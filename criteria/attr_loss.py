import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import torchvision.models as tv_models


class GenderClassifier(nn.Module):
    def __init__(self, ckpt_path: str, num_classes: int = 2):
        super().__init__()
        model = tv_models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        state = torch.load(ckpt_path, map_location="cpu")
        # Support both raw state_dict and wrapped checkpoint formats
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
            # Strip common prefixes added by training frameworks
            state = {k.replace("model.", "").replace("module.", ""): v
                     for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ======================================================================
# DAN — Distract your Attention Network (Wen et al., arXiv:2109.07270)
#
# Reimplemented from https://github.com/yaoing/DAN/blob/main/networks/dan.py
# The architecture uses a ResNet-18 feature backbone followed by
# multi-head cross-attention (spatial + channel) heads.
# ======================================================================


class _SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1, keepdim=True)
        out = x * y
        return out


class _ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid(),
        )

    def forward(self, sa: torch.Tensor) -> torch.Tensor:
        sa = self.gap(sa)
        sa = sa.view(sa.size(0), -1)
        y = self.attention(sa)
        out = sa * y
        return out


class _CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = _SpatialAttention()
        self.ca = _ChannelAttention()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sa = self.sa(x)
        ca = self.ca(sa)
        return ca


class DAN(nn.Module):
    def __init__(self, num_class: int = 8, num_head: int = 4):
        super(DAN, self).__init__()

        resnet = tv_models.resnet18(weights=None)
        # Use all layers except avgpool and fc as the feature backbone
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            setattr(self, "cat_head%d" % i, _CrossAttentionHead())
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self, "cat_head%d" % i)(x))

        heads = torch.stack(heads).permute([1, 0, 2])
        if heads.size(1) > 1:
            heads = F.log_softmax(heads, dim=1)

        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)
        return out


class ExprClassifier(nn.Module):
    def __init__(self, ckpt_path: str, num_classes: int = 8, num_head: int = 4):
        super().__init__()
        self.model = DAN(num_class=num_classes, num_head=num_head)

        state = torch.load(ckpt_path, map_location="cpu")
        # Support both raw state_dict and wrapped checkpoint formats
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            # Strip common prefixes added by training frameworks
            state = {k.replace("model.", "").replace("module.", ""): v
                     for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AttrLoss(nn.Module):
    # ImageNet normalisation used by torchvision ResNets
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        gender_ckpt: str = "./networks/gender_classifier.pth",
        expr_ckpt: str = "./networks/affecnet8_epoch5_acc0.6209.pth",
        gender_classes: int = 2,
        expr_classes: int = 8,
        expr_num_head: int = 4,
        input_size: int = 224,
    ):
        super(AttrLoss, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading ResNet-18 gender classifier for gender loss")
        self.gender_net = GenderClassifier(gender_ckpt, gender_classes)
        self.gender_net.eval()
        self.gender_net.to(device)

        print("Loading DAN expression classifier for expression loss")
        self.expr_net = ExprClassifier(expr_ckpt, expr_classes, expr_num_head)
        self.expr_net.eval()
        self.expr_net.to(device)

        self.input_size = input_size

        # Face region crop pools
        self.pool = nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = nn.AdaptiveAvgPool2d((input_size, input_size))

        # Register normalisation constants as buffers so they follow .to(device)
        self.register_buffer(
            "norm_mean",
            torch.tensor(self.MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "norm_std",
            torch.tensor(self.STD).view(1, 3, 1, 1),
        )

        self.to(device)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.norm_mean.device)

        if x.shape[2] != 256:
            x = self.pool(x)

        x = x[:, :, 35:223, 32:220]  # [B, 3, 188, 188]

        # Resize to classifier input size
        x = self.face_pool(x)

        # Map from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0

        # ImageNet normalisation
        x = (x - self.norm_mean) / self.norm_std

        return x

    def forward(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_pre = self._preprocess(y)
        y_hat_pre = self._preprocess(y_hat)

        # Gender loss (Eq. 7)
        with torch.no_grad():
            g_t = self.gender_net(y_pre)
        g_g = self.gender_net(y_hat_pre)

        n_gd = g_t.shape[1]
        gender_loss = (g_t - g_g).square().sum(dim=1).mean() / n_gd

        # Expression loss (Eq. 8)
        with torch.no_grad():
            e_t = self.expr_net(y_pre)
        e_g = self.expr_net(y_hat_pre)

        n_ex = e_t.shape[1]
        expr_loss = (e_t - e_g).square().sum(dim=1).mean() / n_ex

        return gender_loss, expr_loss
