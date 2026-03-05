import torch
from torch import nn

from models.facial_recognition.model_irse import Backbone


class DeIDLoss(nn.Module):
    def __init__(self, pp: float = 0.0):
        super(DeIDLoss, self).__init__()
        self.pp = pp

        print('Loading ResNet ArcFace for de-identification loss')
        self.facenet = Backbone(input_size=112, num_layers=50,
                                drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load("./networks/model_ir_se50.pth"))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()

    def extract_feats(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # crop to face region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    # Loss calculation follows "Face deidentification with controllable privacy protection" Eq. (6)
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        n_samples = y.shape[0]

        y_feats = self.extract_feats(y).detach()
        y_hat_feats = self.extract_feats(y_hat)

        loss = torch.tensor(0.0, device=y.device)
        total_dist = torch.tensor(0.0, device=y.device)

        for i in range(n_samples):
            dist = (y_feats[i] - y_hat_feats[i]).square().sum()
            total_dist = total_dist + dist

            loss = loss + torch.clamp(self.pp - dist, min=0.0)

        loss = loss / n_samples
        mean_dist = total_dist / n_samples

        return loss, mean_dist
