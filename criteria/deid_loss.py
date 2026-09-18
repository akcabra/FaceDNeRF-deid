import torch
from torch import nn

from models.facial_recognition.model_irse import Backbone


class DeIDLoss(nn.Module):
    def __init__(self, pp: float = 0.0):
        super(DeIDLoss, self).__init__()
        if not 0.0 <= pp <= 1.0:
            raise ValueError(f"pp must be in [0, 1], got {pp}")
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

    def similarity(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return normalized ArcFace cosine similarity for each image pair."""
        y_feats = torch.nn.functional.normalize(
            self.extract_feats(y).detach(), dim=1)
        y_hat_feats = torch.nn.functional.normalize(self.extract_feats(y_hat), dim=1)
        return (y_feats * y_hat_feats).sum(dim=1)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        y_feats = self.extract_feats(y).detach()
        y_hat_feats = self.extract_feats(y_hat)

        # Use normalized ArcFace cosine similarity as the controllable score.
        # Lower pp values request stronger identity modification.
        y_feats = torch.nn.functional.normalize(y_feats, dim=1)
        y_hat_feats = torch.nn.functional.normalize(y_hat_feats, dim=1)
        similarity = (y_feats * y_hat_feats).sum(dim=1)
        loss = (similarity - self.pp).square().mean()
        return loss, similarity.mean()
