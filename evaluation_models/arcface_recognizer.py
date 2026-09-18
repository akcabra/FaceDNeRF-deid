"""Evaluation wrapper for the ArcFace model used by the de-identification loss."""

from pathlib import Path
from typing import Sequence, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from models.facial_recognition.model_irse import Backbone


ImageInput = Union[str, Path, Image.Image]


class ArcFaceRecognizer:
    """Produce normalized embeddings with the pipeline's ArcFace preprocessing."""

    def __init__(self, checkpoint="networks/model_ir_se50.pth", device="cuda"):
        self.device = torch.device(device)
        model = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se")
        model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
        self.model = model.eval().requires_grad_(False).to(self.device)

    @staticmethod
    def _load(image: ImageInput) -> torch.Tensor:
        if isinstance(image, Image.Image):
            rgb = image.convert("RGB")
        else:
            with Image.open(image) as loaded:
                rgb = loaded.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32).copy()
        return torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0

    @torch.no_grad()
    def embed(self, images: Sequence[ImageInput], batch_size: int = 32) -> torch.Tensor:
        if not images:
            raise ValueError("At least one image is required")
        outputs = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack([self._load(x) for x in images[start:start + batch_size]])
            batch = batch.to(self.device)
            if batch.shape[-2:] != (256, 256):
                batch = F.interpolate(batch, (256, 256), mode="area")
            batch = batch[:, :, 35:223, 32:220]
            batch = F.adaptive_avg_pool2d(batch, (112, 112))
            outputs.append(F.normalize(self.model(batch), p=2, dim=1).cpu())
        return torch.cat(outputs, dim=0)

    @staticmethod
    def cosine_similarity(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(first, second, dim=-1)

