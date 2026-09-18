"""Evaluation-only FaceNet recognizer pretrained on VGGFace2.

The wrapper intentionally loads a local checkpoint and never downloads model
weights. Inputs are expected to be aligned face images. They are resized to
160x160 and standardized using the facenet-pytorch fixed-image convention
``(pixel - 127.5) / 128``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .facenet import InceptionResnetV1


ImageInput = Union[str, Path, Image.Image]

DEFAULT_CHECKPOINT = Path("networks/20180402-114759-vggface2.pt")
EXPECTED_SHA256 = "281cebca8662831adb987a874bdcb36e73f5b1c6dc5ee5878f305e985625d99b"


def checkpoint_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resample_bilinear():
    return Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


class FaceNetVGGFace2Recognizer:
    """Produce normalized 512D FaceNet embeddings and cosine similarities."""

    image_size = 160
    embedding_size = 512

    def __init__(
        self,
        checkpoint: Union[str, Path] = DEFAULT_CHECKPOINT,
        device: Union[str, torch.device] = "cpu",
        verify_checksum: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"FaceNet checkpoint not found: {self.checkpoint}")
        if verify_checksum:
            actual = checkpoint_sha256(self.checkpoint)
            if actual != EXPECTED_SHA256:
                raise ValueError(
                    "Unexpected FaceNet checkpoint checksum: "
                    f"expected {EXPECTED_SHA256}, got {actual}"
                )

        self.device = torch.device(device)
        # Creating the classifier head reproduces the checkpoint architecture.
        # classify is disabled after loading so forward() returns embeddings.
        model = InceptionResnetV1(
            pretrained=None, classify=True, num_classes=8631
        )
        state_dict = torch.load(str(self.checkpoint), map_location="cpu")
        if not isinstance(state_dict, dict):
            raise TypeError("FaceNet checkpoint must contain a state dictionary")
        model.load_state_dict(state_dict, strict=True)
        model.classify = False
        model.eval().requires_grad_(False).to(self.device)
        self.model: nn.Module = model

    @staticmethod
    def _open_rgb(image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        with Image.open(image) as loaded:
            return loaded.convert("RGB")

    @classmethod
    def preprocess(cls, image: ImageInput) -> torch.Tensor:
        """Return a standardized CHW tensor for an already aligned face."""
        rgb = cls._open_rgb(image)
        rgb = rgb.resize((cls.image_size, cls.image_size), _resample_bilinear())
        array = np.asarray(rgb, dtype=np.float32).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - 127.5) / 128.0

    @torch.no_grad()
    def embed(self, images: Sequence[ImageInput], batch_size: int = 32) -> torch.Tensor:
        """Return CPU embeddings shaped ``[N, 512]``."""
        if not images:
            raise ValueError("At least one image is required")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        prepared = torch.stack([self.preprocess(image) for image in images])
        outputs = []
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size].to(self.device)
            embeddings = self.model(batch)
            outputs.append(F.normalize(embeddings, p=2, dim=1).cpu())
        result = torch.cat(outputs, dim=0)
        if result.shape != (len(images), self.embedding_size):
            raise RuntimeError(f"Unexpected embedding shape: {tuple(result.shape)}")
        if not torch.isfinite(result).all():
            raise RuntimeError("FaceNet produced non-finite embeddings")
        return result

    @staticmethod
    def cosine_similarity(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape:
            raise ValueError(
                f"Embedding shapes must match, got {tuple(first.shape)} and {tuple(second.shape)}"
            )
        return F.cosine_similarity(first, second, dim=-1)

    def compare(self, first: ImageInput, second: ImageInput) -> float:
        embeddings = self.embed([first, second], batch_size=2)
        return float(self.cosine_similarity(embeddings[0:1], embeddings[1:2]).item())
