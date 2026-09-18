#!/usr/bin/env python3
"""Render fixed yaw offsets from a completed FaceDNeRF de-identification run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import legacy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render fixed views around the input camera of a completed run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--yaw-degrees", type=float, nargs="+", default=[-30, -15, 0, 15, 30])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_run_files(run_dir: Path) -> tuple[Path, Path, Path]:
    config_path = run_dir / "config.json"
    generator_path = run_dir / "checkpoints" / "fintuned_generator.pkl"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run configuration not found: {config_path}")
    if not generator_path.is_file():
        raise FileNotFoundError(f"Final generator not found: {generator_path}")

    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    image_id = Path(config["image_path"]).stem
    latent_path = run_dir / "checkpoints" / f"{image_id}.npy"
    if not latent_path.is_file():
        raise FileNotFoundError(f"Final latent not found: {latent_path}")

    camera_path = Path(config["c_path"])
    if not camera_path.is_absolute():
        camera_path = REPOSITORY_ROOT / camera_path
    if not camera_path.is_file():
        raise FileNotFoundError(f"Input camera not found: {camera_path}")
    return generator_path, latent_path, camera_path


def yaw_offset_camera(camera: np.ndarray, degrees: float, pivot: np.ndarray) -> np.ndarray:
    """Orbit the input camera around the EG3D pivot while preserving intrinsics."""
    result = camera.astype(np.float32, copy=True)
    cam2world = result[:16].reshape(4, 4)
    angle = math.radians(degrees)
    rotation = np.array(
        [[math.cos(angle), 0.0, math.sin(angle)],
         [0.0, 1.0, 0.0],
         [-math.sin(angle), 0.0, math.cos(angle)]],
        dtype=np.float32,
    )
    cam2world[:3, :3] = rotation @ cam2world[:3, :3]
    cam2world[:3, 3] = pivot + rotation @ (cam2world[:3, 3] - pivot)
    result[:16] = cam2world.reshape(-1)
    return result


def yaw_label(degrees: float) -> str:
    sign = "m" if degrees < 0 else "p"
    magnitude = abs(degrees)
    number = f"{magnitude:g}".replace(".", "p")
    return f"yaw_{sign}{number}"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "multiview").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generator_path, latent_path, camera_path = resolve_run_files(run_dir)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    with generator_path.open("rb") as handle:
        generator = legacy.load_network_pkl(handle)["G_ema"]
    generator = generator.eval().requires_grad_(False).to(device)

    latent = torch.from_numpy(np.load(latent_path).astype(np.float32)).to(device)
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)
    expected_ws = generator.backbone.mapping.num_ws
    if latent.shape[1] == 1:
        latent = latent.repeat(1, expected_ws, 1)
    if latent.ndim != 3 or latent.shape[0] != 1 or latent.shape[1] != expected_ws:
        raise ValueError(f"Unexpected W+ shape: {tuple(latent.shape)}")

    base_camera = np.load(camera_path).astype(np.float32)
    if base_camera.shape != (25,) or not np.isfinite(base_camera).all():
        raise ValueError(f"Expected a finite 25-value camera vector: {camera_path}")
    pivot = np.asarray(generator.rendering_kwargs["avg_camera_pivot"], dtype=np.float32)

    images = []
    records = []
    with torch.no_grad():
        for degrees in args.yaw_degrees:
            camera = yaw_offset_camera(base_camera, degrees, pivot)
            conditioning = torch.from_numpy(camera).unsqueeze(0).to(device)
            rendered = generator.synthesis(
                latent, conditioning, noise_mode="const")["image"]
            pixels = (rendered[0].permute(1, 2, 0) * 127.5 + 128.0)
            pixels = pixels.clamp(0, 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(pixels, "RGB")
            filename = f"{yaw_label(degrees)}.png"
            image.save(output_dir / filename)
            images.append(image)
            records.append({"yaw_offset_degrees": degrees, "image": filename})

    strip = Image.new("RGB", (sum(image.width for image in images), images[0].height))
    left = 0
    for image in images:
        strip.paste(image, (left, 0))
        left += image.width
    strip.save(output_dir / "strip.png")

    metadata = {
        "run_dir": str(run_dir),
        "generator": str(generator_path),
        "latent": str(latent_path),
        "base_camera": str(camera_path),
        "camera_protocol": "yaw offsets around the input camera and EG3D average pivot",
        "views": records,
        "strip": "strip.png",
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Rendered {len(images)} views to {output_dir}")


if __name__ == "__main__":
    main()
