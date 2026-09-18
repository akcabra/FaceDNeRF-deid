#!/usr/bin/env python3
"""Evaluate final input-view images from completed de-identification runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import kornia
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation_models.arcface_recognizer import ArcFaceRecognizer
from evaluation_models.facenet_recognizer import FaceNetVGGFace2Recognizer


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--pp-values", type=float, nargs="*")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--arcface-checkpoint", type=Path,
                        default=Path("networks/model_ir_se50.pth"))
    parser.add_argument("--facenet-checkpoint", type=Path,
                        default=Path("networks/20180402-114759-vggface2.pt"))
    return parser.parse_args()


def manifest_rows(path):
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {str(row["image_id"]): row for row in csv.DictReader(handle)}


def resolve_input(config, run_dir, manifest):
    image_id = Path(config["image_path"]).stem
    row = manifest.get(image_id, {})
    for key in ("processed_png", "image_path", "source_image", "preprocessing_input"):
        value = row.get(key)
        if value:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            if candidate.is_file():
                return image_id, candidate, row
    candidate = Path(config["image_path"])
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"Input image not found for {run_dir}: {candidate}")
    return image_id, candidate, row


def final_stage_metrics(run_dir):
    with (run_dir / "metrics.json").open(encoding="utf-8") as handle:
        root_metrics = json.load(handle)
    stages = root_metrics.get("stages", [])
    if not stages:
        raise ValueError(f"No stage metrics in {run_dir / 'metrics.json'}")
    return next((x for x in reversed(stages) if x.get("stage") == "post"), stages[-1])


def image_tensor(path):
    with Image.open(path) as loaded:
        rgb = loaded.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    array = np.asarray(rgb, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def ssim(first, second):
    return float(kornia.metrics.ssim(
        image_tensor(first), image_tensor(second), window_size=11, max_val=1.0).mean())


def main():
    args = arguments()
    manifest = manifest_rows(args.sample_manifest)
    allowed_pp = None if not args.pp_values else {round(x, 8) for x in args.pp_values}
    records = []
    for config_path in sorted(args.results_dir.rglob("config.json")):
        run_dir = config_path.parent
        final_image = run_dir / "final.png"
        inversion_image = run_dir / "inversion" / "final.png"
        metrics_path = run_dir / "metrics.json"
        if (not final_image.is_file() or not inversion_image.is_file() or
                not metrics_path.is_file()):
            continue
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        pp = float(config["pp"])
        if allowed_pp is not None and round(pp, 8) not in allowed_pp:
            continue
        image_id, input_image, sample = resolve_input(config, run_dir, manifest)
        records.append({
            "image_id": image_id, "identity_id": sample.get("identity_id", ""),
            "pp": pp, "seed": config.get("seed", ""), "run_name": config.get("run_name", ""),
            "run_dir": str(run_dir.resolve()), "input_image": str(input_image.resolve()),
            "inversion_image": str(inversion_image.resolve()),
            "final_image": str(final_image.resolve()), "stage": final_stage_metrics(run_dir),
        })
    if not records:
        raise RuntimeError(f"No completed runs found below {args.results_dir}")

    paths = [Path(x[key]) for x in records
             for key in ("input_image", "inversion_image", "final_image")]
    arcface = ArcFaceRecognizer(args.arcface_checkpoint, args.device)
    facenet = FaceNetVGGFace2Recognizer(args.facenet_checkpoint, args.device)
    arc_embeddings = arcface.embed(paths, args.batch_size)
    face_embeddings = facenet.embed(paths, args.batch_size)

    rows = []
    for index, record in enumerate(records):
        first, inversion, second = 3 * index, 3 * index + 1, 3 * index + 2
        arc_inversion_similarity = float(arcface.cosine_similarity(
            arc_embeddings[first:first + 1],
            arc_embeddings[inversion:inversion + 1]))
        arc_similarity = float(arcface.cosine_similarity(
            arc_embeddings[first:first + 1], arc_embeddings[second:second + 1]))
        independent_inversion_similarity = float(facenet.cosine_similarity(
            face_embeddings[first:first + 1],
            face_embeddings[inversion:inversion + 1]))
        independent_similarity = float(facenet.cosine_similarity(
            face_embeddings[first:first + 1], face_embeddings[second:second + 1]))
        stage = record.pop("stage")
        rows.append({
            **record,
            "arcface_inversion_similarity": arc_inversion_similarity,
            "arcface_similarity": arc_similarity,
            "arcface_similarity_reduction": arc_inversion_similarity - arc_similarity,
            "arcface_recorded_similarity": stage.get("arcface_similarity", ""),
            "arcface_calibration_error": abs(arc_similarity - record["pp"]),
            "arcface_at_or_below_pp": arc_similarity <= record["pp"],
            "independent_facenet_inversion_similarity": independent_inversion_similarity,
            "independent_facenet_similarity": independent_similarity,
            "independent_facenet_similarity_reduction": (
                independent_inversion_similarity - independent_similarity),
            "pixel_mse": stage.get("pixel_mse", ""),
            "vgg_distance": stage.get("vgg_distance", ""),
            "ssim": stage.get("ssim", ssim(record["input_image"], record["final_image"])),
            "gender_agreement": stage.get("gender_agreement", ""),
            "expression_agreement": stage.get("expression_agreement", ""),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} input-view rows to {args.output}")


if __name__ == "__main__":
    main()
