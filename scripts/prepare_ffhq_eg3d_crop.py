#!/usr/bin/env python3
"""Recreate an EG3D FFHQ crop and its matching camera label.

The implementation follows EG3D's FFHQ preprocessing: a 1500 px FFHQ-style
realignment with an enlarged quad, followed by Deep3DFaceRecon alignment, a
700 px center crop, and resizing to 512 px. Stored landmarks and crop
parameters are used, so no face detector is needed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _resampling(name: str):
    namespace = getattr(Image, "Resampling", Image)
    return getattr(namespace, name)


def realign_ffhq(image: Image.Image, landmarks, output_size=1500,
                 transform_size=4096) -> Image.Image:
    """Apply the enlarged FFHQ alignment used by EG3D's FFHQ preparation."""
    lm = np.asarray(landmarks, dtype=np.float64)
    eye_left = np.mean(lm[36:42], axis=0)
    eye_right = np.mean(lm[42:48], axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (lm[48] + lm[54]) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0,
             np.hypot(*eye_to_mouth) * 1.8)
    x *= 1.8
    y = np.flipud(x) * [-1, 1]
    center = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([
        center - x - y,
        center - x + y,
        center + x + y,
        center + x - y,
    ])
    qsize = np.hypot(*x) * 2

    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        resized = tuple(int(np.rint(v / shrink)) for v in image.size)
        image = image.resize(resized, _resampling("LANCZOS"))
        quad /= shrink
        qsize /= shrink

    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, image.size[0]),
        min(crop[3] + border, image.size[1]),
    )
    if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
        image = image.crop(crop)
        quad -= crop[:2]

    pad_bounds = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad_bounds[0] + border, 0),
        max(-pad_bounds[1] + border, 0),
        max(pad_bounds[2] - image.size[0] + border, 0),
        max(pad_bounds[3] - image.size[1] + border, 0),
    )
    if max(pad) > border - 4:
        # This branch is not needed for FFHQ 00018 but retains the official
        # reflection/blur padding behavior for boundary cases.
        import cv2
        import scipy.ndimage

        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        pixels = np.pad(
            np.float32(image),
            ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
            "reflect",
        )
        height, width, _ = pixels.shape
        yy, xx, _ = np.ogrid[:height, :width, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(xx) / pad[0],
                             np.float32(width - 1 - xx) / pad[2]),
            1.0 - np.minimum(np.float32(yy) / pad[1],
                             np.float32(height - 1 - yy) / pad[3]),
        )
        low_res = cv2.resize(
            pixels, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        low_res = scipy.ndimage.gaussian_filter(
            low_res, [qsize * 0.002, qsize * 0.002, 0])
        low_res = cv2.resize(
            low_res, (width, height), interpolation=cv2.INTER_LANCZOS4)
        pixels += (low_res - pixels) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        median = np.median(
            cv2.resize(pixels, (0, 0), fx=0.1, fy=0.1,
                       interpolation=cv2.INTER_AREA),
            axis=(0, 1),
        )
        pixels += (median - pixels) * np.clip(mask, 0.0, 1.0)
        image = Image.fromarray(
            np.uint8(np.clip(np.rint(pixels), 0, 255)), "RGB")
        quad += pad[:2]

    image = image.transform(
        (transform_size, transform_size),
        Image.Transform.QUAD if hasattr(Image, "Transform") else Image.QUAD,
        tuple((quad + 0.5).flatten()),
        _resampling("BILINEAR"),
    )
    return image.resize((output_size, output_size), _resampling("LANCZOS"))


def pos(image_points, model_points):
    """Estimate weak-perspective translation and scale."""
    point_count = image_points.shape[1]
    matrix = np.zeros([2 * point_count, 8])
    matrix[0:2 * point_count - 1:2, 0:3] = model_points.T
    matrix[0:2 * point_count - 1:2, 3] = 1
    matrix[1:2 * point_count:2, 4:7] = model_points.T
    matrix[1:2 * point_count:2, 7] = 1
    target = image_points.T.reshape([2 * point_count, 1])
    solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    scale = float((np.linalg.norm(solution[0:3]) +
                   np.linalg.norm(solution[4:7])) / 2)
    translation = np.asarray(
        [solution[3, 0], solution[7, 0]], dtype=np.float64)
    return translation, scale


def final_eg3d_crop(image: Image.Image, crop_params) -> Image.Image:
    """Apply EG3D's stored Deep3DFaceRecon normalization and final crop."""
    landmarks = np.asarray(crop_params["lm"], dtype=np.float64).reshape(-1, 2)
    landmarks[:, 1] = image.size[1] - 1 - landmarks[:, 1]
    model = np.asarray(crop_params["lm3d_std"], dtype=np.float64)
    translation, scale = pos(landmarks.T, model.T)
    scale = float(crop_params["rescale_factor"] / scale)

    width, height = image.size
    resized_width = int(np.int32(width * scale))
    resized_height = int(np.int32(height * scale))
    target_size = 1024
    left = int(np.int32(
        resized_width / 2 - target_size / 2 +
        (translation[0] - width / 2) * scale))
    upper = int(np.int32(
        resized_height / 2 - target_size / 2 +
        (height / 2 - translation[1]) * scale))
    image = image.resize(
        (resized_width, resized_height), _resampling("BICUBIC"))
    image = image.crop(
        (left, upper, left + target_size, upper + target_size))

    crop_size = int(crop_params["center_crop_size"])
    left = int(image.size[0] / 2 - crop_size / 2)
    upper = int(image.size[1] / 2 - crop_size / 2)
    image = image.crop((left, upper, left + crop_size, upper + crop_size))
    output_size = int(crop_params["output_size"])
    return image.resize((output_size, output_size), _resampling("LANCZOS"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-id",
                        help="Five-digit FFHQ identifier, e.g. 00018")
    parser.add_argument("--input", type=Path,
                        help="Original in-the-wild FFHQ image")
    parser.add_argument(
        "--input-dir", type=Path,
        help="Process every numerically named PNG in this directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--save-aligned", action="store_true",
        help="Also retain the intermediate 1500x1500 alignment")
    parser.add_argument(
        "--metadata", type=Path,
        default=Path("data/ffhq-dataset-v2/ffhq-dataset-v2.json"))
    parser.add_argument(
        "--crop-params", type=Path,
        default=Path("data/ffhq-dataset-v2/cropping_params.json"))
    parser.add_argument(
        "--camera-labels", type=Path,
        default=Path("data/ffhq_512/dataset.json"))
    args = parser.parse_args()

    if args.input_dir is not None:
        if args.image_id is not None or args.input is not None:
            parser.error("use either --input-dir or --image-id with --input")
        inputs = sorted(
            path for path in args.input_dir.glob("*.png")
            if path.stem.isdigit())
        if not inputs:
            parser.error(f"no numerically named PNG files found in {args.input_dir}")
    else:
        if args.image_id is None or args.input is None:
            parser.error("--image-id and --input are required without --input-dir")
        inputs = [args.input]

    all_metadata = json.loads(args.metadata.read_text())
    all_crop_params = json.loads(args.crop_params.read_text())
    labels = dict(json.loads(args.camera_labels.read_text())["labels"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in inputs:
        requested_id = args.image_id if args.input_dir is None else input_path.stem
        image_id = f"{int(requested_id):05d}"
        filename = f"{image_id}.png"
        metadata = all_metadata[str(int(image_id))]
        crop_params = all_crop_params[filename]
        if filename not in labels:
            raise KeyError(f"No camera label for {filename}")

        source_metadata = metadata["in_the_wild"]
        if input_path.stat().st_size != source_metadata["file_size"]:
            raise ValueError(
                f"{input_path}: file size does not match FFHQ metadata")
        digest = hashlib.md5()
        with input_path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != source_metadata["file_md5"]:
            raise ValueError(f"{input_path}: MD5 does not match FFHQ metadata")

        image = Image.open(input_path).convert("RGB")
        expected_size = tuple(source_metadata["pixel_size"])
        if image.size != expected_size:
            raise ValueError(
                f"{input_path}: size {image.size} does not match {expected_size}")

        aligned = realign_ffhq(image, source_metadata["face_landmarks"])
        cropped = final_eg3d_crop(aligned, crop_params)
        if args.save_aligned:
            aligned.save(args.output_dir / f"{image_id}_realign1500.png")
        cropped.save(args.output_dir / filename)
        np.save(args.output_dir / f"{image_id}.npy",
                np.asarray(labels[filename], dtype=np.float32))
        print(f"Prepared {image_id}: {args.output_dir / filename}")


if __name__ == "__main__":
    main()
