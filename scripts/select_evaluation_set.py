#!/usr/bin/env python3
"""Create a reproducible, stratified evaluation-set manifest.

The script selects images only; it does not estimate the EG3D camera vectors
required by the FaceDNeRF pipeline. Selected images therefore remain marked as
``pending_preprocessing`` until their PNG and NPY files have been prepared.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a fixed stratified-random subset and write a CSV manifest "
            "plus a JSON file describing the selection protocol."
        )
    )
    parser.add_argument(
        "--attributes",
        type=Path,
        default=Path("data/celeba/list_attr_celeba.csv"),
        help="CSV containing image_id and attribute columns.",
    )
    parser.add_argument(
        "--partitions",
        type=Path,
        default=Path("data/celeba/list_eval_partition.csv"),
        help="Optional CSV containing image_id and partition (0=train, 1=validation, 2=test).",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=2,
        choices=(0, 1, 2),
        help="CelebA partition from which to sample (default: test).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--images-dir",
        type=Path,
        help="Directory containing the source images.",
    )
    source.add_argument(
        "--images-zip",
        type=Path,
        help="ZIP archive containing the source images.",
    )
    parser.add_argument(
        "--identity-csv",
        type=Path,
        help=(
            "Optional CSV with image_id and identity_id. When supplied, no two "
            "selected images will have the same identity."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of images to select (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for selection (default: 42).",
    )
    parser.add_argument(
        "--stratify",
        default="Male,Smiling,Eyeglasses",
        help="Comma-separated binary attribute columns used for stratification.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Development image ID to exclude. May be repeated; an ID matches "
            "with or without its file extension."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/evaluation_set.csv"),
        help="Output CSV manifest (default: evaluation/evaluation_set.csv).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("evaluation/test_data"),
        help="Expected destination for preprocessed PNG/NPY pairs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output manifest and metadata file.",
    )
    return parser.parse_args()


def normalize_id(value: str) -> str:
    return Path(value.strip()).stem


def comparison_id(value: str) -> str:
    """Match numeric IDs regardless of zero padding while preserving other IDs."""
    image_id = normalize_id(value)
    return str(int(image_id)) if image_id.isdigit() else image_id


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def image_column(fieldnames: Sequence[str], path: Path) -> str:
    for candidate in ("image_id", "image", "filename", "file"):
        if candidate in fieldnames:
            return candidate
    raise ValueError(f"{path} must contain an image_id column")


def load_partitions(path: Optional[Path]) -> Dict[str, int]:
    if path is None:
        return {}
    fields, rows = read_csv(path)
    image_key = image_column(fields, path)
    if "partition" not in fields:
        raise ValueError(f"{path} must contain a partition column")
    return {normalize_id(row[image_key]): int(row["partition"]) for row in rows}


def load_identities(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    fields, rows = read_csv(path)
    image_key = image_column(fields, path)
    identity_key = next(
        (key for key in ("identity_id", "identity", "person_id") if key in fields),
        None,
    )
    if identity_key is None:
        raise ValueError(f"{path} must contain an identity_id column")
    return {normalize_id(row[image_key]): row[identity_key].strip() for row in rows}


def available_directory_images(path: Path) -> Dict[str, str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {path}")
    result: Dict[str, str] = {}
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            result.setdefault(item.stem, str(item))
    return result


def available_zip_images(path: Path) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Image archive not found: {path}")
    result: Dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            member = Path(name)
            if member.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                result.setdefault(member.stem, f"{path}::{name}")
    return result


def normalized_attribute(value: str) -> str:
    text = value.strip()
    if text in {"1", "1.0", "true", "True"}:
        return "1"
    if text in {"-1", "-1.0", "0", "0.0", "false", "False"}:
        return "0"
    return text or "missing"


def round_robin_sample(
    groups: Dict[Tuple[str, ...], List[Dict[str, str]]],
    count: int,
    rng: random.Random,
    identities: Dict[str, str],
) -> List[Dict[str, str]]:
    """Sample evenly across nonempty strata with deterministic random order."""
    for rows in groups.values():
        rng.shuffle(rows)
    strata = sorted(groups)
    rng.shuffle(strata)

    selected: List[Dict[str, str]] = []
    used_identities = set()
    positions = {stratum: 0 for stratum in strata}

    while len(selected) < count:
        added = False
        for stratum in strata:
            rows = groups[stratum]
            while positions[stratum] < len(rows):
                row = rows[positions[stratum]]
                positions[stratum] += 1
                image_id = row["_normalized_id"]
                identity = identities.get(image_id)
                if identity is not None and identity in used_identities:
                    continue
                selected.append(row)
                if identity is not None:
                    used_identities.add(identity)
                added = True
                break
            if len(selected) == count:
                return selected
        if not added:
            break
    return selected


def metadata_path(output: Path) -> Path:
    return output.with_suffix(".metadata.json")


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")

    output = args.output.resolve()
    meta_output = metadata_path(output)
    if not args.force and (output.exists() or meta_output.exists()):
        raise FileExistsError(
            f"Output already exists: {output} or {meta_output}; use --force to replace it"
        )

    fields, rows = read_csv(args.attributes)
    image_key = image_column(fields, args.attributes)
    strata_columns = [item.strip() for item in args.stratify.split(",") if item.strip()]
    missing_columns = [item for item in strata_columns if item not in fields]
    if missing_columns:
        raise ValueError(f"Unknown stratification columns: {', '.join(missing_columns)}")

    partitions = load_partitions(args.partitions)
    identities = load_identities(args.identity_csv)
    if args.images_dir:
        available = available_directory_images(args.images_dir)
        source_description = str(args.images_dir.resolve())
        source_kind = "directory"
    else:
        available = available_zip_images(args.images_zip)
        source_description = str(args.images_zip.resolve())
        source_kind = "zip"

    excluded = {comparison_id(item) for item in args.exclude}
    groups: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    eligible = 0
    for original_row in rows:
        image_id = normalize_id(original_row[image_key])
        if comparison_id(image_id) in excluded or image_id not in available:
            continue
        if partitions and partitions.get(image_id) != args.partition:
            continue
        row = dict(original_row)
        row["_normalized_id"] = image_id
        row["_source"] = available[image_id]
        stratum = tuple(normalized_attribute(row[column]) for column in strata_columns)
        row["_stratum"] = "|".join(
            f"{column}={value}" for column, value in zip(strata_columns, stratum)
        )
        groups[stratum].append(row)
        eligible += 1

    selected = round_robin_sample(groups, args.count, random.Random(args.seed), identities)
    if len(selected) < args.count:
        distinct_note = " with distinct identities" if identities else ""
        raise ValueError(
            f"Requested {args.count} images but only {len(selected)} eligible samples"
            f"{distinct_note} could be selected"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    processed_dir = args.processed_dir
    manifest_fields = [
        "selection_rank",
        "image_id",
        "source_image",
        "identity_id",
        "partition",
        "stratum",
        *strata_columns,
        "processed_png",
        "camera_npy",
        "preprocessing_status",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for rank, row in enumerate(selected, start=1):
            image_id = row["_normalized_id"]
            writer.writerow(
                {
                    "selection_rank": rank,
                    "image_id": image_id,
                    "source_image": row["_source"],
                    "identity_id": identities.get(image_id, ""),
                    "partition": partitions.get(image_id, ""),
                    "stratum": row["_stratum"],
                    **{column: normalized_attribute(row[column]) for column in strata_columns},
                    "processed_png": str(processed_dir / f"{image_id}.png"),
                    "camera_npy": str(processed_dir / f"{image_id}.npy"),
                    "preprocessing_status": "pending_preprocessing",
                }
            )

    stratum_counts: Dict[str, int] = defaultdict(int)
    for row in selected:
        stratum_counts[row["_stratum"]] += 1
    metadata = {
        "protocol": "fixed stratified-random image selection",
        "seed": args.seed,
        "requested_count": args.count,
        "selected_count": len(selected),
        "eligible_image_count": eligible,
        "partition": args.partition if partitions else None,
        "stratification_columns": strata_columns,
        "excluded_image_ids": [normalize_id(item) for item in args.exclude],
        "source_kind": source_kind,
        "source": source_description,
        "attributes_csv": str(args.attributes.resolve()),
        "partitions_csv": str(args.partitions.resolve()) if partitions else None,
        "identity_csv": str(args.identity_csv.resolve()) if args.identity_csv else None,
        "identity_uniqueness_enforced": bool(identities),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "important_limitation": (
            "Image-level sampling does not guarantee distinct people because no "
            "identity CSV was supplied."
            if not identities
            else None
        ),
        "next_step": "Estimate EG3D camera parameters and prepare each PNG/NPY pair.",
    }
    with meta_output.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(f"Selected {len(selected)} of {eligible} eligible images using seed {args.seed}.")
    print(f"Manifest: {output}")
    print(f"Metadata: {meta_output}")
    if not identities:
        print(
            "Warning: identity uniqueness was not enforced; supply --identity-csv "
            "for one image per person.",
            file=sys.stderr,
        )
    print("Next: preprocess selected images to create the listed PNG/NPY pairs.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
