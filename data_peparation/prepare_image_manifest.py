#!/usr/bin/env python3
"""Rebase the bundled CTRG 24-slice manifest onto a local dataset root."""

import argparse
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / "dataset" / "total_image_list_final.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preserve the original 24-slice selection while replacing the "
            "authors' absolute paths with paths under a local CTRG dataset."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="CTRG root containing samples/<sample_id>/*.jpg",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Source manifest (default: repository's bundled manifest)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: <dataset-root>/total_image_list_final.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if a rebased image path is missing or a sample has != 24 slices",
    )
    return parser.parse_args()


def rebase_manifest(manifest, dataset_root):
    samples_root = dataset_root.expanduser().resolve() / "samples"
    rebased = {}
    for sample_id, image_paths in manifest.items():
        rebased[str(sample_id)] = [
            str(samples_root / str(sample_id) / Path(image_path).name)
            for image_path in image_paths
        ]
    return rebased


def validate_manifest(manifest):
    errors = []
    for sample_id, image_paths in manifest.items():
        if len(image_paths) != 24:
            errors.append(f"sample {sample_id}: expected 24 paths, found {len(image_paths)}")
        missing = [path for path in image_paths if not Path(path).is_file()]
        if missing:
            errors.append(f"sample {sample_id}: {len(missing)} image(s) missing; first: {missing[0]}")
    return errors


def main():
    args = parse_args()
    output_path = args.output or args.dataset_root / "total_image_list_final.json"

    with args.input.open("r", encoding="utf-8") as file:
        source_manifest = json.load(file)
    if not isinstance(source_manifest, dict):
        raise SystemExit("Input manifest must be a JSON object keyed by sample ID")

    manifest = rebase_manifest(source_manifest, args.dataset_root)
    if args.check:
        errors = validate_manifest(manifest)
        if errors:
            preview = "\n".join(errors[:20])
            remainder = len(errors) - 20
            if remainder > 0:
                preview += f"\n... and {remainder} more error(s)"
            raise SystemExit(f"Manifest validation failed:\n{preview}")

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Wrote {len(manifest)} samples to {output_path}")


if __name__ == "__main__":
    main()
