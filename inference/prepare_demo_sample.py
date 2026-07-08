"""One-time provisioning script for the inference UI's built-in "try demo" case.

Copies one CTRG test-split sample's 24 CT slice images into inference/static/sample/
so a visitor without their own CT scans can still see the whole upload -> report flow
work. This is *not* meant to be run as part of `git pull` on every deployment: the
copied .jpg files (and the ground-truth text derived from them) are real dataset
imagery, so they're gitignored (see inference/static/sample/.gitignore) and must be
(re-)provisioned locally on each host that wants the demo, by running this script
once against that host's own copy of the CTRG-Brain dataset.

Usage (run from anywhere, with the project's Python env):
    python inference/prepare_demo_sample.py --sample-id 5924
"""

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "inference" / "static" / "sample"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default="5924", help="CTRG sample id (test split) to bundle as the demo case")
    parser.add_argument(
        "--split-file",
        default=str(PROJECT_ROOT / "dataset" / "CTRG_SAM_SEG_dataset" / "splits" / "test.json"),
        help="Split JSON to look up the sample's image paths and ground-truth report in",
    )
    args = parser.parse_args()

    samples = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    sample = next((s for s in samples if str(s["id"]) == str(args.sample_id)), None)
    if sample is None:
        raise SystemExit(f"sample id {args.sample_id} not found in {args.split_file}")

    images = sample["images"]
    if len(images) != 24:
        raise SystemExit(f"expected 24 images, sample {args.sample_id} has {len(images)}")

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for old in DEMO_DIR.glob("*.jpg"):
        old.unlink()

    # Index-prefix filenames: some samples reuse the same source file for more than
    # one of the 24 slots, which would otherwise collide when copied flat.
    for i, src in enumerate(images):
        src_path = Path(src)
        if not src_path.exists():
            raise SystemExit(f"missing source image: {src_path}")
        shutil.copyfile(src_path, DEMO_DIR / f"{i:02d}_{src_path.name}")

    meta = {
        "sample_id": sample["id"],
        "ground_truth": {
            "findings": sample.get("findings", ""),
            "impression": sample.get("impression", ""),
        },
    }
    (DEMO_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Provisioned demo sample {args.sample_id}: {len(images)} images -> {DEMO_DIR}")


if __name__ == "__main__":
    main()
