"""Inference entry point called by inference.sh.

--submission-id primary      -> configs/final.yaml (reproduces the primary verification submission)
--submission-id <Kaggle ID>  -> the configuration mapped to that Selected submission in configs/submissions.json
"""
import argparse
import json
import os
import sys
import time

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PACKAGE_ROOT, "src"))

from cuhkx_vqa import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--submission-id", default="primary")
    ap.add_argument("--config", default=os.path.join(PACKAGE_ROOT, "configs", "final.yaml"))
    args = ap.parse_args()

    config = args.config
    if args.submission_id != "primary":
        with open(os.path.join(PACKAGE_ROOT, "configs", "submissions.json"), encoding="utf-8") as fh:
            mapping = json.load(fh)
        if args.submission_id not in mapping:
            sys.exit(f"Unknown submission id {args.submission_id}; known: {sorted(mapping)}")
        config = os.path.join(PACKAGE_ROOT, mapping[args.submission_id])
    t0 = time.time()
    pipeline.run(args.data_dir, args.output_csv, config, PACKAGE_ROOT)
    print(f"[predict] done in {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
