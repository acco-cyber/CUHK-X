"""Training: build artifacts/model_state.json from the official Training data.

Inputs (official layout):   <training_dir>/training_qa.csv
                            <training_dir>/data/HAU.zip   (or an extracted HAU/ folder) - IR videos for durations
Optional:                   --durations-json FILE  pre-computed {"userU/x-y-z": seconds} to skip reading videos

Everything is deterministic (no randomness, pure-Python arithmetic).
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import zipfile
from collections import defaultdict

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PACKAGE_ROOT, "src"))

from cuhkx_vqa import duration, singleton, slot_rule  # noqa: E402
from cuhkx_vqa.common import clip_of, trial_of, user_of  # noqa: E402
from cuhkx_vqa.data import find_test_qa  # noqa: E402
from cuhkx_vqa.knowledge import Knowledge  # noqa: E402

HAU_IR = re.compile(r"(?:^|/)([^/]*)/(user\d+/\d+-\d+-\d+)/IR/IR\.mp4$")


def _harn_path(p):
    """HARn clips are action SEGMENTS of HAU recordings with the same userU/x-y-z id; their shorter videos must never
    be taken as the HAU clip duration."""
    parts = [x.lower() for x in p.replace("\\", "/").split("/")]
    return any(x == "harn" or x.startswith("harn.") or x.startswith("harn_") for x in parts)


def _hau_key(name):
    m = HAU_IR.search(name.replace("\\", "/"))
    if not m or _harn_path(name) or re.match(r"\d+_", m.group(1)):   # "<k>_<Label>/userU/..." is a HARn layout
        return None
    return m.group(2)


def training_durations(training_dir):
    """{"userU/x-y-z": IR seconds} for every HAU training clip found as a file or inside a zip (HARn excluded)."""
    out = {}
    for root, dirs, files in os.walk(training_dir):
        dirs.sort()
        for f in sorted(files):
            full = os.path.join(root, f)
            rel = os.path.relpath(full, training_dir)
            if f.lower().endswith(".mp4"):
                key = _hau_key("/" + rel)
                if key and key not in out:
                    with open(full, "rb") as fh:
                        sec = duration.mp4_seconds(fh.read())
                    if sec:
                        out[key] = sec
            elif f.lower().endswith(".zip"):
                if _harn_path(rel):
                    continue
                try:
                    zf = zipfile.ZipFile(full)
                except (zipfile.BadZipFile, OSError):
                    continue
                with zf:
                    for info in zf.infolist():
                        key = _hau_key("/" + info.filename)
                        if key and key not in out:
                            sec = duration.mp4_seconds(zf.read(info.filename))
                            if sec:
                                out[key] = sec
    return out


def duration_scenes(rows, secs):
    clips = defaultdict(list)
    for r in rows:
        clips[clip_of(r["path"])].append(r)
    scenes = defaultdict(dict)
    for k, rs in clips.items():
        if not k.startswith("HAU/"):
            continue
        key = k.split("/", 1)[1]
        if not secs.get(key):
            continue
        emo = [r for r in rs if r["category"] == "emotion"]
        if not emo:
            continue
        xy, z = trial_of(k).rsplit("-", 1)
        scenes[(user_of(k), xy)][int(z)] = {"d": secs[key], "adv": emo[0][emo[0]["answer"]]}
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-dir", required=True)
    ap.add_argument("--out", default=os.path.join(PACKAGE_ROOT, "artifacts", "model_state.json"))
    ap.add_argument("--durations-json", default=None)
    args = ap.parse_args()
    t0 = time.time()
    path = os.path.join(args.training_dir, "training_qa.csv")
    if not os.path.exists(path):
        path = find_test_qa(args.training_dir, name="training_qa.csv")
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    K = Knowledge.fit(rows)
    stats = slot_rule.SlotStats.fit(rows)
    cmodel = singleton.CountModel.fit(rows)
    if args.durations_json:
        with open(args.durations_json, encoding="utf-8") as fh:
            secs = json.load(fh)
    else:
        secs = training_durations(args.training_dir)
    if not secs:
        sys.exit("no HAU IR durations found (expected data/HAU.zip or an extracted HAU/ folder, or --durations-json)")
    dmodel = duration.DurModel.fit(duration_scenes(rows, secs))
    state = {"format": "cuhkx-vqa-model-state/1", "training_questions": len(rows),
             "training_clips_with_duration": len(secs),
             "knowledge": K.to_dict(), "slot_stats": stats.to_dict(),
             "singleton_count_model": cmodel.to_dict(), "duration_model": dmodel.to_dict()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False)
    print(f"wrote {args.out}: {len(K.scripts)} scene scripts, {len(secs)} training durations, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
