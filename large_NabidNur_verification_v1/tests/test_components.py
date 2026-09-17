"""Self-contained tests (no competition data needed):  python3 tests/test_components.py"""
import csv
import io
import os
import struct
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cuhkx_vqa import duration, pipeline  # noqa: E402
from cuhkx_vqa.common import atoms, closure, opts  # noqa: E402
from cuhkx_vqa.data import VideoIndex, canonical_test_hash  # noqa: E402


def _box(t, p):
    return struct.pack(">I4s", 8 + len(p), t) + p


def _mp4(timescale, units, moov_last=True):
    mvhd = _box(b"mvhd", b"\x00\x00\x00\x00" + b"\x00" * 8 + struct.pack(">II", timescale, units) + b"\x00" * 80)
    parts = [_box(b"ftyp", b"isom\x00\x00\x02\x00"), _box(b"mdat", b"\x00" * 100)]
    moov = _box(b"moov", mvhd)
    return b"".join(parts + [moov]) if moov_last else b"".join([parts[0], moov, parts[1]])


def test_mp4_seconds():
    assert abs(duration.mp4_seconds(_mp4(1000, 12345)) - 12.345) < 1e-12
    assert abs(duration.mp4_seconds(_mp4(10, 66, moov_last=False)) - 6.6) < 1e-12
    assert duration.mp4_seconds(b"not an mp4") is None


def test_video_index_zip_and_folder():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "a", "large_model_track_test", "LM_test_0002", "IR"))
        with open(os.path.join(d, "a", "large_model_track_test", "LM_test_0002", "IR", "IR.mp4"), "wb") as fh:
            fh.write(_mp4(10, 75))
        with zipfile.ZipFile(os.path.join(d, "clips.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("large_model_track_test/LM_test_0001/IR/IR.mp4", _mp4(10, 66))
        vi = VideoIndex(d)
        try:
            assert abs(vi.duration_of("large_model_track_test/LM_test_0001") - 6.6) < 1e-12
            assert abs(vi.duration_of("large_model_track_test/LM_test_0002") - 7.5) < 1e-12
            assert vi.duration_of("large_model_track_test/LM_test_0003") is None
        finally:
            vi.close()


def test_helpers_and_format():
    assert atoms("Walking, Eating ,") == frozenset({"Walking", "Eating"})
    assert ("a", "c") in closure({("a", "b"), ("b", "c")})
    assert opts({"A": "x", "B": "y", "C": "z", "D": ""}) == ["x", "y", "z"]
    assert pipeline.valid("BD", "multi", 4) and not pipeline.valid("DB", "multi", 4)
    assert pipeline.valid("DCBA", "sequence", 4) and not pipeline.valid("DCB", "sequence", 4)
    assert pipeline.valid("C", "single", 3) and not pipeline.valid("D", "single", 3)
    rows = [{"qa_id": "t1", "source": "HAU", "path": "p", "category": "single", "question": "q",
             "A": "a", "B": "b", "C": "c", "D": ""}]
    h1 = canonical_test_hash(rows)
    rows[0]["A"] = "changed"
    assert canonical_test_hash(rows) != h1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
