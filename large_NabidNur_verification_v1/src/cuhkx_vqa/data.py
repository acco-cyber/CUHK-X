"""Locate the official label-free Testing data and read clip durations (extracted folders or zip archives)."""
import csv
import hashlib
import os
import sys
import zipfile

from .duration import mp4_seconds

MODALITY_PREFERENCE = ("IR", "Thermal", "Depth", "Depth_Color")
REQUIRED_COLUMNS = ("qa_id", "source", "path", "category", "A", "B", "C", "D")
KNOWN_CATEGORIES = {"single", "multi", "sequence", "combination", "emotion", "object_interaction"}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def walk(top):
    """os.walk that follows directory symlinks, guards against symlink loops and yields sorted directories."""
    seen = set()
    for root, dirs, files in os.walk(top, followlinks=True):
        real = os.path.realpath(root)
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        dirs.sort()
        yield root, dirs, sorted(files)


def find_test_qa(data_dir, name="test_qa.csv", max_depth=4):
    """The shallowest test_qa.csv at most `max_depth` directory levels below data_dir (ambiguity is an error)."""
    base_depth = os.path.abspath(data_dir).rstrip(os.sep).count(os.sep)
    hits = []
    for root, dirs, files in walk(data_dir):
        depth = os.path.abspath(root).count(os.sep) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        if name in files:
            hits.append((depth, os.path.join(root, name)))
    if not hits:
        raise SystemExit(f"{name} not found within {max_depth} directory levels below {data_dir}")
    hits.sort()
    top = [p for d, p in hits if d == hits[0][0]]
    if len(top) > 1:
        raise SystemExit(f"ambiguous input: several {name} at the same depth: {top}; pass the Testing directory itself")
    return top[0]


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if rows:
        missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
        if missing:
            raise SystemExit(f"{path} is missing required columns {missing}")
        unknown = sorted({r.get("category") or "" for r in rows} - KNOWN_CATEGORIES)
        if unknown:
            log(f"[data] WARNING: unknown question categories {unknown}; they receive a format-valid default answer")
    return rows


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_test_hash(rows):
    """SHA-256 of the parsed question content, independent of row order, line endings, BOM or quoting."""
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r.get("qa_id") or ""):
        fields = [r.get(k) or "" for k in ("qa_id", "source", "path", "category", "question", "A", "B", "C", "D")]
        h.update(("\x1f".join(fields) + "\x1e").encode("utf-8"))
    return h.hexdigest()


def _suffix_keys(path):
    """Every suffix of >= 3 path components ('CLIP/MOD/FILE.mp4', 'PARENT/CLIP/MOD/FILE.mp4', ...), lower-cased."""
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
    return ["/".join(parts[-j:]).lower() for j in range(3, len(parts) + 1)]


class VideoIndex:
    """Maps any path suffix 'PARENTS.../CLIPDIR/MODALITY/FILE.mp4' to an extracted file or a zip member.

    Lookups use the longest suffix of the clip path, so clip directories that repeat under different parents do not
    collide. Unreadable archives and videos are logged and skipped (the affected scene keeps its text answers)."""

    def __init__(self, data_dir):
        self.files = {}
        self.members = {}
        self._zips = {}
        n_files = n_members = 0
        for root, dirs, files in walk(data_dir):
            for f in files:
                full = os.path.join(root, f)
                low = f.lower()
                if low.endswith(".mp4"):
                    n_files += 1
                    for k in _suffix_keys(os.path.relpath(full, data_dir)):
                        self.files.setdefault(k, full)
                elif low.endswith(".zip"):
                    try:
                        zf = zipfile.ZipFile(full)
                    except Exception as ex:  # BadZipFile, truncated download, unsupported archive, permissions
                        log(f"[data] WARNING: skipping unreadable zip {full}: {type(ex).__name__}")
                        continue
                    n_mp4 = 0
                    for info in zf.infolist():
                        if info.filename.lower().endswith(".mp4"):
                            keys = _suffix_keys(info.filename)
                            for k in keys:
                                self.members.setdefault(k, (full, info.filename))
                            n_mp4 += bool(keys)
                    n_members += n_mp4
                    if n_mp4:
                        self._zips[full] = zf
                    else:
                        zf.close()
        log(f"[data] indexed {n_files} extracted mp4 files and {n_members} zipped mp4 members")
        if n_files + n_members == 0:
            log("[data] WARNING: no clip videos found below the data directory; the duration stage cannot run")

    def close(self):
        for zf in self._zips.values():
            zf.close()
        self._zips = {}

    def read(self, key):
        if key in self.files:
            with open(self.files[key], "rb") as fh:
                return fh.read()
        if key in self.members:
            zpath, member = self.members[key]
            return self._zips[zpath].read(member)
        return None

    def _seconds(self, clip_key, mod):
        parts = [p for p in clip_key.replace("\\", "/").split("/") if p not in ("", ".")]
        for j in range(len(parts), 0, -1):
            key = "/".join(parts[-j:] + [mod, mod + ".mp4"]).lower()
            if key in self.files or key in self.members:
                try:
                    data = self.read(key)
                    return mp4_seconds(data) if data else None
                except Exception as ex:  # corrupt member (CRC), truncated or odd MP4, unsupported compression
                    log(f"[data] WARNING: unreadable video {key}: {type(ex).__name__}")
                    return None
        return None

    def duration_of(self, clip_key):
        """Duration of one clip from the first modality that has a readable video."""
        for mod in MODALITY_PREFERENCE:
            sec = self._seconds(clip_key, mod)
            if sec:
                return sec
        return None

    def scene_durations(self, clip_keys):
        """Durations of all clips of one scene from ONE modality (the model is fitted on IR), or None."""
        for mod in MODALITY_PREFERENCE:
            ds = [self._seconds(k, mod) for k in clip_keys]
            if all(d and d > 0 for d in ds):
                return ds
        return None
