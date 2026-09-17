"""Shared helpers (identical semantics to the research decoder used for the Kaggle submissions)."""
import re
from collections import Counter

L = "ABCD"

SLOW = {"slowly", "leisurely", "calmly", "steadily", "gently", "patiently", "unhurriedly", "lazily",
        "peacefully", "quietly", "softly", "relaxedly", "casually", "smoothly", "naturally", "evenly",
        "comfortably", "soothingly", "lightly", "contently"}
FAST = {"quickly", "hurriedly", "hastily", "rapidly", "urgently", "briskly", "frantically", "swiftly",
        "impatiently", "anxiously", "nervously", "restlessly", "tensely", "tensly", "hasitly",
        "forcefully", "eagerly"}


def mclass(a):
    a = a.lower()
    return "slow" if a in SLOW else ("fast" if a in FAST else "careful")


def clip_of(p):
    parts = p.replace("\\", "/").split("/")
    if parts[-1].endswith(".mp4"):
        parts = parts[:-2]
    return "/".join(parts)


def user_of(c):
    m = re.search(r"user(\d+)", c)
    return int(m.group(1)) if m else -1


def trial_of(c):
    return c.split("/")[-1]


def test_id(c):
    m = re.search(r"LM_test_(\d+)", c)
    return int(m.group(1)) if m else None


def opts(r):
    return [r[l] for l in L if (r.get(l) or "") != ""]


def atoms(s):
    return frozenset(x.strip() for x in s.split(",") if x.strip())


def emo_opts(rows):
    return set(x for r in rows if r["category"] == "emotion" for x in opts(r))


def certain_actions(rows_list):
    S = set()
    for rows in rows_list:
        for r in rows:
            if r["category"] == "sequence":
                S |= set(opts(r))
    return S


def support(rows_list):
    sup = Counter()
    for rows in rows_list:
        for r in rows:
            c = r["category"]
            o = opts(r)
            if c in ("single", "multi", "sequence"):
                for x in o:
                    sup[x] += 1
            elif c == "combination":
                for x in o:
                    for a in atoms(x):
                        sup[a] += 0.5
    return sup


def likely_actions(rows_list):
    sup = support(rows_list)
    return certain_actions(rows_list) | {a for a, n in sup.items() if n >= 2}


def closure(pairs):
    """Transitive closure of a precedence relation given as a set of (a, b) pairs."""
    P = set(pairs)
    nodes = set(a for a, _ in P) | set(b for _, b in P)
    changed = True
    while changed:
        changed = False
        for a, b in list(P):
            for c in nodes:
                if (b, c) in P and (a, c) not in P and a != c:
                    P.add((a, c))
                    changed = True
    return P
