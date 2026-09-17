"""Stage 4 - emotion question of a scene with a single clip (repeat index unknown, adverb set unknown).

Count model P(repeat z | adverb) with back-off to a generic manner lexicon; marginalises over z in {2, 3} and over
injective assignments of three of the four options to z = 1, 2, 3. Held-out training users: 0.497 vs 0.405 for the
text decoder's default rule.
"""
import itertools
import math
from collections import Counter, defaultdict

from .common import L, mclass, opts

LEX = {
    "relaxed": "slowly leisurely gently casually unhurriedly relaxedly lazily patiently peacefully softly lightly "
               "soothingly contently comfortably gracefully quietly tenderly delicately serenely idly languidly "
               "sluggishly placidly tranquilly joyfully happily cheerfully",
    "careful": "carefully meticulously cautiously deliberately warily gingerly painstakingly",
    "steady": "steadily calmly smoothly evenly naturally orderly attentively intently neatly seriously methodically "
              "thoroughly precisely diligently earnestly confidently firmly absentmindedly systematically "
              "consistently focusedly purposefully efficiently skillfully normally regularly",
    "fast": "quickly hurriedly hastily rapidly briskly swiftly speedily promptly forcefully energetically vigorously "
            "eagerly suddenly abruptly",
    "agitated": "urgently frantically impatiently anxiously nervously restlessly tensely hectically agitatedly "
                "worriedly frenziedly desperately",
}
LEXC = {w: c for c, ws in LEX.items() for w in ws.split()}
_norm_cache = {}


def _ed(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(a):
    a0 = a.strip().lower()
    if a0 in _norm_cache:
        return _norm_cache[a0]
    out = a0
    if a0 not in LEXC:
        best = min(((_ed(a0, w), w) for w in LEXC if w[0] == a0[:1]), default=(99, a0))
        if best[0] <= 2:
            out = best[1]
    _norm_cache[a0] = out
    return out


def lclass(a):
    return LEXC.get(norm(a), mclass(a).replace("slow", "relaxed").replace("careful", "steady"))


class CountModel:
    def __init__(self, c, cc, alpha=2.0, beta=3.0):
        self.c = defaultdict(Counter, {a: Counter(v) for a, v in c.items()})
        self.cc = defaultdict(Counter, {a: Counter(v) for a, v in cc.items()})
        self.alpha, self.beta = alpha, beta

    @classmethod
    def fit(cls, rows, alpha=2.0, beta=3.0):
        from .common import clip_of, trial_of, user_of
        sc = defaultdict(dict)
        for r in rows:
            if r["source"] != "HAU" or r["category"] != "emotion":
                continue
            k = clip_of(r["path"])
            u = user_of(k)
            xy, z = trial_of(k).rsplit("-", 1)
            uu = u - 10 if u in (16, 17, 18, 19) else u
            sc[(uu, xy)][int(z)] = r[r["answer"]]
        c = defaultdict(Counter)
        cc = defaultdict(Counter)
        for E in dict(sc).values():
            for z, a in E.items():
                c[norm(a)][z] += 1
                cc[lclass(a)][z] += 1
        return cls(c, cc, alpha, beta)

    def lw(self, a, z):
        cl = self.cc[lclass(a)]
        pc = (cl[z] + self.beta / 3) / (sum(cl.values()) + self.beta)
        c = self.c[norm(a)]
        return math.log((c[z] + self.alpha * pc) / (sum(c.values()) + self.alpha))

    def to_dict(self):
        return {"c": {a: {str(z): n for z, n in v.items()} for a, v in self.c.items()},
                "cc": {a: {str(z): n for z, n in v.items()} for a, v in self.cc.items()},
                "alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, d):
        c = {a: {int(z): n for z, n in v.items()} for a, v in d["c"].items()}
        cc = {a: {int(z): n for z, n in v.items()} for a, v in d["cc"].items()}
        return cls(c, cc, d["alpha"], d["beta"])


def decide(model, o, zprior=((2, 0.5), (3, 0.5))):
    if len(o) < 3:
        return None
    m = Counter()
    for zq, pzz in zprior:
        post = [(sum(model.lw(a, z) for a, z in zip(perm, (1, 2, 3))), perm) for perm in itertools.permutations(o, 3)]
        mx = max(s for s, _ in post)
        tot = sum(math.exp(s - mx) for s, _ in post)
        for s, perm in post:
            m[perm[zq - 1]] += pzz * math.exp(s - mx) / tot
    return max(o, key=lambda a: m[a])


def apply(model, hau, info, pred):
    pred = dict(pred)
    changed = {}
    for seg, sk, ratio, best in info:
        if sk is not None or len(seg) != 1:
            continue
        for r in hau[seg[0]][1]:
            if r["category"] != "emotion":
                continue
            o = opts(r)
            g = decide(model, o)
            if g is None:
                continue
            new = L[o.index(g)]
            if pred.get(r["qa_id"]) != new:
                changed[r["qa_id"]] = (pred.get(r["qa_id"]), new)
            pred[r["qa_id"]] = new
    return pred, changed
