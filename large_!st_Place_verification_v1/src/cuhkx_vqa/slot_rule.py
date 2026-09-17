"""Stage 2 - temporal-order questions whose action pairs are not covered by the matched training script.

Substitute-slot rule: two actions that are sequenced in many common training scenes but never appear together
in one training sequence question fill the same slot of a scene script (e.g. Reading / Turning a page). An
action Y missing from the script's precedence closure takes the order of such partners Z against X. The
answer is the highest-scoring permutation (decoder score) among the orders consistent with the closure plus
the slot-implied pairs. Validated on held-out training users: 42/43 missing pairs.
"""
import itertools
from collections import Counter, defaultdict

from .common import L, clip_of, closure, opts, trial_of, user_of


class SlotStats:
    def __init__(self, coq, sc_occ):
        self.coq = coq          # Counter over frozenset({a, b}) -> co-occurrence count in training sequence answers
        self.sc_occ = sc_occ    # action -> set of (user, scene) where it was sequenced

    @classmethod
    def fit(cls, rows):
        coq = Counter()
        sc_occ = defaultdict(set)
        for r in rows:
            if r["source"] != "HAU" or r["category"] != "sequence":
                continue
            k = clip_of(r["path"])
            o = [r[l] for l in r["answer"]]
            for a in o:
                sc_occ[a].add((user_of(k), trial_of(k).rsplit("-", 1)[0]))
            for i in range(4):
                for j in range(i + 1, 4):
                    coq[frozenset((o[i], o[j]))] += 1
        return cls(coq, sc_occ)

    def to_dict(self):
        return {"coq": [sorted(k) + [n] for k, n in self.coq.items()],
                "sc_occ": {a: sorted([list(x) for x in v]) for a, v in self.sc_occ.items()}}

    @classmethod
    def from_dict(cls, d):
        coq = Counter()
        for a, b, n in d["coq"]:
            coq[frozenset((a, b))] = n
        sc_occ = defaultdict(set, {a: set((u, xy) for u, xy in v) for a, v in d["sc_occ"].items()})
        return cls(coq, sc_occ)


def _pairs_of(o):
    return {(o[i], o[j]) for i in range(len(o)) for j in range(i + 1, len(o))}


def apply(K, stats, hau, info, pred):
    pred = dict(pred)
    changed = {}
    for seg, sk, ratio, best in info:
        if sk is None:
            continue
        seq = [r for i in seg for r in hau[i][1] if r["category"] == "sequence" and len(opts(r)) == 4]
        if not seq:
            continue
        Ptw = K.scripts[sk]["P"]
        for r in seq:
            o = opts(r)
            known = sum(((a, b) in Ptw) or ((b, a) in Ptw) for a, b in itertools.combinations(o, 2))
            if known == 6:
                continue
            extra = set()
            for r2 in seq:
                if r2 is r:
                    continue
                o2 = opts(r2)
                if sum(((a, b) in Ptw) or ((b, a) in Ptw) for a, b in itertools.combinations(o2, 2)) == 6:
                    extra |= _pairs_of([o2[L.index(c)] for c in pred[r2["qa_id"]]])
            P = closure(set(Ptw) | extra)
            seqd = set(x for p in P for x in p)
            implied = set()
            for Y, X in itertools.permutations(o, 2):
                if (Y, X) in P or (X, Y) in P or Y in seqd or X not in seqd:
                    continue
                partners = [Z for Z in seqd if Z not in (X, Y) and stats.coq[frozenset((Y, Z))] == 0
                            and len(stats.sc_occ[Y] & stats.sc_occ[Z]) >= 2 and ((Z, X) in P or (X, Z) in P)]
                if not partners:
                    continue
                v = sum(1 if (Z, X) in P else -1 for Z in partners)
                if v > 0:
                    implied.add((Y, X))
                elif v < 0:
                    implied.add((X, Y))
            C = closure(P | implied)
            cons = []
            for perm in itertools.permutations(range(4)):
                s = [o[i] for i in perm]
                if all((s[j], s[i]) not in C for i in range(4) for j in range(i + 1, 4)):
                    cons.append("".join(L[i] for i in perm))
            if cons:
                sc = K.seq_scores(o, Ptw)
                new = max(cons, key=lambda p: sc[p])
                if new != pred[r["qa_id"]]:
                    changed[r["qa_id"]] = (pred[r["qa_id"]], new)
                pred[r["qa_id"]] = new
    return pred, changed
