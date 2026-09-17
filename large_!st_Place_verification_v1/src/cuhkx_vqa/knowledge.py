"""Knowledge learned from the labelled TRAINING questions (the model state of the text decoder).

Built by src/train.py from Training/training_qa.csv and frozen into artifacts/model_state.json, so that
inference needs no training data. Round-trips exactly (dict insertion order is preserved because it
breaks ties in script matching and in HARn object look-ups).
"""
import itertools
import math
from collections import Counter, defaultdict

from .common import L, atoms, clip_of, closure, mclass, opts, trial_of, user_of


class NaiveBayesOptions:
    """P(answer | the other options) from training questions (used for HARn single / object questions)."""

    def __init__(self, ans_c, co, V):
        self.ans_c = Counter(ans_c)
        self.co = defaultdict(Counter, {a: Counter(v) for a, v in co.items()})
        self.V = V

    @classmethod
    def fit(cls, rows):
        ans_c = Counter()
        co = defaultdict(Counter)
        for r in rows:
            a = r[r["answer"]]
            ans_c[a] += 1
            for x in opts(r):
                if x != a:
                    co[a][x] += 1
        V = max(1, len(set(x for r in rows for x in opts(r))))
        return cls(ans_c, co, V)

    def __call__(self, o):
        out = []
        for a in o:
            s = math.log(self.ans_c[a] + 0.5)
            for d in o:
                if d != a:
                    s += math.log((self.co[a][d] + 0.1) / (self.ans_c[a] + 0.1 * self.V))
            out.append(s)
        return out

    def to_dict(self):
        return {"ans_c": dict(self.ans_c), "co": {a: dict(v) for a, v in self.co.items()}, "V": self.V}

    @classmethod
    def from_dict(cls, d):
        return cls(d["ans_c"], d["co"], d["V"])


class Knowledge:
    def __init__(self):
        self.scripts = {}
        self.Pg = Counter()
        self.zc = defaultdict(Counter)
        self.zcl = defaultdict(Counter)
        self.emo_ans = Counter()
        self.emo_opt = Counter()
        self.labels = []
        self.a2l = defaultdict(set)
        self.l2obj = defaultdict(Counter)
        self.nb_s = None
        self.nb_o = None

    # ------------------------------------------------------------------ training
    @classmethod
    def fit(cls, rows):
        K = cls()
        clips = defaultdict(list)
        for r in rows:
            clips[clip_of(r["path"])].append(r)
        for k, rs in clips.items():
            if not k.startswith("HAU/"):
                continue
            u = user_of(k)
            xy, z = trial_of(k).rsplit("-", 1)
            z = int(z)
            s = K.scripts.setdefault((u, xy), {"A": set(), "E": {}, "P": set(), "n": 0})
            s["n"] += 1
            for r in rs:
                c, a = r["category"], r["answer"]
                if c == "single":
                    s["A"].add(r[a])
                elif c == "multi":
                    s["A"] |= {r[l] for l in a}
                elif c == "combination":
                    s["A"] |= set(atoms(r[a]))
                elif c == "sequence":
                    o = [r[l] for l in a]
                    s["A"] |= set(o)
                    for i in range(4):
                        for j in range(i + 1, 4):
                            s["P"].add((o[i], o[j]))
                            K.Pg[(o[i], o[j])] += 1
                elif c == "emotion":
                    s["E"][z] = r[a]
                    K.zc[r[a].lower()][z] += 1
                    K.emo_ans[r[a].lower()] += 1
                    for x in opts(r):
                        K.emo_opt[x.lower()] += 1
        for s in K.scripts.values():
            s["P"] = closure(s["P"])
        for a, c in K.zc.items():
            for z, n in c.items():
                K.zcl[mclass(a)][z] += n
        harn = [r for k, rs in clips.items() if k.startswith("HARn/") for r in rs]
        K.labels = sorted(set(r["path"].split("/")[1] for r in harn))
        for r in harn:
            lab = r["path"].split("/")[1]
            if r["category"] == "single":
                K.a2l[r[r["answer"]]].add(lab)
            else:
                K.l2obj[lab][r[r["answer"]]] += 1
        K.nb_s = NaiveBayesOptions.fit([r for r in harn if r["category"] == "single"])
        K.nb_o = NaiveBayesOptions.fit([r for r in harn if r["category"] == "object_interaction"])
        return K

    # ------------------------------------------------------------------ inference helpers
    def pz(self, adv, z):
        c = self.zc.get(adv.lower(), Counter())
        n = sum(c.values())
        cl = self.zcl[mclass(adv)]
        ncl = sum(cl.values())
        return (c[z] + 2 * (cl[z] + 1) / (ncl + 3)) / (n + 2)

    def answer_rate(self, adv):
        return (self.emo_ans[adv.lower()] + 1) / (self.emo_opt[adv.lower()] + 4)

    def assign(self, advs, positions):
        advs = list(advs)
        n = len(positions)
        if len(advs) < n:
            advs = advs + advs * n
        best = None
        for perm in itertools.permutations(range(len(advs)), n):
            s = sum(math.log(self.pz(advs[perm[i]], positions[i])) for i in range(n))
            if best is None or s > best[0]:
                best = (s, perm)
        return [advs[best[1][i]] for i in range(n)]

    def seq_scores(self, o, Ptw, wt=5.0):
        """Score of every permutation (letters string) of a temporal-order question."""
        out = {}
        for perm in itertools.permutations(range(len(o))):
            s = 0.0
            for i in range(len(o)):
                for j in range(i + 1, len(o)):
                    a, b = o[perm[i]], o[perm[j]]
                    s += wt * (((a, b) in Ptw) - ((b, a) in Ptw))
                    s += math.log((self.Pg[(a, b)] + 1) / (self.Pg[(a, b)] + self.Pg[(b, a)] + 2))
            out["".join(L[i] for i in perm)] = s
        return out

    def seq_order(self, o, Ptw, wt=5.0):
        best = None
        for perm in itertools.permutations(range(len(o))):
            s = 0.0
            for i in range(len(o)):
                for j in range(i + 1, len(o)):
                    a, b = o[perm[i]], o[perm[j]]
                    s += wt * (((a, b) in Ptw) - ((b, a) in Ptw))
                    s += math.log((self.Pg[(a, b)] + 1) / (self.Pg[(a, b)] + self.Pg[(b, a)] + 2))
            if best is None or s > best[0]:
                best = (s, "".join(L[i] for i in perm))
        return best[1]

    # ------------------------------------------------------------------ serialisation
    def to_dict(self):
        return {
            "scripts": [{"user": u, "xy": xy, "A": sorted(s["A"]), "E": {str(z): a for z, a in s["E"].items()},
                         "P": sorted([list(p) for p in s["P"]]), "n": s["n"]} for (u, xy), s in self.scripts.items()],
            "Pg": [[a, b, n] for (a, b), n in self.Pg.items()],
            "zc": {a: {str(z): n for z, n in c.items()} for a, c in self.zc.items()},
            "zcl": {a: {str(z): n for z, n in c.items()} for a, c in self.zcl.items()},
            "emo_ans": dict(self.emo_ans),
            "emo_opt": dict(self.emo_opt),
            "labels": list(self.labels),
            "a2l": {x: sorted(v) for x, v in self.a2l.items()},
            "l2obj": {lab: [[o, n] for o, n in c.items()] for lab, c in self.l2obj.items()},
            "nb_s": self.nb_s.to_dict(),
            "nb_o": self.nb_o.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        K = cls()
        for s in d["scripts"]:
            K.scripts[(s["user"], s["xy"])] = {"A": set(s["A"]), "E": {int(z): a for z, a in s["E"].items()},
                                              "P": set(tuple(p) for p in s["P"]), "n": s["n"]}
        K.Pg = Counter({(a, b): n for a, b, n in d["Pg"]})
        K.zc = defaultdict(Counter, {a: Counter({int(z): n for z, n in c.items()}) for a, c in d["zc"].items()})
        K.zcl = defaultdict(Counter, {a: Counter({int(z): n for z, n in c.items()}) for a, c in d["zcl"].items()})
        K.emo_ans = Counter(d["emo_ans"])
        K.emo_opt = Counter(d["emo_opt"])
        K.labels = list(d["labels"])
        K.a2l = defaultdict(set, {x: set(v) for x, v in d["a2l"].items()})
        K.l2obj = defaultdict(Counter)
        for lab, pairs in d["l2obj"].items():
            c = Counter()
            for o, n in pairs:
                c[o] = n
            K.l2obj[lab] = c
        K.nb_s = NaiveBayesOptions.from_dict(d["nb_s"])
        K.nb_o = NaiveBayesOptions.from_dict(d["nb_o"])
        return K
