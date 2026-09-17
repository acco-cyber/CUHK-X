"""Stage 5 - emotion questions of scenes without a matched script, using clip DURATION.

The three repeats of a scene perform the same actions in different manners (e.g. slowly / steadily / hurriedly);
the hurried repeat is the shortest recording. Duration is read from the MP4 'mvhd' box of each clip's IR video.

Model (fitted on training scenes): log d = mu_scene + e(adverb) + f(repeat) + N(0, sigma^2), e = class effect +
ridge-shrunk adverb deviation. Decoding scores every admissible assignment h of the scene adverbs to the clips:
  sum_z log K.pz(h[z], z) + w * sum_clips -(centred log d - centred prediction)^2 / (2 sigma^2).
Held-out training users (no twin): 3-repeat 0.819 -> 0.922, two-repeat 0.760 -> 0.913; test twin scenes decoded
blind 0.796 -> 0.959.
"""
import itertools
import math
import struct
from collections import defaultdict

from .common import L, mclass, opts

# ----------------------------------------------------------------------------------------------- MP4 duration


def _boxes(buf, start, end):
    end = min(end, len(buf))
    p = start
    while p + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[p:p + 8])
        hdr = 8
        if size == 1:
            if p + 16 > end:
                break
            size = struct.unpack(">Q", buf[p + 8:p + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - p
        if size < hdr:
            break
        yield typ, p + hdr, min(p + size, end)
        p += size


def mp4_seconds(data):
    """Duration in seconds from the movie header (mvhd) of an MP4 file given as bytes; None if absent or truncated."""
    for typ, s, e in _boxes(data, 0, len(data)):
        if typ != b"moov":
            continue
        for t2, s2, e2 in _boxes(data, s, e):
            if t2 == b"mvhd":
                if e2 - s2 < (32 if (e2 > s2 and data[s2] == 1) else 20):
                    return None
                ver = data[s2]
                if ver == 1:
                    ts, dur = struct.unpack(">IQ", data[s2 + 20:s2 + 32])
                else:
                    ts, dur = struct.unpack(">II", data[s2 + 12:s2 + 20])
                return dur / ts if ts else None
    return None


# ----------------------------------------------------------------------------------------------- model


class DurModel:
    def __init__(self, c, delta, f, sigma):
        self.c = dict(c)
        self.delta = dict(delta)
        self.f = {int(z): v for z, v in f.items()}
        self.sigma = sigma

    def eff(self, adv):
        a = adv.lower()
        return self.c[mclass(a)] + self.delta.get(a, 0.0)

    def loglik(self, advs, zs, durs):
        n = len(durs)
        logs = [math.log(x) for x in durs]
        mu = sum(logs) / n
        pred = [self.eff(a) + self.f[z] for a, z in zip(advs, zs)]
        pm = sum(pred) / n
        return sum(-0.5 * ((lg - mu) - (p - pm)) ** 2 / self.sigma ** 2 for lg, p in zip(logs, pred))

    def to_dict(self):
        return {"c": self.c, "delta": self.delta, "f": {str(z): v for z, v in self.f.items()}, "sigma": self.sigma}

    @classmethod
    def from_dict(cls, d):
        return cls(d["c"], d["delta"], d["f"], d["sigma"])

    # ------------------------------------------------------------------ training (ridge, exact Gauss-Jordan)
    @classmethod
    def fit(cls, scenes, lam=4.0):
        """scenes: {(user, xy): {z: {"d": seconds, "adv": adverb}}}"""
        obs = []
        advs = set()
        for (u, xy), d in scenes.items():
            if len(d) < 2:
                continue
            zs = sorted(d)
            obs.append([(d[z]["adv"].lower(), z, math.log(d[z]["d"])) for z in zs])
            advs |= {d[z]["adv"].lower() for z in zs}
        classes = ["slow", "careful", "fast"]
        advs = sorted(advs)
        idx = {}
        for c in classes:
            idx[("c", c)] = len(idx)
        for a in advs:
            idx[("a", a)] = len(idx)
        for z in (1, 2, 3):
            idx[("z", z)] = len(idx)
        P = len(idx)
        XtX = [[0.0] * P for _ in range(P)]
        Xty = [0.0] * P
        rows = []
        for scene in obs:
            phis = []
            for a, z, lg in scene:
                v = defaultdict(float)
                v[idx[("c", mclass(a))]] += 1
                v[idx[("a", a)]] += 1
                v[idx[("z", z)]] += 1
                phis.append(v)
            n = len(scene)
            mean_phi = defaultdict(float)
            for v in phis:
                for k, x in v.items():
                    mean_phi[k] += x / n
            mu = sum(lg for _, _, lg in scene) / n
            for (a, z, lg), v in zip(scene, phis):
                x = dict(mean_phi)
                for k in list(x):
                    x[k] = -x[k]
                for k, val in v.items():
                    x[k] = x.get(k, 0.0) + val
                y = lg - mu
                rows.append((x, y))
                items = [(k, val) for k, val in x.items() if abs(val) > 1e-12]
                for k1, v1 in items:
                    Xty[k1] += v1 * y
                    for k2, v2 in items:
                        XtX[k1][k2] += v1 * v2
        for k, i in idx.items():
            XtX[i][i] += lam if k[0] == "a" else 1e-3
        beta = _solve(XtX, Xty)
        c = {cl: beta[idx[("c", cl)]] for cl in classes}
        delta = {a: beta[idx[("a", a)]] for a in advs}
        f = {z: beta[idx[("z", z)]] for z in (1, 2, 3)}
        res = [y - sum(val * beta[k] for k, val in x.items()) for x, y in rows]
        sigma = math.sqrt(sum(r * r for r in res) / max(1, len(res))) or 0.3
        return cls(c, delta, f, sigma)


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= p
        for r in range(n):
            if r != col and M[r][col] != 0:
                fct = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= fct * M[col][j]
    return [M[i][n] for i in range(n)]


# ----------------------------------------------------------------------------------------------- decoding


def _hyps(pool, zs):
    out = []
    for perm in itertools.permutations(pool, len(zs)):
        h = dict(zip(zs, perm))
        if tuple(zs) == (2, 3) and len(pool) == 3:
            h[1] = next(a for a in pool if a not in perm)
        out.append(h)
    return out


def decode(K, model, pool, zs, durs, w):
    scored = []
    for h in _hyps(pool, zs):
        s = sum(math.log(K.pz(a, z)) for z, a in h.items())
        if w:
            s += w * model.loglik([h[z] for z in zs], zs, durs)
        scored.append((s, h))
    scored.sort(key=lambda x: -x[0])
    return scored[0][1], scored


def apply(K, model, hau, info, pred, scene_durations, weight=1.0, two_repeat_positions=(2, 3)):
    """scene_durations([clip_key, ...]) -> [seconds, ...] from one modality, or None.
    Scenes with a missing duration keep the text prediction."""
    pred = dict(pred)
    changed = {}
    for seg, sk, ratio, best in info:
        if sk is not None or len(seg) < 2 or len(seg) > 3:
            continue
        emo = []
        for i in seg:
            e = [r for r in hau[i][1] if r["category"] == "emotion" and opts(r)]
            if not e:
                break
            emo.append(e[0])
        if len(emo) != len(seg):
            continue
        zs = (1, 2, 3) if len(seg) == 3 else tuple(two_repeat_positions)
        pool = sorted(set.intersection(*[set(opts(r)) for r in emo]))
        if len(pool) != 3:
            continue
        durs = scene_durations([hau[i][0] for i in seg])
        if not durs or any(d is None or d <= 0 for d in durs):
            continue
        bestA, _ = decode(K, model, pool, zs, durs, weight)
        for z, r in zip(zs, emo):
            o = opts(r)
            if bestA[z] not in o:
                continue
            new = L[o.index(bestA[z])]
            if pred.get(r["qa_id"]) != new:
                changed[r["qa_id"]] = (pred.get(r["qa_id"]), new)
            pred[r["qa_id"]] = new
    return pred, changed
