"""Stage 3 - joint consistency decoding of the action questions in scenes WITHOUT a matched training script.

Training regularities: single-question distractors are never scene actions (1/2427); multi false options are
never scene actions (1/1729); the correct combination's atoms are scene actions. The text decoder picks each
question independently; here all single and combination answers of a scene are chosen jointly so that no action
is both a chosen positive and a rejected single option (then a penalty for wrong combination options that
become fully positive). Multi answers = options in positives minus negatives.
Held-out training users: +51/3525 (3-repeat) and +49/2371 (two-repeat) with 0-3 answers broken.
"""
import itertools

from .common import L, atoms, certain_actions, opts, support

CFG = dict(c2=1.0, cs=0.1, ws=1.0, V=100.0, wW=0.5)
MAX_QUESTIONS = 7


def joint_segment(seg_rows, cfg=CFG):
    allrows = [r for rows in seg_rows for r in rows]
    sup = support(seg_rows)
    cert = certain_actions(seg_rows)
    singles = [r for r in allrows if r["category"] == "single" and opts(r)]
    combs = [r for r in allrows if r["category"] == "combination" and opts(r)]
    qs = singles + combs
    if not qs:
        return {}
    choices = []
    for r in qs:
        o = opts(r)
        if r["category"] == "single":
            sc = [sup[x] - 1e-3 * i for i, x in enumerate(o)]
        else:
            sc = [cfg["c2"] * sum(sup[a] >= 2 for a in atoms(x)) + cfg["cs"] * sum(sup[a] for a in atoms(x)) - 1e-3 * i
                  for i, x in enumerate(o)]
        choices.append((r, o, sc))
    best = None
    for H in itertools.product(*[range(len(o)) for _, o, _ in choices]):
        pos = set(cert)
        neg = set()
        s = 0.0
        for (r, o, sc), i in zip(choices, H):
            s += sc[i] * (cfg["ws"] if r["category"] == "single" else 1.0)
            if r["category"] == "single":
                pos.add(o[i])
                neg |= set(o) - {o[i]}
            else:
                pos |= set(atoms(o[i]))
        V = len(pos & neg)
        W = 0
        if cfg["wW"]:
            for (r, o, sc), i in zip(choices, H):
                if r["category"] == "combination":
                    W += sum(1 for j, x in enumerate(o) if j != i and atoms(x) <= pos)
        tot = s - cfg["V"] * V - cfg["wW"] * W
        if best is None or tot > best[0]:
            best = (tot, H, pos, neg)
    _, H, pos, neg = best
    out = {}
    for (r, o, sc), i in zip(choices, H):
        out[r["qa_id"]] = L[i]
    S = pos - neg
    for r in allrows:
        if r["category"] == "multi" and opts(r):
            o = opts(r)
            sel = [L[i] for i, x in enumerate(o) if x in S]
            if not sel:
                cand = [i for i, x in enumerate(o) if x not in neg] or list(range(len(o)))
                sel = [L[max(cand, key=lambda i: (sup[o[i]], -i))]]
            out[r["qa_id"]] = "".join(sel)
    return out


def apply(hau, info, pred):
    pred = dict(pred)
    changed = {}
    for seg, sk, ratio, best in info:
        if sk is not None:
            continue
        seg_rows = [hau[i][1] for i in seg]
        if sum(1 for rows in seg_rows for r in rows if r["category"] in ("single", "combination")) > MAX_QUESTIONS:
            continue
        for q, a in joint_segment(seg_rows).items():
            if pred.get(q) != a:
                changed[q] = (pred.get(q), a)
            pred[q] = a
    return pred, changed
