"""Stage 1 - structure-aware text decoder (HAU scenes and HARn clips).

HAU: consecutive clips (in LM_test id order) are grouped into scenes; each scene is matched against the
training scene scripts; single / multi / combination / sequence / emotion answers are decoded jointly
from the matched script (if any) and from within-scene consistency of the questions.
HARn: clips are label-sorted in id order, so labels are assigned by a (softly) monotone dynamic programme.
"""
from .common import (L, atoms, certain_actions, emo_opts, likely_actions, opts, support)

Z_MAPS = {1: [(1,), (2,), (3,)], 2: [(1, 2), (1, 3), (2, 3)], 3: [(1, 2, 3)]}


def segment(clips):
    segs = []
    cur = [0]
    for i in range(1, len(clips)):
        if len(cur) < 3 and len(emo_opts(clips[i][1]) & emo_opts(clips[i - 1][1])) >= 3:
            cur.append(i)
        else:
            segs.append(cur)
            cur = [i]
    segs.append(cur)
    changed = True
    while changed:
        changed = False
        for si, seg in enumerate(segs):
            if len(seg) != 1:
                continue
            me = clips[seg[0]][1]
            best = None
            for nb in (si - 1, si + 1):
                if nb < 0 or nb >= len(segs) or len(segs[nb]) >= 3:
                    continue
                nrows = [clips[i][1] for i in segs[nb]]
                ov_a = len(likely_actions([me]) & likely_actions(nrows))
                ov_e = len(emo_opts(me) & emo_opts(nrows[0]))
                if ov_a >= 2 or ov_e >= 2:
                    key = (ov_a + ov_e, nb)
                    if best is None or key > best[0]:
                        best = (key, nb)
            if best is not None:
                nb = best[1]
                merged = sorted(segs[nb] + seg)
                lo, hi = min(si, nb), max(si, nb)
                segs = segs[:lo] + [merged] + segs[hi + 1:]
                changed = True
                break
    return segs


def soft(rows, A, E, S_cert=frozenset()):
    A2 = A | S_cert
    s = 0.0
    for r in rows:
        c = r["category"]
        o = opts(r)
        if c == "sequence":
            unseen = len(set(o) - A)
            s += 3 if unseen == 0 else (1 if unseen == 1 else -4)
        elif c == "single":
            k = len(set(o) & A2)
            s += 1 if k == 1 else (-1 if k == 0 else 0)
        elif c == "multi":
            s += 1 if (set(o) & A2) else -1
        elif c == "combination":
            k = sum(atoms(x) <= A2 for x in o)
            s += 1 if k == 1 else (-1 if k == 0 else 0)
        elif c == "emotion":
            k = len(set(o) & E)
            s += {0: -3, 1: -1, 2: 1}.get(k, 2)
    return s


def max_soft(rows):
    w = {"sequence": 3, "single": 1, "multi": 1, "combination": 1, "emotion": 2}
    return sum(w.get(r["category"], 0) for r in rows)


def score_script(seg_rows, s):
    Eall = set(s["E"].values())
    S_cert = certain_actions(seg_rows)
    base = sum(soft(rows, s["A"], Eall, S_cert) for rows in seg_rows)
    n = len(seg_rows)
    zb = max(sum((2 if s["E"].get(zz) in emo_opts(rows) else -2) for zz, rows in zip(mp, seg_rows))
             for mp in Z_MAPS[n])
    return base + zb, base


def scene_key(xy):
    return tuple(int(v) for v in xy.split("-"))


def match_segments(K, clips, segs, ratio_thresh=0.85, repair_thresh=0.45):
    out = []
    for seg in segs:
        seg_rows = [clips[i][1] for i in seg]
        mx = sum(max_soft(rows) for rows in seg_rows)
        best = None
        for sk, s in K.scripts.items():
            tot, base = score_script(seg_rows, s)
            if best is None or tot > best[0]:
                best = (tot, base, sk)
        ratio = best[1] / mx if mx else 0
        out.append([best[2] if ratio >= ratio_thresh else None, ratio, best[2], mx])
    for i, o in enumerate(out):
        if o[0] is not None:
            continue
        prev = next((out[j][0] for j in range(i - 1, -1, -1) if out[j][0] is not None), None)
        nxt = next((out[j][0] for j in range(i + 1, len(out)) if out[j][0] is not None), None)
        if prev is None and nxt is None:
            continue
        if prev is not None and nxt is not None and prev[0] != nxt[0]:
            continue
        u = prev[0] if prev is not None else nxt[0]
        used = {o2[0] for o2 in out if o2[0] is not None}
        lo = scene_key(prev[1]) if prev is not None else (0,)
        hi = scene_key(nxt[1]) if nxt is not None else (99,)
        cand = [sk for sk in K.scripts if sk[0] == u and sk not in used and lo <= scene_key(sk[1]) <= hi]
        if not cand:
            continue
        seg_rows = [clips[j][1] for j in segs[i]]
        sc = max(((score_script(seg_rows, K.scripts[sk])[1], sk) for sk in cand), key=lambda z: z[0])
        if o[3] and sc[0] / o[3] >= repair_thresh:
            o[0] = sc[1]
            o[1] = sc[0] / o[3]
    return [(o[0], o[1], o[2]) for o in out]


def decode_segment(K, seg_rows, script, opt):
    pred = {}
    S_cert = certain_actions(seg_rows)
    sup = support(seg_rows)
    A_tw = set(script["A"]) if script else set()
    S1 = A_tw | S_cert
    S2 = set(S_cert)
    for rows in seg_rows:
        for r in rows:
            c = r["category"]
            o = opts(r)
            if not o:
                continue
            if c == "single":
                cand = [i for i, x in enumerate(o) if x in S1] if script else []
                if not cand:
                    cand = [i for i, x in enumerate(o) if sup[x] >= 2]
                pool = cand if cand else range(len(o))
                i = max(pool, key=lambda i: (sup[o[i]], -i))
                pred[r["qa_id"]] = L[i]
                S2.add(o[i])
            elif c == "combination":
                if script:
                    sc = [(sum(a in S1 for a in atoms(x)), sum(sup[a] for a in atoms(x)), -i) for i, x in enumerate(o)]
                else:
                    sc = [(sum(sup[a] >= 2 for a in atoms(x)), sum(sup[a] for a in atoms(x)), -i) for i, x in enumerate(o)]
                i = max(range(len(o)), key=lambda i: sc[i])
                pred[r["qa_id"]] = L[i]
                S2 |= set(atoms(o[i]))
    S = (S1 | S2) if (script and opt.get("multi_union", True)) else (S1 if script else S2)
    for rows in seg_rows:
        for r in rows:
            c = r["category"]
            o = opts(r)
            if not o:
                continue
            if c == "multi":
                sel = [L[i] for i, x in enumerate(o) if x in S]
                if not sel:
                    sel = [L[max(range(len(o)), key=lambda i: (sup[o[i]], -i))]]
                pred[r["qa_id"]] = "".join(sel)
            elif c == "sequence":
                pred[r["qa_id"]] = K.seq_order(o, script["P"] if script else set())
    n = len(seg_rows)
    emo = [(pos, r) for pos, rows in enumerate(seg_rows) for r in rows if r["category"] == "emotion" and opts(r)]
    scene_advs = set.intersection(*[emo_opts(rows) for rows in seg_rows]) if n >= 2 else None
    two_map = tuple(opt.get("two_repeat_positions", (2, 3)))
    positions = {3: (1, 2, 3), 2: two_map, 1: (2,)}.get(n, tuple(range(1, n + 1)))
    guesses = {}
    if script and script["E"]:
        for pos, zz in enumerate(positions):
            guesses[pos] = script["E"].get(zz)
    for pos, r in emo:
        o = opts(r)
        g = guesses.get(pos)
        if g in o:
            pred[r["qa_id"]] = L[o.index(g)]
            continue
        if script and script["E"]:
            advs = set(script["E"].values()) & set(o)
        elif scene_advs:
            advs = scene_advs & set(o)
        else:
            advs = set()
        if 2 <= n <= 3 and scene_advs and len(scene_advs) >= 1:
            pool = sorted(scene_advs)
            if len(pool) < n:
                pool = sorted(advs | scene_advs)
            assigned = K.assign(pool, positions)
            g = assigned[pos]
            if g in o:
                pred[r["qa_id"]] = L[o.index(g)]
                continue
        cand = [x for x in o if x in advs] if advs else list(o)
        zz = positions[pos] if pos < len(positions) else 2
        g = max(cand, key=lambda x: K.pz(x, zz) * K.answer_rate(x))
        pred[r["qa_id"]] = L[o.index(g)]
    return pred


def action_ratio(rows, s, S_cert=frozenset()):
    rows2 = [r for r in rows if r["category"] != "emotion"]
    mx = max_soft(rows2)
    return soft(rows2, s["A"], set(), S_cert) / mx if mx else 0.0


def merge_by_script(K, clips, segs, matches, thresh=0.75):
    changed = True
    while changed:
        changed = False
        for si, seg in enumerate(segs):
            if len(seg) != 1:
                continue
            me = clips[seg[0]][1]
            for nb in (si - 1, si + 1):
                if nb < 0 or nb >= len(segs) or len(segs[nb]) >= 3 or matches[nb][0] is None:
                    continue
                s = K.scripts[matches[nb][0]]
                S_cert = certain_actions([clips[i][1] for i in segs[nb]])
                if action_ratio(me, s, S_cert) >= thresh:
                    lo, hi = min(si, nb), max(si, nb)
                    merged = sorted(segs[nb] + seg)
                    segs = segs[:lo] + [merged] + segs[hi + 1:]
                    matches = matches[:lo] + [matches[nb]] + matches[hi + 1:]
                    changed = True
                    break
            if changed:
                break
    return segs, matches


def decode_hau(K, clips, opt):
    """Returns predictions and per-segment info: (clip indices, matched script key or None, ratio, best key)."""
    if not clips:
        return {}, []
    segs = segment(clips)
    matches = match_segments(K, clips, segs, opt.get("ratio_thresh", 0.85), opt.get("repair_thresh", 0.45))
    segs, matches = merge_by_script(K, clips, segs, matches)
    pred = {}
    info = []
    for seg, (sk, ratio, best_sk) in zip(segs, matches):
        seg_rows = [clips[i][1] for i in seg]
        script = K.scripts[sk] if sk else None
        pred.update(decode_segment(K, seg_rows, script, opt))
        info.append((seg, sk, ratio, best_sk))
    return pred, info


def decode_harn(K, clips, pen=12.0):
    if not clips:
        return {}, []
    labels = K.labels
    m = len(labels)
    NEG = -1e9

    def cand_labels(rows, unsupported=-8.0):
        per_q = []
        for r in rows:
            o = opts(r)
            d = {}
            if not o:
                continue
            if r["category"] == "single":
                sc = K.nb_s(o)
                mx = max(sc)
                for i, x in enumerate(o):
                    for lab in K.a2l.get(x, ()):
                        d[lab] = max(d.get(lab, -1e9), sc[i] - mx)
            else:
                sc = K.nb_o(o)
                mx = max(sc)
                for lab, objs in K.l2obj.items():
                    if not objs:
                        continue
                    ob = objs.most_common(1)[0][0]
                    if ob in o:
                        d[lab] = max(d.get(lab, -1e9), sc[o.index(ob)] - mx)
            per_q.append(d)
        c = {}
        for lab in set(l for d in per_q for l in d):
            c[lab] = sum(d.get(lab, unsupported) for d in per_q)
        return c

    cands = [cand_labels(rows) for _, rows in clips]
    n = len(clips)
    dp = [[NEG] * m for _ in range(n)]
    bp = [[-1] * m for _ in range(n)]
    for j in range(m):
        dp[0][j] = cands[0].get(labels[j], NEG if cands[0] else 0.0)
    for i in range(1, n):
        prev = dp[i - 1]
        pre = [None] * m
        best = NEG
        bj = -1
        for j in range(m):
            if prev[j] > best:
                best, bj = prev[j], j
            pre[j] = (best, bj)
        suf = [None] * m
        best = NEG
        bj = -1
        for j in range(m - 1, -1, -1):
            suf[j] = (best, bj)
            if prev[j] > best:
                best, bj = prev[j], j
        for j in range(m):
            sc = cands[i].get(labels[j], NEG if cands[i] else 0.0)
            if sc <= NEG:
                continue
            a, ai = pre[j]
            b, bi = suf[j]
            if b > NEG:
                b -= pen
            if a >= b and a > NEG:
                dp[i][j], bp[i][j] = a + sc, ai
            elif b > NEG:
                dp[i][j], bp[i][j] = b + sc, bi
    j = max(range(m), key=lambda j: dp[n - 1][j])
    path = [j]
    for i in range(n - 1, 0, -1):
        j = bp[i][j]
        path.append(j)
    path = path[::-1]
    pred = {}
    assigned = []
    for i, (k, rows) in enumerate(clips):
        lab = labels[path[i]] if path[i] >= 0 else None
        assigned.append(lab)
        for r in rows:
            o = opts(r)
            if not o:
                continue
            if r["category"] == "single":
                hit = [i2 for i2, x in enumerate(o) if lab in K.a2l.get(x, ())]
                pred[r["qa_id"]] = L[hit[0]] if hit else L[max(range(len(o)), key=lambda i2: K.nb_s(o)[i2])]
            else:
                ob = K.l2obj[lab].most_common(1)[0][0] if (lab and K.l2obj.get(lab)) else None
                pred[r["qa_id"]] = L[o.index(ob)] if ob in o else L[max(range(len(o)), key=lambda i2: K.nb_o(o)[i2])]
    return pred, assigned
