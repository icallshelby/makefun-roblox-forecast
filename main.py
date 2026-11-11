#!/usr/bin/env python3
"""Entry for make.fun bounty roblox-up-and-coming-forecast-01.

Forecasts Roblox's Up-and-Coming top 10 for the 72 hours after the history
ends. Model (fit from /input/history.ndjson alone, standard library only):

  * Each game's future rank is modelled in log space: ln(rank at +h hours)
    is Gaussian with mean a + b*ln(rank now) + c*ln(h) plus a momentum
    shift, and a standard deviation that grows with ln(h). a, b, c and the
    spread curve are fit by least squares over every (read, later read)
    pair in the history; a per-gap off-chart rate is fit the same way.
  * Momentum: a game's recent ln-rank drift (weight 0.6) and its recent
    ln(playing) growth (weight 0.2, opposite sign) are extrapolated, with
    the extrapolation capped at 48 hours.
  * The single static forecast averages the per-place distributions over
    h = 1..72 (the scored window is hourly), then each place is scaled to
    an empirical coverage target: the measured probability, by gap, that
    the holder of that place was already on the origin chart, averaged
    over the window and multiplied by 0.95. What is left unnamed is shared
    by the scorer over the games a read's chart holds that we did not
    name, which is where genuinely new games get their probability.

Deterministic: no randomness, no clock, sorted iteration only.
Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR
(else /output), honouring task.json's places and output file name.
"""
import json
import math
import os
from datetime import datetime

PLACES_FALLBACK = 10
MIN_P = 0.0004
KAPPA_RANK = 0.7      # weight on recent ln-rank drift (retuned 0.6 -> 0.7 on the practice origins)
KAPPA_PLAY = 0.2      # weight on recent ln(playing) growth (rank improves as it grows)
SD_MULT = 1.08        # spread calibration
COV_MULT = 0.95       # coverage calibration (retuned 0.9 -> 0.95 with KAPPA_RANK 0.7)
GAP_CENTRES = [3, 5, 8, 12, 18, 27, 40, 60, 90, 135]


def parse_time(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def load_reads(input_dir):
    reads = []
    with open(os.path.join(input_dir, 'history.ndjson'), encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rank = {}
            playing = {}
            for g in r.get('up_and_coming') or []:
                rank[g['universe_id']] = g['rank']
                playing[g['universe_id']] = g.get('playing') or 0
            if rank:
                reads.append({'t': parse_time(r['at']), 'rank': rank, 'playing': playing})
    reads.sort(key=lambda x: x['t'])
    return reads


def bin_mix(centres, x):
    lx = math.log(max(x, 1e-9))
    lc = [math.log(c) for c in centres]
    if lx <= lc[0]:
        return [(0, 1.0)]
    if lx >= lc[-1]:
        return [(len(centres) - 1, 1.0)]
    for i in range(len(lc) - 1):
        if lc[i] <= lx <= lc[i + 1]:
            w = (lx - lc[i]) / (lc[i + 1] - lc[i])
            return [(i, 1.0 - w), (i + 1, w)]
    return [(len(centres) - 1, 1.0)]


def fit_diffusion(reads):
    """Least-squares fit of ln r' ~ a + b ln r + c ln gap, spread by gap,
    and a per-gap off-chart rate. Returns (a, b, c, sd_of, off_p) or None."""
    xs, ys, gs = [], [], []
    off = {}
    for i in range(len(reads)):
        ri = reads[i]
        for j in range(i + 1, len(reads)):
            rj = reads[j]
            gap = (rj['t'] - ri['t']) / 3600.0
            if gap <= 0:
                continue
            lg = math.log(gap)
            key = round(lg * 2) / 2
            t, o = off.get(key, (0, 0))
            for uid, r0 in ri['rank'].items():
                r1 = rj['rank'].get(uid)
                t += 1
                if r1 is None:
                    o += 1
                else:
                    xs.append(math.log(r0))
                    ys.append(math.log(r1))
                    gs.append(lg)
            off[key] = (t, o)
    n = len(xs)
    if n < 50:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    mg = sum(gs) / n

    def cov(a_, b_, ma, mb):
        return sum((x - ma) * (y - mb) for x, y in zip(a_, b_)) / n

    def var(a_, ma):
        return sum((x - ma) ** 2 for x in a_) / n

    b = cov(xs, ys, mx, my) / var(xs, mx)
    resid = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
    mr = sum(resid) / n
    c = cov(gs, resid, mg, mr) / var(gs, mg)
    a = my - b * mx - c * mg
    sq = [(r - mr) ** 2 for r in resid]
    msq = sum(sq) / n
    csq = cov(gs, sq, mg, msq) / var(gs, mg)
    asq = msq - csq * mg

    def sd_of(lg):
        return math.sqrt(max(0.04, asq + csq * lg))

    keys = sorted(off)

    def off_p(gap):
        key = round(math.log(gap) * 2) / 2
        kk = min(keys, key=lambda k: abs(k - key))
        t, o = off[kk]
        return (o + 2) / (t + 40)

    return a, b, c, sd_of, off_p


def drifts(reads, uid):
    """(ln-rank drift per 24h, ln-playing drift per 24h) over the game's
    most recent segment of at least ~15h, shrunk for short segments."""
    b = None
    for idx in range(len(reads) - 1, -1, -1):
        if uid in reads[idx]['rank']:
            b = idx
            break
    if b is None:
        return 0.0, 0.0
    tb = reads[b]['t']
    a = None
    for idx in range(b - 1, -1, -1):
        if uid in reads[idx]['rank'] and (tb - reads[idx]['t']) >= 15 * 3600:
            a = idx
            break
    if a is None:
        return 0.0, 0.0
    dt_days = (reads[b]['t'] - reads[a]['t']) / 86400.0
    if dt_days <= 0:
        return 0.0, 0.0
    shrink = min(1.0, dt_days / 1.5)
    mu_rank = (math.log(reads[b]['rank'][uid]) - math.log(reads[a]['rank'][uid])) / dt_days
    mu_rank = max(-1.2, min(1.2, mu_rank)) * shrink
    pa = reads[a]['playing'].get(uid) or 0
    pb = reads[b]['playing'].get(uid) or 0
    mu_play = 0.0
    if pa > 0 and pb > 0:
        mu_play = max(-2.0, min(2.0, (math.log(pb) - math.log(pa)) / dt_days)) * shrink
    return mu_rank, mu_play


def coverage_targets(reads, places):
    """Per-place target sums: P(the holder of place k at a later read was on
    the origin chart), fit by gap bins and averaged over h = 1..72."""
    num = [[0.0] * places for _ in GAP_CENTRES]
    den = [[0.0] * places for _ in GAP_CENTRES]
    for i in range(len(reads)):
        si = set(reads[i]['rank'])
        for j in range(i + 1, len(reads)):
            gap = (reads[j]['t'] - reads[i]['t']) / 3600.0
            if gap <= 0:
                continue
            gb = min(range(len(GAP_CENTRES)),
                     key=lambda b: abs(math.log(GAP_CENTRES[b]) - math.log(gap)))
            for uid, rk in reads[j]['rank'].items():
                if rk <= places:
                    den[gb][rk - 1] += 1
                    if uid in si:
                        num[gb][rk - 1] += 1
    cov = [[(num[b][k] + 5) / (den[b][k] + 10) for k in range(places)]
           for b in range(len(GAP_CENTRES))]
    hs = list(range(1, 73))
    targets = []
    for k in range(places):
        s = 0.0
        for h in hs:
            for b, w in bin_mix(GAP_CENTRES, h):
                s += w * cov[b][k]
        targets.append(min(0.995, s / len(hs) * COV_MULT))
    return targets


def model_forecast(reads, places):
    fit = fit_diffusion(reads)
    if fit is None:
        return None
    a, b, c, sd_of, off_p = fit
    targets = coverage_targets(reads, places)
    last = reads[-1]
    raw = {}
    hs = list(range(1, 73))
    for uid in sorted(last['rank'], key=lambda u: last['rank'][u]):
        r0 = last['rank'][uid]
        mu_rank, mu_play = drifts(reads, uid)
        drift = KAPPA_RANK * mu_rank - KAPPA_PLAY * mu_play
        acc = [0.0] * places
        for h in hs:
            lg = math.log(h)
            mu = a + b * math.log(r0) + c * lg + drift * min(h, 48.0) / 24.0
            sd = sd_of(lg) * SD_MULT
            keep = 1.0 - off_p(h)
            for k in range(1, places + 1):
                lo = math.log(k - 0.5) if k > 1 else -9.0
                hi = math.log(k + 0.5)
                p = 0.5 * (math.erf((hi - mu) / (sd * math.sqrt(2.0)))
                           - math.erf((lo - mu) / (sd * math.sqrt(2.0))))
                acc[k - 1] += p * keep
        raw[uid] = [x / len(hs) for x in acc]
    return raw, targets


def fallback_forecast(reads, places):
    """House-baseline shape, used only if the fit is impossible."""
    last = reads[-1]
    chart = [uid for uid, _ in sorted(last['rank'].items(), key=lambda kv: kv[1])]
    probs = {}
    for place in range(1, places + 1):
        if place > len(chart):
            break
        probs[(chart[place - 1], place)] = 0.5
        if place == 1:
            if len(chart) > 1:
                probs[(chart[1], place)] = 0.3
            continue
        probs[(chart[place - 2], place)] = 0.15
        if place < len(chart):
            probs[(chart[place], place)] = 0.15
    return probs, chart[:places]


def assemble(raw, targets, places):
    out = {}
    for k in range(places):
        items = [(u, raw[u][k]) for u in raw if raw[u][k] > 0]
        s = sum(p for _, p in items)
        if s <= 0:
            continue
        t = min(targets[k], 0.995)
        items = [(u, p * t / s) for u, p in items]
        items.sort(key=lambda kv: -kv[1])
        for u, p in items[:100]:
            q = math.floor(p * 1e6) / 1e6
            if q >= MIN_P:
                out[(u, k + 1)] = q
    scored = []
    for uid, probs in raw.items():
        p10 = sum(probs)
        if p10 > 0.01:
            exp_place = sum((k + 1) * probs[k] for k in range(places)) / p10
            scored.append((-p10, exp_place, uid))
    scored.sort()
    return out, [u for _, _, u in scored[:places]]


def main():
    input_dir = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    output_dir = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    places = PLACES_FALLBACK
    out_name = 'forecast.json'
    try:
        with open(os.path.join(input_dir, 'task.json'), encoding='utf-8') as fh:
            task = json.load(fh)
        places = int(task.get('places', places))
        out_name = task.get('output', out_name)
    except (OSError, ValueError):
        pass
    reads = load_reads(input_dir)
    built = model_forecast(reads, places) if len(reads) >= 3 else None
    if built is None:
        probs, top10 = fallback_forecast(reads, places)
    else:
        raw, targets = built
        probs, top10 = assemble(raw, targets, places)
        if len(top10) < places:  # pad from the last chart, in order
            have = set(top10)
            for uid, _ in sorted(reads[-1]['rank'].items(), key=lambda kv: kv[1]):
                if uid not in have:
                    top10.append(uid)
                if len(top10) == places:
                    break
    rows = [{'universe_id': uid, 'place': k, 'p': p}
            for (uid, k), p in sorted(probs.items(), key=lambda kv: (kv[0][1], -kv[1], kv[0][0]))]
    payload = {'probabilities': rows, 'top10': top10}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, out_name), 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(payload))
        fh.write('\n')


if __name__ == '__main__':
    main()
