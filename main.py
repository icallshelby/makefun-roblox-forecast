#!/usr/bin/env python3
"""Entry for make.fun bounty roblox-up-and-coming-forecast-01 (v2).

Forecasts Roblox's Up-and-Coming top 10 for the 72 hours after the history
ends. Model (fit from /input/history.ndjson alone, standard library only):

  * Each game's future rank is modelled in log space: ln(rank at +h hours)
    is Gaussian with mean a + b*ln(rank now) + c*ln(h) plus a momentum
    shift, and a standard deviation that grows with ln(h). a, b, c and the
    spread curve are fit by least squares over every (read, later read)
    pair in the history, with pairs weighted toward the recent end of the
    history (exponential decay, tau 5 days): the chart's dynamics drift,
    and the hours nearest the forecast window matter most.
  * Off-chart rates are fit per gap and per origin-rank band (shrunk
    toward the pooled rate): a top-5 game and a rank-60 game do not leave
    the chart at the same rate, and the difference moves real mass.
  * Momentum: a game's recent ln-rank drift (weight 0.7) and its recent
    ln(playing) growth (weight 0.2, opposite sign) over its latest
    segment of at least ~10 hours, shrunk for short segments and
    extrapolated at most 72 hours.
  * Saturation: the chart is sorted by growth, not size, so a game's
    ln(playing) level, z-scored across the chart, shifts its mean rank
    worse by 0.06 per z -- the biggest games on the chart are the ones
    whose growth is already spent. (Measured on the practice pairs: the
    partial coefficient is small but positive everywhere, and the effect
    is large exactly in the freshest windows.)
  * Age-out: only games published within the last 28 days are on the
    chart at all, so a game whose creation date (from the reads'
    details, present in the house recorder's reads) puts it past 28
    days at horizon h contributes no mass at that horizon -- it cannot
    hold a place it is no longer eligible for. Where no read carries a
    creation date the adjustment is inactive.
  * The single static forecast averages the per-place distributions over
    h = 1..72 (the scored window is hourly), then each place is scaled to
    an empirical coverage target: the measured probability, by gap, that
    the holder of that place was already on the origin chart, averaged
    over the window and multiplied by 0.95. What is left unnamed is
    shared by the scorer over the games a read's chart holds that we did
    not name, which is where genuinely new games get their probability.

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
KAPPA_RANK = 0.7      # weight on recent ln-rank drift
KAPPA_PLAY = 0.2      # weight on recent ln(playing) growth (rank improves as it grows)
SD_MULT = 1.08        # spread calibration
COV_MULT = 0.95       # coverage calibration
DRIFT_MIN_H = 10.0    # minimum momentum segment length in hours (v1 used 15)
DRIFT_CAP_H = 72.0    # momentum extrapolation cap in hours (v1 used 48)
RECENCY_TAU_DAYS = 5.0  # pair-weight decay for the diffusion fit
LEVEL_W = 0.06        # saturation: mean-rank penalty per z of ln(playing)
GAP_CENTRES = [3, 5, 8, 12, 18, 27, 40, 60, 90, 135]
RANK_BANDS = [(1, 5), (6, 15), (16, 40), (41, 10 ** 9)]


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
            created = {}
            for key, det in (r.get('details') or {}).items():
                if det and det.get('created'):
                    try:
                        created[int(key)] = parse_time(det['created'])
                    except (ValueError, TypeError):
                        pass
            if rank:
                reads.append({'t': parse_time(r['at']), 'rank': rank,
                              'playing': playing, 'created': created})
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


def band_of(r):
    for bi, (lo, hi) in enumerate(RANK_BANDS):
        if lo <= r <= hi:
            return bi
    return len(RANK_BANDS) - 1


def fit_diffusion(reads):
    """Recency-weighted least squares for ln r' ~ a + b ln r + c ln gap,
    spread by gap, and off-chart rates per (gap, origin-rank band).
    Returns (a, b, c, sd_of, off_p) or None."""
    xs, ys, gs, ws = [], [], [], []
    off = {}
    off_rb = {}
    t_end = reads[-1]['t']
    for i in range(len(reads)):
        ri = reads[i]
        for j in range(i + 1, len(reads)):
            rj = reads[j]
            gap = (rj['t'] - ri['t']) / 3600.0
            if gap <= 0:
                continue
            lg = math.log(gap)
            key = round(lg * 2) / 2
            w = math.exp(-(t_end - rj['t']) / 86400.0 / RECENCY_TAU_DAYS)
            t, o = off.get(key, (0, 0))
            for uid, r0 in ri['rank'].items():
                r1 = rj['rank'].get(uid)
                t += 1
                bk = (key, band_of(r0))
                tb, ob = off_rb.get(bk, (0, 0))
                tb += 1
                if r1 is None:
                    o += 1
                    ob += 1
                else:
                    xs.append(math.log(r0))
                    ys.append(math.log(r1))
                    gs.append(lg)
                    ws.append(w)
                off_rb[bk] = (tb, ob)
            off[key] = (t, o)
    n = len(xs)
    if n < 50:
        return None
    sw = sum(ws)

    def wmean(v):
        return sum(x * w for x, w in zip(v, ws)) / sw

    def wcov(a_, b_, ma, mb):
        return sum(w * (x - ma) * (y - mb) for x, y, w in zip(a_, b_, ws)) / sw

    def wvar(a_, ma):
        return sum(w * (x - ma) ** 2 for x, w in zip(a_, ws)) / sw

    mx, my, mg = wmean(xs), wmean(ys), wmean(gs)
    b = wcov(xs, ys, mx, my) / wvar(xs, mx)
    resid = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
    mr = wmean(resid)
    c = wcov(gs, resid, mg, mr) / wvar(gs, mg)
    a = my - b * mx - c * mg
    sq = [(r - mr) ** 2 for r in resid]
    msq = wmean(sq)
    csq = wcov(gs, sq, mg, msq) / wvar(gs, mg)
    asq = msq - csq * mg

    def sd_of(lg):
        return math.sqrt(max(0.04, asq + csq * lg))

    keys = sorted(off)

    def off_p(gap, r0):
        key = round(math.log(gap) * 2) / 2
        kk = min(keys, key=lambda k: abs(k - key))
        t, o = off[kk]
        pooled = (o + 2) / (t + 40)
        tb, ob = off_rb.get((kk, band_of(r0)), (0, 0))
        if tb >= 30:
            return (ob + 30 * pooled) / (tb + 30)
        return pooled

    return a, b, c, sd_of, off_p


def drifts(reads, uid):
    """(ln-rank drift per 24h, ln-playing drift per 24h) over the game's
    most recent segment of at least ~10h, shrunk for short segments."""
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
        if uid in reads[idx]['rank'] and (tb - reads[idx]['t']) >= DRIFT_MIN_H * 3600:
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
    born = {}
    for rd in reads:
        for uid, ct in rd.get('created', {}).items():
            born[uid] = ct
    lp = {u: math.log(p) for u, p in last['playing'].items() if p and p > 0}
    lp_mu = sum(lp.values()) / len(lp) if lp else 0.0
    lp_sd = math.sqrt(sum((v - lp_mu) ** 2 for v in lp.values()) / len(lp)) if lp else 1.0
    raw = {}
    hs = list(range(1, 73))
    for uid in sorted(last['rank'], key=lambda u: last['rank'][u]):
        r0 = last['rank'][uid]
        mu_rank, mu_play = drifts(reads, uid)
        drift = KAPPA_RANK * mu_rank - KAPPA_PLAY * mu_play
        shift = 0.0
        if uid in lp:
            shift += LEVEL_W * (lp[uid] - lp_mu) / max(lp_sd, 1e-9)
        exit_h = None
        if uid in born:
            exit_h = (28.0 * 86400.0 - (last['t'] - born[uid])) / 3600.0
        acc = [0.0] * places
        for h in hs:
            if exit_h is not None and h > exit_h:
                continue
            lg = math.log(h)
            mu = a + b * math.log(r0) + c * lg + drift * min(h, DRIFT_CAP_H) / 24.0 + shift
            sd = sd_of(lg) * SD_MULT
            keep = 1.0 - off_p(h, r0)
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
