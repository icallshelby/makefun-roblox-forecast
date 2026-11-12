import json
import math
import os
from datetime import datetime

GAP_CENTRES = [3, 5, 8, 12, 18, 27, 40, 60, 90, 135]
RANK_BANDS = [(1, 5), (6, 15), (16, 40), (41, 10 ** 9)]

def parse_time(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()

def load_reads(input_dir):
    path = os.path.join(input_dir, 'history.ndjson')
    if not os.path.exists(path):
        return []
    reads = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rank, playing, created = {}, {}, {}
        for g in r.get('up_and_coming') or []:
            uid = g['universe_id']
            rank[uid] = g['rank']
            playing[uid] = g.get('playing') or 0
        dets = r.get('details') or {}
        for uid_str, d in dets.items():
            if not d:
                continue
            cr = d.get('created')
            if cr:
                try:
                    created[int(uid_str)] = parse_time(cr)
                except Exception:
                    pass
        if rank:
            reads.append({
                't': parse_time(r['at']),
                'rank': rank,
                'playing': playing,
                'created': created
            })
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

def _band_of(r):
    for bi, (lo, hi) in enumerate(RANK_BANDS):
        if lo <= r <= hi:
            return bi
    return len(RANK_BANDS) - 1

def fit_diffusion(reads, cfg, gap_lo=None, gap_hi=None):
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
            if gap_lo is not None and gap < gap_lo:
                continue
            if gap_hi is not None and gap > gap_hi:
                continue
            if cfg.get('fit_max_gap') is not None and gap > cfg['fit_max_gap']:
                continue
            lg = math.log(gap)
            key = round(lg * 2) / 2
            w = 1.0
            tau = cfg.get('fit_recency_tau')
            if tau:
                w = math.exp(-(t_end - rj['t']) / 86400.0 / tau)
            t, o = off.get(key, (0.0, 0.0))
            for uid, r0 in ri['rank'].items():
                r1 = rj['rank'].get(uid)
                t += 1
                bk = (key, _band_of(r0))
                tb, ob = off_rb.get(bk, (0.0, 0.0))
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
        return sum(a * b for a, b in zip(v, ws)) / sw

    mx, my, mg = wmean(xs), wmean(ys), wmean(gs)

    def wvar(a_, ma):
        return sum(w * (x - ma) ** 2 for x, w in zip(a_, ws)) / sw

    def wcov(a_, b_, ma, mb):
        return sum(w * (x - ma) * (y - mb) for x, y, w in zip(a_, b_, ws)) / sw

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

    def off_p(gap, r0=None):
        if not keys:
            return 0.1
        key = round(math.log(gap) * 2) / 2
        kk = min(keys, key=lambda k: abs(k - key))
        if cfg.get('off_by_rank') and r0 is not None:
            tb, ob = off_rb.get((kk, _band_of(r0)), (0.0, 0.0))
            t, o = off[kk]
            pooled = (o + 2) / (t + 40)
            if tb >= 30:
                return (ob + 30 * pooled) / (tb + 30)
            return pooled
        t, o = off[kk]
        return (o + 2) / (t + 40)

    return a, b, c, sd_of, off_p

def drifts(reads, uid, cfg):
    min_h = cfg.get('drift_min_h', 15.0)
    shrink_days = cfg.get('drift_shrink_days', 1.5)
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
        if uid in reads[idx]['rank'] and (tb - reads[idx]['t']) >= min_h * 3600:
            a = idx
            break
    if a is None:
        return 0.0, 0.0
    dt_days = (reads[b]['t'] - reads[a]['t']) / 86400.0
    if dt_days <= 0:
        return 0.0, 0.0
    shrink = min(1.0, dt_days / shrink_days)
    mu_rank = (math.log(reads[b]['rank'][uid]) - math.log(reads[a]['rank'][uid])) / dt_days
    mu_rank = max(-1.2, min(1.2, mu_rank)) * shrink
    pa = reads[a]['playing'].get(uid) or 0
    pb = reads[b]['playing'].get(uid) or 0
    mu_play = 0.0
    if pa > 0 and pb > 0:
        mu_play = max(-2.0, min(2.0, (math.log(pb) - math.log(pa)) / dt_days)) * shrink
    return mu_rank, mu_play

def coverage_targets(reads, cfg, places=10):
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
    split = cfg.get('horizon_split', 12.0)
    for k in range(places):
        s = 0.0
        for h in hs:
            for b, w in bin_mix(GAP_CENTRES, h):
                s += w * cov[b][k]
        targets.append(min(0.995, s / len(hs) * cfg.get('cov_mult', 1.00)))
    return targets

def raw_forecast(reads, cfg, places=10):
    split = cfg.get('horizon_split', 12.0)
    if cfg.get('horizon_mix'):
        fit_s = fit_diffusion(reads, cfg, gap_hi=split)
        fit_l = fit_diffusion(reads, cfg, gap_lo=split)
        if fit_s is None or fit_l is None:
            fit = fit_diffusion(reads, cfg)
            if fit is None:
                return None
            fits = (fit, fit)
        else:
            fits = (fit_s, fit_l)
    else:
        fit = fit_diffusion(reads, cfg)
        if fit is None:
            return None
        fits = (fit, fit)

    last = reads[-1]
    first_seen = {}
    for rd in reads:
        for uid in rd['rank']:
            first_seen.setdefault(uid, rd['t'])
    age_days = {uid: (last['t'] - t0) / 86400.0 for uid, t0 in first_seen.items()}

    # creation times for exit check
    created_map = {}
    for rd in reads:
        for uid, ctime in rd.get('created', {}).items():
            created_map[uid] = ctime

    cand = {uid: r0 for uid, r0 in last['rank'].items()}

    lp = {u: math.log(p) for u, p in last['playing'].items() if p and p > 0}
    lp_mu = sum(lp.values()) / len(lp) if lp else 0.0
    lp_sd = math.sqrt(sum((v - lp_mu) ** 2 for v in lp.values()) / len(lp)) if lp else 1.0

    raw = {}
    hs = list(range(1, 73))
    kappa_rank = cfg.get('kappa_rank', 0.7)
    kappa_play = cfg.get('kappa_play', 0.2)
    sd_mult = cfg.get('sd_mult', 1.08)
    drift_cap_h = cfg.get('drift_cap_h', 48.0)
    tenure_w = cfg.get('tenure_w', 0.03)
    tenure_form = cfg.get('tenure_form', 'drift')
    sat_w = cfg.get('sat_w', 0.0)
    sat_z = cfg.get('sat_z', 0.5)

    for uid in sorted(cand, key=lambda u: cand[u]):
        r0 = cand[uid]
        mu_rank, mu_play = drifts(reads, uid, cfg)
        drift = kappa_rank * mu_rank - kappa_play * mu_play
        shift = 0.0
        if sat_w and uid in lp:
            z = (lp[uid] - lp_mu) / max(lp_sd, 1e-9)
            shift += sat_w * max(0.0, z - sat_z)
        ten = tenure_w * age_days.get(uid, 0.0)
        acc = [0.0] * places
        ctime = created_map.get(uid)
        for h in hs:
            # Exit check: 28 days after creation
            if ctime is not None:
                if (last['t'] + h * 3600) > (ctime + 28 * 86400):
                    continue
            fit_obj = fits[0] if h <= split else fits[1]
            a, b, c, sd_of, off_p = fit_obj
            lg = math.log(h)
            h_eff = min(h, drift_cap_h) / 24.0
            mu = a + b * math.log(r0) + c * lg + drift * h_eff + shift
            if tenure_w:
                if tenure_form == 'drift':
                    mu += ten * h_eff
                else:
                    mu += ten
            sd = sd_of(lg) * sd_mult
            keep = 1.0 - off_p(h, r0)
            for k in range(1, places + 1):
                lo = math.log(k - 0.5) if k > 1 else -9.0
                hi = math.log(k + 0.5)
                p = 0.5 * (math.erf((hi - mu) / (sd * math.sqrt(2.0)))
                           - math.erf((lo - mu) / (sd * math.sqrt(2.0))))
                acc[k - 1] += p * keep
        raw[uid] = [x / len(hs) for x in acc]
    return raw

def assemble(raw, targets, cfg, places=10):
    out = {}
    min_p = cfg.get('min_p', 0.001)
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
            if q >= min_p:
                out[(u, k + 1)] = q
    return out

def fallback_forecast(reads, places=10):
    last = reads[-1] if reads else None
    out = {}
    if not last:
        return out
    holder = {rk: uid for uid, rk in last['rank'].items()}
    for k in range(1, places + 1):
        g = holder.get(k)
        if g is not None:
            out[(g, k)] = 0.5
        if k > 1:
            g_prev = holder.get(k - 1)
            if g_prev is not None:
                out[(g_prev, k)] = 0.15
        if k < places:
            g_next = holder.get(k + 1)
            if g_next is not None:
                out[(g_next, k)] = 0.3
    # Normalize per place
    final_out = {}
    for k in range(1, places + 1):
        place_items = [((u, pk), p) for (u, pk), p in out.items() if pk == k]
        s = sum(p for _, p in place_items)
        if s > 0:
            for (u, pk), p in place_items:
                q = math.floor((p / s) * 1e6) / 1e6
                if q >= 0.001:
                    final_out[(u, pk)] = q
    return final_out

def run_forecast():
    input_dir = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    output_dir = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    reads = load_reads(input_dir)

    if len(reads) < 3:
        forecast_probs = fallback_forecast(reads)
    else:
        # v5 ensemble of v1 and v2 members (0.5 / 0.5)
        c1 = {
            'drift_min_h': 15.0, 'drift_cap_h': 48.0,
            'off_by_rank': False, 'fit_recency_tau': None,
            'horizon_mix': True, 'horizon_split': 12.0,
            'tenure_w': 0.03, 'tenure_form': 'drift',
            'sat_w': 0.0, 'sat_z': 0.5,
            'sd_mult': 1.08, 'cov_mult': 1.00, 'min_p': 0.001
        }
        c2 = {
            'drift_min_h': 10.0, 'drift_cap_h': 72.0,
            'off_by_rank': True, 'fit_recency_tau': 3.0,
            'horizon_mix': True, 'horizon_split': 12.0,
            'tenure_w': 0.03, 'tenure_form': 'drift',
            'sat_w': 0.0, 'sat_z': 0.5,
            'sd_mult': 1.08, 'cov_mult': 1.00, 'min_p': 0.001
        }
        raw1 = raw_forecast(reads, c1)
        raw2 = raw_forecast(reads, c2)
        if raw1 is None and raw2 is None:
            forecast_probs = fallback_forecast(reads)
        else:
            if raw1 is None:
                raw1 = raw2
            if raw2 is None:
                raw2 = raw1
            uids = set(raw1) | set(raw2)
            raw = {}
            for u in uids:
                r1 = raw1.get(u, [0.0] * 10)
                r2 = raw2.get(u, [0.0] * 10)
                raw[u] = [0.5 * r1[k] + 0.5 * r2[k] for k in range(10)]
            t1 = coverage_targets(reads, c1)
            t2 = coverage_targets(reads, c2)
            targets = [0.5 * t1[k] + 0.5 * t2[k] for k in range(10)]
            forecast_probs = assemble(raw, targets, c1)
            if not forecast_probs:
                forecast_probs = fallback_forecast(reads)

    # Build output
    probabilities = []
    for (uid, place), p in sorted(forecast_probs.items(), key=lambda x: (x[0][1], -x[1])):
        probabilities.append({
            "universe_id": int(uid),
            "place": int(place),
            "p": float(p)
        })

    # top10: games ranked by total top-10 probability then expected place
    game_stats = {}
    for (uid, place), p in forecast_probs.items():
        if 1 <= place <= 10:
            st = game_stats.setdefault(uid, {'tot_p': 0.0, 'weighted_p': 0.0})
            st['tot_p'] += p
            st['weighted_p'] += place * p

    sorted_games = []
    for uid, st in game_stats.items():
        tot_p = st['tot_p']
        exp_place = st['weighted_p'] / tot_p if tot_p > 0 else 999.0
        sorted_games.append((uid, tot_p, exp_place))

    sorted_games.sort(key=lambda x: (-x[1], x[2]))
    top10 = [int(uid) for uid, _, _ in sorted_games[:10]]

    out_data = {
        "probabilities": probabilities,
        "top10": top10
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'forecast.json')
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)

if __name__ == '__main__':
    run_forecast()
