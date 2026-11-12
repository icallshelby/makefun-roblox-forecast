# Roblox Up-and-Coming Forecaster

Entry for the make.fun bounty `roblox-up-and-coming-forecast-01`
("Predict Roblox's Up-and-Coming top 10").

`main.py` (Python 3, standard library only, no network) reads
`history.ndjson` and `task.json` from `$MAKEFUN_INPUT_DIR` (else `/input`)
and writes one static forecast to `$MAKEFUN_OUTPUT_DIR` (else `/output`):
for each of the top 10 places, a probability for each game it names being
at that place, plus its guessed top 10 in order. The forecast is scored
against every hourly read of the 72 hours after the history ends.

## Model (v5: v4 ensemble + tenure hazard + horizon mixture)

`main.py` runs both earlier forecasters on the same history and blends
their per-place probability vectors elementwise with fixed weights —
0.5 on v1, 0.5 on v2 — and blends the per-place coverage targets with
the same weights before the shared assembly step.

* v1: future ranks are modelled in log space: `ln(rank at +h hours)` is
  Gaussian around `a + b*ln(rank now) + c*ln(h)`, spread growing with
  `ln(h)`, off-chart probability fit per gap. Plain least squares over
  every (read, later read) pair; momentum from the latest >= ~15h
  segment, extrapolated at most 48h.
* v2: the same diffusion with pairs weighted toward the recent end
  (exponential decay, tau 3 days), off-chart rates also split by
  origin-rank band, a >= ~10h momentum segment extrapolated at most
  72h, a saturation HINGE term (ln(playing) z-scored across the chart,
  mean rank worsens 0.3 per z above z = 0.5 only), and a 28-day age-out
  for games whose creation date a read's details carry.
* v5 adds, to BOTH members: TENURE — a game's chart age (days since
  first seen in the history) shifts its drift by +0.04 ln-rank per day
  of age per day (young games keep climbing, old games fade; measured
  age 0-1d improves ln-rank -0.21 at +24h, age 4-7d worsens +0.09);
  and HORIZON MIXTURE — the diffusion is fit separately on short pairs
  (gap <= 12h: persistence 0.92, resid sd 0.37) and long pairs
  (gap > 12h: 0.64, 0.73), each forecast horizon using the matching
  fit. Short-window breakout momentum was measured and rejected: it
  costs mean at every weight tried.
* Both members average their per-place distributions over h = 1..72;
  the blended forecast is scaled per place to the blended empirical
  coverage target (x1.00, unshrunk). Rows below p = 0.001 are dropped
  at assembly: the scorer's fallback for an unnamed holder averages
  ~0.005, so naming a game below ~0.001 is worse than leaving it
  unnamed. The unnamed remainder is left for the scorer to share over
  unnamed games, which covers genuinely new chart entries.

Local harness (exact RULES scorer, 20 origins): v5 mean -36.727,
headline -38.068, worst -42.571, against v4's -37.606 / -38.586 /
-42.388 (v3: -37.929 / -38.674 / -42.198). On the later half of
origins v5 means -36.502 vs v4's -37.538. The program is deterministic:
two runs on the same history write the same bytes.
