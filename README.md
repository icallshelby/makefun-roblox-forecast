# Roblox Up-and-Coming Forecaster

Entry for the make.fun bounty `roblox-up-and-coming-forecast-01`
("Predict Roblox's Up-and-Coming top 10").

`main.py` (Python 3, standard library only, no network) reads
`history.ndjson` and `task.json` from `$MAKEFUN_INPUT_DIR` (else `/input`)
and writes one static forecast to `$MAKEFUN_OUTPUT_DIR` (else `/output`):
for each of the top 10 places, a probability for each game it names being
at that place, plus its guessed top 10 in order. The forecast is scored
against every hourly read of the 72 hours after the history ends.

## Model

* Future ranks are modelled in log space: `ln(rank at +h hours)` is
  Gaussian around `a + b*ln(rank now) + c*ln(h)`, with a spread that grows
  with `ln(h)` and a probability of leaving the chart fit per gap and
  per origin-rank band. All parameters are fit by least squares over
  every (read, later read) pair in the supplied history, with pairs
  weighted toward the recent end (exponential decay, tau 5 days) —
  nothing is hard-coded to one dataset.
* Each game's recent ln-rank drift (weight 0.7) and ln(playing) growth
  (weight 0.2, opposite sign) over its latest segment of at least ~10h
  shift its mean, extrapolated at most 72h.
* Saturation: a game's ln(playing) level, z-scored across the chart,
  shifts its mean rank worse by 0.06 per z — the chart is sorted by
  growth, so the biggest games on it are the ones whose growth is spent.
* Games whose creation date (when a read's details carry one) puts them
  past the chart's 28-day publication limit contribute no mass at
  horizons beyond it.
* The static forecast averages the per-place distributions over
  h = 1..72, then scales each place to an empirical coverage target: the
  measured chance that the holder of that place was already on the origin
  chart (x0.95). The unnamed remainder is left for the scorer to share
  over unnamed games, which covers genuinely new chart entries.

Local evaluation on the practice history (exact scorer from the rules,
20 forecast origins fit without leakage): log score mean -38.030 and
-38.087 on the headline split (train on all but the last 72h), against
-38.046 / -39.232 for the previous version of this entry and -48.154 /
-50.884 for the house baseline on the same harness.

The program is deterministic: two runs on the same history write the
same bytes.
