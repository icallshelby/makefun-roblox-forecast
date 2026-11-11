# Roblox Up-and-Coming Forecaster

Entry for the make.fun bounty `roblox-up-and-coming-forecast-01`
("Predict Roblox's Up-and-Coming top 10").

`main.py` (Python 3, standard library only, no network) reads
`history.ndjson` and `task.json` from `$MAKEFUN_INPUT_DIR` (else `/input`)
and writes one static forecast to `$MAKEFUN_OUTPUT_DIR` (else `/output`):
for each of the top 10 places, a probability for each game it names being
at that place, plus its guessed top 10 in order. The forecast is scored
against every hourly read of the 72 hours after the history ends.

## Model (v3: fixed-weight ensemble of v1 + v2)

`main.py` runs both earlier forecasters on the same history and blends
their per-place probability vectors elementwise with fixed weights —
0.6 on v1, 0.4 on v2 — and blends the per-place coverage targets with
the same weights before the shared assembly step.

* v1: future ranks are modelled in log space: `ln(rank at +h hours)` is
  Gaussian around `a + b*ln(rank now) + c*ln(h)`, spread growing with
  `ln(h)`, off-chart probability fit per gap. Plain least squares over
  every (read, later read) pair; momentum from the latest >= ~15h
  segment, extrapolated at most 48h.
* v2: the same diffusion with pairs weighted toward the recent end
  (exponential decay, tau 5 days), off-chart rates also split by
  origin-rank band, a >= ~10h momentum segment extrapolated at most
  72h, a saturation term (ln(playing) level z-scored across the chart,
  0.06 per z), and a 28-day age-out for games whose creation date a
  read's details carry.
* Both members average their per-place distributions over h = 1..72;
  the blended forecast is scaled per place to the blended empirical
  coverage target (x0.95). The unnamed remainder is left for the scorer
  to share over unnamed games, which covers genuinely new chart
  entries.

The program is deterministic: two runs on the same history write the
same bytes.
