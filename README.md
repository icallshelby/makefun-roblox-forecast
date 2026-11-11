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
  with `ln(h)` and a per-gap probability of leaving the chart entirely.
  All parameters are fit by least squares over every (read, later read)
  pair in the supplied history — nothing is hard-coded to one dataset.
* Each game's recent ln-rank drift (weight 0.6) and ln(playing) growth
  (weight 0.2, opposite sign) shift its mean, extrapolated at most 48h.
* The static forecast averages the per-place distributions over
  h = 1..72, then scales each place to an empirical coverage target: the
  measured chance that the holder of that place was already on the origin
  chart (x0.9). The unnamed remainder is left for the scorer to share
  over unnamed games, which covers genuinely new chart entries.

The program is deterministic: two runs on the same history write the
same bytes. See `BUILD-REPORT.md` for the local evaluation behind these
choices.
