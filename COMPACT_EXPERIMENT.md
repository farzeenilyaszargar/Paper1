# Active compact experiment — 23 October 2026

This user-approved redesign supersedes the 45-condition full-corpus queue.
Original datasets, checkpoints and results remain intact as exploratory history.
The active sizes are 10K, 50K, 100K and 500K (1M is excluded).

## Budget and task

`models/compact_protocol.json` and
`synthetic_dataset_generator/configs/compact.json` define the frozen design.
There are 5,000 train, 1,000 validation and seven 1,000-example test splits.
Train decisive queries require two parallel forward-rule steps. Unknown queries
retain the matched scaffold convention. AND/OR rules, signed literals, canonical
parent-KB exclusion, reserved symbol pairs, separate held-out names, diamond
structures and 3/4/5-hop tests remain. Three lossless renderings are unchanged.

The new task removes extra dangling branches and disconnected distractor chains,
uses 3–6 facts, and has 128 familiar plus 128 held-out symbol names. It still
contains alternative-polarity paths and balancing components. This changes task
difficulty and is **not** a matched comparison with the original full corpus.
The existing generator defaults retain their original sampling behavior.

Each run has at most 20 epochs: 100,000 training-example presentations versus
3,000,000 previously. This is 30 times fewer presentations, not a promised
wall-clock speedup. AdamW LR 0.003 is an exploratory starting setting informed
by earlier 100K subset calibration, not established optimal settings here.
The effective batch remains 32; microbatch 16, accumulation 2, raw characters.

## Stages and stopping rules

1. Run one 100K symbolic pilot, seed 1, validation only. Select its checkpoint
   by minimum validation loss. If it fails 95% validation accuracy or 90%
   recall in any class, stop expansion and diagnose. Do not silently relax gates.
2. Only after that pilot passes, screen four sizes on symbolic input, one seed
   each, with identical budgets. Evaluate all three target formats on all seven
   tests after validation selection. Independent symbolic IID mastery is required
   before interpreting transfer. Test scores never select a training checkpoint.
3. Select the smallest tested size meeting validation and source IID gates,
   then train two additional source-format runs (function and English). Report
   each source's own mastery before its transfer scores. These six core runs
   follow the separate pilot; the pilot is not counted as a screen replicate.
4. Confirm key findings with more seeds only after reviewing the screen. Until
   then, parameter onset is exploratory, not a robust/universal threshold.

Warmup controls, broad calibration and extensive robustness sweeps are deferred.
No capacity or transfer runs start merely because the pilot finishes: the gate
must pass, and stages are invoked explicitly. If the pilot fails, its failure
is evidence to investigate, not evidence that no transformer can reason.

## Commands

```sh
.venv/bin/python -m synthetic_dataset_generator generate --config synthetic_dataset_generator/configs/compact.json --output datasets/compact --workers 2
.venv/bin/python -m models.compact pilot --execute
# These refuse to expand without the prerequisite gates:
.venv/bin/python -m models.compact screen --execute
.venv/bin/python -m models.compact transfer --execute
```

Generation refuses to overwrite existing data. Without `--execute`, compact
commands print a gated plan only. Existing run folders are not overwritten or
silently resumed. Interrupted compact runs require explicit checkpoint review.
Legacy suite/auxiliary CLI entrypoints now require the deliberate
`--allow-legacy-large-sweep` override. Do not use it for the active experiment.

`graphs/final_baseline_analysis.json` remains the **legacy** full-matrix coverage
report; its 45-run completion requirement is not the compact completion target.
Existing plots are historical measurements, not compact results. Compact plots
and conclusions must be generated from the new run folders and dataset hashes.
