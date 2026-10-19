# Solver-checked robustness diagnostics

Run: `baseline_full_100k_symbolic_seed1`. Scope: full IID robustness diagnostic.

High prediction invariance without joint correctness can reflect a constant classifier, not reasoning. Test diagnostics are not used for hyperparameter selection. This distractor intervention is not a length-matched depth control.

Fact ablation applies only to originally decisive cases; those modified gold labels are all unknown. Denominators and label mixes differ. No universal chance line is appropriate.

| Intervention | Applicable | Accuracy | Original & changed correct | Prediction unchanged | Gold changed | Correctly followed gold change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 10000 | 0.3333 | 3333 | 10000 | 0 | 0 |
| Renamed | 10000 | 0.3333 | 3333 | 10000 | 0 | 0 |
| Reordered | 10000 | 0.3333 | 3333 | 10000 | 0 | 0 |
| Distractors | 10000 | 0.3333 | 3333 | 10000 | 0 | 0 |
| Negated query | 10000 | 0.3334 | 0 | 10000 | 6667 | 0 |
| Fact ablation | 6667 | 0.0000 | 0 | 6667 | 6667 | 0 |
