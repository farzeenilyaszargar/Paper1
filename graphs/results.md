# Measured experiment results

Smoke/pilot results are not final paper evidence. Confidence intervals describe finite test sampling, not training-seed uncertainty.

| Run | Scope | Parameters | Train format | Test format | IID accuracy | Examples |
| --- | --- | ---: | --- | --- | ---: | ---: |
| baseline_full_100k_symbolic_seed1 | full corpus | 99075 | symbolic | symbolic | 0.3333 | 10000 |
| baseline_full_100k_symbolic_seed1 | full corpus | 99075 | symbolic | function | 0.3333 | 10000 |
| baseline_full_100k_symbolic_seed1 | full corpus | 99075 | symbolic | english | 0.3333 | 10000 |
| control_local_slots_100k_seed1 | binding-normalized pilot | 99075 | symbolic | symbolic | 0.3020 | 1000 |
| control_local_slots_100k_seed1 | binding-normalized pilot | 99075 | symbolic | function | 0.3020 | 1000 |
| control_local_slots_100k_seed1 | binding-normalized pilot | 99075 | symbolic | english | 0.3020 | 1000 |
| diagnostic_overfit_100k_seed1 | memorization diagnostic | 99075 | symbolic | symbolic | 0.3000 | 30 |
| diagnostic_overfit_100k_seed1 | memorization diagnostic | 99075 | symbolic | function | 0.3000 | 30 |
| diagnostic_overfit_100k_seed1 | memorization diagnostic | 99075 | symbolic | english | 0.3333 | 30 |
| pilot_500k_symbolic_seed1 | pilot subset | 457251 | symbolic | symbolic | 0.3220 | 1000 |
| pilot_500k_symbolic_seed1 | pilot subset | 457251 | symbolic | function | 0.3220 | 1000 |
| pilot_500k_symbolic_seed1 | pilot subset | 457251 | symbolic | english | 0.3220 | 1000 |
| smoke_100k_seed1 | smoke | 89763 | symbolic | symbolic | 0.4333 | 30 |
| smoke_100k_seed1 | smoke | 89763 | symbolic | function | 0.3667 | 30 |
| smoke_100k_seed1 | smoke | 89763 | symbolic | english | 0.5000 | 30 |
