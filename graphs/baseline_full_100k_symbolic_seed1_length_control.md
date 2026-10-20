# Equal-length depth diagnostics

Run: `baseline_full_100k_symbolic_seed1`.

Diagnostic same-format comparisons with equal length and label but different core KBs. Added distractor, graph and fact/operator-count cues are not fully controlled. Not a causal isolation of depth or a transfer result. Exact paired p-values are checkpoint/test-conditional; Holm covers only this run's three depth comparisons, not other seeds/sizes/controls.

Each depth has its own paired 2-hop reference. Short cores are reused across conditions; these are not independent replications. Unknown examples use scaffold depth. No training-seed uncertainty is estimated.

| Long depth | Pairs | Paired 2-hop accuracy | Long accuracy | Long minus short | Exact McNemar p | Holm p (three depths within run) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 10000 | 0.3333 | 0.3333 | +0.0000 | 1 | 1 |
| 4 | 10000 | 0.3333 | 0.3333 | +0.0000 | 1 | 1 |
| 5 | 10000 | 0.3333 | 0.3333 | +0.0000 | 1 | 1 |

Source report SHA256: `3fe474b22b717e47c7173f3b970ad743cdd055e40b42ce85528ff5ad9e1a2a4d`.
