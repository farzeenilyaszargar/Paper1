# Table 1: audited corpus characteristics

Measured full corpus: 180,000 latent problems, each renderable in three formats.
These are corpus statistics, not model results. Lengths are mean character-token
counts including BOS and EOS, with no padding or truncation. English is a
controlled template grammar, not unrestricted natural language.

| Split | Problems | Scaffold depth | Symbol pool | Additional holdout | Symbolic tokens | Function tokens | English tokens |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| Train | 100,000 | 2 | p000–p511 | None | 287.2862 | 411.6127 | 410.5766 |
| Validation | 10,000 | 2 | p000–p511 | Distinct canonical KBs | 287.9381 | 412.6701 | 411.6460 |
| IID test | 10,000 | 2 | p000–p511 | Distinct canonical KBs | 287.0423 | 411.4103 | 410.3299 |
| Symbol combinations | 10,000 | 2 | p000–p511 | Reserved directed name pairs | 287.3868 | 411.7365 | 410.7137 |
| Graph structures | 10,000 | 2 | p000–p511 | Reconvergent diamond dependencies | 307.7839 | 441.6921 | 437.2698 |
| Held-out symbols | 10,000 | 2 | p512–p767 | All atom names absent from training | 287.3090 | 411.7465 | 410.6689 |
| Depth 3 | 10,000 | 3 | p000–p511 | Longer decisive proofs/scaffolds | 356.4248 | 520.6998 | 519.9181 |
| Depth 4 | 10,000 | 4 | p000–p511 | Longer decisive proofs/scaffolds | 431.9453 | 639.9722 | 639.4771 |
| Depth 5 | 10,000 | 5 | p000–p511 | Longer decisive proofs/scaffolds | 517.4239 | 775.5682 | 775.2063 |

Each 10,000-problem split has 3,334 entailed, 3,333 contradicted and 3,333
unknown cases; training has 33,334/33,333/33,333. For decisive cases, depth is
minimum parallel forward-rule depth. Unknown cases have no decisive proof:
their listed depth describes the matched scaffold, not a nonexistent proof.

The familiar graph family consists of directed forests; diamonds are held out
for the structure test. Reserved name pairs are excluded from training and
other familiar-name splits. Canonical KB deduplication crosses all splits and
ignores query/presentation changes; it does not hold out every alpha-equivalent
graph. New symbol names use the shared character encoding, not novel token IDs.
The three renderings remain paired by latent instance and split.

Length is not balanced across holdouts: deeper problems are also longer, and
diamonds differ in mean length from IID. Use the separately audited 90,000-pair
length control for diagnostic equal-length comparisons; its added logical
distractors do not isolate depth from every structural cue.

## Provenance and calculation

Source: `models/character_exposure.json`, whose full-corpus pass verified all
split file checksums/counts. For each split and representation, sum its
`content_character_counts`, divide by `records[split]`, and add two boundary
tokens. Round only the displayed result to four decimals. Counts and holdout
constraints are also checked by `datasets/validation_summary.json` and the
generator/auditor. This table does not replace those full audits.

- Parent manifest SHA256: `4d3a5fc9bf87a6ef3901285f08afae5c52a272fe7c7ef964779dee53dd7565e1`.
- Character-exposure report SHA256: `4cf2c3d4a2b3f7dcd22be90ec01a34c1d824604f356b66b6bd8e423ae08d9e2b`.
- Table prepared 20 October 2026 after matching current manifest and renderer
  checksums to that report. No new model evaluation was used to build it.
