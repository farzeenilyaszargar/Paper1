If a small transformer learns a reasoning procedure in one representation, can it apply that same reasoning procedure when the underlying problem is expressed differently?

# Can Small Transformers Generalize Reasoning Across Representations?

This repository contains a reproducible reasoning dataset, paired symbolic,
function-syntax and template-English renderers, and a from-scratch transformer
experiment pipeline. Software smoke runs are complete; the main experiment is
still in progress. See `notes.md` and `paper_outline.md` for the research record.

## Quick start

Requires Python 3.11 or newer. Run these commands from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m synthetic_dataset_generator generate --config synthetic_dataset_generator/configs/smoke.json
```

The smoke preset writes and fully audits 540 examples in `datasets/smoke/`.
The full preset writes 180,000 examples, including 100,000 training examples:

```sh
.venv/bin/python -m synthetic_dataset_generator generate
.venv/bin/python -m synthetic_dataset_generator audit datasets/default
```

Customize counts, seed, and output without editing code:

```sh
.venv/bin/python -m synthetic_dataset_generator generate --train-count 1000000 --evaluation-count 10000 --seed 123 --output datasets/million-seed123
```

Dataset size and model parameter count are independent. Use the same frozen
corpus and evaluation splits for your 10K, 100K, and 500K parameter models so
capacity comparisons do not also change the data. `--train-count` controls only
training examples; `--evaluation-count` applies separately to validation and
each of the seven test files. Total records are therefore
`train_count + 8 * evaluation_count`. The command above creates 1,080,000 records.
Choose corpus size based on available storage and training compute, not the
model's parameter count alone.

Output directories must not already exist. Choose a new directory for another
run. An interrupted run is incomplete unless it has a manifest and a passing
full audit. Generation never overwrites an existing corpus or relaxes constraints
to meet a count. The installed `reasoning-data` command provides the same interface.
Classical checks use up to four worker processes by default. Set `--workers 1`
for serial execution; changing worker count does not change corpus bytes.

## Task and semantics

The solver-checked Figure 1 toy shows the same two-hop entailment in all three
renderings: `graphs/task_illustration.png` (SVG and exact input JSON alongside).
Regenerate it with `.venv/bin/python -m models.task_illustration`. It is a
pedagogical example, not a sampled benchmark record or model result; unlike
the benchmark, this minimal illustration omits balancing scaffolds/distractors.

Inputs contain symbolic facts, implications, and a literal query:

```text
FACTS: p017; ~p042. RULES: (p017 & ~p042) -> p083; p083 -> p091. QUERY: p091.
```

The answer is `entailed`, `contradicted`, or `unknown`. `contradicted` means the
query's negation follows, not merely that the query cannot be proved. An unknown
query and its negation are both consistent with the knowledge base. Inconsistent
knowledge bases are rejected. `~` is classical negation, not negation-as-failure.

This is a **controlled propositional fragment, not unrestricted propositional
logic**. Rule antecedents use positive/negated atoms with AND/OR; conclusions and
queries are literals. The internal format supports bounded disjunctive normal
form (up to four clauses, three literals per clause); the current sampler uses
unary, binary AND, and binary OR rules. It does not sample arbitrary formula
nesting, biconditionals, or negated compound expressions.

Facts have depth zero. Applying a rule costs one hop; an AND antecedent uses the
maximum of its premises' depths, while alternative rules/OR clauses use the
minimum successful depth. Thus depth measures parallel inference rounds, not
total proof size or shortest proof under every classical inference system.
Forward inference computes minimum depth, and a separate Z3 classical solver
verifies every label during both generation and audit.

For contradicted queries, depth refers to the proof of the negated query. Unknown
queries have `proof_depth: null`; their `matched_depth` identifies the depth of
the scaffold before core starting facts were withheld. They must not be reported
as having a 2-, 3-, 4-, or 5-hop proof.

## Independent evaluation splits

| Split file | Default count | Condition |
| --- | ---: | --- |
| `train.jsonl` | 100,000 | Familiar symbols/trees; 2-hop decisive proofs |
| `validation.jsonl` | 10,000 | New instances under training conditions |
| `test_iid.jsonl` | 10,000 | Independent within-distribution baseline |
| `test_combinations.jsonl` | 10,000 | Familiar names, entirely reserved directed symbol pairs |
| `test_structures.jsonl` | 10,000 | Held-out conjunctive diamond reconvergence; 2 hops |
| `test_symbols.jsonl` | 10,000 | Entirely held-out symbol names; 2 hops |
| `test_depth_3.jsonl` | 10,000 | Familiar symbols/families; 3 hops |
| `test_depth_4.jsonl` | 10,000 | Familiar symbols/families; 4 hops |
| `test_depth_5.jsonl` | 10,000 | Familiar symbols/families; 5 hops |

In every split, the three label counts differ by at most one example. Familiar names are
`p000`–`p511`; held-out names are `p512`–`p767`. A fixed **character vocabulary**
encodes all names. Every input character must occur in training. Do not replace
this encoding with a tokenizer that assigns each complete name its own learned
embedding: that would introduce untrained embeddings into the held-out-name test.
Held-out names differ in spelling within a shared alphabet; this does not test
unseen character tokens, and their numerical ranges are distinguishable.

The symbol-combination test reserves 20% of directed `(antecedent atom,
conclusion atom)` pairs using a stable seeded hash. Every dependency in that test
uses the reserved partition; no dependency in the other splits does. All its
individual names must occur in training. This is a controlled pair-novelty shift,
not just a previously unseen entire prompt. Negation polarity is ignored when
partitioning pairs.

Familiar graphs are directed forests with branching/merging trees and separate
distractor components. Held-out graphs contain a root splitting into two paths
that reconverge in an AND rule. Audits detect reconverging paths throughout the
graph and confirm the conjunctive diamond. The length and held-out-name tests
use familiar graph families and nonreserved pairs. Shapes may repeat across
ordinary splits, intentionally: only the structure test holds out a graph family.

Exact canonical knowledge bases cannot repeat anywhere in the corpus, even
with a different query, statement ordering, or redundant antecedent ordering.
Canonicalization also removes duplicate/absorbed DNF clauses. This is not a
claim of deduplication under every logical equivalence or symbol renaming;
renamed familiar shapes are necessary to the symbol experiments.

## Storage, reproducibility, and model inputs

Each JSONL record contains `id`, `split`, `input`, `target`, and `metadata`.
Metadata includes the structured facts/rules/query, symbols, directed edges,
canonical knowledge-base `group_id`, graph family, depths, decisive literal,
and derivation indices referring to that record's rule order.

**Train only on `input` and `target`.** Metadata contains answers and proofs and
must never enter the model prompt. Encode the input with `vocabulary.json`,
using zero-based positions in `characters` as IDs and appending the three special
tokens in their listed order. Class targets use the order in `targets`.
Character length is token length before adding optional BOS/EOS. A model training
pipeline must size its context from the recorded length histogram and reject
truncation that could remove premises or queries.

`synthetic_dataset_generator.encoding` supplies `encode`, `decode`, special-token
IDs, and target IDs implementing this contract. `encode` adds BOS/EOS by default
and rejects characters outside the shared vocabulary instead of hiding them as UNK.

Each run writes `config.json`, `vocabulary.json`, `manifest.json`, nine JSONL
files, and (after successful verification) `audit.json`. The manifest records
configuration, seed, generator/solver versions, counts, attempts, and SHA-256
checksums. Independent per-split random streams and ordered serialization make
identical configurations byte-reproducible with the same implementation/runtime.
Tests verify two complete runs byte for byte. Z3 and the build backend are pinned;
record the Python version used for a published experiment too.

Audits verify all labels with Z3, minimum forward depths, stored derivations,
rendered inputs, cross-split duplicates, graph constraints, symbol/pair holdouts,
counts, vocabulary coverage, and checksums. A failed audit exits nonzero.
Reports include per-label means for symbols, rules, facts, OR rules, input length,
and query negation, plus graph-family counts and input length histograms.

Both polarities of the query appear as rule conclusions, with separate candidate
proof paths. Starting facts activate the path supporting the query, its negation,
or neither. Inactive starting facts are replaced with facts in backup components.
All labels receive those components, preserving structural counts for a fixed
random scaffold. Presentation order and query polarity are randomized. These measures
reduce obvious shortcuts but do not prove the absence of all statistical cues;
inspect the audit report and compare simple baselines before drawing conclusions.

Generated corpora are ignored by Git because they can be large. Code, presets,
tests, and small illustrative records under `datasets/example/` are committed.
The examples are inspection samples, not an independently audited training corpus.

The default seed-42 corpus has been generated and fully audited: 180,000 records,
including 100,000 training records. `datasets/validation_summary.json` preserves
the corpus fingerprint, code commit, per-split counts, label statistics, and
maximum input lengths. All 16 unit/integration tests passed, including identical
output from serial and parallel verification. These are data validation results,
not model performance measurements.

## Research workflow

First verify within-representation 2-hop learning with a very small transformer.
Then evaluate each test file independently, reporting overall and per-label
accuracy and the training seed. This generator prepares data for 10K, 100K, and
500K parameter models; it does not assume any model will solve the task. Additional
representations and cross-representation transfer come after the symbolic baseline.

## Transformer experiments

Install the pinned experiment dependencies:

```sh
.venv/bin/python -m pip install -e '.[experiments]'
.venv/bin/python -m unittest discover -s tests -v
```

The experiment environment is Python 3.14.6 on macOS arm64. The observed full
dependency set is pinned in `models/requirements-experiments.lock.txt`.

The updated primary scaling conditions are:

| Preset | Actual trainable parameters |
| --- | ---: |
| `100k` | 99,075 |
| `500k` | 457,251 |
| `1m` | 1,004,547 |
| `5m` | 4,961,923 |
| `10m` | 9,896,643 |

These are causal transformers with a three-class answer head, sinusoidal
positions, and shared character IDs, not pretrained language models. Their
fixed 4096-token positional table is a buffer, not trainable parameters.

Example full-corpus run (choose a new output directory):

```sh
.venv/bin/python -m models.train --size 500k --representation symbolic --epochs 30 --seed 1 --batch-size 32 --output models/runs/full_500k_symbolic_seed1
.venv/bin/python -m models.plots
.venv/bin/python -m models.baselines
.venv/bin/python -m models.paired_statistics
```

Paired statistics require retained local per-instance prediction files. The
export verifies matching IDs, labels and saved confusion matrices, and reports
source-versus-target accuracy differences with exact McNemar tests and Holm
correction within each run's 14 comparisons. Missing legacy predictions are
explicitly unavailable, not reconstructed from aggregate accuracy. These are
exploratory checkpoint-conditional comparisons, not final across-seed inference.

Run `.venv/bin/python -m models.final_analysis` for the fixed 45-condition,
30-epoch primary matrix. Until every condition is present, its output
`graphs/final_baseline_analysis.json` contains only incomplete coverage—not
global statistics or an inferred reasoning onset. Once complete, it verifies
all retained epoch budgets, actual test IDs/golds, confusion-based metrics and
prediction hashes, and applies Holm across the full 630 source-versus-target
paired contrasts. Pilots and warmups cannot enter this family. Behavioral
criteria require source-validation plus IID mastery and success across all three
seeds; the smallest tested passing size is not a universal reasoning threshold.
These tests condition on checkpoints/test records, not uncertainty over seeds.
The analysis verifies artifacts; it does not rerun training or model inference.
New suite invocations refresh this report after each completed condition;
already-running supervisors are not hot-patched and can use the manual command.

For learning-rate/optimization calibration, add `--validation-only` to a new
training run. It saves the selected source-validation metrics without loading
test examples for evaluation, and marks its status `validation_only_completed`
so it cannot be counted as a completed transfer experiment. Dataset checksums
are still verified. Resume rejects switching this mode within a run.

Evaluate a completed raw-character checkpoint under solver-checked interventions:

```sh
.venv/bin/python -m models.robustness models/runs/baseline_full_100k_symbolic_seed1
.venv/bin/python -m models.audit_robustness models/runs/baseline_full_100k_symbolic_seed1
.venv/bin/python -m models.plot_robustness models/runs/baseline_full_100k_symbolic_seed1/robustness.json
```

By default this checks all 10K IID cases on CPU, separately from accelerator
training. It probes bijective familiar-name renaming, presentation order,
disconnected distractors, query negation, and supporting-fact removal. Labels
are recomputed with the classical solver; no proof metadata enters a prompt.
Fact removal may need multiple facts for OR branches and is not applicable to
originally unknown queries. Report joint correctness and label-change response,
not only prediction invariance: a constant classifier can be perfectly invariant.
This distractor probe does not replace a length-matched depth experiment.
Existing robustness reports are never overwritten; `--limit` explicitly marks
an optional subset diagnostic rather than a full evaluation.

### Character-exposure diagnostic

```sh
.venv/bin/python -m models.character_exposure --plot
```

This scans all corpus renderings and counts target-format character occurrences
whose IDs were absent from source-format training. It exports counts and a
3-by-3 IID exposure matrix, **not neural transfer accuracy**. BOS/EOS/PAD are
excluded from the content-token denominator. It uses the full training corpus,
not pilot subsets. Zero unseen IDs does not establish learned syntax, and
nonzero exposure mismatch does not by itself explain an accuracy failure.
Training-only syntax-exposure and compute-matched controls remain separate work.

The separate warmup implementation is available for those controls:

```sh
.venv/bin/python -m models.syntax_warmup --size 100k --source symbolic --treatment source_only --seed 1 --device cpu --output models/warmups/100k_source_only_seed1
.venv/bin/python -m models.syntax_warmup --size 100k --source symbolic --treatment all_formats --seed 1 --device cpu --output models/warmups/100k_all_formats_seed1
.venv/bin/python -m models.plot_warmup models/warmups/100k_source_only_seed1 models/warmups/100k_all_formats_seed1
```

Both CPU warmup arms have completed for seeds 1, 2 and 3. Their verified exposure
figures are `graphs/warmup_100k_symbolic_seed{1,2,3}.png`; downstream matched
supervised comparisons are pending. These commands use protected output directories and will not
overwrite completed runs. Do not start another accelerator job beside the queue.
Both arms predict exactly 2,048,000 next tokens in 1,000 fixed-size updates by
default. A tied LM projection adds no parameters and leaves the classifier head
untouched. LM windows can cut/cross problem boundaries; answer-only inference
still rejects truncation. Only training text is used, with no labels or proofs.

For subsequent source-only supervision, pass `--initialization` pointing to the
chosen warmup's `initialization.pt` to `models.train`. It checks corpus, size,
seed, source, completed budget and file provenance, then resets optimizer moments.
Use matched supervised settings for both arms and a no-warmup reference; choose
settings from validation, not tests. Warmup results have separate scopes and
cannot satisfy the primary baseline queue. `--validation-only` still avoids test
evaluation. Metrics distinguish IDs absent from source supervision from those
absent from all observed training inputs, including warmup.

Warmup checkpoints support `--resume` with identical configuration/code; an
interrupted stream is deterministically replayed to its saved position. Checkpoint
files remain local. Matching token/update budgets does not match the number of
latent problems or isolate character learning from richer unlabeled syntax.
The planned comparison is recorded in `models/syntax_exposure_protocol.json`.

Fresh trainer/warmup invocations check available storage before creating output
directories, including an OS reserve and estimated checkpoint/prediction space.
Atomic checkpoint writes also check capacity before touching a new temporary
file; low space raises an error while preserving the previous checkpoint.
These checks cannot guarantee space stays available, and already-running
processes are not hot-patched by source edits.

### Depth versus input length

Build and independently audit a separate held-out control corpus:

```sh
.venv/bin/python -m models.length_control
.venv/bin/python -m models.audit_length_control
.venv/bin/python -m models.evaluate_length_control models/runs/baseline_full_100k_symbolic_seed1
.venv/bin/python -m models.plot_length_control models/runs/baseline_full_100k_symbolic_seed1/length_control_results.json
```

The full control contains 90K pairs (180K logical inputs): 10K pairs for each
of depths 3/4/5 and each of the three representations. Every pair compares a
2-hop core with a deeper core of the same label, at exactly equal character-token
length, including BOS/EOS. Pairing uses held-out labels and original symbolic
lengths, never model predictions. Unknown examples retain null proof depths.
Both sides receive disconnected logical components using familiar names and
nonreserved dependency pairs. No filler whitespace or truncation is used.
Generation and the independent full-corpus audit are complete; the audit
summary is in `datasets/length_control_summary.json`. The full 100K seed-1
symbolic checkpoint evaluation is complete: all three 10K-pair comparisons
score 33.33% on both sides, with zero paired accuracy differences. This constant
checkpoint failed source learning, so these results do not establish reasoning
or depth generalization. See `graphs/baseline_full_100k_symbolic_seed1_length_control.md`.

This matches length and label, **not the entire underlying problem**: core KBs,
distractor counts, operators and graph details can still differ. Results are
diagnostic same-format comparisons, not a causal isolation of depth or raw
representation transfer. Core knowledge bases may recur across depth conditions;
do not treat those conditions as independent replications.

The auditor reparses every stored input, verifies its preserved core,
disconnected additions, graph/pair/name constraints, minimum proof depth, exact
token length, classical label and absence of training-KB duplicates. Evaluation
requires a passing audit and the model's matching parent-corpus fingerprint.
Bulk controls remain Git-ignored; a passing audit exports the compact tracked
`datasets/length_control_summary.json`. A generation without its final manifest
is incomplete and cannot be evaluated. `--limit` on generation creates an
explicitly labeled optional subset, not a replacement for the full control.

Use `--train-limit`/`--eval-limit` only for explicitly labeled pilots. Use
`--resume` with the same output and configuration to extend a completed epoch
checkpoint. Model changes with incompatible parameter counts are rejected.
The trainer checks dataset hashes, selects checkpoints using only source-format
validation loss, and then evaluates seven independent test splits in all three
representations. Inputs never include proof metadata. Oversized sequences fail
rather than being truncated. MPS is used when available; seeded accelerator
training is not guaranteed bit-identical.

JSON configs, epoch histories, measured results, and plots are committed;
checkpoints remain local and ignored. Graphs identify smoke/pilot scope, and
incomplete runs do not receive fabricated test results. Raw representation
transfer reports source-unseen character IDs because a shared vocabulary alone
does not remove syntax-exposure confounds. Full multi-seed scaling, controls,
and the reasoning-onset analysis remain required before a final conclusion.
The full-corpus exact and surface-feature baseline measurements are stored in
`models/baseline_results.json`. Predictions needed for paired statistical tests
are saved locally per run and ignored by Git, alongside checkpoints.

`models/parsing.py` parses all three declared surface grammars without reading
answer/proof metadata. The full 180K-record round-trip/normalization audit is in
`models/representation_audit.json`. `--encoding local_slots` is a separately
labeled external-parsing and binding-normalization control; its identical
cross-format inputs are not learned raw-format transfer.

The initial fixed-budget full-corpus matrix has 45 conditions: five capacities,
three source representations, and three seeds. Each uses 30 epochs, LR 0.001,
and effective batch size 32. Gradient accumulation keeps that batch constant
while reducing microbatch memory for larger models. To run/resume its queue:

```sh
.venv/bin/python -u -m models.suite --include-pilot
```

The supervisor holds an OS lock, rejects overlapping live trainers, and records
the active child in `models/suite_state.json`; logs are local in
`models/suite_logs/`. Before restarting after an interruption, verify that the
old supervisor and trainer are no longer live. Checkpoints are written atomically.
Input lengths are padded to 128-token buckets to bound accelerator compilation
cache shapes. `--eval-batch-size` controls evaluation separately from the training
microbatch (default: at most four). This preserves all input tokens and EOS
positions; padding is causally invisible to the answer head.
Completion of this baseline queue is not completion of calibration, controls,
or the research conclusion; the source-learning gate and all failures remain
part of the results.
