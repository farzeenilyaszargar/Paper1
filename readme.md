If a small transformer learns a reasoning procedure in one representation, can it apply that same reasoning procedure when the underlying problem is expressed differently?

# Can Small Transformers Generalize Reasoning Across Representations?

This repository starts with a reproducible symbolic reasoning dataset. Transformer
training and the other representations are future stages; no model results are
claimed here.

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

Output directories must not already exist. Choose a new directory for another
run. An interrupted run is incomplete unless it has a manifest and a passing
full audit. Generation never overwrites an existing corpus or relaxes constraints
to meet a count. The installed `reasoning-data` command provides the same interface.
Classical checks use up to four worker processes by default. Set `--workers 1`
for serial execution; changing worker count does not change corpus bytes.

## Task and semantics

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

Every split balances the three labels within one example. Familiar names are
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

## Research workflow

First verify within-representation 2-hop learning with a very small transformer.
Then evaluate each test file independently, reporting overall and per-label
accuracy and the training seed. This generator prepares data for 10K, 100K, and
500K parameter models; it does not assume any model will solve the task. Additional
representations and cross-representation transfer come after the symbolic baseline.
