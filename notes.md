# Research notes - 13 October 2026

## Question and current evidence

Can a transformer trained from scratch on 2-hop propositional inference apply
that procedure under a change of representation, symbols, graph family, or depth?
The corpus has 100K training instances and eight 10K evaluation/validation splits.
The logic fragment, minimum parallel-rule depth, and classical three-way labels
are defined in the README. No model has yet met the learning gate below.

The first 89,763-parameter smoke run used only 300 training examples, two epochs,
and 30 validation examples. Best-validation-loss selection produced 26.7%
validation accuracy. This is a software check, not evidence of learned reasoning
or a meaningful estimate of transfer. A 457,251-parameter pilot on 20K examples
is running; its results must be identified as pilot evidence, not full-corpus evidence.

## Closest papers: question, method, result, difference

### Lin et al., ICML 2026

[Paper](https://proceedings.mlr.press/v306/lin26k.html),
[latest arXiv version](https://arxiv.org/abs/2602.03542).
Reviewed the main methods/results in the v2 PDF, including curriculum ablations;
the v1 HTML is older and must not replace the v2 source for those ablations.
Question: do learned procedures transfer between code, graphs, and language?
Method: isomorphic planning instances, pretrained billion-parameter models,
several post-training methods, and a symbolic-to-language curriculum.
Result: naive transfer is unreliable; a two-stage RL curriculum improves planning.
Difference: our experiment trains much smaller models from scratch, uses
three-way propositional inference, and independently controls names, pairs,
graph family, and proof depth. The paper does not establish that our models
will transfer, or that a supervised curriculum must work.

### Valle and Reid, August 2026

[Protoreasoning in Tiny Transformers](https://arxiv.org/html/2608.04980v1).
Reviewed task definitions, methods, results, trace ablations, and implementation.
Question: can tiny transformers benefit from explicit reasoning traces?
Method: roughly 1M-parameter decoders on two Dyck/tree tasks; ten seeds,
structural and length holdouts, and trace-content ablations.
Result: traces often improve generalization, but extrapolation remains difficult;
filler-token ablations distinguish trace content from added computation.
Difference: our first condition is answer-only propositional inference rather
than pruning traces. Their positional augmentation and multi-seed design motivate
controls here; their successes are not guarantees for our task or smaller scales.

### Ye et al., NeurIPS 2025

[How do Transformers Learn Implicit Reasoning?](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5f1cb1d23261b19cbd45f90f7b4f251f-Abstract-Conference.html).
Reviewed the symbolic construction and behavioral experiments in the full PDF;
mechanistic sections and training appendix require a further detailed reading.
Question: how does implicit multi-hop reasoning emerge?
Method: scratch-trained transformers with targeted relational-composition
omissions, semantic patching, and representation diagnostics.
Result: memorization, ID generalization, and cross-distribution behavior emerge
at different stages; second-hop exposure matters.
Difference: our rules/facts are given in each prompt rather than relying on
memorized relational facts. Learning curves and hop-specific controls are still
necessary, but we cannot import their mechanistic conclusions without probes.

### Lazić et al., April 2026

[To See the Unseen](https://arxiv.org/html/2604.21632v1).
Reviewed the architecture, collapse argument, logic experiments, and limitations.
Question: why does symbolic reasoning fail on unseen tokens?
Method: Horn-rule reasoning with decoder models, embedding interventions,
copy attention, and theoretical/empirical collapse analysis.
Result: unseen-token unembeddings can collapse; combined interventions improve
generalization. Their symbolic models are about 60M-75M parameters.
Difference: our held-out names are combinations of known characters, not unused
token IDs, and our output is a fixed three-class head without symbolic copying.
Our experiment therefore cannot test their unembedding-collapse claim directly.

## Recent additions discovered in the search

- [Universal Transformers for Circuit Computations, August 2026](https://arxiv.org/html/2608.31067v1): examined its architecture/construction sections. It uses explicit circuit-depth information, constrained attention, iterative reduction, and halting. Its Boolean-expression guarantees must not be attributed to ordinary fixed-depth decoders. Detailed experimental review remains pending.
- [What Do Current Systematic Generalization Tasks Miss?, September 2026](https://arxiv.org/html/2609.19212v1): examined task definition and generalization results. TranSGrid separates demands that simple productivity tests can omit. Our task measures controlled deduction, not comprehensive inductive/abductive reasoning. Detailed review remains pending.
- [Algebraic Decomposition Theory for Transformer Length Generalization, August 2026](https://arxiv.org/abs/2608.13433): discovered and screened; full review pending.
- [In-Context Learning Amplifies a Latent Symbolic Circuit, September 2026](https://arxiv.org/abs/2609.36265): discovered and screened; full review pending.

Search date: 13 October 2026. Queries included transformer symbolic reasoning,
cross-representation procedures, tiny transformers, length generalization, and
September/October 2026 additions. Sources: arXiv, PMLR, NeurIPS, and primary
paper PDFs. This is a growing scoped review, not a claim to have read every
paper on the topic. Check references/citations and recent arXiv submissions again
before finalizing the related-work section.

## Experimental protocol and interpretation gates

1. Establish source-format learning before interpreting transfer. Operational
   gate: at least 95% source validation accuracy, with all three class recalls
   at least 90%; confirm once on the independent IID test. Do not tune on tests.
2. Main scales, per the updated objective: approximately 100K, 500K, 1M, 5M,
   and 10M; record actual trainable parameter counts. The old 10K/3M presets
   remain optional diagnostics, not required primary conditions. The 100K
   preset was adjusted after the software smoke test to about 99K parameters;
   the 89,763-parameter smoke checkpoint is not that final architecture.
3. Train separately on symbolic, function syntax, and template English. Render
   the same held-out latent records in all three forms; never move an instance
   between splits. Evaluate all nine source-to-target combinations.
4. Use seeds 1, 2, and 3 for final conditions. Pilot LR/budget choices use only
   source validation. Record example/token exposure and compute because equal
   epochs across representations do not imply equal token or compute budgets.
5. Evaluate unseen familiar-symbol pairs, new symbol names, diamonds, and depths
   2-5 independently. Unknown labels have scaffold depth, not a valid proof depth.
6. Report overall accuracy, balanced accuracy, per-class recalls/confusions,
   finite-test confidence intervals, and variation across training seeds. Pair
   transfer comparisons by instance ID; use paired tests with multiplicity
   correction for confirmatory comparisons, not unpaired overlapping intervals.

## Required controls still to implement/run

- Completed: exact forward-inference baseline, with classical verification in
  the corpus audit; training-majority and expected uniform-random baselines.
- Completed: validation-selected linear classifier on ten count, polarity,
  operator, and length features. On the full 10K IID test it gets 33.30%, and
  on the other test splits 33.10%-33.84%, versus 100% for exact inference.
  This checks these features only; it does not prove every shortcut is absent.
- A parse-and-normalize control to distinguish syntax recognition from inference.
- Syntax/character-exposure control: a shared alphabet alone does not mean all
  English/function punctuation and letters were trained. Current raw-transfer
  metrics explicitly record unseen source character IDs. Low raw-transfer
  accuracy alone cannot establish failure of an abstract procedure.
- Symbol-renaming, statement-order, irrelevant-fact, and premise-removal probes.
- A length-matched distractor control to separate proof depth from context length.
- If source learning fails, investigate encoding/optimization/capacity without
  replacing the main experiment with an easier task or claiming transfer failure.
- Curriculum/mixed-format conditions after the answer-only baseline; identify
  adaptation separately from zero-shot transfer and match exposure budgets.

## Where logical reasoning starts: reporting rule

Report the smallest **tested** capacity that passes the source-learning gate
across all three final seeds. Then separately report the smallest capacity that
passes the same accuracy/recall criterion on each independent holdout and under
the robustness probes. These are different onsets, not one interchangeable claim.
If adjacent tested sizes straddle a criterion, give an interval between their
actual parameter counts; do not invent a precise universal threshold. If no
size passes, report that onset was not observed under the tested training budget.
Training fit alone does not prove reasoning; syntax-normalized success alone
does not prove raw cross-representation understanding. Log every failed run,
architecture change, budget, and seed rather than selecting only successes.

## Completion ledger

Completed: audited corpus; three lossless renderers; causal classifier trainer;
checkpoint/validation selection; smoke training/evaluation; graph exporter;
full-corpus exact and superficial-feature baselines; five primary size presets.
In progress: larger pilot, detailed literature review, source learning gate.
Pending: full multi-seed matrix, controls, final statistical comparisons, final
graphs/tables, and one-sentence result-based conclusion. No final conclusion yet.
