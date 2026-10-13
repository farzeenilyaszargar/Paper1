# Research notes - 13-14 October 2026

## Question and current evidence

Can a transformer trained from scratch on 2-hop propositional inference apply
that procedure under a change of representation, symbols, graph family, or depth?
The corpus has 100K training instances and eight 10K evaluation/validation splits.
The logic fragment, minimum parallel-rule depth, and classical three-way labels
are defined in the README. No model has yet met the learning gate below.

The first 89,763-parameter smoke run used only 300 training examples, two epochs,
and 30 validation examples. Best-validation-loss selection produced 26.7%
validation accuracy. This is a software check, not evidence of learned reasoning
or a meaningful estimate of transfer. The 457,251-parameter pilot on 20K examples
was interrupted after epoch 18 and is being resumed from its verified checkpoint.
A separate 99,075-parameter binding-normalized pilot fit its 2K training examples
(95.35% final online training accuracy), but its selected checkpoint got 30.2%
IID test accuracy on 1K cases. It does not satisfy the learning gate.
The 16-example memorization diagnostic first reached perfect online training
accuracy at epoch 17; this verifies fitting capacity, not reasoning generalization.

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
Reviewed symbolic construction, behavioral experiments, semantic patching,
cosine diagnostics, and the training/scaling appendix in the full PDF.
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

## Recent additions: methods/results review

### Ito et al., August 2026

[Universal Transformers for Circuit Computations](https://arxiv.org/abs/2608.31067).
Reviewed construction, numerical experiments, ordinary-looped-transformer
comparisons, and limitations in the v1 PDF. Question: can a tiny architecture
learn exact depth-general circuit evaluation? Method: explicit parenthesis-depth
gating, constrained attention, discrete states, repeated reduction, and halting.
Result: converged models generalize perfectly on sampled depths beyond training;
Boolean/ListOps convergence is 9/10 seeds, not every initialization. Difference:
our ordinary decoders have no built-in circuit-reduction algorithm or provided
depth. Their 280-parameter Boolean result is not a universal capacity threshold.

### Qi et al., September 2026

[What Do Current Systematic Generalization Tasks Miss?](https://arxiv.org/abs/2609.19212).
Reviewed task, experiments, controlled variants and architecture/data appendices
in v2 (29 September); the earlier v1 HTML is not the current reference.
Question: what reasoning demands do productivity tests miss? Method: interacting
grid actions and executable goal-sequence evaluation with seven encoder-decoder
sizes, plus decoupled/action-explicit variants. Result: the harder systematic
benchmark exposes failures even within training-length ranges. Difference: our
deductive entailment task does not cover induction and abduction comprehensively.
Depth extrapolation is only one axis; it must not be equated with all systematic
reasoning. Their solve rates are not comparable to our three-way accuracy.

### Yang et al., COLM 2026

[Algebraic Decomposition Theory for Transformer Length Generalization](https://arxiv.org/abs/2608.13433).
Reviewed main characterization/algorithm and experimental setup/results in the
v1 PDF; the long formal appendices are not independently proof-checked.
Question: which regular languages admit C-RASP length generalization? Method:
typed-monoid decomposition and a 125-language state-tracking suite.
Result: the characterization predicts the measured language-dependent behavior.
Their empirical selection uses long-test performance and a search for successful
seeds. Difference: our validation-only, all-seed reporting avoids that selection
strategy, and our rule-query task is not the same regular-language setting.
The theorem does not give a parameter-onset value for this experiment.

### Wessel, September 2026

[In-Context Learning Amplifies a Latent Symbolic Circuit](https://arxiv.org/abs/2609.36265).
Reviewed circuit tracking, cross-shot patching, function vectors and limitations
in the v1 PDF. Question: do demonstrations create or strengthen rule circuits?
Method: causal interventions on ABA/ABB rule completion in pretrained models.
Result: functioning components are detectable below high behavioral accuracy,
and intervention can restore performance. Difference: our models train from
scratch on multi-hop entailment, not copying rules in pretrained billion-scale
systems. A behavioral onset criterion is not the first mechanistic appearance
of reasoning. Failure of an accuracy gate cannot prove every relevant circuit
is absent.

Search dates: 13-14 October 2026. Queries included transformer symbolic reasoning,
cross-representation procedures, tiny transformers, length generalization, and
September/October 2026 additions. Sources: arXiv, PMLR, NeurIPS, and primary
paper PDFs. This is a growing scoped review, not a claim to have read every
paper on the topic. Check references/citations and recent arXiv submissions again
before finalizing the related-work section.

Additional screened candidates for later detailed review: [Tower of Hanoi world
models](https://arxiv.org/abs/2608.07077) and [unfaithful arithmetic reasoning](https://arxiv.org/abs/2602.01017).
Their abstracts are relevant to interpreting behavioral failures and trace
faithfulness, but they have not yet received the same methods/results review.

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
- Completed: text-only round-trip and pairwise normalization audit on all 180K
  records (540K rendered forms, 1.62M directed normalizations). This is an encoding
  invariant, not an extra neural transfer success. Parsers never consult targets
  or proof metadata. The binding-normalized pilot remains below chance-level
  generalization despite high training fit.
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
full-corpus exact and superficial-feature baselines; five primary size presets;
lossless text-only parsing audit; memorization and binding-normalized diagnostics.
In progress: resumed pilot, 45-condition full-corpus baseline matrix, further
literature screening, and source learning gate. The matrix uses 30 epochs,
LR 0.001, and effective batch size 32; it is not called optimally calibrated.
Pending: full multi-seed matrix, controls, final statistical comparisons, final
graphs/tables, and one-sentence result-based conclusion. No final conclusion yet.
