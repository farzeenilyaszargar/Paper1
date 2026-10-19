# Research notes - updated 20 October 2026

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
completed 30 epochs after resuming its epoch-18 checkpoint. Its long-English
evaluation exhausted MPS memory with dynamic sequence shapes and batch size 32;
the repeated evaluation completed using smaller batches and 128-token padding
buckets. Selected validation accuracy is 34.2% (1K cases), and symbolic IID test
accuracy is 32.2% (1K cases). It fails the source-learning gate. The memory error
was a runtime failure, not an inference accuracy result.
A separate 99,075-parameter binding-normalized pilot fit its 2K training examples
(95.35% final online training accuracy), but its selected checkpoint got 30.2%
IID test accuracy on 1K cases. It does not satisfy the learning gate.
The 16-example memorization diagnostic first reached perfect online training
accuracy at epoch 17; this verifies fitting capacity, not reasoning generalization.

The first full-corpus condition, `baseline_full_100k_symbolic_seed1` (99,075
parameters), completed 30 epochs on all 100K training examples. Its selected
checkpoint gets 33.33% validation and 33.33% symbolic IID test accuracy (10K
each), predicting only `contradicted` in both. Final online training accuracy
is 33.329%; this is failure to fit, not just overfitting. No onset is established.
On 19 October the old suite/trainer PIDs and supervisor tool handle were missing.
The next 500K job had stopped before its first completed epoch, without a
checkpoint or logged exception; its cause is unknown. Its config/invocation/log
evidence was preserved under `models/interrupted_runs/`. The queue was restarted,
skipping the validated completed 100K condition. No measurements were discarded.

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

### Pereira and Zuidema, August 2026

[Transformers Struggle to Use Their Emergent World Models](https://arxiv.org/abs/2608.07077).
Reviewed the small-model training, state probes, interventions and large-model
comparison in v1; visually checked the probe figure and intervention table.
Question: what fails during Hanoi planning? Method: solution-trace training,
state-distance/per-disk probes, whole-stream patches and activation steering.
Result: useful state information can be present at the prompt yet degrade during
generation. Their intervention is not a surgical test of the decoded 2D subspace.
Difference: our classifier emits one label, without a long planning rollout.
This study warns against equating behavioral failure with absent internal
knowledge; it does not diagnose our current training failure.

### Wang, Alazali and Zhong, February 2026

[How Does Unfaithful Reasoning Emerge from Autoregressive Training?](https://arxiv.org/abs/2602.01017).
Reviewed synthetic construction, consistency/intervention metrics, noise results,
limitations and training details in v1; visually checked the faithfulness definitions.
Question: when do arithmetic traces causally follow intermediate steps?
Method: controlled corruption of modular-arithmetic chains and counterfactual
interventions on the intermediate expression in small autoregressive models.
Result: reasoning modes and faithfulness depend on noise, complexity and training
duration. Difference: our answer-only classifier has no generated trace whose
faithfulness can be scored directly. Verified labels do not establish an internal
stepwise proof; any later trace condition needs an independent causal check.

### Ma et al., revised 14 October 2026

[Making Transformers Logical, v2](https://arxiv.org/html/2608.00418v2).
Reviewed main methods/results, limitations, mask design and training details in
HTML, not an independent proof verification. Question: can attention priors
change Boolean extrapolation? Method: task-designed, learnable additive masks
versus vanilla and QK-initialized transformers, with five-seed Boolean tests.
Result: the masks retain useful interaction patterns and improve unseen-case
accuracy; QK initialization can lose the same prior. Difference: masks encode
domain knowledge explicitly (an acknowledged oracle assumption); our baseline
must learn dependencies from raw prompts. This motivates a separately labeled
structural-prior control, not attributing current failure to insufficient
parameter count or silently adding oracle masks to the primary conditions.

### Zhang et al., revised 27 August 2026

[Why Knowing Both Hops Is Not Enough, v2](https://arxiv.org/html/2608.07261v2).
Reviewed task/split definitions, training dynamics, patching/attention analyses,
looped training, and training details in HTML. Question: why can known atomic
facts fail to compose? Method: memorized entity-relation facts, an 8-layer,
width-768 transformer, independent first/second-hop omissions, and interventions.
Result: second-hop distribution matters; weight sharing improves their failures,
consistent with a representation/upper-layer mismatch. Difference: our facts
arrive in each input, names share character pieces, and outputs are three
classes. Their long training and looped architecture are not a measured capacity
threshold for our task. A binding or optimization diagnosis here still needs
our own evidence, rather than importing their mechanism.

Search update on 19 October screened recent symbolic/tiny-transformer reasoning
results and retrieved these primary sources. This is a documented relevance
search, not a claim to have read every new paper. The Google Research entry for
*A Implies B* and its authors' code repository led to the v4 review below.
Pretrained-billion-scale two-hop blog results and task-policy
sequent-search studies were screened, not treated as direct replications.

### Hong et al., revised 19 June 2025

[A Implies B, v4](https://arxiv.org/html/2411.04105v4).
Reviewed task/method definitions, necessity/sufficiency tests, discussion, and
small-transformer appendices C and D (selected analyses, not all model-specific
appendix interventions). The older title was *How Transformers Solve Propositional
Logic Problems*. Question: which components implement in-context logic?
Method: query/rule/fact counterfactuals and activation patching, plus scratch-trained
attention-only models. Result: sparse modular pathways in pretrained models;
small models solve a controlled template with less modular computations.
Their three-layer success uses width 768, atomic variable tokens, proof generation,
two million training examples, and warmup/cosine optimization—not a 100K-parameter
character classifier. Difference: our independent holdouts, three-class answer-only
objective, and scale range are distinct. This is a close comparator for controls,
but its circuit evidence and smallest successful architecture cannot be imported
as our mechanism or capacity threshold. Input interventions alone are behavioral
tests; our current probes do not constitute activation-patching circuit analysis.

### Maltoni and Ferrara, revised 30 June 2026

[Deductive Logic in Language Models: Horizontal vs Vertical Reasoning, v2](https://arxiv.org/html/2510.09340v2).
Reviewed task construction, models, trace/no-trace experiments, convergence and
bias appendices in HTML; publisher page was rate-limited, so this review uses
arXiv v2. Question: how does supervision change deduction?
Method: implication chains and tree navigation, with attention/residual decoding.
Result: selected proof-supervised models generalize; answer-only variants can
memorize or exploit generation shortcuts. Their 144,384-parameter trace model
has 24-token prompts, and only about 20% of runs converge within 250 epochs;
averages over converged models are not unconditional reliability estimates.
Difference: our longer inputs, multi-character names and three-class labels
exclude generated proofs. Their successes do not set our parameter threshold.
This supports separately testing supervision and optimization, without replacing
the primary answer-only task or selectively dropping failed seeds.

### Nguyen, Dang and Inoue, May 2026

[Revealing Algorithmic Deductive Circuits for Logical Reasoning, v1](https://arxiv.org/html/2605.27824v1).
Reviewed synthetic construction, patching, ablations and limitations in HTML.
Question: which components support symbolic-aided proof generation?
Method: counterfactual prompts, activation/path patching, and targeted versus
random head removal in pretrained models. Result: distinct reading/decision
roles and targeted ablation damage on reasoning benchmarks; combined ablations
also affect MMLU, so reasoning exclusivity is not established.
Difference: their trace tokens and inference-step scores are not our classifier
outputs. Our behavioral interventions alone cannot establish their circuit-level
claims or identify internal proof steps. Any future mechanism claim needs its
own intervention evidence rather than an accuracy threshold or attention picture.

Screening update: added these two directly relevant primary-source reviews.
Majority-function optimization theory, visual recursion, and Bayesian common-cause
papers were screened but not counted as reviewed: their tasks/assumptions require
separate analysis before applying their claims to this in-context logic fragment.

### Zhang, Chen and Chen, May 2026

[Satisfiability Solving with LLMs: A Matched-Pair Evaluation of Reasoning
Capability](https://arxiv.org/html/2605.28602v1), v1, reviewed 20 October.
Read main-text setup, paired-edit protocol, reductions and validity discussion;
supplemental reduction implementations were not independently reproduced.

- Question: does pretrained-model SAT performance survive small label-changing
  edits and equivalent graph/packing encodings?
- Method: solver-verified SAT/UNSAT pairs; ADR requires both answers correct.
  Comparisons include 2-SAT, 3-SAT, Vertex Cover and discrete 3D packing.
- Result: class-biased accuracy and cross-format agreement can conceal failure;
  competence declines with instance size. Main-text pooled CNF/Vertex Cover
  agreement is 75.3%, so do not repeat the abstract's broader >80% wording as
  that particular statistic. Timeout-limited coverage also complicates ranking.
- Difference: ours trains small classifiers from scratch on three-way in-context
  inference, not pretrained SAT decision/witness generation. Our label-changing
  robustness probes already report joint correctness; equal-label length pairs
  are not ADR counterfactuals. Neither joint correctness nor invariance alone
  proves an internal algorithm or eliminates every shortcut.

The 20 October search adds this directly relevant methods review, not a claim
that every recent reasoning paper has been found or read.

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
   The trainer now has a tested `--validation-only` path: calibration can end
   with the selected validation metrics and no test loading/evaluation. This
   prevents routine trainer test evaluation from becoming part of tuning.
   First actual trial launched: 99,075 parameters, symbolic/characters, seed 101,
   LR 0.0003, 20K train/2K validation, 10 epochs, effective batch 32, one CPU
   thread, validation-only. The prespecified three-rate pilot is recorded in
   `models/calibration_protocol.json`; LR 0.001 was launched after LR 0.0003
   completed, while LR 0.003 remains pending. No winning rate or source-learning
   success is claimed. This does
   not replace the full-corpus baseline or calibrate all capacities/formats.
   First-epoch observation only: 33.35% online training accuracy and 32.10%
   source validation accuracy. The 2K validation prefix has class counts
   635/680/685, so its majority-class baseline is 34.25%, not exactly one third.
   Uniform-random expected accuracy is still one third. No final rate selection
   is justified from one epoch or before the other planned rates complete.
   The epoch-2 validation accuracy was 34.35%, but loss was worse than
   at epoch 1 and entailed-class recall is zero. The minimum-validation-loss
   checkpoint had therefore not improved at that point. At epoch 9, validation
   accuracy was 32.50%. The first trial has now completed all 10 epochs:
   the selected minimum-loss checkpoint is epoch 10, validation loss 1.100379,
   accuracy 33.45%, balanced accuracy 32.72%, and entailed recall zero. Its
   report has `validation_only_completed` status and no test evaluations.
   No rate is selected before all three trials finish.
   LR 0.001 has completed its first epoch: validation loss 1.099865, accuracy
   32.30%, balanced accuracy 31.48%, entailed recall zero. This is an early
   observation from the still-running ten-epoch trial, not a completed result.
5. Evaluate unseen familiar-symbol pairs, new symbol names, diamonds, and depths
   2-5 independently. Unknown labels have scaffold depth, not a valid proof depth.
6. Report overall accuracy, balanced accuracy, per-class recalls/confusions,
   finite-test confidence intervals, and variation across training seeds. Pair
   transfer comparisons by instance ID; use paired tests with multiplicity
   correction for confirmatory comparisons, not unpaired overlapping intervals.

Implemented `python -m models.paired_statistics`: compares source-format to the
two alternate formats on each of seven test splits, pairing by exact instance ID
and verifying gold labels, counts, and confusion matrices against saved metrics.
It exports discordance tables, paired accuracy differences, exact two-sided
McNemar p-values and Holm correction over those 14 contrasts **within each run**.
Prediction-file and result hashes are retained. Missing legacy gate fields are
null, not invented successes. Current comparisons are exploratory: this is not
correction across the final size/seed/condition family, nor seed uncertainty.

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
- Completed corpus-level exposure audit: all 180K records rendered in all three
  forms. For symbolic-source training, 41.13% of function IID content-character
  occurrences (17 distinct characters) and 39.55% of English occurrences (15
  characters) use source-unseen IDs. Other cross-format pairs range from 19.55%
  to 40.93%. Every same-format held-out-name test has zero source-unseen
  character occurrences, consistent with its shared-character encoding design.
  BOS/EOS/PAD are excluded from these denominators. This is a measured input
  exposure mismatch, not a neural accuracy result or a causal attribution of
  failure. `models/character_exposure.json` retains exact counts; the figure is
  `graphs/character_exposure.png`. The audit alone does not complete actual
  training-only syntax-exposure and matched-compute controls.
  Implemented the next stage separately in `models.syntax_warmup`: fixed-token,
  fixed-window TRAIN-only next-token exposure, source-only versus all-format
  arms. The functional tied LM projection adds no parameters; the classification
  head is not trained during warmup. No label/proof strings enter LM text.
  Verified initializations can seed source-only supervised runs with reset
  optimizer moments and separate result scopes. Resume replays only the training
  stream. The initial protocol uses 2,048,000 prediction tokens per arm; it is
  not presented as an optimal budget. Both seed-1 scientific warmup arms have
  now completed; downstream matched supervised training/evaluation is pending.
  This matches LM compute/token budgets, not latent-example counts or all content
  cues; it is richer than a character-only intervention and not zero-exposure
  transfer. Primary no-warmup runs remain unchanged. Unit fixtures are not data
  for the paper; report warmup exposure separately from downstream task results.
- Implemented: raw-input symbol-renaming, statement-order, disconnected-distractor,
  query-negation and supporting-fact ablation probes in `models.robustness`.
  They parse text only and independently solver-check each new label. The
  ablation finds an irredundant supporting-fact cut (possibly multiple OR roots),
  scores only originally decisive queries and reports its different denominator.
  Completed the full 10K-IID check of the selected full-corpus 100K checkpoint
  on one CPU thread alongside, not instead of, the accelerator matrix.
  Reports include confusion matrices, joint original/transformed correctness,
  unchanged predictions and correct responses to gold-label changes. Constant
  classifiers can pass invariance but fail the causal-label-change probes.
  These are diagnostic test analyses, not validation or training interventions.
  Retained-prediction audit passes for 56,667 records: 10K for each of the five
  all-case variants, and 6,667 for fact ablation (3,333 unknown originals are
  inapplicable). The model always predicts `contradicted`. Renaming, reordering
  and distractors each score 33.33%, with 100% prediction invariance. Negating
  queries scores 33.34% but zero jointly correct original/changed pairs; it
  correctly follows none of the 6,667 gold-label changes. Fact ablation scores
  0%, again following none of 6,667 changes. This demonstrates why invariance
  alone is insufficient here, not a universal conclusion about 100K models.
  `models.audit_robustness` independently reaggregates stored prediction metrics,
  pair IDs, label-change relations and provenance checksums. It does not rerun
  model inference or reconstruct modified texts; its audit scope is explicit.
- Implemented `models.length_control`, `models.audit_length_control`, and
  `models.evaluate_length_control` for 2-hop versus 3/4/5-hop, equal-token-length
  diagnostics in all three grammars. Full generation completed on 19 October:
  90K pairs/180K logical inputs. The separate full-corpus audit passed on
  20 October, including all nine files, independent text parsing, solver labels,
  proof depths, exact length matching and training-KB exclusion. The tracked
  `datasets/length_control_summary.json` records counts and fingerprints.
  Each pair matches label and exact token count, not core KB. Both sides receive
  disconnected forest components with familiar names and nonreserved edges;
  their original decisive proof depths are retained. No filler whitespace is
  used. The auditor also checks source-core preservation and training-KB leakage.
  The small software integration corpus is byte-reproducible and audited; its
  untrained constant-head evaluation is strictly a test fixture, not paper data.
  The selected full-corpus 100K checkpoint's CPU evaluation is running;
  completed neural results and the scientific control figure are still pending.
  Its first 10K-pair comparison (2 versus 3 hops, symbolic) has finished at
  33.33% accuracy on both sides. The 2-versus-4-hop comparison also finished
  at 33.33% on both sides; its retained paired predictions, confusion matrices,
  checksums and comparison statistics were rechecked. Depth 5 is pending. The saved
  report is explicitly `evaluation_in_progress`. A constant-class checkpoint
  that failed source learning does not demonstrate depth generalization.
  The length-control figure/table exporter now rejects incomplete reports,
  verifies baseline/checkpoint and retained-prediction checksums, reaggregates
  accuracy and paired comparisons, and checks the three-comparison Holm
  correction before writing figures. Its constant-head integration fixture is
  software evidence only; no scientific length-control graph exists yet.
  This control does not equalize fact/operator/distractor counts or graph details,
  and shared 2-hop cores across depths make conditions correlated. Any observed
  accuracy gap is not by itself a causal depth effect.
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

`graphs/dataset_characteristics.md` now provides the measured Table 1 draft:
all nine split sizes, symbol pools, holdout boundaries and mean BOS/EOS-inclusive
token lengths in all three renderings. Counts come from the full-corpus
character-exposure audit with manifest/renderer provenance rechecked. The table
explicitly distinguishes unknown scaffold depth and documents length confounds;
it is not a model result or proof of independent alpha-equivalent graph holdout.

Completed: audited corpus; three lossless renderers; causal classifier trainer;
checkpoint/validation selection; smoke training/evaluation; graph exporter;
full-corpus exact and superficial-feature baselines; five primary size presets;
lossless text-only parsing audit; memorization and binding-normalized diagnostics.
Completed additionally: the 30-epoch raw pilot and its recovered evaluation;
sixteen relevant-paper methods/results reviews, with version distinctions recorded;
the first full-corpus 100K symbolic seed-1 baseline; verified paired-comparison tooling.
Also completed: full-IID behavioral interventions and retained-prediction audit
for that checkpoint, with a measured robustness figure/table; tested validation-only
trainer path and the first optimization pilot (LR 0.0003; no source mastery).
The LR 0.001 pilot is running; the three-rate calibration remains incomplete.
Also completed: full-corpus character-exposure diagnostic and measured matrix,
which quantifies a raw-transfer confound without replacing its required control.
Software completed additionally: TRAIN-only matched warmup arms, provenance-
checked supervised initialization, and CPU checkpoint/replay tests. On 20 October,
the actual 100K symbolic source-only warmup (seed 1) was launched on one CPU
thread with the prespecified 1,000-update/2,048,000-prediction-token budget.
It completed all 1,000 updates: 7,113 symbolic records started, 35 observed
input IDs, final logged minibatch LM loss 0.993857 (not corpus-average loss).
The initialization loader verified the report, budget, corpus fingerprint and
checkpoint SHA256. Its matched all-format counterpart also completed the same
seed and token/update/window/batch budget: 5,515 records started (1,838 symbolic,
1,838 function, 1,839 English), final logged minibatch LM loss 0.976620.
Both initializations pass the provenance loader. The measured
`graphs/warmup_100k_symbolic_seed1` figure/report records both curves and unequal
record counts; the losses are from different distributions, not a ranking of
reasoning ability. The prespecified seed-2 source-only warmup also completed
1,000 updates / 2,048,000 predictions: 7,114 records started and final logged
minibatch LM loss 1.047872. Initialization provenance verification passed.
The seed-2 all-format counterpart also completed the fixed budget: 5,515
records started (1,838 symbolic, 1,839 function, 1,838 English), final logged
minibatch LM loss 0.920960. The paired exporter verifies both initializations,
matched configuration/provenance and history totals before exporting
`graphs/warmup_100k_symbolic_seed2`. The seed-3 source-only warmup completed
the same budget: 7,113 records started, final logged minibatch LM loss 0.990229,
35 observed input IDs. Its initialization provenance check passed; the seed-3
all-format counterpart is now running. All three source-only warmups are done.
Matched supervised comparisons are pending; warmup LM loss is not inference
accuracy or task mastery.
New process invocations have storage preflight/checkpoint guards. On 19 October,
the Mac had approximately 3.6 GiB disk free and 8.8 GiB encrypted swap in use;
the three existing scientific processes were live. No extra warmup workload was
launched under that resource pressure. By 20 October disk free had recovered
to about 12 GiB and system memory free was 25%; the one-thread warmup was
then launched. Existing processes are not hot-patched by source edits.
In progress: 45-condition full-corpus baseline matrix, further
literature screening, and source learning gate. The matrix uses 30 epochs,
LR 0.001, and effective batch size 32; it is not called optimally calibrated.
Also completed: the full length-matched corpus's independent audit. Its neural
evaluation and the second validation-only optimization trial are running; code
and integration tests must not be reported as neural results.
Pending: full multi-seed matrix, controls, final statistical comparisons, final
graphs/tables, and one-sentence result-based conclusion. No final conclusion yet.
