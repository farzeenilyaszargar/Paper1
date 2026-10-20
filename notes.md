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

### Shakhvalieva et al., September 2026

[Recursive-model optimization study](https://arxiv.org/html/2609.39967v1),
reviewed 20 October. Read main methods/ablations and appendices B/E/F;
implementation not reproduced.

- Question: what makes compact recurrent solvers train and generalize reliably?
- Method: shared recurrent blocks, controlled gradient horizons and state
  stabilization; six domains with domain-specific training settings.
- Result: optimization strongly affects generalization; larger blocks and extra
  computation do not help uniformly.
- Caveats: ARC evaluation is transductive; the unsuccessful joint-training
  diagnostic is single-seed and stopped early. The physical-batch/accumulation
  ablation concerns recurrent Game of Life, not ordinary causal classifiers.
- Difference: our parameter sweep uses nonrecurrent answer classifiers. Report
  microbatch/accumulation separately; this study supplies no threshold for our task.

This focused search adds one methods review, not an exhaustive-literature claim.

### He et al., September 2026

[Task-ceiling study](https://arxiv.org/html/2609.33134v1), v1, reviewed 20 October:
main methods, Appendix B/C assumptions and G/H setup/results; no independent
proof verification or reproduction.

- Question: when does chain content supply necessary computation?
- Method: shallow-circuit bounds; finite-group experiments, width 512,
  2/4/8 layers, mostly three seeds; dense no-chain versus trace supervision.
- Result: answer-only training also fails on solvable controls; failed optimization
  is not a theorem test.
- Caveat: log-precision hardness is conditional; Appendix C strengthens to
  almost-everywhere hardness. These are not finite 2-hop impossibility bounds.
- Difference: our answer-only 2–5-hop benchmark supplies no CoT interventions;
  this paper establishes no parameter onset for our task.

### Minegishi et al., May 2026 revision

[Analogy study](https://arxiv.org/html/2602.01992v5), v5. Reviewed main methods,
limitations and appendices A-C/R-S; not reproduced.

- Question: how do learned cross-category mappings emerge?
- Method: entity/relation tokens, isomorphic categories; one-layer width-128
  default, three seeds.
- Result: optimization-sensitive, nonmonotonic generalization; rate sweeps
  rescue deeper models.
- Caveats: correlational mechanisms; unresolved batch-32 (main) versus batch-64
  (Appendix A).
- Difference: ours uses fresh KBs/shared characters and cross-grammar transfer,
  not learned-ID mapping holdouts.

Focused review, not exhaustive coverage.

## Experimental protocol and interpretation gates

1. Establish source-format learning before interpreting transfer. Operational
   gate: at least 95% source validation accuracy, with all three class recalls
   at least 90%; confirm once on the independent IID test. Do not tune on tests.
2. Main scales, aligned with the current goal on October 20: approximately 10K,
   50K, 100K, 500K and 1M (11,491 / 50,771 / 99,075 / 457,251 / 1,004,547
   actual parameters). The previous queue included 5M/10M; neither had trained.
   These and 3M remain optional presets, not required primary conditions. This
   correction follows the failed 100K result and an unfinished 500K run, and is
   not externally preregistered. All existing architectures/data/budgets and
   interpretation gates are unchanged. The 100K
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
   The three-rate 100K pilot is complete: 99,075 parameters, symbolic/characters,
   seed 101, 20K train/2K validation, ten epochs per rate, microbatch 8 with
   accumulation 4, one CPU thread, validation-only. All three selected their
   epoch-10 checkpoint. LR 0.0003 gives validation loss 1.100379 and 33.45%
   accuracy; LR 0.001 gives 1.098881 and 32.35%; LR 0.003 gives 1.098089 and
   34.25%. The recorded minimum-loss rule therefore selects **LR 0.003**.
   None passes source mastery. The winning checkpoint predicts unknown only,
   matching the validation-prefix majority baseline (635/680/685 labels).
   A lower validation loss or a selected rate does not establish inference.
   The audit checked completed validation-only status with empty test results,
   all ten epoch/step/example counters, common configuration/fingerprint,
   history loss minima, confusion-based metrics and referenced file hashes.
   `models/calibration_100k_selection.json` records the choice and provenance;
   histories/curves retain intermediate observations without treating them as
   final results. This is subset calibration, not an optimum for all capacities,
   full-corpus settings or representations. It does not replace the baseline.
   After all six 100K warmups finished, the freed one-thread CPU slot was used
   for the first 1M validation-only pilot (1,004,547 parameters, seed 101,
   symbolic, LR 0.001, 20K train/2K validation, ten epochs, microbatch 4 with
   accumulation 8). `models/calibration_1m_protocol.json` records this exploratory
   three-rate plan before its first run; remaining rates are pending. It is not
   a final baseline, a task-success result, or an optimally calibrated setting.
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
  The selected full-corpus 100K checkpoint's CPU evaluation completed all three
  symbolic 10K-pair comparisons: 33.33% accuracy on both sides for 2 versus
  3, 4 and 5 hops. Every paired accuracy difference is zero; exact McNemar and
  within-run three-comparison Holm p-values are 1. The exporter rechecked all
  retained prediction hashes, paired counts/confusions/accuracy/statistics,
  baseline/checkpoint provenance and completed status. A constant-class
  checkpoint that failed source learning does not demonstrate depth generalization.
  The length-control figure/table exporter now rejects incomplete reports,
  verifies baseline/checkpoint and retained-prediction checksums, reaggregates
  accuracy and paired comparisons, and checks the three-comparison Holm
  correction before writing figures. Its constant-head integration fixture is
  software evidence only. The actual completed checkpoint's measured graph and
  table are `graphs/baseline_full_100k_symbolic_seed1_length_control`.
  This control does not equalize fact/operator/distractor counts or graph details,
  and shared 2-hop cores across depths make conditions correlated. Any observed
  accuracy gap is not by itself a causal depth effect.
- If source learning fails, investigate encoding/optimization/capacity without
  replacing the main experiment with an easier task or claiming transfer failure.
- Curriculum/mixed-format conditions after the answer-only baseline; identify
  adaptation separately from zero-shot transfer and match exposure budgets.

## Where logical reasoning starts: reporting rule

The new `models.final_analysis` tool implements the primary-matrix coverage
gate and descriptive criteria. Its actual current export reports only one of
45 aggregate/config-complete conditions, with no final statistics or onset.
Only after all conditions exist does it verify every retained 30-epoch budget,
parent-file hash, test ID/gold label, per-class confusion/accuracy/recall and
prediction hash; architecture/renderer provenance must match. It hashes each
selected checkpoint and history without claiming to rerun inference/training.
Its fixed primary family is 45 runs × 7 splits × 2 alternate target formats =
630 paired McNemar contrasts, corrected together with Holm. This family was
documented before matrix completion, not externally preregistered. Calibration,
warmups and intervention analyses remain separate; this does not correct every
possible paper claim or supply training-seed uncertainty. Each behavioral
holdout/transfer criterion requires validation plus source-IID mastery and
target success in all three seeds. Nonmonotone size outcomes are preserved,
not forced into a universal threshold. Five targeted tests passed, including
partial matrices, duplicate contrasts, full-family correction, three-seed gates,
retained split-ID/gold checks and truncated-history rejection. Unit fixtures
are not scientific results. New suite invocations refresh the report after
each completed condition; the current live supervisor uses its already-loaded
version, so this turn ran the command manually without interrupting training.
The strict individual verifier also passed on the actual completed 100K
symbolic seed-1 baseline (all 14 contrasts, exact test IDs/golds, full history
and checkpoint/prediction hashes); its source criterion remains false. This is
not a completed matrix or global-family result. The full regression suite now
passes all 56 tests. Latest retained training measurements are epoch 18 for the
500K baseline, completed epoch 10 for the 100K calibration and epoch 8 for the
1M pilot. They still show constant-class answers and one-third balanced accuracy;
no onset is claimed. At these latest epochs all predict unknown only: the full
baseline scores 33.33%, and each 2K-validation pilot scores its 34.25% majority baseline.

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

Figure 1 is now `graphs/task_illustration` (PNG/SVG/JSON), regenerated by
`models.task_illustration`. This minimal toy is deliberately not sampled from
the experimental corpus and omits its balancing scaffolds/distractors. All
three canonical renderings reparse to the same KB; classical SAT and forward
inference verify the entailed query and minimum depth 2. The diagram explicitly
shows the AND premise join, and the new targeted test checks that deleting
either fact changes the answer to unknown. This is methodological illustration,
not a neural result, additional training data, or a corpus-size statistic.
The existing 49-test regression suite passed on 20 October; the added Figure 1
test also passed separately after implementation and the rendered PNG was
visually inspected. No training settings were changed for this figure.

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
nineteen relevant-paper methods/results reviews, with version distinctions recorded;
the first full-corpus 100K symbolic seed-1 baseline; verified paired-comparison tooling.
Also completed: full-IID behavioral interventions and retained-prediction audit
for that checkpoint, with a measured robustness figure/table; tested validation-only
trainer path and the first optimization pilot (LR 0.0003; no source mastery).
All three 100K pilot rates are complete without source mastery. LR 0.003 is
selected by the recorded validation-loss rule for the downstream controls;
the separate 1M calibration is not yet complete.
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
all-format counterpart also completed: 5,515 records started (1,838 symbolic,
1,838 function, 1,839 English), final logged minibatch LM loss 0.910378.
The paired checkpoint/provenance/history checks pass for seed 3, and its measured
figure is `graphs/warmup_100k_symbolic_seed3`. All six 100K warmups are done.
Matched supervised comparisons are pending; warmup LM loss is not inference
accuracy or task mastery.
The downstream protocol is now fixed before any supervised-control result:
99,075 parameters, symbolic source, seeds 1/2/3, full 100K train and 10K
validation, 30 epochs, microbatch 16 × accumulation 2 (effective 32), 93,750
supervised updates and 3M supervised examples per arm. The rate was left unset
until all three original calibration trials finished; LR 0.003 is now selected
only by minimum selected validation loss, with declared-order tie breaking.
Reset optimizer state for supervision and retain the full original 21 test
evaluations only after source-validation checkpoint choice. Run no-warmup,
source-only warmup, then all-format warmup within each seed, on one CPU thread.
The main contrast is between the two matched warmup budgets. No-warmup has
the same supervised budget but less total training compute, so it is a secondary
reference, not an equal-total-compute intervention. Reuse a primary baseline
only if every supervised setting agrees, including physical microbatch and
learning rate, device, thread count, optimizer, batching and learning-code
provenance. The current MPS baseline is not reused as a matched CPU reference.
This is an internal plan after existing diagnostics, not external preregistration.
Subset calibration does not establish optimal full-corpus
settings or mastery when every rate fails. These controls remain separate from
the 45-condition matrix; current baseline budgets and model definitions are
unchanged. The seed-1 no-warmup CPU reference is now training in
`models/runs/exposure_full_100k_symbolic_seed1_no_warmup_lr003`; no completed
supervised-control result exists yet, and the two warmup-supervised arms wait
for their scheduled turn. `--purpose control` gives the no-warmup reference a
separate result scope so it cannot merge into the primary baseline plots.
Resume rejects changing a run's purpose; these additions do not alter optimizer
calculations. The new small-fixture scope/resume tests are software-only.
New process invocations have storage preflight/checkpoint guards. On 19 October,
the Mac had approximately 3.6 GiB disk free and 8.8 GiB encrypted swap in use;
the three existing scientific processes were live. No extra warmup workload was
launched under that resource pressure. By 20 October disk free had recovered
to about 12 GiB and system memory free was 25%; the one-thread warmup was
then launched. Existing processes are not hot-patched by source edits.
In progress: 45-condition full-corpus baseline matrix, further
literature screening, and source learning gate. The matrix uses 30 epochs,
LR 0.001, and effective batch size 32; it is not called optimally calibrated.
Also completed: the full length-matched corpus's independent audit and its first
checkpoint evaluation/verified measured figure. The 100K rate selection is
complete, and its first matched no-warmup reference is running. Code and
integration tests are not neural results.

### Verified recovery on 20 October, approximately 18:26 IST

No suite, trainer or control-evaluation process was live in the system process
table; the old evaluation session handle was missing. This was a verified stop,
not an observation timeout. The control evaluation and LR 0.001 100K pilot had
completed successfully. The 500K baseline retained an epoch-14 checkpoint and
had begun epoch 15; the 1M pilot retained an epoch-4 checkpoint. The cause of
the training stop is unknown: no terminal exception or exit code was retained.
The suite was restarted first and resumed the 500K run from epoch 15; the 1M
pilot was then resumed from epoch 5, and the remaining 100K LR 0.003 trial was
launched. Existing histories/checkpoints were preserved. Unsaved partial epochs
are replayed, so completed-epoch example/token counters exclude any lost partial
work and must not be claimed as exact total physical compute. Resume invocation
records retain the checkpoint starting epoch and full configuration. All three
new processes were checked live after launch; stale suite state alone was not
used as evidence of liveness. The 45-condition objective remains incomplete.
The resumed 500K run subsequently completed epoch 15: validation accuracy
33.34%, balanced accuracy one third, with every answer predicted entailed.
Training continues to the scheduled 30 epochs; this observation is not a final
500K result, a source-learning success or a capacity threshold.
The resumed 1M pilot completed epoch 5: validation accuracy 34.25%, loss
1.098922, balanced accuracy one third, with every answer predicted unknown.
This equals its validation-prefix majority baseline and is not inference
learning. Its remaining five scheduled epochs and other calibration rates
are still pending; no 1M hyperparameter winner is selected yet.

### Verified recovery and control launch after the interrupted turn

At the next process-table inspection no suite or trainer was live, and the old
audit/test session handle was missing. The cause is not inferred from the
interruption message alone. The 500K checkpoint retained epoch 18 (epoch 19 had
started), and the 1M pilot retained epoch 8. They were resumed from epochs 19
and 9 respectively, preserving histories and optimizer state; lost partial
work remains outside retained counters. New invocations record the new source
hashes; the control-scope/resume-purpose changes do not alter optimizer math.
The completed 100K calibration was not restarted. Its three reports were audited
and LR 0.003 selected by validation loss only. The seed-1 no-warmup CPU reference
then launched under the fixed downstream budget, separately labeled with
`--purpose control`; its configuration and liveness were checked. All 56 tests
pass, including no-warmup pilot/full scope and purpose-change resume rejection.
The primary matrix is still incomplete, and no downstream control result or
reasoning onset is claimed.
Pending: full multi-seed matrix, controls, final statistical comparisons, final
graphs/tables, and one-sentence result-based conclusion. No final conclusion yet.

### October 20: current capacity scope and lossless queue handover

The active user goal specifies 10K, 50K, 100K, 500K and 1M, not the previous
100K/500K/1M/5M/10M queue. The revised primary matrix retains 45 conditions
(five sizes x three source formats x three seeds), 30 epochs, LR 0.001 and
effective batch 32. The existing 10K preset has 11,491 parameters; the new
50K preset has 50,771 (width 32, three layers, four heads, FF width 176).
The other three primary architectures are unchanged. Larger presets remain
available but are outside this matrix. This scope correction was made after
the failed 100K baseline and while the 500K run was unfinished; it must not be
described as an externally preregistered size selection or motivated by an
observed favorable 10K/50K result (neither had been trained in the main matrix).

The old supervisor alone was stopped, leaving its 500K accelerator child live
at PID 48920 without interrupting an epoch or resetting its optimizer. The
replacement supervisor uses `--drain-pid 48920`, holds the OS lock, verifies
the same command while waiting, and checks accelerator exclusivity again
before starting the corrected queue. Explicit CPU calibration/control jobs can
coexist; automatic/MPS trainer overlap is rejected. The process parser handles
unquoted unrelated application arguments without confusing them with trainers.
The 500K run had completed epoch 26/30 at inspection; the no-warmup CPU control
had completed epoch 3/30. Both remained near chance. These are interim counts,
not completed scientific conditions or estimates of physical total compute.

Adding a preset changes the transformer file hash without changing existing
model computations. Final analysis now accepts only one explicitly pinned
historical transformer revision when its per-size architecture and the entire
non-registry computation AST match exactly. Unknown source hashes, changed
computation and changed historical head/width/layer/FF settings are rejected.
The actual completed 100K configuration passes this provenance check; tests
cover both acceptance and rejection. Raw historical source hashes remain in
the reports, rather than being rewritten to the new hash.

The first 1M calibration trial (LR 0.001) completed all ten epochs with no test
evaluation. Minimum validation loss was 1.098180 at epoch 9, accuracy 34.25%,
with every answer unknown (balanced accuracy one third); epoch 10 accuracy
was 34.0%, all contradicted. This is not source mastery. Its complete retained
history, empty test-evaluation mapping and minimum-loss selection were checked.
The second trial, LR 0.0003, is now running with the same 20K/2K subset,
seed 101, CPU thread count, microbatch 4 and accumulation 8. LR 0.003 remains
pending; no 1M rate is selected before all three trials finish.

The revised final-analysis coverage report still records only 1/45 complete
main conditions and refuses final statistics/onsets. Software validation has
60 tests, including the requested capacities, exact parameter counts, matched
effective batches, historical provenance checks and safe trainer handover.
No claim of learned 2-hop inference or a capacity threshold is supported yet.

### October 20: literature follow-up and measured curve refresh

The focused primary-source search added the eighteenth reviewed reference
above; it is not an exhaustive-literature claim. No experimental setting was
changed based on this reading. The answer-only training failures remain
optimization observations, not a proof that these finite tasks are impossible.

`python -m models.plots` regenerated actual retained learning curves. The 1M
LR 0.001 calibration curve now contains all ten completed epochs. The 500K
baseline and no-warmup 100K control curves remain partial snapshots at 26/30
and 3/30 epochs respectively, not completed final results. All three changed
PNG plots were visually inspected, including labels, legends and chance line.
The five completed evaluated runs in the generic plotting export include
smoke/pilot/diagnostic conditions; this is not five completed primary conditions.
The validation-only calibration has no transfer/test measurement and is not
added to that evaluated-run count. Existing measured-results tables stayed
unchanged. Live process checks confirmed the main child, revised waiting
supervisor and both CPU jobs; none was restarted or interrupted for plotting.

### October 20: guarded automatic auxiliary handoffs

`models.auxiliary_queue` now schedules the three declared 1M calibration rates
and nine declared 100K downstream syntax-exposure conditions as two separate
single-threaded CPU lanes. This changes scheduling only: no examples, rates,
seeds, epochs, physical/effective batch sizes, optimizer settings or comparison
arms were added or changed. Both lanes check their frozen protocol and use
independent OS locks. They can coexist with the one accelerator job. The queue
processes do not import torch; their idle supervisory memory is small.

The calibration supervisor was verified live at PID 66353, waiting for the
existing LR 0.0003 child PID 63602; the supervision supervisor was verified live
at PID 66357, waiting for the existing no-warmup child PID 50388. Neither child
was interrupted. After each finishes, its lane checks coverage before advancing
to the next planned condition. The current 500K accelerator child and revised
main supervisor were also checked live. An attempted duplicate calibration
supervisor was rejected by its OS lock without changing the active state file.

Queue coverage checks include protocol/rate-selection integrity, all six
completed warmup initialization hashes/budgets, matching configuration, full
retained epoch/example/update counts, minimum-validation-loss selection,
checkpoint existence, and absence of test evaluations for calibration.
Supervised controls must retain all 21 full test groups and prediction files.
File existence/count coverage is not a substitute for the independent retained-
prediction correctness/statistics audit. A partial directory without a saved
epoch checkpoint or a conflicting live output refuses dispatch rather than
being overwritten. CLI plotting now uses one shared lock across all lanes.

The architecture-provenance guard reads literal registry metadata with AST
inspection rather than importing torch, while preserving the pinned historical
computation/preset checks. Actual completed calibration provenance and all six
initialization files were rechecked. All 66 tests pass, including six auxiliary-
queue tests. Unit fixtures for scheduler checks are not scientific training or
prediction outputs.

The 500K run completed epoch 27/30: validation accuracy 33.33%, balanced accuracy
one third, all answers contradicted. This still fails source learning. It is
not a completed 500K result or a capacity threshold. The full primary matrix,
downstream control outcomes, final statistics and paper conclusion remain open.
