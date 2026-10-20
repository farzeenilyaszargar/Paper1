# Can Small Transformers Generalize Reasoning Across Representations?

Working outline; fill numerical claims only from completed, verified experiments.

## Abstract

State the question, controlled task, model scales, and independent evaluation
axes. Report source learning before transfer. Include the actual main outcome
and limitations only after final runs; do not predict the conclusion.

## 1. Introduction

Distinguish solving one representation from reusing a procedure across formats.
Motivate scratch-trained small models and reproducible symbolic data. Position
the contribution relative to Lin et al., tiny-model reasoning, implicit reasoning,
and unseen-symbol studies. Avoid an unsupported claim of being the first study.

## 2. Related work

Cross-representation procedure generalization; tiny-model traces; implicit
multi-hop reasoning; names versus tokens; structure/length generalization.
Use the source-backed notes and explicitly distinguish architectural guarantees
from empirical learning in ordinary transformers.
The twenty reviewed references are collected in `references.bib`; screening and
additional relevant-paper review remain documented in `notes.md`.
Distinguish answer-only versus proof supervision and success-conditioned
mechanistic case studies from unconditional multi-seed reliability.
Include recurrent-model optimization as a distinct architectural comparison,
not a parameter-threshold result for our nonrecurrent task.
Separate conditional asymptotic circuit bounds from finite-depth training failures
(He et al.); neither establishes our empirical onset.
Distinguish learned cross-category analogy (Minegishi et al.) from fresh-KB
cross-format inference and universal capacity claims.
Keep Xu et al.'s stylized identity-bridge assumptions separate from our
in-context classical-entailment benchmark.

## 3. Task and dataset

Define signed literals, bounded AND/OR implications, classical entailment,
contradiction, and unknown. Specify minimum forward-rule depth and unknown
scaffold depth. Describe polarity-balanced candidate paths and distractors.
Explain canonical deduplication and each holdout boundary; state what is not
held out. Show one identical latent problem in all three representations.

Figure 1: paired representation example and dependency graph.
Completed illustrative draft: `graphs/task_illustration.png`/`.svg`, with exact
renderer inputs and independent parser/classical-solver checks in the JSON.
Label it a minimal pedagogical toy without benchmark distractors/scaffolds,
not a sampled corpus item or a model prediction.
Table 1: split counts, depth, symbol pools, structural constraints, input lengths.
Measured draft: `graphs/dataset_characteristics.md`, derived from the audited
full-corpus character counts; it also documents the unequal-length confound.

## 4. Models and experimental design

Describe causal decoder blocks, sinusoidal positions, shared character encoding,
answer-only classification, exact parameter counts (11,491; 50,771; 99,075;
457,251; 1,004,547), optimization, checkpoints,
seeds, and compute. These are transformers with a classifier, not pretrained LLMs.
Specify source-learning gate, validation-only selection, held-out testing, pilot
versus final conditions, and exposure/compute matching. Explain surface novelty
and syntax-exposure controls before interpreting transfer.

## 5. Results

5.1 Source-format 2-hop learning and capacity failures.
5.2 Full 3-by-3 source/target transfer matrix, with each diagonal as a reference.
5.3 Independent symbol-combination and held-out-name results.
5.4 Held-out diamond dependencies.
5.5 Depth 2-5 versus context-length controls.
Use exact-token-length pairs with the same gold label in each grammar. State
that the paired cores differ and that added distractor/count/graph cues are not
fully matched; this is not a pure causal intervention on proof depth.
The full 90K-pair/180K-input control corpus passed its independent audit.
The first full 100K symbolic checkpoint scores 33.33% on both sides of each
10K-pair 2-versus-3/4/5 comparison, with zero paired differences. Its failed
source gate prevents a reasoning/depth-generalization interpretation. The
measured figure/table is `graphs/baseline_full_100k_symbolic_seed1_length_control`.
5.6 Capacity scaling and training-seed variability.

Figures 2-5: learning curves, transfer heatmaps, depth curves, capacity curves.
Tables 2-3: per-condition/per-class results and paired statistical comparisons.
The final primary analysis command is `python -m models.final_analysis`.
It refuses global statistics/onsets until all 45 fixed-budget conditions are
complete, then corrects the 630 primary paired contrasts as one Holm family.
This is not a correction over calibration, warmups or every paper claim, and
the behavioral smallest-tested-size criteria are descriptive, not mechanistic.
Do not fill missing runs with estimated or synthetic measurements.
The first full-corpus 99,075-parameter seed-1 symbolic baseline is complete:
33.33% validation and IID accuracy, with a constant-class selected checkpoint.
This single condition fails the source gate and does not determine an onset.
Its full-IID interventions also retain the constant answer: 100% prediction
invariance, zero jointly correct pairs after query negation, and 0% accuracy
after supporting-fact ablation. Use this measured example to explain why
invariance must be accompanied by correctness and counterfactual sensitivity.
Relate this distinction to Zhang, Chen and Chen's matched SAT-pair evaluation;
their label-changing ADR pairs differ from our equal-label length controls.

## 6. Controls and error analysis

Oracle and superficial-feature baselines; parse-normalized versus raw transfer;
syntax exposure; renaming/order/distractor probes; ablated premises; optional
curriculum and mixed-format adaptation. Use actual error examples with IDs.
Separate evidence for syntax recognition, variable binding, and inference.
Include the measured character-exposure matrix: symbolic-to-function/English
inputs contain substantial source-unseen character occurrences despite shared
IDs. Distinguish this corpus audit from a neural syntax-exposure intervention.
Compare unlabeled multiformat warmup against a matched source-only LM warmup,
not only a no-warmup baseline. Report additional tokens, windows, updates and
record counts; call this exposure/adaptation, not zero-exposure transfer.
Both 100K warmup arms are complete for seeds 1–3, with measured exposure curves.
Their downstream matched supervised comparison is pending: LM loss is not a
source-inference or transfer result, and differs across text distributions.
The first no-warmup CPU reference has started under the fixed control budget.
LR 0.003 was chosen only after all three validation-only calibration trials
completed; none passed source mastery. Report the matched warmup-arm contrast
separately from the lower-total-compute no-warmup reference.

## 7. Discussion and limitations

Answer the question within the actual trained capacities and fragment. A failed
source-learning gate does not isolate transfer. Discuss structured template
language, shared-character rather than unseen-token novelty, limited proof
depths, statistical cues, nondeterministic accelerator kernels, and local compute.
Do not generalize findings to arbitrary propositional logic or human reasoning.

## 8. Conclusion

One sentence based on the final measurements, followed by the supported scope.
Currently pending: no final experiment conclusion is justified.

## Reproducibility appendix

Git commit, dependency/runtime versions, dataset fingerprint, configurations,
all seeds, actual parameter counts, step/token budgets, selection rule, unabridged
results, confidence-interval procedure, and figure-generation commands.
