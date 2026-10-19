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
The thirteen reviewed references are collected in `references.bib`; screening and
additional relevant-paper review remain documented in `notes.md`.

## 3. Task and dataset

Define signed literals, bounded AND/OR implications, classical entailment,
contradiction, and unknown. Specify minimum forward-rule depth and unknown
scaffold depth. Describe polarity-balanced candidate paths and distractors.
Explain canonical deduplication and each holdout boundary; state what is not
held out. Show one identical latent problem in all three representations.

Figure 1: paired representation example and dependency graph.
Table 1: split counts, depth, symbol pools, structural constraints, input lengths.

## 4. Models and experimental design

Describe causal decoder blocks, sinusoidal positions, shared character encoding,
answer-only classification, exact parameter counts, optimization, checkpoints,
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
5.6 Capacity scaling and training-seed variability.

Figures 2-5: learning curves, transfer heatmaps, depth curves, capacity curves.
Tables 2-3: per-condition/per-class results and paired statistical comparisons.
Do not fill missing runs with estimated or synthetic measurements.
The first full-corpus 99,075-parameter seed-1 symbolic baseline is complete:
33.33% validation and IID accuracy, with a constant-class selected checkpoint.
This single condition fails the source gate and does not determine an onset.

## 6. Controls and error analysis

Oracle and superficial-feature baselines; parse-normalized versus raw transfer;
syntax exposure; renaming/order/distractor probes; ablated premises; optional
curriculum and mixed-format adaptation. Use actual error examples with IDs.
Separate evidence for syntax recognition, variable binding, and inference.

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
