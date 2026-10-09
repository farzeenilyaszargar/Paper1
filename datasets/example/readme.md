# Illustrative records

`records.jsonl` contains the first record from each split of the seed-42 smoke
preset (generator 0.1.0). These nine records illustrate the format; this subset
is not a training corpus and is not expected to pass a full-corpus audit.

Regenerate the complete source corpus with:

```sh
.venv/bin/python -m synthetic_dataset_generator generate --config synthetic_dataset_generator/configs/smoke.json
```

Train on `input` and `target` only. `metadata` contains answers and proofs.
