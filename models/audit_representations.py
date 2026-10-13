"""Verify text-only round trips and canonical normalization across the corpus."""

import argparse
import json
from pathlib import Path

from synthetic_dataset_generator.generator import file_checksum, write_json
from models.parsing import normalize, parse
from models.representations import REPRESENTATIONS, format_input


def audit(data, output):
    manifest = json.loads((data / "manifest.json").read_text())
    report = {"passed": True, "scope": "text parser round-trip, not neural transfer accuracy",
              "dataset_manifest_sha256": file_checksum(data / "manifest.json"), "splits": {}}
    for filename, info in manifest["files"].items():
        path = data / filename
        if file_checksum(path) != info["sha256"]:
            raise ValueError("Data checksum mismatch")
        count = 0
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                expected = {k: record["metadata"][k] for k in ("facts", "rules", "query")}
                for representation in REPRESENTATIONS:
                    text = format_input(record, representation)
                    if parse(text, representation)["metadata"] != expected:
                        raise ValueError(f"Non-lossless rendering: {filename}/{record['id']}/{representation}")
                    for target in REPRESENTATIONS:
                        if normalize(text, representation, target) != format_input(record, target):
                            raise ValueError("Normalization changes the underlying model input")
                count += 1
        if count != info["count"]:
            raise ValueError("Record count mismatch")
        report["splits"][filename.removesuffix(".jsonl")] = {"records": count, "parsed_forms": 3*count,
                                                           "pairwise_normalizations": 9*count}
        print(f"Passed {filename}: {count} records; all three grammars", flush=True)
    report["total_records"] = sum(v["records"] for v in report["splits"].values())
    write_json(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/default"))
    parser.add_argument("--output", type=Path, default=Path("models/representation_audit.json"))
    args = parser.parse_args()
    audit(args.data, args.output)
