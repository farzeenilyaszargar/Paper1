"""Command-line entry points; no training code in the generator."""

import argparse
import os
from dataclasses import replace
from pathlib import Path

from .audit import audit
from .generator import Config, generate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="Generate and fully audit a new corpus")
    gen.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/default.json")
    gen.add_argument("--output", type=Path, help="New directory; defaults to datasets/<config name>")
    gen.add_argument("--seed", type=int)
    gen.add_argument("--train-count", type=int)
    gen.add_argument("--evaluation-count", type=int)
    gen.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="Classical-solver processes (default: up to 4)")
    check = commands.add_parser("audit", help="Independently audit an existing corpus")
    check.add_argument("directory", type=Path)
    check.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    try:
        if args.command == "generate":
            config = Config.load(args.config)
            overrides = {key: getattr(args, key) for key in ("seed", "train_count", "evaluation_count")
                         if getattr(args, key) is not None}
            config = replace(config, **overrides)
            output = args.output or Path("datasets") / config.name
            generate(config, output, workers=args.workers)
        else:
            output = args.directory
        report = audit(output, workers=args.workers)
        write_json(output / "audit.json", report)
        print(f"PASS: {report['total_examples']} examples; report: {output / 'audit.json'}")
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
