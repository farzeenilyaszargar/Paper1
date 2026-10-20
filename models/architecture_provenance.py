"""Accept a pinned historical registry only when model computation is unchanged.

Adding a preset changes the file hash but must not invalidate earlier, identical
architectures. This is not a general source-hash bypass: only the exact historical
hash below is accepted, with its pinned computation AST and per-size registry.
"""

import ast
import hashlib
from pathlib import Path

from synthetic_dataset_generator.generator import file_checksum

HISTORICAL_SOURCE = "e218ac74f8f42324ddb80bc79328ef1acdccd7e8671562ecadcaa3adfa35e5a7"
HISTORICAL_COMPUTATION = "eea42cbf50270bca4382d64a463c549efbd2cbf16a5e340d66debe7629909e0a"
HISTORICAL_PRESETS = {
    "10k": (16, 3, 2, 64), "100k": (48, 3, 4, 224),
    "500k": (96, 4, 4, 384), "1m": (128, 5, 4, 512),
    "3m": (192, 7, 6, 768), "5m": (256, 6, 8, 1088),
    "10m": (320, 8, 8, 1280),
}


def computation_digest(source):
    tree = ast.parse(source)
    registry = {"SIZES", "PRIMARY_SIZES", "FF_WIDTH"}
    nodes = [node for node in tree.body if not (
        isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in registry for target in node.targets))]
    canonical = ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_transformer_provenance(config):
    path = Path(__file__).with_name("transformer.py")
    recorded = config["source_checksums"]["transformer.py"]
    if recorded == file_checksum(path):
        return
    # Read literal registry metadata without importing torch in CPU supervisors.
    source = path.read_text()
    registry = {node.targets[0].id:ast.literal_eval(node.value)
                for node in ast.parse(source).body
                if isinstance(node,ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0],ast.Name)
                and node.targets[0].id in ("SIZES","FF_WIDTH")}
    sizes, ff_width = registry["SIZES"], registry["FF_WIDTH"]
    size = config["size"]
    architecture = config.get("architecture", {})
    old = HISTORICAL_PRESETS.get(size)
    current = (*sizes[size], ff_width.get(size, 4*sizes[size][0]))
    saved = tuple(architecture.get(key) for key in ("width", "layers", "heads", "ff_width"))
    if (recorded != HISTORICAL_SOURCE or old is None or old != current or saved != old
            or computation_digest(source) != HISTORICAL_COMPUTATION):
        raise ValueError("Primary architecture provenance differs from pinned computation/preset")
