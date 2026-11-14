"""Summarise the weight shards from their safetensors headers.

Usage: python3 docs/inspect_weights.py [folder holding the .safetensors shards]

Reads only the JSON header at the start of each shard, so it needs no ML
libraries and never loads the weights into memory. Without shards it falls
back to the committed model.safetensors.index.json.
"""
import json
import struct
import sys
from collections import Counter
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
folder = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else repo
bits = json.loads((repo / "config.json").read_text())["quantization"]["bits"]

shards = sorted(folder.glob("model-*.safetensors"))
if not shards:
    index = json.loads((repo / "model.safetensors.index.json").read_text())
    per_shard = Counter(index["weight_map"].values())
    print(f"No shards in {folder}; reading the committed index instead.")
    print(f"Tensors: {len(index['weight_map'])}")
    for name, count in sorted(per_shard.items()):
        print(f"  {name}: {count} tensors")
    print(f"Total size: {index['metadata']['total_size'] / 1e9:.2f} GB")
    sys.exit(0)

tensors = {}
for shard in shards:
    with shard.open("rb") as fh:
        (length,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(length))
    fmt = (header.pop("__metadata__", None) or {}).get("format", "not stated")
    print(f"{shard.name}: {len(header)} tensors, {shard.stat().st_size / 1e9:.2f} GB, format {fmt}")
    tensors.update(header)

dtypes = Counter(t["dtype"] for t in tensors.values())
print(f"Tensors: {len(tensors)} ({', '.join(f'{n} {d}' for d, n in sorted(dtypes.items()))})")

# A quantized layer stores its weight as packed 32-bit words (32 / bits values
# per word) plus one scale and one bias per group of values.
quantized = [n for n, t in tensors.items() if n.endswith(".weight") and t["dtype"] == "U32"]
params = 0
for name, t in tensors.items():
    if name.endswith((".scales", ".biases")):
        continue
    count = 1
    for dim in t["shape"]:
        count *= dim
    params += count * (32 // bits) if t["dtype"] == "U32" else count
print(f"Quantized layers: {len(quantized)} at {bits} bits")
print(f"Model parameters: {params:,} ({params / 1e9:.2f} billion)")
