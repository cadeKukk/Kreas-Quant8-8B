"""Print the committed configuration. No model weights or inference required."""
import json
from pathlib import Path

config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
for label, key in [
    ("Architecture", "architectures"),
    ("Layers", "num_hidden_layers"),
    ("Hidden size", "hidden_size"),
    ("Attention heads", "num_attention_heads"),
    ("Key/value heads", "num_key_value_heads"),
    ("Position limit", "max_position_embeddings"),
]:
    value = config[key]
    print(f"{label}: {', '.join(value) if isinstance(value, list) else value}")
print(f"Quantization: {config['quantization']['bits']} bits")
print(f"Group size: {config['quantization']['group_size']}")
print("Weights: not included in this repository")
