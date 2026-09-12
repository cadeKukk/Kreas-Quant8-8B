# Kreas

Configuration files for a Qwen3-8B model with 8-bit quantization and a group size of 64. This repository does not include model weights or the tokenizer vocabulary, so a clone alone cannot generate text.

## Inspect the configuration

```sh
python3 docs/inspect_config.py
```

This demo reads the committed `config.json` and needs only Python's standard library. Its saved output is in [config-output.txt](docs/config-output.txt).

```text
Architecture: Qwen3ForCausalLM
Layers: 36
Hidden size: 4096
Attention heads: 32
Key/value heads: 8
Position limit: 131072
Quantization: 8 bits
Group size: 64
Weights: not included in this repository
```

## What the files specify

| Setting | Value |
| --- | --- |
| Model type | `qwen3` |
| Intermediate size | 12,288 |
| Vocabulary size | 151,936 |
| Activation | `silu` |
| Declared tensor dtype | `bfloat16` |
| RoPE scaling | YaRN, factor 4, original limit 32,768 |
| Tied embeddings | No |

The position limit is a configuration value. This repository has no benchmark results validating output quality at that length.

## Load the model

Obtain the matching weight shards and `tokenizer.json` separately, then place them alongside the configuration files. The original file set used two shards named `model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors`.

Use a runtime that supports the supplied weight format and quantization. The configuration files do not demonstrate compatibility with every loader. Check the original model's license before distributing the weights.

The repository has no recorded inference demo. The example above reports configuration values and does not imply a successful model run.
