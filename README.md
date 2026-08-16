# Kreas - Kreative Response Engine for Adaptive Solutions

**Kreas** is an ~8.2-billion parameter language model designed for creative problem-solving and adaptive responses across a wide range of tasks. It is built directly on the **Qwen3-8B architecture** and quantized to **8-bit precision (group size 64)**, delivering strong performance while maintaining efficient resource usage.

> **Note**: This repository was previously named `Kreas-Quant4-12B`, which misstated the quantization and parameter count. It has been renamed to `Kreas-Quant8-8B` to match the actual configuration: 8-bit quantized weights on a ~8.2 billion parameter architecture. All specifications below reflect the real configuration shipped in `config.json`.

## Model Overview

- **Model Type**: Causal language model (`qwen3`)
- **Base Architecture**: Qwen3-8B
- **Parameters**: ~8.2 billion
- **Quantization**: 8-bit precision with group size 64
- **Context Length**: 131,072 tokens (128k context window)

## Key Features

### Creative Intelligence
- **Adaptive Problem Solving**: Adjusts node structure from feedback
- **Context-Aware Responses**: Leverages extensive 128k token context window

### Technical Specifications
- **Hidden Size**: 4,096 dimensions
- **Intermediate (MLP) Size**: 12,288 dimensions
- **Attention Heads**: 32 (with 8 key-value heads for efficiency)
- **Hidden Layers**: 36 transformer blocks
- **Vocabulary Size**: 151,936 tokens
- **Activation Function**: SiLU (Swish) for improved gradient flow

These dimensions are identical to Qwen3-8B and work out to approximately 8.2 billion parameters (embeddings and LM head are untied).

### Optimization Features
- **Efficient Attention**: Grouped Query Attention (GQA) for faster inference
- **Extended Context**: YARN scaling enables 4x context extension from base 32k to 128k tokens
- **BFloat16 Precision**: Optimal balance of speed and numerical stability

## Model Configuration

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "num_hidden_layers": 36,
  "hidden_size": 4096,
  "intermediate_size": 12288,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "max_position_embeddings": 131072,
  "vocab_size": 151936,
  "quantization": {
    "bits": 8,
    "group_size": 64
  }
}
```

## Installation & Usage

### Getting the Model Files
**⚠️ Important**: This repository contains only the configuration files. The large model weights are not included to avoid GitHub storage limits.

To use Kreas, you need to obtain the model files separately:
- `model-00001-of-00002.safetensors` (~5.3GB)
- `model-00002-of-00002.safetensors` (~3.4GB)
- `tokenizer.json` (~11MB)

Place these files in the same directory as the configuration files from this repository.

### With LM Studio
1. Ensure you have all model files (safetensors + tokenizer.json) in the directory
2. Load the model in LM Studio
3. Adjust temperature and other sampling parameters
4. Load model into VRAM

### System Requirements
- **VRAM**: Minimum 12GB, recommended 16GB+ (the 8-bit weights alone are ~8.7GB)
- **Storage**: ~9GB for model files
- **GPU**: NVIDIA CUDA or APPLE MLX

### Recommended Settings
- **Temperature**: 0.7-0.9 for creative tasks, 0.1-0.3 for analytical tasks
- **Top-p**: 0.8-0.95
- **Max Tokens**: 128k context token cap

The model employs several advanced architectural features inherited from Qwen3:

- **YARN RoPE Scaling**: Enables extrapolation to longer sequences while maintaining quality
- **Grouped Query Attention**: Reduces memory bandwidth requirements during inference
- **SiLU Activation**: Provides smooth, non-monotonic activation for better gradient flow
- **RMSNorm**: Efficient normalization with epsilon of 1e-06

## Limitations

- Generated content should be verified for factual accuracy
- May occasionally produce inconsistent responses for highly specialized domains
- Performance degrades with extremely long context (approaching 128k limit)
- Quantization may introduce minor quality trade-offs compared to full precision

## License & Usage

This model is derived from the Qwen3-8B architecture. Please refer to the original Qwen3 license terms and ensure compliance with applicable usage policies.

## Technical Support

For issues related to:
- Model loading or compatibility
- Performance optimization
- Integration questions

Please consult your inference platform's documentation or community forums.

---

**Version**: Quantized 8-bit edition (group size 64)  
**Base Architecture**: Qwen3-8B  
**Parameters**: ~8.2 billion  
**Release**: Community optimized version
