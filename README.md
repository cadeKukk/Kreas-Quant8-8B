# Kreas - Kreative Response Engine for Adaptive Solutions

**Kreas** is a powerful 12-billion parameter language model designed for creative problem-solving and adaptive responses across a wide range of tasks. Built on the Qwen3 architecture and optimized through 4-bit quantization, Kreas delivers exceptional performance while maintaining efficient resource usage.

## Model Overview

- **Model Type**: Qwen3 Language interpreter
- **Parameters**: ~12 billion
- **Quantization**: 8-bit precision with 64-group size
- **Context Length**: 131,072 tokens (128k context window)
- **Architecture**: Transformer with advanced features

## Key Features

### Creative Intelligence
- **Adaptive Problem Solving**: Adjusts node structure from feedback
- **Context-Aware Responses**: Leverages extensive 128k token context window

### Technical Specifications
- **Hidden Size**: 4,096 dimensions
- **Attention Heads**: 32 (with 8 key-value heads for efficiency)
- **Hidden Layers**: 36 transformer blocks
- **Activation Function**: SiLU (Swish) for improved gradient flow

### Optimization Features
- **Efficient Attention**: Grouped Query Attention (GQA) for faster inference
- **Extended Context**: YARN scaling enables 4x context extension from base 32k to 128k tokens
- **BFloat16 Precision**: Optimal balance of speed and numerical stability

## Model Configuration

```json
{
  "model_type": "qwen3",
  "num_hidden_layers": 36,
  "hidden_size": 4096,
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
- `model-00001-of-00002.safetensors` (~5GB)
- `model-00002-of-00002.safetensors` (~3.1GB) 
- `tokenizer.json` (~11MB)

Place these files in the same directory as the configuration files from this repository.

### With LM Studio
1. Ensure you have all model files (safetensors + tokenizer.json) in the directory
2. Load the model in LM Studio
3. Adjust temperature and other sampling parameters 
4. Load model into VRAM

### System Requirements
- **VRAM**: Minimum 8GB, recommended 16GB+
- **Storage**: ~6-8GB for model files
- **GPU**: NVIDIA CUDA or APPLE MLX

### Recommended Settings
- **Temperature**: 0.7-0.9 for creative tasks, 0.1-0.3 for analytical tasks
- **Top-p**: 0.8-0.95
- **Max Tokens**: 128k context token cap

The model employs several advanced architectural features:

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

Please refer to the original model's license terms and ensure compliance with applicable usage policies.

## Technical Support

For issues related to:
- Model loading or compatibility
- Performance optimization
- Integration questions

Please consult your inference platform's documentation or community forums.

---

**Version**: Quantized 4-bit edition  
**Base Architecture**: Qwen3  
**Quantization**: 8-bit with group size 64  
**Release**: Community optimized version 