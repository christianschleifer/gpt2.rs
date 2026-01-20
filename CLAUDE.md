# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gpt2-rs is a pure Rust implementation of GPT-2 inference. The project implements BPE tokenization, model loading from HuggingFace
SafeTensors format, and a complete inference pipeline with multi-head attention and KV caching.

## Build Commands

```bash
cargo build                        # Debug build
cargo build --release              # Release build
cargo run --bin gpt2               # Run inference binary
cargo test --bin gpt2              # Run all tests
cargo test --bin gpt2 -- --nocapture  # Run tests with output
cargo test --bin gpt2 tokenize     # Run tests matching pattern
```

## Architecture

All code lives in a single binary: `src/bin/gpt2.rs`

**Components (in source order):**

1. **Tokenizer** (~170 lines) - GPT-2 BPE tokenization
   - Loads vocabulary and merge rules from `resources/tokenizer.json`
   - `tokenize()` converts text to token IDs
   - `detokenize()` converts tokens back to text
   - `bpe()` implements the byte-pair encoding algorithm

2. **SafeTensors** (~60 lines) - Model file parsing
   - Parses HuggingFace SafeTensors binary format
   - Extracts tensors by name with shape metadata

3. **Tensor2D** (~140 lines) - Matrix operations
   - 2D tensor with row/column access
   - `mat_mul()` - matrix multiplication
   - `standardize()` - layer normalization (mean=0, std=1)
   - `soft_max()` - softmax activation
   - `gelu()` - GELU activation function
   - `transpose()` - matrix transpose
   - Operator overloads: `+=`, `*=`, `/=`

4. **TransformerBlock** (~115 lines) - Single transformer layer
   - Layer norm, multi-head attention, and MLP weights
   - `run()` - full forward pass with KV caching for efficient autoregressive generation
   - Supports 12 attention heads (64-dim per head)

5. **LargeLanguageModel** (~120 lines) - Main inference engine
   - Loads model from SafeTensors file via `TryFrom<&Path>`
   - `initialize()` processes initial context tokens
   - `sample_one()` generates next token using greedy decoding (argmax)

## Resources

Model files in `resources/` (gitignored except tokenizer.json):

- `tokenizer.json` - GPT-2 vocabulary (50,257 tokens) from HuggingFace
- `model.safetensors` - 124M parameter GPT-2 weights (548 MB)
- `header.json` - SafeTensors metadata

Download from: https://huggingface.co/openai-community/gpt2/

## Implementation Status

- ✅ Tokenization with BPE algorithm
- ✅ SafeTensors model loading
- ✅ Tensor operations (mat_mul, standardize, soft_max, gelu, transpose)
- ✅ Multi-head attention with KV caching
- ✅ Transformer forward pass (layer norm → attention → MLP)
- ✅ Token sampling (greedy/argmax decoding)

16 passing tests covering tokenization, detokenization, and tensor operations.
