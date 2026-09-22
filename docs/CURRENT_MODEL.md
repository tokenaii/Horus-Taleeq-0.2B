# Horus Taleeq 0.2 Base — Technical Record

Status: Arabic-only SLM development base. This document does not certify a
finished instruction/chat assistant.

## Architecture

| Field | Value |
| --- | ---: |
| Parameters | exactly 199,916,160 (approximately 199.92M) |
| Architecture | Llama-style decoder-only causal Transformer |
| Vocabulary | 128,000 SentencePiece tokens |
| Hidden size | 640 |
| Transformer layers | 24 |
| Attention heads | 10 |
| Key/value heads | 2 |
| Head dimension | 64 |
| MLP intermediate size | 2,048 |
| Activation | SiLU/SwiGLU-style Llama MLP |
| Normalization | RMSNorm, epsilon 1e-5 |
| RoPE theta | 1,000,000 |
| Embeddings | tied input/output |
| Biases | disabled in attention and MLP |
| Training dtype | bfloat16 |
| Context configuration | 4,096 tokens |

## Tokenizer

The base run used the Arabic-first `nour_flash_128k.model` SentencePiece
tokenizer. The role-special tokenizer used for later SFT experiments is not part
of this base repository.

## Optimizer and runtime

- Fused AdamW, betas `(0.9, 0.95)`, epsilon `1e-8`.
- Weight decay `0.1` and gradient clipping `1.0`.
- CUDA bfloat16 autocast and activation checkpointing.
- STAM was research-only and was not used for the verified ledger.

## Quality boundary

The base model can produce inconsistent or repetitive completions and is not
instruction-tuned. Arabic fluency, Egyptian dialect quality, factual QA, safety,
and multi-turn behavior require separate held-out evaluation.
