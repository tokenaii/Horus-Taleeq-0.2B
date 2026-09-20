# Horus Taleeq 0.2B — Current Technical Record

Status date: 2026-09-20. This is an internal development record; it is not a claim
that the current checkpoint is a finished chat model.

## Model configuration

| Field | Value |
|---|---:|
| Model family | Decoder-only Llama-style causal Transformer |
| Approximate parameters | 0.2B (tied input/output embeddings) |
| Vocabulary | 128,000 SentencePiece vocabulary |
| Hidden size | 640 |
| Transformer layers | 24 |
| Attention heads | 10 |
| Key/value heads | 2 (GQA) |
| Head dimension | 64 |
| MLP intermediate size | 2,048 |
| Activation | SwiGLU/SILU-style Llama MLP |
| Normalization | RMSNorm, epsilon 1e-5 |
| Positional encoding | RoPE, theta 1,000,000 |
| Embeddings | Tied input and LM-head embeddings |
| Attention/MLP bias | Disabled |
| Training dtype | bfloat16 |
| Current supported context | 4,096 tokens |
| Initial pre-training sequence length | 2,048 tokens |
| Context continuation | 100,007,936 tokens at sequence length 4,096 |

## Tokenizer and chat format

The tokenizer is `nour_flash_128k.model`, a 128K SentencePiece model trained for
Arabic-first text. The supervised training formatter uses explicit role markers:

```text
<system>\n...\n<user>\n...\n<assistant>\n...
```

Training labels are masked outside assistant spans. The tokenizer and role formatter
were validated on the 70,000-record SFT preparation file: zero malformed records,
128,000 vocabulary entries, and a maximum encoded conversation length of 624 tokens
before the SFT sequence cap.

## Training ledger

| Stage | Tokenizer IDs | Status |
|---|---:|---|
| Initial pre-training | 3,616,305,152 | Complete |
| Raw-data continuation | 2,384,003,072 | Complete |
| **Total base ledger** | **6,000,308,224** | Complete |
| 4K context continuation | 100,007,936 | Complete |
| Clean-repair continuation | Target 400,000,000 | In progress at record time |

The initial and raw continuation streams were not the same as the later cleaned
Parquet export. The base ledger is therefore an accounting record, not yet a fully
reproducible public data release.

## Optimizer and runtime

- Optimizer: fused AdamW, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`.
- Gradient clipping: `1.0`.
- Mixed precision: bfloat16 with CUDA autocast.
- The continuation runs used gradient accumulation and activation checkpointing.
- The AWS development instance used an RTX PRO 6000 Blackwell-class GPU with about
  98 GiB reported VRAM.

STAM was investigated as a research option but is not the optimizer used for the
current verified training ledger. AdamW remains the reproducibility baseline.

## Intended language coverage

The target is Arabic-only generation, with Egyptian Arabic as the first dialect and
additional Arabic dialects represented in later supervised data. Arabic dialect
recognition and quality are not yet certified by a held-out benchmark. The model
must not be described as multilingual or as a production chat assistant yet.

## Current limitations

1. Base generations still show web/SEO boilerplate and weak question grounding.
2. Earlier SFT experiments showed repetition and collapse; those checkpoints are
   not release candidates.
3. The cleaned Parquet export contains 1,302,502,948 tokenizer IDs, not the full
   6B training ledger, so it cannot currently reproduce the base run.
4. Identity, dialect, safety, and multi-turn quality gates remain open.

## Release gate

Do not publish a “final chat model” until the model passes held-out factual QA,
Arabic fluency, Egyptian dialect, dialect identification, multi-turn consistency,
repetition, contamination, and safety evaluations, and a canonical data manifest is
available.
