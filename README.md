# Horus Taleeq 0.2 Base

Arabic-only Small Language Model (SLM) by [TokenAI](https://tokenai.llc/), owned
and developed by **Assem Sabry**.

TokenAI is a startup and nonprofit organization focused on open Arabic AI
research and model development.

- GitHub repository: [tokenaii/Horus-Taleeq-0.2B](https://github.com/tokenaii/Horus-Taleeq-0.2B)
- Hugging Face model: [tokenaii/Horus-Taleeq-0.2-base](https://huggingface.co/tokenaii/Horus-Taleeq-0.2-base)

This repository contains the training code and reproducibility record for
`Horus-Taleeq-0.2-base` only. It does not contain other model lines, chat
adapters, teacher outputs, private data, or credentials.

## Release boundary

This is a base-model development release, ready for continued pretraining,
instruction tuning, and identity tuning. It is not a finished production chat
model. The public record stops at base pretraining and its documented context and
clean-repair continuations.

The original base ledger contains **6,000,308,224 tokenizer IDs**:

- Initial pretraining: `3,616,305,152` IDs
- Raw-data continuation: `2,384,003,072` IDs

Core pretraining wall-clock time was **28 hours, 40 minutes, 20.604 seconds**.
The separate 4K context and clean-repair continuations are documented separately
and are not counted in that total.

## Model configuration

| Field | Value |
| --- | --- |
| Model type | Arabic-only SLM, decoder-only causal Transformer |
| Parameters | approximately 204.6M |
| Layers | 24 |
| Hidden size | 640 |
| Attention heads | 10 |
| Key/value heads | 2 (GQA) |
| MLP size | 2,048 |
| Vocabulary | 128,000 SentencePiece tokens |
| Context configuration | 4,096 tokens |
| Initial pretraining sequence | 2,048 tokens |
| Position encoding | RoPE, theta 1,000,000 |
| Embeddings | tied input/output |
| Training dtype | bfloat16 |

## Repository layout

```text
configs/
  horus_taleeq_0.2_base.yaml
docs/
  CURRENT_MODEL.md
  TRAINING_STAGES.md
  DATA_SOURCES.md
  DATA_AND_REPRODUCIBILITY.md
scripts/
  prepare_tokenizer_input.py
  train_tokenizer.py
  train_pretrain.py
  train_canonical_clean.py
  build_pretrain_parquet.py
  collect_arabic_35b.py
  collect_arabic_50b_multisource.py
  collect_arabic_resume.py
```

## Training stages

1. Arabic-first SentencePiece tokenizer preparation and training.
2. Initial pretraining at sequence length 2,048.
3. Raw-data continuation to the 6.0B-token base ledger.
4. Separate 4,096-token context continuation.
5. Clean-repair continuation on filtered Arabic Parquet data.

AdamW is the reproducibility baseline. STAM was investigated as a research
alternative but was not used for the verified base ledger.

## Data policy

The pipeline is Arabic-only and applies Arabic-ratio checks, exact deduplication,
boilerplate/HTML/SEO rejection, repeated-line filtering, and Quranic-content
exclusion checks. See [DATA_SOURCES.md](docs/DATA_SOURCES.md) and
[DATA_AND_REPRODUCIBILITY.md](docs/DATA_AND_REPRODUCIBILITY.md).

The historical raw streams and later cleaned Parquet export are not identical, so
this repository does not claim full corpus reproducibility from the public export.

## License

The model and repository materials are released under the [MIT License](LICENSE).
