# Horus Taleeq

Arabic-first decoder-only language models by TokenAI. This repository contains the
engineering and reproducibility documentation for the current Horus Taleeq 0.2B
development line and the proposed Horus Taleeq 4B successor.

## Current release status

The 0.2B model is still a development checkpoint, not a public production
conversation model. Its base pre-training reached **6,000,308,224 tokenizer IDs**.
The 4,096-token context continuation and a clean-repair continuation are separate
development checkpoints. SFT and identity tuning are staged but are not yet signed
off by the evaluation gate.

Read [the current model record](docs/CURRENT_MODEL.md) for exact architecture,
data accounting, tokenizer, training settings, and known limitations. The successor
plan is in [HORUS_TALEEQ_4B_PLAN.md](docs/HORUS_TALEEQ_4B_PLAN.md).

## Important data statement

The raw training corpus and the cleaned Parquet export are not currently identical.
The Parquet audit produced 1,302,502,948 tokenizer IDs after filtering, while the
model's training ledger records 6,000,308,224 IDs from the raw training streams.
No dataset or model should be described as a fully reproducible public release until
one canonical, hashed manifest is published.

## Repository layout

```text
docs/
  CURRENT_MODEL.md          Current 0.2B technical record and status
  HORUS_TALEEQ_4B_PLAN.md   Proposed 4B architecture and training plan
  DATA_AND_REPRODUCIBILITY.md  Data, licensing, filtering, and release gates
```

## License and data rights

Model and code licensing will be declared with the first public release. Every
dataset shard must carry source, license, filtering, tokenizer, and SHA-256 metadata.
Do not publish private, restricted, or unlicensed source material.
