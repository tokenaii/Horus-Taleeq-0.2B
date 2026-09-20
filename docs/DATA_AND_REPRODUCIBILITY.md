# Data and Reproducibility Record

## Current corpus accounting

The current 0.2B training ledger records 6,000,308,224 tokenizer IDs. A separate
quality-filtered Parquet export was built from the available source streams and
contains 1,302,502,948 IDs across 44 Parquet files. These are different artifacts.
The export was intentionally not uploaded as a claim of being the complete training
corpus.

Every future release must publish a manifest with:

- source name and immutable revision;
- license and permitted use;
- accepted/rejected row counts and rejection reasons;
- normalization and deduplication algorithm version;
- Quran exclusion audit result;
- tokenizer model SHA-256;
- per-shard token count and SHA-256;
- total token count computed from the release shards.

## Filtering requirements

The Arabic pre-training pipeline must apply Arabic-ratio and minimum-length checks,
exact and near-duplicate removal, boilerplate/SEO detection, URL and HTML filtering,
repeated-line detection, language identification, and a final Quran exclusion audit.
Filtering metadata must be retained instead of silently deleting evidence.

## Canonical dataset policy

There must be one canonical dataset for a model run. A checkpoint may not be called
reproducible if the training stream and the published Parquet stream differ. Before
the next public upload, create a frozen manifest, compute its token total with the
release tokenizer, and run a sample audit from every shard.

## Hugging Face release layout

The private dataset repository should use:

```text
pretraining/
  parquet/part-xxxxx.parquet
  manifest.json
  README.md
models/
  horus-taleeq-0.2b/
  horus-taleeq-4b/       # only after the 4B release gate
```

Never commit access tokens, private keys, raw credentials, or unlicensed source
content to this code repository or the dataset repository.
