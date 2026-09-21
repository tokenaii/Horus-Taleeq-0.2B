# Data and Reproducibility Record

## Accounting

The base training ledger records 6,000,308,224 tokenizer IDs. A separate
quality-filtered Parquet export contained 1,302,502,948 IDs across 44 files. These
are different artifacts; the export must not be described as the complete corpus.

## Required manifest fields

Every reproducible run must publish:

- immutable source revision and license;
- accepted/rejected row counts and rejection reasons;
- normalization, filtering, and deduplication version;
- Quran exclusion audit result;
- tokenizer SHA-256;
- per-shard token count and SHA-256;
- total token count computed with the release tokenizer.

## Canonical dataset policy

One frozen canonical dataset must be used for a reproducible model run. A
checkpoint cannot claim full reproducibility if the training stream and published
Parquet stream differ.

## Security and licensing

Never commit access tokens, private keys, raw credentials, or unlicensed source
content. This repository contains code and documentation, not private corpus
files.
