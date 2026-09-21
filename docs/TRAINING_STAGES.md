# Training Stages and Token Ledger

This ledger covers `Horus-Taleeq-0.2-base` only. Other model plans and chat
alignment artifacts are excluded.

## Core pretraining

| Stage | Tokens | Sequence length | Wall-clock |
| --- | ---: | ---: | ---: |
| Initial pretraining | 3,616,305,152 | 2,048 | 17:44:34.691 |
| Raw-data continuation | 2,384,003,072 | 2,048 | 10:55:45.913 |
| **Core total** | **6,000,308,224** | — | **28:40:20.604** |

Durations are measured from training-log creation to the terminal
`training_complete` event on the AWS host in UTC.

## Post-pretraining continuations

| Stage | Tokens | Sequence length | Wall-clock |
| --- | ---: | ---: | ---: |
| 4K context continuation | 100,007,936 | 4,096 | 00:29:01.616 |
| Clean-repair continuation | 400,007,168 | 4,096 | 01:59:25.735 |

The public base boundary stops before instruction/chat SFT and identity tuning.
The later canonical-clean run is separate and was not included in the public base
release.

## Reproduction

Use `scripts/` with a frozen data manifest and the exact tokenizer. The historical
ledger cannot be reproduced from the cleaned Parquet export alone because the two
streams are different artifacts.
