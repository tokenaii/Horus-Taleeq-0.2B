# Arabic Data Sources and Filtering

The base pipeline is Arabic-only and includes Quranic-content exclusion checks.
Each run must retain source, license, revision, filtering, and token-count
metadata. Raw source content is not redistributed in this code repository.

## Sources used or evaluated

| Source | Role | License / notes |
| --- | --- | --- |
| [ClusterlabAi/101_billion_arabic_words_dataset](https://huggingface.co/datasets/ClusterlabAi/101_billion_arabic_words_dataset) | large Arabic web reservoir | Apache-2.0 metadata; strict filtering required |
| [lightonai/ArabicWeb24](https://huggingface.co/datasets/lightonai/ArabicWeb24) | cleaned Arabic web source | ODC-BY; access conditions apply |
| [mOSCAR](https://oscar-project.github.io/documentation/versions/mOSCAR/) | MSA and dialectal reservoir | terms checked per configuration |
| [TokenHaven/FineWeb-Edu-Arabic](https://huggingface.co/datasets/TokenHaven/FineWeb-Edu-Arabic) | translated educational data | tagged as translated and audited separately |

## Filtering policy

- Arabic character-ratio and minimum-length checks.
- Exact SHA-256 deduplication within a run.
- Boilerplate, SEO, HTML, URL, and repeated-line rejection.
- Low-vocabulary-repetition rejection.
- Quranic site, recitation, surah, mushaf, and related phrase filtering.
- Final token counting with the run tokenizer.

Rejected-row counts and reason codes must be preserved. A dataset is not called
clean or Quran-free solely because an upstream card says so; the final manifest and
audit are authoritative.
