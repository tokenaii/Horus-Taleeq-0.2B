# Horus Taleeq 4B — Architecture and Training Plan

## Objective

Build a strong Arabic-first, decoder-only causal language model with Egyptian-first
dialect coverage, broad Arabic general knowledge, 4K training context followed by an
8K extension, and a separate supervised conversation/identity stage. The current
0.2B model and its audited data are inputs to the new run, not a substitute for the
new run's quality gates.

## Proposed architecture

The first configuration target is approximately 4B parameters:

| Field | Proposed value |
|---|---:|
| Layers | 40 |
| Hidden size | 3,072 |
| Attention heads | 24 |
| Key/value heads | 8 (GQA) |
| Head dimension | 128 |
| MLP intermediate size | 8,192 |
| Vocabulary | 128,000 initially; freeze after tokenizer audit |
| Activation | SwiGLU |
| Normalization | RMSNorm, epsilon 1e-5 |
| Position encoding | RoPE, theta 1,000,000 |
| Embeddings | Tied input/output embeddings unless validation favors untied heads |
| Initial context | 4,096 |
| Later context | 8,192 after a dedicated continuation |
| Precision | bfloat16 |

This configuration is a sizing target, not a final parameter count. Run a parameter
counter and a one-batch forward/backward smoke test before allocating a long run.
The GQA divisibility constraint must be checked (`24 % 8 == 0`).

## Data target and feasibility

The requested target is **100B Arabic tokenizer IDs**. It is not safe to promise that
100B clean, licensed, non-Quran Arabic tokens already exist. A realistic target is:

- carry forward the verified, available 0.2B corpus only through a frozen manifest;
- acquire and audit roughly 94B additional IDs from licensed or permissioned sources;
- expect substantially more than 100B raw IDs before language filtering, deduplication,
  boilerplate removal, and the Quran exclusion audit;
- stop the run if the accepted, licensed corpus cannot reach the target rather than
  padding it with low-quality repeated web text.

Candidate source families must be individually licensed and audited: Arabic
encyclopedic and educational text, public-domain material, scientific and technical
writing, news with permitted redistribution, Arabic web text after aggressive
filtering, and dialectal conversational text with explicit rights. Social and forum
data require special privacy, license, and personal-data review.

## Suggested 100B mixture

The mixture is a planning budget, not permission to scrape or publish any source:

| Family | Budget |
|---|---:|
| High-quality Arabic web and reference text | 30B |
| Educational, encyclopedic, and public-domain text | 20B |
| Scientific, mathematical, and technical Arabic | 15B |
| Licensed news and current-affairs archives | 10B |
| Dialectal Arabic, Egyptian-first | 10B |
| High-quality public discussion and Q&A | 5B |
| Existing verified project data | 6B |
| Reserved quality buffer | 4B |
| **Total** | **100B** |

Conversation templates, teacher distillation, identity examples, and preference data
belong in later supervised stages and must not dominate base pre-training.

## Training stages

### 0. Tokenizer and data freeze

Audit the existing tokenizer, benchmark alternatives, freeze one tokenizer, and
publish a manifest before model training. Enforce source licenses, exact dedup,
language identification, boilerplate filtering, and Quran exclusion.

### 1. Base pre-training (4K)

Train from scratch on the accepted 100B-token mixture. Use packed sequences, Flash
Attention, FSDP/ZeRO-3, activation checkpointing, gradient clipping, and frequent
durable checkpoints. AdamW is the reproducibility baseline; STAM may be evaluated in
a short matched pilot, but it should not replace the baseline without loss, speed,
and stability evidence.

### 2. 8K context continuation

Extend context only after the 4K base passes held-out language and factual tests.
Use a low learning rate, long-context packed samples, and a separate checkpoint.
Verify perplexity and long-document retrieval before proceeding.

### 3. Conversation SFT

Use a held-out, quality-gated Arabic conversation set with explicit role markers and
assistant-only loss masking. Balance Egyptian, Modern Standard Arabic, Levantine,
Gulf, Maghrebi, Iraqi, Sudanese, and Yemeni varieties. Include identity only in a
small, varied slice to avoid template overfitting.

### 4. Preference and safety tuning

Use DPO or an equivalent preference method only after SFT evaluation identifies a
measurable need. Add refusal, privacy, harassment, and dialect-sensitive safety
tests. Profanity may be represented for linguistic understanding, but it must not
become an unconditional response style.

### 5. Release evaluation

Require Arabic factual QA, mathematics/science, Egyptian and other dialect tests,
dialect identification, multi-turn consistency, repetition, contamination, safety,
latency, memory, and 4K/8K context tests. Publish the evaluation version and data
manifest with the model.

## Compute and systems

A 4B model over 100B tokens is not a practical single-GPU run. Plan a multi-GPU
cluster with FSDP or ZeRO-3, fast local NVMe sharding, a streaming data loader, and
checkpoint storage independent of the training volume. The current AWS development
GPU is suitable for smoke tests and small pilots, not for an economical 100B-token
4B production run. Produce a measured tokens/second benchmark on the selected
cluster before committing to a schedule or cost estimate.

## Definition of done

The 4B project starts only after Horus Taleeq 0.2B is a tested, published chat model
and its canonical data manifest is frozen. The 4B model is complete only when the
100B accepted-token ledger, 4K base checkpoint, 8K continuation, SFT/identity model,
evaluation report, and reproducible release artifacts all exist.
