import os
import sentencepiece as spm

root = "/mnt/opet-data/slm-nour-flash"
inp = os.path.join(root, "tokenizer", "train_sample.txt")
prefix = os.path.join(root, "tokenizer", "nour_flash_128k")
spm.SentencePieceTrainer.Train(
    input=inp,
    model_prefix=prefix,
    vocab_size=128000,
    model_type="unigram",
    character_coverage=0.9999,
    byte_fallback=True,
    normalization_rule_name="identity",
    add_dummy_prefix=False,
    remove_extra_whitespaces=False,
    max_sentence_length=16384,
    input_sentence_size=20_000_000,
    shuffle_input_sentence=True,
    user_defined_symbols=["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
    bos_id=1,
    eos_id=2,
    unk_id=0,
    pad_id=3,
)
print({"model": prefix + ".model", "vocab": prefix + ".vocab"})
