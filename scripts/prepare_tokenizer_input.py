"""Extract a bounded, deterministic tokenizer-training sample from raw JSONL shards."""
import glob, json, os

ROOT = r"/mnt/opet-data/slm-nour-flash"
out = os.path.join(ROOT, "tokenizer", "train_sample.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)
max_chars = 2_000_000_000
written = 0
with open(out, "w", encoding="utf-8") as dst:
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "pretrain_raw", "part-*.jsonl"))):
        with open(path, encoding="utf-8") as src:
            for line in src:
                try:
                    text = json.loads(line).get("text", "").strip()
                except Exception:
                    continue
                if len(text) < 40:
                    continue
                remaining = max_chars - written
                if remaining <= 0:
                    break
                if len(text) > remaining:
                    text = text[:remaining]
                dst.write(text.replace("\n", " ") + "\n")
                written += len(text)
        if written >= max_chars:
            break
print({"output": out, "chars": written})
