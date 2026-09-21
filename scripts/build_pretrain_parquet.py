import os, re, json, glob, hashlib, sqlite3, time
from pathlib import Path
import sentencepiece as spm
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(os.getenv('NOUR_ROOT', '/mnt/opet-data/slm-nour-flash'))
OUT = Path(os.getenv('PARQUET_OUT', str(ROOT / 'exports' / 'pretraining_parquet')))
TOK = Path(os.getenv('TOKENIZER_PATH', str(ROOT / 'tokenizer' / 'nour_flash_128k.model')))
TARGET = int(os.getenv('PARQUET_TARGET_TOKENS', '8300000000'))
ROW_SHARD = int(os.getenv('PARQUET_ROWS_PER_SHARD', '100000'))
MIN_CHARS = int(os.getenv('PARQUET_MIN_CHARS', '80'))

QURAN = re.compile(r'القرآن|القران|سورة|سُورَة|آية|اية|بسم الله الرحمن الرحيم|الحمد لله رب العالمين')
BAD = re.compile(r'(?:حقوق الطبع|جميع الحقوق محفوظة|اضغط هنا|اشترك الآن|التصنيفات|الرئيسية\s+اخبار|javascript:|<\/?(?:html|script|style))', re.I)
AR = re.compile(r'[\u0600-\u06ff]')
URL = re.compile(r'https?://|www\.', re.I)

def files():
    pats = [ROOT/'data'/'pretrain_raw'/'part-*.jsonl',
            ROOT/'data'/'dialect_egyptian'/'*.jsonl',
            ROOT/'data'/'pretrain_extra'/'part-*.jsonl']
    return sorted({str(p) for pat in pats for p in glob.glob(str(pat))})

def good(text):
    if not isinstance(text, str): return False
    text = text.strip()
    if len(text) < MIN_CHARS or QURAN.search(text) or BAD.search(text) or URL.search(text): return False
    a = len(AR.findall(text)); letters = sum(c.isalpha() for c in text)
    if letters < 40 or a / max(letters, 1) < 0.72: return False
    if re.search(r'(.)\1{8,}', text): return False
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if len(lines) >= 4 and len(set(lines)) / len(lines) < 0.7: return False
    return True

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(OUT/'dedup.sqlite3'))
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS seen (h BLOB PRIMARY KEY)')
    db.commit()
    sp = spm.SentencePieceProcessor(model_file=str(TOK))
    schema = pa.schema([('text', pa.string()), ('source', pa.string()), ('token_count', pa.int32())])
    writer = None; shard = 0; rows=[]; total=0; seen=0; kept=0; rejected=0; started=time.time()
    for fn in files():
        source = Path(fn).parent.name + '/' + Path(fn).name
        with open(fn, encoding='utf-8') as f:
            for line in f:
                seen += 1
                try: x=json.loads(line); text=x.get('text','')
                except Exception: rejected += 1; continue
                if not good(text): rejected += 1; continue
                norm=' '.join(text.split())
                h=hashlib.sha256(norm.encode('utf-8')).digest()
                try: db.execute('INSERT INTO seen(h) VALUES (?)',(h,))
                except sqlite3.IntegrityError: rejected += 1; continue
                ids=sp.encode(norm, out_type=int); n=len(ids)
                if n == 0: rejected += 1; continue
                rows.append((norm, source, n)); total += n; kept += 1
                if len(rows) >= ROW_SHARD:
                    if writer is None:
                        writer = pq.ParquetWriter(str(OUT/f'part-{shard:05d}.parquet'), schema, compression='zstd')
                    writer.write_table(pa.Table.from_pylist([{'text':a,'source':b,'token_count':c} for a,b,c in rows], schema=schema))
                    rows=[]
                if kept % 10000 == 0:
                    db.commit()
                if kept % 100000 == 0:
                    if writer: writer.close(); writer=None; shard += 1
                    rate=total/max(1,time.time()-started)
                    print(json.dumps({'event':'progress','seen':seen,'kept':kept,'rejected':rejected,'tokens':total,'tokens_per_sec':round(rate,1),'shard':shard},ensure_ascii=False),flush=True)
                if total >= TARGET: break
            if total >= TARGET: break
    if rows:
        if writer is None: writer=pq.ParquetWriter(str(OUT/f'part-{shard:05d}.parquet'), schema, compression='zstd')
        writer.write_table(pa.Table.from_pylist([{'text':a,'source':b,'token_count':c} for a,b,c in rows], schema=schema))
    if writer: writer.close()
    db.commit(); db.close()
    print(json.dumps({'event':'complete','seen':seen,'kept':kept,'rejected':rejected,'tokens':total,'target':TARGET,'output':str(OUT)},ensure_ascii=False),flush=True)

if __name__ == '__main__': main()
