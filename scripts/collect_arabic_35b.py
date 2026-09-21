import os, re, json, time, hashlib, sqlite3
from pathlib import Path
import sentencepiece as spm
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

ROOT=Path('/mnt/opet-data/slm-nour-flash')
OUT=Path(os.getenv('AR35_OUT',str(ROOT/'data'/'pretrain_35b_clean')))
TOK=Path(os.getenv('AR35_TOKENIZER',str(ROOT/'tokenizer'/'nour_flash_128k.model')))
TARGET=int(os.getenv('AR35_TARGET_TOKENS','35000000000'))
START=int(os.getenv('AR35_START_ROW','0'))
ROWS=int(os.getenv('AR35_ROWS_PER_SHARD','100000'))
MIN_CHARS=int(os.getenv('AR35_MIN_CHARS','120'))

AR=re.compile(r'[\u0600-\u06ff]')
QURAN=re.compile(r'القرآن|القران|سورة|سُورَة|آية|اية|بسم الله الرحمن الرحيم|الحمد لله رب العالمين')
BOILER=re.compile(r'حقوق الطبع|جميع الحقوق محفوظة|اضغط هنا|اشترك الآن|التصنيفات|الصفحة الرئيسية|تسجيل الدخول|كلمة المرور|javascript:|<\/?(?:html|script|style)|المنتدى|مشاركات اليوم',re.I)
URL=re.compile(r'https?://|www\.',re.I)

def clean_ok(text):
    if not isinstance(text,str): return False
    text=' '.join(text.split())
    if len(text)<MIN_CHARS or QURAN.search(text) or BOILER.search(text) or URL.search(text): return False
    letters=sum(c.isalpha() for c in text); arab=len(AR.findall(text))
    if letters<60 or arab/max(1,letters)<0.78: return False
    if re.search(r'(.)\1{8,}',text): return False
    # reject obvious scraped index/keyword spam
    words=text.split()
    if len(words)>=20 and len(set(words))/len(words)<0.35: return False
    return text

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(str(OUT/'dedup.sqlite3'),timeout=60)
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE IF NOT EXISTS seen (h BLOB PRIMARY KEY)'); db.commit()
    sp=spm.SentencePieceProcessor(model_file=str(TOK))
    ds=load_dataset('ClusterlabAi/101_billion_arabic_words_dataset',split='train',streaming=True)
    if START: ds=ds.skip(START)
    schema=pa.schema([('text',pa.string()),('source',pa.string()),('url',pa.string()),('token_count',pa.int32())])
    rows=[]; total=0; seen=0; accepted=0; rejected=0; shard=0; started=time.time(); writer=None
    for item in ds:
        seen+=1; text=clean_ok(item.get('text',''))
        if not text: rejected+=1; continue
        h=hashlib.sha256(text.encode()).digest()
        try: db.execute('INSERT INTO seen(h) VALUES (?)',(h,))
        except sqlite3.IntegrityError: rejected+=1; continue
        ids=sp.encode(text,out_type=int); n=len(ids)
        if n<16: rejected+=1; continue
        rows.append({'text':text,'source':'ClusterlabAi/101_billion_arabic_words_dataset','url':item.get('url',''),'token_count':n})
        total+=n; accepted+=1
        if len(rows)>=ROWS:
            if writer is None: writer=pq.ParquetWriter(str(OUT/f'part-{shard:05d}.parquet'),schema,compression='zstd')
            writer.write_table(pa.Table.from_pylist(rows,schema=schema)); rows=[]
        if accepted%10000==0: db.commit()
        if accepted%100000==0:
            if writer: writer.close(); writer=None; shard+=1
            rate=total/max(1,time.time()-started)
            print(json.dumps({'event':'progress','source_row':START+seen,'accepted':accepted,'rejected':rejected,'tokens':total,'rate':round(rate,1),'shard':shard},ensure_ascii=False),flush=True)
        if total>=TARGET: break
    if rows:
        if writer is None: writer=pq.ParquetWriter(str(OUT/f'part-{shard:05d}.parquet'),schema,compression='zstd')
        writer.write_table(pa.Table.from_pylist(rows,schema=schema))
    if writer: writer.close()
    db.commit(); db.close()
    print(json.dumps({'event':'complete','tokens':total,'accepted':accepted,'rejected':rejected,'source_rows':START+seen,'target':TARGET,'output':str(OUT)},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
