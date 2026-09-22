import os, re, json, time, hashlib, sqlite3
from pathlib import Path
import sentencepiece as spm
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

ROOT=Path('/mnt/opet-data/slm-nour-flash')
OUT=Path(os.getenv('AR50_OUT',str(ROOT/'data'/'pretrain_50b_clean')))
TOK=Path(os.getenv('AR50_TOKENIZER',str(ROOT/'tokenizer'/'nour_flash_128k.model')))
TARGET=int(os.getenv('AR50_TARGET_TOKENS','50000000000'))
ROWS=int(os.getenv('AR50_ROWS_PER_SHARD','100000'))
HF_TOKEN=os.getenv('HF_TOKEN') or os.getenv('HF_HUB_TOKEN')

AR=re.compile(r'[\u0600-\u06ff]')
QURAN=re.compile(r'القرآن|القران|سورة|سُورَة|آية|اية|بسم الله الرحمن الرحيم|الحمد لله رب العالمين')
BOILER=re.compile(r'حقوق الطبع|جميع الحقوق محفوظة|اضغط هنا|اشترك الآن|التصنيفات|الصفحة الرئيسية|تسجيل الدخول|كلمة المرور|javascript:|<\/?(?:html|script|style)|المنتدى|مشاركات اليوم',re.I)
URL=re.compile(r'https?://|www\.',re.I)

SOURCES=[
 ('ClusterlabAi/101_billion_arabic_words_dataset','default','original'),
 ('lightonai/ArabicWeb24','default','original'),
 ('oscar-corpus/mOSCAR','arb_Arab','original'),
 ('oscar-corpus/mOSCAR','arz_Arab','dialect'),
 ('oscar-corpus/mOSCAR','acm_Arab','dialect'),
 ('oscar-corpus/mOSCAR','aeb_Arab','dialect'),
 ('oscar-corpus/mOSCAR','ajp_Arab','dialect'),
 ('oscar-corpus/mOSCAR','apc_Arab','dialect'),
 ('oscar-corpus/mOSCAR','ars_Arab','dialect'),
 ('oscar-corpus/mOSCAR','ary_Arab','dialect'),
 ('TokenHaven/FineWeb-Edu-Arabic','default','translated'),
]

def to_text(v):
    if isinstance(v,str): return v
    if isinstance(v,dict): return v.get('text','') if isinstance(v.get('text',''),str) else ''
    if isinstance(v,list): return ' '.join(to_text(x) for x in v)
    return ''

def good(text):
    if not isinstance(text,str): return ''
    text=' '.join(text.split())
    if len(text)<120 or QURAN.search(text) or BOILER.search(text) or URL.search(text): return ''
    letters=sum(c.isalpha() for c in text); arab=len(AR.findall(text))
    if letters<60 or arab/max(1,letters)<0.78: return ''
    if re.search(r'(.)\1{8,}',text): return ''
    words=text.split()
    if len(words)>=20 and len(set(words))/len(words)<0.35: return ''
    return text

def extract(row):
    text=to_text(row.get('text',''))
    url=row.get('url','')
    if not isinstance(url,str): url=''
    if not url:
        meta=row.get('metadata',{})
        if isinstance(meta,dict): url=str(meta.get('url','') or '')
    return text,url

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(str(OUT/'dedup.sqlite3'),timeout=60)
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE IF NOT EXISTS seen (h BLOB PRIMARY KEY)'); db.commit()
    sp=spm.SentencePieceProcessor(model_file=str(TOK))
    schema=pa.schema([('text',pa.string()),('source',pa.string()),('source_type',pa.string()),('url',pa.string()),('token_count',pa.int32())])
    # Resume safely when the collector is restarted. The prior implementation
    # always started at part-00000 and could overwrite already collected shards.
    existing_shards=sorted(OUT.glob('part-*.parquet'))
    shard=(max((int(p.stem.split('-')[-1]) for p in existing_shards), default=-1) + 1)
    total=0
    if existing_shards:
        for p in existing_shards:
            try:
                pf=pq.ParquetFile(str(p))
                total += sum(int(x) for x in pf.read(columns=['token_count']).column('token_count').to_pylist())
            except Exception:
                pass
    accepted=0; rejected=0; started=time.time()
    print(json.dumps({'event':'resume_state','existing_shards':len(existing_shards),'existing_tokens':total,'next_shard':shard,'target':TARGET},ensure_ascii=False),flush=True)
    for repo,config,stype in SOURCES:
        print(json.dumps({'event':'source_start','repo':repo,'config':config,'source_type':stype},ensure_ascii=False),flush=True)
        try:
            kwargs={'split':'train','streaming':True}
            if config!='default': kwargs['name']=config
            if HF_TOKEN: kwargs['token']=HF_TOKEN
            ds=load_dataset(repo,**kwargs)
            for row in ds:
                text,url=extract(row); text=good(text)
                if not text: rejected+=1; continue
                h=hashlib.sha256(text.encode()).digest()
                try: db.execute('INSERT INTO seen(h) VALUES (?)',(h,))
                except sqlite3.IntegrityError: rejected+=1; continue
                n=len(sp.encode(text,out_type=int))
                if n<16: rejected+=1; continue
                rows.append({'text':text,'source':repo,'source_type':stype,'url':url,'token_count':n})
                total+=n; accepted+=1
                if len(rows)>=ROWS:
                    pq.write_table if False else None
                    fn=OUT/f'part-{shard:05d}.parquet'
                    pq.write_table(pa.Table.from_pylist(rows,schema=schema),str(fn),compression='zstd'); rows=[]; shard+=1
                if accepted%10000==0: db.commit()
                if accepted%100000==0:
                    rate=total/max(1,time.time()-started)
                    print(json.dumps({'event':'progress','accepted':accepted,'rejected':rejected,'tokens':total,'rate':round(rate,1),'shard':shard},ensure_ascii=False),flush=True)
                if total>=TARGET: break
            if total>=TARGET: break
        except Exception as e:
            print(json.dumps({'event':'source_error','repo':repo,'config':config,'error':repr(e)},ensure_ascii=False),flush=True)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows,schema=schema),str(OUT/f'part-{shard:05d}.parquet'),compression='zstd'); shard+=1
    db.commit(); db.close()
    print(json.dumps({'event':'complete','tokens':total,'accepted':accepted,'rejected':rejected,'shards':shard,'target':TARGET,'output':str(OUT)},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
