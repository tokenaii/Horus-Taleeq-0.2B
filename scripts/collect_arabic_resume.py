import hashlib, json, os, re, unicodedata
import numpy as np
for n,t in [("long",np.int64),("ulong",np.uint64),("bool",bool),("object",object),("str",str),("int",int),("float",float)]:
    if not hasattr(np,n): setattr(np,n,t)
from datasets import load_dataset

ROOT="/mnt/opet-data/slm-nour-flash"
OUT=os.getenv("EXTRA_OUT",os.path.join(ROOT,"data","pretrain_raw"))
os.makedirs(OUT,exist_ok=True)
SKIP_ROWS=int(os.getenv("SKIP_ROWS","517585"))
target_chars=int(os.getenv("TARGET_CHARS","15_500_000_000"))
qurl=re.compile(r"(quran|tanzil|islamweb|al-eman|quran\.ksu|quran\.com|المصحف|القرآن)",re.I)
qtext=re.compile(r"(القرآن\s*(الكريم|العظيم)?|المصحف|تلاوات|التجويد|حفص\s+عن\s+عاصم)",re.I)
arabic=re.compile(r"[\u0600-\u06ff]")
bad=re.compile(r"(.)\1{12,}")
existing=[x for x in os.listdir(OUT) if x.startswith("part-") and x.endswith(".jsonl")]
idx=max([int(x[5:10]) for x in existing],default=2)+1
f=None; bcount=0; chars=0; kept=0; rejected=0; seen=set()
def open_shard():
    global f,bcount
    if f: f.close()
    f=open(os.path.join(OUT,f"part-{idx:05d}.jsonl"),"w",encoding="utf-8")
    bcount=0
open_shard()
ds=load_dataset("ClusterlabAi/101_billion_arabic_words_dataset",split="train",streaming=True)
for row_id,row in enumerate(ds):
    if row_id<SKIP_ROWS: continue
    if chars>=target_chars: break
    text=unicodedata.normalize("NFKC",str(row.get("text") or "")).strip()
    url=str(row.get("url") or "")
    if len(text)<200 or len(text)>200000 or qurl.search(url) or qtext.search(text):
        rejected+=1; continue
    text=re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"," ",text)
    text=re.sub(r"[ \t]+"," ",text)
    text=re.sub(r"\n{3,}","\n\n",text).strip()
    visible=len(re.findall(r"\S",text))
    if len(arabic.findall(text))/max(visible,1)<0.72 or bad.search(text):
        rejected+=1; continue
    h=hashlib.sha1(text.encode()).digest()
    if h in seen: continue
    if len(seen)<2_000_000: seen.add(h)
    line=json.dumps({"text":text,"source":"ClusterlabAi/101_billion_arabic_words_dataset","source_row":row_id,"url":url},ensure_ascii=False,separators=(",",":"))+"\n"
    size=len(line.encode())
    if bcount+size>1024**3:
        idx+=1; open_shard()
    f.write(line); bcount+=size; chars+=len(text); kept+=1
    if kept%10000==0: print(json.dumps({"row":row_id,"kept":kept,"chars":chars,"est_tokens":int(chars/3.2),"rejected":rejected,"shard":idx}),flush=True)
if f: f.close()
print(json.dumps({"done":True,"kept":kept,"chars":chars,"est_tokens":int(chars/3.2),"rejected":rejected,"last_shard":idx}),flush=True)
