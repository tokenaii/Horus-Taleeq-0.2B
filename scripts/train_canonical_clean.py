"""Continual pretraining on the canonical filtered Arabic corpus only.

This run is intentionally separate from the historical raw/pretraining run:
it resumes from clean-repair, reads only the collector's Parquet output, and
records the exact file manifest and token budget.
"""
import glob, hashlib, json, math, os, random, time
from pathlib import Path
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import LlamaForCausalLM
import sentencepiece as spm

ROOT=Path(os.getenv('NOUR_ROOT','/mnt/opet-data/slm-nour-flash'))
DATA_DIR=Path(os.getenv('NOUR_CANONICAL_DIR',str(ROOT/'data/pretrain_50b_clean')))
BASE=Path(os.getenv('NOUR_RESUME_FROM',str(ROOT/'checkpoints/horus-taleeq-clean-repair-v1/final')))
OUT=Path(os.getenv('NOUR_OUT',str(ROOT/'checkpoints/horus-taleeq-canonical-clean-v1')))
TOK=Path(os.getenv('NOUR_TOKENIZER',str(ROOT/'tokenizer/nour_flash_128k.model')))
SEQ=int(os.getenv('NOUR_SEQ_LEN','2048')); MICRO=int(os.getenv('NOUR_MICRO_BATCH','8')); ACCUM=int(os.getenv('NOUR_GRAD_ACCUM','4'))
LR=float(os.getenv('NOUR_LR','1e-4')); MAX_TOKENS=int(os.getenv('NOUR_MAX_TOKENS','5000000000')); WARMUP=int(os.getenv('NOUR_WARMUP_STEPS','2000')); SAVE=int(os.getenv('NOUR_SAVE_EVERY','250000000'))
SEED=20260921; torch.manual_seed(SEED); random.seed(SEED); torch.backends.cuda.matmul.allow_tf32=True; torch.set_float32_matmul_precision('high')

class Packed(IterableDataset):
    def __init__(self,files,sp,seq): self.files,self.sp,self.seq=files,sp,seq
    def __iter__(self):
        buf=[]; eos=self.sp.eos_id()
        for fn in self.files:
            import pyarrow.parquet as pq
            for b in pq.ParquetFile(fn).iter_batches(batch_size=4096,columns=['text']):
                for text in b.column('text').to_pylist():
                    if not isinstance(text,str) or len(text)<20: continue
                    try: ids=self.sp.encode(text,out_type=int)
                    except Exception: continue
                    buf.extend(ids); buf.append(eos)
                    while len(buf)>=self.seq:
                        x=torch.tensor(buf[:self.seq],dtype=torch.long); del buf[:self.seq]
                        yield {'input_ids':x,'labels':x.clone()}

def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    files=sorted(glob.glob(str(DATA_DIR/'part-*.parquet')))
    if not files: raise RuntimeError('No canonical parquet shards')
    manifest={'data_dir':str(DATA_DIR),'files':len(files),'first':files[0],'last':files[-1],'file_hash':hashlib.sha256('\n'.join(files).encode()).hexdigest(),'base':str(BASE),'seq':SEQ,'micro':MICRO,'accum':ACCUM,'lr':LR,'target_tokens':MAX_TOKENS,'optimizer':'AdamW','quran_filter':'collector_v1'}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    sp=spm.SentencePieceProcessor(model_file=str(TOK))
    model=LlamaForCausalLM.from_pretrained(str(BASE),torch_dtype=torch.bfloat16).cuda(); model.gradient_checkpointing_enable(); model.config.use_cache=False
    try: opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),eps=1e-8,weight_decay=.1,fused=True)
    except Exception: opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(.9,.95),eps=1e-8,weight_decay=.1)
    loader=DataLoader(Packed(files,sp,SEQ),batch_size=MICRO,num_workers=0,pin_memory=True)
    total=0; step=0; micro=0; last_save=0; start=time.time(); opt.zero_grad(set_to_none=True); model.train()
    print(json.dumps({'event':'canonical_pretrain_started',**manifest},ensure_ascii=False),flush=True)
    for batch in loader:
        ids=batch['input_ids'].cuda(non_blocking=True); labels=batch['labels'].cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(input_ids=ids,labels=labels).loss/ACCUM
        loss.backward(); total+=ids.numel(); micro+=1
        if micro%ACCUM==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); step+=1
            if step<=WARMUP: scale=step/WARMUP
            else:
                p=min(1.,max(0.,(total-SEQ*MICRO*ACCUM*WARMUP)/(MAX_TOKENS-SEQ*MICRO*ACCUM*WARMUP))); scale=.1+.9*.5*(1+math.cos(math.pi*p))
            for g in opt.param_groups: g['lr']=LR*scale
            opt.step(); opt.zero_grad(set_to_none=True)
            if step%25==0 or step==1:
                rate=total/max(1,time.time()-start); print(json.dumps({'event':'step','step':step,'tokens':total,'loss':round(float(loss.detach())*ACCUM,5),'lr':g['lr'],'tok_per_s':round(rate,1),'hours_remaining':round(max(0,(MAX_TOKENS-total)/rate/3600),2)},ensure_ascii=False),flush=True)
            if total-last_save>=SAVE:
                p=OUT/f'tokens_{total}'; p.mkdir(exist_ok=True); model.save_pretrained(p); torch.save(opt.state_dict(),p/'optimizer.pt'); last_save=total
        if total>=MAX_TOKENS: break
    model.save_pretrained(OUT/'final'); print(json.dumps({'event':'canonical_pretrain_complete','tokens':total,'steps':step,'output':str(OUT/'final')},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
