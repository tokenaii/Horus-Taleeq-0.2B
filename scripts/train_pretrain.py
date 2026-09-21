import os, json, glob, time, math, random
from pathlib import Path
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import LlamaConfig, LlamaForCausalLM
import sentencepiece as spm

ROOT = Path(os.getenv('NOUR_ROOT', '/mnt/opet-data/slm-nour-flash'))
DATA = ROOT / 'data'
OUT = Path(os.getenv('NOUR_OUT', str(ROOT / 'checkpoints' / 'pretrain')))
TOKENIZER = ROOT / 'tokenizer' / 'nour_flash_128k.model'
RESUME_FROM = os.getenv('NOUR_RESUME_FROM', '')
SEQ = int(os.getenv('NOUR_SEQ_LEN', '2048'))
MICRO = int(os.getenv('NOUR_MICRO_BATCH', '4'))
ACCUM = int(os.getenv('NOUR_GRAD_ACCUM', '8'))
LR = float(os.getenv('NOUR_LR', '3e-4'))
MAX_TOKENS = int(os.getenv('NOUR_MAX_TOKENS', '5330000000'))
WARMUP_STEPS = int(os.getenv('NOUR_WARMUP_STEPS', '1000'))
SAVE_EVERY = int(os.getenv('NOUR_SAVE_EVERY', '250000000'))
SEED = 20260918

torch.manual_seed(SEED); random.seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

class PackedText(IterableDataset):
    def __init__(self, files, sp, seq): self.files, self.sp, self.seq = files, sp, seq
    def __iter__(self):
        buf=[]; eos=self.sp.eos_id()
        for fn in self.files:
            if fn.endswith('.parquet'):
                import pyarrow.parquet as pq
                batches = pq.ParquetFile(fn).iter_batches(batch_size=4096, columns=['text'])
                records = (t for b in batches for t in b.column('text').to_pylist())
            else:
                f = open(fn, encoding='utf-8')
                records = (json.loads(line).get('text','') for line in f)
            try:
                for text in records:
                    if not isinstance(text,str) or len(text)<20: continue
                    if not isinstance(text,str) or len(text)<20: continue
                    try: ids=self.sp.encode(text, out_type=int)
                    except Exception: continue
                    buf.extend(ids); buf.append(eos)
                    while len(buf)>=self.seq:
                        x=torch.tensor(buf[:self.seq], dtype=torch.long); del buf[:self.seq]
                        yield {'input_ids':x, 'labels':x.clone()}
            finally:
                if not fn.endswith('.parquet'):
                    f.close()

def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    OUT.mkdir(parents=True, exist_ok=True)
    sp=spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    if os.getenv('NOUR_PARQUET_ONLY','0') == '1':
        files=sorted(glob.glob(str(ROOT/'exports'/'pretraining_parquet'/'part-*.parquet')))
    elif os.getenv('NOUR_EXTRA_ONLY','0') == '1':
        files=sorted(glob.glob(str(DATA/'pretrain_extra'/'part-*.jsonl')))
    else:
        files=sorted(glob.glob(str(DATA/'pretrain_raw'/'part-*.jsonl')) + glob.glob(str(DATA/'dialect_egyptian'/'*.jsonl')))
    if not files: raise RuntimeError('No training shards found')
    cfg=LlamaConfig(vocab_size=128000, hidden_size=640, intermediate_size=2048,
        num_hidden_layers=24, num_attention_heads=10, num_key_value_heads=2,
        hidden_act='silu', max_position_embeddings=4096, rope_theta=1000000.0,
        rms_norm_eps=1e-5, attention_bias=False, mlp_bias=False,
        tie_word_embeddings=True, torch_dtype=torch.bfloat16)
    model=(LlamaForCausalLM.from_pretrained(RESUME_FROM, torch_dtype=torch.bfloat16).cuda()
           if RESUME_FROM else LlamaForCausalLM(cfg).cuda())
    model.gradient_checkpointing_enable(); model.config.use_cache=False
    opt=torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1, fused=True)
    ds=PackedText(files,sp,SEQ); loader=DataLoader(ds,batch_size=MICRO,num_workers=0)
    total=0; step=0; start=time.time(); last_save=0; model.train(); opt.zero_grad(set_to_none=True)
    print(json.dumps({'event':'training_started','files':len(files),'seq':SEQ,'micro_batch':MICRO,'grad_accum':ACCUM,'target_tokens':MAX_TOKENS,'optimizer':'AdamW'},ensure_ascii=False),flush=True)
    for batch in loader:
        ids=batch['input_ids'].cuda(non_blocking=True); labels=batch['labels'].cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(input_ids=ids,labels=labels).loss/ACCUM
        loss.backward(); total += ids.numel()
        if total // (SEQ*MICRO) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            current_step=step+1
            if current_step <= WARMUP_STEPS: scale=current_step/max(1,WARMUP_STEPS)
            else:
                p=min(1.0,max(0.0,(total-MICRO*SEQ*ACCUM*WARMUP_STEPS)/(MAX_TOKENS-MICRO*SEQ*ACCUM*WARMUP_STEPS)))
                scale=0.1+0.9*0.5*(1+math.cos(math.pi*p))
            for g in opt.param_groups: g['lr']=LR*scale
            opt.step(); opt.zero_grad(set_to_none=True); step=current_step
            if step==1 or step%10==0:
                elapsed=max(1e-6,time.time()-start); rate=total/elapsed
                print(json.dumps({'event':'step','step':step,'tokens':total,'loss':round(float(loss.item()*ACCUM),5),'lr':g['lr'],'tokens_per_sec':round(rate,1),'hours_remaining':round(max(0,(MAX_TOKENS-total)/rate/3600),2)},ensure_ascii=False),flush=True)
            if total-last_save>=SAVE_EVERY:
                p=OUT/f'tokens_{total}'; p.mkdir(exist_ok=True); model.save_pretrained(p); torch.save(opt.state_dict(),p/'optimizer.pt'); last_save=total
        if total>=MAX_TOKENS: break
    model.save_pretrained(OUT/'final'); print(json.dumps({'event':'training_complete','tokens':total,'steps':step},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
