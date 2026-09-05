import torch, time
import vllm._custom_ops as ops
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_with_kv_cache_mla as dec
dev="cuda"; H,PB=32,64; nblocks=256; ntok=nblocks*PB
kv_c=(torch.randn(ntok,512,device=dev)*0.5).to(torch.bfloat16); k_pe=torch.zeros(ntok,64,device=dev,dtype=torch.bfloat16)
cache=torch.zeros(nblocks,PB,656,device=dev,dtype=torch.uint8)
ops.concat_and_cache_mla(kv_c,k_pe,cache,torch.arange(ntok,device=dev),"fp8_ds_mla",torch.tensor(1.0,device=dev))
ws=torch.zeros(256*1024*1024,device=dev,dtype=torch.uint8); KV4=cache.unsqueeze(1)
def call(NT,topk):
    q=torch.randn(NT,1,H,576,device=dev,dtype=torch.bfloat16); q[...,512:]=0
    ii=torch.randint(0,ntok,(NT,1,topk),device=dev,dtype=torch.int32); ll=torch.full((NT,),topk,device=dev,dtype=torch.int32)
    out=torch.empty(NT,1,H,512,device=dev,dtype=torch.bfloat16); lse=torch.empty(NT,1,H,device=dev,dtype=torch.float32)
    dec(query=q,kv_cache=KV4,workspace_buffer=ws,qk_nope_head_dim=256,kv_lora_rank=512,qk_rope_head_dim=64,block_tables=ii,seq_lens=ll,max_seq_len=topk,out=out,bmm1_scale=0.0625,bmm2_scale=1.0,sparse_mla_top_k=topk,kv_scale_format="arbitrary_fp32",lse=lse,return_lse=True)
    return out
for NT,topk in ((72,2048),(72,128),(72,1024),(72,512),(72,2176),(4096,2048),(4096,128)):
    try:
        torch.cuda.synchronize(); t=time.time(); call(NT,topk); torch.cuda.synchronize()
        print(f"prefill NT={NT} topk={topk}: OK {(time.time()-t)*1000:.2f} ms")
    except Exception as e:
        print(f"prefill NT={NT} topk={topk}: FAIL {str(e).splitlines()[-1][:120]}")
# decode chunking cost: 64-token slices, 2 calls each
q=torch.randn(64,1,H,576,device=dev,dtype=torch.bfloat16)
for topk in (2048,128):
    ii=torch.randint(0,ntok,(64,1,topk),device=dev,dtype=torch.int32); ll=torch.full((64,),topk,device=dev,dtype=torch.int32)
    out=torch.empty(64,1,H,512,device=dev,dtype=torch.bfloat16); lse=torch.empty(64,1,H,device=dev,dtype=torch.float32)
    f=lambda: dec(query=q,kv_cache=KV4,workspace_buffer=ws,qk_nope_head_dim=256,kv_lora_rank=512,qk_rope_head_dim=64,block_tables=ii,seq_lens=ll,max_seq_len=topk,out=out,bmm1_scale=0.0625,bmm2_scale=1.0,sparse_mla_top_k=topk,kv_scale_format="arbitrary_fp32",lse=lse,return_lse=True)
    f(); torch.cuda.synchronize(); t=time.time()
    for _ in range(20): f()
    torch.cuda.synchronize(); print(f"decode 64 tokens topk={topk}: {(time.time()-t)/20*1000:.3f} ms")
