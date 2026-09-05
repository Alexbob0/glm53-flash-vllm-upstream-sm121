"""Numerical validation of the main+extra (2048+128) top-k split merged by LSE, SM120 sparse-MLA (GLM).
(i) reference = prefill orchestrator (72 tokens) with 2 segments via the internal API (if supported)
(ii) specialised decode 2048 + 128 merged vs reference
(iii) fallback: 1024 + 1024 merged vs a single 2048 call -> establishes the LSE base.
Run inside the built image: docker run --rm --gpus all -v $PWD/tests:/t --entrypoint python3 <image> /t/test_split_merge.py"""
import torch, time, math, sys
import vllm._custom_ops as ops
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_with_kv_cache_mla as dec
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import _kv_scale_format_for_model
from flashinfer.mla import _core as C

dev = "cuda"
torch.manual_seed(0)
H, TOPK, MAIN, EXTRA, PB = 32, 2176, 2048, 128, 64
NT = 72
nblocks = 256
ntok = nblocks * PB
KSF = _kv_scale_format_for_model("glm5_next")
print("kv_scale_format(glm5_next) =", KSF)
kv_c = (torch.randn(ntok, 512, device=dev) * 0.5).to(torch.bfloat16)
k_pe = torch.zeros(ntok, 64, device=dev, dtype=torch.bfloat16)
cache = torch.zeros(nblocks, PB, 656, device=dev, dtype=torch.uint8)
slots = torch.arange(ntok, device=dev, dtype=torch.int64)
ops.concat_and_cache_mla(kv_c, k_pe, cache, slots, "fp8_ds_mla", torch.tensor(1.0, device=dev))
q = torch.randn(NT, H, 576, device=dev, dtype=torch.bfloat16); q[..., 512:] = 0
idx = torch.stack([torch.randperm(ntok, device=dev)[:TOPK] for _ in range(NT)]).to(torch.int32)
lens = torch.randint(900, TOPK + 1, (NT,), device=dev, dtype=torch.int32)
lens[:4] = torch.tensor([TOPK, MAIN, 1700, MAIN + 1], device=dev, dtype=torch.int32)
ar = torch.arange(TOPK, device=dev).unsqueeze(0)
idx = torch.where(ar < lens.unsqueeze(1), idx, torch.full_like(idx, -1))
ws = torch.zeros(256 * 1024 * 1024, device=dev, dtype=torch.uint8)
sm = 256 ** -0.5
KV4 = cache.view(torch.uint8).unsqueeze(1)

def call(qq, ii, ll, topk):
    out = torch.empty(qq.shape[0], 1, H, 512, device=dev, dtype=torch.bfloat16)
    lse = torch.empty(qq.shape[0], 1, H, device=dev, dtype=torch.float32)
    dec(query=qq.unsqueeze(1), kv_cache=KV4, workspace_buffer=ws,
        qk_nope_head_dim=256, kv_lora_rank=512, qk_rope_head_dim=64,
        block_tables=ii.unsqueeze(1), seq_lens=ll, max_seq_len=topk, out=out,
        bmm1_scale=sm, bmm2_scale=1.0, sparse_mla_top_k=topk, kv_scale_format=KSF,
        lse=lse, return_lse=True)
    return out.squeeze(1), lse.squeeze(1)

def split(ii, ll, main, extra):
    lm = ll.clamp(max=main)
    le_true = (ll - main).clamp(min=0, max=extra)
    ie = ii[:, main:main + extra].clone()
    empty = le_true == 0
    ie[:, 0] = ie[:, 0].masked_fill(empty, 0)
    return ii[:, :main].contiguous(), lm, ie.contiguous(), le_true.clamp(min=1), empty

def merge(o1, l1, o2, l2, empty, base):
    l2 = l2.masked_fill(empty.unsqueeze(1), float("-inf"))
    m = torch.maximum(l1, l2)
    w1 = torch.pow(base, l1 - m).unsqueeze(-1); w2 = torch.pow(base, l2 - m).unsqueeze(-1)
    return (o1.float() * w1 + o2.float() * w2) / (w1 + w2)

# (iii) base des LSE : 1024+1024 vs 2048 (decode, 64 tokens)
n = 64
ref2048, _ = call(q[:n], idx[:n, :MAIN].contiguous(), lens[:n].clamp(max=MAIN), MAIN)
im, lm, ie, le, empty = split(idx[:n, :MAIN], lens[:n].clamp(max=MAIN), 1024, 1024)
o1, l1 = call(q[:n], im, lm, 1024); o2, l2 = call(q[:n], ie, le, 1024)
print("empty extra rows:", int(empty.sum()), "| lse sample", [round(x, 3) for x in l1[0, :2].tolist()])
best = None
for base, name in ((math.e, "ln"), (2.0, "log2")):
    err = (merge(o1, l1, o2, l2, empty, base) - ref2048.float()).abs()
    rel = err.max().item() / ref2048.float().abs().max().item()
    print(f"(iii) 1024+1024 vs 2048, base={name}: max abs {err.max().item():.3e} mean {err.mean().item():.3e} rel {rel:.3e}")
    if best is None or rel < best[0]: best = (rel, base, name)
print("=> LSE base:", best[2])

# (i) ref orchestrateur a 2 segments (2048 + 128) sur 72 tokens
im, lm, ie, le, empty = split(idx, lens, MAIN, EXTRA)
ref = None
try:
    seg = [C._SparseMLASegment(indices=im.unsqueeze(1), lengths=lm),
           C._SparseMLASegment(indices=ie.unsqueeze(1), lengths=le, kv_cache=KV4)]
    out = torch.empty(NT, 1, H, 512, device=dev, dtype=torch.bfloat16)
    C._trtllm_batch_decode_sparse_mla_sm120(query=q.unsqueeze(1), kv_cache=KV4, workspace_buffer=ws,
        sparse_mla_segments=seg, out=out, sm_scale=sm, sinks=None, lse=None, return_lse=False, kv_scale_format=KSF)
    ref = out.squeeze(1)
    print("(i) orchestrateur 2 segments OK, finite:", torch.isfinite(ref.float()).all().item())
except Exception as e:
    print("(i) orchestrateur 2 segments FAIL:", type(e).__name__, str(e)[:260].replace("\n", " "))

# (ii) decode 2048 + 128 fusionne
o1, l1 = call(q[:n], im[:n], lm[:n], MAIN); o2, l2 = call(q[:n], ie[:n], le[:n], EXTRA)
mg = merge(o1, l1, o2, l2, empty[:n], best[1])
if ref is not None:
    err = (mg - ref[:n].float()).abs()
    print(f"(ii) decode 2048+128 vs ref: max abs {err.max().item():.3e} mean {err.mean().item():.3e} rel {err.max().item()/ref[:n].float().abs().max().item():.3e}")
rows = lens[:n] <= MAIN
print("(ii) rows len<=2048: merged == main-only ?", (mg[rows] - o1.float()[rows]).abs().max().item())
torch.cuda.synchronize(); t = time.time()
for _ in range(20):
    call(q[:8], im[:8], lm[:8], MAIN); call(q[:8], ie[:8], le[:8], EXTRA)
torch.cuda.synchronize(); print("2-call decode B=8: %.3f ms" % ((time.time() - t) / 20 * 1000))
torch.cuda.synchronize(); t = time.time()
for _ in range(20): call(q[:8], im[:8], lm[:8], MAIN)
torch.cuda.synchronize(); print("1-call 2048 B=8: %.3f ms" % ((time.time() - t) / 20 * 1000))
