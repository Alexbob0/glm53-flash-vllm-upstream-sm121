"""Fused Triton LSE merge vs the eager reference formula (same file, same inputs).
Run: docker run --rm --gpus all -v $PWD/overlay:/ov:ro -v $PWD/tests:/t:ro --entrypoint python3 <image> /t/test_fused_lse_merge.py"""
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location("mla_ov", "/ov/flashinfer_mla_sparse_sm120.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert m._HAS_TRITON, "triton missing"
torch.manual_seed(0); dev = "cuda"
worst = 0.0; mism = 0; total = 0
for N, H, D in [(1, 32, 512), (7, 32, 512), (64, 32, 512), (333, 32, 512), (1024, 32, 512)]:
    o1 = (torch.randn(N, H, D, device=dev) * 0.7).to(torch.bfloat16)
    o2 = (torch.randn(N, H, D, device=dev) * 0.7).to(torch.bfloat16)
    l1 = torch.randn(N, H, device=dev) * 3 + 10
    l2 = torch.randn(N, H, device=dev) * 3 + 8
    extra_empty = torch.rand(N, device=dev) < 0.3
    empty_rows = torch.rand(N, device=dev) < 0.1
    ref = m._eager_lse_merge(o1, l1, o2, l2, extra_empty, empty_rows, torch.bfloat16)
    got = m._fused_lse_merge(o1, l1, o2, l2, extra_empty, empty_rows)
    torch.cuda.synchronize()
    assert got.shape == ref.shape and got.dtype == ref.dtype
    d = (got.float() - ref.float()).abs()
    ulp = torch.exp2(torch.floor(torch.log2(ref.float().abs().clamp(min=1e-30))) - 7)  # exact bf16 ulp of ref
    ne = (got != ref).sum().item(); mism += ne; total += got.numel()
    worst = max(worst, d.max().item())
    assert (d <= ulp * 1.001 + 1e-9).all(), f"N={N}: max abs {d.max().item()} > 1 bf16 ulp"
    # semantics: empty rows -> exact zeros ; extra_empty rows -> exactly o1
    assert (got[empty_rows] == 0).all()
    sel = extra_empty & ~empty_rows
    assert torch.equal(got[sel], o1[sel]), "extra_empty rows must equal o1 bit-exactly"
    print(f"N={N:5d}: max abs diff {d.max().item():.3e}, bf16 mismatches {ne}/{got.numel()} ({100*ne/got.numel():.4f} %)")
# timing at the prefill shape
N, H, D = 4608, 32, 512
o1 = torch.randn(N, H, D, device=dev).to(torch.bfloat16); o2 = torch.randn(N, H, D, device=dev).to(torch.bfloat16)
l1 = torch.randn(N, H, device=dev); l2 = torch.randn(N, H, device=dev)
ee = torch.rand(N, device=dev) < 0.3; er = torch.rand(N, device=dev) < 0.05
for name, fn in (("eager", lambda: m._eager_lse_merge(o1, l1, o2, l2, ee, er, torch.bfloat16)), ("fused", lambda: m._fused_lse_merge(o1, l1, o2, l2, ee, er))):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(10): fn()
    t1.record(); torch.cuda.synchronize()
    print(f"{name}: {t0.elapsed_time(t1)/10:.2f} ms per merge at [4608,32,512]")
print(f"OK worst abs diff {worst:.3e}, total bf16 mismatches {mism}/{total}")
