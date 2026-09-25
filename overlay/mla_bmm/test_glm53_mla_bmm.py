import sys, torch
sys.path.insert(0, "/w")
import glm53_mla_bmm as m
N, B, P, L, V = 32, 4608, 256, 512, 256
bf = torch.bfloat16; d = "cuda"
for B in (256, 777, 4608):
    q = torch.randn(B, N, P, device=d, dtype=bf)          # (B, N, P) like vLLM's q output
    qn = q.transpose(0, 1)
    Wk = (torch.randn(L, N, P, device=d, dtype=bf) * .05).permute(1, 2, 0)   # W_UK_T vLLM (N, P, L)
    qf = torch.full((B, N, L + 64), 7.0, device=d, dtype=bf)
    qf[..., L:].zero_()
    m.bmm(qn, Wk, qf[..., :L].transpose(0, 1))
    ref = torch.bmm(qn, Wk)                                   # cuBLAS (N, B, L)
    e1 = (qf[..., :L].transpose(0, 1).float() - ref.float()).abs().max().item()
    assert qf[..., L:].abs().max().item() == 0.0
    x = torch.randn(B, N, L, device=d, dtype=bf).transpose(0, 1)
    Wv = (torch.randn(L, N, V, device=d, dtype=bf) * .05).transpose(0, 1)
    o = torch.empty(B, N, V, device=d, dtype=bf)
    m.bmm(x, Wv, o.transpose(0, 1))
    ref2 = torch.empty(B, N, V, device=d, dtype=bf); torch.bmm(x, Wv, out=ref2.transpose(0, 1))
    e2 = (o.float() - ref2.float()).abs().max().item()
    s1 = ref.float().abs().max().item(); s2 = ref2.float().abs().max().item()
    print(f"B={B}: bmm1 err {e1:.4f} (|max| {s1:.1f})  bmm2 err {e2:.4f} (|max| {s2:.1f})")
    assert e1 < 0.02 * s1 and e2 < 0.02 * s2
print("OK")
