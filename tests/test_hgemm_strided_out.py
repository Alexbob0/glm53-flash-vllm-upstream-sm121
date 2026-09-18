"""exllamav3 ext.hgemm into a column view of a wider fp16 buffer (row stride != n) vs torch.matmul.
This is what the DENSE_NOCAT path relies on. Run in the image with --gpus all."""
import torch, exllamav3_ext as ext
dev = "cuda"; torch.manual_seed(0)
ok = True
for rows, k, ns in [(1024, 4096, [512, 4096, 1024]), (4608, 4096, [2048, 2048]), (1500, 2048, [128, 640])]:
    x = (torch.randn(rows, k, device=dev) * 0.1).to(torch.half)
    ws = [(torch.randn(k, n, device=dev) * 0.05).to(torch.half) for n in ns]
    y = torch.empty((rows, sum(ns)), dtype=torch.half, device=dev)
    yc = torch.empty_like(y)
    col = 0
    for w, n in zip(ws, ns):
        ext.hgemm(x, w, y[:, col:col + n])          # strided output view
        yc_i = torch.empty((rows, n), dtype=torch.half, device=dev)
        ext.hgemm(x, w, yc_i)                        # contiguous output (upstream default)
        yc[:, col:col + n] = yc_i
        col += n
    torch.cuda.synchronize()
    ref = torch.cat([x.float() @ w.float() for w in ws], dim=1)
    same = torch.equal(y, yc)
    err = (y.float() - ref).abs().max().item(); rel = err / ref.abs().max().item()
    print(f"rows={rows} k={k} ns={ns}: strided==contiguous {same}, max rel err vs fp32 {rel:.2e}")
    ok &= same and rel < 1e-2
print("OK" if ok else "FAIL")
