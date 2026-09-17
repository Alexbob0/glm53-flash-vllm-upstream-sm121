# Port notes — cooperative MoE on the upstream-nightly stack

This directory is MiaAI Lab's `extensions/cooperative_moe/` (TP2 geometry, AGPL-3.0-or-later)
ported onto **our** stack (`vllm/vllm-openai:nightly` + `overlay/exl3.py` + E3 grouped prefill),
not onto their fork image. The adapter interface matched `overlay/exl3.py` as-is: the layer carries
`_exl3_bits`/`_exl3_k`, `_exl3_inners`, `_exl3_hidden_size`, `_exl3_intermediate_local`,
`_exl3_ptrs`, `_exl3_fused_temps`, and `apply_exl3_experts` calls the module-global
`apply_exl3_fused_moe`, which is what `install()` wraps.

## Repin (toolchain)

The upstream-published `.so` digest is `aa3fe5e9…`. A clean rebuild here with this image's
`nvcc 13.0.88` gives a different binary, so the digest was repinned to **our** build:

| artifact | digest |
|---|---|
| `cooperative_moe.so` (this stack) | `bfc4cfae36ae147010566b6cdb5db48b153dc0cbf1c13403b23f07b0bf6cb0de` |
| `runtime.py` (upstream C1 adapter + repinned `.so` constant) | `6dc835d41575f17db453ced041cdff02701d6b2e68a77f474fad21f90aecaf2f` |
| `overlay/exl3.py` (stock source we generate from) | `9aea5ecedc05ec1ec892f1a65b36283bec59c9dc9739e3f6b932291200309506` |

`runtime.py` and `prepare_profile.py` pin these. An unvalidated binary is still refused.

## Build (host + image, no GPU)

```sh
# 1. archive the pinned exllamav3 headers on the host
git clone --filter=blob:none --no-checkout https://github.com/turboderp-org/exllamav3.git /tmp/ev3
git -C /tmp/ev3 checkout --detach 02aef45cd681b960a00afcd0749a4ab99e6c1bfe
mkdir -p /tmp/coop/{staged,build}
git -C /tmp/ev3 archive HEAD exllamav3/exllamav3_ext | tar -x -C /tmp/coop/staged
printf '%s\n' 02aef45cd681b960a00afcd0749a4ab99e6c1bfe > /tmp/coop/staged/.glm53_coop_upstream_pin

# 2. compile in the recipe image (nvcc, no GPU; /work must be empty)
docker run --rm --network none --cpus 4 --memory 8g --memory-swap 8g --user "$(id -u):$(id -g)" \
  -v "$PWD/extensions/cooperative_moe:/src:ro" -v /tmp/coop/staged:/upstream:ro \
  -v /tmp/coop/build:/work --entrypoint bash glm53-upstream:latest \
  /src/build.sh /upstream /work

# 3. generate the opt-in overlay from our stock exl3.py (absolute container runtime dir)
mkdir -p /tmp/coop/artifacts
cp /tmp/coop/build/cooperative_moe.so extensions/cooperative_moe/runtime.py /tmp/coop/artifacts/
python3 extensions/cooperative_moe/prepare_profile.py \
  --stock overlay/exl3.py --artifacts /tmp/coop/artifacts \
  --runtime-directory /root/.cache/vllm/cooperative_moe \
  --output /tmp/coop/artifacts/exl3-cooperative.py
```

## Deploy (both ranks, identical files)

The overlay footer `run_path`s `/root/.cache/vllm/cooperative_moe/runtime.py`, which is inside the
launcher's JIT-cache mount (`JIT_CACHE=1`). Stage the adapter and the library there and the generated
overlay somewhere both nodes can bind-mount:

```sh
CACHE=$HOME/glm53-upstream-cache/<tag>
mkdir -p "$CACHE/vllm/cooperative_moe" "$CACHE/coop"
install -m 644 /tmp/coop/artifacts/cooperative_moe.so /tmp/coop/artifacts/runtime.py "$CACHE/vllm/cooperative_moe/"
install -m 644 /tmp/coop/artifacts/exl3-cooperative.py "$CACHE/coop/"
# copy the same three files to the worker at the same paths
```

Then select it on both ranks:

```sh
EXL3_OVERLAY_HOST=$HOME/glm53-upstream-cache/<tag>/coop/exl3-cooperative.py \
GLM53_COOP_GEOMETRY=1 ADAPTIVE_K=1 ./supervise.sh    # or ./run.sh head|worker by hand
```

`supervise.sh` forwards `EXL3_OVERLAY_HOST` and `GLM53_COOP_GEOMETRY`, so a supervised restart keeps
the overlay. `EXL3_OVERLAY_HOST` empty = the baked plugin (default).

## Validation status

- Host-side: `test_dispatch.py` (57 checks), `test_profile.py` (8), `python3 -O test_optimized_init.py`
  (`native_init_calls=['cdll','abi','info']` — the optimized-`assert` bug is fixed in this revision).
- Pre-flight import in the recipe image: `_glm53_coop_installed=True`, geometry 1 (A-wide/B-wide).
- **Pending: the packaged GPU gate** (`test_cuda_integration.py`, needs an idle GPU) and a live boot
  with the diagnostic that `eligible` layers > 0 and cooperative is selected at decode rows.
