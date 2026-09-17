#!/usr/bin/env bash
# Build artifacts only; never select a profile or modify a running service.
# The recipe image has nvcc but not git: verify/archive the pin on the host,
# then run this script in the container against the extracted tree.
set -euo pipefail

upstream_checkout=${1:?Usage: build.sh EXLLAMAV3_CHECKOUT EMPTY_OUTPUT_DIRECTORY}
output_dir=${2:?Usage: build.sh EXLLAMAV3_CHECKOUT EMPTY_OUTPUT_DIRECTORY}
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Header archive matching the specialized kernel.cuh. Kernel origin is
# turboderp-org/exllamav3@58d4d732 (two-stage cooperative MoE, MIT).
upstream_pin=02aef45cd681b960a00afcd0749a4ab99e6c1bfe

mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
test -z "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" || {
  echo 'Refusing a nonempty build output directory.' >&2
  exit 1
}

mkdir -- "$output_dir/upstream"
if command -v git >/dev/null 2>&1 && [ -e "$upstream_checkout/.git" ]; then
  test "$(git -C "$upstream_checkout" rev-parse "$upstream_pin^{commit}")" = "$upstream_pin"
  git -C "$upstream_checkout" archive "$upstream_pin" exllamav3/exllamav3_ext |
    tar -x -C "$output_dir/upstream"
else
  test -d "$upstream_checkout/exllamav3/exllamav3_ext"
  if [ -f "$upstream_checkout/.glm53_coop_upstream_pin" ]; then
    test "$(cat "$upstream_checkout/.glm53_coop_upstream_pin")" = "$upstream_pin"
  fi
  cp -a -- "$upstream_checkout/exllamav3" "$output_dir/upstream/exllamav3"
fi

cp -- "$source_dir/native/cooperative_moe.cu" "$output_dir/glm53_coop.cu"
cp -- "$source_dir/native/cooperative_moe_kernel.cuh" \
  "$output_dir/upstream/exllamav3/exllamav3_ext/quant/glm53_coop_kernel.cuh"
cp -- "$source_dir/native/exl3_moe_coop.cuh" \
  "$output_dir/upstream/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cuh"
cp -- "$source_dir/runtime.py" "$output_dir/runtime.py"

"${NVCC:-/usr/local/cuda/bin/nvcc}" -std=c++17 -O3 --use_fast_math -lineinfo --expt-relaxed-constexpr \
  -gencode arch=compute_121a,code=sm_121a -shared -Xcompiler -fPIC --ptxas-options=-v \
  -I "$output_dir/upstream/exllamav3/exllamav3_ext" \
  "$output_dir/glm53_coop.cu" -o "$output_dir/glm53-coop.so" \
  > "$output_dir/cooperative_moe-build.log" 2>&1
mv -- "$output_dir/glm53-coop.so" "$output_dir/cooperative_moe.so"
sha256sum "$output_dir/cooperative_moe.so" "$output_dir/runtime.py"
