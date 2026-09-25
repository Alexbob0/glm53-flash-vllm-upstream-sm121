#!/bin/bash
# Regenerate the MAMBA_FREE overlay from the pristine nightly sources (*.orig.py).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
cp single_type_kv_cache_manager.orig.py single_type_kv_cache_manager.py
cp kv_cache_interface.orig.py kv_cache_interface.py
GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY=$PWD/single_type_kv_cache_manager.py \
GLM53_KV_CACHE_INTERFACE_PY=$PWD/kv_cache_interface.py python3 patch_mamba_free_nightly.py
# idempotence
GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY=$PWD/single_type_kv_cache_manager.py \
GLM53_KV_CACHE_INTERFACE_PY=$PWD/kv_cache_interface.py python3 patch_mamba_free_nightly.py >/dev/null
