#!/bin/bash
# Regenerate the SWA_TAIL overlays: on the pristine nightly file, and on top of the MAMBA_FREE overlay.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
python3 patch_swa_tail.py ../apc/single_type_kv_cache_manager.orig.py single_type_kv_cache_manager.py
python3 patch_swa_tail.py ../mamba_free/single_type_kv_cache_manager.py single_type_kv_cache_manager.mamba_free.py
python3 patch_swa_tail.py single_type_kv_cache_manager.py /dev/null >/dev/null   # idempotence
python3 -m py_compile single_type_kv_cache_manager.py single_type_kv_cache_manager.mamba_free.py
