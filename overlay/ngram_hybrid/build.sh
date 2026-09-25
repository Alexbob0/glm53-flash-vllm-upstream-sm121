#!/bin/bash
# Regenerates the model_runner.py overlay (GLM53_NGRAM) from the original copy extracted from image b4.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
python3 patch_model_runner.py model_runner.orig.py model_runner.py
python3 patch_model_runner.py model_runner.py /dev/null   # idempotence
python3 -m py_compile model_runner.py glm53_ngram.py
