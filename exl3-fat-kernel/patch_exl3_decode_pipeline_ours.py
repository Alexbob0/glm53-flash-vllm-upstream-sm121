#!/usr/bin/env python3
"""[18/09] Port of MiaAI's overlay/patch_exl3_decode_pipeline.py (kit ca85576, commit 2fb0dc0) onto OUR
exllamav3 fork tree (MiaAI 1.4.2 lineage: 3-parameter kernel template <t_bits, MOE_TILESIZE_N, cb> and a
[K][cb-1][N_off] instance table), instead of the 2-parameter c5d9c65 tree their script targets.
Additive only: the stock exl3_moe kernels/instances are untouched. Adds
  * quant/glm53_exl3_moe_fast_kernel.cuh : copy of exl3_moe_kernel.cuh with a 4th template param
    `bool shared_input` (skip the redundant up-input Hadamard, feed gemm_up with the gate transform);
  * quant/comp_units/glm53_exl3_moe_fast.cu : <4,256,cb1|cb2,true|false> instances compiled with
    MOE_FRAG_STAGES 1 / MOE_SH_STAGES 8 (stock 3/3);
  * host dispatch in quant/exl3_moe.cu : K == 4 && N_off == 1 && GLM53_EXL3_MOE_FAST=1 && SM 12.1;
    shared_input := gate_ptrs_suh.data_ptr() == up_ptrs_suh.data_ptr() (aliased by overlay/exl3.py);
  * bindings: glm53_fast_moe_version() == 1.
Usage: patch_exl3_decode_pipeline_ours.py <path to exllamav3/exllamav3_ext>
"""
import sys
from pathlib import Path


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected exactly one anchor: {old!r} (found {text.count(old)})')
    return text.replace(old, new, 1)


def patch(root):
    root = Path(root)
    quant = root / 'quant'
    host_path = quant / 'exl3_moe.cu'
    host = host_path.read_text()
    if 'GLM53_EXL3_MOE_FAST' in host:
        raise RuntimeError('Decode pipeline patch already present; use a clean source tree')
    kernel = (quant / 'exl3_moe_kernel.cuh').read_text()
    kernel = once(kernel, 'template<int t_bits, int MOE_TILESIZE_N, int cb>',
                  'template<int t_bits, int MOE_TILESIZE_N, int cb, bool shared_input>')
    kernel = once(kernel, 'void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)',
                  'void glm53_exl3_moe_fast_kernel(EXL3_MOE_KERNEL_ARGS)')
    redundant = '''                had_hf_r_128_inner<true, false>
                (
                    in_ptr,
                    temp_state_u + 128 * warp_idx,
                    exp_up_suh + 128 * token_off,
                    0.088388347648f
                );'''
    kernel = once(kernel, redundant,
                  '                if constexpr (!shared_input) {\n' + redundant + '\n                }')
    kernel = once(kernel,
                  'gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);',
                  'gemm_up(shared_input ? temp_state_g : temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);')
    wrapper = '''// [glm53] SM121 thin-decode fast instances: shallower register pipeline, deeper smem staging.
#include "exl3_moe_instances.cuh"
#undef MOE_FRAG_STAGES
#define MOE_FRAG_STAGES 1
#undef MOE_SH_STAGES
#define MOE_SH_STAGES 8
#include "../glm53_exl3_moe_fast_kernel.cuh"
fp_exl3_moe_kernel glm53_exl3_moe_fast_k4_n256(int cb, bool shared_input) {
    if (cb == 2)
        return shared_input ? glm53_exl3_moe_fast_kernel<4, 256, 2, true>
                            : glm53_exl3_moe_fast_kernel<4, 256, 2, false>;
    return shared_input ? glm53_exl3_moe_fast_kernel<4, 256, 1, true>
                        : glm53_exl3_moe_fast_kernel<4, 256, 1, false>;
}
'''
    helper = '''
#include <cstdlib>
#include <cstring>

fp_exl3_moe_kernel glm53_exl3_moe_fast_k4_n256(int cb, bool shared_input);

static bool glm53_fast_moe_enabled(int device) {
    static const bool requested = [] {
        const char* value = std::getenv("GLM53_EXL3_MOE_FAST");
        TORCH_CHECK(!value || !std::strcmp(value, "0") || !std::strcmp(value, "1"),
                    "GLM53_EXL3_MOE_FAST must be 0 or 1");
        return value && !std::strcmp(value, "1");
    }();
    if (!requested) return false;
    TORCH_CHECK(device >= 0 && device < MAX_DEVICES, "Unexpected device index");
    static thread_local int supported[MAX_DEVICES] = {};
    if (!supported[device]) {
        int major = 0, minor = 0;
        cuda_check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        cuda_check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
        supported[device] = major == 12 && minor == 1 ? 1 : -1;
    }
    return supported[device] == 1;
}
'''
    host = once(host, '#include <set>', '#include <set>\n' + helper)
    anchor = '    fp_exl3_moe_kernel kernel = exl3_moe_kernel_instances[4 * K + 2 * cb_idx + N_off];'
    host = once(host, anchor, anchor + '''
    if (K == 4 && N_off == 1 && glm53_fast_moe_enabled(device)) {
        kernel = glm53_exl3_moe_fast_k4_n256(
            cb_idx + 1, gate_ptrs_suh.data_ptr() == up_ptrs_suh.data_ptr());
    }
''')
    bindings_path = root / 'bindings.cpp'
    bindings = once(bindings_path.read_text(),
                    '    m.def("exl3_moe", &exl3_moe, "exl3_moe");',
                    '    m.def("exl3_moe", &exl3_moe, "exl3_moe");\n'
                    '    m.def("glm53_fast_moe_version", []() { return 1; });')
    (quant / 'glm53_exl3_moe_fast_kernel.cuh').write_text(kernel)
    (quant / 'comp_units/glm53_exl3_moe_fast.cu').write_text(wrapper)
    host_path.write_text(host)
    bindings_path.write_text(bindings)
    print(f'Installed opt-in SM121 K4/N256 decode pipeline (cb1/cb2) into {root}')


if __name__ == '__main__':
    patch(sys.argv[1])
