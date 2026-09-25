"""Generates kda.py (glm5next/nvidia overlay) from kda.orig.py (image glm53-upstream:b4). See glm53_flashkda.py."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
def sub(old, new):
    global s
    assert s.count(old) == 1, old[:90]
    s = s.replace(old, new)
sub('''            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_kda_with_fused_gate(
                q=_rearr(q_ns),''', '''            if _glm53_flashkda.active():
                # [2026-09-25] FLASHKDA: prefill on the FlashKDA CUDA kernel (output written straight into core_attn_out
                # when the step has no speculative tokens -> no merge copy)
                if not use_spec:
                    ns_out = core_attn_out[:, :num_actual_tokens]
                core_attn_out_non_spec, last_recurrent_state = _glm53_flashkda.prefill(
                    _rearr(q_ns), _rearr(k_ns), _rearr(v_ns), g1_ns, beta_ns,
                    self.A_log, self.dt_bias, lower_bound, initial_state, non_spec_query_start_loc, out=ns_out,
                )
            else:
              (
                core_attn_out_non_spec,
                last_recurrent_state,
              ) = chunk_kda_with_fused_gate(
                q=_rearr(q_ns),''')
sub('\nfrom vllm.platforms import current_platform\n',
    '\nfrom vllm.platforms import current_platform\nfrom vllm.models.glm5next.nvidia import glm53_flashkda as _glm53_flashkda  # [25/09] FLASHKDA\n')
open(dst, "w").write(s); print("OK", dst)
