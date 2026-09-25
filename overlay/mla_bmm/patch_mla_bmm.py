"""Generates mla_attention.py (overlay) from mla_attention.orig.py (image glm53-upstream:b4). See glm53_mla_bmm.py."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
def sub(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
sub('''                if self.q_pad_num_heads is not None:
                    mqa_ql_nope = mqa_q_nope.new_empty((self.q_pad_num_heads, B, L))
                    mqa_ql_nope.resize_((N, B, L))
                else:
                    mqa_ql_nope = mqa_q_nope.new_empty((N, B, L))

                # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
                torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope)

                # Convert from (N, B, L) to (B, N, L)
                mqa_ql_nope = mqa_ql_nope.transpose(0, 1)
''', '''                _glm53_rope_pad = int(getattr(self.impl, "rope_pad", 0) or 0)
                if (
                    self.q_pad_num_heads is None
                    and mqa_q_pe.shape[-1] == 0
                    and _glm53_rope_pad > 0
                    and not (fp8_attention and self.impl.supports_quant_query_input)
                    and self.impl.dcp_world_size == 1
                    and mqa_q_nope.dtype == torch.bfloat16
                    and _glm53_mla_bmm.active(B, logger)
                ):
                    # [2026-09-25] MLA_BMM: q written straight into the padded buffer (B, N, L + pad) -> no cat, no pad
                    _glm53_q_full = mqa_q_nope.new_empty((B, N, L + _glm53_rope_pad))
                    _glm53_q_full[..., L:].zero_()
                    _glm53_mla_bmm.bmm(mqa_q_nope, W_UK_T, _glm53_q_full[..., :L].transpose(0, 1))
                    mqa_ql_nope = _glm53_q_full[..., :L]
                else:
                    _glm53_q_full = None
                    if self.q_pad_num_heads is not None:
                        mqa_ql_nope = mqa_q_nope.new_empty((self.q_pad_num_heads, B, L))
                        mqa_ql_nope.resize_((N, B, L))
                    else:
                        mqa_ql_nope = mqa_q_nope.new_empty((N, B, L))

                    # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
                    torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope)

                    # Convert from (N, B, L) to (B, N, L)
                    mqa_ql_nope = mqa_ql_nope.transpose(0, 1)
''')
sub('''            else:
                mqa_q = (mqa_ql_nope, mqa_q_pe)
''', '''            elif _glm53_q_full is not None:
                mqa_q = _glm53_q_full  # [2026-09-25] MLA_BMM: already (B, N, L + rope_pad)
            else:
                mqa_q = (mqa_ql_nope, mqa_q_pe)
''')
sub('''        else:
            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
            torch.bmm(x, self.W_UV, out=out.transpose(0, 1))
''', '''        elif (
            x.dtype == torch.bfloat16
            and self.W_UV.shape[-1] % 128 == 0
            and _glm53_mla_bmm.active(x.shape[1], logger)
        ):
            # [2026-09-25] MLA_BMM: same product, Triton kernel (cuBLAS on sm121 = slow sm80 wmma)
            _glm53_mla_bmm.bmm(x, self.W_UV, out.transpose(0, 1))
        else:
            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
            torch.bmm(x, self.W_UV, out=out.transpose(0, 1))
''')
sub('''            # Convert from (B, N, P) to (N, B, P)
            mqa_q_nope = mqa_q_nope.transpose(0, 1)
''', '''            # Convert from (B, N, P) to (N, B, P)
            mqa_q_nope = mqa_q_nope.transpose(0, 1)
            _glm53_q_full = None  # [25/09] MLA_BMM
''')
sub('\nimport torch.nn as nn\n', '\nimport torch.nn as nn\n\nfrom vllm.model_executor.layers.attention import glm53_mla_bmm as _glm53_mla_bmm  # [25/09] MLA_BMM\n')
open(dst, "w").write(s)
print("OK", dst)
