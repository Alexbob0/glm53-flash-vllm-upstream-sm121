# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


# main/extra split of the kpool top-k buffer (see forward_mqa).
_SPLIT_MAIN_TOPK = 2048
_SPLIT_DECODE_CHUNK = 64


def _kv_scale_format_for_model(model_type: str | None) -> str:
    if model_type is not None and model_type.startswith("glm"):
        return "arbitrary_fp32"
    return "pow2_fp32"


class FlashInferMLASparseSM120Impl(MLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

    is_sparse = True
    supports_dense_mha_prefill = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        if self.kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        self.rope_pad = 0
        if self.qk_rope_head_dim == 0:
            if self.kv_lora_rank != 512:
                raise NotImplementedError(
                    "FLASHINFER_MLA_SPARSE_SM120 pads NoPE MLA into the "
                    "576-wide GLM_NSA geometry, which requires "
                    f"kv_lora_rank=512; got {self.kv_lora_rank}."
                )
            self.rope_pad = 64
        self.kernel_qk_rope_head_dim = self.qk_rope_head_dim + self.rope_pad
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        model_type = None
        if vllm_config.model_config is not None:
            model_type = getattr(
                vllm_config.model_config.hf_text_config, "model_type", None
            )
        self.kv_scale_format = _kv_scale_format_for_model(model_type)

        # Skip-topk layers are built with indexer=None and get the shared
        # buffer via mla_args instead (cf. FLASHMLA_SPARSE).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer
            if indexer is not None
            else mla_args.get("topk_indices_buffer")
        )
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        assert self.topk_indices_buffer is not None

        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if self.rope_pad:
            q = torch.nn.functional.pad(q, (0, self.rope_pad))

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        topk_indices_physical, topk_lengths = cast(
            tuple[torch.Tensor, torch.Tensor],
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            ),
        )
        sparse_topk_capacity = topk_indices_physical.shape[1]
        empty_rows = topk_lengths == 0
        topk_indices_physical[:, 0] = topk_indices_physical[:, 0].masked_fill(
            empty_rows, 0
        )
        topk_lengths_raw = topk_lengths
        topk_lengths = topk_lengths.clamp(min=1)

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        kv4 = kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1)

        if sparse_topk_capacity <= _SPLIT_MAIN_TOPK:
            output = q.new_empty(
                (num_actual_toks, self.num_heads, self.kv_lora_rank),
                dtype=q.dtype,
            )
            self._sparse_call(
                q, kv4, topk_indices_physical, topk_lengths, sparse_topk_capacity,
                output, None,
            )
            output.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
            return output, None

        # FlashInfer 0.6.18 only has GLM_NSA kernels for topk in
        # {128, 512, 1024, 2048} (decode) and 2048 (prefill). GLM-5.3-Flash's
        # kpool buffer is 2048 + 128 (tail). Split: main = 2048 (one call),
        # extra = the tail in 64-token slices on the (32, 128) decode kernel,
        # then an exact merge through the LSEs (base log2, validated to
        # 5e-4 = bf16 precision).
        main_k = _SPLIT_MAIN_TOPK
        extra_k = sparse_topk_capacity - main_k
        main_idx = topk_indices_physical[:, :main_k].contiguous()
        main_len = topk_lengths.clamp(max=main_k)
        extra_len_true = (topk_lengths_raw - main_k).clamp(min=0, max=extra_k)
        extra_empty = extra_len_true == 0
        extra_idx = topk_indices_physical[:, main_k:].clone()
        extra_idx[:, 0] = extra_idx[:, 0].masked_fill(extra_empty, 0)
        extra_len = extra_len_true.clamp(min=1)

        o1 = q.new_empty((num_actual_toks, self.num_heads, self.kv_lora_rank))
        l1 = q.new_empty((num_actual_toks, self.num_heads), dtype=torch.float32)
        self._sparse_call(q, kv4, main_idx, main_len, main_k, o1, l1)

        o2 = q.new_empty((num_actual_toks, self.num_heads, self.kv_lora_rank))
        l2 = q.new_empty((num_actual_toks, self.num_heads), dtype=torch.float32)
        for s in range(0, num_actual_toks, _SPLIT_DECODE_CHUNK):
            e = min(s + _SPLIT_DECODE_CHUNK, num_actual_toks)
            self._sparse_call(
                q[s:e], kv4, extra_idx[s:e], extra_len[s:e], extra_k, o2[s:e], l2[s:e]
            )

        l2 = l2.masked_fill(extra_empty.unsqueeze(1), float("-inf"))
        m = torch.maximum(l1, l2)
        w1 = torch.exp2(l1 - m).unsqueeze(-1)
        w2 = torch.exp2(l2 - m).unsqueeze(-1)
        out = ((o1.float() * w1 + o2.float() * w2) / (w1 + w2)).to(q.dtype)
        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
        return out, None

    def _sparse_call(
        self,
        q: torch.Tensor,
        kv4: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        topk: int,
        out: torch.Tensor,
        lse: torch.Tensor | None,
    ) -> None:
        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        kwargs = {}
        if lse is not None:
            kwargs = {"lse": lse.unsqueeze(1), "return_lse": True}
        flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv4,
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.kernel_qk_rope_head_dim,
            block_tables=indices.unsqueeze(1),
            seq_lens=lengths,
            max_seq_len=topk,
            out=out.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=topk,
            kv_scale_format=self.kv_scale_format,
            **kwargs,
        )

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if self.rope_pad:
            k_pe = k_pe.new_zeros((k_pe.shape[0], 1, self.rope_pad))
        super().do_kv_cache_update(
            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
        )
