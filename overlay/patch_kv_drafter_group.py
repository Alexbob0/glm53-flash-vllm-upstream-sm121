"""[chantier B-bis] Groupe KV "drafter" pour GLM-5.3-Flash + DFlash2 sur la nightly.

Port du DFLASH2-DRAFTER-GROUP du fork MiaAI (mode "padded slot-share") : les
couches SlidingWindowSpec du draft (block 64, page rembourree a la page MLA)
co-possedent le tenseur MLA i a des ids de blocs disjoints, comme les couches
mamba. Sans cela le chemin generique rembourre les etats KDA a la page du
draft (dizaines de Go) ou echoue sur l'unification des pages.
"""
import pathlib
import sys

p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py")
s = p.read_text()
if "[drafter-group]" in s:
    print("kv_cache_utils.py: deja patche")
    sys.exit(0)

edits = [
    # E1: partition des couches SWA du draft hors du groupe attention
    (
        "    attn_specs = {\n"
        "        name: spec\n"
        "        for name, spec in kv_cache_spec.items()\n"
        "        if not isinstance(spec, (MambaSpec, KpoolTailSpec))\n"
        "    }\n"
        "    if not mamba_specs or not all(\n"
        "        type(spec) is MLAAttentionSpec for spec in attn_specs.values()\n"
        "    ):\n"
        "        return None\n",
        "    # [drafter-group] SWA layers of a DFlash2 drafter (exact type: KpoolTailSpec\n"
        "    # subclasses SlidingWindowSpec) get their own slot-shared group below.\n"
        "    draft_specs = {\n"
        "        name: spec\n"
        "        for name, spec in kv_cache_spec.items()\n"
        "        if type(spec) is SlidingWindowSpec\n"
        "    }\n"
        "    attn_specs = {\n"
        "        name: spec\n"
        "        for name, spec in kv_cache_spec.items()\n"
        "        if not isinstance(spec, (MambaSpec, KpoolTailSpec))\n"
        "        and type(spec) is not SlidingWindowSpec\n"
        "    }\n"
        "    if not mamba_specs or not all(\n"
        "        type(spec) is MLAAttentionSpec for spec in attn_specs.values()\n"
        "    ):\n"
        "        return None\n",
    ),
    # E2: groupe drafter (padded slot-share) ajoute en dernier
    (
        "    return (\n"
        "        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]\n"
        "        + ([tail_group] if tail_group is not None else [])\n"
        "        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)\n"
        "    )\n",
        "    draft_group: KVCacheGroupSpec | None = None\n"
        "    if draft_specs:\n"
        "        any_draft = next(iter(draft_specs.values()))\n"
        "        assert all(spec == any_draft for spec in draft_specs.values()), (\n"
        "            \"drafter SlidingWindowSpec layers must share one spec\"\n"
        "        )\n"
        "        assert len(draft_specs) <= len(mla_names), (\n"
        "            \"drafter layers exceed MLA tensors available for slot-sharing\"\n"
        "        )\n"
        "        # Draft block: during prefill the SWA group must hold the whole prompt before\n"
        "        # trimming to its window; 64-token blocks (fork default) eat 281 ids for an\n"
        "        # 18K prompt and thrash the pool under concurrent prefills. Page must stay\n"
        "        # <= mla_page (padded), i.e. block <= mla_page // bytes_per_token, and the\n"
        "        # block must DIVIDE the MLA block (4608): hybrid prefix-cache hits are aligned\n"
        "        # on the lcm of all group block sizes (1024 -> hits only every 9216 tokens).\n"
        "        draft_bytes_per_token = any_draft.page_size_bytes // any_draft.block_size\n"
        "        compact_block = int(__import__(\"os\").environ.get(\"GLM53_DRAFT_BLOCK\", \"1152\"))\n"
        "        compact_block = max(64, min(compact_block, (mla_page // draft_bytes_per_token) // 64 * 64))\n"
        "        logger.info(\n"
        "            \"[drafter-group] DFlash2 drafter KV: padded slot-share block=%d \"\n"
        "            \"mla_page=%d (was block=%d, %d bytes/token)\",\n"
        "            compact_block,\n"
        "            mla_page,\n"
        "            any_draft.block_size,\n"
        "            draft_bytes_per_token,\n"
        "        )\n"
        "        new_draft_specs: dict[str, KVCacheSpec] = {\n"
        "            name: replace(spec, block_size=compact_block, page_size_padded=mla_page)\n"
        "            for name, spec in draft_specs.items()\n"
        "        }\n"
        "        draft_uniform = UniformTypeKVCacheSpecs.from_specs(new_draft_specs)\n"
        "        assert draft_uniform is not None\n"
        "        draft_group = KVCacheGroupSpec(list(new_draft_specs), draft_uniform)\n"
        "\n"
        "    return (\n"
        "        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]\n"
        "        + ([tail_group] if tail_group is not None else [])\n"
        "        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)\n"
        "        + ([draft_group] if draft_group is not None else [])\n"
        "    )\n",
    ),
    # E3a: layout — annotation de retour (9-uplet)
    (
        "        list[str],\n"
        "        int,\n"
        "    ]\n"
        "    | None\n"
        "):\n"
        "    \"\"\"Recognize the GLM-5.3-Flash grouping after optional PP projection.\"\"\"\n",
        "        list[str],\n"
        "        int,\n"
        "        KVCacheGroupSpec | None,\n"
        "    ]\n"
        "    | None\n"
        "):\n"
        "    \"\"\"Recognize the GLM-5.3-Flash grouping after optional PP projection.\"\"\"\n",
    ),
    # E3b: layout — detection du groupe drafter
    (
        "    tail_group: KVCacheGroupSpec | None = None\n"
        "    for group in uniform_groups:\n"
        "        inner = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs\n"
        "        if all(type(spec) is MLAAttentionSpec for spec in inner.values()):\n"
        "            attn_group = group\n"
        "        elif all(isinstance(spec, KpoolTailSpec) for spec in inner.values()):\n"
        "            tail_group = group\n",
        "    tail_group: KVCacheGroupSpec | None = None\n"
        "    draft_group: KVCacheGroupSpec | None = None\n"
        "    for group in uniform_groups:\n"
        "        inner = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs\n"
        "        if all(type(spec) is MLAAttentionSpec for spec in inner.values()):\n"
        "            attn_group = group\n"
        "        elif all(isinstance(spec, KpoolTailSpec) for spec in inner.values()):\n"
        "            tail_group = group\n"
        "        elif inner and all(type(spec) is SlidingWindowSpec for spec in inner.values()):\n"
        "            draft_group = group  # [drafter-group]\n",
    ),
    # E3c: layout — validation + retour
    (
        "    tail_names: list[str] = []\n"
        "    tail_page = 0\n"
        "    if tail_group is not None:\n"
        "        tail_names = list(tail_group.layer_names)\n"
        "        tail_inner = cast(\n",
        "    if draft_group is not None:\n"
        "        # [drafter-group] padded slot-share: page (padded) == mla_page, one\n"
        "        # MLA tensor per drafter layer.\n"
        "        draft_inner = cast(\n"
        "            UniformTypeKVCacheSpecs, draft_group.kv_cache_spec\n"
        "        ).kv_cache_specs\n"
        "        if any(spec.page_size_bytes != mla_page for spec in draft_inner.values()):\n"
        "            return None\n"
        "        if len(draft_group.layer_names) > len(mla_names):\n"
        "            return None\n"
        "\n"
        "    tail_names: list[str] = []\n"
        "    tail_page = 0\n"
        "    if tail_group is not None:\n"
        "        tail_names = list(tail_group.layer_names)\n"
        "        tail_inner = cast(\n",
    ),
    (
        "        tail_names,\n"
        "        tail_page,\n"
        "    )\n",
        "        tail_names,\n"
        "        tail_page,\n"
        "        draft_group,\n"
        "    )\n",
    ),
    # E4: bytes per block (draft slot-shared: aucun octet en plus)
    (
        "        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = glm5_layout\n",
        "        _, _, mla_names, idx_names, mla_page, idx_page, _, _, _ = glm5_layout\n",
    ),
    # E5: emission des tenseurs — draft layer i sur le tenseur MLA i
    (
        "            tail_names,\n"
        "            _,\n"
        "        ) = glm5_layout\n"
        "        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page\n",
        "            tail_names,\n"
        "            _,\n"
        "            draft_group,\n"
        "        ) = glm5_layout\n"
        "        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page\n",
    ),
    (
        "            for group in mamba_groups:\n"
        "                if index < len(group.layer_names):\n"
        "                    add_tensor(group.layer_names[index], group.kv_cache_spec, offset)\n"
        "\n"
        "        idx_base = len(mla_names) * mla_page * num_blocks\n",
        "            for group in mamba_groups:\n"
        "                if index < len(group.layer_names):\n"
        "                    add_tensor(group.layer_names[index], group.kv_cache_spec, offset)\n"
        "            if draft_group is not None and index < len(draft_group.layer_names):\n"
        "                # [drafter-group] drafter layer i rides MLA tensor i (strided\n"
        "                # view via page_size_padded, disjoint block ids).\n"
        "                draft_name = draft_group.layer_names[index]\n"
        "                draft_inner = cast(\n"
        "                    UniformTypeKVCacheSpecs, draft_group.kv_cache_spec\n"
        "                ).kv_cache_specs\n"
        "                add_tensor(draft_name, draft_inner[draft_name], offset)\n"
        "\n"
        "        idx_base = len(mla_names) * mla_page * num_blocks\n",
    ),
    # E6: usage memoire max — blocs du draft (SWA) comptes
    (
        "            tail_names,\n"
        "            _,\n"
        "        ) = glm5_layout\n"
        "        uniform_spec = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)\n",
        "            tail_names,\n"
        "            _,\n"
        "            draft_group,\n"
        "        ) = glm5_layout\n"
        "        uniform_spec = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)\n",
    ),
    (
        "        if tail_names:\n"
        "            total_blocks += 1\n"
        "        return total_blocks * (len(mla_names) * mla_page + len(idx_names) * idx_page)\n",
        "        if tail_names:\n"
        "            total_blocks += 1\n"
        "        if draft_group is not None:  # [drafter-group]\n"
        "            # Block ids are shared across groups (one id = one page in every\n"
        "            # tensor): a request costs the drafter its SWA window in ids, at\n"
        "            # the full per-block byte cost, not one page per layer.\n"
        "            draft_spec = next(\n"
        "                iter(\n"
        "                    cast(\n"
        "                        UniformTypeKVCacheSpecs, draft_group.kv_cache_spec\n"
        "                    ).kv_cache_specs.values()\n"
        "                )\n"
        "            )\n"
        "            draft_window = min(\n"
        "                getattr(draft_spec, \"sliding_window\", 0) or 0,\n"
        "                vllm_config.model_config.max_model_len,\n"
        "            )\n"
        "            total_blocks += cdiv(draft_window, draft_spec.block_size) + 2\n"
        "        return total_blocks * (len(mla_names) * mla_page + len(idx_names) * idx_page)\n",
    ),
]
for old, new in edits:
    assert s.count(old) == 1, f"kv_cache_utils.py: contexte introuvable/ambigu ({s.count(old)}):\n{old}"
    s = s.replace(old, new)
p.write_text(s)
print("kv_cache_utils.py: patche (drafter-group)")
