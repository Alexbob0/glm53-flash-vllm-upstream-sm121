"""Inserts the GLM53_NGRAM call right before drafts are recorded into req_states (nightly model_runner v2)."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
anchor = "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n"
assert s.count(anchor) == 1, "anchor missing or not unique"
hook = ("            # [25/09] GLM53_NGRAM : brouillon hybride DFlash2 + n-gram (overlay/ngram_hybrid)\n"
        "            from vllm.v1.worker.gpu.spec_decode.glm53_ngram import maybe_override as _glm53_ngram\n"
        "            _glm53_ngram(draft_tokens, input_batch, self.req_states, self.speculator)\n")
if hook not in s:
    s = s.replace(anchor, hook + anchor)
open(dst, "w").write(s)
