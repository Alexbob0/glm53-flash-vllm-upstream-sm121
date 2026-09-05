"""Let GLM-5.3-Flash's KDA and MLA projections see the real quant config.

Upstream builds both with quant_config=None (their projections are BF16 in the fp8
checkpoints). An EXL3 pack quantizes them, and Exl3Config decides per layer via
non_routed_exl3 (BF16 elsewhere), so the explicit projections must see the config.
kda.py: restore self.quant_config after super().__init__; model.py: pass
vllm_config.quant_config to the MLA attention.
"""
import pathlib
import sys

base = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia")

kda = base / "kda.py"
s = kda.read_text()
old = (
    "        finally:\n"
    "            vllm_config.quant_config = saved_quant_config\n"
)
new = old + (
    "        # [dense-overlay] super() froze self.quant_config=None; the\n"
    "        # explicit projections must see the real config (match\n"
    "        # non_routed_exl3).\n"
    "        self.quant_config = saved_quant_config\n"
)
if "[dense-overlay]" in s:
    print("kda.py: deja patche")
else:
    assert s.count(old) == 1, "kda.py: contexte try/finally introuvable"
    kda.write_text(s.replace(old, new))
    print("kda.py: patche")

model = base / "model.py"
s = model.read_text()
old = "                quant_config=None,  # MLA projections are BF16 in checkpoint\n"
new = "                quant_config=vllm_config.quant_config,  # [dense-overlay] exl3 non_routed match\n"
if "[dense-overlay]" in s:
    print("model.py: deja patche")
else:
    assert s.count(old) == 1, "model.py: site MLA quant_config=None introuvable"
    model.write_text(s.replace(old, new))
    print("model.py: patche")
sys.exit(0)
