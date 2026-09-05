#!/usr/bin/env python3
"""Register "exl3" as a first-class vLLM quantization method (Literal + lazy mapping to Exl3Config)."""
from pathlib import Path
p = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/__init__.py")
t = p.read_text()
old = 'QuantizationMethods = Literal[\n    "awq",\n'
new = 'QuantizationMethods = Literal[\n    "exl3",\n    "awq",\n'
assert t.count(old) == 1, "Literal anchor"
t = t.replace(old, new)
old2 = "    return method_to_config[quantization]"
new2 = ('    if quantization == "exl3":\n        from .exl3 import Exl3Config\n'
        '        method_to_config["exl3"] = Exl3Config\n'
        '    return method_to_config[quantization]')
assert t.count(old2) == 1, "return anchor"
p.write_text(t.replace(old2, new2))
print("exl3 registered in quantization registry")
