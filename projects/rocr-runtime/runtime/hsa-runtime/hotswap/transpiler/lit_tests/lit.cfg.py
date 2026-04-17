# -*- Python -*-
"""Lit configuration for Salmon transpiler FileCheck tests (loaded by lit.site.cfg.py)."""

import os
import shutil

import lit.formats

config.name = "Salmon Transpiler"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".ll"]

# `lit.site.cfg.py` is the intended entry point and sets both roots. Only
# fall back to source-tree defaults when `lit.cfg.py` is loaded directly
# (rare — e.g. running `llvm-lit lit_tests/` without reconfiguring). In
# that fallback path we intentionally route `test_exec_root` under the
# user's temp dir so an ad-hoc run never writes `Output/` into the
# source tree.
if not getattr(config, "test_source_root", None):
    config.test_source_root = os.path.dirname(os.path.abspath(__file__))
if not getattr(config, "test_exec_root", None):
    import tempfile
    config.test_exec_root = os.path.join(
        tempfile.gettempdir(), "salmon-transpiler-lit"
    )

config.substitutions.append(("%raise_cli", config.raise_cli))
config.substitutions.append(("%FileCheck", config.file_check))
config.substitutions.append(("%not", config.not_tool))

for _tag, _path in config.fixtures.items():
    config.substitutions.append(("%{}_co".format(_tag), _path))

if not hasattr(config, "available_features") or config.available_features is None:
    config.available_features = set()
if shutil.which("hipcc"):
    config.available_features.add("hipcc")
