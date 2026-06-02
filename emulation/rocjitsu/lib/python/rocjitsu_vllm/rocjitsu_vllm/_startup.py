# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Startup hook executed via .pth file at Python interpreter init.

Patches vllm.logger._should_log_with_scope to avoid a circular import
triggered during RocmPlatform module-level initialization when amdsmi
fails to query GPU ASIC info.
"""

import importlib


def _patch():
    try:
        spec = importlib.util.find_spec("vllm.logger")
        if spec is None:
            return
        mod = importlib.import_module("vllm.logger")
        mod._should_log_with_scope = lambda *_a, **_kw: True
    except Exception:
        pass


_patch()
del _patch
