# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""vLLM out-of-tree platform plugin for rocjitsu GPU simulator.

Bypasses vLLM's built-in ROCm platform detection (which requires amdsmi
to fully initialize) and uses torch.cuda instead. This avoids two issues
in the container environment:
  1. amdsmi's drmGetDevice failure triggers a heap corruption bug in
     libamd_smi.so v26.2.2 (inverted error-handling branch)
  2. vLLM's logger.warning_once circular import when amdsmi_get_gpu_asic_info
     fails and the fallback path is taken

The plugin returns a custom platform class qualname that patches the
module-level _GCN_ARCH variable before importing RocmPlatform, avoiding
the amdsmi code path entirely.
"""


def rocjitsu_platform_plugin() -> "str | None":
    import torch

    if not (hasattr(torch.version, "hip") and torch.version.hip):
        return None
    return "vllm.platforms.rocm.RocmPlatform"
