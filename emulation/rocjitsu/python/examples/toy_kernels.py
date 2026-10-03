# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Small exact-result PyTorch and compiled Triton kernels for API qualification."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _add(x, y, output, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(
        output + offsets,
        tl.load(x + offsets, mask, other=0) + tl.load(y + offsets, mask, other=0),
        mask,
    )


def check(framework, device=0, n=257):
    """Execute a kernel, check every output on the CPU, and report device identity."""
    with torch.cuda.device(device):
        x = torch.arange(n, dtype=torch.float32, device=f"cuda:{device}")
        y = torch.full_like(x, 3)
        if framework == "pytorch":
            output = x + y
        elif framework == "triton":
            output = torch.empty_like(x)
            _add[(triton.cdiv(n, 128),)](x, y, output, n, BLOCK=128)
        else:
            raise ValueError(framework)
        actual = output.cpu()
        expected = torch.arange(n, dtype=torch.float32) + 3
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        props = torch.cuda.get_device_properties(device)
        # Marketing names can be replaced by ROCr's PCI database. Inspect the
        # allocation backing as well as the architecture to distinguish an
        # actual simulator dispatch from execution on an identically named GPU.
        simulated = "memfd:rocjitsu_" in open("/proc/self/maps").read()
        return {
            "framework": framework,
            "pid": os.getpid(),
            "name": props.name,
            "arch": props.gcnArchName,
            "simulated": simulated,
            "elements": n,
            "output": actual,
        }


def gpu_tensor():
    return torch.zeros(1, device="cuda")
