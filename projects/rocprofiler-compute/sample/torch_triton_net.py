# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Sample mixed PyTorch + Triton workload for rocprof-compute --ml-api-trace.

    rocprof-compute profile --experimental --ml-api-trace --no-roof \
        -n torch_triton_net -- python3 ./torch_triton_net.py
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def scale_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)


def triton_scale(x):
    out = torch.empty_like(x)
    n = out.numel()
    scale_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK_SIZE=BLOCK)
    return out


class TorchTritonNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return triton_scale(x)


def main():
    if not torch.cuda.is_available():
        print("GPU is required for this sample. Exiting.")
        sys.exit(1)

    model = TorchTritonNet().cuda()
    inp = torch.randn(64, 64, device="cuda")
    for _ in range(3):
        model(inp)
    torch.cuda.synchronize()
    print("Mixed torch/triton workload completed")


if __name__ == "__main__":
    main()
