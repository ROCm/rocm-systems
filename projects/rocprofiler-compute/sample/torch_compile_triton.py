# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Minimal torch.compile workload that generates Triton kernels."""

import sys

import torch


@torch.compile
def fused(x, y):
    return torch.relu(x) * y + x


def main():
    if not torch.cuda.is_available():
        print("GPU is required for this sample. Exiting.")
        sys.exit(1)

    x = torch.randn(4096, 4096, device="cuda")
    y = torch.randn(4096, 4096, device="cuda")

    # First call compiles; later calls reuse the generated Triton kernels.
    for _ in range(3):
        fused(x, y)

    torch.cuda.synchronize()
    print("Compiled workload completed")


if __name__ == "__main__":
    main()
