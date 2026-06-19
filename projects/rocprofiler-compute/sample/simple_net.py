# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Sample PyTorch workload for simple network training.

Prerequisites:
    PyTorch must be installed with GPU support. Install via:

    ROCm:
        pip install --no-cache-dir --pre torch --index-url \
            https://download.pytorch.org/whl/nightly/rocm7.2

    See https://pytorch.org/get-started/locally/ for more options.

Usage:
    rocprof-compute profile --experimental --torch-trace --no-roof \
        -n simple_net  -- python3 ./simple_net.py
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def main():
    if not torch.cuda.is_available():
        print("GPU is required for this sample. Exiting.")
        sys.exit(1)

    model = SimpleNet().cuda()
    x = torch.randn(5, 10).cuda()

    for _ in range(1):
        output = model(x)
        loss = output.sum()
        loss.backward()

    print("Training completed")


if __name__ == "__main__":
    main()
