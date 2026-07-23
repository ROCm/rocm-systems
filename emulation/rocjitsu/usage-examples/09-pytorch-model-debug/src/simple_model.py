#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""simple_model.py - PyTorch NaN-localization debugging under rocjitsu.

Runs a tiny network's forward pass on the (emulated) GPU and uses per-layer
forward hooks to pinpoint *where* a NaN/Inf first appears and how it propagates:

  * clean pass    - well-formed input, every layer clean -> PASSED
  * corrupt pass  - one layer's weights corrupted (e.g. a bad checkpoint); the
                    hooks show the NaN originates at that layer and propagates
                    to every layer after it -> FAILED

Why inference only: a training step (autograd backward + optimizer) issues far
too many kernels to emulate in practical time. A small forward pass completes in
tens of seconds; training does not. The model is intentionally tiny for the same
reason.

What rocjitsu contributes: it runs the real ROCm PyTorch workload (device='cuda'
-> HIP kernels the emulator executes; no physical GPU), and RJ_LOG=1 shows the
kernel dispatches (incl. 'mfma detected' for the matmuls). The NaN/Inf detection
itself is ordinary host-side PyTorch (torch.isnan / torch.isinf), not a rocjitsu
feature.
"""

import sys
import torch
import torch.nn as nn


class TinyNet(nn.Module):
    """Three tiny linear layers so a NaN can be seen originating and propagating."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 12)
        self.fc2 = nn.Linear(12, 8)
        self.fc3 = nn.Linear(8, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def register_nan_hooks(model, trace):
    """Register a forward hook per named Linear.

    Each hook records (layer_name, bad?) into `trace` in execution order and
    prints the per-layer status, so the output localizes the first bad layer and
    shows propagation. This is plain host-side PyTorch, not a rocjitsu feature.
    """

    def make_hook(name):
        def hook(module, inputs, output):
            bad = bool(torch.isnan(output).any() or torch.isinf(output).any())
            trace.append((name, bad))
            print(f"  [hook] {name} -> {'NaN/Inf' if bad else 'clean'}")

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(make_hook(name))


def report(trace):
    """Summarize a pass: pinpoint the first bad layer and the propagation path."""
    bad_layers = [name for name, bad in trace if bad]
    if not bad_layers:
        print("  result: PASSED (no NaN/Inf)")
        return True
    origin = bad_layers[0]
    propagated = bad_layers[1:]
    print(f"  result: FAILED - NaN/Inf originates at '{origin}'"
          + (f", propagates through: {', '.join(propagated)}" if propagated else ""))
    return False


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("PyTorch NaN-localization debugging under rocjitsu")
    print(f"  torch: {torch.__version__}   cuda_available: {torch.cuda.is_available()}   device: {device}")
    if device == "cpu":
        print("  NOTE: no GPU/HIP backend — install a ROCm PyTorch wheel (torch+rocm).")
    print()

    torch.manual_seed(0)
    model = TinyNet().to(device).eval()
    trace = []
    register_nan_hooks(model, trace)

    x = torch.randn(2, 16, device=device)

    # 1) Clean pass: well-formed input, all layers clean.
    print("[clean] forward pass:")
    trace.clear()
    with torch.no_grad():
        model(x)
    clean_ok = report(trace)
    print()

    # 2) Corrupt pass: simulate a bad layer (e.g. a corrupted checkpoint) by
    #    writing NaN into fc2's weights. The hooks show fc1 clean, then NaN
    #    originating at fc2 and propagating to fc3.
    print("[corrupt] set fc2.weight[0,0] = nan (simulates a corrupted layer):")
    with torch.no_grad():
        model.fc2.weight[0, 0] = float("nan")
    trace.clear()
    with torch.no_grad():
        model(x)
    corrupt_ok = report(trace)
    print()

    # The example "succeeds" if it behaves as designed: clean passes, corrupt is
    # caught and localized.
    if clean_ok and not corrupt_ok:
        print("NaN debugging worked: clean input passed; corrupt layer was localized.")
        return 0
    print("Unexpected: clean should PASS and corrupt should be caught.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
