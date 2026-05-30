"""Minimal torch fixture used by ml_scenarios.rs.

Allocates a tiny CPU tensor (the rocjitsu pipeline doesn't yet
provide a CUDA backend the test harness can rely on), validates the
sum, and prints the sentinel string the test asserts on.
"""

import torch

x = torch.zeros(8)
x.add_(1.0)
assert x.sum().item() == 8.0, x.sum().item()
print("tiny_torch_ok")
