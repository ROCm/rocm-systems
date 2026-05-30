"""Tiny torch fixture used by ml_scenarios.rs.

When run under `LD_PRELOAD=librocjitsu_kmd.so` with `RJ_CONFIG`/
`RJ_SCHEMA` set, ROCm/HIP calls are routed into the rocjitsu
simulated GPU. This script asserts the simulated GPU is visible to
torch and runs a tiny tensor op on it.

Exit codes:
  * 0 + prints `tiny_torch_ok` -> success
  * non-zero on any failure (no silent fallback to CPU)
"""

import sys

import torch

if not torch.cuda.is_available():
    print("FAIL: torch.cuda.is_available() returned False", file=sys.stderr)
    sys.exit(2)

device = torch.device("cuda")
x = torch.zeros(8, device=device)
x.add_(1.0)
total = x.sum().item()
if total != 8.0:
    print(f"FAIL: expected sum 8.0, got {total}", file=sys.stderr)
    sys.exit(3)

print("tiny_torch_ok")
