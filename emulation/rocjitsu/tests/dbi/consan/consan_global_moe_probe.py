#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Reduced TokenSpeed global-memory feasibility probe; not E2E qualification.

Requires the pinned TokenSpeed checkout and tokenspeed-triton package described
in docs/consan/SUPERCOLLIDER_GLOBAL_MEMORY.md. Run on gfx950 hardware
or through the gfx950 emulator. No timing results are reported.
"""

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenspeed-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tokenspeed_dir / "tokenspeed-kernel-amd/python"))

    import torch
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.warp_decode_gluon_kernel import (
        invoke_stage1_warp_decode_gluon,
        invoke_stage2_warp_decode_gluon,
    )

    torch.set_num_threads(4)
    torch.manual_seed(1234)
    target = torch.cuda.get_device_properties(0).gcnArchName
    if not target.startswith("gfx950"):
        raise RuntimeError(f"This probe requires gfx950; found {target}")

    tokens, experts, hidden, intermediate, topk = 2, 4, 512, 256, 2
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    w1 = torch.randn(experts, 2 * intermediate, hidden, dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(experts, hidden, intermediate, dtype=torch.bfloat16) * 0.05
    ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75], [0.625, 0.375]], dtype=torch.float32)
    reference = torch.zeros(tokens, hidden)
    for token in range(tokens):
        for slot in range(topk):
            expert = ids[token, slot].item()
            gate_up = x[token].float() @ w1[expert].float().T
            activation = (
                torch.nn.functional.silu(gate_up[:intermediate])
                * gate_up[intermediate:]
            )
            reference[token] += weights[token, slot] * (
                activation @ w2[expert].float().T
            )

    x, w1, w2, ids, weights = [a.cuda() for a in (x, w1, w2, ids, weights)]
    inter = torch.empty(tokens * topk, intermediate, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    invoke_stage1_warp_decode_gluon(x, w1, ids, inter, topk)
    invoke_stage2_warp_decode_gluon(inter, w2, ids, weights, out, topk)
    torch.cuda.synchronize()
    # Keep the oracle's cast on CPU; no unrelated PyTorch GPU conversion kernel.
    actual = out.cpu().float()
    atol = 0.02 * float(reference.abs().max()) + 0.01
    torch.testing.assert_close(actual, reference, atol=atol, rtol=0.0)
    print(json.dumps({
        "target": target,
        "shape": [tokens, experts, hidden, intermediate, topk],
        "oracle": "cpu-fp32-swiglu-weighted-expert-sum",
        "atol": atol,
        "max_abs_error": float((actual - reference).abs().max()),
        "passed": True,
        "scope": "production two-stage kernels, reduced shape; not E2E qualification",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
