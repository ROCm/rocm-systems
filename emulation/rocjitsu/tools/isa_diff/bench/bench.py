#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Measure rj_isa_diff cost against region size, with a whole-kernel control.

Each kernel is a 2048-instruction straight-line SALU body. The changed region is R
instructions; the new side commutes every s_add_u32 in it, which is equivalent. The
"whole" variant extends the region back to the kernel start (--context), so every case
executes the full kernel: that is the full-fixture control the per-region numbers are
compared against. The generated inputs need llvm-mc; the measured tool does not.

usage: bench.py --tool build/tools/isa_diff/rj_isa_diff --out DIR [--cases N]
"""

import argparse
import json
import random
import subprocess
import time
from pathlib import Path


def body(n, rng, regs):
    ops = []
    for _ in range(n):
        a, b, c = rng.sample(regs, 3)
        kind = rng.randrange(6)
        if kind == 0:
            ops.append(f"s_add_u32 s{a}, s{b}, s{c}")
        elif kind == 1:
            ops.append(f"s_xor_b32 s{a}, s{b}, s{c}")
        elif kind == 2:
            ops.append(f"s_lshl_b32 s{a}, s{b}, {rng.randrange(32)}")
        elif kind == 3:
            ops.append(f"s_and_b32 s{a}, s{b}, s{c}")
        elif kind == 4:
            ops.append(f"s_min_u32 s{a}, s{b}, s{c}")
        else:
            ops.append(f"s_cmp_lg_u32 s{b}, s{c}")
    return ops


def region(n, rng, regs):
    old, new = [], []
    for _ in range(n):
        a, b, c = rng.sample(regs, 3)
        old.append(f"s_add_u32 s{a}, s{b}, s{c}")
        new.append(f"s_add_u32 s{a}, s{c}, s{b}")
    return old, new


def assemble(lines, path, mc, objcopy):
    src = path.with_suffix(".s")
    src.write_text("\n".join(lines) + "\n")
    obj = path.with_suffix(".o")
    subprocess.run(
        [
            mc,
            "-triple=amdgcn-amd-amdhsa",
            "-mcpu=gfx950",
            "-filetype=obj",
            str(src),
            "-o",
            str(obj),
        ],
        check=True,
    )
    subprocess.run(
        [objcopy, "-O", "binary", "--only-section=.text", str(obj), str(path)],
        check=True,
    )
    obj.unlink()


def run(cmd):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    return time.perf_counter() - t0, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--kernel-insts", type=int, default=2048)
    ap.add_argument("--llvm-mc", default="/opt/rocm/llvm/bin/llvm-mc")
    ap.add_argument("--objcopy", default="/opt/rocm/llvm/bin/llvm-objcopy")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    regs = list(range(10, 60))
    rows = []
    variants = [(r, False) for r in (1, 4, 16, 64, 256)] + [(16, True)]
    for size, whole in variants:
        rng = random.Random(1234 + size)
        prefix = body(args.kernel_insts - size - 64, rng, regs)
        old_r, new_r = region(size, rng, regs)
        suffix = body(63, rng, regs) + ["s_endpgm"]
        name = f"r{size}" + ("_whole" if whole else "")
        d = out / name
        d.mkdir(exist_ok=True)
        assemble(prefix + old_r + suffix, d / "old.bin", args.llvm_mc, args.objcopy)
        assemble(prefix + new_r + suffix, d / "new.bin", args.llvm_mc, args.objcopy)
        analyze = [
            args.tool,
            "analyze",
            "--old",
            str(d / "old.bin"),
            "--new",
            str(d / "new.bin"),
            "--out",
            str(d / "case"),
            "--max-region",
            "4096",
        ]
        if whole:
            analyze += ["--context", "100000", "--max-expansions", "100000"]
        t_an, p = run(analyze)
        if p.returncode != 0:
            raise SystemExit(f"analyze failed for {name}: {p.stderr}")
        t_te, p = run(
            [
                args.tool,
                "test",
                "--case",
                str(d / "case"),
                "--cases",
                str(args.cases),
                "--seed",
                "1",
            ]
        )
        if p.returncode not in (0, 1, 2):
            raise SystemExit(f"test failed for {name}: {p.stderr}")
        case = json.loads((d / "case" / "case.json").read_text())
        res = json.loads((d / "case" / "results.json").read_text())
        reg = res["regions"][0]
        t = reg["timing"]
        n = max(reg["counts"]["executed_pairs"], 1)
        per_case_ns = (
            t["propose_ns"]
            + t["state_load_ns"]
            + t["old_execute_ns"]
            + t["new_execute_ns"]
            + t["compare_ns"]
        ) / n
        rows.append(
            {
                "variant": name,
                "kernel_insts": case["artifacts"]["old"]["reachable_insts"],
                "region_old": reg["execution"]["old_region_insts"],
                "region_new": reg["execution"]["new_region_insts"],
                "status": reg["test_status"],
                "contract": reg["contract_status"],
                "valid_pairs": reg["counts"]["valid_pairs"],
                "insts_per_case": (
                    reg["execution"]["old_insts_executed"]
                    + reg["execution"]["new_insts_executed"]
                )
                / n,
                "analyze_wall_ms": t_an * 1e3,
                "decode_cfg_liveness_ms": case["analysis"]["timing"][
                    "decode_cfg_liveness_ns"
                ]
                / 1e6,
                "align_ms": case["analysis"]["timing"]["align_classify_ns"] / 1e6,
                "plan_ms": case["analysis"]["timing"]["plan_ns"] / 1e6,
                "runner_setup_ms": t["setup_ns"] / 1e6,
                "per_case_us": per_case_ns / 1e3,
                "old_exec_us": t["old_execute_ns"] / n / 1e3,
                "new_exec_us": t["new_execute_ns"] / n / 1e3,
                "load_us": t["state_load_ns"] / n / 1e3,
                "propose_us": t["propose_ns"] / n / 1e3,
                "compare_us": t["compare_ns"] / n / 1e3,
                "test_wall_s": t_te,
                "cases_per_s_wall": n / t_te,
            }
        )
    (out / "bench.json").write_text(json.dumps(rows, indent=2) + "\n")
    cols = [
        "variant",
        "kernel_insts",
        "region_old",
        "region_new",
        "status",
        "insts_per_case",
        "analyze_wall_ms",
        "decode_cfg_liveness_ms",
        "runner_setup_ms",
        "per_case_us",
        "old_exec_us",
        "new_exec_us",
        "load_us",
        "propose_us",
        "compare_us",
        "test_wall_s",
        "cases_per_s_wall",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    (out / "bench.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
