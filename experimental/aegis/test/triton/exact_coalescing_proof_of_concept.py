#!/usr/bin/env python3
"""
Exact coalescing proof-of-concept for AegisBit.

This test avoids broad thresholds. It launches tiny Triton kernels with fully
deterministic lane-to-address mappings, then compares AegisBit's reported
cache-line counts against an independent Python oracle.

The proof is intentionally limited to VMEM coalescing. Exact LDS conflict
proofs are possible too, but they require a more controlled shared-memory
kernel/oracle pairing than Triton currently gives us cheaply.

Run:
    python3 test/triton/exact_coalescing_proof_of_concept.py
"""

import os
import re
import subprocess
import sys

import torch
import triton
import triton.language as tl


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
AEGISBIT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
LIB_PATH = os.path.join(AEGISBIT_DIR, "build", "src", "libaegisbit.so")
CACHELINE_BYTES = 128
ELEM_BYTES = 4
LANES = 64


@triton.jit
def exact_coalesced_kernel(Data_ptr, Out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(Data_ptr + offs, mask=mask)
    tl.store(Out_ptr + offs, x, mask=mask)


@triton.jit
def exact_grouped_gather_kernel(Data_ptr, Idx_ptr, Out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    idx = tl.load(Idx_ptr + offs, mask=mask)
    x = tl.load(Data_ptr + idx, mask=mask)
    tl.store(Out_ptr + offs, x, mask=mask)


@triton.jit
def exact_strided_kernel(Data_ptr, Out_ptr, N, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    src_offs = offs * STRIDE
    x = tl.load(Data_ptr + src_offs, mask=mask)
    tl.store(Out_ptr + offs, x, mask=mask)


def expected_cachelines(indices):
    byte_addrs = [int(i) * ELEM_BYTES for i in indices]
    return len({addr // CACHELINE_BYTES for addr in byte_addrs})


def parse_vmem_sites(output):
    sites = []
    current_kernel = None
    for line in output.splitlines():
        m = re.match(r"=== VMEM Coalescing: (\S+) ===", line)
        if m:
            current_kernel = m.group(1)
            continue
        if current_kernel is None:
            continue
        m = re.match(
            r"\s+(.+?)\s+(\d+)×(load\+?s?t?o?r?e?|store|load)\s+"
            r"eff=(\d+)\s*%\s+cachelines=(\d+)\s+(\w+)",
            line,
        )
        if m:
            sites.append(
                {
                    "kernel": current_kernel,
                    "source": m.group(1).strip(),
                    "kind": m.group(3),
                    "efficiency": int(m.group(4)),
                    "cachelines": int(m.group(5)),
                    "pattern": m.group(6),
                }
            )
    return sites


def run_kernels():
    torch.manual_seed(0)

    data = torch.arange(LANES * 64, device="cuda", dtype=torch.float32)
    out = torch.empty(LANES, device="cuda", dtype=torch.float32)

    # Case 1: contiguous fp32 lane access -> 64 * 4B = 256B -> exactly 2 cache lines.
    exact_coalesced_kernel[(1,)](data, out, LANES, BLOCK=LANES)
    torch.cuda.synchronize()
    assert torch.equal(out, data[:LANES])

    # Case 2: deterministic grouped gather touching exactly 4 cache lines.
    grouped_idx = torch.tensor(
        [0] * 16 + [32] * 16 + [96] * 16 + [224] * 16,
        device="cuda",
        dtype=torch.int32,
    )
    out2 = torch.empty(LANES, device="cuda", dtype=torch.float32)
    exact_grouped_gather_kernel[(1,)](data, grouped_idx, out2, LANES, BLOCK=LANES)
    torch.cuda.synchronize()
    assert torch.equal(out2, data[grouped_idx.long()])

    # Case 3: one lane per cache line.
    stride = CACHELINE_BYTES // ELEM_BYTES
    out3 = torch.empty(LANES, device="cuda", dtype=torch.float32)
    exact_strided_kernel[(1,)](data, out3, LANES, STRIDE=stride, BLOCK=LANES)
    torch.cuda.synchronize()
    expected3 = data[torch.arange(LANES, device="cuda") * stride]
    assert torch.equal(out3, expected3)

    print("Kernel execution PASS")


def validate():
    if not os.path.exists(LIB_PATH):
        print(f"ERROR: libaegisbit.so not found at {LIB_PATH}")
        sys.exit(1)

    env = {
        **os.environ,
        "AEGISBIT_ENABLED": "1",
        "AEGISBIT_MODE": "MEMORY_ONLY",
        "AEGISBIT_STRATEGY": "on_gpu_reduce",
        "AEGISBIT_KERNELS": "*exact_coalesced*,*exact_grouped_gather*,*exact_strided*",
        "LD_PRELOAD": LIB_PATH,
    }

    result = subprocess.run(
        [sys.executable, __file__, "--run-kernels"],
        cwd=AEGISBIT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    combined = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        print(combined[-4000:])
        sys.exit(result.returncode)

    sites = parse_vmem_sites(combined)
    if not sites:
        print("ERROR: No VMEM profiling output found.")
        print(combined[-4000:])
        sys.exit(1)

    grouped_idx = [0] * 16 + [32] * 16 + [96] * 16 + [224] * 16
    expectations = [
        (
            "exact_coalesced_kernel",
            "Data_ptr + offs",
            expected_cachelines(list(range(LANES))),
        ),
        (
            "exact_grouped_gather_kernel",
            "Data_ptr + idx",
            expected_cachelines(grouped_idx),
        ),
        (
            "exact_strided_kernel",
            "Data_ptr + src_offs",
            expected_cachelines([i * (CACHELINE_BYTES // ELEM_BYTES) for i in range(LANES)]),
        ),
    ]

    failures = []
    for kernel_name, source_fragment, expected in expectations:
        matches = [
            s
            for s in sites
            if kernel_name in s["kernel"]
            and source_fragment in s["source"]
            and "load" in s["kind"]
        ]
        if not matches:
            failures.append(f"{kernel_name}: no matching load site")
            continue
        actual = matches[0]["cachelines"]
        if actual != expected:
            failures.append(
                f"{kernel_name}: expected {expected} cachelines, got {actual}"
            )
        else:
            print(f"PASS: {kernel_name} exact cachelines = {actual}")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        print("\n--- profiler tail ---")
        print(combined[-4000:])
        sys.exit(1)

    print("\nExact coalescing proof-of-concept PASSED.")


def main():
    if "--run-kernels" in sys.argv:
        run_kernels()
    else:
        validate()


if __name__ == "__main__":
    main()
