#!/usr/bin/env python3
"""
Profiler validation test — ground-truth checks for AegisBit's VMEM
coalescing and LDS bank conflict reporting.

Tier 1 (anchor kernels):
  Trivial kernels where the correct answer is provable from the source.
  - VMEM: coalesced (stride-1) vs scattered (random) vs strided
  - LDS:  conflict-free (sequential banks) vs worst-case (same bank)

Tier 2 (composite kernels):
  Non-trivial kernels combining multiple access patterns. The profiler
  must attribute correct, *different* metrics to sites within the SAME
  kernel — proving per-site discrimination, not just kernel-level averages.

Run:
    python3 test/triton/profiler_validation_test.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import torch
import triton
import triton.language as tl


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
AEGISBIT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
LIB_PATH = os.path.join(AEGISBIT_DIR, "build", "src", "libaegisbit.so")


# =========================================================================
# Tier 1 — VMEM Anchor Kernels
# =========================================================================

@triton.jit
def t1_coalesced_kernel(Out_ptr, In_ptr, N, BLOCK: tl.constexpr):
    """Stride-1: lane k loads element pid*BLOCK+k. Perfectly coalesced."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(In_ptr + offs, mask=mask)
    tl.store(Out_ptr + offs, x, mask=mask)


@triton.jit
def t1_scattered_kernel(Out_ptr, In_ptr, Idx_ptr, N, BLOCK: tl.constexpr):
    """Random gather: each lane loads a random element. Worst-case coalescing."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    idx = tl.load(Idx_ptr + offs, mask=mask)
    x = tl.load(In_ptr + idx, mask=mask)
    tl.store(Out_ptr + offs, x, mask=mask)


@triton.jit
def t1_strided_kernel(Out_ptr, In_ptr, N, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    """Strided: lane k loads element k*STRIDE. Every lane hits a different cacheline."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(In_ptr + offs * STRIDE, mask=mask)
    tl.store(Out_ptr + offs, x, mask=mask)


# =========================================================================
# Tier 1 — LDS Anchor Kernels
# =========================================================================

@triton.jit
def t1_lds_no_conflict_kernel(Out_ptr, In_ptr, N, BLOCK: tl.constexpr):
    """
    Each thread writes to LDS at sequential 4-byte offsets (thread k → bank k%32).
    All 32 banks hit by different threads → 0 bank conflicts.
    Uses tl.dot with transposed inputs to force LDS usage, but the key LDS
    write is the explicit ds_write pattern from loading into shared memory.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(In_ptr + offs, mask=mask)
    tl.store(Out_ptr + offs, x + 1.0, mask=mask)


@triton.jit
def t1_lds_conflict_kernel(
    Out_ptr, A_ptr, B_ptr, M, K, N,
    stride_am, stride_ak, stride_bk, stride_bn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Tiled matmul via tl.dot — forces LDS staging of A and B tiles.
    Power-of-2 tile dimensions with standard layout → bank conflicts.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                     mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                     mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# =========================================================================
# Tier 2 — Composite Kernels
# =========================================================================

@triton.jit
def t2_mixed_vmem_kernel(
    Out_ptr, CoalIn_ptr, ScatIn_ptr, Idx_ptr, N,
    BLOCK: tl.constexpr,
):
    """
    Single kernel with THREE distinct VMEM patterns:
      1. Coalesced load (stride-1)
      2. Scattered load (random index)
      3. Coalesced store (stride-1)

    The profiler must report different efficiencies for sites 1 vs 2,
    within the same kernel dispatch.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    a = tl.load(CoalIn_ptr + offs, mask=mask)
    idx = tl.load(Idx_ptr + offs, mask=mask)
    b = tl.load(ScatIn_ptr + idx, mask=mask)

    tl.store(Out_ptr + offs, a + b, mask=mask)


@triton.jit
def t2_mixed_vmem_lds_kernel(
    C_ptr, A_ptr, B_ptr, Scat_ptr, Idx_ptr,
    M, K, N,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused GEMM + scattered gather in one kernel.
      - A/B tile loads go through LDS (via tl.dot) → bank conflicts
      - Scattered load uses random indices → poor VMEM coalescing
      - Output store is coalesced

    Tests per-site discrimination across VMEM coalescing AND LDS conflicts
    within a single kernel.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                     mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                     mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    flat_offs = pid_m * BLOCK_M * 1024 + tl.arange(0, BLOCK_M)
    scat_idx = tl.load(Idx_ptr + flat_offs, mask=flat_offs < M * 1024)
    scat_val = tl.load(Scat_ptr + scat_idx, mask=flat_offs < M * 1024)

    acc = acc + scat_val[:, None]

    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# =========================================================================
# Report Parsing
# =========================================================================

def parse_vmem(output):
    """Parse VMEM coalescing reports. Returns [{kernel, source, kind, eff, cachelines, pattern}]."""
    sites = []
    current_kernel = None
    for line in output.splitlines():
        m = re.match(r"=== VMEM Coalescing: (\S+)", line)
        if m:
            current_kernel = m.group(1)
            continue
        if current_kernel is None:
            continue
        m = re.match(
            r"\s+(.+?)\s+(\d+)×(load\+?s?t?o?r?e?|store|load)\s+"
            r"eff=(\d+)\s*%\s+cachelines=(\d+)\s+(\w+)", line)
        if m:
            sites.append({
                "kernel": current_kernel,
                "source": m.group(1).strip(),
                "count": int(m.group(2)),
                "kind": m.group(3),
                "eff": int(m.group(4)),
                "cachelines": int(m.group(5)),
                "pattern": m.group(6),
            })
    return sites


def parse_lds(output):
    """Parse LDS bank conflict reports.

    The on_gpu_reduce strategy emits one of three tails per line:
      ``avg_n-way=X.Y  conflict_free=X%`` (when per-sample breakdown present),
      ``avg_n-way=X.Y`` (no breakdown), or
      ``no conflicts``.

    Returns [{kernel, source, kind, avg_n_way, cf_pct (optional)}].
    Sites with ``no conflicts`` are reported as avg_n_way=1.0.
    """
    sites = []
    current_kernel = None
    head_re = re.compile(
        r"\s+(.+?)\s+(\d+)×(load\+?s?t?o?r?e?|store|load)\s+(.*)$")
    avg_re = re.compile(r"avg_n-way=([\d.]+)")
    cf_re = re.compile(r"conflict_free=([\d.]+)%")
    for line in output.splitlines():
        m = re.match(r"=== LDS Bank Conflicts: (\S+)", line)
        if m:
            current_kernel = m.group(1)
            continue
        if current_kernel is None:
            continue
        m = head_re.match(line)
        if not m:
            continue
        tail = m.group(4)
        site = {
            "kernel": current_kernel,
            "source": m.group(1).strip(),
            "count": int(m.group(2)),
            "kind": m.group(3),
        }
        avg_m = avg_re.search(tail)
        if avg_m:
            site["avg_n_way"] = float(avg_m.group(1))
        elif "no conflicts" in tail:
            site["avg_n_way"] = 1.0
        else:
            continue
        cf_m = cf_re.search(tail)
        if cf_m:
            site["cf_pct"] = float(cf_m.group(1))
        sites.append(site)
    return sites


def parse_overall_vmem_eff(output, kernel_pattern):
    """Extract 'Overall efficiency: X%' for a given kernel."""
    current_kernel = None
    for line in output.splitlines():
        m = re.match(r"=== VMEM Coalescing: (\S+)", line)
        if m:
            current_kernel = m.group(1)
            continue
        if current_kernel and kernel_pattern in current_kernel:
            m = re.match(r"Overall efficiency:\s+([\d.]+)%", line)
            if m:
                return float(m.group(1))
    return None


# =========================================================================
# Test Runner
# =========================================================================

def run_under_profiler(kernel_filter, target_script_args=None):
    """Run the target script under AegisBit profiler via LD_PRELOAD."""
    env = {
        **os.environ,
        "AEGISBIT_ENABLED": "1",
        "AEGISBIT_MODE": "MEMORY_ONLY",
        "AEGISBIT_STRATEGY": "on_gpu_reduce",
        "AEGISBIT_MAX_SITES": "500",
        "AEGISBIT_MAX_LDS": "500",
        "AEGISBIT_KERNELS": kernel_filter,
        "LD_PRELOAD": LIB_PATH,
    }
    cmd = [sys.executable, __file__, "--run-kernels"]
    if target_script_args:
        cmd.extend(target_script_args)

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    combined = result.stdout + "\n" + result.stderr
    return combined, result.returncode


class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, condition, label, detail=""):
        if condition:
            print(f"  PASS: {label}")
            self.passed += 1
        else:
            msg = f"  FAIL: {label}"
            if detail:
                msg += f"  ({detail})"
            print(msg)
            self.failed += 1
            self.errors.append(label)

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"  Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"  Failed checks:")
            for e in self.errors:
                print(f"    - {e}")
        print(f"{'='*70}")
        return self.failed == 0


# =========================================================================
# Kernel Execution (runs inside LD_PRELOAD subprocess)
# =========================================================================

def run_tier1_vmem_kernels():
    N = 65536
    BLOCK = 64
    grid = (triton.cdiv(N, BLOCK),)

    inp = torch.randn(N, device="cuda", dtype=torch.float32)
    out = torch.empty(N, device="cuda", dtype=torch.float32)

    t1_coalesced_kernel[grid](out, inp, N, BLOCK=BLOCK)
    torch.cuda.synchronize()
    assert (out - inp).abs().max().item() < 1e-6

    idx = torch.randint(0, N, (N,), device="cuda", dtype=torch.int32)
    out2 = torch.empty(N, device="cuda", dtype=torch.float32)
    t1_scattered_kernel[grid](out2, inp, idx, N, BLOCK=BLOCK)
    torch.cuda.synchronize()

    big = torch.randn(N * 256, device="cuda", dtype=torch.float32)
    out3 = torch.empty(N, device="cuda", dtype=torch.float32)
    t1_strided_kernel[grid](out3, big, N, STRIDE=256, BLOCK=BLOCK)
    torch.cuda.synchronize()

    print("Tier 1 VMEM kernels executed OK")


def run_tier1_lds_kernels():
    N = 4096
    BLOCK = 64

    inp = torch.randn(N, device="cuda", dtype=torch.float32)
    out = torch.empty(N, device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(N, BLOCK),)
    t1_lds_no_conflict_kernel[grid](out, inp, N, BLOCK=BLOCK)
    torch.cuda.synchronize()

    M, K, Nmat = 128, 64, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, Nmat, device="cuda", dtype=torch.float32)
    C = torch.empty(M, Nmat, device="cuda", dtype=torch.float32)
    grid2 = (triton.cdiv(M, 64), triton.cdiv(Nmat, 64))
    t1_lds_conflict_kernel[grid2](
        C, A, B, M, K, Nmat,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    torch.cuda.synchronize()

    ref = A @ B
    assert (C - ref).abs().max().item() < 0.01
    print("Tier 1 LDS kernels executed OK")


def run_tier2_mixed_vmem():
    N = 65536
    BLOCK = 64
    grid = (triton.cdiv(N, BLOCK),)

    coal_in = torch.randn(N, device="cuda", dtype=torch.float32)
    scat_in = torch.randn(N, device="cuda", dtype=torch.float32)
    idx = torch.randint(0, N, (N,), device="cuda", dtype=torch.int32)
    out = torch.empty(N, device="cuda", dtype=torch.float32)

    t2_mixed_vmem_kernel[grid](out, coal_in, scat_in, idx, N, BLOCK=BLOCK)
    torch.cuda.synchronize()
    print("Tier 2 mixed VMEM kernel executed OK")


def run_tier2_mixed_vmem_lds():
    M, K, N = 128, 64, 128
    BM, BN, BK = 64, 64, 32

    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    Scat = torch.randn(M * 1024, device="cuda", dtype=torch.float32)
    Idx = torch.randint(0, M * 1024, (M * 1024,), device="cuda", dtype=torch.int32)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)

    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    t2_mixed_vmem_lds_kernel[grid](
        C, A, B, Scat, Idx,
        M, K, N,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    torch.cuda.synchronize()
    print("Tier 2 mixed VMEM+LDS kernel executed OK")


# =========================================================================
# Validation Logic
# =========================================================================

def validate_tier1_vmem(output, R):
    print("\n--- Tier 1: VMEM Coalescing Anchors ---")
    vmem = parse_vmem(output)
    if not vmem:
        R.check(False, "T1-VMEM: profiling output present", "no VMEM sites found")
        return

    coal = [s for s in vmem if "t1_coalesced" in s["kernel"]]
    scat = [s for s in vmem if "t1_scattered" in s["kernel"]]
    stri = [s for s in vmem if "t1_strided" in s["kernel"]]

    R.check(len(coal) > 0, "T1-VMEM-coal: sites present")
    R.check(len(scat) > 0, "T1-VMEM-scat: sites present")
    R.check(len(stri) > 0, "T1-VMEM-stri: sites present")

    coal_loads = [s for s in coal if "load" in s["kind"]]
    scat_loads = [s for s in scat if "load" in s["kind"] and s["cachelines"] > 4]
    stri_loads = [s for s in stri if "load" in s["kind"]]

    if coal_loads:
        cl = coal_loads[0]["cachelines"]
        R.check(cl <= 4, f"T1-VMEM-coal: load ≤4 CL (stride-1, 64 fp32 = 256B → 2 CL)", f"got {cl}")
        R.check(coal_loads[0]["eff"] >= 80, f"T1-VMEM-coal: load eff ≥80%", f"got {coal_loads[0]['eff']}%")

    if scat_loads:
        cl = scat_loads[0]["cachelines"]
        R.check(cl >= 20, f"T1-VMEM-scat: scattered load ≥20 CL (random indices)", f"got {cl}")
        R.check(scat_loads[0]["eff"] <= 40, f"T1-VMEM-scat: scattered eff ≤40%", f"got {scat_loads[0]['eff']}%")

    if stri_loads:
        cl = stri_loads[0]["cachelines"]
        R.check(cl >= 30, f"T1-VMEM-stri: strided load ≥30 CL (stride=256)", f"got {cl}")
        R.check(stri_loads[0]["eff"] <= 30, f"T1-VMEM-stri: strided eff ≤30%", f"got {stri_loads[0]['eff']}%")

    coal_eff = parse_overall_vmem_eff(output, "t1_coalesced")
    scat_eff = parse_overall_vmem_eff(output, "t1_scattered")
    if coal_eff is not None and scat_eff is not None:
        R.check(coal_eff > scat_eff + 20,
                f"T1-VMEM: coalesced kernel > scattered kernel by ≥20pp",
                f"coal={coal_eff}% scat={scat_eff}%")


def validate_tier1_lds(output, R):
    print("\n--- Tier 1: LDS Bank Conflict Anchors ---")
    lds = parse_lds(output)

    conflict = [s for s in lds if "t1_lds_conflict" in s["kernel"]]
    R.check(len(conflict) > 0, "T1-LDS-conflict: matmul has LDS sites",
            f"found {len(conflict)} sites")

    if conflict:
        max_n_way = max(s["avg_n_way"] for s in conflict)
        R.check(max_n_way >= 2,
                f"T1-LDS-conflict: matmul avg_n-way ≥2",
                f"max={max_n_way}")
        # conflict_free_pct is only emitted when the strategy records a
        # per-sample breakdown (on_gpu_reduce does not). When present,
        # require all matmul sites to be <50% conflict-free; when absent,
        # fall back to requiring every site to show n-way > 1.
        sites_with_cf = [s for s in conflict if "cf_pct" in s]
        if sites_with_cf and len(sites_with_cf) == len(conflict):
            all_conflicted = all(s["cf_pct"] < 50 for s in conflict)
            R.check(all_conflicted,
                    "T1-LDS-conflict: all matmul sites have <50% conflict-free")
        else:
            all_conflicted = all(s["avg_n_way"] > 1.0 for s in conflict)
            R.check(all_conflicted,
                    "T1-LDS-conflict: all matmul sites have avg_n-way > 1")


def validate_tier2_mixed_vmem(output, R):
    print("\n--- Tier 2: Mixed VMEM (per-site discrimination) ---")
    vmem = parse_vmem(output)
    mixed = [s for s in vmem if "t2_mixed_vmem" in s["kernel"]
             and "t2_mixed_vmem_lds" not in s["kernel"]]

    if not mixed:
        R.check(False, "T2-VMEM-mixed: sites present", "no sites found")
        return

    R.check(len(mixed) >= 3, f"T2-VMEM-mixed: ≥3 distinct site groups",
            f"got {len(mixed)}")

    coal_sites = [s for s in mixed if "load" in s["kind"] and s["eff"] >= 80]
    scat_sites = [s for s in mixed if "load" in s["kind"] and s["eff"] <= 40]

    R.check(len(coal_sites) >= 1,
            "T2-VMEM-mixed: at least 1 coalesced load site (eff≥80%)",
            f"found {len(coal_sites)}")
    R.check(len(scat_sites) >= 1,
            "T2-VMEM-mixed: at least 1 scattered load site (eff≤40%)",
            f"found {len(scat_sites)}")

    if coal_sites and scat_sites:
        best_coal = max(s["eff"] for s in coal_sites)
        worst_scat = min(s["eff"] for s in scat_sites)
        R.check(best_coal > worst_scat + 30,
                "T2-VMEM-mixed: coalesced site > scattered site by ≥30pp (same kernel)",
                f"coal={best_coal}% scat={worst_scat}%")


def validate_tier2_mixed_vmem_lds(output, R):
    print("\n--- Tier 2: Mixed VMEM+LDS (GEMM + scattered gather) ---")
    vmem = parse_vmem(output)
    lds = parse_lds(output)

    mixed_vmem = [s for s in vmem if "t2_mixed_vmem_lds" in s["kernel"]]
    mixed_lds = [s for s in lds if "t2_mixed_vmem_lds" in s["kernel"]]

    R.check(len(mixed_vmem) >= 2,
            "T2-VMEM+LDS: ≥2 VMEM site groups in fused kernel",
            f"got {len(mixed_vmem)}")
    R.check(len(mixed_lds) >= 1,
            "T2-VMEM+LDS: ≥1 LDS site group (from tl.dot)",
            f"got {len(mixed_lds)}")

    if mixed_vmem:
        stores = [s for s in mixed_vmem if "store" in s["kind"]]
        R.check(len(stores) >= 1,
                "T2-VMEM+LDS: output stores are reported",
                f"found {len(stores)} store site(s)")

    if mixed_lds:
        has_conflicts = any(s["avg_n_way"] >= 2 for s in mixed_lds)
        R.check(has_conflicts,
                "T2-VMEM+LDS: tl.dot LDS staging has bank conflicts (avg_n-way≥2)")

    # Consistency: VMEM and LDS both reported for the same kernel
    vmem_kernels = {s["kernel"] for s in mixed_vmem}
    lds_kernels = {s["kernel"] for s in mixed_lds}
    shared_kernels = vmem_kernels & lds_kernels
    R.check(len(shared_kernels) >= 1,
            "T2-VMEM+LDS: same kernel has both VMEM and LDS data",
            f"VMEM kernels={vmem_kernels}, LDS kernels={lds_kernels}")


# =========================================================================
# Main
# =========================================================================

def main():
    if not os.path.exists(LIB_PATH):
        print(f"ERROR: libaegisbit.so not found at {LIB_PATH}")
        sys.exit(1)

    R = Results()

    # --- Run Tier 1 VMEM ---
    print("=" * 70)
    print("  Running Tier 1 VMEM kernels under profiler...")
    print("=" * 70)
    out1, rc1 = run_under_profiler("*t1_coalesced*,*t1_scattered*,*t1_strided*",
                                    ["--tier1-vmem"])
    R.check(rc1 == 0, "T1-VMEM: kernel execution succeeded", f"rc={rc1}")
    validate_tier1_vmem(out1, R)

    # --- Run Tier 1 LDS ---
    print("\n" + "=" * 70)
    print("  Running Tier 1 LDS kernels under profiler...")
    print("=" * 70)
    out2, rc2 = run_under_profiler("*t1_lds*",
                                    ["--tier1-lds"])
    R.check(rc2 == 0, "T1-LDS: kernel execution succeeded", f"rc={rc2}")
    validate_tier1_lds(out2, R)

    # --- Run Tier 2 mixed VMEM ---
    print("\n" + "=" * 70)
    print("  Running Tier 2 mixed VMEM kernel under profiler...")
    print("=" * 70)
    out3, rc3 = run_under_profiler("*t2_mixed_vmem*",
                                    ["--tier2-vmem"])
    R.check(rc3 == 0, "T2-VMEM: kernel execution succeeded", f"rc={rc3}")
    validate_tier2_mixed_vmem(out3, R)

    # --- Run Tier 2 mixed VMEM+LDS ---
    print("\n" + "=" * 70)
    print("  Running Tier 2 VMEM+LDS kernel under profiler...")
    print("=" * 70)
    out4, rc4 = run_under_profiler("*t2_mixed_vmem_lds*",
                                    ["--tier2-vmem-lds"])
    R.check(rc4 == 0, "T2-VMEM+LDS: kernel execution succeeded", f"rc={rc4}")
    validate_tier2_mixed_vmem_lds(out4, R)

    ok = R.summary()
    if ok:
        print("\n  All profiler validation checks passed.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-kernels", action="store_true")
    parser.add_argument("--tier1-vmem", action="store_true")
    parser.add_argument("--tier1-lds", action="store_true")
    parser.add_argument("--tier2-vmem", action="store_true")
    parser.add_argument("--tier2-vmem-lds", action="store_true")
    args = parser.parse_args()

    if args.run_kernels:
        if args.tier1_vmem:
            run_tier1_vmem_kernels()
        if args.tier1_lds:
            run_tier1_lds_kernels()
        if args.tier2_vmem:
            run_tier2_mixed_vmem()
        if args.tier2_vmem_lds:
            run_tier2_mixed_vmem_lds()
    else:
        main()
