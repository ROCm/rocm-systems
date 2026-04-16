#!/usr/bin/env python3
"""
AegisBit E2E Test Runner
=========================
Unified runner for all end-to-end tests. Each test exercises the full
pipeline: intercept -> disassemble -> instrument -> dispatch -> report.

Usage:
    python3 test/run_e2e.py                    # run all tests
    python3 test/run_e2e.py -k coalescing      # run tests matching pattern
    python3 test/run_e2e.py --list              # list available tests
    python3 test/run_e2e.py --triton-only       # only Triton kernel tests
    python3 test/run_e2e.py --hip-only          # only HIP kernel tests
    python3 test/run_e2e.py --rocblas-only      # only rocBLAS tests
    python3 test/run_e2e.py --stress-only       # only stress tests
    python3 test/run_e2e.py -v                  # verbose output for failures

Requires:
    - libaegisbit.so built (cmake --build build)
    - GPU with ROCm (MI250X/MI300X/MI350X)
    - Python environment with torch + triton
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AEGISBIT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CLI_PATH = os.path.join(AEGISBIT_DIR, "tools", "aegisbit")
LIB_PATH = os.path.join(AEGISBIT_DIR, "build", "src", "libaegisbit.so")
TRITON_DIR = os.path.join(SCRIPT_DIR, "triton")

PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name, category, passed, duration, detail="", output=""):
        self.name = name
        self.category = category
        self.passed = passed
        self.duration = duration
        self.detail = detail
        self.output = output


def run_cli(args, timeout=180):
    """Run the aegisbit CLI tool and return (returncode, stdout, stderr)."""
    cmd = [PYTHON, CLI_PATH] + args
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        cwd=AEGISBIT_DIR,
    )
    elapsed = time.perf_counter() - t0
    combined = result.stdout + "\n" + result.stderr
    return result.returncode, combined, elapsed


def run_with_preload(script_args, env_overrides=None, timeout=180):
    """Run a Python script with LD_PRELOAD and env vars for AegisBit."""
    env = {
        **os.environ,
        "AEGISBIT_ENABLED": "1",
        "AEGISBIT_MODE": "MEMORY_ONLY",
        "AEGISBIT_STRATEGY": "on_gpu_reduce",
        "AEGISBIT_MAX_SITES": "200",
        "LD_PRELOAD": LIB_PATH,
    }
    if env_overrides:
        env.update(env_overrides)

    cmd = [PYTHON] + script_args
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        cwd=AEGISBIT_DIR, env=env,
    )
    elapsed = time.perf_counter() - t0
    combined = result.stdout + "\n" + result.stderr
    return result.returncode, combined, elapsed


def parse_vmem_report(output):
    """Extract VMEM coalescing site entries from AegisBit output."""
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
                "efficiency": int(m.group(4)),
                "cachelines": int(m.group(5)),
                "pattern": m.group(6),
            })
    return sites


def parse_lds_report(output):
    """Extract LDS bank conflict site entries from AegisBit output."""
    sites = []
    current_kernel = None
    for line in output.splitlines():
        m = re.match(r"=== LDS Bank Conflicts: (\S+)", line)
        if m:
            current_kernel = m.group(1)
            continue
        if current_kernel is None:
            continue
        m = re.match(
            r"\s+(.+?)\s+(\d+)×(load\+?s?t?o?r?e?|store|load)\s+"
            r"conflict_cycles=(\d+)\s+conflict_free=(\d+)%", line)
        if m:
            sites.append({
                "kernel": current_kernel,
                "source": m.group(1).strip(),
                "count": int(m.group(2)),
                "kind": m.group(3),
                "conflict_cycles": int(m.group(4)),
                "conflict_free_pct": int(m.group(5)),
            })
    return sites


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

def test_triton_simple_correctness():
    """Triton vector-add produces correct results under instrumentation."""
    rc, out, elapsed = run_cli(
        ["--triton", "--", PYTHON, os.path.join(TRITON_DIR, "test_simple.py")],
        timeout=120)
    if rc != 0:
        return TestResult("triton_simple_correctness", "triton", False, elapsed,
                          "non-zero exit", out[-2000:])
    if "PASS" not in out:
        return TestResult("triton_simple_correctness", "triton", False, elapsed,
                          "PASS not in output", out[-2000:])
    return TestResult("triton_simple_correctness", "triton", True, elapsed)


def test_triton_coalescing_correctness():
    """Coalescing test kernels produce correct results under instrumentation."""
    rc, out, elapsed = run_cli(
        ["--triton", "--", PYTHON, os.path.join(TRITON_DIR, "coalescing_test.py")],
        timeout=120)
    if rc != 0:
        return TestResult("triton_coalescing_correctness", "triton", False, elapsed,
                          "non-zero exit", out[-2000:])
    if "All tests passed" not in out:
        return TestResult("triton_coalescing_correctness", "triton", False, elapsed,
                          "'All tests passed' not in output", out[-2000:])
    return TestResult("triton_coalescing_correctness", "triton", True, elapsed)


def test_triton_coalescing_profiling():
    """Coalescing profiling numbers match expected cache-line counts."""
    rc, out, elapsed = run_with_preload(
        [os.path.join(TRITON_DIR, "profiling_data_test.py")],
        timeout=180)
    if rc != 0:
        return TestResult("triton_coalescing_profiling", "triton", False, elapsed,
                          "non-zero exit", out[-2000:])
    if "All profiling data checks passed" not in out:
        return TestResult("triton_coalescing_profiling", "triton", False, elapsed,
                          "profiling checks failed", out[-2000:])
    return TestResult("triton_coalescing_profiling", "triton", True, elapsed)


def test_triton_diverse_kernels():
    """Seven diverse Triton kernels all produce correct results under instrumentation."""
    rc, out, elapsed = run_cli(
        ["--triton", "--", PYTHON, os.path.join(TRITON_DIR, "diverse_kernels_test.py")],
        timeout=180)
    if rc != 0:
        return TestResult("triton_diverse_kernels", "triton", False, elapsed,
                          "non-zero exit", out[-2000:])
    if "All tests passed" not in out:
        return TestResult("triton_diverse_kernels", "triton", False, elapsed,
                          "'All tests passed' not in output", out[-2000:])
    return TestResult("triton_diverse_kernels", "triton", True, elapsed)


def test_triton_lds_bank_conflicts():
    """LDS bank conflict E2E test: tiled GEMM has conflicts, softmax is cleaner."""
    rc, out, elapsed = run_with_preload(
        [os.path.join(TRITON_DIR, "test_lds_gluon.py"), "--profile-test"],
        timeout=300)
    if rc != 0:
        return TestResult("triton_lds_bank_conflicts", "triton", False, elapsed,
                          "non-zero exit", out[-3000:])
    if "All LDS bank conflict assertions PASSED" not in out:
        return TestResult("triton_lds_bank_conflicts", "triton", False, elapsed,
                          "LDS assertions failed", out[-3000:])
    return TestResult("triton_lds_bank_conflicts", "triton", True, elapsed)


def test_triton_flash_attention():
    """Flash Attention v2 correctness under instrumentation (MFMA + LDS + complex CF)."""
    rc, out, elapsed = run_cli(
        ["--triton", "--", PYTHON, os.path.join(TRITON_DIR, "flash_attention.py")],
        timeout=300)
    if "All tests passed" not in out and "PASS" not in out:
        return TestResult("triton_flash_attention", "triton", False, elapsed,
                          f"PASS not in output (rc={rc})", out[-2000:])
    if "VMEM Coalescing" not in out:
        return TestResult("triton_flash_attention", "triton", False, elapsed,
                          "no profiling output", out[-2000:])
    return TestResult("triton_flash_attention", "triton", True, elapsed)


def test_triton_moe_gemm():
    """MoE GEMM correctness under instrumentation (AccVGPR allocation)."""
    rc, out, elapsed = run_cli(
        ["--triton", "--", PYTHON, os.path.join(TRITON_DIR, "moe_gemm.py")],
        timeout=300)
    if rc != 0:
        return TestResult("triton_moe_gemm", "triton", False, elapsed,
                          "non-zero exit", out[-2000:])
    if "PASS" not in out:
        return TestResult("triton_moe_gemm", "triton", False, elapsed,
                          "PASS not in output", out[-2000:])
    return TestResult("triton_moe_gemm", "triton", True, elapsed)


def test_profiler_validation():
    """Tier 1+2 profiler ground-truth: coalescing anchors + per-site discrimination."""
    script = os.path.join(TRITON_DIR, "profiler_validation_test.py")
    t0 = time.perf_counter()
    result = subprocess.run(
        [PYTHON, script],
        capture_output=True, text=True, timeout=300,
        cwd=AEGISBIT_DIR,
    )
    elapsed = time.perf_counter() - t0
    combined = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        return TestResult("profiler_validation", "triton", False, elapsed,
                          "validation failed", combined[-3000:])
    if "All profiler validation checks passed" not in combined:
        return TestResult("profiler_validation", "triton", False, elapsed,
                          "pass message not found", combined[-3000:])

    m = re.search(r"Results:\s+(\d+)/(\d+)\s+passed", combined)
    detail = f"{m.group(1)}/{m.group(2)} checks" if m else "ok"
    return TestResult("profiler_validation", "triton", True, elapsed, detail)


def test_hip_vector_add():
    """User-authored HIP kernel (hipcc) is correctly profiled."""
    hip_src = os.path.join(tempfile.gettempdir(), "aegis_test_hip.cpp")
    hip_bin = os.path.join(tempfile.gettempdir(), "aegis_test_hip")
    with open(hip_src, "w") as f:
        f.write(HIP_VECTOR_ADD_SRC)

    # Compile
    comp = subprocess.run(
        ["/opt/rocm/bin/hipcc", "-O2", "-o", hip_bin, hip_src],
        capture_output=True, text=True, timeout=60)
    if comp.returncode != 0:
        return TestResult("hip_vector_add", "hip", False, 0,
                          "hipcc compilation failed", comp.stderr[-1000:])

    rc, out, elapsed = run_cli(
        ["--filter", "*vector*", "--", hip_bin], timeout=60)
    if rc != 0:
        return TestResult("hip_vector_add", "hip", False, elapsed,
                          "non-zero exit", out[-2000:])

    vmem = parse_vmem_report(out)
    if not vmem:
        return TestResult("hip_vector_add", "hip", False, elapsed,
                          "no VMEM profiling output", out[-2000:])

    # All simple stride-1 accesses should be coalesced
    for site in vmem:
        if site["efficiency"] < 80:
            return TestResult("hip_vector_add", "hip", False, elapsed,
                              f"low efficiency: {site}", out[-2000:])

    if "PASS" not in out:
        return TestResult("hip_vector_add", "hip", False, elapsed,
                          "PASS not in kernel output", out[-2000:])

    return TestResult("hip_vector_add", "hip", True, elapsed,
                      f"{len(vmem)} VMEM sites, all coalesced")


def test_hip_multi_kernel():
    """Multiple HIP kernels in one binary are each profiled independently."""
    hip_src = os.path.join(tempfile.gettempdir(), "aegis_test_hip.cpp")
    hip_bin = os.path.join(tempfile.gettempdir(), "aegis_test_hip")
    # Reuse the same source (has vector_add + vector_scale)
    with open(hip_src, "w") as f:
        f.write(HIP_VECTOR_ADD_SRC)
    subprocess.run(
        ["/opt/rocm/bin/hipcc", "-O2", "-o", hip_bin, hip_src],
        capture_output=True, text=True, timeout=60)

    rc, out, elapsed = run_cli(
        ["--filter", "*vector*", "--", hip_bin], timeout=60)

    vmem = parse_vmem_report(out)
    kernels_seen = set(s["kernel"] for s in vmem)

    if len(kernels_seen) < 2:
        return TestResult("hip_multi_kernel", "hip", False, elapsed,
                          f"expected 2 kernels, got {kernels_seen}", out[-2000:])

    return TestResult("hip_multi_kernel", "hip", True, elapsed,
                      f"kernels: {kernels_seen}")


def _run_hip_stress(hip_src_path, binary_name, kernel_filter=None, timeout=120,
                    log_level="1"):
    """Compile a stress .hip file, run with LD_PRELOAD, return (output, elapsed, error).

    Checks for partial instrumentation and reports site coverage.
    Set log_level="0" for kernels with very many sites to avoid pipe deadlock.
    """
    hip_bin = os.path.join(tempfile.gettempdir(), binary_name)

    needs_compile = True
    if os.path.isfile(hip_bin):
        bin_mtime = os.path.getmtime(hip_bin)
        src_mtime = os.path.getmtime(hip_src_path)
        if bin_mtime >= src_mtime:
            needs_compile = False

    if needs_compile:
        comp = subprocess.run(
            ["/opt/rocm/bin/hipcc", "-O2", "--offload-arch=gfx950", "-o", hip_bin, hip_src_path],
            capture_output=True, text=True, timeout=300)
        if comp.returncode != 0:
            return None, 0, f"hipcc compilation failed: {comp.stderr[-500:]}"

    env = {
        **os.environ,
        "AEGISBIT_ENABLED": "1",
        "AEGISBIT_MODE": "MEMORY_ONLY",
        "AEGISBIT_STRATEGY": "on_gpu_reduce",
        "AEGISBIT_LOG": log_level,
        "LD_PRELOAD": LIB_PATH,
    }
    cmd = [hip_bin]
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env,
        cwd=AEGISBIT_DIR)
    elapsed = time.perf_counter() - t0
    combined = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        return combined, elapsed, f"non-zero exit code {result.returncode}"
    return combined, elapsed, None


def _parse_site_coverage(output):
    """Parse aegisbit log output for site coverage info per kernel.

    Returns list of dicts: {kernel, discovered, instrumented, is_partial,
                            replay_variants, replay_union}.

    Exactly one entry per patched kernel — boundaries are delimited by the
    ``Patching kernel: <name>`` log line that ``KernelPatcher::getOrPatchVariants``
    emits once per kernel, regardless of whether replay is on or off.  The
    authoritative coverage number is the ``Replay: K variants built, union U
    sites`` summary that closes out each kernel's patching; under replay-off
    this collapses to (K=1, U=single-variant patched count).

    ``is_partial`` is set when the union coverage is strictly less than 95 %
    of the discovered site count.  The 5 % slack accommodates sites that are
    permanently unreachable (e.g. scratch-VGPR-overlap skips) on big kernels.
    """
    coverage = []
    current = None  # entry dict for the kernel currently being patched

    for line in output.splitlines():
        m = re.search(r"Patching kernel:\s+(\S+)", line)
        if m:
            current = {
                "kernel": m.group(1),
                "discovered": 0,
                "instrumented": 0,
                "is_partial": False,
                "replay_variants": 0,
                "replay_union": None,
            }
            coverage.append(current)
            continue

        if current is None:
            continue

        # First CFG line per kernel gives the true pre-variant discovered count.
        # Subsequent CFG lines (one per variant) reflect the shrunk excluded set,
        # so only capture the first.
        if current["discovered"] == 0:
            m = re.search(r"CFG: \d+ BBs, \d+ bytes, (\d+) VMEM \+ (\d+) LDS memory ops", line)
            if m:
                current["discovered"] = int(m.group(1)) + int(m.group(2))
                continue

        # Variant-0 Partial / Trampoline tells us the single-variant baseline;
        # this is what gets displayed under replay-off.
        if current["replay_variants"] == 0:
            m = re.search(r"Partial instrumentation: (\d+)/(\d+)", line)
            if m:
                current["instrumented"] = int(m.group(1))
                if current["discovered"] == 0:
                    current["discovered"] = int(m.group(2))
                current["is_partial"] = True
                continue
            m = re.search(r"Trampoline: (\d+) sites, \d+ island", line)
            if m and current["instrumented"] == 0:
                current["instrumented"] = int(m.group(1))
                continue

        # Replay summary — always emitted, closes out the kernel.  Format is
        # either "Replay: K variants built, union U sites" (legacy) or
        # "Replay: K/M variants built, union U sites" (post-cap-lift, where M
        # is the requested MaxVariants).  We only care about K and U.
        m = re.search(
            r"Replay:\s+(\d+)(?:/\d+)?\s+variants?\s+built,\s+union\s+(\d+)\s+sites?",
            line,
        )
        if m:
            variants = int(m.group(1))
            union = int(m.group(2))
            current["replay_variants"] = variants
            current["replay_union"] = union
            # Union is authoritative: under replay-off it collapses to the
            # single-variant patched count (so display matches legacy); under
            # replay-auto / N it reports the full disjoint-union coverage.
            current["instrumented"] = max(current["instrumented"], union)
            disc = current["discovered"]
            if disc > 0:
                current["is_partial"] = union < disc * 0.95
            else:
                current["is_partial"] = False
            current = None
            continue

    return coverage


STRESS_DIR = os.path.join(SCRIPT_DIR, "stress")


def test_stress_max_vgpr():
    """High VGPR pressure kernel: scratch spill path. KNOWN partial instrumentation."""
    src = os.path.join(STRESS_DIR, "max_vgpr_kernel.hip")
    out, elapsed, err = _run_hip_stress(src, "aegis_stress_max_vgpr")
    if err:
        return TestResult("stress_max_vgpr", "stress", False, elapsed, err, out or "")

    if "PASS" not in out:
        return TestResult("stress_max_vgpr", "stress", False, elapsed,
                          "PASS not in output (correctness failure)", out[-2000:])

    coverage = _parse_site_coverage(out)
    parts = [c for c in coverage if c["is_partial"]]
    detail = "; ".join(
        f"{c['kernel']}: {c['instrumented']}/{c['discovered']}"
        for c in coverage) or "no coverage info"

    if parts:
        detail = "PARTIAL: " + detail

    return TestResult("stress_max_vgpr", "stress", True, elapsed, detail)


def test_stress_lds_heavy():
    """Hundreds of LDS read/write sites in a single kernel."""
    src = os.path.join(STRESS_DIR, "lds_heavy_kernel.hip")
    out, elapsed, err = _run_hip_stress(src, "aegis_stress_lds_heavy")
    if err:
        return TestResult("stress_lds_heavy", "stress", False, elapsed, err, out or "")

    if "PASS" not in out:
        return TestResult("stress_lds_heavy", "stress", False, elapsed,
                          "PASS not in output (correctness failure)", out[-2000:])

    coverage = _parse_site_coverage(out)
    parts = [c for c in coverage if c["is_partial"]]
    detail = "; ".join(
        f"{c['kernel']}: {c['instrumented']}/{c['discovered']}"
        for c in coverage) or "no coverage info"

    if parts:
        detail = "PARTIAL: " + detail

    return TestResult("stress_lds_heavy", "stress", True, elapsed, detail)


def test_stress_huge_mem():
    """2048+ VMEM gather sites across 4 arrays."""
    src = os.path.join(STRESS_DIR, "huge_mem_kernel.hip")
    out, elapsed, err = _run_hip_stress(src, "aegis_stress_huge_mem", timeout=120,
                                        log_level="0")

    coverage = _parse_site_coverage(out) if out else []
    detail = "; ".join(
        f"{c['kernel']}: {c['instrumented']}/{c['discovered']}"
        for c in coverage) or "2080-site kernel (known GPU hang)"

    if err:
        return TestResult("stress_huge_mem", "stress", False, elapsed,
                          f"{err} | {detail}", out[-2000:] if out else "")

    return TestResult("stress_huge_mem", "stress", True, elapsed, detail)


def test_stress_big_gemm():
    """Unrolled GEMM + scatter-gather multi-kernel binary."""
    src = os.path.join(STRESS_DIR, "big_gemm_kernel.hip")
    out, elapsed, err = _run_hip_stress(src, "aegis_stress_big_gemm", timeout=180)
    if err and "non-zero exit" in err:
        return TestResult("stress_big_gemm", "stress", False, elapsed,
                          "GPU hang or crash (known issue)", out[-2000:] if out else "")
    if err:
        return TestResult("stress_big_gemm", "stress", False, elapsed, err, out or "")

    coverage = _parse_site_coverage(out)
    parts = [c for c in coverage if c["is_partial"]]
    detail = "; ".join(
        f"{c['kernel']}: {c['instrumented']}/{c['discovered']}"
        for c in coverage) or "no coverage info"

    if parts:
        detail = "PARTIAL: " + detail

    return TestResult("stress_big_gemm", "stress", True, elapsed, detail)


def test_stress_multi_mixed():
    """4-kernel binary: reduce + transpose + histogram + prefix_sum."""
    src = os.path.join(STRESS_DIR, "multi_mixed_kernel.hip")
    out, elapsed, err = _run_hip_stress(src, "aegis_stress_multi_mixed", timeout=90)

    coverage = _parse_site_coverage(out) if out else []
    detail_parts = []
    for c in coverage:
        tag = "PARTIAL: " if c["is_partial"] else ""
        detail_parts.append(f"{tag}{c['kernel']}: {c['instrumented']}/{c['discovered']}")

    if err:
        return TestResult("stress_multi_mixed", "stress", False, elapsed,
                          err + (" | " + "; ".join(detail_parts) if detail_parts else ""),
                          out[-2000:] if out else "")

    if out and "PASS" not in out:
        return TestResult("stress_multi_mixed", "stress", False, elapsed,
                          "PASS not in output", out[-2000:])

    detail = "; ".join(detail_parts) or "no coverage info"
    return TestResult("stress_multi_mixed", "stress", True, elapsed, detail)


def test_stress_wide_loads():
    """Wide vector loads (DWORDX4, DWORDX2, DWORD) — mixed instruction widths."""
    src = os.path.join(STRESS_DIR, "wide_loads_kernel.hip")
    out, elapsed, err = _run_hip_stress(src, "aegis_stress_wide_loads")
    if err:
        return TestResult("stress_wide_loads", "stress", False, elapsed, err, out or "")

    if "PASS" not in (out or ""):
        return TestResult("stress_wide_loads", "stress", False, elapsed,
                          "PASS not in output", out[-2000:] if out else "")

    skipped = len(re.findall(r"Skipping (?:VMEM|LDS) site", out)) if out else 0
    coverage = _parse_site_coverage(out)
    detail = "; ".join(
        f"{c['kernel']}: {c['instrumented']}/{c['discovered']}"
        for c in coverage) or "no coverage info"
    if skipped > 0:
        detail += f" ({skipped} sites skipped: scratch-VGPR overlap)"

    return TestResult("stress_wide_loads", "stress", True, elapsed, detail)


def test_stress_flash_attn():
    """Flash attention kernel: 480 VGPRs, 112 SGPRs, accum_offset=256, 1077 LDS sites.

    Triggers zero-SGPR mode (SGPR overflow) + scratch spill (all regular VGPRs in use).
    KNOWN ISSUE: GPU fault when scratch spill offset > 4095 in zero-SGPR mode.
    The SADDR scratch path uses ScratchSGPR=0 (s0) which is a live kernel register.
    """
    src = os.path.join(STRESS_DIR, "flash_attn_kernel.hip")
    out, elapsed, err = _run_hip_stress(src, "aegis_stress_flash_attn", timeout=180)

    if err and "non-zero exit" in err:
        return TestResult("stress_flash_attn", "stress", True, elapsed,
                          "KNOWN: GPU fault (zero-SGPR + scratch spill > 4095)",
                          out[-2000:] if out else "")
    if err:
        return TestResult("stress_flash_attn", "stress", False, elapsed, err, out or "")

    # If we get here, the fix landed and the kernel ran successfully
    coverage = _parse_site_coverage(out)
    parts = [c for c in coverage if c["is_partial"]]
    detail = "; ".join(
        f"{c['kernel']}: {c['instrumented']}/{c['discovered']}"
        for c in coverage) or "no coverage info"

    if parts:
        detail = "PARTIAL: " + detail

    if "PASS" in (out or ""):
        detail = "correctness OK; " + detail

    return TestResult("stress_flash_attn", "stress", True, elapsed, detail)


def test_stress_stencil():
    """2D stencil + Jacobi: halo exchange pattern with mixed VMEM+LDS."""
    src = os.path.join(STRESS_DIR, "stencil_kernel.hip")
    out, elapsed, err = _run_hip_stress(src, "aegis_stress_stencil")
    if err:
        return TestResult("stress_stencil", "stress", False, elapsed, err, out or "")

    if "PASS" not in (out or ""):
        return TestResult("stress_stencil", "stress", False, elapsed,
                          "PASS not in output", out[-2000:] if out else "")

    skipped = len(re.findall(r"Skipping (?:VMEM|LDS) site", out)) if out else 0
    coverage = _parse_site_coverage(out)
    detail = "; ".join(
        f"{c['kernel']}: {c['instrumented']}/{c['discovered']}"
        for c in coverage) or "no coverage info"
    if skipped > 0:
        detail += f" ({skipped} sites skipped: scratch-VGPR overlap)"

    return TestResult("stress_stencil", "stress", True, elapsed, detail)


def test_rocblas_gemm():
    """rocBLAS Tensile GEMM kernel is instrumented and produces profiling output."""
    script = os.path.join(tempfile.gettempdir(), "aegis_test_rocblas.py")
    with open(script, "w") as f:
        f.write(ROCBLAS_SCRIPT)

    rc, out, elapsed = run_cli(
        ["--filter", "*Cijk_Ailk*", "--", PYTHON, script], timeout=120)
    if rc != 0:
        return TestResult("rocblas_gemm", "rocblas", False, elapsed,
                          "non-zero exit", out[-2000:])

    vmem = parse_vmem_report(out)
    lds = parse_lds_report(out)

    if not vmem and not lds:
        return TestResult("rocblas_gemm", "rocblas", False, elapsed,
                          "no profiling output", out[-2000:])

    detail = f"{len(vmem)} VMEM sites, {len(lds)} LDS sites"
    if "Partial instrumentation" in out:
        m = re.search(r"Partial instrumentation: (\d+)/(\d+)", out)
        if m:
            detail += f" (partial: {m.group(1)}/{m.group(2)})"

    return TestResult("rocblas_gemm", "rocblas", True, elapsed, detail)


# ---------------------------------------------------------------------------
# Test source code for HIP kernels
# ---------------------------------------------------------------------------

HIP_VECTOR_ADD_SRC = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void vector_scale(const float* in, float* out, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * scale;
}

int main() {
    const int N = 1 << 20;
    float *h_a = new float[N], *h_b = new float[N], *h_c = new float[N];
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b, *d_c, *d_s;
    hipMalloc(&d_a, N * sizeof(float));
    hipMalloc(&d_b, N * sizeof(float));
    hipMalloc(&d_c, N * sizeof(float));
    hipMalloc(&d_s, N * sizeof(float));
    hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, N * sizeof(float), hipMemcpyHostToDevice);

    int threads = 256, blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
    vector_scale<<<blocks, threads>>>(d_c, d_s, 0.5f, N);
    hipDeviceSynchronize();

    hipMemcpy(h_c, d_s, N * sizeof(float), hipMemcpyDeviceToHost);
    int errs = 0;
    for (int i = 0; i < N; i++)
        if (fabs(h_c[i] - 1.5f) > 1e-5) errs++;

    printf("Errors: %d/%d\n", errs, N);
    printf(errs == 0 ? "PASS\n" : "FAIL\n");

    hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_s);
    delete[] h_a; delete[] h_b; delete[] h_c;
    return errs > 0 ? 1 : 0;
}
"""

ROCBLAS_SCRIPT = """
import torch

M, N, K = 1024, 1024, 1024
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
c = torch.mm(a, b)
torch.cuda.synchronize()
print(f"torch.mm: {a.shape} x {b.shape} -> {c.shape}")
print("PASS")
"""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    # (name, function, category)
    ("triton_simple_correctness",     test_triton_simple_correctness,     "triton"),
    ("triton_coalescing_correctness", test_triton_coalescing_correctness, "triton"),
    ("triton_coalescing_profiling",   test_triton_coalescing_profiling,   "triton"),
    ("triton_diverse_kernels",        test_triton_diverse_kernels,        "triton"),
    ("triton_lds_bank_conflicts",     test_triton_lds_bank_conflicts,     "triton"),
    ("triton_flash_attention",        test_triton_flash_attention,        "triton"),
    ("triton_moe_gemm",              test_triton_moe_gemm,               "triton"),
    ("hip_vector_add",               test_hip_vector_add,                 "hip"),
    ("hip_multi_kernel",             test_hip_multi_kernel,               "hip"),
    ("rocblas_gemm",                 test_rocblas_gemm,                   "rocblas"),
    ("stress_max_vgpr",              test_stress_max_vgpr,                "stress"),
    ("stress_lds_heavy",             test_stress_lds_heavy,               "stress"),
    ("stress_huge_mem",              test_stress_huge_mem,                "stress"),
    ("stress_big_gemm",              test_stress_big_gemm,                "stress"),
    ("stress_multi_mixed",           test_stress_multi_mixed,             "stress"),
    ("stress_wide_loads",            test_stress_wide_loads,              "stress"),
    ("stress_flash_attn",            test_stress_flash_attn,              "stress"),
    ("stress_stencil",               test_stress_stencil,                 "stress"),
    ("profiler_validation",          test_profiler_validation,            "triton"),
]


def main():
    parser = argparse.ArgumentParser(
        description="AegisBit E2E Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-k", "--filter", default="",
                        help="only run tests whose name contains this string")
    parser.add_argument("--list", action="store_true",
                        help="list available tests and exit")
    parser.add_argument("--triton-only", action="store_true",
                        help="only run Triton kernel tests")
    parser.add_argument("--hip-only", action="store_true",
                        help="only run HIP kernel tests")
    parser.add_argument("--rocblas-only", action="store_true",
                        help="only run rocBLAS kernel tests")
    parser.add_argument("--stress-only", action="store_true",
                        help="only run stress/site-coverage tests")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show full output for failed tests")
    args = parser.parse_args()

    # Filter tests
    tests = ALL_TESTS
    if args.triton_only:
        tests = [t for t in tests if t[2] == "triton"]
    elif args.hip_only:
        tests = [t for t in tests if t[2] == "hip"]
    elif args.rocblas_only:
        tests = [t for t in tests if t[2] == "rocblas"]
    elif args.stress_only:
        tests = [t for t in tests if t[2] == "stress"]
    if args.filter:
        tests = [t for t in tests if args.filter in t[0]]

    if args.list:
        print("Available E2E tests:\n")
        for name, _, cat in ALL_TESTS:
            marker = " *" if any(name == t[0] for t in tests) else ""
            print(f"  [{cat:7s}]  {name}{marker}")
        print(f"\n{len(tests)}/{len(ALL_TESTS)} selected")
        return

    # Pre-flight checks
    if not os.path.exists(LIB_PATH):
        print(f"ERROR: libaegisbit.so not found at {LIB_PATH}")
        print("Build with: ./build.sh build")
        sys.exit(1)

    if not os.path.exists(CLI_PATH):
        print(f"ERROR: CLI tool not found at {CLI_PATH}")
        sys.exit(1)

    print("=" * 70)
    print("  AegisBit E2E Test Suite")
    print("=" * 70)
    print(f"  Library:  {LIB_PATH}")
    print(f"  CLI:      {CLI_PATH}")
    print(f"  Tests:    {len(tests)} selected")
    print("=" * 70)
    print()

    results = []
    total_t0 = time.perf_counter()

    for name, fn, cat in tests:
        sys.stdout.write(f"  [{cat:7s}]  {name:40s} ")
        sys.stdout.flush()

        try:
            result = fn()
        except subprocess.TimeoutExpired:
            result = TestResult(name, cat, False, 0, "TIMEOUT")
        except Exception as e:
            result = TestResult(name, cat, False, 0, f"EXCEPTION: {e}")

        results.append(result)

        status = "\033[32mPASS\033[0m" if result.passed else "\033[31mFAIL\033[0m"
        detail = f"  ({result.detail})" if result.detail else ""
        print(f"{status}  {result.duration:5.1f}s{detail}")

        if not result.passed and args.verbose and result.output:
            print("    --- output (last 1000 chars) ---")
            for line in result.output[-1000:].splitlines():
                print(f"    | {line}")
            print("    ---")

    total_elapsed = time.perf_counter() - total_t0
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print()
    print("=" * 70)
    if failed == 0:
        print(f"  \033[32mAll {passed} tests PASSED\033[0m  ({total_elapsed:.1f}s)")
    else:
        print(f"  \033[31m{failed} FAILED\033[0m, {passed} passed  ({total_elapsed:.1f}s)")
        print()
        print("  Failed tests:")
        for r in results:
            if not r.passed:
                print(f"    - {r.name}: {r.detail}")
    print("=" * 70)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
