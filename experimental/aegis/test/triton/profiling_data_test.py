#!/usr/bin/env python3
"""
Profiling data validation — verifies that AegisBit's on_gpu_reduce counters
produce correct cache line counts for kernels with known access patterns.

Runs three kernels with predictable memory patterns, parses the profiling
report from stderr, and checks that cache line counts match expectations.

Run:
    AEGISBIT_ENABLED=1 AEGISBIT_MODE=MEMORY_ONLY \
    AEGISBIT_STRATEGY=on_gpu_reduce AEGISBIT_MAX_SITES=200 \
    AEGISBIT_KERNELS="*coalesced*,*scattered*,*strided*" \
    LD_PRELOAD=build/src/libaegisbit.so \
    python3 test/triton/profiling_data_test.py
"""
import os
import re
import subprocess
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
AEGISBIT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
LIB_PATH = os.path.join(AEGISBIT_DIR, "build", "src", "libaegisbit.so")
TARGET_SCRIPT = os.path.join(THIS_DIR, "coalescing_test.py")
PYTHON = os.path.join(os.path.dirname(sys.executable), "python3")

# Expected cache line counts per kernel per source-line pattern.
# Each entry: (kernel_glob, source_pattern, kind, min_cl, max_cl, label)
EXPECTATIONS = [
    # Coalesced: BLOCK=64, stride-1, 64 fp32 = 256B = 2 cache lines
    ("coalesced_copy", r"tl\.load", "load", 1.5, 2.5, "coalesced load → ~2 CL"),
    ("coalesced_copy", r"tl\.store", "store", 0.5, 1.5, "coalesced store → ~1 CL"),
    # Scattered: random index → nearly every lane hits a different cache line
    ("scattered_copy", r"In_ptr \+ idx", "load", 40, 64, "scattered load → 40-64 CL"),
    # Scattered: the idx load itself is coalesced (stride-1 int32 indices)
    ("scattered_copy", r"Idx_ptr", "load", 1.5, 2.5, "idx load → ~2 CL"),
    # Strided: stride=128 elements × 4B = 512B per lane, every lane in different CL
    ("strided_copy", r"In_ptr \+ src_offs", "load", 55, 64, "strided load → ~64 CL"),
    # Stores are always coalesced in all three kernels
    ("strided_copy", r"tl\.store", "store", 0.5, 1.5, "strided store → ~1 CL"),
]


def parse_profiling_reports(output: str):
    """Parse VMEM Coalescing reports from AegisBit output.

    Returns list of dicts with keys: kernel, source, kind, cachelines, efficiency, pattern.
    """
    reports = []
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
            reports.append({
                "kernel": current_kernel,
                "source": m.group(1).strip(),
                "count": int(m.group(2)),
                "kind": m.group(3),
                "efficiency": int(m.group(4)),
                "cachelines": int(m.group(5)),
                "pattern": m.group(6),
            })

    return reports


def main():
    if not os.path.exists(LIB_PATH):
        print(f"ERROR: libaegisbit.so not found at {LIB_PATH}")
        print("Build with: cmake --build build --target aegisbit_shared")
        sys.exit(1)

    env = {
        **os.environ,
        "AEGISBIT_ENABLED": "1",
        "AEGISBIT_MODE": "MEMORY_ONLY",
        "AEGISBIT_STRATEGY": "on_gpu_reduce",
        "AEGISBIT_MAX_SITES": "200",
        "AEGISBIT_KERNELS": "*coalesced*,*scattered*,*strided*",
        "LD_PRELOAD": LIB_PATH,
    }

    print("Running coalescing_test.py with AegisBit profiling...\n")
    result = subprocess.run(
        [PYTHON, TARGET_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    combined = result.stdout + "\n" + result.stderr
    reports = parse_profiling_reports(combined)

    if not reports:
        print("ERROR: No profiling reports found in output!")
        print("--- stdout ---")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        print("--- stderr ---")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        sys.exit(1)

    print(f"Parsed {len(reports)} site reports:\n")
    for r in reports:
        print(f"  {r['kernel']:30s}  {r['source'][:50]:50s}  "
              f"{r['count']}×{r['kind']:5s}  CL={r['cachelines']:2d}  {r['pattern']}")
    print()

    passed = 0
    failed = 0

    for kernel_pat, src_pat, kind, min_cl, max_cl, label in EXPECTATIONS:
        matches = [
            r for r in reports
            if kernel_pat in r["kernel"]
            and re.search(src_pat, r["source"])
            and kind in r["kind"]
        ]

        if not matches:
            print(f"  FAIL: {label} — no matching site found")
            failed += 1
            continue

        for m in matches:
            cl = m["cachelines"]
            if min_cl <= cl <= max_cl:
                print(f"  PASS: {label}  (got {cl} CL)")
                passed += 1
            else:
                print(f"  FAIL: {label}  (got {cl} CL, expected {min_cl}-{max_cl})")
                failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{passed+failed} passed")
    print(f"{'='*60}")

    if result.returncode != 0:
        print(f"\n  WARNING: coalescing_test.py exited with code {result.returncode}")

    if failed:
        print(f"\n  {failed} check(s) FAILED")
        sys.exit(1)
    else:
        print("\n  All profiling data checks passed.")


if __name__ == "__main__":
    main()
