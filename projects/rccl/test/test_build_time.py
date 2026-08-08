#!/usr/bin/env python3
"""Build-time regression test: clean-build RCCL once per GPU target and fail if
any target takes longer than a threshold (default 5 minutes).

Each target is configured and built from scratch in its own throwaway build
directory, mirroring the Release configuration that ``install.sh`` produces:

    cmake --toolchain=toolchain-linux.cmake -DCMAKE_BUILD_TYPE=Release \\
          -DGPU_TARGETS=<target> -DROCM_PATH=$ROCM_PATH -GNinja

The reported time is configure + build wall clock, since that is what a
developer waits through after ``rm -rf build``.

The target list comes from DEFAULT_GPUS in the top-level CMakeLists.txt, so
archs stay in sync as RCCL adds or drops them. The CTest registration passes
that CMake list on the command line; a standalone run parses it out of the file.

Targets the installed compiler cannot build are SKIPPED rather than failed --
the RCCL default arch list runs ahead of released ROCm, so e.g. gfx1250 is not
buildable on ROCm 7.0.x. Pass --strict (or set RCCL_BUILD_TIME_STRICT=1) on a
runner whose toolchain is expected to cover every target.

The threshold is wall clock, so it is only meaningful relative to the machine:
a clean single-arch build is ~50s on a 256-core host but will exceed 5 minutes
on a small runner. Parallelism defaults to the CPU count and is recorded in the
report; use --jobs to pin it when comparing results across machines.

Usage:
    ./test_build_time.py                      # every arch in DEFAULT_GPUS
    ./test_build_time.py --local-gpu          # only this machine's arch (CI gate)
    ./test_build_time.py --targets gfx942     # one target
    ./test_build_time.py --threshold-sec 600 --jobs 32
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import TextIO

HERE: str = os.path.dirname(os.path.abspath(__file__))
RCCL_ROOT: str = os.path.dirname(HERE)
TOOLCHAIN: str = os.path.join(RCCL_ROOT, "toolchain-linux.cmake")
CMAKELISTS: str = os.path.join(RCCL_ROOT, "CMakeLists.txt")

DEFAULT_THRESHOLD_SEC: int = 300

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

# Ordered most- to least-specific; plain CI is last so it only acts as a
# fallback for runners that set nothing else.
CI_MARKERS: tuple[tuple[str, str, str | None], ...] = (
    ("GITHUB_ACTIONS", "GitHub Actions", "GITHUB_RUN_ID"),
    ("SLURM_JOB_ID", "SLURM", "SLURM_JOB_ID"),
    ("JENKINS_URL", "Jenkins", "BUILD_NUMBER"),
    ("TF_BUILD", "Azure Pipelines", "BUILD_BUILDID"),
    ("GITLAB_CI", "GitLab CI", "CI_JOB_ID"),
    ("CI", "CI (unidentified)", None),
)


@dataclass
class TargetResult:
    """Outcome for one GPU target.

    The timings are None when no build ran at all (skipped, or failed in
    --strict), which is what the report distinguishes with "-" columns.
    """

    target: str
    status: str
    note: str
    configure_s: float | None = None
    build_s: float | None = None


def ci_context() -> str | None:
    """Name the CI system if one is detectable, else None.

    Recorded in the report because the threshold is wall clock: a number is
    only comparable against others from the same runner class.
    """
    for var, name, id_var in CI_MARKERS:
        if os.environ.get(var):
            job_id = os.environ.get(id_var or "")
            return "%s %s" % (name, job_id) if job_id else name
    return None


def cpu_count() -> int:
    """CPU count, treating the unknowable case as a single CPU.

    os.cpu_count() is Optional, and both -j and the report need a real number.
    """
    return os.cpu_count() or 1


def targets_from_cmake(path: str = CMAKELISTS) -> list[str]:
    """Read the DEFAULT_GPUS arch list out of the top-level CMakeLists.txt.

    This keeps a standalone run covering exactly the archs RCCL ships for,
    without a second copy of the list to update. The CTest registration passes
    the same CMake list explicitly, so both entry points agree.

    Deliberately reads DEFAULT_GPUS rather than the post-rocm_check_target_ids
    GPU_TARGETS: archs the local compiler cannot build should show up as an
    explicit SKIP here, not silently vanish from the report.
    """
    with open(path) as f:
        match = re.search(r"set\s*\(\s*DEFAULT_GPUS\s+([^)]*)\)", f.read())
    if not match:
        raise RuntimeError("could not find DEFAULT_GPUS in %s" % path)
    targets = re.findall(r"gfx[0-9a-fA-F]+", match.group(1))
    if not targets:
        raise RuntimeError("DEFAULT_GPUS in %s contained no gfx targets" % path)
    return targets


def local_gpu_targets() -> list[str]:
    """Resolve the archs of the GPUs installed in this machine.

    Mirrors what BUILD_LOCAL_GPU_TARGET_ONLY does via rocm_local_targets, so a
    CI gate can time only the arch its runner actually ships rather than the
    whole DEFAULT_GPUS list.
    """
    bundled = os.path.join(rocm_path(), "bin", "rocm_agent_enumerator")
    enumerator = bundled if os.path.exists(bundled) else shutil.which("rocm_agent_enumerator")
    if not enumerator:
        raise RuntimeError("rocm_agent_enumerator not found; pass --targets explicitly")

    out = subprocess.run([enumerator], capture_output=True, text=True).stdout
    # Deduplicate but keep discovery order; gfx000 is the CPU agent.
    targets: list[str] = []
    for arch in re.findall(r"gfx[0-9a-fA-F]+", out):
        if arch != "gfx000" and arch not in targets:
            targets.append(arch)
    if not targets:
        raise RuntimeError("no local AMD GPUs found; pass --targets explicitly")
    return targets


def rocm_path() -> str:
    return os.environ.get("ROCM_PATH", "/opt/rocm")


def _hipcc_accepts(hipcc: str, extra_args: list[str]) -> bool:
    """Run a syntax-only compile of an empty HIP file; True if it succeeds.

    Empty input keeps this well under a second, and invalid target IDs are
    rejected during argument handling rather than compilation.
    """
    with tempfile.TemporaryDirectory() as tmp:
        empty = os.path.join(tmp, "empty.hip")
        open(empty, "w").close()
        return subprocess.run(
            [hipcc, "-x", "hip"] + extra_args + ["-fsyntax-only", empty],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0


def compiler_works(hipcc: str) -> bool:
    """True if the compiler can compile HIP at all, before any arch selection.

    hipcc dispatches to $ROCM_PATH/lib/llvm/bin/clang++, so a wrong ROCM_PATH
    makes every --offload-arch probe fail. Without this check that misreports
    as "no arch is supported" rather than as the configuration error it is.
    """
    return _hipcc_accepts(hipcc, [])


def compiler_supports(target: str, hipcc: str) -> bool:
    """True if the installed compiler accepts --offload-arch=<target>."""
    return _hipcc_accepts(hipcc, ["--offload-arch=" + target])


def pick_generator() -> tuple[str, list[str]]:
    """Prefer Ninja: it parallelizes the per-arch device pipeline far better."""
    if shutil.which("ninja"):
        return "Ninja", ["ninja"]
    return "Unix Makefiles", ["make"]


def build_target(
    target: str, build_dir: str, jobs: int, log_path: str
) -> tuple[float, float, bool]:
    """Clean-configure and build one target. Returns (configure_s, build_s, ok)."""
    shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir)

    generator, build_cmd = pick_generator()
    configure = [
        "cmake",
        "--toolchain=" + TOOLCHAIN,
        "-G", generator,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGPU_TARGETS=" + target,
        "-DROCM_PATH=" + rocm_path(),
        RCCL_ROOT,
    ]

    log: TextIO
    with open(log_path, "w") as log:
        def run(cmd: list[str]) -> tuple[float, int]:
            log.write("\n$ " + " ".join(cmd) + "\n")
            log.flush()
            start = time.monotonic()
            rc = subprocess.call(cmd, cwd=build_dir, stdout=log, stderr=subprocess.STDOUT)
            return time.monotonic() - start, rc

        configure_s, rc = run(configure)
        if rc != 0:
            return configure_s, 0.0, False

        build_s, rc = run(build_cmd + ["-j", str(jobs)])
        return configure_s, build_s, rc == 0


def fmt(seconds: float) -> str:
    return "%d:%05.2f" % (int(seconds // 60), seconds % 60)


def resolve_targets(args: argparse.Namespace) -> tuple[list[str], str]:
    """Pick the target list and report where it came from."""
    if args.targets:
        return args.targets, "--targets"
    if args.local_gpu:
        return local_gpu_targets(), "local GPUs (rocm_agent_enumerator)"
    if os.environ.get("RCCL_BUILD_TIME_TARGETS", "").split():
        return os.environ["RCCL_BUILD_TIME_TARGETS"].split(), "RCCL_BUILD_TIME_TARGETS"
    return targets_from_cmake(), "DEFAULT_GPUS in CMakeLists.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--targets", nargs="+",
                       help="GPU targets to time (default: DEFAULT_GPUS from CMakeLists.txt)")
    which.add_argument("--local-gpu", action="store_true",
                       help="time only the arch(es) of the GPUs in this machine")
    parser.add_argument("--threshold-sec", type=float,
                        default=float(os.environ.get("RCCL_BUILD_TIME_THRESHOLD_SEC", DEFAULT_THRESHOLD_SEC)))
    parser.add_argument("--jobs", type=int,
                        default=int(os.environ.get("RCCL_BUILD_TIME_JOBS", 0)) or cpu_count())
    parser.add_argument("--build-root", default=os.environ.get("RCCL_BUILD_TIME_ROOT"),
                        help="where to put scratch build dirs (default: a temp dir)")
    parser.add_argument("--keep", action="store_true", help="keep build dirs for inspection")
    parser.add_argument("--strict", action="store_true",
                        default=os.environ.get("RCCL_BUILD_TIME_STRICT") == "1",
                        help="fail, rather than skip, targets the compiler cannot build")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets, targets_from = resolve_targets(args)

    bundled = os.path.join(rocm_path(), "bin", "hipcc")
    hipcc = bundled if os.path.exists(bundled) else shutil.which("hipcc")
    if not hipcc:
        print("ERROR: hipcc not found; set ROCM_PATH or put hipcc on PATH", file=sys.stderr)
        return 1
    if not compiler_works(hipcc):
        print("ERROR: %s cannot compile a trivial HIP file, so no target can be "
              "measured.\n       Check ROCM_PATH (currently %r) and the toolchain "
              "install." % (hipcc, rocm_path()), file=sys.stderr)
        return 1

    build_root: str = args.build_root or tempfile.mkdtemp(prefix="rccl-build-time-")
    os.makedirs(build_root, exist_ok=True)
    generator, _ = pick_generator()

    ci = ci_context()
    print("RCCL build-time regression test")
    print("  context    : %s" % (ci if ci else "local (no CI environment detected)"))
    print("  source     : %s" % RCCL_ROOT)
    print("  ROCM_PATH  : %s" % rocm_path())
    print("  generator  : %s" % generator)
    print("  jobs       : %d (of %d CPUs)" % (args.jobs, cpu_count()))
    print("  threshold  : %s (%.0fs) per target, configure + build" %
          (fmt(args.threshold_sec), args.threshold_sec))
    print("  build root : %s" % build_root)
    print("  targets    : %s (from %s)" % (" ".join(targets), targets_from))
    print()

    results: list[TargetResult] = []
    for target in targets:
        if not compiler_supports(target, hipcc):
            reason = "compiler does not support --offload-arch=%s" % target
            status = FAIL if args.strict else SKIP
            print("[%s] %s - %s%s" % (target, status, reason,
                                      " (--strict)" if args.strict else ""))
            results.append(TargetResult(target, status, reason))
            continue

        build_dir = os.path.join(build_root, target)
        log_path = os.path.join(build_root, "%s.log" % target)
        print("[%s] building ..." % target, flush=True)

        configure_s, build_s, ok = build_target(target, build_dir, args.jobs, log_path)
        total_s = configure_s + build_s

        if not ok:
            status, note = FAIL, "build failed, see %s" % log_path
        elif total_s > args.threshold_sec:
            status, note = FAIL, "exceeded threshold by %s" % fmt(total_s - args.threshold_sec)
        else:
            status, note = PASS, "%s under threshold" % fmt(args.threshold_sec - total_s)

        print("[%s] %s - %s (configure %s + build %s) - %s" %
              (target, status, fmt(total_s), fmt(configure_s), fmt(build_s), note))
        results.append(TargetResult(target, status, note, configure_s, build_s))

        if not args.keep:
            shutil.rmtree(build_dir, ignore_errors=True)

    print()
    print("%-10s %10s %10s %10s  %s" % ("TARGET", "CONFIGURE", "BUILD", "TOTAL", "RESULT"))
    for r in results:
        if r.configure_s is None or r.build_s is None:
            print("%-10s %10s %10s %10s  %s (%s)" % (r.target, "-", "-", "-", r.status, r.note))
        else:
            print("%-10s %10s %10s %10s  %s" %
                  (r.target, fmt(r.configure_s), fmt(r.build_s),
                   fmt(r.configure_s + r.build_s), r.status))

    failed = [r.target for r in results if r.status == FAIL]
    if failed:
        print("\nFAILED: %s" % ", ".join(failed))
        return 1

    built = [r.target for r in results if r.status == PASS]
    if not built:
        print("\nNo targets were buildable with this toolchain; nothing was measured.")
        return 1

    print("\nPASSED: %s" % ", ".join(built))
    return 0


if __name__ == "__main__":
    sys.exit(main())
