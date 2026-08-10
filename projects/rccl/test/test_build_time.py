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

--compare-base sidesteps that: it builds the PR's base commit and its head on
this same runner and gates on the relative change, and a ratio cancels out
machine speed. It costs 2*--repeat builds per target instead of one.

Usage:
    ./test_build_time.py                      # every arch in DEFAULT_GPUS
    ./test_build_time.py --local-gpu          # only this machine's arch (CI gate)
    ./test_build_time.py --targets gfx942     # one target
    ./test_build_time.py --threshold-sec 600 --jobs 32
    ./test_build_time.py --local-gpu --compare-base --max-regression-pct 10
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

    proc = subprocess.run([enumerator], capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"rocm_agent_enumerator failed (rc={proc.returncode}): {msg}")
    out = proc.stdout
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


def build_env() -> dict[str, str]:
    """Environment for build subprocesses, with compiler caching disabled.

    A warm ccache would make the second half of an A/B comparison finish
    almost instantly and silently invalidate the result. RCCL's own build does
    not enable ccache, but TheRock CI does, so force it off rather than trust
    the surrounding configuration.
    """
    env = dict(os.environ)
    env["CCACHE_DISABLE"] = "1"
    return env


def build_target(
    target: str, build_dir: str, jobs: int, log_path: str, source_dir: str = RCCL_ROOT
) -> tuple[float, float, bool]:
    """Clean-configure and build one target. Returns (configure_s, build_s, ok).

    source_dir selects which checkout to build, so a single copy of this script
    can time two revisions; each tree supplies its own toolchain file.
    """
    shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir)

    generator, build_cmd = pick_generator()
    configure = [
        "cmake",
        "--toolchain=" + os.path.join(source_dir, "toolchain-linux.cmake"),
        "-G", generator,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGPU_TARGETS=" + target,
        "-DROCM_PATH=" + rocm_path(),
        source_dir,
    ]

    env = build_env()
    log: TextIO
    with open(log_path, "w") as log:
        def run(cmd: list[str]) -> tuple[float, int]:
            log.write("\n$ " + " ".join(cmd) + "\n")
            log.flush()
            start = time.monotonic()
            rc = subprocess.call(cmd, cwd=build_dir, stdout=log, stderr=subprocess.STDOUT,
                                 env=env)
            return time.monotonic() - start, rc

        configure_s, rc = run(configure)
        if rc != 0:
            return configure_s, 0.0, False

        build_s, rc = run(build_cmd + ["-j", str(jobs)])
        return configure_s, build_s, rc == 0


def fmt(seconds: float) -> str:
    return "%d:%05.2f" % (int(seconds // 60), seconds % 60)


def git(*args: str) -> str:
    """Run git in the RCCL source tree and return stripped stdout."""
    return subprocess.run(["git", "-C", RCCL_ROOT, *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def default_base_rev() -> str:
    """The commit a PR branched from: the merge base with the upstream branch."""
    upstream = os.environ.get("RCCL_BUILD_TIME_UPSTREAM", "origin/develop")
    return git("merge-base", upstream, "HEAD")


def rccl_dir_within(worktree: str) -> str:
    """Locate the RCCL project inside a checkout of the enclosing repository.

    RCCL lives at projects/rccl in the rocm-systems monorepo, so a worktree
    root is not itself a configurable source directory. Resolves to the
    worktree root when RCCL is checked out standalone.
    """
    rel = os.path.relpath(RCCL_ROOT, git("rev-parse", "--show-toplevel"))
    return os.path.normpath(os.path.join(worktree, rel))


@dataclass
class Comparison:
    """Base-vs-head timings for one target, in seconds (min across repeats)."""

    target: str
    status: str
    note: str
    base_s: float | None = None
    head_s: float | None = None

    @property
    def delta_pct(self) -> float | None:
        if self.base_s is None or self.head_s is None or self.base_s == 0:
            return None
        return (self.head_s - self.base_s) / self.base_s * 100.0


def time_best_of(
    target: str, source_dir: str, build_root: str, tag: str, jobs: int, repeat: int,
    round_index: int,
) -> tuple[float, bool]:
    """Build one revision once and return (total_s, ok) for this round."""
    build_dir = os.path.join(build_root, "%s-%s-%d" % (target, tag, round_index))
    log_path = os.path.join(build_root, "%s-%s-%d.log" % (target, tag, round_index))
    configure_s, build_s, ok = build_target(target, build_dir, jobs, log_path, source_dir)
    shutil.rmtree(build_dir, ignore_errors=True)
    return configure_s + build_s, ok


def compare_target(
    target: str, head_dir: str, base_dir: str, build_root: str, jobs: int, repeat: int,
) -> tuple[float, float, bool]:
    """Time base and head alternately; return (base_best, head_best, ok).

    Alternating rather than running all of one revision then the other keeps
    slow drift (thermal, noisy neighbours) from landing on just one side. The
    minimum across repeats is the estimator because build-time noise is
    one-sided: interference can only ever make a build slower.
    """
    base_times: list[float] = []
    head_times: list[float] = []
    for i in range(repeat):
        base_s, ok = time_best_of(target, base_dir, build_root, "base", jobs, repeat, i)
        if not ok:
            return 0.0, 0.0, False
        base_times.append(base_s)

        head_s, ok = time_best_of(target, head_dir, build_root, "head", jobs, repeat, i)
        if not ok:
            return 0.0, 0.0, False
        head_times.append(head_s)

        print("    round %d/%d: base %s  head %s" %
              (i + 1, repeat, fmt(base_s), fmt(head_s)), flush=True)

    return min(base_times), min(head_times), True


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

    compare = parser.add_argument_group(
        "base comparison",
        "Time the PR's base commit and its head on this same runner and gate on the "
        "relative change. A ratio cancels out runner speed, so unlike --threshold-sec "
        "it needs no per-machine tuning.")
    compare.add_argument("--compare-base", nargs="?", const="", metavar="REV",
                         help="compare against REV (default: merge-base with origin/develop)")
    compare.add_argument("--max-regression-pct", type=float,
                         default=float(os.environ.get("RCCL_BUILD_TIME_MAX_REGRESSION_PCT", 10.0)),
                         help="fail if head is more than this %% slower than base (default: 10)")
    compare.add_argument("--repeat", type=int,
                         default=int(os.environ.get("RCCL_BUILD_TIME_REPEAT", 2)),
                         help="alternating base/head rounds; the minimum of each wins (default: 2)")
    return parser.parse_args()


def run_comparison(
    args: argparse.Namespace, targets: list[str], hipcc: str, build_root: str,
) -> int:
    """Time base vs head for each target and gate on the relative change."""
    base_rev = args.compare_base or default_base_rev()
    base_sha = git("rev-parse", base_rev)
    head_sha = git("rev-parse", "HEAD")

    print("  mode       : base-vs-head comparison")
    print("  base       : %s (%s)" % (base_sha[:10], base_rev))
    print("  head       : %s" % head_sha[:10])
    print("  tolerance  : head may be at most %.1f%% slower than base, and must "
          "still finish within %s" % (args.max_regression_pct, fmt(args.threshold_sec)))
    print("  rounds     : %d (alternating; minimum of each wins)" % args.repeat)
    print()

    if base_sha == head_sha:
        print("Base and head are the same commit; nothing to compare.")
        return 0

    worktree = os.path.join(build_root, "base-tree")
    git("worktree", "add", "--detach", "--quiet", worktree, base_sha)
    base_dir = rccl_dir_within(worktree)
    try:
        results: list[Comparison] = []
        for target in targets:
            if not compiler_supports(target, hipcc):
                reason = "compiler does not support --offload-arch=%s" % target
                status = FAIL if args.strict else SKIP
                print("[%s] %s - %s" % (target, status, reason))
                results.append(Comparison(target, status, reason))
                continue

            print("[%s] timing base and head ..." % target, flush=True)
            base_s, head_s, ok = compare_target(
                target, RCCL_ROOT, base_dir, build_root, args.jobs, args.repeat)
            if not ok:
                print("[%s] FAIL - build failed, see %s" % (target, build_root))
                results.append(Comparison(target, FAIL, "build failed"))
                continue

            result = Comparison(target, PASS, "", base_s, head_s)
            pct = result.delta_pct or 0.0

            # The head build is timed here anyway, so this run enforces the
            # absolute budget too and there is no need for a second CI entry.
            problems: list[str] = []
            if pct > args.max_regression_pct:
                problems.append("%.1f%% slower than base (limit %.1f%%)"
                                % (pct, args.max_regression_pct))
            if head_s > args.threshold_sec:
                problems.append("took %s, over the %s budget"
                                % (fmt(head_s), fmt(args.threshold_sec)))
            if problems:
                result.status = FAIL
                result.note = "; ".join(problems)

            print("[%s] %s - base %s -> head %s (%+.1f%%)%s" %
                  (target, result.status, fmt(base_s), fmt(head_s), pct,
                   "  " + result.note if problems else ""))
            results.append(result)
    finally:
        git("worktree", "remove", "--force", worktree)

    print()
    print("%-10s %10s %10s %9s  %s" % ("TARGET", "BASE", "HEAD", "DELTA", "RESULT"))
    for r in results:
        if r.base_s is None or r.head_s is None:
            print("%-10s %10s %10s %9s  %s (%s)" % (r.target, "-", "-", "-", r.status, r.note))
        else:
            print("%-10s %10s %10s %8.1f%%  %s" %
                  (r.target, fmt(r.base_s), fmt(r.head_s), r.delta_pct or 0.0, r.status))

    failed = [r.target for r in results if r.status == FAIL]
    if failed:
        print("\nFAILED: %s" % ", ".join(failed))
        return 1
    if not any(r.status == PASS for r in results):
        print("\nNo targets were buildable with this toolchain; nothing was measured.")
        return 1
    print("\nPASSED: %s" % ", ".join(r.target for r in results if r.status == PASS))
    return 0


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
    print("  build root : %s" % build_root)
    print("  targets    : %s (from %s)" % (" ".join(targets), targets_from))

    if args.compare_base is not None:
        rc = run_comparison(args, targets, hipcc, build_root)
    else:
        rc = run_absolute(args, targets, hipcc, build_root)

    # Each run leaves a few MB of build logs; they are only interesting when
    # something failed, and this runs on every PR.
    if rc == 0 and not args.keep and not args.build_root:
        shutil.rmtree(build_root, ignore_errors=True)
    elif rc != 0:
        print("\nBuild logs kept at %s" % build_root)
    return rc


def run_absolute(
    args: argparse.Namespace, targets: list[str], hipcc: str, build_root: str,
) -> int:
    """Time each target once and gate on the absolute threshold."""
    print("  threshold  : %s (%.0fs) per target, configure + build" %
          (fmt(args.threshold_sec), args.threshold_sec))
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
