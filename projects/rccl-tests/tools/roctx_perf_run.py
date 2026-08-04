#!/usr/bin/env python3
"""Perf-test runner for GPU-timed rccl-tests runs.

For each test x dtype x repeat it runs one of:

  dispatch (default) -- RCCL times its own dispatches, no profiler:
    mpirun -np <N> -x RCCL_KERNEL_TIMING=1 -x RCCL_TESTS_KERNEL_TIMING=<dir>/rank \
      build/<test>_perf <perf-args> -d <dtype>

  profiled -- the same measurement taken by rocprofv3, which costs enough to
  change what it measures:
    mpirun -np <N> -x RCCL_TESTS_ROCTX=1 \
      rocprofv3 --marker-trace --kernel-trace -f csv -d <dir>_rank_%rank% -- \
        build/<test>_perf <perf-args> -d <dtype>

  baseline (--baseline, in addition) -- uninstrumented, host-side timing only.

Running dispatch and baseline together is the honest test of whether the timing
is free: the same build, timed both ways.

Artifacts are saved under a timestamped output directory with a metadata.json
capturing environment, versions, git state, and per-run exit codes.
"""

import argparse
import datetime
import glob
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import textwrap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

KNOWN_DTYPES = ["float", "half", "bfloat16", "double", "int8", "uint8", "int32", "uint32", "int64", "uint64"]

DEFAULT_PERF_ARGS = "-b 8 -e 1G -f 2 -w 5 -n 50 -A 1"

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def find_executable(name, extra_dirs=None, prefer_dirs=None):
    """Find an executable by name.

    Search order:
      1. prefer_dirs (in order) -- checked before PATH
      2. shutil.which (PATH)
      3. extra_dirs (in order) -- fallback when not on PATH
    """
    for d in (prefer_dirs or []):
        candidate = os.path.join(d, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    path = shutil.which(name)
    if path:
        return path
    for d in (extra_dirs or []):
        candidate = os.path.join(d, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def detect_gpu_count():
    enumerator = find_executable("rocm_agent_enumerator", ["/opt/rocm/bin"])
    if not enumerator:
        return None
    try:
        out = subprocess.check_output([enumerator], text=True, timeout=10)
        agents = [l.strip() for l in out.splitlines() if l.strip() and l.strip() != "gfx000"]
        return len(agents) if agents else None
    except Exception:
        return None


def discover_tests(build_dir):
    pattern = os.path.join(build_dir, "*_perf")
    bins = sorted(glob.glob(pattern))
    tests = []
    for b in bins:
        name = os.path.basename(b)
        if name.endswith("_perf"):
            tests.append(name[:-5])
    return tests


def read_cmake_cache_var(build_dir, varname):
    """Read a variable from build_dir/CMakeCache.txt, or None."""
    cache_path = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.isfile(cache_path):
        return None
    pattern = re.compile(rf"^{re.escape(varname)}:\w+=(.+)$")
    try:
        with open(cache_path) as f:
            for line in f:
                m = pattern.match(line.strip())
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def find_mpirun(build_dir):
    """Derive mpirun from CMakeCache.txt, then fall back to PATH.

    Strategy:
      1. MPIEXEC_EXECUTABLE from CMakeCache.txt (set by CMake's FindMPI)
      2. MPI_HOME from CMakeCache.txt -> $MPI_HOME/bin/mpirun
      3. MPI_HOME environment variable -> $MPI_HOME/bin/mpirun
      4. Plain PATH lookup
    """
    mpiexec = read_cmake_cache_var(build_dir, "MPIEXEC_EXECUTABLE")
    if mpiexec and os.path.isfile(mpiexec) and os.access(mpiexec, os.X_OK):
        mpirun = os.path.join(os.path.dirname(mpiexec), "mpirun")
        if os.path.isfile(mpirun):
            return mpirun
        return mpiexec

    for source in [read_cmake_cache_var(build_dir, "MPI_HOME"),
                   os.environ.get("MPI_HOME")]:
        if source:
            candidate = os.path.join(source, "bin", "mpirun")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

    return find_executable("mpirun")


def strip_ansi(text):
    return ANSI_ESCAPE_RE.sub("", text)


def capture_cmd(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return None


LIBRCCL_VERSION_RE = re.compile(r"RCCL version [:\s]*(\S+)")
LIBRCCL_COMPILED_RE = re.compile(r'RCCL version .+ compiled with ROCm "([^"]+)"')
LIBRCCL_HIP_RE = re.compile(r"HIP version\s*:\s*(\S+)")
LIBRCCL_GITHASH_VALUE_RE = re.compile(r"^[a-zA-Z0-9_./-]+:[0-9a-f]{6,12}\+?$")


def collect_ldd_info(binary_path):
    """Run ldd on a binary and return the full output plus resolved librccl path."""
    ldd_output = capture_cmd(["ldd", binary_path], timeout=10)
    librccl_path = None
    if ldd_output:
        for line in ldd_output.splitlines():
            if "librccl" in line:
                parts = line.split("=>")
                if len(parts) == 2:
                    resolved = parts[1].strip().split()[0]
                    if os.path.isfile(resolved):
                        librccl_path = resolved
    return ldd_output, librccl_path


def extract_librccl_info(librccl_path):
    """Extract version strings embedded in librccl.so via strings."""
    if not librccl_path or not os.path.isfile(librccl_path):
        return None
    try:
        out = subprocess.run(
            ["strings", librccl_path],
            capture_output=True, text=True, timeout=30,
        )
        text = out.stdout
    except Exception:
        return None

    real_path = os.path.realpath(librccl_path)
    info = {"path": librccl_path, "realpath": real_path}

    try:
        import hashlib
        h = hashlib.md5()
        with open(real_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 20), b""):
                h.update(chunk)
        info["md5"] = h.hexdigest()
    except Exception:
        pass

    m = LIBRCCL_VERSION_RE.search(text)
    if m:
        info["rccl_version"] = m.group(1)
    m = LIBRCCL_COMPILED_RE.search(text)
    if m:
        info["rocm_build"] = m.group(1)
    m = LIBRCCL_HIP_RE.search(text)
    if m:
        info["hip_build"] = m.group(1)

    for line in text.splitlines():
        if LIBRCCL_GITHASH_VALUE_RE.match(line.strip()):
            info["git_hash"] = line.strip()
            break

    return info


def collect_metadata(args, run_dir):
    meta = {
        "schema_version": 1,
        "timestamp_start": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "run_dir": run_dir,
    }

    meta["args"] = {
        "test": args.test,
        "dtypes": args.dtypes,
        "repeats": args.repeats,
        "np": args.np,
        "map_by": args.map_by,
        "mca_accelerator": args.mca_accelerator,
        "perf_args": args.perf_args,
        "build_dir": os.path.abspath(args.build_dir),
        "modes": args.modes,
        "baseline": args.baseline,
    }

    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    ver_file = os.path.join(rocm_path, ".info", "version")
    rocm_version = None
    if os.path.isfile(ver_file):
        try:
            with open(ver_file) as f:
                rocm_version = f.read().strip()
        except OSError:
            pass

    mpirun_version = capture_cmd([args.mpirun, "--version"])
    rocprofv3_path = args.rocprofv3

    meta["environment"] = {
        "rocm_path": rocm_path,
        "rocm_version": rocm_version,
        "mpirun": os.path.abspath(args.mpirun),
        "mpi_version": mpirun_version.splitlines()[0] if mpirun_version else None,
        "rocprofv3": os.path.abspath(rocprofv3_path) if rocprofv3_path else None,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    }

    git_dir = PROJECT_ROOT
    sha = capture_cmd(["git", "-C", git_dir, "rev-parse", "HEAD"])
    branch = capture_cmd(["git", "-C", git_dir, "rev-parse", "--abbrev-ref", "HEAD"])
    porcelain = capture_cmd(["git", "-C", git_dir, "status", "--porcelain"])
    meta["git"] = {
        "sha": sha,
        "branch": branch,
        "dirty": bool(porcelain) if porcelain is not None else None,
    }

    gpu_info = capture_cmd(["rocm-smi", "--showproductname"], timeout=15)
    meta["gpu_info"] = gpu_info

    first_test = args.test[0]
    perf_binary = os.path.join(os.path.abspath(args.build_dir), f"{first_test}_perf")
    ldd_output, librccl_path = collect_ldd_info(perf_binary)
    meta["ldd"] = ldd_output
    librccl_info = extract_librccl_info(librccl_path)
    meta["librccl"] = librccl_info

    meta["matrix"] = {
        "tests": args.test,
        "dtypes": args.dtypes,
        "repeats": args.repeats,
        "np": args.np,
        "map_by": args.map_by,
        "mca_accelerator": args.mca_accelerator,
        "perf_args": args.perf_args,
        "modes": args.modes,
        "baseline": args.baseline,
    }

    meta["results"] = []
    meta["status"] = "running"
    return meta


def write_metadata(meta, run_dir):
    path = os.path.join(run_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _mpi_env_flags():
    """Return mpirun -x flags to forward select env vars to child processes.

    LD_LIBRARY_PATH must be forwarded explicitly: mpirun does NOT propagate it
    to remote/child ranks on its own, and a rank that resolves a different
    libmpi.so/libopen-pal.so/libopen-rte.so than the one `mpirun` belongs to
    will silently fall back to its own singleton MPI_COMM_WORLD (size 1)
    instead of joining the real N-rank job -- see verify_mpi_coordination().

    ROCR_VISIBLE_DEVICES is forwarded as-is (same value to every rank) so a
    subset of free GPUs (e.g. because one is held by someone else's
    reservation) can be selected. This Open MPI build does have its own
    ROCm-awareness -- it has an "accelerator/rocm" MCA component (built via
    --with-rocm) that talks to libamdhip64.so.7 for GPU-aware buffer support
    -- but that's unrelated to *scheduling*: it has no notion of a GPU being
    "reserved" (that's purely our own bookkeeping, not visible to the driver),
    so it will never route ranks around a busy/held device on its own.
    Restricting visibility via ROCR_VISIBLE_DEVICES is still required, and
    still needs to be the same list on every rank so each one independently
    picks its own device from the restricted set by local rank, same as it
    would from the unrestricted set.
    """
    flags = []
    if os.environ.get("LD_LIBRARY_PATH"):
        flags += ["-x", "LD_LIBRARY_PATH"]
    if os.environ.get("ROCR_VISIBLE_DEVICES"):
        flags += ["-x", "ROCR_VISIBLE_DEVICES"]
    return flags


def build_mpicmd(args, roctx=True):
    mpicmd = [args.mpirun, "-np", str(args.np)]
    if args.map_by:
        # Without an explicit --map-by, Open MPI 5.0.6 packs all ranks onto
        # package[0]: with 8 ranks on a 2-socket host (GPUs 0-3 on NUMA node0,
        # 4-7 on node1), ranks 4-7 end up NUMA-remote from the GPU they own
        # (host-side work -- doorbell writes, proxy threads, kernel-timing
        # harvest reads -- pays the cross-socket distance penalty on every
        # access). "--map-by numa" spreads ranks across NUMA domains so each
        # rank's host binding matches its GPU's socket.
        mpicmd += ["--map-by", args.map_by]
    if args.mca_accelerator:
        # Open MPI's ob1 PML eagerly initializes GPU-aware ("accelerator")
        # support in MPI_Init -- before the app has called cudaSetDevice, so
        # HIP's current device is still its global default (device ordinal
        # 0). Every rank ends up leaking a handful of HSA compute queues onto
        # whichever physical GPU that is, regardless of which GPU the rank
        # actually owns. On an 8-rank single-node job that oversubscribes
        # that one GPU's KFD runlist and causes periodic ~10ms collective
        # stalls (confirmed via HSA-API-level backtraces: the streams come
        # from mca_pml_ob1_accelerator_init, not from RCCL or this harness).
        # rccl-tests never passes GPU pointers through MPI itself (only
        # small host-memory bootstrap traffic), so this support is unused
        # dead weight here; "--mca accelerator null" disables it entirely.
        mpicmd += ["--mca", "accelerator", args.mca_accelerator]
    mpicmd += _mpi_env_flags()
    if roctx:
        mpicmd += ["-x", "RCCL_TESTS_ROCTX=1"]
    return mpicmd

def build_perfcmd(args, test):
    perf_binary = os.path.join(os.path.abspath(args.build_dir), f"{test}_perf")
    perf_args_list = shlex.split(args.perf_args)
    return [perf_binary] + perf_args_list

def build_dispatch_cmd(args, test, dtype, output_dir):
    """Build a run that has RCCL time its own dispatches.

    Each rank appends its pid to the prefix, so one directory per (test, dtype,
    rep) collects the whole run without the ranks colliding.
    """
    cmd = build_mpicmd(args, roctx=False) + [
        "-x", "RCCL_KERNEL_TIMING=1",
        "-x", f"RCCL_TESTS_KERNEL_TIMING={os.path.join(output_dir, 'rank')}",
    ]
    if args.kernel_timing_layout:
        cmd += ["-x", f"RCCL_KERNEL_TIMING_LAYOUT={args.kernel_timing_layout}"]
    return cmd + build_perfcmd(args, test) + ["-d", dtype]

def build_rocprofv3_cmd(args, test, dtype, output_dir):
    mpicmd = build_mpicmd(args)
    rocpcmd = [
        args.rocprofv3,
        "--marker-trace", "--kernel-trace",
        "-f", "csv",
        "-d", output_dir,       # e.g.  results_rank_%rank%
        "--" ]
    perfcmd = build_perfcmd(args, test) + [ "-d", dtype ]
    cmd = mpicmd + rocpcmd + perfcmd
    return cmd

def build_baseline_cmd(args, test, dtype, csv_path):
    """Build command for an uninstrumented baseline run (no timing of any kind)."""
    cmd = build_mpicmd(args, roctx=False) + build_perfcmd(args, test) + [
      "-d",  dtype, "-Z", "csv", "-X", csv_path]
    return cmd

def run_one(cmd, log_path, dry_run=False, cwd=None):
    cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)
    if dry_run:
        print(f"  [dry-run] {cmd_str}")
        return 0, cmd_str

    print(f"  $ {cmd_str}")
    sys.stdout.flush()
    proc = subprocess.run(cmd, capture_output=True, cwd=cwd)
    cleaned = strip_ansi(proc.stdout.decode("utf-8", errors="replace"))
    if proc.stderr:
        cleaned += strip_ansi(proc.stderr.decode("utf-8", errors="replace"))
    with open(log_path, "w") as log_f:
        log_f.write(f"# {cmd_str}\n")
        log_f.write(cleaned)
    return proc.returncode, cmd_str


USING_DEVICES_RE = re.compile(r"^# Using devices$", re.MULTILINE)
RANK_LINE_RE = re.compile(r"^#\s+Rank\s+\d+\s+Group", re.MULTILINE)


def verify_mpi_coordination(args):
    """Launch a throwaway 1-iteration run and check that all `args.np` ranks
    actually joined one MPI job, rather than each falling back to its own
    singleton `MPI_COMM_WORLD` (size 1).

    A real N-rank job prints exactly one gathered "# Using devices" block
    with N "# Rank" lines (see writeDeviceReport()'s MPI_Gather in
    src/util.cu). N processes that never coordinated -- e.g. because the
    ranks resolved a different libmpi.so than the one `mpirun` belongs to --
    each print their own block instead, so this count is a direct behavioral
    check rather than a guess about paths or env vars.
    """
    if args.np < 2:
        return
    probe_cmd = build_mpicmd(args, roctx=False) + [
        os.path.join(os.path.abspath(args.build_dir), f"{args.test[0]}_perf"),
        "-b", "8", "-e", "8", "-w", "0", "-n", "1", "-d", args.dtypes[0],
    ]
    print("Verifying MPI coordination (probe run) ...")
    print(f"  $ {' '.join(shlex.quote(str(c)) for c in probe_cmd)}")
    try:
        proc = subprocess.run(probe_cmd, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        sys.exit(
            "Preflight MPI check timed out. This usually means the ranks did not "
            "coordinate at all (hung waiting for peers); aborting rather than "
            "running the full sweep against a broken MPI setup."
        )
    out = strip_ansi(proc.stdout.decode("utf-8", errors="replace"))
    out += strip_ansi(proc.stderr.decode("utf-8", errors="replace"))

    blocks = len(USING_DEVICES_RE.findall(out))
    rank_lines = len(RANK_LINE_RE.findall(out))

    if proc.returncode == 0 and blocks == 1 and rank_lines == args.np:
        print(f"  OK: one {args.np}-rank job (1 gathered device block, {rank_lines} rank lines).\n")
        return

    ldd_out, _ = collect_ldd_info(os.path.join(os.path.abspath(args.build_dir), f"{args.test[0]}_perf"))
    mpi_lines = "\n".join(l for l in (ldd_out or "").splitlines() if "mpi" in l.lower() or "pal" in l.lower() or "rte" in l.lower())
    sys.exit(
        f"Preflight MPI check failed: expected 1 gathered device block with {args.np} "
        f"rank lines, got {blocks} block(s) and {rank_lines} rank line(s) (rc={proc.returncode}).\n"
        f"This is the signature of each rank falling back to its own singleton "
        f"MPI_COMM_WORLD (size 1) instead of joining one {args.np}-rank job -- typically "
        f"because the test binary resolves a different libmpi.so/libopen-pal.so/libopen-rte.so "
        f"at runtime than the one `{args.mpirun}` belongs to.\n"
        f"mpirun:          {args.mpirun}\n"
        f"MPI_HOME (env):  {os.environ.get('MPI_HOME')}\n"
        f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}\n"
        f"ldd (mpi-related libs actually resolved by the test binary):\n{mpi_lines}\n"
        f"--- probe output ---\n{out}"
    )


def run_matrix(args, meta, run_dir):
    results = []
    errors = 0

    for test in args.test:
        for dtype in args.dtypes:
            # --- RCCL dispatch-timed runs ---
            if "dispatch" in args.modes:
                for rep in range(1, args.repeats + 1):
                    tag = f"{test}/{dtype}/dispatch/rep{rep}"
                    disp_dir = os.path.join(run_dir, f"{test}_{dtype}_rep{rep}_dispatch")
                    log_path = os.path.join(run_dir, f"{test}_{dtype}_dispatch_rep{rep}.log")
                    if not args.dry_run:
                        os.makedirs(disp_dir, exist_ok=True)

                    cmd = build_dispatch_cmd(args, test, dtype, disp_dir)
                    print(f"  [{tag}] dispatch timing ...")
                    rc, cmd_str = run_one(cmd, log_path, dry_run=args.dry_run)

                    entry = {
                        "test": test,
                        "dtype": dtype,
                        "rep": rep,
                        "dispatch": True,
                        "rc": rc,
                        "log": os.path.relpath(log_path, run_dir),
                        "dispatch_dir": os.path.relpath(disp_dir, run_dir),
                    }
                    if rc == 0 and not args.dry_run:
                        traces = len(glob.glob(os.path.join(disp_dir, "*_pid*.csv")))
                        entry["trace_files"] = traces
                        if traces == 0:
                            print(f"  [{tag}] no dispatch traces written -- is this "
                                  f"librccl built with kernel timing?")
                    results.append(entry)
                    meta["results"].append(entry)
                    if not args.dry_run:
                        write_metadata(meta, run_dir)

                    if rc != 0:
                        errors += 1
                        print(f"  [{tag}] exited {rc}")
                    else:
                        print(f"  [{tag}] ok")

            # --- rocprofv3-profiled runs ---
            if "profiled" in args.modes:
                for rep in range(1, args.repeats + 1):
                    tag = f"{test}/{dtype}/rep{rep}"
                    prof_dir = os.path.join(run_dir, f"{test}_{dtype}_rep{rep}_%rank%")
                    log_path = os.path.join(run_dir, f"{test}_{dtype}_rep{rep}.log")

                    cmd = build_rocprofv3_cmd(args, test, dtype, prof_dir)
                    print(f"  [{tag}] profiling ...")
                    rc, cmd_str = run_one(cmd, log_path, dry_run=args.dry_run)

                    entry = {
                        "test": test,
                        "dtype": dtype,
                        "rep": rep,
                        "profiled": True,
                        "rc": rc,
                        "log": os.path.relpath(log_path, run_dir),
                        "profiler_dir": os.path.relpath(prof_dir, run_dir),
                    }
                    if rc == 0 and not args.dry_run:
                        # rocprofv3 keeps intercepting rocTX even when it cannot
                        # trace dispatches against a newer HSA runtime than it
                        # was built for, so a run that "succeeded" can still
                        # carry markers and no kernels at all. Say so here
                        # rather than let it read as a zero-cost profile later.
                        pat = os.path.join(run_dir, f"{test}_{dtype}_rep{rep}_*")
                        kernels = len(glob.glob(os.path.join(pat, "**", "*_kernel_trace.csv"),
                                                recursive=True))
                        markers = len(glob.glob(os.path.join(pat, "**", "*_marker_api_trace.csv"),
                                                recursive=True))
                        entry["kernel_trace_files"] = kernels
                        entry["marker_trace_files"] = markers
                        if kernels == 0:
                            print(f"  [{tag}] rocprofv3 wrote no kernel trace "
                                  f"({markers} marker trace(s)): this run has no kernel "
                                  f"timings. Check that rocprofv3 matches the ROCm "
                                  f"runtime being profiled.")
                    results.append(entry)
                    meta["results"].append(entry)
                    if not args.dry_run:
                        write_metadata(meta, run_dir)

                    if rc != 0:
                        errors += 1
                        print(f"  [{tag}] exited {rc}")
                    else:
                        print(f"  [{tag}] ok")

            # --- baseline (uninstrumented) runs ---
            if args.baseline:
                for rep in range(1, args.repeats + 1):
                    tag = f"{test}/{dtype}/baseline/rep{rep}"
                    csv_path = os.path.join(
                        run_dir, f"{test}_{dtype}_baseline_rep{rep}.csv")
                    log_path = os.path.join(
                        run_dir, f"{test}_{dtype}_baseline_rep{rep}.log")

                    cmd = build_baseline_cmd(args, test, dtype, csv_path)
                    print(f"  [{tag}] baseline ...")
                    rc, cmd_str = run_one(cmd, log_path, dry_run=args.dry_run)

                    entry = {
                        "test": test,
                        "dtype": dtype,
                        "rep": rep,
                        "baseline": True,
                        "rc": rc,
                        "log": os.path.relpath(log_path, run_dir),
                        "baseline_csv": os.path.relpath(csv_path, run_dir),
                    }
                    results.append(entry)
                    meta["results"].append(entry)
                    if not args.dry_run:
                        write_metadata(meta, run_dir)

                    if rc != 0:
                        errors += 1
                        print(f"  [{tag}] exited {rc}")
                    else:
                        print(f"  [{tag}] ok")

    return results, errors


def print_summary(results):
    if not results:
        return

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    max_test = max(len(r["test"]) for r in results)
    max_dtype = max(len(r["dtype"]) for r in results)

    header = f"  {'test':<{max_test}}  {'dtype':<{max_dtype}}  {'kind':<10}  rep  rc"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        status = "ok" if r["rc"] == 0 else f"FAIL({r['rc']})"
        if r.get("baseline"):
            kind = "baseline"
        elif r.get("dispatch"):
            kind = "dispatch"
        else:
            kind = "profiled"
        if kind == "dispatch" and r.get("trace_files") == 0:
            status += " (no traces)"
        print(f"  {r['test']:<{max_test}}  {r['dtype']:<{max_dtype}}  {kind:<10}  {r['rep']:>3}  {status}")

    passed = sum(1 for r in results if r["rc"] == 0)
    total = len(results)
    print(f"\n  {passed}/{total} passed")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="rocTX+rocprofv3 profiled perf-test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s --test all_reduce
              %(prog)s --test all_reduce --dtypes float,half --repeats 4
              %(prog)s --test all_reduce --baseline
              %(prog)s --test all_reduce --mode dispatch,profiled --baseline
              %(prog)s --list-tests
              %(prog)s --dry-run --test broadcast --np 2

            mpirun resolution order:
              1. MPIEXEC_EXECUTABLE from build/CMakeCache.txt
              2. MPI_HOME from CMakeCache.txt  -> $MPI_HOME/bin/mpirun
              3. MPI_HOME environment variable  -> $MPI_HOME/bin/mpirun
              4. PATH lookup
        """),
    )

    parser.add_argument(
        "--list-tests", action="store_true",
        help="List available test binaries in --build-dir and exit",
    )
    parser.add_argument(
        "--test", type=str, default=None,
        help="Comma-separated test names (e.g. all_reduce,broadcast). "
             "Maps to <name>_perf binaries in --build-dir.",
    )
    parser.add_argument(
        "--dtypes", type=str, default="float",
        help="Comma-separated data types (default: float). "
             f"Known types: {', '.join(KNOWN_DTYPES)}",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of profiled repeats per test/dtype (default: 1)",
    )
    parser.add_argument(
        "--np", type=int, default=None,
        help="Number of MPI ranks (default: #GPUs from rocm_agent_enumerator)",
    )
    parser.add_argument(
        "--map-by", type=str, default="numa",
        help="Open MPI '--map-by' policy passed to mpirun for rank/host "
             "placement (default: numa). Without this, Open MPI packs all "
             "ranks onto package[0], leaving high-numbered ranks NUMA-remote "
             "from the GPUs they own on multi-socket hosts. Pass an empty "
             "string ('') to omit --map-by entirely and use Open MPI's own "
             "default.",
    )
    parser.add_argument(
        "--mca-accelerator", type=str, default="null",
        help="Open MPI '--mca accelerator' component (default: null). Open "
             "MPI's ob1 PML eagerly initializes GPU-aware MPI support in "
             "MPI_Init, before the app selects its device, leaking HSA "
             "queues onto whichever GPU is device ordinal 0 on every rank. "
             "rccl-tests never passes GPU pointers through MPI, so this is "
             "unused; 'null' disables it. Pass an empty string ('') to omit "
             "the flag and use Open MPI's own default (auto-detect rocm).",
    )
    parser.add_argument(
        "--perf-args", type=str, default=DEFAULT_PERF_ARGS,
        help=f"Arguments passed to the *_perf binary (default: '{DEFAULT_PERF_ARGS}')",
    )
    parser.add_argument(
        "--build-dir", type=str, default=os.path.join(PROJECT_ROOT, "build"),
        help="Directory containing *_perf binaries (default: <project>/build)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "perf-runs"),
        help="Base directory for output (default: <project>/perf-runs)",
    )
    parser.add_argument(
        "--mode", type=str, default="dispatch",
        help="Instrumented run modes, comma-separated: 'dispatch' (RCCL times "
             "its own dispatches, no profiler), 'profiled' (rocprofv3), or both "
             "(default: dispatch)",
    )
    parser.add_argument(
        "--rocprofv3", type=str, default=None,
        help="Path to rocprofv3 (default: $ROCM_PATH/bin, then PATH). "
             "Only needed for --mode profiled.",
    )
    parser.add_argument(
        "--kernel-timing-layout", type=str, default=None,
        help="Value for RCCL_KERNEL_TIMING_LAYOUT in dispatch runs (1..N pins a "
             "known event layout; RCCL probes for it by default)",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Also run each test with no instrumentation at all, to price the "
             "instrumented runs against.  Passes -Z csv -X file to the perf "
             "binary.  Baseline runs follow the instrumented runs within each "
             "(test, dtype) group.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args(argv)

    # --list-tests: just print and exit
    if args.list_tests:
        tests = discover_tests(args.build_dir)
        if not tests:
            print(f"No *_perf binaries found in {args.build_dir}", file=sys.stderr)
            sys.exit(1)
        print("Available tests:")
        for t in tests:
            print(f"  {t}")
        sys.exit(0)

    # Resolve test names
    if args.test is None:
        parser.error("--test is required (use --list-tests to see available tests)")
    args.test = [t.strip() for t in args.test.split(",") if t.strip()]
    if not args.test:
        parser.error("--test must specify at least one test name")

    # Validate test binaries exist
    available = discover_tests(args.build_dir)
    for t in args.test:
        if t not in available:
            parser.error(
                f"test '{t}' not found in {args.build_dir}. "
                f"Available: {', '.join(available)}"
            )

    # Resolve dtypes
    args.dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]

    # Resolve np
    if args.np is None:
        gpu_count = detect_gpu_count()
        if gpu_count is None or gpu_count == 0:
            parser.error("Could not detect GPU count; specify --np explicitly")
        args.np = gpu_count
        print(f"Detected {args.np} GPUs")

    # Resolve mpirun from CMakeCache.txt or PATH
    args.mpirun = find_mpirun(args.build_dir)
    if args.mpirun is None:
        parser.error(
            "mpirun not found. Set MPI_HOME, build with cmake (populates CMakeCache.txt), "
            "or ensure mpirun is on PATH."
        )
    print(f"Using mpirun: {args.mpirun}")

    # Resolve modes
    args.modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    unknown = [m for m in args.modes if m not in ("dispatch", "profiled")]
    if unknown:
        parser.error(f"unknown --mode value(s): {', '.join(unknown)}")
    if not args.modes and not args.baseline:
        parser.error("--mode must name at least one mode, or pass --baseline")

    # Resolve rocprofv3: prefer $ROCM_PATH/bin over anything on PATH, since system
    # installs under /usr/bin are often stale or mismatched on HPC systems.
    if "profiled" in args.modes and args.rocprofv3 is None:
        rocm_bin = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin")
        args.rocprofv3 = find_executable("rocprofv3", prefer_dirs=[rocm_bin])
        if args.rocprofv3 is None:
            parser.error(
                "rocprofv3 not found; set ROCM_PATH or specify --rocprofv3"
            )

    return args


def main():
    args = parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    final_dir = os.path.join(args.output_dir, stamp)
    tmp_dir = os.path.join(args.output_dir, f".tmp-{stamp}-{os.getpid()}")

    if not args.dry_run:
        os.makedirs(tmp_dir, exist_ok=True)

    run_dir = tmp_dir

    print(f"Output:   {final_dir}  (staging in {os.path.basename(tmp_dir)})")
    print(f"Tests:    {', '.join(args.test)}")
    print(f"Dtypes:   {', '.join(args.dtypes)}")
    print(f"Ranks:    {args.np}")
    print(f"Map-by:   {args.map_by or '(Open MPI default)'}")
    print(f"Accel:    --mca accelerator {args.mca_accelerator}" if args.mca_accelerator else "Accel:    (Open MPI default)")
    print(f"Reps:     {args.repeats}")
    print(f"Args:     {args.perf_args}")
    print(f"Modes:    {', '.join(args.modes) if args.modes else 'none'}")
    if args.baseline:
        print(f"Baseline: enabled (uninstrumented runs with -Z csv -X file)")
    print()

    meta = collect_metadata(args, run_dir)
    meta["final_dir"] = final_dir
    if not args.dry_run:
        write_metadata(meta, run_dir)

    def _signal_handler(signum, frame):
        print(f"\nCaught signal {signum}, writing partial metadata ...")
        meta["status"] = "interrupted"
        meta["timestamp_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            write_metadata(meta, run_dir)
        except Exception:
            pass
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if not args.dry_run:
        verify_mpi_coordination(args)

    results, errors = run_matrix(args, meta, run_dir)

    meta["status"] = "completed" if errors == 0 else "completed_with_errors"
    meta["timestamp_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if not args.dry_run:
        write_metadata(meta, run_dir)

    print_summary(results)

    if not args.dry_run:
        os.rename(tmp_dir, final_dir)
        print(f"Results: {final_dir}")

    if errors:
        print(f"\n{errors} run(s) had non-zero exit codes.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
