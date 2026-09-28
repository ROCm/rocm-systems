# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Automation script that validates the ROCm Systems Profiler (rocprof-sys)
binaries shipped inside a nightly ROCm (TheRock) tarball against the latest
tests + examples from the ``develop`` branch of ROCm/rocm-systems.

What it does (in order), inside a reusable directory ``rocprofiler-systems-tests``
created in the *current working directory*. The directory is reused across runs so
work is incremental:

  1. Resolve the latest nightly ROCm multi-arch tarball from
     https://nightly.repo.amd.com/rocm/core/tarball/. It is downloaded and
     (re)extracted into ``<workdir>/rocm`` (ROCM_PATH, providing
     ``bin/rocprof-sys-*``) ONLY when a newer nightly is available than the one
     already extracted.
  2. Sparse-clone (or ``git fetch``+update) the ``develop`` branch of
     ROCm/rocm-systems, checking out only ``projects/rocprofiler-systems``.
  3. Create/reuse an isolated Python venv with the pytest test dependencies
     (from ``requirements.txt``).
  4. Build the example programs (standalone) and stage the pytest test-suite into
     the ROCm prefix so the suite runs in "install mode". The examples are rebuilt
     ONLY when the ROCm tree was replaced, the source changed, the artifacts are
     missing, or ``--force-rebuild`` is given.
  5. Run the pytest suite in install mode so every test exercises the
     ``rocprof-sys-*`` binaries that live in the downloaded tarball's ``bin/``
     folder (nothing from rocprof-sys itself is rebuilt).

Typical usage on a GPU test machine:

    python3 run-nightly-tarball-tests.py                 # latest multiarch nightly
    python3 run-nightly-tarball-tests.py --variant gfx90a  # smallest per-GPU tarball
    python3 run-nightly-tarball-tests.py --tier quick    # fast smoke subset
    python3 run-nightly-tarball-tests.py --tier full --run-labels mpi
    python3 run-nightly-tarball-tests.py --reruns 2      # retry flaky tests
    python3 run-nightly-tarball-tests.py --rocm-version 7.15.0a20260717
    python3 run-nightly-tarball-tests.py --pytest-args "-m gpu -k transpose"

Validate freshly-built binaries instead of the tarball's shipped ones (QA-style
build-from-source + test; uses the tarball only for ROCm runtime/toolchain):

    python3 run-nightly-tarball-tests.py --build-from-source

Air-gapped cluster (compute nodes have no network) - two phases sharing one
--work-dir on a shared filesystem:

    # Phase 1, on a networked node (e.g. the login node): fetch everything.
    python3 run-nightly-tarball-tests.py --work-dir /scratch/$USER/rocm-test \
        --variant auto --prepare-only

    # Phase 2, on the air-gapped GPU compute node: no network, build + run tests.
    python3 run-nightly-tarball-tests.py --work-dir /scratch/$USER/rocm-test \
        --offline --tier standard

Each run writes a detailed log, a summary log (with per-step timings, test counts,
failed-test names, and the rocprof-sys version under test), and, on failure, a
failures log listing each failing test with its output.

Environment variables:
    The script inherits your shell environment and forwards it.
    It pins the build and tests to the extracted tarball and will override:
      ROCM_PATH, HIP_PATH, HSA_PATH, HIP_CLANG_PATH, CMAKE_PREFIX_PATH,
      ROCPROFSYS_CI, ROCPROFSYS_INSTALL_DIR / ROCPROFSYS_BUILD_DIR, and
      TMPDIR/TMP/TEMP (redirected into <work-dir>/tmp during tests).
    PATH, LD_LIBRARY_PATH, LIBRARY_PATH, PKG_CONFIG_PATH, and CMAKE_MODULE_PATH
    drop entries that belong to any other ROCm/TheRock prefix already in the
    environment (a host /opt/rocm, HIP_PATH, or a leftover therock-tarball on
    PATH). The tarball's bin/ and lib/ directories are then prepended.
    ROCPROFSYS_TMPDIR and GIT_HTTP_LOW_SPEED_* are only set when you have not
    already set them.
    ``--variant auto`` reads the GPU arch from the KFD topology, not from
    whichever rocminfo happens to be first on PATH.

Run ``--help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import platform
import shlex
import socket
import subprocess
import sys
from pathlib import Path

from .build import (
    build_examples,
    build_from_source,
    detect_mpi,
    ensure_trace_processor_shell,
    stage_tests,
)
from .command import begin_run, die, log, open_detailed_log, step
from .constants import (
    DEFAULT_MIN_FREE_GB,
    MULTIARCH_VARIANT,
    REQUIRED_ROCPROFSYS_BINARIES,
    TEST_CATEGORIES_REL,
    TIER_ORDER,
)
from .context import RunContext
from .environment import make_rocm_env
from .gpu import (
    _tool_version,
    capture_rocm_sanity,
    preflight,
    report_under_test,
    resolve_variant,
    smoke_check_rocprofsys,
)
from .reporting import write_summary
from .source import (
    _git_rev,
    capture_pip_freeze,
    install_system_deps,
    make_venv,
    sync_source,
)
from .tarball import (
    cleanup_downloaded_tarballs,
    current_rocm_tarball,
    download_file,
    extract_tarball,
    parse_dist_tarball,
    resolve_tarball,
    set_rocm_tarball,
    verify_tarball,
)
from .testing import (
    clean_previous_logs,
    collect_failed_tests,
    drop_label_excludes,
    load_test_categories,
    parse_junit,
    run_tests,
    tier_selection_args,
    write_failures_log,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--variant",
        default="multiarch",
        help="Tarball GPU variant (default: multiarch). Use 'auto' to pick the "
        "smallest per-family tarball for the detected GPU. A specific arch is "
        "accepted where only its family ships, e.g. 'gfx942' resolves to the "
        "gfx94X-dcgpu tarball. Examples: multiarch, gfx942, gfx94X-dcgpu (MI300), "
        "gfx950-dcgpu, gfx90a, gfx110X-all.",
    )
    p.add_argument(
        "--rocm-version",
        default=None,
        help="Specific nightly version to use (e.g. 7.15.0a20260717 or 20260717). "
        "Default: the latest available for the variant.",
    )
    p.add_argument(
        "--sha256",
        default=None,
        help="Expected SHA-256 of the ROCm tarball. When set, the download is "
        "verified against it before extraction (recommended for pinned runs).",
    )
    p.add_argument(
        "--require-checksum",
        action="store_true",
        help="Abort if no SHA-256 is available (via --sha256 or published next to "
        "the tarball) instead of only warning.",
    )
    p.add_argument(
        "--branch",
        default="develop",
        help="rocm-systems branch to test from (default: develop).",
    )
    p.add_argument(
        "--work-dir",
        default=None,
        help="Override the working directory (default: " "./rocprofiler-systems-tests).",
    )
    p.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=os.cpu_count() or 8,
        help="Parallel build jobs (default: nproc).",
    )
    p.add_argument(
        "--disable-examples",
        default="lulesh",
        help="Comma-separated example directories to skip building (default: "
        "lulesh, which vendors Kokkos and requires C++20 in the standalone "
        "build). Pass '' to build everything.",
    )
    p.add_argument(
        "--install-system-deps",
        action="store_true",
        help="Best-effort apt-get install of build/runtime deps (needs sudo/root).",
    )
    p.add_argument(
        "--tier",
        default="standard",
        choices=tuple(TIER_ORDER),
        help="Test tier to run (default: standard). Tiers are read from "
        f"{TEST_CATEGORIES_REL} in the checked-out source, the same definitions "
        "'ctest -L <tier>' uses: 'quick' is a fast smoke subset, 'standard' is the "
        "PR tier, 'comprehensive' the nightly tier, and 'full' everything that can "
        "run. GPU and CPU-only install-mode tests are both included. Ignored when "
        "--pytest-args is given.",
    )
    p.add_argument(
        "--run-labels",
        default="",
        help="Comma-separated marker labels to stop excluding, e.g. 'mpi' to run the "
        f"MPI tests. {TEST_CATEGORIES_REL} excludes some categories (mpi, annotate, "
        "julia, ...) from every tier including 'full', so re-enabling one is an "
        "explicit override. Ignored when --pytest-args is given.",
    )
    p.add_argument(
        "--reruns",
        type=int,
        default=0,
        help="Re-run each failing test up to N times before marking it failed "
        "(needs pytest-rerunfailures; default: 0).",
    )
    p.add_argument(
        "--reruns-delay",
        type=int,
        default=5,
        help="Seconds to wait between reruns (default: 5).",
    )
    p.add_argument(
        "--rerun-failed",
        type=int,
        default=0,
        help="After the main run, re-run only the failed tests up to N times, each "
        "into its own JUnit artifact (pytest-rerun-<n>.xml). Distinguishes flaky "
        "tests from hard failures (default: 0).",
    )
    p.add_argument(
        "--min-free-gb",
        type=int,
        default=DEFAULT_MIN_FREE_GB,
        help=f"Minimum free disk space (GB) required before a download "
        f"(default: {DEFAULT_MIN_FREE_GB}).",
    )
    p.add_argument(
        "--pytest-args",
        default=None,
        help="Override the pytest selection/args entirely (quoted). Takes precedence "
        "over --tier.",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse the already-extracted ROCm tree in the work dir; never check "
        "for a newer nightly.",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding the examples even when the source and ROCm tarball "
        "are unchanged.",
    )
    p.add_argument(
        "--force-sync",
        action="store_true",
        help="Force a git fetch/reset of the source checkout and treat the source "
        "as changed so examples/tests are rebuilt/refreshed. Requires network "
        "access; cannot be combined with --offline.",
    )
    p.add_argument(
        "--skip-tests",
        action="store_true",
        help="Do everything except run the pytest suite.",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Delete log/report artifacts from previous runs (detailed/summary/"
        "RESULT/failures/sanity/pip-freeze/junit/archives/latest-*) before starting. "
        "Preserves the extracted ROCm tree, source checkout, venv, and build dirs.",
    )
    p.add_argument(
        "--build-from-source",
        action="store_true",
        help="Instead of testing the tarball's shipped binaries, build the full "
        "rocprofiler-systems from source (using the tarball for ROCm runtime + "
        "toolchain) and validate the freshly-built build-tree binaries. Pulls "
        "submodules and takes ~1-2h. Mirrors the QA build+CTest flow.",
    )
    p.add_argument(
        "--cmake-arg",
        action="append",
        default=[],
        dest="cmake_args",
        metavar="ARG",
        help="Extra argument forwarded verbatim to the example / source CMake "
        "configure step. Repeatable. Must use the '=' form, since the value starts "
        "with a dash. Use it for site-specific needs, e.g. pointing "
        "find_package(MPI) at a vendor MPI that ships no mpicc wrapper for CMake to "
        "interrogate (Cray MPICH): --cmake-arg=-DMPI_C_LIB_NAMES=mpi "
        "--cmake-arg=-DMPI_mpi_LIBRARY=$MPICH_DIR/lib/libmpi.so. These come last on "
        "the CMake command line, so they also override what the script picked, e.g. "
        "--cmake-arg=-DROCPROFSYS_USE_MPI=OFF to build without MPI on a host that "
        "has it.",
    )
    # ---- offline / two-phase (air-gapped compute node) workflow ----------- #
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Phase 1 (run on a networked node, e.g. the login node): download + "
        "extract the tarball, clone the source, and create the venv, then STOP "
        "before building/testing. Nothing GPU-specific is done.",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Phase 2 (run on an air-gapped GPU compute node): do no network I/O at "
        "all (no tarball download, no git fetch, no pip). Reuses the "
        "tarball/source/venv staged by an earlier --prepare-only run.",
    )
    return p.parse_args(argv)


def open_run(args: argparse.Namespace) -> RunContext:
    """Create the work directory, open the detailed log, and record run facts."""
    if args.work_dir:
        workdir = Path(args.work_dir).resolve()
    else:
        workdir = Path.cwd() / "rocprofiler-systems-tests"
    workdir.mkdir(parents=True, exist_ok=True)
    rocm_dir = workdir / "rocm"
    started = _dt.datetime.now()
    run_stamp = started.strftime("%Y%m%d-%H%M%S")
    detail_log = workdir / f"detailed-{run_stamp}.log"
    summary_log = workdir / f"summary-{run_stamp}.log"
    ctx = RunContext(
        args=args,
        workdir=workdir,
        rocm_dir=rocm_dir,
        facts={},
        run_stamp=run_stamp,
        detail_log=detail_log,
        summary_log=summary_log,
    )
    begin_run(ctx)
    open_detailed_log(detail_log)
    if args.clean:
        clean_previous_logs(workdir, keep_stamp=run_stamp)

    facts = ctx.facts
    facts.update(
        {
            "started": started.isoformat(timespec="seconds"),
            # shlex.join, not " ".join: keeps quoting so the recorded command can be
            # pasted back verbatim (--pytest-args takes one space-containing value).
            "command": shlex.join(sys.argv),
            "host": socket.gethostname(),
            "work_dir": str(workdir),
            "rocm_prefix": str(rocm_dir),
            "variant": args.variant,
            "branch": args.branch,
            "detailed_log": str(detail_log),
        }
    )
    sched = {
        k: os.environ[k]
        for k in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_JOB_GPUS",
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
        )
        if os.environ.get(k)
    }
    if sched:
        facts["scheduler"] = "; ".join(f"{k}={v}" for k, v in sched.items())

    log(f"Working directory: {workdir}")
    log(f"ROCm prefix (ROCM_PATH): {rocm_dir}")
    log(f"Detailed log: {detail_log}")
    log(f"Summary log:  {summary_log}")
    if args.cmake_args:
        facts["cmake_args"] = " ".join(args.cmake_args)
        log("Extra CMake args: " + " ".join(shlex.quote(a) for a in args.cmake_args))
    return ctx


def prepare_rocm(ctx: RunContext) -> None:
    """Download or reuse the nightly tarball and pin the process environment to it."""
    args = ctx.args
    facts = ctx.facts
    workdir = ctx.workdir
    rocm_dir = ctx.rocm_dir
    run_stamp = ctx.run_stamp
    # ---- 0. Preflight ----------------------------------------------------- #
    archs = preflight(args, workdir, rocm_dir)
    facts["gpu_arch"] = ", ".join(archs) or "none"
    variant = resolve_variant(args.variant, archs)
    facts["variant"] = variant
    facts["tier"] = args.tier
    if args.reruns > 0:
        facts["reruns"] = args.reruns

    facts["mode"] = (
        "prepare-only" if args.prepare_only else ("offline" if args.offline else "normal")
    )

    # ---- 1. Resolve + download + extract the nightly ROCm tarball ---------- #
    step("Download + extract nightly ROCm tarball")
    rocm_updated = False
    current = current_rocm_tarball(workdir)
    if args.skip_download and not (rocm_dir / "bin").is_dir():
        die(
            "--skip-download/--offline was given but there is no extracted ROCm "
            f"tree at {rocm_dir}.\n"
            "       Run once with --prepare-only on a networked node first."
        )
    if args.skip_download and (rocm_dir / "bin").is_dir():
        log(f"--skip-download: reusing existing ROCm tree ({current or 'unknown'}).")
        facts["tarball"] = current or "unknown"
        # report what is staged, not what --variant/--rocm-version asked for: no
        # download happens here, so the extracted tree is the thing under test
        parsed = parse_dist_tarball(current) if current else None
        if parsed:
            facts["variant"], facts["rocm_version"] = parsed
    else:
        # --variant auto is a best-effort hint, so let it degrade to multiarch;
        # an explicitly requested variant must resolve or abort.
        filename, url, variant = resolve_tarball(
            variant,
            args.rocm_version,
            fallback=MULTIARCH_VARIANT if args.variant == "auto" else None,
        )
        facts["variant"] = variant
        facts["tarball"] = filename
        facts["tarball_url"] = url
        parsed = parse_dist_tarball(filename)
        facts["rocm_version"] = parsed[1] if parsed else "unknown"
        if (rocm_dir / "bin").is_dir() and current == filename:
            log(f"ROCm tarball already current ({filename}); reusing extracted tree.")
        else:
            log(f"Selected tarball: {filename}")
            log(f"URL: {url}")
            if current and current != filename:
                log(f"Newer nightly available: {current} -> {filename}")
            tarball = workdir / filename
            download_file(
                url,
                tarball,
                expected_sha256=args.sha256,
                require_checksum=args.require_checksum,
            )
            verify_tarball(url, tarball, args.sha256, args.require_checksum, facts)
            extract_tarball(tarball, rocm_dir)
            set_rocm_tarball(workdir, filename)
            # the archive is now extracted into rocm_dir; drop every downloaded
            # tarball (including this one) to reclaim the multi-GB of disk space.
            cleanup_downloaded_tarballs(workdir)
            rocm_updated = True
    facts["rocm_updated"] = rocm_updated

    # sanity: the profiler binaries must be present in the tarball's bin/
    missing = [
        b for b in REQUIRED_ROCPROFSYS_BINARIES if not (rocm_dir / "bin" / b).exists()
    ]
    if missing:
        die(
            "the downloaded tarball does not contain the expected rocprof-sys "
            f"binaries in {rocm_dir / 'bin'}: {', '.join(missing)}.\n"
            "       Make sure you are using a full dist tarball (not the "
            "'-tests-' variant)."
        )
    log(
        "Verified rocprof-sys binaries present in tarball bin/: "
        + ", ".join(REQUIRED_ROCPROFSYS_BINARIES)
    )

    env = make_rocm_env(os.environ.copy(), rocm_dir)
    version_rc = report_under_test(rocm_dir, env, facts)
    capture_rocm_sanity(rocm_dir, env, workdir / f"rocm-sanity-{run_stamp}.log", facts)

    # Fail fast if the tarball binaries can't start on this machine, judged from the
    # --version probe above. Skipped when building from source (those tarball binaries
    # aren't what's tested) and during prepare-only (which may run on another node).
    if not args.build_from_source and not args.prepare_only:
        smoke_check_rocprofsys(rocm_dir, version_rc)

    facts["validation"] = (
        "build-from-source" if args.build_from_source else "tarball-install"
    )

    ctx.env = env
    ctx.rocm_updated = rocm_updated


def prepare_source(ctx: RunContext) -> None:
    """Sparse-clone or update projects/rocprofiler-systems."""
    args = ctx.args
    facts = ctx.facts
    workdir = ctx.workdir
    env = ctx.env
    # ---- 2. Sparse-clone / update develop --------------------------------- #
    step("Sparse-clone / update rocm-systems (projects/rocprofiler-systems)")
    src_dir, source_changed = sync_source(
        workdir,
        args.branch,
        env,
        no_fetch=args.skip_clone,
        submodules=args.build_from_source,
        force=args.force_sync,
    )
    repo_dir = workdir / "rocm-systems"
    facts["git_revision"] = _git_rev(repo_dir) or "unknown"
    facts["git_subject"] = (
        subprocess.run(
            ["git", "-C", str(repo_dir), "log", "-1", "--format=%s"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        or "unknown"
    )
    facts["source_changed"] = source_changed

    ctx.src_dir = src_dir
    ctx.source_changed = source_changed


def prepare_python(ctx: RunContext) -> None:
    """Create the venv and resolve the selected pytest tier."""
    args = ctx.args
    facts = ctx.facts
    workdir = ctx.workdir
    src_dir = ctx.src_dir
    run_stamp = ctx.run_stamp
    # ---- 3. Required installations for the tests -------------------------- #
    step("Install test dependencies")
    if args.install_system_deps:
        install_system_deps()
    venv_py = make_venv(workdir, src_dir, skip_install=args.skip_pip)
    capture_pip_freeze(venv_py, workdir / f"pip-freeze-{run_stamp}.txt", facts)

    # Resolve the tier now (not just before pytest) so a bad/unreadable tier
    # definition fails before the build instead of after it.
    tiers = load_test_categories(src_dir, venv_py)
    dropped = drop_label_excludes(
        tiers, [x.strip() for x in args.run_labels.split(",") if x.strip()]
    )
    if dropped:
        facts["run_labels"] = ", ".join(dropped)
    if not args.pytest_args:
        log(
            f"Tier '{args.tier}' from {TEST_CATEGORIES_REL}: "
            + " ".join(shlex.quote(a) for a in tier_selection_args(args.tier, tiers))
        )

    ctx.venv_py = venv_py
    ctx.tiers = tiers


def finish_prepare_only(ctx: RunContext) -> int:
    """Stop after network staging so an air-gapped node can run --offline."""
    args = ctx.args
    facts = ctx.facts
    workdir = ctx.workdir
    summary_log = ctx.summary_log
    if args.prepare_only:
        shell = ensure_trace_processor_shell(workdir, skip_download=False)
        if shell is None and platform.machine() in ("x86_64", "AMD64"):
            die(
                "failed to stage trace_processor_shell during --prepare-only.\n"
                "       Install wget or curl and re-run so offline Perfetto tests "
                "have a cached binary."
            )
        step("Prepare-only complete (--prepare-only)")
        log("Network prep done. The tarball, source, and venv are staged under:")
        log(f"  {workdir}")
        log("Now run the tests on the (air-gapped) GPU compute node with:")
        log(
            f"  python3 {Path(sys.argv[0]).name} --work-dir {workdir} --offline "
            f"--tier {args.tier}"
        )
        facts["result"] = "PREPARED (--prepare-only)"
        write_summary(summary_log, facts)
        return 0


def build_and_stage(ctx: RunContext):
    """Build examples or rocprofiler-systems and locate the pytest tree."""
    args = ctx.args
    facts = ctx.facts
    workdir = ctx.workdir
    rocm_dir = ctx.rocm_dir
    env = ctx.env
    venv_py = ctx.venv_py
    src_dir = ctx.src_dir
    rocm_updated = ctx.rocm_updated
    source_changed = ctx.source_changed
    # ---- 4. Build (source or examples) + prepare the test tree ------------ #
    # MPI is built whenever the machine supports it and --tier/--run-labels decide
    # what runs, so it is probed just before each configure below - never during
    # --prepare-only, whose login node has a different MPI environment than the
    # compute node that does the build.
    if args.build_from_source:
        use_mpi = detect_mpi(workdir, env, args.cmake_args, facts)
        step("Build rocprofiler-systems from source")
        _src_disable = [e.strip() for e in args.disable_examples.split(",") if e.strip()]
        build_dir = build_from_source(
            src_dir,
            workdir,
            rocm_dir,
            env,
            venv_py,
            args.jobs,
            use_mpi,
            disable_examples=_src_disable,
            extra_cmake_args=args.cmake_args,
        )
        pytest_dir = build_dir / "share" / "rocprofiler-systems" / "tests" / "pytest"
        if not (pytest_dir / "conftest.py").is_file():
            die(f"source build did not produce a pytest tree at {pytest_dir}")
        test_mode = "build"
        test_root = build_dir
        facts["build_dir"] = str(build_dir)
        # report the freshly-built rocprof-sys version (that's what's under test)
        built_avail = build_dir / "bin" / "rocprof-sys-avail"
        if built_avail.exists():
            facts["rocprofsys_version"] = _tool_version(str(built_avail), env)
        log(f"Testing build-tree binaries in: {build_dir / 'bin'}")
    else:
        step("Build examples + stage test suite into the ROCm prefix")
        disable_examples = [
            e.strip() for e in args.disable_examples.split(",") if e.strip()
        ]
        facts["disabled_examples"] = ", ".join(disable_examples) or "(none)"

        pytest_dir = rocm_dir / "share" / "rocprofiler-systems" / "tests" / "pytest"
        examples_out = rocm_dir / "share" / "rocprofiler-systems" / "examples"
        examples_present = examples_out.is_dir() and any(examples_out.iterdir())

        # Rebuild only when the inputs changed: fresh/updated ROCm tree (examples
        # were installed into the tree that was just replaced), updated source,
        # missing example artifacts, or an explicit --force-rebuild.
        need_build = (
            args.force_rebuild or rocm_updated or source_changed or not examples_present
        )
        if need_build:
            use_mpi = detect_mpi(workdir, env, args.cmake_args, facts)
            reasons = []
            if args.force_rebuild:
                reasons.append("--force-rebuild")
            if rocm_updated:
                reasons.append("rocm updated")
            if source_changed:
                reasons.append("source changed")
            if not examples_present:
                reasons.append("examples missing")
            log("Building examples (" + ", ".join(reasons) + ")")
            build_examples(
                src_dir,
                workdir,
                rocm_dir,
                env,
                args.jobs,
                use_mpi,
                disable_examples,
                extra_cmake_args=args.cmake_args,
            )
        else:
            # nothing will be configured, so the MPI probe is skipped: the examples
            # on disk already carry whatever MPI decision their build made.
            facts["mpi"] = "not probed (examples up to date)"
            log(f"Examples up to date ({facts['git_revision'][:10]}); skipping rebuild.")

        # Staging the pytest tree is cheap; refresh it whenever we rebuilt, the ROCm
        # tree changed, or the suite isn't staged yet.
        if need_build or not (pytest_dir / "conftest.py").is_file():
            pytest_dir = stage_tests(
                src_dir, rocm_dir, workdir, env, skip_download=args.skip_download
            )
        else:
            log("Test suite already staged; skipping.")

        test_mode = "install"
        test_root = rocm_dir
        facts["examples_rebuilt"] = need_build
        facts["examples_installed"] = (
            len(list(examples_out.glob("*"))) if examples_out.is_dir() else 0
        )

    return pytest_dir, test_mode, test_root


def finish_skipped(
    ctx: RunContext, pytest_dir: Path, test_mode: str, test_root: Path
) -> int:
    """Record a run that staged everything and was asked not to execute tests."""
    args = ctx.args
    facts = ctx.facts
    rocm_dir = ctx.rocm_dir
    venv_py = ctx.venv_py
    summary_log = ctx.summary_log
    if args.skip_tests:
        step("Skipping test execution (--skip-tests)")
        var = "ROCPROFSYS_BUILD_DIR" if test_mode == "build" else "ROCPROFSYS_INSTALL_DIR"
        log(f"Everything is staged under: {test_root}")
        log("To run manually:")
        log(f"  {var}={test_root} ROCM_PATH={rocm_dir} \\")
        log(f"  {venv_py} -m pytest {pytest_dir} -v")
        facts["result"] = "SKIPPED (--skip-tests)"
        write_summary(summary_log, facts)
        return 0


def execute_tests(
    ctx: RunContext, pytest_dir: Path, test_mode: str, test_root: Path
) -> int:
    """Run pytest and write the summary, failures log, and archive."""
    args = ctx.args
    facts = ctx.facts
    workdir = ctx.workdir
    rocm_dir = ctx.rocm_dir
    env = ctx.env
    venv_py = ctx.venv_py
    tiers = ctx.tiers
    run_stamp = ctx.run_stamp
    summary_log = ctx.summary_log
    step(
        f"Run pytest suite in {test_mode} mode "
        f"({'build-tree' if test_mode == 'build' else 'tarball'} rocprof-sys binaries)"
    )
    rc, junit = run_tests(
        venv_py,
        pytest_dir,
        rocm_dir,
        env,
        args.pytest_args,
        args.tier,
        tiers,
        args.reruns,
        args.reruns_delay,
        args.rerun_failed,
        workdir,
        facts,
        mode=test_mode,
        test_root=test_root,
    )

    counts = parse_junit(junit)
    facts["pytest_exit"] = rc
    facts["result"] = "PASS" if rc == 0 else "FAIL"
    if counts:
        facts["tests_total"] = counts["tests"]
        facts["passed"] = counts["passed"]
        facts["failed"] = counts["failures"]
        facts["errors"] = counts["errors"]
        facts["skipped"] = counts["skipped"]
        facts["duration_sec"] = round(counts["time"], 1)

    failed = collect_failed_tests(junit)
    if failed:
        failures_log = workdir / f"failures-{run_stamp}.log"
        write_failures_log(failures_log, failed)
        facts["failures_log"] = str(failures_log)
        facts["_failed_tests"] = [tid for tid, _, _ in failed]

    step("Summary")
    write_summary(summary_log, facts)
    return rc


def main(argv=None) -> int:
    """Run one nightly validation from argument parsing through the result archive."""
    args = parse_args(argv)
    # --offline is the single air-gapped switch: reuse the staged tarball, source,
    # and venv, and do no network I/O.
    args.skip_clone = args.offline
    args.skip_pip = args.offline
    if args.offline:
        args.skip_download = True
    if args.force_sync and args.offline:
        die("--force-sync requires network access and cannot be combined with --offline.")
    if args.force_sync:
        args.skip_clone = False

    ctx = open_run(args)
    prepare_rocm(ctx)
    prepare_source(ctx)
    prepare_python(ctx)
    if args.prepare_only:
        return finish_prepare_only(ctx)
    pytest_dir, test_mode, test_root = build_and_stage(ctx)
    if args.skip_tests:
        return finish_skipped(ctx, pytest_dir, test_mode, test_root)
    return execute_tests(ctx, pytest_dir, test_mode, test_root)
