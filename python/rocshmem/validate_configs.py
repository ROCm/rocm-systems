#!/usr/bin/env python3
"""Validate that the standalone rocshmem4py wheel builds and imports against
every meaningful rocSHMEM backend configuration.

The heap allocator is a runtime choice (ROCSHMEM_HEAP_ALLOCATOR_TYPE) since
ROCm-systems PR #9432, so it is not part of this build matrix.

For each configuration this driver, in full isolation:
  1. builds + installs rocSHMEM into a per-config prefix (build dir != prefix),
  2. creates a fresh venv,
  3. builds + installs the rocshmem4py wheel into that venv against the prefix
     (python -m build, run by the venv's own interpreter),
  4. verifies import, version tag, embedded GPU arches, and that the
     configuration's transitive dependencies resolved (NUMA for GDA/SDMA, an
     MPI NEEDED entry for RO),
  5. optionally runs the torch-free api_compat suite under a launcher.

It prints a pass/fail table. Run it with the interpreter you want the venvs to
be cloned from:

    python validate_configs.py --src /path/to/rocm-systems/projects/rocshmem \
        --work /tmp/rs-validate --gpu-targets gfx942 \
        --configs ipc_single,gda,all_backends --run-tests

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent          # python/rocshmem


def _onoff(v):
    return "ON" if v else "OFF"


def _backends(ro, ipc, gda, sdma=0, single_node=None):
    flags = [f"-DUSE_RO={_onoff(ro)}", f"-DUSE_IPC={_onoff(ipc)}",
             f"-DUSE_GDA={_onoff(gda)}", f"-DUSE_SDMA={_onoff(sdma)}"]
    if single_node is not None:
        flags.append(f"-DUSE_SINGLE_NODE={_onoff(single_node)}")
    return flags


# The matrix now varies only the *backend* (RO/IPC/GDA/SDMA). The heap allocator
# is no longer a build option: ROCm-systems PR #9432 removed the five
# USE_HEAP_DEVICE_* flags in favour of the runtime ROCSHMEM_HEAP_ALLOCATOR_TYPE
# env var (finegrained/uncached/coarsegrained/vmm_posix/vmm_fabric), so allocator
# coverage belongs in a runtime test pass, not here.
#
# name -> (rocSHMEM cmake flags, expects_numa). MPI is deliberately NOT keyed off
# the backend: rocSHMEM defaults USE_EXTERNAL_MPI=AUTO, so libmpi is linked for
# *every* config -- IPC-only included -- whenever MPI was present at library
# build time, so `libmpi in NEEDED` is reported for information, not gated.
# expects_numa is True for GDA/SDMA (they export find_dependency(NUMA)).
CONFIGS = {
    "ipc_single":   (_backends(0, 1, 0, single_node=True), False),
    "ro_net":       (_backends(1, 0, 0),                    False),
    "ro_ipc":       (_backends(1, 1, 0),                    False),
    "gda":          (_backends(0, 1, 1),                    True),
    "sdma":         (_backends(0, 1, 0, sdma=1),            True),
    "all_backends": (_backends(1, 1, 1),                    True),
}


def log(msg):
    print(f"[validate] {msg}", flush=True)


def run(cmd, log_path, env=None):
    """Run a command, teeing combined output to log_path. Returns exit code."""
    with open(log_path, "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return p.returncode


def build_library(name, flags, args):
    build_dir = Path(args.work) / f"build-{name}"
    prefix = Path(args.work) / f"prefix-{name}"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    if prefix.exists():
        shutil.rmtree(prefix)

    base = [
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        f"-DGPU_TARGETS={args.gpu_targets}",
        "-DBUILD_FUNCTIONAL_TESTS=OFF", "-DBUILD_UNIT_TESTS=OFF",
        "-DBUILD_EXAMPLES=OFF", "-DBUILD_CTESTS=OFF",
    ]
    env = os.environ.copy()
    env["ROCM_PATH"] = args.rocm_path

    cfg_log = Path(args.work) / f"{name}.lib-configure.log"
    if run(["cmake", "-S", args.src, "-B", str(build_dir), *base, *flags], cfg_log, env):
        return None, f"library configure failed ({cfg_log})"
    bld_log = Path(args.work) / f"{name}.lib-build.log"
    if run(["cmake", "--build", str(build_dir), "--parallel", str(args.jobs)], bld_log, env):
        return None, f"library build failed ({bld_log})"
    if run(["cmake", "--install", str(build_dir)], Path(args.work) / f"{name}.lib-install.log", env):
        return None, "library install failed"
    return prefix, None


def make_venv(name, args):
    venv_dir = Path(args.work) / f"venv-{name}"
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    venv.create(venv_dir, with_pip=True)
    return venv_dir / "bin" / "python"


def build_wheel_into_venv(name, venv_py, prefix, args):
    env = os.environ.copy()
    # rocSHMEM is discovered via find_package(rocshmem CONFIG) on CMAKE_PREFIX_PATH
    # (setup.py forwards this as a -D cache var). Its NUMA / rocprofiler-register /
    # hsakmt transitive deps are resolved by rocSHMEM's own exported config.
    env["CMAKE_PREFIX_PATH"] = str(prefix)
    env["ROCM_PATH"] = args.rocm_path
    # Wipe in-tree build artifacts between configs. setuptools' build_ext caches
    # the CMake build under build/temp*/, and find_package(rocshmem) caches the
    # resolved rocshmem_DIR there; without this a later config silently relinks
    # the previous config's rocSHMEM even after CMAKE_PREFIX_PATH changes.
    for junk in (HERE / "build", HERE / "dist", *HERE.glob("*.egg-info")):
        if junk.exists():
            shutil.rmtree(junk)
    dist = HERE / "dist"
    # Build the wheel with the venv's own interpreter (python -m build --wheel),
    # so the build/install/verify chain all runs under one interpreter. HERE is
    # passed explicitly as the source dir since run() does not set a cwd.
    cmd = [str(venv_py), "-m", "build", "--wheel"]
    if args.no_isolation:
        # python -m build won't fetch build deps under --no-isolation, so
        # preinstall them. setuptools>=61 is required or the wheel is named
        # "UNKNOWN" (older setuptools ignores the PEP 621 [project] table).
        if run([str(venv_py), "-m", "pip", "install", "--upgrade", "build",
                "setuptools>=61.0", "wheel", "cmake>=3.20", "nanobind>=2.12.0,<3.0"],
               Path(args.work) / f"{name}.builddeps.log", env):
            return "build-dep install failed"
        cmd.append("--no-isolation")
    cmd.append(str(HERE))
    if run(cmd, Path(args.work) / f"{name}.wheel.log", env):
        return f"wheel build failed ({Path(args.work) / f'{name}.wheel.log'})"
    wheels = list(dist.glob("*.whl"))
    if not wheels:
        return "no wheel produced"
    if run([str(venv_py), "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheels[0])],
           Path(args.work) / f"{name}.pipinstall.log", env):
        return "wheel install failed"
    return None


def _so_path(venv_py, env, cwd):
    out = subprocess.run(
        [str(venv_py), "-c", "import importlib.util as u; print(u.find_spec('_rocshmem4py').origin)"],
        capture_output=True, text=True, env=env, cwd=cwd)
    return out.stdout.strip() if out.returncode == 0 else ""


def _so_arches(so):
    """GPU arches with actual code objects embedded in the extension. Uses
    roc-obj-ls (ground truth: enumerates the bundled code objects); falls back
    to scanning strings, which can over-report arches that only appear in a
    baked-in default target list rather than as real code objects."""
    objs = subprocess.run(["roc-obj-ls", so], capture_output=True, text=True)
    if objs.returncode == 0 and "gfx" in objs.stdout:
        return sorted(set(re.findall(r"amdgcn-amd-amdhsa--(gfx[0-9a-z]+)", objs.stdout)))
    raw = subprocess.run(["strings", "-a", so], capture_output=True, text=True).stdout
    return sorted(set(re.findall(r"amdgcn-amd-amdhsa--(gfx[0-9a-z]+)", raw)))


def verify(name, venv_py, prefix, expects_numa, args):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [f"{prefix}/lib", f"{args.rocm_path}/lib", env.get("LD_LIBRARY_PATH", "")])
    problems = []
    # Verify from a neutral CWD, never the source tree: a prior `pip install -e .`
    # leaves an in-place _rocshmem4py*.so (and the pure-python rocshmem4py/ pkg)
    # in the source root, which sys.path[0] would otherwise import in preference
    # to the venv's freshly built wheel.
    neutral = str(args.work)

    imp = subprocess.run(
        [str(venv_py), "-c",
         "import rocshmem4py as r; print(r.__version__); print(r.__rocshmem_version__)"],
        capture_output=True, text=True, env=env, cwd=neutral)
    if imp.returncode != 0:
        return [f"import failed: {imp.stderr.strip().splitlines()[-1:] or imp.stdout.strip()}"], {}
    ver, rocshmem_ver = (imp.stdout.strip().splitlines() + ["", ""])[:2]

    info = {"version": ver, "rocshmem": rocshmem_ver}
    # A well-built wheel reports "<base>+rocshmem<libver>" via importlib.metadata.
    # "0.0.0+unknown" means the distribution metadata was not found -- usually a
    # setuptools <61 build that produced an "UNKNOWN" dist name.
    if "unknown" in ver or "rocshmem" not in ver:
        problems.append(f"bad version metadata '{ver}' (setuptools<61 / UNKNOWN dist?)")
    so = _so_path(venv_py, env, neutral)
    if so:
        arches = _so_arches(so)
        raw = subprocess.run(["strings", "-a", so], capture_output=True, text=True).stdout
        githash = (re.findall(r"\b[0-9a-f]{40}\b", raw) or [""])[0]
        info["arches"] = ",".join(arches)
        info["git"] = githash[:10]
        bc = sorted({re.sub(r".*librocshmem_device_(.+)\.bc", r"\1", p.name)
                     for p in (prefix / "lib").glob("librocshmem_device_*.bc")})
        if arches != bc:
            problems.append(f"arch mismatch: .so={arches} vs install={bc}")
        # libmpi is NEEDED for every config built with USE_EXTERNAL_MPI!=OFF
        # (the AUTO default), not just RO -- reported for information, not gated.
        needed = subprocess.run(["readelf", "-dW", so], capture_output=True, text=True).stdout
        info["mpi_needed"] = "libmpi" in needed
    # Confirm the NUMA transitive dep was actually recorded + resolved: the build
    # only got here if find_package(rocshmem) succeeded, but assert the
    # config-level expectation to catch a silently-degraded library. Since PR
    # #9583, rocSHMEM records it two ways depending on the ROCm it was built on:
    #   ROCm >= 7.13: find_dependency(NUMA) (with a rocm_sysdeps prefix inject),
    #   ROCm <  7.13: a baked numa::numa target pulled in via a generated
    #                 rocshmem-numa-targets.cmake include (no config-mode NUMA
    #                 package exists on those hosts).
    cmake_dir = prefix / "lib/cmake/rocshmem"
    cfg = (cmake_dir / "rocshmem-config.cmake").read_text()
    if expects_numa:
        baked = cmake_dir / "rocshmem-numa-targets.cmake"
        numa_ok = "find_dependency(NUMA)" in cfg or (
            "rocshmem-numa-targets.cmake" in cfg and baked.exists())
        if not numa_ok:
            problems.append("NUMA transitive dep not recorded by rocshmem-config.cmake "
                            "(neither find_dependency(NUMA) nor a baked numa::numa target)")
    return problems, info


def run_tests(name, venv_py, prefix, args):
    env = os.environ.copy()
    env["CMAKE_PREFIX_PATH"] = str(prefix)
    env["ROCM_PATH"] = args.rocm_path
    # The statically-linked extension still needs the ROCm runtime (and any
    # rocSHMEM shared deps) on the loader path at import time.
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [f"{prefix}/lib", f"{args.rocm_path}/lib", env.get("LD_LIBRARY_PATH", "")])
    testfile = str(HERE / "tests/test_api_compat.py")
    # api_compat is torch-free and single-PE safe. Launch one PE directly so
    # conftest's rocSHMEM init/finalize path is exercised. torchrun re-invokes
    # this interpreter as workers (needs torch in the venv); mpirun launches it
    # explicitly and sets OMPI_COMM_WORLD_SIZE for conftest's PE detection.
    if args.launcher == "torchrun":
        launch = [str(venv_py), "-m", "torch.distributed.run",
                  "--standalone", "--nnodes=1", "--nproc_per_node=1"]
    else:
        launch = ["mpirun", "--allow-run-as-root", "-n", "1",
                  "-x", "LD_LIBRARY_PATH", str(venv_py)]
    rc = run([*launch, "-m", "pytest", testfile, "-q"],
             Path(args.work) / f"{name}.test.log", env)
    return None if rc == 0 else f"tests failed ({Path(args.work) / f'{name}.test.log'})"


def main(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Path to rocm-systems/projects/rocshmem (library source).")
    p.add_argument("--work", default="/tmp/rs-validate", help="Scratch dir for build dirs, prefixes, venvs.")
    p.add_argument("--rocm-path", default=os.environ.get("ROCM_PATH", "/opt/rocm"))
    p.add_argument("--gpu-targets", default="gfx942", help="e.g. gfx942 or gfx942,gfx950")
    p.add_argument("--configs", default=",".join(CONFIGS),
                   help="Comma-separated subset of: " + ",".join(CONFIGS))
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    p.add_argument("--run-tests", action="store_true", help="Also run api_compat under a launcher.")
    p.add_argument("--launcher", choices=("torchrun", "mpirun"), default="mpirun")
    p.add_argument("--no-isolation", action="store_true",
                   help="Preinstall build deps into each venv and skip pip/build isolation "
                        "(for offline / air-gapped boxes).")
    args = p.parse_args(argv)

    Path(args.work).mkdir(parents=True, exist_ok=True)
    selected = [c.strip() for c in args.configs.split(",") if c.strip()]
    results = []

    for name in selected:
        if name not in CONFIGS:
            results.append((name, "SKIP", "unknown config", {}))
            continue
        flags, exp_numa = CONFIGS[name]
        log(f"=== {name} ===")
        prefix, err = build_library(name, flags, args)
        if err:
            results.append((name, "FAIL", err, {}))
            continue
        venv_py = make_venv(name, args)
        err = build_wheel_into_venv(name, venv_py, prefix, args)
        if err:
            results.append((name, "FAIL", err, {}))
            continue
        problems, info = verify(name, venv_py, prefix, exp_numa, args)
        if not problems and args.run_tests:
            err = run_tests(name, venv_py, prefix, args)
            if err:
                problems.append(err)
        status = "PASS" if not problems else "FAIL"
        results.append((name, status, "; ".join(problems) or "ok", info))
        log(f"{name}: {status} {info}")

    print("\n==================== validation summary ====================")
    print(f"{'config':<14} {'status':<6} {'version':<26} {'arches':<20} detail")
    npass = 0
    for name, status, detail, info in results:
        npass += status == "PASS"
        print(f"{name:<14} {status:<6} {info.get('version',''):<26} "
              f"{info.get('arches',''):<20} {detail if status != 'PASS' else ''}")
    print(f"\n{npass}/{len(results)} configs passed")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
