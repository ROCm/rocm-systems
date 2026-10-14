# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build an amdhip64 with HRR capture compiled in, for the HRR GPU suite.

Interim: TheRock has no capture-enabled build yet. Once rocm-systems CI can
request TheRock's hrr-capture build variant, hrr-gpu.yml calls TheRock's
workflows instead and this script and test_hrr_capture_build.py are deleted.

Release builds compile capture out (HIP_HRR_CAPTURE_ENABLED undefined). This
script installs the ROCm artifacts that rocm-systems' own TheRock CI built for
a recent develop commit, then builds ROCr and CLR (with capture enabled) from
this checkout over them, and builds projects/hrr on top. The compiler, comgr,
the device libraries, rocprofiler-register and the sysdeps stay prebuilt. The
GPU jobs install the same artifacts plus these two runtime libraries, so build
and test use the same libraries, and the binaries' RUNPATH needs no
LD_LIBRARY_PATH.

  fetch-base     Pick the base run and install the build-time artifacts.
  build          Build ROCr, then CLR with capture, into the base; build HRR.
  verify         Check the suite's case list, the embedded GPU targets and
                 which runtime libraries the suite loads.
  install-base   On a GPU machine: install the run-time artifacts of the
                 recorded base run and the runtime libraries built here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

import hrr_test_config

Runner = Callable[[Sequence[str]], None]

BASE_REPO = "ROCm/rocm-systems"
BASE_WORKFLOW = "therock-multi-arch-ci.yml"
BASE_BRANCH = "develop"
# The generic (target-neutral) artifacts are the same in every family group.
BASE_ARTIFACT_GROUP = "gfx94X-dcgpu"

# Artifacts needed to build ROCr, CLR and projects/hrr.
BUILD_ARTIFACTS = (
    "amd-llvm_run", "amd-llvm_lib", "amd-llvm_dev",
    "base_run", "base_lib", "base_dev",
    "sysdeps_lib", "sysdeps_dev",
    "core-runtime_run", "core-runtime_lib", "core-runtime_dev",
    "core-hip_run", "core-hip_lib", "core-hip_dev",
)

# Artifacts the GPU suite loads at run time.
RUNTIME_ARTIFACTS = (
    "amd-llvm_lib",
    "base_run", "base_lib",
    "sysdeps_lib",
    "core-runtime_run", "core-runtime_lib",
    "core-hip_lib",
)

# The libraries built from this checkout that replace the base's copies.
RUNTIME_LIB_GLOBS = ("libamdhip64.so*", "libhsa-runtime64.so*")

# Everything ROCr installs into the base, relative to --dist, so its build
# replaces the base's ROCr files instead of mixing with them.
ROCR_INSTALLED = (
    "include/hsa",
    "include/hsakmt",
    "lib/cmake/hsa-runtime64",
    "lib/cmake/hsakmt",
    "lib/libhsakmt.a",
    "lib/pkgconfig/libhsakmt.pc",
    "lib/libhsa-runtime64.so*",
)

# Its targets.arch is the architecture list the GPU suite is built for.
INTEGRATION_CONFIG = Path("projects/hrr/tests/config/integration.yaml")

# What the GPU suite runs. The host unit suite is built and run by hrr-ci.yml.
HRR_TARGETS = ("hrr-playback", "hrr-integration-tests", "hrr-test-dispatch-tool")

# Uploaded by hrr-gpu.yml's build job, relative to --build.
EXECUTABLES = (
    "hrr/playback/hrr-playback",
    "hrr/tests/integration/hrr-integration-tests",
)


def run(cmd: Sequence[str]) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def pick_base_run(finder_json: dict) -> dict:
    """The base run from find_latest_artifacts.py --json output."""
    if finder_json.get("status") != "found" or not finder_json.get("results"):
        raise SystemExit(f"no {BASE_BRANCH} run of {BASE_WORKFLOW} has the needed artifacts")
    result = finder_json["results"][0]
    return {
        "repo": result["github_repository_name"],
        "run_id": str(result["workflow_run_id"]),
        "commit": result["git_commit_sha"],
        "run_url": result["workflow_run_html_url"],
        "artifact_group": result["artifact_group"],
    }


def find_base_run(therock: Path) -> dict:
    cmd = [
        sys.executable, therock / "build_tools" / "find_latest_artifacts.py",
        "--repo", BASE_REPO,
        "--workflow", BASE_WORKFLOW,
        "--branch", BASE_BRANCH,
        "--artifact-group", BASE_ARTIFACT_GROUP,
        "--json",
    ]
    for artifact in BUILD_ARTIFACTS:
        cmd += ["--require-artifact", f"{artifact}_generic"]
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    # Only stdout is captured: it carries the JSON, stderr the finder's log.
    out = subprocess.run([str(c) for c in cmd], check=True, stdout=subprocess.PIPE, text=True)
    return pick_base_run(json.loads(out.stdout))


def fetch_command(therock: Path, base: dict, dist: Path, artifacts: Sequence[str]) -> list:
    return [
        sys.executable, therock / "build_tools" / "fetch_artifacts.py",
        "--run-id", base["run_id"],
        "--run-github-repo", base["repo"],
        "--artifact-group", base["artifact_group"],
        "--output-dir", dist,
        "--flatten",
        *artifacts,
    ]


def rocr_configure_command(src: Path, build: Path, dist: Path) -> list:
    sysdeps = dist / "lib" / "rocm_sysdeps"
    cmake_dir = dist / "lib" / "llvm" / "lib" / "cmake"
    # The sysdeps provide libelf, libdrm and libnuma, as in TheRock.
    return [
        "env", f"PKG_CONFIG_PATH={sysdeps / 'lib' / 'pkgconfig'}",
        "cmake", "-S", src / "projects" / "rocr-runtime", "-B", build / "rocr",
        "-GNinja", "-Wno-dev",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        f"-DCMAKE_PREFIX_PATH={dist};{sysdeps}",
        f"-DCMAKE_INSTALL_PREFIX={dist}",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        # TheRock's libhsa-runtime64 RUNPATH.
        "-DCMAKE_INSTALL_RPATH=$ORIGIN;$ORIGIN/rocm_sysdeps/lib",
        # The blit and trap-handler kernels need the base's clang; ROCr's own
        # hint (${CMAKE_PREFIX_PATH}/llvm) does not work with a list.
        f"-DClang_DIR={cmake_dir / 'clang'}",
        f"-DLLVM_DIR={cmake_dir / 'llvm'}",
    ]


def clr_configure_command(src: Path, build: Path, dist: Path) -> list:
    # HIP_HRR_CAPTURE_ENABLED is what compiles capture into amdhip64; CLR has
    # no CMake option for it.
    return [
        "cmake", "-S", src / "projects" / "clr", "-B", build / "clr", "-GNinja", "-Wno-dev",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_FLAGS=-DHIP_HRR_CAPTURE_ENABLED",
        f"-DCMAKE_PREFIX_PATH={dist}",
        f"-DCMAKE_INSTALL_PREFIX={dist}",
        # The artifacts use lib/; GNUInstallDirs would pick lib64 on this image.
        "-DCMAKE_INSTALL_LIBDIR=lib",
        # libhiprtc sets no RUNPATH of its own; comgr and ROCr sit next to it.
        "-DCMAKE_INSTALL_RPATH=$ORIGIN",
        f"-DROCM_PATH={dist}",
        f"-DHIP_COMMON_DIR={src / 'projects' / 'hip'}",
        "-DHIP_PLATFORM=amd",
        "-DCLR_BUILD_HIP=ON",
        "-DCLR_BUILD_OCL=OFF",
        "-DHIPCC_BIN_DIR=",
    ]


def hrr_configure_command(src: Path, build: Path, dist: Path, archs: str) -> list:
    llvm = dist / "lib" / "llvm"
    # One compiler for C, C++ and HIP, as TheRock builds HRR: host objects from
    # the system compiler are not position independent, and the HIP link is.
    return [
        "cmake", "-S", src / "projects" / "hrr", "-B", build / "hrr", "-GNinja", "-Wno-dev",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={dist}",
        f"-DROCM_PATH={dist}",
        f"-DCMAKE_C_COMPILER={llvm / 'bin' / 'clang'}",
        f"-DCMAKE_CXX_COMPILER={llvm / 'bin' / 'clang++'}",
        f"-DCMAKE_HIP_COMPILER={llvm / 'bin' / 'clang++'}",
        f"-DCMAKE_HIP_FLAGS=--rocm-device-lib-path={llvm / 'amdgcn' / 'bitcode'}",
        f"-DGPU_TARGETS={archs}",
        f"-DCMAKE_HIP_ARCHITECTURES={archs}",
        "-DHRR_BUILD_PLAYBACK=ON",
        "-DHRR_BUILD_TESTS=ON",
        "-DHRR_BUILD_INTEGRATION_TESTS=ON",
    ]


def remove_paths(root: Path, patterns: Sequence[str]) -> None:
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()


def remove_runtime_libs(lib_dir: Path) -> None:
    """Remove the base's runtime libraries so only the ones built here remain."""
    remove_paths(lib_dir, RUNTIME_LIB_GLOBS)


def copy_runtime_libs(src_dir: Path, dst_dir: Path) -> list[str]:
    """Copy libamdhip64.so* and libhsa-runtime64.so*, keeping symlinks."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for pattern in RUNTIME_LIB_GLOBS:
        found = sorted(src_dir.glob(pattern))
        if not found:
            raise SystemExit(f"no {pattern} in {src_dir}")
        for path in found:
            target = dst_dir / path.name
            if target.is_symlink() or target.exists():
                target.unlink()
            if path.is_symlink():
                os.symlink(os.readlink(path), target)
            else:
                shutil.copy2(path, target)
            copied.append(path.name)
    return copied


def linkage_errors(ldd_output: str, lib_dir: Path) -> list[str]:
    """Each runtime library must resolve to exactly one file, in lib_dir."""
    resolved: dict[str, set[str]] = {}
    for match in re.finditer(r"^\s*(\S+) => (\S+)", ldd_output, re.MULTILINE):
        name, path = match.groups()
        for pattern in RUNTIME_LIB_GLOBS:
            stem = pattern.rstrip("*")
            if name.startswith(stem):
                resolved.setdefault(stem, set()).add(path)
    errors = []
    for pattern in RUNTIME_LIB_GLOBS:
        stem = pattern.rstrip("*")
        paths = resolved.get(stem, set())
        if len(paths) != 1:
            errors.append(f"{stem} resolves to {sorted(paths) or 'nothing'}")
        elif Path(next(iter(paths))).parent != lib_dir:
            errors.append(f"{stem} resolves to {next(iter(paths))}, not into {lib_dir}")
    return errors


def check_linkage(binary: Path, lib_dir: Path) -> int:
    out = subprocess.run(["ldd", str(binary)], check=True, capture_output=True, text=True).stdout
    errors = linkage_errors(out, lib_dir)
    for error in errors:
        print(f"::error::{binary.name}: {error}")
    if not errors:
        print(f"{binary.name} loads the runtime libraries from {lib_dir}")
    return 1 if errors else 0


def embedded_targets(objdump_output: str) -> set[str]:
    return set(re.findall(r"\bgfx[0-9a-f]+\b", objdump_output))


def missing_targets(archs: str, objdump_output: str) -> list[str]:
    found = embedded_targets(objdump_output)
    return [a for a in archs.split(";") if a and a not in found]


def cmd_fetch_base(args: argparse.Namespace, runner: Runner = run) -> int:
    base = json.loads(args.base_json.read_text()) if args.base_json.is_file() else None
    if base is None:
        base = find_base_run(args.therock)
        args.base_json.parent.mkdir(parents=True, exist_ok=True)
        args.base_json.write_text(json.dumps(base, indent=2) + "\n")
    print(f"Base: {base['repo']} run {base['run_id']} at {base['commit']}")
    runner(fetch_command(args.therock, base, args.dist, BUILD_ARTIFACTS))
    return 0


def cmd_build(args: argparse.Namespace, runner: Runner = run) -> int:
    runner(rocr_configure_command(args.src, args.build, args.dist))
    runner(["cmake", "--build", args.build / "rocr"])
    remove_paths(args.dist, ROCR_INSTALLED)
    runner(["cmake", "--install", args.build / "rocr"])
    # CLR finds the ROCr just installed through its hsa-runtime64 package.
    runner(clr_configure_command(args.src, args.build, args.dist))
    runner(["cmake", "--build", args.build / "clr"])
    remove_paths(args.dist / "lib", ("libamdhip64.so*",))
    runner(["cmake", "--install", args.build / "clr"])
    copy_runtime_libs(args.dist / "lib", args.runtime_lib)
    runner(hrr_configure_command(args.src, args.build, args.dist, args.archs))
    runner(["cmake", "--build", args.build / "hrr", "--target", *HRR_TARGETS])
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    exe = args.build / "hrr" / "tests" / "integration" / "hrr-integration-tests"
    run([
        sys.executable, args.src / "projects/hrr/tests/scripts/hrr_test_config.py", "check",
        "--binary", exe,
        "--config", args.src / "projects/hrr/tests/config/integration.yaml",
    ])
    objdump = args.dist / "lib" / "llvm" / "bin" / "llvm-objdump"
    out = subprocess.run(
        [str(objdump), "--offloading", str(exe)], check=True, capture_output=True, text=True
    ).stdout
    missing = missing_targets(args.archs, out)
    print("embedded: " + " ".join(sorted(embedded_targets(out))))
    if missing:
        print(f"::error::missing from the fat binary: {' '.join(missing)}")
        return 1
    return check_linkage(exe, args.dist / "lib")


def restore_executable_bits(build: Path) -> None:
    """Artifact downloads drop the executable bit."""
    for rel in EXECUTABLES:
        path = build / rel
        path.chmod(path.stat().st_mode | 0o111)


def cmd_install_base(args: argparse.Namespace, runner: Runner = run) -> int:
    base = json.loads(args.base_json.read_text())
    print(f"Base: {base['repo']} run {base['run_id']} at {base['commit']}")
    runner(fetch_command(args.therock, base, args.dist, RUNTIME_ARTIFACTS))
    remove_runtime_libs(args.dist / "lib")
    copy_runtime_libs(args.runtime_lib, args.dist / "lib")
    restore_executable_bits(args.build)
    return check_linkage(
        args.build / "hrr" / "tests" / "integration" / "hrr-integration-tests", args.dist / "lib"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("fetch-base", "build", "verify", "install-base"))
    parser.add_argument("--therock", type=Path, default=Path("therock"))
    parser.add_argument("--src", type=Path, default=Path.cwd())
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--runtime-lib", type=Path, default=Path("build/runtime-lib"))
    parser.add_argument("--base-json", type=Path, default=Path("build/base.json"))
    parser.add_argument(
        "--archs", default="",
        help=f"Semicolon-separated GPU targets; default targets.arch of {INTEGRATION_CONFIG}",
    )
    args = parser.parse_args(argv)
    for name in ("therock", "src", "dist", "build", "runtime_lib", "base_json"):
        setattr(args, name, getattr(args, name).resolve())
    if args.command in ("build", "verify") and not args.archs:
        config = hrr_test_config.load(args.src / INTEGRATION_CONFIG)
        args.archs = ";".join(config["targets"]["arch"])
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return {
        "fetch-base": cmd_fetch_base,
        "build": cmd_build,
        "verify": cmd_verify,
        "install-base": cmd_install_base,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
