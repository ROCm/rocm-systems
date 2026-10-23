# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build an amdhip64 with HRR capture compiled in, for the HRR GPU suite.

Interim: this rebuilds CLR with capture enabled on top of TheRock release
artifacts. It stays until TheRock can build and run the HRR integration
suite. The hrr-capture variant alone does not: no workflow selects it, and
it does not install the test binaries. This script and
test_hrr_capture_build.py go away once that suite runs from TheRock artifacts.

Release builds compile capture out (HIP_HRR_CAPTURE_ENABLED undefined). This
script installs the ROCm artifacts that rocm-systems' own TheRock CI built for
a recent develop commit, rebuilds only CLR from this checkout with capture
enabled, installs it over those artifacts, and builds projects/hrr on top. HRR
lives in projects/hrr and projects/clr only; ROCr and everything else stay
prebuilt. The GPU jobs then install the same artifacts plus this amdhip64, so
build and test use the same libraries, and the binaries' RUNPATH needs no
LD_LIBRARY_PATH.

  fetch-base     Pick the base run and install the build-time artifacts.
  build          Build CLR with capture, install it into the base, build HRR.
  verify         Check the suite's case list, the embedded GPU targets and
                 which amdhip64 and ROCr the suite loads.
  install-base   On a GPU machine: install the run-time artifacts of the
                 recorded base run and the capture-enabled amdhip64.
"""

from __future__ import annotations

import argparse
import hashlib
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

# Artifacts needed to build CLR and projects/hrr.
BUILD_ARTIFACTS = (
    "amd-llvm_run",
    "amd-llvm_lib",
    "amd-llvm_dev",
    "base_run",
    "base_lib",
    "base_dev",
    "sysdeps_lib",
    "sysdeps_dev",
    "core-runtime_run",
    "core-runtime_lib",
    "core-runtime_dev",
    "core-hip_run",
    "core-hip_lib",
    "core-hip_dev",
)

# Artifacts the GPU suite loads at run time.
RUNTIME_ARTIFACTS = (
    "amd-llvm_lib",
    "base_run",
    "base_lib",
    "sysdeps_lib",
    "core-runtime_run",
    "core-runtime_lib",
    "core-hip_lib",
)

# What CLR's install writes into lib/: amdhip64 plus hiprtc and its builtins.
# All of it comes from this checkout, so all of it is carried to the test job.
CAPTURE_LIB_GLOBS = ("libamdhip64.so*", "libhiprtc*.so*")

# The suite must load exactly one copy of each, from the installed tree: the
# capture-enabled amdhip64 built here and the base's ROCr.
LOADED_LIBS = ("libamdhip64.so", "libhsa-runtime64.so")

# Its targets.arch is the architecture list the GPU suite is built for.
INTEGRATION_CONFIG = Path("projects/hrr/tests/config/integration.yaml")

# What the GPU suite runs. The host unit suite is built and run by hrr-ci.yml.
HRR_TARGETS = ("hrr-playback", "hrr-integration-tests", "hrr-test-dispatch-tool")

# What the capture cache key hashes. ci.yaml is a suite-selection file and is
# deliberately absent: a family or backend edit must restore this build.
CACHE_SOURCE_PATHS = (
    "projects/clr",
    "projects/hip",
    "projects/hrr/CMakeLists.txt",
    "projects/hrr/include",
    "projects/hrr/playback",
    "projects/hrr/tests/CMakeLists.txt",
    "projects/hrr/tests/include",
    "projects/hrr/tests/integration",
    "projects/hrr/tests/scripts/hrr_capture_build.py",
)

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
        raise SystemExit(
            f"no {BASE_BRANCH} run of {BASE_WORKFLOW} has the needed artifacts"
        )
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
        sys.executable,
        therock / "build_tools" / "find_latest_artifacts.py",
        "--repo",
        BASE_REPO,
        "--workflow",
        BASE_WORKFLOW,
        "--branch",
        BASE_BRANCH,
        "--artifact-group",
        BASE_ARTIFACT_GROUP,
        "--json",
    ]
    for artifact in BUILD_ARTIFACTS:
        cmd += ["--require-artifact", f"{artifact}_generic"]
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    # Only stdout is captured: it carries the JSON, stderr the finder's log.
    out = subprocess.run(
        [str(c) for c in cmd], check=True, stdout=subprocess.PIPE, text=True
    )
    return pick_base_run(json.loads(out.stdout))


def fetch_command(
    therock: Path, base: dict, dist: Path, artifacts: Sequence[str]
) -> list:
    return [
        sys.executable,
        therock / "build_tools" / "fetch_artifacts.py",
        "--run-id",
        base["run_id"],
        "--run-github-repo",
        base["repo"],
        "--artifact-group",
        base["artifact_group"],
        "--output-dir",
        dist,
        "--flatten",
        *artifacts,
    ]


def clr_configure_command(src: Path, build: Path, dist: Path) -> list:
    # HIP_HRR_CAPTURE_ENABLED is what compiles capture into amdhip64; CLR has
    # no CMake option for it.
    return [
        "cmake",
        "-S",
        src / "projects" / "clr",
        "-B",
        build / "clr",
        "-GNinja",
        "-Wno-dev",
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


BUILD_RPATH = "$ORIGIN/../../../../dist/lib;$ORIGIN/../../../dist/lib"


def hrr_configure_command(src: Path, build: Path, dist: Path, archs: str) -> list:
    llvm = dist / "lib" / "llvm"
    # One compiler for C, C++ and HIP, as TheRock builds HRR: host objects from
    # the system compiler are not position independent, and the HIP link is.
    return [
        "cmake",
        "-S",
        src / "projects" / "hrr",
        "-B",
        build / "hrr",
        "-GNinja",
        "-Wno-dev",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={dist}",
        f"-DROCM_PATH={dist}",
        # CMake bakes the absolute dist/lib into RUNPATH, which is the build
        # container's workspace path. The rocJitsu jobs run on the bare runner,
        # whose workspace path differs. These entries reach dist/lib from the
        # integration binary and from the playback binary, wherever the
        # workspace sits.
        f"-DCMAKE_BUILD_RPATH={BUILD_RPATH}",
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


def remove_amdhip64(lib_dir: Path) -> None:
    """Remove the release amdhip64 and hiprtc so only the CLR build's remain."""
    for glob in CAPTURE_LIB_GLOBS:
        for path in lib_dir.glob(glob):
            path.unlink()


def copy_amdhip64(src_dir: Path, dst_dir: Path) -> list[str]:
    """Copy the CAPTURE_LIB_GLOBS libraries from src_dir to dst_dir, keeping symlinks."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for glob in CAPTURE_LIB_GLOBS:
        for path in sorted(src_dir.glob(glob)):
            target = dst_dir / path.name
            if target.is_symlink() or target.exists():
                target.unlink()
            if path.is_symlink():
                os.symlink(os.readlink(path), target)
            else:
                shutil.copy2(path, target)
            copied.append(path.name)
    if not any(name.startswith("libamdhip64.so") for name in copied):
        raise SystemExit(f"no libamdhip64.so* in {src_dir}")
    return copied


def linkage_errors(ldd_output: str, lib_dir: Path) -> list[str]:
    """Each of LOADED_LIBS must resolve to exactly one file, in lib_dir."""
    resolved: dict[str, set[str]] = {}
    for match in re.finditer(r"^\s*(\S+) => (\S+)", ldd_output, re.MULTILINE):
        name, path = match.groups()
        # A relative RUNPATH entry shows up as .../build/hrr/../../dist/lib.
        path = os.path.normpath(path)
        for stem in LOADED_LIBS:
            if name.startswith(stem):
                resolved.setdefault(stem, set()).add(path)
    errors = []
    for stem in LOADED_LIBS:
        paths = resolved.get(stem, set())
        if len(paths) != 1:
            errors.append(f"{stem} resolves to {sorted(paths) or 'nothing'}")
        elif Path(next(iter(paths))).parent != lib_dir:
            errors.append(f"{stem} resolves to {next(iter(paths))}, not into {lib_dir}")
    return errors


def staged_library_sources(ldd_output: str, lib_dir: Path) -> list[Path]:
    """Libraries ldd resolved outside lib_dir and outside the system directories.

    The integration binary's RUNPATH searches dist/lib, which is where it loads
    amdhip64. A cache restore skips the build, so a dependency that the build
    tree had and the base artifacts do not (librocm_kpack.so.0) has to be
    copied here for the restored binary to start.
    """
    system_roots = (Path("/lib"), Path("/lib64"), Path("/usr/lib"), Path("/usr/lib64"))
    sources: list[Path] = []
    for match in re.finditer(r"^\s*\S+ => (\S+)", ldd_output, re.MULTILINE):
        raw = match.group(1)
        if raw == "not":
            continue
        src = Path(raw)
        if src.parent == lib_dir:
            continue
        if any(src.is_relative_to(root) for root in system_roots):
            continue
        sources.append(src)
    return sources


def stage_runtime_libs(
    binary: Path, lib_dir: Path, env: dict | None = None
) -> list[str]:
    """Copy non-system libraries the binary loads into lib_dir. Returns names copied."""
    if not binary.is_file():
        return []
    out = subprocess.run(
        ["ldd", str(binary)], check=True, capture_output=True, text=True, env=env
    ).stdout
    lib_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for src in staged_library_sources(out, lib_dir):
        if not src.is_file():
            continue
        dest = lib_dir / src.name
        if dest.exists():
            continue
        shutil.copy2(src, dest, follow_symlinks=True)
        copied.append(dest.name)
        print(f"staged {dest.name} from {src}")
    return copied


def check_linkage(binary: Path, lib_dir: Path) -> int:
    out = subprocess.run(
        ["ldd", str(binary)], check=True, capture_output=True, text=True
    ).stdout
    errors = linkage_errors(out, lib_dir)
    for error in errors:
        print(f"::error::{binary.name}: {error}")
    if not errors:
        print(f"{binary.name} loads {' and '.join(LOADED_LIBS)} from {lib_dir}")
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
    runner(clr_configure_command(args.src, args.build, args.dist))
    runner(["cmake", "--build", args.build / "clr"])
    remove_amdhip64(args.dist / "lib")
    runner(["cmake", "--install", args.build / "clr"])
    copy_amdhip64(args.dist / "lib", args.capture_lib)
    runner(hrr_configure_command(args.src, args.build, args.dist, args.archs))
    runner(["cmake", "--build", args.build / "hrr", "--target", *HRR_TARGETS])
    stage_runtime_libs(
        args.build / "hrr" / "tests" / "integration" / "hrr-integration-tests",
        args.dist / "lib",
    )
    return 0


def cmd_stage(args: argparse.Namespace) -> int:
    """Stage the runtime libraries again after a cache restore.

    The cache keeps the binary but not the copies build staged into dist/lib.
    Without the build tree, ldd finds the libraries only in the installed
    subtrees (dist/lib/llvm/lib, dist/lib/rocm_sysdeps/lib), so put those on
    its search path.
    """
    lib_dir = args.dist / "lib"
    search = [str(p) for p in sorted(lib_dir.glob("*/lib")) if p.is_dir()]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        search + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    )
    stage_runtime_libs(
        args.build / "hrr" / "tests" / "integration" / "hrr-integration-tests",
        lib_dir,
        env,
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    exe = args.build / "hrr" / "tests" / "integration" / "hrr-integration-tests"
    run(
        [
            sys.executable,
            args.src / "projects/hrr/tests/scripts/hrr_test_config.py",
            "check",
            "--binary",
            exe,
            "--config",
            args.src / "projects/hrr/tests/config/integration.yaml",
        ]
    )
    objdump = args.dist / "lib" / "llvm" / "bin" / "llvm-objdump"
    out = subprocess.run(
        [str(objdump), "--offloading", str(exe)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    missing = missing_targets(args.archs, out)
    print("embedded: " + " ".join(sorted(embedded_targets(out))))
    if missing:
        print(f"::error::missing from the fat binary: {' '.join(missing)}")
        return 1
    return check_linkage(exe, args.dist / "lib")


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        for item in files:
            digest.update(item.relative_to(path).as_posix().encode())
            digest.update(b"\0")
            digest.update(item.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def cache_key_digest(
    src: Path, base: dict, therock_pin: str, image: str, archs: str
) -> str:
    """Content key for the capture build. Not the merge SHA."""
    digest = hashlib.sha256()
    header = "\n".join(
        [
            f"repo={base['repo']}",
            f"run_id={base['run_id']}",
            f"commit={base['commit']}",
            f"artifact_group={base['artifact_group']}",
            f"therock={therock_pin}",
            f"image={image}",
            f"archs={archs}",
        ]
    )
    digest.update(header.encode())
    for rel in CACHE_SOURCE_PATHS:
        digest.update(b"\n")
        digest.update(rel.encode())
        digest.update(b"\n")
        digest.update(_tree_digest(src / rel).encode())
    return digest.hexdigest()


def cmd_cache_key(args: argparse.Namespace) -> int:
    base = json.loads(args.base_json.read_text())
    digest = cache_key_digest(args.src, base, args.therock_pin, args.image, args.archs)
    line = f"digest={digest}\n"
    output = os.environ.get("GITHUB_OUTPUT", "")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(line)
    sys.stdout.write(line)
    return 0


def restore_executable_bits(build: Path) -> None:
    """Artifact downloads drop the executable bit."""
    for rel in EXECUTABLES:
        path = build / rel
        path.chmod(path.stat().st_mode | 0o111)


def cmd_install_base(args: argparse.Namespace, runner: Runner = run) -> int:
    base = json.loads(args.base_json.read_text())
    print(f"Base: {base['repo']} run {base['run_id']} at {base['commit']}")
    runner(fetch_command(args.therock, base, args.dist, RUNTIME_ARTIFACTS))
    remove_amdhip64(args.dist / "lib")
    copy_amdhip64(args.capture_lib, args.dist / "lib")
    restore_executable_bits(args.build)
    return check_linkage(
        args.build / "hrr" / "tests" / "integration" / "hrr-integration-tests",
        args.dist / "lib",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=("fetch-base", "build", "stage", "verify", "install-base", "cache-key"),
    )
    parser.add_argument("--therock", type=Path, default=Path("therock"))
    parser.add_argument("--src", type=Path, default=Path.cwd())
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--capture-lib", type=Path, default=Path("build/capture-lib"))
    parser.add_argument("--base-json", type=Path, default=Path("build/base.json"))
    parser.add_argument(
        "--archs",
        default="",
        help=f"Semicolon-separated GPU targets; default targets.arch of {INTEGRATION_CONFIG}",
    )
    parser.add_argument(
        "--therock-pin", default="", help="TheRock commit the build tools came from"
    )
    parser.add_argument("--image", default="", help="Build container image digest")
    args = parser.parse_args(argv)
    for name in ("therock", "src", "dist", "build", "capture_lib", "base_json"):
        setattr(args, name, getattr(args, name).resolve())
    if args.command in ("build", "verify", "cache-key") and not args.archs:
        config = hrr_test_config.load(args.src / INTEGRATION_CONFIG)
        args.archs = ";".join(config["targets"]["arch"])
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return {
        "fetch-base": cmd_fetch_base,
        "build": cmd_build,
        "stage": cmd_stage,
        "verify": cmd_verify,
        "install-base": cmd_install_base,
        "cache-key": cmd_cache_key,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
