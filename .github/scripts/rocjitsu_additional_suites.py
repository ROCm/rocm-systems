#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Bounded systems-component selections on native gfx942 or rocJITsu.

Selection conventions follow TheRock at THEROCK_REF. This is a prototype:
RCCL covers one simulated or two native GPUs; rocSHMEM covers host units in
simulation or four native GPUs. These topology differences are recorded.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import rocjitsu_suite_runner as runner

THEROCK_REF = "e0e238c0323f76aeabdd348cfd3798a653820fdc"
SUITES = ("aqlprofile", "rocprofiler-sdk", "rocprofiler-compute", "amdsmi",
          "kfdtest", "hipfile", "rccl", "rocshmem", "rocprofiler-systems")


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return path


def pinned_script(therock: Path, name: str) -> Path:
    revision = subprocess.check_output(
        ["git", "-C", str(therock), "rev-parse", "HEAD"], text=True, timeout=15
    ).strip()
    if revision != THEROCK_REF:
        raise RuntimeError(f"Expected TheRock {THEROCK_REF}, got {revision}")
    return require_file(therock / "build_tools/github_actions/test_executable_scripts" / name)


def ctest_command(root: Path, relative: str, label: str) -> list[str]:
    directory = root / relative
    require_file(directory / "CTestTestfile.cmake")
    return ["ctest", "--test-dir", str(directory), "-L", f"^{label}$",
            "--no-tests=error", "--output-on-failure", "--parallel", "1",
            "--timeout", "300", "-V"]


def suite_command(suite: str, root: Path, therock: Path,
                  backend: str = "rocjitsu") -> tuple[list[str], str]:
    if suite == "aqlprofile":
        return ["bash", str(require_file(root / "share/hsa-amd-aqlprofile/run_tests.sh"))], "installed four-case AQL selection"
    if suite == "amdsmi":
        return [str(require_file(root / "share/amd_smi/tests/amdsmitst")), "--gtest_filter=*Unit*"], "AMD SMI unit selection"
    if suite == "kfdtest":
        return [str(require_file(root / "bin/kfdtest")),
                "--gtest_filter=KFDOpenCloseKFDTest.OpenCloseKFD:KFDOpenCloseKFDTest.OpenAlreadyOpenedKFD:KFDCloseKFDTest.CloseAClosedKfd"], "KFD open/close basic selection"
    if suite == "hipfile":
        return ctest_command(root, "share/hipfile/test", "unit"), "hipFile unit label"
    if suite == "rocprofiler-compute":
        return ctest_command(root, "libexec/rocprofiler-compute", "quick"), "compute quick label"
    if suite == "rccl":
        gpu_count = "2" if backend == "native-gfx942" else "1"
        return [str(require_file(root / "bin/all_reduce_perf")), "-b", "8", "-e", "1M",
                "-f", "2", "-g", gpu_count, "-n", "1", "-w", "1", "-c", "1"], f"{gpu_count}-GPU all-reduce correctness smoke; topology differs between backends"
    if suite == "rocshmem":
        if backend == "native-gfx942":
            directory = root / "bin/rocshmem"
            require_file(directory / "CTestTestfile.cmake")
            return ["ctest", "--test-dir", str(directory), "-R", "^unit_tests_n4$",
                    "--no-tests=error", "--output-on-failure", "--timeout", "900", "-V"], "four-GPU MPI unit_tests_n4; topology differs from simulator host units"
        return [str(require_file(root / "bin/rocshmem_unit_tests")),
                "--gtest_filter=EnvVar*"], "singleton host EnvVar unit tests; no GPU communication coverage"
    if suite == "rocprofiler-sdk":
        script = pinned_script(therock, "test_rocprofiler_sdk.py")
        require_file(root / "share/rocprofiler-sdk/tests/CMakeLists.txt")
        return [sys.executable, str(Path(__file__).resolve()), "--sdk-child", str(script)], "installed SDK configure/build/test with upstream ASAN exclusions"
    if suite == "rocprofiler-systems":
        script = pinned_script(therock, "test_runner.py")
        require_file(root / "share/rocprofiler-systems/tests/CTestTestfile.cmake")
        return [sys.executable, str(script)], "systems quick selection with upstream exclusions"
    raise ValueError(f"Unknown suite: {suite}")


def has_executed_tests(log: str, suite: str) -> bool:
    # Match test output, never the wrapper's own synthetic Passed summary.
    patterns = [r"\[\s*PASSED\s*\]\s+[1-9]\d*\s+tests?",
                r"^\d+:.*\bTest\s+#\d+:.*\bPassed\b",
                r"\b[1-9]\d* passed\b"]
    if suite == "aqlprofile":
        patterns.append(r"\b[1-9]\d* tests? run\b")
    if suite == "rccl":
        patterns = [r"#\s*Out of bounds values\s*:\s*0\s+OK"]
    return any(re.search(pattern, log, re.MULTILINE) for pattern in patterns)


def sdk_config_command(module, preload: str) -> list[str]:
    command = module.get_cmake_config_cmd()
    # Installed CTest launchers must retain the simulator interposer as well
    # as ASAN. The pinned upstream script otherwise replaces this with ASAN.
    return [
        "-DROCPROFILER_MEMCHECK_PRELOAD_ENV=LD_PRELOAD=" + preload
        if arg.startswith("-DROCPROFILER_MEMCHECK_PRELOAD_ENV=") else
        "-DROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE=" + preload
        if arg.startswith("-DROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE=") else arg
        for arg in command
    ]


def sdk_child(script: Path) -> int:
    # This process is already running under the selected backend. Every
    # configure/build/test subprocess inherits that same launcher environment.
    spec = importlib.util.spec_from_file_location("pinned_sdk_tests", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path(os.environ["ROCM_ROOT"])
    runtime = str(require_file(Path(os.environ["ASAN_RUNTIME_PATH"])))
    module.THEROCK_CLANG_PATH = require_file(root / "lib/llvm/bin/amdclang")
    module.THEROCK_CLANG_PLUS_PATH = require_file(root / "lib/llvm/bin/amdclang++")
    module.get_asan_runtime_library = lambda: runtime
    module.setup_env()
    preload = os.environ.get("LD_PRELOAD", "")
    if runtime not in preload:
        raise RuntimeError("SDK child did not inherit its ASAN runtime preload")
    print(f"SDK child LD_PRELOAD={preload}", flush=True)
    module._run_command(sdk_config_command(module, preload))
    module.cmake_build()
    command = module.get_ctest_cmd()
    command += ["--no-tests=error", "--timeout", "300"]
    module._run_command(command)
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--sdk-child":
        return sdk_child(Path(sys.argv[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--backend", choices=("native-gfx942", "rocjitsu"), required=True)
    parser.add_argument("--rocm-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--therock-root", type=Path, default=Path("TheRock"))
    parser.add_argument("--rocjitsu")
    parser.add_argument("--config")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.backend == "rocjitsu" and not (args.rocjitsu and args.config):
        parser.error("rocjitsu backend requires --rocjitsu and --config")
    root = args.rocm_root.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{args.suite}.log"
    status = "failed"
    scope = "selection could not start"
    try:
        command, scope = suite_command(args.suite, root, args.therock_root.resolve(), args.backend)
        env = os.environ.copy()
        env.update(ROCM_ROOT=str(root), ROCM_PATH=str(root), HIP_PATH=str(root),
                   THEROCK_BIN_DIR=str(root / "bin"), OUTPUT_ARTIFACTS_DIR=str(root),
                   BUILD_VARIANT="host-asan", TEST_TYPE="quick", TEST_COMPONENT=args.suite,
                   AMDGPU_FAMILIES="gfx942", SHARD_INDEX="1", TOTAL_SHARDS="1",
                   AMDSMI_NON_PRIVILEGED="1", CTEST_NO_TESTS_ACTION="error",
                   ROCPROFSYS_INSTALL_DIR=str(root), ROCPROFSYS_MAX_THREADS="64",
                   OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
        if (root / "bin/mpirun").is_file():
            env.update(OPAL_PREFIX=str(root), PRTE_PREFIX=str(root), PMIX_PREFIX=str(root))
        env["PATH"] = os.pathsep.join((str(Path(sys.executable).parent), str(root / "bin"), env.get("PATH", "")))
        libraries = [root / "lib", root / "lib/rocm_sysdeps/lib",
                     root / "lib/rocprofiler-systems",
                     root / "share/rocprofiler-systems/examples/lib"]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, libraries)) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        if args.backend == "rocjitsu":
            result = runner.run_under_rocjitsu(args.rocjitsu, args.config, command,
                args.suite, args.timeout_seconds, env, str(root))
        else:
            result = runner.run_command(command, args.suite, args.timeout_seconds,
                runner.native_test_env(env), str(root))
        log = "\n".join(result["log_lines"]) + "\n"
        if result["passed"] and has_executed_tests(log, args.suite):
            status = "passed"
        elif result["passed"]:
            log += "FAILED: no evidence of executed tests; empty or all-skipped selections cannot pass.\n"
    except NotImplementedError as error:
        status = "unsupported"
        log = f"UNSUPPORTED: {error}\n"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        log = f"FAILED: {error}\n"
    log_path.write_text(f"Backend: {args.backend}\nSelection: {scope}\nTheRock reference: {THEROCK_REF}\n" + log)
    (args.log_dir / f"{args.suite}.result").write_text(status + "\n")
    (args.log_dir / f"{args.suite}.json").write_text(json.dumps({
        "suite": args.suite, "backend": args.backend, "status": status, "selection": scope,
        "therock_ref": THEROCK_REF, "timeout_seconds": args.timeout_seconds,
    }, indent=2) + "\n")
    print(f"{args.suite}: {status}; {log_path}", flush=True)
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
