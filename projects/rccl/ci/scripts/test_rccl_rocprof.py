#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Smoke tests validating that rocprofv3 can trace RCCL collectives.
# Runs in RCCL CI using TheRock build artifacts (THEROCK_BIN_DIR) or
# locally against a system ROCm install.

import argparse
import csv
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def resolve_paths():
    therock_bin = os.getenv("THEROCK_BIN_DIR")
    if therock_bin:
        therock_bin = Path(therock_bin).resolve()
        therock_path = therock_bin.parent
        return {
            "rocprofv3": str(therock_bin / "rocprofv3"),
            "rccl_unittests": str(therock_bin / "rccl-UnitTests"),
            "lib_path": str(therock_path / "lib"),
            "extra_lib_paths": [],
            "sysdeps_lib_path": str(therock_path / "lib" / "rocm_sysdeps" / "lib"),
            "rocm_path": str(therock_path),
        }
    rocm_path = os.getenv("ROCM_PATH", "/opt/rocm")
    rccl_build_dir = os.getenv("RCCL_BUILD_DIR", "")

    rccl_ut = shutil.which("rccl-UnitTests") or f"{rocm_path}/bin/rccl-UnitTests"

    extra_lib_paths = []
    if rccl_build_dir:
        extra_lib_paths.append(rccl_build_dir)
    elif Path(rccl_ut).is_file():
        candidate = str(Path(rccl_ut).resolve().parent.parent)
        librccl = Path(candidate) / "librccl.so"
        if librccl.exists():
            extra_lib_paths.append(candidate)

    return {
        "rocprofv3": shutil.which("rocprofv3") or f"{rocm_path}/bin/rocprofv3",
        "rccl_unittests": rccl_ut,
        "lib_path": f"{rocm_path}/lib",
        "extra_lib_paths": extra_lib_paths,
        "sysdeps_lib_path": "",
        "rocm_path": rocm_path,
    }


PATHS = resolve_paths()


def _setup_kpack_env(env, rocm_path):
    """Set ROCM_KPACK_PATH if .kpack archives exist in the artifact tree.

    ROCM_KPACK_PATH expects literal file paths, not @GFXARCH@ patterns.
    """
    artifact_dir = Path(rocm_path)
    kpack_files = sorted(artifact_dir.rglob("*.kpack"))
    if not kpack_files:
        return
    kpack_paths = []
    for f in kpack_files:
        resolved = str(f.resolve())
        kpack_paths.append(resolved)
        log.info("kpack archive: %s", resolved)
    if kpack_paths:
        env["ROCM_KPACK_PATH"] = ":".join(kpack_paths)
        log.info("ROCM_KPACK_PATH=%s", env["ROCM_KPACK_PATH"])


def make_env():
    env = os.environ.copy()
    env["ROCM_PATH"] = PATHS["rocm_path"]
    env["HIP_PATH"] = PATHS["rocm_path"]

    ld_paths = []
    for p in PATHS.get("extra_lib_paths", []):
        ld_paths.append(p)
    ld_paths.append(PATHS["lib_path"])
    if PATHS["sysdeps_lib_path"]:
        ld_paths.append(PATHS["sysdeps_lib_path"])
    old_ld = os.getenv("LD_LIBRARY_PATH", "")
    if old_ld:
        ld_paths.append(old_ld)
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths)

    rocprofv3_dir = str(Path(PATHS["rocprofv3"]).parent)
    old_path = os.getenv("PATH", "")
    env["PATH"] = f"{rocprofv3_dir}:{old_path}" if old_path else rocprofv3_dir

    if not env.get("HIP_VISIBLE_DEVICES"):
        env["HIP_VISIBLE_DEVICES"] = "0,1"

    env.pop("GPU_DEVICE_ORDINAL", None)

    _setup_kpack_env(env, PATHS["rocm_path"])

    return env


def run_cmd(cmd, env=None, timeout=300):
    log.info(f"++ Exec: {shlex.join(cmd)}")
    result = subprocess.run(
        cmd,
        env=env or make_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.stdout:
        log.info(f"stdout ({len(result.stdout)} chars):\n{result.stdout[:2000]}")
    if result.stderr:
        log.info(f"stderr ({len(result.stderr)} chars):\n{result.stderr[:2000]}")
    return result


def skip_if_missing(binary_path, name):
    if not Path(binary_path).is_file():
        pytest.skip(f"{name} not found at {binary_path}")


class TestRCCLRocprof:

    def test_rocprofv3_available(self):
        skip_if_missing(PATHS["rocprofv3"], "rocprofv3")
        result = run_cmd([PATHS["rocprofv3"], "--version"])
        assert result.returncode == 0, f"rocprofv3 --version failed: {result.stderr}"
        assert "version" in result.stdout.lower() or "version" in result.stderr.lower()

    def test_rocprofv3_help_shows_rccl_trace(self):
        skip_if_missing(PATHS["rocprofv3"], "rocprofv3")
        result = run_cmd([PATHS["rocprofv3"], "--help"])
        assert result.returncode == 0, f"rocprofv3 --help failed: {result.stderr}"
        combined = result.stdout + result.stderr
        assert "--rccl-trace" in combined, (
            "rocprofv3 --help does not mention --rccl-trace"
        )

    def test_rocprofv3_hip_trace_rccl_unittests(self):
        skip_if_missing(PATHS["rocprofv3"], "rocprofv3")
        skip_if_missing(PATHS["rccl_unittests"], "rccl-UnitTests")

        with tempfile.TemporaryDirectory(prefix="rccl_rocprof_hip_") as tmpdir:
            outprefix = os.path.join(tmpdir, "hip_trace")
            cmd = [
                PATHS["rocprofv3"],
                "--hip-trace",
                "--output-format", "csv",
                "-o", outprefix,
                "--",
                PATHS["rccl_unittests"],
                "--gtest_filter=AllReduce.OutOfPlace",
            ]
            env = make_env()
            env["UT_MIN_GPUS"] = "2"
            env["UT_MAX_GPUS"] = "2"
            result = run_cmd(cmd, env=env, timeout=300)

            assert result.returncode == 0, (
                f"rocprofv3 --hip-trace exited with {result.returncode}\n"
                f"stderr: {result.stderr[-1000:]}"
            )

            csv_files = list(Path(tmpdir).rglob("*.csv"))
            assert len(csv_files) > 0, (
                f"No CSV output files found in {tmpdir}. "
                f"Directory contents: {list(Path(tmpdir).rglob('*'))}"
            )

            total_rows = 0
            for csv_file in csv_files:
                with open(csv_file) as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    data_rows = max(0, len(rows) - 1)
                    total_rows += data_rows
                    log.info(f"  {csv_file.name}: {data_rows} data rows")

            assert total_rows > 0, "CSV output files are empty (no trace data)"

    def test_rocprofv3_rccl_trace(self):
        skip_if_missing(PATHS["rocprofv3"], "rocprofv3")
        skip_if_missing(PATHS["rccl_unittests"], "rccl-UnitTests")

        with tempfile.TemporaryDirectory(prefix="rccl_rocprof_rccl_") as tmpdir:
            outprefix = os.path.join(tmpdir, "rccl_trace")
            cmd = [
                PATHS["rocprofv3"],
                "--rccl-trace",
                "--output-format", "csv",
                "-o", outprefix,
                "--",
                PATHS["rccl_unittests"],
                "--gtest_filter=AllReduce.OutOfPlace",
            ]
            env = make_env()
            env["UT_MIN_GPUS"] = "2"
            env["UT_MAX_GPUS"] = "2"
            result = run_cmd(cmd, env=env, timeout=300)

            assert result.returncode == 0, (
                f"rocprofv3 --rccl-trace exited with {result.returncode}\n"
                f"stderr: {result.stderr[-1000:]}"
            )

            csv_files = list(Path(tmpdir).rglob("*.csv"))
            assert len(csv_files) > 0, (
                f"No CSV output files found in {tmpdir}. "
                f"Directory contents: {list(Path(tmpdir).rglob('*'))}"
            )

            all_content = ""
            for csv_file in csv_files:
                content = csv_file.read_text()
                all_content += content
                log.info(f"  {csv_file.name} ({len(content)} bytes): "
                         f"{content[:500]}")

            rccl_indicators = ["nccl", "rccl", "AllReduce", "ncclAllReduce"]
            found_any = any(
                ind.lower() in all_content.lower() for ind in rccl_indicators
            )
            assert found_any, (
                "No RCCL-specific entries found in trace output. "
                "rocprofiler-register may not be exposing RCCL callbacks. "
                f"Searched for {rccl_indicators} in {len(csv_files)} CSV files "
                f"({len(all_content)} bytes total)."
            )

    @pytest.mark.skip(reason="sys-trace traces all domains and exceeds CI timeout; run locally or in nightly")
    def test_rocprofv3_sys_trace_rccl(self):
        skip_if_missing(PATHS["rocprofv3"], "rocprofv3")
        skip_if_missing(PATHS["rccl_unittests"], "rccl-UnitTests")

        with tempfile.TemporaryDirectory(prefix="rccl_rocprof_sys_") as tmpdir:
            outprefix = os.path.join(tmpdir, "sys_trace")
            cmd = [
                PATHS["rocprofv3"],
                "--sys-trace",
                "--output-format", "json",
                "-o", outprefix,
                "--",
                PATHS["rccl_unittests"],
                "--gtest_filter=AllReduce.OutOfPlace",
            ]
            env = make_env()
            env["UT_MIN_GPUS"] = "2"
            env["UT_MAX_GPUS"] = "2"
            result = run_cmd(cmd, env=env, timeout=600)

            assert result.returncode == 0, (
                f"rocprofv3 --sys-trace exited with {result.returncode}\n"
                f"stderr: {result.stderr[-1000:]}"
            )

            json_files = list(Path(tmpdir).rglob("*.json"))
            assert len(json_files) > 0, (
                f"No JSON output files found in {tmpdir}. "
                f"Directory contents: {list(Path(tmpdir).rglob('*'))}"
            )

            for jf in json_files:
                content = jf.read_text()
                assert len(content) > 10, f"JSON file {jf.name} is too small"
                try:
                    data = json.loads(content)
                    log.info(
                        f"  {jf.name}: valid JSON, "
                        f"top-level keys: "
                        f"{list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
                    )
                except json.JSONDecodeError:
                    lines = content.strip().split("\n")
                    for line in lines[:5]:
                        json.loads(line)
                    log.info(f"  {jf.name}: valid JSONL, {len(lines)} lines")


def main() -> None:
    parser = argparse.ArgumentParser(description="rocprofv3 RCCL smoke tests")
    parser.add_argument(
        "--notify-email",
        type=str,
        default="",
        help="Send summary report to this email address",
    )
    parser.add_argument(
        "--notify-webhook",
        type=str,
        default="",
        help="Send summary report to this Teams webhook URL",
    )
    args = parser.parse_args()

    exit_code = pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO", "--tb=short"])

    if args.notify_email or args.notify_webhook:
        from rccl_ci_utils import send_email_report, send_teams_webhook

        status = "PASSED" if exit_code == 0 else "FAILED"
        report = f"RCCL rocprofv3 smoke test: {status}"
        if args.notify_email:
            send_email_report(report, args.notify_email, status,
                              subject_prefix="RCCL rocprofv3 Smoke Test")
        if args.notify_webhook:
            send_teams_webhook(report, args.notify_webhook, status,
                               subject_prefix="RCCL rocprofv3 Smoke Test")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
