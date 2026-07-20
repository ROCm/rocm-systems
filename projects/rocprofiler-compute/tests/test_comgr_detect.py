# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for double comgr detection and single-comgr forcing.

Covers the detection helpers, the forcing decision for the two workload shapes
that trigger the condition (a torch wheel with a bundled comgr and a native app
with a private ``RPATH`` comgr), and GPU-backed runs of both workloads.
"""

import importlib.util
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch

import common
import pytest
from conftest import require_torch

from utils.comgr_detect import (
    _python_import_names,
    _workload_script_path,
    detect_and_log_double_comgr,
    find_workload_comgr_by_imports,
    output_indicates_double_comgr,
)

# LLVM messages emitted when two comgr libraries collide.
_REAL_ABORT_OUTPUT = (
    "   INFO    |-> [rocprofiler-sdk] : CommandLine Error: Option "
    "'spirv-expand-step' registered more than once!\n"
    "   INFO    |-> [rocprofiler-sdk] LLVM ERROR: inconsistency in registered "
    "CommandLine options\n"
)


def _make_comgr_file(directory: Path, name: str = "libamd_comgr.so.3") -> Path:
    """Create a placeholder comgr file under ``directory`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"\x7fELF")
    return path


class TestDoubleComgrSignatures:
    def test_real_abort_output_is_detected(self) -> None:
        assert output_indicates_double_comgr(_REAL_ABORT_OUTPUT) is True

    def test_benign_output_is_not_flagged(self) -> None:
        assert not output_indicates_double_comgr("Profiling completed successfully")


class TestImportParsing:
    def test_python_import_names_collects_top_level_packages(
        self, tmp_path: Path
    ) -> None:
        script = tmp_path / "workload.py"
        script.write_text(
            textwrap.dedent(
                """
                import torch
                import os.path
                from numpy import array
                from . import sibling      # relative -> skipped
                import torch                # duplicate -> collapsed
                """
            )
        )
        assert _python_import_names(script) == ["torch", "os", "numpy"]

    def test_workload_script_path_for_python_file(self, tmp_path: Path) -> None:
        script = tmp_path / "simple_net.py"
        script.write_text("import torch\n")
        assert _workload_script_path(["python3", str(script)]) == script

    def test_workload_script_path_none_for_module_or_inline(self) -> None:
        assert _workload_script_path(["python3", "-m", "pkg.mod"]) is None
        assert _workload_script_path(["python3", "-c", "print(1)"]) is None


class TestForcingDecision:
    """detect_and_log_double_comgr returns the workload comgr to preload."""

    def _patch_sources(self, *, tool, dynamic, imports, static):
        """Patch the resolver functions used by detect_and_log_double_comgr."""
        return (
            patch("utils.comgr_detect.resolve_tool_comgr", return_value=tool),
            patch(
                "utils.comgr_detect.find_workload_comgr_dynamic",
                return_value=dynamic,
            ),
            patch(
                "utils.comgr_detect.find_workload_comgr_by_imports",
                return_value=imports,
            ),
            patch(
                "utils.comgr_detect.find_workload_comgr_static",
                return_value=static,
            ),
        )

    def test_torch_workload_forces_bundled_comgr(self, tmp_path: Path) -> None:
        tool = _make_comgr_file(tmp_path / "opt/rocm/lib")
        bundled = _make_comgr_file(tmp_path / "venv/site-packages/_rocm_sdk_core/lib")
        patches = self._patch_sources(
            tool=tool, dynamic=[], imports=[bundled], static=[]
        )
        with patches[0], patches[1], patches[2], patches[3], patch(
            "utils.comgr_detect.console_warning"
        ) as warn:
            forced = detect_and_log_double_comgr(
                ["python3", "simple_net.py"], "/tool.so", {}
            )
        assert forced == bundled
        warn.assert_called_once()
        assert "_rocm_sdk_core" in warn.call_args.args[1]

    def test_hiprtc_workload_forces_private_rpath_comgr(self, tmp_path: Path) -> None:
        tool = _make_comgr_file(tmp_path / "opt/rocm/lib")
        privlib = _make_comgr_file(tmp_path / "sample/privlib")
        patches = self._patch_sources(
            tool=tool, dynamic=[], imports=[], static=[privlib]
        )
        with patches[0], patches[1], patches[2], patches[3], patch(
            "utils.comgr_detect.console_warning"
        ) as warn:
            forced = detect_and_log_double_comgr(["./rtc"], "/tool.so", {})
        assert forced == privlib
        warn.assert_called_once()

    def test_matching_comgr_does_not_force(self, tmp_path: Path) -> None:
        tool = _make_comgr_file(tmp_path / "opt/rocm/lib")
        # Workload resolves through a symlink to the same file as the tool.
        link = tmp_path / "alias/libamd_comgr.so.3"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(tool)
        patches = self._patch_sources(tool=tool, dynamic=[link], imports=[], static=[])
        with patches[0], patches[1], patches[2], patches[3], patch(
            "utils.comgr_detect.console_warning"
        ) as warn:
            forced = detect_and_log_double_comgr(["python3", "x.py"], "/tool.so", {})
        assert forced is None
        warn.assert_not_called()

    def test_no_workload_comgr_does_not_force(self, tmp_path: Path) -> None:
        tool = _make_comgr_file(tmp_path / "opt/rocm/lib")
        patches = self._patch_sources(tool=tool, dynamic=[], imports=[], static=[])
        with patches[0], patches[1], patches[2], patches[3]:
            forced = detect_and_log_double_comgr(["python3", "x.py"], "/tool.so", {})
        assert forced is None


class TestImportProbe:
    """A comgr loaded when a workload module is imported must be reported."""

    @staticmethod
    def _find_loadable_so() -> Path:
        """Return a loadable shared object mapped into the current process."""
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.split()
                if len(fields) < 6:
                    continue
                candidate = Path(fields[-1])
                if ".so" in candidate.name and candidate.is_file():
                    return candidate
        pytest.skip("no loadable .so found in /proc/self/maps")

    def test_probe_reports_comgr_loaded_by_import(self, tmp_path: Path) -> None:
        real_so = self._find_loadable_so()
        pkg_root = tmp_path / "pkgroot"
        pkg = pkg_root / "fakerocm"
        pkg.mkdir(parents=True)
        fake_comgr = pkg / "libamd_comgr.so.3"
        shutil.copy(real_so, fake_comgr)
        (pkg / "__init__.py").write_text(
            "import ctypes, os\n"
            "ctypes.CDLL(os.path.join(os.path.dirname(__file__), "
            "'libamd_comgr.so.3'))\n"
        )
        script = tmp_path / "workload.py"
        script.write_text("import fakerocm\n")

        env = dict(os.environ)
        env["PYTHONPATH"] = str(pkg_root)
        found = find_workload_comgr_by_imports(["python3", str(script)], env)

        assert fake_comgr.resolve() in {p.resolve() for p in found}


def _bundled_comgr_dir() -> Path | None:
    """Return the directory of torch's bundled comgr, if it is self-contained."""
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return None
    site_packages = Path(spec.origin).parent.parent
    for hit in site_packages.glob("**/libamd_comgr.so*"):
        if hit.is_file():
            return hit.parent
    return None


class TestDoubleComgrIntegration:
    """GPU-backed profiling runs of the two workloads must not abort."""

    def _assert_no_double_comgr_abort(self, stdout: str, stderr: str) -> None:
        combined = (stdout or "") + (stderr or "")
        assert "registered more than once" not in combined, (
            "Double comgr abort was not prevented"
        )
        assert "inconsistency in registered CommandLine options" not in combined
        if "Double comgr detected" in combined:
            assert "Forcing single comgr via LD_PRELOAD" in combined

    def test_torch_workload_runs_without_double_comgr(
        self, binary_handler_profile_rocprof_compute, tmp_path: Path
    ) -> None:
        require_torch(gpu=True)
        simple_net = Path(common.ROOT) / "tests" / "simple_net.py"
        if not simple_net.exists():
            pytest.skip("tests/simple_net.py not found")
        config = {"comgr_app": ["python3", str(simple_net)]}
        returncode, stdout, stderr = binary_handler_profile_rocprof_compute(
            config,
            str(tmp_path / "torch_workload"),
            options=[],
            check_success=False,
            app_name="comgr_app",
            capture_output=True,
        )
        self._assert_no_double_comgr_abort(stdout, stderr)
        assert returncode == 0, "torch workload profiling failed"

    def test_hiprtc_private_comgr_runs_without_double_comgr(
        self, binary_handler_profile_rocprof_compute, tmp_path: Path
    ) -> None:
        require_torch(gpu=True)
        hipcc = shutil.which("hipcc")
        if hipcc is None:
            pytest.skip("hipcc not available")
        comgr_dir = _bundled_comgr_dir()
        if comgr_dir is None:
            pytest.skip("no self-contained (bundled) comgr found for a private RPATH")

        src = tmp_path / "rtc.cpp"
        src.write_text(
            textwrap.dedent(
                """
                #include <hip/hiprtc.h>
                #include <hip/hip_runtime.h>
                #include <cstdio>

                __global__ void vadd(int* p) { p[threadIdx.x] = threadIdx.x; }

                int main() {
                    const char* code = "extern \\"C\\" __global__ void k(){}";
                    hiprtcProgram prog;
                    hiprtcCreateProgram(&prog, code, "k.cu", 0, nullptr, nullptr);
                    printf("hiprtc rc=%d\\n",
                           (int)hiprtcCompileProgram(prog, 0, nullptr));
                    hiprtcDestroyProgram(&prog);

                    int* d = nullptr;
                    hipMalloc(&d, 64 * sizeof(int));
                    hipLaunchKernelGGL(vadd, dim3(1), dim3(64), 0, 0, d);
                    hipDeviceSynchronize();
                    hipFree(d);
                    printf("done\\n");
                    return 0;
                }
                """
            )
        )
        binary = tmp_path / "rtc"
        build = subprocess.run(
            [
                hipcc,
                str(src),
                "-o",
                str(binary),
                "-lhiprtc",
                "-Wl,--disable-new-dtags",
                f"-Wl,-rpath,{comgr_dir}",
            ],
            text=True,
            capture_output=True,
        )
        if build.returncode != 0:
            pytest.skip(f"hipcc build failed: {build.stderr.strip()[:200]}")

        config = {"comgr_app": [str(binary)]}
        returncode, stdout, stderr = binary_handler_profile_rocprof_compute(
            config,
            str(tmp_path / "hiprtc_workload"),
            options=[],
            check_success=False,
            app_name="comgr_app",
            capture_output=True,
        )
        self._assert_no_double_comgr_abort(stdout, stderr)
        assert returncode == 0, "hipRTC private-comgr workload profiling failed"
