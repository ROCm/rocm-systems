# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the ROCm stack pre-flight.

Covers the double-comgr output signature, the comgr forcing decision, early
failure on a conflicting rocprofiler library, the rocprofv3 redirect and
fail-fast decisions, workload-stack discovery, and GPU-backed end-to-end runs.
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

from utils.rocm_stack_preflight import (
    COMGR_LIB_STEM,
    ROCPROFILER_REGISTER_LIB_STEM,
    ROCPROFILER_SDK_LIB_STEM,
    StackResolution,
    _distinct_workload_lib,
    _run_import_probe,
    comgr_to_force,
    output_indicates_double_comgr,
    plan_rocprofv3,
    resolve_rocm_stacks,
)
from utils.utils_exceptions import IncompatibleRocmStackError

# LLVM messages emitted when two comgr libraries collide.
_REAL_ABORT_OUTPUT = (
    "   INFO    |-> [rocprofiler-sdk] : CommandLine Error: Option "
    "'spirv-expand-step' registered more than once!\n"
    "   INFO    |-> [rocprofiler-sdk] LLVM ERROR: inconsistency in registered "
    "CommandLine options\n"
)


def _make_lib(directory: Path, name: str) -> Path:
    """Create a placeholder ELF file under ``directory`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"\x7fELF")
    return path


def _resolution(
    *,
    tool_comgr=None,
    workload_comgrs=(),
    tool_register=None,
    workload_register=(),
    tool_sdk=None,
    workload_sdk=(),
) -> StackResolution:
    """Build a StackResolution from explicit tool and workload libraries."""
    return StackResolution(
        tool_stack={
            COMGR_LIB_STEM: tool_comgr,
            ROCPROFILER_REGISTER_LIB_STEM: tool_register,
            ROCPROFILER_SDK_LIB_STEM: tool_sdk,
        },
        workload_stack={
            COMGR_LIB_STEM: list(workload_comgrs),
            ROCPROFILER_REGISTER_LIB_STEM: list(workload_register),
            ROCPROFILER_SDK_LIB_STEM: list(workload_sdk),
        },
    )


class TestDoubleComgrSignatures:
    def test_real_abort_output_is_detected(self) -> None:
        assert output_indicates_double_comgr(_REAL_ABORT_OUTPUT) is True

    def test_benign_output_is_not_flagged(self) -> None:
        assert not output_indicates_double_comgr("Profiling completed successfully")


class TestComgrToForce:
    """comgr_to_force selects the workload comgr to preload, or raises."""

    def test_forces_bundled_comgr(self, tmp_path: Path) -> None:
        tool = _make_lib(tmp_path / "opt/rocm/lib", "libamd_comgr.so.3")
        bundled = _make_lib(
            tmp_path / "venv/site-packages/_rocm_sdk_core/lib", "libamd_comgr.so.3"
        )
        with patch("utils.rocm_stack_preflight.console_warning") as warn:
            forced = comgr_to_force(
                _resolution(tool_comgr=tool, workload_comgrs=[bundled])
            )
        assert forced == bundled
        messages = " ".join(str(call.args[1]) for call in warn.call_args_list)
        assert "_rocm_sdk_core" in messages
        assert "Forcing a single 'libamd_comgr' via LD_PRELOAD" in messages

    def test_matching_comgr_does_not_force(self, tmp_path: Path) -> None:
        tool = _make_lib(tmp_path / "opt/rocm/lib", "libamd_comgr.so.3")
        link = tmp_path / "alias/libamd_comgr.so.3"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(tool)
        with patch("utils.rocm_stack_preflight.console_warning") as warn:
            forced = comgr_to_force(
                _resolution(tool_comgr=tool, workload_comgrs=[link])
            )
        assert forced is None
        warn.assert_not_called()

    def test_major_mismatch_does_not_force(self, tmp_path: Path) -> None:
        # A workload comgr with a different soname major is reported, not forced.
        tool = _make_lib(tmp_path / "opt/rocm/lib", "libamd_comgr.so.3")
        bundled = _make_lib(tmp_path / "wheel/lib", "libamd_comgr.so.2")
        with patch("utils.rocm_stack_preflight.console_warning") as warn:
            forced = comgr_to_force(
                _resolution(tool_comgr=tool, workload_comgrs=[bundled])
            )
        assert forced is None
        warnings = " ".join(str(call.args[1]) for call in warn.call_args_list)
        assert "major" in warnings

    def test_rocprofiler_register_major_mismatch_raises(self, tmp_path: Path) -> None:
        tool = _make_lib(tmp_path / "opt/rocm/lib", "librocprofiler-register.so.0.6.0")
        workload = _make_lib(tmp_path / "wheel/lib", "librocprofiler-register.so.1.0.0")
        with pytest.raises(IncompatibleRocmStackError):
            comgr_to_force(
                _resolution(tool_register=tool, workload_register=[workload])
            )

    def test_rocprofiler_same_major_does_not_raise(self, tmp_path: Path) -> None:
        # Same soname major, different filename version: not a conflict.
        tool = _make_lib(tmp_path / "opt/rocm/lib", "librocprofiler-register.so.0.6.0")
        workload = _make_lib(tmp_path / "wheel/lib", "librocprofiler-register.so.0")
        assert (
            comgr_to_force(
                _resolution(tool_register=tool, workload_register=[workload])
            )
            is None
        )


class TestPlanRocprofv3:
    """plan_rocprofv3 redirects, forces comgr, or raises."""

    def test_redirects_to_workload_rocprofv3(self, tmp_path: Path) -> None:
        tool_sdk = _make_lib(tmp_path / "opt/rocm/lib", "librocprofiler-sdk.so.1")
        root = tmp_path / "wheel/_rocm_sdk_core"
        workload_sdk = _make_lib(root / "lib", "librocprofiler-sdk.so.1")
        binary = _make_lib(root / "bin", "rocprofv3")
        binary.chmod(0o755)
        launch = plan_rocprofv3(
            _resolution(tool_sdk=tool_sdk, workload_sdk=[workload_sdk])
        )
        assert launch.rocprofv3 == str(binary)
        assert launch.forced_comgr is None

    def test_fails_fast_without_workload_rocprofv3(self, tmp_path: Path) -> None:
        tool_sdk = _make_lib(tmp_path / "opt/rocm/lib", "librocprofiler-sdk.so.1")
        workload_sdk = _make_lib(tmp_path / "wheel/lib", "librocprofiler-sdk.so.1")
        with pytest.raises(IncompatibleRocmStackError):
            plan_rocprofv3(_resolution(tool_sdk=tool_sdk, workload_sdk=[workload_sdk]))

    def test_major_mismatch_fails_fast(self, tmp_path: Path) -> None:
        tool_sdk = _make_lib(tmp_path / "opt/rocm/lib", "librocprofiler-sdk.so.1")
        root = tmp_path / "wheel/_rocm_sdk_core"
        workload_sdk = _make_lib(root / "lib", "librocprofiler-sdk.so.2")
        binary = _make_lib(root / "bin", "rocprofv3")
        binary.chmod(0o755)
        with pytest.raises(IncompatibleRocmStackError):
            plan_rocprofv3(_resolution(tool_sdk=tool_sdk, workload_sdk=[workload_sdk]))

    def test_forces_comgr_without_sdk_conflict(self, tmp_path: Path) -> None:
        tool_comgr = _make_lib(tmp_path / "opt/rocm/lib", "libamd_comgr.so.3")
        workload_comgr = _make_lib(tmp_path / "wheel/lib", "libamd_comgr.so.3")
        with patch("utils.rocm_stack_preflight.console_warning"):
            launch = plan_rocprofv3(
                _resolution(tool_comgr=tool_comgr, workload_comgrs=[workload_comgr])
            )
        assert launch.rocprofv3 is None
        assert launch.forced_comgr == str(workload_comgr)


class TestDistinctWorkloadLib:
    def test_returns_none_when_same_real_path(self, tmp_path: Path) -> None:
        tool = _make_lib(tmp_path / "opt/rocm/lib", "librocprofiler-sdk.so.1")
        link = tmp_path / "alias/librocprofiler-sdk.so.1"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(tool)
        assert _distinct_workload_lib(tool, [link]) is None

    def test_returns_distinct_lib(self, tmp_path: Path) -> None:
        tool = _make_lib(tmp_path / "opt/rocm/lib", "librocprofiler-sdk.so.1")
        workload = _make_lib(tmp_path / "wheel/lib", "librocprofiler-sdk.so.1")
        assert _distinct_workload_lib(tool, [workload]) == workload


class TestResolveRocmStacks:
    """resolve_rocm_stacks returns the resolved stacks, or None on failure."""

    def test_returns_resolution(self, tmp_path: Path) -> None:
        tool = _make_lib(tmp_path / "opt/rocm/lib", "libamd_comgr.so.3")
        bundled = _make_lib(tmp_path / "wheel/lib", "libamd_comgr.so.3")
        tool_stack = {
            COMGR_LIB_STEM: tool,
            ROCPROFILER_REGISTER_LIB_STEM: None,
            ROCPROFILER_SDK_LIB_STEM: None,
        }
        workload_stack = {
            COMGR_LIB_STEM: [bundled],
            ROCPROFILER_REGISTER_LIB_STEM: [],
            ROCPROFILER_SDK_LIB_STEM: [],
        }
        with patch(
            "utils.rocm_stack_preflight._resolve_tool_stack", return_value=tool_stack
        ), patch(
            "utils.rocm_stack_preflight._discover_workload_stack",
            return_value=workload_stack,
        ):
            resolution = resolve_rocm_stacks(["python3", "x.py"], "/tool.so", {})
        assert resolution is not None
        assert resolution.tool(COMGR_LIB_STEM) == tool
        assert resolution.workload(COMGR_LIB_STEM) == [bundled]

    def test_returns_none_on_discovery_failure(self) -> None:
        with patch(
            "utils.rocm_stack_preflight._resolve_tool_stack",
            side_effect=RuntimeError("boom"),
        ):
            assert resolve_rocm_stacks(["python3", "x.py"], "/tool.so", {}) is None


class TestWorkloadStackDiscovery:
    """The workload stack must exclude the profiler's own ROCm root."""

    def test_excludes_profiler_rocm_root(self, tmp_path: Path, monkeypatch) -> None:
        from utils import rocm_stack_preflight as preflight

        rocm_root = tmp_path / "opt/rocm"
        under_root = _make_lib(rocm_root / "lib", "libamd_comgr.so.3")
        bundled = _make_lib(tmp_path / "wheel/lib", "libamd_comgr.so.3")
        monkeypatch.setenv("ROCM_PATH", str(rocm_root))
        monkeypatch.setattr(
            preflight,
            "_find_workload_libs_dynamic",
            lambda env, stems: {s: [under_root] for s in stems},
        )
        monkeypatch.setattr(preflight, "_run_import_probe", lambda a, e, s: [bundled])
        monkeypatch.setattr(
            preflight,
            "_find_workload_libs_static",
            lambda a, s: {stem: [] for stem in s},
        )
        result = preflight._discover_workload_stack(
            ["python3", "x.py"], {}, (COMGR_LIB_STEM,)
        )
        resolved = {p.resolve() for p in result[COMGR_LIB_STEM]}
        assert under_root.resolve() not in resolved
        assert bundled.resolve() in resolved

    def test_excludes_library_identical_to_tool(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A workload library byte-identical to the tool's copy is excluded."""
        from utils import rocm_stack_preflight as preflight

        tool_lib = _make_lib(tmp_path / "devel/lib", "libamd_comgr.so.3")
        tool_lib.write_bytes(b"identical-comgr-build")
        # Same build shipped in a sibling directory (distinct path, same bytes).
        workload_lib = _make_lib(tmp_path / "core/lib", "libamd_comgr.so.3")
        workload_lib.write_bytes(b"identical-comgr-build")
        # A genuinely different build must still be reported.
        other_lib = _make_lib(tmp_path / "bundled/lib", "libamd_comgr.so.3")
        other_lib.write_bytes(b"a-different-comgr-build")

        monkeypatch.setenv("ROCM_PATH", str(tmp_path / "devel"))
        monkeypatch.setattr(
            preflight,
            "_find_workload_libs_dynamic",
            lambda env, stems: {s: [workload_lib, other_lib] for s in stems},
        )
        monkeypatch.setattr(preflight, "_run_import_probe", lambda a, e, s: [])
        monkeypatch.setattr(
            preflight,
            "_find_workload_libs_static",
            lambda a, s: {stem: [] for stem in s},
        )
        result = preflight._discover_workload_stack(
            ["python3", "x.py"],
            {},
            (COMGR_LIB_STEM,),
            {COMGR_LIB_STEM: tool_lib},
        )
        resolved = {p.resolve() for p in result[COMGR_LIB_STEM]}
        assert workload_lib.resolve() not in resolved
        assert other_lib.resolve() in resolved


class TestImportProbe:
    """A comgr loaded when a workload module is imported must be reported."""

    @staticmethod
    def _find_loadable_so() -> Path:
        """Return a loadable shared object mapped into the current process."""
        with open("/proc/self/maps", encoding="utf-8") as maps:
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
        found = _run_import_probe(["python3", str(script)], env, (COMGR_LIB_STEM,))

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
            assert "Forcing a single 'libamd_comgr' via LD_PRELOAD" in combined

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
