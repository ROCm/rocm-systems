#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN

"""GPU-free unit tests for rocprofv3-doctor's load test and error matching.

The error strings below are the real messages printed by glibc's loader,
ROCr, CLR, rocprofiler-sdk, and Python; if one of those changes wording, the
matching signature in doctor_diagnose.py has to follow it.
"""

import pytest

from conftest import FakeAccessor, make_healthy_accessor

SDK = "/fake/rocm/lib/librocprofiler-sdk.so"
HIP = "/fake/rocm/lib/libamdhip64.so"


@pytest.fixture
def diag(rocprofv3_package):
    from rocprofv3 import doctor_diagnose

    return doctor_diagnose


@pytest.fixture
def checks(rocprofv3_package):
    from rocprofv3 import (
        doctor_checks_counters,
        doctor_checks_runtime,
        doctor_checks_tools,
        doctor_result,
    )

    return {
        "counters": doctor_checks_counters,
        "runtime": doctor_checks_runtime,
        "tools": doctor_checks_tools,
        "status": doctor_result,
    }


def _ids(diagnoses):
    return [item["id"] for item in diagnoses]


# ----------------------------------------------------------------------
# signatures
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "libnuma.so.1: cannot open shared object file: No such file or directory",
            "loader.missing-dependency",
        ),
        (
            SDK + ": undefined symbol: _ZN11rocprofiler8internal4initEv",
            "loader.undefined-symbol",
        ),
        (
            "/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.30' not "
            "found (required by /fake/rocm/lib/libamdhip64.so)",
            "loader.symbol-version",
        ),
        (SDK + ": wrong ELF class: ELFCLASS32", "loader.wrong-elf-class"),
        (SDK + ": invalid ELF header", "loader.bad-file"),
        (
            "HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the "
            "necessary resources.",
            "hsa.out-of-resources",
        ),
        ("hipErrorNoDevice: no ROCm-capable device is detected", "hip.no-device"),
        ("open /dev/kfd failed: Permission denied", "kfd.permission-denied"),
        (
            "Required permission (CAP_PERFMON) is not set, permission denied",
            "sdk.device-lock-permission",
        ),
        (
            "Device 1 could not be locked for profiling due to lack of permissions "
            "(capability SYS_PERFMON). PMC Counters may be inaccurate",
            "sdk.device-lock-permission",
        ),
        (
            "Device 1 has a profiler attached to it. PMC Counters may be inaccurate.",
            "sdk.device-busy",
        ),
        (
            "Driver/Kernel version does not support locking device 1.",
            "sdk.incompatible-kernel",
        ),
        (
            "A service depends on a newer version of KFD (amdgpu kernel driver)",
            "sdk.incompatible-kernel",
        ),
        ("Context has a conflict with another context", "sdk.context-conflict"),
        ("ModuleNotFoundError: No module named 'yaml'", "python.missing-module"),
    ],
)
def test_doctor_diagnose_signature_matches(diag, text, expected):
    diagnoses = diag.diagnose(make_healthy_accessor(), text)
    assert expected in _ids(diagnoses)
    match = diagnoses[_ids(diagnoses).index(expected)]
    assert match["summary"] and match["remediation"] and match["observation"]
    assert match["confidence"] in ("high", "possible")
    assert match["evidence"] in text


def test_doctor_diagnose_unrecognised_text_is_empty(diag):
    assert diag.diagnose(make_healthy_accessor(), "something went wrong") == []


def test_doctor_diagnose_rocm_dependency_suggests_reinstall(diag):
    text = "libhsa-runtime64.so.1: cannot open shared object file: No such file"
    (match,) = diag.diagnose(make_healthy_accessor(), text)
    assert "libhsa-runtime64.so.1" in match["summary"]
    assert "apt-file" not in match["remediation"]
    assert "part of ROCm" in match["remediation"]


def test_doctor_diagnose_system_dependency_suggests_package_search(diag):
    text = "libnuma.so.1: cannot open shared object file: No such file"
    (match,) = diag.diagnose(make_healthy_accessor(), text)
    assert "apt-file search libnuma.so.1" in match["remediation"]


def test_doctor_diagnose_old_libstdcxx_under_conda(diag):
    text = "version `GLIBCXX_3.4.30' not found (required by x)"
    accessor = make_healthy_accessor(env={"CONDA_PREFIX": "/home/me/miniconda3"})
    (match,) = diag.diagnose(accessor, text)
    assert "libstdc++" in match["remediation"]
    assert "/home/me/miniconda3" in match["remediation"]


def test_doctor_diagnose_pip_name_differs_from_module(diag):
    (match,) = diag.diagnose(make_healthy_accessor(), "No module named 'yaml'")
    assert "pip install pyyaml" in match["remediation"]


def test_doctor_diagnose_missing_rocprofv3_module_suggests_pythonpath(diag):
    (match,) = diag.diagnose(make_healthy_accessor(), "No module named 'rocprofv3.avail'")
    assert "PYTHONPATH=/fake/rocm/lib/python3/site-packages" in match["remediation"]


def test_doctor_diagnose_unexplained_crash(diag):
    assert _ids(diag.diagnose(make_healthy_accessor(), "", -11)) == ["process.crashed"]


def test_doctor_diagnose_explained_crash_uses_the_explanation(diag):
    text = SDK + ": undefined symbol: foo"
    assert _ids(diag.diagnose(make_healthy_accessor(), text, -11)) == [
        "loader.undefined-symbol"
    ]


@pytest.mark.parametrize(
    "returncode,expected",
    [
        (-1001, "timed out"),
        (-1002, "could not be started"),
        # review P2: -1 and -2 are SIGHUP and SIGINT, not doctor sentinels
        (-1, "crashed (SIGHUP)"),
        (-2, "crashed (SIGINT)"),
        (-11, "crashed (SIGSEGV)"),
        (-6, "crashed (SIGABRT)"),
        (1, "exited 1"),
    ],
)
def test_doctor_diagnose_describe_returncode(diag, returncode, expected):
    assert diag.describe_returncode(returncode) == expected


def test_doctor_diagnose_explain_failure_falls_back(diag):
    detail, remediation, diagnoses = diag.explain_failure(
        make_healthy_accessor(),
        "tool",
        1,
        "",
        "warming up\nboom: bad thing\n",
        "Run it yourself.",
    )
    # unrecognised output keeps raw evidence and names the next step
    assert detail == "tool exited 1; last output: boom: bad thing"
    assert remediation.startswith("Run it yourself.")
    assert "ROCPROFILER_LOG_LEVEL=info" in remediation
    assert diagnoses == []


def test_doctor_diagnose_explain_failure_names_cause_and_related_checks(diag):
    # /dev/kfd exists but is not accessible: that corroborates the GPU-access cause
    accessor = make_healthy_accessor(cannot_write=["/dev/kfd"])
    detail, remediation, _ = diag.explain_failure(
        accessor,
        "rocprofv3-avail info",
        1,
        "",
        "HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed",
        "Run it yourself.",
    )
    assert detail.startswith("rocprofv3-avail info exited 1: ")
    assert "likely cause: the current user cannot open /dev/kfd" in detail
    assert "usermod" in remediation
    assert "driver.kfd-readable" in remediation


def test_doctor_diagnose_out_of_resources_with_accessible_kfd_is_not_permissions(diag):
    """Review P1: ROCr also returns OUT_OF_RESOURCES for real allocation
    failures. With /dev/kfd accessible, permission fixes are not offered."""
    text = "HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed"
    (match,) = diag.diagnose(make_healthy_accessor(), text)
    assert match["confidence"] == "possible"
    assert "possible cause" in match["summary"]
    assert "usermod" not in match["remediation"]
    assert "not a permission problem" in match["remediation"]


def test_doctor_diagnose_out_of_resources_without_kfd(diag):
    accessor = make_healthy_accessor(files=[], dirs=["/fake/rocm"])
    (match,) = diag.diagnose(accessor, "HSA_STATUS_ERROR_OUT_OF_RESOURCES")
    assert match["confidence"] == "high"
    assert "/dev/kfd does not exist" in match["cause"]


def test_doctor_diagnose_device_lock_permission_is_scoped(diag):
    """Review P1: the KFD lock needs CAP_PERFMON; the sysctl does not govern it,
    and setcap on the rocprofv3 script has no effect."""
    text = "Device 1 could not be locked for profiling due to lack of permissions"
    (match,) = diag.diagnose(make_healthy_accessor(), text)
    assert "perf_event_paranoid" not in match["remediation"]
    assert "sysctl" not in match["remediation"]
    assert "setcap" not in match["remediation"]
    assert "Dispatch counter values may be inaccurate" in match["observation"]
    assert "device-wide" in match["observation"]


def test_doctor_diagnose_undefined_symbol_corroborated_by_environment(diag):
    other_lib = "/opt/rocm-6.2.0/lib"
    accessor = make_healthy_accessor(
        env={"LD_LIBRARY_PATH": other_lib},
        files=list(make_healthy_accessor()._files) + [other_lib + "/libhsa-runtime64.so"],
        dirs=list(make_healthy_accessor()._dirs) + ["/opt/rocm-6.2.0", other_lib],
    )
    (match,) = diag.diagnose(accessor, SDK + ": undefined symbol: foo")
    assert match["confidence"] == "high"
    assert other_lib in match["cause"]
    assert other_lib in match["remediation"]


def test_doctor_diagnose_undefined_symbol_uncorroborated_is_possible(diag):
    (match,) = diag.diagnose(make_healthy_accessor(), SDK + ": undefined symbol: foo")
    assert match["confidence"] == "possible"
    assert "LD_DEBUG" in match["remediation"]


def test_doctor_diagnose_no_device_names_the_mask(diag):
    accessor = make_healthy_accessor(env={"HIP_VISIBLE_DEVICES": "-1"})
    (match,) = diag.diagnose(accessor, "hipErrorNoDevice")
    assert match["confidence"] == "high"
    assert "HIP_VISIBLE_DEVICES=-1" in match["cause"]
    assert "unset HIP_VISIBLE_DEVICES" in match["remediation"]


def test_doctor_diagnose_context_conflict_names_injected_tool(diag):
    accessor = make_healthy_accessor(env={"HSA_TOOLS_LIB": "/x/libother.so"})
    (match,) = diag.diagnose(accessor, "Context has a conflict with another context")
    assert match["confidence"] == "high"
    assert "unset HSA_TOOLS_LIB" in match["remediation"]


def test_doctor_diagnose_related_checks_exist(diag, rocprofv3_package):
    """A typo in a related check id would point the user at nothing."""
    from rocprofv3 import doctor

    known = set(check.id for check in doctor.get_checks())
    related = set()
    for sig in diag.SIGNATURES:
        related.update(sig[3])
    for item in diag.diagnose(make_healthy_accessor(), "", -11):
        related.update(item["related_checks"])
    assert related - known == set()


# ----------------------------------------------------------------------
# runtime.libraries-load
# ----------------------------------------------------------------------
def test_doctor_load_healthy_passes(checks):
    accessor = make_healthy_accessor()
    result = checks["runtime"].check_libraries_load(accessor)
    assert result.status == checks["status"].STATUS_PASS, result.detail
    assert "6 libraries load" in result.detail


def test_doctor_load_runs_isolated_and_side_effect_free(checks):
    accessor = make_healthy_accessor(env={"LD_PRELOAD": "/x/librocprofiler-sdk-tool.so"})
    checks["runtime"].check_libraries_load(accessor)
    loads = [(cmd, env) for cmd, env in accessor.calls if "-c" in cmd]
    assert loads, "no library was loaded"
    for cmd, env in loads:
        assert "-I" in cmd
        assert "RTLD_NOW" in cmd[cmd.index("-c") + 1]
        assert env["LD_PRELOAD"] == ""
        assert env["ROCPROFILER_LIBRARY_CTOR"] == "0"


def test_doctor_load_undefined_symbol_fails_with_diagnosis(checks):
    accessor = make_healthy_accessor(
        runs={SDK: (3, SDK + ": undefined symbol: _ZN11rocprofiler4initEv", "")}
    )
    result = checks["runtime"].check_libraries_load(accessor)
    assert result.status == checks["status"].STATUS_FAIL
    assert "librocprofiler-sdk.so" in result.detail
    assert "_ZN11rocprofiler4initEv" in result.detail
    assert "LD_LIBRARY_PATH" in result.remediation
    sdk = result.data["libraries"]["librocprofiler-sdk.so"]
    assert sdk["diagnoses"][0]["id"] == "loader.undefined-symbol"


def test_doctor_load_optional_library_only_warns(checks):
    accessor = make_healthy_accessor(
        runs={HIP: (3, "libdrm_amdgpu.so.1: cannot open shared object file", "")}
    )
    result = checks["runtime"].check_libraries_load(accessor)
    assert result.status == checks["status"].STATUS_WARN
    assert "libamdhip64.so" in result.detail


def test_doctor_load_lists_every_missing_dependency(checks):
    ldd = (
        "\tlinux-vdso.so.1 (0x00007ffd)\n"
        "\tlibhsa-runtime64.so.1 => not found\n"
        "\tlibamd_comgr.so.3 => not found\n"
        "\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f)\n"
    )
    accessor = make_healthy_accessor(
        runs={
            SDK: (3, "libhsa-runtime64.so.1: cannot open shared object file", ""),
            ("ldd", SDK): (0, ldd, ""),
        }
    )
    result = checks["runtime"].check_libraries_load(accessor)
    assert result.status == checks["status"].STATUS_FAIL
    assert "all missing: libhsa-runtime64.so.1, libamd_comgr.so.3" in result.detail
    sdk = result.data["libraries"]["librocprofiler-sdk.so"]
    assert sdk["missing_dependencies"] == ["libhsa-runtime64.so.1", "libamd_comgr.so.3"]


def test_doctor_load_crash_is_a_failure_not_a_doctor_crash(checks):
    accessor = make_healthy_accessor(runs={SDK: (-11, "", "")})
    result = checks["runtime"].check_libraries_load(accessor)
    assert result.status == checks["status"].STATUS_FAIL
    assert "SIGSEGV" in result.detail


def test_doctor_load_nothing_resolvable_skips(checks):
    accessor = FakeAccessor(rocm_root="/fake/rocm", dirs=["/fake/rocm"])
    result = checks["runtime"].check_libraries_load(accessor)
    assert result.status == checks["status"].STATUS_SKIP


# ----------------------------------------------------------------------
# diagnosis wired into tool-running checks
# ----------------------------------------------------------------------
def test_doctor_avail_runnable_explains_loader_error(checks):
    traceback = (
        "Traceback (most recent call last):\n"
        '  File "/fake/rocm/lib/python3/site-packages/rocprofv3/avail.py", line 1\n'
        "OSError: libhsa-runtime64.so.1: cannot open shared object file: "
        "No such file or directory\n"
    )
    accessor = make_healthy_accessor(runs={"rocprofv3-avail": (1, "", traceback)})
    result = checks["tools"].check_rocprofv3_avail_runnable(accessor)
    assert result.status == checks["status"].STATUS_WARN
    assert "libhsa-runtime64.so.1" in result.detail
    assert "part of ROCm" in result.remediation
    assert result.data["diagnoses"][0]["id"] == "loader.missing-dependency"


def test_doctor_avail_enumeration_explains_gpu_access(checks):
    accessor = make_healthy_accessor(
        cannot_write=["/dev/kfd"],
        runs={
            "rocprofv3-avail": (
                1,
                "",
                "HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate",
            )
        },
    )
    result = checks["counters"].check_avail_enumeration(accessor)
    assert result.status == checks["status"].STATUS_WARN
    assert "cannot open /dev/kfd" in result.detail
    assert "usermod" in result.remediation


def test_doctor_avail_enumeration_unrecognised_error_keeps_fallback(checks):
    accessor = make_healthy_accessor(runs={"rocprofv3-avail": (1, "", "oops")})
    result = checks["counters"].check_avail_enumeration(accessor)
    assert result.status == checks["status"].STATUS_WARN
    assert "rocprofv3-avail info --pmc" in result.remediation


def test_doctor_load_uses_the_launchers_library_path(checks):
    """Review P2: rocprofv3 appends $ROCM_DIR/lib to LD_LIBRARY_PATH for the
    profiled process; the probe must too, or a working setup looks broken."""
    accessor = make_healthy_accessor(env={"LD_LIBRARY_PATH": "/site/lib"})
    result = checks["runtime"].check_libraries_load(accessor)
    loads = [env for cmd, env in accessor.calls if "-c" in cmd]
    assert loads
    for env in loads:
        assert env["LD_LIBRARY_PATH"] == "/site/lib:/fake/rocm/lib"
    assert (
        result.data["probe_environment"]["LD_LIBRARY_PATH"] == "/site/lib:/fake/rocm/lib"
    )
