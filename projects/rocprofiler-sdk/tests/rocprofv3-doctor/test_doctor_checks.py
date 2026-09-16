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
# THE SOFTWARE.

"""GPU-free unit tests for individual rocprofv3-doctor checks.

Each test drives one check against a synthetic machine built with
``FakeAccessor``, so conditions that would be impossible to arrange on the CI
host (no GPU, a wedged driver, a hostile environment) are all reachable.
"""

import pytest

from conftest import FakeAccessor, make_healthy_accessor

RUN_TIMEOUT = -1001  # rocprofv3.doctor_env.RUN_TIMEOUT


@pytest.fixture
def checks(rocprofv3_package):
    from rocprofv3 import (
        doctor_checks_container,
        doctor_checks_counters,
        doctor_checks_driver,
        doctor_checks_environ,
        doctor_checks_filesystem,
        doctor_checks_install,
        doctor_checks_python_env,
        doctor_checks_runtime,
        doctor_checks_tools,
        doctor_result,
    )

    return {
        "container": doctor_checks_container,
        "counters": doctor_checks_counters,
        "driver": doctor_checks_driver,
        "environ": doctor_checks_environ,
        "filesystem": doctor_checks_filesystem,
        "install": doctor_checks_install,
        "python": doctor_checks_python_env,
        "runtime": doctor_checks_runtime,
        "tools": doctor_checks_tools,
        "status": doctor_result,
    }


# ----------------------------------------------------------------------
# install
# ----------------------------------------------------------------------
def test_doctor_checks_rocm_root_pass(checks):
    accessor = make_healthy_accessor()
    result = checks["install"].check_rocm_root(accessor)
    assert result.status == checks["status"].STATUS_PASS


def test_doctor_checks_rocm_root_missing_fails(checks):
    accessor = FakeAccessor(rocm_root="/nonexistent/rocm")
    result = checks["install"].check_rocm_root(accessor)
    assert result.status == checks["status"].STATUS_FAIL
    assert result.remediation, "a failure must tell the user what to do"


def test_doctor_checks_rocm_root_override_is_respected(checks):
    accessor = make_healthy_accessor(rocm_root="/fake/rocm")
    result = checks["install"].check_rocm_root(accessor)
    assert result.data["rocm_root"] == "/fake/rocm"


def test_doctor_checks_sdk_library_pass(checks):
    result = checks["install"].check_sdk_library(make_healthy_accessor())
    assert result.status == checks["status"].STATUS_PASS
    assert result.data["resolved_path"].endswith("librocprofiler-sdk.so")


def test_doctor_checks_sdk_library_missing_fails(checks):
    accessor = make_healthy_accessor(files=[], dirs=["/fake/rocm"])
    result = checks["install"].check_sdk_library(accessor)
    assert result.status == checks["status"].STATUS_FAIL
    assert result.data["searched"], "failure must report where it looked"


def test_doctor_checks_sdk_library_versioned_soname_only(checks):
    """A runtime-only package ships librocprofiler-sdk.so.0.4.0 with no symlink."""
    accessor = make_healthy_accessor(
        files=["/fake/rocm/lib/librocprofiler-sdk.so.0.4.0"], dirs=["/fake/rocm"]
    )
    result = checks["install"].check_sdk_library(accessor)
    assert result.status == checks["status"].STATUS_PASS


def test_doctor_checks_kokkosp_missing_is_only_a_warning(checks):
    """Optional libraries must never fail the run."""
    accessor = make_healthy_accessor(files=[], dirs=["/fake/rocm"])
    result = checks["install"].check_kokkosp_library(accessor)
    assert result.status == checks["status"].STATUS_WARN


def test_doctor_checks_version_consistency_match(checks):
    accessor = make_healthy_accessor(
        files=["/fake/rocm/lib/librocprofiler-sdk.so.1.4.0"],
        dirs=["/fake/rocm"],
        tool_version="1.4.0",
    )
    result = checks["install"].check_version_consistency(accessor)
    assert result.status == checks["status"].STATUS_PASS


def test_doctor_checks_version_consistency_major_skew_fails(checks):
    accessor = make_healthy_accessor(
        files=["/fake/rocm/lib/librocprofiler-sdk.so.2.0.0"],
        dirs=["/fake/rocm"],
        tool_version="1.4.0",
    )
    result = checks["install"].check_version_consistency(accessor)
    assert result.status == checks["status"].STATUS_FAIL


def test_doctor_checks_version_consistency_minor_skew_warns(checks):
    """A minor skew is usually benign, so it must not be reported as a failure."""
    accessor = make_healthy_accessor(
        files=["/fake/rocm/lib/librocprofiler-sdk.so.1.5.0"],
        dirs=["/fake/rocm"],
        tool_version="1.4.0",
    )
    result = checks["install"].check_version_consistency(accessor)
    assert result.status == checks["status"].STATUS_WARN


def test_doctor_checks_version_consistency_unconfigured_skips(checks):
    """Running from an unconfigured source tree must skip, not fail."""
    accessor = make_healthy_accessor(tool_version="@FULL_VERSION_STRING@")
    result = checks["install"].check_version_consistency(accessor)
    assert result.status == checks["status"].STATUS_SKIP


def test_doctor_checks_no_mixed_rocm_warns_on_foreign_ld_library_path(checks):
    accessor = make_healthy_accessor(
        env={"LD_LIBRARY_PATH": "/opt/rocm-1.2/lib"},
    )
    result = checks["install"].check_no_mixed_rocm(accessor)
    assert result.status == checks["status"].STATUS_WARN
    assert "rocm-1.2" in result.detail


def test_doctor_checks_no_mixed_rocm_ignores_non_rocm_entries(checks):
    accessor = make_healthy_accessor(env={"LD_LIBRARY_PATH": "/usr/local/lib"})
    result = checks["install"].check_no_mixed_rocm(accessor)
    assert result.status == checks["status"].STATUS_PASS


def test_doctor_checks_no_mixed_rocm_warns_on_foreign_rocm_path(checks):
    accessor = make_healthy_accessor(env={"ROCM_PATH": "/opt/rocm-6.2.0"})
    result = checks["install"].check_no_mixed_rocm(accessor)
    assert result.status == checks["status"].STATUS_WARN


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def test_doctor_checks_kfd_device_pass(checks):
    result = checks["driver"].check_kfd_device(make_healthy_accessor())
    assert result.status == checks["status"].STATUS_PASS


def test_doctor_checks_kfd_device_missing_fails(checks):
    accessor = make_healthy_accessor(files=[], dirs=["/fake/rocm"])
    result = checks["driver"].check_kfd_device(accessor)
    assert result.status == checks["status"].STATUS_FAIL
    assert "modprobe" in result.remediation or "amdgpu" in result.remediation


def test_doctor_checks_kfd_not_readable_fails(checks):
    accessor = make_healthy_accessor(cannot_write=["/dev/kfd"])
    result = checks["driver"].check_kfd_readable(accessor)
    assert result.status == checks["status"].STATUS_FAIL


def test_doctor_checks_render_group_pass(checks):
    result = checks["driver"].check_render_group(make_healthy_accessor())
    assert result.status == checks["status"].STATUS_PASS


def test_doctor_checks_render_group_stale_session_says_relogin(checks):
    """Member on paper but not in this session: advise re-login, not usermod."""
    accessor = make_healthy_accessor(
        groups=[44],
        group_gids={"render": 109, "video": 44},
        group_members={"render": ["tester"], "video": ["tester"]},
    )
    result = checks["driver"].check_render_group(accessor)
    assert result.status == checks["status"].STATUS_WARN
    assert "newgrp" in result.remediation
    assert "usermod" not in result.remediation


def test_doctor_checks_render_group_non_member_says_usermod(checks):
    accessor = make_healthy_accessor(
        groups=[44],
        group_gids={"render": 109, "video": 44},
        group_members={"render": ["someone-else"], "video": ["tester"]},
    )
    result = checks["driver"].check_render_group(accessor)
    assert result.status == checks["status"].STATUS_WARN
    assert "usermod" in result.remediation


def test_doctor_checks_missing_group_is_a_warning(checks):
    accessor = make_healthy_accessor(groups=[], group_gids={}, group_members={})
    result = checks["driver"].check_render_group(accessor)
    assert result.status == checks["status"].STATUS_WARN


def test_doctor_checks_amdgpu_module_pass(checks):
    result = checks["driver"].check_amdgpu_module(make_healthy_accessor())
    assert result.status == checks["status"].STATUS_PASS
    assert result.data["amdgpu_version"] == "6.16.13"


def test_doctor_checks_amdgpu_module_missing_fails(checks):
    accessor = make_healthy_accessor(dirs=["/fake/rocm"])
    result = checks["driver"].check_amdgpu_module(accessor)
    assert result.status == checks["status"].STATUS_FAIL


def test_doctor_checks_amdgpu_in_tree_driver_without_version_passes(checks):
    """Review P1: the in-tree driver has no /sys/module/amdgpu/version (only
    DKMS sets one); its absence must not be read as "driver not loaded"."""
    accessor = make_healthy_accessor(file_contents={})
    result = checks["driver"].check_amdgpu_module(accessor)
    assert result.status == checks["status"].STATUS_PASS
    assert result.data["amdgpu_version"] is None


def test_doctor_checks_gpu_topology_pass(checks):
    result = checks["driver"].check_gpu_topology(make_healthy_accessor())
    assert result.status == checks["status"].STATUS_PASS
    assert len(result.data["gpu_nodes"]) == 1


def test_doctor_checks_gpu_topology_no_nodes_fails(checks):
    accessor = make_healthy_accessor(files=[], dirs=["/fake/rocm"], file_contents={})
    result = checks["driver"].check_gpu_topology(accessor)
    assert result.status == checks["status"].STATUS_FAIL


def test_doctor_checks_gpu_topology_cpu_only_fails(checks):
    """Nodes exist but every one reports simd_count 0: no GPU."""
    accessor = make_healthy_accessor(
        file_contents={
            "/sys/class/kfd/kfd/topology/nodes/0/properties": (
                "cpu_cores_count 24\nsimd_count 0\ngfx_target_version 0\n"
            ),
            "/sys/class/kfd/kfd/topology/nodes/1/properties": (
                "cpu_cores_count 24\nsimd_count 0\ngfx_target_version 0\n"
            ),
        }
    )
    result = checks["driver"].check_gpu_topology(accessor)
    assert result.status == checks["status"].STATUS_FAIL


# ----------------------------------------------------------------------
# runtime
# ----------------------------------------------------------------------
def test_doctor_checks_aqlprofile_missing_fails(checks):
    accessor = make_healthy_accessor(files=[], dirs=["/fake/rocm"])
    result = checks["runtime"].check_aqlprofile_library(accessor)
    assert result.status == checks["status"].STATUS_FAIL
    assert "aqlprofile" in result.remediation


def test_doctor_checks_hsa_library_outside_rocm_root_warns(checks):
    accessor = make_healthy_accessor(
        files=["/opt/other/lib/libhsa-runtime64.so"],
        dirs=["/fake/rocm"],
        env={"LD_LIBRARY_PATH": "/opt/other/lib"},
    )
    result = checks["runtime"].check_hsa_library(accessor)
    assert result.status == checks["status"].STATUS_WARN


def test_doctor_checks_att_decoder_missing_is_a_warning(checks):
    accessor = make_healthy_accessor(files=[], dirs=["/fake/rocm"])
    result = checks["runtime"].check_att_decoder_library(accessor)
    assert result.status == checks["status"].STATUS_WARN


def test_doctor_checks_att_decoder_env_override(checks):
    accessor = make_healthy_accessor(
        files=["/custom/att/librocprof-trace-decoder.so"],
        env={"ROCPROF_ATT_LIBRARY_PATH": "/custom/att"},
    )
    result = checks["runtime"].check_att_decoder_library(accessor)
    assert result.status == checks["status"].STATUS_PASS


# ----------------------------------------------------------------------
# counters
# ----------------------------------------------------------------------
def test_doctor_checks_no_perf_event_paranoid_check(checks):
    """Review P1: rocprofiler-sdk does not use perf events, so nothing may
    recommend changing perf_event_paranoid."""
    from rocprofv3 import doctor

    ids = [check.id for check in doctor.get_checks()]
    assert "counters.perfmon-paranoid" not in ids


def test_doctor_checks_cap_perfmon_absent_still_passes(checks):
    """Missing CAP_PERFMON is normal for an unprivileged account; do not alarm,
    but say truthfully what it affects."""
    accessor = make_healthy_accessor(cap_eff=0)
    result = checks["counters"].check_cap_perfmon(accessor)
    assert result.status == checks["status"].STATUS_PASS
    assert "device-wide" in result.detail
    assert "dispatch counter collection works" in result.detail


def test_doctor_checks_cap_perfmon_never_suggests_setcap_on_script(checks):
    """Review P1: file capabilities on the rocprofv3 Python script do nothing."""
    result = checks["counters"].check_cap_perfmon(make_healthy_accessor(cap_eff=0))
    assert "setcap" not in result.remediation
    assert "perf_event_paranoid" not in result.remediation


def test_doctor_checks_cap_perfmon_present(checks):
    accessor = make_healthy_accessor(cap_eff=1 << 38)
    result = checks["counters"].check_cap_perfmon(accessor)
    assert result.status == checks["status"].STATUS_PASS
    assert result.data["cap_perfmon"] is True


def test_doctor_checks_avail_enumeration_timeout_fails(checks):
    accessor = make_healthy_accessor(
        runs={"rocprofv3-avail": (RUN_TIMEOUT, "", "timed out after 15 seconds")}
    )
    result = checks["counters"].check_avail_enumeration(accessor)
    assert result.status == checks["status"].STATUS_FAIL
    assert "timed out" in result.detail


def test_doctor_checks_avail_enumeration_success(checks):
    accessor = make_healthy_accessor(
        runs={"rocprofv3-avail": (0, "GPU:0\nCounter_Name:\tSQ_WAVES\n", "")}
    )
    result = checks["counters"].check_avail_enumeration(accessor)
    assert result.status == checks["status"].STATUS_PASS


# ----------------------------------------------------------------------
# environ
# ----------------------------------------------------------------------
def test_doctor_checks_ld_preload_clean(checks):
    result = checks["environ"].check_ld_preload_conflict(make_healthy_accessor())
    assert result.status == checks["status"].STATUS_PASS


def test_doctor_checks_ld_preload_roctracer_conflict_fails(checks):
    accessor = make_healthy_accessor(
        env={"LD_PRELOAD": "/opt/rocm/lib/libroctracer64.so"}
    )
    result = checks["environ"].check_ld_preload_conflict(accessor)
    assert result.status == checks["status"].STATUS_FAIL
    assert "unset LD_PRELOAD" in result.remediation


def test_doctor_checks_ld_preload_sdk_only_passes(checks):
    accessor = make_healthy_accessor(
        env={"LD_PRELOAD": "/opt/rocm/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"}
    )
    result = checks["environ"].check_ld_preload_conflict(accessor)
    assert result.status == checks["status"].STATUS_PASS


def test_doctor_checks_gpu_visibility_empty_warns(checks):
    accessor = make_healthy_accessor(env={"ROCR_VISIBLE_DEVICES": ""})
    result = checks["environ"].check_gpu_visibility(accessor)
    assert result.status == checks["status"].STATUS_WARN
    assert "hides every GPU" in result.detail


def test_doctor_checks_gpu_visibility_unset_passes(checks):
    result = checks["environ"].check_gpu_visibility(make_healthy_accessor())
    assert result.status == checks["status"].STATUS_PASS


def test_doctor_checks_gpu_visibility_restricted_warns(checks):
    accessor = make_healthy_accessor(env={"HIP_VISIBLE_DEVICES": "0"})
    result = checks["environ"].check_gpu_visibility(accessor)
    assert result.status == checks["status"].STATUS_WARN


def test_doctor_checks_rocprof_vars_dump_collects_everything(checks):
    env = {
        "ROCPROF_OUTPUT_PATH": "/tmp/out",
        "ROCPROFILER_METRICS_PATH": "/opt/rocm/share",
        "ROCP_TOOL_LIBRARIES": "/opt/rocm/lib/tool.so",
        "UNRELATED_VAR": "ignored",
    }
    accessor = make_healthy_accessor(env=env)
    result = checks["environ"].check_rocprof_vars_dump(accessor)
    assert result.status == checks["status"].STATUS_PASS
    assert "ROCPROF_OUTPUT_PATH" in result.data
    assert "ROCPROFILER_METRICS_PATH" in result.data
    assert "UNRELATED_VAR" not in result.data


def test_doctor_checks_hsa_tools_lib_foreign_warns(checks):
    accessor = make_healthy_accessor(env={"HSA_TOOLS_LIB": "/opt/other/libfoo.so"})
    result = checks["environ"].check_hsa_tools_lib(accessor)
    assert result.status == checks["status"].STATUS_WARN


# ----------------------------------------------------------------------
# container
# ----------------------------------------------------------------------
def test_doctor_checks_container_not_detected(checks):
    result = checks["container"].check_container_detected(make_healthy_accessor())
    assert result.status == checks["status"].STATUS_PASS
    assert result.data["container_runtime"] is None


def test_doctor_checks_container_docker_detected(checks):
    accessor = make_healthy_accessor(files=["/.dockerenv"])
    result = checks["container"].check_container_detected(accessor)
    assert result.data["container_runtime"] == "docker"


def test_doctor_checks_container_missing_devices_warns(checks):
    accessor = FakeAccessor(files=["/.dockerenv"], dirs=["/fake/rocm"])
    result = checks["container"].check_device_passthrough(accessor)
    assert result.status == checks["status"].STATUS_WARN
    assert "--device=/dev/kfd" in result.remediation


def test_doctor_checks_container_passthrough_ok(checks):
    accessor = make_healthy_accessor(
        files=list(make_healthy_accessor()._files) + ["/.dockerenv"]
    )
    result = checks["container"].check_device_passthrough(accessor)
    assert result.status == checks["status"].STATUS_PASS


# ----------------------------------------------------------------------
# python environment
# ----------------------------------------------------------------------
def test_doctor_checks_python_version_ok(checks):
    result = checks["python"].check_python_version(make_healthy_accessor())
    assert result.status == checks["status"].STATUS_PASS


def test_doctor_checks_python_version_too_old_fails(checks):
    accessor = make_healthy_accessor(python_version=(2, 7, 18))
    result = checks["python"].check_python_version(accessor)
    assert result.status == checks["status"].STATUS_FAIL


def test_doctor_checks_sqlite_importable_but_broken_fails(checks):
    """Minimal images ship a Python whose sqlite3 imports but cannot connect."""
    accessor = make_healthy_accessor(sqlite_ok=False)
    result = checks["python"].check_sqlite3(accessor)
    assert result.status == checks["status"].STATUS_FAIL


def test_doctor_checks_sqlite_ok(checks):
    result = checks["python"].check_sqlite3(make_healthy_accessor())
    assert result.status == checks["status"].STATUS_PASS


def test_doctor_checks_optional_module_missing_is_a_warning(checks):
    accessor = make_healthy_accessor(modules={})
    result = checks["python"].check_pyyaml(accessor)
    assert result.status == checks["status"].STATUS_WARN


def test_doctor_checks_required_module_missing_fails(checks):
    accessor = make_healthy_accessor(modules={})
    result = checks["python"].check_ctypes(accessor)
    assert result.status == checks["status"].STATUS_FAIL


# ----------------------------------------------------------------------
# filesystem
# ----------------------------------------------------------------------
def test_doctor_checks_output_dir_writable(checks):
    accessor = make_healthy_accessor(
        env={"ROCPROF_OUTPUT_PATH": "/tmp"}, writable_dirs=["/tmp"]
    )
    result = checks["filesystem"].check_output_dir_writable(accessor)
    assert result.status == checks["status"].STATUS_PASS


def test_doctor_checks_output_dir_not_writable_fails(checks):
    accessor = make_healthy_accessor(
        env={"ROCPROF_OUTPUT_PATH": "/tmp"}, writable_dirs=[]
    )
    result = checks["filesystem"].check_output_dir_writable(accessor)
    assert result.status == checks["status"].STATUS_FAIL


def test_doctor_checks_disk_space_plenty(checks):
    accessor = make_healthy_accessor(env={"ROCPROF_OUTPUT_PATH": "/tmp"})
    result = checks["filesystem"].check_output_disk_space(accessor)
    assert result.status == checks["status"].STATUS_PASS


def test_doctor_checks_disk_space_low_warns(checks):
    accessor = make_healthy_accessor(
        env={"ROCPROF_OUTPUT_PATH": "/tmp"}, free_bytes=100 * 1024 * 1024
    )
    result = checks["filesystem"].check_output_disk_space(accessor)
    assert result.status == checks["status"].STATUS_WARN


def test_doctor_checks_disk_space_critical_fails(checks):
    accessor = make_healthy_accessor(
        env={"ROCPROF_OUTPUT_PATH": "/tmp"}, free_bytes=1024 * 1024
    )
    result = checks["filesystem"].check_output_disk_space(accessor)
    assert result.status == checks["status"].STATUS_FAIL


# ----------------------------------------------------------------------
# whole-run behaviour on synthetic machines
# ----------------------------------------------------------------------
def test_doctor_checks_healthy_machine_has_no_failures(checks):
    from rocprofv3 import doctor

    accessor = make_healthy_accessor(
        env={"ROCPROF_OUTPUT_PATH": "/tmp"},
        writable_dirs=["/tmp"],
        runs={"rocprofv3-avail": (0, "GPU:0\nCounter_Name:\tSQ_WAVES\n", "")},
    )
    outcomes = doctor.run(accessor)
    failures = [
        check.id for check, result in outcomes if result.status == doctor.STATUS_FAIL
    ]
    assert failures == []
    assert doctor.exit_code(outcomes) == 0


def test_doctor_checks_no_gpu_machine_skips_rather_than_cascading(checks):
    """The headline no-GPU requirement: driver checks skip, nothing crashes."""
    from rocprofv3 import doctor

    accessor = FakeAccessor(
        rocm_root="/fake/rocm",
        dirs=["/fake/rocm", "/tmp"],
        files=["/fake/rocm/lib/librocprofiler-sdk.so"],
        writable_dirs=["/tmp"],
    )
    outcomes = doctor.run(accessor)
    statuses = dict((check.id, result.status) for check, result in outcomes)

    assert statuses["driver.kfd-device"] == doctor.STATUS_FAIL
    for dependent in (
        "driver.kfd-readable",
        "driver.gpu-topology",
        "driver.render-group",
        "counters.cap-perfmon",
    ):
        assert statuses[dependent] == doctor.STATUS_SKIP, dependent


def test_doctor_checks_every_check_returns_a_result(checks):
    """No check may raise on a hostile/empty environment."""
    from rocprofv3 import doctor

    accessor = FakeAccessor(rocm_root="/nonexistent")
    outcomes = doctor.run(accessor)
    assert len(outcomes) == len(doctor.select_checks())
    for check, result in outcomes:
        assert result.status in doctor.STATUS_VALUES, check.id
        assert "traceback" not in result.data, "{} raised: {}".format(
            check.id, result.data.get("traceback")
        )


def test_doctor_checks_failures_carry_remediation(checks):
    """A failing check the user can act on must say how."""
    from rocprofv3 import doctor

    accessor = FakeAccessor(rocm_root="/nonexistent")
    outcomes = doctor.run(accessor)
    for check, result in outcomes:
        if result.status == doctor.STATUS_FAIL:
            assert result.remediation, "{} fails without remediation".format(check.id)


# ----------------------------------------------------------------------
# regression tests for defects found in review
# ----------------------------------------------------------------------
def test_doctor_checks_soversion_ladder_prefers_plain_so(checks):
    """B4: the plain .so devel symlink wins when it exists."""
    accessor = make_healthy_accessor(
        files=[
            "/fake/rocm/lib/librocprofiler-sdk.so",
            "/fake/rocm/lib/librocprofiler-sdk.so.1",
            "/fake/rocm/lib/librocprofiler-sdk.so.1.4.0",
        ],
        dirs=["/fake/rocm"],
        tool_version="1.4.0",
    )
    path, _ = checks["install"].resolve_library(
        accessor, "librocprofiler-sdk.so", ("lib",)
    )
    assert path == "/fake/rocm/lib/librocprofiler-sdk.so"


def test_doctor_checks_soversion_ladder_uses_known_major(checks):
    """B4: with no plain .so, the ladder uses the *known* major version."""
    accessor = make_healthy_accessor(
        files=[
            "/fake/rocm/lib/librocprofiler-sdk.so.1",
            "/fake/rocm/lib/librocprofiler-sdk.so.9",
        ],
        dirs=["/fake/rocm"],
        tool_version="1.4.0",
    )
    path, _ = checks["install"].resolve_library(
        accessor, "librocprofiler-sdk.so", ("lib",)
    )
    assert path == "/fake/rocm/lib/librocprofiler-sdk.so.1", (
        "must pick the soversion matching the tool version, not the "
        "lexicographically largest"
    )


def test_doctor_checks_soversion_ladder_numeric_not_lexicographic(checks):
    """B4 regression: sorted() ranks '.so.9' above '.so.10'; we must not.

    With the tool version unknown the resolver falls back to scanning, and that
    scan has to compare soversions numerically.
    """
    accessor = make_healthy_accessor(
        files=[
            "/fake/rocm/lib/libfoo.so.9",
            "/fake/rocm/lib/libfoo.so.10",
            "/fake/rocm/lib/libfoo.so.10.1.0",
        ],
        dirs=["/fake/rocm"],
        tool_version=None,
    )
    path, _ = checks["install"].resolve_library(accessor, "libfoo.so", ("lib",))
    assert (
        path == "/fake/rocm/lib/libfoo.so.10.1.0"
    ), "lexicographic sort would have picked libfoo.so.9"


def test_doctor_checks_soversion_ladder_tolerates_non_numeric_suffix(checks):
    """B4: a .so.debug sidecar must not be chosen over a real soversion."""
    accessor = make_healthy_accessor(
        files=[
            "/fake/rocm/lib/libfoo.so.2",
            "/fake/rocm/lib/libfoo.so.debug",
        ],
        dirs=["/fake/rocm"],
        tool_version=None,
    )
    path, _ = checks["install"].resolve_library(accessor, "libfoo.so", ("lib",))
    assert path == "/fake/rocm/lib/libfoo.so.2"


def test_doctor_checks_attach_library_is_distinct_from_tool_library(checks):
    """B5: attach-library must check the rocattach injector, not the tool lib.

    Previously both checks resolved librocprofiler-sdk-tool.so, so the attach
    check could never disagree with its sibling.
    """
    accessor = make_healthy_accessor(
        files=["/fake/rocm/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"],
        dirs=["/fake/rocm"],
    )
    tool = checks["install"].check_sdk_tool_library(accessor)
    attach = checks["install"].check_attach_library(accessor)

    assert tool.status == checks["status"].STATUS_PASS
    assert (
        attach.status == checks["status"].STATUS_WARN
    ), "the rocattach injector is absent here, so the two checks must differ"
    assert attach.data["library"] == "librocprofiler-sdk-rocattach.so"


def test_doctor_checks_tool_found_on_path_when_absent_from_rocm_bin(checks):
    """B3: a distro install symlinks the tools into /usr/bin; do not fail."""
    accessor = make_healthy_accessor(
        files=["/usr/bin/rocprofv3-avail"],
        dirs=["/fake/rocm"],
        env={"PATH": "/usr/local/bin:/usr/bin"},
    )
    result = checks["tools"].check_rocprofv3_avail(accessor)
    assert result.status == checks["status"].STATUS_PASS
    assert result.data["path"] == "/usr/bin/rocprofv3-avail"


def test_doctor_checks_tool_prefers_rocm_bin_over_path(checks):
    """B3: the ROCm tree under inspection still wins when it has the tool."""
    accessor = make_healthy_accessor(
        files=["/fake/rocm/bin/rocprofv3-avail", "/usr/bin/rocprofv3-avail"],
        dirs=["/fake/rocm"],
        env={"PATH": "/usr/bin"},
    )
    result = checks["tools"].check_rocprofv3_avail(accessor)
    assert result.data["path"] == "/fake/rocm/bin/rocprofv3-avail"


def test_doctor_checks_tool_missing_everywhere_still_fails(checks):
    """B3: PATH fallback must not mask a genuinely missing tool."""
    accessor = make_healthy_accessor(
        files=[], dirs=["/fake/rocm"], env={"PATH": "/usr/bin"}
    )
    result = checks["tools"].check_rocprofv3_avail(accessor)
    assert result.status == checks["status"].STATUS_FAIL


def test_doctor_checks_healthy_path_only_install_exits_zero(checks):
    """B3 end-to-end: tools on PATH only must not drive exit code 1."""
    from rocprofv3 import doctor

    accessor = make_healthy_accessor(
        files=[
            path
            for path in make_healthy_accessor()._files
            if not path.startswith("/fake/rocm/bin/")
        ]
        + [
            "/usr/bin/rocprofv3",
            "/usr/bin/rocprofv3-avail",
            "/usr/bin/rocprof-attach",
            "/usr/bin/rocpd",
        ],
        env={"PATH": "/usr/bin", "ROCPROF_OUTPUT_PATH": "/tmp"},
        writable_dirs=["/tmp"],
        runs={"rocprofv3-avail": (0, "Counter_Name        :\tSQ_WAVES\n", "")},
    )
    outcomes = doctor.run(accessor)
    failures = [
        check.id for check, result in outcomes if result.status == doctor.STATUS_FAIL
    ]
    assert failures == []


def test_doctor_checks_profiler_lock_detects_shebang_launched_script(checks):
    """B6: rocprofv3 is a #!python3 script, so argv[0] is the interpreter.

    Scanning only argv[0] reported "no other profiler" while a real rocprofv3
    held the device -- a false pass on the exact condition being checked.
    """
    accessor = make_healthy_accessor(
        files=["/proc/9001/cmdline"],
        file_contents={
            "/proc/9001/cmdline": "/usr/bin/python3\x00/opt/rocm/bin/rocprofv3\x00--pmc\x00SQ_WAVES\x00",
        },
    )
    result = checks["counters"].check_no_profiler_lock(accessor)
    assert result.status == checks["status"].STATUS_WARN
    assert "9001" in result.detail


def test_doctor_checks_profiler_lock_ignores_own_pid(checks):
    """B6: self-filter by pid, not by matching 'doctor' in the command line."""
    accessor = make_healthy_accessor(
        pid=4242,
        files=["/proc/4242/cmdline"],
        file_contents={
            "/proc/4242/cmdline": "/usr/bin/python3\x00/opt/rocm/bin/rocprofv3-doctor\x00",
        },
    )
    result = checks["counters"].check_no_profiler_lock(accessor)
    assert result.status == checks["status"].STATUS_PASS


def test_doctor_checks_profiler_lock_does_not_hide_profiler_named_doctor(checks):
    """B6: a real profiler whose args mention 'doctor' must still be reported."""
    accessor = make_healthy_accessor(
        pid=4242,
        files=["/proc/9002/cmdline"],
        file_contents={
            "/proc/9002/cmdline": "/usr/bin/python3\x00/opt/rocm/bin/rocprofv3\x00-d\x00/home/me/doctor-results\x00",
        },
    )
    result = checks["counters"].check_no_profiler_lock(accessor)
    assert result.status == checks["status"].STATUS_WARN
    assert "9002" in result.detail


def test_doctor_checks_profiler_lock_clean_when_nothing_running(checks):
    accessor = make_healthy_accessor(
        files=["/proc/9003/cmdline"],
        file_contents={"/proc/9003/cmdline": "/usr/sbin/sshd\x00-D\x00"},
    )
    result = checks["counters"].check_no_profiler_lock(accessor)
    assert result.status == checks["status"].STATUS_PASS


def test_doctor_checks_avail_enumeration_parses_info_format(checks):
    """B7: match the real `info --pmc` output, not a format it never emits.

    build_counter_string() pads the key to 20 columns, so the line is
    "Counter_Name        :<TAB>SQ_WAVES".
    """
    stdout = (
        "Counter_Name        :\tSQ_WAVES\n"
        "Description         :\tCount number of waves\n"
        "Block               :\tSQ\n"
        "\n"
        "Counter_Name        :\tGRBM_COUNT\n"
        "Description         :\tFree running clock\n"
    )
    accessor = make_healthy_accessor(runs={"rocprofv3-avail": (0, stdout, "")})
    result = checks["counters"].check_avail_enumeration(accessor)
    assert result.status == checks["status"].STATUS_PASS
    assert result.data["counter_entries"] == 2


def test_doctor_checks_avail_enumeration_invokes_info_subcommand(checks):
    """B7: the `list` subcommand prints a bare grid; we must call `info`."""
    seen = {}

    class RecordingAccessor(type(make_healthy_accessor())):
        def run(self, cmd, timeout=10, env=None):
            seen["cmd"] = cmd
            return (0, "Counter_Name        :\tSQ_WAVES\n", "")

    accessor = RecordingAccessor(
        rocm_root="/fake/rocm",
        files=["/fake/rocm/bin/rocprofv3-avail"],
        dirs=["/fake/rocm"],
    )
    checks["counters"].check_avail_enumeration(accessor)
    assert "info" in seen["cmd"]
    assert "list" not in seen["cmd"]


@pytest.mark.parametrize(
    "capability",
    [
        2823004800,  # this dev machine's real value (ASICRevision=1)
        (8 << 22) | 1,  # ASICRevision=8 -- sets bit 25
        (15 << 22) | 1,  # ASICRevision=15 -- sets bit 25
    ],
)
def test_doctor_checks_virt_gpu_no_false_sriov_on_ordinary_gpu(checks, capability):
    """B8: bits 22-25 are the ASICRevision field, not an SR-IOV flag.

    Any GPU with ASIC revision >= 8 sets bit 25, so the old `capability & (1<<25)`
    test warned "SR-IOV virtual function -- hardware counters unavailable" on
    ordinary hardware. See HSA_CAPABILITY in
    source/include/rocprofiler-sdk/cxx/serialization/save.hpp.
    """
    accessor = make_healthy_accessor(
        file_contents={
            "/sys/class/kfd/kfd/topology/nodes/1/properties": (
                "simd_count 256\ngfx_target_version 90000\n"
                "capability {}\n".format(capability)
            ),
        }
    )
    result = checks["container"].check_virt_gpu(accessor)
    assert result.status == checks["status"].STATUS_PASS
    assert "SR-IOV" not in result.detail


def test_doctor_checks_path_containment_is_separator_anchored(rocprofv3_package):
    """B9: /opt/rocm-6.2.0-alt is NOT inside /opt/rocm-6.2.

    Exercises the real SystemAccessor, not the test double, since this is the
    implementation the checks actually call.
    """
    from rocprofv3.doctor_env import SystemAccessor

    accessor = SystemAccessor(rocm_root="/opt/rocm-6.2")
    assert accessor.path_within("/opt/rocm-6.2/lib/libfoo.so", "/opt/rocm-6.2")
    assert accessor.path_within("/opt/rocm-6.2", "/opt/rocm-6.2")
    assert accessor.path_within("/opt/rocm-6.2/", "/opt/rocm-6.2")
    assert not accessor.path_within("/opt/rocm-6.2.0-alt/lib/libfoo.so", "/opt/rocm-6.2")
    assert not accessor.path_within("/opt/rocm-6.20/lib", "/opt/rocm-6.2")


def test_doctor_checks_mixed_rocm_detects_sibling_prefix_conflict(checks):
    """B9: a confusingly-named sibling tree must be reported as a conflict."""
    accessor = make_healthy_accessor(
        rocm_root="/opt/rocm-6.2",
        env={"LD_LIBRARY_PATH": "/opt/rocm-6.2.0-alt/lib"},
    )
    result = checks["install"].check_no_mixed_rocm(accessor)
    assert result.status == checks["status"].STATUS_WARN


def test_doctor_checks_runtime_sibling_prefix_is_outside_root(checks):
    """B9: same anchoring bug in the runtime library location warning."""
    accessor = make_healthy_accessor(
        rocm_root="/opt/rocm-6.2",
        files=["/opt/rocm-6.2.0-alt/lib/libhsa-runtime64.so"],
        dirs=["/opt/rocm-6.2"],
        env={"LD_LIBRARY_PATH": "/opt/rocm-6.2.0-alt/lib"},
    )
    result = checks["runtime"].check_hsa_library(accessor)
    assert result.status == checks["status"].STATUS_WARN


def test_doctor_checks_env_dump_excludes_unrelated_rocm_vars(checks):
    """Lower-priority: HSA_/HIP_/AMD_ bury the signal; only ROCPROF* now."""
    accessor = make_healthy_accessor(
        env={
            "ROCPROF_OUTPUT_PATH": "/tmp/out",
            "ROCPROFILER_METRICS_PATH": "/opt/rocm/share",
            "ROCP_TOOL_LIBRARIES": "/opt/rocm/lib/tool.so",
            "HSA_ENABLE_SDMA": "0",
            "HIP_VISIBLE_DEVICES": "0",
            "AMD_LOG_LEVEL": "4",
        }
    )
    result = checks["environ"].check_rocprof_vars_dump(accessor)
    assert set(result.data.keys()) == {
        "ROCPROF_OUTPUT_PATH",
        "ROCPROFILER_METRICS_PATH",
        "ROCP_TOOL_LIBRARIES",
    }
