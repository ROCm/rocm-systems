#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import re
import socket
import subprocess
import sys
from collections import defaultdict

import pytest

AGENT_KEYS = {
    "array_count",
    "cpu_cores_count",
    "cu_count",
    "cu_per_simd_array",
    "gfx_target_version",
    "gpu_id",
    "grid_max_dim",
    "logical_node_id",
    "logical_node_type_id",
    "max_waves_per_cu",
    "max_waves_per_simd",
    "model_name",
    "name",
    "node_id",
    "num_shader_banks",
    "num_xcc",
    "product_name",
    "runtime_visibility",
    "simd_arrays_per_engine",
    "simd_count",
    "simd_per_cu",
    "vendor_name",
    "wave_front_size",
    "workgroup_max_dim",
}
MAX_COUNTER_BATCH = 32
PC_CONFIG_KEYS = ("method", "unit", "min_interval", "max_interval", "flags")
SPM_CONFIG_KEYS = ("type", "minimum_interval", "maximum_interval")
GPU_RE = re.compile(r"(?m)^[ \t]*GPU[ \t]*:[ \t]*(\d+)[ \t]*$")
FIELD_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")
PMC_CHECK_RE = re.compile(
    r"^Following input counters can be collected together on GPU:(\d+)\s+(.+)$"
)
# rocprofv3 aborts when ROCPROFILER_CI requests tracing options, and the
# ROCPROF_OUTPUT_* variables change where the listing is written.
CONFLICTING_ENVIRONMENT_VARIABLES = (
    "ROCPROFILER_CI",
    "ROCPROF_MPI_RANKS",
    "ROCPROF_OUTPUT_FILE_NAME",
    "ROCPROF_OUTPUT_PATH",
)


def _value(item):
    return getattr(item, "value", item)


def set_library(rocm_path):
    sys.path.append(
        "{}/lib/python{}/site-packages".format(rocm_path, sys.version_info[0])
    )
    os.environ["ROCPROFILER_METRICS_PATH"] = "{}/share/rocprofiler-sdk".format(rocm_path)
    os.environ["ROCPROFILER_PC_SAMPLING_BETA_ENABLED"] = "on"

    from rocprofv3 import avail

    avail.loadLibrary.libname = (
        "{}/lib/rocprofiler-sdk/librocprofv3-list-avail.so".format(rocm_path)
    )
    return avail


@pytest.fixture(scope="session")
def cli(request):
    rocm_path = request.config.getoption("--rocm-path")
    environment = os.environ.copy()
    for key in CONFLICTING_ENVIRONMENT_VARIABLES:
        environment.pop(key, None)
    environment["ROCPROFILER_METRICS_PATH"] = "{}/share/rocprofiler-sdk".format(rocm_path)
    environment["ROCPROFILER_PC_SAMPLING_BETA_ENABLED"] = "on"

    executables = {
        "avail": "{}/bin/rocprofv3-avail".format(rocm_path),
        "rocprofv3": "{}/bin/rocprofv3".format(rocm_path),
    }

    def run(name, *arguments, check=True, cwd=None, extra_environment=None):
        child_environment = dict(environment)
        if extra_environment:
            child_environment.update(extra_environment)
        result = subprocess.run(
            [sys.executable, executables[name]] + [str(arg) for arg in arguments],
            cwd=cwd,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=60,
        )
        if check:
            assert result.returncode == 0, result.stderr
        assert "Traceback" not in result.stderr
        return result

    return run


@pytest.fixture(scope="session")
def inventory(request):
    avail = set_library(request.config.getoption("--rocm-path"))

    agent_info = avail.get_agent_info_map()
    counters = avail.get_counters()
    spm_counters = avail.get_spm_counters()
    logical_ids = {
        handle: int(info["logical_node_type_id"])
        for handle, info in agent_info.items()
        if info["type"] == 2
    }
    assert logical_ids, "No GPU agent was discovered"
    assert len(set(logical_ids.values())) == len(logical_ids)

    names = {
        logical_id: str(agent_info[handle]["name"])
        for handle, logical_id in logical_ids.items()
    }
    counter_records = {
        logical_id: tuple(counters.get(handle, ()))
        for handle, logical_id in logical_ids.items()
    }
    pmc = {
        logical_id: frozenset(counter.name for counter in records)
        for logical_id, records in counter_records.items()
    }
    assert all(pmc.values()), "No PMC counters discovered for a GPU agent"
    basic = {
        logical_id: tuple(
            counter.name
            for counter in records
            if isinstance(counter, avail.basic_counter)
        )
        for logical_id, records in counter_records.items()
    }
    collectable = {
        logical_id: tuple(
            counter.name for counter in records if not _value(counter.is_hw_constant)
        )
        for logical_id, records in counter_records.items()
    }
    handles = {
        logical_id: {
            counter.name: int(_value(counter.counter_handle))
            for counter in records
            if not _value(counter.is_hw_constant)
        }
        for logical_id, records in counter_records.items()
    }

    def is_collectable(agent_handle, counter_names):
        counter_handles = [
            handles[logical_ids[agent_handle]][name] for name in counter_names
        ]
        try:
            return avail.check_pmc({agent_handle: counter_handles})
        except SystemExit:
            return False

    # The first two counters of an agent normally share a block, while a large
    # enough batch exceeds the hardware limits. Both are confirmed against the
    # library so the tests skip rather than misreport on unusual hardware.
    compatible = None
    incompatible = None
    for agent_handle, logical_id in sorted(logical_ids.items(), key=lambda kv: kv[1]):
        candidates = collectable[logical_id]
        if len(candidates) < 2:
            continue
        if compatible is None and is_collectable(agent_handle, candidates[:2]):
            compatible = (logical_id, candidates[:2])
        batch = candidates[:MAX_COUNTER_BATCH]
        if incompatible is None and not is_collectable(agent_handle, batch):
            incompatible = (logical_id, batch)

    spm = {
        logical_id: frozenset(counter.name for counter in spm_counters.get(handle, ()))
        for handle, logical_id in logical_ids.items()
    }
    pc_configs = {
        logical_ids[handle]: tuple(
            sorted(
                (
                    config.method,
                    config.unit,
                    str(_value(config.min_interval)),
                    str(_value(config.max_interval)),
                    "interval pow2" if _value(config.flags) == 1 else "none",
                )
                for config in configs
            )
        )
        for handle, configs in avail.get_pc_sample_configs().items()
        if handle in logical_ids and configs
    }
    spm_configs = {
        logical_ids[handle]: tuple(
            sorted(
                (
                    config.type,
                    str(_value(config.sample_interval_min)),
                    str(_value(config.sample_interval_max)),
                )
                for config in configs
            )
        )
        for handle, configs in avail.get_spm_configs().items()
        if handle in logical_ids and configs
    }
    return {
        "names": names,
        "pmc": pmc,
        "spm": spm,
        "basic": basic,
        "collectable": collectable,
        "handles": handles,
        "incompatible": incompatible,
        "compatible": compatible,
        "pc-sampling": pc_configs,
        "spm-config": spm_configs,
    }


def _gpu_sections(output):
    matches = list(GPU_RE.finditer(output))
    return [
        (
            int(match.group(1)),
            output[
                match.end() : (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(output)
                )
            ],
        )
        for index, match in enumerate(matches)
    ]


def _fields(section):
    fields = defaultdict(list)
    for line in section.splitlines():
        match = FIELD_RE.match(line)
        if match:
            fields[match.group(1).lower()].append(match.group(2).strip())
    return fields


def _agent_listing(output):
    return {logical_id: _fields(section) for logical_id, section in _gpu_sections(output)}


def _listed_counters(output, counter_type):
    marker = re.compile(r"^\s*{}\s*:\s*$".format(counter_type), re.IGNORECASE)
    result = {}
    for logical_id, section in _gpu_sections(output):
        lines = section.splitlines()
        counters = set()
        marker_index = next(
            (index for index, line in enumerate(lines) if marker.match(line)), None
        )
        if marker_index is None:
            assert "No {} counters supported".format(counter_type) in section
        else:
            started = False
            for line in lines[marker_index + 1 :]:
                if not line.strip():
                    if started:
                        break
                    continue
                started = True
                counters.update(line.split())
        result[logical_id] = frozenset(counters)
    return result


def _info_counters(output):
    result = defaultdict(set)
    seen = set()
    for logical_id, section in _gpu_sections(output):
        seen.add(logical_id)
        result[logical_id].update(_fields(section).get("counter_name", ()))
    return {logical_id: frozenset(result[logical_id]) for logical_id in seen}


def _config_records(output, keys):
    result = defaultdict(list)
    for logical_id, section in _gpu_sections(output):
        fields = _fields(section)
        if not any(key in fields for key in keys):
            continue
        assert all(key in fields for key in keys)
        assert len({len(fields[key]) for key in keys}) == 1
        result[logical_id].extend(zip(*(fields[key] for key in keys)))
    return {logical_id: tuple(sorted(records)) for logical_id, records in result.items()}


def _assert_combined_availability(output, inventory):
    assert _info_counters(output) == inventory["pmc"]
    assert _config_records(output, PC_CONFIG_KEYS) == inventory["pc-sampling"]
    assert _config_records(output, SPM_CONFIG_KEYS) == inventory["spm-config"]


def _select_counter(inventory, counter_type):
    for logical_id, counters in sorted(inventory[counter_type].items()):
        if counters:
            return logical_id, sorted(counters)[0]
    pytest.skip("{} counters are unsupported on the available GPUs".format(counter_type))


def _select_collectable_counter(inventory):
    for counter_type in ("basic", "collectable"):
        for logical_id, counters in sorted(inventory[counter_type].items()):
            if counters:
                return logical_id, counters[0]
    pytest.skip("No collectable PMC counter is available")


def _pmc_check_records(output):
    records = {}
    for line in output.splitlines():
        match = PMC_CHECK_RE.match(line)
        if match:
            records[int(match.group(1))] = tuple(match.group(2).split())
    return records


def test_rocprofv3_avail_agent_views_and_device_scope(cli, inventory):
    agent_output = cli("avail", "list", "--agent").stdout
    agents = _agent_listing(agent_output)
    assert set(agents) == set(inventory["names"])
    assert not re.search(r"(?im)^\s*PMC\s*:", agent_output)
    for logical_id, fields in agents.items():
        assert AGENT_KEYS.issubset(fields)
        assert fields["name"] == [inventory["names"][logical_id]]
        assert fields["logical_node_type_id"] == [str(logical_id)]

    logical_id = min(inventory["names"])
    for arguments in (("list", "--agent"), ("info",)):
        scoped = _agent_listing(cli("avail", "-d", logical_id, *arguments).stdout)
        assert set(scoped) == {logical_id}
        assert scoped[logical_id]["name"] == [inventory["names"][logical_id]]

    bare_device = cli("avail", "-d", logical_id, "list").stdout
    assert _listed_counters(bare_device, "pmc") == {
        logical_id: inventory["pmc"][logical_id]
    }


@pytest.mark.parametrize("counter_type", ("pmc", "spm"))
def test_rocprofv3_avail_counter_views_and_filters(cli, inventory, counter_type):
    if not any(inventory[counter_type].values()):
        pytest.skip(
            "{} counters are unsupported on the available GPUs".format(counter_type)
        )

    listed = cli("avail", "list", "--" + counter_type).stdout
    assert _listed_counters(listed, counter_type) == inventory[counter_type]
    if counter_type == "pmc":
        assert cli("avail", "list").stdout == listed

    detailed = cli("avail", "info", "--" + counter_type).stdout
    assert _info_counters(detailed) == inventory[counter_type]

    logical_id, counter = _select_counter(inventory, counter_type)
    scoped = cli("avail", "-d", logical_id, "list", "--" + counter_type).stdout
    assert _listed_counters(scoped, counter_type) == {
        logical_id: inventory[counter_type][logical_id]
    }

    filtered = cli("avail", "-d", logical_id, "info", "--" + counter_type, counter).stdout
    assert _info_counters(filtered) == {logical_id: frozenset((counter,))}


def test_rocprofv3_avail_list_boolean_false(cli):
    assert cli("avail", "list", "--pmc", "false").stdout.strip() == ""


@pytest.mark.parametrize(
    "capability,keys",
    (("pc-sampling", PC_CONFIG_KEYS), ("spm-config", SPM_CONFIG_KEYS)),
)
def test_rocprofv3_avail_capability_views(cli, inventory, capability, keys):
    expected = inventory[capability]
    if not expected:
        pytest.skip("{} is unsupported on the available GPUs".format(capability))

    listed = _agent_listing(cli("avail", "list", "--" + capability).stdout)
    assert set(listed) == set(expected)
    for logical_id, fields in listed.items():
        assert fields["name"] == [inventory["names"][logical_id]]

    detailed = cli("avail", "info", "--" + capability).stdout
    assert _config_records(detailed, keys) == expected


@pytest.mark.parametrize("selection", ("device-option", "qualifier", "multi-counter"))
def test_rocprofv3_avail_pmc_check_device_selection(cli, inventory, selection):
    logical_id, counter = _select_collectable_counter(inventory)
    if selection == "device-option":
        arguments = ("-d", logical_id, "pmc-check", counter)
        expected = (counter,)
    elif selection == "qualifier":
        arguments = ("pmc-check", "{}:device={}".format(counter, logical_id))
        expected = (counter,)
    else:
        if not inventory["compatible"]:
            pytest.skip("No multi-counter compatible set was discovered")
        logical_id, expected = inventory["compatible"]
        arguments = ("-d", logical_id, "pmc-check") + expected

    output = cli("avail", *arguments).stdout
    assert _pmc_check_records(output) == {logical_id: expected}


def test_rocprofv3_avail_pmc_check_all_agents(cli, inventory):
    logical_ids = sorted(inventory["collectable"])
    common = set(inventory["collectable"][logical_ids[0]])
    for logical_id in logical_ids[1:]:
        common.intersection_update(inventory["collectable"][logical_id])
    if not common:
        pytest.skip("No collectable PMC counter is common to every GPU")

    counter = next(name for name in inventory["basic"][logical_ids[0]] if name in common)
    output = cli("avail", "pmc-check", counter).stdout
    assert _pmc_check_records(output) == {
        logical_id: (counter,) for logical_id in logical_ids
    }


NEGATIVE_CASES = (
    pytest.param(("pmc-check",), "Provide counter to check", id="missing-counter"),
    pytest.param(
        ("pmc-check", "{invalid_counter}"),
        "Invalid counter name",
        id="invalid-counter",
    ),
    pytest.param(
        ("-d", "{invalid_device}", "pmc-check", "{counter}"),
        "Invalid device id",
        id="invalid-device",
    ),
    pytest.param(
        ("pmc-check", "{counter}:agent={device}"),
        "Incorrect input format for device index",
        id="malformed-qualifier",
    ),
    pytest.param(
        ("pmc-check", "{counter}:device={device}:extra"),
        "Invalid format for pmc-check",
        id="excess-colons",
    ),
)


@pytest.mark.parametrize("arguments,error", NEGATIVE_CASES)
def test_rocprofv3_avail_pmc_check_rejects_bad_input(cli, inventory, arguments, error):
    logical_id, counter = _select_collectable_counter(inventory)
    invalid_counter = "__rocprofv3_invalid_counter__"
    all_counters = set().union(*inventory["pmc"].values())
    while invalid_counter in all_counters:
        invalid_counter += "_"
    values = {
        "counter": counter,
        "device": logical_id,
        "invalid_counter": invalid_counter,
        "invalid_device": max(inventory["names"]) + 1,
    }
    resolved = tuple(argument.format(**values) for argument in arguments)
    result = cli("avail", *resolved, check=False)

    assert result.returncode != 0
    assert error in result.stderr
    assert "Following input counters can be collected together" not in result.stdout


def test_rocprofv3_avail_pmc_check_rejects_incompatible_set(cli, inventory):
    if not inventory["incompatible"]:
        pytest.skip("No incompatible hardware counter set was discovered")
    logical_id, counters = inventory["incompatible"]

    result = cli("avail", "-d", logical_id, "pmc-check", *counters, check=False)

    assert result.returncode != 0, "CLI accepted a set the library rejected"
    assert "not collected on agent" in result.stderr
    assert "can be collected together" not in result.stdout


def test_rocprofv3_avail_global_cli_contracts(cli, inventory):
    help_result = cli("avail")
    assert "usage:" in help_result.stdout.lower()
    for command in ("list", "info", "pmc-check"):
        assert command in help_result.stdout

    bad_command = cli("avail", "not-a-command", check=False)
    assert bad_command.returncode == 2
    assert "invalid choice" in bad_command.stderr

    out_of_range = max(inventory["names"]) + 1
    assert cli("avail", "-d", out_of_range, "list").stdout.strip() == ""


def test_rocprofv3_list_avail_aliases_are_equivalent(cli, inventory):
    short_output = cli("rocprofv3", "-L").stdout
    assert short_output == cli("rocprofv3", "--list-avail").stdout
    _assert_combined_availability(short_output, inventory)


def _list_avail_file(output_directory):
    files = sorted(output_directory.rglob("*_list_avail.txt"))
    assert len(files) == 1
    return files[0].read_text(encoding="utf-8")


def test_rocprofv3_list_avail_output_file(cli, inventory, tmp_path):
    output_directory = tmp_path / "file-output"
    result = cli(
        "rocprofv3", "-d", output_directory, "-o", "metrics", "--list-avail", cwd=tmp_path
    )

    assert result.stdout == ""
    _assert_combined_availability(_list_avail_file(output_directory), inventory)


def test_rocprofv3_list_avail_inherited_output_variables(cli, inventory, tmp_path):
    inherited = tmp_path / "inherited"
    result = cli(
        "rocprofv3",
        "--list-avail",
        cwd=tmp_path,
        extra_environment={
            "ROCPROF_OUTPUT_PATH": str(inherited),
            "ROCPROF_OUTPUT_FILE_NAME": "metrics",
        },
    )

    # -o/-d default to these variables, but a caller who only exported them
    # still expects the listing on stdout
    _assert_combined_availability(result.stdout, inventory)
    assert sorted(tmp_path.rglob("*_list_avail.txt")) == []


def test_rocprofv3_list_avail_explicit_file_with_inherited_path(cli, inventory, tmp_path):
    inherited = tmp_path / "inherited"
    result = cli(
        "rocprofv3",
        "-o",
        "metrics",
        "--list-avail",
        cwd=tmp_path,
        extra_environment={"ROCPROF_OUTPUT_PATH": str(inherited)},
    )

    # an explicit -o diverts the listing, and the inherited path still decides
    # where it lands, so it stays with the other artifacts of the same run
    assert result.stdout == ""
    _assert_combined_availability(_list_avail_file(inherited), inventory)


def test_rocprofv3_list_avail_with_trace(cli, inventory, request, tmp_path):
    transpose = "{}/bin/transpose".format(request.config.getoption("--rocm-path"))
    if not os.path.isfile(transpose):
        pytest.skip("transpose workload is unavailable")

    output_directory = tmp_path / "trace-output"
    result = cli(
        "rocprofv3",
        "-d",
        output_directory,
        "-o",
        "metrics",
        "--list-avail",
        "--sys-trace",
        "--log-level",
        "warning",
        "--",
        transpose,
        cwd=tmp_path,
    )

    assert not _gpu_sections(result.stdout)
    _assert_combined_availability(_list_avail_file(output_directory), inventory)

    databases = sorted(output_directory.rglob("*.db"))
    assert len(databases) == 1
    assert databases[0].stat().st_size > 0


def test_rocprofv3_list_avail_rejects_non_directory_output(cli, tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    message = "Could not resolve the list-avail output path"

    direct = cli(
        "avail",
        "--output-directory",
        blocker,
        "list",
        "--pmc",
        check=False,
        cwd=tmp_path,
    )
    launched = cli("rocprofv3", "-d", blocker, "--list-avail", check=False, cwd=tmp_path)
    # the file name carries the subdirectory, so the blocker is in the middle of
    # the resolved path rather than at its root
    nested = cli(
        "rocprofv3",
        "-d",
        tmp_path,
        "-o",
        "{}/name".format(blocker.name),
        "--list-avail",
        check=False,
        cwd=tmp_path,
    )

    # every offending directory is diagnosed rather than reaching the abort
    for result in (direct, launched, nested):
        assert result.returncode == 1
        assert message in result.stderr
    assert blocker.read_text(encoding="utf-8") == ""


def test_rocprofv3_list_avail_rejects_non_directory_placeholder(cli, tmp_path):
    (tmp_path / socket.gethostname()).write_text("", encoding="utf-8")

    result = cli(
        "avail",
        "--output-directory",
        "%hostname%",
        "list",
        "--pmc",
        check=False,
        cwd=tmp_path,
    )

    # only the library can expand the placeholder, so it is also the only place
    # that can tell the resolved path is unusable
    assert result.returncode == 1
    assert "Could not resolve the list-avail output path" in result.stderr


def test_rocprofv3_list_avail_echo_runs_nothing(cli, tmp_path):
    result = cli(
        "rocprofv3",
        "-d",
        tmp_path / "echo-output",
        "--list-avail",
        "--echo",
        "--",
        sys.executable,
        "-c",
        "pass",
        cwd=tmp_path,
    )

    echoed = [line for line in result.stdout.splitlines() if line.startswith("command:")]
    assert len(echoed) == 2
    assert any("rocprofv3-avail" in line for line in echoed)
    assert not _gpu_sections(result.stdout)
    assert sorted(tmp_path.rglob("*_list_avail.txt*")) == []


def test_rocprofv3_list_avail_with_application_and_false(cli, inventory):
    sentinel = "ROCPROFV3_LIST_AVAIL_APPLICATION_RAN"
    application = [sys.executable, "-c", "print({!r})".format(sentinel)]

    listed = cli("rocprofv3", "--list-avail", "--", *application)
    assert sentinel in listed.stdout
    _assert_combined_availability(listed.stdout, inventory)

    disabled = cli("rocprofv3", "-L", "false", "--", *application)
    assert sentinel in disabled.stdout
    assert not _gpu_sections(disabled.stdout)
    assert "Counter_Name" not in disabled.stdout


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
