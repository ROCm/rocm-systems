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

import ctypes
import json
import os
import re
import signal
import socket
import sqlite3
import subprocess
import sys
from collections import defaultdict, namedtuple
from pathlib import Path

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
PC_CONFIG_KEYS = ("method", "unit", "min_interval", "max_interval", "flags")
SPM_CONFIG_KEYS = ("type", "minimum_interval", "maximum_interval")
SUCCESS_PREFIX = "Following input counters can be collected together on GPU"
GPU_RE = re.compile(r"(?m)^[ \t]*GPU[ \t]*:[ \t]*(\d+)[ \t]*$")
FIELD_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")
SANITIZED_ENVIRONMENT_VARIABLES = (
    "MPI_LOCALNRANKS",
    "MPI_LOCALRANKID",
    "MPI_NRANKS",
    "MPI_RANK",
    "MPI_RANKID",
    "MPI_SIZE",
    "MV2_COMM_WORLD_RANK",
    "MV2_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "PBS_NODENUM",
    "PBS_O_TASKNUM",
    "PMI_RANK",
    "PMI_SIZE",
    "ROCPROFILER_CI",
    "ROCPROF_MPI_RANKS",
    "ROCPROF_MPI_RANK_VAR",
    "ROCPROF_MPI_SIZE_VAR",
    "SLURM_JOB_ID",
    "SLURM_NTASKS",
    "SLURM_PROCID",
)

Device = namedtuple(
    "Device",
    (
        "handle",
        "logical_id",
        "name",
        "counters",
        "basic_counters",
        "collectable_counters",
        "largest_block_counters",
        "spm_counters",
    ),
)

# A single hardware block exposes more countable events than it has physical
# counters, so this many counters from one block cannot share a pass.
INCOMPATIBLE_SET_SIZE = 24


class Inventory:
    def __init__(self, devices, pc_configs, spm_configs):
        self.devices = devices
        self.pc_configs = pc_configs
        self.spm_configs = spm_configs

    @property
    def pmc_by_device(self):
        return {device.logical_id: device.counters for device in self.devices}

    @property
    def spm_by_device(self):
        return {device.logical_id: device.spm_counters for device in self.devices}

    @property
    def counter_device(self):
        for device in self.devices:
            if device.basic_counters:
                return device
        for device in self.devices:
            if device.collectable_counters:
                return device
        return None

    @property
    def common_counter(self):
        common = set(self.devices[0].collectable_counters)
        for device in self.devices[1:]:
            common.intersection_update(device.collectable_counters)
        for counters in (
            *(device.basic_counters for device in self.devices),
            self.devices[0].collectable_counters,
        ):
            for counter in counters:
                if counter in common:
                    return counter
        return None

    @staticmethod
    def preferred_counter(device):
        counters = device.basic_counters or device.collectable_counters
        return counters[0] if counters else None


class Commands:
    def __init__(self, avail, rocprofv3, environment):
        self.avail = avail
        self.rocprofv3 = rocprofv3
        self.environment = environment

    def _run(self, executable, *arguments, check=True, cwd=None, environment=None):
        command = [sys.executable, str(executable)]
        command.extend(str(argument) for argument in arguments)
        child_environment = self.environment.copy()
        if environment:
            child_environment.update(environment)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            universal_newlines=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                process.kill()
            process.communicate()
            raise
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        result.pid = process.pid
        if check:
            assert result.returncode == 0, (
                f"{' '.join(command)} exited {result.returncode}\n"
                f"stderr:\n{result.stderr}\nstdout tail:\n{result.stdout[-2000:]}"
            )
            assert "Traceback" not in result.stderr
        return result

    def run_avail(self, *arguments, **kwargs):
        return self._run(self.avail, *arguments, **kwargs)

    def run_rocprofv3(self, *arguments, **kwargs):
        return self._run(self.rocprofv3, *arguments, **kwargs)


def _with_paths(environment, key, *paths):
    values = [str(path) for path in paths]
    if environment.get(key):
        values.append(environment[key])
    environment[key] = os.pathsep.join(values)


@pytest.fixture(scope="session")
def commands(request):
    root = Path(request.config.getoption("--rocm-path"))
    environment = os.environ.copy()
    for key in SANITIZED_ENVIRONMENT_VARIABLES + (
        "ROCPROF_OUTPUT_FILE_NAME",
        "ROCPROF_OUTPUT_LIST_AVAIL_FILE",
        "ROCPROF_OUTPUT_PATH",
    ):
        environment.pop(key, None)
    environment["ROCPROFILER_METRICS_PATH"] = str(root / "share" / "rocprofiler-sdk")
    environment["ROCPROF_LIST_AVAIL_TOOL_LIBRARY"] = str(
        root / "lib" / "rocprofiler-sdk" / "librocprofv3-list-avail.so"
    )
    environment["ROCPROFILER_PC_SAMPLING_BETA_ENABLED"] = "on"
    _with_paths(
        environment,
        "LD_LIBRARY_PATH",
        root / "lib",
        root / "lib" / "rocprofiler-sdk",
    )
    _with_paths(
        environment,
        "PYTHONPATH",
        root / "lib" / f"python{sys.version_info.major}" / "site-packages",
    )

    avail = Path(request.config.getoption("--rocprofv3-avail"))
    rocprofv3 = Path(request.config.getoption("--rocprofv3"))
    assert avail.is_file(), f"rocprofv3-avail not found: {avail}"
    assert rocprofv3.is_file(), f"rocprofv3 not found: {rocprofv3}"
    return Commands(avail, rocprofv3, environment)


def _value(item):
    return getattr(item, "value", item)


@pytest.fixture(scope="session")
def inventory(request):
    root = Path(request.config.getoption("--rocm-path"))
    package_path = root / "lib" / f"python{sys.version_info.major}" / "site-packages"
    sys.path.insert(0, str(package_path))
    os.environ["ROCPROFILER_PC_SAMPLING_BETA_ENABLED"] = "on"
    os.environ["ROCPROFILER_METRICS_PATH"] = str(root / "share" / "rocprofiler-sdk")

    from rocprofv3 import avail

    avail.loadLibrary.libname = str(
        root / "lib" / "rocprofiler-sdk" / "librocprofv3-list-avail.so"
    )
    agent_info = avail.get_agent_info_map()
    counters = avail.get_counters()
    spm_counters = avail.get_spm_counters()

    devices = []
    for handle, info in agent_info.items():
        if info["type"] != 2:
            continue
        agent_counters = counters.get(handle, ())
        by_block = defaultdict(list)
        for counter in agent_counters:
            if isinstance(counter, avail.basic_counter) and not _value(
                counter.is_hw_constant
            ):
                by_block[str(counter.block)].append(counter.name)
        devices.append(
            Device(
                handle=handle,
                logical_id=int(info["logical_node_type_id"]),
                name=str(info["name"]),
                counters=frozenset(counter.name for counter in agent_counters),
                basic_counters=tuple(
                    counter.name
                    for counter in agent_counters
                    if isinstance(counter, avail.basic_counter)
                ),
                collectable_counters=tuple(
                    counter.name
                    for counter in agent_counters
                    if not _value(counter.is_hw_constant)
                ),
                largest_block_counters=tuple(
                    sorted(max(by_block.values(), key=len)) if by_block else ()
                ),
                spm_counters=frozenset(
                    counter.name for counter in spm_counters.get(handle, ())
                ),
            )
        )

    assert devices, "No GPU agents discovered"
    logical_ids = [device.logical_id for device in devices]
    assert len(logical_ids) == len(set(logical_ids)), "Duplicate GPU logical IDs"
    handle_to_device = {device.handle: device.logical_id for device in devices}

    pc_configs = {}
    for handle, configs in avail.get_pc_sample_configs().items():
        if handle not in handle_to_device or not configs:
            continue
        pc_configs[handle_to_device[handle]] = tuple(
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

    spm_configs = {}
    for handle, configs in avail.get_spm_configs().items():
        if handle not in handle_to_device or not configs:
            continue
        spm_configs[handle_to_device[handle]] = tuple(
            sorted(
                (
                    config.type,
                    str(_value(config.sample_interval_min)),
                    str(_value(config.sample_interval_max)),
                )
                for config in configs
            )
        )

    return Inventory(tuple(devices), pc_configs, spm_configs)


def _gpu_sections(output):
    matches = list(GPU_RE.finditer(output))
    return [
        (
            int(match.group(1)),
            (
                output[match.end() : matches[index + 1].start()]
                if index + 1 < len(matches)
                else output[match.end() :]
            ),
        )
        for index, match in enumerate(matches)
    ]


def _fields(section):
    values = defaultdict(list)
    for line in section.splitlines():
        match = FIELD_RE.match(line)
        if match:
            values[match.group(1).lower()].append(match.group(2).strip())
    return values


def _listed_counters(output, expected):
    universe = set().union(*expected.values())
    result = {}
    for logical_id, section in _gpu_sections(output):
        assert logical_id not in result
        tokens = {token.strip(",:") for token in section.split()}
        result[logical_id] = frozenset(tokens.intersection(universe))
    return result


def _info_counters(output):
    result = defaultdict(set)
    seen = set()
    for logical_id, section in _gpu_sections(output):
        seen.add(logical_id)
        result[logical_id].update(_fields(section).get("counter_name", []))
    return {logical_id: frozenset(result[logical_id]) for logical_id in seen}


def _config_records(output, keys):
    result = defaultdict(list)
    for logical_id, section in _gpu_sections(output):
        fields = _fields(section)
        present = [key for key in keys if key in fields]
        if not present:
            continue
        assert len(present) == len(keys)
        lengths = {len(fields[key]) for key in keys}
        assert len(lengths) == 1
        result[logical_id].extend(zip(*(fields[key] for key in keys)))
    return {logical_id: tuple(sorted(records)) for logical_id, records in result.items()}


def _agent_listing(output):
    result = {}
    for logical_id, section in _gpu_sections(output):
        assert logical_id not in result
        result[logical_id] = _fields(section)
    return result


def _assert_success(result, expected):
    rows = []
    pattern = re.compile(
        rf"(?m)^{re.escape(SUCCESS_PREFIX)}[ \t]*:[ \t]*(\d+)[ \t]*(.*?)[ \t]*$"
    )
    for match in pattern.finditer(result.stdout):
        rows.append((int(match.group(1)), set(match.group(2).split())))
    assert len(rows) == len(expected)
    assert {logical_id for logical_id, _ in rows} == set(expected)
    for logical_id, names in rows:
        assert expected[logical_id] in names


def _assert_availability(output, inventory):
    assert _info_counters(output) == inventory.pmc_by_device
    assert _config_records(output, PC_CONFIG_KEYS) == inventory.pc_configs
    assert _config_records(output, SPM_CONFIG_KEYS) == inventory.spm_configs


def _assert_routed_output(result, output_file, inventory):
    assert result.stdout == ""
    assert output_file.is_file()
    assert output_file.stat().st_size > 0
    _assert_availability(
        output_file.read_text(encoding="utf-8", errors="replace"), inventory
    )


def _require_counter(inventory):
    device = inventory.counter_device
    if device is None:
        pytest.skip("No GPU exposes a collectable non-constant counter")
    return device, inventory.preferred_counter(device)


def test_default_list_matches_explicit_pmc(commands, inventory):
    default = commands.run_avail("list")
    explicit = commands.run_avail("list", "--pmc")
    expected = inventory.pmc_by_device
    default_counters = _listed_counters(default.stdout, expected)
    explicit_counters = _listed_counters(explicit.stdout, expected)
    assert default_counters == explicit_counters == expected


def test_list_agent_fields_without_pmc(commands, inventory):
    result = commands.run_avail("list", "--agent")
    agents = _agent_listing(result.stdout)
    assert set(agents) == set(inventory.pmc_by_device)
    names = {device.logical_id: device.name for device in inventory.devices}
    for logical_id, fields in agents.items():
        assert AGENT_KEYS.issubset(fields)
        assert fields["logical_node_type_id"] == [str(logical_id)]
        assert fields["name"] == [names[logical_id]]
        assert "pmc" not in fields
    assert not re.search(r"(?mi)^\s*PMC\s*:", result.stdout)


@pytest.mark.parametrize("view", ("list", "info"))
def test_device_scoping(commands, inventory, view):
    device = inventory.devices[0]
    result = commands.run_avail("-d", device.logical_id, view, "--pmc")
    if view == "list":
        actual = _listed_counters(result.stdout, {device.logical_id: device.counters})
    else:
        actual = _info_counters(result.stdout)
    assert actual == {device.logical_id: device.counters}


def test_info_pmc_name_filter(commands, inventory):
    device, counter = _require_counter(inventory)
    result = commands.run_avail("-d", device.logical_id, "info", "--pmc", counter)
    assert _info_counters(result.stdout) == {device.logical_id: frozenset({counter})}
    sections = _gpu_sections(result.stdout)
    assert len(sections) == 1
    descriptions = _fields(sections[0][1])["description"]
    assert len(descriptions) == 1 and descriptions[0]


@pytest.mark.parametrize("capability", ("spm", "pc-sampling", "spm-config"))
def test_supported_capability_views(commands, inventory, capability):
    if capability == "spm":
        expected = inventory.spm_by_device
        if not any(expected.values()):
            pytest.skip("SPM counters are unsupported on the available GPUs")
        listed = commands.run_avail("list", "--spm")
        detailed = commands.run_avail("info", "--spm")
        assert _listed_counters(listed.stdout, expected) == expected
        assert _info_counters(detailed.stdout) == expected

        device = next(item for item in inventory.devices if item.spm_counters)
        counter = sorted(device.spm_counters)[0]
        filtered = commands.run_avail("-d", device.logical_id, "info", "--spm", counter)
        assert _info_counters(filtered.stdout) == {
            device.logical_id: frozenset({counter})
        }
        return

    if capability == "pc-sampling":
        expected = inventory.pc_configs
        keys = PC_CONFIG_KEYS
    else:
        expected = inventory.spm_configs
        keys = SPM_CONFIG_KEYS
    if not expected:
        pytest.skip(f"{capability} is unsupported on the available GPUs")

    listed = commands.run_avail("list", f"--{capability}")
    agents = _agent_listing(listed.stdout)
    assert set(agents) == set(expected)
    names = {device.logical_id: device.name for device in inventory.devices}
    for logical_id, fields in agents.items():
        assert fields["name"] == [names[logical_id]]

    detailed = commands.run_avail("info", f"--{capability}")
    assert _config_records(detailed.stdout, keys) == expected


@pytest.mark.parametrize("selection", ("global-device", "qualifier"))
def test_pmc_check_device_selection(commands, inventory, selection):
    device, counter = _require_counter(inventory)
    if selection == "global-device":
        arguments = ("-d", device.logical_id, "pmc-check", counter)
    else:
        arguments = ("pmc-check", f"{counter}:device={device.logical_id}")
    result = commands.run_avail(*arguments)
    _assert_success(result, {device.logical_id: counter})


def test_pmc_check_default_all_agents(commands, inventory):
    counter = inventory.common_counter
    if counter is None:
        pytest.skip("No collectable counter is common to every GPU")
    result = commands.run_avail("pmc-check", counter)
    _assert_success(result, {device.logical_id: counter for device in inventory.devices})


NEGATIVE_CASES = (
    ("missing-counter", ("pmc-check",), "Provide counter to check"),
    (
        "invalid-counter",
        ("pmc-check", "{invalid_counter}"),
        "Invalid counter name",
    ),
    (
        "invalid-device",
        ("-d", "{invalid_device}", "pmc-check", "{counter}"),
        "Invalid device id",
    ),
    (
        "malformed-qualifier",
        ("pmc-check", "{counter}:agent={device}"),
        "Incorrect input format for device index",
    ),
    (
        "excess-colons",
        ("pmc-check", "{counter}:device={device}:extra"),
        "Invalid format for pmc-check",
    ),
)


@pytest.mark.parametrize(
    "_name,arguments,error",
    NEGATIVE_CASES,
    ids=[case[0] for case in NEGATIVE_CASES],
)
def test_pmc_check_rejects_bad_input(commands, inventory, _name, arguments, error):
    all_names = set().union(*(item.counters for item in inventory.devices))
    invalid_counter = "__rocprofv3_invalid_counter__"
    while invalid_counter in all_names:
        invalid_counter += "_"
    values = {
        "invalid_counter": invalid_counter,
        "invalid_device": max(item.logical_id for item in inventory.devices) + 1,
    }
    if any("{counter}" in argument for argument in arguments):
        device, counter = _require_counter(inventory)
        values.update({"counter": counter, "device": device.logical_id})
    resolved = tuple(argument.format_map(values) for argument in arguments)
    result = commands.run_avail(*resolved, check=False)
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert error in result.stderr
    assert SUCCESS_PREFIX not in combined
    assert "Traceback" not in combined


def test_pmc_check_rejects_incompatible_set(commands, inventory):
    device = next(
        (
            item
            for item in inventory.devices
            if len(item.largest_block_counters) >= INCOMPATIBLE_SET_SIZE
        ),
        None,
    )
    if device is None:
        pytest.skip("No GPU block exposes enough counters to exceed its pass limit")
    counters = device.largest_block_counters[:INCOMPATIBLE_SET_SIZE]
    result = commands.run_avail(
        "-d", device.logical_id, "pmc-check", *counters, check=False
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "not collected on agent" in result.stderr
    assert SUCCESS_PREFIX not in combined
    assert "Traceback" not in combined


@pytest.mark.parametrize("scope", ("disabled-pmc", "unknown-device"))
def test_avail_list_reports_nothing_outside_scope(commands, inventory, scope):
    if scope == "disabled-pmc":
        arguments = ("list", "--pmc", "false")
    else:
        unknown_device = max(item.logical_id for item in inventory.devices) + 1
        arguments = ("-d", unknown_device, "list")
    result = commands.run_avail(*arguments)
    assert _gpu_sections(result.stdout) == []


def test_help_and_bad_subcommand(commands):
    help_result = commands.run_avail()
    assert help_result.returncode == 0
    assert "usage:" in help_result.stdout.lower()
    for command in ("list", "info", "pmc-check"):
        assert command in help_result.stdout

    bad_result = commands.run_avail("not-a-command", check=False)
    assert bad_result.returncode == 2
    assert "usage:" in bad_result.stderr.lower()
    assert "invalid choice" in bad_result.stderr
    assert "Traceback" not in bad_result.stderr


def test_output_formatter_error_contract(commands):
    library = ctypes.CDLL(commands.environment["ROCPROF_LIST_AVAIL_TOOL_LIBRARY"])
    library.format_output_path.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    library.format_output_path.restype = ctypes.c_int

    output = ctypes.c_char_p(os.fsencode("sentinel"))
    assert library.format_output_path(None, None, ctypes.byref(output)) == 0
    assert output.value is None
    assert library.format_output_path(os.fsencode("path"), None, None) == 0


@pytest.mark.parametrize("flag", ("--list-avail", "-L"))
def test_rocprofv3_list_avail_stdout(commands, inventory, flag):
    result = commands.run_rocprofv3(flag)
    _assert_availability(result.stdout, inventory)


def test_rocprofv3_list_avail_false_runs_application(commands, tmp_path):
    result = commands.run_rocprofv3("-L", "false", "--", "/bin/true", cwd=tmp_path)
    assert _info_counters(result.stdout) == {}
    assert not list(tmp_path.rglob("*_list_avail.txt"))


@pytest.mark.parametrize("routing", ("directory-only", "file-only", "placeholder-app"))
def test_rocprofv3_list_avail_output_routing(commands, inventory, tmp_path, routing):
    cwd = tmp_path / routing
    cwd.mkdir()
    environment = {}
    if routing == "directory-only":
        output_directory = cwd / "output"
        arguments = ("-d", output_directory, "--list-avail")
    elif routing == "file-only":
        arguments = ("-o", "availability", "--list-avail")
    else:
        output_directory = (
            cwd / "%hostname%" / "%env{AIROCVAL_ROUTE}%" / "%rank%-%size%-%nid%"
        )
        environment.update(
            {
                "AIROCVAL_RANK": "3",
                "AIROCVAL_ROUTE": "route-token",
                "AIROCVAL_SIZE": "8",
            }
        )
        arguments = (
            "-d",
            output_directory,
            "-o",
            "%argt%_%pid%",
            "--mpi-world-rank-variable",
            "AIROCVAL_RANK",
            "--mpi-world-size-variable",
            "AIROCVAL_SIZE",
            "--list-avail",
            "--",
            "/bin/true",
            "--",
            "payload",
        )

    result = commands.run_rocprofv3(*arguments, cwd=cwd, environment=environment)
    if routing == "directory-only":
        output_file = (
            output_directory / socket.gethostname() / f"{result.pid}_list_avail.txt"
        )
    elif routing == "file-only":
        output_file = cwd / "availability_list_avail.txt"
    else:
        output_file = (
            cwd
            / socket.gethostname()
            / "route-token"
            / "3-8-3"
            / f"truepayload_{result.pid}_list_avail.txt"
        )

    assert sorted(cwd.rglob("*_list_avail.txt")) == [output_file]
    _assert_routed_output(result, output_file, inventory)


def test_rocprofv3_list_avail_with_tracing(commands, inventory, request, tmp_path):
    application = request.config.getoption("--transpose")
    if not application or not Path(application).is_file():
        pytest.skip("transpose application is unavailable")
    output_directory = tmp_path / "traced"
    commands.run_rocprofv3(
        "--list-avail",
        "--sys-trace",
        "-d",
        output_directory,
        "-o",
        "metrics",
        "--",
        application,
        cwd=tmp_path,
    )

    availability = output_directory / "metrics_list_avail.txt"
    _assert_availability(
        availability.read_text(encoding="utf-8", errors="replace"), inventory
    )

    trace = output_directory / "metrics_results.db"
    assert trace.is_file()
    connection = sqlite3.connect(f"file:{trace}?mode=ro", uri=True)
    try:
        counts = [
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("rocpd_kernel_dispatch", "rocpd_region")
        ]
    finally:
        connection.close()
    assert all(count > 0 for count in counts), f"empty trace tables: {counts}"


def test_rocprofv3_list_avail_input_file(commands, inventory, tmp_path):
    output_directory = tmp_path / "input-output"
    config = tmp_path / "input.json"
    config.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "list_avail": True,
                        "output_directory": str(output_directory),
                        "output_file": "from-input",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = commands.run_rocprofv3("-i", config, cwd=tmp_path)
    output_file = output_directory / "from-input_list_avail.txt"
    assert sorted(tmp_path.rglob("*_list_avail.txt")) == [output_file]
    _assert_routed_output(result, output_file, inventory)


def test_rocprofv3_list_avail_multipass_routing(commands, inventory, tmp_path):
    _, counter = _require_counter(inventory)
    output_directory = tmp_path / "multipass"
    result = commands.run_rocprofv3(
        "--list-avail",
        "-d",
        output_directory,
        "-o",
        "availability",
        "--pmc",
        counter,
        "--pmc",
        counter,
        cwd=tmp_path,
    )
    output_files = [
        output_directory / f"pass_{index}" / "availability_list_avail.txt"
        for index in (1, 2)
    ]
    assert sorted(tmp_path.rglob("*_list_avail.txt")) == output_files
    assert result.stdout == ""
    for output_file in output_files:
        assert output_file.is_file()
        _assert_availability(
            output_file.read_text(encoding="utf-8", errors="replace"), inventory
        )


def test_rocprofv3_nested_failure_is_clean(commands, tmp_path):
    missing_library = tmp_path / "missing-list-avail.so"
    result = commands.run_rocprofv3(
        "--list-avail",
        check=False,
        environment={"ROCPROF_LIST_AVAIL_TOOL_LIBRARY": str(missing_library)},
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "rocprofv3-avail exited with status 1" in result.stderr
    assert "Traceback" not in combined


def test_rocprofv3_incompatible_list_avail_library_is_clean(commands):
    list_avail_library = Path(commands.environment["ROCPROF_LIST_AVAIL_TOOL_LIBRARY"])
    incompatible_library = list_avail_library.parent.parent / "librocprofiler-sdk.so"
    assert incompatible_library.is_file()

    environment = {"ROCPROF_LIST_AVAIL_TOOL_LIBRARY": str(incompatible_library)}

    nested_result = commands.run_rocprofv3(
        "--list-avail",
        check=False,
        environment=environment,
    )
    nested_output = nested_result.stdout + nested_result.stderr
    assert nested_result.returncode == 1
    assert "rocprofv3-avail exited with status 1" in nested_result.stderr
    assert "Traceback" not in nested_output
