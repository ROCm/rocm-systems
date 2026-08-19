# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Set, Tuple

import common
import pandas as pd
import pytest

from pc_sampling.pc_sampling_analysis import load_pc_sample_records
from tests.integration import common as integration_common
from utils.file_io import load_pc_sampling_results

config = {}
config["app_1"] = ["./tests/vcopy", "-n", "1048576", "-b", "256", "-i", "3"]
config["app_mat_mul_max"] = ["./tests/mat_mul_max"]
# The update kernel is ~100x cheaper per round than SpMV; at the default 400
# rounds it finishes inside one sampling interval and is sometimes missed
# entirely. 5000 rounds spans roughly nine intervals, so every child is sampled.
CG_ROUNDS = "5000"
config["app_conjugate_gradient"] = [
    "./tests/conjugate_gradient",
    "--processes",
    "3",
    "--kernels",
    "spmv,spmv,update",
    "--rounds",
    CG_ROUNDS,
    "--rotate-code-objects",
]
config["cleanup"] = True
config["COUNTER_LOGGING"] = False
config["METRIC_COMPARE"] = False

num_devices = 1

CG_SPMV_KERNEL_NAME = "kernel_spmv_csr"
CG_UPDATE_KERNEL_NAME = "kernel_cg_update_reduce"
CG_KERNEL_NAMES = frozenset({CG_SPMV_KERNEL_NAME, CG_UPDATE_KERNEL_NAME})
CG_MODULE_A_NAME = "cg_module_a.hsaco"
CG_MODULE_B_NAME = "cg_module_b.hsaco"
CG_MODULE_NAMES = frozenset({CG_MODULE_A_NAME, CG_MODULE_B_NAME})
CG_SUMMARY_PATTERN = re.compile(r"pid=\d+(?: [a-z_]+=\S+)+$")
CODE_OBJECT_INFO_SUFFIX = "_code_obj_info.json"
PC_SAMPLING_RESULTS_SUFFIX = "_ps_file_results.json"


class ChildExpectation(NamedTuple):
    """What one conjugate-gradient child must report and attribute."""

    kernel_option: str
    kernel_name: str
    module_load_order: Tuple[str, str]


# Mirrors config["app_conjugate_gradient"]: --kernels spmv,spmv,update assigns a
# kernel per child, and --rotate-code-objects reverses the module load order for
# odd children. Every child dispatches out of module A regardless of that order.
CG_CHILD_EXPECTATIONS = {
    0: ChildExpectation(
        "spmv", CG_SPMV_KERNEL_NAME, (CG_MODULE_A_NAME, CG_MODULE_B_NAME)
    ),
    1: ChildExpectation(
        "spmv", CG_SPMV_KERNEL_NAME, (CG_MODULE_B_NAME, CG_MODULE_A_NAME)
    ),
    2: ChildExpectation(
        "update", CG_UPDATE_KERNEL_NAME, (CG_MODULE_A_NAME, CG_MODULE_B_NAME)
    ),
}


def pc_sampling_file_pids(file_names: Iterable[str], suffix: str) -> Set[int]:
    """Return unique numeric PID prefixes for files ending in ``suffix``."""
    matching_file_names = [
        file_name for file_name in file_names if file_name.endswith(suffix)
    ]
    process_identifier_prefixes = [
        file_name[: len(file_name) - len(suffix)] for file_name in matching_file_names
    ]
    assert all(
        process_identifier_prefix.isdigit()
        for process_identifier_prefix in process_identifier_prefixes
    ), f"expected numeric PID prefixes for {matching_file_names}"

    process_identifiers = [
        int(process_identifier_prefix)
        for process_identifier_prefix in process_identifier_prefixes
    ]
    assert len(process_identifiers) == len(set(process_identifiers)), (
        f"expected unique PID prefixes for {matching_file_names}"
    )
    return set(process_identifiers)


def _assert_pc_sampling_files(
    file_dict: Mapping[str, object], expected_count: int = 1
) -> None:
    """Assert PID-prefixed PC sampling and code-object output files."""
    file_names = set(file_dict)
    code_object_files = [
        file_name
        for file_name in file_names
        if file_name.endswith(CODE_OBJECT_INFO_SUFFIX)
    ]
    assert len(code_object_files) == expected_count, (
        f"expected {expected_count} *{CODE_OBJECT_INFO_SUFFIX}, got {code_object_files}"
    )
    pc_sampling_results = [
        file_name
        for file_name in file_names
        if file_name.endswith(PC_SAMPLING_RESULTS_SUFFIX)
    ]
    assert len(pc_sampling_results) == expected_count, (
        f"expected {expected_count} *{PC_SAMPLING_RESULTS_SUFFIX}, "
        f"got {pc_sampling_results}"
    )

    code_object_process_ids = pc_sampling_file_pids(
        code_object_files,
        CODE_OBJECT_INFO_SUFFIX,
    )
    result_process_ids = pc_sampling_file_pids(
        pc_sampling_results,
        PC_SAMPLING_RESULTS_SUFFIX,
    )
    assert result_process_ids == code_object_process_ids

    dynamic_files = {*code_object_files, *pc_sampling_results}
    remaining = file_names - dynamic_files
    assert remaining == {"sysinfo.csv"}


def collect_dispatched_cg_symbols(
    tool_data: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """Map dispatched CG kernel IDs to their symbol records."""
    symbols_by_kernel_id = {
        symbol["kernel_id"]: symbol for symbol in tool_data["kernel_symbols"]
    }

    dispatched_symbols = {}
    for dispatch in tool_data["buffer_records"]["kernel_dispatch"]:
        kernel_id = dispatch["dispatch_info"]["kernel_id"]
        symbol = symbols_by_kernel_id.get(kernel_id)
        if symbol is None:
            continue
        if symbol["formatted_kernel_name"] not in CG_KERNEL_NAMES:
            continue
        dispatched_symbols[kernel_id] = symbol
    return dispatched_symbols


def parse_child_summaries(stdout: str) -> Dict[int, Dict[str, str]]:
    """Parse the CG child summary lines into per-child key/value maps."""
    summaries = {}
    for line in stdout.splitlines():
        # Child stdout reaches the test prefixed by the streamed profiler log,
        # so match the summary wherever it starts on the line.
        match = CG_SUMMARY_PATTERN.search(line)
        if match is None:
            continue
        summary = dict(
            token.split("=", maxsplit=1) for token in match.group().split(" ")
        )
        child_index = int(summary["child"])
        assert child_index not in summaries, (
            f"duplicate summary for child {child_index}"
        )
        summaries[child_index] = summary

    assert set(summaries) == set(CG_CHILD_EXPECTATIONS), (
        f"expected summaries for children {sorted(CG_CHILD_EXPECTATIONS)}, "
        f"got {sorted(summaries)}"
    )
    process_ids = [summary["pid"] for summary in summaries.values()]
    assert len(set(process_ids)) == len(process_ids), (
        f"expected unique child PIDs, got {process_ids}"
    )
    assert all(summary["rounds"] == CG_ROUNDS for summary in summaries.values())
    return summaries


def code_object_ids_by_module(tool_data: Dict[str, Any]) -> Dict[str, int]:
    """Map each CG module filename to the process-local code-object ID it loaded."""
    module_code_object_ids: Dict[str, int] = {}
    for code_object in tool_data["code_objects"]:
        module_uri = code_object.get("uri")
        if not isinstance(module_uri, str):
            continue
        module_name = Path(module_uri.split("#offset=", maxsplit=1)[0]).name
        if module_name not in CG_MODULE_NAMES:
            continue
        assert module_name not in module_code_object_ids, (
            f"{module_name} is cataloged more than once"
        )
        module_code_object_ids[module_name] = code_object["code_object_id"]

    assert set(module_code_object_ids) == CG_MODULE_NAMES, (
        f"expected both CG modules in the process catalog, "
        f"got {sorted(module_code_object_ids)}"
    )
    return module_code_object_ids


def sampled_code_object_ids(
    sample_records: pd.DataFrame, kernel_ids: Set[int]
) -> Set[int]:
    """Return the code-object IDs sampled for the given dispatched kernel IDs."""
    kernel_samples = sample_records[sample_records["kernel_id"].isin(kernel_ids)]
    assert not kernel_samples.empty
    return {int(code_object_id) for code_object_id in kernel_samples["code_object_id"]}


def parse_tty_table(output: str, heading: str) -> List[Dict[str, str]]:
    """Parse the fancy_grid table the CLI renders under a section heading."""
    lines = output.splitlines()
    heading_indexes = [
        index for index, line in enumerate(lines) if line.strip().startswith(heading)
    ]
    assert len(heading_indexes) == 1, (
        f"expected exactly one {heading!r} table, found {len(heading_indexes)}"
    )

    headers = None
    rows = []
    for line in lines[heading_indexes[0] + 1 :]:
        stripped_line = line.strip()
        if stripped_line.startswith("╘"):
            break
        if not stripped_line.startswith("│"):
            continue
        cells = [cell.strip() for cell in stripped_line.strip("│").split("│")]
        if headers is None:
            headers = cells
        elif cells[0]:
            # A row whose leading index cell is blank continues a wrapped cell.
            rows.append(dict(zip(headers, cells)))

    assert rows, f"{heading!r} table has no rows"
    return rows


def assert_cli_code_object_attribution(
    output: str,
    module_ids_by_process_id: Mapping[int, Mapping[str, int]],
    kernel_name_by_process_id: Mapping[int, str],
) -> None:
    """Assert the rendered PC sampling table attributes rows per process.

    The terminal table is the surface users read, and it carries the same
    process-local identity the database does: pid, code_object_id, offset.
    """
    sampling_rows = parse_tty_table(output, "21.1 PC Sampling")
    attributions = {
        (int(row["pid"]), int(row["code_object_id"]), row["Kernel_Name"])
        for row in sampling_rows
    }
    expected_attributions = {
        (
            process_id,
            module_ids[CG_MODULE_A_NAME],
            kernel_name_by_process_id[process_id],
        )
        for process_id, module_ids in module_ids_by_process_id.items()
    }
    assert attributions == expected_attributions

    # Module B is loaded but never dispatched, so it reaches no row -- not even
    # where its ID matches a module A store in another process.
    module_b_keys = {
        (process_id, module_ids[CG_MODULE_B_NAME])
        for process_id, module_ids in module_ids_by_process_id.items()
    }
    assert {
        (process_id, code_object_id) for process_id, code_object_id, _ in attributions
    }.isdisjoint(module_b_keys)

    # Rotation splits one kernel across two code-object IDs, and the two
    # non-rotated children give one ID to two different kernels.
    spmv_code_object_ids = {
        code_object_id
        for _, code_object_id, kernel_name in attributions
        if kernel_name == CG_SPMV_KERNEL_NAME
    }
    assert len(spmv_code_object_ids) == 2
    kernels_by_code_object_id: Dict[int, Set[str]] = {}
    for _, code_object_id, kernel_name in attributions:
        kernels_by_code_object_id.setdefault(code_object_id, set()).add(kernel_name)
    assert any(
        len(kernel_names) > 1 for kernel_names in kernels_by_code_object_id.values()
    )

    assert all(int(row["count"]) > 0 for row in sampling_rows)

    # Both SpMV children run identical code, so their shared offsets must stay
    # separate rows rather than collapsing into one.
    spmv_offsets: Dict[int, Set[int]] = {}
    for row in sampling_rows:
        if row["Kernel_Name"] != CG_SPMV_KERNEL_NAME:
            continue
        spmv_offsets.setdefault(int(row["pid"]), set()).add(int(row["offset"], 16))
    assert len(spmv_offsets) == 2
    first_offsets, second_offsets = spmv_offsets.values()
    assert first_offsets & second_offsets


def assert_analyze_code_object_attribution(
    database_path: Path,
    module_ids_by_process_id: Mapping[int, Mapping[str, int]],
    kernel_name_by_process_id: Mapping[int, str],
) -> None:
    """Assert analyze keeps each child's code object and kernel separate.

    ``code_object_id`` is process-local: the rotated child's module A shares an
    ID with another child's module B, and the non-rotated children give the same
    ID to two different kernels. Analyze must key on (pid, code object)
    throughout, never on the ID alone. The HIP runtime's own copy code object is
    ignored here; only the CG kernels are the subject.
    """
    connection = sqlite3.connect(str(database_path))
    try:
        stores = pd.read_sql_query(
            "SELECT pid, code_object_id FROM compute_code_object_store",
            connection,
        )
        symbols = pd.read_sql_query(
            "SELECT cos.pid, cos.code_object_id, k.kernel_name "
            "FROM compute_kernel_symbol ks "
            "JOIN compute_code_object_store cos "
            "ON ks.code_object_uuid = cos.code_object_uuid "
            "JOIN compute_kernel k ON ks.kernel_uuid = k.kernel_uuid",
            connection,
        )
        kernels = pd.read_sql_query(
            "SELECT kernel_name, dispatch_count FROM compute_kernel_view",
            connection,
        )
        samples = pd.read_sql_query(
            "SELECT pid, code_object_id, kernel_name, offset, count "
            "FROM compute_pc_sampling_summary_view",
            connection,
        )
    finally:
        connection.close()

    expected_symbols = {
        (
            process_id,
            module_ids[CG_MODULE_A_NAME],
            kernel_name_by_process_id[process_id],
        )
        for process_id, module_ids in module_ids_by_process_id.items()
    }
    store_keys = set(zip(stores["pid"], stores["code_object_id"]))

    # Module B is loaded but never dispatched, so analyze stores nothing for it
    # -- even where its ID matches a module A store in another process.
    module_b_keys = {
        (process_id, module_ids[CG_MODULE_B_NAME])
        for process_id, module_ids in module_ids_by_process_id.items()
    }
    assert store_keys.isdisjoint(module_b_keys)

    # One symbol per child, in that child's own store, naming the kernel it ran.
    cg_symbols = symbols[symbols["kernel_name"].isin(CG_KERNEL_NAMES)]
    assert (
        set(
            zip(
                cg_symbols["pid"],
                cg_symbols["code_object_id"],
                cg_symbols["kernel_name"],
            )
        )
        == expected_symbols
    )
    assert len(cg_symbols) == len(expected_symbols)

    # Rotation splits one kernel across two code-object IDs, and the two
    # non-rotated children give one ID to two different kernels. Keying on the
    # ID alone would both split and merge the wrong things.
    spmv_symbols = cg_symbols[cg_symbols["kernel_name"] == CG_SPMV_KERNEL_NAME]
    assert len(spmv_symbols) == 2
    assert spmv_symbols["code_object_id"].nunique() == 2
    kernels_by_code_object_id = cg_symbols.groupby("code_object_id")["kernel_name"]
    assert any(names.nunique() > 1 for _, names in kernels_by_code_object_id)

    # Kernels aggregate by name across processes; their dispatches do not.
    dispatch_counts = dict(zip(kernels["kernel_name"], kernels["dispatch_count"]))
    assert dispatch_counts[CG_SPMV_KERNEL_NAME] == 2
    assert dispatch_counts[CG_UPDATE_KERNEL_NAME] == 1

    cg_samples = samples[samples["kernel_name"].isin(CG_KERNEL_NAMES)]
    assert (
        set(
            zip(
                cg_samples["pid"],
                cg_samples["code_object_id"],
                cg_samples["kernel_name"],
            )
        )
        == expected_symbols
    )
    assert (cg_samples.groupby("pid")["count"].sum() > 0).all()

    # Both SpMV children run identical code, so their shared offsets must stay
    # separate rows: the view's identity is (pid, code object, kernel, offset).
    spmv_samples = cg_samples[cg_samples["kernel_name"] == CG_SPMV_KERNEL_NAME]
    offsets_by_process_id = spmv_samples.groupby("pid")["offset"].apply(set)
    assert len(offsets_by_process_id) == 2
    first_offsets, second_offsets = offsets_by_process_id
    assert first_offsets & second_offsets


def is_pc_sampling_not_supported(output):
    """
    To be called with the stdout + stderr after profiling.
    Check whether profiling output said PC sampling is not supported on the machine
    """
    return any(
        marker in output
        for marker in (
            # rocprof-compute's own pre-flight check against the agent configs
            "is not supported on any of the agents on this system",
            # rocprofiler-sdk, when it accepts the run and then rejects the config
            "Given PC sampling configuration is not supported",
        )
    )


def _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir):
    if is_pc_sampling_not_supported(f"{stdout}\n{stderr}"):
        common.clean_output_dir(config["cleanup"], workload_dir)
        pytest.skip("PC sampling is not supported")


def test_pc_sampling_host_trap(binary_handler_profile_rocprof_compute, monkeypatch):
    """
    Test that PC sampling works with --block 21 and --pc-sampling-method host_trap.
    """
    integration_common.require_pc_sampling_gpu()
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        stream=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = integration_common.check_non_pmc_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict)

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_stochastic(binary_handler_profile_rocprof_compute, monkeypatch):
    """
    Test that PC sampling works with --block 21 and --pc-sampling-method stochastic.
    """
    integration_common.require_pc_sampling_gpu(is_stochastic=True)
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "--pc-sampling-method",
        "stochastic",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        stream=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = integration_common.check_non_pmc_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict)

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.parametrize("sampling_method", ["host_trap", "stochastic"])
def test_multiprocess_pc_sampling_distinct_code_objects(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    capsys,
    monkeypatch,
    sampling_method,
):
    """Correlate per-process samples with rotated CG code-object IDs."""
    integration_common.require_pc_sampling_gpu(
        is_stochastic=sampling_method == "stochastic"
    )
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "--pc-sampling-method",
        sampling_method,
    ]
    workload_dir = common.get_output_dir(param_id=sampling_method)

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        stream=True,
        roof=False,
        app_name="app_conjugate_gradient",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    child_summaries = parse_child_summaries(stdout)
    file_dict = integration_common.check_non_pmc_files(
        workload_dir,
        num_devices,
        1,
    )
    _assert_pc_sampling_files(file_dict, expected_count=3)

    result_process_ids = pc_sampling_file_pids(
        file_dict,
        PC_SAMPLING_RESULTS_SUFFIX,
    )
    tool_data_records = load_pc_sampling_results(workload_dir)
    assert len(tool_data_records) == 3
    metadata_process_ids = {
        tool_data["metadata"]["pid"] for tool_data in tool_data_records
    }
    assert metadata_process_ids == result_process_ids

    tool_data_by_process_id = {
        tool_data["metadata"]["pid"]: tool_data for tool_data in tool_data_records
    }
    process_id_by_child = {
        child_index: int(summary["pid"])
        for child_index, summary in child_summaries.items()
    }
    assert set(process_id_by_child.values()) == result_process_ids

    module_a_ids_by_child = {}
    module_ids_by_process_id = {}
    kernel_name_by_process_id = {}
    for child_index, expectation in CG_CHILD_EXPECTATIONS.items():
        summary = child_summaries[child_index]
        assert summary["kernel"] == expectation.kernel_option
        assert summary["modules"] == ",".join(expectation.module_load_order)
        assert summary["target"] == "cg_module_a"

        process_id = process_id_by_child[child_index]
        tool_data = tool_data_by_process_id[process_id]

        # Both modules load in every child; only the load order rotates, and the
        # SDK numbers code objects in load order.
        module_ids = code_object_ids_by_module(tool_data)
        first_module, second_module = expectation.module_load_order
        assert module_ids[first_module] < module_ids[second_module], (
            f"child {child_index} loaded {expectation.module_load_order}, "
            f"but got code object IDs {module_ids}"
        )
        module_a_ids_by_child[child_index] = module_ids[CG_MODULE_A_NAME]
        module_ids_by_process_id[process_id] = module_ids
        kernel_name_by_process_id[process_id] = expectation.kernel_name

        # The child always resolves its kernel out of module A, so the dispatched
        # symbol and every sample for it carry module A's code-object ID.
        dispatched_symbols = collect_dispatched_cg_symbols(tool_data)
        assert len(dispatched_symbols) == 1
        dispatched_symbol = next(iter(dispatched_symbols.values()))
        assert dispatched_symbol["formatted_kernel_name"] == expectation.kernel_name
        assert dispatched_symbol["code_object_id"] == module_ids[CG_MODULE_A_NAME]

        sample_records = load_pc_sample_records(tool_data)
        assert sampled_code_object_ids(sample_records, set(dispatched_symbols)) == {
            module_ids[CG_MODULE_A_NAME]
        }

        # Module B is loaded but never used, so no sample can reference it.
        assert module_ids[CG_MODULE_B_NAME] not in set(sample_records["code_object_id"])

    # Rotation gives module A a different process-local ID in the odd child.
    assert module_a_ids_by_child[0] != module_a_ids_by_child[1]

    # No kernel filter and no row cap: the table then holds every process's
    # rows, which the default 10 highest-count rows would hide.
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--block",
        "21",
        "--pc-sampling-rows",
        "0",
    ])
    assert code == 0

    captured = capsys.readouterr()
    assert "21. PC Sampling" in captured.out
    assert_cli_code_object_attribution(
        captured.out,
        module_ids_by_process_id,
        kernel_name_by_process_id,
    )

    # Kernels aggregate by name across processes; their dispatches do not.
    kernel_rows = parse_tty_table(captured.out, "0.1 Top Kernels")
    dispatch_counts = {row["Kernel_Name"]: float(row["Count"]) for row in kernel_rows}
    assert dispatch_counts[CG_SPMV_KERNEL_NAME] == 2
    assert dispatch_counts[CG_UPDATE_KERNEL_NAME] == 1

    workload_path = Path(workload_dir).resolve()
    dispatch_info_path = workload_path / "pmc_dispatch_info.csv"
    assert dispatch_info_path.exists()
    dispatch_info = pd.read_csv(dispatch_info_path)
    assert "PID" in dispatch_info.columns
    dispatch_process_ids = set(dispatch_info["PID"].astype(int))
    assert dispatch_process_ids == result_process_ids

    # --output-name forbids path separators, so the db lands in the cwd; run
    # from the workload dir to keep it there for clean_output_dir to remove.
    database_name = f"cg_code_objects_{sampling_method}"
    monkeypatch.chdir(workload_path)
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        str(workload_path),
        "--block",
        "21",
        "--output-format",
        "db",
        "--output-name",
        database_name,
    ])
    assert code == 0

    database_path = workload_path / f"{database_name}.db"
    assert database_path.is_file()
    assert_analyze_code_object_attribution(
        database_path,
        module_ids_by_process_id,
        kernel_name_by_process_id,
    )

    common.clean_output_dir(config["cleanup"], str(workload_path))


def test_multi_rank_pc_sampling_only(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that no multi-rank warning is printed when running with only
    --block 21 (PC sampling only mode requires a single pass) with multi-rank.
    """
    integration_common.require_pc_sampling_gpu()
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    workload_dir = common.get_output_dir()

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
    ]

    _, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        app_name="app_1",
        capture_output=True,
        stream=True,
        check_success=False,
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    output = stdout + stderr
    assert "Multi-rank application detected" not in output

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_multi_rank_warning_pc_sampling_with_counters(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that a multi-rank warning is printed when running with --block 21
    and another block (PC sampling with counters mode requires multiple passes)
    with multi-rank.
    """
    integration_common.require_pc_sampling_gpu()
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    workload_dir = common.get_output_dir()

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "2",
        "--pc-sampling-method",
        "host_trap",
    ]

    _, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        app_name="app_1",
        capture_output=True,
        stream=True,
        check_success=False,
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    output = stdout + stderr
    assert "Multi-rank application detected" in output
    assert "Application replay mode" in output
    assert "--iteration-multiplexing" in output
    assert "--block" not in output
    assert "--set" in output

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_profile_then_analyze(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    capsys,
    monkeypatch,
):
    """
    End-to-end: profile with PC sampling (host_trap), then
    run analysis on the profiling output.
    """
    integration_common.require_pc_sampling_gpu()
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        stream=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = integration_common.check_non_pmc_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict)

    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--path",
            workload_dir,
            "--block",
            "21",
        ],
    )
    assert code == 0

    captured = capsys.readouterr()
    assert "0.1 Top Kernels" in captured.out
    assert "0.2 Dispatch List" in captured.out

    workload_path = Path(workload_dir)

    kernel_top_csv = workload_path / "pmc_kernel_top.csv"
    assert kernel_top_csv.exists()
    kernel_top_header = kernel_top_csv.read_text().splitlines()[0]
    assert "Kernel_Name" in kernel_top_header
    assert "Count" in kernel_top_header
    assert "Percent" in kernel_top_header

    dispatch_info_csv = workload_path / "pmc_dispatch_info.csv"
    assert dispatch_info_csv.exists()
    dispatch_info_header = dispatch_info_csv.read_text().splitlines()[0]
    assert "Dispatch_ID" in dispatch_info_header
    assert "Kernel_Name" in dispatch_info_header
    assert "GPU_ID" in dispatch_info_header

    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--path",
            workload_dir,
            "--block",
            "21",
            "--kernel",
            "0",
        ],
    )
    assert code == 0

    captured = capsys.readouterr()
    assert "0.1 Top Kernels" in captured.out
    assert "0.2 Dispatch List" in captured.out
    assert "21. PC Sampling" in captured.out

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_with_sol_block(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    capsys,
    monkeypatch,
):
    """
    PC sampling with counter collection (--block 21 2): profiling produces the
    expected artifacts and analyze renders both counter and PC sampling panels.
    """
    integration_common.require_pc_sampling_gpu()
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "2",
        "--pc-sampling-method",
        "host_trap",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        stream=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = integration_common.check_csv_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict)

    assert common.check_file_pattern("- '21'", f"{workload_dir}/profiling_config.yaml")
    assert common.check_file_pattern("- '2'", f"{workload_dir}/profiling_config.yaml")

    # Analyze with a single kernel so the detailed PC sampling table renders.
    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--path",
            workload_dir,
            "--kernel",
            "0",
        ],
    )
    assert code == 0

    captured = capsys.readouterr()
    assert "2.1 System Speed-of-Light" in captured.out
    assert "21. PC Sampling" in captured.out
    # The "instruction" column header only renders when the table has rows.
    assert "instruction" in captured.out

    common.clean_output_dir(config["cleanup"], workload_dir)
