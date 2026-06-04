# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Profile artifact writer implementations."""

from pathlib import Path

from utils.profile_artifacts.interfaces import ProfilePassContext


class RocpdProfileArtifactWriter:
    """Finalize current ROCPD profile artifacts."""

    def finalize_pass(self, context: ProfilePassContext) -> None:
        from utils import utils_profile as profile_ops

        db_paths = list((context.workload_dir / "out" / "pmc_1").glob("*/*.db"))
        if context.using_native_tool:
            _update_rocpd_native_counter_events(db_paths, context)

        profile_ops.rocpd_data.convert_dbs_to_csv(
            [str(path) for path in db_paths],
            str(_rocpd_counter_collection_path(context)),
            str(_rocpd_marker_trace_path(context)),
        )
        combined_rows = _read_rocpd_counter_rows(context)
        if not combined_rows:
            profile_ops.console_warning(
                "No GPU kernel data collected. "
                "The workload may not have dispatched any GPU kernels."
            )
            profile_ops.shutil.rmtree(
                str(context.workload_dir / "out"),
                ignore_errors=True,
            )
            return

        _normalize_rocpd_counter_rows(combined_rows)
        profile_ops.csv_ops.write_csv_from_dicts(
            str(_rocpd_counter_collection_path(context)),
            combined_rows,
        )
        profile_ops.csv_ops.write_csv_from_dicts(
            str(context.workload_dir / f"results_{context.fbase}.csv"),
            combined_rows,
        )
        profile_ops.console_warning(
            "Intermediate results_*.csv generation from rocpd databases is "
            "deprecated and will be replaced with automatic .db file "
            "retention in a future release."
        )
        if context.torch_trace_enabled:
            profile_ops.save_torch_trace_inputs(
                str(context.workload_dir),
                context.fbase,
                "rocpd",
            )
        if context.retain_rocpd_output:
            _retain_rocpd_databases(db_paths, context)
        profile_ops.shutil.rmtree(str(context.workload_dir / "out"))


class CsvProfileArtifactWriter:
    """Finalize current CSV profile artifacts."""

    def finalize_pass(self, context: ProfilePassContext) -> None:
        from utils import utils_profile as profile_ops

        result_files = _process_csv_outputs(context)
        if context.torch_trace_enabled:
            profile_ops.save_torch_trace_inputs(
                str(context.workload_dir),
                context.fbase,
                "csv",
            )
        if not result_files:
            profile_ops.console_warning(
                f"Cannot write results for {context.fbase}.csv due to no counter "
                "csv files generated."
            )
            return

        combined_results = profile_ops.csv_ops.concat_csv_files(result_files)
        _normalize_csv_counter_rows(combined_results)
        _write_csv_counter_results(combined_results, context)
        _standardize_csv_headers(context)


def _update_rocpd_native_counter_events(
    db_paths: list[Path],
    context: ProfilePassContext,
) -> None:
    from utils import utils_profile as profile_ops

    for db_name in db_paths:
        pid = db_name.stem.split("_")[0]
        counter_csv = (
            context.workload_dir
            / "out"
            / "pmc_1"
            / f"{pid}_native_counter_collection.csv"
        )
        if not counter_csv.is_file():
            profile_ops.console_debug(
                f"No native counter CSV for pid {pid}; "
                f"skipping rocpd update for {db_name}."
            )
            continue
        counter_rows, _ = profile_ops.csv_ops.read_csv_as_dicts(str(counter_csv))
        profile_ops.rocpd_data.update_rocpd_pmc_events(counter_rows, str(db_name))
        profile_ops.console_debug(
            f"Updated rocpd db {db_name} with native tool counters."
        )


def _rocpd_counter_collection_path(context: ProfilePassContext) -> Path:
    return context.workload_dir / "out" / "pmc_1" / (
        f"{context.fbase}_counter_collection.csv"
    )


def _rocpd_marker_trace_path(context: ProfilePassContext) -> Path:
    return context.workload_dir / "out" / "pmc_1" / (
        f"{context.fbase}_marker_api_trace.csv"
    )


def _read_rocpd_counter_rows(context: ProfilePassContext) -> list[dict]:
    from utils import utils_profile as profile_ops

    try:
        combined_rows, _ = profile_ops.csv_ops.read_csv_as_dicts(
            str(_rocpd_counter_collection_path(context))
        )
        return combined_rows
    except (FileNotFoundError, ValueError):
        return []


def _normalize_rocpd_counter_rows(combined_rows: list[dict]) -> None:
    from utils import utils_profile as profile_ops

    profile_ops.csv_ops.assign_group_ids(
        combined_rows,
        [
            "PID",
            "Kernel_Name",
            "Grid_Size",
            "Workgroup_Size",
            "LDS_Per_Workgroup",
            "Start_Timestamp",
            "End_Timestamp",
        ],
        "Dispatch_ID",
    )
    profile_ops.csv_ops.assign_group_ids(
        combined_rows,
        ["Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup"],
        "Kernel_ID",
    )
    profile_ops.csv_ops.drop_column_from_rows(combined_rows, "PID")


def _retain_rocpd_databases(
    db_paths: list[Path],
    context: ProfilePassContext,
) -> None:
    from utils import utils_profile as profile_ops

    profile_ops.console_warning(
        "--retain-rocpd-output is deprecated and will be removed in "
        "a future release. .db files will be retained automatically."
    )
    for db_path in db_paths:
        pid = db_path.stem.split("_")[0]
        retained_path = context.workload_dir / f"{context.fbase}_{pid}.db"
        profile_ops.shutil.copyfile(db_path, retained_path)
        profile_ops.console_warning(
            f"Retaining large raw rocpd database: {retained_path}"
        )


def _process_csv_outputs(context: ProfilePassContext) -> list[str]:
    from utils import utils_profile as profile_ops

    if context.profiler_command == "rocprofiler-sdk":
        return profile_ops.process_rocprofv3_output(
            str(context.workload_dir),
            using_native_tool=context.using_native_tool,
        )

    result_files = profile_ops.process_rocprofv3_output(
        str(context.workload_dir),
        using_native_tool=False,
    )
    if context.kokkos_trace_enabled:
        profile_ops.process_kokkos_trace_output(
            str(context.workload_dir),
            context.fbase,
        )
    return result_files


def _normalize_csv_counter_rows(combined_results: list[dict]) -> None:
    from utils import utils_profile as profile_ops

    profile_ops.csv_ops.add_column_to_rows(
        combined_results,
        "Dispatch_ID",
        list(range(0, len(combined_results))),
    )
    profile_ops.csv_ops.assign_group_ids(
        combined_results,
        ["Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup"],
        "Kernel_ID",
    )


def _write_csv_counter_results(
    combined_results: list[dict],
    context: ProfilePassContext,
) -> None:
    from utils import utils_profile as profile_ops

    profile_ops.csv_ops.write_csv_from_dicts(
        str(context.workload_dir / "out" / "pmc_1" / f"results_{context.fbase}.csv"),
        combined_results,
    )
    if (context.workload_dir / "out").exists():
        profile_ops.shutil.copyfile(
            str(
                context.workload_dir
                / "out"
                / "pmc_1"
                / f"results_{context.fbase}.csv"
            ),
            str(context.workload_dir / f"results_{context.fbase}.csv"),
        )
        profile_ops.shutil.rmtree(str(context.workload_dir / "out"))


def _standardize_csv_headers(context: ProfilePassContext) -> None:
    from utils import utils_profile as profile_ops

    csv_path = context.workload_dir / f"results_{context.fbase}.csv"
    rows, _ = profile_ops.csv_ops.read_csv_as_dicts(str(csv_path))
    profile_ops.csv_ops.rename_columns(rows, _csv_output_headers())
    profile_ops.csv_ops.write_csv_from_dicts(str(csv_path), rows)


def _csv_output_headers() -> dict[str, str]:
    return {
        "KernelName": "Kernel_Name",
        "Index": "Dispatch_ID",
        "grd": "Grid_Size",
        "gpu-id": "GPU_ID",
        "wgr": "Workgroup_Size",
        "lds": "LDS_Per_Workgroup",
        "scr": "Scratch_Per_Workitem",
        "sgpr": "SGPR",
        "arch_vgpr": "Arch_VGPR",
        "accum_vgpr": "Accum_VGPR",
        "BeginNs": "Start_Timestamp",
        "EndNs": "End_Timestamp",
        "GRD": "Grid_Size",
        "WGR": "Workgroup_Size",
        "LDS": "LDS_Per_Workgroup",
        "SCR": "Scratch_Per_Workitem",
        "ACCUM_VGPR": "Accum_VGPR",
    }
