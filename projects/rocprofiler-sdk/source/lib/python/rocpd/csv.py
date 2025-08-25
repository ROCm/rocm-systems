#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

import os
import re
from typing import Tuple

from .importer import RocpdImportData
from .query import export_sqlite_query
from .time_window import apply_time_window
from . import output_config
from . import libpyrocpd


def write_sql_query_to_csv(
    connection: RocpdImportData,
    config,
    query,
    filename="",
    postfix="trace",
) -> None:
    """Write the contents of a SQL query to a CSV file in the specified output path."""

    query_not_empty = f"""
        SELECT EXISTS (
            {query}
        )
    """

    # just return if the result is empty
    if not connection.execute(query_not_empty).fetchone()[0]:
        return

    # call query module to export to csv
    file_prefix = config.output_file + "_" if config.output_file else ""
    file_postfix = "_" + postfix if postfix else ""
    export_path = os.path.join(
        config.output_path, f"{file_prefix}{filename}{file_postfix}.csv"
    )
    export_sqlite_query(connection, query, export_format="csv", export_path=export_path)


def get_header_with_alias(column_name, titled_columns) -> Tuple[str, str]:
    match = re.match(r"(.+?)\s+AS\s+(.+)", column_name, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)

    if titled_columns:
        return column_name, column_name.title()

    return column_name, ""


def write_agent_info_csv(importData, config) -> None:

    # Define mapping of output column name to JSON key
    json_keys = [
        "node_id",
        "logical_node_id",
        "cpu_cores_count",
        "simd_count",
        "cpu_core_id_base",
        "simd_id_base",
        "max_waves_per_simd",
        "lds_size_in_kb",
        "gds_size_in_kb",
        "num_gws",
        "wave_front_size",
        "num_xcc",
        "cu_count",
        "array_count",
        "num_shader_banks",
        "simd_arrays_per_engine",
        "cu_per_simd_array",
        "simd_per_cu",
        "max_slots_scratch_cu",
        "gfx_target_version",
        "vendor_id",
        "device_id",
        "location_id",
        "domain",
        "drm_render_minor",
        "num_sdma_engines",
        "num_sdma_xgmi_engines",
        "num_sdma_queues_per_engine",
        "num_cp_queues",
        "max_engine_clk_ccompute",
        "max_engine_clk_fcompute",
        "sdma_fw_version.uCodeSDMA AS Sdma_Fw_Version",
        "fw_version.uCode AS Fw_Version",
        "cu_per_engine",
        "max_waves_per_cu",
        "workgroup_max_size",
        "family_id",
        "grid_max_size",
        "local_mem_size",
        "hive_id",
        "gpu_id",
        "workgroup_max_dim.x AS Workgroup_Max_Dim_X",
        "workgroup_max_dim.y AS Workgroup_Max_Dim_Y",
        "workgroup_max_dim.z AS Workgroup_Max_Dim_Z",
        "grid_max_dim.x AS Grid_Max_Dim_X",
        "grid_max_dim.y AS Grid_Max_Dim_Y",
        "grid_max_dim.z AS Grid_Max_Dim_Z",
        "vendor_name",
        "product_name",
    ]

    # Build SELECT clause for json_extract columns
    select_json = []
    for column in json_keys:
        column_name, column_alias = get_header_with_alias(column, config.titled_columns)
        alias_postfix = " AS " + (column_alias if column_alias else column_name)
        select_json.append(f"json_extract(extdata, '$.{column_name}'){alias_postfix}")

    # Build Capability value for SELECT clause
    def cap_expr(key, shift, mask=None):
        base = f"COALESCE(json_extract(extdata, '$.capability.{key}'), 0)"
        if mask is not None:
            base = f"({base} & {mask})"
        return f"({base} << {hex(shift)})"

    capability_bits = [
        ("HotPluggable", 0x0),
        ("HSAMMUPresent", 0x1),
        ("SharedWithGraphics", 0x2),
        ("QueueSizePowerOfTwo", 0x3),
        ("QueueSize32bit", 0x4),
        ("QueueIdleEvent", 0x5),
        ("VALimit", 0x6),
        ("WatchPointsSupported", 0x7),
        ("WatchPointsTotalBits", 0x8, 0xF),
        ("DoorbellType", 0xC, 0x3),
        ("AQLQueueDoubleMap", 0xE),
        ("DebugTrapSupported", 0xF),
        ("WaveLaunchTrapOverrideSupported", 0x10),
        ("WaveLaunchModeSupported", 0x11),
        ("PreciseMemoryOperationsSupported", 0x12),
        ("DEPRECATED_SRAM_EDCSupport", 0x13),
        ("Mem_EDCSupport", 0x14),
        ("RASEventNotify", 0x15),
        ("ASICRevision", 0x16, 0xF),
        ("SRAM_EDCSupport", 0x1A),
        ("SVMAPISupported", 0x1B),
        ("CoherentHostAccess", 0x1C),
        ("DebugSupportedFirmware", 0x1D),
        ("PreciseALUOperationsSupported", 0x1E),
        ("PerQueueResetSupported", 0x1F),
    ]

    capability_exprs = [cap_expr(*args) for args in capability_bits]

    select_capability = [" | ".join(capability_exprs) + " AS Capability"]

    # Add non-JSON columns
    fixed_keys = [
        "guid",
        "type AS Agent_Type",
        "name",
        "model_name",
    ]

    fixed_select = []
    for column in fixed_keys:
        column_name, column_alias = get_header_with_alias(column, config.titled_columns)
        alias_postfix = " AS " + column_alias if column_alias else ""
        fixed_select.append(f"{column_name}{alias_postfix}")

    # to mimic the previous order
    select_clause = (
        fixed_select[:1]
        + select_json[:2]
        + fixed_select[1:2]
        + select_json[2:33]
        + select_capability
        + select_json[33:47]
        + fixed_select[2:3]
        + select_json[47:]
        + fixed_select[3:4]
    )

    select_clause = ",\n    ".join(select_clause)

    query = f"""
        SELECT
            {select_clause}
        FROM "rocpd_info_agent"
    """

    write_sql_query_to_csv(importData, config, query, "agent_info", "")


def build_agent_id_string(agent_index_value, prefix=""):

    agent_prefix = prefix + "_" if prefix else ""

    if agent_index_value == libpyrocpd.agent_indexing.node:  # absolute
        return f"'Agent ' || {agent_prefix}agent_abs_index"
    elif (
        agent_index_value == libpyrocpd.agent_indexing.logical_node
    ):  # relative (default)
        return f"'Agent ' || {agent_prefix}agent_log_index"
    elif (
        agent_index_value == libpyrocpd.agent_indexing.logical_node_type
    ):  # type-relative
        return f"{agent_prefix}agent_type || ' ' || {agent_prefix}agent_type_index"
    else:
        return ""


def write_kernel_csv(importData, config) -> None:

    agent_id = build_agent_id_string(config.agent_index_value)

    if config.kernel_rename:
        kernel_name = "region"
    else:
        kernel_name = "name"

    select_columns = [
        "guid",
        "'KERNEL_DISPATCH' AS Kind",
        f"{agent_id} AS Agent_Id",
        "queue_id",
        "stream_id",
        "tid AS Thread_Id",
        "dispatch_id",
        "kernel_Id",
        f"{kernel_name} AS Kernel_Name",
        "stack_id AS Correlation_Id",
        "start AS Start_Timestamp",
        "end AS End_Timestamp",
        "lds_size AS LDS_Block_Size",
        "scratch_size",
        "vgpr_count AS VGPR_Count",
        "accum_vgpr_count AS Accum_VGPR_Count",
        "sgpr_count AS SGPR_Count",
        "workgroup_x AS Workgroup_Size_X",
        "workgroup_y AS Workgroup_Size_Y",
        "workgroup_z AS Workgroup_Size_Z",
        "grid_x AS Grid_Size_X",
        "grid_y AS Grid_Size_Y",
        "grid_z AS Grid_Size_Z",
    ]

    aliased_headers = []
    for column in select_columns:
        column_name, column_alias = get_header_with_alias(column, config.titled_columns)
        alias_postfix = " AS " + column_alias if column_alias else ""
        aliased_headers.append(f"{column_name}{alias_postfix}")

    select_clause = ",\n".join(aliased_headers)

    query = f"""
        SELECT
            {select_clause}
        FROM "kernels"
        ORDER BY
            guid ASC, start ASC, end DESC
    """
    write_sql_query_to_csv(importData, config, query, "kernel")


def write_memory_copy_csv(importData, config) -> None:

    src_agent_id = build_agent_id_string(config.agent_index_value, "src")
    dst_agent_id = build_agent_id_string(config.agent_index_value, "dst")

    query = f"""
        SELECT
            guid AS Guid,
            'MEMORY_COPY' AS Kind,
            name AS Direction,
            stream_id AS Stream_Id,
            {src_agent_id} AS Source_Agent_Id,
            {dst_agent_id} AS Destination_Agent_Id,
            stack_id AS Correlation_Id,
            start AS Start_Timestamp,
            end AS End_Timestamp
        FROM "memory_copies"
        ORDER BY
            guid ASC, start ASC, end DESC
    """
    write_sql_query_to_csv(importData, config, query, "memory_copy")


def write_memory_allocation_csv(importData, config) -> None:

    agent_id = build_agent_id_string(config.agent_index_value)

    query = f"""
        SELECT
            guid AS Guid,
            'MEMORY_ALLOCATION' AS Kind,
            CASE
                WHEN type = 'ALLOC'
                THEN 'MEMORY_ALLOCATION_ALLOCATE'
                ELSE 'MEMORY_ALLOCATION_' || type
            END AS Operation,
            CASE
                WHEN type != 'FREE'
                THEN {agent_id}
                ELSE '"'
            END AS Agent_Id,
            size AS Allocation_Size,
            '0x' || printf('%016X', address) AS Address,
            stack_id AS Correlation_Id,
            start AS Start_Timestamp,
            end AS End_Timestamp
        FROM "memory_allocations"
        ORDER BY
            guid ASC, start ASC, end DESC
    """
    write_sql_query_to_csv(importData, config, query, "memory_allocation")


def write_counters_csv(importData, config) -> None:

    agent_id = build_agent_id_string(config.agent_index_value)

    select_columns = [
        "guid",
        "stack_id AS Correlation_Id",
        "dispatch_id",
        f"{agent_id} AS Agent_Id",
        "queue_id",
        "pid AS Process_Id",
        "tid AS Thread_Id",
        "grid_size",
        "kernel_id",
        "kernel_name",
        "workgroup_size",
        "lds_block_size AS LDS_Block_Size",
        "scratch_size",
        "vgpr_count AS VGPR_Count",
        "accum_vgpr_count AS Accum_VGPR_Count",
        "sgpr_count AS SGPR_Count",
        "counter_name",
        "value AS Counter_Value",
        "start AS Start_Timestamp",
        "end AS End_Timestamp",
    ]

    aliased_headers = []
    for column in select_columns:
        column_name, column_alias = get_header_with_alias(column, config.titled_columns)
        alias_postfix = " AS " + column_alias if column_alias else ""
        aliased_headers.append(f"{column_name}{alias_postfix}")

    select_clause = ",\n".join(aliased_headers)

    query = f"""
        SELECT
            {select_clause}
        FROM "counters_collection"
        ORDER BY
            guid ASC, start ASC, end DESC
    """
    write_sql_query_to_csv(importData, config, query, "counter_collection")


def write_scratch_memory_csv(importData, config) -> None:

    agent_id = build_agent_id_string(config.agent_index_value)

    query = f"""
        SELECT
            guid AS Guid,
            'SCRATCH_MEMORY' AS Kind,
            'SCRATCH_MEMORY_' || operation AS Operation,
            {agent_id} AS Agent_Id,
            queue_id AS Queue_Id,
            tid AS Thread_Id,
            alloc_flags AS Alloc_Flags,
            start AS Start_Timestamp,
            end AS End_Timestamp
        FROM "scratch_memory"
        ORDER BY
            guid ASC, start ASC, end DESC
    """
    write_sql_query_to_csv(importData, config, query, "scratch_memory")


def write_region_csv(importData, config) -> None:

    query = """
        SELECT
            guid AS Guid,
            category AS Domain,
            CASE
                WHEN json_extract(extdata, '$.message') IS NOT NULL
                THEN json_extract(extdata, '$.message')
                ELSE name
            END AS Function,
            pid AS Process_Id,
            tid AS Thread_Id,
            stack_id AS Correlation_Id,
            start AS Start_Timestamp,
            end AS End_Timestamp
        FROM "regions_and_samples"
        ORDER BY
            guid ASC, start ASC, end DESC
    """
    write_sql_query_to_csv(importData, config, query, "region")


def write_csv(importData, config):

    write_agent_info_csv(importData, config)
    write_counters_csv(importData, config)
    write_kernel_csv(importData, config)
    write_memory_allocation_csv(importData, config)
    write_memory_copy_csv(importData, config)
    write_region_csv(importData, config)
    write_scratch_memory_csv(importData, config)


def execute(input, config=None, window_args=None, **kwargs):

    importData = RocpdImportData(input)

    apply_time_window(importData, **window_args)

    config = (
        output_config.output_config(**kwargs)
        if config is None
        else config.update(**kwargs)
    )

    write_csv(importData, config)


def add_args(parser):
    """Add csv arguments."""

    csv_options = parser.add_argument_group("CSV options")
    csv_options.add_argument(
        "--titled-columns",
        action="store_true",
        default=False,
        help="Format column names as titles in the output",
    )

    return ["titled_columns"]


def process_args(args, valid_args):
    ret = {}
    for itr in valid_args:
        if hasattr(args, itr):
            val = getattr(args, itr)
            if val is not None:
                ret[itr] = val
    return ret


def main(argv=None):
    import argparse
    from .time_window import add_args as add_args_time_window
    from .time_window import process_args as process_args_time_window
    from .output_config import add_args as add_args_output_config
    from .output_config import process_args as process_args_output_config
    from .output_config import add_generic_args, process_generic_args

    parser = argparse.ArgumentParser(
        description="Convert rocPD to CSV files",
        allow_abbrev=False,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    required_params = parser.add_argument_group("Required arguments")

    required_params.add_argument(
        "-i",
        "--input",
        required=True,
        type=output_config.check_file_exists,
        nargs="+",
        help="Input path and filename to one or more database(s), separated by spaces",
    )

    valid_out_config_args = add_args_output_config(parser)
    valid_generic_args = add_generic_args(parser)
    valid_time_window_args = add_args_time_window(parser)
    valid_csv_args = add_args(parser)

    args = parser.parse_args(argv)

    out_cfg_args = process_args_output_config(args, valid_out_config_args)
    generic_out_cfg_args = process_generic_args(args, valid_generic_args)
    window_args = process_args_time_window(args, valid_time_window_args)
    csv_args = process_args(args, valid_csv_args)

    all_args = {
        **out_cfg_args,
        **generic_out_cfg_args,
        **csv_args,
    }

    execute(args.input, window_args=window_args, **all_args)


if __name__ == "__main__":
    main()
