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

from .importer import RocpdImportData
from .query import export_sqlite_query
from .time_window import apply_time_window
from . import output_config
from . import libpyrocpd


def write_sql_query_to_csv(
    connection: RocpdImportData,
    query,
    output_path,
    output_file,
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
    file_prefix = output_file + "_" if output_file else ""
    file_postfix = "_" + postfix if postfix else ""
    export_path = os.path.join(output_path, f"{file_prefix}{filename}{file_postfix}.csv")
    export_sqlite_query(connection, query, export_format="csv", export_path=export_path)


def write_agent_info_csv(importData, config) -> None:

    # Define mapping of output column name to JSON key
    json_keys = {
        "Node_Id": "node_id",
        "Logical_Node_Id": "logical_node_id",
        "Cpu_Cores_Count": "cpu_cores_count",
        "Simd_Count": "simd_count",
        "Cpu_Core_Id_Base": "cpu_core_id_base",
        "Simd_Id_Base": "simd_id_base",
        "Max_Waves_Per_Simd": "max_waves_per_simd",
        "Lds_Size_In_Kb": "lds_size_in_kb",
        "Gds_Size_In_Kb": "gds_size_in_kb",
        "Num_Gws": "num_gws",
        "Wave_Front_Size": "wave_front_size",
        "Num_Xcc": "num_xcc",
        "Cu_Count": "cu_count",
        "Array_Count": "array_count",
        "Num_Shader_Banks": "num_shader_banks",
        "Simd_Arrays_Per_Engine": "simd_arrays_per_engine",
        "Cu_Per_Simd_Array": "cu_per_simd_array",
        "Simd_Per_Cu": "simd_per_cu",
        "Max_Slots_Scratch_Cu": "max_slots_scratch_cu",
        "Gfx_Target_Version": "gfx_target_version",
        "Vendor_Id": "vendor_id",
        "Device_Id": "device_id",
        "Location_Id": "location_id",
        "Domain": "domain",
        "Drm_Render_Minor": "drm_render_minor",
        "Num_Sdma_Engines": "num_sdma_engines",
        "Num_Sdma_Xgmi_Engines": "num_sdma_xgmi_engines",
        "Num_Sdma_Queues_Per_Engine": "num_sdma_queues_per_engine",
        "Num_Cp_Queues": "num_cp_queues",
        "Max_Engine_Clk_Ccompute": "max_engine_clk_ccompute",
        "Max_Engine_Clk_Fcompute": "max_engine_clk_fcompute",
        "Sdma_Fw_Version": "sdma_fw_version.uCodeSDMA",
        "Fw_Version": "fw_version.uCode",
        "Cu_per_engine": "cu_per_engine",
        "Max_Waves_Per_Cu": "max_waves_per_cu",
        "Workgroup_Max_Size": "workgroup_max_size",
        "Family_Id": "family_id",
        "Grid_Max_Size": "grid_max_size",
        "Local_Mem_Size": "local_mem_size",
        "Hive_Id": "hive_id",
        "Gpu_Id": "gpu_id",
        "Workgroup_Max_Dim_X": "workgroup_max_dim.x",
        "Workgroup_Max_Dim_Y": "workgroup_max_dim.y",
        "Workgroup_Max_Dim_Z": "workgroup_max_dim.z",
        "Grid_Max_Dim_X": "grid_max_dim.x",
        "Grid_Max_Dim_Y": "grid_max_dim.y",
        "Grid_Max_Dim_Z": "grid_max_dim.z",
        "Vendor_Name": "vendor_name",
        "Product_Name": "product_name",
    }

    # Build SELECT clause for json_extract columns
    select_json = [
        f"json_extract(extdata, '$.{json_key}') AS {col_name}"
        for col_name, json_key in json_keys.items()
    ]

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
    select_fixed = [
        "guid AS Guid",
        "type AS Agent_Type",
        "name AS Name",
        "model_name AS Model_Name",
    ]

    # to mimic the previous order
    select_clause = (
        select_fixed[:1]
        + select_json[:2]
        + select_fixed[1:2]
        + select_json[2:33]
        + select_capability
        + select_json[33:47]
        + select_fixed[2:3]
        + select_json[47:]
        + select_fixed[3:4]
    )

    select_clause = ",\n    ".join(select_clause)

    query = f"""
        SELECT
            {select_clause}
        FROM "rocpd_info_agent"
    """

    write_sql_query_to_csv(
        importData, query, config.output_path, config.output_file, "agent_info", ""
    )


def write_kernel_csv(importData, config) -> None:

    if config.agent_index_value == libpyrocpd.agent_indexing.node:  # absolute
        agent_id = "'Agent ' || agent_abs_index"
    elif (
        config.agent_index_value == libpyrocpd.agent_indexing.logical_node
    ):  # relative (default)
        agent_id = "'Agent ' || agent_log_index"
    elif (
        config.agent_index_value == libpyrocpd.agent_indexing.logical_node_type
    ):  # type-relative
        agent_id = "agent_type || ' ' || agent_type_index"
    else:
        agent_id = ""

    if config.kernel_rename:
        kernel_name = "region"
    else:
        kernel_name = "name"

    query = f"""
        SELECT
            guid AS Guid,
            'KERNEL_DISPATCH' AS Kind,
            {agent_id} AS Agent_Id,
            queue_id AS Queue_Id,
            stream_id AS Stream_Id,
            tid AS Thread_Id,
            dispatch_id AS Dispatch_Id,
            kernel_Id AS Kernel_Id,
            {kernel_name} AS Kernel_Name,
            stack_id AS Correlation_Id,
            start AS Start_Timestamp,
            end AS End_Timestamp,
            lds_size AS LDS_Block_Size,
            scratch_size AS Scratch_Size,
            vgpr_count AS VGPR_Count,
            accum_vgpr_count AS Accum_VGPR_Count,
            sgpr_count AS SGPR_Count,
            workgroup_x AS Workgroup_Size_X,
            workgroup_y AS Workgroup_Size_Y,
            workgroup_z AS Workgroup_Size_Z,
            grid_x AS Grid_Size_X,
            grid_y AS Grid_Size_Y,
            grid_z AS Grid_Size_Z
        FROM "kernels"
        ORDER BY
            guid ASC, start ASC, end DESC
    """
    write_sql_query_to_csv(
        importData, query, config.output_path, config.output_file, "kernel"
    )


def write_memory_copy_csv(importData, config) -> None:

    if config.agent_index_value == libpyrocpd.agent_indexing.node:  # absolute
        src_agent_id = "'Agent ' || src_agent_abs_index"
        dst_agent_id = "'Agent ' || dst_agent_abs_index"
    elif (
        config.agent_index_value == libpyrocpd.agent_indexing.logical_node
    ):  # relative (default)
        src_agent_id = "'Agent ' || src_agent_log_index"
        dst_agent_id = "'Agent ' || dst_agent_log_index"
    elif (
        config.agent_index_value == libpyrocpd.agent_indexing.logical_node_type
    ):  # type-relative
        src_agent_id = "src_agent_type || ' ' || src_agent_type_index"
        dst_agent_id = "dst_agent_type || ' ' || dst_agent_type_index"
    else:
        src_agent_id = ""
        dst_agent_id = ""

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
    write_sql_query_to_csv(
        importData, query, config.output_path, config.output_file, "memory_copy"
    )


def write_memory_allocation_csv(importData, config) -> None:

    if config.agent_index_value == libpyrocpd.agent_indexing.node:  # absolute
        agent_id = "'Agent ' || agent_abs_index"
    elif (
        config.agent_index_value == libpyrocpd.agent_indexing.logical_node
    ):  # relative (default)
        agent_id = "'Agent ' || agent_log_index"
    elif (
        config.agent_index_value == libpyrocpd.agent_indexing.logical_node_type
    ):  # type-relative
        agent_id = "agent_type || ' ' || agent_type_index"
    else:
        agent_id = ""

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
    write_sql_query_to_csv(
        importData, query, config.output_path, config.output_file, "memory_allocation"
    )


def write_counters_csv(importData, config) -> None:

    if config.agent_index_value == libpyrocpd.agent_indexing.node:  # absolute
        agent_id = "'Agent ' || agent_abs_index"
    elif (
        config.agent_index_value == libpyrocpd.agent_indexing.logical_node
    ):  # relative (default)
        agent_id = "'Agent ' || agent_log_index"
    elif (
        config.agent_index_value == libpyrocpd.agent_indexing.logical_node_type
    ):  # type-relative
        agent_id = "agent_type || ' ' || agent_type_index"
    else:
        agent_id = ""

    query = f"""
        SELECT
            guid AS Guid,
            stack_id AS Correlation_Id,
            dispatch_id AS Dispatch_Id,
            {agent_id} AS Agent_Id,
            queue_id AS Queue_Id,
            pid AS Process_Id,
            tid AS Thread_Id,
            grid_size AS Grid_Size,
            kernel_id AS Kernel_Id,
            kernel_name AS Kernel_Name,
            workgroup_size AS Workgroup_Size,
            lds_block_size AS LDS_Block_Size,
            scratch_size AS Scratch_Size,
            vgpr_count AS VGPR_Count,
            accum_vgpr_count AS Accum_VGPR_Count,
            sgpr_count AS SGPR_Count,
            counter_name AS Counter_Name,
            value AS Counter_Value,
            start AS Start_Timestamp,
            end AS End_Timestamp
        FROM "counters_collection"
        ORDER BY
            guid ASC, start ASC, end DESC
    """
    write_sql_query_to_csv(
        importData, query, config.output_path, config.output_file, "counter_collection"
    )


def write_scratch_memory_csv(importData, config) -> None:

    if config.agent_index_value == libpyrocpd.agent_indexing.node:  # absolute
        agent_id = "'Agent ' || agent_abs_index"
    elif (
        config.agent_index_value == libpyrocpd.agent_indexing.logical_node
    ):  # relative (default)
        agent_id = "'Agent ' || agent_log_index"
    elif (
        config.agent_index_value == libpyrocpd.agent_indexing.logical_node_type
    ):  # type-relative
        agent_id = "agent_type || ' ' || agent_type_index"
    else:
        agent_id = ""

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
    write_sql_query_to_csv(
        importData, query, config.output_path, config.output_file, "scratch_memory"
    )


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
    write_sql_query_to_csv(
        importData, query, config.output_path, config.output_file, "region"
    )


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

    return []


def process_args(args, valid_args):
    ret = {}
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
