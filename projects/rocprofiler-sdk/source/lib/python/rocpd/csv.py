#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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
    connection: RocpdImportData, query, output_path, output_file, filename=""
) -> None:
    """Write the contents of a SQL query to a CSV file in the specified output path."""

    # call query module to export to csv
    file_prefix = output_file + "_" if output_file else ""
    export_path = os.path.join(output_path, f"{file_prefix}{filename}_trace.csv")
    export_sqlite_query(connection, query, export_format="csv", export_path=export_path)

def write_agent_info_csv(
    importData, config
) -> None:

    query = """
        SELECT
            guid AS Guid,
            json_extract(extdata, '$.node_id') AS Node_Id,
            json_extract(extdata, '$.logical_node_id') AS Logical_Node_Id,
            type AS Agent_Type,
            json_extract(extdata, '$.cpu_cores_count') AS Cpu_Cores_Count,
            json_extract(extdata, '$.simd_count') AS Simd_Count,
            json_extract(extdata, '$.cpu_core_id_base') AS Cpu_Core_Id_Base,
            json_extract(extdata, '$.simd_id_base') AS Simd_Id_Base,
            json_extract(extdata, '$.max_waves_per_simd') AS Max_Waves_Per_Simd,
            json_extract(extdata, '$.lds_size_in_kb') AS Lds_Size_In_Kb,
            json_extract(extdata, '$.gds_size_in_kb') AS Gds_Size_In_Kb,
            json_extract(extdata, '$.num_gws') AS Num_Gws,
            json_extract(extdata, '$.wave_front_size') AS Wave_Front_Size,
            json_extract(extdata, '$.num_xcc') AS Num_Xcc,
            json_extract(extdata, '$.cu_count') AS Cu_Count,
            json_extract(extdata, '$.array_count') AS Array_Count,
            json_extract(extdata, '$.num_shader_banks') AS Num_Shader_Banks,
            json_extract(extdata, '$.simd_arrays_per_engine') AS Simd_Arrays_Per_Engine,
            json_extract(extdata, '$.cu_per_simd_array') AS Cu_Per_Simd_Array,
            json_extract(extdata, '$.simd_per_cu') AS Simd_Per_Cu,
            json_extract(extdata, '$.max_slots_scratch_cu') AS Max_Slots_Scratch_Cu,
            json_extract(extdata, '$.gfx_target_version') AS Gfx_Target_Version,
            json_extract(extdata, '$.vendor_id') AS Vendor_Id,
            json_extract(extdata, '$.device_id') AS Device_Id,
            json_extract(extdata, '$.location_id') AS Location_Id,
            json_extract(extdata, '$.domain') AS Domain,
            json_extract(extdata, '$.drm_render_minor') AS Drm_Render_Minor,
            json_extract(extdata, '$.num_sdma_engines') AS Num_Sdma_Engines,
            json_extract(extdata, '$.num_sdma_xgmi_engines') AS Num_Sdma_Xgmi_Engines,
            json_extract(extdata, '$.num_sdma_queues_per_engine') AS Num_Sdma_Queues_Per_Engine,
            json_extract(extdata, '$.num_cp_queues') AS Num_Cp_Queues,
            json_extract(extdata, '$.max_engine_clk_ccompute') AS Max_Engine_Clk_Ccompute,
            json_extract(extdata, '$.max_engine_clk_fcompute')  AS Max_Engine_Clk_Fcompute,
            json_extract(extdata, '$.sdma_fw_version.uCodeSDMA') AS Sdma_Fw_Version,
            json_extract(extdata, '$.fw_version.uCode') AS Fw_Version,
            (COALESCE(json_extract(extdata, '$.capability.HotPluggable'), 0) << 0x0) |
            (COALESCE(json_extract(extdata, '$.capability.HSAMMUPresent'), 0) << 0x1) |
            (COALESCE(json_extract(extdata, '$.capability.SharedWithGraphics'), 0) << 0x2) |
            (COALESCE(json_extract(extdata, '$.capability.QueueSizePowerOfTwo'), 0) << 0x3) |
            (COALESCE(json_extract(extdata, '$.capability.QueueSize32bit'), 0) << 0x4) |
            (COALESCE(json_extract(extdata, '$.capability.QueueIdleEvent'), 0) << 0x5) |
            (COALESCE(json_extract(extdata, '$.capability.VALimit'), 0) << 0x6) |
            (COALESCE(json_extract(extdata, '$.capability.WatchPointsSupported'), 0) << 0x7) |
            ((COALESCE(json_extract(extdata, '$.capability.WatchPointsTotalBits'), 0) & 0xF) << 0x8) |
            ((COALESCE(json_extract(extdata, '$.capability.DoorbellType'), 0) & 0x3) << 0xC) |
            (COALESCE(json_extract(extdata, '$.capability.AQLQueueDoubleMap'), 0) << 0xE) |
            (COALESCE(json_extract(extdata, '$.capability.DebugTrapSupported'), 0) << 0xF) |
            (COALESCE(json_extract(extdata, '$.capability.WaveLaunchTrapOverrideSupported'), 0) << 0x10) |
            (COALESCE(json_extract(extdata, '$.capability.WaveLaunchModeSupported'), 0) << 0x11) |
            (COALESCE(json_extract(extdata, '$.capability.PreciseMemoryOperationsSupported'), 0) << 0x12) |
            (COALESCE(json_extract(extdata, '$.capability.DEPRECATED_SRAM_EDCSupport'), 0) << 0x13) |
            (COALESCE(json_extract(extdata, '$.capability.Mem_EDCSupport'), 0) << 0x14) |
            (COALESCE(json_extract(extdata, '$.capability.RASEventNotify'), 0) << 0x15) |
            ((COALESCE(json_extract(extdata, '$.capability.ASICRevision'), 0) & 0xF) << 0x16) |
            (COALESCE(json_extract(extdata, '$.capability.SRAM_EDCSupport'), 0) << 0x1A) |
            (COALESCE(json_extract(extdata, '$.capability.SVMAPISupported'), 0) << 0x1B) |
            (COALESCE(json_extract(extdata, '$.capability.CoherentHostAccess'), 0) << 0x1C) |
            (COALESCE(json_extract(extdata, '$.capability.DebugSupportedFirmware'), 0) << 0x1D) |
            (COALESCE(json_extract(extdata, '$.capability.PreciseALUOperationsSupported'), 0) << 0x1E) |
            (COALESCE(json_extract(extdata, '$.capability.PerQueueResetSupported'), 0) << 0x1F) AS Capability,
            json_extract(extdata, '$.cu_per_engine') AS Cu_Per_Engine,
            json_extract(extdata, '$.max_waves_per_cu') AS Max_Waves_Per_Cu,
            json_extract(extdata, '$.workgroup_max_size') AS Workgroup_Max_Size,
            json_extract(extdata, '$.family_id') AS Family_Id,
            json_extract(extdata, '$.grid_max_size') AS Grid_Max_Size,
            json_extract(extdata, '$.local_mem_size') AS Local_Mem_Size,
            json_extract(extdata, '$.hive_id') AS Hive_Id,
            json_extract(extdata, '$.gpu_id') AS Gpu_Id,
            json_extract(extdata, '$.workgroup_max_dim.x') AS Workgroup_Max_Dim_X,
            json_extract(extdata, '$.workgroup_max_dim.y') AS Workgroup_Max_Dim_Y,
            json_extract(extdata, '$.workgroup_max_dim.z') AS Workgroup_Max_Dim_Z,
            json_extract(extdata, '$.grid_max_dim.x') AS Grid_Max_Dim_X,
            json_extract(extdata, '$.grid_max_dim.y') AS Grid_Max_Dim_Y,
            json_extract(extdata, '$.grid_max_dim.z') AS Grid_Max_Dim_Z,
            name AS Name,
            json_extract(extdata, '$.vendor_name') AS Vendor_Name,
            json_extract(extdata, '$.product_name') AS Product_Name,
            model_name AS Model_Name
        FROM "rocpd_info_agent"
    """
    write_sql_query_to_csv(importData, query, config.output_path, config.output_file, "agent_info")

def write_kernel_csv(
    importData, config
) -> None:

    if config.agent_index_value == libpyrocpd.agent_indexing.node : # absolute
        agent_id = "'Agent ' || agent_abs_index"
    elif config.agent_index_value == libpyrocpd.agent_indexing.logical_node: # relative (default)
        agent_id = "'Agent ' || agent_log_index"
    elif config.agent_index_value == libpyrocpd.agent_indexing.logical_node_type: #  type-relative
        agent_id = "agent_type || ' ' || agent_type_index"
    else:
        agent_id = ""

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
            name AS Kernel_Name,
            stack_id AS Correlation_Id,
            start AS Start_Timestamp,
            end AS End_Timestamp,
            scratch_size AS Private_Segment_Size,
            lds_size AS Group_Segment_Size,
            workgroup_x AS Workgroup_Size_X,
            workgroup_y AS Workgroup_Size_Y,
            workgroup_z AS Workgroup_Size_Z,
            grid_x AS Grid_Size_X,
            grid_y AS Grid_Size_Y,
            grid_z AS Grid_Size_Z
        FROM "kernels"
        ORDER BY
            start ASC, end DESC
    """
    write_sql_query_to_csv(importData, query, config.output_path, config.output_file, "kernel")

def write_memory_copy_csv(
    importData, config
) -> None:

    if config.agent_index_value == libpyrocpd.agent_indexing.node : # absolute
        src_agent_id = "'Agent ' || src_agent_abs_index"
        dst_agent_id = "'Agent ' || dst_agent_abs_index"
    elif config.agent_index_value == libpyrocpd.agent_indexing.logical_node: # relative (default)
        src_agent_id = "'Agent ' || src_agent_log_index"
        dst_agent_id = "'Agent ' || dst_agent_log_index"
    elif config.agent_index_value == libpyrocpd.agent_indexing.logical_node_type: #  type-relative
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
            {src_agent_id}  AS Source_Agent_Id,
            {dst_agent_id}  AS Destination_Agent_Id,
            stack_id AS Correlation_Id,
            start AS Start_Timestamp,
            end AS End_Timestamp
        FROM "memory_copies"
        ORDER BY
            start ASC, end DESC
    """
    write_sql_query_to_csv(importData, query, config.output_path, config.output_file, "memory_copy")

def write_memory_allocation_csv(
    importData, config
) -> None:

    if config.agent_index_value == libpyrocpd.agent_indexing.node : # absolute
        agent_id = "'Agent ' || agent_abs_index"
    elif config.agent_index_value == libpyrocpd.agent_indexing.logical_node: # relative (default)
        agent_id = "'Agent ' || agent_log_index"
    elif config.agent_index_value == libpyrocpd.agent_indexing.logical_node_type: #  type-relative
        agent_id = "agent_type || ' ' || agent_type_index"
    else:
        agent_id = ""

    query = f"""
        SELECT
            guid AS Guid,
            'MEMORY_ALLOCATION' AS Kind,
            'MEMORY_ALLOCATION_' || type AS Operation,
            CASE
                WHEN type != "FREE"
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
            start ASC, end DESC
    """
    write_sql_query_to_csv(importData, query, config.output_path, config.output_file, "memory_allocation")

def write_hip_api_csv(
    importData, config
) -> None:

    query = """
        SELECT
            guid AS Guid,
            category AS Domain,
            name AS Function,
            pid AS Process_Id,
            tid AS Thread_Id,
            stack_id AS Correlation_Id,
            start AS Start_Timestamp,
            end AS End_Timestamp
        FROM "regions"
        WHERE
            category LIKE 'HIP_%'
        ORDER BY
            start ASC, end DESC
    """
    write_sql_query_to_csv(importData, query, config.output_path, config.output_file, "hip_api")

def write_hsa_api_csv(
    importData, config
) -> None:

    query = """
        SELECT
            guid AS Guid,
            category AS Domain,
            name AS Function,
            pid AS Process_Id,
            tid AS Thread_Id,
            stack_id AS Correlation_Id,
            start AS Start_Timestamp,
            end AS End_Timestamp
        FROM "regions"
        WHERE
            category LIKE 'HSA_%'
        ORDER BY
            start ASC, end DESC
    """
    write_sql_query_to_csv(importData, query, config.output_path, config.output_file, "hsa_api")

def write_marker_api_csv(
    importData, config
) -> None:

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
        WHERE
            category LIKE 'MARKER_%'
        ORDER BY
            start ASC, end DESC
    """
    write_sql_query_to_csv(importData, query, config.output_path, config.output_file, "marker_api")

def write_counters_csv(
    importData, config
) -> None:

    if config.agent_index_value == libpyrocpd.agent_indexing.node : # absolute
        agent_id = "'Agent ' || agent_abs_index"
    elif config.agent_index_value == libpyrocpd.agent_indexing.logical_node: # relative (default)
        agent_id = "'Agent ' || agent_log_index"
    elif config.agent_index_value == libpyrocpd.agent_indexing.logical_node_type: #  type-relative
        agent_id = "agent_type || ' ' || agent_type_index"
    else:
        agent_id = ""

    query = f"""
        SELECT
            guid AS Pid,
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
            start ASC, end DESC
    """
    write_sql_query_to_csv(importData, query, config.output_path, config.output_file, "counter_collection")

def write_scratch_memory_csv(
    importData, config
) -> None:

    if config.agent_index_value == libpyrocpd.agent_indexing.node : # absolute
        agent_id = "'Agent ' || agent_abs_index"
    elif config.agent_index_value == libpyrocpd.agent_indexing.logical_node: # relative (default)
        agent_id = "'Agent ' || agent_log_index"
    elif config.agent_index_value == libpyrocpd.agent_indexing.logical_node_type: #  type-relative
        agent_id = "agent_type || ' ' || agent_type_index"
    else:
        agent_id = ""

    query = f"""
        SELECT
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
            start ASC, end DESC
    """
    write_sql_query_to_csv(importData, query, config.output_path, config.output_file, "scratch_memory")

def write_rccl_api_csv(
    importData, config
) -> None:

    query = """
         SELECT
            guid AS Guid,
            category AS Domain,
            name AS Function,
            pid AS Process_Id,
            tid AS Thread_Id,
            stack_id AS Correlation_Id,
            start AS Start_Timestamp,
            end AS End_Timestamp
        FROM "regions"
        WHERE
            category LIKE 'RCCL_%'
        ORDER BY
            start ASC, end DESC
    """
    write_sql_query_to_csv(importData, query, config.output_path, config.output_file, "rccl_api")

def write_rocdecode_api_csv(
    importData, config
) -> None:

    query = """
         SELECT
            guid AS Guid,
            category AS Domain,
            name AS Function,
            pid AS Process_Id,
            tid AS Thread_Id,
            stack_id AS Correlation_Id,
            start AS Start_Timestamp,
            end AS End_Timestamp
        FROM "regions"
        WHERE
            category LIKE 'ROCDECODE_%'
        ORDER BY
            start ASC, end DESC
    """
    write_sql_query_to_csv(importData, query, config.output_path, config.output_file, "rocdecode_api")

def write_rocjpeg_api_csv(
    importData, config
) -> None:

    query = """
        SELECT
            guid AS Guid,
            category AS Domain,
            name AS Function,
            pid AS Process_Id,
            tid AS Thread_Id,
            stack_id AS Correlation_Id,
            start AS Start_Timestamp,
            end AS End_Timestamp
        FROM "regions"
        WHERE
            category LIKE 'ROCJPEG_%'
        ORDER BY
            start ASC, end DESC
    """
    write_sql_query_to_csv(importData, query, config.output_path, config.output_file, "rocjpeg_api")

def write_csv(importData, config):

    write_agent_info_csv(importData, config)
    write_kernel_csv(importData, config)
    write_memory_copy_csv(importData, config)
    write_memory_allocation_csv(importData, config)
    write_hip_api_csv(importData, config)
    write_hsa_api_csv(importData, config)
    write_marker_api_csv(importData, config)
    write_counters_csv(importData, config)
    write_scratch_memory_csv(importData, config)
    write_rccl_api_csv(importData, config)
    write_rocdecode_api_csv(importData, config)
    write_rocjpeg_api_csv(importData, config)

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
