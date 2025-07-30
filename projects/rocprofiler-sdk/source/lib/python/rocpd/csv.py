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
    connection: RocpdImportData, query, output_path, filename=""
) -> None:
    """Write the contents of a SQL query to a CSV file in the specified output path."""

    # call query module to export to csv
    export_path = os.path.join(output_path, f"{filename}.csv")
    export_sqlite_query(connection, query, export_format="csv", export_path=export_path)

def write_agent_info_csv(
    importData, output_path
) -> None:

    query = """
        SELECT
            A.guid AS Guid,
            json_extract(A.extdata, '$.node_id') AS Node_Id,
            json_extract(A.extdata, '$.logical_node_id') AS Logical_Node_Id,
            A.type AS Agent_Type,
            json_extract(A.extdata, '$.cpu_cores_count') AS Cpu_Cores_Count,
            json_extract(A.extdata, '$.simd_count') AS Simd_Count,
            json_extract(A.extdata, '$.cpu_core_id_base') AS Cpu_Core_Id_Base,
            json_extract(A.extdata, '$.simd_id_base') AS Simd_Id_Base,
            json_extract(A.extdata, '$.max_waves_per_simd') AS Max_Waves_Per_Simd,
            json_extract(A.extdata, '$.lds_size_in_kb') AS Lds_Size_In_Kb,
            json_extract(A.extdata, '$.gds_size_in_kb') AS Gds_Size_In_Kb,
            json_extract(A.extdata, '$.num_gws') AS Num_Gws,
            json_extract(A.extdata, '$.wave_front_size') AS Wave_Front_Size,
            json_extract(A.extdata, '$.num_xcc') AS Num_Xcc,
            json_extract(A.extdata, '$.cu_count') AS Cu_Count,
            json_extract(A.extdata, '$.array_count') AS Array_Count,
            json_extract(A.extdata, '$.num_shader_banks') AS Num_Shader_Banks,
            json_extract(A.extdata, '$.simd_arrays_per_engine') AS Simd_Arrays_Per_Engine,
            json_extract(A.extdata, '$.cu_per_simd_array') AS Cu_Per_Simd_Array,
            json_extract(A.extdata, '$.simd_per_cu') AS Simd_Per_Cu,
            json_extract(A.extdata, '$.max_slots_scratch_cu') AS Max_Slots_Scratch_Cu,
            json_extract(A.extdata, '$.gfx_target_version') AS Gfx_Target_Version,
            json_extract(A.extdata, '$.vendor_id') AS Vendor_Id,
            json_extract(A.extdata, '$.device_id') AS Device_Id,
            json_extract(A.extdata, '$.location_id') AS Location_Id,
            json_extract(A.extdata, '$.domain') AS Domain,
            json_extract(A.extdata, '$.drm_render_minor') AS Drm_Render_Minor,
            json_extract(A.extdata, '$.num_sdma_engines') AS Num_Sdma_Engines,
            json_extract(A.extdata, '$.num_sdma_xgmi_engines') AS Num_Sdma_Xgmi_Engines,
            json_extract(A.extdata, '$.num_sdma_queues_per_engine') AS Num_Sdma_Queues_Per_Engine,
            json_extract(A.extdata, '$.num_cp_queues') AS Num_Cp_Queues,
            json_extract(A.extdata, '$.max_engine_clk_ccompute') AS Max_Engine_Clk_Ccompute,
            json_extract(A.extdata, '$.max_engine_clk_fcompute')  AS Max_Engine_Clk_Fcompute,
            json_extract(A.extdata, '$.sdma_fw_version.uCodeSDMA') AS Sdma_Fw_Version,
            json_extract(A.extdata, '$.fw_version.uCode') AS Fw_Version,
            (COALESCE(json_extract(A.extdata, '$.capability.HotPluggable'), 0) << 0x0) |
            (COALESCE(json_extract(A.extdata, '$.capability.HSAMMUPresent'), 0) << 0x1) |
            (COALESCE(json_extract(A.extdata, '$.capability.SharedWithGraphics'), 0) << 0x2) |
            (COALESCE(json_extract(A.extdata, '$.capability.QueueSizePowerOfTwo'), 0) << 0x3) |
            (COALESCE(json_extract(A.extdata, '$.capability.QueueSize32bit'), 0) << 0x4) |
            (COALESCE(json_extract(A.extdata, '$.capability.QueueIdleEvent'), 0) << 0x5) |
            (COALESCE(json_extract(A.extdata, '$.capability.VALimit'), 0) << 0x6) |
            (COALESCE(json_extract(A.extdata, '$.capability.WatchPointsSupported'), 0) << 0x7) |
            ((COALESCE(json_extract(A.extdata, '$.capability.WatchPointsTotalBits'), 0) & 0xF) << 0x8) |
            ((COALESCE(json_extract(A.extdata, '$.capability.DoorbellType'), 0) & 0x3) << 0xC) |
            (COALESCE(json_extract(A.extdata, '$.capability.AQLQueueDoubleMap'), 0) << 0xE) |
            (COALESCE(json_extract(A.extdata, '$.capability.DebugTrapSupported'), 0) << 0xF) |
            (COALESCE(json_extract(A.extdata, '$.capability.WaveLaunchTrapOverrideSupported'), 0) << 0x10) |
            (COALESCE(json_extract(A.extdata, '$.capability.WaveLaunchModeSupported'), 0) << 0x11) |
            (COALESCE(json_extract(A.extdata, '$.capability.PreciseMemoryOperationsSupported'), 0) << 0x12) |
            (COALESCE(json_extract(A.extdata, '$.capability.DEPRECATED_SRAM_EDCSupport'), 0) << 0x13) |
            (COALESCE(json_extract(A.extdata, '$.capability.Mem_EDCSupport'), 0) << 0x14) |
            (COALESCE(json_extract(A.extdata, '$.capability.RASEventNotify'), 0) << 0x15) |
            ((COALESCE(json_extract(A.extdata, '$.capability.ASICRevision'), 0) & 0xF) << 0x16) |
            (COALESCE(json_extract(A.extdata, '$.capability.SRAM_EDCSupport'), 0) << 0x1A) |
            (COALESCE(json_extract(A.extdata, '$.capability.SVMAPISupported'), 0) << 0x1B) |
            (COALESCE(json_extract(A.extdata, '$.capability.CoherentHostAccess'), 0) << 0x1C) |
            (COALESCE(json_extract(A.extdata, '$.capability.DebugSupportedFirmware'), 0) << 0x1D) |
            (COALESCE(json_extract(A.extdata, '$.capability.PreciseALUOperationsSupported'), 0) << 0x1E) |
            (COALESCE(json_extract(A.extdata, '$.capability.PerQueueResetSupported'), 0) << 0x1F) AS Capability,
            json_extract(A.extdata, '$.cu_per_engine') AS Cu_Per_Engine,
            json_extract(A.extdata, '$.max_waves_per_cu') AS Max_Waves_Per_Cu,
            json_extract(A.extdata, '$.workgroup_max_size') AS Workgroup_Max_Size,
            json_extract(A.extdata, '$.family_id') AS Family_Id,
            json_extract(A.extdata, '$.grid_max_size') AS Grid_Max_Size,
            json_extract(A.extdata, '$.local_mem_size') AS Local_Mem_Size,
            json_extract(A.extdata, '$.hive_id') AS Hive_Id,
            json_extract(A.extdata, '$.gpu_id') AS Gpu_Id,
            json_extract(A.extdata, '$.workgroup_max_dim.x') AS Workgroup_Max_Dim_X,
            json_extract(A.extdata, '$.workgroup_max_dim.y') AS Workgroup_Max_Dim_Y,
            json_extract(A.extdata, '$.workgroup_max_dim.z') AS Workgroup_Max_Dim_Z,
            json_extract(A.extdata, '$.grid_max_dim.x') AS Grid_Max_Dim_X,
            json_extract(A.extdata, '$.grid_max_dim.y') AS Grid_Max_Dim_Y,
            json_extract(A.extdata, '$.grid_max_dim.z') AS Grid_Max_Dim_Z,
            A.name AS Name,
            json_extract(A.extdata, '$.vendor_name') AS Vendor_Name,
            json_extract(A.extdata, '$.product_name') AS Product_Name,
            A.model_name AS Model_Name
        FROM "rocpd_info_node" AS N
        INNER JOIN rocpd_info_agent as A
            ON A.guid = N.guid
            AND A.nid = N.id
    """
    write_sql_query_to_csv(importData, query, output_path, "out_agent_info")

def write_kernel_csv(
    importData, output_path
) -> None:

    query = """
        SELECT
            K.guid AS Guid,
            'KERNEL_DISPATCH' AS Kind,
            'Agent ' || K.agent_log_index AS Agent_Id,
            K.queue_id AS Queue_Id,
            K.stream_id AS Stream_Id,
            K.tid AS Thread_Id,
            K.dispatch_id AS Dispatch_Id,
            K.kernel_Id AS Kernel_Id,
            K.name AS Kernel_Name,
            K.stack_id AS Correlation_Id,
            K.start AS Start_Timestamp,
            K.end AS End_Timestamp,
            K.scratch_size AS Private_Segment_Size,
            K.lds_size AS Group_Segment_Size,
            K.workgroup_x AS Workgroup_Size_X,
            K.workgroup_y AS Workgroup_Size_Y,
            K.workgroup_z AS Workgroup_Size_Z,
            K.grid_x AS Grid_Size_X,
            K.grid_y AS Grid_Size_Y,
            K.grid_z AS Grid_Size_Z
        FROM "rocpd_info_node" AS N
        INNER JOIN rocpd_info_process as P
            ON P.guid = N.guid
            AND P.nid = N.id
        INNER JOIN kernels AS K
            ON K.guid = P.guid
            AND K.nid = P.nid
            AND K.pid = P.pid
        ORDER BY
            K.start ASC, K.end DESC
    """
    write_sql_query_to_csv(importData, query, output_path, "out_kernel_trace")

def write_memory_copy_csv(
    importData, output_path
) -> None:

    query = """
        SELECT
            M.guid AS Guid,
            'MEMORY_COPY' AS Kind,
            M.name AS Direction,
            M.stream_id AS Stream_Id,
            'Agent ' || M.src_agent_log_index AS Source_Agent_Id,
            'Agent ' || M.dst_agent_log_index AS Destination_Agent_Id,
            M.stack_id AS Correlation_Id,
            M.start AS Start_Timestamp,
            M.end AS End_Timestamp

        FROM "rocpd_info_node" AS N
        INNER JOIN rocpd_info_process as P
            ON P.guid = N.guid
            AND P.nid = N.id
        INNER JOIN memory_copies AS M
            ON M.guid = P.guid
            AND M.nid = P.nid
            AND M.pid = P.pid
        ORDER BY
            M.start ASC, M.end DESC
    """
    write_sql_query_to_csv(importData, query, output_path, "out_memory_copy_trace")

def write_memory_allocation_csv(
    importData, output_path
) -> None:

    query = """
        SELECT
            A.guid AS Guid,
            'MEMORY_ALLOCATION' AS Kind,
            'MEMORY_ALLOCATION_' || A.type AS Operation,
            CASE
                WHEN A.type != "FREE"
                THEN 'Agent ' || A.agent_log_index
                ELSE '"'
            END AS Agent_Id,
            A.size AS Allocation_Size,
            '0x' || printf('%016X', A.address) AS Address,
            A.stack_id AS Correlation_Id,
            A.start AS Start_Timestamp,
            A.end AS End_Timestamp
        FROM "rocpd_info_node" AS N
        INNER JOIN rocpd_info_process as P
            ON P.guid = N.guid
            AND P.nid = N.id
        INNER JOIN memory_allocations AS A
            ON A.guid = P.guid
            AND A.nid = P.nid
            AND A.pid = P.pid
        ORDER BY
            A.start ASC, A.end DESC
    """
    write_sql_query_to_csv(importData, query, output_path, "out_memory_allocation_trace")

def write_csv(importData, config):

    write_agent_info_csv(importData, config.output_path)
    write_kernel_csv(importData, config.output_path)
    write_memory_copy_csv(importData, config.output_path)
    write_memory_allocation_csv(importData, config.output_path)

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
