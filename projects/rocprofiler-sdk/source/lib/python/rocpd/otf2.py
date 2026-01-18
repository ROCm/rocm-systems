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
import otf2
from otf2.enums import LocationType, LocationGroupType, RegionRole, Paradigm
import shutil
import time
from collections import defaultdict

from .importer import RocpdImportData
from . import output_config


def get_perfetto_category_name(category):
    """Map category names to perfetto category names"""
    category_map = {
        "NONE": "none",
        "HSA_CORE_API": "hsa_api",
        "HSA_AMD_EXT_API": "hsa_api",
        "HSA_IMAGE_EXT_API": "hsa_api",
        "HSA_FINALIZE_EXT_API": "hsa_api",
        "HIP_RUNTIME_API": "hip_api",
        "HIP_COMPILER_API": "hip_api",
        "MARKER_CORE_API": "marker_api",
        "MARKER_CORE_RANGE_API": "marker_api",
        "MARKER_CONTROL_API": "marker_api",
        "MARKER_NAME_API": "marker_api",
        "MEMORY_COPY": "memory_copy",
        "MEMORY_ALLOCATION": "memory_allocation",
        "KERNEL_DISPATCH": "kernel_dispatch",
        "SCRATCH_MEMORY": "scratch_memory",
        "CORRELATION_ID_RETIREMENT": "none",
        "RCCL_API": "rccl_api",
        "OMPT": "openmp",
        "RUNTIME_INITIALIZATION": "none",
        "ROCDECODE_API": "rocdecode_api",
        "ROCJPEG_API": "rocjpeg_api",
        "HIP_STREAM": "hip_api",
        "HIP_RUNTIME_API_EXT": "hip_api",
        "HIP_COMPILER_API_EXT": "hip_api",
        "ROCDECODE_API_EXT": "rocdecode_api",
        "KFD_EVENT_PAGE_MIGRATE": "kfd_events",
        "KFD_EVENT_PAGE_FAULT": "kfd_events",
        "KFD_EVENT_QUEUE": "kfd_events",
        "KFD_EVENT_UNMAP_FROM_GPU": "kfd_events",
        "KFD_EVENT_DROPPED_EVENTS": "kfd_events",
        "KFD_PAGE_MIGRATE": "kfd_events",
        "KFD_PAGE_FAULT": "kfd_events",
        "KFD_QUEUE": "kfd_events",
    }
    return category_map.get(category, "none")


def allocation_level_type_name(level, type):
    if level == "REAL":
        name = "MEMORY"
    elif level == "VIRTUAL":
        name = "MEMORY_VMEM"
    elif level == "SCRATCH":
        name = "SCRATCH_MEMORY"
    else:
        return "UNKNOWN_LEVEL"

    if type == "ALLOC":
        return name + "_ALLOCATE"
    elif type == "FREE":
        return name + "_FREE"
    else:
        return level + "_MEMORY_NONE"

    return name


def write_otf2(importData, config):

    timer_resolution = 1_000_000_000
    trace_dir = getattr(config, "output_path", "./otf_traces")
    trace_file = f"{getattr(config, 'output_file', 'traces')}_results"
    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir)

    if os.path.exists(trace_dir):
        trace_subdir = os.path.join(trace_dir, trace_file)
        if os.path.exists(trace_subdir):
            shutil.rmtree(trace_subdir)

        otf2_file = os.path.join(trace_dir, f"{trace_file}.otf2")
        def_file = os.path.join(trace_dir, f"{trace_file}.def")

        print(f"Writing: {otf2_file}")
        print(f"Writing: {def_file}")
        print(f"Writing: {trace_subdir} directory")

        if os.path.exists(otf2_file):
            os.remove(otf2_file)
        if os.path.exists(def_file):
            os.remove(def_file)
    with otf2.writer.open(
        trace_dir, trace_file, timer_resolution=timer_resolution
    ) as archive:
        conn = getattr(importData, "connection", None)
        if conn is not None:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT MIN(start), MAX(fini) FROM processes;")
                min_start, max_finish = cursor.fetchone()
                with otf2.writer.DefinitionWriter(archive) as global_def_writer:
                    global_offset = min_start
                    duration = max_finish - min_start
                    realtime_timestamp = int(round(time.time() * timer_resolution))
                    global_def_writer.write_clock_properties(
                        timer_resolution, global_offset, duration, realtime_timestamp
                    )
                    perfetto_category = archive.definitions.attribute(
                        name="category", description="tracing category"
                    )
                    memory_copy_attributes = {perfetto_category: "memory_copy"}
                    memory_allocation_attributes = {
                        perfetto_category: "memory_allocation"
                    }
                    kernel_attributes = {perfetto_category: "kernel_dispatch"}
                    kernel_rename = getattr(config, "kernel_rename")
                    agent_index_value = getattr(config, "agent_index_value")

                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT guid, id FROM rocpd_info_node")
                    for row in cursor:
                        guid, nid = row
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT pid, hostname, command FROM processes WHERE guid = ? AND nid = ?",
                            (guid, nid),
                        )
                        for row in cursor:
                            pid, hostname, command = row
                            tree_node = archive.definitions.system_tree_node(
                                name=command, class_name=hostname, parent=None
                            )
                            cpu_location_group = archive.definitions.location_group(
                                name=command,
                                location_group_type=LocationGroupType.PROCESS,
                                system_tree_parent=tree_node,
                            )
                            api_calls = defaultdict(list)
                            memory_copies = defaultdict(list)
                            memory_allocations = defaultdict(list)
                            memory_deallocations = defaultdict(list)
                            memory_unknown = defaultdict(list)
                            kernel_dispatches = defaultdict(list)
                            agents = {}

                            cursor = conn.cursor()
                            cursor.execute(
                                """SELECT tid, start, end, name, category FROM regions
                                WHERE guid = ? AND nid = ? AND pid = ?""",
                                (guid, nid, pid),
                            )
                            for row in cursor:
                                tid, start, end, name, category = row
                                api_calls[tid].append((start, end, name, category))

                            cursor = conn.cursor()
                            cursor.execute(
                                """SELECT tid, dst_agent_abs_index, start, end, name
                                FROM memory_copies WHERE guid = ? AND nid = ?
                                AND pid = ? ORDER BY start ASC""",
                                (guid, nid, pid),
                            )
                            for row in cursor:
                                tid, agent, start, end, name = row
                                memory_copies[(tid, agent)].append((start, end, name))

                            cursor = conn.cursor()
                            cursor.execute(
                                """SELECT tid, agent_abs_index, start, end, level, type
                                FROM memory_allocations WHERE guid = ? AND nid = ?
                                AND pid = ? ORDER BY start ASC""",
                                (guid, nid, pid),
                            )
                            for row in cursor:
                                tid, agent, start, end, level, type = row
                                name = allocation_level_type_name(level, type)
                                if type == "ALLOC":
                                    memory_allocations[(tid, agent)].append(
                                        (start, end, name)
                                    )
                                elif type == "FREE":
                                    memory_deallocations[tid].append((start, end, name))
                                else:
                                    memory_unknown[tid].append((start, end, name))

                            cursor = conn.cursor()
                            cursor.execute(
                                """SELECT tid, agent_abs_index, queue_id,
                                start, end, name, region
                                FROM kernels WHERE guid = ? AND nid = ?
                                AND pid = ? ORDER BY start ASC""",
                                (guid, nid, pid),
                            )
                            for row in cursor:
                                tid, agent, queue, start, end, name, region = row
                                if kernel_rename and region:
                                    kernel_dispatches[(tid, agent, queue)].append(
                                        (start, end, region)
                                    )
                                else:
                                    kernel_dispatches[(tid, agent, queue)].append(
                                        (start, end, name)
                                    )

                            cursor = conn.cursor()
                            cursor.execute(
                                "SELECT id, name, type FROM rocpd_info_agent WHERE guid = ? AND nid = ?",
                                (guid, nid),
                            )
                            for row in cursor:
                                id, name, type = row
                                agents[id] = (
                                    (
                                        f"{type} Agent-{id}"
                                        if agent_index_value
                                        else f"{type} {type}-{id}"
                                    ),
                                    archive.definitions.location_group(
                                        name=name,
                                        location_group_type=LocationGroupType.ACCELERATOR,
                                        system_tree_parent=tree_node,
                                    ),
                                )

                            # Write API Call Events
                            for tid, data in api_calls.items():
                                cpu_location = archive.definitions.location(
                                    name=f"Thread {tid}",
                                    type=LocationType.CPU_THREAD,
                                    group=cpu_location_group,
                                )
                                event_writer = otf2.writer.EventWriter(
                                    archive, cpu_location
                                )
                                events = []
                                for start, end, name, category in data:
                                    attributes = {
                                        perfetto_category: get_perfetto_category_name(
                                            category
                                        )
                                    }

                                    region = archive.definitions.region(
                                        name=name,
                                        region_role=RegionRole.FUNCTION,
                                        paradigm=Paradigm.HIP,
                                    )
                                    events.append((start, "start", region, attributes))
                                    events.append((end, "end", region, None))
                                events.sort(key=lambda x: x[0])
                                for timestamp, event_type, region, attributes in events:
                                    if event_type == "start":
                                        event_writer.enter(
                                            timestamp, region, attributes=attributes
                                        )
                                    else:  # if event_type == "end":
                                        event_writer.leave(timestamp, region)

                            # Write Memory Copy Events
                            for (tid, agent_id), data in memory_copies.items():
                                agent_name, agent_location_group = agents.get(
                                    agent_id, (f"Unknown Agent {agent_id}", None)
                                )
                                memory_copy_location = archive.definitions.location(
                                    name=f"Thread {tid}, Copy to {agent_name}",
                                    type=LocationType.ACCELERATOR_STREAM,
                                    group=agent_location_group,
                                )
                                memory_copy_writer = otf2.writer.EventWriter(
                                    archive, memory_copy_location
                                )
                                memory_copy_events = []
                                for start, end, name in data:
                                    region = archive.definitions.region(
                                        name=name,
                                        region_role=RegionRole.DATA_TRANSFER,
                                        paradigm=Paradigm.HIP,
                                    )
                                    memory_copy_events.append((start, "enter", region))
                                    memory_copy_events.append((end, "leave", region))
                                memory_copy_events.sort(key=lambda x: x[0])
                                for timestamp, event_type, region in memory_copy_events:
                                    if event_type == "enter":
                                        memory_copy_writer.enter(
                                            timestamp,
                                            region,
                                            attributes=memory_copy_attributes,
                                        )
                                    else:  # if event_type == "leave":
                                        memory_copy_writer.leave(timestamp, region)

                            # Write Memory Allocation Events
                            for (tid, agent_id), data in memory_allocations.items():
                                agent_name, agent_location_group = agents.get(
                                    agent_id, (f"Unknown Agent {agent_id}", None)
                                )
                                memory_allocation_location = archive.definitions.location(
                                    name=f"Thread {tid}, Memory Allocate at {agent_name}",
                                    type=LocationType.ACCELERATOR_STREAM,
                                    group=agent_location_group,
                                )
                                memory_allocation_writer = otf2.writer.EventWriter(
                                    archive, memory_allocation_location
                                )
                                memory_allocation_events = []
                                for start, end, name in data:
                                    region = archive.definitions.region(
                                        name=name,
                                        region_role=RegionRole.ALLOCATE,
                                        paradigm=Paradigm.HIP,
                                    )
                                    memory_allocation_events.append(
                                        (start, "enter", region)
                                    )
                                    memory_allocation_events.append(
                                        (end, "leave", region)
                                    )
                                memory_allocation_events.sort(key=lambda x: x[0])
                                for (
                                    timestamp,
                                    event_type,
                                    region,
                                ) in memory_allocation_events:
                                    if event_type == "enter":
                                        memory_allocation_writer.enter(
                                            timestamp,
                                            region,
                                            attributes=memory_allocation_attributes,
                                        )
                                    else:  # if event_type == "leave":
                                        memory_allocation_writer.leave(timestamp, region)

                            # Write Memory Deallocation Events
                            for tid, data in memory_deallocations.items():
                                memory_free_location = archive.definitions.location(
                                    name=f"Thread {tid}, Memory Deallocate (Free)",
                                    type=LocationType.ACCELERATOR_STREAM,
                                    group=cpu_location_group,
                                )
                                memory_free_writer = otf2.writer.EventWriter(
                                    archive, memory_free_location
                                )
                                memory_free_events = []
                                for start, end, name in data:
                                    region = archive.definitions.region(
                                        name=name,
                                        region_role=RegionRole.DEALLOCATE,
                                        paradigm=Paradigm.HIP,
                                    )
                                    memory_free_events.append((start, "enter", region))
                                    memory_free_events.append((end, "leave", region))
                                memory_free_events.sort(key=lambda x: x[0])
                                for timestamp, event_type, region in memory_free_events:
                                    if event_type == "enter":
                                        memory_free_writer.enter(
                                            timestamp,
                                            region,
                                            attributes=memory_allocation_attributes,
                                        )
                                    else:  # if event_type == "leave":
                                        memory_free_writer.leave(timestamp, region)

                            # Write Unknown Memory Events
                            for tid, data in memory_unknown.items():
                                memory_unknown_location = archive.definitions.location(
                                    name=f"Thread {tid}, Memory Operation UNK",
                                    type=LocationType.ACCELERATOR_STREAM,
                                    group=cpu_location_group,
                                )
                                memory_unknown_writer = otf2.writer.EventWriter(
                                    archive, memory_unknown_location
                                )
                                memory_unknown_events = []
                                for start, end, name in data:
                                    region = archive.definitions.region(
                                        name=name,
                                    )
                                    memory_unknown_events.append((start, "enter", region))
                                    memory_unknown_events.append((end, "leave", region))
                                memory_unknown_events.sort(key=lambda x: x[0])
                                for (
                                    timestamp,
                                    event_type,
                                    region,
                                ) in memory_unknown_events:
                                    if event_type == "enter":
                                        memory_unknown_writer.enter(
                                            timestamp,
                                            region,
                                            attributes=memory_allocation_attributes,
                                        )
                                    else:  # if event_type == "leave":
                                        memory_unknown_writer.leave(timestamp, region)

                            # Write Kernel Dispatch Events
                            for (tid, agent_id, queue), data in kernel_dispatches.items():
                                agent_name, agent_location_group = agents.get(
                                    agent_id, (f"Unknown Agent {agent_id}", None)
                                )
                                kernel_location = archive.definitions.location(
                                    name=f"Thread {tid}, Compute on {agent_name}, Queue {queue}",
                                    type=LocationType.ACCELERATOR_STREAM,
                                    group=agent_location_group,
                                )
                                kernel_writer = otf2.writer.EventWriter(
                                    archive, kernel_location
                                )
                                kernel_events = []
                                for start, end, name in data:
                                    region = archive.definitions.region(
                                        name=name,
                                        region_role=RegionRole.FUNCTION,
                                        paradigm=Paradigm.HIP,
                                    )
                                    kernel_events.append((start, "enter", region))
                                    kernel_events.append((end, "leave", region))
                                kernel_events.sort(key=lambda x: x[0])
                                for timestamp, event_type, region in kernel_events:
                                    if event_type == "enter":
                                        kernel_writer.enter(
                                            timestamp,
                                            region,
                                            attributes=kernel_attributes,
                                        )
                                    else:  # if event_type == "leave":
                                        kernel_writer.leave(timestamp, region)
            except Exception as e:
                print("Could not query sqlite database:", e)
        else:
            print("No sqlite connection found in importData.")


def execute(input, config=None, **kwargs):

    config = (
        output_config.output_config(**kwargs)
        if config is None
        else config.update(**kwargs)
    )

    write_otf2(input, config)


def add_args(parser):
    """Add otf2 arguments."""

    # Currently, no otf2 specific args

    # otf2_options = parser.add_argument_group("OTF2 options")

    # otf2_options.add_argument(
    #     "--kernel-rename",
    #     help="Use kernel names from debugging symbols if available",
    #     action="store_true",
    #     default=False,
    # )

    def process_args(input, args):
        valid_args = []
        ret = {}
        for itr in valid_args:
            if hasattr(args, itr):
                val = getattr(args, itr)
                if val is not None:
                    ret[itr] = val
        return ret

    return process_args


def main(argv=None):
    import argparse
    from . import time_window
    from . import output_config

    parser = argparse.ArgumentParser(
        description="Convert rocPD to OTF2 format", allow_abbrev=False
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

    process_out_config_args = output_config.add_args(parser)
    process_otf2_args = add_args(parser)
    process_generic_args = output_config.add_generic_args(parser)
    process_time_window_args = time_window.add_args(parser)

    args = parser.parse_args(argv)

    input = RocpdImportData(
        args.input, automerge_limit=getattr(args, "automerge_limit", None)
    )

    out_cfg_args = process_out_config_args(input, args)
    generic_out_cfg_args = process_generic_args(input, args)
    otf2_args = process_otf2_args(input, args)
    process_time_window_args(input, args)

    all_args = {**out_cfg_args, **otf2_args, **generic_out_cfg_args}

    execute(input, **all_args)


if __name__ == "__main__":
    main()
