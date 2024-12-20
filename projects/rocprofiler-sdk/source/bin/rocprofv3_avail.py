#!/usr/bin/env python3

import ctypes
import pathlib
import os
import io
import csv
import socket
import sys


def fatal_error(msg, exit_code=1):
    sys.stderr.write(f"Fatal error: {msg}\n")
    sys.stderr.flush()
    sys.exit(exit_code)


class derived_counter:

    def __init__(
        self, counter_name, counter_description, counter_expression, counter_dimensions
    ):

        self.name = counter_name
        self.description = counter_description
        self.expression = counter_expression
        self.dimensions = counter_dimensions


class basic_counter:

    def __init__(
        self, counter_name, counter_description, counter_block, counter_dimensions
    ):

        self.name = counter_name
        self.description = counter_description
        self.block = counter_block
        self.dimensions = counter_dimensions


class pc_config:

    def __init__(self, config_method, config_unit, min_interval, max_interval):
        self.method = config_method
        self.unit = config_unit
        self.min_interval = min_interval
        self.max_interval = max_interval


MAX_STR = 256
libname = os.environ.get("ROCPROF_LIST_AVAIL_TOOL_LIBRARY")
c_lib = ctypes.CDLL(libname)

c_lib.get_number_of_counters.restype = ctypes.c_ulong
c_lib.get_number_of_pc_sample_configs.restype = ctypes.c_ulong
c_lib.get_number_of_dimensions.restype = ctypes.c_ulong

c_lib.get_number_of_counters.argtypes = [ctypes.c_int]
c_lib.get_number_of_pc_sample_configs.argtypes = [ctypes.c_int]
c_lib.get_number_of_dimensions.argtypes = [ctypes.c_int]

c_lib.get_pc_sample_config.argtypes = [
    ctypes.c_ulong,
    ctypes.c_ulong,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_char * MAX_STR)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_char * MAX_STR)),
    ctypes.POINTER(ctypes.c_ulong),
    ctypes.POINTER(ctypes.c_ulong),
]

c_lib.get_counters_info.argtypes = [
    ctypes.c_ulong,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_ulong),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_char * MAX_STR)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_char * MAX_STR)),
    ctypes.POINTER(ctypes.c_int),
]

c_lib.get_counter_expression.argtypes = [
    ctypes.c_ulong,
    ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_char * MAX_STR)),
]

c_lib.get_counter_dimension.argtypes = [
    ctypes.c_ulong,
    ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_ulong),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_char * MAX_STR)),
    ctypes.POINTER(ctypes.c_ulong),
]

c_lib.get_counter_block.argtypes = [
    ctypes.c_ulong,
    ctypes.c_ulong,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_char * MAX_STR)),
]

c_lib.get_number_of_agents.restype = ctypes.c_size_t

c_lib.get_agent_node_id.restype = ctypes.c_ulong
c_lib.get_agent_node_id.argtypes = [ctypes.c_int]


agent_derived_counter_map = dict()
agent_basic_counter_map = dict()
agent_pc_sample_config_map = dict()


def get_counters(node_id):

    no_of_counters = c_lib.get_number_of_counters(node_id)

    basic_counters = []
    derived_counters = []
    for counter_idx in range(0, no_of_counters):

        name_args = ctypes.POINTER(ctypes.c_char * MAX_STR)()
        description_args = ctypes.POINTER(ctypes.c_char * MAX_STR)()
        block_args = ctypes.POINTER(ctypes.c_char * MAX_STR)()
        is_derived_args = ctypes.c_int()
        counter_id_args = ctypes.c_ulong()

        c_lib.get_counters_info(
            node_id,
            counter_idx,
            ctypes.byref(counter_id_args),
            name_args,
            description_args,
            ctypes.byref(is_derived_args),
        )

        is_derived = is_derived_args.value
        counter_id = counter_id_args.value
        no_of_dimensions = c_lib.get_number_of_dimensions(counter_id)

        name = ctypes.cast(name_args, ctypes.c_char_p).value.decode("utf-8")
        description = ctypes.cast(description_args, ctypes.c_char_p).value.decode("utf-8")
        dimensions_stream = io.StringIO()

        for dim in range(0, no_of_dimensions):

            dim_name_args = ctypes.POINTER(ctypes.c_char * MAX_STR)()
            dim_instance_args = ctypes.c_ulong()
            dimension_id_args = ctypes.c_ulong()

            c_lib.get_counter_dimension(
                counter_id,
                dim,
                ctypes.byref(dimension_id_args),
                dim_name_args,
                ctypes.byref(dim_instance_args),
            )

            dim_name = ctypes.cast(dim_name_args, ctypes.c_char_p).value.decode("utf-8")
            dim_instance = dim_instance_args.value

            dimensions_stream.write(dim_name)
            dimensions_stream.write("[0:")
            dimensions_stream.write(str(dim_instance))
            dimensions_stream.write("]")
            if dim != no_of_dimensions - 1:
                dimensions_stream.write("\t")

        if is_derived:

            expression_args = ctypes.POINTER(ctypes.c_char * MAX_STR)()
            c_lib.get_counter_expression(node_id, counter_idx, expression_args)
            counter_expression = ctypes.cast(
                expression_args, ctypes.c_char_p
            ).value.decode("utf-8")
            derived_counters.append(
                derived_counter(
                    name, description, counter_expression, dimensions_stream.getvalue()
                )
            )

        else:

            block_args = ctypes.POINTER(ctypes.c_char * MAX_STR)()
            c_lib.get_counter_block(node_id, counter_idx, block_args)
            block = ctypes.cast(block_args, ctypes.c_char_p).value.decode("utf-8")
            basic_counters.append(
                basic_counter(name, description, block, dimensions_stream.getvalue())
            )
        dimensions_stream.close()

    agent_derived_counter_map[node_id] = derived_counters
    agent_basic_counter_map[node_id] = basic_counters


def get_pc_sample_configs(node_id):

    no_of_pc_sample_configs = c_lib.get_number_of_pc_sample_configs(node_id)
    pc_sample_configs = []
    if no_of_pc_sample_configs:
        for config_idx in range(0, no_of_pc_sample_configs):

            method_args = ctypes.POINTER(ctypes.c_char * MAX_STR)()
            unit_args = ctypes.POINTER(ctypes.c_char * MAX_STR)()
            min_interval = ctypes.c_ulong()
            max_interval = ctypes.c_ulong()

            c_lib.get_pc_sample_config(
                node_id,
                config_idx,
                method_args,
                unit_args,
                ctypes.byref(min_interval),
                ctypes.byref(max_interval),
            )

            method = ctypes.cast(method_args, ctypes.c_char_p).value.decode("utf-8")
            unit = ctypes.cast(unit_args, ctypes.c_char_p).value.decode("utf-8")
            pc_sample_configs.append(
                pc_config(method, unit, min_interval.value, max_interval.value)
            )

        agent_pc_sample_config_map[node_id] = pc_sample_configs


def process_filename(file_path, file_type):

    filename = os.environ.get(
        "ROCPROF_OUTPUT_FILE_NAME", socket.gethostname() + "/" + str(os.getpid())
    )

    if os.path.exists(file_path) and os.path.isfile(file_path):
        fatal_error("ROCPROFILER_OUTPUT_PATH already exists and is not a directory")

    elif not os.path.exists(file_path):
        os.makedirs(file_path)

    output_filename = ""
    if file_type == "derived":
        output_filename = filename + "_" + "derived_metrics" + ".csv"
    elif file_type == "basic":
        output_filename = filename + "_" + "basic_metrics" + ".csv"
    elif file_type == "pc_sample_config":
        output_filename = filename + "_" + "pc_sample_config" + ".csv"
    output_path = os.path.join(file_path, output_filename)
    output_path_parent = os.path.dirname(output_path)

    if not os.path.exists(output_path_parent):
        os.makedirs(output_path_parent)

    elif os.path.exists(output_path_parent) and os.path.isfile(output_path_parent):
        fatal_error("ROCPROFILER_OUTPUT_PATH already exists and is not a directory")

    return output_path


def generate_output(agent_ids):

    list_avail_file = os.environ.get("ROCPROF_OUTPUT_LIST_AVAIL_FILE")

    if list_avail_file:

        file_path = os.environ.get("ROCPROF_OUTPUT_PATH")
        derived_output_file = process_filename(file_path, "derived")
        basic_output_file = process_filename(file_path, "basic")
        pc_sample_config_file = process_filename(file_path, "pc_sample_config")

        with open(derived_output_file, "w") as csvfile:
            print(f"Opened result file: {derived_output_file}")
            fieldnames = ["Agent_Id", "Name", "Description", "Expression", "Dimensions"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for node_id, counters in agent_derived_counter_map.items():
                for counter in counters:
                    writer.writerow(
                        {
                            "Agent_Id": node_id,
                            "Name": counter.name,
                            "Description": counter.description,
                            "Expression": counter.expression,
                            "Dimensions": counter.dimensions,
                        }
                    )

        with open(basic_output_file, "w") as csvfile:
            print(f"Opened result file: {basic_output_file}")
            fieldnames = ["Agent_Id", "Name", "Description", "Block", "Dimensions"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for node_id, counters in agent_basic_counter_map.items():
                for counter in counters:
                    if counter.block:
                        writer.writerow(
                            {
                                "Agent_Id": node_id,
                                "Name": counter.name,
                                "Description": counter.description,
                                "Block": counter.block,
                                "Dimensions": counter.dimensions,
                            }
                        )

        with open(pc_sample_config_file, "w") as csvfile:
            print(f"Opened result file: {pc_sample_config_file}")
            fieldnames = [
                "Agent_Id",
                "Method",
                "Unit",
                "Minimum_Interval",
                "Maximum_Interval",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for node_id, configs in agent_pc_sample_config_map.items():
                for config in configs:
                    writer.writerow(
                        {
                            "Agent_Id": node_id,
                            "Method": config.method,
                            "Unit": config.unit,
                            "Minimum_Interval": config.min_interval,
                            "Maximum_Interval": config.max_interval,
                        }
                    )
    else:

        for node_id in agent_ids:
            if node_id in agent_basic_counter_map.keys():
                basic_counters_stream = io.StringIO()
                counters = agent_basic_counter_map[node_id]
                for counter in counters:
                    if counter.block:
                        basic_counters_stream.write(f"gpu-agent:{node_id}\n")
                        basic_counters_stream.write("Name:")
                        basic_counters_stream.write("\t")
                        basic_counters_stream.write(str(counter.name))
                        basic_counters_stream.write("\n")
                        basic_counters_stream.write("Description:")
                        basic_counters_stream.write("\t")
                        basic_counters_stream.write(str(counter.description))
                        basic_counters_stream.write("\n")
                        basic_counters_stream.write("Block:")
                        basic_counters_stream.write("\t")
                        basic_counters_stream.write(str(counter.block))
                        basic_counters_stream.write("\n")
                        basic_counters_stream.write("Dimensions:")
                        basic_counters_stream.write("\t")
                        basic_counters_stream.write(str(counter.dimensions))
                        basic_counters_stream.write("\n\n")
                basic_counters = basic_counters_stream.getvalue()
                print(basic_counters)
                basic_counters_stream.close()

            if node_id in agent_derived_counter_map.keys():
                derived_counters_stream = io.StringIO()
                counters = agent_derived_counter_map[node_id]
                for counter in counters:
                    derived_counters_stream.write(f"gpu-agent:{node_id}\n")
                    derived_counters_stream.write("Name:")
                    derived_counters_stream.write("\t")
                    derived_counters_stream.write(str(counter.name))
                    derived_counters_stream.write("\n")
                    derived_counters_stream.write("Description:")
                    derived_counters_stream.write("\t")
                    derived_counters_stream.write(str(counter.description))
                    derived_counters_stream.write("\n")
                    derived_counters_stream.write("Expression:")
                    derived_counters_stream.write("\t")
                    derived_counters_stream.write(str(counter.expression))
                    derived_counters_stream.write("\n")
                    derived_counters_stream.write("Dimensions:")
                    derived_counters_stream.write("\t")
                    derived_counters_stream.write(str(counter.dimensions))
                    derived_counters_stream.write("\n\n")
                derived_counters = derived_counters_stream.getvalue()
                print(derived_counters)
                derived_counters_stream.close()

            if node_id in agent_pc_sample_config_map.keys():
                pc_sample_config_stream = io.StringIO()
                configs = agent_pc_sample_config_map[node_id]
                for config in configs:
                    pc_sample_config_stream.write("Method:")
                    pc_sample_config_stream.write("\t")
                    pc_sample_config_stream.write(str(config.method))
                    pc_sample_config_stream.write("\n")
                    pc_sample_config_stream.write("Unit:")
                    pc_sample_config_stream.write("\t")
                    pc_sample_config_stream.write(str(config.unit))
                    pc_sample_config_stream.write("\n")
                    pc_sample_config_stream.write("Minimum_Interval:")
                    pc_sample_config_stream.write("\t")
                    pc_sample_config_stream.write(str(config.min_interval))
                    pc_sample_config_stream.write("\n")
                    pc_sample_config_stream.write("Maximum_Interval:")
                    pc_sample_config_stream.write("\t")
                    pc_sample_config_stream.write(str(config.max_interval))
                    pc_sample_config_stream.write("\n")
                pc_sample = pc_sample_config_stream.getvalue()
                print(
                    "List available PC Sample Configurations for node_id\t"
                    + str(node_id)
                    + "\n"
                )
                print(pc_sample)
                print("\n")
                pc_sample_config_stream.close()
            else:
                print("PC Sampling not supported on node_id\t" + str(node_id) + "\n")


if __name__ == "__main__":
    # Load the shared library into ctypes

    c_lib.avail_tool_init()
    no_of_agents = c_lib.get_number_of_agents()
    agent_ids = []
    for idx in range(0, no_of_agents):

        node_id = c_lib.get_agent_node_id(idx)
        agent_ids.append(node_id)
        get_counters(node_id)
        get_pc_sample_configs(node_id)

    generate_output(agent_ids)
