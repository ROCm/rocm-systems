#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
import argparse
import sys
from rocprofv3_avail_module import avail


def format_help(formatter, w=120, h=40):
    """Return a wider HelpFormatter, if possible."""
    try:
        kwargs = {"width": w, "max_help_position": h}
        formatter(None, **kwargs)
        return lambda prog: formatter(prog, **kwargs)
    except TypeError:
        return formatter


def strtobool(val):
    """Convert a string representation of truth to true or false.
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    if isinstance(val, (list, tuple)):
        if len(val) > 1:
            val_type = type(val).__name__
            raise ValueError(f"invalid truth value {val} (type={val_type})")
        else:
            val = val[0]

    if isinstance(val, bool):
        return val
    elif isinstance(val, str) and val.lower() in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif isinstance(val, str) and val.lower() in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        val_type = type(val).__name__
        raise ValueError(f"invalid truth value {val} (type={val_type})")


class booleanArgAction(argparse.Action):
    def __call__(self, parser, args, value, option_string=None):
        setattr(args, self.dest, strtobool(value))


def parse_arguments(args=None):

    usage_examples = """

%(prog)s, e.g.

    $ rocprofv3-avail [<rocprofv3-avail-option> ...]
    $ rocprofv3-avail -- avail-hw-counters

"""

    def add_list_options(subparsers):
        list_command = subparsers.add_parser(
            "list", help="List options for hw counters, agents and pc-sampling support"
        )
        add_parser_bool_argument(list_command, "--pmc", help="List counters")
        add_parser_bool_argument(
            list_command, "--agent", help="List basic info of agents"
        )
        add_parser_bool_argument(
            list_command, "--pc-sampling", help="List agents supporting pc-sampling"
        )
        list_command.set_defaults(func=process_list)

    def add_info_options(subparsers):
        info_command = subparsers.add_parser(
            "info",
            help="Info options for detailed information of counters, agents, and pc-sampling configurations",
        )

        info_command.add_argument("--pmc", nargs="*", help="PMC info")
        info_command.add_argument(
            "--pc-sampling", nargs="*", help="Detailed PC Sampling info"
        )
        info_command.set_defaults(func=process_info)

    def add_pmc_check_options(subparsers):
        pmc_check_command = subparsers.add_parser(
            "pmc-check", help="Checking counters collection support on agents"
        )
        pmc_check_command.add_argument(
            "pmc", nargs="*", default=None, help="List of PMC names"
        )
        pmc_check_command.set_defaults(func=process_pmc_check)

    def add_parser_bool_argument(gparser, *args, **kwargs):
        gparser.add_argument(
            *args,
            **kwargs,
            action=booleanArgAction,
            nargs="?",
            const=True,
            type=str,
            required=False,
            metavar="BOOL",
        )

    # Create the parser
    parser = argparse.ArgumentParser(
        description="ROCProfilerV3-avail Run Script",
        usage="%(prog)s [options] ",
        epilog=usage_examples,
        formatter_class=format_help(argparse.RawTextHelpFormatter),
    )

    parser.add_argument(
        "-d",
        "--device",
        help="device index, device on a node to apply the sub-commands on",
        type=int,
        default=None,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="rocprofv3-avail sub commands"
    )

    add_list_options(subparsers)
    add_info_options(subparsers)
    add_pmc_check_options(subparsers)

    return parser.parse_args(args[:])


def get_basic_agent_info(info):
    basic_info = {}
    req_keys = [
        "cpu_cores_count",
        "simd_count",
        "max_waves_per_simd",
        "runtime_visibility",
        "wave_front_size",
        "num_xcc",
        "cu_count",
        "array_count",
        "num_shader_banks",
        "simd_arrays_per_engine",
        "cu_per_simd_array",
        "simd_per_cu",
        "gfx_target_version",
        "max_waves_per_cu",
        "gpu_id",
        "workgroup_max_dim",
        "grid_max_dim",
        "name",
        "vendor_name",
        "product_name",
        "model_name",
        "runtime_visibility",
        "node_id",
        "logical_node_id",
        "logical_node_type_id",
    ]
    for key in req_keys:
        basic_info.update({key: info[key]})
    return basic_info


def list_basic_agent(args, list_counters):
    def print_agent_counter(counters, info):
        names = ["{:20}".format(counter.name) for counter in counters]
        print("   PMC:\n")
        for idx in range(0, len(names), int(len(counters) / 20)):
            print("              {}".format(" ".join(names[idx : (idx + 5)])))

    def print_basic_info(info):
        print("GPU:{}\n".format(info["logical_node_type_id"]))
        print("\n".join(["   {:20}: {}".format(key, itr) for key, itr in info.items()]))
        if not list_counters:
            print("\n")

    agent_info_map = avail.get_agent_info_map()
    agent_counters = avail.get_counters()

    for agent, info in agent_info_map.items():
        if (
            info["type"] == 2
            and args.device is not None
            and info["logical_node_type_id"] == args.device
        ):
            print_basic_info(get_basic_agent_info(info))
            if list_counters:
                print_agent_counter(agent_counters[agent], info)
            break

        elif info["type"] == 2 and args.device is None:
            print_basic_info(get_basic_agent_info(info))
            if list_counters:
                print_agent_counter(agent_counters[agent], info)


def list_pc_sampling(args):
    sampling_agents = avail.get_pc_sample_configs()
    agent_info_map = avail.get_agent_info_map()
    print("Agents supporting PC Sampling\n")
    for agent in sampling_agents.keys():
        info = agent_info_map[agent]
        print("GPU:{}\nNAME:{}\n".format(info["logical_node_type_id"], info["name"]))


def info_pc_sampling(args):
    sampling_agents = avail.get_pc_sample_configs()
    agent_info_map = avail.get_agent_info_map()
    for agent, configs in sampling_agents.items():
        info = agent_info_map[agent]
        print("GPU:{}\nNAME:{}".format(info["logical_node_type_id"], info["name"]))
        print("configs:")
        for config in configs:
            print(config)
            print("\n")
        print("\n")


def listing(args):
    def print_agent_counter(counters, info):
        names = ["{:20}".format(counter.name) for counter in counters]
        print("PMC:\n")
        for idx in range(0, len(names), int(len(counters) / 20)):
            print("              {}".format(" ".join(names[idx : (idx + 5)])))

    agent_counters = avail.get_counters()
    agent_info_map = avail.get_agent_info_map()

    for agent, info in agent_info_map.items():
        if (
            info["type"] == 2
            and args.device is not None
            and info["logical_node_type_id"] == args.device
        ):
            print("GPU:{}\nNAME:{}".format(info["logical_node_type_id"], info["name"]))
            print_agent_counter(agent_counters[agent], info)
            break
        elif info["type"] == 2 and args.device is None:
            print("GPU:{}\nNAME:{}".format(info["logical_node_type_id"], info["name"]))
            print_agent_counter(agent_counters[agent], info)


def info_pmc(args):
    agent_counters = avail.get_counters()
    agent_info_map = avail.get_agent_info_map()

    def print_pmc_info(args, pmc_counters):

        if not args.pmc:
            for counter in pmc_counters:
                print(counter)
                print("\n")
        else:
            for pmc in args.pmc:
                for counter in pmc_counters:
                    if pmc == counter.get_as_dict()["counter_name"]:
                        print(counter)
                        print("\n")

    for agent, info in agent_info_map.items():
        if (
            info["type"] == 2
            and args.device is not None
            and info["logical_node_type_id"] == args.device
        ):
            print("GPU:{}\nNAME:{}".format(info["logical_node_type_id"], info["name"]))
            print_pmc_info(args, agent_counters[agent])
            break
        elif info["type"] == 2 and args.device is None:
            print("GPU:{}\nNAME:{}".format(info["logical_node_type_id"], info["name"]))
            print_pmc_info(args, agent_counters[agent])


def process_info(args):
    if args.pmc is None and args.pc_sampling is None:
        list_basic_agent(args, True)
    if args.pmc is not None:
        info_pmc(args)
    if args.pc_sampling is not None:
        os.environ["ROCPROFILER_PC_SAMPLING_BETA_ENABLED"] = "on"
        info_pc_sampling(args)


def process_list(args):
    if not args.agent and args.pc_sampling is None:
        listing(args)
    if args.agent:
        list_basic_agent(args, False)
    if args.pc_sampling:
        os.environ["ROCPROFILER_PC_SAMPLING_BETA_ENABLED"] = "on"
        list_pc_sampling(args)


def process_pmc_check(args):
    def get_device_agent(device_id):
        for agent, info in agent_info_map.items():
            if info["type"] == 2 and info["logical_node_type_id"] == device_id:
                return agent
        avail.fatal_error("Invalid device id : {}".format(device_id))

    def get_gpu_agents():
        agent_ids = []
        for agent, info in agent_info_map.items():
            if info["type"] == 2:
                agent_ids.append(agent)
        return agent_ids

    def get_counter_handle(counter_name):
        agent_counters = avail.get_counters()
        for agent, counters in agent_counters.items():
            for counter in counters:
                if counter.get_as_dict()["counter_name"] == counter_name:
                    return counter.counter_handle
        avail.fatal_error("Invalid counter name")

    def get_counter_names(pmc_list, agent):
        agent_counters = avail.get_counters()
        counter_names = []
        for pmc in pmc_list:
            for counter in agent_counters[agent]:
                if counter.counter_handle == pmc:
                    counter_names.append(counter.name)
        return counter_names

    def get_logical_node_type_id(agent_id):
        for agent, info in agent_info_map.items():
            if info["type"] == 2 and info["id"]["handle"] == agent_id:
                return info["logical_node_type_id"]

    def process_qualifiers(pmc):
        res = pmc.split(":")
        if len(res) > 2:
            avail.fatal_error("Invalid format for pmc-check")

        if len(res) == 2:
            qualifiers_list = []
            qualifiers = res[1]
            if "device" not in qualifiers:
                avail.fatal_error("Incorrect input format for device index")
            qualifiers = qualifiers.split(",")
            for qualifier in qualifiers:
                qualifiers_list.append(dict([qualifier.split("=")]))
            return (res[0], qualifiers_list)

        return (res[0], None)

    if not args.pmc:
        avail.fatal_error("Provide counter to check")

    device_pmc = {}
    agent_info_map = avail.get_agent_info_map()

    for pmc in args.pmc:
        counter, qualifiers = process_qualifiers(pmc)
        if qualifiers is None:
            if args.device is not None:
                agent_handle = get_device_agent(args.device)
                if agent_handle not in device_pmc.keys():
                    device_pmc.setdefault(agent_handle, [])
                device_pmc[agent_handle].append(get_counter_handle(counter))
            else:
                agent_ids = get_gpu_agents()
                for agent_handle in agent_ids:
                    if agent_handle not in device_pmc.keys():
                        device_pmc.setdefault(agent_handle, [])
                    device_pmc[agent_handle].append(get_counter_handle(counter))
        else:
            for itr in qualifiers:
                agent_handle = get_device_agent(int(itr["device"]))
                if agent_handle not in device_pmc.keys():
                    device_pmc.setdefault(agent_handle, [])
                device_pmc[agent_handle].append(get_counter_handle(counter))

    if avail.check_pmc(device_pmc) is True:
        for agent, pmc in device_pmc.items():
            device_id = get_logical_node_type_id(agent)
            pmc_names = get_counter_names(pmc, agent)
            print(
                "Following input counters can be collected together on GPU:{}\t{}".format(
                    device_id, "\t".join(pmc_names)
                )
            )


def main(argv=None):
    ROCPROFV3_AVAIL_DIR = os.path.dirname(os.path.realpath(__file__))
    ROCM_DIR = os.path.dirname(ROCPROFV3_AVAIL_DIR)
    ROCPROF_LIST_AVAIL_TOOL_LIBRARY = (
        f"{ROCM_DIR}/libexec/rocprofiler-sdk/librocprofv3-list-avail.so"
    )
    avail.loadLibrary.libname = os.environ.get(
        "ROCPROF_LIST_AVAIL_TOOL_LIBRARY", ROCPROF_LIST_AVAIL_TOOL_LIBRARY
    )
    args = parse_arguments(argv)
    if args.command:
        args.func(args)


if __name__ == "__main__":
    ec = main(sys.argv[1:])
    sys.exit(ec)
