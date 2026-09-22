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

import sys
import os
import pytest
import ctypes


def test_validate_metrics(rocm_path):
    set_library(rocm_path)

    from rocprofv3 import avail

    lib = avail.get_library()

    agent_counters_dict = avail.get_counters()
    for agent, counters in agent_counters_dict.items():
        for counter in counters:
            assert counter.name != ""
            assert counter.description != ""
            if isinstance(counter, avail.derived_counter):
                assert counter.expression != ""
            elif isinstance(counter, avail.basic_counter):
                assert counter.block != ""
            assert counter.dimensions != ""
            if counter.name == "TA_BUSY_min":
                counter.description == "TA block is busy. Min over TA instances."
                counter.expression == "reduce(TA_TA_BUSY,min)"
            if counter.name == "TA_BUSY_min":
                counter.description == "TA block is busy. Min over TA instances."
                counter.expression == "reduce(TA_TA_BUSY,min)"


def test_validate_list_pc_sample_config(rocm_path):
    set_library(rocm_path)

    from rocprofv3 import avail

    lib = avail.get_library()

    pc_sample_configs_dict = avail.get_pc_sample_configs()
    for agent, configs in pc_sample_configs_dict.items():
        for config in configs:
            assert config.method != ""
            assert config.unit != ""
            assert isinstance(config.max_interval, ctypes.c_ulong) == True
            assert isinstance(config.min_interval, ctypes.c_ulong) == True
            if config.method == "stochastic":
                assert config.flags.value == 1
            elif config.method == "host_trap":
                assert config.flags.value == 0
            assert config.max_interval.value >= config.min_interval.value


def test_counter_set(capsys, rocm_path):
    set_library(rocm_path)

    from rocprofv3 import avail

    def get_counter_names(counter_ids):

        counter_names = []
        for counter_id in counter_ids:
            counter = get_counter_info(counter_id)
            if counter.counter_handle == counter_id:
                counter_names.append(counter.name)
        return counter_names

    def get_agent_name(agent_id):
        agent_info_map = get_agent_info_map()
        for agent, info in agent_info_map.items():
            if agent == agent_id:
                return info["name"]

    lib = avail.get_library()
    agent_counters = avail.get_counters()
    pmc_input_1 = []

    for agent, counters in agent_counters.items():
        counter_ids = []
        for counter in counters:
            counter_ids.append(counter.counter_handle)
        input = {agent: [counter_ids[0]]}
        assert avail.check_pmc(input) == True
        with pytest.raises(SystemExit) as excinfo:
            input = {agent: counter_ids}
            avail.check_pmc(input)
            output = capsys.readouterr()
            test = "{} not collected on agent {}".format(
                " ".join(get_counter_names(counter_ids)), get_agent_name(agent)
            )
            assert test == output


def set_library(rocm_path):
    ROCPROFV3_AVAIL_PACKAGE = f"{rocm_path}/lib/python{sys.version_info[0]}/site-packages"
    sys.path.append(ROCPROFV3_AVAIL_PACKAGE)

    from rocprofv3 import avail

    ROCPROF_LIST_AVAIL_TOOL_LIBRARY = (
        f"{rocm_path}/lib/rocprofiler-sdk/librocprofv3-list-avail.so"
    )
    os.environ["ROCPROFILER_METRICS_PATH"] = f"{rocm_path}/share/rocprofiler-sdk"
    avail.loadLibrary.libname = os.environ.get(
        "ROCPROF_LIST_AVAIL_TOOL_LIBRARY", ROCPROF_LIST_AVAIL_TOOL_LIBRARY
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
