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
import pytest
from collections import defaultdict


# helper function
def node_exists(name, data, min_len=1):
    assert name in data
    assert data[name] is not None
    assert len(data[name]) >= min_len


def test_data_structure(input_data):
    """verify minimum amount of expected data is present"""
    node_exists("rocprofiler-sdk-json-tool", input_data)
    rocp_data = input_data
    node_exists("names", rocp_data["rocprofiler-sdk-json-tool"]["callback_records"])
    node_exists("spm_records", rocp_data["rocprofiler-sdk-json-tool"]["callback_records"])


def test_spm_counter_values(input_data):
    data = input_data["rocprofiler-sdk-json-tool"]
    agent_data = data["agents"]
    counter_info = data["counter_info"]
    counter_data = data["callback_records"]["spm_records"]
    agent_counter_map = defaultdict(list)

    def get_counter_value(counters, name):
        for itr in counters:
            if itr["name"] == name:
                return itr["value"]

    def get_name(counter_id):
        for itr in counter_info:
            if itr["id"]["handle"] == counter_id:
                return itr["name"]

    for record in counter_data:
        if record["agent_id"]["handle"] in agent_counter_map.keys():
            found = 0
            for i in range(0, len(agent_counter_map[record["agent_id"]["handle"]])):
                if agent_counter_map[record["agent_id"]["handle"]][i]["name"] == get_name(
                    record["counter_id"]["handle"]
                ):
                    agent_counter_map[record["agent_id"]["handle"]][i]["value"] += record[
                        "value"
                    ]
                    found = 1
            if not found:
                agent_counter_map[record["agent_id"]["handle"]].append(
                    {
                        "name": get_name(record["counter_id"]["handle"]),
                        "value": record["value"],
                    }
                )
        else:
            agent_counter_map[record["agent_id"]["handle"]].append(
                {
                    "name": get_name(record["counter_id"]["handle"]),
                    "value": record["value"],
                }
            )

    for agent, counters in agent_counter_map.items():

        assert 1.0 * get_counter_value(counters, "SQ_INSTS_SALU") > get_counter_value(
            counters, "SQ_WAVES"
        )
        assert 1.0 * get_counter_value(counters, "SQ_INSTS_VALU") > get_counter_value(
            counters, "SQ_WAVES"
        )
        assert 1.0 * get_counter_value(counters, "TA_TA_BUSY") > get_counter_value(
            counters, "TA_TOTAL_WAVEFRONTS"
        )
        assert 1.0 * get_counter_value(counters, "TCP_TOTAL_WRITE") > get_counter_value(
            counters, "SQ_WAVES"
        )
        assert (
            100
            * get_counter_value(counters, "SQC_ICACHE_MISSES")
            / get_counter_value(counters, "SQC_ICACHE_REQ")
        ) < 100
        assert (
            100
            * get_counter_value(counters, "SQC_ICACHE_HITS")
            / get_counter_value(counters, "SQC_ICACHE_REQ")
        ) < 100
        assert (
            100
            * get_counter_value(counters, "SPI_CSN_WINDOW_VALID")
            / get_counter_value(counters, "SQ_CYCLES")
        )
        assert (
            100
            * get_counter_value(counters, "SPI_CSN_BUSY")
            / get_counter_value(counters, "SQ_CYCLES")
        ) < 100
        assert (
            100
            * get_counter_value(counters, "TCC_HIT")
            / get_counter_value(counters, "TCC_REQ")
        ) < 100
        assert (
            100
            * get_counter_value(counters, "TCC_MISS")
            / get_counter_value(counters, "TCC_REQ")
        ) < 100


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
