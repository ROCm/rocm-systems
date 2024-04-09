#!/usr/bin/env python3

import sys
import pytest


# helper function
def node_exists(name, data, min_len=1):
    assert name in data
    assert data[name] is not None
    assert len(data[name]) >= min_len


def test_data_structure(input_data):
    """verify minimum amount of expected data is present"""
    node_exists("rocprofiler-sdk-json-tool", input_data)
    rocp_data = input_data
    node_exists("names", rocp_data["rocprofiler-sdk-json-tool"]["buffer_records"])
    node_exists(
        "counter_collection", rocp_data["rocprofiler-sdk-json-tool"]["buffer_records"]
    )


def test_counter_values(input_data):
    data = input_data
    agent_data = data["rocprofiler-sdk-json-tool"]["agents"]
    counter_data = data["rocprofiler-sdk-json-tool"]["buffer_records"][
        "counter_collection"
    ]

    scaling_factor = 1
    for itr in agent_data:
        if itr["type"] == 2 and itr["wave_front_size"] > 0:
            scaling_factor = 64 / itr["wave_front_size"]
            break

    for itr in counter_data:
        value = itr["counter_value"]
        if int(round(value, 0)) > 0:
            assert int(round(value, 0)) == int(
                round(1 * scaling_factor, 0)
            ), f"agent_data:\n{agent_data}\n\ncounter_data:\n{counter_data}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
