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
    counter_info = {}
    for itr in data["rocprofiler-sdk-json-tool"]["buffer_records"]["counter_collection"]:
        value = itr["counter_value"]
        assert value == 1


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
