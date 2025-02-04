#!/usr/bin/env python3

import sys
import pytest
import re
import os


def test_json_data(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    strings = data["strings"]
    assert "att_filenames" in strings.keys()
    att_files = data["strings"]["att_filenames"]
    assert len(att_files) > 0


def test_code_object_memory(code_object_file_path, json_data, output_path):

    data = json_data["rocprofiler-sdk-tool"]
    tool_memory_load = data["strings"]["code_object_snapshot_filenames"]
    gfx_pattern = "gfx[a-z0-9]+"
    match = re.search(gfx_pattern, tool_memory_load[0])
    assert match != None
    gpu_name = match.group(0)
    tool_memory_load_1 = open(os.path.join(output_path, tool_memory_load[0]), "rb")
    tool_memory_load_2 = open(os.path.join(output_path, tool_memory_load[1]), "rb")
    found = False
    for hsa_file in code_object_file_path["hsa_memory_load"]:

        m = re.search(gfx_pattern, hsa_file)
        assert m != None
        gpu = m.group(0)

        if gpu == gpu_name:
            found = True
            hsa_memory_load = open(hsa_file, "rb")
            hsa_memory_fs = hsa_memory_load.read()
            tool_memory_fs_1 = tool_memory_load_1.read()
            tool_memory_fs_2 = tool_memory_load_2.read()
            assert hsa_memory_fs == tool_memory_fs_2 or hsa_memory_fs == tool_memory_fs_1
            break
    assert found == True


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
