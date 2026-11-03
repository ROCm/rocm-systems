#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest
import json

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list

import re
import os


def pytest_addoption(parser):

    parser.addoption(
        "--input",
        action="store",
        help="Path to JSON file.",
    )
    parser.addoption(
        "--code-object-input",
        action="store",
        help="Path to code object file.",
    )
    parser.addoption(
        "--output-path",
        action="store",
        help="Output Path.",
    )
    parser.addoption(
        "--target-cu",
        action="store",
        default=None,
        help="Target CU for perfcounter validation.",
    )
    parser.addoption(
        "--att-occupancy-event-trace-out-dir",
        action="store",
        help="Path to ATT occupancy event trace output directory.",
    )
    parser.addoption(
        "--att-other-simd-out-dir",
        action="store",
        help="Path to Output directory.",
    )
    parser.addoption(
        "--att-shaderdata-out-dir",
        action="store",
        help="Path to Output directory.",
    )
    parser.addoption(
        "--att-multi-gpu-out-dir",
        action="store",
        help="Path to multi-GPU ATT output directory.",
    )
    parser.addoption(
        "--att-marker-trace-out-dir",
        action="store",
        help="Path to ATT marker trace output directory.",
    )
    parser.addoption(
        "--att-no-intercept-out-dir",
        action="store",
        help="Path to ATT no-intercept output directory.",
    )
    parser.addoption(
        "--att-no-detail-out-dir",
        action="store",
        help="Path to ATT no-detail output directory.",
    )
    parser.addoption(
        "--att-pmc-input",
        action="store",
        help="Path to JSON file from a combined ATT + counter collection run.",
    )


@pytest.fixture
def json_data(request):
    filename = request.config.getoption("--input")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def output_path(request):
    return request.config.getoption("--output-path")


@pytest.fixture
def att_occupancy_event_trace_out_dir_path(request):
    output_dir_path = request.config.getoption("--att-occupancy-event-trace-out-dir")
    if not output_dir_path:
        pytest.skip("--att-occupancy-event-trace-out-dir not provided")
    return output_dir_path


@pytest.fixture
def code_object_file_path(request):
    file_path = request.config.getoption("--code-object-input")
    if file_path is None:
        pytest.skip("--code-object-input not provided")
    # hsa_file_load = re.compile(".*copy.hsaco$")
    code_object_files = {}
    code_object_memory = []
    hsa_memory_load_pattern = "gfx[a-zA-Z0-9_-]+_copy_memory.hsaco"
    for root, dirs, files in os.walk(file_path, topdown=True):
        for file in files:
            filename = os.path.join(root, file)
            if re.search(hsa_memory_load_pattern, filename):
                code_object_memory.append(filename)
    code_object_files["hsa_memory_load"] = code_object_memory
    return code_object_files


@pytest.fixture
def att_other_simd_out_dir_path(request):
    output_dir_path = request.config.getoption("--att-other-simd-out-dir")
    if not output_dir_path:
        pytest.skip("--att-other-simd-out-dir not provided")
    return output_dir_path


@pytest.fixture
def att_shaderdata_out_dir_path(request):
    output_dir_path = request.config.getoption("--att-shaderdata-out-dir")
    if not output_dir_path:
        pytest.skip("--att-shaderdata-out-dir not provided")
    return output_dir_path


@pytest.fixture
def att_multi_gpu_out_dir_path(request):
    output_dir_path = request.config.getoption("--att-multi-gpu-out-dir")
    if not output_dir_path:
        pytest.skip("--att-multi-gpu-out-dir not provided")
    return output_dir_path


@pytest.fixture
def att_marker_trace_out_dir_path(request):
    output_dir_path = request.config.getoption("--att-marker-trace-out-dir")
    if not output_dir_path:
        pytest.skip("--att-marker-trace-out-dir not provided")
    return output_dir_path


def cpu_supports_att_quick_scan():
    """The ATT quick-scan decoder needs a CPU with AVX-512 (vbmi, bw and f).

    On other CPUs rocprofv3 warns and falls back to --att, so there is no
    no-intercept output to validate. If the CPU flags cannot be read, assume
    that quick-scan is supported and let the test run.
    """
    ret = True
    required = {"avx512vbmi", "avx512bw", "avx512f"}
    try:
        with open("/proc/cpuinfo", "r") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("flags"):
                    ret = required.issubset(set(line.split(":", 1)[1].split()))
                    break
    except OSError:
        pass
    return ret


@pytest.fixture
def att_no_intercept_out_dir_path(request):
    output_dir_path = request.config.getoption("--att-no-intercept-out-dir")
    if not output_dir_path:
        pytest.skip("--att-no-intercept-out-dir not provided")
    if not cpu_supports_att_quick_scan():
        pytest.skip("ATT no-intercept quick-scan needs a CPU with AVX-512")
    return output_dir_path


@pytest.fixture
def att_no_detail_out_dir_path(request):
    output_dir_path = request.config.getoption("--att-no-detail-out-dir")
    if not output_dir_path:
        pytest.skip("--att-no-detail-out-dir not provided")
    return output_dir_path


@pytest.fixture
def att_pmc_json_data(request):
    filename = request.config.getoption("--att-pmc-input")
    if not filename:
        pytest.skip("--att-pmc-input not provided")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))
