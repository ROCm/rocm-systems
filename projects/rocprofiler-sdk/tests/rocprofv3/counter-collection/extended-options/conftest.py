#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

import csv
import json

import pytest


def pytest_addoption(parser):
    parser.addoption("--reference-input", action="store")
    parser.addoption("--truncated-input", action="store")
    parser.addoption("--mangled-input", action="store")
    parser.addoption("--counter-input", action="store")
    parser.addoption("--json-input", action="store")
    parser.addoption("--agent-index", action="store")
    parser.addoption("--context-input", action="store")
    parser.addoption("--context-json", action="store")
    parser.addoption("--context-pftrace", action="store")
    parser.addoption("--post-input", action="store")
    parser.addoption("--openmp-input", action="store")
    parser.addoption("--device-input", action="store")


def _read_csv(request, option):
    with open(
        request.config.getoption(option), newline="", encoding="utf-8"
    ) as input_file:
        return list(csv.DictReader(input_file))


@pytest.fixture
def reference_rows(request):
    return _read_csv(request, "--reference-input")


@pytest.fixture
def truncated_rows(request):
    return _read_csv(request, "--truncated-input")


@pytest.fixture
def mangled_rows(request):
    return _read_csv(request, "--mangled-input")


@pytest.fixture
def counter_rows(request):
    return _read_csv(request, "--counter-input")


@pytest.fixture
def json_data(request):
    with open(request.config.getoption("--json-input"), encoding="utf-8") as input_file:
        return json.load(input_file)


@pytest.fixture
def agent_index(request):
    return request.config.getoption("--agent-index")


@pytest.fixture
def context_rows(request):
    return _read_csv(request, "--context-input")


@pytest.fixture
def context_json(request):
    with open(request.config.getoption("--context-json"), encoding="utf-8") as input_file:
        return json.load(input_file)


@pytest.fixture
def context_pftrace(request):
    return request.config.getoption("--context-pftrace")


@pytest.fixture
def post_rows(request):
    return _read_csv(request, "--post-input")


@pytest.fixture
def openmp_rows(request):
    return _read_csv(request, "--openmp-input")


@pytest.fixture
def device_rows(request):
    return _read_csv(request, "--device-input")
