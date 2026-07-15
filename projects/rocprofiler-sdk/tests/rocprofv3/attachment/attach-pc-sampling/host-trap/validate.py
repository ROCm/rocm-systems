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

# Validate rocprofv3 host_trap PC sampling output collected in attach mode.

import sys

import pytest

from rocprofiler_sdk.pc_sampling.attach.csv import normalize_attach_csv
from rocprofiler_sdk.pc_sampling.attach.json import validate_attach_json
from rocprofiler_sdk.pc_sampling.exec_mask_manipulation.csv import (
    exec_mask_manipulation_validate_csv,
)

METHOD = "host_trap"


def test_validate_pc_sampling_attach_csv(input_csv, wave_size):
    # normalize the attach-mode distortions, then reuse the shared exec-mask checks
    df, num_kernels = normalize_attach_csv(input_csv)
    exec_mask_manipulation_validate_csv(df, wave_size=wave_size, num_kernels=num_kernels)


def test_validate_pc_sampling_attach_json(input_csv, input_json):
    # parity plus basic per-record checks; the full exec-mask JSON validator
    # assumes a complete run, so it is not applied to attach captures
    validate_attach_json(input_json, METHOD, len(input_csv))


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
