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

import validate


def _make_inputs(tmp_path, candidates):
    snapshot_bytes = b"matching-code-object"
    snapshot_name = "gfx1250_snapshot.co"
    (tmp_path / snapshot_name).write_bytes(snapshot_bytes)

    hsa_files = []
    for filename, payload in candidates:
        path = tmp_path / filename
        path.write_bytes(payload)
        hsa_files.append(str(path))

    code_object_file_path = {"hsa_memory_load": hsa_files}
    json_data = {
        "rocprofiler-sdk-tool": {
            "strings": {
                "code_object_snapshot_filenames": ["unused", snapshot_name],
            }
        }
    }
    return code_object_file_path, json_data, str(tmp_path)


def test_nonmatching_same_prefix_before_matching(tmp_path):
    code_object_file_path, json_data, output_path = _make_inputs(
        tmp_path,
        [
            ("gfx1250-strict_copy_memory.hsaco", b"non-matching-strict"),
            ("gfx1250_copy_memory.hsaco", b"matching-code-object"),
        ],
    )
    validate.test_code_object_memory(code_object_file_path, json_data, output_path)


def test_matching_same_prefix_first(tmp_path):
    code_object_file_path, json_data, output_path = _make_inputs(
        tmp_path,
        [
            ("gfx1250_copy_memory.hsaco", b"matching-code-object"),
            ("gfx1250-strict_copy_memory.hsaco", b"non-matching-strict"),
        ],
    )
    validate.test_code_object_memory(code_object_file_path, json_data, output_path)


def test_all_same_prefix_candidates_mismatch(tmp_path):
    code_object_file_path, json_data, output_path = _make_inputs(
        tmp_path,
        [
            ("gfx1250-strict_copy_memory.hsaco", b"non-matching-strict"),
            ("gfx1250_copy_memory.hsaco", b"non-matching-base"),
        ],
    )
    with pytest.raises(AssertionError):
        validate.test_code_object_memory(code_object_file_path, json_data, output_path)
