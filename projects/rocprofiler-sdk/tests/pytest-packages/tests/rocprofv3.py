# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

from __future__ import absolute_import


def test_perfetto_data(
    pftrace_data, json_data, categories=("hip", "hsa", "marker", "kernel", "memory_copy")
):

    mapping = {
        "hip": ("hip_api", "hip_api"),
        "hsa": ("hsa_api", "hsa_api"),
        "marker": ("marker_api", "marker_api"),
        "kernel": ("kernel_dispatch", "kernel_dispatch"),
        "memory_copy": ("memory_copy", "memory_copy"),
    }

    # make sure they specified valid categories
    for itr in categories:
        assert itr in mapping.keys()

    for pf_category, js_category in [
        itr for key, itr in mapping.items() if key in categories
    ]:
        _pf_data = pftrace_data.loc[pftrace_data["category"] == pf_category]
        _js_data = json_data["rocprofiler-sdk-tool"]["buffer_records"][js_category]

        assert len(_pf_data) == len(
            _js_data
        ), f"{pf_category} ({len(_pf_data)}):\n\t{_pf_data}\n{js_category} ({len(_js_data)}):\n\t{_js_data}"
