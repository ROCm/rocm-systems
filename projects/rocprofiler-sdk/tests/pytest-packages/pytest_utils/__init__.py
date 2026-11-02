# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

import json
import os
import shutil
import subprocess


def collapse_dict_list(data, key="rocprofiler-sdk-tool"):
    """Collapse a dictionary entry list into a single mapped value"""

    def check_return(_data):
        assert isinstance(_data, dict), "expected dict, type: {}".format(
            type(_data).__name__
        )
        return _data

    if (
        key in data.keys()
        and len(data.keys()) == 1
        and isinstance(data[key], (list, tuple))
        and len(data[key]) == 1
    ):
        return check_return({key: data[key][0]})

    return check_return(data)


def gpu_uses_auto_perf_level(agent):
    """Query GPU 0 only, using its agent metadata to limit the check to gfx11/gfx12."""
    if not agent.get("name", "").startswith(("gfx11", "gfx12")):
        return False

    amd_smi = shutil.which("amd-smi") or os.path.join(
        os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "amd-smi"
    )
    try:
        result = subprocess.run(
            [amd_smi, "metric", "--gpu", "0", "--perf-level", "--json"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=5,
        )
        metrics = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        # An unavailable performance level must not hide a counter regression.
        return False
    if isinstance(metrics, dict):
        metrics = metrics.get("gpu_data", [])
    return (
        isinstance(metrics, list)
        and len(metrics) == 1
        and isinstance(metrics[0], dict)
        and metrics[0].get("perf_level") == "AMDSMI_DEV_PERF_LEVEL_AUTO"
    )
