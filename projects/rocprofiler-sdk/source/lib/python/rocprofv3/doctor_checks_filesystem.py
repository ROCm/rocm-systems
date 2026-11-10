# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Filesystem checks for the profiler's output destination."""

from __future__ import absolute_import

from rocprofv3.doctor_registry import register
from rocprofv3.doctor_result import (
    make_fail,
    make_pass,
    make_warn,
    SEV_ERROR,
    SEV_WARNING,
)

MB = 1024 * 1024
WARN_FREE_BYTES = 500 * MB
FAIL_FREE_BYTES = 50 * MB


def output_path(accessor):
    """The directory rocprofv3 would write its output into."""
    return accessor.getenv("ROCPROF_OUTPUT_PATH") or accessor.realpath(".")


def _format_size(num_bytes):
    if num_bytes >= 1024 * MB:
        return "{:.1f} GB".format(float(num_bytes) / (1024 * MB))
    return "{:.0f} MB".format(float(num_bytes) / MB)


@register(
    id="filesystem.output-dir-writable",
    group="filesystem",
    title="Output directory writable",
    severity=SEV_ERROR,
    order=80,
)
def check_output_dir_writable(accessor):
    path = output_path(accessor)
    data = {"output_path": path, "from_env": accessor.getenv("ROCPROF_OUTPUT_PATH")}

    if not accessor.path_exists(path):
        return make_warn(
            "output directory {} does not exist yet; rocprofv3 will try to "
            "create it".format(path),
            "mkdir -p {}".format(path),
            data,
        )

    if not accessor.path_is_dir(path):
        return make_fail(
            "output path {} exists but is not a directory".format(path),
            "export ROCPROF_OUTPUT_PATH=/path/to/a/directory",
            data,
        )

    if not accessor.touch_probe(path):
        return make_fail(
            "cannot create files in {}".format(path),
            "Choose a writable output directory:\n"
            "  rocprofv3 -d /tmp/rocprof-output ...\n"
            "or fix the permissions:\n"
            "  sudo chown $USER {}".format(path),
            data,
        )
    return make_pass("{} is writable".format(path), "", data)


@register(
    id="filesystem.output-disk-space",
    group="filesystem",
    title="Sufficient free disk space for output",
    severity=SEV_WARNING,
    depends=["filesystem.output-dir-writable"],
    order=81,
)
def check_output_disk_space(accessor):
    path = output_path(accessor)
    free = accessor.free_bytes(path)
    data = {"output_path": path, "free_bytes": free}

    if free is None:
        return make_warn("could not determine free space on {}".format(path), "", data)

    data["free_human"] = _format_size(free)

    if free < FAIL_FREE_BYTES:
        return make_fail(
            "only {} free on {} -- trace output will very likely fail".format(
                data["free_human"], path
            ),
            "Free up space, or write elsewhere:\n  rocprofv3 -d /path/with/space ...",
            data,
        )
    if free < WARN_FREE_BYTES:
        return make_warn(
            "only {} free on {}; large traces may fill the filesystem".format(
                data["free_human"], path
            ),
            "Free up space, or write elsewhere:\n  rocprofv3 -d /path/with/space ...",
            data,
        )
    return make_pass("{} free on {}".format(data["free_human"], path), "", data)


@register(
    id="filesystem.tmp-writable",
    group="filesystem",
    title="/tmp writable",
    severity=SEV_WARNING,
    order=82,
)
def check_tmp_writable(accessor):
    tmpdir = accessor.getenv("TMPDIR") or "/tmp"
    data = {"tmpdir": tmpdir}

    if not accessor.path_is_dir(tmpdir):
        return make_warn(
            "{} is not a directory".format(tmpdir), "export TMPDIR=/tmp", data
        )
    if not accessor.touch_probe(tmpdir):
        return make_warn(
            "cannot create files in {}".format(tmpdir),
            "export TMPDIR=$HOME/tmp && mkdir -p $TMPDIR",
            data,
        )
    return make_pass("{} is writable".format(tmpdir), "", data)
