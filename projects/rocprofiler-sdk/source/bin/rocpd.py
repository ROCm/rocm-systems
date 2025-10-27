#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import sys

"""
Simple Python executable script for invoking `python3 -m @ROCPD_EXE_MODULE@`
"""


def main(argv=sys.argv[1:], environ=dict(os.environ)):
    """
    Executes {sys.executable} -m @ROCPD_EXE_MODULE@ @ROCPD_EXE_MODULE_ARGS@

    Pure Python implementation - works with all Python 3.6+ versions
    """

    ROCPD_MODULE_ARGS = [f"{itr}" for itr in "@ROCPD_EXE_MODULE_ARGS@".split(" ") if itr]

    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Pure Python implementation: version-agnostic path
    # No longer requires version-specific builds (python3.8, python3.9, etc.)
    module_path = os.path.join(
        this_dir,
        "..",
        "@CMAKE_INSTALL_LIBDIR@",
        "python",  # Version-agnostic: works with all Python 3.6+
        "site-packages",
    )

    python_path = [module_path] + os.environ.get("PYTHONPATH", "").split(":")

    # update PYTHONPATH environment variable
    environ["PYTHONPATH"] = ":".join(python_path)

    args = [f"{sys.executable}", "-m", "@ROCPD_EXE_MODULE@"] + ROCPD_MODULE_ARGS + argv

    # does not return
    os.execvpe(args[0], args, env=environ)


if __name__ == "__main__":
    main()
