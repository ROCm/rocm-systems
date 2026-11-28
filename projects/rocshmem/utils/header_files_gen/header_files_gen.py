###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from RMA import generate_RMA_header
from AMO import generate_AMO_header
from SIG_OP import generate_SIG_OP_header
from COLL import generate_COLL_header
from P2P_SYNC import generate_P2P_SYNC_header
from RMA_X import generate_RMA_X_header


copyright = """/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/
"""


# Generated headers are committed formatted, so format them the same way.
# The style (and its required version) is projects/rocshmem/.clang-format.
CLANG_FORMAT_VERSION = 22
GENERATED_HEADERS = [
    "rocshmem_RMA.hpp",
    "rocshmem_AMO.hpp",
    "rocshmem_SIG_OP.hpp",
    "rocshmem_COLL.hpp",
    "rocshmem_P2P_SYNC.hpp",
    "rocshmem_RMA_X.hpp",
]


def format_headers(directory):
    """Run clang-format (rocSHMEM style) in place over the generated headers."""
    style = Path(__file__).resolve().parents[2] / ".clang-format"
    exe = (
        os.environ.get("CLANG_FORMAT")
        or shutil.which(f"clang-format-{CLANG_FORMAT_VERSION}")
        or shutil.which("clang-format")
    )
    if not exe:
        sys.exit(
            f"error: clang-format >= {CLANG_FORMAT_VERSION} not found "
            "(set CLANG_FORMAT, or pass --no-format to skip formatting)"
        )
    out = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"(\d+)\.\d+\.\d+", out)
    if not m or int(m.group(1)) < CLANG_FORMAT_VERSION:
        sys.exit(
            f"error: clang-format >= {CLANG_FORMAT_VERSION} required, found: "
            f"{out.strip()}"
        )
    files = [str(Path(directory) / h) for h in GENERATED_HEADERS]
    subprocess.run([exe, "-i", f"--style=file:{style}"] + files, check=True)


def main():
    parser = argparse.ArgumentParser(
        description='Generate an expanded header files.'
    )
    parser.add_argument(
        'directory', type=str, nargs='?', default='.',
        help='Directory to write the header files to (default: current directory)'
    )
    parser.add_argument(
        '--no-format', action='store_true',
        help='Do not run clang-format on the generated headers'
    )
    args = parser.parse_args()

    generate_RMA_header(args.directory, copyright)
    generate_AMO_header(args.directory, copyright)
    generate_SIG_OP_header(args.directory, copyright)
    generate_COLL_header(args.directory, copyright)
    generate_P2P_SYNC_header(args.directory, copyright)
    generate_RMA_X_header(args.directory, copyright)

    if not args.no_format:
        format_headers(args.directory)

if __name__ == "__main__":
    main()
