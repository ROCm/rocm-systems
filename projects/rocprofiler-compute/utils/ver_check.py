##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

##############################################################################

#!/usr/bin/env python3
#
# Support utility to check VERSION file against a tagname. Used in
# release pipeline.

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, required=True, help="tagname to check")
args = parser.parse_args()

exec_path = Path(__file__).parent
with open(exec_path / "../VERSION") as f:
    repo_ver = f.readline().strip()

repo_check = f"v{repo_ver}"
tag = args.tag

print(f"Current repository version = {repo_ver}")
print(f"-->  tagname               = {tag}")

if repo_check == tag:
    print("OK: exact match")
    exit(0)
elif tag.startswith(repo_check + "-"):
    print("OK: allowed match with extra delimiter")
    exit(0)
elif tag.startswith("rocm-"):
    print("OK: allowed match with 'rocm-' prefix")
    exit(0)
else:
    print("FAIL: no match - double check top-level VERSION file")
    exit(1)
