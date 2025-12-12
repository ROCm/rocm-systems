#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT


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
