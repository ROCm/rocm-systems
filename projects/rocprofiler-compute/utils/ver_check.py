#!/usr/bin/env python3
#
# Support utility to check VERSION file against a tagname. Used in
# release pipeline.

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, required=True, help="tagname to check")
args = parser.parse_args()

execPath = str(Path(__file__).parent)
with open(execPath + "/../VERSION") as f:
    repoVer = f.readline().strip()

repoCheck = "v" + repoVer
tag = args.tag

print("Current repository version = %s" % repoVer)
print("-->  tagname               = %s" % tag)

if repoCheck == tag:
    print("OK: exact match")
    exit(0)
elif tag.startswith(repoCheck + "-"):
    print("OK: allowed match with extra delimiter")
    exit(0)
elif tag.startswith("rocm-"):
    print("OK: allowed match with 'rocm-' prefix")
    exit(0)
else:
    print("FAIL: no match - double check top-level VERSION file")
    exit(1)
