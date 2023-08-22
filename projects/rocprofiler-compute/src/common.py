##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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
##############################################################################el

import os
import sys
import io
from pathlib import Path
import subprocess
import shutil

OMNIPERF_HOME = Path(__file__).resolve().parent

# OMNIPERF INFO
PROG = "omniperf"
SOC_LIST = ["mi50", "mi100", "mi200", "vega10"]
DISTRO_MAP = {"platform:el8": "rhel8", "15.3": "sle15sp3", "20.04": "ubuntu20_04"}


def resolve_rocprof(returnPath=False):
    # ROCPROF INFO
    if not "ROCPROF" in os.environ.keys():
        rocprof_cmd = "rocprof"
    else:
        rocprof_cmd = os.environ["ROCPROF"]

    rocprof_path = subprocess.run(
        ["which", rocprof_cmd], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    if rocprof_path.returncode != 0:
        print("\nError: Unable to resolve path to %s binary" % rocprof_cmd)
        print(
            "Please verify installation or set ROCPROF environment variable with full path."
        )
        sys.exit(1)
    else:
        # Resolve any sym links in file path
        rocprof_path = os.path.realpath(rocprof_path.stdout.decode("utf-8").rstrip("\n"))
        print("ROC Profiler: ", rocprof_path)
        if returnPath:
            return rocprof_path
        else:
            return rocprof_cmd


def getVersion():
    # symantic version info
    version = os.path.join(OMNIPERF_HOME.parent, "VERSION")
    try:
        with open(version, "r") as file:
            VER = file.read().replace("\n", "")
    except EnvironmentError:
        print("ERROR: Cannot find VERSION file at {}".format(version))
        sys.exit(1)

    # git version info
    gitDir = os.path.join(OMNIPERF_HOME.parent, ".git")
    if (shutil.which("git") is not None) and os.path.exists(gitDir):
        gitQuery = subprocess.run(
            ["git", "log", "--pretty=format:%h", "-n", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if gitQuery.returncode != 0:
            SHA = "unknown"
            MODE = "unknown"
        else:
            SHA = gitQuery.stdout.decode("utf-8")
            MODE = "dev"
    else:
        shaFile = os.path.join(OMNIPERF_HOME.parent, "VERSION.sha")
        try:
            with open(shaFile, "r") as file:
                SHA = file.read().replace("\n", "")
        except EnvironmentError:
            print("ERROR: Cannot find VERSION.sha file at {}".format(shaFile))
            sys.exit(1)

        MODE = "release"

    versionData = {"version": VER, "sha": SHA, "mode": MODE}
    return versionData


def getVersionDisplay(version, sha, mode):
    buf = io.StringIO()
    print("-" * 40, file=buf)
    print("Omniperf version: %s (%s)" % (version, mode), file=buf)
    print("Git revision:     %s" % sha, file=buf)
    print("-" * 40, file=buf)
    return buf.getvalue()
