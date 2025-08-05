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


import argparse
import glob
import os

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description="create test_analyze_workloads.py")

    my_parser.add_argument(
        "-p", "--path", dest="path", required=True, type=str, help="Specify directory."
    )

    args = my_parser.parse_args()
    workloads_path = args.path
    workloads = glob.glob(workloads_path + "/*")

    with open("test_analyze_workloads.py", "a") as f:
        for workload in workloads:
            workload_name = workload[workload.rfind("/") + 1 :]
            archs = os.listdir(workload)
            for arch in archs:
                test = (
                    "\n\ndef test_analyze_"
                    + workload_name
                    + "_"
                    + arch
                    + "():"
                    + "\n\twith pytest.raises(SystemExit) as e:"
                    + "\n\t\twith patch('sys.argv',['rocprof-compute', 'analyze', '--path', '"
                    + workload
                    + "/"
                    + arch
                    + "']):\n\t\t\trocprof_compute.main()"
                    + "\n\tassert e.value.code == 0"
                )
                f.write(test)
