# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
from pathlib import Path

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description="create test_analyze_workloads.py")

    my_parser.add_argument(
        "-p", "--path", dest="path", required=True, type=str, help="Specify directory."
    )

    args = my_parser.parse_args()
    workloads_path = Path(args.path)

    with open("test_analyze_workloads.py", "a") as f:
        for workload in sorted(workloads_path.iterdir()):
            workload_name = workload.name
            archs = [p.name for p in workload.iterdir()]
            for arch in archs:
                test = (
                    "\n\ndef test_analyze_"
                    + workload_name
                    + "_"
                    + arch
                    + "():"
                    + "\n\twith pytest.raises(SystemExit) as e:"
                    + (
                        "\n\t\twith patch("
                        "'sys.argv',"
                        "["
                        "'rocprof-compute', "
                        "'analyze', "
                        "'--path', "
                        "'" + str(workload / arch) + "']"
                        "):\n\t\t\trocprof_compute.main()"
                    )
                    + "\n\tassert e.value.code == 0"
                )
                f.write(test)
