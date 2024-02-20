import argparse

import os
import sys
import glob

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description="create test_analyze_workloads.py")

    my_parser.add_argument(
        "-p", "--path", dest="path", required=True, type=str, help="Specify directory."
    )

    args = my_parser.parse_args()
    workloads_path = args.path
    workloads = glob.glob(workloads_path + "/*")

    with open("test_analyze_workloads.py", "a") as f:
        with open("test_saved_analysis.py", "a") as g:
                for workload in workloads:
                    workload_name = workload[workload.rfind("/") + 1 :]
                    if (
                        os.path.exists(workload + "/MI100")
                        and len(os.listdir(workload + "/MI100")) > 0
                    ):
                        test = (
                            "\n\ndef test_analyze_"
                            + workload_name
                            + "_MI100():\n    with patch('sys.argv',['omniperf', 'analyze', '--path', '"
                            + workload
                            + "/MI100']): omniperf.main()"
                        )
                        f.write(test)
                        test = (
                            "\n\ndef test_saved_"
                            + workload_name
                            + "_MI100():\n    compare('"
                            + workload
                            + "/MI100/prev_analysis', '"
                            + workload
                            + "/MI100/saved_analysis')"
                        )
                        g.write(test)
                    if (
                        os.path.exists(workload + "/MI200")
                        and len(os.listdir(workload + "/MI200")) > 0
                    ):
                        test = (
                            "\n\ndef test_"
                            + workload_name
                            + "_MI200():\n    with patch('sys.argv',['omniperf', 'analyze', '--path', '"
                            + workload
                            + "/MI200']): omniperf.main()"
                        )
                        f.write(test)
                        test = (
                            "\n\ndef test_saved_"
                            + workload_name
                            + "_MI200():\n    compare('"
                            + workload
                            + "/MI200/prev_analysis', '"
                            + workload
                            + "/MI200/saved_analysis')"
                        )
                        g.write(test)
