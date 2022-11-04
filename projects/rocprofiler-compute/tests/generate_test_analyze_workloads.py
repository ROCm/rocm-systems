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

    with open("cmake/test_analyze_workloads.py", "a") as f:
        with open("cmake/test_import_workloads.py", "a") as g:
            with open("cmake/test_saved_analysis.py", "a") as h:
                for workload in workloads:
                    workload_name = workload[workload.rfind("/") + 1 :]
                    if (
                        os.path.exists(workload + "/mi100")
                        and len(os.listdir(workload + "/mi100")) > 0
                    ):
                        test = (
                            "\n\ndef test_analyze_"
                            + workload_name
                            + "_mi100():\n    with patch('sys.argv',['omniperf', 'analyze', '--path', '"
                            + workload
                            + "/mi100']): omniperf.main()"
                        )
                        f.write(test)
                        test = (
                            "\n\ndef test_import_"
                            + workload_name
                            + "_mi100():\n    with patch('sys.argv',['omniperf', 'database', '--import', '-H', 'localhost', '-u', 'amd', '-p', 'amd123', '-t', 'asw', '-w', '"
                            + workload
                            + "/mi100']): omniperf.main()"
                        )
                        g.write(test)
                        test = (
                            "\n\ndef test_saved_"
                            + workload_name
                            + "_mi100():\n    compare('"
                            + workload
                            + "/mi100/prev_analysis', '"
                            + workload
                            + "/mi100/saved_analysis')"
                        )
                        h.write(test)
                    if (
                        os.path.exists(workload + "/mi200")
                        and len(os.listdir(workload + "/mi200")) > 0
                    ):
                        test = (
                            "\n\ndef test_"
                            + workload_name
                            + "_mi200():\n    with patch('sys.argv',['omniperf', 'analyze', '--path', '"
                            + workload
                            + "/mi200']): omniperf.main()"
                        )
                        f.write(test)
                        test = (
                            "\n\ndef test_"
                            + workload_name
                            + "_mi100():\n    with patch('sys.argv',['omniperf', 'database', '--import', '-H', 'localhost', '-u', '-p', 'amd123', 'amd', '-t', 'asw', '-w', '"
                            + workload
                            + "/mi100']): omniperf.main()"
                        )
                        g.write(test)
                        test = (
                            "\n\ndef test_saved_"
                            + workload_name
                            + "_mi200():\n    compare('"
                            + workload
                            + "/mi200/prev_analysis', '"
                            + workload
                            + "/mi200/saved_analysis')"
                        )
                        h.write(test)
