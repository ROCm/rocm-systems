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
                archs = os.listdir(workload)
                for arch in archs:
                    test = (
                        "\n\ndef test_analyze_"
                        + workload_name
                        + "_"+arch+"():"+
                        "\n\twith pytest.raises(SystemExit) as e:"+
                        "\n\t\twith patch('sys.argv',['omniperf', 'analyze', '--path', '"
                        + workload
                        + "/"+arch+"']):\n\t\t\tomniperf.main()"+
                        "\n\tassert e.value.code == 0"
                    )
                    f.write(test)
                    test = (
                        "\n\ndef test_saved_"
                        + workload_name
                        + "_"+arch+"():\n\tcompare('"
                        + workload
                        + "/"+arch+"/prev_analysis', '"
                        + "/"+arch+"/prev_analysis', '"
                        + workload
                        + "/"+arch+"/saved_analysis')"
                        + "/"+arch+"/saved_analysis')"
                    )
                    g.write(test)
