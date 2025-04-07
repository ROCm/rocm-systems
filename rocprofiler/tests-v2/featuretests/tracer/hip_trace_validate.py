import pandas as pd
import sys


def validate_hip_trace(filename):
    df = pd.read_csv(filename)

    start_time = df.loc[0, "Start_Timestamp"]
    end_time = df.loc[0, "End_Timestamp"]

    # Validate the data
    if start_time < end_time:
        print("Test Passed: Time stamps are valid.")
        return 0
    else:
        print("Test Failed: Time stamps are not valid.")
        return 1


if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        raise RuntimeError("no input files provided")
    for filename in files:
        ec = validate_hip_trace(filename)
        if ec != 0:
            sys.stderr.write(f"{filename} did not pass validation\n")
            sys.exit(ec)
