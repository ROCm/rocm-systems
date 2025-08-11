import pandas as pd
import sys


def validate_grbm_count(filename):
    df = pd.read_csv(filename)

    grbm_count = df.loc[0, "GRBM_COUNT"]

    # Validate the data
    if not grbm_count < 0:
        print("Test Passed: grbm count is valid.")
        return 0
    else:
        print("Test Failed: grbm count is not valid.")
        return 1


if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        raise RuntimeError("no input files provided")
    for filename in files:
        ec = validate_grbm_count(filename)
        if ec != 0:
            sys.stderr.write(f"{filename} did not pass validation\n")
            sys.exit(ec)
