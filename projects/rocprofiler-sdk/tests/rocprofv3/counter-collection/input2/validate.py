import pandas as pd
import os
import sys
import pytest


def test_validate_counter_collection_pmc2(input_dir: pd.DataFrame):
    directory_path = input_dir

    # Check if the directory is not empty
    assert os.path.isdir(directory_path), f"{directory_path} is not a directory."
    assert os.listdir(directory_path), f"{directory_path} is empty."

    # Check if there are 2 subdirectories pmc_1 and pmc_2
    subdirectories = [
        d
        for d in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, d))
    ]
    assert (
        len(subdirectories) == 2
    ), f"Expected 2 subdirectories, found {len(subdirectories)}."

    # Check if each subdirectory has files
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(directory_path, subdirectory)
        assert os.listdir(subdirectory_path), f"{subdirectory_path} is empty."

        # Check if each file in the subdirectory has some data
        for file_name in os.listdir(subdirectory_path):
            file_path = os.path.join(subdirectory_path, file_name)
            assert os.path.isfile(file_path), f"{file_path} is not a file."

            with open(file_path, "r") as file:
                df = pd.read_csv(file)
                # check if kernel-name is present
                assert len(df["kernel-name"]) > 0
                # check if counter value is positive
                assert len(df["counter_value"]) > 0


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
