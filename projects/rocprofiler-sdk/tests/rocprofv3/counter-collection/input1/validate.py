import pandas as pd
import sys
import pytest

kernel_list = ["addition_kernel", "subtract_kernel", "multiply_kernel", "divide_kernel"]


def test_validate_counter_collection_pmc1(input_data: pd.DataFrame):
    df = input_data

    assert df.empty == False
    assert (df["Agent_Id"].astype(int).values > 0).all()
    assert (df["Queue_Id"].astype(int).values > 0).all()
    assert (df["Process_Id"].astype(int).values > 0).all()
    assert len(df["Kernel-Name"]) > 0
    df_list = df["Kernel-Name"].values.flatten().tolist()
    # Check if each string in kernel_list is present at least once
    missing_kernels = []
    for kernel in kernel_list:
        if kernel not in df_list:
            missing_kernels.append(kernel)

    assert (
        not missing_kernels
    ), f"The following kernel names are missing from the out file: {missing_kernels}"
    assert df["Counter_Name"].str.contains("SQ_WAVES").all()
    assert len(df["Counter_Value"]) > 0


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
