import pandas as pd
import sys
import pytest


def test_validate_counter_collection_pmc1(input_data: pd.DataFrame):
    df = input_data

    assert df.empty == False
    assert df["Agent_Id"].map(type).eq(int).all()
    assert len(df["Kernel-Name"]) > 0
    assert df["Kernel-Name"].str.contains("matrixTranspose").all()
    assert df["Counter_Name"].str.contains("SQ_WAVES").all()


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
