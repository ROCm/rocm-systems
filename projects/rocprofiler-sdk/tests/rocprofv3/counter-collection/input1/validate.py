import pandas as pd
import sys
import pytest


def test_validate_counter_collection_pmc1(input_data: pd.DataFrame):
    df = input_data

    assert df.empty == False
    assert df["agent-id"].map(type).eq(int).all()
    assert len(df["kernel-name"]) > 0


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
