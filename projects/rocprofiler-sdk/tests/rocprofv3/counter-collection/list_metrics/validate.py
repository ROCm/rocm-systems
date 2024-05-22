import pandas as pd
import sys
import pytest


def test_validate_list_basic_metrics(basic_metrics_input_data):
    for row in basic_metrics_input_data:
        assert row["Agent_Id"].isdigit() == True
        assert row["Name"] != ""
        assert row["Description"] != ""
        assert row["Block"] != ""
        assert row["Dimensions"] != ""
        if row["Name"] == "SQ_WAVES":
            row[
                "Description"
            ] == "Count number of waves sent to SQs. (per-simd, emulated, global)"
            row["Block"] == "SQ"


def test_validate_list_derived_metrics(derived_metrics_input_data):
    for row in derived_metrics_input_data:
        assert row["Agent_Id"].isdigit() == True
        assert row["Name"] != ""
        assert row["Description"] != ""
        assert row["Expression"] != ""
        assert row["Dimensions"] != ""
        if row["Name"] == "TA_BUSY_min":
            row["Description"] == "TA block is busy. Min over TA instances."
            row["Expression"] == "reduce(TA_TA_BUSY,min)"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
