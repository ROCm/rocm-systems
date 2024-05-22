import sys
import pytest


def test_validate_counter_collection_plus_tracing(
    json_data, counter_input_data, hsa_input_data
):

    # check if either kernel-name/FUNCTION is present
    assert (
        "Kernel_Name" in counter_input_data.columns
        or "Function" in counter_input_data.columns
    )

    data = json_data["rocprofiler-sdk-tool"]
    hsa_api = data["buffer_records"]["hsa_api"]
    assert len(hsa_input_data) == len(hsa_api)

    counter_collection_data = data["callback_records"]["counter_collection"]
    assert len(counter_collection_data) > 0


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
