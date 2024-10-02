import pandas as pd
import os
import sys
import pytest


def test_agent_info(agent_info_input_data):
    logical_node_id = max([int(itr["Logical_Node_Id"]) for itr in agent_info_input_data])

    assert logical_node_id + 1 == len(agent_info_input_data)

    for row in agent_info_input_data:
        agent_type = row["Agent_Type"]
        assert agent_type in ("CPU", "GPU")
        if agent_type == "CPU":
            assert int(row["Cpu_Cores_Count"]) > 0
            assert int(row["Simd_Count"]) == 0
            assert int(row["Max_Waves_Per_Simd"]) == 0
        else:
            assert int(row["Cpu_Cores_Count"]) == 0
            assert int(row["Simd_Count"]) > 0
            assert int(row["Max_Waves_Per_Simd"]) > 0


def test_validate_counter_collection_yml_pmc(counter_input_data):
    counter_names = ["SQ_WAVES", "GRBM_COUNT", "GRBM_GUI_ACTIVE"]
    di_list = []

    for row in counter_input_data:
        assert int(row["Agent_Id"]) > 0
        assert int(row["Queue_Id"]) > 0
        assert int(row["Process_Id"]) > 0
        assert len(row["Kernel_Name"]) > 0

        assert len(row["Counter_Value"]) > 0
        # assert row["Counter_Name"].contains("SQ_WAVES").all()
        assert row["Counter_Name"] in counter_names
        assert float(row["Counter_Value"]) > 0

        di_list.append(int(row["Dispatch_Id"]))

    # # make sure the dispatch ids are unique and ordered
    di_list = list(dict.fromkeys(di_list))
    di_expect = [idx + 1 for idx in range(len(di_list))]
    assert di_expect == di_list


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
