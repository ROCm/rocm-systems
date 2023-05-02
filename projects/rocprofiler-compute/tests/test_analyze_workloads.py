import pytest
from unittest.mock import patch
from importlib.machinery import SourceFileLoader

omniperf = SourceFileLoader("omniperf", "src/omniperf").load_module()

##################################################
##          Generated tests                     ##
##################################################


def test_analyze_D_str_inv1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_str_inv1/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_D_str_inv1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_str_inv1/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_dev01p3_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/dev01p3/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_SQC_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/SQC/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_SQC_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/SQC/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_Axes2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/Axes2/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_Axes2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/Axes2/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_SQ_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_SQ/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_SQ_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_SQ/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_CPF_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/CPF/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_CPF_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/CPF/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_LDS_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_LDS/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_LDS_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_LDS/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_D_str_inv4_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_str_inv4/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_D_str_inv4_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_str_inv4/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_K_int_inv2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_K_int_inv2/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_SPI_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_SPI/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_SPI_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_SPI/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_K_str_valid_2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/no_roof_K_str_valid_2/mi100",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_K_str_valid_2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/no_roof_K_str_valid_2/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_mixbench1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_mixbench1/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_mixbench1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_mixbench1/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_TA_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_TA/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_TA_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_TA/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_CPF_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_CPF/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_CPF_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_CPF/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_CPC_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_CPC/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_CPC_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_CPC/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_K_str_inv3_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/K_str_inv3/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_K_str_inv3_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/K_str_inv3/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_LDS_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/LDS/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_LDS_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/LDS/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_K_str_valid_3_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/no_roof_K_str_valid_3/mi100",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_D_int_inv2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_D_int_inv2/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_K_str_inv1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_K_str_inv1/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_SQC_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_SQC/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_Axes2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_Axes2/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_Axes2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_Axes2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_HBM_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/HBM/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_HBM_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/HBM/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_TA_CPC_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_TA_CPC/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_D_val_int_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_D_val_int/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_L2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_L2/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_L2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_L2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_L2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/L2/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_L2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/L2/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_dev1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_dev1/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_dev1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_dev1/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_K_str_inv3_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_K_str_inv3/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_K_str_valid_1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_K_str_valid_1/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_CPC_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_CPC/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_Axes3_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_Axes3/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_Axes3_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_Axes3/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_D_str_inv3_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_str_inv3/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_D_str_inv3_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_str_inv3/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_D_int_inv2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_int_inv2/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_D_int_inv2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_int_inv2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_TD_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/TD/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_TD_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/TD/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_D_int_inv1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_D_int_inv1/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_D_val_int2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_val_int2/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_D_val_int2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_val_int2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_mixbench2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_mixbench2/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_mixbench2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_mixbench2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_SPI_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_SPI/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_D_val_int2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_val_int2/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_D_val_int2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_val_int2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_K_str_inv1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/K_str_inv1/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_K_str_inv1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/K_str_inv1/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_TA_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_TA/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_K_str_valid_3_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/K_str_valid_3/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_SQ_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/SQ/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_SQ_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/SQ/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_D_str_inv1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_str_inv1/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_D_str_inv1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_str_inv1/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_dev01p3_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_dev01p3/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_D_val_int2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_D_val_int2/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_D_str_inv4_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_str_inv4/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_D_str_inv4_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_str_inv4/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_CPF_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_CPF/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_mixbench_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/mixbench/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_mixbench_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/mixbench/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_D_str_inv4_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_D_str_inv4/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_kernels_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_kernels/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_kernels_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_kernels/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_TCC_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_TCC/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_TA_CPC_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/TA_CPC/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_TA_CPC_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/TA_CPC/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_SQ_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_SQ/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_K_int_inv2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/K_int_inv2/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_K_int_inv2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/K_int_inv2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_TCP_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_TCP/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_K_str_valid_2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_K_str_valid_2/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_D_int_inv2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_int_inv2/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_D_int_inv2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_int_inv2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_Axes3_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_Axes3/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_dev0_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/dev0/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_dev0_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/dev0/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_K_str_inv2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_K_str_inv2/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_Axes1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/Axes1/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_Axes1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/Axes1/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_HBM_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_HBM/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_D_val_int_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_val_int/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_D_val_int_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_val_int/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_TCC_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_TCC/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_TCC_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_TCC/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_SQC_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_SQC/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_SQC_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_SQC/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_TD_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_TD/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_K_int_inv1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_K_int_inv1/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_K_int_inv1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_K_int_inv1/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_Axes1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_Axes1/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_Axes1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_Axes1/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_SPI_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/SPI/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_SPI_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/SPI/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_D_str_inv3_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_D_str_inv3/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_D_val_int_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_val_int/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_D_val_int_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_val_int/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_K_str_inv2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/K_str_inv2/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_K_str_inv2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/K_str_inv2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_CPC_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/CPC/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_CPC_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/CPC/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_dispatches_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_dispatches/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_mixbench2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_mixbench2/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_Axes4_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/Axes4/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_Axes4_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/Axes4/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_TCP_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_TCP/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_TCP_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_TCP/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_LDS_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_LDS/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_invdev_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/invdev/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_invdev_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/invdev/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_dev0_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_dev0/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_dev0_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_dev0/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_Axes1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_Axes1/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_invdev_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_invdev/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_D_str_inv2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_D_str_inv2/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_K_str_inv3_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_K_str_inv3/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_K_str_inv3_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_K_str_inv3/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_K_str_inv2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_K_str_inv2/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_K_str_inv2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_K_str_inv2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_D_str_inv2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_str_inv2/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_D_str_inv2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_str_inv2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_kernels_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/kernels/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_kernels_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/kernels/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_Axes4_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_Axes4/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_Axes4_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_Axes4/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_CMD_INV_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/CMD_INV/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_K_int_inv1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/K_int_inv1/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_K_int_inv1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/K_int_inv1/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_mixbench2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/mixbench2/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_mixbench2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/mixbench2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_Double_N_flag_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_Double_N_flag/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_TD_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_TD/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_TD_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_TD/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_TCC_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/TCC/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_TCC_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/TCC/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_dev0_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_dev0/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_D_str_inv2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_str_inv2/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_D_str_inv2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_str_inv2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_L2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_L2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_TA_CPC_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_TA_CPC/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_TA_CPC_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_TA_CPC/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_Double_N_flag_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/no_roof_Double_N_flag/mi100",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_Double_N_flag_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/no_roof_Double_N_flag/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_Double_N_flag_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/Double_N_flag/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_Double_N_flag_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/Double_N_flag/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_K_int_inv1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_K_int_inv1/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_K_str_valid_1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/no_roof_K_str_valid_1/mi100",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_K_str_valid_1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/no_roof_K_str_valid_1/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_mixbench1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_mixbench1/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_dev1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/dev1/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_dev1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/dev1/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_K_str_inv1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_K_str_inv1/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_K_str_inv1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_K_str_inv1/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_K_str_valid_1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/K_str_valid_1/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_K_str_valid_1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/K_str_valid_1/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_mixbench1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/mixbench1/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_mixbench1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/mixbench1/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_CMD_INV_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_CMD_INV/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_D_str_inv1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/roof_only_D_str_inv1/mi200",
            ],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_HBM_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_HBM/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_HBM_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_HBM/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_kernels_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_kernels/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_D_int_inv1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_int_inv1/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_D_int_inv1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_int_inv1/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_K_str_valid_2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/K_str_valid_2/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_K_str_valid_2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/K_str_valid_2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_TCP_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/TCP/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_TCP_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/TCP/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_Axes3_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/Axes3/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_Axes3_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/Axes3/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_invdev_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_invdev/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_invdev_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_invdev/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_dispatches_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_dispatches/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_dispatches_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_dispatches/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_D_str_inv3_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_str_inv3/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_D_str_inv3_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/D_str_inv3/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_TA_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/TA/mi100"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_TA_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/TA/mi200"]
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_D_int_inv1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_int_inv1/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_D_int_inv1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_D_int_inv1/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_dispatches_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/dispatches/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_dispatches_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/dispatches/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_roof_only_dev1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/roof_only_dev1/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_analyze_no_roof_K_int_inv2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_K_int_inv2/mi100"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0


def test_no_roof_K_int_inv2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/no_roof_K_int_inv2/mi200"],
        ):
            omniperf.main()
    assert e.type == SystemExit
    assert e.value.code == 0
