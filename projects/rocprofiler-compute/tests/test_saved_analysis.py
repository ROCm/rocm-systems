import sys
import pandas as pd
import glob


def compare(prev, cur):
    prev_csvs = glob.glob(prev + "/*")
    cur_csvs = glob.glob(cur + "/*")
    for prev_csv in prev_csvs:
        csv_name = prev_csv[prev_csv.rfind("/") + 1 :]
        with open(prev_csv, "r") as csv1, open(
            cur + "/" + csv_name, "r"
        ) as csv2:  # Import CSV files
            import1 = csv1.readlines()
            import2 = csv2.readlines()
            for row in import2:
                if row not in import1:
                    sys.exit(1)


##################################################
##          Generated tests                     ##
##  Meant to run after test_analyze_worloads    ##
##################################################


def test_saved_D_str_inv1_mi100():
    compare(
        "cmake/workloads/D_str_inv1/mi100/prev_analysis",
        "cmake/workloads/D_str_inv1/mi100/saved_analysis",
    )


def test_saved_D_str_inv1_mi200():
    compare(
        "cmake/workloads/D_str_inv1/mi200/prev_analysis",
        "cmake/workloads/D_str_inv1/mi200/saved_analysis",
    )


def test_saved_dev01p3_mi100():
    compare(
        "cmake/workloads/dev01p3/mi100/prev_analysis",
        "cmake/workloads/dev01p3/mi100/saved_analysis",
    )


def test_saved_SQC_mi100():
    compare(
        "cmake/workloads/SQC/mi100/prev_analysis",
        "cmake/workloads/SQC/mi100/saved_analysis",
    )


def test_saved_SQC_mi200():
    compare(
        "cmake/workloads/SQC/mi200/prev_analysis",
        "cmake/workloads/SQC/mi200/saved_analysis",
    )


def test_saved_Axes2_mi100():
    compare(
        "cmake/workloads/Axes2/mi100/prev_analysis",
        "cmake/workloads/Axes2/mi100/saved_analysis",
    )


def test_saved_Axes2_mi200():
    compare(
        "cmake/workloads/Axes2/mi200/prev_analysis",
        "cmake/workloads/Axes2/mi200/saved_analysis",
    )


def test_saved_no_roof_SQ_mi100():
    compare(
        "cmake/workloads/no_roof_SQ/mi100/prev_analysis",
        "cmake/workloads/no_roof_SQ/mi100/saved_analysis",
    )


def test_saved_no_roof_SQ_mi200():
    compare(
        "cmake/workloads/no_roof_SQ/mi200/prev_analysis",
        "cmake/workloads/no_roof_SQ/mi200/saved_analysis",
    )


def test_saved_CPF_mi100():
    compare(
        "cmake/workloads/CPF/mi100/prev_analysis",
        "cmake/workloads/CPF/mi100/saved_analysis",
    )


def test_saved_CPF_mi200():
    compare(
        "cmake/workloads/CPF/mi200/prev_analysis",
        "cmake/workloads/CPF/mi200/saved_analysis",
    )


def test_saved_no_roof_LDS_mi100():
    compare(
        "cmake/workloads/no_roof_LDS/mi100/prev_analysis",
        "cmake/workloads/no_roof_LDS/mi100/saved_analysis",
    )


def test_saved_no_roof_LDS_mi200():
    compare(
        "cmake/workloads/no_roof_LDS/mi200/prev_analysis",
        "cmake/workloads/no_roof_LDS/mi200/saved_analysis",
    )


def test_saved_D_str_inv4_mi100():
    compare(
        "cmake/workloads/D_str_inv4/mi100/prev_analysis",
        "cmake/workloads/D_str_inv4/mi100/saved_analysis",
    )


def test_saved_D_str_inv4_mi200():
    compare(
        "cmake/workloads/D_str_inv4/mi200/prev_analysis",
        "cmake/workloads/D_str_inv4/mi200/saved_analysis",
    )


def test_saved_roof_only_K_int_inv2_mi200():
    compare(
        "cmake/workloads/roof_only_K_int_inv2/mi200/prev_analysis",
        "cmake/workloads/roof_only_K_int_inv2/mi200/saved_analysis",
    )


def test_saved_no_roof_SPI_mi100():
    compare(
        "cmake/workloads/no_roof_SPI/mi100/prev_analysis",
        "cmake/workloads/no_roof_SPI/mi100/saved_analysis",
    )


def test_saved_no_roof_SPI_mi200():
    compare(
        "cmake/workloads/no_roof_SPI/mi200/prev_analysis",
        "cmake/workloads/no_roof_SPI/mi200/saved_analysis",
    )


def test_saved_no_roof_K_str_valid_2_mi100():
    compare(
        "cmake/workloads/no_roof_K_str_valid_2/mi100/prev_analysis",
        "cmake/workloads/no_roof_K_str_valid_2/mi100/saved_analysis",
    )


def test_saved_no_roof_K_str_valid_2_mi200():
    compare(
        "cmake/workloads/no_roof_K_str_valid_2/mi200/prev_analysis",
        "cmake/workloads/no_roof_K_str_valid_2/mi200/saved_analysis",
    )


def test_saved_no_roof_mixbench1_mi100():
    compare(
        "cmake/workloads/no_roof_mixbench1/mi100/prev_analysis",
        "cmake/workloads/no_roof_mixbench1/mi100/saved_analysis",
    )


def test_saved_no_roof_mixbench1_mi200():
    compare(
        "cmake/workloads/no_roof_mixbench1/mi200/prev_analysis",
        "cmake/workloads/no_roof_mixbench1/mi200/saved_analysis",
    )


def test_saved_no_roof_TA_mi100():
    compare(
        "cmake/workloads/no_roof_TA/mi100/prev_analysis",
        "cmake/workloads/no_roof_TA/mi100/saved_analysis",
    )


def test_saved_no_roof_TA_mi200():
    compare(
        "cmake/workloads/no_roof_TA/mi200/prev_analysis",
        "cmake/workloads/no_roof_TA/mi200/saved_analysis",
    )


def test_saved_no_roof_CPF_mi100():
    compare(
        "cmake/workloads/no_roof_CPF/mi100/prev_analysis",
        "cmake/workloads/no_roof_CPF/mi100/saved_analysis",
    )


def test_saved_no_roof_CPF_mi200():
    compare(
        "cmake/workloads/no_roof_CPF/mi200/prev_analysis",
        "cmake/workloads/no_roof_CPF/mi200/saved_analysis",
    )


def test_saved_no_roof_CPC_mi100():
    compare(
        "cmake/workloads/no_roof_CPC/mi100/prev_analysis",
        "cmake/workloads/no_roof_CPC/mi100/saved_analysis",
    )


def test_saved_no_roof_CPC_mi200():
    compare(
        "cmake/workloads/no_roof_CPC/mi200/prev_analysis",
        "cmake/workloads/no_roof_CPC/mi200/saved_analysis",
    )


def test_saved_K_str_inv3_mi100():
    compare(
        "cmake/workloads/K_str_inv3/mi100/prev_analysis",
        "cmake/workloads/K_str_inv3/mi100/saved_analysis",
    )


def test_saved_K_str_inv3_mi200():
    compare(
        "cmake/workloads/K_str_inv3/mi200/prev_analysis",
        "cmake/workloads/K_str_inv3/mi200/saved_analysis",
    )


def test_saved_LDS_mi100():
    compare(
        "cmake/workloads/LDS/mi100/prev_analysis",
        "cmake/workloads/LDS/mi100/saved_analysis",
    )


def test_saved_LDS_mi200():
    compare(
        "cmake/workloads/LDS/mi200/prev_analysis",
        "cmake/workloads/LDS/mi200/saved_analysis",
    )


def test_saved_no_roof_K_str_valid_3_mi100():
    compare(
        "cmake/workloads/no_roof_K_str_valid_3/mi100/prev_analysis",
        "cmake/workloads/no_roof_K_str_valid_3/mi100/saved_analysis",
    )


def test_saved_roof_only_D_int_inv2_mi200():
    compare(
        "cmake/workloads/roof_only_D_int_inv2/mi200/prev_analysis",
        "cmake/workloads/roof_only_D_int_inv2/mi200/saved_analysis",
    )


def test_saved_roof_only_K_str_inv1_mi200():
    compare(
        "cmake/workloads/roof_only_K_str_inv1/mi200/prev_analysis",
        "cmake/workloads/roof_only_K_str_inv1/mi200/saved_analysis",
    )


def test_saved_roof_only_SQC_mi200():
    compare(
        "cmake/workloads/roof_only_SQC/mi200/prev_analysis",
        "cmake/workloads/roof_only_SQC/mi200/saved_analysis",
    )


def test_saved_no_roof_Axes2_mi100():
    compare(
        "cmake/workloads/no_roof_Axes2/mi100/prev_analysis",
        "cmake/workloads/no_roof_Axes2/mi100/saved_analysis",
    )


def test_saved_no_roof_Axes2_mi200():
    compare(
        "cmake/workloads/no_roof_Axes2/mi200/prev_analysis",
        "cmake/workloads/no_roof_Axes2/mi200/saved_analysis",
    )


def test_saved_HBM_mi100():
    compare(
        "cmake/workloads/HBM/mi100/prev_analysis",
        "cmake/workloads/HBM/mi100/saved_analysis",
    )


def test_saved_HBM_mi200():
    compare(
        "cmake/workloads/HBM/mi200/prev_analysis",
        "cmake/workloads/HBM/mi200/saved_analysis",
    )


def test_saved_roof_only_TA_CPC_mi200():
    compare(
        "cmake/workloads/roof_only_TA_CPC/mi200/prev_analysis",
        "cmake/workloads/roof_only_TA_CPC/mi200/saved_analysis",
    )


def test_saved_roof_only_D_val_int_mi200():
    compare(
        "cmake/workloads/roof_only_D_val_int/mi200/prev_analysis",
        "cmake/workloads/roof_only_D_val_int/mi200/saved_analysis",
    )


def test_saved_no_roof_L2_mi100():
    compare(
        "cmake/workloads/no_roof_L2/mi100/prev_analysis",
        "cmake/workloads/no_roof_L2/mi100/saved_analysis",
    )


def test_saved_no_roof_L2_mi200():
    compare(
        "cmake/workloads/no_roof_L2/mi200/prev_analysis",
        "cmake/workloads/no_roof_L2/mi200/saved_analysis",
    )


def test_saved_L2_mi100():
    compare(
        "cmake/workloads/L2/mi100/prev_analysis",
        "cmake/workloads/L2/mi100/saved_analysis",
    )


def test_saved_L2_mi200():
    compare(
        "cmake/workloads/L2/mi200/prev_analysis",
        "cmake/workloads/L2/mi200/saved_analysis",
    )


def test_saved_no_roof_dev1_mi100():
    compare(
        "cmake/workloads/no_roof_dev1/mi100/prev_analysis",
        "cmake/workloads/no_roof_dev1/mi100/saved_analysis",
    )


def test_saved_no_roof_dev1_mi200():
    compare(
        "cmake/workloads/no_roof_dev1/mi200/prev_analysis",
        "cmake/workloads/no_roof_dev1/mi200/saved_analysis",
    )


def test_saved_roof_only_K_str_inv3_mi200():
    compare(
        "cmake/workloads/roof_only_K_str_inv3/mi200/prev_analysis",
        "cmake/workloads/roof_only_K_str_inv3/mi200/saved_analysis",
    )


def test_saved_roof_only_K_str_valid_1_mi200():
    compare(
        "cmake/workloads/roof_only_K_str_valid_1/mi200/prev_analysis",
        "cmake/workloads/roof_only_K_str_valid_1/mi200/saved_analysis",
    )


def test_saved_roof_only_CPC_mi200():
    compare(
        "cmake/workloads/roof_only_CPC/mi200/prev_analysis",
        "cmake/workloads/roof_only_CPC/mi200/saved_analysis",
    )


def test_saved_no_roof_Axes3_mi100():
    compare(
        "cmake/workloads/no_roof_Axes3/mi100/prev_analysis",
        "cmake/workloads/no_roof_Axes3/mi100/saved_analysis",
    )


def test_saved_no_roof_Axes3_mi200():
    compare(
        "cmake/workloads/no_roof_Axes3/mi200/prev_analysis",
        "cmake/workloads/no_roof_Axes3/mi200/saved_analysis",
    )


def test_saved_no_roof_D_str_inv3_mi100():
    compare(
        "cmake/workloads/no_roof_D_str_inv3/mi100/prev_analysis",
        "cmake/workloads/no_roof_D_str_inv3/mi100/saved_analysis",
    )


def test_saved_no_roof_D_str_inv3_mi200():
    compare(
        "cmake/workloads/no_roof_D_str_inv3/mi200/prev_analysis",
        "cmake/workloads/no_roof_D_str_inv3/mi200/saved_analysis",
    )


def test_saved_no_roof_D_int_inv2_mi100():
    compare(
        "cmake/workloads/no_roof_D_int_inv2/mi100/prev_analysis",
        "cmake/workloads/no_roof_D_int_inv2/mi100/saved_analysis",
    )


def test_saved_no_roof_D_int_inv2_mi200():
    compare(
        "cmake/workloads/no_roof_D_int_inv2/mi200/prev_analysis",
        "cmake/workloads/no_roof_D_int_inv2/mi200/saved_analysis",
    )


def test_saved_TD_mi100():
    compare(
        "cmake/workloads/TD/mi100/prev_analysis",
        "cmake/workloads/TD/mi100/saved_analysis",
    )


def test_saved_TD_mi200():
    compare(
        "cmake/workloads/TD/mi200/prev_analysis",
        "cmake/workloads/TD/mi200/saved_analysis",
    )


def test_saved_roof_only_D_int_inv1_mi200():
    compare(
        "cmake/workloads/roof_only_D_int_inv1/mi200/prev_analysis",
        "cmake/workloads/roof_only_D_int_inv1/mi200/saved_analysis",
    )


def test_saved_D_val_int2_mi100():
    compare(
        "cmake/workloads/D_val_int2/mi100/prev_analysis",
        "cmake/workloads/D_val_int2/mi100/saved_analysis",
    )


def test_saved_D_val_int2_mi200():
    compare(
        "cmake/workloads/D_val_int2/mi200/prev_analysis",
        "cmake/workloads/D_val_int2/mi200/saved_analysis",
    )


def test_saved_no_roof_mixbench2_mi100():
    compare(
        "cmake/workloads/no_roof_mixbench2/mi100/prev_analysis",
        "cmake/workloads/no_roof_mixbench2/mi100/saved_analysis",
    )


def test_saved_no_roof_mixbench2_mi200():
    compare(
        "cmake/workloads/no_roof_mixbench2/mi200/prev_analysis",
        "cmake/workloads/no_roof_mixbench2/mi200/saved_analysis",
    )


def test_saved_roof_only_SPI_mi200():
    compare(
        "cmake/workloads/roof_only_SPI/mi200/prev_analysis",
        "cmake/workloads/roof_only_SPI/mi200/saved_analysis",
    )


def test_saved_no_roof_D_val_int2_mi100():
    compare(
        "cmake/workloads/no_roof_D_val_int2/mi100/prev_analysis",
        "cmake/workloads/no_roof_D_val_int2/mi100/saved_analysis",
    )


def test_saved_no_roof_D_val_int2_mi200():
    compare(
        "cmake/workloads/no_roof_D_val_int2/mi200/prev_analysis",
        "cmake/workloads/no_roof_D_val_int2/mi200/saved_analysis",
    )


def test_saved_K_str_inv1_mi100():
    compare(
        "cmake/workloads/K_str_inv1/mi100/prev_analysis",
        "cmake/workloads/K_str_inv1/mi100/saved_analysis",
    )


def test_saved_K_str_inv1_mi200():
    compare(
        "cmake/workloads/K_str_inv1/mi200/prev_analysis",
        "cmake/workloads/K_str_inv1/mi200/saved_analysis",
    )


def test_saved_roof_only_TA_mi200():
    compare(
        "cmake/workloads/roof_only_TA/mi200/prev_analysis",
        "cmake/workloads/roof_only_TA/mi200/saved_analysis",
    )


def test_saved_K_str_valid_3_mi100():
    compare(
        "cmake/workloads/K_str_valid_3/mi100/prev_analysis",
        "cmake/workloads/K_str_valid_3/mi100/saved_analysis",
    )


def test_saved_SQ_mi100():
    compare(
        "cmake/workloads/SQ/mi100/prev_analysis",
        "cmake/workloads/SQ/mi100/saved_analysis",
    )


def test_saved_SQ_mi200():
    compare(
        "cmake/workloads/SQ/mi200/prev_analysis",
        "cmake/workloads/SQ/mi200/saved_analysis",
    )


def test_saved_no_roof_D_str_inv1_mi100():
    compare(
        "cmake/workloads/no_roof_D_str_inv1/mi100/prev_analysis",
        "cmake/workloads/no_roof_D_str_inv1/mi100/saved_analysis",
    )


def test_saved_no_roof_D_str_inv1_mi200():
    compare(
        "cmake/workloads/no_roof_D_str_inv1/mi200/prev_analysis",
        "cmake/workloads/no_roof_D_str_inv1/mi200/saved_analysis",
    )


def test_saved_no_roof_dev01p3_mi100():
    compare(
        "cmake/workloads/no_roof_dev01p3/mi100/prev_analysis",
        "cmake/workloads/no_roof_dev01p3/mi100/saved_analysis",
    )


def test_saved_roof_only_D_val_int2_mi200():
    compare(
        "cmake/workloads/roof_only_D_val_int2/mi200/prev_analysis",
        "cmake/workloads/roof_only_D_val_int2/mi200/saved_analysis",
    )


def test_saved_no_roof_D_str_inv4_mi100():
    compare(
        "cmake/workloads/no_roof_D_str_inv4/mi100/prev_analysis",
        "cmake/workloads/no_roof_D_str_inv4/mi100/saved_analysis",
    )


def test_saved_no_roof_D_str_inv4_mi200():
    compare(
        "cmake/workloads/no_roof_D_str_inv4/mi200/prev_analysis",
        "cmake/workloads/no_roof_D_str_inv4/mi200/saved_analysis",
    )


def test_saved_roof_only_CPF_mi200():
    compare(
        "cmake/workloads/roof_only_CPF/mi200/prev_analysis",
        "cmake/workloads/roof_only_CPF/mi200/saved_analysis",
    )


def test_saved_mixbench_mi100():
    compare(
        "cmake/workloads/mixbench/mi100/prev_analysis",
        "cmake/workloads/mixbench/mi100/saved_analysis",
    )


def test_saved_mixbench_mi200():
    compare(
        "cmake/workloads/mixbench/mi200/prev_analysis",
        "cmake/workloads/mixbench/mi200/saved_analysis",
    )


def test_saved_roof_only_D_str_inv4_mi200():
    compare(
        "cmake/workloads/roof_only_D_str_inv4/mi200/prev_analysis",
        "cmake/workloads/roof_only_D_str_inv4/mi200/saved_analysis",
    )


def test_saved_no_roof_kernels_mi100():
    compare(
        "cmake/workloads/no_roof_kernels/mi100/prev_analysis",
        "cmake/workloads/no_roof_kernels/mi100/saved_analysis",
    )


def test_saved_no_roof_kernels_mi200():
    compare(
        "cmake/workloads/no_roof_kernels/mi200/prev_analysis",
        "cmake/workloads/no_roof_kernels/mi200/saved_analysis",
    )


def test_saved_roof_only_TCC_mi200():
    compare(
        "cmake/workloads/roof_only_TCC/mi200/prev_analysis",
        "cmake/workloads/roof_only_TCC/mi200/saved_analysis",
    )


def test_saved_TA_CPC_mi100():
    compare(
        "cmake/workloads/TA_CPC/mi100/prev_analysis",
        "cmake/workloads/TA_CPC/mi100/saved_analysis",
    )


def test_saved_TA_CPC_mi200():
    compare(
        "cmake/workloads/TA_CPC/mi200/prev_analysis",
        "cmake/workloads/TA_CPC/mi200/saved_analysis",
    )


def test_saved_roof_only_SQ_mi200():
    compare(
        "cmake/workloads/roof_only_SQ/mi200/prev_analysis",
        "cmake/workloads/roof_only_SQ/mi200/saved_analysis",
    )


def test_saved_K_int_inv2_mi100():
    compare(
        "cmake/workloads/K_int_inv2/mi100/prev_analysis",
        "cmake/workloads/K_int_inv2/mi100/saved_analysis",
    )


def test_saved_K_int_inv2_mi200():
    compare(
        "cmake/workloads/K_int_inv2/mi200/prev_analysis",
        "cmake/workloads/K_int_inv2/mi200/saved_analysis",
    )


def test_saved_roof_only_TCP_mi200():
    compare(
        "cmake/workloads/roof_only_TCP/mi200/prev_analysis",
        "cmake/workloads/roof_only_TCP/mi200/saved_analysis",
    )


def test_saved_roof_only_K_str_valid_2_mi200():
    compare(
        "cmake/workloads/roof_only_K_str_valid_2/mi200/prev_analysis",
        "cmake/workloads/roof_only_K_str_valid_2/mi200/saved_analysis",
    )


def test_saved_D_int_inv2_mi100():
    compare(
        "cmake/workloads/D_int_inv2/mi100/prev_analysis",
        "cmake/workloads/D_int_inv2/mi100/saved_analysis",
    )


def test_saved_D_int_inv2_mi200():
    compare(
        "cmake/workloads/D_int_inv2/mi200/prev_analysis",
        "cmake/workloads/D_int_inv2/mi200/saved_analysis",
    )


def test_saved_roof_only_Axes3_mi200():
    compare(
        "cmake/workloads/roof_only_Axes3/mi200/prev_analysis",
        "cmake/workloads/roof_only_Axes3/mi200/saved_analysis",
    )


def test_saved_dev0_mi100():
    compare(
        "cmake/workloads/dev0/mi100/prev_analysis",
        "cmake/workloads/dev0/mi100/saved_analysis",
    )


def test_saved_dev0_mi200():
    compare(
        "cmake/workloads/dev0/mi200/prev_analysis",
        "cmake/workloads/dev0/mi200/saved_analysis",
    )


def test_saved_roof_only_K_str_inv2_mi200():
    compare(
        "cmake/workloads/roof_only_K_str_inv2/mi200/prev_analysis",
        "cmake/workloads/roof_only_K_str_inv2/mi200/saved_analysis",
    )


def test_saved_Axes1_mi100():
    compare(
        "cmake/workloads/Axes1/mi100/prev_analysis",
        "cmake/workloads/Axes1/mi100/saved_analysis",
    )


def test_saved_Axes1_mi200():
    compare(
        "cmake/workloads/Axes1/mi200/prev_analysis",
        "cmake/workloads/Axes1/mi200/saved_analysis",
    )


def test_saved_roof_only_HBM_mi200():
    compare(
        "cmake/workloads/roof_only_HBM/mi200/prev_analysis",
        "cmake/workloads/roof_only_HBM/mi200/saved_analysis",
    )


def test_saved_D_val_int_mi100():
    compare(
        "cmake/workloads/D_val_int/mi100/prev_analysis",
        "cmake/workloads/D_val_int/mi100/saved_analysis",
    )


def test_saved_D_val_int_mi200():
    compare(
        "cmake/workloads/D_val_int/mi200/prev_analysis",
        "cmake/workloads/D_val_int/mi200/saved_analysis",
    )


def test_saved_no_roof_TCC_mi100():
    compare(
        "cmake/workloads/no_roof_TCC/mi100/prev_analysis",
        "cmake/workloads/no_roof_TCC/mi100/saved_analysis",
    )


def test_saved_no_roof_TCC_mi200():
    compare(
        "cmake/workloads/no_roof_TCC/mi200/prev_analysis",
        "cmake/workloads/no_roof_TCC/mi200/saved_analysis",
    )


def test_saved_no_roof_SQC_mi100():
    compare(
        "cmake/workloads/no_roof_SQC/mi100/prev_analysis",
        "cmake/workloads/no_roof_SQC/mi100/saved_analysis",
    )


def test_saved_no_roof_SQC_mi200():
    compare(
        "cmake/workloads/no_roof_SQC/mi200/prev_analysis",
        "cmake/workloads/no_roof_SQC/mi200/saved_analysis",
    )


def test_saved_roof_only_TD_mi200():
    compare(
        "cmake/workloads/roof_only_TD/mi200/prev_analysis",
        "cmake/workloads/roof_only_TD/mi200/saved_analysis",
    )


def test_saved_no_roof_K_int_inv1_mi100():
    compare(
        "cmake/workloads/no_roof_K_int_inv1/mi100/prev_analysis",
        "cmake/workloads/no_roof_K_int_inv1/mi100/saved_analysis",
    )


def test_saved_no_roof_K_int_inv1_mi200():
    compare(
        "cmake/workloads/no_roof_K_int_inv1/mi200/prev_analysis",
        "cmake/workloads/no_roof_K_int_inv1/mi200/saved_analysis",
    )


def test_saved_no_roof_Axes1_mi100():
    compare(
        "cmake/workloads/no_roof_Axes1/mi100/prev_analysis",
        "cmake/workloads/no_roof_Axes1/mi100/saved_analysis",
    )


def test_saved_no_roof_Axes1_mi200():
    compare(
        "cmake/workloads/no_roof_Axes1/mi200/prev_analysis",
        "cmake/workloads/no_roof_Axes1/mi200/saved_analysis",
    )


def test_saved_SPI_mi100():
    compare(
        "cmake/workloads/SPI/mi100/prev_analysis",
        "cmake/workloads/SPI/mi100/saved_analysis",
    )


def test_saved_SPI_mi200():
    compare(
        "cmake/workloads/SPI/mi200/prev_analysis",
        "cmake/workloads/SPI/mi200/saved_analysis",
    )


def test_saved_roof_only_D_str_inv3_mi200():
    compare(
        "cmake/workloads/roof_only_D_str_inv3/mi200/prev_analysis",
        "cmake/workloads/roof_only_D_str_inv3/mi200/saved_analysis",
    )


def test_saved_no_roof_D_val_int_mi100():
    compare(
        "cmake/workloads/no_roof_D_val_int/mi100/prev_analysis",
        "cmake/workloads/no_roof_D_val_int/mi100/saved_analysis",
    )


def test_saved_no_roof_D_val_int_mi200():
    compare(
        "cmake/workloads/no_roof_D_val_int/mi200/prev_analysis",
        "cmake/workloads/no_roof_D_val_int/mi200/saved_analysis",
    )


def test_saved_K_str_inv2_mi100():
    compare(
        "cmake/workloads/K_str_inv2/mi100/prev_analysis",
        "cmake/workloads/K_str_inv2/mi100/saved_analysis",
    )


def test_saved_K_str_inv2_mi200():
    compare(
        "cmake/workloads/K_str_inv2/mi200/prev_analysis",
        "cmake/workloads/K_str_inv2/mi200/saved_analysis",
    )


def test_saved_CPC_mi100():
    compare(
        "cmake/workloads/CPC/mi100/prev_analysis",
        "cmake/workloads/CPC/mi100/saved_analysis",
    )


def test_saved_CPC_mi200():
    compare(
        "cmake/workloads/CPC/mi200/prev_analysis",
        "cmake/workloads/CPC/mi200/saved_analysis",
    )


def test_saved_roof_only_dispatches_mi200():
    compare(
        "cmake/workloads/roof_only_dispatches/mi200/prev_analysis",
        "cmake/workloads/roof_only_dispatches/mi200/saved_analysis",
    )


def test_saved_roof_only_mixbench2_mi200():
    compare(
        "cmake/workloads/roof_only_mixbench2/mi200/prev_analysis",
        "cmake/workloads/roof_only_mixbench2/mi200/saved_analysis",
    )


def test_saved_Axes4_mi100():
    compare(
        "cmake/workloads/Axes4/mi100/prev_analysis",
        "cmake/workloads/Axes4/mi100/saved_analysis",
    )


def test_saved_Axes4_mi200():
    compare(
        "cmake/workloads/Axes4/mi200/prev_analysis",
        "cmake/workloads/Axes4/mi200/saved_analysis",
    )


def test_saved_no_roof_TCP_mi100():
    compare(
        "cmake/workloads/no_roof_TCP/mi100/prev_analysis",
        "cmake/workloads/no_roof_TCP/mi100/saved_analysis",
    )


def test_saved_no_roof_TCP_mi200():
    compare(
        "cmake/workloads/no_roof_TCP/mi200/prev_analysis",
        "cmake/workloads/no_roof_TCP/mi200/saved_analysis",
    )


def test_saved_roof_only_LDS_mi200():
    compare(
        "cmake/workloads/roof_only_LDS/mi200/prev_analysis",
        "cmake/workloads/roof_only_LDS/mi200/saved_analysis",
    )


def test_saved_invdev_mi100():
    compare(
        "cmake/workloads/invdev/mi100/prev_analysis",
        "cmake/workloads/invdev/mi100/saved_analysis",
    )


def test_saved_invdev_mi200():
    compare(
        "cmake/workloads/invdev/mi200/prev_analysis",
        "cmake/workloads/invdev/mi200/saved_analysis",
    )


def test_saved_no_roof_dev0_mi100():
    compare(
        "cmake/workloads/no_roof_dev0/mi100/prev_analysis",
        "cmake/workloads/no_roof_dev0/mi100/saved_analysis",
    )


def test_saved_no_roof_dev0_mi200():
    compare(
        "cmake/workloads/no_roof_dev0/mi200/prev_analysis",
        "cmake/workloads/no_roof_dev0/mi200/saved_analysis",
    )


def test_saved_roof_only_Axes1_mi200():
    compare(
        "cmake/workloads/roof_only_Axes1/mi200/prev_analysis",
        "cmake/workloads/roof_only_Axes1/mi200/saved_analysis",
    )


def test_saved_roof_only_invdev_mi200():
    compare(
        "cmake/workloads/roof_only_invdev/mi200/prev_analysis",
        "cmake/workloads/roof_only_invdev/mi200/saved_analysis",
    )


def test_saved_roof_only_D_str_inv2_mi200():
    compare(
        "cmake/workloads/roof_only_D_str_inv2/mi200/prev_analysis",
        "cmake/workloads/roof_only_D_str_inv2/mi200/saved_analysis",
    )


def test_saved_no_roof_K_str_inv3_mi100():
    compare(
        "cmake/workloads/no_roof_K_str_inv3/mi100/prev_analysis",
        "cmake/workloads/no_roof_K_str_inv3/mi100/saved_analysis",
    )


def test_saved_no_roof_K_str_inv3_mi200():
    compare(
        "cmake/workloads/no_roof_K_str_inv3/mi200/prev_analysis",
        "cmake/workloads/no_roof_K_str_inv3/mi200/saved_analysis",
    )


def test_saved_no_roof_K_str_inv2_mi100():
    compare(
        "cmake/workloads/no_roof_K_str_inv2/mi100/prev_analysis",
        "cmake/workloads/no_roof_K_str_inv2/mi100/saved_analysis",
    )


def test_saved_no_roof_K_str_inv2_mi200():
    compare(
        "cmake/workloads/no_roof_K_str_inv2/mi200/prev_analysis",
        "cmake/workloads/no_roof_K_str_inv2/mi200/saved_analysis",
    )


def test_saved_D_str_inv2_mi100():
    compare(
        "cmake/workloads/D_str_inv2/mi100/prev_analysis",
        "cmake/workloads/D_str_inv2/mi100/saved_analysis",
    )


def test_saved_D_str_inv2_mi200():
    compare(
        "cmake/workloads/D_str_inv2/mi200/prev_analysis",
        "cmake/workloads/D_str_inv2/mi200/saved_analysis",
    )


def test_saved_kernels_mi100():
    compare(
        "cmake/workloads/kernels/mi100/prev_analysis",
        "cmake/workloads/kernels/mi100/saved_analysis",
    )


def test_saved_kernels_mi200():
    compare(
        "cmake/workloads/kernels/mi200/prev_analysis",
        "cmake/workloads/kernels/mi200/saved_analysis",
    )


def test_saved_no_roof_Axes4_mi100():
    compare(
        "cmake/workloads/no_roof_Axes4/mi100/prev_analysis",
        "cmake/workloads/no_roof_Axes4/mi100/saved_analysis",
    )


def test_saved_no_roof_Axes4_mi200():
    compare(
        "cmake/workloads/no_roof_Axes4/mi200/prev_analysis",
        "cmake/workloads/no_roof_Axes4/mi200/saved_analysis",
    )


def test_saved_CMD_INV_mi100():
    compare(
        "cmake/workloads/CMD_INV/mi100/prev_analysis",
        "cmake/workloads/CMD_INV/mi100/saved_analysis",
    )


def test_saved_K_int_inv1_mi100():
    compare(
        "cmake/workloads/K_int_inv1/mi100/prev_analysis",
        "cmake/workloads/K_int_inv1/mi100/saved_analysis",
    )


def test_saved_K_int_inv1_mi200():
    compare(
        "cmake/workloads/K_int_inv1/mi200/prev_analysis",
        "cmake/workloads/K_int_inv1/mi200/saved_analysis",
    )


def test_saved_mixbench2_mi100():
    compare(
        "cmake/workloads/mixbench2/mi100/prev_analysis",
        "cmake/workloads/mixbench2/mi100/saved_analysis",
    )


def test_saved_mixbench2_mi200():
    compare(
        "cmake/workloads/mixbench2/mi200/prev_analysis",
        "cmake/workloads/mixbench2/mi200/saved_analysis",
    )


def test_saved_roof_only_Double_N_flag_mi200():
    compare(
        "cmake/workloads/roof_only_Double_N_flag/mi200/prev_analysis",
        "cmake/workloads/roof_only_Double_N_flag/mi200/saved_analysis",
    )


def test_saved_no_roof_TD_mi100():
    compare(
        "cmake/workloads/no_roof_TD/mi100/prev_analysis",
        "cmake/workloads/no_roof_TD/mi100/saved_analysis",
    )


def test_saved_no_roof_TD_mi200():
    compare(
        "cmake/workloads/no_roof_TD/mi200/prev_analysis",
        "cmake/workloads/no_roof_TD/mi200/saved_analysis",
    )


def test_saved_TCC_mi100():
    compare(
        "cmake/workloads/TCC/mi100/prev_analysis",
        "cmake/workloads/TCC/mi100/saved_analysis",
    )


def test_saved_TCC_mi200():
    compare(
        "cmake/workloads/TCC/mi200/prev_analysis",
        "cmake/workloads/TCC/mi200/saved_analysis",
    )


def test_saved_roof_only_dev0_mi200():
    compare(
        "cmake/workloads/roof_only_dev0/mi200/prev_analysis",
        "cmake/workloads/roof_only_dev0/mi200/saved_analysis",
    )


def test_saved_no_roof_D_str_inv2_mi100():
    compare(
        "cmake/workloads/no_roof_D_str_inv2/mi100/prev_analysis",
        "cmake/workloads/no_roof_D_str_inv2/mi100/saved_analysis",
    )


def test_saved_no_roof_D_str_inv2_mi200():
    compare(
        "cmake/workloads/no_roof_D_str_inv2/mi200/prev_analysis",
        "cmake/workloads/no_roof_D_str_inv2/mi200/saved_analysis",
    )


def test_saved_roof_only_L2_mi200():
    compare(
        "cmake/workloads/roof_only_L2/mi200/prev_analysis",
        "cmake/workloads/roof_only_L2/mi200/saved_analysis",
    )


def test_saved_no_roof_TA_CPC_mi100():
    compare(
        "cmake/workloads/no_roof_TA_CPC/mi100/prev_analysis",
        "cmake/workloads/no_roof_TA_CPC/mi100/saved_analysis",
    )


def test_saved_no_roof_TA_CPC_mi200():
    compare(
        "cmake/workloads/no_roof_TA_CPC/mi200/prev_analysis",
        "cmake/workloads/no_roof_TA_CPC/mi200/saved_analysis",
    )


def test_saved_no_roof_Double_N_flag_mi100():
    compare(
        "cmake/workloads/no_roof_Double_N_flag/mi100/prev_analysis",
        "cmake/workloads/no_roof_Double_N_flag/mi100/saved_analysis",
    )


def test_saved_no_roof_Double_N_flag_mi200():
    compare(
        "cmake/workloads/no_roof_Double_N_flag/mi200/prev_analysis",
        "cmake/workloads/no_roof_Double_N_flag/mi200/saved_analysis",
    )


def test_saved_Double_N_flag_mi100():
    compare(
        "cmake/workloads/Double_N_flag/mi100/prev_analysis",
        "cmake/workloads/Double_N_flag/mi100/saved_analysis",
    )


def test_saved_Double_N_flag_mi200():
    compare(
        "cmake/workloads/Double_N_flag/mi200/prev_analysis",
        "cmake/workloads/Double_N_flag/mi200/saved_analysis",
    )


def test_saved_roof_only_K_int_inv1_mi200():
    compare(
        "cmake/workloads/roof_only_K_int_inv1/mi200/prev_analysis",
        "cmake/workloads/roof_only_K_int_inv1/mi200/saved_analysis",
    )


def test_saved_no_roof_K_str_valid_1_mi100():
    compare(
        "cmake/workloads/no_roof_K_str_valid_1/mi100/prev_analysis",
        "cmake/workloads/no_roof_K_str_valid_1/mi100/saved_analysis",
    )


def test_saved_no_roof_K_str_valid_1_mi200():
    compare(
        "cmake/workloads/no_roof_K_str_valid_1/mi200/prev_analysis",
        "cmake/workloads/no_roof_K_str_valid_1/mi200/saved_analysis",
    )


def test_saved_roof_only_mixbench1_mi200():
    compare(
        "cmake/workloads/roof_only_mixbench1/mi200/prev_analysis",
        "cmake/workloads/roof_only_mixbench1/mi200/saved_analysis",
    )


def test_saved_dev1_mi100():
    compare(
        "cmake/workloads/dev1/mi100/prev_analysis",
        "cmake/workloads/dev1/mi100/saved_analysis",
    )


def test_saved_dev1_mi200():
    compare(
        "cmake/workloads/dev1/mi200/prev_analysis",
        "cmake/workloads/dev1/mi200/saved_analysis",
    )


def test_saved_no_roof_K_str_inv1_mi100():
    compare(
        "cmake/workloads/no_roof_K_str_inv1/mi100/prev_analysis",
        "cmake/workloads/no_roof_K_str_inv1/mi100/saved_analysis",
    )


def test_saved_no_roof_K_str_inv1_mi200():
    compare(
        "cmake/workloads/no_roof_K_str_inv1/mi200/prev_analysis",
        "cmake/workloads/no_roof_K_str_inv1/mi200/saved_analysis",
    )


def test_saved_K_str_valid_1_mi100():
    compare(
        "cmake/workloads/K_str_valid_1/mi100/prev_analysis",
        "cmake/workloads/K_str_valid_1/mi100/saved_analysis",
    )


def test_saved_K_str_valid_1_mi200():
    compare(
        "cmake/workloads/K_str_valid_1/mi200/prev_analysis",
        "cmake/workloads/K_str_valid_1/mi200/saved_analysis",
    )


def test_saved_mixbench1_mi100():
    compare(
        "cmake/workloads/mixbench1/mi100/prev_analysis",
        "cmake/workloads/mixbench1/mi100/saved_analysis",
    )


def test_saved_mixbench1_mi200():
    compare(
        "cmake/workloads/mixbench1/mi200/prev_analysis",
        "cmake/workloads/mixbench1/mi200/saved_analysis",
    )


def test_saved_no_roof_CMD_INV_mi100():
    compare(
        "cmake/workloads/no_roof_CMD_INV/mi100/prev_analysis",
        "cmake/workloads/no_roof_CMD_INV/mi100/saved_analysis",
    )


def test_saved_roof_only_D_str_inv1_mi200():
    compare(
        "cmake/workloads/roof_only_D_str_inv1/mi200/prev_analysis",
        "cmake/workloads/roof_only_D_str_inv1/mi200/saved_analysis",
    )


def test_saved_no_roof_HBM_mi100():
    compare(
        "cmake/workloads/no_roof_HBM/mi100/prev_analysis",
        "cmake/workloads/no_roof_HBM/mi100/saved_analysis",
    )


def test_saved_no_roof_HBM_mi200():
    compare(
        "cmake/workloads/no_roof_HBM/mi200/prev_analysis",
        "cmake/workloads/no_roof_HBM/mi200/saved_analysis",
    )


def test_saved_roof_only_kernels_mi200():
    compare(
        "cmake/workloads/roof_only_kernels/mi200/prev_analysis",
        "cmake/workloads/roof_only_kernels/mi200/saved_analysis",
    )


def test_saved_D_int_inv1_mi100():
    compare(
        "cmake/workloads/D_int_inv1/mi100/prev_analysis",
        "cmake/workloads/D_int_inv1/mi100/saved_analysis",
    )


def test_saved_D_int_inv1_mi200():
    compare(
        "cmake/workloads/D_int_inv1/mi200/prev_analysis",
        "cmake/workloads/D_int_inv1/mi200/saved_analysis",
    )


def test_saved_K_str_valid_2_mi100():
    compare(
        "cmake/workloads/K_str_valid_2/mi100/prev_analysis",
        "cmake/workloads/K_str_valid_2/mi100/saved_analysis",
    )


def test_saved_K_str_valid_2_mi200():
    compare(
        "cmake/workloads/K_str_valid_2/mi200/prev_analysis",
        "cmake/workloads/K_str_valid_2/mi200/saved_analysis",
    )


def test_saved_TCP_mi100():
    compare(
        "cmake/workloads/TCP/mi100/prev_analysis",
        "cmake/workloads/TCP/mi100/saved_analysis",
    )


def test_saved_TCP_mi200():
    compare(
        "cmake/workloads/TCP/mi200/prev_analysis",
        "cmake/workloads/TCP/mi200/saved_analysis",
    )


def test_saved_Axes3_mi100():
    compare(
        "cmake/workloads/Axes3/mi100/prev_analysis",
        "cmake/workloads/Axes3/mi100/saved_analysis",
    )


def test_saved_Axes3_mi200():
    compare(
        "cmake/workloads/Axes3/mi200/prev_analysis",
        "cmake/workloads/Axes3/mi200/saved_analysis",
    )


def test_saved_no_roof_invdev_mi100():
    compare(
        "cmake/workloads/no_roof_invdev/mi100/prev_analysis",
        "cmake/workloads/no_roof_invdev/mi100/saved_analysis",
    )


def test_saved_no_roof_invdev_mi200():
    compare(
        "cmake/workloads/no_roof_invdev/mi200/prev_analysis",
        "cmake/workloads/no_roof_invdev/mi200/saved_analysis",
    )


def test_saved_no_roof_dispatches_mi100():
    compare(
        "cmake/workloads/no_roof_dispatches/mi100/prev_analysis",
        "cmake/workloads/no_roof_dispatches/mi100/saved_analysis",
    )


def test_saved_no_roof_dispatches_mi200():
    compare(
        "cmake/workloads/no_roof_dispatches/mi200/prev_analysis",
        "cmake/workloads/no_roof_dispatches/mi200/saved_analysis",
    )


def test_saved_D_str_inv3_mi100():
    compare(
        "cmake/workloads/D_str_inv3/mi100/prev_analysis",
        "cmake/workloads/D_str_inv3/mi100/saved_analysis",
    )


def test_saved_D_str_inv3_mi200():
    compare(
        "cmake/workloads/D_str_inv3/mi200/prev_analysis",
        "cmake/workloads/D_str_inv3/mi200/saved_analysis",
    )


def test_saved_TA_mi100():
    compare(
        "cmake/workloads/TA/mi100/prev_analysis",
        "cmake/workloads/TA/mi100/saved_analysis",
    )


def test_saved_TA_mi200():
    compare(
        "cmake/workloads/TA/mi200/prev_analysis",
        "cmake/workloads/TA/mi200/saved_analysis",
    )


def test_saved_no_roof_D_int_inv1_mi100():
    compare(
        "cmake/workloads/no_roof_D_int_inv1/mi100/prev_analysis",
        "cmake/workloads/no_roof_D_int_inv1/mi100/saved_analysis",
    )


def test_saved_no_roof_D_int_inv1_mi200():
    compare(
        "cmake/workloads/no_roof_D_int_inv1/mi200/prev_analysis",
        "cmake/workloads/no_roof_D_int_inv1/mi200/saved_analysis",
    )


def test_saved_dispatches_mi100():
    compare(
        "cmake/workloads/dispatches/mi100/prev_analysis",
        "cmake/workloads/dispatches/mi100/saved_analysis",
    )


def test_saved_dispatches_mi200():
    compare(
        "cmake/workloads/dispatches/mi200/prev_analysis",
        "cmake/workloads/dispatches/mi200/saved_analysis",
    )


def test_saved_roof_only_dev1_mi200():
    compare(
        "cmake/workloads/roof_only_dev1/mi200/prev_analysis",
        "cmake/workloads/roof_only_dev1/mi200/saved_analysis",
    )


def test_saved_no_roof_K_int_inv2_mi100():
    compare(
        "cmake/workloads/no_roof_K_int_inv2/mi100/prev_analysis",
        "cmake/workloads/no_roof_K_int_inv2/mi100/saved_analysis",
    )


def test_saved_no_roof_K_int_inv2_mi200():
    compare(
        "cmake/workloads/no_roof_K_int_inv2/mi200/prev_analysis",
        "cmake/workloads/no_roof_K_int_inv2/mi200/saved_analysis",
    )
