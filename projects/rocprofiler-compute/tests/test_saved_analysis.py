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
##  Meant to run after test_analyze_workloads   ##
##################################################


def test_saved_dispatch_invalid_MI100():
    compare(
        "workloads/dispatch_invalid/MI100/prev_analysis",
        "workloads/dispatch_invalid/MI100/saved_analysis",
    )


def test_saved_kernel_verbose_4_MI100():
    compare(
        "workloads/kernel_verbose_4/MI100/prev_analysis",
        "workloads/kernel_verbose_4/MI100/saved_analysis",
    )


def test_saved_no_roof_MI100():
    compare(
        "workloads/no_roof/MI100/prev_analysis", "workloads/no_roof/MI100/saved_analysis"
    )


def test_saved_kernel_verbose_5_MI100():
    compare(
        "workloads/kernel_verbose_5/MI100/prev_analysis",
        "workloads/kernel_verbose_5/MI100/saved_analysis",
    )


def test_saved_ipblocks_CPC_MI100():
    compare(
        "workloads/ipblocks_CPC/MI100/prev_analysis",
        "workloads/ipblocks_CPC/MI100/saved_analysis",
    )


def test_saved_ipblocks_SQ_SQC_TCP_CPC_MI100():
    compare(
        "workloads/ipblocks_SQ_SQC_TCP_CPC/MI100/prev_analysis",
        "workloads/ipblocks_SQ_SQC_TCP_CPC/MI100/saved_analysis",
    )


def test_saved_kernel_verbose_2_MI100():
    compare(
        "workloads/kernel_verbose_2/MI100/prev_analysis",
        "workloads/kernel_verbose_2/MI100/saved_analysis",
    )


def test_saved_ipblocks_SQ_MI100():
    compare(
        "workloads/ipblocks_SQ/MI100/prev_analysis",
        "workloads/ipblocks_SQ/MI100/saved_analysis",
    )


def test_saved_kernel_substr_MI100():
    compare(
        "workloads/kernel_substr/MI100/prev_analysis",
        "workloads/kernel_substr/MI100/saved_analysis",
    )


def test_saved_ipblocks_CPF_MI100():
    compare(
        "workloads/ipblocks_CPF/MI100/prev_analysis",
        "workloads/ipblocks_CPF/MI100/saved_analysis",
    )


def test_saved_dispatch_0_1_MI100():
    compare(
        "workloads/dispatch_0_1/MI100/prev_analysis",
        "workloads/dispatch_0_1/MI100/saved_analysis",
    )


def test_saved_kernel_verbose_0_MI100():
    compare(
        "workloads/kernel_verbose_0/MI100/prev_analysis",
        "workloads/kernel_verbose_0/MI100/saved_analysis",
    )


def test_saved_ipblocks_SQC_MI100():
    compare(
        "workloads/ipblocks_SQC/MI100/prev_analysis",
        "workloads/ipblocks_SQC/MI100/saved_analysis",
    )


def test_saved_join_type_grid_MI100():
    compare(
        "workloads/join_type_grid/MI100/prev_analysis",
        "workloads/join_type_grid/MI100/saved_analysis",
    )


def test_saved_ipblocks_TCP_MI100():
    compare(
        "workloads/ipblocks_TCP/MI100/prev_analysis",
        "workloads/ipblocks_TCP/MI100/saved_analysis",
    )


def test_saved_kernel_verbose_3_MI100():
    compare(
        "workloads/kernel_verbose_3/MI100/prev_analysis",
        "workloads/kernel_verbose_3/MI100/saved_analysis",
    )


def test_saved_device_inv_int_MI100():
    compare(
        "workloads/device_inv_int/MI100/prev_analysis",
        "workloads/device_inv_int/MI100/saved_analysis",
    )


def test_saved_ipblocks_SQ_SPI_MI100():
    compare(
        "workloads/ipblocks_SQ_SPI/MI100/prev_analysis",
        "workloads/ipblocks_SQ_SPI/MI100/saved_analysis",
    )


def test_saved_kernel_summaries_MI100():
    compare(
        "workloads/kernel_summaries/MI100/prev_analysis",
        "workloads/kernel_summaries/MI100/saved_analysis",
    )


def test_saved_dispatch_2_MI100():
    compare(
        "workloads/dispatch_2/MI100/prev_analysis",
        "workloads/dispatch_2/MI100/saved_analysis",
    )


def test_saved_kernel_inv_int_MI100():
    compare(
        "workloads/kernel_inv_int/MI100/prev_analysis",
        "workloads/kernel_inv_int/MI100/saved_analysis",
    )


def test_saved_ipblocks_SQ_SPI_TA_TCC_CPF_MI100():
    compare(
        "workloads/ipblocks_SQ_SPI_TA_TCC_CPF/MI100/prev_analysis",
        "workloads/ipblocks_SQ_SPI_TA_TCC_CPF/MI100/saved_analysis",
    )


def test_saved_dispatch_7_MI100():
    compare(
        "workloads/dispatch_7/MI100/prev_analysis",
        "workloads/dispatch_7/MI100/saved_analysis",
    )


def test_saved_kernel_inv_str_MI100():
    compare(
        "workloads/kernel_inv_str/MI100/prev_analysis",
        "workloads/kernel_inv_str/MI100/saved_analysis",
    )


def test_saved_ipblocks_TCC_MI100():
    compare(
        "workloads/ipblocks_TCC/MI100/prev_analysis",
        "workloads/ipblocks_TCC/MI100/saved_analysis",
    )


def test_saved_ipblocks_SQ_TA_MI100():
    compare(
        "workloads/ipblocks_SQ_TA/MI100/prev_analysis",
        "workloads/ipblocks_SQ_TA/MI100/saved_analysis",
    )


def test_saved_ipblocks_SPI_MI100():
    compare(
        "workloads/ipblocks_SPI/MI100/prev_analysis",
        "workloads/ipblocks_SPI/MI100/saved_analysis",
    )


def test_saved_ipblocks_SQ_CPC_MI100():
    compare(
        "workloads/ipblocks_SQ_CPC/MI100/prev_analysis",
        "workloads/ipblocks_SQ_CPC/MI100/saved_analysis",
    )


def test_saved_dispatch_6_8_MI100():
    compare(
        "workloads/dispatch_6_8/MI100/prev_analysis",
        "workloads/dispatch_6_8/MI100/saved_analysis",
    )


def test_saved_dispatch_0_MI100():
    compare(
        "workloads/dispatch_0/MI100/prev_analysis",
        "workloads/dispatch_0/MI100/saved_analysis",
    )


def test_saved_kernel_verbose_1_MI100():
    compare(
        "workloads/kernel_verbose_1/MI100/prev_analysis",
        "workloads/kernel_verbose_1/MI100/saved_analysis",
    )


def test_saved_ipblocks_TA_MI100():
    compare(
        "workloads/ipblocks_TA/MI100/prev_analysis",
        "workloads/ipblocks_TA/MI100/saved_analysis",
    )


def test_saved_ipblocks_TD_MI100():
    compare(
        "workloads/ipblocks_TD/MI100/prev_analysis",
        "workloads/ipblocks_TD/MI100/saved_analysis",
    )


def test_saved_path_MI100():
    compare("workloads/path/MI100/prev_analysis", "workloads/path/MI100/saved_analysis")


def test_saved_join_type_kernel_MI100():
    compare(
        "workloads/join_type_kernel/MI100/prev_analysis",
        "workloads/join_type_kernel/MI100/saved_analysis",
    )
