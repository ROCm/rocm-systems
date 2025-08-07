import numpy as np
import pandas
import os
import glob

current_dir = os.getcwd()
rocprof = "rocprofv2"

expected_filename = "pmc_1/results_histo.csv"
output_folder = "/tmp/tests-v2/pmc"


def test_grbm(csvfile):
    count = np.array(csvfile["GRBM_COUNT"])
    active = np.array(csvfile["GRBM_GUI_ACTIVE"])
    assert np.all(active > 0)  # GPU must always be active
    assert np.all(count >= active)  # Count always increments more than active
    assert np.all(
        count * 0.5 < active
    )  # We can reasonably expect an active GPU during the kernel execution


def test_insts(csvfile):
    waves = np.array(
        csvfile["SQ_WAVES"]
    )  # TODO: 256 for wave32, need to check for wave64
    lds = np.array(csvfile["SQ_INSTS_LDS"])
    valu = np.array(csvfile["SQ_INSTS_VALU"])
    salu = np.array(csvfile["SQ_INSTS_SALU"])
    smem = np.array(csvfile["SQ_INSTS_SMEM"])

    assert np.all(waves == 256) or np.all(waves == 128)

    # Each have executes at least one of these
    assert np.all(lds > waves)
    assert np.all(valu > waves)
    assert np.all(salu > waves)
    assert np.all(smem >= waves)


def test_sqcycles(csvfile):
    tabusy = np.array(csvfile["TA_BUSY_max"])
    grbm = np.array(csvfile["GRBM_GUI_ACTIVE"])
    waves = np.array(csvfile["SQ_WAVES"])
    wait_any = np.array(csvfile["SQ_WAIT_ANY"])
    wave_cycles = np.array(csvfile["SQ_WAVE_CYCLES"])
    vmem_cycles = np.array(csvfile["SQ_INST_CYCLES_VMEM"])

    lds = np.array(csvfile["SQ_INSTS_LDS"])
    valu = np.array(csvfile["SQ_INSTS_VALU"])
    salu = np.array(csvfile["SQ_INSTS_SALU"])
    smem = np.array(csvfile["SQ_INSTS_SMEM"])

    ALU = lds + valu + salu + smem

    assert np.all(
        wave_cycles >= ALU + wait_any
    )  # Each ALU inst takes at least one cycle
    assert np.all(wave_cycles / grbm <= waves)  # Mean occupancy cannot exceed waves
    assert np.all(wait_any >= tabusy)  # Waves are waiting for ta
    assert np.all(
        vmem_cycles >= waves
    )  # Each wave takes at least one cycle to issue vmem


if __name__ == "__main__":
    csv = pandas.read_csv(f"{output_folder}/{expected_filename}")
    test_grbm(csv)
    test_insts(csv)
    # test_sqcycles(csv)

    # if its reached this point, then all tests apssed
    print("Test Passed: All counter correctness tests passed.")
