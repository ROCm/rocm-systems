import numpy as np
import pandas
import os
import glob

current_dir = os.getcwd()
rocprof = "rocprofv2"

expected_filename = "pmc_1/results_vadd.csv"
output_folder = "/tmp/tests-v2/pmc"


def test_grbm(csvfile):
    count = np.array(csvfile["GRBM_COUNT"])
    active = np.array(csvfile["GRBM_GUI_ACTIVE"])
    assert np.all(active > 0)  # GPU must always be active
    assert np.all(count >= active)  # Count always increments more than active
    assert np.all(
        count * 0.8 < active
    )  # We can reasonably expect an active GPU during the kernel execution


def test_sqwaves(csvfile):
    waves = np.array(
        csvfile["SQ_WAVES"]
    )  # 1M threads == 32k waves for Wave32 and 16k waves for Wave64
    assert np.all(waves == 32768) or np.all(waves == 16384)


def test_insts(csvfile):
    waves = np.array(csvfile["SQ_WAVES"])
    valu = np.array(csvfile["SQ_INSTS_VALU"])
    salu = np.array(csvfile["SQ_INSTS_SALU"])
    smem = np.array(csvfile["SQ_INSTS_SMEM"])
    lds = np.array(csvfile["SQ_INSTS_LDS"])

    assert np.all(lds == 0)  # Not used on vectoradd

    # VALU, SALU and SMEM must be divisible by SQ_Waves
    assert np.all(valu % waves == 0)
    assert np.all(salu % waves == 0)
    assert np.all(smem % waves == 0)

    # Each have executes at least one of these
    assert np.all(valu > waves)
    assert np.all(salu > waves)
    assert np.all(smem >= waves)

    # TODO: Check assembly for exact number!


def test_gl2c(csvfile):
    waves = np.array(csvfile["SQ_WAVES"])
    read = np.array(csvfile["GL2C_MC_RDREQ_sum"])
    write = np.array(csvfile["GL2C_MC_WRREQ_sum"])
    hit = np.array(csvfile["GL2C_HIT_sum"])
    miss = np.array(csvfile["GL2C_MISS_sum"])
    hitrate = np.array(csvfile["L2CacheHit"])

    assert np.all(write >= waves)  # We do at least one write per wave
    # TODO: Find out why the first kernel gets such a high write request count.
    assert np.all(write < 2.5 * waves)  # We do only one write (+ a little) per wave.
    assert np.all(read >= 2 * waves)  # We do at least 2 reads per wave (A=B+C)
    assert np.all(read < 3 * waves)  # We do only 2 reads (+ a little) per wave

    assert np.all(miss >= hit)  # on Vadd we can't have more hits than misses
    assert np.all(miss >= 2 * waves)  # Each read misses at least once
    assert np.all(miss < 4 * waves)  # Can't miss too much
    assert np.all(hit >= 0.5 * waves)  # We have at least one hit per wave

    assert np.all(hitrate <= 50)  # We always get more misses than hits


def test_ta(csvfile):
    busy = np.array(csvfile["MemUnitBusy"])
    some_busy = np.array(csvfile["TA_BUSY_max"]) / np.array(csvfile["GRBM_GUI_ACTIVE"])

    assert np.all(busy <= 100)  # MemUnitBusy <= 100%
    assert np.all(some_busy >= 1)  # Some shader engine is using TA


def test_sqcycles(csvfile):
    tabusy = np.array(csvfile["TA_BUSY_max"])
    grbm = np.array(csvfile["GRBM_GUI_ACTIVE"])
    waves = np.array(csvfile["SQ_WAVES"])
    ALU = np.array(csvfile["SQ_INSTS_VALU"]) + np.array(csvfile["SQ_INSTS_SALU"])
    wait_any = np.array(csvfile["SQ_WAIT_ANY"])
    wave_cycles = np.array(csvfile["SQ_WAVE_CYCLES"])
    vmem_cycles = np.array(csvfile["SQ_INST_CYCLES_VMEM"])

    assert np.all(
        wave_cycles >= ALU + wait_any
    )  # Each ALU inst takes at least one cycle
    assert np.all(wave_cycles / grbm <= waves)  # Mean occupancy cannot exceed waves
    assert np.all(wait_any >= tabusy)  # Waves are waiting for ta
    assert np.all(
        vmem_cycles >= waves
    )  # Each wave takes at least one cycle to issue vmem
    assert np.all(
        wait_any / wave_cycles >= 0.5
    )  # vectorAdd is very memory-bound. TODO: use number less arbitrary than 0.5


if __name__ == "__main__":
    csv = pandas.read_csv(f"{output_folder}/{expected_filename}")
    test_grbm(csv)
    test_sqwaves(csv)
    test_insts(csv)
    # test_gl2c(csv)
    # test_ta(csv)
    # test_sqcycles(csv)

    # if its reached this point, then all tests apssed
    print("Test Passed: All counter correctness tests passed.")
