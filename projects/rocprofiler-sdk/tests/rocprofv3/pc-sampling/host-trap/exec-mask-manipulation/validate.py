#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import itertools
import sys
import pytest
import numpy as np
import pandas as pd


# =========================== Validating CSV output


# Keep this in case we decide to revert workgroup_id information
def validate_workgoup_id_x_y_z(df, max_x, max_y, max_z):
    assert (df["Workgroup_Size_X"].astype(int) >= 0).all()
    assert (df["Workgroup_Size_X"].astype(int) <= max_x).all()

    assert (df["Workgroup_Size_Y"].astype(int) >= 0).all()
    assert (df["Workgroup_Size_Y"].astype(int) <= max_y).all()

    assert (df["Workgroup_Size_Z"].astype(int) >= 0).all()
    assert (df["Workgroup_Size_Z"].astype(int) <= max_z).all()


# Keep this in case we decide to revert wave_id information
def validate_wave_id(df, max_wave_id):
    assert (df["Wave_Id"].astype(int) <= max_wave_id).all()


# Keep this in case we decide to revert wave_id information
def validate_chiplet(df, max_chiplet):
    assert (df["Chiplet"].astype(int) <= max_chiplet).all()


def validate_instruction_decoding(
    df,
    inst_str,
    exec_mask_uint64: np.uint64 = None,
    source_code_lines_range: (int, int) = None,
    all_source_lines_samples=False,
):
    # Make a copy, so that we don't work (modify) a view.
    df_inst = df[df["Instruction"].apply(lambda inst: inst.startswith(inst_str))].copy()

    assert not df_inst.empty
    # assert the exec mask if requested
    if exec_mask_uint64 is not None:
        assert (df_inst["Exec_Mask"].astype(np.uint64) == exec_mask_uint64).all()

    # assert whether the samples source code lines belongs to the provided range
    if source_code_lines_range is not None:
        start_range, end_range = source_code_lines_range
        # The instruction comment is isually in the following format: /path/to/source/file.cpp:line_num
        df_inst["source_line_num"] = df_inst["Instruction_Comment"].apply(
            lambda source_line: int(source_line.split(":")[-1])
        )
        assert (df_inst["source_line_num"] >= start_range).all()
        assert (df_inst["source_line_num"] <= end_range).all()
        # if requested, check if all lines from the range are sampled
        if all_source_lines_samples:
            assert len(df_inst["source_line_num"].unique()) == (
                end_range - start_range + 1
            )


def validate_instruction_comment(df):
    # Instruction comment must always be present, since the testing application
    # is built with debug symbols.
    assert (
        (df["Instruction_Comment"] != "") & (df["Instruction_Comment"] != "nullptr")
    ).all()


def validate_instruction_correlation_id_relation(df):
    # Samples with no decoded instructions originates from either
    # blit kernels or self modifying code. The correlation id for this
    # type of samples should alway be zero.
    # Thus, Correlation_Id is 0 `iff`` instruction is not decoded.

    # The previous statement has two implications.
    # Implication 1: If the instruction is not decoded, then correlation id is 0.
    samples_no_instruction_df = df[
        (df["Instruction"] == "") | (df["Instruction"] == "nullptr")
    ]
    assert (samples_no_instruction_df["Correlation_Id"] == 0).all()

    # Implication 2: If the correlation id is 0, then the instruction is not decoded.
    samples_cid_zero_df = df[df["Correlation_Id"] == 0]
    assert (
        (samples_cid_zero_df["Instruction"] == "")
        | (samples_cid_zero_df["Instruction"] == "nullptr")
    ).all()

    assert len(samples_no_instruction_df) == len(samples_cid_zero_df)

    # Since we're not enabling any kind of API tracing,
    # internal correlation id should match the dispatch id
    assert all(df["Correlation_Id"] == df["Dispatch_Id"])


def validate_exec_mask_based_on_correlation_id(df):
    # The function assumes that each kernel launches 1024 blocks.
    # Each block contains number of threads that matches correlation ID of the kernel.
    # The exec mask of a sample should contain number of ones equal to
    # the correlation ID of the kernel during which execution the sample was generated.
    df["active_SIMD_threads"] = df["Exec_Mask"].apply(
        lambda exec_mask: bin(exec_mask).count("1")
    )
    assert (df["active_SIMD_threads"] == df["Correlation_Id"]).all()

    # TODO: Comment out the following code if it causes spurious fails.
    # The more conservative constraint based on the experience follows.
    # The exec mask of sampled instructions of the kernels respect the following pattern:
    # cid -> exec
    # 1 -> 0b1
    # 2 -> 0b11
    # 3 -> 0b111
    # ...
    # 64 -> 0xffffffffffffffff

    df["Exec_Mask2"] = (
        df["Correlation_Id"].astype(int).apply(lambda x: int("0b" + (x * "1"), 2))
    )

    # TODO: exec should be in hex and that will ease the comparison
    assert (df["Exec_Mask"].astype(np.uint64) == df["Exec_Mask2"].astype(np.uint64)).all()


def exec_mask_manipulation_validate_csv(df, all_sampled=False):
    assert not df.empty

    validate_instruction_comment(df)
    validate_instruction_correlation_id_relation(df)

    # Validate samples with non-zero correlation IDs (and with decoded instructions)
    samples_cid_non_zero_df = df[df["Correlation_Id"] != 0]

    # exactly 65 kernels and 65 correlation id
    assert (samples_cid_non_zero_df["Correlation_Id"].astype(int) >= 1).all()
    assert (samples_cid_non_zero_df["Correlation_Id"].astype(int) <= 65).all()
    if all_sampled:
        # all correlation IDs must be sampled
        assert len(samples_cid_non_zero_df["Correlation_Id"].astype(int).unique()) == 65

    first_64_kernels_df = samples_cid_non_zero_df[
        samples_cid_non_zero_df["Correlation_Id"] <= 64
    ]

    # Make a copy, so that we don't work (modify) a view.
    validate_exec_mask_based_on_correlation_id(first_64_kernels_df.copy())

    # validate the last kernel
    kernel_65_df = df[df["Correlation_Id"] == 65]

    # assert that v_rcp instructions are properly decoded
    # the v_rcp is executed by even SIMD threads
    validate_instruction_decoding(
        kernel_65_df,
        "v_rcp_f64",
        exec_mask_uint64=np.uint64(int("5555555555555555", 16)),
        source_code_lines_range=(288, 387),
        all_source_lines_samples=all_sampled,
    )

    # assert that v_rcp_f32 instructions are properly decoded
    # the v_rcp_f32 is executed by odd SIMD threads
    validate_instruction_decoding(
        kernel_65_df,
        "v_rcp_f32",
        exec_mask_uint64=np.uint64(int("AAAAAAAAAAAAAAAA", 16)),
        source_code_lines_range=(391, 490),
        all_source_lines_samples=all_sampled,
    )


def test_validate_pc_sampling_exec_mask_manipulation_csv(
    input_csv: pd.DataFrame, all_sampled: bool
):
    exec_mask_manipulation_validate_csv(input_csv, all_sampled=all_sampled)


# ========================= Validating JSON output


def validate_json_exec_mask_manipulation(data_json, all_sampled=False):
    # Although functional programming might look more elegant,
    # I was trying to avoid multiple iteration over the list of samples.
    # Thus, I decided to use procedural programming instead.
    # Although, it would be more elegant to wrap some of the checks in dedicated functions,
    # I noticed that it can introduce significant overhead, so I decided to inline those checks.

    # the function assume homogenous system
    agents = data_json["agents"]
    gpu_agents = list(filter(lambda agent: agent["type"] == 2, agents))
    # There should be at least one GPU agent
    assert len(gpu_agents) > 0
    first_gpu_agent = gpu_agents[0]
    num_xcc = first_gpu_agent["num_xcc"]
    max_waves_per_simd = first_gpu_agent["max_waves_per_simd"]
    simd_per_cu = first_gpu_agent["simd_per_cu"]

    instructions = data_json["strings"]["pc_sample_instructions"]
    comments = data_json["strings"]["pc_sample_comments"]

    # execution mask where even SIMD lanes are active
    # correspond to the v_rcp_f64 instructions of the last kernel
    even_simds_active_exec_mask = np.uint64(int("5555555555555555", 16))
    # start and end source code lines of the v_rcp_f64 instructions of the last kernel
    v_rcp_f64_start_line_num, v_rcp_f64_end_line_num = 288, 387
    # execution mask where even SIMD lanes are active
    # correspond to the v_rcp_f64 instructions of the last kernel
    odd_simds_active_exec_mask = np.uint64(int("AAAAAAAAAAAAAAAA", 16))
    # start and end source code lines of the v_rcp_f32 0 instructions of the last kernel
    v_rcp_f32_start_line_num, v_rcp_f32_end_line_num = 391, 490

    # sampled wave_ids of the last kernel
    kernel65_sampled_wave_in_grp = set()
    # sampled source lines of the last kernel matching v_rcp_f64 instructions
    kernel65_v_rcp_64_sampled_source_line_set = set()
    # sampled source lines of the last kernel matching v_rcp_f64 instructions
    kernel65_v_rcp_f32_sampled_source_line_set = set()
    # sampled correlation IDs
    sampled_cids_set = set()
    # pairs of sampled SIMD ids and waveslot IDs
    sampled_simd_waveslots_pairs = set()
    # sampled chiplets
    sampled_chiplets = set()
    # sample VMIDs
    sampled_vmids = set()

    for sample in data_json["buffer_records"]["pc_sample_host_trap"]:
        record = sample["record"]
        cid = record["corr_id"]["internal"]

        # pull information from hw_id
        hw_id = record["hw_id"]
        sampled_chiplets.add(hw_id["chiplet"])
        sampled_simd_waveslots_pairs.add((hw_id["simd_id"], hw_id["wave_id"]))
        sampled_vmids.add(hw_id["vm_id"])

        # Checks specific for all samples

        # cids must be non-negative numbers
        assert cid >= 0

        inst_index = sample["inst_index"]

        # Since we're not enabling any kind of API tracing, the internal correlation id should
        # be equal to the dispatch_id
        assert cid == record["dispatch_id"]

        if cid == 0:
            # Samples originates either from a blit kernel or self-modifying code.
            # Thus, code object is uknown, as well as the instruction.
            assert record["pc"]["code_object_id"] == 0
            assert inst_index == -1
        else:
            # Update set of sampled cids
            sampled_cids_set.add(cid)

            # All samples with non-zero correlation ID should pass the following checks
            # code object is know, so as the instruction
            assert record["pc"]["code_object_id"] != 0
            assert inst_index != -1

            wgid = record["wrkgrp_id"]
            # check corrdinates of the workgroup
            assert wgid["x"] >= 0 and wgid["x"] <= 1023
            assert wgid["y"] == 0
            assert wgid["z"] == 0

            wave_in_grp = record["wave_in_grp"]
            exec_mask = record["exec_mask"]

            if cid < 65:
                # checks specific for samples from first 64 kernels
                assert wave_in_grp == 0
                # inline if possible
                # validate_json_exec_mask_based_on_cid(sample.record)

                # The function assumes that each kernel launches 1024 blocks.
                # Each block contains number of threads that matches correlation ID of the kernel.
                # The exec mask of a sample should contain number of ones equal to
                # the correlation ID of the kernel during which execution the sample was generated.
                assert bin(exec_mask).count("1") == cid

                # TODO: Comment out the following code if it causes spurious fails.
                # The more conservative constraint based on the experience follows.
                # The exec mask of sampled instructions of the kernels respect the following pattern:
                # cid -> exec
                # 1 -> 0b1
                # 2 -> 0b11
                # 3 -> 0b111
                # ...
                # 64 -> 0xffffffffffffffff
                exec_mask_str = "0b" + "1" * cid
                assert np.uint64(exec_mask) == np.uint64(int(exec_mask_str, 2))
            else:
                # No more that 65 cids
                assert cid == 65
                # Monitor wave_in_group being sampled
                kernel65_sampled_wave_in_grp.add(wave_in_grp)
                # chekcs specific for samples from the last kernel
                assert wave_in_grp >= 0 and wave_in_grp <= 3

                # validate instruction decoding
                inst = instructions[inst_index]
                comm = comments[inst_index]
                # The instruction comment is isually in the following format:
                # /path/to/source/file.cpp:line_num
                line_num = int(comm.split(":")[-1])
                if inst.startswith("v_rcp_f64"):
                    # even SIMD lanes active
                    assert np.uint64(exec_mask) == even_simds_active_exec_mask
                    assert (
                        line_num >= v_rcp_f64_start_line_num
                        and line_num <= v_rcp_f64_end_line_num
                    )
                    kernel65_v_rcp_64_sampled_source_line_set.add(line_num)
                elif inst.startswith("v_rcp_f32"):
                    # odd SIMD lanes active
                    assert np.uint64(exec_mask) == odd_simds_active_exec_mask
                    assert (
                        line_num >= v_rcp_f32_start_line_num
                        and line_num <= v_rcp_f32_end_line_num
                    )
                    kernel65_v_rcp_f32_sampled_source_line_set.add(line_num)

    if all_sampled:
        # All cids that belongs to the range [1, 65] should be samples
        assert len(sampled_cids_set) == 65

        # all wave_ids that belongs to the range [0, 3] should be sampled for the last kernel
        assert len(kernel65_sampled_wave_in_grp) == 4

        # all source lines matches v_rcp_f64 instructions of the last kernel should be sampled
        assert len(kernel65_v_rcp_64_sampled_source_line_set) == (
            v_rcp_f64_end_line_num - v_rcp_f64_start_line_num + 1
        )
        # all source lines matches v_rcp_f32 instructions of the last kernel should be sampled
        assert len(kernel65_v_rcp_f32_sampled_source_line_set) == (
            v_rcp_f32_end_line_num - v_rcp_f32_start_line_num + 1
        )

        # all chiplets must be sampled
        assert len(sampled_chiplets) == num_xcc
        # all (simd ID, waveslot ID) pairs must be samples
        assert len(sampled_simd_waveslots_pairs) == simd_per_cu * max_waves_per_simd

    # assert chiplet index
    assert all(map(lambda chiplet: 0 <= chiplet < num_xcc, sampled_chiplets))
    # assert (SIMD ID, waveslot ID) combinations
    assert all(
        map(
            lambda simd_waveslot: (0 <= simd_waveslot[0] < simd_per_cu)
            and (0 <= simd_waveslot[1] < max_waves_per_simd),
            sampled_simd_waveslots_pairs,
        )
    )

    # Apparently, not all dispatches must belong to the same VMID,
    # so I'm temporarily disabling the following check.
    # # all samples should belong to the same VMID
    # assert len(sampled_vmids) == 1


def test_validate_pc_sampling_exec_mask_manipulation_json(
    input_json, input_csv: pd.DataFrame, all_sampled: bool
):
    data = input_json["rocprofiler-sdk-tool"]
    # The same amount of samples should be in both CSV and JSON files.
    assert len(input_csv) == len(data["buffer_records"]["pc_sample_host_trap"])
    # # validating JSON output
    validate_json_exec_mask_manipulation(data, all_sampled=all_sampled)


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
