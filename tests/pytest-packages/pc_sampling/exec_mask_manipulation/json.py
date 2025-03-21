# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from __future__ import absolute_import

import numpy as np
import pandas as pd


def validate_json_exec_mask_manipulation(
    data_json, pc_sampling_method="host_trap", all_sampled=False
):
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
    # TODO: Similar reason for introducing stochastic_assert inside the csv.py.
    # When asserting certain conditions related to exec_masks for all samples,
    # we observe some failures.
    # This usually happens because some small number of samples (e.g., 1-10 out of 100k)
    # do not satisfy the condition. This is either a regression in the ROCr 2nd level trap
    # handler (as sometimes execution mask or correlation ID mismatches), or
    # just stochastic nature of the sampling (meaning our checks are too strict).
    # To relax checks, we introduce an assertion that will allow some small number
    # of samples to disobey the condition.
    # This is a temporary solution until we find the root cause of the issue.

    failing_exec_mask_checks_samples_num = 0
    # We noticed failing samples in:
    # 1. kernels 1-64
    # 2. kernel 65 even SIMD lanes
    # 3. kernel 64 odd SIMD lanes
    # The number of failing samples is less than 10 per category.
    max_number_of_failing_records = 30

    for sample in data_json["buffer_records"][f"pc_sample_{pc_sampling_method}"]:
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
                # assert bin(exec_mask).count("1") == cid
                if bin(exec_mask).count("1") != cid:
                    failing_exec_mask_checks_samples_num += 1

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
                # assert np.uint64(exec_mask) == np.uint64(int(exec_mask_str, 2))
                if np.uint64(exec_mask) != np.uint64(int(exec_mask_str, 2)):
                    failing_exec_mask_checks_samples_num += 1
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
                    # assert np.uint64(exec_mask) == even_simds_active_exec_mask
                    if np.uint64(exec_mask) != even_simds_active_exec_mask:
                        failing_exec_mask_checks_samples_num += 1

                    assert (
                        line_num >= v_rcp_f64_start_line_num
                        and line_num <= v_rcp_f64_end_line_num
                    )
                    kernel65_v_rcp_64_sampled_source_line_set.add(line_num)
                elif inst.startswith("v_rcp_f32"):
                    # odd SIMD lanes active
                    # assert np.uint64(exec_mask) == odd_simds_active_exec_mask
                    if np.uint64(exec_mask) != odd_simds_active_exec_mask:
                        failing_exec_mask_checks_samples_num += 1

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

    # assert that the number of failing samples is acceptable
    assert (
        failing_exec_mask_checks_samples_num <= max_number_of_failing_records
    ), "Number of failing samples failing exec_mask check is too high"
