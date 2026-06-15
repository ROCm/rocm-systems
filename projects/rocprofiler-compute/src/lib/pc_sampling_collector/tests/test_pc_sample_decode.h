// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#define ROCPROFILER_SDK_EXPERIMENTAL

#include "gtest/gtest.h"
#include "pc_sample_writer.h"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/pc_sampling.h>

class test_pc_sample_decode_t : public ::testing::Test
{
protected:
    // Fully-populated stochastic record.
    static rocprofiler_pc_sampling_record_stochastic_v0_t make_stochastic_record()
    {
        rocprofiler_pc_sampling_record_stochastic_v0_t rec{};
        rec.size = sizeof(rec);

        rec.hw_id.chiplet          = 3;
        rec.hw_id.wave_id          = 5;
        rec.hw_id.simd_id          = 2;
        rec.hw_id.pipe_id          = 4;
        rec.hw_id.cu_or_wgp_id     = 6;
        rec.hw_id.shader_array_id  = 1;
        rec.hw_id.shader_engine_id = 7;
        rec.hw_id.workgroup_id     = 9;
        rec.hw_id.vm_id            = 11;
        rec.hw_id.queue_id         = 4;
        rec.hw_id.microengine_id   = 2;

        rec.pc.code_object_id     = 2;
        rec.pc.code_object_offset = 8040;

        rec.exec_mask   = 0xFFFFFFFFFFFFFFFFULL;
        rec.timestamp   = 1234567890ULL;
        rec.dispatch_id = 3;

        rec.correlation_id.internal       = 3;
        rec.correlation_id.external.value = 99;

        rec.workgroup_id.x = 142;
        rec.workgroup_id.y = 0;
        rec.workgroup_id.z = 0;

        rec.wave_in_group = 1;
        rec.wave_issued   = 0;
        rec.inst_type     = ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU;
        rec.wave_count    = 12;

        rec.snapshot.reason_not_issued = ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT;
        rec.snapshot.dual_issue_valu      = 1;
        rec.snapshot.arb_state_issue_valu = 1;
        rec.snapshot.arb_state_stall_lds  = 1;

        return rec;
    }

    static rocprofiler_pc_sampling_record_host_trap_v0_t make_host_trap_record()
    {
        rocprofiler_pc_sampling_record_host_trap_v0_t rec{};
        rec.size = sizeof(rec);

        rec.hw_id.chiplet          = 3;
        rec.hw_id.wave_id          = 5;
        rec.hw_id.simd_id          = 2;
        rec.hw_id.pipe_id          = 4;
        rec.hw_id.cu_or_wgp_id     = 6;
        rec.hw_id.shader_array_id  = 1;
        rec.hw_id.shader_engine_id = 7;
        rec.hw_id.workgroup_id     = 9;
        rec.hw_id.vm_id            = 11;
        rec.hw_id.queue_id         = 4;
        rec.hw_id.microengine_id   = 2;

        rec.pc.code_object_id     = 2;
        rec.pc.code_object_offset = 8040;

        rec.exec_mask   = 0xFFFFFFFFFFFFFFFFULL;
        rec.timestamp   = 1234567890ULL;
        rec.dispatch_id = 3;

        rec.correlation_id.internal       = 3;
        rec.correlation_id.external.value = 99;

        rec.workgroup_id.x = 142;
        rec.workgroup_id.y = 0;
        rec.workgroup_id.z = 0;

        rec.wave_in_group = 1;

        return rec;
    }

    // Wrap a record payload into a buffer record header with the given (category, kind).
    static rocprofiler_record_header_t make_header(uint32_t category, uint32_t kind, void* payload)
    {
        rocprofiler_record_header_t header{};
        header.category = category;
        header.kind     = kind;
        header.payload  = payload;
        return header;
    }
};
