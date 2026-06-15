// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_pc_sample_decode.h"

namespace rct = rocprofiler_compute_tool;

TEST_F(test_pc_sample_decode_t, ProvidedStochasticHeader_MapsEveryFieldOneToOne)
{
    auto rec    = make_stochastic_record();
    auto header = make_header(ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING,
                              ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE,
                              &rec);

    const auto decoded = rct::decode_pc_sample_record(header);
    ASSERT_TRUE(decoded.has_value());

    EXPECT_EQ(decoded->kind, rct::pc_sample_kind_t::Stochastic);

    EXPECT_EQ(decoded->hw_id.chiplet, 3u);
    EXPECT_EQ(decoded->hw_id.wave_id, 5u);
    EXPECT_EQ(decoded->hw_id.simd_id, 2u);
    EXPECT_EQ(decoded->hw_id.pipe_id, 4u);
    EXPECT_EQ(decoded->hw_id.cu_or_wgp_id, 6u);
    EXPECT_EQ(decoded->hw_id.shader_array_id, 1u);
    EXPECT_EQ(decoded->hw_id.shader_engine_id, 7u);
    EXPECT_EQ(decoded->hw_id.workgroup_id, 9u);
    EXPECT_EQ(decoded->hw_id.vm_id, 11u);
    EXPECT_EQ(decoded->hw_id.queue_id, 4u);
    EXPECT_EQ(decoded->hw_id.microengine_id, 2u);

    EXPECT_EQ(decoded->pc.code_object_id, 2u);
    EXPECT_EQ(decoded->pc.code_object_offset, 8040u);

    EXPECT_EQ(decoded->exec_mask, 0xFFFFFFFFFFFFFFFFULL);
    EXPECT_EQ(decoded->timestamp, 1234567890ULL);
    EXPECT_EQ(decoded->dispatch_id, 3u);

    EXPECT_EQ(decoded->corr_id.internal, 3u);
    EXPECT_EQ(decoded->corr_id.external, 99u);

    EXPECT_EQ(decoded->wrkgrp_id.x, 142u);
    EXPECT_EQ(decoded->wrkgrp_id.y, 0u);
    EXPECT_EQ(decoded->wrkgrp_id.z, 0u);

    EXPECT_EQ(decoded->wave_in_grp, 1u);
    EXPECT_EQ(decoded->wave_issued, 0u);
    EXPECT_EQ(decoded->wave_cnt, 12u);

    // Decode stores the raw SDK enum value; the name is resolved at serialization.
    EXPECT_EQ(decoded->inst_type, static_cast<uint32_t>(ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU));

    EXPECT_EQ(decoded->snapshot.stall_reason,
              static_cast<uint32_t>(ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT));

    EXPECT_EQ(decoded->snapshot.dual_issue_valu, 1u);
    EXPECT_EQ(decoded->snapshot.arb_state_issue_valu, 1u);
    EXPECT_EQ(decoded->snapshot.arb_state_stall_lds, 1u);
}

TEST_F(test_pc_sample_decode_t, ProvidedHostTrapHeader_MapsCommonFieldsAndLeavesStochasticDefault)
{
    auto rec    = make_host_trap_record();
    auto header = make_header(ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING,
                              ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE,
                              &rec);

    const auto decoded = rct::decode_pc_sample_record(header);
    ASSERT_TRUE(decoded.has_value());

    EXPECT_EQ(decoded->kind, rct::pc_sample_kind_t::HostTrap);

    EXPECT_EQ(decoded->hw_id.chiplet, 3u);
    EXPECT_EQ(decoded->hw_id.wave_id, 5u);
    EXPECT_EQ(decoded->hw_id.shader_engine_id, 7u);

    EXPECT_EQ(decoded->pc.code_object_id, 2u);
    EXPECT_EQ(decoded->pc.code_object_offset, 8040u);

    EXPECT_EQ(decoded->exec_mask, 0xFFFFFFFFFFFFFFFFULL);
    EXPECT_EQ(decoded->timestamp, 1234567890ULL);
    EXPECT_EQ(decoded->dispatch_id, 3u);

    EXPECT_EQ(decoded->corr_id.internal, 3u);
    EXPECT_EQ(decoded->corr_id.external, 99u);

    EXPECT_EQ(decoded->wrkgrp_id.x, 142u);
    EXPECT_EQ(decoded->wave_in_grp, 1u);

    // Stochastic-only fields are left default (0) for host-trap samples.
    EXPECT_EQ(decoded->wave_cnt, 0u);
    EXPECT_EQ(decoded->inst_type, 0u);
    EXPECT_EQ(decoded->snapshot.stall_reason, 0u);
}

TEST_F(test_pc_sample_decode_t, ProvidedNonPcSamplingCategory_ReturnsNullopt)
{
    auto rec    = make_stochastic_record();
    auto header = make_header(ROCPROFILER_BUFFER_CATEGORY_TRACING,
                              ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE,
                              &rec);

    EXPECT_FALSE(rct::decode_pc_sample_record(header).has_value());
}

TEST_F(test_pc_sample_decode_t, ProvidedUnsupportedKind_ReturnsNullopt)
{
    auto rec    = make_stochastic_record();
    auto header = make_header(ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING, 0xFFFFFFFFu, &rec);

    EXPECT_FALSE(rct::decode_pc_sample_record(header).has_value());
}
