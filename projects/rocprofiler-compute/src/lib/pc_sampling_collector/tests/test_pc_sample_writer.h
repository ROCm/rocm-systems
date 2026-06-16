// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "gtest/gtest.h"
#include "pc_sample_writer.h"

#include <rocprofiler-sdk/pc_sampling.h>

class test_pc_sample_writer_t : public ::testing::Test
{
protected:
    rocprofiler_compute_tool::pc_sample_writer_json_t m_writer;

    static rocprofiler_compute_tool::pc_sample_record_t make_stochastic_record()
    {
        rocprofiler_compute_tool::pc_sample_record_t r{};
        r.kind = rocprofiler_compute_tool::pc_sample_kind_t::Stochastic;

        r.flags.has_mem_cnt = 1;

        r.hw_id.chiplet          = 1;
        r.hw_id.wave_id          = 2;
        r.hw_id.simd_id          = 3;
        r.hw_id.pipe_id          = 4;
        r.hw_id.cu_or_wgp_id     = 5;
        r.hw_id.shader_array_id  = 6;
        r.hw_id.shader_engine_id = 7;
        r.hw_id.workgroup_id     = 8;
        r.hw_id.vm_id            = 9;
        r.hw_id.queue_id         = 10;
        r.hw_id.microengine_id   = 11;

        r.pc.code_object_id     = 42;
        r.pc.code_object_offset = 0x1234;

        r.exec_mask   = 0xABCDEF;
        r.timestamp   = 555;
        r.dispatch_id = 77;

        r.corr_id.internal = 3;
        r.corr_id.external = 99;

        r.wrkgrp_id.x = 12;
        r.wrkgrp_id.y = 0;
        r.wrkgrp_id.z = 0;

        r.wave_in_grp = 2;
        r.wave_issued = 1;
        r.inst_type   = ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU;
        r.wave_cnt    = 27;

        r.snapshot.stall_reason    = ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT;
        r.snapshot.dual_issue_valu = 1;

        r.snapshot.arb_state_issue_valu       = 1;
        r.snapshot.arb_state_issue_matrix     = 2;
        r.snapshot.arb_state_issue_lds        = 3;
        r.snapshot.arb_state_issue_lds_direct = 4;
        r.snapshot.arb_state_issue_scalar     = 5;
        r.snapshot.arb_state_issue_vmem_tex   = 6;
        r.snapshot.arb_state_issue_flat       = 7;
        r.snapshot.arb_state_issue_exp        = 8;
        r.snapshot.arb_state_issue_misc       = 9;
        r.snapshot.arb_state_issue_brmsg      = 10;

        r.snapshot.arb_state_stall_valu       = 11;
        r.snapshot.arb_state_stall_matrix     = 12;
        r.snapshot.arb_state_stall_lds        = 13;
        r.snapshot.arb_state_stall_lds_direct = 14;
        r.snapshot.arb_state_stall_scalar     = 15;
        r.snapshot.arb_state_stall_vmem_tex   = 16;
        r.snapshot.arb_state_stall_flat       = 17;
        r.snapshot.arb_state_stall_exp        = 18;
        r.snapshot.arb_state_stall_misc       = 19;
        r.snapshot.arb_state_stall_brmsg      = 20;

        return r;
    }

    static rocprofiler_compute_tool::pc_sample_record_t make_host_trap_record()
    {
        rocprofiler_compute_tool::pc_sample_record_t r{};
        r.kind = rocprofiler_compute_tool::pc_sample_kind_t::HostTrap;

        r.hw_id.chiplet          = 1;
        r.hw_id.wave_id          = 2;
        r.hw_id.simd_id          = 3;
        r.hw_id.pipe_id          = 4;
        r.hw_id.cu_or_wgp_id     = 5;
        r.hw_id.shader_array_id  = 6;
        r.hw_id.shader_engine_id = 7;
        r.hw_id.workgroup_id     = 8;
        r.hw_id.vm_id            = 9;
        r.hw_id.queue_id         = 10;
        r.hw_id.microengine_id   = 11;

        r.pc.code_object_id     = 42;
        r.pc.code_object_offset = 0x1234;

        r.exec_mask   = 0xABCDEF;
        r.timestamp   = 555;
        r.dispatch_id = 77;

        r.corr_id.internal = 3;
        r.corr_id.external = 99;

        r.wrkgrp_id.x = 12;
        r.wrkgrp_id.y = 0;
        r.wrkgrp_id.z = 0;

        r.wave_in_grp = 2;

        return r;
    }

    static rocprofiler_compute_tool::agent_record_t make_agent_record()
    {
        rocprofiler_compute_tool::agent_record_t a{};
        a.size            = 312;
        a.id_handle       = 18942;
        a.type            = 2;  // GPU
        a.node_id         = 2;
        a.logical_node_id = 2;
        a.cu_count        = 304;
        a.gpu_id          = 4567;
        a.wave_front_size = 64;
        a.simd_count      = 1216;
        return a;
    }

    static rocprofiler_compute_tool::kernel_dispatch_record_t make_kernel_dispatch_record()
    {
        rocprofiler_compute_tool::kernel_dispatch_record_t d{};
        d.size            = 184;
        d.kind            = 11;
        d.operation       = 2;
        d.thread_id       = 298597;
        d.corr_internal   = 1;
        d.corr_external   = 0;
        d.start_timestamp = 1987779595190273ULL;
        d.end_timestamp   = 1987779595198565ULL;

        d.dispatch_info_size   = 72;
        d.agent_id_handle      = 18942;
        d.queue_id_handle      = 2;
        d.kernel_id            = 12;
        d.dispatch_id          = 1;
        d.private_segment_size = 0;
        d.group_segment_size   = 0;
        d.workgroup_size       = {256, 1, 1};
        d.grid_size            = {1048576, 1, 1};
        return d;
    }
};
