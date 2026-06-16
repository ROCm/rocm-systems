// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#ifndef ROCPROFILER_SDK_EXPERIMENTAL
#    define ROCPROFILER_SDK_EXPERIMENTAL
#endif

#include "pc_sample_decode.h"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/pc_sampling.h>

using namespace rocprofiler_compute_tool;

namespace
{
pc_sample_hw_id_t map_hw_id(const rocprofiler_pc_sampling_hw_id_v0_t& hw)
{
    pc_sample_hw_id_t out{};
    out.chiplet          = hw.chiplet;
    out.wave_id          = hw.wave_id;
    out.simd_id          = hw.simd_id;
    out.pipe_id          = hw.pipe_id;
    out.cu_or_wgp_id     = hw.cu_or_wgp_id;
    out.shader_array_id  = hw.shader_array_id;
    out.shader_engine_id = hw.shader_engine_id;
    out.workgroup_id     = hw.workgroup_id;
    out.vm_id            = hw.vm_id;
    out.queue_id         = hw.queue_id;
    out.microengine_id   = hw.microengine_id;
    return out;
}

pc_sample_pc_t map_pc(const rocprofiler_pc_t& pc)
{
    pc_sample_pc_t out{};
    out.code_object_id     = pc.code_object_id;
    out.code_object_offset = pc.code_object_offset;
    return out;
}

pc_sample_dim3_t map_dim3(const rocprofiler_dim3_t& d)
{
    pc_sample_dim3_t out{};
    out.x = d.x;
    out.y = d.y;
    out.z = d.z;
    return out;
}

// Templated because both SDK record structs expose these members by the same names.
template<typename RecT>
void map_common_fields(pc_sample_record_t& out, const RecT& rec)
{
    out.hw_id            = map_hw_id(rec.hw_id);
    out.pc               = map_pc(rec.pc);
    out.exec_mask        = rec.exec_mask;
    out.timestamp        = rec.timestamp;
    out.dispatch_id      = rec.dispatch_id;
    out.corr_id.internal = rec.correlation_id.internal;
    out.corr_id.external = rec.correlation_id.external.value;
    out.wrkgrp_id        = map_dim3(rec.workgroup_id);
    out.wave_in_grp      = rec.wave_in_group;
}
}  // namespace

std::optional<pc_sample_record_t> rocprofiler_compute_tool::decode_pc_sample_record(
    const rocprofiler_record_header_t& header)
{
    if (header.category != ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
        return std::nullopt;

    if (header.kind == ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE)
    {
        const auto& rec = *reinterpret_cast<const rocprofiler_pc_sampling_record_stochastic_v0_t*>(
            header.payload);

        pc_sample_record_t out{};
        out.kind = pc_sample_kind_t::Stochastic;
        map_common_fields(out, rec);
        out.flags.has_mem_cnt = rec.flags.has_memory_counter;
        out.wave_issued       = rec.wave_issued;
        out.inst_type         = rec.inst_type;
        out.wave_cnt          = rec.wave_count;

        out.snapshot.stall_reason = rec.snapshot.reason_not_issued;
        // Copy the plain uint32 fields from the single-sourced field list.
#define PC_SAMPLE_SNAPSHOT_COPY(field) out.snapshot.field = rec.snapshot.field;
        PC_SAMPLE_SNAPSHOT_PLAIN_FIELDS(PC_SAMPLE_SNAPSHOT_COPY)
#undef PC_SAMPLE_SNAPSHOT_COPY

        return out;
    }

    if (header.kind == ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE)
    {
        const auto& rec = *reinterpret_cast<const rocprofiler_pc_sampling_record_host_trap_v0_t*>(
            header.payload);

        pc_sample_record_t out{};
        out.kind = pc_sample_kind_t::HostTrap;
        map_common_fields(out, rec);

        return out;
    }

    return std::nullopt;
}

std::optional<kernel_dispatch_record_t> rocprofiler_compute_tool::decode_kernel_dispatch_record(
    const rocprofiler_record_header_t& header)
{
    if (header.category != ROCPROFILER_BUFFER_CATEGORY_TRACING ||
        header.kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        return std::nullopt;

    const auto& rec = *reinterpret_cast<const rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
        header.payload);
    const auto& di = rec.dispatch_info;

    kernel_dispatch_record_t out{};
    out.size            = rec.size;
    out.kind            = rec.kind;
    out.operation       = rec.operation;
    out.thread_id       = rec.thread_id;
    out.corr_internal   = rec.correlation_id.internal;
    out.corr_external   = rec.correlation_id.external.value;
    out.start_timestamp = rec.start_timestamp;
    out.end_timestamp   = rec.end_timestamp;

    out.dispatch_info_size   = di.size;
    out.agent_id_handle      = di.agent_id.handle;
    out.queue_id_handle      = di.queue_id.handle;
    out.kernel_id            = di.kernel_id;
    out.dispatch_id          = di.dispatch_id;
    out.private_segment_size = di.private_segment_size;
    out.group_segment_size   = di.group_segment_size;
    out.workgroup_size       = map_dim3(di.workgroup_size);
    out.grid_size            = map_dim3(di.grid_size);

    return out;
}
