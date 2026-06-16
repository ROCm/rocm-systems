// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "pair_hash.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofiler_compute_tool
{
enum class pc_sample_kind_t : uint8_t
{
    Stochastic,
    HostTrap
};

struct pc_sample_flags_t
{
    uint32_t has_mem_cnt = 0;
};

struct pc_sample_hw_id_t
{
    uint64_t chiplet          = 0;
    uint64_t wave_id          = 0;
    uint64_t simd_id          = 0;
    uint64_t pipe_id          = 0;
    uint64_t cu_or_wgp_id     = 0;
    uint64_t shader_array_id  = 0;
    uint64_t shader_engine_id = 0;
    uint64_t workgroup_id     = 0;
    uint64_t vm_id            = 0;
    uint64_t queue_id         = 0;
    uint64_t microengine_id   = 0;
};

struct pc_sample_pc_t
{
    uint64_t code_object_id     = 0;
    uint64_t code_object_offset = 0;
};

struct pc_sample_correlation_t
{
    uint64_t internal = 0;
    uint64_t external = 0;
};

struct pc_sample_dim3_t
{
    uint64_t x = 0, y = 0, z = 0;
};

// The plain uint32 snapshot fields (every field except stall_reason, which needs
// special handling on both the decode and serialize sides). Single-sourced here
// so the struct definition, the SDK->struct copy in decode, and the JSON emit are
// all driven from one list; adding/renaming a field is a one-line edit.
#define PC_SAMPLE_SNAPSHOT_PLAIN_FIELDS(X)                                                          \
    X(dual_issue_valu)                                                                              \
    X(arb_state_issue_valu)                                                                         \
    X(arb_state_issue_matrix)                                                                       \
    X(arb_state_issue_lds)                                                                          \
    X(arb_state_issue_lds_direct)                                                                   \
    X(arb_state_issue_scalar)                                                                       \
    X(arb_state_issue_vmem_tex)                                                                     \
    X(arb_state_issue_flat)                                                                         \
    X(arb_state_issue_exp)                                                                          \
    X(arb_state_issue_misc)                                                                         \
    X(arb_state_issue_brmsg)                                                                        \
    X(arb_state_stall_valu)                                                                         \
    X(arb_state_stall_matrix)                                                                       \
    X(arb_state_stall_lds)                                                                          \
    X(arb_state_stall_lds_direct)                                                                   \
    X(arb_state_stall_scalar)                                                                       \
    X(arb_state_stall_vmem_tex)                                                                     \
    X(arb_state_stall_flat)                                                                         \
    X(arb_state_stall_exp)                                                                          \
    X(arb_state_stall_misc)                                                                         \
    X(arb_state_stall_brmsg)

struct pc_sample_snapshot_t
{
    uint32_t stall_reason = 0;  // raw SDK not-issued-reason enum; name resolved at serialization
#define PC_SAMPLE_SNAPSHOT_DECLARE(field) uint32_t field = 0;
    PC_SAMPLE_SNAPSHOT_PLAIN_FIELDS(PC_SAMPLE_SNAPSHOT_DECLARE)
#undef PC_SAMPLE_SNAPSHOT_DECLARE
};

struct pc_sample_record_t
{
    pc_sample_kind_t        kind = pc_sample_kind_t::HostTrap;
    pc_sample_flags_t       flags{};
    pc_sample_hw_id_t       hw_id{};
    pc_sample_pc_t          pc{};
    uint64_t                exec_mask = 0, timestamp = 0, dispatch_id = 0;
    pc_sample_correlation_t corr_id{};
    pc_sample_dim3_t        wrkgrp_id{};
    uint32_t                wave_in_grp = 0;
    uint32_t                wave_issued = 0;
    uint32_t inst_type = 0;  // raw SDK instruction-type enum; name resolved at serialization
    uint32_t wave_cnt  = 0;
    pc_sample_snapshot_t snapshot{};
};

struct kernel_symbol_entry_t
{
    uint64_t    code_object_id = 0;
    std::string formatted_kernel_name{};
    uint64_t    kernel_id = 0;
};

// Minimal agent record; the consumer builds its GPU-id map from GPU-type agents
// sorted by node_id.
struct agent_record_t
{
    uint64_t size            = 0;  // sizeof the SDK agent struct (for SDK-shape parity)
    uint64_t id_handle       = 0;
    uint32_t type            = 0;  // raw rocprofiler_agent_type_t value
    uint32_t node_id         = 0;
    int32_t  logical_node_id = 0;
    uint32_t cu_count        = 0;
    uint64_t gpu_id          = 0;  // KFD identifier (GPU only)
    uint32_t wave_front_size = 0;
    uint32_t simd_count      = 0;
};

struct kernel_dispatch_record_t
{
    uint64_t size            = 0;
    uint32_t kind            = 0;  // raw rocprofiler_buffer_tracing_kind_t value
    uint32_t operation       = 0;
    uint64_t thread_id       = 0;
    uint64_t corr_internal   = 0;
    uint64_t corr_external   = 0;
    uint64_t start_timestamp = 0;
    uint64_t end_timestamp   = 0;

    uint64_t         dispatch_info_size   = 0;
    uint64_t         agent_id_handle      = 0;
    uint64_t         queue_id_handle      = 0;
    uint64_t         kernel_id            = 0;
    uint64_t         dispatch_id          = 0;
    uint32_t         private_segment_size = 0;
    uint32_t         group_segment_size   = 0;
    pc_sample_dim3_t workgroup_size{};
    pc_sample_dim3_t grid_size{};
};

class pc_string_table_t
{
public:
    // Dedups by (instruction_text, comment); returns the shared index (position in BOTH arrays).
    size_t insert(const std::string& instruction_text, const std::string& comment);
    const std::deque<std::string>& instructions() const;
    const std::deque<std::string>& comments() const;

private:
    // Canonical strings live in deques (stable addresses); m_index keys on views
    // into them, so a cache hit allocates nothing.
    std::deque<std::string> m_instructions;
    std::deque<std::string> m_comments;
    std::unordered_map<std::pair<std::string_view, std::string_view>, size_t, pair_hash_t> m_index;
};
}  // namespace rocprofiler_compute_tool
