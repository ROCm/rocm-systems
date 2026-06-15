// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#ifndef ROCPROFILER_SDK_EXPERIMENTAL
#    define ROCPROFILER_SDK_EXPERIMENTAL
#endif

#include <rocprofiler-sdk/fwd.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <optional>
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

struct pc_sample_snapshot_t
{
    uint32_t stall_reason    = 0;  // raw SDK not-issued-reason enum; name resolved at serialization
    uint32_t dual_issue_valu = 0;
    uint32_t arb_state_issue_valu = 0, arb_state_issue_matrix = 0, arb_state_issue_lds = 0,
             arb_state_issue_lds_direct = 0, arb_state_issue_scalar = 0,
             arb_state_issue_vmem_tex = 0, arb_state_issue_flat = 0, arb_state_issue_exp = 0,
             arb_state_issue_misc = 0, arb_state_issue_brmsg = 0;
    uint32_t arb_state_stall_valu = 0, arb_state_stall_matrix = 0, arb_state_stall_lds = 0,
             arb_state_stall_lds_direct = 0, arb_state_stall_scalar = 0,
             arb_state_stall_vmem_tex = 0, arb_state_stall_flat = 0, arb_state_stall_exp = 0,
             arb_state_stall_misc = 0, arb_state_stall_brmsg = 0;
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
    size_t               inst_index = 0;
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
    struct view_pair_hash_t
    {
        size_t operator()(const std::pair<std::string_view, std::string_view>& p) const
        {
            const size_t h1 = std::hash<std::string_view>{}(p.first);
            const size_t h2 = std::hash<std::string_view>{}(p.second);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    // Canonical strings live in deques (stable addresses); m_index keys on views
    // into them, so a cache hit allocates nothing.
    std::deque<std::string> m_instructions;
    std::deque<std::string> m_comments;
    std::unordered_map<std::pair<std::string_view, std::string_view>, size_t, view_pair_hash_t> m_index;
};

// Returns std::nullopt when header.category != ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING
// or header.kind is not one of the two supported PC sampling record kinds.
std::optional<pc_sample_record_t> decode_pc_sample_record(const rocprofiler_record_header_t& header);

// Returns std::nullopt when header.category != ROCPROFILER_BUFFER_CATEGORY_TRACING
// or header.kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH.
std::optional<kernel_dispatch_record_t> decode_kernel_dispatch_record(const rocprofiler_record_header_t& header);

class pc_sample_writer_t
{
public:
    static std::shared_ptr<pc_sample_writer_t> create();

    virtual ~pc_sample_writer_t() = default;
    virtual void begin()          = 0;
    // inst_index is the interned (instruction, comment) index for this sample.
    virtual void append_stochastic(const pc_sample_record_t& r, size_t inst_index)              = 0;
    virtual void append_host_trap(const pc_sample_record_t& r, size_t inst_index)               = 0;
    virtual void set_strings(const pc_string_table_t& table)                                    = 0;
    virtual void set_kernel_symbols(const std::vector<kernel_symbol_entry_t>& syms)             = 0;
    virtual void set_agents(const std::vector<agent_record_t>& agents)                          = 0;
    virtual void set_kernel_dispatches(const std::vector<kernel_dispatch_record_t>& dispatches) = 0;
    virtual void set_metadata(int pid)                                                          = 0;
    virtual std::string get_result()                                                            = 0;
    virtual void        flush(const std::filesystem::path& output_file_path)                    = 0;
};

class pc_sample_writer_json_t : public pc_sample_writer_t
{
public:
    void begin() override;
    void append_stochastic(const pc_sample_record_t& r, size_t inst_index) override;
    void append_host_trap(const pc_sample_record_t& r, size_t inst_index) override;
    void set_strings(const pc_string_table_t& table) override;
    void set_kernel_symbols(const std::vector<kernel_symbol_entry_t>& syms) override;
    void set_agents(const std::vector<agent_record_t>& agents) override;
    void set_kernel_dispatches(const std::vector<kernel_dispatch_record_t>& dispatches) override;
    void set_metadata(int pid) override;
    std::string get_result() override;
    void        flush(const std::filesystem::path& output_file_path) override;

private:
    int                                   m_pid = 0;
    std::vector<pc_sample_record_t>       m_stochastic;
    std::vector<pc_sample_record_t>       m_host_trap;
    std::vector<std::string>              m_instructions;
    std::vector<std::string>              m_comments;
    std::vector<kernel_symbol_entry_t>    m_kernel_symbols;
    std::vector<agent_record_t>           m_agents;
    std::vector<kernel_dispatch_record_t> m_kernel_dispatches;
};
}  // namespace rocprofiler_compute_tool
