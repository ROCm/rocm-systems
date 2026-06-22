// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "att_no_intercept.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace tool
{
namespace att_no_intercept
{
struct entry_key
{
    uint64_t code_object_id = 0;
    uint64_t address        = 0;

    friend bool operator==(const entry_key& lhs, const entry_key& rhs)
    {
        return lhs.code_object_id == rhs.code_object_id && lhs.address == rhs.address;
    }
};

struct entry_key_hash
{
    size_t operator()(const entry_key& key) const
    {
        return static_cast<size_t>((key.address >> 2) ^ (key.code_object_id << 24) ^
                                   (key.code_object_id >> 40));
    }
};

struct code_object_record
{
    uint64_t code_object_id = 0;
    int64_t  load_delta     = 0;
    uint64_t load_base      = 0;
    uint64_t load_size      = 0;

    std::unordered_map<std::string, uint64_t> symbol_sizes_by_name  = {};
    std::unordered_map<uint64_t, uint64_t>    symbol_sizes_by_vaddr = {};
};

struct kernel_symbol_record
{
    rocprofiler_kernel_id_t kernel_id                     = 0;
    uint64_t                code_object_id                = 0;
    std::string             kernel_name                   = {};
    uint64_t                kernel_address                = 0;
    int64_t                 kernel_code_entry_byte_offset = 0;
};

struct kernel_symbol_range
{
    rocprofiler_kernel_id_t kernel_id      = 0;
    uint64_t                code_object_id = 0;
    uint64_t                begin          = 0;
    uint64_t                end            = 0;
};

struct agent_stats
{
    std::atomic<uint64_t> chunks_scanned      = {0};
    std::atomic<uint64_t> bytes_scanned       = {0};
    std::atomic<uint64_t> dispatches_seen     = {0};
    std::atomic<uint64_t> unknown_dispatches  = {0};
    std::atomic<uint64_t> target_dispatches   = {0};
    std::atomic<uint64_t> cross_chunk_skips   = {0};
    std::atomic<uint64_t> cuts_built          = {0};
    std::atomic<uint64_t> cut_build_failures  = {0};
    std::atomic<uint64_t> quick_scan_failures = {0};
    std::atomic<uint64_t> buffer_full_events  = {0};
};

struct agent_state
{
    rocprofiler_agent_id_t                            id                  = {};
    uint64_t                                          gpu_index           = 0;
    std::string                                       name                = {};
    rocprofiler_context_id_t                          context             = {};
    std::vector<rocprofiler_thread_trace_parameter_t> parameters          = {};
    rocprofiler_user_data_t                           userdata            = {};
    void*                                             backend             = nullptr;
    uint64_t                                          consecutive_kernels = 0;

    std::shared_mutex mutex     = {};
    bool              started   = false;
    bool              finalized = false;

    std::vector<kernel_symbol_record>                                      kernel_symbols   = {};
    std::unordered_map<entry_key, rocprofiler_kernel_id_t, entry_key_hash> kernels_by_entry = {};
    std::unordered_map<uint64_t, std::vector<kernel_symbol_range>> kernel_ranges_by_code_object =
        {};
    std::unordered_map<uint64_t, code_object_record> code_objects = {};
    agent_stats                                      stats        = {};
};

bool
backend_supported();

void*
backend_create(agent_state& state);

void
backend_destroy(agent_state& state);

void
backend_code_object_load(agent_state&                                                state,
                         const rocprofiler_callback_tracing_code_object_load_data_t& data);

void
backend_shader_data(agent_state&                           state,
                    rocprofiler_thread_trace_shader_data_t shader_data,
                    shader_data_forwarder_t                shader_data_forwarder,
                    kernel_target_filter_t                 kernel_target_filter);
}  // namespace att_no_intercept
}  // namespace tool
}  // namespace rocprofiler
