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

#if !defined(ROCPROFILER_ATT_QUICK_SCAN_ENABLED) || ROCPROFILER_ATT_QUICK_SCAN_ENABLED == 0
typedef uint64_t rocprof_trace_decoder_handle_t;
#else
#    include <rocprof_trace_decoder/rocprof_trace_decoder.h>
#endif

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocprofiler
{
namespace tool
{
namespace att_no_intercept
{
struct entry_key_t
{
    uint64_t code_object_id = 0;
    uint64_t address        = 0;

    friend bool operator==(const entry_key_t& lhs, const entry_key_t& rhs)
    {
        return lhs.code_object_id == rhs.code_object_id && lhs.address == rhs.address;
    }
};

struct entry_key_hash_t
{
    size_t operator()(const entry_key_t& key) const
    {
        return static_cast<size_t>((key.address << 5) ^ key.code_object_id);
    }
};

struct code_object_record_t
{
    int64_t load_delta = 0;

    std::unordered_map<uint64_t, uint64_t> symbol_sizes_by_vaddr = {};
};

struct kernel_symbol_range_t
{
    uint64_t begin    = 0;
    uint64_t end      = 0;
    bool     targeted = false;
};

struct trace_range_t
{
    bool     active               = false;
    uint8_t  me_id                = 0;
    uint8_t  pipe_id              = 0;
    uint64_t offset_begin         = 0;
    uint64_t offset_end           = 0;
    uint64_t remaining_dispatches = 0;
    uint64_t flush_count          = 0;
};

struct agent_state_t
{
    rocprofiler_agent_id_t         id                  = {};
    rocprofiler_context_id_t       context             = {};
    rocprofiler_user_data_t        userdata            = {};
    rocprof_trace_decoder_handle_t decoder             = {};
    uint64_t                       consecutive_kernels = 0;

    std::shared_mutex mutex     = {};
    bool              started   = false;
    bool              finalized = false;

    std::unordered_map<entry_key_t, std::shared_ptr<std::atomic<size_t>>, entry_key_hash_t>
        kernel_iterations_by_entry = {};
    std::unordered_map<uint64_t, std::vector<kernel_symbol_range_t>> kernel_ranges_by_code_object{};
    std::unordered_map<uint64_t, code_object_record_t>               code_objects{};

    std::atomic<bool>     chunk_failed{false};
    std::atomic<uint64_t> chunk_completed{0};
    std::atomic<uint64_t> pending_requests{0};

    std::mutex                                         request_mutex{};
    std::unordered_set<uint64_t>                       chunk_requested{};
    std::mutex                                         data_mutex{};
    std::unordered_map<uint64_t, std::vector<uint8_t>> chunk_data{0};
    std::condition_variable                            data_cv{};
};

bool
backend_supported();

void
backend_create(agent_state_t& state);

void
backend_destroy(agent_state_t& state);

void
backend_code_object_load(agent_state_t&                                              state,
                         const rocprofiler_callback_tracing_code_object_load_data_t& data);

void
backend_shader_data(agent_state_t&                         state,
                    rocprofiler_thread_trace_shader_data_t shader_data,
                    shader_data_forwarder_t                shader_data_forwarder,
                    const std::unordered_set<size_t>&      kernel_filter_range);
}  // namespace att_no_intercept
}  // namespace tool
}  // namespace rocprofiler
