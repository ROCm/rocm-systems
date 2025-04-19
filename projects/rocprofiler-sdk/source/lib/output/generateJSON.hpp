// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "agent_info.hpp"
#include "buffered_output.hpp"
#include "metadata.hpp"
#include "output_config.hpp"
#include "output_stream.hpp"
#include "statistics.hpp"

#include <cstdint>
#include <deque>

namespace rocprofiler
{
namespace tool
{
using JSONOutputArchive = ::cereal::MinimalJSONOutputArchive;

struct json_output
{
    json_output(const output_config&       cfg,
                std::string_view           filename,
                JSONOutputArchive::Options _opts);
    ~json_output();

    json_output(const json_output&)     = delete;
    json_output(json_output&&) noexcept = default;
    json_output& operator=(const json_output&) = delete;
    json_output& operator=(json_output&&) noexcept = default;

    template <typename... Args>
    decltype(auto) operator()(Args&&... args)
    {
        return (*archive)(std::forward<Args>(args)...);
    }

    void           startNode() { archive->startNode(); }
    void           finishNode() { archive->finishNode(); }
    void           makeArray() { archive->makeArray(); }
    decltype(auto) setNextName(const char* name) { archive->setNextName(name); }

    void start_process();
    void finish_process();

    void close();

private:
    output_stream                      stream  = {};
    std::unique_ptr<JSONOutputArchive> archive = {};
};

json_output
open_json(const output_config& cfg);

void
close_json(json_output& ar);

void
write_json(json_output&, const output_config& cfg, const metadata& tool_metadata, uint64_t pid);

void
write_json(
    json_output&                                                            json_ar,
    const output_config&                                                    cfg,
    const metadata&                                                         tool_metadata,
    const domain_stats_vec_t&                                               domain_stats,
    const generator<tool_buffer_tracing_hip_api_ext_record_t>&              hip_api_gen,
    const generator<rocprofiler_buffer_tracing_hsa_api_record_t>&           hsa_api_gen,
    const generator<tool_buffer_tracing_kernel_dispatch_ext_record_t>&      kernel_dispatch_gen,
    const generator<tool_buffer_tracing_memory_copy_ext_record_t>&          memory_copy_gen,
    const generator<tool_counter_record_t>&                                 counter_collection_gen,
    const generator<rocprofiler_buffer_tracing_marker_api_record_t>&        marker_api_gen,
    const generator<rocprofiler_buffer_tracing_scratch_memory_record_t>&    scratch_memory_gen,
    const generator<rocprofiler_buffer_tracing_rccl_api_ext_record_t>&      rccl_api_ext_gen,
    const generator<tool_buffer_tracing_memory_allocation_ext_record_t>&    memory_allocation_gen,
    const generator<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t>& rocdecode_api_gen,
    const generator<rocprofiler_buffer_tracing_rocjpeg_api_record_t>&       rocjpeg_api_gen,
    const generator<rocprofiler_tool_pc_sampling_host_trap_record_t>&  pc_sampling_host_trap_gen,
    const generator<rocprofiler_tool_pc_sampling_stochastic_record_t>& pc_sampling_stochastic_gen);
}  // namespace tool
}  // namespace rocprofiler
