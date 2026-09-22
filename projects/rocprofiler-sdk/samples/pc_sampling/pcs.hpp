// MIT License
//
// Copyright (c) 2023-2025 ROCm Developer Tools
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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <memory>
#include <vector>

namespace client
{
namespace pcs
{
constexpr size_t BUFFER_SIZE_BYTES = 8192;
constexpr size_t WATERMARK         = (BUFFER_SIZE_BYTES / 4);

struct tool_agent_info;
using avail_configs_vec_t         = std::vector<rocprofiler_pc_sampling_configuration_t>;
using tool_agent_info_vec_t       = std::vector<std::unique_ptr<tool_agent_info>>;
using pc_sampling_buffer_id_vec_t = std::vector<rocprofiler_buffer_id_t>;

struct tool_agent_info
{
    rocprofiler_agent_id_t               agent_id;
    std::unique_ptr<avail_configs_vec_t> avail_configs;
    const rocprofiler_agent_t*           agent;
};

// GPU agents supporting some kind of PC sampling.
// Note that for some of these agent, the corresponding context might be invalid,
// meaning we were not able to enable PC sampling service.
// Check the `tool_init` for more information.
extern tool_agent_info_vec_t gpu_agents;

// Must be called first (prior to any other function from this namespace)
void
init();

// Must be called at the end of the `tool_fini`
void
fini();

// Ids of the buffers used as containers for PC sampling records
pc_sampling_buffer_id_vec_t*
get_pc_sampling_buffer_ids();

void
find_all_gpu_agents_supporting_pc_sampling();

/**
 * @brief The return value indicates if the agent supports PC sampling.
 * Check the implementation for more info.
 */
bool
query_avail_configs_for_agent(tool_agent_info* agent_info);

void
configure_pc_sampling_prefer_stochastic(tool_agent_info*         agent_info,
                                        rocprofiler_context_id_t context_id,
                                        rocprofiler_buffer_id_t  buffer_id);

void
rocprofiler_pc_sampling_callback(rocprofiler_context_id_t      context_id,
                                 rocprofiler_buffer_id_t       buffer_id,
                                 rocprofiler_record_header_t** headers,
                                 size_t                        num_headers,
                                 void*                         data,
                                 uint64_t                      drop_count);
}  // namespace pcs
}  // namespace client
