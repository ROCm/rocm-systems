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

#include "generator.hpp"
#include "metadata.hpp"
#include "statistics.hpp"
#include "stream_info.hpp"

namespace rocprofiler
{
namespace tool
{
stats_entry_t
generate_stats(const output_config&                                               cfg,
               const metadata&                                                    tool_metadata,
               const generator<tool_buffer_tracing_kernel_dispatch_ext_record_t>& data);

stats_entry_t
generate_stats(const output_config&                                       cfg,
               const metadata&                                            tool_metadata,
               const generator<tool_buffer_tracing_hip_api_ext_record_t>& data);

stats_entry_t
generate_stats(const output_config&                                          cfg,
               const metadata&                                               tool_metadata,
               const generator<rocprofiler_buffer_tracing_hsa_api_record_t>& data);

stats_entry_t
generate_stats(const output_config&                                           cfg,
               const metadata&                                                tool_metadata,
               const generator<tool_buffer_tracing_memory_copy_ext_record_t>& data);

stats_entry_t
generate_stats(const output_config&                                             cfg,
               const metadata&                                                  tool_metadata,
               const generator<rocprofiler_buffer_tracing_marker_api_record_t>& data);

stats_entry_t
generate_stats(const output_config&                    cfg,
               const metadata&                         tool_metadata,
               const generator<tool_counter_record_t>& data);

stats_entry_t
generate_stats(const output_config&                                                 cfg,
               const metadata&                                                      tool_metadata,
               const generator<rocprofiler_buffer_tracing_scratch_memory_record_t>& data);

stats_entry_t
generate_stats(const output_config&                                               cfg,
               const metadata&                                                    tool_metadata,
               const generator<rocprofiler_buffer_tracing_rccl_api_ext_record_t>& data);

stats_entry_t
generate_stats(const output_config&                                                 cfg,
               const metadata&                                                      tool_metadata,
               const generator<tool_buffer_tracing_memory_allocation_ext_record_t>& data);

stats_entry_t
generate_stats(const output_config& cfg,
               const metadata&      tool_metadata,
               const generator<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t>& data);

stats_entry_t
generate_stats(const output_config&                                              cfg,
               const metadata&                                                   tool_metadata,
               const generator<rocprofiler_buffer_tracing_rocjpeg_api_record_t>& data);

stats_entry_t
generate_stats(const output_config&                                              cfg,
               const metadata&                                                   tool_metadata,
               const generator<rocprofiler_tool_pc_sampling_host_trap_record_t>& data);

stats_entry_t
generate_stats(const output_config&                                               cfg,
               const metadata&                                                    tool_metadata,
               const generator<rocprofiler_tool_pc_sampling_stochastic_record_t>& data);

void
generate_stats(const output_config&      cfg,
               const metadata&           tool_metadata,
               const domain_stats_vec_t& data);
}  // namespace tool
}  // namespace rocprofiler
