// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include "agent_manager.hpp"
#include "core/perfetto_fwd.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/storage_parser.hpp"

#include <memory>
#include <perfetto.h>

namespace rocprofsys
{
namespace trace_cache
{
class perfetto_post_processing
{
public:
    perfetto_post_processing(metadata_registry& metadata, const uint64_t& pid,
                             agent_manager& agent_mngr);

    ~perfetto_post_processing();

    void register_parser_callback(storage_parser& parser);

    void setup_perfetto();

    void start_session();
    void stop_session();
    void post_process(bool& _perfetto_output_error);

private:
    // void post_process_metadata();

    postprocessing_callback get_kernel_dispatch_callback() const;
    postprocessing_callback get_memory_copy_callback() const;
#if(ROCPROFILER_VERSION >= 600)
    postprocessing_callback get_memory_allocate_callback() const;
#endif
    postprocessing_callback get_region_callback() const;
    postprocessing_callback get_cpu_freq_sample_callback() const;
    postprocessing_callback get_backtrace_sample_callback() const;
    postprocessing_callback get_pmc_event_with_sample_callback() const;
    postprocessing_callback get_amd_smi_sample_callback() const;
    postprocessing_callback get_in_time_sample_callback() const;

    metadata_registry&                          m_metadata;
    uint64_t                                    m_process_id;
    agent_manager&                              m_agent_manager;
    ::perfetto::TraceConfig                     m_session_config;
    std::unique_ptr<::perfetto::TracingSession> m_tracing_session{ nullptr };
};
}  // namespace trace_cache
}  // namespace rocprofsys
