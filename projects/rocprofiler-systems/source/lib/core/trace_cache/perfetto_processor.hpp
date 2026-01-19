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
#include "config.hpp"
#include "core/perfetto_fwd.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_processor.hpp"

#include <functional>
#include <memory>
#include <perfetto.h>

namespace rocprofsys
{
namespace trace_cache
{
using char_vec_t = std::vector<char>;

struct pmc_track_info
{
    const char*                                                           default_units;
    std::function<bool(uint64_t)>                                         exists_fn;
    std::function<void(uint64_t, const std::string&, const std::string&)> emplace_fn;
    std::function<void(uint64_t, uint64_t, uint64_t, double)>             trace_fn;
};

class perfetto_processor_t : public processor_t<perfetto_processor_t>
{
public:
    perfetto_processor_t(const std::shared_ptr<metadata_registry>& metadata,
                         const std::shared_ptr<agent_manager>& agent_mngr, int pid,
                         int ppid);

    void prepare_for_processing();
    void finalize_processing();

    void handle(const kernel_dispatch_sample& sample);
    void handle(const scratch_memory_sample& sample);
    void handle(const memory_copy_sample& sample);
    void handle(const memory_allocate_sample& sample);
    void handle(const region_sample& sample);
    void handle(const in_time_sample& sample);
    void handle(const pmc_event_with_sample& sample);
    void handle(const amd_smi_sample& sample);
    void handle(const cpu_freq_sample& sample);
    void handle(const backtrace_region_sample& sample);

private:
    void       initialize_perfetto();
    void       setup_perfetto();
    void       start_session();
    void       stop_session();
    void       flush(bool& perfetto_output_error);
    char_vec_t get_session_data();

    metadata_registry&                          m_metadata;
    uint64_t                                    m_process_id;
    uint64_t                                    m_parrent_pid;
    agent_manager&                              m_agent_manager;
    ::perfetto::TraceConfig                     m_session_config;
    std::shared_ptr<tmp_file>                   m_tmp_file{ nullptr };
    std::unique_ptr<::perfetto::TracingSession> m_tracing_session{ nullptr };
    bool                                        m_use_annotations{ false };

    std::unordered_map<size_t, pmc_track_info> m_pmc_track_map;
};
}  // namespace trace_cache
}  // namespace rocprofsys
