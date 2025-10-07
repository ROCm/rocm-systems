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
#include "core/agent_manager.hpp"
#include "core/node_info.hpp"

#include "core/rocpd/data_processor.hpp"

#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_processor.hpp"
#include "core/trace_cache/sample_type.hpp"
#include <memory>

namespace rocprofsys
{
namespace trace_cache
{

class rocpd_post_processing : public post_processing_t
{
public:
    rocpd_post_processing(metadata_registry& metadata, agent_manager& agent_mngr,
                          const std::string& _database_tag);

    void post_process_metadata() override;

    std::shared_ptr<rocpd::data_processor> get_data_processor() const;

private:
    using primary_key = size_t;

    void handle(const kernel_dispatch_sample& value) override;
    void handle(const memory_copy_sample& value) override;
#if(ROCPROFILER_VERSION >= 600)
    void handle(const memory_allocate_sample& value) override;
#endif
    void handle(const region_sample& value) override;
    void handle(const in_time_sample& value) override;
    void handle(const pmc_event_with_sample& value) override;
    void handle(const amd_smi_sample& value) override;
    void handle(const cpu_freq_sample& value) override;
    void handle(const backtrace_region_sample& value) override;

    inline void rocpd_insert_thread_id(info::thread& t_info, const node_info& n_info,
                                       const info::process& process_info) const;

    metadata_registry&                     m_metadata;
    agent_manager&                         m_agent_manager;
    std::shared_ptr<rocpd::data_processor> m_data_processor;
};

}  // namespace trace_cache
}  // namespace rocprofsys
