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
#include "core/trace_cache/sample_type.hpp"
#include <memory>

namespace rocprofsys::trace_cache
{

struct post_processing_t
{
    virtual void post_process_metadata()                     = 0;
    virtual void handle(const kernel_dispatch_sample& value) = 0;
    virtual void handle(const memory_copy_sample& value)     = 0;
#if(ROCPROFILER_VERSION >= 600)
    virtual void handle(const memory_allocate_sample& value) = 0;
#endif
    virtual void handle(const region_sample& value)           = 0;
    virtual void handle(const in_time_sample& value)          = 0;
    virtual void handle(const pmc_event_with_sample& value)   = 0;
    virtual void handle(const amd_smi_sample& value)          = 0;
    virtual void handle(const cpu_freq_sample& value)         = 0;
    virtual void handle(const backtrace_region_sample& value) = 0;
};

struct sample_processor_t
{
    void add_post_processing(const std::shared_ptr<post_processing_t>& pp);
    void clear_post_processing();
    void post_process_metadata();
    void execute_sample_processing(type_identifier_t type, const cacheable_t& value);

private:
    std::vector<std::shared_ptr<post_processing_t>> m_post_processing_handles;
};

}  // namespace rocprofsys::trace_cache
