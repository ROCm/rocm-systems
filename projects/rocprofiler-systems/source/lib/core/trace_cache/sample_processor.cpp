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

#include "sample_processor.hpp"

namespace rocprofsys::trace_cache
{
void
sample_processor_t::execute_sample_processing(type_identifier_t  type,
                                              const cacheable_t& value)
{
    for(auto& handle : m_post_processing_handles)
    {
        switch(type)
        {
            case type_identifier_t::in_time_sample:
            {
                const auto& casted_value = static_cast<const in_time_sample&>(value);
                handle->handle(casted_value);
                break;
            }
            case type_identifier_t::pmc_event_with_sample:
            {
                const auto& casted_value =
                    static_cast<const pmc_event_with_sample&>(value);
                handle->handle(casted_value);
                break;
            }
            case type_identifier_t::region:
            {
                const auto& casted_value = static_cast<const region_sample&>(value);
                handle->handle(casted_value);
                break;
            }
            case type_identifier_t::kernel_dispatch:
            {
                const auto& casted_value =
                    static_cast<const kernel_dispatch_sample&>(value);
                handle->handle(casted_value);
                break;
            }
            case type_identifier_t::memory_copy:
            {
                const auto& casted_value = static_cast<const memory_copy_sample&>(value);
                handle->handle(casted_value);
                break;
            }
#if(ROCPROFILER_VERSION >= 600)
            case type_identifier_t::memory_alloc:
            {
                const auto& casted_value =
                    static_cast<const memory_allocate_sample&>(value);
                handle->handle(casted_value);
                break;
            }
#endif
            case type_identifier_t::amd_smi_sample:
            {
                const auto& casted_value = static_cast<const amd_smi_sample&>(value);
                handle->handle(casted_value);
                break;
            }
            case type_identifier_t::cpu_freq_sample:
            {
                const auto& casted_value = static_cast<const cpu_freq_sample&>(value);
                handle->handle(casted_value);
                break;
            }
            case type_identifier_t::backtrace_region_sample:
            {
                const auto& casted_value =
                    static_cast<const backtrace_region_sample&>(value);
                handle->handle(casted_value);
                break;
            }
            case type_identifier_t::fragmented_space: break;
        }
    }
}

void
sample_processor_t::add_post_processing(const std::shared_ptr<post_processing_t>& pp)
{
    m_post_processing_handles.push_back(pp);
}

void
sample_processor_t::clear_post_processing()
{
    m_post_processing_handles.clear();
}

void
sample_processor_t::post_process_metadata()
{
    for(auto& handle : m_post_processing_handles)
    {
        handle->post_process_metadata();
    }
}
}  // namespace rocprofsys::trace_cache
