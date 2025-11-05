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
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/sample_type.hpp"
#include <functional>
#include <vector>

namespace rocprofsys
{
namespace trace_cache
{

template <typename T>
struct post_processor_t
{
    void handle(const kernel_dispatch_sample& sample)
    {
        static_cast<T*>(this)->handle(sample);
    }

    void handle(const memory_copy_sample& sample)
    {
        static_cast<T*>(this)->handle(sample);
    }

    void handle(const memory_allocate_sample& sample)
    {
        static_cast<T*>(this)->handle(sample);
    }

    void handle(const region_sample& sample) { static_cast<T*>(this)->handle(sample); }

    void handle(const in_time_sample& sample) { static_cast<T*>(this)->handle(sample); }

    void handle(const pmc_event_with_sample& sample)
    {
        static_cast<T*>(this)->handle(sample);
    }

    void handle(const amd_smi_sample& sample) { static_cast<T*>(this)->handle(sample); }

    void handle(const cpu_freq_sample& sample) { static_cast<T*>(this)->handle(sample); }

    void handle(const backtrace_region_sample& sample)
    {
        static_cast<T*>(this)->handle(sample);
    }

protected:
    ~post_processor_t() = default;
};

struct post_processor_view_t
{
    template <typename T>
    explicit post_processor_view_t(T& t)
    : object{ &t }
    , handle_kernel_dispatch_impl{ [](void* obj, const kernel_dispatch_sample& sample) {
        static_cast<T*>(obj)->handle(sample);
    } }
    , handle_memory_copy_impl{ [](void* obj, const memory_copy_sample& sample) {
        static_cast<T*>(obj)->handle(sample);
    } }
    , handle_memory_allocate_impl{ [](void* obj, const memory_allocate_sample& sample) {
        static_cast<T*>(obj)->handle(sample);
    } }
    , handle_region_impl{ [](void* obj, const region_sample& sample) {
        static_cast<T*>(obj)->handle(sample);
    } }
    , handle_in_time_sample_impl{ [](void* obj, const in_time_sample& sample) {
        static_cast<T*>(obj)->handle(sample);
    } }
    , handle_pmc_event_with_sample_impl{ [](void*                        obj,
                                            const pmc_event_with_sample& sample) {
        static_cast<T*>(obj)->handle(sample);
    } }
    , handle_amd_smi_sample_impl{ [](void* obj, const amd_smi_sample& sample) {
        static_cast<T*>(obj)->handle(sample);
    } }
    , handle_cpu_freq_sample_impl{ [](void* obj, const cpu_freq_sample& sample) {
        static_cast<T*>(obj)->handle(sample);
    } }
    , handle_backtrace_region_sample_impl{ [](void*                          obj,
                                              const backtrace_region_sample& sample) {
        static_cast<T*>(obj)->handle(sample);
    } }
    {}

    void handle(const kernel_dispatch_sample& sample) const
    {
        handle_kernel_dispatch_impl(object, sample);
    }

    void handle(const memory_copy_sample& sample) const
    {
        handle_memory_copy_impl(object, sample);
    }

    void handle(const memory_allocate_sample& sample) const
    {
        handle_memory_allocate_impl(object, sample);
    }

    void handle(const region_sample& sample) const { handle_region_impl(object, sample); }

    void handle(const in_time_sample& sample) const
    {
        handle_in_time_sample_impl(object, sample);
    }

    void handle(const pmc_event_with_sample& sample) const
    {
        handle_pmc_event_with_sample_impl(object, sample);
    }

    void handle(const amd_smi_sample& sample) const
    {
        handle_amd_smi_sample_impl(object, sample);
    }

    void handle(const cpu_freq_sample& sample) const
    {
        handle_cpu_freq_sample_impl(object, sample);
    }

    void handle(const backtrace_region_sample& sample) const
    {
        handle_backtrace_region_sample_impl(object, sample);
    }

private:
    void* object;

    std::function<void(void*, const kernel_dispatch_sample&)> handle_kernel_dispatch_impl;
    std::function<void(void*, const memory_copy_sample&)>     handle_memory_copy_impl;
    std::function<void(void*, const memory_allocate_sample&)> handle_memory_allocate_impl;
    std::function<void(void*, const region_sample&)>          handle_region_impl;
    std::function<void(void*, const in_time_sample&)>         handle_in_time_sample_impl;
    std::function<void(void*, const pmc_event_with_sample&)>
                                                       handle_pmc_event_with_sample_impl;
    std::function<void(void*, const amd_smi_sample&)>  handle_amd_smi_sample_impl;
    std::function<void(void*, const cpu_freq_sample&)> handle_cpu_freq_sample_impl;
    std::function<void(void*, const backtrace_region_sample&)>
        handle_backtrace_region_sample_impl;
};

struct sample_processor_t
{
    void clear_handlers() { m_post_processor_view_list.clear(); }

    template <typename T>
    void add_handler(T& handler)
    {
        m_post_processor_view_list.emplace_back(handler);
    }

    template <typename SampleType>
    void handle_sample(const SampleType& sample) const
    {
        for(const auto& view : m_post_processor_view_list)
            view.handle(sample);
    }

    void execute_sample_processing(type_identifier_t               type_identifier,
                                   const trace_cache::cacheable_t& sample) const
    {
        switch(type_identifier)
        {
            case type_identifier_t::region:
                handle_sample(static_cast<const region_sample&>(sample));
                break;
            case type_identifier_t::kernel_dispatch:
                handle_sample(static_cast<const kernel_dispatch_sample&>(sample));
                break;
            case type_identifier_t::memory_copy:
                handle_sample(static_cast<const memory_copy_sample&>(sample));
                break;
#if ROCPROFILER_VERSION >= 600
            case type_identifier_t::memory_alloc:
                handle_sample(static_cast<const memory_allocate_sample&>(sample));
                break;
#endif
            case type_identifier_t::in_time_sample:
                handle_sample(static_cast<const in_time_sample&>(sample));
                break;
            case type_identifier_t::pmc_event_with_sample:
                handle_sample(static_cast<const pmc_event_with_sample&>(sample));
                break;
            case type_identifier_t::amd_smi_sample:
                handle_sample(static_cast<const amd_smi_sample&>(sample));
                break;
            case type_identifier_t::cpu_freq_sample:
                handle_sample(static_cast<const cpu_freq_sample&>(sample));
                break;
            case type_identifier_t::backtrace_region_sample:
                handle_sample(static_cast<const backtrace_region_sample&>(sample));
                break;
            default: throw std::runtime_error("Unsupported sample type");
        }
    }

private:
    std::vector<post_processor_view_t> m_post_processor_view_list;
};

}  // namespace trace_cache
}  // namespace rocprofsys
