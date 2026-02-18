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

#if ROCPROFSYS_USE_ROCM > 0
#    include <rocprofiler-sdk/version.h>
#endif

#include <vector>

namespace rocprofsys
{
namespace trace_cache
{

template <typename T>
struct processor_t
{
    void handle(const kernel_dispatch_sample& sample)
    {
        static_cast<T*>(this)->handle(sample);
    }

    void handle(const scratch_memory_sample& sample)
    {
        static_cast<T*>(this)->handle(sample);
    }

    void handle(const memory_copy_sample& sample)
    {
        static_cast<T*>(this)->handle(sample);
    }

#if(ROCPROFILER_VERSION >= 600)
    void handle(const memory_allocate_sample& sample)
    {
        static_cast<T*>(this)->handle(sample);
    }
#endif

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

    void prepare_for_processing() { static_cast<T*>(this)->prepare_for_processing(); }

    void finalize_processing() { static_cast<T*>(this)->finalize_processing(); }

protected:
    ~processor_t() = default;
};

struct processor_view_t
{
    using kernel_dispatch_fn_t = void (*)(void*, const kernel_dispatch_sample&) noexcept;
    using scratch_memory_fn_t  = void (*)(void*, const scratch_memory_sample&) noexcept;
    using memory_copy_fn_t     = void (*)(void*, const memory_copy_sample&) noexcept;
#if(ROCPROFILER_VERSION >= 600)
    using memory_allocate_fn_t = void (*)(void*, const memory_allocate_sample&) noexcept;
#endif
    using region_fn_t           = void (*)(void*, const region_sample&) noexcept;
    using in_time_sample_fn_t   = void (*)(void*, const in_time_sample&) noexcept;
    using pmc_event_fn_t        = void (*)(void*, const pmc_event_with_sample&) noexcept;
    using amd_smi_sample_fn_t   = void (*)(void*, const amd_smi_sample&) noexcept;
    using cpu_freq_sample_fn_t  = void (*)(void*, const cpu_freq_sample&) noexcept;
    using backtrace_region_fn_t = void (*)(void*,
                                           const backtrace_region_sample&) noexcept;
    using prepare_for_processing_fn_t = void (*)(void*) noexcept;
    using finalize_processing_fn_t    = void (*)(void*) noexcept;

    struct vtable_t
    {
        kernel_dispatch_fn_t handle_kernel_dispatch;
        scratch_memory_fn_t  handle_scratch_memory;
        memory_copy_fn_t     handle_memory_copy;
#if(ROCPROFILER_VERSION >= 600)
        memory_allocate_fn_t handle_memory_allocate;
#endif
        region_fn_t                 handle_region;
        in_time_sample_fn_t         handle_in_time_sample;
        pmc_event_fn_t              handle_pmc_event;
        amd_smi_sample_fn_t         handle_amd_smi_sample;
        cpu_freq_sample_fn_t        handle_cpu_freq_sample;
        backtrace_region_fn_t       handle_backtrace_region;
        prepare_for_processing_fn_t prepare_for_processing;
        finalize_processing_fn_t    finalize_processing;
    };

    template <typename T>
    explicit processor_view_t(T& t) noexcept
    : m_object{ std::addressof(t) }
    , m_vtable{ std::addressof(get_vtable_for_type<T>()) }
    {
        static_assert(std::is_base_of<processor_t<T>, T>::value,
                      "Type must be derived from processor_t<T>");
    }

    processor_view_t(const processor_view_t&) noexcept            = default;
    processor_view_t(processor_view_t&&) noexcept                 = default;
    processor_view_t& operator=(const processor_view_t&) noexcept = default;
    processor_view_t& operator=(processor_view_t&&) noexcept      = default;

    ROCPROFSYS_INLINE void handle(const kernel_dispatch_sample& sample) const noexcept
    {
        m_vtable->handle_kernel_dispatch(m_object, sample);
    }

    ROCPROFSYS_INLINE void handle(const scratch_memory_sample& sample) const noexcept
    {
        m_vtable->handle_scratch_memory(m_object, sample);
    }

    ROCPROFSYS_INLINE void handle(const memory_copy_sample& sample) const noexcept
    {
        m_vtable->handle_memory_copy(m_object, sample);
    }

#if(ROCPROFILER_VERSION >= 600)
    ROCPROFSYS_INLINE void handle(const memory_allocate_sample& sample) const noexcept
    {
        m_vtable->handle_memory_allocate(m_object, sample);
    }
#endif

    ROCPROFSYS_INLINE void handle(const region_sample& sample) const noexcept
    {
        m_vtable->handle_region(m_object, sample);
    }

    ROCPROFSYS_INLINE void handle(const in_time_sample& sample) const noexcept
    {
        m_vtable->handle_in_time_sample(m_object, sample);
    }

    ROCPROFSYS_INLINE void handle(const pmc_event_with_sample& sample) const noexcept
    {
        m_vtable->handle_pmc_event(m_object, sample);
    }

    ROCPROFSYS_INLINE void handle(const amd_smi_sample& sample) const noexcept
    {
        m_vtable->handle_amd_smi_sample(m_object, sample);
    }

    ROCPROFSYS_INLINE void handle(const cpu_freq_sample& sample) const noexcept
    {
        m_vtable->handle_cpu_freq_sample(m_object, sample);
    }

    ROCPROFSYS_INLINE void handle(const backtrace_region_sample& sample) const noexcept
    {
        m_vtable->handle_backtrace_region(m_object, sample);
    }

    ROCPROFSYS_INLINE void prepare_for_processing() const noexcept
    {
        m_vtable->prepare_for_processing(m_object);
    }

    ROCPROFSYS_INLINE void finalize_processing() const noexcept
    {
        m_vtable->finalize_processing(m_object);
    }

private:
    template <typename T>
    static inline const vtable_t& get_vtable_for_type() noexcept
    {
        static const vtable_t vtable{
            +[](void* obj, const kernel_dispatch_sample& sample) noexcept {
                static_cast<T*>(obj)->handle(sample);
            },
            +[](void* obj, const scratch_memory_sample& sample) noexcept {
                static_cast<T*>(obj)->handle(sample);
            },
            +[](void* obj, const memory_copy_sample& sample) noexcept {
                static_cast<T*>(obj)->handle(sample);
            },
#if(ROCPROFILER_VERSION >= 600)
            +[](void* obj, const memory_allocate_sample& sample) noexcept {
                static_cast<T*>(obj)->handle(sample);
            },
#endif
            +[](void* obj, const region_sample& sample) noexcept {
                static_cast<T*>(obj)->handle(sample);
            },
            +[](void* obj, const in_time_sample& sample) noexcept {
                static_cast<T*>(obj)->handle(sample);
            },
            +[](void* obj, const pmc_event_with_sample& sample) noexcept {
                static_cast<T*>(obj)->handle(sample);
            },
            +[](void* obj, const amd_smi_sample& sample) noexcept {
                static_cast<T*>(obj)->handle(sample);
            },
            +[](void* obj, const cpu_freq_sample& sample) noexcept {
                static_cast<T*>(obj)->handle(sample);
            },
            +[](void* obj, const backtrace_region_sample& sample) noexcept {
                static_cast<T*>(obj)->handle(sample);
            },
            +[](void* obj) noexcept { static_cast<T*>(obj)->prepare_for_processing(); },
            +[](void* obj) noexcept { static_cast<T*>(obj)->finalize_processing(); }
        };
        return vtable;
    }

    void*           m_object;
    const vtable_t* m_vtable;
};

struct sample_processor_t
{
    void clear_handlers() { m_processor_view_list.clear(); }

    template <typename T>
    void add_handler(T& handler)
    {
        m_processor_view_list.emplace_back(handler);
    }

    template <typename SampleType>
    ROCPROFSYS_INLINE void handle_sample(const SampleType& sample) const
    {
        for(const auto& view : m_processor_view_list)
            view.handle(sample);
    }

    ROCPROFSYS_INLINE void prepare_for_processing() const noexcept
    {
        for(const auto& view : m_processor_view_list)
            view.prepare_for_processing();
    }

    ROCPROFSYS_INLINE void finalize_processing() const noexcept
    {
        for(const auto& view : m_processor_view_list)
            view.finalize_processing();
    }

    ROCPROFSYS_INLINE bool is_empty() const noexcept
    {
        return m_processor_view_list.empty();
    }

    ROCPROFSYS_INLINE void execute_sample_processing(
        type_identifier_t type_identifier, const trace_cache::cacheable_t& sample) const
    {
        switch(type_identifier)
        {
            case type_identifier_t::region:
                handle_sample(static_cast<const region_sample&>(sample));
                break;
            case type_identifier_t::kernel_dispatch:
                handle_sample(static_cast<const kernel_dispatch_sample&>(sample));
                break;
            case type_identifier_t::scratch_memory:
                handle_sample(static_cast<const scratch_memory_sample&>(sample));
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
    std::vector<processor_view_t> m_processor_view_list;
};

}  // namespace trace_cache
}  // namespace rocprofsys
