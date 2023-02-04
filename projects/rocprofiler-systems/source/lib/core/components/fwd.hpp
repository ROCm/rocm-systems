// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "core/categories.hpp"
#include "core/common.hpp"
#include "core/defines.hpp"

#include <timemory/api.hpp>
#include <timemory/api/macros.hpp>
#include <timemory/components/base/types.hpp>
#include <timemory/components/data_tracker/types.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/components/user_bundle/types.hpp>
#include <timemory/enum.h>
#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/mpl/types.hpp>

#include <type_traits>

OMNITRACE_DECLARE_COMPONENT(roctracer)
OMNITRACE_DECLARE_COMPONENT(rocprofiler)
OMNITRACE_DECLARE_COMPONENT(rcclp_handle)
OMNITRACE_DECLARE_COMPONENT(comm_data)

OMNITRACE_COMPONENT_ALIAS(comm_data_tracker_t,
                          ::tim::component::data_tracker<float, project::omnitrace>)

namespace omnitrace
{
namespace policy = ::tim::policy;     // NOLINT
namespace comp   = ::tim::component;  // NOLINT

namespace component
{
template <typename Tp, typename ValueT>
using base = ::tim::component::base<Tp, ValueT>;

template <typename... Tp>
using data_tracker = tim::component::data_tracker<Tp...>;

template <typename... Tp>
using functor_t = std::function<void(Tp...)>;

using default_functor_t = functor_t<const char*>;

struct backtrace;
struct backtrace_metrics;
struct backtrace_timestamp;
struct backtrace_wall_clock
{};
struct backtrace_cpu_clock
{};
struct backtrace_fraction
{};
struct backtrace_gpu_busy
{};
struct backtrace_gpu_temp
{};
struct backtrace_gpu_power
{};
struct backtrace_gpu_memory
{};
using sampling_wall_clock = data_tracker<double, backtrace_wall_clock>;
using sampling_cpu_clock  = data_tracker<double, backtrace_cpu_clock>;
using sampling_percent    = data_tracker<double, backtrace_fraction>;
using sampling_gpu_busy   = data_tracker<double, backtrace_gpu_busy>;
using sampling_gpu_temp   = data_tracker<double, backtrace_gpu_temp>;
using sampling_gpu_power  = data_tracker<double, backtrace_gpu_power>;
using sampling_gpu_memory = data_tracker<double, backtrace_gpu_memory>;

template <typename ApiT, typename StartFuncT = default_functor_t,
          typename StopFuncT = default_functor_t>
struct functors;
}  // namespace component
}  // namespace omnitrace

#if !defined(OMNITRACE_USE_ROCTRACER)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, component::roctracer, false_type)
#endif

#if !defined(OMNITRACE_USE_ROCPROFILER)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, component::rocprofiler, false_type)
#endif

#if !defined(OMNITRACE_USE_RCCL)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, category::rocm_rccl, false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, component::rcclp_handle, false_type)
#endif

#if !defined(OMNITRACE_USE_RCCL) && !defined(OMNITRACE_USE_MPI)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, component::comm_data_tracker_t, false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, component::comm_data, false_type)
#endif

#if !defined(TIMEMORY_USE_LIBUNWIND)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, category::sampling, false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, component::backtrace, false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, component::backtrace_metrics, false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, component::backtrace_timestamp, false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_wall_clock, false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_cpu_clock, false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_percent, false_type)
#endif

#if !defined(TIMEMORY_USE_LIBUNWIND) || !defined(OMNITRACE_USE_ROCM_SMI)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_gpu_busy, false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_gpu_temp, false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_gpu_power, false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_gpu_memory, false_type)
#endif

TIMEMORY_SET_COMPONENT_API(omnitrace::component::roctracer, project::omnitrace,
                           tpls::rocm, device::gpu, os::supports_linux,
                           category::external)
TIMEMORY_SET_COMPONENT_API(omnitrace::component::rocprofiler, project::omnitrace,
                           tpls::rocm, device::gpu, os::supports_linux,
                           category::external, category::hardware_counter)
TIMEMORY_SET_COMPONENT_API(omnitrace::component::sampling_wall_clock, project::omnitrace,
                           category::timing, os::supports_unix, category::sampling,
                           category::interrupt_sampling)
TIMEMORY_SET_COMPONENT_API(omnitrace::component::sampling_cpu_clock, project::omnitrace,
                           category::timing, os::supports_unix, category::sampling,
                           category::interrupt_sampling)
TIMEMORY_SET_COMPONENT_API(omnitrace::component::sampling_percent, project::omnitrace,
                           category::timing, os::supports_unix, category::sampling,
                           category::interrupt_sampling)
TIMEMORY_SET_COMPONENT_API(omnitrace::component::sampling_gpu_busy, project::omnitrace,
                           tpls::rocm, device::gpu, os::supports_linux,
                           category::sampling, category::process_sampling)
TIMEMORY_SET_COMPONENT_API(omnitrace::component::sampling_gpu_memory, project::omnitrace,
                           tpls::rocm, device::gpu, os::supports_linux, category::memory,
                           category::sampling, category::process_sampling)
TIMEMORY_SET_COMPONENT_API(omnitrace::component::sampling_gpu_power, project::omnitrace,
                           tpls::rocm, device::gpu, os::supports_linux, category::power,
                           category::sampling, category::process_sampling)
TIMEMORY_SET_COMPONENT_API(omnitrace::component::sampling_gpu_temp, project::omnitrace,
                           tpls::rocm, device::gpu, os::supports_linux,
                           category::temperature, category::sampling,
                           category::process_sampling)

TIMEMORY_PROPERTY_SPECIALIZATION(omnitrace::component::roctracer, OMNITRACE_ROCTRACER,
                                 "roctracer", "omnitrace_roctracer")
TIMEMORY_PROPERTY_SPECIALIZATION(omnitrace::component::rocprofiler, OMNITRACE_ROCPROFILER,
                                 "rocprofiler", "omnitrace_rocprofiler")
TIMEMORY_PROPERTY_SPECIALIZATION(omnitrace::component::sampling_wall_clock,
                                 OMNITRACE_SAMPLING_WALL_CLOCK, "sampling_wall_clock", "")
TIMEMORY_PROPERTY_SPECIALIZATION(omnitrace::component::sampling_cpu_clock,
                                 OMNITRACE_SAMPLING_CPU_CLOCK, "sampling_cpu_clock", "")
TIMEMORY_PROPERTY_SPECIALIZATION(omnitrace::component::sampling_percent,
                                 OMNITRACE_SAMPLING_PERCENT, "sampling_percent", "")
TIMEMORY_PROPERTY_SPECIALIZATION(omnitrace::component::sampling_gpu_busy,
                                 OMNITRACE_SAMPLING_GPU_BUSY, "sampling_gpu_busy",
                                 "sampling_gpu_util")
TIMEMORY_PROPERTY_SPECIALIZATION(omnitrace::component::sampling_gpu_memory,
                                 OMNITRACE_SAMPLING_GPU_MEMORY_USAGE,
                                 "sampling_gpu_memory_usage", "")
TIMEMORY_PROPERTY_SPECIALIZATION(omnitrace::component::sampling_gpu_power,
                                 OMNITRACE_SAMPLING_GPU_POWER, "sampling_gpu_power", "")
TIMEMORY_PROPERTY_SPECIALIZATION(omnitrace::component::sampling_gpu_temp,
                                 OMNITRACE_SAMPLING_GPU_TEMP, "sampling_gpu_temp", "")

TIMEMORY_METADATA_SPECIALIZATION(omnitrace::component::roctracer, "roctracer",
                                 "High-precision ROCm API and kernel tracing", "")
TIMEMORY_METADATA_SPECIALIZATION(omnitrace::component::rocprofiler, "rocprofiler",
                                 "ROCm kernel hardware counters", "")
TIMEMORY_METADATA_SPECIALIZATION(omnitrace::component::sampling_wall_clock,
                                 "sampling_wall_clock", "Wall-clock timing",
                                 "Derived from statistical sampling")
TIMEMORY_METADATA_SPECIALIZATION(omnitrace::component::sampling_cpu_clock,
                                 "sampling_cpu_clock", "CPU-clock timing",
                                 "Derived from statistical sampling")
TIMEMORY_METADATA_SPECIALIZATION(omnitrace::component::sampling_percent,
                                 "sampling_percent",
                                 "Fraction of wall-clock time spent in functions",
                                 "Derived from statistical sampling")
TIMEMORY_METADATA_SPECIALIZATION(omnitrace::component::sampling_gpu_busy,
                                 "sampling_gpu_busy",
                                 "GPU Utilization (% busy) via ROCm-SMI",
                                 "Derived from sampling")
TIMEMORY_METADATA_SPECIALIZATION(omnitrace::component::sampling_gpu_memory,
                                 "sampling_gpu_memory_usage",
                                 "GPU Memory Usage via ROCm-SMI", "Derived from sampling")
TIMEMORY_METADATA_SPECIALIZATION(omnitrace::component::sampling_gpu_power,
                                 "sampling_gpu_power", "GPU Power Usage via ROCm-SMI",
                                 "Derived from sampling")
TIMEMORY_METADATA_SPECIALIZATION(omnitrace::component::sampling_gpu_temp,
                                 "sampling_gpu_temp", "GPU Temperature via ROCm-SMI",
                                 "Derived from sampling")

// statistics type
TIMEMORY_STATISTICS_TYPE(omnitrace::component::sampling_wall_clock, double)
TIMEMORY_STATISTICS_TYPE(omnitrace::component::sampling_cpu_clock, double)
TIMEMORY_STATISTICS_TYPE(omnitrace::component::sampling_gpu_busy, double)
TIMEMORY_STATISTICS_TYPE(omnitrace::component::sampling_gpu_temp, double)
TIMEMORY_STATISTICS_TYPE(omnitrace::component::sampling_gpu_power, double)
TIMEMORY_STATISTICS_TYPE(omnitrace::component::sampling_gpu_memory, double)
TIMEMORY_STATISTICS_TYPE(omnitrace::component::comm_data_tracker_t, float)

// enable timing units
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_timing_category, component::sampling_wall_clock,
                                true_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_timing_category, component::sampling_cpu_clock,
                                true_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_timing_category, component::sampling_percent,
                                true_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::sampling_wall_clock,
                                true_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::sampling_cpu_clock,
                                true_type)

// enable percent units
OMNITRACE_DEFINE_CONCRETE_TRAIT(uses_percent_units, component::sampling_gpu_busy,
                                true_type)

// enable memory units
OMNITRACE_DEFINE_CONCRETE_TRAIT(is_memory_category, component::sampling_gpu_memory,
                                true_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::sampling_gpu_memory,
                                true_type)

// reporting categories (sum)
OMNITRACE_DEFINE_CONCRETE_TRAIT(report_sum, component::sampling_gpu_busy, false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(report_sum, component::sampling_gpu_temp, false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(report_sum, component::sampling_gpu_power, false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(report_sum, component::sampling_gpu_memory, false_type)

// reporting categories (mean)
OMNITRACE_DEFINE_CONCRETE_TRAIT(report_mean, component::sampling_percent, false_type)

// reporting categories (stats)
OMNITRACE_DEFINE_CONCRETE_TRAIT(report_statistics, component::sampling_percent,
                                false_type)

#define OMNITRACE_DECLARE_EXTERN_COMPONENT(NAME, HAS_DATA, ...)                          \
    TIMEMORY_DECLARE_EXTERN_TEMPLATE(                                                    \
        struct tim::component::base<TIMEMORY_ESC(omnitrace::component::NAME),            \
                                    __VA_ARGS__>)                                        \
    TIMEMORY_DECLARE_EXTERN_OPERATIONS(TIMEMORY_ESC(omnitrace::component::NAME),         \
                                       HAS_DATA)                                         \
    TIMEMORY_DECLARE_EXTERN_STORAGE(TIMEMORY_ESC(omnitrace::component::NAME))

#define OMNITRACE_INSTANTIATE_EXTERN_COMPONENT(NAME, HAS_DATA, ...)                      \
    TIMEMORY_INSTANTIATE_EXTERN_TEMPLATE(                                                \
        struct tim::component::base<TIMEMORY_ESC(omnitrace::component::NAME),            \
                                    __VA_ARGS__>)                                        \
    TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(TIMEMORY_ESC(omnitrace::component::NAME),     \
                                           HAS_DATA)                                     \
    TIMEMORY_INSTANTIATE_EXTERN_STORAGE(TIMEMORY_ESC(omnitrace::component::NAME))
