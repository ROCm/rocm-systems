// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

ROCPROFSYS_DECLARE_COMPONENT(rcclp_handle)
ROCPROFSYS_DECLARE_COMPONENT(comm_data)

ROCPROFSYS_COMPONENT_ALIAS(comm_data_tracker_t,
                           ::tim::component::data_tracker<float, project::rocprofsys>)

namespace rocprofsys
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
struct backtrace_gpu_busy_gfx
{};
struct backtrace_gpu_busy_umc
{};
struct backtrace_gpu_busy_mm
{};
struct backtrace_gpu_temp
{};
struct backtrace_gpu_power
{};
struct backtrace_gpu_memory
{};
struct backtrace_gpu_vcn
{};
struct backtrace_gpu_jpeg
{};

using sampling_wall_clock   = data_tracker<double, backtrace_wall_clock>;
using sampling_cpu_clock    = data_tracker<double, backtrace_cpu_clock>;
using sampling_percent      = data_tracker<double, backtrace_fraction>;
using sampling_gpu_busy_gfx = data_tracker<double, backtrace_gpu_busy_gfx>;
using sampling_gpu_busy_umc = data_tracker<double, backtrace_gpu_busy_umc>;
using sampling_gpu_busy_mm  = data_tracker<double, backtrace_gpu_busy_mm>;
using sampling_gpu_temp     = data_tracker<double, backtrace_gpu_temp>;
using sampling_gpu_power    = data_tracker<double, backtrace_gpu_power>;
using sampling_gpu_memory   = data_tracker<double, backtrace_gpu_memory>;
using sampling_gpu_vcn      = data_tracker<double, backtrace_gpu_vcn>;
using sampling_gpu_jpeg     = data_tracker<double, backtrace_gpu_jpeg>;

template <typename ApiT, typename StartFuncT = default_functor_t,
          typename StopFuncT = default_functor_t>
struct functors;
}  // namespace component
}  // namespace rocprofsys

#if !defined(ROCPROFSYS_USE_RCCL)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, category::rocm_rccl, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::rcclp_handle, false_type)
#endif

#if !defined(ROCPROFSYS_USE_RCCL) && !defined(ROCPROFSYS_USE_MPI)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::comm_data_tracker_t, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::comm_data, false_type)
#endif

#if !defined(TIMEMORY_USE_LIBUNWIND)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, category::sampling, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::backtrace, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::backtrace_metrics, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::backtrace_timestamp, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_wall_clock, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_cpu_clock, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_percent, false_type)
#endif

#if !defined(TIMEMORY_USE_LIBUNWIND) || !defined(ROCPROFSYS_USE_ROCM)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_gpu_busy_gfx,
                                 false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_gpu_busy_umc,
                                 false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_gpu_busy_mm,
                                 false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_gpu_temp, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_gpu_power, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_gpu_memory, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_gpu_vcn, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::sampling_gpu_jpeg, false_type)
#endif

TIMEMORY_SET_COMPONENT_API(rocprofsys::component::sampling_wall_clock,
                           project::rocprofsys, category::timing, os::supports_unix,
                           category::sampling, category::interrupt_sampling)
TIMEMORY_SET_COMPONENT_API(rocprofsys::component::sampling_cpu_clock, project::rocprofsys,
                           category::timing, os::supports_unix, category::sampling,
                           category::interrupt_sampling)
TIMEMORY_SET_COMPONENT_API(rocprofsys::component::sampling_percent, project::rocprofsys,
                           category::timing, os::supports_unix, category::sampling,
                           category::interrupt_sampling)
TIMEMORY_SET_COMPONENT_API(rocprofsys::component::sampling_gpu_busy_gfx,
                           project::rocprofsys, tpls::rocm, device::gpu,
                           os::supports_linux, category::sampling,
                           category::process_sampling)
TIMEMORY_SET_COMPONENT_API(rocprofsys::component::sampling_gpu_busy_umc,
                           project::rocprofsys, tpls::rocm, device::gpu,
                           os::supports_linux, category::sampling,
                           category::process_sampling)
TIMEMORY_SET_COMPONENT_API(rocprofsys::component::sampling_gpu_busy_mm,
                           project::rocprofsys, tpls::rocm, device::gpu,
                           os::supports_linux, category::sampling,
                           category::process_sampling)
TIMEMORY_SET_COMPONENT_API(rocprofsys::component::sampling_gpu_memory,
                           project::rocprofsys, tpls::rocm, device::gpu,
                           os::supports_linux, category::memory, category::sampling,
                           category::process_sampling)
TIMEMORY_SET_COMPONENT_API(rocprofsys::component::sampling_gpu_power, project::rocprofsys,
                           tpls::rocm, device::gpu, os::supports_linux, category::power,
                           category::sampling, category::process_sampling)
TIMEMORY_SET_COMPONENT_API(rocprofsys::component::sampling_gpu_temp, project::rocprofsys,
                           tpls::rocm, device::gpu, os::supports_linux,
                           category::temperature, category::sampling,
                           category::process_sampling)
TIMEMORY_SET_COMPONENT_API(rocprofsys::component::sampling_gpu_vcn, project::rocprofsys,
                           tpls::rocm, device::gpu, os::supports_linux,
                           category::sampling, category::process_sampling)
TIMEMORY_SET_COMPONENT_API(rocprofsys::component::sampling_gpu_jpeg, project::rocprofsys,
                           tpls::rocm, device::gpu, os::supports_linux,
                           category::sampling, category::process_sampling)

TIMEMORY_METADATA_SPECIALIZATION(rocprofsys::component::sampling_wall_clock,
                                 "sampling_wall_clock", "Wall-clock timing",
                                 "Derived from statistical sampling")
TIMEMORY_METADATA_SPECIALIZATION(rocprofsys::component::sampling_cpu_clock,
                                 "sampling_cpu_clock", "CPU-clock timing",
                                 "Derived from statistical sampling")
TIMEMORY_METADATA_SPECIALIZATION(rocprofsys::component::sampling_percent,
                                 "sampling_percent",
                                 "Fraction of wall-clock time spent in functions",
                                 "Derived from statistical sampling")
TIMEMORY_METADATA_SPECIALIZATION(rocprofsys::component::sampling_gpu_busy_gfx,
                                 "sampling_gpu_busy_gfx",
                                 "GFX engine GPU Utilization (% busy) via AMD SMI",
                                 "Derived from sampling")
TIMEMORY_METADATA_SPECIALIZATION(rocprofsys::component::sampling_gpu_busy_umc,
                                 "sampling_gpu_busy_umc",
                                 "Memory controller GPU Utilization (% busy) via AMD SMI",
                                 "Derived from sampling")
TIMEMORY_METADATA_SPECIALIZATION(rocprofsys::component::sampling_gpu_busy_mm,
                                 "sampling_gpu_busy_mm",
                                 "Multimedia engine GPU Utilization (% busy) via AMD SMI",
                                 "Derived from sampling")
TIMEMORY_METADATA_SPECIALIZATION(rocprofsys::component::sampling_gpu_memory,
                                 "sampling_gpu_memory_usage",
                                 "GPU Memory Usage via AMD SMI", "Derived from sampling")
TIMEMORY_METADATA_SPECIALIZATION(rocprofsys::component::sampling_gpu_power,
                                 "sampling_gpu_power", "GPU Power Usage via AMD SMI",
                                 "Derived from sampling")
TIMEMORY_METADATA_SPECIALIZATION(rocprofsys::component::sampling_gpu_temp,
                                 "sampling_gpu_temp", "GPU Temperature via AMD SMI",
                                 "Derived from sampling")
TIMEMORY_METADATA_SPECIALIZATION(rocprofsys::component::sampling_gpu_vcn,
                                 "sampling_gpu_vcn",
                                 "GPU VCN Utilization (% activity) via AMD SMI",
                                 "Derived from sampling")
TIMEMORY_METADATA_SPECIALIZATION(rocprofsys::component::sampling_gpu_jpeg,
                                 "sampling_gpu_jpeg",
                                 "GPU JPEG Utilization (% activity) via AMD SMI",
                                 "Derived from sampling")

// statistics type
TIMEMORY_STATISTICS_TYPE(rocprofsys::component::sampling_wall_clock, double)
TIMEMORY_STATISTICS_TYPE(rocprofsys::component::sampling_cpu_clock, double)
TIMEMORY_STATISTICS_TYPE(rocprofsys::component::sampling_gpu_busy_gfx, double)
TIMEMORY_STATISTICS_TYPE(rocprofsys::component::sampling_gpu_busy_umc, double)
TIMEMORY_STATISTICS_TYPE(rocprofsys::component::sampling_gpu_busy_mm, double)
TIMEMORY_STATISTICS_TYPE(rocprofsys::component::sampling_gpu_temp, double)
TIMEMORY_STATISTICS_TYPE(rocprofsys::component::sampling_gpu_power, double)
TIMEMORY_STATISTICS_TYPE(rocprofsys::component::sampling_gpu_memory, double)
TIMEMORY_STATISTICS_TYPE(rocprofsys::component::sampling_gpu_vcn, double)
TIMEMORY_STATISTICS_TYPE(rocprofsys::component::sampling_gpu_jpeg, double)
TIMEMORY_STATISTICS_TYPE(rocprofsys::component::comm_data_tracker_t, float)

// enable timing units
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_timing_category, component::sampling_wall_clock,
                                 true_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_timing_category, component::sampling_cpu_clock,
                                 true_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_timing_category, component::sampling_percent,
                                 true_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::sampling_wall_clock,
                                 true_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::sampling_cpu_clock,
                                 true_type)

// enable percent units
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(uses_percent_units, component::sampling_gpu_busy_gfx,
                                 true_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(uses_percent_units, component::sampling_gpu_busy_umc,
                                 true_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(uses_percent_units, component::sampling_gpu_busy_mm,
                                 true_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(uses_percent_units, component::sampling_percent,
                                 true_type)

// enable memory units
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_memory_category, component::sampling_gpu_memory,
                                 true_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::sampling_gpu_memory,
                                 true_type)

// reporting categories (sum)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_sum, component::sampling_gpu_busy_gfx, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_sum, component::sampling_gpu_busy_umc, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_sum, component::sampling_gpu_busy_mm, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_sum, component::sampling_gpu_temp, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_sum, component::sampling_gpu_power, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_sum, component::sampling_gpu_memory, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_sum, component::sampling_gpu_vcn, false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_sum, component::sampling_gpu_jpeg, false_type)

// reporting categories (mean)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_mean, component::sampling_percent, false_type)

// reporting categories (stats)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_statistics, component::sampling_percent,
                                 false_type)

// reporting categories (self)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_self, component::sampling_percent, false_type)

#define ROCPROFSYS_DECLARE_EXTERN_COMPONENT(NAME, HAS_DATA, ...)                         \
    TIMEMORY_DECLARE_EXTERN_TEMPLATE(                                                    \
        struct tim::component::base<TIMEMORY_ESC(rocprofsys::component::NAME),           \
                                    __VA_ARGS__>)                                        \
    TIMEMORY_DECLARE_EXTERN_OPERATIONS(TIMEMORY_ESC(rocprofsys::component::NAME),        \
                                       HAS_DATA)                                         \
    TIMEMORY_DECLARE_EXTERN_STORAGE(TIMEMORY_ESC(rocprofsys::component::NAME))

#define ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(NAME, HAS_DATA, ...)                     \
    TIMEMORY_INSTANTIATE_EXTERN_TEMPLATE(                                                \
        struct tim::component::base<TIMEMORY_ESC(rocprofsys::component::NAME),           \
                                    __VA_ARGS__>)                                        \
    TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(TIMEMORY_ESC(rocprofsys::component::NAME),    \
                                           HAS_DATA)                                     \
    TIMEMORY_INSTANTIATE_EXTERN_STORAGE(TIMEMORY_ESC(rocprofsys::component::NAME))
