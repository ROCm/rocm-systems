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

#include "common/join.hpp"
#include "library/defines.hpp"

#if defined(TIMEMORY_PERFETTO_CATEGORIES)
#    error "TIMEMORY_PERFETTO_CATEGORIES is already defined. Please include \"" __FILE__ "\" before including any timemory files"
#endif

#include <timemory/api.hpp>
#include <timemory/api/macros.hpp>
#include <timemory/mpl/macros.hpp>
#include <timemory/mpl/types.hpp>

#define OMNITRACE_DEFINE_NAME_TRAIT(NAME, ...)                                           \
    namespace tim                                                                        \
    {                                                                                    \
    namespace trait                                                                      \
    {                                                                                    \
    template <>                                                                          \
    struct perfetto_category<__VA_ARGS__>                                                \
    {                                                                                    \
        static constexpr auto value = NAME;                                              \
    };                                                                                   \
    }                                                                                    \
    }

#define OMNITRACE_DECLARE_CATEGORY(NS, VALUE, NAME)                                      \
    TIMEMORY_DECLARE_NS_API(NS, VALUE)                                                   \
    OMNITRACE_DEFINE_NAME_TRAIT(NAME, NS::VALUE)
#define OMNITRACE_DEFINE_CATEGORY(NS, VALUE, NAME)                                       \
    TIMEMORY_DEFINE_NS_API(NS, VALUE)                                                    \
    OMNITRACE_DEFINE_NAME_TRAIT(NAME, NS::VALUE)

// these are defined by omnitrace
OMNITRACE_DEFINE_CATEGORY(project, omnitrace, "omnitrace")
OMNITRACE_DEFINE_CATEGORY(category, host, "host")
OMNITRACE_DEFINE_CATEGORY(category, user, "user")
OMNITRACE_DEFINE_CATEGORY(category, device, "device")
OMNITRACE_DEFINE_CATEGORY(category, device_hip, "device_hip")
OMNITRACE_DEFINE_CATEGORY(category, device_hsa, "device_hsa")
OMNITRACE_DEFINE_CATEGORY(category, rocm_hip, "rocm_hip")
OMNITRACE_DEFINE_CATEGORY(category, rocm_hsa, "rocm_hsa")
OMNITRACE_DEFINE_CATEGORY(category, rocm_roctx, "rocm_roctx")
OMNITRACE_DEFINE_CATEGORY(category, rocm_smi, "rocm_smi")
OMNITRACE_DEFINE_CATEGORY(category, rocm_rccl, "rccl")
OMNITRACE_DEFINE_CATEGORY(category, roctracer, "roctracer")
OMNITRACE_DEFINE_CATEGORY(category, rocprofiler, "rocprofiler")
OMNITRACE_DEFINE_CATEGORY(category, pthread, "pthread")
OMNITRACE_DEFINE_CATEGORY(category, kokkos, "kokkos")
OMNITRACE_DEFINE_CATEGORY(category, mpi, "mpi")
OMNITRACE_DEFINE_CATEGORY(category, ompt, "ompt")
OMNITRACE_DEFINE_CATEGORY(category, process_sampling, "process_sampling")
OMNITRACE_DEFINE_CATEGORY(category, critical_trace, "critical-trace")
OMNITRACE_DEFINE_CATEGORY(category, host_critical_trace, "host-critical-trace")
OMNITRACE_DEFINE_CATEGORY(category, device_critical_trace, "device-critical-trace")
OMNITRACE_DEFINE_CATEGORY(cpu_freq, cpu_page, "process_page_fault")
OMNITRACE_DEFINE_CATEGORY(cpu_freq, cpu_virt, "process_virtual_memory")
OMNITRACE_DEFINE_CATEGORY(cpu_freq, cpu_peak, "process_memory_hwm")
OMNITRACE_DEFINE_CATEGORY(cpu_freq, cpu_context_switch, "process_context_switch")
OMNITRACE_DEFINE_CATEGORY(cpu_freq, cpu_page_fault, "process_page_fault")
OMNITRACE_DEFINE_CATEGORY(cpu_freq, cpu_user_mode_time, "process_user_cpu_time")
OMNITRACE_DEFINE_CATEGORY(cpu_freq, cpu_kernel_mode_time, "process_kernel_cpu_time")

OMNITRACE_DECLARE_CATEGORY(category, sampling, "sampling")

namespace tim
{
namespace trait
{
template <typename... Tp>
using name = perfetto_category<Tp...>;
}
}  // namespace tim

#define OMNITRACE_PERFETTO_CATEGORIES                                                    \
    perfetto::Category(tim::trait::name<tim::category::host>::value)                     \
        .SetDescription("Host-side function tracing"),                                   \
        perfetto::Category("user").SetDescription("User-defined regions"),               \
        perfetto::Category("sampling").SetDescription("Host-side function sampling"),    \
        perfetto::Category("device_hip")                                                 \
            .SetDescription("Device-side functions submitted via HSA API"),              \
        perfetto::Category("device_hsa")                                                 \
            .SetDescription("Device-side functions submitted via HIP API"),              \
        perfetto::Category("rocm_hip").SetDescription("Host-side HIP functions"),        \
        perfetto::Category("rocm_hsa").SetDescription("Host-side HSA functions"),        \
        perfetto::Category("rocm_roctx").SetDescription("Host-side ROCTX labels"),       \
        perfetto::Category("device_busy")                                                \
            .SetDescription("Busy percentage of a GPU device"),                          \
        perfetto::Category("device_temp")                                                \
            .SetDescription("Temperature of GPU device in degC"),                        \
        perfetto::Category("device_power")                                               \
            .SetDescription("Power consumption of GPU device in watts"),                 \
        perfetto::Category("device_memory_usage")                                        \
            .SetDescription("Memory usage of GPU device in MB"),                         \
        perfetto::Category("thread_peak_memory")                                         \
            .SetDescription(                                                             \
                "Peak memory usage on thread in MB (derived from sampling)"),            \
        perfetto::Category("thread_context_switch")                                      \
            .SetDescription("Context switches on thread (derived from sampling)"),       \
        perfetto::Category("thread_page_fault")                                          \
            .SetDescription("Memory page faults on thread (derived from sampling)"),     \
        perfetto::Category("hardware_counter")                                           \
            .SetDescription("Hardware counter value on thread (derived from sampling)"), \
        perfetto::Category("cpu_freq")                                                   \
            .SetDescription("CPU frequency in MHz (collected in background thread)"),    \
        perfetto::Category("process_page_fault")                                         \
            .SetDescription(                                                             \
                "Memory page faults in process (collected in background thread)"),       \
        perfetto::Category("process_memory_hwm")                                         \
            .SetDescription("Memory High-Water Mark i.e. peak memory usage (collected "  \
                            "in background thread)"),                                    \
        perfetto::Category("process_virtual_memory")                                     \
            .SetDescription("Virtual memory usage in process in MB (collected in "       \
                            "background thread)"),                                       \
        perfetto::Category("process_context_switch")                                     \
            .SetDescription(                                                             \
                "Context switches in process (collected in background thread)"),         \
        perfetto::Category("process_page_fault")                                         \
            .SetDescription(                                                             \
                "Memory page faults in process (collected in background thread)"),       \
        perfetto::Category("process_user_cpu_time")                                      \
            .SetDescription("CPU time of functions executing in user-space in process "  \
                            "in seconds (collected in background thread)"),              \
        perfetto::Category("process_kernel_cpu_time")                                    \
            .SetDescription("CPU time of functions executing in kernel-space in "        \
                            "process in seconds (collected in background thread)"),      \
        perfetto::Category("pthread").SetDescription("Pthread functions"),               \
        perfetto::Category("kokkos").SetDescription("Kokkos regions"),                   \
        perfetto::Category("mpi").SetDescription("MPI regions"),                         \
        perfetto::Category("ompt").SetDescription("OpenMP Tools regions"),               \
        perfetto::Category("rccl").SetDescription(                                       \
            "ROCm Communication Collectives Library (RCCL) regions"),                    \
        perfetto::Category("comm_data")                                                  \
            .SetDescription(                                                             \
                "MPI/RCCL counters for tracking amount of data sent or received"),       \
        perfetto::Category("critical-trace").SetDescription("Combined critical traces"), \
        perfetto::Category("host-critical-trace")                                        \
            .SetDescription("Host-side critical traces"),                                \
        perfetto::Category("device-critical-trace")                                      \
            .SetDescription("Device-side critical traces"),                              \
        perfetto::Category("timemory").SetDescription("Events from the timemory API")

#if defined(TIMEMORY_USE_PERFETTO)
#    define TIMEMORY_PERFETTO_CATEGORIES OMNITRACE_PERFETTO_CATEGORIES
#endif
