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

#define OMNITRACE_DEFINE_NAME_TRAIT(NAME, DESC, ...)                                     \
    namespace tim                                                                        \
    {                                                                                    \
    namespace trait                                                                      \
    {                                                                                    \
    template <>                                                                          \
    struct perfetto_category<__VA_ARGS__>                                                \
    {                                                                                    \
        static constexpr auto value       = NAME;                                        \
        static constexpr auto description = DESC;                                        \
    };                                                                                   \
    }                                                                                    \
    }

#define OMNITRACE_DECLARE_CATEGORY(NS, VALUE, NAME, DESC)                                \
    TIMEMORY_DECLARE_NS_API(NS, VALUE)                                                   \
    OMNITRACE_DEFINE_NAME_TRAIT(NAME, DESC, NS::VALUE)
#define OMNITRACE_DEFINE_CATEGORY(NS, VALUE, NAME, DESC)                                 \
    TIMEMORY_DEFINE_NS_API(NS, VALUE)                                                    \
    OMNITRACE_DEFINE_NAME_TRAIT(NAME, DESC, NS::VALUE)

// clang-format off
// these are defined by omnitrace
OMNITRACE_DEFINE_CATEGORY(project, omnitrace, "omnitrace", "Omnitrace project")
OMNITRACE_DEFINE_CATEGORY(category, host, "host", "Host-side function tracing")
OMNITRACE_DEFINE_CATEGORY(category, user, "user", "User-defined regions")
OMNITRACE_DEFINE_CATEGORY(category, device_hip, "device_hip", "Device-side functions submitted via HIP API")
OMNITRACE_DEFINE_CATEGORY(category, device_hsa, "device_hsa", "Device-side functions submitted via HSA API")
OMNITRACE_DEFINE_CATEGORY(category, rocm_hip, "rocm_hip", "Host-side HIP functions")
OMNITRACE_DEFINE_CATEGORY(category, rocm_hsa, "rocm_hsa", "Host-side HSA functions")
OMNITRACE_DEFINE_CATEGORY(category, rocm_roctx, "rocm_roctx", "ROCTx labels")
OMNITRACE_DEFINE_CATEGORY(category, rocm_smi, "rocm_smi", "rocm-smi data")
OMNITRACE_DEFINE_CATEGORY(category, rocm_smi_busy, "device_busy", "Busy percentage of a GPU device")
OMNITRACE_DEFINE_CATEGORY(category, rocm_smi_temp, "device_temp",   "Temperature of a GPU device")
OMNITRACE_DEFINE_CATEGORY(category, rocm_smi_power, "device_power", "Power consumption of a GPU device")
OMNITRACE_DEFINE_CATEGORY(category, rocm_smi_memory_usage, "device_memory_usage", "Memory usage of a GPU device")
OMNITRACE_DEFINE_CATEGORY(category, rocm_rccl, "rccl", "ROCm Communication Collectives Library (RCCL) regions")
OMNITRACE_DEFINE_CATEGORY(category, roctracer, "roctracer", "Kernel tracing provided by roctracer")
OMNITRACE_DEFINE_CATEGORY(category, rocprofiler, "rocprofiler", "HW counter data provided by rocprofiler")
OMNITRACE_DEFINE_CATEGORY(category, pthread, "pthread", "POSIX threading functions")
OMNITRACE_DEFINE_CATEGORY(category, kokkos, "kokkos", "KokkosTools regions")
OMNITRACE_DEFINE_CATEGORY(category, mpi, "mpi", "MPI regions")
OMNITRACE_DEFINE_CATEGORY(category, ompt, "ompt", "OpenMP tools regions")
OMNITRACE_DEFINE_CATEGORY(category, process_sampling, "process_sampling", "Process-level data")
OMNITRACE_DEFINE_CATEGORY(category, comm_data, "comm_data", "MPI/RCCL counters for tracking amount of data sent or received")
OMNITRACE_DEFINE_CATEGORY(category, critical_trace, "critical-trace", "Critical trace data")
OMNITRACE_DEFINE_CATEGORY(category, host_critical_trace, "host-critical-trace", "Host-side critical trace data")
OMNITRACE_DEFINE_CATEGORY(category, device_critical_trace, "device-critical-trace", "Device-side critical trace data")
OMNITRACE_DEFINE_CATEGORY(category, causal, "causal", "Causal profiling data")
OMNITRACE_DEFINE_CATEGORY(category, cpu_freq, "cpu_frequency", "CPU frequency (collected in background thread)")
OMNITRACE_DEFINE_CATEGORY(category, process_page, "process_page_fault", "Memory page faults in process (collected in background thread)")
OMNITRACE_DEFINE_CATEGORY(category, process_virt, "process_virtual_memory", "Virtual memory usage in process in MB (collected in background thread)")
OMNITRACE_DEFINE_CATEGORY(category, process_peak, "process_memory_hwm", "Memory High-Water Mark i.e. peak memory usage (collected in background thread)")
OMNITRACE_DEFINE_CATEGORY(category, process_context_switch, "process_context_switch", "Context switches in process (collected in background thread)")
OMNITRACE_DEFINE_CATEGORY(category, process_page_fault, "process_page_fault", "Memory page faults in process (collected in background thread)")
OMNITRACE_DEFINE_CATEGORY(category, process_user_mode_time, "process_user_cpu_time", "CPU time of functions executing in user-space in process in seconds (collected in background thread)")
OMNITRACE_DEFINE_CATEGORY(category, process_kernel_mode_time, "process_kernel_cpu_time", "CPU time of functions executing in kernel-space in process in seconds (collected in background thread)")
OMNITRACE_DEFINE_CATEGORY(category, thread_page_fault, "thread_page_fault", "Memory page faults on thread (derived from sampling)")
OMNITRACE_DEFINE_CATEGORY(category, thread_peak_memory, "thread_peak_memory", "Peak memory usage on thread in MB (derived from sampling)")
OMNITRACE_DEFINE_CATEGORY(category, thread_context_switch, "thread_context_switch", "Context switches on thread (derived from sampling)")
OMNITRACE_DEFINE_CATEGORY(category, thread_hardware_counter, "thread_hardware_counter", "Hardware counter value on thread (derived from sampling)")
OMNITRACE_DEFINE_CATEGORY(category, kernel_hardware_counter, "kernel_hardware_counter", "Hardware counter value for kernel (deterministic)")
OMNITRACE_DEFINE_CATEGORY(category, numa, "numa", "Non-unified memory architecture")

OMNITRACE_DECLARE_CATEGORY(category, sampling, "sampling", "Host-side call-stack sampling")
// clang-format on

namespace tim
{
namespace trait
{
template <typename... Tp>
using name = perfetto_category<Tp...>;
}
}  // namespace tim

#define OMNITRACE_PERFETTO_CATEGORY(TYPE)                                                \
    ::perfetto::Category(::tim::trait::perfetto_category<::tim::TYPE>::value)            \
        .SetDescription(::tim::trait::perfetto_category<::tim::TYPE>::description)

#define OMNITRACE_PERFETTO_CATEGORIES                                                    \
    OMNITRACE_PERFETTO_CATEGORY(category::host),                                         \
        OMNITRACE_PERFETTO_CATEGORY(category::user),                                     \
        OMNITRACE_PERFETTO_CATEGORY(category::sampling),                                 \
        OMNITRACE_PERFETTO_CATEGORY(category::device_hip),                               \
        OMNITRACE_PERFETTO_CATEGORY(category::device_hsa),                               \
        OMNITRACE_PERFETTO_CATEGORY(category::rocm_hip),                                 \
        OMNITRACE_PERFETTO_CATEGORY(category::rocm_hsa),                                 \
        OMNITRACE_PERFETTO_CATEGORY(category::rocm_roctx),                               \
        OMNITRACE_PERFETTO_CATEGORY(category::rocm_smi),                                 \
        OMNITRACE_PERFETTO_CATEGORY(category::rocm_smi_busy),                            \
        OMNITRACE_PERFETTO_CATEGORY(category::rocm_smi_temp),                            \
        OMNITRACE_PERFETTO_CATEGORY(category::rocm_smi_power),                           \
        OMNITRACE_PERFETTO_CATEGORY(category::rocm_smi_memory_usage),                    \
        OMNITRACE_PERFETTO_CATEGORY(category::rocm_rccl),                                \
        OMNITRACE_PERFETTO_CATEGORY(category::roctracer),                                \
        OMNITRACE_PERFETTO_CATEGORY(category::rocprofiler),                              \
        OMNITRACE_PERFETTO_CATEGORY(category::pthread),                                  \
        OMNITRACE_PERFETTO_CATEGORY(category::kokkos),                                   \
        OMNITRACE_PERFETTO_CATEGORY(category::mpi),                                      \
        OMNITRACE_PERFETTO_CATEGORY(category::ompt),                                     \
        OMNITRACE_PERFETTO_CATEGORY(category::sampling),                                 \
        OMNITRACE_PERFETTO_CATEGORY(category::process_sampling),                         \
        OMNITRACE_PERFETTO_CATEGORY(category::comm_data),                                \
        OMNITRACE_PERFETTO_CATEGORY(category::critical_trace),                           \
        OMNITRACE_PERFETTO_CATEGORY(category::host_critical_trace),                      \
        OMNITRACE_PERFETTO_CATEGORY(category::device_critical_trace),                    \
        OMNITRACE_PERFETTO_CATEGORY(category::causal),                                   \
        OMNITRACE_PERFETTO_CATEGORY(category::cpu_freq),                                 \
        OMNITRACE_PERFETTO_CATEGORY(category::process_page),                             \
        OMNITRACE_PERFETTO_CATEGORY(category::process_virt),                             \
        OMNITRACE_PERFETTO_CATEGORY(category::process_peak),                             \
        OMNITRACE_PERFETTO_CATEGORY(category::process_context_switch),                   \
        OMNITRACE_PERFETTO_CATEGORY(category::process_page_fault),                       \
        OMNITRACE_PERFETTO_CATEGORY(category::process_user_mode_time),                   \
        OMNITRACE_PERFETTO_CATEGORY(category::process_kernel_mode_time),                 \
        OMNITRACE_PERFETTO_CATEGORY(category::thread_page_fault),                        \
        OMNITRACE_PERFETTO_CATEGORY(category::thread_peak_memory),                       \
        OMNITRACE_PERFETTO_CATEGORY(category::thread_context_switch),                    \
        OMNITRACE_PERFETTO_CATEGORY(category::thread_hardware_counter),                  \
        OMNITRACE_PERFETTO_CATEGORY(category::kernel_hardware_counter),                  \
        OMNITRACE_PERFETTO_CATEGORY(category::numa),                                     \
        perfetto::Category("timemory").SetDescription("Events from the timemory API")

#if defined(TIMEMORY_USE_PERFETTO)
#    define TIMEMORY_PERFETTO_CATEGORIES OMNITRACE_PERFETTO_CATEGORIES
#endif
