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

#include "core/defines.hpp"

#include <timemory/backends/papi.hpp>

#include <cstdint>
#include <linux/perf_event.h>

#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
#    include <sys/syscall.h>
#    define gettid() syscall(SYS_gettid)
#endif

// Workaround for missing hw_breakpoint.h include file:
//   This include file just defines constants used to configure watchpoint registers.
//   This will be constant across x86 systems.
enum
{
    HW_BREAKPOINT_X = 4
};

namespace rocprofsys
{
namespace perf
{
/// An enum class with all the available sampling data
enum class sample : uint64_t
{
    ip        = PERF_SAMPLE_IP,
    pid_tid   = PERF_SAMPLE_TID,
    time      = PERF_SAMPLE_TIME,
    addr      = PERF_SAMPLE_ADDR,
    id        = PERF_SAMPLE_ID,
    stream_id = PERF_SAMPLE_STREAM_ID,
    cpu       = PERF_SAMPLE_CPU,
    period    = PERF_SAMPLE_PERIOD,

#if defined(PERF_SAMPLE_READ)
    read = PERF_SAMPLE_READ,
#else
    read = 0,
#endif

    callchain = PERF_SAMPLE_CALLCHAIN,
    raw       = PERF_SAMPLE_RAW,

#if defined(PERF_SAMPLE_BRANCH_STACK)
    branch_stack = PERF_SAMPLE_BRANCH_STACK,
#else
    branch_stack = 0,
#endif

#if defined(PERF_SAMPLE_REGS_USER)
    regs = PERF_SAMPLE_REGS_USER,
#else
    regs = 0,
#endif

#if defined(PERF_SAMPLE_STACK_USER)
    stack = PERF_SAMPLE_STACK_USER,
#else
    stack = 0,
#endif

#if defined(PERF_SAMPLE_WEIGHT)
    weight = PERF_SAMPLE_WEIGHT,
#else
    weight = 0,
#endif

#if defined(PERF_SAMPLE_DATA_SRC)
    data_src = PERF_SAMPLE_DATA_SRC,
#else
    data_src = 0,
#endif

#if defined(PERF_SAMPLE_IDENTIFIER)
    identifier = PERF_SAMPLE_IDENTIFIER,
#else
    identifier = 0,
#endif

#if defined(PERF_SAMPLE_TRANSACTION)
    transaction = PERF_SAMPLE_TRANSACTION,
#else
    transaction = 0,
#endif

#if defined(PERF_SAMPLE_REGS_INTR)
    regs_intr = PERF_SAMPLE_REGS_INTR,
#else
    regs_intr = 0,
#endif

#if defined(PERF_SAMPLE_PHYS_ADDR)
    phys_addr = PERF_SAMPLE_PHYS_ADDR,
#else
    phys_addr = 0,
#endif

#if defined(PERF_SAMPLE_CGROUP)
    cgroup = PERF_SAMPLE_CGROUP,
#else
    cgroup = 0,
#endif

    last = PERF_SAMPLE_MAX
};

enum class event_type : int
{
    hardware   = PERF_TYPE_HARDWARE,
    software   = PERF_TYPE_SOFTWARE,
    tracepoint = PERF_TYPE_TRACEPOINT,
    hw_cache   = PERF_TYPE_HW_CACHE,
    raw        = PERF_TYPE_RAW,
    breakpoint = PERF_TYPE_BREAKPOINT,
    max        = PERF_TYPE_MAX,
};

enum class hw_config : int
{
    cpu_cycles              = PERF_COUNT_HW_CPU_CYCLES,
    instructions            = PERF_COUNT_HW_INSTRUCTIONS,
    cache_references        = PERF_COUNT_HW_CACHE_REFERENCES,
    cache_misses            = PERF_COUNT_HW_CACHE_MISSES,
    branch_instructions     = PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
    branch_misses           = PERF_COUNT_HW_BRANCH_MISSES,
    bus_cycles              = PERF_COUNT_HW_BUS_CYCLES,
    stalled_cycles_frontend = PERF_COUNT_HW_STALLED_CYCLES_FRONTEND,
    stalled_cycles_backend  = PERF_COUNT_HW_STALLED_CYCLES_BACKEND,
    reference_cpu_cycles    = PERF_COUNT_HW_REF_CPU_CYCLES,
    max                     = PERF_COUNT_HW_MAX,
};

enum class sw_config : int
{
    cpu_clock         = PERF_COUNT_SW_CPU_CLOCK,
    task_clock        = PERF_COUNT_SW_TASK_CLOCK,
    page_faults       = PERF_COUNT_SW_PAGE_FAULTS,
    context_switches  = PERF_COUNT_SW_CONTEXT_SWITCHES,
    cpu_migrations    = PERF_COUNT_SW_CPU_MIGRATIONS,
    page_faults_minor = PERF_COUNT_SW_PAGE_FAULTS_MIN,
    page_faults_major = PERF_COUNT_SW_PAGE_FAULTS_MAJ,
    alignment_faults  = PERF_COUNT_SW_ALIGNMENT_FAULTS,
    emulation_faults  = PERF_COUNT_SW_EMULATION_FAULTS,
    max               = PERF_COUNT_SW_MAX,
};

enum class hw_cache_config : int
{
    l1d  = PERF_COUNT_HW_CACHE_L1D,
    l1i  = PERF_COUNT_HW_CACHE_L1I,
    ll   = PERF_COUNT_HW_CACHE_LL,
    dtlb = PERF_COUNT_HW_CACHE_DTLB,
    itlb = PERF_COUNT_HW_CACHE_ITLB,
    bpu  = PERF_COUNT_HW_CACHE_BPU,
    node = PERF_COUNT_HW_CACHE_NODE,
    max  = PERF_COUNT_HW_CACHE_MAX,
};

enum class hw_cache_op : int
{
    read     = PERF_COUNT_HW_CACHE_OP_READ,
    write    = PERF_COUNT_HW_CACHE_OP_WRITE,
    prefetch = PERF_COUNT_HW_CACHE_OP_PREFETCH,
    max      = PERF_COUNT_HW_CACHE_OP_MAX,
};

enum class hw_cache_op_result : int
{
    access = PERF_COUNT_HW_CACHE_RESULT_ACCESS,
    miss   = PERF_COUNT_HW_CACHE_RESULT_MISS,
    max    = PERF_COUNT_HW_CACHE_RESULT_MAX,
};

/// An enum to distinguish types of records in the mmapped ring buffer
enum class record_type
{
    mmap       = PERF_RECORD_MMAP,
    lost       = PERF_RECORD_LOST,
    comm       = PERF_RECORD_COMM,
    exit       = PERF_RECORD_EXIT,
    throttle   = PERF_RECORD_THROTTLE,
    unthrottle = PERF_RECORD_UNTHROTTLE,
    fork       = PERF_RECORD_FORK,
    read       = PERF_RECORD_READ,
    sample     = PERF_RECORD_SAMPLE,

#if defined(PERF_RECORD_MMAP2)
    mmap2 = PERF_RECORD_MMAP2,
#else
    mmap2 = 0,
#endif

#if defined(PERF_RECORD_AUX)
    aux = PERF_RECORD_AUX,
#else
    aux = 0,
#endif

#if defined(PERF_RECORD_ITRACE_START)
    itrace_start = PERF_RECORD_ITRACE_START,
#else
    itrace_start = 0,
#endif

#if defined(PERF_RECORD_LOST_SAMPLES)
    lost_samples = PERF_RECORD_LOST_SAMPLES,
#else
    lost_samples = 0,
#endif

#if defined(PERF_RECORD_SWITCH)
    switch_record = PERF_RECORD_SWITCH,
#else
    switch_record = 0,
#endif

#if defined(PERF_RECORD_SWITCH_CPU_WIDE)
    switch_cpu_wide = PERF_RECORD_SWITCH_CPU_WIDE,
#else
    switch_cpu_wide = 0,
#endif

#if defined(PERF_RECORD_NAMESPACES)
    namespaces = PERF_RECORD_NAMESPACES,
#else
    namespaces = 0,
#endif

#if defined(PERF_RECORD_KSYMBOL)
    ksymbol = PERF_RECORD_KSYMBOL,
#else
    ksymbol = 0,
#endif

#if defined(PERF_RECORD_BPF_EVENT)
    bpf_event = PERF_RECORD_BPF_EVENT,
#else
    bpf_event = 0,
#endif

#if defined(PERF_RECORD_CGROUP)
    cgroup = PERF_RECORD_CGROUP,
#else
    cgroup = 0,
#endif

#if defined(PERF_RECORD_TEXT_POKE)
    text_poke = PERF_RECORD_TEXT_POKE,
#else
    text_poke = 0,
#endif
};

std::vector<std::string>
get_config_choices();

event_type get_event_type(std::string_view);
hw_config  get_hw_config(std::string_view);
sw_config  get_sw_config(std::string_view);
int        get_hw_cache_config(std::string_view);

void
config_overflow_sampling(struct perf_event_attr&, std::string_view, double);
}  // namespace perf
}  // namespace rocprofsys
