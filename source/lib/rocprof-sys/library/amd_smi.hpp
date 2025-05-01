// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/defines.hpp"
#include "core/state.hpp"
#include "library/thread_data.hpp"

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

#include <chrono>
#include <cstdint>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <ratio>
#include <thread>
#include <tuple>
#include <type_traits>

namespace rocprofsys
{
namespace amd_smi
{
void
setup();

void
config();

void
sample();

void
shutdown();

void
post_process();

void set_state(State);

struct settings
{
    bool busy          = true;
    bool temp          = true;
    bool power         = true;
    bool mem_usage     = true;
    bool vcn_activity  = true;
    bool jpeg_activity = true;
};

struct data
{
    using msec_t    = std::chrono::milliseconds;
    using usec_t    = std::chrono::microseconds;
    using nsec_t    = std::chrono::nanoseconds;
    using promise_t = std::promise<void>;

    using timestamp_t = int64_t;
    using power_t     = uint32_t;
    using busy_perc_t = uint32_t;
    using mem_usage_t = uint64_t;
    using temp_t      = int64_t;

    ROCPROFSYS_DEFAULT_OBJECT(data)

    explicit data(uint32_t _dev_id);

    void sample(uint32_t _dev_id);
    void print(std::ostream& _os) const;

    static void post_process(uint32_t _dev_id);

    uint32_t              m_dev_id       = std::numeric_limits<uint32_t>::max();
    timestamp_t           m_ts           = 0;
    temp_t                m_temp         = 0;
    mem_usage_t           m_mem_usage    = 0;
    std::vector<uint16_t> m_vcn_metrics  = {};
    std::vector<uint16_t> m_jpeg_metrics = {};
#if ROCPROFSYS_USE_ROCM > 0
    amdsmi_engine_usage_t m_busy_perc = {};
    amdsmi_power_info_t   m_power     = {};
#else
    std::vector<busy_perc_t> m_busy_perc = {};
    std::vector<power_t>     m_power     = {};
#endif

    friend std::ostream& operator<<(std::ostream& _os, const data& _v)
    {
        _v.print(_os);
        return _os;
    }

private:
    friend void rocprofsys::amd_smi::setup();
    friend void rocprofsys::amd_smi::config();
    friend void rocprofsys::amd_smi::sample();
    friend void rocprofsys::amd_smi::shutdown();
    friend void rocprofsys::amd_smi::post_process();

    static size_t                        device_count;
    static std::set<uint32_t>            device_list;
    static std::unique_ptr<promise_t>    polling_finished;
    static std::vector<data>&            get_initial();
    static std::unique_ptr<std::thread>& get_thread();
    static bool                          setup();
    static bool                          shutdown();
};

#if !defined(ROCPROFSYS_USE_ROCM) || ROCPROFSYS_USE_ROCM == 0

inline void
setup()
{}

inline void
config()
{}

inline void
sample()
{}

inline void
shutdown()
{}

inline void
post_process()
{}

inline void set_state(State) {}
#endif
}  // namespace amd_smi
}  // namespace rocprofsys

#if defined(ROCPROFSYS_USE_ROCM) && ROCPROFSYS_USE_ROCM > 0
#    if !defined(ROCPROFSYS_EXTERN_COMPONENTS) ||                                        \
        (defined(ROCPROFSYS_EXTERN_COMPONENTS) && ROCPROFSYS_EXTERN_COMPONENTS > 0)

#        include <timemory/components/base.hpp>
#        include <timemory/components/data_tracker/components.hpp>
#        include <timemory/operations.hpp>

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_gfx>),
    true, double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_umc>),
    true, double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_mm>),
    true, double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_temp>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_power>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_memory>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_vcn>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_jpeg>), true,
    double)

#    endif
#endif
