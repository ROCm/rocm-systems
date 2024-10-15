// Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All Rights Reserved.
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
namespace rocm_smi
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

uint32_t
device_count();

struct settings
{
    bool busy      = true;
    bool temp      = true;
    bool power     = true;
    bool mem_usage = true;
};

struct data
{
    using msec_t    = std::chrono::milliseconds;
    using usec_t    = std::chrono::microseconds;
    using nsec_t    = std::chrono::nanoseconds;
    using promise_t = std::promise<void>;

    using timestamp_t = int64_t;
    using power_t     = uint64_t;
    using busy_perc_t = uint32_t;
    using mem_usage_t = uint64_t;
    using temp_t      = int64_t;

    ROCPROFSYS_DEFAULT_OBJECT(data)

    explicit data(uint32_t _dev_id);

    void sample(uint32_t _dev_id);
    void print(std::ostream& _os) const;

    static void post_process(uint32_t _dev_id);

    uint32_t    m_dev_id    = std::numeric_limits<uint32_t>::max();
    timestamp_t m_ts        = 0;
    busy_perc_t m_busy_perc = 0;
    temp_t      m_temp      = 0;
    power_t     m_power     = 0;
    mem_usage_t m_mem_usage = 0;

    friend std::ostream& operator<<(std::ostream& _os, const data& _v)
    {
        _v.print(_os);
        return _os;
    }

private:
    friend void rocprofsys::rocm_smi::setup();
    friend void rocprofsys::rocm_smi::config();
    friend void rocprofsys::rocm_smi::sample();
    friend void rocprofsys::rocm_smi::shutdown();
    friend void rocprofsys::rocm_smi::post_process();

    static size_t                        device_count;
    static std::set<uint32_t>            device_list;
    static std::unique_ptr<promise_t>    polling_finished;
    static std::vector<data>&            get_initial();
    static std::unique_ptr<std::thread>& get_thread();
    static bool                          setup();
    static bool                          shutdown();
};

#if !defined(ROCPROFSYS_USE_ROCM_SMI)
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
}  // namespace rocm_smi
}  // namespace rocprofsys

#if defined(ROCPROFSYS_USE_ROCM_SMI) && ROCPROFSYS_USE_ROCM_SMI > 0
#    if !defined(ROCPROFSYS_EXTERN_COMPONENTS) ||                                        \
        (defined(ROCPROFSYS_EXTERN_COMPONENTS) && ROCPROFSYS_EXTERN_COMPONENTS > 0)

#        include <timemory/components/base.hpp>
#        include <timemory/components/data_tracker/components.hpp>
#        include <timemory/operations.hpp>

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_temp>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_power>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_memory>), true,
    double)

#    endif
#endif
