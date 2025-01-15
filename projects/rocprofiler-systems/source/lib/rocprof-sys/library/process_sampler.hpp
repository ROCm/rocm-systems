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

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/defines.hpp"
#include "core/state.hpp"
#include "library/thread_data.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <type_traits>
#include <vector>

namespace rocprofsys
{
namespace process_sampler
{
struct instance
{
    std::function<void()> setup        = []() {};
    std::function<void()> shutdown     = []() {};
    std::function<void()> config       = []() {};
    std::function<void()> sample       = []() {};
    std::function<void()> post_process = []() {};
};
//
struct sampler
{
    using msec_t    = std::chrono::milliseconds;
    using usec_t    = std::chrono::microseconds;
    using nsec_t    = std::chrono::nanoseconds;
    using promise_t = std::promise<void>;
    using future_t  = std::future<void>;
    using state_t   = State;

    using timestamp_t = int64_t;

    template <typename Tp                                                      = nsec_t,
              std::enable_if_t<!std::is_same_v<std::decay_t<Tp>, nsec_t>, int> = 0>
    static void poll(std::atomic<state_t>* _state, Tp&& _interval, promise_t*);

    static void setup();
    static void shutdown();
    static void post_process();
    static void set_state(state_t);
    static void poll(std::atomic<state_t>* _state, nsec_t _interval, promise_t*);
};
//
template <
    typename Tp,
    std::enable_if_t<!std::is_same_v<std::decay_t<Tp>, std::chrono::nanoseconds>, int>>
void
sampler::poll(std::atomic<state_t>* _state, Tp&& _interval, promise_t* _prom)
{
    poll(_state, std::chrono::duration_cast<nsec_t>(_interval), _prom);
}
//
inline void
setup()
{
    sampler::setup();
}

inline void
shutdown()
{
    sampler::shutdown();
}

inline void
post_process()
{
    sampler::post_process();
}
//
}  // namespace process_sampler
}  // namespace rocprofsys
