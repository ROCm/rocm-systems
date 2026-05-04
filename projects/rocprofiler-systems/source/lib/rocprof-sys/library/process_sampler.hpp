// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/common.hpp"
#include "core/components/fwd.hpp"
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
    std::function<void()> pause        = []() {};
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
    static void pause();
    static void resume();
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

inline void
pause()
{
    sampler::pause();
}

inline void
resume()
{
    sampler::resume();
}
//
}  // namespace process_sampler
}  // namespace rocprofsys
