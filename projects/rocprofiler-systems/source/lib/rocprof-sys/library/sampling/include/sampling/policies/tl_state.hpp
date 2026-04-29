// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Typed per-thread sampling state pointers.
//
// The signal handler reads `sampler` and `offload` to dispatch a backtrace
// sample without taking any mutex (NFR-TS-2). Definitions of the static
// thread_locals live in services_accessor.cpp (one TU) so the static-TLS
// model is used and dlopen-loaded libraries do not fault.
//
// One instantiation per Policies set; in production exactly one
// (default_sampling_policies). Tests instantiate with their own Policies.

#include "sampling/sampling_policies.hpp"
#include "sampling/src/thread_sampler_state.hpp"

#include <cstdint>

namespace rocprofsys::sampling
{

template <class Policies>
struct tl_state
{
    using state_t   = thread_sampler_state<Policies>;
    using offload_t = typename Policies::offload;

    static thread_local state_t*   sampler;
    static thread_local offload_t* offload;
    static thread_local int64_t    logical_tid;
};

// C++17: inline + thread_local definitions of static template members ensure
// exactly one instance per (Policies, thread) without per-TU ODR conflicts.
template <class Policies>
inline thread_local typename tl_state<Policies>::state_t* tl_state<Policies>::sampler =
    nullptr;
template <class Policies>
inline thread_local typename tl_state<Policies>::offload_t* tl_state<Policies>::offload =
    nullptr;
template <class Policies>
inline thread_local int64_t tl_state<Policies>::logical_tid = -1;

}  // namespace rocprofsys::sampling
