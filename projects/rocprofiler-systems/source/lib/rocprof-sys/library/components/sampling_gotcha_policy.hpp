// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common.hpp"
#include "core/config.hpp"
#include "core/timemory.hpp"
#include "library/components/sampling_gotcha.hpp"
#include "library/sampling.hpp"

#include <timemory/components/gotcha/backends.hpp>
#include <timemory/components/gotcha/components.hpp>
#include <timemory/process/threading.hpp>
#include <timemory/variadic/lightweight_tuple.hpp>

#include <csignal>
#include <pthread.h>
#include <set>
#include <tuple>

namespace rocprofsys
{
struct DefaultSamplingPolicy
{
    using gotcha_data_t = tim::component::gotcha_data;
    using component_t   = component::sampling_gotcha<DefaultSamplingPolicy>;
    using gotcha_t =
        comp::gotcha<component_t::gotcha_capacity, std::tuple<>, component_t>;
    using gotcha_bundle_t = tim::lightweight_tuple<gotcha_t>;

    // Binding failures are expected in processes that never load the GPU runtime, so
    // they are only worth reporting to someone who asked for the detail.
    static bool suppress_binding_warnings()
    {
        return get_verbose_env() < 3 && !get_debug_env();
    }

    // Returns null when this thread has no sampling signals registered, which is the
    // case before the sampler is set up and after it is torn down. Returned by
    // pointer so that the wrapper does not allocate on every wrapped call.
    static const std::set<int>* get_sampling_signals()
    {
        const auto& _signals = sampling::get_signal_types(threading::get_id());
        return (_signals) ? _signals.get() : nullptr;
    }

    static int block_signals(const sigset_t* _blocked, sigset_t* _prev)
    {
        return pthread_sigmask(SIG_BLOCK, _blocked, _prev);
    }

    static int restore_signals(const sigset_t* _prev)
    {
        return pthread_sigmask(SIG_SETMASK, _prev, nullptr);
    }
};
}  // namespace rocprofsys
