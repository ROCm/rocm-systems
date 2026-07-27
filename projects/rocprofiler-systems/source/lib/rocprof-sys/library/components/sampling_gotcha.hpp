// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <timemory/components/base/declaration.hpp>

#include <csignal>
#include <cstddef>
#include <set>
#include <string>
#include <type_traits>

namespace rocprofsys::component::sampling_concepts
{
template <typename Policy>
concept HasGotchaData = requires { typename Policy::gotcha_data_t; };

template <typename Policy>
concept HasGotcha = requires { typename Policy::gotcha_t; };

template <typename Policy>
concept HasGotchaBundle = requires { typename Policy::gotcha_bundle_t; };
}  // namespace rocprofsys::component::sampling_concepts

namespace rocprofsys
{
namespace component
{
// Blocks the sampling signals for the scope. Restores the caller's mask verbatim
// instead of unblocking, since callers such as sampler setup, thread creation and MPI
// finalize hold these signals blocked across regions that reach a wrapped function.
template <typename Policy>
struct scoped_sampling_block
{
    scoped_sampling_block()
    {
        if(s_depth++ != 0) return;

        const auto* _signals = Policy::get_sampling_signals();
        if(!_signals || _signals->empty()) return;

        auto _blocked = sigset_t{};
        sigemptyset(&_blocked);
        for(auto itr : *_signals)
            sigaddset(&_blocked, itr);

        m_restore = (Policy::block_signals(&_blocked, &m_prev_mask) == 0);
    }

    ~scoped_sampling_block()
    {
        if(--s_depth == 0 && m_restore) Policy::restore_signals(&m_prev_mask);
    }

    scoped_sampling_block(const scoped_sampling_block&)            = delete;
    scoped_sampling_block(scoped_sampling_block&&)                 = delete;
    scoped_sampling_block& operator=(const scoped_sampling_block&) = delete;
    scoped_sampling_block& operator=(scoped_sampling_block&&)      = delete;

private:
    static thread_local int s_depth;

    sigset_t m_prev_mask{};
    bool     m_restore = false;
};

template <typename Policy>
thread_local int scoped_sampling_block<Policy>::s_depth = 0;

// Wraps function calls that can fail if a sampling signal is delivered during their
// execution. Only hsa_init is covered for now; see configure() for why that is the
// entry point and how to widen the set.
template <typename Policy>
struct sampling_gotcha : tim::component::base<sampling_gotcha<Policy>, void>
{
    // gotcha_t and gotcha_bundle_t are checked in the member functions rather than
    // here: the policy names this component to size its own gotcha_t, so those types
    // do not exist yet while this class body is being instantiated
    static_assert(sampling_concepts::HasGotchaData<Policy>,
                  "Policy must have a gotcha_data_t type");

    // Binding slots, one per wrapped function. To cover another entry point, add a
    // slot here and a matching configure() call in the initializer; the capacity below
    // follows along on its own, so the two cannot drift apart.
    enum wrapped_call : size_t
    {
        hsa_init_slot = 0,
        wrapped_call_count
    };

    static constexpr size_t gotcha_capacity = wrapped_call_count;

    using gotcha_data_t = typename Policy::gotcha_data_t;

    // string id for component
    static std::string label() { return "sampling_gotcha"; }

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    // activate / deactivate the gotcha bindings
    static void start();
    static void stop();

    // mask sampling signals -> call the real function -> restore signals
    template <typename Ret, typename... Args>
    Ret operator()(const gotcha_data_t&, Ret (*)(Args...), Args...) const;
};

namespace detail
{
template <typename Policy>
auto&
get_sampling_gotcha()
{
    static auto _v = typename Policy::gotcha_bundle_t{};
    return _v;
}
}  // namespace detail

template <typename Policy>
void
sampling_gotcha<Policy>::configure()
{
    static_assert(sampling_concepts::HasGotcha<Policy>,
                  "Policy must have a gotcha_t type");

    using gotcha_t = typename Policy::gotcha_t;

    // A process that never loads the GPU runtime has no such symbol to bind, which is
    // not a fault, so stay quiet about a failed binding unless asked for detail
    if(Policy::suppress_binding_warnings())
    {
        for(size_t i = 0; i < gotcha_t::capacity(); ++i)
        {
            auto* itr = static_cast<gotcha_data_t*>(gotcha_t::at(i));
            if(itr) itr->verbose = -1;
        }
    }

    gotcha_t::get_initializer() = []() {
        // hsa_init is the choke point behind HIP entry points, and it is where the
        // device discovery that talks to the driver happens.
        gotcha_t::template configure<hsa_init_slot, int>("hsa_init");
    };
}

template <typename Policy>
void
sampling_gotcha<Policy>::shutdown()
{
    static_assert(sampling_concepts::HasGotcha<Policy>,
                  "Policy must have a gotcha_t type");

    Policy::gotcha_t::disable();
}

template <typename Policy>
void
sampling_gotcha<Policy>::start()
{
    static_assert(sampling_concepts::HasGotcha<Policy>,
                  "Policy must have a gotcha_t type");
    static_assert(sampling_concepts::HasGotchaBundle<Policy>,
                  "Policy must have a gotcha_bundle_t type");

    using gotcha_t = typename Policy::gotcha_t;

    if(!detail::get_sampling_gotcha<Policy>().template get<gotcha_t>()->get_is_running())
    {
        configure();
        detail::get_sampling_gotcha<Policy>().template get<gotcha_t>()->start();
    }
}

template <typename Policy>
void
sampling_gotcha<Policy>::stop()
{
    static_assert(sampling_concepts::HasGotcha<Policy>,
                  "Policy must have a gotcha_t type");
    static_assert(sampling_concepts::HasGotchaBundle<Policy>,
                  "Policy must have a gotcha_bundle_t type");

    using gotcha_t = typename Policy::gotcha_t;

    detail::get_sampling_gotcha<Policy>().template get<gotcha_t>()->stop();
}

template <typename Policy>
template <typename Ret, typename... Args>
Ret
sampling_gotcha<Policy>::operator()(const gotcha_data_t&, Ret (*_func)(Args...),
                                    Args... _args) const
{
    if constexpr(std::is_void_v<Ret>)
    {
        scoped_sampling_block<Policy> _block{};
        (*_func)(_args...);
    }
    else
    {
        scoped_sampling_block<Policy> _block{};
        return (*_func)(_args...);
    }
}
}  // namespace component
}  // namespace rocprofsys
