// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "common/static_object.hpp"
#include "logger/debug.hpp"
#include "utility.hpp"

#include <spdlog/fmt/fmt.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys
{
enum class process_lifecycle_state : std::uint16_t
{
    PreInit = 0,
    Init,
    Active,
    Finalized,
    Disabled,
};

enum class thread_lifecycle_state : std::uint16_t
{
    Enabled = 0,
    Internal,
    Completed,
    Disabled,
};

enum class process_mode : std::uint16_t
{
    Trace = 0,
    Sampling,
    Causal,
    Coverage,
};

enum class process_causal_backend : std::uint16_t
{
    Perf = 0,
    Timer,
    Auto,
};

enum class process_causal_mode : std::uint16_t
{
    Line = 0,
    Function,
};
}  // namespace rocprofsys

template <>
struct fmt::formatter<rocprofsys::process_lifecycle_state>
: fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(rocprofsys::process_lifecycle_state _v, FormatContext& ctx) const
    {
        std::string_view _s = {};
        switch(_v)
        {
            case rocprofsys::process_lifecycle_state::PreInit: _s = "PreInit"; break;
            case rocprofsys::process_lifecycle_state::Init: _s = "Init"; break;
            case rocprofsys::process_lifecycle_state::Active: _s = "Active"; break;
            case rocprofsys::process_lifecycle_state::Disabled: _s = "Disabled"; break;
            case rocprofsys::process_lifecycle_state::Finalized: _s = "Finalized"; break;
        }
        return fmt::formatter<std::string_view>::format(_s, ctx);
    }
};

template <>
struct fmt::formatter<rocprofsys::thread_lifecycle_state>
: fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(rocprofsys::thread_lifecycle_state _v, FormatContext& ctx) const
    {
        std::string_view _s = {};
        switch(_v)
        {
            case rocprofsys::thread_lifecycle_state::Enabled: _s = "Enabled"; break;
            case rocprofsys::thread_lifecycle_state::Internal: _s = "Internal"; break;
            case rocprofsys::thread_lifecycle_state::Completed: _s = "Completed"; break;
            case rocprofsys::thread_lifecycle_state::Disabled: _s = "Disabled"; break;
        }
        return fmt::formatter<std::string_view>::format(_s, ctx);
    }
};

template <>
struct fmt::formatter<rocprofsys::process_mode> : fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(rocprofsys::process_mode _v, FormatContext& ctx) const
    {
        std::string_view _s = {};
        switch(_v)
        {
            case rocprofsys::process_mode::Trace: _s = "Trace"; break;
            case rocprofsys::process_mode::Sampling: _s = "Sampling"; break;
            case rocprofsys::process_mode::Causal: _s = "Causal"; break;
            case rocprofsys::process_mode::Coverage: _s = "Coverage"; break;
        }
        return fmt::formatter<std::string_view>::format(_s, ctx);
    }
};

template <>
struct fmt::formatter<rocprofsys::process_causal_mode> : fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(rocprofsys::process_causal_mode _v, FormatContext& ctx) const
    {
        std::string_view _s = {};
        switch(_v)
        {
            case rocprofsys::process_causal_mode::Line: _s = "Line"; break;
            case rocprofsys::process_causal_mode::Function: _s = "Function"; break;
        }
        return fmt::formatter<std::string_view>::format(_s, ctx);
    }
};

namespace rocprofsys
{
class config_policy
{
public:
    static bool get_debug_init();
};

// Storage is keyed on Policy (see storage() below) so basic_process_state<mock_policy>
// in tests never shares the common::static_object<..., Policy> instance with the
// production process_state alias.
template <typename Policy = config_policy>
class basic_process_state final
{
public:
    using State         = process_lifecycle_state;
    using Mode          = process_mode;
    using CausalBackend = process_causal_backend;
    using CausalMode    = process_causal_mode;

    basic_process_state()                                      = delete;
    basic_process_state(const basic_process_state&)            = delete;
    basic_process_state& operator=(const basic_process_state&) = delete;
    basic_process_state(basic_process_state&&)                 = delete;
    basic_process_state& operator=(basic_process_state&&)      = delete;

    [[gnu::hot]] static State get() noexcept
    {
        return storage().load(std::memory_order_relaxed);
    }

    [[gnu::cold]] static State set(State _n)
    {
        auto is_debug_init = Policy::get_debug_init();
        if(is_debug_init)
        {
            LOG_DEBUG("Setting state :: {} -> {}", get(), _n);
        }
        if(_n < get())
        {
            throw std::runtime_error(fmt::format(
                "State is being assigned to a lesser value :: {} -> {}", get(), _n));
        }
        auto _prior = get();
        storage().store(_n, std::memory_order_relaxed);
        return _prior;
    }

    [[gnu::cold]] static State reset()
    {
        auto is_debug_init = Policy::get_debug_init();
        if(is_debug_init)
        {
            LOG_DEBUG("Resetting state :: {} -> PreInit", get());
        }
        auto _prior = get();
        storage().store(State::PreInit, std::memory_order_relaxed);
        return _prior;
    }

private:
    static std::atomic<State>& storage() noexcept
    {
        static auto*& _v = common::static_object<std::atomic<State>, Policy>::construct(
            common::do_not_destroy{}, State::PreInit);
        return *_v;
    }
};

using process_state = basic_process_state<>;

class thread_state final
{
public:
    using State = thread_lifecycle_state;

    thread_state()                               = delete;
    thread_state(const thread_state&)            = delete;
    thread_state& operator=(const thread_state&) = delete;
    thread_state(thread_state&&)                 = delete;
    thread_state& operator=(thread_state&&)      = delete;

    [[gnu::hot]] static State get() noexcept { return current(); }

    [[gnu::hot]] static State set(State _n) noexcept
    {
        auto _prior = current();
        current()   = _n;
        return _prior;
    }

    [[gnu::hot]] static State push(State _v)
    {
        if(get() >= State::Completed) return get();
        return history().emplace_back(set(_v));
    }

    [[gnu::hot]] static State pop()
    {
        if(get() >= State::Completed) return get();
        auto& _hist = history();
        if(!_hist.empty())
        {
            set(_hist.back());
            _hist.pop_back();
        }
        return get();
    }

    class [[nodiscard]] scoped_guard
    {
    public:
        [[gnu::always_inline]] explicit scoped_guard(State _v) { thread_state::push(_v); }

        [[gnu::always_inline]] ~scoped_guard() { thread_state::pop(); }

        scoped_guard(const scoped_guard&)            = delete;
        scoped_guard& operator=(const scoped_guard&) = delete;
        scoped_guard(scoped_guard&&)                 = delete;
        scoped_guard& operator=(scoped_guard&&)      = delete;
    };

    [[nodiscard]] [[gnu::hot]] static scoped_guard scoped(State _v)
    {
        return scoped_guard{ _v };
    }

private:
    static State& current() noexcept
    {
        static thread_local auto _v = State::Enabled;
        return _v;
    }

    static std::vector<State>& history()
    {
        auto _idx = utility::get_thread_index();

        static auto _v = utility::get_filled_array<ROCPROFSYS_MAX_THREADS>(
            []() { return utility::get_reserved_vector<State>(32); });

        if(_idx >= ROCPROFSYS_MAX_THREADS)
        {
            static thread_local auto _tl_v = utility::get_reserved_vector<State>(32);
            return _tl_v;
        }

        return _v.at(_idx);
    }
};
}  // namespace rocprofsys
