// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/static_object.hpp"
#include "logger/debug.hpp"
#include "utility.hpp"

#include <spdlog/fmt/bundled/base.h>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/fmt/fmt.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
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
    auto format(rocprofsys::process_lifecycle_state pl_state, FormatContext& ctx) const
    {
        std::string_view str = {};
        switch(pl_state)
        {
            case rocprofsys::process_lifecycle_state::PreInit: str = "PreInit"; break;
            case rocprofsys::process_lifecycle_state::Init: str = "Init"; break;
            case rocprofsys::process_lifecycle_state::Active: str = "Active"; break;
            case rocprofsys::process_lifecycle_state::Disabled: str = "Disabled"; break;
            case rocprofsys::process_lifecycle_state::Finalized: str = "Finalized"; break;
        }
        return fmt::formatter<std::string_view>::format(str, ctx);
    }
};

template <>
struct fmt::formatter<rocprofsys::thread_lifecycle_state>
: fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(rocprofsys::thread_lifecycle_state tl_state, FormatContext& ctx) const
    {
        std::string_view str = {};
        switch(tl_state)
        {
            case rocprofsys::thread_lifecycle_state::Enabled: str = "Enabled"; break;
            case rocprofsys::thread_lifecycle_state::Internal: str = "Internal"; break;
            case rocprofsys::thread_lifecycle_state::Completed: str = "Completed"; break;
            case rocprofsys::thread_lifecycle_state::Disabled: str = "Disabled"; break;
        }
        return fmt::formatter<std::string_view>::format(str, ctx);
    }
};

template <>
struct fmt::formatter<rocprofsys::process_mode> : fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(rocprofsys::process_mode p_mode, FormatContext& ctx) const
    {
        std::string_view str = {};
        switch(p_mode)
        {
            case rocprofsys::process_mode::Trace: str = "Trace"; break;
            case rocprofsys::process_mode::Sampling: str = "Sampling"; break;
            case rocprofsys::process_mode::Causal: str = "Causal"; break;
            case rocprofsys::process_mode::Coverage: str = "Coverage"; break;
        }
        return fmt::formatter<std::string_view>::format(str, ctx);
    }
};

template <>
struct fmt::formatter<rocprofsys::process_causal_mode> : fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(rocprofsys::process_causal_mode c_mode, FormatContext& ctx) const
    {
        std::string_view str = {};
        switch(c_mode)
        {
            case rocprofsys::process_causal_mode::Line: str = "Line"; break;
            case rocprofsys::process_causal_mode::Function: str = "Function"; break;
        }
        return fmt::formatter<std::string_view>::format(str, ctx);
    }
};

namespace rocprofsys
{

class process_state final
{
public:
    using State         = process_lifecycle_state;
    using Mode          = process_mode;
    using CausalBackend = process_causal_backend;
    using CausalMode    = process_causal_mode;

    process_state()                                = delete;
    process_state(const process_state&)            = delete;
    process_state& operator=(const process_state&) = delete;
    process_state(process_state&&)                 = delete;
    process_state& operator=(process_state&&)      = delete;
    ~process_state()                               = default;

    [[gnu::hot]] static State get() noexcept
    {
        return storage().load(std::memory_order_relaxed);
    }

    [[gnu::cold]] static State set(State state_to_set)
    {
        auto last_state = get();
        if(state_to_set < last_state)
        {
            throw std::runtime_error(
                fmt::format("State is being assigned to a lesser value :: {} -> {}",
                            last_state, state_to_set));
        }
        storage().store(state_to_set, std::memory_order_relaxed);
        LOG_DEBUG("Setting state :: {} -> {}", last_state, state_to_set);
        return last_state;
    }

    [[gnu::cold]] static State reset()
    {
        auto last_state = get();
        storage().store(State::PreInit, std::memory_order_relaxed);
        LOG_DEBUG("Resetting state :: {} -> PreInit", get());
        return last_state;
    }

private:
    static std::atomic<State>& storage() noexcept
    {
        static auto*& atomic_state = common::static_object<std::atomic<State>>::construct(
            common::do_not_destroy{}, State::PreInit);
        return *atomic_state;
    }
};

class thread_state final
{
public:
    using State = thread_lifecycle_state;

    thread_state()                               = delete;
    thread_state(const thread_state&)            = delete;
    thread_state& operator=(const thread_state&) = delete;
    thread_state(thread_state&&)                 = delete;
    thread_state& operator=(thread_state&&)      = delete;
    ~thread_state()                              = default;

    [[gnu::hot]] static State get() noexcept { return current(); }

    [[gnu::hot]] static State set(State state_to_set) noexcept
    {
        auto last_state = current();
        current()       = state_to_set;
        return last_state;
    }

    [[gnu::hot]] static State push(State state_to_push)
    {
        if(get() >= State::Completed)
        {
            return get();
        }
        return history().emplace_back(set(state_to_push));
    }

    [[gnu::hot]] static State pop()
    {
        if(get() >= State::Completed)
        {
            return get();
        }
        auto& state_history = history();
        if(!state_history.empty())
        {
            set(state_history.back());
            state_history.pop_back();
        }
        return get();
    }

    class [[nodiscard]] scoped_guard
    {
    public:
        [[gnu::always_inline]] explicit scoped_guard(State state_to_push)
        {
            thread_state::push(state_to_push);
        }

        [[gnu::always_inline]] ~scoped_guard() { thread_state::pop(); }

        scoped_guard(const scoped_guard&)            = delete;
        scoped_guard& operator=(const scoped_guard&) = delete;
        scoped_guard(scoped_guard&&)                 = delete;
        scoped_guard& operator=(scoped_guard&&)      = delete;
    };

    [[nodiscard]] [[gnu::hot]] static scoped_guard scoped(State state_to_set)
    {
        return scoped_guard{ state_to_set };
    }

private:
    static State& current() noexcept
    {
        static thread_local auto current_state = State::Enabled;
        return current_state;
    }

    static std::vector<State>& history()
    {
        auto thread_index = utility::get_thread_index();

        static auto state_history_array =
            utility::get_filled_array<ROCPROFSYS_MAX_THREADS>(
                []() { return utility::get_reserved_vector<State>(32); });

        if(thread_index >= ROCPROFSYS_MAX_THREADS)
        {
            static thread_local auto local_vector =
                utility::get_reserved_vector<State>(32);
            return local_vector;
        }

        return state_history_array.at(thread_index);
    }
};
}  // namespace rocprofsys
