// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/static_object.hpp"
#include "logger/debug.hpp"
#include "utility.hpp"

#include <spdlog/fmt/fmt.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace rocprofsys::state
{

enum class process_lifecycle : std::uint16_t
{
    PreInit = 0,
    Init,
    Active,
    Finalized,
    Disabled,
};

enum class thread_lifecycle : std::uint16_t
{
    Enabled = 0,
    Internal,
    Completed,
    Disabled,
};
}  // namespace rocprofsys::state

namespace rocprofsys::mode
{

enum class process : std::uint16_t
{
    Trace = 0,
    Sampling,
    Causal,
    Coverage,
};

enum class process_causal : std::uint16_t
{
    Line = 0,
    Function,
};
}  // namespace rocprofsys::mode

namespace rocprofsys::backend
{
enum class causal : std::uint16_t
{
    Perf = 0,
    Timer,
    Auto,
};

}  // namespace rocprofsys::backend

template <>
struct fmt::formatter<rocprofsys::state::process_lifecycle>
: fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(rocprofsys::state::process_lifecycle pl_state, FormatContext& ctx) const
    {
        std::string_view str = {};
        switch(pl_state)
        {
            case rocprofsys::state::process_lifecycle::PreInit: str = "PreInit"; break;
            case rocprofsys::state::process_lifecycle::Init: str = "Init"; break;
            case rocprofsys::state::process_lifecycle::Active: str = "Active"; break;
            case rocprofsys::state::process_lifecycle::Disabled: str = "Disabled"; break;
            case rocprofsys::state::process_lifecycle::Finalized:
                str = "Finalized";
                break;
        }
        return fmt::formatter<std::string_view>::format(str, ctx);
    }
};

template <>
struct fmt::formatter<rocprofsys::state::thread_lifecycle>
: fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(rocprofsys::state::thread_lifecycle tl_state, FormatContext& ctx) const
    {
        std::string_view str = {};
        switch(tl_state)
        {
            case rocprofsys::state::thread_lifecycle::Enabled: str = "Enabled"; break;
            case rocprofsys::state::thread_lifecycle::Internal: str = "Internal"; break;
            case rocprofsys::state::thread_lifecycle::Completed: str = "Completed"; break;
            case rocprofsys::state::thread_lifecycle::Disabled: str = "Disabled"; break;
        }
        return fmt::formatter<std::string_view>::format(str, ctx);
    }
};

template <>
struct fmt::formatter<rocprofsys::mode::process> : fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(rocprofsys::mode::process p_mode, FormatContext& ctx) const
    {
        std::string_view str = {};
        switch(p_mode)
        {
            case rocprofsys::mode::process::Trace: str = "Trace"; break;
            case rocprofsys::mode::process::Sampling: str = "Sampling"; break;
            case rocprofsys::mode::process::Causal: str = "Causal"; break;
            case rocprofsys::mode::process::Coverage: str = "Coverage"; break;
        }
        return fmt::formatter<std::string_view>::format(str, ctx);
    }
};

template <>
struct fmt::formatter<rocprofsys::mode::process_causal> : fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(rocprofsys::mode::process_causal c_mode, FormatContext& ctx) const
    {
        std::string_view str = {};
        switch(c_mode)
        {
            case rocprofsys::mode::process_causal::Line: str = "Line"; break;
            case rocprofsys::mode::process_causal::Function: str = "Function"; break;
        }
        return fmt::formatter<std::string_view>::format(str, ctx);
    }
};

namespace rocprofsys::state
{

class process final
{
public:
    using State         = process_lifecycle;
    using Mode          = mode::process;
    using CausalBackend = backend::causal;
    using CausalMode    = mode::process_causal;

    // Explicit aliases instead of `using enum` — GCC added `using enum`
    // support only in GCC 11; the CI matrix still builds with GCC 10.3.
    static constexpr State PreInit   = State::PreInit;
    static constexpr State Init      = State::Init;
    static constexpr State Active    = State::Active;
    static constexpr State Finalized = State::Finalized;
    static constexpr State Disabled  = State::Disabled;

    process()                          = delete;
    process(const process&)            = delete;
    process& operator=(const process&) = delete;
    process(process&&)                 = delete;
    process& operator=(process&&)      = delete;
    ~process()                         = default;

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
        storage().store(PreInit, std::memory_order_relaxed);
        LOG_DEBUG("Resetting state :: {} -> PreInit", get());
        return last_state;
    }

private:
    static std::atomic<State>& storage() noexcept
    {
        static auto*& atomic_state = common::static_object<std::atomic<State>>::construct(
            common::do_not_destroy{}, PreInit);
        return *atomic_state;
    }
};

class thread final
{
public:
    using State = thread_lifecycle;

    // Explicit aliases instead of `using enum` — GCC added `using enum`
    // support only in GCC 11; the CI matrix still builds with GCC 10.3.
    static constexpr State Enabled   = State::Enabled;
    static constexpr State Internal  = State::Internal;
    static constexpr State Completed = State::Completed;
    static constexpr State Disabled  = State::Disabled;

    thread()                         = delete;
    thread(const thread&)            = delete;
    thread& operator=(const thread&) = delete;
    thread(thread&&)                 = delete;
    thread& operator=(thread&&)      = delete;
    ~thread()                        = default;

    [[gnu::hot]] static State get() noexcept { return current(); }

    [[gnu::hot]] static State set(State state_to_set) noexcept
    {
        auto last_state = current();
        current()       = state_to_set;
        return last_state;
    }

    [[gnu::hot]] static State push(State state_to_push)
    {
        if(get() >= Completed)
        {
            return get();
        }
        return history().emplace_back(set(state_to_push));
    }

    [[gnu::hot]] static State pop()
    {
        if(get() >= Completed)
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
            thread::push(state_to_push);
        }

        [[gnu::always_inline]] ~scoped_guard() { thread::pop(); }

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
        static thread_local auto current_state = Enabled;
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
}  // namespace rocprofsys::state
