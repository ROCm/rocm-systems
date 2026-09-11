// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/static_object.hpp"
#include "logger/debug.hpp"
#include "utility.hpp"

#include <fmt/format.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace rocprofsys::state
{

enum class process_lifecycle : std::uint16_t
{
    pre_init = 0,
    init,
    active,
    finalized,
    disabled,
};

enum class thread_lifecycle : std::uint16_t
{
    enabled = 0,
    internal,
    completed,
    disabled,
};
}  // namespace rocprofsys::state

namespace rocprofsys::mode
{

enum class process : std::uint16_t
{
    trace = 0,
    sampling,
    causal,
    coverage,
};

enum class process_causal : std::uint16_t
{
    line = 0,
    function,
};
}  // namespace rocprofsys::mode

namespace rocprofsys::backend
{
enum class causal : std::uint16_t
{
    perf = 0,
    timer,
    automatic,
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
            case rocprofsys::state::process_lifecycle::pre_init: str = "PreInit"; break;
            case rocprofsys::state::process_lifecycle::init: str = "Init"; break;
            case rocprofsys::state::process_lifecycle::active: str = "Active"; break;
            case rocprofsys::state::process_lifecycle::disabled: str = "Disabled"; break;
            case rocprofsys::state::process_lifecycle::finalized:
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
            case rocprofsys::state::thread_lifecycle::enabled: str = "Enabled"; break;
            case rocprofsys::state::thread_lifecycle::internal: str = "Internal"; break;
            case rocprofsys::state::thread_lifecycle::completed: str = "Completed"; break;
            case rocprofsys::state::thread_lifecycle::disabled: str = "Disabled"; break;
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
            case rocprofsys::mode::process::trace: str = "Trace"; break;
            case rocprofsys::mode::process::sampling: str = "Sampling"; break;
            case rocprofsys::mode::process::causal: str = "Causal"; break;
            case rocprofsys::mode::process::coverage: str = "Coverage"; break;
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
            case rocprofsys::mode::process_causal::line: str = "Line"; break;
            case rocprofsys::mode::process_causal::function: str = "Function"; break;
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
    static constexpr State PreInit   = State::pre_init;
    static constexpr State Init      = State::init;
    static constexpr State Active    = State::active;
    static constexpr State Finalized = State::finalized;
    static constexpr State Disabled  = State::disabled;

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
        LOG_DEBUG("Resetting state :: {} -> PreInit", last_state);
        storage().store(PreInit, std::memory_order_relaxed);
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
    static constexpr State Enabled   = State::enabled;
    static constexpr State Internal  = State::internal;
    static constexpr State Completed = State::completed;
    static constexpr State Disabled  = State::disabled;

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

    struct alignas(64) padded_history_t
    {
        std::vector<State> entries;
    };

    static std::vector<State>& history()
    {
        auto thread_index = utility::get_thread_index();

        static auto state_history_array =
            utility::get_filled_array<ROCPROFSYS_MAX_THREADS>([]() {
                return padded_history_t{ utility::get_reserved_vector<State>(32) };
            });

        if(thread_index >= ROCPROFSYS_MAX_THREADS)
        {
            static thread_local auto local_vector =
                utility::get_reserved_vector<State>(32);
            return local_vector;
        }

        return state_history_array.at(thread_index).entries;
    }
};
}  // namespace rocprofsys::state
