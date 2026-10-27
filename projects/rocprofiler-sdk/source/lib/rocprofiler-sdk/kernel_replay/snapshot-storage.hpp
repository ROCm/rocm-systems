// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/kernel_replay/snapshot-plan.hpp"

#include <cstddef>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace rocprofiler
{
namespace kernel_replay
{
namespace memory_snapshot
{
namespace storage
{
enum class error_code : uint8_t
{
    none,
    out_of_resources,
    hard_failure,
    invalid_state,
};

struct commit_result
{
    error_code error         = error_code::none;
    size_t     storage_index = std::numeric_limits<size_t>::max();

    explicit operator bool() const { return error == error_code::none; }
};

template <typename Handle>
struct allocation_result
{
    std::optional<Handle> value{};
    error_code            error = error_code::none;

    explicit operator bool() const { return value.has_value(); }
};

template <typename Policy, typename BackingAllocator>
class arena
{
public:
    using policy_type            = Policy;
    using backing_allocator_t    = BackingAllocator;
    using handle_type            = typename backing_allocator_t::handle_type;

    static constexpr bool has_backing = true;
    static constexpr bool is_terminal = false;

    // The handle and its allocator move as one ownership unit.
    static_assert(std::is_nothrow_move_constructible<backing_allocator_t>::value,
                  "backing allocators must be nothrow move constructible");
    static_assert(std::is_nothrow_move_constructible<handle_type>::value,
                  "backing handles must be nothrow move constructible");

    template <typename... Args,
              std::enable_if_t<std::is_constructible<backing_allocator_t, Args...>::value, int> = 0>
    explicit arena(Policy policy, Args&&... args)
    : m_policy{std::move(policy)}
    , m_allocator{std::forward<Args>(args)...}
    , m_state{state::planning}
    {}

    planning::result_t reserve(size_t size, size_t alignment)
    {
        if(m_state != state::planning)
            return planning::result_t{
                .error = planning::error_t{planning::error_code::invalid_state, size}};
        return m_policy.reserve(size, alignment);
    }

    size_t used() const { return m_policy.used(); }

    commit_result commit()
    {
        if(m_state != state::planning) return commit_result{.error = error_code::invalid_state};
        if(used() == 0)
        {
            m_state = state::committed;
            return {};
        }

        auto result = m_allocator.allocate(used());
        if(!result)
            return commit_result{.error = result.error == error_code::none
                                              ? error_code::hard_failure
                                              : result.error};
        m_backing = std::move(result.value);
        m_state   = state::committed;
        return {};
    }

    std::optional<handle_type> release_backing()
    {
        if(m_state != state::committed) return std::nullopt;
        m_state = state::released;
        return std::exchange(m_backing, std::nullopt);
    }

    void reset() noexcept
    {
        if(m_backing)
        {
            m_allocator.release(std::move(*m_backing));
            m_backing.reset();
        }
        m_policy.reset();
        m_state = state::planning;
    }

    ~arena() { reset(); }

    arena(const arena&) = delete;
    arena& operator=(const arena&) = delete;

    arena(arena&& rhs) noexcept(std::is_nothrow_move_constructible<Policy>::value&&
                                    std::is_nothrow_move_constructible<backing_allocator_t>::value&&
                                        std::is_nothrow_move_constructible<handle_type>::value)
    : m_policy{std::move(rhs.m_policy)}
    , m_allocator{std::move(rhs.m_allocator)}
    , m_state{std::exchange(rhs.m_state, state::planning)}
    {
        if(rhs.m_backing)
        {
            m_backing.emplace(std::move(*rhs.m_backing));
            rhs.m_backing.reset();
        }
    }

    arena& operator=(arena&&) = delete;

private:
    enum class state
    {
        planning,
        committed,
        released,
    };

    Policy                     m_policy;
    backing_allocator_t        m_allocator;
    std::optional<handle_type> m_backing{};
    state                      m_state = state::planning;
};

class null_allocator
{
public:
    static constexpr bool has_backing = false;
    static constexpr bool is_terminal = true;

    planning::result_t reserve(size_t size, size_t)
    {
        return planning::result_t{
            .error = planning::error_t{planning::error_code::no_backing_capacity, size}};
    }

    size_t        used() const { return 0; }
    commit_result commit() { return {}; }
    void          reset() noexcept {}
};

template <typename... Allocators>
class fallback_allocator
{
public:
    using tuple_type = std::tuple<Allocators...>;

    static constexpr size_t allocator_count = sizeof...(Allocators);

    template <size_t Index>
    using allocator_type = std::tuple_element_t<Index, tuple_type>;

    static_assert(allocator_count > 0, "a fallback allocator needs a terminal allocator");
    static_assert(allocator_type<allocator_count - 1>::is_terminal,
                  "the final fallback allocator must be terminal");

    explicit fallback_allocator(Allocators... allocators)
    : m_allocators{std::move(allocators)...}
    {}

    planning::result_t reserve(size_t requested_size, size_t alignment)
    {
        return reserve_from<0>(requested_size, alignment);
    }

    commit_result commit() { return commit_from<0>(); }

    template <size_t Index>
    auto& get()
    {
        return std::get<Index>(m_allocators);
    }

    template <size_t Index>
    const auto& get() const
    {
        return std::get<Index>(m_allocators);
    }

    void reset() noexcept { reset_from<allocator_count>(); }

private:
    template <size_t Index>
    planning::result_t reserve_from(size_t requested_size, size_t alignment)
    {
        static_assert(Index < allocator_count, "the terminal allocator must stop fallback");

        auto result = std::get<Index>(m_allocators).reserve(requested_size, alignment);
        if(result)
        {
            result->storage_index = Index;
            return result;
        }
        using current_allocator = allocator_type<Index>;
        if constexpr(current_allocator::is_terminal)
        {
            return result;
        }
        else
        {
            if(result.error.code != planning::error_code::exhausted) return result;
            return reserve_from<Index + 1>(requested_size, alignment);
        }
    }

    template <size_t Index>
    commit_result commit_from()
    {
        if constexpr(Index == allocator_count)
        {
            return {};
        }
        else
        {
            auto result = std::get<Index>(m_allocators).commit();
            if(!result)
            {
                result.storage_index = Index;
                reset();
                return result;
            }
            return commit_from<Index + 1>();
        }
    }

    template <size_t Count>
    void reset_from() noexcept
    {
        if constexpr(Count > 0)
        {
            std::get<Count - 1>(m_allocators).reset();
            reset_from<Count - 1>();
        }
    }

    tuple_type m_allocators;
};
}  // namespace storage
}  // namespace memory_snapshot
}  // namespace kernel_replay
}  // namespace rocprofiler
