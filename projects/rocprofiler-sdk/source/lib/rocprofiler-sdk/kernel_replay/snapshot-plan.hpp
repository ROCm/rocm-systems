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

#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace kernel_replay
{
namespace memory_snapshot
{
namespace planning
{
enum class error_code : uint8_t
{
    none,
    exhausted,
    no_backing_capacity,
    overflow,
    metadata_failure,
};

struct error_t
{
    error_code code         = error_code::none;
    size_t     requested    = 0;
    size_t     region_index = std::numeric_limits<size_t>::max();
};

struct slice_t
{
    size_t       size   = 0;
    size_t       offset = 0;
    storage_kind tier   = storage_kind::none;
};

struct result_t
{
    std::optional<slice_t> value{};
    error_t                error{};

    explicit       operator bool() const { return value.has_value(); }
    const slice_t& operator*() const { return *value; }
    const slice_t* operator->() const { return &*value; }
};

template <storage_kind Tier>
class bounded_allocator
{
public:
    explicit bounded_allocator(size_t capacity)
    : m_capacity{capacity}
    {}

    result_t allocate(size_t size, size_t alignment)
    {
        if(alignment == 0) return result_t{.error = error_t{error_code::overflow, size}};

        const auto remainder = m_used % alignment;
        const auto padding   = remainder == 0 ? size_t{0} : alignment - remainder;
        if(m_used > std::numeric_limits<size_t>::max() - padding)
            return result_t{.error = error_t{error_code::overflow, size}};
        const auto offset = m_used + padding;
        if(size > std::numeric_limits<size_t>::max() - offset)
            return result_t{.error = error_t{error_code::overflow, size}};
        const auto end = offset + size;
        if(end > m_capacity) return result_t{.error = error_t{error_code::exhausted, size}};

        m_used = end;
        return result_t{.value = slice_t{size, offset, Tier}};
    }

    size_t used() const { return m_used; }

private:
    size_t m_capacity = 0;
    size_t m_used     = 0;
};

class null_allocator
{
public:
    result_t allocate(size_t size, size_t)
    {
        return result_t{.error = error_t{error_code::no_backing_capacity, size}};
    }

    size_t used() const { return 0; }
};

template <typename... Allocators>
class fallback_allocator
{
public:
    explicit fallback_allocator(Allocators... allocators)
    : m_allocators{std::move(allocators)...}
    {}

    result_t allocate(size_t size, size_t alignment) { return allocate_from<0>(size, alignment); }

    template <size_t Index>
    auto& get()
    {
        return std::get<Index>(m_allocators);
    }

private:
    template <size_t Index>
    result_t allocate_from(size_t size, size_t alignment)
    {
        if constexpr(Index == sizeof...(Allocators))
        {
            return result_t{.error = error_t{error_code::no_backing_capacity, size}};
        }
        else
        {
            auto result = std::get<Index>(m_allocators).allocate(size, alignment);
            if(result || result.error.code != error_code::exhausted) return result;
            return allocate_from<Index + 1>(size, alignment);
        }
    }

    std::tuple<Allocators...> m_allocators;
};

struct request_t
{
    size_t    size         = 0;
    size_t    region_index = 0;
    uintptr_t order_key    = 0;
};

struct placement_t
{
    size_t       size           = 0;
    size_t       region_index   = 0;
    storage_kind tier           = storage_kind::none;
    size_t       backing_offset = 0;
};

struct plan_t
{
    std::vector<placement_t> placements{};
    size_t                   gpu_bytes  = 0;
    size_t                   host_bytes = 0;
    error_t                  error{};

    bool complete() const { return error.code == error_code::none; }
};

template <typename Allocator>
plan_t
build(std::vector<request_t> requests, Allocator& allocator, size_t alignment)
{
    auto plan = plan_t{};
    try
    {
        std::sort(requests.begin(), requests.end(), [](const auto& lhs, const auto& rhs) {
            if(lhs.order_key != rhs.order_key) return lhs.order_key < rhs.order_key;
            return lhs.region_index < rhs.region_index;
        });
        plan.placements.reserve(requests.size());

        for(const auto& request : requests)
        {
            auto result = allocator.allocate(request.size, alignment);
            if(!result)
            {
                plan.error              = result.error;
                plan.error.region_index = request.region_index;
                return plan;
            }
            plan.placements.emplace_back(
                placement_t{request.size, request.region_index, result->tier, result->offset});
        }
    } catch(const std::bad_alloc&)
    {
        plan.error = error_t{error_code::metadata_failure};
        return plan;
    }

    plan.gpu_bytes  = allocator.template get<0>().used();
    plan.host_bytes = allocator.template get<1>().used();
    return plan;
}

using gpu_allocator      = bounded_allocator<storage_kind::gpu_local>;
using pinned_allocator   = bounded_allocator<storage_kind::pinned_host>;
using snapshot_allocator = fallback_allocator<gpu_allocator, pinned_allocator, null_allocator>;
}  // namespace planning
}  // namespace memory_snapshot
}  // namespace kernel_replay
}  // namespace rocprofiler
