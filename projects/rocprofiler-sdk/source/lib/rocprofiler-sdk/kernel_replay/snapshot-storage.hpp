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

#include <algorithm>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace kernel_replay
{
namespace memory_snapshot
{
namespace storage
{
// Packs ranges back to back, so the backing only has to cover the high-water mark. `used` never
// exceeds `capacity`, which keeps the offset arithmetic free of overflow.
template <typename BackingAllocator>
struct bump_suballocator
{
    using handle_type = typename BackingAllocator::handle_type;

    size_t                     capacity  = 0;
    size_t                     alignment = 1;
    BackingAllocator           backing;
    size_t                     used   = 0;
    std::optional<handle_type> handle = {};

    size_t padding() const
    {
        const auto remainder = used % alignment;
        return remainder == 0 ? 0 : alignment - remainder;
    }

    size_t available() const
    {
        const auto free_bytes = capacity - used;
        return free_bytes > padding() ? free_bytes - padding() : 0;
    }

    size_t reserve(size_t size)
    {
        const auto offset = used + padding();
        used              = offset + size;
        return offset;
    }

    bool commit()
    {
        if(used == 0) return true;
        handle = backing.allocate(used);
        return handle.has_value();
    }

    void reset()
    {
        handle.reset();
        used = 0;
    }

    size_t                     backing_size() const { return used; }
    std::optional<handle_type> release_backing() { return std::exchange(handle, std::nullopt); }
};

// One tier. The admission policy decides how much of a request to take; the suballocator decides
// where it goes and obtains the backing at commit.
template <typename Admission, typename Suballocator>
struct arena
{
    static constexpr bool terminal = false;

    Admission    admission;
    Suballocator suballocator;
    bool         enabled = true;

    std::optional<planning::reservation_t> reserve(const planning::request_t& request)
    {
        if(!enabled) return std::nullopt;
        const auto available = suballocator.available();
        // Clamped so a misbehaving policy cannot overrun the arena.
        const auto size = std::min({admission.admit(request, available), request.size, available});
        if(size == 0) return std::nullopt;
        return planning::reservation_t{size, suballocator.reserve(size)};
    }

    bool commit() { return suballocator.commit(); }
    void reset() { suballocator.reset(); }
    void disable() { enabled = false; }
};

// Anything that reaches this tier has nowhere to go, so planning fails.
struct null_tier
{
    static constexpr bool terminal = true;

    std::optional<planning::reservation_t> reserve(const planning::request_t&)
    {
        return std::nullopt;
    }

    bool commit() { return true; }
    void reset() {}
    void disable() {}
};

// Offers what is left of each request to the tiers in order, ending in the null tier.
template <typename... Tiers>
struct fallback
{
    static constexpr size_t tier_count = sizeof...(Tiers);
    static_assert(std::tuple_element_t<tier_count - 1, std::tuple<Tiers...>>::terminal,
                  "the last tier must be the null tier");

    std::tuple<Tiers...> tiers;

    explicit fallback(Tiers... t)
    : tiers{std::move(t)...}
    {}

    // A tier is asked again until it declines, so it can take one request in several pieces.
    // Returns false when bytes are left over at the null tier.
    bool place(planning::request_t request, std::vector<planning::placement_t>& placements)
    {
        for_each([&](auto& tier, size_t index) {
            while(request.size > 0)
            {
                const auto taken = tier.reserve(request);
                if(!taken) break;
                placements.push_back(planning::placement_t{request.region_index,
                                                           request.region_offset,
                                                           taken->size,
                                                           index,
                                                           taken->backing_offset});
                request.region_offset += taken->size;
                request.size -= taken->size;
            }
        });
        return request.size == 0;
    }

    // On failure, releases every tier and returns the index of the one that could not
    // materialize its backing.
    std::optional<size_t> commit()
    {
        auto failed = std::optional<size_t>{};
        for_each([&](auto& tier, size_t index) {
            if(!failed && !tier.commit()) failed = index;
        });
        if(failed) reset();
        return failed;
    }

    void reset()
    {
        for_each([](auto& tier, size_t) { tier.reset(); });
    }

    void disable(size_t index)
    {
        for_each([index](auto& tier, size_t tier_index) {
            if(tier_index == index) tier.disable();
        });
    }

    // Calls `fn(tier_index, size, handle)` for every committed tier, moving the backing out of the
    // storage.
    template <typename Fn>
    void release_backings(Fn&& fn)
    {
        for_each([&](auto& tier, size_t index) {
            if constexpr(!std::decay_t<decltype(tier)>::terminal)
            {
                if(auto handle = tier.suballocator.release_backing())
                    fn(index, tier.suballocator.backing_size(), std::move(*handle));
            }
        });
    }

    template <typename Fn>
    void for_each(Fn&& fn)
    {
        std::apply(
            [&](auto&... tier) {
                size_t index = 0;
                (fn(tier, index++), ...);
            },
            tiers);
    }
};
}  // namespace storage
}  // namespace memory_snapshot
}  // namespace kernel_replay
}  // namespace rocprofiler
