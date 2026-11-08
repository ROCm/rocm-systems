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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace rocprofiler
{
namespace kernel_replay
{
namespace memory_snapshot
{
namespace planning
{
// A piece of one region to place. Lower priority values are placed first. Callers that need
// non-contiguous placement, such as hot or dirty pages, emit several requests for one region.
struct request_t
{
    size_t   region_index  = 0;
    size_t   region_offset = 0;
    size_t   size          = 0;
    uint32_t priority      = 0;
};

// The bytes one tier took from the front of a request, and where they start in its backing.
struct reservation_t
{
    size_t size           = 0;
    size_t backing_offset = 0;
};

struct placement_t
{
    size_t region_index   = 0;
    size_t region_offset  = 0;
    size_t size           = 0;
    size_t tier           = 0;
    size_t backing_offset = 0;
};

// Admission policies return how many bytes from the front of a request one tier takes, at most
// `available`. Anything they leave is offered to the next tier.

// The request fits whole or moves on to the next tier.
struct whole_region
{
    size_t admit(const request_t& request, size_t available) const
    {
        return request.size <= available ? request.size : 0;
    }
};

// The largest prefix that fits. The split point is rounded down to a granule in region
// coordinates, so both sides stay aligned for DMA and for the blit's 16-byte items.
struct prefix_chunking
{
    size_t granularity = 4096;

    size_t admit(const request_t& request, size_t available) const
    {
        if(request.size <= available) return request.size;
        const auto granule = std::max<size_t>(granularity, 1);
        const auto split   = (request.region_offset + available) / granule * granule;
        return split > request.region_offset ? split - request.region_offset : 0;
    }
};

// Places every request through `storage` in priority order. On failure the storage is reset.
template <typename Storage>
std::optional<std::vector<placement_t>>
plan(Storage& storage, std::vector<request_t> requests)
{
    std::sort(requests.begin(), requests.end(), [](const request_t& lhs, const request_t& rhs) {
        return std::tie(lhs.priority, lhs.region_index, lhs.region_offset) <
               std::tie(rhs.priority, rhs.region_index, rhs.region_offset);
    });

    auto placements = std::vector<placement_t>{};
    placements.reserve(requests.size());
    for(const auto& request : requests)
    {
        if(!storage.place(request, placements))
        {
            storage.reset();
            return std::nullopt;
        }
    }
    return placements;
}

// Tier capacities are advisory until backing exists. If a tier cannot materialize what it was
// given, it is disabled and every request is planned again.
template <typename Storage>
std::optional<std::vector<placement_t>>
plan_and_commit(Storage& storage, const std::vector<request_t>& requests)
{
    for(size_t attempt = 0; attempt < Storage::tier_count; ++attempt)
    {
        auto placements = plan(storage, requests);
        if(!placements) return std::nullopt;

        const auto failed = storage.commit();
        if(!failed) return placements;
        storage.disable(*failed);
    }
    return std::nullopt;
}
}  // namespace planning
}  // namespace memory_snapshot
}  // namespace kernel_replay
}  // namespace rocprofiler
