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

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
class AgentCache;
class Queue;
}  // namespace hsa

namespace kernel_replay
{
namespace blit
{
struct copy_region_t
{
    void*       dst  = nullptr;
    const void* src  = nullptr;
    size_t      size = 0;
};

hsa_status_t
prepare(const hsa::Queue& queue);

class packet_info
{
public:
    packet_info();
    ~packet_info();

    packet_info(const packet_info&) = delete;
    packet_info& operator=(const packet_info&) = delete;
    packet_info(packet_info&&) noexcept;
    packet_info& operator=(packet_info&&) noexcept;

    const void*  get_raw();
    uint64_t     packet_count() const;
    hsa_status_t wait();
    hsa_status_t retire();

private:
    struct implementation;
    explicit packet_info(std::unique_ptr<implementation> impl);

    std::unique_ptr<implementation> m_impl;

    friend std::optional<packet_info> create(const hsa::Queue&, const std::vector<copy_region_t>&);
};

std::optional<packet_info>
create(const hsa::Queue& queue, const std::vector<copy_region_t>& regions);

// Submit all GPU-backed regions through one mixed-size raw AQL blit packet. The completion signal
// and kernarg storage are reused from per-agent state after the dependent target kernel drains.
hsa_status_t
copy(const hsa::Queue&                     queue,
     hsa_amd_queue_intercept_packet_writer writer,
     const std::vector<copy_region_t>&     regions,
     std::vector<packet_info>&             pending);
}  // namespace blit
}  // namespace kernel_replay
}  // namespace rocprofiler
