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
    size_t      size   = 0;
    const void* source = nullptr;
    void*       dest   = nullptr;
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

    friend struct packet_builder;
};

// Submit all regions as raw AQL packets: three dispatches that cut them into 16 KiB tiles on the
// GPU, one that copies the tiles a wave each, and a barrier on its completion. The completion
// signal, argument buffer and tile workspace are reused from per-agent state after the dependent
// target kernel drains.
hsa_status_t
copy(const hsa::Queue&                     queue,
     hsa_amd_queue_intercept_packet_writer writer,
     const std::vector<copy_region_t>&     regions,
     std::vector<packet_info>&             pending);

// Like copy(), and the tile list stays on the GPU for the restores below until another tile list
// replaces it. With `digests`, the copy also sums every tile it reads into a digest for
// restore_changed(), restore_listed() and check_unlisted(); without room for them, the capture
// keeps just its list.
hsa_status_t
capture(const hsa::Queue&                     queue,
        hsa_amd_queue_intercept_packet_writer writer,
        const std::vector<copy_region_t>&     regions,
        bool                                  digests,
        std::vector<packet_info>&             pending);

// Whether the tile list on this queue's agent is a capture's, and whether it has digests.
bool
has_kept_list(const hsa::Queue& queue);

bool
has_digests(const hsa::Queue& queue);

// Forgets that tile list. A replay calls this before its capture, which copies nothing when there
// is nothing to capture, so it never restores from an older capture's list.
void
forget_kept_list(const hsa::Queue& queue);

// Copies every tile of that capture back, one dispatch over its list.
hsa_status_t
restore_all(const hsa::Queue&                     queue,
            hsa_amd_queue_intercept_packet_writer writer,
            std::vector<packet_info>&             pending);

// Hashes every tile of that capture, copies back the ones whose digest changed, and keeps a list
// of them. Each call makes a new list.
hsa_status_t
restore_changed(const hsa::Queue&                     queue,
                hsa_amd_queue_intercept_packet_writer writer,
                std::vector<packet_info>&             pending);

// Copies back the tiles on the last list, without hashing.
hsa_status_t
restore_listed(const hsa::Queue&                     queue,
               hsa_amd_queue_intercept_packet_writer writer,
               std::vector<packet_info>&             pending);

// Counts the tiles off the last list whose digest changed anyway: restores from the list missed
// them.
hsa_status_t
check_unlisted(const hsa::Queue&                     queue,
               hsa_amd_queue_intercept_packet_writer writer,
               std::vector<packet_info>&             pending);

struct digest_counts_t
{
    uint32_t tiles            = 0;
    uint32_t changed          = 0;  // on the last list
    uint32_t unlisted_changed = 0;  // found by check_unlisted()
};

// Reads the counts after the restores and the check complete.
std::optional<digest_counts_t>
read_digest_counts(const hsa::Queue& queue);
}  // namespace blit
}  // namespace kernel_replay
}  // namespace rocprofiler
