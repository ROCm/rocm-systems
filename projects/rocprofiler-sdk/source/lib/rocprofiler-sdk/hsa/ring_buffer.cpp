// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hsa/ring_buffer.hpp"

#include <hsa/hsa.h>

#include <cstring>
#include <thread>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
RingCursor::RingCursor(RingView view, uint64_t submit_pos, volatile const uint64_t* real_rdid)
: view_{view}
, submit_pos_{submit_pos}
, real_rdid_{real_rdid}
{}

void
RingCursor::wait_for_free_slot()
{
    while(true)
    {
        auto real_rdid = __atomic_load_n(real_rdid_, __ATOMIC_ACQUIRE);
        auto ring_used = submit_pos_ - real_rdid;
        if(ring_used < view_.size)
        {
            return;
        }
        std::this_thread::yield();
    }
}

bool
RingCursor::write(const void* pkt)
{
    if(submit_pos_ >= reservation_end_)
    {
        // Reservation exhausted; refuse to overrun into a foreign window.
        return false;
    }
    wait_for_free_slot();
    auto        slot = submit_pos_ & view_.mask;
    auto*       dst  = static_cast<char*>(view_.buf) + (slot * view_.pkt_size);
    const auto* src  = static_cast<const char*>(pkt);
    if(dst != src) memcpy(dst, src, view_.pkt_size);
    ++submit_pos_;
    return true;
}

void
RingCursor::write_gap()
{
    wait_for_free_slot();
    auto  slot = submit_pos_ & view_.mask;
    auto* dst  = static_cast<char*>(view_.buf) + (slot * view_.pkt_size);
    // Zero the whole slot first, then release-store the header so the CP
    // never observes a stale header with a zero body (or vice versa).
    memset(dst, 0, view_.pkt_size);
    uint16_t header = static_cast<uint16_t>(HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE);
    __atomic_store_n(reinterpret_cast<uint16_t*>(dst), header, __ATOMIC_RELEASE);
    ++submit_pos_;
}

void
RingCursor::pad_to_reservation()
{
    while(submit_pos_ < reservation_end_)
    {
        write_gap();
    }
}

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
