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

#pragma once

#include <cstdint>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
/**
 * @brief Plain description of a ring buffer's geometry.
 *
 * Pure value type — no ownership, no cursor state. Pass by value.
 * `size` must be a power of two and `mask == size - 1`.
 *
 * This is Layer 1 of the queue_intercept refactor: pure ring-buffer math
 * decoupled from the `QueueState` struct.
 */
struct RingView
{
    void*    buf      = nullptr;  ///< Base of the ring (packet array)
    uint32_t size     = 0;        ///< Packet slot count; power of 2
    uint32_t mask     = 0;        ///< size - 1
    uint32_t pkt_size = 64;       ///< Bytes per slot (64 for AQL, 256 for metadata)

    /// Pure read: returns the address of slot (logical_pos & mask).
    /// Does NOT advance any cursor and has no side effects.
    const void* read_slot(uint64_t logical_pos) const
    {
        return static_cast<const char*>(buf) + ((logical_pos & mask) * pkt_size);
    }
};

/**
 * @brief Stateful write cursor into a ring buffer.
 *
 * Tracks a `submit_pos` as packets are copied in. Uses `real_rdid` (the
 * hardware read-doorbell-id pointer) as a backpressure source: `write()`
 * spins until `submit_pos - *real_rdid < size` before copying the packet.
 *
 * Optionally tracks an exclusive upper bound (`reservation_end`) that
 * prevents `write()` from advancing beyond a reserved window. Defaults to
 * UINT64_MAX (no limit); call `set_reservation_end` to install a bound for
 * subsequent writes.
 *
 * NOT thread-safe. Callers in the queue_intercept pipeline serialize access
 * via `QueueState::gate_lock`.
 *
 * Usage:
 *   RingCursor cur{view, state->published_pos, state->real_rdid};
 *   cur.set_reservation_end(cur.submit_pos() + N);
 *   cur.write(pkt);      // returns bool; refuses if reservation exhausted
 *   cur.pad_to_reservation();
 *   state->published_pos = cur.submit_pos();
 */
class RingCursor
{
public:
    RingCursor(RingView view, uint64_t submit_pos, volatile const uint64_t* real_rdid);

    /// Copy `view_.pkt_size` bytes from `pkt` into `ring[submit_pos & mask]`.
    /// Waits for a free slot via `real_rdid` before writing.
    /// Returns true and advances submit_pos by 1 on success.
    /// Returns false (and does NOT advance submit_pos) if submit_pos has
    /// already reached `reservation_end`.
    bool write(const void* pkt);

    /// Write a zeroed packet with header = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE).
    /// Header is stored with `__ATOMIC_RELEASE` so the CP observes the full zeroed
    /// body before the type flips off INVALID. Advances submit_pos by 1.
    /// Not constrained by reservation_end.
    void write_gap();

    /// Fill remaining slots in the current reservation with gap packets
    /// (BARRIER_AND header, zeroed body) until submit_pos == reservation_end.
    /// No-op if already at or beyond reservation_end.
    void pad_to_reservation();

    /// Set the exclusive upper bound for subsequent `write()` calls. Any
    /// `write()` call with `submit_pos_ >= pos` will return false. Defaults
    /// to UINT64_MAX, which disables enforcement.
    void set_reservation_end(uint64_t pos) { reservation_end_ = pos; }

    /// Pure read through the underlying RingView (see RingView::read_slot).
    const void* read_slot(uint64_t logical_pos) const { return view_.read_slot(logical_pos); }

    uint64_t        submit_pos() const { return submit_pos_; }
    uint64_t        reservation_end() const { return reservation_end_; }
    const RingView& view() const { return view_; }

private:
    void wait_for_free_slot();

    RingView                 view_;
    uint64_t                 submit_pos_;
    volatile const uint64_t* real_rdid_;
    uint64_t                 reservation_end_ = UINT64_MAX;
};

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
