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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/rocprofiler-sdk/kfd/dlog_drain.hpp"
#include "lib/rocprofiler-sdk/kfd/record_chunk_pool.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <new>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace kfd
{
// In-band reader marker: one fw_record slot written by one stream call and
// published with its pass. Firmware writes only record_type 0, 1 and 2, so this
// value cannot come from the ring.
constexpr uint32_t kRecReaderMarker = 0x524B4D31;  // 'RKM1'

// Decoded form of the marker. `seen_at_ns` is the pass-start clock, sampled
// before the segment's copies, and is the eviction watermark for the segment.
struct reader_marker
{
    uint64_t epoch      = 0;
    uint64_t seen_at_ns = 0;
};

// The slot carries {ts_lo, ts_hi} = epoch and {dispatch_id, doorbell_off} =
// seen_at_ns (low, high).
inline fw_record
encode_marker(const reader_marker& m) noexcept
{
    return fw_record{static_cast<uint32_t>(m.epoch),
                     static_cast<uint32_t>(m.epoch >> 32),
                     kRecReaderMarker,
                     static_cast<uint32_t>(m.seen_at_ns),
                     static_cast<uint32_t>(m.seen_at_ns >> 32)};
}

// Inverse of encode_marker. Asserts the type: firmware writes only record_type
// 0, 1 and 2, so a foreign record with the marker value is a firmware ABI break.
inline reader_marker
decode_marker(const fw_record& s) noexcept
{
    assert(s.record_type == kRecReaderMarker);
    reader_marker m;
    m.epoch = static_cast<uint64_t>(s.ts_lo) | (static_cast<uint64_t>(s.ts_hi) << 32);
    m.seen_at_ns =
        static_cast<uint64_t>(s.dispatch_id) | (static_cast<uint64_t>(s.doorbell_off) << 32);
    return m;
}

// Owner sizing: chunks for one maximal ring pass plus its closing marker, from
// any current fill. ring_bytes/20 slots total, minus the R wrap exclusions,
// plus the one marker slot.
constexpr size_t
dlog_pass_reserve(uint64_t ring_bytes, uint32_t regions, size_t k) noexcept
{
    const uint64_t per_region = ring_bytes / (uint64_t{20} * regions);
    const uint64_t slots      = regions * (per_region - 1) + 1;
    return static_cast<size_t>((slots + k - 1) / k);
}

// Arrivals during one maximal 1 s processor gap at the 1.8M rec/s design rate:
// ceil(1.8e6 / kRecords) = 110, plus the one marker slot.
inline constexpr size_t kDlogGapReserveChunks = 111;

inline size_t
dlog_spare_target(uint64_t ring_bytes, uint32_t regions) noexcept
{
    return dlog_pass_reserve(ring_bytes, regions, basic_record_chunk_pool<>::chunk::kRecords) +
           kDlogGapReserveChunks;
}

// Why the reader allocated at all: a sizing defect or a slow consumer, counted
// and never silent. min_spares is the smallest spares_available() observed at a
// pass start; SIZE_MAX until the first pass. Published relaxed: the reader is
// the only writer, and teardown reads both for its warning.
struct reader_alloc_log
{
    std::atomic<uint64_t> reader_allocs = {0};
    std::atomic<size_t>   min_spares    = {SIZE_MAX};
};

// One marker-delimited batch: exactly today's batch for pair_records.
struct stream_segment
{
    std::vector<fw_record> records    = {};
    uint64_t               end_pos    = 0;
    uint64_t               epoch      = 0;
    uint64_t               seen_at_ns = 0;
};

// Processor-private backlog for one stream: data runs bulk-insert into the
// open segment, every marker slot closes it. Allocation failure here terminates
// the process through copy_out's noexcept.
class stream_backlog
{
public:
    // Slots at [pos, pos + n) were published; ingest them in order. Marker
    // slots are handled one at a time; pure data runs are bulk-inserted.
    void absorb(const fw_record* p, size_t n, uint64_t pos)
    {
        size_t i = 0;
        while(i < n)
        {
            size_t j = i;
            while(j < n && p[j].record_type != kRecReaderMarker)
                ++j;
            if(j > i) open.records.insert(open.records.end(), p + i, p + j);
            if(j < n)
            {
                close(decode_marker(p[j]), pos + j + 1);
                ++j;
            }
            i = j;
        }
    }

    // process_unit returns a paired segment's vector here; absorb starts the
    // next open segment from the free list so backlog growth stays flat.
    void recycle_vector(std::vector<fw_record>&& v)
    {
        if(recycled.size() < kMaxRecycledVectors)
        {
            v.clear();
            recycled.push_back(std::move(v));
        }
    }

    std::deque<stream_segment>          closed;
    stream_segment                      open;
    std::vector<std::vector<fw_record>> recycled;

private:
    static constexpr size_t kMaxRecycledVectors = 4;

    // A marker always closes the open segment; an idle flush closes the empty
    // one, which pair_records handles exactly like base's empty batch. The new
    // open segment reuses a recycled vector (moved out before pairing).
    void close(const reader_marker& m, uint64_t end_pos)
    {
        stream_segment seg;
        seg.records    = std::move(open.records);
        seg.end_pos    = end_pos;
        seg.epoch      = m.epoch;
        seg.seen_at_ns = m.seen_at_ns;
        closed.push_back(std::move(seg));
        stream_segment next;
        if(!recycled.empty())
        {
            next.records = std::move(recycled.back());
            recycled.pop_back();
        }
        open = std::move(next);
    }
};

// One GPU's SPSC chain of chunks: raw firmware records plus reader markers,
// reader -> processor. Ownership: the stream owns every chunk from m_head along
// `next` to m_tail; `next` is non-owning; spares belong to m_pool.
// preallocate/reset/abandon_in_child only while no worker touches the stream.
//
// MEMORY ORDERING: the reader's release publish makes all slot bytes below the
// position visible; the release link of chunk::next publishes the successor's
// base_pos; the processor's release publish_processed publishes paired records.
// Everything else is single-writer.
template <typename Mapping = chunk_mapping>
class basic_record_stream
{
public:
    using pool_t    = basic_record_chunk_pool<Mapping>;
    using chunk     = typename pool_t::chunk;
    using chunk_ptr = typename pool_t::chunk_ptr;

    static constexpr size_t kK = chunk::kRecords;

    // 32 + 111 = 143 chunks covers every accepted region count, including R = 1.
    static_assert(dlog_pass_reserve(kDlogMaxRingBytes, 1, kK) + kDlogGapReserveChunks <=
                      pool_t::kCapacity,
                  "spare target of the largest ring must fit the pool's spare ring");

    // ---- setup / teardown: no worker may touch the stream ----

    basic_record_stream() = default;

    // Both workers are joined or never started.
    ~basic_record_stream() { reset(); }

    basic_record_stream(const basic_record_stream&) = delete;
    basic_record_stream& operator=(const basic_record_stream&) = delete;

    // Current chunk + spare_target spares; on any failure frees all, false.
    // reset() first: re-preallocation must not leak the old chain or keep stale
    // positions, and a first-map failure must leave a clean stream behind.
    bool preallocate(size_t spare_target) noexcept
    {
        reset();
        auto first = pool_t::allocate();
        if(!first) return false;
        first->base_pos  = 0;
        chunk* const raw = first.get();
        m_head           = raw;
        m_tail           = raw;
        m_write_pos      = 0;
        m_taken_pos      = 0;
        // The chain owns the current chunk from here on: release it so a later
        // failure's reset() (or the destructor) frees it exactly once.
        first.release();
        for(size_t i = 0; i < spare_target; ++i)
        {
            auto c = pool_t::allocate();
            if(!c || !m_pool.push(std::move(c)))
            {
                reset();
                return false;
            }
        }
        m_armed.store(true, std::memory_order_release);
        return true;
    }

    bool armed() const noexcept { return m_armed.load(std::memory_order_acquire); }

    // Frees the chain and the spares; back to the default state.
    void reset() noexcept
    {
        chunk* c = m_head;
        while(c != nullptr)
        {
            chunk* const next = c->next.load(std::memory_order_relaxed);
            typename pool_t::chunk_deleter{}(c);
            c = next;
        }
        m_head = m_tail = nullptr;
        m_write_pos     = 0;
        m_taken_pos     = 0;
        m_pool.clear();
        m_armed.store(false, std::memory_order_relaxed);
        m_published.store(0, std::memory_order_relaxed);
        m_processed.store(0, std::memory_order_relaxed);
    }

    // Fork child: forget every chunk without munmap or any chunk access, in
    // case a link is half-written.
    void abandon_in_child() noexcept
    {
        m_head      = nullptr;
        m_tail      = nullptr;
        m_write_pos = 0;
        m_taken_pos = 0;
        m_pool.abandon();
    }

    // ---- producer: reader thread only ----

    // Slots left in the current tail chunk.
    size_t room() const noexcept
    {
        assert(m_tail != nullptr);
        return m_tail->base_pos + kK - m_write_pos;
    }

    // Appends n records from a byte-continuous source, linking at room() == 0.
    // Byte-addressed: the ring is an external mapping, not a fw_record array.
    // Throws std::bad_alloc when a needed link cannot be allocated.
    void append(const void* src, size_t n, reader_alloc_log& log)
    {
        assert(m_tail != nullptr);
        const auto* bytes = static_cast<const uint8_t*>(src);
        while(n > 0)
        {
            if(room() == 0) link_next(log);
            const size_t k = std::min(n, room());
            std::memcpy(
                &m_tail->slots[m_write_pos - m_tail->base_pos], bytes, k * sizeof(fw_record));
            m_write_pos += k;
            bytes += k * sizeof(fw_record);
            n -= k;
        }
    }

    // Writes the one marker slot through the same link path as append.
    void append_marker(const reader_marker& m, reader_alloc_log& log)
    {
        const fw_record slot = encode_marker(m);
        append(&slot, 1, log);
    }

    // Publishes everything written so far; no store at all when nothing moved.
    void publish() noexcept
    {
        if(m_published.load(std::memory_order_relaxed) != m_write_pos)
            m_published.store(m_write_pos, std::memory_order_release);
    }

    size_t spares_available() const noexcept { return m_pool.available(); }

    // ---- consumer: processor thread only ----

    // Feeds every published, untaken run to on_run(pos) in order, runs split
    // only at chunk boundaries, then retires fully-taken linked chunks (pushed
    // while occupancy() < spare_target, else freed; never before on_run
    // returns). A throw from on_run terminates the process.
    template <typename OnRun>
    void copy_out(OnRun&& on_run, size_t spare_target) noexcept
    {
        const uint64_t target = m_published.load(std::memory_order_acquire);
        chunk*         c      = m_head;
        while(c != nullptr)
        {
            const uint64_t chunk_end = c->base_pos + kK;
            const uint64_t upto      = std::min(target, chunk_end);
            if(m_taken_pos < upto)
            {
                on_run(&c->slots[m_taken_pos - c->base_pos],
                       static_cast<size_t>(upto - m_taken_pos),
                       m_taken_pos);
                m_taken_pos = upto;
            }
            if(m_taken_pos < chunk_end) break;  // chunk not fully taken: stays head
            chunk* const next = c->next.load(std::memory_order_acquire);
            if(next == nullptr) break;  // an unlinked chunk is never retired
            c->next.store(nullptr, std::memory_order_relaxed);
            chunk_ptr retiring{c};
            if(m_pool.occupancy() < spare_target)
                m_pool.push(std::move(retiring));  // a full ring frees instead
            m_head = next;
            c      = next;
        }
    }

    void publish_processed(uint64_t pos) noexcept
    {
        assert(pos <= m_taken_pos);
        m_processed.store(pos, std::memory_order_release);
    }

    uint64_t taken() const noexcept { return m_taken_pos; }

    // ---- any thread ----

    uint64_t published() const noexcept { return m_published.load(std::memory_order_acquire); }

    uint64_t processed() const noexcept { return m_processed.load(std::memory_order_acquire); }

    // ---- tests only: both workers parked at a barrier ----

    size_t chain_length_quiescent() const noexcept
    {
        size_t n = 0;
        chunk* c = m_head;
        while(c != nullptr)
        {
            ++n;
            c = c->next.load(std::memory_order_relaxed);
        }
        return n;
    }

    size_t spares_quiescent() const noexcept { return m_pool.occupancy(); }

    uint64_t write_pos_quiescent() const noexcept { return m_write_pos; }

private:
    // Links a spare (or allocates) at room()==0; throws std::bad_alloc if
    // neither works.
    void link_next(reader_alloc_log& log)
    {
        chunk_ptr c = m_pool.pop();
        if(!c)
        {
            c = pool_t::allocate();
            if(!c) throw std::bad_alloc();
            log.reader_allocs.fetch_add(1, std::memory_order_relaxed);
        }
        c->base_pos = m_tail->base_pos + kK;  // the link happens exactly at the boundary
        m_tail->next.store(c.get(), std::memory_order_release);
        m_tail = c.release();
    }

    chunk*            m_tail      = nullptr;  // reader: last linked chunk
    uint64_t          m_write_pos = 0;        // reader-private
    chunk*            m_head      = nullptr;  // processor-private (set by preallocate)
    uint64_t          m_taken_pos = 0;        // processor-private
    pool_t            m_pool      = {};
    std::atomic<bool> m_armed     = {false};
    alignas(64) std::atomic<uint64_t> m_published = {0};
    alignas(64) std::atomic<uint64_t> m_processed = {0};
};

using record_stream = basic_record_stream<>;

// One reader pass over one session's ring: sample the clock, update min_spares,
// drain the ring into the stream, close the pass with the one-slot marker
// {epoch, seen_at_ns} when records were copied or a flush is pending, and
// publish once. Every publish therefore ends at a marker, so the published prefix
// never ends inside a segment. SessionT needs: info.num_regions,
// info.region_record_count, cursors, stream, alloc_log and
// pairing_flush_pending. `now_ns` is sampled by the caller (the pass-start
// clock, as base does) so lap telemetry uses the same instant. Returns the
// records copied.
template <typename SessionT>
uint64_t
dlog_pass(const uint8_t*       records_base,
          const volatile void* wptr_base,
          SessionT&            s,
          uint64_t             drain_epoch,
          uint64_t             now_ns)
{
    const uint64_t _epoch  = drain_epoch + 1;
    const size_t   _spares = s.stream.spares_available();
    if(_spares < s.alloc_log.min_spares.load(std::memory_order_relaxed))
        s.alloc_log.min_spares.store(_spares, std::memory_order_relaxed);

    const auto _copied = copy_pipes(records_base,
                                    s.info.num_regions,
                                    s.info.region_record_count,
                                    wptr_base,
                                    s.cursors,
                                    s.stream,
                                    s.alloc_log);

    // copied > 0 closes a DATA marker; copied == 0 with a pending flush closes
    // the idle-flush marker; nothing pending appends nothing.
    if(_copied > 0 || s.pairing_flush_pending)
        s.stream.append_marker({_epoch, now_ns}, s.alloc_log);
    s.pairing_flush_pending = _copied > 0;
    s.stream.publish();
    return _copied;
}

}  // namespace kfd
}  // namespace rocprofiler
