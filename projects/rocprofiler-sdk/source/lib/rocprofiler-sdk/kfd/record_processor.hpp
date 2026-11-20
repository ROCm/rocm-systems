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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

// Processor-side mechanics for the dispatch-log chunk stream (copy-out into
// per-stream backlogs, marker-delimited segment units, the fence / GC
// predicates), factored out of kfd_reader.cpp so unit tests can drive them
// without a GPU. Streams expose .stream and .info(buffer_size, num_regions).
//
// RE-ENTRANCY: copy-out may re-enter from process_unit's EOP-flush hook, but it
// touches only streams and backlogs -- the segment under pairing was moved out
// beforehand, and recycle_vector runs only after pairing returns.

#include "lib/rocprofiler-sdk/kfd/dlog_drain.hpp"
#include "lib/rocprofiler-sdk/kfd/record_stream.hpp"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace rocprofiler
{
namespace kfd
{
// Retained-START aging cadence: an unmatched START belongs to a dispatch whose
// EOP never arrived (or a live kernel); eviction is throttled per GPU by the
// segment's own copy time, never the wall clock.
inline constexpr uint64_t kProcessorEvictIntervalNs = 1'000'000'000ull;  // 1 s

// Copy every armed stream's published runs into its backlog (quarantined
// streams included: they keep draining until reset). Asserts the invariant:
// every publish ends at a marker, so no open segment survives a copy-out.
template <typename Streams, typename Backlogs>
void
copy_out_all(Streams& streams, Backlogs& backlogs, size_t count)
{
    for(size_t i = 0; i < count; ++i)
    {
        auto& _stream = streams[i].stream;
        if(!_stream.armed()) continue;
        const size_t _target =
            dlog_spare_target(streams[i].info.buffer_size, streams[i].info.num_regions);
        _stream.copy_out(
            [&](const fw_record* p, size_t n, uint64_t pos) { backlogs[i].absorb(p, n, pos); },
            _target);
        assert(backlogs[i].open.records.empty());
    }
}

// One segment popped from one backlog's closed deque.
struct proc_unit
{
    bool           found = false;
    size_t         slot  = 0;
    stream_segment seg   = {};
};

// Round-robin over the slots with a non-empty closed deque and pop_front
// (move). The segment leaves the backlog before pairing, so a service during
// pairing cannot overwrite it.
template <typename Backlogs>
proc_unit
next_unit(Backlogs& backlogs, size_t count, size_t& rr)
{
    if(count == 0) return {};
    for(size_t k = 0; k < count; ++k)
    {
        const size_t i = (rr + k) % count;
        if(backlogs[i].closed.empty()) continue;
        rr = (i + 1) % count;
        proc_unit u;
        u.found = true;
        u.slot  = i;
        u.seg   = std::move(backlogs[i].closed.front());
        backlogs[i].closed.pop_front();
        return u;
    }
    return {};
}

// One segment = one base batch. Pairs under the segment's marker epoch, then
// final_flush (the caller's buffered-EOP resolution) runs BEFORE anything is
// published, so a fence that observes processed >= end_pos can never pass while
// an EOP sits unresolved in the caller's buffer.
template <typename Stream, typename OnRecord, typename FinalFlush, typename PublishDeferred>
void
process_unit(Stream&           stream,
             stream_backlog&   backlog,
             pair_state&       pairing,
             stream_segment&   seg,
             uint64_t          start_max_age_ns,
             OnRecord&&        on_record,
             FinalFlush&&      final_flush,
             PublishDeferred&& publish_deferred)
{
    // Evict stale retained STARTs BEFORE pairing, watermarked by THIS segment's
    // own copy time (never the wall clock), throttled per GPU. Ordering it
    // ahead of pairing keeps a START the watermark has condemned from binding
    // this segment's recycled EOP first.
    if(seg.seen_at_ns - pairing.last_evict_ns >= kProcessorEvictIntervalNs)
    {
        pairing.starts_evicted += pairing.evict_stale(seg.seen_at_ns, start_max_age_ns);
        pairing.last_evict_ns = seg.seen_at_ns;
    }

    pair_records(
        seg.records.data(),
        seg.records.size(),
        pairing,
        seg.seen_at_ns,
        [&](const drained_record& rec) { on_record(rec); },
        /*lookahead=*/true,
        seg.epoch);

    // Buffered EOPs resolve BEFORE anything is published: the fence reads
    // processed before the epoch, so an unresolved EOP behind a published
    // position would be drained early by signal_less_disable_permanently().
    final_flush();
    // Held epoch published BEFORE the processed position, so a fence that
    // observes processed >= end_pos also sees this unit's epoch store.
    publish_deferred();
    stream.publish_processed(seg.end_pos);
    // Only after pairing returns: a service during pairing takes vectors from
    // the free list, so nothing pair_records reads may be overwritten.
    backlog.recycle_vector(std::move(seg.records));
}

// GC gate: closed-window GC may run only when the reader is between passes and
// every armed stream is fully copied out and fully decoded. Pure over loaded
// values; the seq_cst load feeding reader_pass_open (loaded first) and the
// session_count acquire live at the call site. Held EOP keys stay excluded
// through the caller's held_eop_keys().
template <typename Streams, typename Backlogs>
bool
gc_gate_open_streams(bool            reader_pass_open,
                     const Streams&  streams,
                     const Backlogs& backlogs,
                     size_t          count)
{
    if(reader_pass_open) return false;
    for(size_t i = 0; i < count; ++i)
    {
        const auto& _stream = streams[i].stream;
        if(!_stream.armed()) continue;
        if(_stream.taken() != _stream.published() || !backlogs[i].closed.empty()) return false;
    }
    return true;
}

// The GC gate with its load discipline in one place: reader_pass_open FIRST
// (seq_cst), then the session count (acquire). Sampling the count first could
// miss a session published between the two loads, opening the gate over an
// undrained stream.
template <typename PassOpen, typename Count, typename Streams, typename Backlogs>
bool
gc_gate_open_pass(const PassOpen& reader_pass_open,
                  const Streams&  streams,
                  const Backlogs& backlogs,
                  const Count&    session_count)
{
    const bool   _pass_open = reader_pass_open.load(std::memory_order_seq_cst);
    const size_t _n         = session_count.load(std::memory_order_acquire);
    return gc_gate_open_streams(_pass_open, streams, backlogs, _n);
}

// Fence Stage 2 over streams: has the processor processed every published
// position captured in `target` AND released every record held for lookahead
// from drain passes <= through_epoch? Newer traffic cannot prolong the wait: it
// is not below any captured target, and held epochs only rise. Read processed
// BEFORE the held epoch: the processor stores the epoch before
// publish_processed, so a processed >= target observation pairs with the epoch
// store of the same unit.
template <typename Streams>
bool
processor_drained_through(const Streams&               streams,
                          size_t                       count,
                          const uint64_t*              target,
                          const std::atomic<uint64_t>& deferred_epoch,
                          uint64_t                     through_epoch)
{
    for(size_t i = 0; i < count; ++i)
    {
        const auto& _stream = streams[i].stream;
        if(!_stream.armed()) continue;
        if(_stream.processed() < target[i]) return false;
    }
    const auto pending = deferred_epoch.load(std::memory_order_acquire);
    return pending == UINT64_MAX || pending > through_epoch;
}

// After reader_done, nothing is published again, and one copy-out takes
// everything published (it walks to published and stops only at an unlinked
// tail); then pair every closed segment. `process_one(slot, seg)` pairs one
// unit.
template <typename Streams, typename Backlogs, typename ProcessOne>
void
drain_for_exit(Streams& streams, Backlogs& backlogs, size_t count, ProcessOne&& process_one)
{
    copy_out_all(streams, backlogs, count);
    size_t _rr = 0;
    for(auto _u = next_unit(backlogs, count, _rr); _u.found; _u = next_unit(backlogs, count, _rr))
        process_one(_u.slot, _u.seg);
}

}  // namespace kfd
}  // namespace rocprofiler
