/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

/*
 * hrr_region_map.h — replay-side consumer of external region annotations.
 *
 * Loads every .hrrr sidecar in an archive's regions/ directory (format in
 * hrr_regions.h), merges them into one timestamp-ordered stream, and replays
 * that stream in lockstep with the captured events so that at any point during
 * playback the map holds the set of regions that were live at the corresponding
 * instant of the original run.
 *
 * Two things come out of that:
 *
 *   - Bounds. A kernel-argument pointer that lands inside a captured segment
 *     but inside no live block is an intra-segment out-of-bounds or stale
 *     pointer. Replay cannot see this on its own, because the segment is one
 *     contiguous allocation and the block layout inside it never crossed a HIP
 *     API. classify() reports it.
 *   - Reachability. A segment the producer declared but that HRR never observed
 *     (HIP bypassed: direct HSA allocation, a foreign VMM pool, imported
 *     memory) is materialised, so pointers into it translate instead of
 *     reaching a kernel as null.
 *
 * Ordering assumes a totally ordered replay stream, so the map is used only in
 * single-threaded replay (the default). It is inert under --multi-thread.
 */

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "hrr_regions.h"

struct PlaybackContext;

namespace hrr {

class RegionMap {
  public:
    // Where a recorded address falls relative to the currently live regions.
    enum class Class {
        None,             // not inside any annotated segment
        InBlock,          // inside a live block — the expected case
        InSegmentNoBlock  // inside a segment, in no live block: OOB or stale
    };

    // Load and merge every regions/*.hrrr under `archive_dir`. Returns the
    // number of records loaded; 0 (with streams()==0) is the ordinary case for
    // a capture that ran without an external producer.
    size_t load(const std::string& archive_dir);

    bool   empty()   const { return records_.empty(); }
    size_t streams() const { return streams_; }
    size_t records() const { return records_.size(); }

    // Apply every record with mono_ns <= ts. Monotonic: a timestamp older than
    // the last one is ignored rather than rewinding the map. Use rewind() to
    // reset the cursor for a second pass. Records stamped mono_ns == 0 are the
    // producer's baseline and are applied on first call.
    void advance_to(PlaybackContext& ctx, int64_t ts);

    // Reset the live-set cursor so a second pass (the timed kernel-filter pass
    // after warm-up) re-applies the timeline from the start. Does not free
    // materialised buffers; those stay registered in PlaybackContext::alloc_map.
    void rewind();

    // Back the segment containing `rec_addr`, if one is declared live and has
    // not been backed already, and return the live address for `rec_addr`.
    // Returns nullptr when no live segment contains it or the allocation fails.
    //
    // Deliberately lazy. A producer declares its segments uniformly, because it
    // cannot know which of its allocations crossed a HIP API — and a segment's
    // ADD record generally arrives *before* replay has reached the hipMalloc
    // that created it, so asking at that moment whether the base already
    // resolves gets the wrong answer for every captured segment. Waiting until a
    // pointer has actually failed to translate removes the ambiguity: by then
    // every allocation the archive knows about has been replayed, so failing to
    // translate is proof that HIP was bypassed.
    void* materialize_for(PlaybackContext& ctx, uint64_t rec_addr);

    // Segments still backed when replay ends need no teardown here: they are
    // registered in PlaybackContext::alloc_map as ordinary device allocations,
    // so the replayer's normal cleanup frees them exactly once. Only a segment
    // the producer explicitly freed mid-run is released early, by advance_to.

    // Classify a recorded device address against the live set. On InBlock the
    // owning block's recorded base and size are returned through the
    // out-parameters, which is what the block guard needs to relocate it.
    Class classify(uint64_t rec_addr, uint64_t* blk_base, uint64_t* blk_size) const;

    // Bounds of the archive's own event timestamps, used to detect a producer
    // whose clock does not match amd::Os::timeNanos (CLOCK_MONOTONIC). Passing
    // an empty span disables the check.
    void check_clock_against(uint64_t first_event_ns, uint64_t last_event_ns);

    // ---- Reporting ----
    size_t blocks_live()   const { return blocks_.size(); }
    size_t segments_live() const { return segments_.size(); }
    size_t segments_materialized() const { return materialized_total_; }
    size_t applied()       const { return cursor_; }

  private:
    void apply(PlaybackContext& ctx, const hrr_region_rec& r);

    std::vector<hrr_region_rec> records_;  // merged, stable-sorted by mono_ns
    size_t  streams_  = 0;
    size_t  cursor_   = 0;
    int64_t last_ts_  = INT64_MIN;

    // What a live region is, beyond the base it is keyed on.
    struct Extent {
        uint64_t size;
        uint8_t  device;  // ordinal the producer recorded the region on
    };

    // Live sets, keyed by recorded base. std::map so classify() can find the
    // tightest enclosing entry with upper_bound instead of a linear scan.
    //
    // Keyed by base alone, not by (device, base). Sidecars are read from one
    // archive's regions/ directory, so every record in the map comes from a
    // single recorded process, and within a process a device VA identifies its
    // allocation regardless of which GPU it lives on — two devices are never
    // handed the same address. The ordinal therefore cannot disambiguate a
    // lookup; what it does decide is where a materialised segment is placed,
    // which is where materialize_for uses it.
    std::map<uint64_t, Extent> blocks_;
    std::map<uint64_t, Extent> segments_;

    // Segments this map allocated because HRR never saw them. Recorded base ->
    // live pointer; the entry is also registered in PlaybackContext::alloc_map
    // so ordinary pointer translation resolves into it.
    std::map<uint64_t, void*> materialized_;
    size_t materialized_total_ = 0;  // cumulative, including released ones

    // Segments whose materialisation failed, so the allocation is attempted
    // once rather than on every pointer that lands in them.
    std::set<uint64_t> materialize_failed_;
};

}  // namespace hrr
