/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once


#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "hrr/hrr_regions.h"

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
    size_t load(const std::string& archive_dir);

    bool   empty()   const { return records_.empty(); }
    size_t streams() const { return streams_; }
    size_t records() const { return records_.size(); }

    // Apply every record with mono_ns <= ts. Monotonic: a timestamp older than
    // the last one is ignored rather than rewinding the map. Use rewind() to
    void advance_to(PlaybackContext& ctx, int64_t ts);

    // Reset the live-set cursor so a second pass (the timed kernel-filter pass
    // after warm-up) re-applies the timeline from the start. Does not free
    void rewind();

    // Back the segment containing `rec_addr`, if one is declared live and has
    // not been backed already, and return the live address for `rec_addr`.
    void* materialize_for(PlaybackContext& ctx, uint64_t rec_addr);

    // Segments still backed when replay ends need no teardown here: they are
    // registered in PlaybackContext::alloc_map as ordinary device allocations,

    // Classify a recorded device address against the live set. On InBlock the
    // owning block's recorded base and size are returned through the
    Class classify(uint64_t rec_addr, uint64_t* blk_base, uint64_t* blk_size) const;

    // Bounds of the archive's own event timestamps, used to detect a producer
    // whose clock does not match amd::Os::timeNanos (CLOCK_MONOTONIC). Passing
    void check_clock_against(uint64_t first_event_ns, uint64_t last_event_ns);

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
    std::map<uint64_t, Extent> blocks_;
    std::map<uint64_t, Extent> segments_;

    // Segments this map allocated because HRR never saw them. Recorded base ->
    // live pointer; the entry is also registered in PlaybackContext::alloc_map
    std::map<uint64_t, void*> materialized_;
    size_t materialized_total_ = 0;  // cumulative, including released ones

    // Segments whose materialisation failed, so the allocation is attempted
    // once rather than on every pointer that lands in them.
    std::set<uint64_t> materialize_failed_;
};

}  // namespace hrr
