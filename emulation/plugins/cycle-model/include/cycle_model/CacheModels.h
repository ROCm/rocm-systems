// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file CacheModels.h
/// @brief Shared timing primitives used by both the per-CU MemorySystem (L1) and the shared
/// SharedMemModel (L2/HBM): a set-associative timing cache (simdojo RuntimeSetAssocTags +
/// per-line fill-readiness), and a sustained-throughput bandwidth queue.

#pragma once

#include "cycle_model/UarchConfig.h"
#include "simdojo/components/runtime_set_assoc_tags.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cycle_model {

/// One cache level for the timing model: residency + LRU come from simdojo's
/// RuntimeSetAssocTags; this adds the *timing overlay* the mechanism deliberately
/// omits — a per-line fill-readiness cycle (`ready_cyc`) plus the level's latency
/// parameters. `ready_cyc` is the cycle the line's fill returns: probes before it
/// coalesce onto the in-flight miss (complete at ready_cyc); probes at/after it are
/// real hits. Indexed by the stable slot index the tag store returns.
struct TimedTagCache {
  simdojo::RuntimeSetAssocTags tags_;
  std::vector<uint64_t>        ready_cyc_;          // slot-indexed; sized to tags_.slot_count()
  uint32_t line_bytes = 0, hit_latency = 0, miss_to_next = 0;

  void init(const CacheSpec& s) {
    line_bytes = s.line_bytes; hit_latency = s.hit_latency; miss_to_next = s.miss_to_next_level;
    // Unconfigured geometry (e.g. a flat-latency test config) -> no modeled cache.
    if (!s.size_kb || !s.ways || !s.line_bytes) { tags_.configure(0, 0, 0); ready_cyc_.clear(); return; }
    uint32_t sets = (s.size_kb * 1024u) / (s.ways * s.line_bytes);
    tags_.configure(sets, s.ways, s.line_bytes);
    ready_cyc_.assign(tags_.slot_count(), 0);
  }
  bool      configured() const { return tags_.configured(); }
  int       lookup(uint64_t line_base) const { return tags_.lookup(line_base); }   // slot or -1, NO touch
  bool      present(uint64_t line_base) const { return tags_.present(line_base); }
  uint32_t  victim(uint64_t line_base) const { return tags_.victim(line_base); }    // slot, NO touch/install
  void      install(uint32_t slot, uint64_t line_base) { tags_.install(slot, line_base); }  // touches
  void      touch(uint32_t slot) { tags_.touch(slot); }                       // hit-path MRU promote
  uint64_t& ready(uint32_t slot) { return ready_cyc_[slot]; }
  void      reset() { tags_.reset(); std::fill(ready_cyc_.begin(), ready_cyc_.end(), 0); }
};

/// A bandwidth-limited resource (the L2 port, or one HBM channel) modeled as a single
/// availability watermark with sub-cycle byte accumulation.
///
/// CONTRACT — this is a *sustained-throughput* model (bytes/cycle), NOT a per-
/// transaction minimum-occupancy model. A transaction of `bytes` arriving at `arrive`
/// starts at max(next_free, arrive); its bytes accumulate in `rem` and advance
/// next_free by floor(rem / bytes_per_cycle) whole cycles, carrying the partial-cycle
/// remainder forward. So:
///   - A sub-rate transaction (bytes < bytes_per_cycle) advances next_free by 0 — many
///     small transactions in the same cycle are free until their *aggregate* bytes
///     exceed bytes_per_cycle, at which point serialization kicks in. This is the
///     intended aggregate-bandwidth behavior, not a missing per-packet occupancy bug.
///   - An idle gap (arrive > next_free) discards the carried remainder (the resource
///     drained while idle) and restarts accumulation at `arrive`.
/// Per-transaction *latency* (channel access / DRAM latency) is added by the caller on
/// top of the value returned here; this queue models only occupancy/throughput.
struct BwQueue {
  uint64_t next_free = 0;
  uint64_t rem = 0;                  // accumulated sub-cycle bytes (carried remainder)
  uint32_t bytes_per_cycle = 1;

  uint64_t schedule(uint64_t arrive, uint64_t bytes) {
    uint64_t start = arrive > next_free ? arrive : next_free;
    if (arrive > next_free) rem = 0;             // channel was idle -> partial work drained
    rem += bytes;
    uint64_t cyc = rem / bytes_per_cycle;
    rem %= bytes_per_cycle;
    next_free = start + cyc;
    return next_free;                            // service_done
  }
  void reset() { next_free = 0; rem = 0; }       // bytes_per_cycle (config) is preserved
};

/// One coalesced cache line that missed/bypassed L1 and must be serviced by the
/// shared L2/HBM hierarchy. `arrive_cyc` is when the request reaches L2 (now for
/// UC/NT, now + L1 hit_latency for a cold L1 miss). `skip_l2` = UC (bypass L2).
struct SharedReq {
  uint64_t line_base = 0;
  uint64_t arrive_cyc = 0;
  bool     skip_l2 = false;
};

}  // namespace cycle_model
