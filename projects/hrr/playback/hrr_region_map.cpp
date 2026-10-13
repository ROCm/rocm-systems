/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hrr_region_map.h"

#include "hip_playback.h"
#include "hrr_reader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace hrr {

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

size_t RegionMap::load(const std::string& archive_dir) {
    const std::vector<std::string> paths = find_region_streams(archive_dir);
    if (paths.empty()) return 0;

    for (const auto& path : paths) {
        FILE* f = open_record_stream(path, HRR_REGION_MAGIC, HRR_REGION_VERSION, nullptr);
        if (!f) continue;  // logged by open_record_stream; other streams still load

        size_t before = records_.size();
        std::vector<uint8_t> raw;
        while (true) {
            const RecordStatus st = read_raw_record(f, raw);
            if (st == RecordStatus::EndOfStream) break;
            if (st == RecordStatus::Torn) {
                // A producer that died mid-write leaves exactly one partial
                // batch at the tail. Everything before it is intact.
                fprintf(stderr, "[HRR] regions: torn tail in %s — kept %zu records\n",
                        path.c_str(), records_.size() - before);
                break;
            }
            if (raw.size() < sizeof(hrr_region_batch)) continue;
            const auto* b = reinterpret_cast<const hrr_region_batch*>(raw.data());
            if (b->hdr.event_type != HRR_REGION_EVENT) continue;

            // Trust the record count only as far as the bytes actually present.
            const size_t avail = (raw.size() - sizeof(hrr_region_batch)) /
                                 sizeof(hrr_region_rec);
            const size_t n = std::min<size_t>(b->n, avail);
            const auto* recs = reinterpret_cast<const hrr_region_rec*>(
                raw.data() + sizeof(hrr_region_batch));
            records_.insert(records_.end(), recs, recs + n);
        }
        fclose(f);
        ++streams_;
    }

    // Stable so that records sharing a timestamp — a producer emitting a free
    // and the reallocation of the same address in one batch — keep the order
    // the producer wrote them in. Baseline records (mono_ns == 0) sort first.
    std::stable_sort(records_.begin(), records_.end(),
                     [](const hrr_region_rec& a, const hrr_region_rec& b) {
                         return a.mono_ns < b.mono_ns;
                     });
    return records_.size();
}

void RegionMap::check_clock_against(uint64_t first_event_ns, uint64_t last_event_ns) {
    if (records_.empty() || first_event_ns == 0 || last_event_ns < first_event_ns)
        return;

    // Baseline records are deliberately stamped 0 and carry no clock evidence.
    size_t stamped = 0, inside = 0;
    for (const auto& r : records_) {
        if (r.mono_ns <= 0) continue;
        ++stamped;
        const uint64_t t = static_cast<uint64_t>(r.mono_ns);
        if (t >= first_event_ns && t <= last_event_ns) ++inside;
    }
    if (stamped == 0) return;

    // Region timestamps must come from the same clock as the event headers
    // (CLOCK_MONOTONIC via amd::Os::timeNanos). A producer that stamped
    // CLOCK_REALTIME, or converted with a stale offset, lands wholly outside
    // the capture's own span — and then every block looks live at the wrong
    // instant, which silently turns the classification into noise.
    if (inside * 10 < stamped) {
        fprintf(stderr,
                "[HRR] WARNING: %zu of %zu region timestamps fall outside the "
                "archive's event span [%llu, %llu] — the producer's clock is "
                "probably not CLOCK_MONOTONIC. Block classification will be "
                "unreliable.\n",
                stamped - inside, stamped,
                static_cast<unsigned long long>(first_event_ns),
                static_cast<unsigned long long>(last_event_ns));
    }
}

// ---------------------------------------------------------------------------
// Timeline application
// ---------------------------------------------------------------------------

// Tightest enclosing entry: the largest base <= addr whose extent covers addr.
template <typename Extent>
static bool enclosing(const std::map<uint64_t, Extent>& m, uint64_t addr,
                      uint64_t* base, uint64_t* size, int* device = nullptr) {
    if (m.empty()) return false;
    auto it = m.upper_bound(addr);
    if (it == m.begin()) return false;
    --it;
    if (addr >= it->first + it->second.size) return false;
    if (base)   *base   = it->first;
    if (size)   *size   = it->second.size;
    if (device) *device = static_cast<int>(it->second.device);
    return true;
}

void RegionMap::apply(PlaybackContext& ctx, const hrr_region_rec& r) {
    if (r.kind != HRR_REGION_BLOCK && r.kind != HRR_REGION_SEGMENT)
        return;  // unknown kind: ignore forward
    if (r.size == 0 && r.op == HRR_REGION_ADD) return;

    auto& live_set = (r.kind == HRR_REGION_BLOCK) ? blocks_ : segments_;
    if (r.op == HRR_REGION_ADD) {
        live_set[r.base] = Extent{r.size, r.device};
        return;
    }
    live_set.erase(r.base);
    // A segment the producer freed mid-run. Release the buffer standing in for
    // it, so a later pointer into that address is reported rather than quietly
    // resolving into memory the recorded program had already given up.
    if (r.kind == HRR_REGION_SEGMENT) {
        auto it = materialized_.find(r.base);
        if (it != materialized_.end()) {
            hrr_release_region(ctx, r.base, it->second);
            materialized_.erase(it);
        }
    }
}

void RegionMap::advance_to(PlaybackContext& ctx, int64_t ts) {
    if (records_.empty()) return;
    // Monotonic only. Events are dispatched in sequence order, which is the
    // capture's call order, so a timestamp going backwards means clock jitter
    // between threads rather than a genuine rewind. Call rewind() to start a
    // second pass over the same timeline.
    if (ts < last_ts_) return;
    last_ts_ = ts;
    while (cursor_ < records_.size() && records_[cursor_].mono_ns <= ts) {
        apply(ctx, records_[cursor_]);
        ++cursor_;
    }
}

void RegionMap::rewind() {
    cursor_  = 0;
    last_ts_ = INT64_MIN;
    blocks_.clear();
    segments_.clear();
    // Keep materialized_ / materialize_failed_ / materialized_total_: a warm-up
    // pass may already have allocated stand-in buffers and registered them in
    // alloc_map. Clearing those would leak the GPU allocations or double-malloc
    // on the timed pass. apply() on SEGMENT DEL still releases a buffer the
    // producer freed mid-run, as on the first pass.
}

void* RegionMap::materialize_for(PlaybackContext& ctx, uint64_t rec_addr) {
    uint64_t base = 0, size = 0;
    int device = 0;
    if (!enclosing(segments_, rec_addr, &base, &size, &device)) return nullptr;

    auto it = materialized_.find(base);
    if (it == materialized_.end()) {
        if (materialize_failed_.count(base)) return nullptr;
        void* live = nullptr;
        if (hrr_materialize_region(ctx, base, static_cast<size_t>(size), device,
                                   &live) != hipSuccess) {
            materialize_failed_.insert(base);
            return nullptr;
        }
        it = materialized_.emplace(base, live).first;
        ++materialized_total_;
    }
    return static_cast<char*>(it->second) + (rec_addr - base);
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

RegionMap::Class RegionMap::classify(uint64_t rec_addr, uint64_t* blk_base,
                                     uint64_t* blk_size) const {
    if (enclosing(blocks_, rec_addr, blk_base, blk_size)) return Class::InBlock;
    if (enclosing(segments_, rec_addr, nullptr, nullptr))
        return Class::InSegmentNoBlock;
    return Class::None;
}

}  // namespace hrr
