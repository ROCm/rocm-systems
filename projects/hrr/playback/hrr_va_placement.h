/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

/*
 * hrr_va_placement.h — replay device allocations at their capture-time address.
 *
 * Replay translates every pointer it can see: kernel arguments, copy endpoints,
 * VMM calls. It cannot see a device address the recorded program stored in a
 * buffer. That address reaches the GPU inside an H2D payload, restored byte for
 * byte, and names memory in the capturing process. vLLM's block table is the
 * case that made this matter (ROCM-31827): a device pointer copied into device
 * memory with hipMemcpy, then dereferenced by _compute_slot_mappings_kernel.
 *
 * Placement makes the stored address true again. Before hipInit the replayer
 * holds every recorded allocation range with a PROT_NONE placeholder, so nothing
 * else in the process can land there. After hipInit it swaps each placeholder
 * for a hipMemAddressReserve at that address, and each recorded allocation is
 * then a VMM mapping inside the reservation at exactly its recorded base.
 * Translation of a placed allocation is the identity.
 *
 * An allocation whose range could not be held, or that shares a page with one
 * still live, falls back to an ordinary allocation at a fresh address and is
 * reported by name. Managed and fine-grained memory have no VMM equivalent and
 * are never placed.
 *
 * The planning half of this file is pure and CPU-only, so the unit tests can
 * reach it without a GPU. The VaPlacement class is the HIP half.
 * Placement needs mmap(MAP_FIXED_NOREPLACE) to hold ranges before hipInit, so
 * on Windows hold() declines and replay keeps its old behaviour.
 */

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "hrr/hrr_api_args.h"
#include "hrr_reader.h"

namespace hrr {

// Half-open [base, end).
struct VaRange {
    uint64_t base = 0;
    uint64_t end  = 0;
    bool operator==(const VaRange& o) const { return base == o.base && end == o.end; }
};

// The smallest unit placement can hold or map. VRAM VMM maps at page
// granularity, and the runtime is asked for its own minimum after hipInit.
constexpr uint64_t kPlacePage = 4096;

// The user half of a 48-bit address space. A recorded value outside it cannot
// be a device allocation this process could be given.
constexpr uint64_t kPlaceVaLimit = 1ull << 47;
// Below this the kernel refuses mappings anyway (vm.mmap_min_addr).
constexpr uint64_t kPlaceVaFloor = 1ull << 20;

inline uint64_t va_floor(uint64_t v, uint64_t g) { return v - (v % g); }

// Rounds up, or returns 0 when that would wrap.
inline uint64_t va_ceil(uint64_t v, uint64_t g) {
    const uint64_t r = v % g;
    if (r == 0) return v;
    return (v > UINT64_MAX - (g - r)) ? 0 : v + (g - r);
}

// Round every range out to `page`, then merge the ones that overlap or touch.
// Empty ranges, wrapping ranges and ranges outside user VA are dropped.
inline std::vector<VaRange> va_round_merge(std::vector<VaRange> in, uint64_t page) {
    std::vector<VaRange> r;
    r.reserve(in.size());
    for (const auto& x : in) {
        if (x.end <= x.base) continue;
        const uint64_t b = va_floor(x.base, page);
        const uint64_t e = va_ceil(x.end, page);
        if (e == 0 || b < kPlaceVaFloor || e > kPlaceVaLimit) continue;
        r.push_back({b, e});
    }
    std::sort(r.begin(), r.end(),
              [](const VaRange& a, const VaRange& b) { return a.base < b.base; });
    std::vector<VaRange> out;
    for (const auto& x : r) {
        if (!out.empty() && x.base <= out.back().end)
            out.back().end = std::max(out.back().end, x.end);
        else
            out.push_back(x);
    }
    return out;
}

// `from` minus `minus`. Both sorted and merged, as va_round_merge returns them.
inline std::vector<VaRange> va_subtract(const std::vector<VaRange>& from,
                                        const std::vector<VaRange>& minus) {
    std::vector<VaRange> out;
    size_t j = 0;
    for (const auto& f : from) {
        uint64_t cur = f.base;
        while (j < minus.size() && minus[j].end <= cur) ++j;
        size_t k = j;
        while (cur < f.end && k < minus.size() && minus[k].base < f.end) {
            if (minus[k].base > cur) out.push_back({cur, minus[k].base});
            cur = std::max(cur, minus[k].end);
            ++k;
        }
        if (cur < f.end) out.push_back({cur, f.end});
    }
    return out;
}

// The range in `sorted` that contains all of [b, e), or nullptr.
inline const VaRange* va_find_containing(const std::vector<VaRange>& sorted,
                                         uint64_t b, uint64_t e) {
    auto it = std::upper_bound(sorted.begin(), sorted.end(), b,
                               [](uint64_t v, const VaRange& r) { return v < r.base; });
    if (it == sorted.begin()) return nullptr;
    --it;
    return (b >= it->base && e <= it->end) ? &*it : nullptr;
}

// Whether any range in `sorted` overlaps [b, e).
inline bool va_overlaps(const std::vector<VaRange>& sorted, uint64_t b, uint64_t e) {
    auto it = std::upper_bound(sorted.begin(), sorted.end(), b,
                               [](uint64_t v, const VaRange& r) { return v < r.base; });
    if (it != sorted.begin() && std::prev(it)->end > b) return true;
    return it != sorted.end() && it->base < e;
}

// Which allocation APIs are placed, and what range one recorded call claimed.
// Only successful calls count: a failed one returned no address. Managed
// memory, and hipExtMallocWithFlags with any flag set (fine-grained, uncached,
// signal, contiguous), have no VMM equivalent and keep their old path.
enum class PlaceKind { None, Alloc, Vmm };

inline PlaceKind placement_event_range(const Event& ev, VaRange* out) {
    const auto& p = ev.raw_payload;
    if (p.size() < sizeof(hrr_event_header)) return PlaceKind::None;
    uint64_t base = 0, size = 0;
    int32_t  ret  = 0;
    PlaceKind kind = PlaceKind::Alloc;
    switch (ev.header().event_type) {
        case HRR_API_HIPMALLOC: {
            hrr_args_hipMalloc a;
            if (p.size() < sizeof(a)) return PlaceKind::None;
            std::memcpy(&a, p.data(), sizeof(a));
            ret = a.ret; base = a.ptr; size = a.size;
            break;
        }
        case HRR_API_HIPEXTMALLOCWITHFLAGS: {
            hrr_args_hipExtMallocWithFlags a;
            if (p.size() < sizeof(a)) return PlaceKind::None;
            std::memcpy(&a, p.data(), sizeof(a));
            if (a.flags != 0) return PlaceKind::None;
            ret = a.ret; base = a.ptr; size = a.sizeBytes;
            break;
        }
        case HRR_API_HIPMALLOCASYNC: {
            hrr_args_hipMallocAsync a;
            if (p.size() < sizeof(a)) return PlaceKind::None;
            std::memcpy(&a, p.data(), sizeof(a));
            ret = a.ret; base = a.dev_ptr; size = a.size;
            break;
        }
        case HRR_API_HIPMALLOCFROMPOOLASYNC: {
            hrr_args_hipMallocFromPoolAsync a;
            if (p.size() < sizeof(a)) return PlaceKind::None;
            std::memcpy(&a, p.data(), sizeof(a));
            ret = a.ret; base = a.dev_ptr; size = a.size;
            break;
        }
        case HRR_API_HIPMEMADDRESSRESERVE: {
            hrr_args_hipMemAddressReserve a;
            if (p.size() < sizeof(a)) return PlaceKind::None;
            std::memcpy(&a, p.data(), sizeof(a));
            ret = a.ret; base = a.ptr; size = a.size;
            kind = PlaceKind::Vmm;
            break;
        }
        default:
            return PlaceKind::None;
    }
    if (ret != 0 || base == 0 || size == 0 || base > UINT64_MAX - size)
        return PlaceKind::None;
    out->base = base;
    out->end  = base + size;
    return kind;
}

// The address a recorded call exported for another process, or 0. Both APIs
// refuse VMM memory, so the allocation holding that address keeps the old path.
inline uint64_t placement_exported_ptr(const Event& ev) {
    const auto& p = ev.raw_payload;
    if (p.size() < sizeof(hrr_event_header)) return 0;
    switch (ev.header().event_type) {
        case HRR_API_HIPIPCGETMEMHANDLE: {
            hrr_args_hipIpcGetMemHandle a;
            if (p.size() < sizeof(a)) return 0;
            std::memcpy(&a, p.data(), sizeof(a));
            return a.ret == 0 ? a.devPtr : 0;
        }
        case HRR_API_HIPMEMPOOLEXPORTPOINTER: {
            hrr_args_hipMemPoolExportPointer a;
            if (p.size() < sizeof(a)) return 0;
            std::memcpy(&a, p.data(), sizeof(a));
            return a.ret == 0 ? a.dev_ptr : 0;
        }
        default:
            return 0;
    }
}

struct PlacementPlan {
    // Ranges for the hipMalloc family and the region sidecar's segments. These
    // become placement-owned reservations after hipInit.
    std::vector<VaRange> alloc;
    // Ranges the recording reserved with hipMemAddressReserve, minus `alloc`.
    // These stay placeholders until the replayed reserve asks for its address.
    std::vector<VaRange> vmm;
    // Ranges of the allocations HIP_HRR_REPLAY_PLACE_DENY named. Left out of
    // both lists and held for the whole replay, as if something else in the
    // process had taken them.
    std::vector<VaRange> denied;
    // Ranges of the allocations the recording exported with hipIpcGetMemHandle
    // or hipMemPoolExportPointer. Neither works on VMM memory, so these are
    // left out of `alloc`, not held, and replay where the runtime puts them.
    std::vector<VaRange> exported;
    size_t alloc_events = 0;
    size_t vmm_events   = 0;
    size_t segments     = 0;
};

// Every range replay will want at its recorded address, rounded and merged.
// `segments` are the region sidecar's declared segments. `deny` holds addresses
// whose enclosing recorded allocation is to be treated as already taken.
inline PlacementPlan plan_placement(const std::vector<Event>& events,
                                    const std::vector<VaRange>& segments,
                                    const std::vector<uint64_t>& deny,
                                    uint64_t page = kPlacePage) {
    PlacementPlan plan;
    std::vector<VaRange> alloc, vmm, denied, exported;
    std::vector<uint64_t> exports;
    for (const auto& ev : events)
        if (const uint64_t x = placement_exported_ptr(ev)) exports.push_back(x);
    auto note = [](const std::vector<uint64_t>& addrs, const VaRange& r,
                   std::vector<VaRange>* out) {
        for (uint64_t d : addrs)
            if (d >= r.base && d < r.end) { out->push_back(r); return; }
    };
    for (const auto& ev : events) {
        VaRange r;
        switch (placement_event_range(ev, &r)) {
            case PlaceKind::Alloc:
                alloc.push_back(r); ++plan.alloc_events;
                note(deny, r, &denied); note(exports, r, &exported);
                break;
            case PlaceKind::Vmm:
                vmm.push_back(r); ++plan.vmm_events;
                note(deny, r, &denied);
                break;
            case PlaceKind::None:
                break;
        }
    }
    for (const auto& s : segments) {
        if (s.end <= s.base) continue;
        alloc.push_back(s);
        ++plan.segments;
        note(deny, s, &denied);
    }
    plan.denied   = va_round_merge(std::move(denied), page);
    plan.exported = va_subtract(va_round_merge(std::move(exported), page), plan.denied);
    plan.alloc    = va_subtract(va_subtract(va_round_merge(std::move(alloc), page),
                                            plan.denied),
                                plan.exported);
    // A VA the recording used for a hipMalloc at one moment and a reservation
    // at another is held as an allocation range. The reservation replayed
    // there then lands elsewhere and is reported as a fallback.
    plan.vmm = va_subtract(va_subtract(va_round_merge(std::move(vmm), page), plan.alloc),
                           plan.denied);
    return plan;
}

// Parse HIP_HRR_REPLAY_PLACE_DENY: addresses separated by commas, any base
// strtoull accepts with base 0.
inline std::vector<uint64_t> parse_place_deny(const char* s) {
    std::vector<uint64_t> out;
    if (!s) return out;
    while (*s) {
        char* end = nullptr;
        const unsigned long long v = std::strtoull(s, &end, 0);
        if (end == s) break;
        if (v) out.push_back(v);
        s = end;
        while (*s == ',' || *s == ' ') ++s;
    }
    return out;
}

}  // namespace hrr

// ---------------------------------------------------------------------------
// The HIP half, implemented in hrr_va_placement.cpp.
// ---------------------------------------------------------------------------
namespace hrr {

// Back `len` bytes at `va`, inside a reservation the caller already holds,
// with fresh device memory on `device`, readable and writable from it. `len`
// must be a multiple of the VMM granularity. On failure nothing is left
// mapped. Shared by placement and by the segment-tail guard.
hipError_t hrr_vmm_map_into(void* va, size_t len, int device,
                            hipMemGenericAllocationHandle_t* out_handle);

class VaPlacement {
  public:
    // Before hipInit: hold every planned range with a placeholder. Returns
    // false when the platform cannot (Windows), which leaves placement off.
    bool hold(PlacementPlan plan);

    // After hipInit, before the first replayed event: swap each allocation
    // placeholder for a reservation at the same address, and check the
    // address the runtime returned, because it falls back silently.
    void reserve(int device);

    bool active() const { return active_; }

    // Map `size` bytes at recorded address `rec` on `device`. On success
    // *live == rec. Returns false when the allocation has to fall back, after
    // reporting why; the caller then allocates the old way.
    bool map_at(uint64_t rec, size_t size, int device, const char* api, void** live);

    // Note a fallback decided outside map_at, e.g. during graph capture.
    void fell_back(uint64_t rec, size_t size, const char* api, const char* why);

    // If `live` is a placed mapping, unmap it and release its handle. The
    // reservation stays, so the next allocation recorded there lands again.
    bool unmap(void* live);
    // Whether `live` is the base of a placed mapping.
    bool is_mapped(void* live);

    // hipMemAddressReserve: give back the placeholder over [base, base+size) so
    // the reserve at that hint can take it. False when placement does not hold
    // that range.
    bool release_vmm_hold(uint64_t base, size_t size);
    // hipMemAddressFree: hold the range again for a later reserve there.
    void restore_vmm_hold(uint64_t base, size_t size);
    void note_vmm(bool placed, uint64_t rec, size_t size, uint64_t live);

    // Teardown of the reservations themselves. Mappings are released first by
    // the ordinary free path.
    void release_all();

    uint64_t placed() const { return placed_.load(); }
    uint64_t fallbacks() const { return fallbacks_.load(); }
    uint64_t held_bytes() const { return held_bytes_; }
    size_t   held_ranges() const { return reserved_.size(); }
    size_t   lost_ranges() const { return lost_ranges_; }

  private:
    struct Mapping {
        uint64_t end;
        uint64_t rec;
        hipMemGenericAllocationHandle_t handle;
    };

    std::mutex mu_;
    bool active_ = false;
    uint64_t gran_ = kPlacePage;
    PlacementPlan plan_;
    std::vector<VaRange> reserved_;        // placement-owned reservations
    std::map<uint64_t, Mapping> mapped_;   // page base -> live mapping
    std::vector<VaRange> vmm_held_;        // placeholders awaiting a reserve
    std::vector<VaRange> denied_held_;     // HIP_HRR_REPLAY_PLACE_DENY, held to the end
    std::atomic<uint64_t> placed_{0};
    std::atomic<uint64_t> fallbacks_{0};
    uint64_t held_bytes_ = 0;
    size_t   lost_ranges_ = 0;
};

}  // namespace hrr
