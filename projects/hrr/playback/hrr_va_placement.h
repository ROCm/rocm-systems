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
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
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
    if (from.empty()) return out;
    // Both lists are sorted and their ranges disjoint, so ends ascend too.
    // Start at the first range of `minus` that ends past `from`, rather than
    // walking up to it: one range taken from a long list is called once per
    // range to hold.
    size_t j = std::partition_point(minus.begin(), minus.end(),
                                    [&](const VaRange& r) { return r.end <= from.front().base; }) -
               minus.begin();
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

// The VMM mappings a replay made, base -> size, kept disjoint. A recorded
// hipMemUnmap can cover several hipMemMap pieces, or only part of one, so
// forgetting a range drops every entry it touches and keeps what lies outside.
inline void va_untrack_mapping(std::map<uint64_t, size_t>& m, uint64_t va, size_t size) {
    const uint64_t e = va + size;
    auto it = m.lower_bound(va);
    if (it != m.begin() && std::prev(it)->first + std::prev(it)->second > va) --it;
    while (it != m.end() && it->first < e) {
        const uint64_t b0 = it->first, e0 = b0 + it->second;
        it = m.erase(it);
        if (b0 < va) m[b0] = va - b0;
        if (e0 > e) m[e] = e0 - e;
    }
}

inline void va_track_mapping(std::map<uint64_t, size_t>& m, uint64_t va, size_t size) {
    va_untrack_mapping(m, va, size);
    m[va] = size;
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
    // Every placed allocation's own page-rounded range, sorted, before any
    // merging. When one merged `alloc` range cannot be reserved as a whole,
    // reserve() falls back to reserving these one by one.
    std::vector<VaRange> pieces;
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
    std::vector<VaRange> alloc, segs, vmm, denied, exported;
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
        segs.push_back(s);
        ++plan.segments;
        note(deny, s, &denied);
    }
    plan.denied   = va_round_merge(std::move(denied), page);
    plan.exported = va_subtract(va_round_merge(std::move(exported), page), plan.denied);
    const std::vector<VaRange> vmm_m   = va_round_merge(std::move(vmm), page);
    const std::vector<VaRange> alloc_m = va_round_merge(alloc, page);
    // The region sidecar declares a segment for every allocation it saw,
    // hipMemAddressReserve ranges included. Those belong to the replayed
    // reserve, so they come out of the segments before the merge. A VA the
    // recording used for a hipMalloc at one moment and a reservation at
    // another stays an allocation range; the reserve replayed there lands
    // elsewhere and is reported as a fallback.
    std::vector<VaRange> all = alloc_m;
    for (const auto& r : va_subtract(va_round_merge(segs, page), vmm_m))
        all.push_back(r);
    const std::vector<VaRange> keep_out =
        va_round_merge([&] { auto v = plan.denied;
                             v.insert(v.end(), plan.exported.begin(), plan.exported.end());
                             return v; }(), page);
    plan.alloc = va_subtract(va_round_merge(std::move(all), page), keep_out);
    plan.vmm   = va_subtract(va_subtract(vmm_m, plan.alloc), plan.denied);
    for (const auto* list : {&alloc, &segs}) {
        for (const auto& a : *list) {
            if (a.end <= a.base) continue;
            const uint64_t b = va_floor(a.base, page), e = va_ceil(a.end, page);
            if (e == 0 || b < kPlaceVaFloor || e > kPlaceVaLimit ||
                va_overlaps(keep_out, b, e) || (list == &segs && va_overlaps(vmm_m, b, e)))
                continue;
            plan.pieces.push_back({b, e});
        }
    }
    std::sort(plan.pieces.begin(), plan.pieces.end(),
              [](const VaRange& x, const VaRange& y) { return x.base < y.base; });
    return plan;
}

// The ranges reserve() asks for one by one when the merged range [b, e) could
// not be reserved whole: each allocation's own range rounded out to the VMM
// granularity and clipped to [b, e). Two allocations that round into the same
// granule share one reservation; ranges that only touch stay separate, so one
// that cannot be reserved costs only itself.
inline std::vector<VaRange> va_reserve_pieces(const std::vector<VaRange>& pieces,
                                              uint64_t b, uint64_t e, uint64_t gran) {
    std::vector<VaRange> r;
    for (const auto& p : pieces) {
        if (p.end <= b || p.base >= e) continue;
        uint64_t pb = va_floor(p.base, gran), pe = va_ceil(p.end, gran);
        if (pe == 0) continue;
        pb = std::max(pb, b);
        pe = std::min(pe, e);
        if (pb < pe) r.push_back({pb, pe});
    }
    std::sort(r.begin(), r.end(),
              [](const VaRange& x, const VaRange& y) { return x.base < y.base; });
    std::vector<VaRange> out;
    for (const auto& x : r) {
        if (!out.empty() && x.base < out.back().end)
            out.back().end = std::max(out.back().end, x.end);
        else
            out.push_back(x);
    }
    return out;
}

// The ranges /proc/self/maps lists, sorted and merged. Text in, so the tests
// can feed it a fixture.
inline std::vector<VaRange> parse_proc_maps(const std::string& text) {
    std::vector<VaRange> r;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(pos, nl - pos);
        pos = nl + 1;
        char* end = nullptr;
        const unsigned long long b = std::strtoull(line.c_str(), &end, 16);
        if (!end || *end != '-') continue;
        const unsigned long long e = std::strtoull(end + 1, nullptr, 16);
        if (e > b) r.push_back({b, e});
    }
    std::sort(r.begin(), r.end(),
              [](const VaRange& x, const VaRange& y) { return x.base < y.base; });
    std::vector<VaRange> out;
    for (const auto& x : r) {
        if (!out.empty() && x.base <= out.back().end)
            out.back().end = std::max(out.back().end, x.end);
        else
            out.push_back(x);
    }
    return out;
}

// A recorded call that allocated device memory placement does not place, with
// the range it claimed (rec may be 0 when the API returns no plain pointer)
// and why. Only successful calls count.
struct Unplaced {
    uint64_t    rec  = 0;
    uint64_t    size = 0;
    const char* api  = "";
    std::string why;
};

inline bool placement_unplaced(const uint8_t* data, size_t n, Unplaced* out) {
    if (n < sizeof(hrr_event_header) + sizeof(int32_t)) return false;
    hrr_event_header hdr;
    std::memcpy(&hdr, data, sizeof(hdr));
    int32_t ret = 0;
    std::memcpy(&ret, data + sizeof(hrr_event_header), sizeof(ret));
    if (ret != 0) return false;
    auto get = [&](auto* a) {
        if (n < sizeof(*a)) return false;
        std::memcpy(a, data, sizeof(*a));
        return true;
    };
    auto extent = [](const uint8_t* b) {
        hipExtent x{};
        std::memcpy(&x, b, std::min(sizeof(x), size_t(24)));
        return uint64_t(x.width) * std::max<size_t>(x.height, 1) * std::max<size_t>(x.depth, 1);
    };
    static const char* kArray = "arrays have an opaque layout VMM cannot map";
    switch (hdr.event_type) {
        case HRR_API_HIPMALLOCMANAGED: {
            hrr_args_hipMallocManaged a;
            if (!get(&a)) return false;
            *out = {a.dev_ptr, a.size, "hipMallocManaged",
                    "managed memory has no VMM equivalent"};
            return true;
        }
        case HRR_API_HIPEXTMALLOCWITHFLAGS: {
            hrr_args_hipExtMallocWithFlags a;
            if (!get(&a) || a.flags == 0) return false;
            char why[128];
            snprintf(why, sizeof(why),
                     "flags 0x%x (fine-grained, uncached, signal or contiguous) "
                     "have no VMM equivalent", a.flags);
            *out = {a.ptr, a.sizeBytes, "hipExtMallocWithFlags", why};
            return true;
        }
        case HRR_API_HIPMALLOCPITCH: {
            hrr_args_hipMallocPitch a;
            if (!get(&a)) return false;
            *out = {a.ptr, a.pitch * a.height, "hipMallocPitch",
                    "pitched allocations are not placed"};
            return true;
        }
        case HRR_API_HIPMEMALLOCPITCH: {
            hrr_args_hipMemAllocPitch a;
            if (!get(&a)) return false;
            *out = {a.dptr, a.pitch * a.height, "hipMemAllocPitch",
                    "pitched allocations are not placed"};
            return true;
        }
        case HRR_API_HIPMALLOC3D: {
            hrr_args_hipMalloc3D a;
            if (!get(&a)) return false;
            *out = {0, extent(a.extent_bytes), "hipMalloc3D",
                    "pitched allocations are not placed"};
            return true;
        }
        case HRR_API_HIPMALLOCARRAY: {
            hrr_args_hipMallocArray a;
            if (!get(&a)) return false;
            *out = {0, a.width * std::max<uint64_t>(a.height, 1), "hipMallocArray", kArray};
            return true;
        }
        case HRR_API_HIPMALLOC3DARRAY: {
            hrr_args_hipMalloc3DArray a;
            if (!get(&a)) return false;
            *out = {0, extent(a.extent_bytes), "hipMalloc3DArray", kArray};
            return true;
        }
        case HRR_API_HIPMALLOCMIPMAPPEDARRAY: {
            hrr_args_hipMallocMipmappedArray a;
            if (!get(&a)) return false;
            *out = {0, extent(a.extent_bytes), "hipMallocMipmappedArray", kArray};
            return true;
        }
        case HRR_API_HIPARRAYCREATE:
            *out = {0, 0, "hipArrayCreate", kArray};
            return true;
        case HRR_API_HIPARRAY3DCREATE:
            *out = {0, 0, "hipArray3DCreate", kArray};
            return true;
        case HRR_API_HIPMIPMAPPEDARRAYCREATE:
            *out = {0, 0, "hipMipmappedArrayCreate", kArray};
            return true;
        case HRR_API_HIPGRAPHADDMEMALLOCNODE: {
            hrr_args_hipGraphAddMemAllocNode a;
            if (!get(&a)) return false;
            hipMemAllocNodeParams np{};
            if (a.pNodeParams_present)
                std::memcpy(&np, a.pNodeParams_bytes,
                            std::min(sizeof(np), sizeof(a.pNodeParams_bytes)));
            *out = {reinterpret_cast<uint64_t>(np.dptr), np.bytesize,
                    "hipGraphAddMemAllocNode",
                    "graph memory nodes allocate when the graph runs"};
            return true;
        }
        default:
            return false;
    }
}

inline bool placement_unplaced(const Event& ev, Unplaced* out) {
    return placement_unplaced(ev.raw_payload.data(), ev.raw_payload.size(), out);
}

// Whether the recording reached memory on more than one device: two devices
// selected, peer access enabled, a pool shared, or a peer copy. Placed
// mappings are then made accessible to every peer that can reach them, as
// enabling peer access would have done for ordinary allocations.
inline bool placement_needs_peer_access(const std::vector<Event>& events) {
    int first = -1;
    for (const auto& ev : events) {
        const auto& p = ev.raw_payload;
        if (p.size() < sizeof(hrr_event_header)) continue;
        switch (ev.header().event_type) {
            case HRR_API_HIPSETDEVICE: {
                hrr_args_hipSetDevice a;
                if (p.size() < sizeof(a)) break;
                std::memcpy(&a, p.data(), sizeof(a));
                if (a.ret != 0) break;
                if (first < 0) first = a.deviceId;
                else if (a.deviceId != first) return true;
                break;
            }
            case HRR_API_HIPDEVICEENABLEPEERACCESS:
            case HRR_API_HIPCTXENABLEPEERACCESS:
            case HRR_API_HIPMEMPOOLSETACCESS:
            case HRR_API_HIPMEMCPYPEER:
            case HRR_API_HIPMEMCPYPEERASYNC:
                return true;
            default:
                break;
        }
    }
    return false;
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
// with fresh device memory on `device`, readable and writable from it and from
// every device in `peers`. `len` must be a multiple of the VMM granularity. On
// failure nothing is left mapped. Shared by placement and by the segment-tail
// guard.
hipError_t hrr_vmm_map_into(void* va, size_t len, int device,
                            hipMemGenericAllocationHandle_t* out_handle,
                            const std::vector<int>& peers = {});

// The ranges mapped in this process now, from /proc/self/maps.
std::vector<VaRange> read_proc_maps();

// Hold every free piece of [b, e) with a placeholder and append what was held
// to `out`. `occupied` is what read_proc_maps returned, so ranges already
// mapped are skipped without a failed mmap each; a piece that still collides
// (something mapped since) is split in half until the free part is found.
void hold_free_pieces(uint64_t b, uint64_t e, const std::vector<VaRange>& occupied,
                      std::vector<VaRange>* out);

// A placed mapping: page range [key, end), the recorded allocation base, its
// physical handle and the device it lives on. `unmapping` marks a freed one
// whose hipMemUnmap is running outside the lock: it still occupies its range.
// A mapping freed with hipFreeAsync remembers the live stream it was freed on
// and that stream's device (`on_stream`, `stream`, `stream_device`: every
// device's null stream has the same handle) and, when no capture was open, an
// event recorded on that stream after the free (`event`): what an allocation
// taking the mapping back on another stream has to wait for.
struct PlacedMapping {
    uint64_t end;
    uint64_t rec;
    hipMemGenericAllocationHandle_t handle;
    int  device    = 0;
    bool unmapping = false;
    bool on_stream = false;
    hipStream_t stream = nullptr;
    int stream_device  = -1;
    hipEvent_t  event  = nullptr;
};
using PlacedMap = std::map<uint64_t, PlacedMapping>;

// The mapping in `m` that overlaps [pb, pe), or nullptr. Only the one nearest
// below pe can, because mappings never overlap each other.
inline const PlacedMapping* va_mapping_overlapping(const PlacedMap& m, uint64_t pb,
                                                   uint64_t pe) {
    auto it = m.lower_bound(pe);
    if (it == m.begin()) return nullptr;
    --it;
    return it->second.end > pb ? &it->second : nullptr;
}

// The page bases of every mapping in `m` that overlaps [pb, pe), in order.
inline std::vector<uint64_t> va_mappings_overlapping(const PlacedMap& m, uint64_t pb,
                                                     uint64_t pe) {
    std::vector<uint64_t> out;
    auto it = m.lower_bound(pb);
    if (it != m.begin() && std::prev(it)->second.end > pb) --it;
    for (; it != m.end() && it->first < pe; ++it) out.push_back(it->first);
    return out;
}

// The device of `stream`: the stream's own, or the current device's for the
// null stream.
int hrr_stream_device(hipStream_t stream);

// Create an event and record it on `stream`, with the stream's device current
// around both. On failure nothing is left.
hipError_t hrr_record_free_event(hipStream_t stream, hipEvent_t* event);

#ifdef HRR_VA_PLACEMENT_TESTING
// Tests only: the HIP calls placement makes, replaced to run without a GPU.
struct VmmOps {
    hipError_t (*map)(void* va, size_t len, int device, hipMemGenericAllocationHandle_t* h,
                      const std::vector<int>& peers) = hrr_vmm_map_into;
    hipError_t (*unmap)(void* base, size_t size)              = hipMemUnmap;
    hipError_t (*release)(hipMemGenericAllocationHandle_t h) = hipMemRelease;
    void (*clear_error)()                                     = nullptr;
    hipError_t (*record_event)(hipStream_t s, hipEvent_t* e) = hrr_record_free_event;
    hipError_t (*wait_event)(hipStream_t s, hipEvent_t e, unsigned int flags) =
        hipStreamWaitEvent;
    hipError_t (*sync_event)(hipEvent_t e)                    = hipEventSynchronize;
    hipError_t (*destroy_event)(hipEvent_t e)                 = hipEventDestroy;
    hipError_t (*address_free)(void* va, size_t size)         = hipMemAddressFree;
    int (*stream_device)(hipStream_t s)                       = hrr_stream_device;
};
// How many times read_proc_maps() has read /proc/self/maps.
size_t proc_maps_reads_for_test();
#endif

class VaPlacement {
  public:
    // Before hipInit: hold every planned range with a placeholder. Returns
    // false when the platform cannot (Windows), which leaves placement off.
    bool hold(PlacementPlan plan);
    // Placement off, HIP_HRR_REPLAY_PLACE_DENY set: hold only the denied
    // ranges, before hipInit, so the runtime cannot return those addresses.
    // Placement stays inactive; release_all() drops the holds. Returns false
    // when the platform cannot (Windows).
    bool hold_denied(const std::vector<VaRange>& denied);

    // After hipInit, before the first replayed event: check that every device
    // supports VMM, then swap each allocation placeholder for a reservation at
    // the same address, and check the address the runtime returned, because
    // it falls back silently. With `peer_access`, each device's mappings are
    // also made accessible to every peer that can reach them. Returns false,
    // after one line saying why, when placement had to turn off.
    bool reserve(int device_count, bool peer_access);

    bool active() const { return active_; }
    // Under --verbose every fallback is named, not only the first 16.
    void set_verbose(bool v) { verbose_ = v; }
    int device_count() const { return device_count_; }

    // Map `size` bytes at recorded address `rec` on `device`. On success
    // *live == rec. Returns false when the allocation has to fall back, after
    // reporting why; the caller then allocates the old way. `stream` is the
    // live stream a stream-ordered allocation is made on, and nullptr for
    // hipMalloc, hipExtMallocWithFlags and region segments.
    //
    // A freed mapping still waiting for its unmap that starts at the same page
    // on the same device and ends where the allocation's pages do is taken
    // back as it is: no unmap, no new map. A stream-ordered allocation also
    // takes back one up to 9/8 of its own pages, keeping the mapping's end, as
    // the pool hands a block back for a request up to 12.5% smaller. The
    // allocation is ordered after the free first:
    //  - freed by hipFreeAsync on `stream` itself: nothing to wait for;
    //  - otherwise, with an event recorded after the free and no capture
    //    open: `stream` waits for that event, or the host does for an
    //    allocation with no stream;
    //  - otherwise it is handled as any other overlap.
    // Any other overlap with a freed mapping unmaps it first, unless
    // `capturing` says a graph capture is open, because hipMemUnmap would wait
    // for the capturing stream; the allocation then falls back. When the map
    // runs out of memory and no capture is open, the freed mappings are
    // unmapped and the map is tried once more.
    bool map_at(uint64_t rec, size_t size, int device, const char* api, void** live,
                bool capturing = false, const hipStream_t* stream = nullptr);

    // Note a fallback decided outside map_at, e.g. during graph capture.
    void fell_back(uint64_t rec, size_t size, const char* api, const char* why);

    // If `live` is a placed mapping, unmap it and release its handle. The
    // reservation stays, so the next allocation recorded there lands again.
    // With `defer` it moves to a list drain_deferred() unmaps later, and
    // map_at may take it back. Replay defers a hipFree while a graph capture
    // is open, since hipMemUnmap would wait for the capturing stream.
    bool unmap(void* live, bool defer = false);
    // hipFreeAsync of `live` on live stream `stream`: always deferred, since
    // hipMemUnmap waits for every stream. Keeps the stream and, unless
    // `capturing`, an event recorded on it now, for map_at to order a later
    // allocation on another stream after the free.
    bool unmap_async(void* live, hipStream_t stream, bool capturing);
    // A live stream is being destroyed: a deferred free made on it no longer
    // counts as made on whatever stream gets its handle next.
    void stream_destroyed(hipStream_t stream);
    // Unmap everything unmap() deferred, and retry unmaps that failed. Call
    // only when no capture is open. Returns how many were unmapped.
    size_t drain_deferred();
    // An allocation that did not go through placement returned `r`. If it ran
    // out of memory and no capture is open, unmap the deferred frees, which
    // still hold memory. Returns how many were unmapped: when that is not 0,
    // the caller tries the allocation once more.
    size_t drain_for_retry(hipError_t r, bool capturing);

    // hipMemAddressReserve: give back the placeholder over [base, base+size) so
    // the reserve at that hint can take it. False when placement does not hold
    // that range. Between this call and the reserve, the range is free to any
    // mmap in the process, and another replay thread's allocation can land
    // there first; the reserve then misses its hint and falls back, named.
    bool release_vmm_hold(uint64_t base, size_t size);
    // hipMemAddressFree: hold the range again for a later reserve there.
    void restore_vmm_hold(uint64_t base, size_t size);
    // The same for many ranges, with one read of /proc/self/maps.
    void restore_vmm_holds(const std::vector<VaRange>& ranges);
    // The replayed hipMemAddressReserve returned `live`. Counts it, and when
    // `held` and the runtime put it elsewhere, holds the recorded range again.
    void vmm_reserved(uint64_t rec, size_t size, bool held, uint64_t live);

    // Teardown: unmap whatever is still mapped, then free the reservations
    // and drop every placeholder, the denied ones too when placement is off.
    void release_all();

    // Between the warm-up pass and the timed pass. The line budget starts
    // again too, so the timed pass names its own fallbacks.
    void reset_counts() { placed_ = 0; fallbacks_ = 0; lines_ = 0; deferred_total_ = 0; }

    uint64_t placed() const { return placed_.load(); }
    uint64_t fallbacks() const { return fallbacks_.load(); }
    uint64_t deferred_unmaps() const { return deferred_total_.load(); }
    uint64_t held_bytes() const { return held_bytes_; }
    size_t   held_ranges() const { return reserved_.size(); }
    size_t   lost_ranges() const { return lost_ranges_; }
    std::vector<uint64_t> mapped_bases();

#ifdef HRR_VA_PLACEMENT_TESTING
    // Tests only: replace the VMM calls; take [b, e) as reserved on
    // `devices` devices, as reserve() would; take [rec, rec + size) as a
    // placed mapping on `device`, as map_at would.
    void set_vmm_ops_for_test(VmmOps ops) { ops_ = ops; }
    void adopt_reservation_for_test(uint64_t b, uint64_t e, int devices);
    void adopt_mapping_for_test(uint64_t rec, size_t size, int device = 0);
    // Whether `live` is the base of a live placed mapping.
    bool is_mapped(void* live);
#endif

  private:
    hipError_t vmm_map(uint64_t pb, uint64_t pe, int device,
                       hipMemGenericAllocationHandle_t* h);
    void clear_error();
    bool unmap_one(uint64_t pb, const PlacedMapping& m);
    // Destroy the event a deferred free kept, if any.
    void drop_event(hipEvent_t e);
    // hrr_stream_device, or its stand-in in tests.
    int stream_device(hipStream_t stream);
    // Order work after `e`: `stream` waits for it, or the host does when
    // there is no stream. Never under mu_.
    void wait_for_free(hipEvent_t e, const hipStream_t* stream);
    bool unmap_impl(void* live, bool defer, bool on_stream, hipStream_t stream,
                    bool capturing);
    // Unmap the deferred mappings at `keys` that no other thread is
    // unmapping. They stay in deferred_, marked, while hipMemUnmap runs with
    // mu_ released; `lk` holds mu_ again on return. Returns how many went.
    size_t drain_locked(std::unique_lock<std::mutex>& lk, const std::vector<uint64_t>& keys);
    bool any_unmapping() const;
    void reserve_line(bool whole, uint64_t b, uint64_t e, const char* why);

    std::mutex mu_;
    std::condition_variable cv_;           // an unmap outside mu_ finished
#ifdef HRR_VA_PLACEMENT_TESTING
    VmmOps ops_;
#endif
    bool active_  = false;
    bool verbose_ = false;
    int  device_count_ = 0;
    uint64_t gran_ = kPlacePage;
    PlacementPlan plan_;
    std::vector<VaRange> reserved_;        // placement-owned reservations, sorted
    std::vector<VaRange> alloc_holds_;     // allocation placeholders left after reserve()
    PlacedMap mapped_;                     // page base -> live mapping
    PlacedMap deferred_;                   // freed, unmap deferred, running or failed
    std::vector<std::vector<int>> peers_;  // device -> peers granted access
    std::vector<VaRange> vmm_held_;        // placeholders awaiting a reserve
    std::map<uint64_t, std::vector<VaRange>> vmm_released_;  // reserve base -> dropped
    std::vector<VaRange> denied_held_;     // HIP_HRR_REPLAY_PLACE_DENY, held to the end
    std::atomic<uint64_t> placed_{0};
    std::atomic<uint64_t> fallbacks_{0};
    std::atomic<uint64_t> lines_{0};
    std::atomic<uint64_t> deferred_total_{0};
    uint64_t held_bytes_ = 0;
    size_t   lost_ranges_ = 0;
    size_t   reserve_lines_ = 0;
};

}  // namespace hrr
