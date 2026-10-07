/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
/* hrr_va_placement.cpp — the HIP half of capture-address placement. */

#include "hrr_va_placement.h"

#include <cstdio>

#ifndef _WIN32
#include <sys/mman.h>
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif
#endif

namespace hrr {

namespace {

// Report at most this many fallbacks one by one. The rest are counted in the
// summary: past a handful, the lines stop helping anyone find the one that
// matters.
constexpr uint64_t kFallbackLines = 16;

#ifndef _WIN32
// Hold [b, e) with an inaccessible placeholder. MAP_FIXED_NOREPLACE fails
// instead of moving, and a kernel too old to know the flag treats the address
// as a hint, so the returned address is checked either way.
bool hold_exact(uint64_t b, uint64_t e) {
    void* want = reinterpret_cast<void*>(b);
    void* p = mmap(want, e - b, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE,
                   -1, 0);
    if (p == MAP_FAILED) return false;
    if (p != want) { munmap(p, e - b); return false; }
    return true;
}

// Hold what can be held of [b, e). A range that collides with something
// already mapped is split in half until the free pieces are found, so one
// library mapped into the middle of a recorded range costs only the pages it
// covers.
void hold_pieces(uint64_t b, uint64_t e, std::vector<VaRange>* out) {
    if (b >= e) return;
    if (hold_exact(b, e)) { out->push_back({b, e}); return; }
    if (e - b <= kPlacePage) return;
    const uint64_t mid = va_floor(b + (e - b) / 2, kPlacePage);
    hold_pieces(b, mid, out);
    hold_pieces(mid, e, out);
}

void drop_hold(uint64_t b, uint64_t e) {
    if (b < e) munmap(reinterpret_cast<void*>(b), e - b);
}
#endif

}  // namespace

hipError_t hrr_vmm_map_into(void* va, size_t len, int device,
                            hipMemGenericAllocationHandle_t* out_handle) {
    hipMemAllocationProp prop{};
    prop.type          = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id   = device;

    hipMemGenericAllocationHandle_t handle{};
    hipError_t r = hipMemCreate(&handle, len, &prop, 0);
    if (r != hipSuccess) return r;

    r = hipMemMap(va, len, 0, handle, 0);
    if (r != hipSuccess) { (void)hipMemRelease(handle); return r; }

    hipMemAccessDesc desc{};
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id   = device;
    desc.flags         = hipMemAccessFlagsProtReadWrite;
    r = hipMemSetAccess(va, len, &desc, 1);
    if (r != hipSuccess) {
        (void)hipMemUnmap(va, len);
        (void)hipMemRelease(handle);
        return r;
    }
    *out_handle = handle;
    return hipSuccess;
}

bool VaPlacement::hold(PlacementPlan plan) {
#ifdef _WIN32
    (void)plan;
    return false;
#else
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<VaRange> held;
    for (const auto& r : plan.alloc) hold_pieces(r.base, r.end, &held);
    plan.alloc = va_round_merge(std::move(held), kPlacePage);
    held.clear();
    for (const auto& r : plan.vmm) hold_pieces(r.base, r.end, &held);
    vmm_held_ = va_round_merge(std::move(held), kPlacePage);
    // A denied range is held too, so the fallback cannot land back on it by
    // chance: the runtime has to put that allocation somewhere else.
    held.clear();
    for (const auto& r : plan.denied) hold_pieces(r.base, r.end, &held);
    denied_held_ = std::move(held);
    plan_ = std::move(plan);
    active_ = true;
    return true;
#endif
}

void VaPlacement::reserve(int device) {
#ifndef _WIN32
    std::lock_guard<std::mutex> lk(mu_);
    if (!active_) return;
    hipMemAllocationProp prop{};
    prop.type          = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id   = device;
    size_t g = 0;
    if (hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum)
            == hipSuccess && g >= kPlacePage)
        gran_ = g;

    for (const auto& r : plan_.alloc) {
        // A granularity coarser than the page shrinks the range to whole
        // granules. The edges keep their placeholders and the allocations
        // there fall back.
        const uint64_t b = va_ceil(r.base, gran_);
        const uint64_t e = va_floor(r.end, gran_);
        if (b == 0 || b >= e) { ++lost_ranges_; continue; }
        drop_hold(b, e);
        void* va = nullptr;
        // ROCr asks the thunk for exactly this address and, when that fails,
        // silently reserves elsewhere. Only the returned address says which.
        const hipError_t err = hipMemAddressReserve(&va, e - b, 0,
                                                    reinterpret_cast<void*>(b), 0);
        if (err == hipSuccess && reinterpret_cast<uint64_t>(va) == b) {
            reserved_.push_back({b, e});
            held_bytes_ += e - b;
            continue;
        }
        if (err == hipSuccess) (void)hipMemAddressFree(va, e - b);
        ++lost_ranges_;
        fprintf(stderr,
                "[HRR] Placement: could not reserve 0x%llx-0x%llx at its recorded "
                "address (%s); allocations there will move\n",
                (unsigned long long)b, (unsigned long long)e,
                err == hipSuccess ? "the runtime returned another address"
                                  : hipGetErrorString(err));
    }
#else
    (void)device;
#endif
}

void VaPlacement::fell_back(uint64_t rec, size_t size, const char* api,
                            const char* why) {
    const uint64_t n = ++fallbacks_;
    if (n <= kFallbackLines)
        fprintf(stderr,
                "[HRR] Placement: %s 0x%llx (%zu bytes) not placed at its recorded "
                "address: %s. It replays elsewhere, so a copy of its address "
                "stored in device memory is stale (HIP_HRR_REPLAY_SCAN_H2D=1 "
                "finds those)\n",
                api, (unsigned long long)rec, size, why);
    else if (n == kFallbackLines + 1)
        fprintf(stderr, "[HRR] Placement: further fallbacks are only counted\n");
}

bool VaPlacement::map_at(uint64_t rec, size_t size, int device, const char* api,
                         void** live) {
    if (!active_ || size == 0 || rec > UINT64_MAX - size) return false;
    const uint64_t pb = va_floor(rec, gran_);
    const uint64_t pe = va_ceil(rec + size, gran_);
    const char* why = nullptr;
    char buf[96];
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (pe != 0 && va_overlaps(plan_.exported, pb, pe)) {
            why = "the recording exports it to another process, which VMM memory "
                  "does not support";
        } else if (pe == 0 || !va_find_containing(reserved_, pb, pe)) {
            why = "its range could not be held";
        } else {
            // The mapping nearest below pe is the only one that can overlap.
            auto it = mapped_.lower_bound(pe);
            if (it != mapped_.begin()) {
                --it;
                if (it->second.end > pb) {
                    if (it->second.rec == rec && it->first == pb && it->second.end == pe) {
                        // The same allocation again with no free in between:
                        // the warm-up pass replays every event a second time.
                        *live = reinterpret_cast<void*>(rec);
                        ++placed_;
                        return true;
                    }
                    snprintf(buf, sizeof(buf),
                             "it shares a page with live allocation 0x%llx",
                             (unsigned long long)it->second.rec);
                    why = buf;
                }
            }
            if (!why) {
                hipMemGenericAllocationHandle_t h{};
                const hipError_t r = hrr_vmm_map_into(reinterpret_cast<void*>(pb),
                                                      pe - pb, device, &h);
                if (r == hipSuccess) {
                    mapped_[pb] = {pe, rec, h};
                    *live = reinterpret_cast<void*>(rec);
                    ++placed_;
                    return true;
                }
                snprintf(buf, sizeof(buf), "mapping it failed (%s)",
                         hipGetErrorString(r));
                why = buf;
            }
        }
    }
    fell_back(rec, size, api, why);
    return false;
}

bool VaPlacement::unmap(void* live) {
    if (!active_ || !live) return false;
    const uint64_t v = reinterpret_cast<uint64_t>(live);
    uint64_t pb = 0;
    Mapping m{};
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = mapped_.upper_bound(v);
        if (it == mapped_.begin()) return false;
        --it;
        if (it->second.rec != v) return false;
        pb = it->first;
        m  = it->second;
        mapped_.erase(it);
    }
    // hipMemUnmap waits for every stream, so nothing still queued can touch
    // the pages once they are gone. The reservation stays.
    (void)hipMemUnmap(reinterpret_cast<void*>(pb), m.end - pb);
    (void)hipMemRelease(m.handle);
    return true;
}

bool VaPlacement::is_mapped(void* live) {
    if (!active_ || !live) return false;
    const uint64_t v = reinterpret_cast<uint64_t>(live);
    std::lock_guard<std::mutex> lk(mu_);
    auto it = mapped_.upper_bound(v);
    if (it == mapped_.begin()) return false;
    --it;
    return it->second.rec == v;
}

bool VaPlacement::release_vmm_hold(uint64_t base, size_t size) {
#ifndef _WIN32
    if (!active_ || size == 0) return false;
    std::lock_guard<std::mutex> lk(mu_);
    const uint64_t b = base, e = base + size;
    bool any = false;
    for (const auto& r : vmm_held_) {
        const uint64_t lo = std::max(r.base, b), hi = std::min(r.end, e);
        if (lo < hi) { drop_hold(lo, hi); any = true; }
    }
    if (any) vmm_held_ = va_subtract(vmm_held_, {{b, e}});
    return any;
#else
    (void)base; (void)size;
    return false;
#endif
}

void VaPlacement::restore_vmm_hold(uint64_t base, size_t size) {
#ifndef _WIN32
    if (!active_ || size == 0) return;
    std::lock_guard<std::mutex> lk(mu_);
    if (!hold_exact(base, base + size)) return;
    std::vector<VaRange> v = vmm_held_;
    v.push_back({base, base + size});
    vmm_held_ = va_round_merge(std::move(v), kPlacePage);
#else
    (void)base; (void)size;
#endif
}

void VaPlacement::note_vmm(bool placed, uint64_t rec, size_t size, uint64_t live) {
    if (placed) { ++placed_; return; }
    char why[96];
    snprintf(why, sizeof(why), "the runtime reserved 0x%llx instead",
             (unsigned long long)live);
    fell_back(rec, size, "hipMemAddressReserve", why);
}

void VaPlacement::release_all() {
#ifndef _WIN32
    std::lock_guard<std::mutex> lk(mu_);
    if (!active_) return;
    for (auto& [pb, m] : mapped_) {
        (void)hipMemUnmap(reinterpret_cast<void*>(pb), m.end - pb);
        (void)hipMemRelease(m.handle);
    }
    mapped_.clear();
    for (const auto& r : reserved_)
        (void)hipMemAddressFree(reinterpret_cast<void*>(r.base), r.end - r.base);
    reserved_.clear();
    // The granule edges reserve() left as placeholders, and every VMM range
    // the recording never got round to reserving.
    for (const auto& r : plan_.alloc) {
        drop_hold(r.base, std::min(r.end, va_ceil(r.base, gran_)));
        drop_hold(std::max(r.base, va_floor(r.end, gran_)), r.end);
    }
    for (const auto& r : vmm_held_) drop_hold(r.base, r.end);
    vmm_held_.clear();
    for (const auto& r : denied_held_) drop_hold(r.base, r.end);
    denied_held_.clear();
    active_ = false;
#endif
}

}  // namespace hrr
