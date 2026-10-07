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

// Report at most this many fallbacks, and as many reserve failures, one by
// one unless --verbose. The rest are counted: past a handful, the lines stop
// helping anyone find the one that matters.
constexpr uint64_t kFallbackLines = 16;

using ull = unsigned long long;

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
// mapped is split in half until the free pieces are found.
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
#else
bool hold_exact(uint64_t, uint64_t) { return false; }
void drop_hold(uint64_t, uint64_t) {}
#endif

}  // namespace

std::vector<VaRange> read_proc_maps() {
#ifndef _WIN32
    std::string text;
    if (FILE* f = fopen("/proc/self/maps", "r")) {
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
        fclose(f);
    }
    return parse_proc_maps(text);
#else
    return {};
#endif
}

void hold_free_pieces(uint64_t b, uint64_t e, const std::vector<VaRange>& occupied,
                      std::vector<VaRange>* out) {
#ifndef _WIN32
    if (b >= e) return;
    for (const auto& r : va_subtract({{b, e}}, occupied)) hold_pieces(r.base, r.end, out);
#else
    (void)b; (void)e; (void)occupied; (void)out;
#endif
}

hipError_t hrr_vmm_map_into(void* va, size_t len, int device,
                            hipMemGenericAllocationHandle_t* out_handle,
                            const std::vector<int>& peers) {
    hipMemAllocationProp prop{};
    prop.type          = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id   = device;

    hipMemGenericAllocationHandle_t handle{};
    hipError_t r = hipMemCreate(&handle, len, &prop, 0);
    if (r != hipSuccess) return r;

    r = hipMemMap(va, len, 0, handle, 0);
    if (r != hipSuccess) { (void)hipMemRelease(handle); return r; }

    std::vector<hipMemAccessDesc> desc(1 + peers.size());
    for (size_t i = 0; i < desc.size(); ++i) {
        desc[i].location.type = hipMemLocationTypeDevice;
        desc[i].location.id   = i == 0 ? device : peers[i - 1];
        desc[i].flags         = hipMemAccessFlagsProtReadWrite;
    }
    r = hipMemSetAccess(va, len, desc.data(), desc.size());
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
    // One read of the address map serves all three lists: they are disjoint.
    const std::vector<VaRange> occ = read_proc_maps();
    std::vector<VaRange> held;
    for (const auto& r : plan.alloc) hold_free_pieces(r.base, r.end, occ, &held);
    plan.alloc = va_round_merge(std::move(held), kPlacePage);
    alloc_holds_ = plan.alloc;
    held.clear();
    for (const auto& r : plan.vmm) hold_free_pieces(r.base, r.end, occ, &held);
    vmm_held_ = va_round_merge(std::move(held), kPlacePage);
    // A denied range is held too, so the fallback cannot land back on it by
    // chance: the runtime has to put that allocation somewhere else.
    held.clear();
    for (const auto& r : plan.denied) hold_free_pieces(r.base, r.end, occ, &held);
    denied_held_ = std::move(held);
    plan_ = std::move(plan);
    active_ = true;
    return true;
#endif
}

void VaPlacement::reserve_line(bool whole, uint64_t b, uint64_t e, const char* why) {
    if (++reserve_lines_ > kFallbackLines && !verbose_) return;
    if (whole)
        fprintf(stderr,
                "[HRR] Placement: could not reserve 0x%llx-0x%llx as one range (%s); "
                "reserving each allocation in it on its own\n",
                (ull)b, (ull)e, why);
    else
        fprintf(stderr,
                "[HRR] Placement: could not reserve 0x%llx-0x%llx at its recorded "
                "address (%s); allocations there will move\n",
                (ull)b, (ull)e, why);
}

bool VaPlacement::reserve(int device_count, bool peer_access) {
#ifndef _WIN32
    std::unique_lock<std::mutex> lk(mu_);
    if (!active_) return false;
    for (int d = 0; d < device_count; ++d) {
        int vmm = 0;
        if (hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported,
                                  d) != hipSuccess || !vmm) {
            fprintf(stderr,
                    "[HRR] Placement : off (device %d does not support virtual memory "
                    "management)\n", d);
            lk.unlock();
            release_all();
            return false;
        }
    }
    device_count_ = device_count;
    for (int d = 0; d < device_count; ++d) {
        hipMemAllocationProp prop{};
        prop.type          = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id   = d;
        size_t g = 0;
        if (hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum)
                == hipSuccess && g > gran_)
            gran_ = g;
    }
    peers_.assign(device_count, {});
    if (peer_access && device_count > 1)
        for (int d = 0; d < device_count; ++d)
            for (int p = 0; p < device_count; ++p) {
                int can = 0;
                // Can device p reach memory that lives on device d?
                if (p != d && hipDeviceCanAccessPeer(&can, p, d) == hipSuccess && can)
                    peers_[d].push_back(p);
            }

    std::vector<VaRange> holds;
    for (const auto& r : plan_.alloc) {
        // A granularity coarser than the page shrinks the range to whole
        // granules. The edges keep their placeholders and the allocations
        // there fall back.
        const uint64_t b = va_ceil(r.base, gran_);
        const uint64_t e = va_floor(r.end, gran_);
        if (b == 0 || b >= e) { holds.push_back(r); ++lost_ranges_; continue; }
        if (r.base < b) holds.push_back({r.base, b});
        if (e < r.end) holds.push_back({e, r.end});
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
        reserve_line(true, b, e,
                     err == hipSuccess ? "the runtime returned another address"
                                       : hipGetErrorString(err));
        // Hold the range again, then reserve each allocation's own granules,
        // so one obstacle costs only the allocations it touches.
        std::vector<VaRange> re;
        hold_free_pieces(b, e, read_proc_maps(), &re);
        re = va_round_merge(std::move(re), kPlacePage);
        for (const auto& p : va_reserve_pieces(plan_.pieces, b, e, gran_)) {
            if (!va_find_containing(re, p.base, p.end)) {
                ++lost_ranges_;
                reserve_line(false, p.base, p.end, "something else is mapped there");
                continue;
            }
            drop_hold(p.base, p.end);
            void* pv = nullptr;
            const hipError_t pr = hipMemAddressReserve(&pv, p.end - p.base, 0,
                                                       reinterpret_cast<void*>(p.base), 0);
            if (pr == hipSuccess && reinterpret_cast<uint64_t>(pv) == p.base) {
                reserved_.push_back(p);
                held_bytes_ += p.end - p.base;
                re = va_subtract(re, {p});
                continue;
            }
            if (pr == hipSuccess) (void)hipMemAddressFree(pv, p.end - p.base);
            ++lost_ranges_;
            reserve_line(false, p.base, p.end,
                         pr == hipSuccess ? "the runtime returned another address"
                                          : hipGetErrorString(pr));
            if (!hold_exact(p.base, p.end)) re = va_subtract(re, {p});
        }
        holds.insert(holds.end(), re.begin(), re.end());
    }
    if (reserve_lines_ > kFallbackLines && !verbose_)
        fprintf(stderr,
                "[HRR] Placement: %zu more reserve failure(s) not shown; --verbose "
                "names every one\n", reserve_lines_ - kFallbackLines);
    std::sort(reserved_.begin(), reserved_.end(),
              [](const VaRange& x, const VaRange& y) { return x.base < y.base; });
    alloc_holds_ = std::move(holds);
    return true;
#else
    (void)device_count; (void)peer_access;
    return false;
#endif
}

void VaPlacement::fell_back(uint64_t rec, size_t size, const char* api,
                            const char* why) {
    const bool first = fallbacks_++ == 0;
    const uint64_t n = ++lines_;
    char where[40] = "";
    if (rec) snprintf(where, sizeof(where), " 0x%llx", (ull)rec);
    if (n <= kFallbackLines || verbose_) {
        char bytes[40] = "";
        if (size) snprintf(bytes, sizeof(bytes), " (%zu bytes)", size);
        fprintf(stderr,
                "[HRR] Placement: %s%s%s not placed at its recorded address: %s. It "
                "replays elsewhere, so a copy of its address stored in device memory "
                "is stale\n",
                api, where, bytes, why);
    } else if (n == kFallbackLines + 1) {
        fprintf(stderr,
                "[HRR] Placement: further fallbacks are only counted; --verbose "
                "names every one\n");
    }
    // The first fallback is what turns the H2D scan on (replay_memcpy_impl),
    // so say so once, and name it, rather than on every fallback line.
    if (first)
        fprintf(stderr,
                "[HRR] Placement: from here on, replay scans the payload of each "
                "host-to-device hipMemcpy, hipMemcpyAsync, hipMemcpyHtoD, "
                "hipMemcpyHtoDAsync and hipMemcpyWithStream for addresses of allocations that moved, "
                "because %s%s did not land at its recorded address\n",
                api, where);
}

bool VaPlacement::map_at(uint64_t rec, size_t size, int device, const char* api,
                         void** live, bool capturing) {
    if (!active_ || size == 0 || rec > UINT64_MAX - size) return false;
    const uint64_t pb = va_floor(rec, gran_);
    const uint64_t pe = va_ceil(rec + size, gran_);
    char buf[128];
    const char* why = nullptr;
    for (int attempt = 0; attempt < 2 && !why; ++attempt) {
        uint64_t freed = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (pe != 0 && va_overlaps(plan_.exported, pb, pe)) {
                why = "the recording exports it to another process, which VMM memory "
                      "does not support";
                break;
            }
            if (pe == 0 || !va_find_containing(reserved_, pb, pe)) {
                why = "its range could not be held";
                break;
            }
            if (device < 0 || device >= device_count_) {
                snprintf(buf, sizeof(buf), "device %d is not one replay can see", device);
                why = buf;
                break;
            }
            if (const auto* m = va_mapping_overlapping(deferred_, pb, pe)) {
                freed = m->rec;
            } else if (const auto* l = va_mapping_overlapping(mapped_, pb, pe)) {
                snprintf(buf, sizeof(buf), "it shares a page with live allocation 0x%llx",
                         (ull)l->rec);
                why = buf;
                break;
            } else {
                hipMemGenericAllocationHandle_t h{};
                const hipError_t r = hrr_vmm_map_into(reinterpret_cast<void*>(pb), pe - pb,
                                                      device, &h, peers_[device]);
                if (r == hipSuccess) {
                    mapped_[pb] = {pe, rec, h};
                    *live = reinterpret_cast<void*>(rec);
                    ++placed_;
                    return true;
                }
                snprintf(buf, sizeof(buf), "mapping it failed (%s)", hipGetErrorString(r));
                why = buf;
                break;
            }
        }
        // A mapping freed there is still mapped, its unmap deferred. Unmap the
        // deferred ones now if no capture is open, then look again; never map
        // over one, because a capture or a stream may still use it. The drain
        // can find nothing when another thread drained first; the second look
        // tells.
        if (capturing || attempt == 1) {
            snprintf(buf, sizeof(buf), "allocation 0x%llx, freed there, is still mapped (%s)",
                     (ull)freed, capturing ? "a graph capture is open" : "its unmap failed");
            why = buf;
        } else {
            (void)drain_deferred();
        }
    }
    fell_back(rec, size, api, why);
    return false;
}

bool VaPlacement::unmap_one(uint64_t pb, const PlacedMapping& m) {
    // hipMemUnmap waits for every stream, so nothing still queued can touch
    // the pages once they are gone. The reservation stays.
    hipError_t r = hipMemUnmap(reinterpret_cast<void*>(pb), m.end - pb);
    if (r != hipSuccess) {
        fprintf(stderr,
                "[HRR] Placement: hipMemUnmap of 0x%llx (%llu bytes) failed (%s); the "
                "mapping stays and nothing is placed over it\n",
                (ull)m.rec, (ull)(m.end - pb), hipGetErrorString(r));
        return false;
    }
    r = hipMemRelease(m.handle);
    if (r != hipSuccess)
        fprintf(stderr,
                "[HRR] Placement: hipMemRelease for 0x%llx failed (%s); its memory is "
                "lost until exit\n",
                (ull)m.rec, hipGetErrorString(r));
    return true;
}

bool VaPlacement::unmap(void* live, bool defer) {
    if (!active_ || !live) return false;
    const uint64_t v = reinterpret_cast<uint64_t>(live);
    uint64_t pb = 0;
    PlacedMapping m{};
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = mapped_.upper_bound(v);
        if (it == mapped_.begin()) return false;
        --it;
        if (it->second.rec != v) return false;
        pb = it->first;
        m  = it->second;
        mapped_.erase(it);
        if (defer) {
            deferred_[pb] = m;
            ++deferred_total_;
            return true;
        }
    }
    if (!unmap_one(pb, m)) {
        // Still mapped: keep it where map_at will not place over it and the
        // next drain tries again.
        std::lock_guard<std::mutex> lk(mu_);
        deferred_[pb] = m;
    }
    return true;
}

size_t VaPlacement::drain_deferred() {
    PlacedMap work;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (deferred_.empty()) return 0;
        work.swap(deferred_);
    }
    size_t n = 0;
    PlacedMap kept;
    for (const auto& [pb, m] : work) {
        if (unmap_one(pb, m)) ++n;
        else kept[pb] = m;
    }
    if (!kept.empty()) {
        std::lock_guard<std::mutex> lk(mu_);
        deferred_.insert(kept.begin(), kept.end());
    }
    return n;
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

std::vector<uint64_t> VaPlacement::mapped_bases() {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<uint64_t> out;
    for (const auto& kv : mapped_) out.push_back(kv.second.rec);
    return out;
}

bool VaPlacement::release_vmm_hold(uint64_t base, size_t size) {
#ifndef _WIN32
    if (!active_ || size == 0) return false;
    std::lock_guard<std::mutex> lk(mu_);
    const uint64_t b = base, e = base + size;
    std::vector<VaRange> dropped;
    for (const auto& r : vmm_held_) {
        const uint64_t lo = std::max(r.base, b), hi = std::min(r.end, e);
        if (lo < hi) { drop_hold(lo, hi); dropped.push_back({lo, hi}); }
    }
    if (dropped.empty()) return false;
    vmm_held_ = va_subtract(vmm_held_, {{b, e}});
    vmm_released_[base] = std::move(dropped);
    return true;
#else
    (void)base; (void)size;
    return false;
#endif
}

void VaPlacement::restore_vmm_hold(uint64_t base, size_t size) {
#ifndef _WIN32
    if (!active_ || size == 0) return;
    std::lock_guard<std::mutex> lk(mu_);
    // Hold again exactly what release_vmm_hold gave up, minus whatever the
    // runtime has put there since.
    std::vector<VaRange> want;
    auto it = vmm_released_.find(base);
    if (it != vmm_released_.end()) {
        want = std::move(it->second);
        vmm_released_.erase(it);
    } else {
        want.push_back({base, base + size});
    }
    const std::vector<VaRange> occ = read_proc_maps();
    std::vector<VaRange> got = vmm_held_;
    for (const auto& r : want) hold_free_pieces(r.base, r.end, occ, &got);
    vmm_held_ = va_round_merge(std::move(got), kPlacePage);
#else
    (void)base; (void)size;
#endif
}

void VaPlacement::vmm_reserved(uint64_t rec, size_t size, bool held, uint64_t live) {
    if (live == rec) { ++placed_; return; }
    char why[96];
    snprintf(why, sizeof(why), "the runtime reserved 0x%llx instead", (ull)live);
    fell_back(rec, size, "hipMemAddressReserve", why);
    // The recorded range is free again: hold it for a later reserve there.
    if (held) restore_vmm_hold(rec, size);
}

void VaPlacement::release_all() {
#ifndef _WIN32
    std::lock_guard<std::mutex> lk(mu_);
    if (!active_) return;
    for (const auto* m : {&mapped_, &deferred_})
        for (const auto& [pb, mm] : *m) (void)unmap_one(pb, mm);
    mapped_.clear();
    deferred_.clear();
    for (const auto& r : reserved_)
        (void)hipMemAddressFree(reinterpret_cast<void*>(r.base), r.end - r.base);
    reserved_.clear();
    for (const auto& r : alloc_holds_) drop_hold(r.base, r.end);
    alloc_holds_.clear();
    for (const auto& r : vmm_held_) drop_hold(r.base, r.end);
    vmm_held_.clear();
    vmm_released_.clear();
    for (const auto& r : denied_held_) drop_hold(r.base, r.end);
    denied_held_.clear();
    active_ = false;
#endif
}

}  // namespace hrr
