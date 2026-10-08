/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
/* hrr_va_placement.cpp — the HIP half of capture-address placement. */

#include "hrr_va_placement.h"

#include <cerrno>
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
enum class Hold { Held, Taken, Failed };

// Hold [b, e) with an inaccessible placeholder. MAP_FIXED_NOREPLACE fails
// instead of moving, and a kernel too old to know the flag treats the address
// as a hint, so the returned address is checked either way. Taken means
// something is mapped in the range; Failed is any other refusal, such as
// ENOMEM under RLIMIT_AS, which a smaller piece would meet too.
Hold hold_exact(uint64_t b, uint64_t e) {
    void* want = reinterpret_cast<void*>(b);
    void* p = mmap(want, e - b, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE,
                   -1, 0);
    if (p == MAP_FAILED) return errno == EEXIST ? Hold::Taken : Hold::Failed;
    if (p != want) { munmap(p, e - b); return Hold::Taken; }
    return Hold::Held;
}

// Hold what can be held of [b, e). A range that collides with something
// mapped is split in half until the free pieces are found. Any other failure
// stops there: splitting would repeat it once per page.
void hold_pieces(uint64_t b, uint64_t e, std::vector<VaRange>* out) {
    if (b >= e) return;
    const Hold h = hold_exact(b, e);
    if (h == Hold::Held) { out->push_back({b, e}); return; }
    if (h == Hold::Failed || e - b <= kPlacePage) return;
    const uint64_t mid = va_floor(b + (e - b) / 2, kPlacePage);
    hold_pieces(b, mid, out);
    hold_pieces(mid, e, out);
}

void drop_hold(uint64_t b, uint64_t e) {
    if (b < e) munmap(reinterpret_cast<void*>(b), e - b);
}
#endif

}  // namespace

#ifdef HRR_VA_PLACEMENT_TESTING
namespace {
std::atomic<size_t> g_proc_maps_reads{0};
}  // namespace
size_t proc_maps_reads_for_test() { return g_proc_maps_reads.load(); }
#endif

std::vector<VaRange> read_proc_maps() {
#ifndef _WIN32
#ifdef HRR_VA_PLACEMENT_TESTING
    ++g_proc_maps_reads;
#endif
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

hipError_t hrr_record_free_event(hipStream_t stream, hipEvent_t* event) {
    hipEvent_t e = nullptr;
    hipError_t r = hipEventCreateWithFlags(&e, hipEventDisableTiming);
    if (r != hipSuccess) return r;
    r = hipEventRecord(e, stream);
    if (r != hipSuccess) {
        (void)hipEventDestroy(e);
        return r;
    }
    *event = e;
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

bool VaPlacement::hold_denied(const std::vector<VaRange>& denied) {
#ifdef _WIN32
    (void)denied;
    return false;
#else
    std::lock_guard<std::mutex> lk(mu_);
    const std::vector<VaRange> occ = read_proc_maps();
    for (const auto& r : denied) hold_free_pieces(r.base, r.end, occ, &denied_held_);
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
            if (hold_exact(p.base, p.end) != Hold::Held) re = va_subtract(re, {p});
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

hipError_t VaPlacement::vmm_map(uint64_t pb, uint64_t pe, int device,
                                hipMemGenericAllocationHandle_t* h) {
    void* va = reinterpret_cast<void*>(pb);
#ifdef HRR_VA_PLACEMENT_TESTING
    return ops_.map(va, pe - pb, device, h, peers_[device]);
#else
    return hrr_vmm_map_into(va, pe - pb, device, h, peers_[device]);
#endif
}

// A failed HIP call leaves a sticky error. The recorded allocation succeeded,
// so the replayed program must not find one in its next hipGetLastError.
void VaPlacement::clear_error() {
#ifdef HRR_VA_PLACEMENT_TESTING
    if (ops_.clear_error) ops_.clear_error();
#else
    (void)hipGetLastError();
#endif
}

void VaPlacement::drop_event(hipEvent_t e) {
    if (!e) return;
#ifdef HRR_VA_PLACEMENT_TESTING
    const hipError_t r = ops_.destroy_event(e);
#else
    const hipError_t r = hipEventDestroy(e);
#endif
    if (r != hipSuccess) clear_error();
}

void VaPlacement::wait_for_free(hipEvent_t e, const hipStream_t* stream) {
#ifdef HRR_VA_PLACEMENT_TESTING
    hipError_t (*wait)(hipStream_t, hipEvent_t, unsigned int) = ops_.wait_event;
    hipError_t (*sync)(hipEvent_t) = ops_.sync_event;
#else
    hipError_t (*wait)(hipStream_t, hipEvent_t, unsigned int) = hipStreamWaitEvent;
    hipError_t (*sync)(hipEvent_t) = hipEventSynchronize;
#endif
    // A stream that cannot be made to wait is waited for by the host instead.
    if (stream && wait(*stream, e, 0) == hipSuccess) return;
    if (stream) clear_error();
    if (sync(e) != hipSuccess) clear_error();
}

bool VaPlacement::any_unmapping() const {
    for (const auto& kv : deferred_)
        if (kv.second.unmapping) return true;
    return false;
}

bool VaPlacement::map_at(uint64_t rec, size_t size, int device, const char* api,
                         void** live, bool capturing, const hipStream_t* stream) {
    if (!active_ || size == 0 || rec > UINT64_MAX - size) return false;
    const uint64_t pb = va_floor(rec, gran_);
    const uint64_t pe = va_ceil(rec + size, gran_);
    char buf[160];
    const char* why = nullptr;
    bool drained_overlap = false;  // this call unmapped the freed mappings here
    bool drained_oom     = false;  // this call unmapped every freed mapping
    std::vector<uint64_t> drained;
    std::unique_lock<std::mutex> lk(mu_);
    while (!why) {
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
        // A live allocation there means a fallback whatever else is freed
        // there, so nothing is unmapped for it.
        if (const auto* l = va_mapping_overlapping(mapped_, pb, pe)) {
            snprintf(buf, sizeof(buf), "it shares a page with live allocation 0x%llx",
                     (ull)l->rec);
            why = buf;
            break;
        }
        const std::vector<uint64_t> freed = va_mappings_overlapping(deferred_, pb, pe);
        bool busy = false;
        for (uint64_t k : freed) busy |= deferred_.at(k).unmapping;
        if (busy) {
            // Another thread's unmap there is still running: the pages are
            // still mapped. Look again when it is over.
            cv_.wait(lk);
            continue;
        }
        if (freed.size() == 1) {
            auto it = deferred_.find(freed[0]);
            const PlacedMapping& d = it->second;
            // Freed from the same first page on this device, and big enough:
            // a stream-ordered pool hands a block back for a request up to
            // 12.5% smaller. Take the mapping back as it is, keeping its end,
            // as the pool kept the whole block. That needs no unmap, so no
            // device-wide wait. The recording's pool reused the block only
            // once the free was done, or ordered after it, so replay orders
            // the allocation after the free too: nothing to do on the stream
            // that freed it; on another, or for an allocation with no stream,
            // wait for the event the free left, unless a capture is open,
            // where that wait would sync inside it.
            const bool same_stream = stream && d.on_stream && d.stream == *stream;
            if (it->first == pb && pe <= d.end && d.device == device &&
                (same_stream || (!capturing && d.event))) {
                PlacedMapping m = d;
                const hipEvent_t ev = d.event;
                m.rec = rec;
                m.on_stream = false;
                m.stream = nullptr;
                m.event = nullptr;
                deferred_.erase(it);
                mapped_[pb] = m;
                ++placed_;
                lk.unlock();
                if (ev && !same_stream) wait_for_free(ev, stream);
                drop_event(ev);
                *live = reinterpret_cast<void*>(rec);
                return true;
            }
            // Anything else over it is an ordinary overlap, below: unmapped
            // first, since hipMemUnmap waits for every stream, or a fallback
            // while a capture is open.
        }
        if (!freed.empty()) {
            // Never map over a freed mapping: a capture or a stream may still
            // use it. Unmap it first if no capture is open, then look again.
            if (capturing || drained_overlap) {
                const uint64_t k = freed[0];
                const bool failed =
                    std::find(drained.begin(), drained.end(), k) != drained.end();
                snprintf(buf, sizeof(buf), "allocation 0x%llx, freed there, is still mapped (%s)",
                         (ull)deferred_.at(k).rec,
                         capturing ? "a graph capture is open, and unmapping it would sync inside it"
                         : failed  ? "its unmap failed"
                                   : "another thread freed it there while this one unmapped");
                why = buf;
                break;
            }
            drained_overlap = true;
            drained = freed;
            (void)drain_locked(lk, freed);
            continue;
        }
        PlacedMapping m{pe, rec, {}, device, false};
        const hipError_t r = vmm_map(pb, pe, device, &m.handle);
        if (r == hipSuccess) {
            mapped_[pb] = m;
            *live = reinterpret_cast<void*>(rec);
            ++placed_;
            return true;
        }
        clear_error();
        // Freed mappings still hold their memory until a sync drains them.
        // Out of memory with some waiting: unmap them all, then try again.
        if (r == hipErrorOutOfMemory && !capturing && !drained_oom && !deferred_.empty()) {
            drained_oom = true;
            cv_.wait(lk, [&] { return !any_unmapping(); });
            std::vector<uint64_t> all;
            for (const auto& kv : deferred_) all.push_back(kv.first);
            (void)drain_locked(lk, all);
            continue;
        }
        snprintf(buf, sizeof(buf), "mapping it failed (%s)", hipGetErrorString(r));
        why = buf;
    }
    lk.unlock();
    fell_back(rec, size, api, why);
    return false;
}

bool VaPlacement::unmap_one(uint64_t pb, const PlacedMapping& m) {
    // hipMemUnmap waits for every stream, so nothing still queued can touch
    // the pages once they are gone. The reservation stays.
#ifdef HRR_VA_PLACEMENT_TESTING
    hipError_t r = ops_.unmap(reinterpret_cast<void*>(pb), m.end - pb);
#else
    hipError_t r = hipMemUnmap(reinterpret_cast<void*>(pb), m.end - pb);
#endif
    if (r != hipSuccess) {
        fprintf(stderr,
                "[HRR] Placement: hipMemUnmap of 0x%llx (%llu bytes) failed (%s); the "
                "mapping stays and nothing is placed over it\n",
                (ull)m.rec, (ull)(m.end - pb), hipGetErrorString(r));
        clear_error();
        return false;
    }
#ifdef HRR_VA_PLACEMENT_TESTING
    r = ops_.release(m.handle);
#else
    r = hipMemRelease(m.handle);
#endif
    if (r != hipSuccess) {
        fprintf(stderr,
                "[HRR] Placement: hipMemRelease for 0x%llx failed (%s); its memory is "
                "lost until exit\n",
                (ull)m.rec, hipGetErrorString(r));
        clear_error();
    }
    return true;
}

// hipMemUnmap waits for every stream on the device, which can take as long as
// the longest kernel queued there, so it never runs under mu_: another
// thread's unrelated alloc or free would wait for it too, and a kernel that
// needs a later replayed event to finish would never be reached. The mapping
// stays in deferred_, marked, while the unmap runs. Until hipMemUnmap returns
// the pages are still mapped, and map_at does not place over them.
size_t VaPlacement::drain_locked(std::unique_lock<std::mutex>& lk,
                                 const std::vector<uint64_t>& keys) {
    std::vector<std::pair<uint64_t, PlacedMapping>> work;
    for (uint64_t k : keys) {
        auto it = deferred_.find(k);
        if (it == deferred_.end() || it->second.unmapping) continue;
        it->second.unmapping = true;
        work.emplace_back(it->first, it->second);
    }
    if (work.empty()) return 0;
    lk.unlock();
    std::vector<char> ok(work.size());
    for (size_t i = 0; i < work.size(); ++i) {
        ok[i] = unmap_one(work[i].first, work[i].second);
        // The unmap waited for every stream, the freeing one included. A
        // failed one keeps its event for the next try.
        if (ok[i]) drop_event(work[i].second.event);
    }
    lk.lock();
    size_t n = 0;
    for (size_t i = 0; i < work.size(); ++i) {
        auto it = deferred_.find(work[i].first);
        if (it == deferred_.end()) continue;
        // A failed unmap stays, still mapped, for the next drain to retry.
        if (ok[i]) { deferred_.erase(it); ++n; }
        else it->second.unmapping = false;
    }
    cv_.notify_all();
    return n;
}

bool VaPlacement::unmap_impl(void* live, bool defer, bool on_stream, hipStream_t stream,
                             bool capturing) {
    if (!active_ || !live) return false;
    const uint64_t v = reinterpret_cast<uint64_t>(live);
    std::unique_lock<std::mutex> lk(mu_);
    auto it = mapped_.upper_bound(v);
    if (it == mapped_.begin()) return false;
    --it;
    if (it->second.rec != v) return false;
    const uint64_t pb = it->first;
    PlacedMapping m = it->second;
    m.on_stream = on_stream;
    m.stream = stream;
    m.event = nullptr;
    // An event on the freeing stream, recorded now, marks the end of the
    // stream's work on this memory. Recording one inside a capture would add
    // it to the graph, so a free made then keeps none.
    if (on_stream && !capturing) {
#ifdef HRR_VA_PLACEMENT_TESTING
        const hipError_t r = ops_.record_event(stream, &m.event);
#else
        const hipError_t r = hrr_record_free_event(stream, &m.event);
#endif
        if (r != hipSuccess) {
            m.event = nullptr;
            clear_error();
        }
    }
    deferred_[pb] = m;
    mapped_.erase(it);
    if (defer) {
        ++deferred_total_;
        return true;
    }
    (void)drain_locked(lk, {pb});
    return true;
}

bool VaPlacement::unmap(void* live, bool defer) {
    return unmap_impl(live, defer, false, nullptr, false);
}

bool VaPlacement::unmap_async(void* live, hipStream_t stream, bool capturing) {
    return unmap_impl(live, true, true, stream, capturing);
}

void VaPlacement::stream_destroyed(hipStream_t stream) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& kv : deferred_)
        if (kv.second.on_stream && kv.second.stream == stream) {
            kv.second.on_stream = false;
            kv.second.stream = nullptr;
        }
}

size_t VaPlacement::drain_deferred() {
    std::unique_lock<std::mutex> lk(mu_);
    std::vector<uint64_t> keys;
    for (const auto& kv : deferred_) keys.push_back(kv.first);
    return drain_locked(lk, keys);
}

size_t VaPlacement::drain_for_retry(hipError_t r, bool capturing) {
    if (!active_ || r != hipErrorOutOfMemory || capturing) return 0;
    return drain_deferred();
}

#ifdef HRR_VA_PLACEMENT_TESTING
void VaPlacement::adopt_reservation_for_test(uint64_t b, uint64_t e, int devices) {
    std::lock_guard<std::mutex> lk(mu_);
    active_       = true;
    device_count_ = devices;
    peers_.assign(devices, {});
    reserved_.push_back({b, e});
    std::sort(reserved_.begin(), reserved_.end(),
              [](const VaRange& x, const VaRange& y) { return x.base < y.base; });
}

void VaPlacement::adopt_mapping_for_test(uint64_t rec, size_t size, int device) {
    std::lock_guard<std::mutex> lk(mu_);
    active_ = true;
    mapped_[va_floor(rec, gran_)] = {va_ceil(rec + size, gran_), rec, {}, device, false};
}
#endif

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
    if (size) restore_vmm_holds({{base, base + size}});
}

void VaPlacement::restore_vmm_holds(const std::vector<VaRange>& ranges) {
#ifndef _WIN32
    if (!active_ || ranges.empty()) return;
    std::lock_guard<std::mutex> lk(mu_);
    // One read of the address map for every range. A range held since is
    // found by hold_free_pieces itself, which splits around what collides.
    const std::vector<VaRange> occ = read_proc_maps();
    std::vector<VaRange> got = vmm_held_;
    for (const auto& r : ranges) {
        if (r.base >= r.end) continue;
        // Hold again exactly what release_vmm_hold gave up, minus whatever
        // the runtime has put there since.
        std::vector<VaRange> want;
        auto it = vmm_released_.find(r.base);
        if (it != vmm_released_.end()) {
            want = std::move(it->second);
            vmm_released_.erase(it);
        } else {
            want.push_back(r);
        }
        for (const auto& w : want) hold_free_pieces(w.base, w.end, occ, &got);
    }
    vmm_held_ = va_round_merge(std::move(got), kPlacePage);
#else
    (void)ranges;
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
    std::unique_lock<std::mutex> lk(mu_);
    for (const auto& r : denied_held_) drop_hold(r.base, r.end);
    denied_held_.clear();
    if (!active_) return;
    cv_.wait(lk, [&] { return !any_unmapping(); });
    PlacedMap left = std::move(mapped_);
    left.insert(deferred_.begin(), deferred_.end());
    mapped_.clear();
    deferred_.clear();
    lk.unlock();
    for (const auto& [pb, mm] : left) {
        (void)unmap_one(pb, mm);
        drop_event(mm.event);
    }
    lk.lock();
    for (const auto& r : reserved_) {
#ifdef HRR_VA_PLACEMENT_TESTING
        (void)ops_.address_free(reinterpret_cast<void*>(r.base), r.end - r.base);
#else
        (void)hipMemAddressFree(reinterpret_cast<void*>(r.base), r.end - r.base);
#endif
    }
    reserved_.clear();
    for (const auto& r : alloc_holds_) drop_hold(r.base, r.end);
    alloc_holds_.clear();
    for (const auto& r : vmm_held_) drop_hold(r.base, r.end);
    vmm_held_.clear();
    vmm_released_.clear();
    active_ = false;
#endif
}

}  // namespace hrr
