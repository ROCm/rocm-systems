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
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "hrr/hrr_api_args.h"
#include "hrr_reader.h"

struct StreamCaptureFlag {
    int n = 0;
    void begin(uint64_t) { ++n; }
    bool end(uint64_t) { if (!n) return false; n = 0; return true; }
    bool any() const { return n > 0; }
};
inline uint64_t hrr_capture_key(uint64_t s, uint64_t) { return s; }

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

inline uint64_t va_floor(uint64_t v, uint64_t g)  { return 0; }

// Rounds up, or returns 0 when that would wrap.
inline uint64_t va_ceil(uint64_t v, uint64_t g) { return 0; }

// Round every range out to `page`, then merge the ones that overlap or touch.
// Empty ranges, wrapping ranges and ranges outside user VA are dropped.
inline std::vector<VaRange> va_round_merge(std::vector<VaRange> in, uint64_t page) { return {}; }

// `from` minus `minus`. Both sorted and merged, as va_round_merge returns them.
inline std::vector<VaRange> va_subtract(const std::vector<VaRange>& from,
                                        const std::vector<VaRange>& minus) { return {}; }

// The range in `sorted` that contains all of [b, e), or nullptr.
inline const VaRange* va_find_containing(const std::vector<VaRange>& sorted,
                                         uint64_t b, uint64_t e) { return nullptr; }

// Whether any range in `sorted` overlaps [b, e).
inline bool va_overlaps(const std::vector<VaRange>& sorted, uint64_t b, uint64_t e) { return false; }

// Which allocation APIs are placed, and what range one recorded call claimed.
// Only successful calls count: a failed one returned no address. Managed
// memory, and hipExtMallocWithFlags with any flag set (fine-grained, uncached,
// signal, contiguous), have no VMM equivalent and keep their old path.
enum class PlaceKind { None, Alloc, Vmm };

inline PlaceKind placement_event_range(const Event& ev, VaRange* out) { return PlaceKind::None; }

// The address a recorded call exported for another process, or 0. Both APIs
// refuse VMM memory, so the allocation holding that address keeps the old path.
inline uint64_t placement_exported_ptr(const Event& ev) { return 0; }

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
                                    uint64_t page = kPlacePage) { return {}; }

// The ranges reserve() asks for one by one when the merged range [b, e) could
// not be reserved whole: each allocation's own range rounded out to the VMM
// granularity and clipped to [b, e). Two allocations that round into the same
// granule share one reservation; ranges that only touch stay separate, so one
// that cannot be reserved costs only itself.
inline std::vector<VaRange> va_reserve_pieces(const std::vector<VaRange>& pieces,
                                              uint64_t b, uint64_t e, uint64_t gran) { return {}; }

// The ranges /proc/self/maps lists, sorted and merged. Text in, so the tests
// can feed it a fixture.
inline std::vector<VaRange> parse_proc_maps(const std::string& text) { return {}; }

// A recorded call that allocated device memory placement does not place, with
// the range it claimed (rec may be 0 when the API returns no plain pointer)
// and why. Only successful calls count.
struct Unplaced {
    uint64_t    rec  = 0;
    uint64_t    size = 0;
    const char* api  = "";
    std::string why;
};

inline bool placement_unplaced(const uint8_t* data, size_t n, Unplaced* out) { return false; }

inline bool placement_unplaced(const Event& ev, Unplaced* out) { return false; }

// Whether the recording reached memory on more than one device: two devices
// selected, peer access enabled, a pool shared, or a peer copy. Placed
// mappings are then made accessible to every peer that can reach them, as
// enabling peer access would have done for ordinary allocations.
inline bool placement_needs_peer_access(const std::vector<Event>& events) { return false; }

// Parse HIP_HRR_REPLAY_PLACE_DENY: addresses separated by commas, any base
// strtoull accepts with base 0.
inline std::vector<uint64_t> parse_place_deny(const char* s) { return {}; }

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

// A placed mapping: page range [key, end), the recorded allocation base, and
// its physical handle.
struct PlacedMapping {
    uint64_t end;
    uint64_t rec;
    hipMemGenericAllocationHandle_t handle;
    int  device    = 0;
    bool unmapping = false;
};
using PlacedMap = std::map<uint64_t, PlacedMapping>;

// The mapping in `m` that overlaps [pb, pe), or nullptr. Only the one nearest
// below pe can, because mappings never overlap each other.
inline const PlacedMapping* va_mapping_overlapping(const PlacedMap& m, uint64_t pb,
                                                   uint64_t pe) { return nullptr; }

inline void va_untrack_mapping(std::map<uint64_t, size_t>& m, uint64_t va, size_t) { m.erase(va); }
inline void va_track_mapping(std::map<uint64_t, size_t>& m, uint64_t va, size_t size) { m[va] = size; }
inline hipError_t hrr_record_free_event(hipStream_t, hipEvent_t*) { return hipErrorNotSupported; }
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
    int (*stream_device)(hipStream_t s)                       = nullptr;
    void (*flush_tlb)()                                       = nullptr;
};
using UnmapOps = VmmOps;
inline size_t proc_maps_reads_for_test() { return 0; }

class VaPlacement {
  public:
    void set_unmap_ops_for_test(UnmapOps) {}
    void adopt_mapping_for_test(uint64_t, size_t, int = 0) {}
    void set_vmm_ops_for_test(VmmOps) {}
    void adopt_reservation_for_test(uint64_t, uint64_t, int) {}
    void restore_vmm_holds(const std::vector<VaRange>&) {}
    size_t drain_for_retry(hipError_t, bool) { return 0; }
    bool unmap_async(void* live, hipStream_t, bool) { return unmap(live, true); }
    void stream_destroyed(hipStream_t) {}
    bool map_at(uint64_t rec, size_t size, int device, const char* api, void** live,
                bool capturing, const hipStream_t*) {
        return map_at(rec, size, device, api, live, capturing);
    }
    bool hold_denied(const std::vector<VaRange>&) { return true; }
    // Before hipInit: hold every planned range with a placeholder. Returns
    // false when the platform cannot (Windows), which leaves placement off.
    bool hold(PlacementPlan plan);

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
    // reporting why; the caller then allocates the old way. `capturing` says
    // whether any thread is inside a graph capture, when unmapping is illegal.
    bool map_at(uint64_t rec, size_t size, int device, const char* api, void** live,
                bool capturing = false);

    // Note a fallback decided outside map_at, e.g. during graph capture.
    void fell_back(uint64_t rec, size_t size, const char* api, const char* why);

    // If `live` is a placed mapping, unmap it and release its handle. The
    // reservation stays, so the next allocation recorded there lands again.
    // With `defer` (a graph capture is open, and hipMemUnmap would wait for
    // the capturing stream) it moves to a list drain_deferred() unmaps later;
    // until then nothing is placed over it.
    bool unmap(void* live, bool defer = false);
    // Whether `live` is the base of a live placed mapping.
    bool is_mapped(void* live);
    // Unmap everything unmap() deferred. Call only when no capture is open.
    size_t drain_deferred();

    // hipMemAddressReserve: give back the placeholder over [base, base+size) so
    // the reserve at that hint can take it. False when placement does not hold
    // that range.
    bool release_vmm_hold(uint64_t base, size_t size);
    // hipMemAddressFree: hold the range again for a later reserve there.
    void restore_vmm_hold(uint64_t base, size_t size);
    // The replayed hipMemAddressReserve returned `live`. Counts it, and when
    // `held` and the runtime put it elsewhere, holds the recorded range again.
    void vmm_reserved(uint64_t rec, size_t size, bool held, uint64_t live);

    // Teardown: unmap whatever is still mapped, then free the reservations
    // and drop every placeholder.
    void release_all();

    // Between the warm-up pass and the timed pass.
    void reset_counts() { placed_ = 0; fallbacks_ = 0; deferred_total_ = 0; }

    uint64_t placed() const { return placed_.load(); }
    uint64_t fallbacks() const { return fallbacks_.load(); }
    uint64_t deferred_unmaps() const { return deferred_total_.load(); }
    uint64_t held_bytes() const { return held_bytes_; }
    size_t   held_ranges() const { return reserved_.size(); }
    size_t   lost_ranges() const { return lost_ranges_; }
    std::vector<uint64_t> mapped_bases();

  private:
    bool unmap_one(uint64_t pb, const PlacedMapping& m);
    void reserve_line(bool whole, uint64_t b, uint64_t e, const char* why);

    std::mutex mu_;
    bool active_  = false;
    bool verbose_ = false;
    int  device_count_ = 0;
    uint64_t gran_ = kPlacePage;
    PlacementPlan plan_;
    std::vector<VaRange> reserved_;        // placement-owned reservations, sorted
    std::vector<VaRange> alloc_holds_;     // allocation placeholders left after reserve()
    PlacedMap mapped_;                     // page base -> live mapping
    PlacedMap deferred_;                   // freed under capture, still mapped
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
