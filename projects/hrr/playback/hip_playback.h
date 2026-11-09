/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <hip/hip_runtime.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <new>
#include <map>
#include <set>
#include <unordered_map>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include <unordered_set>

#include "hrr/hrr_api_args.h"  // for HRR_API_COUNT, hrr_api_id_t
#include "hrr_region_map.h"    // external region annotations (regions/*.hrrr)
#include "hrr_replay_limits.h" // bounds on archive-driven waits and retries
#include "hrr_jit_options.h"   // which JIT options carry a pointer

// Draining is skipped during stream graph capture, where a synchronize would break the capture.
inline bool hrr_replayed_h2d_needs_drain(bool in_graph_capture) {
    return !in_graph_capture;
}

// The zero-init injected after a host-synchronous allocation is drained, except during graph capture.
inline bool hrr_zero_init_needs_drain(bool zero_init_enabled,
                                      bool in_graph_capture) {
    return zero_init_enabled && !in_graph_capture;
}

// Byte-exact by default: a recorded blob carries no element type, so reading it as floats would be a guess.
struct HrrD2HTol {
    enum Type { kExact, kF32, kBF16, kF16, kF64 };
    Type   type = kExact;
    double atol = 0.0;
    double rtol = 0.0;
};

// Leaves *out unchanged and returns false unless spec has a known type and two finite non-negative tolerances.
bool hrr_parse_d2h_tol(const char* spec, HrrD2HTol* out);

// Half-open [base, end).
struct HrrVaRange { uint64_t base = 0, end = 0; };

// Ranges to hold before hipInit: recorded allocations rounded out to `page`, merged,
// minus any that overlap an IPC-exported allocation.
inline std::vector<HrrVaRange> hrr_place_plan(std::vector<HrrVaRange> allocs,
                                              const std::vector<uint64_t>& exported,
                                              uint64_t page = 4096) {
    std::sort(allocs.begin(), allocs.end(),
              [](const HrrVaRange& a, const HrrVaRange& b) { return a.base < b.base; });
    for (auto& r : allocs) {
        r.base -= r.base % page;
        r.end += (page - r.end % page) % page;
    }
    std::vector<HrrVaRange> ipc;
    for (const auto& r : allocs)
        if (std::any_of(exported.begin(), exported.end(),
                        [&](uint64_t x) { return x >= r.base && x < r.end; }))
            ipc.push_back(r);
    std::vector<HrrVaRange> out;
    for (const auto& r : allocs) {
        if (r.base == 0 || r.end <= r.base || r.end > (1ull << 47)) continue;
        if (std::any_of(ipc.begin(), ipc.end(),
                        [&](const HrrVaRange& x) { return r.base < x.end && x.base < r.end; }))
            continue;
        if (!out.empty() && r.base <= out.back().end)
            out.back().end = std::max(out.back().end, r.end);
        else
            out.push_back(r);
    }
    return out;
}

// The granule-aligned range [base, end) that placing `size` bytes at the recorded address `rec` maps.
// False if `size` is 0 or the range overflows.
inline bool hrr_place_range(uint64_t rec, uint64_t size, uint64_t gran,
                            uint64_t* base, uint64_t* end) {
    if (size == 0) return false;
    uint64_t last;  // one past the last byte
    if (__builtin_add_overflow(rec, size, &last)) return false;
    uint64_t up;
    if (__builtin_add_overflow(last, gran - 1, &up)) return false;
    *base = rec - rec % gran;
    *end  = up / gran * gran;
    return true;
}

// The placeholders still standing, base -> end, kept merged.
struct HrrHeldRanges {
    std::map<uint64_t, uint64_t> r;
    void add(uint64_t b, uint64_t e) {
        auto it = r.upper_bound(b);
        if (it != r.begin() && std::prev(it)->second >= b) {
            --it;
            b = it->first;
            e = std::max(e, it->second);
            it = r.erase(it);
        }
        while (it != r.end() && it->first <= e) {
            e = std::max(e, it->second);
            it = r.erase(it);
        }
        r[b] = e;
    }
    // Drop [b, e), e > b, if one held range covers all of it.
    bool take(uint64_t b, uint64_t e) {
        auto it = r.upper_bound(b);
        if (it == r.begin()) return false;
        --it;
        const uint64_t hb = it->first, he = it->second;
        if (e > he) return false;
        r.erase(it);
        if (hb < b) r[hb] = b;
        if (e < he) r[e] = he;
        return true;
    }
};


// A recorded pointer that resolves in no allocation, VMM reservation or region.
// Replay does not know that memory.
struct HrrUnmappedNote {
    uint64_t first = 0;  // first recorded pointer the event could not translate
    unsigned count = 0;  // how many it met
};
// Per thread, because each replay thread dispatches its own events.
inline thread_local HrrUnmappedNote hrr_unmapped_note;

// How live_ptr was obtained. It selects the API that releases it at teardown.
enum class AllocKind : uint8_t {
    Device,        // hipMalloc / hipMallocPitch                    -> hipFree
    Managed,       // hipMallocManaged: host-backed, charged to host_budget -> hipFree
    HostMalloc,    // hipHostMalloc / hipMallocHost                 -> hipHostFree
    HostRegister,  // hipHostRegister backing buffer  -> hipHostUnregister
    DevicePtrAlias // hipHostGetDevicePointer result  -> not separately freed
};

struct AllocEntry {
    uint64_t  rec_base;  // recorded GPU base address
    void*     live_ptr;  // live replay GPU base address
    size_t    size;
    AllocKind kind = AllocKind::Device;
};

struct PlaybackContext {
    std::string archive_dir;

    // Protected by map_mutex: shared_lock for reads, unique_lock for writes.
    mutable std::shared_mutex map_mutex;
    // Orders a pinned-host snapshot copy against the free of its destination.
    std::shared_mutex host_free_mu;
    std::unordered_map<uint64_t, hipStream_t>    stream_map;
    std::unordered_map<uint64_t, hipEvent_t>     event_map;
    std::unordered_map<uint64_t, hipModule_t>    module_map;
    std::unordered_map<uint64_t, hipFunction_t>  func_map;
    std::unordered_map<uint64_t, hipMemPool_t>   mempool_map;
    std::unordered_map<uint64_t, hipArray_t>     array_map;
    std::unordered_map<uint64_t, hipMipmappedArray_t> mipmapped_map;
    std::unordered_map<uint64_t, hipGraph_t>     graph_map;
    std::unordered_map<uint64_t, hipGraphNode_t> graph_node_map;
    std::unordered_map<uint64_t, hipGraphExec_t> graph_exec_map;
    std::unordered_map<uint64_t, hipSurfaceObject_t> surface_map;
    std::unordered_map<uint64_t, hipTextureObject_t> texture_map;
    std::unordered_map<uint64_t, hipCtx_t>       ctx_map;
    // Live linker states, so hipLinkComplete and hipLinkDestroy get this process's handle.
    std::unordered_map<uint64_t, hipLinkState_t> link_state_map;

    std::unordered_map<uint64_t, AllocEntry>     alloc_map;

    // __device__ globals, keyed by the recorded host shadow address that
    // hipGetSymbolAddress and hipMemcpy*Symbol received.
    struct SymbolEntry {
        std::string name;
        void*       live_ptr = nullptr;
        size_t      size     = 0;
    };
    std::unordered_map<uint64_t, SymbolEntry>    symbol_map;

    // Code-object modules loaded by hash (not by recorded module handle)
    std::unordered_map<std::string, hipModule_t> co_modules;

    std::unordered_set<std::string> co_no_device_code;

    std::unordered_map<std::string, hipFunction_t> func_cache;

    bool timing            = false;
    bool skip_device_sync  = false;
    bool sync_after_launch = false;  // hipDeviceSynchronize after every kernel launch
    bool sync_after_event  = false;  // hipDeviceSynchronize after EVERY dispatched event
    // Keep replaying after a handler error. Off by default:
    // for a debugging replay the first error is the one that matters.
    bool continue_on_error = false;
    std::atomic<size_t> events_failed{0};
    std::atomic<size_t> code_objects_failed{0};
    // Max wall-clock ms to wait for a device synchronize before declaring the GPU wedged.
    // 0 disables the watchdog.
    unsigned sync_watchdog_ms = 0;
    // When non-zero, dump each pointer argument as recorded -> live
    // for the kernel launch with this 1-based ordinal.
    size_t dump_ptrs_ordinal = 0;
    // When non-zero, read back each pointer argument of the launch with this
    // 1-based ordinal and report any 8-byte word that is a recorded address.
    size_t scan_args_ordinal = 0;
    size_t scan_args_bytes   = 4096;  // per argument
    // Report recorded addresses inside replayed H2D payloads. They are restored verbatim,
    // so a pointer among them reaches the GPU untranslated.
    bool scan_h2d = false;
    // Report each kernel that takes a pointer into host memory.
    // Replay reallocates those buffers but cannot refill them.
    bool audit_host_args = false;
    bool verbose           = false;
    bool validate_d2h      = false;  // perform D2H validation against captured expected data
    HrrD2HTol d2h_tol;               // byte-exact unless HIP_HRR_D2H_TOL opts a float type in
    // The archive header has HRR_FILE_FLAG_PACKED_HOST_RECTS: the host blobs of
    // the pitched copies hold only the copied rows, packed end to end.
    bool packed_host_rects = false;
    // The archive header has HRR_FILE_FLAG_EXT_LAUNCH_EVENT_MARKERS: capture recorded
    // the hipEventRecord start/stop events of hipExtModuleLaunchKernel.
    bool ext_launch_event_markers = false;
    std::string kernel_filter;

    bool trace_kernels         = false;  // one compact line before every kernel launch
    bool trace_sync            = false;  // mark sync begin/done around each launched kernel
    size_t progress_kernel_interval = 0; // print progress every N launched kernels
    double progress_seconds_interval = 0.0; // also print progress at most every N seconds
    std::chrono::steady_clock::time_point progress_start_time{};
    std::chrono::steady_clock::time_point progress_last_time{};
    std::mutex progress_mutex;

    // Parsed --replace-kernel NAME=path pairs: the kernel recorded as NAME
    // launches from the code object at `path`.
    std::vector<std::pair<std::string, std::string>> kernel_replacements;
    // Resolved replacement functions, keyed by the recorded kernel name.
    // Guarded by map_mutex. Modules backing them are owned in replacement_modules.
    std::unordered_map<std::string, hipFunction_t> replacement_funcs;
    std::vector<hipModule_t> replacement_modules;  // unloaded at teardown

    bool in_graph_capture  = false;

    // Global submission order for MT replay.
    std::atomic<uint64_t> next_seq{0};

    // Set to true by dispatch_event on the first HIP error. All replay threads
    // check this at the top of their event loop and exit immediately.
    mutable std::atomic<bool> fatal_error{false};

    // Host memory held for the archive, against --max-host-bytes (hrr_replay_limits.h).
    // A refused charge reports the site on stderr and ends the replay.
    hrr::ByteBudget host_budget{hrr::kDefaultMaxHostBytes};
    bool charge_host(uint64_t bytes, const char* what) {
        if (host_budget.try_charge(bytes)) return true;
        report_host_refusal(bytes, what);
        return false;
    }
    // The cap limits the whole run, so a refusal ends the replay even with --continue-on-error.
    void report_host_refusal(uint64_t bytes, const char* what) {
        fprintf(stderr,
                "[HRR] Fatal: %s needs %llu host bytes; %llu of the %llu-byte host "
                "memory cap (--max-host-bytes) is already in use\n",
                what, (unsigned long long)bytes, (unsigned long long)host_budget.used(),
                (unsigned long long)host_budget.cap());
        fatal_error.store(true, std::memory_order_release);
    }

    uint64_t max_seq_waits      = hrr::kDefaultMaxSeqWaits;
    // Calls in progress, ordered or not. A wait for the next turn does not
    // count against max_seq_waits while this is non-zero.
    std::atomic<int> calls_in_flight{0};
    uint64_t max_query_attempts = hrr::kDefaultMaxQueryAttempts;

    std::atomic<size_t> kernels_launched{0};
    std::atomic<size_t> graphs_launched{0};
    double              total_kernel_ms  = 0.0;  // guarded by map_mutex when ctx.timing
    double              total_graph_ms   = 0.0;  // guarded by map_mutex when ctx.timing
    std::atomic<size_t> d2h_pass{0};
    // Subset of d2h_pass that matched within HIP_HRR_D2H_TOL but not byte-exactly.
    std::atomic<size_t> d2h_pass_tol{0};
    std::atomic<size_t> d2h_fail{0};
    // Every D2H event with a captured blob hash: pass + fail + skipped (missing pointer or blob).
    std::atomic<size_t> d2h_attempted{0};
    // Pinned host snapshot records written back, refused, and not applied
    // because the launch was replayed into a graph capture.
    std::atomic<size_t> host_snapshots_restored{0};
    std::atomic<size_t> host_snapshots_refused{0};
    std::atomic<size_t> host_snapshots_skipped{0};

    // Set by note_d2h_fail when the D2H-failure fraction crosses the divergence-abort threshold.
    std::atomic<bool> diverged{false};

    // Records one D2H validation failure. Sets `diverged` when the failure fraction
    // exceeds HIP_HRR_REPLAY_DIVERGENCE_ABORT.
    void note_d2h_fail(uint64_t seq);

    // One timing event pair per replay thread, created on its first launch and reused.
    // Registered here so teardown destroys them.
    std::vector<hipEvent_t> owned_timing_events;

    // Registered hipHostRegister replay buffers: recorded ptr -> block in
    // host_reg_blocks. Needed so hipHostUnregister can find the buffer.
    std::unordered_map<uint64_t, void*> host_reg_bufs;

    // APIs replay refused to reproduce (UNREPLAYABLE_PLAYBACK_APIS), and APIs
    // whose recorded call had already failed at capture.
    std::map<std::string, std::string> unreplayable_apis;  // api -> reason
    std::map<std::string, int>         reproduced_errors;  // api -> recorded hipError_t

    // VMM replay maps (guarded by map_mutex):
    std::unordered_map<uint64_t, hipMemGenericAllocationHandle_t> vmm_handle_map;
    struct VmmLive { uint64_t host_charge; int maps; bool released; };
    std::unordered_map<hipMemGenericAllocationHandle_t, VmmLive> vmm_live_map;
    struct VmmMapping { size_t size; hipMemGenericAllocationHandle_t live; };
    std::map<uintptr_t, VmmMapping> vmm_mappings;
    struct VmmVA { void* live; size_t size; };
    std::unordered_map<uint64_t, VmmVA> vmm_va_map;

    void* translate_vmm_va(uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = vmm_va_map.find(rec);
        return it != vmm_va_map.end() ? it->second.live : nullptr;
    }
    hipMemGenericAllocationHandle_t translate_vmm_handle(uint64_t rec) const {
        if (rec == 0) return {};
        std::shared_lock lk(map_mutex);
        auto it = vmm_handle_map.find(rec);
        return it != vmm_handle_map.end() ? it->second : hipMemGenericAllocationHandle_t{};
    }

    // The charge of a live handle goes back once it is released and unmapped.
    // Caller holds map_mutex exclusively.
    void vmm_drop_if_free_locked(hipMemGenericAllocationHandle_t live) {
        auto it = vmm_live_map.find(live);
        if (it == vmm_live_map.end() || !it->second.released || it->second.maps > 0) return;
        host_budget.release(it->second.host_charge);
        vmm_live_map.erase(it);
    }
    void vmm_note_create(uint64_t rec, hipMemGenericAllocationHandle_t live, uint64_t charge) {
        std::unique_lock lk(map_mutex);
        vmm_handle_map[rec] = live;
        vmm_live_map[live] = {charge, 0, false};
    }
    // hipMemRelease succeeded on `live`: it is no longer reachable by name, and
    // its charge stays while a mapping holds it.
    void vmm_note_release(hipMemGenericAllocationHandle_t live) {
        std::unique_lock lk(map_mutex);
        auto it = vmm_live_map.find(live);
        if (it == vmm_live_map.end()) return;
        it->second.released = true;
        vmm_drop_if_free_locked(live);
    }
    void vmm_note_map(void* live_va, size_t size, hipMemGenericAllocationHandle_t live) {
        std::unique_lock lk(map_mutex);
        auto it = vmm_live_map.find(live);
        if (it == vmm_live_map.end()) return;
        ++it->second.maps;
        vmm_mappings[reinterpret_cast<uintptr_t>(live_va)] = {size, live};
    }
    // hipMemUnmap succeeded on [live_va, live_va + size): every mapping wholly
    // inside gives its reference back.
    void vmm_note_unmap(void* live_va, size_t size) {
        std::unique_lock lk(map_mutex);
        const uintptr_t lo = reinterpret_cast<uintptr_t>(live_va);
        const uintptr_t hi = lo + size;
        for (auto it = vmm_mappings.lower_bound(lo); it != vmm_mappings.end() && it->first < hi;) {
            if (it->first + it->second.size > hi) { ++it; continue; }
            const auto live = it->second.live;
            it = vmm_mappings.erase(it);
            auto lit = vmm_live_map.find(live);
            if (lit != vmm_live_map.end()) --lit->second.maps;
            vmm_drop_if_free_locked(live);
        }
    }

    // What a producer outside libamdhip64 knew that the HIP dispatch table could not:
    // the block layout inside a framework allocator's segments.
    hrr::RegionMap regions;
    // Set when a sidecar was loaded. Separate from regions.empty() so --no-regions
    // and --multi-thread can switch the feature off.
    bool regions_enabled = false;
    // Count out-of-bounds findings toward the exit code. Off by default:
    // the finding describes the recorded program, not a replay failure.
    bool regions_strict = false;
    std::atomic<uint64_t> region_ptrs_checked{0};
    std::atomic<uint64_t> region_oob_ptrs{0};

    // Kernel-argument pointers that resolved in no allocation, VMM reservation or region.
    // A non-zero count means the capture lost allocations.
    bool warn_untranslated_args = false;
    std::atomic<uint64_t> untranslated_ptr_args{0};

    // Events not replayed as recorded because a pointer argument was unmapped.
    // strict_unmapped aborts the replay instead.
    bool strict_unmapped = false;
    std::atomic<size_t> unmapped_events{0};

    // Both off by default: they move memory the replay otherwise places exactly as recorded,
    // to catch out-of-bounds accesses.
    bool   guard_segments = false;  // VMM-back every allocation, guard its tail
    bool   guard_blocks   = false;  // relocate annotated blocks behind a guard
    size_t guard_min_bytes = 0;     // skip blocks smaller than this
    size_t guard_max_bytes = 0;     // 0 = no upper bound
    size_t guard_budget_bytes = (4ull << 30);  // per launch
    // Reproduce the recorded pointer's offset within a granule exactly, rather
    // than only its alignment. Tighter fidelity, looser guard.
    bool   guard_exact_align = false;
    std::atomic<uint64_t> guard_blocks_relocated{0};
    std::atomic<uint64_t> guard_blind_max{0};  // largest unguarded tail seen

    // VMM-backed allocations made for --guard-segments or placement, keyed by the pointer
    // handed out. hipFree cannot release them, so the free path unmaps them.
    struct GuardAlloc {
        void*  va_base  = nullptr;  // reserved VA base (== mapped base)
        size_t reserved = 0;        // total reserved VA (mapped + guard)
        size_t mapped   = 0;        // mapped/backed bytes
        hipMemGenericAllocationHandle_t handle{};
        bool   placed   = false;    // a placed allocation, not a guarded one
        bool   async    = false;    // placed for hipMallocAsync
    };
    std::unordered_map<void*, GuardAlloc> guard_allocs;

    // Capture-address placement. Placed allocations live in guard_allocs.
    bool place = false;           // set by hrr_place_hold
    std::mutex place_mutex;       // guards place_held
    HrrHeldRanges place_held;
    std::atomic<uint64_t> placed{0}, place_fell_back{0};

    // Graphs replay could not build faithfully: the recording added a node
    // that no handler here can reconstruct.
    std::set<uint64_t> incomplete_graphs;

    void mark_graph_incomplete(uint64_t rec_graph, const char* api) {
        if (!rec_graph) return;
        bool first;
        {
            std::unique_lock lk(map_mutex);
            first = incomplete_graphs.insert(rec_graph).second;
        }
        if (first)
            fprintf(stderr,
                    "[HRR] %s: this node was not reconstructed, so graph "
                    "0x%llx is incomplete; instantiating it will be refused "
                    "rather than replayed as a graph missing work.\n",
                    api, static_cast<unsigned long long>(rec_graph));
    }
    bool graph_is_incomplete(uint64_t rec_graph) const {
        std::shared_lock lk(map_mutex);
        return incomplete_graphs.count(rec_graph) != 0;
    }

    // IPC handles: recorded 64 bytes -> the 64 bytes this replay's exporter produced.
    // Filled only from legacy archives.
    std::map<std::string, std::string> ipc_handle_map;

    static std::string ipc_key(const void* bytes, size_t n) {
        return std::string(static_cast<const char*>(bytes), n);
    }
    void record_ipc_handle(const std::string& recorded, const std::string& live) {
        std::unique_lock lk(map_mutex);
        ipc_handle_map[recorded] = live;
    }
    // Empty when the archive holds no export for this handle: the importer is in
    // another process, out of scope for a single archive.
    std::string translate_ipc_handle(const std::string& recorded) const {
        std::shared_lock lk(map_mutex);
        auto it = ipc_handle_map.find(recorded);
        return it != ipc_handle_map.end() ? it->second : std::string();
    }

    // Translates a recorded GPU address to a live pointer and notes a miss
    // for the unmapped-pointer policy.
    void* translate_ptr(uint64_t rec) const {
        void* live = lookup_ptr(rec);
        if (!live && rec != 0) note_unmapped(rec);
        return live;
    }
    static void note_unmapped(uint64_t rec) {
        if (hrr_unmapped_note.count++ == 0) hrr_unmapped_note.first = rec;
    }

    // The same lookup without the note, for callers to whom a miss is an answer:
    // argument scans and releases.
    void* lookup_ptr(uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = alloc_map.find(rec);
        if (it != alloc_map.end()) return it->second.live_ptr;
        // Range search for sub-allocations. Recorded sizes are padded, so ranges can overlap;
        // take the highest base to pick the tightest.
        {
            const AllocEntry* best = nullptr;
            for (auto& [base, entry] : alloc_map) {
                if (rec >= base && rec < base + entry.size &&
                    (!best || base > best->rec_base))
                    best = &entry;
            }
            if (best)
                return static_cast<char*>(best->live_ptr) +
                       static_cast<ptrdiff_t>(rec - best->rec_base);
        }
        // Fall back to VMM reserved-VA map (exact + tightest-enclosing range)
        auto vit = vmm_va_map.find(rec);
        if (vit != vmm_va_map.end()) return vit->second.live;
        {
            uint64_t best_base = 0; const VmmVA* best = nullptr;
            for (auto& [base, va] : vmm_va_map) {
                if (rec >= base && rec < base + va.size &&
                    (!best || base > best_base)) {
                    best = &va; best_base = base;
                }
            }
            if (best)
                return static_cast<char*>(best->live) +
                       static_cast<ptrdiff_t>(rec - best_base);
        }
        return nullptr;
    }

    bool in_pinned_host(uint64_t v) const {
        std::shared_lock lk(map_mutex);
        for (auto& [base, e] : alloc_map)
            if ((e.kind == AllocKind::HostMalloc || e.kind == AllocKind::HostRegister) &&
                v - base < e.size)
                return true;
        return false;
    }

    // Whether v falls inside an allocation as recorded, before translation.
    bool is_recorded_va(uint64_t v, uint64_t* base_out = nullptr,
                        size_t* size_out = nullptr) const {
        if (v < 0x10000ULL) return false;
        std::shared_lock lk(map_mutex);
        for (auto& [base, entry] : alloc_map) {
            if (v >= base && v < base + entry.size) {
                if (base_out) *base_out = base;
                if (size_out) *size_out = entry.size;
                return true;
            }
        }
        for (auto& [base, va] : vmm_va_map) {
            if (v >= base && v < base + va.size) {
                if (base_out) *base_out = base;
                if (size_out) *size_out = va.size;
                return true;
            }
        }
        return false;
    }

    // Recorded allocation ranges sorted by base. Testing one word walks every allocation,
    // so scans that test many words use this.
    std::vector<std::pair<uint64_t, uint64_t>> recorded_ranges() const {
        std::vector<std::pair<uint64_t, uint64_t>> r;
        {
            std::shared_lock lk(map_mutex);
            r.reserve(alloc_map.size() + vmm_va_map.size());
            for (auto& [base, e] : alloc_map) r.emplace_back(base, base + e.size);
            for (auto& [base, va] : vmm_va_map) r.emplace_back(base, base + va.size);
        }
        std::sort(r.begin(), r.end());
        return r;
    }

    static bool range_contains(
        const std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t v,
        uint64_t* base_out = nullptr, size_t* size_out = nullptr) {
        if (ranges.empty() || v < ranges.front().first || v >= ranges.back().second)
            return false;
        auto it = std::upper_bound(
            ranges.begin(), ranges.end(), v,
            [](uint64_t x, const std::pair<uint64_t, uint64_t>& p) {
                return x < p.first;
            });
        if (it == ranges.begin()) return false;
        --it;
        if (v >= it->second) return false;
        if (base_out) *base_out = it->first;
        if (size_out) *size_out = static_cast<size_t>(it->second - it->first);
        return true;
    }

    // The live allocation containing a live pointer.
    // A kernel argument usually points into the middle of a buffer.
    bool live_alloc_of(const void* p, void** base_out, size_t* size_out,
                       uint64_t* rec_base_out,
                       AllocKind* kind_out = nullptr) const {
        uint64_t v = reinterpret_cast<uint64_t>(p);
        std::shared_lock lk(map_mutex);
        for (auto& [base, e] : alloc_map) {
            uint64_t lb = reinterpret_cast<uint64_t>(e.live_ptr);
            if (!lb || v < lb || v >= lb + e.size) continue;
            if (base_out) *base_out = e.live_ptr;
            if (size_out) *size_out = e.size;
            if (rec_base_out) *rec_base_out = base;
            if (kind_out) *kind_out = e.kind;
            return true;
        }
        return false;
    }

    static const char* alloc_kind_name(AllocKind k) {
        switch (k) {
            case AllocKind::HostMalloc:     return "pinned host";
            case AllocKind::HostRegister:   return "registered host";
            case AllocKind::DevicePtrAlias: return "pinned host alias";
            case AllocKind::Managed:        return "managed";
            default:                        return "device";
        }
    }

    // Describes an address as a capture-time address and as one of this process's
    // live allocations, for fault reports.
    std::string explain_addr(uint64_t v) const {
        char buf[320];
        std::string out;
        uint64_t rbase = 0; size_t rsize = 0;
        if (is_recorded_va(v, &rbase, &rsize)) {
            void* live = lookup_ptr(rbase);
            snprintf(buf, sizeof(buf),
                     "recorded allocation 0x%llx+%zu (live base %p)",
                     (unsigned long long)rbase, rsize, live);
            out += buf;
        }
        {
            std::shared_lock lk(map_mutex);
            for (auto& [base, entry] : alloc_map) {
                uint64_t lb = reinterpret_cast<uint64_t>(entry.live_ptr);
                if (!lb || v < lb || v >= lb + entry.size) continue;
                snprintf(buf, sizeof(buf),
                         "%slive allocation %p+%zu (recorded base 0x%llx)",
                         out.empty() ? "" : "; ", entry.live_ptr, entry.size,
                         (unsigned long long)base);
                out += buf;
                break;
            }
        }
        return out.empty() ? std::string("in no allocation this replay knows") : out;
    }

    size_t alloc_bytes_from(void* live_ptr) const {
        if (!live_ptr) return 0;
        uint64_t addr = reinterpret_cast<uint64_t>(live_ptr);
        std::shared_lock lk(map_mutex);
        for (auto& [base, entry] : alloc_map) {
            uint64_t live_base = reinterpret_cast<uint64_t>(entry.live_ptr);
            if (addr >= live_base && addr < live_base + entry.size)
                return entry.size - static_cast<size_t>(addr - live_base);
        }
        return 0;
    }

    hipStream_t   translate_stream  (uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = stream_map.find(rec); return it != stream_map.end() ? it->second : nullptr;
    }
    hipEvent_t    translate_event   (uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = event_map.find(rec); return it != event_map.end() ? it->second : nullptr;
    }
    hipModule_t   translate_module  (uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = module_map.find(rec); return it != module_map.end() ? it->second : nullptr;
    }
    hipFunction_t translate_func    (uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = func_map.find(rec); return it != func_map.end() ? it->second : nullptr;
    }
    hipMemPool_t  translate_mempool (uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = mempool_map.find(rec); return it != mempool_map.end() ? it->second : nullptr;
    }
    hipArray_t    translate_array   (uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = array_map.find(rec); return it != array_map.end() ? it->second : nullptr;
    }
    hipMipmappedArray_t translate_mipmapped(uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = mipmapped_map.find(rec); return it != mipmapped_map.end() ? it->second : nullptr;
    }
    hipGraph_t    translate_graph   (uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = graph_map.find(rec); return it != graph_map.end() ? it->second : nullptr;
    }
    hipGraphNode_t translate_graph_node(uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = graph_node_map.find(rec);
        return it != graph_node_map.end() ? it->second : nullptr;
    }
    hipGraphExec_t translate_graph_exec(uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = graph_exec_map.find(rec); return it != graph_exec_map.end() ? it->second : nullptr;
    }
    hipSurfaceObject_t translate_surface(uint64_t rec) const {
        std::shared_lock lk(map_mutex);
        auto it = surface_map.find(rec); return it != surface_map.end() ? it->second : 0;
    }
    hipTextureObject_t translate_texture(uint64_t rec) const {
        std::shared_lock lk(map_mutex);
        auto it = texture_map.find(rec); return it != texture_map.end() ? it->second : 0;
    }
    hipCtx_t translate_ctx(uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = ctx_map.find(rec); return it != ctx_map.end() ? it->second : nullptr;
    }
    hipLinkState_t translate_link_state(uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = link_state_map.find(rec);
        return it != link_state_map.end() ? it->second : nullptr;
    }

    void record_alloc(uint64_t rec, void* live, size_t sz,
                      AllocKind kind = AllocKind::Device) {
        std::unique_lock lk(map_mutex);
        // A host entry still under this address keeps its charge:
        // its buffer is still live.
        alloc_map[rec] = {rec, live, sz, kind};
    }
    void remove_alloc(uint64_t rec) {
        std::unique_lock lk(map_mutex);
        auto it = alloc_map.find(rec);
        if (it == alloc_map.end()) return;
        erase_alloc_locked(it);
    }
    // Removes the entry only while it still maps to `live`. An allocation recorded at
    // the same address earlier in the stream may have replaced it.
    bool remove_alloc_if_live(uint64_t rec, void* live) {
        std::unique_lock lk(map_mutex);
        auto it = alloc_map.find(rec);
        if (it == alloc_map.end() || it->second.live_ptr != live) return false;
        erase_alloc_locked(it);
        return true;
    }
    // Caller holds map_mutex exclusively.
    void erase_alloc_locked(std::unordered_map<uint64_t, AllocEntry>::iterator it) {
        // Host allocations (and managed memory, which the runtime may back
        // from host memory) were charged to host_budget before they were made.
        if (it->second.kind == AllocKind::HostMalloc ||
            it->second.kind == AllocKind::Managed)
            host_budget.release(it->second.size);
        alloc_map.erase(it);
    }
    bool has_alloc(uint64_t rec) const {
        std::shared_lock lk(map_mutex);
        return alloc_map.find(rec) != alloc_map.end();
    }

    // A graph node keeps the host pointer it was given, so a block is never
    // moved or freed before teardown.
    struct HostBlockFree {
        void operator()(uint8_t* p) const { ::operator delete[](p, std::align_val_t{64}); }
    };
    using HostBlockPtr = std::unique_ptr<uint8_t[], HostBlockFree>;
    struct HostBlock {
        HostBlockPtr cur;   // the block new nodes are given
        size_t size = 0;
        // Blocks an earlier node may still point into, outgrown but not freed.
        std::vector<HostBlockPtr> retired;
    };
    std::unordered_map<uint64_t, HostBlock> host_landing_buffers;
    // hipHostRegister backing buffers, kept the same way: a graph node may
    // still point into one after hipHostUnregister.
    std::unordered_map<uint64_t, HostBlock> host_reg_blocks;

    // The block for a recorded host address, at least sz bytes, zero-filled when new.
    // Null when the allocation fails, which also ends the replay.
    void* host_block(std::unordered_map<uint64_t, HostBlock>& blocks, uint64_t rec,
                     size_t sz, const char* what) {
        std::unique_lock lk(map_mutex);
        auto& buf = blocks[rec];
        if (!buf.cur || buf.size < sz) {
            // Charged whole, since the outgrown block stays resident.
            if (!charge_host(sz, what)) return nullptr;
            HostBlockPtr next(new (std::align_val_t{64}, std::nothrow) uint8_t[sz ? sz : 1]());
            if (!next) {
                host_budget.release(sz);
                fprintf(stderr, "[HRR] no %zu-byte %s for recorded host address 0x%llx\n", sz,
                        what, static_cast<unsigned long long>(rec));
                return nullptr;
            }
            if (buf.cur) buf.retired.push_back(std::move(buf.cur));
            buf.cur = std::move(next);
            buf.size = sz;
        }
        return buf.cur.get();
    }

    // A recorded key of 0 is a null host pointer from a call that failed at capture.
    // It gets no buffer.
    void* host_landing_buffer(uint64_t rec, size_t sz) {
        if (!rec) return nullptr;
        return host_block(host_landing_buffers, rec, sz, "graph host operand buffer");
    }

    void record_symbol(uint64_t rec_host_var, const std::string& name,
                       void* live, size_t sz) {
        std::unique_lock lk(map_mutex);
        symbol_map[rec_host_var] = {name, live, sz};
    }
    // Device address of the global the capturing process knew by this host
    // shadow address, or nullptr if the archive never registered it.
    void* translate_symbol(uint64_t rec_host_var, size_t* sz_out = nullptr) const {
        std::shared_lock lk(map_mutex);
        auto it = symbol_map.find(rec_host_var);
        if (it == symbol_map.end()) return nullptr;
        if (sz_out) *sz_out = it->second.size;
        return it->second.live_ptr;
    }
    // Resolve a symbol by name across every module the replay has loaded,
    // whether it came from a fat binary or an explicit hipModuleLoad.
    void* resolve_symbol_by_name(const char* name, size_t* sz_out) const;

    void record_stream  (uint64_t rec, hipStream_t     live) { std::unique_lock lk(map_mutex); stream_map[rec]     = live; }
    void record_event   (uint64_t rec, hipEvent_t      live) { std::unique_lock lk(map_mutex); event_map[rec]      = live; }
    void record_module  (uint64_t rec, hipModule_t     live) { std::unique_lock lk(map_mutex); module_map[rec]     = live; }
    void record_func    (uint64_t rec, hipFunction_t   live) { std::unique_lock lk(map_mutex); func_map[rec]       = live; }
    void record_mempool (uint64_t rec, hipMemPool_t    live) { std::unique_lock lk(map_mutex); mempool_map[rec]    = live; }
    void record_array   (uint64_t rec, hipArray_t      live) { std::unique_lock lk(map_mutex); array_map[rec]      = live; }
    void record_mipmapped(uint64_t rec, hipMipmappedArray_t live) { std::unique_lock lk(map_mutex); mipmapped_map[rec] = live; }
    void record_graph   (uint64_t rec, hipGraph_t      live) { std::unique_lock lk(map_mutex); graph_map[rec]      = live; }
    void record_graph_node(uint64_t rec, hipGraphNode_t live){ std::unique_lock lk(map_mutex); graph_node_map[rec] = live; }
    void record_graph_exec(uint64_t rec, hipGraphExec_t live){ std::unique_lock lk(map_mutex); graph_exec_map[rec] = live; }
    void record_surface (uint64_t rec, hipSurfaceObject_t live) { std::unique_lock lk(map_mutex); surface_map[rec] = live; }
    void record_texture (uint64_t rec, hipTextureObject_t live) { std::unique_lock lk(map_mutex); texture_map[rec] = live; }
    void record_ctx     (uint64_t rec, hipCtx_t        live) { std::unique_lock lk(map_mutex); ctx_map[rec]        = live; }
    void record_link_state(uint64_t rec, hipLinkState_t live){ std::unique_lock lk(map_mutex); link_state_map[rec] = live; }

    void remove_stream  (uint64_t rec) { std::unique_lock lk(map_mutex); stream_map.erase(rec); }
    void remove_event   (uint64_t rec) { std::unique_lock lk(map_mutex); event_map.erase(rec); }
    void remove_module  (uint64_t rec) { std::unique_lock lk(map_mutex); module_map.erase(rec); }
    void remove_func    (uint64_t rec) { std::unique_lock lk(map_mutex); func_map.erase(rec); }
    void remove_mempool (uint64_t rec) { std::unique_lock lk(map_mutex); mempool_map.erase(rec); }
    void remove_array   (uint64_t rec) { std::unique_lock lk(map_mutex); array_map.erase(rec); }
    void remove_mipmapped(uint64_t rec){ std::unique_lock lk(map_mutex); mipmapped_map.erase(rec); }
    void remove_graph   (uint64_t rec) { std::unique_lock lk(map_mutex); graph_map.erase(rec); }
    void remove_graph_node(uint64_t rec){ std::unique_lock lk(map_mutex); graph_node_map.erase(rec); }
    void remove_graph_exec(uint64_t rec){ std::unique_lock lk(map_mutex); graph_exec_map.erase(rec); }
    void remove_surface (uint64_t rec) { std::unique_lock lk(map_mutex); surface_map.erase(rec); }
    void remove_texture (uint64_t rec) { std::unique_lock lk(map_mutex); texture_map.erase(rec); }
    void remove_ctx     (uint64_t rec) { std::unique_lock lk(map_mutex); ctx_map.erase(rec); }
    void remove_link_state(uint64_t rec){ std::unique_lock lk(map_mutex); link_state_map.erase(rec); }

    // Load a blob from archive_dir/blobs/<2-char-prefix>/<hash>.blob
    // Returns nullptr if not found. Memory is owned by the context (cached).
    const void* load_blob(uint64_t hash_lo, uint64_t hash_hi,
                          size_t* sz_out = nullptr) const;
    const void* load_code_object(uint64_t hash_lo, uint64_t hash_hi,
                                 size_t* sz_out) const;
    // Read `path` into blob_cache_ under `key`, charged to the host budget.
    const void* cache_file(const std::string& key, const std::string& path,
                           size_t* sz_out, const char* what) const;
    hipModule_t load_module(uint64_t hash_lo, uint64_t hash_hi);

    // Parameter sizes the code object declares for `func`, in order. Empty means ask HIP
    // (hipKernelGetParamInfo); a handler test can override it.
    std::function<void(hipFunction_t, std::vector<size_t>&)> kernel_param_sizes;

    hipFunction_t resolve_replacement(const std::string& kernel_name);

private:
    mutable std::unordered_map<std::string, std::vector<uint8_t>> blob_cache_;
};

// Region materialisation — back an annotated segment HRR never observed.
hipError_t hrr_materialize_region(PlaybackContext& ctx, uint64_t rec_base,
                                  size_t size, int device, void** out_live);

// Release a buffer created by hrr_materialize_region and drop its alloc_map
// entry. `live` is the pointer that call returned.
void hrr_release_region(PlaybackContext& ctx, uint64_t rec_base, void* live);

void hrr_free_device_alloc(PlaybackContext& ctx, void* live);

// Capture-address placement. hrr_place_hold runs before hipInit and holds the planned
// ranges with PROT_NONE placeholders (a no-op on Windows).
void hrr_place_hold(PlaybackContext& ctx, const std::vector<HrrVaRange>& plan);
void hrr_place_reset(PlaybackContext& ctx);

// Calls `once(arg)` until it stops returning hipErrorNotReady,
// at most ctx.max_query_attempts times.
hipError_t replay_query_until_success(PlaybackContext& ctx, const char* api,
                                      hipError_t (*once)(void*), void* arg);

void hrr_note_unreplayable(PlaybackContext& ctx, const char* api,
                           const char* reason);

void hrr_note_recorded_error(PlaybackContext& ctx, const char* api,
                             int recorded_ret);

// hrr_runtime_symbol — address of an exported function in the HIP runtime this
// process loaded, or nullptr when it does not export it.
void* hrr_runtime_symbol(const char* name);

bool hrr_replayed_recorded_error(PlaybackContext& ctx, const char* api,
                                 int32_t recorded_ret, hipError_t replayed);

// The kernel-argument decoder that a launch and a graph kernel node share.
void hrr_decode_kernel_args(PlaybackContext& ctx, const uint8_t* p,
                            const uint8_t* end, uint16_t num_args,
                            std::vector<void*>& arg_ptrs,
                            std::vector<std::vector<uint8_t>>& arg_storage);

hipCtx_t hrr_live_ctx(uint64_t recorded);

// Replaces the srcArray and dstArray handles of a recorded hipMemcpy3DParms
// with the live arrays replay created for them.
bool hrr_translate_memcpy3d_arrays(const PlaybackContext& ctx, const char* api,
                                   hipMemcpy3DParms& parms);

// Thread-local sequence ID — set by dispatch_event before calling any handler.
extern thread_local uint64_t hrr_dispatch_seq;

// Kernel-launch handoff for dispatch_event. A launch gives up the global turn before
// its handler runs, so it holds host_free_mu shared across the handoff.
void hrr_launch_hold(PlaybackContext& ctx);
void hrr_launch_release();
bool hrr_is_launch_handler(hipError_t (*fn)(PlaybackContext&, const uint8_t*));

// Hands the global turn on for one event. An unordered event hands it on at
// construction; an ordered one, at destruction.
class HrrTurnGuard {
 public:
  HrrTurnGuard(PlaybackContext& ctx, uint64_t next, bool ordered, bool launch)
      : ctx_(ctx), next_(next), ordered_(ordered), launch_(launch) {
    if (!ordered_) {
      if (launch_) hrr_launch_hold(ctx_);
      ctx_.next_seq.store(next_, std::memory_order_release);
    }
  }
  ~HrrTurnGuard() {
    if (ordered_) ctx_.next_seq.store(next_, std::memory_order_release);
    if (launch_) hrr_launch_release();
  }
  HrrTurnGuard(const HrrTurnGuard&) = delete;
  HrrTurnGuard& operator=(const HrrTurnGuard&) = delete;

 private:
  PlaybackContext& ctx_;
  uint64_t next_;
  bool ordered_;
  bool launch_;
};

// Device synchronize with an optional watchdog.
// With ctx.sync_watchdog_ms == 0 it is a plain hipDeviceSynchronize().
hipError_t hrr_watchdog_device_sync(PlaybackContext& ctx, const char* what);

// One pinned host snapshot record of a kernel launch event: u64 ptr_handle,
// u64 offset, u64 length, u64 hash_lo, u64 hash_hi, u8 direction (0).
inline constexpr size_t kHostSnapRecordSize = 41;

inline bool hrr_host_snapshots_fit(const uint8_t* p, const uint8_t* end, uint16_t n) {
    return p <= end && static_cast<size_t>(end - p) / kHostSnapRecordSize >= n;
}

// Where replay writes the blob of the record at `rec`, or nullptr with *why saying why
// it refuses the record. The archive is untrusted input.
inline void* hrr_host_snapshot_target(const PlaybackContext& ctx, const uint8_t* rec,
                                      const void* blob, size_t blob_len,
                                      const char** why) {
    uint64_t ptr, off, len;
    memcpy(&ptr, rec, 8);
    memcpy(&off, rec + 8, 8);
    memcpy(&len, rec + 16, 8);
    void*  live = nullptr;
    size_t size = 0;
    {
        std::shared_lock lk(ctx.map_mutex);
        auto it = ctx.alloc_map.find(ptr);
        if (it != ctx.alloc_map.end() && (it->second.kind == AllocKind::HostMalloc ||
                                          it->second.kind == AllocKind::HostRegister)) {
            live = it->second.live_ptr;
            size = it->second.size;
        }
    }
    if (rec[40] != 0)
        *why = "has an unknown direction";
    else if (!live)
        *why = "names no pinned host allocation replay made";
    else if (off > size || len > size - off)
        *why = "lies outside its allocation";
    else if (!blob || blob_len != len)
        *why = "does not match its blob";
    else
        return static_cast<uint8_t*>(live) + off;
    return nullptr;
}


typedef hipError_t (*hrr_playback_fn_t)(PlaybackContext& ctx, const uint8_t* payload);

// Indexed by hrr_api_id_t — defined in hip_playback_generated.cpp
extern hrr_playback_fn_t hrr_playback_dispatch[HRR_API_COUNT];
extern const uint32_t    hrr_api_min_payload_size[HRR_API_COUNT];
