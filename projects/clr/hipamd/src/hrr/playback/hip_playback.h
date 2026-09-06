/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
/* hip_playback.h — PlaybackContext and dispatch table for HRR playback. */
#pragma once

#include <hip/hip_runtime.h>
#include <map>
#include <set>
#include <unordered_map>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <chrono>

#include "hrr_api_args.h"     // for HRR_API_COUNT, hrr_api_id_t
#include "hrr_region_map.h"   // external region annotations (regions/*.hrrr)

// Whether a replayed H2D blob restore must be drained before subsequent
// replay work. Draining is skipped while a stream graph capture is active,
// where a device/stream synchronize is illegal (HIP 900/901). Kept as a
// small pure predicate so the graph-capture guard is unit-testable without a
// GPU. See hrr_sync_after_replayed_h2d() in hip_playback.cpp.
inline bool hrr_replayed_h2d_needs_drain(bool in_graph_capture) {
    return !in_graph_capture;
}

// Whether the zero-init injected after a host-synchronous replay allocation
// (hipMalloc / hipMallocManaged / hipExtMallocWithFlags) must be drained before
// the allocation handler returns. hipMemset on a fresh, non-offset device
// allocation is promoted to asynchronous by ihipMemset(), so it is only
// enqueued on the null stream; a recorded hipStreamNonBlocking stream never
// synchronizes with the null stream, leaving the zero-init unordered against
// every later replayed event. Skipped while a stream graph capture is active,
// where the zero-init is not issued at all. Kept as a small pure predicate so
// the guard is unit-testable without a GPU. See hrr_zero_init_alloc() in
// hip_playback.cpp.
inline bool hrr_zero_init_needs_drain(bool zero_init_enabled,
                                      bool in_graph_capture) {
    return zero_init_enabled && !in_graph_capture;
}

// ---------------------------------------------------------------------------
// PlaybackContext — central replay state
// ---------------------------------------------------------------------------

// How an alloc_map entry's live_ptr was obtained — determines which API must
// release it at teardown. Mixing them up (e.g. hipFree on a host pointer)
// returns errors and can corrupt allocator bookkeeping.
enum class AllocKind : uint8_t {
    Device,        // hipMalloc / hipMallocManaged / hipMallocPitch -> hipFree
    HostMalloc,    // hipHostMalloc / hipMallocHost                 -> hipHostFree
    HostRegister,  // hipHostRegister backing buffer  -> hipHostUnregister + free
                   //   (released via host_reg_bufs; skipped in the alloc_map loop)
    DevicePtrAlias // hipHostGetDevicePointer result  -> not separately freed
                   //   (alias into an already-tracked pinned host allocation)
};

struct AllocEntry {
    uint64_t  rec_base;  // recorded GPU base address
    void*     live_ptr;  // live replay GPU base address
    size_t    size;
    AllocKind kind = AllocKind::Device;
};

struct PlaybackContext {
    std::string archive_dir;

    // ---- Handle translation maps (recorded raw ptr -> live handle) ----
    // Protected by map_mutex: shared_lock for reads, unique_lock for writes.
    mutable std::shared_mutex map_mutex;
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
    // Driver-API contexts. Without this map the family replayed asymmetrically
    // — hipCtxCreate and hipCtxPopCurrent no-ops, hipCtxPushCurrent real — so
    // replay pushed a capture-time hipCtx_t and the thread was left with no
    // current device, which killed the process several events later.
    std::unordered_map<uint64_t, hipCtx_t>       ctx_map;
    // JIT linker states. hipLinkCreate used to throw its output away, so
    // hipLinkComplete and hipLinkDestroy were handed the capturing process's
    // pointer and failed no matter what the linker inputs were.
    std::unordered_map<uint64_t, hipLinkState_t> link_state_map;

    // Device allocations: recorded base address -> {live ptr, size}
    std::unordered_map<uint64_t, AllocEntry>     alloc_map;

    // __device__ globals, keyed by the recorded host shadow address the
    // capturing process passed to hipGetSymbolAddress and the hipMemcpy*Symbol
    // family. A replay has no host shadow of its own — the code object is
    // loaded through hipModuleLoadData — so the name recorded beside the
    // address is what resolves it.
    struct SymbolEntry {
        std::string name;
        void*       live_ptr = nullptr;
        size_t      size     = 0;
    };
    std::unordered_map<uint64_t, SymbolEntry>    symbol_map;

    // Code-object modules loaded by hash (not by recorded module handle)
    std::unordered_map<std::string, hipModule_t> co_modules;

    // Kernel function cache: mangled name -> resolved hipFunction_t.
    // Populated on first launch of each kernel; avoids repeated hipModuleGetFunction
    // searches across module_map / co_modules on every launch.
    std::unordered_map<std::string, hipFunction_t> func_cache;

    // Options propagated from hrr_replay / hrr_bench / hrr_fullreplay
    bool timing            = false;
    bool skip_device_sync  = false;
    bool sync_after_launch = false;  // hipDeviceSynchronize after every kernel launch
    bool sync_after_event  = false;  // hipDeviceSynchronize after EVERY dispatched event
    // Keep replaying after a handler returns an error instead of stopping at
    // the first one. Off by default: for a debugging replay the first error is
    // the answer, and continuing past it only produces cascading noise. Turn
    // it on to survey a whole archive — which API fails, and what the ones
    // after it do — rather than to reproduce a fault.
    bool continue_on_error = false;
    std::atomic<size_t> events_failed{0};
    std::atomic<size_t> code_objects_failed{0};
    // Sync watchdog: max wall-clock ms to wait for a device synchronize before
    // declaring the GPU wedged (0 = disabled / wait forever). Surfaces hung
    // kernels (e.g. a StreamK producer/consumer flag spin-wait) as a diagnostic
    // + hard exit instead of an indefinite hang. See hrr_watchdog_device_sync.
    unsigned sync_watchdog_ms = 0;
    // When non-zero, dump each pointer argument's recorded->translated(live)
    // value for the kernel launch with this 1-based ordinal. Used to diff the
    // captured vs replay pointer contract (e.g. a StreamK synchronizer base).
    size_t dump_ptrs_ordinal = 0;
    bool verbose           = false;
    bool validate_d2h      = false;  // perform D2H validation against captured expected data
    std::string kernel_filter;

    // Lightweight replay tracing. These are intentionally separate from
    // verbose mode, which dumps every event and every kernel argument.
    bool trace_kernels         = false;  // one compact line before every kernel launch
    bool trace_sync            = false;  // mark sync begin/done around each launched kernel
    size_t progress_kernel_interval = 0; // print progress every N launched kernels
    double progress_seconds_interval = 0.0; // also print progress at most every N seconds
    std::chrono::steady_clock::time_point progress_start_time{};
    std::chrono::steady_clock::time_point progress_last_time{};
    std::mutex progress_mutex;

    // ---- Kernel replacement (playback-time override) ----
    // Parsed "NAME=path" pairs from --replace-kernel. The recorded kernel whose
    // name is exactly NAME launches from the replacement code object at `path`
    // instead of the recorded one, while grid/block/shared/args/pointers still
    // come from the recording. NAME must be the full recorded symbol. The archive
    // is never modified. Empty => feature disabled (no overhead on the hot path).
    std::vector<std::pair<std::string, std::string>> kernel_replacements;
    // Resolved replacement functions, keyed by the recorded kernel name.
    // Guarded by map_mutex. Modules backing them are owned in replacement_modules.
    std::unordered_map<std::string, hipFunction_t> replacement_funcs;
    std::vector<hipModule_t> replacement_modules;  // unloaded at teardown

    // Set true between hipStreamBeginCapture and hipStreamEndCapture.
    // HIP event timing must be skipped during graph capture: recording an
    // event on a captured stream inserts it into the graph and invalidates
    // the capture state, causing error 901 on all subsequent operations.
    bool in_graph_capture  = false;

    // Global submission order for MT replay.
    // Each thread spin-waits until next_seq reaches its event's sequence_id,
    // dispatches the event, then advances next_seq. This prevents any thread
    // from getting ahead of the global capture order without forcing strict
    // 1-at-a-time serialisation — a thread only waits when it IS ahead.
    std::atomic<uint64_t> next_seq{0};

    // Set to true by dispatch_event on the first HIP error. All replay threads
    // check this at the top of their event loop and exit immediately.
    // The spin-wait in dispatch_event also checks it to avoid deadlock when a
    // thread that was supposed to advance next_seq has already aborted.
    std::atomic<bool> fatal_error{false};

    // Stats — atomic for safe concurrent increment from replay threads.
    // total_kernel_ms is guarded by map_mutex (unique_lock) in the timing path.
    std::atomic<size_t> kernels_launched{0};
    std::atomic<size_t> graphs_launched{0};
    double              total_kernel_ms  = 0.0;  // guarded by map_mutex when ctx.timing
    double              total_graph_ms   = 0.0;  // guarded by map_mutex when ctx.timing
    std::atomic<size_t> d2h_pass{0};
    // Subset of d2h_pass that were NOT byte-exact but matched within numeric
    // tolerance (benign floating-point nondeterminism from non-associative GPU
    // reductions). Tracked separately so the summary can distinguish exact
    // replay fidelity from "numerically equivalent".
    std::atomic<size_t> d2h_pass_tol{0};
    std::atomic<size_t> d2h_fail{0};
    // Incremented for every D2H event that had a captured blob hash (i.e., validation
    // was expected). Includes pass + fail + skipped (missing ptr / missing blob).
    // If d2h_attempted > 0 but d2h_pass == 0 && d2h_fail == 0, every check was
    // skipped — pointer translation or blob loading failed for all D2H events.
    std::atomic<size_t> d2h_attempted{0};

    // Set true by note_d2h_fail when the running D2H-failure fraction crosses the
    // configured divergence-abort threshold. Distinguishes a clean "replay
    // diverged" stop (replay-fidelity limit) from a genuine HIP error abort.
    std::atomic<bool> diverged{false};

    // Records one D2H validation failure (replaces a bare d2h_fail++). When the
    // running failure fraction exceeds HIP_HRR_REPLAY_DIVERGENCE_ABORT (after a
    // minimum sample count), sets `diverged` + `fatal_error` so the replay stops
    // cleanly before a downstream GPU fault instead of dying unrecoverably.
    // `seq` is the recorded event sequence id (hrr_dispatch_seq) for diagnostics.
    void note_d2h_fail(uint64_t seq);

    // Timing events — one pair per replay thread, created on first kernel launch
    // and reused for every subsequent launch on that thread. Registered here so
    // cleanup can destroy them without per-kernel create/destroy overhead.
    // Appended under map_mutex (unique_lock); no lock needed to read (single writer).
    std::vector<hipEvent_t> owned_timing_events;

    // Backing buffers for hipHostRegister replay: recorded ptr -> malloc'd host buffer.
    // Needed so hipHostUnregister can call hipHostUnregister then free the buffer.
    // Guarded by map_mutex.
    std::unordered_map<uint64_t, void*> host_reg_bufs;

    // APIs this replay refused to reproduce (UNREPLAYABLE_PLAYBACK_APIS), and
    // APIs whose recorded call had already failed at capture and failed the
    // same way here. Both are printed in the summary: the first is what the
    // replay could not do, the second is what it faithfully reproduced but a
    // reader would otherwise misread as a replay error. Guarded by map_mutex.
    std::map<std::string, std::string> unreplayable_apis;  // api -> reason
    std::map<std::string, int>         reproduced_errors;  // api -> recorded hipError_t

    // VMM replay maps (guarded by map_mutex):
    //   vmm_handle_map: recorded hipMemGenericAllocationHandle_t (as u64) -> live handle
    //   vmm_va_map    : recorded reserved-VA base (u64) -> {live_va, size}
    std::unordered_map<uint64_t, hipMemGenericAllocationHandle_t> vmm_handle_map;
    struct VmmVA { void* live; size_t size; };
    std::unordered_map<uint64_t, VmmVA> vmm_va_map;

    // Translate a recorded VA (from AddressReserve) to the live replay VA.
    // Returns nullptr if not found or if rec is 0.
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

    // ---- External region annotations (regions/*.hrrr, see hrr_regions.h) ----
    // What a producer outside libamdhip64 knew and the HIP dispatch table could
    // not: the per-object block layout inside a framework allocator's segments,
    // and segments that never crossed a HIP API at all. Replayed in lockstep
    // with the event stream so the live set matches the captured instant.
    hrr::RegionMap regions;
    // Set when a sidecar was actually loaded. Kept separate from
    // regions.empty() so --no-regions and --multi-thread can switch the whole
    // feature off without discarding what was read.
    bool regions_enabled = false;
    // Count intra-segment out-of-bounds findings toward the exit code. Off by
    // default: the finding is a diagnostic about the recorded program, not a
    // failure of the replay to reproduce it.
    bool regions_strict = false;
    std::atomic<uint64_t> region_ptrs_checked{0};
    std::atomic<uint64_t> region_oob_ptrs{0};

    // Kernel-argument pointers that resolved in no map at all — neither an
    // allocation, a VMM reservation, nor an annotated region. A non-zero count
    // is the measurement that says a capture needs region annotations: those
    // pointers reach the GPU as null. See --warn-untranslated-args.
    bool warn_untranslated_args = false;
    std::atomic<uint64_t> untranslated_ptr_args{0};

    // ---- Guard pages ----
    // Both off by default: they trade the exact memory layout the replay
    // otherwise reproduces for the ability to make an out-of-bounds access
    // fault. See the guard section of hip_playback.cpp.
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

    // VMM-backed allocations created for --guard-segments, keyed by the mapped
    // base that was handed out. hipFree cannot release these, so the free path
    // has to recognise them. Guarded by map_mutex.
    struct GuardAlloc {
        void*  va_base  = nullptr;  // reserved VA base (== mapped base)
        size_t reserved = 0;        // total reserved VA (mapped + guard)
        size_t mapped   = 0;        // mapped/backed bytes
        hipMemGenericAllocationHandle_t handle{};
    };
    std::unordered_map<void*, GuardAlloc> guard_allocs;

    // Graphs this replay could not build faithfully: a node the recording
    // added that no handler here can reconstruct. Before the node API was
    // replayed at all, a graph's mere absence from graph_map was the signal
    // hipGraphInstantiate refused on. Now that most construction calls do
    // build something, absence no longer distinguishes "built by the node API"
    // from "built correctly", so the ones that fell short say so explicitly
    // and instantiate still refuses. Keyed by the recorded hipGraph_t.
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

    // IPC handles, recorded 64 bytes -> the 64 bytes this replay's exporter
    // produced. An importer's handle names an export in the process that made
    // it, so replaying hipIpcOpenMemHandle with the recorded bytes opens
    // nothing; it has to be matched to the handle the replayed exporter
    // returned. Keyed by the recorded bytes because that is all the importing
    // event carries. Guarded by map_mutex.
    std::map<std::string, std::string> ipc_handle_map;

    static std::string ipc_key(const void* bytes, size_t n) {
        return std::string(static_cast<const char*>(bytes), n);
    }
    void record_ipc_handle(const std::string& recorded, const std::string& live) {
        std::unique_lock lk(map_mutex);
        ipc_handle_map[recorded] = live;
    }
    // Empty when the archive holds no export for this handle: the importer is
    // then in another process, which is out of scope for a single-archive
    // replay.
    std::string translate_ipc_handle(const std::string& recorded) const {
        std::shared_lock lk(map_mutex);
        auto it = ipc_handle_map.find(recorded);
        return it != ipc_handle_map.end() ? it->second : std::string();
    }

    // ---- Pointer translation ----
    // Translates a recorded GPU address to a live pointer.
    // Checks alloc_map (exact + range) then vmm_va_map (exact + range).
    void* translate_ptr(uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = alloc_map.find(rec);
        if (it != alloc_map.end()) return it->second.live_ptr;
        // Range search for sub-allocations. Allocations are recorded with padded
        // sizes, so synthetic ranges can overlap; iteration order over an
        // unordered_map is unspecified. Pick the *tightest* enclosing entry
        // (largest base <= rec) so the result is deterministic and points at
        // the true home allocation rather than a neighbour's over-extended pad.
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

    // Returns bytes available from a live GPU pointer to end of its backing
    // alloc_map entry. Returns 0 if the pointer is not within any known alloc.
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

    // ---- Handle resolution (shared lock — concurrent reads safe) ----
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

    // ---- Allocation registration (exclusive lock) ----
    void record_alloc(uint64_t rec, void* live, size_t sz,
                      AllocKind kind = AllocKind::Device) {
        std::unique_lock lk(map_mutex);
        alloc_map[rec] = {rec, live, sz, kind};
    }
    void remove_alloc(uint64_t rec) {
        std::unique_lock lk(map_mutex);
        alloc_map.erase(rec);
    }
    // True if an allocation is already tracked under this recorded address.
    bool has_alloc(uint64_t rec) const {
        std::shared_lock lk(map_mutex);
        return alloc_map.find(rec) != alloc_map.end();
    }

    // A graph node's host operand has to stay alive until the graph is
    // launched, which is long after the call that built the node returned.
    // Keyed by the recorded host address so the same recorded destination
    // reuses one buffer instead of growing a new one per event.
    std::unordered_map<uint64_t, std::vector<uint8_t>> host_landing_buffers;

    void* host_landing_buffer(uint64_t rec, size_t sz) {
        std::unique_lock lk(map_mutex);
        auto& buf = host_landing_buffers[rec];
        if (buf.size() < sz) buf.resize(sz);
        return buf.data();
    }

    // ---- Symbol registration and lookup ----
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

    // ---- Handle registration (exclusive lock) ----
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

    // ---- Handle removal (exclusive lock) ----
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

    // ---- Blob/code-object loading ----
    // Load a blob from archive_dir/blobs/<2-char-prefix>/<hash>.blob
    // Returns nullptr if not found. Memory is owned by the context (cached).
    const void* load_blob(uint64_t hash_lo, uint64_t hash_hi,
                          size_t* sz_out = nullptr) const;
    // Load a code object from archive_dir/code_objects/<hash>.hsaco
    const void* load_code_object(uint64_t hash_lo, uint64_t hash_hi,
                                 size_t* sz_out) const;
    // Load a code object and cache the resulting hipModule_t
    hipModule_t load_module(uint64_t hash_lo, uint64_t hash_hi);

    // Resolve a playback-time replacement function for a recorded kernel name.
    // Returns a hipFunction_t loaded from a user-supplied .hsaco if `kernel_name`
    // matches a --replace-kernel pattern, or nullptr if no replacement applies
    // (or the replacement failed to load — caller then falls back to the recorded
    // kernel). Lazily loads + caches the replacement module on first match.
    hipFunction_t resolve_replacement(const std::string& kernel_name);

private:
    mutable std::unordered_map<std::string, std::vector<uint8_t>> blob_cache_;
};

// ---------------------------------------------------------------------------
// Region materialisation — back an annotated segment HRR never observed.
//
// Called by RegionMap when a producer declares a segment whose base resolves in
// no map, which means the allocation bypassed the HIP dispatch table (a direct
// HSA allocation, a foreign VMM pool, memory imported from another process).
// Allocates a live buffer of the recorded size on `device` — the ordinal the
// producer recorded the segment on — applies the replay fill byte (nothing in
// the archive can say what the contents were) and registers it in alloc_map so
// ordinary pointer translation resolves into it.
hipError_t hrr_materialize_region(PlaybackContext& ctx, uint64_t rec_base,
                                  size_t size, int device, void** out_live);

// Release a buffer created by hrr_materialize_region and drop its alloc_map
// entry. `live` is the pointer that call returned.
void hrr_release_region(PlaybackContext& ctx, uint64_t rec_base, void* live);

// Release a device allocation the replay created, whichever way it was created.
// Under --guard-segments an allocation is a VMM mapping rather than a hipMalloc
// and hipFree cannot release it, so every teardown path has to go through here.
void hrr_free_device_alloc(PlaybackContext& ctx, void* live);

// ---------------------------------------------------------------------------
// hrr_note_unreplayable — this API cannot be reproduced, and here is why.
//
// Called by the UNREPLAYABLE_PLAYBACK_APIS handlers before they return
// hipErrorNotSupported. Prints once per API:
//
//   [HRR] hipStreamAddCallback: NOT REPLAYABLE — the callback is a host
//         function pointer belonging to the capturing process. ...
//
// and records it so the replay summary can list every such API rather than
// leaving the reader to grep a long log for warnings.
void hrr_note_unreplayable(PlaybackContext& ctx, const char* api,
                           const char* reason);

// ---------------------------------------------------------------------------
// hrr_note_recorded_error — replay reproduced a call that also failed at
// capture, which is fidelity rather than failure.
//
// Most capture shims only record calls that succeeded, so a handler returning
// an error normally means replay diverged. The exceptions are the hand-written
// shims that record the call before the runtime rejects it (hipArrayCreate on a
// part with no image support, say). Returning that same error to dispatch_event
// would abort the replay over a call the recording shows failing identically.
void hrr_note_recorded_error(PlaybackContext& ctx, const char* api,
                             int recorded_ret);

// ---------------------------------------------------------------------------
// hrr_replayed_recorded_error — did this call fail exactly as the recording
// says it failed?
//
// The generated handlers get this from the generator; the hand-written ones
// call it. Both are needed because the hand-written capture shims are the ones
// that record a call the runtime rejected, so their handlers are the ones most
// likely to meet a recorded error.
bool hrr_replayed_recorded_error(PlaybackContext& ctx, const char* api,
                                 int32_t recorded_ret, hipError_t replayed);

// ---------------------------------------------------------------------------
// hrr_live_ctx — a usable hipCtx_t for a recorded one.
//
// A recorded hipCtx_t names a context object in the capturing process and
// there is no map from it to anything here. Passing it through would hand the
// runtime a stale address; passing nullptr fails outright, because the
// driver-API graph node calls reject a null context. Replay therefore
// substitutes the context the calling thread is already on, which on AMD is
// the primary context of the current device — the same device the archive was
// recorded against. A recording that used a null context keeps one, so a call
// that capture shows failing that way fails the same way here.
hipCtx_t hrr_live_ctx(uint64_t recorded);

// Thread-local sequence ID — set by dispatch_event before calling any handler.
// Kernel-launch handlers read this to wait for their submission turn at the
// exact point of the HIP call, allowing preparation work to run in parallel.
extern thread_local uint64_t hrr_dispatch_seq;

// Device synchronize with an optional watchdog. When ctx.sync_watchdog_ms == 0
// this is a plain hipDeviceSynchronize(). Otherwise the (potentially blocking)
// sync runs on a helper thread and is bounded by the timeout; on timeout it
// prints a hung-kernel diagnostic (using `what` as context) and hard-exits so a
// deadlocked kernel surfaces instead of hanging the replay forever. Normal
// completion — including a genuine GPU fault — is returned to the caller.
hipError_t hrr_watchdog_device_sync(PlaybackContext& ctx, const char* what);

// ---------------------------------------------------------------------------
// Playback function signature and dispatch table
// ---------------------------------------------------------------------------

// Each playback shim receives:
//   ctx      — replay state (mutable)
//   payload  — pointer to the full hrr_args_* struct (header + fields); cast directly:
//              const auto* a = reinterpret_cast<const hrr_args_foo*>(payload);
// Returns hipSuccess (0) on success, or a hipError_t on failure.
typedef hipError_t (*hrr_playback_fn_t)(PlaybackContext& ctx, const uint8_t* payload);

// Indexed by hrr_api_id_t — defined in hip_playback_generated.cpp
extern hrr_playback_fn_t hrr_playback_dispatch[HRR_API_COUNT];
extern const uint32_t    hrr_api_min_payload_size[HRR_API_COUNT];
