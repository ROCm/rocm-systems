/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
/* hip_playback.h — PlaybackContext and dispatch table for HRR playback. */
#pragma once

#include <hip/hip_runtime.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <atomic>
#include <mutex>
#include <shared_mutex>

#include "hrr_api_args.h"  // for HRR_API_COUNT, hrr_api_id_t

// ---------------------------------------------------------------------------
// PlaybackContext — central replay state
// ---------------------------------------------------------------------------

struct AllocEntry {
    uint64_t rec_base;  // recorded GPU base address
    void*    live_ptr;  // live replay GPU base address
    size_t   size;
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
    std::unordered_map<uint64_t, hipGraphExec_t> graph_exec_map;
    std::unordered_map<uint64_t, hipSurfaceObject_t> surface_map;
    std::unordered_map<uint64_t, hipTextureObject_t> texture_map;

    // Device allocations: recorded base address -> {live ptr, size}
    std::unordered_map<uint64_t, AllocEntry>     alloc_map;

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
    bool verbose           = false;
    bool validate_d2h      = false;  // perform D2H validation against captured expected data
    std::string kernel_filter;

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
    std::atomic<size_t> d2h_fail{0};
    // Incremented for every D2H event that had a captured blob hash (i.e., validation
    // was expected). Includes pass + fail + skipped (missing ptr / missing blob).
    // If d2h_attempted > 0 but d2h_pass == 0 && d2h_fail == 0, every check was
    // skipped — pointer translation or blob loading failed for all D2H events.
    std::atomic<size_t> d2h_attempted{0};

    // Timing events — one pair per replay thread, created on first kernel launch
    // and reused for every subsequent launch on that thread. Registered here so
    // cleanup can destroy them without per-kernel create/destroy overhead.
    // Appended under map_mutex (unique_lock); no lock needed to read (single writer).
    std::vector<hipEvent_t> owned_timing_events;

    // Backing buffers for hipHostRegister replay: recorded ptr -> malloc'd host buffer.
    // Needed so hipHostUnregister can call hipHostUnregister then free the buffer.
    // Guarded by map_mutex.
    std::unordered_map<uint64_t, void*> host_reg_bufs;

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

    // ---- Pointer translation ----
    // Translates a recorded GPU address to a live pointer.
    // Checks alloc_map (exact + range) then vmm_va_map (exact + range).
    void* translate_ptr(uint64_t rec) const {
        if (rec == 0) return nullptr;
        std::shared_lock lk(map_mutex);
        auto it = alloc_map.find(rec);
        if (it != alloc_map.end()) return it->second.live_ptr;
        // Range search for sub-allocations
        for (auto& [base, entry] : alloc_map) {
            if (rec >= base && rec < base + entry.size)
                return static_cast<char*>(entry.live_ptr) +
                       static_cast<ptrdiff_t>(rec - base);
        }
        // Fall back to VMM reserved-VA map (exact + range)
        auto vit = vmm_va_map.find(rec);
        if (vit != vmm_va_map.end()) return vit->second.live;
        for (auto& [base, va] : vmm_va_map) {
            if (rec >= base && rec < base + va.size)
                return static_cast<char*>(va.live) +
                       static_cast<ptrdiff_t>(rec - base);
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

    // ---- Allocation registration (exclusive lock) ----
    void record_alloc(uint64_t rec, void* live, size_t sz) {
        std::unique_lock lk(map_mutex);
        alloc_map[rec] = {rec, live, sz};
    }
    void remove_alloc(uint64_t rec) {
        std::unique_lock lk(map_mutex);
        alloc_map.erase(rec);
    }

    // ---- Handle registration (exclusive lock) ----
    void record_stream  (uint64_t rec, hipStream_t     live) { std::unique_lock lk(map_mutex); stream_map[rec]     = live; }
    void record_event   (uint64_t rec, hipEvent_t      live) { std::unique_lock lk(map_mutex); event_map[rec]      = live; }
    void record_module  (uint64_t rec, hipModule_t     live) { std::unique_lock lk(map_mutex); module_map[rec]     = live; }
    void record_func    (uint64_t rec, hipFunction_t   live) { std::unique_lock lk(map_mutex); func_map[rec]       = live; }
    void record_mempool (uint64_t rec, hipMemPool_t    live) { std::unique_lock lk(map_mutex); mempool_map[rec]    = live; }
    void record_array   (uint64_t rec, hipArray_t      live) { std::unique_lock lk(map_mutex); array_map[rec]      = live; }
    void record_mipmapped(uint64_t rec, hipMipmappedArray_t live) { std::unique_lock lk(map_mutex); mipmapped_map[rec] = live; }
    void record_graph   (uint64_t rec, hipGraph_t      live) { std::unique_lock lk(map_mutex); graph_map[rec]      = live; }
    void record_graph_exec(uint64_t rec, hipGraphExec_t live){ std::unique_lock lk(map_mutex); graph_exec_map[rec] = live; }
    void record_surface (uint64_t rec, hipSurfaceObject_t live) { std::unique_lock lk(map_mutex); surface_map[rec] = live; }
    void record_texture (uint64_t rec, hipTextureObject_t live) { std::unique_lock lk(map_mutex); texture_map[rec] = live; }

    // ---- Handle removal (exclusive lock) ----
    void remove_stream  (uint64_t rec) { std::unique_lock lk(map_mutex); stream_map.erase(rec); }
    void remove_event   (uint64_t rec) { std::unique_lock lk(map_mutex); event_map.erase(rec); }
    void remove_module  (uint64_t rec) { std::unique_lock lk(map_mutex); module_map.erase(rec); }
    void remove_func    (uint64_t rec) { std::unique_lock lk(map_mutex); func_map.erase(rec); }
    void remove_mempool (uint64_t rec) { std::unique_lock lk(map_mutex); mempool_map.erase(rec); }
    void remove_array   (uint64_t rec) { std::unique_lock lk(map_mutex); array_map.erase(rec); }
    void remove_mipmapped(uint64_t rec){ std::unique_lock lk(map_mutex); mipmapped_map.erase(rec); }
    void remove_graph   (uint64_t rec) { std::unique_lock lk(map_mutex); graph_map.erase(rec); }
    void remove_graph_exec(uint64_t rec){ std::unique_lock lk(map_mutex); graph_exec_map.erase(rec); }
    void remove_surface (uint64_t rec) { std::unique_lock lk(map_mutex); surface_map.erase(rec); }
    void remove_texture (uint64_t rec) { std::unique_lock lk(map_mutex); texture_map.erase(rec); }

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

private:
    mutable std::unordered_map<std::string, std::vector<uint8_t>> blob_cache_;
};

// Thread-local sequence ID — set by dispatch_event before calling any handler.
// Kernel-launch handlers read this to wait for their submission turn at the
// exact point of the HIP call, allowing preparation work to run in parallel.
extern thread_local uint64_t hrr_dispatch_seq;

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
