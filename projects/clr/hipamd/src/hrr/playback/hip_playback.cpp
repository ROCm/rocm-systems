/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
//
// hip_playback.cpp — Manual playback implementations for APIs that need
// complex handling: kernel launches, H2D memcpy with blobs, module load
// from code objects, and hipModuleGetFunction name resolution.
//
// Also implements PlaybackContext helpers: load_blob, load_code_object,
// load_module.

#include "hip_playback.h"
#include "hrr_api_args.h"
#include "hrr_reader.h"   // hrr::hash_hex

#include <hip/hip_runtime.h>
// hipExtModuleLaunchKernel is declared in <hip/hip_ext.h>. That header redeclares
// symbols whose attributes trip -Werror=attributes against the runtime headers
// already pulled in above, so wrap the include in a diagnostic guard rather than
// hand-declaring the exported symbol (a hand declaration silently diverges from
// the library ABI if the signature ever changes).
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif
#include <hip/hip_ext.h>
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <future>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <mutex>
#include <set>
#ifdef _WIN32
#include <process.h>  // _exit
#else
#include <unistd.h>   // _exit
#endif

// Thread-local sequence ID — set by dispatch_event before calling any handler.
// Kernel-launch handlers use this to wait for their submission turn and then
// immediately unblock the next thread before doing timing/sync.
thread_local uint64_t hrr_dispatch_seq = 0;

void hrr_note_recorded_error(PlaybackContext& ctx, const char* api,
                             int recorded_ret) {
    {
        std::unique_lock lk(ctx.map_mutex);
        if (!ctx.reproduced_errors.emplace(api, recorded_ret).second) return;
    }
    fprintf(stderr,
            "[HRR] %s: returned %d (%s) at replay, which is what it returned at "
            "capture — the recorded failure was reproduced faithfully.\n",
            api, recorded_ret,
            hipGetErrorString(static_cast<hipError_t>(recorded_ret)));
}

bool hrr_replayed_recorded_error(PlaybackContext& ctx, const char* api,
                                 int32_t recorded_ret, hipError_t replayed) {
    if (replayed == hipSuccess || recorded_ret == 0 ||
        static_cast<int32_t>(replayed) != recorded_ret)
        return false;
    hrr_note_recorded_error(ctx, api, recorded_ret);
    return true;
}

hipCtx_t hrr_live_ctx(uint64_t recorded) {
    if (recorded == 0) return nullptr;
    hipCtx_t cur = nullptr;
    if (hipCtxGetCurrent(&cur) != hipSuccess) return nullptr;
    return cur;
}

// ---------------------------------------------------------------------------
// HIP error checking — returns the hipError_t so callers can branch on it.
// Usage:  HRR_HIP_CHECK(hipFoo(...));                      // log only
//         if (HRR_HIP_CHECK(hipFoo(...)) != hipSuccess) {} // log + branch
// ---------------------------------------------------------------------------

static inline hipError_t hrr_hip_check(hipError_t e, const char* call,
                                        const char* file, int line) {
    if (e != hipSuccess)
        fprintf(stderr, "[HRR] HIP error %d (%s): %s (%s:%d)\n",
                e, hipGetErrorString(e), call, file, line);
    return e;
}
#define HRR_HIP_CHECK(call) hrr_hip_check((call), #call, __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Sync watchdog
// ---------------------------------------------------------------------------
// hipDeviceSynchronize() blocks forever if a kernel is deadlocked (e.g. a
// StreamK producer/consumer flag spin-wait where the producer's flag store and
// the consumer's poll resolve to different addresses). When a watchdog timeout
// is configured, run the sync on a helper thread and bound the wait; on timeout
// the GPU is wedged, so we print an actionable diagnostic and hard-exit rather
// than hang the whole replay. A normally-completing sync — including one that
// reports a genuine GPU fault — is returned to the caller unchanged.
hipError_t hrr_watchdog_device_sync(PlaybackContext& ctx, const char* what) {
    const unsigned timeout_ms = ctx.sync_watchdog_ms;
    if (timeout_ms == 0)
        return hipDeviceSynchronize();

    auto fut = std::async(std::launch::async, [] { return hipDeviceSynchronize(); });
    if (fut.wait_for(std::chrono::milliseconds(timeout_ms)) ==
        std::future_status::ready) {
        return fut.get();
    }

    // Timed out. The async sync thread is stuck in the driver and cannot be
    // joined; abandon it and exit hard (_exit skips destructors, so the
    // detached future does not block trying to join the wedged thread).
    fflush(stdout);
    fprintf(stderr,
            "\n[HRR][WATCHDOG] GPU sync did not complete within %u ms at: %s\n"
            "[HRR][WATCHDOG] Treating this as a hung/deadlocked kernel (e.g. a StreamK\n"
            "[HRR][WATCHDOG] producer/consumer flag spin-wait). Re-run with --trace-kernels\n"
            "[HRR][WATCHDOG] to see the last launch, or attach rocgdb to inspect wavefronts.\n",
            timeout_ms, (what && *what) ? what : "device synchronize");
    fflush(stderr);
    _exit(124);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

static hipError_t hrr_sync_after_replayed_h2d(PlaybackContext& ctx,
                                                const char* what) {
    // Replay substitutes captured blobs for capture-time host pointers. Drain
    // the restore so following kernels see the replayed input bytes. This is
    // skipped during graph capture because device/stream sync is illegal there.
    if (!hrr_replayed_h2d_needs_drain(ctx.in_graph_capture)) return hipSuccess;
    // Clear any pre-existing sticky error so we attribute only an error surfaced
    // by this drain, matching the kernel-launch sync path.
    (void)hipGetLastError();
    hipError_t r = hrr_watchdog_device_sync(ctx, what);
    hipError_t last_r = hipGetLastError();
    if (r == hipSuccess && last_r != hipSuccess) r = last_r;
    return r;
}

static std::string compact_kernel_name(const std::string& name) {
    constexpr size_t kMax = 120;
    if (name.size() <= kMax) return name;
    return name.substr(0, 96) + "..." + name.substr(name.size() - 21);
}

// Build path: archive_dir/blobs/<2-char-prefix>/<hex>.blob
static std::string blob_path(const std::string& archive_dir,
                             uint64_t hash_lo, uint64_t hash_hi) {
    std::string hex = hrr::hash_hex(hash_lo, hash_hi);
    return archive_dir + "/blobs/" + hex.substr(0, 2) + "/" + hex + ".blob";
}

// Build path: archive_dir/code_objects/<hex>.hsaco
static std::string co_path(const std::string& archive_dir,
                            uint64_t hash_lo, uint64_t hash_hi) {
    return archive_dir + "/code_objects/" + hrr::hash_hex(hash_lo, hash_hi) + ".hsaco";
}

static std::vector<uint8_t> read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return {}; }
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    if (fread(buf.data(), 1, buf.size(), f) != buf.size()) { fclose(f); return {}; }
    fclose(f);
    return buf;
}

static void maybe_trace_progress(PlaybackContext& ctx, size_t kernel_ordinal,
                                 const std::string& kernel_name) {
    const bool by_count = ctx.progress_kernel_interval != 0 &&
                          (kernel_ordinal == 1 ||
                           kernel_ordinal % ctx.progress_kernel_interval == 0);
    const bool by_time_enabled = ctx.progress_seconds_interval > 0.0;
    if (!by_count && !by_time_enabled) return;

    auto now = std::chrono::steady_clock::now();
    bool by_time = false;
    double elapsed_s = 0.0;
    {
        std::lock_guard<std::mutex> lk(ctx.progress_mutex);
        if (ctx.progress_start_time.time_since_epoch().count() == 0) {
            ctx.progress_start_time = now;
            ctx.progress_last_time = now;
            by_time = true;
        } else {
            elapsed_s = std::chrono::duration<double>(
                now - ctx.progress_start_time).count();
            if (by_time_enabled) {
                double since_last = std::chrono::duration<double>(
                    now - ctx.progress_last_time).count();
                if (since_last >= ctx.progress_seconds_interval) {
                    ctx.progress_last_time = now;
                    by_time = true;
                }
            }
        }
    }

    if (!by_count && !by_time) return;
    fprintf(stderr,
            "[HRR progress] elapsed_s=%.1f seq=%llu kernels=%zu d2h_pass=%zu "
            "d2h_fail=%zu d2h_attempted=%zu last=\"%s\"\n",
            elapsed_s,
            (unsigned long long)hrr_dispatch_seq,
            kernel_ordinal,
            ctx.d2h_pass.load(std::memory_order_relaxed),
            ctx.d2h_fail.load(std::memory_order_relaxed),
            ctx.d2h_attempted.load(std::memory_order_relaxed),
            compact_kernel_name(kernel_name).c_str());
    fflush(stderr);
}

}  // namespace

// ---------------------------------------------------------------------------
// PlaybackContext: blob / code-object loading
// ---------------------------------------------------------------------------

const void* PlaybackContext::load_blob(uint64_t hash_lo, uint64_t hash_hi,
                                       size_t* sz_out) const {
    if (!hash_lo && !hash_hi) return nullptr;
    std::string key = hrr::hash_hex(hash_lo, hash_hi);
    {
        std::shared_lock lk(map_mutex);
        auto it = blob_cache_.find(key);
        if (it != blob_cache_.end()) {
            if (sz_out) *sz_out = it->second.size();
            return it->second.data();
        }
    }
    // Not cached — read from disk, then insert under exclusive lock
    auto data = read_file(blob_path(archive_dir, hash_lo, hash_hi));
    if (data.empty()) return nullptr;
    std::unique_lock lk(map_mutex);
    auto it = blob_cache_.emplace(key, std::move(data)).first;
    if (sz_out) *sz_out = it->second.size();
    return it->second.data();
}

const void* PlaybackContext::load_code_object(uint64_t hash_lo, uint64_t hash_hi,
                                              size_t* sz_out) const {
    if (!hash_lo && !hash_hi) return nullptr;
    // Prefix with "co:" to avoid colliding with blobs of the same hash in blob_cache_.
    std::string key = "co:" + hrr::hash_hex(hash_lo, hash_hi);
    {
        std::shared_lock lk(map_mutex);
        auto it = blob_cache_.find(key);
        if (it != blob_cache_.end()) {
            if (sz_out) *sz_out = it->second.size();
            return it->second.data();
        }
    }
    auto data = read_file(co_path(archive_dir, hash_lo, hash_hi));
    if (data.empty()) return nullptr;
    std::unique_lock lk(map_mutex);
    auto [it, inserted] = blob_cache_.emplace(key, std::move(data));
    (void)inserted;
    if (sz_out) *sz_out = it->second.size();
    return it->second.data();
}

hipModule_t PlaybackContext::load_module(uint64_t hash_lo, uint64_t hash_hi) {
    std::string hex = hrr::hash_hex(hash_lo, hash_hi);
    {
        std::shared_lock lk(map_mutex);
        auto it = co_modules.find(hex);
        if (it != co_modules.end()) return it->second;
    }

    // Cache miss — load without holding any lock (disk I/O + GPU call)
    // Drain any deferred GPU error before loading: hipModuleLoadData triggers
    // a driver state check that surfaces faults from prior async kernel launches
    // even after hipDeviceSynchronize() has returned success.
    {
        hipError_t pre_sync = hipDeviceSynchronize();
        hipError_t pre_err  = hipGetLastError();
        if (pre_sync != hipSuccess || pre_err != hipSuccess)
            fprintf(stderr, "[HRR] load_module %s: pre-drain sync=%d err=%d\n",
                    hex.c_str(), pre_sync, pre_err);
    }

    size_t sz = 0;
    const void* data = load_code_object(hash_lo, hash_hi, &sz);
    if (!data || sz == 0) {
        fprintf(stderr, "[HRR] Code object %s not found in archive\n", hex.c_str());
        return nullptr;
    }
    hipModule_t mod = nullptr;
    hipError_t err = hipModuleLoadData(&mod, data);
    if (err != hipSuccess) {
        fprintf(stderr, "[HRR] Failed to load code object %s: %d (%s)\n",
                hex.c_str(), err, hipGetErrorString(err));
        return nullptr;
    }

    // Re-acquire with exclusive lock; if a concurrent thread already loaded
    // this module, discard ours to avoid a double-load leak.
    std::unique_lock lk(map_mutex);
    auto [it, inserted] = co_modules.emplace(hex, mod);
    if (!inserted) {
        (void)hipModuleUnload(mod);
        return it->second;
    }
    if (verbose)
        fprintf(stderr, "[HRR] Loaded code object %s (%zu bytes)\n", hex.c_str(), sz);
    return mod;
}

hipFunction_t PlaybackContext::resolve_replacement(const std::string& kernel_name) {
    // Fast path: nothing to do if no replacements were requested.
    if (kernel_replacements.empty()) return nullptr;

    // Cache hit — already resolved (or already resolved to "no replacement").
    {
        std::shared_lock lk(map_mutex);
        auto it = replacement_funcs.find(kernel_name);
        if (it != replacement_funcs.end()) return it->second;
    }

    // Find the replacement whose NAME exactly equals the recorded kernel name.
    // Exact match (not substring) so a replacement can never accidentally apply
    // to an unintended kernel; the NAME must be the full recorded symbol (the
    // mangled C++ name for chevron/HIP-RTC kernels, or the module symbol for
    // hipModuleGetFunction kernels) — the same string hipModuleGetFunction is
    // called with below.
    const std::string* path = nullptr;
    for (auto& [name, p] : kernel_replacements) {
        if (kernel_name == name) { path = &p; break; }
    }
    // No pattern matched: cache the negative result so we don't re-scan every launch.
    if (!path) {
        std::unique_lock lk(map_mutex);
        replacement_funcs.emplace(kernel_name, nullptr);
        return nullptr;
    }

    // Load the replacement code object from the filesystem path and resolve the
    // function by the recorded (same) symbol name. Mirrors load_module()'s
    // hipModuleLoadData + hipModuleGetFunction pattern; the input here is a path,
    // not an archive hash, so we read the file directly.
    auto data = read_file(*path);
    hipFunction_t func = nullptr;
    if (data.empty()) {
        fprintf(stderr, "[HRR] --replace-kernel: cannot read '%s' for kernel '%s' — "
                "using recorded kernel\n", path->c_str(), kernel_name.c_str());
    } else {
        hipModule_t mod = nullptr;
        hipError_t err = hipModuleLoadData(&mod, data.data());
        if (err != hipSuccess) {
            fprintf(stderr, "[HRR] --replace-kernel: failed to load '%s': %d (%s) — "
                    "using recorded kernel\n", path->c_str(), err, hipGetErrorString(err));
        } else if (hipModuleGetFunction(&func, mod, kernel_name.c_str()) != hipSuccess
                   || !func) {
            fprintf(stderr, "[HRR] --replace-kernel: symbol '%s' not found in '%s' — "
                    "using recorded kernel\n", kernel_name.c_str(), path->c_str());
            func = nullptr;
            (void)hipModuleUnload(mod);
        } else {
            std::unique_lock lk(map_mutex);
            replacement_modules.push_back(mod);
            // Announce the successful replacement on stdout: it is a deliberate,
            // user-requested replay action (a result of --replace-kernel), not a
            // diagnostic. stdout is also what carries the replay summary callers
            // parse; the fallback warnings above stay on stderr.
            printf("[HRR] Replacing kernel '%s' with %s (symbol %s)\n",
                   kernel_name.c_str(), path->c_str(), kernel_name.c_str());
            fflush(stdout);
        }
    }

    // Cache result (func or nullptr-on-failure) so each kernel is resolved once.
    std::unique_lock lk(map_mutex);
    replacement_funcs[kernel_name] = func;
    return func;
}

// ---------------------------------------------------------------------------
// Kernel launch — shared implementation used by all four launch APIs
// ---------------------------------------------------------------------------
// Kernel launch payload (raw_payload, bytes after 32-byte EventHeader):
//   [0..7]   stream_handle (uint64_t)
//   [8..9]   name_len (uint16_t)
//   [10..]   kernel_name (name_len bytes, no NUL)
//   [+0..7]  co_hash_lo (uint64_t)
//   [+8..15] co_hash_hi (uint64_t)
//   [+0..11] grid[3]   (uint32_t[3])
//   [+12..23] block[3] (uint32_t[3])
//   [+24..27] shared_mem (uint32_t)
//   [+28..29] num_args (uint16_t)
//   [+30..31] num_snapshots (uint16_t, always 0)
//   per arg:  u8 value_kind, u16 size, <size> bytes data
//             value_kind: 0=scalar, 1=gpu-pointer, 2=hidden,
//                         3=scalar/struct with embedded gpu pointer(s);
//             kind 3 appends u16 n_ptrs then n_ptrs * u16 byte offsets.

// ext_global_worksize: the captured grid[] holds *global work-item counts*
// (HSA/OpenCL semantics, as passed to hipExtModuleLaunchKernel), NOT workgroup
// counts. When false the grid[] holds workgroup counts (hipModuleLaunchKernel /
// hipLaunchKernel semantics). This MUST match the API the launch was captured
// from: replaying an Ext launch (global work items) through
// hipModuleLaunchKernel (which treats the dims as workgroup counts) over-launches
// the grid by a factor of the block size in each dimension. For persistent,
// co-resident kernels (e.g. hipBLASLt StreamK producer/consumer flag handshakes)
// that blow-up deadlocks the kernel: only the first wave of workgroups is
// resident and the spinning consumers wait forever on producers that live in
// later, never-scheduled waves.
// hipExtModuleLaunchKernel is declared via <hip/hip_ext.h> (included at the top
// of this file behind a -Wattributes diagnostic guard) so the prototype always
// tracks the library ABI instead of a hand-maintained copy.
// launch_ex: the event came from hipDrvLaunchKernelEx or hipLaunchKernelExC.
// Their recorded attribute list is applied by replaying through
// hipDrvLaunchKernelEx with a rebuilt descriptor, so a cluster dimension or a
// cooperative flag the program asked for is not silently dropped. The C
// spelling replays through the driver entry point too: its host function
// address means nothing here, while the resolved hipFunction_t does.

// ---- External region annotations at the translation sites -------------------
//
// Every recorded device pointer a kernel receives passes through one of the
// translation sites in replay_kernel_launch, which makes them the one place where
// a pointer can be compared against what an external producer said the memory
// layout was. Two questions get answered here:
//
//   - Does this pointer land inside an annotated segment but in no live block?
//     That is an intra-segment out-of-bounds or stale pointer. Replay cannot
//     see it otherwise: the segment is a single contiguous allocation, so the
//     access is in bounds as far as HIP is concerned.
//   - Should the owning block be relocated behind a guard page so an overrun
//     faults instead of hitting a neighbour? Only when block guarding is on.

// One VMM-backed buffer standing in for a block, with an unmapped guard span
// after it. Populated by hrr_block_guard_alloc.
struct HrrBlockGuard {
    void*  va        = nullptr;  // reserved VA base
    size_t reserved  = 0;        // total reserved VA (mapped + guard)
    size_t mapped    = 0;        // mapped/backed bytes
    void*  data      = nullptr;  // where the block itself starts inside `va`
    size_t size      = 0;        // block size in bytes
    void*  orig_live = nullptr;  // the real block, to copy results back to
    hipMemGenericAllocationHandle_t handle{};
};

// Per-launch guard bookkeeping, owned by replay_kernel_launch. Null on the
// graph kernel-node path: a node's arguments outlive the call that built it, so
// a relocation there could never be undone.
struct RegionLaunchState {
    std::vector<HrrBlockGuard>          guards;     // torn down after the launch
    std::unordered_map<uint64_t, void*> relocated;  // rec block base -> buffer
};

static void* hrr_block_guard_relocate(PlaybackContext& ctx, RegionLaunchState& rls,
                                      uint64_t rec_ptr, void* live,
                                      uint64_t blk_base, uint64_t blk_size);
static void  hrr_block_guard_teardown(const HrrBlockGuard& g);
static hipError_t hrr_block_guard_resolve(PlaybackContext& ctx,
                                          RegionLaunchState& rls,
                                          const std::string& kernel_name,
                                          size_t kernel_ordinal);

// Returns the pointer the kernel should actually receive: `live` unchanged
// unless block guarding relocated its owning block. Cheap no-op when no region
// sidecar was loaded.
static void* hrr_region_check_ptr(PlaybackContext& ctx, RegionLaunchState* rls,
                                  const std::string& kernel_name,
                                  unsigned arg_index, uint64_t rec_ptr,
                                  void* live) {
    if (!ctx.regions_enabled || rec_ptr < 0x10000ULL) return live;

    uint64_t blk_base = 0, blk_size = 0;
    const auto cls = ctx.regions.classify(rec_ptr, &blk_base, &blk_size);
    if (cls == hrr::RegionMap::Class::None) return live;

    ctx.region_ptrs_checked.fetch_add(1, std::memory_order_relaxed);

    if (cls == hrr::RegionMap::Class::InSegmentNoBlock) {
        ctx.region_oob_ptrs.fetch_add(1, std::memory_order_relaxed);
        static std::mutex mu;
        static std::set<std::string> warned;
        char key[192];
        snprintf(key, sizeof(key), "%s#%u", kernel_name.c_str(), arg_index);
        bool first;
        { std::lock_guard<std::mutex> lk(mu); first = warned.insert(key).second; }
        if (first)
            fprintf(stderr,
                    "[HRR] region OOB: '%s' arg[%u] recorded 0x%llx is inside an "
                    "annotated segment but in no live block — an intra-segment "
                    "out-of-bounds or stale pointer\n",
                    compact_kernel_name(kernel_name).c_str(), arg_index,
                    (unsigned long long)rec_ptr);
        return live;
    }

    // Never while a graph capture is active: the launch is being recorded into a
    // graph rather than executed, so the relocation would still be referenced
    // long after this call returns, and the copy-in and post-launch sync the
    // guard needs are both illegal on a capturing stream (HIP 901).
    if (rls && live && blk_size > 0 && ctx.guard_blocks && !ctx.in_graph_capture)
        live = hrr_block_guard_relocate(ctx, *rls, rec_ptr, live, blk_base, blk_size);
    return live;
}

// Resolve the hipFunction_t a recorded kernel name (plus the code-object hash
// that disambiguates it) refers to in this process, or nullptr with a message
// naming the kernel. Shared by kernel launches and graph kernel nodes, which
// name their kernel the same way and for the same reason: the recorded host
// function address belongs to the capturing process.
static hipFunction_t resolve_kernel_function(PlaybackContext& ctx,
                                             const std::string& kernel_name,
                                             uint64_t co_hash_lo,
                                             uint64_t co_hash_hi) {
    hipFunction_t func = nullptr;

    // Playback-time kernel override: if this kernel matches a --replace-kernel
    // pattern, launch the user-supplied code object instead of the recorded one.
    // All recorded inputs (grid/block/shared/args/pointers) are still used below.
    // resolve_replacement returns nullptr when no replacement applies (or it
    // failed to load), in which case we fall through to the recorded kernel.
    if (!ctx.kernel_replacements.empty())
        func = ctx.resolve_replacement(kernel_name);

    // Resolve by (co_hash, name). Kernel symbol names are NOT globally unique:
    // Triton/inductor emits the generic entry symbol "triton_" from many distinct
    // code objects, so caching/searching by name alone binds every "triton_"
    // launch to one arbitrary kernel and faults (HIP 719 / VM fault). The recorded
    // code-object hash disambiguates which code object this launch came from.
    std::string cache_key = kernel_name;
    if (co_hash_lo || co_hash_hi) {
        char hpfx[34];
        snprintf(hpfx, sizeof(hpfx), "%016llx%016llx:",
                 (unsigned long long)co_hash_hi, (unsigned long long)co_hash_lo);
        cache_key.assign(hpfx);
        cache_key += kernel_name;
    }

    if (!func) {
        std::shared_lock lk(ctx.map_mutex);
        auto it = ctx.func_cache.find(cache_key);
        if (it != ctx.func_cache.end())
            func = it->second;
    }

    if (!func) {
        // Cache miss. Prefer the exact code object identified by the recorded
        // co_hash so non-unique names ("triton_") bind to the correct kernel;
        // only fall back to a name-only search across all loaded modules when no
        // hash was recorded (older recordings) or the hashed module lacks it.
        if (co_hash_lo || co_hash_hi) {
            hipModule_t mod = ctx.load_module(co_hash_lo, co_hash_hi);
            if (mod) (void)hipModuleGetFunction(&func, mod, kernel_name.c_str());
        }
        if (!func) {
            std::shared_lock lk(ctx.map_mutex);
            for (auto& [rec_mod, live_mod] : ctx.module_map) {
                if (hipModuleGetFunction(&func, live_mod, kernel_name.c_str()) == hipSuccess
                    && func) break;
                func = nullptr;
            }
            if (!func) {
                for (auto& [hex, mod] : ctx.co_modules) {
                    if (hipModuleGetFunction(&func, mod, kernel_name.c_str()) == hipSuccess
                        && func) break;
                    func = nullptr;
                }
            }
        }
        if (!func) {
            fprintf(stderr, "[HRR] Kernel '%s' not found in any loaded module\n",
                    kernel_name.c_str());
            return nullptr;
        }
        std::unique_lock lk(ctx.map_mutex);
        ctx.func_cache.emplace(cache_key, func);
    }
    return func;
}

// Decode the recorded argument list at `p` into `arg_storage` (owning) and
// `arg_ptrs` (what a launch or node-parameter build wants), translating every
// recorded device address on the way. `p` is advanced past the arguments.
//
// Shared by the kernel-launch path and the graph kernel-node path: a kernel
// node's arguments are the same bytes with the same pointers inside them, and
// a second decoder would be a second place for the pointer heuristics below to
// drift out of agreement with what capture recorded.
//
// `rls` is the caller's per-launch guard bookkeeping, or null on the graph
// kernel-node path where a relocation could never be undone.
static void decode_kernel_args(
    PlaybackContext& ctx, const uint8_t*& p, const uint8_t* end,
    uint16_t num_args, const std::string& kernel_name,
    std::vector<void*>& arg_ptrs,
    std::vector<std::vector<uint8_t>>& arg_storage,
    RegionLaunchState* rls = nullptr,
    std::vector<std::tuple<unsigned, uint64_t, void*>>* dbg_ptrs_out = nullptr) {
    (void)kernel_name;
    std::vector<std::tuple<unsigned, uint64_t, void*>> dbg_sink;
    const bool dbg_dump_ptrs = (dbg_ptrs_out != nullptr);
    auto& dbg_ptrs = dbg_ptrs_out ? *dbg_ptrs_out : dbg_sink;

    for (uint16_t i = 0; i < num_args; i++) {
        if (p + 3 > end) break;
        uint8_t  value_kind = *p++;
        uint16_t arg_size;
        memcpy(&arg_size, p, 2); p += 2;
        if (p + arg_size > end) break;

        const uint8_t* data = p;
        p += arg_size;

        // value_kind 3 carries a trailing list of embedded-pointer byte offsets.
        std::vector<uint16_t> ptr_offsets;
        if (value_kind == 3) {
            if (p + 2 > end) break;
            uint16_t n_ptrs; memcpy(&n_ptrs, p, 2); p += 2;
            for (uint16_t k = 0; k < n_ptrs; k++) {
                if (p + 2 > end) { n_ptrs = k; break; }
                uint16_t off; memcpy(&off, p, 2); p += 2;
                ptr_offsets.push_back(off);
            }
        }

        if (value_kind == 2) {  // hidden arg — skip
            continue;
        }
        arg_storage.emplace_back();
        auto& storage = arg_storage.back();
        if (value_kind == 1 && arg_size >= 8) {  // whole-arg GPU pointer
            uint64_t rec_ptr; memcpy(&rec_ptr, data, 8);
            void* live = ctx.translate_ptr(rec_ptr);
            // Nothing in the archive covers this address. If a producer said a
            // segment lived there, HIP was bypassed for it — back it now and
            // translate into it. Asked here rather than when the segment was
            // declared, because only a translation failure at the moment of use
            // proves no captured allocation covers the address.
            if (!live && rec_ptr != 0 && ctx.regions_enabled)
                live = ctx.regions.materialize_for(ctx, rec_ptr);

            if (!live && rec_ptr != 0) {
                // Counted and reported, but still handed to the kernel as null.
                // Forwarding the recorded address instead would be worse than
                // useless: the driver tends to reproduce a VA layout across
                // runs, so a capture-time address is quite likely to be mapped
                // in the replay process too, and the kernel would then scribble
                // over an unrelated live buffer with nothing to show for it. A
                // null dereference stops at the first kernel that uses the
                // pointer, which is the failure worth having.
                ctx.untranslated_ptr_args.fetch_add(1, std::memory_order_relaxed);
                static std::mutex mu;
                static std::set<std::string> warned;
                char key[160];
                snprintf(key, sizeof(key), "%s#%u", kernel_name.c_str(), i);
                bool first;
                { std::lock_guard<std::mutex> lk(mu); first = warned.insert(key).second; }
                if (first)
                    fprintf(stderr,
                            "[HRR] '%s' arg[%u]: recorded 0x%llx is in no known "
                            "allocation — passing null\n",
                            compact_kernel_name(kernel_name).c_str(), i,
                            (unsigned long long)rec_ptr);
            } else {
                // Region check: an intra-segment OOB/stale pointer, or a block to be
                // relocated behind a guard page. No-op when no sidecar was loaded.
                live = hrr_region_check_ptr(ctx, rls, kernel_name, i, rec_ptr, live);
            }
            if (dbg_dump_ptrs) dbg_ptrs.emplace_back(i, rec_ptr, live);
            storage.resize(sizeof(void*));
            memcpy(storage.data(), &live, sizeof(void*));
            if (ctx.verbose)
                fprintf(stderr, "[HRR]   arg[%u]: ptr 0x%llx -> %p%s\n",
                        i, (unsigned long long)rec_ptr, live,
                        live ? "" : " (MISSING!)");
        } else if (value_kind == 3) {  // scalar/struct with embedded gpu pointer(s)
            storage.assign(data, data + arg_size);

            // Translate the 8-byte word at `off` (read from the *original*
            // recorded bytes) and write the live pointer into storage, but only
            // when the recorded value actually resolves to a known allocation.
            //
            // The capture-side detector is a value-based heuristic: any 8-byte
            // word that happened to fall inside a live device VA was flagged. If
            // the recorded value does not resolve to a known allocation here, it
            // may be a genuine scalar (a large count, a double, a packed value)
            // that was mis-flagged — overwriting it with null would silently
            // corrupt it. Only rewrite the word when it actually resolves.
            // Reject packed-integer false positives. The capture-side detector
            // flags any 8-byte word that resolves to a device VA, but two adjacent
            // 32-bit struct fields {uint32 lo, uint32 hi} can coincidentally form
            // such a value: ATen elementwise kernels embed an OffsetCalculator
            // (per-arg uint32 strides/sizes + IntDivider magic constants) in the
            // functor. When `hi` happens to hold a value whose top bits match the
            // device-VA prefix (0x7e../0x7f..) and `lo` holds a small integer (a
            // stride/size/dim), the combined 64-bit word lands inside a real
            // allocation and gets "translated" — corrupting the OffsetCalculator
            // and producing an out-of-bounds VM fault (e.g. the recurring
            // elementwise_kernel_manual_unroll<...MulFunctor> crash).
            //
            // A genuine 64-bit device pointer carries a full 48-bit address, so its
            // low 32 bits are part of that address and are effectively never this
            // small. The FP, by construction, needs its HIGH word to be the VA
            // prefix and its LOW word to be a small scalar — so a tiny low-32 value
            // is the reliable FP signature. (Set HIP_HRR_PTR_RELAX=1 to disable.)
            static const bool ptr_relax =
                (std::getenv("HIP_HRR_PTR_RELAX") != nullptr);
            auto pointer_like = [&](uint64_t v) -> bool {
                if (ptr_relax) return true;
                return (v & 0xFFFFFFFFULL) >= 0x10000ULL;
            };
            auto try_translate_word = [&](size_t off, const char* src) -> bool {
                if (off + 8 > arg_size) return false;
                uint64_t rec_ptr; memcpy(&rec_ptr, data + off, 8);
                if (rec_ptr < 0x10000ULL) return false;  // null/small — never a VA
                if (!pointer_like(rec_ptr)) return false;  // packed-int false positive
                void* live = ctx.translate_ptr(rec_ptr);
                // Same bypassed-segment materialisation as a whole-pointer
                // argument: a struct field pointing into memory HIP never saw
                // resolves in no map until the segment is backed. Only an
                // address a producer declared materialises, so a mis-flagged
                // scalar still resolves nowhere and is left untouched below.
                if (!live && ctx.regions_enabled)
                    live = ctx.regions.materialize_for(ctx, rec_ptr);
                if (!live) return false;
                // Same region check as a whole-pointer argument: a device
                // address embedded in a by-value struct addresses a tensor
                // block just as much as one passed directly.
                live = hrr_region_check_ptr(ctx, rls, kernel_name, i, rec_ptr, live);
                memcpy(storage.data() + off, &live, sizeof(void*));
                if (dbg_dump_ptrs) dbg_ptrs.emplace_back(i, rec_ptr, live);
                if (ctx.verbose)
                    fprintf(stderr, "[HRR]   arg[%u]: embedded ptr @+%zu 0x%llx -> %p [%s]\n",
                            i, off, (unsigned long long)rec_ptr, live, src);
                return true;
            };

            // First honor the capture-recorded offsets (these may be unaligned).
            std::vector<char> handled(arg_size, 0);
            for (uint16_t off : ptr_offsets) {
                if (try_translate_word(off, "captured"))
                    for (int b = 0; b < 8 && static_cast<size_t>(off) + b < arg_size; b++)
                        handled[off + b] = 1;
                else if (ctx.verbose)
                    fprintf(stderr, "[HRR]   arg[%u]: embedded ptr @+%u unresolved — left as-is (possible scalar)\n",
                            i, off);
            }

            // Defensive rescan: the capture-side value-based detector can MISS an
            // embedded pointer. Its per-offset verdict is cached, so a struct slot
            // that held a non-resolving value on an early launch is frozen as a
            // scalar; when a later launch reuses that slot for a real device
            // pointer (e.g. the reused addresses[] slots in ATen's
            // multi_tensor_apply TensorListMetadata across launch waves), the
            // offset is never flagged and the stale recorded pointer reaches the
            // GPU — a guaranteed VM fault. Here we have the full recorded
            // allocation map, so we re-scan every word and translate any that
            // resolves. Genuine scalars (small counts/shapes) never fall inside a
            // recorded device VA, so this does not corrupt them.
            static const bool no_rescan =
                (std::getenv("HIP_HRR_REPLAY_NO_RESCAN") != nullptr);
            for (size_t off = 0; !no_rescan && off + 8 <= arg_size; ) {
                if (handled[off]) { off += 1; continue; }
                if (try_translate_word(off, "rescan")) off += 8;
                else off += 1;
            }
        } else {
            storage.assign(data, data + arg_size);
            if (ctx.verbose) {
                // Print scalar args as hex bytes for debugging
                fprintf(stderr, "[HRR]   arg[%u]: scalar %u bytes = ", i, arg_size);
                for (uint16_t b = 0; b < arg_size && b < 8; b++)
                    fprintf(stderr, "%02x", data[b]);
                if (arg_size > 8) fprintf(stderr, "...");
                // Also print as u32/u64 for convenience
                if (arg_size == 4) { uint32_t v; memcpy(&v, data, 4); fprintf(stderr, " (u32=%u)", v); }
                if (arg_size == 8) { uint64_t v; memcpy(&v, data, 8); fprintf(stderr, " (u64=%llu)", (unsigned long long)v); }
                fprintf(stderr, "\n");
            }
        }
        arg_ptrs.push_back(storage.data());
    }
}

static hipError_t replay_kernel_launch(PlaybackContext& ctx, const uint8_t* pl,
                                       bool ext_global_worksize = false,
                                       bool launch_ex = false,
                                       bool cooperative = false) {
    // Skip the 32-byte header; kernel launch has a variable-length binary format.
    const auto* hdr = reinterpret_cast<const hrr_event_header*>(pl);
    const uint8_t* p   = pl + sizeof(hrr_event_header);
    const uint8_t* end = pl + hdr->payload_length;

    if (p + 8 > end) return hipErrorInvalidValue;
    uint64_t stream_rec; memcpy(&stream_rec, p, 8); p += 8;

    if (p + 2 > end) return hipErrorInvalidValue;
    uint16_t name_len; memcpy(&name_len, p, 2); p += 2;
    if (p + name_len > end) return hipErrorInvalidValue;
    std::string kernel_name(reinterpret_cast<const char*>(p), name_len);
    p += name_len;

    // Workaround for recordings made before the capture side tagged Ext launches:
    // hipBLASLt/Tensile ("Cijk_*") StreamK kernels are launched via
    // hipExtModuleLaunchKernel, whose grid[] is *global work-item counts*. If such
    // a launch was collapsed into the generic (workgroup-count) launch event, the
    // grid is over-launched by blockDim and the persistent producer/consumer
    // handshake deadlocks. Setting HIP_HRR_REPLAY_FORCE_EXT_CIJK=1 reinterprets
    // these grids as global work items (replay through the Ext API).
    //
    // SUNSET: this is a backward-compat escape hatch only. New recordings record
    // hipExtModuleLaunchKernel under HRR_API_HIPEXTMODULELAUNCHKERNEL, whose
    // dedicated playback handler already passes ext_global_worksize=true (see
    // playback_hipExtModuleLaunchKernel below), so they never need this path. The
    // heuristic depends on the third-party Tensile/hipBLASLt "Cijk_" naming
    // convention, which can change without notice. Safe to delete once no archive
    // predating the capture-side Ext-tagging fix is still being replayed (i.e.
    // every recording in use routes Ext launches through their own event id).
    if (!ext_global_worksize && kernel_name.compare(0, 5, "Cijk_") == 0 &&
        std::getenv("HIP_HRR_REPLAY_FORCE_EXT_CIJK"))
        ext_global_worksize = true;

    uint64_t co_hash_lo = 0, co_hash_hi = 0;
    if (p + 16 <= end) {
        memcpy(&co_hash_lo, p, 8); p += 8;
        memcpy(&co_hash_hi, p, 8); p += 8;
    }

    if (p + 32 > end) return hipErrorInvalidValue;
    uint32_t grid[3], block[3], shared_mem;
    memcpy(grid,       p, 12); p += 12;
    memcpy(block,      p, 12); p += 12;
    memcpy(&shared_mem, p, 4); p +=  4;

    uint16_t num_args, num_snapshots;
    memcpy(&num_args,       p, 2); p += 2;
    memcpy(&num_snapshots,  p, 2); p += 2;

    // Apply kernel filter if set
    if (!ctx.kernel_filter.empty() &&
        kernel_name.find(ctx.kernel_filter) == std::string::npos)
        return hipSuccess;

    hipFunction_t func = resolve_kernel_function(ctx, kernel_name,
                                                 co_hash_lo, co_hash_hi);
    if (!func) return hipErrorNotFound;

    // Build kernelParams[] from captured args, translating GPU pointers.
    std::vector<void*>                arg_ptrs;
    std::vector<std::vector<uint8_t>> arg_storage;
    // Guarded blocks this launch relocated, if block guarding is on. Scoped to
    // the launch so the relocation is undone before the next event runs.
    RegionLaunchState rls;
    // Optional recorded->live pointer dump for one target kernel (diff tooling).
    const bool dbg_dump_ptrs = (ctx.dump_ptrs_ordinal != 0);
    std::vector<std::tuple<unsigned, uint64_t, void*>> dbg_ptrs;  // (arg_idx, recorded, live)
    decode_kernel_args(ctx, p, end, num_args, kernel_name, arg_ptrs,
                       arg_storage, &rls, dbg_dump_ptrs ? &dbg_ptrs : nullptr);

    // Launch-attribute tail: u32 count, u32 per-entry stride, then the
    // entries. Every launch payload carries it (count 0 for a plain launch).
    std::vector<hipLaunchAttribute> launch_attrs;
    if (p + 8 <= end) {
        uint32_t n_attrs = 0, stride = 0;
        memcpy(&n_attrs, p, 4); p += 4;
        memcpy(&stride,  p, 4); p += 4;
        if (n_attrs) {
            const size_t have = static_cast<size_t>(end - p);
            if (stride != sizeof(hipLaunchAttribute) ||
                have < static_cast<size_t>(n_attrs) * stride) {
                fprintf(stderr,
                        "[HRR] '%s': launch attributes recorded with a %u-byte "
                        "entry (this build expects %zu) — launching without "
                        "them; the replayed launch is not the recorded one\n",
                        kernel_name.c_str(), stride, sizeof(hipLaunchAttribute));
            } else {
                launch_attrs.resize(n_attrs);
                memcpy(launch_attrs.data(), p,
                       static_cast<size_t>(n_attrs) * stride);
                p += static_cast<size_t>(n_attrs) * stride;
                for (auto& at : launch_attrs) {
                    // Two attribute values are themselves pointers. The access
                    // window names a device buffer, which the allocation map
                    // can resolve; the prefetch config is a host struct that
                    // was never recorded, so the attribute is dropped rather
                    // than passed as a capture-time address.
                    if (at.id == hipLaunchAttributeAccessPolicyWindow) {
                        at.val.accessPolicyWindow.base_ptr = ctx.translate_ptr(
                            reinterpret_cast<uint64_t>(
                                at.val.accessPolicyWindow.base_ptr));
                    } else if (at.id == hipLaunchAttributeExtDynDataPrefetch) {
                        fprintf(stderr,
                                "[HRR] '%s': dropping the dynamic-data-prefetch "
                                "launch attribute; its config lives in host "
                                "memory that the archive does not carry\n",
                                kernel_name.c_str());
                        at.id = hipLaunchAttributeIgnore;
                    }
                }
            }
        }
    }

    hipStream_t stream = ctx.translate_stream(stream_rec);
    const size_t kernel_ordinal =
        ctx.kernels_launched.load(std::memory_order_relaxed) + 1;

    if (dbg_dump_ptrs && kernel_ordinal == ctx.dump_ptrs_ordinal) {
        fprintf(stderr,
                "[HRR ptr-dump] kernel #%zu \"%s\" recorded->live pointer args:\n",
                kernel_ordinal, compact_kernel_name(kernel_name).c_str());
        for (auto& [idx, rec, live] : dbg_ptrs)
            fprintf(stderr, "[HRR ptr-dump]   arg[%u] recorded=0x%llx -> live=%p\n",
                    idx, (unsigned long long)rec, live);
        fflush(stderr);
    }


    // Skip HIP event timing during graph capture: recording events on a
    // captured stream inserts them into the graph and invalidates the
    // capture state (error 901 on all subsequent operations).
    const bool do_timing = ctx.timing && !ctx.in_graph_capture;

    // Timing events are created once per replay thread and reused for every
    // kernel launch on that thread — no per-launch create/destroy overhead.
    // thread_local gives each replay thread its own independent pair.
    thread_local hipEvent_t tl_start = nullptr;
    thread_local hipEvent_t tl_stop  = nullptr;

    bool timing_ok = do_timing;
    if (timing_ok && !tl_start) {
        if (HRR_HIP_CHECK(hipEventCreate(&tl_start)) != hipSuccess ||
            HRR_HIP_CHECK(hipEventCreate(&tl_stop))  != hipSuccess) {
            tl_start = tl_stop = nullptr;
            timing_ok = false;
        } else {
            std::unique_lock lk(ctx.map_mutex);
            ctx.owned_timing_events.push_back(tl_start);
            ctx.owned_timing_events.push_back(tl_stop);
        }
    }

    if (timing_ok)
        timing_ok = (HRR_HIP_CHECK(hipEventRecord(tl_start, stream)) == hipSuccess);

    if (ctx.trace_kernels) {
        fprintf(stderr,
                "[HRR launch] seq=%llu kernel=%zu launch=%s grid=[%u,%u,%u] "
                "block=[%u,%u,%u] shared=%u args=%u snapshots=%u name=\"%s\"\n",
                (unsigned long long)hrr_dispatch_seq,
                kernel_ordinal,
                ext_global_worksize ? "EXT" : "MOD",
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                shared_mem,
                num_args,
                num_snapshots,
                compact_kernel_name(kernel_name).c_str());
        fflush(stderr);
    }

    // Launch the kernel.
    //
    // SP3 assembly kernels (MIOpen Sp3Asm*) directly index the kernarg buffer at
    // known offsets and also read hidden args (hidden_global_offset_x/y/z, etc.)
    // that must be zero. Using kernelParams[] leaves hidden slots uninitialized in
    // the ring-buffer allocator, which causes GPU faults. Instead, build a packed
    // kernarg buffer using the AMDGPU ABI layout rule: each argument is placed at
    // the next offset that is a multiple of its own size (natural alignment).
    // This matches exactly what amd::KernelSignature::at(i).offset_ reports.
    //
    // HIP C++ kernels (clang-compiled) work correctly with kernelParams[]: the
    // runtime handles hidden args internally, so no packed buffer is needed.
    hipError_t r;
    // The extensible descriptor is rebuilt here rather than at parse time so
    // it sees the translated stream and the argument buffer chosen below.
    HIP_LAUNCH_CONFIG ex_cfg{};
    if (launch_ex) {
        ex_cfg.gridDimX = grid[0];  ex_cfg.gridDimY = grid[1];  ex_cfg.gridDimZ = grid[2];
        ex_cfg.blockDimX = block[0]; ex_cfg.blockDimY = block[1]; ex_cfg.blockDimZ = block[2];
        ex_cfg.sharedMemBytes = shared_mem;
        ex_cfg.hStream  = stream;
        ex_cfg.attrs    = launch_attrs.empty() ? nullptr : launch_attrs.data();
        ex_cfg.numAttrs = static_cast<unsigned int>(launch_attrs.size());
    }
    {
        // The cooperative entry point takes no `extra`, so the packed-kernarg
        // route below is not available to it. Cooperative kernels are HIP C++
        // in every case seen here, which is the kernelParams[] path anyway.
        bool is_sp3 = !cooperative &&
                      (kernel_name.find("Sp3") != std::string::npos ||
                       kernel_name.find("sp3") != std::string::npos);
        if (is_sp3 && !arg_ptrs.empty()) {
            // Compute kernarg layout from captured arg sizes using natural alignment.
            // Each arg aligns to its own size (max 8). Hidden args are zero-padded
            // at the end by over-allocating the buffer.
            uint32_t cursor = 0;
            std::vector<uint32_t> koffsets(arg_storage.size());
            for (size_t i = 0; i < arg_storage.size(); ++i) {
                uint32_t sz = static_cast<uint32_t>(arg_storage[i].size());
                uint32_t align = (sz >= 8) ? 8u : sz ? sz : 1u;
                cursor = (cursor + align - 1) & ~(align - 1);
                koffsets[i] = cursor;
                cursor += sz;
            }
            // Round up to 64-byte alignment; add 256 bytes for hidden arg space.
            uint32_t kbuf_sz = ((cursor + 63) & ~63u) + 256;
            std::vector<uint8_t> kbuf(kbuf_sz, 0);
            for (size_t i = 0; i < arg_storage.size(); ++i) {
                const auto& s = arg_storage[i];
                if (koffsets[i] + s.size() <= kbuf_sz)
                    memcpy(kbuf.data() + koffsets[i], s.data(), s.size());
            }
            size_t extra_sz = kbuf_sz;
            void* extra[5] = {
                HIP_LAUNCH_PARAM_BUFFER_POINTER, kbuf.data(),
                HIP_LAUNCH_PARAM_BUFFER_SIZE,    &extra_sz,
                HIP_LAUNCH_PARAM_END
            };
            if (launch_ex) {
                r = hipDrvLaunchKernelEx(&ex_cfg, func, nullptr, extra);
            } else if (ext_global_worksize) {
                // grid[] = global work-item counts: replay through the Ext API.
                r = hipExtModuleLaunchKernel(
                    func,
                    grid[0], grid[1], grid[2],
                    block[0], block[1], block[2],
                    shared_mem, stream,
                    nullptr, extra,
                    nullptr, nullptr, 0);
            } else {
                r = hipModuleLaunchKernel(
                    func,
                    grid[0], grid[1], grid[2],
                    block[0], block[1], block[2],
                    shared_mem, stream,
                    nullptr, extra);
            }
        } else {
            // HIP C++ kernels: kernelParams[] path — runtime handles hidden args.
            if (cooperative) {
                r = hipModuleLaunchCooperativeKernel(
                    func,
                    grid[0], grid[1], grid[2],
                    block[0], block[1], block[2],
                    shared_mem, stream,
                    arg_ptrs.empty() ? nullptr : arg_ptrs.data());
            } else if (launch_ex) {
                r = hipDrvLaunchKernelEx(
                    &ex_cfg, func,
                    arg_ptrs.empty() ? nullptr : arg_ptrs.data(), nullptr);
            } else if (ext_global_worksize) {
                // grid[] = global work-item counts: replay through the Ext API.
                r = hipExtModuleLaunchKernel(
                    func,
                    grid[0], grid[1], grid[2],
                    block[0], block[1], block[2],
                    shared_mem, stream,
                    arg_ptrs.empty() ? nullptr : arg_ptrs.data(),
                    nullptr,
                    nullptr, nullptr, 0);
            } else {
                r = hipModuleLaunchKernel(
                    func,
                    grid[0], grid[1], grid[2],
                    block[0], block[1], block[2],
                    shared_mem, stream,
                    arg_ptrs.empty() ? nullptr : arg_ptrs.data(),
                    nullptr);
            }
        }
    }

    if (timing_ok)
        timing_ok = (HRR_HIP_CHECK(hipEventRecord(tl_stop, stream)) == hipSuccess);

    if (r != hipSuccess) {
        fprintf(stderr, "[HRR] Kernel '%s' launch error: %d (%s) func=%p"
                " grid=[%u,%u,%u] block=[%u,%u,%u]\n",
                kernel_name.c_str(), r, hipGetErrorString(r), (void*)func,
                grid[0], grid[1], grid[2], block[0], block[1], block[2]);
        for (const auto& g : rls.guards) hrr_block_guard_teardown(g);
        return r;
    }

    // Resolve any guarded blocks before the next event runs, so this launch is
    // the only one whose memory was moved.
    if (hipError_t gr = hrr_block_guard_resolve(ctx, rls, kernel_name, kernel_ordinal);
        gr != hipSuccess)
        return gr;

    if (timing_ok)
        timing_ok = (HRR_HIP_CHECK(hipEventSynchronize(tl_stop)) == hipSuccess);
    if (timing_ok) {
        float ms = 0.f;
        if (HRR_HIP_CHECK(hipEventElapsedTime(&ms, tl_start, tl_stop)) == hipSuccess) {
            std::unique_lock lk(ctx.map_mutex);
            ctx.total_kernel_ms += ms;
        }
    }

    // Skip the debug sync while a graph capture is active: hipDeviceSynchronize
    // is illegal during stream capture and invalidates it (HIP 901). The original
    // run never synced here either; syncs resume once capture ends.
    if (ctx.sync_after_launch && !ctx.in_graph_capture) {
        // Clear any pre-existing error before sync so we get a clean error code.
        (void)hipGetLastError();
        if (ctx.trace_sync) {
            fprintf(stderr, "[HRR sync begin] seq=%llu kernel=%zu name=\"%s\"\n",
                    (unsigned long long)hrr_dispatch_seq,
                    kernel_ordinal,
                    compact_kernel_name(kernel_name).c_str());
            fflush(stderr);
        }
        if (ctx.sync_watchdog_ms) {
            char wd_what[512];
            snprintf(wd_what, sizeof(wd_what),
                     "kernel #%zu \"%s\" (seq=%llu, grid=[%u,%u,%u] block=[%u,%u,%u])",
                     kernel_ordinal, compact_kernel_name(kernel_name).c_str(),
                     (unsigned long long)hrr_dispatch_seq,
                     grid[0], grid[1], grid[2], block[0], block[1], block[2]);
            r = hrr_watchdog_device_sync(ctx, wd_what);
        } else {
            r = hipDeviceSynchronize();
        }
        hipError_t last_r = hipGetLastError();
        if (r == hipSuccess && last_r != hipSuccess) r = last_r;
        if (r != hipSuccess)
            fprintf(stderr, "[HRR] GPU error after '%s': %d (%s) last=%d (%s)\n",
                    kernel_name.c_str(), r, hipGetErrorString(r),
                    (int)last_r, hipGetErrorString(last_r));
        else if (ctx.trace_sync) {
            fprintf(stderr, "[HRR sync done] seq=%llu kernel=%zu status=success\n",
                    (unsigned long long)hrr_dispatch_seq,
                    kernel_ordinal);
            fflush(stderr);
        }
        else if (ctx.verbose)
            fprintf(stderr, "[HRR] Kernel '%s' OK\n", kernel_name.c_str());
    }

    const size_t completed_kernel =
        ctx.kernels_launched.fetch_add(1, std::memory_order_relaxed) + 1;
    maybe_trace_progress(ctx, completed_kernel, kernel_name);
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: kernel launches (all four variants share the same payload)
// ---------------------------------------------------------------------------

hipError_t playback_hipModuleLaunchKernel(PlaybackContext& ctx,
                                          const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload);
}

hipError_t playback_hipExtModuleLaunchKernel(PlaybackContext& ctx,
                                             const uint8_t* payload) {
    // hipExtModuleLaunchKernel is captured with HSA/OpenCL semantics: the grid[]
    // dims are *global work-item counts*, not workgroup counts. Replay through
    // the matching API so the grid is not over-launched by a factor of blockDim.
    return replay_kernel_launch(ctx, payload, /*ext_global_worksize=*/true);
}

hipError_t playback_hipLaunchKernel(PlaybackContext& ctx,
                                    const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload);
}

hipError_t playback_hipLaunchByPtr(PlaybackContext& ctx,
                                   const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload);
}

hipError_t playback_hipLaunchKernel_spt(PlaybackContext& ctx,
                                        const uint8_t* payload) {
    // Which stream the launch goes on is recorded in the payload, so the
    // stream-per-thread spelling replays through the same path as the plain
    // one. Only the event id differs.
    return replay_kernel_launch(ctx, payload);
}

hipError_t playback_hipLaunchCooperativeKernel(PlaybackContext& ctx,
                                               const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload, /*ext_global_worksize=*/false,
                                /*launch_ex=*/false, /*cooperative=*/true);
}

hipError_t playback_hipLaunchCooperativeKernel_spt(PlaybackContext& ctx,
                                                   const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload, /*ext_global_worksize=*/false,
                                /*launch_ex=*/false, /*cooperative=*/true);
}

hipError_t playback_hipDrvLaunchKernelEx(PlaybackContext& ctx,
                                         const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload, /*ext_global_worksize=*/false,
                                /*launch_ex=*/true);
}

hipError_t playback_hipLaunchKernelExC(PlaybackContext& ctx,
                                       const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload, /*ext_global_worksize=*/false,
                                /*launch_ex=*/true);
}

hipError_t playback_hipModuleLaunchCooperativeKernel(PlaybackContext& ctx,
                                                     const uint8_t* payload) {
    // A cooperative launch has to go back through the cooperative entry point:
    // it is what reserves the whole grid as co-resident, and a grid-wide
    // barrier replayed through the ordinary launch hangs instead of failing.
    return replay_kernel_launch(ctx, payload, /*ext_global_worksize=*/false,
                                /*launch_ex=*/false, /*cooperative=*/true);
}

// ---------------------------------------------------------------------------
// Manual playback: __hipRegisterFatBinary
// ---------------------------------------------------------------------------
// Load the fat binary blob via hipModuleLoadData so all embedded kernel names
// become resolvable at kernel launch replay time.
// Stored in co_modules keyed by the full 32-char hex hash — collision-free
// and consistent with load_module(), so kernel name scans find it automatically.

hipError_t playback___hipRegisterFatBinary(PlaybackContext& ctx,
                                           const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args___hipRegisterFatBinary*>(payload);
    uint64_t blob_hash_lo = a->blob_hash_lo;
    uint64_t blob_hash_hi = a->blob_hash_hi;

    if (!blob_hash_lo && !blob_hash_hi) return hipSuccess;  // no blob — skip
    if (!a->blob_size) return hipSuccess;

    std::string hex = hrr::hash_hex(blob_hash_lo, blob_hash_hi);

    // Deduplicate: if already loaded (e.g. multiple __hipRegisterFatBinary events
    // for the same binary), skip the load.
    {
        std::shared_lock lk(ctx.map_mutex);
        if (ctx.co_modules.count(hex)) return hipSuccess;
    }

    size_t sz = 0;
    const void* blob = ctx.load_blob(blob_hash_lo, blob_hash_hi, &sz);
    if (!blob || sz == 0) {
        fprintf(stderr, "[HRR] __hipRegisterFatBinary: blob not found in archive\n");
        return hipSuccess;  // non-fatal — kernels will fail at launch but don't abort
    }

    hipModule_t mod = nullptr;
    hipError_t err = hipModuleLoadData(&mod, blob);
    if (err != hipSuccess) {
        fprintf(stderr, "[HRR] __hipRegisterFatBinary: hipModuleLoadData failed: %d (%s)\n",
                err, hipGetErrorString(err));
        return hipSuccess;  // non-fatal
    }

    {
        std::unique_lock lk(ctx.map_mutex);
        auto [it, inserted] = ctx.co_modules.emplace(hex, mod);
        if (!inserted) {
            // Another thread raced us between the shared_lock check and here — discard ours.
            (void)hipModuleUnload(mod);
        }
    }
    if (ctx.verbose)
        fprintf(stderr, "[HRR] Loaded fat binary blob (%zu bytes) -> hipModule_t\n", sz);
    return hipSuccess;
}

// ---------------------------------------------------------------------------
// Symbol resolution
// ---------------------------------------------------------------------------
// A __device__ global is named in the recording by the host shadow address the
// compiler emitted, which means nothing here. The code object carrying the
// global is loaded, though, so the name recorded beside that address resolves
// it — the same lazy name lookup kernel launches already use for functions.

void* PlaybackContext::resolve_symbol_by_name(const char* name,
                                              size_t* sz_out) const {
    if (!name || !*name) return nullptr;
    std::shared_lock lk(map_mutex);
    for (const auto& [hex, mod] : co_modules) {
        (void)hex;
        hipDeviceptr_t dptr = nullptr;
        size_t bytes = 0;
        if (hipModuleGetGlobal(&dptr, &bytes, mod, name) == hipSuccess && dptr) {
            if (sz_out) *sz_out = bytes;
            return dptr;
        }
    }
    for (const auto& [rec_mod, mod] : module_map) {
        (void)rec_mod;
        hipDeviceptr_t dptr = nullptr;
        size_t bytes = 0;
        if (hipModuleGetGlobal(&dptr, &bytes, mod, name) == hipSuccess && dptr) {
            if (sz_out) *sz_out = bytes;
            return dptr;
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Manual playback: __hipRegisterVar
// ---------------------------------------------------------------------------
// Registrations fire at the capturing process's static-init time, before the
// capture shims are live, so what the archive holds is the post-registration
// sweep hip_capture_init() writes: one event per __device__ global, carrying
// its name, its host shadow address and its capture-time device address.
//
// Resolving the name here and recording both mappings is what makes the symbol
// family replayable. The device-address mapping matters most: hipMemcpyToSymbol
// and friends are recorded as an inner hipMemcpy against the symbol's device
// address, and without this that source translated to nothing.

hipError_t playback___hipRegisterVar(PlaybackContext& ctx,
                                     const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args___hipRegisterVar*>(payload);
    if (!a->deviceVar_present || a->deviceVar_bytes[0] == '\0') return hipSuccess;

    const char* name = reinterpret_cast<const char*>(a->deviceVar_bytes);
    size_t live_size = 0;
    void* live = ctx.resolve_symbol_by_name(name, &live_size);
    if (!live) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr,
                    "[HRR] __hipRegisterVar: symbol '%s' is not in any module "
                    "this replay loaded, so copies naming it will fail rather "
                    "than write somewhere else.\n", name);
        }
        return hipSuccess;
    }

    if (live_size == 0) live_size = static_cast<size_t>(a->size);
    ctx.record_symbol(a->var, name, live, live_size);
    if (a->dev_addr)
        ctx.record_alloc(a->dev_addr, live, live_size);
    if (ctx.verbose)
        fprintf(stderr, "[HRR] symbol '%s': 0x%llx -> %p (%zu bytes)\n", name,
                (unsigned long long)a->dev_addr, live, live_size);
    return hipSuccess;
}

// ---------------------------------------------------------------------------
// Manual playback: hipGetSymbolAddress / hipGetSymbolSize
// ---------------------------------------------------------------------------
// Both take the host shadow address, which the symbol sweep has already tied
// to a live global. Answering from that map also registers the recorded device
// address the capturing process got back, so a later copy against it resolves.

hipError_t playback_hipGetSymbolAddress(PlaybackContext& ctx,
                                        const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args_hipGetSymbolAddress*>(payload);
    size_t sz = 0;
    void* live = ctx.translate_symbol(a->symbol, &sz);
    if (!live) {
        fprintf(stderr,
                "[HRR] hipGetSymbolAddress: symbol 0x%llx was never registered "
                "in this archive.\n", (unsigned long long)a->symbol);
        return hipErrorInvalidSymbol;
    }
    if (a->devPtr) ctx.record_alloc(a->devPtr, live, sz);
    return hipSuccess;
}

hipError_t playback_hipGetSymbolSize(PlaybackContext& ctx,
                                     const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args_hipGetSymbolSize*>(payload);
    size_t sz = 0;
    if (!ctx.translate_symbol(a->symbol, &sz)) {
        fprintf(stderr,
                "[HRR] hipGetSymbolSize: symbol 0x%llx was never registered in "
                "this archive.\n", (unsigned long long)a->symbol);
        return hipErrorInvalidSymbol;
    }
    return hipSuccess;
}

// ---------------------------------------------------------------------------
// Manual playback: hipGraphAddMemcpyNodeToSymbol / FromSymbol
// ---------------------------------------------------------------------------
// The node is built against the live global the symbol registry resolved. The
// host side of the copy is the recorded blob for the to-symbol spelling and a
// context-owned landing buffer for the from-symbol one, which has to outlive
// this call: the copy happens when the graph is launched, not here.
//
// The node is added through the 1D spelling rather than the symbol one. The
// symbol entry points want the host shadow address the compiler emitted for
// the global, and a replay has no such shadow — it loaded the code object
// through hipModuleLoadData. Passing the device address instead is rejected
// with hipErrorInvalidDeviceSymbol, so the copy is expressed directly against
// the resolved device address, which is what the symbol spelling decays to
// inside the runtime anyway.

static hipGraphNode_t* dep_array(PlaybackContext& ctx, const uint8_t* bytes,
                                 uint8_t present, uint32_t n,
                                 std::vector<hipGraphNode_t>& out) {
    if (!present || n == 0) return nullptr;
    out.resize(n);
    std::memcpy(out.data(), bytes, n * sizeof(hipGraphNode_t));
    for (auto& node : out)
        node = ctx.translate_graph_node(reinterpret_cast<uint64_t>(node));
    return out.data();
}

hipError_t playback_hipGraphAddMemcpyNodeToSymbol(PlaybackContext& ctx,
                                                  const uint8_t* payload) {
    const auto* a =
        reinterpret_cast<const hrr_args_hipGraphAddMemcpyNodeToSymbol*>(payload);
    const auto kind = static_cast<hipMemcpyKind>(a->kind);
    void* symbol = ctx.translate_symbol(a->symbol);
    if (!symbol) {
        fprintf(stderr,
                "[HRR] hipGraphAddMemcpyNodeToSymbol: symbol 0x%llx is not in "
                "this archive's symbol registry.\n",
                (unsigned long long)a->symbol);
        return hipErrorInvalidSymbol;
    }

    const void* src = nullptr;
    if (a->blob_hash_lo || a->blob_hash_hi) {
        size_t blob_sz = 0;
        src = ctx.load_blob(a->blob_hash_lo, a->blob_hash_hi, &blob_sz);
        if (!src || blob_sz < a->count) {
            fprintf(stderr,
                    "[HRR] hipGraphAddMemcpyNodeToSymbol: host source blob "
                    "missing from the archive.\n");
            return hipErrorInvalidValue;
        }
    } else {
        src = ctx.translate_ptr(a->src);
        if (!src) {
            fprintf(stderr,
                    "[HRR] hipGraphAddMemcpyNodeToSymbol: device source 0x%llx "
                    "is not mapped.\n", (unsigned long long)a->src);
            return hipErrorInvalidValue;
        }
    }

    std::vector<hipGraphNode_t> deps;
    hipGraphNode_t* dep_ptr = dep_array(ctx, a->pDependencies_bytes,
                                        a->pDependencies_present,
                                        a->pDependencies_n, deps);
    hipGraphNode_t node = nullptr;
    hipError_t r = hipGraphAddMemcpyNode1D(
        &node, ctx.translate_graph(a->graph), dep_ptr, deps.size(),
        static_cast<char*>(symbol) + a->offset, src,
        static_cast<size_t>(a->count), kind);
    if (r == hipSuccess && a->pGraphNode)
        ctx.record_graph_node(a->pGraphNode, node);
    return r;
}

hipError_t playback_hipGraphAddMemcpyNodeFromSymbol(PlaybackContext& ctx,
                                                    const uint8_t* payload) {
    const auto* a =
        reinterpret_cast<const hrr_args_hipGraphAddMemcpyNodeFromSymbol*>(payload);
    const auto kind = static_cast<hipMemcpyKind>(a->kind);
    void* symbol = ctx.translate_symbol(a->symbol);
    if (!symbol) {
        fprintf(stderr,
                "[HRR] hipGraphAddMemcpyNodeFromSymbol: symbol 0x%llx is not in "
                "this archive's symbol registry.\n",
                (unsigned long long)a->symbol);
        return hipErrorInvalidSymbol;
    }

    void* dst = nullptr;
    if (kind == hipMemcpyDeviceToHost || kind == hipMemcpyHostToHost) {
        dst = ctx.host_landing_buffer(a->dst, static_cast<size_t>(a->count));
    } else {
        dst = ctx.translate_ptr(a->dst);
        if (!dst) {
            fprintf(stderr,
                    "[HRR] hipGraphAddMemcpyNodeFromSymbol: device destination "
                    "0x%llx is not mapped.\n", (unsigned long long)a->dst);
            return hipErrorInvalidValue;
        }
    }

    std::vector<hipGraphNode_t> deps;
    hipGraphNode_t* dep_ptr = dep_array(ctx, a->pDependencies_bytes,
                                        a->pDependencies_present,
                                        a->pDependencies_n, deps);
    hipGraphNode_t node = nullptr;
    hipError_t r = hipGraphAddMemcpyNode1D(
        &node, ctx.translate_graph(a->graph), dep_ptr, deps.size(), dst,
        static_cast<char*>(symbol) + a->offset, static_cast<size_t>(a->count),
        kind);
    if (r == hipSuccess && a->pGraphNode)
        ctx.record_graph_node(a->pGraphNode, node);
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: the four *MemcpyNodeSetParams*Symbol spellings
// ---------------------------------------------------------------------------
// Same substitution as the two construction handlers above, for the same
// reason: the symbol entry points need a host shadow address, so the mutation
// is expressed against the resolved device address through the 1D spelling.

namespace {

// The host source for a to-symbol mutation: the recorded blob when the copy
// came from host memory, the translated device address otherwise.
template <typename A>
static const void* to_symbol_source(PlaybackContext& ctx, const A* a,
                                    const char* api) {
    if (a->blob_hash_lo || a->blob_hash_hi) {
        size_t blob_sz = 0;
        const void* blob = ctx.load_blob(a->blob_hash_lo, a->blob_hash_hi,
                                         &blob_sz);
        if (!blob || blob_sz < a->count) {
            fprintf(stderr, "[HRR] %s: host source blob missing from the "
                    "archive.\n", api);
            return nullptr;
        }
        return blob;
    }
    const void* src = ctx.translate_ptr(a->src);
    if (!src)
        fprintf(stderr, "[HRR] %s: device source 0x%llx is not mapped.\n", api,
                (unsigned long long)a->src);
    return src;
}

static void* symbol_or_complain(PlaybackContext& ctx, uint64_t rec,
                                const char* api) {
    void* symbol = ctx.translate_symbol(rec);
    if (!symbol)
        fprintf(stderr, "[HRR] %s: symbol 0x%llx is not in this archive's "
                "symbol registry.\n", api, (unsigned long long)rec);
    return symbol;
}

}  // namespace

hipError_t playback_hipGraphMemcpyNodeSetParamsToSymbol(PlaybackContext& ctx,
                                                        const uint8_t* payload) {
    const char* kApi = "hipGraphMemcpyNodeSetParamsToSymbol";
    const auto* a =
        reinterpret_cast<const hrr_args_hipGraphMemcpyNodeSetParamsToSymbol*>(payload);
    void* symbol = symbol_or_complain(ctx, a->symbol, kApi);
    if (!symbol) return hipErrorInvalidSymbol;
    const void* src = to_symbol_source(ctx, a, kApi);
    if (!src) return hipErrorInvalidValue;
    hipGraphNode_t node = ctx.translate_graph_node(a->node);
    if (!node) {
        fprintf(stderr, "[HRR] %s: node 0x%llx was never built at replay.\n",
                kApi, (unsigned long long)a->node);
        return hipErrorInvalidValue;
    }
    return hipGraphMemcpyNodeSetParams1D(
        node, static_cast<char*>(symbol) + a->offset, src,
        static_cast<size_t>(a->count), static_cast<hipMemcpyKind>(a->kind));
}

hipError_t playback_hipGraphMemcpyNodeSetParamsFromSymbol(PlaybackContext& ctx,
                                                          const uint8_t* payload) {
    const char* kApi = "hipGraphMemcpyNodeSetParamsFromSymbol";
    const auto* a =
        reinterpret_cast<const hrr_args_hipGraphMemcpyNodeSetParamsFromSymbol*>(payload);
    void* symbol = symbol_or_complain(ctx, a->symbol, kApi);
    if (!symbol) return hipErrorInvalidSymbol;
    hipGraphNode_t node = ctx.translate_graph_node(a->node);
    if (!node) {
        fprintf(stderr, "[HRR] %s: node 0x%llx was never built at replay.\n",
                kApi, (unsigned long long)a->node);
        return hipErrorInvalidValue;
    }
    const auto kind = static_cast<hipMemcpyKind>(a->kind);
    void* dst = (kind == hipMemcpyDeviceToHost || kind == hipMemcpyHostToHost)
                    ? ctx.host_landing_buffer(a->dst, static_cast<size_t>(a->count))
                    : ctx.translate_ptr(a->dst);
    if (!dst) {
        fprintf(stderr, "[HRR] %s: destination 0x%llx is not mapped.\n", kApi,
                (unsigned long long)a->dst);
        return hipErrorInvalidValue;
    }
    return hipGraphMemcpyNodeSetParams1D(
        node, dst, static_cast<char*>(symbol) + a->offset,
        static_cast<size_t>(a->count), kind);
}

hipError_t playback_hipGraphExecMemcpyNodeSetParamsToSymbol(
    PlaybackContext& ctx, const uint8_t* payload) {
    const char* kApi = "hipGraphExecMemcpyNodeSetParamsToSymbol";
    const auto* a =
        reinterpret_cast<const hrr_args_hipGraphExecMemcpyNodeSetParamsToSymbol*>(payload);
    void* symbol = symbol_or_complain(ctx, a->symbol, kApi);
    if (!symbol) return hipErrorInvalidSymbol;
    const void* src = to_symbol_source(ctx, a, kApi);
    if (!src) return hipErrorInvalidValue;
    hipGraphExec_t exec = ctx.translate_graph_exec(a->hGraphExec);
    hipGraphNode_t node = ctx.translate_graph_node(a->node);
    if (!exec || !node) {
        fprintf(stderr, "[HRR] %s: exec 0x%llx / node 0x%llx was never built "
                "at replay.\n", kApi, (unsigned long long)a->hGraphExec,
                (unsigned long long)a->node);
        return hipErrorInvalidValue;
    }
    return hipGraphExecMemcpyNodeSetParams1D(
        exec, node, static_cast<char*>(symbol) + a->offset, src,
        static_cast<size_t>(a->count), static_cast<hipMemcpyKind>(a->kind));
}

hipError_t playback_hipGraphExecMemcpyNodeSetParamsFromSymbol(
    PlaybackContext& ctx, const uint8_t* payload) {
    const char* kApi = "hipGraphExecMemcpyNodeSetParamsFromSymbol";
    const auto* a =
        reinterpret_cast<const hrr_args_hipGraphExecMemcpyNodeSetParamsFromSymbol*>(payload);
    void* symbol = symbol_or_complain(ctx, a->symbol, kApi);
    if (!symbol) return hipErrorInvalidSymbol;
    hipGraphExec_t exec = ctx.translate_graph_exec(a->hGraphExec);
    hipGraphNode_t node = ctx.translate_graph_node(a->node);
    if (!exec || !node) {
        fprintf(stderr, "[HRR] %s: exec 0x%llx / node 0x%llx was never built "
                "at replay.\n", kApi, (unsigned long long)a->hGraphExec,
                (unsigned long long)a->node);
        return hipErrorInvalidValue;
    }
    const auto kind = static_cast<hipMemcpyKind>(a->kind);
    void* dst = (kind == hipMemcpyDeviceToHost || kind == hipMemcpyHostToHost)
                    ? ctx.host_landing_buffer(a->dst, static_cast<size_t>(a->count))
                    : ctx.translate_ptr(a->dst);
    if (!dst) {
        fprintf(stderr, "[HRR] %s: destination 0x%llx is not mapped.\n", kApi,
                (unsigned long long)a->dst);
        return hipErrorInvalidValue;
    }
    return hipGraphExecMemcpyNodeSetParams1D(
        exec, node, dst, static_cast<char*>(symbol) + a->offset,
        static_cast<size_t>(a->count), kind);
}

// ---------------------------------------------------------------------------
// Manual playback: hipLinkAddData
// ---------------------------------------------------------------------------
// The image comes back from its blob and the linker state from the map
// hipLinkCreate fills. Option values are replayed as recorded: an option whose
// value is a pointer into the capturing process cannot be reconstructed, and
// the link then fails here rather than producing a silently different binary.

hipError_t playback_hipLinkAddData(PlaybackContext& ctx,
                                   const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args_hipLinkAddData*>(payload);
    hipLinkState_t state = ctx.translate_link_state(a->state);
    if (!state) {
        fprintf(stderr,
                "[HRR] hipLinkAddData: linker state 0x%llx was never created "
                "at replay.\n", (unsigned long long)a->state);
        return hipErrorInvalidValue;
    }

    size_t blob_sz = 0;
    const void* image = (a->blob_hash_lo || a->blob_hash_hi)
                            ? ctx.load_blob(a->blob_hash_lo, a->blob_hash_hi,
                                            &blob_sz)
                            : nullptr;
    if (!image) {
        fprintf(stderr,
                "[HRR] hipLinkAddData: the linker input image is not in this "
                "archive, so the link cannot be reproduced.\n");
        return hipErrorInvalidValue;
    }

    hipJitOption options[32]{};
    void* option_values[32]{};
    uint32_t n_opts = a->options_n > 32u ? 32u : a->options_n;
    if (a->options_present)
        std::memcpy(options, a->options_bytes, n_opts * sizeof(hipJitOption));
    if (a->optionValues_present)
        std::memcpy(option_values, a->optionValues_bytes,
                    n_opts * sizeof(void*));

    hipError_t r = hipLinkAddData(
        state, static_cast<hipJitInputType>(a->type), const_cast<void*>(image),
        blob_sz,
        a->name_present ? reinterpret_cast<const char*>(a->name_bytes)
                        : nullptr,
        n_opts, n_opts ? options : nullptr, n_opts ? option_values : nullptr);
    if (hrr_replayed_recorded_error(ctx, "hipLinkAddData", a->ret, r))
        return hipSuccess;
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipModuleGetFunction
// ---------------------------------------------------------------------------
// Intentional no-op at playback.
//
// Function handles are resolved lazily by name at kernel launch time:
// replay_kernel_launch() searches module_map + co_modules + func_cache.
// Recording a handle map here would require translating the recorded module
// handle to a live hipModule_t at the time of this call, which is fragile;
// the lazy lookup at launch is simpler and more robust.
//
// LIMITATION: if the captured application calls hipFuncGetAttributes or
// similar APIs on the returned function handle, those calls will silently
// receive a null handle and may fail or no-op during playback.

hipError_t playback_hipModuleGetFunction(PlaybackContext& ctx,
                                         const uint8_t* payload) {
    (void)ctx; (void)payload;
    return hipSuccess;
}

// ---------------------------------------------------------------------------
// Manual playback: hipModuleLoadData / hipModuleLoadDataEx / hipModuleLoad
// ---------------------------------------------------------------------------
// Payload layout (after 32-byte EventHeader):
//   hipModuleLoadData / hipModuleLoadDataEx:
//     ret(4) module(8) image(8) co_hash_lo(8) co_hash_hi(8) [module_id(4)]
//   hipModuleLoad:
//     ret(4) module(8) fname(8) co_hash_lo(8) co_hash_hi(8) [module_id(4)]
//
// The recorded module handle is at offset +4 (8 bytes).
// co_hash_lo is at offset +20 (8 bytes), co_hash_hi at +28 (8 bytes).

static hipError_t replay_module_load(PlaybackContext& ctx,
                                     const uint8_t* payload) {
    // hipModuleLoad and hipModuleLoadData share the same layout for the fields we need
    const auto* a = reinterpret_cast<const hrr_args_hipModuleLoadData*>(payload);
    uint64_t rec_module = a->module;
    uint64_t co_hash_lo = a->co_hash_lo;
    uint64_t co_hash_hi = a->co_hash_hi;

    if (!co_hash_lo && !co_hash_hi) {
        fprintf(stderr, "[HRR] hipModuleLoad: no code object hash in payload\n");
        return hipErrorInvalidValue;
    }

    hipModule_t mod = ctx.load_module(co_hash_lo, co_hash_hi);
    if (!mod) return hipErrorSharedObjectInitFailed;

    ctx.record_module(rec_module, mod);
    return hipSuccess;
}

hipError_t playback_hipModuleLoadData(PlaybackContext& ctx,
                                      const uint8_t* payload) {
    return replay_module_load(ctx, payload);
}

hipError_t playback_hipModuleLoadDataEx(PlaybackContext& ctx,
                                        const uint8_t* payload) {
    // hipModuleLoadDataEx has extra fields (numOptions/options/optionValues)
    // before co_hash — use the correct struct type.
    const auto* a = reinterpret_cast<const hrr_args_hipModuleLoadDataEx*>(payload);
    uint64_t rec_module = a->module;
    uint64_t co_hash_lo = a->co_hash_lo;
    uint64_t co_hash_hi = a->co_hash_hi;

    if (!co_hash_lo && !co_hash_hi) {
        fprintf(stderr, "[HRR] hipModuleLoadDataEx: no code object hash in payload\n");
        return hipErrorInvalidValue;
    }

    hipModule_t mod = ctx.load_module(co_hash_lo, co_hash_hi);
    if (!mod) return hipErrorSharedObjectInitFailed;

    ctx.record_module(rec_module, mod);
    return hipSuccess;
}

hipError_t playback_hipModuleLoad(PlaybackContext& ctx,
                                  const uint8_t* payload) {
    return replay_module_load(ctx, payload);
}

// ---------------------------------------------------------------------------
// Manual playback: hipMalloc / hipMallocManaged / hipHostMalloc
// ---------------------------------------------------------------------------
// Payload: ret(4) ptr(8) size(8) [additional fields for managed/host variants]
// ptr at +4, size at +12

// GPU allocation padding multiplier for replay.
//
// MIOpen / ROCBlas kernels are often launched with grids larger than the
// batch size (e.g. grid[0] = batch * n_groups with n_groups=96; if batch=1
// but grid is 256×n_groups, the kernel's s88 = blockIdx.x/n_groups sweeps
// 256 "virtual batches" and accesses memory at up to 256× the single-batch
// tensor size).  In the original application, a framework memory pool
// allocates GPU VA contiguously so the adjacent pages are all mapped;
// replay hipMallocs are independent and the adjacent pages are unmapped,
// causing GPU page faults.
//
// Fix (optional): over-allocate by HIP_HRR_REPLAY_ALLOC_PAD_FACTOR (legacy default
// was 256, capped per allocation) so pool-style kernels have headroom.  Default
// factor is now **1** (exact recorded sizes) so large captures replay without
// multiplying VRAM; set HIP_HRR_REPLAY_ALLOC_PAD_FACTOR=256 for MIOpen-style
// workloads that may fault without padding.  The extra memory is zero-initialized
// when factor > 1.
//
// SP3AsmConv stride2 on 64×112×112 input sweeps 256 virtual batches:
//   256 × 64 × 112 × 112 × 4 = ~781 MB from in_ptr.
// A 1 GB cap ensures any sub-allocation has ≥800 MB headroom.
// With 46 GB GPU and ≤30 pool allocations: 30 × 1 GB = 30 GB — within budget.
//
// With factor 256 and a 1 GiB cap, many medium allocs each replay as 1 GiB, so
// cumulative VRAM can exceed HBM early on large LLM captures; factor **1** avoids that.
// Tunables: HIP_HRR_REPLAY_ALLOC_PAD_FACTOR (default **1**) and
// HIP_HRR_REPLAY_ALLOC_PAD_MAX (default 1073741824).
static void hrr_replay_alloc_pad_params(size_t* factor_out, size_t* max_out) {
    static std::once_flag once;
    static constexpr size_t kDefaultFactor = 1;
    static constexpr size_t kDefaultMax   = 1ULL * 1024 * 1024 * 1024;
    static size_t g_factor = kDefaultFactor;
    static size_t g_max    = kDefaultMax;
    std::call_once(once, [] {
        if (const char* e = std::getenv("HIP_HRR_REPLAY_ALLOC_PAD_FACTOR")) {
            char* end = nullptr;
            unsigned long v = std::strtoul(e, &end, 0);
            if (end != e) {
                if (v <= 1)
                    g_factor = 1;
                else if (v <= 4096)
                    g_factor = static_cast<size_t>(v);
            }
        }
        if (const char* e = std::getenv("HIP_HRR_REPLAY_ALLOC_PAD_MAX")) {
            char* end = nullptr;
            unsigned long v = std::strtoul(e, &end, 0);
            if (end != e && v > 0) g_max = static_cast<size_t>(v);
        }
        if (g_factor != kDefaultFactor || g_max != kDefaultMax) {
            fprintf(stderr,
                    "[HRR] replay alloc pad: factor=%zu max_bytes=%zu "
                    "(HIP_HRR_REPLAY_ALLOC_PAD_* env)\n",
                    g_factor, g_max);
        }
    });
    *factor_out = g_factor;
    *max_out    = g_max;
}

static size_t replay_padded_alloc_size(size_t orig_sz) {
    size_t fac, mx;
    hrr_replay_alloc_pad_params(&fac, &mx);
    size_t pad_sz = std::min(orig_sz * fac, mx);
    return std::max(orig_sz, pad_sz);
}

// Zero-initialise freshly allocated replay device memory.
//
// hipMalloc/hipMallocAsync do NOT guarantee zeroed memory: AMD only scrubs a
// physical page on its FIRST allocation (for cross-process security). Memory
// that is reused within the process (vLLM allocates/frees constantly) comes
// back holding stale bytes from a previous replay allocation. Any kernel that
// reads a region the recorded stream never explicitly wrote — e.g. a workload
// that implicitly relies on first-touch-zeroed memory, or a reduction/argmax
// scratch buffer — then sees run-to-run-varying garbage. That nondeterministic
// divergence cascades (a flipped argmax tie -> a different token -> a different
// block table -> a slot-mapping kernel writing out of bounds), surfacing as the
// intermittent "_compute_slot_mapping_kernel" memory fault at a low address.
//
// Zeroing makes replay deterministic and matches the first-touch-zeroed
// semantics these workloads implicitly assume. Default on; set
// HIP_HRR_REPLAY_ZERO_INIT=0 to skip it (faster, but reintroduces the garbage).
static bool hrr_replay_zero_init() {
    static std::once_flag once;
    static bool g_enabled = true;
    std::call_once(once, [] {
        if (const char* e = std::getenv("HIP_HRR_REPLAY_ZERO_INIT")) {
            if (e[0] == '0' && e[1] == '\0') {
                g_enabled = false;
                fprintf(stderr, "[HRR] replay zero-init DISABLED "
                                "(HIP_HRR_REPLAY_ZERO_INIT=0)\n");
            }
        }
    });
    return g_enabled;
}

// Zero-initialise a host-synchronous replay allocation, ordered.
//
// ROCM-27985. hipMalloc is host-synchronous by contract, so the recorded
// program is free to use the returned pointer immediately from any stream
// without establishing an ordering edge. The zero-init injected here must
// therefore be complete before the allocation handler returns.
//
// A bare hipMemset does not give that. ihipMemset() promotes a memset on a
// fresh, non-offset device allocation to asynchronous ("spec says hipMemset
// will be asynchronous when destination memory is device memory and pointer is
// non-offseted"), so it is only enqueued on the null stream and the host
// returns immediately. Streams the capture created with hipStreamNonBlocking do
// not synchronize with the null stream, so a later replayed H2D restore or
// kernel launch on such a stream races the zero-init. When the zero-init lands
// last it overwrites the restored input with zeros and the consuming kernel
// computes from zeros, which surfaces downstream as a replay D2H validation
// mismatch against the captured output.
//
// Draining after the H2D restore (hrr_sync_after_replayed_h2d) does not fix
// this: it waits for both operations to finish but does not order them. The
// ordering edge has to be established here, at the allocation.
static void hrr_zero_init_alloc(PlaybackContext& ctx, void* live, size_t sz) {
    if (!live || sz == 0) return;  // nothing written, so nothing to order
    if (!hrr_zero_init_needs_drain(hrr_replay_zero_init(), ctx.in_graph_capture))
        return;
    if (hipMemsetAsync(live, 0, sz, nullptr) != hipSuccess) return;
    (void)hipStreamSynchronize(nullptr);
}

// ---- External region materialisation ----------------------------------------

// Make `device` current and return the ordinal to restore afterwards, or -1 if
// nothing was switched. An ordinal this system does not have is reported once
// and ignored rather than failing the replay: a capture taken on an eight-GPU
// node is still worth replaying on a smaller one, just not with the segment on
// the device the producer named.
static int hrr_set_region_device(int device) {
    int count = 0;
    if (device < 0 || hipGetDeviceCount(&count) != hipSuccess) return -1;
    if (device >= count) {
        static std::once_flag once;
        std::call_once(once, [&] {
            fprintf(stderr,
                    "[HRR] regions: annotation names device %d but this system "
                    "has %d — materialising on the current device instead\n",
                    device, count);
        });
        return -1;
    }
    int cur = 0;
    if (hipGetDevice(&cur) != hipSuccess || cur == device) return -1;
    if (HRR_HIP_CHECK(hipSetDevice(device)) != hipSuccess) return -1;
    return cur;
}

// A segment a producer declared but that HRR never observed. Everything the
// archive knows about it is its base and size: no capture shim ran, so no
// contents were recorded and no allocation event exists. Replay gives it a
// buffer of the right size so pointers into it translate — a kernel then reads
// fill bytes instead of faulting on null, which is a diagnosable wrong answer
// rather than a crash that says only that something was missing.
hipError_t hrr_materialize_region(PlaybackContext& ctx, uint64_t rec_base,
                                  size_t size, int device, void** out_live) {
    if (out_live) *out_live = nullptr;
    if (rec_base == 0 || size == 0) return hipErrorInvalidValue;

    // Place the buffer on the device the producer said the segment was on.
    // Leaving it on whatever device happens to be current would put it on the
    // wrong GPU as soon as the recorded program used more than one: the segment
    // is materialised by the first pointer that fails to translate, so the
    // placement would be decided by whichever kernel touched it first rather
    // than by where the memory actually lived.
    const int prev = hrr_set_region_device(device);

    void* live = nullptr;
    hipError_t r = HRR_HIP_CHECK(hipMalloc(&live, size));
    if (r != hipSuccess) {
        if (prev >= 0) (void)hipSetDevice(prev);
        fprintf(stderr,
                "[HRR] regions: could not materialise segment 0x%llx (%zu bytes) "
                "on device %d: %s\n",
                static_cast<unsigned long long>(rec_base), size, device,
                hipGetErrorString(r));
        return r;
    }
    // Still on the segment's device: the fill runs on the null stream of the
    // current device, which has to be the one owning the buffer.
    hrr_zero_init_alloc(ctx, live, size);
    if (prev >= 0) (void)hipSetDevice(prev);

    ctx.record_alloc(rec_base, live, size);
    if (out_live) *out_live = live;

    static std::once_flag once;
    std::call_once(once, [] {
        fprintf(stderr,
                "[HRR] regions: materialising segments that bypassed the HIP API "
                "— their contents are not in the archive and are filled with the "
                "replay fill byte (HIP_HRR_REPLAY_FILL_BYTE)\n");
    });
    return hipSuccess;
}

void hrr_release_region(PlaybackContext& ctx, uint64_t rec_base, void* live) {
    ctx.remove_alloc(rec_base);
    if (live) (void)hipFree(live);
}

// ---- Guard pages ------------------------------------------------------------
//
// Replay allocates each recorded segment as one contiguous buffer, exactly as
// the recording did, so an access that ran off the end of an object inside that
// segment is in bounds as far as the hardware is concerned and lands in
// whatever the allocator put next. Guard pages convert that silence into a
// fault, at the cost of moving memory the replay would otherwise place exactly
// where the recording had it. Two granularities, both opt-in:
//
//   --guard-segments  every device allocation is VMM-backed with an unmapped
//                     span after it. Layout inside the segment is untouched, so
//                     this only catches a run off the end of a whole segment.
//   --guard-blocks    needs region annotations. For the duration of one launch,
//                     each argument that resolves into a live block is handed a
//                     copy of that block placed against an unmapped guard, so an
//                     overrun past the object itself faults. The copy is written
//                     back and released before the next event, which keeps the
//                     divergence to exactly the launch being examined.
//
// The device ordinal comes from the calling thread rather than being pinned to
// 0: a replay that switched device would otherwise reserve and map on the wrong
// one, and hipMemSetAccess would grant access to a device the kernel is not on.
static int hrr_current_device() {
    int dev = 0;
    (void)hipGetDevice(&dev);
    return dev;
}

static size_t hrr_vmm_granularity(int device) {
    hipMemAllocationProp prop{};
    prop.type          = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id   = device;
    size_t gran = 0;
    if (hipMemGetAllocationGranularity(&gran, &prop,
                                       hipMemAllocationGranularityMinimum)
            == hipSuccess && gran)
        return gran;
    return (2ull << 20);
}

// Reserve `mapped + gran` of VA, back only the first `mapped` bytes, and leave
// the tail span unmapped. Any access past the mapped region traps.
static hipError_t hrr_guard_map(size_t mapped, size_t gran, int device,
                                void** out_va, size_t* out_reserved,
                                hipMemGenericAllocationHandle_t* out_handle) {
    const size_t reserved = mapped + gran;
    hipMemAllocationProp prop{};
    prop.type          = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id   = device;

    void* va = nullptr;
    hipError_t r = hipMemAddressReserve(&va, reserved, 0, nullptr, 0);
    if (r != hipSuccess || !va) return r != hipSuccess ? r : hipErrorOutOfMemory;

    hipMemGenericAllocationHandle_t handle{};
    r = hipMemCreate(&handle, mapped, &prop, 0);
    if (r != hipSuccess) { (void)hipMemAddressFree(va, reserved); return r; }

    r = hipMemMap(va, mapped, 0, handle, 0);
    if (r != hipSuccess) {
        (void)hipMemRelease(handle);
        (void)hipMemAddressFree(va, reserved);
        return r;
    }

    hipMemAccessDesc desc{};
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id   = device;
    desc.flags         = hipMemAccessFlagsProtReadWrite;
    r = hipMemSetAccess(va, mapped, &desc, 1);
    if (r != hipSuccess) {
        (void)hipMemUnmap(va, mapped);
        (void)hipMemRelease(handle);
        (void)hipMemAddressFree(va, reserved);
        return r;
    }

    *out_va       = va;
    *out_reserved = reserved;
    *out_handle   = handle;
    return hipSuccess;
}

static void hrr_guard_unmap(void* va, size_t mapped, size_t reserved,
                            hipMemGenericAllocationHandle_t handle) {
    (void)hipMemUnmap(va, mapped);
    (void)hipMemRelease(handle);
    (void)hipMemAddressFree(va, reserved);
}

// ---- Segment-tail guard (--guard-segments) ----------------------------------

static hipError_t hrr_guard_alloc(PlaybackContext& ctx, size_t want, void** out) {
    const int    dev    = hrr_current_device();
    const size_t gran   = hrr_vmm_granularity(dev);
    const size_t mapped = ((want + gran - 1) / gran) * gran;

    void* va = nullptr;
    size_t reserved = 0;
    hipMemGenericAllocationHandle_t handle{};
    hipError_t r = hrr_guard_map(mapped, gran, dev, &va, &reserved, &handle);
    if (r != hipSuccess) return r;

    {
        std::unique_lock lk(ctx.map_mutex);
        ctx.guard_allocs[va] = {va, reserved, mapped, handle};
    }
    static std::once_flag once;
    std::call_once(once, [&] {
        fprintf(stderr,
                "[HRR] guard-segments active: first VMM-backed allocation "
                "want=%zu mapped=%zu guard=%zu\n", want, mapped, reserved - mapped);
    });
    *out = va;
    return hipSuccess;
}

// Returns true if `live` was a guarded allocation and was torn down.
static bool hrr_guard_free(PlaybackContext& ctx, void* live) {
    PlaybackContext::GuardAlloc g;
    {
        std::unique_lock lk(ctx.map_mutex);
        auto it = ctx.guard_allocs.find(live);
        if (it == ctx.guard_allocs.end()) return false;
        g = it->second;
        ctx.guard_allocs.erase(it);
    }
    hrr_guard_unmap(g.va_base, g.mapped, g.reserved, g.handle);
    return true;
}

void hrr_free_device_alloc(PlaybackContext& ctx, void* live) {
    if (!live) return;
    if (hrr_guard_free(ctx, live)) return;
    (void)hipFree(live);
}

// ---- Block guard (--guard-blocks) -------------------------------------------

// Largest power of two dividing v, capped at `cap`. This is the alignment the
// recorded pointer actually had, and reproducing it is what keeps a kernel on
// the same code path: hipBLASLt tile selection and vectorised loads both branch
// on how aligned their operands are, so handing a kernel a differently-aligned
// copy of its data can change what it executes or make it fault for a reason
// the recording never had.
static size_t hrr_alignment_of(uint64_t v, size_t cap) {
    if (v == 0) return cap;
    size_t a = static_cast<size_t>(v & (~v + 1));  // v & -v
    return a > cap ? cap : a;
}

// Place `size` bytes of block whose recorded base is `blk_base` against an
// unmapped guard. Fills *blind with the number of bytes between the block's end
// and the guard — an overrun smaller than that is still not caught.
static hipError_t hrr_block_guard_alloc(PlaybackContext& ctx, uint64_t blk_base,
                                        size_t size, HrrBlockGuard* g,
                                        size_t* blind) {
    if (size == 0) return hipErrorInvalidValue;
    const int    dev  = hrr_current_device();
    const size_t gran = hrr_vmm_granularity(dev);

    // Where inside the mapped span the block starts. Two policies, both of
    // which keep the guard immediately after the mapped region:
    //   default  right-align to the block's own alignment, so the guard is at
    //            most (alignment - 1) bytes past the block's end.
    //   exact    reproduce blk_base's offset within a granule bit for bit, at
    //            the cost of a blind spot of up to one granule.
    size_t pad;
    if (ctx.guard_exact_align) {
        pad = static_cast<size_t>(blk_base & (gran - 1));
    } else {
        const size_t align  = hrr_alignment_of(blk_base, gran);
        const size_t mapped = ((size + gran - 1) / gran) * gran;
        pad = (mapped - size) & ~(align - 1);
    }
    const size_t mapped = ((pad + size + gran - 1) / gran) * gran;
    if (blind) *blind = mapped - (pad + size);

    void* va = nullptr;
    size_t reserved = 0;
    hipMemGenericAllocationHandle_t handle{};
    hipError_t r = hrr_guard_map(mapped, gran, dev, &va, &reserved, &handle);
    if (r != hipSuccess) return r;

    g->va        = va;
    g->reserved  = reserved;
    g->mapped    = mapped;
    g->handle    = handle;
    g->data      = static_cast<char*>(va) + pad;
    g->size      = size;
    g->orig_live = nullptr;
    return hipSuccess;
}

static void hrr_block_guard_teardown(const HrrBlockGuard& g) {
    hrr_guard_unmap(g.va, g.mapped, g.reserved, g.handle);
}

// Hand this argument a guarded copy of its owning block. One buffer per
// distinct block per launch, so two arguments pointing into the same object
// still see the same memory as they did at capture.
static void* hrr_block_guard_relocate(PlaybackContext& ctx, RegionLaunchState& rls,
                                      uint64_t rec_ptr, void* live,
                                      uint64_t blk_base, uint64_t blk_size) {
    if (blk_size < ctx.guard_min_bytes) return live;
    if (ctx.guard_max_bytes && blk_size > ctx.guard_max_bytes) return live;

    auto it = rls.relocated.find(blk_base);
    if (it != rls.relocated.end())
        return static_cast<char*>(it->second) + (rec_ptr - blk_base);

    // A launch can take a hundred pointers; without a ceiling, guarding all of
    // them at granule resolution exhausts VRAM before the kernel runs.
    size_t outstanding = 0;
    for (const auto& g : rls.guards) outstanding += g.mapped;
    if (outstanding + blk_size > ctx.guard_budget_bytes) {
        static std::once_flag once;
        std::call_once(once, [&] {
            fprintf(stderr,
                    "[HRR] guard-blocks: budget of %zu MiB reached in a single "
                    "launch — remaining blocks are left unguarded "
                    "(--guard-budget-mb)\n",
                    ctx.guard_budget_bytes >> 20);
        });
        return live;
    }

    void* live_base = ctx.translate_ptr(blk_base);
    if (!live_base) return live;

    HrrBlockGuard g{};
    size_t blind = 0;
    if (hrr_block_guard_alloc(ctx, blk_base, static_cast<size_t>(blk_size), &g,
                              &blind) != hipSuccess)
        return live;

    // Seed the guarded copy so a kernel that reads or accumulates into the
    // block still computes from the values the recording had.
    if (hipMemcpy(g.data, live_base, blk_size, hipMemcpyDeviceToDevice)
            != hipSuccess) {
        hrr_block_guard_teardown(g);
        return live;
    }
    g.orig_live = live_base;
    rls.relocated[blk_base] = g.data;
    rls.guards.push_back(g);
    ctx.guard_blocks_relocated.fetch_add(1, std::memory_order_relaxed);
    if (blind > ctx.guard_blind_max.load(std::memory_order_relaxed))
        ctx.guard_blind_max.store(blind, std::memory_order_relaxed);

    return static_cast<char*>(g.data) + (rec_ptr - blk_base);
}

// After the launch: did anything run off the end of a guarded block? Sync so
// the fault is attributed to this kernel rather than to whatever event happens
// to synchronize next. On a clean launch the results are copied back to the
// real blocks and the guards released, leaving the rest of the replay
// byte-identical to an unguarded run.
static hipError_t hrr_block_guard_resolve(PlaybackContext& ctx,
                                          RegionLaunchState& rls,
                                          const std::string& kernel_name,
                                          size_t kernel_ordinal) {
    if (rls.guards.empty()) return hipSuccess;

    if (ctx.in_graph_capture) {
        // hipDeviceSynchronize and blocking D2D hipMemcpy are illegal during
        // stream capture (HIP 901). Relocate already skips when capturing, so
        // this is defence in depth: tear the reservations down without a device
        // round-trip and without treating a capture abort as a guard fault.
        for (const auto& g : rls.guards) hrr_block_guard_teardown(g);
        rls.guards.clear();
        rls.relocated.clear();
        return hipSuccess;
    }

    (void)hipGetLastError();
    hipError_t gs = hipDeviceSynchronize();
    hipError_t ge = hipGetLastError();
    if (gs != hipSuccess || ge != hipSuccess) {
        fprintf(stderr,
                "[HRR] GUARD FAULT: kernel #%zu \"%s\" ran past the end of a "
                "guarded block (sync=%d last=%d); %zu block(s) were guarded. "
                "This is the intra-segment out-of-bounds access the contiguous "
                "replay segment would otherwise have absorbed.\n",
                kernel_ordinal, compact_kernel_name(kernel_name).c_str(),
                static_cast<int>(gs), static_cast<int>(ge), rls.guards.size());
        // The context is gone; unmapping now would only add errors on the way
        // out. Leave the reservations to process teardown.
        rls.guards.clear();
        rls.relocated.clear();
        return gs != hipSuccess ? gs : ge;
    }

    for (const auto& g : rls.guards) {
        (void)hipMemcpy(g.orig_live, g.data, g.size, hipMemcpyDeviceToDevice);
        hrr_block_guard_teardown(g);
    }
    rls.guards.clear();
    rls.relocated.clear();
    return hipSuccess;
}

// ---- Divergence-abort guard -------------------------------------------------
// Replaying a numerically-unstable workload (e.g. a model emitting degenerate
// output) cannot reproduce bit-identical results from nondeterministic GPU
// reductions, so data diverges wholesale and a downstream kernel eventually
// writes out of bounds, killing the GPU context unrecoverably. Rather than die
// on that fault, watch the D2H validation failure fraction and stop cleanly
// once it is clearly broken — this turns the intermittent fault into a
// deterministic, diagnosable "replay diverged" exit.
//
// HIP_HRR_REPLAY_DIVERGENCE_ABORT : fail fraction in [0,1]; default 0.25.
//                                   0 disables the guard.
// HIP_HRR_REPLAY_DIVERGENCE_MIN_SAMPLES : min D2H attempts before the ratio is
//                                   evaluated (avoids tripping on noise);
//                                   default 64.
static double hrr_divergence_abort_frac() {
    static std::once_flag once;
    static double frac = 0.25;
    std::call_once(once, [] {
        if (const char* e = std::getenv("HIP_HRR_REPLAY_DIVERGENCE_ABORT")) {
            char* end = nullptr;
            double v = std::strtod(e, &end);
            if (end != e && v >= 0.0 && v <= 1.0) {
                frac = v;
                fprintf(stderr,
                        "[HRR] replay divergence-abort threshold = %.3f%s\n",
                        frac, frac == 0.0 ? " (DISABLED)" : "");
            }
        }
    });
    return frac;
}

static size_t hrr_divergence_min_samples() {
    static std::once_flag once;
    static size_t n = 64;
    std::call_once(once, [] {
        if (const char* e = std::getenv("HIP_HRR_REPLAY_DIVERGENCE_MIN_SAMPLES")) {
            char* end = nullptr;
            unsigned long v = std::strtoul(e, &end, 10);
            if (end != e && v > 0)
                n = static_cast<size_t>(v);
        }
    });
    return n;
}

void PlaybackContext::note_d2h_fail(uint64_t seq) {
    size_t fail = d2h_fail.fetch_add(1, std::memory_order_relaxed) + 1;
    double frac = hrr_divergence_abort_frac();
    if (frac <= 0.0)
        return;  // guard disabled
    size_t att = d2h_attempted.load(std::memory_order_relaxed);
    if (att < hrr_divergence_min_samples())
        return;
    if (static_cast<double>(fail) < frac * static_cast<double>(att))
        return;
    // Threshold crossed — flag once and stop the replay cleanly.
    if (!diverged.exchange(true, std::memory_order_acq_rel)) {
        fprintf(stderr,
                "[HRR] replay DIVERGED at recorded event seq %llu: %zu/%zu D2H "
                "validations failed (%.1f%% >= %.1f%% threshold). Aborting "
                "cleanly before a downstream GPU fault. This is a replay-fidelity "
                "divergence (e.g. nondeterministic GPU reductions in an unstable "
                "model state), not an HRR translation/memory bug. Set "
                "HIP_HRR_REPLAY_DIVERGENCE_ABORT=0 to disable this guard.\n",
                static_cast<unsigned long long>(seq), fail, att,
                100.0 * static_cast<double>(fail) / static_cast<double>(att),
                100.0 * frac);
        fatal_error.store(true, std::memory_order_release);
    }
}

static hipError_t replay_malloc(PlaybackContext& ctx, const uint8_t* pl,
                                bool managed = false) {
    const auto* a = reinterpret_cast<const hrr_args_hipMalloc*>(pl);
    size_t orig_sz = static_cast<size_t>(a->size);
    size_t pad_sz  = replay_padded_alloc_size(orig_sz);
    void* live = nullptr;
    hipError_t r;
    // --guard-segments: back the allocation with VMM and leave an unmapped span
    // after it, so running off the end of the segment traps here instead of
    // landing in whatever the driver placed next. Managed memory has no VMM
    // equivalent, and a failed reservation falls back to a plain allocation
    // rather than failing the replay.
    if (!managed && ctx.guard_segments &&
        hrr_guard_alloc(ctx, pad_sz, &live) == hipSuccess) {
        r = hipSuccess;
    } else if (managed) {
        r = hipMallocManaged(&live, pad_sz);
    } else {
        r = hipMalloc(&live, pad_sz);
    }
    if (r == hipSuccess) {
        // hipMalloc does NOT guarantee zeroed memory (only first-touch pages are
        // scrubbed; reused allocations carry stale bytes). Zero so replay is
        // deterministic and matches first-touch-zeroed assumptions. See
        // hrr_replay_zero_init(). The zero-init is skipped during graph capture,
        // where the original run never issued it and an injected memset would
        // invalidate the capture (HIP 901) for every subsequent op in the graph.
        hrr_zero_init_alloc(ctx, live, pad_sz);
        ctx.record_alloc(a->ptr, live, pad_sz);
        if (ctx.verbose && pad_sz > orig_sz)
            fprintf(stderr, "[HRR] hipMalloc 0x%llx: orig=%zu padded=%zu\n",
                    (unsigned long long)a->ptr, orig_sz, pad_sz);
    }
    return r;
}

hipError_t playback_hipMalloc(PlaybackContext& ctx, const uint8_t* pl) {
    return replay_malloc(ctx, pl);
}
hipError_t playback_hipMallocManaged(PlaybackContext& ctx, const uint8_t* pl) {
    return replay_malloc(ctx, pl, /*managed=*/true);
}

// ---------------------------------------------------------------------------
// Manual playback: hipExtMallocWithFlags
// ---------------------------------------------------------------------------
// A real device allocation (preserving the recorded flags) that must land in
// alloc_map, otherwise any H2D/D2H copy or kernel-arg pointer derived from the
// returned buffer would translate to nullptr. Mirrors replay_malloc for padding
// and zero-init so its fidelity matches hipMalloc.
hipError_t playback_hipExtMallocWithFlags(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipExtMallocWithFlags*>(pl);
    size_t orig_sz = static_cast<size_t>(a->sizeBytes);
    size_t pad_sz  = replay_padded_alloc_size(orig_sz);
    void* live = nullptr;
    hipError_t r = hipExtMallocWithFlags(&live, pad_sz, a->flags);
    if (r == hipSuccess) {
        hrr_zero_init_alloc(ctx, live, pad_sz);
        ctx.record_alloc(a->ptr, live, pad_sz);
        if (ctx.verbose && pad_sz > orig_sz)
            fprintf(stderr, "[HRR] hipExtMallocWithFlags 0x%llx: orig=%zu padded=%zu\n",
                    (unsigned long long)a->ptr, orig_sz, pad_sz);
    }
    return r;
}


// ---------------------------------------------------------------------------
// Manual playback: hipMallocAsync / hipMallocFromPoolAsync
// ---------------------------------------------------------------------------
// hipMallocAsync:  ret(4) dev_ptr(8) size(8) stream(8)
// hipMallocFromPoolAsync: ret(4) dev_ptr(8) size(8) mem_pool(8) stream(8)
//
// These do not need hrr_zero_init_alloc()'s drain (ROCM-27985): the zero-init is
// enqueued on the allocating stream, and stream-ordered allocations are only
// usable on that stream until the recorded program itself establishes an
// ordering edge to another stream. Replaying that edge carries the zero-init
// with it, so it is already ordered ahead of every recorded use.

hipError_t playback_hipMallocAsync(PlaybackContext& ctx,
                                   const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMallocAsync*>(pl);
    hipStream_t stream = ctx.translate_stream(a->stream);
    void* live = nullptr;
    size_t orig_sz = static_cast<size_t>(a->size);
    size_t pad_sz  = replay_padded_alloc_size(orig_sz);
    hipError_t r = hipMallocAsync(&live, pad_sz, stream);
    if (r == hipSuccess) {
        if (hrr_replay_zero_init() && !ctx.in_graph_capture)
            (void)hipMemsetAsync(live, 0, pad_sz, stream);
        ctx.record_alloc(a->dev_ptr, live, pad_sz);
    }
    return r;
}

hipError_t playback_hipMallocFromPoolAsync(PlaybackContext& ctx,
                                           const uint8_t* pl) {
    const auto* a  = reinterpret_cast<const hrr_args_hipMallocFromPoolAsync*>(pl);
    hipMemPool_t pool   = ctx.translate_mempool(a->mem_pool);
    hipStream_t  stream = ctx.translate_stream(a->stream);
    void* live = nullptr;
    size_t orig_sz = static_cast<size_t>(a->size);
    size_t pad_sz  = replay_padded_alloc_size(orig_sz);
    hipError_t r = hipMallocFromPoolAsync(&live, pad_sz, pool, stream);
    if (r == hipSuccess) {
        if (hrr_replay_zero_init() && !ctx.in_graph_capture)
            (void)hipMemsetAsync(live, 0, pad_sz, stream);
        ctx.record_alloc(a->dev_ptr, live, pad_sz);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipMemPoolSetAttribute / hipMemPoolGetAttribute
// value is void*; stored inline as value_u64 (8 bytes covers all attr sizes).
// GetAttribute is a no-op at playback (output only; pool state matches capture).
// ---------------------------------------------------------------------------

hipError_t playback_hipMemPoolSetAttribute(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemPoolSetAttribute*>(pl);
    hipMemPool_t pool = ctx.translate_mempool(a->mem_pool);
    return hipMemPoolSetAttribute(pool, (hipMemPoolAttr)a->attr,
                                  const_cast<void*>(static_cast<const void*>(&a->value_u64)));
}

hipError_t playback_hipMemPoolGetAttribute(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemPoolGetAttribute*>(pl);
    hipMemPool_t pool = ctx.translate_mempool(a->mem_pool);
    uint64_t scratch = 0;
    return hipMemPoolGetAttribute(pool, (hipMemPoolAttr)a->attr, &scratch);
}

// ---------------------------------------------------------------------------
// Manual playback: hipMemPoolCreate
// pool_props stored inline as pool_props_bytes[88] — reconstruct and pass.
// ---------------------------------------------------------------------------

hipError_t playback_hipMemPoolCreate(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemPoolCreate*>(pl);
    hipMemPoolProps props{};
    static_assert(sizeof(props) <= sizeof(a->pool_props_bytes),
                  "hipMemPoolProps larger than pool_props_bytes[88]");
    std::memcpy(&props, a->pool_props_bytes, sizeof(props));
    hipMemPool_t live = nullptr;
    hipError_t r = hipMemPoolCreate(&live, &props);
    if (r == hipSuccess)
        ctx.record_mempool(a->mem_pool, live);
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipHostMalloc / hipMallocHost
// ---------------------------------------------------------------------------
// hipHostMalloc:  ret(4) ptr(8) size(8) flags(4)
// hipMallocHost:  ret(4) ptr(8) size(8)

hipError_t playback_hipHostMalloc(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipHostMalloc*>(pl);
    void* live = nullptr;
    hipError_t r = hipHostMalloc(&live, static_cast<size_t>(a->size), a->flags);
    if (r == hipSuccess)
        ctx.record_alloc(a->ptr, live, static_cast<size_t>(a->size), AllocKind::HostMalloc);
    return r;
}

hipError_t playback_hipMallocHost(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMallocHost*>(pl);
    void* live = nullptr;
    hipError_t r = hipMallocHost(&live, static_cast<size_t>(a->size));
    if (r == hipSuccess)
        ctx.record_alloc(a->ptr, live, static_cast<size_t>(a->size), AllocKind::HostMalloc);
    return r;
}

hipError_t playback_hipFreeHost(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipFreeHost*>(pl);
    void* live = ctx.translate_ptr(a->ptr);
    if (!live) return hipSuccess;
    hipError_t r = hipFreeHost(live);
    if (r == hipSuccess) ctx.remove_alloc(a->ptr);
    return r;
}

hipError_t playback_hipHostFree(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipHostFree*>(pl);
    void* live = ctx.translate_ptr(a->ptr);
    if (!live) return hipSuccess;
    hipError_t r = hipHostFree(live);
    if (r == hipSuccess) ctx.remove_alloc(a->ptr);
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipHostRegister / hipHostUnregister
// ---------------------------------------------------------------------------
// hipHostRegister recorded a snapshot of the host memory as a blob.
// At replay we allocate a fresh host buffer (malloc), restore the blob
// into it, call hipHostRegister on it, and track the (recorded -> live)
// mapping so kernel-arg pointer translations work.
// hipHostUnregister unregisters, frees the backing buffer, and removes the entry.

hipError_t playback_hipHostRegister(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipHostRegister*>(pl);
    size_t sz = static_cast<size_t>(a->sizeBytes);
    if (sz == 0) return hipSuccess;

    // Allocate backing host buffer aligned to 64 bytes (page-register friendly).
    void* buf = nullptr;
#ifdef _WIN32
    buf = _aligned_malloc(sz, 64);
#else
    if (posix_memalign(&buf, 64, sz) != 0) buf = nullptr;
#endif
    if (!buf) return hipErrorMemoryAllocation;

    // Restore snapshot into the buffer.
    if (a->blob_hash_lo || a->blob_hash_hi) {
        size_t blob_sz = 0;
        const void* blob = ctx.load_blob(a->blob_hash_lo, a->blob_hash_hi, &blob_sz);
        if (blob && blob_sz == sz)
            std::memcpy(buf, blob, sz);
        else
            std::memset(buf, 0, sz);
    } else {
        std::memset(buf, 0, sz);
    }

    hipError_t r = hipHostRegister(buf, sz, a->flags);
    if (r == hipSuccess) {
        ctx.record_alloc(a->hostPtr, buf, sz, AllocKind::HostRegister);
        std::unique_lock lk(ctx.map_mutex);
        ctx.host_reg_bufs[a->hostPtr] = buf;
    } else {
#ifdef _WIN32
        _aligned_free(buf);
#else
        free(buf);
#endif
    }
    return r;
}

hipError_t playback_hipHostUnregister(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipHostUnregister*>(pl);

    // Retrieve the backing buffer regardless of whether translate_ptr succeeds —
    // we must free it even if the alloc_map entry was already removed.
    void* buf = nullptr;
    {
        std::unique_lock lk(ctx.map_mutex);
        auto it = ctx.host_reg_bufs.find(a->hostPtr);
        if (it != ctx.host_reg_bufs.end()) {
            buf = it->second;
            ctx.host_reg_bufs.erase(it);
        }
    }

    void* live = buf ? buf : ctx.translate_ptr(a->hostPtr);
    if (!live) return hipSuccess;

    hipError_t r = hipHostUnregister(live);
    if (r == hipSuccess) ctx.remove_alloc(a->hostPtr);

#ifdef _WIN32
    _aligned_free(buf);
#else
    free(buf);
#endif
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipHostGetDevicePointer
// ---------------------------------------------------------------------------
// The generated shim would pass the raw recorded host ptr to the real API,
// which fails because it's a stale captured address.  We need to translate
// it through host_reg_bufs first, then record the returned device pointer
// in alloc_map so future translate_ptr calls work.

hipError_t playback_hipHostGetDevicePointer(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipHostGetDevicePointer*>(pl);

    // Check host_reg_bufs first (hipHostRegister path), then alloc_map
    // (hipHostMalloc path — already pinned, no register step needed).
    void* live_host = nullptr;
    {
        std::unique_lock lk(ctx.map_mutex);
        auto it = ctx.host_reg_bufs.find(a->hstPtr);
        if (it != ctx.host_reg_bufs.end())
            live_host = it->second;
    }
    if (!live_host)
        live_host = ctx.translate_ptr(a->hstPtr);

    if (!live_host) {
        fprintf(stderr, "[HRR] hipHostGetDevicePointer: no live buf for recorded hstPtr %llx\n",
                (unsigned long long)a->hstPtr);
        return hipErrorInvalidValue;
    }

    void* dev_ptr = nullptr;
    hipError_t r = hipHostGetDevicePointer(&dev_ptr, live_host, a->flags);
    if (r == hipSuccess) {
        // For zero-copy host allocations the device pointer equals the host
        // pointer, so a->devPtr collides with the recorded address of the
        // owning hipHostMalloc/hipMalloc entry. Recording an alias here would
        // overwrite that entry and downgrade its AllocKind to DevicePtrAlias,
        // losing the info teardown needs to hipHostFree the pinned buffer —
        // leaking it on any path that frees via alloc_map (e.g. the
        // divergence-abort teardown, which runs before the replayed hipHostFree
        // event). Only record a genuinely distinct alias.
        if (!ctx.has_alloc(a->devPtr))
            ctx.record_alloc(a->devPtr, dev_ptr, 0, AllocKind::DevicePtrAlias);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipFree / hipFreeAsync
// ---------------------------------------------------------------------------
// hipFree:       ret(4) ptr(8)
// hipFreeAsync:  ret(4) dev_ptr(8) stream(8)

hipError_t playback_hipFree(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipFree*>(pl);
    void* live = ctx.translate_ptr(a->ptr);
    if (!live) return hipSuccess;
    if (hrr_guard_free(ctx, live)) {  // --guard-segments: a VMM mapping
        ctx.remove_alloc(a->ptr);
        return hipSuccess;
    }
    hipError_t r = hipFree(live);
    if (r == hipSuccess) ctx.remove_alloc(a->ptr);
    return r;
}

hipError_t playback_hipFreeAsync(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a  = reinterpret_cast<const hrr_args_hipFreeAsync*>(pl);
    void*       live   = ctx.translate_ptr(a->dev_ptr);
    hipStream_t stream = ctx.translate_stream(a->stream);
    if (!live) return hipSuccess;
    hipError_t r = hipFreeAsync(live, stream);
    if (r == hipSuccess) ctx.remove_alloc(a->dev_ptr);
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipMemcpy / hipMemcpyAsync / hipMemcpyHtoD / hipMemcpyHtoDAsync
// ---------------------------------------------------------------------------

// is_async: true  -> call hipMemcpyAsync(stream) regardless of whether stream is null
//           false -> call synchronous hipMemcpy (no stream argument)
// ---------------------------------------------------------------------------
// D2H validation with numeric tolerance.
//
// A byte-exact memcmp is the wrong oracle for buffers produced by GPU kernels
// that use non-associative reductions (atomicAdd in backward passes, split-K
// GEMM accumulation, etc.): those are nondeterministic at the ULP level, so a
// faithful replay legitimately produces slightly different bytes than capture.
// (Verified by a replay-vs-replay control: the same recording replayed twice
// produces different bytes for the same tensors, so the nondeterminism is in
// the kernels, not in HRR.) Reporting that as "FAIL" is misleading.
//
// Instead we classify a mismatch numerically: a buffer passes if every element
// is within  |actual - expected| <= atol + rtol*|expected|. The recorded blob
// carries no dtype, so we try candidate float encodings (fp32, bf16, fp16,
// fp64) and accept if any encoding fits — a wrong encoding turns small diffs
// into garbage/inf and is rejected, so the true dtype is the one that fits.
// Genuine corruption (wrong pointer, shifted/zeroed data) produces large,
// structured differences that fit no encoding and still FAILs.
//
// Tunable via HIP_HRR_D2H_ATOL / HIP_HRR_D2H_RTOL; HIP_HRR_D2H_EXACT=1 forces
// the old byte-exact behavior.
struct HrrD2HTol { double atol; double rtol; bool exact_only; };
static const HrrD2HTol& hrr_d2h_tol() {
    static const HrrD2HTol t = [] {
        HrrD2HTol d{};
        const char* a = std::getenv("HIP_HRR_D2H_ATOL");
        const char* r = std::getenv("HIP_HRR_D2H_RTOL");
        d.atol = a ? std::atof(a) : 1e-3;
        d.rtol = r ? std::atof(r) : 1e-3;
        d.exact_only = std::getenv("HIP_HRR_D2H_EXACT") != nullptr;
        return d;
    }();
    return t;
}

static inline float hrr_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t man  = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) { f = sign; }
        else {  // subnormal
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; exp--; }
            man &= 0x3ffu;
            f = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1fu) {
        f = sign | 0x7f800000u | (man << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}

// Stats for one candidate-dtype interpretation of a mismatching buffer.
struct HrrD2HScan { size_t n_diff; size_t n_bad; double max_abs; double max_rel; };

template <typename DecodeFn>
static HrrD2HScan hrr_scan_elems(const uint8_t* a, const uint8_t* e, size_t n,
                                 size_t esz, DecodeFn dec,
                                 double atol, double rtol) {
    HrrD2HScan s{0, 0, 0.0, 0.0};
    for (size_t i = 0; i + esz <= n; i += esz) {
        if (memcmp(a + i, e + i, esz) == 0) continue;  // identical bytes
        s.n_diff++;
        double av = dec(a + i), ev = dec(e + i);
        bool av_nan = (av != av), ev_nan = (ev != ev);
        if (av_nan && ev_nan) continue;                // both NaN — equivalent
        double d = av - ev; if (d < 0) d = -d;
        double ev_abs = ev < 0 ? -ev : ev;
        double tol = atol + rtol * ev_abs;
        if (!(d <= tol)) s.n_bad++;                    // also catches inf/NaN d
        if (d == d) {                                  // finite/representable
            if (d > s.max_abs) s.max_abs = d;
            if (ev_abs > 0) { double r = d / ev_abs; if (r > s.max_rel) s.max_rel = r; }
        }
    }
    return s;
}

// Validate one D2H buffer: updates ctx counters and emits a message only for a
// genuine (out-of-tolerance) failure. Returns true if the buffer is acceptable
// (byte-exact or within tolerance).
static bool hrr_d2h_validate(PlaybackContext& ctx, const char* tag, uint64_t seq,
                             const uint8_t* actual, const uint8_t* expected, size_t n) {
    if (n == 0 || memcmp(actual, expected, n) == 0) {
        ctx.d2h_pass++;
        if (ctx.verbose)
            fprintf(stderr, "[HRR] %s D2H validate: %zu bytes OK (exact)\n", tag, n);
        return true;
    }

    const HrrD2HTol& tol = hrr_d2h_tol();
    size_t ndiff_bytes = 0;
    for (size_t i = 0; i < n; i++) if (actual[i] != expected[i]) ndiff_bytes++;
    size_t first_diff = 0;
    while (first_diff < n && actual[first_diff] == expected[first_diff]) ++first_diff;

    if (!tol.exact_only) {
        // Try candidate float encodings; accept on the first that fits, tracking
        // the best (fewest out-of-tolerance elements) for the failure report.
        auto dec_f32 = [](const uint8_t* p) { float v; memcpy(&v, p, 4); return (double)v; };
        auto dec_f64 = [](const uint8_t* p) { double v; memcpy(&v, p, 8); return v; };
        auto dec_bf16 = [](const uint8_t* p) {
            uint16_t h; memcpy(&h, p, 2); uint32_t u = (uint32_t)h << 16;
            float v; memcpy(&v, &u, 4); return (double)v; };
        auto dec_f16 = [](const uint8_t* p) {
            uint16_t h; memcpy(&h, p, 2); return (double)hrr_half_to_float(h); };

        struct Cand { const char* name; size_t esz; };
        const Cand cands[] = { {"f32", 4}, {"bf16", 2}, {"f16", 2}, {"f64", 8} };
        HrrD2HScan best{0, SIZE_MAX, 0.0, 0.0}; const char* best_name = "?";
        for (const auto& c : cands) {
            if (n % c.esz != 0) continue;
            HrrD2HScan s;
            if (c.esz == 4)      s = hrr_scan_elems(actual, expected, n, 4, dec_f32, tol.atol, tol.rtol);
            else if (c.esz == 8) s = hrr_scan_elems(actual, expected, n, 8, dec_f64, tol.atol, tol.rtol);
            else if (c.name[0] == 'b') s = hrr_scan_elems(actual, expected, n, 2, dec_bf16, tol.atol, tol.rtol);
            else                 s = hrr_scan_elems(actual, expected, n, 2, dec_f16, tol.atol, tol.rtol);
            if (s.n_bad < best.n_bad) { best = s; best_name = c.name; }
            if (s.n_bad == 0) {  // fits this encoding → numerically equivalent
                ctx.d2h_pass++;
                ctx.d2h_pass_tol++;
                if (ctx.verbose)
                    fprintf(stderr,
                            "[HRR] %s D2H validate: %zu bytes ~OK within tol as %s "
                            "(%zu elems differ, max|d|=%.3g maxrel=%.3g, atol=%g rtol=%g)\n",
                            tag, n, c.name, s.n_diff, s.max_abs, s.max_rel,
                            tol.atol, tol.rtol);
                return true;
            }
        }
        // No encoding fit — a real divergence.
        ctx.note_d2h_fail(seq);
        fprintf(stderr,
                "[HRR] %s D2H FAIL seq=%llu: %zu bytes, %zu/%zu bytes differ (%.2f%%), "
                "first@%zu (got 0x%02x exp 0x%02x); best fit %s: %zu/%zu elems exceed "
                "tol (atol=%g rtol=%g), max|d|=%.4g maxrel=%.4g\n",
                tag, (unsigned long long)seq, n, ndiff_bytes, n,
                100.0 * (double)ndiff_bytes / (double)n, first_diff,
                actual[first_diff], expected[first_diff], best_name,
                best.n_bad, best.n_diff, tol.atol, tol.rtol, best.max_abs, best.max_rel);
        return false;
    }

    // Exact-only mode: any byte mismatch is a failure.
    ctx.note_d2h_fail(seq);
    fprintf(stderr,
            "[HRR] %s D2H FAIL seq=%llu (exact): %zu bytes, %zu/%zu bytes differ, "
            "first@%zu (got 0x%02x exp 0x%02x)\n",
            tag, (unsigned long long)seq, n, ndiff_bytes, n, first_diff,
            actual[first_diff], expected[first_diff]);
    return false;
}

// Translate, and if nothing in the archive covers the address, back a
// sidecar-declared segment and translate into that — what the whole-pointer
// kernel-argument path already does. A copy can be the first touch of an
// allocation HIP never saw, and dispatch_event advances the region timeline
// before every handler precisely so that memcpys see the same view as a launch.
static void* translate_or_materialize(PlaybackContext& ctx, uint64_t rec_addr) {
    void* live = ctx.translate_ptr(rec_addr);
    if (!live && rec_addr != 0 && ctx.regions_enabled)
        live = ctx.regions.materialize_for(ctx, rec_addr);
    return live;
}

// This mirrors the captured API exactly — hipMemcpyAsync on the default stream
// (stream_rec==0, translated to nullptr) must still use the async variant.
static hipError_t replay_memcpy_impl(PlaybackContext& ctx,
                                     uint64_t dst_rec, uint64_t src_rec,
                                     uint64_t size, int32_t kind,
                                     bool is_async, hipStream_t stream,
                                     uint64_t hash_lo, uint64_t hash_hi) {
    void*      dst = translate_or_materialize(ctx, dst_rec);
    hipError_t r   = hipSuccess;


    if (kind == hipMemcpyHostToDevice && (hash_lo || hash_hi)) {
        size_t blob_sz = 0;
        const void* blob = ctx.load_blob(hash_lo, hash_hi, &blob_sz);
        if (!blob) {
            fprintf(stderr, "[HRR] H2D blob %016llx%016llx not found\n",
                    (unsigned long long)hash_lo, (unsigned long long)hash_hi);
            return hipErrorNotFound;
        }
        if (!dst) { fprintf(stderr, "[HRR] H2D dst 0x%llx not mapped (size=%llu blob_sz=%zu)\n",
                            (unsigned long long)dst_rec, (unsigned long long)size, blob_sz);
                    return hipErrorInvalidValue; }
        size_t copy_sz = static_cast<size_t>(size);
        if (copy_sz > blob_sz) copy_sz = blob_sz;
        size_t avail = ctx.alloc_bytes_from(dst);
        if (avail > 0 && copy_sz > avail) {
            fprintf(stderr, "[HRR] H2D dst 0x%llx: copy_sz=%zu > avail=%zu — clamping\n",
                    (unsigned long long)dst_rec, copy_sz, avail);
            copy_sz = avail;
        }
        if (is_async)
            r = hipMemcpyAsync(dst, blob, copy_sz, hipMemcpyHostToDevice, stream);
        else
            r = hipMemcpy(dst, blob, copy_sz, hipMemcpyHostToDevice);
        if (r != hipSuccess) {
            fprintf(stderr, "[HRR] H2D memcpy failed: %d (%s) dst=%p copy_sz=%zu blob_sz=%zu avail=%zu\n",
                    r, hipGetErrorString(r), dst, copy_sz, blob_sz, avail);
        } else {
            r = hrr_sync_after_replayed_h2d(ctx, "replayed H2D memcpy");
        }
    } else if (kind == hipMemcpyDeviceToDevice) {
        void* src = translate_or_materialize(ctx, src_rec);
        if (!dst) fprintf(stderr, "[HRR] D2D dst 0x%llx not mapped\n", (unsigned long long)dst_rec);
        if (!src) fprintf(stderr, "[HRR] D2D src 0x%llx not mapped\n", (unsigned long long)src_rec);
        if (dst && src) {
            size_t copy_sz = static_cast<size_t>(size);
            size_t dst_avail = ctx.alloc_bytes_from(dst);
            size_t src_avail = ctx.alloc_bytes_from(src);
            if (dst_avail > 0 && copy_sz > dst_avail) {
                fprintf(stderr, "[HRR] D2D dst_avail=%zu < copy_sz=%zu — clamping\n", dst_avail, copy_sz);
                copy_sz = dst_avail;
            }
            if (src_avail > 0 && copy_sz > src_avail)
                copy_sz = src_avail;
            if (is_async)
                r = hipMemcpyAsync(dst, src, copy_sz,
                                   hipMemcpyDeviceToDevice, stream);
            else
                r = hipMemcpy(dst, src, copy_sz,
                              hipMemcpyDeviceToDevice);
            if (r != hipSuccess)
                fprintf(stderr, "[HRR] D2D memcpy failed: %d (%s)\n", r, hipGetErrorString(r));
        }
    } else if (kind == hipMemcpyDeviceToHost && ctx.validate_d2h &&
               (hash_lo || hash_hi)) {
        // D2H validation: copy from live device src into a local host buffer,
        // then compare against the expected data blob captured at record time.
        ctx.d2h_attempted++;
        void* src_dev = translate_or_materialize(ctx, src_rec);
        if (!src_dev) {
            fprintf(stderr, "[HRR] D2H validate FAIL: src 0x%llx not mapped — pointer translation bug\n",
                    (unsigned long long)src_rec);
            ctx.note_d2h_fail(hrr_dispatch_seq);
            return hipErrorInvalidValue;
        } else {
            size_t copy_sz = static_cast<size_t>(size);
            size_t blob_sz = 0;
            const void* expected = ctx.load_blob(hash_lo, hash_hi, &blob_sz);
            if (!expected) {
                fprintf(stderr, "[HRR] D2H validate FAIL: expected blob not found in archive\n");
                ctx.note_d2h_fail(hrr_dispatch_seq);
            } else {
                copy_sz = std::min(copy_sz, blob_sz);
                std::vector<uint8_t> actual(copy_sz);
                // For async memcpy the stream may not yet have completed — sync it so
                // all preceding GPU work has finished before reading back.
                // Synchronous hipMemcpy already guarantees completion; no extra sync needed.
                if (is_async) (void)hipStreamSynchronize(stream);
                r = hipMemcpy(actual.data(), src_dev, copy_sz, hipMemcpyDeviceToHost);
                if (r != hipSuccess) {
                    fprintf(stderr, "[HRR] D2H validate: hipMemcpy failed: %d (%s)\n",
                            r, hipGetErrorString(r));
                    ctx.note_d2h_fail(hrr_dispatch_seq);
                } else {
                    hrr_d2h_validate(ctx, "kernarg", hrr_dispatch_seq, actual.data(),
                                     static_cast<const uint8_t*>(expected), copy_sz);
                }
            }
        }
    }
    // H2H / unhandled: no-op
    return r;
}

hipError_t playback_hipMemcpy(PlaybackContext& ctx,
                              const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpy*>(pl);
    return replay_memcpy_impl(ctx, a->dst, a->src, a->sizeBytes, a->kind,
                              /*is_async=*/false, nullptr,
                              a->blob_hash_lo, a->blob_hash_hi);
}

hipError_t playback_hipMemcpyAsync(PlaybackContext& ctx,
                                   const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyAsync*>(pl);
    return replay_memcpy_impl(ctx, a->dst, a->src, a->sizeBytes, a->kind,
                              /*is_async=*/true, ctx.translate_stream(a->stream),
                              a->blob_hash_lo, a->blob_hash_hi);
}

hipError_t playback_hipMemcpyHtoD(PlaybackContext& ctx,
                                  const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyHtoD*>(pl);
    return replay_memcpy_impl(ctx, a->dst, a->src, a->sizeBytes,
                              hipMemcpyHostToDevice,
                              /*is_async=*/false, nullptr,
                              a->blob_hash_lo, a->blob_hash_hi);
}

hipError_t playback_hipMemcpyHtoDAsync(PlaybackContext& ctx,
                                       const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyHtoDAsync*>(pl);
    return replay_memcpy_impl(ctx, a->dst, a->src, a->sizeBytes,
                              hipMemcpyHostToDevice,
                              /*is_async=*/true, ctx.translate_stream(a->stream),
                              a->blob_hash_lo, a->blob_hash_hi);
}

// ---------------------------------------------------------------------------
// Manual playback: hipMemcpyWithStream
// ---------------------------------------------------------------------------
// Synchronous copy with stream. Captured by manual shim (has blob_hash fields).
// Routes through replay_memcpy_impl exactly like hipMemcpy/hipMemcpyAsync.
// is_async=true so the stream is passed through (even if it translates to null).
hipError_t playback_hipMemcpyWithStream(PlaybackContext& ctx,
                                        const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyWithStream*>(pl);
    return replay_memcpy_impl(ctx, a->dst, a->src, a->sizeBytes, a->kind,
                              /*is_async=*/true, ctx.translate_stream(a->stream),
                              a->blob_hash_lo, a->blob_hash_hi);
}

// ---------------------------------------------------------------------------
// Manual playback: hipMemcpyDtoH / hipMemcpyDtoHAsync
// ---------------------------------------------------------------------------
// dst is a host pointer (captured as raw address, not in alloc_map).
// We copy from the live device src into a temp host buffer and compare against
// the expected blob captured at record time (D2H validation).
hipError_t playback_hipMemcpyDtoH(PlaybackContext& ctx,
                                  const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyDtoH*>(pl);
    uint64_t hash_lo = a->blob_hash_lo;
    uint64_t hash_hi = a->blob_hash_hi;
    if (hash_lo || hash_hi) ctx.d2h_attempted++;

    void* src_dev = ctx.translate_ptr(a->src);
    if (!src_dev) {
        fprintf(stderr, "[HRR] hipMemcpyDtoH: src 0x%llx not mapped — D2H validate FAIL\n",
                (unsigned long long)a->src);
        if (hash_lo || hash_hi) ctx.note_d2h_fail(hrr_dispatch_seq);
        return hipErrorInvalidValue;
    }
    size_t sz = static_cast<size_t>(a->sizeBytes);
    std::vector<uint8_t> actual(sz);
    hipError_t r = hipMemcpyDtoH(actual.data(), (hipDeviceptr_t)src_dev, sz);
    if (r != hipSuccess) {
        fprintf(stderr, "[HRR] hipMemcpyDtoH failed: %d (%s)\n", r, hipGetErrorString(r));
        if (hash_lo || hash_hi) ctx.note_d2h_fail(hrr_dispatch_seq);
        return r;
    }
    if (hash_lo || hash_hi) {
        size_t blob_sz = 0;
        const void* expected = ctx.load_blob(hash_lo, hash_hi, &blob_sz);
        if (!expected) {
            fprintf(stderr, "[HRR] D2H validate FAIL: expected blob not found in archive\n");
            ctx.note_d2h_fail(hrr_dispatch_seq);
        } else {
            size_t cmp_sz = std::min(sz, blob_sz);
            hrr_d2h_validate(ctx, "DtoH", hrr_dispatch_seq, actual.data(),
                             static_cast<const uint8_t*>(expected), cmp_sz);
        }
    }
    return hipSuccess;
}

hipError_t playback_hipMemcpyDtoHAsync(PlaybackContext& ctx,
                                       const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyDtoHAsync*>(pl);
    uint64_t hash_lo = a->blob_hash_lo;
    uint64_t hash_hi = a->blob_hash_hi;
    if (hash_lo || hash_hi) ctx.d2h_attempted++;

    void* src_dev = ctx.translate_ptr(a->src);
    if (!src_dev) {
        fprintf(stderr, "[HRR] hipMemcpyDtoHAsync: src 0x%llx not mapped — D2H validate FAIL\n",
                (unsigned long long)a->src);
        if (hash_lo || hash_hi) ctx.note_d2h_fail(hrr_dispatch_seq);
        return hipErrorInvalidValue;
    }
    size_t sz = static_cast<size_t>(a->sizeBytes);
    std::vector<uint8_t> actual(sz);
    hipStream_t stream = ctx.translate_stream(a->stream);
    hipError_t r = hipMemcpyDtoHAsync(actual.data(), (hipDeviceptr_t)src_dev, sz, stream);
    if (r == hipSuccess) (void)hipStreamSynchronize(stream);
    if (r != hipSuccess) {
        fprintf(stderr, "[HRR] hipMemcpyDtoHAsync failed: %d (%s)\n", r, hipGetErrorString(r));
        if (hash_lo || hash_hi) ctx.note_d2h_fail(hrr_dispatch_seq);
        return r;
    }
    if (hash_lo || hash_hi) {
        size_t blob_sz = 0;
        const void* expected = ctx.load_blob(hash_lo, hash_hi, &blob_sz);
        if (!expected) {
            fprintf(stderr, "[HRR] D2H validate FAIL: expected blob not found in archive\n");
            ctx.note_d2h_fail(hrr_dispatch_seq);
        } else {
            size_t cmp_sz = std::min(sz, blob_sz);
            hrr_d2h_validate(ctx, "DtoHAsync", hrr_dispatch_seq, actual.data(),
                             static_cast<const uint8_t*>(expected), cmp_sz);
        }
    }
    return hipSuccess;
}

// ---------------------------------------------------------------------------
// Manual playback: stream create/destroy
// ---------------------------------------------------------------------------
// hipStreamCreate:              ret(4) stream(8)
// hipStreamCreateWithFlags:     ret(4) stream(8) flags(4)
// hipStreamCreateWithPriority:  ret(4) stream(8) flags(4) priority(4)
// hipStreamDestroy:             ret(4) stream(8)

hipError_t playback_hipStreamCreate(PlaybackContext& ctx,
                                    const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamCreate*>(pl);
    hipStream_t s = nullptr;
    hipError_t r = hipStreamCreate(&s);
    if (r == hipSuccess) ctx.record_stream(a->stream, s);
    return r;
}

hipError_t playback_hipStreamCreateWithFlags(PlaybackContext& ctx,
                                             const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamCreateWithFlags*>(pl);
    hipStream_t s = nullptr;
    hipError_t r  = hipStreamCreateWithFlags(&s, a->flags);
    if (r == hipSuccess) {
        ctx.record_stream(a->stream, s);
        if (ctx.verbose)
            fprintf(stderr, "[HRR] StreamCreateWithFlags: rec=0x%llx -> live=%p\n",
                    (unsigned long long)a->stream, (void*)s);
    }
    return r;
}

hipError_t playback_hipStreamCreateWithPriority(PlaybackContext& ctx,
                                                const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamCreateWithPriority*>(pl);
    hipStream_t s = nullptr;
    hipError_t  r = hipStreamCreateWithPriority(&s, a->flags, a->priority);
    if (r == hipSuccess) ctx.record_stream(a->stream, s);
    return r;
}

hipError_t playback_hipStreamDestroy(PlaybackContext& ctx,
                                     const uint8_t* pl) {
    const auto* a  = reinterpret_cast<const hrr_args_hipStreamDestroy*>(pl);
    hipStream_t stream = ctx.translate_stream(a->stream);
    hipError_t r = hipSuccess;
    if (stream) r = hipStreamDestroy(stream);
    ctx.remove_stream(a->stream);
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipStreamEndCapture / hipGraphInstantiate
// ---------------------------------------------------------------------------
// Stream-capture flow:
//   hipStreamBeginCapture — generated shim calls real API (no handle output)
//   hipStreamEndCapture   — calls real API, records resulting hipGraph_t handle
//   hipGraphInstantiate   — calls real API, records resulting hipGraphExec_t handle
//   hipGraphLaunch        — generated shim translates both handles; works once above succeed
//
// hrr_args_hipStreamEndCapture layout (after 32-byte EventHeader):
//   ret(4) stream(8) pGraph(8)       — pGraph = recorded *pGraph output value
//
// hrr_args_hipGraphInstantiate layout:
//   ret(4) pGraphExec(8) graph(8) pErrorNode(8) pLogBuffer(8) bufferSize(8)

// hrr_args_hipStreamBeginCapture payload (after 32-byte EventHeader):
//   ret(4) stream(8) mode(4)
hipError_t playback_hipStreamBeginCapture(PlaybackContext& ctx,
                                          const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamBeginCapture*>(payload);
    if (a->ret != hipSuccess) return hipSuccess;  // original failed — skip

    hipStream_t stream = ctx.translate_stream(a->stream);
    if (!stream && a->stream != 0) {
        // Stream handle not in map — create a temporary stream for graph capture
        fprintf(stderr, "[HRR] hipStreamBeginCapture: stream 0x%llx not found, "
                "creating temp stream for graph capture\n",
                (unsigned long long)a->stream);
        hipError_t cr = hipStreamCreate(&stream);
        if (cr != hipSuccess) {
            fprintf(stderr, "[HRR] hipStreamBeginCapture: failed to create temp stream: %d\n", cr);
            return hipSuccess;  // non-fatal
        }
        ctx.record_stream(a->stream, stream);
    }

    hipStreamCaptureMode mode = (hipStreamCaptureMode)a->mode;
    hipError_t r = hipStreamBeginCapture(stream, mode);
    if (r != hipSuccess && mode != hipStreamCaptureModeGlobal) {
        // ThreadLocal may fail in replay context — try Global
        r = hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
        if (r != hipSuccess)
            fprintf(stderr, "[HRR] hipStreamBeginCapture failed (both modes): %d (%s)\n",
                    r, hipGetErrorString(r));
    }
    if (r == hipSuccess)
        ctx.in_graph_capture = true;
    return r;
}

hipError_t playback_hipStreamEndCapture(PlaybackContext& ctx,
                                        const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamEndCapture*>(payload);
    if (a->ret != hipSuccess) return hipSuccess;  // original call failed — skip

    hipStream_t stream = ctx.translate_stream(a->stream);
    if (!stream) {
        fprintf(stderr, "[HRR] hipStreamEndCapture: stream 0x%llx not found in map\n",
                (unsigned long long)a->stream);
        return hipSuccess;  // non-fatal
    }
    ctx.in_graph_capture = false;
    hipGraph_t live_graph = nullptr;
    hipError_t r = hipStreamEndCapture(stream, &live_graph);
    if (r == hipSuccess && live_graph) {
        ctx.record_graph(a->pGraph, live_graph);
        if (ctx.verbose)
            fprintf(stderr, "[HRR] hipStreamEndCapture: recorded graph 0x%llx\n",
                    (unsigned long long)a->pGraph);
    } else {
        fprintf(stderr, "[HRR] hipStreamEndCapture failed: %d (%s)\n",
                r, hipGetErrorString(r));
    }
    return r;
}

hipError_t playback_hipGraphInstantiate(PlaybackContext& ctx,
                                        const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args_hipGraphInstantiate*>(payload);
    if (a->ret != hipSuccess) return hipSuccess;  // original call failed — skip

    hipGraph_t graph = ctx.translate_graph(a->graph);
    if (!graph) {
        // graph_map holds every graph this replay built, whether by stream
        // capture (hipStreamEndCapture) or by the node API (hipGraphCreate).
        // A miss means the graph does not exist here at all.
        fprintf(stderr,
                "[HRR] hipGraphInstantiate: graph 0x%llx was never built at "
                "replay, so there is nothing to instantiate. Aborting rather "
                "than running an empty graph.\n",
                (unsigned long long)a->graph);
        return hipErrorNotSupported;
    }
    if (ctx.graph_is_incomplete(a->graph)) {
        // The graph exists but is missing at least one node HRR could not
        // reconstruct (each one said so when it was skipped). Instantiating it
        // would run a graph short of work and quietly produce wrong buffers.
        fprintf(stderr,
                "[HRR] hipGraphInstantiate: graph 0x%llx is missing nodes this "
                "replay could not reconstruct (see the earlier per-node "
                "messages). Refusing to instantiate a graph that would run "
                "with work missing.\n",
                (unsigned long long)a->graph);
        return hipErrorNotSupported;
    }

    hipGraphExec_t exec = nullptr;
    // Use the simplified WithFlags variant; pErrorNode/pLogBuffer are optional at replay
    hipError_t r = hipGraphInstantiateWithFlags(&exec, graph, 0);
    if (r == hipSuccess && exec) {
        ctx.record_graph_exec(a->pGraphExec, exec);
        if (ctx.verbose)
            fprintf(stderr, "[HRR] hipGraphInstantiate: recorded exec 0x%llx\n",
                    (unsigned long long)a->pGraphExec);
    } else {
        fprintf(stderr, "[HRR] hipGraphInstantiate (via WithFlags) failed: %d (%s)\n",
                r, hipGetErrorString(r));
    }
    return r;
}

hipError_t playback_hipGraphInstantiateWithFlags(PlaybackContext& ctx,
                                                 const uint8_t* payload) {
    const auto* a =
        reinterpret_cast<const hrr_args_hipGraphInstantiateWithFlags*>(payload);
    if (a->ret != hipSuccess) return hipSuccess;  // original call failed — skip

    hipGraph_t graph = ctx.translate_graph(a->graph);
    if (!graph) {
        fprintf(stderr,
                "[HRR] hipGraphInstantiateWithFlags: graph 0x%llx was never "
                "built at replay. Aborting replay.\n",
                (unsigned long long)a->graph);
        return hipErrorNotSupported;
    }
    if (ctx.graph_is_incomplete(a->graph)) {
        // See playback_hipGraphInstantiate.
        fprintf(stderr,
                "[HRR] hipGraphInstantiateWithFlags: graph 0x%llx is missing "
                "nodes this replay could not reconstruct. Refusing to "
                "instantiate it.\n",
                (unsigned long long)a->graph);
        return hipErrorNotSupported;
    }

    hipGraphExec_t exec = nullptr;
    hipError_t r = hipGraphInstantiateWithFlags(&exec, graph,
                                                static_cast<unsigned long long>(a->flags));
    if (r == hipSuccess && exec) {
        ctx.record_graph_exec(a->pGraphExec, exec);
    } else {
        fprintf(stderr, "[HRR] hipGraphInstantiateWithFlags failed: %d (%s)\n",
                r, hipGetErrorString(r));
    }
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipGraphLaunch
// ---------------------------------------------------------------------------
// Payload layout (after 32-byte EventHeader):
//   ret(4) graphExec(8) stream(8)
hipError_t playback_hipGraphLaunch(PlaybackContext& ctx,
                                   const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args_hipGraphLaunch*>(payload);
    if (a->ret != hipSuccess) return hipSuccess;  // original call failed — skip

    hipGraphExec_t exec = ctx.translate_graph_exec(a->graphExec);
    if (!exec) {
        if (ctx.verbose)
            fprintf(stderr, "[HRR] hipGraphLaunch: graphExec 0x%llx not found in map\n",
                    (unsigned long long)a->graphExec);
        return hipSuccess;  // non-fatal — exec not yet created
    }

    hipStream_t stream = ctx.translate_stream(a->stream);

    thread_local hipEvent_t tl_g_start = nullptr;
    thread_local hipEvent_t tl_g_stop  = nullptr;
    bool timing_ok = ctx.timing;
    if (timing_ok && !tl_g_start) {
        if (HRR_HIP_CHECK(hipEventCreate(&tl_g_start)) != hipSuccess ||
            HRR_HIP_CHECK(hipEventCreate(&tl_g_stop))  != hipSuccess) {
            tl_g_start = tl_g_stop = nullptr;
            timing_ok = false;
        } else {
            std::unique_lock lk(ctx.map_mutex);
            ctx.owned_timing_events.push_back(tl_g_start);
            ctx.owned_timing_events.push_back(tl_g_stop);
        }
    }
    if (timing_ok)
        timing_ok = (HRR_HIP_CHECK(hipEventRecord(tl_g_start, stream)) == hipSuccess);

    hipError_t r = hipGraphLaunch(exec, stream);

    if (timing_ok)
        timing_ok = (HRR_HIP_CHECK(hipEventRecord(tl_g_stop, stream)) == hipSuccess);

    if (r != hipSuccess) {
        fprintf(stderr, "[HRR] hipGraphLaunch failed: %d (%s) exec=0x%llx stream=0x%llx\n",
                r, hipGetErrorString(r),
                (unsigned long long)a->graphExec, (unsigned long long)a->stream);
        return r;
    }

    ctx.graphs_launched.fetch_add(1, std::memory_order_relaxed);

    if (timing_ok)
        timing_ok = (HRR_HIP_CHECK(hipEventSynchronize(tl_g_stop)) == hipSuccess);
    if (timing_ok) {
        float ms = 0.f;
        if (HRR_HIP_CHECK(hipEventElapsedTime(&ms, tl_g_start, tl_g_stop)) == hipSuccess) {
            std::unique_lock lk(ctx.map_mutex);
            ctx.total_graph_ms += ms;
        }
    }

    if (ctx.verbose)
        fprintf(stderr, "[HRR] hipGraphLaunch: exec 0x%llx on stream 0x%llx -> OK\n",
                (unsigned long long)a->graphExec, (unsigned long long)a->stream);
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: event create/destroy
// ---------------------------------------------------------------------------
// hipEventCreate:            ret(4) event(8)
// hipEventCreateWithFlags:   ret(4) event(8) flags(4)
// hipEventDestroy:           ret(4) event(8)

hipError_t playback_hipEventCreate(PlaybackContext& ctx,
                                   const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipEventCreate*>(pl);
    hipEvent_t e = nullptr;
    hipError_t r = hipEventCreate(&e);
    if (r == hipSuccess) ctx.record_event(a->event, e);
    return r;
}

hipError_t playback_hipEventCreateWithFlags(PlaybackContext& ctx,
                                            const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipEventCreateWithFlags*>(pl);
    hipEvent_t e  = nullptr;
    hipError_t r  = hipEventCreateWithFlags(&e, a->flags);
    if (r == hipSuccess) ctx.record_event(a->event, e);
    return r;
}

hipError_t playback_hipEventDestroy(PlaybackContext& ctx,
                                    const uint8_t* pl) {
    const auto* a  = reinterpret_cast<const hrr_args_hipEventDestroy*>(pl);
    hipEvent_t event = ctx.translate_event(a->event);
    hipError_t r = hipSuccess;
    if (event) r = hipEventDestroy(event);
    ctx.remove_event(a->event);
    return r;
}

// Capture only records hipEventQuery / hipStreamQuery when they returned
// hipSuccess (see hip_capture_generated.cpp).  Replay drives the API trace
// faster than the original CPU often did relative to GPU completion, so the
// same call can transiently return hipErrorNotReady (600).  Spin until
// hipSuccess to match the captured observable return.
static hipError_t replay_query_until_success(hipError_t (*once)(void*), void* arg) {
    int spin = 0;
    for (;;) {
        hipError_t r = once(arg);
        if (r == hipSuccess)
            return hipSuccess;
        if (r != hipErrorNotReady)
            return r;
        if (++spin < 1000)
            std::this_thread::yield();
        else
            std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}

struct replay_event_query_ctx {
    hipEvent_t event;
};

static hipError_t replay_event_query_once(void* p) {
    auto* c = static_cast<replay_event_query_ctx*>(p);
    return hipEventQuery(c->event);
}

hipError_t playback_hipEventQuery(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipEventQuery*>(pl);
    replay_event_query_ctx c{ctx.translate_event(a->event)};
    return replay_query_until_success(replay_event_query_once, &c);
}

struct replay_stream_query_ctx {
    hipStream_t stream;
};

static hipError_t replay_stream_query_once(void* p) {
    auto* c = static_cast<replay_stream_query_ctx*>(p);
    return hipStreamQuery(c->stream);
}

static hipError_t replay_stream_query_spt_once(void* p) {
    auto* c = static_cast<replay_stream_query_ctx*>(p);
    return hipStreamQuery_spt(c->stream);
}

hipError_t playback_hipStreamQuery(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamQuery*>(pl);
    replay_stream_query_ctx c{(hipStream_t)ctx.translate_stream(a->stream)};
    return replay_query_until_success(replay_stream_query_once, &c);
}

hipError_t playback_hipStreamQuery_spt(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamQuery_spt*>(pl);
    replay_stream_query_ctx c{(hipStream_t)ctx.translate_stream(a->stream)};
    return replay_query_until_success(replay_stream_query_spt_once, &c);
}

// ---------------------------------------------------------------------------
// Manual playback: hipMemcpy3D / hipMemcpy3DAsync
// ---------------------------------------------------------------------------

// Shared D2H validation logic for all 3D memcpy variants.
// Copies byte_count bytes from src_live (device) into a host buffer, then
// validates against the expected blob stored at d2h_hash_lo/hi.
// `tag` names the calling API in the diagnostics. It has no default: several
// APIs share this body, and the messages here are the only thing that tells
// them apart in a replay log.
static hipError_t replay_memcpy3d_d2h(PlaybackContext& ctx,
                                       void* src_live, size_t byte_count,
                                       uint64_t d2h_hash_lo, uint64_t d2h_hash_hi,
                                       hipStream_t stream, bool is_async,
                                       const char* tag) {
    std::vector<uint8_t> actual(byte_count ? byte_count : 1);
    hipError_t r;
    if (is_async) {
        r = hipMemcpyAsync(actual.data(), src_live, byte_count,
                           hipMemcpyDeviceToHost, stream);
        // A recorded default stream translates to nullptr, and that is still the
        // stream this readback was issued on, so sync unconditionally or the
        // comparison below races the copy. Propagating the sync failure keeps a
        // dead device from being reported as a data mismatch. Both match the
        // sibling 2D path in replay_memcpy2d().
        if (r == hipSuccess) r = hipStreamSynchronize(stream);
    } else {
        r = hipMemcpy(actual.data(), src_live, byte_count, hipMemcpyDeviceToHost);
    }
    if (r != hipSuccess) {
        fprintf(stderr, "[HRR] %s D2H: device readback failed: %d (%s)\n",
                tag, r, hipGetErrorString(r));
        ctx.note_d2h_fail(hrr_dispatch_seq);
        return r;
    }
    if (!ctx.validate_d2h || !(d2h_hash_lo || d2h_hash_hi))
        return hipSuccess;  // no expected blob — just execute, no comparison

    ctx.d2h_attempted++;
    size_t blob_sz = 0;
    const void* expected = ctx.load_blob(d2h_hash_lo, d2h_hash_hi, &blob_sz);
    if (!expected) {
        fprintf(stderr, "[HRR] %s D2H validate FAIL: expected blob not found in archive\n", tag);
        ctx.note_d2h_fail(hrr_dispatch_seq);
        return hipSuccess;
    }
    size_t cmp_sz = std::min(byte_count, blob_sz);
    hrr_d2h_validate(ctx, tag, hrr_dispatch_seq, actual.data(),
                     static_cast<const uint8_t*>(expected), cmp_sz);
    return hipSuccess;
}

hipError_t playback_hipMemcpy3D(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpy3D*>(pl);
    hipMemcpy3DParms parms{};
    std::memcpy(&parms, a->parms_bytes, sizeof(parms));

    if (parms.kind == hipMemcpyHostToDevice && a->blob_hash_lo != 0) {
        size_t blob_sz = 0;
        const void* blob = ctx.load_blob(a->blob_hash_lo, a->blob_hash_hi, &blob_sz);
        if (blob) parms.srcPtr.ptr = const_cast<void*>(blob);
        parms.dstPtr.ptr = ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.dstPtr.ptr));
        hipError_t r = hipMemcpy3D(&parms);
        if (r == hipSuccess)
            r = hrr_sync_after_replayed_h2d(ctx, "replayed 3D H2D memcpy");
        return r;
    }
    if (parms.kind == hipMemcpyDeviceToHost) {
        uint64_t src_rec = reinterpret_cast<uint64_t>(parms.srcPtr.ptr);
        void* src_live = ctx.translate_ptr(src_rec);
        if (!src_live) {
            fprintf(stderr, "[HRR] hipMemcpy3D D2H validate FAIL: src 0x%llx not mapped — pointer translation bug\n",
                    (unsigned long long)src_rec);
            ctx.d2h_attempted++;
            ctx.note_d2h_fail(hrr_dispatch_seq);
            return hipSuccess;
        }
        size_t byte_count = parms.extent.width * parms.extent.height * parms.extent.depth;
        return replay_memcpy3d_d2h(ctx, src_live, byte_count,
                                   a->d2h_hash_lo, a->d2h_hash_hi,
                                   nullptr, false, "hipMemcpy3D");
    }
    // D2D: translate both pointers
    parms.srcPtr.ptr = ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.srcPtr.ptr));
    parms.dstPtr.ptr = ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.dstPtr.ptr));
    return hipMemcpy3D(&parms);
}

hipError_t playback_hipMemcpy3DAsync(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpy3DAsync*>(pl);
    hipMemcpy3DParms parms{};
    std::memcpy(&parms, a->parms_bytes, sizeof(parms));
    hipStream_t stream = ctx.translate_stream(a->stream);

    if (parms.kind == hipMemcpyHostToDevice && a->blob_hash_lo != 0) {
        size_t blob_sz = 0;
        const void* blob = ctx.load_blob(a->blob_hash_lo, a->blob_hash_hi, &blob_sz);
        if (blob) parms.srcPtr.ptr = const_cast<void*>(blob);
        parms.dstPtr.ptr = ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.dstPtr.ptr));
        hipError_t r = hipMemcpy3DAsync(&parms, stream);
        if (r == hipSuccess)
            r = hrr_sync_after_replayed_h2d(ctx, "replayed 3D async H2D memcpy");
        return r;
    }
    if (parms.kind == hipMemcpyDeviceToHost) {
        uint64_t src_rec = reinterpret_cast<uint64_t>(parms.srcPtr.ptr);
        void* src_live = ctx.translate_ptr(src_rec);
        if (!src_live) {
            fprintf(stderr, "[HRR] hipMemcpy3DAsync D2H validate FAIL: src 0x%llx not mapped — pointer translation bug\n",
                    (unsigned long long)src_rec);
            ctx.d2h_attempted++;
            ctx.note_d2h_fail(hrr_dispatch_seq);
            return hipSuccess;
        }
        size_t byte_count = parms.extent.width * parms.extent.height * parms.extent.depth;
        return replay_memcpy3d_d2h(ctx, src_live, byte_count,
                                   a->d2h_hash_lo, a->d2h_hash_hi,
                                   stream, true, "hipMemcpy3DAsync");
    }
    // D2D: translate both pointers
    parms.srcPtr.ptr = ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.srcPtr.ptr));
    parms.dstPtr.ptr = ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.dstPtr.ptr));
    return hipMemcpy3DAsync(&parms, stream);
}

// ---------------------------------------------------------------------------
// Manual playback: hipDrvMemcpy3D / hipDrvMemcpy3DAsync / hipDrvMemcpy2DUnaligned
// Driver-style struct copies. Mirror hipMemcpy3D: reconstruct the struct from
// the inline bytes, translate the embedded device pointers (srcDevice/dstDevice)
// via the alloc map, load the H2D blob into srcHost, or validate D2H against the
// expected blob. Keyed off srcMemoryType/dstMemoryType. Host and device memory
// types are in scope; array-typed rects are declined (see below).
// ---------------------------------------------------------------------------

// Host-side byte footprint of a driver-copy rect, measured from the host base
// pointer. Mirrors capture's drvmemcpy_host_byte_count() and, underneath it,
// amd::BufferRect::create(): the rect spans
//   z*slice + y*row + x  ..  + (depth-1)*slice + (height-1)*row + width
// Replay substitutes the captured blob for srcHost while keeping the recorded
// pitches and offsets, so the blob must be at least this large or the runtime
// strides past its end.
static size_t drvmemcpy_host_bytes(size_t pitch, size_t pitch_height,
                                   size_t x, size_t y, size_t z,
                                   size_t width, size_t height, size_t depth) {
    if (width == 0 || height == 0 || depth == 0) return 0;
    size_t row = (pitch != 0) ? pitch : width;
    if (row < width) row = width;
    size_t slice = pitch * pitch_height;
    if (slice < row * height) slice = row * height;  // 0 => runtime default
    return z * slice + y * row + x + (depth - 1) * slice + row * (height - 1) + width;
}

// Resolve the H2D source blob for a driver copy. Returns nullptr when there is
// nothing faithful to substitute: no blob recorded, or a blob smaller than the
// recorded source rect (an archive captured before the blob-footprint fix).
// Skipping matches replay_memcpy2d's H2D policy: never fall back to the stale
// capture-time host VA, and never hand the runtime a short buffer to stride off.
static const void* drvmemcpy_h2d_src_blob(PlaybackContext& ctx, const char* api,
                                          uint64_t hash_lo, uint64_t hash_hi,
                                          size_t need) {
    size_t blob_sz = 0;
    const void* blob = (hash_lo || hash_hi)
                           ? ctx.load_blob(hash_lo, hash_hi, &blob_sz)
                           : nullptr;
    if (!blob) {
        fprintf(stderr, "[HRR] %s H2D: no blob to substitute, skipped\n", api);
        return nullptr;
    }
    if (blob_sz < need) {
        fprintf(stderr,
                "[HRR] %s H2D: blob covers %zu of the %zu bytes the recorded rect "
                "spans, skipped\n", api, blob_sz, need);
        return nullptr;
    }
    return blob;
}

// Array-typed rects are out of scope for these APIs, so decline them up front
// rather than let them reach a branch that cannot describe them. An array is
// addressed through srcArray / dstArray, and the matching srcDevice / dstDevice
// is unset: the device-to-host branch below would translate that unset pointer
// and report the resulting null as a translation bug, and the device-to-device
// branch would translate two unset pointers. hipMemoryType has no separate
// texture enumerator (a texture is read through the array it is bound to), so
// this covers the texture case too. Warned once per API, like the no-op
// handlers, so a replay with many such copies does not spam stderr.
static bool drvmemcpy_declines_array_rect(const char* api, bool& warned,
                                          hipMemoryType src_type, hipMemoryType dst_type) {
    if (src_type != hipMemoryTypeArray && dst_type != hipMemoryTypeArray) return false;
    if (!warned) {
        warned = true;
        fprintf(stderr, "[HRR] %s: array memory type is not replayed by HRR; the copy is "
                        "skipped and results may differ from capture.\n", api);
    }
    return true;
}

// Shared body for hipDrvMemcpy3D / hipDrvMemcpy3DAsync.
static hipError_t replay_drvmemcpy3d(PlaybackContext& ctx, HIP_MEMCPY3D& parms,
                                     const char* api, uint64_t blob_hash_lo,
                                     uint64_t blob_hash_hi, uint64_t d2h_hash_lo,
                                     uint64_t d2h_hash_hi, hipStream_t stream,
                                     bool is_async) {
    if (parms.srcMemoryType == hipMemoryTypeHost) {
        if (parms.dstMemoryType == hipMemoryTypeHost) {
            // Host-to-host touches no device state, and the recorded dstHost VA
            // is meaningless at replay (writing through it would be a wild
            // store). Nothing to reproduce.
            fprintf(stderr, "[HRR] %s: host-to-host copy, skipped\n", api);
            return hipSuccess;
        }
        size_t need = drvmemcpy_host_bytes(parms.srcPitch, parms.srcHeight,
                                           parms.srcXInBytes, parms.srcY, parms.srcZ,
                                           parms.WidthInBytes, parms.Height, parms.Depth);
        const void* blob = drvmemcpy_h2d_src_blob(ctx, api, blob_hash_lo, blob_hash_hi, need);
        if (!blob) return hipSuccess;
        parms.srcHost = blob;
        parms.dstDevice = reinterpret_cast<hipDeviceptr_t>(
            ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.dstDevice)));
        hipError_t r = is_async ? hipDrvMemcpy3DAsync(&parms, stream)
                                : hipDrvMemcpy3D(&parms);
        if (r == hipSuccess)
            r = hrr_sync_after_replayed_h2d(ctx, is_async ? "replayed driver 3D async H2D memcpy"
                                                          : "replayed driver 3D H2D memcpy");
        return r;
    }
    if (parms.dstMemoryType == hipMemoryTypeHost) {
        uint64_t src_rec = reinterpret_cast<uint64_t>(parms.srcDevice);
        void* src_live = ctx.translate_ptr(src_rec);
        if (!src_live) {
            fprintf(stderr, "[HRR] %s D2H validate FAIL: src 0x%llx not mapped - "
                            "pointer translation bug\n",
                    api, (unsigned long long)src_rec);
            ctx.d2h_attempted++;
            ctx.note_d2h_fail(hrr_dispatch_seq);
            return hipSuccess;
        }
        size_t byte_count = parms.WidthInBytes * parms.Height * parms.Depth;
        return replay_memcpy3d_d2h(ctx, src_live, byte_count, d2h_hash_lo, d2h_hash_hi,
                                   stream, is_async, api);
    }
    parms.srcDevice = reinterpret_cast<hipDeviceptr_t>(
        ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.srcDevice)));
    parms.dstDevice = reinterpret_cast<hipDeviceptr_t>(
        ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.dstDevice)));
    return is_async ? hipDrvMemcpy3DAsync(&parms, stream) : hipDrvMemcpy3D(&parms);
}

hipError_t playback_hipDrvMemcpy3D(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipDrvMemcpy3D*>(pl);
    HIP_MEMCPY3D parms{};
    std::memcpy(&parms, a->drv3d_bytes, sizeof(parms));
    static bool array_warned = false;
    if (drvmemcpy_declines_array_rect("hipDrvMemcpy3D", array_warned,
                                      parms.srcMemoryType, parms.dstMemoryType))
        return hipSuccess;
    return replay_drvmemcpy3d(ctx, parms, "hipDrvMemcpy3D",
                              a->blob_hash_lo, a->blob_hash_hi,
                              a->d2h_hash_lo, a->d2h_hash_hi,
                              nullptr, /*is_async=*/false);
}

hipError_t playback_hipDrvMemcpy3DAsync(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipDrvMemcpy3DAsync*>(pl);
    HIP_MEMCPY3D parms{};
    std::memcpy(&parms, a->drv3d_bytes, sizeof(parms));
    static bool array_warned = false;
    if (drvmemcpy_declines_array_rect("hipDrvMemcpy3DAsync", array_warned,
                                      parms.srcMemoryType, parms.dstMemoryType))
        return hipSuccess;
    return replay_drvmemcpy3d(ctx, parms, "hipDrvMemcpy3DAsync",
                              a->blob_hash_lo, a->blob_hash_hi,
                              a->d2h_hash_lo, a->d2h_hash_hi,
                              ctx.translate_stream(a->stream), /*is_async=*/true);
}

// The three spellings of the driver 2D copy — hipDrvMemcpy2DUnaligned,
// hipMemcpyParam2D and hipMemcpyParam2DAsync — take the same hip_Memcpy2D and
// need the same treatment, so `issue` is the only thing that varies: it runs
// the rebuilt descriptor through whichever entry point was recorded.
template <typename T, typename Issue>
static hipError_t replay_drvmemcpy2d(PlaybackContext& ctx, const T* a,
                                     const char* api, hipStream_t stream,
                                     bool is_async, Issue&& issue) {
    hip_Memcpy2D parms{};
    std::memcpy(&parms, a->drv2d_bytes, sizeof(parms));

    static bool array_warned = false;
    if (drvmemcpy_declines_array_rect(api, array_warned,
                                      parms.srcMemoryType, parms.dstMemoryType))
        return hipSuccess;

    if (parms.srcMemoryType == hipMemoryTypeHost) {
        if (parms.dstMemoryType == hipMemoryTypeHost) {
            fprintf(stderr, "[HRR] %s: host-to-host copy, skipped\n", api);
            return hipSuccess;
        }
        // The runtime widens hip_Memcpy2D to a HIP_MEMCPY3D with Depth == 1 and
        // srcHeight == 0, defaulting the pitch to x + WidthInBytes. See
        // hip::getDrvMemcpy3DDesc().
        size_t pitch = parms.srcPitch ? parms.srcPitch
                                      : parms.srcXInBytes + parms.WidthInBytes;
        size_t need = drvmemcpy_host_bytes(pitch, /*pitch_height=*/0, parms.srcXInBytes,
                                           parms.srcY, /*z=*/0, parms.WidthInBytes,
                                           parms.Height, /*depth=*/1);
        const void* blob = drvmemcpy_h2d_src_blob(ctx, api, a->blob_hash_lo,
                                                  a->blob_hash_hi, need);
        if (!blob) return hipSuccess;
        parms.srcHost = blob;
        parms.dstDevice = reinterpret_cast<hipDeviceptr_t>(
            ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.dstDevice)));
        hipError_t r = issue(&parms);
        if (r == hipSuccess)
            r = hrr_sync_after_replayed_h2d(ctx, "replayed driver 2D H2D memcpy");
        return r;
    }
    if (parms.dstMemoryType == hipMemoryTypeHost) {
        uint64_t src_rec = reinterpret_cast<uint64_t>(parms.srcDevice);
        void* src_live = ctx.translate_ptr(src_rec);
        if (!src_live) {
            fprintf(stderr, "[HRR] %s D2H validate FAIL: src 0x%llx not mapped - "
                            "pointer translation bug\n",
                    api, (unsigned long long)src_rec);
            ctx.d2h_attempted++;
            ctx.note_d2h_fail(hrr_dispatch_seq);
            return hipSuccess;
        }
        size_t byte_count = parms.WidthInBytes * parms.Height;
        return replay_memcpy3d_d2h(ctx, src_live, byte_count,
                                   a->d2h_hash_lo, a->d2h_hash_hi,
                                   stream, is_async, api);
    }
    parms.srcDevice = reinterpret_cast<hipDeviceptr_t>(
        ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.srcDevice)));
    parms.dstDevice = reinterpret_cast<hipDeviceptr_t>(
        ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.dstDevice)));
    return issue(&parms);
}

hipError_t playback_hipDrvMemcpy2DUnaligned(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipDrvMemcpy2DUnaligned*>(pl);
    return replay_drvmemcpy2d(ctx, a, "hipDrvMemcpy2DUnaligned", nullptr,
                              /*is_async=*/false,
                              [](const hip_Memcpy2D* p) {
                                  return hipDrvMemcpy2DUnaligned(p);
                              });
}

hipError_t playback_hipMemcpyParam2D(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyParam2D*>(pl);
    return replay_drvmemcpy2d(ctx, a, "hipMemcpyParam2D", nullptr,
                              /*is_async=*/false,
                              [](const hip_Memcpy2D* p) {
                                  return hipMemcpyParam2D(p);
                              });
}

hipError_t playback_hipMemcpyParam2DAsync(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyParam2DAsync*>(pl);
    hipStream_t stream = ctx.translate_stream(a->stream);
    return replay_drvmemcpy2d(ctx, a, "hipMemcpyParam2DAsync", stream,
                              /*is_async=*/true,
                              [stream](const hip_Memcpy2D* p) {
                                  return hipMemcpyParam2DAsync(p, stream);
                              });
}

// ---------------------------------------------------------------------------
// Manual playback: hipMemcpy3DBatchAsync
//
// Each operand of each op is a tagged union: a pointer with a layout and a
// location hint, or an array handle. Which member holds the recorded address
// therefore depends on the tag beside it, which is why the op list is restored
// here rather than by a ptr_members rewrite in the generator.
// ---------------------------------------------------------------------------

static void translate_batch_operand(PlaybackContext& ctx,
                                    hipMemcpy3DOperand& operand) {
    if (operand.type == hipMemcpyOperandTypePointer) {
        operand.op.ptr.ptr = ctx.translate_ptr(
            reinterpret_cast<uint64_t>(operand.op.ptr.ptr));
    } else {
        operand.op.array.array = ctx.translate_array(
            reinterpret_cast<uint64_t>(operand.op.array.array));
    }
}

hipError_t playback_hipMemcpy3DBatchAsync(PlaybackContext& ctx,
                                          const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpy3DBatchAsync*>(pl);
    if (!a->opList_present || a->opList_n == 0) return hipSuccess;

    uint32_t n = a->opList_n;
    if (n > 16u) n = 16u;
    std::vector<hipMemcpy3DBatchOp> ops(n);
    std::memcpy(ops.data(), a->opList_bytes, n * sizeof(hipMemcpy3DBatchOp));
    for (auto& op : ops) {
        translate_batch_operand(ctx, op.src);
        translate_batch_operand(ctx, op.dst);
    }

    size_t fail_idx = 0;
    return hipMemcpy3DBatchAsync(n, ops.data(), &fail_idx,
                                 static_cast<unsigned long long>(a->flags),
                                 ctx.translate_stream(a->stream));
}

// ---------------------------------------------------------------------------
// Manual playback: hipMemcpy2D / hipMemcpy2DAsync
//
// H2D: the recorded host `src` VA is meaningless at replay; substitute the
//      captured blob (laid out with the recorded `spitch`) and copy into the
//      translated device `dst`.
// D2H: read the device `src` back with the recorded pitches and validate against
//      the captured expected-output blob.
// ---------------------------------------------------------------------------

static size_t memcpy2d_host_bytes(uint64_t pitch, uint64_t width, uint64_t height) {
    if (height == 0 || width == 0) return 0;
    if (pitch < width) pitch = width;
    return static_cast<size_t>(pitch * (height - 1) + width);
}

template <typename T>
static hipError_t replay_memcpy2d(PlaybackContext& ctx, const T* a,
                                   hipStream_t stream, bool is_async) {
    const auto kind   = static_cast<hipMemcpyKind>(a->kind);
    const size_t dpitch = static_cast<size_t>(a->dpitch);
    const size_t spitch = static_cast<size_t>(a->spitch);
    const size_t width  = static_cast<size_t>(a->width);
    const size_t height = static_cast<size_t>(a->height);

    if (kind == hipMemcpyHostToDevice) {
        void* dst = ctx.translate_ptr(a->dst);
        size_t blob_sz = 0;
        const void* blob = (a->blob_hash_lo || a->blob_hash_hi)
                               ? ctx.load_blob(a->blob_hash_lo, a->blob_hash_hi, &blob_sz)
                               : nullptr;
        if (!blob) {
            // No captured source data — nothing faithful to write. Skip rather
            // than copy from a stale capture-time host VA.
            fprintf(stderr, "[HRR] hipMemcpy2D%s H2D: no blob to substitute — skipped\n",
                    is_async ? "Async" : "");
            return hipSuccess;
        }
        hipError_t r = hipSuccess;
        if (is_async)
            r = hipMemcpy2DAsync(dst, dpitch, blob, spitch, width, height,
                                 hipMemcpyHostToDevice, stream);
        else
            r = hipMemcpy2D(dst, dpitch, blob, spitch, width, height,
                            hipMemcpyHostToDevice);
        if (r == hipSuccess)
            r = hrr_sync_after_replayed_h2d(ctx, is_async ? "replayed 2D async H2D memcpy"
                                                          : "replayed 2D H2D memcpy");
        return r;
    }

    if (kind == hipMemcpyDeviceToHost) {
        void* src = ctx.translate_ptr(a->src);
        if (!src) {
            fprintf(stderr, "[HRR] hipMemcpy2D%s D2H validate FAIL: src 0x%llx not mapped\n",
                    is_async ? "Async" : "", (unsigned long long)a->src);
            ctx.d2h_attempted++;
            ctx.note_d2h_fail(hrr_dispatch_seq);
            return hipSuccess;
        }
        size_t n = memcpy2d_host_bytes(a->dpitch, a->width, a->height);
        std::vector<uint8_t> actual(n ? n : 1);
        hipError_t r;
        if (is_async) {
            r = hipMemcpy2DAsync(actual.data(), dpitch, src, spitch, width, height,
                                 hipMemcpyDeviceToHost, stream);
            if (r == hipSuccess) r = hipStreamSynchronize(stream);
        } else {
            r = hipMemcpy2D(actual.data(), dpitch, src, spitch, width, height,
                            hipMemcpyDeviceToHost);
        }
        if (r != hipSuccess) {
            fprintf(stderr, "[HRR] hipMemcpy2D%s D2H: device readback failed: %d (%s)\n",
                    is_async ? "Async" : "", r, hipGetErrorString(r));
            ctx.note_d2h_fail(hrr_dispatch_seq);
            return r;
        }
        if (!ctx.validate_d2h || !(a->d2h_hash_lo || a->d2h_hash_hi))
            return hipSuccess;
        ctx.d2h_attempted++;
        size_t blob_sz = 0;
        const void* expected = ctx.load_blob(a->d2h_hash_lo, a->d2h_hash_hi, &blob_sz);
        if (!expected) {
            fprintf(stderr, "[HRR] hipMemcpy2D D2H validate FAIL: expected blob not found\n");
            ctx.note_d2h_fail(hrr_dispatch_seq);
            return hipSuccess;
        }
        size_t cmp_sz = std::min(n, blob_sz);
        hrr_d2h_validate(ctx, "2D", hrr_dispatch_seq, actual.data(),
                         static_cast<const uint8_t*>(expected), cmp_sz);
        return hipSuccess;
    }

    // D2D / H2H: translate both ends (host ptrs translate to themselves-as-null
    // and fall through to the recorded value, matching the generated behavior).
    void* dst = ctx.translate_ptr(a->dst);
    void* src = ctx.translate_ptr(a->src);
    if (!dst) dst = reinterpret_cast<void*>(a->dst);
    if (!src) src = reinterpret_cast<void*>(a->src);
    if (is_async)
        return hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
    return hipMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

hipError_t playback_hipMemcpy2D(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpy2D*>(pl);
    return replay_memcpy2d(ctx, a, nullptr, /*is_async=*/false);
}

hipError_t playback_hipMemcpy2DAsync(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpy2DAsync*>(pl);
    hipStream_t stream = ctx.translate_stream(a->stream);
    return replay_memcpy2d(ctx, a, stream, /*is_async=*/true);
}

hipError_t playback_hipMemcpy3D_spt(PlaybackContext& ctx, const uint8_t* pl) {
    return playback_hipMemcpy3D(ctx, pl);
}

hipError_t playback_hipMemcpy3DAsync_spt(PlaybackContext& ctx, const uint8_t* pl) {
    return playback_hipMemcpy3DAsync(ctx, pl);
}

// ---------------------------------------------------------------------------
// Manual playback: hipArrayCreate / hipArray3DCreate
// ---------------------------------------------------------------------------

hipError_t playback_hipArrayCreate(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipArrayCreate*>(pl);
    HIP_ARRAY_DESCRIPTOR desc{};
    std::memcpy(&desc, a->array_desc_bytes, sizeof(desc));
    hipArray_t arr = nullptr;
    hipError_t r = hipArrayCreate(&arr, &desc);
    if (r == hipSuccess) ctx.record_array(a->pHandle, arr);
    if (hrr_replayed_recorded_error(ctx, "hipArrayCreate", a->ret, r))
        return hipSuccess;
    return r;
}

hipError_t playback_hipArray3DCreate(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipArray3DCreate*>(pl);
    HIP_ARRAY3D_DESCRIPTOR desc{};
    std::memcpy(&desc, a->array3d_desc_bytes, sizeof(desc));
    hipArray_t arr = nullptr;
    hipError_t r = hipArray3DCreate(&arr, &desc);
    if (r == hipSuccess) ctx.record_array(a->array, arr);
    if (hrr_replayed_recorded_error(ctx, "hipArray3DCreate", a->ret, r))
        return hipSuccess;
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipFreeArray — skip if handle not in array_map
// ---------------------------------------------------------------------------
hipError_t playback_hipFreeArray(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipFreeArray*>(pl);
    hipArray_t arr = ctx.translate_array(a->array);
    if (!arr) return hipSuccess;  // nooped alloc (hipMallocArray noop)
    return hipFreeArray(arr);
}

// ---------------------------------------------------------------------------
// Manual playback: hipStreamSetAttribute
// ---------------------------------------------------------------------------

hipError_t playback_hipStreamSetAttribute(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamSetAttribute*>(pl);
    hipStream_t stream = ctx.translate_stream(a->stream);
    hipStreamAttrValue val{};
    std::memcpy(&val, a->stream_attr_bytes, sizeof(val));
    return hipStreamSetAttribute(stream, static_cast<hipStreamAttrID>(a->attr), &val);
}

// ---------------------------------------------------------------------------
// Manual playback: hipMemGetAllocationGranularity
// ---------------------------------------------------------------------------

hipError_t playback_hipMemGetAllocationGranularity(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemGetAllocationGranularity*>(pl);
    hipMemAllocationProp prop{};
    std::memcpy(&prop, a->alloc_prop_bytes, sizeof(prop));
    size_t granularity = 0;
    return hipMemGetAllocationGranularity(&granularity, &prop,
                                          static_cast<hipMemAllocationGranularity_flags>(a->option));
}

// ---------------------------------------------------------------------------
// Manual playback: hipMemPoolSetAccess / hipMemSetAccess
// ---------------------------------------------------------------------------

hipError_t playback_hipMemPoolSetAccess(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemPoolSetAccess*>(pl);
    hipMemPool_t pool = ctx.translate_mempool(a->mem_pool);
    if (!pool) return hipSuccess;
    hipMemAccessDesc desc{};
    std::memcpy(&desc, a->access_desc_bytes, sizeof(desc));
    return hipMemPoolSetAccess(pool, &desc, 1);
}

hipError_t playback_hipMemSetAccess(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemSetAccess*>(pl);
    // VMM reserved address — look in vmm_va_map first, then fall back to alloc_map
    void* ptr = ctx.translate_vmm_va(a->ptr);
    if (!ptr) ptr = ctx.translate_ptr(a->ptr);
    if (!ptr) return hipSuccess;  // VA not found — address space not rebuilt yet, skip
    hipMemAccessDesc desc{};
    std::memcpy(&desc, a->access_desc_bytes, sizeof(desc));
    return hipMemSetAccess(ptr, static_cast<size_t>(a->size), &desc, 1);
}

// ---------------------------------------------------------------------------
// Manual playback: Virtual Memory Management (VMM) address/allocation APIs
// ---------------------------------------------------------------------------
// These APIs require tracking: recorded VA -> live VA, and recorded handle ->
// live hipMemGenericAllocationHandle_t.  The generated shims cannot do this
// because they discard output pointers and pass stale handles as-is.

hipError_t playback_hipMemAddressReserve(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemAddressReserve*>(pl);
    uint64_t rec_ptr = a->ptr;  // recorded output pointer-to-pointer; contains the reserved VA
    // Interpret a->ptr as the *value* of the reserved VA (stored by generator as uint64_t ptr)
    // The recorded ptr field stores the *pointer* output, which at capture time held the VA.
    // We use it as the recorded-VA key.
    void* live_va = nullptr;
    hipError_t r = hipMemAddressReserve(&live_va,
                                        static_cast<size_t>(a->size),
                                        static_cast<size_t>(a->alignment),
                                        nullptr,  // hint addr — don't try to match capture VA
                                        static_cast<unsigned long long>(a->flags));
    if (r == hipSuccess && live_va) {
        std::unique_lock lk(ctx.map_mutex);
        ctx.vmm_va_map[rec_ptr] = {live_va, static_cast<size_t>(a->size)};
    }
    return r;
}

hipError_t playback_hipMemAddressFree(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemAddressFree*>(pl);
    void* live_va = ctx.translate_vmm_va(a->devPtr);
    if (!live_va) return hipSuccess;  // already freed or not tracked
    hipError_t r = hipMemAddressFree(live_va, static_cast<size_t>(a->size));
    if (r == hipSuccess) {
        std::unique_lock lk(ctx.map_mutex);
        ctx.vmm_va_map.erase(a->devPtr);
    }
    return r;
}

hipError_t playback_hipMemCreate(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemCreate*>(pl);
    uint64_t rec_handle = a->handle;  // recorded output handle
    // The allocation property is carried inline (DEREF_FIELDS). It used to be
    // hardcoded to Pinned on device 0, which builds the wrong topology without
    // saying so for a heap that maps cross-node peers.
    hipMemAllocationProp prop{};
    if (a->prop_present) {
        std::memcpy(&prop, a->prop_bytes, sizeof(prop));
    } else {
        prop.type          = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id   = 0;
    }
    // A recorded device ordinal that does not exist here would otherwise be
    // answered by the runtime with a bare error code, or worse, silently
    // satisfied from the wrong device.
    if (prop.location.type == hipMemLocationTypeDevice) {
        int ndev = 0;
        (void)hipGetDeviceCount(&ndev);
        if (prop.location.id >= ndev) {
            fprintf(stderr,
                    "[HRR] hipMemCreate: the recording allocated on device %d "
                    "and this replay has %d device(s) — refusing to allocate "
                    "somewhere else\n", prop.location.id, ndev);
            return hipErrorInvalidDevice;
        }
    }
    hipMemGenericAllocationHandle_t live_handle{};
    hipError_t r = hipMemCreate(&live_handle, static_cast<size_t>(a->size), &prop,
                                static_cast<unsigned long long>(a->flags));
    if (r == hipSuccess) {
        std::unique_lock lk(ctx.map_mutex);
        ctx.vmm_handle_map[rec_handle] = live_handle;
    }
    return r;
}

hipError_t playback_hipMemRelease(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemRelease*>(pl);
    hipMemGenericAllocationHandle_t live = ctx.translate_vmm_handle(a->handle);
    if (!live) return hipSuccess;
    hipError_t r = hipMemRelease(live);
    if (r == hipSuccess) {
        std::unique_lock lk(ctx.map_mutex);
        ctx.vmm_handle_map.erase(a->handle);
    }
    return r;
}

hipError_t playback_hipMemMap(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemMap*>(pl);
    void* live_va = ctx.translate_vmm_va(a->ptr);
    if (!live_va) return hipSuccess;  // VA not tracked, skip
    hipMemGenericAllocationHandle_t live_handle = ctx.translate_vmm_handle(a->handle);
    if (!live_handle) return hipSuccess;  // handle not tracked, skip
    return hipMemMap(live_va,
                     static_cast<size_t>(a->size),
                     static_cast<size_t>(a->offset),
                     live_handle,
                     static_cast<unsigned long long>(a->flags));
}

hipError_t playback_hipMemUnmap(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemUnmap*>(pl);
    void* live_va = ctx.translate_vmm_va(a->ptr);
    if (!live_va) return hipSuccess;
    return hipMemUnmap(live_va, static_cast<size_t>(a->size));
}

// ---------------------------------------------------------------------------
// Manual playback: hipStreamBatchMemOp
//
// The op list is carried inline (DEREF_FIELDS). Every entry's address is a
// device pointer recorded in the capturing process, and which union member
// holds it depends on the op type — so the rewrite is done here rather than by
// a generic ptr_members pass.
// ---------------------------------------------------------------------------
hipError_t playback_hipStreamBatchMemOp(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamBatchMemOp*>(pl);
    const uint32_t inline_max =
        static_cast<uint32_t>(sizeof(a->paramArray_bytes) /
                              sizeof(hipStreamBatchMemOpParams));
    uint32_t n = a->paramArray_n > inline_max ? inline_max : a->paramArray_n;
    if (!a->paramArray_present || n == 0) return hipSuccess;

    std::vector<hipStreamBatchMemOpParams> ops(n);
    std::memcpy(ops.data(), a->paramArray_bytes,
                static_cast<size_t>(n) * sizeof(hipStreamBatchMemOpParams));

    for (auto& op : ops) {
        hipDeviceptr_t* addr = nullptr;
        switch (op.operation) {
            case hipStreamMemOpWaitValue32:
            case hipStreamMemOpWaitValue64:
                addr = &op.waitValue.address;  break;
            case hipStreamMemOpWriteValue32:
            case hipStreamMemOpWriteValue64:
                addr = &op.writeValue.address; break;
            default:
                // Barrier and flush ops carry no address.
                continue;
        }
        void* live = ctx.translate_ptr(reinterpret_cast<uint64_t>(*addr));
        if (!live) {
            // A batch the runtime cannot address is rejected whole, so one
            // untranslatable entry would cost the rest of the archive.
            static bool warned = false;
            if (!warned) {
                warned = true;
                fprintf(stderr,
                        "[HRR] hipStreamBatchMemOp: op address 0x%llx is not in "
                        "any recorded allocation — skipping this batch\n",
                        (unsigned long long)reinterpret_cast<uint64_t>(*addr));
            }
            return hipSuccess;
        }
        *addr = live;
        // The alias field is documented as unused on AMD and holds whatever
        // the capturing process left there; a stale address in it is a
        // pointer the runtime must never see.
        if (op.operation == hipStreamMemOpWaitValue32 ||
            op.operation == hipStreamMemOpWaitValue64)
            op.waitValue.alias = nullptr;
        else
            op.writeValue.alias = nullptr;
    }

    return hipStreamBatchMemOp(ctx.translate_stream(a->stream), n, ops.data(),
                               a->flags);
}

// ---------------------------------------------------------------------------
// Manual playback: IPC memory handles
//
// The handle is 64 opaque bytes naming an export in the process that made it.
// Replay re-exports the live allocation and pairs the recorded bytes with the
// live ones, so an import later in the same archive has something to open. An
// import whose export is in another process (the cross-rank case) has no such
// pairing and says so rather than opening a handle from a dead process.
// ---------------------------------------------------------------------------
hipError_t playback_hipIpcGetMemHandle(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipIpcGetMemHandle*>(pl);
    void* live_ptr = ctx.translate_ptr(a->devPtr);
    if (!live_ptr) {
        fprintf(stderr,
                "[HRR] hipIpcGetMemHandle: recorded 0x%llx is not in any live "
                "allocation — skipping the export\n",
                (unsigned long long)a->devPtr);
        return hipSuccess;
    }
    hipIpcMemHandle_t live{};
    hipError_t r = hipIpcGetMemHandle(&live, live_ptr);
    if (r != hipSuccess) return r;
    if (a->handle_present)
        ctx.record_ipc_handle(
            PlaybackContext::ipc_key(a->handle_bytes, sizeof(live)),
            PlaybackContext::ipc_key(&live, sizeof(live)));
    return hipSuccess;
}

hipError_t playback_hipIpcOpenMemHandle(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipIpcOpenMemHandle*>(pl);
    hipIpcMemHandle_t handle{};
    const std::string live_bytes = ctx.translate_ipc_handle(
        PlaybackContext::ipc_key(a->handle_bytes, sizeof(handle)));
    if (live_bytes.size() != sizeof(handle)) {
        fprintf(stderr,
                "[HRR] hipIpcOpenMemHandle: no replayed export matches this "
                "handle. It was exported by another process, which a "
                "single-archive replay does not reproduce — skipping the "
                "import; anything reading through this pointer will differ "
                "from the recording.\n");
        return hipSuccess;
    }
    std::memcpy(&handle, live_bytes.data(), sizeof(handle));
    void* live = nullptr;
    hipError_t r = hipIpcOpenMemHandle(&live, handle, a->flags);
    if (r == hipSuccess && live)
        ctx.record_alloc(a->devPtr, live, 0, AllocKind::DevicePtrAlias);
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: graph nodes whose parameters are more than one pointer deep
//
// The generated handlers cover every node kind whose parameter struct is flat
// enough for DEREF_FIELDS. These four are not: a kernel node names a host
// function and an argument array, a batch-memory-operation node points at an
// op array, a memory-allocation node hands back an address that only this
// replay's pool can choose, and a driver memcpy node decides between host and
// device operands by a member the archive cannot resolve for the host case.
// ---------------------------------------------------------------------------

// Resolve a recorded dependency list into live nodes. Returns false when a
// dependency names a node this replay never built — the ordering constraint
// would be silently dropped, which is the class of bug the node map exists to
// prevent, so the caller marks the graph incomplete instead.
static bool translate_node_deps(PlaybackContext& ctx, const char* api,
                                const uint8_t* bytes, uint32_t n,
                                uint8_t present,
                                std::vector<hipGraphNode_t>& out) {
    out.clear();
    if (!present || n == 0) return true;
    out.resize(n);
    std::memcpy(out.data(), bytes, static_cast<size_t>(n) * sizeof(hipGraphNode_t));
    for (auto& node : out) {
        const uint64_t rec = reinterpret_cast<uint64_t>(node);
        node = ctx.translate_graph_node(rec);
        if (!node && rec) {
            fprintf(stderr,
                    "[HRR] %s: dependency 0x%llx was never built at replay, so "
                    "this node's ordering constraint cannot be reproduced.\n",
                    api, (unsigned long long)rec);
            return false;
        }
    }
    return true;
}

// What a hand-written graph-node handler returns when the runtime refused the
// call. Unlike the generated shims, these record the event even when the call
// failed at capture, so meeting the same refusal here is fidelity. Any other
// error leaves the graph a node short, which is what the incompleteness flag
// exists to catch at instantiation.
static hipError_t node_add_failed(PlaybackContext& ctx, const char* api,
                                  int32_t recorded_ret, hipError_t r,
                                  uint64_t graph) {
    if (hrr_replayed_recorded_error(ctx, api, recorded_ret, r))
        return hipSuccess;
    ctx.mark_graph_incomplete(graph, api);
    return r;
}

// Rebuild hipKernelNodeParams from the recorded struct plus the kernel tail.
// `storage` and `arg_ptrs` own what the returned params point at and must
// outlive the call that consumes them.
static bool rebuild_kernel_node_params(
    PlaybackContext& ctx, const char* api, const uint8_t* payload,
    size_t fixed_size, const uint8_t* params_bytes, uint8_t params_present,
    hipKernelNodeParams& out,
    std::vector<void*>& arg_ptrs,
    std::vector<std::vector<uint8_t>>& arg_storage) {
    if (!params_present) {
        fprintf(stderr, "[HRR] %s: the node parameters were not recorded.\n", api);
        return false;
    }
    std::memcpy(&out, params_bytes, sizeof(hipKernelNodeParams));

    const auto* hdr = reinterpret_cast<const hrr_event_header*>(payload);
    const uint8_t* p   = payload + fixed_size;
    const uint8_t* end = payload + hdr->payload_length;

    if (p + 2 > end) { fprintf(stderr, "[HRR] %s: truncated kernel tail.\n", api); return false; }
    uint16_t name_len; memcpy(&name_len, p, 2); p += 2;
    if (p + name_len > end) { fprintf(stderr, "[HRR] %s: truncated kernel name.\n", api); return false; }
    std::string kernel_name(reinterpret_cast<const char*>(p), name_len);
    p += name_len;
    if (name_len == 0) {
        fprintf(stderr,
                "[HRR] %s: the recording could not name this node's kernel "
                "(its host function did not resolve at capture).\n", api);
        return false;
    }

    if (p + 16 > end) return false;
    uint64_t co_lo, co_hi;
    memcpy(&co_lo, p, 8); p += 8;
    memcpy(&co_hi, p, 8); p += 8;

    if (p + 2 > end) return false;
    uint16_t num_args; memcpy(&num_args, p, 2); p += 2;

    hipFunction_t func = resolve_kernel_function(ctx, kernel_name, co_lo, co_hi);
    if (!func) return false;

    arg_storage.reserve(num_args);
    decode_kernel_args(ctx, p, end, num_args, kernel_name, arg_ptrs, arg_storage);

    // CLR's GraphKernelNode::getFunc first asks the statically-registered
    // code-object table what host address this is, and falls back to reading
    // the field as a hipFunction_t when that lookup says "not a symbol".
    // The recorded host address belongs to the capturing process, so the
    // resolved function is what goes in.
    out.func         = reinterpret_cast<void*>(func);
    out.kernelParams = arg_ptrs.empty() ? nullptr : arg_ptrs.data();
    out.extra        = nullptr;
    return true;
}

hipError_t playback_hipGraphAddKernelNode(PlaybackContext& ctx,
                                          const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipGraphAddKernelNode*>(pl);
    hipGraph_t graph = ctx.translate_graph(a->graph);
    if (!graph) {
        fprintf(stderr, "[HRR] hipGraphAddKernelNode: graph 0x%llx was never "
                "built at replay; skipping this node.\n",
                (unsigned long long)a->graph);
        return hipSuccess;
    }

    hipKernelNodeParams knp{};
    std::vector<void*> arg_ptrs;
    std::vector<std::vector<uint8_t>> arg_storage;
    std::vector<hipGraphNode_t> deps;
    if (!rebuild_kernel_node_params(ctx, "hipGraphAddKernelNode", pl,
                                    sizeof(*a), a->pNodeParams_bytes,
                                    a->pNodeParams_present, knp, arg_ptrs,
                                    arg_storage) ||
        !translate_node_deps(ctx, "hipGraphAddKernelNode",
                             a->pDependencies_bytes, a->pDependencies_n,
                             a->pDependencies_present, deps)) {
        ctx.mark_graph_incomplete(a->graph, "hipGraphAddKernelNode");
        return hipSuccess;
    }

    hipGraphNode_t node = nullptr;
    hipError_t r = hipGraphAddKernelNode(&node, graph,
                                         deps.empty() ? nullptr : deps.data(),
                                         deps.size(), &knp);
    if (r != hipSuccess)
        return node_add_failed(ctx, "hipGraphAddKernelNode", a->ret, r,
                               a->graph);
    ctx.record_graph_node(a->pGraphNode, node);
    return hipSuccess;
}

hipError_t playback_hipGraphKernelNodeSetParams(PlaybackContext& ctx,
                                                const uint8_t* pl) {
    const auto* a =
        reinterpret_cast<const hrr_args_hipGraphKernelNodeSetParams*>(pl);
    hipGraphNode_t node = ctx.translate_graph_node(a->node);
    if (!node) return hipSuccess;  // the node was never built; already reported

    hipKernelNodeParams knp{};
    std::vector<void*> arg_ptrs;
    std::vector<std::vector<uint8_t>> arg_storage;
    if (!rebuild_kernel_node_params(ctx, "hipGraphKernelNodeSetParams", pl,
                                    sizeof(*a), a->pNodeParams_bytes,
                                    a->pNodeParams_present, knp, arg_ptrs,
                                    arg_storage))
        return hipSuccess;
    hipError_t r = hipGraphKernelNodeSetParams(node, &knp);
    return hrr_replayed_recorded_error(ctx, "hipGraphKernelNodeSetParams",
                                       a->ret, r) ? hipSuccess : r;
}

hipError_t playback_hipGraphExecKernelNodeSetParams(PlaybackContext& ctx,
                                                    const uint8_t* pl) {
    const auto* a =
        reinterpret_cast<const hrr_args_hipGraphExecKernelNodeSetParams*>(pl);
    hipGraphExec_t exec = ctx.translate_graph_exec(a->hGraphExec);
    hipGraphNode_t node = ctx.translate_graph_node(a->node);
    if (!exec || !node) return hipSuccess;

    hipKernelNodeParams knp{};
    std::vector<void*> arg_ptrs;
    std::vector<std::vector<uint8_t>> arg_storage;
    if (!rebuild_kernel_node_params(ctx, "hipGraphExecKernelNodeSetParams", pl,
                                    sizeof(*a), a->pNodeParams_bytes,
                                    a->pNodeParams_present, knp, arg_ptrs,
                                    arg_storage))
        return hipSuccess;
    hipError_t r = hipGraphExecKernelNodeSetParams(exec, node, &knp);
    return hrr_replayed_recorded_error(ctx, "hipGraphExecKernelNodeSetParams",
                                       a->ret, r) ? hipSuccess : r;
}

// Rebuild hipBatchMemOpNodeParams: the struct is recorded inline, the op array
// follows as a tail, and each op's address is translated the same way
// hipStreamBatchMemOp's is.
static bool rebuild_batch_memop_params(
    PlaybackContext& ctx, const char* api, const uint8_t* payload,
    size_t fixed_size, const uint8_t* params_bytes, uint8_t params_present,
    hipBatchMemOpNodeParams& out,
    std::vector<hipStreamBatchMemOpParams>& ops) {
    if (!params_present) {
        fprintf(stderr, "[HRR] %s: the node parameters were not recorded.\n", api);
        return false;
    }
    std::memcpy(&out, params_bytes, sizeof(hipBatchMemOpNodeParams));

    const auto* hdr = reinterpret_cast<const hrr_event_header*>(payload);
    const uint8_t* p   = payload + fixed_size;
    const uint8_t* end = payload + hdr->payload_length;
    if (p + 8 > end) return false;
    uint32_t n = 0, stride = 0;
    memcpy(&n, p, 4); p += 4;
    memcpy(&stride, p, 4); p += 4;
    if (n && (stride != sizeof(hipStreamBatchMemOpParams) ||
              static_cast<size_t>(end - p) < static_cast<size_t>(n) * stride)) {
        fprintf(stderr,
                "[HRR] %s: the op array was recorded with a %u-byte entry "
                "(this build expects %zu).\n",
                api, stride, sizeof(hipStreamBatchMemOpParams));
        return false;
    }

    ops.assign(n, hipStreamBatchMemOpParams{});
    if (n) memcpy(ops.data(), p, static_cast<size_t>(n) * stride);
    for (auto& op : ops) {
        hipDeviceptr_t* addr = nullptr;
        switch (op.operation) {
            case hipStreamMemOpWaitValue32:
            case hipStreamMemOpWaitValue64:
                addr = &op.waitValue.address;  break;
            case hipStreamMemOpWriteValue32:
            case hipStreamMemOpWriteValue64:
                addr = &op.writeValue.address; break;
            default:
                continue;  // barrier / flush carry no address
        }
        void* live = ctx.translate_ptr(reinterpret_cast<uint64_t>(*addr));
        if (!live) {
            fprintf(stderr,
                    "[HRR] %s: op address 0x%llx is not in any recorded "
                    "allocation.\n", api,
                    (unsigned long long)reinterpret_cast<uint64_t>(*addr));
            return false;
        }
        *addr = live;
        // The alias member is unused on AMD and still holds a capture-time
        // address; the runtime must never see it.
        if (op.operation == hipStreamMemOpWaitValue32 ||
            op.operation == hipStreamMemOpWaitValue64)
            op.waitValue.alias = nullptr;
        else
            op.writeValue.alias = nullptr;
    }
    // hipGraphAddBatchMemOpNode rejects a null context (hip_graph.cpp:3736),
    // so the recorded capture-time address is swapped for the live one rather
    // than cleared. A recording made with a null context keeps it, and fails
    // here exactly as it failed there.
    out.ctx        = hrr_live_ctx(reinterpret_cast<uint64_t>(out.ctx));
    out.count      = n;
    out.paramArray = ops.empty() ? nullptr : ops.data();
    return true;
}

hipError_t playback_hipGraphAddBatchMemOpNode(PlaybackContext& ctx,
                                              const uint8_t* pl) {
    const auto* a =
        reinterpret_cast<const hrr_args_hipGraphAddBatchMemOpNode*>(pl);
    hipGraph_t graph = ctx.translate_graph(a->hGraph);
    if (!graph) return hipSuccess;

    hipBatchMemOpNodeParams bnp{};
    std::vector<hipStreamBatchMemOpParams> ops;
    std::vector<hipGraphNode_t> deps;
    if (!rebuild_batch_memop_params(ctx, "hipGraphAddBatchMemOpNode", pl,
                                    sizeof(*a), a->nodeParams_bytes,
                                    a->nodeParams_present, bnp, ops) ||
        !translate_node_deps(ctx, "hipGraphAddBatchMemOpNode",
                             a->dependencies_bytes, a->dependencies_n,
                             a->dependencies_present, deps)) {
        ctx.mark_graph_incomplete(a->hGraph, "hipGraphAddBatchMemOpNode");
        return hipSuccess;
    }

    hipGraphNode_t node = nullptr;
    hipError_t r = hipGraphAddBatchMemOpNode(&node, graph,
                                             deps.empty() ? nullptr : deps.data(),
                                             deps.size(), &bnp);
    if (r != hipSuccess)
        return node_add_failed(ctx, "hipGraphAddBatchMemOpNode", a->ret, r,
                               a->hGraph);
    ctx.record_graph_node(a->phGraphNode, node);
    return hipSuccess;
}

hipError_t playback_hipGraphBatchMemOpNodeSetParams(PlaybackContext& ctx,
                                                    const uint8_t* pl) {
    const auto* a =
        reinterpret_cast<const hrr_args_hipGraphBatchMemOpNodeSetParams*>(pl);
    hipGraphNode_t node = ctx.translate_graph_node(a->hNode);
    if (!node) return hipSuccess;
    hipBatchMemOpNodeParams bnp{};
    std::vector<hipStreamBatchMemOpParams> ops;
    if (!rebuild_batch_memop_params(ctx, "hipGraphBatchMemOpNodeSetParams", pl,
                                    sizeof(*a), a->nodeParams_bytes,
                                    a->nodeParams_present, bnp, ops))
        return hipSuccess;
    hipError_t r = hipGraphBatchMemOpNodeSetParams(node, &bnp);
    return hrr_replayed_recorded_error(ctx, "hipGraphBatchMemOpNodeSetParams",
                                       a->ret, r) ? hipSuccess : r;
}

hipError_t playback_hipGraphExecBatchMemOpNodeSetParams(PlaybackContext& ctx,
                                                        const uint8_t* pl) {
    const auto* a =
        reinterpret_cast<const hrr_args_hipGraphExecBatchMemOpNodeSetParams*>(pl);
    hipGraphExec_t exec = ctx.translate_graph_exec(a->hGraphExec);
    hipGraphNode_t node = ctx.translate_graph_node(a->hNode);
    if (!exec || !node) return hipSuccess;
    hipBatchMemOpNodeParams bnp{};
    std::vector<hipStreamBatchMemOpParams> ops;
    if (!rebuild_batch_memop_params(ctx, "hipGraphExecBatchMemOpNodeSetParams",
                                    pl, sizeof(*a), a->nodeParams_bytes,
                                    a->nodeParams_present, bnp, ops))
        return hipSuccess;
    hipError_t r = hipGraphExecBatchMemOpNodeSetParams(exec, node, &bnp);
    return hrr_replayed_recorded_error(
               ctx, "hipGraphExecBatchMemOpNodeSetParams", a->ret, r)
               ? hipSuccess : r;
}

hipError_t playback_hipGraphAddMemAllocNode(PlaybackContext& ctx,
                                            const uint8_t* pl) {
    const auto* a =
        reinterpret_cast<const hrr_args_hipGraphAddMemAllocNode*>(pl);
    hipGraph_t graph = ctx.translate_graph(a->graph);
    if (!graph) return hipSuccess;

    std::vector<hipGraphNode_t> deps;
    if (!a->pNodeParams_present ||
        !translate_node_deps(ctx, "hipGraphAddMemAllocNode",
                             a->pDependencies_bytes, a->pDependencies_n,
                             a->pDependencies_present, deps)) {
        ctx.mark_graph_incomplete(a->graph, "hipGraphAddMemAllocNode");
        return hipSuccess;
    }

    hipMemAllocNodeParams anp{};
    std::memcpy(&anp, a->pNodeParams_bytes, sizeof(anp));
    const uint64_t rec_dptr = reinterpret_cast<uint64_t>(anp.dptr);
    anp.dptr = nullptr;  // written by the call; the recorded value is not ours
    if (anp.accessDescs && anp.accessDescCount) {
        // The descriptor array is a second pointer hop the archive does not
        // carry. Dropping it would quietly give the allocation different peer
        // visibility than the recording had.
        fprintf(stderr,
                "[HRR] hipGraphAddMemAllocNode: the node's %zu access "
                "descriptors were not recorded, so its peer visibility cannot "
                "be reproduced.\n", anp.accessDescCount);
        ctx.mark_graph_incomplete(a->graph, "hipGraphAddMemAllocNode");
        return hipSuccess;
    }
    anp.accessDescs     = nullptr;
    anp.accessDescCount = 0;

    hipGraphNode_t node = nullptr;
    hipError_t r = hipGraphAddMemAllocNode(&node, graph,
                                           deps.empty() ? nullptr : deps.data(),
                                           deps.size(), &anp);
    if (r != hipSuccess)
        return node_add_failed(ctx, "hipGraphAddMemAllocNode", a->ret, r,
                               a->graph);
    ctx.record_graph_node(a->pGraphNode, node);
    // The pool picks the address, so it is not the recorded one; a later free
    // node or memcpy naming the recorded address needs the pairing.
    if (rec_dptr && anp.dptr)
        ctx.record_alloc(rec_dptr, anp.dptr,
                         static_cast<size_t>(anp.bytesize),
                         AllocKind::DevicePtrAlias);
    return hipSuccess;
}

// Translate a recorded HIP_MEMCPY3D in place. A host operand is a payload the
// archive never carried, so it is reported rather than passed through.
static bool translate_drv_memcpy3d(PlaybackContext& ctx, const char* api,
                                   HIP_MEMCPY3D& c) {
    struct Side { const char* what; hipMemoryType type; const void* host;
                  hipDeviceptr_t* dev; hipArray_t* arr; };
    Side sides[2] = {
        {"source",      c.srcMemoryType, c.srcHost, &c.srcDevice, &c.srcArray},
        {"destination", c.dstMemoryType, c.dstHost, &c.dstDevice, &c.dstArray},
    };
    for (const Side& s : sides) {
        if (s.type == hipMemoryTypeHost || s.host) {
            fprintf(stderr,
                    "[HRR] %s: the %s is host memory, whose contents the "
                    "archive does not carry.\n", api, s.what);
            return false;
        }
        if (*s.dev) {
            void* live = ctx.translate_ptr(reinterpret_cast<uint64_t>(*s.dev));
            if (!live) {
                fprintf(stderr,
                        "[HRR] %s: the %s address 0x%llx is not in any recorded "
                        "allocation.\n", api, s.what,
                        (unsigned long long)reinterpret_cast<uint64_t>(*s.dev));
                return false;
            }
            *s.dev = live;
        }
        if (*s.arr)
            *s.arr = ctx.translate_array(reinterpret_cast<uint64_t>(*s.arr));
    }
    return true;
}

hipError_t playback_hipDrvGraphAddMemcpyNode(PlaybackContext& ctx,
                                             const uint8_t* pl) {
    const auto* a =
        reinterpret_cast<const hrr_args_hipDrvGraphAddMemcpyNode*>(pl);
    hipGraph_t graph = ctx.translate_graph(a->hGraph);
    if (!graph) return hipSuccess;

    HIP_MEMCPY3D copy{};
    std::vector<hipGraphNode_t> deps;
    if (!a->copyParams_present) {
        ctx.mark_graph_incomplete(a->hGraph, "hipDrvGraphAddMemcpyNode");
        return hipSuccess;
    }
    std::memcpy(&copy, a->copyParams_bytes, sizeof(copy));
    if (!translate_drv_memcpy3d(ctx, "hipDrvGraphAddMemcpyNode", copy) ||
        !translate_node_deps(ctx, "hipDrvGraphAddMemcpyNode",
                             a->dependencies_bytes, a->dependencies_n,
                             a->dependencies_present, deps)) {
        ctx.mark_graph_incomplete(a->hGraph, "hipDrvGraphAddMemcpyNode");
        return hipSuccess;
    }

    hipGraphNode_t node = nullptr;
    hipError_t r = hipDrvGraphAddMemcpyNode(&node, graph,
                                            deps.empty() ? nullptr : deps.data(),
                                            deps.size(), &copy,
                                            hrr_live_ctx(a->ctx));
    if (r != hipSuccess)
        return node_add_failed(ctx, "hipDrvGraphAddMemcpyNode", a->ret, r,
                               a->hGraph);
    ctx.record_graph_node(a->phGraphNode, node);
    return hipSuccess;
}

hipError_t playback_hipDrvGraphMemcpyNodeSetParams(PlaybackContext& ctx,
                                                   const uint8_t* pl) {
    const auto* a =
        reinterpret_cast<const hrr_args_hipDrvGraphMemcpyNodeSetParams*>(pl);
    hipGraphNode_t node = ctx.translate_graph_node(a->hNode);
    if (!node || !a->nodeParams_present) return hipSuccess;
    HIP_MEMCPY3D copy{};
    std::memcpy(&copy, a->nodeParams_bytes, sizeof(copy));
    if (!translate_drv_memcpy3d(ctx, "hipDrvGraphMemcpyNodeSetParams", copy))
        return hipSuccess;
    hipError_t r = hipDrvGraphMemcpyNodeSetParams(node, &copy);
    return hrr_replayed_recorded_error(ctx, "hipDrvGraphMemcpyNodeSetParams",
                                       a->ret, r) ? hipSuccess : r;
}

hipError_t playback_hipDrvGraphExecMemcpyNodeSetParams(PlaybackContext& ctx,
                                                       const uint8_t* pl) {
    const auto* a =
        reinterpret_cast<const hrr_args_hipDrvGraphExecMemcpyNodeSetParams*>(pl);
    hipGraphExec_t exec = ctx.translate_graph_exec(a->hGraphExec);
    hipGraphNode_t node = ctx.translate_graph_node(a->hNode);
    if (!exec || !node || !a->copyParams_present) return hipSuccess;
    HIP_MEMCPY3D copy{};
    std::memcpy(&copy, a->copyParams_bytes, sizeof(copy));
    if (!translate_drv_memcpy3d(ctx, "hipDrvGraphExecMemcpyNodeSetParams", copy))
        return hipSuccess;
    hipError_t r = hipDrvGraphExecMemcpyNodeSetParams(exec, node, &copy,
                                                      hrr_live_ctx(a->ctx));
    return hrr_replayed_recorded_error(
               ctx, "hipDrvGraphExecMemcpyNodeSetParams", a->ret, r)
               ? hipSuccess : r;
}
