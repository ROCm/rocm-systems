/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr/hrr_host_rect.h"
#include "hrr_reader.h"   // hrr::hash_hex, hrr::read_file_capped
#include "hrr_payload_bounds.h"
#include "hrr_replay_limits.h"

#include <hip/hip_runtime.h>
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif
#include <hip/hip_ext.h>
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <mutex>
#include <set>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>    // EnumProcessModules
#include <process.h>  // _exit
#else
#include <dlfcn.h>    // dlsym
#include <unistd.h>   // _exit
#include <sys/mman.h> // capture-address placement
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif
#endif

// Set by dispatch_event before each handler runs. Kernel-launch handlers wait on it for their turn,
// then unblock the next thread before timing and sync.
thread_local uint64_t hrr_dispatch_seq = 0;

void hrr_note_unreplayable(PlaybackContext& ctx, const char* api,
                           const char* reason) {
    {
        std::unique_lock lk(ctx.map_mutex);
        if (!ctx.unreplayable_apis.emplace(api, reason ? reason : "").second)
            return;
    }
    fprintf(stderr,
            "[HRR] %s: NOT REPLAYABLE — %s. The call is in the archive but its "
            "effect cannot be reproduced here, so it is skipped; this replay is "
            "not a faithful reproduction of the recording.\n",
            api, reason ? reason : "(unspecified)");
}

void* hrr_runtime_symbol(const char* name) {
#ifdef _WIN32
    // The runtime DLL's file name carries the HIP major version, so ask every
    // module in the process rather than hard-coding one.
    HMODULE mods[1024];
    DWORD needed = 0;
    if (!EnumProcessModules(GetCurrentProcess(), mods, sizeof(mods), &needed))
        return nullptr;
    const size_t n = std::min<size_t>(needed / sizeof(HMODULE), 1024);
    for (size_t i = 0; i < n; ++i) {
        if (FARPROC p = GetProcAddress(mods[i], name))
            return reinterpret_cast<void*>(p);
    }
    return nullptr;
#else
    return dlsym(RTLD_DEFAULT, name);
#endif
}

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


static inline hipError_t hrr_hip_check(hipError_t e, const char* call,
                                        const char* file, int line) {
    if (e != hipSuccess)
        fprintf(stderr, "[HRR] HIP error %d (%s): %s (%s:%d)\n",
                e, hipGetErrorString(e), call, file, line);
    return e;
}
#define HRR_HIP_CHECK(call) hrr_hip_check((call), #call, __FILE__, __LINE__)

hipError_t hrr_watchdog_device_sync(PlaybackContext& ctx, const char* what) {
    const unsigned timeout_ms = ctx.sync_watchdog_ms;
    if (timeout_ms == 0)
        return hipDeviceSynchronize();

    auto fut = std::async(std::launch::async, [] { return hipDeviceSynchronize(); });
    if (fut.wait_for(std::chrono::milliseconds(timeout_ms)) ==
        std::future_status::ready) {
        return fut.get();
    }

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


namespace {

static hipError_t hrr_sync_after_replayed_h2d(PlaybackContext& ctx,
                                                const char* what) {
    // Replay substitutes captured blobs for capture-time host pointers. Drain the restore
    // so following kernels see the replayed input bytes.
    if (!hrr_replayed_h2d_needs_drain(ctx.in_graph_capture)) return hipSuccess;
    // Clears a sticky error so only an error from this drain is reported.
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

// *refused is set when the file was not read because of the file cap or an
// allocation failure (not because it is missing): the caller ends the replay.
static std::vector<uint8_t> read_file(const std::string& path, bool* refused = nullptr) {
    std::vector<uint8_t> buf;
    std::string err;
    if (!hrr::read_file_capped(path, buf, &err, refused)) {
        fprintf(stderr, "[HRR] %s\n", err.c_str());
        return {};
    }
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
    return cache_file(key, blob_path(archive_dir, hash_lo, hash_hi), sz_out, "blob cache");
}

const void* PlaybackContext::cache_file(const std::string& key, const std::string& path,
                                        size_t* sz_out, const char* what) const {
    auto* self = const_cast<PlaybackContext*>(this);
    std::error_code ec;
    const uintmax_t size = std::filesystem::file_size(path, ec);
    const uint64_t charge = ec ? 0 : static_cast<uint64_t>(size);
    if (!self->charge_host(charge, what)) return nullptr;
    bool refused = false;
    auto data = read_file(path, &refused);
    if (refused) {
        // Not a missing file: the replay cannot continue faithfully, so end it
        // now rather than after the consuming handler reaches its HIP call.
        fatal_error.store(true, std::memory_order_release);
    }
    if (data.empty()) {
        self->host_budget.release(charge);
        return nullptr;
    }
    std::unique_lock lk(map_mutex);
    auto [it, inserted] = blob_cache_.emplace(key, std::move(data));
    if (!inserted) self->host_budget.release(charge);
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
    return cache_file(key, co_path(archive_dir, hash_lo, hash_hi), sz_out, "code object cache");
}

hipModule_t PlaybackContext::load_module(uint64_t hash_lo, uint64_t hash_hi) {
    std::string hex = hrr::hash_hex(hash_lo, hash_hi);
    {
        std::shared_lock lk(map_mutex);
        auto it = co_modules.find(hex);
        if (it != co_modules.end()) return it->second;
    }

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
    if (kernel_replacements.empty()) return nullptr;

    {
        std::shared_lock lk(map_mutex);
        auto it = replacement_funcs.find(kernel_name);
        if (it != replacement_funcs.end()) return it->second;
    }

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
            printf("[HRR] Replacing kernel '%s' with %s (symbol %s)\n",
                   kernel_name.c_str(), path->c_str(), kernel_name.c_str());
            fflush(stdout);
        }
    }

    std::unique_lock lk(map_mutex);
    replacement_funcs[kernel_name] = func;
    return func;
}

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
//   [+30..31] num_snapshots (uint16_t)
//   per arg:  u8 value_kind, u16 size, <size> bytes data
//             value_kind: 0=scalar, 1=gpu-pointer, 2=hidden,
//                         3=scalar/struct with embedded gpu pointer(s);
//             kind 3 appends u16 n_ptrs then n_ptrs * u16 byte offsets.
//   per snapshot (v10): u64 ptr_handle, u64 offset, u64 length, u64 hash_lo,
//             u64 hash_hi, u8 direction (0)
//   u32 num_attrs, u32 attr stride, num_attrs * stride bytes

// (destination, blob, length) of each copy; blob_cache_ is never evicted.
using HostSnapCopies = std::vector<std::tuple<void*, const void*, size_t>>;
static void host_snapshot_copy(void* data) {
    auto* copies = static_cast<HostSnapCopies*>(data);
    for (auto& [dst, src, len] : *copies) memcpy(dst, src, len);
    delete copies;
}

// A kernel launch's hold on host_free_mu, taken before the launch hands the
// turn on and kept until its launch call returns. Per thread, as the launch is.
static thread_local std::shared_lock<std::shared_mutex> t_launch_hold;

void hrr_launch_hold(PlaybackContext& ctx) {
    t_launch_hold = std::shared_lock<std::shared_mutex>(ctx.host_free_mu);
}

void hrr_launch_release() {
    if (t_launch_hold.owns_lock()) t_launch_hold.unlock();
}

// Write a launch's pinned host snapshots back when its stream reaches it, by a host
// function queued ahead of the kernel. The replay thread does not wait for the stream,
// which may wait on work it has yet to submit.
static hipError_t restore_host_snapshots(PlaybackContext& ctx, const uint8_t* rec,
                                         uint16_t n, hipStream_t stream,
                                         const std::string& kernel_name) {
    hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &cs) != hipSuccess || cs != hipStreamCaptureStatusNone) {
        ctx.host_snapshots_skipped += n;
        return hipSuccess;
    }
    auto copies = std::make_unique<HostSnapCopies>();
    for (uint16_t i = 0; i < n; i++, rec += kHostSnapRecordSize) {
        uint64_t hash[2];
        memcpy(hash, rec + 24, 16);
        // Capture left the bytes out, as it does over HIP_HRR_CAPTURE_MAX_BLOB_MB.
        if (!hash[0] && !hash[1]) {
            fprintf(stderr, "[HRR] '%s': pinned host snapshot %u was captured without its "
                            "contents; refusing the launch\n",
                    compact_kernel_name(kernel_name).c_str(), i);
            return hipErrorNotFound;
        }
        size_t blob_len = 0;
        const void* blob = ctx.load_blob(hash[0], hash[1], &blob_len);
        // Refused for the host memory cap or a read error, not merely absent.
        if (!blob && ctx.fatal_error.load(std::memory_order_acquire))
            return hipErrorUnknown;
        const char* why  = nullptr;
        void* dst = hrr_host_snapshot_target(ctx, rec, blob, blob_len, &why);
        if (!dst) {
            ctx.host_snapshots_refused++;
            fprintf(stderr, "[HRR] '%s': pinned host snapshot %u %s — refused\n",
                    compact_kernel_name(kernel_name).c_str(), i, why);
            continue;
        }
        copies->emplace_back(dst, blob, blob_len);
    }
    if (copies->empty()) return hipSuccess;
    const size_t k = copies->size();
    hipError_t r = HRR_HIP_CHECK(hipLaunchHostFunc(stream, host_snapshot_copy, copies.get()));
    if (r != hipSuccess) return r;
    copies.release();
    ctx.host_snapshots_restored += k;
    return hipSuccess;
}

// ext_global_worksize: grid[] holds global work-item counts (HSA/OpenCL semantics, as
// hipExtModuleLaunchKernel takes them), not workgroup counts.

// Every recorded device pointer a kernel receives passes through this check.
// It flags a pointer inside an annotated segment but in no live block, which HIP cannot see as out of bounds.
// With block guarding on, it relocates the owning block behind a guard page.

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

// Per-launch guard bookkeeping, owned by replay_kernel_launch. Null on the graph
// kernel-node path: a node's arguments outlive the call that built it, so a relocation
// there could never be undone.
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

// Returns the pointer the kernel should actually receive: `live` unchanged unless block
// guarding relocated its owning block. Cheap no-op when no region sidecar was loaded.
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

    if (rls && live && blk_size > 0 && ctx.guard_blocks && !ctx.in_graph_capture)
        live = hrr_block_guard_relocate(ctx, *rls, rec_ptr, live, blk_base, blk_size);
    return live;
}

static hipFunction_t resolve_kernel_function(PlaybackContext& ctx,
                                             const std::string& kernel_name,
                                             uint64_t co_hash_lo,
                                             uint64_t co_hash_hi) {
    hipFunction_t func = nullptr;

    if (!ctx.kernel_replacements.empty())
        func = ctx.resolve_replacement(kernel_name);

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

// Zeroed bytes after each decoded argument; see decode_kernel_args.
static constexpr size_t kArgOverreadPad = 8;
static size_t recorded_arg_size(const std::vector<uint8_t>& storage) {
    return storage.size() - kArgOverreadPad;
}

static void decode_kernel_args(
    PlaybackContext& ctx, const uint8_t*& p, const uint8_t* end,
    uint16_t num_args, const std::string& kernel_name,
    std::vector<void*>& arg_ptrs,
    std::vector<std::vector<uint8_t>>& arg_storage,
    RegionLaunchState* rls = nullptr,
    std::vector<std::tuple<unsigned, uint64_t, void*>>* dbg_ptrs_out = nullptr,
    std::string* dbg_args_out = nullptr) {
    (void)kernel_name;
    std::vector<std::tuple<unsigned, uint64_t, void*>> dbg_sink;
    const bool dbg_dump_ptrs = (dbg_ptrs_out != nullptr);
    auto& dbg_ptrs = dbg_ptrs_out ? *dbg_ptrs_out : dbg_sink;

    // The previous 4-byte scalar argument, kept in case it is the low half of a pointer.
    struct { bool valid = false; uint16_t idx = 0;
             const uint8_t* data = nullptr; size_t store = 0; } prev_half;
    static const bool no_rescan =
        (std::getenv("HIP_HRR_REPLAY_NO_RESCAN") != nullptr);

    // Reject packed-integer false positives.
    static const bool ptr_relax =
        (std::getenv("HIP_HRR_PTR_RELAX") != nullptr);
    auto pointer_like = [&](uint64_t v) -> bool {
        if (ptr_relax) return true;
        return (v & 0xFFFFFFFFULL) >= 0x10000ULL;
    };
    // Is this value shaped like an address at all? Below 0x10000 no device VA ever lands,
    // and a packed integer fails pointer_like.
    auto va_shaped = [&](uint64_t v) -> bool {
        return v >= 0x10000ULL && pointer_like(v);
    };

    for (uint16_t i = 0; i < num_args; i++) {
        if (end - p < 3) return;
        uint8_t  value_kind = *p++;
        uint16_t arg_size;
        memcpy(&arg_size, p, 2); p += 2;
        if (end - p < arg_size) return;

        const uint8_t* data = p;
        p += arg_size;

        // Every argument as capture classified it, for --dump-ptrs. A pointer reaching
        // the GPU untranslated is invisible in the translated-pointer list precisely
        // because nothing translated it, so the raw bytes are what identifies it.
        if (dbg_args_out) {
            char line[128];
            snprintf(line, sizeof(line), "  arg[%u] kind=%u size=%u", i,
                     value_kind, arg_size);
            *dbg_args_out += line;
            if (arg_size >= 8) {
                *dbg_args_out += " words:";
                for (uint16_t off = 0; off + 8 <= arg_size; off += 8) {
                    uint64_t w; memcpy(&w, data + off, 8);
                    snprintf(line, sizeof(line), " 0x%llx", (unsigned long long)w);
                    *dbg_args_out += line;
                }
            }
            // Raw bytes as well: a pointer split across two 4-byte arguments is invisible
            // as words, and those are exactly the arguments nothing translates.
            *dbg_args_out += " bytes:";
            for (uint16_t off = 0; off < arg_size && off < 32; off++) {
                snprintf(line, sizeof(line), " %02x", data[off]);
                *dbg_args_out += line;
            }
            *dbg_args_out += "\n";
        }

        std::vector<uint16_t> ptr_offsets;
        if (value_kind == 3) {
            if (end - p < 2) return;
            uint16_t n_ptrs; memcpy(&n_ptrs, p, 2); p += 2;
            for (uint16_t k = 0; k < n_ptrs; k++) {
                if (end - p < 2) return;
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
            void* live = ctx.lookup_ptr(rec_ptr);
            // Nothing in the archive covers this address. If a producer said a segment
            // lived there, HIP was bypassed for it — back it now and translate into it.
            if (!live && rec_ptr != 0 && ctx.regions_enabled)
                live = ctx.regions.materialize_for(ctx, rec_ptr);

            // A pointer-typed argument may hold a sentinel (e.g. a literal 1). Keep sentinels as-is.
            // A VA-shaped value that resolves nowhere goes in as null so it faults at first use.
            if (!live && rec_ptr != 0) {
                ctx.untranslated_ptr_args.fetch_add(1, std::memory_order_relaxed);
                const bool sentinel = !va_shaped(rec_ptr);
                if (sentinel) live = reinterpret_cast<void*>(rec_ptr);
                static std::mutex mu;
                static std::set<std::string> warned;
                char key[160];
                snprintf(key, sizeof(key), "%s#%u", kernel_name.c_str(), i);
                bool first;
                { std::lock_guard<std::mutex> lk(mu); first = warned.insert(key).second; }
                if (first)
                    fprintf(stderr,
                            "[HRR] '%s' arg[%u]: recorded 0x%llx is in no known "
                            "allocation — %s\n",
                            compact_kernel_name(kernel_name).c_str(), i,
                            (unsigned long long)rec_ptr,
                            sentinel ? "not an address, passing it through unchanged"
                                     : "passing null so it faults at first use");
            } else {
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

            auto try_translate_word = [&](size_t off, const char* src, bool flagged) -> bool {
                if (off + 8 > arg_size) return false;
                uint64_t rec_ptr; memcpy(&rec_ptr, data + off, 8);
                // null, small, or a packed-integer false positive. A word capture flagged
                // inside a pinned host allocation need not look like a device VA: a host
                // address can have small low 32 bits.
                if (!va_shaped(rec_ptr) &&
                    !(flagged && ctx.in_pinned_host(rec_ptr)))
                    return false;
                void* live = ctx.lookup_ptr(rec_ptr);
                // Same bypassed-segment materialisation as a whole-pointer argument: a
                // struct field pointing into memory HIP never saw resolves in no map
                // until the segment is backed.
                if (!live && ctx.regions_enabled)
                    live = ctx.regions.materialize_for(ctx, rec_ptr);
                if (!live) return false;
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
                if (try_translate_word(off, "captured", true))
                    for (int b = 0; b < 8 && static_cast<size_t>(off) + b < arg_size; b++)
                        handled[off + b] = 1;
                else if (ctx.verbose)
                    fprintf(stderr, "[HRR]   arg[%u]: embedded ptr @+%u unresolved — left as-is (possible scalar)\n",
                            i, off);
            }

            // Defensive rescan: the capture-side value-based detector can MISS an
            // embedded pointer.
            static const bool no_rescan =
                (std::getenv("HIP_HRR_REPLAY_NO_RESCAN") != nullptr);
            for (size_t off = 0; !no_rescan && off + 8 <= arg_size; ) {
                if (handled[off]) { off += 1; continue; }
                if (try_translate_word(off, "rescan", false)) off += 8;
                else off += 1;
            }
        } else {
            storage.assign(data, data + arg_size);
            if (ctx.verbose) {
                fprintf(stderr, "[HRR]   arg[%u]: scalar %u bytes = ", i, arg_size);
                for (uint16_t b = 0; b < arg_size && b < 8; b++)
                    fprintf(stderr, "%02x", data[b]);
                if (arg_size > 8) fprintf(stderr, "...");
                if (arg_size == 4) { uint32_t v; memcpy(&v, data, 4); fprintf(stderr, " (u32=%u)", v); }
                if (arg_size == 8) { uint64_t v; memcpy(&v, data, 8); fprintf(stderr, " (u64=%llu)", (unsigned long long)v); }
                fprintf(stderr, "\n");
            }
        }
        storage.resize(storage.size() + kArgOverreadPad, 0);
        arg_ptrs.push_back(storage.data());

        const size_t store_idx = arg_storage.size() - 1;
        if (!no_rescan && value_kind == 0 && arg_size == 4) {
            bool joined = false;
            if (prev_half.valid && prev_half.idx + 1 == i) {
                uint32_t lo, hi;
                memcpy(&lo, prev_half.data, 4);
                memcpy(&hi, data, 4);
                const uint64_t rec = (static_cast<uint64_t>(hi) << 32) | lo;
                void* live = nullptr;
                if (va_shaped(rec)) {
                    live = ctx.lookup_ptr(rec);
                    if (!live && ctx.regions_enabled)
                        live = ctx.regions.materialize_for(ctx, rec);
                    if (live)
                        live = hrr_region_check_ptr(ctx, rls, kernel_name,
                                                    prev_half.idx, rec, live);
                }
                if (live) {
                    const uint64_t lv = reinterpret_cast<uint64_t>(live);
                    const uint32_t l32 = static_cast<uint32_t>(lv);
                    const uint32_t h32 = static_cast<uint32_t>(lv >> 32);
                    memcpy(arg_storage[prev_half.store].data(), &l32, 4);
                    memcpy(arg_storage[store_idx].data(), &h32, 4);
                    if (dbg_dump_ptrs) dbg_ptrs.emplace_back(prev_half.idx, rec, live);
                    static std::mutex mu;
                    static std::set<std::string> warned;
                    char key[160];
                    snprintf(key, sizeof(key), "%s#%u", kernel_name.c_str(),
                             prev_half.idx);
                    bool first;
                    { std::lock_guard<std::mutex> lk(mu); first = warned.insert(key).second; }
                    if (first)
                        fprintf(stderr,
                                "[HRR] '%s' arg[%u]+arg[%u]: two scalar halves "
                                "form 0x%llx, a recorded address the metadata "
                                "does not call a pointer — translated to %p\n",
                                compact_kernel_name(kernel_name).c_str(),
                                prev_half.idx, i, (unsigned long long)rec, live);
                    joined = true;
                }
            }
            prev_half = joined ? decltype(prev_half){}
                               : decltype(prev_half){true, i, data, store_idx};
        } else {
            prev_half.valid = false;
        }
    }
}

// Test seam: decode_kernel_args without the launch's guard bookkeeping.
void hrr_decode_kernel_args(PlaybackContext& ctx, const uint8_t* p,
                            const uint8_t* end, uint16_t num_args,
                            std::vector<void*>& arg_ptrs,
                            std::vector<std::vector<uint8_t>>& arg_storage) {
    decode_kernel_args(ctx, p, end, num_args, "", arg_ptrs, arg_storage);
}

static bool kernel_args_match(const std::vector<size_t>& recorded,
                              const std::vector<size_t>& declared,
                              std::string& why) {
    char buf[160];
    if (recorded.size() < declared.size()) {
        snprintf(buf, sizeof(buf),
                 "the record carries %zu arguments, the kernel declares %zu",
                 recorded.size(), declared.size());
        why = buf;
        return false;
    }
    for (size_t i = 0; i < declared.size(); i++) {
        if (recorded[i] >= declared[i]) continue;
        snprintf(buf, sizeof(buf),
                 "argument %zu is %zu bytes, the kernel declares %zu", i,
                 recorded[i], declared[i]);
        why = buf;
        return false;
    }
    return true;
}

// A short list or an argument narrower than its parameter would over-read replay's own vectors.
static bool kernel_args_fit_signature(
    PlaybackContext& ctx, const char* api, const std::string& kernel_name,
    hipFunction_t func,
    const std::vector<std::vector<uint8_t>>& arg_storage) {
    std::vector<size_t> declared;
    if (ctx.kernel_param_sizes) {
        ctx.kernel_param_sizes(func, declared);
    } else {
        // hipKernelGetParamInfo has no count: on a resolved function it answers
        // hipSuccess for each parameter and hipErrorInvalidValue for the first index past
        // the end.
        for (size_t i = 0;; i++) {
            size_t off = 0, size = 0;
            if (hipKernelGetParamInfo(reinterpret_cast<hipKernel_t>(func), i,
                                      &off, &size) != hipSuccess)
                break;
            declared.push_back(size);
        }
        (void)hipGetLastError();
    }
    std::vector<size_t> recorded;
    recorded.reserve(arg_storage.size());
    for (const auto& a : arg_storage) recorded.push_back(recorded_arg_size(a));
    std::string why;
    if (kernel_args_match(recorded, declared, why)) return true;
    fprintf(stderr,
            "[HRR] %s seq=%llu kernel '%s': %s — launch refused; its arguments "
            "would not cover the kernel's signature\n",
            api, (unsigned long long)hrr_dispatch_seq,
            compact_kernel_name(kernel_name).c_str(), why.c_str());
    return false;
}

static hipError_t replay_kernel_launch(PlaybackContext& ctx, const uint8_t* pl,
                                       bool ext_global_worksize = false,
                                       bool launch_ex = false,
                                       bool cooperative = false) {
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

    // Backward compatibility for recordings made before capture tagged Ext launches.
    // hipBLASLt/Tensile "Cijk_*" StreamK kernels launch via hipExtModuleLaunchKernel, so grid[] holds
    // global work-item counts. HIP_HRR_REPLAY_FORCE_EXT_CIJK=1 reads it that way. Delete once no old archives remain.
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

    if (!ctx.kernel_filter.empty() &&
        kernel_name.find(ctx.kernel_filter) == std::string::npos)
        return hipSuccess;

    hipFunction_t func = resolve_kernel_function(ctx, kernel_name,
                                                 co_hash_lo, co_hash_hi);
    if (!func) return hipErrorNotFound;

    std::vector<void*>                arg_ptrs;
    std::vector<std::vector<uint8_t>> arg_storage;
    // Guarded blocks this launch relocated, if block guarding is on. Scoped to
    // the launch so the relocation is undone before the next event runs.
    RegionLaunchState rls;
    const bool dbg_dump_ptrs = (ctx.dump_ptrs_ordinal != 0) || ctx.audit_host_args;
    std::vector<std::tuple<unsigned, uint64_t, void*>> dbg_ptrs;  // (arg_idx, recorded, live)
    std::string dbg_args;
    decode_kernel_args(
        ctx, p, end, num_args, kernel_name, arg_ptrs, arg_storage, &rls,
        dbg_dump_ptrs ? &dbg_ptrs : nullptr, dbg_dump_ptrs ? &dbg_args : nullptr);
    if (!kernel_args_fit_signature(ctx, "kernel launch", kernel_name, func,
                                   arg_storage)) {
        for (const auto& g : rls.guards) hrr_block_guard_teardown(g);
        return hipErrorInvalidValue;
    }

    const uint8_t* snapshots = p;
    if (!hrr_host_snapshots_fit(p, end, num_snapshots)) {
        fprintf(stderr, "[HRR] '%s': %u pinned host snapshot records run past the "
                "event — refused\n", kernel_name.c_str(), num_snapshots);
        ctx.host_snapshots_refused += num_snapshots;
        num_snapshots = 0;
    }
    p += static_cast<size_t>(num_snapshots) * kHostSnapRecordSize;

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
    if (hipError_t sr = num_snapshots ? restore_host_snapshots(ctx, snapshots, num_snapshots,
                                                               stream, kernel_name)
                                      : hipSuccess;
        sr != hipSuccess) {
        for (const auto& g : rls.guards) hrr_block_guard_teardown(g);
        return sr;
    }

    if (ctx.audit_host_args) {
        for (auto& [idx, rec, live] : dbg_ptrs) {
            void* abase = nullptr; size_t asize = 0; uint64_t arec = 0;
            AllocKind akind = AllocKind::Device;
            if (!live || !ctx.live_alloc_of(live, &abase, &asize, &arec, &akind))
                continue;
            if (akind == AllocKind::Device || akind == AllocKind::Managed) continue;
            static std::mutex mu;
            static std::set<std::string> seen;
            std::string key = kernel_name + "#" + std::to_string(idx);
            bool first;
            { std::lock_guard<std::mutex> lk(mu); first = seen.insert(key).second; }
            if (first)
                fprintf(stderr,
                        "[HRR host-audit] kernel #%zu '%s' arg[%u] reads %s "
                        "allocation 0x%llx+%zu, which replay reallocated but "
                        "could not refill\n",
                        kernel_ordinal, compact_kernel_name(kernel_name).c_str(),
                        idx, PlaybackContext::alloc_kind_name(akind),
                        (unsigned long long)arec, asize);
        }
    }

    if (dbg_dump_ptrs && kernel_ordinal == ctx.dump_ptrs_ordinal) {
        fprintf(stderr,
                "[HRR ptr-dump] kernel #%zu \"%s\" recorded->live pointer args:\n",
                kernel_ordinal, compact_kernel_name(kernel_name).c_str());
        for (auto& [idx, rec, live] : dbg_ptrs) {
            void* abase = nullptr; size_t asize = 0; uint64_t arec = 0;
            AllocKind akind = AllocKind::Device;
            if (live && ctx.live_alloc_of(live, &abase, &asize, &arec, &akind))
                fprintf(stderr, "[HRR ptr-dump]   arg[%u] recorded=0x%llx -> live=%p "
                        "(%s allocation 0x%llx+%zu, +%lld)\n", idx,
                        (unsigned long long)rec, live,
                        PlaybackContext::alloc_kind_name(akind),
                        (unsigned long long)arec, asize,
                        (long long)(reinterpret_cast<uint64_t>(live) -
                                    reinterpret_cast<uint64_t>(abase)));
        }
        for (auto& [idx, rec, live] : dbg_ptrs)
            fprintf(stderr, "[HRR ptr-dump]   arg[%u] recorded=0x%llx -> live=%p\n",
                    idx, (unsigned long long)rec, live);
        fprintf(stderr, "[HRR ptr-dump] all recorded arguments as captured:\n%s",
                dbg_args.c_str());
        fflush(stderr);
    }

    // Read back what this kernel is about to read, looking for capture-time addresses.
    if (ctx.scan_args_ordinal && kernel_ordinal == ctx.scan_args_ordinal) {
        (void)hipDeviceSynchronize();
        if (const char* a = std::getenv("HIP_HRR_REPLAY_EXPLAIN_ADDR")) {
            uint64_t v = strtoull(a, nullptr, 0);
            fprintf(stderr, "[HRR arg-scan] 0x%llx is %s\n",
                    (unsigned long long)v, ctx.explain_addr(v).c_str());
            uint64_t probe = 0;
            hipError_t pr = hipMemcpy(&probe, reinterpret_cast<void*>(v),
                                      sizeof(probe), hipMemcpyDeviceToHost);
            fprintf(stderr, "[HRR arg-scan] reading 8 bytes at 0x%llx: %s",
                    (unsigned long long)v, hipGetErrorString(pr));
            if (pr == hipSuccess)
                fprintf(stderr, " (value 0x%llx)", (unsigned long long)probe);
            fprintf(stderr, "\n");
        }
        const size_t cap = ctx.scan_args_bytes;
        const auto ranges = ctx.recorded_ranges();
        std::vector<uint8_t> host(cap);
        fprintf(stderr, "[HRR arg-scan] kernel #%zu \"%s\": reading %zu bytes "
                "behind each pointer argument\n", kernel_ordinal,
                compact_kernel_name(kernel_name).c_str(), cap);
        for (auto& [idx, rec, live] : dbg_ptrs) {
            if (!live) continue;
            // Scan from the start of the enclosing allocation, not from the
            // argument, so bytes ahead of the pointer are covered too.
            void*    abase    = live;
            size_t   asize    = 0;
            uint64_t arec     = rec;
            ctx.live_alloc_of(live, &abase, &asize, &arec);
            size_t n = asize ? std::min(cap, asize) : cap;
            n -= n % 8;
            if (!n) continue;
            if (hipMemcpy(host.data(), abase, n, hipMemcpyDeviceToHost) != hipSuccess) {
                fprintf(stderr, "[HRR arg-scan]   arg[%u]: unreadable\n", idx);
                continue;
            }
            size_t hits = 0;
            for (size_t off = 0; off + 8 <= n; off += 8) {
                uint64_t w; memcpy(&w, host.data() + off, 8);
                uint64_t base = 0; size_t sz = 0;
                if (!PlaybackContext::range_contains(ranges, w, &base, &sz)) continue;
                if (++hits <= 8)
                    fprintf(stderr, "[HRR arg-scan]   arg[%u] allocation 0x%llx "
                            "+%zu holds 0x%llx — recorded allocation 0x%llx+%zu\n",
                            idx, (unsigned long long)arec, off,
                            (unsigned long long)w, (unsigned long long)base, sz);
            }
            if (hits)
                fprintf(stderr, "[HRR arg-scan]   arg[%u]: %zu recorded addresses "
                        "in %zu bytes of allocation 0x%llx\n", idx, hits, n,
                        (unsigned long long)arec);
        }
        fflush(stderr);
    }

    const bool do_timing = ctx.timing && !ctx.in_graph_capture;

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
        // The cooperative entry point takes no `extra`, so the packed-kernarg route below
        // is not available to it. Cooperative kernels are HIP C++ in every case seen
        // here, which is the kernelParams[] path anyway.
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
                uint32_t sz = static_cast<uint32_t>(recorded_arg_size(arg_storage[i]));
                uint32_t align = (sz >= 8) ? 8u : sz ? sz : 1u;
                cursor = (cursor + align - 1) & ~(align - 1);
                koffsets[i] = cursor;
                cursor += sz;
            }
            // Round up to 64-byte alignment; add 256 bytes for hidden arg space.
            uint32_t kbuf_sz = ((cursor + 63) & ~63u) + 256;
            std::vector<uint8_t> kbuf(kbuf_sz + kArgOverreadPad, 0);
            for (size_t i = 0; i < arg_storage.size(); ++i) {
                const auto& s = arg_storage[i];
                const size_t n = recorded_arg_size(s);
                if (koffsets[i] + n <= kbuf_sz)
                    memcpy(kbuf.data() + koffsets[i], s.data(), n);
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

    // The kernel and its snapshot copies are queued: a host free may go ahead.
    // Not held across the syncs below, which can wait on events yet to replay.
    hrr_launch_release();

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

    // Skip the debug sync while a graph capture is active: hipDeviceSynchronize is
    // illegal during stream capture and invalidates it (HIP 901). The original run never
    // synced here either; syncs resume once capture ends.
    if (ctx.sync_after_launch && !ctx.in_graph_capture) {
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


hipError_t playback_hipModuleLaunchKernel(PlaybackContext& ctx,
                                          const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload);
}

hipError_t playback_hipExtModuleLaunchKernel(PlaybackContext& ctx,
                                             const uint8_t* payload) {
    // Captured with HSA/OpenCL semantics: grid[] holds global work-item counts, so replay goes through
    // the matching Ext API to avoid over-launching by a factor of blockDim.
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
    // The stream is in the payload, so the stream-per-thread spelling replays through the same path.
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
    return replay_kernel_launch(ctx, payload, /*ext_global_worksize=*/false,
                                /*launch_ex=*/false, /*cooperative=*/true);
}

bool hrr_is_launch_handler(hipError_t (*fn)(PlaybackContext&, const uint8_t*)) {
    return fn == playback_hipModuleLaunchKernel ||
           fn == playback_hipExtModuleLaunchKernel ||
           fn == playback_hipLaunchKernel ||
           fn == playback_hipLaunchByPtr ||
           fn == playback_hipLaunchKernel_spt ||
           fn == playback_hipLaunchCooperativeKernel ||
           fn == playback_hipLaunchCooperativeKernel_spt ||
           fn == playback_hipDrvLaunchKernelEx ||
           fn == playback_hipLaunchKernelExC ||
           fn == playback_hipModuleLaunchCooperativeKernel;
}


// A well-formed offload bundle starts with one of the two magics the bundler
// emits: the clang text magic, or the compressed-bundle "CCOB" magic.
static bool is_offload_bundle(const void* p, size_t n) {
    if (!p) return false;
    static const char kClangMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
    const size_t kClangMagicLen = sizeof(kClangMagic) - 1;  // 24, no NUL
    if (n >= kClangMagicLen && std::memcmp(p, kClangMagic, kClangMagicLen) == 0) return true;
    if (n >= 4 && std::memcmp(p, "CCOB", 4) == 0) return true;
    return false;
}

hipError_t playback___hipRegisterFatBinary(PlaybackContext& ctx,
                                           const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args___hipRegisterFatBinary*>(payload);
    uint64_t blob_hash_lo = a->blob_hash_lo;
    uint64_t blob_hash_hi = a->blob_hash_hi;

    if (!blob_hash_lo && !blob_hash_hi) return hipSuccess;  // no blob — skip
    if (!a->blob_size) return hipSuccess;

    std::string hex = hrr::hash_hex(blob_hash_lo, blob_hash_hi);

    // Deduplicate: if already loaded (e.g. multiple __hipRegisterFatBinary events for the
    // same binary), skip the load. Binaries already known to hold no code for this GPU
    // are skipped the same way.
    {
        std::shared_lock lk(ctx.map_mutex);
        if (ctx.co_modules.count(hex)) return hipSuccess;
        if (ctx.co_no_device_code.count(hex)) return hipSuccess;
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
        // A bundle with nothing built for the live GPU is expected, not a defect: see
        // co_no_device_code.
        if ((err == hipErrorInvalidImage || err == hipErrorNoBinaryForGpu) &&
            is_offload_bundle(blob, sz)) {
            {
                std::unique_lock lk(ctx.map_mutex);
                ctx.co_no_device_code.insert(hex);
            }
            if (ctx.verbose)
                fprintf(stderr, "[HRR] Fat binary %s holds no code for this GPU — skipped\n",
                        hex.c_str());
            return hipSuccess;
        }
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

// A __device__ global is named in the recording by the host shadow address the compiler
// emitted, which means nothing here.

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


hipError_t playback___hipRegisterVar(PlaybackContext& ctx,
                                     const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args___hipRegisterVar*>(payload);
    if (!a->deviceVar_present || a->deviceVar_bytes[0] == '\0') return hipSuccess;

    // The record's name field need not end in a NUL: copy it out bounded.
    const std::string name_str =
        hrr::bounded_cstr(a->deviceVar_bytes, sizeof(a->deviceVar_bytes));
    const char* name = name_str.c_str();
    size_t live_size = 0;
    void* live = ctx.resolve_symbol_by_name(name, &live_size);
    if (!live) {
        // Replay dispatches events from more than one thread, so the log-once
        // flag has to be atomic even though it only guards a diagnostic.
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
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

// Both take the host shadow address, which the symbol sweep has already tied to a live
// global. Answering from that map also registers the recorded device address the
// capturing process got back, so a later copy against it resolves.

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


static bool dep_array(PlaybackContext& ctx, const char* api, const uint8_t* bytes,
                      size_t field_bytes, uint8_t present, uint32_t n, uint64_t recorded_n,
                      std::vector<hipGraphNode_t>& out, hipGraphNode_t** ptr) {
    *ptr = nullptr;
    out.clear();
    if (!present || n == 0) return true;
    if (!hrr::deps_complete(recorded_n, n)) {
        fprintf(stderr,
                "[HRR] %s: event with sequence number %llu was built with %llu dependencies but its record "
                "holds %u; the others were dropped at capture, so the ordering cannot "
                "be reproduced; refusing the call\n",
                api, (unsigned long long)hrr_dispatch_seq, (unsigned long long)recorded_n, n);
        return false;
    }
    if (!hrr::inline_count_fits(n, field_bytes, sizeof(hipGraphNode_t))) {
        fprintf(stderr,
                "[HRR] %s: event with sequence number %llu records %u dependencies but its record holds "
                "at most %zu; refusing the call\n",
                api, (unsigned long long)hrr_dispatch_seq, n,
                hrr::inline_capacity(field_bytes, sizeof(hipGraphNode_t)));
        return false;
    }
    out.resize(n);
    std::memcpy(out.data(), bytes, static_cast<size_t>(n) * sizeof(hipGraphNode_t));
    for (auto& node : out)
        node = ctx.translate_graph_node(reinterpret_cast<uint64_t>(node));
    *ptr = out.data();
    return true;
}

static hipError_t resolve_symbol_range(PlaybackContext& ctx, uint64_t rec,
                                       uint64_t offset, uint64_t count,
                                       const char* api, void** out) {
    size_t size = 0;
    void* symbol = ctx.translate_symbol(rec, &size);
    if (!symbol) {
        fprintf(stderr,
                "[HRR] %s: symbol 0x%llx is not in this archive's symbol "
                "registry.\n", api, (unsigned long long)rec);
        return hipErrorInvalidSymbol;
    }
    if (offset > size || count > size - offset) {
        fprintf(stderr,
                "[HRR] %s: copy of %llu bytes at offset %llu does not fit "
                "symbol 0x%llx (%zu bytes registered); refusing the call\n",
                api, (unsigned long long)count, (unsigned long long)offset,
                (unsigned long long)rec, size);
        return hipErrorInvalidValue;
    }
    *out = symbol;
    return hipSuccess;
}

static bool host_contents_left_out(const char* api, uint64_t size, uint64_t hash_lo,
                                   uint64_t hash_hi) {
    if (size == 0 || hash_lo || hash_hi) return false;
    fprintf(stderr, "[HRR] %s: %llu bytes of host memory were captured without their "
                    "contents; refusing the call\n",
            api, (unsigned long long)size);
    return true;
}

namespace {

// The source for a to-symbol copy: the recorded blob when the copy came from host memory,
// the translated device address otherwise. Null, said on stderr, when there is neither.
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
    // Capture stores the source by the same rule (memcpy_src_is_host), so a host
    // source with no blob was left out.
    const auto kind = static_cast<hipMemcpyKind>(a->kind);
    hipPointerAttribute_t attr{};
    const bool from_host =
        kind == hipMemcpyHostToDevice || kind == hipMemcpyHostToHost ||
        (kind == hipMemcpyDefault && src &&
         (hipPointerGetAttributes(&attr, src) != hipSuccess ||
          (attr.type != hipMemoryTypeDevice && attr.type != hipMemoryTypeUnified)));
    if (from_host && host_contents_left_out(api, a->count, a->blob_hash_lo, a->blob_hash_hi))
        return nullptr;
    if (!src)
        fprintf(stderr, "[HRR] %s: device source 0x%llx is not mapped.\n", api,
                (unsigned long long)a->src);
    return src;
}

}  // namespace

hipError_t playback_hipGraphAddMemcpyNodeToSymbol(PlaybackContext& ctx,
                                                  const uint8_t* payload) {
    const auto* a =
        reinterpret_cast<const hrr_args_hipGraphAddMemcpyNodeToSymbol*>(payload);
    const auto kind = static_cast<hipMemcpyKind>(a->kind);
    void* symbol = nullptr;
    if (hipError_t e = resolve_symbol_range(ctx, a->symbol, a->offset, a->count,
                                            "hipGraphAddMemcpyNodeToSymbol", &symbol);
        e != hipSuccess)
        return e;
    const void* src = to_symbol_source(ctx, a, "hipGraphAddMemcpyNodeToSymbol");
    if (!src) return hipErrorInvalidValue;

    std::vector<hipGraphNode_t> deps;
    hipGraphNode_t* dep_ptr = nullptr;
    if (!dep_array(ctx, "hipGraphAddMemcpyNodeToSymbol", a->pDependencies_bytes,
                   sizeof(a->pDependencies_bytes), a->pDependencies_present,
                   a->pDependencies_n, a->numDependencies, deps, &dep_ptr)) {
        ctx.mark_graph_incomplete(a->graph, "hipGraphAddMemcpyNodeToSymbol");
        return hipSuccess;
    }
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
    void* symbol = nullptr;
    if (hipError_t e = resolve_symbol_range(ctx, a->symbol, a->offset, a->count,
                                            "hipGraphAddMemcpyNodeFromSymbol", &symbol);
        e != hipSuccess)
        return e;

    void* dst = nullptr;
    if (kind == hipMemcpyDeviceToHost || kind == hipMemcpyHostToHost) {
        dst = ctx.host_landing_buffer(a->dst, static_cast<size_t>(a->count));
        if (!dst) {
            fprintf(stderr,
                    "[HRR] hipGraphAddMemcpyNodeFromSymbol: the recorded host "
                    "destination is null, so this node is not reproduced.\n");
            return hipErrorInvalidValue;
        }
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
    hipGraphNode_t* dep_ptr = nullptr;
    if (!dep_array(ctx, "hipGraphAddMemcpyNodeFromSymbol", a->pDependencies_bytes,
                   sizeof(a->pDependencies_bytes), a->pDependencies_present,
                   a->pDependencies_n, a->numDependencies, deps, &dep_ptr)) {
        ctx.mark_graph_incomplete(a->graph, "hipGraphAddMemcpyNodeFromSymbol");
        return hipSuccess;
    }
    hipGraphNode_t node = nullptr;
    hipError_t r = hipGraphAddMemcpyNode1D(
        &node, ctx.translate_graph(a->graph), dep_ptr, deps.size(), dst,
        static_cast<char*>(symbol) + a->offset, static_cast<size_t>(a->count),
        kind);
    if (r == hipSuccess && a->pGraphNode)
        ctx.record_graph_node(a->pGraphNode, node);
    return r;
}


hipError_t playback_hipGraphMemcpyNodeSetParamsToSymbol(PlaybackContext& ctx,
                                                        const uint8_t* payload) {
    const char* kApi = "hipGraphMemcpyNodeSetParamsToSymbol";
    const auto* a =
        reinterpret_cast<const hrr_args_hipGraphMemcpyNodeSetParamsToSymbol*>(payload);
    void* symbol = nullptr;
    if (hipError_t e = resolve_symbol_range(ctx, a->symbol, a->offset, a->count,
                                            kApi, &symbol);
        e != hipSuccess)
        return e;
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
    void* symbol = nullptr;
    if (hipError_t e = resolve_symbol_range(ctx, a->symbol, a->offset, a->count,
                                            kApi, &symbol);
        e != hipSuccess)
        return e;
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
    void* symbol = nullptr;
    if (hipError_t e = resolve_symbol_range(ctx, a->symbol, a->offset, a->count,
                                            kApi, &symbol);
        e != hipSuccess)
        return e;
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
    void* symbol = nullptr;
    if (hipError_t e = resolve_symbol_range(ctx, a->symbol, a->offset, a->count,
                                            kApi, &symbol);
        e != hipSuccess)
        return e;
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

// The image comes back from its blob and the linker state from the map hipLinkCreate
// fills.

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

    // Recorded option values may point into the dead capture process, so any non-null value is refused.
    for (uint32_t i = 0; i < n_opts; ++i) {
        if (option_values[i]) {
            fprintf(stderr,
                    "[HRR] hipLinkAddData: option %u was recorded with a "
                    "capture-process address (%p) that replay cannot "
                    "reconstruct, so the link is not reproduced.\n",
                    i, option_values[i]);
            return hipErrorInvalidValue;
        }
    }

    // The record's name field need not end in a NUL: copy it out bounded.
    const std::string link_name =
        hrr::bounded_cstr(a->name_bytes, sizeof(a->name_bytes));
    hipError_t r = hipLinkAddData(
        state, static_cast<hipJitInputType>(a->type), const_cast<void*>(image),
        blob_sz,
        a->name_present ? link_name.c_str() : nullptr,
        n_opts, n_opts ? options : nullptr, n_opts ? option_values : nullptr);
    if (hrr_replayed_recorded_error(ctx, "hipLinkAddData", a->ret, r))
        return hipSuccess;
    return r;
}

// Intentional no-op at playback. Function handles are resolved lazily by name at kernel
// launch time: replay_kernel_launch() searches module_map + co_modules + func_cache.

hipError_t playback_hipModuleGetFunction(PlaybackContext& ctx,
                                         const uint8_t* payload) {
    (void)ctx; (void)payload;
    return hipSuccess;
}

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

// Payload: ret(4) ptr(8) size(8) [additional fields for managed/host variants]
// ptr at +4, size at +12

// Pad factor: over-allocate so pool-style kernels that sweep past one tensor do not fault.
// HIP_HRR_REPLAY_ALLOC_PAD_FACTOR (default 1) and HIP_HRR_REPLAY_ALLOC_PAD_MAX (default 1073741824).
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

// Zero-initialise freshly allocated replay device memory. hipMalloc/hipMallocAsync do NOT
// guarantee zeroed memory: AMD only scrubs a physical page on its FIRST allocation (for
// cross-process security).
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

static int hrr_replay_fill_byte() {
    static std::once_flag once;
    static int g_byte = 0;
    std::call_once(once, [] {
        if (const char* e = std::getenv("HIP_HRR_REPLAY_FILL_BYTE")) {
            g_byte = static_cast<int>(strtoul(e, nullptr, 0) & 0xff);
            fprintf(stderr, "[HRR] filling fresh allocations with 0x%02x "
                            "(HIP_HRR_REPLAY_FILL_BYTE)\n", g_byte);
        }
    });
    return g_byte;
}

// hipMalloc is host-synchronous, so the zero-init must finish before the handler returns (ROCM-27985).
// A bare hipMemset is async on the null stream and can race a later H2D restore on a non-blocking stream.
static void hrr_zero_init_alloc(PlaybackContext& ctx, void* live, size_t sz) {
    if (!live || sz == 0) return;  // nothing written, so nothing to order
    if (!hrr_zero_init_needs_drain(hrr_replay_zero_init(), ctx.in_graph_capture))
        return;
    if (hipMemsetAsync(live, hrr_replay_fill_byte(), sz, nullptr) != hipSuccess)
        return;
    (void)hipStreamSynchronize(nullptr);
}


// Make `device` current and return the ordinal to restore afterwards, or -1 if nothing
// was switched.
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

// A segment a producer declared but that HRR never observed. Everything the archive knows
// about it is its base and size: no capture shim ran, so no contents were recorded and no
// allocation event exists.
hipError_t hrr_materialize_region(PlaybackContext& ctx, uint64_t rec_base,
                                  size_t size, int device, void** out_live) {
    if (out_live) *out_live = nullptr;
    if (rec_base == 0 || size == 0) return hipErrorInvalidValue;

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
    (void)ctx.remove_alloc_if_live(rec_base, live);
    if (live) (void)hipFree(live);
}

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

// Reserve `mapped + gran` of VA, back only the first `mapped` bytes, and leave the tail
// span unmapped. Any access past the mapped region traps.
static hipError_t hrr_guard_map(size_t mapped, size_t gran, int device, void* hint,
                                void** out_va, size_t* out_reserved,
                                hipMemGenericAllocationHandle_t* out_handle,
                                bool peers = false) {
    const size_t reserved = mapped + gran;
    hipMemAllocationProp prop{};
    prop.type          = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id   = device;

    void* va = nullptr;
    hipError_t r = hipMemAddressReserve(&va, reserved, 0, hint, 0);
    if (r != hipSuccess || !va) return r != hipSuccess ? r : hipErrorOutOfMemory;
    if (hint && va != hint) { (void)hipMemAddressFree(va, reserved); return hipErrorOutOfMemory; }

    hipMemGenericAllocationHandle_t handle{};
    r = hipMemCreate(&handle, mapped, &prop, 0);
    if (r != hipSuccess) { (void)hipMemAddressFree(va, reserved); return r; }

    r = hipMemMap(va, mapped, 0, handle, 0);
    if (r != hipSuccess) {
        (void)hipMemRelease(handle);
        (void)hipMemAddressFree(va, reserved);
        return r;
    }

    int ndev = 1;
    if (!peers || hipGetDeviceCount(&ndev) != hipSuccess) ndev = 1;
    std::vector<hipMemAccessDesc> desc(1);
    desc[0].location.id = device;
    for (int d = 0; d < ndev; ++d) {
        int can = 0;
        if (d != device && hipDeviceCanAccessPeer(&can, d, device) == hipSuccess && can)
            desc.emplace_back().location.id = d;
    }
    for (auto& a : desc) {
        a.location.type = hipMemLocationTypeDevice;
        a.flags         = hipMemAccessFlagsProtReadWrite;
    }
    r = hipMemSetAccess(va, mapped, desc.data(), desc.size());
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

// hipMalloc and hipMallocAsync land at their recorded address, as a VMM mapping in place
// of the PROT_NONE placeholder hrr_place_hold left there before hipInit. See DESIGN.md,
// "GPU Allocator Address Non-Determinism".

#ifndef _WIN32
// MAP_FIXED_NOREPLACE fails rather than move; a kernel too old to know the flag
// takes the address as a hint, hence the check.
static bool hrr_place_mmap(uint64_t b, uint64_t e) {
    void* want = reinterpret_cast<void*>(b);
    void* p = mmap(want, e - b, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED) return false;
    if (p != want) { munmap(p, e - b); return false; }
    return true;
}
#endif

void hrr_place_hold(PlaybackContext& ctx, const std::vector<HrrVaRange>& plan) {
#ifndef _WIN32
    // A padded allocation would run over its neighbours' recorded addresses.
    if (replay_padded_alloc_size(1) > 1) return;
    std::lock_guard<std::mutex> lk(ctx.place_mutex);
    for (const auto& r : plan)
        if (hrr_place_mmap(r.base, r.end)) ctx.place_held.add(r.base, r.end);
    ctx.place = true;
#else
    (void)ctx; (void)plan;
#endif
}

// Map the granules holding `size` bytes at the recorded address `rec` (a
// stream-ordered allocation need not start on one). False: allocate as before.
static bool hrr_place_alloc(PlaybackContext& ctx, uint64_t rec, size_t size, void** out,
                            bool async = false) {
#ifndef _WIN32
    if (!ctx.place) return false;
    const int      dev  = hrr_current_device();
    const size_t   gran = hrr_vmm_granularity(dev);
    uint64_t base, end;
    if (!hrr_place_range(rec, size, gran, &base, &end)) {
        // hipMalloc(&p, 0) is legal and recorded; it has nothing to place.
        if (size != 0)
            fprintf(stderr,
                    "[HRR] Placement: seq=%llu: refusing 0x%llx + %zu bytes (the range wraps); "
                    "allocating without placement\n",
                    (unsigned long long)hrr_dispatch_seq, (unsigned long long)rec, size);
        ctx.place_fell_back++;
        return false;
    }
    const uint64_t len = end - base;
    bool held;
    {
        std::lock_guard<std::mutex> lk(ctx.place_mutex);
        held = ctx.place_held.take(base, base + len);
        if (held) munmap(reinterpret_cast<void*>(base), len);
    }
    void* va = nullptr;
    size_t reserved = 0;
    hipMemGenericAllocationHandle_t handle{};
    if (held && hrr_guard_map(len, 0, dev, reinterpret_cast<void*>(base), &va, &reserved,
                              &handle, /*peers=*/true) == hipSuccess) {
        {
            std::unique_lock lk(ctx.map_mutex);
            ctx.guard_allocs[reinterpret_cast<void*>(rec)] = {va, reserved, len, handle,
                                                              /*placed=*/true, async};
        }
        ctx.placed++;
        *out = reinterpret_cast<void*>(rec);
        return true;
    }
    if (ctx.verbose)
        fprintf(stderr, "[HRR] Placement: 0x%llx (%zu bytes) fell back: %s\n",
                (unsigned long long)rec, size, held ? "not mapped there" : "not held");
    if (held) {
        (void)hipGetLastError();  // the recorded call succeeded
        std::lock_guard<std::mutex> lk(ctx.place_mutex);
        if (hrr_place_mmap(base, base + len)) ctx.place_held.add(base, base + len);
    }
    ctx.place_fell_back++;
#else
    (void)ctx; (void)rec; (void)size; (void)out; (void)async;
#endif
    return false;
}

static void hrr_place_released(PlaybackContext& ctx, uint64_t b, uint64_t e) {
#ifndef _WIN32
    {
        std::lock_guard<std::mutex> lk(ctx.place_mutex);
        if (hrr_place_mmap(b, e)) ctx.place_held.add(b, e);
    }
    if (ctx.in_graph_capture) return;
    void* p = nullptr;
    if (hipMalloc(&p, size_t(4) << 20) == hipSuccess) (void)hipFree(p);
    else (void)hipGetLastError();
#else
    (void)ctx; (void)b; (void)e;
#endif
}

static bool hrr_is_placed(PlaybackContext& ctx, void* live) {
    std::shared_lock lk(ctx.map_mutex);
    auto it = ctx.guard_allocs.find(live);
    return it != ctx.guard_allocs.end() && it->second.placed;
}

// hipFree waits for the device before it frees hipMalloc memory, though not
// hipMallocAsync memory; unmapping a VMM allocation in its place does not.
static bool hrr_free_needs_sync(PlaybackContext& ctx, void* live) {
    std::shared_lock lk(ctx.map_mutex);
    auto it = ctx.guard_allocs.find(live);
    return it != ctx.guard_allocs.end() && !it->second.async;
}


static hipError_t hrr_guard_alloc(PlaybackContext& ctx, size_t want, void** out) {
    const int    dev    = hrr_current_device();
    const size_t gran   = hrr_vmm_granularity(dev);
    const size_t mapped = ((want + gran - 1) / gran) * gran;

    void* va = nullptr;
    size_t reserved = 0;
    hipMemGenericAllocationHandle_t handle{};
    hipError_t r = hrr_guard_map(mapped, gran, dev, nullptr, &va, &reserved, &handle);
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
    if (g.placed) {
        const uint64_t b = reinterpret_cast<uint64_t>(g.va_base);
        hrr_place_released(ctx, b, b + g.mapped);
    }
    return true;
}

void hrr_free_device_alloc(PlaybackContext& ctx, void* live) {
    if (!live) return;
    if (hrr_guard_free(ctx, live)) return;
    (void)hipFree(live);
}

void hrr_place_reset(PlaybackContext& ctx) {
    std::vector<void*> placed;
    {
        std::shared_lock lk(ctx.map_mutex);
        for (const auto& [va, g] : ctx.guard_allocs)
            if (g.placed) placed.push_back(va);
    }
    for (void* va : placed) {
        hrr_guard_free(ctx, va);
        ctx.remove_alloc(reinterpret_cast<uint64_t>(va));  // placed: live == recorded
    }
    ctx.placed = 0;
    ctx.place_fell_back = 0;
}


static size_t hrr_alignment_of(uint64_t v, size_t cap) {
    if (v == 0) return cap;
    size_t a = static_cast<size_t>(v & (~v + 1));  // v & -v
    return a > cap ? cap : a;
}

// Place `size` bytes of block whose recorded base is `blk_base` against an unmapped
// guard. Fills *blind with the number of bytes between the block's end and the guard — an
// overrun smaller than that is still not caught.
static hipError_t hrr_block_guard_alloc(PlaybackContext& ctx, uint64_t blk_base,
                                        size_t size, HrrBlockGuard* g,
                                        size_t* blind) {
    if (size == 0) return hipErrorInvalidValue;
    const int    dev  = hrr_current_device();
    const size_t gran = hrr_vmm_granularity(dev);

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
    hipError_t r = hrr_guard_map(mapped, gran, dev, nullptr, &va, &reserved, &handle);
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

// Hand this argument a guarded copy of its owning block. One buffer per distinct block
// per launch, so two arguments pointing into the same object still see the same memory as
// they did at capture.
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

    void* live_base = ctx.lookup_ptr(blk_base);
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

// After the launch: did anything run off the end of a guarded block? Sync so the fault is
// attributed to this kernel rather than to whatever event happens to synchronize next.
static hipError_t hrr_block_guard_resolve(PlaybackContext& ctx,
                                          RegionLaunchState& rls,
                                          const std::string& kernel_name,
                                          size_t kernel_ordinal) {
    if (rls.guards.empty()) return hipSuccess;

    if (ctx.in_graph_capture) {
        // hipDeviceSynchronize and blocking D2D hipMemcpy are illegal during stream
        // capture (HIP 901).
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

static void free_replaced_host_alloc(PlaybackContext& ctx, uint64_t rec);

static hipError_t replay_malloc(PlaybackContext& ctx, const uint8_t* pl,
                                bool managed = false) {
    const auto* a = reinterpret_cast<const hrr_args_hipMalloc*>(pl);
    size_t orig_sz = static_cast<size_t>(a->size);
    size_t pad_sz  = replay_padded_alloc_size(orig_sz);
    void* live = nullptr;
    hipError_t r;
    if (managed) {
        free_replaced_host_alloc(ctx, a->ptr);
        if (!ctx.charge_host(pad_sz, "hipMallocManaged")) return hipErrorMemoryAllocation;
    }
    if (!managed && hrr_place_alloc(ctx, a->ptr, pad_sz, &live)) {
        r = hipSuccess;
    } else if (!managed && ctx.guard_segments &&
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
        // deterministic and matches first-touch-zeroed assumptions.
        hrr_zero_init_alloc(ctx, live, pad_sz);
        ctx.record_alloc(a->ptr, live, pad_sz,
                         managed ? AllocKind::Managed : AllocKind::Device);
        if (ctx.verbose && pad_sz > orig_sz)
            fprintf(stderr, "[HRR] hipMalloc 0x%llx: orig=%zu padded=%zu\n",
                    (unsigned long long)a->ptr, orig_sz, pad_sz);
    } else if (managed) {
        ctx.host_budget.release(pad_sz);
    }
    return r;
}

hipError_t playback_hipMalloc(PlaybackContext& ctx, const uint8_t* pl) {
    return replay_malloc(ctx, pl);
}
hipError_t playback_hipMallocManaged(PlaybackContext& ctx, const uint8_t* pl) {
    return replay_malloc(ctx, pl, /*managed=*/true);
}

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
    // Inside a graph capture this is a graph allocation node; not placed.
    const bool placed = !ctx.in_graph_capture &&
                        hrr_place_alloc(ctx, a->dev_ptr, pad_sz, &live, /*async=*/true);
    hipError_t r = placed ? hipSuccess : hipMallocAsync(&live, pad_sz, stream);
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

// hipHostMalloc:  ret(4) ptr(8) size(8) flags(4)
// hipMallocHost:  ret(4) ptr(8) size(8)

static void free_replaced_host_alloc(PlaybackContext& ctx, uint64_t rec) {
    std::unique_lock host_free(ctx.host_free_mu);
    void* host_ptr = nullptr;
    void* reg_buf = nullptr;
    size_t size = 0;
    AllocKind kind = AllocKind::Device;
    {
        std::unique_lock lk(ctx.map_mutex);
        auto it = ctx.alloc_map.find(rec);
        if (it == ctx.alloc_map.end()) return;
        kind = it->second.kind;
        if (kind != AllocKind::HostMalloc && kind != AllocKind::HostRegister &&
            kind != AllocKind::Managed)
            return;
        host_ptr = it->second.live_ptr;
        size = it->second.size;
        ctx.alloc_map.erase(it);
        auto rb = ctx.host_reg_bufs.find(rec);
        if (rb != ctx.host_reg_bufs.end()) {
            reg_buf = rb->second;
            ctx.host_reg_bufs.erase(rb);
        }
    }
    if (kind == AllocKind::HostMalloc) {
        if (hipHostFree(host_ptr) == hipSuccess) ctx.host_budget.release(size);
        return;
    }
    if (kind == AllocKind::Managed) {
        if (hipFree(host_ptr) == hipSuccess) ctx.host_budget.release(size);
        return;
    }
    void* buf = reg_buf ? reg_buf : host_ptr;
    if (hipHostUnregister(buf) == hipSuccess) return;
    // Still registered: the next registration here gets a new block.
    std::unique_lock lk(ctx.map_mutex);
    auto& block = ctx.host_reg_blocks[rec];
    if (block.cur.get() == buf) block.retired.push_back(std::move(block.cur));
}

hipError_t playback_hipHostMalloc(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipHostMalloc*>(pl);
    void* live = nullptr;
    free_replaced_host_alloc(ctx, a->ptr);
    if (!ctx.charge_host(a->size, "hipHostMalloc")) return hipErrorMemoryAllocation;
    hipError_t r = hipHostMalloc(&live, static_cast<size_t>(a->size), a->flags);
    if (r == hipSuccess)
        ctx.record_alloc(a->ptr, live, static_cast<size_t>(a->size), AllocKind::HostMalloc);
    else
        ctx.host_budget.release(a->size);
    // A call that failed the same way at capture is not a replay failure.
    if (r != hipSuccess && a->ret != 0 && static_cast<int32_t>(r) == a->ret) {
        hrr_note_recorded_error(ctx, "hipHostMalloc", a->ret);
        return hipSuccess;
    }
    return r;
}

hipError_t playback_hipMallocHost(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMallocHost*>(pl);
    void* live = nullptr;
    free_replaced_host_alloc(ctx, a->ptr);
    if (!ctx.charge_host(a->size, "hipMallocHost")) return hipErrorMemoryAllocation;
    hipError_t r = hipMallocHost(&live, static_cast<size_t>(a->size));
    if (r == hipSuccess)
        ctx.record_alloc(a->ptr, live, static_cast<size_t>(a->size), AllocKind::HostMalloc);
    else
        ctx.host_budget.release(a->size);
    // A call that failed the same way at capture is not a replay failure.
    if (r != hipSuccess && a->ret != 0 && static_cast<int32_t>(r) == a->ret) {
        hrr_note_recorded_error(ctx, "hipMallocHost", a->ret);
        return hipSuccess;
    }
    return r;
}

// hipHostFree and hipFreeHost differ only in the runtime call. The exclusive lock keeps a
// launch that is queuing a pinned host snapshot for this buffer from racing the free (see
// hrr_launch_hold).
static hipError_t free_host_memory(PlaybackContext& ctx, uint64_t rec_ptr, int32_t recorded_ret,
                                   hipError_t (*free_fn)(void*), const char* api) {
    std::unique_lock host_free(ctx.host_free_mu);
    void* live = ctx.lookup_ptr(rec_ptr);
    if (!live) return hipSuccess;
    hipError_t r = free_fn(live);
    if (r != hipSuccess && recorded_ret != 0 && static_cast<int32_t>(r) == recorded_ret) {
        hrr_note_recorded_error(ctx, api, recorded_ret);
        return hipSuccess;
    }
    if (r == hipSuccess) ctx.remove_alloc(rec_ptr);
    return r;
}

hipError_t playback_hipHostFree(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipHostFree*>(pl);
    return free_host_memory(ctx, a->ptr, a->ret, hipHostFree, "hipHostFree");
}

hipError_t playback_hipFreeHost(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipFreeHost*>(pl);
    return free_host_memory(ctx, a->ptr, a->ret, hipFreeHost, "hipFreeHost");
}

// hipHostRegister recorded a snapshot of the host memory as a blob.

hipError_t playback_hipHostRegister(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipHostRegister*>(pl);
    size_t sz = static_cast<size_t>(a->sizeBytes);
    if (sz == 0) return hipSuccess;
    free_replaced_host_alloc(ctx, a->hostPtr);
    if (host_contents_left_out("hipHostRegister", sz, a->blob_hash_lo, a->blob_hash_hi))
        return hipErrorNotFound;
    void* buf = ctx.host_block(ctx.host_reg_blocks, a->hostPtr, sz,
                               "hipHostRegister backing buffer");
    if (!buf) return hipErrorMemoryAllocation;

    size_t blob_sz = 0;
    const void* blob = ctx.load_blob(a->blob_hash_lo, a->blob_hash_hi, &blob_sz);
    if (blob && blob_sz == sz)
        std::memcpy(buf, blob, sz);
    else
        std::memset(buf, 0, sz);
    if (!blob && ctx.fatal_error.load(std::memory_order_acquire)) {
        // The blob was refused (cap or allocation), not merely absent:
        // do not register zeros in its place.
        return hipErrorUnknown;
    }

    hipError_t r = hipHostRegister(buf, sz, a->flags);
    if (r == hipSuccess) {
        ctx.record_alloc(a->hostPtr, buf, sz, AllocKind::HostRegister);
        std::unique_lock lk(ctx.map_mutex);
        ctx.host_reg_bufs[a->hostPtr] = buf;
    }
    return r;
}

hipError_t playback_hipHostUnregister(PlaybackContext& ctx, const uint8_t* pl) {
    std::unique_lock host_free(ctx.host_free_mu);
    const auto* a = reinterpret_cast<const hrr_args_hipHostUnregister*>(pl);

    // Leave the maps alone until the runtime call succeeds: a failed
    // unregister did not release the registration.
    void* buf = nullptr;
    {
        std::shared_lock lk(ctx.map_mutex);
        auto it = ctx.host_reg_bufs.find(a->hostPtr);
        if (it != ctx.host_reg_bufs.end()) buf = it->second;
    }

    void* live = buf ? buf : ctx.lookup_ptr(a->hostPtr);
    if (!live) return hipSuccess;

    hipError_t r = hipHostUnregister(live);
    if (r != hipSuccess) return r;

    {
        std::unique_lock lk(ctx.map_mutex);
        ctx.host_reg_bufs.erase(a->hostPtr);
    }
    ctx.remove_alloc(a->hostPtr);
    return r;
}

// The generated shim would pass the raw recorded host ptr to the real API, which fails
// because it's a stale captured address.

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
        // For zero-copy host allocations the device pointer equals the host pointer, so
        // a->devPtr collides with the recorded address of the owning
        // hipHostMalloc/hipMalloc entry.
        if (!ctx.has_alloc(a->devPtr))
            ctx.record_alloc(a->devPtr, dev_ptr, 0, AllocKind::DevicePtrAlias);
    }
    return r;
}

// hipFree:       ret(4) ptr(8)
// hipFreeAsync:  ret(4) dev_ptr(8) stream(8)

hipError_t playback_hipFree(PlaybackContext& ctx, const uint8_t* pl) {
    std::unique_lock host_free(ctx.host_free_mu);
    const auto* a = reinterpret_cast<const hrr_args_hipFree*>(pl);
    void* live = ctx.lookup_ptr(a->ptr);
    if (!live) return hipSuccess;
    if (!ctx.in_graph_capture && hrr_free_needs_sync(ctx, live))
        if (hipError_t sr = hipDeviceSynchronize(); sr != hipSuccess) return sr;
    if (hrr_guard_free(ctx, live)) {  // placed or --guard-segments: a VMM mapping
        ctx.remove_alloc(a->ptr);
        return hipSuccess;
    }
    hipError_t r = hipFree(live);
    if (r == hipSuccess) ctx.remove_alloc(a->ptr);
    return r;
}

hipError_t playback_hipFreeAsync(PlaybackContext& ctx, const uint8_t* pl) {
    std::unique_lock host_free(ctx.host_free_mu);
    const auto* a  = reinterpret_cast<const hrr_args_hipFreeAsync*>(pl);
    void*       live   = ctx.lookup_ptr(a->dev_ptr);
    hipStream_t stream = ctx.translate_stream(a->stream);
    if (!live) return hipSuccess;
    if (hrr_is_placed(ctx, live)) {
        // Unmapped at once, after the work queued on its stream. A graph being
        // captured may still use it, so it then stays mapped.
        if (!ctx.in_graph_capture) {
            if (hipError_t sr = hipStreamSynchronize(stream); sr != hipSuccess) return sr;
            hrr_guard_free(ctx, live);
        }
        ctx.remove_alloc(a->dev_ptr);
        return hipSuccess;
    }
    hipError_t r = hipFreeAsync(live, stream);
    if (r == hipSuccess) ctx.remove_alloc(a->dev_ptr);
    return r;
}


// The default oracle is byte-exact. A blob carries no element type, so a guessed tolerance could pass a zeroed buffer.
// Non-associative reductions (atomicAdd, split-K) differ at the ULP level between replays of one recording.
// For those, HIP_HRR_D2H_TOL=<type>:<atol>:<rtol> accepts elements with |actual - expected| <= atol + rtol*|expected|.
bool hrr_parse_d2h_tol(const char* spec, HrrD2HTol* out) {
    static const struct { const char* name; HrrD2HTol::Type type; } kTypes[] = {
        {"f32", HrrD2HTol::kF32}, {"bf16", HrrD2HTol::kBF16},
        {"f16", HrrD2HTol::kF16}, {"f64", HrrD2HTol::kF64}};
    if (!spec) return false;
    const char* colon = std::strchr(spec, ':');
    if (!colon) return false;
    HrrD2HTol t;
    bool known = false;
    for (const auto& k : kTypes)
        if (std::strlen(k.name) == static_cast<size_t>(colon - spec) &&
            std::strncmp(spec, k.name, colon - spec) == 0) {
            t.type = k.type;
            known = true;
        }
    if (!known) return false;
    char* end = nullptr;
    t.atol = std::strtod(colon + 1, &end);
    if (end == colon + 1 || *end != ':') return false;
    const char* r = end + 1;
    t.rtol = std::strtod(r, &end);
    if (end == r || *end != '\0') return false;
    // An infinite tolerance would accept every finite value.
    if (!std::isfinite(t.atol) || !std::isfinite(t.rtol) || t.atol < 0 || t.rtol < 0)
        return false;
    *out = t;
    return true;
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

// Updates ctx counters and prints only on failure. True if the buffer is exact or within HIP_HRR_D2H_TOL.
static bool hrr_d2h_validate(PlaybackContext& ctx, const char* tag, uint64_t seq,
                             const uint8_t* actual, const uint8_t* expected, size_t n) {
    if (n == 0 || memcmp(actual, expected, n) == 0) {
        ctx.d2h_pass++;
        if (ctx.verbose)
            fprintf(stderr, "[HRR] %s D2H validate: %zu bytes OK (exact)\n", tag, n);
        return true;
    }

    const HrrD2HTol& tol = ctx.d2h_tol;
    size_t ndiff_bytes = 0;
    for (size_t i = 0; i < n; i++) if (actual[i] != expected[i]) ndiff_bytes++;
    size_t first_diff = 0;
    while (first_diff < n && actual[first_diff] == expected[first_diff]) ++first_diff;

    auto dec_f32 = [](const uint8_t* p) { float v; memcpy(&v, p, 4); return (double)v; };
    auto dec_f64 = [](const uint8_t* p) { double v; memcpy(&v, p, 8); return v; };
    auto dec_bf16 = [](const uint8_t* p) {
        uint16_t h; memcpy(&h, p, 2); uint32_t u = (uint32_t)h << 16;
        float v; memcpy(&v, &u, 4); return (double)v; };
    auto dec_f16 = [](const uint8_t* p) {
        uint16_t h; memcpy(&h, p, 2); return (double)hrr_half_to_float(h); };

    const char* type_name = nullptr;
    size_t esz = 0;
    switch (tol.type) {
    case HrrD2HTol::kF32:  type_name = "f32";  esz = 4; break;
    case HrrD2HTol::kBF16: type_name = "bf16"; esz = 2; break;
    case HrrD2HTol::kF16:  type_name = "f16";  esz = 2; break;
    case HrrD2HTol::kF64:  type_name = "f64";  esz = 8; break;
    case HrrD2HTol::kExact: break;
    }
    // A buffer that is not a whole number of elements is not of the named type.
    if (esz && n % esz == 0) {
        HrrD2HScan s;
        switch (tol.type) {
        case HrrD2HTol::kF32:  s = hrr_scan_elems(actual, expected, n, 4, dec_f32, tol.atol, tol.rtol); break;
        case HrrD2HTol::kBF16: s = hrr_scan_elems(actual, expected, n, 2, dec_bf16, tol.atol, tol.rtol); break;
        case HrrD2HTol::kF16:  s = hrr_scan_elems(actual, expected, n, 2, dec_f16, tol.atol, tol.rtol); break;
        default:               s = hrr_scan_elems(actual, expected, n, 8, dec_f64, tol.atol, tol.rtol); break;
        }
        if (s.n_bad == 0) {
            ctx.d2h_pass++;
            ctx.d2h_pass_tol++;
            if (ctx.verbose)
                fprintf(stderr,
                        "[HRR] %s D2H validate: %zu bytes ~OK within tol as %s "
                        "(%zu elems differ, max|d|=%.3g maxrel=%.3g, atol=%g rtol=%g)\n",
                        tag, n, type_name, s.n_diff, s.max_abs, s.max_rel,
                        tol.atol, tol.rtol);
            return true;
        }
        ctx.note_d2h_fail(seq);
        fprintf(stderr,
                "[HRR] %s D2H FAIL seq=%llu: %zu bytes, %zu/%zu bytes differ (%.2f%%), "
                "first@%zu (got 0x%02x exp 0x%02x); as %s: %zu/%zu elems exceed "
                "tol (atol=%g rtol=%g), max|d|=%.4g maxrel=%.4g\n",
                tag, (unsigned long long)seq, n, ndiff_bytes, n,
                100.0 * (double)ndiff_bytes / (double)n, first_diff,
                actual[first_diff], expected[first_diff], type_name,
                s.n_bad, s.n_diff, tol.atol, tol.rtol, s.max_abs, s.max_rel);
        return false;
    }

    ctx.note_d2h_fail(seq);
    fprintf(stderr,
            "[HRR] %s D2H FAIL seq=%llu (exact): %zu bytes, %zu/%zu bytes differ, "
            "first@%zu (got 0x%02x exp 0x%02x)\n",
            tag, (unsigned long long)seq, n, ndiff_bytes, n, first_diff,
            actual[first_diff], expected[first_diff]);
    return false;
}

// A linear copy's blob is the whole copy.
static bool hrr_d2h_blob_short(const char* tag, size_t copy_sz, size_t blob_sz) {
    if (blob_sz >= copy_sz) return false;
    fprintf(stderr,
            "[HRR] %s D2H: expected blob holds %zu of the %zu bytes copied (truncated), "
            "not validated\n",
            tag, blob_sz, copy_sz);
    return true;
}

static void* translate_or_materialize(PlaybackContext& ctx, uint64_t rec_addr) {
    void* live = ctx.lookup_ptr(rec_addr);
    if (!live && rec_addr != 0 && ctx.regions_enabled)
        live = ctx.regions.materialize_for(ctx, rec_addr);
    // Unmapped only once the region has had its chance.
    if (!live && rec_addr != 0) PlaybackContext::note_unmapped(rec_addr);
    return live;
}

// A replayed H2D payload keeps any device pointer the application stored in it as a capture-time address, which translation never sees.
// Report those words instead of translating them: a payload is opaque bytes, and a value that looks like an address may be data.
static void scan_h2d_payload(PlaybackContext& ctx, const void* blob, size_t n,
                             uint64_t dst_rec) {
    const uint8_t* p = static_cast<const uint8_t*>(blob);
    const auto ranges = ctx.recorded_ranges();
    size_t hits = 0;
    for (size_t off = 0; off + 8 <= n; off += 8) {
        uint64_t w;
        memcpy(&w, p + off, 8);
        uint64_t base = 0;
        size_t   sz   = 0;
        if (!PlaybackContext::range_contains(ranges, w, &base, &sz)) continue;
        if (++hits <= 4)
            fprintf(stderr,
                    "[HRR h2d-scan] payload for 0x%llx +%zu holds 0x%llx — a "
                    "recorded address in allocation 0x%llx+%zu\n",
                    (unsigned long long)dst_rec, off, (unsigned long long)w,
                    (unsigned long long)base, sz);
    }
    if (hits)
        fprintf(stderr,
                "[HRR h2d-scan] payload for 0x%llx: %zu recorded addresses in "
                "%zu bytes\n",
                (unsigned long long)dst_rec, hits, n);
}

// Mirrors the captured API: hipMemcpyAsync on the default stream (stream_rec==0, nullptr) still uses the async variant.
static hipError_t replay_memcpy_impl(PlaybackContext& ctx, const char* api,
                                     uint64_t dst_rec, uint64_t src_rec,
                                     uint64_t size, int32_t kind,
                                     bool is_async, hipStream_t stream,
                                     uint64_t hash_lo, uint64_t hash_hi) {
    if (kind == hipMemcpyHostToDevice && host_contents_left_out(api, size, hash_lo, hash_hi))
        return hipErrorNotFound;
    const bool dev_dst = kind == hipMemcpyHostToDevice || kind == hipMemcpyDeviceToDevice;
    void*      dst = dev_dst ? translate_or_materialize(ctx, dst_rec) : nullptr;
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
        if (ctx.scan_h2d) scan_h2d_payload(ctx, blob, copy_sz, dst_rec);
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
            } else if (!hrr_d2h_blob_short("kernarg", copy_sz, blob_sz)) {
                hrr::ScopedCharge readback(ctx.host_budget, copy_sz);
                if (!readback.ok()) {
                    ctx.report_host_refusal(copy_sz, "D2H validation buffer");
                    return hipErrorMemoryAllocation;
                }
                std::vector<uint8_t> actual(copy_sz);
                // An async copy may not have finished, so sync the stream before reading back. Synchronous hipMemcpy already completes.
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
    return r;
}

hipError_t playback_hipMemcpy(PlaybackContext& ctx,
                              const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpy*>(pl);
    return replay_memcpy_impl(ctx, "hipMemcpy", a->dst, a->src, a->sizeBytes, a->kind,
                              /*is_async=*/false, nullptr,
                              a->blob_hash_lo, a->blob_hash_hi);
}

hipError_t playback_hipMemcpyAsync(PlaybackContext& ctx,
                                   const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyAsync*>(pl);
    return replay_memcpy_impl(ctx, "hipMemcpyAsync", a->dst, a->src, a->sizeBytes, a->kind,
                              /*is_async=*/true, ctx.translate_stream(a->stream),
                              a->blob_hash_lo, a->blob_hash_hi);
}

hipError_t playback_hipMemcpyHtoD(PlaybackContext& ctx,
                                  const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyHtoD*>(pl);
    return replay_memcpy_impl(ctx, "hipMemcpyHtoD", a->dst, a->src, a->sizeBytes,
                              hipMemcpyHostToDevice,
                              /*is_async=*/false, nullptr,
                              a->blob_hash_lo, a->blob_hash_hi);
}

hipError_t playback_hipMemcpyHtoDAsync(PlaybackContext& ctx,
                                       const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyHtoDAsync*>(pl);
    return replay_memcpy_impl(ctx, "hipMemcpyHtoDAsync", a->dst, a->src, a->sizeBytes,
                              hipMemcpyHostToDevice,
                              /*is_async=*/true, ctx.translate_stream(a->stream),
                              a->blob_hash_lo, a->blob_hash_hi);
}

// is_async is true so the stream is passed through even when it translates to null.
hipError_t playback_hipMemcpyWithStream(PlaybackContext& ctx,
                                        const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpyWithStream*>(pl);
    return replay_memcpy_impl(ctx, "hipMemcpyWithStream", a->dst, a->src, a->sizeBytes, a->kind,
                              /*is_async=*/true, ctx.translate_stream(a->stream),
                              a->blob_hash_lo, a->blob_hash_hi);
}

// dst is a host pointer, recorded as a raw address. Copies the live device source to a temp buffer and compares it with the expected blob.
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
    hrr::ScopedCharge readback(ctx.host_budget, sz);
    if (!readback.ok()) {
        ctx.report_host_refusal(sz, "hipMemcpyDtoH read-back buffer");
        return hipErrorMemoryAllocation;
    }
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
        } else if (!hrr_d2h_blob_short("DtoH", sz, blob_sz)) {
            hrr_d2h_validate(ctx, "DtoH", hrr_dispatch_seq, actual.data(),
                             static_cast<const uint8_t*>(expected), sz);
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
    hrr::ScopedCharge readback(ctx.host_budget, sz);
    if (!readback.ok()) {
        ctx.report_host_refusal(sz, "hipMemcpyDtoHAsync read-back buffer");
        return hipErrorMemoryAllocation;
    }
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
        } else if (!hrr_d2h_blob_short("DtoHAsync", sz, blob_sz)) {
            hrr_d2h_validate(ctx, "DtoHAsync", hrr_dispatch_seq, actual.data(),
                             static_cast<const uint8_t*>(expected), sz);
        }
    }
    return hipSuccess;
}

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

// hrr_args_hipStreamEndCapture layout: ret(4) stream(8) pGraph(8), where pGraph is the recorded *pGraph output.
// hrr_args_hipGraphInstantiate layout: ret(4) pGraphExec(8) graph(8) pErrorNode(8) pLogBuffer(8) bufferSize(8)

// hrr_args_hipStreamBeginCapture payload (after 32-byte EventHeader):
//   ret(4) stream(8) mode(4)
hipError_t playback_hipStreamBeginCapture(PlaybackContext& ctx,
                                          const uint8_t* payload) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamBeginCapture*>(payload);
    if (a->ret != hipSuccess) return hipSuccess;  // original failed — skip

    hipStream_t stream = ctx.translate_stream(a->stream);
    if (!stream && a->stream != 0) {
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
    // Capture records a failed EndCapture only when the failure ended the
    // capture (invalidated or unjoined); every recorded EndCapture ends ours.
    ctx.in_graph_capture = false;

    hipStream_t stream = ctx.translate_stream(a->stream);
    if (!stream) {
        fprintf(stderr, "[HRR] hipStreamEndCapture: stream 0x%llx not found in map\n",
                (unsigned long long)a->stream);
        return hipSuccess;  // non-fatal
    }
    hipGraph_t live_graph = nullptr;
    hipError_t r = hipStreamEndCapture(stream, &live_graph);
    if (a->ret != hipSuccess) {
        // The application got no graph, so nothing later refers to this one.
        if (live_graph) (void)hipGraphDestroy(live_graph);
        (void)hipGetLastError();
        return hipSuccess;
    }
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
        // graph_map holds every graph this replay built, whether by stream capture
        // (hipStreamEndCapture) or by the node API (hipGraphCreate). A miss means the
        // graph does not exist here at all.
        fprintf(stderr,
                "[HRR] hipGraphInstantiate: graph 0x%llx was never built at "
                "replay, so there is nothing to instantiate. Aborting rather "
                "than running an empty graph.\n",
                (unsigned long long)a->graph);
        return hipErrorNotSupported;
    }
    if (ctx.graph_is_incomplete(a->graph)) {
        // The graph exists but is missing at least one node HRR could not reconstruct
        // (each one said so when it was skipped). Instantiating it would run a graph
        // short of work and quietly produce wrong buffers.
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

hipError_t playback_hipEventElapsedTime(PlaybackContext& ctx,
                                         const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipEventElapsedTime*>(pl);
    float ms = 0.f;
    hipEvent_t start = ctx.translate_event(a->start);
    hipEvent_t stop  = ctx.translate_event(a->stop);
    hipError_t r = hipEventElapsedTime(&ms, start, stop);
    if (r == hipSuccess) return hipSuccess;
    // HIP returns hipErrorInvalidHandle when either event has no marker.
    // hipExtModuleLaunchKernel sets those markers inside the runtime. An archive recorded
    // before that side effect was stored has no hipEventRecord for the pair, and the
    // captured call still succeeded.
    if (r == hipErrorInvalidHandle && a->ret == 0 && start && stop &&
        !ctx.ext_launch_event_markers) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            fprintf(stderr,
                    "[HRR] hipEventElapsedTime: no marker on a live pair "
                    "(start=0x%llx stop=0x%llx; archive predates ext-launch "
                    "event markers, recorded success, replay returned "
                    "hipErrorInvalidHandle); continuing\n",
                    (unsigned long long)a->start, (unsigned long long)a->stop);
        }
        (void)hipGetLastError();
        return hipSuccess;
    }
    if (hrr_replayed_recorded_error(ctx, "hipEventElapsedTime", a->ret, r))
        return hipSuccess;
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

// Capture only records hipEventQuery / hipStreamQuery when they returned hipSuccess (see
// hip_capture_generated.cpp). Replay drives the API trace faster than the original CPU
// often did relative to GPU completion, so the same call can transiently return
// hipErrorNotReady (600).
hipError_t replay_query_until_success(PlaybackContext& ctx, const char* api,
                                             hipError_t (*once)(void*), void* arg) {
    hipError_t r = hipSuccess;
    const bool done = hrr::retry_bounded(ctx.max_query_attempts, [&] {
        if (ctx.fatal_error.load(std::memory_order_acquire)) {
            r = hipErrorUnknown;  // another thread failed: stop spinning
            return true;
        }
        r = once(arg);
        return r != hipErrorNotReady;
    });
    if (done)
        return r;
    fprintf(stderr,
            "[HRR] Fatal: %s still returned hipErrorNotReady after %llu attempts "
            "(--max-query-attempts); the recorded work did not complete\n",
            api, (unsigned long long)ctx.max_query_attempts);
    // Fatal for the whole replay, even under --continue-on-error: each further
    // query of an archive like this would spend the full bound again.
    ctx.fatal_error.store(true, std::memory_order_release);
    return hipErrorNotReady;
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
    return replay_query_until_success(ctx, "hipEventQuery", replay_event_query_once, &c);
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
    return replay_query_until_success(ctx, "hipStreamQuery", replay_stream_query_once, &c);
}

hipError_t playback_hipStreamQuery_spt(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamQuery_spt*>(pl);
    replay_stream_query_ctx c{(hipStream_t)ctx.translate_stream(a->stream)};
    return replay_query_until_success(ctx, "hipStreamQuery_spt", replay_stream_query_spt_once, &c);
}


// The host side of each descriptor, given the dense layout of a packed blob.
static void hrr_pack_host_side(hipPitchedPtr& ptr, hipPos& pos, const hipExtent& extent) {
    ptr.pitch = extent.width;
    ptr.xsize = extent.width;
    ptr.ysize = extent.height;
    pos = make_hipPos(0, 0, 0);
}

static void hrr_pack_host_src(HIP_MEMCPY3D& p) {
    p.srcXInBytes = p.srcY = p.srcZ = 0;
    p.srcPitch = p.WidthInBytes;
    p.srcHeight = p.Height;
}

static void hrr_pack_host_dst(HIP_MEMCPY3D& p) {
    p.dstXInBytes = p.dstY = p.dstZ = 0;
    p.dstPitch = p.WidthInBytes;
    p.dstHeight = p.Height;
}

static void hrr_pack_host_src(hip_Memcpy2D& p) {
    p.srcXInBytes = p.srcY = 0;
    p.srcPitch = p.WidthInBytes;
}

static void hrr_pack_host_dst(hip_Memcpy2D& p) {
    p.dstXInBytes = p.dstY = 0;
    p.dstPitch = p.WidthInBytes;
}

static void hrr_host_rect_pack(const HrrHostRect& rect, const uint8_t* base, uint8_t* out) {
    if (rect.width == 0) return;  // nothing copied, and `out` may be null
    for (size_t z = 0; z < rect.depth; ++z)
        for (size_t y = 0; y < rect.height; ++y, out += rect.width)
            std::memmove(out, base + rect.first + z * rect.slice + y * rect.row, rect.width);
}

template <typename Issue>
static hipError_t replay_pitched_d2h(PlaybackContext& ctx, const char* api,
                                     const HrrHostRect& dst, int32_t recorded_ret,
                                     uint64_t hash_lo, uint64_t hash_hi,
                                     hipStream_t stream, bool is_async,
                                     Issue&& issue) {
    // A copy that failed at capture wrote nothing, so its blob is no output.
    const bool validate = ctx.validate_d2h && (hash_lo || hash_hi) && recorded_ret == 0;
    if (!dst.ok) {
        fprintf(stderr, "[HRR] %s D2H: recorded host rect overflows size_t, skipped\n", api);
        if (validate) ctx.d2h_attempted++;
        return hipSuccess;
    }
    // The extent comes from the archive and can be far larger than the rows the copy
    // writes.
    const uint64_t scratch_bytes = dst.extent ? dst.extent : 1;
    hrr::ScopedCharge scratch_charge(ctx.host_budget, scratch_bytes);
    if (!scratch_charge.ok()) {
        ctx.report_host_refusal(scratch_bytes, api);
        return hipErrorMemoryAllocation;
    }
    std::unique_ptr<uint8_t[]> host(new (std::nothrow) uint8_t[dst.extent ? dst.extent : 1]);
    if (!host) {
        fprintf(stderr, "[HRR] %s D2H: no %zu-byte scratch buffer for the recorded host rect, "
                "skipped\n", api, dst.extent);
        if (validate) ctx.d2h_attempted++;
        return hipSuccess;
    }
    hipError_t r = issue(host.get());
    if (r == hipSuccess && is_async) r = hipStreamSynchronize(stream);
    if (hrr_replayed_recorded_error(ctx, api, recorded_ret, r)) return hipSuccess;
    if (r != hipSuccess) {
        fprintf(stderr, "[HRR] %s D2H: device readback failed: %d (%s)\n",
                api, r, hipGetErrorString(r));
        if (validate) {
            ctx.d2h_attempted++;
            ctx.note_d2h_fail(hrr_dispatch_seq);
        }
        return r;
    }
    if (!validate) return hipSuccess;  // no expected blob: just execute

    ctx.d2h_attempted++;
    size_t blob_sz = 0;
    const auto* expected =
        static_cast<const uint8_t*>(ctx.load_blob(hash_lo, hash_hi, &blob_sz));
    if (expected && blob_sz < dst.extent) {
        // An archive recorded before D2H blobs covered the rect holds only the first
        // width*height*depth bytes of the host buffer.
        fprintf(stderr,
                "[HRR] %s D2H: expected blob holds %zu of the %zu bytes the recorded "
                "rect spans (captured before rect-shaped D2H blobs, or truncated), "
                "not validated\n",
                api, blob_sz, dst.extent);
        return hipSuccess;
    }
    if (!expected) {
        fprintf(stderr, "[HRR] %s D2H validate FAIL: expected blob not found in archive\n", api);
        ctx.note_d2h_fail(hrr_dispatch_seq);
        return hipSuccess;
    }
    const size_t n = dst.width * dst.height * dst.depth;
    if (hrr_host_rect_dense(&dst)) {
        hrr_d2h_validate(ctx, api, hrr_dispatch_seq, host.get() + dst.first,
                         expected + dst.first, n);
        return hipSuccess;
    }
    // Like the scratch buffer, one that cannot be had is a skipped check.
    hrr::ScopedCharge want_charge(ctx.host_budget, n ? n : 1);
    if (!want_charge.ok()) {
        ctx.report_host_refusal(n ? n : 1, api);
        return hipErrorMemoryAllocation;
    }
    std::unique_ptr<uint8_t[]> want(new (std::nothrow) uint8_t[n]);
    if (!want) {
        fprintf(stderr, "[HRR] %s D2H: no %zu-byte buffer for the expected rows, skipped\n",
                api, n);
        return hipSuccess;
    }
    hrr_host_rect_pack(dst, host.get(), host.get());
    hrr_host_rect_pack(dst, expected, want.get());
    hrr_d2h_validate(ctx, api, hrr_dispatch_seq, host.get(), want.get(), n);
    return hipSuccess;
}

static void note_unmapped_d2h_src(PlaybackContext& ctx, const char* api, uint64_t src_rec,
                                  int32_t recorded_ret, uint64_t hash_lo, uint64_t hash_hi) {
    if (recorded_ret != 0) return;
    if (ctx.validate_d2h && (hash_lo || hash_hi)) {
        fprintf(stderr, "[HRR] %s D2H validate FAIL: src 0x%llx not mapped - "
                        "pointer translation bug\n",
                api, (unsigned long long)src_rec);
        ctx.d2h_attempted++;
        ctx.note_d2h_fail(hrr_dispatch_seq);
    } else {
        fprintf(stderr, "[HRR] %s D2H: src 0x%llx not mapped, not replayed\n", api,
                (unsigned long long)src_rec);
    }
}


// The host destination of a hipMemcpy3D, as hip::getDrvMemcpy3DDesc() widens it for
// pointer copies: dstPos in bytes, rows dstPtr.pitch apart and slices
// dstPtr.pitch*dstPtr.ysize apart.
static HrrHostRect memcpy3d_host_dst(const hipMemcpy3DParms& p) {
    return hrr_host_rect(p.dstPtr.pitch, p.dstPtr.ysize, p.dstPos.x, p.dstPos.y,
                         p.dstPos.z, p.extent.width, p.extent.height, p.extent.depth);
}

// Defined with the driver-copy helpers below.
static hipError_t drvmemcpy_h2d_src_blob(PlaybackContext& ctx, const char* api,
                                         uint64_t hash_lo, uint64_t hash_hi,
                                         size_t need, const void** blob);

static hipError_t memcpy3d_h2d_src(PlaybackContext& ctx, const char* api, uint64_t hash_lo,
                                   uint64_t hash_hi, hipMemcpy3DParms& p) {
    if (ctx.packed_host_rects) hrr_pack_host_side(p.srcPtr, p.srcPos, p.extent);
    size_t need = hrr_host_rect_extent(p.srcPtr.pitch, p.srcPtr.ysize, p.srcPos.x, p.srcPos.y,
                                       p.srcPos.z, p.extent.width, p.extent.height,
                                       p.extent.depth);
    const void* blob = nullptr;
    const hipError_t r = drvmemcpy_h2d_src_blob(ctx, api, hash_lo, hash_hi, need, &blob);
    p.srcPtr.ptr = const_cast<void*>(blob);
    return r;
}

static bool translate_array_or_refuse(const PlaybackContext& ctx, const char* api,
                                      const char* what, hipArray_t* arr) {
    if (!*arr) return true;
    const uint64_t rec = reinterpret_cast<uint64_t>(*arr);
    hipArray_t live = ctx.translate_array(rec);
    if (!live) {
        fprintf(stderr,
                "[HRR] %s: the %s array 0x%llx was not created at replay, so the "
                "copy is refused rather than hand a recorded address to HIP\n",
                api, what, (unsigned long long)rec);
        return false;
    }
    *arr = live;
    return true;
}

// hipMemcpy3D reads srcArray / dstArray as array handles, and CLR dereferences
// them to size the element and the layered layout.
bool hrr_translate_memcpy3d_arrays(const PlaybackContext& ctx, const char* api,
                                   hipMemcpy3DParms& parms) {
    return translate_array_or_refuse(ctx, api, "source", &parms.srcArray) &&
           translate_array_or_refuse(ctx, api, "destination", &parms.dstArray);
}

static bool memcpy3d_is_host_array(const hipMemcpy3DParms& p) {
    return (p.kind == hipMemcpyHostToDevice && p.dstArray) ||
           (p.kind == hipMemcpyDeviceToHost && p.srcArray);
}

static hipError_t refuse_host_array_copy(PlaybackContext& ctx, const char* api,
                                         const hipMemcpy3DParms& p, uint64_t hash_lo,
                                         uint64_t hash_hi) {
    if (p.kind == hipMemcpyDeviceToHost && ctx.validate_d2h && (hash_lo || hash_hi)) {
        ctx.d2h_attempted++;
        ctx.note_d2h_fail(hrr_dispatch_seq);
    }
    hrr_note_unreplayable(ctx, api,
                          "a copy between host memory and an array has its extent in array "
                          "elements but its recorded host rect and blob in bytes, so issuing it "
                          "would read or write past them");
    return hipErrorNotSupported;
}

static hipError_t replay_memcpy3d(PlaybackContext& ctx, const uint8_t* pl, bool spt) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpy3D*>(pl);
    hipMemcpy3DParms parms{};
    std::memcpy(&parms, a->parms_bytes, sizeof(parms));
    hipError_t (*const copy)(const hipMemcpy3DParms*) = spt ? hipMemcpy3D_spt : hipMemcpy3D;
    if (!hrr_translate_memcpy3d_arrays(ctx, spt ? "hipMemcpy3D_spt" : "hipMemcpy3D", parms))
        return hipErrorInvalidValue;
    if (memcpy3d_is_host_array(parms))
        return refuse_host_array_copy(ctx, spt ? "hipMemcpy3D_spt" : "hipMemcpy3D", parms,
                                      a->d2h_hash_lo, a->d2h_hash_hi);

    if (parms.kind == hipMemcpyHostToDevice) {
        hipError_t r =
            memcpy3d_h2d_src(ctx, "hipMemcpy3D", a->blob_hash_lo, a->blob_hash_hi, parms);
        if (!parms.srcPtr.ptr) return r;
        parms.dstPtr.ptr = ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.dstPtr.ptr));
        r = copy(&parms);
        if (r == hipSuccess)
            r = hrr_sync_after_replayed_h2d(ctx, "replayed 3D H2D memcpy");
        return r;
    }
    if (parms.kind == hipMemcpyDeviceToHost) {
        uint64_t src_rec = reinterpret_cast<uint64_t>(parms.srcPtr.ptr);
        void* src_live = ctx.translate_ptr(src_rec);
        if (!src_live) {
            note_unmapped_d2h_src(ctx, "hipMemcpy3D", src_rec, a->ret, a->d2h_hash_lo,
                                  a->d2h_hash_hi);
            return hipSuccess;
        }
        parms.srcPtr.ptr = src_live;
        if (ctx.packed_host_rects) hrr_pack_host_side(parms.dstPtr, parms.dstPos, parms.extent);
        return replay_pitched_d2h(ctx, "hipMemcpy3D", memcpy3d_host_dst(parms), a->ret,
                                  a->d2h_hash_lo, a->d2h_hash_hi, nullptr, false,
                                  [&parms, copy](void* host) {
                                      parms.dstPtr.ptr = host;
                                      return copy(&parms);
                                  });
    }
    parms.srcPtr.ptr = ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.srcPtr.ptr));
    parms.dstPtr.ptr = ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.dstPtr.ptr));
    return copy(&parms);
}

hipError_t playback_hipMemcpy3D(PlaybackContext& ctx, const uint8_t* pl) {
    return replay_memcpy3d(ctx, pl, /*spt=*/false);
}

static hipStream_t memcpy3d_async_stream(PlaybackContext& ctx, uint64_t rec, bool spt) {
    if (rec == reinterpret_cast<uint64_t>(hipStreamPerThread) || (spt && rec == 0))
        return hipStreamPerThread;
    return ctx.translate_stream(rec);
}

static hipError_t replay_memcpy3d_async(PlaybackContext& ctx, const uint8_t* pl, bool spt) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpy3DAsync*>(pl);
    hipMemcpy3DParms parms{};
    std::memcpy(&parms, a->parms_bytes, sizeof(parms));
    if (!hrr_translate_memcpy3d_arrays(
            ctx, spt ? "hipMemcpy3DAsync_spt" : "hipMemcpy3DAsync", parms))
        return hipErrorInvalidValue;
    if (memcpy3d_is_host_array(parms))
        return refuse_host_array_copy(ctx, spt ? "hipMemcpy3DAsync_spt" : "hipMemcpy3DAsync",
                                      parms, a->d2h_hash_lo, a->d2h_hash_hi);
    hipStream_t stream = memcpy3d_async_stream(ctx, a->stream, spt);

    if (parms.kind == hipMemcpyHostToDevice) {
        hipError_t r = memcpy3d_h2d_src(ctx, "hipMemcpy3DAsync", a->blob_hash_lo,
                                        a->blob_hash_hi, parms);
        if (!parms.srcPtr.ptr) return r;
        parms.dstPtr.ptr = ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.dstPtr.ptr));
        r = hipMemcpy3DAsync(&parms, stream);
        if (r == hipSuccess)
            r = hrr_sync_after_replayed_h2d(ctx, "replayed 3D async H2D memcpy");
        return r;
    }
    if (parms.kind == hipMemcpyDeviceToHost) {
        uint64_t src_rec = reinterpret_cast<uint64_t>(parms.srcPtr.ptr);
        void* src_live = ctx.translate_ptr(src_rec);
        if (!src_live) {
            note_unmapped_d2h_src(ctx, "hipMemcpy3DAsync", src_rec, a->ret, a->d2h_hash_lo,
                                  a->d2h_hash_hi);
            return hipSuccess;
        }
        parms.srcPtr.ptr = src_live;
        if (ctx.packed_host_rects) hrr_pack_host_side(parms.dstPtr, parms.dstPos, parms.extent);
        return replay_pitched_d2h(ctx, "hipMemcpy3DAsync", memcpy3d_host_dst(parms), a->ret,
                                  a->d2h_hash_lo, a->d2h_hash_hi, stream, true,
                                  [&parms, stream](void* host) {
                                      parms.dstPtr.ptr = host;
                                      return hipMemcpy3DAsync(&parms, stream);
                                  });
    }
    parms.srcPtr.ptr = ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.srcPtr.ptr));
    parms.dstPtr.ptr = ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.dstPtr.ptr));
    return hipMemcpy3DAsync(&parms, stream);
}

hipError_t playback_hipMemcpy3DAsync(PlaybackContext& ctx, const uint8_t* pl) {
    return replay_memcpy3d_async(ctx, pl, /*spt=*/false);
}


static hipError_t drvmemcpy_h2d_src_blob(PlaybackContext& ctx, const char* api,
                                         uint64_t hash_lo, uint64_t hash_hi,
                                         size_t need, const void** blob) {
    *blob = nullptr;
    if (host_contents_left_out(api, need, hash_lo, hash_hi)) return hipErrorNotFound;
    size_t blob_sz = 0;
    const void* found = (hash_lo || hash_hi)
                            ? ctx.load_blob(hash_lo, hash_hi, &blob_sz)
                            : nullptr;
    if (!found) {
        fprintf(stderr, "[HRR] %s H2D: no blob to substitute, skipped\n", api);
        return hipSuccess;
    }
    if (blob_sz < need) {
        fprintf(stderr,
                "[HRR] %s H2D: blob covers %zu of the %zu bytes the recorded rect "
                "spans, skipped\n", api, blob_sz, need);
        return hipSuccess;
    }
    *blob = found;
    return hipSuccess;
}

// Array-typed rects are out of scope for these APIs, so decline them up front rather than
// let them reach a branch that cannot describe them.
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

static hipError_t replay_drvmemcpy3d(PlaybackContext& ctx, HIP_MEMCPY3D& parms,
                                     const char* api, int32_t recorded_ret,
                                     uint64_t blob_hash_lo, uint64_t blob_hash_hi,
                                     uint64_t d2h_hash_lo, uint64_t d2h_hash_hi,
                                     hipStream_t stream, bool is_async) {
    if (parms.srcMemoryType == hipMemoryTypeHost) {
        if (parms.dstMemoryType == hipMemoryTypeHost) {
            // Host-to-host touches no device state, and the recorded dstHost VA is
            // meaningless at replay (writing through it would be a wild store). Nothing
            // to reproduce.
            fprintf(stderr, "[HRR] %s: host-to-host copy, skipped\n", api);
            return hipSuccess;
        }
        if (ctx.packed_host_rects) hrr_pack_host_src(parms);
        size_t need = hrr_host_rect_extent(parms.srcPitch, parms.srcHeight,
                                           parms.srcXInBytes, parms.srcY, parms.srcZ,
                                           parms.WidthInBytes, parms.Height, parms.Depth);
        const void* blob = nullptr;
        hipError_t r =
            drvmemcpy_h2d_src_blob(ctx, api, blob_hash_lo, blob_hash_hi, need, &blob);
        if (!blob) return r;
        parms.srcHost = blob;
        parms.dstDevice = reinterpret_cast<hipDeviceptr_t>(
            ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.dstDevice)));
        r = is_async ? hipDrvMemcpy3DAsync(&parms, stream) : hipDrvMemcpy3D(&parms);
        if (r == hipSuccess)
            r = hrr_sync_after_replayed_h2d(ctx, is_async ? "replayed driver 3D async H2D memcpy"
                                                          : "replayed driver 3D H2D memcpy");
        return r;
    }
    if (parms.dstMemoryType == hipMemoryTypeHost) {
        uint64_t src_rec = reinterpret_cast<uint64_t>(parms.srcDevice);
        void* src_live = ctx.translate_ptr(src_rec);
        if (!src_live) {
            note_unmapped_d2h_src(ctx, api, src_rec, recorded_ret, d2h_hash_lo, d2h_hash_hi);
            return hipSuccess;
        }
        parms.srcDevice = reinterpret_cast<hipDeviceptr_t>(src_live);
        if (ctx.packed_host_rects) hrr_pack_host_dst(parms);
        const HrrHostRect dst = hrr_host_rect(parms.dstPitch, parms.dstHeight,
                                              parms.dstXInBytes, parms.dstY, parms.dstZ,
                                              parms.WidthInBytes, parms.Height, parms.Depth);
        return replay_pitched_d2h(ctx, api, dst, recorded_ret, d2h_hash_lo, d2h_hash_hi,
                                  stream, is_async, [&parms, stream, is_async](void* host) {
                                      parms.dstHost = host;
                                      return is_async ? hipDrvMemcpy3DAsync(&parms, stream)
                                                      : hipDrvMemcpy3D(&parms);
                                  });
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
    return replay_drvmemcpy3d(ctx, parms, "hipDrvMemcpy3D", a->ret,
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
    return replay_drvmemcpy3d(ctx, parms, "hipDrvMemcpy3DAsync", a->ret,
                              a->blob_hash_lo, a->blob_hash_hi,
                              a->d2h_hash_lo, a->d2h_hash_hi,
                              ctx.translate_stream(a->stream), /*is_async=*/true);
}

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
        // The runtime widens hip_Memcpy2D to a HIP_MEMCPY3D with Depth == 1 and srcHeight
        // == 0, defaulting the pitch to x + WidthInBytes. See hip::getDrvMemcpy3DDesc().
        if (ctx.packed_host_rects) hrr_pack_host_src(parms);
        size_t pitch = parms.srcPitch ? parms.srcPitch
                                      : parms.srcXInBytes + parms.WidthInBytes;
        size_t need = hrr_host_rect_extent(pitch, /*pitch_height=*/0, parms.srcXInBytes,
                                           parms.srcY, /*z=*/0, parms.WidthInBytes,
                                           parms.Height, /*depth=*/1);
        const void* blob = nullptr;
        hipError_t r = drvmemcpy_h2d_src_blob(ctx, api, a->blob_hash_lo, a->blob_hash_hi,
                                              need, &blob);
        if (!blob) return r;
        parms.srcHost = blob;
        parms.dstDevice = reinterpret_cast<hipDeviceptr_t>(
            ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.dstDevice)));
        r = issue(&parms);
        if (r == hipSuccess)
            r = hrr_sync_after_replayed_h2d(ctx, "replayed driver 2D H2D memcpy");
        return r;
    }
    if (parms.dstMemoryType == hipMemoryTypeHost) {
        uint64_t src_rec = reinterpret_cast<uint64_t>(parms.srcDevice);
        void* src_live = ctx.translate_ptr(src_rec);
        if (!src_live) {
            note_unmapped_d2h_src(ctx, api, src_rec, a->ret, a->d2h_hash_lo, a->d2h_hash_hi);
            return hipSuccess;
        }
        parms.srcDevice = reinterpret_cast<hipDeviceptr_t>(src_live);
        if (ctx.packed_host_rects) hrr_pack_host_dst(parms);
        size_t pitch = parms.dstPitch ? parms.dstPitch
                                      : parms.dstXInBytes + parms.WidthInBytes;
        const HrrHostRect dst = hrr_host_rect(pitch, /*pitch_height=*/0, parms.dstXInBytes,
                                              parms.dstY, /*z=*/0, parms.WidthInBytes,
                                              parms.Height, /*depth=*/1);
        return replay_pitched_d2h(ctx, api, dst, a->ret, a->d2h_hash_lo, a->d2h_hash_hi,
                                  stream, is_async, [&parms, &issue](void* host) {
                                      parms.dstHost = host;
                                      return issue(&parms);
                                  });
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


static bool translate_batch_operand(PlaybackContext& ctx, const char* what,
                                    hipMemcpy3DOperand& operand) {
    if (operand.type == hipMemcpyOperandTypePointer) {
        operand.op.ptr.ptr = ctx.translate_ptr(
            reinterpret_cast<uint64_t>(operand.op.ptr.ptr));
        return true;
    }
    if (operand.type != hipMemcpyOperandTypeArray) {
        fprintf(stderr, "[HRR] hipMemcpy3DBatchAsync: the %s operand has type %d, which is "
                        "neither pointer nor array, so the batch is refused\n",
                what, static_cast<int>(operand.type));
        return false;
    }
    if (!operand.op.array.array) {
        fprintf(stderr, "[HRR] hipMemcpy3DBatchAsync: the %s array operand is null, so the "
                        "batch is refused\n", what);
        return false;
    }
    return translate_array_or_refuse(ctx, "hipMemcpy3DBatchAsync", what,
                                     &operand.op.array.array);
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
        if (!translate_batch_operand(ctx, "source", op.src) ||
            !translate_batch_operand(ctx, "destination", op.dst))
            return hipErrorInvalidValue;
    }

    size_t fail_idx = 0;
    return hipMemcpy3DBatchAsync(n, ops.data(), &fail_idx,
                                 static_cast<unsigned long long>(a->flags),
                                 ctx.translate_stream(a->stream));
}


template <typename T>
static hipError_t replay_memcpy2d(PlaybackContext& ctx, const T* a,
                                   hipStream_t stream, bool is_async) {
    const auto kind   = static_cast<hipMemcpyKind>(a->kind);
    const size_t width  = static_cast<size_t>(a->width);
    const size_t height = static_cast<size_t>(a->height);
    // A packed host blob is read and written with the host pitch == width.
    const size_t dpitch = (ctx.packed_host_rects && kind == hipMemcpyDeviceToHost)
                              ? width : static_cast<size_t>(a->dpitch);
    const size_t spitch = (ctx.packed_host_rects && kind == hipMemcpyHostToDevice)
                              ? width : static_cast<size_t>(a->spitch);

    if (kind == hipMemcpyHostToDevice) {
        // The runtime reads `height` rows `spitch` apart from the blob, so it
        // must reach spitch * (height - 1) + width bytes.
        const size_t need = hrr_host_rect_extent(spitch, /*pitch_height=*/0, 0, 0, 0,
                                                 width, height, /*depth=*/1);
        const void* blob = nullptr;
        hipError_t r = drvmemcpy_h2d_src_blob(
            ctx, is_async ? "hipMemcpy2DAsync" : "hipMemcpy2D", a->blob_hash_lo,
            a->blob_hash_hi, need, &blob);
        if (!blob) return r;
        void* dst = ctx.translate_ptr(a->dst);
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
            note_unmapped_d2h_src(ctx, is_async ? "hipMemcpy2DAsync" : "hipMemcpy2D", a->src,
                                  a->ret, a->d2h_hash_lo, a->d2h_hash_hi);
            return hipSuccess;
        }
        const HrrHostRect dst = hrr_host_rect(dpitch, /*pitch_height=*/0, 0, 0, 0,
                                              width, height, /*depth=*/1);
        return replay_pitched_d2h(ctx, is_async ? "hipMemcpy2DAsync" : "hipMemcpy2D", dst,
                                  a->ret, a->d2h_hash_lo, a->d2h_hash_hi, stream, is_async,
                                  [=](void* host) {
                                      return is_async
                                          ? hipMemcpy2DAsync(host, dpitch, src, spitch, width,
                                                             height, hipMemcpyDeviceToHost, stream)
                                          : hipMemcpy2D(host, dpitch, src, spitch, width, height,
                                                        hipMemcpyDeviceToHost);
                                  });
    }

    if (width == 0 || height == 0) return hipSuccess;
    // A host-to-host copy is a no-op, as in the 1D path (replay_memcpy_impl):
    // plain host memory is in no map, and the copy touches nothing replay tracks.
    if (kind == hipMemcpyHostToHost) return hipSuccess;

    void* dst = ctx.translate_ptr(a->dst);
    void* src = ctx.translate_ptr(a->src);
    if (!dst || !src) {
        fprintf(stderr,
                "[HRR] hipMemcpy2D%s: %s 0x%llx is not mapped at replay, so the copy "
                "is refused rather than run against a recorded address\n",
                is_async ? "Async" : "", !dst ? "destination" : "source",
                (unsigned long long)(!dst ? a->dst : a->src));
        return hipErrorInvalidValue;
    }
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
    static_assert(sizeof(hrr_args_hipMemcpy3D_spt) == sizeof(hrr_args_hipMemcpy3D),
                  "the _spt payload is replayed as the plain one");
    return replay_memcpy3d(ctx, pl, /*spt=*/true);
}

hipError_t playback_hipMemcpy3DAsync_spt(PlaybackContext& ctx, const uint8_t* pl) {
    static_assert(sizeof(hrr_args_hipMemcpy3DAsync_spt) == sizeof(hrr_args_hipMemcpy3DAsync),
                  "the _spt payload is replayed as the plain one");
    return replay_memcpy3d_async(ctx, pl, /*spt=*/true);
}


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

hipError_t playback_hipFreeArray(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipFreeArray*>(pl);
    hipArray_t arr = ctx.translate_array(a->array);
    if (!arr) return hipSuccess;  // nooped alloc (hipMallocArray noop)
    hipError_t r = hipFreeArray(arr);
    // A later copy naming this handle must not translate to the freed array.
    // Dropped whatever the result: the runtime forgets an array a free reaches.
    ctx.remove_array(a->array);
    return r;
}


hipError_t playback_hipStreamSetAttribute(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipStreamSetAttribute*>(pl);
    hipStream_t stream = ctx.translate_stream(a->stream);
    hipStreamAttrValue val{};
    std::memcpy(&val, a->stream_attr_bytes, sizeof(val));
    // The access policy window's base_ptr is a device address of the capturing process;
    // translate it as hipLaunchKernelExC does.
    if (a->attr == hipStreamAttributeAccessPolicyWindow)
        val.accessPolicyWindow.base_ptr = ctx.translate_ptr(
            reinterpret_cast<uint64_t>(val.accessPolicyWindow.base_ptr));
    return hipStreamSetAttribute(stream, static_cast<hipStreamAttrID>(a->attr), &val);
}


hipError_t playback_hipMemGetAllocationGranularity(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemGetAllocationGranularity*>(pl);
    hipMemAllocationProp prop{};
    std::memcpy(&prop, a->alloc_prop_bytes, sizeof(prop));
    size_t granularity = 0;
    return hipMemGetAllocationGranularity(&granularity, &prop,
                                          static_cast<hipMemAllocationGranularity_flags>(a->option));
}


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
    void* ptr = ctx.translate_vmm_va(a->ptr);
    if (!ptr) ptr = ctx.translate_ptr(a->ptr);
    if (!ptr) return hipSuccess;  // VA not found — address space not rebuilt yet, skip
    hipMemAccessDesc desc{};
    std::memcpy(&desc, a->access_desc_bytes, sizeof(desc));
    return hipMemSetAccess(ptr, static_cast<size_t>(a->size), &desc, 1);
}

// These APIs require tracking: recorded VA -> live VA, and recorded handle -> live
// hipMemGenericAllocationHandle_t. The generated shims cannot do this because they
// discard output pointers and pass stale handles as-is.

hipError_t playback_hipMemAddressReserve(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemAddressReserve*>(pl);
    uint64_t rec_ptr = a->ptr;  // recorded output pointer-to-pointer; contains the reserved VA
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
    // The allocation property is carried inline (DEREF_FIELDS); hardcoding Pinned/device
    // 0 would build the wrong topology for a heap that maps cross-node peers.
    hipMemAllocationProp prop{};
    if (a->prop_present) {
        std::memcpy(&prop, a->prop_bytes, sizeof(prop));
    } else {
        prop.type          = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id   = 0;
    }
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
    // A host-located handle is host-backed physical memory (pinned, GTT), so it
    // counts against --max-host-bytes. A device-located one does not.
    const bool host_backed = prop.location.type != hipMemLocationTypeDevice;
    // A handle recorded again under a name still held: release the old one first
    // so neither the handle nor its charge is left behind.
    {
        hipMemGenericAllocationHandle_t old{};
        {
            std::unique_lock lk(ctx.map_mutex);
            auto it = ctx.vmm_handle_map.find(rec_handle);
            if (it != ctx.vmm_handle_map.end()) {
                old = it->second;
                ctx.vmm_handle_map.erase(it);
            }
        }
        if (old) {
            (void)hipMemRelease(old);
            ctx.vmm_note_release(old);  // its charge stays while a mapping holds it
        }
    }
    const uint64_t charge = host_backed ? a->size : 0;
    if (host_backed && !ctx.charge_host(charge, "hipMemCreate (host location)"))
        return hipErrorMemoryAllocation;
    hipMemGenericAllocationHandle_t live_handle{};
    hipError_t r = hipMemCreate(&live_handle, static_cast<size_t>(a->size), &prop,
                                static_cast<unsigned long long>(a->flags));
    if (r == hipSuccess) {
        ctx.vmm_note_create(rec_handle, live_handle, charge);
    } else {
        ctx.host_budget.release(charge);
    }
    return r;
}

hipError_t playback_hipMemRelease(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemRelease*>(pl);
    hipMemGenericAllocationHandle_t live = ctx.translate_vmm_handle(a->handle);
    if (!live) return hipSuccess;
    hipError_t r = hipMemRelease(live);
    if (r == hipSuccess) {
        {
            std::unique_lock lk(ctx.map_mutex);
            ctx.vmm_handle_map.erase(a->handle);
        }
        ctx.vmm_note_release(live);
    }
    return r;
}

hipError_t playback_hipMemMap(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemMap*>(pl);
    void* live_va = ctx.translate_vmm_va(a->ptr);
    if (!live_va) {
        // Not a reservation base.
        (void)ctx.translate_ptr(a->ptr);
        return hipSuccess;
    }
    hipMemGenericAllocationHandle_t live_handle = ctx.translate_vmm_handle(a->handle);
    if (!live_handle) return hipSuccess;  // handle not tracked, skip
    hipError_t r = hipMemMap(live_va,
                             static_cast<size_t>(a->size),
                             static_cast<size_t>(a->offset),
                             live_handle,
                             static_cast<unsigned long long>(a->flags));
    if (r == hipSuccess) ctx.vmm_note_map(live_va, static_cast<size_t>(a->size), live_handle);
    return r;
}

hipError_t playback_hipMemUnmap(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMemUnmap*>(pl);
    void* live_va = ctx.translate_vmm_va(a->ptr);
    if (!live_va) return hipSuccess;
    hipError_t r = hipMemUnmap(live_va, static_cast<size_t>(a->size));
    if (r == hipSuccess) ctx.vmm_note_unmap(live_va, static_cast<size_t>(a->size));
    return r;
}

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
            // A batch the runtime cannot address is rejected whole, so one untranslatable
            // entry would cost the rest of the archive. Atomic for the same reason as the
            // __hipRegisterVar warning: replay is multi-threaded and this flag is shared.
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true)) {
                fprintf(stderr,
                        "[HRR] hipStreamBatchMemOp: op address 0x%llx is not in "
                        "any recorded allocation — skipping this batch\n",
                        (unsigned long long)reinterpret_cast<uint64_t>(*addr));
            }
            return hipSuccess;
        }
        *addr = live;
        if (op.operation == hipStreamMemOpWaitValue32 ||
            op.operation == hipStreamMemOpWaitValue64)
            op.waitValue.alias = nullptr;
        else
            op.writeValue.alias = nullptr;
    }

    return hipStreamBatchMemOp(ctx.translate_stream(a->stream), n, ops.data(),
                               a->flags);
}

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
    // Capture no longer records the handle and leaves the field zero. Such an export
    // pairs with nothing, so every import is reported as unmatched, which is what it was:
    // a process cannot open its own handle.
    const bool recorded = std::any_of(std::begin(a->handle_bytes), std::end(a->handle_bytes),
                                      [](uint8_t b) { return b != 0; });
    if (a->handle_present && recorded)
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


static bool translate_node_deps(PlaybackContext& ctx, const char* api,
                                const uint8_t* bytes, size_t field_bytes,
                                uint32_t n, uint8_t present, uint64_t recorded_n,
                                std::vector<hipGraphNode_t>& out) {
    out.clear();
    if (!present || n == 0) return true;
    // Capture stores at most what the inline field holds and does not say so, but the
    // record also carries the caller's own count.
    if (!hrr::deps_complete(recorded_n, n)) {
        fprintf(stderr,
                "[HRR] %s: event with sequence number %llu was built with %llu dependencies but its record "
                "holds %u; the others were dropped at capture, so the ordering cannot "
                "be reproduced; refusing the call\n",
                api, (unsigned long long)hrr_dispatch_seq, (unsigned long long)recorded_n, n);
        return false;
    }
    // The record holds its dependencies inline; a larger count is malformed.
    if (!hrr::inline_count_fits(n, field_bytes, sizeof(hipGraphNode_t))) {
        fprintf(stderr,
                "[HRR] %s: event with sequence number %llu records %u dependencies but its record holds "
                "at most %zu; refusing the call\n",
                api, (unsigned long long)hrr_dispatch_seq, n,
                hrr::inline_capacity(field_bytes, sizeof(hipGraphNode_t)));
        return false;
    }
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

// What a hand-written graph-node handler returns when the runtime refused the call.
// Unlike the generated shims, these record the event even when the call failed at
// capture, so meeting the same refusal here is fidelity.
static hipError_t node_add_failed(PlaybackContext& ctx, const char* api,
                                  int32_t recorded_ret, hipError_t r,
                                  uint64_t graph) {
    if (hrr_replayed_recorded_error(ctx, api, recorded_ret, r))
        return hipSuccess;
    ctx.mark_graph_incomplete(graph, api);
    return r;
}

// What a kernel set-params handler returns when it could not rebuild the parameters:
// hipErrorInvalidValue, with the node left on its old arguments.
static hipError_t set_params_refused(PlaybackContext& ctx, const char* api,
                                     int32_t recorded_ret) {
    return hrr_replayed_recorded_error(ctx, api, recorded_ret,
                                       hipErrorInvalidValue)
               ? hipSuccess : hipErrorInvalidValue;
}

// Rebuild hipKernelNodeParams from the recorded struct plus the kernel tail. `storage`
// and `arg_ptrs` own what the returned params point at and must outlive the call that
// consumes them.
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
    if (!kernel_args_fit_signature(ctx, api, kernel_name, func, arg_storage))
        return false;

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
                             a->pDependencies_bytes, sizeof(a->pDependencies_bytes),
                             a->pDependencies_n, a->pDependencies_present,
                             a->numDependencies, deps)) {
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
        return set_params_refused(ctx, "hipGraphKernelNodeSetParams", a->ret);
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
        return set_params_refused(ctx, "hipGraphExecKernelNodeSetParams",
                                  a->ret);
    hipError_t r = hipGraphExecKernelNodeSetParams(exec, node, &knp);
    return hrr_replayed_recorded_error(ctx, "hipGraphExecKernelNodeSetParams",
                                       a->ret, r) ? hipSuccess : r;
}

// Rebuild hipBatchMemOpNodeParams: the struct is recorded inline, the op array follows as
// a tail, and each op's address is translated the same way hipStreamBatchMemOp's is.
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
    // hipGraphAddBatchMemOpNode rejects a null context (hip_graph.cpp:3736), so the
    // recorded capture-time address is swapped for the live one rather than cleared. A
    // recording made with a null context keeps it, and fails here exactly as it failed
    // there.
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
                             a->dependencies_bytes, sizeof(a->dependencies_bytes),
                             a->dependencies_n, a->dependencies_present,
                             a->numDependencies, deps)) {
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
                             a->pDependencies_bytes, sizeof(a->pDependencies_bytes),
                             a->pDependencies_n, a->pDependencies_present,
                             a->numDependencies, deps)) {
        ctx.mark_graph_incomplete(a->graph, "hipGraphAddMemAllocNode");
        return hipSuccess;
    }

    hipMemAllocNodeParams anp{};
    std::memcpy(&anp, a->pNodeParams_bytes, sizeof(anp));
    const uint64_t rec_dptr = reinterpret_cast<uint64_t>(anp.dptr);
    anp.dptr = nullptr;  // written by the call; the recorded value is not ours
    if (anp.accessDescs && anp.accessDescCount) {
        // The descriptor array is a second pointer hop the archive does not carry.
        // Dropping it would quietly give the allocation different peer visibility than
        // the recording had.
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
// archive never carried, so it is reported rather than passed through. An array
// replay never created becomes null, which CLR refuses with hipErrorInvalidValue
// (ihipDrvMemcpy3D_validate), also for the graph set-params calls.
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
                             a->dependencies_bytes, sizeof(a->dependencies_bytes),
                             a->dependencies_n, a->dependencies_present,
                             a->numDependencies, deps)) {
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
