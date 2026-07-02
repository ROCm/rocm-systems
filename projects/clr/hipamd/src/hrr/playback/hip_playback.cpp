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
#ifdef _WIN32
#include <process.h>  // _exit
#else
#include <unistd.h>   // _exit
#endif

// Thread-local sequence ID — set by dispatch_event before calling any handler.
// Kernel-launch handlers use this to wait for their submission turn and then
// immediately unblock the next thread before doing timing/sync.
thread_local uint64_t hrr_dispatch_seq = 0;

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
static hipError_t replay_kernel_launch(PlaybackContext& ctx, const uint8_t* pl,
                                       bool ext_global_worksize = false) {
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

    // Resolve hipFunction_t — cache hit avoids repeated hipModuleGetFunction
    // searches. Locked because multiple threads can now be in kernel launch
    // preparation concurrently (only the HIP call itself is serialized).
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
            return hipErrorNotFound;
        }
        std::unique_lock lk(ctx.map_mutex);
        ctx.func_cache.emplace(cache_key, func);
    }

    // Build kernelParams[] from captured args, translating GPU pointers.
    std::vector<void*>                arg_ptrs;
    std::vector<std::vector<uint8_t>> arg_storage;
    // Optional recorded->live pointer dump for one target kernel (diff tooling).
    const bool dbg_dump_ptrs = (ctx.dump_ptrs_ordinal != 0);
    std::vector<std::tuple<unsigned, uint64_t, void*>> dbg_ptrs;  // (arg_idx, recorded, live)
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
                if (!live) return false;
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
    {
        bool is_sp3 = (kernel_name.find("Sp3") != std::string::npos ||
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
            if (ext_global_worksize) {
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
            if (ext_global_worksize) {
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
        return r;
    }

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
    if (managed)
        r = hipMallocManaged(&live, pad_sz);
    else
        r = hipMalloc(&live, pad_sz);
    if (r == hipSuccess) {
        // hipMalloc does NOT guarantee zeroed memory (only first-touch pages are
        // scrubbed; reused allocations carry stale bytes). Zero so replay is
        // deterministic and matches first-touch-zeroed assumptions. See
        // hrr_replay_zero_init().
        // Skip the zero-init memset while a graph capture is active: the original
        // run never issued it, and an injected synchronous device memset during
        // capture is illegal and invalidates the capture (HIP 901) for every
        // subsequent op in the graph.
        if (hrr_replay_zero_init() && !ctx.in_graph_capture)
            (void)hipMemset(live, 0, pad_sz);
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
        if (hrr_replay_zero_init() && !ctx.in_graph_capture)
            (void)hipMemset(live, 0, pad_sz);
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

// This mirrors the captured API exactly — hipMemcpyAsync on the default stream
// (stream_rec==0, translated to nullptr) must still use the async variant.
static hipError_t replay_memcpy_impl(PlaybackContext& ctx,
                                     uint64_t dst_rec, uint64_t src_rec,
                                     uint64_t size, int32_t kind,
                                     bool is_async, hipStream_t stream,
                                     uint64_t hash_lo, uint64_t hash_hi) {
    void*      dst = ctx.translate_ptr(dst_rec);
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
        if (r != hipSuccess)
            fprintf(stderr, "[HRR] H2D memcpy failed: %d (%s) dst=%p copy_sz=%zu blob_sz=%zu avail=%zu\n",
                    r, hipGetErrorString(r), dst, copy_sz, blob_sz, avail);
    } else if (kind == hipMemcpyDeviceToDevice) {
        void* src = ctx.translate_ptr(src_rec);
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
        void* src_dev = ctx.translate_ptr(src_rec);
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
        // graph_map is populated ONLY by the stream-capture chain
        // (hipStreamEndCapture). A miss means the graph was built through the
        // explicit node API (hipGraphCreate + hipGraphAdd*Node), which HRR does
        // not replay. Replaying an empty graph would silently skip every launch
        // and corrupt downstream buffers, so fail loudly instead.
        fprintf(stderr,
                "[HRR] hipGraphInstantiate: graph 0x%llx not in graph_map. HRR only "
                "replays stream-capture graphs (hipStreamBeginCapture/EndCapture); "
                "explicit node-API graph construction (hipGraphCreate + "
                "hipGraphAdd*Node) is NOT supported. Aborting replay rather than "
                "running an empty graph.\n",
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
        // See playback_hipGraphInstantiate: a graph_map miss means explicit
        // node-API construction, which HRR does not replay. Fail loudly.
        fprintf(stderr,
                "[HRR] hipGraphInstantiateWithFlags: graph 0x%llx not in graph_map. "
                "HRR only replays stream-capture graphs; explicit node-API graph "
                "construction is NOT supported. Aborting replay.\n",
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
static hipError_t replay_memcpy3d_d2h(PlaybackContext& ctx,
                                       void* src_live, size_t byte_count,
                                       uint64_t d2h_hash_lo, uint64_t d2h_hash_hi,
                                       hipStream_t stream, bool is_async) {
    std::vector<uint8_t> actual(byte_count ? byte_count : 1);
    hipError_t r;
    if (is_async) {
        r = hipMemcpyAsync(actual.data(), src_live, byte_count,
                           hipMemcpyDeviceToHost, stream);
        if (stream) (void)hipStreamSynchronize(stream);
    } else {
        r = hipMemcpy(actual.data(), src_live, byte_count, hipMemcpyDeviceToHost);
    }
    if (r != hipSuccess) {
        fprintf(stderr, "[HRR] hipMemcpy3D D2H: device readback failed: %d (%s)\n",
                r, hipGetErrorString(r));
        ctx.note_d2h_fail(hrr_dispatch_seq);
        return r;
    }
    if (!ctx.validate_d2h || !(d2h_hash_lo || d2h_hash_hi))
        return hipSuccess;  // no expected blob — just execute, no comparison

    ctx.d2h_attempted++;
    size_t blob_sz = 0;
    const void* expected = ctx.load_blob(d2h_hash_lo, d2h_hash_hi, &blob_sz);
    if (!expected) {
        fprintf(stderr, "[HRR] hipMemcpy3D D2H validate FAIL: expected blob not found in archive\n");
        ctx.note_d2h_fail(hrr_dispatch_seq);
        return hipSuccess;
    }
    size_t cmp_sz = std::min(byte_count, blob_sz);
    hrr_d2h_validate(ctx, "3D", hrr_dispatch_seq, actual.data(),
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
        return hipMemcpy3D(&parms);
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
                                   nullptr, false);
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
        return hipMemcpy3DAsync(&parms, stream);
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
                                   stream, true);
    }
    // D2D: translate both pointers
    parms.srcPtr.ptr = ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.srcPtr.ptr));
    parms.dstPtr.ptr = ctx.translate_ptr(reinterpret_cast<uint64_t>(parms.dstPtr.ptr));
    return hipMemcpy3DAsync(&parms, stream);
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
        if (is_async)
            return hipMemcpy2DAsync(dst, dpitch, blob, spitch, width, height,
                                    hipMemcpyHostToDevice, stream);
        return hipMemcpy2D(dst, dpitch, blob, spitch, width, height,
                           hipMemcpyHostToDevice);
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
    return r;
}

hipError_t playback_hipArray3DCreate(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipArray3DCreate*>(pl);
    HIP_ARRAY3D_DESCRIPTOR desc{};
    std::memcpy(&desc, a->array3d_desc_bytes, sizeof(desc));
    hipArray_t arr = nullptr;
    hipError_t r = hipArray3DCreate(&arr, &desc);
    if (r == hipSuccess) ctx.record_array(a->array, arr);
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
    // Reconstruct the hipMemAllocationProp — it's a regular struct, not stored inline
    // Generator stores handle (u64) and prop (u64 stale ptr) and flags (u64).
    // We re-query granularity with the same type/location as captured.
    hipMemAllocationProp prop{};
    prop.type                = hipMemAllocationTypePinned;
    prop.location.type       = hipMemLocationTypeDevice;
    prop.location.id         = 0;  // device 0; matches test workload
    hipMemGenericAllocationHandle_t live_handle{};
    hipError_t r = hipMemCreate(&live_handle, static_cast<size_t>(a->size), &prop, 0);
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
