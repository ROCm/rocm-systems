/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */
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
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

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
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

static inline uint32_t rd32(const uint8_t* p, size_t off) {
    uint32_t v; memcpy(&v, p + off, 4); return v;
}
static inline uint64_t rd64(const uint8_t* p, size_t off) {
    uint64_t v; memcpy(&v, p + off, 8); return v;
}
static inline int32_t rdi32(const uint8_t* p, size_t off) {
    int32_t v; memcpy(&v, p + off, 4); return v;
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

}  // namespace

// ---------------------------------------------------------------------------
// PlaybackContext: blob / code-object loading
// ---------------------------------------------------------------------------

const void* PlaybackContext::load_blob(uint64_t hash_lo, uint64_t hash_hi,
                                       size_t* sz_out) const {
    if (!hash_lo && !hash_hi) return nullptr;
    auto key = hash_lo ^ (hash_hi << 1);
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
    auto key = hash_lo ^ ((hash_hi << 1) | 1u);
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
    auto it = blob_cache_.emplace(key, std::move(data)).first;
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
        hipModuleUnload(mod);
        return it->second;
    }
    if (verbose)
        fprintf(stderr, "[HRR] Loaded code object %s (%zu bytes)\n", hex.c_str(), sz);
    return mod;
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

static hipError_t replay_kernel_launch(PlaybackContext& ctx,
                                       const uint8_t* pl, size_t pl_len) {
    const uint8_t* p   = pl;
    const uint8_t* end = pl + pl_len;

    if (p + 8 > end) return hipErrorInvalidValue;
    uint64_t stream_rec; memcpy(&stream_rec, p, 8); p += 8;

    if (p + 2 > end) return hipErrorInvalidValue;
    uint16_t name_len; memcpy(&name_len, p, 2); p += 2;
    if (p + name_len > end) return hipErrorInvalidValue;
    std::string kernel_name(reinterpret_cast<const char*>(p), name_len);
    p += name_len;

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
    {
        std::shared_lock lk(ctx.map_mutex);
        auto it = ctx.func_cache.find(kernel_name);
        if (it != ctx.func_cache.end())
            func = it->second;
    }

    if (!func) {
        // Cache miss: search module_map then co_modules.
        {
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
        if (!func && (co_hash_lo || co_hash_hi)) {
            hipModule_t mod = ctx.load_module(co_hash_lo, co_hash_hi);
            if (mod) hipModuleGetFunction(&func, mod, kernel_name.c_str());
        }
        if (!func) {
            fprintf(stderr, "[HRR] Kernel '%s' not found in any loaded module\n",
                    kernel_name.c_str());
            return hipErrorNotFound;
        }
        std::unique_lock lk(ctx.map_mutex);
        ctx.func_cache.emplace(kernel_name, func);
    }

    // Build kernelParams[] from captured args, translating GPU pointers
    std::vector<void*>                arg_ptrs;
    std::vector<std::vector<uint8_t>> arg_storage;
    for (uint16_t i = 0; i < num_args; i++) {
        if (p + 3 > end) break;
        uint8_t  value_kind = *p++;
        uint16_t arg_size;
        memcpy(&arg_size, p, 2); p += 2;
        if (p + arg_size > end) break;

        if (value_kind == 2) {  // hidden arg — skip
            p += arg_size;
            continue;
        }
        arg_storage.emplace_back();
        auto& storage = arg_storage.back();
        if (value_kind == 1 && arg_size >= 8) {  // GPU pointer
            uint64_t rec_ptr; memcpy(&rec_ptr, p, 8);
            void* live = ctx.translate_ptr(rec_ptr);
            storage.resize(sizeof(void*));
            memcpy(storage.data(), &live, sizeof(void*));
            if (ctx.verbose)
                fprintf(stderr, "[HRR]   arg[%u]: ptr 0x%llx -> %p%s\n",
                        i, (unsigned long long)rec_ptr, live,
                        live ? "" : " (MISSING!)");
        } else {
            storage.assign(p, p + arg_size);
        }
        arg_ptrs.push_back(storage.data());
        p += arg_size;
    }

    hipStream_t stream = ctx.translate_stream(stream_rec);

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

    hipError_t r = hipModuleLaunchKernel(
        func,
        grid[0], grid[1], grid[2],
        block[0], block[1], block[2],
        shared_mem, stream,
        arg_ptrs.empty() ? nullptr : arg_ptrs.data(),
        nullptr);

    if (timing_ok)
        timing_ok = (HRR_HIP_CHECK(hipEventRecord(tl_stop, stream)) == hipSuccess);

    if (r != hipSuccess) {
        fprintf(stderr, "[HRR] Kernel '%s' launch error: %d (%s)\n",
                kernel_name.c_str(), r, hipGetErrorString(r));
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

    if (ctx.sync_after_launch) {
        r = hipDeviceSynchronize();
        if (r != hipSuccess)
            fprintf(stderr, "[HRR] GPU error after '%s': %d (%s)\n",
                    kernel_name.c_str(), r, hipGetErrorString(r));
        else if (ctx.verbose)
            fprintf(stderr, "[HRR] Kernel '%s' OK\n", kernel_name.c_str());
    }

    ctx.kernels_launched++;
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: kernel launches (all four variants share the same payload)
// ---------------------------------------------------------------------------

hipError_t playback_hipModuleLaunchKernel(PlaybackContext& ctx,
                                          const uint8_t* payload, size_t pl_len) {
    return replay_kernel_launch(ctx, payload, pl_len);
}

hipError_t playback_hipExtModuleLaunchKernel(PlaybackContext& ctx,
                                             const uint8_t* payload, size_t pl_len) {
    return replay_kernel_launch(ctx, payload, pl_len);
}

hipError_t playback_hipLaunchKernel(PlaybackContext& ctx,
                                    const uint8_t* payload, size_t pl_len) {
    return replay_kernel_launch(ctx, payload, pl_len);
}

hipError_t playback_hipLaunchByPtr(PlaybackContext& ctx,
                                   const uint8_t* payload, size_t pl_len) {
    return replay_kernel_launch(ctx, payload, pl_len);
}

// ---------------------------------------------------------------------------
// Manual playback: __hipRegisterFatBinary
// ---------------------------------------------------------------------------
// Payload layout (raw_payload bytes after 32-byte EventHeader):
//   ret(8) blob_hash_lo(8) blob_hash_hi(8) blob_size(8)
//
// Load the fat binary blob via hipModuleLoadData so all embedded kernel names
// become resolvable at kernel launch replay time.

hipError_t playback___hipRegisterFatBinary(PlaybackContext& ctx,
                                           const uint8_t* payload, size_t pl_len) {
    if (pl_len < 32) return hipSuccess;  // too short — skip gracefully
    uint64_t blob_hash_lo = rd64(payload,  8);
    uint64_t blob_hash_hi = rd64(payload, 16);
    uint64_t blob_size    = rd64(payload, 24);

    if (!blob_hash_lo && !blob_hash_hi) return hipSuccess;  // no blob — skip
    if (!blob_size) return hipSuccess;

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

    // Store under a synthetic key (use hash_lo as key since we have no recorded handle)
    ctx.record_module(blob_hash_lo, mod);
    if (ctx.verbose)
        fprintf(stderr, "[HRR] Loaded fat binary blob (%zu bytes) -> hipModule_t\n", sz);
    return hipSuccess;
}

// ---------------------------------------------------------------------------
// Manual playback: hipModuleGetFunction
// ---------------------------------------------------------------------------
// The function handle is stored by name — not as a fixed uint64_t mapping.
// We ignore the call during replay; functions are looked up by name at launch time.

hipError_t playback_hipModuleGetFunction(PlaybackContext& ctx,
                                         const uint8_t* payload, size_t pl_len) {
    (void)ctx; (void)payload; (void)pl_len;
    // Function handles are resolved by name at kernel launch time.
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
                                     const uint8_t* payload, size_t pl_len) {
    if (pl_len < 36) return hipErrorInvalidValue;
    uint64_t rec_module = rd64(payload,  4);  // recorded hipModule_t raw ptr
    uint64_t co_hash_lo = rd64(payload, 20);
    uint64_t co_hash_hi = rd64(payload, 28);

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
                                      const uint8_t* payload, size_t pl_len) {
    return replay_module_load(ctx, payload, pl_len);
}

hipError_t playback_hipModuleLoadDataEx(PlaybackContext& ctx,
                                        const uint8_t* payload, size_t pl_len) {
    return replay_module_load(ctx, payload, pl_len);
}

hipError_t playback_hipModuleLoad(PlaybackContext& ctx,
                                  const uint8_t* payload, size_t pl_len) {
    return replay_module_load(ctx, payload, pl_len);
}

// ---------------------------------------------------------------------------
// Manual playback: hipMalloc / hipMallocManaged / hipHostMalloc
// ---------------------------------------------------------------------------
// Payload: ret(4) ptr(8) size(8) [additional fields for managed/host variants]
// ptr at +4, size at +12

static hipError_t replay_malloc(PlaybackContext& ctx, const uint8_t* pl, size_t pl_len,
                                bool managed = false, bool host = false) {
    if (pl_len < 20) return hipErrorInvalidValue;
    uint64_t rec_ptr = rd64(pl,  4);
    uint64_t size    = rd64(pl, 12);

    void* live = nullptr;
    hipError_t r;
    if (managed)
        r = hipMallocManaged(&live, static_cast<size_t>(size));
    else if (host)
        r = hipHostMalloc(&live, static_cast<size_t>(size));
    else
        r = hipMalloc(&live, static_cast<size_t>(size));

    if (r == hipSuccess)
        ctx.record_alloc(rec_ptr, live, static_cast<size_t>(size));
    return r;
}

hipError_t playback_hipMalloc(PlaybackContext& ctx, const uint8_t* pl, size_t pl_len) {
    return replay_malloc(ctx, pl, pl_len);
}
hipError_t playback_hipMallocManaged(PlaybackContext& ctx, const uint8_t* pl, size_t pl_len) {
    return replay_malloc(ctx, pl, pl_len, /*managed=*/true);
}

// ---------------------------------------------------------------------------
// Manual playback: hipMallocAsync / hipMallocFromPoolAsync
// ---------------------------------------------------------------------------
// hipMallocAsync:  ret(4) dev_ptr(8) size(8) stream(8)
// hipMallocFromPoolAsync: ret(4) dev_ptr(8) size(8) mem_pool(8) stream(8)

hipError_t playback_hipMallocAsync(PlaybackContext& ctx,
                                   const uint8_t* pl, size_t pl_len) {
    if (pl_len < 28) return hipErrorInvalidValue;
    uint64_t rec_ptr     = rd64(pl,  4);
    uint64_t size        = rd64(pl, 12);
    uint64_t stream_rec  = rd64(pl, 20);
    hipStream_t stream   = ctx.translate_stream(stream_rec);

    void* live = nullptr;
    hipError_t r = hipMallocAsync(&live, static_cast<size_t>(size), stream);
    if (r == hipSuccess)
        ctx.record_alloc(rec_ptr, live, static_cast<size_t>(size));
    return r;
}

hipError_t playback_hipMallocFromPoolAsync(PlaybackContext& ctx,
                                           const uint8_t* pl, size_t pl_len) {
    if (pl_len < 36) return hipErrorInvalidValue;
    uint64_t rec_ptr    = rd64(pl,  4);
    uint64_t size       = rd64(pl, 12);
    uint64_t pool_rec   = rd64(pl, 20);
    uint64_t stream_rec = rd64(pl, 28);
    hipMemPool_t  pool   = ctx.translate_mempool(pool_rec);
    hipStream_t   stream = ctx.translate_stream(stream_rec);

    void* live = nullptr;
    hipError_t r = hipMallocFromPoolAsync(&live, static_cast<size_t>(size), pool, stream);
    if (r == hipSuccess)
        ctx.record_alloc(rec_ptr, live, static_cast<size_t>(size));
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipFree / hipFreeAsync
// ---------------------------------------------------------------------------
// hipFree:       ret(4) ptr(8)
// hipFreeAsync:  ret(4) dev_ptr(8) stream(8)

hipError_t playback_hipFree(PlaybackContext& ctx, const uint8_t* pl, size_t pl_len) {
    if (pl_len < 12) return hipErrorInvalidValue;
    uint64_t rec_ptr = rd64(pl, 4);
    void* live = ctx.translate_ptr(rec_ptr);
    if (!live) return hipSuccess;  // already freed or not tracked
    hipError_t r = hipFree(live);
    if (r == hipSuccess) ctx.remove_alloc(rec_ptr);
    return r;
}

hipError_t playback_hipFreeAsync(PlaybackContext& ctx, const uint8_t* pl, size_t pl_len) {
    if (pl_len < 20) return hipErrorInvalidValue;
    uint64_t rec_ptr    = rd64(pl,  4);
    uint64_t stream_rec = rd64(pl, 12);
    void*       live   = ctx.translate_ptr(rec_ptr);
    hipStream_t stream = ctx.translate_stream(stream_rec);
    if (!live) return hipSuccess;
    hipError_t r = hipFreeAsync(live, stream);
    if (r == hipSuccess) ctx.remove_alloc(rec_ptr);
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: hipMemcpy / hipMemcpyAsync / hipMemcpyHtoD / hipMemcpyHtoDAsync
// ---------------------------------------------------------------------------
// hipMemcpy:        ret(4) dst(8) src(8) sizeBytes(8) kind(4) hash_lo(8) hash_hi(8)
// hipMemcpyAsync:   ret(4) dst(8) src(8) sizeBytes(8) kind(4) stream(8) hash_lo(8) hash_hi(8)
// hipMemcpyHtoD:    ret(4) dst(8) src(8) sizeBytes(8) hash_lo(8) hash_hi(8)
// hipMemcpyHtoDAsync: ret(4) dst(8) src(8) sizeBytes(8) stream(8) hash_lo(8) hash_hi(8)

static hipError_t replay_memcpy(PlaybackContext& ctx,
                                const uint8_t* pl, size_t pl_len,
                                bool is_htod_explicit,
                                bool is_async, bool has_kind) {
    // Minimum layout depends on variant
    size_t expected = 4 + 8 + 8 + 8 + (has_kind ? 4 : 0) + (is_async ? 8 : 0) + 8 + 8;
    if (pl_len < expected) return hipErrorInvalidValue;

    size_t off = 4;
    uint64_t dst_rec  = rd64(pl, off);  off += 8;
    uint64_t src_rec  = rd64(pl, off);  off += 8;
    uint64_t size     = rd64(pl, off);  off += 8;
    int32_t  kind     = is_htod_explicit ? 1 /* hipMemcpyHostToDevice */
                                         : rdi32(pl, off);
    if (has_kind) off += 4;
    uint64_t stream_rec = 0;
    if (is_async) { stream_rec = rd64(pl, off); off += 8; }
    uint64_t hash_lo = rd64(pl, off); off += 8;
    uint64_t hash_hi = rd64(pl, off);

    hipStream_t stream = ctx.translate_stream(stream_rec);
    void*       dst    = ctx.translate_ptr(dst_rec);
    hipError_t  r      = hipSuccess;

    if (kind == hipMemcpyHostToDevice && (hash_lo || hash_hi)) {
        size_t blob_sz = 0;
        const void* blob = ctx.load_blob(hash_lo, hash_hi, &blob_sz);
        if (!blob) {
            fprintf(stderr, "[HRR] H2D blob %016llx%016llx not found\n",
                    (unsigned long long)hash_lo, (unsigned long long)hash_hi);
            return hipErrorNotFound;
        }
        if (!dst) { fprintf(stderr, "[HRR] H2D dst 0x%llx not mapped\n",
                            (unsigned long long)dst_rec); return hipErrorInvalidValue; }
        size_t copy_sz = static_cast<size_t>(size);
        if (copy_sz > blob_sz) copy_sz = blob_sz;
        if (stream)
            r = hipMemcpyAsync(dst, blob, copy_sz, hipMemcpyHostToDevice, stream);
        else
            r = hipMemcpy(dst, blob, copy_sz, hipMemcpyHostToDevice);
    } else if (kind == hipMemcpyDeviceToDevice) {
        void* src = ctx.translate_ptr(src_rec);
        if (dst && src) {
            if (stream)
                r = hipMemcpyAsync(dst, src, static_cast<size_t>(size),
                                   hipMemcpyDeviceToDevice, stream);
            else
                r = hipMemcpy(dst, src, static_cast<size_t>(size),
                              hipMemcpyDeviceToDevice);
        }
    } else if (kind == hipMemcpyDeviceToHost && ctx.validate_d2h &&
               (hash_lo || hash_hi)) {
        // D2H validation: copy from live device src into a local host buffer,
        // then compare against the expected data blob captured at record time.
        void* src_dev = ctx.translate_ptr(src_rec);
        if (!src_dev) {
            if (ctx.verbose)
                fprintf(stderr, "[HRR] D2H validate: src 0x%llx not mapped — skip\n",
                        (unsigned long long)src_rec);
        } else {
            size_t copy_sz = static_cast<size_t>(size);
            size_t blob_sz = 0;
            const void* expected = ctx.load_blob(hash_lo, hash_hi, &blob_sz);
            if (!expected) {
                fprintf(stderr, "[HRR] D2H validate: expected blob not found — skip\n");
            } else {
                copy_sz = std::min(copy_sz, blob_sz);
                std::vector<uint8_t> actual(copy_sz);
                r = hipMemcpy(actual.data(), src_dev, copy_sz, hipMemcpyDeviceToHost);
                if (r != hipSuccess) {
                    fprintf(stderr, "[HRR] D2H validate: hipMemcpy failed: %d (%s)\n",
                            r, hipGetErrorString(r));
                    ctx.d2h_fail++;
                } else if (memcmp(actual.data(), expected, copy_sz) == 0) {
                    ctx.d2h_pass++;
                    if (ctx.verbose)
                        fprintf(stderr, "[HRR] D2H validate: %zu bytes OK\n", copy_sz);
                } else {
                    ctx.d2h_fail++;
                    // Find first differing byte for diagnostics
                    size_t first_diff = 0;
                    const uint8_t* exp = static_cast<const uint8_t*>(expected);
                    while (first_diff < copy_sz && actual[first_diff] == exp[first_diff])
                        ++first_diff;
                    fprintf(stderr,
                            "[HRR] D2H validate FAIL: %zu bytes, first diff at byte %zu "
                            "(got 0x%02x expected 0x%02x)\n",
                            copy_sz, first_diff,
                            actual[first_diff], exp[first_diff]);
                }
            }
        }
    }
    // H2H / unhandled: no-op
    return r;
}

hipError_t playback_hipMemcpy(PlaybackContext& ctx,
                              const uint8_t* pl, size_t pl_len) {
    return replay_memcpy(ctx, pl, pl_len,
                         /*is_htod_explicit=*/false, /*is_async=*/false, /*has_kind=*/true);
}

hipError_t playback_hipMemcpyAsync(PlaybackContext& ctx,
                                   const uint8_t* pl, size_t pl_len) {
    return replay_memcpy(ctx, pl, pl_len,
                         /*is_htod_explicit=*/false, /*is_async=*/true, /*has_kind=*/true);
}

hipError_t playback_hipMemcpyHtoD(PlaybackContext& ctx,
                                  const uint8_t* pl, size_t pl_len) {
    return replay_memcpy(ctx, pl, pl_len,
                         /*is_htod_explicit=*/true, /*is_async=*/false, /*has_kind=*/false);
}

hipError_t playback_hipMemcpyHtoDAsync(PlaybackContext& ctx,
                                       const uint8_t* pl, size_t pl_len) {
    return replay_memcpy(ctx, pl, pl_len,
                         /*is_htod_explicit=*/true, /*is_async=*/true, /*has_kind=*/false);
}

// ---------------------------------------------------------------------------
// Manual playback: stream create/destroy
// ---------------------------------------------------------------------------
// hipStreamCreate:              ret(4) stream(8)
// hipStreamCreateWithFlags:     ret(4) stream(8) flags(4)
// hipStreamCreateWithPriority:  ret(4) stream(8) flags(4) priority(4)
// hipStreamDestroy:             ret(4) stream(8)

hipError_t playback_hipStreamCreate(PlaybackContext& ctx,
                                    const uint8_t* pl, size_t pl_len) {
    if (pl_len < 12) return hipErrorInvalidValue;
    uint64_t rec = rd64(pl, 4);
    hipStream_t s = nullptr;
    hipError_t r = hipStreamCreate(&s);
    if (r == hipSuccess) ctx.record_stream(rec, s);
    return r;
}

hipError_t playback_hipStreamCreateWithFlags(PlaybackContext& ctx,
                                             const uint8_t* pl, size_t pl_len) {
    if (pl_len < 16) return hipErrorInvalidValue;
    uint64_t rec   = rd64(pl, 4);
    uint32_t flags = rd32(pl, 12);
    hipStream_t s  = nullptr;
    hipError_t r   = hipStreamCreateWithFlags(&s, flags);
    if (r == hipSuccess) {
        ctx.record_stream(rec, s);
        if (ctx.verbose)
            fprintf(stderr, "[HRR] StreamCreateWithFlags: rec=0x%llx -> live=%p\n",
                    (unsigned long long)rec, (void*)s);
    }
    return r;
}

hipError_t playback_hipStreamCreateWithPriority(PlaybackContext& ctx,
                                                const uint8_t* pl, size_t pl_len) {
    if (pl_len < 20) return hipErrorInvalidValue;
    uint64_t rec    = rd64(pl,  4);
    uint32_t flags  = rd32(pl, 12);
    int32_t  pri    = rdi32(pl, 16);
    hipStream_t s   = nullptr;
    hipError_t  r   = hipStreamCreateWithPriority(&s, flags, pri);
    if (r == hipSuccess) ctx.record_stream(rec, s);
    return r;
}

hipError_t playback_hipStreamDestroy(PlaybackContext& ctx,
                                     const uint8_t* pl, size_t pl_len) {
    if (pl_len < 12) return hipErrorInvalidValue;
    uint64_t rec       = rd64(pl, 4);
    hipStream_t stream = ctx.translate_stream(rec);
    hipError_t r = hipSuccess;
    if (stream) r = hipStreamDestroy(stream);
    ctx.remove_stream(rec);
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
                                          const uint8_t* payload, size_t pl_len) {
    if (pl_len < 16) return hipSuccess;
    int32_t  recorded_ret  = (int32_t)rd32(payload,  0);
    uint64_t recorded_strm = rd64(payload,  4);
    int32_t  recorded_mode = (int32_t)rd32(payload, 12);

    if (recorded_ret != hipSuccess) return hipSuccess;  // original failed — skip

    hipStream_t stream = ctx.translate_stream(recorded_strm);
    if (!stream && recorded_strm != 0) {
        // Stream handle not in map — create a temporary stream for graph capture
        fprintf(stderr, "[HRR] hipStreamBeginCapture: stream 0x%llx not found, "
                "creating temp stream for graph capture\n",
                (unsigned long long)recorded_strm);
        hipError_t cr = hipStreamCreate(&stream);
        if (cr != hipSuccess) {
            fprintf(stderr, "[HRR] hipStreamBeginCapture: failed to create temp stream: %d\n", cr);
            return hipSuccess;  // non-fatal
        }
        ctx.record_stream(recorded_strm, stream);
    }

    hipStreamCaptureMode mode = (hipStreamCaptureMode)recorded_mode;
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
                                        const uint8_t* payload, size_t pl_len) {
    // Payload layout (after 32-byte EventHeader):
    //   ret(4) stream(8) pGraph(8)
    if (pl_len < 20) return hipSuccess;
    int32_t  rec_ret   = (int32_t)rd32(payload,  0);
    uint64_t rec_strm  = rd64(payload,  4);
    uint64_t rec_pgraph = rd64(payload, 12);

    if (rec_ret != hipSuccess) return hipSuccess;  // original call failed — skip

    hipStream_t stream = ctx.translate_stream(rec_strm);
    if (!stream) {
        fprintf(stderr, "[HRR] hipStreamEndCapture: stream 0x%llx not found in map\n",
                (unsigned long long)rec_strm);
        return hipSuccess;  // non-fatal
    }
    ctx.in_graph_capture = false;
    hipGraph_t live_graph = nullptr;
    hipError_t r = hipStreamEndCapture(stream, &live_graph);
    if (r == hipSuccess && live_graph) {
        ctx.record_graph(rec_pgraph, live_graph);
        if (ctx.verbose)
            fprintf(stderr, "[HRR] hipStreamEndCapture: recorded graph 0x%llx\n",
                    (unsigned long long)rec_pgraph);
    } else {
        fprintf(stderr, "[HRR] hipStreamEndCapture failed: %d (%s)\n",
                r, hipGetErrorString(r));
    }
    return r;
}

hipError_t playback_hipGraphInstantiate(PlaybackContext& ctx,
                                        const uint8_t* payload, size_t pl_len) {
    // Payload layout (after 32-byte EventHeader):
    //   ret(4) pGraphExec(8) graph(8) pErrorNode(8) pLogBuffer(8) bufferSize(8)
    if (pl_len < 20) return hipSuccess;
    int32_t  rec_ret   = (int32_t)rd32(payload,  0);
    uint64_t rec_pexec = rd64(payload,  4);
    uint64_t rec_graph = rd64(payload, 12);

    if (rec_ret != hipSuccess) return hipSuccess;  // original call failed — skip

    hipGraph_t graph = ctx.translate_graph(rec_graph);
    if (!graph) {
        fprintf(stderr, "[HRR] hipGraphInstantiate: graph 0x%llx not found in map\n",
                (unsigned long long)rec_graph);
        return hipSuccess;  // non-fatal — launches will be skipped
    }

    hipGraphExec_t exec = nullptr;
    // Use the simplified WithFlags variant; pErrorNode/pLogBuffer are optional at replay
    hipError_t r = hipGraphInstantiateWithFlags(&exec, graph, 0);
    if (r == hipSuccess && exec) {
        ctx.record_graph_exec(rec_pexec, exec);
        if (ctx.verbose)
            fprintf(stderr, "[HRR] hipGraphInstantiate: recorded exec 0x%llx\n",
                    (unsigned long long)rec_pexec);
    } else {
        fprintf(stderr, "[HRR] hipGraphInstantiate (via WithFlags) failed: %d (%s)\n",
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
                                   const uint8_t* payload, size_t pl_len) {
    if (pl_len < 20) return hipSuccess;
    int32_t  rec_ret    = (int32_t)rd32(payload,  0);
    uint64_t rec_exec   = rd64(payload,  4);
    uint64_t rec_strm   = rd64(payload, 12);

    if (rec_ret != hipSuccess) return hipSuccess;  // original call failed — skip

    hipGraphExec_t exec = ctx.translate_graph_exec(rec_exec);
    if (!exec) {
        if (ctx.verbose)
            fprintf(stderr, "[HRR] hipGraphLaunch: graphExec 0x%llx not found in map\n",
                    (unsigned long long)rec_exec);
        return hipSuccess;  // non-fatal — exec not yet created
    }

    hipStream_t stream = ctx.translate_stream(rec_strm);

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
                (unsigned long long)rec_exec, (unsigned long long)rec_strm);
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
                (unsigned long long)rec_exec, (unsigned long long)rec_strm);
    return r;
}

// ---------------------------------------------------------------------------
// Manual playback: event create/destroy
// ---------------------------------------------------------------------------
// hipEventCreate:            ret(4) event(8)
// hipEventCreateWithFlags:   ret(4) event(8) flags(4)
// hipEventDestroy:           ret(4) event(8)

hipError_t playback_hipEventCreate(PlaybackContext& ctx,
                                   const uint8_t* pl, size_t pl_len) {
    if (pl_len < 12) return hipErrorInvalidValue;
    uint64_t rec = rd64(pl, 4);
    hipEvent_t e = nullptr;
    hipError_t r = hipEventCreate(&e);
    if (r == hipSuccess) ctx.record_event(rec, e);
    return r;
}

hipError_t playback_hipEventCreateWithFlags(PlaybackContext& ctx,
                                            const uint8_t* pl, size_t pl_len) {
    if (pl_len < 16) return hipErrorInvalidValue;
    uint64_t rec   = rd64(pl, 4);
    uint32_t flags = rd32(pl, 12);
    hipEvent_t e   = nullptr;
    hipError_t r   = hipEventCreateWithFlags(&e, flags);
    if (r == hipSuccess) ctx.record_event(rec, e);
    return r;
}

hipError_t playback_hipEventDestroy(PlaybackContext& ctx,
                                    const uint8_t* pl, size_t pl_len) {
    if (pl_len < 12) return hipErrorInvalidValue;
    uint64_t rec     = rd64(pl, 4);
    hipEvent_t event = ctx.translate_event(rec);
    hipError_t r = hipSuccess;
    if (event) r = hipEventDestroy(event);
    ctx.remove_event(rec);
    return r;
}
