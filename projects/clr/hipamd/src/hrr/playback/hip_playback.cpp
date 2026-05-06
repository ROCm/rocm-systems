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
#include <cstdlib>
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

static hipError_t replay_kernel_launch(PlaybackContext& ctx, const uint8_t* pl) {
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
                                          const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload);
}

hipError_t playback_hipExtModuleLaunchKernel(PlaybackContext& ctx,
                                             const uint8_t* payload) {
    return replay_kernel_launch(ctx, payload);
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
            hipModuleUnload(mod);
        }
    }
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
                                         const uint8_t* payload) {
    (void)ctx; (void)payload; 
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
    return replay_module_load(ctx, payload);
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

static hipError_t replay_malloc(PlaybackContext& ctx, const uint8_t* pl,
                                bool managed = false) {
    const auto* a = reinterpret_cast<const hrr_args_hipMalloc*>(pl);
    void* live = nullptr;
    hipError_t r;
    if (managed)
        r = hipMallocManaged(&live, static_cast<size_t>(a->size));
    else
        r = hipMalloc(&live, static_cast<size_t>(a->size));
    if (r == hipSuccess)
        ctx.record_alloc(a->ptr, live, static_cast<size_t>(a->size));
    return r;
}

hipError_t playback_hipMalloc(PlaybackContext& ctx, const uint8_t* pl) {
    return replay_malloc(ctx, pl);
}
hipError_t playback_hipMallocManaged(PlaybackContext& ctx, const uint8_t* pl) {
    return replay_malloc(ctx, pl, /*managed=*/true);
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
    hipError_t r = hipMallocAsync(&live, static_cast<size_t>(a->size), stream);
    if (r == hipSuccess)
        ctx.record_alloc(a->dev_ptr, live, static_cast<size_t>(a->size));
    return r;
}

hipError_t playback_hipMallocFromPoolAsync(PlaybackContext& ctx,
                                           const uint8_t* pl) {
    const auto* a  = reinterpret_cast<const hrr_args_hipMallocFromPoolAsync*>(pl);
    hipMemPool_t pool   = ctx.translate_mempool(a->mem_pool);
    hipStream_t  stream = ctx.translate_stream(a->stream);
    void* live = nullptr;
    hipError_t r = hipMallocFromPoolAsync(&live, static_cast<size_t>(a->size), pool, stream);
    if (r == hipSuccess)
        ctx.record_alloc(a->dev_ptr, live, static_cast<size_t>(a->size));
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
    if (r == hipSuccess) ctx.record_alloc(a->ptr, live, static_cast<size_t>(a->size));
    return r;
}

hipError_t playback_hipMallocHost(PlaybackContext& ctx, const uint8_t* pl) {
    const auto* a = reinterpret_cast<const hrr_args_hipMallocHost*>(pl);
    void* live = nullptr;
    hipError_t r = hipMallocHost(&live, static_cast<size_t>(a->size));
    if (r == hipSuccess) ctx.record_alloc(a->ptr, live, static_cast<size_t>(a->size));
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
        ctx.record_alloc(a->hostPtr, buf, sz);
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
        if (!dst) { fprintf(stderr, "[HRR] H2D dst 0x%llx not mapped\n",
                            (unsigned long long)dst_rec); return hipErrorInvalidValue; }
        size_t copy_sz = static_cast<size_t>(size);
        if (copy_sz > blob_sz) copy_sz = blob_sz;
        if (is_async)
            r = hipMemcpyAsync(dst, blob, copy_sz, hipMemcpyHostToDevice, stream);
        else
            r = hipMemcpy(dst, blob, copy_sz, hipMemcpyHostToDevice);
    } else if (kind == hipMemcpyDeviceToDevice) {
        void* src = ctx.translate_ptr(src_rec);
        if (dst && src) {
            if (is_async)
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
                // For async memcpy the stream may not yet have completed — sync it so
                // all preceding GPU work has finished before reading back.
                // Synchronous hipMemcpy already guarantees completion; no extra sync needed.
                if (is_async) hipStreamSynchronize(stream);
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
        fprintf(stderr, "[HRR] hipGraphInstantiate: graph 0x%llx not found in map\n",
                (unsigned long long)a->graph);
        return hipSuccess;  // non-fatal — launches will be skipped
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
