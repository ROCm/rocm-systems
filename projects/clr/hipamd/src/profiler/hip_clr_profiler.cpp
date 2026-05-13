/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * hip_clr_profiler.cpp — Built-in HIP CLR profiling layer.
 *
 * Activated by GPU_CLR_PROFILE_OUTPUT=<path> or programmatically via the
 * hipProfiler*Ext extension API declared in hip_profiler_ext.h.
 *
 * Design mirrors the reference ICD tracer (hip_tracer_core.cpp):
 *
 * CPU timing — dispatch table wrappers in hip_clr_dispatch_wrappers.cpp:
 *   auto* record = HipGetActiveRecordExt(api_id);   // allocs slot N, sets correlation_id TLS = N
 *   auto _r = g_next.hipFoo_fn(...);                // GPU command inherits correlation_id N
 *   record->end_ns = NowNs();
 *
 * GPU timing — ReportActivityCallback (ACTIVITY_DOMAIN_HIP_OPS):
 *   ar->correlation_id == N  →  index directly into g_records[N/chunk][N%chunk]
 *   No map, no TLS sentinel, no pending table.
 *
 * Chunk storage: g_records holds hipApiRecordExt arrays.
 *   Op-1 lives in rec.gpu; ops 2..N are a linked list via rec.gpu.next (graph launches only).
 *   Each node is individually heap-allocated. Freed by FreeChunk walking the list.
 */

#include "hip_clr_profiler.hpp"
#include "hip/amd_detail/hip_api_trace.hpp"
#include "platform/activity.hpp"
#include "../hip_internal.hpp"
#include "../hip_global.hpp"

#include "rocclr/os/os.hpp"
#include "platform/runtime.hpp"
#include "utils/flags.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <thread>
#if !defined(_WIN32)
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <map>
#include <unordered_map>
#include <unordered_set>

// Forward-declare CLR's comgr-based demangler (defined in hip_comgr_helper.cpp).
namespace hip { namespace helpers {
bool demangleName(const std::string& mangledName, std::string& demangledName);
} }


extern "C" void hipRegisterTracerCallback(int (*function)(activity_domain_t domain,
                                                          uint32_t operation_id, void* data));

// ================================================================================================
// Internal state
namespace {

inline uint64_t NowNs() { return amd::Os::timeNanos(); }

constexpr size_t   kChunkSize           = 10000;
// Empirical delivery margin: records whose chunk_id is at least this far behind
// the highest GPU-callback chunk_id are considered fully complete.
constexpr uint32_t kDeliveryMargin      = 3;

// Two independent enable paths:
//   g_enable_refcount  — incremented by hipProfilerEnableExt, decremented by Disable.
// Recording is active when EITHER is live.
std::atomic<int>         g_enable_refcount{0};
std::atomic<bool>        g_callback_registered{false};

inline bool IsProfilingActive() {
  return !flagIsDefault(GPU_CLR_PROFILE_OUTPUT) ||
         g_enable_refcount.load(std::memory_order_acquire) > 0;
}

// Previously registered callback saved before we register ours.
// Forwarded to at the end of HipActivityCallbackExt so we can coexist
// with roctracer / rocprofiler that may have registered first.
using activity_callback_t = int(*)(activity_domain_t, uint32_t, void*);
std::atomic<activity_callback_t> g_prev_callback{nullptr};

// Kernel name interning table — maps mangled name string to an owned copy.
// Keyed by value (std::string) so both the HSA activity callback path and the
// graph-node instantiation path can intern names into the same table.
// Values start as the mangled name and are replaced with demangled in
// PreDemangleKernelNames().  Pointers into values (std::string::c_str()) are
// stable for the lifetime of the map entry (unordered_map never moves values).
std::unordered_map<std::string, std::string> g_kernel_names;
std::mutex                                   g_kernel_names_mtx;

// Chunks of hipApiRecordExt.  Must be reserved to at least kMaxChunks before any
// recording starts.  HipGetActiveRecordExt reads g_records.size() and dereferences
// g_records[idx] without holding g_alloc_mtx (fast path).  If push_back ever
// reallocates the pointer vector, those bare reads race with the reallocation —
// undefined behaviour.  reserve() keeps capacity above the watermark so push_back
// never triggers a realloc.
std::vector<hipApiRecordExt*> g_records;
constexpr size_t kMaxChunks = 100000;  // hard cap: 100000 * 10000 = 1B records max
std::atomic<size_t>           g_rec_counter{0};
std::mutex                    g_alloc_mtx;

// Maximum chunk_id seen across all GPU activity callbacks.  Updated with a fetch_max
// (CAS loop) under memory_order_release in HipActivityCallbackExt so the delivery
// thread's acquire load establishes happens-before with all GPU writes for records
// up to and including that id.
static std::atomic<uint32_t> g_max_gpu_chunk_id{0};

// ── Chunk callback delivery ───────────────────────────────────────────────────
// When g_chunk_clients is non-empty, a background thread delivers records in
// chunk_id order as GPU activity completes, invoking every registered client.
// When empty, records are buffered and written at exit (existing buffered path).
struct HipChunkClient {
  hipProfilerChunkCallback cb;
  void*                    user_data;
};
static std::vector<HipChunkClient> g_chunk_clients;
static std::mutex                  g_chunk_clients_mtx;
// Monotonically increasing sequence number assigned to each new record slot.
static std::atomic<uint32_t>    g_next_chunk_id{0};

static std::mutex               g_chunk_mtx;
static std::condition_variable  g_chunk_cv;
static std::thread              g_chunk_thread;
static std::atomic<bool>        g_chunk_thread_stop{false};

// ================================================================================================
hipApiRecordExt* AllocChunk() {
  void* raw = ::operator new[](kChunkSize * sizeof(hipApiRecordExt));
  hipApiRecordExt* chunk = static_cast<hipApiRecordExt*>(raw);
  std::memset(chunk, 0, kChunkSize * sizeof(hipApiRecordExt));
  return chunk;
}

// ================================================================================================
void FreeChunk(hipApiRecordExt* chunk) {
  if (!chunk) return;
  for (size_t i = 0; i < kChunkSize; ++i) {
    // Free spill node kernel_args blobs and the nodes themselves.
    // All kernel_args blobs are owned: single launches by HipCaptureKernelArgsExt,
    // graph launches by a copy made in fill_dispatch_info.
    const hipGpuActivityExt* node = chunk[i].gpu.next;
    while (node) {
      const hipGpuActivityExt* next = node->next;
      delete[] node->kernel_args;
      delete node;
      node = next;
    }
    // Free the first-op kernel_args blob (owned by this record).
    delete[] chunk[i].gpu.kernel_args;
  }
  ::operator delete[](static_cast<void*>(chunk));
}

// ================================================================================================
// Free the resources owned by a single record (GPU spill list + kernel_args blobs).
// Does NOT free the record struct itself — records live in chunk arrays owned by g_records.
// Called by the delivery thread after the chunk callback returns.
static void FreeRecordResources(hipApiRecordExt* rec) {
  // Free spill node kernel_args blobs and the nodes themselves.
  const hipGpuActivityExt* node = rec->gpu.next;
  while (node) {
    const hipGpuActivityExt* nxt = node->next;
    delete[] node->kernel_args;
    delete node;
    node = nxt;
  }
  rec->gpu.next = nullptr;
  rec->_spill_tail = nullptr;
  // Free the first-op kernel_args blob.
  delete[] rec->gpu.kernel_args;
  rec->gpu.kernel_args = nullptr;
  rec->gpu.kernel_args_size = 0;
}

// Forward declaration — PfChunkCallback is defined later in the file and
// registered as an internal chunk client when pftrace output is enabled.
static FILE*                    g_pf_file           = nullptr;
static std::atomic<uint32_t>   g_pf_slabs_written{0}; // incremented each time PfChunkCallback delivers
static void  PfChunkCallback(const hipApiRecordExt*, uint32_t, uint32_t, void*);

// ================================================================================================
// Background thread: delivers completed records to all registered g_chunk_clients.
//
// Uses two counters for synchronisation:
//   g_max_gpu_chunk_id — highest chunk_id whose GPU callback has completed (fetch_max,
//                         release store in HipActivityCallbackExt).
//   next_slab          — local to this thread; index of the next slab to deliver.
//
// Delivery granularity is always one full kChunkSize slab. We never deliver a partial slab
// mid-stream — only the last slab at exit may be partial (records allocated but not kChunkSize).
//
// Mid-stream watermark: (g_max_gpu_chunk_id - kDeliveryMargin) / kChunkSize
//   Only slabs whose last record is at least kDeliveryMargin behind the max GPU callback
//   chunk_id are considered complete and safe to deliver.
//
// Stop-flush: deliver all remaining slabs up to the last allocated record.
static void ChunkDeliveryThread() {
  uint32_t next_slab = 0;

  while (true) {
    {
      std::unique_lock<std::mutex> lk(g_chunk_mtx);
      g_chunk_cv.wait(lk, [&] {
        if (g_chunk_thread_stop.load(std::memory_order_relaxed)) return true;
        uint32_t max_id  = g_max_gpu_chunk_id.load(std::memory_order_acquire);
        // Wake when the GPU callback watermark is kDeliveryMargin full slabs
        // ahead of next_slab.  Slab-granular margin (not record-granular) gives
        // ~kChunkSize * kDeliveryMargin records of safety, making it safe even
        // when GPU callbacks arrive significantly out of order.
        uint32_t max_slab = max_id / static_cast<uint32_t>(kChunkSize);
        return max_slab > next_slab + kDeliveryMargin;
      });
    }

    bool stopping = g_chunk_thread_stop.load(std::memory_order_relaxed);
    uint32_t total = g_next_chunk_id.load(std::memory_order_acquire);

    // Snapshot client list once per wakeup.
    std::vector<HipChunkClient> clients;
    {
      std::lock_guard<std::mutex> lk(g_chunk_clients_mtx);
      clients = g_chunk_clients;
    }

    auto deliver_slab = [&](uint32_t slab_idx, uint32_t count) {
      if (slab_idx >= g_records.size() || !g_records[slab_idx]) return;
      hipApiRecordExt* ptr = g_records[slab_idx];
      for (auto& c : clients)
        c.cb(ptr, count, slab_idx, c.user_data);
      for (uint32_t i = 0; i < count; ++i)
        FreeRecordResources(&ptr[i]);
      g_records[slab_idx] = nullptr;
    };

    // Deliver all complete slabs that are kDeliveryMargin full slabs behind the
    // GPU callback watermark.  max_slab is the slab of the highest GPU callback;
    // we only deliver slabs strictly before max_slab - kDeliveryMargin.
    uint32_t max_id   = g_max_gpu_chunk_id.load(std::memory_order_acquire);
    uint32_t max_slab = max_id / static_cast<uint32_t>(kChunkSize);
    if (total >= static_cast<uint32_t>(kChunkSize) &&
        max_slab > next_slab + kDeliveryMargin) {
      uint32_t watermark_slab = max_slab - kDeliveryMargin - 1;
      while (next_slab <= watermark_slab) {
        deliver_slab(next_slab, static_cast<uint32_t>(kChunkSize));
        ++next_slab;
      }
    }

    if (stopping) {
      // Flush the partial last slab (records allocated but slab not yet full).
      uint32_t remainder = total % static_cast<uint32_t>(kChunkSize);
      if (remainder > 0) deliver_slab(next_slab, remainder);
      break;
    }
  }
}

// ================================================================================================
// Convert internal CL_COMMAND_* kind to the public HipCopyKindExt enum.
// Called once per copy activity record; result stored in record at capture time
// so the public API and JSON writer never see raw OpenCL constants.
static HipCopyKindExt ToCopyKindExt(uint32_t cl_kind) {
  switch (cl_kind) {
    case CL_COMMAND_WRITE_BUFFER:           return HIP_COPY_KIND_H2D_EXT;
    case CL_COMMAND_WRITE_BUFFER_RECT:      return HIP_COPY_KIND_H2D_RECT_EXT;
    case CL_COMMAND_WRITE_IMAGE:            return HIP_COPY_KIND_H2D_IMAGE_EXT;
    case CL_COMMAND_READ_BUFFER:            return HIP_COPY_KIND_D2H_EXT;
    case CL_COMMAND_READ_BUFFER_RECT:       return HIP_COPY_KIND_D2H_RECT_EXT;
    case CL_COMMAND_READ_IMAGE:             return HIP_COPY_KIND_D2H_IMAGE_EXT;
    case CL_COMMAND_COPY_BUFFER:            return HIP_COPY_KIND_D2D_EXT;
    case CL_COMMAND_COPY_BUFFER_RECT:       return HIP_COPY_KIND_D2D_RECT_EXT;
    case CL_COMMAND_COPY_IMAGE:             return HIP_COPY_KIND_D2D_IMAGE_EXT;
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:   return HIP_COPY_KIND_BUFFER_TO_IMAGE_EXT;
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:   return HIP_COPY_KIND_IMAGE_TO_BUFFER_EXT;
    case CL_COMMAND_FILL_BUFFER:            return HIP_COPY_KIND_FILL_EXT;
    default:                                return HIP_COPY_KIND_UNKNOWN_EXT;
  }
}

// ── Graph exec → node info table ─────────────────────────────────────────────
// Populated at hipGraphInstantiate time; erased at hipGraphExecDestroy.
// Looked up in HipActivityCallbackExt to fill per-node dims+kargs in spill nodes.
// Declared here (before HipActivityCallbackExt) so the callback lambdas can
// access the map directly under the lock without calling HipGetGraphExecNodesExt.
std::unordered_map<uintptr_t, std::vector<HipGraphNodeInfoExt>> g_graph_exec_nodes;
std::mutex g_graph_exec_mtx;

// ============================================================
// GPU ops callback — same logic as reference ReportActivityCallback.
// correlation_id == slot index → direct array lookup, no map needed.
// ============================================================
int HipActivityCallbackExt(activity_domain_t domain, uint32_t op_id, void* data) {
  // Return 1 (disabled) for HIP_API domain so api_callbacks_spawner_t does NOT
  // overwrite amd::activity_prof::correlation_id with its own auto-increment value.
  // Our slot index written in HipGetActiveRecordExt must survive intact.
  if (domain != ACTIVITY_DOMAIN_HIP_OPS) return 1;

  // When disabled, forward to prev (if any) and return its answer.
  // This lets roctracer/rocprofiler stay active even when we are not.
  if (!IsProfilingActive()) {
    auto* prev = g_prev_callback.load(std::memory_order_acquire);
    return prev ? prev(domain, op_id, data) : 1;
  }

  // IsEnabled probe — we are active; also forward so prev can enable GPU ops.
  if (data == nullptr) {
    auto* prev = g_prev_callback.load(std::memory_order_acquire);
    if (prev) prev(domain, op_id, data);
    return 0;
  }

  // CommitRecord sentinel (0x1) — not needed by our design; forward to prev.
  if (reinterpret_cast<uintptr_t>(data) == 1) {
    auto* prev = g_prev_callback.load(std::memory_order_acquire);
    if (prev) prev(domain, op_id, data);
    return 0;
  }

  auto* ar = static_cast<activity_record_t*>(data);
  uint64_t slot = ar->correlation_id;

  size_t idx = slot / kChunkSize;
  if (idx >= g_records.size()) return 0;

  // g_records[idx] may be nullptr if the chunk was already delivered and freed
  // by the streaming delivery thread — ignore late GPU callbacks for freed slabs.
  if (!g_records[idx]) return 0;

  hipApiRecordExt* rec = &g_records[idx][slot % kChunkSize];

  // OP_ID_BARRIER maps exclusively to CL_COMMAND_MARKER, which is used for all
  // GPU-side synchronization (graph node barriers, hipStreamWaitEvent, hipEventRecord
  // waits). Markers execute on the queue and carry real begin/end timestamps — record
  // them all so the trace shows where barriers land on the GPU timeline.

  if (ar->op == OP_ID_DISPATCH && ar->kernel_name) {
    // Eagerly copy the mangled name string now — ar->kernel_name is a pointer into
    // the HIP runtime kernel object which may be freed before WriteJsonTraceImpl runs
    // at process exit.  Demangling still happens lazily at write time, but only using
    // the owned copy in g_kernel_names::second (never gop.kernel_name directly).
    // Calling COMGR from within the GPU activity callback is unsafe (re-entrancy risk).
    std::lock_guard<std::mutex> lk(g_kernel_names_mtx);
    auto [it, ok] = g_kernel_names.emplace(ar->kernel_name, std::string{ar->kernel_name});
    (void)ok;
    // Store pointer into the key (mangled name), which is stable for the map's lifetime.
    // The value (it->second) gets replaced with the demangled name at write time, so
    // gop.kernel_name must point to the key — not the value — for find() to work after
    // PreDemangleKernelNames() runs.
    ar->kernel_name = it->first.c_str();
  }

  // Helper: populate dims and kernel args for a dispatch GPU op.
  // For single launches, dims come from the CPU record; for graph launches,
  // look up the node info table using the graphExec stored in rec->memory1.
  //
  // Distinguish single vs. graph launch by checking api_name: hipGraphLaunch
  // records have "hipGraphLaunch" (or the _spt variant) as their api_name.
  // All other kernel-launching wrappers set grid_x via the union, which would
  // alias memory1 for graph launches and produce garbage dims.
  auto is_graph_launch = [&]() -> bool {
    const char* n = rec->api_name;
    return n && (strncmp(n, "hipGraphLaunch", 14) == 0);
  };

  auto fill_dispatch_info = [&](hipGpuActivityExt* gact) {
    if (ar->op != OP_ID_DISPATCH) return;
    gact->kernel_name = ar->kernel_name;
    if (!is_graph_launch()) {
      // Single launch: dims and kernel_args were already written by the wrapper
      // directly into rec->gpu (which IS gact for the first-op branch).
      // Spill nodes don't occur for single launches, so no action needed here.
    } else {
      // Graph launch: look up node info by exec handle stored in memory1.
      // Copy the kernel_args blob so every hipGpuActivityExt owns its blob
      // and FreeChunk can unconditionally delete[] it.
      auto* exec = reinterpret_cast<hipGraphExec_t>(rec->memory1);
      std::lock_guard<std::mutex> lk(g_graph_exec_mtx);
      auto it = g_graph_exec_nodes.find(reinterpret_cast<uintptr_t>(exec));
      if (it != g_graph_exec_nodes.end() && ar->kernel_name) {
        for (const auto& ni : it->second) {
          if (ni.gpu.op == HIP_OP_DISPATCH_EXT &&
              ni.gpu.kernel_name &&
              strcmp(ni.gpu.kernel_name, ar->kernel_name) == 0) {
            gact->grid_x  = ni.gpu.grid_x;
            gact->grid_y  = ni.gpu.grid_y;
            gact->grid_z  = ni.gpu.grid_z;
            gact->block_x = ni.gpu.block_x;
            gact->block_y = ni.gpu.block_y;
            gact->block_z = ni.gpu.block_z;
            if (ni.gpu.kernel_args && ni.gpu.kernel_args_size > 0) {
              uint8_t* copy = new uint8_t[ni.gpu.kernel_args_size];
              std::memcpy(copy, ni.gpu.kernel_args, ni.gpu.kernel_args_size);
              gact->kernel_args      = copy;
              gact->kernel_args_size = ni.gpu.kernel_args_size;
            }
            break;
          }
        }
      }
    }
  };

  // Helper: populate src/dst for a copy GPU op.
  // For direct (non-graph) copies, src/dst come from the CPU record (set by the wrapper).
  // For graph copy nodes, look up the matching copy node info captured at instantiate time
  // by matching on copy_kind and bytes; first match wins (best-effort for duplicate nodes).
  auto fill_copy_info = [&](hipGpuActivityExt* gact) {
    if (ar->op != OP_ID_COPY) return;
    if (!is_graph_launch()) {
      gact->src = rec->memory2;
      gact->dst = rec->memory1;
    } else {
      auto* exec = reinterpret_cast<hipGraphExec_t>(rec->memory1);
      std::lock_guard<std::mutex> lk(g_graph_exec_mtx);
      auto it = g_graph_exec_nodes.find(reinterpret_cast<uintptr_t>(exec));
      if (it != g_graph_exec_nodes.end()) {
        HipCopyKindExt ck = static_cast<HipCopyKindExt>(gact->copy_kind);
        for (const auto& ni : it->second) {
          if (ni.gpu.op == HIP_OP_COPY_EXT &&
              static_cast<HipCopyKindExt>(ni.gpu.copy_kind) == ck &&
              ni.gpu.bytes == gact->bytes) {
            gact->src = ni.gpu.src;
            gact->dst = ni.gpu.dst;
            break;
          }
        }
      }
    }
  };

  if (rec->gpu.gpu_op_count == 0) {
    // First op: write directly into the embedded gpu field — no heap alloc.
    rec->gpu.op        = ar->op;
    rec->gpu.begin_ns  = ar->begin_ns;
    rec->gpu.end_ns    = ar->end_ns;
    rec->gpu.device_id = ar->device_id;
    rec->gpu.queue_id  = ar->queue_id;
    if (ar->op == OP_ID_DISPATCH) {
      fill_dispatch_info(&rec->gpu);
    } else if (ar->op == OP_ID_COPY) {
      rec->gpu.bytes     = ar->bytes;
      rec->gpu.copy_kind = ToCopyKindExt(ar->kind);
      fill_copy_info(&rec->gpu);
    }
    // OP_ID_BARRIER: no payload fields — begin/end timestamps already written above.
    rec->gpu.gpu_op_count  = 1;
    rec->has_gpu_activity  = 1;
  } else {
    // Subsequent op (graph launch spill): allocate a new node and O(1)-append via _spill_tail.
    hipGpuActivityExt* node = new hipGpuActivityExt{};
    node->op        = ar->op;
    node->begin_ns  = ar->begin_ns;
    node->end_ns    = ar->end_ns;
    node->device_id = ar->device_id;
    node->queue_id  = ar->queue_id;
    if (ar->op == OP_ID_DISPATCH) {
      fill_dispatch_info(node);
    } else if (ar->op == OP_ID_COPY) {
      node->bytes     = ar->bytes;
      node->copy_kind = ToCopyKindExt(ar->kind);
      fill_copy_info(node);
    }
    node->next = nullptr;

    // _spill_tail caches the list tail so append is O(1).
    hipGpuActivityExt* tail =
      const_cast<hipGpuActivityExt*>(rec->_spill_tail);
    if (tail)
      tail->next = node;
    else
      rec->gpu.next = node;  // first spill node → head
    rec->_spill_tail = node;

    rec->gpu.gpu_op_count++;
  }

  // Update g_max_gpu_chunk_id to rec->chunk_id (fetch_max via CAS loop, release store).
  // The release store ensures all GPU writes above are visible to the delivery thread
  // after its acquire load of g_max_gpu_chunk_id.
  if (!g_chunk_clients.empty()) {
    uint32_t id   = rec->chunk_id;
    uint32_t prev = g_max_gpu_chunk_id.load(std::memory_order_relaxed);
    while (id > prev && !g_max_gpu_chunk_id.compare_exchange_weak(
               prev, id, std::memory_order_release, std::memory_order_relaxed)) {}
    g_chunk_cv.notify_one();
  }

  // Forward to previously registered callback (e.g. roctracer / rocprofiler).
  auto* prev = g_prev_callback.load(std::memory_order_acquire);
  if (prev) prev(domain, op_id, data);
  return 0;
}


// ================================================================================================
static const char* CopyKindName(uint32_t kind) {
  switch (static_cast<HipCopyKindExt>(kind)) {
    case HIP_COPY_KIND_H2D_EXT:             return "H2D";
    case HIP_COPY_KIND_H2D_RECT_EXT:        return "H2D_Rect";
    case HIP_COPY_KIND_H2D_IMAGE_EXT:       return "H2D_Image";
    case HIP_COPY_KIND_D2H_EXT:             return "D2H";
    case HIP_COPY_KIND_D2H_RECT_EXT:        return "D2H_Rect";
    case HIP_COPY_KIND_D2H_IMAGE_EXT:       return "D2H_Image";
    case HIP_COPY_KIND_D2D_EXT:             return "D2D";
    case HIP_COPY_KIND_D2D_RECT_EXT:        return "D2D_Rect";
    case HIP_COPY_KIND_D2D_IMAGE_EXT:       return "D2D_Image";
    case HIP_COPY_KIND_BUFFER_TO_IMAGE_EXT: return "BufferToImage";
    case HIP_COPY_KIND_IMAGE_TO_BUFFER_EXT: return "ImageToBuffer";
    case HIP_COPY_KIND_FILL_EXT:            return "Fill";
    default:                                return "Unknown";
  }
}

// ── Shared helpers for both JSON and proto writers ───────────────────────────

static bool IsAllocApi(const char* n) { return n && strncmp(n, "hipMalloc", 9) == 0; }
static bool IsFreeApi (const char* n) { return n && strncmp(n, "hipFree",   7) == 0; }

// Pre-demangle all kernel names in g_kernel_names under a single lock acquisition.
// Both writers call this once at the start so the per-event lookup needs no demangling.
// ================================================================================================
static void PreDemangleKernelNames() {
  std::lock_guard<std::mutex> lk(g_kernel_names_mtx);
  for (auto& kv : g_kernel_names) {
    if (kv.second.size() > 0 && kv.second[0] == '_') {
      std::string demangled;
      if (hip::helpers::demangleName(kv.second, demangled))
        kv.second = std::move(demangled);
    }
  }
}

// ================================================================================================
void WriteJsonTraceImpl(const char* filepath) {
  PreDemangleKernelNames();

  const char* path = (filepath && filepath[0]) ? filepath : "hip_clr_trace.json";
  std::ofstream trace(path, std::fstream::out);
  trace << std::fixed << std::setprecision(3);
  if (!trace.is_open()) return;

  const char* kGpuEvents[] = {"Dispatch", "Copy", "Barrier", "Unknown"};

  trace << "{\n  \"traceEvents\": [";

  size_t total      = g_rec_counter.load(std::memory_order_acquire);
  // Track only the (device_id, gpu_tid) pairs that actually received events.
  // Key = device_id, Value = set of gpu tids (lane*2+sdma) with events.
  std::unordered_map<int, std::unordered_set<uint64_t>> device_gpu_tids;
  // Map (device_id, gpu_tid) -> last seen hipStream_t for lane labeling.
  std::map<std::pair<int,uint64_t>, hipStream_t> gpu_tid_stream;
  std::unordered_set<uint64_t> tid_set;  // unique CPU thread ids seen
  uint64_t flow_id  = 0;  // unique id for each CPU→GPU flow arrow pair
  bool first = true;
  // Compact lane index per unique stream (same scheme as pftrace writer).
  std::unordered_map<uintptr_t, uint64_t> stream_lane_idx;
  uint64_t next_stream_lane = 0;
  auto compact_stream_lane = [&](hipStream_t stream, uint64_t queue_id) -> uint64_t {
    uintptr_t key = stream ? reinterpret_cast<uintptr_t>(stream) : (0x8000ULL | queue_id);
    auto [it, ins] = stream_lane_idx.emplace(key, next_stream_lane);
    if (ins) ++next_stream_lane;
    return it->second;
  };

  // ── Memory lifetime map ───────────────────────────────────────────────────
  // Built in a first pass over all records before emission.
  // pid 2048 "GPU Memory" — one tid per allocation (low 20 bits of ptr).
  static constexpr int kMemPid = 2048;
  struct AllocInfo {
    uint64_t start_ns;   // hipMalloc call start
    uint64_t end_ns;     // hipFree call start; 0 = still live at trace end
    uint64_t size;       // bytes
    uint64_t mem_fid;    // flow id for the "alloc→lifetime" arrow (filled later)
  };
  // Key = device pointer; value = all lifetime entries for that ptr (handles reuse).
  std::unordered_map<uint64_t, std::vector<AllocInfo>> alloc_map;
  // First pass: find base_ns (earliest timestamp) and populate alloc_map.
  // base_ns is subtracted from every ts so values stay small (avoids JS float64
  // precision loss for large absolute ns timestamps in the Perfetto UI).
  uint64_t base_ns = UINT64_MAX;
  for (size_t c = 0; c < g_records.size(); ++c) {
    hipApiRecordExt* chunk = g_records[c];
    size_t base  = c * kChunkSize;
    size_t valid = (total > base) ? std::min(total - base, kChunkSize) : 0;
    for (size_t i = 0; i < valid; ++i) {
      if (chunk[i].start_ns && chunk[i].start_ns < base_ns)
        base_ns = chunk[i].start_ns;
    }
  }
  if (base_ns == UINT64_MAX) base_ns = 0;
  for (size_t c = 0; c < g_records.size(); ++c) {
    hipApiRecordExt* chunk = g_records[c];
    size_t base  = c * kChunkSize;
    size_t valid = (total > base) ? std::min(total - base, kChunkSize) : 0;
    for (size_t i = 0; i < valid; ++i) {
      const hipApiRecordExt& rec = chunk[i];
      if (!rec.api_name) continue;
      uint64_t ptr = reinterpret_cast<uintptr_t>(rec.memory1);
      if (IsAllocApi(rec.api_name) && ptr) {
        alloc_map[ptr].push_back(AllocInfo{rec.start_ns, 0, rec.size, 0});
      } else if (IsFreeApi(rec.api_name) && ptr) {
        auto it = alloc_map.find(ptr);
        if (it != alloc_map.end()) {
          // Close the most recent open (unfreed) lifetime for this ptr.
          for (auto rit = it->second.rbegin(); rit != it->second.rend(); ++rit) {
            if (rit->end_ns == 0) { rit->end_ns = rec.start_ns; break; }
          }
        }
      }
    }
  }
  // Close any allocations still live at trace end (hipFree never called).
  uint64_t trace_end_ns = 0;
  for (auto& kv : alloc_map)
    for (auto& ai : kv.second)
      if (ai.end_ns > trace_end_ns) trace_end_ns = ai.end_ns;
  // Also check all record end times for a proper upper bound.
  for (size_t c = 0; c < g_records.size(); ++c) {
    hipApiRecordExt* chunk = g_records[c];
    size_t base  = c * kChunkSize;
    size_t valid = (total > base) ? std::min(total - base, kChunkSize) : 0;
    for (size_t i = 0; i < valid; ++i)
      if (chunk[i].end_ns > trace_end_ns) trace_end_ns = chunk[i].end_ns;
  }
  for (auto& kv : alloc_map)
    for (auto& ai : kv.second)
      if (ai.end_ns == 0) ai.end_ns = trace_end_ns;

  // Assign a stable tid to each allocation: low 20 bits of ptr shifted right
  // by alignment (device allocs are typically 4KB-aligned, so bits 12+).
  // Collisions are harmless — they just share a lane.
  auto alloc_tid = [](uint64_t ptr) -> uint64_t {
    return (ptr >> 12) & 0xFFFFF;
  };

  for (size_t c = 0; c < g_records.size(); ++c) {
    hipApiRecordExt* chunk = g_records[c];
    size_t base  = c * kChunkSize;
    size_t valid = (total > base) ? std::min(total - base, kChunkSize) : 0;
    if (valid == 0) continue;

    for (size_t i = 0; i < valid; ++i) {
      const hipApiRecordExt& rec = chunk[i];
      tid_set.insert(rec.thread_id);

      double s_time  = (rec.start_ns - base_ns) / 1000.0;
      double dur_us  = (rec.end_ns > rec.start_ns)
                       ? std::max(0.001, (rec.end_ns - rec.start_ns) / 1000.0) : 0.001;

      if (!first) trace << ",";
      first = false;

      trace << "\n{\"name\":\"" << rec.api_name
            << "\",\"ph\":\"X\",\"pid\":1024,\"tid\":" << rec.thread_id
            << ",\"ts\":" << s_time << ",\"dur\":" << dur_us;
      {
        bool first_cpu_arg = true;
        auto cpu_sep = [&]() {
          if (first_cpu_arg) { trace << ",\"args\":{"; first_cpu_arg = false; }
          else trace << ",";
        };
        if (rec.memory1) {
          cpu_sep();
          trace << "\"ptr\":\"0x" << std::hex
                << reinterpret_cast<uintptr_t>(rec.memory1) << std::dec << "\"";
        }
        if (rec.memory2) {
          cpu_sep();
          trace << "\"src\":\"0x" << std::hex
                << reinterpret_cast<uintptr_t>(rec.memory2) << std::dec << "\"";
        }
        if (rec.size) {
          cpu_sep();
          trace << "\"size\":" << rec.size;
        }
        if (rec.stream) {
          cpu_sep();
          trace << "\"stream\":\"0x" << std::hex
                << reinterpret_cast<uintptr_t>(rec.stream) << std::dec << "\"";
        }
        if (!first_cpu_arg) trace << "}";
      }
      trace << "}";

      // Emit one GPU op event: flow start (ph:s) on CPU side, GPU X event,
      // flow finish (ph:t) on GPU side.  Dims and kernel args are read directly
      // from the hipGpuActivityExt struct (populated by HipActivityCallbackExt).
      // Returns {gpu_tid, gpu_ts, has_ts} for chaining node→node flow arrows.
      // emit_host_arrow: draw ph:s/ph:t dep arrow from CPU event to this GPU op.
      //   For graph launches only the first op gets the host arrow; subsequent nodes
      //   are connected via node→node graph arrows instead.
      struct GpuOpInfo { uint64_t tid; double ts; bool has_ts; };
      auto emit_gpu_op = [&](const hipGpuActivityExt& gop, hipStream_t stream,
                             bool emit_host_arrow = true) -> GpuOpInfo {
        uint32_t op_idx  = gop.op < 3 ? gop.op : 3;
        int sdma = (op_idx == OP_ID_COPY) &&
                   hipCopyKindIsSDMAExt(static_cast<HipCopyKindExt>(gop.copy_kind)) ? 1 : 0;
        double gpu_dur = (gop.end_ns > gop.begin_ns)
                         ? std::max(0.001, (gop.end_ns - gop.begin_ns) / 1000.0) : 0.001;
        double gpu_ts  = (gop.begin_ns > base_ns) ? (gop.begin_ns - base_ns) / 1000.0 : 0.0;
        // Use compact lane index per stream so reused queue_ids appear on separate lanes.
        uint64_t gpu_tid = compact_stream_lane(stream, gop.queue_id) * 2 + sdma;

        const char* gpu_name_cstr = (op_idx == OP_ID_COPY) ? CopyKindName(gop.copy_kind)
                                                            : kGpuEvents[op_idx];
        std::string gpu_name = gpu_name_cstr;
        if (op_idx == OP_ID_DISPATCH && gop.kernel_name) {
          // PreDemangleKernelNames() already ran — just look up the (now-demangled) copy.
          // Never use gop.kernel_name at write time — it may be dangling.
          std::lock_guard<std::mutex> lk(g_kernel_names_mtx);
          auto it = g_kernel_names.find(gop.kernel_name);
          if (it != g_kernel_names.end()) gpu_name = it->second;
        }

        // Track stream→lane mapping for metadata labels.
        if (stream) gpu_tid_stream[{static_cast<int>(gop.device_id), gpu_tid}] = stream;

        // Only draw flow arrow when GPU timestamps are valid (begin_ns == 0
        // means ReportActivity never populated the record — no arrow to draw).
        const bool has_ts = (gop.begin_ns > 0);
        uint64_t fid = (has_ts && emit_host_arrow) ? flow_id++ : 0;
        if (has_ts && emit_host_arrow)
          trace << ",\n{\"ph\":\"s\",\"id\":" << fid
                << ",\"pid\":1024,\"tid\":" << rec.thread_id
                << ",\"ts\":" << s_time << ",\"name\":\"dep\"}";
        // GPU X event.
        trace << ",\n{\"name\":\"" << gpu_name
              << "\",\"ph\":\"X\",\"pid\":" << gop.device_id
              << ",\"tid\":" << gpu_tid
              << ",\"ts\":" << gpu_ts << ",\"dur\":" << gpu_dur
              << ",\"args\":{";
        bool first_arg = true;
        auto sep = [&]() { if (!first_arg) trace << ","; first_arg = false; };
        sep();
        trace << "\"queue_id\":" << gop.queue_id;
        if (op_idx == OP_ID_DISPATCH && gop.grid_x) {
          sep();
          trace << "\"grid\":\"" << gop.grid_x << "x" << gop.grid_y << "x" << gop.grid_z << "\""
                << ",\"block\":\"" << gop.block_x << "x" << gop.block_y << "x"
                << gop.block_z << "\"";
        }
        if (op_idx == OP_ID_COPY) {
          sep();
          trace << "\"copy_kind\":\"" << CopyKindName(gop.copy_kind)
                << "\",\"bytes\":" << gop.bytes;
          if (gop.dst) {
            sep();
            trace << "\"dst\":\"0x" << std::hex
                  << reinterpret_cast<uintptr_t>(gop.dst) << std::dec << "\"";
          }
          if (gop.src) {
            sep();
            trace << "\"src\":\"0x" << std::hex
                  << reinterpret_cast<uintptr_t>(gop.src) << std::dec << "\"";
          }
        }
        if (stream) {
          sep();
          trace << "\"stream\":\"0x" << std::hex
                << reinterpret_cast<uintptr_t>(stream) << std::dec << "\"";
        }
        // Kernel args on GPU dispatch event (positional, same format as CPU record).
        if (op_idx == OP_ID_DISPATCH && gop.kernel_args && gop.kernel_args_size > 0) {
          const uint8_t* p   = gop.kernel_args;
          const uint8_t* end = p + gop.kernel_args_size;
          int kidx = 0;
          while (p + sizeof(uint32_t) <= end) {
            uint32_t sz; std::memcpy(&sz, p, sizeof(uint32_t)); p += sizeof(uint32_t);
            if (p + sz > end) break;
            sep();
            trace << "\"" << kidx++ << "\":";
            if (sz == 8) {
              uint64_t v; std::memcpy(&v, p, 8);
              trace << "\"0x" << std::hex << v << std::dec << "\"";
            } else if (sz == 4) {
              uint32_t v; std::memcpy(&v, p, 4);
              trace << v;
            } else {
              trace << "\"0x" << std::hex;
              for (uint32_t b = 0; b < sz; ++b) trace << static_cast<unsigned>(p[b]);
              trace << std::dec << "\"";
            }
            p += sz;
          }
        }
        trace << "}}";
        if (has_ts && emit_host_arrow)
          trace << ",\n{\"ph\":\"t\",\"id\":" << fid
                << ",\"pid\":" << gop.device_id
                << ",\"tid\":" << gpu_tid
                << ",\"ts\":" << gpu_ts << ",\"name\":\"dep\"}";

        // ── GPU→memory flow arrows ────────────────────────────────────────
        // Emit arrows from this GPU event to the lifetime slice of each
        // allocation it references.  Dispatch: scan kargs for 8-byte ptrs.
        // Copy: use gop.dst / gop.src (populated from CPU record for direct copies,
        // or from graph node info for graph copy nodes).
        if (has_ts) {
          uint64_t seen[18]; int nseen = 0;
          auto try_mem_arrow = [&](uint64_t ptr) {
            if (!ptr) return;
            for (int si = 0; si < nseen; ++si) if (seen[si] == ptr) return;
            auto it = alloc_map.find(ptr);
            if (it == alloc_map.end()) return;
            // Find the lifetime entry that was live when this GPU op ran.
            AllocInfo* ai_match = nullptr;
            for (auto& ai : it->second)
              if (ai.start_ns <= gop.begin_ns && (ai.end_ns == 0 || gop.begin_ns <= ai.end_ns))
                { ai_match = &ai; break; }
            if (!ai_match) return;
            seen[nseen++] = ptr;
            uint64_t afid    = flow_id++;
            uint64_t mem_tid = alloc_tid(ptr);
            trace << ",\n{\"ph\":\"s\",\"id\":" << afid
                  << ",\"pid\":" << gop.device_id << ",\"tid\":" << gpu_tid
                  << ",\"ts\":" << gpu_ts << ",\"name\":\"mem\",\"cat\":\"mem\"}";
            trace << ",\n{\"ph\":\"f\",\"bp\":\"e\",\"id\":" << afid
                  << ",\"pid\":" << kMemPid << ",\"tid\":" << mem_tid
                  << ",\"ts\":" << gpu_ts << ",\"name\":\"mem\",\"cat\":\"mem\"}";
          };
          if (op_idx == OP_ID_DISPATCH && gop.kernel_args && gop.kernel_args_size > 0) {
            const uint8_t* p   = gop.kernel_args;
            const uint8_t* end = p + gop.kernel_args_size;
            while (p + sizeof(uint32_t) <= end) {
              uint32_t sz; std::memcpy(&sz, p, sizeof(uint32_t)); p += sizeof(uint32_t);
              if (p + sz > end) break;
              if (sz == 8) { uint64_t v; std::memcpy(&v, p, 8); try_mem_arrow(v); }
              p += sz;
            }
          } else if (op_idx == OP_ID_COPY) {
            try_mem_arrow(reinterpret_cast<uintptr_t>(gop.dst));
            try_mem_arrow(reinterpret_cast<uintptr_t>(gop.src));
          }
        }

        device_gpu_tids[static_cast<int>(gop.device_id)].insert(gpu_tid);
        return GpuOpInfo{gpu_tid, gpu_ts, has_ts};
      };

      // Op-1 lives directly in rec.gpu; ops 2..N are in the spill linked list.
      // All dims and kernel args are now in each hipGpuActivityExt node.
      const bool graph_launch = rec.api_name && strncmp(rec.api_name, "hipGraphLaunch", 14) == 0;
      if (rec.gpu.gpu_op_count > 0) {
        GpuOpInfo prev = emit_gpu_op(rec.gpu, rec.stream, /*emit_host_arrow=*/true);

        for (const hipGpuActivityExt* node = rec.gpu.next; node; node = node->next) {
          GpuOpInfo cur = emit_gpu_op(*node, rec.stream, /*emit_host_arrow=*/false);
          // Node→node flow arrow within this graph launch.
          if (graph_launch && prev.has_ts && cur.has_ts) {
            uint64_t gfid = flow_id++;
            trace << ",\n{\"ph\":\"s\",\"id\":" << gfid
                  << ",\"pid\":" << rec.gpu.device_id
                  << ",\"tid\":" << prev.tid
                  << ",\"ts\":" << prev.ts << ",\"name\":\"graph\",\"cat\":\"graph\"}";
            trace << ",\n{\"ph\":\"f\",\"bp\":\"e\",\"id\":" << gfid
                  << ",\"pid\":" << rec.gpu.device_id
                  << ",\"tid\":" << cur.tid
                  << ",\"ts\":" << cur.ts << ",\"name\":\"graph\",\"cat\":\"graph\"}";
          }
          prev = cur;
        }
      }
    }
  }

  // ── Memory lifetime slices (pid 2048 "GPU Memory") ───────────────────────
  // One ph:"X" slice per allocation spanning malloc→free, on a per-ptr tid lane.
  for (auto& kv : alloc_map) {
    uint64_t ptr     = kv.first;
    uint64_t mem_tid = alloc_tid(ptr);
    char ptrbuf[32];
    snprintf(ptrbuf, sizeof(ptrbuf), "0x%llx", static_cast<unsigned long long>(ptr));
    for (const AllocInfo& ai : kv.second) {
    double ts_us   = (ai.start_ns > base_ns) ? (ai.start_ns - base_ns) / 1000.0 : 0.0;
    double end_us  = (ai.end_ns   > base_ns) ? (ai.end_ns   - base_ns) / 1000.0 : 0.0;
    double dur_us  = (end_us > ts_us) ? (end_us - ts_us) : 0.001;
    char sizebuf[32];
    if (ai.size >= 1024 * 1024)
      snprintf(sizebuf, sizeof(sizebuf), "%.1f MB", ai.size / (1024.0 * 1024.0));
    else if (ai.size >= 1024)
      snprintf(sizebuf, sizeof(sizebuf), "%.1f KB", ai.size / 1024.0);
    else
      snprintf(sizebuf, sizeof(sizebuf), "%llu B",
               static_cast<unsigned long long>(ai.size));
    trace << ",\n{\"name\":\"" << ptrbuf << " (" << sizebuf << ")"
          << "\",\"ph\":\"X\""
          << ",\"pid\":" << kMemPid << ",\"tid\":" << mem_tid
          << ",\"ts\":" << ts_us << ",\"dur\":" << dur_us
          << ",\"args\":{\"ptr\":\"" << ptrbuf << "\",\"size\":" << ai.size << "}}";
    } // end inner for (ai)
    // Thread name label once per ptr (use size of first entry for the label).
    if (!kv.second.empty()) {
      const AllocInfo& ai0 = kv.second[0];
      char sizebuf[32];
      if (ai0.size >= 1024*1024) snprintf(sizebuf, sizeof(sizebuf), "%.1f MB", ai0.size/(1024.0*1024.0));
      else if (ai0.size >= 1024)  snprintf(sizebuf, sizeof(sizebuf), "%.1f KB", ai0.size/1024.0);
      else snprintf(sizebuf, sizeof(sizebuf), "%llu B", static_cast<unsigned long long>(ai0.size));
      trace << ",\n{\"name\":\"thread_name\",\"ph\":\"M\""
            << ",\"pid\":" << kMemPid << ",\"tid\":" << mem_tid
            << ",\"args\":{\"name\":\"" << ptrbuf << " (" << sizebuf << ")\"}}";
    }
  }
  if (!alloc_map.empty()) {
    uint64_t total_bytes = 0;
    for (auto& kv : alloc_map) for (auto& ai : kv.second) total_bytes += ai.size;
    char total_sizebuf[32];
    if (total_bytes >= 1024 * 1024)
      snprintf(total_sizebuf, sizeof(total_sizebuf), "%.1f MB", total_bytes / (1024.0 * 1024.0));
    else if (total_bytes >= 1024)
      snprintf(total_sizebuf, sizeof(total_sizebuf), "%.1f KB", total_bytes / 1024.0);
    else
      snprintf(total_sizebuf, sizeof(total_sizebuf), "%llu B",
               static_cast<unsigned long long>(total_bytes));
    trace << ",\n{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":" << kMemPid
          << ",\"args\":{\"name\":\"GPU Memory (" << total_sizebuf << " total)\"}}";
    trace << ",\n{\"name\":\"process_sort_index\",\"ph\":\"M\",\"pid\":" << kMemPid
          << ",\"args\":{\"sort_index\":999}}";
  }

  // Emit CPU thread name metadata using kernel thread IDs
  for (uint64_t tid : tid_set) {
    trace << ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1024,\"tid\":" << tid
          << ",\"args\":{\"name\":\"HIP Thread " << tid << "\"}}";
  }

  // Returns {targetId, hip_device_index} for a given HSA device_id.
  auto get_gfxip = [&](int dev_id) -> std::pair<std::string, int> {
    for (int pass = 0; pass < 2; ++pass) {
      for (size_t gi = 0; gi < hip::g_devices.size(); ++gi) {
        auto* hdev = hip::g_devices[gi];
        if (!hdev || hdev->devices().empty()) continue;
        bool match = (pass == 0) ? (hdev->deviceId() == dev_id)
                                 : (static_cast<int>(gi) == dev_id % static_cast<int>(hip::g_devices.size()));
        if (!match) continue;
        const char* tgt = hdev->devices()[0]->isa().targetId();
        if (tgt && tgt[0]) return {std::string(tgt), static_cast<int>(gi)};
      }
    }
    for (size_t gi = 0; gi < hip::g_devices.size(); ++gi) {
      auto* hdev = hip::g_devices[gi];
      if (!hdev || hdev->devices().empty()) continue;
      const char* tgt = hdev->devices()[0]->isa().targetId();
      if (tgt && tgt[0]) return {std::string(tgt), static_cast<int>(gi)};
    }
    return {"", -1};
  };

  // CPU process metadata (sort_index 0 so it appears first)
  std::string proc_name, proc_path;
  amd::Os::getAppPathAndFileName(proc_name, proc_path);
  std::string cpu_label = proc_name.empty() ? "CPU HIP" : ("CPU HIP [" + proc_name + "]");
  trace << ",\n{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1024,"
           "\"args\":{\"name\":\"" << cpu_label << "\"}}";
  trace << ",\n{\"name\":\"process_sort_index\",\"ph\":\"M\",\"pid\":1024,"
           "\"args\":{\"sort_index\":0}}";

  // Per-device GPU process metadata — only name tids that actually received events.
  int gpu_sort = 1;
  for (auto& kv : device_gpu_tids) {
    int dev_id = kv.first;
    const auto& active_tids = kv.second;

    auto [gfxip, hip_idx] = get_gfxip(dev_id);
    std::string label = gfxip.empty() ? ("GPU " + std::to_string(dev_id))
                                      : (gfxip + " [Device " + std::to_string(hip_idx) + "]");

    trace << ",\n{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":" << dev_id
          << ",\"args\":{\"name\":\"" << label << "\"}}";
    trace << ",\n{\"name\":\"process_sort_index\",\"ph\":\"M\",\"pid\":" << dev_id
          << ",\"args\":{\"sort_index\":" << gpu_sort++ << "}}";

    // Emit a thread_name only for tids that actually have events.
    // Even tid = Compute (queue_id*2), odd tid = SDMA (queue_id*2+1).
    // Use the hipStream_t address if known; fall back to queue index label.
    for (uint64_t gpu_tid : active_tids) {
      bool is_sdma = (gpu_tid & 1) != 0;
      uint64_t q   = gpu_tid / 2;
      std::string lane_name;
      auto sit = gpu_tid_stream.find({dev_id, gpu_tid});
      if (sit != gpu_tid_stream.end() && sit->second) {
        char buf[32];
        snprintf(buf, sizeof(buf), "0x%llx",
                 static_cast<unsigned long long>(
                   reinterpret_cast<uintptr_t>(sit->second)));
        // SDMA lane shares the stream prefix so it sorts next to its compute lane.
        lane_name = std::string("Stream ") + buf + (is_sdma ? " [DMA]" : "");
      } else if (!is_sdma && q == 0) {
        lane_name = "Default Stream";
      } else if (is_sdma && q == 0) {
        lane_name = "Default Stream [DMA]";
      } else {
        lane_name = std::string(is_sdma ? "SDMA " : "Compute ") + std::to_string(q);
      }
      trace << ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":" << dev_id
            << ",\"tid\":" << gpu_tid
            << ",\"args\":{\"name\":\"" << lane_name << "\"}}";
    }
  }

  trace << "\n  ],\n  \"displayTimeUnit\": \"us\"\n}\n";
  trace.close();
}

// ================================================================================================
static void DrainAllDevices() {
  // Drain all GPU work on every device using the internal CLR path.
  // Mirrors hipDeviceSynchronize() without going through HIP_INIT_API or our
  // dispatch table wrappers, so no spurious profiling records are created and
  // there is no re-entrancy risk.
  for (auto* dev : hip::g_devices) {
    constexpr bool kWaitForCpu = true;
    dev->SyncAllStreams(kWaitForCpu);
  }
}

// ============================================================
// Minimal protobuf encoder (no external dependency)
// ============================================================
struct ProtoMsg {
  std::string buf;

  void varint(uint64_t v) {
    while (v >= 0x80) { buf += static_cast<char>((v & 0x7f) | 0x80); v >>= 7; }
    buf += static_cast<char>(v);
  }
  void tag(uint32_t field, uint32_t wtype) { varint((uint64_t(field) << 3) | wtype); }
  void u64(uint32_t field, uint64_t v)  { tag(field, 0); varint(v); }
  void u32(uint32_t field, uint32_t v)  { tag(field, 0); varint(v); }
  void str(uint32_t field, const std::string& s) {
    tag(field, 2); varint(s.size()); buf += s;
  }
  void str(uint32_t field, const char* s) {
    if (!s) return;
    tag(field, 2); size_t n = strlen(s); varint(n); buf.append(s, n);
  }
  void msg(uint32_t field, const ProtoMsg& m) {
    tag(field, 2); varint(m.buf.size()); buf += m.buf;
  }
  // packed repeated uint64 (wire type 2, content = concatenated varints)
  void packed_u64(uint32_t field, const std::vector<uint64_t>& vals) {
    if (vals.empty()) return;
    ProtoMsg inner;
    for (uint64_t v : vals) inner.varint(v);
    tag(field, 2); varint(inner.buf.size()); buf += inner.buf;
  }
};

// ── Perfetto proto field numbers (verified against perfetto/trace protos) ────
// TracePacket (trace_packet.proto)
static constexpr uint32_t kPkt_timestamp      = 8;   // uint64
static constexpr uint32_t kPkt_track_event    = 11;  // TrackEvent
static constexpr uint32_t kPkt_seq_id         = 10;  // trusted_packet_sequence_id
static constexpr uint32_t kPkt_seq_flags      = 13;  // sequence_flags
static constexpr uint32_t kPkt_track_desc     = 60;  // TrackDescriptor
// kPkt_clock_snapshot (6) and kPkt_ts_clock_id (58) intentionally omitted —
// we rely on Perfetto's default trace clock (no conversion needed).

// sequence_flags values (TracePacket.sequence_flags)
static constexpr uint32_t kSeqFlag_Cleared    = 1;  // SEQ_INCREMENTAL_STATE_CLEARED
static constexpr uint32_t kSeqFlag_NeedsState = 2;  // SEQ_NEEDS_INCREMENTAL_STATE

// TrackDescriptor (track_descriptor.proto)
static constexpr uint32_t kDesc_uuid    = 1;
static constexpr uint32_t kDesc_name    = 2;
static constexpr uint32_t kDesc_process = 3;
static constexpr uint32_t kDesc_thread  = 4;
static constexpr uint32_t kDesc_parent  = 5;

// ProcessDescriptor (process_descriptor.proto)
static constexpr uint32_t kProc_pid  = 1;
static constexpr uint32_t kProc_name = 6;

// ThreadDescriptor (thread_descriptor.proto)
static constexpr uint32_t kThrd_pid  = 1;
static constexpr uint32_t kThrd_tid  = 2;
static constexpr uint32_t kThrd_name = 5;

// TrackEvent (track_event.proto)
static constexpr uint32_t kEvt_type         = 9;
static constexpr uint32_t kEvt_track_uuid   = 11;
static constexpr uint32_t kEvt_name_iid     = 10;  // uint64 interned name reference
static constexpr uint32_t kEvt_cat_iids     = 3;   // repeated uint64 interned category refs
static constexpr uint32_t kEvt_dbg_ann      = 4;   // repeated DebugAnnotation
static constexpr uint32_t kEvt_legacy_event = 6;   // TrackEvent.legacy_event (LegacyEvent msg)
static constexpr uint32_t kEvt_TYPE_BEGIN   = 1;
static constexpr uint32_t kEvt_TYPE_END     = 2;

// LegacyEvent (legacy_event.proto) — used for Chrome-format flow arrows (ph='s'/'f')
static constexpr uint32_t kLeg_phase    = 2;   // int32 — Chrome phase char ('s'=115, 'f'=102)
static constexpr uint32_t kLeg_id       = 6;   // uint64 — unscoped flow binding id
static constexpr uint32_t kLeg_flow_dir = 13;  // uint32 — FlowDirection: OUT=1, IN=2

// DebugAnnotation (debug_annotation.proto)
static constexpr uint32_t kAnn_name_iid = 1;   // uint64 interned name reference
static constexpr uint32_t kAnn_uint     = 3;   // uint64 uint_value
static constexpr uint32_t kAnn_str_val  = 6;   // string string_value

// InternedData (interned_data.proto) in TracePacket — string intern table
// Field layout from interned_data.proto:
//   1 = event_categories (repeated EventCategory)
//   2 = event_names      (repeated EventName)
//   3 = debug_annotation_names (repeated DebugAnnotationName)
static constexpr uint32_t kPkt_interned_data = 12;  // InternedData message
static constexpr uint32_t kIData_cat_names   = 1;   // repeated EventCategory
static constexpr uint32_t kIData_evt_names   = 2;   // repeated EventName
static constexpr uint32_t kIData_ann_names   = 3;   // repeated DebugAnnotationName
// EventName / DebugAnnotationName / EventCategory all share the same layout:
static constexpr uint32_t kIName_iid  = 1;  // uint64 interned ID
static constexpr uint32_t kIName_name = 2;  // string

// ClockSnapshot — not used; we emit no clock_snapshot and no timestamp_clock_id.
// Perfetto treats timestamps without a clock ID as nanoseconds on the default trace
// clock (displayed as-is).  Emitting CLOCK_MONOTONIC (id=3) with a snapshot taken
// at atexit time causes Perfetto to reject events whose timestamps predate the snapshot.

// UUID helpers: keep out of collision with each other
static uint64_t CpuProcessUuid()                        { return 0x1000ULL; }
static uint64_t CpuThreadUuid(uint32_t tid)             { return 0x1000ULL << 20 | tid; }
static uint64_t GpuProcessUuid(int dev)                 { return 0x2000ULL | uint64_t(dev); }
static uint64_t GpuThreadUuid(int dev, uint64_t q_tid) {
  return (0x2000ULL | uint64_t(dev)) << 20 | q_tid;
}
static uint64_t MemProcessUuid()                        { return 0x3000ULL; }
static uint64_t MemThreadUuid(uint64_t tid) {
  return 0x3000ULL << 20 | (tid & 0xFFFFF);
}

// ================================================================================================
static void AppendPacket(std::string& out, const ProtoMsg& pkt) {
  // Each TracePacket is field 1 (length-delimited) of the Trace message.
  ProtoMsg wrapper;
  wrapper.msg(1, pkt);
  out += wrapper.buf;
}

// Sorted packet accumulator — collects (timestamp_ns, serialized_packet) pairs.
// After all events are collected, sort by timestamp and flush to output.
// This ensures the packet stream is monotonically non-decreasing in timestamp,
// which Perfetto requires for correct rendering within a trusted_packet_sequence.
using SortedPktList = std::vector<std::pair<uint64_t, std::string>>;

static void AppendSorted(SortedPktList& pkts, uint64_t ts_ns, const ProtoMsg& pkt) {
  ProtoMsg wrapper;
  wrapper.msg(1, pkt);
  pkts.emplace_back(ts_ns, std::move(wrapper.buf));
}

static void FlushSorted(std::string& out, SortedPktList& pkts) {
  std::stable_sort(pkts.begin(), pkts.end(),
                   [](const auto& a, const auto& b) { return a.first < b.first; });
  for (auto& kv : pkts) out += kv.second;
  pkts.clear();
}

// All track events share a single sequence (cleared by the ClockSnapshot packet).
static constexpr uint32_t kGlobalSeq = 1;

// ================================================================================================
static void EmitTrackDesc(std::string& out, uint64_t uuid, uint64_t parent,
                           const std::string& name,
                           int pid = -1, int tid = -1, bool is_process = false) {
  ProtoMsg desc;
  desc.u64(kDesc_uuid, uuid);
  if (parent) desc.u64(kDesc_parent, parent);
  desc.str(kDesc_name, name);
  if (is_process && pid >= 0) {
    ProtoMsg proc;
    proc.u32(kProc_pid, uint32_t(pid));
    proc.str(kProc_name, name);
    desc.msg(kDesc_process, proc);
  } else if (!is_process && pid >= 0) {
    ProtoMsg thrd;
    thrd.u32(kThrd_pid, uint32_t(pid));
    if (tid >= 0) thrd.u32(kThrd_tid, uint32_t(tid));
    thrd.str(kThrd_name, name);
    desc.msg(kDesc_thread, thrd);
  }

  ProtoMsg pkt;
  pkt.msg(kPkt_track_desc, desc);
  pkt.u32(kPkt_seq_id, kGlobalSeq);
  AppendPacket(out, pkt);
}

// ================================================================================================
static ProtoMsg MakeFlowEventPkt(uint64_t uuid, uint64_t ts_ns, uint64_t fid, bool is_start) {
  ProtoMsg leg;
  leg.u32(kLeg_phase,    is_start ? 115u : 102u);  // 's' or 'f'
  leg.u64(kLeg_id,       fid);
  leg.u32(kLeg_flow_dir, is_start ? 1u   : 2u);    // FLOW_OUT or FLOW_IN
  ProtoMsg evt;
  evt.u64(kEvt_track_uuid, uuid);
  evt.msg(kEvt_legacy_event, leg);
  ProtoMsg pkt;
  pkt.u64(kPkt_timestamp, ts_ns);
  pkt.msg(kPkt_track_event, evt);
  pkt.u32(kPkt_seq_id, kGlobalSeq);
  return pkt;
}

static void EmitFlowEvent(std::string& out, uint64_t uuid, uint64_t ts_ns,
                           uint64_t fid, bool is_start) {
  // Emit a Chrome-format flow arrow event (ph='s' or ph='f') as a LegacyEvent instant.
  // Mirrors the JSON writer's {"ph":"s"} / {"ph":"f"} events — known to work in Perfetto UI.
  AppendPacket(out, MakeFlowEventPkt(uuid, ts_ns, fid, is_start));
}

static void EmitFlowEventSorted(SortedPktList& pkts, uint64_t uuid, uint64_t ts_ns,
                                 uint64_t fid, bool is_start) {
  AppendSorted(pkts, ts_ns, MakeFlowEventPkt(uuid, ts_ns, fid, is_start));
}

// ================================================================================================
// name_iid / cat_iid / ann_key_iid: interned string IDs (0 = not interned, use direct string)
static void EmitSlice(std::string& out, uint64_t uuid, uint32_t /*seq_id*/,
                       uint64_t ts_ns, uint64_t dur_ns,
                       const std::string& name, uint64_t name_iid,
                       uint64_t cat_iid,
                       const std::vector<std::pair<uint64_t,std::string>>& anns,
                       const std::vector<uint64_t>& out_flows = {},
                       const std::vector<uint64_t>& in_flows  = {}) {
  // BEGIN
  {
    ProtoMsg evt;
    evt.u32(kEvt_type, kEvt_TYPE_BEGIN);
    evt.u64(kEvt_track_uuid, uuid);
    if (name_iid) evt.u64(kEvt_name_iid, name_iid);
    else          evt.str(23 /*name direct*/, name);
    if (cat_iid)  evt.u64(kEvt_cat_iids, cat_iid);
    for (const auto& a : anns) {
      ProtoMsg ann;
      ann.u64(kAnn_name_iid, a.first);   // always interned for annotation keys
      ann.str(kAnn_str_val,  a.second);
      evt.msg(kEvt_dbg_ann, ann);
    }
    ProtoMsg pkt;
    pkt.u64(kPkt_timestamp, ts_ns);
    pkt.msg(kPkt_track_event, evt);
    pkt.u32(kPkt_seq_id, kGlobalSeq);
    pkt.u32(kPkt_seq_flags, kSeqFlag_NeedsState);  // packet uses interned IIDs
    AppendPacket(out, pkt);
  }
  // Flow events — separate packets, matching JSON writer's ph='s'/'f' approach
  for (uint64_t fid : out_flows) EmitFlowEvent(out, uuid, ts_ns, fid, true);
  for (uint64_t fid : in_flows)  EmitFlowEvent(out, uuid, ts_ns, fid, false);
  // END
  {
    ProtoMsg evt;
    evt.u32(kEvt_type, kEvt_TYPE_END);
    evt.u64(kEvt_track_uuid, uuid);
    ProtoMsg pkt;
    pkt.u64(kPkt_timestamp, ts_ns + dur_ns);
    pkt.msg(kPkt_track_event, evt);
    pkt.u32(kPkt_seq_id, kGlobalSeq);
    AppendPacket(out, pkt);
  }
}

// Sorted variant: collects BEGIN, flow events, and END into pkts keyed by their timestamps.
// Caller must call FlushSorted(out, pkts) after all events are accumulated.
static void EmitSliceSorted(SortedPktList& pkts, uint64_t uuid,
                             uint64_t ts_ns, uint64_t dur_ns,
                             const std::string& name, uint64_t name_iid,
                             uint64_t cat_iid,
                             const std::vector<std::pair<uint64_t,std::string>>& anns,
                             const std::vector<uint64_t>& out_flows = {},
                             const std::vector<uint64_t>& in_flows  = {}) {
  // BEGIN
  {
    ProtoMsg evt;
    evt.u32(kEvt_type, kEvt_TYPE_BEGIN);
    evt.u64(kEvt_track_uuid, uuid);
    if (name_iid) evt.u64(kEvt_name_iid, name_iid);
    else          evt.str(23 /*name direct*/, name);
    if (cat_iid)  evt.u64(kEvt_cat_iids, cat_iid);
    for (const auto& a : anns) {
      ProtoMsg ann;
      ann.u64(kAnn_name_iid, a.first);
      ann.str(kAnn_str_val,  a.second);
      evt.msg(kEvt_dbg_ann, ann);
    }
    ProtoMsg pkt;
    pkt.u64(kPkt_timestamp, ts_ns);
    pkt.msg(kPkt_track_event, evt);
    pkt.u32(kPkt_seq_id, kGlobalSeq);
    pkt.u32(kPkt_seq_flags, kSeqFlag_NeedsState);
    AppendSorted(pkts, ts_ns, pkt);
  }
  for (uint64_t fid : out_flows) EmitFlowEventSorted(pkts, uuid, ts_ns, fid, true);
  for (uint64_t fid : in_flows)  EmitFlowEventSorted(pkts, uuid, ts_ns, fid, false);
  // END
  {
    ProtoMsg evt;
    evt.u32(kEvt_type, kEvt_TYPE_END);
    evt.u64(kEvt_track_uuid, uuid);
    ProtoMsg pkt;
    pkt.u64(kPkt_timestamp, ts_ns + dur_ns);
    pkt.msg(kPkt_track_event, evt);
    pkt.u32(kPkt_seq_id, kGlobalSeq);
    AppendSorted(pkts, ts_ns + dur_ns, pkt);
  }
}

// ============================================================
// Streaming pftrace state — persistent across chunk deliveries.
// All guarded by g_pf_mtx (the chunk callback is called from the delivery
// thread; ProfilerAtExit accesses from the main thread after thread join).
// ============================================================

// File handle opened at streaming init; closed in ProfilerAtExit.
// Forward-declared above ChunkDeliveryThread — definition here.
// Buffer flushed to file at each chunk boundary.
static std::string g_pf_buf;

// Intern tables (event names, annotation keys, categories).
// IIDs start at 1; 0 means "not interned".
static std::unordered_map<std::string, uint64_t> g_pf_evt_iid;
static std::unordered_map<std::string, uint64_t> g_pf_ann_iid;
static std::unordered_map<std::string, uint64_t> g_pf_cat_iid;
static uint64_t                                   g_pf_next_iid{1};

// Track UUIDs already emitted as TrackDescriptors.
static std::unordered_set<uint64_t> g_pf_emitted_tracks;

// Compact stream-lane index (same scheme as batch writer).
static std::unordered_map<uintptr_t, uint64_t> g_pf_stream_lane;
static uint64_t                                 g_pf_next_lane{0};

// Compact CPU thread index.
static std::unordered_map<uint64_t, uint32_t> g_pf_tid_map;
static uint32_t                                g_pf_next_tid{0};

// Monotonically increasing flow ID (separate counter from batch writer).
static uint64_t g_pf_next_fid{1};

// Alloc lifetime map — built incrementally; memory slices emitted at final flush.
struct PfAllocInfo { uint64_t start_ns, end_ns, size; };
static std::unordered_map<uint64_t, std::vector<PfAllocInfo>> g_pf_alloc_map;
static uint64_t g_pf_trace_end_ns{0};

// Mutex protecting all g_pf_* state.
static std::mutex g_pf_mtx;
static bool       g_pf_first_interned{true};  // kSeqFlag_Cleared only on first InternedData packet

// ── Streaming intern helpers ──────────────────────────────────────────────────
// Returns the IID for a string, adding it to the new-IID list if first seen.
// Caller passes new_evts/new_anns/new_cats vectors that accumulate entries for
// the current chunk's InternedData packet.
static uint64_t PfInternEvt(const std::string& s,
                              std::vector<std::pair<uint64_t,std::string>>& new_evts) {
  auto [it, ins] = g_pf_evt_iid.emplace(s, 0);
  if (ins) { it->second = g_pf_next_iid++; new_evts.push_back({it->second, s}); }
  return it->second;
}
static uint64_t PfInternAnn(const std::string& s,
                              std::vector<std::pair<uint64_t,std::string>>& new_anns) {
  auto [it, ins] = g_pf_ann_iid.emplace(s, 0);
  if (ins) { it->second = g_pf_next_iid++; new_anns.push_back({it->second, s}); }
  return it->second;
}
static uint64_t PfInternCat(const std::string& s,
                              std::vector<std::pair<uint64_t,std::string>>& new_cats) {
  auto [it, ins] = g_pf_cat_iid.emplace(s, 0);
  if (ins) { it->second = g_pf_next_iid++; new_cats.push_back({it->second, s}); }
  return it->second;
}

// ── Streaming track descriptor helpers ───────────────────────────────────────
// Emits a TrackDescriptor packet the first time a UUID is seen.
static void PfEnsureTrack(uint64_t uuid, uint64_t parent, const std::string& name,
                           int pid = -1, int tid = -1, bool is_process = false) {
  if (!g_pf_emitted_tracks.insert(uuid).second) return;  // already emitted
  EmitTrackDesc(g_pf_buf, uuid, parent, name, pid, tid, is_process);
}

// ── Lane helpers (same logic as batch writer) ─────────────────────────────────
static uint64_t PfCompactLane(hipStream_t stream, uint64_t queue_id) {
  uintptr_t key = stream ? reinterpret_cast<uintptr_t>(stream) : (0x8000ULL | queue_id);
  auto [it, ins] = g_pf_stream_lane.emplace(key, g_pf_next_lane);
  if (ins) ++g_pf_next_lane;
  return it->second;
}
static uint32_t PfCompactTid(uint64_t raw) {
  auto [it, ins] = g_pf_tid_map.emplace(raw, g_pf_next_tid);
  if (ins) ++g_pf_next_tid;
  return it->second;
}

// ── get_gfxip — minimal version (same logic as batch writer) ─────────────────
static std::pair<std::string,int> PfGetGfxip(int dev_id) {
  for (int pass = 0; pass < 2; ++pass) {
    for (size_t gi = 0; gi < hip::g_devices.size(); ++gi) {
      auto* hdev = hip::g_devices[gi];
      if (!hdev || hdev->devices().empty()) continue;
      bool match = (pass == 0) ? (hdev->deviceId() == dev_id)
                               : (static_cast<int>(gi) == dev_id % static_cast<int>(hip::g_devices.size()));
      if (!match) continue;
      const char* tgt = hdev->devices()[0]->isa().targetId();
      if (tgt && tgt[0]) return {std::string(tgt), static_cast<int>(gi)};
    }
  }
  return {"", -1};
}

// ── CPU process + thread tracks ───────────────────────────────────────────────
static void PfEnsureCpuProcess() {
  if (g_pf_emitted_tracks.count(CpuProcessUuid())) return;
  std::string proc_name, proc_path;
  amd::Os::getAppPathAndFileName(proc_name, proc_path);
  std::string label = proc_name.empty() ? "CPU HIP" : ("CPU HIP [" + proc_name + "]");
  PfEnsureTrack(CpuProcessUuid(), 0, label, 1024, -1, true);
}
static void PfEnsureCpuThread(uint32_t ctid) {
  PfEnsureCpuProcess();
  uint64_t uuid = CpuThreadUuid(ctid);
  PfEnsureTrack(uuid, CpuProcessUuid(), "HIP Thread " + std::to_string(ctid),
                1024, int(ctid), false);
}
static void PfEnsureGpuThread(int dev_id, uint64_t gtid, hipStream_t stream) {
  uint64_t proc_uuid = GpuProcessUuid(dev_id);
  if (!g_pf_emitted_tracks.count(proc_uuid)) {
    auto [gfxip, hip_idx] = PfGetGfxip(dev_id);
    std::string label = gfxip.empty() ? ("GPU " + std::to_string(dev_id))
                                      : (gfxip + " [Device " + std::to_string(hip_idx) + "]");
    PfEnsureTrack(proc_uuid, 0, label, dev_id, -1, true);
  }
  uint64_t t_uuid = GpuThreadUuid(dev_id, gtid);
  if (!g_pf_emitted_tracks.count(t_uuid)) {
    bool is_sdma = (gtid & 1) != 0;
    std::string lane;
    if (stream) {
      char buf[32]; snprintf(buf, sizeof(buf), "0x%llx",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(stream)));
      lane = std::string("Stream ") + buf + (is_sdma ? " [DMA]" : "");
    } else {
      lane = std::string(is_sdma ? "SDMA " : "Compute ") + std::to_string(gtid / 2);
    }
    PfEnsureTrack(t_uuid, proc_uuid, lane, dev_id, int(gtid), false);
  }
}

// ── Chunk callback — incremental pftrace writer ───────────────────────────────
// Processes one delivery batch: emits track descriptors, InternedData, and
// sorted event packets directly into g_pf_buf, then writes to g_pf_file.
// Called from the ChunkDeliveryThread (not on the main thread).
static void PfChunkCallback(const hipApiRecordExt* records, uint32_t count, uint32_t /*chunk_id*/, void* /*ud*/) {
  if (!g_pf_file || count == 0) return;
  g_pf_slabs_written.fetch_add(1, std::memory_order_relaxed);

  std::lock_guard<std::mutex> lk(g_pf_mtx);

  // Demangle any new kernel names in this batch.
  {
    std::lock_guard<std::mutex> klk(g_kernel_names_mtx);
    for (auto& kv : g_kernel_names) {
      if (kv.second.size() > 0 && kv.second[0] == '_') {
        std::string demangled;
        if (hip::helpers::demangleName(kv.second, demangled))
          kv.second = std::move(demangled);
      }
    }
  }

  // ── Accumulate new intern entries for this chunk ──────────────────────────
  std::vector<std::pair<uint64_t,std::string>> new_evts, new_anns, new_cats;

  // Pre-intern fixed strings.
  auto cat_hip    = PfInternCat("hip",    new_cats);
  auto cat_gpu    = PfInternCat("gpu",    new_cats);
  (void)cat_hip; (void)cat_gpu;
  // Pre-intern common annotation keys.
  auto iid_queue_id  = PfInternAnn("queue_id",  new_anns);
  auto iid_grid      = PfInternAnn("grid",      new_anns);
  auto iid_block     = PfInternAnn("block",     new_anns);
  auto iid_copy_kind = PfInternAnn("copy_kind", new_anns);
  auto iid_bytes     = PfInternAnn("bytes",     new_anns);
  auto iid_dst       = PfInternAnn("dst",       new_anns);
  auto iid_src       = PfInternAnn("src",       new_anns);
  auto iid_stream    = PfInternAnn("stream",    new_anns);
  auto iid_ptr       = PfInternAnn("ptr",       new_anns);
  auto iid_size_key  = PfInternAnn("size",      new_anns);
  std::array<uint64_t,16> iid_kidx;
  for (int k = 0; k < 16; ++k) iid_kidx[k] = PfInternAnn(std::to_string(k), new_anns);

  // Pre-intern event names that appear in this batch.
  for (uint32_t ri = 0; ri < count; ++ri) {
    const hipApiRecordExt& rec = records[ri];
    if (rec.api_name) PfInternEvt(rec.api_name, new_evts);
    if (rec.gpu.gpu_op_count == 0) continue;
    auto scan = [&](const hipGpuActivityExt& gop) {
      if (gop.op == OP_ID_DISPATCH && gop.kernel_name) {
        std::lock_guard<std::mutex> klk(g_kernel_names_mtx);
        auto it = g_kernel_names.find(gop.kernel_name);
        if (it != g_kernel_names.end()) PfInternEvt(it->second, new_evts);
      } else if (gop.op == OP_ID_COPY) {
        PfInternEvt(CopyKindName(gop.copy_kind), new_evts);
      } else {
        PfInternEvt("Barrier", new_evts);
      }
    };
    scan(rec.gpu);
    for (const hipGpuActivityExt* n = rec.gpu.next; n; n = n->next) scan(*n);
  }

  // ── Update alloc map ──────────────────────────────────────────────────────
  for (uint32_t ri = 0; ri < count; ++ri) {
    const hipApiRecordExt& rec = records[ri];
    if (!rec.api_name) continue;
    uint64_t ptr = reinterpret_cast<uintptr_t>(rec.memory1);
    if (IsAllocApi(rec.api_name) && ptr) {
      g_pf_alloc_map[ptr].push_back({rec.start_ns, 0, rec.size});
    } else if (IsFreeApi(rec.api_name) && ptr) {
      auto it = g_pf_alloc_map.find(ptr);
      if (it != g_pf_alloc_map.end()) {
        for (auto rit = it->second.rbegin(); rit != it->second.rend(); ++rit)
          if (rit->end_ns == 0) { rit->end_ns = rec.start_ns; break; }
      }
    }
    g_pf_trace_end_ns = std::max(g_pf_trace_end_ns, rec.end_ns);
  }

  // ── Emit InternedData packet if any new strings ────────────────────────────
  if (!new_evts.empty() || !new_anns.empty() || !new_cats.empty()) {
    ProtoMsg idata;
    for (auto& kv : new_evts) {
      ProtoMsg e; e.u64(kIName_iid, kv.first); e.str(kIName_name, kv.second);
      idata.msg(kIData_evt_names, e);
    }
    for (auto& kv : new_anns) {
      ProtoMsg e; e.u64(kIName_iid, kv.first); e.str(kIName_name, kv.second);
      idata.msg(kIData_ann_names, e);
    }
    for (auto& kv : new_cats) {
      ProtoMsg e; e.u64(kIName_iid, kv.first); e.str(kIName_name, kv.second);
      idata.msg(kIData_cat_names, e);
    }
    ProtoMsg pkt;
    pkt.msg(kPkt_interned_data, idata);
    pkt.u32(kPkt_seq_id, kGlobalSeq);
    // kSeqFlag_Cleared only on the first InternedData packet — subsequent chunks
    // only add new names (incremental), so Cleared must not be re-sent or Perfetto
    // discards all IIDs from prior chunks.
    uint32_t iflags = kSeqFlag_NeedsState;
    if (g_pf_first_interned) { iflags |= kSeqFlag_Cleared; g_pf_first_interned = false; }
    pkt.u32(kPkt_seq_flags, iflags);
    AppendPacket(g_pf_buf, pkt);
  }

  // ── Build within-chunk CPU→GPU flow map ──────────────────────────────────
  // Map from slot index (== chunk_id) to flow id for CPU→GPU arrow.
  std::unordered_map<uint32_t, uint64_t> chunk_cpu_gpu_fid;   // chunk_id → fid
  std::unordered_map<uint32_t, uint32_t> chunk_cpu_gpu_ord;   // chunk_id → target op ordinal
  for (uint32_t ri = 0; ri < count; ++ri) {
    const hipApiRecordExt& rec = records[ri];
    if (!rec.api_name || rec.gpu.gpu_op_count == 0) continue;
    uint32_t min_ord = UINT32_MAX; uint64_t min_ns = UINT64_MAX; uint32_t ord = 0;
    auto scan_b = [&](const hipGpuActivityExt& gop) {
      if (gop.begin_ns > 0 && gop.begin_ns < min_ns) { min_ns = gop.begin_ns; min_ord = ord; }
      ++ord;
    };
    scan_b(rec.gpu);
    for (const hipGpuActivityExt* n = rec.gpu.next; n; n = n->next) scan_b(*n);
    if (min_ord != UINT32_MAX) {
      chunk_cpu_gpu_fid[rec.chunk_id] = g_pf_next_fid++;
      chunk_cpu_gpu_ord[rec.chunk_id] = min_ord;
    }
  }

  // ── Collect and sort all event packets ───────────────────────────────────
  SortedPktList sorted_pkts;

  for (uint32_t ri = 0; ri < count; ++ri) {
    const hipApiRecordExt& rec = records[ri];
    if (!rec.api_name) continue;

    uint32_t ctid     = PfCompactTid(rec.thread_id);
    uint64_t cpu_uuid = CpuThreadUuid(ctid);
    PfEnsureCpuThread(ctid);

    uint64_t ts_ns  = rec.start_ns;
    uint64_t dur_ns = (rec.end_ns > rec.start_ns) ? (rec.end_ns - rec.start_ns) : 1;

    // CPU annotations
    std::vector<std::pair<uint64_t,std::string>> cpu_anns;
    char buf[32];
    if (rec.memory1) {
      snprintf(buf, sizeof(buf), "0x%llx",
               static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(rec.memory1)));
      cpu_anns.push_back({iid_ptr, buf});
    }
    if (rec.memory2) {
      snprintf(buf, sizeof(buf), "0x%llx",
               static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(rec.memory2)));
      cpu_anns.push_back({iid_src, buf});
    }
    if (rec.size) cpu_anns.push_back({iid_size_key, std::to_string(rec.size)});
    if (rec.stream) {
      snprintf(buf, sizeof(buf), "0x%llx",
               static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(rec.stream)));
      cpu_anns.push_back({iid_stream, buf});
    }

    // CPU→GPU outflow
    std::vector<uint64_t> cpu_out;
    auto cfit = chunk_cpu_gpu_fid.find(rec.chunk_id);
    if (cfit != chunk_cpu_gpu_fid.end()) cpu_out.push_back(cfit->second);

    uint64_t name_iid = g_pf_evt_iid.count(rec.api_name) ? g_pf_evt_iid[rec.api_name] : 0;
    EmitSliceSorted(sorted_pkts, cpu_uuid, ts_ns, dur_ns, rec.api_name,
                    name_iid, cat_hip, cpu_anns, cpu_out, {});

    if (rec.gpu.gpu_op_count == 0) continue;

    // GPU op slices
    uint32_t cpu_target_ord = chunk_cpu_gpu_ord.count(rec.chunk_id)
                              ? chunk_cpu_gpu_ord[rec.chunk_id] : UINT32_MAX;
    uint32_t op_ord = 0;
    auto emit_gpu = [&](const hipGpuActivityExt& gop) {
      if (gop.begin_ns == 0) { ++op_ord; return; }
      int sdma = (gop.op == OP_ID_COPY) &&
                 hipCopyKindIsSDMAExt(static_cast<HipCopyKindExt>(gop.copy_kind)) ? 1 : 0;
      uint64_t lane     = PfCompactLane(rec.stream, gop.queue_id);
      uint64_t gtid     = lane * 2 + sdma;
      uint64_t gpu_uuid = GpuThreadUuid(static_cast<int>(gop.device_id), gtid);
      PfEnsureGpuThread(static_cast<int>(gop.device_id), gtid, rec.stream);

      uint64_t g_ts  = gop.begin_ns;
      uint64_t g_dur = (gop.end_ns > gop.begin_ns) ? (gop.end_ns - gop.begin_ns) : 1;

      std::string gpu_name;
      if (gop.op == OP_ID_DISPATCH && gop.kernel_name) {
        std::lock_guard<std::mutex> klk(g_kernel_names_mtx);
        auto it = g_kernel_names.find(gop.kernel_name);
        if (it != g_kernel_names.end()) gpu_name = it->second;
      } else if (gop.op == OP_ID_COPY) {
        gpu_name = CopyKindName(gop.copy_kind);
      } else {
        gpu_name = "Barrier";
      }

      std::vector<std::pair<uint64_t,std::string>> gpu_anns;
      gpu_anns.push_back({iid_queue_id, std::to_string(gop.queue_id)});
      if (gop.op == OP_ID_DISPATCH && gop.grid_x) {
        snprintf(buf, sizeof(buf), "%ux%ux%u", gop.grid_x, gop.grid_y, gop.grid_z);
        gpu_anns.push_back({iid_grid, buf});
        snprintf(buf, sizeof(buf), "%ux%ux%u", gop.block_x, gop.block_y, gop.block_z);
        gpu_anns.push_back({iid_block, buf});
      }
      if (gop.op == OP_ID_COPY) {
        gpu_anns.push_back({iid_copy_kind, CopyKindName(gop.copy_kind)});
        gpu_anns.push_back({iid_bytes, std::to_string(gop.bytes)});
        if (gop.dst) {
          snprintf(buf, sizeof(buf), "0x%llx",
                   static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(gop.dst)));
          gpu_anns.push_back({iid_dst, buf});
        }
        if (gop.src) {
          snprintf(buf, sizeof(buf), "0x%llx",
                   static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(gop.src)));
          gpu_anns.push_back({iid_src, buf});
        }
      }
      if (rec.stream) {
        snprintf(buf, sizeof(buf), "0x%llx",
                 static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(rec.stream)));
        gpu_anns.push_back({iid_stream, buf});
      }
      if (gop.op == OP_ID_DISPATCH && gop.kernel_args && gop.kernel_args_size > 0) {
        const uint8_t* p   = gop.kernel_args;
        const uint8_t* end = p + gop.kernel_args_size;
        int kidx = 0;
        while (p + sizeof(uint32_t) <= end) {
          uint32_t sz; std::memcpy(&sz, p, sizeof(uint32_t)); p += sizeof(uint32_t);
          if (p + sz > end) break;
          char vbuf[32];
          if (sz == 8)      { uint64_t v; std::memcpy(&v, p, 8); snprintf(vbuf, sizeof(vbuf), "0x%llx", static_cast<unsigned long long>(v)); }
          else if (sz == 4) { uint32_t v; std::memcpy(&v, p, 4); snprintf(vbuf, sizeof(vbuf), "%u", v); }
          else               { snprintf(vbuf, sizeof(vbuf), "(sz=%u)", sz); }
          uint64_t key_iid = (kidx < 16) ? iid_kidx[kidx] : PfInternAnn(std::to_string(kidx), new_anns);
          gpu_anns.push_back({key_iid, vbuf});
          ++kidx; p += sz;
        }
      }

      // CPU→GPU flow terminates on the op with the earliest begin_ns.
      std::vector<uint64_t> in_flows;
      if (op_ord == cpu_target_ord && cfit != chunk_cpu_gpu_fid.end())
        in_flows.push_back(cfit->second);

      uint64_t gpu_name_iid = g_pf_evt_iid.count(gpu_name) ? g_pf_evt_iid[gpu_name] : 0;
      EmitSliceSorted(sorted_pkts, gpu_uuid, g_ts, g_dur, gpu_name,
                      gpu_name_iid, cat_gpu, gpu_anns, {}, in_flows);
      ++op_ord;
    };

    emit_gpu(rec.gpu);
    for (const hipGpuActivityExt* n = rec.gpu.next; n; n = n->next) emit_gpu(*n);
  }

  FlushSorted(g_pf_buf, sorted_pkts);

  // Write accumulated buffer to file.
  if (!g_pf_buf.empty()) {
    fwrite(g_pf_buf.data(), 1, g_pf_buf.size(), g_pf_file);
    fflush(g_pf_file);
    g_pf_buf.clear();
  }
}

// ── PfFlushAllocSlices — called at final close to emit memory lifetime slices.
static void PfFlushAllocSlices() {
  if (!g_pf_file || g_pf_alloc_map.empty()) return;
  // Close any still-open alloc lifetimes.
  for (auto& kv : g_pf_alloc_map)
    for (auto& ai : kv.second)
      if (ai.end_ns == 0) ai.end_ns = g_pf_trace_end_ns;

  // Ensure memory process track.
  if (!g_pf_emitted_tracks.count(MemProcessUuid())) {
    uint64_t total_bytes = 0;
    for (auto& kv : g_pf_alloc_map) for (auto& ai : kv.second) total_bytes += ai.size;
    char lbl[64];
    if (total_bytes >= 1024*1024)
      snprintf(lbl, sizeof(lbl), "GPU Memory (%.1f MB total)", total_bytes/(1024.0*1024.0));
    else
      snprintf(lbl, sizeof(lbl), "GPU Memory (%llu B total)",
               static_cast<unsigned long long>(total_bytes));
    EmitTrackDesc(g_pf_buf, MemProcessUuid(), 0, lbl, 2048, -1, true);
    g_pf_emitted_tracks.insert(MemProcessUuid());
  }

  std::vector<std::pair<uint64_t,std::string>> new_evts, new_anns, new_cats;
  auto iid_ptr      = PfInternAnn("ptr",  new_anns);
  auto iid_size_key = PfInternAnn("size", new_anns);
  auto cat_mem      = PfInternCat("memory", new_cats);

  SortedPktList sorted_pkts;

  for (auto& kv : g_pf_alloc_map) {
    uint64_t ptr = kv.first;
    uint64_t mem_tid  = (ptr >> 12) & 0xFFFFF;
    uint64_t mem_uuid = MemThreadUuid(mem_tid);
    char ptrbuf[32], sizebuf[32];
    snprintf(ptrbuf, sizeof(ptrbuf), "0x%llx", static_cast<unsigned long long>(ptr));
    uint64_t label_size = kv.second.empty() ? 0 : kv.second[0].size;
    if (label_size >= 1024*1024) snprintf(sizebuf, sizeof(sizebuf), "%.1f MB", label_size/(1024.0*1024.0));
    else if (label_size >= 1024) snprintf(sizebuf, sizeof(sizebuf), "%.1f KB", label_size/1024.0);
    else                         snprintf(sizebuf, sizeof(sizebuf), "%llu B", static_cast<unsigned long long>(label_size));
    EmitTrackDesc(g_pf_buf, mem_uuid, MemProcessUuid(),
                  std::string(ptrbuf)+" ("+sizebuf+")", 2048, int(mem_tid), false);
    g_pf_emitted_tracks.insert(mem_uuid);

    for (const PfAllocInfo& ai : kv.second) {
      std::string alloc_name = std::string(ptrbuf) + " (" + sizebuf + ")";
      uint64_t alloc_iid = PfInternEvt(alloc_name, new_evts);
      uint64_t dur_ns = (ai.end_ns > ai.start_ns) ? (ai.end_ns - ai.start_ns) : 1;
      std::vector<std::pair<uint64_t,std::string>> anns;
      anns.push_back({iid_ptr, ptrbuf});
      anns.push_back({iid_size_key, std::to_string(ai.size)});
      EmitSliceSorted(sorted_pkts, mem_uuid, ai.start_ns, dur_ns,
                      alloc_name, alloc_iid, cat_mem, anns, {}, {});
    }
  }

  // Emit InternedData for any new strings from alloc slices.
  if (!new_evts.empty() || !new_anns.empty() || !new_cats.empty()) {
    ProtoMsg idata;
    for (auto& kv : new_evts) { ProtoMsg e; e.u64(kIName_iid, kv.first); e.str(kIName_name, kv.second); idata.msg(kIData_evt_names, e); }
    for (auto& kv : new_anns) { ProtoMsg e; e.u64(kIName_iid, kv.first); e.str(kIName_name, kv.second); idata.msg(kIData_ann_names, e); }
    for (auto& kv : new_cats) { ProtoMsg e; e.u64(kIName_iid, kv.first); e.str(kIName_name, kv.second); idata.msg(kIData_cat_names, e); }
    ProtoMsg pkt;
    pkt.msg(kPkt_interned_data, idata);
    pkt.u32(kPkt_seq_id, kGlobalSeq);
    pkt.u32(kPkt_seq_flags, kSeqFlag_Cleared | kSeqFlag_NeedsState);
    AppendPacket(g_pf_buf, pkt);
  }

  FlushSorted(g_pf_buf, sorted_pkts);
  if (!g_pf_buf.empty()) {
    fwrite(g_pf_buf.data(), 1, g_pf_buf.size(), g_pf_file);
    g_pf_buf.clear();
  }
}

// ================================================================================================
void WriteProtoTraceImpl(const char* filepath) {
  PreDemangleKernelNames();

  std::string out;

  size_t total = g_rec_counter.load(std::memory_order_acquire);

  // ── String intern table ───────────────────────────────────────────────────
  // Assign a small integer IID to each unique string (event names, ann keys,
  // categories). All interned strings are emitted once in a single InternedData
  // packet before any events, replacing repeated full-string copies with 1-3
  // byte varint references.
  std::unordered_map<std::string, uint64_t> evt_iid_map;  // event name → iid
  std::unordered_map<std::string, uint64_t> ann_iid_map;  // annotation key → iid
  std::unordered_map<std::string, uint64_t> cat_iid_map;  // category → iid
  uint64_t next_iid = 1;
  auto intern_evt = [&](const std::string& s) -> uint64_t {
    auto [it, ins] = evt_iid_map.emplace(s, 0);
    if (ins) it->second = next_iid++;
    return it->second;
  };
  auto intern_ann = [&](const std::string& s) -> uint64_t {
    auto [it, ins] = ann_iid_map.emplace(s, 0);
    if (ins) it->second = next_iid++;
    return it->second;
  };
  auto intern_cat = [&](const std::string& s) -> uint64_t {
    auto [it, ins] = cat_iid_map.emplace(s, 0);
    if (ins) it->second = next_iid++;
    return it->second;
  };

  // Pre-intern fixed strings used everywhere
  const uint64_t kCatHip    = intern_cat("hip");
  const uint64_t kCatGpu    = intern_cat("gpu");
  const uint64_t kCatMemory = intern_cat("memory");
  // Pre-intern common annotation keys
  auto iid_queue_id   = intern_ann("queue_id");
  auto iid_grid       = intern_ann("grid");
  auto iid_block      = intern_ann("block");
  auto iid_copy_kind  = intern_ann("copy_kind");
  auto iid_bytes      = intern_ann("bytes");
  auto iid_dst        = intern_ann("dst");
  auto iid_src        = intern_ann("src");
  auto iid_stream     = intern_ann("stream");
  auto iid_ptr        = intern_ann("ptr");
  auto iid_size_key   = intern_ann("size");
  // Pre-intern karg index keys "0".."15"
  std::array<uint64_t,16> iid_kidx;
  for (int k = 0; k < 16; ++k) iid_kidx[k] = intern_ann(std::to_string(k));

  // ── Pass 1: build all maps needed before any emission ────────────────────
  std::unordered_map<uint64_t, uint32_t> tid_map;
  uint32_t next_tid = 0;
  auto compact_tid = [&](uint64_t raw) -> uint32_t {
    auto [it, ins] = tid_map.emplace(raw, next_tid);
    if (ins) ++next_tid;
    return it->second;
  };
  std::unordered_map<int, std::unordered_set<uint64_t>> device_gpu_tids;
  std::map<std::pair<int,uint64_t>, hipStream_t> gpu_tid_stream;
  std::unordered_map<uintptr_t, uint64_t> stream_lane_idx;
  uint64_t next_stream_lane = 0;
  auto compact_stream_lane = [&](hipStream_t stream, uint64_t queue_id) -> uint64_t {
    uintptr_t key = stream ? reinterpret_cast<uintptr_t>(stream) : (0x8000ULL | queue_id);
    auto [it, ins] = stream_lane_idx.emplace(key, next_stream_lane);
    if (ins) ++next_stream_lane;
    return it->second;
  };

  struct AllocInfoP { uint64_t start_ns, end_ns, size; };
  std::unordered_map<uint64_t, std::vector<AllocInfoP>> alloc_map;

  uint64_t t_end = 0;
  for (size_t c = 0; c < g_records.size(); ++c) {
    hipApiRecordExt* chunk = g_records[c];
    size_t base  = c * kChunkSize;
    size_t valid = (total > base) ? std::min(total - base, kChunkSize) : 0;
    for (size_t i = 0; i < valid; ++i) {
      const hipApiRecordExt& rec = chunk[i];
      t_end = std::max(t_end, rec.end_ns);
      compact_tid(rec.thread_id);
      if (rec.api_name) {
        intern_evt(rec.api_name);  // pre-intern all event names in pass 1
        uint64_t ptr = reinterpret_cast<uintptr_t>(rec.memory1);
        if (IsAllocApi(rec.api_name) && ptr) {
          alloc_map[ptr].push_back({rec.start_ns, 0, rec.size});
        } else if (IsFreeApi(rec.api_name) && ptr) {
          auto it = alloc_map.find(ptr);
          if (it != alloc_map.end()) {
            for (auto rit = it->second.rbegin(); rit != it->second.rend(); ++rit)
              if (rit->end_ns == 0) { rit->end_ns = rec.start_ns; break; }
          }
        }
      }
      if (rec.gpu.gpu_op_count == 0) continue;
      auto scan = [&](const hipGpuActivityExt& gop) {
        int sdma = (gop.op == OP_ID_COPY) &&
                   hipCopyKindIsSDMAExt(static_cast<HipCopyKindExt>(gop.copy_kind)) ? 1 : 0;
        uint64_t lane = compact_stream_lane(rec.stream, gop.queue_id);
        uint64_t gtid = lane * 2 + sdma;
        device_gpu_tids[static_cast<int>(gop.device_id)].insert(gtid);
        if (rec.stream) gpu_tid_stream[{static_cast<int>(gop.device_id), gtid}] = rec.stream;
      };
      scan(rec.gpu);
      for (const hipGpuActivityExt* n = rec.gpu.next; n; n = n->next) scan(*n);
    }
  }
  for (auto& kv : alloc_map)
    for (auto& ai : kv.second)
      if (ai.end_ns == 0) ai.end_ns = t_end;

  // ── Pass 2: pre-assign flow IDs ──────────────────────────────────────────
  // CPU→GPU: CPU slice emits ph='s', GPU slice emits ph='f' (forward in time).
  // Memory→GPU: alloc slice emits ph='s' (at alloc time), GPU slice emits ph='f'
  //   (at GPU time > alloc time). Arrow goes forward: Memory → GPU op that used it.
  uint64_t flow_id = 1;
  std::unordered_map<uint64_t, uint64_t> cpu_gpu_flow;        // slot → fid
  std::unordered_map<uint64_t, uint32_t> cpu_gpu_target_ord;  // slot → target ordinal (min begin_ns)
  // alloc_out_flows[(ptr, period_start_ns)] = fids where that period's memory slice emits ph='s'.
  // Keyed per-period so each alloc slice only emits the fids for GPU ops live during that period.
  std::map<std::pair<uint64_t,uint64_t>, std::vector<uint64_t>> alloc_out_flows;
  // gpu_recv_flows[slot*1000+op_ord] = fids where GPU slice emits ph='f' (arrow ends at GPU op)
  std::unordered_map<uint64_t, std::vector<uint64_t>> gpu_recv_flows;
  // Graph node→node flows: sender op emits ph='s', next op emits ph='f'.
  // Keyed by slot*1000+actual_ordinal (position in linked list).
  std::unordered_map<uint64_t, uint64_t> graph_node_out_flows; // sender key → fid
  std::unordered_map<uint64_t, uint64_t> graph_node_in_flows;  // receiver key → fid

  for (size_t c = 0; c < g_records.size(); ++c) {
    hipApiRecordExt* chunk = g_records[c];
    size_t base  = c * kChunkSize;
    size_t valid = (total > base) ? std::min(total - base, kChunkSize) : 0;
    for (size_t i = 0; i < valid; ++i) {
      const hipApiRecordExt& rec = chunk[i];
      if (!rec.api_name || rec.gpu.gpu_op_count == 0) continue;
      uint64_t slot = base + i;

      const bool is_graph_rec = rec.api_name && strncmp(rec.api_name, "hipGraphLaunch", 14) == 0;
      // CPU→GPU arrow: target the op with the earliest begin_ns (first to start executing),
      // not necessarily ordinal 0 (first callback to arrive). GPU callbacks fire in completion
      // order which can differ from dispatch order (e.g. SDMA ops on a separate engine).
      {
        uint32_t min_ord = UINT32_MAX;
        uint64_t min_ns  = UINT64_MAX;
        uint32_t ord = 0;
        auto scan_begin = [&](const hipGpuActivityExt& gop) {
          if (gop.begin_ns > 0 && gop.begin_ns < min_ns) { min_ns = gop.begin_ns; min_ord = ord; }
          ++ord;
        };
        scan_begin(rec.gpu);
        for (const hipGpuActivityExt* n = rec.gpu.next; n; n = n->next) scan_begin(*n);
        if (min_ord != UINT32_MAX) {
          uint64_t fid = flow_id++;
          cpu_gpu_flow[slot]       = fid;
          cpu_gpu_target_ord[slot] = min_ord;
        }
      }

      // GPU→memory flows
      uint32_t op_ord = 0;
      auto assign_mem_flows = [&](const hipGpuActivityExt& gop) {
        uint64_t key = slot * 1000 + op_ord++;
        uint64_t seen[18]; int nseen = 0;
        auto try_ptr = [&](uint64_t ptr) {
          if (!ptr) return;
          for (int s = 0; s < nseen; ++s) if (seen[s]==ptr) return;
          auto ait = alloc_map.find(ptr);
          if (ait == alloc_map.end()) return;
          // Find the alloc period that was live when this GPU op ran.
          // Only create an arrow if gop.begin_ns falls within an alloc period for ptr.
          uint64_t period_start = 0;
          for (const auto& ai : ait->second) {
            if (ai.start_ns <= gop.begin_ns && gop.begin_ns <= ai.end_ns) {
              period_start = ai.start_ns; break;
            }
          }
          if (!period_start) return;  // GPU op outside all alloc periods for this ptr
          seen[nseen++] = ptr;
          uint64_t fid = flow_id++;
          gpu_recv_flows[key].push_back(fid);
          alloc_out_flows[{ptr, period_start}].push_back(fid);
        };
        if (gop.op == OP_ID_DISPATCH && gop.kernel_args && gop.kernel_args_size > 0) {
          const uint8_t* p = gop.kernel_args, *end = p + gop.kernel_args_size;
          while (p + sizeof(uint32_t) <= end) {
            uint32_t sz; std::memcpy(&sz, p, sizeof(uint32_t)); p += sizeof(uint32_t);
            if (p + sz > end) break;
            if (sz == 8) { uint64_t v; std::memcpy(&v, p, 8); try_ptr(v); }
            p += sz;
          }
        } else if (gop.op == OP_ID_COPY) {
          try_ptr(reinterpret_cast<uintptr_t>(gop.dst));
          try_ptr(reinterpret_cast<uintptr_t>(gop.src));
        }
      };
      assign_mem_flows(rec.gpu);
      for (const hipGpuActivityExt* n = rec.gpu.next; n; n = n->next) assign_mem_flows(*n);

      // Graph node→node flows sorted by begin_ns so arrows reflect execution order
      // regardless of callback arrival order (e.g. SDMA on a separate hardware engine).
      if (is_graph_rec && rec.gpu.gpu_op_count >= 2) {
        // Collect (begin_ns, ordinal) for each dispatch/copy op with valid timing.
        std::vector<std::pair<uint64_t,uint32_t>> sorted_ops;
        uint32_t ord = 0;
        auto collect = [&](const hipGpuActivityExt& gop) {
          if (gop.begin_ns > 0 &&
              (gop.op == HIP_OP_DISPATCH_EXT || gop.op == HIP_OP_COPY_EXT))
            sorted_ops.push_back({gop.begin_ns, ord});
          ++ord;
        };
        collect(rec.gpu);
        for (const hipGpuActivityExt* n = rec.gpu.next; n; n = n->next) collect(*n);
        std::stable_sort(sorted_ops.begin(), sorted_ops.end());
        for (size_t k = 0; k + 1 < sorted_ops.size(); ++k) {
          uint32_t from_ord = sorted_ops[k].second;
          uint32_t to_ord   = sorted_ops[k + 1].second;
          uint64_t fid = flow_id++;
          graph_node_out_flows[slot * 1000 + from_ord] = fid;
          graph_node_in_flows [slot * 1000 + to_ord]   = fid;
        }
      }
    }
  }

  // ── Emit ALL track descriptors before any events ──────────────────────────
  // Returns {targetId, hip_device_index} for a given HSA device_id.
  auto get_gfxip = [&](int dev_id) -> std::pair<std::string, int> {
    for (int pass = 0; pass < 2; ++pass) {
      for (size_t gi = 0; gi < hip::g_devices.size(); ++gi) {
        auto* hdev = hip::g_devices[gi];
        if (!hdev || hdev->devices().empty()) continue;
        bool match = (pass == 0) ? (hdev->deviceId() == dev_id)
                                 : (static_cast<int>(gi) == dev_id % static_cast<int>(hip::g_devices.size()));
        if (!match) continue;
        const char* tgt = hdev->devices()[0]->isa().targetId();
        if (tgt && tgt[0]) return {std::string(tgt), static_cast<int>(gi)};
      }
    }
    for (size_t gi = 0; gi < hip::g_devices.size(); ++gi) {
      auto* hdev = hip::g_devices[gi];
      if (!hdev || hdev->devices().empty()) continue;
      const char* tgt = hdev->devices()[0]->isa().targetId();
      if (tgt && tgt[0]) return {std::string(tgt), static_cast<int>(gi)};
    }
    return {"", -1};
  };

  std::string proc_name, proc_path;
  amd::Os::getAppPathAndFileName(proc_name, proc_path);
  std::string cpu_label = proc_name.empty() ? "CPU HIP" : ("CPU HIP [" + proc_name + "]");
  EmitTrackDesc(out, CpuProcessUuid(), 0, cpu_label, 1024, -1, true);
  for (auto& kv : tid_map) {
    EmitTrackDesc(out, CpuThreadUuid(kv.second), CpuProcessUuid(),
                  "HIP Thread " + std::to_string(kv.second), 1024, int(kv.second), false);
  }
  for (auto& kv : device_gpu_tids) {
    int dev_id = kv.first;
    auto [gfxip, hip_idx] = get_gfxip(dev_id);
    std::string label = gfxip.empty() ? ("GPU " + std::to_string(dev_id))
                                      : (gfxip + " [Device " + std::to_string(hip_idx) + "]");
    EmitTrackDesc(out, GpuProcessUuid(dev_id), 0, label, dev_id, -1, true);
    for (uint64_t gtid : kv.second) {
      bool is_sdma = (gtid & 1) != 0;
      std::string lane;
      auto sit = gpu_tid_stream.find({dev_id, gtid});
      if (sit != gpu_tid_stream.end() && sit->second) {
        char buf[32]; snprintf(buf, sizeof(buf), "0x%llx",
          static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(sit->second)));
        lane = std::string("Stream ") + buf + (is_sdma ? " [DMA]" : "");
      } else {
        lane = std::string(is_sdma ? "SDMA " : "Compute ") + std::to_string(gtid / 2);
      }
      EmitTrackDesc(out, GpuThreadUuid(dev_id, gtid), GpuProcessUuid(dev_id),
                    lane, dev_id, int(gtid), false);
    }
  }
  if (!alloc_map.empty()) {
    uint64_t total_bytes = 0;
    for (auto& kv : alloc_map) for (auto& ai : kv.second) total_bytes += ai.size;
    char lbl[64];
    if (total_bytes >= 1024*1024)
      snprintf(lbl, sizeof(lbl), "GPU Memory (%.1f MB total)", total_bytes/(1024.0*1024.0));
    else
      snprintf(lbl, sizeof(lbl), "GPU Memory (%llu B total)",
               static_cast<unsigned long long>(total_bytes));
    EmitTrackDesc(out, MemProcessUuid(), 0, lbl, 2048, -1, true);
    for (auto& kv : alloc_map) {
      uint64_t ptr = kv.first;
      uint64_t mem_tid = (ptr >> 12) & 0xFFFFF;
      char ptrbuf[32], sizebuf[32];
      snprintf(ptrbuf, sizeof(ptrbuf), "0x%llx", static_cast<unsigned long long>(ptr));
      uint64_t label_size = kv.second.empty() ? 0 : kv.second[0].size;
      if (label_size >= 1024 * 1024) {
        snprintf(sizebuf, sizeof(sizebuf), "%.1f MB", label_size / (1024.0 * 1024.0));
      } else if (label_size >= 1024) {
        snprintf(sizebuf, sizeof(sizebuf), "%.1f KB", label_size / 1024.0);
      } else {
        snprintf(sizebuf, sizeof(sizebuf), "%llu B", (unsigned long long)label_size);
      }
      EmitTrackDesc(out, MemThreadUuid(mem_tid), MemProcessUuid(),
                    std::string(ptrbuf)+" ("+sizebuf+")", 2048, int(mem_tid), false);
    }
  }

  // ── Pre-intern all dynamic event names so InternedData is complete ────────
  // GPU dispatch: demangled kernel names from g_kernel_names
  {
    std::lock_guard<std::mutex> lk(g_kernel_names_mtx);
    for (auto& kv : g_kernel_names) intern_evt(kv.second);
  }
  // GPU copy: all possible CopyKindName values
  for (int k = 0; k <= 12; ++k) intern_evt(CopyKindName(static_cast<uint32_t>(k)));
  intern_evt("Barrier");
  // Memory alloc names: "0xADDR (SIZE unit)"
  for (auto& kv : alloc_map) {
    if (kv.second.empty()) continue;
    uint64_t ptr = kv.first;
    uint64_t sz  = kv.second[0].size;
    char ptrbuf[32], sizebuf[32];
    snprintf(ptrbuf, sizeof(ptrbuf), "0x%llx", static_cast<unsigned long long>(ptr));
    if (sz >= 1024 * 1024) snprintf(sizebuf, sizeof(sizebuf), "%.1f MB", sz / (1024.0 * 1024.0));
    else if (sz >= 1024)   snprintf(sizebuf, sizeof(sizebuf), "%.1f KB", sz / 1024.0);
    else                   snprintf(sizebuf, sizeof(sizebuf), "%llu B", (unsigned long long)sz);
    intern_evt(std::string(ptrbuf) + " (" + sizebuf + ")");
  }

  // ── Emit InternedData packet (all IIDs now known) ─────────────────────────
  // One packet with SEQ_INCREMENTAL_STATE_CLEARED carries the full intern table.
  // Must come before any event packets that reference IIDs.
  {
    ProtoMsg idata;
    for (auto& kv : evt_iid_map) {
      ProtoMsg e; e.u64(kIName_iid, kv.second); e.str(kIName_name, kv.first);
      idata.msg(kIData_evt_names, e);
    }
    for (auto& kv : ann_iid_map) {
      ProtoMsg e; e.u64(kIName_iid, kv.second); e.str(kIName_name, kv.first);
      idata.msg(kIData_ann_names, e);
    }
    for (auto& kv : cat_iid_map) {
      ProtoMsg e; e.u64(kIName_iid, kv.second); e.str(kIName_name, kv.first);
      idata.msg(kIData_cat_names, e);
    }
    ProtoMsg pkt;
    pkt.msg(kPkt_interned_data, idata);
    pkt.u32(kPkt_seq_id, kGlobalSeq);
    // Both flags: Cleared resets intern state, NeedsState marks this as an interned-data carrier
    pkt.u32(kPkt_seq_flags, kSeqFlag_Cleared | kSeqFlag_NeedsState);
    AppendPacket(out, pkt);
  }

  // ── Pass 3: emit events ───────────────────────────────────────────────────
  // Collect all event packets into a sorted list keyed by timestamp, then flush
  // in order. Perfetto requires monotonically non-decreasing timestamps within a
  // trusted_packet_sequence; without sorting, long-duration events (e.g.
  // hipLaunchKernel) would emit their END packet before short events that ran later
  // but are stored in later slots, causing visual crossing in the Perfetto UI.
  SortedPktList sorted_pkts;
  std::vector<std::pair<uint64_t,std::string>> gpu_anns;
  std::vector<uint64_t> in_flows_vec;
  for (size_t c = 0; c < g_records.size(); ++c) {
    hipApiRecordExt* chunk = g_records[c];
    size_t base  = c * kChunkSize;
    size_t valid = (total > base) ? std::min(total - base, kChunkSize) : 0;
    for (size_t i = 0; i < valid; ++i) {
      const hipApiRecordExt& rec = chunk[i];
      if (!rec.api_name) continue;
      uint64_t slot     = base + i;
      uint32_t ctid     = compact_tid(rec.thread_id);
      uint64_t cpu_uuid = CpuThreadUuid(ctid);
      uint64_t ts_ns    = rec.start_ns;
      uint64_t dur_ns   = (rec.end_ns > rec.start_ns) ? (rec.end_ns - rec.start_ns) : 1;

      std::vector<uint64_t> cpu_out;
      auto cfit = cpu_gpu_flow.find(slot);
      if (cfit != cpu_gpu_flow.end()) cpu_out.push_back(cfit->second);

      std::vector<std::pair<uint64_t,std::string>> cpu_anns;
      {
        char buf[32];
        if (rec.memory1) {
          snprintf(buf, sizeof(buf), "0x%llx",
                   static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(rec.memory1)));
          cpu_anns.push_back({iid_ptr, buf});
        }
        if (rec.memory2) {
          snprintf(buf, sizeof(buf), "0x%llx",
                   static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(rec.memory2)));
          cpu_anns.push_back({iid_src, buf});
        }
        if (rec.size)
          cpu_anns.push_back({iid_size_key, std::to_string(rec.size)});
        if (rec.stream) {
          snprintf(buf, sizeof(buf), "0x%llx",
                   static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(rec.stream)));
          cpu_anns.push_back({iid_stream, buf});
        }
      }
      EmitSliceSorted(sorted_pkts, cpu_uuid, ts_ns, dur_ns, rec.api_name,
                      intern_evt(rec.api_name), kCatHip, cpu_anns, cpu_out, {});

      // GPU op slices
      auto ctoit = cpu_gpu_target_ord.find(slot);
      uint32_t cpu_target_ord = (ctoit != cpu_gpu_target_ord.end()) ? ctoit->second : UINT32_MAX;
      uint32_t op_ord = 0;
      auto emit_gpu = [&](const hipGpuActivityExt& gop) {
        if (gop.begin_ns == 0) { ++op_ord; return; }
        int sdma = (gop.op == OP_ID_COPY) &&
                   hipCopyKindIsSDMAExt(static_cast<HipCopyKindExt>(gop.copy_kind)) ? 1 : 0;
        uint64_t lane     = compact_stream_lane(rec.stream, gop.queue_id);
        uint64_t gtid     = lane * 2 + sdma;
        uint64_t gpu_uuid = GpuThreadUuid(static_cast<int>(gop.device_id), gtid);
        uint64_t g_ts     = gop.begin_ns;
        uint64_t g_dur    = (gop.end_ns > gop.begin_ns) ? (gop.end_ns - gop.begin_ns) : 1;

        std::string gpu_name;
        if (gop.op == OP_ID_DISPATCH && gop.kernel_name) {
          // PreDemangleKernelNames() already ran — just look up the (now-demangled) copy.
          // Never use gop.kernel_name at write time — it may be dangling.
          std::lock_guard<std::mutex> lk(g_kernel_names_mtx);
          auto it = g_kernel_names.find(gop.kernel_name);
          if (it != g_kernel_names.end()) gpu_name = it->second;
          // If not found in map, kernel_name is dangling — leave gpu_name empty.
        } else if (gop.op == OP_ID_COPY) {
          gpu_name = CopyKindName(gop.copy_kind);
        } else { gpu_name = "Barrier"; }

        gpu_anns.clear();
        {
          char buf[32];
          gpu_anns.push_back({iid_queue_id, std::to_string(gop.queue_id)});
          if (gop.op == OP_ID_DISPATCH && gop.grid_x) {
            snprintf(buf, sizeof(buf), "%ux%ux%u", gop.grid_x, gop.grid_y, gop.grid_z);
            gpu_anns.push_back({iid_grid, buf});
            snprintf(buf, sizeof(buf), "%ux%ux%u", gop.block_x, gop.block_y, gop.block_z);
            gpu_anns.push_back({iid_block, buf});
          }
          if (gop.op == OP_ID_COPY) {
            gpu_anns.push_back({iid_copy_kind, CopyKindName(gop.copy_kind)});
            gpu_anns.push_back({iid_bytes, std::to_string(gop.bytes)});
            if (gop.dst) {
              snprintf(buf, sizeof(buf), "0x%llx",
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(gop.dst)));
              gpu_anns.push_back({iid_dst, buf});
            }
            if (gop.src) {
              snprintf(buf, sizeof(buf), "0x%llx",
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(gop.src)));
              gpu_anns.push_back({iid_src, buf});
            }
          }
          if (rec.stream) {
            snprintf(buf, sizeof(buf), "0x%llx",
              static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(rec.stream)));
            gpu_anns.push_back({iid_stream, buf});
          }
        }
        // Kernel args — same positional format as JSON writer
        if (gop.op == OP_ID_DISPATCH && gop.kernel_args && gop.kernel_args_size > 0) {
          const uint8_t* p   = gop.kernel_args;
          const uint8_t* end = p + gop.kernel_args_size;
          int kidx = 0;
          while (p + sizeof(uint32_t) <= end) {
            uint32_t sz; std::memcpy(&sz, p, sizeof(uint32_t)); p += sizeof(uint32_t);
            if (p + sz > end) break;
            char vbuf[32];
            if (sz == 8) {
              uint64_t v; std::memcpy(&v, p, 8);
              snprintf(vbuf, sizeof(vbuf), "0x%llx", static_cast<unsigned long long>(v));
            } else if (sz == 4) {
              uint32_t v; std::memcpy(&v, p, 4);
              snprintf(vbuf, sizeof(vbuf), "%u", v);
            } else {
              snprintf(vbuf, sizeof(vbuf), "(sz=%u)", sz);
            }
            uint64_t key_iid = (kidx < 16) ? iid_kidx[kidx] : intern_ann(std::to_string(kidx));
            gpu_anns.push_back({key_iid, vbuf});
            ++kidx; p += sz;
          }
        }

        // GPU emits ph='f' for: CPU→GPU flow AND memory→GPU flows (all terminate here)
        in_flows_vec.clear();
        // CPU→GPU arrow terminates on the op with the earliest begin_ns (pre-computed
        // in Pass 2 as cpu_gpu_target_ord). This correctly handles graphs where callbacks
        // fire out of dispatch order (e.g. SDMA on a separate hardware engine).
        if (op_ord == cpu_target_ord && cfit != cpu_gpu_flow.end())
          in_flows_vec.push_back(cfit->second);
        auto rfit = gpu_recv_flows.find(slot * 1000 + op_ord);
        if (rfit != gpu_recv_flows.end())
          in_flows_vec.insert(in_flows_vec.end(), rfit->second.begin(), rfit->second.end());
        // Graph node→node: this op receives from previous node
        auto gnif = graph_node_in_flows.find(slot * 1000 + op_ord);
        if (gnif != graph_node_in_flows.end()) in_flows_vec.push_back(gnif->second);

        // Graph node→node: this op sends to next node
        std::vector<uint64_t> out_flows_vec;
        auto gnof = graph_node_out_flows.find(slot * 1000 + op_ord);
        if (gnof != graph_node_out_flows.end()) out_flows_vec.push_back(gnof->second);

        ++op_ord;

        uint64_t gpu_name_iid = intern_evt(gpu_name);
        EmitSliceSorted(sorted_pkts, gpu_uuid, g_ts, g_dur, gpu_name,
                        gpu_name_iid, kCatGpu, gpu_anns, out_flows_vec, in_flows_vec);
      };

      if (rec.gpu.gpu_op_count > 0) {
        emit_gpu(rec.gpu);
        for (const hipGpuActivityExt* n = rec.gpu.next; n; n = n->next) emit_gpu(*n);
      }
    }
  }

  // ── Memory lifetime slices with terminating flows ─────────────────────────
  for (auto& kv : alloc_map) {
    uint64_t ptr = kv.first;
    uint64_t mem_tid  = (ptr >> 12) & 0xFFFFF;
    uint64_t mem_uuid = MemThreadUuid(mem_tid);
    char ptrbuf[32];
    snprintf(ptrbuf, sizeof(ptrbuf), "0x%llx", static_cast<unsigned long long>(ptr));
    for (const AllocInfoP& ai : kv.second) {
      uint64_t dur_ns = (ai.end_ns > ai.start_ns) ? (ai.end_ns - ai.start_ns) : 1;
      char sizebuf[32];
      if (ai.size >= 1024 * 1024) snprintf(sizebuf, sizeof(sizebuf), "%.1f MB", ai.size/(1024.0*1024.0));
      else if (ai.size >= 1024)   snprintf(sizebuf, sizeof(sizebuf), "%.1f KB", ai.size/1024.0);
      else                        snprintf(sizebuf, sizeof(sizebuf), "%llu B", (unsigned long long)ai.size);
      std::string alloc_name = std::string(ptrbuf) + " (" + sizebuf + ")";
      std::vector<std::pair<uint64_t,std::string>> anns;
      anns.push_back({iid_ptr, ptrbuf}); anns.push_back({iid_size_key, std::to_string(ai.size)});
      std::vector<uint64_t> out_flows;
      auto afit = alloc_out_flows.find({ptr, ai.start_ns});
      if (afit != alloc_out_flows.end()) out_flows = afit->second;
      uint64_t alloc_name_iid = intern_evt(alloc_name);
      EmitSliceSorted(sorted_pkts, mem_uuid, ai.start_ns, dur_ns, alloc_name,
                      alloc_name_iid, kCatMemory, anns, out_flows, {});
    }
  }

  FlushSorted(out, sorted_pkts);

  std::ofstream f(filepath, std::ios::binary);
  if (f.is_open()) f.write(out.data(), out.size());
}

// atexit handler — registered only when GPU_CLR_PROFILE_OUTPUT is set.
// Runs before static destructors so HIP devices are still alive for DrainAllDevices().
// Insert PID before the file extension: "trace.json" → "trace_1234.json"
// ================================================================================================
static std::string AddPidToPath(const std::string& path) {
  auto dot = path.rfind('.');
  std::string pid_str = "_" + std::to_string(amd::Os::getProcessId());
  if (dot == std::string::npos) {
    return path + pid_str;
  }
  return path.substr(0, dot) + pid_str + path.substr(dot);
}

// ================================================================================================
// Fires before Device::tearDown() via RuntimeTearDown::RegisterTearDownCallback.
// Drains all queues when profiling was active; writes trace if GPU_CLR_PROFILE_OUTPUT is set.
static void ProfilerAtExit() {
#if !defined(_WIN32)
  // Drain in-flight GPU work whenever profiling was active (env var or API),
  // so all ReportActivity callbacks arrive before we serialise records.
  // Skipped on Windows where KFD streams may already be partially torn down.
  if (IsProfilingActive()) DrainAllDevices();
#endif

  // Stop the chunk delivery thread (if running) and flush all remaining records.
  // Must happen after DrainAllDevices so all GPU callbacks have fired before the
  // final flush; the thread's stop path delivers with watermark = UINT32_MAX.
  if (g_chunk_thread.joinable()) {
    {
      std::lock_guard<std::mutex> lk(g_chunk_mtx);
      g_chunk_thread_stop.store(true, std::memory_order_relaxed);
    }
    g_chunk_cv.notify_one();
    g_chunk_thread.join();
    g_chunk_thread_stop.store(false, std::memory_order_relaxed);
  }

  // Write trace only when the env-var output path was configured.
  if (flagIsDefault(GPU_CLR_PROFILE_OUTPUT)) return;
  std::string path = AddPidToPath(GPU_CLR_PROFILE_OUTPUT);
  const char* ext = strrchr(path.c_str(), '.');
  bool pftrace_mode = (ext && strcmp(ext, ".pftrace") == 0);

  if (pftrace_mode && g_pf_file && g_pf_slabs_written.load(std::memory_order_relaxed) > 0) {
    // Streaming pftrace mode: chunk thread delivered records incrementally.
    // Flush memory lifetime slices and close the file.
    PfFlushAllocSlices();
    fclose(g_pf_file);
    g_pf_file = nullptr;
  } else if (pftrace_mode) {
    // No chunks were streamed (short run, or file open failed) — batch write.
    if (g_pf_file) { fclose(g_pf_file); g_pf_file = nullptr; }
    WriteProtoTraceImpl(path.c_str());
  } else {
    // JSON mode: always buffered batch write.
    WriteJsonTraceImpl(path.c_str());
  }
}

struct HipClrProfilerFinalizer {
  ~HipClrProfilerFinalizer() {
    for (auto* chunk : g_records) FreeChunk(chunk);
  }
} g_finalizer;

}  // anonymous namespace

// ================================================================================================
void HipCaptureKernelArgsExt(hipGpuActivityExt* gact, hipFunction_t func, void** args) {
  if (!gact || !func || !args) return;

  amd::Kernel* kernel = hip::asKernel(func);
  if (!kernel) return;

  const amd::KernelSignature& sig = kernel->signature();
  uint32_t nparams = sig.numParameters();   // user-visible params only (hidden excluded)
  if (nparams == 0) return;

  // Blob layout per param (positional order, 0..N-1):
  //   uint32_t value_size      — byte size of argument value
  //   uint8_t  value[value_size] — raw little-endian bytes
  size_t total = 0;
  for (uint32_t i = 0; i < nparams; ++i)
    total += sizeof(uint32_t) + sig.at(i).size_;

  uint8_t* blob = new uint8_t[total];
  uint8_t* p = blob;
  for (uint32_t i = 0; i < nparams; ++i) {
    const amd::KernelParameterDescriptor& desc = sig.at(i);
    uint32_t val_size = static_cast<uint32_t>(desc.size_);
    std::memcpy(p, &val_size, sizeof(uint32_t));
    p += sizeof(uint32_t);
    if (args[i])
      std::memcpy(p, args[i], val_size);
    else
      std::memset(p, 0, val_size);
    p += val_size;
  }

  gact->kernel_args      = blob;
  gact->kernel_args_size = static_cast<uint32_t>(total);
}

// ================================================================================================
void HipCaptureKernelArgsPackedExt(hipGpuActivityExt* gact, hipFunction_t func,
                                   const void* kernargs, size_t kernargs_size) {
  if (!gact || !func || !kernargs || kernargs_size == 0) {
    return;
  }

  amd::Kernel* kernel = hip::asKernel(func);
  if (!kernel) return;

  const amd::KernelSignature& sig = kernel->signature();
  uint32_t nparams = sig.numParameters();
  if (nparams == 0) return;

  // Build the same {size, bytes...} blob, reading each arg from its ABI offset.
  size_t total = 0;
  for (uint32_t i = 0; i < nparams; ++i)
    total += sizeof(uint32_t) + sig.at(i).size_;

  uint8_t* blob = new uint8_t[total];
  uint8_t* p    = blob;
  const uint8_t* buf = static_cast<const uint8_t*>(kernargs);
  for (uint32_t i = 0; i < nparams; ++i) {
    const amd::KernelParameterDescriptor& desc = sig.at(i);
    uint32_t val_size = static_cast<uint32_t>(desc.size_);
    std::memcpy(p, &val_size, sizeof(uint32_t));
    p += sizeof(uint32_t);
    if (desc.offset_ + desc.size_ <= kernargs_size)
      std::memcpy(p, buf + desc.offset_, val_size);
    else
      std::memset(p, 0, val_size);
    p += val_size;
  }

  gact->kernel_args      = blob;
  gact->kernel_args_size = static_cast<uint32_t>(total);
}

// ================================================================================================
void HipStoreGraphExecNodesExt(hipGraphExec_t exec, std::vector<HipGraphNodeInfoExt> nodes) {
  std::lock_guard<std::mutex> lk(g_graph_exec_mtx);
  g_graph_exec_nodes[reinterpret_cast<uintptr_t>(exec)] = std::move(nodes);
}

// ================================================================================================
void HipEraseGraphExecNodesExt(hipGraphExec_t exec) {
  std::lock_guard<std::mutex> lk(g_graph_exec_mtx);
  g_graph_exec_nodes.erase(reinterpret_cast<uintptr_t>(exec));
}

const std::vector<HipGraphNodeInfoExt>* HipGetGraphExecNodesExt(hipGraphExec_t exec) {
  std::lock_guard<std::mutex> lk(g_graph_exec_mtx);
  auto it = g_graph_exec_nodes.find(reinterpret_cast<uintptr_t>(exec));
  return (it != g_graph_exec_nodes.end()) ? &it->second : nullptr;
}

// ================================================================================================
// Capture kernel name and args for one graph kernel node into info->gpu.
// Interns the mangled name into g_kernel_names and stores a stable const char*
// in info->gpu.kernel_name.  Writes the arg blob into info->gpu.kernel_args/size.
// args may be NULL (stream-captured graphs may expose NULL kp.kernelParams).
void HipCaptureGraphNodeArgsExt(HipGraphNodeInfoExt* info, hipFunction_t func, void** args) {
  if (!info || !func) return;

  amd::Kernel* kernel = hip::asKernel(func);
  if (!kernel) return;

  // Intern the mangled name into g_kernel_names (same table used by the callback
  // and the trace writers).  Strip any trailing '\0' from the CLR std::string.
  // Pointers into unordered_map values are stable — store c_str() directly.
  {
    const std::string& raw = kernel->name();
    size_t len = raw.size();
    while (len > 0 && raw[len - 1] == '\0') {
      --len;
    }
    std::string mangled(raw.data(), len);
    // ar->kernel_name from the HSA callback is the same mangled name as kernel->name().
    // Use the mangled name as key so fill_dispatch_info's strcmp finds a match.
    // Demangling happens lazily at write time in PreDemangleKernelNames.
    std::lock_guard<std::mutex> lk(g_kernel_names_mtx);
    auto [it, ok] = g_kernel_names.emplace(mangled, mangled);
    (void)ok;
    info->gpu.kernel_name = it->first.c_str();
  }

  if (!args) return;

  const amd::KernelSignature& sig = kernel->signature();
  uint32_t nparams = sig.numParameters();
  if (nparams == 0) return;

  size_t total = 0;
  for (uint32_t i = 0; i < nparams; ++i) {
    total += sizeof(uint32_t) + sig.at(i).size_;
  }

  uint8_t* blob = new uint8_t[total];
  uint8_t* p = blob;
  for (uint32_t i = 0; i < nparams; ++i) {
    const amd::KernelParameterDescriptor& desc = sig.at(i);
    uint32_t val_size = static_cast<uint32_t>(desc.size_);
    std::memcpy(p, &val_size, sizeof(uint32_t));
    p += sizeof(uint32_t);
    if (args[i]) {
      std::memcpy(p, args[i], val_size);
    } else {
      std::memset(p, 0, val_size);
    }
    p += val_size;
  }

  info->gpu.kernel_args      = blob;
  info->gpu.kernel_args_size = static_cast<uint32_t>(total);
}

// ================================================================================================
// Called from each *Layer wrapper — mirrors reference GetActiveRecord().
// Allocates a record slot, writes slot index into correlation_id TLS so the
// GPU command that follows inherits it, stamps start_ns, returns the record.
hipApiRecordExt* HipGetActiveRecordExt(uint32_t api_id) {
  size_t slot = g_rec_counter.fetch_add(1, std::memory_order_relaxed);
  size_t idx  = slot / kChunkSize;

  if (idx == g_records.size()) {
    std::lock_guard<std::mutex> lk(g_alloc_mtx);
    if (idx == g_records.size()) {
      assert(idx < kMaxChunks && "HIP profiler record capacity exhausted (kMaxChunks reached)");
      if (idx < kMaxChunks) {
        g_records.push_back(AllocChunk());
      } else {
        idx = kMaxChunks - 1;  // clamp: slots alias but g_records stays race-free
      }
    }
  }

  hipApiRecordExt* rec = &g_records[idx][slot % kChunkSize];
  rec->api_name    = (api_id < kHipApiNamesCountExt) ? kHipApiNamesExt[api_id] : "unknown";
  rec->_flags_u64  = 0;
  rec->chunk_id    = g_next_chunk_id.fetch_add(1, std::memory_order_relaxed);
#if defined(_WIN32)
  static thread_local uint64_t cached_tid = static_cast<uint64_t>(GetCurrentThreadId());
  rec->thread_id   = cached_tid;
#else
  static thread_local uint64_t cached_tid = static_cast<uint64_t>(syscall(SYS_gettid));
  rec->thread_id   = cached_tid;
#endif

  // Tell the HIP runtime to tag the next GPU command with this slot index.
  // Mirrors: next_layer.hipRegisterTracerId(slot) in the reference tracer.
  amd::activity_prof::correlation_id = static_cast<activity_correlation_id_t>(slot);

  rec->start_ns = NowNs();
  return rec;
}

// ============================================================
// Internal API
// ============================================================
namespace hip {
  const HipDispatchTable*         GetHipDispatchTable();
  const HipCompilerDispatchTable* GetHipCompilerDispatchTable();
}


// ================================================================================================
// Shared helper — registers callback once and installs wrappers.
static void EnsureCallbackAndWrappers() {
  {
    std::lock_guard<std::mutex> lk(g_alloc_mtx);
    if (g_records.empty()) {
      // Correctness requirement: prevents reallocation that would race with the
      // lock-free reads of g_records.size() / g_records[idx] in HipGetActiveRecordExt.
      g_records.reserve(kMaxChunks);
      g_records.push_back(AllocChunk());
    }
  }
  if (!g_callback_registered.exchange(true, std::memory_order_acq_rel)) {
    // Save whatever callback is already registered (e.g. roctracer) so we
    // can chain to it and coexist with other profiling tools.
    g_prev_callback.store(
        amd::activity_prof::report_activity.load(std::memory_order_acquire),
        std::memory_order_release);
    hipRegisterTracerCallback(HipActivityCallbackExt);
  }
  HipProfilerInstallWrappersExt(const_cast<HipDispatchTable*>(hip::GetHipDispatchTable()));
  HipProfilerInstallCompilerWrappersExt(
      const_cast<HipCompilerDispatchTable*>(hip::GetHipCompilerDispatchTable()));
}

// ================================================================================================
void HipProfilerInitExt() {
  // Build the wrapper table once from the live dispatch table.
  HipProfilerBuildWrapperTableExt(const_cast<HipDispatchTable*>(hip::GetHipDispatchTable()));

#if defined(_WIN32)
  // RuntimeTearDown::~RuntimeTearDown() does not fire reliably during DLL
  // unload on Windows — use atexit instead.
  std::atexit(ProfilerAtExit);
#else
  // On Linux RuntimeTearDown fires before device teardown, which is the
  // correct point to drain queues and write the trace.
  amd::RuntimeTearDown::RegisterTearDownCallback("HipClrProfiler", ProfilerAtExit);
#endif

  // GPU_CLR_PROFILE_OUTPUT=<path>: presence enables env-var profiling mode and
  // sets the output path for the automatic trace written at process exit.
  if (flagIsDefault(GPU_CLR_PROFILE_OUTPUT)) return;

  EnsureCallbackAndWrappers();

  // For .pftrace output, register the internal streaming chunk callback.
  // Records are written incrementally as GPU activity completes; at exit
  // ProfilerAtExit flushes memory lifetime slices and closes the file.
  // For .json output (and all other extensions) buffered mode is used.
  {
    // Replicate AddPidToPath inline (the helper is static inside the anon namespace).
    std::string raw_path = GPU_CLR_PROFILE_OUTPUT;
    std::string pid_str  = "_" + std::to_string(amd::Os::getProcessId());
    auto dot = raw_path.rfind('.');
    std::string path = (dot == std::string::npos)
                       ? raw_path + pid_str
                       : raw_path.substr(0, dot) + pid_str + raw_path.substr(dot);
    const char* ext = strrchr(path.c_str(), '.');
    if (ext && strcmp(ext, ".pftrace") == 0) {
      FILE* f = fopen(path.c_str(), "wb");
      if (f) {
        g_pf_file = f;
        // Register PfChunkCallback as an internal client so it goes through
        // the same delivery path as external clients.
        {
          std::lock_guard<std::mutex> lk(g_chunk_clients_mtx);
          g_chunk_clients.push_back({PfChunkCallback, nullptr});
        }
        g_chunk_thread_stop = false;
        g_chunk_thread      = std::thread(ChunkDeliveryThread);
      } else {
        fprintf(stderr, "[profiler] warning: cannot open pftrace file %s — "
                        "falling back to batch write at exit\n", path.c_str());
      }
    }
  }
}

// ================================================================================================
uint64_t HipProfilerEnableExt() {
  uint64_t start_id = g_rec_counter.load(std::memory_order_acquire);
  int prev = g_enable_refcount.fetch_add(1, std::memory_order_acq_rel);
  if (prev == 0) {
    EnsureCallbackAndWrappers();
  }
  return start_id;
}

// ================================================================================================
uint64_t HipProfilerDisableExt() {
  // Always drain before decrementing so IsProfilingActive() remains true
  // during the drain and all ReportActivity callbacks are captured.
  DrainAllDevices();
  int prev = g_enable_refcount.fetch_sub(1, std::memory_order_acq_rel);
  if (prev <= 0) {
    g_enable_refcount.store(0, std::memory_order_relaxed);
    return g_rec_counter.load(std::memory_order_acquire);
  }
  if (prev == 1) {
    HipProfilerRemoveWrappersExt(const_cast<HipDispatchTable*>(hip::GetHipDispatchTable()));
    HipProfilerRemoveCompilerWrappersExt(
        const_cast<HipCompilerDispatchTable*>(hip::GetHipCompilerDispatchTable()));
  }
  return g_rec_counter.load(std::memory_order_acquire);
}

// ============================================================
// Public C extension API
// ============================================================
extern "C" {

// ================================================================================================
hipError_t hipProfilerEnableExt(uint64_t* start_record_id, uint64_t state) {
  (void)state;  // reserved for future feature flags; ignored in this version
  uint64_t id = HipProfilerEnableExt();
  if (start_record_id) *start_record_id = id;
  return hipSuccess;
}

// ================================================================================================
hipError_t hipProfilerDisableExt(uint64_t* end_record_id) {
  uint64_t id = HipProfilerDisableExt();
  if (end_record_id) *end_record_id = id;
  return hipSuccess;
}

// ================================================================================================
hipError_t hipProfilerGetRecordsExt(const hipApiRecordExt* const** chunks,
                                     size_t* chunk_count,
                                     size_t* chunk_size,
                                     size_t* total_count) {
  if (!chunks || !chunk_count || !chunk_size || !total_count)
    return hipErrorInvalidValue;

  // Snapshot under alloc lock so chunk_count and total_count are consistent.
  size_t nchunks, total;
  {
    std::lock_guard<std::mutex> lk(g_alloc_mtx);
    nchunks = g_records.size();
    total   = g_rec_counter.load(std::memory_order_relaxed);
  }

  *chunks      = reinterpret_cast<const hipApiRecordExt* const*>(g_records.data());
  *chunk_count = nchunks;
  *chunk_size  = kChunkSize;
  *total_count = total;
  return hipSuccess;
}

// ================================================================================================
hipError_t hipProfilerRegisterChunkCallbackExt(hipProfilerChunkCallback cb, void* user_data) {
  if (!cb) return hipErrorInvalidValue;
  bool first;
  {
    std::lock_guard<std::mutex> lk(g_chunk_clients_mtx);
    first = g_chunk_clients.empty();
    g_chunk_clients.push_back({cb, user_data});
  }
  // Start the delivery thread only if not already running (pftrace env-var
  // path may have started it before any external client registered).
  if (!g_chunk_thread.joinable()) {
    g_chunk_thread_stop = false;
    g_chunk_thread      = std::thread(ChunkDeliveryThread);
  }
  return hipSuccess;
}

}  // extern "C"
