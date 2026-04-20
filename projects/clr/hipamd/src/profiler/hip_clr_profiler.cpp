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
 * Chunk storage: g_records holds HipApiRecordExt arrays.
 *   _pad1[80] is repurposed to hold a std::vector<HipGpuActivityExt> (placement new).
 *   Op-1 lives in rec.gpu; ops 2..N spill into this vector (graph launches only).
 *   Accessed via GpuOps() helpers in hip_clr_profiler.hpp.
 */

#include "hip_clr_profiler.hpp"
#include "hip/amd_detail/hip_api_trace.hpp"
#include "platform/activity.hpp"
#include "../hip_internal.hpp"

#include "rocclr/os/os.hpp"
#include "utils/flags.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hip { const HipDispatchTable* GetHipDispatchTable(); }

extern "C" void hipRegisterTracerCallback(int (*function)(activity_domain_t domain,
                                                          uint32_t operation_id, void* data));

// ============================================================
// Internal state
// ============================================================
namespace {

inline uint64_t NowNs() { return amd::Os::timeNanos(); }

constexpr size_t kChunkSize = 10000;

// Two independent enable paths:
//   g_env_output_path — set by GPU_CLR_PROFILE_OUTPUT=<path> at Init(), never cleared.
//   g_api_enabled     — toggled by hipProfilerEnableExt/Disable.
// Recording is active when EITHER is live.
std::atomic<bool>        g_api_enabled{false};
std::atomic<bool>        g_callback_registered{false};
std::string              g_env_output_path;  // written once at init; read-only afterward

inline bool IsProfilingActive() {
  return !g_env_output_path.empty() ||
         g_api_enabled.load(std::memory_order_acquire);
}

// Previously registered callback saved before we register ours.
// Forwarded to at the end of HipActivityCallbackExt so we can coexist
// with roctracer / rocprofiler that may have registered first.
using activity_callback_t = int(*)(activity_domain_t, uint32_t, void*);
std::atomic<activity_callback_t> g_prev_callback{nullptr};

// Chunks of HipApiRecordExt.  _pad1[96] in each slot holds a
// std::vector<HipGpuActivityExt> (placement-new'd by AllocChunk).
// Pre-reserved so push_back never reallocates — required for the
// lock-free fast path in HipGetActiveRecordExt (double-checked locking).
std::vector<HipApiRecordExt*> g_records;
constexpr size_t kMaxChunks = 1024;  // 1024 * 10000 = 10M records max
std::atomic<size_t>           g_rec_counter{0};
std::mutex                    g_alloc_mtx;


std::unordered_map<const char*, std::string> g_kernel_names;
std::mutex                                   g_kernel_names_mtx;

// ============================================================
// Chunk lifecycle — placement-new the gpu_ops vector in _pad1.
// ============================================================
HipApiRecordExt* AllocChunk() {
  void* raw = ::operator new[](kChunkSize * sizeof(HipApiRecordExt));
  HipApiRecordExt* chunk = static_cast<HipApiRecordExt*>(raw);
  std::memset(chunk, 0, kChunkSize * sizeof(HipApiRecordExt));
  return chunk;
}

void FreeChunk(HipApiRecordExt* chunk) {
  for (size_t i = 0; i < kChunkSize; ++i)
    GpuOps(chunk[i]).~vector<HipGpuActivityExt>();
  ::operator delete[](static_cast<void*>(chunk));
}

// ============================================================
// Convert internal CL_COMMAND_* kind to the public HipCopyKindExt enum.
// Called once per copy activity record; result stored in record at capture time
// so the public API and JSON writer never see raw OpenCL constants.
// ============================================================
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

  HipApiRecordExt* rec = &g_records[idx][slot % kChunkSize];

  // OP_ID_BARRIER maps exclusively to CL_COMMAND_MARKER, which is used for all
  // GPU-side synchronization (graph node barriers, hipStreamWaitEvent, hipEventRecord
  // waits). Markers execute on the queue and carry real begin/end timestamps — record
  // them all so the trace shows where barriers land on the GPU timeline.

  if (ar->op == OP_ID_DISPATCH && ar->kernel_name) {
    std::lock_guard<std::mutex> lk(g_kernel_names_mtx);
    g_kernel_names.emplace(ar->kernel_name, std::string(ar->kernel_name));
  }

  if (rec->gpu.gpu_op_count == 0) {
    // First op: write directly into the embedded gpu field — no heap alloc.
    rec->gpu.op        = ar->op;
    rec->gpu.begin_ns  = ar->begin_ns;
    rec->gpu.end_ns    = ar->end_ns;
    rec->gpu.device_id = ar->device_id;
    rec->gpu.queue_id  = ar->queue_id;
    if (ar->op == OP_ID_DISPATCH)
      rec->gpu.kernel_name = ar->kernel_name;
    else if (ar->op == OP_ID_COPY) {
      rec->gpu.bytes     = ar->bytes;
      rec->gpu.copy_kind = ToCopyKindExt(ar->kind);
    }
    // OP_ID_BARRIER: no payload fields — begin/end timestamps already written above.
    rec->gpu.gpu_op_count = 1;
  } else {
    // Subsequent op (graph launch): spill into the overflow vector in _pad1.
    HipGpuActivityExt entry{};  // gpu_op_count=0, gpu_ops=nullptr, _pad1={0}
    entry.op        = ar->op;
    entry.begin_ns  = ar->begin_ns;
    entry.end_ns    = ar->end_ns;
    entry.device_id = ar->device_id;
    entry.queue_id  = ar->queue_id;
    if (ar->op == OP_ID_DISPATCH)
      entry.kernel_name = ar->kernel_name;
    else if (ar->op == OP_ID_COPY) {
      entry.bytes     = ar->bytes;
      entry.copy_kind = ToCopyKindExt(ar->kind);
    }
    GpuOps(*rec).push_back(entry);
    rec->gpu.gpu_op_count++;
  }

  // Forward to previously registered callback (e.g. roctracer / rocprofiler).
  auto* prev = g_prev_callback.load(std::memory_order_acquire);
  if (prev) prev(domain, op_id, data);
  return 0;
}

// ============================================================
// JSON output — Chrome Trace Event format (matches reference GoogleTrace())
// ============================================================
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

void WriteJsonTraceImpl(const char* filepath) {
  const char* path = (filepath && filepath[0]) ? filepath : "hip_clr_trace.json";
  std::ofstream trace(path, std::fstream::out);
  if (!trace.is_open()) return;

  const char* kGpuEvents[] = {"Dispatch", "Copy", "Barrier", "Unknown"};

  trace << "{\n  \"traceEvents\": [";

  size_t total      = g_rec_counter.load(std::memory_order_acquire);
  size_t event_id   = 0;
  // Track only the (device_id, gpu_tid) pairs that actually received events.
  // Key = device_id, Value = set of gpu tids (queue_id*2+sdma) with events.
  std::unordered_map<int, std::unordered_set<uint64_t>> device_gpu_tids;
  // Map raw hash thread ids to compact sequential ints (JS safe-integer limit)
  std::unordered_map<uint64_t, uint32_t> tid_map;
  uint32_t next_tid = 0;
  bool first = true;

  auto compact_tid = [&](uint64_t raw) -> uint32_t {
    auto it = tid_map.find(raw);
    if (it != tid_map.end()) return it->second;
    uint32_t id = next_tid++;
    tid_map[raw] = id;
    return id;
  };

  for (size_t c = 0; c < g_records.size(); ++c) {
    HipApiRecordExt* chunk = g_records[c];
    size_t base  = c * kChunkSize;
    size_t valid = (total > base) ? std::min(total - base, kChunkSize) : 0;
    if (valid == 0) continue;

    for (size_t i = 0; i < valid; ++i) {
      const HipApiRecordExt& rec = chunk[i];

      uint64_t s_time  = rec.start_ns / 1000;  // ns → µs
      uint64_t dur_us  = (rec.end_ns > rec.start_ns)
                         ? (rec.end_ns - rec.start_ns) / 1000 : 1;

      if (!first) trace << ",";
      first = false;

      uint32_t ctid = compact_tid(rec.thread_id);
      trace << "\n{\"name\":\"" << rec.api_name
            << "\",\"ph\":\"X\",\"pid\":1024,\"tid\":" << ctid
            << ",\"ts\":" << s_time << ",\"dur\":" << dur_us;
      if (rec.stream)
        trace << ",\"args\":{\"stream\":\"0x" << std::hex << reinterpret_cast<uintptr_t>(rec.stream) << std::dec << "\"}";
      trace << "}";

      // Emit one GPU op event (flow arrow + X event + flow finish).
      // Called for op-1 (from rec.gpu) and ops 2..N (from GpuOps vector).
      auto emit_gpu_op = [&](uint32_t op, uint64_t begin_ns, uint64_t end_ns,
                              int device_id, uint64_t queue_id,
                              const char* kernel_name, size_t bytes,
                              uint32_t copy_kind) {
        uint32_t op_idx  = op < 3 ? op : 3;
        // H2D / D2H transfers cross PCIe → SDMA engine.
        // D2D, buffer↔image copies use the GPU blit/compute engine.
        int sdma = (op_idx == OP_ID_COPY) &&
                   hipCopyKindIsSDMAExt(static_cast<HipCopyKindExt>(copy_kind)) ? 1 : 0;
        uint64_t gpu_dur = (end_ns > begin_ns) ? (end_ns - begin_ns) / 1000 : 1;
        uint64_t gpu_ts  = begin_ns / 1000;

        trace << ",\n{\"ts\":" << s_time
              << ",\"ph\":\"s\",\"id\":" << event_id
              << ",\"pid\":1024,\"tid\":" << ctid << ",\"name\":\"dep\"}";

        const char* gpu_name_cstr = (op_idx == OP_ID_COPY) ? CopyKindName(copy_kind)
                                                            : kGpuEvents[op_idx];
        std::string gpu_name = gpu_name_cstr;
        if (op_idx == OP_ID_DISPATCH && kernel_name) {
          std::lock_guard<std::mutex> lk(g_kernel_names_mtx);
          auto it = g_kernel_names.find(kernel_name);
          if (it != g_kernel_names.end()) gpu_name = it->second;
        }

        trace << ",\n{\"name\":\"" << gpu_name
              << "\",\"ph\":\"X\",\"pid\":" << device_id
              << ",\"tid\":" << (queue_id * 2 + sdma)
              << ",\"ts\":" << gpu_ts << ",\"dur\":" << gpu_dur;
        if (op_idx == OP_ID_COPY)
          trace << ",\"args\":{\"copy_kind\":\"" << CopyKindName(copy_kind)
                << "\",\"bytes\":" << bytes << "}";
        trace << "}";

        trace << ",\n{\"ts\":" << gpu_ts
              << ",\"ph\":\"f\",\"bp\":\"e\",\"id\":" << event_id
              << ",\"pid\":" << device_id
              << ",\"tid\":" << (queue_id * 2 + sdma)
              << ",\"name\":\"dep\"}";

        device_gpu_tids[device_id].insert(queue_id * 2 + sdma);
        ++event_id;
      };

      // Op-1 lives directly in rec.gpu; ops 2..N are in GpuOps(rec).
      if (rec.gpu.gpu_op_count > 0)
        emit_gpu_op(rec.gpu.op, rec.gpu.begin_ns, rec.gpu.end_ns,
                    rec.gpu.device_id, rec.gpu.queue_id,
                    rec.gpu.kernel_name, rec.gpu.bytes, rec.gpu.copy_kind);
      for (const auto& gop : GpuOps(rec))
        emit_gpu_op(gop.op, gop.begin_ns, gop.end_ns,
                    gop.device_id, gop.queue_id,
                    gop.kernel_name, gop.bytes, gop.copy_kind);
    }
  }

  // Emit CPU thread name metadata using the same compact tid mapping
  for (auto& kv : tid_map) {
    trace << ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1024,\"tid\":" << kv.second
          << ",\"args\":{\"name\":\"HIP Thread " << kv.second << "\"}}";
  }

  // Build device_id -> gfxip name map.
  // Try device_id directly as HIP index; also try device_id-1 (some backends
  // use 1-based device ids in activity records).
  auto get_gfxip = [&](int idx) -> std::string {
    if (idx < 0 || idx >= static_cast<int>(hip::g_devices.size())) return "";
    auto* hdev = hip::g_devices[idx];
    if (!hdev || hdev->devices().empty()) return "";
    const char* tgt = hdev->devices()[0]->isa().targetId();
    return (tgt && tgt[0]) ? std::string(tgt) : "";
  };

  // CPU process metadata (sort_index 0 so it appears first)
  trace << ",\n{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1024,"
           "\"args\":{\"name\":\"CPU HIP\"}}";
  trace << ",\n{\"name\":\"process_sort_index\",\"ph\":\"M\",\"pid\":1024,"
           "\"args\":{\"sort_index\":0}}";

  // Per-device GPU process metadata — only name tids that actually received events.
  int gpu_sort = 1;
  for (auto& kv : device_gpu_tids) {
    int dev_id = kv.first;
    const auto& active_tids = kv.second;

    std::string label = get_gfxip(dev_id);
    if (label.empty()) label = get_gfxip(dev_id - 1);  // try 1-based offset
    if (label.empty()) label = "GPU " + std::to_string(dev_id);

    trace << ",\n{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":" << dev_id
          << ",\"args\":{\"name\":\"" << label << "\"}}";
    trace << ",\n{\"name\":\"process_sort_index\",\"ph\":\"M\",\"pid\":" << dev_id
          << ",\"args\":{\"sort_index\":" << gpu_sort++ << "}}";

    // Emit a thread_name only for tids that actually have events.
    // Even tid = Compute (queue_id*2), odd tid = SDMA (queue_id*2+1).
    // queue_id==0 is the default/null HIP stream on the compute engine.
    for (uint64_t gpu_tid : active_tids) {
      bool is_sdma = (gpu_tid & 1) != 0;
      uint64_t q   = gpu_tid / 2;
      std::string lane_name;
      if (!is_sdma && q == 0)
        lane_name = "Default Stream";
      else
        lane_name = std::string(is_sdma ? "SDMA " : "Compute ") + std::to_string(q);
      trace << ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":" << dev_id
            << ",\"tid\":" << gpu_tid
            << ",\"args\":{\"name\":\"" << lane_name << "\"}}";
    }
  }

  trace << "\n  ],\n  \"displayTimeUnit\": \"us\"\n}\n";
  trace.close();
}

// Drain all GPU work on every device using the internal CLR path.
// Mirrors hipDeviceSynchronize() without going through HIP_INIT_API or our
// dispatch table wrappers, so no spurious profiling records are created and
// there is no re-entrancy risk.
static void DrainAllDevices() {
  for (auto* dev : hip::g_devices) {
    constexpr bool kWaitForCpu = false;
    dev->SyncAllStreams(kWaitForCpu);
  }
}

struct HipClrProfilerFinalizer {
  ~HipClrProfilerFinalizer() {
    // Auto-write when activated by env var.
    if (!g_env_output_path.empty()) {
      DrainAllDevices();
      WriteJsonTraceImpl(g_env_output_path.c_str());
    }
    for (auto* chunk : g_records) FreeChunk(chunk);
  }
} g_finalizer;

}  // anonymous namespace

// ============================================================
// Called from each *Layer wrapper — mirrors reference GetActiveRecord().
// Allocates a record slot, writes slot index into correlation_id TLS so the
// GPU command that follows inherits it, stamps start_ns, returns the record.
// ============================================================
HipApiRecordExt* HipGetActiveRecordExt(uint32_t api_id) {
  size_t slot = g_rec_counter.fetch_add(1, std::memory_order_relaxed);
  size_t idx  = slot / kChunkSize;

  if (idx == g_records.size()) {
    std::lock_guard<std::mutex> lk(g_alloc_mtx);
    if (idx == g_records.size())
      g_records.push_back(AllocChunk());
  }

  HipApiRecordExt* rec = &g_records[idx][slot % kChunkSize];
  rec->api_name    = (api_id < kHipApiNamesCountExt) ? kHipApiNamesExt[api_id] : "unknown";
  rec->_flags_u64  = 0;
  rec->thread_id   = static_cast<uint64_t>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));

  // Tell the HIP runtime to tag the next GPU command with this slot index.
  // Mirrors: next_layer.hipRegisterTracerId(slot) in the reference tracer.
  amd::activity_prof::correlation_id = static_cast<activity_correlation_id_t>(slot);

  rec->start_ns = NowNs();
  return rec;
}

// ============================================================
// Internal API
// ============================================================
// Shared helper — registers callback once and installs wrappers.
static void EnsureCallbackAndWrappers() {
  {
    std::lock_guard<std::mutex> lk(g_alloc_mtx);
    if (g_records.empty()) {
      g_records.reserve(kMaxChunks);  // prevent reallocation; required for lock-free fast path
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
}

void HipProfilerInitExt() {
  // GPU_CLR_PROFILE_OUTPUT=<path>: presence (non-empty) enables profiling;
  // the value is the output file path written at process exit.
  if (flagIsDefault(GPU_CLR_PROFILE_OUTPUT)) return;

  g_env_output_path = GPU_CLR_PROFILE_OUTPUT;
  EnsureCallbackAndWrappers();
}

void HipProfilerEnableExt() {
  g_api_enabled.store(true, std::memory_order_release);
  EnsureCallbackAndWrappers();
}

void HipProfilerDisableExt() {
  // Drain all outstanding GPU work before clearing the flag so that
  // ReportActivity callbacks for in-flight commands are delivered while
  // the profiler callback is still active.
  DrainAllDevices();
  g_api_enabled.store(false, std::memory_order_release);
  HipProfilerRemoveWrappersExt(const_cast<HipDispatchTable*>(hip::GetHipDispatchTable()));
}

void HipProfilerResetExt() {
  std::lock_guard<std::mutex> lk(g_alloc_mtx);
  for (auto* chunk : g_records) FreeChunk(chunk);
  g_records.clear();
  g_rec_counter.store(0, std::memory_order_relaxed);
}

void HipProfilerWriteJsonExt(const char* filepath) {
  WriteJsonTraceImpl(filepath);
}

// ============================================================
// Public C extension API
// ============================================================
extern "C" {

hipError_t hipProfilerEnableExt()  { HipProfilerEnableExt();  return hipSuccess; }
hipError_t hipProfilerDisableExt() { HipProfilerDisableExt(); return hipSuccess; }
hipError_t hipProfilerResetExt()   { HipProfilerResetExt();   return hipSuccess; }

hipError_t hipProfilerWriteJsonExt(const char* filepath) {
  HipProfilerWriteJsonExt(filepath);
  return hipSuccess;
}

hipError_t hipProfilerGetRecordsExt(const HipApiRecordExt* const** chunks,
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

  *chunks      = reinterpret_cast<const HipApiRecordExt* const*>(g_records.data());
  *chunk_count = nchunks;
  *chunk_size  = kChunkSize;
  *total_count = total;
  return hipSuccess;
}

}  // extern "C"
