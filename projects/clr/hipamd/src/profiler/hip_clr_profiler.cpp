/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * hip_clr_profiler.cpp — Built-in HIP CLR profiling layer.
 *
 * Activated by GPU_CLR_PROFILE=1 or programmatically via the
 * hipClrProfiler* extension API declared in hip_clr_profiler_ext.h.
 *
 * Design mirrors the reference ICD tracer (hip_tracer_core.cpp):
 *
 * CPU timing — dispatch table wrappers in hip_clr_dispatch_wrappers.cpp:
 *   auto* record = HipClrGetActiveRecord(api_id);   // allocs slot N, sets correlation_id TLS = N
 *   auto _r = g_next.hipFoo_fn(...);                // GPU command inherits correlation_id N
 *   record->end_ = high_resolution_clock::now();
 *
 * GPU timing — ReportActivityCallback (ACTIVITY_DOMAIN_HIP_OPS):
 *   ar->correlation_id == N  →  index directly into g_records[N/chunk][N%chunk]
 *   No map, no TLS sentinel, no pending table.
 */

#include "hip_clr_profiler.hpp"
#include "hip/amd_detail/hip_clr_profiler_ext.h"
#include "hip/amd_detail/hip_api_trace.hpp"
#include "platform/activity.hpp"
#include "../hip_internal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace hip { const HipDispatchTable* GetHipDispatchTable(); }

extern "C" void hipRegisterTracerCallback(int (*function)(activity_domain_t domain,
                                                          uint32_t operation_id, void* data));

// ============================================================
// Internal state
// ============================================================
namespace {

using Clock = std::chrono::high_resolution_clock;

constexpr size_t kChunkSize = 10000;

// Two independent enable bits:
//   g_env_enabled  — set by GPU_CLR_PROFILE=1 at Init(), never cleared at runtime.
//   g_api_enabled  — toggled by hipClrProfilerEnable/Disable and hipProfilerStart/Stop.
// Recording is active when EITHER bit is set.
// Wrappers are installed when EITHER is set, removed only when BOTH are clear.
std::atomic<bool>        g_env_enabled{false};
std::atomic<bool>        g_api_enabled{false};
std::atomic<bool>        g_callback_registered{false};
std::string              g_env_output_path;   // path from GPU_CLR_PROFILE value

inline bool IsProfilingActive() {
  return g_env_enabled.load(std::memory_order_acquire) ||
         g_api_enabled.load(std::memory_order_acquire);
}

// Previously registered callback saved before we register ours.
// Forwarded to at the end of HipClrActivityCallback so we can coexist
// with roctracer / rocprofiler that may have registered first.
using activity_callback_t = int(*)(activity_domain_t, uint32_t, void*);
std::atomic<activity_callback_t> g_prev_callback{nullptr};

std::vector<HipClrProfRecord*> g_records;
std::atomic<size_t>            g_rec_counter{0};
std::mutex                     g_alloc_mtx;

std::vector<HipClrApiRecord>   g_export_buf;
std::mutex                     g_export_mtx;

std::unordered_map<const char*, std::string> g_kernel_names;
std::mutex                                   g_kernel_names_mtx;

// ============================================================
// GPU ops callback — same logic as reference ReportActivityCallback.
// correlation_id == slot index → direct array lookup, no map needed.
// ============================================================
int HipClrActivityCallback(activity_domain_t domain, uint32_t op_id, void* data) {
  // Return 1 (disabled) for HIP_API domain so api_callbacks_spawner_t does NOT
  // overwrite amd::activity_prof::correlation_id with its own auto-increment value.
  // Our slot index written in HipClrGetActiveRecord must survive intact.
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

  HipClrProfRecord* rec = &g_records[idx][slot % kChunkSize];
  rec->has_gpu       = true;
  rec->gpu.op        = ar->op;
  rec->gpu.begin_ns  = ar->begin_ns;
  rec->gpu.end_ns    = ar->end_ns;
  rec->gpu.device_id = ar->device_id;
  rec->gpu.queue_id  = ar->queue_id;
  if (ar->op == OP_ID_DISPATCH) {
    rec->gpu.kernel_name = ar->kernel_name;
    if (ar->kernel_name) {
      std::lock_guard<std::mutex> lk(g_kernel_names_mtx);
      g_kernel_names.emplace(ar->kernel_name, std::string(ar->kernel_name));
    }
  } else {
    rec->gpu.bytes = ar->bytes;
  }

  // Forward to previously registered callback (e.g. roctracer / rocprofiler).
  auto* prev = g_prev_callback.load(std::memory_order_acquire);
  if (prev) prev(domain, op_id, data);
  return 0;
}

// ============================================================
// JSON output — Chrome Trace Event format (matches reference GoogleTrace())
// ============================================================
void WriteJsonTraceImpl(const char* filepath) {
  const char* path = (filepath && filepath[0]) ? filepath : "hip_clr_trace.json";
  std::ofstream trace(path, std::fstream::out);
  if (!trace.is_open()) return;

  const char* kGpuEvents[] = {"Dispatch", "Copy", "Barrier", "Unknown"};

  trace << "{\n  \"traceEvents\": [";

  size_t total      = g_rec_counter.load(std::memory_order_acquire);
  size_t event_id   = 0;
  std::unordered_map<int, uint64_t> device_queue_max;  // device_id -> max queue_id seen
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
    HipClrProfRecord* chunk = g_records[c];
    size_t base  = c * kChunkSize;
    size_t valid = (total > base) ? std::min(total - base, kChunkSize) : 0;
    if (valid == 0) continue;

    for (size_t i = 0; i < valid; ++i) {
      const HipClrProfRecord& rec = chunk[i];

      // Convert to µs using chrono — handles any clock period correctly.
      uint64_t s_time = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              rec.start_.time_since_epoch()).count());
      uint64_t duration = static_cast<uint64_t>(
          std::chrono::duration<double, std::micro>(rec.end_ - rec.start_).count());
      if (duration == 0) duration = 1;

      if (!first) trace << ",";
      first = false;

      uint32_t ctid = compact_tid(rec.thread_id);
      const char* api_name = (rec.api_id < kHipClrApiNamesCount)
                             ? kHipClrApiNames[rec.api_id] : "unknown";
      trace << "\n{\"name\":\"" << api_name
            << "\",\"ph\":\"X\",\"pid\":1024,\"tid\":" << ctid
            << ",\"ts\":" << s_time << ",\"dur\":" << duration << "}";

      if (rec.has_gpu) {
        uint32_t op  = rec.gpu.op < 3 ? rec.gpu.op : 3;
        int    sdma  = (op == OP_ID_DISPATCH) ? 0 : 1;
        uint64_t gpu_dur = (rec.gpu.end_ns > rec.gpu.begin_ns)
                           ? (rec.gpu.end_ns - rec.gpu.begin_ns) / 1000 : 1;
        uint64_t gpu_ts  = rec.gpu.begin_ns / 1000;

        trace << ",\n{\"ts\":" << s_time
              << ",\"ph\":\"s\",\"id\":" << event_id
              << ",\"pid\":1024,\"tid\":" << ctid << ",\"name\":\"dep\"}";

        std::string gpu_name = kGpuEvents[op];
        if (!sdma && rec.gpu.kernel_name) {
          std::lock_guard<std::mutex> lk(g_kernel_names_mtx);
          auto it = g_kernel_names.find(rec.gpu.kernel_name);
          if (it != g_kernel_names.end()) gpu_name = it->second;
        }

        trace << ",\n{\"name\":\"" << gpu_name
              << "\",\"ph\":\"X\",\"pid\":" << rec.gpu.device_id
              << ",\"tid\":" << (rec.gpu.queue_id * 2 + sdma)
              << ",\"ts\":" << gpu_ts << ",\"dur\":" << gpu_dur;
        if (sdma) trace << ",\"args\":{\"Copy size\":" << rec.gpu.bytes << "}";
        trace << "}";

        trace << ",\n{\"ts\":" << gpu_ts
              << ",\"ph\":\"f\",\"bp\":\"e\",\"id\":" << event_id
              << ",\"pid\":" << rec.gpu.device_id
              << ",\"tid\":" << (rec.gpu.queue_id * 2 + sdma)
              << ",\"name\":\"dep\"}";

        auto& qmax = device_queue_max[rec.gpu.device_id];
        qmax = std::max(qmax, rec.gpu.queue_id);
        ++event_id;
      }
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
  std::unordered_map<int, std::string> dev_to_name;  // device_id -> label
  auto get_gfxip = [&](int idx) -> std::string {
    if (idx < 0 || idx >= static_cast<int>(hip::g_devices.size())) return "";
    auto* hdev = hip::g_devices[idx];
    if (!hdev || hdev->devices().empty()) return "";
    const char* tid = hdev->devices()[0]->isa().targetId();
    return (tid && tid[0]) ? std::string(tid) : "";
  };
  for (auto& kv : device_queue_max) {
    int dev_id = kv.first;
    std::string name = get_gfxip(dev_id);
    if (name.empty()) name = get_gfxip(dev_id - 1);  // try 1-based offset
    if (name.empty()) name = "GPU " + std::to_string(dev_id);
    dev_to_name[dev_id] = name;
  }

  // CPU process metadata (sort_index 0 so it appears first)
  trace << ",\n{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1024,"
           "\"args\":{\"name\":\"CPU HIP\"}}";
  trace << ",\n{\"name\":\"process_sort_index\",\"ph\":\"M\",\"pid\":1024,"
           "\"args\":{\"sort_index\":0}}";

  // Per-device GPU process metadata — use device_id as pid (already in events)
  int gpu_sort = 1;
  for (auto& kv : device_queue_max) {
    int      dev_id = kv.first;
    uint64_t q_max  = kv.second;
    const std::string& label = dev_to_name.count(dev_id) ? dev_to_name[dev_id]
                                                          : ("GPU " + std::to_string(dev_id));

    trace << ",\n{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":" << dev_id
          << ",\"args\":{\"name\":\"" << label << "\"}}";
    trace << ",\n{\"name\":\"process_sort_index\",\"ph\":\"M\",\"pid\":" << dev_id
          << ",\"args\":{\"sort_index\":" << gpu_sort++ << "}}";
    for (uint64_t q = 0; q <= q_max; ++q) {
      trace << ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":" << dev_id
            << ",\"tid\":" << (q * 2)
            << ",\"args\":{\"name\":\"Compute " << q << "\"}}";
      trace << ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":" << dev_id
            << ",\"tid\":" << (q * 2 + 1)
            << ",\"args\":{\"name\":\"SDMA " << q << "\"}}";
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
    // Auto-write JSON only when GPU_CLR_PROFILE activated tracing.
    // App-controlled mode uses explicit hipClrProfilerWriteJson().
    if (g_env_enabled.load(std::memory_order_acquire)) {
      DrainAllDevices();
      WriteJsonTraceImpl(g_env_output_path.c_str());
    }
    for (auto* chunk : g_records) delete[] chunk;
  }
} g_finalizer;

}  // anonymous namespace

// ============================================================
// Called from each *Layer wrapper — mirrors reference GetActiveRecord().
// Allocates a record slot, writes slot index into correlation_id TLS so the
// GPU command that follows inherits it, stamps start_, returns the record.
// ============================================================
HipClrProfRecord* HipClrGetActiveRecord(uint32_t api_id) {
  if (!IsProfilingActive()) return nullptr;

  size_t slot = g_rec_counter.fetch_add(1, std::memory_order_relaxed);
  size_t idx  = slot / kChunkSize;

  {
    std::lock_guard<std::mutex> lk(g_alloc_mtx);
    if (idx == g_records.size()) {
      g_records.push_back(new HipClrProfRecord[kChunkSize]());
    }
  }

  HipClrProfRecord* rec = &g_records[idx][slot % kChunkSize];
  rec->api_id    = api_id;
  rec->thread_id = static_cast<uint64_t>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
  rec->has_gpu   = false;
  rec->result    = hipSuccess;

  // Tell the HIP runtime to tag the next GPU command with this slot index.
  // Mirrors: next_layer.hipRegisterTracerId(slot) in the reference tracer.
  amd::activity_prof::correlation_id = static_cast<activity_correlation_id_t>(slot);

  rec->start_ = Clock::now();
  return rec;
}

// ============================================================
// Internal API
// ============================================================
// Shared helper — registers callback once and installs wrappers.
static void EnsureCallbackAndWrappers() {
  {
    std::lock_guard<std::mutex> lk(g_alloc_mtx);
    if (g_records.empty())
      g_records.push_back(new HipClrProfRecord[kChunkSize]());
  }
  if (!g_callback_registered.exchange(true, std::memory_order_acq_rel)) {
    // Save whatever callback is already registered (e.g. roctracer) so we
    // can chain to it and coexist with other profiling tools.
    g_prev_callback.store(
        amd::activity_prof::report_activity.load(std::memory_order_acquire),
        std::memory_order_release);
    hipRegisterTracerCallback(HipClrActivityCallback);
  }
  HipClrProfilerInstallWrappers(const_cast<HipDispatchTable*>(hip::GetHipDispatchTable()));
}

void HipClrProfilerInit() {
  const char* env = getenv("GPU_CLR_PROFILE");
  if (!env || env[0] == '\0') return;

  // If value looks like a path (contains '/' or '.'), use it; otherwise use default.
  g_env_output_path = (strchr(env, '/') || strchr(env, '\\') || strchr(env, '.'))
                      ? env : "hip_clr_trace.json";
  g_env_enabled.store(true, std::memory_order_release);
  EnsureCallbackAndWrappers();
}

void HipClrProfilerEnable() {
  g_api_enabled.store(true, std::memory_order_release);
  EnsureCallbackAndWrappers();
}

void HipClrProfilerDisable() {
  // Drain all outstanding GPU work before clearing the flag so that
  // ReportActivity callbacks for in-flight commands are delivered while
  // the profiler callback is still active.
  DrainAllDevices();
  g_api_enabled.store(false, std::memory_order_release);
  // Only remove wrappers if GPU_CLR_PROFILE is also not holding them.
  if (!g_env_enabled.load(std::memory_order_acquire)) {
    HipClrProfilerRemoveWrappers(const_cast<HipDispatchTable*>(hip::GetHipDispatchTable()));
  }
}

void HipClrProfilerReset() {
  std::lock_guard<std::mutex> lk(g_alloc_mtx);
  for (auto* chunk : g_records) delete[] chunk;
  g_records.clear();
  g_rec_counter.store(0, std::memory_order_relaxed);
  std::lock_guard<std::mutex> elk(g_export_mtx);
  g_export_buf.clear();
}

void HipClrProfilerWriteJson(const char* filepath) {
  WriteJsonTraceImpl(filepath);
}

void HipClrProfilerGetRecords(const HipClrProfRecord** out_records, size_t* out_count) {
  *out_records = g_records.empty() ? nullptr : g_records[0];
  *out_count   = g_rec_counter.load(std::memory_order_acquire);
}

// ============================================================
// Public C extension API
// ============================================================
extern "C" {

hipError_t hipClrProfilerEnable()  { HipClrProfilerEnable();  return hipSuccess; }
hipError_t hipClrProfilerDisable() { HipClrProfilerDisable(); return hipSuccess; }
hipError_t hipClrProfilerReset()   { HipClrProfilerReset();   return hipSuccess; }

hipError_t hipClrProfilerWriteJson(const char* filepath) {
  HipClrProfilerWriteJson(filepath);
  return hipSuccess;
}

hipError_t hipClrProfilerGetRecords(const HipClrApiRecord** records, size_t* count) {
  if (!records || !count) return hipErrorInvalidValue;

  size_t total = g_rec_counter.load(std::memory_order_acquire);
  std::lock_guard<std::mutex> lk(g_export_mtx);
  g_export_buf.clear();
  g_export_buf.reserve(total);

  for (size_t c = 0; c < g_records.size(); ++c) {
    size_t base  = c * kChunkSize;
    size_t valid = (total > base) ? std::min(total - base, kChunkSize) : 0;
    if (valid == 0) continue;

    for (size_t i = 0; i < valid; ++i) {
      const HipClrProfRecord& src = g_records[c][i];

      HipClrApiRecord dst{};
      dst.api_id    = static_cast<uint32_t>(src.api_id);
      dst.thread_id = src.thread_id;
      dst.start_ns  = static_cast<uint64_t>(src.start_.time_since_epoch().count());
      dst.end_ns    = static_cast<uint64_t>(src.end_.time_since_epoch().count());
      dst.has_gpu_activity = src.has_gpu ? 1 : 0;
      if (src.has_gpu) {
        dst.gpu.op          = src.gpu.op;
        dst.gpu.begin_ns    = src.gpu.begin_ns;
        dst.gpu.end_ns      = src.gpu.end_ns;
        dst.gpu.device_id   = src.gpu.device_id;
        dst.gpu.queue_id    = src.gpu.queue_id;
        if (src.gpu.op == OP_ID_DISPATCH) {
          dst.gpu.kernel_name = src.gpu.kernel_name;
          dst.gpu.bytes       = 0;
        } else {
          dst.gpu.bytes       = src.gpu.bytes;
          dst.gpu.kernel_name = nullptr;
        }
      }
      g_export_buf.push_back(dst);
    }
  }

  *records = g_export_buf.data();
  *count   = g_export_buf.size();
  return hipSuccess;
}

}  // extern "C"
