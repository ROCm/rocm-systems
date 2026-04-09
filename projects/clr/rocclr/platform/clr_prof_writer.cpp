/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "platform/clr_prof_writer.hpp"
#include "platform/clr_prof_event_bus.hpp"
#include "platform/clr_prof_interface.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <unistd.h>

namespace amd::clr_prof {

// ---------------------------------------------------------------------------
// Chrome Trace Event Format writer
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
// ---------------------------------------------------------------------------

namespace {

enum class OutputFormat { kNone, kJson };

struct WriterState {
  FILE*               file{nullptr};
  OutputFormat        format{OutputFormat::kNone};
  clr_prof_subscriber_t subscriber{nullptr};
  std::mutex          mu;
  bool                first_event{true};  // for JSON comma-separation
  std::atomic<bool>   finalized{false};
};

WriterState g_writer;

// ── JSON helpers ─────────────────────────────────────────────────────────────

// Escape a C-string for JSON embedding.
static std::string JsonEscape(const char* s) {
  if (!s) return "null";
  std::string out;
  out.reserve(std::strlen(s) + 2);
  for (const char* p = s; *p; ++p) {
    switch (*p) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:   out += *p;
    }
  }
  return out;
}

// Emit one JSON trace event object.  Caller holds g_writer.mu.
static void JsonEmitEvent(const char* ph, const char* cat, const char* name,
                           uint64_t ts_ns, uint64_t dur_ns,
                           int32_t pid, uint32_t tid, uint64_t id,
                           const char* extra_key, const char* extra_val) {
  if (!g_writer.file) return;
  if (!g_writer.first_event) std::fputs(",\n", g_writer.file);
  g_writer.first_event = false;

  // Convert ns to µs (Chrome Trace uses microseconds).
  double ts_us  = static_cast<double>(ts_ns)  / 1e3;
  double dur_us = static_cast<double>(dur_ns)  / 1e3;

  std::fprintf(g_writer.file,
               R"({"ph":"%s","cat":"%s","name":"%s","ts":%.3f,"dur":%.3f,"pid":%d,"tid":%u,"id":%llu)",
               ph, cat, name, ts_us, dur_us, pid, tid,
               static_cast<unsigned long long>(id));

  if (extra_key && extra_val)
    std::fprintf(g_writer.file, R"(,"args":{"%s":"%s"})", extra_key, extra_val);

  std::fputc('}', g_writer.file);
}

// ── Subscriber callbacks ───────────────────────────────────────────────────

// We accumulate API enter timestamps in a simple TLS struct to avoid a map.
struct ApiEnterInfo {
  uint64_t timestamp_ns{0};
  uint32_t pid{0};
  uint32_t tid{0};
};
thread_local ApiEnterInfo tls_api_enter{};

static void OnGpuActivity(const clr_prof_gpu_record_t* rec, void* /*ud*/) {
  if (!rec || g_writer.finalized.load(std::memory_order_acquire)) return;

  const char* name = clr_prof_gpu_op_name(rec->op);
  const char* kname = (rec->op == CLR_PROF_OP_KERNEL_DISPATCH && rec->kernel_name)
                          ? rec->kernel_name
                          : nullptr;
  uint64_t dur_ns = (rec->end_ns >= rec->begin_ns) ? rec->end_ns - rec->begin_ns : 0;

  std::string escaped = kname ? JsonEscape(kname) : std::string{};

  std::lock_guard lock(g_writer.mu);
  if (g_writer.format == OutputFormat::kJson) {
    JsonEmitEvent("X",  // Complete event (has dur)
                  "gpu", kname ? escaped.c_str() : name,
                  rec->begin_ns, dur_ns,
                  rec->device_id, static_cast<uint32_t>(rec->queue_id),
                  rec->correlation_id,
                  kname ? nullptr : nullptr, nullptr);
  }
}

static void OnHipApi(const clr_prof_api_record_t* rec, void* /*ud*/) {
  if (!rec || g_writer.finalized.load(std::memory_order_acquire)) return;

  if (rec->phase == CLR_PROF_PHASE_ENTER) {
    // Save enter info in TLS; emit the complete event on EXIT when we have dur.
    tls_api_enter.timestamp_ns = rec->timestamp_ns;
    tls_api_enter.pid          = rec->pid;
    tls_api_enter.tid          = rec->tid;
    return;
  }

  // EXIT: emit complete event.
  uint64_t begin_ns = tls_api_enter.timestamp_ns;
  uint64_t dur_ns   = (rec->timestamp_ns >= begin_ns) ? rec->timestamp_ns - begin_ns : 0;
  const char* name  = clr_prof_api_name(rec->api_id);

  std::lock_guard lock(g_writer.mu);
  if (g_writer.format == OutputFormat::kJson) {
    JsonEmitEvent("X", "hip", name ? name : "unknown",
                  begin_ns, dur_ns,
                  static_cast<int32_t>(rec->pid), rec->tid,
                  rec->correlation_id, nullptr, nullptr);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void WriterInit() {
  const char* path = std::getenv("CLR_TRACE_OUTPUT");
  if (!path || path[0] == '\0') return;

  // Determine format from extension.
  std::string_view sv(path);
  OutputFormat fmt = OutputFormat::kNone;
  if (sv.ends_with(".json"))
    fmt = OutputFormat::kJson;
  else if (sv.ends_with(".perfetto"))
    // Perfetto proto support is a future extension; fall back to JSON for now.
    fmt = OutputFormat::kJson;
  else
    fmt = OutputFormat::kJson;  // Default: treat as JSON.

  FILE* f = std::fopen(path, "w");
  if (!f) {
    std::fprintf(stderr, "[CLR] Warning: cannot open trace output file '%s'\n", path);
    return;
  }

  {
    std::lock_guard lock(g_writer.mu);
    g_writer.file        = f;
    g_writer.format      = fmt;
    g_writer.first_event = true;
    g_writer.finalized.store(false, std::memory_order_relaxed);
  }

  // Write JSON header.
  if (fmt == OutputFormat::kJson) std::fputs("{\"traceEvents\":[\n", f);

  // Register with the EventBus.
  clr_prof_callbacks_t cbs{};
  cbs.struct_size  = sizeof(cbs);
  cbs.gpu_activity = OnGpuActivity;
  cbs.hip_api      = OnHipApi;
  cbs.user_data    = nullptr;

  g_writer.subscriber = EventBus::instance().subscribe(&cbs, nullptr);
}

void WriterFini() {
  if (g_writer.finalized.exchange(true, std::memory_order_acq_rel)) return;

  if (g_writer.subscriber) {
    EventBus::instance().unsubscribe(g_writer.subscriber);
    g_writer.subscriber = nullptr;
  }

  std::lock_guard lock(g_writer.mu);
  if (!g_writer.file) return;

  if (g_writer.format == OutputFormat::kJson) {
    // Close the traceEvents array and add metadata.
    std::fprintf(g_writer.file,
                 "\n],\n"
                 "\"displayTimeUnit\":\"ns\",\n"
                 "\"otherData\":{\"generator\":\"CLR clr_prof built-in writer\"}\n"
                 "}\n");
  }

  std::fclose(g_writer.file);
  g_writer.file = nullptr;
}

}  // namespace amd::clr_prof
