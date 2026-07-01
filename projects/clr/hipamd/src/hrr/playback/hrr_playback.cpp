/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// hrr-playback: Full HIP workload replay with D2H data validation.
//
// Replays a .hrr archive using one CPU thread per captured thread.
// GPU-side parallelism is preserved via stream handles exactly as during
// capture. CPU-side synchronisation between threads (mutexes, barriers,
// atomics) is NOT replicated — only GPU-visible ordering is guaranteed.
//
// When --kernel-filter is used, a silent full warm-up pass runs first to
// populate all GPU buffers (including intermediate kernel outputs) before
// the filtered timed pass begins.
//
// Usage: hrr-playback <capture.hrr> [options]
//   --verbose             Print each event as it is processed
//   --skip-device-sync    Skip hipDeviceSynchronize / hipStreamSynchronize
//   --single-thread       Force single-threaded replay
//   --timing              Report wall time and total GPU kernel time
//   --kernel-filter STR   Only launch kernels whose name contains STR
//                         (full warm-up pass runs first to set up GPU state)
//   --replace-kernel N=P  Launch the kernel whose recorded name is exactly N
//                         from code object P (.hsaco) instead of the recorded
//                         one (repeatable). Recorded args/grid/block are reused;
//                         the archive is untouched.
//   --sync-after-launch   hipDeviceSynchronize() after every kernel (debug)
//   --trace-kernels       Print one compact line before every kernel launch
//   --trace-sync          Print sync begin/done markers around launched kernels
//   --progress-kernels N  Print heartbeat every N launched kernels
//   --progress-seconds S  Print heartbeat at most every S seconds
//   --help                Show this message
//
// Exit code: 0 = all D2H checks passed (or none present), 1 = any failure.

#include "hrr_reader.h"
#include "hip_playback.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

#define HIP_CHECK(call)                                                       \
  do {                                                                        \
    hipError_t _e = (call);                                                   \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "[HRR] HIP error %d (%s) at %s:%d\n",                  \
              _e, hipGetErrorString(_e), __FILE__, __LINE__);                 \
      return 1;                                                                \
    }                                                                         \
  } while (0)

static bool env_flag_enabled(const char* name) {
  const char* v = std::getenv(name);
  return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}

static size_t env_size_or(const char* name, size_t fallback) {
  const char* v = std::getenv(name);
  if (!v || v[0] == '\0') return fallback;
  char* end = nullptr;
  unsigned long long n = std::strtoull(v, &end, 10);
  return (end && *end == '\0') ? static_cast<size_t>(n) : fallback;
}

static double env_double_or(const char* name, double fallback) {
  const char* v = std::getenv(name);
  if (!v || v[0] == '\0') return fallback;
  char* end = nullptr;
  double n = std::strtod(v, &end);
  return (end && *end == '\0') ? n : fallback;
}

// ---------------------------------------------------------------------------
// --info mode: print archive summary without touching the GPU
// ---------------------------------------------------------------------------

struct ProcessInfo {
  uint64_t pid = 0;
  uint64_t parent_pid = 0;
  uint64_t event_count = 0;
  uint64_t blob_count = 0;
  bool complete = false;
};

static bool read_process_manifest(const fs::path& path, ProcessInfo& out) {
  FILE* f = fopen(path.string().c_str(), "r");
  if (!f) return false;

  ProcessInfo info{};
  bool saw_pid = false;
  char line[256];
  while (fgets(line, sizeof(line), f)) {
    unsigned long long u = 0;
    char b[8] = {};
    if (sscanf(line, " \"pid\": %llu", &u) == 1) {
      info.pid = static_cast<uint64_t>(u);
      saw_pid = true;
      continue;
    }
    if (sscanf(line, " \"parent_pid\": %llu", &u) == 1) {
      info.parent_pid = static_cast<uint64_t>(u);
      continue;
    }
    if (sscanf(line, " \"event_count\": %llu", &u) == 1) {
      info.event_count = static_cast<uint64_t>(u);
      continue;
    }
    if (sscanf(line, " \"blob_count\": %llu", &u) == 1) {
      info.blob_count = static_cast<uint64_t>(u);
      continue;
    }
    if (sscanf(line, " \"complete\": %7[^,\n ]", b) == 1) {
      info.complete = (strcmp(b, "true") == 0);
      continue;
    }
  }
  fclose(f);
  if (!saw_pid) return false;
  out = info;
  return true;
}

static bool read_root_owner_pid(const fs::path& root, uint64_t& owner_pid) {
  FILE* f = fopen((root / "manifest.json").string().c_str(), "r");
  if (!f) return false;

  bool found = false;
  char line[256];
  while (fgets(line, sizeof(line), f)) {
    unsigned long long u = 0;
    if (sscanf(line, " \"owner_pid\": %llu", &u) == 1) {
      owner_pid = static_cast<uint64_t>(u);
      found = true;
      break;
    }
  }
  fclose(f);
  return found;
}

static uint64_t derive_owner_pid(const std::vector<std::pair<fs::path, ProcessInfo>>& processes) {
  if (processes.empty()) return 0;
  for (const auto& candidate : processes) {
    for (const auto& child : processes) {
      if (child.second.parent_pid == candidate.second.pid)
        return candidate.second.pid;
    }
  }
  return processes.front().second.pid;
}

static bool print_root_info(const std::string& archive_path) {
  fs::path root(archive_path);
  if (fs::exists(root / "events.bin") || !fs::exists(root) || !fs::is_directory(root))
    return false;

  std::vector<std::pair<fs::path, ProcessInfo>> processes;
  for (const auto& ent : fs::directory_iterator(root)) {
    if (!ent.is_directory()) continue;
    const std::string name = ent.path().filename().string();
    if (name.rfind("pid-", 0) != 0) continue;
    ProcessInfo info{};
    if (!read_process_manifest(ent.path() / "manifest.json", info)) {
      // Keep incomplete/crashed directories visible even if their manifest was
      // never written; derive pid from the directory name when possible.
      char* end = nullptr;
      unsigned long long pid = std::strtoull(name.c_str() + 4, &end, 10);
      if (end && *end == '\0') info.pid = static_cast<uint64_t>(pid);
    }
    processes.emplace_back(ent.path(), info);
  }
  if (processes.empty()) return false;

  std::sort(processes.begin(), processes.end(),
            [](const auto& a, const auto& b) {
              return a.first.filename().string() < b.first.filename().string();
            });

  uint64_t owner_pid = 0;
  if (!read_root_owner_pid(root, owner_pid))
    owner_pid = derive_owner_pid(processes);

  printf("HRR Archive Root: %s\n", archive_path.c_str());
  printf("========================================\n");
  printf("Capture Mode: in-tree\n");
  printf("Owner PID:    %llu\n", static_cast<unsigned long long>(owner_pid));
  printf("Processes:    %zu\n\n", processes.size());
  printf("  %-12s %-12s %-10s %-12s %-10s %s\n",
         "PID", "Parent PID", "Complete", "Events", "Blobs", "Path");
  printf("  %-12s %-12s %-10s %-12s %-10s %s\n",
         "---", "----------", "--------", "------", "-----", "----");
  for (const auto& [path, info] : processes) {
    printf("  %-12llu %-12llu %-10s %-12llu %-10llu %s\n",
           static_cast<unsigned long long>(info.pid),
           static_cast<unsigned long long>(info.parent_pid),
           info.complete ? "yes" : "NO",
           static_cast<unsigned long long>(info.event_count),
           static_cast<unsigned long long>(info.blob_count),
           path.string().c_str());
  }
  printf("\n");
  return true;
}

static void print_info(const hrr::Archive& archive, bool show_events) {
  printf("HRR Archive: %s\n", archive.path.c_str());
  printf("========================================\n");
  printf("Complete:     %s\n",
         archive.complete ? "yes (clean shutdown)"
                          : (archive.truncated
                               ? "NO (crash-truncated; trailing torn record discarded)"
                               : "NO (no shutdown trailer; capture likely crashed)"));
  printf("Recovered:    %zu events\n", archive.event_count);
  printf("Events:       %zu\n", archive.event_count);
  printf("Kernels:      %zu\n", archive.kernel_count);
  printf("Blobs:        %zu\n", archive.blob_count);
  printf("Code Objects: %zu\n", archive.code_object_count);
  printf("Threads:      %zu\n", archive.threads.size());
  printf("\n");

  // Event type breakdown
  std::map<uint16_t, size_t> type_counts;
  for (const auto& ev : archive.events)
    type_counts[ev.header().event_type]++;

  printf("Event Type Breakdown:\n");
  printf("  %-22s %s\n", "Type", "Count");
  printf("  %-22s %s\n", "----", "-----");
  for (auto& [type, count] : type_counts)
    printf("  %-22s %zu\n", hrr::event_type_name(type), count);
  printf("\n");

  // Kernel summary
  if (archive.kernel_count > 0) {
    printf("Kernel Summary (first 20):\n");
    printf("  %-4s %-50s %-15s %-15s %s\n", "ID", "Kernel", "Grid", "Block", "SharedMem");
    printf("  %-4s %-50s %-15s %-15s %s\n", "--", "------", "----", "-----", "---------");

    size_t kid = 0;
    std::map<std::string, size_t> kernel_calls;
    for (const auto& ev : archive.events) {
      if (!ev.kernel_launch)
        continue;
      const auto& kl = *ev.kernel_launch;
      kernel_calls[kl.kernel_name]++;
      if (kid < 20) {
        char grid_str[32], block_str[32];
        snprintf(grid_str,  sizeof(grid_str),  "[%u,%u,%u]", kl.grid[0],  kl.grid[1],  kl.grid[2]);
        snprintf(block_str, sizeof(block_str), "[%u,%u,%u]", kl.block[0], kl.block[1], kl.block[2]);
        std::string name = kl.kernel_name;
        if (name.size() > 50) name = name.substr(0, 47) + "...";
        printf("  %-4zu %-50s %-15s %-15s %u\n", kid, name.c_str(), grid_str, block_str, kl.shared_mem);
      }
      kid++;
    }
    if (kid > 20) printf("  ... and %zu more\n", kid - 20);

    printf("\nKernel Call Counts:\n");
    printf("  %-60s %s\n", "Kernel", "Calls");
    printf("  %-60s %s\n", "------", "-----");
    for (auto& [name, count] : kernel_calls) {
      std::string d = name;
      if (d.size() > 60) d = d.substr(0, 57) + "...";
      printf("  %-60s %zu\n", d.c_str(), count);
    }
    printf("\n");
  }

  if (!show_events) return;

  printf("Event Log:\n");
  printf("  %-6s %-10s %-16s %-22s %s\n", "Seq", "Thread", "Timestamp(ns)", "Type", "Details");
  for (const auto& ev : archive.events) {
    printf("  %-6llu %-10llu %-16llu %-22s",
           (unsigned long long)ev.header().sequence_id,
           (unsigned long long)ev.header().thread_id,
           (unsigned long long)ev.header().timestamp_ns,
           hrr::event_type_name(ev.header().event_type));

    switch (ev.header().event_type) {
      case HRR_API_HIPMALLOC:
        printf(" handle=0x%llx size=%llu",
               (unsigned long long)ev.malloc_ev.ptr_handle,
               (unsigned long long)ev.malloc_ev.size);
        break;
      case HRR_API_HIPFREE:
        printf(" handle=0x%llx", (unsigned long long)ev.malloc_ev.ptr_handle);
        break;
      case HRR_API_HIPMEMCPY:
      case HRR_API_HIPMEMCPYASYNC:
      case HRR_API_HIPMEMCPYHTOD:
      case HRR_API_HIPMEMCPYHTODASYNC:
        printf(" dst=0x%llx src=0x%llx size=%llu kind=%d",
               (unsigned long long)ev.memcpy_ev.dst_addr,
               (unsigned long long)ev.memcpy_ev.src_addr,
               (unsigned long long)ev.memcpy_ev.size,
               ev.memcpy_ev.kind);
        break;
      case HRR_API_HIPMODULELAUNCHKERNEL:
      case HRR_API_HIPEXTMODULELAUNCHKERNEL:
      case HRR_API_HIPLAUNCHKERNEL:
      case HRR_API_HIPLAUNCHBYPTR:
        if (ev.kernel_launch) {
          const auto& kl = *ev.kernel_launch;
          std::string name = kl.kernel_name;
          if (name.size() > 40) name = name.substr(0, 37) + "...";
          printf(" stream=0x%llx %s [%u,%u,%u]/[%u,%u,%u] args=%zu",
                 (unsigned long long)ev.stream_handle,
                 name.c_str(),
                 kl.grid[0], kl.grid[1], kl.grid[2],
                 kl.block[0], kl.block[1], kl.block[2],
                 kl.args.size());
          // Per-arg detail: print pointer values so a kernel input can be
          // matched to the H2D copy (or prior kernel output) that filled it.
          for (size_t ai = 0; ai < kl.args.size(); ++ai) {
            const auto& arg = kl.args[ai];
            if (arg.value_kind == 1 && arg.data.size() >= 8) {
              uint64_t p;
              memcpy(&p, arg.data.data(), 8);
              printf(" arg[%zu]=ptr:0x%llx", ai, (unsigned long long)p);
            } else if (arg.value_kind == 3) {
              for (uint16_t off : arg.ptr_offsets) {
                if (off + 8u <= arg.data.size()) {
                  uint64_t p;
                  memcpy(&p, arg.data.data() + off, 8);
                  printf(" arg[%zu]+%u=ptr:0x%llx", ai, off,
                         (unsigned long long)p);
                }
              }
            }
          }
        }
        break;
      case HRR_API_HIPSTREAMCREATE:
      case HRR_API_HIPSTREAMCREATEWITHFLAGS:
      case HRR_API_HIPSTREAMCREATEWITHPRIORITY:
        printf(" stream=0x%llx flags=0x%x pri=%d",
               (unsigned long long)ev.stream_create_ev.stream_handle,
               ev.stream_create_ev.flags, ev.stream_create_ev.priority);
        break;
      case HRR_API_HIPMODULELOADDATA:
      case HRR_API_HIPMODULELOADDATAEX:
      case HRR_API_HIPMODULELOAD:
        printf(" mod=0x%llx", (unsigned long long)ev.module_load_ev.module_handle);
        break;
      default:
        if (ev.handle64)
          printf(" handle=0x%llx", (unsigned long long)ev.handle64);
        break;
    }
    printf("\n");
  }
}

// ---------------------------------------------------------------------------
// Special-case events handled outside the dispatch table
// ---------------------------------------------------------------------------

static bool is_special(uint16_t etype) {
  switch (etype) {
    case HRR_API_HIPDEVICESYNCHRONIZE:
    case HRR_API_HIPSTREAMSYNCHRONIZE:
    case HRR_API_HIPMODULEUNLOAD:
    case HRR_API_HIPSETDEVICE:
    case HRR_API_HIPGETLASTERROR:
    case HRR_API_HIPPEEKATLASTERROR:
      return true;
    default:
      return false;
  }
}

static hipError_t handle_special(PlaybackContext& ctx, const hrr::Event& ev) {
  switch (ev.header().event_type) {

    case HRR_API_HIPDEVICESYNCHRONIZE:
      if (!ctx.skip_device_sync) {
        hipError_t r = hrr_watchdog_device_sync(ctx, "recorded hipDeviceSynchronize event");
        if (r != hipSuccess) {
          fprintf(stderr, "[HRR] hipDeviceSynchronize error %d (%s)\n",
                  r, hipGetErrorString(r));
          return r;
        }
      }
      return hipSuccess;

    case HRR_API_HIPSTREAMSYNCHRONIZE:
      if (!ctx.skip_device_sync) {
        hipError_t r = hipStreamSynchronize(ctx.translate_stream(ev.handle64));
        if (r != hipSuccess) {
          fprintf(stderr, "[HRR] hipStreamSynchronize error %d (%s)\n",
                  r, hipGetErrorString(r));
          return r;
        }
      }
      return hipSuccess;

    case HRR_API_HIPMODULEUNLOAD:
      ctx.remove_module(ev.handle64);
      return hipSuccess;

    case HRR_API_HIPSETDEVICE: {
      if (ev.raw_payload.size() >= sizeof(hrr_args_hipSetDevice)) {
        const auto* a = reinterpret_cast<const hrr_args_hipSetDevice*>(ev.raw_payload.data());
        int dev = a->deviceId;
        int n = 0; (void)hipGetDeviceCount(&n);
        if (dev >= n) dev = 0;
        (void)hipSetDevice(dev);
      }
      return hipSuccess;
    }

    case HRR_API_HIPGETLASTERROR:
    case HRR_API_HIPPEEKATLASTERROR:
      return hipSuccess;  // skip silently

    default: return hipSuccess;
  }
}

// ---------------------------------------------------------------------------
// Dispatch one event
// ---------------------------------------------------------------------------

// Events that create or destroy handles written into PlaybackContext maps.
// These must be submitted in global capture order so that handle translations
// are available before any thread that depends on them runs.
// Kernel launches and syncs are excluded — GPU stream ordering handles them.
static bool needs_ordering(uint16_t etype) {
  switch (etype) {
    // Memory alloc / free
    case HRR_API_HIPMALLOC:
    case HRR_API_HIPMALLOCASYNC:
    case HRR_API_HIPMALLOCFROMPOOLASYNC:
    case HRR_API_HIPMALLOCMANAGED:
    case HRR_API_HIPMALLOCHOST:
    case HRR_API_HIPMALLOC3D:
    case HRR_API_HIPMALLOC3DARRAY:
    case HRR_API_HIPMALLOCARRAY:
    case HRR_API_HIPMALLOCMIPMAPPEDARRAY:
    case HRR_API_HIPMALLOCPITCH:
    case HRR_API_HIPFREE:
    case HRR_API_HIPFREEASYNC:
    case HRR_API_HIPFREEARRAY:
    case HRR_API_HIPFREEHOST:
    case HRR_API_HIPFREEMIPMAPPEDARRAY:
    case HRR_API_HIPHOSTFREE:
    case HRR_API_HIPMEMADDRESSFREE:
    case HRR_API_HIPMEMRELEASE:
    // Host/pinned + VMM handle creation. These populate shared maps
    // (alloc_map / host_reg_bufs / vmm_va_map / vmm_handle_map) that later
    // consumers (e.g. hipMemMap, hipHostGetDevicePointer) translate against.
    // They must be ordered so a cross-thread consumer can never run before the
    // create populates the map (their destroy/free counterparts above are
    // already ordered — this restores the symmetry).
    case HRR_API_HIPHOSTREGISTER:
    case HRR_API_HIPHOSTUNREGISTER:
    case HRR_API_HIPHOSTGETDEVICEPOINTER:
    case HRR_API_HIPHOSTMALLOC:
    case HRR_API_HIPMEMADDRESSRESERVE:
    case HRR_API_HIPMEMCREATE:
    // Stream create / destroy
    case HRR_API_HIPSTREAMCREATE:
    case HRR_API_HIPSTREAMCREATEWITHFLAGS:
    case HRR_API_HIPSTREAMCREATEWITHPRIORITY:
    case HRR_API_HIPSTREAMDESTROY:
    // Event create / destroy
    case HRR_API_HIPEVENTCREATE:
    case HRR_API_HIPEVENTCREATEWITHFLAGS:
    case HRR_API_HIPEVENTDESTROY:
    // Module load / unload
    case HRR_API_HIPMODULELOAD:
    case HRR_API_HIPMODULELOADDATA:
    case HRR_API_HIPMODULELOADDATAEX:
    case HRR_API_HIPMODULELOADFATBINARY:
    case HRR_API_HIPMODULEUNLOAD:
    case HRR_API_HIPREGISTERFATBINARY:
    // Graph / graph-exec create
    case HRR_API_HIPSTREAMBEGINCAPTURE:
    case HRR_API_HIPSTREAMENDCAPTURE:
    case HRR_API_HIPGRAPHINSTANTIATE:
    case HRR_API_HIPGRAPHINSTANTIATEWITHFLAGS:
    case HRR_API_HIPGRAPHINSTANTIATEWITHPARAMS:
    case HRR_API_HIPGRAPHEXECDESTROY:
    case HRR_API_HIPGRAPHDESTROY:
    case HRR_API_HIPLINKDESTROY:
    // MemPool create / destroy
    case HRR_API_HIPMEMPOOLCREATE:
    case HRR_API_HIPMEMPOOLDESTROY:
    // Array / mipmapped array create / destroy
    case HRR_API_HIPARRAY3DCREATE:
    case HRR_API_HIPARRAYCREATE:
    case HRR_API_HIPARRAYDESTROY:
    case HRR_API_HIPMIPMAPPEDARRAYCREATE:
    case HRR_API_HIPMIPMAPPEDARRAYDESTROY:
    // Texture / surface object create / destroy
    case HRR_API_HIPCREATETEXTUREOBJECT:
    case HRR_API_HIPCREATESURFACEOBJECT:
    case HRR_API_HIPDESTROYSURFACEOBJECT:
    case HRR_API_HIPDESTROYTEXTUREOBJECT:
    case HRR_API_HIPTEXOBJECTDESTROY:
      return true;
    default:
      return false;
  }
}

// Returns hipSuccess, or the first non-success error encountered.
// On any error, ctx.fatal_error is set to true so all replay threads stop.
static hipError_t dispatch_event(PlaybackContext& ctx, const hrr::Event& ev,
                                 size_t idx, bool log) {
  uint16_t etype = ev.header().event_type;

  // Give kernel-launch handlers the sequence ID so they can wait and advance
  // next_seq at the exact point of the HIP call.
  hrr_dispatch_seq = ev.header().sequence_id;
  auto order = needs_ordering(etype);

  // Spin-wait for our turn in global capture order.
  // Also check fatal_error: if another thread failed and advanced next_seq
  // past our slot, we must not spin forever — bail out immediately.
  // After ~1000 failed yields, sleep for 1 µs to avoid burning a full core
  // while waiting for a slow ordering predecessor.
  {
    int spin = 0;
    while (ctx.next_seq.load(std::memory_order_acquire) != ev.header().sequence_id) {
      if (ctx.fatal_error.load(std::memory_order_acquire))
        return hipErrorUnknown;
      if (++spin < 1000)
        std::this_thread::yield();
      else
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
  }

  // If a fatal error was already flagged by another thread that ran before us,
  // advance next_seq so the thread waiting on the next event can also exit,
  // then bail out without executing this event.
  if (ctx.fatal_error.load(std::memory_order_acquire)) {
    ctx.next_seq.store(ev.header().sequence_id + 1, std::memory_order_release);
    return hipErrorUnknown;
  }

  // RAII guard: non-ordering events advance immediately (constructor) so the
  // next thread can proceed while this call is still in-flight; ordered events
  // advance on scope exit so the next thread is unblocked on every return path.
  struct SeqAdvance {
    PlaybackContext& ctx;
    uint64_t next;
    bool order;
    SeqAdvance(PlaybackContext& c, uint64_t n, bool o) : ctx(c), next(n), order(o) {
      if (!order)
        ctx.next_seq.store(next, std::memory_order_release);
    }
    ~SeqAdvance() {
      if (order)
        ctx.next_seq.store(next, std::memory_order_release);
    }
  } seq_guard{ctx, ev.header().sequence_id + 1, order};

  if (is_special(etype)) {
    hipError_t r = handle_special(ctx, ev);
    if (r != hipSuccess) {
      ctx.fatal_error.store(true, std::memory_order_release);
      fprintf(stderr, "[HRR] Fatal: aborting replay after error in %s\n",
              hrr::event_type_name(etype));
    }
    return r;
  }

  if (etype >= HRR_API_COUNT || !hrr_playback_dispatch[etype]) {
    if (log && ctx.verbose)
      fprintf(stderr, "[HRR] T%llu Event %zu: no handler for type %u\n",
              (unsigned long long)ev.header().thread_id, idx, etype);
    return hipSuccess;
  }

  // Guard: the reader's SIZE_OK macro skips typed decode for undersized payloads
  // but still retains the event in the archive.  Validate against the per-type
  // minimum size (sizeof(hrr_args_*)) so handlers can safely cast raw_payload
  // without reading past the end on malformed archives.
  const uint32_t min_size = (etype < HRR_API_COUNT)
                              ? hrr_api_min_payload_size[etype]
                              : static_cast<uint32_t>(sizeof(hrr_event_header));
  if (ev.raw_payload.size() < min_size) {
    if (log)
      fprintf(stderr, "[HRR] T%llu Event %zu (%s): payload too small (%zu < %u bytes) — skipping\n",
              (unsigned long long)ev.header().thread_id, idx,
              hrr::event_type_name(etype), ev.raw_payload.size(), min_size);
    return hipSuccess;
  }

  hipError_t r = hrr_playback_dispatch[etype](ctx, ev.raw_payload.data());

  if (r != hipSuccess) {
    ctx.fatal_error.store(true, std::memory_order_release);
    if (log)
      fprintf(stderr, "[HRR] Fatal: T%llu Event %zu (%s) returned %d (%s) — aborting replay\n",
              (unsigned long long)ev.header().thread_id, idx,
              hrr::event_type_name(etype), r, hipGetErrorString(r));
    return r;
  }

  // --sync-after-event: flush the GPU after every dispatched event and check
  // for async errors. This makes GPU faults show up at the exact causal event
  // rather than surfacing later on a sync or the next API call.
  if (ctx.sync_after_event) {
    hipError_t se = hrr_watchdog_device_sync(ctx, "sync-after-event");
    if (se == hipSuccess) se = hipGetLastError();
    if (se != hipSuccess) {
      ctx.fatal_error.store(true, std::memory_order_release);
      if (log)
        fprintf(stderr,
                "[HRR] Fatal: GPU error after T%llu Event %zu (%s): %d (%s) — aborting\n",
                (unsigned long long)ev.header().thread_id, idx,
                hrr::event_type_name(etype), se, hipGetErrorString(se));
      return se;
    }
  }

  return r;
}

// ---------------------------------------------------------------------------
// Per-thread replay worker (MT path)
// ---------------------------------------------------------------------------

static void replay_thread(PlaybackContext& ctx,
                          const std::vector<const hrr::Event*>& events,
                          bool log) {
  for (size_t i = 0; i < events.size(); i++) {
    if (ctx.fatal_error.load(std::memory_order_acquire))
      return;
    const hrr::Event& ev = *events[i];
    if (log && ctx.verbose)
      fprintf(stderr, "[HRR] T%llu [%zu] %s\n",
              (unsigned long long)ev.header().thread_id, i,
              hrr::event_type_name(ev.header().event_type));
    (void)dispatch_event(ctx, ev, i, log);
  }
}

// ---------------------------------------------------------------------------
// Pre-pass device state snapshot
// ---------------------------------------------------------------------------
// Clears any pending GPU error, syncs the device, and logs available memory.
// Called before every replay pass so we always start from a known-clean state.
// Returns false if the device is already in an error state before we begin.

static bool save_device_state(const char* label) {
  // Drain any error left over from previous operations
  hipError_t pending = hipGetLastError();
  if (pending != hipSuccess)
    fprintf(stderr, "[HRR] [%s] WARNING: pending GPU error cleared: %d (%s)\n",
            label, pending, hipGetErrorString(pending));

  // Sync to ensure all prior GPU work has completed
  hipError_t sync_err = hipDeviceSynchronize();
  if (sync_err != hipSuccess) {
    fprintf(stderr, "[HRR] [%s] FATAL: hipDeviceSynchronize failed: %d (%s)\n",
            label, sync_err, hipGetErrorString(sync_err));
    return false;
  }

  // Check again after sync (async errors surface here)
  hipError_t post_sync = hipGetLastError();
  if (post_sync != hipSuccess) {
    fprintf(stderr, "[HRR] [%s] FATAL: GPU error after sync: %d (%s)\n",
            label, post_sync, hipGetErrorString(post_sync));
    return false;
  }

  // Log available device memory
  size_t free_bytes = 0, total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
    printf("[HRR] [%s] Device memory: %zu MB free / %zu MB total\n",
           label,
           free_bytes  / (1024 * 1024),
           total_bytes / (1024 * 1024));
  }

  printf("[HRR] [%s] Device state OK — beginning replay\n", label);
  return true;
}

// ---------------------------------------------------------------------------
// One full pass through the archive (single- or multi-threaded)
// ---------------------------------------------------------------------------

// Returns false if a fatal HIP error occurred during the pass.
static bool run_pass(PlaybackContext& ctx,
                     const hrr::Archive& archive,
                     const std::unordered_map<uint64_t,
                           std::vector<const hrr::Event*>>& thread_events,
                     bool use_mt, bool log) {
  // Pre-load all code objects before any GPU work begins.
  // hipModuleLoadData() triggers internal driver state checks that can surface
  // deferred GPU faults from prior async kernel launches, even after
  // hipDeviceSynchronize() returns success (WDDM-specific behavior on Windows).
  // Loading all code objects up front ensures no code object load happens after
  // any kernel launch, preventing error 719 from corrupting module loads.
  if (log) printf("[HRR] Pre-loading %zu code objects...\n", archive.code_objects.size());
  for (const auto& [hex, fpath] : archive.code_objects) {
      // hex is "llllllllllllllllhhhhhhhhhhhhhhhh" (lo first, hi second)
      uint64_t lo = 0, hi = 0;
      if (hex.size() == 32) {
          lo = strtoull(hex.substr(0,16).c_str(), nullptr, 16);
          hi = strtoull(hex.substr(16).c_str(), nullptr, 16);
      }
      hipModule_t mod = ctx.load_module(lo, hi);
      if (!mod) {
          fprintf(stderr, "[HRR] Fatal: failed to pre-load code object %s — aborting\n",
                  hex.c_str());
          return false;
      }
  }
  if (log) printf("[HRR] All code objects pre-loaded OK.\n");

  // Reset the sequence counter to the first event's seq_id before every pass
  // so MT threads start waiting at the right value, and ST kernel-filter
  // passes don't hang after a warm-up pass left next_seq at the end.
  if (!archive.events.empty())
    ctx.next_seq.store(archive.events.front().header().sequence_id,
                       std::memory_order_relaxed);

  if (use_mt) {

    std::vector<std::thread> threads;
    threads.reserve(archive.threads.size());
    for (uint64_t tid : archive.threads) {
      const auto& evlist = thread_events.at(tid);
      threads.emplace_back(replay_thread, std::ref(ctx),
                           std::cref(evlist), log);
    }
    for (auto& t : threads) t.join();
  } else {
    for (size_t i = 0; i < archive.events.size(); i++) {
      if (ctx.fatal_error.load(std::memory_order_acquire))
        break;
      const auto& ev = archive.events[i];
      if (log && ctx.verbose)
        fprintf(stderr, "[HRR] Event %zu: %s\n", i,
                hrr::event_type_name(ev.header().event_type));
      (void)dispatch_event(ctx, ev, i, log);
    }
  }

  if (ctx.fatal_error.load(std::memory_order_acquire))
    return false;

  hipError_t r = hrr_watchdog_device_sync(ctx, "end-of-pass device synchronize");
  if (r != hipSuccess) {
    fprintf(stderr, "[HRR] Fatal: hipDeviceSynchronize after pass failed: %d (%s)\n",
            r, hipGetErrorString(r));
    ctx.fatal_error.store(true, std::memory_order_release);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// --repair mode: rewrite a crash-truncated archive into a clean one.
//
// Loads the archive with the tolerant reader (recovering all complete records),
// then rewrites events.bin trimmed to the last complete record, appends the
// clean-shutdown trailer, and writes a complete manifest.json. After repair the
// archive looks exactly like one produced by a normal process exit. Replay
// works on truncated archives without repair too; this just "blesses" them.
// ---------------------------------------------------------------------------

static bool write_u(FILE* f, const void* p, size_t n) {
  return fwrite(p, 1, n, f) == n;
}

static uint64_t pid_from_archive_dir(const fs::path& archive_dir) {
  const std::string name = archive_dir.filename().string();
  if (name.rfind("pid-", 0) != 0) return 0;

  char* end = nullptr;
  unsigned long long pid = std::strtoull(name.c_str() + 4, &end, 10);
  if (!end || *end != '\0') return 0;
  return static_cast<uint64_t>(pid);
}

static int repair_archive(const hrr::Archive& archive) {
  std::string events_path = archive.path + "/events.bin";
  std::string tmp_path    = events_path + ".repair.tmp";

  if (archive.complete) {
    printf("[HRR] Archive already complete (%zu events); nothing to repair\n",
           archive.event_count);
    return 0;
  }

  FILE* out = fopen(tmp_path.c_str(), "wb");
  if (!out) {
    fprintf(stderr, "[HRR] repair: cannot open %s for writing\n", tmp_path.c_str());
    return 1;
  }

  hrr_file_header fh{HRR_MAGIC, HRR_VERSION, 0};
  bool ok = write_u(out, &fh, sizeof(fh));

  uint64_t max_seq = 0;
  for (const auto& ev : archive.events) {
    if (!ok) break;
    // load_archive only ever stores complete records, so raw_payload always
    // holds at least a full header. Assert it before ev.header() casts the
    // bytes, so a future reader bug surfaces here instead of as silent UB.
    assert(ev.raw_payload.size() >= sizeof(hrr_event_header) &&
           "repair: event raw_payload smaller than hrr_event_header");
    ok = write_u(out, ev.raw_payload.data(), ev.raw_payload.size());
    if (ev.header().sequence_id > max_seq) max_seq = ev.header().sequence_id;
  }

  // Clean-shutdown trailer. Built via the shared hrr_make_eof_record() helper so
  // the trailer layout cannot drift from the capture writer (writer::flush()).
  // The offline tool leaves timestamp_ns / thread_id at 0.
  if (ok) {
    hrr_eof_record rec = hrr_make_eof_record(max_seq + 1, archive.events.size());
    ok = write_u(out, &rec, sizeof(rec));
  }
  fflush(out);
  fclose(out);

  if (!ok) {
    fprintf(stderr, "[HRR] repair: write failed; leaving original untouched\n");
    remove(tmp_path.c_str());
    return 1;
  }

  // Atomically replace the original events.bin. POSIX rename() overwrites an
  // existing destination, but Windows rename()/MoveFile without
  // MOVEFILE_REPLACE_EXISTING fails when the target already exists. Use
  // std::filesystem::rename (which maps to a replacing move on Windows) and, if
  // that still fails, fall back to removing the destination first.
  std::error_code ec;
  fs::rename(tmp_path, events_path, ec);
  if (ec) {
    fs::remove(events_path, ec);
    fs::rename(tmp_path, events_path, ec);
  }
  if (ec) {
    fprintf(stderr, "[HRR] repair: cannot replace %s: %s\n",
            events_path.c_str(), ec.message().c_str());
    remove(tmp_path.c_str());
    return 1;
  }

  fs::path archive_dir(archive.path);
  std::string manifest_path = (archive_dir / "manifest.json").string();
  ProcessInfo info{};
  if (!read_process_manifest(manifest_path, info))
    info.pid = pid_from_archive_dir(archive_dir);

  FILE* mf = fopen(manifest_path.c_str(), "w");
  if (mf) {
    fprintf(mf,
            "{\n"
            "  \"pid\": %llu,\n"
            "  \"parent_pid\": %llu,\n"
            "  \"complete\": true,\n"
            "  \"event_count\": %zu,\n"
            "  \"blob_count\": %zu\n"
            "}\n",
            static_cast<unsigned long long>(info.pid),
            static_cast<unsigned long long>(info.parent_pid),
            archive.events.size(), archive.blob_count);
    fclose(mf);
  }

  printf("[HRR] Repaired archive: %zu events kept, clean trailer + manifest written\n",
         archive.events.size());
  return 0;
}

static void print_usage(const char* argv0) {
  fprintf(stderr,
    "Usage: %s <capture.hrr> [options]\n"
    "\n"
    "Options:\n"
    "  --info                Print archive summary and exit (no GPU required)\n"
    "  --repair              Rewrite a crash-truncated archive as a clean one and exit\n"
    "  --events              With --info: also print the full event log\n"
    "  --verbose             Print each event as it is processed\n"
    "  --skip-device-sync    Skip device/stream synchronize events\n"
    "  --multi-thread        Enable multi-threaded replay (default: single-threaded)\n"
    "  --timing              Report wall time and total GPU kernel time\n"
    "  --kernel-filter STR   Only launch kernels whose name contains STR\n"
    "                        (silent full warm-up pass runs first)\n"
    "  --replace-kernel N=P  Launch the kernel whose recorded name is exactly N\n"
    "                        from code object P (.hsaco) instead of the recorded\n"
    "                        one. N must be the full recorded symbol (see --info;\n"
    "                        mangled for C++/chevron kernels). Repeatable for\n"
    "                        multiple kernels. Recorded args/grid/block are reused;\n"
    "                        the archive is not modified.\n"
    "  --sync-after-launch   hipDeviceSynchronize after every kernel launch\n"
    "  --sync-after-event    hipDeviceSynchronize after EVERY event (slowest, most precise)\n"
    "  --sync-watchdog-ms N  Abort with a diagnostic if any device synchronize does\n"
    "                        not complete within N ms (catches hung/deadlocked\n"
    "                        kernels, e.g. StreamK flag spin-waits). 0 = disabled.\n"
    "  --trace-kernels       Print one compact line before every kernel launch\n"
    "  --trace-sync          Print sync begin/done markers around kernel syncs\n"
    "  --progress-kernels N  Print heartbeat every N launched kernels\n"
    "  --progress-seconds S  Print heartbeat at most every S seconds\n"
    "  --help                Show this message\n"
    "\n"
    "Environment (replay device allocation padding — see hip_playback.cpp):\n"
    "  HIP_HRR_REPLAY_ALLOC_PAD_FACTOR   unsigned, default 1 (exact recorded sizes).\n"
    "                                    Use 256 for legacy pool-style headroom (more VRAM).\n"
    "  HIP_HRR_REPLAY_ALLOC_PAD_MAX      bytes, default 1073741824 (1 GiB cap per alloc).\n"
    "\n"
    "Environment (lightweight replay tracing):\n"
    "  HIP_HRR_REPLAY_TRACE_KERNELS=1       same as --trace-kernels\n"
    "  HIP_HRR_REPLAY_TRACE_SYNC=1          same as --trace-sync\n"
    "  HIP_HRR_REPLAY_PROGRESS_KERNELS=N    same as --progress-kernels N\n"
    "  HIP_HRR_REPLAY_PROGRESS_SECONDS=S    same as --progress-seconds S\n"
    "  HIP_HRR_REPLAY_SYNC_WATCHDOG_MS=N    same as --sync-watchdog-ms N\n"
    "\n"
    "Default mode: single-threaded, serialize GPU after pass, abort on first error.\n"
    "Use --sync-after-event to pinpoint the exact event causing a GPU fault or hang.\n",
    argv0);
}

int main(int argc, char** argv) {
  if (argc < 2) { print_usage(argv[0]); return 1; }

  std::string archive_path;
  PlaybackContext ctx;
  ctx.validate_d2h = true;
  bool single_thread = true;   // default: single-threaded for safety; use --multi-thread to opt in
  bool show_info     = false;
  bool show_events   = false;
  bool do_repair     = false;

  for (int i = 1; i < argc; i++) {
    if      (!strcmp(argv[i], "--info"))              show_info              = true;
    else if (!strcmp(argv[i], "--repair"))            do_repair              = true;
    else if (!strcmp(argv[i], "--events"))            show_events            = true;
    else if (!strcmp(argv[i], "--verbose"))           ctx.verbose            = true;
    else if (!strcmp(argv[i], "--skip-device-sync"))  ctx.skip_device_sync   = true;
    else if (!strcmp(argv[i], "--multi-thread"))      single_thread          = false;
    else if (!strcmp(argv[i], "--timing"))            ctx.timing             = true;
    else if (!strcmp(argv[i], "--sync-after-launch")) ctx.sync_after_launch  = true;
    else if (!strcmp(argv[i], "--sync-after-event"))  ctx.sync_after_event   = true;
    else if (!strcmp(argv[i], "--sync-watchdog-ms") && i + 1 < argc) {
      char* end = nullptr;
      unsigned long long n = std::strtoull(argv[++i], &end, 10);
      if (!end || *end != '\0') {
        fprintf(stderr, "[HRR] --sync-watchdog-ms expects an integer (milliseconds)\n");
        return 1;
      }
      ctx.sync_watchdog_ms = static_cast<unsigned>(n);
    }
    else if (!strcmp(argv[i], "--trace-kernels"))     ctx.trace_kernels      = true;
    else if (!strcmp(argv[i], "--trace-sync"))        ctx.trace_sync         = true;
    else if (!strcmp(argv[i], "--progress-kernels") && i + 1 < argc) {
      char* end = nullptr;
      unsigned long long n = std::strtoull(argv[++i], &end, 10);
      if (!end || *end != '\0') {
        fprintf(stderr, "[HRR] --progress-kernels expects an integer\n");
        return 1;
      }
      ctx.progress_kernel_interval = static_cast<size_t>(n);
    }
    else if (!strcmp(argv[i], "--progress-seconds") && i + 1 < argc) {
      char* end = nullptr;
      double n = std::strtod(argv[++i], &end);
      if (!end || *end != '\0') {
        fprintf(stderr, "[HRR] --progress-seconds expects a number\n");
        return 1;
      }
      ctx.progress_seconds_interval = n;
    }
    else if (!strcmp(argv[i], "--kernel-filter") && i + 1 < argc)
      ctx.kernel_filter = argv[++i];
    else if (!strcmp(argv[i], "--replace-kernel") && i + 1 < argc) {
      // Split "NAME=path" on the LAST '=' so paths containing '=' (or a
      // Windows drive letter) stay intact. Both sides must be non-empty.
      std::string spec = argv[++i];
      size_t eq = spec.rfind('=');
      if (eq == std::string::npos || eq == 0 || eq + 1 >= spec.size()) {
        fprintf(stderr, "[HRR] --replace-kernel expects NAME=PATH, got '%s'\n",
                spec.c_str());
        print_usage(argv[0]);
        return 1;
      }
      ctx.kernel_replacements.emplace_back(spec.substr(0, eq), spec.substr(eq + 1));
    }
    else if (!strcmp(argv[i], "--help")) { print_usage(argv[0]); return 0; }
    else if (argv[i][0] != '-') archive_path = argv[i];
  }

  ctx.trace_kernels |= env_flag_enabled("HIP_HRR_REPLAY_TRACE_KERNELS");
  ctx.trace_sync |= env_flag_enabled("HIP_HRR_REPLAY_TRACE_SYNC");
  ctx.progress_kernel_interval =
      env_size_or("HIP_HRR_REPLAY_PROGRESS_KERNELS", ctx.progress_kernel_interval);
  ctx.progress_seconds_interval =
      env_double_or("HIP_HRR_REPLAY_PROGRESS_SECONDS", ctx.progress_seconds_interval);
  ctx.sync_watchdog_ms = static_cast<unsigned>(
      env_size_or("HIP_HRR_REPLAY_SYNC_WATCHDOG_MS", ctx.sync_watchdog_ms));
  ctx.dump_ptrs_ordinal =
      env_size_or("HIP_HRR_REPLAY_DUMP_PTRS_ORDINAL", ctx.dump_ptrs_ordinal);

  if (archive_path.empty()) {
    fprintf(stderr, "[HRR] No archive path specified\n");
    print_usage(argv[0]);
    return 1;
  }

  // --info on an archive root prints the common process summary. Direct
  // pid-<pid> paths continue through load_archive for detailed event info.
  if (show_info && print_root_info(archive_path))
    return 0;

  hrr::Archive archive;
  if (!hrr::load_archive(archive_path, archive)) return 1;

  // --info: print summary and exit without touching the GPU
  if (show_info) {
    print_info(archive, show_events);
    return 0;
  }

  // --repair: rewrite a crash-truncated archive as a clean one and exit
  if (do_repair) {
    return repair_archive(archive);
  }

  ctx.archive_dir = archive.path;

  printf("[HRR] Archive : %zu events, %zu kernels, %zu blobs, %zu code objects\n",
         archive.event_count, archive.kernel_count,
         archive.blob_count, archive.code_object_count);
  printf("[HRR] Threads : %zu captured\n", archive.threads.size());

  HIP_CHECK(hipInit(0));

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) { fprintf(stderr, "[HRR] No GPU devices found\n"); return 1; }

  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  printf("[HRR] Device  : %s (%s)\n", props.name, props.gcnArchName);

  // Partition events by thread_id — O(n), no re-scan needed at replay time
  std::unordered_map<uint64_t, std::vector<const hrr::Event*>> thread_events;
  for (uint64_t tid : archive.threads) thread_events[tid];
  for (const auto& ev : archive.events)
    thread_events[ev.header().thread_id].push_back(&ev);

  const bool use_mt = !single_thread && archive.threads.size() > 1;
  printf("[HRR] Mode    : %s\n", use_mt ? "multi-threaded" : "single-threaded");

  // Module pre-pass: process fat binary and explicit module load events in
  // global sequence order, single-threaded, before the parallel replay begins.
  // Without this, a timing delay on one thread (e.g. hipEventSynchronize) can
  // cause another thread's kernel launch to race ahead of the module load that
  // populates module_map, resulting in "kernel not found" errors.
  //
  // dispatch_event() spin-waits on ctx.next_seq matching the event's sequence_id.
  // The pre-pass skips non-module events, so next_seq would never naturally
  // reach a module event's sequence_id — causing a deadlock. Fix: advance
  // next_seq to each event's sequence_id immediately before dispatching it so
  // the spin-wait passes unconditionally in this single-threaded context.
  if (use_mt) {
    for (const auto& ev : archive.events) {
      uint16_t t = ev.header().event_type;
      if (t == HRR_API_HIPREGISTERFATBINARY ||
          t == HRR_API_HIPMODULELOADDATA    ||
          t == HRR_API_HIPMODULELOADDATAEX  ||
          t == HRR_API_HIPMODULELOAD) {
        ctx.next_seq.store(ev.header().sequence_id, std::memory_order_release);
        if (dispatch_event(ctx, ev, 0, /*log=*/true) != hipSuccess) {
          fprintf(stderr, "[HRR] Fatal error in module pre-pass — aborting\n");
          return 1;
        }
      }
    }
    // After the pre-pass, reset next_seq to 0 so the main MT replay starts
    // from the beginning of the sequence ordering.
    ctx.next_seq.store(0, std::memory_order_release);
    ctx.fatal_error.store(false, std::memory_order_release);
  }

  // Warm-up pass: when a kernel filter is active, run the full archive once
  // without the filter so all GPU buffers (including intermediate kernel
  // outputs) are correctly populated before the timed filtered pass.
  if (!ctx.kernel_filter.empty()) {
    printf("[HRR] Warm-up : full pass to populate GPU state...\n");
    if (!save_device_state("warm-up")) return 1;
    std::string filter = ctx.kernel_filter;
    ctx.kernel_filter.clear();
    if (!run_pass(ctx, archive, thread_events, use_mt, /*log=*/false)) {
      fprintf(stderr, "[HRR] Fatal error in warm-up pass — aborting\n");
      return 1;
    }
    ctx.fatal_error.store(false, std::memory_order_release);
    ctx.kernel_filter    = filter;
    ctx.kernels_launched = 0;
    ctx.total_kernel_ms  = 0.0;
    printf("[HRR] Warm-up done. Running filtered pass...\n");
  }

  // Timed replay pass
  if (!save_device_state("replay")) return 1;
  auto wall_start = std::chrono::high_resolution_clock::now();
  bool pass_ok = run_pass(ctx, archive, thread_events, use_mt, /*log=*/true);
  auto wall_end = std::chrono::high_resolution_clock::now();
  double wall_ms = std::chrono::duration<double, std::milli>(
                       wall_end - wall_start).count();

  if (!pass_ok) {
    if (ctx.diverged.load(std::memory_order_acquire)) {
      fprintf(stderr,
              "[HRR] Replay stopped: data divergence exceeded threshold "
              "(%zu/%zu D2H validations failed). This is a replay-fidelity "
              "limit (nondeterministic GPU reductions in an unstable workload), "
              "not an HRR bug — see the [HRR] replay DIVERGED message above.\n",
              ctx.d2h_fail.load(), ctx.d2h_attempted.load());
      return 2;
    }
    fprintf(stderr, "[HRR] Replay aborted due to fatal HIP error — exiting\n");
    return 1;
  }

  // ---------------------------------------------------------------------------
  // Summary
  // ---------------------------------------------------------------------------
  printf("\n");
  printf("[HRR] -- Replay summary ------------------------------\n");
  printf("[HRR]   Wall time      : %.1f ms\n", wall_ms);
  printf("[HRR]   Threads used   : %zu\n", use_mt ? archive.threads.size() : (size_t)1);
  printf("[HRR]   Kernels launched: %zu\n", ctx.kernels_launched.load());
  printf("[HRR]   Graphs launched : %zu\n", ctx.graphs_launched.load());
  if (ctx.timing) {
    printf("[HRR]   GPU kernel time : %.1f ms\n", ctx.total_kernel_ms);
    printf("[HRR]   GPU graph time  : %.1f ms\n", ctx.total_graph_ms);
    printf("[HRR]   GPU total time  : %.1f ms\n", ctx.total_kernel_ms + ctx.total_graph_ms);
  }
  printf("[HRR]   D2H checks     : %zu pass (%zu exact, %zu within tol), %zu fail, %zu skipped\n",
         ctx.d2h_pass.load(),
         ctx.d2h_pass.load() - ctx.d2h_pass_tol.load(), ctx.d2h_pass_tol.load(),
         ctx.d2h_fail.load(),
         ctx.d2h_attempted.load() - ctx.d2h_pass.load() - ctx.d2h_fail.load());

  bool ok = (ctx.d2h_fail == 0);
  if (ctx.d2h_pass == 0 && ctx.d2h_fail == 0) {
    if (ctx.d2h_attempted > 0) {
      // D2H events with captured blobs existed, but every validation was skipped
      // (source pointer translation failed or expected blob was missing for all).
      // This indicates a pointer-translation bug or archive corruption.
      fprintf(stderr,
              "[HRR] ERROR: %zu D2H validation event(s) attempted but 0 pass/fail recorded "
              "(all skipped — check pointer translation and blob files)\n",
              ctx.d2h_attempted.load());
      ok = false;
    } else {
      printf("[HRR]   (no D2H validation blobs in archive -- re-capture to enable)\n");
    }
  }

  printf("[HRR] %s\n", ok ? "PASS" : "FAIL");

  // Cleanup.
  for (auto& [rec, gexec] : ctx.graph_exec_map) (void)hipGraphExecDestroy(gexec);
  for (auto& [rec, graph] : ctx.graph_map)      (void)hipGraphDestroy(graph);

  // alloc_map mixes device and host allocations; each must be released with the
  // matching API. Dispatch on AllocEntry::kind:
  //   Device         -> hipFree
  //   HostMalloc     -> hipHostFree
  //   HostRegister   -> released below via host_reg_bufs (hipHostUnregister+free)
  //   DevicePtrAlias -> not separately freed (alias into a pinned host alloc)
  for (auto& [rec, entry] : ctx.alloc_map) {
    switch (entry.kind) {
      case AllocKind::Device:        (void)hipFree(entry.live_ptr);     break;
      case AllocKind::HostMalloc:    (void)hipHostFree(entry.live_ptr); break;
      case AllocKind::HostRegister:                                     break;
      case AllocKind::DevicePtrAlias:                                   break;
    }
  }
  // Captures routinely end mid-stream with hipHostRegister'd buffers still live;
  // playback_hipHostUnregister never ran for them, so unregister + free each
  // remaining backing buffer here to avoid leaking both the pinned registration
  // and the malloc'd buffer every run.
  for (auto& [rec, buf] : ctx.host_reg_bufs) {
    if (!buf) continue;
    (void)hipHostUnregister(buf);
#ifdef _WIN32
    _aligned_free(buf);
#else
    free(buf);
#endif
  }
  ctx.host_reg_bufs.clear();
  for (auto& [rec, str]   : ctx.stream_map)  (void)hipStreamDestroy(str);
  for (auto& [rec, ev2]   : ctx.event_map)   (void)hipEventDestroy(ev2);
  for (hipEvent_t e : ctx.owned_timing_events) (void)hipEventDestroy(e);

  // Unload all unique hipModule_t values across both maps.
  // co_modules holds modules loaded from code objects (by hash).
  // module_map holds fat-binary modules (not in co_modules) plus
  // duplicates of modules from explicit hipModuleLoad calls (already in
  // co_modules). The set deduplicates so each module is unloaded once.
  {
    std::unordered_set<hipModule_t> mods;
    for (auto& [hex, mod] : ctx.co_modules) mods.insert(mod);
    for (auto& [rec, mod] : ctx.module_map) mods.insert(mod);
    for (hipModule_t m : ctx.replacement_modules) mods.insert(m);
    for (hipModule_t m : mods) (void)hipModuleUnload(m);
  }

  return ok ? 0 : 1;
}
