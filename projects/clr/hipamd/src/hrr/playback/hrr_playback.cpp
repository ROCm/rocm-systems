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
//   --sync-after-launch   hipDeviceSynchronize() after every kernel (debug)
//   --help                Show this message
//
// Exit code: 0 = all D2H checks passed (or none present), 1 = any failure.

#include "hrr_reader.h"
#include "hip_playback.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define HIP_CHECK(call)                                                       \
  do {                                                                        \
    hipError_t _e = (call);                                                   \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "[HRR] HIP error %d (%s) at %s:%d\n",                  \
              _e, hipGetErrorString(_e), __FILE__, __LINE__);                 \
      return 1;                                                                \
    }                                                                         \
  } while (0)

// ---------------------------------------------------------------------------
// --info mode: print archive summary without touching the GPU
// ---------------------------------------------------------------------------

static void print_info(const hrr::Archive& archive, bool show_events) {
  printf("HRR Archive: %s\n", archive.path.c_str());
  printf("========================================\n");
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
        hipError_t r = hipDeviceSynchronize();
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
    hipError_t se = hipDeviceSynchronize();
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

  hipError_t r = hipDeviceSynchronize();
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

static void print_usage(const char* argv0) {
  fprintf(stderr,
    "Usage: %s <capture.hrr> [options]\n"
    "\n"
    "Options:\n"
    "  --info                Print archive summary and exit (no GPU required)\n"
    "  --events              With --info: also print the full event log\n"
    "  --verbose             Print each event as it is processed\n"
    "  --skip-device-sync    Skip device/stream synchronize events\n"
    "  --multi-thread        Enable multi-threaded replay (default: single-threaded)\n"
    "  --timing              Report wall time and total GPU kernel time\n"
    "  --kernel-filter STR   Only launch kernels whose name contains STR\n"
    "                        (silent full warm-up pass runs first)\n"
    "  --sync-after-launch   hipDeviceSynchronize after every kernel launch\n"
    "  --sync-after-event    hipDeviceSynchronize after EVERY event (slowest, most precise)\n"
    "  --help                Show this message\n"
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

  for (int i = 1; i < argc; i++) {
    if      (!strcmp(argv[i], "--info"))              show_info              = true;
    else if (!strcmp(argv[i], "--events"))            show_events            = true;
    else if (!strcmp(argv[i], "--verbose"))           ctx.verbose            = true;
    else if (!strcmp(argv[i], "--skip-device-sync"))  ctx.skip_device_sync   = true;
    else if (!strcmp(argv[i], "--multi-thread"))      single_thread          = false;
    else if (!strcmp(argv[i], "--timing"))            ctx.timing             = true;
    else if (!strcmp(argv[i], "--sync-after-launch")) ctx.sync_after_launch  = true;
    else if (!strcmp(argv[i], "--sync-after-event"))  ctx.sync_after_event   = true;
    else if (!strcmp(argv[i], "--kernel-filter") && i + 1 < argc)
      ctx.kernel_filter = argv[++i];
    else if (!strcmp(argv[i], "--help")) { print_usage(argv[0]); return 0; }
    else if (argv[i][0] != '-') archive_path = argv[i];
  }

  if (archive_path.empty()) {
    fprintf(stderr, "[HRR] No archive path specified\n");
    print_usage(argv[0]);
    return 1;
  }

  hrr::Archive archive;
  if (!hrr::load_archive(archive_path, archive)) return 1;

  // --info: print summary and exit without touching the GPU
  if (show_info) {
    print_info(archive, show_events);
    return 0;
  }

  ctx.archive_dir = archive_path;

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
  printf("[HRR]   D2H checks     : %zu pass, %zu fail, %zu skipped\n",
         ctx.d2h_pass.load(), ctx.d2h_fail.load(),
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

  // Cleanup
  for (auto& [rec, gexec] : ctx.graph_exec_map) (void)hipGraphExecDestroy(gexec);
  for (auto& [rec, graph] : ctx.graph_map)      (void)hipGraphDestroy(graph);
  for (auto& [rec, entry] : ctx.alloc_map)   (void)hipFree(entry.live_ptr);
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
    for (hipModule_t m : mods) (void)hipModuleUnload(m);
  }

  return ok ? 0 : 1;
}
