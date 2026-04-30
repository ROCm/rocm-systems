/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

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
    type_counts[ev.header.event_type]++;

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
      if (ev.header.event_type != HRR_API_HIPMODULELAUNCHKERNEL || !ev.kernel_launch)
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
           (unsigned long long)ev.header.sequence_id,
           (unsigned long long)ev.header.thread_id,
           (unsigned long long)ev.header.timestamp_ns,
           hrr::event_type_name(ev.header.event_type));

    switch (ev.header.event_type) {
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

static void handle_special(PlaybackContext& ctx, const hrr::Event& ev) {
  switch (ev.header.event_type) {

    case HRR_API_HIPDEVICESYNCHRONIZE:
      if (!ctx.skip_device_sync) hipDeviceSynchronize();
      return;

    case HRR_API_HIPSTREAMSYNCHRONIZE:
      if (!ctx.skip_device_sync)
        hipStreamSynchronize(ctx.translate_stream(ev.handle64));
      return;

    case HRR_API_HIPMODULEUNLOAD:
      ctx.remove_module(ev.handle64);
      return;

    case HRR_API_HIPSETDEVICE: {
      if (ev.raw_payload.size() >= 8) {
        int32_t dev = 0;
        memcpy(&dev, ev.raw_payload.data() + 4, 4);
        int n = 0; hipGetDeviceCount(&n);
        if (dev >= n) dev = 0;
        hipSetDevice(dev);
      }
      return;
    }

    case HRR_API_HIPGETLASTERROR:
    case HRR_API_HIPPEEKATLASTERROR:
      return;  // skip silently

    default: return;
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

static void dispatch_event(PlaybackContext& ctx, const hrr::Event& ev,
                           size_t idx, bool log, const hrr::Event* next_ev) {
  uint16_t etype = ev.header.event_type;

  // Give kernel-launch handlers the sequence ID so they can wait and advance
  // next_seq at the exact point of the HIP call.
  hrr_dispatch_seq = ev.header.sequence_id;
  auto order = needs_ordering(etype);
  auto next_order = next_ev && needs_ordering(next_ev->header.event_type);
  //order = (order ^ next_order);

  // Only "create" events need ordering — wait for turn then advance immediately
  // (before the call) so the next thread can start preparing while we execute.
  while (ctx.next_seq.load(std::memory_order_acquire) != ev.header.sequence_id)
    std::this_thread::yield();

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
  } seq_guard{ctx, ev.header.sequence_id + 1, order};

  if (is_special(etype)) {
    handle_special(ctx, ev);
    return;
  }

  if (etype >= HRR_API_COUNT || !hrr_playback_dispatch[etype]) {
    if (log && ctx.verbose)
      fprintf(stderr, "[HRR] T%llu Event %zu: no handler for type %u\n",
              (unsigned long long)ev.header.thread_id, idx, etype);
    return;
  }

  hipError_t r = hrr_playback_dispatch[etype](
      ctx, ev.raw_payload.data(), ev.raw_payload.size());

  if (r != hipSuccess && log) {
    fprintf(stderr, "[HRR] T%llu Event %zu (%s) error %d (%s)\n",
            (unsigned long long)ev.header.thread_id, idx,
            hrr::event_type_name(etype), r, hipGetErrorString(r));
  }
}

// ---------------------------------------------------------------------------
// Per-thread replay worker (MT path)
// ---------------------------------------------------------------------------

static void replay_thread(PlaybackContext& ctx,
                          const std::vector<const hrr::Event*>& events,
                          bool log) {
  for (size_t i = 0; i < events.size(); i++) {
    const hrr::Event& ev = *events[i];
    if (log && ctx.verbose)
      fprintf(stderr, "[HRR] T%llu [%zu] %s\n",
              (unsigned long long)ev.header.thread_id, i,
              hrr::event_type_name(ev.header.event_type));
    const hrr::Event* next_ev = (i + 1 < events.size()) ? events[i + 1] : nullptr;
    dispatch_event(ctx, ev, i, log, next_ev);
  }
}

// ---------------------------------------------------------------------------
// One full pass through the archive (single- or multi-threaded)
// ---------------------------------------------------------------------------

static void run_pass(PlaybackContext& ctx,
                     const hrr::Archive& archive,
                     const std::unordered_map<uint64_t,
                           std::vector<const hrr::Event*>>& thread_events,
                     bool use_mt, bool log) {
  if (use_mt) {
    // Reset the global sequence counter to the first event's seq_id so
    // threads start waiting at the right value for each pass.
    if (!archive.events.empty())
      ctx.next_seq.store(archive.events.front().header.sequence_id,
                         std::memory_order_relaxed);

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
      const auto& ev = archive.events[i];
      if (log && ctx.verbose)
        fprintf(stderr, "[HRR] Event %zu: %s\n", i,
                hrr::event_type_name(ev.header.event_type));
      const hrr::Event* next_ev = (i + 1 < archive.events.size()) ? &archive.events[i + 1] : nullptr;
      dispatch_event(ctx, ev, i, log, next_ev);
    }
  }
  hipDeviceSynchronize();
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
    "  --single-thread       Force single-threaded replay\n"
    "  --timing              Report wall time and total GPU kernel time\n"
    "  --kernel-filter STR   Only launch kernels whose name contains STR\n"
    "                        (silent full warm-up pass runs first)\n"
    "  --sync-after-launch   Sync after every kernel (surfaces GPU errors immediately)\n"
    "  --help                Show this message\n",
    argv0);
}

int main(int argc, char** argv) {
  if (argc < 2) { print_usage(argv[0]); return 1; }

  std::string archive_path;
  PlaybackContext ctx;
  ctx.validate_d2h = true;
  bool single_thread = false;
  bool show_info     = false;
  bool show_events   = false;

  for (int i = 1; i < argc; i++) {
    if      (!strcmp(argv[i], "--info"))              show_info             = true;
    else if (!strcmp(argv[i], "--events"))            show_events           = true;
    else if (!strcmp(argv[i], "--verbose"))           ctx.verbose           = true;
    else if (!strcmp(argv[i], "--skip-device-sync"))  ctx.skip_device_sync  = true;
    else if (!strcmp(argv[i], "--single-thread"))     single_thread         = true;
    else if (!strcmp(argv[i], "--timing"))            ctx.timing            = true;
    else if (!strcmp(argv[i], "--sync-after-launch")) ctx.sync_after_launch = true;
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
    thread_events[ev.header.thread_id].push_back(&ev);

  const bool use_mt = !single_thread && archive.threads.size() > 1;
  printf("[HRR] Mode    : %s\n", use_mt ? "multi-threaded" : "single-threaded");

  // Module pre-pass: process fat binary and explicit module load events in
  // global sequence order, single-threaded, before the parallel replay begins.
  // Without this, a timing delay on one thread (e.g. hipEventSynchronize) can
  // cause another thread's kernel launch to race ahead of the module load that
  // populates module_map, resulting in "kernel not found" errors.
  if (use_mt) {
    for (const auto& ev : archive.events) {
      uint16_t t = ev.header.event_type;
      if (t == HRR_API_HIPREGISTERFATBINARY ||
          t == HRR_API_HIPMODULELOADDATA    ||
          t == HRR_API_HIPMODULELOADDATAEX  ||
          t == HRR_API_HIPMODULELOAD) {
        dispatch_event(ctx, ev, 0, /*log=*/false, /*next_ev=*/nullptr);
      }
    }
  }

  // Warm-up pass: when a kernel filter is active, run the full archive once
  // without the filter so all GPU buffers (including intermediate kernel
  // outputs) are correctly populated before the timed filtered pass.
  if (!ctx.kernel_filter.empty()) {
    printf("[HRR] Warm-up : full pass to populate GPU state...\n");
    std::string filter = ctx.kernel_filter;
    ctx.kernel_filter.clear();
    run_pass(ctx, archive, thread_events, use_mt, /*log=*/false);
    ctx.kernel_filter    = filter;
    ctx.kernels_launched = 0;
    ctx.total_kernel_ms  = 0.0;
    printf("[HRR] Warm-up done. Running filtered pass...\n");
  }

  // Timed replay pass
  auto wall_start = std::chrono::high_resolution_clock::now();
  run_pass(ctx, archive, thread_events, use_mt, /*log=*/true);
  auto wall_end = std::chrono::high_resolution_clock::now();
  double wall_ms = std::chrono::duration<double, std::milli>(
                       wall_end - wall_start).count();

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
  printf("[HRR]   D2H checks     : %zu pass, %zu fail\n",
         ctx.d2h_pass.load(), ctx.d2h_fail.load());

  bool ok = (ctx.d2h_fail == 0);
  if (ctx.d2h_pass == 0 && ctx.d2h_fail == 0)
    printf("[HRR]   (no D2H validation blobs in archive -- re-capture to enable)\n");

  printf("[HRR] %s\n", ok ? "PASS" : "FAIL");

  // Cleanup
  for (auto& [rec, entry] : ctx.alloc_map)   hipFree(entry.live_ptr);
  for (auto& [rec, str]   : ctx.stream_map)  hipStreamDestroy(str);
  for (auto& [rec, ev2]   : ctx.event_map)   hipEventDestroy(ev2);
  for (hipEvent_t e : ctx.owned_timing_events) hipEventDestroy(e);

  // Unload all unique hipModule_t values across both maps.
  // co_modules holds modules loaded from code objects (by hash).
  // module_map holds fat-binary modules (not in co_modules) plus
  // duplicates of modules from explicit hipModuleLoad calls (already in
  // co_modules). The set deduplicates so each module is unloaded once.
  {
    std::unordered_set<hipModule_t> mods;
    for (auto& [hex, mod] : ctx.co_modules) mods.insert(mod);
    for (auto& [rec, mod] : ctx.module_map) mods.insert(mod);
    for (hipModule_t m : mods) hipModuleUnload(m);
  }

  return ok ? 0 : 1;
}
