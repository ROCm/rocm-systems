/*
 * ACCL Profiler Plugin — Combined GPU kernel + proxy/network decomposition.
 *
 * Subscribes to: Coll, KernelCh, ProxyOp, ProxyStep
 * Output: JSONL with per-collective timing decomposition.
 *
 * Build: see README.md or CMakeLists.txt
 */

#include "accl_profiler.h"

#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <sys/stat.h>
#include <unistd.h>

#include "accl_shim.h"

#define __hidden __attribute__((visibility("hidden")))

static ncclDebugLogger_t gLogFn;

// INFO stays on NCCL_PROFILE (0x4000): routine chatter, opt-in. WARN uses
// NCCL_ALL, as RCCL's own WARN does -- the default subsys mask is
// INIT|BOOTSTRAP|ENV, so a NCCL_PROFILE warning is dropped unless the user set
// NCCL_DEBUG_SUBSYS, and every warning here reports lost profiling data.
#define ACCL_INFO(...)  do { if (gLogFn) gLogFn(4, 0x4000, __func__, __LINE__, __VA_ARGS__); } while(0)
#define ACCL_WARN(...)  do { if (gLogFn) gLogFn(3, ~0UL, __func__, __LINE__, __VA_ARGS__); } while(0)

// rcclTelemetrySwCapture() from librccl, or NULL when RCCL network telemetry is
// unavailable (older librccl, ACCL_PROFILER_NET=0). Resolved once at first init.
typedef int (*acclNetCaptureFn)(RcclTelemetrySwSnapshot* out, int maxDevs);
static acclNetCaptureFn gNetCapture = NULL;
static pthread_once_t gNetCaptureOnce = PTHREAD_ONCE_INIT;
static void* gLogFnAddr = NULL;  // an address inside librccl, for dladdr()

static inline const char* safeStr(const char* s) { return s ? s : ""; }

// Parse ACCL_PROFILER_MIN_SIZE_BYTES. Returns the threshold, or 0 with *bad set
// for anything that is not a plain non-negative decimal integer fitting size_t.
// atol() failed silently in both directions: "-1" became SIZE_MAX, which filters
// out every collective and leaves an empty file that reads as a clean run, while
// "abc" became 0, indistinguishable from a deliberate 0.
static size_t acclParseMinSize(const char* env, int* bad) {
  *bad = 0;
  const char* p = env;
  // isspace, not just ' ' and '\t': strtoull skips the whole whitespace class,
  // so a leading '\n' or '\v' would otherwise walk past the sign check below
  // and let "-1" through as ULLONG_MAX without setting ERANGE.
  while (isspace((unsigned char)*p)) {
    p++;
  }
  // An all-blank value is the usual shell idiom for "off", not a typo, so it
  // takes the default without complaint.
  if (*p == '\0') {
    return 0;
  }
  // strtoull accepts a leading '-' and wraps it to a huge positive, which is the
  // silent filter-everything case above, so reject the sign before parsing.
  if (*p == '-') {
    *bad = 1;
    return 0;
  }
  errno = 0;
  char* end = NULL;
  unsigned long long v = strtoull(p, &end, 10);
  if (end == p || errno == ERANGE) {
    *bad = 1;
    return 0;
  }
  while (isspace((unsigned char)*end)) {
    end++;
  }
  if (*end != '\0') {
    *bad = 1;
    return 0;
  }
#if SIZE_MAX < ULLONG_MAX
  if (v > (unsigned long long)SIZE_MAX) {
    *bad = 1;
    return 0;
  }
#endif
  return (size_t)v;
}

// mkdir -p for the output directory. Returns 0 on success, -1 with errno set.
// A bare mkdir() only ever creates the last component, so the multi-level path
// the README documents fails ENOENT whenever two or more levels are missing.
static int acclMkdirRecursive(const char* path) {
  if (!path || !path[0]) { errno = EINVAL; return -1; }
  char buf[1024];
  size_t len = strlen(path);
  if (len >= sizeof(buf)) { errno = ENAMETOOLONG; return -1; }
  memcpy(buf, path, len + 1);
  // Trailing slashes would make the loop try to create the same level twice.
  while (len > 1 && buf[len - 1] == '/') buf[--len] = '\0';
  // Start at buf+1 so a leading '/' is never mkdir'd as the empty string.
  for (char* p = buf + 1; *p; p++) {
    if (*p != '/') continue;
    *p = '\0';
    if (mkdir(buf, 0755) != 0 && errno != EEXIST) return -1;
    *p = '/';
  }
  if (mkdir(buf, 0755) != 0 && errno != EEXIST) return -1;
  return 0;
}

// ============================================================================
// Growable slot pool (see struct acclPool)
// ============================================================================

// Adds one chunk and pushes its slots on the free list. Caller holds mutex.
static int acclPoolGrow(struct acclPool* pool) {
  if (pool->nChunks >= pool->maxChunks) return -1;
  int c = pool->nChunks;
  int n = pool->chunkSlots;
  char* chunk = (char*)calloc((size_t)n, pool->elemSize);
  int* used = (int*)calloc((size_t)n, sizeof(int));
  int* next = (int*)malloc((size_t)n * sizeof(int));
  if (!chunk || !used || !next) {
    free(chunk);
    free(used);
    free(next);
    return -1;
  }
  if (pool->slotMutexOff >= 0) {
    for (int i = 0; i < n; i++) {
      pthread_mutex_init((pthread_mutex_t*)(chunk + (size_t)i * pool->elemSize +
                                            pool->slotMutexOff), NULL);
    }
  }
  for (int i = 0; i < n; i++) next[i] = c * n + i + 1;
  next[n - 1] = pool->freeHead;
  pool->chunks[c] = chunk;
  pool->used[c] = used;
  pool->next[c] = next;
  pool->freeHead = c * n;
  pool->nChunks = c + 1;
  return 0;
}

static int acclPoolInit(struct acclPool* pool, size_t elemSize, int chunkSlots,
                        int capacity, long slotMutexOff) {
  pthread_mutex_init(&pool->mutex, NULL);
  pool->elemSize = elemSize;
  pool->chunkSlots = chunkSlots;
  pool->maxChunks = capacity / chunkSlots;
  pool->slotMutexOff = slotMutexOff;
  pool->freeHead = -1;
  return acclPoolGrow(pool);
}

static void acclPoolDestroy(struct acclPool* pool) {
  for (int c = 0; c < pool->nChunks; c++) {
    if (pool->slotMutexOff >= 0) {
      for (int i = 0; i < pool->chunkSlots; i++) {
        pthread_mutex_destroy((pthread_mutex_t*)(pool->chunks[c] +
                              (size_t)i * pool->elemSize + pool->slotMutexOff));
      }
    }
    free(pool->chunks[c]);
    free(pool->used[c]);
    free(pool->next[c]);
  }
  pool->nChunks = 0;
  pthread_mutex_destroy(&pool->mutex);
}

// Slots ever allocated, in use or not. Caller holds mutex.
static inline int acclPoolSlots(const struct acclPool* pool) {
  return pool->nChunks * pool->chunkSlots;
}

// No lock needed: idx came from an alloc, which published its chunk first.
static inline void* acclPoolSlot(const struct acclPool* pool, int idx) {
  return pool->chunks[idx / pool->chunkSlots] +
         (size_t)(idx % pool->chunkSlots) * pool->elemSize;
}

static inline int acclPoolIsUsed(const struct acclPool* pool, int idx) {
  return pool->used[idx / pool->chunkSlots][idx % pool->chunkSlots];
}

// Index of the slot at p, or -1 if p is not a slot of this pool. Caller holds mutex.
static int acclPoolIndex(const struct acclPool* pool, const void* p) {
  const char* cp = (const char*)p;
  size_t chunkBytes = (size_t)pool->chunkSlots * pool->elemSize;
  for (int c = 0; c < pool->nChunks; c++) {
    if (cp >= pool->chunks[c] && cp < pool->chunks[c] + chunkBytes) {
      return c * pool->chunkSlots + (int)((size_t)(cp - pool->chunks[c]) / pool->elemSize);
    }
  }
  return -1;
}

// Takes a free slot, growing the pool if it has none. Returns -1 at capacity.
// Caller holds mutex.
static int acclPoolTake(struct acclPool* pool) {
  if (pool->freeHead < 0 && acclPoolGrow(pool) != 0) return -1;
  int idx = pool->freeHead;
  pool->freeHead = pool->next[idx / pool->chunkSlots][idx % pool->chunkSlots];
  pool->used[idx / pool->chunkSlots][idx % pool->chunkSlots] = 1;
  return idx;
}

// Returns a slot to the free list. A slot already free is left alone, so a
// second release cannot put it on the list twice. Caller holds mutex.
static void acclPoolRelease(struct acclPool* pool, int idx) {
  if (idx < 0 || idx >= acclPoolSlots(pool) || !acclPoolIsUsed(pool, idx)) return;
  pool->used[idx / pool->chunkSlots][idx % pool->chunkSlots] = 0;
  pool->next[idx / pool->chunkSlots][idx % pool->chunkSlots] = pool->freeHead;
  pool->freeHead = idx;
}

// Forward declarations for cross-referenced pool functions
static void acclFreeProxyOp(struct acclCommContext* ctx, struct acclProxyOpInfo* op);
static void acclNetDiff(const struct acclCollInfo* coll, struct acclCompletedRecord* rec);

static void acclFreeCollProxyOps(struct acclCommContext* ctx,
                                 struct acclCollInfo* coll) {
  for (int i = 0; i < coll->nProxyOps; i++) {
    int idx = coll->proxyOpIndices[i];
    if (idx >= 0 && idx < ACCL_PROXY_OP_POOL_SIZE) {
      acclFreeProxyOp(ctx, (struct acclProxyOpInfo*)acclPoolSlot(&ctx->proxyOpPool, idx));
    }
  }
}

// ============================================================================
// Per-communicator pool allocators
// ============================================================================

// Release one context reference. The last holder tears the context down, so a
// callback that outlives acclPluginFinalize still finds live mutexes and file.
static void acclCtxUnref(struct acclCommContext* ctx) {
  if (__atomic_sub_fetch(&ctx->refCount, 1, __ATOMIC_SEQ_CST) != 0) {
    return;
  }
  pthread_mutex_lock(&ctx->outputMutex);
  if (ctx->outputFile) {
    fclose(ctx->outputFile);
    ctx->outputFile = NULL;
  }
  pthread_mutex_unlock(&ctx->outputMutex);
  acclPoolDestroy(&ctx->collPool);
  acclPoolDestroy(&ctx->proxyOpPool);
  acclPoolDestroy(&ctx->proxyStepPool);
  pthread_mutex_destroy(&ctx->outputMutex);
  free(ctx);
}

static struct acclCollInfo* acclAllocColl(struct acclCommContext* ctx) {
  pthread_mutex_lock(&ctx->collPool.mutex);
  int i = acclPoolTake(&ctx->collPool);
  if (i >= 0) {
    // Clear around the mutex, never through it, and hold it while clearing: a
    // stale KernelCh stop writes nKernelChCompleted under this lock, and that
    // path does not take collPool.mutex. POSIX does not define copying a
    // pthread_mutex_t.
    struct acclCollInfo* slot = (struct acclCollInfo*)acclPoolSlot(&ctx->collPool, i);
    const size_t muOff = offsetof(struct acclCollInfo, mutex);
    const size_t muEnd = muOff + sizeof(slot->mutex);
    pthread_mutex_lock(&slot->mutex);
    memset(slot, 0, muOff);
    memset((char*)slot + muEnd, 0, sizeof(*slot) - muEnd);
    pthread_mutex_unlock(&slot->mutex);
    __atomic_add_fetch(&ctx->refCount, 1, __ATOMIC_SEQ_CST);
    pthread_mutex_unlock(&ctx->collPool.mutex);
    return slot;
  }
  // Pool is full. Warn once: a per-drop WARN emits one line per collective for
  // the rest of the run, which buries the very message it is trying to deliver.
  // The end-of-run summary carries the totals.
  ctx->droppedCollectives++;
  int firstExhaustion = !ctx->poolExhaustedWarned;
  ctx->poolExhaustedWarned = 1;
  pthread_mutex_unlock(&ctx->collPool.mutex);
  if (firstExhaustion) {
    ACCL_WARN("ACCL Profiler: coll pool exhausted (%d slots). Profiling output for this "
              "communicator is now INCOMPLETE and must not be compared against a full "
              "run. Further drops are counted in the end-of-run summary only.",
              ACCL_COLL_POOL_SIZE);
  }
  return NULL;
}

static void acclFreeColl(struct acclCommContext* ctx, struct acclCollInfo* coll) {
  if (!coll) return;
  pthread_mutex_lock(&ctx->collPool.mutex);
  acclPoolRelease(&ctx->collPool, acclPoolIndex(&ctx->collPool, coll));
  pthread_mutex_unlock(&ctx->collPool.mutex);
  acclCtxUnref(ctx);
}

// Shared tail of both proxy allocators: count the drop, warn once per pool.
static void acclProxyPoolExhausted(uint64_t* dropped, int* warned,
                                   const char* noun, int poolSize) {
  __atomic_add_fetch(dropped, 1, __ATOMIC_SEQ_CST);
  if (__atomic_exchange_n(warned, 1, __ATOMIC_SEQ_CST) == 0) {
    ACCL_WARN("ACCL Profiler: proxy %s pool exhausted (%d slots). Proxy timing for this "
              "communicator is now INCOMPLETE. Further drops are counted in the "
              "end-of-run summary only.", noun, poolSize);
  }
}

static struct acclProxyOpInfo* acclAllocProxyOp(struct acclCommContext* ctx) {
  pthread_mutex_lock(&ctx->proxyOpPool.mutex);
  int i = acclPoolTake(&ctx->proxyOpPool);
  if (i >= 0) {
    __atomic_add_fetch(&ctx->refCount, 1, __ATOMIC_SEQ_CST);
    // Clear around the mutex, never through it, and hold it while clearing: a
    // stale ProxyStep stop reads generation under this lock, so the clear
    // must not run beside it. POSIX does not define copying a pthread_mutex_t.
    struct acclProxyOpInfo* slot = (struct acclProxyOpInfo*)acclPoolSlot(&ctx->proxyOpPool, i);
    const size_t muOff = offsetof(struct acclProxyOpInfo, mutex);
    const size_t muEnd = muOff + sizeof(slot->mutex);
    pthread_mutex_lock(&slot->mutex);
    uint64_t gen = slot->generation;
    memset(slot, 0, muOff);
    memset((char*)slot + muEnd, 0, sizeof(*slot) - muEnd);
    // One tenancy per alloc. A step that outlived the previous tenant is
    // rejected at stop rather than charged to this one.
    slot->generation = gen + 1;
    slot->poolIdx = i;
    pthread_mutex_unlock(&slot->mutex);
    pthread_mutex_unlock(&ctx->proxyOpPool.mutex);
    return slot;
  }
  pthread_mutex_unlock(&ctx->proxyOpPool.mutex);
  acclProxyPoolExhausted(&ctx->droppedProxyOps, &ctx->proxyOpPoolWarned,
                         "op", ACCL_PROXY_OP_POOL_SIZE);
  return NULL;
}

static void acclFreeProxyOp(struct acclCommContext* ctx, struct acclProxyOpInfo* op) {
  if (!op) return;
  // Not destroyed here: a ProxyStep stop still locks it after the slot is
  // released. Destroyed once in acclCtxUnref, with the last context reference.
  pthread_mutex_lock(&ctx->proxyOpPool.mutex);
  acclPoolRelease(&ctx->proxyOpPool, op->poolIdx);
  pthread_mutex_unlock(&ctx->proxyOpPool.mutex);
  acclCtxUnref(ctx);
}

static struct acclProxyStepInfo* acclAllocProxyStep(struct acclCommContext* ctx) {
  pthread_mutex_lock(&ctx->proxyStepPool.mutex);
  int i = acclPoolTake(&ctx->proxyStepPool);
  if (i >= 0) {
    struct acclProxyStepInfo* slot = (struct acclProxyStepInfo*)acclPoolSlot(&ctx->proxyStepPool, i);
    __atomic_add_fetch(&ctx->refCount, 1, __ATOMIC_SEQ_CST);
    memset(slot, 0, sizeof(*slot));
    pthread_mutex_unlock(&ctx->proxyStepPool.mutex);
    return slot;
  }
  pthread_mutex_unlock(&ctx->proxyStepPool.mutex);
  acclProxyPoolExhausted(&ctx->droppedProxySteps, &ctx->proxyStepPoolWarned,
                         "step", ACCL_PROXY_STEP_POOL_SIZE);
  return NULL;
}

static void acclFreeProxyStep(struct acclCommContext* ctx, struct acclProxyStepInfo* step) {
  if (!step) return;
  pthread_mutex_lock(&ctx->proxyStepPool.mutex);
  acclPoolRelease(&ctx->proxyStepPool, acclPoolIndex(&ctx->proxyStepPool, step));
  pthread_mutex_unlock(&ctx->proxyStepPool.mutex);
  acclCtxUnref(ctx);
}

// Charge `elapsed` to the state the step was IN over that interval. RCCL
// announces a state on ENTRY (src/transport/net.cc:1783,1849,1877,2059,2092,
// 2187), so the interval that just closed belongs to the previously announced
// state, not to the one being entered.
static void acclProxyStepChargeState(struct acclProxyStepInfo* step, int state, uint64_t elapsed) {
  switch (state) {
  case ncclProfilerProxyStepSendGPUWait:
    step->gpuWaitUs += elapsed;
    break;
  case ncclProfilerProxyStepSendPeerWait_v4:
    step->peerWaitUs += elapsed;
    break;
  case ncclProfilerProxyStepSendWait:
    step->sendWaitUs += elapsed;
    break;
  case ncclProfilerProxyStepRecvWait:
    step->recvWaitUs += elapsed;
    break;
  case ncclProfilerProxyStepRecvFlushWait:
    step->flushWaitUs += elapsed;
    break;
  case ncclProfilerProxyStepRecvGPUWait:
    step->gpuRecvWaitUs += elapsed;
    break;
  default:
    // -1 before the first transition: the buffer-post/irecv-post interval that
    // precedes it belongs to no named state, so it is deliberately dropped.
    break;
  }
}

// ============================================================================
// Datatype size helper
// ============================================================================
static int acclDatatypeSize(const char* dt) {
  if (!dt) return 4;
  // Must match ncclDatatypeToString() in src/collectives.cc.
  // ncclUint8 falls through to "Unknown" there (same enum value as
  // ncclChar in NCCL, but distinct in RCCL).
  static const struct { const char* name; int size; } table[] = {
    {"ncclInt8",        1},
    {"ncclFloat8e4m3",  1}, {"ncclFloat8e5m2",  1},
    {"ncclFloat16",     2}, {"ncclBfloat16",    2},
    {"ncclInt32",       4}, {"ncclUint32",      4}, {"ncclFloat32", 4},
    {"ncclInt64",       8}, {"ncclUint64",      8}, {"ncclFloat64", 8},
    {"Unknown",         1},
  };
  for (size_t i = 0; i < sizeof(table)/sizeof(table[0]); i++) {
    if (strcmp(dt, table[i].name) == 0) return table[i].size;
  }
  return 4;
}

// ============================================================================
// BusBW factor (from NCCL conventions)
// ============================================================================
static double acclBusBwFactor(const char* func, int nRanks) {
  if (!func || nRanks <= 1) return 1.0;
  double n = (double)nRanks;
  // Strings must match ncclFuncToString() in src/collectives.cc
  if (strcmp(func, "AllReduce") == 0)       return 2.0 * (n - 1.0) / n;
  if (strcmp(func, "ReduceScatter") == 0)   return (n - 1.0) / n;
  if (strcmp(func, "AllGather") == 0)       return (n - 1.0) / n;
  if (strcmp(func, "Broadcast") == 0)       return 1.0;
  if (strcmp(func, "Reduce") == 0)          return 1.0;
  if (strcmp(func, "AlltoAll") == 0)        return (n - 1.0) / n;
  if (strcmp(func, "AlltoAllv") == 0)       return (n - 1.0) / n;
  if (strcmp(func, "Gather") == 0)          return 1.0;
  if (strcmp(func, "Scatter") == 0)         return 1.0;
  return 1.0;
}

// Expansions of ACCL_DECOMP_FIELDS (accl_profiler.h) used by acclWriteRecord().
#define ACCL_DECOMP_KEY(ctype, key, fmt, member)      "\"" #key "\":" fmt ","
#define ACCL_DECOMP_KEY_LAST(ctype, key, fmt, member) "\"" #key "\":" fmt
#define ACCL_DECOMP_ARG(ctype, key, fmt, member)      , rec->member

// ============================================================================
// JSON output for a completed collective
// ============================================================================
static void acclWriteRecord(struct acclCommContext* ctx,
                            struct acclCompletedRecord* rec) {
  pthread_mutex_lock(&ctx->outputMutex);
  if (!ctx->outputFile) {
    pthread_mutex_unlock(&ctx->outputMutex);
    return;
  }

  double algoBw = (rec->totalExecUs > 0)
    ? ((double)rec->msgSizeBytes / 1e9) / (rec->totalExecUs / 1e6) : 0;
  double busBw = algoBw * acclBusBwFactor(rec->func, rec->nRanks);

  fprintf(ctx->outputFile,
    "{\"header\":{\"rank\":%d,\"n_ranks\":%d,\"host\":\"%s\",\"pid\":%d,\"comm_hash\":\"0x%lx\"},"
    "\"coll_perf\":{"
    "\"coll\":\"%s\",\"coll_sn\":%lu,\"coll_msg_size_bytes\":%zu,"
    "\"coll_algo\":\"%s\",\"coll_proto\":\"%s\","
    "\"coll_n_channels\":%d,\"coll_n_channels_reported\":%d,"
    "\"coll_exec_time_us\":%.2f,"
    "\"coll_algobw_gbs\":%.6f,\"coll_busbw_gbs\":%.6f,"
    "\"coll_timing_source\":\"%s\","
    "\"decomposition\":{"
      ACCL_DECOMP_FIELDS(ACCL_DECOMP_KEY, ACCL_DECOMP_KEY_LAST)
    "},",
    rec->rank, rec->nRanks, ctx->hostname, (int)getpid(), (unsigned long)ctx->commHash,
    safeStr(rec->func), (unsigned long)rec->seqNumber, rec->msgSizeBytes,
    safeStr(rec->algo), safeStr(rec->proto),
    rec->nChannels, rec->nChannelsRaw,
    rec->totalExecUs,
    algoBw, busBw,
    rec->hasGpuTiming ? "gpu_globaltimer" : "cpu_wallclock"
    ACCL_DECOMP_FIELDS(ACCL_DECOMP_ARG, ACCL_DECOMP_ARG)
  );

  fprintf(ctx->outputFile,
    "\"host_ts\":{\"clock\":\"CLOCK_MONOTONIC\",\"enqueue_us\":%lu,"
    "\"exec_start_us\":%lu,\"exec_end_us\":%lu},",
    (unsigned long)rec->tsEnqueueUs, (unsigned long)rec->tsExecStartUs,
    (unsigned long)rec->tsExecEndUs);

  // Per-NIC SW counter deltas. Only when telemetry was capturing and the
  // collective went through the proxy (i.e. used the network).
  if (rec->tsNetStartUs != 0 && rec->tsNetStopUs != 0) {
    fprintf(ctx->outputFile,
      "\"net\":{\"source\":\"rccl_telemetry_sw\",\"window_start_us\":%lu,"
      "\"window_end_us\":%lu,\"devices\":[",
      (unsigned long)rec->tsNetStartUs, (unsigned long)rec->tsNetStopUs);
    for (int i = 0; i < rec->nNetDevs; i++) {
      const RcclTelemetrySwSnapshot* d = &rec->netDelta[i];
      fprintf(ctx->outputFile,
        "%s{\"device_id\":%d,\"tx_bytes\":%lu,\"rx_bytes\":%lu,\"wqe_sent\":%lu,"
        "\"recv_wqe\":%lu,\"wqe_rcvd\":%lu,\"wqe_completed\":%lu,\"wqe_sampled\":%lu,"
        "\"cq_errors\":%lu,\"completion_hist\":[",
        i > 0 ? "," : "", d->device_id,
        (unsigned long)d->tx_bytes, (unsigned long)d->rx_bytes, (unsigned long)d->wqe_sent,
        (unsigned long)d->recv_wqe, (unsigned long)d->wqe_rcvd, (unsigned long)d->wqe_completed,
        (unsigned long)d->wqe_sampled, (unsigned long)d->num_cq_errors);
      int last = RCCL_TELEMETRY_HISTOGRAM_SIZE - 1;
      while (last >= 0 && d->wqe_completion_histogram[last] == 0) last--;
      for (int h = 0; h <= last; h++)
        fprintf(ctx->outputFile, "%s%lu", h > 0 ? "," : "", (unsigned long)d->wqe_completion_histogram[h]);
      fprintf(ctx->outputFile, "]}");
    }
    fprintf(ctx->outputFile, "]},");
  }

  // Kernel events array
  fprintf(ctx->outputFile, "\"event_trace_ts\":{\"kernel_events\":[");
  for (int i = 0; i < rec->nKernelEvents; i++) {
    if (i > 0) fprintf(ctx->outputFile, ",");
    fprintf(ctx->outputFile,
      "{\"channel_id\":%d,\"kernel_start_ts\":%lu,\"kernel_stop_ts\":%lu,\"duration_us\":%lu}",
      rec->kernelEvents[i].channelId,
      (unsigned long)rec->kernelEvents[i].startGpuClk,
      (unsigned long)rec->kernelEvents[i].stopGpuClk,
      (unsigned long)rec->kernelEvents[i].durationUs);
  }
  fprintf(ctx->outputFile, "]}}}\n");
  fflush(ctx->outputFile);
  pthread_mutex_unlock(&ctx->outputMutex);
}

// ============================================================================
// Finalize a collective: compute decomposition, emit record
// ============================================================================
static void acclFinalizeCollective(struct acclCollInfo* coll) {
  struct acclCommContext* ctx = (struct acclCommContext*)coll->commCtx;
  if (!ctx) return;

  struct acclCompletedRecord rec;
  memset(&rec, 0, sizeof(rec));

  rec.func = coll->func;
  rec.algo = coll->algo;
  rec.proto = coll->proto;
  rec.seqNumber = coll->seqNumber;
  rec.msgSizeBytes = coll->msgSizeBytes;
  rec.nChannels = coll->nChannels;
  rec.nChannelsRaw = coll->nChannelsRaw;
  rec.rank = ctx->rank;
  rec.nRanks = ctx->nRanks;

  // Kernel timing
  uint64_t firstKernelCpuStartUs = UINT64_MAX;
  uint64_t lastCpuStop = 0;
  uint64_t firstGpuClk = UINT64_MAX;
  uint64_t lastGpuClk = 0;
  double kernelDurSum = 0;
  double kernelDurMin = 1e18;
  double kernelDurMax = 0;
  int nKernelEvents = 0;
  int hasGpuTiming = 0;

  for (uint32_t ch = 0; ch < ACCL_MAX_CHANNELS; ch++) {
    struct acclKernelChInfo* kch = &coll->kernelCh[ch];
    if (kch->tsStartUs == 0) continue;
    if (kch->tsStopUs == 0) continue;

    double dur;
    if (kch->startGpuClk != 0 && kch->stopGpuClk != 0 && kch->stopGpuClk > kch->startGpuClk) {
      // AMD wall_clock64() runs at 100 MHz (10 ns/tick)
      dur = (double)(kch->stopGpuClk - kch->startGpuClk) / 100.0;
      hasGpuTiming = 1;
      if (kch->startGpuClk < firstGpuClk) firstGpuClk = kch->startGpuClk;
      if (kch->stopGpuClk > lastGpuClk) lastGpuClk = kch->stopGpuClk;
    } else {
      dur = (double)(kch->tsStopUs - kch->tsStartUs);
    }

    if (kch->tsStartUs < firstKernelCpuStartUs) firstKernelCpuStartUs = kch->tsStartUs;
    if (kch->tsStopUs > lastCpuStop) lastCpuStop = kch->tsStopUs;

    kernelDurSum += dur;
    if (dur < kernelDurMin) kernelDurMin = dur;
    if (dur > kernelDurMax) kernelDurMax = dur;

    if (nKernelEvents < ACCL_MAX_CHANNELS) {
      rec.kernelEvents[nKernelEvents].channelId = kch->channelId;
      rec.kernelEvents[nKernelEvents].startGpuClk = kch->startGpuClk;
      rec.kernelEvents[nKernelEvents].stopGpuClk = kch->stopGpuClk;
      rec.kernelEvents[nKernelEvents].durationUs = (uint64_t)dur;
      nKernelEvents++;
    }
  }

  rec.nKernelEvents = nKernelEvents;
  rec.hasGpuTiming = hasGpuTiming;

  if (nKernelEvents > 0) {
    rec.gpuKernelUs = kernelDurSum / nKernelEvents;
    rec.gpuKernelMinUs = kernelDurMin;
    rec.gpuKernelMaxUs = kernelDurMax;
    if (hasGpuTiming && lastGpuClk > firstGpuClk) {
      rec.totalExecUs = (double)(lastGpuClk - firstGpuClk) / 100.0;
    } else {
      rec.totalExecUs = (double)(lastCpuStop - firstKernelCpuStartUs);
    }
    rec.enqueueToKernelUs = (firstKernelCpuStartUs > coll->tsCollStartUs)
      ? (double)(firstKernelCpuStartUs - coll->tsCollStartUs) : 0;
  } else {
    rec.totalExecUs = (coll->tsCollStopUs > coll->tsCollStartUs)
      ? (double)(coll->tsCollStopUs - coll->tsCollStartUs) : 0;
  }

  // Each component is averaged over the op class that can produce it, not over
  // every proxy op: send ops only pass through SendGPUWait/SendPeerWait/SendWait
  // and recv ops only through RecvWait/RecvFlushWait/RecvGPUWait, so dividing a
  // one-sided total by nProxyOps scales it by that class's share of the mix. On
  // a ring (one send + one recv op per channel) these are per-channel costs --
  // the same scale as gpu_kernel_avg_us.
  double sendGpuWait = 0, sendPeerWait = 0, sendNetwork = 0;
  double recvFlush = 0, recvGpuWait = 0, recvNetwork = 0;
  int nSend = 0, nRecv = 0;
  uint64_t lastProxyStopUs = 0;

  for (int i = 0; i < coll->nProxyOps; i++) {
    int opIdx = coll->proxyOpIndices[i];
    if (opIdx < 0 || opIdx >= ACCL_PROXY_OP_POOL_SIZE) continue;
    struct acclProxyOpInfo* op = (struct acclProxyOpInfo*)acclPoolSlot(&ctx->proxyOpPool, opIdx);
    // Under op->mutex: a step of this op can still be stopping and adding to
    // these accumulators, and that path writes them under the same lock.
    pthread_mutex_lock(&op->mutex);
    if (op->tsStopUs > lastProxyStopUs) lastProxyStopUs = op->tsStopUs;
    if (op->isSend) {
      nSend++;
      sendGpuWait += (double)op->totalGpuWaitUs;
      sendPeerWait += (double)op->totalPeerWaitUs;
      sendNetwork += (double)op->totalNetworkUs;
    } else {
      nRecv++;
      recvFlush += (double)op->totalFlushUs;
      recvGpuWait += (double)op->totalGpuRecvWaitUs;
      recvNetwork += (double)op->totalNetworkUs;
    }
    pthread_mutex_unlock(&op->mutex);
  }

  // A zero denominator only happens when the class has no ops at all, in which
  // case its numerator is zero too, so reporting 0 states the truth rather than
  // papering over a division; the guard exists only to avoid 0/0.
  double meanSendNet = nSend > 0 ? sendNetwork / nSend : 0;
  double meanRecvNet = nRecv > 0 ? recvNetwork / nRecv : 0;

  int nOps = coll->nProxyOps;
  rec.proxyGpuWaitUs = nSend > 0 ? sendGpuWait / nSend : 0;
  rec.proxyNetworkUs = meanSendNet + meanRecvNet;
  rec.proxyPeerWaitUs = nSend > 0 ? sendPeerWait / nSend : 0;
  rec.proxyFlushUs = nRecv > 0 ? recvFlush / nRecv : 0;
  rec.proxyGpuRecvWaitUs = nRecv > 0 ? recvGpuWait / nRecv : 0;
  rec.nProxyOps = nOps;
  rec.nSendOps = nSend;
  rec.nRecvOps = nRecv;

  rec.tsEnqueueUs = coll->tsCollStartUs;
  // Proxy op starts are left out on purpose (see acclPluginRecordEventState):
  // the collective starts running when its kernel or its first data step does.
  uint64_t firstNetUs = coll->tsNetStartUs ? coll->tsNetStartUs : UINT64_MAX;
  uint64_t execStart = firstKernelCpuStartUs < firstNetUs ? firstKernelCpuStartUs : firstNetUs;
  uint64_t execEnd = lastCpuStop > lastProxyStopUs ? lastCpuStop : lastProxyStopUs;
  rec.tsExecStartUs = execStart != UINT64_MAX ? execStart : 0;
  rec.tsExecEndUs = execEnd;

  if (coll->tsNetStartUs != 0 && coll->tsNetStopUs != 0) acclNetDiff(coll, &rec);

  acclWriteRecord(ctx, &rec);
}

// Returns 1 (under coll->mutex) if this coll should be finalized now.
// Sets the finalized flag to prevent double finalization.
static inline int acclShouldFinalize(struct acclCollInfo* coll) {
  if (coll->finalized) return 0;
  if (!coll->collStopped) return 0;
  if (coll->nKernelChCompleted < coll->nChannels) return 0;
  if (coll->nProxyOpsCompleted < coll->nProxyOpsStarted) return 0;
  coll->finalized = 1;
  return 1;
}

static void acclFinalizeAndFree(struct acclCommContext* ctx,
                                struct acclCollInfo* coll) {
  acclFinalizeCollective(coll);
  acclFreeCollProxyOps(ctx, coll);
  acclFreeColl(ctx, coll);
}

// ============================================================================
// RCCL network telemetry (per-collective SW counter snapshots)
// ============================================================================

// librccl exports rcclTelemetrySwCapture(), but RTLD_DEFAULT only finds it when
// librccl is in the global symbol scope. A Python extension pulls librccl in
// with RTLD_LOCAL, so look it up through the library that owns the logger
// callback RCCL handed us first, and fall back to the global scope.
static void acclNetCaptureResolve(void) {
  const char* env = getenv("ACCL_PROFILER_NET");
  if (env && atoi(env) == 0) return;

  void* sym = NULL;
  Dl_info info;
  if (gLogFnAddr && dladdr(gLogFnAddr, &info) && info.dli_fname) {
    void* lib = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
    if (lib) {
      sym = dlsym(lib, "rcclTelemetrySwCapture");
      dlclose(lib);  // RTLD_NOLOAD took a reference; librccl stays loaded
    }
  }
  if (!sym) sym = dlsym(RTLD_DEFAULT, "rcclTelemetrySwCapture");
  gNetCapture = (acclNetCaptureFn)sym;
  if (!gNetCapture) {
    ACCL_INFO("ACCL Profiler: rcclTelemetrySwCapture not found; per-collective "
              "network counters disabled");
  }
}

// Capture the SW counters of every NIC RCCL telemetry tracks. Returns the
// number of devices written; 0 when telemetry is off (RCCL_TELEMETRY_ENABLE
// unset) or unavailable.
static inline int acclNetCapture(RcclTelemetrySwSnapshot* out) {
  return gNetCapture ? gNetCapture(out, ACCL_NET_MAX_DEVS) : 0;
}

// rec->netDelta = stop - start for each device present in both snapshots.
// Devices with no traffic in the window are left out. Min/max completion times
// are whole-run values in the snapshot and cannot be differenced, so they are
// not carried over; the histogram is.
static void acclNetDiff(const struct acclCollInfo* coll, struct acclCompletedRecord* rec) {
  rec->nNetDevs = 0;
  rec->tsNetStartUs = coll->tsNetStartUs;
  rec->tsNetStopUs = coll->tsNetStopUs;
  for (int i = 0; i < coll->nNetDevsStop; i++) {
    const RcclTelemetrySwSnapshot* b = &coll->netStop[i];
    const RcclTelemetrySwSnapshot* a = NULL;
    for (int j = 0; j < coll->nNetDevsStart; j++) {
      if (coll->netStart[j].device_id == b->device_id) { a = &coll->netStart[j]; break; }
    }
    if (!a) continue;
    RcclTelemetrySwSnapshot* d = &rec->netDelta[rec->nNetDevs];
    memset(d, 0, sizeof(*d));
    d->device_id     = b->device_id;
    d->tx_bytes      = b->tx_bytes - a->tx_bytes;
    d->rx_bytes      = b->rx_bytes - a->rx_bytes;
    d->num_cq_errors = b->num_cq_errors - a->num_cq_errors;
    d->wqe_sent      = b->wqe_sent - a->wqe_sent;
    d->recv_wqe      = b->recv_wqe - a->recv_wqe;
    d->wqe_rcvd      = b->wqe_rcvd - a->wqe_rcvd;
    d->wqe_completed = b->wqe_completed - a->wqe_completed;
    d->wqe_sampled   = b->wqe_sampled - a->wqe_sampled;
    for (int h = 0; h < RCCL_TELEMETRY_HISTOGRAM_SIZE; h++)
      d->wqe_completion_histogram[h] = b->wqe_completion_histogram[h] - a->wqe_completion_histogram[h];
    if (d->tx_bytes == 0 && d->rx_bytes == 0 && d->wqe_sent == 0 && d->recv_wqe == 0 &&
        d->wqe_completed == 0 && d->num_cq_errors == 0)
      continue;
    rec->nNetDevs++;
  }
}

// ============================================================================
// Plugin callbacks
// ============================================================================

__hidden ncclResult_t acclPluginInit(void** context, uint64_t commHash,
                                     int* eActivationMask, const char* commName,
                                     int nNodes, int nRanks, int rank,
                                     ncclDebugLogger_t logfn) {
  gLogFn = logfn;
  gLogFnAddr = (void*)logfn;
  pthread_once(&gNetCaptureOnce, acclNetCaptureResolve);

  struct acclCommContext* ctx = (struct acclCommContext*)calloc(1, sizeof(*ctx));
  if (!ctx) return ncclSuccess;

  // Read into the context, not a file static: with several communicators in one
  // process a static is last-init-wins, and because the assignment is
  // conditional it is also never reset once set.
  const char* env;
  if ((env = getenv("ACCL_PROFILER_MIN_SIZE_BYTES")) != NULL) {
    int badMinSize = 0;
    ctx->minMsgSize = acclParseMinSize(env, &badMinSize);
    if (badMinSize) {
      // Fail open, to 0 rather than a guess: too large a threshold profiles
      // nothing and no summary field marks that file as wrong, while 0 shows up
      // as more data than asked for.
      ACCL_WARN("ACCL Profiler: ACCL_PROFILER_MIN_SIZE_BYTES=\"%s\" is not a "
                "non-negative byte count; ignoring it and profiling every "
                "collective. A negative or out-of-range value would otherwise "
                "filter out every collective and leave an empty output file.",
                env);
    }
  }

  ctx->refCount = 1;
  // One chunk each up front, as the fixed pools had; the rest on demand.
  int poolErr = acclPoolInit(&ctx->collPool, sizeof(struct acclCollInfo),
                             ACCL_COLL_POOL_CHUNK, ACCL_COLL_POOL_SIZE,
                             (long)offsetof(struct acclCollInfo, mutex));
  poolErr |= acclPoolInit(&ctx->proxyOpPool, sizeof(struct acclProxyOpInfo),
                          ACCL_PROXY_OP_POOL_CHUNK, ACCL_PROXY_OP_POOL_SIZE,
                          (long)offsetof(struct acclProxyOpInfo, mutex));
  poolErr |= acclPoolInit(&ctx->proxyStepPool, sizeof(struct acclProxyStepInfo),
                          ACCL_PROXY_STEP_POOL_CHUNK, ACCL_PROXY_STEP_POOL_SIZE, -1);
  if (poolErr) {
    acclPoolDestroy(&ctx->collPool);
    acclPoolDestroy(&ctx->proxyOpPool);
    acclPoolDestroy(&ctx->proxyStepPool);
    free(ctx);
    return ncclSuccess;
  }
  ctx->commHash = commHash;
  ctx->rank = rank;
  ctx->nRanks = nRanks;
  ctx->nNodes = nNodes;
  if (commName) strncpy(ctx->commName, commName, sizeof(ctx->commName) - 1);

  pthread_mutex_init(&ctx->outputMutex, NULL);

  // Open output file (write mode — fresh file per init to avoid stale data)
  const char* outDir = getenv("ACCL_PROFILER_OUTPUT_DIR");
  // Empty as well as unset: "" would assemble a path at the filesystem root.
  if (!outDir || !outDir[0]) outDir = "/tmp";

  gethostname(ctx->hostname, sizeof(ctx->hostname) - 1);

  // fopen() below reports only ENOENT on the assembled filename, which does not
  // say the directory is the problem, so keep this errno to name the real cause.
  int dirErrno = (acclMkdirRecursive(outDir) == 0) ? 0 : errno;

  int pathLen = snprintf(ctx->outputPath, sizeof(ctx->outputPath),
    "%s/accl_profiler_rank%d_%s_pid%d_0x%lx.jsonl",
    outDir, rank, ctx->hostname, (int)getpid(), (unsigned long)commHash);
  if (pathLen < 0 || (size_t)pathLen >= sizeof(ctx->outputPath)) {
    // Truncation cuts the rank/pid/hash suffix, so every rank in the job would
    // open one shared name and interleave records. Refuse the path instead.
    ACCL_WARN("ACCL Profiler: ACCL_PROFILER_OUTPUT_DIR is too long: output path "
              "needs %d bytes but only %zu are available. No profiling output "
              "will be written.", pathLen, sizeof(ctx->outputPath));
    ctx->outputPath[0] = '\0';
    ctx->outputFile = NULL;
  } else {
    ctx->outputFile = fopen(ctx->outputPath, "w");
    if (!ctx->outputFile) {
      int openErrno = errno;
      if (dirErrno != 0) {
        ACCL_WARN("ACCL Profiler: cannot create output directory %s (from "
                  "ACCL_PROFILER_OUTPUT_DIR): %s. No profiling output will be "
                  "written.", outDir, strerror(dirErrno));
      } else {
        ACCL_WARN("ACCL Profiler: Failed to open %s: %s", ctx->outputPath,
                  strerror(openErrno));
      }
    }
  }

  *context = ctx;
  *eActivationMask = ncclProfileColl | ncclProfileKernelCh
                   | ncclProfileProxyOp | ncclProfileProxyStep;

  ACCL_INFO("ACCL Profiler: init rank=%d nRanks=%d nNodes=%d "
            "output=%s minSize=%zu netCounters=%s",
            rank, nRanks, nNodes, ctx->outputPath, ctx->minMsgSize,
            gNetCapture ? "on" : "off");
  return ncclSuccess;
}

__hidden ncclResult_t acclPluginFinalize(void* context) {
  struct acclCommContext* ctx = (struct acclCommContext*)context;
  if (!ctx) return ncclSuccess;

  ACCL_INFO("ACCL Profiler: finalize rank=%d output=%s", ctx->rank, ctx->outputPath);

  // Drain any coll slots still in use (orphaned by teardown).
  pthread_mutex_lock(&ctx->collPool.mutex);
  for (int i = 0; i < acclPoolSlots(&ctx->collPool); i++) {
    if (acclPoolIsUsed(&ctx->collPool, i)) {
      struct acclCollInfo* coll = (struct acclCollInfo*)acclPoolSlot(&ctx->collPool, i);
      // Take the claim under the lock every writer uses, not collPool.mutex. An
      // already-claimed slot belongs to a thread between acclShouldFinalize and
      // acclFreeColl, which releases the slot and its reference itself; touching
      // it here double-releases both. Read and set in one critical section so no
      // writer can claim a slot this drain has decided to release.
      pthread_mutex_lock(&coll->mutex);
      int claimed = coll->finalized;
      coll->finalized = 1;
      pthread_mutex_unlock(&coll->mutex);
      if (claimed) {
        // Its owner is between acclShouldFinalize and acclWriteRecord, and the
        // file closes below -- after which acclWriteRecord drops every write.
        // That record may never reach disk, so count it rather than let a run
        // that lost one still read as clean.
        ctx->inflightCollectives++;
        continue;
      }
      // Never reached its completion predicate — teardown skipped one or more of
      // its kernel-channel events. Count it so the loss is visible; no record is
      // emitted for it.
      ctx->leakedCollectives++;
      acclFreeCollProxyOps(ctx, coll);
      acclPoolRelease(&ctx->collPool, i);
      __atomic_sub_fetch(&ctx->refCount, 1, __ATOMIC_SEQ_CST);
    }
  }
  pthread_mutex_unlock(&ctx->collPool.mutex);

  // Count, but deliberately do NOT release, proxy ops and steps still held at
  // finalize. Three classes never reach acclFreeProxyOp/acclFreeProxyStep on
  // their own: an op with no coll parent, an op whose stop never arrives
  // (coll->proxyOpIndices is filled at op STOP, so the coll drain above cannot
  // see it), and any unstopped step.
  //
  // Draining them would be a use-after-free. Unlike the profiler thread's work
  // queue, which ncclProfilerThreadDestroy quiesces and purges by context
  // before we are called, proxy op and step callbacks are issued straight from
  // the proxy progress thread (src/plugin/profiler.cc:748,783,794,812,845), and
  // that thread is shared through comm->sharedRes: ncclProxyDestroy only tears
  // it down for the last comm using it, so a split-comm teardown finalizes this
  // context with events for it still in flight. Releasing their references here
  // could take refCount to zero and free the context under a live caller.
  //
  // So the reference stays, and whichever stop arrives last frees the context.
  // If a stop never arrives -- an aborted proxy thread -- this context is
  // retained for the life of the process. The cost is bounded: one
  // acclCommContext per affected communicator, and the output file is closed
  // below regardless, so no descriptor is held.
  pthread_mutex_lock(&ctx->proxyOpPool.mutex);
  for (int i = 0; i < acclPoolSlots(&ctx->proxyOpPool); i++) {
    if (acclPoolIsUsed(&ctx->proxyOpPool, i)) ctx->outstandingProxyOps++;
  }
  pthread_mutex_unlock(&ctx->proxyOpPool.mutex);

  pthread_mutex_lock(&ctx->proxyStepPool.mutex);
  for (int i = 0; i < acclPoolSlots(&ctx->proxyStepPool); i++) {
    if (acclPoolIsUsed(&ctx->proxyStepPool, i)) ctx->outstandingProxySteps++;
  }
  pthread_mutex_unlock(&ctx->proxyStepPool.mutex);

  // Write the drop summary while the output file is still open, under the lock
  // acclWriteRecord uses, so the summary cannot land inside a record. The file
  // is closed by acclCtxUnref once the last reference goes away.
  pthread_mutex_lock(&ctx->outputMutex);
  if (ctx->outputFile) {
    // Emit the summary unconditionally, including on a clean run: a consumer
    // that finds no summary line cannot distinguish "nothing was lost" from
    // "the process died before finalize".
    uint64_t dOps   = __atomic_load_n(&ctx->droppedProxyOps, __ATOMIC_SEQ_CST);
    uint64_t dSteps = __atomic_load_n(&ctx->droppedProxySteps, __ATOMIC_SEQ_CST);
    uint64_t oOps   = __atomic_load_n(&ctx->overflowProxyOps, __ATOMIC_SEQ_CST);
    uint64_t sSteps = __atomic_load_n(&ctx->staleProxySteps, __ATOMIC_SEQ_CST);
    // Complete only if nothing was lost anywhere. Proxy loss understates the
    // decomposition with no marker on the affected records, and ferror() is the
    // one place a discarded fprintf failure (ENOSPC, EIO) is still visible, so
    // both clear the flag.
    int writeError = ferror(ctx->outputFile) != 0;
    int complete = (ctx->droppedCollectives == 0 && ctx->leakedCollectives == 0 &&
                    ctx->inflightCollectives == 0 &&
                    dOps == 0 && dSteps == 0 && oOps == 0 && sSteps == 0 &&
                    ctx->outstandingProxyOps == 0 &&
                    ctx->outstandingProxySteps == 0 &&
                    !writeError);
    fprintf(ctx->outputFile,
      "{\"summary\":{\"dropped_collectives\":%lu,\"leaked_collectives\":%lu,"
      "\"inflight_collectives\":%lu,"
      "\"dropped_proxy_ops\":%lu,\"dropped_proxy_steps\":%lu,"
      "\"overflow_proxy_ops\":%lu,\"stale_proxy_steps\":%lu,"
      "\"outstanding_proxy_ops\":%lu,\"outstanding_proxy_steps\":%lu,"
      "\"write_error\":%s,"
      "\"coll_pool_size\":%d,\"proxy_op_pool_size\":%d,\"proxy_step_pool_size\":%d,"
      "\"max_proxy_ops_per_coll\":%d,\"complete\":%s}}\n",
      (unsigned long)ctx->droppedCollectives,
      (unsigned long)ctx->leakedCollectives,
      (unsigned long)ctx->inflightCollectives,
      (unsigned long)dOps, (unsigned long)dSteps, (unsigned long)oOps,
      (unsigned long)sSteps,
      (unsigned long)ctx->outstandingProxyOps,
      (unsigned long)ctx->outstandingProxySteps,
      writeError ? "true" : "false",
      ACCL_COLL_POOL_SIZE, ACCL_PROXY_OP_POOL_SIZE, ACCL_PROXY_STEP_POOL_SIZE,
      ACCL_MAX_PROXY_OPS, complete ? "true" : "false");
    fflush(ctx->outputFile);
    if (!complete) {
      ACCL_WARN("ACCL Profiler: rank=%d output INCOMPLETE: %lu collectives dropped "
                "(coll pool exhausted), %lu slots leaked (teardown-skipped kernel "
                "events), %lu collectives in flight at finalize (record may be "
                "missing), %lu proxy ops dropped, %lu proxy steps dropped, %lu proxy "
                "ops discarded (more than %d on one collective), %lu proxy steps "
                "stale (parent slot reissued), %lu proxy ops and %lu proxy steps "
                "still outstanding, write_error=%d",
                ctx->rank, (unsigned long)ctx->droppedCollectives,
                (unsigned long)ctx->leakedCollectives,
                (unsigned long)ctx->inflightCollectives,
                (unsigned long)dOps, (unsigned long)dSteps, (unsigned long)oOps,
                ACCL_MAX_PROXY_OPS, (unsigned long)sSteps,
                (unsigned long)ctx->outstandingProxyOps,
                (unsigned long)ctx->outstandingProxySteps, writeError);
    }
    if (writeError) {
      // Say it separately: when writes are failing the summary above may not
      // reach disk at all, so the log is the only channel left.
      ACCL_WARN("ACCL Profiler: rank=%d hit a write error on %s; the records on "
                "disk are truncated", ctx->rank, ctx->outputPath);
    }
    // Close here rather than leaving it to acclCtxUnref. A context held by an
    // outstanding proxy op or step is not freed at finalize and may never be,
    // so deferring would hold a descriptor for the life of the process -- and
    // descriptors, unlike the fixed-size context, are a shared process limit.
    // Nothing legitimate is lost: the summary above is the documented last line
    // of the file, so a record written after it would violate that contract,
    // and acclWriteRecord already drops writes once outputFile is NULL.
    fclose(ctx->outputFile);
    ctx->outputFile = NULL;
  }
  pthread_mutex_unlock(&ctx->outputMutex);

  // Drop the init reference. Outstanding proxy ops and steps hold their own, so
  // the context survives here exactly when one of them may still be delivered.
  acclCtxUnref(ctx);
  return ncclSuccess;
}

__hidden ncclResult_t acclPluginStartEvent(void* context, void** eHandle,
                                            ncclProfilerEventDescr_v5_t* eDescr) {
  if (!context || !eDescr) {
    *eHandle = NULL;
    return ncclSuccess;
  }

  struct acclCommContext* ctx = (struct acclCommContext*)context;

  if (eDescr->type == ncclProfileColl) {
    // Check message size filter
    size_t msgSize = (size_t)acclDatatypeSize(eDescr->coll.datatype) * eDescr->coll.count;
    if (msgSize < ctx->minMsgSize) {
      *eHandle = NULL;
      return ncclSuccess;
    }

    struct acclCollInfo* coll = acclAllocColl(ctx);
    if (!coll) {
      *eHandle = NULL;
      return ncclSuccess;
    }
    coll->type = ncclProfileColl;
    coll->func = eDescr->coll.func;
    coll->algo = eDescr->coll.algo;
    coll->proto = eDescr->coll.proto;
    coll->seqNumber = eDescr->coll.seqNumber;
    // eDescr->coll.nChannels is uint8_t (profiler_v5.h), but RCCL's
    // ncclTaskColl::nChannels is uint16_t (enqueue.cc) and MAXCHANNELS is 256 —
    // deliberately raised so NCCL_MAX_NCHANNELS=256 is a supported setting. A
    // 256-channel collective therefore arrives here as 0.
    //
    // Promote a reported 0 to ACCL_MAX_CHANNELS instead of taking it literally.
    // Erring high only delays finalization: the slot stays allocated and is
    // reclaimed by the drain in acclPluginFinalize(). Erring low would satisfy
    // acclShouldFinalize() at coll-stop and free the slot while the proxy
    // thread is still delivering KernelCh events into it — RCCL fires coll-stop
    // immediately after ncclProxyStart(), not after the collective completes.
    coll->nChannelsRaw = eDescr->coll.nChannels;
    coll->nChannels = eDescr->coll.nChannels ? eDescr->coll.nChannels
                                             : ACCL_MAX_CHANNELS;
    coll->msgSizeBytes = msgSize;
    coll->tsCollStartUs = acclGetTimeUs();
    coll->commCtx = ctx;

    *eHandle = coll;
    return ncclSuccess;
  }

  if (eDescr->type == ncclProfileKernelCh) {
    if (!eDescr->parentObj) {
      *eHandle = NULL;
      return ncclSuccess;
    }
    uint64_t parentType = *(uint64_t*)eDescr->parentObj;
    if (parentType != ncclProfileColl) {
      *eHandle = NULL;
      return ncclSuccess;
    }

    struct acclCollInfo* coll = (struct acclCollInfo*)eDescr->parentObj;
    uint8_t chId = eDescr->kernelCh.channelId;
    if (chId >= ACCL_MAX_CHANNELS) {
      *eHandle = NULL;
      return ncclSuccess;
    }

    pthread_mutex_lock(&coll->mutex);
    struct acclKernelChInfo* kch = &coll->kernelCh[chId];
    kch->type = ncclProfileKernelCh;
    kch->parentObj = coll;
    kch->channelId = chId;
    kch->startGpuClk = eDescr->kernelCh.pTimer;
    kch->tsStartUs = acclGetTimeUs();
    coll->nKernelChStarted++;
    pthread_mutex_unlock(&coll->mutex);

    *eHandle = kch;
    return ncclSuccess;
  }

  if (eDescr->type == ncclProfileProxyOp) {
    // parentObj points to our acclCollInfo handle (or NULL for non-coll ops)
    struct acclCollInfo* parentColl = NULL;
    if (eDescr->parentObj) {
      uint64_t parentType = *(uint64_t*)eDescr->parentObj;
      if (parentType == ncclProfileColl) {
        parentColl = (struct acclCollInfo*)eDescr->parentObj;
      }
    }

    struct acclProxyOpInfo* op = acclAllocProxyOp(ctx);
    if (!op) {
      *eHandle = NULL;
      return ncclSuccess;
    }
    op->type = ncclProfileProxyOp;
    op->parentObj = parentColl;
    op->commCtx = ctx;
    if (parentColl) {
      pthread_mutex_lock(&parentColl->mutex);
      parentColl->nProxyOpsStarted++;
      pthread_mutex_unlock(&parentColl->mutex);
    }
    op->channelId = eDescr->proxyOp.channelId;
    op->peer = eDescr->proxyOp.peer;
    op->nSteps = eDescr->proxyOp.nSteps;
    op->chunkSize = eDescr->proxyOp.chunkSize;
    op->isSend = eDescr->proxyOp.isSend;
    op->tsStartUs = acclGetTimeUs();

    *eHandle = op;
    return ncclSuccess;
  }

  if (eDescr->type == ncclProfileProxyStep) {
    // parentObj points to our acclProxyOpInfo handle
    struct acclProxyStepInfo* step = acclAllocProxyStep(ctx);
    if (!step) {
      *eHandle = NULL;
      return ncclSuccess;
    }
    step->type = ncclProfileProxyStep;
    step->parentObj = eDescr->parentObj;
    step->commCtx = ctx;
    // Pin the parent's current tenancy: its slot can be freed and reissued
    // before this step stops, and only the generation separates the two.
    struct acclProxyOpInfo* parentOp = (struct acclProxyOpInfo*)eDescr->parentObj;
    if (parentOp) {
      pthread_mutex_lock(&parentOp->mutex);
      step->parentGen = parentOp->generation;
      pthread_mutex_unlock(&parentOp->mutex);
    }
    step->step = eDescr->proxyStep.step;
    step->tsStartUs = acclGetTimeUs();
    step->lastStateTs = step->tsStartUs;
    step->prevState = -1;

    *eHandle = step;
    return ncclSuccess;
  }

  *eHandle = NULL;
  return ncclSuccess;
}

__hidden ncclResult_t acclPluginStopEvent(void* eHandle) {
  if (!eHandle) return ncclSuccess;

  uint64_t type = *(uint64_t*)eHandle;

  if (type == ncclProfileColl) {
    struct acclCollInfo* coll = (struct acclCollInfo*)eHandle;
    struct acclCommContext* ctx = (struct acclCommContext*)coll->commCtx;
    coll->tsCollStopUs = acclGetTimeUs();

    pthread_mutex_lock(&coll->mutex);
    if (coll->finalized) {
      pthread_mutex_unlock(&coll->mutex);
      return ncclSuccess;
    }
    coll->collStopped = 1;
    int shouldFinalize = acclShouldFinalize(coll);
    pthread_mutex_unlock(&coll->mutex);

    if (shouldFinalize) {
      acclFinalizeAndFree(ctx, coll);
    }
    return ncclSuccess;
  }

  if (type == ncclProfileKernelCh) {
    struct acclKernelChInfo* kch = (struct acclKernelChInfo*)eHandle;
    kch->tsStopUs = acclGetTimeUs();

    struct acclCollInfo* coll = (struct acclCollInfo*)kch->parentObj;
    if (!coll) return ncclSuccess;

    pthread_mutex_lock(&coll->mutex);
    coll->nKernelChCompleted++;
    int shouldFinalize = acclShouldFinalize(coll);
    pthread_mutex_unlock(&coll->mutex);

    if (shouldFinalize) {
      struct acclCommContext* ctx = (struct acclCommContext*)coll->commCtx;
      acclFinalizeAndFree(ctx, coll);
    }
    return ncclSuccess;
  }

  if (type == ncclProfileProxyOp) {
    struct acclProxyOpInfo* op = (struct acclProxyOpInfo*)eHandle;
    op->tsStopUs = acclGetTimeUs();

    struct acclCollInfo* coll = (struct acclCollInfo*)op->parentObj;
    struct acclCommContext* ctx = (struct acclCommContext*)op->commCtx;

    if (coll) {
      pthread_mutex_lock(&coll->mutex);
      if (coll->finalized) {
        pthread_mutex_unlock(&coll->mutex);
        acclFreeProxyOp(ctx, op);
        return ncclSuccess;
      }
      int opIdx = op->poolIdx;
      if (coll->nProxyOps < ACCL_MAX_PROXY_OPS) {
        coll->proxyOpIndices[coll->nProxyOps] = opIdx;
        coll->nProxyOps++;
      } else {
        // The op completed normally; the collective simply has no room to record
        // it, so its timing is lost and the decomposition understates the total.
        __atomic_add_fetch(&ctx->overflowProxyOps, 1, __ATOMIC_SEQ_CST);
        acclFreeProxyOp(ctx, op);
      }
      // Closing the last outstanding op closes the network window. If more ops
      // start afterwards, the next "last" overwrites this snapshot.
      if (coll->nProxyOpsCompleted + 1 == coll->nProxyOpsStarted && coll->tsNetStartUs != 0) {
        coll->nNetDevsStop = acclNetCapture(coll->netStop);
        coll->tsNetStopUs = acclGetTimeUs();
      }
      coll->nProxyOpsCompleted++;
      int shouldFinalize = acclShouldFinalize(coll);
      pthread_mutex_unlock(&coll->mutex);
      if (shouldFinalize) {
        acclFinalizeAndFree(ctx, coll);
      }
    } else {
      acclFreeProxyOp(ctx, op);
    }
    return ncclSuccess;
  }

  if (type == ncclProfileProxyStep) {
    struct acclProxyStepInfo* step = (struct acclProxyStepInfo*)eHandle;
    step->tsStopUs = acclGetTimeUs();
    // Close the last state: no further transition is announced, so the interval
    // from the last transition to the stop belongs to the state still in effect
    // (SendWait on the send side, RecvGPUWait on the recv side).
    acclProxyStepChargeState(step, step->prevState, step->tsStopUs - step->lastStateTs);
    step->prevState = -1;

    // Accumulate step timing into parent proxy op under lock, but only while
    // the op is still the tenant this step started under. The slot is freed
    // from three paths a step knows nothing about, so it can be reissued
    // first; charging a stale step to the new tenant is silent corruption of
    // an unrelated collective's decomposition, so drop it and count it.
    struct acclProxyOpInfo* op = (struct acclProxyOpInfo*)step->parentObj;
    int stale = 0;
    if (op) {
      pthread_mutex_lock(&op->mutex);
      if (op->generation == step->parentGen) {
        op->totalGpuWaitUs += step->gpuWaitUs;
        op->totalPeerWaitUs += step->peerWaitUs;
        op->totalNetworkUs += step->sendWaitUs + step->recvWaitUs;
        op->totalFlushUs += step->flushWaitUs;
        op->totalGpuRecvWaitUs += step->gpuRecvWaitUs;
        op->stepsCompleted++;
      } else {
        stale = 1;
      }
      pthread_mutex_unlock(&op->mutex);
    }

    struct acclCommContext* ctx = (struct acclCommContext*)step->commCtx;
    if (stale) __atomic_add_fetch(&ctx->staleProxySteps, 1, __ATOMIC_SEQ_CST);
    acclFreeProxyStep(ctx, step);
    return ncclSuccess;
  }

  return ncclSuccess;
}

__hidden ncclResult_t acclPluginRecordEventState(void* eHandle,
                                                  ncclProfilerEventState_v5_t eState,
                                                  ncclProfilerEventStateArgs_v5_t* eStateArgs) {
  if (!eHandle) return ncclSuccess;

  uint64_t type = *(uint64_t*)eHandle;

  if (type == ncclProfileKernelCh && (int)eState == (int)ncclProfilerKernelChStop) {
    struct acclKernelChInfo* kch = (struct acclKernelChInfo*)eHandle;
    if (eStateArgs) {
      kch->stopGpuClk = eStateArgs->kernelCh.pTimer;
    }
    return ncclSuccess;
  }

  // ProxyStep state transitions — accumulate time per state
  if (type == ncclProfileProxyStep) {
    struct acclProxyStepInfo* step = (struct acclProxyStepInfo*)eHandle;
    // RCCL signals a state on ENTRY, so the interval that just closed was spent
    // in step->prevState, not eState. Confirmed against the emission sites in
    // src/transport/net.cc and against plugins/profiler/example, whose spans run
    // ts(state) -> ts(next state). The final interval is charged at step stop.
    uint64_t now = acclGetTimeUs();
    acclProxyStepChargeState(step, step->prevState, now - step->lastStateTs);
    step->lastStateTs = now;
    step->prevState = (int)eState;

    // The first step that moves data opens the collective's network window: a
    // send posted to the NIC, or received data landing. Not the proxy op start:
    // RCCL posts the ops of every queued collective at once, so op starts of
    // back-to-back collectives all bunch up before any of them sends.
    int moving = (int)eState == ncclProfilerProxyStepSendWait ||
                 (int)eState == ncclProfilerProxyStepRecvFlushWait ||
                 (int)eState == ncclProfilerProxyStepRecvGPUWait;
    struct acclProxyOpInfo* op = (struct acclProxyOpInfo*)step->parentObj;
    struct acclCollInfo* coll = NULL;
    if (moving && op && gNetCapture) {
      // Same tenancy check as at step stop: the op slot may have been reissued.
      pthread_mutex_lock(&op->mutex);
      if (op->generation == step->parentGen) coll = (struct acclCollInfo*)op->parentObj;
      pthread_mutex_unlock(&op->mutex);
    }
    if (coll && coll->tsNetStartUs == 0) {
      // Captured under the lock so a concurrent finalize can never read a
      // half-written snapshot.
      pthread_mutex_lock(&coll->mutex);
      if (coll->tsNetStartUs == 0 && !coll->finalized) {
        coll->nNetDevsStart = acclNetCapture(coll->netStart);
        if (coll->nNetDevsStart > 0) coll->tsNetStartUs = acclGetTimeUs();
      }
      pthread_mutex_unlock(&coll->mutex);
    }
    return ncclSuccess;
  }

  return ncclSuccess;
}

// ============================================================================
// Plugin export — v5 interface (compatible with RCCL on ROCm)
// ============================================================================
ncclProfiler_v5_t ncclProfiler_v5 = {
  "ACCL-Profiler",
  acclPluginInit,
  acclPluginStartEvent,
  acclPluginStopEvent,
  acclPluginRecordEventState,
  acclPluginFinalize,
};
