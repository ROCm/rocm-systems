/*
 * ACCL Profiler Plugin — Combined GPU kernel + proxy/network decomposition.
 *
 * Subscribes to: Coll, KernelCh, ProxyOp, ProxyStep
 * Output: JSONL with per-collective timing decomposition.
 *
 * Build: see README.md or CMakeLists.txt
 */

#include "accl_profiler.h"

#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

#include "accl_shim.h"

#define __hidden __attribute__((visibility("hidden")))

static ncclDebugLogger_t gLogFn;

#define ACCL_INFO(...)  do { if (gLogFn) gLogFn(4, 0x4000, __func__, __LINE__, __VA_ARGS__); } while(0)
#define ACCL_WARN(...)  do { if (gLogFn) gLogFn(3, 0x4000, __func__, __LINE__, __VA_ARGS__); } while(0)

// Env vars
static size_t gMinMsgSize = 0;  // ACCL_PROFILER_MIN_SIZE_BYTES

static inline const char* safeStr(const char* s) { return s ? s : ""; }

// Forward declarations for cross-referenced pool functions
static void acclFreeProxyOp(struct acclCommContext* ctx, struct acclProxyOpInfo* op);

static void acclFreeCollProxyOps(struct acclCommContext* ctx,
                                 struct acclCollInfo* coll) {
  for (int i = 0; i < coll->nProxyOps; i++) {
    int idx = coll->proxyOpIndices[i];
    if (idx >= 0 && idx < ACCL_PROXY_OP_POOL_SIZE) {
      acclFreeProxyOp(ctx, &ctx->proxyOpPool[idx]);
    }
  }
}

// ============================================================================
// Per-communicator pool allocators
// ============================================================================

static struct acclCollInfo* acclAllocColl(struct acclCommContext* ctx) {
  pthread_mutex_lock(&ctx->collPoolMutex);
  for (int i = 0; i < ACCL_COLL_POOL_SIZE; i++) {
    if (!ctx->collPoolUsed[i]) {
      ctx->collPoolUsed[i] = 1;
      memset(&ctx->collPool[i], 0, sizeof(ctx->collPool[i]));
      pthread_mutex_init(&ctx->collPool[i].mutex, NULL);
      __atomic_add_fetch(&ctx->refCount, 1, __ATOMIC_SEQ_CST);
      pthread_mutex_unlock(&ctx->collPoolMutex);
      return &ctx->collPool[i];
    }
  }
  // Pool is full. Warn once: a per-drop WARN emits one line per collective for
  // the rest of the run, which buries the very message it is trying to deliver.
  // The end-of-run summary carries the totals.
  ctx->droppedCollectives++;
  int firstExhaustion = !ctx->poolExhaustedWarned;
  ctx->poolExhaustedWarned = 1;
  pthread_mutex_unlock(&ctx->collPoolMutex);
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
  pthread_mutex_destroy(&coll->mutex);
  pthread_mutex_lock(&ctx->collPoolMutex);
  int idx = (int)(coll - ctx->collPool);
  if (idx >= 0 && idx < ACCL_COLL_POOL_SIZE) {
    ctx->collPoolUsed[idx] = 0;
  }
  pthread_mutex_unlock(&ctx->collPoolMutex);
  __atomic_sub_fetch(&ctx->refCount, 1, __ATOMIC_SEQ_CST);
}

static struct acclProxyOpInfo* acclAllocProxyOp(struct acclCommContext* ctx) {
  pthread_mutex_lock(&ctx->proxyOpPoolMutex);
  for (int i = 0; i < ACCL_PROXY_OP_POOL_SIZE; i++) {
    if (!ctx->proxyOpPoolUsed[i]) {
      ctx->proxyOpPoolUsed[i] = 1;
      memset(&ctx->proxyOpPool[i], 0, sizeof(ctx->proxyOpPool[i]));
      pthread_mutex_init(&ctx->proxyOpPool[i].mutex, NULL);
      pthread_mutex_unlock(&ctx->proxyOpPoolMutex);
      return &ctx->proxyOpPool[i];
    }
  }
  pthread_mutex_unlock(&ctx->proxyOpPoolMutex);
  ACCL_WARN("ACCL Profiler: proxy op pool exhausted (%d slots)", ACCL_PROXY_OP_POOL_SIZE);
  return NULL;
}

static void acclFreeProxyOp(struct acclCommContext* ctx, struct acclProxyOpInfo* op) {
  if (!op) return;
  pthread_mutex_destroy(&op->mutex);
  pthread_mutex_lock(&ctx->proxyOpPoolMutex);
  int idx = (int)(op - ctx->proxyOpPool);
  if (idx >= 0 && idx < ACCL_PROXY_OP_POOL_SIZE) {
    ctx->proxyOpPoolUsed[idx] = 0;
  }
  pthread_mutex_unlock(&ctx->proxyOpPoolMutex);
}

static struct acclProxyStepInfo* acclAllocProxyStep(struct acclCommContext* ctx) {
  pthread_mutex_lock(&ctx->proxyStepPoolMutex);
  for (int i = 0; i < ACCL_PROXY_STEP_POOL_SIZE; i++) {
    if (!ctx->proxyStepPoolUsed[i]) {
      ctx->proxyStepPoolUsed[i] = 1;
      memset(&ctx->proxyStepPool[i], 0, sizeof(ctx->proxyStepPool[i]));
      pthread_mutex_unlock(&ctx->proxyStepPoolMutex);
      return &ctx->proxyStepPool[i];
    }
  }
  pthread_mutex_unlock(&ctx->proxyStepPoolMutex);
  ACCL_WARN("ACCL Profiler: proxy step pool exhausted (%d slots)", ACCL_PROXY_STEP_POOL_SIZE);
  return NULL;
}

static void acclFreeProxyStep(struct acclCommContext* ctx, struct acclProxyStepInfo* step) {
  if (!step) return;
  pthread_mutex_lock(&ctx->proxyStepPoolMutex);
  int idx = (int)(step - ctx->proxyStepPool);
  if (idx >= 0 && idx < ACCL_PROXY_STEP_POOL_SIZE) {
    ctx->proxyStepPoolUsed[idx] = 0;
  }
  pthread_mutex_unlock(&ctx->proxyStepPoolMutex);
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
    "{\"header\":{\"rank\":%d,\"n_ranks\":%d},"
    "\"coll_perf\":{"
    "\"coll\":\"%s\",\"coll_sn\":%lu,\"coll_msg_size_bytes\":%zu,"
    "\"coll_algo\":\"%s\",\"coll_proto\":\"%s\","
    "\"coll_n_channels\":%d,\"coll_n_channels_reported\":%d,"
    "\"coll_exec_time_us\":%.2f,"
    "\"coll_algobw_gbs\":%.6f,\"coll_busbw_gbs\":%.6f,"
    "\"coll_timing_source\":\"%s\","
    "\"decomposition\":{"
      "\"enqueue_to_kernel_us\":%.2f,"
      "\"gpu_kernel_avg_us\":%.2f,"
      "\"gpu_kernel_min_us\":%.2f,"
      "\"gpu_kernel_max_us\":%.2f,"
      "\"proxy_gpu_wait_us\":%.2f,"
      "\"proxy_network_us\":%.2f,"
      "\"proxy_peer_wait_us\":%.2f,"
      "\"proxy_flush_us\":%.2f,"
      "\"proxy_gpu_recv_wait_us\":%.2f,"
      "\"n_proxy_ops\":%d,"
      "\"n_send_ops\":%d,"
      "\"n_recv_ops\":%d"
    "},",
    rec->rank, rec->nRanks,
    safeStr(rec->func), (unsigned long)rec->seqNumber, rec->msgSizeBytes,
    safeStr(rec->algo), safeStr(rec->proto),
    rec->nChannels, rec->nChannelsRaw,
    rec->totalExecUs,
    algoBw, busBw,
    rec->hasGpuTiming ? "gpu_globaltimer" : "cpu_wallclock",
    rec->enqueueToKernelUs,
    rec->gpuKernelUs,
    rec->gpuKernelMinUs,
    rec->gpuKernelMaxUs,
    rec->proxyGpuWaitUs,
    rec->proxyNetworkUs,
    rec->proxyPeerWaitUs,
    rec->proxyFlushUs,
    rec->proxyGpuRecvWaitUs,
    rec->nProxyOps,
    rec->nSendOps,
    rec->nRecvOps
  );

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

  // Proxy decomposition
  double totalGpuWait = 0, totalNetwork = 0, totalPeerWait = 0;
  double totalFlush = 0, totalGpuRecvWait = 0;
  int nSend = 0, nRecv = 0;

  for (int i = 0; i < coll->nProxyOps; i++) {
    int opIdx = coll->proxyOpIndices[i];
    if (opIdx < 0 || opIdx >= ACCL_PROXY_OP_POOL_SIZE) continue;
    struct acclProxyOpInfo* op = &ctx->proxyOpPool[opIdx];
    totalGpuWait += (double)op->totalGpuWaitUs;
    totalNetwork += (double)op->totalNetworkUs;
    totalPeerWait += (double)op->totalPeerWaitUs;
    totalFlush += (double)op->totalFlushUs;
    totalGpuRecvWait += (double)op->totalGpuRecvWaitUs;
    if (op->isSend) nSend++; else nRecv++;
  }

  int nOps = coll->nProxyOps;
  rec.proxyGpuWaitUs = nOps > 0 ? totalGpuWait / nOps : 0;
  rec.proxyNetworkUs = nOps > 0 ? totalNetwork / nOps : 0;
  rec.proxyPeerWaitUs = nOps > 0 ? totalPeerWait / nOps : 0;
  rec.proxyFlushUs = nOps > 0 ? totalFlush / nOps : 0;
  rec.proxyGpuRecvWaitUs = nOps > 0 ? totalGpuRecvWait / nOps : 0;
  rec.nProxyOps = nOps;
  rec.nSendOps = nSend;
  rec.nRecvOps = nRecv;

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
// Plugin callbacks
// ============================================================================

__hidden ncclResult_t acclPluginInit(void** context, uint64_t commHash,
                                     int* eActivationMask, const char* commName,
                                     int nNodes, int nRanks, int rank,
                                     ncclDebugLogger_t logfn) {
  gLogFn = logfn;

  const char* env;
  if ((env = getenv("ACCL_PROFILER_MIN_SIZE_BYTES")) != NULL) {
    gMinMsgSize = (size_t)atol(env);
  }

  struct acclCommContext* ctx = (struct acclCommContext*)calloc(1, sizeof(*ctx));
  if (!ctx) return ncclSuccess;

  ctx->refCount = 1;
  ctx->commHash = commHash;
  ctx->rank = rank;
  ctx->nRanks = nRanks;
  ctx->nNodes = nNodes;
  if (commName) strncpy(ctx->commName, commName, sizeof(ctx->commName) - 1);

  pthread_mutex_init(&ctx->outputMutex, NULL);
  pthread_mutex_init(&ctx->collPoolMutex, NULL);
  pthread_mutex_init(&ctx->proxyOpPoolMutex, NULL);
  pthread_mutex_init(&ctx->proxyStepPoolMutex, NULL);

  // Open output file (write mode — fresh file per init to avoid stale data)
  const char* outDir = getenv("ACCL_PROFILER_OUTPUT_DIR");
  if (!outDir) outDir = "/tmp";

  char hostname[256] = {0};
  gethostname(hostname, sizeof(hostname) - 1);

  mkdir(outDir, 0755);

  snprintf(ctx->outputPath, sizeof(ctx->outputPath),
    "%s/accl_profiler_rank%d_%s_pid%d_0x%lx.jsonl",
    outDir, rank, hostname, (int)getpid(), (unsigned long)commHash);
  ctx->outputFile = fopen(ctx->outputPath, "w");
  if (!ctx->outputFile) {
    ACCL_WARN("ACCL Profiler: Failed to open %s: %s", ctx->outputPath, strerror(errno));
  }

  *context = ctx;
  *eActivationMask = ncclProfileColl | ncclProfileKernelCh
                   | ncclProfileProxyOp | ncclProfileProxyStep;

  ACCL_INFO("ACCL Profiler: init rank=%d nRanks=%d nNodes=%d "
            "output=%s minSize=%zu",
            rank, nRanks, nNodes, ctx->outputPath, gMinMsgSize);
  return ncclSuccess;
}

__hidden ncclResult_t acclPluginFinalize(void* context) {
  struct acclCommContext* ctx = (struct acclCommContext*)context;
  if (!ctx) return ncclSuccess;

  ACCL_INFO("ACCL Profiler: finalize rank=%d output=%s", ctx->rank, ctx->outputPath);

  // Drain any coll slots still in use (orphaned by teardown).
  pthread_mutex_lock(&ctx->collPoolMutex);
  for (int i = 0; i < ACCL_COLL_POOL_SIZE; i++) {
    if (ctx->collPoolUsed[i]) {
      struct acclCollInfo* coll = &ctx->collPool[i];
      if (!coll->finalized) {
        // Never reached its completion predicate — teardown skipped one or more
        // of its kernel-channel events. Count it so the loss is visible; no
        // record is emitted for it.
        ctx->leakedCollectives++;
        acclFreeCollProxyOps(ctx, coll);
      }
      coll->finalized = 1;
      pthread_mutex_destroy(&coll->mutex);
      ctx->collPoolUsed[i] = 0;
      __atomic_sub_fetch(&ctx->refCount, 1, __ATOMIC_SEQ_CST);
    }
  }
  pthread_mutex_unlock(&ctx->collPoolMutex);

  // Write drop summary and close output before tearing down mutexes.
  if (ctx->outputFile) {
    // Emit the summary unconditionally, including on a clean run: a consumer
    // that finds no summary line cannot distinguish "nothing was lost" from
    // "the process died before finalize".
    int complete = (ctx->droppedCollectives == 0 && ctx->leakedCollectives == 0);
    fprintf(ctx->outputFile,
      "{\"summary\":{\"dropped_collectives\":%lu,\"leaked_collectives\":%lu,"
      "\"pool_size\":%d,\"complete\":%s}}\n",
      (unsigned long)ctx->droppedCollectives,
      (unsigned long)ctx->leakedCollectives,
      ACCL_COLL_POOL_SIZE, complete ? "true" : "false");
    fflush(ctx->outputFile);
    if (!complete) {
      ACCL_WARN("ACCL Profiler: rank=%d output INCOMPLETE — %lu collectives dropped "
                "(pool exhausted), %lu slots leaked (teardown-skipped kernel events)",
                ctx->rank, (unsigned long)ctx->droppedCollectives,
                (unsigned long)ctx->leakedCollectives);
    }
    fclose(ctx->outputFile);
    ctx->outputFile = NULL;
  }

  // Tear down context only after refcount reaches zero.
  if (__atomic_sub_fetch(&ctx->refCount, 1, __ATOMIC_SEQ_CST) == 0) {
    pthread_mutex_destroy(&ctx->outputMutex);
    pthread_mutex_destroy(&ctx->collPoolMutex);
    pthread_mutex_destroy(&ctx->proxyOpPoolMutex);
    pthread_mutex_destroy(&ctx->proxyStepPoolMutex);
    free(ctx);
  }
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
    if (msgSize < gMinMsgSize) {
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
    step->step = eDescr->proxyStep.step;
    step->tsStartUs = acclGetTimeUs();
    step->lastStateTs = step->tsStartUs;

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
      int opIdx = (int)(op - ctx->proxyOpPool);
      if (coll->nProxyOps < ACCL_MAX_PROXY_OPS) {
        coll->proxyOpIndices[coll->nProxyOps] = opIdx;
        coll->nProxyOps++;
      } else {
        acclFreeProxyOp(ctx, op);
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

    // Accumulate step timing into parent proxy op under lock
    struct acclProxyOpInfo* op = (struct acclProxyOpInfo*)step->parentObj;
    if (op) {
      pthread_mutex_lock(&op->mutex);
      op->totalGpuWaitUs += step->gpuWaitUs;
      op->totalPeerWaitUs += step->peerWaitUs;
      op->totalNetworkUs += step->sendWaitUs + step->recvWaitUs;
      op->totalFlushUs += step->flushWaitUs;
      op->totalGpuRecvWaitUs += step->gpuRecvWaitUs;
      op->stepsCompleted++;
      pthread_mutex_unlock(&op->mutex);
    }

    struct acclCommContext* ctx = (struct acclCommContext*)step->commCtx;
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
    uint64_t now = acclGetTimeUs();
    uint64_t elapsed = now - step->lastStateTs;
    step->lastStateTs = now;

    switch ((int)eState) {
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
