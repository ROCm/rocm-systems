/*
 * ACCL Profiler Plugin — Combined GPU kernel + proxy/network decomposition.
 *
 * Subscribes to: Coll, KernelCh, ProxyOp, ProxyStep
 * Output: JSONL with per-collective timing decomposition.
 *
 * Build: see build.sh
 */

#include "accl_profiler.h"
#include <dlfcn.h>
#include <unistd.h>
#include <errno.h>

// Pull in the NCCL profiler v5 types from the RCCL headers.
// We use v5 since the inspector plugin proved it works on ROCm.
// The v5 descriptor is a subset of v6 (no CE/proxyDiag arms),
// but includes all the fields we need.
#include "nccl/profiler.h"

#define __hidden __attribute__((visibility("hidden")))

typedef void (*ncclDebugLogger_t)(int level, unsigned long flags,
                                  const char* file, int line,
                                  const char* fmt, ...);
static ncclDebugLogger_t gLogFn;

#define ACCL_INFO(...)  do { if (gLogFn) gLogFn(4, 0x4000, __func__, __LINE__, __VA_ARGS__); } while(0)
#define ACCL_WARN(...)  do { if (gLogFn) gLogFn(3, 0x4000, __func__, __LINE__, __VA_ARGS__); } while(0)

// Env vars
static size_t gMinMsgSize = 0;  // ACCL_PROFILER_MIN_SIZE_BYTES
static int    gWarmupIters = 5; // ACCL_PROFILER_WARMUP_ITERS

static inline const char* safeStr(const char* s) { return s ? s : ""; }

// ============================================================================
// Collective pool — simple fixed-size pool to avoid malloc in hot path
// ============================================================================
#define ACCL_COLL_POOL_SIZE 256
static struct acclCollInfo gCollPool[ACCL_COLL_POOL_SIZE];
static int gCollPoolUsed[ACCL_COLL_POOL_SIZE];
static pthread_mutex_t gCollPoolMutex = PTHREAD_MUTEX_INITIALIZER;

static struct acclCollInfo* acclAllocColl() {
  pthread_mutex_lock(&gCollPoolMutex);
  for (int i = 0; i < ACCL_COLL_POOL_SIZE; i++) {
    if (!gCollPoolUsed[i]) {
      gCollPoolUsed[i] = 1;
      memset(&gCollPool[i], 0, sizeof(gCollPool[i]));
      pthread_mutex_init(&gCollPool[i].mutex, NULL);
      pthread_mutex_unlock(&gCollPoolMutex);
      return &gCollPool[i];
    }
  }
  pthread_mutex_unlock(&gCollPoolMutex);
  return NULL;
}

static void acclFreeColl(struct acclCollInfo* coll) {
  if (!coll) return;
  pthread_mutex_destroy(&coll->mutex);
  pthread_mutex_lock(&gCollPoolMutex);
  int idx = (int)(coll - gCollPool);
  if (idx >= 0 && idx < ACCL_COLL_POOL_SIZE) {
    gCollPoolUsed[idx] = 0;
  }
  pthread_mutex_unlock(&gCollPoolMutex);
}

// ============================================================================
// ProxyOp pool
// ============================================================================
#define ACCL_PROXY_OP_POOL_SIZE 1024
static struct acclProxyOpInfo gProxyOpPool[ACCL_PROXY_OP_POOL_SIZE];
static int gProxyOpPoolUsed[ACCL_PROXY_OP_POOL_SIZE];
static pthread_mutex_t gProxyOpPoolMutex = PTHREAD_MUTEX_INITIALIZER;

static struct acclProxyOpInfo* acclAllocProxyOp() {
  pthread_mutex_lock(&gProxyOpPoolMutex);
  for (int i = 0; i < ACCL_PROXY_OP_POOL_SIZE; i++) {
    if (!gProxyOpPoolUsed[i]) {
      gProxyOpPoolUsed[i] = 1;
      memset(&gProxyOpPool[i], 0, sizeof(gProxyOpPool[i]));
      pthread_mutex_unlock(&gProxyOpPoolMutex);
      return &gProxyOpPool[i];
    }
  }
  pthread_mutex_unlock(&gProxyOpPoolMutex);
  return NULL;
}

static void acclFreeProxyOp(struct acclProxyOpInfo* op) {
  if (!op) return;
  pthread_mutex_lock(&gProxyOpPoolMutex);
  int idx = (int)(op - gProxyOpPool);
  if (idx >= 0 && idx < ACCL_PROXY_OP_POOL_SIZE) {
    gProxyOpPoolUsed[idx] = 0;
  }
  pthread_mutex_unlock(&gProxyOpPoolMutex);
}

// ============================================================================
// ProxyStep pool
// ============================================================================
#define ACCL_PROXY_STEP_POOL_SIZE 4096
static struct acclProxyStepInfo gProxyStepPool[ACCL_PROXY_STEP_POOL_SIZE];
static int gProxyStepPoolUsed[ACCL_PROXY_STEP_POOL_SIZE];
static pthread_mutex_t gProxyStepPoolMutex = PTHREAD_MUTEX_INITIALIZER;

static struct acclProxyStepInfo* acclAllocProxyStep() {
  pthread_mutex_lock(&gProxyStepPoolMutex);
  for (int i = 0; i < ACCL_PROXY_STEP_POOL_SIZE; i++) {
    if (!gProxyStepPoolUsed[i]) {
      gProxyStepPoolUsed[i] = 1;
      memset(&gProxyStepPool[i], 0, sizeof(gProxyStepPool[i]));
      pthread_mutex_unlock(&gProxyStepPoolMutex);
      return &gProxyStepPool[i];
    }
  }
  pthread_mutex_unlock(&gProxyStepPoolMutex);
  return NULL;
}

static void acclFreeProxyStep(struct acclProxyStepInfo* step) {
  if (!step) return;
  pthread_mutex_lock(&gProxyStepPoolMutex);
  int idx = (int)(step - gProxyStepPool);
  if (idx >= 0 && idx < ACCL_PROXY_STEP_POOL_SIZE) {
    gProxyStepPoolUsed[idx] = 0;
  }
  pthread_mutex_unlock(&gProxyStepPoolMutex);
}

// ============================================================================
// Datatype size helper
// ============================================================================
static int acclDatatypeSize(const char* dt) {
  if (!dt) return 4;
  if (strstr(dt, "int8") || strstr(dt, "Int8") || strstr(dt, "uint8") ||
      strstr(dt, "Uint8") || strstr(dt, "fp8") || strstr(dt, "Fp8"))
    return 1;
  if (strstr(dt, "float16") || strstr(dt, "Float16") ||
      strstr(dt, "bfloat16") || strstr(dt, "Bfloat16"))
    return 2;
  if (strstr(dt, "int64") || strstr(dt, "Int64") || strstr(dt, "uint64") ||
      strstr(dt, "Uint64") || strstr(dt, "float64") || strstr(dt, "Float64"))
    return 8;
  return 4;  // int32, uint32, float32 default
}

// ============================================================================
// BusBW factor (from NCCL conventions)
// ============================================================================
static double acclBusBwFactor(const char* func, int nRanks) {
  if (!func || nRanks <= 1) return 1.0;
  double n = (double)nRanks;
  if (strcmp(func, "AllReduce") == 0)       return 2.0 * (n - 1.0) / n;
  if (strcmp(func, "ReduceScatter") == 0)   return (n - 1.0) / n;
  if (strcmp(func, "AllGather") == 0)       return (n - 1.0) / n;
  if (strcmp(func, "Broadcast") == 0)       return 1.0;
  if (strcmp(func, "Reduce") == 0)          return 1.0;
  if (strcmp(func, "AllToAll") == 0)        return (n - 1.0) / n;
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
    "\"coll_n_channels\":%d,"
    "\"coll_exec_time_us\":%.2f,"
    "\"coll_algobw_gbs\":%.6f,\"coll_busbw_gbs\":%.6f,"
    "\"coll_timing_source\":\"%s\","
    "\"decomposition\":{"
      "\"launch_overhead_us\":%.2f,"
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
    rec->nChannels,
    rec->totalExecUs,
    algoBw, busBw,
    rec->hasGpuTiming ? "gpu_globaltimer" : "cpu_wallclock",
    rec->launchOverheadUs,
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
  rec.rank = ctx->rank;
  rec.nRanks = ctx->nRanks;

  // Kernel timing
  uint64_t firstKernelStart = UINT64_MAX;
  uint64_t lastKernelStop = 0;
  double kernelDurSum = 0;
  double kernelDurMin = 1e18;
  double kernelDurMax = 0;
  int nKernelEvents = 0;
  int hasGpuTiming = 0;

  for (uint32_t ch = 0; ch < coll->nChannels && ch < ACCL_MAX_CHANNELS; ch++) {
    struct acclKernelChInfo* kch = &coll->kernelCh[ch];
    if (kch->tsStartUs == 0) continue;

    double dur;
    if (kch->startGpuClk != 0 && kch->stopGpuClk != 0 && kch->stopGpuClk > kch->startGpuClk) {
      dur = (double)(kch->tsStopUs - kch->tsStartUs);
      hasGpuTiming = 1;
    } else {
      dur = (double)(kch->tsStopUs - kch->tsStartUs);
    }

    if (kch->tsStartUs < firstKernelStart) firstKernelStart = kch->tsStartUs;
    if (kch->tsStopUs > lastKernelStop) lastKernelStop = kch->tsStopUs;

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
    rec.totalExecUs = (double)(lastKernelStop - firstKernelStart);
    rec.launchOverheadUs = (firstKernelStart > coll->tsCollStartUs)
      ? (double)(firstKernelStart - coll->tsCollStartUs) : 0;
  } else {
    rec.totalExecUs = (coll->tsCollStopUs > coll->tsCollStartUs)
      ? (double)(coll->tsCollStopUs - coll->tsCollStartUs) : 0;
  }

  // Proxy decomposition
  double totalGpuWait = 0, totalNetwork = 0, totalPeerWait = 0;
  double totalFlush = 0, totalGpuRecvWait = 0;
  int nSend = 0, nRecv = 0;

  for (int i = 0; i < coll->nProxyOps; i++) {
    struct acclProxyOpInfo* op = &coll->proxyOps[i];
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

// ============================================================================
// Reference counting for collective records
// ============================================================================
static inline void acclCollRef(struct acclCollInfo* coll) {
  pthread_mutex_lock(&coll->mutex);
  coll->refCount++;
  pthread_mutex_unlock(&coll->mutex);
}

// Returns 1 if refcount hit zero and caller should finalize+free
static inline int acclCollDeref(struct acclCollInfo* coll) {
  pthread_mutex_lock(&coll->mutex);
  coll->refCount--;
  int done = (coll->refCount == 0);
  pthread_mutex_unlock(&coll->mutex);
  return done;
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
  if ((env = getenv("ACCL_PROFILER_MIN_SIZE_BYTES")) != NULL)
    gMinMsgSize = (size_t)atol(env);
  if ((env = getenv("ACCL_PROFILER_WARMUP_ITERS")) != NULL)
    gWarmupIters = atoi(env);

  struct acclCommContext* ctx = (struct acclCommContext*)calloc(1, sizeof(*ctx));
  if (!ctx) return ncclSuccess;

  ctx->commHash = commHash;
  ctx->rank = rank;
  ctx->nRanks = nRanks;
  ctx->nNodes = nNodes;
  if (commName) strncpy(ctx->commName, commName, sizeof(ctx->commName) - 1);

  pthread_mutex_init(&ctx->outputMutex, NULL);
  pthread_mutex_init(&ctx->ringMutex, NULL);

  // Open output file
  const char* outDir = getenv("ACCL_PROFILER_OUTPUT_DIR");
  if (!outDir) outDir = "/tmp";

  char hostname[256] = {0};
  gethostname(hostname, sizeof(hostname) - 1);

  snprintf(ctx->outputPath, sizeof(ctx->outputPath),
    "%s/accl_profiler_rank%d_%s.jsonl", outDir, rank, hostname);
  ctx->outputFile = fopen(ctx->outputPath, "a");
  if (!ctx->outputFile) {
    ACCL_WARN("ACCL Profiler: Failed to open %s: %s", ctx->outputPath, strerror(errno));
  }

  *context = ctx;
  *eActivationMask = ncclProfileColl | ncclProfileKernelCh
                   | ncclProfileProxyOp | ncclProfileProxyStep;

  ACCL_INFO("ACCL Profiler: init rank=%d nRanks=%d nNodes=%d "
            "output=%s minSize=%zu warmup=%d",
            rank, nRanks, nNodes, ctx->outputPath, gMinMsgSize, gWarmupIters);
  return ncclSuccess;
}

__hidden ncclResult_t acclPluginFinalize(void* context) {
  struct acclCommContext* ctx = (struct acclCommContext*)context;
  if (!ctx) return ncclSuccess;

  ACCL_INFO("ACCL Profiler: finalize rank=%d output=%s", ctx->rank, ctx->outputPath);

  if (ctx->outputFile) {
    fclose(ctx->outputFile);
    ctx->outputFile = NULL;
  }
  pthread_mutex_destroy(&ctx->outputMutex);
  pthread_mutex_destroy(&ctx->ringMutex);
  free(ctx);
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

    struct acclCollInfo* coll = acclAllocColl();
    if (!coll) {
      *eHandle = NULL;
      return ncclSuccess;
    }
    coll->type = ncclProfileColl;
    coll->refCount = 1; // self-ref
    coll->func = eDescr->coll.func;
    coll->algo = eDescr->coll.algo;
    coll->proto = eDescr->coll.proto;
    coll->seqNumber = eDescr->coll.seqNumber;
    coll->nChannels = (eDescr->coll.nChannels < ACCL_MAX_CHANNELS)
                       ? eDescr->coll.nChannels : ACCL_MAX_CHANNELS;
    coll->msgSizeBytes = msgSize;
    coll->tsCollStartUs = acclGetTimeUs();
    coll->commCtx = ctx;

    // Extra ref if we have channels (kernel completion will deref)
    if (coll->nChannels > 0)
      coll->refCount++;

    *eHandle = coll;
    return ncclSuccess;
  }

  if (eDescr->type == ncclProfileKernelCh) {
    if (eDescr->kernelCh.channelId >= ACCL_MAX_CHANNELS) {
      *eHandle = NULL;
      return ncclSuccess;
    }
    // parentObj is our acclCollInfo handle
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

    pthread_mutex_lock(&coll->mutex);
    struct acclKernelChInfo* kch = &coll->kernelCh[chId];
    kch->type = ncclProfileKernelCh;
    kch->parentObj = coll;
    kch->parentType = ncclProfileColl;
    kch->channelId = chId;
    kch->startGpuClk = eDescr->kernelCh.pTimer;
    kch->tsStartUs = acclGetTimeUs();
    coll->nKernelChStarted++;
    coll->refCount++; // deref'd on KernelCh stop
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

    struct acclProxyOpInfo* op = acclAllocProxyOp();
    if (!op) {
      *eHandle = NULL;
      return ncclSuccess;
    }
    op->type = ncclProfileProxyOp;
    op->parentObj = parentColl;
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
    struct acclProxyStepInfo* step = acclAllocProxyStep();
    if (!step) {
      *eHandle = NULL;
      return ncclSuccess;
    }
    step->type = ncclProfileProxyStep;
    step->parentObj = eDescr->parentObj;
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
    coll->tsCollStopUs = acclGetTimeUs();
    if (acclCollDeref(coll)) {
      acclFinalizeCollective(coll);
      acclFreeColl(coll);
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
    int allDone = (coll->nKernelChCompleted == coll->nKernelChStarted &&
                   coll->nKernelChCompleted == coll->nChannels);
    pthread_mutex_unlock(&coll->mutex);

    int shouldFinalize = 0;
    if (allDone) {
      // Deref the "kernel completion" ref
      shouldFinalize = acclCollDeref(coll);
    }
    // Deref the per-channel ref
    if (acclCollDeref(coll) || shouldFinalize) {
      acclFinalizeCollective(coll);
      acclFreeColl(coll);
    }
    return ncclSuccess;
  }

  if (type == ncclProfileProxyOp) {
    struct acclProxyOpInfo* op = (struct acclProxyOpInfo*)eHandle;
    op->tsStopUs = acclGetTimeUs();

    // Link this proxy op to its parent collective
    struct acclCollInfo* coll = (struct acclCollInfo*)op->parentObj;
    if (coll) {
      pthread_mutex_lock(&coll->mutex);
      if (coll->nProxyOps < ACCL_MAX_PROXY_OPS) {
        memcpy(&coll->proxyOps[coll->nProxyOps], op, sizeof(*op));
        coll->nProxyOps++;
      }
      pthread_mutex_unlock(&coll->mutex);
    }
    acclFreeProxyOp(op);
    return ncclSuccess;
  }

  if (type == ncclProfileProxyStep) {
    struct acclProxyStepInfo* step = (struct acclProxyStepInfo*)eHandle;
    step->tsStopUs = acclGetTimeUs();

    // Accumulate step timing into parent proxy op
    struct acclProxyOpInfo* op = (struct acclProxyOpInfo*)step->parentObj;
    if (op) {
      op->totalGpuWaitUs += step->gpuWaitUs;
      op->totalPeerWaitUs += step->peerWaitUs;
      op->totalNetworkUs += step->sendWaitUs + step->recvWaitUs;
      op->totalFlushUs += step->flushWaitUs;
      op->totalGpuRecvWaitUs += step->gpuRecvWaitUs;
      op->stepCount++;
    }
    acclFreeProxyStep(step);
    return ncclSuccess;
  }

  return ncclSuccess;
}

__hidden ncclResult_t acclPluginRecordEventState(void* eHandle,
                                                  ncclProfilerEventState_v5_t eState,
                                                  ncclProfilerEventStateArgs_v5_t* eStateArgs) {
  if (!eHandle) return ncclSuccess;

  uint64_t type = *(uint64_t*)eHandle;

  // KernelCh stop record — capture GPU stop timestamp
  if (type == ncclProfileKernelCh && (int)eState == (int)ncclProfilerKernelChStop) {
    struct acclKernelChInfo* kch = (struct acclKernelChInfo*)eHandle;
    if (eStateArgs) {
      kch->stopGpuClk = eStateArgs->kernelCh.pTimer;
    }
    // Deref the GPU-timestamp ref (paired with startGpuClk != 0 check)
    // We handle this simply: the stop event will handle the main deref
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
