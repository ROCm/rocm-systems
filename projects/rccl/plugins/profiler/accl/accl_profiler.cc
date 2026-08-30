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
#include <stddef.h>
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
  for (int i = 0; i < ACCL_COLL_POOL_SIZE; i++) {
    pthread_mutex_destroy(&ctx->collPool[i].mutex);
  }
  for (int i = 0; i < ACCL_PROXY_OP_POOL_SIZE; i++) {
    pthread_mutex_destroy(&ctx->proxyOpPool[i].mutex);
  }
  pthread_mutex_destroy(&ctx->outputMutex);
  pthread_mutex_destroy(&ctx->collPoolMutex);
  pthread_mutex_destroy(&ctx->proxyOpPoolMutex);
  pthread_mutex_destroy(&ctx->proxyStepPoolMutex);
  free(ctx);
}

static struct acclCollInfo* acclAllocColl(struct acclCommContext* ctx) {
  pthread_mutex_lock(&ctx->collPoolMutex);
  for (int i = 0; i < ACCL_COLL_POOL_SIZE; i++) {
    if (!ctx->collPoolUsed[i]) {
      ctx->collPoolUsed[i] = 1;
      // Clear the slot around the mutex, never through it. The mutex is created
      // once in acclPluginInit and outlives every tenancy; a stale KernelCh
      // start can be inside pthread_mutex_lock on it at this instant, since that
      // path does not take collPoolMutex, so writing its bytes here — by memset
      // or by a save/restore struct copy — is a data race either way. POSIX also
      // does not define copying a pthread_mutex_t at all.
      struct acclCollInfo* slot = &ctx->collPool[i];
      const size_t muOff = offsetof(struct acclCollInfo, mutex);
      const size_t muEnd = muOff + sizeof(slot->mutex);
      memset(slot, 0, muOff);
      memset((char*)slot + muEnd, 0, sizeof(*slot) - muEnd);
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
  pthread_mutex_lock(&ctx->collPoolMutex);
  int idx = (int)(coll - ctx->collPool);
  if (idx >= 0 && idx < ACCL_COLL_POOL_SIZE) {
    ctx->collPoolUsed[idx] = 0;
  }
  pthread_mutex_unlock(&ctx->collPoolMutex);
  acclCtxUnref(ctx);
}

static struct acclProxyOpInfo* acclAllocProxyOp(struct acclCommContext* ctx) {
  pthread_mutex_lock(&ctx->proxyOpPoolMutex);
  for (int i = 0; i < ACCL_PROXY_OP_POOL_SIZE; i++) {
    if (!ctx->proxyOpPoolUsed[i]) {
      ctx->proxyOpPoolUsed[i] = 1;
      __atomic_add_fetch(&ctx->refCount, 1, __ATOMIC_SEQ_CST);
      // Clear the slot around the mutex, never through it. The mutex is created
      // once in acclPluginInit and outlives every tenancy; a stale ProxyStep
      // stop can be inside pthread_mutex_lock on it at this instant, since that
      // path does not take proxyOpPoolMutex, so writing its bytes here — by
      // memset or by a save/restore struct copy — is a data race either way.
      struct acclProxyOpInfo* slot = &ctx->proxyOpPool[i];
      const size_t muOff = offsetof(struct acclProxyOpInfo, mutex);
      const size_t muEnd = muOff + sizeof(slot->mutex);
      memset(slot, 0, muOff);
      memset((char*)slot + muEnd, 0, sizeof(*slot) - muEnd);
      pthread_mutex_unlock(&ctx->proxyOpPoolMutex);
      return &ctx->proxyOpPool[i];
    }
  }
  pthread_mutex_unlock(&ctx->proxyOpPoolMutex);
  __atomic_add_fetch(&ctx->droppedProxyOps, 1, __ATOMIC_SEQ_CST);
  if (__atomic_exchange_n(&ctx->proxyOpPoolWarned, 1, __ATOMIC_SEQ_CST) == 0) {
    ACCL_WARN("ACCL Profiler: proxy op pool exhausted (%d slots). Proxy timing for this "
              "communicator is now INCOMPLETE. Further drops are counted in the "
              "end-of-run summary only.", ACCL_PROXY_OP_POOL_SIZE);
  }
  return NULL;
}

static void acclFreeProxyOp(struct acclCommContext* ctx, struct acclProxyOpInfo* op) {
  if (!op) return;
  // The mutex is not destroyed here. A ProxyStep stop still locks it after the
  // op's slot has been released, so destroying per free leaves that path locking
  // a destroyed mutex once the slot is reissued. It is destroyed once, in
  // acclCtxUnref, when the last context reference drops.
  pthread_mutex_lock(&ctx->proxyOpPoolMutex);
  int idx = (int)(op - ctx->proxyOpPool);
  if (idx >= 0 && idx < ACCL_PROXY_OP_POOL_SIZE) {
    ctx->proxyOpPoolUsed[idx] = 0;
  }
  pthread_mutex_unlock(&ctx->proxyOpPoolMutex);
  acclCtxUnref(ctx);
}

static struct acclProxyStepInfo* acclAllocProxyStep(struct acclCommContext* ctx) {
  pthread_mutex_lock(&ctx->proxyStepPoolMutex);
  for (int i = 0; i < ACCL_PROXY_STEP_POOL_SIZE; i++) {
    if (!ctx->proxyStepPoolUsed[i]) {
      ctx->proxyStepPoolUsed[i] = 1;
      __atomic_add_fetch(&ctx->refCount, 1, __ATOMIC_SEQ_CST);
      memset(&ctx->proxyStepPool[i], 0, sizeof(ctx->proxyStepPool[i]));
      pthread_mutex_unlock(&ctx->proxyStepPoolMutex);
      return &ctx->proxyStepPool[i];
    }
  }
  pthread_mutex_unlock(&ctx->proxyStepPoolMutex);
  __atomic_add_fetch(&ctx->droppedProxySteps, 1, __ATOMIC_SEQ_CST);
  if (__atomic_exchange_n(&ctx->proxyStepPoolWarned, 1, __ATOMIC_SEQ_CST) == 0) {
    ACCL_WARN("ACCL Profiler: proxy step pool exhausted (%d slots). Proxy timing for this "
              "communicator is now INCOMPLETE. Further drops are counted in the "
              "end-of-run summary only.", ACCL_PROXY_STEP_POOL_SIZE);
  }
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
  acclCtxUnref(ctx);
}

// Charge `elapsed` to the bucket of the state the step was IN over that interval.
// RCCL announces a proxy-step state on ENTRY (src/transport/net.cc:1783,1849,1877,
// 2059,2092,2187), so the interval that just closed belongs to `state`, the
// previously announced one, not to the state being entered.  See the note above
// the switch in acclPluginRecordEventState.
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
    "{\"header\":{\"rank\":%d,\"n_ranks\":%d},"
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
    rec->rank, rec->nRanks,
    safeStr(rec->func), (unsigned long)rec->seqNumber, rec->msgSizeBytes,
    safeStr(rec->algo), safeStr(rec->proto),
    rec->nChannels, rec->nChannelsRaw,
    rec->totalExecUs,
    algoBw, busBw,
    rec->hasGpuTiming ? "gpu_globaltimer" : "cpu_wallclock"
    ACCL_DECOMP_FIELDS(ACCL_DECOMP_ARG, ACCL_DECOMP_ARG)
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

  // Proxy decomposition. Each component is averaged over the ops that can
  // contribute to it, not over every proxy op: a send op only ever passes
  // through the SendGPUWait/SendPeerWait/SendWait states and a recv op only
  // through RecvWait/RecvFlushWait/RecvGPUWait, so dividing a one-sided total
  // by nProxyOps scales it by that class's share of the op mix. A ring
  // collective posts one send and one recv op per channel, so the per-class
  // means below are per-channel costs, which is the scale gpu_kernel_avg_us is
  // already on and the scale accl_report.py's classifier compares against.
  double sendGpuWait = 0, sendPeerWait = 0, sendNetwork = 0;
  double recvFlush = 0, recvGpuWait = 0, recvNetwork = 0;
  int nSend = 0, nRecv = 0;

  for (int i = 0; i < coll->nProxyOps; i++) {
    int opIdx = coll->proxyOpIndices[i];
    if (opIdx < 0 || opIdx >= ACCL_PROXY_OP_POOL_SIZE) continue;
    struct acclProxyOpInfo* op = &ctx->proxyOpPool[opIdx];
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
  for (int i = 0; i < ACCL_COLL_POOL_SIZE; i++) {
    pthread_mutex_init(&ctx->collPool[i].mutex, NULL);
  }
  for (int i = 0; i < ACCL_PROXY_OP_POOL_SIZE; i++) {
    pthread_mutex_init(&ctx->proxyOpPool[i].mutex, NULL);
  }
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

  // fopen() below reports only ENOENT on the assembled filename, which does not
  // say the directory is the problem, so keep this errno to name the real cause.
  int dirErrno = (acclMkdirRecursive(outDir) == 0) ? 0 : errno;

  int pathLen = snprintf(ctx->outputPath, sizeof(ctx->outputPath),
    "%s/accl_profiler_rank%d_%s_pid%d_0x%lx.jsonl",
    outDir, rank, hostname, (int)getpid(), (unsigned long)commHash);
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
      // Take the claim under the lock every writer uses, not under
      // collPoolMutex. A slot already claimed belongs to a thread between
      // acclShouldFinalize and acclFreeColl; it releases the slot and its
      // reference itself, so touching it here double-releases both. Reading and
      // setting in one critical section leaves no window for a writer to claim
      // a slot this drain has decided to release.
      pthread_mutex_lock(&coll->mutex);
      int claimed = coll->finalized;
      coll->finalized = 1;
      pthread_mutex_unlock(&coll->mutex);
      if (claimed) continue;
      // Never reached its completion predicate — teardown skipped one or more of
      // its kernel-channel events. Count it so the loss is visible; no record is
      // emitted for it.
      ctx->leakedCollectives++;
      acclFreeCollProxyOps(ctx, coll);
      ctx->collPoolUsed[i] = 0;
      __atomic_sub_fetch(&ctx->refCount, 1, __ATOMIC_SEQ_CST);
    }
  }
  pthread_mutex_unlock(&ctx->collPoolMutex);

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
    // A run is complete only if nothing was lost anywhere. Proxy loss leaves the
    // decomposition understated with no marker on the affected records, so it
    // has to clear this flag too.
    // Records go out through fprintf/fflush whose results are discarded on the
    // hot path; the stream's sticky error flag is the one place a write failure
    // (ENOSPC, EIO) is still visible, so fold it into the verdict here rather
    // than branching per record.
    int writeError = ferror(ctx->outputFile) != 0;
    int complete = (ctx->droppedCollectives == 0 && ctx->leakedCollectives == 0 &&
                    dOps == 0 && dSteps == 0 && oOps == 0 && !writeError);
    fprintf(ctx->outputFile,
      "{\"summary\":{\"dropped_collectives\":%lu,\"leaked_collectives\":%lu,"
      "\"dropped_proxy_ops\":%lu,\"dropped_proxy_steps\":%lu,"
      "\"overflow_proxy_ops\":%lu,"
      "\"coll_pool_size\":%d,\"proxy_op_pool_size\":%d,\"proxy_step_pool_size\":%d,"
      "\"max_proxy_ops_per_coll\":%d,\"complete\":%s}}\n",
      (unsigned long)ctx->droppedCollectives,
      (unsigned long)ctx->leakedCollectives,
      (unsigned long)dOps, (unsigned long)dSteps, (unsigned long)oOps,
      ACCL_COLL_POOL_SIZE, ACCL_PROXY_OP_POOL_SIZE, ACCL_PROXY_STEP_POOL_SIZE,
      ACCL_MAX_PROXY_OPS, complete ? "true" : "false");
    fflush(ctx->outputFile);
    if (!complete) {
      ACCL_WARN("ACCL Profiler: rank=%d output INCOMPLETE: %lu collectives dropped "
                "(coll pool exhausted), %lu slots leaked (teardown-skipped kernel "
                "events), %lu proxy ops dropped, %lu proxy steps dropped, %lu proxy "
                "ops discarded (more than %d on one collective)",
                ctx->rank, (unsigned long)ctx->droppedCollectives,
                (unsigned long)ctx->leakedCollectives,
                (unsigned long)dOps, (unsigned long)dSteps, (unsigned long)oOps,
                ACCL_MAX_PROXY_OPS);
    }
    if (writeError) {
      // Say it separately: when writes are failing the summary above may not
      // reach disk at all, so the log is the only channel left.
      ACCL_WARN("ACCL Profiler: rank=%d hit a write error on %s; the records on "
                "disk are truncated", ctx->rank, ctx->outputPath);
    }
  }
  pthread_mutex_unlock(&ctx->outputMutex);

  // Drop the init reference; outstanding proxy ops/steps hold their own.
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
      int opIdx = (int)(op - ctx->proxyOpPool);
      if (coll->nProxyOps < ACCL_MAX_PROXY_OPS) {
        coll->proxyOpIndices[coll->nProxyOps] = opIdx;
        coll->nProxyOps++;
      } else {
        // The op completed normally; the collective simply has no room to record
        // it, so its timing is lost and the decomposition understates the total.
        __atomic_add_fetch(&ctx->overflowProxyOps, 1, __ATOMIC_SEQ_CST);
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
    // Close the last state: no further transition is announced, so the interval
    // from the last transition to the stop belongs to the state still in effect
    // (SendWait on the send side, RecvGPUWait on the recv side).
    acclProxyStepChargeState(step, step->prevState, step->tsStopUs - step->lastStateTs);
    step->prevState = -1;

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
    // RCCL signals a proxy-step state when the step ENTERS it, so the interval
    // that just closed was spent in step->prevState, not in eState.  Verified
    // against the emission sites in src/transport/net.cc and against the
    // reference consumer plugins/profiler/example, whose chrome-trace spans run
    // ts(state) -> ts(next state) (print_event.cc:118-152).  The final interval
    // is charged in the ProxyStep stop path.
    uint64_t now = acclGetTimeUs();
    acclProxyStepChargeState(step, step->prevState, now - step->lastStateTs);
    step->lastStateTs = now;
    step->prevState = (int)eState;
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
