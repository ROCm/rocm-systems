#ifndef ACCL_PROFILER_H_
#define ACCL_PROFILER_H_

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

// Limits
#define ACCL_MAX_CHANNELS 256
#define ACCL_MAX_PROXY_OPS 256

static inline uint64_t acclGetTimeUs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

// Per-channel kernel timing
struct acclKernelChInfo {
  uint64_t type;
  void*    parentObj;
  uint8_t  channelId;
  uint64_t startGpuClk;
  uint64_t stopGpuClk;
  uint64_t tsStartUs;
  uint64_t tsStopUs;
};

// Per proxy step timing — accumulates time in each state
struct acclProxyStepInfo {
  uint64_t type;
  void*    parentObj;    // points to acclProxyOpInfo
  void*    commCtx;      // owning acclCommContext (for pool free)
  int      step;
  uint64_t tsStartUs;
  uint64_t tsStopUs;
  uint64_t lastStateTs;
  int      prevState;    // state the step is currently IN; -1 before the first transition
  // Accumulated time in each proxy step state (us)
  uint64_t gpuWaitUs;
  uint64_t peerWaitUs;
  uint64_t sendWaitUs;
  uint64_t recvWaitUs;
  uint64_t flushWaitUs;
  uint64_t gpuRecvWaitUs;
};

// Per proxy op (one per channel per send/recv direction)
struct acclProxyOpInfo {
  uint64_t type;
  void*    parentObj;    // points to acclCollInfo (the coll event handle)
  void*    commCtx;      // owning acclCommContext (for pool free)
  uint8_t  channelId;
  int      peer;
  int      nSteps;
  int      chunkSize;
  int      isSend;
  uint64_t tsStartUs;
  uint64_t tsStopUs;
  // Aggregated from steps (protected by mutex)
  pthread_mutex_t mutex;
  uint64_t totalGpuWaitUs;
  uint64_t totalPeerWaitUs;
  uint64_t totalNetworkUs;   // sendWait + recvWait
  uint64_t totalFlushUs;
  uint64_t totalGpuRecvWaitUs;
  int      stepsCompleted;
};

// Per collective record
struct acclCollInfo {
  uint64_t    type;         // ncclProfileColl
  int         collStopped;
  int         finalized;
  pthread_mutex_t mutex;

  // Collective metadata
  const char* func;
  const char* algo;
  const char* proto;
  uint64_t    seqNumber;
  size_t      msgSizeBytes;
  // Completion target for acclShouldFinalize(). Wider than the v5 ABI's uint8_t
  // so it can hold ACCL_MAX_CHANNELS; see acclPluginStartEvent() for why a
  // reported 0 is promoted rather than taken at face value.
  uint16_t    nChannels;
  uint8_t     nChannelsRaw;  // value exactly as the v5 descriptor reported it

  // Host timestamps
  uint64_t    tsCollStartUs;
  uint64_t    tsCollStopUs;

  // Kernel channel data
  uint32_t    nKernelChStarted;
  uint32_t    nKernelChCompleted;
  struct acclKernelChInfo kernelCh[ACCL_MAX_CHANNELS];

  // Proxy ops linked to this collective (indices into ctx->proxyOpPool)
  int         nProxyOps;
  int         nProxyOpsStarted;
  int         nProxyOpsCompleted;
  int         proxyOpIndices[ACCL_MAX_PROXY_OPS];

  // Comm info backpointer
  void*       commCtx;
};

// Schema for the "decomposition" JSON object: one row per field, driving the
// acclCompletedRecord members, the JSON keys and the fprintf argument list from
// one list so a key can never drift onto another field's value. XE marks the
// last row, which emits no trailing comma. Rows are (C type, JSON key, printf
// conversion, acclCompletedRecord member); the row order is the emission order.
//
// The macro body cannot carry // comments, since the line-continuation backslash
// would be swallowed by them, so the field meanings live here:
//   gpu_kernel_avg_us      avg kernel duration across channels
//   proxy_gpu_wait_us      proxy waiting for the GPU to produce data
//   proxy_network_us       actual network send/recv
//   proxy_peer_wait_us     waiting for the remote FIFO
//   proxy_flush_us         GDR flush
//   proxy_gpu_recv_wait_us proxy waiting for the GPU to consume
// The n_* counts are totals over all proxy ops. Each proxy_* row is a mean over
// the op class that can produce it — the send-only states over n_send_ops, the
// recv-only ones over n_recv_ops — so it is a per-channel cost for the usual
// one-send-plus-one-recv-per-channel ring, not a per-op cost. proxy_network_us
// spans both classes and is the sum of the two per-class means.
#define ACCL_DECOMP_FIELDS(X, XE)                                    \
  X (double, enqueue_to_kernel_us,   "%.2f", enqueueToKernelUs)      \
  X (double, gpu_kernel_avg_us,      "%.2f", gpuKernelUs)            \
  X (double, gpu_kernel_min_us,      "%.2f", gpuKernelMinUs)         \
  X (double, gpu_kernel_max_us,      "%.2f", gpuKernelMaxUs)         \
  X (double, proxy_gpu_wait_us,      "%.2f", proxyGpuWaitUs)         \
  X (double, proxy_network_us,       "%.2f", proxyNetworkUs)         \
  X (double, proxy_peer_wait_us,     "%.2f", proxyPeerWaitUs)        \
  X (double, proxy_flush_us,         "%.2f", proxyFlushUs)           \
  X (double, proxy_gpu_recv_wait_us, "%.2f", proxyGpuRecvWaitUs)     \
  X (int,    n_proxy_ops,            "%d",   nProxyOps)              \
  X (int,    n_send_ops,             "%d",   nSendOps)               \
  XE(int,    n_recv_ops,             "%d",   nRecvOps)

#define ACCL_DECOMP_DECL(ctype, key, fmt, member) ctype member;

// Completed record for output
struct acclCompletedRecord {
  // Metadata
  const char* func;
  const char* algo;
  const char* proto;
  uint64_t    seqNumber;
  size_t      msgSizeBytes;
  uint16_t    nChannels;
  uint8_t     nChannelsRaw;
  int         rank;
  int         nRanks;

  // Timing decomposition (microseconds)
  double      totalExecUs;
  ACCL_DECOMP_FIELDS(ACCL_DECOMP_DECL, ACCL_DECOMP_DECL)

  // Per-channel kernel events
  struct {
    uint8_t  channelId;
    uint64_t startGpuClk;
    uint64_t stopGpuClk;
    uint64_t durationUs;
  } kernelEvents[ACCL_MAX_CHANNELS];
  int nKernelEvents;

  // Timing source
  int hasGpuTiming;
};

// Pool sizes (per communicator)
#define ACCL_COLL_POOL_SIZE 256
#define ACCL_PROXY_OP_POOL_SIZE 1024
#define ACCL_PROXY_STEP_POOL_SIZE 4096

// Per-communicator context (owns all pools)
struct acclCommContext {
  int         refCount;
  uint64_t    droppedCollectives;   // never allocated a slot: pool was full
  uint64_t    leakedCollectives;    // allocated but never finalized; freed by the drain
  int         poolExhaustedWarned;  // one-shot guard for the pool-exhaustion WARN
  // Proxy-side loss. Written from the proxy thread under three different locks,
  // so these are atomic rather than adopting any one of them.
  uint64_t    droppedProxyOps;      // proxy-op pool was full: this op is unprofiled
  uint64_t    droppedProxySteps;    // proxy-step pool was full: this step is unprofiled
  uint64_t    overflowProxyOps;     // op completed but the coll already held ACCL_MAX_PROXY_OPS
  int         proxyOpPoolWarned;    // one-shot guards, as for the coll pool above
  int         proxyStepPoolWarned;
  uint64_t    commHash;
  // ACCL_PROFILER_MIN_SIZE_BYTES as read when THIS communicator was created.
  // Per-comm, not process-global: a later communicator created after the
  // variable changed must not retroactively re-filter this one, and the
  // minSize= this comm echoed in its init log has to keep describing it.
  size_t      minMsgSize;
  int         rank;
  int         nRanks;
  int         nNodes;
  char        commName[256];

  // Output
  FILE*       outputFile;
  char        outputPath[1024];
  pthread_mutex_t outputMutex;

  // Per-comm pools
  struct acclCollInfo      collPool[ACCL_COLL_POOL_SIZE];
  int                      collPoolUsed[ACCL_COLL_POOL_SIZE];
  pthread_mutex_t          collPoolMutex;

  struct acclProxyOpInfo   proxyOpPool[ACCL_PROXY_OP_POOL_SIZE];
  int                      proxyOpPoolUsed[ACCL_PROXY_OP_POOL_SIZE];
  pthread_mutex_t          proxyOpPoolMutex;

  struct acclProxyStepInfo proxyStepPool[ACCL_PROXY_STEP_POOL_SIZE];
  int                      proxyStepPoolUsed[ACCL_PROXY_STEP_POOL_SIZE];
  pthread_mutex_t          proxyStepPoolMutex;
};

#endif
