#ifndef ACCL_PROFILER_H_
#define ACCL_PROFILER_H_

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "net_telemetry.h"  // RcclTelemetrySwSnapshot

// Limits
#define ACCL_MAX_CHANNELS 256
#define ACCL_MAX_PROXY_OPS 256
// NICs captured per network snapshot. A node has at most a handful; devices
// past this are not recorded (their traffic is still in the telemetry JSON).
#define ACCL_NET_MAX_DEVS 16

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
  // Parent's tenancy of its pool slot, read at start. The slot can be freed and
  // reissued before this step stops; comparing at stop is what keeps a stale
  // step out of the new tenant's accumulators.
  uint64_t parentGen;
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
  int      poolIdx;      // this slot's index in ctx->proxyOpPool
  uint8_t  channelId;
  int      peer;
  int      nSteps;
  int      chunkSize;
  int      isSend;
  uint64_t tsStartUs;
  uint64_t tsStopUs;
  // Aggregated from steps (protected by mutex)
  pthread_mutex_t mutex;
  // Tenancy counter, bumped under mutex on every alloc and never cleared.
  // Outside the mutex bytes, so acclAllocProxyOp's clear steps over both.
  uint64_t generation;
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

  // Network window: from the first proxy step that moves data (a send posted or
  // received data landing) to the last proxy op stop.
  // RCCL telemetry SW counters are captured at both ends (see
  // acclNetCapture()); the difference is this collective's NIC traffic.
  uint64_t    tsNetStartUs;
  uint64_t    tsNetStopUs;
  int         nNetDevsStart;
  int         nNetDevsStop;
  RcclTelemetrySwSnapshot netStart[ACCL_NET_MAX_DEVS];
  RcclTelemetrySwSnapshot netStop[ACCL_NET_MAX_DEVS];

  // Comm info backpointer
  void*       commCtx;
};

// Schema for the "decomposition" JSON object. Drives the acclCompletedRecord
// members, the JSON keys and the fprintf argument list from one list, so a key
// cannot drift onto another field's value. Rows are (C type, JSON key, printf
// conversion, member) in emission order; XE marks the last, which emits no
// trailing comma. Field meanings live here because the macro body cannot carry
// // comments -- the line-continuation backslash would be swallowed:
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

  // Host CLOCK_MONOTONIC timestamps (us), for lining records up with the RCCL
  // telemetry hw_samples series. execStart/execEnd span the first kernel or
  // data-moving proxy step to the last kernel or proxy op stop; 0 when nothing
  // was observed.
  uint64_t    tsEnqueueUs;
  uint64_t    tsExecStartUs;
  uint64_t    tsExecEndUs;

  // Per-NIC SW counter deltas over the network window (see acclCollInfo).
  uint64_t    tsNetStartUs;
  uint64_t    tsNetStopUs;
  int         nNetDevs;
  RcclTelemetrySwSnapshot netDelta[ACCL_NET_MAX_DEVS];
};

// Pool capacities (per communicator). Each pool starts with one chunk and adds
// another when it runs out, up to the capacity. RCCL can queue hundreds of
// collectives ahead of the GPU and every queued one already holds its proxy
// ops, so a fixed pool sized for the common case drops the deep-queue ones.
#define ACCL_COLL_POOL_CHUNK 256
#define ACCL_COLL_POOL_SIZE (ACCL_COLL_POOL_CHUNK * 16)
#define ACCL_PROXY_OP_POOL_CHUNK 1024
#define ACCL_PROXY_OP_POOL_SIZE (ACCL_PROXY_OP_POOL_CHUNK * 64)
#define ACCL_PROXY_STEP_POOL_CHUNK 4096
#define ACCL_PROXY_STEP_POOL_SIZE (ACCL_PROXY_STEP_POOL_CHUNK * 16)
#define ACCL_POOL_MAX_CHUNKS 64

// Slots are addressed by index: chunk * chunkSlots + offset. Chunks never move
// or shrink, so a slot pointer handed to RCCL stays valid for the context's
// life. Free slots form a LIFO list through next[], so alloc and free stay O(1)
// however far the pool has grown. Every field is guarded by mutex.
struct acclPool {
  pthread_mutex_t mutex;
  size_t   elemSize;
  int      chunkSlots;
  int      maxChunks;
  long     slotMutexOff;   // offset of each slot's pthread_mutex_t, or -1
  int      nChunks;
  int      freeHead;       // -1 when every slot of every chunk is in use
  char*    chunks[ACCL_POOL_MAX_CHUNKS];
  int*     used[ACCL_POOL_MAX_CHUNKS];
  int*     next[ACCL_POOL_MAX_CHUNKS];
};

// Per-communicator context (owns all pools)
struct acclCommContext {
  int         refCount;
  uint64_t    droppedCollectives;   // never allocated a slot: pool was full
  uint64_t    leakedCollectives;    // allocated but never finalized; freed by the drain
  // Claimed by an owner mid-finalize, so its record may never have been written:
  // the drain leaves it alone, but the file closes before the owner writes.
  uint64_t    inflightCollectives;
  int         poolExhaustedWarned;  // one-shot guard for the pool-exhaustion WARN
  // Proxy-side loss. Written from the proxy thread under three different locks,
  // so these are atomic rather than adopting any one of them.
  uint64_t    droppedProxyOps;      // proxy-op pool was full: this op is unprofiled
  uint64_t    droppedProxySteps;    // proxy-step pool was full: this step is unprofiled
  uint64_t    overflowProxyOps;     // op completed but the coll already held ACCL_MAX_PROXY_OPS
  uint64_t    staleProxySteps;      // stopped after its parent op's slot was reissued
  // Still held at finalize: their stop never arrived. Counted there, never
  // released there -- see acclPluginFinalize. Written only on that thread.
  uint64_t    outstandingProxyOps;
  uint64_t    outstandingProxySteps;
  int         proxyOpPoolWarned;    // one-shot guards, as for the coll pool above
  int         proxyStepPoolWarned;
  uint64_t    commHash;
  // ACCL_PROFILER_MIN_SIZE_BYTES as read when THIS communicator was created.
  // Per-comm, not process-global: a comm created later must not retroactively
  // re-filter an earlier one, whose init log already echoed its own minSize=.
  size_t      minMsgSize;
  int         rank;
  int         nRanks;
  int         nNodes;
  char        commName[256];
  char        hostname[256];

  // Output
  FILE*       outputFile;
  char        outputPath[1024];
  pthread_mutex_t outputMutex;

  // Per-comm pools
  struct acclPool collPool;       // struct acclCollInfo
  struct acclPool proxyOpPool;    // struct acclProxyOpInfo
  struct acclPool proxyStepPool;  // struct acclProxyStepInfo
};

#endif
