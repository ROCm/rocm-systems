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

// Per collective record. Also stands in for one launch of a collective that RCCL
// lowered to P2P tasks (isFanout): every task of that call in one kernel plan
// returns this slot as its P2p handle, so its KernelCh and ProxyOp children land
// here exactly as a ring collective's do.
struct acclCollInfo {
  uint64_t    type;         // ncclProfileColl
  int         collStopped;
  int         finalized;
  pthread_mutex_t mutex;
  // Tenancy counter, bumped under mutex on every alloc and never cleared, so a
  // fan-out call can tell whether the slot it last joined is still that launch.
  uint64_t    generation;

  // Fan-out launch bookkeeping. RCCL starts every task of a plan before it
  // stops any, so the first task stop closes the launch to further joins.
  int         isFanout;
  uint32_t    nTasksStarted;
  uint32_t    nTasksStopped;

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

  // Kernel channel data. A fan-out launch can post several KernelCh events on
  // one channel; they share the slot, which then spans the earliest start to the
  // latest stop. Completion counts events, not channels.
  uint32_t    nKernelChExpected;
  uint32_t    nKernelChStarted;
  uint32_t    nKernelChCompleted;
  struct acclKernelChInfo kernelCh[ACCL_MAX_CHANNELS];

  // Proxy ops are folded in here at their stop and their slot released at once,
  // so the op pool holds only ops still in flight.
  int         nProxyOpsStarted;
  int         nProxyOpsCompleted;
  int         nSendOps;
  int         nRecvOps;
  uint64_t    sendGpuWaitUs;
  uint64_t    sendPeerWaitUs;
  uint64_t    sendNetworkUs;
  uint64_t    recvFlushUs;
  uint64_t    recvGpuWaitUs;
  uint64_t    recvNetworkUs;

  // Comm info backpointer
  void*       commCtx;
};

// Returned as the P2pApi handle of every task a fanned-out collective posts, so
// those tasks' P2p events find the call through their parent. Typed P2pApi
// because that is the event it answers for.
struct acclFanoutCall {
  uint64_t    type;           // ncclProfileP2pApi
  const char* func;
  const char* datatype;
  size_t      count;
  int         graphCaptured;
  uint64_t    seqNumber;
  // One reference for the open CollApi bracket and one per task not yet
  // started. A graph-captured call is never released: every replay restarts its
  // tasks against this handle. Guarded by ctx->apiMutex, as are the fields below.
  int         refCount;
  struct acclCollInfo* openLaunch;
  uint64_t    openLaunchGen;
  struct acclFanoutCall* nextFree;
  struct acclFanoutCall* nextAll;
};

// CollApi handle. Thread-local, because API events start and stop on the
// calling thread and do not nest; ordinary collectives therefore cost nothing,
// and an acclFanoutCall is created only when a P2pApi event starts inside.
struct acclCollApiStub {
  uint64_t    type;           // ncclProfileCollApi
  void*       ctx;
  int         open;
  const char* func;
  const char* datatype;
  size_t      count;
  int         graphCaptured;
  struct acclFanoutCall* call;
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
  // Claimed by an owner mid-finalize, so its record may never have been written:
  // the drain leaves it alone, but the file closes before the owner writes.
  uint64_t    inflightCollectives;
  int         poolExhaustedWarned;  // one-shot guard for the pool-exhaustion WARN
  // Proxy-side loss. Written from the proxy thread under three different locks,
  // so these are atomic rather than adopting any one of them.
  uint64_t    droppedProxyOps;      // proxy-op pool was full: this op is unprofiled
  uint64_t    droppedProxySteps;    // proxy-step pool was full: this step is unprofiled
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

  // Peak slots in use, each under its pool's mutex, so a clean run still shows
  // how close it came to dropping.
  int         collPoolInUse, collPoolHighWater;
  int         proxyOpPoolInUse, proxyOpPoolHighWater;
  int         proxyStepPoolInUse, proxyStepPoolHighWater;

  // Fan-out calls: every one allocated (freed with the context) and the free
  // list they are recycled through. RCCL keeps no sequence number for these
  // calls, so fanoutSeq numbers them per communicator.
  pthread_mutex_t          apiMutex;
  struct acclFanoutCall*   fanoutAll;
  struct acclFanoutCall*   fanoutFree;
  uint64_t                 fanoutSeq;
};

#endif
