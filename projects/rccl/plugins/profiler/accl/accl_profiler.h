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
  double      enqueueToKernelUs;
  double      gpuKernelUs;        // avg kernel duration across channels
  double      gpuKernelMinUs;
  double      gpuKernelMaxUs;

  // Proxy decomposition (aggregated across all proxy ops)
  double      proxyGpuWaitUs;     // proxy waiting for GPU to produce data
  double      proxyNetworkUs;     // actual network send/recv
  double      proxyPeerWaitUs;    // waiting for remote FIFO
  double      proxyFlushUs;       // GDR flush
  double      proxyGpuRecvWaitUs; // proxy waiting for GPU to consume
  int         nProxyOps;
  int         nSendOps;
  int         nRecvOps;

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
  uint64_t    commHash;
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
