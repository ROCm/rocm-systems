#ifndef ACCL_PROFILER_H_
#define ACCL_PROFILER_H_

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

// NCCL profiler event type bits (from nccl_profiler.h)
enum {
  ncclProfileGroup          = (1 << 0),
  ncclProfileColl           = (1 << 1),
  ncclProfileP2p            = (1 << 2),
  ncclProfileProxyOp        = (1 << 3),
  ncclProfileProxyStep      = (1 << 4),
  ncclProfileProxyCtrl      = (1 << 5),
  ncclProfileKernelCh       = (1 << 6),
  ncclProfileNetPlugin      = (1 << 7),
};

// NCCL profiler event states we care about
typedef enum {
  ncclProfilerProxyStepSendGPUWait     = 8,
  ncclProfilerProxyStepSendPeerWait_v4 = 20,
  ncclProfilerProxyStepSendWait        = 9,
  ncclProfilerProxyStepRecvWait        = 10,
  ncclProfilerProxyStepRecvFlushWait   = 11,
  ncclProfilerProxyStepRecvGPUWait     = 12,
  ncclProfilerProxyCtrlIdle            = 13,
  ncclProfilerProxyCtrlActive          = 14,
  ncclProfilerKernelChStop             = 22,
} acclProfilerEventState_t;

// Limits
#define ACCL_MAX_CHANNELS      64
#define ACCL_MAX_PROXY_OPS     256
#define ACCL_MAX_STEPS_PER_OP  32
#define ACCL_RING_SIZE         1024

static inline uint64_t acclGetTimeUs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

// Per-channel kernel timing
struct acclKernelChInfo {
  uint64_t type;
  void*    parentObj;
  uint64_t parentType;
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
  uint8_t  channelId;
  int      peer;
  int      nSteps;
  int      chunkSize;
  int      isSend;
  uint64_t tsStartUs;
  uint64_t tsStopUs;
  // Aggregated from steps
  uint64_t totalGpuWaitUs;
  uint64_t totalPeerWaitUs;
  uint64_t totalNetworkUs;   // sendWait + recvWait
  uint64_t totalFlushUs;
  uint64_t totalGpuRecvWaitUs;
  int      stepCount;
};

// Per collective record
struct acclCollInfo {
  uint64_t    type;         // ncclProfileColl
  int         refCount;
  pthread_mutex_t mutex;

  // Collective metadata
  const char* func;
  const char* algo;
  const char* proto;
  uint64_t    seqNumber;
  size_t      msgSizeBytes;
  uint8_t     nChannels;

  // Host timestamps
  uint64_t    tsCollStartUs;
  uint64_t    tsCollStopUs;

  // Kernel channel data
  uint32_t    nKernelChStarted;
  uint32_t    nKernelChCompleted;
  struct acclKernelChInfo kernelCh[ACCL_MAX_CHANNELS];

  // Proxy ops linked to this collective
  int         nProxyOps;
  struct acclProxyOpInfo proxyOps[ACCL_MAX_PROXY_OPS];

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
  uint8_t     nChannels;
  int         rank;
  int         nRanks;

  // Timing decomposition (microseconds)
  double      totalExecUs;
  double      launchOverheadUs;
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

// Ring buffer for completed records
struct acclCompletedRing {
  struct acclCompletedRecord records[ACCL_RING_SIZE];
  int head;
  int tail;
  int count;
};

// Per-communicator context
struct acclCommContext {
  uint64_t    commHash;
  int         rank;
  int         nRanks;
  int         nNodes;
  char        commName[256];

  // Output
  FILE*       outputFile;
  char        outputPath[1024];
  pthread_mutex_t outputMutex;

  // Completed records ring
  struct acclCompletedRing ring;
  pthread_mutex_t ringMutex;

  // Dump thread
  pthread_t   dumpThread;
  int         dumpThreadRunning;
  pthread_mutex_t dumpMutex;
  pthread_cond_t  dumpCond;
};

#endif // ACCL_PROFILER_H_
