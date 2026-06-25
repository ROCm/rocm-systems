/* Minimal NCCL profiler types needed by accl-profiler plugin.
 * Sourced from RCCL's nccl_profiler.h + profiler_v5.h */

#ifndef ACCL_NCCL_PROFILER_H_
#define ACCL_NCCL_PROFILER_H_

#include <stdint.h>
#include <stddef.h>

typedef int ncclResult_t;
#define ncclSuccess 0

typedef int ncclDataType_t;
enum {
  ncclInt8 = 0, ncclChar = 0,
  ncclUint8 = 1,
  ncclInt32 = 2, ncclInt = 2,
  ncclUint32 = 3,
  ncclInt64 = 4,
  ncclUint64 = 5,
  ncclFloat16 = 6, ncclHalf = 6,
  ncclFloat32 = 7, ncclFloat = 7,
  ncclFloat64 = 8, ncclDouble = 8,
  ncclBfloat16 = 9,
  ncclFloat8e4m3 = 10,
  ncclFloat8e5m2 = 11,
  ncclNumTypes = 12
};

typedef void (*ncclDebugLogger_t)(int level, unsigned long flags,
                                  const char* file, int line,
                                  const char* fmt, ...);

typedef unsigned long ncclPid_t;

// Profiler v5 event descriptor (same layout as RCCL)
typedef struct {
  uint64_t type;
  void* parentObj;
  int rank;
  union {
    struct {
      bool graphCaptured;
      int groupDepth;
    } groupApi;
    struct {
      const char* func;
      size_t count;
      const char* datatype;
      int root;
      void* stream;
      bool graphCaptured;
    } collApi;
    struct {
      const char* func;
      size_t count;
      const char* datatype;
      void* stream;
      bool graphCaptured;
    } p2pApi;
    struct {
      void* stream;
    } kernelLaunch;
    struct {
      uint64_t seqNumber;
      const char* func;
      void const* sendBuff;
      void* recvBuff;
      size_t count;
      int root;
      const char* datatype;
      uint8_t nChannels;
      uint8_t nWarps;
      const char* algo;
      const char* proto;
      void* parentGroup;
    } coll;
    struct {
      const char* func;
      void* buff;
      const char* datatype;
      size_t count;
      int peer;
      uint8_t nChannels;
      void* parentGroup;
    } p2p;
    struct {
      ncclPid_t pid;
      uint8_t channelId;
      int peer;
      int nSteps;
      int chunkSize;
      int isSend;
    } proxyOp;
    struct {
      int step;
    } proxyStep;
    struct {
      uint8_t channelId;
      uint64_t pTimer;
    } kernelCh;
    struct {
      int64_t id;
      void* data;
    } netPlugin;
  };
} ncclProfilerEventDescr_v5_t;

typedef enum {
  ncclProfilerProxyStepSendGPUWait_     = 8,
  ncclProfilerProxyStepSendWait_        = 9,
  ncclProfilerProxyStepRecvWait_        = 10,
  ncclProfilerProxyStepRecvFlushWait_   = 11,
  ncclProfilerProxyStepRecvGPUWait_     = 12,
  ncclProfilerProxyStepSendPeerWait_v4_ = 20,
  ncclProfilerKernelChStop_             = 22,
} ncclProfilerEventState_v5_t;

typedef union {
  struct { size_t transSize; } proxyStep;
  struct { int appendedProxyOps; } proxyCtrl;
  struct { void* data; } netPlugin;
  struct { uint64_t pTimer; } kernelCh;
} ncclProfilerEventStateArgs_v5_t;

typedef struct {
  const char* name;
  ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask,
                       const char* commName, int nNodes, int nranks, int rank,
                       ncclDebugLogger_t logfn);
  ncclResult_t (*startEvent)(void* context, void** eHandle,
                              ncclProfilerEventDescr_v5_t* eDescr);
  ncclResult_t (*stopEvent)(void* eHandle);
  ncclResult_t (*recordEventState)(void* eHandle,
                                    ncclProfilerEventState_v5_t eState,
                                    ncclProfilerEventStateArgs_v5_t* eStateArgs);
  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v5_t;

#endif // ACCL_NCCL_PROFILER_H_
