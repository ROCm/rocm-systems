/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *
 * NCCL profiler v6 event descriptor (upstream) with RCCL extensions:
 *   - Extended proxyOp (proxy-diag / proxy-trace)
 *   - CE collective / sync / batch descriptor arms
 *************************************************************************/

#ifndef PROFILER_V6_H_
#define PROFILER_V6_H_

#include "profiler_v5.h"

typedef struct {
  uint64_t type;                // event type descriptor
  void* parentObj;              // pointer to the profiler parent object
  int rank;                     // originating rank
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
      /* RCCL: proxy-diag / proxy-trace plugin (zero when unused) */
      uint64_t commHash;
      int64_t opCount;
      int32_t traceFuncIdx;
      int32_t traceProtocol;
      int32_t tracePattern;
      uint32_t traceTotalBytes;
      uint32_t proxyNbytes;
      int rawProxyNsteps;
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

    /* CE (upstream v6) + RCCL */
    struct {
      uint64_t seqNumber;
      const char* func;
      void const* sendBuff;
      void* recvBuff;
      size_t count;
      int root;
      const char* datatype;
      const char* syncStrategy;
      bool intraBatchSync;
      uint32_t batchSize;
      uint32_t numBatches;
      uint32_t ceSeqNum;
      void* stream;
    } ceColl;

    struct {
      bool isComplete;
      int nRanks;
    } ceCollSync;

    struct {
      int numOps;
      size_t totalBytes;
      bool useIntraSync;
    } ceCollBatch;
  };
} ncclProfilerEventDescr_v6_t;

/* RCCL: proxy-diag / proxy-trace counter payloads (v6-only; not part of narrow v5 plugins). */
typedef union {
  struct {
    size_t transSize;
  } proxyStep;

  struct {
    int appendedProxyOps;
  } proxyCtrl;

  struct {
    void* data;
  } netPlugin;

  struct {
    uint64_t pTimer;
  } kernelCh;

  struct {
    uint8_t counterKind;
    uint8_t reserved;
    uint16_t flags; /* bit0: record wall-clock timestamp for this counter */
    int64_t value;
    int64_t value2;
  } proxyDiag;
} ncclProfilerEventStateArgs_v6_t;

typedef struct {
  const char* name;

  ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask, const char* commName, int nNodes, int nranks, int rank, ncclDebugLogger_t logfn);

  ncclResult_t (*startEvent)(void* context, void** eHandle, ncclProfilerEventDescr_v6_t* eDescr);

  ncclResult_t (*stopEvent)(void* eHandle);

  ncclResult_t (*recordEventState)(void* eHandle, ncclProfilerEventState_v6_t eState, ncclProfilerEventStateArgs_v6_t* eStateArgs);

  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v6_t;

#endif // PROFILER_V6_H_
