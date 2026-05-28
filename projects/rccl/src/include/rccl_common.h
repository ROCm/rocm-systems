/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef RCCL_COMMON_H_
#define RCCL_COMMON_H_
#include "nccl_common.h"
#include "nccl.h"
#include "param.h"
#include "core.h"
typedef enum RcclTunableColls {
  RCCL_UNSUPPORTED_TUNABLE = -1,
  RCCL_RS_TUNABLE = 0,    // reduce_scatter index
  RCCL_AG_TUNABLE = 1,    // all_gather index
  RCCL_AR_TUNABLE = 2,    // all_reduce index
  RCCL_RE_TUNABLE = 3,    // reduce index
  RCCL_BR_TUNABLE = 4,    // broadcast index
  RCCL_TUNABLE_COLLS = 5  // LL/LL64/LL128 tunable collectives count
} rcclTunableIndex_t;

#define CHAN_THRESHOLDS_UNDEFINED 0
#define RCCL_CHANNELS_TUNABLE_ENTRIES 9 // 2,4,8,16,32,40,48,56,64 channels

#define RCCL_LL_LIMITS_UNDEFINED 0
#define RCCL_PROTOCOL_ENTRY_SIZE 4
#define RCCL_PROTOCOL_MIN_IDX 0
#define RCCL_PROTOCOL_MAX_IDX 1
#define RCCL_PROTOCOL_FACTOR_IDX 2
#define RCCL_PROTOCOL_THREAD_THRESHOLD_IDX 3

#define RCCL_SINGLE_NODE_MAX_NTHREADS 256
#define RCCL_GFX950_MAX_NTHREADS 512  // for Simple and LL64/LL128 gfx950
#define RCCL_DEFAULT_MAX_NTHREADS 256 // for Simple and LL64/LL128 other archs
#define RCCL_LL_MAX_NTHREADS 256
#define RCCL_P2P_MAX_NTHREADS 256

typedef enum {
  RCCL_VALUE_UNSET = -2,
  RCCL_VALUE_INVALID = -1
} rcclValueState_t;

typedef enum {
  RCCL_DIRECT_ALLGATHER = NCCL_NUM_ALGORITHMS, // Direct AllGather
  RCCL_MSCCL,
  RCCL_MSCCLPP,
  RCCL_ALGO_COUNT
} rcclAddonAlgos_t;

#ifdef RCCL_EXPOSE_STATIC
#define RCCL_STATIC_EXPOSE_CHECK()
#else
#define RCCL_STATIC_EXPOSE_CHECK() \
  do { \
    WARN("Attempting to use internal logic while required static functions are not exposed. Rebuild with RCCL_EXPOSE_STATIC enabled"); \
    return ncclInvalidUsage; \
  } while (0)
#endif

inline rcclTunableIndex_t rcclGetTunableIndex(ncclFunc_t const& func) {
  switch (func) {
    case ncclFuncReduceScatter:
      return RCCL_RS_TUNABLE;
    case ncclFuncAllGather:
      return RCCL_AG_TUNABLE;
    case ncclFuncAllReduce:
      return RCCL_AR_TUNABLE;
    case ncclFuncReduce:
      return RCCL_RE_TUNABLE;
    case ncclFuncBroadcast:
      return RCCL_BR_TUNABLE;
    default:
      return RCCL_UNSUPPORTED_TUNABLE; // Invalid or unsupported function
  }
}

inline size_t rcclGetSizePerRank(ncclFunc_t const& func, size_t const& nBytes, int const& nRanks) {
  // Normalize the comparison to sizePerRank as this is essentially what matters in determining protocol choice for the impacted collectives
  // For AG, this is the send size per rank
  // For RS, this is the recv size per rank
  // For AR, this is the send/recv size per rank
  return (func == ncclFuncReduceScatter || func == ncclFuncAllGather || func == ncclFuncBroadcast || func == ncclFuncReduce) ? nBytes / nRanks : nBytes;
}
ncclResult_t rcclOverrideChannels(struct ncclComm* comm, ncclFunc_t coll, size_t nBytes, int& nc);
ncclResult_t rcclGetAlgoProtoIndex(const char *envStr, const char* algoProtoString[], int nEntries, int& result);
ncclResult_t rcclOverrideProtocol(const char* ncclProtoStr[], float table[][NCCL_NUM_PROTOCOLS], struct ncclTaskColl* info);
ncclResult_t rcclOverrideAlgorithm(const char* ncclAlgoStr[], float table[][NCCL_NUM_PROTOCOLS], struct ncclTaskColl* info);
void rcclUpdateCollectiveProtocol(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info);
void rcclUpdateThreadThreshold(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info, int& threadThreshold);
void rcclSetPipelining(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info);
void rcclGetMaxNthreads(struct ncclComm* comm, int maxNthreads[]);
void rcclOptThreadBlockSize(struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes, int& nThreads);
void rcclSetDefaultBuffSizes(struct ncclComm* comm, int defaultBuffSizes[]);
NCCL_API(ncclResult_t, rcclGetAlgoInfo, struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType, int collNetSupport, int nvlsSupport, int numPipeOps, int* algo, int* protocol, int* maxChannels);
NCCL_API(ncclResult_t, rcclGetAlgoName, int algo, const char** algoName);
NCCL_API(ncclResult_t, rcclGetProtocolName, int protocol, const char** algoName);
bool rcclUseAllGatherDirect(struct ncclComm* comm, size_t& msgSize);
void rcclSetPxn(struct ncclComm* comm,  int& rcclPxnDisable);
void rcclSetP2pNetChunkSize(struct ncclComm* comm,  int& rcclP2pNetChunkSize);
ncclResult_t rcclFuncMaxSendRecvCount(ncclFunc_t func, int nRanks, size_t count, size_t& maxCount);
ncclResult_t commSetUnrollFactor(struct ncclComm* comm);
bool validHsaScratchEnvSetting(const char*hsaScratchEnv, int hipRuntimeVersion, int firmwareVersion, const char* archName);
int parseFirmwareVersion();
bool rcclIsArchSupportedForFunc(struct ncclTaskColl* info, char const* archName);
#endif
