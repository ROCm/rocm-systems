/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__

#define NCCL_TESTS_VERSION "2.19.6"

#include "rccl/rccl.h"
// nccl_device.h provides the device-API public types referenced below
// (ncclCommProperties_t, full ncclDevCommRequirements, NCCL_GIN_TYPE_NONE, ...).
// As of NCCL 2.29 these types are referenced unconditionally by the test engine
// signature and per-collective DevCommRequirements helpers, so the include must
// not require -DENABLE_DEVICE_API on 2.29+.
#if (defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)) \
    || NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
#include "nccl_device.h"
#if defined(RCCL_DEVCOMM_EXPECT_FABRIC_FIELDS)
#include <cstddef>
// Fails if common.cu compiled against /opt/rocm headers that lack the fabric
// DDA tail (240-byte ncclDevComm vs librccl 288-byte memset).
static_assert(offsetof(struct ncclDevComm, ginFabricPeerScratch) < sizeof(struct ncclDevComm),
              "ncclDevComm is missing ginFabricPeerScratch; rccl_common include path is picking the wrong headers");
#endif
#endif
#include <stdio.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <thread>
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif
#include "nccl1_compat.h"
#include "rccl_compat.h"  // Weak symbols forward declarations
#include "timer.h"
#include "os.h"
#include <string>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <map>

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if( err != cudaSuccess ) {                        \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test CUDA failure %s:%d '%s'\n",    \
         hostname,                                  \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    return testCudaError;                           \
  }                                                 \
} while(0)

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,13,0)
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test NCCL failure %s:%d "           \
           "'%s / %s'\n",                           \
           hostname,__FILE__,__LINE__,              \
           ncclGetErrorString(res),                 \
           ncclGetLastError(NULL));                 \
    return testNcclError;                           \
  }                                                 \
} while(0)
#else
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test NCCL failure %s:%d '%s'\n",    \
         hostname,                                  \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    return testNcclError;                           \
  }                                                 \
} while(0)
#endif

typedef enum {
  testSuccess = 0,
  testInternalError = 1,
  testCudaError = 2,
  testNcclError = 3,
  testTimeout = 4,
  testNotImplemented = 5,
  testInvalidUsage = 6,
  testNumResults = 7, // Must be last
} testResult_t;

// Relay errors up and trace
#define TESTCHECK(cmd) do {                         \
  testResult_t r = cmd;                             \
  if (r!= testSuccess) {                            \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf(" .. %s pid %d: Test failure %s:%d\n",   \
         hostname, ncclTestGetPid(),                \
        __FILE__,__LINE__);                         \
    return r;                                       \
  }                                                 \
} while(0)

struct testColl {
  const char name[20];
  void (*getCollByteCount)(
      size_t *sendcount, size_t *recvcount, size_t *paramcount,
      size_t *sendInplaceOffset, size_t *recvInplaceOffset,
      size_t count, size_t eltSize, int nranks);
  testResult_t (*initData)(struct threadArgs* args, ncclDataType_t type,
      ncclRedOp_t op, int root, int rep, int in_place);
  void (*getBw)(size_t count, size_t typesize, double sec, double* algBw, double* busBw, int nranks);
  testResult_t (*runColl)(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset,
      size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int implIndex, void* bias);
  testResult_t (*getAlgoProtoChannels)(ncclComm_t comm, size_t count, ncclDataType_t type, int* algo, int* proto, int* nchannels);
  testResult_t (*getSymkInfo)(ncclComm_t comm, size_t count, ncclDataType_t type, ncclRedOp_t op, int* algo, int* proto, int* nchannels);
  // Reports the actual backend RCCL will dispatch (CE / DDA / symmetric / kernel),
  // op/buffer/graph-capture aware, via rcclGetCollImplInfo. NULL when the collective
  // does not wire it or the library lacks the symbol (older librccl).
  testResult_t (*getCollImplInfo)(ncclComm_t comm, size_t count, ncclDataType_t type, ncclRedOp_t op,
      const void* sendbuff, void* recvbuff, int graphCapturing, int* algo, int* proto, int* nchannels);
  // Optional device-side (in-kernel wall_clock64) timing hook. Non-null only for
  // collectives that implement it (AllToAll, AllGather). Driven by BenchTime via
  // --device_timing (0=off, 1=augment, 2=device-time-only):
  //   - outDeltaSec == nullptr (mode 1): prints an extra device-only
  //     latency/busbw line alongside the normal graph/hipEvent numbers (report,
  //     not replace).
  //   - outDeltaSec != nullptr (mode 2): stores the measured per-iteration
  //     device latency (seconds, max across ranks) in *outDeltaSec; BenchTime
  //     then reports that as THE metric, having skipped the graph/hipEvent
  //     timed loop.
  // Other collectives leave it nullptr via aggregate initialization, so no other
  // struct needs to change.
  testResult_t (*deviceTime)(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec);
};
extern struct testColl allReduceTest;
extern struct testColl allGatherTest;
extern struct testColl reduceScatterTest;
extern struct testColl broadcastTest;
extern struct testColl reduceTest;
extern struct testColl alltoAllTest;

class Reporter {
  public:
    Reporter(std::string fileName, std::string outputFormat);
    ~Reporter() { if (_outputValid) { _out.close(); } };
    void setParameters(const size_t numCycle, const char* name, const char* typeName, const char* opName);
    void addResult(int gpusPerRank, int ranksPerNode, int totalRanks, size_t numBytes, int inPlace, double timeUsec, double algBw, double busBw, int64_t wrongElts = -1);
    void writeFile();

  private:
    bool isMainThread();
    template<typename T> std::pair<std::string, std::string> makeValueKeyPair(T v, std::string k) { return std::make_pair(std::to_string(v), k); };
    template <> std::pair<std::string, std::string> makeValueKeyPair<std::string>(std::string v, std::string k) { return std::make_pair("\"" + v + "\"", k); };

    bool _outputValid = false;
    std::ofstream _out;
    std::string _outputFormat;
    size_t _numCycle = 0;
    std::string _collectiveName;
    std::string _typeName;
    std::string _opName;
    std::vector<std::vector<std::pair<std::string, std::string>>> _outputData;
};

struct testEngine {
  void (*getBuffSize)(size_t *sendcount, size_t *recvcount, size_t count, int nranks);
  testResult_t (*runTest)(struct threadArgs* args, int root, ncclDataType_t type,
      const char* typeName, ncclRedOp_t op, const char* opName);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
  /* Optional; called from initComms after common fields are set on ncclConfig_t. */
  void (*initCommConfig)(ncclConfig_t* config);
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
  testResult_t (*getDevCommRequirements)(int deviceImpl, ncclDevCommRequirements* reqs, ncclComm_t comm);
#elif defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  bool (*getDevCommRequirements)(int deviceImpl, ncclDevCommRequirements* reqs);
#endif
};

extern struct testEngine ncclTestEngine;

struct threadArgs {
  size_t nbytes;
  size_t minbytes;
  size_t maxbytes;
  size_t stepbytes;
  size_t stepfactor;

  int totalProcs;
  int nProcs;
  int proc;
  int nThreads;
  int thread;
  int nGpus;
  int* gpus;
  int localRank;
  int enable_out_of_place;
  int enable_in_place;
  int enable_cache_flush;
  int enable_rotating_tensor;
  void** sendbuffs;
  size_t sendBytes;
  size_t sendInplaceOffset;
  void** recvbuffs;
  size_t recvInplaceOffset;
  ncclUniqueId ncclId;
  ncclComm_t* comms;
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  ncclDevComm* devComms;
#endif
  cudaStream_t* streams;
  void** bias;
  cudaEvent_t* events;
  float* ms;

  void** expected;
  size_t expectedBytes;
  int* errors;
  double* bw;
  int* bw_count;

  int reportErrors;

  struct testColl* collTest;

  Reporter* reporter;

  int64_t* initGpuMem;
  int64_t* bufferMemory;
  int64_t* devMemUsed;

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
  void** sendRegHandles;
  void** recvRegHandles;
  void** biasRegHandles;
#endif
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,7)
  // Mode-1 device-timing augment line (#[a2a-devtime]); buffered here and
  // flushed after writeBenchmarkLineTerminator so it does not split the row.
  char devtimeAugmentLine[512];
#endif
};

typedef testResult_t (*threadFunc_t)(struct threadArgs* args);
struct testThread {
  std::thread thread;
  threadFunc_t func;
  struct threadArgs args;
  testResult_t ret;
};

// Provided by common.cu
extern void Barrier(struct threadArgs* args);
extern testResult_t testStreamSynchronize(int ngpus, cudaStream_t* streams, ncclComm_t* comms);
// Inter-thread/process reduce: average 0=bcast(r0),1=avg,2=min,3=max,4=sum.
// Serializes MPI to the last thread (MPI is MPI_THREAD_SINGLE) and writes the
// combined value back to every thread. Instantiated for double / long long.
template<typename T> void Allreduce(struct threadArgs* args, T* value, int average);
extern testResult_t TimeTest(struct threadArgs* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op,  const char* opName, int root);
extern testResult_t InitDataReduce(void* data, const size_t count, const size_t offset, ncclDataType_t type, ncclRedOp_t op, const uint64_t seed, const int nranks);
extern testResult_t InitDataApplyBias(void* expected, void* bias, const size_t count, const size_t offset, ncclDataType_t type, ncclRedOp_t op);
extern testResult_t InitData(void* data, const size_t count, size_t offset, ncclDataType_t type, ncclRedOp_t op, const uint64_t seed, const int nranks, const int rank);
extern testResult_t AllocateBuffs(void **sendbuff, size_t sendBytes, void **recvbuff, size_t recvBytes, void **expected, size_t nbytes, void **bias);

static void getHostName(char* hostname, int maxlen) {
  ncclTestGetHostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '\0') {
      return;
    }
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)       [Linux]
 * $(hostname)$(MachineGuid from registry)                 [Windows]
 *
 */
static uint64_t getHostHash(const char* hostname) {
  return ncclTestGetHostHash(hostname);
}

#if NCCL_MAJOR >= 2 && RCCL_BFLOAT16 == 1
#define HAVE_BF16 1
#else
#define HAVE_BF16 0
#endif
#if NCCL_MAJOR >= 2 && RCCL_FLOAT8 == 1
#define HAVE_FP8 1
#else
#define HAVE_FP8 0
#endif

#if NCCL_MAJOR >= 2
  #if defined(__CUDA_BF16_TYPES_EXIST__) && NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
    #undef HAVE_BF16
    #define HAVE_BF16 1
    #if defined(__CUDA_FP8_TYPES_EXIST__) && NCCL_VERSION_CODE >= NCCL_VERSION(2,24,0)
      #undef HAVE_FP8
      #define HAVE_FP8 1
    #endif
  #endif
#endif

static size_t wordSize(ncclDataType_t type) {
  switch(type) {
    case ncclChar:
#if NCCL_MAJOR >= 2
    //case ncclInt8:
    case ncclUint8:
#if HAVE_FP8
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
#endif
#endif
      return 1;
    case ncclHalf:
#if HAVE_BF16
    case ncclBfloat16:
#endif
    //case ncclFloat16:
      return 2;
    case ncclInt:
    case ncclFloat:
#if NCCL_MAJOR >= 2
    //case ncclInt32:
    case ncclUint32:
    //case ncclFloat32:
#endif
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclDouble:
    //case ncclFloat64:
      return 8;
    default: return 0;
  }
}

extern int test_ncclVersion; // init'd with ncclGetVersion()
typedef enum { ncclCoarse        = 0,
               ncclFine          = 1,
               ncclHost          = 2,
               ncclManaged       = 3,
               nccl_NUM_MTYPES   = 4 } ncclMemoryType_t;
extern const char *test_memorytypes[nccl_NUM_MTYPES];
extern int deviceCtaCount; // number of CTAs for device implementation
extern int deviceImpl;     // selected -D device implementation (0 = host); lets per-collective
                           // device-timing hooks pick the matching timed kernel.
// In-kernel device timing (AllToAll -D 3/4): --device_timing 0=off, 1=augment, 2=device-only.
extern int deviceTimingMode;
extern int devtimeLoop;       // --devtime_loop (default 10)
extern int devtimeSkip;       // --devtime_skip (default 10)
extern int devtimeLoopMid;    // --devtime_loop_mid: loop at per-peer >= 8 MiB (0=disabled)
extern int devtimeLoopLarge;  // --devtime_loop_large: loop at per-peer >= 64 MiB (0=disabled)
extern int devtimeSkipMid;    // --devtime_skip_mid (-1: min(base skip, 2))
extern int devtimeSkipLarge;  // --devtime_skip_large (-1: min(base skip, 1))
extern int devtimeCheck;      // --devtime_check: validate timed-kernel output
constexpr int test_opNumMax = (int)ncclNumOps + (NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0) ? 1 : 0);
extern int test_opnum;
extern int test_typenum;
extern ncclDataType_t test_types[ncclNumTypes];
extern const char *test_typenames[ncclNumTypes];
extern ncclRedOp_t test_ops[];
extern const char *test_opnames[];

static int ncclstringtotype(char *str) {
    for (int t=0; t<test_typenum; t++) {
      if (strcmp(str, test_typenames[t]) == 0) {
        return t;
      }
    }
    if (strcmp(str, "all") == 0) {
      return -1;
    }
    printf("invalid type %s, defaulting to %s .. \n", str, test_typenames[ncclFloat]);
    return ncclFloat;
}

static int ncclstringtoop (char *str) {
    for (int o=0; o<test_opnum; o++) {
      if (strcmp(str, test_opnames[o]) == 0) {
        return o;
      }
    }
    if (strcmp(str, "all") == 0) {
      return -1;
    }
    printf("invalid op %s, defaulting to %s .. \n", str, test_opnames[ncclSum]);
    return ncclSum;
}

static int ncclstringtoroot (char *str) {
    if (strcmp(str, "all") == 0) {
      return -1;
    }
    return strtol(str, NULL, 0);
}

static int ncclstringtomtype (char *str) {
    for (int o=0; o<nccl_NUM_MTYPES; o++) {
      if (strcmp(str, test_memorytypes[o]) == 0) {
        return o;
      }
    }
    printf("invalid memorytype %s, defaulting to %s .. \n", str, test_memorytypes[ncclCoarse]);
    return ncclCoarse;
}

extern int is_main_proc;
extern thread_local int is_main_thread;

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
template <typename F>
testResult_t testLaunchDeviceKernel(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;

  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  kernel<<<deviceCtaCount, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm);
  return testSuccess;
}

#define SPECIALIZE_KERNEL(kernel, type, op) \
  ( op != ncclSum ? nullptr : \
   type == ncclInt8 ? kernel<int8_t> : \
   type == ncclUint8 ? kernel<uint8_t> : \
   type == ncclInt32 ? kernel<int32_t> : \
   type == ncclUint32 ? kernel<uint32_t> : \
   type == ncclInt64 ? kernel<int64_t> : \
   type == ncclUint64 ? kernel<uint64_t> : \
   type == ncclFloat16 ? kernel<half> : \
   type == ncclFloat32 ? kernel<float> : \
   type == ncclFloat64 ? kernel<double> : \
   nullptr \
  )
#else
template <typename F>
testResult_t testLaunchDeviceKernel(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  return testNotImplemented;
}
#define SPECIALIZE_KERNEL(kernel, type, op) nullptr
#endif

typedef enum {
  ncclFuncBroadcast = 0,
  ncclFuncReduce = 1,
  ncclFuncAllGather = 2,
  ncclFuncReduceScatter = 3,
  ncclFuncAllReduce = 4,
  ncclFuncAllReduceWithBias = 5,
  ncclFuncSendRecv = 6,
  ncclFuncSend = 7,
  ncclFuncRecv = 8,
  ncclFuncAllToAllPivot = 9,
  ncclNumFuncs = 10
} ncclFunc_t;

typedef ncclResult_t (*rcclTestsGetAlgoInfo_t)(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
                                          int collNetSupport, int nvlsSupport, int numPipeOps,
                                          int* algo, int* protocol, int* maxChannels);
typedef ncclResult_t (*rcclTestsGetAlgoName_t)(int algo, const char** algoName);
typedef ncclResult_t (*rcclTestsGetProtocolName_t)(int protocol, const char** protocolName);
typedef ncclResult_t (*rcclTestsGetSymkInfo_t)(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType, ncclRedOp_t op,
    int* algo, int* protocol, int* maxChannels);
// Newer librccl: reports the backend actually dispatched (CE/DDA/symmetric/kernel).
typedef ncclResult_t (*rcclTestsGetCollImplInfo_t)(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
    ncclRedOp_t op, const void* sendbuff, void* recvbuff, int graphCapturing, int* algo, int* protocol, int* maxChannels);

extern rcclTestsGetAlgoInfo_t rcclTestsGetAlgoInfo;
extern rcclTestsGetProtocolName_t rcclTestsGetProtocolName;
extern rcclTestsGetAlgoName_t rcclTestsGetAlgoName;
extern rcclTestsGetSymkInfo_t rcclTestsGetSymkInfo;
extern rcclTestsGetCollImplInfo_t rcclTestsGetCollImplInfo;

// Network counter collector (self-contained, see collector.h for full API)
#include "collector.h"

// rccl-tests wrappers that bridge threadArgs ↔ collector API
extern NetworkCounterContext NetCounterCollectBefore(struct threadArgs* args);
extern void NetCounterCollectAfterAndPrint(struct threadArgs* args, const NetworkCounterContext& ctx);

#endif
