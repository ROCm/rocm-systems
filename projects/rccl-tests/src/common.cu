
/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "rccl_float8.h"
#include <hip/hip_bfloat16.h>
#include "common.h"
#include <pthread.h>
#include <cstdio>
#include <type_traits>
#include <limits>
#include <getopt.h>
#include <libgen.h>
#include <string.h>
#include <ctype.h>
#include "cuda.h"
#include <vector>
#include <utility>
#include <errno.h>     /* program_invocation_short_name */
#include <dlfcn.h>
//#define DEBUG_PRINT

#include "verifiable.h"
#include "util.h"

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))

int test_ncclVersion = 0; // init'd with ncclGetVersion()
int32_t gpu_block3;
size_t cache_bytes = 192 * 1024 * 1024; // Use 192MB

rcclTestsGetAlgoInfo_t rcclTestsGetAlgoInfo = NULL;
rcclTestsGetProtocolName_t rcclTestsGetProtocolName = NULL;
rcclTestsGetAlgoName_t rcclTestsGetAlgoName= NULL;
static void loadRcclSyms() {
  static void* handle = NULL;
  const char* libname = "librccl.so";
  if (!handle) {
    handle = dlopen(libname, RTLD_LAZY | RTLD_LOCAL);
      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return;
      }
  }
  rcclTestsGetAlgoInfo      = (rcclTestsGetAlgoInfo_t)     dlsym(handle, "rcclGetAlgoInfo");
  rcclTestsGetAlgoName      = (rcclTestsGetAlgoName_t)     dlsym(handle,  "rcclGetAlgoName");
  rcclTestsGetProtocolName  = (rcclTestsGetProtocolName_t) dlsym(handle,  "rcclGetProtocolName");
}

// RCCL_FLOAT8 support
bool rccl_float8_useFnuz = false;
bool IsArchMatch(char const* arch, char const* target) {
  // helper function to reduce clutter in code elsewhere.  Returns true on match.
  return (strncmp(arch, target, strlen(target)) == 0);
}

#if NCCL_MAJOR >= 2
  ncclDataType_t test_types[ncclNumTypes] = {
    ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclHalf, ncclFloat, ncclDouble
  #if HAVE_BF16
    , ncclBfloat16
  #endif
  #if HAVE_FP8
    , ncclFloat8e4m3, ncclFloat8e5m2
  #endif
  };
  const char *test_typenames[ncclNumTypes] = {
    "int8", "uint8", "int32", "uint32", "int64", "uint64", "half", "float", "double"
  #if HAVE_BF16
    , "bfloat16"
  #endif
  #if HAVE_FP8
    , "fp8_e4m3", "fp8_e5m2"
  #endif
  };
  int test_typenum = -1;

  const char *test_opnames[] = {"sum", "prod", "max", "min", "avg", "mulsum"};
  ncclRedOp_t test_ops[] = {ncclSum, ncclProd, ncclMax, ncclMin
  #if NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
    , ncclAvg
  #endif
  #if NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0)
    , ncclNumOps // stand in for ncclRedOpCreatePreMulSum() created on-demand
  #endif
  };
  int test_opnum = -1;
#else
  ncclDataType_t test_types[ncclNumTypes] = {ncclChar, ncclInt, ncclHalf, ncclFloat, ncclDouble, ncclInt64, ncclUint64};
  const char *test_typenames[ncclNumTypes] = {"char", "int", "half", "float", "double", "int64", "uint64"};
  int test_typenum = 7;
  const char *test_opnames[] = {"sum", "prod", "max", "min"};
  ncclRedOp_t test_ops[] = {ncclSum, ncclProd, ncclMax, ncclMin};
  int test_opnum = 4;
#endif

const char *test_memorytypes[nccl_NUM_MTYPES] = {"coarse", "fine", "host", "managed"};

// For libnccl's < 2.13
extern "C" __attribute__((weak)) char const* ncclGetLastError(ncclComm_t comm) {
  return "";
}

int is_main_proc = 0;
thread_local int is_main_thread = 0;

// Command line parameter defaults
int nThreads = 1;
int nGpus = 1;
size_t minBytes = 32*1024*1024;
size_t maxBytes = 32*1024*1024;
size_t stepBytes = 1*1024*1024;
size_t stepFactor = 1;
int datacheck = 1;
int warmup_iters = 1;
int iters = 20;
int agg_iters = 1;
static int run_cycles = 1;
static int ncclop = ncclSum;
static int nccltype = ncclFloat;
static int ncclroot = 0;
int parallel_init = 0;
int blocking_coll = 0;
static int streamnull = 0;
static int timeout = 0;
int cudaGraphLaunches = 0;
static int output_algo_proto_channels = 0;
static int memorytype = 0;
static uint32_t cumask[4];
std::string rccl_output_file;
std::string rccl_output_format;
static int report_cputime = 0;
static int report_timestamps = 0;
static int deviceImpl = 0;
int memory_report = 0;

int deviceCtaCount = 16; // Default number of CTAs for device implementation

// Report average iteration time: (0=RANK0,1=AVG,2=MIN,3=MAX)
static int average = 1;
static int numDevices = 1;
static int delay_inout_place = 0;
static int enable_out_of_place = 1;
static int enable_in_place = 1;
static int enable_cache_flush = 0;
static int enable_rotating_tensor = 0;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
#define LOCAL_REGISTER 1
#define SYMMETRIC_REGISTER 2
static int local_register = 0;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
static int ctaPolicy = -1;
#endif
#endif
static int minCudaArch = 1<<30;

// Test bias
static int test_bias = 0;

// RCCL-tests Reporter class implementation
Reporter::Reporter(std::string fileName, std::string outputFormat) : _outputFormat(outputFormat) {
  if (!fileName.empty()) {
    if (isMainThread()) {
      _out = std::ofstream(fileName, std::ios_base::out);
      _outputValid = true;
    }
  }
}

void Reporter::setParameters(const size_t numCycle, const char* name, const char* typeName, const char* opName) {
  if (!isMainThread() || !_outputValid)
    return;

  _numCycle = numCycle;
  _collectiveName = name;
  _typeName = typeName;
  _opName = opName;
}

void Reporter::addResult(int gpusPerRank, int ranksPerNode, int totalRanks, size_t numBytes, int inPlace, double timeUsec, double algBw, double busBw, int64_t wrongElts) {
  if (!isMainThread() || !_outputValid)
    return;

  std::vector<std::pair<std::string, std::string>> outputValuesKeys;
  std::string wrongEltsStr = (wrongElts == -1) ? "N/A" : std::to_string(wrongElts);
  int nodes = totalRanks / ranksPerNode;

  outputValuesKeys.push_back(makeValueKeyPair(_numCycle, "numCycle"));
  outputValuesKeys.push_back(makeValueKeyPair(_collectiveName, "name"));
#ifdef MPI_SUPPORT
  outputValuesKeys.push_back(makeValueKeyPair(nodes, "nodes"));
  outputValuesKeys.push_back(makeValueKeyPair(totalRanks, "ranks"));
  outputValuesKeys.push_back(makeValueKeyPair(ranksPerNode, "ranksPerNode"));
  outputValuesKeys.push_back(makeValueKeyPair(gpusPerRank, "gpusPerRank"));
#else
  outputValuesKeys.push_back(makeValueKeyPair(gpusPerRank, "gpus"));
#endif
  outputValuesKeys.push_back(makeValueKeyPair(numBytes, "size"));
  outputValuesKeys.push_back(makeValueKeyPair(_typeName, "type"));
  outputValuesKeys.push_back(makeValueKeyPair(_opName, "redop"));
  outputValuesKeys.push_back(makeValueKeyPair(inPlace, "inPlace"));
  outputValuesKeys.push_back(makeValueKeyPair(timeUsec, "time"));
  outputValuesKeys.push_back(makeValueKeyPair(algBw, "algBw"));
  outputValuesKeys.push_back(makeValueKeyPair(busBw, "busBw"));
  outputValuesKeys.push_back(makeValueKeyPair(wrongEltsStr, "wrong"));

  _outputData.push_back(outputValuesKeys);
}

void Reporter::writeFile() {
  if (!isMainThread() || !_outputValid)
    return;

  if (_outputFormat == "csv") {
    _out << "numCycle,";
    _out << "collective,";
#ifdef MPI_SUPPORT
    _out << "ranks,rankspernode,gpusperrank,";
#else
    _out << "gpus,";
#endif
    _out << "size,type,redop,inplace,time,algbw,busbw,#wrong\n";
    for (auto iterEntries = _outputData.begin(); iterEntries != _outputData.end(); ++iterEntries) {
      for (auto iterVals = (*iterEntries).begin(); iterVals != (*iterEntries).end(); ++iterVals) {
	_out << iterVals->first;
	if (std::next(iterVals) != (*iterEntries).end()) {
	  _out << ",";
	}
      }
      _out << std::endl;
    }
  } else { //json
    _out << "[" << std::endl;
    for (auto iterEntries = _outputData.begin(); iterEntries != _outputData.end(); ++iterEntries) {
      for (auto iterVals = (*iterEntries).begin(); iterVals != (*iterEntries).end(); ++iterVals) {
        if (iterVals == (*iterEntries).begin()) {
          _out << "{";
        }
        _out << "\"" << iterVals->second << "\":" << iterVals->first;
        if (std::next(iterVals) != (*iterEntries).end()) {
          _out << ", ";
	}
      }
      if (std::next(iterEntries) != _outputData.end()) {
        _out << "}," << std::endl;
      } else {
	_out << "}" << std::endl;
      }
    }
    _out << "]" << std::endl;
  }
}

bool Reporter::isMainThread() { return is_main_thread == 1; }

#define NUM_BLOCKS 32

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

extern "C" __global__ void flush_icache()
{
    asm __volatile__("s_icache_inv \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t" ::
                         :);
}

// NCCL-tests JSON output format
enum output_file_type_t {
  JSON_FILE_OUTPUT,
  UNSPECIFIED_FILE_OUTPUT
};

// Return pointer to extension in `path` if one is found. An extension
// is the last `.` in the `path`, if there is no `/` following the `.`
// and there are characters after `.`.
//
// Therefore: returns 0 if no meaningful extension was found, or returns offset
// into string where extension begins
static const char *getExtension(const char *path) {
  if (path == nullptr) return nullptr;
  int last_dot = -1;
  int last_slash = -1;

  int pos;
  for (pos = 0; path[pos] != '\0'; ++pos) {
    switch (path[pos]) {
    case '.':
      last_dot = pos;
      break;
    case '/':
      last_slash = pos;
      break;
    default:
      break;
    }
  }

  if (last_dot > last_slash && last_dot + 1 != pos) {
    return path + last_dot + 1;
  }

  return nullptr;
}

static output_file_type_t classifyOutputFile(const char *filename) {
  const char *extension = getExtension(filename);
  if (extension != nullptr && strcasecmp(extension, "json") == 0) {
    return JSON_FILE_OUTPUT;
  }

  return UNSPECIFIED_FILE_OUTPUT;
}

static void outputFileInit(output_file_type_t output_file_type,
                           const char *output_file, int argc, char **argv, char **envp) {
  switch (output_file_type) {
  case JSON_FILE_OUTPUT:
    jsonOutputInit(output_file, argc, argv, envp);
    break;
  case UNSPECIFIED_FILE_OUTPUT:
  default:
    break;
  }
}

static void outputFileFinalize(output_file_type_t output_file_type) {
  switch (output_file_type) {
  case JSON_FILE_OUTPUT:
    jsonOutputFinalize();
    break;
  case UNSPECIFIED_FILE_OUTPUT:
  default:
    break;
  }
}

testResult_t initComms(ncclComm_t* comms, int nComms, int firstRank, int nRanks, int* cudaDevs, ncclUniqueId& ncclId) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
  if (ctaPolicy >= 0)
    config.CTAPolicy = ctaPolicy;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  config.nvlinkCentricSched = 1;
#endif
#endif
#endif

  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<nComms; i++) {
    int rank = firstRank + i;
    CUDACHECK(cudaSetDevice(cudaDevs[i]));
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
    NCCLCHECK(ncclCommInitRankConfig(comms+i, nRanks, ncclId, rank, &config));
#else
    NCCLCHECK(ncclCommInitRank(comms+i, nRanks, ncclId, rank));
#endif
  }
  NCCLCHECK(ncclGroupEnd());
  return testSuccess;
}

// NOTE: We use the binary system, so M=Mebibytes and G=Gibibytes
static double parsesize(const char *value) {
    long long int units;
    double size;
    char size_lit[2];

    int count = sscanf(value, "%lf %1s", &size, size_lit);

    switch (count) {
    case 2:
      switch (size_lit[0]) {
      case 'G':
      case 'g':
        units = 1024*1024*1024;
        break;
      case 'M':
      case 'm':
        units = 1024*1024;
        break;
      case 'K':
      case 'k':
        units = 1024;
        break;
      default:
        return -1.0;
      };
      break;
    case 1:
      units = 1;
      break;
    default:
      return -1.0;
    }

    return size * units;
}

static bool minReqVersion(int rmajor, int rminor, int rpatch)
{
  int version;
  int major, minor, patch, rem;
  ncclGetVersion(&version);

  if (version < 10000) {
    major = version/1000;
    rem   = version%1000;
    minor = rem/100;
    patch = rem%100;
  }
  else {
    major = version/10000;
    rem   = version%10000;
    minor = rem/100;
    patch = rem%100;
  }

  if (major < rmajor)      return false;
  else if (major > rmajor) return true;

  // major == rmajor
  if (minor < rminor)      return false;
  else if (minor > rminor) return true;

  // major == rmajor && minor == rminor
  if (patch < rpatch)      return false;

  return true;
}

testResult_t CheckDelta(void* results, void* expected, size_t count, size_t offset, ncclDataType_t type, ncclRedOp_t op, uint64_t seed, int nranks, int64_t *wrongEltN) {
  CUDACHECK(ncclVerifiableVerify(results, expected, count, (int)type, (int)op, nranks, seed, offset, wrongEltN, cudaStreamDefault));
  CUDACHECK(cudaDeviceSynchronize());
  return testSuccess;
}

testResult_t InitDataReduce(void* data, const size_t count, const size_t offset, ncclDataType_t type, ncclRedOp_t op, uint64_t seed, int nranks) {
  CUDACHECK(ncclVerifiablePrepareExpected(data, count, (int)type, (int)op, nranks, seed, offset, cudaStreamDefault));
  return testSuccess;
}

testResult_t InitDataApplyBias(void* expected, void* bias, const size_t count, const size_t offset, ncclDataType_t type, ncclRedOp_t op) {
  ncclVerifiableApplyBias(expected, bias, count, (int)type, (int)op, offset, cudaStreamDefault);
  return testSuccess;
}

testResult_t InitData(void* data, const size_t count, size_t offset, ncclDataType_t type, ncclRedOp_t op, uint64_t seed, int nranks, int rank) {
  CUDACHECK(ncclVerifiablePrepareInput(data, count, (int)type, (int)op, nranks, rank, seed, offset, cudaStreamDefault));
  return testSuccess;
}

void Barrier(struct threadArgs *args) {
  thread_local int epoch = 0;
  static pthread_mutex_t lock[2] = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER};
  static pthread_cond_t cond[2] = {PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER};
  static int counter[2] = {0, 0};

  pthread_mutex_lock(&lock[epoch]);
  if(++counter[epoch] == args->nThreads)
    pthread_cond_broadcast(&cond[epoch]);

  if(args->thread+1 == args->nThreads) {
    while(counter[epoch] != args->nThreads)
      pthread_cond_wait(&cond[epoch], &lock[epoch]);
    #ifdef MPI_SUPPORT
      MPI_Barrier(MPI_COMM_WORLD);
    #endif
    counter[epoch] = 0;
    pthread_cond_broadcast(&cond[epoch]);
  }
  else {
    while(counter[epoch] != 0)
      pthread_cond_wait(&cond[epoch], &lock[epoch]);
  }
  pthread_mutex_unlock(&lock[epoch]);
  epoch ^= 1;
}

// Inter-thread/process barrier+allreduce. The quality of the return value
// for average=0 (which means broadcast from rank=0) is dubious. The returned
// value will actually be the result of process-local broadcast from the local thread=0.
template<typename T>
void Allreduce(struct threadArgs* args, T* value, int average) {
  thread_local int epoch = 0;
  static pthread_mutex_t lock[2] = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER};
  static pthread_cond_t cond[2] = {PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER};
  static T accumulator[2];
  static int counter[2] = {0, 0};

  pthread_mutex_lock(&lock[epoch]);
  if(counter[epoch] == 0) {
    if(average != 0 || args->thread == 0) accumulator[epoch] = *value;
  } else {
    switch(average) {
    case /*r0*/ 0: if(args->thread == 0) accumulator[epoch] = *value; break;
    case /*avg*/1: accumulator[epoch] += *value; break;
    case /*min*/2: accumulator[epoch] = std::min<T>(accumulator[epoch], *value); break;
    case /*max*/3: accumulator[epoch] = std::max<T>(accumulator[epoch], *value); break;
    case /*sum*/4: accumulator[epoch] += *value; break;
    }
  }

  if(++counter[epoch] == args->nThreads)
    pthread_cond_broadcast(&cond[epoch]);

  if(args->thread+1 == args->nThreads) {
    while(counter[epoch] != args->nThreads)
      pthread_cond_wait(&cond[epoch], &lock[epoch]);

    #ifdef MPI_SUPPORT
    if(average != 0) {
      static_assert(std::is_same<T, long long>::value || std::is_same<T, double>::value, "Allreduce<T> only for T in {long long, double}");
      MPI_Datatype ty = std::is_same<T, long long>::value ? MPI_LONG_LONG :
                        std::is_same<T, double>::value ? MPI_DOUBLE :
                        MPI_Datatype();
      MPI_Op op = average == 1 ? MPI_SUM :
                  average == 2 ? MPI_MIN :
                  average == 3 ? MPI_MAX :
                  average == 4 ? MPI_SUM : MPI_Op();
      MPI_Allreduce(MPI_IN_PLACE, (void*)&accumulator[epoch], 1, ty, op, MPI_COMM_WORLD);
    }
    #endif

    if(average == 1) accumulator[epoch] /= args->totalProcs*args->nThreads;
    counter[epoch] = 0;
    pthread_cond_broadcast(&cond[epoch]);
  }
  else {
    while(counter[epoch] != 0)
      pthread_cond_wait(&cond[epoch], &lock[epoch]);
  }
  pthread_mutex_unlock(&lock[epoch]);

  *value = accumulator[epoch];
  epoch ^= 1;
}

testResult_t CheckData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int64_t *wrongElts) {
  int nranks = args->nProcs*args->nGpus*args->nThreads;
  size_t count = args->expectedBytes/wordSize(type);

  int64_t *wrongPerGpu = nullptr;
  CUDACHECK(hipHostMalloc((void**)&wrongPerGpu, args->nGpus*sizeof(int64_t), cudaHostAllocMapped));

  for (int i=0; i<args->nGpus; i++) {
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    void *data = in_place ? ((void *)((uintptr_t)args->recvbuffs[i] + args->recvInplaceOffset*rank)) : args->recvbuffs[i];

    TESTCHECK(CheckDelta(data, args->expected[i], count, 0, type, op, 0, nranks, wrongPerGpu+i));

#if 1 && defined(DEBUG_PRINT)
    if (args->reportErrors && wrongPerGpu[i] != 0) {
      printf("rank=%d #wrong=%d\n", rank, (int)wrongPerGpu[i]);
      char *expectedHost = (char*)malloc(args->expectedBytes);
      char *dataHost = (char*)malloc(args->expectedBytes);
      int eltsz = wordSize(type);
      cudaMemcpy(expectedHost, args->expected[i], args->expectedBytes, cudaMemcpyDeviceToHost);
      cudaMemcpy(dataHost, data, args->expectedBytes, cudaMemcpyDeviceToHost);

      for(int j=0; j<args->expectedBytes/eltsz; j++) {
        unsigned long long want, got;
        want = 0;
        memcpy(&want, expectedHost + j*eltsz, eltsz);
        got = 0;
        memcpy(&got, dataHost + j*eltsz, eltsz);
        if(want != got) {
          printf(" rank=%d elt[%d]: want=0x%llx got=0x%llx\n", rank, j, want, got);
        }
      }
      free(expectedHost);
      free(dataHost);
    }
#endif
  }

  *wrongElts = 0;
  for (int i=0; i < args->nGpus; i++) *wrongElts += wrongPerGpu[i];
  cudaFreeHost(wrongPerGpu);

  if (args->reportErrors && *wrongElts) args->errors[0]++;
  return testSuccess;
}

testResult_t testStreamSynchronize(int ngpus, cudaStream_t* streams, ncclComm_t* comms) {
  cudaError_t cudaErr;
  int remaining = ngpus;
  int* done = (int*)malloc(sizeof(int)*ngpus);
  memset(done, 0, sizeof(int)*ngpus);
  timer tim;

  while (remaining) {
   int idle = 1;
   for (int i=0; i<ngpus; i++) {
     if (done[i]) continue;

     cudaErr = cudaStreamQuery(streams[i]);
     if (cudaErr == cudaSuccess) {
       done[i] = 1;
       remaining--;
       idle = 0;
       continue;
     }

     if (cudaErr != cudaErrorNotReady) CUDACHECK(cudaErr);

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,4,0)
     if (test_ncclVersion >= NCCL_VERSION(2,4,0) && comms) {
       ncclResult_t ncclAsyncErr;
       NCCLCHECK(ncclCommGetAsyncError(comms[i], &ncclAsyncErr));
       if (ncclAsyncErr != ncclSuccess) {
         // An asynchronous error happened. Stop the operation and destroy
         // the communicator
         for (int i=0; i<ngpus; i++)
           NCCLCHECK(ncclCommAbort(comms[i]));
         // Abort the perf test
         NCCLCHECK(ncclAsyncErr);
       }
     }
     double delta = tim.elapsed();
     if (delta > timeout && timeout > 0) {
       for (int i=0; i<ngpus; i++)
         NCCLCHECK(ncclCommAbort(comms[i]));
       char hostname[1024];
       getHostName(hostname, 1024);
       printf("%s: Test timeout (%ds) %s:%d\n",
           hostname,
           timeout,
           __FILE__,__LINE__);
       free(done);
       return testTimeout;
     }
#endif
   }

   // We might want to let other threads (including NCCL threads) use the CPU.
   if (idle) sched_yield();
  }
  free(done);
  return testSuccess;
}

testResult_t startColl(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t opIndex, int root, int in_place, int iter) {
  size_t count = args->nbytes / wordSize(type);

  // Try to change offset for each iteration so that we avoid cache effects and catch race conditions in ptrExchange
  size_t shift = 0;
  if(enable_rotating_tensor) {
    shift = cache_bytes * (iter % 2);
  }
  else {
    size_t totalnbytes = std::max(args->sendBytes, args->expectedBytes);
    size_t steps = totalnbytes ? args->maxbytes / totalnbytes : 1;
    shift = totalnbytes * (iter % steps);
  }

  if (args->nGpus > 1) NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < args->nGpus; i++) {
#ifndef NCCL_MAJOR
    CUDACHECK(cudaSetDevice(args->gpus[i]));
#endif
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    char* recvBuff = ((char*)args->recvbuffs[i]) + shift;
    char* sendBuff = ((char*)args->sendbuffs[i]) + shift;
    char* bias = ((char*)args->bias[i]) + shift;
    ncclRedOp_t op;

    if(opIndex < ncclNumOps) {
      op = opIndex;
    }
    #if NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0)
    else {
      union {
        int8_t i8; uint8_t u8; int32_t i32; uint32_t u32; int64_t i64; uint64_t u64;
        half f16; float f32; double f64;
        #if HAVE_BF16
        hip_bfloat16 bf16;
        #endif
        #if HAVE_FP8
        rccl_float8 fp8_e4m3; rccl_bfloat8 fp8_e5m2;
        #endif
      };
      switch(type) {
      case ncclInt8: i8 = ncclVerifiablePremulScalar<int8_t>(rank); break;
      case ncclUint8: u8 = ncclVerifiablePremulScalar<uint8_t>(rank); break;
      case ncclInt32: i32 = ncclVerifiablePremulScalar<int32_t>(rank); break;
      case ncclUint32: u32 = ncclVerifiablePremulScalar<uint32_t>(rank); break;
      case ncclInt64: i64 = ncclVerifiablePremulScalar<int64_t>(rank); break;
      case ncclUint64: u64 = ncclVerifiablePremulScalar<uint64_t>(rank); break;
      case ncclFloat16: f16 = ncclVerifiablePremulScalar<half>(rank); break;
      case ncclFloat32: f32 = ncclVerifiablePremulScalar<float>(rank); break;
      case ncclFloat64: f64 = ncclVerifiablePremulScalar<double>(rank); break;
      #if HAVE_BF16
      case ncclBfloat16: bf16 = ncclVerifiablePremulScalar<hip_bfloat16>(rank); break;
      #endif
      #if HAVE_FP8
      case ncclFloat8e4m3: fp8_e4m3 = ncclVerifiablePremulScalar<rccl_float8>(rank); break;
      case ncclFloat8e5m2 : fp8_e5m2 = ncclVerifiablePremulScalar<rccl_bfloat8>(rank); break;
      #endif
      default: break; // Just to silence clang
      }
      NCCLCHECK(ncclRedOpCreatePreMulSum(&op, &u64, type, ncclScalarHostImmediate, args->comms[i]));
    }
    #endif

    if(enable_cache_flush > 0 && ((iter % enable_cache_flush) == 0)) {
      hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0, args->streams[i]);
    }

    if (deviceImpl == 0) {
      TESTCHECK(args->collTest->runColl(
            (void*)(in_place ? recvBuff : sendBuff), in_place ? args->sendInplaceOffset*rank : 0,
            (void*)recvBuff, in_place ? args->recvInplaceOffset*rank : 0,
            count, type, op, root, args->comms[i], args->streams[i], 0, bias));
    } else {
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
      void* sendwin = args->sendRegHandles[i];
      void* recvwin = args->recvRegHandles[i];
      CUDACHECK(cudaSetDevice(args->gpus[i]));
      TESTCHECK(args->collTest->runColl(
            (void*)(in_place ? recvwin : sendwin), shift + (in_place ? args->sendInplaceOffset*rank : 0),
            (void*)recvwin, shift + (in_place ? args->recvInplaceOffset*rank : 0),
            count, type, op, root, (ncclComm_t)(args->devComms+i), args->streams[i], deviceImpl, bias));
#endif
    }

    #if NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0)
    if(opIndex >= ncclNumOps) {
      NCCLCHECK(ncclRedOpDestroy(op, args->comms[i]));
    }
    #endif
  }
  if (args->nGpus > 1) NCCLCHECK(ncclGroupEnd());

  if (blocking_coll) {
    // Complete op before returning
    TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms));
  }
  if (blocking_coll) Barrier(args);
  return testSuccess;
}

testResult_t completeColl(struct threadArgs* args) {
  if (blocking_coll) return testSuccess;

  TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms));
  return testSuccess;
}

testResult_t BenchTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place) {
  size_t count = args->nbytes / wordSize(type);
  if (datacheck) {
    // Initialize sendbuffs, recvbuffs and expected
    TESTCHECK(args->collTest->initData(args, type, op, root, 99, in_place));
  }

  if (warmup_iters) {
    // Sync
    TESTCHECK(startColl(args, type, op, root, in_place, 0));
    TESTCHECK(completeColl(args));
  }

  Barrier(args);

#if HIP_VERSION >= 50221310
  std::vector<cudaGraph_t> graphs(args->nGpus);
  std::vector<cudaGraphExec_t> graphExec(args->nGpus);
  if (cudaGraphLaunches >= 1) {
    // Begin cuda graph capture
    for (int i=0; i<args->nGpus; i++) {
      // Thread local mdoe is needed for:
      // - Multi-thread mode: where graph capture and instantiation can happen concurrently across threads
      // - P2P pre-connect: when there is no warm-up, P2P pre-connect is done during graph capture.
      //   Since pre-connect calls cudaMalloc, we cannot use global capture mode
      CUDACHECK(cudaStreamBeginCapture(args->streams[i], cudaStreamCaptureModeThreadLocal));
    }
  }
#endif

  // Performance Benchmark
  timer tim;
  for (int iter = 0; iter < iters; iter++) {
    if (agg_iters>1) NCCLCHECK(ncclGroupStart());
    for (int aiter = 0; aiter < agg_iters; aiter++) {
      TESTCHECK(startColl(args, type, op, root, in_place, iter*agg_iters+aiter));
    }
    if (agg_iters>1) NCCLCHECK(ncclGroupEnd());
  }

#if HIP_VERSION >= 50221310
  if (cudaGraphLaunches >= 1) {
    // End cuda graph capture
    for (int i=0; i<args->nGpus; i++) {
      CUDACHECK(cudaStreamEndCapture(args->streams[i], graphs.data()+i));
    }
    // Instantiate cuda graph
    for (int i=0; i<args->nGpus; i++) {
      CUDACHECK(cudaGraphInstantiate(graphExec.data()+i, graphs[i], NULL, NULL, 0));
    }
    // Resync CPU, restart timing, launch cuda graph
    Barrier(args);
    tim.reset();
    for (int l=0; l<cudaGraphLaunches; l++) {
      for (int i=0; i<args->nGpus; i++) {
        CUDACHECK(cudaGraphLaunch(graphExec[i], args->streams[i]));
      }
    }
  }
#endif

  double cputimeSec = tim.elapsed()/(iters*agg_iters);
  TESTCHECK(completeColl(args));

  double deltaSec = tim.elapsed();
  deltaSec = deltaSec/(iters*agg_iters);
  if (cudaGraphLaunches >= 1) deltaSec = deltaSec/cudaGraphLaunches;
  Allreduce(args, &deltaSec, average);

#if HIP_VERSION >= 50221310
  if (cudaGraphLaunches >= 1) {
    //destroy cuda graph
    for (int i=0; i<args->nGpus; i++) {
      CUDACHECK(cudaGraphExecDestroy(graphExec[i]));
      CUDACHECK(cudaGraphDestroy(graphs[i]));
    }
  }
#endif

  double algBw, busBw;
  args->collTest->getBw(count, wordSize(type), deltaSec, &algBw, &busBw, args->nProcs*args->nThreads*args->nGpus);

  Barrier(args);

  int64_t wrongElts = 0;
  static __thread int rep = 0;
  rep++;
  for (int c = 0; c < datacheck; c++) {
      // Initialize sendbuffs, recvbuffs and expected
      TESTCHECK(args->collTest->initData(args, type, op, root, rep, in_place));

#if HIP_VERSION >= 50221310
      if (cudaGraphLaunches >= 1) {
        // Begin cuda graph capture for data check
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaStreamBeginCapture(args->streams[i], args->nThreads > 1 ? cudaStreamCaptureModeThreadLocal : cudaStreamCaptureModeGlobal));
        }
      }
#endif

      //test validation in single itertion, should ideally be included into the multi-iteration run
      TESTCHECK(startColl(args, type, op, root, in_place, 0));

#if HIP_VERSION >= 50221310
      if (cudaGraphLaunches >= 1) {
        // End cuda graph capture
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaStreamEndCapture(args->streams[i], graphs.data()+i));
        }
        // Instantiate cuda graph
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaGraphInstantiate(graphExec.data()+i, graphs[i], NULL, NULL, 0));
        }
        // Launch cuda graph
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaGraphLaunch(graphExec[i], args->streams[i]));
        }
      }
#endif

      TESTCHECK(completeColl(args));

#if HIP_VERSION >= 50221310
      if (cudaGraphLaunches >= 1) {
        //destroy cuda graph
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaGraphExecDestroy(graphExec[i]));
          CUDACHECK(cudaGraphDestroy(graphs[i]));
        }
      }
#endif

      TESTCHECK(CheckData(args, type, op, root, in_place, &wrongElts));

      //aggregate delta from all threads and procs
      long long wrongElts1 = wrongElts;
      //if (wrongElts) fprintf(stderr, "\nERROR: Data corruption : rank %d size %ld wrongElts %ld\n", args->proc, args->expectedBytes, wrongElts);
      Allreduce(args, &wrongElts1, /*sum*/4);
      wrongElts = wrongElts1;
      if (wrongElts) break;
  }

  double timeUsec = (report_cputime ? cputimeSec : deltaSec)*1.0E6;
  writeBenchmarkLineBody(timeUsec, algBw, busBw, args->reportErrors, wrongElts, report_cputime, report_timestamps, in_place==0);

  auto largestMessageSize = std::max(args->sendBytes, args->expectedBytes);
  if (args->reporter) {
    if (args->reportErrors) {
      args->reporter->addResult((args->nThreads * args->nGpus), args->nProcs, args->totalProcs, largestMessageSize, in_place, timeUsec, algBw, busBw, wrongElts);
    } else {
      args->reporter->addResult((args->nThreads * args->nGpus), args->nProcs, args->totalProcs, largestMessageSize, in_place, timeUsec, algBw, busBw);
    }
  }

  args->bw[0] += busBw;
  args->bw_count[0]++;
  return testSuccess;
}

void setupArgs(size_t size, ncclDataType_t type, struct threadArgs* args) {
  int nranks = args->nProcs*args->nGpus*args->nThreads;
  size_t count, sendCount, recvCount, paramCount, sendInplaceOffset, recvInplaceOffset;

  count = size / wordSize(type);
  args->collTest->getCollByteCount(&sendCount, &recvCount, &paramCount, &sendInplaceOffset, &recvInplaceOffset, (size_t)count, wordSize(type), (size_t)nranks);

  args->nbytes = paramCount * wordSize(type);
  args->sendBytes = sendCount * wordSize(type);
  args->expectedBytes = recvCount * wordSize(type);
  args->sendInplaceOffset = sendInplaceOffset * wordSize(type);
  args->recvInplaceOffset = recvInplaceOffset * wordSize(type);
}

testResult_t TimeTest(struct threadArgs* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName, int root) {
  // Sync to avoid first-call timeout
  Barrier(args);

  // Warm-up for all sizes (using a stepfactor of 2)
  for (size_t size = args->minbytes; size <= args->maxbytes; size = size * 2) {
    setupArgs(size, type, args);
#if HIP_VERSION >= 50221310
    std::vector<cudaGraph_t> graphs(args->nGpus);
    std::vector<cudaGraphExec_t> graphExec(args->nGpus);
    if (cudaGraphLaunches >= 1) {
      // Begin cuda graph capture
      for (int i=0; i<args->nGpus; i++) {
        // Thread local mode is needed for:
        // - Multi-thread mode: where graph capture and instantiation can happen concurrently across threads
        // - P2P pre-connect: when there is no warm-up, P2P pre-connect is done during graph capture.
        //   Since pre-connect calls cudaMalloc, we cannot use global capture mode
        CUDACHECK(cudaStreamBeginCapture(args->streams[i], cudaStreamCaptureModeThreadLocal));
      }
    }
#endif
    for (int iter = 0; iter < warmup_iters; iter++) {
      TESTCHECK(startColl(args, type, op, root, 0, iter));
    }

#if HIP_VERSION >= 50221310
    if (cudaGraphLaunches >= 1) {
      // End cuda graph capture
      for (int i=0; i<args->nGpus; i++) {
        CUDACHECK(cudaStreamEndCapture(args->streams[i], graphs.data()+i));
      }
      // Instantiate cuda graph
      for (int i=0; i<args->nGpus; i++) {
        CUDACHECK(cudaGraphInstantiate(graphExec.data()+i, graphs[i], NULL, NULL, 0));
      }
      // Resync CPU, restart timing, launch cuda graph
      Barrier(args);
      for (int l=0; l<cudaGraphLaunches; l++) {
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaGraphLaunch(graphExec[i], args->streams[i]));
        }
      }
    }
#endif

    TESTCHECK(completeColl(args));

#if HIP_VERSION >= 50221310
    if (cudaGraphLaunches >= 1) {
      //destroy cuda graph
      for (int i=0; i<args->nGpus; i++) {
        CUDACHECK(cudaGraphExecDestroy(graphExec[i]));
        CUDACHECK(cudaGraphDestroy(graphs[i]));
      }
    }
#endif
  }

  // Benchmark
  long repeat = run_cycles;
  size_t iter = 0;

  do {
    if (run_cycles > 1) PRINT("# Testing %lu cycle.\n", iter+1);
    if (args->reporter) {
      args->reporter->setParameters(iter, args->collTest->name, typeName, opName);
    }
    for (size_t size = args->minbytes; size<=args->maxbytes; size = ((args->stepfactor > 1) ? size*args->stepfactor : size+args->stepbytes)) {
      setupArgs(size, type, args);
      writeBenchmarkLinePreamble(std::max(args->sendBytes, args->expectedBytes), args->nbytes / wordSize(type), typeName, opName, root);
      if (enable_out_of_place) {
        TESTCHECK(BenchTime(args, type, op, root, 0));
        usleep(delay_inout_place);
      }
      if (enable_in_place)
        TESTCHECK(BenchTime(args, type, op, root, 1));
      if(output_algo_proto_channels) {
        if(args->collTest->getAlgoProtoChannels) {
          int algo, proto, nchannels;
          const char* algoName = NULL;
          const char* protoName = NULL;
          TESTCHECK(args->collTest->getAlgoProtoChannels(args->comms[0], args->nbytes / wordSize(type), type, &algo, &proto, &nchannels));
          NCCLCHECK(rcclTestsGetAlgoName(algo, &algoName));
          NCCLCHECK(rcclTestsGetProtocolName(proto, &protoName));
          PRINT("%8s  %8s  %10d", algoName, protoName, nchannels);
        } else {
          PRINT("%8s  %8s  %10s","N/A", "N/A", "N/A");
        }
      }
      writeBenchmarkLineTerminator(iters, "");
    }
    --repeat;
    ++iter;
  } while(repeat != 0);

  return testSuccess;
}

static void getGPUMemoryInfo(int64_t* ptotalGpuMem, int64_t* pfreeGpuMem) {
  size_t freeGpuMem, totalGpuMem = 0;
  cudaMemGetInfo(&freeGpuMem, &totalGpuMem);
  if (ptotalGpuMem != nullptr) *ptotalGpuMem = totalGpuMem;
  if (pfreeGpuMem != nullptr) *pfreeGpuMem = freeGpuMem;
}

testResult_t threadRunTests(struct threadArgs* args) {
  //  capture the free memory before
  int64_t* totalGpuFreeMem = (int64_t*)calloc(args->nGpus*2, sizeof(int64_t));
  for (int g = 0; g < args->nGpus; ++g) {
    CUDACHECK(cudaSetDevice(args->gpus[g]));
    getGPUMemoryInfo(nullptr, &totalGpuFreeMem[g]);
  }

  // Set device to the first of our GPUs. If we don't do that, some operations
  // will be done on the current GPU (by default : 0) and if the GPUs are in
  // exclusive mode those operations will fail.
  CUDACHECK(cudaSetDevice(args->gpus[0]));
  TESTCHECK(ncclTestEngine.runTest(args, ncclroot, (ncclDataType_t)nccltype, test_typenames[nccltype], (ncclRedOp_t)ncclop, test_opnames[ncclop]));

  // Capture the memory used by the GPUs
  for (int g = 0; g < args->nGpus; ++g) {
    CUDACHECK(cudaSetDevice(args->gpus[g]));
    getGPUMemoryInfo(nullptr, &totalGpuFreeMem[g + args->nGpus]);
    *args->devMemUsed = std::max(*args->devMemUsed, totalGpuFreeMem[g] - totalGpuFreeMem[g + args->nGpus]);
  }
  free(totalGpuFreeMem);
  return testSuccess;
}

testResult_t threadInit(struct threadArgs* args) {
  int nranks =  args->nProcs*args->nThreads*args->nGpus;

  //set main thread again
  is_main_thread = (is_main_proc && args->thread == 0) ? 1 : 0;

  jsonIdentifyWriter(is_main_thread);

  // Capture GPU memory before initializing the NCCL communicators
  int64_t* initFreeGpuMem = (int64_t*)calloc(args->nGpus*3, sizeof(int64_t));
  for (int g = 0; g < args->nGpus; ++g) {
    CUDACHECK(cudaSetDevice(args->gpus[g]));
    getGPUMemoryInfo(nullptr, &initFreeGpuMem[g]);
  }

  int firstRank = args->proc*args->nThreads*args->nGpus + args->thread*args->nGpus;
  TESTCHECK(initComms(args->comms, args->nGpus, firstRank, nranks, args->gpus, args->ncclId));

  // Capture the memory used by the GPUs after initializing the NCCL communicators
  for (int g = 0; g < args->nGpus; ++g) {
    CUDACHECK(cudaSetDevice(args->gpus[g]));
    getGPUMemoryInfo(nullptr, &initFreeGpuMem[g + args->nGpus]);
    *args->initGpuMem = std::max(*args->initGpuMem, initFreeGpuMem[g] - initFreeGpuMem[g + args->nGpus]);
  }

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<args->nGpus; i++) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
    if (test_ncclVersion >= NCCL_VERSION(2,27,0) && (local_register == SYMMETRIC_REGISTER)) {
      NCCLCHECK(ncclCommWindowRegister(args->comms[i], args->sendbuffs[i], args->maxbytes, (ncclWindow_t*)&args->sendRegHandles[i], NCCL_WIN_COLL_SYMMETRIC));
      NCCLCHECK(ncclCommWindowRegister(args->comms[i], args->recvbuffs[i], args->maxbytes, (ncclWindow_t*)&args->recvRegHandles[i], NCCL_WIN_COLL_SYMMETRIC));
    } else
#endif
    {
      if (local_register) NCCLCHECK(ncclCommRegister(args->comms[i], args->sendbuffs[i], args->maxbytes, &args->sendRegHandles[i]));
      if (local_register) NCCLCHECK(ncclCommRegister(args->comms[i], args->recvbuffs[i], args->maxbytes, &args->recvRegHandles[i]));
    }
  }
  NCCLCHECK(ncclGroupEnd());
#endif
  // Capture memory used by test buffers
  for (int g = 0; g < args->nGpus; ++g) {
    CUDACHECK(cudaSetDevice(args->gpus[g]));
    getGPUMemoryInfo(nullptr, &initFreeGpuMem[g + args->nGpus*2]);
    args->bufferMemory[args->thread] = std::max(args->bufferMemory[args->thread], initFreeGpuMem[g + args->nGpus] - initFreeGpuMem[g + args->nGpus*2]);
  }
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  /* Create device communicators based on test-specific requirements */
  if (deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
    if (test_ncclVersion < NCCL_VERSION(2,29,0)) {
      fprintf(stderr,
        "Incompatible NCCL versions. nccl-tests was compiled with NCCL %d, but is running with NCCL %d. "
        "The %d Device API is not compatible with versions before 2.29.\n",
        NCCL_VERSION_CODE, test_ncclVersion, NCCL_VERSION_CODE);
      return testInvalidUsage;
    }
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    if (!ncclTestEngine.getDevCommRequirements) {
      fprintf(stderr, "Device implementation %d is not supported by this test\n", deviceImpl);
      return testNotImplemented;
    }
    ncclCommProperties commProperties = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCLCHECK(ncclCommQueryProperties(args->comms[0], &commProperties));
    TESTCHECK(ncclTestEngine.getDevCommRequirements(deviceImpl, &reqs, &commProperties));
#else
    if (test_ncclVersion >= NCCL_VERSION(2,29,0)) {
      fprintf(stderr, "Incompatible NCCL versions. nccl-tests was compiled with NCCL 2.28, but is running with NCCL %d. "
        "The 2.28 Device API is not compatible with later.\n",
        test_ncclVersion);
      return testInvalidUsage;
    }
    ncclDevCommRequirements reqs = {};
    if (!ncclTestEngine.getDevCommRequirements ||
        !ncclTestEngine.getDevCommRequirements(deviceImpl, &reqs)) {
      fprintf(stderr, "Device implementation %d is not supported by this test\n", deviceImpl);
      return testNotImplemented;
    }
#endif

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < args->nGpus; i++) {
      NCCLCHECK(ncclDevCommCreate(args->comms[i], &reqs, args->devComms+i));
    }
    NCCLCHECK(ncclGroupEnd());
  }
  // Capture memory used by test buffers
  int64_t deviceCommMaxMem = 0;
  for (int g = 0; g < args->nGpus; ++g) {
    CUDACHECK(cudaSetDevice(args->gpus[g]));
    int64_t freeGpuMem;
    getGPUMemoryInfo(nullptr, &freeGpuMem);
    deviceCommMaxMem = std::max(deviceCommMaxMem, initFreeGpuMem[g + args->nGpus*2] - freeGpuMem);
  }
  *args->initGpuMem += deviceCommMaxMem;
#endif
  free(initFreeGpuMem);

  TESTCHECK(threadRunTests(args));

  // Cleanup: deregister buffers and destroy communicators
  for (int i=0; i<args->nGpus; i++) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
    if (test_ncclVersion >= NCCL_VERSION(2,27,0) && (local_register == SYMMETRIC_REGISTER)) {
      NCCLCHECK(ncclCommWindowDeregister(args->comms[i], (ncclWindow_t)args->sendRegHandles[i]));
      NCCLCHECK(ncclCommWindowDeregister(args->comms[i], (ncclWindow_t)args->recvRegHandles[i]));
    } else
#endif
    {
      if (local_register) NCCLCHECK(ncclCommDeregister(args->comms[i], args->sendRegHandles[i]));
      if (local_register) NCCLCHECK(ncclCommDeregister(args->comms[i], args->recvRegHandles[i]));
    }
#endif
    NCCLCHECK(ncclCommDestroy(args->comms[i]));
  }

  return testSuccess;
}

void* threadLauncher(void* thread_) {
  struct testThread* thread = (struct testThread*)thread_;
  thread->ret = thread->func(&thread->args);
  return NULL;
}
testResult_t threadLaunch(struct testThread* thread) {
  pthread_create(&thread->thread, NULL, threadLauncher, thread);
  return testSuccess;
}

testResult_t AllocateBuffs(void **sendbuff, size_t sendBytes, void **recvbuff, size_t recvBytes, void **expected, size_t nbytes, void **bias) {
  if(enable_rotating_tensor) {
    recvBytes = recvBytes + cache_bytes;
    nbytes = nbytes + cache_bytes;
  }
  if (memorytype == ncclFine) {
    if(HIP_VERSION >= 50700000) {
      CUDACHECK(hipExtMallocWithFlags(sendbuff, nbytes, hipDeviceMallocUncached));
      CUDACHECK(hipExtMallocWithFlags(recvbuff, nbytes, hipDeviceMallocUncached));
      if (bias) CUDACHECK(hipExtMallocWithFlags(bias, nbytes, hipDeviceMallocUncached));
      if (datacheck) CUDACHECK(hipExtMallocWithFlags(expected, recvBytes, hipDeviceMallocUncached));
    }
    else {
      CUDACHECK(hipExtMallocWithFlags(sendbuff, nbytes, hipDeviceMallocFinegrained));
      CUDACHECK(hipExtMallocWithFlags(recvbuff, nbytes, hipDeviceMallocFinegrained));
      if (bias) CUDACHECK(hipExtMallocWithFlags(bias, nbytes, hipDeviceMallocFinegrained));
      if (datacheck) CUDACHECK(hipExtMallocWithFlags(expected, recvBytes, hipDeviceMallocFinegrained));
    }
  }
  else if (memorytype == ncclHost) {
    CUDACHECK(hipHostMalloc(sendbuff, nbytes));
    CUDACHECK(hipHostMalloc(recvbuff, nbytes));
    if (bias) CUDACHECK(hipHostMalloc(bias, nbytes));
    if (datacheck) CUDACHECK(hipHostMalloc(expected, recvBytes));
  }
  else if (memorytype == ncclManaged) {
    CUDACHECK(cudaMallocManaged(sendbuff, nbytes));
    CUDACHECK(cudaMallocManaged(recvbuff, nbytes));
    if (bias) CUDACHECK(cudaMallocManaged(bias, nbytes));
    if (datacheck) CUDACHECK(cudaMallocManaged(expected, recvBytes));
#if 0
    CUDACHECK(cudaMemset(*sendbuff, 0, nbytes));
    CUDACHECK(cudaMemset(*recvbuff, 0, nbytes));
    if (datacheck) CUDACHECK(cudaMemset(*expected, 0, recvBytes));
#endif
  }
  else {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
    NCCLCHECK(ncclMemAlloc(sendbuff, nbytes));
    NCCLCHECK(ncclMemAlloc(recvbuff, nbytes));
    if (bias) CUDACHECK(cudaMalloc(bias, nbytes));
    if (datacheck) NCCLCHECK(ncclMemAlloc(expected, recvBytes));
#else
    CUDACHECK(cudaMalloc(sendbuff, nbytes));
    CUDACHECK(cudaMalloc(recvbuff, nbytes));
    if (bias) CUDACHECK(cudaMalloc(bias, nbytes));
    if (datacheck) CUDACHECK(cudaMalloc(expected, recvBytes));
#endif
  }
  CUDACHECK(hipMemset(*sendbuff, 1, nbytes));
  if (bias) CUDACHECK(hipMemset(*bias, 1, nbytes));
  if (datacheck) CUDACHECK(hipMemset(*expected, 1, recvBytes));
  return testSuccess;
}

testResult_t run(); // Main function

int main(int argc, char* argv[], char **envp) {
  // Make sure everyline is flushed so that we see the progress of the test
  setlinebuf(stdout);

  #if NCCL_VERSION_CODE >= NCCL_VERSION(2,4,0)
    ncclGetVersion(&test_ncclVersion);
  #else
    test_ncclVersion = NCCL_VERSION_CODE;
  #endif
  //printf("# nccl-tests version %s NCCL_VERSION_CODE=%d ncclGetVersion=%d\n", NCCL_TESTS_VERSION, NCCL_VERSION_CODE, test_ncclVersion);
  #if NCCL_VERSION_CODE >= NCCL_VERSION(2,0,0)
    test_opnum = 4;
    test_typenum = 9;
    if (NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0) && test_ncclVersion >= NCCL_VERSION(2,10,0)) {
      test_opnum++; // ncclAvg
    }
    if (NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0) && test_ncclVersion >= NCCL_VERSION(2,11,0)) {
      test_opnum++; // PreMulSum
    }
    #if defined(RCCL_BFLOAT16)
    if (NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0) && test_ncclVersion >= NCCL_VERSION(2,10,0)) {
      test_typenum++; // bfloat16
    }
    #endif
    #if defined(RCCL_FLOAT8)
    if (NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0) && test_ncclVersion >= NCCL_VERSION(2,10,0)) {
      test_typenum += 2; // fp8 e4m3,e5m2
    }
    #endif
  #endif
  loadRcclSyms();
  // Parse args
  double parsed;
  int longindex;
  char *output_file = nullptr;

  static struct option longopts[] = {
    {"nthreads", required_argument, 0, 't'},
    {"ngpus", required_argument, 0, 'g'},
    {"minbytes", required_argument, 0, 'b'},
    {"maxbytes", required_argument, 0, 'e'},
    {"stepbytes", required_argument, 0, 'i'},
    {"stepfactor", required_argument, 0, 'f'},
    {"iters", required_argument, 0, 'n'},
    {"agg_iters", required_argument, 0, 'm'},
    {"warmup_iters", required_argument, 0, 'w'},
    {"run_cycles", required_argument, 0, 'N'},
    {"parallel_init", required_argument, 0, 'p'},
    {"check", required_argument, 0, 'c'},
    {"op", required_argument, 0, 'o'},
    {"datatype", required_argument, 0, 'd'},
    {"root", required_argument, 0, 'r'},
    {"blocking", required_argument, 0, 'z'},
    {"stream_null", required_argument, 0, 'y'},
    {"timeout", required_argument, 0, 'T'},
    {"cudagraph", required_argument, 0, 'G'},
    {"report_cputime", required_argument, 0, 'C'},
    {"report_timestamps", required_argument, 0, 'S'},
    {"output_file", required_argument, 0, 'J'},
    {"average", required_argument, 0, 'a'},
    {"local_register", required_argument, 0, 'R'},
    {"cta_policy", required_argument, 0, 'x'},
    {"device_implementation", required_argument, 0, 'D'},
    {"device_cta_count", required_argument, 0, 'V'},
    {"memory_report", required_argument, 0, 'M'},
    {"memory_type", required_argument, 0, 'Y'},                     //RCCL
    {"cumask", required_argument, 0, 'u'},                          //RCCL
    {"out_of_place", required_argument, 0, 'O'},                    //RCCL
    {"delay_inout_place", required_argument, 0, 'q'},               //RCCL
    {"cache_flush", required_argument, 0, 'F'},                     //RCCL
    {"rotating_tensor", required_argument, 0, 'E'},                 //RCCL
    {"rccl_output_format", required_argument, 0, 'Z'},              //RCCL
    {"rccl_output_file", required_argument, 0, 'X'},                //RCCL (output file for Reporter class)
    {"output_algo_proto_channels", required_argument, 0, 'A'},      //RCCL (changed from M)
    {"help", no_argument, 0, 'h'},
    {}
  };

  while(1) {
    int c;
    c = getopt_long(argc, argv, "t:g:b:e:i:f:n:m:w:N:p:c:o:d:r:z:y:T:hG:C:a:R:x:D:V:J:S:M:Y:u:O:q:F:E:Z:X:A:", longopts, &longindex);

    if (c == -1)
      break;

    switch(c) {
      case 't':
        nThreads = strtol(optarg, NULL, 0);
        break;
      case 'g':
        nGpus = strtol(optarg, NULL, 0);
        break;
      case 'b':
        parsed = parsesize(optarg);
        if (parsed < 0) {
          fprintf(stderr, "invalid size specified for 'minbytes'\n");
          return -1;
        }
        minBytes = (size_t)parsed;
        break;
      case 'e':
        parsed = parsesize(optarg);
        if (parsed < 0) {
          fprintf(stderr, "invalid size specified for 'maxbytes'\n");
          return -1;
        }
        maxBytes = (size_t)parsed;
        break;
      case 'i':
        parsed = parsesize(optarg);
        if (parsed < 0) {
          fprintf(stderr, "invalid size specified for 'stepBytes'\n");
          return -1;
        }
        stepBytes = (size_t)parsed;
        break;
      case 'f':
        stepFactor = strtol(optarg, NULL, 0);
        break;
      case 'n':
        iters = (int)strtol(optarg, NULL, 0);
        break;
      case 'm':
#if NCCL_MAJOR > 2 || (NCCL_MAJOR >= 2 && NCCL_MINOR >= 2)
        agg_iters = (int)strtol(optarg, NULL, 0);
#else
        fprintf(stderr, "Option -m not supported before NCCL 2.2. Ignoring\n");
#endif
        break;
      case 'w':
        warmup_iters = (int)strtol(optarg, NULL, 0);
        break;
      case 'N':
        run_cycles = (int)strtol(optarg, NULL, 0);
        break;
      case 'p':
        parallel_init = (int)strtol(optarg, NULL, 0);
        break;
      case 'c':
        datacheck = (int)strtol(optarg, NULL, 0);
        break;
      case 'o':
        ncclop = ncclstringtoop(optarg);
        break;
      case 'd':
        nccltype = ncclstringtotype(optarg);
        break;
      case 'r':
        ncclroot = ncclstringtoroot(optarg);
        break;
      case 'z':
        blocking_coll = strtol(optarg, NULL, 0);
        break;
      case 'y':
        streamnull = strtol(optarg, NULL, 0);
        break;
      case 'T':
        timeout = strtol(optarg, NULL, 0);
        break;
      case 'G':
#if (NCCL_MAJOR > 2 || (NCCL_MAJOR >= 2 && NCCL_MINOR >= 9)) && HIP_VERSION >= 50221310
        cudaGraphLaunches = strtol(optarg, NULL, 0);
#else
        printf("Option -G (HIP graph) not supported before NCCL 2.9 + ROCm 5.2 Ignoring\n");
#endif
        break;
      case 'C':
        report_cputime = strtol(optarg, NULL, 0);
        break;
      case 'J':
        output_file = strdup(optarg);
        break;
      case 'S':
        report_timestamps = strtol(optarg, NULL, 0);
        break;
      case 'a':
        average = (int)strtol(optarg, NULL, 0);
        break;
      case 'R':
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
        local_register = (int)strtol(optarg, NULL, 0);
        if (local_register == SYMMETRIC_REGISTER && test_ncclVersion < NCCL_VERSION(2,27,0)) {
          printf("Option -R 2 (symmetric) is not supported before NCCL 2.27. Defaulting to local registration\n");
          local_register = LOCAL_REGISTER;
        }
#else
        printf("Option -R (register) is not supported before NCCL 2.19. Ignoring\n");
#endif
        break;
      case 'Y':
        memorytype = ncclstringtomtype(optarg);
        break;
      case 'u':
        {
          int nmasks = 0;
          char *mask = strtok(optarg, ",");
          while (mask != NULL && nmasks < 4) {
            cumask[nmasks++] = strtol(mask, NULL, 16);
            mask = strtok(NULL, ",");
          };
        }
	break;
      case 'O':
        enable_out_of_place = strtol(optarg, NULL, 0);
        enable_in_place = enable_out_of_place ? 0 : 1;
        break;
      case 'q':
        delay_inout_place = (int)strtol(optarg, NULL, 10);
      	break;
      case 'F':
        enable_cache_flush = strtol(optarg, NULL, 0);
        if (enable_cache_flush > 0) {
          hipDeviceProp_t deviceProps;
          CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProps, 0));
          gpu_block3 = deviceProps.multiProcessorCount * 60;
        }
        break;
      case 'E':
        enable_rotating_tensor = strtol(optarg, NULL, 0);
        break;
      case 'Z':
        rccl_output_format = optarg;
        break;
      case 'X':
        rccl_output_file = optarg;
        break;
      case 'A':
        output_algo_proto_channels = strtol(optarg, NULL, 0);
        if(rcclTestsGetAlgoInfo == NULL || rcclTestsGetAlgoName == NULL || rcclTestsGetProtocolName == NULL) output_algo_proto_channels = 0;
        break;
      case 'M':
        memory_report = (int)strtol(optarg, NULL, 0);
        break;
      case 'x':
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
        ctaPolicy = (int)strtol(optarg, NULL, 0);
        if (ctaPolicy > 1 && test_ncclVersion < NCCL_VERSION(2,28,0)) {
          printf("Option -x (cta_policy) %d is not supported before NCCL 2.28. Ignoring\n", ctaPolicy);
          ctaPolicy = -1;
        }
#else
        printf("Option -x (cta_policy) is not supported before NCCL 2.27. Ignoring\n");
#endif
        break;
      case 'D':
        if (test_ncclVersion >= NCCL_VERSION(2,28,0)) {
          deviceImpl = (int)strtol(optarg, NULL, 0);
        } else {
          fprintf(stderr, "Option -D (device implementation) requires NCCL >= 2.28.0\n");
          return -1;
        }
        break;
      case 'V':
        if (test_ncclVersion >= NCCL_VERSION(2,28,0)) {
          deviceCtaCount = (int)strtol(optarg, NULL, 0);
          if (deviceCtaCount <= 0 || deviceCtaCount > 128) {
            fprintf(stderr, "device_cta_count (-V) must be positive and less than 128, got %d. "
                    "Using default value 16.\n", deviceCtaCount);
            deviceCtaCount = 16;
          }
        } else {
          fprintf(stderr, "Option -V (device CTA count) requires NCCL >= 2.28.0\n");
          return -1;
        }
        break;
      case 'h':
      default:
        if (c != 'h') printf("invalid option '%c'\n", c);
        printf("USAGE: %s \n\t"
            "[-t,--nthreads <num threads>] \n\t"
            "[-g,--ngpus <gpus per thread>] \n\t"
            "[-b,--minbytes <min size in bytes>] \n\t"
            "[-e,--maxbytes <max size in bytes>] \n\t"
            "[-i,--stepbytes <increment size>] \n\t"
            "[-f,--stepfactor <increment factor>] \n\t"
            "[-n,--iters <iteration count>] \n\t"
            "[-m,--agg_iters <aggregated iteration count>] \n\t"
            "[-w,--warmup_iters <warmup iteration count>] \n\t"
            "[-N,--run_cycles <cycle count> run & print each cycle (default: 1; 0=infinite)] \n\t"
            "[-p,--parallel_init <0/1>] \n\t"
            "[-c,--check <check iteration count>] \n\t"
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0)
            "[-o,--op <sum/prod/min/max/avg/mulsum/all>] \n\t"
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
            "[-o,--op <sum/prod/min/max/avg/all>] \n\t"
#else
            "[-o,--op <sum/prod/min/max/all>] \n\t"
#endif
            "[-d,--datatype <nccltype/all>] \n\t"
            "[-r,--root <root/all>] \n\t"
            "[-z,--blocking <0/1>] \n\t"
            "[-y,--stream_null <0/1>] \n\t"
            "[-T,--timeout <time in seconds>] \n\t"
            "[-G,--cudagraph <num graph launches>] \n\t"
            "[-C,--report_cputime <0/1>] \n\t"
            "[-S,--report_timestamps <0/1> report timestamps (default 0)] \n\t"
            "[-J,--output_file <file> write output to filepath, if accessible. Infer type from suffix (only json supported presently.)] \n\t"
            "[-a,--average <0/1/2/3> report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>] \n\t"
            "[-R,--local_register <0/1/2> enable local (1) or symmetric (2) buffer registration on send/recv buffers (default: disable (0))] \n\t"
            "[-x,--cta_policy <0/1/2> set CTA policy (NCCL_CTA_POLICY_DEFAULT (0), NCCL_CTA_POLICY_EFFICIENCY (1), NCCL_CTA_POLICY_ZERO (2)) (default: do not set)] \n\t"
            "[-D,--device_implementation <implementation number> enable device implementation (default: 0, use NCCL implementation; requires -R 2 if > 0)] \n\t"
            "[-V,--device_cta_count <number> set number of CTAs for device implementation (default: 16)] \n\t"
            "[-M,--memory_report <0/1> enable memory usage report (default: 0)] \n\t"
            "[-Y,--memory_type <coarse/fine/host/managed>] \n\t"                                                    //RCCL
            "[-u,--cumask <d0,d1,d2,d3>] \n\t"                                                                      //RCCL
            "[-O,--out_of_place <0/1>] \n\t"                                                                        //RCCL
            "[-q,--delay_inout_place <delay between out-of-place and in-place in microseconds>] \n\t"               //RCCL
            "[-F,--cache_flush <number of iterations between instruction cache flush>] \n\t"                        //RCCL
            "[-E,--rotating_tensor <0/1>] \n\t"                                                                     //RCCL
            "[-Z,--rccl_output_format <output format <csv|json>] \n\t"                                              //RCCL
            "[-X,--rccl_output_file <file> RCCL Reporter output file for csv/json (used with -Z)] \n\t"             //RCCL
            "[-A,--output_algo_proto_channels <0/1> enable algorithm/protocol/channels output (default: 0)] \n\t"   //RCCL
            "[-h,--help]\n",
          basename(argv[0]));
        return 0;
    }
  }

  CUDACHECK(cudaGetDeviceCount(&numDevices));
#ifndef MPI_SUPPORT
  if (nGpus > numDevices)
  {
      fprintf(stderr, "[ERROR] The number of requested GPUs (%d) is greater than the number of GPUs available (%d)\n", nGpus, numDevices);
      return testNcclError;
  }
#endif
  if (minBytes > maxBytes) {
    fprintf(stderr, "invalid sizes for 'minbytes' and 'maxbytes': %llu > %llu\n",
           (unsigned long long)minBytes,
           (unsigned long long)maxBytes);
    return -1;
  }
  if (!rccl_output_format.empty()) {
    if (!(rccl_output_format == "csv" || rccl_output_format == "json")) {
      std::cerr << "Invalid --rccl_output_format: " << rccl_output_format << "\n";
      return -1;
    }
  }
  if (deviceImpl > 0 && (local_register != SYMMETRIC_REGISTER)) {
    fprintf(stderr, "device implementation (-D > 0) requires enabling symmetric memory registration (-R 2)\n");
    return -1;
  }

#ifdef MPI_SUPPORT
  MPI_Init(&argc, &argv);
#endif

  const output_file_type_t output_file_type = classifyOutputFile(output_file);
  outputFileInit(output_file_type, output_file, argc, argv, envp);

  if(output_file) {
    free(output_file);
    output_file = nullptr;
  }

  testResult_t result = run();

  outputFileFinalize(output_file_type);

  TESTCHECK(result);

  return 0;
}

#ifdef MPI_SUPPORT
// parse int for base 2/10/16, will ignore first whitespaces
static bool parseInt(char *s, int *num) {
  char *p = NULL;
  if (!s || !num)
    return false;
  while (*s && isspace(*s)) ++s;
  if (!*s) return false;

  if (strncasecmp(s, "0b", 2) == 0)
    *num = (int)strtoul(s + 2, &p, 2);
  else
    *num = (int)strtoul(s, &p, 0);

  if (p == s)
    return false;
  return true;
}
#endif

testResult_t run() {
  int totalProcs = 1, proc = 0, ncclProcs = 1, ncclProc = 0, color = 0;
  int localRank = 0;
  int localSize = 0;
  char hostname[1024];
  getHostName(hostname, 1024);

  hipDeviceProp_t devProp;
  CUDACHECK(hipGetDeviceProperties(&devProp, 0));
  if (IsArchMatch(devProp.gcnArchName, "gfx942")) {
    PRINT("On gfx942 architecture, using FNUZ FP8 types");
    rccl_float8_useFnuz = true;
  }

#ifdef MPI_SUPPORT
  MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  std::vector<uint64_t> hostHashs(totalProcs);
  hostHashs[proc] = getHostHash(hostname);
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs.data(), sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
  for (int p=0; p<totalProcs; p++) {
    if (p == proc) break;
    if (hostHashs[p] == hostHashs[proc]) localRank++;
  }

  char *splitMaskEnv = NULL;
  if ((splitMaskEnv = getenv("NCCL_TESTS_SPLIT_MASK"))) {
    color = proc & strtoul(splitMaskEnv, NULL, 16);
  } else if ((splitMaskEnv = getenv("NCCL_TESTS_SPLIT"))) {
    if (
      (strncasecmp(splitMaskEnv, "AND", strlen("AND")) == 0 && parseInt(splitMaskEnv + strlen("AND"), &color)) ||
      (strncasecmp(splitMaskEnv, "&", strlen("&")) == 0 && parseInt(splitMaskEnv + strlen("&"), &color))
    )
        color = proc & color;
    if (
      (strncasecmp(splitMaskEnv, "OR", strlen("OR")) == 0 && parseInt(splitMaskEnv + strlen("OR"), &color)) ||
      (strncasecmp(splitMaskEnv, "|", strlen("|")) == 0 && parseInt(splitMaskEnv + strlen("|"), &color))
    )
        color = proc | color;
    if (
      (strncasecmp(splitMaskEnv, "MOD", strlen("MOD")) == 0 && parseInt(splitMaskEnv + strlen("MOD"), &color)) ||
      (strncasecmp(splitMaskEnv, "%", strlen("%")) == 0 && parseInt(splitMaskEnv + strlen("%"), &color))
    )
        color = proc % color;
    if (
      (strncasecmp(splitMaskEnv, "DIV", strlen("DIV")) == 0 && parseInt(splitMaskEnv + strlen("DIV"), &color)) ||
      (strncasecmp(splitMaskEnv, "/", strlen("/")) == 0 && parseInt(splitMaskEnv + strlen("/"), &color))
    )
        color = proc / color;
  }

  MPI_Comm mpi_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, proc, &mpi_comm);
  MPI_Comm_size(mpi_comm, &ncclProcs);
  MPI_Comm_rank(mpi_comm, &ncclProc);

  for (int p=0; p<totalProcs; p++) {
    if (hostHashs[p] == hostHashs[proc]) localSize++;
  }
  if (nGpus * localSize > numDevices && numDevices != 1)
  {
      fprintf(stderr, "[ERROR] The number of requested GPUs (%d) is greater than the number of GPUs available (%d) on node (%s)\n", nGpus*localSize, numDevices, hostname);
      return testNcclError;
  }
#endif
  is_main_thread = is_main_proc = (proc == 0) ? 1 : 0;

  jsonIdentifyWriter(is_main_thread);

  size_t maxMem = ~0;
  testResult_t report_result = writeDeviceReport(&maxMem, localRank, proc, totalProcs, color, hostname, program_invocation_short_name);
  if(report_result != testSuccess) {
    return report_result;
  }

  // Reserve 1GiB of memory for each 16GiB installed, but limit to a max of 4GiB
  const size_t GB = (1ULL << 30);
  size_t reserveMem =  std::min(DIVUP(maxMem, 16*GB) * 1*GB, 4*GB);
  // If the program is all_reduce_bias, enable bias
  if (strcmp(program_invocation_short_name, "all_reduce_bias_perf") == 0) test_bias = 1;
  // We need sendbuff, recvbuff, expected (when datacheck enabled), bias (when bias enabled), plus 1G for the rest.
  size_t memMaxBytes = (maxMem - reserveMem - 1*GB) / (datacheck ? (test_bias ? 4 : 3) : (test_bias ? 3 : 2));
  if (maxBytes > memMaxBytes) {
    maxBytes = memMaxBytes;
    if (minBytes > maxBytes) minBytes = maxBytes;
    if (proc == 0) printf("#\n# Reducing maxBytes to %ld due to memory limitation\n", maxBytes);
  }

  ncclUniqueId ncclId;
  if (ncclProc == 0) {
    NCCLCHECK(ncclGetUniqueId(&ncclId));
  }
#ifdef MPI_SUPPORT
  MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, mpi_comm);
  MPI_Barrier(MPI_COMM_WORLD); // Ensure Bcast is complete for HCOLL
#endif

  std::vector<int> gpus(nGpus*nThreads);
  std::vector<cudaStream_t> streams(nGpus*nThreads);
  std::vector<void*> sendbuffs(nGpus*nThreads);
  std::vector<void*> recvbuffs(nGpus*nThreads);
  std::vector<void*> bias(nGpus*nThreads);
  std::vector<void*> expected(nGpus*nThreads);
  size_t sendBytes, recvBytes;

  ncclTestEngine.getBuffSize(&sendBytes, &recvBytes, (size_t)maxBytes, (size_t)ncclProcs*nGpus*nThreads);

  char* envstr = getenv("NCCL_TESTS_DEVICE");
  int gpu0 = envstr ? atoi(envstr) : -1;
  for (int i=0; i<nGpus*nThreads; i++) {
    gpus[i] = ((gpu0 != -1 ? gpu0 : localRank*nThreads*nGpus) + i)%numDevices;
    CUDACHECK(cudaSetDevice(gpus[i]));
    if(test_bias) {
      TESTCHECK(AllocateBuffs(sendbuffs.data()+i, sendBytes, recvbuffs.data()+i, recvBytes, expected.data()+i, (size_t)maxBytes, bias.data()+i));
    } else {
      TESTCHECK(AllocateBuffs(sendbuffs.data()+i, sendBytes, recvbuffs.data()+i, recvBytes, expected.data()+i, (size_t)maxBytes, NULL));
    }
    if (streamnull) {
      streams[i] = NULL;
    }
    else {
      CUDACHECK(cudaStreamCreateWithFlags(streams.data()+i, cudaStreamNonBlocking));
    }
    int archMajor, archMinor;
    CUDACHECK(cudaDeviceGetAttribute(&archMajor, cudaDevAttrComputeCapabilityMajor, gpus[i]));
    CUDACHECK(cudaDeviceGetAttribute(&archMinor, cudaDevAttrComputeCapabilityMinor, gpus[i]));
    minCudaArch = std::min(minCudaArch, 100*archMajor + 10*archMinor);
  }

#ifdef MPI_SUPPORT
  MPI_Allreduce(MPI_IN_PLACE, &minCudaArch, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
#if defined(RCCL_FLOAT8)
  if (NCCL_VERSION_CODE >= NCCL_VERSION(2,24,0) && test_ncclVersion >= NCCL_VERSION(2,24,0)) {
    if (minCudaArch < 900) { // Filter out fp8 on pre-Hopper hardware
      int n = 0;
      for (int i=0; i < test_typenum; i++) {
        if (!(test_types[i] == ncclFloat8e4m3 || test_types[i] == ncclFloat8e5m2)) {
          test_types[n] = test_types[i];
          test_typenames[n] = test_typenames[i];
          n += 1;
        }
      }
      test_typenum = n;
    }
  }
#endif

  //if parallel init is not selected, use main thread to initialize NCCL
  ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nThreads*nGpus);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
  std::vector<void*> sendRegHandles(nThreads*nGpus, nullptr);
  std::vector<void*> recvRegHandles(nThreads*nGpus, nullptr);
#endif
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  std::vector<ncclDevComm> devComms(nThreads*nGpus);
#endif
  std::vector<int64_t> initGpuMem(nThreads, 0);
  std::vector<int64_t> bufferMemory(nThreads, 0);
  if (!parallel_init) {
    // Capture the memory used by the GPUs before initializing the NCCL communicators
    int64_t* initFreeGpuMem = (int64_t*)calloc(nGpus*3, sizeof(int64_t));
    for (int g = 0; g < nGpus; ++g) {
      CUDACHECK(cudaSetDevice(gpus[g]));
      getGPUMemoryInfo(nullptr, &initFreeGpuMem[g]);
    }
    //if parallel init is not selected, use main thread to initialize NCCL
    TESTCHECK(initComms(comms, nGpus*nThreads, ncclProc*nThreads*nGpus, ncclProcs*nThreads*nGpus, gpus.data(), ncclId));

     // Capture the memory used by the GPUs after initializing the NCCL communicators
     for (int g = 0; g < nGpus; ++g) {
       CUDACHECK(cudaSetDevice(gpus[g]));
       getGPUMemoryInfo(nullptr, &initFreeGpuMem[g + nGpus]);
     }
     for ( size_t t = 0; t < nThreads; ++t) {
       for (int g = 0; g < nGpus; ++g) {
         initGpuMem[t] = std::max(initGpuMem[t], initFreeGpuMem[g] - initFreeGpuMem[g + nGpus]);
       }
     }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
     NCCLCHECK(ncclGroupStart());
     for (int i=0; i<nGpus*nThreads; i++) {
       CUDACHECK(cudaSetDevice(gpus[i]));
       if(test_bias) {
         TESTCHECK(AllocateBuffs(sendbuffs.data()+i, sendBytes, recvbuffs.data()+i, recvBytes, expected.data()+i, (size_t)maxBytes, bias.data()+i));
       } else {
         TESTCHECK(AllocateBuffs(sendbuffs.data()+i, sendBytes, recvbuffs.data()+i, recvBytes, expected.data()+i, (size_t)maxBytes, NULL));
       }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
       if (test_ncclVersion >= NCCL_VERSION(2,27,0) && (local_register == SYMMETRIC_REGISTER)) {
         NCCLCHECK(ncclCommWindowRegister(comms[i], sendbuffs[i], maxBytes, (ncclWindow_t*)&sendRegHandles[i], NCCL_WIN_COLL_SYMMETRIC));
         NCCLCHECK(ncclCommWindowRegister(comms[i], recvbuffs[i], maxBytes, (ncclWindow_t*)&recvRegHandles[i], NCCL_WIN_COLL_SYMMETRIC));
       } else
#endif
       {
         if (local_register) NCCLCHECK(ncclCommRegister(comms[i], sendbuffs[i], maxBytes, &sendRegHandles[i]));
         if (local_register) NCCLCHECK(ncclCommRegister(comms[i], recvbuffs[i], maxBytes, &recvRegHandles[i]));
       }
     }
     NCCLCHECK(ncclGroupEnd());
#endif
     // Capture memory used by after allocating buffers
     for (int g = 0; g < nGpus; ++g) {
       CUDACHECK(cudaSetDevice(gpus[g]));
       getGPUMemoryInfo(nullptr, &initFreeGpuMem[g + nGpus*2]);
     }
     for ( size_t t = 0; t < nThreads; ++t) {
      for (int g = 0; g < nGpus; ++g) {
        bufferMemory[t] = std::max(bufferMemory[t], initFreeGpuMem[g + nGpus] - initFreeGpuMem[g + nGpus*2]);
      }
     }
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
     /* Create device communicators based on test-specific requirements */
     if (deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
      if (test_ncclVersion < NCCL_VERSION(2,29,0)) {
        fprintf(stderr,
          "Incompatible NCCL versions. nccl-tests was compiled with NCCL %d, but is running with NCCL %d. "
          "The %d Device API is not compatible with versions before 2.29.\n",
          NCCL_VERSION_CODE, test_ncclVersion, NCCL_VERSION_CODE);
        return testInvalidUsage;
      }
      ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      if (!ncclTestEngine.getDevCommRequirements) {
        fprintf(stderr, "Device implementation %d is not supported by this test\n", deviceImpl);
        return testNotImplemented;
      }
      ncclCommProperties commProperties = NCCL_COMM_PROPERTIES_INITIALIZER;
      NCCLCHECK(ncclCommQueryProperties(comms[0], &commProperties));
      TESTCHECK(ncclTestEngine.getDevCommRequirements(deviceImpl, &reqs, &commProperties));
#else
      if (test_ncclVersion >= NCCL_VERSION(2,29,0)) {
        fprintf(stderr, "Incompatible NCCL versions. nccl-tests was compiled with NCCL 2.28, but is running with NCCL %d. "
          "The 2.28 Device API is not compatible with later versions.\n", test_ncclVersion);
        return testInvalidUsage;
      }
      ncclDevCommRequirements reqs = {};
      if (!ncclTestEngine.getDevCommRequirements ||
          !ncclTestEngine.getDevCommRequirements(deviceImpl, &reqs)) {
        fprintf(stderr, "Device implementation %d is not supported by this test\n", deviceImpl);
        return testNotImplemented;
      }
#endif

       NCCLCHECK(ncclGroupStart());
       for (int i = 0; i < nGpus * nThreads; i++) {
         NCCLCHECK(ncclDevCommCreate(comms[i], &reqs, devComms.data()+i));
       }
       NCCLCHECK(ncclGroupEnd());
     }
     int64_t deviceCommMaxMem = 0;
     for (int g = 0; g < nGpus; ++g) {
       CUDACHECK(cudaSetDevice(gpus[g]));
       int64_t freeGpuMem;
       getGPUMemoryInfo(nullptr, &freeGpuMem);
       deviceCommMaxMem = std::max(deviceCommMaxMem, initFreeGpuMem[g + nGpus*2] - freeGpuMem);
     }
     for ( size_t t = 0; t < nThreads; ++t) {
       initGpuMem[t] += deviceCommMaxMem;
     }
#endif
    free(initFreeGpuMem);
  }

  std::vector<int> errors(nThreads);
  std::vector<double> bw(nThreads);
  std::vector<int64_t> devMemUsed(nThreads);
  double* delta;
  CUDACHECK(hipHostMalloc(&delta, sizeof(double)*nThreads*NUM_BLOCKS, cudaHostAllocPortable | cudaHostAllocMapped));
  std::vector<int> bw_count(nThreads);
  for (int t=0; t<nThreads; t++) {
    bw[t] = 0.0;
    errors[t] = bw_count[t] = 0;
    devMemUsed[t] = std::numeric_limits<int64_t>::min();
  }

  fflush(stdout);
  
  // RCCL: Call NCCL's refactored header function with RCCL-specific parameters
  writeResultHeader(report_cputime, report_timestamps, enable_out_of_place, enable_in_place, output_algo_proto_channels);
  
  // RCCL: Initialize Reporter for file output (-Z flag)
  Reporter reporter(rccl_output_file, rccl_output_format);

  std::vector<testThread> threads(nThreads);
  memset(threads.data(), 0, sizeof(struct testThread)*nThreads);

  for (int t=nThreads-1; t>=0; t--) {
    threads[t].args.minbytes=minBytes;
    threads[t].args.maxbytes=maxBytes;
    threads[t].args.stepbytes=stepBytes;
    threads[t].args.stepfactor=stepFactor;
    threads[t].args.localRank = localRank;

    threads[t].args.totalProcs = totalProcs;
    threads[t].args.nProcs=ncclProcs;
    threads[t].args.proc=ncclProc;
    threads[t].args.nThreads=nThreads;
    threads[t].args.thread=t;
    threads[t].args.nGpus=nGpus;
    threads[t].args.gpus=gpus.data()+t*nGpus;
    threads[t].args.sendbuffs = sendbuffs.data()+t*nGpus;
    threads[t].args.recvbuffs = recvbuffs.data()+t*nGpus;
    threads[t].args.bias = bias.data()+t*nGpus;
    threads[t].args.expected = expected.data()+t*nGpus;
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
    threads[t].args.devComms = devComms.data()+t*nGpus;
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
    threads[t].args.sendRegHandles = sendRegHandles.data()+t*nGpus;
    threads[t].args.recvRegHandles = recvRegHandles.data()+t*nGpus;
#endif
    threads[t].args.ncclId = ncclId;
    threads[t].args.comms=comms+t*nGpus;
    threads[t].args.streams=streams.data()+t*nGpus;
    threads[t].args.enable_out_of_place=enable_out_of_place;
    threads[t].args.enable_in_place=enable_in_place;
    threads[t].args.enable_cache_flush = enable_cache_flush;
    threads[t].args.enable_rotating_tensor = enable_rotating_tensor;
    threads[t].args.errors=errors.data()+t;
    threads[t].args.bw=bw.data()+t;
    threads[t].args.bw_count=bw_count.data()+t;
    threads[t].args.initGpuMem = initGpuMem.data() + t;
    threads[t].args.bufferMemory = bufferMemory.data() + t;
    threads[t].args.devMemUsed = devMemUsed.data() + t;

    threads[t].args.reportErrors = datacheck;
    threads[t].args.reporter = &reporter;

    threads[t].func = parallel_init ? threadInit : threadRunTests;
    if (t)
      TESTCHECK(threadLaunch(threads.data()+t));
    else
      TESTCHECK(threads[t].func(&threads[t].args));
  }

  // Wait for other threads and accumulate stats and errors
  for (int t=nThreads-1; t>=0; t--) {
    if (t) pthread_join(threads[t].thread, NULL);
    TESTCHECK(threads[t].ret);
    if (t) {
      errors[0] += errors[t];
      bw[0] += bw[t];
      bw_count[0] += bw_count[t];
      devMemUsed[0] = std::max(devMemUsed[0], devMemUsed[t]);
      initGpuMem[0] = std::max(initGpuMem[0], initGpuMem[t]);
      bufferMemory[0] = std::max(bufferMemory[0], bufferMemory[t]);
    }
  }

#ifdef MPI_SUPPORT
  MPI_Allreduce(MPI_IN_PLACE, &errors[0], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &devMemUsed[0], 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, initGpuMem.data(), 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, bufferMemory.data(), 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
#endif

  if (!parallel_init) {
    for(int i=0; i<nGpus*nThreads; ++i) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
      if (test_ncclVersion >= NCCL_VERSION(2,27,0) && (local_register == SYMMETRIC_REGISTER)) {
        NCCLCHECK(ncclCommWindowDeregister(comms[i], (ncclWindow_t)sendRegHandles[i]));
        NCCLCHECK(ncclCommWindowDeregister(comms[i], (ncclWindow_t)recvRegHandles[i]));
      } else
#endif
      {
        if (local_register) NCCLCHECK(ncclCommDeregister(comms[i], sendRegHandles[i]));
        if (local_register) NCCLCHECK(ncclCommDeregister(comms[i], recvRegHandles[i]));
      }
#endif
      NCCLCHECK(ncclCommDestroy(comms[i]));
    }
  }
  free(comms);

  // Free off CUDA allocated memory
  for (int i=0; i<nGpus*nThreads; i++) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
    if (sendbuffs[i]) NCCLCHECK(ncclMemFree((char*)sendbuffs[i]));
    if (recvbuffs[i]) NCCLCHECK(ncclMemFree((char*)recvbuffs[i]));
    if (datacheck) NCCLCHECK(ncclMemFree(expected[i]));
#else
    if (sendbuffs[i]) CUDACHECK(cudaFree((char*)sendbuffs[i]));
    if (recvbuffs[i]) CUDACHECK(cudaFree((char*)recvbuffs[i]));
    if (bias[i]) CUDACHECK(cudaFree((char*)bias[i]));
    if (datacheck) CUDACHECK(cudaFree(expected[i]));
#endif
  }
  CUDACHECK(cudaFreeHost(delta));
  envstr = getenv("NCCL_TESTS_MIN_BW");
  const double check_avg_bw = envstr ? atof(envstr) : -1;
  bw[0] /= bw_count[0];

  writeResultFooter(errors.data(), bw.data(), check_avg_bw, program_invocation_short_name);
  if (memory_report) {
    memInfo_t memInfos[3];
    memInfos[0] = { initGpuMem[0], "Initialization" };
    memInfos[1] = { bufferMemory[0], "User-Allocated" };
    memInfos[2] = { devMemUsed[0], "Collective" };
    writeMemInfo(memInfos, 3);
  }
  finalizeFooter();

#ifdef MPI_SUPPORT
  MPI_Comm_free(&mpi_comm);
  MPI_Finalize();
#endif

  reporter.writeFile();
  writeErrors();

  // 'cuda-memcheck --leak-check full' requires this
  cudaDeviceReset();

  if (errors[0] || bw[0] < check_avg_bw*(0.9))
    return testNumResults;
  else
    return testSuccess;
}
