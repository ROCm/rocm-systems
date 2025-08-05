
/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "rccl_float8.h"
#include <cstdio>
#include <ctype.h>
#include <getopt.h>
#include <hip/hip_bfloat16.h>
#include <libgen.h>
#include <pthread.h>
#include <string.h>
#include <type_traits>
#include <utility>
#include <vector>

// #define DEBUG_PRINT

#include "git_version.h"
#include "verifiable.h"

int     test_ncclVersion = 0;  // init'd with ncclGetVersion()
int32_t gpu_block3;
size_t  cache_bytes = 192 * 1024 * 1024;  // Use 192MB

#if NCCL_MAJOR >= 2
ncclDataType_t test_types[ncclNumTypes] = { ncclInt8,
                                            ncclUint8,
                                            ncclInt32,
                                            ncclUint32,
                                            ncclInt64,
                                            ncclUint64,
                                            ncclHalf,
                                            ncclFloat,
                                            ncclDouble
#    if RCCL_BFLOAT16 == 1
                                            ,
                                            ncclBfloat16
#    endif
#    if RCCL_FLOAT8 == 1
                                            ,
                                            ncclFp8E4M3,
                                            ncclFp8E5M2
#    endif
};
const char* test_typenames[ncclNumTypes] = { "int8",
                                             "uint8",
                                             "int32",
                                             "uint32",
                                             "int64",
                                             "uint64",
                                             "half",
                                             "float",
                                             "double"
#    if RCCL_BFLOAT16 == 1
                                             ,
                                             "bfloat16"
#    endif
#    if RCCL_FLOAT8 == 1
                                             ,
                                             "fp8_e4m3",
                                             "fp8_e5m2"
#    endif
};
int test_typenum = -1;

const char* test_opnames[] = { "sum", "prod", "max", "min", "avg", "mulsum" };
ncclRedOp_t test_ops[]     = {
    ncclSum,
    ncclProd,
    ncclMax,
    ncclMin
#    if NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
    ,
    ncclAvg
#    endif
#    if NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0)
    ,
    ncclNumOps  // stand in for ncclRedOpCreatePreMulSum() created on-demand
#    endif
};
int test_opnum = -1;
#else
ncclDataType_t test_types[ncclNumTypes] = { ncclChar,   ncclInt,   ncclHalf,  ncclFloat,
                                            ncclDouble, ncclInt64, ncclUint64 };
const char*    test_typenames[ncclNumTypes] = { "char",   "int",   "half",  "float",
                                                "double", "int64", "uint64" };
int            test_typenum                 = 7;
const char*    test_opnames[]               = { "sum", "prod", "max", "min" };
ncclRedOp_t    test_ops[]                   = { ncclSum, ncclProd, ncclMax, ncclMin };
int            test_opnum                   = 4;
#endif

const char* test_memorytypes[nccl_NUM_MTYPES] = { "coarse", "fine", "host", "managed" };

// For libnccl's < 2.13
extern "C" __attribute__((weak)) char const*
ncclGetLastError(ncclComm_t comm)
{
    return "";
}

int              is_main_proc   = 0;
thread_local int is_main_thread = 0;

// Command line parameter defaults
static int      nThreads      = 1;
static int      nGpus         = 1;
static size_t   minBytes      = 32 * 1024 * 1024;
static size_t   maxBytes      = 32 * 1024 * 1024;
static size_t   stepBytes     = 1 * 1024 * 1024;
static size_t   stepFactor    = 1;
static int      datacheck     = 1;
static int      warmup_iters  = 5;
static int      iters         = 20;
static int      agg_iters     = 1;
static int      run_cycles    = 1;
static int      ncclop        = ncclSum;
static int      nccltype      = ncclFloat;
static int      ncclroot      = 0;
static int      parallel_init = 0;
static int      blocking_coll = 0;
static int      memorytype    = 0;
static uint32_t cumask[4];
static int      streamnull        = 0;
static int      timeout           = 0;
static int      cudaGraphLaunches = 0;
std::string     output_file;
std::string     output_format;
static int      report_cputime = 0;
// Report average iteration time: (0=RANK0,1=AVG,2=MIN,3=MAX)
static int average                = 1;
static int numDevices             = 1;
static int delay_inout_place      = 0;
static int enable_out_of_place    = 1;
static int enable_in_place        = 1;
static int enable_cache_flush     = 0;
static int enable_rotating_tensor = 0;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
static int local_register = 0;
#endif

Reporter::Reporter(std::string fileName, std::string outputFormat)
: _outputFormat(outputFormat)
{
    if(!fileName.empty())
    {
        if(isMainThread())
        {
            _out         = std::ofstream(fileName, std::ios_base::out);
            _outputValid = true;
            if(_outputFormat == "csv")
            {
                _out << "numCycle, ";
                _out << "collective, ";
#ifdef MPI_SUPPORT
                _out << "ranks, rankspernode, gpusperrank, ";
#else
                _out << "gpus, ";
#endif
                _out << "size, type, redop, inplace, time, algbw, busbw, #wrong\n";
            }
        }
    }
}

void
Reporter::setParameters(const size_t numCycle, const char* name, const char* typeName,
                        const char* opName)
{
    if(!isMainThread() || !_outputValid) return;

    _numCycle       = numCycle;
    _collectiveName = name;
    _typeName       = typeName;
    _opName         = opName;
}

void
Reporter::addResult(int gpusPerRank, int ranksPerNode, int totalRanks, size_t numBytes,
                    int inPlace, double timeUsec, double algBw, double busBw,
                    int64_t wrongElts)
{
    if(!isMainThread() || !_outputValid) return;

    std::vector<std::pair<std::string, std::string>> outputValuesKeys;
    std::string wrongEltsStr = (wrongElts == -1) ? "N/A" : std::to_string(wrongElts);
    int         nodes        = totalRanks / ranksPerNode;

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

    for(auto iter = outputValuesKeys.begin(); iter != outputValuesKeys.end(); ++iter)
    {
        if(_outputFormat == "csv")
        {
            _out << iter->first;
            if(std::next(iter) != outputValuesKeys.end())
            {
                _out << ", ";
            }
        }
        else
        {  // json
            if(iter == outputValuesKeys.begin())
            {
                _out << "{";
            }
            _out << "\"" << iter->second << "\":" << iter->first;
            if(std::next(iter) != outputValuesKeys.end())
            {
                _out << ", ";
            }
            else
            {
                _out << "}";
            }
        }
    }
    _out << std::endl;
}

bool
Reporter::isMainThread()
{
    return is_main_thread == 1;
}

#define NUM_BLOCKS 32

#ifndef CHECK_HIP_ERROR
#    define CHECK_HIP_ERROR(error)                                                       \
        if(error != hipSuccess)                                                          \
        {                                                                                \
            fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n", hipGetErrorString(error),  \
                    error, __FILE__, __LINE__);                                          \
            exit(EXIT_FAILURE);                                                          \
        }
#endif

extern "C" __global__ void
flush_icache()
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

static double
parsesize(const char* value)
{
    long long int units;
    double        size;
    char          size_lit[2];

    int count = sscanf(value, "%lf %1s", &size, size_lit);

    switch(count)
    {
        case 2:
            switch(size_lit[0])
            {
                case 'G':
                case 'g': units = 1024 * 1024 * 1024; break;
                case 'M':
                case 'm': units = 1024 * 1024; break;
                case 'K':
                case 'k': units = 1024; break;
                default: return -1.0;
            };
            break;
        case 1: units = 1; break;
        default: return -1.0;
    }

    return size * units;
}

static bool
minReqVersion(int rmajor, int rminor, int rpatch)
{
    int version;
    int major, minor, patch, rem;
    ncclGetVersion(&version);

    if(version < 10000)
    {
        major = version / 1000;
        rem   = version % 1000;
        minor = rem / 100;
        patch = rem % 100;
    }
    else
    {
        major = version / 10000;
        rem   = version % 10000;
        minor = rem / 100;
        patch = rem % 100;
    }

    if(major < rmajor)
        return false;
    else if(major > rmajor)
        return true;

    // major == rmajor
    if(minor < rminor)
        return false;
    else if(minor > rminor)
        return true;

    // major == rmajor && minor == rminor
    if(patch < rpatch) return false;

    return true;
}

testResult_t
CheckDelta(void* results, void* expected, size_t count, size_t offset,
           ncclDataType_t type, ncclRedOp_t op, uint64_t seed, int nranks,
           int64_t* wrongEltN)
{
    ncclVerifiableVerify(results, expected, count, (int) type, (int) op, nranks, seed,
                         offset, wrongEltN, cudaStreamDefault);
    CUDACHECK(cudaDeviceSynchronize());
    return testSuccess;
}

testResult_t
InitDataReduce(void* data, const size_t count, const size_t offset, ncclDataType_t type,
               ncclRedOp_t op, uint64_t seed, int nranks)
{
    ncclVerifiablePrepareExpected(data, count, (int) type, (int) op, nranks, seed, offset,
                                  cudaStreamDefault);
    return testSuccess;
}

testResult_t
InitData(void* data, const size_t count, size_t offset, ncclDataType_t type,
         ncclRedOp_t op, uint64_t seed, int nranks, int rank)
{
    ncclVerifiablePrepareInput(data, count, (int) type, (int) op, nranks, rank, seed,
                               offset, cudaStreamDefault);
    return testSuccess;
}

void
Barrier(struct threadArgs* args)
{
    thread_local int       epoch      = 0;
    static pthread_mutex_t lock[2]    = { PTHREAD_MUTEX_INITIALIZER,
                                          PTHREAD_MUTEX_INITIALIZER };
    static pthread_cond_t  cond[2]    = { PTHREAD_COND_INITIALIZER,
                                          PTHREAD_COND_INITIALIZER };
    static int             counter[2] = { 0, 0 };

    pthread_mutex_lock(&lock[epoch]);
    if(++counter[epoch] == args->nThreads) pthread_cond_broadcast(&cond[epoch]);

    if(args->thread + 1 == args->nThreads)
    {
        while(counter[epoch] != args->nThreads)
            pthread_cond_wait(&cond[epoch], &lock[epoch]);
#ifdef MPI_SUPPORT
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        counter[epoch] = 0;
        pthread_cond_broadcast(&cond[epoch]);
    }
    else
    {
        while(counter[epoch] != 0)
            pthread_cond_wait(&cond[epoch], &lock[epoch]);
    }
    pthread_mutex_unlock(&lock[epoch]);
    epoch ^= 1;
}

// Inter-thread/process barrier+allreduce. The quality of the return value
// for average=0 (which means broadcast from rank=0) is dubious. The returned
// value will actually be the result of process-local broadcast from the local thread=0.
template <typename T>
void
Allreduce(struct threadArgs* args, T* value, int average)
{
    thread_local int       epoch   = 0;
    static pthread_mutex_t lock[2] = { PTHREAD_MUTEX_INITIALIZER,
                                       PTHREAD_MUTEX_INITIALIZER };
    static pthread_cond_t  cond[2] = { PTHREAD_COND_INITIALIZER,
                                       PTHREAD_COND_INITIALIZER };
    static T               accumulator[2];
    static int             counter[2] = { 0, 0 };

    pthread_mutex_lock(&lock[epoch]);
    if(counter[epoch] == 0)
    {
        if(average != 0 || args->thread == 0) accumulator[epoch] = *value;
    }
    else
    {
        switch(average)
        {
            case /*r0*/ 0:
                if(args->thread == 0) accumulator[epoch] = *value;
                break;
            case /*avg*/ 1: accumulator[epoch] += *value; break;
            case /*min*/ 2:
                accumulator[epoch] = std::min<T>(accumulator[epoch], *value);
                break;
            case /*max*/ 3:
                accumulator[epoch] = std::max<T>(accumulator[epoch], *value);
                break;
            case /*sum*/ 4: accumulator[epoch] += *value; break;
        }
    }

    if(++counter[epoch] == args->nThreads) pthread_cond_broadcast(&cond[epoch]);

    if(args->thread + 1 == args->nThreads)
    {
        while(counter[epoch] != args->nThreads)
            pthread_cond_wait(&cond[epoch], &lock[epoch]);

#ifdef MPI_SUPPORT
        if(average != 0)
        {
            static_assert(std::is_same<T, long long>::value ||
                              std::is_same<T, double>::value,
                          "Allreduce<T> only for T in {long long, double}");
            MPI_Datatype ty = std::is_same<T, long long>::value ? MPI_LONG_LONG
                              : std::is_same<T, double>::value  ? MPI_DOUBLE
                                                                : MPI_Datatype();
            MPI_Op       op = average == 1   ? MPI_SUM
                              : average == 2 ? MPI_MIN
                              : average == 3 ? MPI_MAX
                              : average == 4 ? MPI_SUM
                                             : MPI_Op();
            MPI_Allreduce(MPI_IN_PLACE, (void*) &accumulator[epoch], 1, ty, op,
                          MPI_COMM_WORLD);
        }
#endif

        if(average == 1) accumulator[epoch] /= args->totalProcs * args->nThreads;
        counter[epoch] = 0;
        pthread_cond_broadcast(&cond[epoch]);
    }
    else
    {
        while(counter[epoch] != 0)
            pthread_cond_wait(&cond[epoch], &lock[epoch]);
    }
    pthread_mutex_unlock(&lock[epoch]);

    *value = accumulator[epoch];
    epoch ^= 1;
}

testResult_t
CheckData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root,
          int in_place, int64_t* wrongElts)
{
    int    nranks = args->nProcs * args->nGpus * args->nThreads;
    size_t count  = args->expectedBytes / wordSize(type);

    int64_t* wrongPerGpu = nullptr;
    CUDACHECK(hipHostMalloc((void**) &wrongPerGpu, args->nGpus * sizeof(int64_t),
                            cudaHostAllocMapped));

    for(int i = 0; i < args->nGpus; i++)
    {
        int rank = ((args->proc * args->nThreads + args->thread) * args->nGpus + i);
        CUDACHECK(cudaSetDevice(args->gpus[i]));
        void* data = in_place ? ((void*) ((uintptr_t) args->recvbuffs[i] +
                                          args->recvInplaceOffset * rank))
                              : args->recvbuffs[i];

        TESTCHECK(CheckDelta(data, args->expected[i], count, 0, type, op, 0, nranks,
                             wrongPerGpu + i));

#if 1 && DEBUG_PRINT
        if(args->reportErrors && wrongPerGpu[i] != 0)
        {
            printf("rank=%d #wrong=%d\n", rank, (int) wrongPerGpu[i]);
            char* expectedHost = (char*) malloc(args->expectedBytes);
            char* dataHost     = (char*) malloc(args->expectedBytes);
            int   eltsz        = wordSize(type);
            cudaMemcpy(expectedHost, args->expected[i], args->expectedBytes,
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(dataHost, data, args->expectedBytes, cudaMemcpyDeviceToHost);

            for(int j = 0; j < args->expectedBytes / eltsz; j++)
            {
                unsigned long long want, got;
                want = 0;
                memcpy(&want, expectedHost + j * eltsz, eltsz);
                got = 0;
                memcpy(&got, dataHost + j * eltsz, eltsz);
                if(want != got)
                {
                    printf(" rank=%d elt[%d]: want=0x%llx got=0x%llx\n", rank, j, want,
                           got);
                }
            }
            free(expectedHost);
            free(dataHost);
        }
#endif
    }

    *wrongElts = 0;
    for(int i = 0; i < args->nGpus; i++)
        *wrongElts += wrongPerGpu[i];
    cudaFreeHost(wrongPerGpu);

    if(args->reportErrors && *wrongElts) args->errors[0]++;
    return testSuccess;
}

testResult_t
testStreamSynchronize(int ngpus, cudaStream_t* streams, ncclComm_t* comms)
{
    cudaError_t cudaErr;
    int         remaining = ngpus;
    int*        done      = (int*) malloc(sizeof(int) * ngpus);
    memset(done, 0, sizeof(int) * ngpus);
    timer tim;

    while(remaining)
    {
        int idle = 1;
        for(int i = 0; i < ngpus; i++)
        {
            if(done[i]) continue;

            cudaErr = cudaStreamQuery(streams[i]);
            if(cudaErr == cudaSuccess)
            {
                done[i] = 1;
                remaining--;
                idle = 0;
                continue;
            }

            if(cudaErr != cudaErrorNotReady) CUDACHECK(cudaErr);

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 4, 0)
            if(test_ncclVersion >= NCCL_VERSION(2, 4, 0) && comms)
            {
                ncclResult_t ncclAsyncErr;
                NCCLCHECK(ncclCommGetAsyncError(comms[i], &ncclAsyncErr));
                if(ncclAsyncErr != ncclSuccess)
                {
                    // An asynchronous error happened. Stop the operation and destroy
                    // the communicator
                    for(int i = 0; i < ngpus; i++)
                        NCCLCHECK(ncclCommAbort(comms[i]));
                    // Abort the perf test
                    NCCLCHECK(ncclAsyncErr);
                }
            }
            double delta = tim.elapsed();
            if(delta > timeout && timeout > 0)
            {
                for(int i = 0; i < ngpus; i++)
                    NCCLCHECK(ncclCommAbort(comms[i]));
                char hostname[1024];
                getHostName(hostname, 1024);
                printf("%s: Test timeout (%ds) %s:%d\n", hostname, timeout, __FILE__,
                       __LINE__);
                free(done);
                return testTimeout;
            }
#endif
        }

        // We might want to let other threads (including NCCL threads) use the CPU.
        if(idle) sched_yield();
    }
    free(done);
    return testSuccess;
}

testResult_t
startColl(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t opIndex, int root,
          int in_place, int iter)
{
    size_t count = args->nbytes / wordSize(type);

    // Try to change offset for each iteration so that we avoid cache effects and catch
    // race conditions in ptrExchange
    size_t shift = 0;
    if(enable_rotating_tensor)
    {
        shift = cache_bytes * (iter % 2);
    }
    else
    {
        size_t totalnbytes = std::max(args->sendBytes, args->expectedBytes);
        size_t steps       = totalnbytes ? args->maxbytes / totalnbytes : 1;
        shift              = totalnbytes * (iter % steps);
    }

    if(args->nGpus > 1) NCCLCHECK(ncclGroupStart());
    for(int i = 0; i < args->nGpus; i++)
    {
#ifndef NCCL_MAJOR
        CUDACHECK(cudaSetDevice(args->gpus[i]));
#endif
        int   rank     = ((args->proc * args->nThreads + args->thread) * args->nGpus + i);
        char* recvBuff = ((char*) args->recvbuffs[i]) + shift;
        char* sendBuff = ((char*) args->sendbuffs[i]) + shift;
        ncclRedOp_t op;

        if(opIndex < ncclNumOps)
        {
            op = opIndex;
        }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0)
        else
        {
            union
            {
                int8_t   i8;
                uint8_t  u8;
                int32_t  i32;
                uint32_t u32;
                int64_t  i64;
                uint64_t u64;
                half     f16;
                float    f32;
                double   f64;
#    if defined(RCCL_BFLOAT16)
                hip_bfloat16 bf16;
#    endif
#    if defined(RCCL_FLOAT8)
                rccl_float8  fp8_e4m3;
                rccl_bfloat8 fp8_e5m2;
#    endif
            };
            switch(type)
            {
                case ncclInt8: i8 = ncclVerifiablePremulScalar<int8_t>(rank); break;
                case ncclUint8: u8 = ncclVerifiablePremulScalar<uint8_t>(rank); break;
                case ncclInt32: i32 = ncclVerifiablePremulScalar<int32_t>(rank); break;
                case ncclUint32: u32 = ncclVerifiablePremulScalar<uint32_t>(rank); break;
                case ncclInt64: i64 = ncclVerifiablePremulScalar<int64_t>(rank); break;
                case ncclUint64: u64 = ncclVerifiablePremulScalar<uint64_t>(rank); break;
                case ncclFloat16: f16 = ncclVerifiablePremulScalar<half>(rank); break;
                case ncclFloat32: f32 = ncclVerifiablePremulScalar<float>(rank); break;
                case ncclFloat64: f64 = ncclVerifiablePremulScalar<double>(rank); break;
#    if defined(RCCL_BFLOAT16)
                case ncclBfloat16:
                    bf16 = ncclVerifiablePremulScalar<hip_bfloat16>(rank);
                    break;
#    endif
#    if defined(RCCL_FLOAT8)
                case ncclFp8E4M3:
                    fp8_e4m3 = ncclVerifiablePremulScalar<rccl_float8>(rank);
                    break;
                case ncclFp8E5M2:
                    fp8_e5m2 = ncclVerifiablePremulScalar<rccl_bfloat8>(rank);
                    break;
#    endif
                case ncclNumTypes: break;
            }
            NCCLCHECK(ncclRedOpCreatePreMulSum(&op, &u64, type, ncclScalarHostImmediate,
                                               args->comms[i]));
        }
#endif

        if(enable_cache_flush > 0 && ((iter % enable_cache_flush) == 0))
        {
            hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0,
                               args->streams[i]);
        }

        TESTCHECK(args->collTest->runColl(
            (void*) (in_place ? recvBuff + args->sendInplaceOffset * rank : sendBuff),
            (void*) (in_place ? recvBuff + args->recvInplaceOffset * rank : recvBuff),
            count, type, op, root, args->comms[i], args->streams[i]));

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0)
        if(opIndex >= ncclNumOps)
        {
            NCCLCHECK(ncclRedOpDestroy(op, args->comms[i]));
        }
#endif
    }
    if(args->nGpus > 1) NCCLCHECK(ncclGroupEnd());

    if(blocking_coll)
    {
        // Complete op before returning
        TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms));
    }
    if(blocking_coll) Barrier(args);
    return testSuccess;
}

testResult_t
completeColl(struct threadArgs* args)
{
    if(blocking_coll) return testSuccess;

    TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms));
    return testSuccess;
}

testResult_t
BenchTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root,
          int in_place)
{
    size_t count = args->nbytes / wordSize(type);
    if(datacheck)
    {
        // Initialize sendbuffs, recvbuffs and expected
        TESTCHECK(args->collTest->initData(args, type, op, root, 99, in_place));
    }

    if(warmup_iters)
    {
        // Sync
        TESTCHECK(startColl(args, type, op, root, in_place, 0));
        TESTCHECK(completeColl(args));
    }

    Barrier(args);

#if HIP_VERSION >= 50221310
    std::vector<cudaGraph_t>     graphs(args->nGpus);
    std::vector<cudaGraphExec_t> graphExec(args->nGpus);
    if(cudaGraphLaunches >= 1)
    {
        // Begin cuda graph capture
        for(int i = 0; i < args->nGpus; i++)
        {
            // Thread local mdoe is needed for:
            // - Multi-thread mode: where graph capture and instantiation can happen
            // concurrently across threads
            // - P2P pre-connect: when there is no warm-up, P2P pre-connect is done during
            // graph capture.
            //   Since pre-connect calls cudaMalloc, we cannot use global capture mode
            CUDACHECK(cudaStreamBeginCapture(args->streams[i],
                                             cudaStreamCaptureModeThreadLocal));
        }
    }
#endif

    // Performance Benchmark
    timer tim;
    for(int iter = 0; iter < iters; iter++)
    {
        if(agg_iters > 1) NCCLCHECK(ncclGroupStart());
        for(int aiter = 0; aiter < agg_iters; aiter++)
        {
            TESTCHECK(
                startColl(args, type, op, root, in_place, iter * agg_iters + aiter));
        }
        if(agg_iters > 1) NCCLCHECK(ncclGroupEnd());
    }

#if HIP_VERSION >= 50221310
    if(cudaGraphLaunches >= 1)
    {
        // End cuda graph capture
        for(int i = 0; i < args->nGpus; i++)
        {
            CUDACHECK(cudaStreamEndCapture(args->streams[i], graphs.data() + i));
        }
        // Instantiate cuda graph
        for(int i = 0; i < args->nGpus; i++)
        {
            CUDACHECK(
                cudaGraphInstantiate(graphExec.data() + i, graphs[i], NULL, NULL, 0));
        }
        // Resync CPU, restart timing, launch cuda graph
        Barrier(args);
        tim.reset();
        for(int l = 0; l < cudaGraphLaunches; l++)
        {
            for(int i = 0; i < args->nGpus; i++)
            {
                CUDACHECK(cudaGraphLaunch(graphExec[i], args->streams[i]));
            }
        }
    }
#endif

    double cputimeSec = tim.elapsed() / (iters * agg_iters);
    TESTCHECK(completeColl(args));

    double deltaSec = tim.elapsed();
    deltaSec        = deltaSec / (iters * agg_iters);
    if(cudaGraphLaunches >= 1) deltaSec = deltaSec / cudaGraphLaunches;
    Allreduce(args, &deltaSec, average);

#if HIP_VERSION >= 50221310
    if(cudaGraphLaunches >= 1)
    {
        // destroy cuda graph
        for(int i = 0; i < args->nGpus; i++)
        {
            CUDACHECK(cudaGraphExecDestroy(graphExec[i]));
            CUDACHECK(cudaGraphDestroy(graphs[i]));
        }
    }
#endif

    double algBw, busBw;
    args->collTest->getBw(count, wordSize(type), deltaSec, &algBw, &busBw,
                          args->nProcs * args->nThreads * args->nGpus);

    Barrier(args);

    int64_t             wrongElts = 0;
    static __thread int rep       = 0;
    rep++;
    for(int c = 0; c < datacheck; c++)
    {
        // Initialize sendbuffs, recvbuffs and expected
        TESTCHECK(args->collTest->initData(args, type, op, root, rep, in_place));

#if HIP_VERSION >= 50221310
        if(cudaGraphLaunches >= 1)
        {
            // Begin cuda graph capture for data check
            for(int i = 0; i < args->nGpus; i++)
            {
                CUDACHECK(cudaStreamBeginCapture(args->streams[i],
                                                 args->nThreads > 1
                                                     ? cudaStreamCaptureModeThreadLocal
                                                     : cudaStreamCaptureModeGlobal));
            }
        }
#endif

        // test validation in single itertion, should ideally be included into the
        // multi-iteration run
        TESTCHECK(startColl(args, type, op, root, in_place, 0));

#if HIP_VERSION >= 50221310
        if(cudaGraphLaunches >= 1)
        {
            // End cuda graph capture
            for(int i = 0; i < args->nGpus; i++)
            {
                CUDACHECK(cudaStreamEndCapture(args->streams[i], graphs.data() + i));
            }
            // Instantiate cuda graph
            for(int i = 0; i < args->nGpus; i++)
            {
                CUDACHECK(
                    cudaGraphInstantiate(graphExec.data() + i, graphs[i], NULL, NULL, 0));
            }
            // Launch cuda graph
            for(int i = 0; i < args->nGpus; i++)
            {
                CUDACHECK(cudaGraphLaunch(graphExec[i], args->streams[i]));
            }
        }
#endif

        TESTCHECK(completeColl(args));

#if HIP_VERSION >= 50221310
        if(cudaGraphLaunches >= 1)
        {
            // destroy cuda graph
            for(int i = 0; i < args->nGpus; i++)
            {
                CUDACHECK(cudaGraphExecDestroy(graphExec[i]));
                CUDACHECK(cudaGraphDestroy(graphs[i]));
            }
        }
#endif

        TESTCHECK(CheckData(args, type, op, root, in_place, &wrongElts));

        // aggregate delta from all threads and procs
        long long wrongElts1 = wrongElts;
        // if (wrongElts) fprintf(stderr, "\nERROR: Data corruption : rank %d size %ld
        // wrongElts %ld\n", args->proc, args->expectedBytes, wrongElts);
        Allreduce(args, &wrongElts1, /*sum*/ 4);
        wrongElts = wrongElts1;
        if(wrongElts) break;
    }

    double timeUsec = (report_cputime ? cputimeSec : deltaSec) * 1.0E6;
    char   timeStr[100];
    if(timeUsec >= 10000.0)
    {
        sprintf(timeStr, "%7.0f", timeUsec);
    }
    else if(timeUsec >= 100.0)
    {
        sprintf(timeStr, "%7.1f", timeUsec);
    }
    else
    {
        sprintf(timeStr, "%7.2f", timeUsec);
    }
    if(args->reportErrors)
    {
        PRINT("  %7s  %6.2f  %6.2f  %5g", timeStr, algBw, busBw, (double) wrongElts);
    }
    else
    {
        PRINT("  %7s  %6.2f  %6.2f  %5s", timeStr, algBw, busBw, "N/A");
    }

    auto largestMessageSize = std::max(args->sendBytes, args->expectedBytes);
    if(args->reporter)
    {
        if(args->reportErrors)
        {
            args->reporter->addResult((args->nThreads * args->nGpus), args->nProcs,
                                      args->totalProcs, largestMessageSize, in_place,
                                      timeUsec, algBw, busBw, wrongElts);
        }
        else
        {
            args->reporter->addResult((args->nThreads * args->nGpus), args->nProcs,
                                      args->totalProcs, largestMessageSize, in_place,
                                      timeUsec, algBw, busBw);
        }
    }

    args->bw[0] += busBw;
    args->bw_count[0]++;
    return testSuccess;
}

void
setupArgs(size_t size, ncclDataType_t type, struct threadArgs* args)
{
    int    nranks = args->nProcs * args->nGpus * args->nThreads;
    size_t count, sendCount, recvCount, paramCount, sendInplaceOffset, recvInplaceOffset;

    count = size / wordSize(type);
    args->collTest->getCollByteCount(&sendCount, &recvCount, &paramCount,
                                     &sendInplaceOffset, &recvInplaceOffset,
                                     (size_t) count, wordSize(type), (size_t) nranks);

    args->nbytes            = paramCount * wordSize(type);
    args->sendBytes         = sendCount * wordSize(type);
    args->expectedBytes     = recvCount * wordSize(type);
    args->sendInplaceOffset = sendInplaceOffset * wordSize(type);
    args->recvInplaceOffset = recvInplaceOffset * wordSize(type);
}

testResult_t
TimeTest(struct threadArgs* args, ncclDataType_t type, const char* typeName,
         ncclRedOp_t op, const char* opName, int root)
{
    // Sync to avoid first-call timeout
    Barrier(args);

    // Warm-up for large size
    setupArgs(args->maxbytes, type, args);
#if HIP_VERSION >= 50221310
    std::vector<cudaGraph_t>     graphs(args->nGpus);
    std::vector<cudaGraphExec_t> graphExec(args->nGpus);
    if(cudaGraphLaunches >= 1)
    {
        // Begin cuda graph capture
        for(int i = 0; i < args->nGpus; i++)
        {
            // Thread local mode is needed for:
            // - Multi-thread mode: where graph capture and instantiation can happen
            // concurrently across threads
            // - P2P pre-connect: when there is no warm-up, P2P pre-connect is done during
            // graph capture.
            //   Since pre-connect calls cudaMalloc, we cannot use global capture mode
            CUDACHECK(cudaStreamBeginCapture(args->streams[i],
                                             cudaStreamCaptureModeThreadLocal));
        }
    }
#endif
    for(int iter = 0; iter < warmup_iters; iter++)
    {
        TESTCHECK(startColl(args, type, op, root, 0, iter));
    }

#if HIP_VERSION >= 50221310
    if(cudaGraphLaunches >= 1)
    {
        // End cuda graph capture
        for(int i = 0; i < args->nGpus; i++)
        {
            CUDACHECK(cudaStreamEndCapture(args->streams[i], graphs.data() + i));
        }
        // Instantiate cuda graph
        for(int i = 0; i < args->nGpus; i++)
        {
            CUDACHECK(
                cudaGraphInstantiate(graphExec.data() + i, graphs[i], NULL, NULL, 0));
        }
        // Resync CPU, restart timing, launch cuda graph
        Barrier(args);
        for(int l = 0; l < cudaGraphLaunches; l++)
        {
            for(int i = 0; i < args->nGpus; i++)
            {
                CUDACHECK(cudaGraphLaunch(graphExec[i], args->streams[i]));
            }
        }
    }
#endif

    TESTCHECK(completeColl(args));

#if HIP_VERSION >= 50221310
    if(cudaGraphLaunches >= 1)
    {
        // destroy cuda graph
        for(int i = 0; i < args->nGpus; i++)
        {
            CUDACHECK(cudaGraphExecDestroy(graphExec[i]));
            CUDACHECK(cudaGraphDestroy(graphs[i]));
        }
    }
#endif

    // Warm-up for small size
    setupArgs(args->minbytes, type, args);
#if HIP_VERSION >= 50221310
    if(cudaGraphLaunches >= 1)
    {
        // Begin cuda graph capture
        for(int i = 0; i < args->nGpus; i++)
        {
            // Thread local mode is needed for:
            // - Multi-thread mode: where graph capture and instantiation can happen
            // concurrently across threads
            // - P2P pre-connect: when there is no warm-up, P2P pre-connect is done during
            // graph capture.
            //   Since pre-connect calls cudaMalloc, we cannot use global capture mode
            CUDACHECK(cudaStreamBeginCapture(args->streams[i],
                                             cudaStreamCaptureModeThreadLocal));
        }
    }
#endif
    for(int iter = 0; iter < warmup_iters; iter++)
    {
        TESTCHECK(startColl(args, type, op, root, iter < warmup_iters / 2 ? 0 : 1, iter));
    }

#if HIP_VERSION >= 50221310
    if(cudaGraphLaunches >= 1)
    {
        // End cuda graph capture
        for(int i = 0; i < args->nGpus; i++)
        {
            CUDACHECK(cudaStreamEndCapture(args->streams[i], graphs.data() + i));
        }
        // Instantiate cuda graph
        for(int i = 0; i < args->nGpus; i++)
        {
            CUDACHECK(
                cudaGraphInstantiate(graphExec.data() + i, graphs[i], NULL, NULL, 0));
        }
        // Resync CPU, restart timing, launch cuda graph
        Barrier(args);
        for(int l = 0; l < cudaGraphLaunches; l++)
        {
            for(int i = 0; i < args->nGpus; i++)
            {
                CUDACHECK(cudaGraphLaunch(graphExec[i], args->streams[i]));
            }
        }
    }
#endif

    TESTCHECK(completeColl(args));

#if HIP_VERSION >= 50221310
    if(cudaGraphLaunches >= 1)
    {
        // destroy cuda graph
        for(int i = 0; i < args->nGpus; i++)
        {
            CUDACHECK(cudaGraphExecDestroy(graphExec[i]));
            CUDACHECK(cudaGraphDestroy(graphs[i]));
        }
    }
#endif

    // Benchmark
    long   repeat = run_cycles;
    size_t iter   = 0;

    do
    {
        if(run_cycles > 1) PRINT("# Testing %lu cycle.\n", iter + 1);
        if(args->reporter)
        {
            args->reporter->setParameters(iter, args->collTest->name, typeName, opName);
        }
        for(size_t size = args->minbytes; size <= args->maxbytes;
            size        = ((args->stepfactor > 1) ? size * args->stepfactor
                                                  : size + args->stepbytes))
        {
            setupArgs(size, type, args);
            char rootName[100];
            sprintf(rootName, "%6i", root);
            PRINT("%12li  %12li  %8s  %6s  %6s",
                  std::max(args->sendBytes, args->expectedBytes),
                  args->nbytes / wordSize(type), typeName, opName, rootName);
            if(enable_out_of_place)
            {
                TESTCHECK(BenchTime(args, type, op, root, 0));
                usleep(delay_inout_place);
            }
            if(enable_in_place) TESTCHECK(BenchTime(args, type, op, root, 1));
            PRINT("\n");
        }
        --repeat;
        ++iter;
    } while(repeat != 0);

    return testSuccess;
}

testResult_t
threadRunTests(struct threadArgs* args)
{
    // Set device to the first of our GPUs. If we don't do that, some operations
    // will be done on the current GPU (by default : 0) and if the GPUs are in
    // exclusive mode those operations will fail.
    CUDACHECK(cudaSetDevice(args->gpus[0]));
    TESTCHECK(ncclTestEngine.runTest(args, ncclroot, (ncclDataType_t) nccltype,
                                     test_typenames[nccltype], (ncclRedOp_t) ncclop,
                                     test_opnames[ncclop]));
    return testSuccess;
}

testResult_t
threadInit(struct threadArgs* args)
{
    char hostname[1024];
    getHostName(hostname, 1024);
    int nranks = args->nProcs * args->nThreads * args->nGpus;

    // set main thread again
    is_main_thread = (is_main_proc && args->thread == 0) ? 1 : 0;

    NCCLCHECK(ncclGroupStart());
    for(int i = 0; i < args->nGpus; i++)
    {
        int rank =
            args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus + i;
        CUDACHECK(cudaSetDevice(args->gpus[i]));
        NCCLCHECK(ncclCommInitRank(args->comms + i, nranks, args->ncclId, rank));
    }
    NCCLCHECK(ncclGroupEnd());
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
    void** sendRegHandles =
        (local_register) ? (void**) malloc(sizeof(*sendRegHandles) * args->nGpus) : NULL;
    void** recvRegHandles =
        (local_register) ? (void**) malloc(sizeof(*recvRegHandles) * args->nGpus) : NULL;
    for(int i = 0; i < args->nGpus; i++)
    {
        if(local_register)
            NCCLCHECK(ncclCommRegister(args->comms[i], args->sendbuffs[i], args->maxbytes,
                                       &sendRegHandles[i]));
        if(local_register)
            NCCLCHECK(ncclCommRegister(args->comms[i], args->recvbuffs[i], args->maxbytes,
                                       &recvRegHandles[i]));
    }
#endif

    TESTCHECK(threadRunTests(args));

    for(int i = 0; i < args->nGpus; i++)
    {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
        if(local_register)
            NCCLCHECK(ncclCommDeregister(args->comms[i], sendRegHandles[i]));
        if(local_register)
            NCCLCHECK(ncclCommDeregister(args->comms[i], recvRegHandles[i]));
#endif
        NCCLCHECK(ncclCommDestroy(args->comms[i]));
    }
    return testSuccess;
}

void*
threadLauncher(void* thread_)
{
    struct testThread* thread = (struct testThread*) thread_;
    thread->ret               = thread->func(&thread->args);
    return NULL;
}
testResult_t
threadLaunch(struct testThread* thread)
{
    pthread_create(&thread->thread, NULL, threadLauncher, thread);
    return testSuccess;
}

testResult_t
AllocateBuffs(void** sendbuff, size_t sendBytes, void** recvbuff, size_t recvBytes,
              void** expected, size_t nbytes)
{
    if(enable_rotating_tensor)
    {
        recvBytes = recvBytes + cache_bytes;
        nbytes    = nbytes + cache_bytes;
    }
    if(memorytype == ncclFine)
    {
        if(HIP_VERSION >= 50700000)
        {
            CUDACHECK(hipExtMallocWithFlags(sendbuff, nbytes, hipDeviceMallocUncached));
            CUDACHECK(hipExtMallocWithFlags(recvbuff, nbytes, hipDeviceMallocUncached));
            if(datacheck)
                CUDACHECK(
                    hipExtMallocWithFlags(expected, recvBytes, hipDeviceMallocUncached));
        }
        else
        {
            CUDACHECK(
                hipExtMallocWithFlags(sendbuff, nbytes, hipDeviceMallocFinegrained));
            CUDACHECK(
                hipExtMallocWithFlags(recvbuff, nbytes, hipDeviceMallocFinegrained));
            if(datacheck)
                CUDACHECK(hipExtMallocWithFlags(expected, recvBytes,
                                                hipDeviceMallocFinegrained));
        }
    }
    else if(memorytype == ncclHost)
    {
        CUDACHECK(hipHostMalloc(sendbuff, nbytes));
        CUDACHECK(hipHostMalloc(recvbuff, nbytes));
        if(datacheck) CUDACHECK(hipHostMalloc(expected, recvBytes));
    }
    else if(memorytype == ncclManaged)
    {
        CUDACHECK(cudaMallocManaged(sendbuff, nbytes));
        CUDACHECK(cudaMallocManaged(recvbuff, nbytes));
        if(datacheck) CUDACHECK(cudaMallocManaged(expected, recvBytes));
#if 0
    CUDACHECK(cudaMemset(*sendbuff, 0, nbytes));
    CUDACHECK(cudaMemset(*recvbuff, 0, nbytes));
    if (datacheck) CUDACHECK(cudaMemset(*expected, 0, recvBytes));
#endif
    }
    else
    {
        CUDACHECK(cudaMalloc(sendbuff, nbytes));
        CUDACHECK(cudaMalloc(recvbuff, nbytes));
        if(datacheck) CUDACHECK(cudaMalloc(expected, recvBytes));
    }
    CUDACHECK(hipMemset(*sendbuff, 1, nbytes));
    if(datacheck) CUDACHECK(hipMemset(*expected, 1, recvBytes));
    return testSuccess;
}

testResult_t
run();  // Main function

int
main(int argc, char* argv[])
{
    // Make sure everyline is flushed so that we see the progress of the test
    setlinebuf(stdout);

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 4, 0)
    ncclGetVersion(&test_ncclVersion);
#else
    test_ncclVersion = NCCL_VERSION_CODE;
#endif

// printf("# NCCL_VERSION_CODE=%d ncclGetVersion=%d\n", NCCL_VERSION_CODE,
// test_ncclVersion);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 0, 0)
    test_opnum   = 4;
    test_typenum = 9;
    if(NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0) &&
       test_ncclVersion >= NCCL_VERSION(2, 10, 0))
    {
        test_opnum++;  // ncclAvg
#    if defined(RCCL_BFLOAT16)
        test_typenum++;  // bfloat16
#    endif
#    if defined(RCCL_FLOAT8)
        test_typenum++;  // fp8_e4m3
        test_typenum++;  // fp8_e5m2
#    endif
    }
    if(NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0) &&
       test_ncclVersion >= NCCL_VERSION(2, 11, 0))
    {
        test_opnum++;  // PreMulSum
    }
#endif

    // Parse args
    // Replace getopt_long with manual argument parsing
    double parsed;
    for(int argi = 1; argi < argc; ++argi)
    {
        const char* arg = argv[argi];
        if(strcmp(arg, "-t") == 0 || strcmp(arg, "--nthreads") == 0)
        {
            nThreads = strtol(argv[++argi], NULL, 0);
        }
        else if(strcmp(arg, "-g") == 0 || strcmp(arg, "--ngpus") == 0)
        {
            nGpus = strtol(argv[++argi], NULL, 0);
        }
        else if(strcmp(arg, "-b") == 0 || strcmp(arg, "--minbytes") == 0)
        {
            parsed = parsesize(argv[++argi]);
            if(parsed < 0)
            {
                fprintf(stderr, "invalid size specified for 'minbytes'\n");
                return -1;
            }
            minBytes = (size_t) parsed;
        }
        else if(strcmp(arg, "-e") == 0 || strcmp(arg, "--maxbytes") == 0)
        {
            parsed = parsesize(argv[++argi]);
            if(parsed < 0)
            {
                fprintf(stderr, "invalid size specified for 'maxbytes'\n");
                return -1;
            }
            maxBytes = (size_t) parsed;
        }
        else if(strcmp(arg, "-i") == 0 || strcmp(arg, "--stepbytes") == 0)
        {
            parsed = parsesize(argv[++argi]);
            if(parsed < 0)
            {
                fprintf(stderr, "invalid size specified for 'stepBytes'\n");
                return -1;
            }
            stepBytes = (size_t) parsed;
        }
        else if(strcmp(arg, "-f") == 0 || strcmp(arg, "--stepfactor") == 0)
        {
            stepFactor = strtol(argv[++argi], NULL, 0);
        }
        else if(strcmp(arg, "-n") == 0 || strcmp(arg, "--iters") == 0)
        {
            iters = (int) strtol(argv[++argi], NULL, 0);
        }
        else if(strcmp(arg, "-m") == 0 || strcmp(arg, "--agg_iters") == 0)
        {
#if NCCL_MAJOR > 2 || (NCCL_MAJOR >= 2 && NCCL_MINOR >= 2)
            agg_iters = (int) strtol(argv[++argi], NULL, 0);
#else
            fprintf(stderr, "Option -m not supported before NCCL 2.2. Ignoring\n");
            ++argi;
#endif
        }
        else if(strcmp(arg, "-w") == 0 || strcmp(arg, "--warmup_iters") == 0)
        {
            warmup_iters = (int) strtol(argv[++argi], NULL, 0);
        }
        else if(strcmp(arg, "-N") == 0 || strcmp(arg, "--run_cycles") == 0)
        {
            run_cycles = (int) strtol(argv[++argi], NULL, 0);
        }
        else if(strcmp(arg, "-p") == 0 || strcmp(arg, "--parallel_init") == 0)
        {
            parallel_init = (int) strtol(argv[++argi], NULL, 0);
        }
        else if(strcmp(arg, "-c") == 0 || strcmp(arg, "--check") == 0)
        {
            datacheck = (int) strtol(argv[++argi], NULL, 0);
        }
        else if(strcmp(arg, "-o") == 0 || strcmp(arg, "--op") == 0)
        {
            ncclop = ncclstringtoop(argv[++argi]);
        }
        else if(strcmp(arg, "-d") == 0 || strcmp(arg, "--datatype") == 0)
        {
            nccltype = ncclstringtotype(argv[++argi]);
        }
        else if(strcmp(arg, "-r") == 0 || strcmp(arg, "--root") == 0)
        {
            ncclroot = ncclstringtoroot(argv[++argi]);
        }
        else if(strcmp(arg, "-z") == 0 || strcmp(arg, "--blocking") == 0)
        {
            blocking_coll = strtol(argv[++argi], NULL, 0);
        }
        else if(strcmp(arg, "-y") == 0 || strcmp(arg, "--stream_null") == 0)
        {
            streamnull = strtol(argv[++argi], NULL, 0);
        }
        else if(strcmp(arg, "-T") == 0 || strcmp(arg, "--timeout") == 0)
        {
            timeout = strtol(argv[++argi], NULL, 0);
        }
        else if(strcmp(arg, "-G") == 0 || strcmp(arg, "--cudagraph") == 0)
        {
#if(NCCL_MAJOR > 2 || (NCCL_MAJOR >= 2 && NCCL_MINOR >= 9)) && HIP_VERSION >= 50221310
            cudaGraphLaunches = strtol(argv[++argi], NULL, 0);
#else
            printf("Option -G (HIP graph) not supported before NCCL 2.9 + ROCm 5.2 "
                   "Ignoring\n");
            ++argi;
#endif
        }
        else if(strcmp(arg, "-C") == 0 || strcmp(arg, "--report_cputime") == 0)
        {
            report_cputime = strtol(argv[++argi], NULL, 0);
        }
        else if(strcmp(arg, "-a") == 0 || strcmp(arg, "--average") == 0)
        {
            average = (int) strtol(argv[++argi], NULL, 0);
        }
        else if(strcmp(arg, "-R") == 0 || strcmp(arg, "--local_register") == 0)
        {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
            if((int) strtol(argv[++argi], NULL, 0))
            {
                local_register = 1;
            }
#else
            printf("Option -R (register) is not supported before NCCL 2.19. Ignoring\n");
            ++argi;
#endif
        }
        else if(strcmp(arg, "-Y") == 0 || strcmp(arg, "--memory_type") == 0)
        {
            memorytype = ncclstringtomtype(argv[++argi]);
        }
        else if(strcmp(arg, "-u") == 0 || strcmp(arg, "--cumask") == 0)
        {
            int   nmasks  = 0;
            char* maskstr = argv[++argi];
            char* mask    = strtok(maskstr, ",");
            while(mask != NULL && nmasks < 4)
            {
                cumask[nmasks++] = strtol(mask, NULL, 16);
                mask             = strtok(NULL, ",");
            }
        }
        else if(strcmp(arg, "-O") == 0 || strcmp(arg, "--out_of_place") == 0)
        {
            enable_out_of_place = strtol(argv[++argi], NULL, 0);
            enable_in_place     = enable_out_of_place ? 0 : 1;
        }
        else if(strcmp(arg, "-q") == 0 || strcmp(arg, "--delay_inout_place") == 0)
        {
            delay_inout_place = (int) strtol(argv[++argi], NULL, 10);
        }
        else if(strcmp(arg, "-F") == 0 || strcmp(arg, "--cache_flush") == 0)
        {
            enable_cache_flush = strtol(argv[++argi], NULL, 0);
            if(enable_cache_flush > 0)
            {
                hipDeviceProp_t deviceProps;
                CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProps, 0));
                gpu_block3 = deviceProps.multiProcessorCount * 60;
            }
        }
        else if(strcmp(arg, "-E") == 0 || strcmp(arg, "--rotating_tensor") == 0)
        {
            enable_rotating_tensor = strtol(argv[++argi], NULL, 0);
        }
        else if(strcmp(arg, "-x") == 0 || strcmp(arg, "--output_file") == 0)
        {
            output_file = argv[++argi];
        }
        else if(strcmp(arg, "-Z") == 0 || strcmp(arg, "--output_format") == 0)
        {
            output_format = argv[++argi];
        }
        else if(strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0)
        {
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
                   "[-N,--run_cycles <cycle count> run & print each cycle (default: 1; "
                   "0=infinite)] \n\t"
                   "[-p,--parallel_init <0/1>] \n\t"
                   "[-c,--check <check iteration count>] \n\t"
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0)
                   "[-o,--op <sum/prod/min/max/avg/mulsum/all>] \n\t"
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
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
                   "[-a,--average <0/1/2/3> report average iteration time "
                   "<0=RANK0/1=AVG/2=MIN/3=MAX>] \n\t"
                   "[-R,--local_register <1/0> enable local buffer registration on "
                   "send/recv buffers (default: disable)] \n\t"
                   "[-Y,--memory_type <coarse/fine/host/managed>] \n\t"
                   "[-u,--cumask <d0,d1,d2,d3>] \n\t"
                   "[-O,--out_of_place <0/1>] \n\t"
                   "[-q,--delay <delay between out-of-place and in-place in "
                   "microseconds>] \n\t"
                   "[-F,--cache_flush <number of iterations between instruction cache "
                   "flush>] \n\t"
                   "[-E,--rotating_tensor <0/1>] \n\t"
                   "[-x,--output_file <output file name>] \n\t"
                   "[-Z,--output_format <output format <csv|json>] \n\t"
                   "[-h,--help]\n",
                   basename(argv[0]));
            return 0;
        }
    }

    CUDACHECK(cudaGetDeviceCount(&numDevices));
#ifndef MPI_SUPPORT
    if(nGpus > numDevices)
    {
        fprintf(stderr,
                "[ERROR] The number of requested GPUs (%d) is greater than the number of "
                "GPUs available (%d)\n",
                nGpus, numDevices);
        return testNcclError;
    }
#endif
    if(minBytes > maxBytes)
    {
        fprintf(stderr, "invalid sizes for 'minbytes' and 'maxbytes': %llu > %llu\n",
                (unsigned long long) minBytes, (unsigned long long) maxBytes);
        return -1;
    }
    if(!output_format.empty())
    {
        if(!(output_format == "csv" || output_format == "json"))
        {
            std::cerr << "Invalid --output_format: " << output_format << "\n";
            return -1;
        }
    }
#ifdef MPI_SUPPORT
    MPI_Init(&argc, &argv);
#endif

    TESTCHECK(run());
    return 0;
}

#ifdef MPI_SUPPORT
// parse int for base 2/10/16, will ignore first whitespaces
static bool
parseInt(char* s, int* num)
{
    char* p = NULL;
    if(!s || !num) return false;
    while(*s && isspace(*s))
        ++s;
    if(!*s) return false;

    if(strncasecmp(s, "0b", 2) == 0)
        *num = (int) strtoul(s + 2, &p, 2);
    else
        *num = (int) strtoul(s, &p, 0);

    if(p == s) return false;
    return true;
}
#endif

testResult_t
run()
{
    int  totalProcs = 1, proc = 0, ncclProcs = 1, ncclProc = 0, color = 0;
    int  localRank = 0;
    int  localSize = 0;
    char hostname[1024];
    getHostName(hostname, 1024);

#ifdef MPI_SUPPORT
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    std::vector<uint64_t> hostHashs(totalProcs);
    hostHashs[proc] = getHostHash(hostname);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs.data(), sizeof(uint64_t),
                  MPI_BYTE, MPI_COMM_WORLD);
    for(int p = 0; p < totalProcs; p++)
    {
        if(p == proc) break;
        if(hostHashs[p] == hostHashs[proc]) localRank++;
    }

    char* splitMaskEnv = NULL;
    if((splitMaskEnv = getenv("NCCL_TESTS_SPLIT_MASK")))
    {
        color = proc & strtoul(splitMaskEnv, NULL, 16);
    }
    else if((splitMaskEnv = getenv("NCCL_TESTS_SPLIT")))
    {
        if((strncasecmp(splitMaskEnv, "AND", strlen("AND")) == 0 &&
            parseInt(splitMaskEnv + strlen("AND"), &color)) ||
           (strncasecmp(splitMaskEnv, "&", strlen("&")) == 0 &&
            parseInt(splitMaskEnv + strlen("&"), &color)))
            color = proc & color;
        if((strncasecmp(splitMaskEnv, "OR", strlen("OR")) == 0 &&
            parseInt(splitMaskEnv + strlen("OR"), &color)) ||
           (strncasecmp(splitMaskEnv, "|", strlen("|")) == 0 &&
            parseInt(splitMaskEnv + strlen("|"), &color)))
            color = proc | color;
        if((strncasecmp(splitMaskEnv, "MOD", strlen("MOD")) == 0 &&
            parseInt(splitMaskEnv + strlen("MOD"), &color)) ||
           (strncasecmp(splitMaskEnv, "%", strlen("%")) == 0 &&
            parseInt(splitMaskEnv + strlen("%"), &color)))
            color = proc % color;
        if((strncasecmp(splitMaskEnv, "DIV", strlen("DIV")) == 0 &&
            parseInt(splitMaskEnv + strlen("DIV"), &color)) ||
           (strncasecmp(splitMaskEnv, "/", strlen("/")) == 0 &&
            parseInt(splitMaskEnv + strlen("/"), &color)))
            color = proc / color;
    }

    MPI_Comm mpi_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, proc, &mpi_comm);
    MPI_Comm_size(mpi_comm, &ncclProcs);
    MPI_Comm_rank(mpi_comm, &ncclProc);

    for(int p = 0; p < totalProcs; p++)
    {
        if(hostHashs[p] == hostHashs[proc]) localSize++;
    }
    if(nGpus * localSize > numDevices && numDevices != 1)
    {
        fprintf(stderr,
                "[ERROR] The number of requested GPUs (%d) is greater than the number of "
                "GPUs available (%d) on node (%s)\n",
                nGpus * localSize, numDevices, hostname);
        return testNcclError;
    }
#endif
    is_main_thread = is_main_proc = (proc == 0) ? 1 : 0;

    PRINT("# nThread %d nGpus %d minBytes %ld maxBytes %ld step: %ld(%s) warmup iters: "
          "%d iters: %d agg iters: %d validation: %d graph: %d\n",
          nThreads, nGpus, minBytes, maxBytes, (stepFactor > 1) ? stepFactor : stepBytes,
          (stepFactor > 1) ? "factor" : "bytes", warmup_iters, iters, agg_iters,
          datacheck, cudaGraphLaunches);
    if(blocking_coll)
        PRINT("# Blocking Enabled: wait for completion and barrier after each collective "
              "\n");
    if(parallel_init)
        PRINT("# Parallel Init Enabled: threads call into NcclInitRank concurrently \n");
    PRINT("#\n");
    PRINT("rccl-tests: Version %s\n", rcclTestsGitHash);
    PRINT("# Using devices\n");
#define MAX_LINE 2048
    char   line[MAX_LINE];
    int    len    = 0;
    size_t maxMem = ~0;
    char*  envstr = getenv("NCCL_TESTS_DEVICE");
    int    gpu0   = envstr ? atoi(envstr) : -1;
    for(int i = 0; i < nThreads * nGpus; i++)
    {
        int cudaDev =
            ((gpu0 != -1 ? gpu0 : localRank * nThreads * nGpus) + i) % numDevices;
        int            rank = proc * nThreads * nGpus + i;
        cudaDeviceProp prop;
        CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
        // char busIdStr[] = "00000000:00:00.0";
        // CUDACHECK(cudaDeviceGetPCIBusId(busIdStr, sizeof(busIdStr), cudaDev));
        // len += snprintf(line+len, MAX_LINE>len ? MAX_LINE-len : 0, "#  Rank %2d Group
        // %2d Pid %6d on %10s device %2d [%04x:%s:%02x] %s\n",
        //                 rank, color, getpid(), hostname, cudaDev, prop.pciDomainID,
        //                 busIdStr, prop.pciDeviceID, prop.name);
        len += snprintf(
            line + len, MAX_LINE > len ? MAX_LINE - len : 0,
            "#  Rank %2d Group %2d Pid %6d on %10s device %2d [%04x:%02x:%02x] %s\n",
            rank, color, getpid(), hostname, cudaDev, prop.pciDomainID, prop.pciBusID,
            prop.pciDeviceID, prop.name);
        maxMem = std::min(maxMem, prop.totalGlobalMem);
    }
#if MPI_SUPPORT
    char* lines = (proc == 0) ? (char*) malloc(totalProcs * MAX_LINE) : NULL;
    // Gather all output in rank order to root (0)
    MPI_Gather(line, MAX_LINE, MPI_BYTE, lines, MAX_LINE, MPI_BYTE, 0, MPI_COMM_WORLD);
    if(proc == 0)
    {
        for(int p = 0; p < totalProcs; p++)
            PRINT("%s", lines + MAX_LINE * p);
        free(lines);
    }
    MPI_Allreduce(MPI_IN_PLACE, &maxMem, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
#else
    PRINT("%s", line);
#endif

    // We need sendbuff, recvbuff, expected (when datacheck enabled), plus 1G for the
    // rest.
    size_t memMaxBytes = (maxMem - (1 << 30)) / (datacheck ? 3 : 2);
    if(maxBytes > memMaxBytes)
    {
        maxBytes = memMaxBytes;
        if(proc == 0)
            printf("#\n# Reducing maxBytes to %ld due to memory limitation\n", maxBytes);
    }

    ncclUniqueId ncclId;
    if(ncclProc == 0)
    {
        NCCLCHECK(ncclGetUniqueId(&ncclId));
    }
#ifdef MPI_SUPPORT
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, mpi_comm);
    MPI_Barrier(MPI_COMM_WORLD);  // Ensure Bcast is complete for HCOLL
#endif

    std::vector<int>          gpus(nGpus * nThreads);
    std::vector<cudaStream_t> streams(nGpus * nThreads);
    std::vector<void*>        sendbuffs(nGpus * nThreads);
    std::vector<void*>        recvbuffs(nGpus * nThreads);
    std::vector<void*>        expected(nGpus * nThreads);
    size_t                    sendBytes, recvBytes;

    ncclTestEngine.getBuffSize(&sendBytes, &recvBytes, (size_t) maxBytes,
                               (size_t) ncclProcs * nGpus * nThreads);

    envstr = getenv("NCCL_TESTS_DEVICE");
    gpu0   = envstr ? atoi(envstr) : -1;
    for(int i = 0; i < nGpus * nThreads; i++)
    {
        gpus[i] = ((gpu0 != -1 ? gpu0 : localRank * nThreads * nGpus) + i) % numDevices;
        CUDACHECK(cudaSetDevice(gpus[i]));
        TESTCHECK(AllocateBuffs(sendbuffs.data() + i, sendBytes, recvbuffs.data() + i,
                                recvBytes, expected.data() + i, (size_t) maxBytes));
        if(streamnull)
            streams[i] = NULL;
        else
            CUDACHECK(
                cudaStreamCreateWithFlags(streams.data() + i, cudaStreamNonBlocking));
    }

    // if parallel init is not selected, use main thread to initialize NCCL
    ncclComm_t* comms = (ncclComm_t*) malloc(sizeof(ncclComm_t) * nThreads * nGpus);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
    void** sendRegHandles = NULL;
    void** recvRegHandles = NULL;
#endif
    if(!parallel_init)
    {
        if(ncclProcs == 1)
        {
            NCCLCHECK(ncclCommInitAll(comms, nGpus * nThreads, gpus.data()));
        }
        else
        {
            NCCLCHECK(ncclGroupStart());
            for(int i = 0; i < nGpus * nThreads; i++)
            {
                CUDACHECK(cudaSetDevice(gpus[i]));
                NCCLCHECK(ncclCommInitRank(comms + i, ncclProcs * nThreads * nGpus,
                                           ncclId, ncclProc * nThreads * nGpus + i));
            }
            NCCLCHECK(ncclGroupEnd());
        }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
        sendRegHandles = (local_register)
                             ? (void**) malloc(sizeof(*sendRegHandles) * nThreads * nGpus)
                             : NULL;
        recvRegHandles = (local_register)
                             ? (void**) malloc(sizeof(*recvRegHandles) * nThreads * nGpus)
                             : NULL;
        for(int i = 0; i < nGpus * nThreads; i++)
        {
            if(local_register)
                NCCLCHECK(ncclCommRegister(comms[i], &sendbuffs[i], maxBytes,
                                           &sendRegHandles[i]));
            if(local_register)
                NCCLCHECK(ncclCommRegister(comms[i], &recvbuffs[i], maxBytes,
                                           &recvRegHandles[i]));
        }
#endif
    }

    std::vector<int>    errors(nThreads);
    std::vector<double> bw(nThreads);
    double*             delta;
    CUDACHECK(hipHostMalloc(&delta, sizeof(double) * nThreads * NUM_BLOCKS,
                            cudaHostAllocPortable | cudaHostAllocMapped));
    std::vector<int> bw_count(nThreads);
    for(int t = 0; t < nThreads; t++)
    {
        bw[t]     = 0.0;
        errors[t] = bw_count[t] = 0;
    }

    fflush(stdout);

    const char* timeStr = report_cputime ? "cputime" : "time";
    PRINT("#\n");
    if(enable_out_of_place && enable_in_place)
    {
        PRINT("# %10s  %12s  %8s  %6s  %6s           out-of-place                       "
              "in-place          \n",
              "", "", "", "", "");
        PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s %6s  %7s  %6s  %6s %6s\n",
              "size", "count", "type", "redop", "root", timeStr, "algbw", "busbw",
              "#wrong", timeStr, "algbw", "busbw", "#wrong");
        PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n",
              "(B)", "(elements)", "", "", "", "(us)", "(GB/s)", "(GB/s)", "", "(us)",
              "(GB/s)", "(GB/s)", "");
    }
    else if(enable_out_of_place)
    {
        PRINT("# %10s  %12s  %8s  %6s  %6s           out-of-place      \n", "", "", "",
              "", "");
        PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s %6s\n", "size", "count", "type",
              "redop", "root", timeStr, "algbw", "busbw", "#wrong");
        PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s  %5s\n", "(B)", "(elements)",
              "", "", "", "(us)", "(GB/s)", "(GB/s)", "");
    }
    else
    {
        PRINT("# %10s  %12s  %8s  %6s  %6s           in-place          \n", "", "", "",
              "", "");
        PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s %6s\n", "size", "count", "type",
              "redop", "root", timeStr, "algbw", "busbw", "#wrong");
        PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s  %5s\n", "(B)", "(elements)",
              "", "", "", "(us)", "(GB/s)", "(GB/s)", "");
    }
    Reporter reporter(output_file, output_format);

    std::vector<testThread> threads(nThreads);
    memset(threads.data(), 0, sizeof(struct testThread) * nThreads);

    for(int t = nThreads - 1; t >= 0; t--)
    {
        threads[t].args.minbytes   = minBytes;
        threads[t].args.maxbytes   = maxBytes;
        threads[t].args.stepbytes  = stepBytes;
        threads[t].args.stepfactor = stepFactor;
        threads[t].args.localRank  = localRank;

        threads[t].args.totalProcs             = totalProcs;
        threads[t].args.nProcs                 = ncclProcs;
        threads[t].args.proc                   = ncclProc;
        threads[t].args.nThreads               = nThreads;
        threads[t].args.thread                 = t;
        threads[t].args.nGpus                  = nGpus;
        threads[t].args.gpus                   = gpus.data() + t * nGpus;
        threads[t].args.sendbuffs              = sendbuffs.data() + t * nGpus;
        threads[t].args.recvbuffs              = recvbuffs.data() + t * nGpus;
        threads[t].args.expected               = expected.data() + t * nGpus;
        threads[t].args.ncclId                 = ncclId;
        threads[t].args.comms                  = comms + t * nGpus;
        threads[t].args.streams                = streams.data() + t * nGpus;
        threads[t].args.enable_out_of_place    = enable_out_of_place;
        threads[t].args.enable_in_place        = enable_in_place;
        threads[t].args.enable_cache_flush     = enable_cache_flush;
        threads[t].args.enable_rotating_tensor = enable_rotating_tensor;
        threads[t].args.errors                 = errors.data() + t;
        threads[t].args.bw                     = bw.data() + t;
        threads[t].args.bw_count               = bw_count.data() + t;

        threads[t].args.reportErrors = datacheck;
        threads[t].args.reporter     = &reporter;

        threads[t].func = parallel_init ? threadInit : threadRunTests;
        if(t)
            TESTCHECK(threadLaunch(threads.data() + t));
        else
            TESTCHECK(threads[t].func(&threads[t].args));
    }

    // Wait for other threads and accumulate stats and errors
    for(int t = nThreads - 1; t >= 0; t--)
    {
        if(t) pthread_join(threads[t].thread, NULL);
        TESTCHECK(threads[t].ret);
        if(t)
        {
            errors[0] += errors[t];
            bw[0] += bw[t];
            bw_count[0] += bw_count[t];
        }
    }

#ifdef MPI_SUPPORT
    MPI_Allreduce(MPI_IN_PLACE, &errors[0], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

    if(!parallel_init)
    {
        for(int i = 0; i < nGpus * nThreads; ++i)
        {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
            if(local_register) NCCLCHECK(ncclCommDeregister(comms[i], sendRegHandles[i]));
            if(local_register) NCCLCHECK(ncclCommDeregister(comms[i], recvRegHandles[i]));
#endif
            NCCLCHECK(ncclCommDestroy(comms[i]));
        }
        free(comms);
    }

    // Free off CUDA allocated memory
    for(int i = 0; i < nGpus * nThreads; i++)
    {
        if(sendbuffs[i]) CUDACHECK(cudaFree((char*) sendbuffs[i]));
        if(recvbuffs[i]) CUDACHECK(cudaFree((char*) recvbuffs[i]));
        if(datacheck) CUDACHECK(cudaFree(expected[i]));
    }
    CUDACHECK(cudaFreeHost(delta));
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
    free(sendRegHandles);
    free(recvRegHandles);
#endif

    envstr              = getenv("NCCL_TESTS_MIN_BW");
    double check_avg_bw = envstr ? atof(envstr) : -1;
    bw[0] /= bw_count[0];

    if(datacheck)
        PRINT("# Errors with asterisks indicate errors that have exceeded the maximum "
              "threshold.\n");
    PRINT("# Out of bounds values : %d %s\n", errors[0], errors[0] ? "FAILED" : "OK");
    PRINT("# Avg bus bandwidth    : %g %s\n", bw[0],
          check_avg_bw == -1 ? "" : (bw[0] < check_avg_bw * (0.9) ? "FAILED" : "OK"));
    PRINT("#\n");
#ifdef MPI_SUPPORT
    MPI_Comm_free(&mpi_comm);
    MPI_Finalize();
#endif

    // 'cuda-memcheck --leak-check full' requires this
    PRINT("%s\n", ncclGetLastError(NULL));
    cudaDeviceReset();

    if(errors[0] || bw[0] < check_avg_bw * (0.9))
        exit(EXIT_FAILURE);
    else
        exit(EXIT_SUCCESS);
}
