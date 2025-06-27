/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__

#include "rccl/rccl.h"
#include <algorithm>
#include <cstdint>
#include <stdio.h>
#ifdef MPI_SUPPORT
#    include "mpi.h"
#endif
#include "nccl1_compat.h"
#include "timer.h"
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <string>

// Ensures backward compatibility for FP8 types in RCCL 2.24.3 and later
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 24, 3)
#    define ncclFp8E4M3 ncclFloat8e4m3
#    define ncclFp8E5M2 ncclFloat8e5m2
#endif

// For nccl.h < 2.13 since we define a weak fallback
extern "C" char const*
ncclGetLastError(ncclComm_t comm);

#define CUDACHECK(cmd)                                                                   \
    do                                                                                   \
    {                                                                                    \
        cudaError_t err = cmd;                                                           \
        if(err != cudaSuccess)                                                           \
        {                                                                                \
            char hostname[1024];                                                         \
            getHostName(hostname, 1024);                                                 \
            printf("%s: Test CUDA failure %s:%d '%s'\n", hostname, __FILE__, __LINE__,   \
                   cudaGetErrorString(err));                                             \
            return testCudaError;                                                        \
        }                                                                                \
    } while(0)

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 13, 0)
#    define NCCLCHECK(cmd)                                                               \
        do                                                                               \
        {                                                                                \
            ncclResult_t res = cmd;                                                      \
            if(res != ncclSuccess)                                                       \
            {                                                                            \
                char hostname[1024];                                                     \
                getHostName(hostname, 1024);                                             \
                printf("%s: Test NCCL failure %s:%d "                                    \
                       "'%s / %s'\n",                                                    \
                       hostname, __FILE__, __LINE__, ncclGetErrorString(res),            \
                       ncclGetLastError(NULL));                                          \
                return testNcclError;                                                    \
            }                                                                            \
        } while(0)
#else
#    define NCCLCHECK(cmd)                                                               \
        do                                                                               \
        {                                                                                \
            ncclResult_t res = cmd;                                                      \
            if(res != ncclSuccess)                                                       \
            {                                                                            \
                char hostname[1024];                                                     \
                getHostName(hostname, 1024);                                             \
                printf("%s: Test NCCL failure %s:%d '%s'\n", hostname, __FILE__,         \
                       __LINE__, ncclGetErrorString(res));                               \
                return testNcclError;                                                    \
            }                                                                            \
        } while(0)
#endif

typedef enum
{
    testSuccess       = 0,
    testInternalError = 1,
    testCudaError     = 2,
    testNcclError     = 3,
    testTimeout       = 4,
    testNumResults    = 5
} testResult_t;

// Relay errors up and trace
#define TESTCHECK(cmd)                                                                   \
    do                                                                                   \
    {                                                                                    \
        testResult_t r = cmd;                                                            \
        if(r != testSuccess)                                                             \
        {                                                                                \
            char hostname[1024];                                                         \
            getHostName(hostname, 1024);                                                 \
            printf(" .. %s pid %d: Test failure %s:%d\n", hostname, getpid(), __FILE__,  \
                   __LINE__);                                                            \
            return r;                                                                    \
        }                                                                                \
    } while(0)

struct testColl
{
    const char name[20];
    void (*getCollByteCount)(size_t* sendcount, size_t* recvcount, size_t* paramcount,
                             size_t* sendInplaceOffset, size_t* recvInplaceOffset,
                             size_t count, size_t eltSize, int nranks);
    testResult_t (*initData)(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op,
                             int root, int rep, int in_place);
    void (*getBw)(size_t count, int typesize, double sec, double* algBw, double* busBw,
                  int nranks);
    testResult_t (*runColl)(void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t type, ncclRedOp_t op, int root,
                            ncclComm_t comm, cudaStream_t stream);
};
extern struct testColl allReduceTest;
extern struct testColl allGatherTest;
extern struct testColl reduceScatterTest;
extern struct testColl broadcastTest;
extern struct testColl reduceTest;
extern struct testColl alltoAllTest;

class Reporter
{
public:
    Reporter(std::string fileName, std::string outputFormat);
    ~Reporter()
    {
        if(_outputValid)
        {
            _out.close();
        }
    };
    void setParameters(const size_t numCycle, const char* name, const char* typeName,
                       const char* opName);
    void addResult(int gpusPerRank, int ranksPerNode, int totalRanks, size_t numBytes,
                   int inPlace, double timeUsec, double algBw, double busBw,
                   int64_t wrongElts = -1);

private:
    bool isMainThread();
    template <typename T>
    std::pair<std::string, std::string> makeValueKeyPair(T v, std::string k)
    {
        return std::make_pair(std::to_string(v), k);
    };
    template <>
    std::pair<std::string, std::string> makeValueKeyPair<std::string>(std::string v,
                                                                      std::string k)
    {
        return std::make_pair("\"" + v + "\"", k);
    };

    bool          _outputValid = false;
    std::ofstream _out;
    std::string   _outputFormat;
    size_t        _numCycle = 0;
    std::string   _collectiveName;
    std::string   _typeName;
    std::string   _opName;
};

struct testEngine
{
    void (*getBuffSize)(size_t* sendcount, size_t* recvcount, size_t count, int nranks);
    testResult_t (*runTest)(struct threadArgs* args, int root, ncclDataType_t type,
                            const char* typeName, ncclRedOp_t op, const char* opName);
};

extern struct testEngine ncclTestEngine;

struct threadArgs
{
    size_t nbytes;
    size_t minbytes;
    size_t maxbytes;
    size_t stepbytes;
    size_t stepfactor;

    int           totalProcs;
    int           nProcs;
    int           proc;
    int           nThreads;
    int           thread;
    int           nGpus;
    int*          gpus;
    int           localRank;
    int           enable_out_of_place;
    int           enable_in_place;
    int           enable_cache_flush;
    int           enable_rotating_tensor;
    void**        sendbuffs;
    size_t        sendBytes;
    size_t        sendInplaceOffset;
    void**        recvbuffs;
    size_t        recvInplaceOffset;
    ncclUniqueId  ncclId;
    ncclComm_t*   comms;
    cudaStream_t* streams;

    void**  expected;
    size_t  expectedBytes;
    int*    errors;
    double* bw;
    int*    bw_count;

    int reportErrors;

    struct testColl* collTest;

    Reporter* reporter;
};

typedef testResult_t (*threadFunc_t)(struct threadArgs* args);
struct testThread
{
    pthread_t         thread;
    threadFunc_t      func;
    struct threadArgs args;
    testResult_t      ret;
};

// Provided by common.cu
extern void
Barrier(struct threadArgs* args);
extern testResult_t
TimeTest(struct threadArgs* args, ncclDataType_t type, const char* typeName,
         ncclRedOp_t op, const char* opName, int root);
extern testResult_t
InitDataReduce(void* data, const size_t count, const size_t offset, ncclDataType_t type,
               ncclRedOp_t op, const uint64_t seed, const int nranks);
extern testResult_t
InitData(void* data, const size_t count, size_t offset, ncclDataType_t type,
         ncclRedOp_t op, const uint64_t seed, const int nranks, const int rank);
extern void
AllocateBuffs(void** sendbuff, void** recvbuff, void** expected, void** expectedHost,
              size_t nbytes, int nranks);

#include <unistd.h>

static void
getHostName(char* hostname, int maxlen)
{
    gethostname(hostname, maxlen);
    for(int i = 0; i < maxlen; i++)
    {
        if(hostname[i] == '.')
        {
            hostname[i] = '\0';
            return;
        }
    }
}

#include <stdint.h>

static uint64_t
getHash(const char* string, size_t n)
{
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for(size_t c = 0; c < n; c++)
    {
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 */
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static uint64_t
getHostHash(const char* hostname)
{
    char hostHash[1024];

    // Fall back is the hostname if something fails
    (void) strncpy(hostHash, hostname, sizeof(hostHash));
    int offset = strlen(hostHash);

    FILE* file = fopen(HOSTID_FILE, "r");
    if(file != NULL)
    {
        char* p;
        if(fscanf(file, "%ms", &p) == 1)
        {
            strncpy(hostHash + offset, p, sizeof(hostHash) - offset - 1);
            free(p);
        }
    }
    fclose(file);

    // Make sure the string is terminated
    hostHash[sizeof(hostHash) - 1] = '\0';

    return getHash(hostHash, strlen(hostHash));
}

static size_t
wordSize(ncclDataType_t type)
{
    switch(type)
    {
        case ncclChar:
#if NCCL_MAJOR >= 2
        // case ncclInt8:
        case ncclUint8:
#    if NCCL_MAJOR >= 2 && RCCL_FLOAT8 == 1
        case ncclFp8E4M3:
        case ncclFp8E5M2:
#    endif
#endif
            return 1;
        case ncclHalf:
#if NCCL_MAJOR >= 2 && RCCL_BFLOAT16 == 1
        case ncclBfloat16:
#endif
            // case ncclFloat16:
            return 2;
        case ncclInt:
        case ncclFloat:
#if NCCL_MAJOR >= 2
        // case ncclInt32:
        case ncclUint32:
            // case ncclFloat32:
#endif
            return 4;
        case ncclInt64:
        case ncclUint64:
        case ncclDouble:
            // case ncclFloat64:
            return 8;
        default: return 0;
    }
}

extern int test_ncclVersion;  // init'd with ncclGetVersion()
typedef enum
{
    ncclCoarse      = 0,
    ncclFine        = 1,
    ncclHost        = 2,
    ncclManaged     = 3,
    nccl_NUM_MTYPES = 4
} ncclMemoryType_t;
extern const char* test_memorytypes[nccl_NUM_MTYPES];
constexpr int      test_opNumMax =
    (int) ncclNumOps + (NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0) ? 1 : 0);
extern int            test_opnum;
extern int            test_typenum;
extern ncclDataType_t test_types[ncclNumTypes];
extern const char*    test_typenames[ncclNumTypes];
extern ncclRedOp_t    test_ops[];
extern const char*    test_opnames[];

static int
ncclstringtotype(char* str)
{
    for(int t = 0; t < test_typenum; t++)
    {
        if(strcmp(str, test_typenames[t]) == 0)
        {
            return t;
        }
    }
    if(strcmp(str, "all") == 0)
    {
        return -1;
    }
    printf("invalid type %s, defaulting to %s .. \n", str, test_typenames[ncclFloat]);
    return ncclFloat;
}

static int
ncclstringtoop(char* str)
{
    for(int o = 0; o < test_opnum; o++)
    {
        if(strcmp(str, test_opnames[o]) == 0)
        {
            return o;
        }
    }
    if(strcmp(str, "all") == 0)
    {
        return -1;
    }
    printf("invalid op %s, defaulting to %s .. \n", str, test_opnames[ncclSum]);
    return ncclSum;
}

static int
ncclstringtoroot(char* str)
{
    if(strcmp(str, "all") == 0)
    {
        return -1;
    }
    return strtol(str, NULL, 0);
}

static int
ncclstringtomtype(char* str)
{
    for(int o = 0; o < nccl_NUM_MTYPES; o++)
    {
        if(strcmp(str, test_memorytypes[o]) == 0)
        {
            return o;
        }
    }
    printf("invalid memorytype %s, defaulting to %s .. \n", str,
           test_memorytypes[ncclCoarse]);
    return ncclCoarse;
}

extern int              is_main_proc;
extern thread_local int is_main_thread;
#define PRINT                                                                            \
    if(is_main_thread) printf

#endif
