/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "cuda_runtime.h"

void
AlltoAllGetCollByteCount(size_t* sendcount, size_t* recvcount, size_t* paramcount,
                         size_t* sendInplaceOffset, size_t* recvInplaceOffset,
                         size_t count, size_t eltSize, int nranks)
{
    *paramcount        = (count / nranks) & -(16 / eltSize);
    *sendcount         = nranks * (*paramcount);
    *recvcount         = *sendcount;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
}

testResult_t
AlltoAllInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root,
                 int rep, int in_place)
{
    size_t sendcount = args->sendBytes / wordSize(type);
    size_t recvcount = args->expectedBytes / wordSize(type);
    int    nranks    = args->nProcs * args->nThreads * args->nGpus;

    for(int i = 0; i < args->nGpus; i++)
    {
        CUDACHECK(cudaSetDevice(args->gpus[i]));
        int rank = ((args->proc * args->nThreads + args->thread) * args->nGpus + i);
        CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
        void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
        TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33 * rep + rank, 1, 0));
        for(int j = 0; j < nranks; j++)
        {
            size_t partcount = sendcount / nranks;
            TESTCHECK(InitData((char*) args->expected[i] + j * partcount * wordSize(type),
                               partcount, rank * partcount, type, ncclSum, 33 * rep + j,
                               1, 0));
        }
        CUDACHECK(cudaDeviceSynchronize());
    }
    // We don't support in-place alltoall
    args->reportErrors = in_place ? 0 : 1;
    return testSuccess;
}

void
AlltoAllGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw,
              int nranks)
{
    double baseBw = (double) (count * nranks * typesize) / 1.0E9 / sec;

    *algBw        = baseBw;
    double factor = ((double) (nranks - 1)) / ((double) (nranks));
    *busBw        = baseBw * factor;
}

testResult_t
AlltoAllRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type,
                ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
{
    NCCLCHECK(ncclAllToAll(sendbuff, recvbuff, count, type, comm, stream));
    return testSuccess;
}

struct testColl alltoAllTest = { "AlltoAll", AlltoAllGetCollByteCount, AlltoAllInitData,
                                 AlltoAllGetBw, AlltoAllRunColl };

void
AlltoAllGetBuffSize(size_t* sendcount, size_t* recvcount, size_t count, int nranks)
{
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    AlltoAllGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset,
                             &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t
AlltoAllRunTest(struct threadArgs* args, int root, ncclDataType_t type,
                const char* typeName, ncclRedOp_t op, const char* opName)
{
    args->collTest = &alltoAllTest;
    ncclDataType_t* run_types;
    const char**    run_typenames;
    int             type_count;

    if((int) type != -1)
    {
        type_count    = 1;
        run_types     = &type;
        run_typenames = &typeName;
    }
    else
    {
        type_count    = test_typenum;
        run_types     = test_types;
        run_typenames = test_typenames;
    }

    for(int i = 0; i < type_count; i++)
    {
        TESTCHECK(
            TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t) 0, "none", -1));
    }
    return testSuccess;
}

struct testEngine ncclTestEngine = { AlltoAllGetBuffSize, AlltoAllRunTest };
