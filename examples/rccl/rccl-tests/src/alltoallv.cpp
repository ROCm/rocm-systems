/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "cuda_runtime.h"

#define USE_RCCL_GATHER_SCATTER

void
AlltoAllvGetCollByteCount(size_t* sendcount, size_t* recvcount, size_t* paramcount,
                          size_t* sendInplaceOffset, size_t* recvInplaceOffset,
                          size_t count, size_t eltSize, int nranks)
{
    if(count < nranks * nranks / 2)
    {
        *sendcount         = 0;
        *recvcount         = 0;
        *sendInplaceOffset = 0;
        *recvInplaceOffset = 0;
        *paramcount        = 0;
    }
    else
    {
        *paramcount        = (count / nranks) & -(16 / eltSize);
        *sendcount         = nranks * (*paramcount);
        *recvcount         = *sendcount;
        *sendInplaceOffset = 0;
        *recvInplaceOffset = 0;
    }
}

testResult_t
AlltoAllvInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root,
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

#if 0
    int *dataHost = (int *)malloc(args->sendBytes);
    cudaMemcpy(dataHost, data, args->sendBytes, cudaMemcpyDeviceToHost);
    printf(" Rank [%d] Original: ", rank);
    for(int j=0; j<sendcount; j++) {
	    printf("%d:%d ", j, dataHost[j]);
    }
    printf("\n");
    free(dataHost);
#endif

        size_t rdisp      = 0;
        size_t data_count = sendcount * 2 / nranks;
        size_t chunksize  = data_count / nranks;
        for(int j = 0; j < nranks; j++)
        {
            size_t scount = 0, rcount = ((j + rank) % nranks) * chunksize;
            if((j + rank) % nranks == 0)
                rcount += (sendcount - chunksize * (nranks - 1) * nranks / 2);
            size_t sdisp = 0;
            for(int k = 0; k < nranks; k++)
            {
                scount = ((k + j) % nranks) * chunksize;
                if((k + j) % nranks == 0)
                    scount += (sendcount - chunksize * (nranks - 1) * nranks / 2);
                if(k == rank) break;
                sdisp += scount;
            }
            TESTCHECK(InitData(((char*) args->expected[i]) + rdisp * wordSize(type),
                               rcount, sdisp, type, ncclSum, 33 * rep + j, 1, 0));
            rdisp += rcount;
        }
        CUDACHECK(cudaDeviceSynchronize());
    }
    // We don't support in-place alltoall
    args->reportErrors = in_place ? 0 : 1;
    return testSuccess;
}

void
AlltoAllvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw,
               int nranks)
{
    double baseBw = (double) (count * nranks * typesize) / 1.0E9 / sec;

    *algBw        = baseBw;
    double factor = ((double) (nranks - 1)) / ((double) (nranks));
    *busBw        = baseBw * factor;
}

testResult_t
AlltoAllvRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type,
                 ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
{
    int nranks;
    NCCLCHECK(ncclCommCount(comm, &nranks));
    int rank;
    NCCLCHECK(ncclCommUserRank(comm, &rank));

    if(count == 0) return testSuccess;

    size_t *sendcounts, *recvcounts, *sdispls, *rdispls;
    sendcounts = (size_t*) malloc(nranks * nranks * sizeof(size_t));
    recvcounts = (size_t*) malloc(nranks * nranks * sizeof(size_t));
    sdispls    = (size_t*) malloc(nranks * nranks * sizeof(size_t));
    rdispls    = (size_t*) malloc(nranks * nranks * sizeof(size_t));
    if(sendcounts == nullptr || recvcounts == nullptr || sdispls == nullptr ||
       rdispls == nullptr)
    {
        printf("failed to allocate buffers for alltoallv\n");
        return testNcclError;
    }

    size_t disp      = 0;
    size_t chunksize = count * 2 / nranks;
    for(int i = 0; i < nranks; i++)
    {
        size_t scount = ((i + rank) % nranks) * chunksize;
        if((i + rank) % nranks == 0)
            scount += (count * nranks - chunksize * (nranks - 1) * nranks / 2);
        sendcounts[i + rank * nranks] = recvcounts[i + rank * nranks] = scount;
        sdispls[i + rank * nranks] = rdispls[i + rank * nranks] = disp;
        disp += scount;
        // printf("%d->%d: sendcounts/recvcounts %lx sdispls/rdispls %lx\n", rank, i,
        // sendcounts[i+rank*nranks]*wordSize(type),
        // sdispls[i+rank*nranks]*wordSize(type));
    }

#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
    printf(
        "NCCL 2.7 or later is needed for alltoallv. This test was compiled with %d.%d.\n",
        NCCL_MAJOR, NCCL_MINOR);
    return testNcclError;
#else
#    if defined(RCCL_ALLTOALLV) && defined(USE_RCCL_GATHER_SCATTER)
    NCCLCHECK(ncclAllToAllv(sendbuff, sendcounts + rank * nranks, sdispls + rank * nranks,
                            recvbuff, recvcounts + rank * nranks, rdispls + rank * nranks,
                            type, comm, stream));
#    else
    NCCLCHECK(ncclGroupStart());
    for(int r = 0; r < nranks; r++)
    {
        if(sendcounts[r + rank * nranks] != 0)
        {
            NCCLCHECK(
                ncclSend(((char*) sendbuff) + sdispls[r + rank * nranks] * wordSize(type),
                         sendcounts[r + rank * nranks], type, r, comm, stream));
        }
        if(recvcounts[r + rank * nranks] != 0)
        {
            NCCLCHECK(
                ncclRecv(((char*) recvbuff) + rdispls[r + rank * nranks] * wordSize(type),
                         recvcounts[r + rank * nranks], type, r, comm, stream));
        }
    }
    NCCLCHECK(ncclGroupEnd());
#    endif
#endif
    free(sendcounts);
    free(recvcounts);
    free(sdispls);
    free(rdispls);
    return testSuccess;
}

struct testColl alltoAllTest = { "AlltoAllv", AlltoAllvGetCollByteCount,
                                 AlltoAllvInitData, AlltoAllvGetBw, AlltoAllvRunColl };

void
AlltoAllvGetBuffSize(size_t* sendcount, size_t* recvcount, size_t count, int nranks)
{
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    AlltoAllvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset,
                              &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t
AlltoAllvRunTest(struct threadArgs* args, int root, ncclDataType_t type,
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
        type_count    = ncclNumTypes;
        run_types     = test_types;
        run_typenames = test_typenames;
    }

    for(int i = 0; i < type_count; i++)
    {
        TESTCHECK(
            TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t) 0, "", -1));
    }
    return testSuccess;
}

struct testEngine ncclTestEngine = { AlltoAllvGetBuffSize, AlltoAllvRunTest };
