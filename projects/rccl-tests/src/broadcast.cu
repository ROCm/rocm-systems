/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

// Set RCCL_TESTS_ALLGATHERV_ENABLE=1 to run grouped-broadcast (AllGatherV fusion) mode
static int broadcast_grouped = [] {
  const char* s = getenv("RCCL_TESTS_ALLGATHERV_ENABLE");
  return s != nullptr && s[0] == '1';
}();

void BroadcastGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  if (broadcast_grouped) {
    if (count < (size_t)nranks) {
      *sendcount = *recvcount = *paramcount = *sendInplaceOffset = *recvInplaceOffset = 0;
      return;
    }
    size_t chunkElems = (count / nranks) & -(16 / eltSize);
    if (chunkElems == 0) {
      *sendcount = *recvcount = *paramcount = *sendInplaceOffset = *recvInplaceOffset = 0;
      return;
    }
    *sendcount  = chunkElems;
    *recvcount  = chunkElems * nranks;
    *paramcount = chunkElems;
    *sendInplaceOffset = chunkElems;
    *recvInplaceOffset = 0;
  } else {
    *sendcount = count;
    *recvcount = count;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = *sendcount;
  }
}

testResult_t BroadcastInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  if (broadcast_grouped) {
    size_t chunkElems = args->sendBytes / wordSize(type);
    int nranks = args->nProcs * args->nThreads * args->nGpus;
    for (int i = 0; i < args->nGpus; i++) {
      CUDACHECK(cudaSetDevice(args->gpus[i]));
      int rank = (args->proc * args->nThreads + args->thread) * args->nGpus + i;
      CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
      void* data = in_place ? ((char*)args->recvbuffs[i]) + rank * args->sendBytes : args->sendbuffs[i];
      TESTCHECK(InitData(data, chunkElems, 0, type, ncclSum, rank + 1, 1, 0));
      for (int k = 0; k < nranks; k++) {
        char* slot = (char*)args->expected[i] + (size_t)k * chunkElems * wordSize(type);
        TESTCHECK(InitData(slot, chunkElems, 0, type, ncclSum, k + 1, 1, 0));
      }
      CUDACHECK(cudaDeviceSynchronize());
    }
    return testSuccess;
  }

  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    if (rank == root) TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, rep, 1, 0));
    TESTCHECK(InitData(args->expected[i], recvcount, 0, type, ncclSum, rep, 1, 0));
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

testResult_t  BroadcastGetAlgoProtoChannels(ncclComm_t comm, size_t count, ncclDataType_t type, int* algo, int* proto, int* nchannels) {
  if(rcclTestsGetAlgoInfo == NULL) return testInternalError;
  NCCLCHECK(rcclTestsGetAlgoInfo(comm, ncclFuncBroadcast , count, type , 0, 0, 1, algo, proto, nchannels));
  return testSuccess;
}

void BroadcastGetBw(size_t count, size_t typesize, double sec, double* algBw, double* busBw, int nranks) {
  if (broadcast_grouped) {
    double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;
    *algBw = baseBw;
    double factor = (double)(nranks - 1) / (double)nranks;
    *busBw = baseBw * factor;
  } else {
    double baseBw = (double)(count * typesize) / 1.0E9 / sec;
    *algBw = baseBw;
    double factor = 1;
    *busBw = baseBw * factor;
  }
}

testResult_t BroadcastRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  if (broadcast_grouped) {
    // nranks broadcasts with distinct roots inside one group — fuses into AllGatherV ring kernel
    if (deviceImpl != 0) return testNotImplemented;
    int nranks;
    NCCLCHECK(ncclCommCount(comm, &nranks));
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
    NCCLCHECK(ncclGroupStart());
    for (int k = 0; k < nranks; k++) {
      NCCLCHECK(ncclBroadcast(sptr, rptr + (size_t)k * count * wordSize(type),
                              count, type, /*root=*/k, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    return testSuccess;
  }

  if (deviceImpl == 0) {
    int rank;
    NCCLCHECK(ncclCommUserRank(comm, &rank));

    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
#if NCCL_MAJOR >= 2 && NCCL_MINOR >= 2
    NCCLCHECK(ncclBroadcast(sptr, rptr, count, type, root, comm, stream));
#else
    if (rank == root) {
      NCCLCHECK(ncclBcast(sptr, count, type, root, comm, stream));
    } else {
      NCCLCHECK(ncclBcast(rptr, count, type, root, comm, stream));
    }
#endif
  } else {
    return testNotImplemented;
  }
  return testSuccess;
}

struct testColl broadcastTest = {
  "Broadcast",
  BroadcastGetCollByteCount,
  BroadcastInitData,
  BroadcastGetBw,
  BroadcastRunColl,
  BroadcastGetAlgoProtoChannels,
  NULL
};

void BroadcastGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  BroadcastGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t BroadcastRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &broadcastTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if (broadcast_grouped) {
    for (int i = 0; i < type_count; i++)
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
    return testSuccess;
  }

  int begin_root, end_root;
  if (root != -1) {
    begin_root = end_root = root;
  } else {
    begin_root = 0;
    end_root = args->nProcs*args->nThreads*args->nGpus-1;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=begin_root; j<=end_root; j++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", j));
    }
  }
  return testSuccess;
}

NCCL_WEAK struct testEngine ncclTestEngine = {
  /* .getBuffSize = */ BroadcastGetBuffSize,
  /* .runTest = */ BroadcastRunTest
};
