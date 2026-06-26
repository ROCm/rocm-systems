/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>

#define USE_RCCL_GATHER_SCATTER

// Build a deterministic sparse "send-size" matrix M[i][j] (sender i -> receiver j),
// identical on every rank, where every row and every column sums to exactly `total`.
//
// Method (Birkhoff--von Neumann): a doubly-stochastic matrix is a weighted sum of
// permutation matrices. Each random permutation contributes its weight to exactly one
// cell per row and per column, so all row/column sums equal the sum of the weights.
// Using integer weights that sum to `total` makes every row/column sum exactly `total`,
// which keeps the send/recv buffers exactly filled.
//
// Two tuning knobs (overridable via environment, but defaulted so all ranks agree):
//   RCCL_TESTS_A2AV_SPARSITY   fraction of (i,j) pairs forced to zero  [default 0.5]
//   RCCL_TESTS_A2AV_SIZESPREAD max/min ratio knob for nonzero weights  [default 3.0]
//   RCCL_TESTS_A2AV_SEED       shared RNG seed                         [default 2602]
//   RCCL_TESTS_A2AV_VERBOSE    if set, rank 0 prints the size matrix once
static void AlltoAllvGenSizeMatrix(int nranks, int rank, size_t total, std::vector<size_t>& M) {
  M.assign((size_t)nranks * nranks, 0);
  if (nranks <= 0 || total == 0) return;
  if (nranks == 1) { M[0] = total; return; }

  double sparsity = 0.5;
  double sizeSpread = 3.0;
  unsigned long seed = 2602;
  const char* s;
  if ((s = getenv("RCCL_TESTS_A2AV_SPARSITY")))   sparsity   = atof(s);
  if ((s = getenv("RCCL_TESTS_A2AV_SIZESPREAD"))) sizeSpread = atof(s);
  if ((s = getenv("RCCL_TESTS_A2AV_SEED")))       seed       = strtoul(s, NULL, 0);

  if (sparsity < 0.0)  sparsity = 0.0;
  if (sparsity > 0.95) sparsity = 0.95;
  if (sizeSpread < 1.0) sizeSpread = 1.0;

  // Number of permutation terms controls density. With k independent random
  // permutations, the expected fraction of zero cells is (1 - 1/N)^k, so
  // k ~= ln(sparsity) / ln(1 - 1/N) hits the requested sparsity.
  int k;
  if (sparsity > 0.0) {
    k = (int)std::lround(std::log(sparsity) / std::log(1.0 - 1.0 / nranks));
  } else {
    k = 4 * nranks; // effectively dense
  }
  if (k < 1) k = 1;

  std::mt19937_64 rng(seed);

  // Raw weights spread geometrically so max/min ~= sizeSpread.
  std::vector<double> raw(k);
  double rawSum = 0.0;
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  for (int t = 0; t < k; t++) {
    raw[t] = std::pow(sizeSpread, uni(rng)); // in [1, sizeSpread]
    rawSum += raw[t];
  }

  // Integer weights summing exactly to `total`.
  std::vector<size_t> w(k, 0);
  size_t assigned = 0;
  for (int t = 0; t < k; t++) {
    w[t] = (size_t)std::floor(raw[t] / rawSum * (double)total);
    assigned += w[t];
  }
  for (size_t rem = total - assigned, t = 0; rem > 0; rem--, t = (t + 1) % k) w[t] += 1;

  // Accumulate weighted random permutation matrices.
  std::vector<int> perm(nranks);
  for (int i = 0; i < nranks; i++) perm[i] = i;
  for (int t = 0; t < k; t++) {
    for (int i = nranks - 1; i > 0; i--) { // Fisher-Yates
      std::uniform_int_distribution<int> d(0, i);
      std::swap(perm[i], perm[d(rng)]);
    }
    for (int i = 0; i < nranks; i++) M[(size_t)i * nranks + perm[i]] += w[t];
  }

  if (rank == 0 && getenv("RCCL_TESTS_A2AV_VERBOSE")) {
    static bool printed = false;
    if (!printed) {
      printed = true;
      size_t nz = 0, mn = (size_t)-1, mx = 0;
      for (size_t e = 0; e < M.size(); e++) {
        if (M[e]) { nz++; if (M[e] < mn) mn = M[e]; if (M[e] > mx) mx = M[e]; }
      }
      if (mn == (size_t)-1) mn = 0;
      printf("AlltoAllv size matrix (rows=sender, cols=receiver):\n");
      printf("config:    total=%zu  requestedSparsity=%.1f%%  sizeSpread=%.2f  N=%d  seed=%lu\n",
             total, 100.0 * sparsity, sizeSpread, nranks, seed);
      printf("realized:  sparsity=%.1f%%  sizeSpreadRatio(max/min)=%.2f  nonzero[min..max]=[%zu..%zu]  terms=%d\n",
             100.0 * (1.0 - (double)nz / M.size()),
             mn ? (double)mx / (double)mn : 0.0, mn, mx, k);
      for (int i = 0; i < nranks; i++) {
        for (int j = 0; j < nranks; j++) printf("%10zu", M[(size_t)i * nranks + j]);
        printf("\n");
      }
    }
  }
}

void AlltoAllvGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  if (count < nranks*nranks/2) {
    *sendcount = 0;
    *recvcount = 0;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = 0;
  } else {
    *paramcount = (count/nranks) & -(16/eltSize);
    *sendcount = nranks*(*paramcount);
    *recvcount = *sendcount;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
  }
}

testResult_t AlltoAllvInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33*rep+rank, 1, 0));

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

    // Shared sparse size matrix: M[j][r] = bytes sender j sends to receiver r.
    // `expected` for this rank is column `rank`, concatenated in sender order.
    std::vector<size_t> M;
    AlltoAllvGenSizeMatrix(nranks, rank, sendcount, M);

    size_t rdisp = 0;
    for (int j=0; j<nranks; j++) {
      size_t rcount = M[(size_t)j*nranks + rank];
      // Displacement of this chunk inside sender j's send buffer (prefix sum of row j).
      size_t sdisp = 0;
      for (int d=0; d<rank; d++) sdisp += M[(size_t)j*nranks + d];
      TESTCHECK(InitData(((char*)args->expected[i])+rdisp*wordSize(type), rcount, sdisp, type, ncclSum, 33*rep+j, 1, 0));
      rdisp += rcount;
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  // We don't support in-place alltoall
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void AlltoAllvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

testResult_t AlltoAllvRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl, void* bias = nullptr) {
  char* sptr = (char*)sendbuff + sendoffset;
  char* rptr = (char*)recvbuff + recvoffset;
  
  int nranks;
  NCCLCHECK(ncclCommCount(comm, &nranks));
  int rank;
  NCCLCHECK(ncclCommUserRank(comm, &rank));

  if (count == 0) return testSuccess;

  std::vector<size_t> sendcounts, recvcounts, sdispls, rdispls;
  try {
    sendcounts = std::vector<size_t>(nranks * nranks);
    recvcounts = std::vector<size_t>(nranks * nranks);
    sdispls = std::vector<size_t>(nranks * nranks);
    rdispls = std::vector<size_t>(nranks * nranks);
  } catch (const std::bad_alloc&) {
    printf("failed to allocate buffers for alltoallv\n");
    return testNcclError;
  }

  // Shared sparse size matrix: row `rank` = what we send, column `rank` = what we recv.
  std::vector<size_t> M;
  AlltoAllvGenSizeMatrix(nranks, rank, count*nranks, M);

  size_t sdisp = 0, rdisp = 0;
  for (int i = 0; i < nranks; i++) {
      size_t scount = M[(size_t)rank*nranks + i]; // rank -> i
      size_t rcount = M[(size_t)i*nranks + rank]; // i -> rank
      sendcounts[i+rank*nranks] = scount;
      recvcounts[i+rank*nranks] = rcount;
      sdispls[i+rank*nranks] = sdisp;
      rdispls[i+rank*nranks] = rdisp;
      sdisp += scount;
      rdisp += rcount;
      //printf("%d->%d: send %zu @ %zu, recv %zu @ %zu\n", rank, i, scount, sdispls[i+rank*nranks], rcount, rdispls[i+rank*nranks]);
  }

#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
  printf("NCCL 2.7 or later is needed for alltoallv. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
  return testNcclError;
#else
#if defined(RCCL_ALLTOALLV) && defined(USE_RCCL_GATHER_SCATTER) && NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
  if (test_ncclVersion >= NCCL_VERSION(2,28,0)) {
    NCCLCHECK(ncclAlltoAllv(sptr, sendcounts.data()+rank*nranks, sdispls.data()+rank*nranks, rptr, recvcounts.data()+rank*nranks, rdispls.data()+rank*nranks, type, comm, stream));
    return testSuccess;
  }
  if (test_ncclVersion >= NCCL_VERSION(2,19,0)) {
    NCCLCHECK(ncclAllToAllv(sptr, sendcounts.data()+rank*nranks, sdispls.data()+rank*nranks, rptr, recvcounts.data()+rank*nranks, rdispls.data()+rank*nranks, type, comm, stream));
    return testSuccess;
  }
  printf("RCCL 2.19 or later is needed for RCCL_ALLTOALLV. This test was compiled with %d.%d, but is running with RCCL %d.\n", NCCL_MAJOR, NCCL_MINOR, test_ncclVersion);
  return testNcclError;
#else
  NCCLCHECK(ncclGroupStart());
  for (int r=0; r<nranks; r++) {
    if (sendcounts[r+rank*nranks] != 0) {
      NCCLCHECK(ncclSend(
          sptr + sdispls[r+rank*nranks] * wordSize(type),
          sendcounts[r+rank*nranks],
          type,
          r,
          comm,
          stream));
    }
    if (recvcounts[r+rank*nranks] != 0) {
      NCCLCHECK(ncclRecv(
          rptr + rdispls[r+rank*nranks] * wordSize(type),
          recvcounts[r+rank*nranks],
          type,
          r,
          comm,
          stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());
#endif
#endif
  return testSuccess;
}

struct testColl alltoAllTest = {
  "AlltoAllv",
  AlltoAllvGetCollByteCount,
  AlltoAllvInitData,
  AlltoAllvGetBw,
  AlltoAllvRunColl,
  NULL,
  NULL
};

void AlltoAllvGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AlltoAllvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AlltoAllvRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &alltoAllTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = ncclNumTypes;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  for (int i=0; i<type_count; i++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
  }
  return testSuccess;
}

struct testEngine ncclTestEngine = {
  .getBuffSize = AlltoAllvGetBuffSize,
  .runTest = AlltoAllvRunTest
};
