/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the GIN-SDMA fakes. See gin_sdma_fakes.h.

#include "gin_sdma_fakes.h"

#include "algorithms/gin/gin_all_gather.h"
#include "algorithms/gin/gin_alltoall.h"
#include "algorithms/gin/gin_all_reduce.h"

#include "fail_loud.h"
#include "signature-drift.h"

#if defined(ENABLE_ROCSHMEM_GIN)
ASSERT_HOOK_MATCHES_PROD(g_allGatherGinSdmaEligible, ncclAllGatherGinSdmaEligible);
ASSERT_HOOK_MATCHES_PROD(g_allGatherGinSdma, ncclAllGatherGinSdma);
ASSERT_HOOK_MATCHES_PROD(g_allGatherGinSdmaCtaCount, ncclAllGatherGinSdmaCtaCount);

ASSERT_HOOK_MATCHES_PROD(g_allToAllGinSdmaEligible, ncclAllToAllGinSdmaEligible);
ASSERT_HOOK_MATCHES_PROD(g_allToAllGinSdma, ncclAllToAllGinSdma);

ASSERT_HOOK_MATCHES_PROD(g_allReduceGinSdmaEligible, ncclAllReduceGinSdmaEligible);
ASSERT_HOOK_MATCHES_PROD(g_allReduceGinSdma, ncclAllReduceGinSdma);
#endif /* ENABLE_ROCSHMEM_GIN */
#undef ASSERT_HOOK_MATCHES_PROD

#if defined(ENABLE_ROCSHMEM_GIN)
namespace {
// Generic lambda can be assigned to all the GinSdmaEligible hooks below.
// Unlike a template function, does not need explicit template arguments.
constexpr auto DefaultGinSdmaEligible = [](auto&&...) -> bool { return false; };

// The only differences between the defaults for the GinSdma hooks are the symbol names.
constexpr auto DefaultGinSdmaFunc(const char* symbol) {
  // Capture the symbol argument into the generic lambda.
  return [symbol](auto&&...) -> ncclResult_t { FailLoudUnfaked("gin_sdma_fakes", symbol); };
}
constexpr auto DefaultAllGatherGinSdma = DefaultGinSdmaFunc("ncclAllGatherGinSdma");
constexpr auto DefaultAllToAllGinSdma = DefaultGinSdmaFunc("ncclAllToAllGinSdma");
constexpr auto DefaultAllReduceGinSdma = DefaultGinSdmaFunc("ncclAllReduceGinSdma");

int DefaultAllGatherGinSdmaCtaCount(int, size_t) { return 0; }
}  // namespace

// src/algorithms/gin/sdma/gin_all_gather_sdma.cu
std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t)>
    g_allGatherGinSdmaEligible = DefaultGinSdmaEligible;
std::function<ncclResult_t(const void*, void*, size_t, ncclDataType_t, ncclComm*, hipStream_t)>
    g_allGatherGinSdma = DefaultAllGatherGinSdma;
bool ncclAllGatherGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                  ncclDataType_t datatype) {
  return g_allGatherGinSdmaEligible(comm, sendbuff, recvbuff, sendcount, datatype);
}
ncclResult_t ncclAllGatherGinSdma(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                  ncclComm* comm, hipStream_t stream) {
  return g_allGatherGinSdma(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}
std::function<int(int, size_t)> g_allGatherGinSdmaCtaCount = DefaultAllGatherGinSdmaCtaCount;
int ncclAllGatherGinSdmaCtaCount(int nRanks, size_t bytes) { return g_allGatherGinSdmaCtaCount(nRanks, bytes); }

// src/algorithms/gin/sdma/gin_alltoall_sdma.cu
std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t)>
    g_allToAllGinSdmaEligible = DefaultGinSdmaEligible;
std::function<ncclResult_t(const void*, void*, size_t, ncclDataType_t, ncclComm*, hipStream_t)>
    g_allToAllGinSdma = DefaultAllToAllGinSdma;
bool ncclAllToAllGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype) {
  return g_allToAllGinSdmaEligible(comm, sendbuff, recvbuff, count, datatype);
}
ncclResult_t ncclAllToAllGinSdma(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                 ncclComm* comm, hipStream_t stream) {
  return g_allToAllGinSdma(sendbuff, recvbuff, count, datatype, comm, stream);
}

// src/algorithms/gin/sdma/gin_all_reduce_sdma.cu
std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t)>
    g_allReduceGinSdmaEligible = DefaultGinSdmaEligible;
std::function<ncclResult_t(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm*, hipStream_t)>
    g_allReduceGinSdma = DefaultAllReduceGinSdma;
bool ncclAllReduceGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype, ncclRedOp_t op) {
  return g_allReduceGinSdmaEligible(comm, sendbuff, recvbuff, count, datatype, op);
}
ncclResult_t ncclAllReduceGinSdma(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                  ncclRedOp_t op, ncclComm* comm, hipStream_t stream) {
  return g_allReduceGinSdma(sendbuff, recvbuff, count, datatype, op, comm, stream);
}
bool ncclAllReduceGinSdmaYieldToDda(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t) {
  return false;
}
#endif /* ENABLE_ROCSHMEM_GIN */

void ResetGinSdmaFakes() {
#if defined(ENABLE_ROCSHMEM_GIN)
  g_allGatherGinSdmaEligible = DefaultGinSdmaEligible;
  g_allGatherGinSdma = DefaultAllGatherGinSdma;
  g_allGatherGinSdmaCtaCount = DefaultAllGatherGinSdmaCtaCount;
  g_allToAllGinSdmaEligible = DefaultGinSdmaEligible;
  g_allToAllGinSdma = DefaultAllToAllGinSdma;
  g_allReduceGinSdmaEligible = DefaultGinSdmaEligible;
  g_allReduceGinSdma = DefaultAllReduceGinSdma;
#endif /* ENABLE_ROCSHMEM_GIN */
}
