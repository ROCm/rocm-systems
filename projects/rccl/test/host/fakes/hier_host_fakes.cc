/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "hier_host_fakes.h"

#include "comm.h"
#include "fail_loud.h"
#include "hier_host.h"
#include "signature-drift.h"

std::function<bool(ncclComm*, size_t, ncclDataType_t, ncclRedOp_t)> g_hostHierAllReduceEligible =
    [](ncclComm*, size_t, ncclDataType_t, ncclRedOp_t) { return false; };
bool rcclHostHierAllReduceEligible(ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op) {
  return g_hostHierAllReduceEligible(comm, count, datatype, op);
}
ASSERT_HOOK_MATCHES_PROD(g_hostHierAllReduceEligible, rcclHostHierAllReduceEligible);

std::function<ncclResult_t(ncclComm*)> g_hostHierEnsure = [](ncclComm*) -> ncclResult_t {
  FailLoudUnfaked("hier_host_fakes", "rcclHostHierEnsure");
};
ncclResult_t rcclHostHierEnsure(ncclComm* comm) { return g_hostHierEnsure(comm); }
ASSERT_HOOK_MATCHES_PROD(g_hostHierEnsure, rcclHostHierEnsure);

std::function<ncclResult_t(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm*, hipStream_t)>
    g_hostHierAllReduce = [](const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm*,
                             hipStream_t) -> ncclResult_t { FailLoudUnfaked("hier_host_fakes", "rcclHostHierAllReduce"); };
ncclResult_t rcclHostHierAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclRedOp_t op, ncclComm* comm, hipStream_t stream) {
  return g_hostHierAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}
ASSERT_HOOK_MATCHES_PROD(g_hostHierAllReduce, rcclHostHierAllReduce);

std::function<ncclResult_t(ncclComm*)> g_hostHierTeardown = [](ncclComm* comm) {
  comm->hostHier = nullptr;
  return ncclSuccess;
};
ncclResult_t rcclHostHierTeardown(ncclComm* comm) { return g_hostHierTeardown(comm); }
ASSERT_HOOK_MATCHES_PROD(g_hostHierTeardown, rcclHostHierTeardown);

#undef ASSERT_HOOK_MATCHES_PROD
