/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the DDA all-to-all paths launched from ncclAllToAll.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_ALLTOALL_H_
#define DDA_ALLTOALL_H_

#include "nccl.h"

#include <cstdint>

struct ncclComm;

// Tuning / attribution knobs of the DDA AllToAll fabric paths (block counts, unrolls, pieces,
// negative controls). Production builds compile each knob to its default (no env lookup, the
// dead branches fold away); build with -DRCCL_DDA_A2A_TUNING=1 to read RCCL_<env> again.
// Env that stays in production (documented in docs/api-reference/env-variables.rst):
// RCCL_DDA_A2A_TIMEOUT (alltoall_dda_watchdog.h), RCCL_MEMALLOC_FABRIC.
// Selection thresholds live in rcclArchThresholds (a2a* fields), not in env.
// Define each knob in exactly one translation unit (RCCL_PARAM has external linkage).
#ifndef RCCL_DDA_A2A_TUNING
#define RCCL_DDA_A2A_TUNING 0
#endif
#if RCCL_DDA_A2A_TUNING
#include "param.h"
#define RCCL_A2A_KNOB(name, env, deftVal) RCCL_PARAM(name, env, deftVal)
#else
#define RCCL_A2A_KNOB(name, env, deftVal) \
  static constexpr int64_t rcclParam##name() { return (int64_t)(deftVal); }
#endif

/**
 * Check if DDA alltoall is eligible for the given parameters
 */
bool ncclAllToAllDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                ncclDataType_t datatype);

/**
 * Execute DDA alltoall operation using IPC
 */
ncclResult_t ncclAllToAllDdaIpc(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                ncclComm* comm, cudaStream_t stream);

// Total CTAs (grid blocks) each DDA alltoall launcher would use for the given
// operands. Delegates to the shared *Geom() helper, so block counts are
// guaranteed to match the actual launch grid.
uint32_t ncclAllToAllDdaIpcBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype);
uint32_t ncclAllToAllDdaFabricBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype);
uint32_t ncclAllToAllDdaFabricLLBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype);
uint32_t ncclAllToAllDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype);

/**
 * Check if DDA alltoall is eligible for the fabric/VMM path (runtime nRanks
 * up to kDdaMaxNranks, single- or multi-node within an MNNVL clique).
 */
bool ncclAllToAllDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                   ncclDataType_t datatype);

/**
 * Execute DDA alltoall operation using the fabric/VMM path
 */
ncclResult_t ncclAllToAllDdaFabric(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclComm* comm, cudaStream_t stream);

// LL-protocol fabric path (small-chunk fast lane, 16B lines, no barrier).
bool ncclAllToAllDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                     ncclDataType_t datatype);

ncclResult_t ncclAllToAllDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                     ncclComm* comm, cudaStream_t stream);

// LL128-protocol fabric path (mid-chunk fast lane, 128B lines, no barrier).
bool ncclAllToAllDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                        ncclDataType_t datatype);

ncclResult_t ncclAllToAllDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                        ncclComm* comm, cudaStream_t stream);

// Per-comm device error word (alltoall_dda_watchdog.h): comm->ddaA2aErrHost mirrors the wait-site id
// of the first failure. Two A2A kernels of one comm in flight at once is the caller's ordering error
// (ncclInvalidUsage); every wait site is a peer that did not arrive in time (ncclRemoteError).
// A wait that observed the abort flag stores kDdaA2aErrAborted: not reported by ncclCommGetAsyncError
// (the caller aborted), but a later ncclAlltoAll on a comm that outlived the flag (shrink-abort, or
// a sibling sharing it) is refused, since that comm's A2A protocol state is out of step.
constexpr uint32_t kDdaA2aErrConcurrent = 30;
constexpr uint32_t kDdaA2aErrAborted = 0xAB;
inline ncclResult_t rcclDdaA2aErrorResult(uint32_t site) {
  return site == 0 ? ncclSuccess
       : (site == kDdaA2aErrConcurrent || site == kDdaA2aErrAborted) ? ncclInvalidUsage
                                                                        : ncclRemoteError;
}
// What ncclCommGetAsyncError reports for a site (aborts are the caller's own doing).
inline ncclResult_t rcclDdaA2aAsyncResult(uint32_t site) {
  return site == kDdaA2aErrAborted ? ncclSuccess : rcclDdaA2aErrorResult(site);
}
const char* ncclDdaA2aWdSiteName(uint32_t site);

#endif
