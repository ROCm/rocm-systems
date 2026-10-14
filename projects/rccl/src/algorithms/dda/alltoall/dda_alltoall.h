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
// RCCL_DDA_A2A_DIRECT / _PUSH (kill switches), RCCL_DDA_A2A_PUSH_BUFFER_SIZE, RCCL_DDA_A2A_TIMEOUT
// (alltoall_dda_watchdog.h), RCCL_MEMALLOC_FABRIC.
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

// Creates the push staging buffer and the Direct flag buffer on every rank (collective: every
// rank must call it on the same AllToAll; dda_alltoall_fabric_lazy_host.h). No-op unless
// comm->ddaA2aLazy is dda_a2a_lazy::kUntried. Returns an error only if bootstrap failed.
ncclResult_t ncclDdaA2aLazyInit(ncclComm* comm);

// One-shot PUSH fabric path (gfx1250, mid/large sizes): own fabric buffer
// (ncclDdaA2aLazyInit). Above the buffer the op runs in rounds (see
// ncclAllToAllDdaFabricPushRoundBytes).
ncclResult_t ncclDdaA2aPushCommFini(ncclComm* comm);
bool ncclAllToAllDdaFabricPushEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                       ncclDataType_t datatype);
uint32_t ncclAllToAllDdaFabricPushBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype);
// Per-peer bytes moved by one push launch (== perPeer unless the op exceeds the agreed buffer).
size_t ncclAllToAllDdaFabricPushRoundBytes(ncclComm* comm, size_t perPeer);
ncclResult_t ncclAllToAllDdaFabricPush(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                       ncclComm* comm, cudaStream_t stream);

// Zero-copy one-shot push into the peers' recvbuff in a symmetric window (gfx1250); flag buffer
// from ncclDdaA2aLazyInit (dda_alltoall_fabric_direct_host.h).
ncclResult_t ncclDdaA2aDirectCommFini(ncclComm* comm);
// Shape + "recvbuff in a symmetric window", independent of the path's state (the lazy trigger);
// no side effects.
bool ncclAllToAllDdaFabricDirectWindowed(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                         ncclDataType_t datatype);
// Windowed + the agreed flag buffer + geometry + every peer's mapping; no side effects.
bool ncclAllToAllDdaFabricDirectEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                         ncclDataType_t datatype);
uint32_t ncclAllToAllDdaFabricDirectBlocks(ncclComm* comm);
// Any failure poisons the comm (ncclDdaA2aPoison, kDdaA2aErrLaunch), as a push call does.
ncclResult_t ncclAllToAllDdaFabricDirect(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         ncclComm* comm, cudaStream_t stream);

// Same-package sibling of `rank` from topology, for the CU-split push kernel's sibling row and the
// two-rank push-over-Direct rule (rcclSelectAlltoAllFabric). The two logical GPUs of one gfx1250 DPX
// package share the host and the PCI bus/device, and differ in the function digit of busId (e.g.
// 0x104000 / 0x104001); separate physical GPUs (SPX) differ in bus/device. Group = ranks with the same host hash and busId >> 4; a rank has a sibling only if its
// group is exactly two ranks with different busIds (two ranks on one device are not siblings).
// Symmetric by construction; -1 when there is none. hostOf(r) / busOf(r): peer r's host hash / busId.
template <class HostOf, class BusOf>
int rcclDdaA2aTopoSibling(int nRanks, int rank, HostOf hostOf, BusOf busOf) {
  int sib = -1;
  for (int r = 0; r < nRanks; ++r) {
    if (r == rank || hostOf(r) != hostOf(rank) || (busOf(r) >> 4) != (busOf(rank) >> 4)) continue;
    if (sib >= 0 || busOf(r) == busOf(rank)) return -1;
    sib = r;
  }
  return sib;
}

// One push call: the rounds (per-peer bytes [off, off + c) each, launch(off, len, last)), stopping
// at the first error. Any error -- c == 0 included -- calls poison() before returning it: peers may
// already have launched this call, so this rank's device epoch is now out of step with theirs and
// its next push op could match their newer flags and copy stale data. poison() marks the
// communicator failed (ncclDdaA2aPoison), so later AllToAlls are refused instead.
template <class Launch, class Poison>
ncclResult_t rcclDdaA2aPushRun(size_t perPeer, size_t c, Launch launch, Poison poison) {
  ncclResult_t r = c == 0 ? ncclInvalidUsage : ncclSuccess;
  for (size_t off = 0; r == ncclSuccess && off < perPeer; off += c) {
    const size_t len = perPeer - off < c ? perPeer - off : c;
    r = launch(off, len, off + len >= perPeer);
  }
  if (r != ncclSuccess) poison();
  return r;
}

// Per-comm device error word (alltoall_dda_watchdog.h): comm->ddaA2aErrHost mirrors the wait-site id
// of the first failure. Two A2A kernels of one comm in flight at once is the caller's ordering error
// (ncclInvalidUsage); every wait site is a peer that did not arrive in time (ncclRemoteError).
// A wait that observed the abort flag stores kDdaA2aErrAborted: not reported by ncclCommGetAsyncError
// (the caller aborted), but a later ncclAlltoAll on a comm that outlived the flag (shrink-abort, or
// a sibling sharing it) is refused, since that comm's A2A protocol state is out of step.
constexpr uint32_t kDdaA2aErrConcurrent = 30;
// Host-side: a push / Direct call failed on the host (launch error) after the comm's device epoch may
// have gone out of step with its peers (rcclDdaA2aPushRun, ncclAllToAllDdaFabricDirect):
// ncclUnhandledCudaError from then on.
constexpr uint32_t kDdaA2aErrLaunch = 31;
constexpr uint32_t kDdaA2aErrAborted = 0xAB;
inline ncclResult_t rcclDdaA2aErrorResult(uint32_t site) {
  return site == 0 ? ncclSuccess
       : (site == kDdaA2aErrConcurrent || site == kDdaA2aErrAborted) ? ncclInvalidUsage
       : site == kDdaA2aErrLaunch                                    ? ncclUnhandledCudaError
                                                                        : ncclRemoteError;
}
// Host mirror half of a poison: stores `site` if the word is still healthy (the first failure wins).
inline void rcclDdaA2aPoisonHostWord(uint32_t* word, uint32_t site) {
  uint32_t expected = 0;
  (void)__atomic_compare_exchange_n(word, &expected, site, /*weak=*/false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}
// Marks the comm's AllToAll state failed with `site`: host mirror (later ncclAlltoAll calls refuse,
// ncclCommGetAsyncError reports it) and, best effort and asynchronously, the device word (kernels of
// earlier graph captures then return at entry). fabric_init.cu.
void ncclDdaA2aPoison(ncclComm* comm, uint32_t site);
// What ncclCommGetAsyncError reports for a site (aborts are the caller's own doing).
inline ncclResult_t rcclDdaA2aAsyncResult(uint32_t site) {
  return site == kDdaA2aErrAborted ? ncclSuccess : rcclDdaA2aErrorResult(site);
}
const char* ncclDdaA2aWdSiteName(uint32_t site);

#endif
