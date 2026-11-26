/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef RCCL_HIER_HOST_H_
#define RCCL_HIER_HOST_H_

#include "nccl.h"

#include <cstddef>

struct ncclComm;

// Host-hierarchical collectives group ranks by physical host (peerInfo hostHash).
// A scale-up fabric such as gfx1250 MNNVL reports the whole clique as
// comm->nNodes == 1, so the nNodes-based hierarchical paths never see the hosts.

// comm->hostHierState values.
enum rcclHostHierState {
  rcclHostHierUnprobed = 0,
  rcclHostHierOff, // No usable host layout, or a host-hierarchical child comm
  rcclHostHierLayout, // Layout known, child comms not built yet
  rcclHostHierReady, // Child comms built on every rank
  rcclHostHierFailed, // Building the child comms failed on some rank. Not retried.
};

struct rcclHostLayout {
  int nHosts;
  int ranksPerHost;
  int hostIndex; // Hosts are numbered in order of their first rank
  int localRank; // Position among this host's ranks, in rank order
};

struct rcclHostHier {
  struct rcclHostLayout layout;
  ncclComm_t intra; // Ranks of this host, ordered by local rank
  ncclComm_t inter; // The same local rank on every host, ordered by host. Null with one host
  // Shared by every host-hierarchical collective on the comm, so running them
  // concurrently on two streams is unsafe.
  void* scratch;
  size_t scratchBytes;
  // Recorded after each host-hierarchical collective, on whichever stream it ran. Consecutive collectives on the comm
  // are ordered on the device, so the latest record follows every earlier use of scratch.
  cudaEvent_t scratchUsed;
};

// Returns true and fills layout when every host holds the same number of ranks,
// and at least two. Does not read or change comm->hostHierState.
bool rcclHostLayoutGet(const struct ncclComm* comm, struct rcclHostLayout* layout);

// Cached rcclHostLayoutGet. Null when the comm has no usable layout, is a child
// of a host-hierarchical comm, or failed to build its child comms.
const struct rcclHostLayout* rcclHostHierGetLayout(struct ncclComm* comm);

// Builds the intra-host and inter-host comms on the first call. Collective over
// comm: every rank must call it together, outside a group. The ranks agree on the
// result, so either all of them use the children or none do.
ncclResult_t rcclHostHierEnsure(struct ncclComm* comm);

// Destroys the child comms, or aborts them when comm is being aborted. Called from commFree.
ncclResult_t rcclHostHierTeardown(struct ncclComm* comm);

// Stores the comm's child comms in children and returns how many there are, at most two. Split with splitShare = 0,
// they keep their own abort flags, which the parent's abort and revoke must also set.
int rcclHostHierChildren(const struct ncclComm* comm, ncclComm_t children[2]);

// Sizes, in bytes, at which host-hierarchical AllReduce beat the default path in
// gfx1250 MNNVL fp32 sum measurements with 8 ranks per host. False for unmeasured host counts.
bool rcclHostHierAllReduceAutoWindow(int nHosts, size_t bytes);

// Whether RCCL_HOST_HIER_ALLREDUCE and the operands allow host-hierarchical
// AllReduce. registeredBuffers: the send or receive buffer is in a registered
// window, which auto mode leaves to the paths that use it. The caller also checks
// the dispatch context: group depth 0, no graph capture, no enqueue-only mode.
bool rcclHostHierAllReduceEligible(struct ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                                   bool registeredBuffers);

// ReduceScatter within the host, AllReduce across hosts, then AllGather within
// the host. Needs rcclHostHierEnsure to have succeeded.
ncclResult_t rcclHostHierAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclRedOp_t op, struct ncclComm* comm, cudaStream_t stream);

#endif
