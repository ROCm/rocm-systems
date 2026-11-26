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

// Scratch at least doubles each time it grows, so this covers any size.
constexpr int kHostHierMaxRetired = 64;

struct rcclHostHier {
  struct rcclHostLayout layout;
  ncclComm_t intra; // Ranks of this host, ordered by local rank
  ncclComm_t inter; // The same local rank on every host, ordered by host. Null with one host
  int* hostRanks; // Comm rank of local rank l on host h at h * ranksPerHost + l
  // Shared by every host-hierarchical collective on the comm, so running them
  // concurrently on two streams is unsafe.
  void* scratch;
  size_t scratchBytes;
  void* retired[kHostHierMaxRetired]; // Outgrown scratch buffers, freed at teardown
  int nRetired;
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

// Sizes, in bytes, at which host-hierarchical AllReduce beat the default path in gfx1250
// MNNVL fp32 sum measurements with 8 ranks per host on 2, 4 and 8 hosts and with 4 ranks
// per host on 8 hosts. The rcclHostHierAllReduceAutoType combinations gain at the
// same sizes. False for unmeasured layouts. Auto mode also needs the SPX tuning table
// for 4 ranks per host, and on two hosts the fused kernel between 2 and 8 MiB for the
// types the kernel takes.
bool rcclHostHierAllReduceAutoWindow(int nHosts, int ranksPerHost, size_t bytes);

// Sizes, in bytes, at which the fused DDA kernel beat the three collectives in gfx1250 fp32, fp16 and bf16 sum
// measurements on one and two hosts with 8 ranks per host. With 4 ranks per host it was slower at every size.
bool rcclHostHierAllReduceFusedWindow(int nHosts, int ranksPerHost, size_t bytes);

// Datatype and op combinations that auto mode uses at the fp32-sum sizes of rcclHostHierAllReduceAutoWindow, from
// gfx1250 measurements. With 8 ranks per host (DPX), sum on fp32, fp16, bf16 and fp8, which gained at every size of
// the window on two, four and eight hosts: prod, max, min, avg and integer sums were slower at 64-128 MiB on two and
// four hosts. With 4 ranks per host (SPX), every split op except fp8 avg,
// prod, max and min and int8 / uint8 max and min: on eight hosts fp8 avg, max and min were slower at 256-512 MiB and
// int8 and uint8 max and min at 256 MiB, and fp8 prod is unmeasured. The caller still applies opSplitsAcrossStages,
// which drops integer avg and PreMulSum.
bool rcclHostHierAllReduceAutoType(int ranksPerHost, ncclDataType_t datatype, ncclRedOp_t op);

// Whether RCCL_HOST_HIER_ALLREDUCE and the operands allow host-hierarchical
// AllReduce. registeredBuffers: the send or receive buffer is in a registered
// window, which auto mode leaves to the paths that use it. The caller also checks
// the dispatch context: group depth 0, no graph capture, no enqueue-only mode.
bool rcclHostHierAllReduceEligible(struct ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                                   bool registeredBuffers);

// ReduceScatter within the host, AllReduce across hosts, then AllGather within
// the host: in one fused DDA kernel where RCCL_HOST_HIER_ALLREDUCE_FUSED selects
// it, otherwise as three collectives on the child comms. Needs
// rcclHostHierEnsure to have succeeded.
ncclResult_t rcclHostHierAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclRedOp_t op, struct ncclComm* comm, cudaStream_t stream);

#endif
