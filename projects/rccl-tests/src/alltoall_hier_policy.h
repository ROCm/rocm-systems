/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Peer-to-DevComm split for the hierarchical alltoall (alltoall.cu, -D 5).
// Free of ncclDevComm and GPU intrinsics so HierAlltoAllKernel and the host
// unit test (alltoall_hier_host_test.cpp) run the same code.

#ifndef RCCL_TESTS_ALLTOALL_HIER_POLICY_H_
#define RCCL_TESTS_ALLTOALL_HIER_POLICY_H_

#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(ALLTOALL_HIER_HOST_ONLY)
#define ALLTOALL_HIER_HD __host__ __device__
#else
#define ALLTOALL_HIER_HD
#endif

namespace alltoall_hier {

// CTA 0 drives the network DevComm; CTAs [netCtas, nCtas) drive Anvil SDMA. Each
// network CTA's world barrier signals every peer, so a second network CTA would be
// a second poster on each peer QP, which --gin_resource_sharing 0 (THREAD) forbids.
constexpr int netCtas = 1;

// Peers outside the LSA team [lsaBase, lsaBase + lsaSize) use the network
// DevComm. With groupSize > 0, so do LSA peers outside this rank's group of
// groupSize consecutive LSA ranks. The relation is symmetric, so both ends of a
// pair use the same DevComm.
ALLTOALL_HIER_HD inline bool peerUsesNet(int peer, int rank, int lsaBase, int lsaSize, int groupSize) {
  if (peer < lsaBase || peer >= lsaBase + lsaSize) return true;
  return groupSize > 0 && (peer - lsaBase) / groupSize != (rank - lsaBase) / groupSize;
}

// Puts this rank receives on one DevComm, counting its put to itself. Senders
// pick a DevComm with peerUsesNet, so by symmetry this is the number of peers
// for which peerUsesNet matches net.
ALLTOALL_HIER_HD inline int peerCount(bool net, int rank, int nRanks, int lsaBase, int lsaSize, int groupSize) {
  int n = 0;
  for (int r = 0; r < nRanks; r++) n += peerUsesNet(r, rank, lsaBase, lsaSize, groupSize) == net;
  return n;
}

}  // namespace alltoall_hier

#endif  // RCCL_TESTS_ALLTOALL_HIER_POLICY_H_
