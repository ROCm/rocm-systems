/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-side mock communicator for arch-gated logic that reads the architecture
// from the topology (comm->topo->nodes[GPU].nodes[0].gpu.gcn) rather than from
// comm->archName. Promoted out of RcclWrapTests.cpp so other fixtures can reuse it.

#ifndef RCCL_TEST_MOCK_COMM_HPP
#define RCCL_TEST_MOCK_COMM_HPP

#include <rccl/rccl.h>

#include <cstring>

#include "comm.h"
#include "graph/topo.h"
#include "rccl_common.h"

namespace RcclUnitTesting
{

// Caller owns mockTopo and mockGpuNode. mockComm is heap allocated and released
// by CleanupMockComm.
inline void CreateMockComm(
    ncclComm_t&            mockComm,
    struct ncclTopoSystem& mockTopo,
    struct ncclTopoNode&   mockGpuNode,
    const char*            arch,
    int                    nRanks
)
{
    // Allocate memory for the communicator
    mockComm = new ncclComm();
    memset(mockComm, 0, sizeof(ncclComm));

    // Initialize basic communicator fields
    mockComm->nRanks = nRanks;
    mockComm->nNodes = 1; // Default to single node for P2P tests
    mockComm->rank   = 0; // Default rank

    mockComm->pxnDisable      = RCCL_VALUE_UNSET;
    mockComm->p2pNetChunkSize = RCCL_VALUE_UNSET;

    // Initialize topology
    memset(&mockTopo, 0, sizeof(mockTopo));
    mockComm->topo = &mockTopo;

    // Initialize GPU node
    mockTopo.nodes[GPU].count = 1;
    memset(&mockGpuNode, 0, sizeof(mockGpuNode));

    // Set GPU architecture
    strncpy(mockGpuNode.gpu.gcn, arch, sizeof(mockGpuNode.gpu.gcn) - 1);
    mockGpuNode.gpu.gcn[sizeof(mockGpuNode.gpu.gcn) - 1] = '\0';

    // Copy the node into the topology array
    mockTopo.nodes[GPU].nodes[0] = mockGpuNode;

    // Initialize other required fields for tests
    memset(mockComm->minMaxLLRange, 0, sizeof(mockComm->minMaxLLRange));
}

// Helper function to cleanup mock communicator
inline void CleanupMockComm(ncclComm_t& mockComm)
{
    if(mockComm)
    {
        delete mockComm;
        mockComm = nullptr;
    }
}

// Attach shared resources the way init.cc does for a top-level communicator
// (ncclCalloc + owner = comm). Needed by anything that dereferences
// comm->sharedRes, such as ncclTopoComputeP2pChannels. Caller owns the storage.
inline void AttachMockSharedRes(ncclComm_t mockComm, struct ncclSharedResources& sharedRes)
{
    memset(&sharedRes, 0, sizeof(sharedRes));
    sharedRes.owner     = mockComm;
    sharedRes.tpNRanks  = mockComm->nRanks;
    mockComm->sharedRes = &sharedRes;
}

// Make the comm look multi-node. topo->nRanks is the total rank count the per-arch
// multi-node caps in ncclTopoComputeP2pChannels match on.
inline void SetMockNodes(ncclComm_t mockComm, int nNodes, int topoNRanks)
{
    mockComm->nNodes       = nNodes;
    mockComm->topo->nRanks = topoNRanks;
}

} // namespace RcclUnitTesting

#endif // RCCL_TEST_MOCK_COMM_HPP
