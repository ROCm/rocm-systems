/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for src/transport/p2p-scratch.cc.
//
// Unlike rccl-UnitTests, this binary does NOT link librccl.so. The goal is to
// eventually compile p2p-scratch.cc directly into the test binary so that
// internal symbols (e.g. the static ipcRegisterBuffer) can be exercised with
// link-time-substituted fakes for the HIP driver API and the NCCL proxy layer.
//
// For now this file just stands up the test target with a trivial smoke test;
// real coverage of ipcRegisterBuffer will be added once the stub layer for
// ncclProxyCallBlocking / ncclProxyConnect / the HIP driver seam is in place.

#include <gtest/gtest.h>

// Pull in the hipified copy of p2p.cc (cudaXxx -> hipXxx rewrites already
// applied by the hipify pass that runs as part of the main RCCL build).
// P2P_CC_PATH is defined by this target's CMakeLists.txt as a string
// literal pointing at ${PROJECT_BINARY_DIR}/hipify/src/transport/p2p.cc.
#include P2P_CC_PATH

TEST(P2pScratchSmoke, BinaryLinksAndRuns)
{
    EXPECT_EQ(1 + 1, 2);
}

// ---------------------------------------------------------------------------
// ipcRegisterBuffer: cheapest real path -- regRecord == nullptr.
//
// With no registration record to consult, the function should fall through
// the whole per-peer loop without touching the proxy or driver and just
// zero out the OUT params. This confirms the static symbol is reachable
// from the test TU (via the #include of p2p.cc above) without needing any
// of the fakes' "real" behaviour.
// ---------------------------------------------------------------------------
TEST(IpcRegisterBuffer, NullRegRecordIsNoOp)
{
    // Comm is never dereferenced on this path, but pass a non-null pointer
    // to be safe against future defensive null-checks.
    ncclComm dummyComm{};

    int       peerRanks[1]    = {0};
    int       regBufFlag      = 0xdead;
    uintptr_t offsetOut       = 0xdead;
    uintptr_t* peerRmtAddrs   = reinterpret_cast<uintptr_t*>(0xdead);
    bool      isLegacyIpc     = true;

    ncclResult_t r = ipcRegisterBuffer(
        /*comm=*/        &dummyComm,
        /*userbuff=*/    reinterpret_cast<const void*>(0x1000),
        /*buffSize=*/    4096,
        /*peerRanks=*/   peerRanks,
        /*nPeers=*/      1,
        /*type=*/        NCCL_IPC_COLLECTIVE,
        /*regRecord=*/   nullptr,
        /*regBufFlag=*/  &regBufFlag,
        /*offsetOut=*/   &offsetOut,
        /*peerRmtAddrsOut=*/ &peerRmtAddrs,
        /*isLegacyIpc=*/ &isLegacyIpc);

    EXPECT_EQ(r,             ncclSuccess);
    EXPECT_EQ(regBufFlag,    0);
    EXPECT_EQ(offsetOut,     0u);
    EXPECT_EQ(peerRmtAddrs,  nullptr);
    EXPECT_FALSE(isLegacyIpc);
}
