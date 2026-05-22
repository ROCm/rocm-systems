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

// ---------------------------------------------------------------------------
// ipcRegisterBuffer: reuse path for SENDRECV.
//
// Pre-populate regRecord->ipcInfos[peerLocalRank] so the per-peer loop hits
// the "we already have IPC info for peerLocalRank" branch -- no driver, no
// proxy, no device-stream work. For type == NCCL_IPC_SENDRECV the post-loop
// block is also trivial: it just returns the host-side remote address as
// *peerRmtAddrsOut. This pins down the cheapest path through the function
// that still actually populates all the outputs.
// ---------------------------------------------------------------------------
TEST(IpcRegisterBuffer, SendrecvReusesExistingIpcInfo)
{
    constexpr int kPeerRank      = 3;
    constexpr int kPeerLocalRank = 2;
    constexpr uintptr_t kBegAddr     = 0x10000;
    constexpr uintptr_t kBuffOffset  = 0x40;        // userbuff = begAddr + 0x40
    constexpr uintptr_t kRmtRegAddr  = 0xdeadbeef00ull;

    // Build a comm with enough state for the reuse path:
    //   - rankToLocalRank[kPeerRank] -> kPeerLocalRank
    ncclComm comm{};
    int rankToLocalRank[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    rankToLocalRank[kPeerRank] = kPeerLocalRank;
    comm.rankToLocalRank = rankToLocalRank;

    // Pre-built remote-address table (host side). For NCCL_IPC_SENDRECV the
    // function returns hostPeerRmtAddrs[peerLocalRank] CAST to uintptr_t*,
    // not the table itself.
    uintptr_t hostPeerRmtAddrs[NCCL_MAX_LOCAL_RANKS] = {0};
    hostPeerRmtAddrs[kPeerLocalRank] = kRmtRegAddr;

    // Pre-built per-peer IPC info -- the thing the reuse branch keys off.
    ncclIpcRegInfo existingInfo{};
    existingInfo.peerRank             = kPeerRank;
    existingInfo.impInfo.rmtRegAddr   = reinterpret_cast<void*>(kRmtRegAddr);
    existingInfo.impInfo.legacyIpcCap = true;       // should surface in isLegacyIpc

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    regRecord.ipcInfos[kPeerLocalRank]    = &existingInfo;
    regRecord.regIpcAddrs.hostPeerRmtAddrs = hostPeerRmtAddrs;

    int       peerRanks[1]  = {kPeerRank};
    int       regBufFlag    = 0xdead;
    uintptr_t offsetOut     = 0xdead;
    uintptr_t* peerRmtAddrs = reinterpret_cast<uintptr_t*>(0xdead);
    bool      isLegacyIpc   = false;

    ncclResult_t r = ipcRegisterBuffer(
        /*comm=*/        &comm,
        /*userbuff=*/    reinterpret_cast<const void*>(kBegAddr + kBuffOffset),
        /*buffSize=*/    256,
        /*peerRanks=*/   peerRanks,
        /*nPeers=*/      1,
        /*type=*/        NCCL_IPC_SENDRECV,
        /*regRecord=*/   &regRecord,
        /*regBufFlag=*/  &regBufFlag,
        /*offsetOut=*/   &offsetOut,
        /*peerRmtAddrsOut=*/ &peerRmtAddrs,
        /*isLegacyIpc=*/ &isLegacyIpc);

    EXPECT_EQ(r,            ncclSuccess);
    EXPECT_EQ(regBufFlag,   1);
    EXPECT_EQ(offsetOut,    kBuffOffset);                      // userbuff - begAddr
    EXPECT_EQ(reinterpret_cast<uintptr_t>(peerRmtAddrs),
              kRmtRegAddr);                                    // raw remote addr, not a pointer
    EXPECT_TRUE(isLegacyIpc);                                  // propagated from existingInfo
}

// ---------------------------------------------------------------------------
// ipcRegisterBuffer: reuse path for COLLECTIVE.
//
// Same per-peer reuse branch as test 2, but type == NCCL_IPC_COLLECTIVE
// changes the post-loop output marshalling:
//   - returns the *device-side* peer-address table (regIpcAddrs.devPeerRmtAddrs)
//     instead of a single host-side address.
//   - would normally allocate/refresh devPeerRmtAddrs via ncclCudaCallocAsync
//     + ncclCudaMemcpyAsync, but those are only triggered when
//     devPeerRmtAddrs == nullptr OR needUpdate == true. The reuse branch
//     never flips needUpdate (that's a new-registration thing), so by
//     pre-populating devPeerRmtAddrs we keep the whole strong-stream block
//     skipped. Our fakes for those functions return ncclSuccess anyway, so
//     even if it were entered the test would still pass -- but staying out
//     of it keeps the test honest about which code path it covers.
// ---------------------------------------------------------------------------
TEST(IpcRegisterBuffer, CollectiveReuseReturnsDevicePeerAddrTable)
{
    constexpr int kPeerRank      = 1;
    constexpr int kPeerLocalRank = 1;
    constexpr uintptr_t kBegAddr     = 0x20000;
    constexpr uintptr_t kBuffOffset  = 0x80;
    constexpr uintptr_t kRmtRegAddr  = 0xcafef00d00ull;

    ncclComm comm{};
    int rankToLocalRank[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    rankToLocalRank[kPeerRank] = kPeerLocalRank;
    comm.rankToLocalRank = rankToLocalRank;
    comm.localRanks      = NCCL_MAX_LOCAL_RANKS;   // unused on this path, set defensively

    // host + dev peer-addr tables both pre-populated -> needUpdate stays
    // false, devPeerRmtAddrs is non-null, so the strong-stream block is
    // entirely skipped.
    uintptr_t hostPeerRmtAddrs[NCCL_MAX_LOCAL_RANKS] = {0};
    uintptr_t devPeerRmtAddrs[NCCL_MAX_LOCAL_RANKS]  = {0};
    hostPeerRmtAddrs[kPeerLocalRank] = kRmtRegAddr;

    ncclIpcRegInfo existingInfo{};
    existingInfo.peerRank             = kPeerRank;
    existingInfo.impInfo.rmtRegAddr   = reinterpret_cast<void*>(kRmtRegAddr);
    existingInfo.impInfo.legacyIpcCap = false;

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x2000;
    regRecord.ipcInfos[kPeerLocalRank]      = &existingInfo;
    regRecord.regIpcAddrs.hostPeerRmtAddrs  = hostPeerRmtAddrs;
    regRecord.regIpcAddrs.devPeerRmtAddrs   = devPeerRmtAddrs;

    int       peerRanks[1]  = {kPeerRank};
    int       regBufFlag    = 0xdead;
    uintptr_t offsetOut     = 0xdead;
    uintptr_t* peerRmtAddrs = reinterpret_cast<uintptr_t*>(0xdead);
    bool      isLegacyIpc   = true;                            // start true to see it cleared

    ncclResult_t r = ipcRegisterBuffer(
        /*comm=*/        &comm,
        /*userbuff=*/    reinterpret_cast<const void*>(kBegAddr + kBuffOffset),
        /*buffSize=*/    512,
        /*peerRanks=*/   peerRanks,
        /*nPeers=*/      1,
        /*type=*/        NCCL_IPC_COLLECTIVE,
        /*regRecord=*/   &regRecord,
        /*regBufFlag=*/  &regBufFlag,
        /*offsetOut=*/   &offsetOut,
        /*peerRmtAddrsOut=*/ &peerRmtAddrs,
        /*isLegacyIpc=*/ &isLegacyIpc);

    EXPECT_EQ(r,           ncclSuccess);
    EXPECT_EQ(regBufFlag,  1);
    EXPECT_EQ(offsetOut,   kBuffOffset);
    EXPECT_EQ(peerRmtAddrs, devPeerRmtAddrs);                  // the table itself, not an addr
    EXPECT_FALSE(isLegacyIpc);
}
