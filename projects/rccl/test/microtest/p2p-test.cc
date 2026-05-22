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

#include "fakes/p2p_fakes.h"

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

// ---------------------------------------------------------------------------
// ipcRegisterBuffer: collective reuse, devPeerRmtAddrs missing -> enters the
// strong-stream allocation block (lines 1004-1015 in the hipified p2p.cc).
//
// Same shape as CollectiveReuseReturnsDevicePeerAddrTable, but deliberately
// leave regRecord.regIpcAddrs.devPeerRmtAddrs == nullptr so the
// `if (devPeerRmtAddrs == NULL || needUpdate)` branch goes True. Our existing
// stubs would normally let the block complete "successfully" (returning a
// null hipStream_t), but the very next call -- ncclCudaCallocAsync -- is a
// header-only template that hits the real HIP runtime and there is no GPU
// here, so it would fail mid-block with confusing diagnostics.
//
// Instead we promote ncclStrongStreamAcquire to a controllable seam (see
// fakes/p2p_fakes.h) and have the test install a hook that:
//   (a) records that the block was entered (proves coverage of line 1004's
//       True side and line 1006), and
//   (b) returns ncclSystemError on the first call, so control flows cleanly
//       through NCCLCHECKGOTO into the `fail:` epilogue (lines 1028-1034).
//
// Net coverage gain: line 1004 True side, line 1006 strong-stream call site,
// the entire `fail:` block including the `if (newInfo) free(newInfo)` branch
// at line 1032, and output-zeroing semantics on the failure path.
// ---------------------------------------------------------------------------
class IpcRegisterBufferFixture : public ::testing::Test {
protected:
    void TearDown() override { ResetP2pFakes(); }
};

TEST_F(IpcRegisterBufferFixture, CollectiveReuseEntersStrongStreamBlockWhenDevTableMissing)
{
    constexpr int kPeerRank      = 1;
    constexpr int kPeerLocalRank = 1;
    constexpr uintptr_t kBegAddr     = 0x30000;
    constexpr uintptr_t kBuffOffset  = 0x100;
    constexpr uintptr_t kRmtRegAddr  = 0xfeedface00ull;

    // Comm + sharedRes: the strong-stream calls dereference
    // comm->sharedRes->{hostStream,deviceStream,scratchEvent}. The hook
    // doesn't actually read them, but the address-taking in the call site
    // (`&comm->sharedRes->hostStream`) requires sharedRes to be non-null.
    ncclSharedResources sharedRes{};
    ncclComm comm{};
    comm.sharedRes = &sharedRes;
    int rankToLocalRank[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    rankToLocalRank[kPeerRank] = kPeerLocalRank;
    comm.rankToLocalRank = rankToLocalRank;
    comm.localRanks      = NCCL_MAX_LOCAL_RANKS;

    // Reuse path: ipcInfos[peer] is pre-populated, hostPeerRmtAddrs is set,
    // but devPeerRmtAddrs is intentionally left null -> triggers line 1004.
    uintptr_t hostPeerRmtAddrs[NCCL_MAX_LOCAL_RANKS] = {0};
    hostPeerRmtAddrs[kPeerLocalRank] = kRmtRegAddr;

    ncclIpcRegInfo existingInfo{};
    existingInfo.peerRank             = kPeerRank;
    existingInfo.impInfo.rmtRegAddr   = reinterpret_cast<void*>(kRmtRegAddr);
    existingInfo.impInfo.legacyIpcCap = false;

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x4000;
    regRecord.ipcInfos[kPeerLocalRank]      = &existingInfo;
    regRecord.regIpcAddrs.hostPeerRmtAddrs  = hostPeerRmtAddrs;
    regRecord.regIpcAddrs.devPeerRmtAddrs   = nullptr;        // <-- key

    // Spy + failure injector. Counts entries to the strong-stream block,
    // and on the first call returns an error so NCCLCHECKGOTO routes us to
    // `fail:` before we hit the header-template ncclCudaCallocAsync that
    // would otherwise try to run real HIP.
    int acquireCalls = 0;
    g_strongStreamAcquire = [&](struct ncclCudaGraph,
                                struct ncclStrongStream*,
                                bool,
                                hipStream_t* stream) -> ncclResult_t {
        ++acquireCalls;
        if (stream) *stream = nullptr;
        return ncclSystemError;
    };

    int       peerRanks[1]  = {kPeerRank};
    int       regBufFlag    = 0xdead;
    uintptr_t offsetOut     = 0xdead;
    uintptr_t* peerRmtAddrs = reinterpret_cast<uintptr_t*>(0xdead);
    bool      isLegacyIpc   = true;

    ncclResult_t r = ipcRegisterBuffer(
        /*comm=*/        &comm,
        /*userbuff=*/    reinterpret_cast<const void*>(kBegAddr + kBuffOffset),
        /*buffSize=*/    1024,
        /*peerRanks=*/   peerRanks,
        /*nPeers=*/      1,
        /*type=*/        NCCL_IPC_COLLECTIVE,
        /*regRecord=*/   &regRecord,
        /*regBufFlag=*/  &regBufFlag,
        /*offsetOut=*/   &offsetOut,
        /*peerRmtAddrsOut=*/ &peerRmtAddrs,
        /*isLegacyIpc=*/ &isLegacyIpc);

    // The strong-stream block was entered -- proves line 1004 True side and
    // line 1006 are now covered.
    EXPECT_EQ(acquireCalls, 1);

    // Error propagated out of the NCCLCHECKGOTO at line 1006.
    EXPECT_EQ(r, ncclSystemError);

    // `fail:` epilogue zeroed the OUT params (lines 1029-1031).
    EXPECT_EQ(regBufFlag,    0);
    EXPECT_EQ(offsetOut,     0u);
    EXPECT_EQ(peerRmtAddrs,  nullptr);
}

// ---------------------------------------------------------------------------
// ipcRegisterBuffer: fresh-registration entry, fails on the first HIP call.
//
// All previous tests stayed inside the reuse branch (ipcInfos[peer] != null)
// or skipped the per-peer loop entirely. This test crosses into the
// fresh-registration `else` block at line 905, which today has zero branch
// coverage:
//
//   - Build a regRecord with ipcInfos[peerLocalRank] == nullptr.
//   - With baseAddr starting NULL (line 887) and no prior iteration, the
//     `if (baseAddr == NULL)` branch at line 910 goes True and we hit the
//     CUCHECKGOTO(hipMemGetAddressRange(...)) at line 911.
//   - hipMemGetAddressRange is a real HIP runtime call, NOT a PFN seam --
//     it's resolved through hip::host at link time and called directly
//     here. We pass a bogus userbuff (0xBADADDR) so it fails with a HIP
//     error regardless of whether the host has a GPU; CUCHECKGOTO turns
//     that into ncclUnhandledCudaError and jumps to `fail:`.
//
// What this newly covers:
//   - Branch 894 False is already covered; True (entering the regRecord
//     block) was already covered too -- but every prior True path went
//     through the reuse arm. This is the first test where the False side
//     of branch 900 (`ipcInfos[peerLocalRank]` is null) is taken.
//   - Line 910 branch True (first iteration of the loop, baseAddr is null).
//   - Line 911 call site execution.
//   - The CUCHECKGOTO failure edge through `fail:` from the fresh-reg arm
//     (the prior fail: coverage came in via the strong-stream block, a
//     different goto site).
// ---------------------------------------------------------------------------
TEST(IpcRegisterBuffer, FreshRegistrationFailsOnAddressRangeLookup)
{
    constexpr int kPeerRank      = 4;
    constexpr int kPeerLocalRank = 3;
    constexpr uintptr_t kBegAddr    = 0x50000;
    constexpr uintptr_t kBuffOffset = 0x10;
    // Deliberately bogus address: we want hipMemGetAddressRange to fail,
    // not crash. Any non-registered host/device pointer works -- HIP
    // returns hipErrorInvalidValue (or similar) regardless of GPU presence.
    constexpr uintptr_t kBogusUserbuff = 0xBADADD0ull;

    ncclComm comm{};
    int rankToLocalRank[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    rankToLocalRank[kPeerRank] = kPeerLocalRank;
    comm.rankToLocalRank = rankToLocalRank;
    comm.localRanks      = NCCL_MAX_LOCAL_RANKS;

    // Fresh-registration path: ipcInfos[peerLocalRank] left NULL so the
    // `else` at line 905 is taken.
    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    // ipcInfos[kPeerLocalRank] is NULL by virtue of the {} initializer.

    int       peerRanks[1]  = {kPeerRank};
    int       regBufFlag    = 0xdead;
    uintptr_t offsetOut     = 0xdead;
    uintptr_t* peerRmtAddrs = reinterpret_cast<uintptr_t*>(0xdead);
    bool      isLegacyIpc   = true;

    ncclResult_t r = ipcRegisterBuffer(
        /*comm=*/        &comm,
        /*userbuff=*/    reinterpret_cast<const void*>(kBogusUserbuff + kBuffOffset),
        /*buffSize=*/    256,
        /*peerRanks=*/   peerRanks,
        /*nPeers=*/      1,
        /*type=*/        NCCL_IPC_SENDRECV,
        /*regRecord=*/   &regRecord,
        /*regBufFlag=*/  &regBufFlag,
        /*offsetOut=*/   &offsetOut,
        /*peerRmtAddrsOut=*/ &peerRmtAddrs,
        /*isLegacyIpc=*/ &isLegacyIpc);

    // CUCHECKGOTO turns any non-hipSuccess into ncclUnhandledCudaError.
    EXPECT_EQ(r, ncclUnhandledCudaError);

    // `fail:` zeroed the OUT params (lines 1029-1031).
    EXPECT_EQ(regBufFlag,    0);
    EXPECT_EQ(offsetOut,     0u);
    EXPECT_EQ(peerRmtAddrs,  nullptr);

    // Prologue cleared isLegacyIpc (line 893) before any failure path.
    EXPECT_FALSE(isLegacyIpc);
}
