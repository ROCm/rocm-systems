/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file SuspendResumeMPITests.cpp
 * @brief Multi-process tests for ncclCommMemSuspend / ncclCommMemResume / ncclCommMemStats.
 *
 * The single-process unit-test suite in `test/mem_manager/MemManagerTests.cpp`
 * covers all early-return error paths, `ncclCommMemStats`, and the manager
 * state machine via fake / real-VMM entries that don't need a bootstrap.
 *
 * This file extends coverage to the bootstrap-coupled paths in
 * `ncclCommMemSuspend` / `ncclCommMemResume`:
 *   - Cross-rank `bootstrapBarrier(0xBEEF)` on entry to Suspend.
 *   - Cross-rank `bootstrapBarrier(0xBEEF)` after local restore in Resume.
 *   - `bootstrapAllGather` of per-rank export counts in Resume Step 3.
 *   - `bootstrapSend`/`bootstrapRecv(0xFEED)` handle-info exchange.
 *   - Final `bootstrapBarrier(0xCAFE)` after peer re-import.
 *   - The peer-import matching loop in Step 4 (both positive and negative).
 *   - `ncclProxyClientGetFdBlocking` round-trip used by Step 4 to convert
 *     a peer's CUmemGenericAllocationHandle into a POSIX FD via UDS.
 *   - End-to-end data preservation: rank 0 fills an `ncclMemOffload` buffer
 *     with a canary, both ranks Suspend (CPU backup) + Resume (restore from
 *     backup, re-export, peer re-import), rank 1 reads the canary back
 *     through its imported VA.
 *
 * The tests allocate VMM directly via raw HIP (hipMemCreate / Reserve / Map /
 * SetAccess) instead of going through ncclCuMemAlloc - the wrapper has its
 * own issues on ROCm 7.0/7.2 and the goal is to exercise mem_manager logic,
 * not the allocator. If the runtime really lacks raw VMM, hipMemCreate fails
 * with an explicit HIP error code instead of a misleading version skip.
 *
 * Two suites:
 *   - MemManagerAnyRanks - each rank runs the same local Suspend/Resume
 *     scenario in lockstep. The cross-rank bootstrapBarrier inside
 *     ncclCommMemSuspend / ncclCommMemResume keeps all ranks in sync, so
 *     every test in this suite passes on any np >= 1.
 *   - MemManagerTwoRank   - peer-coupled scenarios that need exactly 2 ranks
 *     (rank 0 exporter, rank 1 importer).
 *
 * Run (a single mpirun -np 2 invocation runs the entire file):
 *   mpirun -np 2 ./rccl-UnitTestsMPI --gtest_filter='MemManager*'
 *
 * Or, if you want to focus on one suite:
 *   mpirun -np 1 ./rccl-UnitTestsMPI --gtest_filter='MemManagerAnyRanks.*'
 *   mpirun -np 2 ./rccl-UnitTestsMPI --gtest_filter='MemManagerTwoRank.*'
 */

#include "MPITestBase.hpp"
#include "TestChecks.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <string>
#include <unistd.h>
#include <vector>

#ifdef MPI_TESTS_ENABLED

#include "bootstrap.h"
#include "comm.h"
#include "mem_manager.h"
#include "proxy.h"

using namespace MPITestConstants;

// ---------------------------------------------------------------------------
// VMM helpers
//
// We talk to the HIP driver directly (hipMemCreate / Reserve / Map / SetAccess)
// instead of going through ncclCuMemAlloc. The wrapper has rollback paths and
// a ROCm-2550 zero-fill memset that hide the actual VMM call site, and it
// segfaults inside cuMemCreate on ROCm 7.0/7.2 even when raw VMM works.
// Same approach the standalone unit test takes (Track_RealVmm_PosixFd).
// ---------------------------------------------------------------------------

namespace
{
struct VmmAlloc
{
    void*                           ptr    = nullptr;
    hipDeviceptr_t                  pdev   = 0;
    size_t                          size   = 0;
    hipMemGenericAllocationHandle_t handle = 0;
};

void allocateVmmPosixFd(int dev, size_t requestedSize, VmmAlloc* out)
{
    ASSERT_NE(out, nullptr);

    hipMemAllocationProp prop            = {};
    prop.type                            = hipMemAllocationTypePinned;
    prop.location.type                   = hipMemLocationTypeDevice;
    prop.location.id                     = dev;
    prop.requestedHandleTypes            = hipMemHandleTypePosixFileDescriptor;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    size_t granularity = 0;
    ASSERT_EQ(hipMemGetAllocationGranularity(&granularity, &prop,
                                             hipMemAllocationGranularityMinimum),
              hipSuccess);
    ASSERT_GT(granularity, 0u);
    size_t size = ((requestedSize + granularity - 1) / granularity) * granularity;

    hipMemGenericAllocationHandle_t handle = 0;
    ASSERT_EQ(hipMemCreate(&handle, size, &prop, 0), hipSuccess);

    hipDeviceptr_t pdev = 0;
    ASSERT_EQ(hipMemAddressReserve(&pdev, size, granularity, 0, 0), hipSuccess);
    ASSERT_EQ(hipMemMap(pdev, size, 0, handle, 0), hipSuccess);

    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type    = hipMemLocationTypeDevice;
    accessDesc.location.id      = dev;
    accessDesc.flags            = hipMemAccessFlagsProtReadWrite;
    ASSERT_EQ(hipMemSetAccess(pdev, size, &accessDesc, 1), hipSuccess);

    out->ptr    = reinterpret_cast<void*>(pdev);
    out->pdev   = pdev;
    out->size   = size;
    out->handle = handle;
}

// Reserve only (no physical handle / no map). Used to set up a "released" VA
// region as if Suspend had already torn the physical backing down. Lets a test
// drive Resume's restore path without needing a Suspend/Resume roundtrip.
void reserveVaOnly(int dev, size_t requestedSize, VmmAlloc* out)
{
    ASSERT_NE(out, nullptr);

    hipMemAllocationProp prop            = {};
    prop.type                            = hipMemAllocationTypePinned;
    prop.location.type                   = hipMemLocationTypeDevice;
    prop.location.id                     = dev;
    prop.requestedHandleTypes            = hipMemHandleTypePosixFileDescriptor;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    size_t granularity = 0;
    ASSERT_EQ(hipMemGetAllocationGranularity(&granularity, &prop,
                                             hipMemAllocationGranularityMinimum),
              hipSuccess);
    size_t size = ((requestedSize + granularity - 1) / granularity) * granularity;

    hipDeviceptr_t pdev = 0;
    ASSERT_EQ(hipMemAddressReserve(&pdev, size, granularity, 0, 0), hipSuccess);

    out->ptr    = reinterpret_cast<void*>(pdev);
    out->pdev   = pdev;
    out->size   = size;
    out->handle = 0;
}

// Tear down whatever Resume / the test left mapped at out->pdev. Safe whether
// the entry is in the Active state (mapped) or Released state (unmapped) -
// hipMemUnmap returns an error on a reserved-but-unmapped VA which we ignore.
void releaseVmm(VmmAlloc* a)
{
    if (a == nullptr || a->pdev == 0) return;
    (void)hipMemUnmap(a->pdev, a->size);
    if (a->handle != 0) (void)hipMemRelease(a->handle);
    (void)hipMemAddressFree(a->pdev, a->size);
    a->pdev   = 0;
    a->handle = 0;
    a->ptr    = nullptr;
    a->size   = 0;
}

// Grant local-device RW access to a previously reserved + mapped VA range.
// Used after hipMemMap on an imported handle to make the VA usable.
void applyDeviceRwAccess(int dev, hipDeviceptr_t pdev, size_t size)
{
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type    = hipMemLocationTypeDevice;
    accessDesc.location.id      = dev;
    accessDesc.flags            = hipMemAccessFlagsProtReadWrite;
    ASSERT_EQ(hipMemSetAccess(pdev, size, &accessDesc, 1), hipSuccess);
}

// Pour `pattern` over the entire VA on the device, then DToH + verify every
// byte equals `pattern`. Uses bulk std::all_of so failures don't spam gtest
// with millions of EXPECT_EQ lines on big buffers. `what` is taken by const-
// reference so callers can pass either a string literal or a `std::to_string`-
// composed message without forcing a `.c_str()` everywhere.
void verifyCanaryReadback(void* ptr, size_t size, uint8_t pattern, const std::string& what)
{
    std::vector<uint8_t> host(size, static_cast<uint8_t>(~pattern));
    ASSERT_EQ(hipMemcpy(host.data(), ptr, size, hipMemcpyDeviceToHost), hipSuccess) << what;
    bool ok = std::all_of(host.begin(), host.end(),
                          [pattern](uint8_t b) { return b == pattern; });
    EXPECT_TRUE(ok) << what << ": canary 0x" << std::hex << static_cast<int>(pattern)
                    << " not preserved";
}

// Cleanup pattern shared by every test that does Track + later Untrack on a
// VMM entry that may have gone through Suspend/Resume:
//   - Resume installs a NEW handle into entry->handle (the original one we
//     created via hipMemCreate is long gone after Suspend's hipMemRelease).
//   - ncclMemUntrack drops the linked-list entry but does NOT unmap or release
//     anything HIP-side.
// So we snapshot the live handle out of the entry FIRST, then Untrack, then
// releaseVmm() with that handle.
void untrackAndRelease(struct ncclComm* comm, void* ptr, VmmAlloc* a)
{
    ncclMemManager* mgr = comm->memManager;
    ASSERT_NE(mgr, nullptr);
    ncclDynMemEntry* entry = mgr->entries;
    while (entry != nullptr && entry->ptr != ptr) entry = entry->next;
    if (entry != nullptr) a->handle = entry->handle;
    ASSERT_EQ(ncclMemUntrack(mgr, ptr, a->size), ncclSuccess);
    releaseVmm(a);
}

// ---------------------------------------------------------------------------
// Peer export/import helpers - shared by the multi-rank tests.
//
// PeerMeta is what the owner ships over bootstrapSend so the importer can
// reconstruct the matching ncclMemTrackImportFromPeer call: handle value
// (round-tripped through the proxy GetFd UDS for FD conversion), VA address
// (matched verbatim by Resume Step 4's matching loop), aligned size, and the
// owner's cudaDev.
// ---------------------------------------------------------------------------

struct PeerMeta
{
    uint64_t handleVal;  // owner's hipMemGenericAllocationHandle_t value
    uint64_t ownerPtr;   // owner's VA (uintptr_t) - matched verbatim in Step 4
    uint64_t size;       // owner's aligned size  - matched verbatim
    int      ownerDev;   // owner's cudaDev       - stored in entry.desc.imported
};

// Owner side: Track the buffer, MarkExportToPeer, and ship the metadata over
// bootstrapSend. The owner keeps `vmm` alive through Suspend/Resume.
// Each (peerRank, tag) pair must be unique across concurrent calls.
void exportAndShipMeta(struct ncclComm* comm, const VmmAlloc& vmm, int peerRank,
                       ncclMemType_t memType, int tag)
{
    ASSERT_EQ(ncclMemTrack(comm->memManager, vmm.ptr, vmm.size, vmm.handle,
                           hipMemHandleTypePosixFileDescriptor, memType),
              ncclSuccess);
    ASSERT_EQ(ncclDynMemMarkExportToPeer(comm->memManager, vmm.ptr, peerRank), ncclSuccess);

    PeerMeta meta{};
    meta.handleVal = (uint64_t)(uintptr_t)vmm.handle;
    meta.ownerPtr  = reinterpret_cast<uintptr_t>(vmm.ptr);
    meta.size      = static_cast<uint64_t>(vmm.size);
    meta.ownerDev  = comm->cudaDev;
    ASSERT_EQ(bootstrapSend(comm->bootstrap, peerRank, tag, &meta, sizeof(meta)),
              ncclSuccess);
}

// Importer side: bootstrapRecv the metadata, ask owner's proxy for a POSIX
// FD, import + map + setAccess into a fresh local VA, then Track as imported.
// Fills `out` with VA + size + the imported handle so untrackAndRelease() can
// free everything later.
void recvAndImportPeerBuffer(struct ncclComm* comm, int ownerRank, ncclMemType_t memType,
                             int tag, VmmAlloc* out)
{
    PeerMeta meta{};
    ASSERT_EQ(bootstrapRecv(comm->bootstrap, ownerRank, tag, &meta, sizeof(meta)),
              ncclSuccess);
    ASSERT_GT(meta.size, 0u);
    ASSERT_NE(meta.ownerPtr, 0u);

    int                             fd       = -1;
    hipMemGenericAllocationHandle_t peerHval =
        (hipMemGenericAllocationHandle_t)(uintptr_t)meta.handleVal;
    ASSERT_EQ(ncclProxyClientGetFdBlocking(comm, ownerRank, &peerHval, &fd), ncclSuccess);
    ASSERT_GE(fd, 0) << "proxy GetFd from rank " << ownerRank << " returned invalid FD";

    hipMemGenericAllocationHandle_t imported = 0;
    ASSERT_EQ(hipMemImportFromShareableHandle(&imported, (void*)(uintptr_t)fd,
                                              hipMemHandleTypePosixFileDescriptor),
              hipSuccess);
    close(fd);

    reserveVaOnly(comm->cudaDev, static_cast<size_t>(meta.size), out);
    ASSERT_EQ(hipMemMap(out->pdev, out->size, 0, imported, 0), hipSuccess);
    applyDeviceRwAccess(comm->cudaDev, out->pdev, out->size);
    out->handle = imported;

    ASSERT_EQ(ncclMemTrackImportFromPeer(comm->memManager, out->ptr, out->size, imported,
                                         hipMemHandleTypePosixFileDescriptor, memType,
                                         ownerRank, meta.ownerDev,
                                         reinterpret_cast<void*>(meta.ownerPtr)),
              ncclSuccess);
}
} // namespace

// ===========================================================================
// AnyRanks suite (np >= 1) - local-only Suspend/Resume scenarios.
//
// Every test here only touches its own rank's manager (no cross-rank exports).
// The bootstrapBarrier(0xBEEF/0xCAFE) inside ncclCommMemSuspend/Resume keeps
// all ranks in sync, AllGather/Send-Recv in Resume Step 3 short-circuits when
// nobody has exports, so the same body works identically on np=1, np=2, np=N.
// ===========================================================================

class MemManagerAnyRanks : public MPITestBase
{
protected:
    void SetUp() override
    {
        MPITestBase::SetUp();
        if (!validateTestPrerequisites(/*min_processes=*/1, kNoProcessLimit)) {
            GTEST_SKIP() << "AnyRanks suite needs at least 1 rank";
        }
        ASSERT_EQ(createTestCommunicator(), ncclSuccess);
    }
};

TEST_F(MemManagerAnyRanks, SuspendResume_LocalScratch)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm, nullptr);
    ASSERT_NE(comm->memManager, nullptr);

    VmmAlloc a{};
    allocateVmmPosixFd(comm->cudaDev, 1 << 20, &a);

    ASSERT_EQ(ncclMemTrack(comm->memManager, a.ptr, a.size, a.handle,
                           hipMemHandleTypePosixFileDescriptor, ncclMemScratch),
              ncclSuccess);

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->released, 1);

    // The single tracked entry is the one we just added.
    ncclDynMemEntry* entry = comm->memManager->entries;
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->state, ncclDynMemStateReleased);
    EXPECT_EQ(static_cast<void*>(entry->handle), nullptr);

    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->released, 0);
    EXPECT_EQ(entry->state, ncclDynMemStateActive);
    EXPECT_NE(static_cast<void*>(entry->handle), nullptr);

    // Verify the VA is mapped again - a roundtrip memset/D2H must succeed.
    ASSERT_EQ(hipMemset(a.ptr, 0xAA, a.size), hipSuccess);
    std::vector<uint8_t> host(a.size, 0);
    ASSERT_EQ(hipMemcpy(host.data(), a.ptr, a.size, hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(host[0], 0xAA);
    EXPECT_EQ(host[a.size - 1], 0xAA);

    ASSERT_EQ(ncclMemUntrack(comm->memManager, a.ptr, a.size), ncclSuccess);
    releaseVmm(&a);
}

TEST_F(MemManagerAnyRanks, SuspendResume_LocalOffload_DataPreserved)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    VmmAlloc a{};
    allocateVmmPosixFd(comm->cudaDev, 64 * 1024, &a);

    ASSERT_EQ(ncclMemTrack(comm->memManager, a.ptr, a.size, a.handle,
                           hipMemHandleTypePosixFileDescriptor, ncclMemOffload),
              ncclSuccess);

    // Pre-fill device buffer with a recognisable pattern so we can verify that
    // Suspend's D2H copy and Resume's H2D restore round-trip the bytes.
    std::vector<uint8_t> pattern(a.size);
    for (size_t i = 0; i < a.size; i++) pattern[i] = static_cast<uint8_t>(i * 31u);
    ASSERT_EQ(hipMemcpy(a.ptr, pattern.data(), a.size, hipMemcpyHostToDevice), hipSuccess);

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);

    ncclDynMemEntry* entry = comm->memManager->entries;
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->state, ncclDynMemStateReleased);
    EXPECT_NE(entry->cpuBackup, nullptr);
    EXPECT_EQ(comm->memManager->cpuBackupUsage, a.size);

    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
    EXPECT_EQ(entry->state, ncclDynMemStateActive);
    EXPECT_EQ(entry->cpuBackup, nullptr);
    EXPECT_EQ(comm->memManager->cpuBackupUsage, 0u);

    std::vector<uint8_t> roundtrip(a.size, 0);
    ASSERT_EQ(hipMemcpy(roundtrip.data(), a.ptr, a.size, hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(std::memcmp(roundtrip.data(), pattern.data(), a.size), 0);

    ASSERT_EQ(ncclMemUntrack(comm->memManager, a.ptr, a.size), ncclSuccess);
    releaseVmm(&a);
}

TEST_F(MemManagerAnyRanks, SuspendResume_PersistUntouched)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    constexpr size_t persistSize = 8192;
    void* persistPtr             = reinterpret_cast<void*>(0xDEADBEEF00ULL);
    ASSERT_EQ(ncclMemTrack(comm->memManager, persistPtr, persistSize,
                           /*handle=*/0, hipMemHandleTypePosixFileDescriptor,
                           ncclMemPersist),
              ncclSuccess);

    uint64_t persistBefore = 0;
    ASSERT_EQ(ncclCommMemStats(comm, ncclStatGpuMemPersist, &persistBefore), ncclSuccess);
    EXPECT_EQ(persistBefore, persistSize);

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);

    uint64_t persistAfter = 0;
    ASSERT_EQ(ncclCommMemStats(comm, ncclStatGpuMemPersist, &persistAfter), ncclSuccess);
    EXPECT_EQ(persistAfter, persistSize);

    uint64_t suspended = 0;
    ASSERT_EQ(ncclCommMemStats(comm, ncclStatGpuMemSuspended, &suspended), ncclSuccess);
    EXPECT_EQ(suspended, 1u);

    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);

    ASSERT_EQ(ncclMemUntrack(comm->memManager, persistPtr, persistSize), ncclSuccess);
}

TEST_F(MemManagerAnyRanks, SuspendResume_DoubleCalls_Rejected)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    EXPECT_EQ(ncclCommMemSuspend(comm), ncclInvalidUsage);

    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
    EXPECT_EQ(ncclCommMemResume(comm), ncclInvalidUsage);
}

TEST_F(MemManagerAnyRanks, MemStats_FlipsAcrossSuspendResume)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    uint64_t suspended = ~0ULL;
    ASSERT_EQ(ncclCommMemStats(comm, ncclStatGpuMemSuspended, &suspended), ncclSuccess);
    EXPECT_EQ(suspended, 0u);

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    ASSERT_EQ(ncclCommMemStats(comm, ncclStatGpuMemSuspended, &suspended), ncclSuccess);
    EXPECT_EQ(suspended, 1u);

    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
    ASSERT_EQ(ncclCommMemStats(comm, ncclStatGpuMemSuspended, &suspended), ncclSuccess);
    EXPECT_EQ(suspended, 0u);
}

// ===========================================================================
// Two-rank suite (-np 2) - exercises bootstrap-coupled paths
// ===========================================================================

class MemManagerTwoRank : public MPITestBase
{
protected:
    void SetUp() override
    {
        MPITestBase::SetUp();
        if (!validateTestPrerequisites(/*min_processes=*/2, /*max_processes=*/2)) {
            GTEST_SKIP() << "Two-rank suite, run with mpirun -np 2";
        }
        ASSERT_EQ(createTestCommunicator(), ncclSuccess);
    }
};

// Both ranks own only local entries. No exports, no peer imports.
// Verifies:
//   - bootstrapBarrier(0xBEEF) on Suspend entry succeeds across ranks.
//   - bootstrapBarrier(0xBEEF) after Resume Step 1 succeeds.
//   - Resume Step 3 short-circuits when nobody has exports
//     (allCounts all zero, totalInfoCount == 0, no Send/Recv).
//   - Final bootstrapBarrier(0xCAFE) succeeds.
TEST_F(MemManagerTwoRank, SuspendResume_LocalOnly)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    VmmAlloc a{};
    allocateVmmPosixFd(comm->cudaDev, 1 << 20, &a);
    ASSERT_EQ(ncclMemTrack(comm->memManager, a.ptr, a.size, a.handle,
                           hipMemHandleTypePosixFileDescriptor, ncclMemScratch),
              ncclSuccess);

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->released, 1);
    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->released, 0);

    // VA must be live again on both ranks.
    ASSERT_EQ(hipMemset(a.ptr, 0x55, a.size), hipSuccess);

    ASSERT_EQ(ncclMemUntrack(comm->memManager, a.ptr, a.size), ncclSuccess);
    releaseVmm(&a);
}

// Inject a synthetic peer-imported entry on rank 1 whose ownerPtr has no
// matching local export on rank 0. Resume Step 4 must surface this as
// ncclSystemError on rank 1 (RCCL divergence from NCCL upstream silent skip)
// and leave the entry in Released state. Rank 0 has no entries -> ncclSuccess.
TEST_F(MemManagerTwoRank, Resume_PeerImport_NoMatch_ReturnsErrorEntryStaysReleased)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);
    int rank = comm->rank;

    // Rank 1 reserves a VA region (no physical backing) and registers it as
    // a peer import that points at a fake ownerPtr on rank 0. Rank 0 does
    // nothing - it'll have no exports to broadcast.
    VmmAlloc fakeImport{};
    if (rank == 1) {
        reserveVaOnly(comm->cudaDev, 64 * 1024, &fakeImport);
        void* fakeOwnerPtr = reinterpret_cast<void*>(0xFEEDFACE0000ULL);
        ASSERT_EQ(ncclMemTrackImportFromPeer(
                      comm->memManager, fakeImport.ptr, fakeImport.size,
                      /*handle=*/0, hipMemHandleTypePosixFileDescriptor,
                      ncclMemScratch, /*ownerRank=*/0, /*ownerDev=*/0, fakeOwnerPtr),
                  ncclSuccess);

        // Pretend Suspend already ran for this entry so Resume Step 4 considers
        // it a candidate for re-import (which is then expected to no-match).
        ncclDynMemEntry* entry = comm->memManager->entries;
        ASSERT_NE(entry, nullptr);
        entry->state = ncclDynMemStateReleased;
    }

    // Drive the full Suspend / Resume cycle. We expect:
    //   - Suspend on both ranks succeeds (rank 1's "released" entry is skipped
    //     in Pass 1 since state already == Released; rank 0 has no entries).
    //   - Resume on rank 0 succeeds (no peer-imports to re-map).
    //   - Resume on rank 1 returns ncclSystemError because the fake entry has
    //     no matching broadcast info; entry stays Released so caller can clean
    //     up via Destroy.
    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    ncclResult_t resumeRet = ncclCommMemResume(comm);

    if (rank == 0) {
        ASSERT_EQ(resumeRet, ncclSuccess) << "Rank 0 has no entries, Resume must succeed";
    } else {
        ASSERT_EQ(resumeRet, ncclSystemError)
            << "Rank 1 has a no-match peer-import; Resume must surface the error";

        ncclDynMemEntry* entry = comm->memManager->entries;
        ASSERT_NE(entry, nullptr);
        EXPECT_EQ(entry->state, ncclDynMemStateReleased)
            << "Resume must leave a no-match peer entry in Released state";

        // released flag must remain set so the manager is in a clearly
        // suspended state and the caller knows to retry or Destroy.
        EXPECT_EQ(comm->memManager->released, 1);

        // Cleanup: drop the fake entry and free the VA reservation.
        ASSERT_EQ(ncclMemUntrack(comm->memManager, fakeImport.ptr, fakeImport.size),
                  ncclSuccess);
        releaseVmm(&fakeImport);
    }
}

// End-to-end happy path of the Suspend/Resume P2P protocol with a real,
// mutual peer import driven through the proxy:
//
//   Setup (before Suspend):
//     rank 0:  hipMemCreate(POSIX_FD) -> hipMemMap -> hipMemSetAccess
//              -> ncclMemTrack(..., ncclMemOffload)
//              -> ncclDynMemMarkExportToPeer(peer=1)
//              -> hipMemset(canary)
//              -> bootstrapSend(meta = {handleVal, ownerPtr, size, ownerDev}, tag=0xC0FFEE)
//
//     rank 1:  bootstrapRecv(meta, tag=0xC0FFEE)
//              -> ncclProxyClientGetFdBlocking(rank 0, &handleVal, &fd)   <-- UDS round-trip
//              -> hipMemImportFromShareableHandle(fd) -> hipMemMap + SetAccess
//              -> ncclMemTrackImportFromPeer(..., ncclMemOffload, owner=rank 0)
//              -> (sanity) hipMemcpy DToH and verify canary on imported VA
//
//   Drive Suspend then Resume on both ranks. This exercises:
//     - Suspend Step 1: rank 1 unmaps the imported entry, releases its handle.
//     - Suspend Step 2: rank 0 copies the offload buffer to CPU backup,
//       closes the shareable FD, unmaps + releases the physical handle.
//     - Resume Step 1: rank 0 re-creates a fresh physical handle, re-maps to
//       the same VA, re-applies POSIX-FD path's empty same-process peer-ACL
//       loop (cross-process here, so no cuMemSetAccess), and restores data
//       from the CPU backup. rank 1 has no local entries to restore.
//     - Resume Step 3: AllGather counts (rank 0 reports 1 export, rank 1
//       reports 0), Send/Recv the new ncclDynMemP2pHandleInfo, both ranks
//       end up with allInfos[0] == rank 0's new handle.
//     - Resume Step 4 (positive matching): rank 1 finds the matching entry
//       (ownerRank/ownerPtr/size all match), calls ncclProxyClientGetFdBlocking
//       on the new handle, hipMemImportFromShareableHandle, ncclCuMemMapAndSetAccess
//       to the same imported VA. Entry transitions Released -> Active.
//     - Final 0xCAFE barrier completes.
//
//   Verify: rank 1 hipMemcpy DToH from imported VA -> canary preserved.
//
// Failure of any step short-circuits and asserts so we get a precise diagnosis
// instead of a silent stale-data success.
TEST_F(MemManagerTwoRank, SuspendResume_PeerExportImport_RoundtripCanary)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    constexpr size_t  kRequestedSize = 1u << 20;
    constexpr uint8_t kCanary        = 0xAB;
    constexpr int     kMetaTag       = 0xC0FFEE;

    if (comm->rank == 0) {
        VmmAlloc owner{};
        allocateVmmPosixFd(comm->cudaDev, kRequestedSize, &owner);
        ASSERT_EQ(hipMemset(owner.ptr, kCanary, owner.size), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        exportAndShipMeta(comm, owner, /*peerRank=*/1, ncclMemOffload, kMetaTag);

        ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
        EXPECT_EQ(comm->memManager->released, 1);
        ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
        EXPECT_EQ(comm->memManager->released, 0);

        untrackAndRelease(comm, owner.ptr, &owner);
        return;
    }

    ASSERT_EQ(comm->rank, 1);
    VmmAlloc imp{};
    recvAndImportPeerBuffer(comm, /*ownerRank=*/0, ncclMemOffload, kMetaTag, &imp);

    // Sanity-check the primary import path BEFORE Suspend/Resume - if this
    // fails, the GetFd / import / map chain is broken and any post-Resume
    // verification would be a stale-data success.
    verifyCanaryReadback(imp.ptr, imp.size, kCanary, "primary peer import");

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->released, 1);
    {
        ncclDynMemEntry* e = comm->memManager->entries;
        ASSERT_NE(e, nullptr);
        EXPECT_EQ(e->state, ncclDynMemStateReleased);
        EXPECT_EQ(static_cast<void*>(e->handle), nullptr)
            << "Suspend must release imported handle";
    }

    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->released, 0);

    ncclDynMemEntry* entry = comm->memManager->entries;
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->state, ncclDynMemStateActive)
        << "Resume Step 4 must transition the matched peer entry back to Active";
    EXPECT_NE(static_cast<void*>(entry->handle), nullptr)
        << "Resume must install the re-imported handle";

    verifyCanaryReadback(imp.ptr, imp.size, kCanary,
                         "post-Resume imported VA (rank 0's offload data)");

    untrackAndRelease(comm, imp.ptr, &imp);
}

// ===========================================================================
// Stress / coverage tests for AnyRanks (np >= 1)
// ===========================================================================

// A: empty manager - no entries tracked, but Suspend/Resume must still drive
// all the bootstrap barriers and toggle the released flag. Catches a corner
// case where Pass 1 / Pass 2 / Step 1 / Step 4 loops crash on a null entries
// pointer (manager->entries == nullptr from start).
TEST_F(MemManagerAnyRanks, EmptyManager_SuspendResume_NoOp)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);
    ASSERT_EQ(comm->memManager->entries, nullptr);
    ASSERT_EQ(comm->memManager->numEntries, 0);

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->released, 1);
    EXPECT_EQ(comm->memManager->entries, nullptr) << "Suspend must not invent entries";

    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->released, 0);
    EXPECT_EQ(comm->memManager->entries, nullptr) << "Resume must not invent entries";
    EXPECT_EQ(comm->memManager->cpuBackupUsage, 0u);
}

// B: Persist + Scratch + Offload tracked together. Suspend / Resume releases
// + restores Scratch and Offload but Persist is invisible to it (lives outside
// the linked list). Verifies stat invariants:
//   - Total = Persist + Scratch + Offload    (constant before and after)
//   - Persist                                 (constant)
//   - Suspend = Scratch + Offload            (constant - tracked, not freed)
//   - Suspended toggles 0 -> 1 -> 0
TEST_F(MemManagerAnyRanks, MixedMemTypes_StatsConsistent)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    constexpr size_t  kPersistReq  = 8 * 1024;
    constexpr size_t  kScratchReq  = 256 * 1024;
    constexpr size_t  kOffloadReq  = 128 * 1024;
    constexpr uint8_t kPersistByte = 0xC1;

    VmmAlloc persistVmm{};
    allocateVmmPosixFd(comm->cudaDev, kPersistReq, &persistVmm);
    ASSERT_EQ(hipMemset(persistVmm.ptr, kPersistByte, persistVmm.size), hipSuccess);
    ASSERT_EQ(ncclMemTrack(comm->memManager, persistVmm.ptr, persistVmm.size,
                           persistVmm.handle, hipMemHandleTypePosixFileDescriptor,
                           ncclMemPersist),
              ncclSuccess);

    VmmAlloc scratchVmm{};
    allocateVmmPosixFd(comm->cudaDev, kScratchReq, &scratchVmm);
    ASSERT_EQ(ncclMemTrack(comm->memManager, scratchVmm.ptr, scratchVmm.size,
                           scratchVmm.handle, hipMemHandleTypePosixFileDescriptor,
                           ncclMemScratch),
              ncclSuccess);

    VmmAlloc offloadVmm{};
    allocateVmmPosixFd(comm->cudaDev, kOffloadReq, &offloadVmm);
    ASSERT_EQ(ncclMemTrack(comm->memManager, offloadVmm.ptr, offloadVmm.size,
                           offloadVmm.handle, hipMemHandleTypePosixFileDescriptor,
                           ncclMemOffload),
              ncclSuccess);

    auto readStat = [&](ncclCommMemStat_t s) {
        uint64_t v = ~0ULL;
        EXPECT_EQ(ncclCommMemStats(comm, s, &v), ncclSuccess);
        return v;
    };

    const uint64_t expectedTotal   = persistVmm.size + scratchVmm.size + offloadVmm.size;
    const uint64_t expectedPersist = persistVmm.size;
    const uint64_t expectedDynamic = scratchVmm.size + offloadVmm.size;

    EXPECT_EQ(readStat(ncclStatGpuMemTotal),     expectedTotal);
    EXPECT_EQ(readStat(ncclStatGpuMemPersist),   expectedPersist);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspend),   expectedDynamic);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspended), 0u);

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);

    // Tracked sizes don't change across Suspend - the entries are still in the
    // linked list, just with state == Released. Persist counter is independent.
    EXPECT_EQ(readStat(ncclStatGpuMemTotal),     expectedTotal);
    EXPECT_EQ(readStat(ncclStatGpuMemPersist),   expectedPersist);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspend),   expectedDynamic);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspended), 1u);

    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
    EXPECT_EQ(readStat(ncclStatGpuMemTotal),     expectedTotal);
    EXPECT_EQ(readStat(ncclStatGpuMemPersist),   expectedPersist);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspend),   expectedDynamic);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspended), 0u);

    // Persist physical memory must still be readable with its canary intact.
    verifyCanaryReadback(persistVmm.ptr, persistVmm.size, kPersistByte,
                         "persist after Suspend/Resume");

    untrackAndRelease(comm, scratchVmm.ptr, &scratchVmm);
    untrackAndRelease(comm, offloadVmm.ptr, &offloadVmm);
    // Persist isn't in the linked list - Untrack only adjusts the counter.
    ASSERT_EQ(ncclMemUntrack(comm->memManager, persistVmm.ptr, persistVmm.size),
              ncclSuccess);
    releaseVmm(&persistVmm);

    EXPECT_EQ(readStat(ncclStatGpuMemTotal), 0u);
}

// C: multiple Suspend/Resume cycles on the same Offload buffer. Verifies that
// the CPU-backup -> GPU restore loop is idempotent across many cycles - no
// cpuBackup leaks, no handle leaks, canary preserved every iteration.
TEST_F(MemManagerAnyRanks, MultipleSuspendResumeCycles_DataPreserved)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    constexpr size_t  kSize    = 64 * 1024;
    constexpr int     kCycles  = 5;
    constexpr uint8_t kInitial = 0x5A;

    VmmAlloc a{};
    allocateVmmPosixFd(comm->cudaDev, kSize, &a);

    std::vector<uint8_t> seed(a.size);
    for (size_t i = 0; i < a.size; ++i) {
        seed[i] = static_cast<uint8_t>(kInitial ^ (i & 0xFFu));
    }
    ASSERT_EQ(hipMemcpy(a.ptr, seed.data(), a.size, hipMemcpyHostToDevice), hipSuccess);

    ASSERT_EQ(ncclMemTrack(comm->memManager, a.ptr, a.size, a.handle,
                           hipMemHandleTypePosixFileDescriptor, ncclMemOffload),
              ncclSuccess);

    for (int cycle = 0; cycle < kCycles; ++cycle) {
        SCOPED_TRACE("cycle=" + std::to_string(cycle));

        ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
        EXPECT_EQ(comm->memManager->cpuBackupUsage, a.size)
            << "Suspend must allocate exactly one CPU backup";

        ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
        EXPECT_EQ(comm->memManager->cpuBackupUsage, 0u)
            << "Resume must free the CPU backup after restore";

        std::vector<uint8_t> readback(a.size, 0u);
        ASSERT_EQ(hipMemcpy(readback.data(), a.ptr, a.size, hipMemcpyDeviceToHost),
                  hipSuccess);
        EXPECT_EQ(std::memcmp(readback.data(), seed.data(), a.size), 0)
            << "data corrupted after cycle " << cycle;
    }

    untrackAndRelease(comm, a.ptr, &a);
}

// D: stress - many small Scratch + Offload entries tracked together, single
// Suspend/Resume cycle. Catches O(n^2) regressions in the linked-list scan,
// cumulative leaks, granularity-padding bugs, and quietly-skipped entries.
TEST_F(MemManagerAnyRanks, Suspend_LargeNumberOfEntries_AllRestored)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    constexpr int kNumEntries = 32;
    constexpr int kNumOffload = 8;  // first kNumOffload are Offload, rest are Scratch

    std::vector<VmmAlloc> entries(kNumEntries);
    std::vector<uint8_t>  canaries(kNumEntries);
    uint64_t              expectedDynamicBytes = 0;
    uint64_t              expectedOffloadBytes = 0;

    for (int i = 0; i < kNumEntries; ++i) {
        // Mix sizes from 4 KiB to ~256 KiB - granularity rounding still applies.
        size_t requested = (4 * 1024) + (i * 8 * 1024);
        allocateVmmPosixFd(comm->cudaDev, requested, &entries[i]);

        canaries[i] = static_cast<uint8_t>(0x40 + i);
        ASSERT_EQ(hipMemset(entries[i].ptr, canaries[i], entries[i].size), hipSuccess);

        bool          isOffload = (i < kNumOffload);
        ncclMemType_t type      = isOffload ? ncclMemOffload : ncclMemScratch;
        ASSERT_EQ(ncclMemTrack(comm->memManager, entries[i].ptr, entries[i].size,
                               entries[i].handle,
                               hipMemHandleTypePosixFileDescriptor, type),
                  ncclSuccess);
        expectedDynamicBytes += entries[i].size;
        if (isOffload) expectedOffloadBytes += entries[i].size;
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(comm->memManager->numEntries, kNumEntries);

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->cpuBackupUsage, expectedOffloadBytes)
        << "CPU backup pool must equal sum of Offload sizes";

    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->cpuBackupUsage, 0u);

    // Every Offload entry must come back with its canary preserved. Scratch
    // entries lose data by design - just check the VA is mapped + writable.
    for (int i = 0; i < kNumEntries; ++i) {
        SCOPED_TRACE("entry=" + std::to_string(i));
        if (i < kNumOffload) {
            verifyCanaryReadback(entries[i].ptr, entries[i].size, canaries[i],
                                 "offload after Resume");
        } else {
            ASSERT_EQ(hipMemset(entries[i].ptr, 0xAA, entries[i].size), hipSuccess);
        }
    }

    auto readStat = [&](ncclCommMemStat_t s) {
        uint64_t v = ~0ULL;
        EXPECT_EQ(ncclCommMemStats(comm, s, &v), ncclSuccess);
        return v;
    };
    EXPECT_EQ(readStat(ncclStatGpuMemSuspend), expectedDynamicBytes);

    for (auto& e : entries) untrackAndRelease(comm, e.ptr, &e);
    EXPECT_EQ(comm->memManager->numEntries, 0);
}

// E: Persist physical memory must survive Suspend/Resume completely untouched
// (it lives outside the linked list, so neither Pass nor Step iterates over
// it). PersistUntouched (counter-only) leaves room for a regression where the
// VA gets accidentally unmapped; this catches it via canary roundtrip.
TEST_F(MemManagerAnyRanks, Persist_PhysicalMemoryUntouched)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    constexpr size_t  kSize   = 32 * 1024;
    constexpr uint8_t kCanary = 0xE7;

    VmmAlloc persistVmm{};
    allocateVmmPosixFd(comm->cudaDev, kSize, &persistVmm);
    ASSERT_EQ(hipMemset(persistVmm.ptr, kCanary, persistVmm.size), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    ASSERT_EQ(ncclMemTrack(comm->memManager, persistVmm.ptr, persistVmm.size,
                           persistVmm.handle, hipMemHandleTypePosixFileDescriptor,
                           ncclMemPersist),
              ncclSuccess);

    EXPECT_EQ(comm->memManager->numEntries, 0)
        << "Persist must not enter the linked list (only counter)";

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->cpuBackupUsage, 0u)
        << "Persist must NOT be CPU-backed up";

    // VA must still be live - manager skipped it, so handle/map are untouched.
    verifyCanaryReadback(persistVmm.ptr, persistVmm.size, kCanary, "persist mid-suspend");

    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
    verifyCanaryReadback(persistVmm.ptr, persistVmm.size, kCanary, "persist post-resume");

    ASSERT_EQ(ncclMemUntrack(comm->memManager, persistVmm.ptr, persistVmm.size),
              ncclSuccess);
    releaseVmm(&persistVmm);
}

// ===========================================================================
// Stress / coverage tests for TwoRank (np == 2)
// ===========================================================================

// F: rank 0 exports N=3 distinct buffers, rank 1 imports all of them, full
// Suspend/Resume cycle, every canary preserved. Exercises:
//   - Suspend Step 1: rank 1's Pass-1 unmap loop iterates over multiple
//     peer-imported entries.
//   - Suspend Step 2: rank 0's Pass-2 offload loop runs CPU-backup over N
//     distinct entries.
//   - Resume Step 3: rank 0's localBroadcastCount = N, localInfos buffer
//     holds N elements, AllGather counts come out as [N, 0].
//   - Resume Step 4: rank 1's matching loop matches N times in the same
//     allInfos array (positive branch, multiple iterations).
TEST_F(MemManagerTwoRank, Multiple_PeerEntries_RoundtripCanary)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    constexpr int     kNumBuffers  = 3;
    constexpr int     kBaseTag     = 0xC0FFEE;
    const     size_t  kSizes[kNumBuffers]    = { 64 * 1024, 96 * 1024, 128 * 1024 };
    const     uint8_t kCanaries[kNumBuffers] = { 0xA1, 0xB2, 0xC3 };

    if (comm->rank == 0) {
        std::vector<VmmAlloc> owners(kNumBuffers);
        for (int i = 0; i < kNumBuffers; ++i) {
            allocateVmmPosixFd(comm->cudaDev, kSizes[i], &owners[i]);
            ASSERT_EQ(hipMemset(owners[i].ptr, kCanaries[i], owners[i].size), hipSuccess);
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        for (int i = 0; i < kNumBuffers; ++i) {
            exportAndShipMeta(comm, owners[i], /*peerRank=*/1,
                              ncclMemOffload, kBaseTag + i);
        }

        ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
        ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);

        for (auto& o : owners) untrackAndRelease(comm, o.ptr, &o);
        return;
    }

    ASSERT_EQ(comm->rank, 1);
    std::vector<VmmAlloc> imports(kNumBuffers);
    for (int i = 0; i < kNumBuffers; ++i) {
        recvAndImportPeerBuffer(comm, /*ownerRank=*/0, ncclMemOffload, kBaseTag + i,
                                &imports[i]);
        verifyCanaryReadback(imports[i].ptr, imports[i].size, kCanaries[i],
                             "primary import of buffer #" + std::to_string(i));
    }

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);

    int activeCount = 0;
    for (auto* e = comm->memManager->entries; e != nullptr; e = e->next) {
        EXPECT_EQ(e->state, ncclDynMemStateActive);
        EXPECT_TRUE(e->isImportedFromPeer);
        ++activeCount;
    }
    EXPECT_EQ(activeCount, kNumBuffers);

    for (int i = 0; i < kNumBuffers; ++i) {
        verifyCanaryReadback(imports[i].ptr, imports[i].size, kCanaries[i],
                             "post-Resume import of buffer #" + std::to_string(i));
    }
    for (auto& imp : imports) untrackAndRelease(comm, imp.ptr, &imp);
}

// G: bidirectional - both ranks export to each other. Each rank has BOTH a
// local exported buffer AND an imported one. Exercises Resume Step 3 with
// localBroadcastCount > 0 on EVERY rank simultaneously: localInfos contains
// our own export, allInfos receives both ranks' contributions, allCounts
// comes out as [1, 1] (not [N, 0] like in F).
TEST_F(MemManagerTwoRank, Bidirectional_PeerExchange_RoundtripCanary)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);
    int rank = comm->rank;
    int peer = 1 - rank;

    constexpr size_t kSize       = 64 * 1024;
    const     uint8_t myCanary   = (rank == 0) ? 0xD1 : 0xD2;
    const     uint8_t peerCanary = (rank == 0) ? 0xD2 : 0xD1;
    // One tag per sender so the two bootstrapSends can race in flight without
    // collision at the bootstrap layer.
    const int myExportTag   = 0xC0FFEE0 + rank;
    const int peerExportTag = 0xC0FFEE0 + peer;

    VmmAlloc myExport{};
    allocateVmmPosixFd(comm->cudaDev, kSize, &myExport);
    ASSERT_EQ(hipMemset(myExport.ptr, myCanary, myExport.size), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Send my export's metadata BEFORE the recv to avoid deadlock - both ranks
    // do the same so each side's bootstrapSend buffers up at the bootstrap
    // layer until the matching bootstrapRecv fires.
    exportAndShipMeta(comm, myExport, peer, ncclMemOffload, myExportTag);

    VmmAlloc peerImport{};
    recvAndImportPeerBuffer(comm, peer, ncclMemOffload, peerExportTag, &peerImport);
    verifyCanaryReadback(peerImport.ptr, peerImport.size, peerCanary,
                         "primary import of peer's buffer");

    EXPECT_EQ(comm->memManager->numEntries, 2)
        << "Each rank should now own 1 local + 1 imported entry";

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);

    verifyCanaryReadback(peerImport.ptr, peerImport.size, peerCanary,
                         "post-Resume import of peer's buffer");

    untrackAndRelease(comm, myExport.ptr,   &myExport);
    untrackAndRelease(comm, peerImport.ptr, &peerImport);
}

// H: rank 0 exports 2 buffers, but rank 1 only does a real import for the
// FIRST one - the second is a synthetic stale import that points at a fake
// ownerPtr (nothing actually exported with that VA). Resume Step 4 must
// successfully match buffer #0 (positive branch) AND skip-with-WARN buffer
// #1 (negative branch), in the SAME matching-loop pass.
TEST_F(MemManagerTwoRank, PartialPeerImport_MixedMatch)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    constexpr size_t  kSize       = 64 * 1024;
    constexpr uint8_t kCanary     = 0x9E;
    constexpr int     kRealTag    = 0xC0FFEE;

    if (comm->rank == 0) {
        VmmAlloc realOwner{};
        allocateVmmPosixFd(comm->cudaDev, kSize, &realOwner);
        ASSERT_EQ(hipMemset(realOwner.ptr, kCanary, realOwner.size), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        exportAndShipMeta(comm, realOwner, /*peerRank=*/1, ncclMemOffload, kRealTag);

        ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
        ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);

        untrackAndRelease(comm, realOwner.ptr, &realOwner);
        return;
    }

    ASSERT_EQ(comm->rank, 1);
    VmmAlloc realImport{};
    recvAndImportPeerBuffer(comm, /*ownerRank=*/0, ncclMemOffload, kRealTag, &realImport);

    // Inject a synthetic "stale" peer-imported entry that points at an
    // ownerPtr rank 0 doesn't have. Step 4 will gather rank 0's only
    // localInfo (matching realImport), the matching loop will then hit our
    // stale entry, fail to find a match, and WARN+skip it.
    VmmAlloc stale{};
    reserveVaOnly(comm->cudaDev, kSize, &stale);
    void* fakeOwnerPtr = reinterpret_cast<void*>(0xDEADBEEF000ULL);
    ASSERT_EQ(ncclMemTrackImportFromPeer(
                  comm->memManager, stale.ptr, stale.size, /*handle=*/0,
                  hipMemHandleTypePosixFileDescriptor, ncclMemOffload,
                  /*ownerRank=*/0, /*ownerDev=*/0, fakeOwnerPtr),
              ncclSuccess);

    // Pretend Suspend already ran for the stale entry so Step 4 considers it
    // a candidate. realImport will go through the normal Suspend/Resume path.
    for (auto* e = comm->memManager->entries; e != nullptr; e = e->next) {
        if (e->ptr == stale.ptr) {
            e->state = ncclDynMemStateReleased;
            break;
        }
    }

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);

    int activeCount   = 0;
    int releasedCount = 0;
    for (auto* e = comm->memManager->entries; e != nullptr; e = e->next) {
        if      (e->state == ncclDynMemStateActive)   ++activeCount;
        else if (e->state == ncclDynMemStateReleased) ++releasedCount;
    }
    EXPECT_EQ(activeCount,   1) << "Resume must match-and-restore exactly the real import";
    EXPECT_EQ(releasedCount, 1) << "Stale entry must remain Released (no-match WARN branch)";

    verifyCanaryReadback(realImport.ptr, realImport.size, kCanary,
                         "real import survived mixed-match Resume");

    ASSERT_EQ(ncclMemUntrack(comm->memManager, stale.ptr, stale.size), ncclSuccess);
    releaseVmm(&stale);
    untrackAndRelease(comm, realImport.ptr, &realImport);
}

#endif // MPI_TESTS_ENABLED
