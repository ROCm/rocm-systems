/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file SuspendResumeMPITests.cpp
 * @brief Multi-process tests for the suspend/resume feature -- both the
 *        internal ncclCommMemSuspend/Resume/Stats path and the public
 *        ncclCommSuspend/Resume/MemStats API.
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
 * Suites:
 *   - MemManagerAnyRanks - each rank runs the same local Suspend/Resume
 *     scenario in lockstep. The cross-rank bootstrapBarrier inside
 *     ncclCommMemSuspend / ncclCommMemResume keeps all ranks in sync, so
 *     every test in this suite passes on any np >= 1.
 *   - MemManagerTwoRank   - peer-coupled scenarios that need exactly 2 ranks
 *     (rank 0 exporter, rank 1 importer).
 *   - SuspendResumeBasic / MemStats / CollectiveIntegrity / Lifecycle / Group
 *     / ArgValidation / MemStatsValidation - public API tests for
 *     ncclCommSuspend / ncclCommResume / ncclCommMemStats. Need np >= 2.
 *
 * Run (a single mpirun -np 2 invocation runs the entire file):
 *   mpirun -np 2 ./rccl-UnitTestsMPI --gtest_filter='MemManager*:SuspendResume*'
 *
 * Or, if you want to focus on one group of suites:
 *   mpirun -np 1 ./rccl-UnitTestsMPI --gtest_filter='MemManagerAnyRanks.*'
 *   mpirun -np 2 ./rccl-UnitTestsMPI --gtest_filter='MemManagerTwoRank.*'
 *   mpirun -np 2 ./rccl-UnitTestsMPI --gtest_filter='SuspendResume*'
 */

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#ifdef MPI_TESTS_ENABLED

#include "bootstrap.h"
#include "comm.h"
#include "mem_manager.h"
#include "proxy.h"

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

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
    prop.requestedHandleType             = hipMemHandleTypePosixFileDescriptor;
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
    prop.requestedHandleType             = hipMemHandleTypePosixFileDescriptor;
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

// Mirrors ncclCuMemFree: recover the live handle from the VA (cached
// a->handle is stale after Suspend/Resume). Retain bumps refcount by 1, so
// release twice. Reserved-but-unmapped VA falls through to AddressFree.
void releaseVmm(VmmAlloc* a)
{
    if (a == nullptr || a->pdev == 0) return;

    hipMemGenericAllocationHandle_t live = 0;
    if (hipMemRetainAllocationHandle(&live, a->ptr) == hipSuccess) {
        (void)hipMemRelease(live);
        (void)hipMemUnmap(a->pdev, a->size);
        (void)hipMemRelease(live);
    } else {
        (void)hipMemUnmap(a->pdev, a->size);
    }
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

// ncclMemUntrack + releaseVmm. Mirrors the ncclCudaFree pattern.
void untrackAndRelease(struct ncclComm* comm, void* ptr, VmmAlloc* a)
{
    ASSERT_NE(comm, nullptr);
    ASSERT_NE(comm->memManager, nullptr);
    ASSERT_EQ(ncclMemUntrack(comm->memManager, ptr, a->size), ncclSuccess);
    releaseVmm(a);
}

// Snapshot of manager state taken before the test adds its own entries.
// With NCCL_CUMEM_ENABLE=1 a freshly created comm already tracks internal
// RCCL allocations (proxy P2P buffers, channel buffers etc.), so absolute
// counters/sizes can't be asserted directly - tests must check deltas.
struct MgrBaseline
{
    int      numEntries     = 0;
    uint64_t cpuBackupUsage = 0;
    uint64_t total          = 0;
    uint64_t persist        = 0;
    uint64_t suspendBytes   = 0;
};

MgrBaseline captureBaseline(struct ncclComm* comm)
{
    MgrBaseline b{};
    EXPECT_NE(comm, nullptr);
    EXPECT_NE(comm->memManager, nullptr);
    b.numEntries     = comm->memManager->numEntries;
    b.cpuBackupUsage = comm->memManager->cpuBackupUsage;
    EXPECT_EQ(ncclCommMemStats(comm, ncclStatGpuMemTotal,   &b.total),        ncclSuccess);
    EXPECT_EQ(ncclCommMemStats(comm, ncclStatGpuMemPersist, &b.persist),      ncclSuccess);
    EXPECT_EQ(ncclCommMemStats(comm, ncclStatGpuMemSuspend, &b.suspendBytes), ncclSuccess);
    return b;
}

// Run one Suspend/Resume cycle to learn how much CPU backup the comm's
// internal Offload entries consume - tests with their own Offload Track must
// add their size on top of this baseline when checking cpuBackupUsage.
uint64_t captureInternalOffloadCpuBackup(struct ncclComm* comm)
{
    EXPECT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    uint64_t v = comm->memManager->cpuBackupUsage;
    EXPECT_EQ(ncclCommMemResume(comm), ncclSuccess);
    return v;
}

// Find the entry tracking `ptr`. With internals tracked it's no longer
// guaranteed to be at manager->entries[0].
ncclDynMemEntry* findEntryByPtr(struct ncclComm* comm, void* ptr)
{
    for (auto* e = comm->memManager->entries; e != nullptr; e = e->next) {
        if (e->ptr == ptr) return e;
    }
    return nullptr;
}

// Snapshot of GPU + CPU memory usage at one point in a Suspend/Resume cycle.
// Combines what the manager tracks (ncclCommMemStats + cpuBackupUsage) with
// what the OS reports for the process (hipMemGetInfo) so the caller can see
// both bookkeeping and actual physical impact of Suspend/Resume.
struct MemFootprint
{
    uint64_t mgrTotal        = 0;  // ncclStatGpuMemTotal     (bytes)
    uint64_t mgrPersist      = 0;  // ncclStatGpuMemPersist   (bytes)
    uint64_t mgrSuspendable  = 0;  // ncclStatGpuMemSuspend   (bytes)
    uint64_t mgrSuspended    = 0;  // ncclStatGpuMemSuspended (0 / 1 flag)
    uint64_t cpuBackupUsage  = 0;  // manager->cpuBackupUsage (bytes)
    size_t   gpuFree         = 0;  // hipMemGetInfo free      (bytes)
    size_t   gpuTotal        = 0;  // hipMemGetInfo total     (bytes)
    size_t   gpuUsedByProc   = 0;  // gpuTotal - gpuFree      (bytes)
};

MemFootprint captureMemFootprint(struct ncclComm* comm)
{
    MemFootprint f{};
    EXPECT_NE(comm, nullptr);
    EXPECT_NE(comm->memManager, nullptr);
    EXPECT_EQ(ncclCommMemStats(comm, ncclStatGpuMemTotal,     &f.mgrTotal),       ncclSuccess);
    EXPECT_EQ(ncclCommMemStats(comm, ncclStatGpuMemPersist,   &f.mgrPersist),     ncclSuccess);
    EXPECT_EQ(ncclCommMemStats(comm, ncclStatGpuMemSuspend,   &f.mgrSuspendable), ncclSuccess);
    EXPECT_EQ(ncclCommMemStats(comm, ncclStatGpuMemSuspended, &f.mgrSuspended),   ncclSuccess);
    f.cpuBackupUsage = comm->memManager->cpuBackupUsage;
    EXPECT_EQ(hipMemGetInfo(&f.gpuFree, &f.gpuTotal), hipSuccess);
    f.gpuUsedByProc = (f.gpuTotal > f.gpuFree) ? (f.gpuTotal - f.gpuFree) : 0;
    return f;
}

// Pretty-print one footprint line. Kept on std::cout so the host test wrapper
// captures it the same way as gtest's own output. Drop these prints (and the
// helpers above) once we no longer need to eyeball the numbers.
void printMemFootprint(int rank, const char* tag, const MemFootprint& f)
{
    constexpr double MiB = 1024.0 * 1024.0;
    std::cout << "[MemFootprint rank=" << rank << "] " << tag << '\n'
              << "    mgr.Total       = " << std::fixed << std::setprecision(2)
              << (f.mgrTotal       / MiB) << " MiB\n"
              << "    mgr.Persist     = " << (f.mgrPersist     / MiB) << " MiB\n"
              << "    mgr.Suspendable = " << (f.mgrSuspendable / MiB) << " MiB\n"
              << "    mgr.Suspended   = " << f.mgrSuspended << '\n'
              << "    cpuBackupUsage  = " << (f.cpuBackupUsage / MiB) << " MiB\n"
              << "    gpu.UsedByProc  = " << (f.gpuUsedByProc  / MiB) << " MiB"
              << " (free " << (f.gpuFree / MiB)
              << " / total " << (f.gpuTotal / MiB) << " MiB)" << std::endl;
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

    const uint64_t internalOffload = captureInternalOffloadCpuBackup(comm);

    VmmAlloc a{};
    allocateVmmPosixFd(comm->cudaDev, 64 * 1024, &a);

    ASSERT_EQ(ncclMemTrack(comm->memManager, a.ptr, a.size, a.handle,
                           hipMemHandleTypePosixFileDescriptor, ncclMemOffload),
              ncclSuccess);

    std::vector<uint8_t> pattern(a.size);
    for (size_t i = 0; i < a.size; i++) pattern[i] = static_cast<uint8_t>(i * 31u);
    ASSERT_EQ(hipMemcpy(a.ptr, pattern.data(), a.size, hipMemcpyHostToDevice), hipSuccess);

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);

    ncclDynMemEntry* entry = findEntryByPtr(comm, a.ptr);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->state, ncclDynMemStateReleased);
    EXPECT_NE(entry->cpuBackup, nullptr);
    EXPECT_EQ(comm->memManager->cpuBackupUsage, internalOffload + a.size);

    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
    EXPECT_EQ(entry->state, ncclDynMemStateActive);
    EXPECT_EQ(entry->cpuBackup, nullptr);
    EXPECT_EQ(comm->memManager->cpuBackupUsage, 0u);

    std::vector<uint8_t> roundtrip(a.size, 0);
    ASSERT_EQ(hipMemcpy(roundtrip.data(), a.ptr, a.size, hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(std::memcmp(roundtrip.data(), pattern.data(), a.size), 0);

    untrackAndRelease(comm, a.ptr, &a);
}

TEST_F(MemManagerAnyRanks, SuspendResume_PersistUntouched)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    const MgrBaseline base = captureBaseline(comm);

    constexpr size_t persistSize = 8192;
    void* persistPtr             = reinterpret_cast<void*>(0xDEADBEEF00ULL);
    ASSERT_EQ(ncclMemTrack(comm->memManager, persistPtr, persistSize,
                           /*handle=*/0, hipMemHandleTypePosixFileDescriptor,
                           ncclMemPersist),
              ncclSuccess);

    uint64_t persistBefore = 0;
    ASSERT_EQ(ncclCommMemStats(comm, ncclStatGpuMemPersist, &persistBefore), ncclSuccess);
    EXPECT_EQ(persistBefore - base.persist, persistSize);

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);

    uint64_t persistAfter = 0;
    ASSERT_EQ(ncclCommMemStats(comm, ncclStatGpuMemPersist, &persistAfter), ncclSuccess);
    EXPECT_EQ(persistAfter - base.persist, persistSize);

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

// Capture and print the memory footprint at three checkpoints: before Suspend,
// after Suspend, after Resume. Allocates a handful of real VMM buffers as
// ncclMemOffload so Suspend actually moves bytes to CPU and frees the GPU
// backing - otherwise the gpu.UsedByProc delta would be invisible.
//
// The test is non-strict on absolute numbers (other ranks share the GPU on
// shared boxes, internal RCCL allocations vary per build) and instead checks
// the invariants we care about across the cycle:
//   - mgr.Suspended  flips 0 -> 1 -> 0.
//   - cpuBackupUsage rises during Suspend, drops to baseline on Resume.
//   - gpu.UsedByProc is lower after Suspend than before, and recovers on
//     Resume to within a small slack of the pre-Suspend value.
TEST_F(MemManagerAnyRanks, MemoryFootprint_BeforeSuspendAfterSuspendAfterResume)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    // Allocate enough Offload buffers to make the GPU-memory drop on Suspend
    // visible against general device noise (other processes, driver pools).
    constexpr int    kNumBuffers = 4;
    constexpr size_t kBufBytes   = 4u * 1024u * 1024u;  // 4 MiB each, 16 MiB total
    constexpr uint8_t kPattern   = 0xC7;

    std::vector<VmmAlloc> bufs(kNumBuffers);
    for (int i = 0; i < kNumBuffers; ++i) {
        allocateVmmPosixFd(comm->cudaDev, kBufBytes, &bufs[i]);
        ASSERT_EQ(ncclMemTrack(comm->memManager, bufs[i].ptr, bufs[i].size, bufs[i].handle,
                               hipMemHandleTypePosixFileDescriptor, ncclMemOffload),
                  ncclSuccess);
        ASSERT_EQ(hipMemset(bufs[i].ptr, kPattern, bufs[i].size), hipSuccess);
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    MemFootprint before = captureMemFootprint(comm);
    //printMemFootprint(comm->rank, "BEFORE SUSPEND", before);

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);

    MemFootprint suspended = captureMemFootprint(comm);
    //printMemFootprint(comm->rank, "AFTER SUSPEND ", suspended);

    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);

    MemFootprint resumed = captureMemFootprint(comm);
    //printMemFootprint(comm->rank, "AFTER RESUME  ", resumed);

    EXPECT_EQ(before.mgrSuspended,    0u);
    EXPECT_EQ(suspended.mgrSuspended, 1u);
    EXPECT_EQ(resumed.mgrSuspended,   0u);

    EXPECT_GE(suspended.cpuBackupUsage,
              before.cpuBackupUsage + static_cast<uint64_t>(kNumBuffers) * kBufBytes)
        << "Suspend must back up at least our Offload buffers to CPU";
    EXPECT_EQ(resumed.cpuBackupUsage, before.cpuBackupUsage);

    EXPECT_LT(suspended.gpuUsedByProc, before.gpuUsedByProc)
        << "Suspend should free GPU backing for Scratch + Offload entries";

    // Resume must restore the exact pre-Suspend footprint: manager-tracked
    // totals don't change at all across the cycle (entries stay in the list,
    // only `state` toggles), and real GPU usage measured by hipMemGetInfo
    // returns to baseline once cuMemCreate + cuMemMap re-back the released VAs.
    EXPECT_EQ(resumed.mgrTotal,       before.mgrTotal);
    EXPECT_EQ(resumed.mgrPersist,     before.mgrPersist);
    EXPECT_EQ(resumed.mgrSuspendable, before.mgrSuspendable);
    EXPECT_EQ(resumed.gpuUsedByProc,  before.gpuUsedByProc)
        << "Resume must restore the exact GPU memory footprint";

    for (int i = 0; i < kNumBuffers; ++i) {
        untrackAndRelease(comm, bufs[i].ptr, &bufs[i]);
    }
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

// A: no user-tracked entries. Suspend/Resume must drive the bootstrap barriers
// + flip released without inventing or losing entries. Pre-CUMEM the manager
// was always empty here; with NCCL_CUMEM_ENABLE=1 internal entries exist, so
// we check that the entry count and cpuBackup are preserved across the cycle.
TEST_F(MemManagerAnyRanks, EmptyManager_SuspendResume_NoOp)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);
    const MgrBaseline base = captureBaseline(comm);

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->released, 1);
    EXPECT_EQ(comm->memManager->numEntries, base.numEntries)
        << "Suspend must not invent or lose entries";

    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->released, 0);
    EXPECT_EQ(comm->memManager->numEntries, base.numEntries)
        << "Resume must not invent or lose entries";
    EXPECT_EQ(comm->memManager->cpuBackupUsage, base.cpuBackupUsage);
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

    const MgrBaseline base = captureBaseline(comm);

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

    const uint64_t deltaTotal   = persistVmm.size + scratchVmm.size + offloadVmm.size;
    const uint64_t deltaPersist = persistVmm.size;
    const uint64_t deltaDynamic = scratchVmm.size + offloadVmm.size;

    EXPECT_EQ(readStat(ncclStatGpuMemTotal)   - base.total,        deltaTotal);
    EXPECT_EQ(readStat(ncclStatGpuMemPersist) - base.persist,      deltaPersist);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspend) - base.suspendBytes, deltaDynamic);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspended), 0u);

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);

    EXPECT_EQ(readStat(ncclStatGpuMemTotal)   - base.total,        deltaTotal);
    EXPECT_EQ(readStat(ncclStatGpuMemPersist) - base.persist,      deltaPersist);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspend) - base.suspendBytes, deltaDynamic);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspended), 1u);

    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
    EXPECT_EQ(readStat(ncclStatGpuMemTotal)   - base.total,        deltaTotal);
    EXPECT_EQ(readStat(ncclStatGpuMemPersist) - base.persist,      deltaPersist);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspend) - base.suspendBytes, deltaDynamic);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspended), 0u);

    verifyCanaryReadback(persistVmm.ptr, persistVmm.size, kPersistByte,
                         "persist after Suspend/Resume");

    untrackAndRelease(comm, scratchVmm.ptr, &scratchVmm);
    untrackAndRelease(comm, offloadVmm.ptr, &offloadVmm);
    ASSERT_EQ(ncclMemUntrack(comm->memManager, persistVmm.ptr, persistVmm.size),
              ncclSuccess);
    releaseVmm(&persistVmm);

    EXPECT_EQ(readStat(ncclStatGpuMemTotal),   base.total);
    EXPECT_EQ(readStat(ncclStatGpuMemPersist), base.persist);
}

// C: multiple Suspend/Resume cycles on the same Offload buffer. Verifies that
// the CPU-backup -> GPU restore loop is idempotent across many cycles - no
// cpuBackup leaks, no handle leaks, canary preserved every iteration.
TEST_F(MemManagerAnyRanks, MultipleSuspendResumeCycles_DataPreserved)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    const uint64_t internalOffload = captureInternalOffloadCpuBackup(comm);

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
        EXPECT_EQ(comm->memManager->cpuBackupUsage, internalOffload + a.size)
            << "Suspend must allocate one CPU backup per Offload entry";

        ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
        EXPECT_EQ(comm->memManager->cpuBackupUsage, 0u)
            << "Resume must free every CPU backup after restore";

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

    const MgrBaseline base            = captureBaseline(comm);
    const uint64_t    internalOffload = captureInternalOffloadCpuBackup(comm);

    constexpr int kNumEntries = 32;
    constexpr int kNumOffload = 8;  // first kNumOffload are Offload, rest are Scratch

    std::vector<VmmAlloc> entries(kNumEntries);
    std::vector<uint8_t>  canaries(kNumEntries);
    uint64_t              testDynamicBytes = 0;
    uint64_t              testOffloadBytes = 0;

    for (int i = 0; i < kNumEntries; ++i) {
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
        testDynamicBytes += entries[i].size;
        if (isOffload) testOffloadBytes += entries[i].size;
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(comm->memManager->numEntries - base.numEntries, kNumEntries);

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->cpuBackupUsage, internalOffload + testOffloadBytes)
        << "CPU backup pool must equal sum of all Offload sizes (internal + test)";

    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->cpuBackupUsage, 0u);

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
    EXPECT_EQ(readStat(ncclStatGpuMemSuspend) - base.suspendBytes, testDynamicBytes);

    for (auto& e : entries) untrackAndRelease(comm, e.ptr, &e);
    EXPECT_EQ(comm->memManager->numEntries, base.numEntries);
}

// E: Persist physical memory must survive Suspend/Resume completely untouched
// (it lives outside the linked list, so neither Pass nor Step iterates over
// it). PersistUntouched (counter-only) leaves room for a regression where the
// VA gets accidentally unmapped; this catches it via canary roundtrip.
TEST_F(MemManagerAnyRanks, Persist_PhysicalMemoryUntouched)
{
    ncclComm_t comm = getActiveCommunicator();
    ASSERT_NE(comm->memManager, nullptr);

    const MgrBaseline base            = captureBaseline(comm);
    const uint64_t    internalOffload = captureInternalOffloadCpuBackup(comm);

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

    EXPECT_EQ(comm->memManager->numEntries, base.numEntries)
        << "Persist must not enter the linked list (only counter)";

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager->cpuBackupUsage, internalOffload)
        << "Persist must NOT contribute to CPU backup pool";

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

    // Look up each test-owned import explicitly. With NCCL_CUMEM_ENABLE=1 the
    // manager also tracks ~64 internal peer-imports from src/transport/p2p.cc
    // (P2P proxy setup), so we cannot count by `isImportedFromPeer` alone.
    int activeCount = 0;
    for (int i = 0; i < kNumBuffers; ++i) {
        auto* e = findEntryByPtr(comm, imports[i].ptr);
        ASSERT_NE(e, nullptr) << "Import #" << i << " missing from manager";
        EXPECT_TRUE(e->isImportedFromPeer);
        EXPECT_EQ(e->state, ncclDynMemStateActive);
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

    const MgrBaseline base = captureBaseline(comm);

    constexpr size_t kSize       = 64 * 1024;
    const     uint8_t myCanary   = (rank == 0) ? 0xD1 : 0xD2;
    const     uint8_t peerCanary = (rank == 0) ? 0xD2 : 0xD1;
    const int myExportTag   = 0xC0FFEE0 + rank;
    const int peerExportTag = 0xC0FFEE0 + peer;

    VmmAlloc myExport{};
    allocateVmmPosixFd(comm->cudaDev, kSize, &myExport);
    ASSERT_EQ(hipMemset(myExport.ptr, myCanary, myExport.size), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    exportAndShipMeta(comm, myExport, peer, ncclMemOffload, myExportTag);

    VmmAlloc peerImport{};
    recvAndImportPeerBuffer(comm, peer, ncclMemOffload, peerExportTag, &peerImport);
    verifyCanaryReadback(peerImport.ptr, peerImport.size, peerCanary,
                         "primary import of peer's buffer");

    EXPECT_EQ(comm->memManager->numEntries - base.numEntries, 2)
        << "Each rank should add 1 local + 1 imported entry on top of baseline";

    ASSERT_EQ(ncclCommMemSuspend(comm), ncclSuccess);
    ASSERT_EQ(ncclCommMemResume(comm), ncclSuccess);

    verifyCanaryReadback(peerImport.ptr, peerImport.size, peerCanary,
                         "post-Resume import of peer's buffer");

    untrackAndRelease(comm, myExport.ptr,   &myExport);
    untrackAndRelease(comm, peerImport.ptr, &peerImport);
}

// Rank 1 has 1 real peer-import + 1 stale (fake ownerPtr). Per RCCL divergence
// (mem_manager.cc:905), Step 4 partial-success: real -> Active, stale stays
// Released, ncclCommMemResume returns ncclSystemError.
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
    // Per RCCL divergence: stale's no-match -> ncclSystemError. Real import is
    // still re-imported successfully in the same pass (partial-success).
    EXPECT_EQ(ncclCommMemResume(comm), ncclSystemError)
        << "Stale no-match peer-import must surface ncclSystemError";
    EXPECT_EQ(comm->memManager->released, 1)
        << "Partial-failure must leave manager in suspended state";

    // Look up the two test-owned entries explicitly. With NCCL_CUMEM_ENABLE=1
    // the manager also tracks ~64 internal peer-imports (proxy P2P from
    // src/transport/p2p.cc:325,380) so we cannot count by isImportedFromPeer.
    auto* realEntry  = findEntryByPtr(comm, realImport.ptr);
    auto* staleEntry = findEntryByPtr(comm, stale.ptr);
    ASSERT_NE(realEntry,  nullptr);
    ASSERT_NE(staleEntry, nullptr);
    EXPECT_EQ(realEntry->state,  ncclDynMemStateActive)
        << "Real import must be re-imported despite stale's failure";
    EXPECT_EQ(staleEntry->state, ncclDynMemStateReleased)
        << "Stale entry must stay Released (no-match branch)";

    verifyCanaryReadback(realImport.ptr, realImport.size, kCanary,
                         "real import survived mixed-match Resume");

    ASSERT_EQ(ncclMemUntrack(comm->memManager, stale.ptr, stale.size), ncclSuccess);
    releaseVmm(&stale);
    untrackAndRelease(comm, realImport.ptr, &realImport);
}

// ============================================================================
// Public API tests (ncclCommSuspend / ncclCommResume / ncclCommMemStats).
//
// The internal ncclCommMemSuspend/Resume/Stats path is exercised in detail by
// the MemManagerAnyRanks / MemManagerTwoRank suites above (peer export/import
// canary, manager state machine, etc). The suites below target the public API:
//
//  - ncclCommSuspend(comm, NCCL_SUSPEND_MEM)
//  - ncclCommResume(comm)
//  - ncclCommMemStats(comm, stat, &value)
//
// Coverage matrix:
//
//  Happy paths
//    - SuspendResumeBasic.BasicCycle              one Suspend -> Resume round-trip
//    - SuspendResumeBasic.MultipleCycles          5 back-to-back cycles
//    - SuspendResumeBasic.BarrierSync             rank 0 sleeps before suspend;
//                                                 every rank should observe the
//                                                 cross-rank bootstrap barrier
//    - SuspendResumeMemStats.BeforeAndAfter       ncclStatGpuMem* across the
//                                                 active/suspended/resumed
//                                                 transitions
//    - SuspendResumeMemStats.Conservation         total + suspended bytes is
//                                                 conserved across a Suspend
//    - SuspendResumeCollectiveIntegrity.AllReduceAfterResume
//                                                 AllReduce produces correct
//                                                 result after Suspend/Resume
//    - SuspendResumeCollectiveIntegrity.AllReduceTwoCycles
//                                                 same after multiple cycles
//    - SuspendResumeCollectiveIntegrity.AllReduceP2pAutoMark
//                                                 verifies that p2p collectives
//                                                 auto-mark exported buffers via
//                                                 ncclDynMemMarkExportToPeer
//                                                 (Bucket 1)
//    - SuspendResumeLifecycle.DestroyWhileSuspended
//                                                 ncclCommDestroy succeeds on a
//                                                 still-suspended comm (drains
//                                                 pending tasks + manager
//                                                 destruction frees released VAs)
//    - SuspendResumeGroup.AtomicSuspend           ncclGroupStart() + Suspend +
//                                                 ncclGroupEnd() drains the
//                                                 pending suspend task
//    - SuspendResumeGroup.AtomicResume            same for Resume
//    - SuspendResumeGroup.MixedSuspendResume      Suspend + Resume in the same
//                                                 group leaves the comm active
//
//  Argument validation (no bootstrap barrier reached, so individual ranks can
//  fail without deadlocking peers)
//    - SuspendResumeArgValidation.ZeroFlags       Suspend(comm, 0) is rejected
//    - SuspendResumeArgValidation.UnknownFlagsRejected  bogus high bits rejected
//    - SuspendResumeArgValidation.DoubleSuspend   Suspend twice -> 2nd rejected
//    - SuspendResumeArgValidation.ResumeWhenActive Resume on active comm rejected
//    - SuspendResumeArgValidation.DoubleResume    Resume twice -> 2nd rejected
//    - SuspendResumeMemStatsValidation.NullValuePtr  NULL output pointer rejected
//    - SuspendResumeMemStatsValidation.UnknownStat   bogus enum value rejected
// ============================================================================

namespace SuspendResumeTestConfig
{
constexpr int    kMinRanks        = 2;     // multi-rank tests need >= 2
constexpr size_t kAllReduceCount  = 1024;  // 4 KiB float buffer
constexpr float  kEpsilon         = 1e-3f;
constexpr int    kMultiCycleCount = 5;
constexpr int    kBarrierSleepMs  = 200;   // rank-0 stagger time
} // namespace SuspendResumeTestConfig

using namespace SuspendResumeTestConfig;

/**
 * @class SuspendResumePublicAPITestBase
 * @brief Shared fixture for all public-API suspend/resume MPI tests.
 *
 * Wraps MPITestBase and adds a convenience accessor for ncclCommMemStats and
 * a self-checking AllReduce helper.
 */
class SuspendResumePublicAPITestBase : public MPITestBase
{
protected:
    // Read one ncclCommMemStat as uint64_t. Fails the test on any non-success
    // return so callers can use the value directly.
    uint64_t readStat(ncclComm_t comm, ncclCommMemStat_t stat)
    {
        uint64_t v = ~uint64_t{0};
        ncclResult_t r = ncclCommMemStats(comm, stat, &v);
        EXPECT_EQ(r, ncclSuccess) << "ncclCommMemStats failed for stat " << (int)stat;
        return v;
    }

    // Run an AllReduce with float-sum and verify each element equals
    // sum(1..nranks) = nranks * (nranks + 1) / 2. Returns true on success.
    bool runAllReduceAndVerify(ncclComm_t comm, hipStream_t stream)
    {
        const size_t n     = kAllReduceCount;
        const int    rank  = MPIEnvironment::world_rank;
        const int    nrnks = MPIEnvironment::world_size;

        void* sendbuf = nullptr;
        void* recvbuf = nullptr;
        if (hipMalloc(&sendbuf, n * sizeof(float)) != hipSuccess) return false;
        if (hipMalloc(&recvbuf, n * sizeof(float)) != hipSuccess)
        {
            (void)hipFree(sendbuf);
            return false;
        }
        auto sendGuard = makeDeviceBufferAutoGuard(sendbuf);
        auto recvGuard = makeDeviceBufferAutoGuard(recvbuf);

        if (initializeBufferWithPattern<float>(
                sendbuf, n,
                [rank](size_t) { return static_cast<float>(rank + 1); }) != hipSuccess)
            return false;
        if (zeroInitializeBuffer<float>(recvbuf, n) != hipSuccess) return false;

        if (ncclAllReduce(sendbuf, recvbuf, n, ncclFloat32, ncclSum, comm, stream)
            != ncclSuccess)
            return false;
        if (hipStreamSynchronize(stream) != hipSuccess) return false;

        const float expected = static_cast<float>(nrnks * (nrnks + 1) / 2);
        return verifyBufferData<float>(
            recvbuf, n,
            [expected](size_t) { return expected; },
            0,
            static_cast<double>(kEpsilon * expected));
    }
};

// ----------------------------------------------------------------------------
// 1. Basic happy-path tests
// ----------------------------------------------------------------------------

class SuspendResumeBasic : public SuspendResumePublicAPITestBase {};

TEST_F(SuspendResumeBasic, BasicCycle)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended))
        << "Newly initialized comm must report not-suspended";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    EXPECT_EQ(1u, readStat(comm, ncclStatGpuMemSuspended))
        << "After Suspend, ncclStatGpuMemSuspended must be 1";

    // ncclStatGpuMemSuspend == totalScratch + totalOffload of *tracked* entries
    // With NCCL_CUMEM_ENABLE=0 the comm's internal allocations are not
    // VMM-backed so the count is legitimately zero; with CUMEM=1 it is > 0.
    uint64_t suspBytes = readStat(comm, ncclStatGpuMemSuspend);
    TEST_INFO("Rank %d ncclStatGpuMemSuspend after Suspend = %llu bytes",
              MPIEnvironment::world_rank, (unsigned long long)suspBytes);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));
    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended))
        << "After Resume, ncclStatGpuMemSuspended must be 0";
}

TEST_F(SuspendResumeBasic, MultipleCycles)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    for (int i = 0; i < kMultiCycleCount; ++i)
    {
        TEST_INFO("Cycle %d: Suspend", i);
        ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
        EXPECT_EQ(1u, readStat(comm, ncclStatGpuMemSuspended)) << "cycle " << i;

        TEST_INFO("Cycle %d: Resume", i);
        ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));
        EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended)) << "cycle " << i;
    }
}

TEST_F(SuspendResumeBasic, BarrierSync)
{
    // Rank 0 stages a deliberate delay before calling Suspend. Without the
    // bootstrapBarrier inside ncclCommMemSuspend, the other ranks would return
    // immediately and the elapsed time on those ranks would be much smaller
    // than rank 0's. With the barrier wired up correctly, every rank must
    // observe at least kBarrierSleepMs of elapsed wall-clock between the
    // (cross-rank) start of the call and its return.
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::steady_clock::now();

    if (MPIEnvironment::world_rank == 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(kBarrierSleepMs));
    }
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    auto t1 = std::chrono::steady_clock::now();

    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    // Generous slack: this is a synchronization assertion, not a microbenchmark.
    // Without a barrier, non-zero ranks would typically observe < 10 ms.
    EXPECT_GE(elapsed_ms, kBarrierSleepMs / 2)
        << "Rank " << MPIEnvironment::world_rank
        << ": Suspend returned in " << elapsed_ms
        << " ms (rank 0 staged a " << kBarrierSleepMs
        << " ms delay; expected the barrier to hold every rank)";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));
}

// ----------------------------------------------------------------------------
// 2. Memory-statistics correctness
// ----------------------------------------------------------------------------

class SuspendResumeMemStats : public SuspendResumePublicAPITestBase {};

TEST_F(SuspendResumeMemStats, BeforeAndAfter)
{
    // ncclCommMemStats reads per-type counters on the memory manager.
    // The counters cover *tracked* allocations only. With NCCL_CUMEM_ENABLE=0
    // RCCL skips tracking on most internal allocations so all counters
    // legitimately read zero; the suspended boolean still toggles. With
    // CUMEM=1, counters are non-zero and persist across the suspend boundary
    // (Suspend doesn't change totalScratch/totalOffload -- it only flips
    // released).
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    uint64_t totalActive = readStat(comm, ncclStatGpuMemTotal);
    uint64_t suspActive  = readStat(comm, ncclStatGpuMemSuspend);
    uint64_t persActive  = readStat(comm, ncclStatGpuMemPersist);
    uint64_t isSusActive = readStat(comm, ncclStatGpuMemSuspended);
    EXPECT_EQ(isSusActive, 0u) << "fresh comm must not be suspended";
    EXPECT_EQ(totalActive, persActive + suspActive)
        << "Total = persist + (scratch+offload) invariant";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));

    uint64_t totalSusp = readStat(comm, ncclStatGpuMemTotal);
    uint64_t suspSusp  = readStat(comm, ncclStatGpuMemSuspend);
    uint64_t isSusSusp = readStat(comm, ncclStatGpuMemSuspended);
    EXPECT_EQ(isSusSusp, 1u) << "after Suspend, suspended boolean must be 1";
    EXPECT_EQ(totalSusp, totalActive)
        << "Suspend does not change tracked byte counters (mirrors NCCL)";
    EXPECT_EQ(suspSusp, suspActive)
        << "Suspend does not change scratch/offload byte counters";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));

    uint64_t totalResumed = readStat(comm, ncclStatGpuMemTotal);
    uint64_t isSusResumed = readStat(comm, ncclStatGpuMemSuspended);
    EXPECT_EQ(isSusResumed, 0u);
    EXPECT_EQ(totalResumed, totalActive)
        << "Resume does not change tracked byte counters either";

    TEST_INFO("rank %d: total=%llu persist=%llu suspendable=%llu "
              "[scratch+offload] (CUMEM gates whether non-zero)",
              MPIEnvironment::world_rank,
              (unsigned long long)totalActive,
              (unsigned long long)persActive,
              (unsigned long long)suspActive);
}

TEST_F(SuspendResumeMemStats, Conservation)
{
    // ncclCommMemStats reports per-type byte counters; Suspend doesn't move
    // bytes between persist / scratch / offload buckets, it only flips the
    // released boolean. So Total = Persist + (Scratch+Offload) holds in every
    // state, and Total/Persist/Suspend are constant across a Suspend->Resume
    // round-trip. (CPU backup memory used by ncclMemOffload entries is tracked
    // separately on manager->cpuBackupUsage and not part of the public stats.)
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    uint64_t totalA = readStat(comm, ncclStatGpuMemTotal);
    uint64_t persA  = readStat(comm, ncclStatGpuMemPersist);
    uint64_t suspA  = readStat(comm, ncclStatGpuMemSuspend);
    EXPECT_EQ(totalA, persA + suspA);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    uint64_t totalS = readStat(comm, ncclStatGpuMemTotal);
    uint64_t persS  = readStat(comm, ncclStatGpuMemPersist);
    uint64_t suspS  = readStat(comm, ncclStatGpuMemSuspend);
    EXPECT_EQ(totalS, totalA);
    EXPECT_EQ(persS,  persA);
    EXPECT_EQ(suspS,  suspA);
    EXPECT_EQ(totalS, persS + suspS);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));

    uint64_t totalR = readStat(comm, ncclStatGpuMemTotal);
    EXPECT_EQ(totalR, totalA);
}

// ----------------------------------------------------------------------------
// 3. Collective integrity across a suspend/resume cycle
// ----------------------------------------------------------------------------

class SuspendResumeCollectiveIntegrity : public SuspendResumePublicAPITestBase
{
protected:
    void SetUp() override
    {
        SuspendResumePublicAPITestBase::SetUp();
        if (!ncclCuMemEnable()) {
            GTEST_SKIP() << "NCCL_CUMEM_ENABLE=0 — skipped";
        }
    }
};

TEST_F(SuspendResumeCollectiveIntegrity, AllReduceAfterResume)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    EXPECT_TRUE(runAllReduceAndVerify(comm, stream))
        << "AllReduce baseline failed before any Suspend";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));

    EXPECT_TRUE(runAllReduceAndVerify(comm, stream))
        << "AllReduce produced wrong values after Suspend/Resume cycle";
}

TEST_F(SuspendResumeCollectiveIntegrity, AllReduceTwoCycles)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    EXPECT_TRUE(runAllReduceAndVerify(comm, stream)) << "baseline AllReduce";

    for (int i = 0; i < 2; ++i)
    {
        ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));

        EXPECT_TRUE(runAllReduceAndVerify(comm, stream))
            << "AllReduce wrong after cycle " << i;
    }
}

// Regression: ncclP2pImportShareableBuffer (and the intra-PID different-GPU
// branch in p2pMap) must call ncclMemTrackImportFromPeer so the importer side
// of every cuMem-backed proxy buffer is visible to ncclMemManager. Without
// this wiring Resume Step 4 has nothing to re-import: after Suspend the owner
// rebuilds its physical pages, but the importer's local VA stays mapped to
// the OLD physical pages, owner/importer drift apart, and the next collective
// hangs in hipStreamSynchronize waiting on flags that never converge across
// the diverged buffers.
//
// We catch the bug structurally instead of as a hang:
//   1. Force p2p proxy connections via a warm-up AllReduce.
//   2. Walk manager->entries and require >= 1 entry with isImportedFromPeer
//      and a non-zero live handle. Pre-fix this set is empty under CUMEM=1;
//      post-fix every cross-PID p2p connection adds one.
//   3. Suspend/Resume and verify that those imported entries return Active
//      with rebuilt (still non-zero) handles and a non-stale ownerPtr.
TEST_F(SuspendResumeCollectiveIntegrity, P2pPeerImports_TrackedAndRebuilt)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    ASSERT_NE(comm->memManager, nullptr);

    EXPECT_TRUE(runAllReduceAndVerify(comm, stream))
        << "warm-up AllReduce must succeed before peer-import inspection";

    auto countPeerImports = [&](int* nActive, int* nReleased) {
        *nActive = *nReleased = 0;
        std::lock_guard<std::mutex> lk(comm->memManager->lock);
        for (auto* e = comm->memManager->entries; e != nullptr; e = e->next) {
            if (!e->isImportedFromPeer) continue;
            if (e->state == ncclDynMemStateActive)        (*nActive)++;
            else if (e->state == ncclDynMemStateReleased) (*nReleased)++;
        }
    };

    int activeBefore = 0, releasedBefore = 0;
    countPeerImports(&activeBefore, &releasedBefore);
    EXPECT_EQ(releasedBefore, 0)
        << "no entry should be Released while comm is unsuspended";
    EXPECT_GT(activeBefore, 0)
        << "ncclP2pImportShareableBuffer must register every cross-PID p2p "
           "import via ncclMemTrackImportFromPeer (regression: 0 imports "
           "tracked => Resume Step 4 is a no-op => post-Resume collectives "
           "hang on stale peer mappings)";

    if (activeBefore == 0) return;

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));

    int activeMid = 0, releasedMid = 0;
    countPeerImports(&activeMid, &releasedMid);
    EXPECT_EQ(activeMid, 0)
        << "Suspend must move every peer-imported entry to Released";
    EXPECT_EQ(releasedMid, activeBefore)
        << "Suspend must Release exactly the entries that were Active before";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));

    int activeAfter = 0, releasedAfter = 0;
    countPeerImports(&activeAfter, &releasedAfter);
    EXPECT_EQ(releasedAfter, 0)
        << "Resume must rebuild every peer-imported entry";
    EXPECT_EQ(activeAfter, activeBefore)
        << "Resume must restore the same number of peer-imported entries";

    {
        std::lock_guard<std::mutex> lk(comm->memManager->lock);
        for (auto* e = comm->memManager->entries; e != nullptr; e = e->next) {
            if (!e->isImportedFromPeer) continue;
            EXPECT_NE(e->ptr, nullptr) << "imported VA must be preserved";
            EXPECT_NE(e->desc.imported.ownerPtr, nullptr)
                << "ownerPtr must survive Suspend/Resume so Step 4 can match";
            // handle == 0 is legal for the intra-PID different-GPU branch
            // (handle pre-released by p2pMap), so we don't assert non-zero
            // here. The cross-PID branch will have a fresh non-zero handle.
        }
    }

    EXPECT_TRUE(runAllReduceAndVerify(comm, stream))
        << "post-Resume AllReduce must complete (no hang on stale imports)";
}

TEST_F(SuspendResumeCollectiveIntegrity, AllReduceP2pAutoMark)
{
    // Bucket 1 invariant: p2pSendProxySetup / p2pRecvProxySetup pass req->peerRank
    // into ncclP2pAllocateShareableBuffer, which in turn calls
    // ncclDynMemMarkExportToPeer on every cuMem-allocated proxy buffer. After
    // Suspend, the manager's Step 3 in ncclCommMemResume re-exports those
    // buffers to the same peer using the recorded peerRank, so the post-Resume
    // AllReduce can rebuild peer mappings without manual MarkExport calls.
    //
    // This test exercises the full path end-to-end: an AllReduce that uses the
    // p2p transport must continue to work after a full Suspend/Resume cycle.
    // If peerRank is lost this test still passes when CUMEM=0
    // (no peer marking happens at all) but exposes the regression
    // as a SIGSEGV during Resume Step 4 with CUMEM=1 + nRanks >= 2.
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    // Warm-up AllReduce so the p2p proxy buffers are actually allocated and
    // marked for export before we suspend.
    EXPECT_TRUE(runAllReduceAndVerify(comm, stream)) << "warm-up AllReduce";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));

    EXPECT_TRUE(runAllReduceAndVerify(comm, stream))
        << "AllReduce broken after Suspend/Resume - peer-export marking "
           "regression?";
}

// ----------------------------------------------------------------------------
// 4. Lifecycle: destroy while suspended
// ----------------------------------------------------------------------------

class SuspendResumeLifecycle : public SuspendResumePublicAPITestBase
{
protected:
    // We tear down our own comm in this test rather than relying on the base
    // class, so that we drive ncclCommDestroy from a suspended state.
    void TearDown() override
    {
        // Skip MPITestBase::TearDown's cleanupTestCommunicator since the test
        // already destroyed it.
        test_comm_   = nullptr;
        test_stream_ = nullptr;
    }
};

TEST_F(SuspendResumeLifecycle, DestroyWhileSuspended)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    EXPECT_EQ(1u, readStat(comm, ncclStatGpuMemSuspended));

    // Tear the comm down without resuming. commFree drains pending suspend/resume
    // tasks and ncclMemManagerDestroy reclaims VAs of Released entries, so the
    // destructor walk must not SIGSEGV.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommDestroy(comm));
    test_comm_ = nullptr;
    if (stream != nullptr)
    {
        (void)hipStreamDestroy(stream);
        test_stream_ = nullptr;
    }
}

// ----------------------------------------------------------------------------
// 5. Group context (Bucket 3): Suspend/Resume inside ncclGroupStart/End
// ----------------------------------------------------------------------------

class SuspendResumeGroup : public SuspendResumePublicAPITestBase {};

TEST_F(SuspendResumeGroup, AtomicSuspend)
{
    // ncclCommSuspend always wraps the body in
    // ncclGroupStartInternal/EndInternal, so a user-level
    // ncclGroupStart/Suspend/ncclGroupEnd nesting must still drain the pending
    // suspend task and leave the comm in the suspended state.
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());

    EXPECT_EQ(1u, readStat(comm, ncclStatGpuMemSuspended))
        << "Suspend inside a group must drain in groupLaunch";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));
    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended));
}

TEST_F(SuspendResumeGroup, AtomicResume)
{
    // Same as AtomicSuspend but for Resume.
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    EXPECT_EQ(1u, readStat(comm, ncclStatGpuMemSuspended));

    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());

    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended))
        << "Resume inside a group must drain in groupLaunch";
}

TEST_F(SuspendResumeGroup, MixedSuspendResume)
{
    // Suspend + Resume in the same group: groupLaunch drains suspendTaskQueue
    // first, then resumeTaskQueue, so the net effect is "active" again.
    // The RCCL pre-check in ncclCommResume_impl accepts this case because at
    // the moment Resume is called there is still a Suspend task pending in
    // suspendTaskQueue (i.e. the comm is "going to be suspended" — not
    // currently active in the sense of the validation).
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended));

    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());

    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended))
        << "After Suspend+Resume in the same group, comm must be active";
}

// ----------------------------------------------------------------------------
// 6. Argument validation / corner cases
//
// Each of these returns from inside ncclCommSuspend/Resume/MemStats BEFORE the
// bootstrapBarrier is reached, so individual ranks failing the call do not
// leave their peers blocked on a barrier.
// ----------------------------------------------------------------------------

class SuspendResumeArgValidation : public SuspendResumePublicAPITestBase {};

TEST_F(SuspendResumeArgValidation, ZeroFlags)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    ASSERT_MPI_EQ(ncclInvalidArgument, ncclCommSuspend(comm, 0));

    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended))
        << "rejected Suspend should not transition the comm";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));
}

TEST_F(SuspendResumeArgValidation, UnknownFlagsRejected)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    ASSERT_MPI_EQ(ncclInvalidArgument, ncclCommSuspend(comm, 0xffff));
    ASSERT_MPI_EQ(ncclInvalidArgument,
                  ncclCommSuspend(comm, NCCL_SUSPEND_MEM | 0x80));

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));
}

TEST_F(SuspendResumeArgValidation, DoubleSuspend)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));

    // Second suspend rejects on every rank before reaching the barrier
    // (state check sees comm->memManager->released==1).
    ASSERT_MPI_EQ(ncclInvalidUsage, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));

    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));
    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended));
}

TEST_F(SuspendResumeArgValidation, ResumeWhenActive)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    // Resume on an already-active comm rejects before any barrier.
    ASSERT_MPI_EQ(ncclInvalidUsage, ncclCommResume(comm));

    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended))
        << "rejected Resume should not transition the comm";
}

TEST_F(SuspendResumeArgValidation, DoubleResume)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommResume(comm));

    // Second resume rejects: comm is no longer suspended.
    ASSERT_MPI_EQ(ncclInvalidUsage, ncclCommResume(comm));

    EXPECT_EQ(0u, readStat(comm, ncclStatGpuMemSuspended));
}

// ----- ncclCommMemStats validation --------------------------------------

class SuspendResumeMemStatsValidation : public SuspendResumePublicAPITestBase {};

TEST_F(SuspendResumeMemStatsValidation, NullValuePtr)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    EXPECT_EQ(ncclInvalidArgument,
              ncclCommMemStats(comm, ncclStatGpuMemTotal, nullptr));
    EXPECT_EQ(ncclInvalidArgument,
              ncclCommMemStats(comm, ncclStatGpuMemSuspend, nullptr));
}

TEST_F(SuspendResumeMemStatsValidation, UnknownStat)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks)) << "Need >= 2 ranks";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    uint64_t v = 0;
    EXPECT_EQ(ncclInvalidArgument,
              ncclCommMemStats(comm, static_cast<ncclCommMemStat_t>(999), &v));
    EXPECT_EQ(ncclInvalidArgument,
              ncclCommMemStats(comm, static_cast<ncclCommMemStat_t>(-1), &v));
}

#endif // MPI_TESTS_ENABLED
