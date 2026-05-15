/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>
#include <fcntl.h>
#include <random>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "alloc.h"
#include "comm.h"
#include "mem_manager.h"
#include "utils.h"

#include "ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting
{

// Synthetic constants used by pure-state tests. The manager only stores and
// compares ptr / handle, so any sufficiently distinct value works.
namespace
{
constexpr uintptr_t              kFakePtrBase    = 0xDEAD0000ULL;
const hipMemAllocationHandleType kFakeHandleType = hipMemHandleTypePosixFileDescriptor;

hipMemGenericAllocationHandle_t fakeHandle()
{
    return reinterpret_cast<hipMemGenericAllocationHandle_t>(0xCAFEF00DULL);
}

void* fakePtr(uintptr_t i) { return reinterpret_cast<void*>(kFakePtrBase + i); }
} // namespace

// Pure-state fixture: no HIP init, no real allocations. Suitable for fast logic
// checks against the manager's bookkeeping (counters, linked-list, refCount).
class MemManagerTest : public ::testing::Test
{
protected:
    ncclComm* comm = nullptr;

    void SetUp() override
    {
        comm = new ncclComm();
        comm->cudaDev = 0;
    }

    void TearDown() override
    {
        if(comm && comm->memManager) {
            ncclMemManagerDestroy(comm);
        }
        delete comm;
        comm = nullptr;
    }
};

// ---------------------------------------------------------------------------
// Init / Destroy (pure state)
// ---------------------------------------------------------------------------

TEST_F(MemManagerTest, Init_Success)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ASSERT_NE(comm->memManager, nullptr);

    ncclMemManager* m = comm->memManager;
    EXPECT_EQ(m->initialized, 1);
    EXPECT_EQ(m->refCount, 1);
    EXPECT_EQ(m->numEntries, 0);
    EXPECT_EQ(m->entries, nullptr);
    EXPECT_EQ(m->commCudaDev, 0);
    EXPECT_EQ(m->totalPersist, 0u);
    EXPECT_EQ(m->totalPersistImported, 0u);
    EXPECT_EQ(m->totalScratch, 0u);
    EXPECT_EQ(m->totalScratchImported, 0u);
    EXPECT_EQ(m->totalOffload, 0u);
    EXPECT_EQ(m->totalOffloadImported, 0u);
    EXPECT_EQ(m->cpuBackupUsage, 0u);
}

TEST_F(MemManagerTest, Init_NullComm)
{
    EXPECT_EQ(ncclMemManagerInit(nullptr), ncclInvalidArgument);
}

TEST_F(MemManagerTest, Destroy_NullComm)
{
    EXPECT_EQ(ncclMemManagerDestroy(nullptr), ncclInvalidArgument);
}

TEST_F(MemManagerTest, Destroy_NullManager)
{
    ASSERT_EQ(comm->memManager, nullptr);
    EXPECT_EQ(ncclMemManagerDestroy(comm), ncclSuccess);
}

TEST_F(MemManagerTest, Destroy_RefCountDecrement)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* shared = comm->memManager;

    // Simulate parent->shareResources path: bump refCount and reuse manager.
    ncclComm* child   = new ncclComm();
    child->cudaDev    = 0;
    child->memManager = shared;
    ncclAtomicRefCountIncrement(&shared->refCount);
    ASSERT_EQ(shared->refCount, 2);

    ASSERT_EQ(ncclMemManagerDestroy(child), ncclSuccess);
    EXPECT_EQ(child->memManager, nullptr);
    EXPECT_EQ(shared->refCount, 1);
    EXPECT_EQ(shared->initialized, 1);
    delete child;

    ASSERT_EQ(ncclMemManagerDestroy(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager, nullptr);
}

TEST_F(MemManagerTest, RefCount_MultipleChildren_VariousDestroyOrder)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* shared = comm->memManager;

    constexpr int          kChildren = 3;
    std::vector<ncclComm*> children;
    for(int i = 0; i < kChildren; ++i) {
        ncclComm* c   = new ncclComm();
        c->cudaDev    = 0;
        c->memManager = shared;
        ncclAtomicRefCountIncrement(&shared->refCount);
        children.push_back(c);
    }
    ASSERT_EQ(shared->refCount, 1 + kChildren);

    // Destroy children in non-sequential order: 0, 2, 1. Manager must stay
    // alive until refCount drops to 0 (only parent left).
    const int order[kChildren] = {0, 2, 1};
    int       expectedRefCount = 1 + kChildren;
    for(int idx : order) {
        ASSERT_EQ(ncclMemManagerDestroy(children[idx]), ncclSuccess);
        EXPECT_EQ(children[idx]->memManager, nullptr);
        --expectedRefCount;
        EXPECT_EQ(shared->refCount, expectedRefCount);
        EXPECT_EQ(shared->initialized, 1) << "shared manager freed prematurely";
    }
    for(auto* c : children) {
        delete c;
    }

    ASSERT_EQ(shared->refCount, 1);
    ASSERT_EQ(ncclMemManagerDestroy(comm), ncclSuccess);
    EXPECT_EQ(comm->memManager, nullptr);
}

// ---------------------------------------------------------------------------
// Track / Untrack (pure state, synthetic pointers)
// ---------------------------------------------------------------------------

TEST_F(MemManagerTest, Track_PersistOnlyUpdatesCounter)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    constexpr size_t kSize = 4096;
    ASSERT_EQ(ncclMemTrack(m, fakePtr(1), kSize, fakeHandle(), kFakeHandleType, ncclMemPersist),
              ncclSuccess);

    EXPECT_EQ(m->numEntries, 0);
    EXPECT_EQ(m->entries, nullptr);
    EXPECT_EQ(m->totalPersist, kSize);
    EXPECT_EQ(m->totalScratch, 0u);
    EXPECT_EQ(m->totalOffload, 0u);
}

TEST_F(MemManagerTest, TrackImportFromPeer_FillsImportedDesc)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    constexpr size_t kSize     = 2048;
    constexpr int    kOwnerRk  = 7;
    constexpr int    kOwnerDev = 3;
    void*            p         = fakePtr(4);
    void*            ownerPtr  = fakePtr(0xABCD);

    ASSERT_EQ(ncclMemTrackImportFromPeer(m, p, kSize, fakeHandle(), kFakeHandleType,
                                         ncclMemScratch, kOwnerRk, kOwnerDev, ownerPtr),
              ncclSuccess);

    ASSERT_NE(m->entries, nullptr);
    EXPECT_EQ(m->numEntries, 1);
    EXPECT_TRUE(m->entries->isImportedFromPeer);
    EXPECT_EQ(m->entries->desc.imported.ownerRank, kOwnerRk);
    EXPECT_EQ(m->entries->desc.imported.ownerDev, kOwnerDev);
    EXPECT_EQ(m->entries->desc.imported.ownerPtr, ownerPtr);
    EXPECT_EQ(m->totalScratchImported, kSize);
    EXPECT_EQ(m->totalScratch, 0u);

    ASSERT_EQ(ncclMemTrackImportFromPeer(m, fakePtr(5), kSize, fakeHandle(), kFakeHandleType,
                                         ncclMemPersist, kOwnerRk, kOwnerDev, ownerPtr),
              ncclSuccess);
    EXPECT_EQ(m->totalPersistImported, kSize);
    EXPECT_EQ(m->totalPersist, 0u);
}

TEST_F(MemManagerTest, TrackImport_CountersIncrement_AllMemTypes)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    constexpr size_t kP = 1024;
    constexpr size_t kS = 2048;
    constexpr size_t kO = 4096;

    ASSERT_EQ(ncclMemTrackImportFromPeer(m, fakePtr(0x40), kP, fakeHandle(), kFakeHandleType,
                                         ncclMemPersist, /*ownerRank=*/1, /*ownerDev=*/0,
                                         /*ownerPtr=*/fakePtr(0x140)),
              ncclSuccess);
    ASSERT_EQ(ncclMemTrackImportFromPeer(m, fakePtr(0x41), kS, fakeHandle(), kFakeHandleType,
                                         ncclMemScratch, /*ownerRank=*/2, /*ownerDev=*/0,
                                         /*ownerPtr=*/fakePtr(0x141)),
              ncclSuccess);
    ASSERT_EQ(ncclMemTrackImportFromPeer(m, fakePtr(0x42), kO, fakeHandle(), kFakeHandleType,
                                         ncclMemOffload, /*ownerRank=*/3, /*ownerDev=*/0,
                                         /*ownerPtr=*/fakePtr(0x142)),
              ncclSuccess);

    EXPECT_EQ(m->totalPersistImported, kP);
    EXPECT_EQ(m->totalScratchImported, kS);
    EXPECT_EQ(m->totalOffloadImported, kO);
    // Local counters stay untouched: imported allocations don't bump them.
    EXPECT_EQ(m->totalPersist, 0u);
    EXPECT_EQ(m->totalScratch, 0u);
    EXPECT_EQ(m->totalOffload, 0u);
    // Persist isn't kept in the linked list, only Scratch and Offload are.
    EXPECT_EQ(m->numEntries, 2);

    ASSERT_EQ(ncclMemUntrack(m, fakePtr(0x41), kS), ncclSuccess);
    ASSERT_EQ(ncclMemUntrack(m, fakePtr(0x42), kO), ncclSuccess);
    EXPECT_EQ(m->totalScratchImported, 0u);
    EXPECT_EQ(m->totalOffloadImported, 0u);
}

TEST_F(MemManagerTest, Untrack_PersistPath)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    constexpr size_t kSize = 1024;
    void*            p     = fakePtr(7);
    ASSERT_EQ(ncclMemTrack(m, p, kSize, fakeHandle(), kFakeHandleType, ncclMemPersist),
              ncclSuccess);
    ASSERT_EQ(m->totalPersist, kSize);

    // Persistent allocations are not in the linked list; Untrack falls into
    // the "not found" branch which decrements totalPersist by the passed size.
    ASSERT_EQ(ncclMemUntrack(m, p, kSize), ncclSuccess);
    EXPECT_EQ(m->totalPersist, 0u);
    EXPECT_EQ(m->numEntries, 0);
}

TEST_F(MemManagerTest, Untrack_NullManagerReturnsError)
{
    EXPECT_EQ(ncclMemUntrack(nullptr, fakePtr(8), 1024), ncclInternalError);
}

TEST_F(MemManagerTest, Untrack_NullPtrReturnsError)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    EXPECT_EQ(ncclMemUntrack(comm->memManager, nullptr, 1024), ncclInternalError);
}

TEST_F(MemManagerTest, Untrack_HeadMiddleTail_LinkedListRemoval)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    // Track A,B,C,D in order. Track prepends, so the list is
    // head=D -> C -> B -> A=tail.
    constexpr size_t kSize = 1024;
    void*            a     = fakePtr(0x300);
    void*            b     = fakePtr(0x301);
    void*            c     = fakePtr(0x302);
    void*            d     = fakePtr(0x303);
    for(void* p : {a, b, c, d}) {
        ASSERT_EQ(ncclMemTrack(m, p, kSize, fakeHandle(), kFakeHandleType, ncclMemScratch),
                  ncclSuccess);
    }
    ASSERT_EQ(m->numEntries, 4);
    ASSERT_EQ(m->entries->ptr, d);                                // head
    ASSERT_EQ(m->entries->next->ptr, c);
    ASSERT_EQ(m->entries->next->next->ptr, b);
    ASSERT_EQ(m->entries->next->next->next->ptr, a);              // tail

    // 1) Untrack the head: hits the prev == nullptr branch.
    ASSERT_EQ(ncclMemUntrack(m, d, kSize), ncclSuccess);
    EXPECT_EQ(m->numEntries, 3);
    EXPECT_EQ(m->entries->ptr, c);

    // 2) Untrack the middle: prev != nullptr, entry->next != nullptr.
    ASSERT_EQ(ncclMemUntrack(m, b, kSize), ncclSuccess);
    EXPECT_EQ(m->numEntries, 2);
    EXPECT_EQ(m->entries->ptr, c);
    EXPECT_EQ(m->entries->next->ptr, a);

    // 3) Untrack the tail: prev != nullptr, entry->next == nullptr.
    ASSERT_EQ(ncclMemUntrack(m, a, kSize), ncclSuccess);
    EXPECT_EQ(m->numEntries, 1);
    EXPECT_EQ(m->entries->ptr, c);
    EXPECT_EQ(m->entries->next, nullptr);

    // 4) Untrack the singleton: prev == nullptr AND head becomes nullptr.
    ASSERT_EQ(ncclMemUntrack(m, c, kSize), ncclSuccess);
    EXPECT_EQ(m->numEntries, 0);
    EXPECT_EQ(m->entries, nullptr);
    EXPECT_EQ(m->totalScratch, 0u);
}

TEST_F(MemManagerTest, Untrack_SizeMismatchUsesTrackedSize)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    constexpr size_t kTracked = 4096;
    void*            p        = fakePtr(0x310);
    ASSERT_EQ(ncclMemTrack(m, p, kTracked, fakeHandle(), kFakeHandleType, ncclMemScratch),
              ncclSuccess);
    ASSERT_EQ(m->totalScratch, kTracked);

    // Pass a wildly wrong size: manager logs a mismatch but uses the tracked
    // entry->size for accounting. This documents the contract — callers may
    // pass 0 or a stale size and the totals stay consistent.
    ASSERT_EQ(ncclMemUntrack(m, p, /*passedSize=*/123), ncclSuccess);
    EXPECT_EQ(m->totalScratch, 0u);
    EXPECT_EQ(m->numEntries, 0);
}

TEST_F(MemManagerTest, Track_DuplicatePtr_CreatesTwoEntries)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    // Manager doesn't dedupe Track — two entries with the same ptr can legally
    // co-exist (e.g. recycled VA after Untrack/Track in flight). Removal is
    // LIFO because Track prepends.
    constexpr size_t kSizeA = 1024;
    constexpr size_t kSizeB = 2048;
    void*            p      = fakePtr(0x320);

    ASSERT_EQ(ncclMemTrack(m, p, kSizeA, fakeHandle(), kFakeHandleType, ncclMemScratch),
              ncclSuccess);
    ASSERT_EQ(ncclMemTrack(m, p, kSizeB, fakeHandle(), kFakeHandleType, ncclMemScratch),
              ncclSuccess);
    EXPECT_EQ(m->numEntries, 2);
    EXPECT_EQ(m->totalScratch, kSizeA + kSizeB);

    // First Untrack hits the head (prepended last) — kSizeB.
    ASSERT_EQ(ncclMemUntrack(m, p, kSizeB), ncclSuccess);
    EXPECT_EQ(m->numEntries, 1);
    EXPECT_EQ(m->totalScratch, kSizeA);
    EXPECT_EQ(m->entries->size, kSizeA);

    // Second Untrack hits the remaining (originally first) entry.
    ASSERT_EQ(ncclMemUntrack(m, p, kSizeA), ncclSuccess);
    EXPECT_EQ(m->numEntries, 0);
    EXPECT_EQ(m->totalScratch, 0u);
}

// ---------------------------------------------------------------------------
// MarkExportToPeer
// ---------------------------------------------------------------------------

TEST_F(MemManagerTest, MarkExport_AddsPeer)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    void* p = fakePtr(9);
    ASSERT_EQ(ncclMemTrack(m, p, 4096, fakeHandle(), kFakeHandleType, ncclMemScratch),
              ncclSuccess);

    ASSERT_EQ(ncclDynMemMarkExportToPeer(m, p, /*peerRank=*/1), ncclSuccess);
    EXPECT_EQ(m->entries->desc.local.numExportedPeers, 1);
    EXPECT_EQ(m->entries->desc.local.exportedPeersCapacity, NCCL_MEM_EXPORT_PEERS_INIT);
    EXPECT_EQ(m->entries->desc.local.exportedPeerRanks[0], 1);
}

TEST_F(MemManagerTest, MarkExport_GrowsCapacity)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    void* p = fakePtr(10);
    ASSERT_EQ(ncclMemTrack(m, p, 4096, fakeHandle(), kFakeHandleType, ncclMemScratch),
              ncclSuccess);

    constexpr int kPeers = NCCL_MEM_EXPORT_PEERS_INIT + 1;
    for(int peer = 0; peer < kPeers; ++peer) {
        ASSERT_EQ(ncclDynMemMarkExportToPeer(m, p, peer), ncclSuccess) << "peer=" << peer;
    }
    EXPECT_EQ(m->entries->desc.local.numExportedPeers, kPeers);
    EXPECT_EQ(m->entries->desc.local.exportedPeersCapacity, NCCL_MEM_EXPORT_PEERS_INIT * 2);
    for(int peer = 0; peer < kPeers; ++peer) {
        EXPECT_EQ(m->entries->desc.local.exportedPeerRanks[peer], peer);
    }
}

TEST_F(MemManagerTest, MarkExport_DuplicatePeerRejected)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    void* p = fakePtr(11);
    ASSERT_EQ(ncclMemTrack(m, p, 4096, fakeHandle(), kFakeHandleType, ncclMemScratch),
              ncclSuccess);

    ASSERT_EQ(ncclDynMemMarkExportToPeer(m, p, 2), ncclSuccess);
    EXPECT_EQ(ncclDynMemMarkExportToPeer(m, p, 2), ncclInternalError);
    EXPECT_EQ(m->entries->desc.local.numExportedPeers, 1);
}

TEST_F(MemManagerTest, MarkExport_PersistRejected)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    void* p = fakePtr(12);
    ASSERT_EQ(ncclMemTrack(m, p, 4096, fakeHandle(), kFakeHandleType, ncclMemPersist),
              ncclSuccess);
    EXPECT_EQ(ncclDynMemMarkExportToPeer(m, p, 1), ncclInternalError);
}

TEST_F(MemManagerTest, MarkExport_ImportedRejected)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    void* p = fakePtr(13);
    ASSERT_EQ(ncclMemTrackImportFromPeer(m, p, 4096, fakeHandle(), kFakeHandleType,
                                         ncclMemScratch, /*ownerRank=*/4, /*ownerDev=*/0,
                                         /*ownerPtr=*/fakePtr(0xBEEF)),
              ncclSuccess);
    EXPECT_EQ(ncclDynMemMarkExportToPeer(m, p, 1), ncclInternalError);
}

TEST_F(MemManagerTest, MarkExport_GrowsCapacity_LargeN)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    void* p = fakePtr(0x330);
    ASSERT_EQ(ncclMemTrack(m, p, 4096, fakeHandle(), kFakeHandleType, ncclMemScratch),
              ncclSuccess);

    // 65 unique peer ranks. Capacity grows 0 -> 8 -> 16 -> 32 -> 64 -> 128 as
    // ncclRealloc doubles each time numExportedPeers reaches the cap.
    constexpr int kPeers = 65;
    for(int peer = 0; peer < kPeers; ++peer) {
        ASSERT_EQ(ncclDynMemMarkExportToPeer(m, p, peer), ncclSuccess) << "peer=" << peer;
    }
    EXPECT_EQ(m->entries->desc.local.numExportedPeers, kPeers);
    EXPECT_EQ(m->entries->desc.local.exportedPeersCapacity, 128);
    for(int peer = 0; peer < kPeers; ++peer) {
        EXPECT_EQ(m->entries->desc.local.exportedPeerRanks[peer], peer) << "peer=" << peer;
    }
}

TEST_F(MemManagerTest, MarkExport_FreedOnUntrack)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    void* p = fakePtr(0x340);
    ASSERT_EQ(ncclMemTrack(m, p, 4096, fakeHandle(), kFakeHandleType, ncclMemScratch),
              ncclSuccess);
    for(int peer = 0; peer < 4; ++peer) {
        ASSERT_EQ(ncclDynMemMarkExportToPeer(m, p, peer), ncclSuccess);
    }
    ASSERT_NE(m->entries->desc.local.exportedPeerRanks, nullptr);
    ASSERT_EQ(m->entries->desc.local.numExportedPeers, 4);

    // Untrack must free the exportedPeerRanks array along with the entry.
    // ASan/leak-sanitizer would flag a missed free here.
    ASSERT_EQ(ncclMemUntrack(m, p, 4096), ncclSuccess);
    EXPECT_EQ(m->numEntries, 0);
    EXPECT_EQ(m->entries, nullptr);

    // Re-track the same ptr: the new entry must start with a clean slate, no
    // stale capacity carried over from the freed entry.
    ASSERT_EQ(ncclMemTrack(m, p, 4096, fakeHandle(), kFakeHandleType, ncclMemScratch),
              ncclSuccess);
    EXPECT_EQ(m->entries->desc.local.numExportedPeers, 0);
    EXPECT_EQ(m->entries->desc.local.exportedPeersCapacity, 0);
    EXPECT_EQ(m->entries->desc.local.exportedPeerRanks, nullptr);
}

// ---------------------------------------------------------------------------
// Concurrency smoke (pure state)
// ---------------------------------------------------------------------------

TEST_F(MemManagerTest, ConcurrentTrackUntrack)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    constexpr int    kThreads   = 8;
    constexpr int    kPerThread = 256;
    constexpr size_t kSize      = 64;
    std::atomic<int> trackFailures{0};
    std::atomic<int> untrackFailures{0};

    auto worker = [&](int tid) {
        for(int i = 0; i < kPerThread; ++i) {
            void* p = fakePtr(static_cast<uintptr_t>(tid) * 0x10000ULL + i + 1);
            if(ncclMemTrack(m, p, kSize, fakeHandle(), kFakeHandleType, ncclMemScratch)
               != ncclSuccess) {
                ++trackFailures;
            }
            if(ncclMemUntrack(m, p, kSize) != ncclSuccess) {
                ++untrackFailures;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for(int t = 0; t < kThreads; ++t) {
        threads.emplace_back(worker, t);
    }
    for(auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(trackFailures.load(), 0);
    EXPECT_EQ(untrackFailures.load(), 0);
    EXPECT_EQ(m->numEntries, 0);
    EXPECT_EQ(m->entries, nullptr);
    EXPECT_EQ(m->totalScratch, 0u);
}

TEST_F(MemManagerTest, ConcurrentTrackUntrackMarkExport)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    // A long-lived scratch entry that MarkExport workers hammer with disjoint
    // peer ranks. Track/Untrack workers operate on private synthetic ptrs.
    // All three paths share the same manager mutex.
    void* shared = fakePtr(0x500);
    ASSERT_EQ(ncclMemTrack(m, shared, 4096, fakeHandle(), kFakeHandleType, ncclMemScratch),
              ncclSuccess);

    constexpr int    kTrackThreads   = 4;
    constexpr int    kExportThreads  = 4;
    constexpr int    kPerThread      = 256;
    constexpr int    kRanksPerThread = 32;
    std::atomic<int> trackFailures{0};
    std::atomic<int> untrackFailures{0};
    std::atomic<int> exportFailures{0};

    auto trackWorker = [&](int tid) {
        for(int i = 0; i < kPerThread; ++i) {
            void* p = fakePtr(0x10000ULL + static_cast<uintptr_t>(tid) * 0x10000ULL + i + 1);
            if(ncclMemTrack(m, p, 64, fakeHandle(), kFakeHandleType, ncclMemScratch)
               != ncclSuccess) {
                ++trackFailures;
            }
            if(ncclMemUntrack(m, p, 64) != ncclSuccess) {
                ++untrackFailures;
            }
        }
    };

    auto exportWorker = [&](int tid) {
        for(int i = 0; i < kRanksPerThread; ++i) {
            // Unique rank across all export threads — no duplicate-rejection.
            int rank = tid * kRanksPerThread + i + 1;
            if(ncclDynMemMarkExportToPeer(m, shared, rank) != ncclSuccess) {
                ++exportFailures;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kTrackThreads + kExportThreads);
    for(int t = 0; t < kTrackThreads; ++t) {
        threads.emplace_back(trackWorker, t);
    }
    for(int t = 0; t < kExportThreads; ++t) {
        threads.emplace_back(exportWorker, t);
    }
    for(auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(trackFailures.load(), 0);
    EXPECT_EQ(untrackFailures.load(), 0);
    EXPECT_EQ(exportFailures.load(), 0);
    EXPECT_EQ(m->numEntries, 1);                 // only `shared` survived
    EXPECT_EQ(m->entries->ptr, shared);
    EXPECT_EQ(m->entries->desc.local.numExportedPeers, kExportThreads * kRanksPerThread);

    ASSERT_EQ(ncclMemUntrack(m, shared, 4096), ncclSuccess);
    EXPECT_EQ(m->numEntries, 0);
    EXPECT_EQ(m->totalScratch, 0u);
}

TEST_F(MemManagerTest, StressFuzz_RandomOps_TotalsZeroAfterTeardown)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ncclMemManager* m = comm->memManager;

    // Deterministic seed for reproducible fuzz.
    std::mt19937                       rng(0xC0FFEEu);
    std::uniform_int_distribution<int> ptrDist(0, 127);
    std::uniform_int_distribution<int> opDist(0, 99);
    std::uniform_int_distribution<int> sizeDist(1, 4096);

    struct LiveEntry
    {
        size_t        size;
        ncclMemType_t type;
        int           nextRank;
    };
    std::unordered_map<void*, LiveEntry> live;

    constexpr int kIterations = 10000;
    for(int it = 0; it < kIterations; ++it) {
        void* p  = fakePtr(0xFAFA0000ULL + ptrDist(rng));
        int   op = opDist(rng);

        auto found = live.find(p);
        if(found == live.end()) {
            // Not currently tracked → Track. Persist intentionally omitted: its
            // Untrack semantics use the passed size and a single counter, which
            // would skew the invariants we check at the end.
            ncclMemType_t type = (op < 60) ? ncclMemScratch : ncclMemOffload;
            size_t        sz   = static_cast<size_t>(sizeDist(rng));
            ASSERT_EQ(ncclMemTrack(m, p, sz, fakeHandle(), kFakeHandleType, type), ncclSuccess);
            live.emplace(p, LiveEntry{sz, type, /*nextRank=*/0});
        } else if(op < 50) {
            ASSERT_EQ(ncclMemUntrack(m, p, found->second.size), ncclSuccess);
            live.erase(found);
        } else {
            // Fresh peer rank → no duplicate rejection.
            int rank = found->second.nextRank++;
            ASSERT_EQ(ncclDynMemMarkExportToPeer(m, p, rank), ncclSuccess);
        }
    }

    // Drain whatever is still tracked.
    for(auto& [p, e] : live) {
        ASSERT_EQ(ncclMemUntrack(m, p, e.size), ncclSuccess);
    }

    EXPECT_EQ(m->numEntries, 0);
    EXPECT_EQ(m->entries, nullptr);
    EXPECT_EQ(m->totalScratch, 0u);
    EXPECT_EQ(m->totalOffload, 0u);
    EXPECT_EQ(m->totalPersist, 0u);
    EXPECT_EQ(m->cpuBackupUsage, 0u);
}

// ---------------------------------------------------------------------------
// Real GPU memory flow: hipMalloc + Track + Untrack + hipFree
// ---------------------------------------------------------------------------

TEST(MemManagerRealMem, Track_RealHipMalloc_Scratch)
{
    RUN_ISOLATED_TEST("MemManager_Track_RealHipMalloc_Scratch", []() {
        ASSERT_EQ(hipSetDevice(0), hipSuccess);

        ncclComm* comm = new ncclComm();
        comm->cudaDev  = 0;
        ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);

        constexpr size_t kSize = 4096;
        void*            p     = nullptr;
        ASSERT_EQ(hipMalloc(&p, kSize), hipSuccess);
        ASSERT_NE(p, nullptr);

        ASSERT_EQ(ncclMemTrack(comm->memManager, p, kSize, fakeHandle(), kFakeHandleType,
                               ncclMemScratch),
                  ncclSuccess);
        EXPECT_EQ(comm->memManager->numEntries, 1);
        ASSERT_NE(comm->memManager->entries, nullptr);
        EXPECT_EQ(comm->memManager->entries->ptr, p);
        EXPECT_EQ(comm->memManager->entries->size, kSize);
        EXPECT_EQ(comm->memManager->entries->memType, ncclMemScratch);
        EXPECT_EQ(comm->memManager->entries->state, ncclDynMemStateActive);
        EXPECT_EQ(comm->memManager->totalScratch, kSize);

        ASSERT_EQ(ncclMemUntrack(comm->memManager, p, kSize), ncclSuccess);
        EXPECT_EQ(comm->memManager->numEntries, 0);
        EXPECT_EQ(comm->memManager->totalScratch, 0u);

        ASSERT_EQ(hipFree(p), hipSuccess);
        ASSERT_EQ(ncclMemManagerDestroy(comm), ncclSuccess);
        delete comm;
    });
}

TEST(MemManagerRealMem, Track_RealHipMalloc_Offload)
{
    RUN_ISOLATED_TEST("MemManager_Track_RealHipMalloc_Offload", []() {
        ASSERT_EQ(hipSetDevice(0), hipSuccess);

        ncclComm* comm = new ncclComm();
        comm->cudaDev  = 0;
        ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);

        constexpr size_t kSize = 16384;
        void*            p     = nullptr;
        ASSERT_EQ(hipMalloc(&p, kSize), hipSuccess);

        ASSERT_EQ(ncclMemTrack(comm->memManager, p, kSize, fakeHandle(), kFakeHandleType,
                               ncclMemOffload),
                  ncclSuccess);
        EXPECT_EQ(comm->memManager->entries->memType, ncclMemOffload);
        EXPECT_EQ(comm->memManager->entries->cpuBackup, nullptr);
        EXPECT_EQ(comm->memManager->totalOffload, kSize);

        ASSERT_EQ(ncclMemUntrack(comm->memManager, p, kSize), ncclSuccess);
        EXPECT_EQ(comm->memManager->totalOffload, 0u);

        ASSERT_EQ(hipFree(p), hipSuccess);
        ASSERT_EQ(ncclMemManagerDestroy(comm), ncclSuccess);
        delete comm;
    });
}

TEST(MemManagerRealMem, Track_RealHipMalloc_MultipleEntries)
{
    RUN_ISOLATED_TEST("MemManager_Track_RealHipMalloc_Multiple", []() {
        ASSERT_EQ(hipSetDevice(0), hipSuccess);

        ncclComm* comm = new ncclComm();
        comm->cudaDev  = 0;
        ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);

        constexpr int    kN       = 4;
        constexpr size_t kSize    = 1024;
        void*            ptrs[kN] = {};

        for(int i = 0; i < kN; ++i) {
            ASSERT_EQ(hipMalloc(&ptrs[i], kSize), hipSuccess);
            ASSERT_EQ(ncclMemTrack(comm->memManager, ptrs[i], kSize, fakeHandle(),
                                   kFakeHandleType, ncclMemScratch),
                      ncclSuccess);
        }
        EXPECT_EQ(comm->memManager->numEntries, kN);
        EXPECT_EQ(comm->memManager->totalScratch, kN * kSize);

        // Untrack in original Track order: ptrs[0] is at the tail (Track
        // prepends), so the first Untrack walks the full list and exercises
        // the prev != nullptr branch in the linked-list removal.
        for(int i = 0; i < kN; ++i) {
            ASSERT_EQ(ncclMemUntrack(comm->memManager, ptrs[i], kSize), ncclSuccess);
        }
        EXPECT_EQ(comm->memManager->numEntries, 0);
        EXPECT_EQ(comm->memManager->totalScratch, 0u);

        for(int i = 0; i < kN; ++i) {
            ASSERT_EQ(hipFree(ptrs[i]), hipSuccess);
        }
        ASSERT_EQ(ncclMemManagerDestroy(comm), ncclSuccess);
        delete comm;
    });
}

// ---------------------------------------------------------------------------
// Real VMM flow: hipMemCreate(POSIX_FD) + Track + Untrack + hipMemUnmap/Release.
// Bypasses ncclCuMemAlloc/Free wrappers and talks to HIP VMM directly so the
// test exercises the manager against a real handle without depending on the
// RCCL allocator implementation.
// ---------------------------------------------------------------------------

namespace
{
struct VmmPosixAllocation
{
    void*                           ptr    = nullptr;
    hipDeviceptr_t                  pdev   = 0;
    size_t                          size   = 0;
    hipMemGenericAllocationHandle_t handle = 0;
};

// Allocates a chunk of device memory via the HIP VMM API with a POSIX-fd
// shareable handle. Mirrors the prop layout of ncclCuMemAlloc:
//   - Pinned + Device location
//   - requestedHandleTypes = POSIX_FILE_DESCRIPTOR
//   - allocFlags.gpuDirectRDMACapable = 1 (ROCM-2550 workaround; without it
//     hipMemMap can SIGSEGV on AMD)
// `requestedSize` is rounded up to the minimum granularity reported by HIP.
// Caller must pass the result to ReleaseVmmPosixFd().
inline void AllocateVmmPosixFd(int dev, size_t requestedSize, VmmPosixAllocation* out)
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

// Releases a VmmPosixAllocation in HIP-required order:
//   hipMemUnmap -> hipMemRelease(handle) -> hipMemAddressFree(va).
inline void ReleaseVmmPosixFd(const VmmPosixAllocation& a)
{
    ASSERT_EQ(hipMemUnmap(a.pdev, a.size), hipSuccess);
    ASSERT_EQ(hipMemRelease(a.handle), hipSuccess);
    ASSERT_EQ(hipMemAddressFree(a.pdev, a.size), hipSuccess);
}
} // namespace

TEST(MemManagerRealMem, Track_RealVmm_PosixFd)
{
    RUN_ISOLATED_TEST("MemManager_Track_RealVmm_PosixFd", []() {
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        int dev = 0;
        ASSERT_EQ(hipGetDevice(&dev), hipSuccess);

        VmmPosixAllocation va;
        AllocateVmmPosixFd(dev, /*requestedSize=*/65536, &va);
        ASSERT_NE(va.ptr, nullptr);

        ncclComm* comm = new ncclComm();
        comm->cudaDev  = dev;
        ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);

        ASSERT_EQ(ncclMemTrack(comm->memManager, va.ptr, va.size, va.handle,
                               hipMemHandleTypePosixFileDescriptor, ncclMemScratch),
                  ncclSuccess);
        EXPECT_EQ(comm->memManager->numEntries, 1);
        ASSERT_NE(comm->memManager->entries, nullptr);
        EXPECT_EQ(comm->memManager->entries->ptr, va.ptr);
        EXPECT_EQ(comm->memManager->entries->handle, va.handle);
        EXPECT_EQ(comm->memManager->entries->handleType, hipMemHandleTypePosixFileDescriptor);
        EXPECT_EQ(comm->memManager->totalScratch, va.size);

        ASSERT_EQ(ncclMemUntrack(comm->memManager, va.ptr, va.size), ncclSuccess);
        EXPECT_EQ(comm->memManager->numEntries, 0);
        EXPECT_EQ(comm->memManager->totalScratch, 0u);

        ReleaseVmmPosixFd(va);

        ASSERT_EQ(ncclMemManagerDestroy(comm), ncclSuccess);
        delete comm;
    });
}

// ---------------------------------------------------------------------------
// Real VMM allocation + MarkExport to several peers + Destroy. Exercises the
// exportedPeerRanks free path on Destroy with a real handle attached to the
// entry. We own the lifetime of the underlying VMM resources and release them
// after Destroy, since the manager only tracks bookkeeping (not the mapping).
// ---------------------------------------------------------------------------

TEST(MemManagerRealMem, MarkExport_OnRealVmmAllocation_AndDestroy)
{
    RUN_ISOLATED_TEST("MemManager_MarkExport_OnRealVmmAllocation_AndDestroy", []() {
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        int dev = 0;
        ASSERT_EQ(hipGetDevice(&dev), hipSuccess);

        VmmPosixAllocation va;
        AllocateVmmPosixFd(dev, /*requestedSize=*/65536, &va);
        ASSERT_NE(va.ptr, nullptr);

        ncclComm* comm = new ncclComm();
        comm->cudaDev  = dev;
        ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);

        ASSERT_EQ(ncclMemTrack(comm->memManager, va.ptr, va.size, va.handle,
                               hipMemHandleTypePosixFileDescriptor, ncclMemScratch),
                  ncclSuccess);
        for(int peer : {1, 2, 3, 5, 8, 13}) {
            ASSERT_EQ(ncclDynMemMarkExportToPeer(comm->memManager, va.ptr, peer), ncclSuccess);
        }
        EXPECT_EQ(comm->memManager->entries->desc.local.numExportedPeers, 6);

        ASSERT_EQ(ncclMemManagerDestroy(comm), ncclSuccess);
        EXPECT_EQ(comm->memManager, nullptr);
        delete comm;

        // Caller still owns the VMM resources — manager only tracks
        // bookkeeping, not the mapping itself.
        ReleaseVmmPosixFd(va);
    });
}

// ---------------------------------------------------------------------------
// Cleanup branches: cpuBackup free + shareableHandle.fd close.
// We populate these fields manually to mirror what ncclCommSuspend (PR5) and
// ncclCuMemAlloc (PR2) will set, and verify the manager's destructors release
// the underlying real resources (cudaFreeHost / close).
// ---------------------------------------------------------------------------

TEST(MemManagerRealMem, Untrack_FreesCpuBackup)
{
    RUN_ISOLATED_TEST("MemManager_Untrack_FreesCpuBackup", []() {
        ASSERT_EQ(hipSetDevice(0), hipSuccess);

        ncclComm* comm = new ncclComm();
        comm->cudaDev  = 0;
        ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);

        constexpr size_t kSize = 4096;
        void*            p     = fakePtr(0x100);
        ASSERT_EQ(ncclMemTrack(comm->memManager, p, kSize, fakeHandle(), kFakeHandleType,
                               ncclMemOffload),
                  ncclSuccess);

        // Simulate ncclCommSuspend attaching a host-side backup buffer.
        char* backup = nullptr;
        ASSERT_EQ(ncclCudaHostCalloc(&backup, kSize), ncclSuccess);
        ASSERT_NE(backup, nullptr);
        comm->memManager->entries->cpuBackup = backup;
        comm->memManager->cpuBackupUsage += kSize;

        ASSERT_EQ(ncclMemUntrack(comm->memManager, p, kSize), ncclSuccess);
        // Manager calls ncclCudaHostFree(backup) and decrements usage.
        // Address sanitizer would catch a double-free or leak here.
        EXPECT_EQ(comm->memManager->cpuBackupUsage, 0u);

        ASSERT_EQ(ncclMemManagerDestroy(comm), ncclSuccess);
        delete comm;
    });
}

TEST(MemManagerRealMem, Untrack_ClosesShareableFd)
{
    RUN_ISOLATED_TEST("MemManager_Untrack_ClosesShareableFd", []() {
        ASSERT_EQ(hipSetDevice(0), hipSuccess);

        ncclComm* comm = new ncclComm();
        comm->cudaDev  = 0;
        ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);

        constexpr size_t kSize = 4096;
        void*            p     = fakePtr(0x101);
        ASSERT_EQ(ncclMemTrack(comm->memManager, p, kSize, fakeHandle(),
                               hipMemHandleTypePosixFileDescriptor, ncclMemScratch),
                  ncclSuccess);

        // Simulate ncclCuMemAlloc populating the shareable POSIX fd.
        int fd = dup(STDOUT_FILENO);
        ASSERT_GE(fd, 0);
        ASSERT_NE(fcntl(fd, F_GETFD), -1);

        comm->memManager->entries->desc.local.shareableHandle.fd   = fd;
        comm->memManager->entries->desc.local.shareableHandleValid = true;

        ASSERT_EQ(ncclMemUntrack(comm->memManager, p, kSize), ncclSuccess);

        // fd must be closed by the manager.
        errno = 0;
        EXPECT_EQ(fcntl(fd, F_GETFD), -1);
        EXPECT_EQ(errno, EBADF);

        ASSERT_EQ(ncclMemManagerDestroy(comm), ncclSuccess);
        delete comm;
    });
}

TEST(MemManagerRealMem, Destroy_FreesPopulatedEntries)
{
    RUN_ISOLATED_TEST("MemManager_Destroy_FreesPopulatedEntries", []() {
        ASSERT_EQ(hipSetDevice(0), hipSuccess);

        ncclComm* comm = new ncclComm();
        comm->cudaDev  = 0;
        ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);

        constexpr size_t kSize = 4096;

        // Entry 1: scratch with cpuBackup (simulates suspended scratch buffer).
        void* p1 = fakePtr(0x200);
        ASSERT_EQ(ncclMemTrack(comm->memManager, p1, kSize, fakeHandle(),
                               hipMemHandleTypePosixFileDescriptor, ncclMemScratch),
                  ncclSuccess);
        char* backup1 = nullptr;
        ASSERT_EQ(ncclCudaHostCalloc(&backup1, kSize), ncclSuccess);
        comm->memManager->entries->cpuBackup = backup1;
        comm->memManager->cpuBackupUsage += kSize;

        // Entry 2: offload with a real shareable fd. Track prepends, so this
        // becomes the new head and we patch desc.local on the head entry.
        void* p2 = fakePtr(0x201);
        ASSERT_EQ(ncclMemTrack(comm->memManager, p2, kSize, fakeHandle(),
                               hipMemHandleTypePosixFileDescriptor, ncclMemOffload),
                  ncclSuccess);
        int fd2 = dup(STDOUT_FILENO);
        ASSERT_GE(fd2, 0);
        comm->memManager->entries->desc.local.shareableHandle.fd   = fd2;
        comm->memManager->entries->desc.local.shareableHandleValid = true;

        // Entry 3: scratch with exported peers (exercises exportedPeerRanks free).
        void* p3 = fakePtr(0x202);
        ASSERT_EQ(ncclMemTrack(comm->memManager, p3, kSize, fakeHandle(),
                               hipMemHandleTypePosixFileDescriptor, ncclMemScratch),
                  ncclSuccess);
        ASSERT_EQ(ncclDynMemMarkExportToPeer(comm->memManager, p3, 1), ncclSuccess);
        ASSERT_EQ(ncclDynMemMarkExportToPeer(comm->memManager, p3, 2), ncclSuccess);

        EXPECT_EQ(comm->memManager->numEntries, 3);

        // Destroy walks the full list and releases every attached resource.
        ASSERT_EQ(ncclMemManagerDestroy(comm), ncclSuccess);
        EXPECT_EQ(comm->memManager, nullptr);

        // fd2 must now be closed.
        errno = 0;
        EXPECT_EQ(fcntl(fd2, F_GETFD), -1);
        EXPECT_EQ(errno, EBADF);

        delete comm;
    });
}

// ---------------------------------------------------------------------------
// Suspend / Resume - early-return error paths (no HIP, no bootstrap)
//
// All of these exercise the validation block at the top of ncclCommMemSuspend /
// ncclCommMemResume, which returns before touching any HIP API or bootstrap.
// ---------------------------------------------------------------------------

TEST_F(MemManagerTest, Suspend_NullComm)
{
    EXPECT_EQ(ncclCommMemSuspend(nullptr), ncclInvalidArgument);
}

TEST_F(MemManagerTest, Suspend_NullManager)
{
    ASSERT_EQ(comm->memManager, nullptr);
    EXPECT_EQ(ncclCommMemSuspend(comm), ncclInvalidUsage);
}

TEST_F(MemManagerTest, Suspend_AlreadyReleased)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    comm->memManager->released = 1;
    EXPECT_EQ(ncclCommMemSuspend(comm), ncclInvalidUsage);
}

TEST_F(MemManagerTest, Resume_NullComm)
{
    EXPECT_EQ(ncclCommMemResume(nullptr), ncclInvalidArgument);
}

TEST_F(MemManagerTest, Resume_NullManager)
{
    ASSERT_EQ(comm->memManager, nullptr);
    EXPECT_EQ(ncclCommMemResume(comm), ncclInvalidUsage);
}

TEST_F(MemManagerTest, Resume_NotReleased)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    ASSERT_EQ(comm->memManager->released, 0);
    EXPECT_EQ(ncclCommMemResume(comm), ncclInvalidUsage);
}

// ---------------------------------------------------------------------------
// MemStats fixture: ncclCommMemStats calls CommCheck + ncclCommEnsureReady, so
// the comm needs valid magic markers and a non-null abortFlag. We still avoid
// any HIP/bootstrap activity - all paths in MemStats are purely manager-state.
// ---------------------------------------------------------------------------

class MemManagerStatsTest : public ::testing::Test
{
protected:
    ncclComm* comm     = nullptr;
    uint32_t  abortVal = 0;

    void SetUp() override
    {
        comm             = new ncclComm();
        comm->cudaDev    = 0;
        comm->startMagic = NCCL_MAGIC;
        comm->endMagic   = NCCL_MAGIC;
        comm->abortFlag  = &abortVal;
    }

    void TearDown() override
    {
        if(comm && comm->memManager) {
            ncclMemManagerDestroy(comm);
        }
        delete comm;
        comm = nullptr;
    }

    uint64_t readStat(ncclCommMemStat_t s)
    {
        uint64_t v = ~0ULL;
        EXPECT_EQ(ncclCommMemStats(comm, s, &v), ncclSuccess);
        return v;
    }
};

TEST_F(MemManagerStatsTest, Empty)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);

    EXPECT_EQ(readStat(ncclStatGpuMemTotal), 0u);
    EXPECT_EQ(readStat(ncclStatGpuMemPersist), 0u);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspend), 0u);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspended), 0u);
}

TEST_F(MemManagerStatsTest, Counters_PersistScratchOffload)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);

    constexpr size_t kPersist = 1024;
    constexpr size_t kScratch = 4096;
    constexpr size_t kOffload = 2048;

    ASSERT_EQ(ncclMemTrack(comm->memManager, fakePtr(1), kPersist,
                           fakeHandle(), kFakeHandleType, ncclMemPersist),
              ncclSuccess);
    ASSERT_EQ(ncclMemTrack(comm->memManager, fakePtr(2), kScratch,
                           fakeHandle(), kFakeHandleType, ncclMemScratch),
              ncclSuccess);
    ASSERT_EQ(ncclMemTrack(comm->memManager, fakePtr(3), kOffload,
                           fakeHandle(), kFakeHandleType, ncclMemOffload),
              ncclSuccess);

    EXPECT_EQ(readStat(ncclStatGpuMemPersist), kPersist);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspend), kScratch + kOffload);
    EXPECT_EQ(readStat(ncclStatGpuMemTotal),   kPersist + kScratch + kOffload);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspended), 0u);

    ASSERT_EQ(ncclMemUntrack(comm->memManager, fakePtr(2), kScratch), ncclSuccess);
    ASSERT_EQ(ncclMemUntrack(comm->memManager, fakePtr(3), kOffload), ncclSuccess);

    EXPECT_EQ(readStat(ncclStatGpuMemSuspend), 0u);
    EXPECT_EQ(readStat(ncclStatGpuMemPersist), kPersist);
    EXPECT_EQ(readStat(ncclStatGpuMemTotal),   kPersist);
}

TEST_F(MemManagerStatsTest, NullManager_ReturnsZero)
{
    ASSERT_EQ(comm->memManager, nullptr);
    uint64_t v = ~0ULL;
    EXPECT_EQ(ncclCommMemStats(comm, ncclStatGpuMemTotal, &v), ncclSuccess);
    EXPECT_EQ(v, 0u);
}

TEST_F(MemManagerStatsTest, NullValue_Rejected)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    EXPECT_EQ(ncclCommMemStats(comm, ncclStatGpuMemTotal, nullptr),
              ncclInvalidArgument);
}

TEST_F(MemManagerStatsTest, InvalidStat_Rejected)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    uint64_t v = 0;
    EXPECT_EQ(ncclCommMemStats(comm, static_cast<ncclCommMemStat_t>(99), &v),
              ncclInvalidArgument);
}

TEST_F(MemManagerStatsTest, SuspendedFlag_FlipsWithReleased)
{
    ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    EXPECT_EQ(readStat(ncclStatGpuMemSuspended), 0u);

    // Simulate ncclCommMemSuspend's only externally observable effect on stats
    // without paying for the bootstrap+VMM round-trip (covered by the MPI suite).
    comm->memManager->released = 1;
    EXPECT_EQ(readStat(ncclStatGpuMemSuspended), 1u);

    comm->memManager->released = 0;
    EXPECT_EQ(readStat(ncclStatGpuMemSuspended), 0u);
}

} // namespace RcclUnitTesting
