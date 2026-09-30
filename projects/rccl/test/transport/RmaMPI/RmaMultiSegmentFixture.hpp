/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_RMA_MULTISEGMENT_FIXTURE_HPP
#define RCCL_TEST_RMA_MULTISEGMENT_FIXTURE_HPP

#ifdef MPI_TESTS_ENABLED
#ifdef RCCL_HAS_RMA_IB_PROXY

#include "RmaMPITestBase.hpp"
#include "RmaMultiSegmentHelpers.hpp"
#include "HybridVmmHelpers.hpp"
#include "MPIHelpers.hpp"
#include "../../../src/transport/net_ib/gin.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <strings.h>
#include <vector>

namespace RCCLRmaTests
{

inline constexpr size_t kSegRequestBytes = 2u * 1024 * 1024;
inline constexpr int    kNumSegments     = 4;
inline constexpr size_t kSignalSize      = 64;
inline constexpr size_t kMiB             = 1024u * 1024;

// INFO marker emitted by the backend when the per-segment path fires.
inline constexpr const char* kMultiSegMarker = "multi-segment buffer";

// Edge-case payload sizes from 0 up to `maxBytes`, anchored around byte/word,
// page (4K), 64K, and the per-segment boundary `seg`. Deduplicated + sorted.
inline std::vector<size_t> EdgeCaseSizes(size_t seg, size_t maxBytes)
{
    std::vector<size_t> v;
    auto add = [&](size_t s) { if (s <= maxBytes) v.push_back(s); };
    for (size_t s : {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{7},
                     size_t{63}, size_t{64}, size_t{65}, size_t{255}, size_t{256},
                     size_t{4095}, size_t{4096}, size_t{4097},
                     size_t{65535}, size_t{65536}, size_t{65537}})
        add(s);
    // Per-segment boundary neighbourhood (the split points under test).
    if (seg >= 1)     { add(seg - 1); add(seg); add(seg + 1); add(seg + 4096); }
    if (2 * seg >= 1) { add(2 * seg - 1); add(2 * seg); add(2 * seg + 1); }
    add(3 * seg);
    add(maxBytes ? maxBytes - 1 : 0);
    add(maxBytes);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

// RMA proxy fixture + NCCL INFO log capture to confirm the per-segment path
// fired (vs single-MR fallback when cuMem enumeration is unavailable).
class RmaMultiSegmentMPITest : public RmaMPITestBase
{
protected:
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             cuMemGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugSubsysGuard_;
    std::unique_ptr<MPIHelpers::TestLogAssertionContext> logCtx_;

    int GetNumContexts() const override { return 1; }

    void SetUp() override
    {
        // Per-segment enumeration needs the cuMem path; the marker gate below
        // covers cases where the param was already cached process-wide.
        cuMemGuard_       = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_CUMEM_ENABLE",  "1");
        debugGuard_       = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG",         "INFO");
        debugSubsysGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG_SUBSYS",  "ALL");

        RmaMPITestBase::SetUp();

        logCtx_ = std::make_unique<MPIHelpers::TestLogAssertionContext>(
            MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));
    }

    void TearDown() override
    {
        // Deregister IB MRs (base TearDown) BEFORE releasing their backing VMM;
        // freeing VMM under a live DMA-BUF MR aborts/stalls cleanup (AIRUNTIME-2351).
        RmaMPITestBase::TearDown();
        for (auto& b : vmmBuffers_)
            FreeMultiSegmentVmm(*b);
        vmmBuffers_.clear();
        for (auto& b : hybridBuffers_)
            RCCLHybridVmmTests::FreeHybridVmm(*b);
        hybridBuffers_.clear();
        logCtx_.reset();
        debugSubsysGuard_.reset();
        debugGuard_.reset();
        cuMemGuard_.reset();
    }

    std::string readAllLogs() const
    {
        if (!logCtx_) return {};
        return logCtx_->readNcclDebugLog() + logCtx_->readPerRankStderrLog();
    }

    // Collective skip: if ANY rank wants to skip, all ranks return true so they
    // GTEST_SKIP together (a unilateral skip would hang peers).
    bool SyncSkip(bool wantSkip)
    {
        return MPIHelpers::anyRankTrue(wantSkip);
    }

    // True only if EVERY rank observed the per-segment registration marker.
    bool AllTookMultiSegPath()
    {
        return MPIHelpers::allRanksTrue(
            readAllLogs().find(kMultiSegMarker) != std::string::npos);
    }

    // Allocate a fixture-owned N-segment VMM window (freed in TearDown after MR
    // dereg). Returns nullptr on failure so the caller can SyncSkip. Uses the
    // rank's CURRENT GPU (round-robin assigned by the harness), not the IB-device
    // index defaultDevice_, or rank>0 would fault touching dev-0 memory.
    MultiSegmentVmmBuffer* AllocSym(int nSegments, size_t segBytes)
    {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess)
            return nullptr;
        auto buf = std::make_unique<MultiSegmentVmmBuffer>();
        if (!AllocMultiSegmentVmm(dev, nSegments, segBytes, buf.get()))
            return nullptr;
        vmmBuffers_.push_back(std::move(buf));
        return vmmBuffers_.back().get();
    }

    MultiSegmentVmmBuffer* AllocDeepEpElastic(size_t gpuBytes, size_t cpuBytes)
    {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess)
            return nullptr;
        auto buf = std::make_unique<MultiSegmentVmmBuffer>();
        if (!AllocDeepEpElasticVmm(dev, gpuBytes, cpuBytes, buf.get()))
            return nullptr;
        vmmBuffers_.push_back(std::move(buf));
        return vmmBuffers_.back().get();
    }

    RCCLHybridVmmTests::HybridVmmBuffer* AllocHybrid(
        size_t gpuBytes, size_t localCpuBytes, std::string* reason)
    {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess)
            return nullptr;
        if (!RCCLHybridVmmTests::CheckHybridVmmRuntimeSupport(dev, reason))
            return nullptr;
        auto buf = std::make_unique<RCCLHybridVmmTests::HybridVmmBuffer>();
        if (!RCCLHybridVmmTests::AllocHybridVmm(
                dev, gpuBytes, localCpuBytes, buf.get(), reason))
            return nullptr;
        hybridBuffers_.push_back(std::move(buf));
        return hybridBuffers_.back().get();
    }

    bool AllocHybridForLocalRanks(
        size_t gpuBytes, size_t localCpuBytes, int expectedLocalRanks,
        RCCLHybridVmmTests::HybridVmmBuffer** out, std::string* reason)
    {
        *out = nullptr;
        int dev = 0;
        std::string localReason;
        // The support check runs collectives; call it even if hipGetDevice failed.
        const bool haveDev = hipGetDevice(&dev) == hipSuccess;
        bool supported = RCCLHybridVmmTests::CheckHybridVmmRuntimeSupport(dev, reason) && haveDev;
        if (SyncSkip(!supported)) {
            if (reason && reason->empty())
                *reason = "hybrid VMM runtime support is unavailable on another rank";
            return false;
        }
        auto buf = std::make_unique<RCCLHybridVmmTests::HybridVmmBuffer>();
        const bool ok = RCCLHybridVmmTests::AllocHybridForLocalRanks(
            dev, gpuBytes, localCpuBytes, expectedLocalRanks, buf.get(), &localReason);
        if (reason && reason->empty())
            *reason = localReason;
        // TearDown is the only FreeHybridVmm for a successful alloc. A peer skip
        // after this rank succeeded used to destroy the unique_ptr without that call.
        if (ok) hybridBuffers_.push_back(std::move(buf));
        if (SyncSkip(!ok)) {
            if (reason && reason->empty())
                *reason = "hybrid VMM allocation failed on another rank";
            return false;
        }
        *out = hybridBuffers_.back().get();
        return true;
    }

    bool AllocSymPair(MultiSegmentVmmBuffer** src, MultiSegmentVmmBuffer** dst,
                      int nSegments = kNumSegments,
                      size_t segBytes = kSegRequestBytes)
    {
        *src = AllocSym(nSegments, segBytes);
        *dst = AllocSym(nSegments, segBytes);
        return !SyncSkip(*src == nullptr || *dst == nullptr);
    }

    // Skip only when no rank took the per-segment path. A mixed result is a
    // failure: SyncSkip(ANY miss) would hide a unilateral-success bug.
    bool MultiSegmentPathAvailable()
    {
        const bool local =
            readAllLogs().find(kMultiSegMarker) != std::string::npos;
        if (!MPIHelpers::anyRankTrue(local)) return false;
        if (!AllTookMultiSegPath()) {
            ADD_FAILURE() << "multi-segment registration was asymmetric across ranks";
            return true;
        }
        return true;
    }

    void ExpectPayloadIsolated(const void* window, size_t totalSize,
                               size_t offset, size_t size, int seed,
                               uint8_t sentinel, const std::string& context)
    {
        SCOPED_TRACE(context);
        const auto* bytes = static_cast<const uint8_t*>(window);
        EXPECT_TRUE(VerifyBuf(bytes + offset, size, seed));
        EXPECT_TRUE(AllSentinel(window, offset, sentinel));
        EXPECT_TRUE(AllSentinel(bytes + offset + size,
                                totalSize - offset - size, sentinel));
    }

    void RunIPutSizeSweep(MultiSegmentVmmBuffer* src,
                          MultiSegmentVmmBuffer* dst,
                          void* srcMh, void* dstMh, size_t offset,
                          uint8_t seedBase, uint8_t sentinel)
    {
        const size_t total = src->totalSize;
        const size_t seg = src->segSize;
        const std::vector<size_t> sizes = EdgeCaseSizes(seg, total - offset);
        for (size_t idx = 0; idx < sizes.size(); ++idx)
        {
            const size_t size = sizes[idx];
            const uint8_t seed =
                static_cast<uint8_t>(seedBase + (idx & 0x3F));
            const std::string context =
                "size=" + std::to_string(size) +
                " offset=" + std::to_string(offset);
            SCOPED_TRACE(context);

            if (worldRank_ == 0 && size > 0)
                FillBuf(static_cast<uint8_t*>(src->ptr) + offset, size, seed);
            if (worldRank_ == 1)
                FillSentinel(dst->ptr, total, sentinel);

            Barrier();
            bool putOk = true;
            if (worldRank_ == 0)
            {
                void* req = nullptr;
                putOk = rma_->iput(rmaCtx_, 0, offset, srcMh, size,
                                   offset, dstMh, 1, ncclRmaOptFlagsDefault, &req) == ncclSuccess;
                if (putOk) putOk = PollUntilDone(req);
            }
            if (!MPIHelpers::allRanksTrue(putOk))
            {
                ADD_FAILURE() << "iput sweep failed " << context;
                return;
            }
            Barrier();

            if (worldRank_ == 1)
                ExpectPayloadIsolated(dst->ptr, total, offset, size,
                                      seed, sentinel, context);
            Barrier();
        }
    }

    std::vector<std::unique_ptr<MultiSegmentVmmBuffer>> vmmBuffers_;
    std::vector<std::unique_ptr<RCCLHybridVmmTests::HybridVmmBuffer>> hybridBuffers_;
};

} // namespace RCCLRmaTests

#endif // RCCL_HAS_RMA_IB_PROXY
#endif // MPI_TESTS_ENABLED
#endif // RCCL_TEST_RMA_MULTISEGMENT_FIXTURE_HPP
