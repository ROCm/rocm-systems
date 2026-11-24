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
#include "HybridVmmHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MultiSegmentTestSupport.hpp"
#include "MultiSegmentVmmHelpers.hpp"
#include "transport/net_ib_multiseg/ib_multiseg_mseg.h"
#include "rccl_common.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <strings.h>
#include <vector>

namespace RCCLRmaTests
{

using RCCLTestHelpers::BoundaryStress;
using RCCLTestHelpers::MsegPrereq;
using RCCLTestHelpers::MultiSegmentVmmBuffer;
using RCCLTestHelpers::OwnedVmmPool;

constexpr size_t kSegRequestBytes = 2u * 1024 * 1024;

// Segments in a wide test window, more than any fixed per-handle table would hold.
constexpr int kWideSegments = 17;
constexpr int    kNumSegments     = 4;
constexpr size_t kSignalSize      = 64;
constexpr size_t kMiB             = 1024u * 1024;

// DeepEP hybrid window: one GPU segment plus one CPU segment per local rank.
constexpr size_t kHybridGpuBytes  = 8 * kMiB;

constexpr size_t kHybridCpuBytes  = 2 * kMiB;

constexpr const char* kVmmAllocFailed = "a multi-segment VMM allocation failed";

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
    {
        add(s);
    }
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

// Iteration i of a DeepEP IGet stress: payload size, offset inside a CPU segment, and GPU destination offset.
struct StressStep
{
    size_t len, cpuOff, gpuOff;
};

inline StressStep StressOffsets(int i, size_t gpuBytes, size_t cpuBytes)
{
    static constexpr size_t kStressSizes[] = {1, 63, 4095, 4096, 65535, 65536, 131072, 262144};
    const size_t n = static_cast<size_t>(i);
    const size_t len = kStressSizes[n % (sizeof(kStressSizes) / sizeof(kStressSizes[0]))];
    return {len, 4096 + (n * 131071) % (cpuBytes - len - 4096), 65536 + (n * 65537) % (gpuBytes - len - 65536)};
}

// A full-size transfer one byte too long, then shifted one byte past the end on either side.
struct OutOfRangeCase
{
    size_t      firstOff, size, secondOff;
    const char* why;
};

inline std::vector<OutOfRangeCase> OutOfRangeCases(size_t total)
{
    return {{0, total + 1, 0, "oversized transfer"},
            {1, total, 0, "source-side offset overrun"},
            {0, total, 1, "destination-side offset overrun"}};
}

struct SignalWord
{
    uint64_t off, value;
};

// Returns on a failure and skips on the reason a Prepare* helper gives.
#define RMA_MSEG_PREPARE_OR_SKIP(expr)                    \
    do                                                    \
    {                                                     \
        const std::string skipReason_ = (expr);           \
        if (HasFailure()) return;                         \
        if (!skipReason_.empty()) GTEST_SKIP() << skipReason_; \
    } while (0)

// RMA proxy fixture. The per-segment path is the MR handle's segment count.
class RmaMultiSegmentMPITest : public RmaMPITestBase
{
protected:
    std::unique_ptr<MPIHelpers::MpiEnvGuard> cuMemGuard_;
    // RCCL_TEST_RMA_MSEG_REQUIRE=1 (CI) turns skips that would hide the multi-segment path into failures.
    const MsegPrereq prereq_{"RCCL_TEST_RMA_MSEG_REQUIRE"};

    int GetNumContexts() const override { return 1; }

    void OnMissingRmaPrereq(const char* what) override { prereq_.FailIfRequired(what); }

    void SetUp() override
    {
        // Per-segment enumeration needs cuMem. NCCL_PARAM caches it for the process.
        cuMemGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_CUMEM_ENABLE", "1");
        RmaMPITestBase::SetUp();
        // The segment count is read from IB-MSEG MR handles only.
        if (rma_ != nullptr && rma_ != &IbMsegRmaIbProxy)
        {
            prereq_.FailIfRequired("the IB-MSEG RMA backend is not selected");
            GTEST_SKIP() << "requires the IB-MSEG RMA backend (NCCL_NET=IB-MSEG or NCCL_MULTI_SEGMENT_REGISTER=1)";
        }
    }

    void TearDown() override
    {
        // Deregister IB MRs (base TearDown) BEFORE releasing their backing VMM;
        // freeing VMM under a live DMA-BUF MR aborts/stalls cleanup (AIRUNTIME-2351).
        RmaMPITestBase::TearDown();
        pool_.Release();
        for (auto& b : hybridBuffers_)
        {
            for (int fd : b->segFds)
            {
                if (fd >= 0) rcclHostSegmentFdDeregister(fd);
            }
            RCCLHybridVmmTests::FreeHybridVmm(*b);
        }
        hybridBuffers_.clear();
        cuMemGuard_.reset();
    }

    // True when a registration on this rank returned more than one segment.
    bool RegisteredMultiSegment() const
    {
        for (void* mh : registeredMhandles_)
        {
            if (rcclRmaIbProxyMrSegmentCount(mh) > 1) return true;
        }
        return false;
    }

    // True only if EVERY rank registered more than one segment.
    bool AllTookMultiSegPath()
    {
        return MPIHelpers::allRanksTrue(RegisteredMultiSegment());
    }

    // TearDown frees the buffer; AllocHybridCollective already freed it on a collective failure.
    RCCLHybridVmmTests::HybridVmmStatus AllocFixtureHybrid(
        size_t gpuBytes, size_t localCpuBytes, int expectedLocalRanks,
        RCCLHybridVmmTests::HybridVmmBuffer** out, std::string* reason)
    {
        *out = nullptr;
        auto buf = std::make_unique<RCCLHybridVmmTests::HybridVmmBuffer>();
        const auto status = RCCLHybridVmmTests::AllocHybridCollective(
            gpuBytes, localCpuBytes, expectedLocalRanks, buf.get(), reason);
        if (status != RCCLHybridVmmTests::HybridVmmStatus::Ok) return status;
        // Plugin registration has no window, so the importer publishes the owner's DMA-BUF fds.
        uintptr_t va = reinterpret_cast<uintptr_t>(buf->ptr);
        for (size_t s = 0; s < buf->segSizes.size(); ++s) {
            if (buf->segFds[s] >= 0)
            {
                rcclHostSegmentFdRegister(reinterpret_cast<void*>(va), buf->segSizes[s], buf->segFds[s]);
            }
            va += buf->segSizes[s];
        }
        hybridBuffers_.push_back(std::move(buf));
        *out = hybridBuffers_.back().get();
        return status;
    }

    bool AllocSymPair(MultiSegmentVmmBuffer** src, MultiSegmentVmmBuffer** dst,
                      int nSegments = kNumSegments,
                      size_t segBytes = kSegRequestBytes)
    {
        *src = pool_.AllocUniform(nSegments, segBytes);
        *dst = pool_.AllocUniform(nSegments, segBytes);
        return !prereq_.Skip(*src == nullptr || *dst == nullptr, kVmmAllocFailed);
    }

    // Skip only when no rank took the per-segment path; a mixed result fails, as does any skip under
    // RCCL_TEST_RMA_MSEG_REQUIRE=1.
    bool MultiSegmentPathAvailable()
    {
        const bool local = RegisteredMultiSegment();
        if (!MPIHelpers::anyRankTrue(local))
        {
            prereq_.FailIfRequired("no rank took the multi-segment path");
            return false;
        }
        if (!AllTookMultiSegPath())
        {
            ADD_FAILURE() << "multi-segment registration was asymmetric across ranks";
            return true;
        }
        return true;
    }

    // Allocates and registers a symmetric pair over its whole size. Returns a skip reason, or "" when ready;
    // requirePath also skips unless every rank took the per-segment path.
    std::string PreparePair(MultiSegmentVmmBuffer** src, MultiSegmentVmmBuffer** dst, void** srcMh, void** dstMh,
                            int nSegments = kNumSegments, bool requirePath = true)
    {
        if (!AllocSymPair(src, dst, nSegments)) return "Multi-segment VMM allocation unavailable on this host";
        EXPECT_EQ(ncclSuccess, RegMr((*src)->ptr, (*src)->totalSize, srcMh));
        EXPECT_EQ(ncclSuccess, RegMr((*dst)->ptr, (*dst)->totalSize, dstMh));
        if (HasFailure()) return "registration failed";
        if (requirePath && !MultiSegmentPathAvailable()) return "multi-segment path not exercised on this host";
        return "";
    }

    // Registers a 2-segment window to confirm the per-segment path is live, else a negative check is moot.
    std::string ProbeMultiSegmentPath()
    {
        MultiSegmentVmmBuffer* probe = pool_.AllocUniform(2, kSegRequestBytes);
        if (prereq_.Skip(probe == nullptr, kVmmAllocFailed)) return "multi-segment VMM allocation unavailable on this host";
        void* mh = nullptr;
        EXPECT_EQ(ncclSuccess, RegMr(probe->ptr, probe->totalSize, &mh));
        return MultiSegmentPathAvailable() ? "" : "multi-segment path not exercised on this host";
    }

    // Registers a DeepEP [GPU][CPU] window over its whole size on the per-segment path.
    std::string PrepareElastic(size_t gpuBytes, size_t cpuBytes, MultiSegmentVmmBuffer** window, void** mh)
    {
        *window = pool_.AllocDeepEpElastic(gpuBytes, cpuBytes);
        if (prereq_.Skip(*window == nullptr, kVmmAllocFailed))
            return "DeepEP-style GPU+CPU VMM allocation unavailable on this runtime";
        EXPECT_EQ(ncclSuccess, RegMr((*window)->ptr, (*window)->totalSize, mh));
        return MultiSegmentPathAvailable() ? "" : "DeepEP window did not take the multi-segment RMA registration path";
    }

    // Registers a hybrid window from AllocFixtureHybrid; requirePath skips unless it took the per-segment path.
    std::string PrepareHybrid(RCCLHybridVmmTests::HybridVmmBuffer* window, void** mh, bool requirePath = true)
    {
        EXPECT_EQ(ncclSuccess, RegMr(window->ptr, window->totalSize, mh));
        if (HasFailure()) return "registration failed";
        if (requirePath && !MultiSegmentPathAvailable()) return "hybrid window did not take the multi-segment RMA path";
        return "";
    }

    // A zeroed, registered signal window of nSeg segments.
    std::string PrepareSignalWindow(int nSeg, MultiSegmentVmmBuffer** sw, void** mh)
    {
        *sw = pool_.AllocUniform(nSeg, kSegRequestBytes);
        if (prereq_.Skip(*sw == nullptr, kVmmAllocFailed)) return "multi-segment signal window allocation unavailable";
        FillSentinel((*sw)->ptr, (*sw)->totalSize, 0);
        EXPECT_EQ(ncclSuccess, RegMr((*sw)->ptr, (*sw)->totalSize, mh));
        return "";
    }

    // Each listed word of `sw` holds its value and every other word is still zero.
    void ExpectOnlySignals(const MultiSegmentVmmBuffer* sw, const std::vector<SignalWord>& want)
    {
        std::vector<uint64_t> words(sw->totalSize / sizeof(uint64_t));
        ASSERT_EQ(hipSuccess, hipMemcpy(words.data(), sw->ptr, sw->totalSize, hipMemcpyDeviceToHost));
        for (const SignalWord& w : want)
        {
            EXPECT_EQ(words[w.off / sizeof(uint64_t)], w.value) << "signal word at offset " << w.off;
            words[w.off / sizeof(uint64_t)] = 0;
        }
        size_t stray = 0, firstStray = 0;
        for (size_t i = 0; i < words.size(); i++)
        {
            if (words[i] != 0 && stray++ == 0) firstStray = i * sizeof(uint64_t);
        }
        EXPECT_EQ(stray, 0u) << "stray signal writes; first at offset " << firstStray;
    }

    // Rank 0 puts to, or gets from, rank 1; src holds the data either way (the remote side of a get).
    ncclResult_t PostToRank1(bool get, size_t srcOff, void* srcMh, size_t size, size_t dstOff, void* dstMh,
                             void** req)
    {
        *req = nullptr;
        return get ? rma_->iget(rmaCtx_, 0, srcOff, srcMh, size, dstOff, dstMh, 1, ncclRmaOptFlagsDefault, req)
                   : rma_->iput(rmaCtx_, 0, srcOff, srcMh, size, dstOff, dstMh, 1, ncclRmaOptFlagsDefault, req);
    }

    // Rank 0 transfers and waits, between barriers; fails on every rank when it failed.
    void Rank0Xfer(bool get, size_t srcOff, void* srcMh, size_t size, size_t dstOff, void* dstMh)
    {
        Barrier();
        bool ok = true;
        if (worldRank_ == 0)
        {
            void* req = nullptr;
            const ncclResult_t r = PostToRank1(get, srcOff, srcMh, size, dstOff, dstMh, &req);
            EXPECT_EQ(ncclSuccess, r) << (get ? "iget" : "iput") << " post failed";
            ok = r == ncclSuccess && PollUntilDone(req);
        }
        ASSERT_TRUE(MPIHelpers::allRanksTrue(ok)) << (get ? "iget" : "iput") << " failed on at least one rank";
        Barrier();
    }
    void Rank0Put(size_t srcOff, void* srcMh, size_t size, size_t dstOff, void* dstMh)
    {
        Rank0Xfer(false, srcOff, srcMh, size, dstOff, dstMh);
    }

    // The source rank fills src[srcOff, +size) with `seed` and the target sentinels the first `dstChecked` bytes of
    // dst (0: all); after rank 0's transfer (and a rank-1 flush when `flushOff` is set) the payload must land alone.
    void Rank0Round(bool get, const MultiSegmentVmmBuffer* src, void* srcMh, size_t srcOff, size_t size,
                    const MultiSegmentVmmBuffer* dst, void* dstMh, size_t dstOff, int seed, uint8_t sentinel,
                    const std::string& context, size_t dstChecked = 0, bool* flushOff = nullptr)
    {
        SCOPED_TRACE(context);
        const int target = get ? 0 : 1;
        const size_t checked = dstChecked ? dstChecked : dst->totalSize;
        if (worldRank_ == 1 - target && size > 0) FillBuf(static_cast<uint8_t*>(src->ptr) + srcOff, size, seed);
        if (worldRank_ == target) FillSentinel(dst->ptr, checked, sentinel);
        ASSERT_NO_FATAL_FAILURE(Rank0Xfer(get, srcOff, srcMh, size, dstOff, dstMh));
        if (flushOff) *flushOff = Rank1Flush(dstMh);
        if (worldRank_ == target) ExpectPayloadIsolated(dst->ptr, checked, dstOff, size, seed, sentinel, context);
        Barrier();
    }

    // Rank 0 posts each out-of-range put (or get) of a `total`-byte window; none may be accepted or posted.
    void ExpectOutOfRangeRejected(bool get, void* srcMh, void* dstMh, size_t total)
    {
        const char* op = get ? "iget" : "iput";
        Barrier();
        for (const OutOfRangeCase& c : worldRank_ == 0 ? OutOfRangeCases(total) : std::vector<OutOfRangeCase>{})
        {
            void* req = nullptr;
            EXPECT_EQ(ncclInvalidArgument, PostToRank1(get, c.firstOff, srcMh, c.size, c.secondOff, dstMh, &req))
                << op << " " << c.why << " must be rejected";
            EXPECT_EQ(req, nullptr) << "rejected " << op << " (" << c.why << ") must not produce a request";
        }
        Barrier();
    }

    // A pool window of nSeg uniform segments; returns `skip` when the allocation failed.
    std::string AllocWindow(int nSeg, MultiSegmentVmmBuffer** b, const std::string& skip)
    {
        *b = pool_.AllocUniform(nSeg, kSegRequestBytes);
        return prereq_.Skip(*b == nullptr, kVmmAllocFailed) ? skip : "";
    }

    // Rank 1 flushes mh toward rank 0. True when GDR flush is off and no flush WR was built, which fails under
    // RCCL_TEST_RMA_MSEG_REQUIRE=1.
    bool Rank1Flush(void* mh)
    {
        if (worldRank_ != 1) return false;
        void* freq = nullptr;
        EXPECT_EQ(ncclSuccess, rma_->iflush(rmaCtx_, 0, mh, /*peerRank=*/0, &freq)) << "iflush post failed";
        const bool off = freq == nullptr;
        if (off) prereq_.FailIfRequired("iflush returned no request");
        if (!off) EXPECT_TRUE(PollUntilDone(freq)) << "flush did not complete";
        return off;
    }

    // Plain 1 MiB hipMalloc pair (one segment each), registered; rank 0 puts `seed`'s pattern into recvBuf.
    void PreparePlainPair(int seed, void** recvBuf, void** recvMh)
    {
        void* sendBuf = AllocBuf(kMiB);
        *recvBuf = AllocBuf(kMiB);
        ASSERT_NE(sendBuf, nullptr);
        ASSERT_NE(*recvBuf, nullptr);
        if (worldRank_ == 0)
            FillBuf(sendBuf, kMiB, seed);
        void* sendMh = nullptr;
        ASSERT_EQ(ncclSuccess, RegMr(sendBuf, kMiB, &sendMh));
        ASSERT_EQ(ncclSuccess, RegMr(*recvBuf, kMiB, recvMh));
        ASSERT_NO_FATAL_FAILURE(Rank0Put(0, sendMh, kMiB, 0, *recvMh));
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

    enum class MidSegmentOp { Put, Get, PutThenFlush };

    // A transfer from mid-first-segment to mid-last-segment, so the WR builder splits on a non-zero
    // per-segment offset on both sides; the target window is checked against sentinels.
    void RunMidSegmentTransfer(MidSegmentOp op, int seed, uint8_t sentinel)
    {
        MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
        void *srcMh = nullptr, *dstMh = nullptr;
        RMA_MSEG_PREPARE_OR_SKIP(PreparePair(&sb, &rb, &srcMh, &dstMh));

        const size_t off  = sb->segSize / 2;
        const size_t size = sb->totalSize - sb->segSize;
        const bool   get  = op == MidSegmentOp::Get;
        bool flushOff = false;
        ASSERT_NO_FATAL_FAILURE(Rank0Round(
            get, sb, srcMh, off, size, rb, dstMh, off, seed, sentinel,
            op == MidSegmentOp::PutThenFlush ? std::string("partial IPut followed by flush")
                                             : (get ? "IGet at offset " : "IPut at offset ") + std::to_string(off),
            0, op == MidSegmentOp::PutThenFlush ? &flushOff : nullptr));
        if (MPIHelpers::anyRankTrue(flushOff)) GTEST_SKIP() << "GDR flush is disabled; iflush built no flush WR";
    }

    void RunIPutSizeSweep(MultiSegmentVmmBuffer* src, MultiSegmentVmmBuffer* dst, void* srcMh, void* dstMh,
                          size_t offset, uint8_t seedBase, uint8_t sentinel)
    {
        const std::vector<size_t> sizes = EdgeCaseSizes(src->segSize, src->totalSize - offset);
        for (size_t idx = 0; idx < sizes.size(); ++idx)
        {
            const std::string context = "size=" + std::to_string(sizes[idx]) + " offset=" + std::to_string(offset);
            ASSERT_NO_FATAL_FAILURE(Rank0Round(false, src, srcMh, offset, sizes[idx], dst, dstMh, offset,
                                               static_cast<uint8_t>(seedBase + (idx & 0x3F)), sentinel, context));
        }
    }

    OwnedVmmPool pool_;
    std::vector<std::unique_ptr<RCCLHybridVmmTests::HybridVmmBuffer>> hybridBuffers_;
};

} // namespace RCCLRmaTests

#endif // RCCL_HAS_RMA_IB_PROXY
#endif // MPI_TESTS_ENABLED
#endif // RCCL_TEST_RMA_MULTISEGMENT_FIXTURE_HPP
