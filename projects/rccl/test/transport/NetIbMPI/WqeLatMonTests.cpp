/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "NetIbMPITestBase.hpp"
#include "NetIbCastInspect.hpp"

#ifdef MPI_TESTS_ENABLED

// Skip unless the WQE-latency monitor is armed. NCCL_IB_WQE_LATENCY_THRESHOLD_NS
// is read once per process inside wqe_lat_mon.cc's ensureInitialized(), gated by
// std::call_once -- a runtime setenv() from inside a test body would have no
// effect, so this must be set via the MPI launch environment.
#define WQE_LAT_ENV_CHECK_OR_SKIP()                                                        \
    do {                                                                                   \
        const char* _thr = getenv("NCCL_IB_WQE_LATENCY_THRESHOLD_NS");                     \
        long long   _thrVal = (_thr && _thr[0]) ? std::atoll(_thr) : 0;                    \
        if (_thrVal <= 0) {                                                                \
            GTEST_SKIP() << "Requires NCCL_IB_WQE_LATENCY_THRESHOLD_NS > 0, set via the "  \
                            "MPI launch environment (ensureInitialized() in "              \
                            "wqe_lat_mon.cc caches it once per process via "                \
                            "std::call_once).";                                            \
        }                                                                                   \
    } while (0)

// Skip unless QP sharing is enabled AND forced into a single group. With
// RCCL_IB_COMM_NGROUPS=1, IbCastQpSharingSenderSetup's groupIdx = totalRefs %
// ngroups always resolves to group 0, so a second SetupCastConnection between
// the same rank pair deterministically becomes a QP-sharing secondary of the
// first, rather than depending on luck across ngroups>1.
#define QP_SHARING_ENV_CHECK_OR_SKIP()                                                     \
    do {                                                                                   \
        struct { const char* name; const char* required; } _vars[] = {                    \
            { "RCCL_IB_QP_SHARING_ENABLE", "1" },                                         \
            { "RCCL_IB_COMM_NGROUPS",      "1" },                                         \
        };                                                                                 \
        for (auto& _v : _vars) {                                                          \
            const char* _val = getenv(_v.name);                                           \
            if (!_val || strcmp(_val, _v.required) != 0) {                                \
                GTEST_SKIP() << "Requires " << _v.name << "=" << _v.required              \
                             << " (current: " << (_val ? _val : "<unset>") << ").";       \
            }                                                                              \
        }                                                                                  \
    } while (0)

// =============================================================================
// Test: WqeLatMonNonSharingStampComplete
//
// Black-box: non-sharing connection, kNSends small signaled sends below the
// split-data threshold (one WQE per logical send). Verifies via
// ncclIbCastGetWqeLatState that the monitor is enabled with the configured
// threshold, that it recorded exactly one completion per send, and that the
// per-QP percentile ladder (p50<=p90<=p99<=p999<=max) is internally
// consistent.
// =============================================================================
TEST_F(NetIbMPITest, WqeLatMonNonSharingStampComplete) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit);

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    WQE_LAT_ENV_CHECK_OR_SKIP();

    constexpr int    kNSends = 12;
    constexpr size_t kMsgSz  = 256;
    if (GetSplitDataMin() == 0) {
        GTEST_SKIP() << "RCCL_IB_QP_SCHED_SPLIT_DATA_MIN=0: every message is split, "
                        "breaking the one-WQE-per-send assumption this test relies on";
    }
    ASSERT_LT(kMsgSz, GetSplitDataMin())
        << "message must stay under the split-data threshold: one WQE per send";

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm, &sendComm, &recvComm);

    char sendBuf[kNSends * kMsgSz], recvBuf[kNSends * kMsgSz];
    for (size_t i = 0; i < sizeof(sendBuf); i++) sendBuf[i] = static_cast<char>(i & 0xFF);
    memset(recvBuf, 0, sizeof(recvBuf));

    void* comm    = (rank == kCastReceiverRank) ? recvComm : sendComm;
    void* baseBuf = (rank == kCastReceiverRank) ? static_cast<void*>(recvBuf)
                                                 : static_cast<void*>(sendBuf);
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, baseBuf, sizeof(sendBuf), NCCL_PTR_HOST, &mhandle), ncclSuccess);

    CastDoSyncSendRecv(rank, sendComm, recvComm, sendBuf, recvBuf, kMsgSz, kNSends, 400, mhandle);

    if (rank == kCastReceiverRank)
        EXPECT_EQ(memcmp(sendBuf, recvBuf, sizeof(sendBuf)), 0) << "data mismatch";

    if (rank == kCastSenderRank) {
        struct ncclIbCastWqeLatState state = {};
        ASSERT_EQ(ncclIbCastGetWqeLatState(sendComm, &state), ncclSuccess);
        EXPECT_TRUE(state.enabled);
        EXPECT_GT(state.thresholdNs, 0u);
        ASSERT_GT(state.nqps, 0);

        // The WRR scheduler (mandatory for cast tests, see CAST_ENV_CHECK_OR_SKIP)
        // fans every logical send out across all nqps QPs of the connection, so
        // each QP sees one physical signaled WQE -- and one counted completion --
        // per logical send.
        uint64_t totalCount = 0;
        for (int i = 0; i < state.nqps; i++) {
            const auto& qp = state.qps[i];
            totalCount += qp.count;
            EXPECT_EQ(qp.count, static_cast<uint64_t>(kNSends))
                << "qp " << qp.qpNum << ": expected one completion per signaled send";
            if (qp.count == 0) continue;
            EXPECT_LE(qp.p50Ns, qp.p90Ns) << "qp " << qp.qpNum << ": p50 > p90";
            EXPECT_LE(qp.p90Ns, qp.p99Ns) << "qp " << qp.qpNum << ": p90 > p99";
            EXPECT_LE(qp.p99Ns, qp.p999Ns) << "qp " << qp.qpNum << ": p99 > p999";
            EXPECT_LE(qp.p999Ns, qp.maxNs) << "qp " << qp.qpNum << ": p999 > observed max";
            EXPECT_GE(qp.meanNs, 0.0);
            EXPECT_GE(qp.stddevNs, 0.0);
        }
        EXPECT_EQ(totalCount, static_cast<uint64_t>(kNSends) * static_cast<uint64_t>(state.nqps))
            << "latMon must record exactly one completion per signaled send, per QP";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: WqeLatMonQpSharingPerCommIsolation
//
// White-box: forces two logical comms (sendComm1/sendComm2) between the same
// rank pair into QP sharing via RCCL_IB_COMM_NGROUPS=1 (deterministic group-0
// assignment). Each comm drives its own independent batch of sends. Verifies
// via ncclIbCastGetWqeLatState, queried separately per comm, that each comm's
// latMon count reflects only its own traffic -- no cross-comm contamination --
// which holds by construction since completions are routed to the owning
// logical comm (by wr_id/commId/immData) before any latency code runs. Also
// covers comm-close teardown (scenario d): closing both sharing comms in
// sequence must not crash/double-free, which is safe because
// IbCastWqeLatReportQpSummary is called unconditionally per comm in the
// teardown loop (connect.cc IbCastCloseSend), independent of the physical-QP
// refcount that gates the actual ibv_destroy_qp.
// =============================================================================
TEST_F(NetIbMPITest, WqeLatMonQpSharingPerCommIsolation) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit);

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    WQE_LAT_ENV_CHECK_OR_SKIP();
    QP_SHARING_ENV_CHECK_OR_SKIP();

    constexpr int    kNSends = 12;
    constexpr size_t kMsgSz  = 256;
    if (GetSplitDataMin() == 0) {
        GTEST_SKIP() << "RCCL_IB_QP_SCHED_SPLIT_DATA_MIN=0: every message is split, "
                        "breaking the one-WQE-per-send assumption this test relies on";
    }
    ASSERT_LT(kMsgSz, GetSplitDataMin());

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm1 = nullptr; void* sendComm1 = nullptr; void* recvComm1 = nullptr;
    void* listenComm2 = nullptr; void* sendComm2 = nullptr; void* recvComm2 = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm1, &sendComm1, &recvComm1);
    SetupCastConnection(/*dev=*/0, &listenComm2, &sendComm2, &recvComm2);

    char sendBuf1[kNSends * kMsgSz], recvBuf1[kNSends * kMsgSz];
    char sendBuf2[kNSends * kMsgSz], recvBuf2[kNSends * kMsgSz];
    for (size_t i = 0; i < sizeof(sendBuf1); i++) {
        sendBuf1[i] = static_cast<char>(i & 0xFF);
        sendBuf2[i] = static_cast<char>((i + 1) & 0xFF);  // distinct pattern per connection
    }
    memset(recvBuf1, 0, sizeof(recvBuf1));
    memset(recvBuf2, 0, sizeof(recvBuf2));

    void* comm1    = (rank == kCastReceiverRank) ? recvComm1 : sendComm1;
    void* baseBuf1 = (rank == kCastReceiverRank) ? static_cast<void*>(recvBuf1)
                                                  : static_cast<void*>(sendBuf1);
    void* comm2    = (rank == kCastReceiverRank) ? recvComm2 : sendComm2;
    void* baseBuf2 = (rank == kCastReceiverRank) ? static_cast<void*>(recvBuf2)
                                                  : static_cast<void*>(sendBuf2);
    void* mhandle1 = nullptr;
    void* mhandle2 = nullptr;
    ASSERT_EQ(RegisterMemory(comm1, baseBuf1, sizeof(sendBuf1), NCCL_PTR_HOST, &mhandle1), ncclSuccess);
    ASSERT_EQ(RegisterMemory(comm2, baseBuf2, sizeof(sendBuf2), NCCL_PTR_HOST, &mhandle2), ncclSuccess);

    CastDoSyncSendRecv(rank, sendComm1, recvComm1, sendBuf1, recvBuf1, kMsgSz, kNSends, 500, mhandle1);
    CastDoSyncSendRecv(rank, sendComm2, recvComm2, sendBuf2, recvBuf2, kMsgSz, kNSends, 600, mhandle2);

    if (rank == kCastReceiverRank) {
        EXPECT_EQ(memcmp(sendBuf1, recvBuf1, sizeof(sendBuf1)), 0) << "comm1 data mismatch";
        EXPECT_EQ(memcmp(sendBuf2, recvBuf2, sizeof(sendBuf2)), 0) << "comm2 data mismatch";
    }

    // Confirm QP sharing actually activated for this pair before asserting
    // isolation: at least one physical qpNum must be observed on both logical
    // comms. Non-fatal (EXPECT_, not ASSERT_) so rank 1's result can still be
    // forwarded into the synced skip decision below.
    int shouldSkip = 0;
    struct ncclIbCastWqeLatState st1 = {};
    struct ncclIbCastWqeLatState st2 = {};
    if (rank == kCastSenderRank) {
        EXPECT_EQ(ncclIbCastGetWqeLatState(sendComm1, &st1), ncclSuccess);
        EXPECT_EQ(ncclIbCastGetWqeLatState(sendComm2, &st2), ncclSuccess);
        EXPECT_TRUE(st1.enabled);
        EXPECT_TRUE(st2.enabled);
        EXPECT_GT(st1.nqps, 0);
        EXPECT_GT(st2.nqps, 0);

        bool sharedQpFound = false;
        for (int i = 0; i < st1.nqps && !sharedQpFound; i++) {
            for (int j = 0; j < st2.nqps; j++) {
                if (st1.qps[i].qpNum != 0 && st1.qps[i].qpNum == st2.qps[j].qpNum) {
                    sharedQpFound = true;
                    break;
                }
            }
        }
        shouldSkip = sharedQpFound ? 0 : 1;
    }

    MPI_Allreduce(MPI_IN_PLACE, &shouldSkip, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (shouldSkip) {
        TeardownConnection(recvComm1, listenComm1, sendComm1, mhandle1);
        TeardownConnection(recvComm2, listenComm2, sendComm2, mhandle2);
        GTEST_SKIP() << "QP sharing did not activate for this connection pair (no shared "
                        "physical qpNum observed between sendComm1/sendComm2); driver/env "
                        "may not support it here.";
    }

    if (rank == kCastSenderRank) {
        uint64_t total1 = 0, total2 = 0;
        for (int i = 0; i < st1.nqps; i++) total1 += st1.qps[i].count;
        for (int i = 0; i < st2.nqps; i++) total2 += st2.qps[i].count;
        // WRR fans each logical send across all nqps QPs of its own comm (see
        // WqeLatMonNonSharingStampComplete); isolation means comm1's total only
        // reflects comm1's own fan-out, not comm2's.
        EXPECT_EQ(total1, static_cast<uint64_t>(kNSends) * static_cast<uint64_t>(st1.nqps))
            << "comm1 latMon count must reflect only comm1's own sends";
        EXPECT_EQ(total2, static_cast<uint64_t>(kNSends) * static_cast<uint64_t>(st2.nqps))
            << "comm2 latMon count must reflect only comm2's own sends";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Scenario (d): sequential teardown of two QP-sharing comms must not
    // crash/double-free.
    TeardownConnection(recvComm1, listenComm1, sendComm1, mhandle1);
    TeardownConnection(recvComm2, listenComm2, sendComm2, mhandle2);
}

// =============================================================================
// Test: WqeLatMonNoSpuriousCompletionBeforePoll
//
// Black-box: posts one signaled send, then deliberately delays the caller's
// own poll (not the library's post) past the configured threshold before
// calling test()/WaitForCompletion for the first time. ncclIbCastFaultSetQpDelay
// was evaluated for this but is unsuitable: its usleep() in p2p.cc executes
// *before* IbCastWqeLatMonStampSend/wrap_ibv_post_send, so it postpones when
// isend() returns rather than widening the monitor's measured post->poll
// window. Delaying our own poll instead produces a genuine measured stall
// without any fault-injection dependency. Verifies the monitor's count is
// untouched by elapsed wall-clock time alone (completions are only processed
// inside an actual poll/test() call), then that exactly one completion (and
// one slow-count) is recorded once the delayed poll actually runs.
// =============================================================================
TEST_F(NetIbMPITest, WqeLatMonNoSpuriousCompletionBeforePoll) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit);

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    WQE_LAT_ENV_CHECK_OR_SKIP();

    constexpr size_t kMsgSz = 256;
    if (GetSplitDataMin() == 0) {
        GTEST_SKIP() << "RCCL_IB_QP_SCHED_SPLIT_DATA_MIN=0: every message is split, "
                        "breaking the one-WQE-per-send assumption this test relies on";
    }
    ASSERT_LT(kMsgSz, GetSplitDataMin())
        << "message must stay under the split-data threshold: one WQE per send";

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm, &sendComm, &recvComm);

    char sendBuf[kMsgSz], recvBuf[kMsgSz];
    memset(sendBuf, 0xAB, sizeof(sendBuf));
    memset(recvBuf, 0, sizeof(recvBuf));

    void* comm    = (rank == kCastReceiverRank) ? recvComm : sendComm;
    void* buf     = (rank == kCastReceiverRank) ? static_cast<void*>(recvBuf)
                                                 : static_cast<void*>(sendBuf);
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMsgSz, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    if (rank == kCastSenderRank) {
        struct ncclIbCastWqeLatState before = {};
        ASSERT_EQ(ncclIbCastGetWqeLatState(sendComm, &before), ncclSuccess);
        ASSERT_TRUE(before.enabled);
        ASSERT_GT(before.thresholdNs, 0u);
        uint64_t baselineCount = 0, baselineSlow = 0;
        for (int i = 0; i < before.nqps; i++) {
            baselineCount += before.qps[i].count;
            baselineSlow  += before.qps[i].slowCount;
        }

        void* req = nullptr;
        PostSendWithRetry(sendComm, sendBuf, kMsgSz, 700, mhandle, &req);

        // Sleep past the configured threshold before our first poll. Derived
        // from the already-fetched threshold rather than hardcoded, so this
        // works regardless of the env value configured at launch.
        const uint64_t sleepUs = (before.thresholdNs / 1000ULL) + 50000ULL;
        usleep(static_cast<useconds_t>(sleepUs));

        struct ncclIbCastWqeLatState mid = {};
        ASSERT_EQ(ncclIbCastGetWqeLatState(sendComm, &mid), ncclSuccess);
        uint64_t midCount = 0;
        for (int i = 0; i < mid.nqps; i++) midCount += mid.qps[i].count;
        EXPECT_EQ(midCount, baselineCount)
            << "latMon must not record a completion merely because wall-clock time "
               "exceeded the threshold; only an actual poll/test() call processes completions";

        int sz = 0;
        ASSERT_EQ(WaitForCompletion(req, &sz, 30000), ncclSuccess);

        struct ncclIbCastWqeLatState after = {};
        ASSERT_EQ(ncclIbCastGetWqeLatState(sendComm, &after), ncclSuccess);
        uint64_t afterCount = 0, afterSlow = 0;
        for (int i = 0; i < after.nqps; i++) {
            afterCount += after.qps[i].count;
            afterSlow  += after.qps[i].slowCount;
        }
        // WRR fans the one logical send across all nqps QPs (see
        // WqeLatMonNonSharingStampComplete), so one completion is recorded per QP.
        EXPECT_EQ(afterCount, baselineCount + static_cast<uint64_t>(before.nqps))
            << "exactly one completion per QP must be recorded once the delayed poll runs";
        EXPECT_GE(afterSlow, baselineSlow + static_cast<uint64_t>(before.nqps))
            << "the deliberately-delayed send must be classified as exceeding the latency threshold on every QP";
    } else {
        void*  bufs[1]    = {recvBuf};
        size_t sizes[1]   = {kMsgSz};
        int    tags[1]    = {700};
        void*  handles[1] = {mhandle};
        void*  req        = nullptr;
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &req), ncclSuccess);
        ASSERT_NE(req, nullptr);
        int sz = 0;
        ASSERT_EQ(WaitForCompletion(req, &sz, 30000), ncclSuccess);
        EXPECT_EQ(memcmp(sendBuf, recvBuf, kMsgSz), 0) << "data mismatch";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

#endif  // MPI_TESTS_ENABLED
