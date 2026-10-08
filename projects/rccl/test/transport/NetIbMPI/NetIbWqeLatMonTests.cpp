/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "NetIbMPITestBase.hpp"

#ifdef MPI_TESTS_ENABLED

#include "net_ib_wqe_lat_inspect.h"

/*
 * Functional coverage of the net_ib per-QP WQE post-to-poll latency monitor
 * (NCCL_IB_WQE_LATENCY_THRESHOLD_NS / NCCL_IB_WQE_LATENCY_REPORT) on live
 * net_ib connections between two ranks.
 *
 * Rank 0 accepts (recv comm), rank 1 connects (send comm). Every isend posts
 * one signaled RDMA_WRITE_WITH_IMM on each of nqpsPerRequest QPs; the monitor
 * stamps the first signaled WQE on an idle QP at post time and closes the
 * sample when ncclIbTest polls its CQE. The CQ is only polled from test(), so
 * a test that posts and then sleeps before polling produces a known minimum
 * latency.
 *
 * The monitor's switches are process globals set once from the environment.
 * Tests that need it on (or off) regardless of the launch environment
 * override the globals for their scope; WqeLatMonEnvEnabledRecordsSends runs
 * only when the environment itself enables the monitor.
 *
 * net_ib only: the fixture always drives ncclNetIb, whatever NCCL_NET says.
 */

namespace {

constexpr uint64_t kNoSlowThresholdNs = UINT64_MAX;

// See ScopedWqeLatGlobals in NetIbWqeLatMonUnitTests.cpp: fire the env-driven
// std::call_once before saving, so it can not clobber the override later.
// Must outlive every connection created under it.
struct ScopedWqeLatGlobals {
    bool savedEnabled;
    uint64_t savedThresholdNs;
    bool savedReport;
    ScopedWqeLatGlobals(bool enabled, uint64_t thresholdNs) {
        struct ncclIbWqeLatMon dummy;
        ncclIbWqeLatMonInit(&dummy);
        savedEnabled = ncclIbWqeLatEnabled;
        savedThresholdNs = ncclIbWqeLatThresholdNs;
        savedReport = ncclIbWqeLatReportEnabled;
        ncclIbWqeLatEnabled = enabled;
        ncclIbWqeLatThresholdNs = enabled ? thresholdNs : 0;
        ncclIbWqeLatReportEnabled = enabled;
    }
    ~ScopedWqeLatGlobals() {
        ncclIbWqeLatEnabled = savedEnabled;
        ncclIbWqeLatThresholdNs = savedThresholdNs;
        ncclIbWqeLatReportEnabled = savedReport;
    }
};

struct WqeLatTotals {
    uint64_t count = 0;
    uint64_t slow = 0;
    uint64_t inflight = 0;
    int tracking = 0;
    int qpsWithSamples = 0;
};

WqeLatTotals sumWqeLat(const struct ncclIbWqeLatCommState& st) {
    WqeLatTotals t;
    for (int q = 0; q < st.nqps; q++) {
        t.count += st.qps[q].stats.count;
        t.slow += st.qps[q].stats.slowCount;
        t.inflight += st.qps[q].inflight;
        t.tracking += st.qps[q].tracking ? 1 : 0;
        t.qpsWithSamples += st.qps[q].stats.count > 0 ? 1 : 0;
    }
    return t;
}

void logWqeLatState(const char* tag, const struct ncclIbWqeLatCommState& st) {
    for (int q = 0; q < st.nqps; q++) {
        const struct ncclIbWqeLatQpState& s = st.qps[q];
        TEST_INFO("Rank %d %s: %s qp[%d] qpn=%u dev=%d n=%lu slow=%lu inflight=%u tracking=%d "
                  "mean=%.0f p50=%lu p99=%lu max=%lu",
                  MPIEnvironment::world_rank, tag, st.isSend ? "send" : "recv", q, s.qpNum, s.devIndex,
                  (unsigned long)s.stats.count, (unsigned long)s.stats.slowCount, s.inflight, s.tracking ? 1 : 0,
                  s.stats.meanNs, (unsigned long)s.stats.p50Ns, (unsigned long)s.stats.p99Ns,
                  (unsigned long)s.stats.maxNs);
    }
}

}  // namespace

class NetIbWqeLatMonTest : public NetIbMPITest {
protected:
    static constexpr int kWqeLatTagBase = 2013;

    void SetUp() override {
        NetIbMPITest::SetUp();
        net_ = &ncclNetIb;
    }

    // One send/recv round trip per size, each fully completed (both ranks
    // polled to done) before the next is posted. buf must hold the largest size.
    // Polls without sleeping: the sleeping wait's 10 ms backoff would become
    // every sample, and the summaries would report the poll interval.
    void RunSyncTransfers(const ConnectionPair& pair, void* buf, void* mhandle, const std::vector<size_t>& sizes,
                          int tagBase) {
        const int rank = MPIEnvironment::world_rank;
        for (size_t i = 0; i < sizes.size(); i++) {
            const int tag = tagBase + (int)i;
            void* request = nullptr;
            if (rank == 0) {
                PostSingleRecv(pair.recvComm, buf, sizes[i], tag, mhandle, &request);
            } else {
                PostSendWithRetry(pair.sendComm, buf, sizes[i], tag, mhandle, &request);
            }
            int recvSizes[1] = {0};
            const ThreadResult r = WorkerWaitBusy(request, recvSizes, kLargeTransferTimeoutMs);
            EXPECT_TRUE(r.ok) << "transfer " << i << " size " << sizes[i] << ": " << r.msg;
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // Sync transfers followed by the per-QP accounting every enabled run must
    // satisfy: one sample per signaled WQE, nothing left in flight, and no
    // samples on the receive side (which never posts through ncclIbMultiSend).
    void CheckSyncTransfersAreCounted(const char* tag, bool expectNoSlow) {
        const int rank = MPIEnvironment::world_rank;
        ConnectionPair pair;
        NetConnectionGuard connGuard(net_);
        SetupConnectionWithGuard(0, pair, connGuard);

        const size_t kMaxSize = 1 << 20;
        void* buf = malloc(kMaxSize);
        ASSERT_NE(buf, nullptr);
        HostBufferAutoGuard bufGuard = makeHostBufferAutoGuard(buf);
        memset(buf, 0x5a, kMaxSize);
        void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        void* mhandle = nullptr;
        ASSERT_EQ(RegisterMemory(comm, buf, kMaxSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
        NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

        // Small sends exercise the zero-length WRs posted on the extra QPs
        // under split-data; 1 MB exercises the real split.
        std::vector<size_t> sizes;
        for (int i = 0; i < 20; i++) sizes.push_back(kSmallBufferSize);
        for (int i = 0; i < 20; i++) sizes.push_back(kMaxSize);
        RunSyncTransfers(pair, buf, mhandle, sizes, kWqeLatTagBase);

        struct ncclIbWqeLatCommState st;
        EXPECT_EQ(ncclIbWqeLatGetCommState(comm, &st), ncclSuccess);
        logWqeLatState(tag, st);
        const WqeLatTotals t = sumWqeLat(st);
        EXPECT_EQ(t.inflight, 0u);
        EXPECT_EQ(t.tracking, 0);
        if (rank == 1) {
            EXPECT_EQ(st.isSend, 1);
            EXPECT_EQ(t.count, sizes.size() * (uint64_t)st.nqpsPerRequest)
                << "nqps=" << st.nqps << " nqpsPerRequest=" << st.nqpsPerRequest;
            EXPECT_LE(t.slow, t.count);
            if (expectNoSlow) {
                EXPECT_EQ(t.slow, 0u);
            }
            if (st.nqpsPerRequest == st.nqps) {
                for (int q = 0; q < st.nqps; q++) {
                    EXPECT_EQ(st.qps[q].stats.count, (uint64_t)sizes.size()) << "qp " << q;
                }
            }
            // The four percentiles come from independent P^2 estimators, so
            // only bounds by max are invariant; ordering is a unit-test concern.
            for (int q = 0; q < st.nqps; q++) {
                const struct ncclIbWqeLatStats& s = st.qps[q].stats;
                if (s.count == 0) continue;
                EXPECT_GT(s.meanNs, 0.0) << "qp " << q;
                EXPECT_LE((double)s.meanNs, (double)s.maxNs) << "qp " << q;
                EXPECT_LE(s.p50Ns, s.maxNs) << "qp " << q;
                EXPECT_LE(s.p90Ns, s.maxNs) << "qp " << q;
                EXPECT_LE(s.p99Ns, s.maxNs) << "qp " << q;
                EXPECT_LE(s.p999Ns, s.maxNs) << "qp " << q;
                EXPECT_LE((double)s.maxNs, (double)kLargeTransferTimeoutMs * 1e6) << "qp " << q;
            }
        } else {
            EXPECT_EQ(st.isSend, 0);
            EXPECT_EQ(t.count, 0u);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

// =============================================================================
// Test: WqeLatMonEnvKnobs
//
// The globals the data path checks must reflect NCCL_IB_WQE_LATENCY_* after
// the one-time init: off (and quiet) unless the threshold is > 0, and then
// REPORT defaults to on.
// =============================================================================
TEST_F(NetIbWqeLatMonTest, WqeLatMonEnvKnobs) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);

    struct ncclIbWqeLatMon dummy;
    ncclIbWqeLatMonInit(&dummy);

    const char* thrEnv = getenv("NCCL_IB_WQE_LATENCY_THRESHOLD_NS");
    const char* repEnv = getenv("NCCL_IB_WQE_LATENCY_REPORT");
    const long long thr = thrEnv ? strtoll(thrEnv, nullptr, 0) : 0;
    const bool report = repEnv ? strtoll(repEnv, nullptr, 0) != 0 : true;
    TEST_INFO("Rank %d: NCCL_IB_WQE_LATENCY_THRESHOLD_NS=%s NCCL_IB_WQE_LATENCY_REPORT=%s -> enabled=%d thr=%lu "
              "report=%d",
              MPIEnvironment::world_rank, thrEnv ? thrEnv : "(unset)", repEnv ? repEnv : "(unset)",
              ncclIbWqeLatEnabled ? 1 : 0, (unsigned long)ncclIbWqeLatThresholdNs, ncclIbWqeLatReportEnabled ? 1 : 0);

    if (thr > 0) {
        EXPECT_TRUE(ncclIbWqeLatEnabled);
        EXPECT_EQ(ncclIbWqeLatThresholdNs, (uint64_t)thr);
        EXPECT_EQ(ncclIbWqeLatReportEnabled, report);
    } else {
        EXPECT_FALSE(ncclIbWqeLatEnabled);
        EXPECT_EQ(ncclIbWqeLatThresholdNs, 0u);
        EXPECT_FALSE(ncclIbWqeLatReportEnabled);
    }
}

// =============================================================================
// Test: WqeLatMonDisabledRecordsNothing
//
// With the monitor off the data path must not touch the per-QP state: after
// real transfers every QP on both sides has no samples and nothing in flight.
// =============================================================================
TEST_F(NetIbWqeLatMonTest, WqeLatMonDisabledRecordsNothing) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
    ScopedWqeLatGlobals g(false, 0);
    AssertInitAndGetDevices(nullptr);

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(0, pair, connGuard);

    void* buf = malloc(kSmallBufferSize);
    ASSERT_NE(buf, nullptr);
    HostBufferAutoGuard bufGuard = makeHostBufferAutoGuard(buf);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kSmallBufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    RunSyncTransfers(pair, buf, mhandle, std::vector<size_t>(20, kSmallBufferSize), kWqeLatTagBase);

    struct ncclIbWqeLatCommState st;
    EXPECT_EQ(ncclIbWqeLatGetCommState(comm, &st), ncclSuccess);
    logWqeLatState("disabled", st);
    const WqeLatTotals t = sumWqeLat(st);
    EXPECT_EQ(t.count, 0u);
    EXPECT_EQ(t.slow, 0u);
    EXPECT_EQ(t.inflight, 0u);
    EXPECT_EQ(t.tracking, 0);
    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Test: WqeLatMonSyncSendsCountEverySignaledWqe
//
// With nothing else in flight, every signaled WQE is tracked, so after N
// synchronous sends the send comm holds exactly N * nqpsPerRequest samples
// (N per QP when each request uses every QP), with percentiles bounded by max.
// A threshold no completion can exceed keeps slow at 0.
// =============================================================================
TEST_F(NetIbWqeLatMonTest, WqeLatMonSyncSendsCountEverySignaledWqe) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
    ScopedWqeLatGlobals g(true, kNoSlowThresholdNs);
    AssertInitAndGetDevices(nullptr);

    CheckSyncTransfersAreCounted("sync", /*expectNoSlow=*/true);
}

// =============================================================================
// Test: WqeLatMonDelayedPollIsSlow
//
// The sender posts one send and sleeps 20 ms before polling. Until polled, each
// QP the request used is still tracking with no samples; after completion
// each of them holds one sample of at least 20 ms, all counted slow against a
// 1 ms threshold.
// =============================================================================
TEST_F(NetIbWqeLatMonTest, WqeLatMonDelayedPollIsSlow) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
    constexpr uint64_t kThrNs = 1000000;
    constexpr int kDelayMs = 20;
    ScopedWqeLatGlobals g(true, kThrNs);
    AssertInitAndGetDevices(nullptr);

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(0, pair, connGuard);

    void* buf = malloc(kSmallBufferSize);
    ASSERT_NE(buf, nullptr);
    HostBufferAutoGuard bufGuard = makeHostBufferAutoGuard(buf);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kSmallBufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    void* request = nullptr;
    struct ncclIbWqeLatCommState before{};
    if (rank == 0) {
        PostSingleRecv(pair.recvComm, buf, kSmallBufferSize, kWqeLatTagBase, mhandle, &request);
    } else {
        PostSendWithRetry(pair.sendComm, buf, kSmallBufferSize, kWqeLatTagBase, mhandle, &request);
        std::this_thread::sleep_for(std::chrono::milliseconds(kDelayMs));
        EXPECT_EQ(ncclIbWqeLatGetCommState(comm, &before), ncclSuccess);
        logWqeLatState("before-poll", before);
        const WqeLatTotals t = sumWqeLat(before);
        EXPECT_EQ(t.count, 0u);
        EXPECT_EQ(t.tracking, before.nqpsPerRequest);
        EXPECT_EQ(t.inflight, (uint64_t)before.nqpsPerRequest);
    }
    int recvSizes[1] = {0};
    EXPECT_EQ(WaitForCompletion(request, recvSizes, kLargeTransferTimeoutMs), ncclSuccess);

    if (rank == 1) {
        struct ncclIbWqeLatCommState after;
        EXPECT_EQ(ncclIbWqeLatGetCommState(comm, &after), ncclSuccess);
        logWqeLatState("after-poll", after);
        const WqeLatTotals t = sumWqeLat(after);
        EXPECT_EQ(t.count, (uint64_t)after.nqpsPerRequest);
        EXPECT_EQ(t.slow, t.count);
        EXPECT_EQ(t.inflight, 0u);
        EXPECT_EQ(t.tracking, 0);
        for (int q = 0; q < after.nqps; q++) {
            if (!before.qps[q].tracking) continue;
            EXPECT_EQ(after.qps[q].stats.count, 1u) << "qp " << q;
            EXPECT_GE(after.qps[q].stats.maxNs, (uint64_t)kDelayMs * 1000000ULL) << "qp " << q;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Test: WqeLatMonPipelinedSendsSampleOncePerQp
//
// K sends posted back to back before any poll: only the first signaled WQE on
// each QP is tracked (one tracking window per QP), the rest only add to
// inflight. After completing all K every used QP holds exactly one sample and
// inflight is drained to 0, so a following synchronous round starts with
// pendingBefore 0 and samples every WQE again.
// =============================================================================
TEST_F(NetIbWqeLatMonTest, WqeLatMonPipelinedSendsSampleOncePerQp) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
    constexpr int kPipelined = 8;
    constexpr int kSyncAfter = 4;
    ScopedWqeLatGlobals g(true, kNoSlowThresholdNs);
    AssertInitAndGetDevices(nullptr);

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(0, pair, connGuard);

    const size_t bufSize = kSmallBufferSize * kPipelined;
    void* buf = malloc(bufSize);
    ASSERT_NE(buf, nullptr);
    HostBufferAutoGuard bufGuard = makeHostBufferAutoGuard(buf);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, bufSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    std::vector<void*> requests(kPipelined, nullptr);
    for (int i = 0; i < kPipelined; i++) {
        char* slot = (char*)buf + i * kSmallBufferSize;
        if (rank == 0) {
            PostSingleRecv(pair.recvComm, slot, kSmallBufferSize, kWqeLatTagBase + i, mhandle, &requests[i]);
        } else {
            PostSendWithRetry(pair.sendComm, slot, kSmallBufferSize, kWqeLatTagBase + i, mhandle, &requests[i]);
        }
    }
    struct ncclIbWqeLatCommState posted{};
    if (rank == 1) {
        EXPECT_EQ(ncclIbWqeLatGetCommState(comm, &posted), ncclSuccess);
        logWqeLatState("posted", posted);
        const WqeLatTotals t = sumWqeLat(posted);
        EXPECT_EQ(t.count, 0u);
        EXPECT_EQ(t.inflight, (uint64_t)kPipelined * posted.nqpsPerRequest);
    }
    for (int i = 0; i < kPipelined; i++) {
        int recvSizes[1] = {0};
        EXPECT_EQ(WaitForCompletion(requests[i], recvSizes, kLargeTransferTimeoutMs), ncclSuccess) << "request " << i;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        struct ncclIbWqeLatCommState st;
        EXPECT_EQ(ncclIbWqeLatGetCommState(comm, &st), ncclSuccess);
        logWqeLatState("pipelined", st);
        const WqeLatTotals t = sumWqeLat(st);
        EXPECT_EQ(t.inflight, 0u);
        EXPECT_EQ(t.tracking, 0);
        EXPECT_GE(t.qpsWithSamples, st.nqpsPerRequest);
        for (int q = 0; q < st.nqps; q++) {
            EXPECT_EQ(st.qps[q].stats.count, posted.qps[q].tracking ? 1u : 0u) << "qp " << q;
            EXPECT_EQ(st.qps[q].pendingBefore, 0u) << "qp " << q;
        }
    }

    RunSyncTransfers(pair, buf, mhandle, std::vector<size_t>(kSyncAfter, kSmallBufferSize),
                     kWqeLatTagBase + kPipelined);
    if (rank == 1) {
        struct ncclIbWqeLatCommState st;
        EXPECT_EQ(ncclIbWqeLatGetCommState(comm, &st), ncclSuccess);
        logWqeLatState("after-sync", st);
        const WqeLatTotals t = sumWqeLat(st);
        EXPECT_EQ(t.count, (uint64_t)sumWqeLat(posted).tracking + (uint64_t)kSyncAfter * st.nqpsPerRequest);
        EXPECT_EQ(t.inflight, 0u);
        EXPECT_EQ(t.tracking, 0);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Test: WqeLatMonEnvEnabledRecordsSends
//
// End-to-end enablement through the environment alone (no global overrides):
// when NCCL_IB_WQE_LATENCY_THRESHOLD_NS > 0 the monitor must record every
// signaled WQE of synchronous sends. Skipped when the variable is unset.
// =============================================================================
TEST_F(NetIbWqeLatMonTest, WqeLatMonEnvEnabledRecordsSends) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
    const char* thrEnv = getenv("NCCL_IB_WQE_LATENCY_THRESHOLD_NS");
    if (thrEnv == nullptr || strtoll(thrEnv, nullptr, 0) <= 0) {
        GTEST_SKIP() << "NCCL_IB_WQE_LATENCY_THRESHOLD_NS not set > 0";
    }
    AssertInitAndGetDevices(nullptr);
    ASSERT_TRUE(ncclIbWqeLatEnabled);

    CheckSyncTransfersAreCounted("env", /*expectNoSlow=*/false);
}

#endif  // MPI_TESTS_ENABLED
