/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "NetIbMPITestBase.hpp"

#if defined(MPI_TESTS_ENABLED) && defined(ENABLE_FAULT_INJECTION)

#include "ibvcore.h"
#include "net_ib_gid_inspect.h"

namespace {

constexpr int kDevStateOk = 0;
constexpr int kDevStateRecovered = 4;
constexpr int kRecoveryPollIters = 4000;
constexpr int kRecoveryPollUs = 10000;
constexpr int kPostRecoveryMsgs = 20;
constexpr size_t kMsgSize = 8192;

struct GidTransport {
    const char* name;
    ncclNet_t* net;
    ncclResult_t (*getDev)(int, ncclIbGidState*);
    ncclResult_t (*setDev)(int, const ncclIbGidState*);
    ncclResult_t (*getComm)(void*, int, ncclIbGidState*);
    ncclResult_t (*setComm)(void*, int, const ncclIbGidState*);
    ncclResult_t (*changeEvent)(int);
    ncclResult_t (*getQpState)(void*, ncclIbGidQpState*);
    ncclResult_t (*getDevState)(void*, int, int*);
    ncclResult_t (*driveQpToError)(void*, int);
};

const GidTransport kNetIb = {
    "NetIb", &ncclNetIb, ncclIbGidGetDev, ncclIbGidSetDev, ncclIbGidGetComm, ncclIbGidSetComm,
    ncclIbGidChangeEvent, ncclIbGidGetQpState, ncclIbGidGetDevState, ncclIbGidDriveQpToError,
};

const GidTransport kNetIbCast = {
    "NetIbCast", &netIbCast, ncclIbCastGidGetDev, ncclIbCastGidSetDev, ncclIbCastGidGetComm, ncclIbCastGidSetComm,
    ncclIbCastGidChangeEvent, ncclIbCastGidGetQpState, ncclIbCastGidGetDevState, ncclIbCastGidDriveQpToError,
};

bool SameGid(const ncclIbGidState& a, const ncclIbGidState& b) {
    return a.linkLayer == b.linkLayer && a.gidIndex == b.gidIndex && memcmp(a.gid, b.gid, sizeof(a.gid)) == 0;
}

bool IsZeroGid(const ncclIbGidState& s) {
    static const uint8_t zero[sizeof(s.gid)] = {};
    return memcmp(s.gid, zero, sizeof(s.gid)) == 0;
}

ncclIbGidState MakeStale(const ncclIbGidState& real) {
    ncclIbGidState stale = real;
    stale.gidIndex = real.gidIndex + 1;
    memset(stale.gid, 0xAB, sizeof(stale.gid));
    return stale;
}

bool AllRanks(bool ok) {
    int local = ok ? 1 : 0, all = 0;
    MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return all == 1;
}

}  // namespace

class NetIbGidChangeTest : public NetIbMPITest, public ::testing::WithParamInterface<const GidTransport*> {
protected:
    const GidTransport& T() const { return *GetParam(); }

    void SetUp() override {
        NetIbMPITest::SetUp();
        net_ = T().net;
    }

    int CreateMergedDevicePair(int totalDevs) {
        int mergedDev = -1;
        if (totalDevs >= 2) {
            ncclNetVDeviceProps_t vProps = {};
            vProps.ndevs = 2;
            vProps.devs[0] = 0;
            vProps.devs[1] = 1;
            net_->makeVDevice(&mergedDev, &vProps);
        }
        int minDev = 0;
        MPI_Allreduce(&mergedDev, &minDev, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return minDev < 0 ? -1 : mergedDev;
    }

    bool QpsUseGidIndex(void* comm, int devIndex, int gidIndex, int* checked) {
        ncclIbGidQpState qps = {};
        EXPECT_EQ(T().getQpState(comm, &qps), ncclSuccess);
        bool ok = true;
        *checked = 0;
        for (int i = 0; i < qps.nqps; i++) {
            if (qps.devIndex[i] != devIndex || !qps.queryOk[i]) continue;
            (*checked)++;
            EXPECT_EQ(qps.sgidIndex[i], gidIndex) << "qp " << i << " devIndex " << devIndex;
            ok = ok && qps.sgidIndex[i] == gidIndex;
        }
        return ok;
    }
};

TEST_P(NetIbGidChangeTest, DeviceCacheInitializedAtDiscovery) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
    AssertInitAndGetDevices(nullptr);
    const int nPhys = GetPhysicalDeviceCount();
    ASSERT_GT(nPhys, 0);

    bool ok = true;
    for (int dev = 0; dev < nPhys; dev++) {
        ncclIbGidState s = {};
        ASSERT_EQ(T().getDev(dev, &s), ncclSuccess);
        EXPECT_GE(s.gidIndex, 0) << "dev " << dev;
        ok = ok && s.gidIndex >= 0;
        if (s.linkLayer == IBV_LINK_LAYER_ETHERNET) {
            EXPECT_FALSE(IsZeroGid(s)) << "dev " << dev;
            ok = ok && !IsZeroGid(s);
        }
    }
    ncclIbGidState s = {};
    EXPECT_EQ(T().getDev(nPhys, &s), ncclInvalidArgument);
    EXPECT_TRUE(AllRanks(ok));
}

TEST_P(NetIbGidChangeTest, GidChangeEventRefreshesDeviceCache) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
    AssertInitAndGetDevices(nullptr);
    const int nPhys = GetPhysicalDeviceCount();
    ASSERT_GT(nPhys, 0);

    bool ok = true;
    for (int dev = 0; dev < nPhys; dev++) {
        ncclIbGidState real = {}, after = {};
        ASSERT_EQ(T().getDev(dev, &real), ncclSuccess);
        const ncclIbGidState stale = MakeStale(real);
        ASSERT_EQ(T().setDev(dev, &stale), ncclSuccess);

        EXPECT_EQ(T().changeEvent(dev), ncclSuccess) << "dev " << dev;
        ASSERT_EQ(T().getDev(dev, &after), ncclSuccess);
        const bool expectRefresh = real.linkLayer == IBV_LINK_LAYER_ETHERNET;
        const bool devOk = SameGid(after, expectRefresh ? real : stale);
        EXPECT_TRUE(devOk) << "dev " << dev << " linkLayer " << real.linkLayer << " gidIndex " << after.gidIndex
                           << " expected " << (expectRefresh ? real.gidIndex : stale.gidIndex);
        ok = ok && devOk;

        ASSERT_EQ(T().setDev(dev, &real), ncclSuccess);
    }
    EXPECT_TRUE(AllRanks(ok));
}

TEST_P(NetIbGidChangeTest, ConnectSnapshotsDeviceCache) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
    const int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm = nullptr;
    void* recvComm = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm, &sendComm, &recvComm);

    std::vector<char> buf(kMsgSize, 0x5A);
    void* comm = (rank == 0) ? recvComm : sendComm;
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf.data(), buf.size(), NCCL_PTR_HOST, &mhandle), ncclSuccess);
    CastDoSendRecv(rank, sendComm, recvComm, buf.data(), kMsgSize, /*tag=*/2201, mhandle);

    ncclIbGidState snap = {}, dev = {};
    ASSERT_EQ(T().getComm(comm, 0, &snap), ncclSuccess);
    ASSERT_EQ(T().getDev(snap.ibDev, &dev), ncclSuccess);
    bool ok = SameGid(snap, dev);
    EXPECT_TRUE(ok) << "comm gidIndex " << snap.gidIndex << " device gidIndex " << dev.gidIndex;

    if (dev.linkLayer == IBV_LINK_LAYER_ETHERNET) {
        int checked = 0;
        ok = QpsUseGidIndex(comm, 0, dev.gidIndex, &checked) && ok;
        EXPECT_GT(checked, 0);
        ok = ok && checked > 0;
    }
    EXPECT_TRUE(AllRanks(ok));

    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

TEST_P(NetIbGidChangeTest, PortRecoveryRefreshesStaleCommGid) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
    const char* failoverEnv = getenv("NCCL_IB_RESILIENCY_PORT_FAILOVER");
    const char* recoveryEnv = getenv("NCCL_IB_RESILIENCY_PORT_RECOVERY");
    if (!failoverEnv || strcmp(failoverEnv, "1") != 0) GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_FAILOVER=1";
    if (!recoveryEnv || strcmp(recoveryEnv, "1") != 0) GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_RECOVERY=1";

    const int rank = MPIEnvironment::world_rank;
    int totalDevs = 0;
    AssertInitAndGetDevices(&totalDevs);
    const int mergedDev = CreateMergedDevicePair(totalDevs);
    if (mergedDev < 0) GTEST_SKIP() << "Requires NIC Fusion (ndevs >= 2), found " << totalDevs;

    void* listenComm = nullptr;
    void* sendComm = nullptr;
    void* recvComm = nullptr;
    SetupCastConnection(mergedDev, &listenComm, &sendComm, &recvComm);

    const size_t bufSize = kMsgSize * (kPostRecoveryMsgs + 1);
    std::vector<char> sendBuf(bufSize), recvBuf(bufSize, 0);
    for (size_t i = 0; i < bufSize; i++) sendBuf[i] = static_cast<char>((i * 37 + 19) & 0xFF);
    void* comm = (rank == 0) ? recvComm : sendComm;
    char* regBuf = (rank == 0) ? recvBuf.data() : sendBuf.data();
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, regBuf, bufSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    CastDoSendRecv(rank, sendComm, recvComm, regBuf, kMsgSize, /*tag=*/2300, mhandle);

    ncclIbGidQpState qps = {};
    ASSERT_EQ(T().getQpState(comm, &qps), ncclSuccess);
    ASSERT_GT(qps.nqps, 0);
    const int failDev = qps.devIndex[0];
    ASSERT_GE(failDev, 0);

    ncclIbGidState real = {};
    ASSERT_EQ(T().getComm(comm, failDev, &real), ncclSuccess);
    const ncclIbGidState stale = MakeStale(real);
    ASSERT_EQ(T().setComm(comm, failDev, &stale), ncclSuccess);
    ASSERT_EQ(T().changeEvent(real.ibDev), ncclSuccess);
    ncclIbGidState devAfterEvent = {};
    ASSERT_EQ(T().getDev(real.ibDev, &devAfterEvent), ncclSuccess);
    ASSERT_TRUE(SameGid(devAfterEvent, real));

    void* recvReq = nullptr;
    if (rank == 0) {
        void* bufs[1] = {regBuf};
        size_t sizes[1] = {kMsgSize};
        int tags[1] = {2301};
        void* handles[1] = {mhandle};
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &recvReq), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_EQ(T().driveQpToError(comm, 0), ncclSuccess);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        void* sendReq = nullptr;
        for (int attempt = 0; attempt < kMaxRetryAttempts && sendReq == nullptr; attempt++) {
            if (PostSend(sendComm, regBuf, kMsgSize, 2301, mhandle, &sendReq) != ncclSuccess) break;
            if (sendReq == nullptr) usleep(kPollIntervalUs);
        }
        DrainRecvRequest(sendReq);
    } else {
        DrainRecvRequest(recvReq, 1500);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int state = -1;
    for (int poll = 0; poll < kRecoveryPollIters; poll++) {
        ASSERT_EQ(T().getDevState(comm, failDev, &state), ncclSuccess);
        if (state == kDevStateOk || state == kDevStateRecovered) break;
        usleep(kRecoveryPollUs);
    }
    const bool recovered = state == kDevStateOk || state == kDevStateRecovered;
    EXPECT_TRUE(recovered) << "devIndex " << failDev << " state " << state;
    if (!AllRanks(recovered)) {
        ADD_FAILURE() << "Port recovery did not complete on all ranks";
        TeardownConnection(recvComm, listenComm, sendComm, mhandle);
        return;
    }

    ncclIbGidState snap = {};
    ASSERT_EQ(T().getComm(comm, failDev, &snap), ncclSuccess);
    bool ok = SameGid(snap, real);
    EXPECT_TRUE(ok) << "comm gidIndex after recovery " << snap.gidIndex << " expected " << real.gidIndex;
    if (real.linkLayer == IBV_LINK_LAYER_ETHERNET) {
        int checked = 0;
        ok = QpsUseGidIndex(comm, failDev, real.gidIndex, &checked) && ok;
    }

    for (int m = 0; m < kPostRecoveryMsgs; m++) {
        const size_t off = (m + 1) * kMsgSize;
        void* req = nullptr;
        int sz = 0;
        if (rank == 0) {
            void* bufs[1] = {recvBuf.data() + off};
            size_t sizes[1] = {kMsgSize};
            int tags[1] = {2310 + m};
            void* handles[1] = {mhandle};
            ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &req), ncclSuccess);
            ASSERT_EQ(WaitForCompletion(req, &sz, 10000), ncclSuccess) << "recv " << m;
        } else {
            PostSendWithRetry(sendComm, sendBuf.data() + off, kMsgSize, 2310 + m, mhandle, &req);
            ASSERT_EQ(WaitForCompletion(req, &sz, 10000), ncclSuccess) << "send " << m;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        for (int m = 0; m < kPostRecoveryMsgs; m++) {
            const size_t off = (m + 1) * kMsgSize;
            const bool same = memcmp(recvBuf.data() + off, sendBuf.data() + off, kMsgSize) == 0;
            EXPECT_TRUE(same) << "post-recovery message " << m;
            ok = ok && same;
        }
    }
    EXPECT_TRUE(AllRanks(ok));

    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

INSTANTIATE_TEST_SUITE_P(Transport, NetIbGidChangeTest, ::testing::Values(&kNetIb, &kNetIbCast),
                         [](const ::testing::TestParamInfo<const GidTransport*>& info) {
                             return std::string(info.param->name);
                         });

#endif /* MPI_TESTS_ENABLED && ENABLE_FAULT_INJECTION */
