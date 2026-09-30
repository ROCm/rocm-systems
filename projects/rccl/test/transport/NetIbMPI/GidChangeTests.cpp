/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "NetIbMPITestBase.hpp"

#if defined(MPI_TESTS_ENABLED) && defined(ENABLE_FAULT_INJECTION)

#include <fstream>

#include "ibvcore.h"
#include "net_ib_gid_inspect.h"

namespace {

constexpr int kDevStateOk = 0;         // ncclIbResiliencyDevStateOk
constexpr int kDevStateRecovered = 4;  // ncclIbResiliencyDevStateRecovered
constexpr int kRecoveryPollIters = 4000;
constexpr int kRecoveryPollUs = 10000;
constexpr int kPostRecoveryMsgs = 20;
constexpr size_t kMsgSize = 8192;
constexpr int kMaxGidTableEntries = 256;

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
    ncclResult_t (*getRecoveryGidIndex)(void*, int, int*);
    ncclResult_t (*driveQpToError)(void*, int);
};

const GidTransport kNetIb = {
    "NetIb", &ncclNetIb, ncclIbGidGetDev, ncclIbGidSetDev, ncclIbGidGetComm, ncclIbGidSetComm,
    ncclIbGidChangeEvent, ncclIbGidGetQpState, ncclIbGidGetDevState, ncclIbGidGetRecoveryGidIndex,
    ncclIbGidDriveQpToError,
};

const GidTransport kNetIbCast = {
    "NetIbCast", &netIbCast, ncclIbCastGidGetDev, ncclIbCastGidSetDev, ncclIbCastGidGetComm, ncclIbCastGidSetComm,
    ncclIbCastGidChangeEvent, ncclIbCastGidGetQpState, ncclIbCastGidGetDevState, ncclIbCastGidGetRecoveryGidIndex,
    ncclIbCastGidDriveQpToError,
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

bool ReadSysfsLine(const std::string& path, std::string* out) {
    std::ifstream f(path);
    return static_cast<bool>(std::getline(f, *out));
}

// Restores a device's process-global GID cache entry when the test leaves scope.
class DevGidGuard {
public:
    DevGidGuard(const GidTransport& t, int ibDev, const ncclIbGidState& saved) : t_(t), ibDev_(ibDev), saved_(saved) {}
    ~DevGidGuard() { t_.setDev(ibDev_, &saved_); }
    DevGidGuard(const DevGidGuard&) = delete;
    DevGidGuard& operator=(const DevGidGuard&) = delete;

private:
    const GidTransport& t_;
    int ibDev_;
    ncclIbGidState saved_;
};

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

    // Another valid entry of the same port carrying the same address under a different RoCE type
    // (the v1/v2 twin), i.e. a GID reindex that keeps the port reachable.
    bool FindTwinGid(const ncclIbGidState& real, ncclIbGidState* twin) {
        if (real.linkLayer != IBV_LINK_LAYER_ETHERNET) return false;
        ncclNetProperties_t props = {};
        if (GetDeviceProperties(real.ibDev, &props) != ncclSuccess || props.name == nullptr) return false;
        const std::string port = std::string("/sys/class/infiniband/") + props.name + "/ports/" +
                                 std::to_string(props.port);
        std::string realType;
        if (!ReadSysfsLine(port + "/gid_attrs/types/" + std::to_string(real.gidIndex), &realType)) return false;
        for (int i = 0; i < kMaxGidTableEntries; i++) {
            if (i == real.gidIndex) continue;
            std::string text, type;
            uint8_t gid[16];
            if (!ReadSysfsLine(port + "/gids/" + std::to_string(i), &text)) break;
            if (!ParseGidText(text.c_str(), gid) || memcmp(gid, real.gid, sizeof(gid)) != 0) continue;
            if (!ReadSysfsLine(port + "/gid_attrs/types/" + std::to_string(i), &type) || type.empty() ||
                type == realType)
                continue;
            *twin = real;
            twin->gidIndex = i;
            return true;
        }
        return false;
    }

    bool QpsUseGidIndex(void* comm, int devIndex, int gidIndex) {
        ncclIbGidQpState qps = {};
        EXPECT_EQ(T().getQpState(comm, &qps), ncclSuccess);
        bool ok = true;
        int checked = 0;
        for (int i = 0; i < qps.nqps; i++) {
            if (qps.devIndex[i] != devIndex) continue;
            checked++;
            EXPECT_EQ(qps.rtrGidIndex[i], gidIndex) << "qp " << i << " devIndex " << devIndex;
            ok = ok && qps.rtrGidIndex[i] == gidIndex;
            if (!qps.queryOk[i]) continue;
            EXPECT_EQ(qps.sgidIndex[i], gidIndex) << "qp " << i << " devIndex " << devIndex;
            ok = ok && qps.sgidIndex[i] == gidIndex;
        }
        EXPECT_GT(checked, 0) << "no QP on devIndex " << devIndex;
        return ok && checked > 0;
    }
};

// Device GID cache is filled at init: valid GID index for every NIC, non-zero GID on RoCE,
// out-of-range device rejected.
TEST_P(NetIbGidChangeTest, DeviceCacheInitializedAtDiscovery) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
    AssertInitAndGetDevices(nullptr);
    const int nPhys = GetPhysicalDeviceCount();
    ASSERT_GT(nPhys, 0);

    bool ok = true;
    for (int dev = 0; dev < nPhys; dev++) {
        ncclIbGidState s = {};
        EXPECT_EQ(T().getDev(dev, &s), ncclSuccess);
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

// A stale device cache entry is overwritten by the IBV_EVENT_GID_CHANGE handler: re-read from
// hardware on RoCE, left untouched on IB (as upstream).
TEST_P(NetIbGidChangeTest, GidChangeEventRefreshesDeviceCache) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
    AssertInitAndGetDevices(nullptr);
    const int nPhys = GetPhysicalDeviceCount();
    ASSERT_GT(nPhys, 0);

    bool ok = true;
    for (int dev = 0; dev < nPhys; dev++) {
        ncclIbGidState real = {}, after = {};
        EXPECT_EQ(T().getDev(dev, &real), ncclSuccess);
        DevGidGuard guard(T(), dev, real);
        const ncclIbGidState stale = MakeStale(real);
        EXPECT_EQ(T().setDev(dev, &stale), ncclSuccess);

        EXPECT_EQ(T().changeEvent(dev), ncclSuccess) << "dev " << dev;
        EXPECT_EQ(T().getDev(dev, &after), ncclSuccess);
        const bool expectRefresh = real.linkLayer == IBV_LINK_LAYER_ETHERNET;
        const bool devOk = SameGid(after, expectRefresh ? real : stale);
        EXPECT_TRUE(devOk) << "dev " << dev << " linkLayer " << real.linkLayer << " gidIndex " << after.gidIndex
                           << " expected " << (expectRefresh ? real.gidIndex : stale.gidIndex);
        ok = ok && devOk;
    }
    EXPECT_TRUE(AllRanks(ok));
}

// Connect takes the GID from the device cache: with the cache moved to the v1/v2 twin index, the
// comm snapshot and every QP of the new connection use the twin.
TEST_P(NetIbGidChangeTest, ConnectSnapshotsDeviceCache) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
    const int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    ncclIbGidState real = {}, expected = {};
    ASSERT_EQ(T().getDev(0, &real), ncclSuccess);
    real.ibDev = 0;
    DevGidGuard guard(T(), 0, real);
    const bool useTwin = AllRanks(FindTwinGid(real, &expected));
    if (useTwin) {
        EXPECT_EQ(T().setDev(0, &expected), ncclSuccess);
    } else {
        expected = real;
    }

    void* listenComm = nullptr;
    void* sendComm = nullptr;
    void* recvComm = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm, &sendComm, &recvComm);

    std::vector<char> buf(kMsgSize, 0x5A);
    void* comm = (rank == 0) ? recvComm : sendComm;
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf.data(), buf.size(), NCCL_PTR_HOST, &mhandle), ncclSuccess);

    ncclIbGidState snap = {};
    EXPECT_EQ(T().getComm(comm, 0, &snap), ncclSuccess);
    snap.ibDev = 0;
    bool ok = SameGid(snap, expected);
    EXPECT_TRUE(ok) << "comm gidIndex " << snap.gidIndex << " expected " << expected.gidIndex
                    << (useTwin ? " (device cache moved to twin)" : "");
    if (expected.linkLayer == IBV_LINK_LAYER_ETHERNET) ok = QpsUseGidIndex(comm, 0, expected.gidIndex) && ok;
    EXPECT_TRUE(AllRanks(ok));

    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// Port recovery applies the current device GID: after the cache moves to the twin index and QP 0
// fails, recovery completes, the comm snapshot, data QPs and recovery path use the twin, and
// traffic flows. Without a twin (IB), a stale comm snapshot is replaced by the real GID.
TEST_P(NetIbGidChangeTest, PortRecoveryRefreshesStaleCommGid) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses, false, kMinGpusPerNode, kNoNodeLimit);
    const char* failoverEnv = getenv("NCCL_IB_RESILIENCY_PORT_FAILOVER");
    const char* recoveryEnv = getenv("NCCL_IB_RESILIENCY_PORT_RECOVERY");
    if (!failoverEnv || strcmp(failoverEnv, "1") != 0) GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_FAILOVER=1";
    if (!recoveryEnv || strcmp(recoveryEnv, "1") != 0) GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_RECOVERY=1";
    if (T().net == &ncclNetIb && !AllRanks(!rcclUseAinic()))
        GTEST_SKIP() << "net_ib port recovery resets QPs, which AINIC does not support";

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

    // With a twin, the device cache is reindexed so recovery has to move every QP and the recovery
    // path of failDev off their connect-time index. Without one, only the comm snapshot is stale.
    ncclIbGidState real = {}, expected = {};
    ASSERT_EQ(T().getComm(comm, failDev, &real), ncclSuccess);
    DevGidGuard guard(T(), real.ibDev, real);
    const bool useTwin = AllRanks(FindTwinGid(real, &expected));
    if (useTwin) {
        EXPECT_EQ(T().setDev(real.ibDev, &expected), ncclSuccess);
    } else {
        expected = real;
        const ncclIbGidState stale = MakeStale(real);
        EXPECT_EQ(T().setComm(comm, failDev, &stale), ncclSuccess);
        EXPECT_EQ(T().changeEvent(real.ibDev), ncclSuccess);
    }
    ncclIbGidState devBeforeRecovery = {};
    EXPECT_EQ(T().getDev(real.ibDev, &devBeforeRecovery), ncclSuccess);
    devBeforeRecovery.ibDev = real.ibDev;
    EXPECT_TRUE(SameGid(devBeforeRecovery, expected));

    bool ok = true;
    void* recvReq = nullptr;
    if (rank == 0) {
        void* bufs[1] = {regBuf};
        size_t sizes[1] = {kMsgSize};
        int tags[1] = {2301};
        void* handles[1] = {mhandle};
        const bool posted = PostRecv(recvComm, 1, bufs, sizes, tags, handles, &recvReq) == ncclSuccess;
        EXPECT_TRUE(posted);
        ok = ok && posted;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    EXPECT_EQ(T().driveQpToError(comm, 0), ncclSuccess);
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
        if (T().getDevState(comm, failDev, &state) != ncclSuccess) break;
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
    EXPECT_EQ(T().getComm(comm, failDev, &snap), ncclSuccess);
    const bool snapOk = SameGid(snap, expected);
    EXPECT_TRUE(snapOk) << "comm gidIndex after recovery " << snap.gidIndex << " expected " << expected.gidIndex;
    ok = ok && snapOk;
    if (expected.linkLayer == IBV_LINK_LAYER_ETHERNET) {
        ok = QpsUseGidIndex(comm, failDev, expected.gidIndex) && ok;
        int recoveryGidIndex = -1;
        EXPECT_EQ(T().getRecoveryGidIndex(comm, failDev, &recoveryGidIndex), ncclSuccess);
        EXPECT_EQ(recoveryGidIndex, expected.gidIndex) << "port recovery path of devIndex " << failDev;
        ok = ok && recoveryGidIndex == expected.gidIndex;
    }

    bool trafficOk = true;
    for (int m = 0; m < kPostRecoveryMsgs && trafficOk; m++) {
        const size_t off = (m + 1) * kMsgSize;
        void* req = nullptr;
        int sz = 0;
        if (rank == 0) {
            void* bufs[1] = {recvBuf.data() + off};
            size_t sizes[1] = {kMsgSize};
            int tags[1] = {2310 + m};
            void* handles[1] = {mhandle};
            trafficOk = PostRecv(recvComm, 1, bufs, sizes, tags, handles, &req) == ncclSuccess &&
                        WaitForCompletion(req, &sz, 10000) == ncclSuccess;
        } else {
            PostSendWithRetry(sendComm, sendBuf.data() + off, kMsgSize, 2310 + m, mhandle, &req);
            trafficOk = req != nullptr && WaitForCompletion(req, &sz, 10000) == ncclSuccess;
        }
        EXPECT_TRUE(trafficOk) << "post-recovery message " << m;
    }
    ok = ok && trafficOk;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 && trafficOk) {
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
