/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Multi-node whitebox tests for the remote GDR flush flag (RCCL_IB_GDR_FLUSH_FLAG=1)
// when consecutive operations on the same NET connection switch protocol.
//
// On a connection that negotiated the flag, every SIMPLE receive step asks for it
// and publishes a descriptor into gdrFlushSeq[slot] that prims_simple waits on.
// LL/LL128 steps leave the slot as it was, relying on any stale descriptor naming
// an already-completed step.
//
// Ops are enqueued back-to-back on one stream without synchronizing, so the
// proxy and kernels of one protocol are still in flight when the next op starts
// on the other protocol. Each test then checks data on every rank and walks the
// NET recv connectors: every published descriptor must name QPs the connection
// can use and already be satisfied by its flags.
//
// Run with >= 2 nodes. NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 selects LL128 (instead of
// LL) for small P2P on gfx942/gfx950; run under both settings and under each
// NCCL_IB_QPS_PER_CONNECTION / NCCL_IB_SPLIT_DATA_ON_QPS configuration.

#ifdef MPI_TESTS_ENABLED

#include "MPITestBase.hpp"

#include "comm.h"
#include "device.h"
#include "transport.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <mpi.h>

using namespace MPITestConstants;

// Records a failed call and returns false from the enclosing lambda, so the
// ranks can agree on the outcome before any of them leaves the loop.
#define PROTO_SWITCH_CHECK(expr)          \
    do {                                  \
        if (!(expr)) {                    \
            ADD_FAILURE() << #expr;       \
            return false;                 \
        }                                 \
    } while (0)

namespace {

class GdrFlushFlagProtoSwitchMPITest : public MPITestBase {
protected:
    static int envInt(const char* name, int def) {
        const char* v = getenv(name);
        return (v && *v) ? atoi(v) : def;
    }
    static int configuredQps() { return std::max(1, envInt("NCCL_IB_QPS_PER_CONNECTION", 1)); }
    // Whether `count` flags from `first` are what a connection with either the
    // regular or the P2P (RCCL_IB_QPS_PER_P2P) QP count would use.
    static bool validQpSet(int first, int count) {
        const bool split = envInt("NCCL_IB_SPLIT_DATA_ON_QPS", 0) != 0;
        for (int nqps : {configuredQps(), envInt("RCCL_IB_QPS_PER_P2P", 0)}) {
            if (nqps <= 0) continue;
            if (count == (split ? nqps : 1) && first % count == 0 && first + count <= nqps) return true;
        }
        return false;
    }
    static uint64_t loadFlag(const uint64_t* flags, int q) {
        return __atomic_load_n(flags + q * (NCCL_GDR_FLUSH_FLAG_STRIDE / sizeof(uint64_t)), __ATOMIC_ACQUIRE);
    }

    // Collective skip reason, empty to run.
    std::string SkipReason() {
        std::string why;
        if (!validateTestPrerequisites(kMinProcessesForMPI, kNoProcessLimit, kNoPowerOfTwoRequired, /*min_nodes=*/2))
            why = "requires >= 2 processes on >= 2 nodes";
        else if (envInt("RCCL_IB_GDR_FLUSH_FLAG", 0) == 0)
            why = "requires RCCL_IB_GDR_FLUSH_FLAG=1";
        else if (envInt("NCCL_IB_SPLIT_DATA_ON_QPS", 0) != 0 &&
                 std::max(configuredQps(), envInt("RCCL_IB_QPS_PER_P2P", 0)) > NCCL_GDR_FLUSH_MAX_WAIT_QPS)
            why = "the flag needs at most NCCL_GDR_FLUSH_MAX_WAIT_QPS QPs per request";
        int skip = why.empty() ? 0 : 1;
        MPI_Allreduce(MPI_IN_PLACE, &skip, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (skip && why.empty()) why = "another rank cannot run this test";
        return skip ? why : std::string();
    }

    struct FlagState {
        int flagConns = 0;       // NET recv connectors with the flag mapped
        int publishedConns = 0;  // ... of which have a non-zero descriptor
        int badDescs = 0;        // descriptors with wrong QPs or an unmet flag
        std::string firstBad;
    };

    // Walks every NET recv connector. After the stream drained, any published
    // descriptor belongs to a completed flagged step, so its flags are at or
    // past its sequence.
    FlagState InspectFlagState(ncclComm_t comm) {
        FlagState st;
        for (int c = 0; c < MAXCHANNELS; c++) {
            struct ncclChannel* ch = &comm->channels[c];
            if (ch->peers == nullptr) continue;
            for (int peer = 0; peer < comm->nRanks; peer++) {
                struct ncclChannelPeer* p = ch->peers[peer];
                if (p == nullptr) continue;
                for (int ci = 0; ci < NCCL_MAX_CONNS; ci++) {
                    struct ncclConnector* rc = &p->recv[ci];
                    if (!rc->connected || rc->transportComm != &netTransport.recv) continue;
                    const uint64_t* flags = rc->conn.gdrFlushFlag;
                    const uint64_t* seqs = rc->conn.gdrFlushSeq;
                    if (flags == nullptr || seqs == nullptr) continue;
                    st.flagConns++;
                    bool published = false;
                    for (int s = 0; s < NCCL_STEPS; s++) {
                        const uint64_t desc = __atomic_load_n(seqs + s, __ATOMIC_ACQUIRE);
                        if (desc == 0) continue;
                        published = true;
                        const int n = (int)(desc >> NCCL_GDR_FLUSH_COUNT_SHIFT);
                        const int first = (int)((desc >> NCCL_GDR_FLUSH_FIRST_SHIFT) & 0xff);
                        const uint64_t seq = desc & NCCL_GDR_FLUSH_SEQ_MASK;
                        bool ok = validQpSet(first, n);
                        for (int q = first; ok && q < first + n; q++) ok = loadFlag(flags, q) >= seq;
                        if (!ok) {
                            if (st.badDescs++ == 0) {
                                st.firstBad = "channel " + std::to_string(c) + " peer " + std::to_string(peer) +
                                              " conn " + std::to_string(ci) + " slot " + std::to_string(s) +
                                              ": first " + std::to_string(first) + " count " + std::to_string(n) +
                                              " seq " + std::to_string(seq) + " flag" + std::to_string(first) + " " +
                                              std::to_string(loadFlag(flags, first));
                            }
                        }
                    }
                    if (published) st.publishedConns++;
                }
            }
        }
        return st;
    }

    static uint8_t p2pByte(int src, int dst, int op, int iter, size_t i) {
        return (uint8_t)(src * 31 + dst * 17 + op * 7 + iter * 3 + i);
    }

    // Collective verdict: true on every rank only if `ok` holds on every rank.
    static bool AllOk(bool ok) {
        int v = ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return v != 0;
    }
};

// All-to-all ncclSend/ncclRecv where every peer's size flips between the
// latency protocol (LL128, or LL without NCCL_ALLOC_P2P_NET_LL_BUFFERS) and
// SIMPLE on each consecutive op, and peers in the same group use different
// protocols, so flagged SIMPLE steps follow LL/LL128 steps on every connection.
TEST_F(GdrFlushFlagProtoSwitchMPITest, P2pAllToAll_Ll128SimpleSwitch_BackToBack) {
    const std::string why = SkipReason();
    if (!why.empty()) GTEST_SKIP() << why;
    ASSERT_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    const int rank = MPIEnvironment::world_rank;
    const int nRanks = MPIEnvironment::world_size;
    if (rank == 0)
        TEST_INFO("allocP2pNetLLBuffers=%d nChannels=%d p2pnChannels=%d", comm->allocP2pNetLLBuffers,
                  comm->nChannels, comm->p2pnChannels);

    const size_t kSmall[] = {1, 100, 4096, 12288};
    const size_t kLarge[] = {(64 << 10) + 3, (1 << 20) + 7, 4 << 20};
    const size_t maxSize = 4 << 20;
    constexpr int kOps = 12;
    constexpr int kIters = 16;
    const int nPeers = nRanks - 1;

    auto sizeFor = [&](int op, int j, int iter) {
        return ((op + j) & 1) ? kLarge[(op / 2 + j + iter) % 3] : kSmall[(op / 2 + j + iter) % 4];
    };

    std::vector<void*> sendBufs((size_t)kOps * nPeers, nullptr), recvBufs((size_t)kOps * nPeers, nullptr);
    bool allocOk = true;
    for (size_t b = 0; b < sendBufs.size(); b++) {
        allocOk &= hipMalloc(&sendBufs[b], maxSize) == hipSuccess;
        allocOk &= hipMalloc(&recvBufs[b], maxSize) == hipSuccess;
    }
    auto freeAll = [&]() {
        for (size_t b = 0; b < sendBufs.size(); b++) {
            if (sendBufs[b]) (void)hipFree(sendBufs[b]);
            if (recvBufs[b]) (void)hipFree(recvBufs[b]);
        }
    };
    if (!AllOk(allocOk)) {
        ADD_FAILURE() << "hipMalloc failed on at least one rank";
        freeAll();
        return;
    }

    std::vector<uint8_t> host(maxSize);
    bool dataOk = true;
    std::string firstErr;
    auto fillBuffers = [&](int iter) -> bool {
        for (int op = 0; op < kOps; op++) {
            for (int j = 0; j < nPeers; j++) {
                const int dst = (rank + 1 + j) % nRanks;
                const size_t n = sizeFor(op, j, iter);
                for (size_t i = 0; i < n; i++) host[i] = p2pByte(rank, dst, op, iter, i);
                PROTO_SWITCH_CHECK(hipMemcpy(sendBufs[(size_t)op * nPeers + j], host.data(), n,
                                             hipMemcpyHostToDevice) == hipSuccess);
                PROTO_SWITCH_CHECK(hipMemset(recvBufs[(size_t)op * nPeers + j], 0xA5, n) == hipSuccess);
            }
        }
        return true;
    };
    auto runOps = [&](int iter) -> bool {
        for (int op = 0; op < kOps; op++) {
            PROTO_SWITCH_CHECK(ncclGroupStart() == ncclSuccess);
            for (int j = 0; j < nPeers; j++) {
                const int dst = (rank + 1 + j) % nRanks;
                const int src = (rank - 1 - j + 2 * nRanks) % nRanks;
                PROTO_SWITCH_CHECK(ncclSend(sendBufs[(size_t)op * nPeers + j], sizeFor(op, j, iter), ncclUint8, dst,
                                            comm, stream) == ncclSuccess);
                PROTO_SWITCH_CHECK(ncclRecv(recvBufs[(size_t)op * nPeers + j], sizeFor(op, j, iter), ncclUint8, src,
                                            comm, stream) == ncclSuccess);
            }
            PROTO_SWITCH_CHECK(ncclGroupEnd() == ncclSuccess);
        }
        PROTO_SWITCH_CHECK(hipStreamSynchronize(stream) == hipSuccess);
        return true;
    };
    auto verify = [&](int iter) -> bool {
        for (int op = 0; op < kOps && dataOk; op++) {
            for (int j = 0; j < nPeers && dataOk; j++) {
                const int src = (rank - 1 - j + 2 * nRanks) % nRanks;
                const size_t n = sizeFor(op, j, iter);
                PROTO_SWITCH_CHECK(hipMemcpy(host.data(), recvBufs[(size_t)op * nPeers + j], n,
                                             hipMemcpyDeviceToHost) == hipSuccess);
                for (size_t i = 0; i < n; i++) {
                    if (host[i] != p2pByte(src, rank, op, iter, i)) {
                        dataOk = false;
                        firstErr = "iter " + std::to_string(iter) + " op " + std::to_string(op) + " from rank " +
                                   std::to_string(src) + " size " + std::to_string(n) + ": byte " +
                                   std::to_string(i) + " got " + std::to_string(host[i]) + " expected " +
                                   std::to_string(p2pByte(src, rank, op, iter, i));
                        break;
                    }
                }
            }
        }
        return true;
    };
    for (int iter = 0; iter < kIters && dataOk; iter++) {
        if (!AllOk(fillBuffers(iter))) break;
        MPI_Barrier(MPI_COMM_WORLD);
        const bool ran = runOps(iter) && verify(iter);
        if (!AllOk(ran)) break;
        dataOk = AllOk(dataOk) && dataOk;
    }
    EXPECT_TRUE(dataOk) << "rank " << rank << ": " << firstErr;

    const FlagState st = InspectFlagState(comm);
    EXPECT_EQ(st.badDescs, 0) << "rank " << rank << ": " << st.badDescs << " unmet descriptor(s), first " << st.firstBad;
    int totals[2] = {st.flagConns, st.publishedConns};
    MPI_Allreduce(MPI_IN_PLACE, totals, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) TEST_INFO("flag-mapped NET recv connections: %d, with published descriptors: %d", totals[0], totals[1]);
    EXPECT_GT(totals[0], 0) << "no NET recv connection negotiated the flag";
    EXPECT_GT(totals[1], 0) << "SIMPLE P2P steps never published a flag descriptor";

    freeAll();
    MPI_Barrier(MPI_COMM_WORLD);
}

// Collective connections switching between latency protocols and flagged
// SIMPLE: small AllReduce, large AllReduce/ReduceScatter and small P2P sends
// interleaved, all queued back-to-back. Stale gdrFlushSeq slots left by
// LL/LL128 steps must never make the GPU read data early, and every published
// descriptor must be met.
TEST_F(GdrFlushFlagProtoSwitchMPITest, Collectives_LatencyToFlaggedSimpleSwitch_BackToBack) {
    const std::string why = SkipReason();
    if (!why.empty()) GTEST_SKIP() << why;
    ASSERT_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    const int rank = MPIEnvironment::world_rank;
    const int nRanks = MPIEnvironment::world_size;

    // 36 Mi floats = 144 MiB, deep in the SIMPLE range.
    const size_t kLargeCount = (size_t)36 << 20;
    const size_t kSmallCounts[] = {16, 1000, 8192, 65536};
    const size_t kP2pBytes = 6000;
    constexpr int kIters = 6;
    const size_t rsCount = kLargeCount / nRanks;

    float *arSmallSend = nullptr, *arSmallRecv = nullptr, *arLargeSend = nullptr, *arLargeRecv = nullptr,
          *rsRecv = nullptr;
    void *p2pSend = nullptr, *p2pRecv = nullptr;
    bool allocOk = true;
    allocOk &= hipMalloc((void**)&arSmallSend, 65536 * sizeof(float)) == hipSuccess;
    allocOk &= hipMalloc((void**)&arSmallRecv, 4 * 65536 * sizeof(float)) == hipSuccess;
    allocOk &= hipMalloc((void**)&arLargeSend, kLargeCount * sizeof(float)) == hipSuccess;
    allocOk &= hipMalloc((void**)&arLargeRecv, kLargeCount * sizeof(float)) == hipSuccess;
    allocOk &= hipMalloc((void**)&rsRecv, rsCount * sizeof(float)) == hipSuccess;
    allocOk &= hipMalloc(&p2pSend, kP2pBytes) == hipSuccess;
    allocOk &= hipMalloc(&p2pRecv, 2 * kP2pBytes) == hipSuccess;
    auto freeAll = [&]() {
        if (arSmallSend) (void)hipFree(arSmallSend);
        if (arSmallRecv) (void)hipFree(arSmallRecv);
        if (arLargeSend) (void)hipFree(arLargeSend);
        if (arLargeRecv) (void)hipFree(arLargeRecv);
        if (rsRecv) (void)hipFree(rsRecv);
        if (p2pSend) (void)hipFree(p2pSend);
        if (p2pRecv) (void)hipFree(p2pRecv);
    };
    if (!AllOk(allocOk)) {
        ADD_FAILURE() << "hipMalloc failed on at least one rank";
        freeAll();
        return;
    }

    auto inVal = [&](int r, size_t i, int iter) { return (float)((r + 1) * (int)((i + iter) % 13 + 1)); };
    auto sumVal = [&](size_t i, int iter) {
        return (float)((nRanks * (nRanks + 1) / 2) * (int)((i + iter) % 13 + 1));
    };

    std::vector<float> hf(kLargeCount);
    std::vector<uint8_t> hb(2 * kP2pBytes);
    const int next = (rank + 1) % nRanks, prev = (rank - 1 + nRanks) % nRanks;
    bool dataOk = true;
    std::string firstErr;
    auto fillBuffers = [&](int iter) -> bool {
        for (size_t i = 0; i < kLargeCount; i++) hf[i] = inVal(rank, i, iter);
        PROTO_SWITCH_CHECK(hipMemcpy(arLargeSend, hf.data(), kLargeCount * sizeof(float), hipMemcpyHostToDevice) ==
                           hipSuccess);
        PROTO_SWITCH_CHECK(hipMemcpy(arSmallSend, hf.data(), 65536 * sizeof(float), hipMemcpyHostToDevice) ==
                           hipSuccess);
        for (size_t i = 0; i < kP2pBytes; i++) hb[i] = p2pByte(rank, next, 0, iter, i);
        PROTO_SWITCH_CHECK(hipMemcpy(p2pSend, hb.data(), kP2pBytes, hipMemcpyHostToDevice) == hipSuccess);
        PROTO_SWITCH_CHECK(hipMemset(arLargeRecv, 0, kLargeCount * sizeof(float)) == hipSuccess);
        PROTO_SWITCH_CHECK(hipMemset(rsRecv, 0, rsCount * sizeof(float)) == hipSuccess);
        PROTO_SWITCH_CHECK(hipMemset(arSmallRecv, 0, 4 * 65536 * sizeof(float)) == hipSuccess);
        return true;
    };
    // latency AR -> flagged AR -> latency AR + P2P -> flagged RS -> latency AR -> P2P
    auto runOps = [&](int iter) -> bool {
        PROTO_SWITCH_CHECK(ncclAllReduce(arSmallSend, arSmallRecv, kSmallCounts[iter % 4], ncclFloat, ncclSum, comm,
                                         stream) == ncclSuccess);
        PROTO_SWITCH_CHECK(ncclAllReduce(arLargeSend, arLargeRecv, kLargeCount, ncclFloat, ncclSum, comm, stream) ==
                           ncclSuccess);
        PROTO_SWITCH_CHECK(ncclAllReduce(arSmallSend, arSmallRecv + 65536, kSmallCounts[(iter + 1) % 4], ncclFloat,
                                         ncclSum, comm, stream) == ncclSuccess);
        PROTO_SWITCH_CHECK(ncclGroupStart() == ncclSuccess);
        PROTO_SWITCH_CHECK(ncclSend(p2pSend, kP2pBytes, ncclUint8, next, comm, stream) == ncclSuccess);
        PROTO_SWITCH_CHECK(ncclRecv(p2pRecv, kP2pBytes, ncclUint8, prev, comm, stream) == ncclSuccess);
        PROTO_SWITCH_CHECK(ncclGroupEnd() == ncclSuccess);
        PROTO_SWITCH_CHECK(ncclReduceScatter(arLargeSend, rsRecv, rsCount, ncclFloat, ncclSum, comm, stream) ==
                           ncclSuccess);
        PROTO_SWITCH_CHECK(ncclAllReduce(arSmallSend, arSmallRecv + 2 * 65536, kSmallCounts[(iter + 2) % 4],
                                         ncclFloat, ncclSum, comm, stream) == ncclSuccess);
        PROTO_SWITCH_CHECK(ncclGroupStart() == ncclSuccess);
        PROTO_SWITCH_CHECK(ncclSend(p2pSend, kP2pBytes, ncclUint8, next, comm, stream) == ncclSuccess);
        PROTO_SWITCH_CHECK(ncclRecv((char*)p2pRecv + kP2pBytes, kP2pBytes, ncclUint8, prev, comm, stream) ==
                           ncclSuccess);
        PROTO_SWITCH_CHECK(ncclGroupEnd() == ncclSuccess);
        PROTO_SWITCH_CHECK(hipStreamSynchronize(stream) == hipSuccess);
        return true;
    };
    auto verify = [&](int iter) -> bool {
        auto fail = [&](const std::string& what, size_t i, float got, float exp) {
            dataOk = false;
            firstErr = "iter " + std::to_string(iter) + " " + what + "[" + std::to_string(i) +
                       "] got " + std::to_string(got) + " expected " + std::to_string(exp);
        };
        PROTO_SWITCH_CHECK(hipMemcpy(hf.data(), arLargeRecv, kLargeCount * sizeof(float), hipMemcpyDeviceToHost) ==
                           hipSuccess);
        for (size_t i = 0; i < kLargeCount && dataOk; i++)
            if (hf[i] != sumVal(i, iter)) fail("large allreduce", i, hf[i], sumVal(i, iter));
        PROTO_SWITCH_CHECK(hipMemcpy(hf.data(), rsRecv, rsCount * sizeof(float), hipMemcpyDeviceToHost) ==
                           hipSuccess);
        for (size_t j = 0; j < rsCount && dataOk; j++)
            if (hf[j] != sumVal(rank * rsCount + j, iter)) fail("reducescatter", j, hf[j], sumVal(rank * rsCount + j, iter));
        PROTO_SWITCH_CHECK(hipMemcpy(hf.data(), arSmallRecv, 3 * 65536 * sizeof(float), hipMemcpyDeviceToHost) ==
                           hipSuccess);
        for (int k = 0; k < 3 && dataOk; k++) {
            const size_t cnt = kSmallCounts[(iter + k) % 4];
            for (size_t i = 0; i < cnt && dataOk; i++)
                if (hf[k * 65536 + i] != sumVal(i, iter)) fail("small allreduce " + std::to_string(k), i,
                                                              hf[k * 65536 + i], sumVal(i, iter));
        }
        PROTO_SWITCH_CHECK(hipMemcpy(hb.data(), p2pRecv, 2 * kP2pBytes, hipMemcpyDeviceToHost) == hipSuccess);
        for (size_t i = 0; i < 2 * kP2pBytes && dataOk; i++) {
            const uint8_t exp = p2pByte(prev, rank, 0, iter, i % kP2pBytes);
            if (hb[i] != exp) fail("p2p", i, hb[i], exp);
        }
        return true;
    };
    for (int iter = 0; iter < kIters && dataOk; iter++) {
        if (!AllOk(fillBuffers(iter))) break;
        MPI_Barrier(MPI_COMM_WORLD);
        const bool ran = runOps(iter) && verify(iter);
        if (!AllOk(ran)) break;
        dataOk = AllOk(dataOk) && dataOk;
    }
    EXPECT_TRUE(dataOk) << "rank " << rank << ": " << firstErr;

    const FlagState st = InspectFlagState(comm);
    EXPECT_EQ(st.badDescs, 0) << "rank " << rank << ": " << st.badDescs << " unmet descriptor(s), first " << st.firstBad;
    int totals[2] = {st.flagConns, st.publishedConns};
    MPI_Allreduce(MPI_IN_PLACE, totals, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) TEST_INFO("flag-mapped NET recv connections: %d, with published descriptors: %d", totals[0], totals[1]);
    EXPECT_GT(totals[0], 0) << "no NET recv connection negotiated the flag";
    EXPECT_GT(totals[1], 0) << "SIMPLE steps never published a flag descriptor";

    freeAll();
    MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace

#endif  // MPI_TESTS_ENABLED
