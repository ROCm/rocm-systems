/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Multi-node whitebox tests for the remote GDR flush flag (RCCL_IB_GDR_FLUSH_FLAG=1)
// when consecutive operations on the same NET connection switch protocol.
//
// The proxy decides the flag per receive step: only SIMPLE steps of large
// AllReduce/ReduceScatter ask for it (gdrFlushFlagForOp); LL/LL128 steps and all
// P2P steps keep the RDMA read flush. A flagged step publishes a descriptor into
// gdrFlushSeq[slot] that prims_simple waits on; an unflagged step leaves the slot
// as it was, relying on any stale descriptor naming an already-completed step.
//
// Ops are enqueued back-to-back on one stream without synchronizing, so the
// proxy and kernels of one protocol are still in flight when the next op starts
// on the other protocol. Each test then checks data on every rank and walks the
// NET recv connectors: every published descriptor must use the expected QP count
// and already be satisfied by its flags, and P2P-only traffic must never publish.
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

namespace {

class GdrFlushFlagProtoSwitchMPITest : public MPITestBase {
protected:
    static int envInt(const char* name, int def) {
        const char* v = getenv(name);
        return (v && *v) ? atoi(v) : def;
    }
    static int expectedFlagCount() {
        return envInt("NCCL_IB_SPLIT_DATA_ON_QPS", 0) ? std::max(1, envInt("NCCL_IB_QPS_PER_CONNECTION", 1)) : 1;
    }

    // Collective skip reason, empty to run.
    std::string SkipReason() {
        std::string why;
        if (!validateTestPrerequisites(kMinProcessesForMPI, kNoProcessLimit, kNoPowerOfTwoRequired, /*min_nodes=*/2))
            why = "requires >= 2 processes on >= 2 nodes";
        else if (envInt("RCCL_IB_GDR_FLUSH_FLAG", 0) == 0)
            why = "requires RCCL_IB_GDR_FLUSH_FLAG=1";
        int skip = why.empty() ? 0 : 1;
        MPI_Allreduce(MPI_IN_PLACE, &skip, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (skip && why.empty()) why = "another rank cannot run this test";
        return skip ? why : std::string();
    }

    struct FlagState {
        int flagConns = 0;       // NET recv connectors with the flag mapped
        int publishedConns = 0;  // ... of which have a non-zero descriptor
        int badDescs = 0;        // descriptors with a wrong QP count or an unmet flag
        std::string firstBad;
    };

    // Walks every NET recv connector. After the stream drained, any published
    // descriptor belongs to a completed flagged step, so its flags are at or
    // past its sequence.
    FlagState InspectFlagState(ncclComm_t comm) {
        FlagState st;
        const int count = expectedFlagCount();
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
                        const uint64_t seq = desc & NCCL_GDR_FLUSH_SEQ_MASK;
                        bool ok = (n == count);
                        for (int q = 0; ok && q < n; q++) ok = __atomic_load_n(flags + q, __ATOMIC_ACQUIRE) >= seq;
                        if (!ok) {
                            if (st.badDescs++ == 0) {
                                st.firstBad = "channel " + std::to_string(c) + " peer " + std::to_string(peer) +
                                              " conn " + std::to_string(ci) + " slot " + std::to_string(s) +
                                              ": count " + std::to_string(n) + " (expected " +
                                              std::to_string(count) + ") seq " + std::to_string(seq) +
                                              " flag0 " + std::to_string(__atomic_load_n(flags, __ATOMIC_ACQUIRE));
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
// protocols. P2P never asks for the flag, so the negotiated connections must
// fall back to the read flush on every SIMPLE step and never publish.
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
    for (size_t b = 0; b < sendBufs.size(); b++) {
        ASSERT_EQ(hipMalloc(&sendBufs[b], maxSize), hipSuccess);
        ASSERT_EQ(hipMalloc(&recvBufs[b], maxSize), hipSuccess);
    }
    auto freeAll = [&]() {
        for (size_t b = 0; b < sendBufs.size(); b++) {
            (void)hipFree(sendBufs[b]);
            (void)hipFree(recvBufs[b]);
        }
    };

    std::vector<uint8_t> host(maxSize);
    bool dataOk = true;
    std::string firstErr;
    for (int iter = 0; iter < kIters && dataOk; iter++) {
        for (int op = 0; op < kOps; op++) {
            for (int j = 0; j < nPeers; j++) {
                const int dst = (rank + 1 + j) % nRanks;
                const size_t n = sizeFor(op, j, iter);
                for (size_t i = 0; i < n; i++) host[i] = p2pByte(rank, dst, op, iter, i);
                ASSERT_EQ(hipMemcpy(sendBufs[(size_t)op * nPeers + j], host.data(), n, hipMemcpyHostToDevice),
                          hipSuccess);
                ASSERT_EQ(hipMemset(recvBufs[(size_t)op * nPeers + j], 0xA5, n), hipSuccess);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        for (int op = 0; op < kOps; op++) {
            ASSERT_EQ(ncclGroupStart(), ncclSuccess);
            for (int j = 0; j < nPeers; j++) {
                const int dst = (rank + 1 + j) % nRanks;
                const int src = (rank - 1 - j + 2 * nRanks) % nRanks;
                ASSERT_EQ(ncclSend(sendBufs[(size_t)op * nPeers + j], sizeFor(op, j, iter), ncclUint8, dst, comm,
                                   stream), ncclSuccess);
                ASSERT_EQ(ncclRecv(recvBufs[(size_t)op * nPeers + j], sizeFor(op, j, iter), ncclUint8, src, comm,
                                   stream), ncclSuccess);
            }
            ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
        }
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        for (int op = 0; op < kOps && dataOk; op++) {
            for (int j = 0; j < nPeers && dataOk; j++) {
                const int src = (rank - 1 - j + 2 * nRanks) % nRanks;
                const size_t n = sizeFor(op, j, iter);
                ASSERT_EQ(hipMemcpy(host.data(), recvBufs[(size_t)op * nPeers + j], n, hipMemcpyDeviceToHost),
                          hipSuccess);
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
        dataOk = AllOk(dataOk) && dataOk;
    }
    EXPECT_TRUE(dataOk) << "rank " << rank << ": " << firstErr;

    const FlagState st = InspectFlagState(comm);
    EXPECT_EQ(st.publishedConns, 0) << "P2P-only traffic published a flag descriptor on " << st.publishedConns
                                    << " connection(s); first: " << st.firstBad;
    EXPECT_EQ(st.badDescs, 0) << st.firstBad;
    int flagConns = st.flagConns;
    MPI_Allreduce(MPI_IN_PLACE, &flagConns, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) TEST_INFO("NET recv connections with the flag mapped (all ranks): %d", flagConns);
    EXPECT_GT(flagConns, 0) << "no NET recv connection negotiated the flag";

    freeAll();
    MPI_Barrier(MPI_COMM_WORLD);
}

// Collective connections switching between latency protocols and flagged
// SIMPLE: small AllReduce, large AllReduce/ReduceScatter above the default
// flag threshold, and small P2P sends interleaved, all queued back-to-back.
// Stale gdrFlushSeq slots left by LL/LL128 and unflagged steps must never
// make the GPU read data early, and every published descriptor must be met.
TEST_F(GdrFlushFlagProtoSwitchMPITest, Collectives_LatencyToFlaggedSimpleSwitch_BackToBack) {
    const std::string why = SkipReason();
    if (!why.empty()) GTEST_SKIP() << why;
    ASSERT_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    const int rank = MPIEnvironment::world_rank;
    const int nRanks = MPIEnvironment::world_size;

    // 36 Mi floats = 144 MiB, above the default 128 MiB flag threshold.
    const size_t kLargeCount = (size_t)36 << 20;
    const size_t kSmallCounts[] = {16, 1000, 8192, 65536};
    const size_t kP2pBytes = 6000;
    constexpr int kIters = 6;
    const size_t rsCount = kLargeCount / nRanks;

    float *arSmallSend = nullptr, *arSmallRecv = nullptr, *arLargeSend = nullptr, *arLargeRecv = nullptr,
          *rsRecv = nullptr;
    void *p2pSend = nullptr, *p2pRecv = nullptr;
    ASSERT_EQ(hipMalloc((void**)&arSmallSend, 65536 * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc((void**)&arSmallRecv, 4 * 65536 * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc((void**)&arLargeSend, kLargeCount * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc((void**)&arLargeRecv, kLargeCount * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc((void**)&rsRecv, rsCount * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&p2pSend, kP2pBytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&p2pRecv, 2 * kP2pBytes), hipSuccess);

    auto inVal = [&](int r, size_t i, int iter) { return (float)((r + 1) * (int)((i + iter) % 13 + 1)); };
    auto sumVal = [&](size_t i, int iter) {
        return (float)((nRanks * (nRanks + 1) / 2) * (int)((i + iter) % 13 + 1));
    };

    std::vector<float> hf(kLargeCount);
    std::vector<uint8_t> hb(2 * kP2pBytes);
    const int next = (rank + 1) % nRanks, prev = (rank - 1 + nRanks) % nRanks;
    bool dataOk = true;
    std::string firstErr;
    for (int iter = 0; iter < kIters && dataOk; iter++) {
        for (size_t i = 0; i < kLargeCount; i++) hf[i] = inVal(rank, i, iter);
        ASSERT_EQ(hipMemcpy(arLargeSend, hf.data(), kLargeCount * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(arSmallSend, hf.data(), 65536 * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
        for (size_t i = 0; i < kP2pBytes; i++) hb[i] = p2pByte(rank, next, 0, iter, i);
        ASSERT_EQ(hipMemcpy(p2pSend, hb.data(), kP2pBytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(arLargeRecv, 0, kLargeCount * sizeof(float)), hipSuccess);
        ASSERT_EQ(hipMemset(rsRecv, 0, rsCount * sizeof(float)), hipSuccess);
        ASSERT_EQ(hipMemset(arSmallRecv, 0, 4 * 65536 * sizeof(float)), hipSuccess);
        MPI_Barrier(MPI_COMM_WORLD);

        // latency AR -> flagged AR -> latency AR + P2P -> flagged RS -> latency AR -> P2P
        ASSERT_EQ(ncclAllReduce(arSmallSend, arSmallRecv, kSmallCounts[iter % 4], ncclFloat, ncclSum, comm, stream),
                  ncclSuccess);
        ASSERT_EQ(ncclAllReduce(arLargeSend, arLargeRecv, kLargeCount, ncclFloat, ncclSum, comm, stream),
                  ncclSuccess);
        ASSERT_EQ(ncclAllReduce(arSmallSend, arSmallRecv + 65536, kSmallCounts[(iter + 1) % 4], ncclFloat, ncclSum,
                                comm, stream), ncclSuccess);
        ASSERT_EQ(ncclGroupStart(), ncclSuccess);
        ASSERT_EQ(ncclSend(p2pSend, kP2pBytes, ncclUint8, next, comm, stream), ncclSuccess);
        ASSERT_EQ(ncclRecv(p2pRecv, kP2pBytes, ncclUint8, prev, comm, stream), ncclSuccess);
        ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
        ASSERT_EQ(ncclReduceScatter(arLargeSend, rsRecv, rsCount, ncclFloat, ncclSum, comm, stream), ncclSuccess);
        ASSERT_EQ(ncclAllReduce(arSmallSend, arSmallRecv + 2 * 65536, kSmallCounts[(iter + 2) % 4], ncclFloat,
                                ncclSum, comm, stream), ncclSuccess);
        ASSERT_EQ(ncclGroupStart(), ncclSuccess);
        ASSERT_EQ(ncclSend(p2pSend, kP2pBytes, ncclUint8, next, comm, stream), ncclSuccess);
        ASSERT_EQ(ncclRecv((char*)p2pRecv + kP2pBytes, kP2pBytes, ncclUint8, prev, comm, stream), ncclSuccess);
        ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        auto fail = [&](const std::string& what, size_t i, float got, float exp) {
            dataOk = false;
            firstErr = "iter " + std::to_string(iter) + " " + what + "[" + std::to_string(i) +
                       "] got " + std::to_string(got) + " expected " + std::to_string(exp);
        };
        ASSERT_EQ(hipMemcpy(hf.data(), arLargeRecv, kLargeCount * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
        for (size_t i = 0; i < kLargeCount && dataOk; i++)
            if (hf[i] != sumVal(i, iter)) fail("large allreduce", i, hf[i], sumVal(i, iter));
        ASSERT_EQ(hipMemcpy(hf.data(), rsRecv, rsCount * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
        for (size_t j = 0; j < rsCount && dataOk; j++)
            if (hf[j] != sumVal(rank * rsCount + j, iter)) fail("reducescatter", j, hf[j], sumVal(rank * rsCount + j, iter));
        ASSERT_EQ(hipMemcpy(hf.data(), arSmallRecv, 3 * 65536 * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
        for (int k = 0; k < 3 && dataOk; k++) {
            const size_t cnt = kSmallCounts[(iter + k) % 4];
            for (size_t i = 0; i < cnt && dataOk; i++)
                if (hf[k * 65536 + i] != sumVal(i, iter)) fail("small allreduce " + std::to_string(k), i,
                                                              hf[k * 65536 + i], sumVal(i, iter));
        }
        ASSERT_EQ(hipMemcpy(hb.data(), p2pRecv, 2 * kP2pBytes, hipMemcpyDeviceToHost), hipSuccess);
        for (size_t i = 0; i < 2 * kP2pBytes && dataOk; i++) {
            const uint8_t exp = p2pByte(prev, rank, 0, iter, i % kP2pBytes);
            if (hb[i] != exp) fail("p2p", i, hb[i], exp);
        }
        dataOk = AllOk(dataOk) && dataOk;
    }
    EXPECT_TRUE(dataOk) << "rank " << rank << ": " << firstErr;

    const FlagState st = InspectFlagState(comm);
    EXPECT_EQ(st.badDescs, 0) << "rank " << rank << ": " << st.badDescs << " unmet descriptor(s), first " << st.firstBad;
    int totals[2] = {st.flagConns, st.publishedConns};
    MPI_Allreduce(MPI_IN_PLACE, totals, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) TEST_INFO("flag-mapped NET recv connections: %d, with published descriptors: %d", totals[0], totals[1]);
    EXPECT_GT(totals[0], 0) << "no NET recv connection negotiated the flag";
    EXPECT_GT(totals[1], 0) << "large AllReduce/ReduceScatter never published a flag descriptor";

    (void)hipFree(arSmallSend);
    (void)hipFree(arSmallRecv);
    (void)hipFree(arLargeSend);
    (void)hipFree(arLargeRecv);
    (void)hipFree(rsRecv);
    (void)hipFree(p2pSend);
    (void)hipFree(p2pRecv);
    MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace

#endif  // MPI_TESTS_ENABLED
