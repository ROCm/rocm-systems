/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// MPI test suite for QP sharing (RCCL_IB_COMM_NGROUPS /
// RCCL_IB_QP_DEPTH_MULTIPLIER).  2-rank, single-IB-device (dev=0).
//
// Three categories:
//
//   QpShareAlgo*    group-assignment and PRIMARY/SECONDARY bookkeeping,
//                   validated against QpShareRefModel. No data phase.
//   QpShareData*    the data path must be correct at every parameter point.
//   QpShareStress*  scale and churn, with the scale under env control.
//
// Every test derives its expectations from RCCL_IB_COMM_NGROUPS rather than
// pinning one hand-picked value, so all three categories are valid at any
// (ngroups, depth, nconns). 
//
// Both tunables are RCCL_PARAM-backed and therefore cached for the process
// lifetime (src/include/param.h): a sweep over either needs one mpirun per
// point. The connection count is *not* cached and can be varied in-process,
// which is why it is an ordinary env knob (QPSHARE_TEST_NCONNS /
// QPSHARE_TEST_ITERS).

#include "NetIbMPITestBase.hpp"
#include "NetIbQpSharingInspect.hpp"
#include "QpShareRefModel.hpp"

#ifdef MPI_TESTS_ENABLED

namespace {

// Connections opened by the algo sweep: enough for every group to hold two
// comms plus one, so PRIMARY-only, PRIMARY+SECONDARY and (at ngroups=1)
// deep-stacking layouts all occur in one run. Capped so a large ngroups does
// not turn a state-only test into a connection-setup benchmark.
constexpr int kAlgoMaxConns = 64;

// Live connections held by the churn test. Kept at two per group so the test
// stays about pool bookkeeping rather than about RQ depth.
constexpr int kChurnMaxLive = 32;

// Connections the flush test may open. It needs ngroups+1 to guarantee one
// shared pair; past this cap it reports that no pair was formed rather than
// opening an unbounded number of GDR-registered connections.
constexpr int kFlushMaxConns = 128;

int Clamp(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

std::string Stage(const char* what, int i) {
    return std::string(what) + " " + std::to_string(i);
}

// One rank's view of a set of cast connections, paired with the reference
// model's prediction for each.
struct QpShareConns {
    std::vector<void*> listen;
    std::vector<void*> send;
    std::vector<void*> recv;
    std::vector<void*> mine;   // recvComm on rank 0, sendComm on rank 1
    std::vector<QpShareRefModel::Placement> place;

    size_t size() const { return mine.size(); }

    void Add(int rank, void* l, void* s, void* r, QpShareRefModel::Placement p) {
        listen.push_back(l);
        send.push_back(s);
        recv.push_back(r);
        mine.push_back(rank == 0 ? r : s);
        place.push_back(p);
    }

    void Erase(size_t i) {
        listen.erase(listen.begin() + i);
        send.erase(send.begin() + i);
        recv.erase(recv.begin() + i);
        mine.erase(mine.begin() + i);
        place.erase(place.begin() + i);
    }
};

}  // namespace

// =============================================================================
// Test: QpShareAlgoLayoutSweep  (category: algo)
//
// The group-assignment contract, validated at whatever RCCL_IB_COMM_NGROUPS is
// configured. Opens connections one at a time up to min(2*ngroups+1, 64), and
// after every single create re-checks *every* live comm against
// QpShareRefModel: group index, PRIMARY/SECONDARY role, refcount, commId, and
// the physical-QP identity relation (same group => same QP, different groups
// => different QPs). Then tears down in creation order -- which releases each
// group's PRIMARY while its SECONDARYs are still live -- re-checking after
// every destroy.
//
// Subsumes the old QpShareNGroupsOne (ngroups=1), QpSharePrimarySecondaryMix
// (ngroups=2) and QpShareNGroupsExceedsConns (ngroups > nconns) as three
// points of the same sweep rather than three hand-computed layouts.
// =============================================================================
TEST_F(NetIbMPITest, QpShareAlgoLayoutSweep) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    QPSHARE_ENV_CHECK_OR_SKIP();
    const int rank    = MPIEnvironment::world_rank;
    const int ngroups = QpShareEnvNGroups();
    const int target  = std::min(2 * ngroups + 1, kAlgoMaxConns);
    if (2 * ngroups + 1 > kAlgoMaxConns && rank == 0) {
        printf("[QPSHARE-COV] algo sweep capped at %d conns (ngroups=%d would need %d "
               "for two per group)\n", kAlgoMaxConns, ngroups, 2 * ngroups + 1);
    }

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    QpShareRefModel model(ngroups);
    QpShareConns    cs;

    for (int c = 0; c < target; c++) {
        QpShareRefModel::Placement p = model.AssignOnCreate();
        void* l = nullptr; void* s = nullptr; void* r = nullptr;
        if (!TryCastConnection(/*dev=*/0, &l, &s, &r)) {
            model.Release(p);
            ADD_FAILURE()
                << "connection " << c << " of " << target << " failed to establish at "
                << "RCCL_IB_COMM_NGROUPS=" << ngroups << " RCCL_IB_QP_DEPTH_MULTIPLIER="
                << QpShareEnvDepthMultiplier() << ". It would have been comm #"
                << (model.Load(p.group) + 1) << " in group " << p.group
                << ". This test never approaches the shared-QP pool's capacity (target <= "
                << kAlgoMaxConns << " connections), so a connect() failure here means either "
                   "a pool slot leaked by an earlier test in this process (slots are reclaimed "
                   "on close, so this should not occur under normal operation) or a genuine "
                   "hard-fail on pool exhaustion -- check NCCL_DEBUG=WARN for \"shared-QP pool "
                   "exhausted\".";
            break;
        }
        cs.Add(rank, l, s, r, p);
        ExpectQpShareLayout(cs.mine, cs.place, model, Stage("after create", c).c_str());
    }

    ReportQpShareCoverage("algo_layout_sweep", static_cast<int>(cs.size()),
                          model.PeakLoad(), model.Spread());

    // Teardown from the front: releases group PRIMARYs ahead of their
    // SECONDARYs, which is where refcount and pool bookkeeping are most likely
    // to diverge from the model.
    for (int c = 0; cs.size() > 0; c++) {
        model.Release(cs.place[0]);
        CloseCastConnection(cs.listen[0], cs.send[0], cs.recv[0]);
        cs.Erase(0);
        MPI_Barrier(MPI_COMM_WORLD);
        ExpectQpShareLayout(cs.mine, cs.place, model, Stage("after destroy", c).c_str());
    }
}

// =============================================================================
// Test: QpShareAlgoChurnLayout  (category: algo)
//
// Group assignment under interleaved create/destroy, including releases from
// the middle of the live set rather than only the tail. This is where
// occupancy-modulo (groupIdx = live totalRefs % ngroups) stops looking like
// round-robin: after a non-tail release the next comm can land back in a group
// that already has members while another sits empty. That behaviour is
// reproduced by QpShareRefModel and asserted; the resulting imbalance is
// measured and reported (group_spread) but deliberately not asserted -- how
// evenly the rule spreads comms is a design question, not this suite's call.
//
// Also the sharpest check on pool bookkeeping: a slot freed on close but still
// counted by IbCastCountPeerTotalRefcount shifts every subsequent assignment
// and shows up here as a group mismatch.
// =============================================================================
TEST_F(NetIbMPITest, QpShareAlgoChurnLayout) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    QPSHARE_ENV_CHECK_OR_SKIP();
    const int rank    = MPIEnvironment::world_rank;
    const int ngroups = QpShareEnvNGroups();
    const int maxLive = Clamp(2 * ngroups, 2, kChurnMaxLive);
    const int steps   = 3 * maxLive + 6;

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    QpShareRefModel model(ngroups);
    QpShareConns    cs;
    int  observedSpread = 0;
    bool broke          = false;

    // Deterministic and rank-identical: the decision depends only on `step` and
    // the live count, both of which evolve the same way on both ranks.
    for (int step = 0; step < steps && !broke; step++) {
        const bool create = cs.size() == 0 ||
                            (static_cast<int>(cs.size()) < maxLive && (step % 3) != 2);
        if (create) {
            QpShareRefModel::Placement p = model.AssignOnCreate();
            void* l = nullptr; void* s = nullptr; void* r = nullptr;
            if (!TryCastConnection(/*dev=*/0, &l, &s, &r)) {
                model.Release(p);
                ADD_FAILURE()
                    << "step " << step << ": connect failed with " << cs.size()
                    << " live comms at RCCL_IB_COMM_NGROUPS=" << ngroups
                    << " RCCL_IB_QP_DEPTH_MULTIPLIER=" << QpShareEnvDepthMultiplier()
                    << " (would have been comm #" << (model.Load(p.group) + 1)
                    << " in group " << p.group << ")";
                broke = true;
                break;
            }
            cs.Add(rank, l, s, r, p);
        } else {
            // Rotating, non-tail victim: exercises releasing a comm that still
            // has live neighbours in its group.
            const size_t victim = static_cast<size_t>(step / 2) % cs.size();
            model.Release(cs.place[victim]);
            CloseCastConnection(cs.listen[victim], cs.send[victim], cs.recv[victim]);
            cs.Erase(victim);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        observedSpread = std::max(observedSpread, model.Spread());
        ExpectQpShareLayout(cs.mine, cs.place, model, Stage("churn step", step).c_str());
    }

    ReportQpShareCoverage("algo_churn_layout", static_cast<int>(cs.size()),
                          model.PeakLoad(), observedSpread);

    while (cs.size() > 0) {
        model.Release(cs.place[0]);
        CloseCastConnection(cs.listen[0], cs.send[0], cs.recv[0]);
        cs.Erase(0);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// =============================================================================
// Test: QpShareDataAllConnsTransfer  (category: data path)
//
// The core data-path gate: at the configured (ngroups, depth), open nconns
// connections and require every one of them to connect and to move data
// correctly across a size sweep covering zero-length, single-byte and
// powers-of-two +/-1 up to 1 MB.
// =============================================================================
TEST_F(NetIbMPITest, QpShareDataAllConnsTransfer) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    QPSHARE_ENV_CHECK_OR_SKIP();
    const int rank    = MPIEnvironment::world_rank;
    const int ngroups = QpShareEnvNGroups();
    const int nconns  = std::max(1, QpShareEnvInt("QPSHARE_TEST_NCONNS",
                                                  Clamp(ngroups + 2, 4, 32)));

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    QpShareRefModel model(ngroups);
    QpShareConns    cs;

    for (int c = 0; c < nconns; c++) {
        QpShareRefModel::Placement p = model.AssignOnCreate();
        void* l = nullptr; void* s = nullptr; void* r = nullptr;
        if (!TryCastConnection(/*dev=*/0, &l, &s, &r)) {
            model.Release(p);
            ADD_FAILURE()
                << "connection " << c << " of " << nconns << " failed to establish at "
                << "RCCL_IB_COMM_NGROUPS=" << ngroups << " RCCL_IB_QP_DEPTH_MULTIPLIER="
                << QpShareEnvDepthMultiplier() << " -- it would have been comm #"
                << (model.Load(p.group) + 1) << " in group " << p.group
                << ", and the shared QP is sized for " << QpShareEnvDepthMultiplier() << "."
                << (c == 0
                        ? " This is the FIRST connection, so nothing is shared yet:"
                          " the depth multiplier itself exceeds what the device can"
                          " allocate (a known depth-sizing limit, not pool exhaustion)."
                        : " Pool exhaustion is now a hard connect() failure by design"
                          " (net_ib_cast no longer falls back to unshared silently), so"
                          " this is expected only if the shared-QP pool is genuinely"
                          " full at this scale -- otherwise it indicates a leaked slot.");
            break;
        }
        cs.Add(rank, l, s, r, p);
    }

    const int established = static_cast<int>(cs.size());
    ExpectQpShareLayout(cs.mine, cs.place, model, "all conns established");
    ReportQpShareCoverage("data_all_conns_transfer", established,
                          established > 0 ? MaxObservedSharingDepth(cs.mine) : 0,
                          model.Spread());

    if (established > 0) {
        const std::vector<size_t> kSizes = {0, 1, 127, 128, 4095, 4096, 65535, 65536, 1 << 20};
        const size_t kMaxSz = kSizes.back();

        std::vector<std::vector<char>> sendBufs(established, std::vector<char>(kMaxSz));
        std::vector<std::vector<char>> recvBufs(established, std::vector<char>(kMaxSz));
        std::vector<void*> mhandles(established, nullptr);
        bool regOk = true;
        for (int c = 0; c < established && regOk; c++) {
            char* regBuf = (rank == 0) ? recvBufs[c].data() : sendBufs[c].data();
            if (RegisterMemory(cs.mine[c], regBuf, kMaxSz, NCCL_PTR_HOST, &mhandles[c]) != ncclSuccess) {
                ADD_FAILURE() << "RegisterMemory failed for conn " << c;
                regOk = false;
            }
        }
        ASSERT_TRUE(RankAgree(regOk)) << "RegisterMemory failed on one rank only";

        constexpr int kTagBase = 5300;
        for (size_t i = 0; i < kSizes.size(); i++) {
            for (int c = 0; c < established; c++) {
                if (rank == 0) memset(recvBufs[c].data(), 0, kSizes[i]);
                DoSendRecv(cs.send[c], cs.recv[c],
                           sendBufs[c].data(), recvBufs[c].data(),
                           kSizes[i], kTagBase + static_cast<int>(i) * established + c,
                           mhandles[c], mhandles[c],
                           /*patternSeed=*/static_cast<int>(i) * 31 + c);
            }
        }

        for (int c = 0; c < established; c++)
            EXPECT_EQ(DeregisterMemory(cs.mine[c], mhandles[c]), ncclSuccess);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    while (cs.size() > 0) {
        CloseCastConnection(cs.listen[0], cs.send[0], cs.recv[0]);
        cs.Erase(0);
    }
}

// =============================================================================
// Test: QpShareDataUnsharedGroups  (category: data path)
//
// The nconns <= ngroups regime, where occupancy-modulo gives every connection
// a group of its own and no QP is shared at all.
//
// nconns is derived as min(ngroups, 8) so the test is in the unshared regime
// whatever ngroups is set to -- there is nothing to skip.
// =============================================================================
TEST_F(NetIbMPITest, QpShareDataUnsharedGroups) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    QPSHARE_ENV_CHECK_OR_SKIP();
    const int rank    = MPIEnvironment::world_rank;
    const int ngroups = QpShareEnvNGroups();
    const int nconns  = Clamp(ngroups, 1, 8);

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    QpShareRefModel model(ngroups);
    QpShareConns    cs;

    for (int c = 0; c < nconns; c++) {
        QpShareRefModel::Placement p = model.AssignOnCreate();
        void* l = nullptr; void* s = nullptr; void* r = nullptr;
        if (!TryCastConnection(/*dev=*/0, &l, &s, &r)) {
            model.Release(p);
            ADD_FAILURE() << "connection " << c << " of " << nconns
                          << " failed to establish at RCCL_IB_COMM_NGROUPS=" << ngroups
                          << " -- nothing is even shared at this connection count";
            break;
        }
        cs.Add(rank, l, s, r, p);
    }

    const int established = static_cast<int>(cs.size());
    ExpectQpShareLayout(cs.mine, cs.place, model, "unshared groups established");
    for (int c = 0; c < established; c++) {
        EXPECT_EQ(model.Load(cs.place[c].group), 1)
            << "test bug: conn " << c << " was expected to be alone in its group";
        EXPECT_TRUE(cs.place[c].primary);
    }
    ReportQpShareCoverage("data_unshared_groups", established,
                          established > 0 ? MaxObservedSharingDepth(cs.mine) : 0);

    if (established > 0) {
        constexpr size_t kMsgSz = 64 * 1024;
        std::vector<std::vector<char>> sendBufs(established, std::vector<char>(kMsgSz));
        std::vector<std::vector<char>> recvBufs(established, std::vector<char>(kMsgSz));
        std::vector<void*> mhandles(established, nullptr);
        bool regOk = true;
        for (int c = 0; c < established && regOk; c++) {
            char* regBuf = (rank == 0) ? recvBufs[c].data() : sendBufs[c].data();
            if (RegisterMemory(cs.mine[c], regBuf, kMsgSz, NCCL_PTR_HOST, &mhandles[c]) != ncclSuccess) {
                ADD_FAILURE() << "RegisterMemory failed for conn " << c;
                regOk = false;
            }
        }
        ASSERT_TRUE(RankAgree(regOk)) << "RegisterMemory failed on one rank only";
        constexpr int kTagBase = 5100;
        for (int c = 0; c < established; c++) {
            DoSendRecv(cs.send[c], cs.recv[c], sendBufs[c].data(), recvBufs[c].data(),
                       kMsgSz, kTagBase + c, mhandles[c], mhandles[c],
                       /*patternSeed=*/c + 11);
        }
        for (int c = 0; c < established; c++)
            EXPECT_EQ(DeregisterMemory(cs.mine[c], mhandles[c]), ncclSuccess);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    while (cs.size() > 0) {
        CloseCastConnection(cs.listen[0], cs.send[0], cs.recv[0]);
        cs.Erase(0);
    }
}

// =============================================================================
// Test: QpShareDataFlushRouting  (category: data path)
//
// GDR flush completion routing across a shared CQ. A flush QP is per-comm and
// is not itself shared, but its completion still lands on the group's shared CQ.
// =============================================================================
TEST_F(NetIbMPITest, QpShareDataFlushRouting) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    QPSHARE_ENV_CHECK_OR_SKIP();
    const int rank    = MPIEnvironment::world_rank;
    const int ngroups = QpShareEnvNGroups();
    const int nconns  = std::min(ngroups + 1, kFlushMaxConns);

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    ncclNetProperties_t props;
    ASSERT_EQ(GetDeviceProperties(0, &props), ncclSuccess);
    if (!(props.ptrSupport & NCCL_PTR_CUDA)) {
        // Hardware capability, not a tunable: without GDR there is no flush to
        // route. This is the one thing in the suite a parameter cannot fix.
        GTEST_SKIP() << "GDR not supported on this device, skipping flush routing test";
    }

    const size_t kSz = (nconns <= 8) ? (1024 * 1024) : (64 * 1024);

    QpShareRefModel model(ngroups);
    QpShareConns    cs;
    for (int c = 0; c < nconns; c++) {
        QpShareRefModel::Placement p = model.AssignOnCreate();
        void* l = nullptr; void* s = nullptr; void* r = nullptr;
        if (!TryCastConnection(/*dev=*/0, &l, &s, &r)) {
            model.Release(p);
            ADD_FAILURE() << "connection " << c << " of " << nconns
                          << " failed to establish at RCCL_IB_COMM_NGROUPS=" << ngroups
                          << " RCCL_IB_QP_DEPTH_MULTIPLIER=" << QpShareEnvDepthMultiplier();
            break;
        }
        cs.Add(rank, l, s, r, p);
    }
    const int established = static_cast<int>(cs.size());
    ExpectQpShareLayout(cs.mine, cs.place, model, "flush conns established");

    // Locate a PRIMARY/SECONDARY pair and an unshared baseline from the model.
    int primaryIdx = -1, secondaryIdx = -1, baselineIdx = -1;
    for (int c = 0; c < established && secondaryIdx < 0; c++) {
        if (cs.place[c].primary) continue;
        secondaryIdx = c;
        for (int q = 0; q < c; q++)
            if (cs.place[q].group == cs.place[c].group && cs.place[q].primary) primaryIdx = q;
    }
    if (primaryIdx >= 0)
        for (int c = 0; c < established; c++)
            if (cs.place[c].group != cs.place[primaryIdx].group) { baselineIdx = c; break; }

    const bool havePair = (primaryIdx >= 0 && secondaryIdx >= 0);
    ReportQpShareCoverage("data_flush_routing", established,
                          established > 0 ? MaxObservedSharingDepth(cs.mine) : 0);
    if (!havePair && rank == 0) {
        printf("[QPSHARE-COV] data_flush_routing: no PRIMARY/SECONDARY pair formed at "
               "ngroups=%d with %d conns (cap %d) -- shared-CQ flush routing was NOT "
               "exercised; only the unshared flush path was\n",
               ngroups, established, kFlushMaxConns);
        fflush(stdout);
    }

    if (established > 0) {
        std::vector<void*> devBufs(established, nullptr);
        std::vector<DeviceBufferAutoGuard> bufGuards;
        bufGuards.reserve(established);
        std::vector<void*> mhandles(established, nullptr);
        bool setupOk = true;
        for (int c = 0; c < established && setupOk; c++) {
            if (hipMalloc(&devBufs[c], kSz) != hipSuccess) {
                ADD_FAILURE() << "hipMalloc failed for conn " << c;
                setupOk = false;
                break;
            }
            bufGuards.push_back(makeDeviceBufferAutoGuard(devBufs[c]));
            if (RegisterMemory(cs.mine[c], devBufs[c], kSz, NCCL_PTR_CUDA, &mhandles[c]) != ncclSuccess) {
                ADD_FAILURE() << "RegisterMemory failed for conn " << c;
                setupOk = false;
                break;
            }
            if (rank == 1) {
                if (initializeBufferWithPattern<uint8_t>(devBufs[c], kSz,
                                                          makeBytePattern(c + 700)) != hipSuccess) {
                    ADD_FAILURE() << "initializeBufferWithPattern failed for conn " << c;
                    setupOk = false;
                    break;
                }
            }
        }
        ASSERT_TRUE(RankAgree(setupOk)) << "buffer/registration setup failed on one rank only";

        // Transfer every connection concurrently (post-all, wait-all).
        std::vector<void*> reqs(established, nullptr);
        for (int c = 0; c < established; c++) {
            if (rank == 0) PostSingleRecv(cs.recv[c], devBufs[c], kSz, 730 + c, mhandles[c], &reqs[c]);
            else           PostSendWithRetry(cs.send[c], devBufs[c], kSz, 730 + c, mhandles[c], &reqs[c]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        for (int c = 0; c < established; c++) {
            int sz = 0;
            EXPECT_EQ(WaitForCompletion(reqs[c], &sz, kLargeTransferTimeoutMs), ncclSuccess)
                << "conn " << c << " transfer completion";
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0 && havePair) {
            void* fReqP = nullptr; void* fReqS = nullptr;
            int fszP = static_cast<int>(kSz), fszS = static_cast<int>(kSz);
            EXPECT_EQ(FlushRecv(cs.recv[primaryIdx], 1, &devBufs[primaryIdx], &fszP,
                                &mhandles[primaryIdx], &fReqP), ncclSuccess);
            EXPECT_EQ(FlushRecv(cs.recv[secondaryIdx], 1, &devBufs[secondaryIdx], &fszS,
                                &mhandles[secondaryIdx], &fReqS), ncclSuccess);
            EXPECT_NE(fReqP, nullptr);
            EXPECT_NE(fReqS, nullptr);

            // Wait SECONDARY first: completions must be routed by commId, not by
            // the order the requests were posted or polled.
            int wS = 0, wP = 0;
            EXPECT_EQ(WaitForCompletion(fReqS, &wS, kLargeTransferTimeoutMs), ncclSuccess)
                << "SECONDARY (conn " << secondaryIdx << ") flush completion misrouted or lost";
            EXPECT_EQ(WaitForCompletion(fReqP, &wP, kLargeTransferTimeoutMs), ncclSuccess)
                << "PRIMARY (conn " << primaryIdx << ") flush completion misrouted or lost";

            EXPECT_TRUE(verifyBufferData<uint8_t>(devBufs[primaryIdx], kSz,
                                                  makeBytePattern(primaryIdx + 700)))
                << "PRIMARY conn " << primaryIdx << " data wrong after flush -- cross-comm misroute";
            EXPECT_TRUE(verifyBufferData<uint8_t>(devBufs[secondaryIdx], kSz,
                                                  makeBytePattern(secondaryIdx + 700)))
                << "SECONDARY conn " << secondaryIdx << " data wrong after flush -- cross-comm misroute";
        }

        // Unshared baseline: flush on a solo group must behave exactly like the
        // non-sharing path. Falls back to conn 0 when every comm shares a group.
        const int soloIdx = (baselineIdx >= 0) ? baselineIdx : (havePair ? -1 : 0);
        if (rank == 0 && soloIdx >= 0) {
            void* fReq = nullptr;
            int fsz = static_cast<int>(kSz);
            EXPECT_EQ(FlushRecv(cs.recv[soloIdx], 1, &devBufs[soloIdx], &fsz,
                                &mhandles[soloIdx], &fReq), ncclSuccess);
            int w = 0;
            EXPECT_EQ(WaitForCompletion(fReq, &w, kLargeTransferTimeoutMs), ncclSuccess)
                << "unshared baseline conn " << soloIdx << " flush completion";
            EXPECT_TRUE(verifyBufferData<uint8_t>(devBufs[soloIdx], kSz,
                                                  makeBytePattern(soloIdx + 700)));
        }

        MPI_Barrier(MPI_COMM_WORLD);
        for (int c = 0; c < established; c++)
            EXPECT_EQ(DeregisterMemory(cs.mine[c], mhandles[c]), ncclSuccess);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    while (cs.size() > 0) {
        CloseCastConnection(cs.listen[0], cs.send[0], cs.recv[0]);
        cs.Erase(0);
    }
}

// =============================================================================
// Test: QpShareStressManyConns  (category: stress)
//
// Flagship regression guard: many connections to one peer under sustained
// batched concurrent traffic on all of them. At ngroups=2 the default 100
// connections put ~50 comms on each shared QP.
//
// Scale with QPSHARE_TEST_NCONNS.
// =============================================================================
TEST_F(NetIbMPITest, QpShareStressManyConns) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    QPSHARE_ENV_CHECK_OR_SKIP();
    const int rank    = MPIEnvironment::world_rank;
    const int ngroups = QpShareEnvNGroups();
    const int nconns  = std::max(1, QpShareEnvInt("QPSHARE_TEST_NCONNS", 100));

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    QpShareRefModel model(ngroups);
    QpShareConns    cs;
    for (int c = 0; c < nconns; c++) {
        QpShareRefModel::Placement p = model.AssignOnCreate();
        void* l = nullptr; void* s = nullptr; void* r = nullptr;
        if (!TryCastConnection(/*dev=*/0, &l, &s, &r)) {
            model.Release(p);
            ADD_FAILURE() << "connection " << c << " of " << nconns
                          << " failed to establish at RCCL_IB_COMM_NGROUPS=" << ngroups
                          << " RCCL_IB_QP_DEPTH_MULTIPLIER=" << QpShareEnvDepthMultiplier()
                          << " (comm #" << (model.Load(p.group) + 1) << " in group "
                          << p.group << ")";
            break;
        }
        cs.Add(rank, l, s, r, p);
    }
    const int established = static_cast<int>(cs.size());
    ExpectQpShareLayout(cs.mine, cs.place, model, "stress conns established");
    ReportQpShareCoverage("stress_many_conns", established,
                          established > 0 ? MaxObservedSharingDepth(cs.mine) : 0,
                          model.Spread());

    if (established > 0) {
        constexpr int    kNMsgs = 20;
        constexpr size_t kMsgSz = 4096;
        constexpr size_t kBufSz = kNMsgs * kMsgSz;

        std::vector<std::vector<char>> sendBufs(established, std::vector<char>(kBufSz));
        std::vector<std::vector<char>> recvBufs(established, std::vector<char>(kBufSz));
        for (int c = 0; c < established; c++) {
            for (size_t i = 0; i < kBufSz; i++)
                sendBufs[c][i] = static_cast<char>(((i + c) * 5 + 11) & 0xFF);
            memset(recvBufs[c].data(), 0, kBufSz);
        }

        std::vector<void*> mhandles(established, nullptr);
        bool regOk = true;
        for (int c = 0; c < established && regOk; c++) {
            char* regBuf = (rank == 0) ? recvBufs[c].data() : sendBufs[c].data();
            if (RegisterMemory(cs.mine[c], regBuf, kBufSz, NCCL_PTR_HOST, &mhandles[c]) != ncclSuccess) {
                ADD_FAILURE() << "RegisterMemory failed for conn " << c;
                regOk = false;
            }
        }
        ASSERT_TRUE(RankAgree(regOk)) << "RegisterMemory failed on one rank only";

        constexpr int kTagBase = 6000;
        for (int c = 0; c < established; c++) {
            CastDoBatchSendRecv(rank, cs.send[c], cs.recv[c],
                                sendBufs[c].data(), recvBufs[c].data(),
                                kMsgSz, kNMsgs, kTagBase + c * kNMsgs, mhandles[c]);
        }

        if (rank == 0) {
            for (int c = 0; c < established; c++)
                EXPECT_EQ(memcmp(recvBufs[c].data(), sendBufs[c].data(), kBufSz), 0)
                    << "data corruption on conn " << c;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        for (int c = 0; c < established; c++)
            EXPECT_EQ(DeregisterMemory(cs.mine[c], mhandles[c]), ncclSuccess);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    while (cs.size() > 0) {
        CloseCastConnection(cs.listen[0], cs.send[0], cs.recv[0]);
        cs.Erase(0);
    }
}

// =============================================================================
// Test: QpShareStressSharedRqSaturation  (category: stress)
//
// Deliberately tries to starve the shared receive queue: one message posted
// from every connection simultaneously (post-all before any wait), repeated
// over several rounds.
//
// Scale with QPSHARE_TEST_NCONNS (connections) and QPSHARE_TEST_ITERS (rounds).
// =============================================================================
TEST_F(NetIbMPITest, QpShareStressSharedRqSaturation) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    QPSHARE_ENV_CHECK_OR_SKIP();
    const int rank    = MPIEnvironment::world_rank;
    const int ngroups = QpShareEnvNGroups();
    const int nconns  = std::max(1, QpShareEnvInt("QPSHARE_TEST_NCONNS", 50));
    const int rounds  = std::max(1, QpShareEnvInt("QPSHARE_TEST_ITERS", 10));

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    QpShareRefModel model(ngroups);
    QpShareConns    cs;
    for (int c = 0; c < nconns; c++) {
        QpShareRefModel::Placement p = model.AssignOnCreate();
        void* l = nullptr; void* s = nullptr; void* r = nullptr;
        if (!TryCastConnection(/*dev=*/0, &l, &s, &r)) {
            model.Release(p);
            ADD_FAILURE() << "connection " << c << " of " << nconns
                          << " failed to establish at RCCL_IB_COMM_NGROUPS=" << ngroups
                          << " RCCL_IB_QP_DEPTH_MULTIPLIER=" << QpShareEnvDepthMultiplier();
            break;
        }
        cs.Add(rank, l, s, r, p);
    }
    const int established = static_cast<int>(cs.size());
    ReportQpShareCoverage("stress_rq_saturation", established,
                          established > 0 ? MaxObservedSharingDepth(cs.mine) : 0,
                          model.Spread());

    if (established > 0) {
        constexpr size_t kMsgSz = 512;
        std::vector<std::vector<char>> sendBufs(established, std::vector<char>(kMsgSz));
        std::vector<std::vector<char>> recvBufs(established, std::vector<char>(kMsgSz));
        std::vector<void*> mhandles(established, nullptr);
        bool regOk = true;
        for (int c = 0; c < established && regOk; c++) {
            char* regBuf = (rank == 0) ? recvBufs[c].data() : sendBufs[c].data();
            if (RegisterMemory(cs.mine[c], regBuf, kMsgSz, NCCL_PTR_HOST, &mhandles[c]) != ncclSuccess) {
                ADD_FAILURE() << "RegisterMemory failed for conn " << c;
                regOk = false;
            }
        }
        ASSERT_TRUE(RankAgree(regOk)) << "RegisterMemory failed on one rank only";

        constexpr int kTagBase = 7000;
        for (int round = 0; round < rounds; round++) {
            std::vector<void*> reqs(established, nullptr);
            const int tagRoundBase = kTagBase + round * established;
            bool postOk = true;
            if (rank == 0) {
                for (int c = 0; c < established; c++) {
                    void*  bufs[1]    = {recvBufs[c].data()};
                    size_t sizes[1]   = {kMsgSz};
                    int    tags[1]    = {tagRoundBase + c};
                    void*  handles[1] = {mhandles[c]};
                    if (PostRecv(cs.recv[c], 1, bufs, sizes, tags, handles, &reqs[c]) != ncclSuccess
                        || reqs[c] == nullptr) {
                        ADD_FAILURE() << "PostRecv failed for round " << round << " conn " << c;
                        postOk = false;
                        break;
                    }
                }
            } else {
                for (int c = 0; c < established; c++) {
                    fillHostBufferWithPattern<uint8_t>(sendBufs[c].data(), kMsgSz,
                                                       makeBytePattern(round * established + c));
                    PostSendWithRetry(cs.send[c], sendBufs[c].data(), kMsgSz,
                                      tagRoundBase + c, mhandles[c], &reqs[c]);
                }
            }
            ASSERT_TRUE(RankAgree(postOk)) << "PostRecv failed on one rank only (round " << round << ")";
            for (int c = 0; c < established; c++) {
                int sz = 0;
                EXPECT_EQ(WaitForCompletion(reqs[c], &sz, kStressTimeoutMs), ncclSuccess)
                    << "round " << round << " conn " << c;
            }

            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0) {
                for (int c = 0; c < established; c++) {
                    size_t errIdx; uint8_t errExp, errGot;
                    EXPECT_TRUE(verifyHostBufferData<uint8_t>(
                        recvBufs[c].data(), kMsgSz, makeBytePattern(round * established + c),
                        0, 0.0, &errIdx, &errExp, &errGot))
                        << "round " << round << " conn " << c << " data mismatch at byte " << errIdx;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        for (int c = 0; c < established; c++)
            EXPECT_EQ(DeregisterMemory(cs.mine[c], mhandles[c]), ncclSuccess);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    while (cs.size() > 0) {
        CloseCastConnection(cs.listen[0], cs.send[0], cs.recv[0]);
        cs.Erase(0);
    }
}

// =============================================================================
// Test: QpShareStressConnectionChurn  (category: stress)
//
// Sequential connect -> transfer -> close cycles, by default 2048 of them.
// Only one connection is live at a time, so nothing accumulates that the
// transport is entitled to keep, so every cycle is fully torn down before the
// next begins. What the checkpoints measure is whether that teardown returns
// the device resources it took.
//
// Iterations are set by QPSHARE_TEST_ITERS;
// =============================================================================
TEST_F(NetIbMPITest, QpShareStressConnectionChurn) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    QPSHARE_ENV_CHECK_OR_SKIP();
    const int rank  = MPIEnvironment::world_rank;
    const int iters = std::max(1, QpShareEnvInt("QPSHARE_TEST_ITERS", 2048));

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    const size_t sz = 1024;
    std::vector<char> buf(sz);

    // Warmup connect+transfer+close so driver-internal CQs settle before baselining.
    {
        void* l = nullptr; void* s = nullptr; void* r = nullptr;
        if (!TryCastConnection(/*dev=*/0, &l, &s, &r)) {
            // The shared-QP pool reclaims slots on close (free-stack), so this
            // should not happen from cross-test accumulation anymore. A failure
            // here now points at a slot leaked by an earlier test's teardown.
            ADD_FAILURE() << "warmup connection failed before churn even started "
                             "(shared-QP pool slot leak from an earlier test in this "
                             "process? slots are reclaimed on close, so this should "
                             "not occur under normal operation)";
            return;
        }
        void* mh = nullptr;
        bool regOk = (RegisterMemory((rank == 0) ? r : s, buf.data(), sz, NCCL_PTR_HOST, &mh) == ncclSuccess);
        if (!regOk) ADD_FAILURE() << "RegisterMemory failed during warmup";
        ASSERT_TRUE(RankAgree(regOk)) << "RegisterMemory failed on one rank only during warmup";
        DoSendRecv(s, r, buf.data(), buf.data(), sz, /*tag=*/599, mh, mh, /*seed=*/0);
        MPI_Barrier(MPI_COMM_WORLD);
        TeardownConnection(r, l, s, mh);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    RdmaResourceCounts before = CaptureRdmaResources();
    MPI_Barrier(MPI_COMM_WORLD);

    // Checkpoint every quarter, so a failure localises without depending on the
    // iteration count being the default.
    const int stride = std::max(1, iters / 4);

    for (int iter = 0; iter < iters; iter++) {
        void* l = nullptr; void* s = nullptr; void* r = nullptr;
        if (!TryCastConnection(/*dev=*/0, &l, &s, &r)) {
            ADD_FAILURE()
                << "connect failed at churn iteration " << iter << " of " << iters
                << ". Only one connection is live at a time, so every earlier"
                   " cycle was fully closed: a resource acquired per cycle was"
                   " not returned. Check the RDMA-resource checkpoints above and"
                   " run with NCCL_DEBUG=WARN -- pass -x"
                   " NCCL_DEBUG_FILE=<dir>/nccl_%h_%p.log and read the per-rank"
                   " files, since a merged mpirun log usually carries rank 0"
                   " only and the absence of a WARN there is not evidence that"
                   " none was emitted";
            break;
        }
        void* comm = (rank == 0) ? r : s;
        void* mh   = nullptr;
        bool regOk = (RegisterMemory(comm, buf.data(), sz, NCCL_PTR_HOST, &mh) == ncclSuccess);
        if (!regOk) ADD_FAILURE() << "RegisterMemory failed at churn iteration " << iter;
        ASSERT_TRUE(RankAgree(regOk)) << "RegisterMemory failed on one rank only at iteration " << iter;

        DoSendRecv(s, r, buf.data(), buf.data(), sz, /*tag=*/iter % 1000, mh, mh, iter);

        const bool checkpoint = ((iter + 1) % stride == 0) || (iter == iters - 1);
        if (checkpoint) {
            struct ncclIbQpSharingState st = GetActualQpSharingState(comm);
            EXPECT_GT(st.refcount, 0)
                << "refcount dropped to 0 at iter " << iter
                << " -- the comm is live but no longer tracked in the shared-QP pool";
            EXPECT_NE(st.commId, 0)
                << "commId cleared at iter " << iter
                << " -- the comm silently fell off the sharing path";
        }

        MPI_Barrier(MPI_COMM_WORLD);
        TeardownConnection(r, l, s, mh);

        if (checkpoint) {
            MPI_Barrier(MPI_COMM_WORLD);
            RdmaResourceCounts now = CaptureRdmaResources();
            AssertNoRdmaLeaks(before, now, ("churn@" + std::to_string(iter + 1)).c_str());
        }
    }

    // Exactly one comm is live at any instant, so this test never stacks two
    // comms on one QP -- achieved sharing depth is 1 by construction, not by
    // accident, and the "NO QP WAS SHARED" banner is the correct label. Report
    // it rather than leaving the driver's max/grp column blank: a blank column
    // invites the reading that churn says something about sharing depth. It
    // does not. It exercises the create/destroy lifecycle of a shared group
    // whose membership never exceeds one, which is the occupancy that reaches
    // the group-release path on every single cycle (the same path
    // QpShareStressBatchCreateDestroy shows leaking a PD).
    ReportQpShareCoverage("stress_connection_churn", iters, /*maxCommsPerGroup=*/1);
}

// =============================================================================
// Test: QpShareStressBatchCreateDestroy  (category: stress)
//
// Resource lifetime under a bursty cadence: batches of connections created
// together, used, and destroyed together. Defaults to 110 batches of 10.
// RDMA-resource and refcount checkpoints every 25 batches.
//
// What this test is positioned to catch is anything that survives a completed
// batch, which is why the checkpoints compare RDMA object counts against a
// pre-batch baseline.
//
// Batch count from QPSHARE_TEST_ITERS, batch size from QPSHARE_TEST_NCONNS.
// Unlike the sequential churn test this one holds several comms live at once,
// so the batch size also has to fit the configured depth multiplier.
// =============================================================================
TEST_F(NetIbMPITest, QpShareStressBatchCreateDestroy) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    QPSHARE_ENV_CHECK_OR_SKIP();
    const int rank     = MPIEnvironment::world_rank;
    const int ngroups  = QpShareEnvNGroups();
    const int batches  = std::max(1, QpShareEnvInt("QPSHARE_TEST_ITERS", 110));
    const int perBatch = std::max(1, QpShareEnvInt("QPSHARE_TEST_NCONNS", 10));

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    const size_t sz = 512;
    std::vector<char> buf(sz);

    {
        void* l = nullptr; void* s = nullptr; void* r = nullptr;
        if (!TryCastConnection(/*dev=*/0, &l, &s, &r)) {
            // The shared-QP pool reclaims slots on close (free-stack), so this
            // should not happen from cross-test accumulation anymore. A failure
            // here now points at a slot leaked by an earlier test's teardown.
            ADD_FAILURE() << "warmup connection failed before batching even started "
                             "(shared-QP pool slot leak from an earlier test in this "
                             "process? slots are reclaimed on close, so this should "
                             "not occur under normal operation)";
            return;
        }
        void* mh = nullptr;
        bool regOk = (RegisterMemory((rank == 0) ? r : s, buf.data(), sz, NCCL_PTR_HOST, &mh) == ncclSuccess);
        if (!regOk) ADD_FAILURE() << "RegisterMemory failed during warmup";
        ASSERT_TRUE(RankAgree(regOk)) << "RegisterMemory failed on one rank only during warmup";
        DoSendRecv(s, r, buf.data(), buf.data(), sz, /*tag=*/699, mh, mh, /*seed=*/0);
        MPI_Barrier(MPI_COMM_WORLD);
        TeardownConnection(r, l, s, mh);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    RdmaResourceCounts before = CaptureRdmaResources();
    MPI_Barrier(MPI_COMM_WORLD);

    bool broke = false;
    int  peakDepth = 0;   // deepest any group got in any batch
    int  peakSpread = 0;
    for (int batch = 0; batch < batches && !broke; batch++) {
        QpShareRefModel model(ngroups);
        QpShareConns    cs;
        std::vector<void*> mhandles;

        for (int c = 0; c < perBatch; c++) {
            QpShareRefModel::Placement p = model.AssignOnCreate();
            void* l = nullptr; void* s = nullptr; void* r = nullptr;
            if (!TryCastConnection(/*dev=*/0, &l, &s, &r)) {
                model.Release(p);
                ADD_FAILURE()
                    << "connect failed at batch " << batch << " of " << batches
                    << ", connection " << c << " of " << perBatch
                    << " (cumulative connection #" << (batch * perBatch + c)
                    << "). ngroups=" << ngroups << " depth="
                    << QpShareEnvDepthMultiplier()
                    << ". Every previous batch was fully torn down, so a failure "
                       "here means a per-batch resource was not returned -- check "
                       "the RDMA-count checkpoints above for which one.";
                broke = true;
                break;
            }
            cs.Add(rank, l, s, r, p);
            void* mh = nullptr;
            bool regOk = (RegisterMemory(cs.mine.back(), buf.data(), sz, NCCL_PTR_HOST, &mh) == ncclSuccess);
            if (!regOk) {
                ADD_FAILURE() << "RegisterMemory failed at batch " << batch
                              << ", connection " << c;
            }
            if (!RankAgree(regOk)) {
                broke = true;
                break;
            }
            mhandles.push_back(mh);
        }

        const int live = static_cast<int>(cs.size());
        peakDepth  = std::max(peakDepth, model.PeakLoad());
        peakSpread = std::max(peakSpread, model.Spread());
        for (int c = 0; c < live; c++) {
            DoSendRecv(cs.send[c], cs.recv[c], buf.data(), buf.data(), sz,
                       /*tag=*/c, mhandles[c], mhandles[c], batch * perBatch + c);
        }

        const bool checkpoint = ((batch + 1) % 25 == 0) || (batch == batches - 1);
        if (checkpoint && live > 0) {
            // Every batch starts from an empty live set, so the layout must be
            // identical to batch 0's. Drift here means teardown left refcounts
            // behind that IbCastCountPeerTotalRefcount still sees.
            ExpectQpShareLayout(cs.mine, cs.place, model,
                                Stage("batch checkpoint", batch).c_str());
            struct ncclIbQpSharingState st = GetActualQpSharingState(cs.mine[0]);
            EXPECT_GT(st.refcount, 0)
                << "refcount dropped to 0 at batch " << batch
                << " -- comms are live but no longer tracked in the shared-QP pool";
            EXPECT_NE(st.commId, 0) << "commId cleared at batch " << batch;
        }

        for (int c = 0; c < live; c++) {
            EXPECT_EQ(DeregisterMemory(cs.mine[c], mhandles[c]), ncclSuccess);
            CloseCastConnection(cs.listen[c], cs.send[c], cs.recv[c]);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (checkpoint && !broke) {
            RdmaResourceCounts now = CaptureRdmaResources();
            AssertNoRdmaLeaks(before, now, ("batch@" + std::to_string(batch + 1)).c_str());
        }
    }

    // Depth reached inside a batch, which is what sets how often the
    // group-release path runs: perBatch/ngroups comms per group means only
    // 1-in-that-many teardowns reaches it. At the default 10 conns / ngroups=2
    // that is the mild leak rate; at ngroups=10 every group holds one comm and
    // the rate is the churn test's. Reported so the two are comparable.
    ReportQpShareCoverage("stress_batch_create_destroy", perBatch, peakDepth, peakSpread);
}

#endif /* MPI_TESTS_ENABLED */
