/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/rma/rma_proxy_progress.cc.
//
// AICOMRCCL-1854: add coverage for the NCCL 2.30.7 fix of NVIDIA/nccl issue
// #2119 ("one-sided host API requests dropped at a high message rate"). The
// functions under test are all file-static, so this TU reaches them by
// #include-ing the production .cc directly (via RMA_PROXY_PROGRESS_CC_PATH),
// the standard rccl-UnitTestsMicro pattern (see MICROTEST_README.md).
//
// The one hardware dependency -- the network -- is already a function-pointer
// vtable (ncclRma_t). The test double, FakeNet, models exactly what #2119 is
// about: a bounded request pool. With a small poolSize and several queued
// puts, "high message rate" backpressure becomes a deterministic, GPU-free
// unit test.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <vector>

#include "ScopedHook.h"
#include "fakes/rma_fakes.h"

#include "nccl.h"
#include "comm.h"
#include "rma/rma_proxy.h"

// Pull the unit under test in directly so its file-static functions are
// reachable. Must come after the fakes/headers above are in scope.
#include RMA_PROXY_PROGRESS_CC_PATH

namespace {

// ===========================================================================
// FakeNet -- scriptable stand-in for the RMA network behind the ncclRma_t
// vtable. Models a bounded request pool (issue #2119's MAX_REQUESTS):
//
//   - issue():   if outstanding >= poolSize, reproduce the pre-fix failure
//                mode (return ncclInternalError, leave *request == NULL);
//                otherwise hand out a fresh request and bump `outstanding`.
//   - testReq(): report a request done only if the test asked for it (via
//                completeNext()/completeRequest()); a completed request frees
//                one pool slot.
//
// The vtable's iput/iputSignal/test entry points are plain C function pointers,
// so the trampolines recover the FakeNet from the void* ctx the production code
// threads through (ctx->rmaCtx for puts, ctx->rmaCollComm for test).
// ===========================================================================
struct FakeReq {
    uint32_t targetRank = 0;
    bool completed = false;
};

struct FakeNet {
    int  poolSize          = 256;   // model MAX_REQUESTS
    int  outstanding       = 0;     // live requests not yet completed
    bool forceNullOnSuccess = false;  // test #5: success but NULL request

    int  issueCalls        = 0;     // total iput/iputSignal calls
    int  testCalls         = 0;     // total test() calls

    // FIFO of live requests, in issue order, for completeNext().
    std::deque<FakeReq*> live;
    std::vector<std::unique_ptr<FakeReq>> owned;

    ncclResult_t issue(uint32_t targetRank, void** request) {
        ++issueCalls;
        if (outstanding >= poolSize) {
            *request = nullptr;          // pre-fix drop symptom
            return ncclInternalError;
        }
        if (forceNullOnSuccess) {
            *request = nullptr;
            return ncclSuccess;
        }
        auto r = std::make_unique<FakeReq>();
        r->targetRank = targetRank;
        FakeReq* p = r.get();
        owned.push_back(std::move(r));
        live.push_back(p);
        ++outstanding;
        *request = p;
        return ncclSuccess;
    }

    ncclResult_t testReq(void* request, int* done) {
        ++testCalls;
        // A real plugin dereferences the handle; NULL means the progress code
        // dropped its request-NULL skip (e.g. in the group-completion loop),
        // which would fault or double-count in production. Demand a live handle.
        EXPECT_NE(request, nullptr) << "test() called with a NULL request handle";
        auto* r = static_cast<FakeReq*>(request);
        *done = (r != nullptr && r->completed) ? 1 : 0;
        if (*done) {
            --outstanding;
            for (auto it = live.begin(); it != live.end(); ++it) {
                if (*it == r) { live.erase(it); break; }
            }
        }
        return ncclSuccess;
    }

    // Mark the oldest still-live request complete (FIFO completion order).
    void completeNext() {
        ASSERT_FALSE(live.empty());
        live.front()->completed = true;
    }

    void completeRequest(void* request) {
        static_cast<FakeReq*>(request)->completed = true;
    }

    // Build a vtable wired to this FakeNet. Only iput/iputSignal/test are used
    // by the non-persistent progress path exercised here.
    ncclRma_t vtable() {
        ncclRma_t v{};
        v.iput       = &FakeNet::TrampIput;
        v.iputSignal = &FakeNet::TrampIputSignal;
        v.test       = &FakeNet::TrampTest;
        return v;
    }

    // Production's rmaCtx and rmaCollComm are distinct objects: rmaCollComm
    // from connect(), rmaCtx from createContext(collComm, ...) (rma_v14.h:33,36).
    // Tag them so a swapped handle (rmaCtx passed where rmaCollComm is expected,
    // or vice-versa) fails loudly instead of silently passing.
    struct Handle { FakeNet* net; enum Kind { Ctx, CollComm } kind; };
    Handle ctxH{this, Handle::Ctx};
    Handle collH{this, Handle::CollComm};

private:
    static FakeNet* Net(void* h, Handle::Kind want) {
        auto* handle = static_cast<Handle*>(h);
        EXPECT_EQ(handle->kind, want) << "RMA handle passed to the wrong entry point";
        return handle->net;
    }
    static ncclResult_t TrampIput(void* rmaCtx, int, uint64_t, void*, size_t,
                                  uint64_t, void*, uint32_t rank, uint32_t, void** request) {
        return Net(rmaCtx, Handle::Ctx)->issue(rank, request);
    }
    static ncclResult_t TrampIputSignal(void* rmaCtx, int, uint64_t, void*, size_t,
                                        uint64_t, void*, uint32_t rank, uint64_t,
                                        void*, uint64_t, uint32_t, bool, uint32_t, void** request) {
        return Net(rmaCtx, Handle::Ctx)->issue(rank, request);
    }
    static ncclResult_t TrampTest(void* collComm, void* request, int* done) {
        return Net(collComm, Handle::CollComm)->testReq(request, done);
    }
};

// ===========================================================================
// Fixture: hand-builds a minimal ncclRmaProxyCtx + ncclComm and a FakeNet.
//
// Only the fields the non-persistent progress path reads are populated:
// circularBuffers / cis / pis / inProgressQueues / inflightRequests /
// maxInflightRequests / queueSize / comm / rmaCtx / rmaCollComm.
// ===========================================================================
class RmaProxyProgressTest : public ::testing::Test {
protected:
    static constexpr int kQueueSize = 8;  // power of two

    int nRanks_ = 2;

    std::unique_ptr<ncclComm> comm_;
    std::unique_ptr<ncclRmaProxyCtx> ctx_;
    FakeNet net_;
    ncclRma_t rma_{};

    // Backing storage owned by the fixture.
    std::vector<uint32_t> cis_;
    std::vector<uint32_t> pis_;
    std::vector<uint32_t> inflight_;
    std::vector<ncclRmaProxyDesc*> circular_;
    std::vector<ncclIntruQueue<ncclRmaProxyDesc, &ncclRmaProxyDesc::next>> inProgress_;
    // Descriptors the test allocates (kept alive here; destruction is faked).
    std::vector<std::unique_ptr<ncclRmaProxyDesc>> descs_;
    // Sequence storage for descriptors that need a readySeq pointer.
    std::deque<uint64_t> seqStore_;
    // doneSeq storage for in-progress descriptors under completion polling.
    std::deque<uint64_t> doneSeqStore_;
    // Backing storage for group descriptors' ops arrays (deque keeps element
    // vectors stable so ops.data() stays valid).
    std::deque<std::vector<ncclRmaPutSignalOp>> groupOpsStore_;

    void SetUp() override {
        comm_ = std::make_unique<ncclComm>();
        comm_->rank = 0;
        comm_->nRanks = nRanks_;

        cis_.assign(nRanks_, 0);
        pis_.assign(nRanks_, 0);
        inflight_.assign(nRanks_, 0);
        circular_.assign(static_cast<size_t>(nRanks_) * kQueueSize, nullptr);
        inProgress_.resize(nRanks_);
        for (auto& q : inProgress_) {
            ncclIntruQueueConstruct(&q);
        }

        ctx_ = std::make_unique<ncclRmaProxyCtx>();
        ctx_->comm = comm_.get();
        ctx_->queueSize = kQueueSize;
        ctx_->circularBuffers = circular_.data();
        ctx_->cis = cis_.data();
        ctx_->pis = pis_.data();
        ctx_->inProgressQueues = inProgress_.data();
        ctx_->inflightRequests = inflight_.data();
        ctx_->maxInflightRequests = 256;
        ctx_->rmaCtx = &net_.ctxH;
        ctx_->rmaCollComm = &net_.collH;

        net_.poolSize = 256;
        rma_ = net_.vtable();
    }

    void TearDown() override { ResetRmaFakes(); }

    // Allocate a single PutSignal descriptor targeting `targetRank`, ready to
    // issue (readySeq >= opSeq), and place it at the current pending head for
    // `peer`. Bumps pis[peer] so the circular buffer reports non-empty.
    ncclRmaProxyDesc* PushPendingPutSignal(int peer, uint32_t targetRank,
                                           bool withSignal = false) {
        auto d = std::make_unique<ncclRmaProxyDesc>();
        std::memset(d.get(), 0, sizeof(ncclRmaProxyDesc));
        d->rmaDescType = ncclRmaDescTypePutSignal;
        d->rmaDescState = ncclRmaDescStateReady;
        d->putSignal.targetRank = static_cast<int>(targetRank);
        d->putSignal.signal.op = withSignal ? 1 : 0;
        d->putSignal.request = nullptr;

        uint32_t pi = pis_[peer];
        d->opSeq = pi;
        seqStore_.push_back(pi);          // readyVal == opSeq -> ready
        d->readySeq = &seqStore_.back();

        uint32_t idx = pi & (ctx_->queueSize - 1);
        circular_[static_cast<size_t>(peer) * kQueueSize + idx] = d.get();
        pis_[peer] = pi + 1;

        ncclRmaProxyDesc* raw = d.get();
        descs_.push_back(std::move(d));
        return raw;
    }

    // Allocate a single Ready-typed PutSignal descriptor with an explicit
    // opSeq and a caller-owned *shared* readySeq location, placed at the
    // current pending head for `peer`. This mirrors production, where every
    // pending desc for a peer reads one per-peer readySeq that the GPU advances
    // monotonically (rma_proxy_launch.cc: readySeq = &readySeqs[peer]; the GPU
    // stores desc->opSeq into it once the desc's data is ready). A desc is
    // "ready" only once *sharedReady >= opSeq. Bumps pis[peer] so the circular
    // buffer reports non-empty; opSeq is independent of the buffer index.
    ncclRmaProxyDesc* PushPendingPutSignalShared(int peer, uint32_t targetRank,
                                                 uint64_t opSeq,
                                                 uint64_t* sharedReady) {
        auto d = std::make_unique<ncclRmaProxyDesc>();
        std::memset(d.get(), 0, sizeof(ncclRmaProxyDesc));
        d->rmaDescType = ncclRmaDescTypePutSignal;
        d->rmaDescState = ncclRmaDescStateReady;
        d->putSignal.targetRank = static_cast<int>(targetRank);
        d->putSignal.signal.op = 0;
        d->putSignal.request = nullptr;
        d->opSeq = opSeq;
        d->readySeq = sharedReady;

        uint32_t pi = pis_[peer];
        uint32_t idx = pi & (ctx_->queueSize - 1);
        circular_[static_cast<size_t>(peer) * kQueueSize + idx] = d.get();
        pis_[peer] = pi + 1;

        ncclRmaProxyDesc* raw = d.get();
        descs_.push_back(std::move(d));
        return raw;
    }

    // Allocate a single PutSignal descriptor already issued to the network
    // (a live request from FakeNet) and enqueue it on `peer`'s in-progress
    // queue, bumping inflightRequests[targetRank] to match. Gives the desc a
    // doneSeq slot (initialised to a sentinel != opSeq) so completion can be
    // observed to publish opSeq into it.
    ncclRmaProxyDesc* PushInProgressPutSignal(int peer, uint32_t targetRank,
                                              uint64_t opSeq = 1) {
        auto d = std::make_unique<ncclRmaProxyDesc>();
        std::memset(d.get(), 0, sizeof(ncclRmaProxyDesc));
        d->rmaDescType = ncclRmaDescTypePutSignal;
        d->rmaDescState = ncclRmaDescStateInProgress;
        d->putSignal.targetRank = static_cast<int>(targetRank);
        d->putSignal.signal.op = 0;
        d->opSeq = opSeq;

        doneSeqStore_.push_back(~opSeq);  // sentinel distinct from opSeq
        d->doneSeq = &doneSeqStore_.back();

        // Issue a real request through the fake so completion has something to
        // test, and account the credit as the issue path would have.
        EXPECT_EQ(net_.issue(targetRank, &d->putSignal.request), ncclSuccess);
        inflight_[targetRank]++;

        ncclIntruQueueEnqueue(&inProgress_[peer], d.get());
        ncclRmaProxyDesc* raw = d.get();
        descs_.push_back(std::move(d));
        return raw;
    }

    // Allocate a Ready pending PutSignalGroup descriptor whose ops target the
    // ranks in `targets`, and place it at the current pending head for `peer`.
    ncclRmaProxyDesc* PushPendingPutGroup(int peer,
                                          std::vector<uint32_t> targets) {
        auto d = std::make_unique<ncclRmaProxyDesc>();
        std::memset(d.get(), 0, sizeof(ncclRmaProxyDesc));
        d->rmaDescType = ncclRmaDescTypePutSignalGroup;
        d->rmaDescState = ncclRmaDescStateReady;

        groupOpsStore_.emplace_back(targets.size());
        std::vector<ncclRmaPutSignalOp>& ops = groupOpsStore_.back();
        for (size_t i = 0; i < targets.size(); i++) {
            ops[i].targetRank = static_cast<int>(targets[i]);
            ops[i].signal.op = 0;
            ops[i].request = nullptr;
        }
        d->putSignalGroup.nOps = static_cast<int>(targets.size());
        d->putSignalGroup.ops = ops.data();
        d->putSignalGroup.nIssued = 0;
        d->putSignalGroup.nCompleted = 0;

        uint32_t pi = pis_[peer];
        d->opSeq = pi;
        seqStore_.push_back(pi);          // readyVal == opSeq -> ready
        d->readySeq = &seqStore_.back();

        uint32_t idx = pi & (ctx_->queueSize - 1);
        circular_[static_cast<size_t>(peer) * kQueueSize + idx] = d.get();
        pis_[peer] = pi + 1;

        ncclRmaProxyDesc* raw = d.get();
        descs_.push_back(std::move(d));
        return raw;
    }

    // Allocate a PutSignalGroup descriptor with all `targets.size()` ops
    // already issued to the network (live FakeNet requests) and enqueue it on
    // `peer`'s in-progress queue. nIssued == nOps, nCompleted == 0, so the
    // group is awaiting completion. Gives the desc a doneSeq slot (sentinel
    // != opSeq) so publication of opSeq on full completion is observable.
    ncclRmaProxyDesc* PushInProgressPutGroup(int peer,
                                             std::vector<uint32_t> targets,
                                             uint64_t opSeq = 1) {
        auto d = std::make_unique<ncclRmaProxyDesc>();
        std::memset(d.get(), 0, sizeof(ncclRmaProxyDesc));
        d->rmaDescType = ncclRmaDescTypePutSignalGroup;
        d->rmaDescState = ncclRmaDescStateInProgress;
        d->opSeq = opSeq;

        doneSeqStore_.push_back(~opSeq);  // sentinel distinct from opSeq
        d->doneSeq = &doneSeqStore_.back();

        groupOpsStore_.emplace_back(targets.size());
        std::vector<ncclRmaPutSignalOp>& ops = groupOpsStore_.back();
        for (size_t i = 0; i < targets.size(); i++) {
            ops[i].targetRank = static_cast<int>(targets[i]);
            ops[i].signal.op = 0;
            ops[i].request = nullptr;
            EXPECT_EQ(net_.issue(targets[i], &ops[i].request), ncclSuccess);
            inflight_[targets[i]]++;
        }
        d->putSignalGroup.nOps = static_cast<int>(targets.size());
        d->putSignalGroup.ops = ops.data();
        d->putSignalGroup.nIssued = static_cast<int>(targets.size());
        d->putSignalGroup.nCompleted = 0;

        ncclIntruQueueEnqueue(&inProgress_[peer], d.get());
        ncclRmaProxyDesc* raw = d.get();
        descs_.push_back(std::move(d));
        return raw;
    }

    ncclRmaProxyDesc* InProgressHead(int peer) {
        return ncclIntruQueueHead(&inProgress_[peer]);
    }
};

TEST_F(RmaProxyProgressTest, SinglePut_CreditAvailable_MovesToInProgressAndAdvancesCI) {
    const int peer = 1;
    const uint32_t target = 1;
    ncclRmaProxyDesc* desc = PushPendingPutSignal(peer, target);

    ASSERT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);

    EXPECT_EQ(net_.issueCalls, 1);
    EXPECT_EQ(InProgressHead(peer), desc);
    EXPECT_EQ(cis_[peer], 1u);
}

TEST_F(RmaProxyProgressTest, SinglePut_NoCredit_StaysPending) {
    const int peer = 1;
    const uint32_t target = 1;

    // The target's credit budget is fully consumed.
    ctx_->maxInflightRequests = 1;
    inflight_[target] = 1;

    PushPendingPutSignal(peer, target);

    EXPECT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);

    EXPECT_EQ(net_.issueCalls, 0);
    EXPECT_EQ(cis_[peer], 0u);
    EXPECT_EQ(InProgressHead(peer), nullptr);
}

TEST_F(RmaProxyProgressTest, CompletedHead_PublishesDoneSeqDequeuesAndDestroys) {
    const int peer = 1;
    const uint32_t target = 1;
    const uint64_t opSeq = 7;

    ncclRmaProxyDesc* desc = PushInProgressPutSignal(peer, target, opSeq);

    ncclRmaProxyDesc* destroyed = nullptr;
    ScopedHook destroy(g_rmaDestroyDesc,
        [&](struct ncclComm*, struct ncclRmaProxyDesc** d) -> ncclResult_t {
            destroyed = *d;
            *d = nullptr;
            return ncclSuccess;
        });

    net_.completeRequest(desc->putSignal.request);
    EXPECT_EQ(ncclRmaProxyPollNonPersistCompletion(&rma_, ctx_.get(), peer),
              ncclSuccess);

    EXPECT_EQ(doneSeqStore_.back(), opSeq);       // published for the GPU
    EXPECT_EQ(InProgressHead(peer), nullptr);     // dequeued
    EXPECT_EQ(destroyed, desc);                   // destroyed
    EXPECT_EQ(destroy.calls, 1);
}

TEST_F(RmaProxyProgressTest, SinglePut_IssuesOnceCompletionFreesCredit) {
    const int peer = 1;
    const uint32_t target = 1;

    // Credit fully consumed by one in-flight op; a second put waits pending.
    ctx_->maxInflightRequests = 1;
    ncclRmaProxyDesc* inflightOp = PushInProgressPutSignal(peer, target);
    PushPendingPutSignal(peer, target);
    const int baseIssue = net_.issueCalls;

    // At the credit ceiling the pending op is not issued.
    EXPECT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(net_.issueCalls, baseIssue);
    EXPECT_EQ(cis_[peer], 0u);

    // Completing the in-flight op frees a credit.
    net_.completeRequest(inflightOp->putSignal.request);
    EXPECT_EQ(ncclRmaProxyPollNonPersistCompletion(&rma_, ctx_.get(), peer),
              ncclSuccess);

    // The previously-blocked op now issues and advances the consumer index.
    EXPECT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(net_.issueCalls, baseIssue + 1);
    EXPECT_EQ(cis_[peer], 1u);
}

TEST_F(RmaProxyProgressTest, IssuePutSignal_NullRequestOnSuccess_ReturnsError) {
    const uint32_t target = 1;

    // The network violates its contract: reports success but hands back a
    // NULL request handle.
    net_.forceNullOnSuccess = true;

    ncclRmaPutSignalOp op{};
    op.targetRank = static_cast<int>(target);
    op.signal.op = 0;  // iput path

    // The defensive guard rejects it rather than counting a bogus credit.
    EXPECT_EQ(ncclRmaProxyIssuePutSignal(&rma_, ctx_.get(), &op), ncclInternalError);
    EXPECT_EQ(inflight_[target], 0u);
}

// inflightRequests[] is indexed by TARGET rank while circularBuffers / cis /
// pis / inProgressQueues are indexed by QUEUE peer. A target at its ceiling must
// block the op even when the peer's own credit slot is idle.
TEST_F(RmaProxyProgressTest, SinglePut_CreditIsPerTargetNotPerPeer) {
    const int peer = 1;
    const uint32_t target = 0;       // deliberately != peer

    ctx_->maxInflightRequests = 1;
    inflight_[target] = 1;           // the target is at its ceiling
    inflight_[peer]   = 0;           // the peer's own slot is idle

    PushPendingPutSignal(peer, target);

    EXPECT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(net_.issueCalls, 0);   // blocked by the TARGET's credit
    EXPECT_EQ(cis_[peer], 0u);
    EXPECT_EQ(InProgressHead(peer), nullptr);
}

// Mixed-target group: each op is gated by its own target's credit, so a shortage
// on one target head-of-line-blocks the group rather than letting a later op
// through on a different target's budget.
TEST_F(RmaProxyProgressTest, GroupPut_MixedTargets_GatedByEachOpsOwnTarget) {
    const int peer = 1;

    ctx_->maxInflightRequests = 2;
    inflight_[1] = 2;                // target 1 exhausted; target 0 idle

    ncclRmaProxyDesc* desc = PushPendingPutGroup(peer, {0, 1});

    ASSERT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(desc->putSignalGroup.nIssued, 1);   // only the target-0 op issued
    EXPECT_EQ(inflight_[0], 1u);
    EXPECT_EQ(inflight_[1], 2u);                  // untouched
    EXPECT_EQ(cis_[peer], 0u);                    // group stays pending
    EXPECT_EQ(InProgressHead(peer), nullptr);
}

TEST_F(RmaProxyProgressTest, GroupPut_AllOpsFit_MovesToInProgressAndAdvancesCI) {
    const int peer = 1;

    // A group of ops with enough credit for all of them.
    ncclRmaProxyDesc* desc = PushPendingPutGroup(peer, {1, 1});

    EXPECT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);

    EXPECT_EQ(net_.issueCalls, 2);            // both ops issued in one pass
    EXPECT_EQ(InProgressHead(peer), desc);    // group moved to in-progress
    EXPECT_EQ(cis_[peer], 1u);                // consumer index advanced once
}

// A group whose ops don't all fit under the per-target credit budget stays
// pending -- the proxy issues only what credit allows, does not enqueue the
// half-issued group to the in-progress queue, does not advance the consumer
// index, and issues the remainder on a later poll once credit frees up.
TEST_F(RmaProxyProgressTest, GroupPut_PartialCredit_StaysPendingUntilCreditFrees) {
    const int peer = 1;
    const uint32_t target = 1;

    // Budget of 2 credits for `target`; a group of 3 ops all targeting it.
    ctx_->maxInflightRequests = 2;
    ncclRmaProxyDesc* desc = PushPendingPutGroup(peer, {target, target, target});

    // First poll: credit covers only 2 of the 3 ops, so the group stays
    // pending -- not enqueued, CI not advanced.
    ASSERT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(net_.issueCalls, 2);                 // only credit-worth issued
    EXPECT_EQ(InProgressHead(peer), nullptr);      // NOT enqueued while partial
    EXPECT_EQ(cis_[peer], 0u);                      // CI NOT advanced

    // Re-polling with no freed credit makes no further progress.
    ASSERT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(net_.issueCalls, 2);
    EXPECT_EQ(InProgressHead(peer), nullptr);
    EXPECT_EQ(cis_[peer], 0u);

    // Completing the two in-flight ops frees their credit; the next poll issues
    // the held-back third op and only then moves the group to in-progress.
    net_.completeRequest(desc->putSignalGroup.ops[0].request);
    net_.completeRequest(desc->putSignalGroup.ops[1].request);

    ASSERT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(net_.issueCalls, 3);                 // remaining op now issued
    EXPECT_EQ(InProgressHead(peer), desc);         // fully issued -> in-progress
    EXPECT_EQ(cis_[peer], 1u);                      // CI advanced exactly once
}

// A group in-progress with several ops completes incrementally: while ops are
// still outstanding, completion polling makes no progress on the group -- it
// stays at the head of the in-progress queue, is not destroyed, and its doneSeq
// is not yet published. Only once the final op completes (nCompleted == nOps)
// does the poll dequeue, publish doneSeq, and destroy the group.
TEST_F(RmaProxyProgressTest, GroupCompletion_PartialThenFull_StaysInProgressUntilAllOpsComplete) {
    const int peer = 1;
    const uint32_t target = 1;
    const uint64_t opSeq = 5;

    ncclRmaProxyDesc* desc = PushInProgressPutGroup(peer, {target, target, target}, opSeq);

    int destroyCalls = 0;
    ncclRmaProxyDesc* destroyed = nullptr;
    ScopedHook destroy(g_rmaDestroyDesc,
        [&](struct ncclComm*, struct ncclRmaProxyDesc** d) -> ncclResult_t {
            destroyed = *d;
            *d = nullptr;
            ++destroyCalls;
            return ncclSuccess;
        });

    // Poll with no ops complete: group stays put, nothing published/destroyed.
    ASSERT_EQ(ncclRmaProxyPollNonPersistCompletion(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(InProgressHead(peer), desc);
    EXPECT_EQ(destroyCalls, 0);
    EXPECT_EQ(doneSeqStore_.back(), ~opSeq);

    // Complete the first op only: still partial -> no progress on the group.
    net_.completeRequest(desc->putSignalGroup.ops[0].request);
    ASSERT_EQ(ncclRmaProxyPollNonPersistCompletion(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(InProgressHead(peer), desc);
    EXPECT_EQ(destroyCalls, 0);
    EXPECT_EQ(doneSeqStore_.back(), ~opSeq);

    // Complete the second op: two of three done, still not fully done.
    net_.completeRequest(desc->putSignalGroup.ops[1].request);
    ASSERT_EQ(ncclRmaProxyPollNonPersistCompletion(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(InProgressHead(peer), desc);
    EXPECT_EQ(destroyCalls, 0);
    EXPECT_EQ(doneSeqStore_.back(), ~opSeq);

    // Complete the final op: nCompleted == nOps -> group is fully done, so the
    // poll publishes doneSeq, dequeues, and destroys it.
    net_.completeRequest(desc->putSignalGroup.ops[2].request);
    ASSERT_EQ(ncclRmaProxyPollNonPersistCompletion(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(doneSeqStore_.back(), opSeq);       // published for the GPU
    EXPECT_EQ(InProgressHead(peer), nullptr);     // dequeued
    EXPECT_EQ(destroyed, desc);                   // destroyed
    EXPECT_EQ(destroyCalls, 1);
}

// A completion reported by the network while no inflight credit is accounted
// for the target must be rejected -- the underflow guard returns
// ncclInternalError (and WARNs) rather than decrementing the credit counter
// below zero.
TEST_F(RmaProxyProgressTest, Completion_NoInflightCredit_ReturnsErrorWithoutUnderflow) {
    const int peer = 1;
    const uint32_t target = 1;

    ncclRmaProxyDesc* desc = PushInProgressPutSignal(peer, target);

    // Drop the credit the helper accounted so the network reports a completion
    // the proxy has no record of -- the underflow precondition.
    inflight_[target] = 0;

    net_.completeRequest(desc->putSignal.request);

    EXPECT_EQ(ncclRmaProxyPollNonPersistCompletion(&rma_, ctx_.get(), peer),
              ncclInternalError);
    EXPECT_EQ(inflight_[target], 0u);          // guard prevents underflow
}

// Readiness gating on the pending scan. Every pending desc for a peer reads one
// shared, monotonically-advanced readySeq; a desc is issued only once
// readySeq >= its opSeq (rma_proxy_progress.cc: `if (readySeq < opSeq) break`).
// Because the descs sit in a FIFO circular buffer consumed head-first, a head
// whose data the GPU has not yet marked ready blocks every op queued behind it.
// Raising readySeq one opSeq at a time releases the descs strictly in order.
TEST_F(RmaProxyProgressTest, PendingScan_ReadySeqGatesIssueInFifoOrder) {
    const int peer = 1;
    const uint32_t target = 1;

    // One shared per-peer readiness counter, initially behind both ops.
    uint64_t readySeq = 0;
    PushPendingPutSignalShared(peer, target, /*opSeq=*/1, &readySeq);
    PushPendingPutSignalShared(peer, target, /*opSeq=*/2, &readySeq);

    // GPU hasn't signalled readiness for the head: the scan stops at the head
    // even though the buffer is non-empty -- nothing issues, CI stays put.
    ASSERT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(net_.issueCalls, 0);
    EXPECT_EQ(cis_[peer], 0u);
    EXPECT_EQ(InProgressHead(peer), nullptr);

    // Readiness advances to cover only the head (opSeq 1): the head issues and
    // moves to in-progress, but the second op (opSeq 2 > readySeq) is gated and
    // held behind it -- FIFO order preserved.
    readySeq = 1;
    ASSERT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(net_.issueCalls, 1);
    EXPECT_EQ(cis_[peer], 1u);

    // Readiness advances to cover the second op: it now issues too.
    readySeq = 2;
    ASSERT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(net_.issueCalls, 2);
    EXPECT_EQ(cis_[peer], 2u);
}

// Head-of-line completion gate. The in-progress queue drains head-first: while
// the head op is still outstanding, completion polling must not "peek behind"
// it (rma_proxy_progress.cc: `if (!fullyDone) break`). Completing a later op
// first therefore publishes nothing; only once the head completes does the poll
// drain in FIFO order, publishing each doneSeq and destroying each desc.
TEST_F(RmaProxyProgressTest, Completion_HeadOfLineGate_DrainsInFifoOrder) {
    const int peer = 1;
    const uint32_t target = 1;
    const uint64_t headSeq = 3;
    const uint64_t tailSeq = 4;

    ncclRmaProxyDesc* head = PushInProgressPutSignal(peer, target, headSeq);
    ncclRmaProxyDesc* tail = PushInProgressPutSignal(peer, target, tailSeq);

    std::vector<ncclRmaProxyDesc*> destroyOrder;
    ScopedHook destroy(g_rmaDestroyDesc,
        [&](struct ncclComm*, struct ncclRmaProxyDesc** d) -> ncclResult_t {
            destroyOrder.push_back(*d);
            *d = nullptr;
            return ncclSuccess;
        });

    // Complete the *tail* op only. The head is still outstanding, so the poll
    // stops at the head and publishes/destroys nothing behind it.
    net_.completeRequest(tail->putSignal.request);
    ASSERT_EQ(ncclRmaProxyPollNonPersistCompletion(&rma_, ctx_.get(), peer),
              ncclSuccess);
    EXPECT_EQ(InProgressHead(peer), head);       // head still at the front
    EXPECT_TRUE(destroyOrder.empty());           // nothing drained
    EXPECT_EQ(doneSeqStore_[0], ~headSeq);       // head doneSeq unpublished
    EXPECT_EQ(doneSeqStore_[1], ~tailSeq);       // tail doneSeq unpublished

    // Completing the head unblocks the queue: both drain in FIFO order.
    net_.completeRequest(head->putSignal.request);
    ASSERT_EQ(ncclRmaProxyPollNonPersistCompletion(&rma_, ctx_.get(), peer),
              ncclSuccess);
    EXPECT_EQ(InProgressHead(peer), nullptr);              // fully drained
    ASSERT_EQ(destroyOrder.size(), 2u);
    EXPECT_EQ(destroyOrder[0], head);                      // head first
    EXPECT_EQ(destroyOrder[1], tail);                      // then tail
    EXPECT_EQ(doneSeqStore_[0], headSeq);                  // both published
    EXPECT_EQ(doneSeqStore_[1], tailSeq);
}

// Network-side backpressure: credit permits the op but the network's request
// pool is exhausted (#2119's regime). The poll propagates the error leaving no
// residue -- no CI advance, no enqueue, no credit -- and resumes once a slot frees.
TEST_F(RmaProxyProgressTest, SinglePut_NetworkPoolExhausted_PropagatesErrorAndResumes) {
    const int peer = 1;
    const uint32_t target = 1;

    ctx_->maxInflightRequests = 8;   // credit is not the limiter; the network is
    net_.poolSize = 1;

    // Occupy the pool's only slot, with no proxy-side bookkeeping.
    void* hog = nullptr;
    ASSERT_EQ(net_.issue(target, &hog), ncclSuccess);
    const int baseIssue = net_.issueCalls;

    ncclRmaProxyDesc* desc = PushPendingPutSignal(peer, target);

    EXPECT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer),
              ncclInternalError);

    EXPECT_EQ(net_.issueCalls, baseIssue + 1);
    EXPECT_EQ(cis_[peer], 0u);                     // CI NOT advanced
    EXPECT_EQ(InProgressHead(peer), nullptr);      // NOT enqueued
    EXPECT_EQ(inflight_[target], 0u);              // failed op holds no credit
    EXPECT_EQ(desc->putSignal.request, nullptr);

    // The NIC retires the request, freeing the slot.
    net_.completeRequest(hog);
    int done = 0;
    ASSERT_EQ(net_.testReq(hog, &done), ncclSuccess);
    ASSERT_EQ(done, 1);

    ASSERT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(net_.issueCalls, baseIssue + 2);     // retried exactly once
    EXPECT_EQ(cis_[peer], 1u);
    EXPECT_EQ(InProgressHead(peer), desc);
    EXPECT_EQ(inflight_[target], 1u);
}

// Mid-group network refusal: 2 of 3 ops fit. The failed op must be neither
// counted as issued nor charged a credit -- otherwise the retry skips past it and
// the op is silently dropped, which is #2119's symptom.
TEST_F(RmaProxyProgressTest, GroupPut_NetworkPoolExhaustedMidGroup_HoldsPartialProgressAndResumes) {
    const int peer = 1;
    const uint32_t target = 1;

    ctx_->maxInflightRequests = 8;   // credit is not the limiter
    net_.poolSize = 2;               // the network is: only 2 of 3 ops fit

    ncclRmaProxyDesc* desc = PushPendingPutGroup(peer, {target, target, target});

    EXPECT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer),
              ncclInternalError);

    EXPECT_EQ(net_.issueCalls, 3);                       // 2 issued, 3rd attempted
    EXPECT_EQ(desc->putSignalGroup.nIssued, 2);          // failed op NOT counted
    EXPECT_EQ(inflight_[target], 2u);                    // only live ops hold credit
    EXPECT_EQ(desc->putSignalGroup.ops[2].request, nullptr);
    EXPECT_EQ(cis_[peer], 0u);
    EXPECT_EQ(InProgressHead(peer), nullptr);            // group stays pending

    // A completion frees a slot; only the remaining op may issue.
    net_.completeRequest(desc->putSignalGroup.ops[0].request);

    ASSERT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);
    EXPECT_EQ(net_.issueCalls, 4);                       // exactly one more issue
    EXPECT_EQ(desc->putSignalGroup.nIssued, 3);
    EXPECT_EQ(desc->putSignalGroup.nCompleted, 1);
    EXPECT_EQ(InProgressHead(peer), desc);
    EXPECT_EQ(cis_[peer], 1u);
}

// Success with a NULL request, from inside the poll: the guard must reject it
// before a credit is counted, and the poll must propagate rather than advance.
TEST_F(RmaProxyProgressTest, PollDesc_NullRequestOnSuccess_PropagatesWithoutCredit) {
    const int peer = 1;
    const uint32_t target = 1;

    net_.forceNullOnSuccess = true;
    PushPendingPutSignal(peer, target);

    EXPECT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer),
              ncclInternalError);

    EXPECT_EQ(net_.issueCalls, 1);
    EXPECT_EQ(inflight_[target], 0u);          // no credit for a phantom request
    EXPECT_EQ(cis_[peer], 0u);
    EXPECT_EQ(InProgressHead(peer), nullptr);
}

}  // namespace
