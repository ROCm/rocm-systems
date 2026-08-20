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

private:
    static ncclResult_t TrampIput(void* rmaCtx, int, uint64_t, void*, size_t,
                                  uint64_t, void*, uint32_t rank, void** request) {
        return static_cast<FakeNet*>(rmaCtx)->issue(rank, request);
    }
    static ncclResult_t TrampIputSignal(void* rmaCtx, int, uint64_t, void*, size_t,
                                        uint64_t, void*, uint32_t rank, uint64_t,
                                        void*, uint64_t, uint32_t, bool, void** request) {
        return static_cast<FakeNet*>(rmaCtx)->issue(rank, request);
    }
    static ncclResult_t TrampTest(void* collComm, void* request, int* done) {
        return static_cast<FakeNet*>(collComm)->testReq(request, done);
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
        ctx_->rmaCtx = &net_;
        ctx_->rmaCollComm = &net_;

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

TEST_F(RmaProxyProgressTest, GroupPut_AllOpsFit_MovesToInProgressAndAdvancesCI) {
    const int peer = 1;

    // A group of ops with enough credit for all of them.
    ncclRmaProxyDesc* desc = PushPendingPutGroup(peer, {1, 1});

    EXPECT_EQ(ncclRmaProxyPollNonPersistDesc(&rma_, ctx_.get(), peer), ncclSuccess);

    EXPECT_EQ(net_.issueCalls, 2);            // both ops issued in one pass
    EXPECT_EQ(InProgressHead(peer), desc);    // group moved to in-progress
    EXPECT_EQ(cis_[peer], 1u);                // consumer index advanced once
}

}  // namespace
