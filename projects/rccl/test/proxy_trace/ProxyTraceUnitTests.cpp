/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "proxy.h"
#include "proxy_trace/proxy_trace.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <unistd.h>
namespace RcclUnitTesting {

class ProxyTraceTestFixture : public ::testing::Test {
public:
  ncclProxyState *proxyState;
  ncclProxySubArgs *sub1, *sub2;
  uint64_t commHash = 123456789;
  int64_t opCount = 31;
  int nSteps = 10;
  void SetUp() override {
    proxyState = new ncclProxyState();
    facebook_rccl::proxyTraceInit(proxyState->proxyTrace, 0, commHash);
    EXPECT_NE(proxyState->proxyTrace, nullptr);
    sub1 = new ncclProxySubArgs();
    sub2 = new ncclProxySubArgs();
    sub1->traceKey = {commHash, opCount, -1};
    sub2->traceKey = {commHash, opCount, -1};
    sub1->nsteps = nSteps;
    sub2->nsteps = nSteps;
  }
  void TearDown() override {
    delete sub1;
    delete sub2;
    delete proxyState;
  }
  void AddTraceOp(ncclProxySubArgs *sub, facebook_rccl::ProxyOpType opType) {
    facebook_rccl::addNewProxyOp(proxyState->proxyTrace, sub->traceKey,
                                 sub->traceInfo, opType, sub->channelId,
                                 sub->nsteps, sub->nbytes, sub->peer);
  }
};

TEST_F(ProxyTraceTestFixture, nonEmptySingleton) {
  const auto &tracer = proxyState->proxyTrace;
  EXPECT_NE(tracer, nullptr);
}

TEST_F(ProxyTraceTestFixture, addTraceOp) {
  auto &tracer = proxyState->proxyTrace;
  EXPECT_EQ(tracer->getOrCreateProxyOpId(sub1->traceKey.commHash,
                                         sub1->traceKey.opCount),
            0);
  AddTraceOp(sub1, facebook_rccl::ProxyOpType::SEND);
  EXPECT_EQ(sub1->traceKey.proxyOpId, 0);
  AddTraceOp(sub2, facebook_rccl::ProxyOpType::RECV);
  EXPECT_EQ(sub2->traceKey.proxyOpId, 1);
  EXPECT_EQ(tracer->getOrCreateProxyOpId(sub1->traceKey.commHash,
                                         sub1->traceKey.opCount),
            2);
  auto traceRecordPtr = tracer->getProxyTraceOpPtr(sub1->traceKey);
  EXPECT_EQ(traceRecordPtr->opType, facebook_rccl::ProxyOpType::SEND);
}

TEST_F(ProxyTraceTestFixture, getMapSizeMB) {
  auto &tracer = proxyState->proxyTrace;
  AddTraceOp(sub1, facebook_rccl::ProxyOpType::SEND);
  auto size1 = tracer->getMapSizeMB();
  EXPECT_GT(size1, 0);
  AddTraceOp(sub2, facebook_rccl::ProxyOpType::RECV);
  auto size2 = tracer->getMapSizeMB();
  EXPECT_GT(size2, size1);
  // finish sub1
  sub1->done = nSteps;
  facebook_rccl::updateProxyOpCounter(tracer, sub1->traceKey,
                                      facebook_rccl::ProxyCounterTypes::DONE,
                                      sub1->done);
  // sub1 is now serialized and should be moved from activeOps to finishedOps
  auto size3 = tracer->getMapSizeMB();
  EXPECT_GT(size3, size1);
}

TEST_F(ProxyTraceTestFixture, updateTraceOp) {
  auto &tracer = proxyState->proxyTrace;
  AddTraceOp(sub1, facebook_rccl::ProxyOpType::SEND);
  facebook_rccl::updateProxyOpCounter(
      tracer, sub1->traceKey,
      facebook_rccl::ProxyCounterTypes::KERNEL_COPY_READY, 1);
  facebook_rccl::updateProxyOpCounter(
      tracer, sub1->traceKey, facebook_rccl::ProxyCounterTypes::POSTED, 3);
  facebook_rccl::updateProxyOpCounter(
      tracer, sub1->traceKey, facebook_rccl::ProxyCounterTypes::TRANSMITTED, 2);

  auto traceRecordPtr = tracer->getProxyTraceOpPtr(sub1->traceKey);
  EXPECT_NE(traceRecordPtr, nullptr);
  EXPECT_EQ(traceRecordPtr->counters[facebook_rccl::ProxyCounterTypes::POSTED],
            3);
  EXPECT_EQ(
      traceRecordPtr->counters[facebook_rccl::ProxyCounterTypes::TRANSMITTED],
      2);
  EXPECT_EQ(traceRecordPtr
                ->counters[facebook_rccl::ProxyCounterTypes::KERNEL_COPY_READY],
            1);
  EXPECT_GE(traceRecordPtr->lastUpdateTs, traceRecordPtr->startTs);
}

TEST_F(ProxyTraceTestFixture, updateTraceOp2) {
  auto &tracer = proxyState->proxyTrace;
  AddTraceOp(sub1, facebook_rccl::ProxyOpType::SEND);
  int64_t rand = 123456789;
  sub1->posted = rand;
  facebook_rccl::updateProxyOpCounter(tracer, sub1->traceKey,
                                      facebook_rccl::ProxyCounterTypes::POSTED,
                                      sub1->posted);
  auto traceRecordPtr = tracer->getProxyTraceOpPtr(sub1->traceKey);
  EXPECT_EQ(traceRecordPtr->counters[facebook_rccl::ProxyCounterTypes::POSTED],
            rand);
}

TEST_F(ProxyTraceTestFixture, memoryReclaim) {
  auto &tracer = proxyState->proxyTrace;
  tracer->resetAll();
  AddTraceOp(sub1, facebook_rccl::ProxyOpType::SEND);
  sub1->done = nSteps;
  facebook_rccl::updateProxyOpCounter(tracer, sub1->traceKey,
                                      facebook_rccl::ProxyCounterTypes::DONE,
                                      sub1->done);
  auto traceRecordPtr = tracer->getProxyTraceOpPtr(sub1->traceKey);
  EXPECT_EQ(traceRecordPtr, nullptr);
  EXPECT_GT(tracer->getMapSizeMB(), 0);
}

} // namespace RcclUnitTesting