/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "proxy_trace/proxy_trace.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <unistd.h>
namespace RcclUnitTesting {

class ProxyTraceTestFixture : public ::testing::Test {
public:
  std::unique_ptr<facebook_rccl::ProxyTrace> tracer;
  facebook_rccl::ProxyTraceRecordKey traceKey1;
  facebook_rccl::ProxyTraceRecordKey traceKey2;
  uint64_t commHash = 123456789;
  int64_t opCount = 31;
  int nSteps = 10;
  void SetUp() override {
    tracer = std::make_unique<facebook_rccl::ProxyTrace>(0);
    EXPECT_NE(tracer, nullptr);
    traceKey1 = {commHash, opCount, -1};
    traceKey2 = {commHash, opCount, -1};
  }
  void TearDown() override { tracer.reset(); }
  void AddTraceOp(facebook_rccl::ProxyTraceRecordKey& key, facebook_rccl::ProxyOpType opType) {
    facebook_rccl::ProxyTraceExtraInfo ex{};
    tracer->addNewProxyOp(key, ex, opType, 0, nSteps, 100, 0);
  }
};

TEST_F(ProxyTraceTestFixture, nonEmptySingleton) { EXPECT_NE(tracer, nullptr); }

TEST_F(ProxyTraceTestFixture, addTraceOp) {
  AddTraceOp(traceKey1, facebook_rccl::ProxyOpType::SEND);
  EXPECT_EQ(traceKey1.proxyOpId, 0);
  AddTraceOp(traceKey2, facebook_rccl::ProxyOpType::RECV);
  EXPECT_EQ(traceKey2.proxyOpId, 1);
  auto traceRecordPtr = tracer->getProxyTraceOpPtr(traceKey1);
  EXPECT_EQ(traceRecordPtr->opType, facebook_rccl::ProxyOpType::SEND);
}

TEST_F(ProxyTraceTestFixture, getMapSizeMB) {
  AddTraceOp(traceKey1, facebook_rccl::ProxyOpType::SEND);
  auto size1 = tracer->getMapSizeMB();
  EXPECT_GT(size1, 0);
  AddTraceOp(traceKey2, facebook_rccl::ProxyOpType::RECV);
  auto size2 = tracer->getMapSizeMB();
  EXPECT_GT(size2, size1);
  tracer->updateProxyOpCounter(traceKey1, facebook_rccl::ProxyCounterTypes::DONE, nSteps);
  auto size3 = tracer->getMapSizeMB();
  EXPECT_GT(size3, size1);
}

TEST_F(ProxyTraceTestFixture, updateTraceOp) {
  AddTraceOp(traceKey1, facebook_rccl::ProxyOpType::SEND);
  tracer->updateProxyOpCounter(traceKey1, facebook_rccl::ProxyCounterTypes::KERNEL_COPY_READY, 1);
  tracer->updateProxyOpCounter(traceKey1, facebook_rccl::ProxyCounterTypes::POSTED, 3);
  tracer->updateProxyOpCounter(traceKey1, facebook_rccl::ProxyCounterTypes::TRANSMITTED, 2);

  auto traceRecordPtr = tracer->getProxyTraceOpPtr(traceKey1);
  EXPECT_NE(traceRecordPtr, nullptr);
  EXPECT_EQ(traceRecordPtr->counters[facebook_rccl::ProxyCounterTypes::POSTED], 3);
  EXPECT_EQ(traceRecordPtr->counters[facebook_rccl::ProxyCounterTypes::TRANSMITTED], 2);
  EXPECT_EQ(traceRecordPtr->counters[facebook_rccl::ProxyCounterTypes::KERNEL_COPY_READY], 1);
  EXPECT_GE(traceRecordPtr->lastUpdateTs, traceRecordPtr->startTs);
}

TEST_F(ProxyTraceTestFixture, updateTraceOp2) {
  AddTraceOp(traceKey1, facebook_rccl::ProxyOpType::SEND);
  int64_t rand = 123456789;
  tracer->updateProxyOpCounter(traceKey1, facebook_rccl::ProxyCounterTypes::POSTED, rand);
  auto traceRecordPtr = tracer->getProxyTraceOpPtr(traceKey1);
  EXPECT_EQ(traceRecordPtr->counters[facebook_rccl::ProxyCounterTypes::POSTED], rand);
}

} // namespace RcclUnitTesting
