/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>

#include <initializer_list>
#include <vector>

#include <nccl.h>

namespace {

struct AsyncReply {
  ncclResult_t queryResult;
  ncclResult_t state;
};

std::vector<AsyncReply> replies;
size_t queryCount = 0;
size_t commExpressionCount = 0;
ncclComm_t fakeComm = reinterpret_cast<ncclComm_t>(1);

ncclComm_t countedComm() {
  ++commExpressionCount;
  return fakeComm;
}

ncclResult_t mockCommGetAsyncError(ncclComm_t comm, ncclResult_t* state) {
  EXPECT_EQ(fakeComm, comm);
  if (queryCount >= replies.size()) {
    ADD_FAILURE() << "readiness helper queried after the terminal reply";
    return ncclInternalError;
  }
  const AsyncReply reply = replies[queryCount++];
  *state = reply.state;
  return reply.queryResult;
}

} // namespace

/* Isolate the real inline helper in this translation unit. Other test and
 * library code continues to call the real NCCL query and readiness helper. */
#define ncclCommGetAsyncError mockCommGetAsyncError
#define m2nWaitCommReady testM2nWaitCommReady
#include "test_helpers.h"
#undef m2nWaitCommReady
#undef ncclCommGetAsyncError

namespace {

class NonblockingCommWaitTest : public ::testing::Test {
protected:
  void SetUp() override {
    replies.clear();
    queryCount = 0;
    commExpressionCount = 0;
  }

  void setReplies(std::initializer_list<AsyncReply> values) {
    replies = values;
  }
};

TEST_F(NonblockingCommWaitTest, SuccessReturnStillChecksPendingState) {
  setReplies({{ncclSuccess, ncclInProgress}, {ncclSuccess, ncclSuccess}});
  ASSERT_NCCL(ncclSuccess, fakeComm);
  EXPECT_EQ(2u, queryCount);
}

TEST_F(NonblockingCommWaitTest, InProgressReturnWaitsForReadiness) {
  setReplies({{ncclSuccess, ncclInProgress}, {ncclSuccess, ncclInProgress}, {ncclSuccess, ncclSuccess}});
  ASSERT_NCCL(ncclInProgress, fakeComm);
  EXPECT_EQ(3u, queryCount);
}

TEST_F(NonblockingCommWaitTest, AlreadyReadyNeedsOneQuery) {
  setReplies({{ncclSuccess, ncclSuccess}});
  ASSERT_NCCL(ncclSuccess, fakeComm);
  EXPECT_EQ(1u, queryCount);
}

TEST_F(NonblockingCommWaitTest, AssertsTerminalAsyncError) {
  setReplies({{ncclSuccess, ncclInProgress}, {ncclSuccess, ncclRemoteError}});
  EXPECT_FATAL_FAILURE(ASSERT_NCCL(ncclSuccess, fakeComm), "ncclSuccess");
  EXPECT_EQ(2u, queryCount);
}

TEST_F(NonblockingCommWaitTest, AssertsQueryErrorWithoutRetry) {
  setReplies({{ncclSystemError, ncclInProgress}});
  EXPECT_FATAL_FAILURE(ASSERT_NCCL(ncclSuccess, fakeComm), "ncclSuccess");
  EXPECT_EQ(1u, queryCount);
}

TEST_F(NonblockingCommWaitTest, CallErrorDoesNotQueryCommunicator) {
  EXPECT_FATAL_FAILURE(ASSERT_NCCL(ncclInvalidArgument, countedComm()), "ncclInvalidArgument");
  EXPECT_EQ(0u, queryCount);
  EXPECT_EQ(0u, commExpressionCount);
}

TEST_F(NonblockingCommWaitTest, EvaluatesCallOnceBeforeCommunicator) {
  setReplies({{ncclSuccess, ncclSuccess}});
  ncclComm_t comm = nullptr;
  int callCount = 0;
  int commCount = 0;
  auto init = [&]() {
    ++callCount;
    comm = fakeComm;
    return ncclInProgress;
  };
  ASSERT_NCCL(init(), (++commCount, comm));
  EXPECT_EQ(1, callCount);
  EXPECT_EQ(1, commCount);
  EXPECT_EQ(1u, queryCount);
}

TEST_F(NonblockingCommWaitTest, PlainAssertionDoesNotQueryCommunicator) {
  ASSERT_NCCL(ncclSuccess);
  EXPECT_EQ(0u, queryCount);
}

} // namespace
