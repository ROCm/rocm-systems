/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include <vector>

#include "gtest/gtest.h"
#include "../src/sdma_agreement.hpp"

using rocshmem::agreeSdmaStatus;
using rocshmem::SdmaDecision;
using rocshmem::kSdmaConnectFailed;
using rocshmem::kSdmaDisabledByEnv;
using rocshmem::kSdmaReady;

namespace {

rocshmem::SdmaAgreement agree(const std::vector<int>& statuses) {
  return agreeSdmaStatus(statuses.data(), static_cast<int>(statuses.size()));
}

}  // namespace

TEST(SdmaAgreement, AllReadyCommits) {
  const auto a = agree({kSdmaReady, kSdmaReady, kSdmaReady, kSdmaReady});
  EXPECT_EQ(a.decision, SdmaDecision::kCommit);
  EXPECT_TRUE(a.ranks.empty());
}

TEST(SdmaAgreement, SingleRankCommitsOrDisables) {
  EXPECT_EQ(agree({kSdmaReady}).decision, SdmaDecision::kCommit);
  EXPECT_EQ(agree({kSdmaDisabledByEnv}).decision, SdmaDecision::kAllDisabled);
  EXPECT_EQ(agree({kSdmaConnectFailed}).decision, SdmaDecision::kFallbackConnectFailed);
}

TEST(SdmaAgreement, AllDisabledByEnvNeedsNoFallback) {
  const auto a = agree({kSdmaDisabledByEnv, kSdmaDisabledByEnv, kSdmaDisabledByEnv});
  EXPECT_EQ(a.decision, SdmaDecision::kAllDisabled);
  EXPECT_TRUE(a.ranks.empty());
}

TEST(SdmaAgreement, OneFailedConnectFallsBackEveryRank) {
  // Queue over-subscription usually fails only the ranks that initialize last.
  const auto a = agree({kSdmaReady, kSdmaReady, kSdmaReady, kSdmaConnectFailed});
  EXPECT_EQ(a.decision, SdmaDecision::kFallbackConnectFailed);
  EXPECT_EQ(a.ranks, "3");
}

TEST(SdmaAgreement, NamesEveryFailedRankInOrder) {
  const auto a = agree({kSdmaConnectFailed, kSdmaReady, kSdmaConnectFailed, kSdmaConnectFailed});
  EXPECT_EQ(a.decision, SdmaDecision::kFallbackConnectFailed);
  EXPECT_EQ(a.ranks, "0,2,3");
}

TEST(SdmaAgreement, ConnectFailureOutranksEnvMismatch) {
  const auto a = agree({kSdmaDisabledByEnv, kSdmaReady, kSdmaConnectFailed});
  EXPECT_EQ(a.decision, SdmaDecision::kFallbackConnectFailed);
  EXPECT_EQ(a.ranks, "2");
}

TEST(SdmaAgreement, PartialEnvDisableFallsBackEveryRank) {
  // A rank that skipped SDMA runs the pSync alltoall; peers that kept it would wait forever.
  const auto a = agree({kSdmaReady, kSdmaDisabledByEnv, kSdmaReady, kSdmaDisabledByEnv});
  EXPECT_EQ(a.decision, SdmaDecision::kFallbackEnvMismatch);
  EXPECT_EQ(a.ranks, "1,3");
}

TEST(SdmaAgreement, UnknownStatusCountsAsFailedConnect) {
  const auto a = agree({kSdmaReady, 7, kSdmaReady, -1});
  EXPECT_EQ(a.decision, SdmaDecision::kFallbackConnectFailed);
  EXPECT_EQ(a.ranks, "1,3");
}
