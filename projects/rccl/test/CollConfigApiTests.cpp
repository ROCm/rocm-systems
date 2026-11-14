/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The per-collective nccl*Config() entry points on a live communicator. CollConfigTests.cpp pins the parser, and
// that each entry point reaches it, with a null communicator; these run the collectives. On a one-rank
// communicator every collective is a copy, but taskAppend() checks the per-call algorithm selection before it
// takes that shortcut.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <string>
#include <vector>

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {
namespace {

std::string gpuSkipReason() {
  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1) return "requires at least one GPU";
  return "";
}

// A one-rank communicator on GPU 0; each collective copies send into recv.
struct OneRank {
  static constexpr size_t kCount = 1024, kBytes = kCount * sizeof(float);
  ncclComm_t comm = nullptr;
  hipStream_t stream = nullptr;
  float *send = nullptr, *recv = nullptr;
  std::vector<float> expected = std::vector<float>(kCount);

  void setUp() {
    for (size_t i = 0; i < kCount; i++) expected[i] = float(i + 1);
    ncclUniqueId id;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
    ASSERT_EQ(ncclCommInitRank(&comm, 1, id, 0), ncclSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    ASSERT_EQ(hipMalloc(&send, kBytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&recv, kBytes), hipSuccess);
    ASSERT_EQ(hipMemcpy(send, expected.data(), kBytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(recv, 0, kBytes), hipSuccess);
  }
  // Whether the last collective landed in recv; clears recv for the next one.
  bool received() {
    std::vector<float> got(kCount);
    bool ok = hipStreamSynchronize(stream) == hipSuccess &&
              hipMemcpy(got.data(), recv, kBytes, hipMemcpyDeviceToHost) == hipSuccess && got == expected;
    return hipMemset(recv, 0, kBytes) == hipSuccess && ok;
  }
  ~OneRank() {
    if (comm != nullptr) ncclCommDestroy(comm);
    if (stream != nullptr) (void)hipStreamDestroy(stream);
    (void)hipFree(send);
    (void)hipFree(recv);
  }
};

} // namespace

// The same harness through the plain API.
TEST(CollConfigApiTests, ControlPlainAllReduce) {
  RUN_ISOLATED_TEST("CollConfigApiTests.ControlPlainAllReduce", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    OneRank r;
    r.setUp();
    if (::testing::Test::HasFatalFailure()) return;
    ASSERT_EQ(ncclAllReduce(r.send, r.recv, r.kCount, ncclFloat32, ncclSum, r.comm, r.stream), ncclSuccess);
    EXPECT_TRUE(r.received());
  });
}

TEST(CollConfigApiTests, PerCallConfigIsValidated) {
  RUN_ISOLATED_TEST("CollConfigApiTests.PerCallConfigIsValidated", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    OneRank r;
    r.setUp();
    if (::testing::Test::HasFatalFailure()) return;
    auto reduce = [&](const ncclCollConfig_t* cfg) {
      return ncclAllReduceConfig(r.send, r.recv, r.kCount, ncclFloat32, ncclSum, r.comm, r.stream, cfg);
    };

    // No config is the plain API; a valid config runs the collective too.
    ASSERT_EQ(reduce(nullptr), ncclSuccess);
    EXPECT_TRUE(r.received());
    ncclCollConfig_t cfg = NCCL_COLLCONFIG_INITIALIZER;
    cfg.algSelection = "ring";
    ASSERT_EQ(reduce(&cfg), ncclSuccess);
    EXPECT_TRUE(r.received());

    // A config not set up as an ncclCollConfig_t is rejected.
    ncclCollConfig_t bad = cfg;
    bad.magic = 0;
    EXPECT_EQ(reduce(&bad), ncclInvalidArgument);
    bad = cfg;
    bad.size = sizeof(bad.size);
    EXPECT_EQ(reduce(&bad), ncclInvalidArgument);

    // PAT is registered for AllGather and ReduceScatter only: a forced PAT AllReduce is an
    // error, AllGather takes it, and an unforced AllReduce falls back to automatic selection.
    cfg.algSelection = "PAT";
    EXPECT_EQ(reduce(&cfg), ncclInvalidArgument);
    EXPECT_EQ(ncclAllGatherConfig(r.send, r.recv, r.kCount, ncclFloat32, r.comm, r.stream, &cfg), ncclSuccess);
    EXPECT_TRUE(r.received());
    cfg.forceAlgSelection = 0;
    EXPECT_EQ(reduce(&cfg), ncclSuccess);
    EXPECT_TRUE(r.received());
  });
}

} // namespace RcclUnitTesting
