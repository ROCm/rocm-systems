/******************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

// Suite F: gin_anvil_sdma_factory unit tests (validation, mocks, optional HW path).

#include <gin_anvil/sdma_factory.h>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* prev = getenv(name)) {
      had_ = true;
      prev_ = prev;
    }
    setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (had_) {
      setenv(name_, prev_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  std::string prev_;
  bool had_{false};
};

int mockAllgatherFail(void*, void*, size_t) { return -1; }

int mockAllgatherLeaveInvalidPeer(void*, void* buf, size_t bytes_per_rank) {
  const int nRanks = static_cast<int>(bytes_per_rank / sizeof(int));
  auto* devs = static_cast<int*>(buf);
  for (int i = 0; i < nRanks; ++i) {
    if (i == 1) devs[i] = -1;
  }
  return 0;
}

int mockAllgatherFillDevices(void*, void* buf, size_t bytes_per_rank) {
  const int nRanks = static_cast<int>(bytes_per_rank / sizeof(int));
  auto* devs = static_cast<int*>(buf);
  for (int i = 0; i < nRanks; ++i) {
    if (devs[i] < 0) devs[i] = 0;
  }
  return 0;
}

struct CreateOut {
  gin_anvil_sdma_handle_t handle{nullptr};
  void* gpu_handles{nullptr};
  uint64_t* sdma_dirty{nullptr};
};

int tryCreate(int nRanks, int myRank, int deviceId, int (*allgather)(void*, void*, size_t),
              void* allgather_ctx, int num_channels, CreateOut* out) {
  return gin_anvil_sdma_create(nRanks, myRank, deviceId, allgather, allgather_ctx, num_channels,
                               &out->handle, &out->gpu_handles, &out->sdma_dirty);
}

void destroyOut(CreateOut* out) {
  if (out->handle) gin_anvil_sdma_destroy(out->handle);
  out->handle = nullptr;
  out->gpu_handles = nullptr;
  out->sdma_dirty = nullptr;
}

class GinAnvilSdmaFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int ndev = 0;
    ASSERT_EQ(hipGetDeviceCount(&ndev), hipSuccess);
    if (ndev > 0) {
      ASSERT_EQ(hipSetDevice(0), hipSuccess);
    }
  }
};

}  // namespace

// F1: probe returns 0 or 1 depending on HIP + Anvil init.
TEST_F(GinAnvilSdmaFactoryTest, Probe_ReturnsZeroOrOne) {
  const int rc = gin_anvil_sdma_probe();
  EXPECT_TRUE(rc == 0 || rc == 1);
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) {
    EXPECT_EQ(rc, 0);
  }
}

// F2: null / invalid output and rank arguments.
TEST_F(GinAnvilSdmaFactoryTest, Create_NullOutParams) {
  CreateOut out{};
  EXPECT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, nullptr, 1, nullptr), -1);
  EXPECT_EQ(gin_anvil_sdma_create(1, 0, 0, mockAllgatherFillDevices, nullptr, 1, &out.handle,
                                  nullptr, &out.sdma_dirty),
            -1);
  EXPECT_EQ(gin_anvil_sdma_create(1, 0, 0, mockAllgatherFillDevices, nullptr, 1, &out.handle,
                                  &out.gpu_handles, nullptr),
            -1);
  EXPECT_EQ(gin_anvil_sdma_create(1, 0, 0, nullptr, nullptr, 1, &out.handle, &out.gpu_handles,
                                  &out.sdma_dirty),
            -1);
}

TEST_F(GinAnvilSdmaFactoryTest, Create_InvalidRankArgs) {
  CreateOut out{};
  EXPECT_EQ(tryCreate(0, 0, 0, mockAllgatherFillDevices, nullptr, 1, &out), -1);
  EXPECT_EQ(tryCreate(2, -1, 0, mockAllgatherFillDevices, nullptr, 1, &out), -1);
  EXPECT_EQ(tryCreate(2, 2, 0, mockAllgatherFillDevices, nullptr, 1, &out), -1);
}

// F3: allgather failure.
TEST_F(GinAnvilSdmaFactoryTest, Create_AllgatherFail) {
  CreateOut out{};
  EXPECT_EQ(tryCreate(2, 0, 0, mockAllgatherFail, nullptr, 1, &out), -1);
  EXPECT_EQ(out.handle, nullptr);
}

// F4: post-allgather invalid device id.
TEST_F(GinAnvilSdmaFactoryTest, Create_InvalidPeerDev) {
  CreateOut out{};
  EXPECT_EQ(tryCreate(2, 0, 0, mockAllgatherLeaveInvalidPeer, nullptr, 1, &out), -1);
  EXPECT_EQ(out.handle, nullptr);
}

// F5–F7: success path (skipped when probe fails).
TEST_F(GinAnvilSdmaFactoryTest, Create_Success_1Rank) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed (no GPU or USE_SDMA off)";
  }

  CreateOut out{};
  ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, nullptr, 1, &out), 0);
  EXPECT_NE(out.handle, nullptr);
  EXPECT_NE(out.gpu_handles, nullptr);
  EXPECT_NE(out.sdma_dirty, nullptr);

  EXPECT_EQ(gin_anvil_sdma_get_n_ranks(out.handle), 1);
  EXPECT_EQ(gin_anvil_sdma_get_num_channels(out.handle), 1);
  EXPECT_GE(gin_anvil_sdma_get_channel_stride(out.handle), 0);
  EXPECT_LE(gin_anvil_sdma_get_channel_stride(out.handle), 1);

  uint64_t dirty = 0xDEADBEEFULL;
  ASSERT_EQ(hipMemcpy(&dirty, out.sdma_dirty, sizeof(dirty), hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(dirty, 0ULL);

  destroyOut(&out);
}

TEST_F(GinAnvilSdmaFactoryTest, Create_NumChannelsClamp) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }

  CreateOut out{};
  ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, nullptr, 0, &out), 0);
  EXPECT_EQ(gin_anvil_sdma_get_num_channels(out.handle), 1);
  destroyOut(&out);

  ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, nullptr, 99, &out), 0);
  EXPECT_EQ(gin_anvil_sdma_get_num_channels(out.handle), 8);
  destroyOut(&out);
}

TEST_F(GinAnvilSdmaFactoryTest, SpreadChannels_Env) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }

  {
    ScopedEnv spread("NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS", "0");
    CreateOut out{};
    ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, nullptr, 1, &out), 0);
    EXPECT_EQ(gin_anvil_sdma_get_channel_stride(out.handle), 0);
    destroyOut(&out);
  }

  {
    ScopedEnv spread("NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS", "1");
    CreateOut out{};
    ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, nullptr, 1, &out), 0);
    EXPECT_EQ(gin_anvil_sdma_get_channel_stride(out.handle), 1);
    destroyOut(&out);
  }

  unsetenv("NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS");
  CreateOut outDefault{};
  if (tryCreate(1, 0, 0, mockAllgatherFillDevices, nullptr, 1, &outDefault) == 0) {
    EXPECT_EQ(gin_anvil_sdma_get_channel_stride(outDefault.handle), 1);
    destroyOut(&outDefault);
  }
}

// F8: destroy null and valid.
TEST_F(GinAnvilSdmaFactoryTest, Destroy_NullAndValid) {
  gin_anvil_sdma_destroy(nullptr);
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }
  CreateOut out{};
  ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, nullptr, 1, &out), 0);
  destroyOut(&out);
}

// F9: getters on null handle.
TEST_F(GinAnvilSdmaFactoryTest, Getters_NullHandle) {
  EXPECT_EQ(gin_anvil_sdma_get_n_ranks(nullptr), 0);
  EXPECT_EQ(gin_anvil_sdma_get_num_channels(nullptr), 0);
  EXPECT_EQ(gin_anvil_sdma_get_channel_stride(nullptr), 0);
}

// F10: multi-rank mock allgather (exercises connect loop when SDMA available).
TEST_F(GinAnvilSdmaFactoryTest, Create_MultiRankMock) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }
  int ndev = 0;
  ASSERT_EQ(hipGetDeviceCount(&ndev), hipSuccess);
  if (ndev < 1) {
    GTEST_SKIP() << "No HIP devices";
  }

  const int nRanks = ndev >= 2 ? 2 : 1;
  CreateOut out{};
  ASSERT_EQ(tryCreate(nRanks, 0, 0, mockAllgatherFillDevices, nullptr, 2, &out), 0);
  EXPECT_EQ(gin_anvil_sdma_get_n_ranks(out.handle), nRanks);
  EXPECT_EQ(gin_anvil_sdma_get_num_channels(out.handle), 2);
  destroyOut(&out);
}

// F11: spread env atoi paths (non-zero/non-one numeric and invalid string).
TEST_F(GinAnvilSdmaFactoryTest, SpreadChannels_EnvAtoi) {
  if (gin_anvil_sdma_probe() <= 0) {
    GTEST_SKIP() << "Anvil SDMA probe failed";
  }

  {
    ScopedEnv spread("NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS", "2");
    CreateOut out{};
    ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, nullptr, 1, &out), 0);
    EXPECT_EQ(gin_anvil_sdma_get_channel_stride(out.handle), 1);
    destroyOut(&out);
  }

  {
    ScopedEnv spread("NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS", "x");
    CreateOut out{};
    ASSERT_EQ(tryCreate(1, 0, 0, mockAllgatherFillDevices, nullptr, 1, &out), 0);
    EXPECT_EQ(gin_anvil_sdma_get_channel_stride(out.handle), 0);
    destroyOut(&out);
  }
}
