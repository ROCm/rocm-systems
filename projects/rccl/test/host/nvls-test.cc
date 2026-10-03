/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// Host-only microtests for src/transport/nvls.cc as RCCL builds it.
//
// NVLS (NVLink SHARP) multicast exists only behind `#if CUDART_VERSION >= 12010`,
// which no RCCL build defines, so RCCL always compiles the stub branch of nvls.cc.
// NCCL 2.29 reports a failed cuMulticastBindMem as WARN plus an initialization
// error for NCCL_NVLS_ENABLE=1 and =2. These tests pin the RCCL side of that: the
// stubs never read NCCL_NVLS_ENABLE, so no value turns NVLS on, no multicast bind
// is ever attempted, and every NVLS entry point succeeds without effect.
//
// The suite compiles the hipified nvls.cc directly (via NVLS_CC_PATH), with
// NCCL_PARAM redirected through g_loadParam so each case can detect a stub that
// starts reading NCCL_NVLS_ENABLE.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "ScopedHook.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"

#include NVLS_CC_PATH

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12010
#error "nvls.cc compiled its CUDA multicast branch; RCCL must build the NVLS stubs"
#endif

namespace {

class NvlsHipStubTest : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> commStorage_;
  ncclComm* comm_ = nullptr;
  std::vector<std::string> nvlsEnableLookups_;

  void SetUp() override {
    ResetNcclFakes();
    commStorage_ = std::make_unique<ncclComm>();
    comm_ = commStorage_.get();
  }

  // A g_loadParam hook that records each NVLS_ENABLE lookup. No stub reads the param, so its
  // value cannot change their behavior; the empty-lookups checks pin that invariant instead.
  auto RecordNvlsEnableLookups() {
    return [this](const char* env, int64_t deft) -> int64_t {
      if (std::strcmp(env, "NVLS_ENABLE") == 0) nvlsEnableLookups_.emplace_back(env);
      return deft;
    };
  }
};

TEST_F(NvlsHipStubTest, DefaultMatchesNccl) {
  int64_t seenDefault = -1;
  ScopedHook param(g_loadParam, [&seenDefault](const char* env, int64_t deft) -> int64_t {
    if (std::strcmp(env, "NVLS_ENABLE") == 0) seenDefault = deft;
    return deft;
  });

  EXPECT_EQ(ncclParamNvlsEnable(), 2);
  EXPECT_EQ(seenDefault, 2);
}

TEST_F(NvlsHipStubTest, InitLeavesNvlsOff) {
  ScopedHook param(g_loadParam, RecordNvlsEnableLookups());
  comm_->nvlsChannels = 16;

  EXPECT_EQ(ncclNvlsInit(comm_), ncclSuccess);

  EXPECT_EQ(comm_->nvlsChannels, 0);
  EXPECT_TRUE(nvlsEnableLookups_.empty()) << "the stubs must not consult NCCL_NVLS_ENABLE";
}

TEST_F(NvlsHipStubTest, SetupPathSucceedsWithoutAllocating) {
  ScopedHook param(g_loadParam, RecordNvlsEnableLookups());

  ASSERT_EQ(ncclNvlsInit(comm_), ncclSuccess);
  EXPECT_EQ(ncclNvlsSetup(comm_, /*parent=*/nullptr), ncclSuccess);
  EXPECT_EQ(ncclNvlsBufferSetup(comm_), ncclSuccess);
  EXPECT_EQ(ncclNvlsTreeConnect(comm_), ncclSuccess);
  EXPECT_EQ(ncclNvlsFree(comm_), ncclSuccess);

  // Tripwire: the stubs never write nvlsResources; this fails only if one starts allocating.
  EXPECT_EQ(comm_->nvlsResources, nullptr);
  EXPECT_TRUE(nvlsEnableLookups_.empty());
}

TEST_F(NvlsHipStubTest, BufferRegistrationFallsBackToNonNvls) {
  ScopedHook param(g_loadParam, RecordNvlsEnableLookups());
  alignas(64) char send[64] = {};
  alignas(64) char recv[64] = {};
  int regUsed = 1;
  void* regSend = send;
  void* regRecv = recv;

  EXPECT_EQ(ncclNvlsLocalRegisterBuffer(comm_, send, recv, sizeof(send), sizeof(recv), &regUsed, &regSend,
                                        &regRecv),
            ncclSuccess);
  EXPECT_EQ(regUsed, 0);

  int recChannels = -1;
  EXPECT_EQ(ncclNvlsRegResourcesQuery(comm_, ncclFuncAllReduce, &recChannels), ncclSuccess);
  EXPECT_EQ(recChannels, 0);
  EXPECT_TRUE(nvlsEnableLookups_.empty());
}

}  // namespace
