/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/commDump.cc: ncclCommDump's null guards and the profiler proxy-trace dump.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../common/LogCapture.hpp"
#include "comm.h"
#include "fakes/profiler_fakes.h"

#include COMM_DUMP_CC_PATH

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

class CommDumpMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetProfilerFakes();
    comm_ = std::make_unique<ncclComm>();
    comm_->proxyState = reinterpret_cast<ncclProxyState*>(0x50);
    comm_->profilerContext = reinterpret_cast<void*>(0xc7);
  }

  std::unique_ptr<ncclComm> comm_;
  std::unordered_map<std::string, std::string> map_{{"key", "value"}};
};

TEST_F(CommDumpMicrotest, DumpsTheProfilerProxyTrace) {
  ncclResult_t res = ncclInternalError;
  std::string log = CaptureLog([&] { res = ncclCommDump(comm_.get(), map_); });

  EXPECT_EQ(res, ncclSuccess);
  EXPECT_EQ(ProfilerProxyTraceDumps(), std::vector<void*>{reinterpret_cast<void*>(0xc7)});
  EXPECT_TRUE(LogHas(log, "ncclCommDump() proxy-trace (profiler plugin):")) << log;
  EXPECT_EQ(map_, (std::unordered_map<std::string, std::string>{{"key", "value"}}));
}

TEST_F(CommDumpMicrotest, NullCommIsSkipped) {
  EXPECT_EQ(ncclCommDump(nullptr, map_), ncclSuccess);
  EXPECT_TRUE(ProfilerProxyTraceDumps().empty());
}

TEST_F(CommDumpMicrotest, MissingProxyStateIsSkipped) {
  comm_->proxyState = nullptr;

  EXPECT_EQ(ncclCommDump(comm_.get(), map_), ncclSuccess);
  EXPECT_TRUE(ProfilerProxyTraceDumps().empty());
}

}  // namespace
