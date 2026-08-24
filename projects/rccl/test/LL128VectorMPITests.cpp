/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file LL128VectorMPITests.cpp
 * @brief Integration tests for explicit 128-bit vector load/store paths in LL / LL128.
 *
 * Run examples:
 *   mpirun -np 2 ./rccl-UnitTestsMPI --gtest_filter=LL128VectorMPITest.*
 *   mpirun -np 2 -x NCCL_LOCAL_REGISTER=1 -x RCCL_LL128_FORCE_ENABLE=1 \
 *     ./rccl-UnitTestsMPI --gtest_filter=LL128VectorMPITest.RegisteredAllReduce_LL128_IntraNode
 */

#ifdef MPI_TESTS_ENABLED

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;

namespace RcclUnitTesting
{

namespace
{

static bool deviceArchIsGfx1250() {
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) return false;
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, device) != hipSuccess) return false;
  return std::string(prop.gcnArchName).rfind("gfx1250", 0) == 0;
}

static void initSendBuffer(float* buf, size_t count, int rank) {
  for (size_t i = 0; i < count; ++i) {
    buf[i] = static_cast<float>(rank + 1) + static_cast<float>(i) * 0.001f;
  }
}

static bool verifyAllReduceResult(float* buf, size_t count, int nRanks) {
  const float expectedScale = static_cast<float>(nRanks * (nRanks + 1)) / 2.0f;
  std::vector<float> h(count);
  if (hipMemcpy(h.data(), buf, count * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    const float expected = expectedScale + static_cast<float>(i) * 0.001f * static_cast<float>(nRanks);
    if (h[i] != expected) return false;
  }
  return true;
}

// enqueue.cc emits NCCL_TUNING lines like:
//   AllReduce: 16384 Bytes -> Algo RING proto LL128 channel{Lo..Hi}={0..0}
static bool logsContainForcedProto(const std::string& log, const char* proto) {
  if (log.find("AllReduce:") == std::string::npos) return false;
  const std::string protoTag = std::string("proto ") + proto + " ";
  return log.find(protoTag) != std::string::npos;
}

class LL128VectorMPITest : public MPITestBase
{
protected:
  std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugGuard_;
  std::unique_ptr<MPIHelpers::MpiEnvGuard>             tuningGuard_;
  std::unique_ptr<MPIHelpers::TestLogAssertionContext> logCtx_;

  void SetUp() override {
    MPITestBase::SetUp();
    debugGuard_  = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG", "INFO");
    tuningGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG_SUBSYS", "TUNING");
    logCtx_ = std::make_unique<MPIHelpers::TestLogAssertionContext>(
        MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));
  }

  void TearDown() override {
    MPITestBase::TearDown();
    logCtx_.reset();
    tuningGuard_.reset();
    debugGuard_.reset();
  }

  std::string readAllLogs() const {
    return logCtx_->readNcclDebugLog() + logCtx_->readPerRankStderrLog();
  }
};

} // namespace

TEST_F(LL128VectorMPITest, RegisteredAllReduce_LL128_IntraNode) {
  if (!validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kRequireSingleNode)) {
    GTEST_SKIP() << "Requires 2+ ranks on a single node";
  }

  MPIHelpers::MpiEnvGuard protoGuard("NCCL_PROTO", "LL128");
  MPIHelpers::MpiEnvGuard algoGuard("NCCL_ALGO", "Ring");
  MPIHelpers::MpiEnvGuard ll128ForceGuard("RCCL_LL128_FORCE_ENABLE", "1");
  MPIHelpers::MpiEnvGuard localRegGuard("NCCL_LOCAL_REGISTER", "1");

  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  int rank = 0, nRanks = 0;
  ncclCommUserRank(getActiveCommunicator(), &rank);
  ncclCommCount(getActiveCommunicator(), &nRanks);

  // Small enough to stay in the LL128 latency window on typical configs.
  const size_t count = 4096;
  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  void* sendHandle = nullptr;
  void* recvHandle = nullptr;

  ASSERT_MPI_EQ(ncclSuccess, allocateDeviceBuffer(&sendBuf, count * sizeof(float)));
  ASSERT_MPI_EQ(ncclSuccess, allocateDeviceBuffer(&recvBuf, count * sizeof(float)));
  auto cleanup = makeScopeGuard([&]() {
    if (sendHandle) ncclCommDeregister(getActiveCommunicator(), sendHandle);
    if (recvHandle) ncclCommDeregister(getActiveCommunicator(), recvHandle);
    if (sendBuf) freeDeviceBuffer(sendBuf);
    if (recvBuf) freeDeviceBuffer(recvBuf);
  });

  ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommRegister(getActiveCommunicator(), sendBuf, count * sizeof(float), &sendHandle));
  ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommRegister(getActiveCommunicator(), recvBuf, count * sizeof(float), &recvHandle));

  initSendBuffer(static_cast<float*>(sendBuf), count, rank);

  ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, getActiveCommunicator(),
                                getActiveStream()));
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

  ASSERT_TRUE(verifyAllReduceResult(static_cast<float*>(recvBuf), count, nRanks));
  ASSERT_TRUE(logsContainForcedProto(readAllLogs(), "LL128"))
      << "Expected NCCL_TUNING AllReduce line with forced LL128 protocol";
}

TEST_F(LL128VectorMPITest, AllReduce_LL_IntraNode) {
  if (!validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kRequireSingleNode)) {
    GTEST_SKIP() << "Requires 2+ ranks on a single node";
  }

  MPIHelpers::MpiEnvGuard protoGuard("NCCL_PROTO", "LL");
  MPIHelpers::MpiEnvGuard algoGuard("NCCL_ALGO", "Ring");
  MPIHelpers::MpiEnvGuard llBufGuard("NCCL_ALLOC_P2P_NET_LL_BUFFERS", "0");

  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  int rank = 0, nRanks = 0;
  ncclCommUserRank(getActiveCommunicator(), &rank);
  ncclCommCount(getActiveCommunicator(), &nRanks);

  const size_t count = 2048;
  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  ASSERT_MPI_EQ(ncclSuccess, allocateDeviceBuffer(&sendBuf, count * sizeof(float)));
  ASSERT_MPI_EQ(ncclSuccess, allocateDeviceBuffer(&recvBuf, count * sizeof(float)));
  auto cleanup = makeScopeGuard([&]() {
    if (sendBuf) freeDeviceBuffer(sendBuf);
    if (recvBuf) freeDeviceBuffer(recvBuf);
  });

  initSendBuffer(static_cast<float*>(sendBuf), count, rank);

  ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, getActiveCommunicator(),
                                getActiveStream()));
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

  ASSERT_TRUE(verifyAllReduceResult(static_cast<float*>(recvBuf), count, nRanks));
  ASSERT_TRUE(logsContainForcedProto(readAllLogs(), "LL"))
      << "Expected NCCL_TUNING AllReduce line with forced LL protocol";
}

TEST_F(LL128VectorMPITest, RegisteredAllReduce_LL128_Gfx1250Only) {
#if !__has_builtin(__builtin_amdgcn_cooperative_atomic_load_8x16B)
  GTEST_SKIP() << "Cooperative atomic 128-bit builtins unavailable in this toolchain";
#endif
  if (!deviceArchIsGfx1250()) {
    GTEST_SKIP() << "Registered LL128 cooperative-atomic path is gfx1250-specific";
  }

  if (!validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kRequireSingleNode)) {
    GTEST_SKIP() << "Requires 2+ ranks on a single node";
  }

  MPIHelpers::MpiEnvGuard protoGuard("NCCL_PROTO", "LL128");
  MPIHelpers::MpiEnvGuard algoGuard("NCCL_ALGO", "Ring");
  MPIHelpers::MpiEnvGuard ll128ForceGuard("RCCL_LL128_FORCE_ENABLE", "1");
  MPIHelpers::MpiEnvGuard localRegGuard("NCCL_LOCAL_REGISTER", "1");

  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  int rank = 0, nRanks = 0;
  ncclCommUserRank(getActiveCommunicator(), &rank);
  ncclCommCount(getActiveCommunicator(), &nRanks);

  const size_t count = 8192;
  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  void* sendHandle = nullptr;
  void* recvHandle = nullptr;

  ASSERT_MPI_EQ(ncclSuccess, allocateDeviceBuffer(&sendBuf, count * sizeof(float)));
  ASSERT_MPI_EQ(ncclSuccess, allocateDeviceBuffer(&recvBuf, count * sizeof(float)));
  auto cleanup = makeScopeGuard([&]() {
    if (sendHandle) ncclCommDeregister(getActiveCommunicator(), sendHandle);
    if (recvHandle) ncclCommDeregister(getActiveCommunicator(), recvHandle);
    if (sendBuf) freeDeviceBuffer(sendBuf);
    if (recvBuf) freeDeviceBuffer(recvBuf);
  });

  ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommRegister(getActiveCommunicator(), sendBuf, count * sizeof(float), &sendHandle));
  ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommRegister(getActiveCommunicator(), recvBuf, count * sizeof(float), &recvHandle));

  initSendBuffer(static_cast<float*>(sendBuf), count, rank);

  ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, getActiveCommunicator(),
                                getActiveStream()));
  ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

  ASSERT_TRUE(verifyAllReduceResult(static_cast<float*>(recvBuf), count, nRanks));
  ASSERT_TRUE(logsContainForcedProto(readAllLogs(), "LL128"))
      << "Expected NCCL_TUNING AllReduce line with forced LL128 protocol on gfx1250";
}

} // namespace RcclUnitTesting

#endif // MPI_TESTS_ENABLED
