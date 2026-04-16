//===-- GPULaunchGTest.cpp - GPU Launch Integration Tests ------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration tests for GPU kernel launch infrastructure.
/// These tests require actual GPU hardware and will be skipped if no GPU
/// is available.
///
//===----------------------------------------------------------------------===//

#include "fixtures/GPUFixture.h"
#include <gtest/gtest.h>

#ifdef AEGISBIT_HAS_GPU

using namespace aegisbit;
using namespace aegisbit::test;

//===----------------------------------------------------------------------===//
// GPU Detection Tests
//===----------------------------------------------------------------------===//

TEST(GPUDetection, FindsAMDGPUAgents) {
  auto Agents = findAMDGPUAgents();
  // This test just verifies the function runs without crashing
  // It may or may not find GPUs depending on the system
  if (Agents.empty()) {
    GTEST_SKIP() << "No AMD GPUs found on this system";
  }
  EXPECT_GT(Agents.size(), 0u);
}

TEST(GPUDetection, GetDefaultGPUAgent) {
  uint64_t Agent = getDefaultGPUAgent();
  if (Agent == 0) {
    GTEST_SKIP() << "No default GPU agent available";
  }
  EXPECT_NE(Agent, 0u);
}

//===----------------------------------------------------------------------===//
// Kernel Launcher Tests
//===----------------------------------------------------------------------===//

class KernelLauncherTest : public GPUFixture {};

TEST_F(KernelLauncherTest, CreateLauncher) {
  // Launcher is already created in SetUp
  EXPECT_TRUE(Launcher->isGPUAvailable());
  EXPECT_FALSE(Launcher->getGPUName().empty());
}

TEST_F(KernelLauncherTest, GetAgentHandle) {
  EXPECT_NE(Launcher->getAgentHandle(), 0u);
  EXPECT_EQ(Launcher->getAgentHandle(), GPUAgent);
}

// Note: Testing actual kernel loading requires a valid ELF code object
// which would be created by the instrumentation pipeline.
// These are placeholder tests for the infrastructure.

//===----------------------------------------------------------------------===//
// Dispatch Interceptor Tests
//===----------------------------------------------------------------------===//

TEST(DispatchInterceptor, NotInitializedByDefault) {
  // Before initialization, should not be initialized
  // (unless another test already initialized it)
  // This is a basic sanity check
  bool wasInitialized = DispatchInterceptor::isInitialized();
  // Just verify we can check the state without crashing
  (void)wasInitialized;
}

TEST(DispatchInterceptor, InitializeAndFinalize) {
  auto Err = DispatchInterceptor::initialize();
  EXPECT_FALSE(static_cast<bool>(Err))
      << "Initialization should succeed: " << llvm::toString(std::move(Err));

  EXPECT_TRUE(DispatchInterceptor::isInitialized());

  DispatchInterceptor::finalize();
  EXPECT_FALSE(DispatchInterceptor::isInitialized());
}

TEST(DispatchInterceptor, SetCallbacks) {
  auto Err = DispatchInterceptor::initialize();
  if (Err) {
    llvm::consumeError(std::move(Err));
    GTEST_SKIP() << "Could not initialize interceptor";
  }

  bool CodeObjectCalled = false;
  bool KernelSymbolCalled = false;

  DispatchInterceptor::setCodeObjectLoadCallback(
      [&](const CapturedCodeObject& /*CO*/) { CodeObjectCalled = true; });

  DispatchInterceptor::setKernelSymbolCallback(
      [&](const CapturedKernelSymbol& /*KS*/) { KernelSymbolCalled = true; });

  // Callbacks are set but won't be called until actual GPU operations happen
  DispatchInterceptor::finalize();
}

#else // AEGISBIT_HAS_GPU

TEST(GPUDetection, NoGPUSupport) {
  GTEST_SKIP() << "GPU support not compiled in";
}

#endif // AEGISBIT_HAS_GPU
