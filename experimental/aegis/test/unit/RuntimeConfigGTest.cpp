//===-- RuntimeConfigGTest.cpp - RuntimeConfig Tests -----------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for RuntimeConfig environment variable parsing.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/RuntimeConfig.h"
#include <gtest/gtest.h>
#include <cstdlib>

using namespace aegisbit;

class RuntimeConfigTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Clear environment variables before each test
    unsetenv("AEGISBIT_ENABLED");
    unsetenv("AEGISBIT_MODE");
    unsetenv("AEGISBIT_KERNELS");
    unsetenv("AEGISBIT_OUTPUT");
    unsetenv("AEGISBIT_BUFFER_MB");
    unsetenv("AEGISBIT_LOG");
    unsetenv("AEGISBIT_REPLAY");
    unsetenv("AEGISBIT_REPLAY_MAX");
  }

  void TearDown() override {
    // Clean up after each test
    unsetenv("AEGISBIT_ENABLED");
    unsetenv("AEGISBIT_MODE");
    unsetenv("AEGISBIT_KERNELS");
    unsetenv("AEGISBIT_OUTPUT");
    unsetenv("AEGISBIT_BUFFER_MB");
    unsetenv("AEGISBIT_LOG");
    unsetenv("AEGISBIT_REPLAY");
    unsetenv("AEGISBIT_REPLAY_MAX");
  }
};

//===----------------------------------------------------------------------===//
// Default Values Tests
//===----------------------------------------------------------------------===//

TEST_F(RuntimeConfigTest, DefaultValues) {
  RuntimeConfig::initialize();
  const RuntimeConfig& Cfg = RuntimeConfig::getInstance();

  EXPECT_TRUE(Cfg.Enabled);
  EXPECT_EQ(Cfg.Mode, InstrumentationMode::MEMORY_ONLY);
  EXPECT_EQ(Cfg.BufferSizeBytes, 64 * 1024 * 1024);
  EXPECT_FALSE(Cfg.LogEnabled);
  EXPECT_EQ(Cfg.OutputDir.string(), "./aegisbit_traces/");
  EXPECT_EQ(Cfg.KernelPatterns.size(), 1u);
  EXPECT_EQ(Cfg.KernelPatterns[0], "*");
}

//===----------------------------------------------------------------------===//
// AEGISBIT_ENABLED Tests
//===----------------------------------------------------------------------===//

TEST_F(RuntimeConfigTest, EnabledDefault) {
  RuntimeConfig::initialize();
  EXPECT_TRUE(RuntimeConfig::getInstance().Enabled);
}

TEST_F(RuntimeConfigTest, EnabledExplicit1) {
  setenv("AEGISBIT_ENABLED", "1", 1);
  RuntimeConfig::initialize();
  EXPECT_TRUE(RuntimeConfig::getInstance().Enabled);
}

TEST_F(RuntimeConfigTest, EnabledExplicit0) {
  setenv("AEGISBIT_ENABLED", "0", 1);
  RuntimeConfig::initialize();
  EXPECT_FALSE(RuntimeConfig::getInstance().Enabled);
}

TEST_F(RuntimeConfigTest, EnabledFalseString) {
  setenv("AEGISBIT_ENABLED", "false", 1);
  RuntimeConfig::initialize();
  EXPECT_FALSE(RuntimeConfig::getInstance().Enabled);
}

//===----------------------------------------------------------------------===//
// AEGISBIT_MODE Tests
//===----------------------------------------------------------------------===//

TEST_F(RuntimeConfigTest, ModeDefault) {
  RuntimeConfig::initialize();
  EXPECT_EQ(RuntimeConfig::getInstance().Mode, InstrumentationMode::MEMORY_ONLY);
}

TEST_F(RuntimeConfigTest, ModeMemoryOnly) {
  setenv("AEGISBIT_MODE", "MEMORY_ONLY", 1);
  RuntimeConfig::initialize();
  EXPECT_EQ(RuntimeConfig::getInstance().Mode, InstrumentationMode::MEMORY_ONLY);
}

TEST_F(RuntimeConfigTest, ModeAlwaysMemoryOnly) {
  setenv("AEGISBIT_MODE", "FULL", 1);
  RuntimeConfig::initialize();
  EXPECT_EQ(RuntimeConfig::getInstance().Mode, InstrumentationMode::MEMORY_ONLY);
}

//===----------------------------------------------------------------------===//
// AEGISBIT_KERNELS Tests
//===----------------------------------------------------------------------===//

TEST_F(RuntimeConfigTest, KernelsDefault) {
  RuntimeConfig::initialize();
  const RuntimeConfig& Cfg = RuntimeConfig::getInstance();
  ASSERT_EQ(Cfg.KernelPatterns.size(), 1u);
  EXPECT_EQ(Cfg.KernelPatterns[0], "*");
}

TEST_F(RuntimeConfigTest, KernelsSingle) {
  setenv("AEGISBIT_KERNELS", "myKernel", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig& Cfg = RuntimeConfig::getInstance();
  ASSERT_EQ(Cfg.KernelPatterns.size(), 1u);
  EXPECT_EQ(Cfg.KernelPatterns[0], "myKernel");
}

TEST_F(RuntimeConfigTest, KernelsMultiple) {
  setenv("AEGISBIT_KERNELS", "kernel1,kernel2,kernel3", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig& Cfg = RuntimeConfig::getInstance();
  ASSERT_EQ(Cfg.KernelPatterns.size(), 3u);
  EXPECT_EQ(Cfg.KernelPatterns[0], "kernel1");
  EXPECT_EQ(Cfg.KernelPatterns[1], "kernel2");
  EXPECT_EQ(Cfg.KernelPatterns[2], "kernel3");
}

TEST_F(RuntimeConfigTest, KernelsWithSpaces) {
  setenv("AEGISBIT_KERNELS", " kernel1 , kernel2 ", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig& Cfg = RuntimeConfig::getInstance();
  ASSERT_EQ(Cfg.KernelPatterns.size(), 2u);
  EXPECT_EQ(Cfg.KernelPatterns[0], "kernel1");
  EXPECT_EQ(Cfg.KernelPatterns[1], "kernel2");
}

//===----------------------------------------------------------------------===//
// AEGISBIT_BUFFER_MB Tests
//===----------------------------------------------------------------------===//

TEST_F(RuntimeConfigTest, BufferDefault) {
  RuntimeConfig::initialize();
  EXPECT_EQ(RuntimeConfig::getInstance().BufferSizeBytes, 64 * 1024 * 1024);
  EXPECT_EQ(RuntimeConfig::getInstance().getBufferSizeMB(), 64u);
}

TEST_F(RuntimeConfigTest, BufferCustom) {
  setenv("AEGISBIT_BUFFER_MB", "128", 1);
  RuntimeConfig::initialize();
  EXPECT_EQ(RuntimeConfig::getInstance().BufferSizeBytes, 128 * 1024 * 1024);
  EXPECT_EQ(RuntimeConfig::getInstance().getBufferSizeMB(), 128u);
}

TEST_F(RuntimeConfigTest, BufferInvalid) {
  setenv("AEGISBIT_BUFFER_MB", "invalid", 1);
  RuntimeConfig::initialize();
  // Falls back to default
  EXPECT_EQ(RuntimeConfig::getInstance().BufferSizeBytes, 64 * 1024 * 1024);
}

TEST_F(RuntimeConfigTest, BufferTooLarge) {
  setenv("AEGISBIT_BUFFER_MB", "10000", 1);  // > 4096 MB cap
  RuntimeConfig::initialize();
  // Falls back to default
  EXPECT_EQ(RuntimeConfig::getInstance().BufferSizeBytes, 64 * 1024 * 1024);
}

//===----------------------------------------------------------------------===//
// AEGISBIT_OUTPUT Tests
//===----------------------------------------------------------------------===//

TEST_F(RuntimeConfigTest, OutputDefault) {
  RuntimeConfig::initialize();
  EXPECT_EQ(RuntimeConfig::getInstance().OutputDir.string(), "./aegisbit_traces/");
}

TEST_F(RuntimeConfigTest, OutputCustom) {
  setenv("AEGISBIT_OUTPUT", "/tmp/my_traces", 1);
  RuntimeConfig::initialize();
  EXPECT_EQ(RuntimeConfig::getInstance().OutputDir.string(), "/tmp/my_traces");
}

//===----------------------------------------------------------------------===//
// shouldTraceKernel Tests
//===----------------------------------------------------------------------===//

TEST_F(RuntimeConfigTest, ShouldTraceWildcard) {
  RuntimeConfig::initialize();
  const RuntimeConfig& Cfg = RuntimeConfig::getInstance();
  EXPECT_TRUE(Cfg.shouldTraceKernel("anyKernel"));
  EXPECT_TRUE(Cfg.shouldTraceKernel("anotherKernel"));
}

TEST_F(RuntimeConfigTest, ShouldTraceExactMatch) {
  setenv("AEGISBIT_KERNELS", "myKernel", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig& Cfg = RuntimeConfig::getInstance();
  EXPECT_TRUE(Cfg.shouldTraceKernel("myKernel"));
  EXPECT_FALSE(Cfg.shouldTraceKernel("otherKernel"));
}

TEST_F(RuntimeConfigTest, ShouldTracePrefixPattern) {
  setenv("AEGISBIT_KERNELS", "my*", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig& Cfg = RuntimeConfig::getInstance();
  EXPECT_TRUE(Cfg.shouldTraceKernel("myKernel"));
  EXPECT_TRUE(Cfg.shouldTraceKernel("myOtherKernel"));
  EXPECT_FALSE(Cfg.shouldTraceKernel("otherKernel"));
}

TEST_F(RuntimeConfigTest, ShouldTraceSuffixPattern) {
  setenv("AEGISBIT_KERNELS", "*Kernel", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig& Cfg = RuntimeConfig::getInstance();
  EXPECT_TRUE(Cfg.shouldTraceKernel("myKernel"));
  EXPECT_TRUE(Cfg.shouldTraceKernel("otherKernel"));
  EXPECT_FALSE(Cfg.shouldTraceKernel("myFunction"));
}

TEST_F(RuntimeConfigTest, ShouldTraceContainsPattern) {
  setenv("AEGISBIT_KERNELS", "*add*", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig& Cfg = RuntimeConfig::getInstance();
  EXPECT_TRUE(Cfg.shouldTraceKernel("vectorAdd"));
  EXPECT_TRUE(Cfg.shouldTraceKernel("addVectors"));
  EXPECT_TRUE(Cfg.shouldTraceKernel("myAddKernel"));
  EXPECT_FALSE(Cfg.shouldTraceKernel("multiply"));
}

TEST_F(RuntimeConfigTest, ShouldTraceMultiplePatterns) {
  setenv("AEGISBIT_KERNELS", "kernel1,kernel2,*add*", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig& Cfg = RuntimeConfig::getInstance();
  EXPECT_TRUE(Cfg.shouldTraceKernel("kernel1"));
  EXPECT_TRUE(Cfg.shouldTraceKernel("kernel2"));
  EXPECT_TRUE(Cfg.shouldTraceKernel("vectorAdd"));
  EXPECT_FALSE(Cfg.shouldTraceKernel("kernel3"));
}

TEST_F(RuntimeConfigTest, ShouldTraceDisabled) {
  setenv("AEGISBIT_ENABLED", "0", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig& Cfg = RuntimeConfig::getInstance();
  EXPECT_FALSE(Cfg.shouldTraceKernel("anyKernel"));
}

//===----------------------------------------------------------------------===//
// AEGISBIT_REPLAY Tests (instrumentation replay env knob, Phase 5)
//
// Matrix:
//   unset / "0" / "false"  → ReplayVariants=0, ReplayAuto=false (legacy)
//   "auto" / "AUTO"        → ReplayAuto=true,  ReplayVariants=0  (plateau-capped)
//   positive integer       → ReplayVariants=N, ReplayAuto=false (fixed count)
//   non-numeric truthy     → both default (ignored — avoids silently mis-parsing)
//===----------------------------------------------------------------------===//

TEST_F(RuntimeConfigTest, ReplayDefault) {
  RuntimeConfig::initialize();
  const RuntimeConfig &Cfg = RuntimeConfig::getInstance();
  EXPECT_EQ(Cfg.Debug.ReplayVariants, 0u);
  EXPECT_FALSE(Cfg.Debug.ReplayAuto);
}

TEST_F(RuntimeConfigTest, ReplayOff0) {
  setenv("AEGISBIT_REPLAY", "0", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig &Cfg = RuntimeConfig::getInstance();
  EXPECT_EQ(Cfg.Debug.ReplayVariants, 0u);
  EXPECT_FALSE(Cfg.Debug.ReplayAuto);
}

TEST_F(RuntimeConfigTest, ReplayOffFalse) {
  setenv("AEGISBIT_REPLAY", "false", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig &Cfg = RuntimeConfig::getInstance();
  EXPECT_EQ(Cfg.Debug.ReplayVariants, 0u);
  EXPECT_FALSE(Cfg.Debug.ReplayAuto);
}

TEST_F(RuntimeConfigTest, ReplayAutoLowercase) {
  setenv("AEGISBIT_REPLAY", "auto", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig &Cfg = RuntimeConfig::getInstance();
  EXPECT_TRUE(Cfg.Debug.ReplayAuto);
  EXPECT_EQ(Cfg.Debug.ReplayVariants, 0u);
}

TEST_F(RuntimeConfigTest, ReplayAutoUppercase) {
  setenv("AEGISBIT_REPLAY", "AUTO", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig &Cfg = RuntimeConfig::getInstance();
  EXPECT_TRUE(Cfg.Debug.ReplayAuto);
}

TEST_F(RuntimeConfigTest, ReplayFixedCount) {
  setenv("AEGISBIT_REPLAY", "3", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig &Cfg = RuntimeConfig::getInstance();
  EXPECT_EQ(Cfg.Debug.ReplayVariants, 3u);
  EXPECT_FALSE(Cfg.Debug.ReplayAuto);
}

TEST_F(RuntimeConfigTest, ReplayGarbageString) {
  // Non-numeric, non-"auto" truthy value: parser ignores (keeps defaults).
  setenv("AEGISBIT_REPLAY", "banana", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig &Cfg = RuntimeConfig::getInstance();
  EXPECT_EQ(Cfg.Debug.ReplayVariants, 0u);
  EXPECT_FALSE(Cfg.Debug.ReplayAuto);
}

//===----------------------------------------------------------------------===//
// AEGISBIT_REPLAY_MAX Tests (auto-mode safety cap, post-cap-lift refactor)
//
// Matrix:
//   unset            → ReplayMax = 32 (default)
//   positive integer → ReplayMax = N
//   "0"              → ReplayMax = 32 (bogus; fall back to default)
//   garbage string   → ReplayMax = 32 (getEnvU32 fallback)
//===----------------------------------------------------------------------===//

TEST_F(RuntimeConfigTest, ReplayMaxDefault) {
  RuntimeConfig::initialize();
  const RuntimeConfig &Cfg = RuntimeConfig::getInstance();
  EXPECT_EQ(Cfg.Debug.ReplayMax, 32u);
}

TEST_F(RuntimeConfigTest, ReplayMaxExplicit) {
  setenv("AEGISBIT_REPLAY_MAX", "16", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig &Cfg = RuntimeConfig::getInstance();
  EXPECT_EQ(Cfg.Debug.ReplayMax, 16u);
}

TEST_F(RuntimeConfigTest, ReplayMaxZeroFallsBack) {
  // A 0 max would starve even the first variant; fall back to default.
  setenv("AEGISBIT_REPLAY_MAX", "0", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig &Cfg = RuntimeConfig::getInstance();
  EXPECT_EQ(Cfg.Debug.ReplayMax, 32u);
}

TEST_F(RuntimeConfigTest, ReplayMaxGarbageFallsBack) {
  setenv("AEGISBIT_REPLAY_MAX", "banana", 1);
  RuntimeConfig::initialize();
  const RuntimeConfig &Cfg = RuntimeConfig::getInstance();
  EXPECT_EQ(Cfg.Debug.ReplayMax, 32u);
}
