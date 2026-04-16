//===-- GPUFixture.h - GPU Integration Test Fixture ------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Test fixture for GPU integration tests that require actual GPU hardware.
///
/// Provides:
/// - GPU availability detection
/// - Automatic test skipping when no GPU is present
/// - Helper methods for creating test kernels
/// - Trace data validation utilities
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_TEST_GPU_FIXTURE_H
#define AEGISBIT_TEST_GPU_FIXTURE_H

#include <gtest/gtest.h>
#include <set>
#include <string>
#include <vector>

#ifdef AEGISBIT_HAS_GPU
#include "aegisbit/DispatchInterceptor.h"
#include "aegisbit/KernelLauncher.h"
#endif

#include "aegisbit/CodeObjectHandler.h"
#include "aegisbit/Disassembler.h"
#include "aegisbit/Types.h"

namespace aegisbit {
namespace test {

/// Base fixture for GPU integration tests.
///
/// Tests using this fixture will be skipped if no GPU is available.
/// This allows the test suite to run on systems without GPU hardware.
class GPUFixture : public ::testing::Test {
protected:
  void SetUp() override {
#ifdef AEGISBIT_HAS_GPU
    // Try to find a GPU
    GPUAgent = getDefaultGPUAgent();
    HasGPU = (GPUAgent != 0);

    if (HasGPU) {
      auto LauncherOrErr = KernelLauncher::create(GPUAgent);
      if (LauncherOrErr) {
        Launcher = std::move(*LauncherOrErr);
        GPUName = Launcher->getGPUName();
      } else {
        llvm::consumeError(LauncherOrErr.takeError());
        HasGPU = false;
      }
    }
#else
    HasGPU = false;
#endif

    if (!HasGPU) {
      GTEST_SKIP() << "No GPU available, skipping GPU integration test";
    }
  }

  void TearDown() override {
    Launcher.reset();
  }

  /// Check if GPU is available.
  bool hasGPU() const { return HasGPU; }

  /// Get the GPU architecture name (e.g., "gfx942").
  const std::string& getGPUName() const { return GPUName; }

  /// Create a disassembler for the detected GPU.
  llvm::Expected<std::unique_ptr<Disassembler>> createDisassembler() {
    return Disassembler::create("amdgcn-amd-amdhsa", GPUName,
                                 "+wavefrontsize64");
  }

#ifdef AEGISBIT_HAS_GPU
  /// Get the kernel launcher.
  KernelLauncher* getLauncher() { return Launcher.get(); }
#endif

  /// Validate that trace data contains expected number of records.
  bool validateTraceRecordCount(const std::vector<uint8_t>& TraceData,
                                 size_t RecordSize,
                                 size_t ExpectedCount) {
    if (TraceData.size() < RecordSize * ExpectedCount) {
      return false;
    }
    return true;
  }

  /// Extract kernel ID from a trace record.
  static uint32_t extractKernelID(const uint8_t* Record) {
    // TraceArgs layout: BufferPtr(8) + BufferSize(8) + WriteOffsetPtr(8) + KernelID(4)
    // But trace records are: wavefront_id(4) + data(4+)
    return static_cast<uint32_t>(Record[0]) |
           (static_cast<uint32_t>(Record[1]) << 8) |
           (static_cast<uint32_t>(Record[2]) << 16) |
           (static_cast<uint32_t>(Record[3]) << 24);
  }

protected:
  bool HasGPU = false;
  uint64_t GPUAgent = 0;
  std::string GPUName;
#ifdef AEGISBIT_HAS_GPU
  std::unique_ptr<KernelLauncher> Launcher;
#endif
};

/// Fixture for trace validation tests.
class TraceValidationFixture : public GPUFixture {
protected:
  /// Validate that all wavefront IDs in trace are valid.
  bool validateWavefrontIDs(const std::vector<uint8_t>& TraceData,
                             size_t RecordSize,
                             uint32_t MaxWavefrontID) {
    for (size_t i = 0; i + RecordSize <= TraceData.size(); i += RecordSize) {
      uint32_t WfID = extractKernelID(&TraceData[i]);
      if (WfID > MaxWavefrontID) {
        return false;
      }
    }
    return true;
  }

  /// Count unique wavefront IDs in trace.
  size_t countUniqueWavefronts(const std::vector<uint8_t>& TraceData,
                                size_t RecordSize) {
    std::set<uint32_t> Seen;
    for (size_t i = 0; i + RecordSize <= TraceData.size(); i += RecordSize) {
      uint32_t WfID = extractKernelID(&TraceData[i]);
      Seen.insert(WfID);
    }
    return Seen.size();
  }

};

} // namespace test
} // namespace aegisbit

#endif // AEGISBIT_TEST_GPU_FIXTURE_H
