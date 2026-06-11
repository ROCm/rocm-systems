/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_GENERIC_BLIT_CODEOBJ_H_
#define ROCRTST_SUITES_FUNCTIONAL_GENERIC_BLIT_CODEOBJ_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

/// @brief Test class to validate generic blit code objects
///
/// This test validates that blit operations (memory copy and fill) work
/// correctly when using generic code objects (COV 6+) instead of
/// per-target code objects. Generic code objects reduce the number of
/// binaries needed while maintaining compatibility across all GPUs
/// within an architecture generation.
class GenericBlitCodeObjTest : public TestBase {
 public:
  GenericBlitCodeObjTest();
  virtual ~GenericBlitCodeObjTest();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

  /// @brief Test async memory copy with generic blit shaders
  /// Verifies hsa_amd_memory_async_copy works correctly
  void AsyncMemoryCopyTest(void);

  /// @brief Test memory fill operation with generic blit shaders
  /// Verifies hsa_amd_memory_fill works correctly
  void MemoryFillTest(void);

  /// @brief Test various copy sizes to exercise different blit kernel paths
  /// (aligned, misaligned, small, large)
  void VariousCopySizesTest(void);

 private:
  hsa_agent_t cpu_agent_;
  hsa_agent_t gpu_agent_;
  hsa_amd_memory_pool_t cpu_pool_;
  hsa_amd_memory_pool_t gpu_pool_;
  bool has_gpu_;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_GENERIC_BLIT_CODEOBJ_H_
