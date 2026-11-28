/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_MEMORY_COPY_ENGINE_H_
#define ROCRTST_SUITES_FUNCTIONAL_MEMORY_COPY_ENGINE_H_

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "suites/test_common/test_base.h"

/// Functional tests for choosing the copy engine of an async copy:
/// hsa_amd_memory_get_preferred_copy_engine, hsa_amd_memory_copy_engine_status
/// and hsa_amd_memory_async_copy_on_engine, and for the async copy profiling
/// timestamps that those copies produce.
class MemoryCopyEngineTest : public TestBase {
 public:
  MemoryCopyEngineTest();
  virtual ~MemoryCopyEngineTest();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

  /// @brief Copy host to device, device to host and device to device on an
  /// engine picked from the engine status and preferred engine queries, and
  /// check the data. The same-device copy runs on a blit kernel and on SDMA.
  void CopyOnEngineTest();

  /// @brief Check that invalid engine ids, signals, agents and pointers are
  /// rejected by hsa_amd_memory_async_copy_on_engine and that nothing is
  /// copied by a rejected call.
  void CopyOnEngineNegativeTest();

  /// @brief Enable async copy profiling, run copies with and without an
  /// explicit engine, and check the start and end timestamps.
  void AsyncCopyProfilingTest();

  /// @brief Copy between two GPUs with hsa_amd_memory_async_copy and on an
  /// engine. Skipped when fewer than two GPUs can access each other's memory.
  void PeerCopyTest();
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_MEMORY_COPY_ENGINE_H_
