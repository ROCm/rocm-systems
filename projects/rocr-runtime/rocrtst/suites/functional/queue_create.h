/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_QUEUE_CREATE_H
#define ROCRTST_SUITES_FUNCTIONAL_QUEUE_CREATE_H

#include "suites/test_common/test_base.h"

class QueueCreateTest : public TestBase {
 public:
  explicit QueueCreateTest();
  virtual ~QueueCreateTest();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

  /// @brief Create a queue with system memory (default flags=0) and verify
  /// it can dispatch a kernel correctly.
  void SystemMemQueueTest();

  /// @brief Create a queue with HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF
  /// flag and verify kernel dispatch works correctly.
  void DeviceMemRingBufQueueTest();

  /// @brief Batch-create multiple queues with different flag combinations
  /// in a single hsa_amd_queue_create call.
  void BatchQueueCreateTest();

  /// @brief Create a unified SDMA queue descriptor and verify the returned
  /// hsa_queue_t can be destroyed through the standard queue API.
  void SdmaQueueCreateDestroyTest();

  /// @brief Validate descriptor argument checks for hsa_amd_queue_create.
  void InvalidArgsTest();

  /// @brief Query hsa_amd_queue_get_info attributes on a compute (AQL) queue
  /// and check that attributes that do not apply to it are rejected.
  void AqlQueueInfoTest();

  /// @brief Change the priority and the CU mask of a compute queue with
  /// hsa_amd_queue_set_priority and hsa_amd_queue_cu_set_mask, and check that
  /// the queue still dispatches correctly after each change.
  void AqlQueuePriorityAndCuMaskTest();

 private:
  void DispatchAndVerify(hsa_queue_t* queue, const char* test_label);
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_QUEUE_CREATE_H
