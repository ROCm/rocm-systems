/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ROCRTST_SUITES_FUNCTIONAL_QUEUE_MODIFY_H_
#define ROCRTST_SUITES_FUNCTIONAL_QUEUE_MODIFY_H_

#include "common/base_rocr.h"
#include "suites/test_common/test_base.h"

// Exercises the queue-modification path via the public APIs that reconfigure a
// live queue without destroy/recreate: hsa_amd_queue_cu_set_mask /
// hsa_amd_queue_cu_get_mask and hsa_amd_queue_set_priority.
//
// In DRM user-queue mode these drive DrmDriver::ModifyQueue ->
// amdgpu_modify_userqueue (AMDGPU_USERQ_OP_MODIFY). A successful
// hsa_amd_queue_cu_set_mask/set_priority in DRM mode therefore proves the
// modify ioctl round-trips end to end; the get_mask read-back confirms the
// modification took effect. Runs in both KFD and DRM modes, so it guards the
// DRM ModifyQueue path against a KFD-passes / DRM-fails divergence.
class QueueModifyTest : public TestBase {
 public:
  QueueModifyTest(void);
  virtual ~QueueModifyTest(void);

  virtual void SetUp(void);
  virtual void Run(void);
  virtual void DisplayTestInfo(void);
  virtual void DisplayResults(void) const;
  virtual void Close(void);

  void TestQueueModify(void);
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_QUEUE_MODIFY_H_
