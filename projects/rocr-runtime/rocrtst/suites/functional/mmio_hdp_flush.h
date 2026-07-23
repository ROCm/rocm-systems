/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ROCRTST_SUITES_FUNCTIONAL_MMIO_HDP_FLUSH_H_
#define ROCRTST_SUITES_FUNCTIONAL_MMIO_HDP_FLUSH_H_

#include "common/base_rocr.h"
#include "suites/test_common/test_base.h"

// Verifies the MMIO_REMAP allocation path by checking the HDP flush
// registers it backs. The GPU's MMIO_REMAP heap is mapped at init time
// (libhsakmt map_mmio) and surfaced via HSA_AMD_AGENT_INFO_HDP_FLUSH
// (amd_gpu_agent.cpp). If MMIO_REMAP allocation/mapping does not complete
// (observed in DRM user-queue mode, where the BO allocation reports
// success but the page is never usably mapped), the HDP flush pointer
// comes back NULL. This test runs in both KFD and DRM modes and therefore
// guards against a KFD-passes / DRM-fails divergence.
class MMIOHdpFlushTest : public TestBase {
 public:
  MMIOHdpFlushTest(void);
  virtual ~MMIOHdpFlushTest(void);

  virtual void SetUp(void);
  virtual void Run(void);
  virtual void DisplayTestInfo(void);
  virtual void DisplayResults(void) const;
  virtual void Close(void);

  // Query HSA_AMD_AGENT_INFO_HDP_FLUSH, assert the pointers are non-null,
  // and perform the HDP flush write the runtime itself issues.
  void TestHdpFlushMapped(void);
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_MMIO_HDP_FLUSH_H_
