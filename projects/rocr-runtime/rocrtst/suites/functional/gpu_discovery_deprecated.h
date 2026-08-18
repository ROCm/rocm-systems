/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_GPU_DISCOVERY_DEPRECATED_H_
#define ROCRTST_SUITES_FUNCTIONAL_GPU_DISCOVERY_DEPRECATED_H_

#include "suites/test_common/test_base.h"
#include "hsa/hsa.h"

// Tests that HSA initialization succeeds even when deprecated or unsupported
// GPU devices (e.g. pre-Vega with DoorbellType != 2) are present in the system.
//
// Background: GpuAgent construction throws an exception for GPUs with deprecated
// doorbell types. If DiscoverGpu does not catch this exception, a single
// unsupported GPU will abort HSA initialization for ALL devices, including
// supported ones. This test verifies the graceful-skip behavior.
//
// See: amd_gpu_agent.cpp (doorbell type check)
//      amd_topology.cpp  (DiscoverGpu exception handler)
class GpuDiscoveryDeprecatedTest : public TestBase {
 public:
  GpuDiscoveryDeprecatedTest();
  virtual ~GpuDiscoveryDeprecatedTest();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_GPU_DISCOVERY_DEPRECATED_H_
