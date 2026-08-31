// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_TEST_UNIT_CUID_GPU_TEST_H_
#define CUID_TEST_UNIT_CUID_GPU_TEST_H_

#include "test_base.h"

// Regression coverage for CuidGpu::discover_single render_node trimming across
// the GIM PCI-device, DRM /sys/class/drm, and arbitrary path forms.
class TestCuidGpuRenderNode : public TestBase {
 public:
  TestCuidGpuRenderNode();
  void SetUp() override;
  void Run() override;
};

#endif  // CUID_TEST_UNIT_CUID_GPU_TEST_H_
