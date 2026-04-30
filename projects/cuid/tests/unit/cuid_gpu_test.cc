/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "src/cuid_gpu.h"

#include <gtest/gtest.h>
#include <string>

// Regression test for the render_node trim behavior in
// CuidGpu::discover_single. Paths produced by the GIM enumeration path are of
// the form "/sys/bus/pci/devices/<bdf>" and must be preserved verbatim,
// otherwise every GIM-only GPU collapses to the parent directory and the CUID
// file ends up with an ambiguous device_node.
TEST(CuidGpuRenderNodeTest, PciDevicePathIsPreserved) {
  amdcuid_gpu_info info{};
  const std::string pci_path = "/sys/bus/pci/devices/0000:65:00.0";
  ASSERT_EQ(CuidGpu::discover_single(&info, pci_path),
            AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(info.render_node, pci_path)
      << "GIM-style PCI device paths must not be trimmed";
}

// Paths produced by /sys/class/drm enumeration end in "/device" and the
// trailing component must be stripped so that downstream consumers can
// re-append "/device/<attribute>" themselves.
TEST(CuidGpuRenderNodeTest, DrmDevicePathStripsTrailingDevice) {
  amdcuid_gpu_info info{};
  // Use a card name unlikely to collide with a real DRM card on the test
  // host so the sysfs reads inside discover_single fail cleanly. The
  // render_node trim logic still runs and is what we want to exercise.
  const std::string drm_path = "/sys/class/drm/card999/device";
  ASSERT_EQ(CuidGpu::discover_single(&info, drm_path),
            AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(info.render_node, "/sys/class/drm/card999")
      << "DRM enumeration paths must have the trailing /device stripped";
}

// A path that does not end in "/device" and is not a DRM card path must be
// returned verbatim. This guards against a future regression where a partial
// suffix match (e.g. "...evice") accidentally matches.
TEST(CuidGpuRenderNodeTest, ArbitraryPathPreserved) {
  amdcuid_gpu_info info{};
  const std::string path = "/some/custom/path";
  ASSERT_EQ(CuidGpu::discover_single(&info, path), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(info.render_node, path);
}
