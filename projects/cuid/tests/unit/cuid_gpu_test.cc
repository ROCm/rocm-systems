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

#include "unit/cuid_gpu_test.h"

#include <gtest/gtest.h>

#include <string>

#include "src/cuid_gpu.h"

TestCuidGpuRenderNode::TestCuidGpuRenderNode() {
  SetTitle("CuidGpu Render Node");
  SetDescription(
      "Verify CuidGpu::normalize_render_node preserves GIM PCI-device paths, "
      "strips the trailing /device from DRM paths, and returns arbitrary "
      "paths verbatim.");
}

void TestCuidGpuRenderNode::SetUp() {}

void TestCuidGpuRenderNode::Run() {
  // GIM enumeration passes "/sys/bus/pci/devices/<bdf>" and must be kept
  // verbatim; trimming would collapse every GIM-only GPU to the parent dir.
  {
    const std::string pci_path = "/sys/bus/pci/devices/0000:65:00.0";
    EXPECT_EQ(CuidGpu::normalize_render_node(pci_path), pci_path)
        << "GIM-style PCI device paths must not be trimmed";
  }

  // DRM enumeration passes "/sys/class/drm/<card>/device"; the trailing
  // "/device" must be stripped. card999 is unlikely to exist, so no renderD
  // node is resolved and the trimmed card path is returned unchanged.
  {
    const std::string drm_path = "/sys/class/drm/card999/device";
    EXPECT_EQ(CuidGpu::normalize_render_node(drm_path), "/sys/class/drm/card999")
        << "DRM enumeration paths must have the trailing /device stripped";
  }

  // A path that is neither form must be returned verbatim.
  {
    const std::string path = "/some/custom/path";
    EXPECT_EQ(CuidGpu::normalize_render_node(path), path);
  }
}
