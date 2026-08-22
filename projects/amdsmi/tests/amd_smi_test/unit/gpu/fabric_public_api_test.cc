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

// Argument rejection on the public fabric surface. The functional fabric suites cover this
// logic only behind a hardware-and-root gate, so it never runs in CI.

#include <gtest/gtest.h>

#include "amd_smi/amdsmi.h"

TEST(GpuUnit, FabricPublicApiGetRejectsNullArguments) {
  auto info = amdsmi_fabric_info_t{};
  info.fabric_version = AMDSMI_FABRIC_INFO_VERSION_2;

  EXPECT_EQ(amdsmi_get_gpu_fabric_info(nullptr, &info), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amdsmi_get_gpu_fabric_info(nullptr, nullptr), AMDSMI_STATUS_INVAL);
}
