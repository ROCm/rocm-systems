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

// Unit tests for WSLGPUBackend activation and detection logic.
// No-op unless the WSL backend was compiled in (ENABLE_WSL_BACKEND=ON).

#include "config/amd_smi_config.h"

#if defined(ENABLE_WSL_BACKEND)

#include <gtest/gtest.h>
#include <unistd.h>

#include <set>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_processor.h"
#include "amd_smi/impl/amd_smi_socket.h"
#include "amd_smi/impl/amd_smi_wsl_device.h"

using amd::smi::AMDSmiProcessor;
using amd::smi::AMDSmiSocket;
using amd::smi::WSLGPUBackend;

// IsActive() is false before any TryPopulate() call.
TEST(GpuUnit,InactiveByDefault) { EXPECT_FALSE(WSLGPUBackend::IsActive()); }

// TryPopulate() on a machine without /dev/dxg returns NOT_SUPPORTED.
// Skipped on real WSL machines where /dev/dxg is present.
TEST(GpuUnit,TryPopulateWithoutDxg) {
  if (access("/dev/dxg", F_OK) == 0) {
    GTEST_SKIP() << "/dev/dxg present — skipped on WSL machines";
  }
  std::vector<AMDSmiSocket*> sockets;
  std::set<AMDSmiProcessor*> processors;
  amdsmi_status_t r = WSLGPUBackend::TryPopulate(sockets, processors);
  EXPECT_EQ(r, AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(sockets.empty());
  EXPECT_TRUE(processors.empty());
  EXPECT_FALSE(WSLGPUBackend::IsActive());
}

#endif  // ENABLE_WSL_BACKEND
