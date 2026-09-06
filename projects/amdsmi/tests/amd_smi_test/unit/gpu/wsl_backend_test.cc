// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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

// is_active() is false before any try_populate() call.
TEST(GpuUnit, InactiveByDefault) { EXPECT_FALSE(WSLGPUBackend::is_active()); }

// try_populate() on a machine without /dev/dxg returns NOT_SUPPORTED.
// Skipped on real WSL machines where /dev/dxg is present.
TEST(GpuUnit, TryPopulateWithoutDxg) {
  if (access("/dev/dxg", F_OK) == 0) {
    GTEST_SKIP() << "/dev/dxg present — skipped on WSL machines";
  }
  std::vector<AMDSmiSocket*> sockets;
  std::set<AMDSmiProcessor*> processors;
  amdsmi_status_t r = WSLGPUBackend::try_populate(sockets, processors);
  EXPECT_EQ(r, AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(sockets.empty());
  EXPECT_TRUE(processors.empty());
  EXPECT_FALSE(WSLGPUBackend::is_active());
}

#endif  // ENABLE_WSL_BACKEND
