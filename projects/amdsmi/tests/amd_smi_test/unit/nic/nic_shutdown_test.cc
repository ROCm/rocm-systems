// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * The NIC debug-log sink points into the library. Once the library is shut down nothing may
 * still route NIC messages through it.
 */

#include <gtest/gtest.h>

#include <string>

#include "amd_smi/amdsmi.h"
#include "nic_unit_fixture.h"
#include "rocm_smi/rocm_smi.h"
#include "smi_nic_log.h"

namespace {

void discard_message(const std::string& /* message */) {}

TEST_F(NicUnit, ShutdownUninstallsTheNicLogSink) {
  amd::smi::nic::log::set_sink(&discard_message);
  ASSERT_EQ(amdsmi_init(AMDSMI_INIT_AMD_NICS), AMDSMI_STATUS_SUCCESS);

  ASSERT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);

  EXPECT_FALSE(amd::smi::nic::log::is_enabled());
}

// The GPU branch of the shutdown reports a failure only after the NIC branch has run.
TEST_F(NicUnit, ShutdownUninstallsTheNicLogSinkWhenTheGpuShutdownFails) {
  amd::smi::nic::log::set_sink(&discard_message);
  if (amdsmi_init(AMDSMI_INIT_AMD_GPUS | AMDSMI_INIT_AMD_NICS) != AMDSMI_STATUS_SUCCESS) {
    GTEST_SKIP() << "needs a host with a GPU";
  }
  // Release the library's own ROCm SMI reference, so the shutdown below finds it already closed.
  ASSERT_EQ(rsmi_shut_down(), RSMI_STATUS_SUCCESS);

  EXPECT_NE(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);

  EXPECT_FALSE(amd::smi::nic::log::is_enabled());
}

}  // namespace
