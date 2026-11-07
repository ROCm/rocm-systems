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
#include "smi_nic_log.h"

namespace {

void discard_message(const std::string& /* message */) {}

TEST_F(NicUnit, ShutdownUninstallsTheNicLogSink) {
  amd::smi::nic::log::set_sink(&discard_message);
  ASSERT_EQ(amdsmi_init(AMDSMI_INIT_AMD_NICS), AMDSMI_STATUS_SUCCESS);

  ASSERT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);

  EXPECT_FALSE(amd::smi::nic::log::is_enabled());
}

}  // namespace
