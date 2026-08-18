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

#include "telemetry_read.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

// Finds the first AMD NIC handle across all sockets; empty if none present.
static std::vector<amdsmi_processor_handle> nic_handles() {
  std::vector<amdsmi_processor_handle> nics;
  uint32_t socket_count = 0;
  if (amdsmi_get_socket_handles(&socket_count, nullptr) != AMDSMI_STATUS_SUCCESS) {
    return nics;
  }
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  if (amdsmi_get_socket_handles(&socket_count, sockets.data()) != AMDSMI_STATUS_SUCCESS) {
    return nics;
  }
  for (auto socket : sockets) {
    uint32_t count = 0;
    if (amdsmi_get_processor_handles_by_type(socket, AMDSMI_PROCESSOR_TYPE_AMD_NIC, nullptr,
                                             &count) != AMDSMI_STATUS_SUCCESS ||
        count == 0) {
      continue;
    }
    std::vector<amdsmi_processor_handle> handles(count);
    if (amdsmi_get_processor_handles_by_type(socket, AMDSMI_PROCESSOR_TYPE_AMD_NIC, handles.data(),
                                             &count) == AMDSMI_STATUS_SUCCESS) {
      nics.insert(nics.end(), handles.begin(), handles.end());
    }
  }
  return nics;
}

TestNicTelemetryRead::TestNicTelemetryRead() : TestBase() {
  set_title("AMDSMI NIC Telemetry Read Test");
  set_description(
      "This test verifies the public NIC telemetry / firmware bridge: "
      "amdsmi_get_nic_telemetry() and amdsmi_get_nic_fw_info() return sane, "
      "bounded values and reject null output pointers. It references "
      "amdsmi_nic_telemetry_t by field so a field dropped from the getter's "
      "copy bridge surfaces here.");
}

TestNicTelemetryRead::~TestNicTelemetryRead(void) {}

// NIC telemetry needs the NIC backend, not the default GPU init.
void TestNicTelemetryRead::SetUp(void) { TestBase::SetUp(AMDSMI_INIT_AMD_NICS); }

void TestNicTelemetryRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestNicTelemetryRead::DisplayResults(void) const { TestBase::DisplayResults(); }

void TestNicTelemetryRead::Close() { TestBase::Close(); }

void TestNicTelemetryRead::Run(void) {
  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) { std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl; }
    return;
  }

  std::vector<amdsmi_processor_handle> nics = nic_handles();
  if (nics.empty()) {
    IF_VERB(STANDARD) { std::cout << "\t**No AMD NIC present. Skipping.**" << std::endl; }
    return;
  }
  amdsmi_processor_handle nic = nics[0];

  // ── amdsmi_get_nic_telemetry: succeeds and returns sane values ──────────
  {
    amdsmi_nic_telemetry_t telem;
    ASSERT_EQ(amdsmi_get_nic_telemetry(nic, &telem), AMDSMI_STATUS_SUCCESS);

    // Health state is one of the defined enum values.
    EXPECT_LE(telem.health.state, AMDSMI_NIC_HEALTH_UNSUPPORTED);

    // reporter is NUL-terminated within its fixed buffer.
    EXPECT_LT(strnlen(telem.health.reporter, sizeof(telem.health.reporter)),
              sizeof(telem.health.reporter));

    // Each temperature is either a plausible reading or the unsupported sentinel.
    for (uint16_t t : {telem.temperature.asic_temp_c, telem.temperature.transceiver_temp_c,
                       telem.temperature.board_temp_c}) {
      EXPECT_TRUE(t == UINT16_MAX || t <= 150) << "implausible temperature: " << t;
    }

    // Bind every remaining public field by name so a field dropped or renamed
    // from the copy bridge fails to compile here (companion to the sizeof guard).
    EXPECT_TRUE(telem.port_split.splittable == UINT8_MAX || telem.port_split.splittable <= 1)
        << "implausible splittable: " << static_cast<unsigned>(telem.port_split.splittable);
    const uint8_t split_count = telem.port_split.split_count;
    const uint32_t error_count = telem.health.error_count;  // any value is valid
    (void)split_count;
    (void)error_count;
  }

  // ── amdsmi_get_nic_fw_info: succeeds and is bounded ─────────────────────
  {
    amdsmi_nic_fw_info_t fw;
    ASSERT_EQ(amdsmi_get_nic_fw_info(nic, &fw), AMDSMI_STATUS_SUCCESS);
    EXPECT_LE(fw.num_fw, static_cast<uint32_t>(AMDSMI_MAX_NIC_FW));
  }

  // ── null output pointers are rejected ───────────────────────────────────
  {
    EXPECT_EQ(amdsmi_get_nic_telemetry(nic, nullptr), AMDSMI_STATUS_INVAL);
    EXPECT_EQ(amdsmi_get_nic_fw_info(nic, nullptr), AMDSMI_STATUS_INVAL);
  }
}
