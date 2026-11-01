// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "nic_type_read.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "functional/nic/nic_handles.h"
#include "test_common.h"

namespace {

auto is_known_nic_type(amdsmi_nic_type_t type) -> bool {
  return ((type == AMDSMI_NIC_TYPE_AINIC) || (type == AMDSMI_NIC_TYPE_UALOE) ||
          (type == AMDSMI_NIC_TYPE_OTHER));
}

}  // namespace

TestNicTypeRead::TestNicTypeRead() : TestBase() {
  set_title("AMDSMI NIC Type Read Test");
  set_description(
      "This test verifies amdsmi_get_nic_type() and amdsmi_get_nic_capability(): they reject a "
      "null output pointer, a NIC reports a known type, and a UALoE endpoint never reports a "
      "host network port.");
}

TestNicTypeRead::~TestNicTypeRead(void) {}

// NIC queries need the NIC backend, not the default GPU init.
void TestNicTypeRead::SetUp(void) { TestBase::SetUp(AMDSMI_INIT_AMD_NICS); }

void TestNicTypeRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestNicTypeRead::DisplayResults(void) const { TestBase::DisplayResults(); }

void TestNicTypeRead::Close() { TestBase::Close(); }

void TestNicTypeRead::Run(void) {
  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) { std::cout << "** SetUp Failed for this test. Skipping.**\n"; }
    return;
  }

  // The pointer is checked before the handle, so this needs no NIC.
  EXPECT_EQ(amdsmi_get_nic_type(nullptr, nullptr), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amdsmi_get_nic_capability(nullptr, nullptr), AMDSMI_STATUS_INVAL);
  auto no_nic_capability = uint32_t{0};
  EXPECT_EQ(amdsmi_get_nic_capability(nullptr, &no_nic_capability), AMDSMI_STATUS_INVAL);

  const auto nics = nic_handles();
  if (nics.empty()) {
    IF_VERB(STANDARD) { std::cout << "\t**No AMD NIC present. Skipping.**\n"; }
    return;
  }

  for (const auto nic : nics) {
    auto type = AMDSMI_NIC_TYPE_UNKNOWN;
    ASSERT_EQ(amdsmi_get_nic_type(nic, &type), AMDSMI_STATUS_SUCCESS);
    EXPECT_TRUE(is_known_nic_type(type)) << "type " << static_cast<int>(type);

    auto capability = uint32_t{0};
    ASSERT_EQ(amdsmi_get_nic_capability(nic, &capability), AMDSMI_STATUS_SUCCESS);

    // A UALoE endpoint is portless, so it never exposes a host network port; any other NIC that
    // lists a port reports the NETDEV capability.
    if (type == AMDSMI_NIC_TYPE_UALOE) {
      EXPECT_EQ((capability & AMDSMI_NIC_CAP_NETDEV), 0u);
    } else {
      auto port_info = amdsmi_nic_port_info_t{};
      ASSERT_EQ(amdsmi_get_nic_port_info(nic, &port_info), AMDSMI_STATUS_SUCCESS);
      if (port_info.num_ports > 0) {
        EXPECT_NE((capability & AMDSMI_NIC_CAP_NETDEV), 0u);
      }
    }
  }
}
