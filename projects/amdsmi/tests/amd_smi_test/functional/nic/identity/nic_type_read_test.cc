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
      "This test verifies amdsmi_get_nic_type(): it rejects a null output pointer, "
      "reports a known type for every NIC, agrees with the type field of "
      "amdsmi_get_nic_asic_info(), and never reports a host network port for a "
      "UALoE endpoint.");
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

  const auto nics = nic_handles();
  if (nics.empty()) {
    IF_VERB(STANDARD) { std::cout << "\t**No AMD NIC present. Skipping.**\n"; }
    return;
  }

  for (const auto nic : nics) {
    auto type = AMDSMI_NIC_TYPE_UNKNOWN;
    ASSERT_EQ(amdsmi_get_nic_type(nic, &type), AMDSMI_STATUS_SUCCESS);
    EXPECT_TRUE(is_known_nic_type(type)) << "type " << static_cast<int>(type);

    auto asic = amdsmi_nic_asic_info_t{};
    ASSERT_EQ(amdsmi_get_nic_asic_info(nic, &asic), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(static_cast<int>(type), static_cast<int>(asic.type));

    // A UALoE endpoint is portless, so it never exposes a host network port.
    if (type == AMDSMI_NIC_TYPE_UALOE) {
      EXPECT_EQ((asic.capability & AMDSMI_NIC_CAP_NETDEV), 0u);
    }
  }
}
