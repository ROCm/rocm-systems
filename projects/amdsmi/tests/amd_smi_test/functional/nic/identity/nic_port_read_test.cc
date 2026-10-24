// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "nic_port_read.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <limits>

#include "amd_smi/amdsmi.h"
#include "functional/nic/nic_handles.h"
#include "test_common.h"

namespace {

constexpr auto kMaxByteIfindex = uint32_t{std::numeric_limits<uint8_t>::max()};

// The vendor statistics count of the default tier.
auto read_vendor_count(amdsmi_processor_handle nic, uint32_t port_index, uint32_t* count)
    -> amdsmi_status_t {
  return amdsmi_get_nic_vendor_statistics(nic, port_index, count, nullptr);
}

// The vendor statistics count of the given tier.
auto read_scoped_count(amdsmi_processor_handle nic, uint32_t port_index,
                       amdsmi_nic_stat_scope_t scope, uint32_t* count) -> amdsmi_status_t {
  return amdsmi_get_nic_vendor_statistics_with_scope(nic, port_index, scope, count, nullptr);
}

}  // namespace

TestNicPortRead::TestNicPortRead() : TestBase() {
  set_title("AMDSMI NIC Port Read Test");
  set_description(
      "This test verifies amdsmi_get_nic_port_ifindex() and "
      "amdsmi_get_nic_vendor_statistics_with_scope(): they "
      "reject a null handle, the full interface index agrees with the one-byte struct field, and "
      "the extended statistics tier is a superset of the default one.");
}

TestNicPortRead::~TestNicPortRead(void) {}

// NIC queries need the NIC backend, not the default GPU init.
void TestNicPortRead::SetUp(void) { TestBase::SetUp(AMDSMI_INIT_AMD_NICS); }

void TestNicPortRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestNicPortRead::DisplayResults(void) const { TestBase::DisplayResults(); }

void TestNicPortRead::Close() { TestBase::Close(); }

void TestNicPortRead::Run(void) {
  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) { std::cout << "** SetUp Failed for this test. Skipping.**\n"; }
    return;
  }

  // Neither call reaches a device with a null handle, so these need no NIC.
  EXPECT_EQ(amdsmi_get_nic_port_ifindex(nullptr, 0, nullptr), AMDSMI_STATUS_INVAL);
  auto no_nic_ifindex = uint32_t{0};
  EXPECT_EQ(amdsmi_get_nic_port_ifindex(nullptr, 0, &no_nic_ifindex), AMDSMI_STATUS_INVAL);
  auto no_nic_count = uint32_t{0};
  EXPECT_EQ(amdsmi_get_nic_vendor_statistics_with_scope(nullptr, 0, AMDSMI_NIC_STAT_SCOPE_DEFAULT,
                                                        &no_nic_count, nullptr),
            AMDSMI_STATUS_INVAL);

  const auto nics = nic_handles();
  if (nics.empty()) {
    IF_VERB(STANDARD) { std::cout << "\t**No AMD NIC present. Skipping.**\n"; }
    return;
  }

  for (const auto nic : nics) {
    auto port_info = amdsmi_nic_port_info_t{};
    ASSERT_EQ(amdsmi_get_nic_port_info(nic, &port_info), AMDSMI_STATUS_SUCCESS);

    // The struct holds one byte; the getter returns the whole index. An index that does not
    // fit reads 0 in the struct, and so does a port with no driver bound, which the struct
    // reports as unavailable while the getter reads sysfs directly.
    for (auto i = uint32_t{0}; i < port_info.num_ports; ++i) {
      auto ifindex = uint32_t{0};
      ASSERT_EQ(amdsmi_get_nic_port_ifindex(nic, i, &ifindex), AMDSMI_STATUS_SUCCESS);
      const auto expected = (ifindex <= kMaxByteIfindex) ? ifindex : uint32_t{0};
      const auto in_struct = static_cast<uint32_t>(port_info.ports[i].ifindex);
      EXPECT_TRUE((in_struct == expected) || (in_struct == 0)) << "port " << i;
    }
    auto past_the_last = uint32_t{0};
    EXPECT_EQ(amdsmi_get_nic_port_ifindex(nic, port_info.num_ports, &past_the_last),
              AMDSMI_STATUS_NOT_FOUND);

    if (port_info.num_ports == 0) {
      continue;
    }

    // The extended tier is a superset of the default one, and the unscoped call reads the default.
    auto default_count = uint32_t{0};
    if (read_scoped_count(nic, 0, AMDSMI_NIC_STAT_SCOPE_DEFAULT, &default_count) !=
        AMDSMI_STATUS_SUCCESS) {
      IF_VERB(STANDARD) { std::cout << "\t**No vendor statistics on this NIC. Skipping.**\n"; }
      continue;
    }
    auto extended_count = uint32_t{0};
    ASSERT_EQ(read_scoped_count(nic, 0, AMDSMI_NIC_STAT_SCOPE_EXTENDED, &extended_count),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_GE(extended_count, default_count);

    auto unscoped_count = uint32_t{0};
    ASSERT_EQ(read_vendor_count(nic, 0, &unscoped_count), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(unscoped_count, default_count);
  }
}
