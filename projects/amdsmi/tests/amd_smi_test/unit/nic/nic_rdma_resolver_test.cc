// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * was_rdma_port_resolved() maps the flat RDMA port index of the public API to the unified NIC
 * library's (netdev port, InfiniBand device, RDMA port) triple.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

#include "amd_smi/impl/amd_smi_rdma_port.h"
#include "nic_unit_fixture.h"

namespace {

using amd::smi::RdmaPortRef_t;
using amd::smi::was_rdma_port_resolved;

constexpr auto kUnsetIndex = uint32_t{0xFFFFFFFF};
constexpr auto kNetdevName = "enP1p3s0f3";
constexpr auto kOtherNetdevName = "enP2p3s0f3";

// The structs hold hundreds of KB of char arrays, so they live on the heap.
struct RdmaPortTopology_t {
  std::unique_ptr<amdsmi_nic_rdma_devices_info_t> rdma =
      std::make_unique<amdsmi_nic_rdma_devices_info_t>();
  std::unique_ptr<amdsmi_nic_port_info_t> ports = std::make_unique<amdsmi_nic_port_info_t>();

  RdmaPortTopology_t() {
    *rdma = {};
    *ports = {};
  }

  auto add_netdev_port(const std::string& netdev) -> void {
    std::snprintf(ports->ports[ports->num_ports].netdev, AMDSMI_MAX_STRING_LENGTH, "%s",
                  netdev.c_str());
    ++ports->num_ports;
  }

  auto add_rdma_device(const std::string& netdev, uint8_t num_ports) -> void {
    auto& dev = rdma->rdma_dev_info[rdma->num_rdma_dev];
    dev.num_rdma_ports = num_ports;
    for (auto k = uint8_t{0}; k < num_ports; ++k) {
      std::snprintf(dev.rdma_port_info[k].netdev, AMDSMI_MAX_STRING_LENGTH, "%s", netdev.c_str());
    }
    ++rdma->num_rdma_dev;
  }

  auto was_port_resolved(uint32_t flat_index, RdmaPortRef_t* ref) const -> bool {
    return was_rdma_port_resolved(*rdma, *ports, flat_index, ref);
  }
};

// A result the resolver must overwrite, so an untouched one cannot equal an expected index.
auto make_unset_ref() -> RdmaPortRef_t {
  return RdmaPortRef_t{kUnsetIndex, kUnsetIndex, kUnsetIndex};
}

auto is_ref(const RdmaPortRef_t& ref, uint32_t port, uint32_t ib, uint32_t rdma_port)
    -> ::testing::AssertionResult {
  if ((ref.port_index == port) && (ref.ib_index == ib) && (ref.rdma_port_index == rdma_port)) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure()
         << "got (" << ref.port_index << "," << ref.ib_index << "," << ref.rdma_port_index
         << "), want (" << port << "," << ib << "," << rdma_port << ")";
}

TEST_F(NicUnit, RdmaResolverMapsTheOnlyPortToTheOrigin) {
  auto topology = RdmaPortTopology_t{};
  topology.add_netdev_port(kNetdevName);
  topology.add_rdma_device(kNetdevName, 1);

  auto ref = make_unset_ref();
  ASSERT_TRUE(topology.was_port_resolved(0, &ref));
  EXPECT_TRUE(is_ref(ref, 0, 0, 0));
}

TEST_F(NicUnit, RdmaResolverRejectsAnIndexPastTheLastPort) {
  auto topology = RdmaPortTopology_t{};
  topology.add_netdev_port(kNetdevName);
  topology.add_rdma_device(kNetdevName, 1);

  auto ref = make_unset_ref();
  EXPECT_FALSE(topology.was_port_resolved(1, &ref));
}

TEST_F(NicUnit, RdmaResolverMapsEachNetdevPortToItsOwnDevice) {
  auto topology = RdmaPortTopology_t{};
  topology.add_netdev_port(kNetdevName);
  topology.add_netdev_port(kOtherNetdevName);
  topology.add_rdma_device(kNetdevName, 1);
  topology.add_rdma_device(kOtherNetdevName, 1);

  auto ref = make_unset_ref();
  ASSERT_TRUE(topology.was_port_resolved(1, &ref));
  EXPECT_TRUE(is_ref(ref, 1, 0, 0));
}

TEST_F(NicUnit, RdmaResolverAdvancesTheDeviceWhenDevicesShareANetdevPort) {
  auto topology = RdmaPortTopology_t{};
  topology.add_netdev_port(kNetdevName);
  topology.add_rdma_device(kNetdevName, 1);
  topology.add_rdma_device(kNetdevName, 1);

  auto ref = make_unset_ref();
  ASSERT_TRUE(topology.was_port_resolved(1, &ref));
  EXPECT_TRUE(is_ref(ref, 0, 1, 0));
}

TEST_F(NicUnit, RdmaResolverAdvancesTheRdmaPortWithinOneDevice) {
  auto topology = RdmaPortTopology_t{};
  topology.add_netdev_port(kNetdevName);
  topology.add_rdma_device(kNetdevName, 2);

  auto ref = make_unset_ref();
  ASSERT_TRUE(topology.was_port_resolved(1, &ref));
  EXPECT_TRUE(is_ref(ref, 0, 0, 1));
  EXPECT_FALSE(topology.was_port_resolved(2, &ref));
}

TEST_F(NicUnit, RdmaResolverCountsPortsAcrossDevices) {
  auto topology = RdmaPortTopology_t{};
  topology.add_netdev_port(kNetdevName);
  topology.add_netdev_port(kOtherNetdevName);
  topology.add_rdma_device(kNetdevName, 2);
  topology.add_rdma_device(kOtherNetdevName, 1);

  auto ref = make_unset_ref();
  ASSERT_TRUE(topology.was_port_resolved(2, &ref));
  EXPECT_TRUE(is_ref(ref, 1, 0, 0));
}

TEST_F(NicUnit, RdmaResolverRejectsADeviceWhoseNetdevIsNotANicPort) {
  auto topology = RdmaPortTopology_t{};
  topology.add_netdev_port(kNetdevName);
  topology.add_rdma_device("unknown0", 1);

  auto ref = make_unset_ref();
  EXPECT_FALSE(topology.was_port_resolved(0, &ref));
}

TEST_F(NicUnit, RdmaResolverRejectsATopologyWithNoRdmaDevice) {
  auto topology = RdmaPortTopology_t{};

  auto ref = make_unset_ref();
  EXPECT_FALSE(topology.was_port_resolved(0, &ref));
}

TEST_F(NicUnit, RdmaResolverRejectsANullResultPointer) {
  auto topology = RdmaPortTopology_t{};
  topology.add_netdev_port(kNetdevName);
  topology.add_rdma_device(kNetdevName, 1);

  EXPECT_FALSE(topology.was_port_resolved(0, nullptr));
}

}  // namespace
