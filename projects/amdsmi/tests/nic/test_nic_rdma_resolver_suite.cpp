// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * Unit tests for amd::smi::was_rdma_port_resolved(): the flat RDMA port index of the public API
 * mapped to the unified NIC library's (netdev port, InfiniBand device, RDMA port) triple.
 * No hardware, no root.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "amd_smi/impl/amd_smi_rdma_port.h"

namespace {

using amd::smi::RdmaPortRef_t;
using amd::smi::was_rdma_port_resolved;

auto g_tests_run = int{0};
auto g_tests_failed = int{0};

auto check(const std::string& name, bool was_passed) -> void {
  ++g_tests_run;
  if (!was_passed) {
    ++g_tests_failed;
  }
  std::cout << (was_passed ? "  PASS: " : "  FAIL: ") << name << "\n";
}

// The structs hold hundreds of KB of char arrays, so they live on the heap.
struct Fixture_t {
  std::unique_ptr<amdsmi_nic_rdma_devices_info_t> rdma =
      std::make_unique<amdsmi_nic_rdma_devices_info_t>();
  std::unique_ptr<amdsmi_nic_port_info_t> ports = std::make_unique<amdsmi_nic_port_info_t>();

  Fixture_t() {
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

  auto resolve(uint32_t flat_index, RdmaPortRef_t* ref) const -> bool {
    return was_rdma_port_resolved(*rdma, *ports, flat_index, ref);
  }
};

constexpr auto kUnsetIndex = uint32_t{0xFFFFFFFF};

// A result the resolver must overwrite, so an untouched one cannot equal an expected index.
auto make_unset_ref() -> RdmaPortRef_t {
  return RdmaPortRef_t{kUnsetIndex, kUnsetIndex, kUnsetIndex};
}

auto is_ref(const RdmaPortRef_t& ref, uint32_t port, uint32_t ib, uint32_t rdma_port) -> bool {
  return ((ref.port_index == port) && (ref.ib_index == ib) && (ref.rdma_port_index == rdma_port));
}

auto test_one_device_one_port() -> void {
  auto fx = Fixture_t{};
  fx.add_netdev_port("enP1p3s0f3");
  fx.add_rdma_device("enP1p3s0f3", 1);

  auto ref = make_unset_ref();
  check("single device: index 0 resolves", fx.resolve(0, &ref));
  check("single device: index 0 is (0,0,0)", is_ref(ref, 0, 0, 0));
  check("single device: index 1 is out of range", !fx.resolve(1, &ref));
}

auto test_two_netdev_ports_each_with_a_device() -> void {
  auto fx = Fixture_t{};
  fx.add_netdev_port("enP1p3s0f3");
  fx.add_netdev_port("enP2p3s0f3");
  fx.add_rdma_device("enP1p3s0f3", 1);
  fx.add_rdma_device("enP2p3s0f3", 1);

  auto ref = make_unset_ref();
  check("two netdev ports: index 1 resolves", fx.resolve(1, &ref));
  check("two netdev ports: index 1 is (1,0,0)", is_ref(ref, 1, 0, 0));
}

auto test_two_devices_on_one_netdev_port() -> void {
  auto fx = Fixture_t{};
  fx.add_netdev_port("enP1p3s0f3");
  fx.add_rdma_device("enP1p3s0f3", 1);
  fx.add_rdma_device("enP1p3s0f3", 1);

  auto ref = make_unset_ref();
  check("shared netdev port: index 1 resolves", fx.resolve(1, &ref));
  check("shared netdev port: index 1 is (0,1,0)", is_ref(ref, 0, 1, 0));
}

auto test_one_device_with_two_rdma_ports() -> void {
  auto fx = Fixture_t{};
  fx.add_netdev_port("enP1p3s0f3");
  fx.add_rdma_device("enP1p3s0f3", 2);

  auto ref = make_unset_ref();
  check("two RDMA ports: index 1 resolves", fx.resolve(1, &ref));
  check("two RDMA ports: index 1 is (0,0,1)", is_ref(ref, 0, 0, 1));
  check("two RDMA ports: index 2 is out of range", !fx.resolve(2, &ref));
}

auto test_index_counts_ports_across_devices() -> void {
  auto fx = Fixture_t{};
  fx.add_netdev_port("enP1p3s0f3");
  fx.add_netdev_port("enP2p3s0f3");
  fx.add_rdma_device("enP1p3s0f3", 2);
  fx.add_rdma_device("enP2p3s0f3", 1);

  auto ref = make_unset_ref();
  check("ports across devices: index 2 resolves", fx.resolve(2, &ref));
  check("ports across devices: index 2 is (1,0,0)", is_ref(ref, 1, 0, 0));
}

auto test_unresolvable_inputs() -> void {
  auto fx = Fixture_t{};
  fx.add_netdev_port("enP1p3s0f3");
  fx.add_rdma_device("unknown0", 1);

  auto ref = make_unset_ref();
  check("a device whose netdev is not a NIC port does not resolve", !fx.resolve(0, &ref));

  auto empty = Fixture_t{};
  check("no RDMA device does not resolve", !empty.resolve(0, &ref));
  check("a null result pointer does not resolve", !fx.resolve(0, nullptr));
}

}  // namespace

auto main() -> int {
  test_one_device_one_port();
  test_two_netdev_ports_each_with_a_device();
  test_two_devices_on_one_netdev_port();
  test_one_device_with_two_rdma_ports();
  test_index_counts_ports_across_devices();
  test_unresolvable_inputs();

  std::cout << "\n" << g_tests_run << " tests run, " << g_tests_failed << " failed\n";
  return (g_tests_failed == 0) ? 0 : 1;
}
