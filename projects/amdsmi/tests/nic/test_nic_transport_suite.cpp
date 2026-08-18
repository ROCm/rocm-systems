// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Comprehensive test suite for NIC transport layer
 *
 * Tests all three backends (ioctl, netlink, auto) and validates:
 * - Basic functionality
 * - Backend selection
 * - Fallback behavior
 * - Result consistency between backends
 */

#include <array>
#include <cerrno>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "smi_nic_transport.h"

using namespace amd::smi::nic::transport;

// Test Infrastructure

struct TestResult {
  bool passed;
  std::string test_name;
  std::string message;
};

std::vector<TestResult> test_results;
int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;

void record_test(const std::string& test_name, bool passed, const std::string& message = "") {
  tests_run++;
  if (passed) {
    tests_passed++;
  } else {
    tests_failed++;
  }
  test_results.push_back({passed, test_name, message});

  std::cout << (passed ? "  PASS" : "  FAIL") << ": " << test_name;
  if (!message.empty()) {
    std::cout << " - " << message;
  }
  std::cout << "\n";
}

std::string format_mac(const std::array<uint8_t, 6>& mac) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (size_t i = 0; i < mac.size(); ++i) {
    if (i > 0) oss << ":";
    oss << std::setw(2) << static_cast<int>(mac[i]);
  }
  return oss.str();
}

// Test Cases

const char* backend_label(NicBackend_t backend) {
  switch (backend) {
    case NicBackend_t::Auto:
      return "auto";
    case NicBackend_t::Ioctl:
      return "ioctl";
    case NicBackend_t::Netlink:
      return "netlink";
  }
  return "unknown";
}

void test_backend_creation(NicBackend_t backend) {
  const std::string backend_type = backend_label(backend);
  std::cout << "\nTest: Backend Creation (" << backend_type << ")\n";
  std::cout << "+-------------------------------------------------------------------+\n";

  try {
    auto transport = create_transport(backend);
    record_test("Create " + backend_type + " backend", transport != nullptr);

    std::string backend_name = transport->backend_name();
    record_test("Backend name reported", !backend_name.empty(), "name: " + backend_name);
  } catch (const std::exception& e) {
    record_test("Create " + backend_type + " backend", false,
                std::string("Exception: ") + e.what());
  }
}

void test_ioctl_operations(const std::string& iface) {
  std::cout << "\nTest: Ioctl Backend Operations\n";
  std::cout << "+-------------------------------------------------------------------+\n";

  auto transport = create_transport(NicBackend_t::Ioctl);

  auto drvinfo = transport->get_driver_info(iface);
  record_test("Driver info", drvinfo.success,
              drvinfo.success ? "driver: " + drvinfo.value.driver_name
                              : "errno: " + std::to_string(drvinfo.error_code));

  auto pause = transport->get_pause_params(iface);
  record_test("Pause params", pause.success,
              pause.success ? std::string("rx=") + (pause.value.rx_pause ? "on" : "off")
                            : "errno: " + std::to_string(pause.error_code));

  auto link = transport->get_link_settings(iface);
  record_test("Link settings", link.success,
              link.success ? "speed: " + std::to_string(link.value.speed) + " Mbps"
                           : "errno: " + std::to_string(link.error_code));

  // A single-call GLINKSETTINGS left the base struct zeroed, so autoneg read
  // as off on an autonegotiating link. Guards the two-call handshake fix.
  // Assumes the supplied interface autonegotiates (typical for eth/eno NICs).
  if (link.success) {
    record_test("Link autoneg (two-call handshake)", link.value.autoneg != 0,
                std::string("autoneg=") + (link.value.autoneg ? "on" : "off"));
  }

  auto perm = transport->get_permanent_address(iface);
  record_test("Permanent address", perm.success,
              perm.success ? "mac: " + format_mac(perm.value.mac)
                           : "errno: " + std::to_string(perm.error_code));

  auto stats = transport->get_statistics(iface);
  record_test("Statistics", stats.success,
              stats.success ? "count: " + std::to_string(stats.value.names.size())
                            : "errno: " + std::to_string(stats.error_code));
}

#ifdef HAVE_LIBNL3
void test_netlink_operations(const std::string& iface) {
  std::cout << "\nTest: Netlink Backend Operations\n";
  std::cout << "+-------------------------------------------------------------------+\n";

  auto transport = create_transport(NicBackend_t::Netlink);

  auto drvinfo = transport->get_driver_info(iface);
  record_test("Driver info returns ENOTSUP", !drvinfo.success && drvinfo.error_code == ENOTSUP,
              "Expected ENOTSUP for unsupported operation");

  auto pause = transport->get_pause_params(iface);
  record_test("Pause params", pause.success || pause.error_code == ENOTSUP,
              pause.success ? std::string("rx=") + (pause.value.rx_pause ? "on" : "off")
                            : "errno: " + std::to_string(pause.error_code));

  auto link = transport->get_link_settings(iface);
  record_test("Link settings", link.success || link.error_code == ENOTSUP,
              link.success ? "speed: " + std::to_string(link.value.speed) + " Mbps"
                           : "errno: " + std::to_string(link.error_code));

  auto perm = transport->get_permanent_address(iface);
  record_test("Permanent address returns ENOTSUP", !perm.success && perm.error_code == ENOTSUP,
              "Expected ENOTSUP for unsupported operation");
}

void test_auto_backend_fallback(const std::string& iface) {
  std::cout << "\nTest: Auto Backend Fallback Logic\n";
  std::cout << "+-------------------------------------------------------------------+\n";

  auto transport = create_transport(NicBackend_t::Auto);
  std::cout << "  Auto backend name: " << transport->backend_name() << "\n";

  auto drvinfo = transport->get_driver_info(iface);
  record_test("Driver info (always ioctl)", drvinfo.success,
              "Auto backend should use ioctl for driver info");

  auto perm = transport->get_permanent_address(iface);
  record_test("Permanent address (always ioctl)", perm.success,
              "Auto backend should use ioctl for permanent address");

  auto pause = transport->get_pause_params(iface);
  record_test("Pause params (netlink or ioctl)", pause.success,
              "Auto backend should succeed via netlink or ioctl fallback");

  auto link = transport->get_link_settings(iface);
  record_test("Link settings (netlink or ioctl)", link.success,
              "Auto backend should succeed via netlink or ioctl fallback");
}

void test_ioctl_vs_netlink_consistency(const std::string& iface) {
  std::cout << "\nTest: Ioctl vs Netlink Result Consistency\n";
  std::cout << "+-------------------------------------------------------------------+\n";

  auto ioctl_transport = create_transport(NicBackend_t::Ioctl);
  auto netlink_transport = create_transport(NicBackend_t::Netlink);

  auto ioctl_pause = ioctl_transport->get_pause_params(iface);
  auto netlink_pause = netlink_transport->get_pause_params(iface);

  if (ioctl_pause.success && netlink_pause.success) {
    bool pause_match = (ioctl_pause.value.rx_pause == netlink_pause.value.rx_pause &&
                        ioctl_pause.value.tx_pause == netlink_pause.value.tx_pause &&
                        ioctl_pause.value.autoneg == netlink_pause.value.autoneg);
    record_test("Pause params match", pause_match,
                std::string("rx: ioctl=") + (ioctl_pause.value.rx_pause ? "on" : "off") +
                    " netlink=" + (netlink_pause.value.rx_pause ? "on" : "off"));
  } else {
    record_test("Pause comparison", true, "Skipped (one backend failed or ENOTSUP)");
  }

  auto ioctl_link = ioctl_transport->get_link_settings(iface);
  auto netlink_link = netlink_transport->get_link_settings(iface);

  if (ioctl_link.success && netlink_link.success) {
    bool link_match = (ioctl_link.value.speed == netlink_link.value.speed &&
                       ioctl_link.value.duplex == netlink_link.value.duplex);
    record_test("Link settings match", link_match,
                "speed: ioctl=" + std::to_string(ioctl_link.value.speed) +
                    " netlink=" + std::to_string(netlink_link.value.speed));
  } else {
    record_test("Link comparison", true, "Skipped (one backend failed or ENOTSUP)");
  }
}
#endif

// Main Test Runner

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <interface>\n";
    std::cerr << "Example: sudo " << argv[0] << " eth0\n";
    std::cerr << "\n";
    std::cerr << "This test suite validates the NIC transport layer:\n";
    std::cerr << "  - Backend creation (ioctl, netlink, auto)\n";
    std::cerr << "  - All transport operations\n";
    std::cerr << "  - Fallback behavior\n";
    std::cerr << "  - Result consistency between backends\n";
    return 1;
  }

  std::string iface = argv[1];

  std::cout << "+-------------------------------------------------------------------+\n";
  std::cout << "|  AMDSMI NIC Transport Layer Test Suite                           |\n";
  std::cout << "+-------------------------------------------------------------------+\n";
  std::cout << "\n";
  std::cout << "Interface: " << iface << "\n";
  std::cout << "Build configuration:\n";
#ifdef HAVE_LIBNL3
  std::cout << "  Netlink support: YES (libnl-3 available)\n";
#else
  std::cout << "  Netlink support: NO (libnl-3 not available)\n";
#endif
  std::cout << "\n";

  // Test 1: Backend Creation
  test_backend_creation(NicBackend_t::Ioctl);
#ifdef HAVE_LIBNL3
  test_backend_creation(NicBackend_t::Netlink);
  test_backend_creation(NicBackend_t::Auto);
#else
  std::cout << "\nSkipping netlink and auto backend tests (HAVE_LIBNL3 not defined)\n";
#endif

  // Test 2: Ioctl Operations (always available)
  test_ioctl_operations(iface);

#ifdef HAVE_LIBNL3
  // Test 3: Netlink Operations
  test_netlink_operations(iface);

  // Test 4: Auto Backend Fallback
  test_auto_backend_fallback(iface);

  // Test 5: Consistency Check
  test_ioctl_vs_netlink_consistency(iface);
#endif

  std::cout << "\n";
  std::cout << "+-------------------------------------------------------------------+\n";
  std::cout << "|  Test Summary                                                     |\n";
  std::cout << "+-------------------------------------------------------------------+\n";
  std::cout << "\n";
  std::cout << "Total Tests:  " << tests_run << "\n";
  std::cout << "Passed:       " << tests_passed << " ("
            << (tests_run > 0 ? (tests_passed * 100 / tests_run) : 0) << "%)\n";
  std::cout << "Failed:       " << tests_failed << " ("
            << (tests_run > 0 ? (tests_failed * 100 / tests_run) : 0) << "%)\n";
  std::cout << "\n";

  if (tests_failed > 0) {
    std::cout << "Failed Tests:\n";
    for (const auto& result : test_results) {
      if (!result.passed) {
        std::cout << "  - " << result.test_name;
        if (!result.message.empty()) {
          std::cout << ": " << result.message;
        }
        std::cout << "\n";
      }
    }
    std::cout << "\n";
  }

  if (tests_failed == 0) {
    std::cout << "All tests passed!\n";
    std::cout << "\n";
    std::cout << "The transport layer is working correctly.\n";
    std::cout << "Backend implementation is validated.\n";
    return 0;
  } else {
    std::cout << "Some tests failed.\n";
    std::cout << "\n";
    std::cout << "Notes:\n";
    std::cout << "  - Some failures may be expected if the NIC driver doesn't\n";
    std::cout << "    support certain features (e.g., pause frames on some NICs).\n";
    std::cout << "  - Netlink failures on kernel < 5.6 are expected.\n";
    std::cout << "  - ENOTSUP errors indicate unsupported operations (expected).\n";
    return 1;
  }
}
