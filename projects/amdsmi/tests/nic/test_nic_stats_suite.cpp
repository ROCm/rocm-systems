// SPDX-License-Identifier: MIT
/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * Unit tests for the tiered vendor-stat mechanism (StatTable_t filtering,
 * query-time refresh, FEC merge). No hardware, no root: the transport is a
 * fake returning canned VendorStatistics/FecStatistics_t.
 */

#include <cerrno>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "smi_nic.h"
#include "smi_nic_interface.h"
#include "smi_nic_stats.h"
#include "smi_nic_transport.h"
#include "vendors/broadcom/broadcom_stats.h"
#include "vendors/pensando/pensando_stats.h"

namespace tp = amd::smi::nic::transport;

static int tests_run = 0;
static int tests_failed = 0;

static void check(const std::string& name, bool passed, const std::string& detail = "") {
  tests_run++;
  if (!passed) {
    tests_failed++;
  }
  std::cout << (passed ? "  PASS: " : "  FAIL: ") << name;
  if (!detail.empty()) {
    std::cout << " - " << detail;
  }
  std::cout << "\n";
}

// Fake transport: returns whatever the test pre-loads, no I/O.
class FakeTransport : public tp::NicTransport {
 public:
  tp::Result<tp::VendorStatistics> stats{true, {}, 0};
  tp::Result<tp::FecStatistics_t> fec_stats{false, {}, ENOTSUP};

  tp::Result<tp::PauseParams> get_pause_params(const std::string&) override {
    return {false, {}, ENOTSUP};
  }
  tp::Result<tp::LinkSettings> get_link_settings(const std::string&) override {
    return {false, {}, ENOTSUP};
  }
  tp::Result<tp::DriverInfo> get_driver_info(const std::string&) override {
    return {false, {}, ENOTSUP};
  }
  tp::Result<tp::VendorStatistics> get_statistics(const std::string&) override { return stats; }
  tp::Result<tp::FecStatistics_t> get_fec_statistics(const std::string&) override {
    return fec_stats;
  }
  tp::Result<tp::PermanentAddress> get_permanent_address(const std::string&) override {
    return {false, {}, ENOTSUP};
  }
  std::string backend_name() const override { return "fake"; }
};

static const StatTable_t kTestTable = {
    {"tx_packets", StatTier_t::Default},
    {"rx_packets", StatTier_t::Default},
    {"tx_bytes", StatTier_t::Extended},
    {"corrected_blocks", StatTier_t::Extended},
    {"ring_only_not_in_table", StatTier_t::Default},  // deliberately never present in fake stats
};

static SmiNicPort make_port(std::shared_ptr<FakeTransport> transport, const StatTable_t* table) {
  return SmiNicPort("faketh0", "0000:00:00.0", "/tmp", "/tmp", transport, table);
}

static void test_default_scope_excludes_extended() {
  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {true, {{"tx_packets", "rx_packets", "tx_bytes"}, {1, 2, 3}}, 0};

  auto port = make_port(transport, &kTestTable);
  port.collect_vendor_statistics();
  auto result = port.get_vendor_stats_map(StatTier_t::Default);

  check("default scope has tx_packets", result.count("tx_packets") == 1);
  check("default scope has rx_packets", result.count("rx_packets") == 1);
  check("default scope excludes tx_bytes (Extended)", result.count("tx_bytes") == 0);
  check("default scope excludes name absent from stats",
        result.count("ring_only_not_in_table") == 0);
  check("default scope size", result.size() == 2, std::to_string(result.size()));
}

static void test_extended_scope_is_superset() {
  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {true, {{"tx_packets", "rx_packets", "tx_bytes"}, {1, 2, 3}}, 0};

  auto port = make_port(transport, &kTestTable);
  port.collect_vendor_statistics();
  auto result = port.get_vendor_stats_map(StatTier_t::Extended);

  check("extended scope has tx_packets", result.count("tx_packets") == 1);
  check("extended scope has tx_bytes", result.count("tx_bytes") == 1);
  check("extended scope size", result.size() == 3, std::to_string(result.size()));
}

static void test_names_outside_table_are_dropped() {
  auto transport = std::make_shared<FakeTransport>();
  // "[0]: rx_ucast_packets"-shaped and out-of-table names must never appear.
  transport->stats = {
      true, {{"tx_packets", "[0]: rx_ucast_packets", "some_unrelated_counter"}, {1, 2, 3}}, 0};

  auto port = make_port(transport, &kTestTable);
  port.collect_vendor_statistics();
  auto result = port.get_vendor_stats_map(StatTier_t::Extended);

  check("out-of-table ring row dropped", result.count("[0]: rx_ucast_packets") == 0);
  check("out-of-table unrelated counter dropped", result.count("some_unrelated_counter") == 0);
  check("in-table counter kept", result.count("tx_packets") == 1);
}

static void test_refresh_sees_changed_counters() {
  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {true, {{"tx_packets"}, {1}}, 0};

  auto port = make_port(transport, &kTestTable);
  port.collect_vendor_statistics();
  auto first = port.get_vendor_stats_map(StatTier_t::Default);
  check("first read sees value 1", first.at("tx_packets") == 1);

  transport->stats.value.values[0] = 42;
  port.collect_vendor_statistics();
  auto second = port.get_vendor_stats_map(StatTier_t::Default);
  check("second read (after refresh) sees value 42", second.at("tx_packets") == 42,
        std::to_string(second.at("tx_packets")));
}

static void test_fec_merge_ionic_shape() {
  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {true, {{"tx_packets"}, {1}}, 0};
  transport->fec_stats = {true, {{"corrected_blocks"}, {99}}, 0};

  auto port = make_port(transport, &kTestTable);
  port.collect_vendor_statistics();
  auto result = port.get_vendor_stats_map(StatTier_t::Extended);

  check("FEC counter merged in", result.count("corrected_blocks") == 1);
  check("FEC counter value correct", result.at("corrected_blocks") == 99);
}

// Guards the emplace-only regression where a FEC value seeded once at
// discovery time never updates again: refresh must overwrite it, not
// leave the first-collected value frozen.
static void test_fec_counters_refresh_on_subsequent_collect() {
  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {true, {{"tx_packets"}, {1}}, 0};
  transport->fec_stats = {true, {{"corrected_blocks"}, {10}}, 0};

  auto port = make_port(transport, &kTestTable);
  port.collect_vendor_statistics();
  auto first = port.get_vendor_stats_map(StatTier_t::Extended);
  check("first FEC read sees value 10", first.at("corrected_blocks") == 10,
        std::to_string(first.at("corrected_blocks")));

  transport->fec_stats.value.values[0] = 20;
  port.collect_vendor_statistics();
  auto second = port.get_vendor_stats_map(StatTier_t::Extended);
  check("second FEC read (after refresh) sees value 20, not frozen at 10",
        second.at("corrected_blocks") == 20, std::to_string(second.at("corrected_blocks")));
}

static void test_fec_failure_is_not_an_error() {
  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {true, {{"tx_packets"}, {1}}, 0};
  transport->fec_stats = {false, {}, ENOTSUP};  // ioctl backend / no libnl-3 shape

  auto port = make_port(transport, &kTestTable);
  port.collect_vendor_statistics();
  auto result = port.get_vendor_stats_map(StatTier_t::Extended);

  check("tx_packets still present when FEC call fails", result.count("tx_packets") == 1);
  check("no FEC counter present when the call fails", result.count("corrected_blocks") == 0);
}

static void test_no_table_yields_empty_map() {
  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {true, {{"tx_packets"}, {1}}, 0};

  auto port = make_port(transport, nullptr);  // IFoE shape: no ports, no table
  port.collect_vendor_statistics();
  auto result = port.get_vendor_stats_map(StatTier_t::Extended);

  check("no table -> empty map", result.empty());
}

// Guards the filter mechanism itself, not SMI_NIC_MAX_STATISTICS (that cap is
// applied one layer up, in smi_nic_interface.cpp's fixed-size copy).
static void test_large_table_not_truncated_by_mechanism() {
  StatTable_t big_table;
  tp::VendorStatistics stats;
  for (int i = 0; i < 200; ++i) {
    // Static storage isn't needed here: the table only needs to outlive this
    // function, and std::string's SSO would dangle a `const char*` taken from
    // a temporary, so build names into an owning vector first.
    stats.names.push_back("stat_" + std::to_string(i));
    stats.values.push_back(static_cast<uint64_t>(i));
  }
  static std::vector<std::string> owned_names = stats.names;  // keep c_str() storage alive
  for (const auto& name : owned_names) {
    big_table.push_back({name.c_str(), StatTier_t::Default});
  }

  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {true, stats, 0};

  auto port = make_port(transport, &big_table);
  port.collect_vendor_statistics();
  auto result = port.get_vendor_stats_map(StatTier_t::Default);

  check("200-entry table not truncated by the mechanism", result.size() == 200,
        std::to_string(result.size()));
}

// Guards the real SMI_NIC_MAX_STATISTICS cap in smi_nic_interface.h, which
// test_large_table_not_truncated_by_mechanism above does not touch.
static void test_real_tables_fit_within_cap() {
  check("pensando table fits SMI_NIC_MAX_STATISTICS",
        kPensandoStatTable.size() <= SMI_NIC_MAX_STATISTICS);
  check("broadcom table fits SMI_NIC_MAX_STATISTICS",
        kBroadcomStatTable.size() <= SMI_NIC_MAX_STATISTICS);
}

int main() {
  test_default_scope_excludes_extended();
  test_extended_scope_is_superset();
  test_names_outside_table_are_dropped();
  test_refresh_sees_changed_counters();
  test_fec_merge_ionic_shape();
  test_fec_counters_refresh_on_subsequent_collect();
  test_fec_failure_is_not_an_error();
  test_no_table_yields_empty_map();
  test_large_table_not_truncated_by_mechanism();
  test_real_tables_fit_within_cap();

  std::cout << "\n" << tests_run << " tests run, " << tests_failed << " failed\n";
  return tests_failed == 0 ? 0 : 1;
}
