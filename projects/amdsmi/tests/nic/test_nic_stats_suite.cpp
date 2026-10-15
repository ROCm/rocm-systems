// SPDX-License-Identifier: MIT
/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * Unit tests for the tiered vendor-stat mechanism (StatTable_t filtering,
 * query-time refresh, FEC merge). No hardware, no root: the transport is a
 * fake returning canned VendorStatistics/FecStatistics_t.
 */

#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "nic_check.h"
#include "smi_nic.h"
#include "smi_nic_interface.h"
#include "smi_nic_log.h"
#include "smi_nic_stats.h"
#include "smi_nic_testing.h"
#include "smi_nic_transport.h"
#include "vendors/broadcom/broadcom_stats.h"
#include "vendors/pensando/pensando_stats.h"

namespace tp = amd::smi::nic::transport;

using nic_test::check;

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

static std::vector<std::string> g_log_lines;
static void capture_log(const std::string& msg) { g_log_lines.push_back(msg); }

static bool has_log_line_with(const std::vector<std::string>& needles) {
  for (const auto& line : g_log_lines) {
    bool is_match = true;
    for (const auto& needle : needles) {
      is_match = (is_match && (line.find(needle) != std::string::npos));
    }
    if (is_match) {
      return true;
    }
  }
  return false;
}

// A FEC query that returns the wrong card's numbers, or nothing, is invisible
// without the interface and outcome in the log.
static void test_fec_query_is_logged_with_interface_and_outcome() {
  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {true, {{"tx_packets"}, {1}}, 0};
  auto port = make_port(transport, &kTestTable);

  g_log_lines.clear();
  amd::smi::nic::log::set_sink(capture_log);

  transport->fec_stats = {true, {{"corrected_blocks", "corrected_bits"}, {16, 12}}, 0};
  port.collect_vendor_statistics();
  check("FEC success line names the interface and values",
        has_log_line_with({"faketh0", "corrected_blocks=16", "corrected_bits=12"}));

  g_log_lines.clear();
  transport->fec_stats = {false, {}, ENODATA};
  port.collect_vendor_statistics();
  check("FEC failure line names the interface and the error code",
        has_log_line_with({"faketh0", "error=" + std::to_string(ENODATA)}));

  amd::smi::nic::log::set_sink(nullptr);
  g_log_lines.clear();
  port.collect_vendor_statistics();
  check("no FEC line when no log sink is installed", g_log_lines.empty());
}

static void write_counter(const std::filesystem::path& dir, const std::string& name,
                          uint64_t value) {
  std::ofstream(dir / name) << value << "\n";
}

// The kernel keeps its stats refresh interval in hw_counters/lifespan; it is a setting, not a
// counter.
static void test_rdma_hw_counters_skip_lifespan_and_refresh() {
  namespace fs = std::filesystem;
  const fs::path root =
      fs::temp_directory_path() / ("amdsmi-rdma-hw-" + std::to_string(::getpid()));
  const fs::path counters = root / "hw_counters";
  fs::create_directories(counters);
  write_counter(counters, "rx_rdma_ucast_pkts", 5);
  write_counter(counters, "lifespan", 10);

  std::string netdev = "enP1p3s0f3";
  std::string name = "1";
  SmiInfiniBandPort port(netdev, name, root.string());
  port.collect_hw_counters();
  check("lifespan is not reported as a counter", port.get_hw_counters_map().count("lifespan") == 0);
  check("a real counter is read", port.get_hw_counters_map().count("rx_rdma_ucast_pkts") == 1);

  write_counter(counters, "rx_rdma_ucast_pkts", 9);
  write_counter(counters, "tx_rdma_ucast_pkts", 1);
  port.collect_hw_counters();
  check("a second collect sees the changed value",
        port.get_hw_counters_map().at("rx_rdma_ucast_pkts") == 9);
  check("a second collect adds a new counter",
        port.get_hw_counters_map().count("tx_rdma_ucast_pkts") == 1);

  fs::remove(counters / "tx_rdma_ucast_pkts");
  port.collect_hw_counters();
  check("a counter the driver stops reporting disappears",
        port.get_hw_counters_map().count("tx_rdma_ucast_pkts") == 0);
  fs::remove_all(root);
}

// Every statistics read ends in a C API call, where an exception means std::terminate. A sysfs
// error other than "absent" (a symlink loop here) must read as no data, not throw.
static bool is_collect_throwing(const std::function<void()>& collect) {
  try {
    collect();
  } catch (...) {
    return true;
  }
  return false;
}

static void test_standard_statistics_survive_sysfs_errors() {
  namespace fs = std::filesystem;
  const fs::path root =
      fs::temp_directory_path() / ("amdsmi-std-stats-" + std::to_string(::getpid()));
  fs::remove_all(root);

  // statistics itself is a symlink loop.
  const fs::path looped = root / "looped";
  fs::create_directories(looped);
  fs::create_symlink("statistics", looped / "statistics");
  SmiNicPort looped_port("faketh0", "0000:00:00.0", looped.string(), looped.string());
  check("a looping statistics dir does not throw",
        !is_collect_throwing([&] { looped_port.collect_standard_statistics(); }));
  check("a looping statistics dir reads as no counters",
        looped_port.get_standard_stats_map().empty());

  // One unreadable entry must not hide the counters beside it.
  const fs::path mixed = root / "mixed";
  fs::create_directories(mixed / "statistics");
  fs::create_symlink("bad", mixed / "statistics" / "bad");
  write_counter(mixed / "statistics", "rx_bytes", 7);
  SmiNicPort mixed_port("faketh0", "0000:00:00.0", mixed.string(), mixed.string());
  check("a looping entry does not throw",
        !is_collect_throwing([&] { mixed_port.collect_standard_statistics(); }));
  check("the counters beside a looping entry are still read",
        (mixed_port.get_standard_stats_map().count("rx_bytes") == 1));
  fs::remove_all(root);
}

static void test_rdma_hw_counters_survive_sysfs_errors() {
  namespace fs = std::filesystem;
  const fs::path root =
      fs::temp_directory_path() / ("amdsmi-hw-errs-" + std::to_string(::getpid()));
  fs::remove_all(root);
  std::string netdev = "enP1p3s0f3";
  std::string name = "1";

  const fs::path looped = root / "looped";
  fs::create_directories(looped);
  fs::create_symlink("hw_counters", looped / "hw_counters");
  SmiInfiniBandPort looped_port(netdev, name, looped.string());
  check("a looping hw_counters dir does not throw",
        !is_collect_throwing([&] { looped_port.collect_hw_counters(); }));
  check("a looping hw_counters dir reads as no counters",
        looped_port.get_hw_counters_map().empty());

  const fs::path mixed = root / "mixed";
  fs::create_directories(mixed / "hw_counters");
  fs::create_symlink("bad", mixed / "hw_counters" / "bad");
  write_counter(mixed / "hw_counters", "rx_rdma_ucast_pkts", 3);
  SmiInfiniBandPort mixed_port(netdev, name, mixed.string());
  check("a looping hw counter entry does not throw",
        !is_collect_throwing([&] { mixed_port.collect_hw_counters(); }));
  check("the counters beside a looping entry are still read",
        (mixed_port.get_hw_counters_map().count("rx_rdma_ucast_pkts") == 1));
  fs::remove_all(root);
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
  const auto owned_names = stats.names;  // keep c_str() storage alive
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

// Interface layer: the smi_get_nic_*_statistics_* entry points, reached through a prepared system.
// One NIC at 0000:03:00.0 with one port whose driver counts as loaded; no hardware.
constexpr auto kFakeNicBdf = uint64_t{0x300};

class FakeSubsystem : public SmiNicSubsystem {
 public:
  void discover(const std::string&, const std::string&,
                std::shared_ptr<tp::NicTransport>) override {}
  NicVendor vendor() const override { return NicVendor::AMD; }
  bool is_driver_loaded(const std::string& bdf, DriverType) const override {
    return (m_unbound.count(bdf) == 0);
  }
  const std::vector<std::unique_ptr<SmiNic>>& get_nics() const override { return nics_; }
  void add(std::unique_ptr<SmiNic> nic) { nics_.push_back(std::move(nic)); }
  void unbind(const std::string& bdf) { m_unbound.insert(bdf); }

 private:
  std::vector<std::unique_ptr<SmiNic>> nics_;
  std::set<std::string> m_unbound;
};

struct InterfaceFixture_t {
  InterfaceFixture_t(const std::string& class_path, std::shared_ptr<FakeTransport> transport,
                     const StatTable_t* table, const SmiInfiniBand* infiniband = nullptr) {
    auto nic = std::make_unique<SmiNic>("faketh0", "0000:03:00.0");
    auto port = SmiNicPort("faketh0", "0000:03:00.1", class_path, "/tmp", transport, table);
    if (infiniband != nullptr) {
      port.add_infiniband(*infiniband);
    }
    nic->add_nic_port(std::move(port));
    auto subsystem = std::make_unique<FakeSubsystem>();
    subsystem->add(std::move(nic));
    auto system = std::make_unique<SmiNicSystem>("/tmp", "/tmp");
    system->register_subsystem(std::move(subsystem));
    system->discover_nics();
    smi_nic_create_context_for_testing(&ctx, std::move(system));
  }
  ~InterfaceFixture_t() {
    if (ctx != nullptr) {
      smi_nic_destroy_context(ctx);
    }
  }
  InterfaceFixture_t(const InterfaceFixture_t&) = delete;
  InterfaceFixture_t& operator=(const InterfaceFixture_t&) = delete;

  smi_nic_ctx_t ctx = nullptr;
};

static void test_interface_reads_vendor_statistics_from_a_prepared_system() {
  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {true, {{"tx_packets", "rx_packets"}, {5, 6}}, 0};
  InterfaceFixture_t fixture("/tmp", transport, &kTestTable);

  auto count = uint32_t{0};
  check("vendor count succeeds",
        smi_get_nic_vendor_statistics_count(fixture.ctx, kFakeNicBdf, 0, SMI_NIC_STAT_SCOPE_DEFAULT,
                                            &count) == SMI_NIC_STATUS_SUCCESS);
  check("vendor count sees both counters", count == 2, std::to_string(count));

  auto info = std::make_unique<smi_nic_stat_info_t>();
  check("vendor list succeeds",
        smi_get_nic_vendor_statistics_list(fixture.ctx, kFakeNicBdf, 0, SMI_NIC_STAT_SCOPE_DEFAULT,
                                           info.get()) == SMI_NIC_STATUS_SUCCESS);
  check("vendor list carries both counters", info->count == 2, std::to_string(info->count));
}

static void test_interface_reports_a_failed_vendor_read_as_an_error() {
  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {false, {}, EPERM};
  InterfaceFixture_t fixture("/tmp", transport, &kTestTable);

  auto count = uint32_t{0};
  check("a failed vendor read is an error, not zero counters",
        smi_get_nic_vendor_statistics_count(fixture.ctx, kFakeNicBdf, 0, SMI_NIC_STAT_SCOPE_DEFAULT,
                                            &count) == SMI_NIC_STATUS_ERROR);

  auto info = std::make_unique<smi_nic_stat_info_t>();
  check("a failed vendor read is an error from the list too",
        smi_get_nic_vendor_statistics_list(fixture.ctx, kFakeNicBdf, 0, SMI_NIC_STAT_SCOPE_DEFAULT,
                                           info.get()) == SMI_NIC_STATUS_ERROR);
}

// A driver without ethtool statistics has none to report; that is not a failure.
static void test_interface_treats_an_unsupported_vendor_read_as_no_counters() {
  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {false, {}, ENOTSUP};
  InterfaceFixture_t fixture("/tmp", transport, &kTestTable);

  auto count = uint32_t{99};
  check("an unsupported vendor read still succeeds",
        smi_get_nic_vendor_statistics_count(fixture.ctx, kFakeNicBdf, 0, SMI_NIC_STAT_SCOPE_DEFAULT,
                                            &count) == SMI_NIC_STATUS_SUCCESS);
  check("an unsupported vendor read reports zero counters", count == 0, std::to_string(count));
}

// A caller sizes its array from the count, so the count cannot promise more than the list returns.
static void test_interface_count_matches_the_cap_on_the_list() {
  namespace fs = std::filesystem;
  const fs::path root =
      fs::temp_directory_path() / ("amdsmi-stat-cap-" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "statistics");
  const auto num_counters = uint32_t{SMI_NIC_MAX_STATISTICS} + 44;
  for (auto i = uint32_t{0}; i < num_counters; ++i) {
    write_counter(root / "statistics", "counter_" + std::to_string(i), i);
  }
  InterfaceFixture_t fixture(root.string(), std::make_shared<FakeTransport>(), &kTestTable);

  auto count = uint32_t{0};
  check("port count succeeds", smi_get_nic_port_statistics_count(fixture.ctx, kFakeNicBdf, 0,
                                                                 &count) == SMI_NIC_STATUS_SUCCESS);
  check("port count stops at the cap", count == SMI_NIC_MAX_STATISTICS, std::to_string(count));

  auto info = std::make_unique<smi_nic_stat_info_t>();
  check("port list succeeds",
        smi_get_nic_port_statistics_list(fixture.ctx, kFakeNicBdf, 0, info.get()) ==
            SMI_NIC_STATUS_SUCCESS);
  check("port list stops at the cap too", info->count == SMI_NIC_MAX_STATISTICS,
        std::to_string(info->count));
  fs::remove_all(root);
}

static void test_interface_rdma_count_matches_the_cap_on_the_list() {
  namespace fs = std::filesystem;
  const fs::path root =
      fs::temp_directory_path() / ("amdsmi-rdma-cap-" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "hw_counters");
  const auto num_counters = uint32_t{SMI_NIC_MAX_STATISTICS} + 44;
  for (auto i = uint32_t{0}; i < num_counters; ++i) {
    write_counter(root / "hw_counters", "hw_" + std::to_string(i), i);
  }
  std::string netdev = "faketh0";
  std::string name = "1";
  std::string ib_name = "fakerdma0";
  std::string ib_path = root.string();
  SmiInfiniBand infiniband(ib_name, ib_path);
  infiniband.add_port(SmiInfiniBandPort(netdev, name, ib_path));
  InterfaceFixture_t fixture("/tmp", std::make_shared<FakeTransport>(), &kTestTable, &infiniband);

  auto count = uint32_t{0};
  check("rdma count succeeds",
        smi_get_nic_rdma_port_statistics_count(fixture.ctx, kFakeNicBdf, 0, 0, 0, &count) ==
            SMI_NIC_STATUS_SUCCESS);
  check("rdma count stops at the cap", count == SMI_NIC_MAX_STATISTICS, std::to_string(count));

  auto info = std::make_unique<smi_nic_stat_info_t>();
  check("rdma list succeeds",
        smi_get_nic_rdma_port_statistics_list(fixture.ctx, kFakeNicBdf, 0, 0, 0, info.get()) ==
            SMI_NIC_STATUS_SUCCESS);
  check("rdma list stops at the cap too", info->count == SMI_NIC_MAX_STATISTICS,
        std::to_string(info->count));
  fs::remove_all(root);
}

// An empty set reads the same from the count and the list, so a caller that probes with the count
// and then asks for the list is not surprised by a different status.
static void test_interface_vendor_list_of_nothing_succeeds_with_zero() {
  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {true, {}, 0};
  InterfaceFixture_t fixture("/tmp", transport, &kTestTable);

  auto count = uint32_t{99};
  check("an empty vendor count succeeds",
        smi_get_nic_vendor_statistics_count(fixture.ctx, kFakeNicBdf, 0, SMI_NIC_STAT_SCOPE_DEFAULT,
                                            &count) == SMI_NIC_STATUS_SUCCESS);
  check("an empty vendor count is zero", count == 0, std::to_string(count));

  auto info = std::make_unique<smi_nic_stat_info_t>();
  info->count = 99;
  check("an empty vendor list succeeds like the count",
        smi_get_nic_vendor_statistics_list(fixture.ctx, kFakeNicBdf, 0, SMI_NIC_STAT_SCOPE_DEFAULT,
                                           info.get()) == SMI_NIC_STATUS_SUCCESS);
  check("an empty vendor list is zero", info->count == 0, std::to_string(info->count));
}

static void test_interface_port_list_of_nothing_succeeds_with_zero() {
  namespace fs = std::filesystem;
  const fs::path root =
      fs::temp_directory_path() / ("amdsmi-std-empty-" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "statistics");
  InterfaceFixture_t fixture(root.string(), std::make_shared<FakeTransport>(), &kTestTable);

  auto count = uint32_t{99};
  check("an empty port count succeeds",
        smi_get_nic_port_statistics_count(fixture.ctx, kFakeNicBdf, 0, &count) ==
            SMI_NIC_STATUS_SUCCESS);
  check("an empty port count is zero", count == 0, std::to_string(count));

  auto info = std::make_unique<smi_nic_stat_info_t>();
  info->count = 99;
  check("an empty port list succeeds like the count",
        smi_get_nic_port_statistics_list(fixture.ctx, kFakeNicBdf, 0, info.get()) ==
            SMI_NIC_STATUS_SUCCESS);
  check("an empty port list is zero", info->count == 0, std::to_string(info->count));
  fs::remove_all(root);
}

static void test_interface_rdma_list_of_nothing_succeeds_with_zero() {
  namespace fs = std::filesystem;
  const fs::path root =
      fs::temp_directory_path() / ("amdsmi-rdma-empty-" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "hw_counters");
  std::string netdev = "faketh0";
  std::string name = "1";
  std::string ib_name = "fakerdma0";
  std::string ib_path = root.string();
  SmiInfiniBand infiniband(ib_name, ib_path);
  infiniband.add_port(SmiInfiniBandPort(netdev, name, ib_path));
  InterfaceFixture_t fixture("/tmp", std::make_shared<FakeTransport>(), &kTestTable, &infiniband);

  auto count = uint32_t{99};
  check("an empty rdma count succeeds",
        smi_get_nic_rdma_port_statistics_count(fixture.ctx, kFakeNicBdf, 0, 0, 0, &count) ==
            SMI_NIC_STATUS_SUCCESS);
  check("an empty rdma count is zero", count == 0, std::to_string(count));

  auto info = std::make_unique<smi_nic_stat_info_t>();
  info->count = 99;
  check("an empty rdma list succeeds like the count",
        smi_get_nic_rdma_port_statistics_list(fixture.ctx, kFakeNicBdf, 0, 0, 0, info.get()) ==
            SMI_NIC_STATUS_SUCCESS);
  check("an empty rdma list is zero", info->count == 0, std::to_string(info->count));
  fs::remove_all(root);
}

// The count and the list answer an index past the last port the same way.
static void test_interface_out_of_range_index_reads_the_same_from_count_and_list() {
  auto transport = std::make_shared<FakeTransport>();
  InterfaceFixture_t fixture("/tmp", transport, &kTestTable);
  const auto kPastLastPort = uint32_t{7};

  auto count = uint32_t{0};
  auto info = std::make_unique<smi_nic_stat_info_t>();
  check("a port count past the last port is not found",
        smi_get_nic_port_statistics_count(fixture.ctx, kFakeNicBdf, kPastLastPort, &count) ==
            SMI_NIC_STATUS_NOT_FOUND);
  check("a port list past the last port is not found",
        smi_get_nic_port_statistics_list(fixture.ctx, kFakeNicBdf, kPastLastPort, info.get()) ==
            SMI_NIC_STATUS_NOT_FOUND);

  std::string netdev = "faketh0";
  std::string name = "1";
  std::string ib_name = "fakerdma0";
  std::string ib_path = "/tmp";
  SmiInfiniBand infiniband(ib_name, ib_path);
  infiniband.add_port(SmiInfiniBandPort(netdev, name, ib_path));
  InterfaceFixture_t rdma_fixture("/tmp", transport, &kTestTable, &infiniband);
  check("an rdma count past the last rdma port is not found",
        smi_get_nic_rdma_port_statistics_count(rdma_fixture.ctx, kFakeNicBdf, 0, 0, kPastLastPort,
                                               &count) == SMI_NIC_STATUS_NOT_FOUND);
  check("an rdma list past the last rdma port is not found",
        smi_get_nic_rdma_port_statistics_list(rdma_fixture.ctx, kFakeNicBdf, 0, 0, kPastLastPort,
                                              info.get()) == SMI_NIC_STATUS_NOT_FOUND);
}

// A counter the kernel stops exposing must not linger at its last value.
static void test_standard_statistics_drop_a_counter_that_disappears() {
  namespace fs = std::filesystem;
  const fs::path root =
      fs::temp_directory_path() / ("amdsmi-std-gone-" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "statistics");
  write_counter(root / "statistics", "rx_bytes", 7);
  write_counter(root / "statistics", "tx_bytes", 9);
  SmiNicPort port("faketh0", "0000:00:00.0", root.string(), root.string());

  port.collect_standard_statistics();
  check("both counters are read at first", port.get_standard_stats_map().size() == 2,
        std::to_string(port.get_standard_stats_map().size()));

  fs::remove(root / "statistics" / "tx_bytes");
  port.collect_standard_statistics();
  check("a counter the kernel stops exposing disappears",
        port.get_standard_stats_map().count("tx_bytes") == 0);
  check("the counter that remains is still read",
        port.get_standard_stats_map().count("rx_bytes") == 1);
  fs::remove_all(root);
}

// Two ports on one NIC: ethA (0000:03:00.1) has no driver bound, ethB (0000:03:00.2) is healthy.
struct UnboundPortFixture_t {
  UnboundPortFixture_t() {
    auto transport_a = std::make_shared<FakeTransport>();
    transport_a->stats = {true, {{"tx_packets"}, {111}}, 0};
    auto transport_b = std::make_shared<FakeTransport>();
    transport_b->stats = {true, {{"tx_packets"}, {222}}, 0};

    auto nic = std::make_unique<SmiNic>("ethA", "0000:03:00.0");
    nic->add_nic_port(SmiNicPort("ethA", "0000:03:00.1", "/tmp", "/tmp", transport_a, &kTestTable));
    nic->add_nic_port(SmiNicPort("ethB", "0000:03:00.2", "/tmp", "/tmp", transport_b, &kTestTable));
    auto subsystem = std::make_unique<FakeSubsystem>();
    subsystem->unbind("0000:03:00.1");
    subsystem->add(std::move(nic));
    auto system = std::make_unique<SmiNicSystem>("/tmp", "/tmp");
    system->register_subsystem(std::move(subsystem));
    system->discover_nics();
    smi_nic_create_context_for_testing(&ctx, std::move(system));
  }
  ~UnboundPortFixture_t() {
    if (ctx != nullptr) {
      smi_nic_destroy_context(ctx);
    }
  }
  UnboundPortFixture_t(const UnboundPortFixture_t&) = delete;
  UnboundPortFixture_t& operator=(const UnboundPortFixture_t&) = delete;

  smi_nic_ctx_t ctx = nullptr;
};

static void test_interface_port_info_index_names_the_port_that_statistics_read() {
  UnboundPortFixture_t fixture;

  auto info = std::make_unique<smi_nic_port_info_t>();
  check("port info succeeds with one port unbound",
        smi_get_nic_port_info(fixture.ctx, kFakeNicBdf, info.get()) == SMI_NIC_STATUS_SUCCESS);
  check("port info keeps the unbound port in its slot", info->num_ports == 2);
  check("info index 0 is ethA", std::string(info->ports[0].netdev) == "ethA");
  check("info index 1 is ethB", std::string(info->ports[1].netdev) == "ethB");
  check("an unbound port reports N/A link state", std::string(info->ports[0].link_state) == "N/A");

  auto count = uint32_t{0};
  check("statistics on the unbound index report the driver as not loaded",
        smi_get_nic_vendor_statistics_count(fixture.ctx, kFakeNicBdf, 0, SMI_NIC_STAT_SCOPE_DEFAULT,
                                            &count) == SMI_NIC_STATUS_DRIVER_NOT_LOADED);

  auto stats = std::make_unique<smi_nic_stat_info_t>();
  check("statistics on index 1 succeed",
        smi_get_nic_vendor_statistics_list(fixture.ctx, kFakeNicBdf, 1, SMI_NIC_STAT_SCOPE_DEFAULT,
                                           stats.get()) == SMI_NIC_STATUS_SUCCESS);
  check("statistics on index 1 are ethB's counters", stats->stats[0].value == 222);
}

static void test_interface_port_info_reports_driver_not_loaded_when_every_port_is_unbound() {
  // The only port has no driver bound.
  auto nic = std::make_unique<SmiNic>("ethA", "0000:03:00.0");
  nic->add_nic_port(SmiNicPort("ethA", "0000:03:00.1", "/tmp", "/tmp",
                               std::make_shared<FakeTransport>(), &kTestTable));
  auto subsystem = std::make_unique<FakeSubsystem>();
  subsystem->unbind("0000:03:00.1");
  subsystem->add(std::move(nic));
  auto system = std::make_unique<SmiNicSystem>("/tmp", "/tmp");
  system->register_subsystem(std::move(subsystem));
  system->discover_nics();
  auto ctx = smi_nic_ctx_t{nullptr};
  smi_nic_create_context_for_testing(&ctx, std::move(system));

  auto info = std::make_unique<smi_nic_port_info_t>();
  check("port info reports DRIVER_NOT_LOADED when no port has a driver",
        smi_get_nic_port_info(ctx, kFakeNicBdf, info.get()) == SMI_NIC_STATUS_DRIVER_NOT_LOADED);
  smi_nic_destroy_context(ctx);
}

int main() {
  test_default_scope_excludes_extended();
  test_extended_scope_is_superset();
  test_names_outside_table_are_dropped();
  test_refresh_sees_changed_counters();
  test_fec_merge_ionic_shape();
  test_fec_counters_refresh_on_subsequent_collect();
  test_fec_failure_is_not_an_error();
  test_fec_query_is_logged_with_interface_and_outcome();
  test_rdma_hw_counters_skip_lifespan_and_refresh();
  test_standard_statistics_survive_sysfs_errors();
  test_rdma_hw_counters_survive_sysfs_errors();
  test_no_table_yields_empty_map();
  test_large_table_not_truncated_by_mechanism();
  test_real_tables_fit_within_cap();
  test_interface_reads_vendor_statistics_from_a_prepared_system();
  test_interface_reports_a_failed_vendor_read_as_an_error();
  test_interface_treats_an_unsupported_vendor_read_as_no_counters();
  test_interface_count_matches_the_cap_on_the_list();
  test_interface_rdma_count_matches_the_cap_on_the_list();
  test_interface_vendor_list_of_nothing_succeeds_with_zero();
  test_interface_port_list_of_nothing_succeeds_with_zero();
  test_interface_rdma_list_of_nothing_succeeds_with_zero();
  test_interface_out_of_range_index_reads_the_same_from_count_and_list();
  test_standard_statistics_drop_a_counter_that_disappears();
  test_interface_port_info_index_names_the_port_that_statistics_read();
  test_interface_port_info_reports_driver_not_loaded_when_every_port_is_unbound();

  return nic_test::report();
}
