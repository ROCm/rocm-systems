// SPDX-License-Identifier: MIT
/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * Unit tests for the tiered vendor-stat mechanism (StatTable_t filtering,
 * query-time refresh, FEC merge). No hardware, no root: the transport is a
 * fake returning canned VendorStatistics/FecStatistics_t.
 */

#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "amd_smi/impl/nic/amd_smi_nic_interface.h"
#include "nic_check.h"
#include "smi_ibverbs.h"
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

  auto port = make_port(transport, nullptr);  // UALoE shape: no ports, no table
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
  bool driver_loaded(const std::string& bdf, DriverType) const override {
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
        smi_get_nic_vendor_statistics_count(fixture.ctx, kFakeNicBdf, 0, &count) ==
            SMI_NIC_STATUS_SUCCESS);
  check("vendor count sees both counters", count == 2, std::to_string(count));

  auto info = std::make_unique<smi_nic_stat_info_t>();
  check("vendor list succeeds",
        smi_get_nic_vendor_statistics_list(fixture.ctx, kFakeNicBdf, 0, info.get()) ==
            SMI_NIC_STATUS_SUCCESS);
  check("vendor list carries both counters", info->count == 2, std::to_string(info->count));
}

static void test_interface_link_type_reports_unknown_for_a_device_outside_sysfs() {
  // dead:be:1f.7 is in no real sysfs, so the answer is the unknown link, not an error.
  constexpr auto kOutsideSysfsBdf = uint64_t{0xdeadbeff};
  auto nic =
      std::make_unique<SmiNic>("ethA", "dead:be:1f.7", NicType::Ethernet, "", "", NicVendor::AMD);
  auto subsystem = std::make_unique<FakeSubsystem>();
  subsystem->add(std::move(nic));
  auto system = std::make_unique<SmiNicSystem>("/tmp", "/tmp");
  system->register_subsystem(std::move(subsystem));
  system->discover_nics();
  auto ctx = smi_nic_ctx_t{nullptr};
  smi_nic_create_context_for_testing(&ctx, std::move(system));

  auto type = SMI_NIC_LINK_TYPE_PCIE;
  check("a NIC without a port still answers",
        smi_topo_get_nic_link_type(ctx, kOutsideSysfsBdf, 0x400, &type) == SMI_NIC_STATUS_SUCCESS);
  check("the link is unknown", type == SMI_NIC_LINK_TYPE_UNKNOWN);
  check("an unregistered source is not found",
        smi_topo_get_nic_link_type(ctx, 0xffffffffffffffffULL, 0x400, &type) ==
            SMI_NIC_STATUS_NOT_FOUND);
  check("no out parameter is rejected",
        smi_topo_get_nic_link_type(ctx, kOutsideSysfsBdf, 0x400, nullptr) ==
            SMI_NIC_STATUS_WRONG_PARAM);
  check("a null context is rejected",
        smi_topo_get_nic_link_type(nullptr, kOutsideSysfsBdf, 0x400, &type) ==
            SMI_NIC_STATUS_WRONG_PARAM);
  smi_nic_destroy_context(ctx);
}

static void test_interface_reports_each_nics_kind_and_capability() {
  auto ainic = std::make_unique<SmiNic>("ethA", "0000:03:00.0", NicType::Ethernet, "", "",
                                        NicVendor::AMD, NicProduct::AINIC);
  ainic->add_nic_port(SmiNicPort("ethA", "0000:03:00.1", "/tmp", "/tmp",
                                 std::make_shared<FakeTransport>(), &kTestTable));
  auto ualoe = std::make_unique<SmiNic>("", "0000:04:00.0", NicType::Fabric, "", "", NicVendor::AMD,
                                        NicProduct::AINIC);
  auto broadcom = std::make_unique<SmiNic>("ethC", "0000:05:00.0", NicType::Ethernet, "", "",
                                           NicVendor::Broadcom);
  broadcom->add_nic_port(SmiNicPort("ethC", "0000:05:00.1", "/tmp", "/tmp",
                                    std::make_shared<FakeTransport>(), &kTestTable));
  auto subsystem = std::make_unique<FakeSubsystem>();
  subsystem->add(std::move(ainic));
  subsystem->add(std::move(ualoe));
  subsystem->add(std::move(broadcom));
  auto system = std::make_unique<SmiNicSystem>("/tmp", "/tmp");
  system->register_subsystem(std::move(subsystem));
  system->discover_nics();
  auto ctx = smi_nic_ctx_t{nullptr};
  smi_nic_create_context_for_testing(&ctx, std::move(system));

  constexpr auto kAinicBdf = uint64_t{0x300};
  constexpr auto kUaloeBdf = uint64_t{0x400};
  constexpr auto kBroadcomBdf = uint64_t{0x500};
  auto kind = uint8_t{0xff};
  auto capability = uint32_t{0xffffffff};

  check("kind of an AINIC", (smi_get_nic_kind(ctx, kAinicBdf, &kind) == SMI_NIC_STATUS_SUCCESS) &&
                                (kind == SMI_NIC_KIND_AINIC));
  check("kind of a UALoE endpoint",
        (smi_get_nic_kind(ctx, kUaloeBdf, &kind) == SMI_NIC_STATUS_SUCCESS) &&
            (kind == SMI_NIC_KIND_UALOE));
  check("kind of another vendor",
        (smi_get_nic_kind(ctx, kBroadcomBdf, &kind) == SMI_NIC_STATUS_SUCCESS) &&
            (kind == SMI_NIC_KIND_OTHER));
  check("a NIC with a port has the netdev capability",
        (smi_get_nic_capability(ctx, kAinicBdf, &capability) == SMI_NIC_STATUS_SUCCESS) &&
            (capability == SMI_NIC_CAP_NETDEV));
  check("a NIC without a port has no capability",
        (smi_get_nic_capability(ctx, kUaloeBdf, &capability) == SMI_NIC_STATUS_SUCCESS) &&
            (capability == 0));
  check("kind of an unregistered BDF is not found",
        smi_get_nic_kind(ctx, 0xffffffffffffffffULL, &kind) == SMI_NIC_STATUS_NOT_FOUND);
  check(
      "capability of an unregistered BDF is not found",
      smi_get_nic_capability(ctx, 0xffffffffffffffffULL, &capability) == SMI_NIC_STATUS_NOT_FOUND);
  check("kind with no out parameter is rejected",
        smi_get_nic_kind(ctx, kAinicBdf, nullptr) == SMI_NIC_STATUS_WRONG_PARAM);
  check("capability with no out parameter is rejected",
        smi_get_nic_capability(ctx, kAinicBdf, nullptr) == SMI_NIC_STATUS_WRONG_PARAM);
  check("kind on a null context is rejected",
        smi_get_nic_kind(nullptr, kAinicBdf, &kind) == SMI_NIC_STATUS_WRONG_PARAM);
  smi_nic_destroy_context(ctx);
}

static void test_interface_vendor_statistics_scope_is_chosen_per_call() {
  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {true, {{"tx_packets", "rx_packets", "tx_bytes"}, {1, 2, 3}}, 0};
  InterfaceFixture_t fixture("/tmp", transport, &kTestTable);

  auto default_count = uint32_t{0};
  auto extended_count = uint32_t{0};
  auto unscoped_count = uint32_t{0};
  check("default scope count succeeds", smi_get_nic_vendor_statistics_count_with_scope(
                                            fixture.ctx, kFakeNicBdf, 0, SMI_NIC_STAT_SCOPE_DEFAULT,
                                            &default_count) == SMI_NIC_STATUS_SUCCESS);
  check("extended scope count succeeds",
        smi_get_nic_vendor_statistics_count_with_scope(fixture.ctx, kFakeNicBdf, 0,
                                                       SMI_NIC_STAT_SCOPE_EXTENDED,
                                                       &extended_count) == SMI_NIC_STATUS_SUCCESS);
  check("unscoped count succeeds",
        smi_get_nic_vendor_statistics_count(fixture.ctx, kFakeNicBdf, 0, &unscoped_count) ==
            SMI_NIC_STATUS_SUCCESS);
  check("default scope excludes the extended counter", default_count == 2,
        std::to_string(default_count));
  check("extended scope adds the extended counter", extended_count == 3,
        std::to_string(extended_count));
  check("the unscoped call reads the default tier", unscoped_count == default_count);

  auto info = std::make_unique<smi_nic_stat_info_t>();
  check("extended scope list succeeds",
        smi_get_nic_vendor_statistics_list_with_scope(fixture.ctx, kFakeNicBdf, 0,
                                                      SMI_NIC_STAT_SCOPE_EXTENDED,
                                                      info.get()) == SMI_NIC_STATUS_SUCCESS);
  check("extended scope list carries all three counters", info->count == 3,
        std::to_string(info->count));
  check("the next call is not affected by the previous scope",
        (smi_get_nic_vendor_statistics_list(fixture.ctx, kFakeNicBdf, 0, info.get()) ==
         SMI_NIC_STATUS_SUCCESS) &&
            (info->count == 2));
}

static void test_interface_reports_a_failed_vendor_read_as_an_error() {
  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {false, {}, EPERM};
  InterfaceFixture_t fixture("/tmp", transport, &kTestTable);

  auto count = uint32_t{0};
  check("a failed vendor read is an error, not zero counters",
        smi_get_nic_vendor_statistics_count(fixture.ctx, kFakeNicBdf, 0, &count) ==
            SMI_NIC_STATUS_ERROR);

  auto info = std::make_unique<smi_nic_stat_info_t>();
  check("a failed vendor read is an error from the list too",
        smi_get_nic_vendor_statistics_list(fixture.ctx, kFakeNicBdf, 0, info.get()) ==
            SMI_NIC_STATUS_ERROR);
}

// A driver without ethtool statistics has none to report; that is not a failure.
static void test_interface_treats_an_unsupported_vendor_read_as_no_counters() {
  auto transport = std::make_shared<FakeTransport>();
  transport->stats = {false, {}, ENOTSUP};
  InterfaceFixture_t fixture("/tmp", transport, &kTestTable);

  auto count = uint32_t{99};
  check("an unsupported vendor read still succeeds",
        smi_get_nic_vendor_statistics_count(fixture.ctx, kFakeNicBdf, 0, &count) ==
            SMI_NIC_STATUS_SUCCESS);
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

// A vendor RDMA provider can crash or hang while a device is opened. The probe runs in a child
// process, so the failure costs the MTU and nothing else.
static void test_isolated_probe_returns_the_child_answer() {
  const auto mtu = smi_ibverbs::run_isolated(
      []() -> std::optional<smi_ibverbs::PortMtu> { return smi_ibverbs::PortMtu{1024, 4096}; },
      std::chrono::milliseconds(2000));
  check("the child's MTU reaches the parent",
        mtu.has_value() && mtu->active_mtu == 1024 && mtu->max_mtu == 4096);

  const auto none = smi_ibverbs::run_isolated(
      []() -> std::optional<smi_ibverbs::PortMtu> { return std::nullopt; },
      std::chrono::milliseconds(2000));
  check("a probe without an answer gives no MTU", !none.has_value());
}

static void test_isolated_probe_survives_a_crashing_provider() {
  const auto mtu = smi_ibverbs::run_isolated(
      []() -> std::optional<smi_ibverbs::PortMtu> {
        ::raise(SIGSEGV);
        return smi_ibverbs::PortMtu{1024, 4096};
      },
      std::chrono::milliseconds(2000));
  check("a crash in the probe leaves the caller alive with no MTU", !mtu.has_value());
}

// A child killed by a signal is logged by the kernel and by systemd-coredump on every init. The
// child must end itself instead, which also keeps the crash out of the system logs.
static void test_isolated_probe_ends_the_child_itself_on_a_fatal_signal() {
  const auto mtu = smi_ibverbs::run_isolated(
      []() -> std::optional<smi_ibverbs::PortMtu> {
        for (const int signal_number : {SIGSEGV, SIGBUS, SIGABRT, SIGILL, SIGFPE}) {
          struct sigaction current = {};
          ::sigaction(signal_number, nullptr, &current);
          if (current.sa_handler == SIG_DFL) {
            return std::nullopt;
          }
        }
        return smi_ibverbs::PortMtu{1, 1};
      },
      std::chrono::milliseconds(2000));
  check("the child handles every fatal signal itself", mtu.has_value());
}

static void test_isolated_probe_gives_up_on_a_hung_provider_and_reaps_it() {
  const auto start = std::chrono::steady_clock::now();
  const auto mtu = smi_ibverbs::run_isolated(
      []() -> std::optional<smi_ibverbs::PortMtu> {
        ::sleep(60);
        return smi_ibverbs::PortMtu{1024, 4096};
      },
      std::chrono::milliseconds(200));
  const auto elapsed = std::chrono::steady_clock::now() - start;
  check("a hung probe gives no MTU", !mtu.has_value());
  check("a hung probe is abandoned at the timeout", elapsed < std::chrono::seconds(5));

  auto status = int{0};
  errno = 0;
  check("the killed child is reaped, not left as a zombie",
        (::waitpid(-1, &status, WNOHANG) == -1) && (errno == ECHILD));
}

// The MTU of an RDMA port comes from libibverbs; the sysfs files the kernel may expose are the
// fallback when the query has no answer.
static void test_infiniband_port_mtu_comes_from_the_ibverbs_query() {
  const auto root =
      std::filesystem::temp_directory_path() / ("amdsmi-ibv-mtu-" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  const auto port_dir = root / "infiniband/fakerdma0/ports/1";
  std::filesystem::create_directories(port_dir);

  auto asked_device = std::string{};
  auto asked_port = uint8_t{0};
  smi_nic_set_port_mtu_query_for_testing(
      [&](const std::string& rdma_dev, uint8_t port_num) -> std::optional<smi_ibverbs::PortMtu> {
        asked_device = rdma_dev;
        asked_port = port_num;
        return smi_ibverbs::PortMtu{1024, 4096};
      });
  auto netdev = std::string{"faketh0"};
  auto name = std::string{"1"};
  const auto port = SmiInfiniBandPort(netdev, name, port_dir.string());

  check("the query names the RDMA device of the port", asked_device == "fakerdma0", asked_device);
  check("the query names the port number", asked_port == 1, std::to_string(asked_port));
  check("max MTU comes from ibverbs", port.max_mtu() == std::optional<uint16_t>(4096));
  check("active MTU comes from ibverbs", port.active_mtu() == std::optional<uint16_t>(1024));

  smi_nic_set_port_mtu_query_for_testing({});
  std::filesystem::remove_all(root);
}

static void test_infiniband_port_mtu_falls_back_to_sysfs_without_an_ibverbs_answer() {
  const auto root = std::filesystem::temp_directory_path() /
                    ("amdsmi-ibv-fallback-" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  const auto port_dir = root / "infiniband/fakerdma0/ports/1";
  std::filesystem::create_directories(port_dir);
  smi_nic_set_port_mtu_query_for_testing(
      [](const std::string&, uint8_t) -> std::optional<smi_ibverbs::PortMtu> {
        return std::nullopt;
      });
  auto netdev = std::string{"faketh0"};
  auto name = std::string{"1"};

  const auto without_files = SmiInfiniBandPort(netdev, name, port_dir.string());
  check("no ibverbs answer and no sysfs file leaves the MTU unknown",
        !without_files.max_mtu().has_value() && !without_files.active_mtu().has_value());

  std::ofstream(port_dir / "max_mtu") << "2048\n";
  std::ofstream(port_dir / "active_mtu") << "1024\n";
  const auto with_files = SmiInfiniBandPort(netdev, name, port_dir.string());
  check("the sysfs max MTU is the fallback", with_files.max_mtu() == std::optional<uint16_t>(2048));
  check("the sysfs active MTU is the fallback",
        with_files.active_mtu() == std::optional<uint16_t>(1024));

  smi_nic_set_port_mtu_query_for_testing({});
  std::filesystem::remove_all(root);
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
        smi_get_nic_vendor_statistics_count(fixture.ctx, kFakeNicBdf, 0, &count) ==
            SMI_NIC_STATUS_SUCCESS);
  check("an empty vendor count is zero", count == 0, std::to_string(count));

  auto info = std::make_unique<smi_nic_stat_info_t>();
  info->count = 99;
  check("an empty vendor list succeeds like the count",
        smi_get_nic_vendor_statistics_list(fixture.ctx, kFakeNicBdf, 0, info.get()) ==
            SMI_NIC_STATUS_SUCCESS);
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
        smi_get_nic_vendor_statistics_count(fixture.ctx, kFakeNicBdf, 0, &count) ==
            SMI_NIC_STATUS_DRIVER_NOT_LOADED);

  auto stats = std::make_unique<smi_nic_stat_info_t>();
  check("statistics on index 1 succeed",
        smi_get_nic_vendor_statistics_list(fixture.ctx, kFakeNicBdf, 1, stats.get()) ==
            SMI_NIC_STATUS_SUCCESS);
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

static void test_interface_discovery_reports_each_nics_vendor() {
  auto amd_nic =
      std::make_unique<SmiNic>("ethA", "0000:03:00.0", NicType::Ethernet, "", "", NicVendor::AMD);
  auto brcm_nic = std::make_unique<SmiNic>("ethB", "0000:04:00.0", NicType::Ethernet, "", "",
                                           NicVendor::Broadcom);
  auto subsystem = std::make_unique<FakeSubsystem>();
  subsystem->add(std::move(amd_nic));
  subsystem->add(std::move(brcm_nic));
  auto system = std::make_unique<SmiNicSystem>("/tmp", "/tmp");
  system->register_subsystem(std::move(subsystem));
  system->discover_nics();
  auto ctx = smi_nic_ctx_t{nullptr};
  smi_nic_create_context_for_testing(&ctx, std::move(system));

  auto discovery = std::make_unique<smi_nic_discovery_t>();
  check("discovery succeeds", smi_discover_nics(ctx, discovery.get()) == SMI_NIC_STATUS_SUCCESS);
  check("discovery lists both NICs", discovery->count == 2);

  auto vendor_of = [&](const std::string& bdf) {
    for (uint32_t i = 0; i < discovery->count; ++i) {
      if (bdf == discovery->devices[i].bdf) {
        return discovery->devices[i].vendor;
      }
    }
    return SMI_NIC_VENDOR_UNKNOWN;
  };
  check("an AMD NIC reports the AMD vendor", vendor_of("0000:03:00.0") == SMI_NIC_VENDOR_AMD);
  check("a Broadcom NIC reports the Broadcom vendor",
        vendor_of("0000:04:00.0") == SMI_NIC_VENDOR_BROADCOM);
  smi_nic_destroy_context(ctx);
}

// A sysfs class dir holding only an ifindex file.
class IfindexDir_t {
 public:
  explicit IfindexDir_t(uint32_t ifindex)
      : m_path(std::filesystem::temp_directory_path() /
               ("amdsmi-ifindex-" + std::to_string(::getpid()) + "-" + std::to_string(ifindex))) {
    std::filesystem::remove_all(m_path);
    std::filesystem::create_directories(m_path);
    std::ofstream(m_path / "ifindex") << ifindex << "\n";
  }
  ~IfindexDir_t() { std::filesystem::remove_all(m_path); }
  IfindexDir_t(const IfindexDir_t&) = delete;
  IfindexDir_t& operator=(const IfindexDir_t&) = delete;

  std::string path() const { return m_path.string(); }

 private:
  std::filesystem::path m_path;
};

// The public port struct keeps ifindex in one byte, as in the previous release.
constexpr auto kMaxByteIfindex = uint32_t{255};

static uint32_t read_ifindex_from_port_info(uint32_t ifindex) {
  IfindexDir_t dir(ifindex);
  InterfaceFixture_t fixture(dir.path(), std::make_shared<FakeTransport>(), &kTestTable);
  auto info = std::make_unique<smi_nic_port_info_t>();
  check("port info succeeds for ifindex " + std::to_string(ifindex),
        smi_get_nic_port_info(fixture.ctx, kFakeNicBdf, info.get()) == SMI_NIC_STATUS_SUCCESS);
  return info->ports[0].ifindex;
}

static void test_interface_port_info_keeps_an_ifindex_that_fits_a_byte() {
  check("an ifindex of 255 reads 255 in the port struct",
        read_ifindex_from_port_info(kMaxByteIfindex) == kMaxByteIfindex);
}

// 300 and 511 truncate to 44 and 255, so a plain cast to one byte cannot pass for the clamp.
static void test_interface_port_info_reads_zero_for_an_ifindex_above_a_byte() {
  for (const auto ifindex : {kMaxByteIfindex + 1, uint32_t{300}, uint32_t{511}}) {
    check("an ifindex of " + std::to_string(ifindex) + " reads 0 in the port struct",
          read_ifindex_from_port_info(ifindex) == 0);
  }
}

static void test_interface_port_ifindex_returns_the_full_value() {
  const auto ifindex = kMaxByteIfindex + 45;
  IfindexDir_t dir(ifindex);
  InterfaceFixture_t fixture(dir.path(), std::make_shared<FakeTransport>(), &kTestTable);

  auto value = uint32_t{0};
  check("port ifindex succeeds",
        smi_get_nic_port_ifindex(fixture.ctx, kFakeNicBdf, 0, &value) == SMI_NIC_STATUS_SUCCESS);
  check("port ifindex returns a value above a byte whole", value == ifindex, std::to_string(value));
}

// Two ports on one NIC, each with its own sysfs class dir.
class TwoPortIfindexFixture_t {
 public:
  TwoPortIfindexFixture_t(const std::string& first_class_path,
                          const std::string& second_class_path) {
    auto nic = std::make_unique<SmiNic>("faketh0", "0000:03:00.0");
    nic->add_nic_port(SmiNicPort("faketh0", "0000:03:00.1", first_class_path, "/tmp",
                                 std::make_shared<FakeTransport>(), &kTestTable));
    nic->add_nic_port(SmiNicPort("faketh1", "0000:03:00.2", second_class_path, "/tmp",
                                 std::make_shared<FakeTransport>(), &kTestTable));
    auto subsystem = std::make_unique<FakeSubsystem>();
    subsystem->add(std::move(nic));
    auto system = std::make_unique<SmiNicSystem>("/tmp", "/tmp");
    system->register_subsystem(std::move(subsystem));
    system->discover_nics();
    smi_nic_create_context_for_testing(&m_ctx, std::move(system));
  }
  ~TwoPortIfindexFixture_t() {
    if (m_ctx != nullptr) {
      smi_nic_destroy_context(m_ctx);
    }
  }
  TwoPortIfindexFixture_t(const TwoPortIfindexFixture_t&) = delete;
  TwoPortIfindexFixture_t& operator=(const TwoPortIfindexFixture_t&) = delete;

  smi_nic_ctx_t ctx() const { return m_ctx; }

 private:
  smi_nic_ctx_t m_ctx = nullptr;
};

static void test_interface_port_ifindex_follows_the_port_index() {
  const auto first_ifindex = kMaxByteIfindex + 45;
  const auto second_ifindex = first_ifindex + 1;
  IfindexDir_t first_dir(first_ifindex);
  IfindexDir_t second_dir(second_ifindex);
  TwoPortIfindexFixture_t fixture(first_dir.path(), second_dir.path());

  auto first = uint32_t{0};
  auto second = uint32_t{0};
  check("port 0 ifindex succeeds",
        smi_get_nic_port_ifindex(fixture.ctx(), kFakeNicBdf, 0, &first) == SMI_NIC_STATUS_SUCCESS);
  check("port 1 ifindex succeeds",
        smi_get_nic_port_ifindex(fixture.ctx(), kFakeNicBdf, 1, &second) == SMI_NIC_STATUS_SUCCESS);
  check("port 0 reads its own ifindex", first == first_ifindex, std::to_string(first));
  check("port 1 reads its own ifindex", second == second_ifindex, std::to_string(second));
}

static void test_interface_port_ifindex_rejects_a_bad_request() {
  IfindexDir_t dir(1);
  InterfaceFixture_t fixture(dir.path(), std::make_shared<FakeTransport>(), &kTestTable);

  auto value = uint32_t{0};
  check("port ifindex past the last port is not found",
        smi_get_nic_port_ifindex(fixture.ctx, kFakeNicBdf, 1, &value) == SMI_NIC_STATUS_NOT_FOUND);
  check(
      "port ifindex needs an output pointer",
      smi_get_nic_port_ifindex(fixture.ctx, kFakeNicBdf, 0, nullptr) == SMI_NIC_STATUS_WRONG_PARAM);
  check("port ifindex needs a context",
        smi_get_nic_port_ifindex(nullptr, kFakeNicBdf, 0, &value) == SMI_NIC_STATUS_WRONG_PARAM);
  check("port ifindex on an unknown NIC is not found",
        smi_get_nic_port_ifindex(fixture.ctx, kFakeNicBdf + 1, 0, &value) ==
            SMI_NIC_STATUS_NOT_FOUND);
}

int main() {
  test_interface_discovery_reports_each_nics_vendor();
  test_isolated_probe_returns_the_child_answer();
  test_isolated_probe_survives_a_crashing_provider();
  test_isolated_probe_ends_the_child_itself_on_a_fatal_signal();
  test_isolated_probe_gives_up_on_a_hung_provider_and_reaps_it();
  test_infiniband_port_mtu_comes_from_the_ibverbs_query();
  test_infiniband_port_mtu_falls_back_to_sysfs_without_an_ibverbs_answer();
  test_interface_link_type_reports_unknown_for_a_device_outside_sysfs();
  test_interface_reports_each_nics_kind_and_capability();
  test_interface_vendor_statistics_scope_is_chosen_per_call();
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
  test_interface_port_info_keeps_an_ifindex_that_fits_a_byte();
  test_interface_port_info_reads_zero_for_an_ifindex_above_a_byte();
  test_interface_port_ifindex_returns_the_full_value();
  test_interface_port_ifindex_follows_the_port_index();
  test_interface_port_ifindex_rejects_a_bad_request();

  return nic_test::report();
}
