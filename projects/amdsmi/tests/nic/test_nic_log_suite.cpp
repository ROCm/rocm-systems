// SPDX-License-Identifier: MIT
/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * Unit tests for the NIC debug-log seam and the read chokepoints that feed it.
 * No hardware, no root: reads go against real files in a tmpdir and the sink is
 * a capture function standing in for libamd_smi's LOG_DEBUG adapter.
 */

#include <linux/ethtool.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef HAVE_LIBNL3
#include <linux/genetlink.h>
#include <netlink/genl/ctrl.h>

#include "netlink_generic.h"
#include "smi_devlink_netlink.h"
#endif
#include "smi_ethtool_ioctl.h"
#include "smi_nic.h"
#include "smi_nic_log.h"
#include "smi_nic_subsystem.h"
#include "smi_nic_transport.h"
#include "smi_sysfs.h"

namespace fs = std::filesystem;
namespace nlog = amd::smi::nic::log;
namespace tp = amd::smi::nic::transport;

namespace {

const auto kMacText = std::string{"04:90:81:b4:ee:e0"};
const auto kSerialText = std::string{"FPK2615006E"};
const auto kSerialMasked = std::string{"****006E"};
const auto kTempMilliC = std::string{"26000"};
const auto kIface = std::string{"eth0"};
constexpr auto kLinkSpeedMbps = uint32_t{400000};
constexpr auto kMacLastByte = uint8_t{0xe0};
// A generic-netlink family id the kernel never assigns.
constexpr auto kBogusFamilyId = 0x7ffe;

auto tests_run = 0;
auto tests_failed = 0;

auto check(const std::string& name, bool passed, const std::string& detail = "") -> void {
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

auto g_captured = std::vector<std::string>{};

auto capture_sink(const std::string& msg) -> void { g_captured.push_back(msg); }

auto has_substr(const std::string& haystack, const std::string& needle) -> bool {
  return (haystack.find(needle) != std::string::npos);
}

auto joined() -> std::string {
  auto all = std::string{};
  for (const auto& line : g_captured) {
    all += line + "\n";
  }
  return all;
}

// RAII tmpdir plus sink installation, so a failing check never leaks state.
class LogFixture_t {
 public:
  LogFixture_t() {
    g_captured.clear();
    m_dir = fs::temp_directory_path() / ("nic_log_suite_" + std::to_string(::getpid()));
    fs::create_directories(m_dir);
    nlog::set_sink(capture_sink);
  }
  ~LogFixture_t() {
    nlog::set_sink(nullptr);
    auto ec = std::error_code{};
    fs::remove_all(m_dir, ec);
  }
  auto write(const std::string& name, const std::string& content) const -> std::string {
    const auto path = m_dir / name;
    std::ofstream(path) << content;
    return path.string();
  }
  auto path_of(const std::string& name) const -> std::string { return (m_dir / name).string(); }

 private:
  fs::path m_dir;
};

// snprintf always NUL-terminates, unlike strncpy at sizeof(dst) - 1 on a non-zeroed buffer.
template <size_t N>
auto set_cstr(char (&dst)[N], const std::string& src) -> void {
  std::snprintf(dst, N, "%s", src.c_str());
}

auto test_no_sink_is_disabled_and_silent() -> void {
  std::cout << "\nSeam: no sink installed\n";
  g_captured.clear();
  nlog::set_sink(nullptr);

  check("is_enabled is false", !nlog::is_enabled());
  nlog::debug("void fn(int)", "dropped");
  check("debug is a no-op", g_captured.empty());
}

auto test_sink_receives_message() -> void {
  std::cout << "\nSeam: sink installed\n";
  const auto fx = LogFixture_t{};

  check("is_enabled is true", nlog::is_enabled());
  nlog::debug("void fn(int)", "hello");
  check("one message delivered", (g_captured.size() == 1));
  check("message is function | text",
        (!g_captured.empty() && (g_captured[0] == "void fn(int) | hello")));
}

auto test_mask_tail() -> void {
  std::cout << "\nSeam: mask_tail keeps only the last 4 characters\n";

  check("serial masked", (nlog::mask_tail(kSerialText) == kSerialMasked),
        "got " + nlog::mask_tail(kSerialText));
  check("short value fully masked", (nlog::mask_tail("ab") == "****"));
  check("empty stays empty", nlog::mask_tail("").empty());
}

auto test_sysfs_success_logged() -> void {
  std::cout << "\nSysfs: successful read logs full path, status and value\n";
  const auto fx = LogFixture_t{};
  const auto path = fx.write("temp1_input", kTempMilliC + "\n");

  auto value = SmiSysfsReader::SysfsValue{};
  const auto status = SmiSysfsReader::readLine(path, value);

  check("read still succeeds", (status == SmiSysfsReader::SysfsStatus::Success));
  check("one line logged", (g_captured.size() == 1), "got " + std::to_string(g_captured.size()));
  const auto all = joined();
  check("full path present", has_substr(all, path), all);
  check("SUCCESS present", has_substr(all, "SUCCESS"), all);
  check("value present", has_substr(all, kTempMilliC), all);
  check("line names the reader function",
        has_substr(all, "readLine") && has_substr(all, " | sysfs read "), all);
}

auto test_sysfs_missing_file_logged() -> void {
  std::cout << "\nSysfs: missing file logs full path and FAIL\n";
  const auto fx = LogFixture_t{};
  const auto path = fx.path_of("hwmon");

  auto value = SmiSysfsReader::SysfsValue{};
  const auto status = SmiSysfsReader::readLine(path, value);

  check("read reports FileNotFound", (status == SmiSysfsReader::SysfsStatus::FileNotFound));
  const auto all = joined();
  check("full path present", has_substr(all, path), all);
  check("FAIL present", has_substr(all, "FAIL"), all);
  check("no SUCCESS", !has_substr(all, "SUCCESS"), all);
}

auto test_sysfs_mac_is_masked() -> void {
  std::cout << "\nSysfs: MAC read from an 'address' file is masked in the log\n";
  const auto fx = LogFixture_t{};
  const auto path = fx.write("address", kMacText + "\n");

  auto value = SmiSysfsReader::SysfsValue{};
  SmiSysfsReader::readLine(path, value);

  const auto all = joined();
  check("path still logged", has_substr(all, path), all);
  check("full MAC absent from log", !has_substr(all, kMacText), all);
  check("masked tail present", has_substr(all, nlog::mask_tail(kMacText)), all);
  check("value itself unchanged for the caller",
        (std::holds_alternative<std::string>(value) && (std::get<std::string>(value) == kMacText)));
}

auto test_sysfs_logs_raw_text_not_parsed_number() -> void {
  std::cout << "\nSysfs: a hex value is logged as read, not as its decimal parse\n";
  const auto fx = LogFixture_t{};
  const auto path = fx.write("vendor", "0x1022\n");

  auto value = SmiSysfsReader::SysfsValue{};
  SmiSysfsReader::readLine(path, value);

  const auto all = joined();
  check("raw text present", has_substr(all, "value=0x1022"), all);
  check("caller still gets the parsed int",
        (std::holds_alternative<int>(value) && (std::get<int>(value) == 0x1022)));
}

auto test_sysfs_rdma_guids_are_masked() -> void {
  std::cout << "\nSysfs: MAC-derived RDMA GUIDs are masked in the log\n";
  const auto fx = LogFixture_t{};
  const auto guid = std::string{"ac1f:6bff:fe12:3456"};

  for (const auto* name : {"node_guid", "sys_image_guid"}) {
    g_captured.clear();
    const auto path = fx.write(name, guid + "\n");
    auto value = SmiSysfsReader::SysfsValue{};
    SmiSysfsReader::readLine(path, value);

    const auto all = joined();
    check(std::string(name) + " path logged", has_substr(all, path), all);
    check(std::string(name) + " full value absent", !has_substr(all, guid), all);
    check(std::string(name) + " masked tail present", has_substr(all, nlog::mask_tail(guid)), all);
  }
}

// Stands in for a logger that does syscalls (stat, open, write) and so overwrites errno.
auto errno_clobbering_sink(const std::string&) -> void { errno = 0; }

auto test_ioctl_preserves_errno_across_logging() -> void {
  std::cout << "\nEthtool ioctl: errno survives the logging done after the failing call\n";
  auto drvinfo = ethtool_drvinfo{};
  drvinfo.cmd = ETHTOOL_GDRVINFO;
  nlog::set_sink(nullptr);
  errno = 0;
  smi_ethtool_ioctl("nonexistent0", &drvinfo);
  const auto baseline_errno = errno;

  nlog::set_sink(errno_clobbering_sink);
  errno = 0;
  smi_ethtool_ioctl("nonexistent0", &drvinfo);
  const auto logged_errno = errno;
  nlog::set_sink(nullptr);

  check("baseline errno is set", (baseline_errno != 0));
  check("errno unchanged by logging", (logged_errno == baseline_errno),
        "baseline=" + std::to_string(baseline_errno) + " logged=" + std::to_string(logged_errno));
}

// Test double exposing SmiNicSubsystem::read_pci_ids, which discovery calls for every PCI device.
class PciIdProbe_t : public SmiNicSubsystem {
 public:
  auto probe(const std::string& bus_path) const -> std::pair<uint16_t, uint16_t> {
    return read_pci_ids(bus_path);
  }
  auto discover(const std::string&, const std::string&, std::shared_ptr<tp::NicTransport>)
      -> void override {}
  auto vendor() const -> NicVendor override { return NicVendor::Unknown; }
  auto is_driver_loaded(const std::string&, DriverType) const -> bool override { return false; }
  auto get_nics() const -> const std::vector<std::unique_ptr<SmiNic>>& override { return m_nics; }

 private:
  std::vector<std::unique_ptr<SmiNic>> m_nics;
};

auto test_discovery_pci_scan_logs_failures_only() -> void {
  std::cout << "\nDiscovery: successful PCI id reads are silent, failed ones are logged\n";
  const auto fx = LogFixture_t{};
  fx.write("vendor", "0x1dd8\n");
  fx.write("device", "0x1002\n");
  const auto good_dir = fs::path(fx.path_of("vendor")).parent_path().string();
  const auto probe = PciIdProbe_t{};

  const auto ids = probe.probe(good_dir);

  check("ids still parsed", (ids.first == 0x1dd8) && (ids.second == 0x1002));
  check("successful scan is not logged", g_captured.empty(), joined());

  const auto missing_dir = fx.path_of("no-such-device");
  const auto missing_ids = probe.probe(missing_dir);

  check("missing device reads as unknown", (missing_ids.first == 0) && (missing_ids.second == 0));
  const auto all = joined();
  check("failure is logged with the path", has_substr(all, missing_dir + "/vendor"), all);
  check("failure is marked FAIL", has_substr(all, "FAIL"), all);
}

auto test_sysfs_silent_without_sink() -> void {
  std::cout << "\nSysfs: no sink means no logging and an unchanged result\n";
  const auto fx = LogFixture_t{};
  nlog::set_sink(nullptr);  // the fixture installs the capture sink; this test needs none
  const auto path = fx.write("x", "7\n");

  auto value = SmiSysfsReader::SysfsValue{};
  const auto status = SmiSysfsReader::readLine(path, value);

  check("read succeeds", (status == SmiSysfsReader::SysfsStatus::Success));
  check("nothing captured", g_captured.empty());
}

auto test_ioctl_failure_logged() -> void {
  std::cout << "\nEthtool ioctl: failing call logs device, command and FAIL\n";
  const auto fx = LogFixture_t{};
  auto drvinfo = ethtool_drvinfo{};
  drvinfo.cmd = ETHTOOL_GDRVINFO;

  const auto ret = smi_ethtool_ioctl("nonexistent0", &drvinfo);

  check("call still returns -1", (ret == -1));
  const auto all = joined();
  check("device present", has_substr(all, "nonexistent0"), all);
  check("command present", has_substr(all, "cmd=0x3"), all);
  check("line names the ioctl function",
        has_substr(all, "smi_ethtool_ioctl") && has_substr(all, " | ethtool ioctl "), all);
  check("FAIL present", has_substr(all, "FAIL"), all);
  check("no SUCCESS", !has_substr(all, "SUCCESS"), all);
}

auto test_ioctl_silent_without_sink() -> void {
  std::cout << "\nEthtool ioctl: no sink means no logging\n";
  g_captured.clear();
  nlog::set_sink(nullptr);
  auto drvinfo = ethtool_drvinfo{};
  drvinfo.cmd = ETHTOOL_GDRVINFO;

  check("call still returns -1", (smi_ethtool_ioctl("nonexistent0", &drvinfo) == -1));
  check("nothing captured", g_captured.empty());
}

auto test_describe_drvinfo() -> void {
  std::cout << "\nEthtool describe: drvinfo shows driver, versions and bus\n";
  auto drvinfo = ethtool_drvinfo{};
  std::strncpy(drvinfo.driver, "ionic", sizeof(drvinfo.driver) - 1);
  std::strncpy(drvinfo.version, "1.2.3", sizeof(drvinfo.version) - 1);
  std::strncpy(drvinfo.fw_version, "fw-9", sizeof(drvinfo.fw_version) - 1);
  std::strncpy(drvinfo.bus_info, "0001:44:00.0", sizeof(drvinfo.bus_info) - 1);

  const auto text = smi_ethtool_describe(drvinfo);

  check("driver", has_substr(text, "driver=ionic"), text);
  check("version", has_substr(text, "version=1.2.3"), text);
  check("fw_version", has_substr(text, "fw_version=fw-9"), text);
  check("bus_info", has_substr(text, "bus_info=0001:44:00.0"), text);
}

auto test_describe_perm_addr_is_masked() -> void {
  std::cout << "\nEthtool describe: permanent address is masked\n";
  const auto mac = std::array<uint8_t, 6>{0x04, 0x90, 0x81, 0xb4, 0xee, kMacLastByte};
  auto raw = std::vector<uint8_t>(sizeof(ethtool_perm_addr) + mac.size());
  auto* perm = reinterpret_cast<ethtool_perm_addr*>(raw.data());
  perm->size = static_cast<uint32_t>(mac.size());
  std::memcpy(perm->data, mac.data(), mac.size());

  const auto text = smi_ethtool_describe(*perm);

  check("full MAC absent", !has_substr(text, kMacText), text);
  check("masked tail present", has_substr(text, nlog::mask_tail(kMacText)), text);
}

#ifdef HAVE_LIBNL3
namespace nl = amd::nic::netlink;

auto ignore_msg(struct nl_msg*, void*) -> int { return NL_OK; }

auto test_netlink_query_success_logged() -> void {
  std::cout << "\nNetlink query: successful dump logs command and SUCCESS\n";
  const auto fx = LogFixture_t{};
  auto client = nl::GenericNetlinkClient{};
  check("connect", (client.connect() == 0));
  const auto family = client.resolve_family_id("nlctrl");
  check("nlctrl resolves", family.has_value());
  if (!family.has_value()) {
    return;
  }
  g_captured.clear();

  const auto ret = client.query(*family, CTRL_CMD_GETFAMILY, 1, nullptr, ignore_msg, nullptr,
                                (NLM_F_REQUEST | NLM_F_DUMP));

  check("query succeeds", (ret == 0), "ret=" + std::to_string(ret));
  const auto all = joined();
  check("command present", has_substr(all, "cmd=" + std::to_string(CTRL_CMD_GETFAMILY)), all);
  check("line names the query function",
        has_substr(all, "GenericNetlinkClient::query") && has_substr(all, " | netlink query "),
        all);
  check("SUCCESS present", has_substr(all, "SUCCESS"), all);
}

auto test_netlink_query_failure_logged() -> void {
  std::cout << "\nNetlink query: rejected request logs FAIL\n";
  const auto fx = LogFixture_t{};
  auto client = nl::GenericNetlinkClient{};
  check("connect", (client.connect() == 0));
  g_captured.clear();

  const auto ret = client.query(kBogusFamilyId, 1, 1, nullptr, ignore_msg, nullptr, NLM_F_REQUEST);

  check("query fails", (ret < 0), "ret=" + std::to_string(ret));
  const auto all = joined();
  check("FAIL present", has_substr(all, "FAIL"), all);
  check("no SUCCESS", !has_substr(all, "SUCCESS"), all);
}

auto test_devlink_describe_reporters() -> void {
  std::cout << "\nDevlink describe: reporters\n";
  auto fw = nl::DevlinkReporter{};
  std::strncpy(fw.name, "fw", sizeof(fw.name) - 1);
  fw.healthy = 1;
  fw.error_count = 0;

  check("reporter shown", has_substr(nl::devlink_describe(std::vector<nl::DevlinkReporter>{fw}),
                                     "fw healthy=1 errors=0"));
  check("empty set is explicit",
        has_substr(nl::devlink_describe(std::vector<nl::DevlinkReporter>{}), "reporters=0"));
}

auto test_devlink_describe_port_split_unknown() -> void {
  std::cout << "\nDevlink describe: unreported split state reads as unknown\n";
  const auto text = nl::devlink_describe(nl::DevlinkPortSplit{});

  check("splittable unknown", has_substr(text, "splittable=unknown"), text);
  check("split_count unknown", has_substr(text, "split_count=unknown"), text);
}

auto test_devlink_describe_device_info_masks_serials() -> void {
  std::cout << "\nDevlink describe: serials are masked, versions are shown\n";
  const auto eui_serial = std::string{"0490 81b4eee0"};
  auto info = nl::DevlinkDeviceInfo{};
  set_cstr(info.driver_name, "ifoe");
  set_cstr(info.serial_number, eui_serial);
  set_cstr(info.board_serial_number, kSerialText);
  info.version_count = 2;
  info.versions[0].type = static_cast<uint8_t>(nl::DevlinkVersionType::Running);
  set_cstr(info.versions[0].name, "fw.mgmt");
  set_cstr(info.versions[0].value, "0.21.6.0");
  info.versions[1].type = static_cast<uint8_t>(nl::DevlinkVersionType::Fixed);
  set_cstr(info.versions[1].name, "board.serial_number");
  set_cstr(info.versions[1].value, kSerialText);

  const auto text = nl::devlink_describe(info);

  check("driver shown", has_substr(text, "driver=ifoe"), text);
  check("version shown", has_substr(text, "fw.mgmt=0.21.6.0"), text);
  check("board serial absent", !has_substr(text, kSerialText), text);
  check("EUI serial absent", !has_substr(text, eui_serial), text);
  check("board serial masked tail", has_substr(text, kSerialMasked), text);
}
#endif  // HAVE_LIBNL3

/**
 * Fake inner transport: canned results, no I/O. It is the boundary under the
 * logging decorator, which is the subject of these tests.
 */
class CannedTransport_t : public tp::NicTransport {
 public:
  tp::Result<tp::PauseParams> pause{true, {true, true, false}, 0};
  tp::Result<tp::LinkSettings> link{true, {kLinkSpeedMbps, 1, 1, 0x10, 0x20}, 0};
  tp::Result<tp::DriverInfo> drv{true, {"ionic", "1.2.3", "fw-9", "0001:44:00.0", 2}, 0};
  tp::Result<tp::VendorStatistics> stats{true, {{"rx_pkts", "tx_pkts"}, {11, 22}}, 0};
  tp::Result<tp::FecStatistics_t> fec{false, {}, ENOTSUP};
  tp::Result<tp::PermanentAddress> perm{true, {{{0x04, 0x90, 0x81, 0xb4, 0xee, kMacLastByte}}}, 0};

  auto get_pause_params(const std::string&) -> tp::Result<tp::PauseParams> override {
    return pause;
  }
  auto get_link_settings(const std::string&) -> tp::Result<tp::LinkSettings> override {
    return link;
  }
  auto get_driver_info(const std::string&) -> tp::Result<tp::DriverInfo> override { return drv; }
  auto get_statistics(const std::string&) -> tp::Result<tp::VendorStatistics> override {
    return stats;
  }
  auto get_fec_statistics(const std::string&) -> tp::Result<tp::FecStatistics_t> override {
    return fec;
  }
  auto get_permanent_address(const std::string&) -> tp::Result<tp::PermanentAddress> override {
    return perm;
  }
  auto backend_name() const -> std::string override { return "canned"; }
};

auto test_transport_log_passthrough() -> void {
  std::cout << "\nTransport log: results and backend name pass through unchanged\n";
  const auto fx = LogFixture_t{};
  auto logged = tp::with_debug_logging(std::make_shared<CannedTransport_t>());

  check("backend name forwarded", (logged->backend_name() == "canned"));
  const auto link = logged->get_link_settings(kIface);
  check("link success preserved", (link.success && (link.value.speed == kLinkSpeedMbps)));
  const auto fec = logged->get_fec_statistics(kIface);
  check("failure preserved", (!fec.success && (fec.error_code == ENOTSUP)));
}

auto test_transport_log_content() -> void {
  std::cout << "\nTransport log: each operation logs interface, status and parsed content\n";
  const auto fx = LogFixture_t{};
  auto logged = tp::with_debug_logging(std::make_shared<CannedTransport_t>());

  logged->get_pause_params(kIface);
  logged->get_link_settings(kIface);
  logged->get_driver_info(kIface);
  logged->get_fec_statistics(kIface);
  const auto all = joined();

  check("iface present", has_substr(all, kIface), all);
  check("lines name the transport method",
        has_substr(all, "get_pause_params") && has_substr(all, "get_link_settings") &&
            has_substr(all, " | transport "),
        all);
  check("pause content", has_substr(all, "autoneg=1 rx_pause=1 tx_pause=0"), all);
  check("link content", has_substr(all, "speed=" + std::to_string(kLinkSpeedMbps)), all);
  check("driver content", has_substr(all, "driver=ionic"), all);
  check("failure logged with errno", (has_substr(all, "FAIL") && has_substr(all, "not supported")),
        all);
  check("numeric code logged", has_substr(all, "code=" + std::to_string(ENOTSUP)), all);
}

auto test_transport_log_stats_every_counter() -> void {
  std::cout << "\nTransport log: statistics log the count and every counter\n";
  const auto fx = LogFixture_t{};
  auto logged = tp::with_debug_logging(std::make_shared<CannedTransport_t>());

  logged->get_statistics(kIface);
  const auto all = joined();

  check("count present", has_substr(all, "count=2"), all);
  check("first counter", has_substr(all, "rx_pkts=11"), all);
  check("second counter", has_substr(all, "tx_pkts=22"), all);
}

auto test_transport_log_permanent_address_masked() -> void {
  std::cout << "\nTransport log: permanent address is masked\n";
  const auto fx = LogFixture_t{};
  auto logged = tp::with_debug_logging(std::make_shared<CannedTransport_t>());

  const auto perm = logged->get_permanent_address(kIface);
  const auto all = joined();

  check("caller still gets the full MAC", (perm.success && (perm.value.mac[5] == kMacLastByte)));
  check("full MAC absent from log", !has_substr(all, kMacText), all);
  check("masked tail present", has_substr(all, nlog::mask_tail(kMacText)), all);
}

auto test_transport_log_silent_without_sink() -> void {
  std::cout << "\nTransport log: no sink means no logging\n";
  g_captured.clear();
  nlog::set_sink(nullptr);
  auto logged = tp::with_debug_logging(std::make_shared<CannedTransport_t>());

  const auto stats = logged->get_statistics(kIface);

  check("stats preserved", (stats.success && (stats.value.names.size() == 2)));
  check("nothing captured", g_captured.empty());
}

// Minimal PCI VPD image: identifier string, VPD-R with PN and SN, end tag.
auto make_vpd_image(const std::string& part, const std::string& serial) -> std::string {
  auto vpd_r = std::string{};
  vpd_r += std::string("PN") + static_cast<char>(part.size()) + part;
  vpd_r += std::string("SN") + static_cast<char>(serial.size()) + serial;
  auto img = std::string{};
  img += std::string("\x82\x05\x00", 3) + "NIC01";
  img += std::string("\x90", 1) + static_cast<char>(vpd_r.size()) + std::string(1, '\0') + vpd_r;
  img += '\x78';
  return img;
}

auto test_vpd_read_logged_and_serial_masked() -> void {
  std::cout << "\nVPD: read logs path, size and decoded fields; serial is masked\n";
  const auto fx = LogFixture_t{};
  const auto vpd_path = fx.write("vpd", make_vpd_image("100-700000006", kSerialText));
  const auto bus_dir = fs::path(vpd_path).parent_path().string();
  const auto nic = SmiNic{kIface, "0000:03:00.0", NicType::Ethernet, "", bus_dir};

  const auto is_readable = nic.is_vpd_readable();

  check("vpd readable", is_readable);
  const auto all = joined();
  check("full path present", has_substr(all, vpd_path), all);
  check("SUCCESS present", has_substr(all, "SUCCESS"), all);
  check("part number shown", has_substr(all, "part=100-700000006"), all);
  check("line names the VPD function",
        has_substr(all, "read_device_vpd") && has_substr(all, " | vpd read "), all);
  check("serial absent", !has_substr(all, kSerialText), all);
  check("serial masked tail", has_substr(all, nlog::mask_tail(kSerialText)), all);
}

auto test_vpd_missing_logged() -> void {
  std::cout << "\nVPD: unreadable vpd logs the path and FAIL\n";
  const auto fx = LogFixture_t{};
  const auto dir = fs::path(fx.path_of("novpd"));
  fs::create_directories(dir);
  const auto nic = SmiNic{kIface, "0000:03:00.0", NicType::Ethernet, "", dir.string()};

  check("vpd not readable", !nic.is_vpd_readable());
  const auto all = joined();
  check("full path present", has_substr(all, (dir / "vpd").string()), all);
  check("FAIL present", has_substr(all, "FAIL"), all);
}

auto test_hwmon_scan_logged() -> void {
  std::cout << "\nHwmon scan: logs the directory and the chosen temp file\n";
  const auto fx = LogFixture_t{};
  const auto dev = fs::path(fx.path_of("dev"));
  fs::create_directories(dev / "hwmon" / "hwmon3");
  std::ofstream(dev / "hwmon" / "hwmon3" / "temp1_input") << kTempMilliC << "\n";
  const auto nic = SmiNic{kIface, "0000:03:00.0", NicType::Ethernet, "", dev.string()};

  const auto path = nic.hwmon_temp_path(NicTempSensor::Asic);

  check("path resolved", path.has_value());
  const auto all = joined();
  check("hwmon dir present", has_substr(all, (dev / "hwmon").string()), all);
  check("chosen file present", has_substr(all, "temp1_input"), all);
  check("line names the hwmon function",
        has_substr(all, "hwmon_temp_path") && has_substr(all, " | hwmon scan "), all);
  check("SUCCESS present", has_substr(all, "SUCCESS"), all);
}

auto test_hwmon_scan_missing_logged() -> void {
  std::cout << "\nHwmon scan: absent hwmon directory logs the path and FAIL\n";
  const auto fx = LogFixture_t{};
  const auto dev = fs::path(fx.path_of("nohwmon"));
  fs::create_directories(dev);
  const auto nic = SmiNic{kIface, "0000:03:00.0", NicType::Ethernet, "", dev.string()};

  check("no path", !nic.hwmon_temp_path(NicTempSensor::Asic).has_value());
  const auto all = joined();
  check("hwmon dir present", has_substr(all, (dev / "hwmon").string()), all);
  check("FAIL present", has_substr(all, "FAIL"), all);
}

}  // namespace

int main() {
  test_no_sink_is_disabled_and_silent();
  test_sink_receives_message();
  test_mask_tail();
  test_sysfs_success_logged();
  test_sysfs_missing_file_logged();
  test_sysfs_mac_is_masked();
  test_sysfs_logs_raw_text_not_parsed_number();
  test_sysfs_rdma_guids_are_masked();
  test_discovery_pci_scan_logs_failures_only();
  test_ioctl_preserves_errno_across_logging();
  test_sysfs_silent_without_sink();
  test_ioctl_failure_logged();
  test_ioctl_silent_without_sink();
  test_describe_drvinfo();
  test_describe_perm_addr_is_masked();
  test_transport_log_passthrough();
  test_transport_log_content();
  test_transport_log_stats_every_counter();
  test_transport_log_permanent_address_masked();
  test_transport_log_silent_without_sink();
  test_vpd_read_logged_and_serial_masked();
  test_vpd_missing_logged();
  test_hwmon_scan_logged();
  test_hwmon_scan_missing_logged();
#ifdef HAVE_LIBNL3
  test_netlink_query_success_logged();
  test_netlink_query_failure_logged();
  test_devlink_describe_reporters();
  test_devlink_describe_port_split_unknown();
  test_devlink_describe_device_info_masks_serials();
#endif

  std::cout << "\n" << tests_run << " tests run, " << tests_failed << " failed\n";
  return (tests_failed == 0) ? 0 : 1;
}
