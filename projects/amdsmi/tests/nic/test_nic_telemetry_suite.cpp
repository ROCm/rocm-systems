// SPDX-License-Identifier: MIT
/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * Unit tests for the NIC telemetry facade (temperature / health / port-split).
 *
 * These tests need no hardware and no root: devlink is exercised through a fake
 * IDevlinkClient, and sysfs/hwmon through fake files written to a temp dir. They
 * validate the facade's uniform-schema-plus-unsupported contract, not any real
 * device. The one exception is the context-entry-point test, which builds a real
 * context and so walks the host's /sys; its assertions hold whether or not that
 * walk finds anything.
 */

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include "amd_smi/impl/nic/amd_smi_nic_interface.h"
#include "nic_check.h"
#include "smi_nic_hwmon.h"
#include "smi_nic_interface.h"
#include "smi_nic_telemetry.h"
#include "smi_nic_testing.h"

namespace tp = amd::smi::nic::transport;
namespace tel = amd::smi::nic::telemetry;
namespace dl = amd::nic::netlink;
namespace hw = amd::smi::nic::hwmon;

// Layout tripwire. The public bridge (amd_smi.cc, amdsmi_get_nic_telemetry) copies this
// snapshot field-by-field into amdsmi_nic_telemetry_t; no static_assert can prove that
// copy is complete (a field added inside existing padding is invisible here). If a field
// is added and a size below changes, update the public copy and its field-population test
// before touching these numbers.
static_assert(sizeof(tel::NicTemperature) == 6, "NicTemperature grew; revisit public C2 copy");
static_assert(sizeof(tel::NicHealth) == 72, "NicHealth grew; revisit public C2 copy");
static_assert(sizeof(tel::NicPortSplit) == 2, "NicPortSplit grew; revisit public C2 copy");
static_assert(sizeof(tel::NicTelemetrySnapshot) == 84,
              "NicTelemetrySnapshot grew; revisit public C2 copy");

using nic_test::check;

// Fakes

// Builds a DevlinkReporter with the fixed-size name field populated safely.
static dl::DevlinkReporter make_reporter(const std::string& name, uint8_t healthy,
                                         uint32_t error_count) {
  dl::DevlinkReporter r{};
  std::snprintf(r.name, sizeof(r.name), "%s", name.c_str());
  r.healthy = healthy;
  r.error_count = error_count;
  return r;
}

// Configurable devlink client: tests set the results the facade will observe,
// and read back the device each query was addressed to.
class FakeDevlinkClient : public dl::IDevlinkClient {
 public:
  tp::Result<dl::DevlinkPortSplit> port_split{false, {}, ENOTSUP};
  tp::Result<std::vector<dl::DevlinkReporter>> reporters{false, {}, ENOTSUP};
  tp::Result<dl::DevlinkDeviceInfo> device_info{false, {}, ENOTSUP};

  std::string port_split_dev;
  std::string reporters_dev;
  std::string device_info_dev;

  tp::Result<dl::DevlinkPortSplit> get_port_split(const std::string& dev) override {
    port_split_dev = dev;
    return port_split;
  }
  tp::Result<std::vector<dl::DevlinkReporter>> get_health_reporters(
      const std::string& dev) override {
    reporters_dev = dev;
    return reporters;
  }
  tp::Result<dl::DevlinkDeviceInfo> get_device_info(const std::string& dev) override {
    device_info_dev = dev;
    return device_info;
  }
};

// Appends a version entry to a DevlinkDeviceInfo, mirroring how the netlink
// handler fills the fixed-size store.
static void add_version(dl::DevlinkDeviceInfo& info, dl::DevlinkVersionType type,
                        const std::string& name, const std::string& value) {
  auto& v = info.versions[info.version_count++];
  v.type = static_cast<uint8_t>(type);
  std::snprintf(v.name, sizeof(v.name), "%s", name.c_str());
  std::snprintf(v.value, sizeof(v.value), "%s", value.c_str());
}

// NIC whose sensor mapping is driven by the test. hwmon_temp_path remains a
// virtual override point, which is what lets these tests inject temp files
// without touching real sysfs.
class FakeNic : public SmiNic {
 public:
  FakeNic() : SmiNic("eth0", "0000:03:00.0") {}

  std::map<NicTempSensor, std::string> temp_paths;

  std::optional<std::string> hwmon_temp_path(NicTempSensor sensor) const override {
    auto it = temp_paths.find(sensor);
    if (it == temp_paths.end()) {
      return std::nullopt;
    }
    return it->second;
  }
};

static std::string g_tmpdir;

static std::string write_temp_file(const std::string& name, const std::string& content) {
  const std::string path = g_tmpdir + "/" + name;
  std::ofstream f(path);
  f << content;
  f.close();
  return path;
}

// Writes `content` to g_tmpdir/<rel>, creating the parent directories.
static void write_sysfs_file(const std::string& rel, const std::string& content) {
  const auto path = std::filesystem::path(g_tmpdir) / rel;
  std::filesystem::create_directories(path.parent_path());
  std::ofstream f(path);
  f << content;
}

// An ionic hwmon node as `sensors` reported it for the fourth AINIC. Returns the hwmon root,
// the directory that holds hwmon3.
static std::string write_ionic_hwmon(const std::string& root_rel) {
  const std::string node = root_rel + "/hwmon3/";
  write_sysfs_file(node + "name", "ionic\n");
  write_sysfs_file(node + "in0_input", "12060\n");
  write_sysfs_file(node + "in0_label", "input voltage\n");
  write_sysfs_file(node + "in1_input", "744\n");
  write_sysfs_file(node + "in1_label", "core voltage\n");
  write_sysfs_file(node + "temp1_input", "39000\n");
  write_sysfs_file(node + "temp1_label", "board temperature\n");
  write_sysfs_file(node + "temp1_crit", "80000\n");
  write_sysfs_file(node + "temp2_input", "59000\n");
  write_sysfs_file(node + "temp2_label", "die temperature\n");
  write_sysfs_file(node + "temp2_crit", "105000\n");
  write_sysfs_file(node + "power1_input", "37700000\n");
  write_sysfs_file(node + "power1_label", "total power\n");
  write_sysfs_file(node + "power1_max", "40000000\n");
  write_sysfs_file(node + "power2_input", "11000000\n");
  write_sysfs_file(node + "power2_label", "core power\n");
  return g_tmpdir + "/" + root_rel;
}

// Test cases

static void test_temperature_partial() {
  std::cout << "\nTemperature: asic supported, others absent\n";
  FakeNic nic;
  nic.temp_paths[NicTempSensor::Asic] = write_temp_file("temp1_input", "45000\n");

  tel::NicTelemetry telemetry(std::make_shared<FakeDevlinkClient>());
  auto r = telemetry.get_temperature(nic);

  check("call succeeds", r.success, "errno=" + std::to_string(r.error_code));
  check("asic = 45C", r.value.asic_temp_c == 45, "got " + std::to_string(r.value.asic_temp_c));
  check("transceiver = reserved value", r.value.transceiver_temp_c == tel::kTempUnsupported);
  check("board = reserved value", r.value.board_temp_c == tel::kTempUnsupported);
}

static void test_temperature_unsupported() {
  std::cout << "\nTemperature: vendor exposes no sensor\n";
  FakeNic nic;  // no temp_paths

  tel::NicTelemetry telemetry(std::make_shared<FakeDevlinkClient>());
  auto r = telemetry.get_temperature(nic);

  check("call reports unsupported", !r.success);
  check("errno = ENOTSUP", r.error_code == ENOTSUP, "got " + std::to_string(r.error_code));
}

static void test_temperature_present_but_unreadable() {
  std::cout << "\nTemperature: sensor mapped but file missing\n";
  FakeNic nic;
  nic.temp_paths[NicTempSensor::Asic] = g_tmpdir + "/does_not_exist";

  tel::NicTelemetry telemetry(std::make_shared<FakeDevlinkClient>());
  auto r = telemetry.get_temperature(nic);

  check("call still succeeds (sensor is supported)", r.success);
  check("asic = reserved value (unreadable)", r.value.asic_temp_c == tel::kTempUnsupported);
}

static void test_health_supported() {
  std::cout << "\nHealth: single healthy reporter (bnxt 'fw')\n";
  FakeNic nic;

  auto fake = std::make_shared<FakeDevlinkClient>();
  fake->reporters = {true, {make_reporter("fw", /*healthy=*/1, /*errors=*/3)}, 0};

  tel::NicTelemetry telemetry(fake);
  auto r = telemetry.get_health(nic);

  check("call succeeds", r.success, "errno=" + std::to_string(r.error_code));
  check("state = Healthy", r.value.state == static_cast<uint8_t>(tel::HealthState::Healthy));
  check("error_count = 3", r.value.error_count == 3);
  check("reporter name copied", std::strcmp(r.value.reporter, "fw") == 0,
        std::string("got '") + r.value.reporter + "'");
}

static void test_health_reporter_name_varies() {
  std::cout << "\nHealth: reporter named 'fw_reset' (bnxt on other firmware)\n";
  FakeNic nic;

  auto fake = std::make_shared<FakeDevlinkClient>();
  fake->reporters = {true, {make_reporter("fw_reset", /*healthy=*/1, /*errors=*/0)}, 0};

  tel::NicTelemetry telemetry(fake);
  auto r = telemetry.get_health(nic);

  check("call succeeds", r.success);
  check("enumerated name surfaced (not hardcoded 'fw')",
        std::strcmp(r.value.reporter, "fw_reset") == 0,
        std::string("got '") + r.value.reporter + "'");
}

static void test_health_unhealthy() {
  std::cout << "\nHealth: single unhealthy reporter\n";
  FakeNic nic;

  auto fake = std::make_shared<FakeDevlinkClient>();
  fake->reporters = {true, {make_reporter("fw", /*healthy=*/0, /*errors=*/7)}, 0};

  tel::NicTelemetry telemetry(fake);
  auto r = telemetry.get_health(nic);

  check("call succeeds", r.success);
  check("state = Error", r.value.state == static_cast<uint8_t>(tel::HealthState::Error));
  check("error_count = 7", r.value.error_count == 7);
}

static void test_health_multi_reporter_aggregate() {
  std::cout << "\nHealth: multiple reporters aggregate to worst state\n";
  FakeNic nic;

  auto fake = std::make_shared<FakeDevlinkClient>();
  fake->reporters = {true,
                     {make_reporter("fw", /*healthy=*/1, /*errors=*/1),
                      make_reporter("hw_err", /*healthy=*/0, /*errors=*/4),
                      make_reporter("rx", /*healthy=*/1, /*errors=*/2)},
                     0};

  tel::NicTelemetry telemetry(fake);
  auto r = telemetry.get_health(nic);

  check("call succeeds", r.success);
  check("state = Error (any unhealthy)",
        r.value.state == static_cast<uint8_t>(tel::HealthState::Error));
  check("error_count summed = 7", r.value.error_count == 7,
        "got " + std::to_string(r.value.error_count));
  check("name = first unhealthy reporter", std::strcmp(r.value.reporter, "hw_err") == 0,
        std::string("got '") + r.value.reporter + "'");
}

static void test_health_error_count_saturates_below_sentinel() {
  std::cout << "\nHealth: overflowing error_count saturates below the unsupported reserved value\n";
  FakeNic nic;

  // Two reporters each at UINT32_MAX: the u64 sum overflows uint32, so it must
  // clamp. It must NOT clamp to kErrorCountUnsupported (that means "not exposed").
  auto fake = std::make_shared<FakeDevlinkClient>();
  fake->reporters = {true,
                     {make_reporter("fw", /*healthy=*/1, /*errors=*/UINT32_MAX),
                      make_reporter("rx", /*healthy=*/1, /*errors=*/UINT32_MAX)},
                     0};

  tel::NicTelemetry telemetry(fake);
  auto r = telemetry.get_health(nic);

  check("call succeeds", r.success);
  check("error_count is not the unsupported reserved value",
        r.value.error_count != tel::kErrorCountUnsupported,
        "got " + std::to_string(r.value.error_count));
  check("error_count saturated to reserved value minus 1",
        r.value.error_count == (tel::kErrorCountUnsupported - 1),
        "got " + std::to_string(r.value.error_count));
}

static void test_health_no_reporter() {
  std::cout << "\nHealth: device exposes no reporter (ionic)\n";
  FakeNic nic;

  auto fake = std::make_shared<FakeDevlinkClient>();
  fake->reporters = {true, {}, 0};  // enumeration succeeded but empty

  tel::NicTelemetry telemetry(fake);
  auto r = telemetry.get_health(nic);

  check("call reports unsupported", !r.success);
  check("errno = ENOTSUP", r.error_code == ENOTSUP);
}

static void test_health_devlink_failure() {
  std::cout << "\nHealth: devlink query itself fails\n";
  FakeNic nic;

  auto fake = std::make_shared<FakeDevlinkClient>();
  fake->reporters = {false, {}, ENOTSUP};

  tel::NicTelemetry telemetry(fake);
  auto r = telemetry.get_health(nic);

  check("call fails", !r.success);
  check("errno propagated", r.error_code == ENOTSUP);
}

static void test_port_split_active() {
  std::cout << "\nPort-split: splittable port currently split\n";
  FakeNic nic;

  auto fake = std::make_shared<FakeDevlinkClient>();
  fake->port_split = {true, {/*splittable=*/1, /*split_count=*/4}, 0};

  tel::NicTelemetry telemetry(fake);
  auto r = telemetry.get_port_split(nic);

  check("call succeeds", r.success, "errno=" + std::to_string(r.error_code));
  check("splittable", r.value.splittable == 1);
  check("split_count = 4", r.value.split_count == 4, "got " + std::to_string(r.value.split_count));
}

static void test_port_split_not_splittable() {
  std::cout << "\nPort-split: port present but not splittable (the observed fleet)\n";
  FakeNic nic;

  auto fake = std::make_shared<FakeDevlinkClient>();
  fake->port_split = {true, {/*splittable=*/0, /*split_count=*/0}, 0};

  tel::NicTelemetry telemetry(fake);
  auto r = telemetry.get_port_split(nic);

  check("call succeeds", r.success);
  check("not splittable", r.value.splittable == 0);
  check("split_count = 0", r.value.split_count == 0);
}

static void test_port_split_attrs_absent() {
  std::cout << "\nPort-split: port listed but kernel sent no split attributes (ionic, ifoe)\n";
  FakeNic nic;

  // Value-initialised, exactly what the port dump handler yields when neither
  // SPLITTABLE nor SPLIT_COUNT is present.
  auto fake = std::make_shared<FakeDevlinkClient>();
  fake->port_split = {true, {}, 0};

  tel::NicTelemetry telemetry(fake);
  auto r = telemetry.get_port_split(nic);

  check("call succeeds", r.success);
  check("splittable = N/A marker", r.value.splittable == tel::kCountUnsupported,
        "got " + std::to_string(r.value.splittable));
  check("split_count = N/A marker", r.value.split_count == tel::kCountUnsupported,
        "got " + std::to_string(r.value.split_count));
}

static void test_port_split_unsupported() {
  std::cout << "\nPort-split: device exposes no port object (pds_core)\n";
  FakeNic nic;

  tel::NicTelemetry telemetry(std::make_shared<FakeDevlinkClient>());  // default ENOTSUP
  auto r = telemetry.get_port_split(nic);

  check("call fails", !r.success);
  check("errno = ENOTSUP", r.error_code == ENOTSUP);
}

static void test_snapshot_all_supported() {
  std::cout << "\nSnapshot: all three metrics supported (bnxt-like)\n";
  FakeNic nic;
  nic.temp_paths[NicTempSensor::Asic] = write_temp_file("temp1_input", "45000\n");

  auto fake = std::make_shared<FakeDevlinkClient>();
  fake->reporters = {true, {make_reporter("fw", /*healthy=*/1, /*errors=*/0)}, 0};
  fake->port_split = {true, {/*splittable=*/0, /*split_count=*/0}, 0};

  tel::NicTelemetry telemetry(fake);
  auto r = telemetry.get_snapshot(nic);

  check("snapshot succeeds", r.success, "errno=" + std::to_string(r.error_code));
  check("temp asic = 45C", r.value.temperature.asic_temp_c == 45,
        "got " + std::to_string(r.value.temperature.asic_temp_c));
  check("health = Healthy",
        r.value.health.state == static_cast<uint8_t>(tel::HealthState::Healthy));
  check("health reporter = fw", std::strcmp(r.value.health.reporter, "fw") == 0);
  check("port not splittable", r.value.port_split.splittable == 0);
}

static void test_snapshot_mixed() {
  std::cout << "\nSnapshot: health only, temp and port-split absent (pds_core)\n";
  FakeNic nic;  // no temp sensors

  auto fake = std::make_shared<FakeDevlinkClient>();
  fake->reporters = {true, {make_reporter("fw", /*healthy=*/1, /*errors=*/0)}, 0};
  // port_split left at default ENOTSUP

  tel::NicTelemetry telemetry(fake);
  auto r = telemetry.get_snapshot(nic);

  check("snapshot still succeeds", r.success);
  check("temp asic = reserved value", r.value.temperature.asic_temp_c == tel::kTempUnsupported);
  check("temp board = reserved value", r.value.temperature.board_temp_c == tel::kTempUnsupported);
  check("health present = Healthy",
        r.value.health.state == static_cast<uint8_t>(tel::HealthState::Healthy));
  check("port splittable = reserved value",
        r.value.port_split.splittable == tel::kCountUnsupported);
  check("port split_count = reserved value",
        r.value.port_split.split_count == tel::kCountUnsupported);
}

static void test_snapshot_all_unsupported() {
  std::cout << "\nSnapshot: nothing exposed (ionic)\n";
  FakeNic nic;  // no temp sensors

  auto fake = std::make_shared<FakeDevlinkClient>();
  fake->reporters = {true, {}, 0};  // enumeration succeeded but empty -> unsupported
  // port_split left at default ENOTSUP

  tel::NicTelemetry telemetry(fake);
  auto r = telemetry.get_snapshot(nic);

  check("snapshot still succeeds (per-field reserved values, not whole failure)", r.success);
  check("temp asic = reserved value", r.value.temperature.asic_temp_c == tel::kTempUnsupported);
  check("health state = Unsupported (distinct from Unknown)",
        r.value.health.state == static_cast<uint8_t>(tel::HealthState::Unsupported));
  check("health error_count = reserved value",
        r.value.health.error_count == tel::kErrorCountUnsupported);
  check("health reporter = empty", r.value.health.reporter[0] == '\0');
  check("port splittable = reserved value",
        r.value.port_split.splittable == tel::kCountUnsupported);
}

static void test_device_info_supported() {
  std::cout << "\nDevice info: serial + fixed/running/stored versions (pds_core-like)\n";
  FakeNic nic;

  dl::DevlinkDeviceInfo di{};
  std::snprintf(di.driver_name, sizeof(di.driver_name), "%s", "pds_core");
  std::snprintf(di.serial_number, sizeof(di.serial_number), "%s", "FLM2231000X");
  add_version(di, dl::DevlinkVersionType::Fixed, "asic.id", "0x1002");
  add_version(di, dl::DevlinkVersionType::Running, "fw", "1.60.0-73");
  add_version(di, dl::DevlinkVersionType::Stored, "fw.mainfwa", "1.60.0-73");

  auto fake = std::make_shared<FakeDevlinkClient>();
  fake->device_info = {true, di, 0};

  tel::NicTelemetry telemetry(fake);
  auto r = telemetry.get_device_info(nic);

  check("call succeeds", r.success, "errno=" + std::to_string(r.error_code));
  check("driver forwarded", std::strcmp(r.value.driver_name, "pds_core") == 0,
        std::string("got '") + r.value.driver_name + "'");
  check("serial forwarded", std::strcmp(r.value.serial_number, "FLM2231000X") == 0,
        std::string("got '") + r.value.serial_number + "'");
  check("version_count = 3", r.value.version_count == 3,
        "got " + std::to_string(r.value.version_count));
  check("running fw name/value forwarded",
        (r.value.versions[1].type == static_cast<uint8_t>(dl::DevlinkVersionType::Running)) &&
            (std::strcmp(r.value.versions[1].name, "fw") == 0) &&
            (std::strcmp(r.value.versions[1].value, "1.60.0-73") == 0));
}

static void test_device_info_unsupported() {
  std::cout << "\nDevice info: device has no devlink instance\n";
  FakeNic nic;

  tel::NicTelemetry telemetry(std::make_shared<FakeDevlinkClient>());  // default ENOTSUP
  auto r = telemetry.get_device_info(nic);

  check("call reports unsupported", !r.success);
  check("errno = ENOTSUP", r.error_code == ENOTSUP, "got " + std::to_string(r.error_code));
}

static void set_devlink_version(dl::DevlinkDeviceInfo& info, uint8_t index,
                                dl::DevlinkVersionType type, const char* name, const char* value) {
  info.versions[index].type = static_cast<uint8_t>(type);
  std::snprintf(info.versions[index].name, sizeof(info.versions[index].name), "%s", name);
  std::snprintf(info.versions[index].value, sizeof(info.versions[index].value), "%s", value);
}

static void test_fw_info_keeps_each_devlink_version_with_its_type() {
  std::cout << "\nFirmware: devlink versions map to the shim struct\n";
  auto dev = dl::DevlinkDeviceInfo{};
  set_devlink_version(dev, 0, dl::DevlinkVersionType::Fixed, "asic.id", "0x1780");
  set_devlink_version(dev, 1, dl::DevlinkVersionType::Running, "fw", "1.0.2.8");
  set_devlink_version(dev, 2, dl::DevlinkVersionType::Stored, "fw", "1.0.2.9");
  dev.version_count = 3;

  auto fw = smi_nic_fw_info_t{};
  smi_nic_fw_info_from_devlink_for_testing(dev, &fw);

  check("three versions are copied", fw.count == 3, "got " + std::to_string(fw.count));
  check("a fixed version keeps its type", fw.versions[0].type == SMI_NIC_FW_VERSION_TYPE_FIXED);
  check("a running version keeps its type", fw.versions[1].type == SMI_NIC_FW_VERSION_TYPE_RUNNING);
  check("a stored version keeps its type", fw.versions[2].type == SMI_NIC_FW_VERSION_TYPE_STORED);
  check("the name is copied", std::string(fw.versions[0].name) == "asic.id");
  check("the value lands in version", std::string(fw.versions[1].version) == "1.0.2.8");
  check("two entries named fw stay separate",
        (std::string(fw.versions[1].name) == "fw") && (std::string(fw.versions[2].name) == "fw") &&
            (std::string(fw.versions[2].version) == "1.0.2.9"));
}

static void test_fw_info_of_a_device_without_versions_is_empty() {
  auto dev = dl::DevlinkDeviceInfo{};
  auto fw = smi_nic_fw_info_t{};
  fw.count = 7;  // stale value must not survive
  smi_nic_fw_info_from_devlink_for_testing(dev, &fw);
  check("no versions give a zero count", fw.count == 0);
}

static void test_telemetry_shim_struct_copies_every_snapshot_field() {
  std::cout << "\nTelemetry: snapshot maps to the shim struct\n";
  auto snapshot = tel::NicTelemetrySnapshot{};
  snapshot.temperature = {52, tel::kTempUnsupported, 40};
  snapshot.health.state = static_cast<uint8_t>(tel::HealthState::Healthy);
  snapshot.health.error_count = 7;
  std::snprintf(snapshot.health.reporter, sizeof(snapshot.health.reporter), "%s", "fw");
  snapshot.port_split = {1, 4};

  auto telemetry = smi_nic_telemetry_t{};
  smi_nic_telemetry_from_snapshot_for_testing(snapshot, &telemetry);

  check("asic temperature", telemetry.temperature.asic_temp_c == 52);
  check("an unexposed transceiver temperature keeps the reserved value",
        telemetry.temperature.transceiver_temp_c == tel::kTempUnsupported);
  check("board temperature", telemetry.temperature.board_temp_c == 40);
  check("health state", telemetry.health.state == static_cast<uint8_t>(tel::HealthState::Healthy));
  check("error count", telemetry.health.error_count == 7);
  check("reporter name", std::string(telemetry.health.reporter) == "fw");
  check("splittable", telemetry.port_split.splittable == 1);
  check("split count", telemetry.port_split.split_count == 4);
}

static hw::SensorRecord_t make_sensor_record(uint32_t channel, int64_t value, const char* label) {
  return {hw::Category_t::Temperature,
          hw::Attribute_t::Input,
          hw::Unit_t::MilliDegreeCelsius,
          channel,
          3,
          label,
          value};
}

static void test_sensor_shim_struct_copies_every_record_field() {
  std::cout << "\nSensors: records map to the shim struct\n";
  auto records = std::vector<hw::SensorRecord_t>{};
  records.push_back(make_sensor_record(1, 39000, "board temperature"));
  records.push_back(
      {hw::Category_t::Voltage, hw::Attribute_t::Crit, hw::Unit_t::MilliVolt, 0, 4, "", -5});

  auto info = std::make_unique<smi_nic_sensor_info_t>();
  const auto status = smi_nic_sensor_info_from_records_for_testing(records, info.get());

  check("the conversion succeeds", status == SMI_NIC_STATUS_SUCCESS);
  check("count", info->count == 2, std::to_string(info->count));
  check("category",
        info->sensors[0].category == static_cast<uint32_t>(hw::Category_t::Temperature));
  check("attribute", info->sensors[1].attribute == static_cast<uint32_t>(hw::Attribute_t::Crit));
  check("unit", info->sensors[1].unit == static_cast<uint32_t>(hw::Unit_t::MilliVolt));
  check("channel", info->sensors[0].channel == 1);
  check("instance", info->sensors[1].instance == 4);
  check("label", std::string(info->sensors[0].label) == "board temperature");
  check("a missing label is empty", std::string(info->sensors[1].label).empty());
  check("a negative value survives", info->sensors[1].value == -5);
}

static void test_sensor_shim_struct_reports_overflow_and_keeps_what_fits() {
  auto records = std::vector<hw::SensorRecord_t>{};
  for (uint32_t i = 0; i < (SMI_NIC_MAX_SENSORS + 1); ++i) {
    records.push_back(make_sensor_record(i, i, "t"));
  }

  auto info = std::make_unique<smi_nic_sensor_info_t>();
  const auto status = smi_nic_sensor_info_from_records_for_testing(records, info.get());

  check("more records than the struct holds report no resource",
        status == SMI_NIC_STATUS_NO_RESOURCE);
  check("the records that fit are kept", info->count == SMI_NIC_MAX_SENSORS);
  check("the last kept record is intact",
        info->sensors[SMI_NIC_MAX_SENSORS - 1].value == (SMI_NIC_MAX_SENSORS - 1));
}

static void test_sensor_shim_struct_truncates_a_long_label_with_a_terminator() {
  auto records = std::vector<hw::SensorRecord_t>{};
  records.push_back(
      make_sensor_record(1, 1, std::string(SMI_NIC_MAX_STRING_LENGTH + 20, 'x').c_str()));

  auto info = std::make_unique<smi_nic_sensor_info_t>();
  smi_nic_sensor_info_from_records_for_testing(records, info.get());

  check("a long label is cut to fit with its terminator",
        std::string(info->sensors[0].label).size() == (SMI_NIC_MAX_STRING_LENGTH - 1));
}

static void test_sensor_shim_struct_of_a_nic_without_hwmon_is_empty() {
  auto info = std::make_unique<smi_nic_sensor_info_t>();
  info->count = 9;  // stale value must not survive
  const auto status = smi_nic_sensor_info_from_records_for_testing({}, info.get());
  check("no records succeed", status == SMI_NIC_STATUS_SUCCESS);
  check("no records give a zero count", info->count == 0);
}

// The context entry points are how libamd_smi reaches telemetry. They resolve
// the NIC under the context lock, so every rejection has to happen there rather
// than in a caller holding a pointer discovery handed out.
static void test_context_entry_points_reject_bad_input() {
  std::cout << "\nContext entry points: rejection paths\n";
  // Formats to a 16-character BDF, which parse_bdf rejects on length, so no
  // sysfs-discovered device can ever occupy it on any host.
  constexpr uint64_t kUnregisteredBdf = 0xffffffffffffffffULL;

  auto telemetry = smi_nic_telemetry_t{};
  smi_nic_fw_info_t info{};
  auto sensors = std::make_unique<smi_nic_sensor_info_t>();

  check("telemetry on a null context is rejected",
        smi_get_nic_telemetry(nullptr, 0, &telemetry) == SMI_NIC_STATUS_WRONG_PARAM);
  check("firmware on a null context is rejected",
        smi_get_nic_fw_info(nullptr, 0, &info) == SMI_NIC_STATUS_WRONG_PARAM);
  check("hwmon sensors on a null context is rejected",
        smi_get_nic_hwmon_sensors(nullptr, 0, sensors.get()) == SMI_NIC_STATUS_WRONG_PARAM);

  smi_nic_ctx_t ctx = nullptr;
  const auto create_status = smi_nic_create_context(&ctx);
  check("context creation succeeds", create_status == SMI_NIC_STATUS_SUCCESS);
  if (create_status != SMI_NIC_STATUS_SUCCESS) {
    return;
  }

  check("telemetry with no out parameter is rejected",
        smi_get_nic_telemetry(ctx, kUnregisteredBdf, nullptr) == SMI_NIC_STATUS_WRONG_PARAM);
  check("firmware with no out parameter is rejected",
        smi_get_nic_fw_info(ctx, kUnregisteredBdf, nullptr) == SMI_NIC_STATUS_WRONG_PARAM);
  check("hwmon sensors with no out parameter is rejected",
        smi_get_nic_hwmon_sensors(ctx, kUnregisteredBdf, nullptr) == SMI_NIC_STATUS_WRONG_PARAM);
  check("telemetry for an unregistered BDF reports not found",
        smi_get_nic_telemetry(ctx, kUnregisteredBdf, &telemetry) == SMI_NIC_STATUS_NOT_FOUND);
  check("firmware for an unregistered BDF reports not found",
        smi_get_nic_fw_info(ctx, kUnregisteredBdf, &info) == SMI_NIC_STATUS_NOT_FOUND);
  check(
      "hwmon sensors for an unregistered BDF reports not found",
      smi_get_nic_hwmon_sensors(ctx, kUnregisteredBdf, sensors.get()) == SMI_NIC_STATUS_NOT_FOUND);

  smi_nic_destroy_context(ctx);
}

// A multi-function card registers its devlink instance on the port function,
// not on the bridge its handle names. Querying the bridge is what made port
// split and firmware read unsupported on hardware. Health is addressed at the
// management function instead, and falls back to the port when a card has none,
// which is the path this covers.
static void test_devlink_addresses_the_port_function() {
  std::cout << "\nAddressing: devlink queries the port, not the bridge\n";
  const std::string kBridge = "0000:01:00.0";
  const std::string kPort = "0000:04:00.0";

  SmiNic nic("", kBridge, NicType::PCIBridge);
  nic.add_nic_port(SmiNicPort("enP0p4s0", kPort, "", ""));

  auto fake = std::make_shared<FakeDevlinkClient>();
  tel::NicTelemetry telemetry(fake);

  telemetry.get_health(nic);
  check("health falls back to the port when no management function is known",
        fake->reporters_dev == kPort, "got " + fake->reporters_dev);

  telemetry.get_port_split(nic);
  check("port split queries the port", fake->port_split_dev == kPort,
        "got " + fake->port_split_dev);

  telemetry.get_device_info(nic);
  check("device info queries the port", fake->device_info_dev == kPort,
        "got " + fake->device_info_dev);
}

// The health reporter belongs to the card's management function (pds_core),
// behind the other downstream port of the internal switch. The ionic port has a
// devlink instance but registers no reporter, so addressing health at the port
// is what made HEALTH read UNSUPPORTED on hardware. Port split and device info
// are answered by the port and must not move with it.
static void test_devlink_health_addresses_the_management_function() {
  std::cout << "\nAddressing: health queries the management function\n";
  const std::string kPort = "0000:04:00.0";
  const std::string kMgmt = "0000:03:00.2";

  SmiNic nic("", "0000:01:00.0", NicType::PCIBridge);
  nic.add_nic_port(SmiNicPort("enP0p4s0", kPort, "", ""));
  nic.set_mgmt_bdf(kMgmt);

  auto fake = std::make_shared<FakeDevlinkClient>();
  tel::NicTelemetry telemetry(fake);

  telemetry.get_health(nic);
  check("health queries the management function", fake->reporters_dev == kMgmt,
        "got " + fake->reporters_dev);

  telemetry.get_port_split(nic);
  check("port split still queries the port", fake->port_split_dev == kPort,
        "got " + fake->port_split_dev);

  telemetry.get_device_info(nic);
  check("device info still queries the port", fake->device_info_dev == kPort,
        "got " + fake->device_info_dev);
}

// A portless fwctl-only NIC registers its devlink instance on its own function,
// so it must keep being addressed there rather than delegating to a port that
// does not exist.
static void test_devlink_addresses_a_portless_nic_itself() {
  std::cout << "\nAddressing: a portless NIC keeps its own address\n";
  const std::string kSelf = "0001:01:00.1";

  SmiNic nic("", kSelf);

  auto fake = std::make_shared<FakeDevlinkClient>();
  tel::NicTelemetry telemetry(fake);

  telemetry.get_health(nic);
  check("health queries the NIC itself", fake->reporters_dev == kSelf,
        "got " + fake->reporters_dev);

  telemetry.get_device_info(nic);
  check("device info queries the NIC itself", fake->device_info_dev == kSelf,
        "got " + fake->device_info_dev);
}

struct ExpectedSensor {
  hw::Category_t category;
  hw::Attribute_t attribute;
  uint32_t channel;
  int64_t value;
};

static void test_hwmon_scan_ionic_layout() {
  std::cout << "\nHwmon scan: ionic node as sensors reports it\n";
  const auto root = write_ionic_hwmon("scan_ionic");
  const auto records = hw::scan_sensors(root);

  const std::vector<ExpectedSensor> expected = {
      {hw::Category_t::Temperature, hw::Attribute_t::Input, 1, 39000},
      {hw::Category_t::Temperature, hw::Attribute_t::Crit, 1, 80000},
      {hw::Category_t::Temperature, hw::Attribute_t::Input, 2, 59000},
      {hw::Category_t::Temperature, hw::Attribute_t::Crit, 2, 105000},
      {hw::Category_t::Power, hw::Attribute_t::Input, 1, 37700000},
      {hw::Category_t::Power, hw::Attribute_t::Max, 1, 40000000},
      {hw::Category_t::Power, hw::Attribute_t::Input, 2, 11000000},
      {hw::Category_t::Voltage, hw::Attribute_t::Input, 0, 12060},
      {hw::Category_t::Voltage, hw::Attribute_t::Input, 1, 744},
  };
  check("one record per recognised attribute", records.size() == expected.size(),
        "got " + std::to_string(records.size()));
  if (records.size() != expected.size()) {
    return;
  }
  bool is_order_and_values_match = true;
  for (size_t i = 0; i < expected.size(); ++i) {
    is_order_and_values_match =
        is_order_and_values_match && (records[i].category == expected[i].category) &&
        (records[i].attribute == expected[i].attribute) &&
        (records[i].channel == expected[i].channel) && (records[i].value == expected[i].value) &&
        (records[i].instance == 3);
  }
  check("ordered by category then channel then attribute, instance = 3", is_order_and_values_match);
  check("board label carried to the input record", records[0].label == "board temperature");
  check("board label carried to its threshold record", records[1].label == "board temperature");
  check("die label on channel 2", records[2].label == "die temperature");
  check("temperature unit is milli-degrees", records[0].unit == hw::Unit_t::MilliDegreeCelsius);
  check("voltage unit is millivolts", records[7].unit == hw::Unit_t::MilliVolt);
  check("power unit is microwatts", records[4].unit == hw::Unit_t::MicroWatt);
}

// File set of a bnxt_en hwmon node as listed on b20-31.
static void test_hwmon_scan_bnxt_layout_reports_shutdown() {
  std::cout << "\nHwmon scan: bnxt node with shutdown threshold and alarm files\n";
  write_sysfs_file("scan_bnxt/hwmon0/temp1_input", "37000\n");
  write_sysfs_file("scan_bnxt/hwmon0/temp1_max", "95000\n");
  write_sysfs_file("scan_bnxt/hwmon0/temp1_crit", "100000\n");
  write_sysfs_file("scan_bnxt/hwmon0/temp1_emergency", "105000\n");
  write_sysfs_file("scan_bnxt/hwmon0/temp1_shutdown", "115000\n");
  write_sysfs_file("scan_bnxt/hwmon0/temp1_max_alarm", "0\n");
  write_sysfs_file("scan_bnxt/hwmon0/temp1_crit_alarm", "0\n");
  write_sysfs_file("scan_bnxt/hwmon0/temp1_emergency_alarm", "0\n");
  write_sysfs_file("scan_bnxt/hwmon0/temp1_shutdown_alarm", "0\n");
  const auto records = hw::scan_sensors(g_tmpdir + "/scan_bnxt");

  check("input, max, crit, emergency and shutdown; alarms skipped", records.size() == 5,
        "got " + std::to_string(records.size()));
  const auto has_shutdown =
      std::any_of(records.begin(), records.end(), [](const hw::SensorRecord_t& record) {
        return ((record.attribute == hw::Attribute_t::Shutdown) && (record.value == 115000));
      });
  check("shutdown threshold is reported", has_shutdown);
}

static void test_hwmon_scan_two_instances_stay_distinct() {
  std::cout << "\nHwmon scan: same channel on two hwmon nodes\n";
  write_sysfs_file("scan_two/hwmon1/temp1_input", "30000\n");
  write_sysfs_file("scan_two/hwmon2/temp1_input", "40000\n");
  const auto records = hw::scan_sensors(g_tmpdir + "/scan_two");

  check("two records", records.size() == 2, "got " + std::to_string(records.size()));
  if (records.size() == 2) {
    check("first is instance 1", (records[0].instance == 1) && (records[0].value == 30000));
    check("second is instance 2", (records[1].instance == 2) && (records[1].value == 40000));
  }
}

static void test_hwmon_scan_missing_label_is_empty() {
  std::cout << "\nHwmon scan: no label file\n";
  write_sysfs_file("scan_nolabel/hwmon0/temp1_input", "46000\n");
  const auto records = hw::scan_sensors(g_tmpdir + "/scan_nolabel");

  check("one record", records.size() == 1);
  check("label is empty", !records.empty() && records[0].label.empty());
}

static void test_hwmon_scan_skips_unreadable_and_unknown_files() {
  std::cout << "\nHwmon scan: unreadable values and files outside the scope\n";
  write_sysfs_file("scan_skip/hwmon0/temp1_input", "not-a-number\n");
  write_sysfs_file("scan_skip/hwmon0/temp1_crit", "105000\n");
  write_sysfs_file("scan_skip/hwmon0/temp1_max_alarm", "0\n");
  write_sysfs_file("scan_skip/hwmon0/power1_average", "5000000\n");
  write_sysfs_file("scan_skip/hwmon0/fan1_input", "1200\n");
  write_sysfs_file("scan_skip/hwmon0/intrusion0_alarm", "0\n");
  write_sysfs_file("scan_skip/hwmon0/name", "x\n");
  const auto records = hw::scan_sensors(g_tmpdir + "/scan_skip");

  check("only the readable crit threshold remains", records.size() == 1,
        "got " + std::to_string(records.size()));
  check("it is temp1 crit", !records.empty() && (records[0].attribute == hw::Attribute_t::Crit));
}

static void test_hwmon_scan_negative_and_wide_values() {
  std::cout << "\nHwmon scan: negative temperature and a power above 32 bits\n";
  write_sysfs_file("scan_wide/hwmon0/temp1_input", "-5000\n");
  write_sysfs_file("scan_wide/hwmon0/power1_input", "3000000000\n");
  const auto records = hw::scan_sensors(g_tmpdir + "/scan_wide");

  check("two records", records.size() == 2);
  if (records.size() == 2) {
    check("negative temperature kept", records[0].value == -5000);
    check("value above INT32_MAX kept", records[1].value == 3000000000LL);
  }
}

static void test_hwmon_scan_absent_root_is_empty() {
  std::cout << "\nHwmon scan: directory missing\n";
  check("no records", hw::scan_sensors(g_tmpdir + "/does_not_exist").empty());
}

static void test_temperature_role_ionic() {
  std::cout << "\nTemperature role: labelled ionic node\n";
  const auto root = write_ionic_hwmon("role_ionic");
  const auto asic = hw::find_temperature_input(root, hw::TempRole_t::Asic);
  const auto board = hw::find_temperature_input(root, hw::TempRole_t::Board);

  check("asic is the die channel",
        asic.has_value() && (asic.value() == root + "/hwmon3/temp2_input"));
  check("board is the board channel",
        board.has_value() && (board.value() == root + "/hwmon3/temp1_input"));
}

static void test_temperature_role_unlabelled_driver() {
  std::cout << "\nTemperature role: driver that labels nothing\n";
  write_sysfs_file("role_plain/hwmon0/temp1_input", "46000\n");
  const auto root = g_tmpdir + "/role_plain";

  const auto asic = hw::find_temperature_input(root, hw::TempRole_t::Asic);
  check("asic falls back to temp1",
        asic.has_value() && (asic.value() == root + "/hwmon0/temp1_input"));
  check("board is absent", !hw::find_temperature_input(root, hw::TempRole_t::Board).has_value());
}

static void test_temperature_role_board_only_is_not_asic() {
  std::cout << "\nTemperature role: only a board channel exists\n";
  write_sysfs_file("role_boardonly/hwmon0/temp1_input", "39000\n");
  write_sysfs_file("role_boardonly/hwmon0/temp1_label", "board temperature\n");
  const auto root = g_tmpdir + "/role_boardonly";

  check("asic must not borrow the board channel",
        !hw::find_temperature_input(root, hw::TempRole_t::Asic).has_value());
  check("board still found", hw::find_temperature_input(root, hw::TempRole_t::Board).has_value());
}

static void test_temperature_role_unlabeled_non_first_channel() {
  std::cout << "\nTemperature role: unlabelled driver with only temp2\n";
  write_sysfs_file("role_temp2only/hwmon0/temp2_input", "46000\n");
  const auto root = g_tmpdir + "/role_temp2only";

  check("asic does not fall back to temp2",
        !hw::find_temperature_input(root, hw::TempRole_t::Asic).has_value());
}

static void test_temperature_role_mixed_labels_skip_fallback() {
  std::cout << "\nTemperature role: unlabelled temp1 beside a labelled die channel\n";
  write_sysfs_file("role_mixed/hwmon0/temp1_input", "39000\n");
  write_sysfs_file("role_mixed/hwmon0/temp2_input", "59000\n");
  write_sysfs_file("role_mixed/hwmon0/temp2_label", "die temperature\n");
  const auto root = g_tmpdir + "/role_mixed";

  const auto asic = hw::find_temperature_input(root, hw::TempRole_t::Asic);
  check("asic is the labelled die channel, not temp1",
        asic.has_value() && (asic.value() == root + "/hwmon0/temp2_input"));
  check("board is absent", !hw::find_temperature_input(root, hw::TempRole_t::Board).has_value());
}

static void test_temperature_role_label_case_insensitive() {
  std::cout << "\nTemperature role: labels in upper and mixed case\n";
  write_sysfs_file("role_case/hwmon0/temp1_input", "39000\n");
  write_sysfs_file("role_case/hwmon0/temp1_label", "BOARD Temperature\n");
  write_sysfs_file("role_case/hwmon0/temp2_input", "59000\n");
  write_sysfs_file("role_case/hwmon0/temp2_label", "Die Temperature\n");
  const auto root = g_tmpdir + "/role_case";

  const auto asic = hw::find_temperature_input(root, hw::TempRole_t::Asic);
  const auto board = hw::find_temperature_input(root, hw::TempRole_t::Board);
  check("asic matches 'Die Temperature'",
        asic.has_value() && (asic.value() == root + "/hwmon0/temp2_input"));
  check("board matches 'BOARD Temperature'",
        board.has_value() && (board.value() == root + "/hwmon0/temp1_input"));
}

static void test_temperature_role_absent_root() {
  std::cout << "\nTemperature role: directory missing\n";
  const auto root = g_tmpdir + "/does_not_exist";
  check("asic absent", !hw::find_temperature_input(root, hw::TempRole_t::Asic).has_value());
  check("board absent", !hw::find_temperature_input(root, hw::TempRole_t::Board).has_value());
}

int main() {
  char tmpl[] = "/tmp/amdsmi_telemetry_testXXXXXX";
  if (mkdtemp(tmpl) == nullptr) {
    std::cerr << "failed to create temp dir\n";
    return 2;
  }
  g_tmpdir = tmpl;

  std::cout << "NIC telemetry facade tests\n";
  std::cout << "temp dir: " << g_tmpdir << "\n";

  test_temperature_partial();
  test_temperature_unsupported();
  test_temperature_present_but_unreadable();
  test_health_supported();
  test_health_reporter_name_varies();
  test_health_unhealthy();
  test_health_multi_reporter_aggregate();
  test_health_error_count_saturates_below_sentinel();
  test_health_no_reporter();
  test_health_devlink_failure();
  test_port_split_active();
  test_port_split_not_splittable();
  test_port_split_attrs_absent();
  test_port_split_unsupported();
  test_device_info_supported();
  test_device_info_unsupported();
  test_fw_info_keeps_each_devlink_version_with_its_type();
  test_fw_info_of_a_device_without_versions_is_empty();
  test_telemetry_shim_struct_copies_every_snapshot_field();
  test_sensor_shim_struct_copies_every_record_field();
  test_sensor_shim_struct_reports_overflow_and_keeps_what_fits();
  test_sensor_shim_struct_truncates_a_long_label_with_a_terminator();
  test_sensor_shim_struct_of_a_nic_without_hwmon_is_empty();
  test_context_entry_points_reject_bad_input();
  test_devlink_addresses_the_port_function();
  test_devlink_health_addresses_the_management_function();
  test_devlink_addresses_a_portless_nic_itself();
  test_snapshot_all_supported();
  test_snapshot_mixed();
  test_snapshot_all_unsupported();
  test_hwmon_scan_ionic_layout();
  test_hwmon_scan_bnxt_layout_reports_shutdown();
  test_hwmon_scan_two_instances_stay_distinct();
  test_hwmon_scan_missing_label_is_empty();
  test_hwmon_scan_skips_unreadable_and_unknown_files();
  test_hwmon_scan_negative_and_wide_values();
  test_hwmon_scan_absent_root_is_empty();
  test_temperature_role_ionic();
  test_temperature_role_unlabelled_driver();
  test_temperature_role_board_only_is_not_asic();
  test_temperature_role_unlabeled_non_first_channel();
  test_temperature_role_mixed_labels_skip_fallback();
  test_temperature_role_label_case_insensitive();
  test_temperature_role_absent_root();

  auto ec = std::error_code{};
  std::filesystem::remove_all(g_tmpdir, ec);
  return nic_test::report();
}
