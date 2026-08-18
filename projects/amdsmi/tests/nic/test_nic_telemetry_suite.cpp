// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Unit tests for the NIC telemetry facade (temperature / health / port-split).
 *
 * These tests are hardware-independent: devlink is exercised through a fake
 * IDevlinkClient, and sysfs/hwmon through fake files written to a temp dir. They
 * validate the facade's uniform-schema-plus-unsupported contract, not any real
 * device.
 */

#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "smi_nic_telemetry.h"

namespace tp = amd::smi::nic::transport;
namespace tel = amd::smi::nic::telemetry;
namespace dl = amd::nic::netlink;

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

// Test Infrastructure

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

// Configurable devlink client: tests set the results the facade will observe.
class FakeDevlinkClient : public dl::IDevlinkClient {
 public:
  tp::Result<dl::DevlinkPortSplit> port_split{false, {}, ENOTSUP};
  tp::Result<std::vector<dl::DevlinkReporter>> reporters{false, {}, ENOTSUP};
  tp::Result<dl::DevlinkDeviceInfo> device_info{false, {}, ENOTSUP};

  tp::Result<dl::DevlinkPortSplit> get_port_split(const std::string& /*dev*/) override {
    return port_split;
  }
  tp::Result<std::vector<dl::DevlinkReporter>> get_health_reporters(
      const std::string& /*dev*/) override {
    return reporters;
  }
  tp::Result<dl::DevlinkDeviceInfo> get_device_info(const std::string& /*dev*/) override {
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

// Test cases

static void test_temperature_partial() {
  std::cout << "\nTemperature: asic supported, others absent\n";
  FakeNic nic;
  nic.temp_paths[NicTempSensor::Asic] = write_temp_file("temp1_input", "45000\n");

  tel::NicTelemetry telemetry(std::make_shared<FakeDevlinkClient>());
  auto r = telemetry.get_temperature(nic);

  check("call succeeds", r.success, "errno=" + std::to_string(r.error_code));
  check("asic = 45C", r.value.asic_temp_c == 45, "got " + std::to_string(r.value.asic_temp_c));
  check("transceiver = sentinel", r.value.transceiver_temp_c == tel::kTempUnsupported);
  check("board = sentinel", r.value.board_temp_c == tel::kTempUnsupported);
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
  check("asic = sentinel (unreadable)", r.value.asic_temp_c == tel::kTempUnsupported);
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
  std::cout << "\nHealth: overflowing error_count saturates below the unsupported sentinel\n";
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
  check("error_count is not the unsupported sentinel",
        r.value.error_count != tel::kErrorCountUnsupported,
        "got " + std::to_string(r.value.error_count));
  check("error_count saturated to sentinel-1",
        r.value.error_count == tel::kErrorCountUnsupported - 1,
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
  check("temp asic = sentinel", r.value.temperature.asic_temp_c == tel::kTempUnsupported);
  check("temp board = sentinel", r.value.temperature.board_temp_c == tel::kTempUnsupported);
  check("health present = Healthy",
        r.value.health.state == static_cast<uint8_t>(tel::HealthState::Healthy));
  check("port splittable = sentinel", r.value.port_split.splittable == tel::kCountUnsupported);
  check("port split_count = sentinel", r.value.port_split.split_count == tel::kCountUnsupported);
}

static void test_snapshot_all_unsupported() {
  std::cout << "\nSnapshot: nothing exposed (ionic)\n";
  FakeNic nic;  // no temp sensors

  auto fake = std::make_shared<FakeDevlinkClient>();
  fake->reporters = {true, {}, 0};  // enumeration succeeded but empty -> unsupported
  // port_split left at default ENOTSUP

  tel::NicTelemetry telemetry(fake);
  auto r = telemetry.get_snapshot(nic);

  check("snapshot still succeeds (per-field sentinels, not whole failure)", r.success);
  check("temp asic = sentinel", r.value.temperature.asic_temp_c == tel::kTempUnsupported);
  check("health state = Unsupported (distinct from Unknown)",
        r.value.health.state == static_cast<uint8_t>(tel::HealthState::Unsupported));
  check("health error_count = sentinel", r.value.health.error_count == tel::kErrorCountUnsupported);
  check("health reporter = empty", r.value.health.reporter[0] == '\0');
  check("port splittable = sentinel", r.value.port_split.splittable == tel::kCountUnsupported);
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
        r.value.versions[1].type == static_cast<uint8_t>(dl::DevlinkVersionType::Running) &&
            std::strcmp(r.value.versions[1].name, "fw") == 0 &&
            std::strcmp(r.value.versions[1].value, "1.60.0-73") == 0);
}

static void test_device_info_unsupported() {
  std::cout << "\nDevice info: device has no devlink instance\n";
  FakeNic nic;

  tel::NicTelemetry telemetry(std::make_shared<FakeDevlinkClient>());  // default ENOTSUP
  auto r = telemetry.get_device_info(nic);

  check("call reports unsupported", !r.success);
  check("errno = ENOTSUP", r.error_code == ENOTSUP, "got " + std::to_string(r.error_code));
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
  test_port_split_unsupported();
  test_device_info_supported();
  test_device_info_unsupported();
  test_snapshot_all_supported();
  test_snapshot_mixed();
  test_snapshot_all_unsupported();

  std::cout << "\n" << (tests_run - tests_failed) << "/" << tests_run << " checks passed\n";
  return tests_failed == 0 ? 0 : 1;
}
