/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "smi_nic_telemetry.h"

#include <cerrno>
#include <cstdio>
#include <utility>
#include <variant>
#include <vector>

#include "smi_sysfs.h"

/**
 * The devlink client factory lives here (always compiled) rather than in
 * smi_devlink_netlink.cpp, which the build excludes when libnl-3 is absent.
 */
namespace amd::nic::netlink {

#ifndef HAVE_LIBNL3
namespace {
/**
 * Fallback used when the library is built without libnl-3: devlink queries are
 * unavailable, so every call reports ENOTSUP.
 */
class UnsupportedDevlinkClient : public IDevlinkClient {
 public:
  transport::Result<DevlinkPortSplit> get_port_split(const std::string& /*dev*/) override {
    return {false, {}, ENOTSUP};
  }
  transport::Result<std::vector<DevlinkReporter>> get_health_reporters(
      const std::string& /*dev*/) override {
    return {false, {}, ENOTSUP};
  }
  transport::Result<DevlinkDeviceInfo> get_device_info(const std::string& /*dev*/) override {
    return {false, {}, ENOTSUP};
  }
};
}  // namespace
#endif  // !HAVE_LIBNL3

std::shared_ptr<IDevlinkClient> create_devlink_client() {
#ifdef HAVE_LIBNL3
  return std::make_shared<DevlinkNetlinkClient>();
#else
  return std::make_shared<UnsupportedDevlinkClient>();
#endif
}

}  // namespace amd::nic::netlink

namespace amd::smi::nic::telemetry {

NicTelemetry::NicTelemetry(std::shared_ptr<::amd::nic::netlink::IDevlinkClient> devlink)
    : devlink_(std::move(devlink)) {}

transport::Result<NicTemperature> NicTelemetry::get_temperature(const ::SmiNic& nic) const {
  NicTemperature temp{kTempUnsupported, kTempUnsupported, kTempUnsupported};

  const NicTempSensor sensors[] = {NicTempSensor::Asic, NicTempSensor::Transceiver,
                                   NicTempSensor::Board};
  uint16_t* const fields[] = {&temp.asic_temp_c, &temp.transceiver_temp_c, &temp.board_temp_c};

  bool any_supported = false;
  for (size_t i = 0; i < 3; ++i) {
    const auto path = nic.hwmon_temp_path(sensors[i]);
    if (!path.has_value()) {
      continue;  // this vendor does not expose this sensor
    }
    any_supported = true;

    SmiSysfsReader::SysfsValue value;
    if (SmiSysfsReader::readLine(path.value(), value) != SmiSysfsReader::SysfsStatus::Success) {
      continue;  // present but unreadable: leave the sentinel
    }
    if (!std::holds_alternative<int>(value)) {
      continue;
    }
    const int millidegrees = std::get<int>(value);  // hwmon tempN_input units
    if (millidegrees < 0) {
      continue;
    }
    *fields[i] = static_cast<uint16_t>(millidegrees / 1000);
  }

  if (!any_supported) {
    return {false, {}, ENOTSUP};
  }
  return {true, temp, 0};
}

transport::Result<NicHealth> NicTelemetry::get_health(const ::SmiNic& nic) const {
  const auto reporters = devlink_->get_health_reporters(nic.bdf());
  if (!reporters.success) {
    return {false, {}, reporters.error_code};
  }
  if (reporters.value.empty()) {
    return {false, {}, ENOTSUP};  // device exposes no reporter (e.g. ionic)
  }

  /**
   * Aggregate across reporters: the NIC is Error if any reporter is unhealthy;
   * error_count is the total; the reported name is the first unhealthy reporter,
   * else the first reporter. A single-reporter NIC (the common case) reduces to
   * copying that reporter through.
   */
  NicHealth health{};
  health.state = static_cast<uint8_t>(HealthState::Healthy);
  uint64_t error_sum = 0;
  const char* name = reporters.value.front().name;
  for (const auto& r : reporters.value) {
    error_sum += r.error_count;
    if (!r.healthy && health.state != static_cast<uint8_t>(HealthState::Error)) {
      health.state = static_cast<uint8_t>(HealthState::Error);
      name = r.name;
    }
  }
  /**
   * Saturate one below the "unsupported" sentinel so a huge (or overflowing)
   * count is never misread as kErrorCountUnsupported. UINT32_MAX is
   * reserved to mean "not exposed" and must not be producible by a real reading.
   */
  constexpr uint32_t kMaxErrorCount = kErrorCountUnsupported - 1;
  health.error_count =
      error_sum >= kMaxErrorCount ? kMaxErrorCount : static_cast<uint32_t>(error_sum);
  std::snprintf(health.reporter, sizeof(health.reporter), "%s", name);
  return {true, health, 0};
}

transport::Result<NicPortSplit> NicTelemetry::get_port_split(const ::SmiNic& nic) const {
  const auto result = devlink_->get_port_split(nic.bdf());
  if (!result.success) {
    return {false, {}, result.error_code};
  }

  NicPortSplit split{};
  split.splittable = result.value.splittable;
  split.split_count = result.value.split_count > kCountUnsupported
                          ? kCountUnsupported
                          : static_cast<uint8_t>(result.value.split_count);
  return {true, split, 0};
}

transport::Result<::amd::nic::netlink::DevlinkDeviceInfo> NicTelemetry::get_device_info(
    const ::SmiNic& nic) const {
  return devlink_->get_device_info(nic.bdf());
}

transport::Result<NicTelemetrySnapshot> NicTelemetry::get_snapshot(const ::SmiNic& nic) const {
  NicTelemetrySnapshot snap{};

  const auto temp = get_temperature(nic);
  snap.temperature = temp.success
                         ? temp.value
                         : NicTemperature{kTempUnsupported, kTempUnsupported, kTempUnsupported};

  const auto health = get_health(nic);
  if (health.success) {
    snap.health = health.value;
  } else {
    /**
     * No reporter exposed: distinct Unsupported state (not Unknown, which means
     * "reporter present but indeterminate"), with the error_count sentinel and
     * the empty reporter left by value-initialization.
     */
    snap.health.state = static_cast<uint8_t>(HealthState::Unsupported);
    snap.health.error_count = kErrorCountUnsupported;
  }

  const auto split = get_port_split(nic);
  snap.port_split =
      split.success ? split.value : NicPortSplit{kCountUnsupported, kCountUnsupported};

  return {true, snap, 0};
}

}  // namespace amd::smi::nic::telemetry
