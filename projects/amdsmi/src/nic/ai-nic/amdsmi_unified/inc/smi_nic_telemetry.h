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

#ifndef AMDSMI_UNIFIED_NIC_TELEMETRY_H_
#define AMDSMI_UNIFIED_NIC_TELEMETRY_H_

#include <cstdint>
#include <memory>

#include "smi_devlink_netlink.h"
#include "smi_nic.h"
#include "smi_nic_transport.h"

namespace amd::smi::nic::telemetry {

/**
 * Sentinels: a field a NIC does not expose is reported with these rather than a
 * fabricated value, so a caller can distinguish "unsupported" from a real read.
 */
constexpr uint16_t kTempUnsupported = UINT16_MAX;
constexpr uint32_t kErrorCountUnsupported = UINT32_MAX;
constexpr uint8_t kCountUnsupported = UINT8_MAX;

/**
 * The full set is intentional even though devlink's health-reporter state is
 * binary today (healthy vs error): the public mirror exposes all five, with the
 * two currently-unproduced values documented as reserved so they can be filled
 * later without a breaking enum change.
 */
enum class HealthState : uint8_t {
  Unknown = 0,  // reserved: a reporter exists but its state is indeterminate (no source yet)
  Healthy = 1,
  Warning = 2,  // reserved: no source yet
  Error = 3,
  Unsupported = 4,  // the device exposes no health reporter at all (distinct from Unknown)
};

/**
 * The structs below are plain, fixed-size, and standard-layout on purpose: a
 * future public amdsmi_nic_*_t can bridge to them via reinterpret_cast plus a
 * sizeof static_assert, matching the existing smi_nic_*_t <-> amdsmi_nic_*_t
 * pattern. Keep them free of std::string/std::vector.
 */

/** Board temperatures in whole degrees Celsius; kTempUnsupported if not exposed. */
struct NicTemperature {
  uint16_t asic_temp_c;
  uint16_t transceiver_temp_c;
  uint16_t board_temp_c;
};

struct NicHealth {
  uint8_t state;         // a HealthState value
  uint32_t error_count;  // kErrorCountUnsupported if not exposed
  char reporter[64];     // devlink reporter name, NUL-terminated; "" if none
};

struct NicPortSplit {
  uint8_t splittable;   // 1/0, or kCountUnsupported if unknown
  uint8_t split_count;  // current sub-ports (0 if not split), kCountUnsupported if unknown
};

/**
 * All live telemetry in one read. Standard-layout (three standard-layout PODs),
 * so a future public bridge can map it field-for-field. A metric the NIC does
 * not expose is written as sentinels in its sub-struct (asic/transceiver/board =
 * kTempUnsupported; health state Unknown + error_count kErrorCountUnsupported +
 * empty reporter; splittable/split_count = kCountUnsupported), so the snapshot
 * always populates for a valid NIC rather than failing whole.
 */
struct NicTelemetrySnapshot {
  NicTemperature temperature;
  NicHealth health;
  NicPortSplit port_split;
};

/**
 * Live, on-demand NIC telemetry. Every getter reads at call time (no init-time
 * caching): temperature via sysfs/hwmon, health and port-split via devlink.
 *
 * A metric a vendor does not expose at all yields Result.success == false with
 * error_code == ENOTSUP. A partially-supported metric yields success with the
 * unavailable fields set to their sentinel.
 */
class NicTelemetry {
 public:
  explicit NicTelemetry(std::shared_ptr<::amd::nic::netlink::IDevlinkClient> devlink);

  transport::Result<NicTemperature> get_temperature(const ::SmiNic& nic) const;
  transport::Result<NicHealth> get_health(const ::SmiNic& nic) const;
  transport::Result<NicPortSplit> get_port_split(const ::SmiNic& nic) const;

  /**
   * Reads identity/firmware via devlink (DEVLINK_CMD_INFO_GET). This is the only
   * source for a portless (fwctl-only) card with no host netdev. Unlike the
   * getters above it performs no aggregation — the version list is an
   * intentionally raw, vendor-defined name/value set — so it forwards the devlink
   * result unchanged rather than mirroring it into a separate telemetry struct.
   */
  transport::Result<::amd::nic::netlink::DevlinkDeviceInfo> get_device_info(
      const ::SmiNic& nic) const;

  /**
   * One-shot read of all three metrics. Always succeeds for a valid NIC; a
   * metric the NIC does not expose is reported via its sub-struct sentinels
   * (see NicTelemetrySnapshot). This is the seam a future public bridge forwards
   * to, keeping the sentinel-fallback policy next to the code that produces it.
   */
  transport::Result<NicTelemetrySnapshot> get_snapshot(const ::SmiNic& nic) const;

 private:
  std::shared_ptr<::amd::nic::netlink::IDevlinkClient> devlink_;
};

}  // namespace amd::smi::nic::telemetry

#endif  // AMDSMI_UNIFIED_NIC_TELEMETRY_H_
