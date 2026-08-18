/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AMDSMI_UNIFIED_DEVLINK_NETLINK_H_
#define AMDSMI_UNIFIED_DEVLINK_NETLINK_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "smi_nic_transport.h"

#ifdef HAVE_LIBNL3
#include "netlink_generic.h"
#endif

namespace amd::nic::netlink {

/**
 * The transport result wrapper lives in amd::smi::nic::transport; alias it so it
 * reads as transport:: here, consistent with the rest of the netlink layer.
 */
namespace transport = ::amd::smi::nic::transport;

/**
 * Port-split capability/state, read from devlink (DEVLINK_CMD_PORT_GET). devlink
 * exposes only the capability bit and the current count (there is no "max
 * sub-ports" attribute), so the schema stops there rather than fabricating one.
 */
struct DevlinkPortSplit {
  uint8_t splittable;    // DEVLINK_ATTR_PORT_SPLITTABLE: 1 if the port can split
  uint32_t split_count;  // DEVLINK_ATTR_PORT_SPLIT_COUNT: current sub-ports (0 if not split)
};

/**
 * One devlink health reporter and its state, read from devlink
 * (DEVLINK_CMD_HEALTH_REPORTER_GET). A device may expose zero or more reporters,
 * and their names are not stable across drivers or firmware (e.g. bnxt_en names
 * it "fw" on some parts and "fw_reset" on others; ionic exposes none), so
 * callers enumerate them rather than querying a fixed name.
 */
struct DevlinkReporter {
  char name[32];         // reporter name, NUL-terminated (e.g. "fw", "fw_reset")
  uint8_t healthy;       // 1 if the reporter state is healthy, else 0
  uint32_t error_count;  // reporter error count, saturated to UINT32_MAX
};

/**
 * Upper bound on version entries retained from DEVLINK_CMD_INFO_GET. Real
 * devices expose well under this (pds_core ~6, bnxt_en ~8); excess entries in a
 * pathological reply are dropped rather than growing the fixed-size struct.
 */
constexpr int kMaxDevlinkVersions = 32;

/** Which devlink version class an entry came from. */
enum class DevlinkVersionType : uint8_t {
  Fixed = 0,    // DEVLINK_ATTR_INFO_VERSION_FIXED: immutable board/asic identity
  Running = 1,  // DEVLINK_ATTR_INFO_VERSION_RUNNING: firmware currently running
  Stored = 2,   // DEVLINK_ATTR_INFO_VERSION_STORED: firmware staged in flash
};

/**
 * One devlink version pair. name/value are vendor-defined and not stable across
 * drivers (e.g. "asic.id", "fw", "fw.mainfwa"), so callers match on name at the
 * point of use rather than assuming a fixed schema.
 */
struct DevlinkVersion {
  uint8_t type;    // a DevlinkVersionType
  char name[32];   // version key, NUL-terminated
  char value[64];  // version string, NUL-terminated
};

/**
 * Device identity and firmware from devlink (DEVLINK_CMD_INFO_GET). This is the
 * only netlink-reachable source for a portless (fwctl-only) card that has no
 * host netdev to query via ethtool. String fields are "" when the device omits
 * them; versions holds version_count entries.
 *
 * serial_number is the DEVLINK_ATTR_INFO_SERIAL_NUMBER value, which is the
 * MAC-derived EUI, not the board's printed serial; board_serial_number
 * (DEVLINK_ATTR_INFO_BOARD_SERIAL_NUMBER) is the physical board serial.
 */
struct DevlinkDeviceInfo {
  char driver_name[32];          // DEVLINK_ATTR_INFO_DRIVER_NAME, "" if absent
  char serial_number[64];        // DEVLINK_ATTR_INFO_SERIAL_NUMBER (EUI), "" if absent
  char board_serial_number[64];  // DEVLINK_ATTR_INFO_BOARD_SERIAL_NUMBER, "" if absent
  uint8_t version_count;         // number of valid entries in versions[]
  DevlinkVersion versions[kMaxDevlinkVersions];
};

/**
 * Abstraction over the devlink generic-netlink family. Declared without any
 * libnl dependency so the telemetry facade and its unit tests can depend on it
 * unconditionally; only the concrete DevlinkNetlinkClient needs libnl-3. `dev`
 * is the devlink device name, i.e. the PCI BDF (e.g. "0000:03:00.0").
 */
class IDevlinkClient {
 public:
  virtual ~IDevlinkClient() = default;

  virtual transport::Result<DevlinkPortSplit> get_port_split(const std::string& dev) = 0;

  /**
   * Enumerates every health reporter for `dev`. Success with an empty vector
   * means the device exposes no reporter (report as unsupported upstream).
   */
  virtual transport::Result<std::vector<DevlinkReporter>> get_health_reporters(
      const std::string& dev) = 0;

  /**
   * Reads identity and firmware versions for `dev` (DEVLINK_CMD_INFO_GET).
   * Failure (device without a devlink instance) is reported as ENOTSUP.
   */
  virtual transport::Result<DevlinkDeviceInfo> get_device_info(const std::string& dev) = 0;
};

/**
 * Returns the netlink-backed client when built with HAVE_LIBNL3, otherwise a
 * stub whose queries report ENOTSUP. Never returns nullptr.
 */
std::shared_ptr<IDevlinkClient> create_devlink_client();

#ifdef HAVE_LIBNL3
/**
 * devlink generic-netlink client. Not thread-safe; one instance per thread.
 * Connects and resolves the "devlink" family lazily on first query.
 */
class DevlinkNetlinkClient : public IDevlinkClient {
 public:
  DevlinkNetlinkClient() = default;
  ~DevlinkNetlinkClient() override = default;

  transport::Result<DevlinkPortSplit> get_port_split(const std::string& dev) override;
  transport::Result<std::vector<DevlinkReporter>> get_health_reporters(
      const std::string& dev) override;
  transport::Result<DevlinkDeviceInfo> get_device_info(const std::string& dev) override;

 private:
  /**
   * Connects the socket and resolves the "devlink" family id on first use.
   * Returns 0 on success or a negative libnl error code.
   */
  int init();

  GenericNetlinkClient client_;
  int family_id_ = -1;
  bool initialized_ = false;
};
#endif  // HAVE_LIBNL3

}  // namespace amd::nic::netlink

#endif  // AMDSMI_UNIFIED_DEVLINK_NETLINK_H_
