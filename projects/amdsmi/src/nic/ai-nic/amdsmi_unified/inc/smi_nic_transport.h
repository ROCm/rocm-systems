// SPDX-License-Identifier: MIT
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

#ifndef AMDSMI_UNIFIED_NIC_TRANSPORT_H_
#define AMDSMI_UNIFIED_NIC_TRANSPORT_H_

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace amd::smi::nic::transport {

// Success/failure state plus error info for a transport operation.
template <typename T>
struct Result {
  bool success;
  T value;         // valid only if success
  int error_code;  // errno; 0 on success

  std::optional<T> as_optional() const { return success ? std::optional<T>(value) : std::nullopt; }
};

// Flow-control pause frame configuration. ethtool -a
struct PauseParams {
  bool autoneg;
  bool rx_pause;
  bool tx_pause;
};

// Physical-layer link configuration and status. ethtool <iface>
struct LinkSettings {
  uint32_t speed;        // Mbps; 0 if link down
  uint8_t duplex;        // DUPLEX_HALF / DUPLEX_FULL / DUPLEX_UNKNOWN
  uint8_t autoneg;       // AUTONEG_DISABLE / AUTONEG_ENABLE
  uint32_t supported;    // bitmask of link modes
  uint32_t advertising;  // bitmask of link modes
};

// NIC driver and firmware information. ethtool -i
struct DriverInfo {
  std::string driver_name;
  std::string version;
  std::string fw_version;
  std::string bus_info;
  uint32_t n_stats;
};

// Driver-specific statistics counters. ethtool -S
struct VendorStatistics {
  std::vector<std::string> names;  // parallel to values
  std::vector<uint64_t> values;    // parallel to names
};

// Factory-programmed MAC address. ethtool -P
struct PermanentAddress {
  std::array<uint8_t, 6> mac;
};

/**
 * Parse a successful ETHTOOL_GPERMADDR result. reported_size is the byte count
 * the kernel writes back; anything other than the 6-byte MAC length means no
 * permanent address is programmed, so we fail (ENODATA) rather than fabricate
 * 00:00:00:00:00:00. Pure logic, no syscall; the ioctl I/O stays in the backend.
 */
Result<PermanentAddress> parse_perm_addr(uint32_t reported_size, const uint8_t* mac_bytes);

/**
 * Abstract transport for querying network device parameters. Implementations
 * differ by mechanism (ioctl, netlink). Not internally thread-safe: the netlink
 * backend holds a persistent socket with per-request sequence numbers, so a
 * shared instance must be serialized by the caller (each smi_nic_ctx does this
 * under its ctx_mutex).
 */
class NicTransport {
 public:
  virtual ~NicTransport() = default;

  // ethtool -a; returns autoneg, rx, tx in a single call.
  virtual Result<PauseParams> get_pause_params(const std::string& iface) = 0;

  // ethtool <iface>
  virtual Result<LinkSettings> get_link_settings(const std::string& iface) = 0;

  // ethtool -i
  virtual Result<DriverInfo> get_driver_info(const std::string& iface) = 0;

  // ethtool -S; internally queries driver info, stat names, and stat values.
  virtual Result<VendorStatistics> get_statistics(const std::string& iface) = 0;

  // ethtool -P
  virtual Result<PermanentAddress> get_permanent_address(const std::string& iface) = 0;

  // Backend identifier, e.g. "ioctl", "netlink", "auto".
  virtual std::string backend_name() const = 0;
};

/**
 * Transport backend selector. Netlink needs kernel 5.6+ and libnl-3
 * (HAVE_LIBNL3); without it, Netlink and Auto fall back to ioctl.
 */
enum class NicBackend_t { Auto, Ioctl, Netlink };

/**
 * Creates a transport backend. Auto tries netlink first, then ioctl.
 * Thread-safe; each instance is independent.
 */
std::shared_ptr<NicTransport> create_transport(NicBackend_t backend = NicBackend_t::Auto);

// Internal factory functions for backend implementations
namespace ioctl_internal {
std::shared_ptr<NicTransport> create_ioctl_backend();
}

#ifdef HAVE_LIBNL3
namespace internal {
std::shared_ptr<NicTransport> create_netlink_backend();
std::shared_ptr<NicTransport> create_auto_backend();
}  // namespace internal
#endif

}  // namespace amd::smi::nic::transport

#endif  // AMDSMI_UNIFIED_NIC_TRANSPORT_H_
