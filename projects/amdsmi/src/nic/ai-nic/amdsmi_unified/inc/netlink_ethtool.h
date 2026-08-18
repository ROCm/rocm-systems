// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Ethtool netlink client. Requires Linux kernel 5.6+ with ethtool netlink.
 */

#ifndef AMDSMI_UNIFIED_NETLINK_ETHTOOL_H_
#define AMDSMI_UNIFIED_NETLINK_ETHTOOL_H_

#include <linux/ethtool_netlink.h>

#include <string>

#include "netlink_generic.h"
#include "smi_nic_transport.h"

namespace amd::nic::netlink {
/**
 * The transport data structs live in amd::smi::nic::transport; alias them so
 * they read as transport:: here, consistent with the netlink layer's namespace.
 */
namespace transport = ::amd::smi::nic::transport;

/**
 * Ethtool-specific netlink queries.
 * Not thread-safe: one instance per thread.
 */
class EthtoolNetlinkClient {
 public:
  EthtoolNetlinkClient();
  ~EthtoolNetlinkClient() = default;

  EthtoolNetlinkClient(const EthtoolNetlinkClient&) = delete;
  EthtoolNetlinkClient& operator=(const EthtoolNetlinkClient&) = delete;
  EthtoolNetlinkClient(EthtoolNetlinkClient&&) = default;
  EthtoolNetlinkClient& operator=(EthtoolNetlinkClient&&) = default;

  // Connects the socket and resolves the ethtool family. Returns 0 or -errno.
  int init();

  bool is_initialized() const { return initialized_; }

  transport::Result<transport::LinkSettings> get_link_settings(const std::string& iface);
  transport::Result<transport::PauseParams> get_pause_params(const std::string& iface);

  // Not available over netlink; always fails with ENOTSUP.
  transport::Result<transport::DriverInfo> get_driver_info(const std::string& iface);
  transport::Result<transport::VendorStatistics> get_statistics(const std::string& iface);

 private:
  GenericNetlinkClient client_;
  int family_id_;
  bool initialized_;

  /**
   * Adds the per-message header nest (header_attr, an ETHTOOL_A_*_HEADER id)
   * carrying the interface name.
   */
  int build_header(NLMessage& msg, const std::string& iface, int header_attr);
};

}  // namespace amd::nic::netlink

#endif  // AMDSMI_UNIFIED_NETLINK_ETHTOOL_H_
