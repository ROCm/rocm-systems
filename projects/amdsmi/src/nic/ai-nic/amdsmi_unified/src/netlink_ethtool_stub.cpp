// SPDX-License-Identifier: MIT
/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * Ethtool netlink client for builds whose UAPI headers predate ethtool netlink. init() fails, so
 * the auto backend uses ioctl and every query reports ENOTSUP.
 */

#include <cerrno>
#include <string>

#include "netlink_ethtool.h"

namespace amd::nic::netlink {

EthtoolNetlinkClient::EthtoolNetlinkClient() : client_(), family_id_(-1), is_initialized_(false) {}

int EthtoolNetlinkClient::init() { return -ENOTSUP; }

transport::Result<transport::LinkSettings> EthtoolNetlinkClient::get_link_settings(
    const std::string& /* iface */) {
  return {false, {}, ENOTSUP};
}

transport::Result<transport::PauseParams> EthtoolNetlinkClient::get_pause_params(
    const std::string& /* iface */) {
  return {false, {}, ENOTSUP};
}

transport::Result<transport::DriverInfo> EthtoolNetlinkClient::get_driver_info(
    const std::string& /* iface */) {
  return {false, {}, ENOTSUP};
}

transport::Result<transport::VendorStatistics> EthtoolNetlinkClient::get_statistics(
    const std::string& /* iface */) {
  return {false, {}, ENOTSUP};
}

transport::Result<transport::FecStatistics_t> EthtoolNetlinkClient::get_fec_statistics(
    const std::string& /* iface */) {
  return {false, {}, ENOTSUP};
}

}  // namespace amd::nic::netlink
