// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "smi_ethtool_ioctl.h"

#include <fcntl.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "smi_nic_log.h"

template int smi_ethtool_ioctl<ethtool_stats>(const std::string& device, ethtool_stats* data);
template int smi_ethtool_ioctl<ethtool_gstrings>(const std::string& device, ethtool_gstrings* data);
template int smi_ethtool_ioctl<ethtool_drvinfo>(const std::string& device, ethtool_drvinfo* data);
template int smi_ethtool_ioctl<ethtool_pauseparam>(const std::string& device,
                                                   ethtool_pauseparam* data);
template int smi_ethtool_ioctl<ethtool_fecparam>(const std::string& device, ethtool_fecparam* data);
template int smi_ethtool_ioctl<ethtool_link_settings>(const std::string& device,
                                                      ethtool_link_settings* data);
template int smi_ethtool_ioctl<ethtool_perm_addr>(const std::string& device,
                                                  ethtool_perm_addr* data);

namespace {

// ":xx" plus NUL
constexpr size_t kOctetTextLen = 4;
// "0x" plus 8 hex digits plus NUL
constexpr size_t kCmdTextLen = 16;

// ethtool fixed-size text fields are not guaranteed NUL-terminated.
template <size_t N>
std::string text_field(const char (&field)[N]) {
  return std::string(field, strnlen(field, N));
}

}  // namespace

std::string smi_ethtool_describe(const ethtool_stats& v) {
  return "n_stats=" + std::to_string(v.n_stats);
}

std::string smi_ethtool_describe(const ethtool_gstrings& v) {
  return "string_set=" + std::to_string(v.string_set) + " len=" + std::to_string(v.len);
}

std::string smi_ethtool_describe(const ethtool_drvinfo& v) {
  return "driver=" + text_field(v.driver) + " version=" + text_field(v.version) +
         " fw_version=" + text_field(v.fw_version) + " bus_info=" + text_field(v.bus_info);
}

std::string smi_ethtool_describe(const ethtool_pauseparam& v) {
  return "autoneg=" + std::to_string(v.autoneg) + " rx_pause=" + std::to_string(v.rx_pause) +
         " tx_pause=" + std::to_string(v.tx_pause);
}

std::string smi_ethtool_describe(const ethtool_fecparam& v) {
  return "active_fec=" + std::to_string(v.active_fec) + " fec=" + std::to_string(v.fec);
}

std::string smi_ethtool_describe(const ethtool_link_settings& v) {
  return "speed=" + std::to_string(v.speed) + " duplex=" + std::to_string(v.duplex) +
         " autoneg=" + std::to_string(v.autoneg) + " port=" + std::to_string(v.port);
}

std::string smi_ethtool_describe(const ethtool_perm_addr& v) {
  std::string mac;
  for (uint32_t i = 0; i < v.size; ++i) {
    char octet[kOctetTextLen];
    std::snprintf(octet, sizeof(octet), "%s%02x", (i == 0) ? "" : ":", v.data[i]);
    mac += octet;
  }
  return "size=" + std::to_string(v.size) + " addr=" + amd::smi::nic::log::mask_tail(mac);
}

template <typename T>
int smi_ethtool_ioctl(const std::string& device, T* data) {
  struct ifreq ifr{};

  int sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock < 0) {
    return -1;
  }

  device.copy(ifr.ifr_name, IFNAMSIZ - 1);
  ifr.ifr_data = reinterpret_cast<char*>(data);

  const int ret = ioctl(sock, SIOCETHTOOL, &ifr);
  const int saved_errno = errno;
  close(sock);

  if (amd::smi::nic::log::is_enabled()) {
    char cmd[kCmdTextLen];
    std::snprintf(cmd, sizeof(cmd), "0x%x", data->cmd);
    std::string msg = "ethtool ioctl " + device + " cmd=" + cmd + " -> ";
    msg += (ret == -1) ? ("FAIL " + std::string(std::strerror(saved_errno)))
                       : ("SUCCESS " + smi_ethtool_describe(*data));
    NIC_LOG_DEBUG(msg);
  }

  if (ret == -1) {
    // The callers read errno after this returns; logging must not change it.
    errno = saved_errno;
    return -1;
  }
  return 0;
}
