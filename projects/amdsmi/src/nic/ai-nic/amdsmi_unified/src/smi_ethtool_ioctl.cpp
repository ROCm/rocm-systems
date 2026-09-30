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

#include <cstring>

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

template <typename T>
int smi_ethtool_ioctl(const std::string& device, T* data) {
  struct ifreq ifr{};

  int sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock < 0) {
    return -1;
  }

  device.copy(ifr.ifr_name, IFNAMSIZ - 1);
  ifr.ifr_data = reinterpret_cast<char*>(data);

  if (ioctl(sock, SIOCETHTOOL, &ifr) == -1) {
    close(sock);
    return -1;
  }

  close(sock);
  return 0;
}
