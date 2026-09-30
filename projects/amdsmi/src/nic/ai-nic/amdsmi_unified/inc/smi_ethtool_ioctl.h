// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef __SMI_ETHTOOL_IOCTL_H__
#define __SMI_ETHTOOL_IOCTL_H__

#include <fcntl.h>
#include <linux/ethtool.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <string>

/**
 * @brief Generic template function to perform ethtool ioctl on network devices.
 *
 * @tparam T The ethtool structure type (e.g., ethtool_stats, ethtool_drvinfo, etc.)
 * @param device Network device name (e.g., "eth0")
 * @param data Pointer to ethtool data structure to be populated or used for the ioctl
 *
 * @return 0 on success, -1 on failure
 *
 * @note The caller must properly initialize the data structure's cmd field before calling.
 * @note Supported for various ethtool structures including ethtool_stats, ethtool_gstrings,
 *       ethtool_drvinfo, ethtool_pauseparam, ethtool_fecparam, ethtool_link_settings,...
 */
template <typename T>
int smi_ethtool_ioctl(const std::string& device, T* data);

#endif  // __SMI_ETHTOOL_IOCTL_H__
