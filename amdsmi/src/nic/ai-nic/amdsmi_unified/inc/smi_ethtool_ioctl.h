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

#ifndef __SMI_ETHTOOL_IOCTL_H__
#define __SMI_ETHTOOL_IOCTL_H__

#include <unistd.h>
#include <cstring>

#include <iostream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <string>

#include <linux/ethtool.h>
#include <net/if.h>

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

#endif // __SMI_ETHTOOL_IOCTL_H__
