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

#include "smi_ethtool_ioctl.h"

#include <cstring>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/ethtool.h>
#include <net/if.h>
#include <sys/socket.h>
#include <linux/sockios.h>

template int smi_ethtool_ioctl<ethtool_stats>(const std::string& device, ethtool_stats* data);
template int smi_ethtool_ioctl<ethtool_gstrings>(const std::string& device, ethtool_gstrings* data);
template int smi_ethtool_ioctl<ethtool_drvinfo>(const std::string& device, ethtool_drvinfo* data);
template int smi_ethtool_ioctl<ethtool_pauseparam>(const std::string& device, ethtool_pauseparam* data);
template int smi_ethtool_ioctl<ethtool_fecparam>(const std::string& device, ethtool_fecparam* data);
template int smi_ethtool_ioctl<ethtool_link_settings>(const std::string& device, ethtool_link_settings* data);
template int smi_ethtool_ioctl<ethtool_perm_addr>(const std::string& device, ethtool_perm_addr* data);

template <typename T>
int smi_ethtool_ioctl(const std::string& device, T* data)
{
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
