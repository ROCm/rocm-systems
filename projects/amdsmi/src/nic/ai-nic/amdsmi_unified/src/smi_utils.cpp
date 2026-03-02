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

#include "smi_utils.h"

#include <regex>
#include <filesystem>
#include <sstream>
#include <iomanip>

static const std::string SYSFS_PCI_BUS_PATH = "/sys/bus/pci/devices/";

namespace smi_utils {

bool is_valid_bdf(const std::string& bdf)
{
	static const std::regex bdf_pattern("^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\\.[0-9a-fA-F]$");
	return std::regex_match(bdf, bdf_pattern);
}

uint64_t parse_bdf(const std::string& bdf)
{
	if (!is_valid_bdf(bdf)) return 0;

	try {
		uint64_t domain = std::stoul(bdf.substr(0, 4), nullptr, 16);
		uint64_t bus = std::stoul(bdf.substr(5, 2), nullptr, 16);
		uint64_t device = std::stoul(bdf.substr(8, 2), nullptr, 16);
		uint64_t function = std::stoul(bdf.substr(11, 1), nullptr, 16);

		return (domain << 16) | (bus << 8) | (device << 3) | function;
	} catch (const std::exception&) {
		return 0;
	}
}

std::string format_bdf(uint64_t bdf)
{
	uint64_t function = bdf & 0x7;
	uint64_t device = (bdf >> 3) & 0x1F;
	uint64_t bus = (bdf >> 8) & 0xFF;
	uint64_t domain = (bdf >> 16) & 0xFFFF;

	std::ostringstream oss;
	oss << std::hex << std::setfill('0')
	    << std::setw(4) << domain << ":"
	    << std::setw(2) << bus << ":"
	    << std::setw(2) << device << "."
	    << std::setw(1) << function;
	return oss.str();
}

std::optional<int> get_numa_node_from_bdf(const std::string& bdf)
{
	if (!is_valid_bdf(bdf)) {
		return std::nullopt;
	}

	return get_sysfs_data<int>(SYSFS_PCI_BUS_PATH + bdf + "/numa_node");
}

std::optional<std::string> get_pcie_parent_bdf(const std::string& bdf)
{
	if (!is_valid_bdf(bdf)) {
		return std::nullopt;
	}

	std::string sysfsPath = SYSFS_PCI_BUS_PATH + bdf;
	std::error_code ec;

	// Read the symlink target - points to actual device path in /sys/devices
	// e.g., ../../../devices/pci0000:00/0000:00:01.0/0000:01:00.0
	auto linkTarget = std::filesystem::read_symlink(sysfsPath, ec);
	if (ec) return std::nullopt;

	// Parent BDF is the second-to-last path component
	std::string parentName = linkTarget.parent_path().filename().string();

	if (is_valid_bdf(parentName)) {
		return parentName;
	}

	return std::nullopt;
}

std::string nic_type_to_string(NicType type)
{
	switch (type) {
	case NicType::PCIBridge:
		return "PCI Bridge";
	case NicType::Ethernet:
		return "Ethernet";
	case NicType::InfiniBand:
		return "InfiniBand";
	default:
		return "Unknown";
	}
}

} // namespace smi_utils


