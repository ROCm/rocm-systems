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

#include "smi_nic_subsystem.h"

#include <climits>
#include <unistd.h>

#include <filesystem>

#include "smi_sysfs.h"

namespace fs = std::filesystem;

std::pair<uint16_t, uint16_t> SmiNicSubsystem::read_pci_ids(const std::string& sysfs_bus_path) const
{
	uint16_t vendor_id = 0, device_id = 0;

	std::string vendor_path = sysfs_bus_path + "/vendor";
	std::string device_path = sysfs_bus_path + "/device";

	SmiSysfsReader::SysfsValue vendor_val, device_val;
	if (SmiSysfsReader::readLine(vendor_path, vendor_val) == SmiSysfsReader::SysfsStatus::Success &&
		SmiSysfsReader::readLine(device_path, device_val) == SmiSysfsReader::SysfsStatus::Success) {

		if (std::holds_alternative<int>(vendor_val)) {
			vendor_id = static_cast<uint16_t>(std::get<int>(vendor_val));
		} else if (std::holds_alternative<std::string>(vendor_val)) {
			vendor_id = static_cast<uint16_t>(std::stoul(std::get<std::string>(vendor_val), nullptr, 0));
		}

		if (std::holds_alternative<int>(device_val)) {
			device_id = static_cast<uint16_t>(std::get<int>(device_val));
		} else if (std::holds_alternative<std::string>(device_val)) {
			device_id = static_cast<uint16_t>(std::stoul(std::get<std::string>(device_val), nullptr, 0));
		}
	}

	return {vendor_id, device_id};
}

bool SmiNicSubsystem::resolve_bdf(const std::string& symlink, std::string& bdf) const
{
	char resolved_path[PATH_MAX];
	ssize_t len = readlink(symlink.c_str(), resolved_path, sizeof(resolved_path) - 1);

	if (len == -1) {
		return false;
	}
	resolved_path[len] = '\0';

	try {
		fs::path symlink_dir = fs::path(symlink).parent_path();
		fs::path target_path = symlink_dir / resolved_path;
		std::string full_path = fs::canonical(target_path);
		bdf = fs::path(full_path).filename();
		return true;
	} catch (const fs::filesystem_error&) {
		return false;
	}
}

// PENSANDO

NicVendor SmiNicSubsystemPensando::vendor() const
{
	return NicVendor::AMD;
}

bool SmiNicSubsystemPensando::driver_loaded(const std::string& bdf, DriverType driver_type) const
{
	std::error_code ec;
	std::string driver_dir;

	switch (driver_type) {
		case DriverType::IONIC:
			driver_dir = "/sys/bus/pci/drivers/ionic";
			break;
		case DriverType::IONIC_RDMA:
			driver_dir = "/sys/bus/auxiliary/drivers/ionic_rdma.rdma";
			break;
		default:
			return false;
	}

	if (!fs::exists(driver_dir, ec) || !fs::is_directory(driver_dir, ec)) {
		return false;
	}

	try {
		for (const auto& entry : fs::directory_iterator(driver_dir, ec)) {
			if (ec) {
				continue;
			}

			if (!fs::is_symlink(entry, ec)) {
				continue;
			}

			std::string symlink_target = fs::read_symlink(entry.path(), ec).string();
			if (ec) {
				continue;
			}

			if (driver_type == DriverType::IONIC) {
				if (entry.path().filename().string() == bdf) {
					return true;
				}
			}
			else if (driver_type == DriverType::IONIC_RDMA) {
				fs::path full_target_path = entry.path().parent_path() / symlink_target;
				std::string canonical_target = fs::canonical(full_target_path, ec).string();
				if (ec) {
					continue;
				}

				if (canonical_target.find("/" + bdf + "/") != std::string::npos) {
					return true;
				}
			}
		}
	} catch (const fs::filesystem_error&) {
		return false;
	}

	return false;
}

void SmiNicSubsystemPensando::discover(const std::string& pci_path, const std::string& net_path)
{
	nics_.clear();
	std::error_code ec;

	for (const auto& entry : fs::directory_iterator(pci_path, ec)) {
		if (ec) {
			continue;
		}

		std::string bdf = entry.path().filename().string();
		std::string sysfs_bus_path = entry.path().string();
		auto [vendor_id, device_id] = read_pci_ids(sysfs_bus_path);

		if (vendor_id == VENDOR_ID && device_id == DEVICE_ID) {
			auto nic = std::make_unique<SmiNicPensando>("", bdf, NicType::PCIBridge, "", sysfs_bus_path,
			          NicVendor::AMD, NicProduct::AINIC);

			discover_ports(*nic, bdf, pci_path, net_path);
			if (nic->nic_ports_num() > 0) {
				nics_.push_back(std::move(nic));
			}
		}
	}
}

const std::vector<std::unique_ptr<SmiNic>>& SmiNicSubsystemPensando::get_nics() const
{
	return nics_;
}

void SmiNicSubsystemPensando::discover_ports(SmiNic& nic, const std::string& bridge_bdf, 
					     const std::string& pci_path, const std::string& net_path)
{
	std::error_code ec;

	for (const auto& net_entry : fs::directory_iterator(net_path, ec)) {
		if (ec) {
			continue;
		}

		const std::string iface_name = net_entry.path().filename().string();
		std::string device_symlink = net_entry.path().string() + "/device";
		std::string sysfs_class_path = net_entry.path().string();

		if (fs::exists(device_symlink, ec) && fs::is_symlink(device_symlink, ec)) {
			std::string port_bdf;
			if (resolve_bdf(device_symlink, port_bdf)) {
				std::string port_sysfs_bus_path = pci_path + "/" + port_bdf;
				auto [port_vendor_id, port_device_id] = read_pci_ids(port_sysfs_bus_path);

				if (port_vendor_id == VENDOR_ID && port_device_id == PORT_ID) {
					if (downstream_port(port_bdf, bridge_bdf, pci_path)) {
						SmiNicPort port(iface_name, port_bdf, sysfs_class_path, port_sysfs_bus_path);
						port.discover_infiniband();
						port.collect_vendor_statistics();
						port.collect_standard_statistics();
						nic.add_nic_port(port);
					}
				}
			}
		}
	}
}

bool SmiNicSubsystemPensando::downstream_port(const std::string& port_bdf, const std::string& bridge_bdf,
					      const std::string& pci_path) const
{
	std::error_code ec;
	std::string port_path = pci_path + "/" + port_bdf;

	if (!fs::exists(port_path, ec) || !fs::is_symlink(port_path, ec)) {
		return false;
	}

	try {
		std::string port_canon_path = fs::canonical(port_path, ec).string();
		if (ec) {
			return false;
		}

		std::string bridge = "/" + bridge_bdf + "/";
		return port_canon_path.find(bridge) != std::string::npos;

	} catch (const fs::filesystem_error&) {
		return false;
	}
}

// BROADCOM

NicVendor SmiNicSubsystemBroadcom::vendor() const
{
	return NicVendor::Broadcom;
}

bool SmiNicSubsystemBroadcom::driver_loaded(const std::string& bdf, DriverType driver_type) const
{
	(void)bdf;
	(void)driver_type;
	return false;
}

void SmiNicSubsystemBroadcom::discover(const std::string& pci_path, const std::string& net_path)
{
	(void)pci_path;
	(void)net_path;
	nics_.clear();
	// TODO: broadcom - discovery
}

const std::vector<std::unique_ptr<SmiNic>>& SmiNicSubsystemBroadcom::get_nics() const
{
	return nics_;
}
