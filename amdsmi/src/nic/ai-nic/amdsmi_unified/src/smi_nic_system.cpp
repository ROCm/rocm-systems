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

#include "smi_nic_system.h"
#include "smi_nic_subsystem.h"

#include <cstdio>

#include <algorithm>
#include <set>
#include <regex>
#include <iomanip>
#include <sstream>
#include <string>

#include "smi_sysfs.h"

namespace fs = std::filesystem;

uint64_t parse_bdf(const std::string& bdf)
{
	if (bdf.length() != 12) {
		return 0;
	}

	if (bdf[4] != ':' || bdf[7] != ':' || bdf[10] != '.') {
		return 0;
	}

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

SmiNicSystem::SmiNicSystem() : net_path_("/sys/class/net"), pci_path_("/sys/bus/pci/devices")
{
	register_subsystem(std::make_unique<SmiNicSubsystemPensando>());
	// TODO: broadcom
	// register_subsystem(std::make_unique<SmiNicSubsystemBroadcom>());
}

void SmiNicSystem::register_subsystem(std::unique_ptr<SmiNicSubsystem> subsystem)
{
	subsystems_.push_back(std::move(subsystem));
}

bool SmiNicSystem::interface_exists(const std::string& iface)
{
	std::error_code ec;
	return fs::exists(fs::path(net_path_) / fs::path(iface).string(), ec);
}

bool SmiNicSystem::driver_loaded(const std::string& bdf, DriverType driver_type) const
{
	const SmiNic* nic = nullptr;

	for (const auto& entry : nics_) {
		if (entry->bdf() == bdf) {
			nic = entry;
			break;
		}

		for (const auto& port : entry->nic_ports()) {
			if (port.bdf() == bdf) {
				nic = entry;
				break;
			}
		}

		if (nic) {
			break;
		}
	}

	if (!nic) {
		return false;
	}

	for (const auto& subsystem : subsystems_) {
		if (subsystem->vendor() == nic->vendor()) {
			return subsystem->driver_loaded(bdf, driver_type);
		}
	}

	return false;
}

void SmiNicSystem::discover_nics()
{
	std::error_code ec;

	if (!fs::exists(pci_path_, ec) || !fs::is_directory(pci_path_, ec)) {
		return;
	}

	nics_.clear();
	for (auto& subsystem : subsystems_) {
		subsystem->discover(pci_path_, net_path_);
		const auto& subsys_nics = subsystem->get_nics();
		for (const auto& nic : subsys_nics) {
			nics_.push_back(nic.get());
		}
	}

	// Sort NICs by BDF
	std::sort(nics_.begin(), nics_.end(), [](const SmiNic* x, const SmiNic* y) {
		return parse_bdf(x->bdf()) < parse_bdf(y->bdf());
	});

	// Sort ports within each NIC by BDF for consistency
	for (auto* nic : nics_) {
		auto& ports = const_cast<std::vector<SmiNicPort>&>(nic->nic_ports());
		std::sort(ports.begin(), ports.end(), [](const SmiNicPort& x, const SmiNicPort& y) {
			return parse_bdf(x.bdf()) < parse_bdf(y.bdf());
		});
	}
}

const std::vector<const SmiNic*>& SmiNicSystem::get_nics() const
{
	return nics_;
}

std::vector<std::string> SmiNicSystem::list_bdfs()
{
	std::vector<std::string> bdfs;
	for (const auto* nic : nics_) {
		bdfs.push_back(nic->bdf());
	}
	return bdfs;
}

const SmiNic* SmiNicSystem::get_nic_by_interface(const std::string& iface) const
{
	for (const auto* nic : nics_) {
		if (nic->interface() == iface) {
			return nic;
		}
	}

	return nullptr;
}

const SmiNic* SmiNicSystem::get_nic_by_bdf(const std::string& bdf) const
{
	for (const auto* nic : nics_) {
		if (nic->bdf() == bdf) {
			return nic;
		}
	}

	return nullptr;
}

const SmiNic* SmiNicSystem::get_nic_by_bdf(uint64_t bdf) const
{
	uint64_t function_number = bdf & 0x7;
	uint64_t device_number = (bdf >> 3) & 0x1F;
	uint64_t bus_number = (bdf >> 8) & 0xFF;
	uint64_t domain_number = (bdf >> 16) & 0xFFFFFFFF;
	std::ostringstream oss;

	oss << std::hex << std::setfill('0')
	    << std::setw(4) << domain_number << ":"
	    << std::setw(2) << bus_number << ":"
	    << std::setw(2) << device_number << "."
	    << std::setw(1) << function_number;

	return get_nic_by_bdf(oss.str());
}
