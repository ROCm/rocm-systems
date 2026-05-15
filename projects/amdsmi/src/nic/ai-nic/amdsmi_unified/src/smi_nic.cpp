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

#include "smi_nic.h"

#include <cstring>

#include <filesystem>
#include <algorithm>
#include <fstream>
#include <string>
#include <memory>
#include <array>
#include <sstream>
#include <iomanip>
#include <unordered_map>

#include <linux/ethtool.h>
#include <linux/if_arp.h>
#include "smi_sysfs.h"
#include "smi_ethtool_ioctl.h"

static std::string nic_type_to_string(NicType type)
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

template <typename T>
static std::optional<T> get_sysfs_data(const std::string& path)
{
	SmiSysfsReader::SysfsValue val;
	if (SmiSysfsReader::readLine(path, val) == SmiSysfsReader::SysfsStatus::Success) {
		if constexpr (std::is_same_v<T, std::string>) {
			if (std::holds_alternative<std::string>(val)) {
				return std::get<std::string>(val);
			}
			if (std::holds_alternative<int>(val)) {
				return std::to_string(std::get<int>(val));
			}
		} else {
			if (std::holds_alternative<int>(val)) {
				return static_cast<T>(std::get<int>(val));
			}
			if (std::holds_alternative<std::string>(val)) {
				return static_cast<T>(std::stoul(std::get<std::string>(val), nullptr, 0));
			}
		}
	}

	return std::nullopt;
}

// **** SmiNicPort ****

SmiNicPort::SmiNicPort(const std::string& iface, const std::string& bdf, const std::string& sysfs_class_path, const std::string& sysfs_bus_path)
	: iface_(iface), bdf_(bdf), sysfs_class_path_(sysfs_class_path), sysfs_bus_path_(sysfs_bus_path)
{
	port_num_ = get_sysfs_data<uint32_t>(sysfs_class_path_ + "/dev_port");
	auto type_value = get_sysfs_data<int>(sysfs_class_path_ + "/type");

	if (type_value.has_value()) {
		if (type_value.value() == ARPHRD_ETHER) {
			type_ = NicType::Ethernet;
		} else if (type_value.value() == ARPHRD_INFINIBAND) {
			type_ = NicType::InfiniBand;
		} else {
			type_ = NicType::Unknown;
		}
	} else {
		type_ = NicType::Unknown;
	}
}

const std::string& SmiNicPort::interface() const
{
	return iface_;
}

const std::string& SmiNicPort::bdf() const
{
	return bdf_;
}

const std::string& SmiNicPort::sysfs_class_path() const
{
	return sysfs_class_path_;
}

const std::string& SmiNicPort::sysfs_bus_path() const
{
	return sysfs_bus_path_;
}

std::optional<std::string> SmiNicPort::mac_address() const
{
	return get_sysfs_data<std::string>(sysfs_class_path_ + "/address");
}

std::optional<uint32_t> SmiNicPort::port_num() const
{
	return port_num_;
}

std::optional<uint8_t> SmiNicPort::ifindex() const
{
	return get_sysfs_data<uint8_t>(sysfs_class_path_ + "/ifindex");
}

std::optional<uint8_t> SmiNicPort::carrier() const
{
	return get_sysfs_data<uint8_t>(sysfs_class_path_ + "/carrier");
}

std::optional<uint16_t> SmiNicPort::mtu() const
{
	return get_sysfs_data<uint16_t>(sysfs_class_path_ + "/mtu");
}

std::optional<std::string> SmiNicPort::link_state() const
{
	return get_sysfs_data<std::string>(sysfs_class_path_ + "/operstate");
}

std::optional<uint32_t> SmiNicPort::link_speed() const
{
	return get_sysfs_data<uint32_t>(sysfs_class_path_ + "/speed");
}

const std::string SmiNicPort::port_type() const
{
	return nic_type_to_string(type_);
}

std::string SmiNicPort::flavour() const
{
	return "N/A";
}

std::optional<uint32_t> SmiNicPort::active_fec() const
{
	struct ethtool_fecparam fec;
	fec.cmd = ETHTOOL_GFECPARAM;

	int ret = smi_ethtool_ioctl(iface_, &fec);
	if (ret == 0) {
		return fec.active_fec;
	}
	return std::nullopt;
}

std::optional<std::string> SmiNicPort::autoneg() const
{
	struct ethtool_link_settings link_settings;
	link_settings.cmd = ETHTOOL_GLINKSETTINGS;

	int ret = smi_ethtool_ioctl(iface_, &link_settings);
	if (ret == 0) {
		return link_settings.autoneg ? "on" : "off";
	}
	return std::nullopt;
}

std::optional<std::string> SmiNicPort::pause_autoneg() const
{
	struct ethtool_pauseparam pause;
	pause.cmd = ETHTOOL_GPAUSEPARAM;

	int ret = smi_ethtool_ioctl(iface_, &pause);
	if (ret == 0) {
		return pause.autoneg ? "on" : "off";
	}
	return std::nullopt;
}

std::optional<std::string> SmiNicPort::pause_rx() const
{
	struct ethtool_pauseparam pause;
	pause.cmd = ETHTOOL_GPAUSEPARAM;

	int ret = smi_ethtool_ioctl(iface_, &pause);
	if (ret == 0) {
		return pause.rx_pause ? "on" : "off";
	}
	return std::nullopt;
}

std::optional<std::string> SmiNicPort::pause_tx() const
{
	struct ethtool_pauseparam pause;
	pause.cmd = ETHTOOL_GPAUSEPARAM;

	int ret = smi_ethtool_ioctl(iface_, &pause);
	if (ret == 0) {
		return pause.tx_pause ? "on" : "off";
	}
	return std::nullopt;
}

void SmiNicPort::discover_infiniband()
{
	std::string infiniband_path = sysfs_bus_path_ + "/infiniband";
	if (!std::filesystem::exists(infiniband_path) || !std::filesystem::is_directory(infiniband_path)) {
		return;
	}

	for (const auto& entry : std::filesystem::directory_iterator(infiniband_path)) {
		if (entry.is_directory()) {

			std::string name = entry.path().filename().string();
			std::string sysfs_path = entry.path().string();
			SmiInfiniBand ib(name, sysfs_path);

			std::string ports_path = sysfs_path + "/ports";
			if (std::filesystem::exists(ports_path) && std::filesystem::is_directory(ports_path)) {
				for (const auto& port_entry : std::filesystem::directory_iterator(ports_path)) {
					if (port_entry.is_directory()) {
						std::string port_name = port_entry.path().filename().string();
						std::string port_sysfs_path = port_entry.path().string();
						SmiInfiniBandPort port(iface_, port_name, port_sysfs_path);
						port.collect_hw_counters();
						ib.add_port(port);
					}
				}
			}
			add_infiniband(ib);
		}
	}
}

void SmiNicPort::add_infiniband(const SmiInfiniBand& infiniband)
{
	infiniband_.push_back(infiniband);
}

const std::vector<SmiInfiniBand>& SmiNicPort::infiniband() const
{
	return infiniband_;
}

uint8_t SmiNicPort::infiniband_num() const
{
	return static_cast<uint8_t>(infiniband_.size());
}

void SmiNicPort::collect_vendor_statistics()
{
	int ret = 0;
	uint32_t stats_num = 0;

	auto drvinfo = std::make_unique<ethtool_drvinfo>();
	drvinfo->cmd = ETHTOOL_GDRVINFO;

	ret = smi_ethtool_ioctl(iface_, drvinfo.get());
	if (ret != 0 || !drvinfo) {
		return;
	}
	stats_num = drvinfo->n_stats;

	size_t strings_len = sizeof(ethtool_gstrings) + stats_num * ETH_GSTRING_LEN;
	std::unique_ptr<ethtool_gstrings, decltype(&free)> strings(
		static_cast<ethtool_gstrings*>(std::calloc(1, strings_len)), &free);
	strings->cmd = ETHTOOL_GSTRINGS;
	strings->string_set = ETH_SS_STATS;
	strings->len = static_cast<__u32>(stats_num);

	ret = smi_ethtool_ioctl(iface_, strings.get());
	if (ret != 0 || !strings) {
		return;
	}

	size_t stats_len = sizeof(ethtool_stats) + stats_num * sizeof(uint64_t);
	std::unique_ptr<ethtool_stats, decltype(&free)> stats(
		static_cast<ethtool_stats*>(std::calloc(1, stats_len)), &free);
	stats->cmd = ETHTOOL_GSTATS;
	stats->n_stats = static_cast<__u32>(stats_num);

	ret = smi_ethtool_ioctl(iface_, stats.get());
	if (ret != 0 || !stats) {
		return;
	}

	add_vendor_statistic(strings.get(), stats.get());
}

void SmiNicPort::add_vendor_statistic(struct ethtool_gstrings *strings, struct ethtool_stats *stats)
{
	if (!strings || !stats) {
		return;
	}
	for (unsigned int i = 0; i < stats->n_stats; ++i) {
		std::string key(reinterpret_cast<char*>(&strings->data[i * ETH_GSTRING_LEN]), ETH_GSTRING_LEN);
		key.erase(std::find(key.begin(), key.end(), '\0'), key.end());
		if (vendor_stat_allowed(key)) {
			uint64_t value = stats->data[i];
			vendor_stats_map_[key] = value;
		}
	}
}

const std::map<std::string, uint64_t>& SmiNicPort::get_vendor_stats_map() const
{
	return vendor_stats_map_;
}

void SmiNicPort::collect_standard_statistics()
{
	std::string stats_path = sysfs_class_path_ + "/statistics";

	if (!std::filesystem::exists(stats_path) || !std::filesystem::is_directory(stats_path)) {
		return;
	}

	for (const auto& entry : std::filesystem::directory_iterator(stats_path)) {
		if (entry.is_regular_file()) {
			std::string stat_name = entry.path().filename().string();
			auto stat_value = get_sysfs_data<uint64_t>(entry.path().string());
			if (stat_value.has_value()) {
				standard_stats_map_[stat_name] = stat_value.value();
			}
		}
	}
}

const std::map<std::string, uint64_t>& SmiNicPort::get_standard_stats_map() const
{
	return standard_stats_map_;
}

std::optional<std::string> SmiNicPort::read_vpd_content() const
{
	auto vpd = get_sysfs_data<std::string>(sysfs_bus_path_ + "/vpd");
	if (!vpd) {
		return std::nullopt;
	}

	std::string content = vpd.value();
	content.erase(std::remove_if(content.begin(), content.end(),
		[](char c) { return !(std::isprint(static_cast<unsigned char>(c)) || c == '\n'); }), content.end());

	return content;
}

std::string SmiNicPort::map_vendor_stat_to_string(SmiVendorStat stat) const
{
	static const std::unordered_map<SmiVendorStat, std::string> stat_map = {
		{SmiVendorStat::TX_PACKETS, "tx_packets"},
		{SmiVendorStat::RX_PACKETS, "rx_packets"},
		{SmiVendorStat::TX_BYTES, "tx_bytes"},
		{SmiVendorStat::RX_BYTES, "rx_bytes"},
		{SmiVendorStat::TX_CSUM_NONE, "tx_csum_none"},
		{SmiVendorStat::RX_CSUM_NONE, "rx_csum_none"},
		{SmiVendorStat::TX_CSUM, "tx_csum"},
		{SmiVendorStat::TX_TSO, "tx_tso"},
		{SmiVendorStat::TX_TSO_BYTES, "tx_tso_bytes"}
	};

	auto it = stat_map.find(stat);
	return (it != stat_map.end()) ? it->second : "";
}

bool SmiNicPort::vendor_stat_allowed(const std::string& stat_name) const
{
	for (int i = static_cast<int>(SmiVendorStat::TX_PACKETS);
	     i <= static_cast<int>(SmiVendorStat::TX_TSO_BYTES); i++) {
		SmiVendorStat stat = static_cast<SmiVendorStat>(i);
		if (map_vendor_stat_to_string(stat) == stat_name) {
			return true;
		}
	}
	return false;
}

// **** SmiInfiniBandPort ****

SmiInfiniBandPort::SmiInfiniBandPort(std::string& netdev, std::string& name, const std::string& sysfs_path)
	: netdev_(netdev), name_(name), sysfs_path_(sysfs_path)
{
}

const std::string& SmiInfiniBandPort::name() const
{
	return name_;
}

const std::string& SmiInfiniBandPort::netdev() const
{
	return netdev_;
}

std::optional<uint8_t> SmiInfiniBandPort::port_num() const
{
	try {
		return static_cast<uint8_t>(std::stoul(name_));
	} catch (const std::exception&) {
		return std::nullopt;
	}
}

std::optional<std::string> SmiInfiniBandPort::state() const
{
	auto raw_state = get_sysfs_data<std::string>(sysfs_path_ + "/state");
	if (!raw_state.has_value()) {
		return std::nullopt;
	}

	const std::string& state = raw_state.value();
	auto pos = state.find(": ");

	if (pos != std::string::npos) {
		return state.substr(pos + 2);
	}

	return state;
}

std::optional<uint16_t> SmiInfiniBandPort::max_mtu() const
{
	return get_sysfs_data<uint16_t>(sysfs_path_ + "/max_mtu");
}

std::optional<uint16_t> SmiInfiniBandPort::active_mtu() const
{
	return get_sysfs_data<uint16_t>(sysfs_path_ + "/active_mtu");
}

void SmiInfiniBandPort::collect_hw_counters()
{
	std::string hw_counters_path = sysfs_path_ + "/hw_counters";

	if (!std::filesystem::exists(hw_counters_path) || !std::filesystem::is_directory(hw_counters_path)) {
		return;
	}

	for (const auto& entry : std::filesystem::directory_iterator(hw_counters_path)) {
		if (entry.is_regular_file()) {
			std::string counter_name = entry.path().filename().string();
			auto counter_value = get_sysfs_data<uint64_t>(entry.path().string());
			if (counter_value.has_value()) {
				hw_counters_map_[counter_name] = counter_value.value();
			}
		}
	}
}

const std::map<std::string, uint64_t>& SmiInfiniBandPort::get_hw_counters_map() const
{
	return hw_counters_map_;
}

// **** SmiInfiniBand ****

SmiInfiniBand::SmiInfiniBand(std::string& name, const std::string& sysfs_path)
	: name_(name), sysfs_path_(sysfs_path)
{
}

std::string SmiInfiniBand::rdma_dev() const
{
	return name_;
}

std::optional<std::string> SmiInfiniBand::node_guid() const
{
	return get_sysfs_data<std::string>(sysfs_path_ + "/node_guid");
}

std::optional<std::string> SmiInfiniBand::node_type() const
{
	auto raw_node_type = get_sysfs_data<std::string>(sysfs_path_ + "/node_type");
	if (!raw_node_type.has_value()) {
		return std::nullopt;
	}

	const std::string& node_type = raw_node_type.value();
	auto pos = node_type.find(": ");

	if (pos != std::string::npos) {
		return node_type.substr(pos + 2);
	}

	return node_type;
}

std::optional<std::string> SmiInfiniBand::sys_image_guid() const
{
	return get_sysfs_data<std::string>(sysfs_path_ + "/sys_image_guid");
}

std::optional<std::string> SmiInfiniBand::fw_ver() const
{
	return get_sysfs_data<std::string>(sysfs_path_ + "/fw_ver");
}

void SmiInfiniBand::add_port(const SmiInfiniBandPort& port)
{
	ports_.push_back(port);
}

const std::vector<SmiInfiniBandPort>& SmiInfiniBand::ports() const
{
	return ports_;
}

uint8_t SmiInfiniBand::ports_num() const
{
	return static_cast<uint8_t>(ports_.size());
}

// **** SmiNic ****

SmiNic::SmiNic(const std::string& iface, const std::string& bdf, NicType type,
	       const std::string& sysfs_class_path, const std::string& sysfs_bus_path,
	       NicVendor vendor, NicProduct product)
	: iface_(iface), bdf_(bdf), type_(type), vendor_(vendor), product_(product),
	  sysfs_class_path_(sysfs_class_path), sysfs_bus_path_(sysfs_bus_path)
{
}

const std::string& SmiNic::interface() const
{
	return iface_;
}

const std::string& SmiNic::bdf() const
{
	return bdf_;
}

NicType SmiNic::type() const
{
	return type_;
}

NicVendor SmiNic::vendor() const
{
	return vendor_;
}

NicProduct SmiNic::product() const
{
	return product_;
}

const std::string SmiNic::port_type() const
{
	return nic_type_to_string(type_);
}

const std::string& SmiNic::sysfs_class_path() const
{
	return sysfs_class_path_;
}

const std::string& SmiNic::sysfs_bus_path() const
{
	return sysfs_bus_path_;
}

void SmiNic::add_nic_port(const SmiNicPort& port)
{
	ports_.push_back(port);
}

const std::vector<SmiNicPort>& SmiNic::nic_ports() const
{
	return ports_;
}

uint8_t SmiNic::nic_ports_num() const
{
	return static_cast<uint8_t>(ports_.size());
}

std::optional<uint16_t> SmiNic::vendor_id() const
{
	return get_sysfs_data<uint16_t>(sysfs_bus_path_ + "/vendor");
}

std::optional<uint16_t> SmiNic::subvendor_id() const
{
	return get_sysfs_data<uint16_t>(sysfs_bus_path_ + "/subsystem_vendor");
}

std::optional<uint16_t> SmiNic::device_id() const
{
	return get_sysfs_data<uint16_t>(sysfs_bus_path_ + "/device");
}

std::optional<uint16_t> SmiNic::subsystem_id() const
{
	return get_sysfs_data<uint16_t>(sysfs_bus_path_ + "/subsystem_device");
}

std::optional<uint8_t> SmiNic::revision() const
{
	return get_sysfs_data<uint8_t>(sysfs_bus_path_ + "/revision");
}

std::optional<std::string> SmiNic::perm_address() const
{
	if (ports_.empty()) {
		return std::nullopt;
	}

	const std::string& port_iface = ports_[0].interface();
	struct ethtool_perm_addr permaddr;
	permaddr.cmd = ETHTOOL_GPERMADDR;
	permaddr.size = 6;

	int ret = smi_ethtool_ioctl(port_iface, &permaddr);
	if (ret != 0) {
		return std::nullopt;
	}

	if (permaddr.size == 6) {
		std::stringstream ss;
		ss << std::hex << std::setfill('0');
		for (int i = 0; i < 6; i++) {
			if (i > 0) ss << ":";
			ss << std::setw(2) << static_cast<unsigned int>(permaddr.data[i]);
		}
		return ss.str();
	}

	return std::nullopt;
}

std::optional<uint32_t> SmiNic::pcie_class() const
{
	return get_sysfs_data<uint32_t>(sysfs_bus_path_ + "/class");
}

std::optional<uint8_t> SmiNic::max_pcie_width() const {
	return get_sysfs_data<uint8_t>(sysfs_bus_path_ + "/max_link_width");
}

std::optional<uint32_t> SmiNic::max_pcie_speed() const {
	return get_sysfs_data<uint32_t>(sysfs_bus_path_ + "/max_link_speed");
}

std::optional<uint8_t> SmiNic::numa_node() const {
	return get_sysfs_data<uint8_t>(sysfs_bus_path_ + "/numa_node");
}

std::optional<std::string> SmiNic::numa_affinity(uint8_t node) const
{
	std::string path = "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist";
	return get_sysfs_data<std::string>(path);
}

std::optional<std::string> SmiNic::product_name() const
{
	return std::nullopt;
}

std::optional<std::string> SmiNic::part_number() const
{
	return std::nullopt;
}

std::optional<std::string> SmiNic::serial_number() const
{
	return std::nullopt;
}

std::optional<std::string> SmiNic::vendor_name() const
{
	return std::nullopt;
}

// **** SmiNicPensando ****

SmiNicPensando::SmiNicPensando(const std::string& iface, const std::string& bdf, NicType type,
			       const std::string& sysfs_class_path, const std::string& sysfs_bus_path,
			       NicVendor vendor, NicProduct product)
	: SmiNic(iface, bdf, type, sysfs_class_path, sysfs_bus_path, vendor, product)
{
}

std::optional<std::string> SmiNicPensando::vendor_name() const
{
	return std::string("AMD Pensando Systems, Inc.");
}

std::optional<std::string> SmiNicPensando::product_name() const
{
	if (ports_.empty()) {
		return std::nullopt;
	}

	auto vpd = ports_[0].read_vpd_content();
	if (!vpd) {
		return std::nullopt;
	}

	const std::string& content = vpd.value();
	size_t pn_pos = content.find("PN");

	if (pn_pos != std::string::npos) {
		std::string product_name = content.substr(0, pn_pos);
		product_name.erase(product_name.find_last_not_of(" \n\r\t") + 1);
		product_name.erase(0, product_name.find_first_not_of(" \n\r\t"));
		return product_name;
	}

	return std::nullopt;
}

std::optional<std::string> SmiNicPensando::part_number() const
{
	if (ports_.empty()) {
		return std::nullopt;
	}

	auto vpd = ports_[0].read_vpd_content();
	if (!vpd) {
		return std::nullopt;
	}

	const std::string& content = vpd.value();
	size_t pn_pos = content.find("PN");
	size_t sn_pos = content.find("SN", pn_pos);

	if (pn_pos != std::string::npos && sn_pos != std::string::npos) {
		std::string part_number = content.substr(pn_pos + 2, sn_pos - (pn_pos + 2));
		part_number.erase(part_number.find_last_not_of(" \n\r\t") + 1);
		part_number.erase(0, part_number.find_first_not_of(" \n\r\t"));
		return part_number;
	}

	return std::nullopt;
}

std::optional<std::string> SmiNicPensando::serial_number() const
{
	if (ports_.empty()) {
		return std::nullopt;
	}

	auto vpd = ports_[0].read_vpd_content();
	if (!vpd) {
		return std::nullopt;
	}

	const std::string& content = vpd.value();
	size_t sn_pos = content.find("SN");
	size_t mdt_pos = content.find("MDT", sn_pos);

	if (sn_pos != std::string::npos && mdt_pos != std::string::npos) {
		std::string serial_number = content.substr(sn_pos + 2, mdt_pos - (sn_pos + 2));
		serial_number.erase(serial_number.find_last_not_of(" \n\r\t") + 1);
		serial_number.erase(0, serial_number.find_first_not_of(" \n\r\t"));
		return serial_number;
	}

	return std::nullopt;
}

// **** SmiNicBroadcom ****

SmiNicBroadcom::SmiNicBroadcom(const std::string& iface, const std::string& bdf, NicType type,
			       const std::string& sysfs_class_path, const std::string& sysfs_bus_path,
			       NicVendor vendor, NicProduct product)
	: SmiNic(iface, bdf, type, sysfs_class_path, sysfs_bus_path, vendor, product)
{
}

std::optional<std::string> SmiNicBroadcom::vendor_name() const
{
	// TODO: broadcom - get vendor name
	return std::string("Broadcom Inc.");
}

std::optional<std::string> SmiNicBroadcom::product_name() const
{
	// TODO: broadcom - get product name
	return std::nullopt;
}

std::optional<std::string> SmiNicBroadcom::part_number() const
{
	// TODO: broadcom - get part number
	return std::nullopt;
}

std::optional<std::string> SmiNicBroadcom::serial_number() const
{
	// TODO: broadcom - get serial number
	return std::nullopt;
}
