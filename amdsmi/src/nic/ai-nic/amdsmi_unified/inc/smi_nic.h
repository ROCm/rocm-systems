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

#ifndef __SMI_NIC__H__
#define __SMI_NIC__H__

#include <string>
#include <optional>
#include <cstdint>
#include <vector>
#include <map>

#include "smi_ethtool_ioctl.h"

enum class NicType {
	Unknown,
	PCIBridge,
	Ethernet,
	InfiniBand,
};

enum class NicVendor {
	Unknown,
	AMD,
	Broadcom
};

// TODO: broadcom - update enum with the right products
enum class NicProduct {
	Unknown,
	AINIC,  // AMD Pensando AINIC
	Thor,   // Broadcom Thor
};

class SmiInfiniBandPort {
public:
	SmiInfiniBandPort(std::string& netdev, std::string& name, const std::string& sysfs_path_);

	const std::string& netdev() const;
	const std::string& name() const;
	std::optional<uint8_t> port_num() const;
	std::optional<std::string> state() const;
	std::optional<uint16_t> max_mtu() const;
	std::optional<uint16_t> active_mtu() const;
	void collect_hw_counters();
	const std::map<std::string, uint64_t>& get_hw_counters_map() const;

private:
	std::string netdev_;
	std::string name_;
	std::string sysfs_path_;
	std::map<std::string, uint64_t> hw_counters_map_;
};

class SmiInfiniBand {
public:
	SmiInfiniBand(std::string& name, const std::string& sysfs_path);

	std::string rdma_dev() const;
	std::optional<std::string> node_guid() const;
	std::optional<std::string> node_type() const;
	std::optional<std::string> sys_image_guid() const;
	std::optional<std::string> fw_ver() const;

	void add_port(const SmiInfiniBandPort& port);
	const std::vector<SmiInfiniBandPort>& ports() const;
	uint8_t ports_num() const;

private:
	std::string name_;
	std::string sysfs_path_;
	NicType type_ = NicType::InfiniBand;
	std::vector<SmiInfiniBandPort> ports_;
};

class SmiNicPort {
public:
	SmiNicPort(const std::string& iface, const std::string& bdf, const std::string& sysfs_class_path, const std::string& sysfs_bus_path);

	const std::string& interface() const;
	const std::string& bdf() const;
	const std::string& sysfs_class_path() const;
	const std::string& sysfs_bus_path() const;

	std::optional<std::string> mac_address() const;
	std::optional<uint32_t> port_num() const;
	std::optional<uint8_t> ifindex() const;
	std::optional<uint8_t> carrier() const;
	std::optional<uint16_t> mtu() const;
	std::optional<std::string> link_state() const;
	std::optional<uint32_t> link_speed() const;

	const std::string port_type() const;
	std::string flavour() const;

	std::optional<uint32_t> active_fec() const;
	std::optional<std::string> autoneg() const;
	std::optional<std::string> pause_autoneg() const;
	std::optional<std::string> pause_rx() const;
	std::optional<std::string> pause_tx() const;

	void discover_infiniband();
	void add_infiniband(const SmiInfiniBand& infiniband);
	const std::vector<SmiInfiniBand>& infiniband() const;
	uint8_t infiniband_num() const;
	void collect_vendor_statistics();
	void add_vendor_statistic(struct ethtool_gstrings *strings, struct ethtool_stats *stats);
	const std::map<std::string, uint64_t>& get_vendor_stats_map() const;
	void collect_standard_statistics();
	const std::map<std::string, uint64_t>& get_standard_stats_map() const;
	std::optional<std::string> read_vpd_content() const;

private:
	enum class SmiVendorStat {
		TX_PACKETS,
		RX_PACKETS,
		TX_BYTES,
		RX_BYTES,
		TX_CSUM_NONE,
		RX_CSUM_NONE,
		TX_CSUM,
		TX_TSO,
		TX_TSO_BYTES
	};

	std::string map_vendor_stat_to_string(SmiVendorStat stat) const;
	bool vendor_stat_allowed(const std::string& stat_name) const;

	std::string iface_;
	std::string bdf_;
	NicType type_;
	std::string sysfs_class_path_;
	std::string sysfs_bus_path_;
	std::optional<uint32_t> port_num_;
	std::vector<SmiInfiniBand> infiniband_;
	std::map<std::string, uint64_t> vendor_stats_map_;
	std::map<std::string, uint64_t> standard_stats_map_;
};

class SmiNic {
public:
	SmiNic(const std::string& iface, const std::string& bdf, NicType type = NicType::Unknown,
	       const std::string& sysfs_class_path = "", const std::string& sysfs_bus_path = "",
	       NicVendor vendor = NicVendor::Unknown, NicProduct product = NicProduct::Unknown);
	virtual ~SmiNic() = default;

	const std::string& interface() const;
	const std::string& bdf() const;
	NicType type() const;
	NicVendor vendor() const;
	NicProduct product() const;
	const std::string port_type() const;
	const std::string& sysfs_class_path() const;
	const std::string& sysfs_bus_path() const;

	void add_nic_port(const SmiNicPort& port);
	const std::vector<SmiNicPort>& nic_ports() const;
	uint8_t nic_ports_num() const;

	std::optional<uint16_t> vendor_id() const;
	std::optional<uint16_t> subvendor_id() const;
	std::optional<uint16_t> device_id() const;
	std::optional<uint16_t> subsystem_id() const;
	std::optional<uint8_t> revision() const;
	std::optional<std::string> perm_address() const;
	std::optional<uint32_t> pcie_class() const;
	std::optional<uint8_t> max_pcie_width() const;
	std::optional<uint32_t> max_pcie_speed() const;
	std::optional<uint8_t> numa_node() const;
	std::optional<std::string> numa_affinity(uint8_t node) const;
	// Vendor specific
	virtual std::optional<std::string> product_name() const;
	virtual std::optional<std::string> vendor_name() const;
	virtual std::optional<std::string> part_number() const;
	virtual std::optional<std::string> serial_number() const;

protected:
	std::string iface_;
	std::string bdf_;
	NicType type_;
	NicVendor vendor_;
	NicProduct product_;
	std::string sysfs_class_path_;
	std::string sysfs_bus_path_;
	std::vector<SmiNicPort> ports_;
};

class SmiNicPensando : public SmiNic {
public:
	SmiNicPensando(const std::string& iface, const std::string& bdf, NicType type = NicType::Unknown,
		       const std::string& sysfs_class_path = "", const std::string& sysfs_bus_path = "",
		       NicVendor vendor = NicVendor::AMD, NicProduct product = NicProduct::AINIC);

	std::optional<std::string> vendor_name() const override;
	std::optional<std::string> product_name() const override;
	std::optional<std::string> part_number() const override;
	std::optional<std::string> serial_number() const override;
};

class SmiNicBroadcom : public SmiNic {
public:
	SmiNicBroadcom(const std::string& iface, const std::string& bdf, NicType type = NicType::Unknown,
		       const std::string& sysfs_class_path = "", const std::string& sysfs_bus_path = "",
		       NicVendor vendor = NicVendor::Broadcom, NicProduct product = NicProduct::Thor);

	std::optional<std::string> vendor_name() const override;
	std::optional<std::string> product_name() const override;
	std::optional<std::string> part_number() const override;
	std::optional<std::string> serial_number() const override;
};

#endif // __SMI_NIC_H__
