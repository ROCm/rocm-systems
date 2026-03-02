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

#ifndef __SMI_NIC_SUBSYSTEM_H__
#define __SMI_NIC_SUBSYSTEM_H__

#include <cstdint>

#include <string>
#include <vector>
#include <utility>
#include <memory>

#include "smi_nic.h"

enum class DriverType {
	IONIC,
	IONIC_RDMA,
	BNXT_EN,
	BNXT_RE
};

class SmiNicSubsystem {
public:
	virtual ~SmiNicSubsystem() = default;

	virtual void discover(const std::string& pci_path, const std::string& net_path) = 0;
	virtual NicVendor vendor() const = 0;
	virtual bool driver_loaded(DriverType driver_type) const = 0;
	virtual bool driver_loaded(const std::string& bdf, DriverType driver_type) const = 0;
	virtual const std::vector<std::unique_ptr<SmiNic>>& get_nics() const = 0;
protected:
	virtual std::unique_ptr<SmiNic> create_nic(uint16_t device_id, const std::string& bdf, const std::string& sysfs_bus_path) const = 0;
	std::pair<uint16_t, uint16_t> read_pci_ids(const std::string& sysfs_bus_path) const;
	bool resolve_bdf(const std::string& symlink, std::string& bdf) const;
};

class SmiNicSubsystemPensando : public SmiNicSubsystem {
public:
	SmiNicSubsystemPensando() = default;
	~SmiNicSubsystemPensando() override = default;

	void discover(const std::string& pci_path, const std::string& net_path) override;
	NicVendor vendor() const override;
	bool driver_loaded(DriverType driver_type) const override;
	bool driver_loaded(const std::string& bdf, DriverType driver_type) const override;
	const std::vector<std::unique_ptr<SmiNic>>& get_nics() const override;

protected:
	std::unique_ptr<SmiNic> create_nic(uint16_t device_id, const std::string& bdf, const std::string& sysfs_bus_path) const override;

private:
	static constexpr uint16_t VENDOR_ID = 0x1dd8;
	static constexpr uint16_t PORT_ID = 0x1002;
	static constexpr uint16_t DEVICE_ID_POLLARA = 0x0008;

	bool downstream_port(const std::string& port_bdf, const std::string& bridge_bdf, const std::string& pci_path) const;
	void discover_ports(SmiNic& nic, const std::string& bridge_bdf, const std::string& pci_path, const std::string& net_path);

	std::vector<std::unique_ptr<SmiNic>> nics_;
};

class SmiNicSubsystemBroadcom : public SmiNicSubsystem {
public:
	SmiNicSubsystemBroadcom() = default;
	~SmiNicSubsystemBroadcom() override = default;

	void discover(const std::string& pci_path, const std::string& net_path) override;
	NicVendor vendor() const override;
	bool driver_loaded(DriverType driver_type) const override;
	bool driver_loaded(const std::string& bdf, DriverType driver_type) const override;
	const std::vector<std::unique_ptr<SmiNic>>& get_nics() const override;

protected:
	std::unique_ptr<SmiNic> create_nic(uint16_t device_id, const std::string& bdf, const std::string& sysfs_bus_path) const override;

private:
	static constexpr uint16_t VENDOR_ID = 0x14e4;
	static constexpr uint16_t PORT_ID = VENDOR_ID;
	static constexpr uint16_t SWITCH_VENDOR_ID = 0x1000;
	static constexpr uint16_t SWITCH_DEVICE_ID = 0x00b2;
	static constexpr uint16_t DEVICE_ID_THOR2 = 0x1760;

	void discover_ports(SmiNic& nic, const std::string& device_bdf, uint16_t device_id, const std::string& pci_path, const std::string& net_path);

	std::vector<std::unique_ptr<SmiNic>> nics_;
};

#endif // __SMI_NIC_SUBSYSTEM_H__

