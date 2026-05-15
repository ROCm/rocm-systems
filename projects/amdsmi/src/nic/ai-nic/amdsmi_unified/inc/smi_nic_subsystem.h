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
#include <map>
#include <utility>
#include <memory>

#include "smi_nic.h"

enum class DriverType {
	IONIC,
	IONIC_RDMA,
};

class SmiNicSubsystem {
public:
	virtual ~SmiNicSubsystem() = default;

	virtual void discover(const std::string& pci_path, const std::string& net_path) = 0;
	virtual NicVendor vendor() const = 0;
	virtual bool driver_loaded(const std::string& bdf, DriverType driver_type) const = 0;
	virtual const std::vector<std::unique_ptr<SmiNic>>& get_nics() const = 0;
protected:
	std::pair<uint16_t, uint16_t> read_pci_ids(const std::string& sysfs_bus_path) const;
	bool resolve_bdf(const std::string& symlink, std::string& bdf) const;
};


class SmiNicSubsystemPensando : public SmiNicSubsystem {
public:
	SmiNicSubsystemPensando() = default;
	~SmiNicSubsystemPensando() override = default;

	void discover(const std::string& pci_path, const std::string& net_path) override;
	NicVendor vendor() const override;
	const std::vector<std::unique_ptr<SmiNic>>& get_nics() const override;
private:
	static constexpr uint16_t VENDOR_ID = 0x1dd8;
	static constexpr uint16_t DEVICE_ID = 0x0008;
	static constexpr uint16_t PORT_ID = 0x1002;

	bool driver_loaded(const std::string& bdf, DriverType driver_type) const override;
	bool downstream_port(const std::string& port_bdf, const std::string& bridge_bdf, const std::string& pci_path) const;
	void discover_ports(SmiNic& nic, const std::string& bridge_bdf, const std::string& pci_path, const std::string& net_path);

	std::vector<std::unique_ptr<SmiNic>> nics_;
};

// TODO: broadcom - add subsystem
class SmiNicSubsystemBroadcom : public SmiNicSubsystem {
public:
	SmiNicSubsystemBroadcom() = default;
	~SmiNicSubsystemBroadcom() override = default;

	bool driver_loaded(const std::string& bdf, DriverType driver_type) const override;
	void discover(const std::string& pci_path, const std::string& net_path) override;
	NicVendor vendor() const override;
	const std::vector<std::unique_ptr<SmiNic>>& get_nics() const override;
private:
	// TODO: broadcom
	std::vector<std::unique_ptr<SmiNic>> nics_;
};

#endif // __SMI_NIC_SUBSYSTEM_H__

