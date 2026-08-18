/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AMDSMI_UNIFIED_VENDORS_PENSANDO_SUBSYSTEM_H_
#define AMDSMI_UNIFIED_VENDORS_PENSANDO_SUBSYSTEM_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "smi_nic.h"
#include "smi_nic_subsystem.h"

class SmiNicSubsystemPensando : public SmiNicSubsystem {
 public:
  // driver_sysfs_root prefixes the driver sysfs dirs; "" means the real /sys.
  // Non-empty is used only by tests to redirect lookups to a tmpdir tree.
  explicit SmiNicSubsystemPensando(std::string driver_sysfs_root = "")
      : driver_sysfs_root_(std::move(driver_sysfs_root)) {}
  ~SmiNicSubsystemPensando() override = default;

  void discover(const std::string& pci_path, const std::string& net_path,
                std::shared_ptr<amd::smi::nic::transport::NicTransport> transport) override;
  NicVendor vendor() const override;
  const std::vector<std::unique_ptr<SmiNic>>& get_nics() const override;

 private:
  static constexpr uint16_t VENDOR_ID = 0x1dd8;
  static constexpr uint16_t DEVICE_ID = 0x0008;
  static constexpr uint16_t PORT_ID = 0x1002;

  bool driver_loaded(const std::string& bdf, DriverType driver_type) const override;
  bool downstream_port(const std::string& port_bdf, const std::string& bridge_bdf,
                       const std::string& pci_path) const;
  void discover_ports(SmiNic& nic, const std::string& bridge_bdf, const std::string& pci_path,
                      const std::string& net_path,
                      const std::shared_ptr<amd::smi::nic::transport::NicTransport>& transport);

  std::string driver_sysfs_root_;
  std::vector<std::unique_ptr<SmiNic>> nics_;
};

#endif  // AMDSMI_UNIFIED_VENDORS_PENSANDO_SUBSYSTEM_H_
