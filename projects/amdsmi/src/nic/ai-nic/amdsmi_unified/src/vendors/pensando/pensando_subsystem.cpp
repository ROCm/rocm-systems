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

#include "pensando_subsystem.h"

#include <filesystem>

namespace fs = std::filesystem;

NicVendor SmiNicSubsystemPensando::vendor() const { return NicVendor::AMD; }

bool SmiNicSubsystemPensando::driver_loaded(const std::string& bdf, DriverType driver_type) const {
  switch (driver_type) {
    case DriverType::Main:
      return driver_binds_bdf(driver_sysfs_root_ + "/sys/bus/pci/drivers/ionic", bdf,
                              /*match_canonical=*/false);
    case DriverType::Rdma:
      return driver_binds_bdf(driver_sysfs_root_ + "/sys/bus/auxiliary/drivers/ionic_rdma.rdma",
                              bdf,
                              /*match_canonical=*/true);
    default:
      return false;
  }
}

void SmiNicSubsystemPensando::discover(
    const std::string& pci_path, const std::string& net_path,
    std::shared_ptr<amd::smi::nic::transport::NicTransport> transport) {
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

      discover_ports(*nic, bdf, pci_path, net_path, transport);
      // A fwctl-only POLLARA (pds_core, no host netdev) exposes zero ports but is
      // still a manageable NIC: identity/version come from PCI sysfs + devlink.
      // Keep it regardless of port count; ports_num()==0 is what marks it
      // fwctl-only for the accessors and CLI downstream.
      nics_.push_back(std::move(nic));
    }
  }
}

const std::vector<std::unique_ptr<SmiNic>>& SmiNicSubsystemPensando::get_nics() const {
  return nics_;
}

void SmiNicSubsystemPensando::discover_ports(
    SmiNic& nic, const std::string& bridge_bdf, const std::string& pci_path,
    const std::string& net_path,
    const std::shared_ptr<amd::smi::nic::transport::NicTransport>& transport) {
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
            SmiNicPort port(iface_name, port_bdf, sysfs_class_path, port_sysfs_bus_path, transport);
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

bool SmiNicSubsystemPensando::downstream_port(const std::string& port_bdf,
                                              const std::string& bridge_bdf,
                                              const std::string& pci_path) const {
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
