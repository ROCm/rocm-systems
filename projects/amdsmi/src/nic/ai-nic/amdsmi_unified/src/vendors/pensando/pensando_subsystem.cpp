// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "pensando_subsystem.h"

#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

NicVendor SmiNicSubsystemPensando::vendor() const { return NicVendor::AMD; }

bool SmiNicSubsystemPensando::is_bridge_device(uint16_t device_id) {
  return std::find(BRIDGE_DEVICE_IDS.begin(), BRIDGE_DEVICE_IDS.end(), device_id) !=
         BRIDGE_DEVICE_IDS.end();
}

bool SmiNicSubsystemPensando::is_driver_loaded(const std::string& bdf,
                                               DriverType driver_type) const {
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

    if ((vendor_id == VENDOR_ID) && is_bridge_device(device_id)) {
      auto nic = std::make_unique<SmiNicPensando>("", bdf, NicType::PCIBridge, "", sysfs_bus_path,
                                                  NicVendor::AMD, NicProduct::AINIC);

      discover_ports(*nic, bdf, pci_path, net_path, transport);
      discover_mgmt_function(*nic, bdf, pci_path);
      // Keep a card whose port walk found nothing: ports_num()==0 is what marks
      // it fwctl-only downstream, and identity still reads from PCI sysfs and
      // devlink. No card observed so far enumerates this way, so the branch is
      // a deliberate allowance rather than a path any hardware has exercised.
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
          if (is_downstream_port(port_bdf, bridge_bdf, pci_path)) {
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

// pds_core exposes no netdev, so the /sys/class/net walk that finds the ports
// cannot see it; it is found by PCI id instead, with the same ancestry test
// keeping one card from adopting a neighbour's.
void SmiNicSubsystemPensando::discover_mgmt_function(SmiNic& nic, const std::string& bridge_bdf,
                                                     const std::string& pci_path) const {
  std::error_code ec;

  for (const auto& entry : fs::directory_iterator(pci_path, ec)) {
    if (ec) {
      continue;
    }

    const std::string bdf = entry.path().filename().string();
    auto [vendor_id, device_id] = read_pci_ids(entry.path().string());

    if ((vendor_id == VENDOR_ID) && (device_id == MGMT_ID) &&
        is_downstream_port(bdf, bridge_bdf, pci_path)) {
      nic.set_mgmt_bdf(bdf);
      return;
    }
  }
}

bool SmiNicSubsystemPensando::is_downstream_port(const std::string& port_bdf,
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
