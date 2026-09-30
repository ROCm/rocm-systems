// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "broadcom/broadcom_subsystem.h"

#include <filesystem>

#include "broadcom_stats.h"

namespace fs = std::filesystem;

NicVendor SmiNicSubsystemBroadcom::vendor() const { return NicVendor::Broadcom; }

bool SmiNicSubsystemBroadcom::is_driver_loaded(const std::string& bdf,
                                               DriverType driver_type) const {
  switch (driver_type) {
    case DriverType::Main:
      return is_driver_bound_to_bdf(driver_sysfs_root_ + "/sys/bus/pci/drivers/bnxt_en", bdf,
                                    /*match_canonical=*/false);
    case DriverType::Rdma:
      return is_driver_bound_to_bdf(driver_sysfs_root_ + "/sys/bus/auxiliary/drivers/bnxt_re.rdma",
                                    bdf,
                                    /*match_canonical=*/true);
    default:
      return false;
  }
}

void SmiNicSubsystemBroadcom::discover(
    const std::string& pci_path, const std::string& net_path,
    std::shared_ptr<amd::smi::nic::transport::NicTransport> transport) {
  nics_.clear();
  std::error_code ec;

  // Manual increment(ec): operator++ on directory_iterator throws on a
  // mid-scan read error; the non-throwing form only guards construction.
  for (auto it = fs::directory_iterator(net_path, ec); (!ec && (it != fs::directory_iterator()));
       it.increment(ec)) {
    const std::string iface = it->path().filename().string();
    const std::string device_symlink = it->path().string() + "/device";
    if (!fs::exists(device_symlink, ec) || !fs::is_symlink(device_symlink, ec)) {
      continue;
    }

    std::string bdf;
    if (!was_bdf_resolved(device_symlink, bdf)) {
      continue;
    }

    const std::string sysfs_bus_path = pci_path + "/" + bdf;
    auto [vendor_id, device_id] = read_pci_ids(sysfs_bus_path);
    (void)device_id;  // bnxt spans many device ids; vendor + driver bind identify it.
    if (vendor_id != VENDOR_ID || !is_bound_to_bnxt_en(sysfs_bus_path)) {
      continue;
    }

    auto nic = std::make_unique<SmiNic>(iface, bdf, NicType::Ethernet, it->path().string(),
                                        sysfs_bus_path, NicVendor::Broadcom, NicProduct::Unknown);
    SmiNicPort port(iface, bdf, it->path().string(), sysfs_bus_path, transport,
                    &kBroadcomStatTable);
    port.discover_infiniband();
    nic->add_nic_port(port);
    nics_.push_back(std::move(nic));
  }
}

bool SmiNicSubsystemBroadcom::is_bound_to_bnxt_en(const std::string& sysfs_bus_path) const {
  std::error_code ec;
  fs::path driver_link = fs::path(sysfs_bus_path) / "driver";
  if (!fs::is_symlink(driver_link, ec)) {
    return false;
  }
  return fs::read_symlink(driver_link, ec).filename().string() == "bnxt_en" && !ec;
}

const std::vector<std::unique_ptr<SmiNic>>& SmiNicSubsystemBroadcom::get_nics() const {
  return nics_;
}
