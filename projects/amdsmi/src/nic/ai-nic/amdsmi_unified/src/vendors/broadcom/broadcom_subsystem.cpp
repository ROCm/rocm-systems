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

#include "broadcom/broadcom_subsystem.h"

#include <filesystem>

namespace fs = std::filesystem;

NicVendor SmiNicSubsystemBroadcom::vendor() const { return NicVendor::Broadcom; }

bool SmiNicSubsystemBroadcom::driver_loaded(const std::string& bdf, DriverType driver_type) const {
  switch (driver_type) {
    case DriverType::Main:
      return driver_binds_bdf(driver_sysfs_root_ + "/sys/bus/pci/drivers/bnxt_en", bdf,
                              /*match_canonical=*/false);
    case DriverType::Rdma:
      return driver_binds_bdf(driver_sysfs_root_ + "/sys/bus/auxiliary/drivers/bnxt_re.rdma", bdf,
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

  for (const auto& net_entry : fs::directory_iterator(net_path, ec)) {
    if (ec) {
      continue;
    }

    const std::string iface = net_entry.path().filename().string();
    const std::string device_symlink = net_entry.path().string() + "/device";
    if (!fs::exists(device_symlink, ec) || !fs::is_symlink(device_symlink, ec)) {
      continue;
    }

    std::string bdf;
    if (!resolve_bdf(device_symlink, bdf)) {
      continue;
    }

    const std::string sysfs_bus_path = pci_path + "/" + bdf;
    auto [vendor_id, device_id] = read_pci_ids(sysfs_bus_path);
    (void)device_id;  // bnxt spans many device ids; vendor + driver bind identify it.
    if (vendor_id != VENDOR_ID || !bound_to_bnxt_en(sysfs_bus_path)) {
      continue;
    }

    auto nic = std::make_unique<SmiNic>(iface, bdf, NicType::Ethernet, net_entry.path().string(),
                                        sysfs_bus_path, NicVendor::Broadcom, NicProduct::Unknown);
    SmiNicPort port(iface, bdf, net_entry.path().string(), sysfs_bus_path, transport);
    port.discover_infiniband();
    port.collect_vendor_statistics();
    port.collect_standard_statistics();
    nic->add_nic_port(port);
    nics_.push_back(std::move(nic));
  }
}

bool SmiNicSubsystemBroadcom::bound_to_bnxt_en(const std::string& sysfs_bus_path) const {
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
