// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "broadcom/broadcom_subsystem.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "broadcom_stats.h"

namespace fs = std::filesystem;

namespace {
// bng_en is the newer driver of the same Broadcom Ethernet family; both expose the same netdev
// and hwmon layout.
constexpr std::array<const char*, 2> kBroadcomMainDrivers = {"bnxt_en", "bng_en"};
constexpr std::array<const char*, 2> kBroadcomRdmaAuxDrivers = {"bnxt_re.rdma", "bng_re.rdma"};

// A netdev bound to a Broadcom function, held until every netdev is known so they group by
// function.
struct Netdev_t {
  std::string iface;
  std::string bdf;
  std::string class_path;
};
}  // namespace

NicVendor SmiNicSubsystemBroadcom::vendor() const { return NicVendor::Broadcom; }

bool SmiNicSubsystemBroadcom::driver_loaded(const std::string& bdf, DriverType driver_type) const {
  switch (driver_type) {
    case DriverType::Main:
      return std::any_of(
          kBroadcomMainDrivers.begin(), kBroadcomMainDrivers.end(), [&](const char* driver) {
            return is_driver_bound_to_bdf(driver_sysfs_root_ + kPciDriversDir + driver, bdf,
                                          /*is_canonical_match=*/false);
          });
    case DriverType::Rdma:
      return std::any_of(
          kBroadcomRdmaAuxDrivers.begin(), kBroadcomRdmaAuxDrivers.end(), [&](const char* driver) {
            return is_driver_bound_to_bdf(driver_sysfs_root_ + kAuxDriversDir + driver, bdf,
                                          /*is_canonical_match=*/true);
          });
    default:
      return false;
  }
}

auto SmiNicSubsystemBroadcom::is_driver_present(DriverType driver_type) const -> bool {
  const auto is_any_present = [&](const auto& drivers, const std::string& drivers_dir) {
    return std::any_of(drivers.begin(), drivers.end(), [&](const char* driver) {
      return is_directory_present(driver_sysfs_root_ + drivers_dir + driver);
    });
  };
  switch (driver_type) {
    case DriverType::Main:
      return is_any_present(kBroadcomMainDrivers, kPciDriversDir);
    case DriverType::Rdma:
      return is_any_present(kBroadcomRdmaAuxDrivers, kAuxDriversDir);
    default:
      return false;
  }
}

void SmiNicSubsystemBroadcom::discover(
    const std::string& pci_path, const std::string& net_path,
    std::shared_ptr<amd::smi::nic::transport::NicTransport> transport) {
  nics_.clear();
  std::error_code ec;
  auto netdevs = std::vector<Netdev_t>{};

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
    if (!resolve_bdf(device_symlink, bdf)) {
      continue;
    }

    const std::string sysfs_bus_path = pci_path + "/" + bdf;
    // Broadcom spans many device ids; vendor + driver bind identify it.
    [[maybe_unused]] auto [vendor_id, device_id] = read_pci_ids(sysfs_bus_path);
    if ((vendor_id != VENDOR_ID) || !is_bound_to_broadcom_main_driver(sysfs_bus_path)) {
      continue;
    }

    netdevs.push_back({iface, bdf, it->path().string()});
  }

  // Directory order is unspecified; sorting by name makes the netdev that names the NIC stable.
  std::sort(netdevs.begin(), netdevs.end(),
            [](const Netdev_t& lhs, const Netdev_t& rhs) { return (lhs.iface < rhs.iface); });

  // A function can expose several netdevs (a PF with switchdev representors); they are ports of
  // one NIC, not NICs that share a BDF.
  auto nic_by_bdf = std::map<std::string, SmiNic*>{};
  for (const auto& netdev : netdevs) {
    const auto sysfs_bus_path = pci_path + "/" + netdev.bdf;
    auto port = SmiNicPort{netdev.iface,   netdev.bdf, netdev.class_path,
                           sysfs_bus_path, transport,  &kBroadcomStatTable};
    port.discover_infiniband();

    auto found = nic_by_bdf.find(netdev.bdf);
    if (found == nic_by_bdf.end()) {
      auto nic =
          std::make_unique<SmiNic>(netdev.iface, netdev.bdf, NicType::Ethernet, netdev.class_path,
                                   sysfs_bus_path, NicVendor::Broadcom, NicProduct::Unknown);
      found = nic_by_bdf.emplace(netdev.bdf, nic.get()).first;
      nics_.push_back(std::move(nic));
    }
    found->second->add_nic_port(port);
  }
}

bool SmiNicSubsystemBroadcom::is_bound_to_broadcom_main_driver(
    const std::string& sysfs_bus_path) const {
  std::error_code ec;
  fs::path driver_link = fs::path(sysfs_bus_path) / "driver";
  if (!fs::is_symlink(driver_link, ec)) {
    return false;
  }
  const std::string bound_driver = fs::read_symlink(driver_link, ec).filename().string();
  return (!ec && std::any_of(kBroadcomMainDrivers.begin(), kBroadcomMainDrivers.end(),
                             [&](const char* driver) { return (bound_driver == driver); }));
}

const std::vector<std::unique_ptr<SmiNic>>& SmiNicSubsystemBroadcom::get_nics() const {
  return nics_;
}
