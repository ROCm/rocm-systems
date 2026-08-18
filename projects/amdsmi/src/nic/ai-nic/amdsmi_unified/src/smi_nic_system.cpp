// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "smi_nic_system.h"

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <regex>
#include <set>
#include <sstream>
#include <string>

#include "smi_nic_subsystem.h"
#include "smi_sysfs.h"
#include "vendor_registry.h"

namespace fs = std::filesystem;

SmiNicSystem::SmiNicSystem() : SmiNicSystem("/sys/bus/pci/devices", "/sys/class/net") {}

SmiNicSystem::SmiNicSystem(const std::string& pci_path, const std::string& net_path)
    : net_path_(net_path),
      pci_path_(pci_path),
      transport_(amd::smi::nic::transport::create_transport(
          amd::smi::nic::transport::NicBackend_t::Auto)) {
  for (auto& plugin : make_default_vendor_plugins()) {
    register_subsystem(std::move(plugin));
  }
}

void SmiNicSystem::register_subsystem(std::unique_ptr<SmiNicSubsystem> subsystem) {
  subsystems_.push_back(std::move(subsystem));
}

bool SmiNicSystem::interface_exists(const std::string& iface) {
  std::error_code ec;
  return fs::exists(fs::path(net_path_) / fs::path(iface).string(), ec);
}

bool SmiNicSystem::is_driver_loaded(const std::string& bdf, DriverType driver_type) const {
  const SmiNic* nic = nullptr;

  for (const auto& entry : nics_) {
    if (entry->bdf() == bdf) {
      nic = entry;
      break;
    }

    for (const auto& port : entry->nic_ports()) {
      if (port.bdf() == bdf) {
        nic = entry;
        break;
      }
    }

    if (nic) {
      break;
    }
  }

  if (!nic) {
    return false;
  }

  // Resolve to the plugin that discovered this NIC. Keying on the vendor enum
  // instead would hand the query to the first plugin reporting that vendor, so
  // any later plugin sharing it (IFoE behind Pensando, both AMD) is shadowed
  // and answers against the wrong driver path.
  for (const auto& subsystem : subsystems_) {
    for (const auto& owned : subsystem->get_nics()) {
      if (owned.get() == nic) {
        return subsystem->is_driver_loaded(bdf, driver_type);
      }
    }
  }

  return false;
}

void SmiNicSystem::discover_nics(bool ainic_only) {
  std::error_code ec;

  if (!fs::exists(pci_path_, ec) || !fs::is_directory(pci_path_, ec)) {
    return;
  }

  nics_.clear();
  for (auto& subsystem : subsystems_) {
    subsystem->discover(pci_path_, net_path_, transport_);
    const auto& subsys_nics = subsystem->get_nics();
    for (const auto& nic : subsys_nics) {
      if (ainic_only && nic->product() != NicProduct::AINIC) {
        continue;
      }
      nics_.push_back(nic.get());
    }
  }

  // Sort NICs by BDF. This orders the vector this class exposes, not the index
  // amd-smi prints: that follows the socket walk, and a fabric endpoint sharing
  // its GPU's socket takes that socket's position regardless of the order here.
  std::sort(nics_.begin(), nics_.end(), [](const SmiNic* x, const SmiNic* y) {
    return parse_bdf(x->bdf()) < parse_bdf(y->bdf());
  });
}

const std::vector<const SmiNic*>& SmiNicSystem::get_nics() const { return nics_; }

std::vector<std::string> SmiNicSystem::list_bdfs() {
  std::vector<std::string> bdfs;
  for (const auto* nic : nics_) {
    bdfs.push_back(nic->bdf());
  }
  return bdfs;
}

const SmiNic* SmiNicSystem::get_nic_by_interface(const std::string& iface) const {
  for (const auto* nic : nics_) {
    if (nic->interface() == iface) {
      return nic;
    }
  }

  return nullptr;
}

const SmiNic* SmiNicSystem::get_nic_by_bdf(const std::string& bdf) const {
  for (const auto* nic : nics_) {
    if (nic->bdf() == bdf) {
      return nic;
    }
  }

  return nullptr;
}

const SmiNic* SmiNicSystem::get_nic_by_bdf(uint64_t bdf) const {
  uint64_t function_number = bdf & 0x7;
  uint64_t device_number = (bdf >> 3) & 0x1F;
  uint64_t bus_number = (bdf >> 8) & 0xFF;
  uint64_t domain_number = (bdf >> 16) & 0xFFFFFFFF;
  std::ostringstream oss;

  oss << std::hex << std::setfill('0') << std::setw(4) << domain_number << ":" << std::setw(2)
      << bus_number << ":" << std::setw(2) << device_number << "." << std::setw(1)
      << function_number;

  return get_nic_by_bdf(oss.str());
}
