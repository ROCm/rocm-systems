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

#include "smi_nic.h"

#include <linux/if_arp.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "smi_nic_transport.h"
#include "smi_nic_vpd.h"
#include "smi_sysfs.h"

static std::string nic_type_to_string(NicType type) {
  switch (type) {
    case NicType::PCIBridge:
      return "PCI Bridge";
    case NicType::Ethernet:
      return "Ethernet";
    case NicType::InfiniBand:
      return "InfiniBand";
    default:
      return "Unknown";
  }
}

template <typename T>
static std::optional<T> get_sysfs_data(const std::string& path) {
  SmiSysfsReader::SysfsValue val;
  if (SmiSysfsReader::readLine(path, val) == SmiSysfsReader::SysfsStatus::Success) {
    if constexpr (std::is_same_v<T, std::string>) {
      if (std::holds_alternative<std::string>(val)) {
        return std::get<std::string>(val);
      }
      if (std::holds_alternative<int>(val)) {
        return std::to_string(std::get<int>(val));
      }
    } else {
      if (std::holds_alternative<int>(val)) {
        return static_cast<T>(std::get<int>(val));
      }
      if (std::holds_alternative<std::string>(val)) {
        /**
         * A numeric field can legitimately hold a non-numeric string (e.g. the
         * PCI core reports "Unknown speed" for links it cannot classify).
         * stoul would throw across the extern "C" boundary, so treat an
         * unparsable value as absent rather than propagating the exception.
         */
        try {
          return static_cast<T>(std::stoul(std::get<std::string>(val), nullptr, 0));
        } catch (const std::invalid_argument&) {
          return std::nullopt;
        } catch (const std::out_of_range&) {
          return std::nullopt;
        }
      }
    }
  }

  return std::nullopt;
}

// Reads the PCI device's VPD image (device-level attribute, present even for a
// portless fwctl-only NIC) and decodes its identity fields. Absent/unreadable
// VPD yields all-nullopt fields.
static amd::smi::nic::vpd::VpdFields read_device_vpd(const std::string& sysfs_bus_path) {
  std::ifstream file(sysfs_bus_path + "/vpd", std::ios::binary);
  if (!file) {
    return {};
  }
  std::vector<uint8_t> image((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
  return amd::smi::nic::vpd::parse_pci_vpd(image);
}

// **** SmiNicPort ****

SmiNicPort::SmiNicPort(const std::string& iface, const std::string& bdf,
                       const std::string& sysfs_class_path, const std::string& sysfs_bus_path,
                       std::shared_ptr<amd::smi::nic::transport::NicTransport> transport)
    : iface_(iface),
      bdf_(bdf),
      sysfs_class_path_(sysfs_class_path),
      sysfs_bus_path_(sysfs_bus_path),
      transport_(transport ? std::move(transport)
                           : amd::smi::nic::transport::create_transport(
                                 amd::smi::nic::transport::NicBackend_t::Auto)) {
  port_num_ = get_sysfs_data<uint32_t>(sysfs_class_path_ + "/dev_port");
  auto type_value = get_sysfs_data<int>(sysfs_class_path_ + "/type");

  if (type_value.has_value()) {
    if (type_value.value() == ARPHRD_ETHER) {
      type_ = NicType::Ethernet;
    } else if (type_value.value() == ARPHRD_INFINIBAND) {
      type_ = NicType::InfiniBand;
    } else {
      type_ = NicType::Unknown;
    }
  } else {
    type_ = NicType::Unknown;
  }
}

const std::string& SmiNicPort::interface() const { return iface_; }

const std::string& SmiNicPort::bdf() const { return bdf_; }

const std::string& SmiNicPort::sysfs_class_path() const { return sysfs_class_path_; }

const std::string& SmiNicPort::sysfs_bus_path() const { return sysfs_bus_path_; }

std::optional<std::string> SmiNicPort::mac_address() const {
  return get_sysfs_data<std::string>(sysfs_class_path_ + "/address");
}

std::optional<uint32_t> SmiNicPort::port_num() const { return port_num_; }

std::optional<uint32_t> SmiNicPort::ifindex() const {
  return get_sysfs_data<uint32_t>(sysfs_class_path_ + "/ifindex");
}

std::optional<uint8_t> SmiNicPort::carrier() const {
  return get_sysfs_data<uint8_t>(sysfs_class_path_ + "/carrier");
}

std::optional<uint16_t> SmiNicPort::mtu() const {
  return get_sysfs_data<uint16_t>(sysfs_class_path_ + "/mtu");
}

std::optional<std::string> SmiNicPort::link_state() const {
  return get_sysfs_data<std::string>(sysfs_class_path_ + "/operstate");
}

std::optional<uint32_t> SmiNicPort::link_speed() const {
  return get_sysfs_data<uint32_t>(sysfs_class_path_ + "/speed");
}

const std::string SmiNicPort::port_type() const { return nic_type_to_string(type_); }

std::string SmiNicPort::flavour() const { return "N/A"; }

std::optional<bool> SmiNicPort::autoneg() const {
  auto result = transport_->get_link_settings(iface_);
  return result.success ? std::optional<bool>(result.value.autoneg != 0) : std::nullopt;
}

std::optional<amd::smi::nic::transport::PauseParams> SmiNicPort::pause_params() const {
  auto result = transport_->get_pause_params(iface_);
  return result.success ? std::optional<amd::smi::nic::transport::PauseParams>(result.value)
                        : std::nullopt;
}

std::optional<std::string> SmiNicPort::permanent_address() const {
  auto result = transport_->get_permanent_address(iface_);
  if (!result.success) {
    return std::nullopt;
  }

  std::stringstream ss;
  ss << std::hex << std::setfill('0');
  for (size_t i = 0; i < result.value.mac.size(); i++) {
    if (i > 0) ss << ":";
    ss << std::setw(2) << static_cast<unsigned int>(result.value.mac[i]);
  }
  return ss.str();
}

void SmiNicPort::discover_infiniband() {
  std::string infiniband_path = sysfs_bus_path_ + "/infiniband";
  if (!std::filesystem::exists(infiniband_path) ||
      !std::filesystem::is_directory(infiniband_path)) {
    return;
  }

  for (const auto& entry : std::filesystem::directory_iterator(infiniband_path)) {
    if (entry.is_directory()) {
      std::string name = entry.path().filename().string();
      std::string sysfs_path = entry.path().string();
      SmiInfiniBand ib(name, sysfs_path);

      std::string ports_path = sysfs_path + "/ports";
      if (std::filesystem::exists(ports_path) && std::filesystem::is_directory(ports_path)) {
        for (const auto& port_entry : std::filesystem::directory_iterator(ports_path)) {
          if (port_entry.is_directory()) {
            std::string port_name = port_entry.path().filename().string();
            std::string port_sysfs_path = port_entry.path().string();
            SmiInfiniBandPort port(iface_, port_name, port_sysfs_path);
            port.collect_hw_counters();
            ib.add_port(port);
          }
        }
      }
      add_infiniband(ib);
    }
  }
}

void SmiNicPort::add_infiniband(const SmiInfiniBand& infiniband) {
  infiniband_.push_back(infiniband);
}

const std::vector<SmiInfiniBand>& SmiNicPort::infiniband() const { return infiniband_; }

uint8_t SmiNicPort::infiniband_num() const { return static_cast<uint8_t>(infiniband_.size()); }

void SmiNicPort::collect_vendor_statistics() {
  auto result = transport_->get_statistics(iface_);
  if (!result.success) {
    return;
  }

  for (size_t i = 0; i < result.value.names.size(); ++i) {
    const std::string& key = result.value.names[i];
    if (vendor_stat_allowed(key)) {
      vendor_stats_map_[key] = result.value.values[i];
    }
  }
}

const std::map<std::string, uint64_t>& SmiNicPort::get_vendor_stats_map() const {
  return vendor_stats_map_;
}

void SmiNicPort::collect_standard_statistics() {
  std::string stats_path = sysfs_class_path_ + "/statistics";

  if (!std::filesystem::exists(stats_path) || !std::filesystem::is_directory(stats_path)) {
    return;
  }

  for (const auto& entry : std::filesystem::directory_iterator(stats_path)) {
    if (entry.is_regular_file()) {
      std::string stat_name = entry.path().filename().string();
      auto stat_value = get_sysfs_data<uint64_t>(entry.path().string());
      if (stat_value.has_value()) {
        standard_stats_map_[stat_name] = stat_value.value();
      }
    }
  }
}

const std::map<std::string, uint64_t>& SmiNicPort::get_standard_stats_map() const {
  return standard_stats_map_;
}

std::string SmiNicPort::map_vendor_stat_to_string(SmiVendorStat stat) const {
  static const std::unordered_map<SmiVendorStat, std::string> stat_map = {
      {SmiVendorStat::TX_PACKETS, "tx_packets"},     {SmiVendorStat::RX_PACKETS, "rx_packets"},
      {SmiVendorStat::TX_BYTES, "tx_bytes"},         {SmiVendorStat::RX_BYTES, "rx_bytes"},
      {SmiVendorStat::TX_CSUM_NONE, "tx_csum_none"}, {SmiVendorStat::RX_CSUM_NONE, "rx_csum_none"},
      {SmiVendorStat::TX_CSUM, "tx_csum"},           {SmiVendorStat::TX_TSO, "tx_tso"},
      {SmiVendorStat::TX_TSO_BYTES, "tx_tso_bytes"}};

  auto it = stat_map.find(stat);
  return (it != stat_map.end()) ? it->second : "";
}

bool SmiNicPort::vendor_stat_allowed(const std::string& stat_name) const {
  for (int i = static_cast<int>(SmiVendorStat::TX_PACKETS);
       i <= static_cast<int>(SmiVendorStat::TX_TSO_BYTES); i++) {
    SmiVendorStat stat = static_cast<SmiVendorStat>(i);
    if (map_vendor_stat_to_string(stat) == stat_name) {
      return true;
    }
  }
  return false;
}

// **** SmiInfiniBandPort ****

SmiInfiniBandPort::SmiInfiniBandPort(std::string& netdev, std::string& name,
                                     const std::string& sysfs_path)
    : netdev_(netdev), name_(name), sysfs_path_(sysfs_path) {}

const std::string& SmiInfiniBandPort::name() const { return name_; }

const std::string& SmiInfiniBandPort::netdev() const { return netdev_; }

std::optional<uint8_t> SmiInfiniBandPort::port_num() const {
  try {
    return static_cast<uint8_t>(std::stoul(name_));
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<std::string> SmiInfiniBandPort::state() const {
  auto raw_state = get_sysfs_data<std::string>(sysfs_path_ + "/state");
  if (!raw_state.has_value()) {
    return std::nullopt;
  }

  const std::string& state = raw_state.value();
  auto pos = state.find(": ");

  if (pos != std::string::npos) {
    return state.substr(pos + 2);
  }

  return state;
}

std::optional<uint16_t> SmiInfiniBandPort::max_mtu() const {
  return get_sysfs_data<uint16_t>(sysfs_path_ + "/max_mtu");
}

std::optional<uint16_t> SmiInfiniBandPort::active_mtu() const {
  return get_sysfs_data<uint16_t>(sysfs_path_ + "/active_mtu");
}

void SmiInfiniBandPort::collect_hw_counters() {
  std::string hw_counters_path = sysfs_path_ + "/hw_counters";

  if (!std::filesystem::exists(hw_counters_path) ||
      !std::filesystem::is_directory(hw_counters_path)) {
    return;
  }

  for (const auto& entry : std::filesystem::directory_iterator(hw_counters_path)) {
    if (entry.is_regular_file()) {
      std::string counter_name = entry.path().filename().string();
      auto counter_value = get_sysfs_data<uint64_t>(entry.path().string());
      if (counter_value.has_value()) {
        hw_counters_map_[counter_name] = counter_value.value();
      }
    }
  }
}

const std::map<std::string, uint64_t>& SmiInfiniBandPort::get_hw_counters_map() const {
  return hw_counters_map_;
}

// **** SmiInfiniBand ****

SmiInfiniBand::SmiInfiniBand(std::string& name, const std::string& sysfs_path)
    : name_(name), sysfs_path_(sysfs_path) {}

std::string SmiInfiniBand::rdma_dev() const { return name_; }

std::optional<std::string> SmiInfiniBand::node_guid() const {
  return get_sysfs_data<std::string>(sysfs_path_ + "/node_guid");
}

std::optional<std::string> SmiInfiniBand::node_type() const {
  auto raw_node_type = get_sysfs_data<std::string>(sysfs_path_ + "/node_type");
  if (!raw_node_type.has_value()) {
    return std::nullopt;
  }

  const std::string& node_type = raw_node_type.value();
  auto pos = node_type.find(": ");

  if (pos != std::string::npos) {
    return node_type.substr(pos + 2);
  }

  return node_type;
}

std::optional<std::string> SmiInfiniBand::sys_image_guid() const {
  return get_sysfs_data<std::string>(sysfs_path_ + "/sys_image_guid");
}

std::optional<std::string> SmiInfiniBand::fw_ver() const {
  return get_sysfs_data<std::string>(sysfs_path_ + "/fw_ver");
}

void SmiInfiniBand::add_port(const SmiInfiniBandPort& port) { ports_.push_back(port); }

const std::vector<SmiInfiniBandPort>& SmiInfiniBand::ports() const { return ports_; }

uint8_t SmiInfiniBand::ports_num() const { return static_cast<uint8_t>(ports_.size()); }

NicType SmiInfiniBand::type() const { return type_; }

// **** SmiNic ****

SmiNic::SmiNic(const std::string& iface, const std::string& bdf, NicType type,
               const std::string& sysfs_class_path, const std::string& sysfs_bus_path,
               NicVendor vendor, NicProduct product)
    : iface_(iface),
      bdf_(bdf),
      type_(type),
      vendor_(vendor),
      product_(product),
      sysfs_class_path_(sysfs_class_path),
      sysfs_bus_path_(sysfs_bus_path) {}

const std::string& SmiNic::interface() const { return iface_; }

const std::string& SmiNic::bdf() const { return bdf_; }

NicType SmiNic::type() const { return type_; }

NicVendor SmiNic::vendor() const { return vendor_; }

NicProduct SmiNic::product() const { return product_; }

const std::string SmiNic::port_type() const { return nic_type_to_string(type_); }

const std::string& SmiNic::sysfs_class_path() const { return sysfs_class_path_; }

const std::string& SmiNic::sysfs_bus_path() const { return sysfs_bus_path_; }

void SmiNic::add_nic_port(const SmiNicPort& port) { ports_.push_back(port); }

const std::vector<SmiNicPort>& SmiNic::nic_ports() const { return ports_; }

uint8_t SmiNic::nic_ports_num() const { return static_cast<uint8_t>(ports_.size()); }

uint32_t SmiNic::capabilities() const {
  uint32_t caps = 0;
  if (nic_ports_num() > 0) {
    caps |= SMI_NIC_CAP_NETDEV;
  }
  return caps;
}

std::optional<uint16_t> SmiNic::vendor_id() const {
  return get_sysfs_data<uint16_t>(sysfs_bus_path_ + "/vendor");
}

std::optional<uint16_t> SmiNic::subvendor_id() const {
  return get_sysfs_data<uint16_t>(sysfs_bus_path_ + "/subsystem_vendor");
}

std::optional<uint16_t> SmiNic::device_id() const {
  return get_sysfs_data<uint16_t>(sysfs_bus_path_ + "/device");
}

std::optional<uint16_t> SmiNic::subsystem_id() const {
  return get_sysfs_data<uint16_t>(sysfs_bus_path_ + "/subsystem_device");
}

std::optional<uint8_t> SmiNic::revision() const {
  return get_sysfs_data<uint8_t>(sysfs_bus_path_ + "/revision");
}

std::optional<std::string> SmiNic::perm_address() const {
  if (ports_.empty()) {
    return std::nullopt;
  }
  return ports_[0].permanent_address();
}

std::optional<uint32_t> SmiNic::pcie_class() const {
  return get_sysfs_data<uint32_t>(sysfs_bus_path_ + "/class");
}

std::optional<uint8_t> SmiNic::max_pcie_width() const {
  return get_sysfs_data<uint8_t>(sysfs_bus_path_ + "/max_link_width");
}

std::optional<uint32_t> SmiNic::max_pcie_speed() const {
  return get_sysfs_data<uint32_t>(sysfs_bus_path_ + "/max_link_speed");
}

std::optional<uint8_t> SmiNic::numa_node() const {
  return get_sysfs_data<uint8_t>(sysfs_bus_path_ + "/numa_node");
}

std::optional<std::string> SmiNic::numa_affinity(uint8_t node) const {
  std::string path = "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist";
  return get_sysfs_data<std::string>(path);
}

std::optional<std::string> SmiNic::product_name() const {
  return read_device_vpd(sysfs_bus_path_).product_name;
}

std::optional<std::string> SmiNic::part_number() const {
  return read_device_vpd(sysfs_bus_path_).part_number;
}

std::optional<std::string> SmiNic::serial_number() const {
  return read_device_vpd(sysfs_bus_path_).serial_number;
}

std::optional<std::string> SmiNic::vendor_name() const {
  // Vendors with richer sources (e.g. Pensando VPD) override this; the base
  // resolves the discovered vendor enum so plain-SmiNic vendors (bnxt) still
  // report a name instead of falling through to "N/A".
  switch (vendor_) {
    case NicVendor::AMD:
      return std::string("AMD");
    case NicVendor::Broadcom:
      return std::string("Broadcom");
    case NicVendor::Unknown:
      break;
  }
  return std::nullopt;
}

/**
 * Generic hwmon discovery: standard NIC drivers (e.g. bnxt_en) register a hwmon
 * node under the PCI device exposing the ASIC die temperature in tempN_input
 * (millidegrees C). Vendors without such a node (e.g. Pensando pds_core, ionic)
 * return nullopt and report unsupported. Only the ASIC sensor has a generic
 * source; transceiver/board temperatures need a vendor override.
 */
std::optional<std::string> SmiNic::hwmon_temp_path(NicTempSensor sensor) const {
  if (sensor != NicTempSensor::Asic) {
    return std::nullopt;
  }

  const std::string hwmon_root = sysfs_bus_path_ + "/hwmon";
  std::error_code ec;
  if (!std::filesystem::is_directory(hwmon_root, ec)) {
    return std::nullopt;
  }

  for (const auto& entry : std::filesystem::directory_iterator(hwmon_root, ec)) {
    std::filesystem::path input = entry.path() / "temp1_input";
    if (std::filesystem::exists(input, ec)) {
      return input.string();
    }
  }
  return std::nullopt;
}

// **** SmiNicPensando ****

SmiNicPensando::SmiNicPensando(const std::string& iface, const std::string& bdf, NicType type,
                               const std::string& sysfs_class_path,
                               const std::string& sysfs_bus_path, NicVendor vendor,
                               NicProduct product)
    : SmiNic(iface, bdf, type, sysfs_class_path, sysfs_bus_path, vendor, product) {}

uint32_t SmiNicPensando::capabilities() const { return SmiNic::capabilities() | SMI_NIC_CAP_FWCTL; }

std::optional<std::string> SmiNicPensando::vendor_name() const {
  return std::string("AMD Pensando Systems, Inc.");
}
