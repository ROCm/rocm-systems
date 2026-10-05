// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "smi_nic.h"

#include <linux/if_arp.h>

#include <algorithm>
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
#include <vector>

#include "smi_nic_log.h"
#include "smi_nic_transport.h"
#include "smi_nic_vpd.h"
#include "smi_sysfs.h"

uint64_t parse_bdf(const std::string& bdf) {
  if (bdf.length() != 12) {
    return 0;
  }

  if ((bdf[4] != ':') || (bdf[7] != ':') || (bdf[10] != '.')) {
    return 0;
  }

  try {
    uint64_t domain = std::stoul(bdf.substr(0, 4), nullptr, 16);
    uint64_t bus = std::stoul(bdf.substr(5, 2), nullptr, 16);
    uint64_t device = std::stoul(bdf.substr(8, 2), nullptr, 16);
    uint64_t function = std::stoul(bdf.substr(11, 1), nullptr, 16);

    return (domain << 16) | (bus << 8) | (device << 3) | function;
  } catch (const std::exception&) {
    return 0;
  }
}

static std::string nic_type_to_string(NicType type) {
  switch (type) {
    case NicType::PCIBridge:
      return "PCI Bridge";
    case NicType::Ethernet:
      return "Ethernet";
    case NicType::InfiniBand:
      return "InfiniBand";
    case NicType::Fabric:
      return "Fabric";
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
  const std::string vpd_path = sysfs_bus_path + "/vpd";
  std::ifstream file(vpd_path, std::ios::binary);
  if (!file) {
    if (amd::smi::nic::log::is_enabled()) {
      NIC_LOG_DEBUG("vpd read " + vpd_path + " -> FAIL cannot open");
    }
    return {};
  }
  std::vector<uint8_t> image((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
  auto fields = amd::smi::nic::vpd::parse_pci_vpd(image);
  fields.is_vpd_readable = true;
  if (amd::smi::nic::log::is_enabled()) {
    NIC_LOG_DEBUG("vpd read " + vpd_path + " -> SUCCESS bytes=" + std::to_string(image.size()) +
                  " product=" + fields.product_name.value_or("absent") +
                  " part=" + fields.part_number.value_or("absent") + " serial=" +
                  (fields.serial_number.has_value()
                       ? amd::smi::nic::log::mask_tail(fields.serial_number.value())
                       : "absent"));
  }
  return fields;
}

// **** SmiNicPort ****

SmiNicPort::SmiNicPort(const std::string& iface, const std::string& bdf,
                       const std::string& sysfs_class_path, const std::string& sysfs_bus_path,
                       std::shared_ptr<amd::smi::nic::transport::NicTransport> transport,
                       const StatTable_t* stat_table)
    : iface_(iface),
      bdf_(bdf),
      sysfs_class_path_(sysfs_class_path),
      sysfs_bus_path_(sysfs_bus_path),
      transport_(transport ? std::move(transport)
                           : amd::smi::nic::transport::create_transport(
                                 amd::smi::nic::transport::NicBackend_t::Auto)),
      stat_table_(stat_table) {
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
    if (i > 0) {
      ss << ":";
    }
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

  std::error_code ec;
  // Manual increment(ec): operator++ on directory_iterator throws on a
  // mid-scan read error; the non-throwing form only guards construction.
  for (auto it = std::filesystem::directory_iterator(infiniband_path, ec);
       (!ec && (it != std::filesystem::directory_iterator())); it.increment(ec)) {
    if (it->is_directory()) {
      std::string name = it->path().filename().string();
      std::string sysfs_path = it->path().string();
      SmiInfiniBand ib(name, sysfs_path);

      std::string ports_path = sysfs_path + "/ports";
      if (std::filesystem::exists(ports_path) && std::filesystem::is_directory(ports_path)) {
        std::error_code port_ec;
        for (auto port_it = std::filesystem::directory_iterator(ports_path, port_ec);
             (!port_ec && (port_it != std::filesystem::directory_iterator()));
             port_it.increment(port_ec)) {
          if (port_it->is_directory()) {
            std::string port_name = port_it->path().filename().string();
            std::string port_sysfs_path = port_it->path().string();
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

void SmiNicPort::collect_vendor_statistics() const {
  // Built fresh and assigned wholesale (not merged) so a counter the driver
  // stops reporting disappears instead of keeping its last value forever,
  // and so FEC counters (previously merged via emplace into a map that
  // already held them from a prior call) get refreshed instead of frozen.
  std::map<std::string, uint64_t> merged;

  auto result = transport_->get_statistics(iface_);
  if (result.success) {
    for (size_t i = 0; i < result.value.names.size(); ++i) {
      merged.emplace(result.value.names[i], result.value.values[i]);
    }
  }

  // FEC stats are netlink-only; a backend without them (ioctl, no libnl-3)
  // reports failure here, which is not an error for the port; the FEC
  // counters simply stay absent, same as any other name the table allows but
  // the driver does not expose.
  auto fec_result = transport_->get_fec_statistics(iface_);
  if (amd::smi::nic::log::is_enabled()) {
    std::string fec_text = "fec stats " + iface_ + ":";
    if (fec_result.success) {
      for (size_t i = 0; i < fec_result.value.names.size(); ++i) {
        fec_text +=
            " " + fec_result.value.names[i] + "=" + std::to_string(fec_result.value.values[i]);
      }
    } else {
      fec_text += " error=" + std::to_string(fec_result.error_code);
    }
    NIC_LOG_DEBUG(fec_text);
  }
  if (fec_result.success) {
    for (size_t i = 0; i < fec_result.value.names.size(); ++i) {
      // `-S` wins on a name collision: emplace is a no-op if the key already
      // exists from the loop above.
      merged.emplace(fec_result.value.names[i], fec_result.value.values[i]);
    }
  }

  vendor_stats_map_ = std::move(merged);
}

std::map<std::string, uint64_t> SmiNicPort::get_vendor_stats_map(StatTier_t scope) const {
  std::map<std::string, uint64_t> result;
  if (!stat_table_) {
    return result;
  }

  for (const auto& entry : *stat_table_) {
    if ((scope == StatTier_t::Default) && (entry.tier != StatTier_t::Default)) {
      continue;
    }
    auto it = vendor_stats_map_.find(entry.name);
    if (it != vendor_stats_map_.end()) {
      result.emplace(entry.name, it->second);
    }
  }
  return result;
}

void SmiNicPort::collect_standard_statistics() const {
  std::string stats_path = sysfs_class_path_ + "/statistics";

  if (!std::filesystem::exists(stats_path) || !std::filesystem::is_directory(stats_path)) {
    return;
  }

  std::error_code ec;
  // Manual increment(ec): operator++ on directory_iterator throws on a
  // mid-scan read error; the non-throwing form only guards construction.
  for (auto it = std::filesystem::directory_iterator(stats_path, ec);
       (!ec && (it != std::filesystem::directory_iterator())); it.increment(ec)) {
    if (it->is_regular_file()) {
      std::string stat_name = it->path().filename().string();
      auto stat_value = get_sysfs_data<uint64_t>(it->path().string());
      if (stat_value.has_value()) {
        standard_stats_map_[stat_name] = stat_value.value();
      }
    }
  }
}

const std::map<std::string, uint64_t>& SmiNicPort::get_standard_stats_map() const {
  return standard_stats_map_;
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

  std::error_code ec;
  // Manual increment(ec): operator++ on directory_iterator throws on a
  // mid-scan read error; the non-throwing form only guards construction.
  for (auto it = std::filesystem::directory_iterator(hw_counters_path, ec);
       (!ec && (it != std::filesystem::directory_iterator())); it.increment(ec)) {
    if (it->is_regular_file()) {
      std::string counter_name = it->path().filename().string();
      auto counter_value = get_sysfs_data<uint64_t>(it->path().string());
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

// Inserts in BDF order: the netdev walk that feeds this yields readdir order,
// and port 0 is what telemetry and the permanent address resolve through.
void SmiNic::add_nic_port(const SmiNicPort& port) {
  const uint64_t key = parse_bdf(port.bdf());
  const auto pos = std::upper_bound(
      ports_.begin(), ports_.end(), key,
      [](uint64_t lhs, const SmiNicPort& rhs) { return lhs < parse_bdf(rhs.bdf()); });
  ports_.insert(pos, port);
}

const std::vector<SmiNicPort>& SmiNic::nic_ports() const { return ports_; }

uint8_t SmiNic::nic_ports_num() const { return static_cast<uint8_t>(ports_.size()); }

const std::string& SmiNic::telemetry_bdf() const {
  return ports_.empty() ? bdf_ : ports_.front().bdf();
}

const std::string& SmiNic::mgmt_bdf() const {
  return mgmt_bdf_.empty() ? telemetry_bdf() : mgmt_bdf_;
}

void SmiNic::set_mgmt_bdf(const std::string& bdf) {
  if (bdf.empty()) {
    return;
  }
  mgmt_bdf_ = bdf;
}

const std::string& SmiNic::telemetry_sysfs_bus_path() const {
  return ports_.empty() ? sysfs_bus_path_ : ports_.front().sysfs_bus_path();
}

SmiNicKind SmiNic::kind() const {
  // Checked first: SmiNicIfoe is also built with NicProduct::AINIC.
  if (type_ == NicType::Fabric) {
    return SMI_NIC_KIND_UALOE;
  }
  if (product_ == NicProduct::AINIC) {
    return SMI_NIC_KIND_AINIC;
  }
  return SMI_NIC_KIND_OTHER;
}

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

// No port-0 fallback here, unlike the VPD identity strings below: these are PCI
// config-space properties of the registered function itself (the bridge, for a
// Pensando card), not marketing identity that could equally live on either function.
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

// The registered NIC is the PCIe bridge above its ports, and on a Salina card
// only the ionic function beneath it exposes a vpd node. Prefer the NIC's own
// image field by field, so a bridge that carries only some keywords still
// answers with the port's value for the rest.
static amd::smi::nic::vpd::VpdFields read_identity_vpd(const std::string& sysfs_bus_path,
                                                       const std::vector<SmiNicPort>& ports) {
  auto fields = read_device_vpd(sysfs_bus_path);
  // Elides the second read only; the merge below never overwrites a present field,
  // so dropping is_complete would cost an extra VPD open, not change any answer.
  const bool is_complete = (fields.product_name.has_value() && fields.part_number.has_value() &&
                            fields.serial_number.has_value());
  if (is_complete || ports.empty()) {
    return fields;
  }

  const auto port_fields = read_device_vpd(ports.front().sysfs_bus_path());
  if (!fields.product_name.has_value()) {
    fields.product_name = port_fields.product_name;
  }
  if (!fields.part_number.has_value()) {
    fields.part_number = port_fields.part_number;
  }
  if (!fields.serial_number.has_value()) {
    fields.serial_number = port_fields.serial_number;
  }
  fields.is_vpd_readable = (fields.is_vpd_readable || port_fields.is_vpd_readable);
  return fields;
}

std::optional<std::string> SmiNic::product_name() const {
  return read_identity_vpd(sysfs_bus_path_, ports_).product_name;
}

std::optional<std::string> SmiNic::part_number() const {
  return read_identity_vpd(sysfs_bus_path_, ports_).part_number;
}

std::optional<std::string> SmiNic::serial_number() const {
  return read_identity_vpd(sysfs_bus_path_, ports_).serial_number;
}

bool SmiNic::is_vpd_readable() const {
  return read_identity_vpd(sysfs_bus_path_, ports_).is_vpd_readable;
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
 * Generic hwmon discovery: standard NIC drivers (bnxt_en, ionic) register a
 * hwmon node under the PCI device exposing the ASIC die temperature in
 * tempN_input (millidegrees C). A device without such a node returns nullopt
 * and reports unsupported. Only the ASIC sensor has a generic source;
 * transceiver/board temperatures need a vendor override.
 */
std::optional<std::string> SmiNic::hwmon_temp_path(NicTempSensor sensor) const {
  if (sensor != NicTempSensor::Asic) {
    return std::nullopt;
  }

  const std::string hwmon_root = telemetry_sysfs_bus_path() + "/hwmon";
  std::error_code ec;
  if (!std::filesystem::is_directory(hwmon_root, ec)) {
    if (amd::smi::nic::log::is_enabled()) {
      NIC_LOG_DEBUG("hwmon scan " + hwmon_root + " -> FAIL not a directory");
    }
    return std::nullopt;
  }

  // Manual increment(ec): operator++ on directory_iterator throws on a
  // mid-scan read error; the non-throwing form only guards construction.
  for (auto it = std::filesystem::directory_iterator(hwmon_root, ec);
       (!ec && (it != std::filesystem::directory_iterator())); it.increment(ec)) {
    std::filesystem::path input = it->path() / "temp1_input";
    if (std::filesystem::exists(input, ec)) {
      if (amd::smi::nic::log::is_enabled()) {
        NIC_LOG_DEBUG("hwmon scan " + hwmon_root + " -> SUCCESS " + input.string());
      }
      return input.string();
    }
  }
  if (amd::smi::nic::log::is_enabled()) {
    NIC_LOG_DEBUG("hwmon scan " + hwmon_root + " -> FAIL no temp1_input");
  }
  return std::nullopt;
}

// **** SmiNicPensando ****

SmiNicPensando::SmiNicPensando(const std::string& iface, const std::string& bdf, NicType type,
                               const std::string& sysfs_class_path,
                               const std::string& sysfs_bus_path, NicVendor vendor,
                               NicProduct product)
    : SmiNic(iface, bdf, type, sysfs_class_path, sysfs_bus_path, vendor, product) {}

uint32_t SmiNicPensando::capabilities() const {
  uint32_t caps = SmiNic::capabilities();
  if (!mgmt_bdf_.empty()) {
    caps |= SMI_NIC_CAP_FWCTL;
  }
  return caps;
}

std::optional<std::string> SmiNicPensando::vendor_name() const {
  return std::string("AMD Pensando Systems, Inc.");
}
