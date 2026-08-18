// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_UNIFIED_SMI_NIC_H_
#define AMDSMI_UNIFIED_SMI_NIC_H_

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "smi_nic_transport.h"

/**
 * @brief Convert BDF string format to uint64_t
 *
 * Converts a BDF string to uint64_t format:
 * (domain << 16) | (bus << 8) | (device << 3) | function
 *
 * @param bdf BDF string
 * @return uint64_t BDF value, or 0 if parsing fails
 */
uint64_t parse_bdf(const std::string& bdf);

enum class NicType {
  Unknown,
  PCIBridge,
  Ethernet,
  InfiniBand,
  Fabric,  // accelerator-fabric endpoint (IFoE), no host network port
};

enum class NicVendor { Unknown, AMD, Broadcom };

enum class NicProduct {
  Unknown,
  AINIC,  // AMD Pensando AINIC
};

// Bitmask of a NIC's management/connectivity capabilities. Mirrors the public
// amdsmi_nic_capability_bits_t; the bit values MUST stay in sync with it.
enum SmiNicCapability : uint32_t {
  SMI_NIC_CAP_FWCTL = 1u << 0,   // firmware-control mgmt function (POLLARA/pds_core)
  SMI_NIC_CAP_NETDEV = 1u << 1,  // exposes host network port(s)
};

/**
 * Board temperature sensors a NIC may expose. A vendor that lacks a given
 * sensor reports it as unsupported (sentinel) rather than fabricating a value.
 */
enum class NicTempSensor : uint8_t { Asic, Transceiver, Board };

class SmiInfiniBandPort {
 public:
  SmiInfiniBandPort(std::string& netdev, std::string& name, const std::string& sysfs_path_);

  const std::string& netdev() const;
  const std::string& name() const;
  std::optional<uint8_t> port_num() const;
  std::optional<std::string> state() const;
  std::optional<uint16_t> max_mtu() const;
  std::optional<uint16_t> active_mtu() const;
  void collect_hw_counters();
  const std::map<std::string, uint64_t>& get_hw_counters_map() const;

 private:
  std::string netdev_;
  std::string name_;
  std::string sysfs_path_;
  std::map<std::string, uint64_t> hw_counters_map_;
};

class SmiInfiniBand {
 public:
  SmiInfiniBand(std::string& name, const std::string& sysfs_path);

  std::string rdma_dev() const;
  std::optional<std::string> node_guid() const;
  std::optional<std::string> node_type() const;
  std::optional<std::string> sys_image_guid() const;
  std::optional<std::string> fw_ver() const;

  void add_port(const SmiInfiniBandPort& port);
  const std::vector<SmiInfiniBandPort>& ports() const;
  uint8_t ports_num() const;
  NicType type() const;

 private:
  std::string name_;
  std::string sysfs_path_;
  NicType type_ = NicType::InfiniBand;
  std::vector<SmiInfiniBandPort> ports_;
};

class SmiNicPort {
 public:
  SmiNicPort(const std::string& iface, const std::string& bdf, const std::string& sysfs_class_path,
             const std::string& sysfs_bus_path,
             std::shared_ptr<amd::smi::nic::transport::NicTransport> transport = nullptr);

  const std::string& interface() const;
  const std::string& bdf() const;
  const std::string& sysfs_class_path() const;
  const std::string& sysfs_bus_path() const;

  std::optional<std::string> mac_address() const;
  std::optional<uint32_t> port_num() const;
  std::optional<uint32_t> ifindex() const;
  std::optional<uint8_t> carrier() const;
  std::optional<uint16_t> mtu() const;
  std::optional<std::string> link_state() const;
  std::optional<uint32_t> link_speed() const;

  const std::string port_type() const;
  std::string flavour() const;

  std::optional<bool> autoneg() const;
  std::optional<amd::smi::nic::transport::PauseParams> pause_params() const;
  std::optional<std::string> permanent_address() const;

  void discover_infiniband();
  void add_infiniband(const SmiInfiniBand& infiniband);
  const std::vector<SmiInfiniBand>& infiniband() const;
  uint8_t infiniband_num() const;
  void collect_vendor_statistics();
  const std::map<std::string, uint64_t>& get_vendor_stats_map() const;
  void collect_standard_statistics();
  const std::map<std::string, uint64_t>& get_standard_stats_map() const;

 private:
  enum class SmiVendorStat {
    TX_PACKETS,
    RX_PACKETS,
    TX_BYTES,
    RX_BYTES,
    TX_CSUM_NONE,
    RX_CSUM_NONE,
    TX_CSUM,
    TX_TSO,
    TX_TSO_BYTES
  };

  std::string map_vendor_stat_to_string(SmiVendorStat stat) const;
  bool vendor_stat_allowed(const std::string& stat_name) const;

  std::string iface_;
  std::string bdf_;
  NicType type_;
  std::string sysfs_class_path_;
  std::string sysfs_bus_path_;
  std::optional<uint32_t> port_num_;
  std::vector<SmiInfiniBand> infiniband_;
  std::map<std::string, uint64_t> vendor_stats_map_;
  std::map<std::string, uint64_t> standard_stats_map_;
  std::shared_ptr<amd::smi::nic::transport::NicTransport> transport_;
};

class SmiNic {
 public:
  SmiNic(const std::string& iface, const std::string& bdf, NicType type = NicType::Unknown,
         const std::string& sysfs_class_path = "", const std::string& sysfs_bus_path = "",
         NicVendor vendor = NicVendor::Unknown, NicProduct product = NicProduct::Unknown);
  virtual ~SmiNic() = default;

  const std::string& interface() const;
  const std::string& bdf() const;
  NicType type() const;
  NicVendor vendor() const;
  NicProduct product() const;
  const std::string port_type() const;
  const std::string& sysfs_class_path() const;
  const std::string& sysfs_bus_path() const;

  void add_nic_port(const SmiNicPort& port);
  const std::vector<SmiNicPort>& nic_ports() const;
  uint8_t nic_ports_num() const;

  /**
   * Address of the function telemetry registers on. Where the handle is the
   * PCIe bridge above the ports, hwmon and the devlink instance live on the
   * port function beneath it, not on the bridge; a portless fwctl-only NIC
   * answers on its own address. For a single-function NIC the two coincide.
   *
   * On a multi-port card this reports port 0 only, and which port is 0 is
   * decided by add_nic_port's BDF-ordered insert, not here.
   */
  const std::string& telemetry_bdf() const;
  const std::string& telemetry_sysfs_bus_path() const;

  /**
   * Address of the card's management function, which is where devlink
   * registers the health reporter. On a Pensando card that is pds_core, a
   * function behind the other downstream port of the card's internal switch,
   * so it is neither the bridge handle nor a sibling of the ionic port.
   * Falls back to telemetry_bdf() for a card with no separate such function.
   */
  const std::string& mgmt_bdf() const;
  // Ignores an empty argument, so an unset mgmt_bdf_ means "no such function on
  // this card" and nothing else; capabilities() reads it as exactly that.
  void set_mgmt_bdf(const std::string& bdf);

  // Capability bitmask (SmiNicCapability). Base reports NETDEV iff a host port
  // exists; vendors that expose firmware-control management override to add it.
  virtual uint32_t capabilities() const;

  std::optional<uint16_t> vendor_id() const;
  std::optional<uint16_t> subvendor_id() const;
  std::optional<uint16_t> device_id() const;
  std::optional<uint16_t> subsystem_id() const;
  std::optional<uint8_t> revision() const;
  std::optional<std::string> perm_address() const;
  std::optional<uint32_t> pcie_class() const;
  std::optional<uint8_t> max_pcie_width() const;
  std::optional<uint32_t> max_pcie_speed() const;
  std::optional<uint8_t> numa_node() const;
  std::optional<std::string> numa_affinity(uint8_t node) const;
  // Vendor specific
  virtual std::optional<std::string> product_name() const;
  virtual std::optional<std::string> vendor_name() const;
  virtual std::optional<std::string> part_number() const;
  virtual std::optional<std::string> serial_number() const;

  /**
   * Absolute path of the hwmon `tempN_input` file (millidegrees C) backing
   * `sensor`, or nullopt if this NIC has no such sensor. The base performs
   * generic hwmon discovery under the PCI device, which covers standard NIC
   * drivers (e.g. bnxt_en) for the ASIC sensor; a vendor whose sensors live
   * elsewhere overrides this. (Health reporters are enumerated over devlink and
   * need no per-vendor mapping, so there is no reporter-name resolver here.)
   */
  virtual std::optional<std::string> hwmon_temp_path(NicTempSensor sensor) const;

 protected:
  std::string iface_;
  std::string bdf_;
  NicType type_;
  NicVendor vendor_;
  NicProduct product_;
  std::string sysfs_class_path_;
  std::string sysfs_bus_path_;
  std::string mgmt_bdf_;
  std::vector<SmiNicPort> ports_;
};

class SmiNicPensando : public SmiNic {
 public:
  SmiNicPensando(const std::string& iface, const std::string& bdf, NicType type = NicType::Unknown,
                 const std::string& sysfs_class_path = "", const std::string& sysfs_bus_path = "",
                 NicVendor vendor = NicVendor::AMD, NicProduct product = NicProduct::AINIC);

  std::optional<std::string> vendor_name() const override;

  // Adds FWCTL when discovery found this card's pds_core function, which is the
  // surface the bit promises; host netdev ports are independent of it.
  uint32_t capabilities() const override;
};

#endif  // AMDSMI_UNIFIED_SMI_NIC_H_
