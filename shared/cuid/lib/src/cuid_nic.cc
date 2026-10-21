// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cuid_nic.h"

#include <dirent.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "cuid_util.h"
#include "pci_util.h"

CuidNic::CuidNic(const amdcuid_nic_info& i) : m_info(i) {}

amdcuid_status_t CuidNic::discover(std::vector<DevicePtr>& nics) {
  return discover("/sys/class/net", nics);
}

amdcuid_status_t CuidNic::discover(const std::string& nic_base_path, std::vector<DevicePtr>& nics) {
  DIR* dir = opendir(nic_base_path.c_str());
  if (!dir) return AMDCUID_STATUS_DEVICE_NOT_FOUND;

  // One PCI function can carry several interfaces (switchdev representors,
  // multi-port functions), all with one BDF and so one CUID. Keep one per BDF,
  // the first by name, so the choice does not depend on readdir order.
  std::map<std::string, std::string> by_bdf;
  struct dirent* entry;
  // This call site owns its DIR*, which is all POSIX requires for readdir to be
  // safe; readdir_r is deprecated and must not be adopted.
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  while ((entry = readdir(dir)) != NULL) {
    if (entry->d_name[0] != '.') {
      std::string device_path = nic_base_path + "/" + entry->d_name + "/device";
      // lo and other virtual interfaces have no PCI function behind them, so
      // the device link, not the name, is what excludes them.
      const std::string bdf = CuidUtilities::readlink_bdf(device_path);
      if (bdf.empty()) {
        continue;
      }
      auto found = by_bdf.emplace(bdf, device_path).first;
      if (device_path < found->second) found->second = device_path;
    }
  }
  closedir(dir);
  for (const auto& nic : by_bdf) {
    amdcuid_nic_info info = {};
    const auto status = discover_single(&info, nic.second);
    if (status != AMDCUID_STATUS_SUCCESS) return status;
    nics.emplace_back(std::make_shared<CuidNic>(info));
  }
  if (nics.size() == 0) return AMDCUID_STATUS_DEVICE_NOT_FOUND;

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::discover_single(amdcuid_nic_info* nic_info,
                                          const std::string& device_path) {
  amdcuid_nic_info info = {};
  info.header.device_type = AMDCUID_DEVICE_TYPE_NIC;

  std::string bdf = CuidUtilities::readlink_bdf(device_path);

  const auto identity_status =
      PciUtil::read_identity(device_path, bdf, info.header.fields.nic.vendor_id,
                             info.header.fields.nic.device_id, info.header.fields.nic.revision_id);
  if (identity_status != AMDCUID_STATUS_SUCCESS) return identity_status;

  std::string pci_class = CuidUtilities::read_sysfs_file(device_path + "/class");
  uint16_t pci_class_integer = 0;
  if (pci_class.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    uint8_t class_id_bytes[2] = {0};
    const uint16_t offset = 0xa;
    amdcuid_status_t status = PciUtil::read_pci_config_space(bdf, class_id_bytes, 2, offset);
    uint16_t class_id_int = PciUtil::load_le16(class_id_bytes);
    pci_class_integer = (status == AMDCUID_STATUS_SUCCESS) ? class_id_int : 0;
  } else {
    // sysfs class file returns 24-bit value (class:subclass:prog_if), shift
    // right by 8 to get 16-bit class:subclass
    pci_class_integer = (uint16_t)(strtol(pci_class.c_str(), nullptr, 16) >> 8);
  }
  info.header.fields.nic.pci_class = pci_class_integer;

  info.bdf = bdf;
  std::string full_device_node;
  size_t last_slash = device_path.rfind('/');
  if (last_slash != std::string::npos && last_slash > 0) {
    full_device_node = device_path.substr(0, last_slash);
  } else {
    full_device_node = device_path;
  }
  info.network_interface = full_device_node;
  *nic_info = info;

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_hardware_fingerprint(uint64_t& fingerprint) const {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }

  uint16_t offset = 0;
  amdcuid_status_t status = PciUtil::get_pci_dsn_cap_offset(m_info.bdf, offset);
  if (status != AMDCUID_STATUS_SUCCESS) {
    // attempt to get fingerprint through VSEC fallback if DSN capability is not
    // found
    status =
        PciUtil::get_pci_vsec_cap_offset(m_info.bdf, m_info.header.fields.nic.vendor_id, offset);
  }
  if (status == AMDCUID_STATUS_SUCCESS) {
    const uint8_t fingerprint_size = 8;
    uint8_t fingerprint_bytes[fingerprint_size] = {0};
    status =
        PciUtil::read_pci_config_space(m_info.bdf, fingerprint_bytes, fingerprint_size, offset);
    if (status == AMDCUID_STATUS_SUCCESS) {
      fingerprint = PciUtil::load_le64(fingerprint_bytes);
      if (CuidUtilities::validate_fingerprint(fingerprint) == AMDCUID_STATUS_SUCCESS) {
        return AMDCUID_STATUS_SUCCESS;
      }
      // An all-zero DSN means the capability is present but unprogrammed. Fall
      // through and try the MAC address instead of claiming identity zero.
      fingerprint = 0;
    }
  }
  // TODO: serial ID could be found in FRU_ID so should attempt to look there
  // pci config space file does not exist or read failed, so create fingerprint
  // from MAC address
  std::string mac_address;
  status = get_mac_address(mac_address);
  if (status != AMDCUID_STATUS_SUCCESS) {
    // No DSN, no VSEC and no MAC leaves the device with no serial to report,
    // which is the auxiliary case. Any other failure is the caller's problem
    // and is returned, so a device is never called auxiliary on a bad read.
    if (status == AMDCUID_STATUS_UNSUPPORTED) {
      return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
    }
    return status;
  }
  return fingerprint_from_mac(mac_address, fingerprint);
}

amdcuid_status_t CuidNic::fingerprint_from_mac(const std::string& mac_address,
                                               uint64_t& fingerprint) {
  if (!mac_address.empty()) {
    // convert MAC address string to bytes
    uint8_t mac_bytes[6] = {0};
    // An unchecked sscanf leaves mac_bytes partly or wholly zero on a
    // malformed address, and the caller would then derive a CUID from that
    // silently. All six octets must convert.
    // cert-err34-c wants strtoul; the concern it encodes is a conversion that
    // fails silently, which the count check below rules out.
    // NOLINTNEXTLINE(cert-err34-c)
    const int converted = sscanf(  // NOLINT(cert-err34-c)
        mac_address.c_str(), "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx", &mac_bytes[0], &mac_bytes[1],
        &mac_bytes[2], &mac_bytes[3], &mac_bytes[4], &mac_bytes[5]);
    if (converted != 6) {
      fingerprint = 0;
      return AMDCUID_STATUS_INVALID_FORMAT;
    }
    // Octet 0 of the address occupies payload bits 0:7, the same orientation
    // as every other little-endian field, and not the host's byte order.
    uint64_t mac_fingerprint = 0;
    for (size_t i = 0; i < sizeof(mac_bytes); ++i) {
      mac_fingerprint |= static_cast<uint64_t>(mac_bytes[i]) << (8 * i);
    }
    fingerprint = mac_fingerprint;
    // An all-zero MAC is an unconfigured interface, not an identity.
    return CuidUtilities::validate_fingerprint(fingerprint);
  }

  return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
}

amdcuid_status_t CuidNic::get_primary_cuid(amdcuid_primary_id& id) const {
  // A device with no serial has no primary (HW_FINGERPRINT_NOT_FOUND) and
  // takes its temporary CUID; a serial this caller may not read is returned as
  // that error, since the identity a device presents must not depend on the
  // privilege of the caller asking.
  uint64_t fingerprint = 0;
  amdcuid_status_t status = get_hardware_fingerprint(fingerprint);
  if (status != AMDCUID_STATUS_SUCCESS) return status;

  uint16_t unit_id = 0;
  get_unit_id(unit_id);
  status = CuidUtilities::generate_primary_cuid(
      fingerprint, unit_id, m_info.header.fields.nic.revision_id,
      m_info.header.fields.nic.device_id, m_info.header.fields.nic.vendor_id,
      AMDCUID_DEVICE_TYPE_NIC, &id, false);
  if (status != AMDCUID_STATUS_SUCCESS) std::memset(&id, 0, sizeof(id));
  return status;
}

amdcuid_status_t CuidNic::get_temporary_input(CuidUtilities::AuxiliaryInput& aux) const {
  std::string bdf;
  const amdcuid_status_t status = this->get_bdf(bdf);
  if (status != AMDCUID_STATUS_SUCCESS) return status;

  aux = {};
  aux.format = CuidUtilities::kAuxFormatPcie;
  CuidUtilities::set_pcie_routing(aux, bdf);
  aux.revision_id = m_info.header.fields.nic.revision_id;
  aux.device_id = m_info.header.fields.nic.device_id;
  aux.vendor_id = m_info.header.fields.nic.vendor_id;
  aux.component_type = static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_NIC);
  uint16_t unit_id = 0;
  get_unit_id(unit_id);
  aux.unit_id = unit_id;
  return AMDCUID_STATUS_SUCCESS;
}

const amdcuid_nic_info& CuidNic::get_info() const { return m_info; }

amdcuid_status_t CuidNic::get_vendor_id(uint16_t& vendor_id) const {
  vendor_id = m_info.header.fields.nic.vendor_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_device_id(uint16_t& device_id) const {
  device_id = m_info.header.fields.nic.device_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_pci_class(uint16_t& pci_class) const {
  pci_class = m_info.header.fields.nic.pci_class;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_revision_id(uint8_t& revision_id) const {
  revision_id = m_info.header.fields.nic.revision_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_unit_id(uint16_t& unit_id) const {
  const std::string& bdf = m_info.bdf;
  const bool has_function =
      bdf.size() >= 2 && bdf[bdf.size() - 2] == '.' && bdf.back() >= '0' && bdf.back() <= '7';
  unit_id = has_function ? static_cast<uint16_t>(bdf.back() - '0') : 0;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_bdf(std::string& bdf) const {
  if (m_info.bdf.empty()) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  bdf = m_info.bdf;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_device_path(std::string& path) const {
  if (m_info.network_interface.empty()) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  path = m_info.network_interface;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_mac_address(std::string& mac_address) const {
  // Extract interface name from the sysfs path (e.g. /sys/class/net/eth0 ->
  // eth0)
  std::string iface = m_info.network_interface;
  size_t last_slash = iface.rfind('/');
  if (last_slash != std::string::npos) iface = iface.substr(last_slash + 1);

  if (iface.empty() || iface.size() >= IFNAMSIZ) return AMDCUID_STATUS_UNSUPPORTED;

  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) return AMDCUID_STATUS_UNSUPPORTED;

  struct ethtool_perm_addr* epaddr = reinterpret_cast<struct ethtool_perm_addr*>(
      new uint8_t[sizeof(struct ethtool_perm_addr) + ETH_ALEN]);
  epaddr->cmd = ETHTOOL_GPERMADDR;
  epaddr->size = ETH_ALEN;

  struct ifreq ifr = {};
  strncpy(ifr.ifr_name, iface.c_str(), IFNAMSIZ - 1);
  ifr.ifr_data = reinterpret_cast<char*>(epaddr);

  if (ioctl(fd, SIOCETHTOOL, &ifr) < 0) {
    close(fd);
    delete[] reinterpret_cast<uint8_t*>(epaddr);
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  close(fd);

  // Six %02x plus five separators is exactly 17 characters plus the NUL.
  char buf[18];
  // NOLINTNEXTLINE(cert-err33-c)
  snprintf(buf, sizeof(buf), "%02x:%02x:%02x:%02x:%02x:%02x", epaddr->data[0], epaddr->data[1],
           epaddr->data[2], epaddr->data[3], epaddr->data[4], epaddr->data[5]);
  delete[] reinterpret_cast<uint8_t*>(epaddr);

  mac_address = buf;
  return AMDCUID_STATUS_SUCCESS;
}
