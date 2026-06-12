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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "cuid_npu.h"
#include "cuid_file.h"
#include "cuid_util.h"
#include "pci_util.h"
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <unistd.h>

// AMD vendor ID used to filter for AMD NPU devices
static const uint16_t AMD_VENDOR_ID = 0x1022;

// PCI base classes that AMD NPUs may present as:
//   0x11 = Signal processing controller (e.g., Strix/Strix Halo NPU 0x1180)
//   0x12 = Processing accelerators
static const uint16_t PCI_CLASS_SIGNAL_PROCESSING = 0x1100;
static const uint16_t PCI_CLASS_PROCESSING_ACCEL = 0x1200;
static const uint16_t PCI_CLASS_BASE_MASK = 0xFF00;

CuidNpu::CuidNpu(const amdcuid_npu_info &i) : m_info(i) {}

// Helper to check if a /sys/class/accel entry name is an accel device
// (e.g., "accel0", "accel1").
static bool is_accel_entry(const char *name) {
  if (strncmp(name, "accel", 5) != 0 || !isdigit(name[5]))
    return false;
  for (size_t i = 5; name[i] != '\0'; ++i) {
    if (!isdigit(name[i]))
      return false;
  }
  return true;
}

// Discover NPU devices via /sys/class/accel (when amdxdna driver is loaded)
static amdcuid_status_t discover_via_accel(std::vector<DevicePtr> &npus) {
  const char *accel_path = "/sys/class/accel";
  DIR *dir = opendir(accel_path);
  if (!dir)
    return AMDCUID_STATUS_UNSUPPORTED;

  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    if (is_accel_entry(entry->d_name)) {
      std::string accel_name(entry->d_name);
      std::string device_path =
          std::string(accel_path) + "/" + accel_name + "/device";

      // Read vendor to filter for AMD NPU devices only
      std::string vendor_str =
          CuidUtilities::read_sysfs_file(device_path + "/vendor");
      if (vendor_str.empty())
        continue;
      uint16_t vendor_id =
          static_cast<uint16_t>(strtol(vendor_str.c_str(), nullptr, 16));
      if (vendor_id != AMD_VENDOR_ID)
        continue;

      amdcuid_npu_info info = {};
      amdcuid_status_t status = CuidNpu::discover_single(&info, device_path);
      if (status != AMDCUID_STATUS_SUCCESS)
        continue;

      npus.emplace_back(std::make_shared<CuidNpu>(info));
    }
  }
  closedir(dir);

  return npus.empty() ? AMDCUID_STATUS_UNSUPPORTED : AMDCUID_STATUS_SUCCESS;
}

// Fallback: discover NPU devices by scanning PCI bus for AMD processing
// accelerator class devices. This handles systems where /sys/class/accel
// is not populated (e.g., driver not loaded or accel subsystem absent).
static amdcuid_status_t discover_via_pci_bus(std::vector<DevicePtr> &npus) {
  const char *pci_path = "/sys/bus/pci/devices";
  DIR *dir = opendir(pci_path);
  if (!dir)
    return AMDCUID_STATUS_UNSUPPORTED;

  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    if (entry->d_name[0] == '.')
      continue;

    std::string device_path = std::string(pci_path) + "/" + entry->d_name;

    std::string vendor_str =
        CuidUtilities::read_sysfs_file(device_path + "/vendor");
    if (vendor_str.empty())
      continue;
    uint16_t vendor_id =
        static_cast<uint16_t>(strtol(vendor_str.c_str(), nullptr, 16));
    if (vendor_id != AMD_VENDOR_ID)
      continue;

    std::string class_str =
        CuidUtilities::read_sysfs_file(device_path + "/class");
    if (class_str.empty())
      continue;
    // sysfs class is 24-bit (class:subclass:prog_if), shift to get
    // class:subclass
    uint16_t pci_class =
        static_cast<uint16_t>(strtol(class_str.c_str(), nullptr, 16) >> 8);
    uint16_t base_class = pci_class & PCI_CLASS_BASE_MASK;
    if (base_class != PCI_CLASS_PROCESSING_ACCEL &&
        base_class != PCI_CLASS_SIGNAL_PROCESSING)
      continue;

    amdcuid_npu_info info = {};
    amdcuid_status_t status = CuidNpu::discover_single(&info, device_path);
    if (status != AMDCUID_STATUS_SUCCESS)
      continue;

    npus.emplace_back(std::make_shared<CuidNpu>(info));
  }
  closedir(dir);

  return npus.empty() ? AMDCUID_STATUS_UNSUPPORTED : AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNpu::discover(std::vector<DevicePtr> &npus) {
  // Try /sys/class/accel first (when amdxdna driver is loaded)
  amdcuid_status_t status = discover_via_accel(npus);
  if (status == AMDCUID_STATUS_SUCCESS)
    return status;

  // Fallback: scan PCI bus for processing accelerator class devices
  return discover_via_pci_bus(npus);
}

amdcuid_status_t CuidNpu::discover_single(amdcuid_npu_info *npu_info,
                                          const std::string &device_path) {
  amdcuid_npu_info info = {};
  std::string bdf = CuidUtilities::readlink_bdf(device_path);

  std::string vendor = CuidUtilities::read_sysfs_file(device_path + "/vendor");
  if (vendor.empty() && !bdf.empty()) {
    uint8_t vendor_id_bytes[2] = {0};
    const uint16_t offset = 0x0;
    amdcuid_status_t status =
        PciUtil::read_pci_config_space(bdf, vendor_id_bytes, 2, offset);
    uint16_t vendor_id_int =
        PciUtil::le16_to_be16(*reinterpret_cast<uint16_t *>(vendor_id_bytes));
    info.header.fields.npu.vendor_id =
        (status == AMDCUID_STATUS_SUCCESS) ? vendor_id_int : 0;
  } else {
    info.header.fields.npu.vendor_id =
        static_cast<uint16_t>(strtol(vendor.c_str(), nullptr, 16));
  }

  std::string device = CuidUtilities::read_sysfs_file(device_path + "/device");
  if (device.empty() && !bdf.empty()) {
    uint8_t device_id_bytes[2] = {0};
    const uint16_t offset = 0x2;
    amdcuid_status_t status =
        PciUtil::read_pci_config_space(bdf, device_id_bytes, 2, offset);
    uint16_t device_id_int =
        PciUtil::le16_to_be16(*reinterpret_cast<uint16_t *>(device_id_bytes));
    info.header.fields.npu.device_id =
        (status == AMDCUID_STATUS_SUCCESS) ? device_id_int : 0;
  } else {
    info.header.fields.npu.device_id =
        static_cast<uint16_t>(strtol(device.c_str(), nullptr, 16));
  }

  std::string pci_class =
      CuidUtilities::read_sysfs_file(device_path + "/class");
  uint16_t pci_class_integer = 0;
  if (pci_class.empty() && !bdf.empty()) {
    uint8_t class_id_bytes[2] = {0};
    const uint16_t offset = 0xa;
    amdcuid_status_t status =
        PciUtil::read_pci_config_space(bdf, class_id_bytes, 2, offset);
    uint16_t class_id_int =
        PciUtil::le16_to_be16(*reinterpret_cast<uint16_t *>(class_id_bytes));
    pci_class_integer = (status == AMDCUID_STATUS_SUCCESS) ? class_id_int : 0;
  } else {
    // sysfs class file returns 24-bit value (class:subclass:prog_if),
    // shift right by 8 to get 16-bit class:subclass
    pci_class_integer =
        static_cast<uint16_t>(strtol(pci_class.c_str(), nullptr, 16) >> 8);
  }
  info.header.fields.npu.pci_class = pci_class_integer;

  std::string revision_id =
      CuidUtilities::read_sysfs_file(device_path + "/revision");
  if (revision_id.empty() && !bdf.empty()) {
    uint8_t revision_id_bytes[2] = {0};
    const uint16_t offset = 0x8;
    amdcuid_status_t status =
        PciUtil::read_pci_config_space(bdf, revision_id_bytes, 2, offset);
    uint16_t revision_id_int =
        PciUtil::le16_to_be16(*reinterpret_cast<uint16_t *>(revision_id_bytes));
    info.header.fields.npu.revision_id =
        (status == AMDCUID_STATUS_SUCCESS)
            ? static_cast<uint8_t>(revision_id_int)
            : 0;
  } else {
    info.header.fields.npu.revision_id =
        static_cast<uint8_t>(strtol(revision_id.c_str(), nullptr, 16));
  }

  // Determine the device node path.
  // If discovered through /sys/class/accel, strip the /device suffix.
  // If discovered through /sys/bus/pci/devices, try to resolve the accel
  // node; if that fails, store the PCI device path as-is.
  std::string full_device_node;
  if (device_path.find("/sys/class/accel/") != std::string::npos) {
    size_t last_slash = device_path.rfind('/');
    if (last_slash != std::string::npos && last_slash > 0) {
      full_device_node = device_path.substr(0, last_slash);
    } else {
      full_device_node = device_path;
    }
  } else if (!bdf.empty()) {
    // Try to resolve /sys/bus/pci/devices/<bdf>/accel/accelN
    std::string resolved =
        CuidUtilities::bdf_to_device_path(bdf, AMDCUID_DEVICE_TYPE_NPU);
    if (!resolved.empty()) {
      full_device_node = resolved;
    } else {
      full_device_node = device_path;
    }
  } else {
    full_device_node = device_path;
  }

  info.header.device_type = AMDCUID_DEVICE_TYPE_NPU;
  info.bdf = bdf;
  info.accel_node = full_device_node;

  *npu_info = info;

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t
CuidNpu::get_hardware_fingerprint(uint64_t &fingerprint) const {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }

  // Attempt to get fingerprint through PCI Config Space DSN capability
  uint16_t offset = 0;
  amdcuid_status_t status = PciUtil::get_pci_dsn_cap_offset(m_info.bdf, offset);
  if (status != AMDCUID_STATUS_SUCCESS) {
    // attempt to get fingerprint through VSEC fallback if DSN capability is not
    // found
    status = PciUtil::get_pci_vsec_cap_offset(m_info.bdf, offset);
    if (status != AMDCUID_STATUS_SUCCESS) {
      fingerprint = 0;
      return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
    }
  }
  if (status == AMDCUID_STATUS_SUCCESS) {
    const uint8_t fingerprint_size = 8;
    uint8_t fingerprint_bytes[fingerprint_size] = {0};
    status = PciUtil::read_pci_config_space(m_info.bdf, fingerprint_bytes,
                                            fingerprint_size, offset);
    if (status == AMDCUID_STATUS_SUCCESS) {
      uint64_t fingerprint_value = 0;
      std::memcpy(&fingerprint_value, fingerprint_bytes, fingerprint_size);
      fingerprint = PciUtil::le64_to_be64(fingerprint_value);
      return AMDCUID_STATUS_SUCCESS;
    } else {
      fingerprint = 0;
      return status;
    }
  }

  return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
}

amdcuid_status_t CuidNpu::get_primary_cuid(amdcuid_primary_id &id) const {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }

  // Attempt to read the CUID from the file first
  std::string cuid_file_path = CuidUtilities::priv_cuid_file();
  CuidFile primary_file(cuid_file_path, false);
  primary_file.load();

  CuidFileEntry entry;
  amdcuid_status_t status =
      primary_file.find_by_device_node(m_info.accel_node, entry);
  if (status == AMDCUID_STATUS_SUCCESS) {
    id.UUIDv8_representation = entry.primary_cuid;
    CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, id.raw_bits);
    return AMDCUID_STATUS_SUCCESS;
  }

  // Primary CUID not found in file, so generate it
  uint64_t fingerprint = 0;
  status = get_hardware_fingerprint(fingerprint);
  if (status != AMDCUID_STATUS_SUCCESS) {
    std::memset(&id, 0, sizeof(id));
    return status;
  }

  status = CuidUtilities::generate_primary_cuid(
      fingerprint,
      0, // unit_id: NPUs are not partitioned
      m_info.header.fields.npu.revision_id, m_info.header.fields.npu.device_id,
      m_info.header.fields.npu.vendor_id,
      static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_NPU), &id);
  if (status != AMDCUID_STATUS_SUCCESS) {
    std::memset(&id, 0, sizeof(id));
    return status;
  }

  return status;
}

const amdcuid_npu_info &CuidNpu::get_info() const { return m_info; }

amdcuid_status_t CuidNpu::get_vendor_id(uint16_t &vendor_id) const {
  vendor_id = m_info.header.fields.npu.vendor_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNpu::get_device_id(uint16_t &device_id) const {
  device_id = m_info.header.fields.npu.device_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNpu::get_pci_class(uint16_t &pci_class) const {
  pci_class = m_info.header.fields.npu.pci_class;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNpu::get_revision_id(uint8_t &revision_id) const {
  revision_id = m_info.header.fields.npu.revision_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNpu::get_bdf(std::string &bdf) const {
  if (m_info.bdf.empty()) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  bdf = m_info.bdf;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNpu::get_device_path(std::string &path) const {
  if (m_info.accel_node.empty()) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  path = m_info.accel_node;
  return AMDCUID_STATUS_SUCCESS;
}
