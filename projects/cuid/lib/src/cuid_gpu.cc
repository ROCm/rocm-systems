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

#include "cuid_gpu.h"
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

CuidGpu::CuidGpu(const amdcuid_gpu_info &i) : m_info(i) {}

amdcuid_status_t CuidGpu::discover(std::vector<DevicePtr> &gpus) {
  const char *drm_path = "/sys/class/drm";
  DIR *dir = opendir(drm_path);
  if (!dir)
    return AMDCUID_STATUS_UNSUPPORTED;

  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    if (strncmp(entry->d_name, "renderD", 7) == 0 &&
        isdigit(entry->d_name[7])) {
      std::string render_name(entry->d_name);
      std::string device_path =
          std::string(drm_path) + "/" + render_name + "/device";
      amdcuid_gpu_info info = {};
      discover_single(&info, device_path);

      gpus.emplace_back(std::make_shared<CuidGpu>(info));
    }
  }
  closedir(dir);
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::discover_single(amdcuid_gpu_info *gpu_info,
                                          const std::string &device_path) {

  amdcuid_gpu_info info = {};
  std::string bdf = CuidUtilities::readlink_bdf(device_path);

  // Determine unit_id from SR-IOV VF (Virtual Function) status via ioctl.
  // In bare metal or passthrough, unit_id is 0.
  // For VFs, unit_id is the 1-based VF index.
  // Falls back to 0 if VF detection is unavailable.
  info.header.fields.gpu.unit_id = CuidUtilities::get_gpu_vf_id(device_path);

  std::string vendor = CuidUtilities::read_sysfs_file(device_path + "/vendor");
  if (vendor.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    uint8_t vendor_id_bytes[2] = {0};
    const uint16_t offset = 0x0;
    amdcuid_status_t status =
        PciUtil::read_pci_config_space(bdf, vendor_id_bytes, 2, offset);
    uint16_t vendor_id_int =
        PciUtil::le16_to_be16(*reinterpret_cast<uint16_t *>(vendor_id_bytes));
    info.header.fields.gpu.vendor_id =
        (status == AMDCUID_STATUS_SUCCESS) ? vendor_id_int : 0;
  } else {
    info.header.fields.gpu.vendor_id =
        (uint16_t)strtol(vendor.c_str(), nullptr, 16);
  }

  std::string device = CuidUtilities::read_sysfs_file(device_path + "/device");
  if (device.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    uint8_t device_id_bytes[2] = {0};
    const uint16_t offset = 0x2;
    amdcuid_status_t status =
        PciUtil::read_pci_config_space(bdf, device_id_bytes, 2, offset);
    uint16_t device_id_int =
        PciUtil::le16_to_be16(*reinterpret_cast<uint16_t *>(device_id_bytes));
    info.header.fields.gpu.device_id =
        (status == AMDCUID_STATUS_SUCCESS) ? device_id_int : 0;
  } else {
    info.header.fields.gpu.device_id =
        (uint16_t)strtol(device.c_str(), nullptr, 16);
  }

  std::string pci_class =
      CuidUtilities::read_sysfs_file(device_path + "/class");
  uint16_t pci_class_integer = 0;
  if (pci_class.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    uint8_t class_id_bytes[2] = {0};
    const uint16_t offset = 0xa;
    amdcuid_status_t status =
        PciUtil::read_pci_config_space(bdf, class_id_bytes, 2, offset);
    uint16_t class_id_int =
        PciUtil::le16_to_be16(*reinterpret_cast<uint16_t *>(class_id_bytes));
    pci_class_integer = (status == AMDCUID_STATUS_SUCCESS) ? class_id_int : 0;
  } else {
    // sysfs class file returns 24-bit value (class:subclass:prog_if), shift
    // right by 8 to get 16-bit class:subclass
    pci_class_integer = (uint16_t)(strtol(pci_class.c_str(), nullptr, 16) >> 8);
  }
  info.header.fields.gpu.pci_class = pci_class_integer;

  std::string revision_id =
      CuidUtilities::read_sysfs_file(device_path + "/revision");
  if (revision_id.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    uint8_t revision_id_bytes[2] = {0};
    const uint16_t offset = 0x8;
    amdcuid_status_t status =
        PciUtil::read_pci_config_space(bdf, revision_id_bytes, 2, offset);
    uint16_t revision_id_int =
        PciUtil::le16_to_be16(*reinterpret_cast<uint16_t *>(revision_id_bytes));
    info.header.fields.gpu.revision_id =
        (status == AMDCUID_STATUS_SUCCESS) ? revision_id_int : 0;
  } else {
    info.header.fields.gpu.revision_id =
        (uint16_t)strtol(revision_id.c_str(), nullptr, 16);
  }

  // we use the original device path to get render node
  std::string full_device_node;
  size_t last_slash = device_path.rfind('/');
  if (last_slash != std::string::npos && last_slash > 0) {
    // Trim to just /sys/class/drm/renderDXXX;
    full_device_node = device_path.substr(0, last_slash);
  } else {
    full_device_node = device_path;
  }

  info.header.device_type = AMDCUID_DEVICE_TYPE_GPU;
  info.bdf = bdf;
  info.render_node = full_device_node;

  *gpu_info = info;

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t
CuidGpu::get_hardware_fingerprint(uint64_t &fingerprint) const {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }

  std::string unique_id_path = m_info.render_node + "/device/unique_id";

  // Try to read the unique_id from the device sysfs file
  std::ifstream fin(unique_id_path);

  // If not available and this is a VF, try the PF's unique_id via physfn
  if (!fin.is_open() && m_info.header.fields.gpu.unit_id != 0) {
    std::string physfn_path = m_info.render_node + "/device/physfn/unique_id";
    fin.open(physfn_path);
  }
  if (fin.is_open()) {
    std::string hex_str;
    std::getline(fin, hex_str);
    fin.close();
    if (hex_str.empty()) {
      fingerprint = 0;
      return AMDCUID_STATUS_UNSUPPORTED;
    }
    // Parse as 64-bit hex value (if possible)
    try {
      fingerprint = std::stoull(hex_str, nullptr, 16);
    } catch (...) {
      fingerprint = 0;
      return AMDCUID_STATUS_UNSUPPORTED;
    }
  } else if (m_info.header.fields.gpu.unit_id == 0) {
    // attempt to get fingerprint through PCI Config Space if not a partition
    uint16_t offset = 0;
    amdcuid_status_t status =
        PciUtil::get_pci_cap_offset(m_info.bdf, 0x03, offset);
    if (status != AMDCUID_STATUS_SUCCESS) {
      return status;
    }

    uint8_t fingerprint_size = 8;
    uint8_t *fingerprint_buffer = new uint8_t[fingerprint_size];
    status = PciUtil::read_pci_config_space(m_info.bdf, fingerprint_buffer,
                                            fingerprint_size, offset);
    if (status != AMDCUID_STATUS_SUCCESS) {
      fingerprint = 0;
      delete[] fingerprint_buffer;
      return status;
    }
    // pcie config file is little endian, so need to convert to big endian
    fingerprint = PciUtil::le64_to_be64(
        *reinterpret_cast<uint64_t *>(fingerprint_buffer));
    delete[] fingerprint_buffer;
  } else {
    // partitioned device without unique_id file cannot get fingerprint
    fingerprint = 0;
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_primary_cuid(amdcuid_primary_id &id) const {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }

  // attempt to read the CUID from the file first
  std::string cuid_file_path = CuidUtilities::priv_cuid_file();
  CuidFile primary_file(cuid_file_path, false);
  primary_file.load();
  std::vector<CuidFileEntry> entries = primary_file.get_entries();

  CuidFileEntry entry;
  amdcuid_status_t status =
      primary_file.find_by_device_node(m_info.render_node, entry);
  if (status == AMDCUID_STATUS_SUCCESS) {
    id.UUIDv8_representation = entry.primary_cuid;
    CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, id.raw_bits);
    return AMDCUID_STATUS_SUCCESS;
  }

  // primary CUID not found in file so generate it
  uint64_t fingerprint = 0;
  status = get_hardware_fingerprint(fingerprint);
  if (status != AMDCUID_STATUS_SUCCESS) {
    std::memset(&id, 0, sizeof(id));
    return status;
  }
  // Use header fields for the rest
  amdcuid_primary_id result = {};
  const auto &h = m_info.header;
  CuidUtilities::generate_primary_cuid(
      fingerprint, h.fields.gpu.unit_id, h.fields.gpu.revision_id,
      h.fields.gpu.device_id, h.fields.gpu.vendor_id,
      static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_GPU), &result);

  id = result;
  return AMDCUID_STATUS_SUCCESS;
}

const amdcuid_gpu_info &CuidGpu::get_info() const { return m_info; }

amdcuid_status_t CuidGpu::get_vendor_id(uint16_t &vendor_id) const {
  vendor_id = m_info.header.fields.gpu.vendor_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_device_id(uint16_t &device_id) const {
  device_id = m_info.header.fields.gpu.device_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_pci_class(uint16_t &pci_class) const {
  pci_class = m_info.header.fields.gpu.pci_class;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_revision_id(uint8_t &revision_id) const {
  revision_id = m_info.header.fields.gpu.revision_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_unit_id(uint16_t &unit_id) const {
  unit_id = m_info.header.fields.gpu.unit_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_bdf(std::string &bdf) const {
  if (m_info.bdf.empty()) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  bdf = m_info.bdf;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_device_path(std::string &path) const {
  if (m_info.render_node.empty()) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  path = m_info.render_node;
  return AMDCUID_STATUS_SUCCESS;
}
