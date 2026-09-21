// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "pci_util.h"

#include <unistd.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "cuid_device.h"
#include "cuid_device_manager.h"
#include "cuid_gpu.h"
#include "cuid_nic.h"
#include "cuid_util.h"
#include "include/amd_cuid.h"

// Endianness conversion functions
uint64_t PciUtil::le64_to_be64(uint64_t value) {
  return ((value & 0x00000000000000FFULL) << 56) | ((value & 0x000000000000FF00ULL) << 40) |
         ((value & 0x0000000000FF0000ULL) << 24) | ((value & 0x00000000FF000000ULL) << 8) |
         ((value & 0x000000FF00000000ULL) >> 8) | ((value & 0x0000FF0000000000ULL) >> 24) |
         ((value & 0x00FF000000000000ULL) >> 40) | ((value & 0xFF00000000000000ULL) >> 56);
}

// This function should only work with PCI devices, which currently includes
// GPUs, NICs, and NPUs. It may later include storage devices as well.
amdcuid_status_t PciUtil::read_pci_config_space(std::string bdf, uint8_t* buffer,
                                                size_t buffer_size, uint16_t offset) {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }
  if (bdf.empty() || !buffer || buffer_size == 0) return AMDCUID_STATUS_INVALID_ARGUMENT;

  std::string pci_config_path = "/sys/bus/pci/devices/" + bdf + "/config";

  // Read the PCI config space
  std::ifstream config_file(pci_config_path, std::ios::binary);
  if (!config_file) {
    config_file.close();
    return AMDCUID_STATUS_PCI_ERROR;
  }

  config_file.seekg(0, std::ios::end);
  std::ifstream::pos_type length = config_file.tellg();
  if (length < 0) {
    config_file.close();
    return AMDCUID_STATUS_PCI_ERROR;
  }
  config_file.seekg(0, std::ios::beg);
  if (!config_file) {
    config_file.close();
    return AMDCUID_STATUS_PCI_ERROR;
  }

  // Written so it cannot wrap: offset + buffer_size would overflow size_t for
  // a large buffer_size and let an out-of-range read through.
  const size_t file_size = static_cast<size_t>(length);
  if (buffer_size > file_size || offset > file_size - buffer_size) {
    config_file.close();
    return AMDCUID_STATUS_PCI_ERROR;
  }

  config_file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!config_file) {
    config_file.close();
    return AMDCUID_STATUS_PCI_ERROR;
  }

  config_file.read(reinterpret_cast<char*>(buffer), static_cast<std::streamsize>(buffer_size));
  if (config_file.fail() || static_cast<size_t>(config_file.gcount()) != buffer_size) {
    config_file.close();
    return AMDCUID_STATUS_PCI_ERROR;
  }
  return AMDCUID_STATUS_SUCCESS;
}

// iterate capabilities list to find the relevant capability
amdcuid_status_t PciUtil::get_pci_dsn_cap_offset(std::string bdf, uint16_t& offset) {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }
  if (bdf.empty()) return AMDCUID_STATUS_INVALID_ARGUMENT;

  // Get the whole PCI config space header
  uint8_t config_space[kPciConfigSpaceSize] = {0};
  amdcuid_status_t status = read_pci_config_space(bdf, config_space, sizeof(config_space), 0);
  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }

  // Device Serial Number (cap_id 0x0003) is a PCIe Extended Capability.
  const uint16_t dsn_cap_id = 0x0003;  // PCIe DSN capability ID

  // Traverse the extended capability list starting at offset 0x100.
  uint16_t cap_ptr = 0x100;
  for (size_t hops = 0; hops < 1024; ++hops) {
    // Widen to size_t before comparing against sizeof(): cap_ptr promotes to
    // int, which would otherwise be an implicit signed/unsigned comparison.
    if (cap_ptr < 0x100 || static_cast<size_t>(cap_ptr) + 3 >= sizeof(config_space)) {
      return AMDCUID_STATUS_UNSUPPORTED;
    }

    uint32_t header = static_cast<uint32_t>(config_space[cap_ptr]) |
                      (static_cast<uint32_t>(config_space[cap_ptr + 1]) << 8) |
                      (static_cast<uint32_t>(config_space[cap_ptr + 2]) << 16) |
                      (static_cast<uint32_t>(config_space[cap_ptr + 3]) << 24);

    uint16_t cap_id_local = static_cast<uint16_t>(header & 0xFFFF);
    uint16_t next_ptr = static_cast<uint16_t>((header >> 20) & 0x0FFF);

    if (cap_id_local == static_cast<uint16_t>(dsn_cap_id)) {
      offset = cap_ptr + 4;
      return AMDCUID_STATUS_SUCCESS;
    }

    // End of list.
    if (next_ptr == 0) {
      break;
    }

    // Guard against malformed/cyclic lists.
    if (next_ptr == cap_ptr) {
      break;
    }

    cap_ptr = next_ptr;
  }

  return AMDCUID_STATUS_UNSUPPORTED;
}

amdcuid_status_t PciUtil::get_pci_vsec_cap_offset(std::string bdf, uint16_t& offset) {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }
  if (bdf.empty()) return AMDCUID_STATUS_INVALID_ARGUMENT;

  // Get the whole PCI config space header
  uint8_t config_space[kPciConfigSpaceSize] = {0};
  amdcuid_status_t status = read_pci_config_space(bdf, config_space, sizeof(config_space), 0);
  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }

  // Vendor-Specific Extended Capability (cap_id 0x000B) is a PCIe Extended
  // Capability.
  const uint16_t vsec_cap_id = 0x0b;  // PCIe VSEC capability ID

  // Traverse the extended capability list starting at offset 0x100.
  uint16_t cap_ptr = 0x100;
  for (size_t hops = 0; hops < 1024; ++hops) {
    // Widen to size_t before comparing against sizeof(): cap_ptr promotes to
    // int, which would otherwise be an implicit signed/unsigned comparison.
    if (cap_ptr < 0x100 || static_cast<size_t>(cap_ptr) + 7 >= sizeof(config_space)) {
      return AMDCUID_STATUS_UNSUPPORTED;
    }

    uint32_t cap_header = static_cast<uint32_t>(config_space[cap_ptr]) |
                          (static_cast<uint32_t>(config_space[cap_ptr + 1]) << 8) |
                          (static_cast<uint32_t>(config_space[cap_ptr + 2]) << 16) |
                          (static_cast<uint32_t>(config_space[cap_ptr + 3]) << 24);

    uint16_t cap_id_local = static_cast<uint16_t>(cap_header & 0xFFFF);
    uint16_t next_ptr = static_cast<uint16_t>((cap_header >> 20) & 0x0FFF);

    if (cap_id_local == vsec_cap_id) {
      // check vsec header now for correct fields
      uint32_t vsec_header = static_cast<uint32_t>(config_space[cap_ptr + 4]) |
                             (static_cast<uint32_t>(config_space[cap_ptr + 5]) << 8) |
                             (static_cast<uint32_t>(config_space[cap_ptr + 6]) << 16) |
                             (static_cast<uint32_t>(config_space[cap_ptr + 7]) << 24);

      // if the VSEC has the required length for our fingerprint, return the
      // offset. Since VSEC ID for serial number/id not typically defined,
      // we rely on the length field to to simply get enough bytes
      // for the fingerprint rather than checking an ID field in the VSEC
      // header. If one is defined in the future, we will add that check here as
      // well.
      uint16_t vsec_length = static_cast<uint16_t>((vsec_header >> 20) & 0x0FFF);
      if (vsec_length >= 8) {
        offset = cap_ptr + 8;
        return AMDCUID_STATUS_SUCCESS;
      }
    }

    // Guard against malformed/cyclic lists or end of list.
    if (next_ptr == 0 || next_ptr == cap_ptr) {
      break;
    }

    cap_ptr = next_ptr;
  }

  return AMDCUID_STATUS_UNSUPPORTED;
}
