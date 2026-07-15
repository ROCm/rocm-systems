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

#include "pci_util.h"
#include "cuid_device.h"
#include "cuid_device_manager.h"
#include "cuid_gpu.h"
#include "cuid_nic.h"
#include "cuid_util.h"
#include "include/amd_cuid.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>

// Endianness conversion functions
uint16_t PciUtil::le16_to_be16(uint16_t value) {
  return ((value & 0x00FF) << 8) | ((value & 0xFF00) >> 8);
}

uint64_t PciUtil::le64_to_be64(uint64_t value) {
  return ((value & 0x00000000000000FFULL) << 56) |
         ((value & 0x000000000000FF00ULL) << 40) |
         ((value & 0x0000000000FF0000ULL) << 24) |
         ((value & 0x00000000FF000000ULL) << 8) |
         ((value & 0x000000FF00000000ULL) >> 8) |
         ((value & 0x0000FF0000000000ULL) >> 24) |
         ((value & 0x00FF000000000000ULL) >> 40) |
         ((value & 0xFF00000000000000ULL) >> 56);
}

// This function should only work with PCI devices, which currently includes
// GPUs, NICs, and NPUs. It may later include storage devices as well.
amdcuid_status_t PciUtil::read_pci_config_space(std::string bdf,
                                                uint8_t *buffer,
                                                size_t buffer_size,
                                                uint16_t offset) {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }
  if (bdf.empty() || !buffer || buffer_size == 0)
    return AMDCUID_STATUS_INVALID_ARGUMENT;

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

  if (static_cast<std::ifstream::pos_type>(offset + buffer_size) > length) {
    config_file.close();
    return AMDCUID_STATUS_PCI_ERROR;
  }

  config_file.seekg(offset, std::ios::beg);
  if (!config_file) {
    config_file.close();
    return AMDCUID_STATUS_PCI_ERROR;
  }

  config_file.read(reinterpret_cast<char *>(buffer), buffer_size);
  if (config_file.fail() ||
      static_cast<size_t>(config_file.gcount()) != buffer_size) {
    config_file.close();
    return AMDCUID_STATUS_PCI_ERROR;
  }
  return AMDCUID_STATUS_SUCCESS;
}

// iterate capabilities list to find the relevant capability
amdcuid_status_t PciUtil::get_pci_dsn_cap_offset(std::string bdf,
                                                 uint16_t &offset) {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }
  if (bdf.empty())
    return AMDCUID_STATUS_INVALID_ARGUMENT;

  // Get the whole PCI config space header
  uint8_t config_space[4096] = {0};
  amdcuid_status_t status = read_pci_config_space(bdf, config_space, 4096, 0);
  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }

  // Device Serial Number (cap_id 0x0003) is a PCIe Extended Capability.
  const uint16_t dsn_cap_id = 0x0003; // PCIe DSN capability ID

  // Traverse the extended capability list starting at offset 0x100.
  uint16_t cap_ptr = 0x100;
  for (size_t hops = 0; hops < 1024; ++hops) {
    if (cap_ptr < 0x100 || cap_ptr + 3 >= sizeof(config_space)) {
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

amdcuid_status_t PciUtil::get_pci_vsec_cap_offset(std::string bdf,
                                                  uint16_t &offset) {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }
  if (bdf.empty())
    return AMDCUID_STATUS_INVALID_ARGUMENT;

  // Get the whole PCI config space header
  uint8_t config_space[4096] = {0};
  amdcuid_status_t status = read_pci_config_space(bdf, config_space, 4096, 0);
  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }

  // Vendor-Specific Extended Capability (cap_id 0x000B) is a PCIe Extended
  // Capability.
  const uint16_t vsec_cap_id = 0x0b; // PCIe VSEC capability ID

  // Traverse the extended capability list starting at offset 0x100.
  uint16_t cap_ptr = 0x100;
  for (size_t hops = 0; hops < 1024; ++hops) {
    if (cap_ptr < 0x100 || cap_ptr + 7 >= sizeof(config_space)) {
      return AMDCUID_STATUS_UNSUPPORTED;
    }

    uint32_t cap_header =
        static_cast<uint32_t>(config_space[cap_ptr]) |
        (static_cast<uint32_t>(config_space[cap_ptr + 1]) << 8) |
        (static_cast<uint32_t>(config_space[cap_ptr + 2]) << 16) |
        (static_cast<uint32_t>(config_space[cap_ptr + 3]) << 24);

    uint16_t cap_id_local = static_cast<uint16_t>(cap_header & 0xFFFF);
    uint16_t next_ptr = static_cast<uint16_t>((cap_header >> 20) & 0x0FFF);

    if (cap_id_local == vsec_cap_id) {
      // check vsec header now for correct fields
      uint32_t vsec_header =
          static_cast<uint32_t>(config_space[cap_ptr + 4]) |
          (static_cast<uint32_t>(config_space[cap_ptr + 5]) << 8) |
          (static_cast<uint32_t>(config_space[cap_ptr + 6]) << 16) |
          (static_cast<uint32_t>(config_space[cap_ptr + 7]) << 24);

      // if the VSEC has the required length for our fingerprint, return the
      // offset. Since VSEC ID for serial number/id not typically defined,
      // we rely on the length field to to simply get enough bytes
      // for the fingerprint rather than checking an ID field in the VSEC
      // header. If one is defined in the future, we will add that check here as
      // well.
      uint16_t vsec_length =
          static_cast<uint16_t>((vsec_header >> 20) & 0x0FFF);
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
