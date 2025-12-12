#ifndef PCI_UTIL_H
#define PCI_UTIL_H

#include "cuid.h"
#include <cstdint>
#include <string>
#include <vector>

class PciUtil {
public:
    static amdcuid_status_t read_pci_config_space(std::string bdf, uint8_t *buffer, size_t buffer_size, uint16_t offset);
    static amdcuid_status_t get_pci_bdf_from_handle(amdcuid_handle handle, std::string &bdf);
    static amdcuid_status_t get_pci_cap_offset(std::string bdf, uint32_t cap_id, uint16_t &offset);

    // Endianness conversion utilities
    static uint16_t le16_to_be16(uint16_t value);
    static uint32_t le32_to_be32(uint32_t value);
    static uint64_t le64_to_be64(uint64_t value);
};

#endif // PCI_UTIL_H