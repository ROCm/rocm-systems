#ifndef CUID_SPEC_H
#define CUID_SPEC_H

#include "cuid.h"

struct amdcuid_header {
    uint16_t device_type;
};

struct amdcuid_header_cpu {
    struct amdcuid_header hdr;
    uint16_t vid;
    uint16_t family;
    uint16_t model;
    uint8_t revision;
    uint16_t core;
    uint16_t physical_id;
} __attribute__((packed));

struct amdcuid_header_gpu {
    struct amdcuid_header hdr;
    uint16_t vid;
    uint16_t did;
    uint16_t pci_class;
    uint16_t revision_id;
    uint32_t partition_info;
} __attribute__((packed));

struct amdcuid_header_nic {
    struct amdcuid_header hdr;
    uint16_t vid;
    uint16_t did;
    uint16_t pci_class;
    uint16_t revision_id;
} __attribute__((packed));

struct amdcuid_header_platform {
    struct amdcuid_header hdr;
    uint8_t system_information[14]; // System Information (Type1)
} __attribute__((packed));

/**
 * @brief Generates a secondary CUID (Component Unique Identifier) based on a given salt and primary CUID.
 *
 * This function computes a secondary CUID using the provided salt and the primary CUID.
 * It is typically used to derive a unique identifier for a component or entity that is
 * related to the primary identifier, ensuring uniqueness within a specific context.
 *
 * @param salt The salt value used to derive the secondary CUID.
 * @param primary_id Pointer to the primary CUID from which the secondary CUID is derived.
 * @return The computed secondary CUID.
 */
amdcuid get_secondary_cuid(amdcuid_salt_t salt, const amdcuid* primary_id);

/**
 * @brief Generates a primary CUID (Component Unique Identifier) based on hardware identifiers.
 *
 * This function creates a unique identifier for a component using various hardware-specific parameters,
 * such as serial number, unit IDs, revision, device, and vendor information.
 *
 * @param serial_number   The 64-bit serial number of the component.
 * @param unit_id_part1   The first 8-bit part of the unit identifier.
 * @param unit_id_part2   The second 8-bit part of the unit identifier.
 * @param revision_id     The 8-bit revision identifier of the component.
 * @param device_id       The 16-bit device identifier.
 * @param vendor_id       The 16-bit vendor identifier.
 * @param component_type  The 8-bit type identifier for the component.
 * @return amdcuid        The generated primary CUID.
 */
amdcuid generate_primary_cuid(uint64_t serial_number, uint8_t unit_id_part1, uint8_t unit_id_part2,
                                 uint8_t revision_id, uint16_t device_id, uint16_t vendor_id,
                                 uint8_t component_type);

#endif // CUID_SPEC_H