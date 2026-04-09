

#ifndef LIB_CUID_INTERNAL_H_
#define LIB_CUID_INTERNAL_H_

#include <cstdint>
#include "include/amd_cuid.h"

typedef struct {
    uint16_t vendor_id;      ///< CPU Vendor ID (from CPUID EAX=0)
    uint16_t family;         ///< CPU Family (from CPUID EAX=1)
    uint16_t model;          ///< CPU Model (from CPUID EAX=1)
    uint16_t device_id;      ///< DeviceID = Family & Model combined
    uint8_t revision_id;     ///< CPU Stepping/RevisionID (from CPUID EAX=1)
    uint16_t unit_id;        ///< UnitID from APIC ID (from ACPI MADT)
    uint16_t core;           ///< Core number within package
    uint16_t physical_id;    ///< Physical package/socket ID
} amdcuid_cuid_public_fields_cpu;

typedef struct {
    uint16_t vendor_id;
    uint16_t device_id;
    uint16_t pci_class;
    uint8_t revision_id;
    uint16_t unit_id;
} amdcuid_cuid_public_fields_gpu;

typedef struct {
    uint16_t vendor_id;
    uint16_t device_id;
    uint16_t pci_class;
    uint8_t revision_id;
} amdcuid_cuid_public_fields_nic;

typedef struct {
    uint16_t vendor_id;
} amdcuid_cuid_public_fields_platform;

typedef struct amdcuid_cuid_public_fields {
    amdcuid_device_type_t device_type;
    union {
        amdcuid_cuid_public_fields_cpu cpu;
        amdcuid_cuid_public_fields_gpu gpu;
        amdcuid_cuid_public_fields_nic nic;
        amdcuid_cuid_public_fields_platform platform;
    } fields;
} amdcuid_cuid_public_fields;

typedef struct amdcuid_primary_id {
    uint8_t raw_bits[16];               // 122 bits of raw data
    amdcuid_id_t UUIDv8_representation; // UUIDv8 representation of the primary CUID which adds in the UUIDv8 required bits
} amdcuid_primary_id;

typedef struct amdcuid_derived_id {
    uint8_t hash[14];                   // 110 LSB bits of HMAC hash
    uint8_t raw_bits[16];               // 122 bits which adds back the unit id bits
    amdcuid_id_t UUIDv8_representation; // UUIDv8 representation of the derived CUID which adds in the UUIDv8 required bits
} amdcuid_derived_id;
#endif // LIB_CUID_INTERNAL_H_