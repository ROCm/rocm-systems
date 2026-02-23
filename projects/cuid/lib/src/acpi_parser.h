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

#ifndef ACPI_PARSER_H
#define ACPI_PARSER_H

#include "include/amd_cuid.h"
#include <cstdint>
#include <string>
#include <vector>

/**
 * @file acpi_parser.h
 * @brief ACPI table parser for extracting CPU identification information
 * 
 * Parses binary ACPI tables (MADT/APIC, DSDT) to extract processor-specific
 * identifiers like APIC IDs and processor UIDs needed for CPU CUID generation.
 * 
 * ACPI tables parsed:
 * - MADT (Multiple APIC Description Table): Contains Local APIC entries with APIC IDs
 * - DSDT (Differentiated System Description Table): Contains Processor objects with _UID
 * 
 * References:
 * - ACPI Specification 6.5: https://uefi.org/specs/ACPI/6.5/
 * - MADT format: Section 5.2.12
 * - AML bytecode: Chapter 20
 */

/**
 * @brief Standard ACPI table header (common to all ACPI tables)
 */
struct AcpiTableHeader {
    char signature[4];      ///< Table signature (e.g., "APIC", "DSDT")
    uint32_t length;        ///< Total table length including header
    uint8_t revision;       ///< Table revision
    uint8_t checksum;       ///< Checksum of entire table (must sum to 0)
    char oem_id[6];         ///< OEM identification
    char oem_table_id[8];   ///< OEM table identifier
    uint32_t oem_revision;  ///< OEM revision
    char creator_id[4];     ///< Creator utility ID
    uint32_t creator_revision; ///< Creator utility revision
} __attribute__((packed));

/**
 * @brief MADT (Multiple APIC Description Table) header
 */
struct MadtHeader {
    AcpiTableHeader header;
    uint32_t local_apic_address; ///< Physical address of local APIC
    uint32_t flags;              ///< Multiple APIC flags
} __attribute__((packed));

/**
 * @brief MADT interrupt controller structure header
 */
struct MadtEntryHeader {
    uint8_t type;   ///< Entry type (0 = Local APIC, 9 = Local x2APIC, etc.)
    uint8_t length; ///< Entry length including header
} __attribute__((packed));

/**
 * @brief MADT Local APIC entry (type 0)
 */
struct MadtLocalApic {
    MadtEntryHeader header;
    uint8_t acpi_processor_uid; ///< ACPI Processor UID
    uint8_t apic_id;            ///< Local APIC ID
    uint32_t flags;             ///< Flags (bit 0: enabled, bit 1: online capable)
} __attribute__((packed));

/**
 * @brief MADT Local x2APIC entry (type 9)
 */
struct MadtLocalX2Apic {
    MadtEntryHeader header;
    uint16_t reserved;          ///< Reserved (must be zero)
    uint32_t x2apic_id;         ///< x2APIC ID
    uint32_t flags;             ///< Flags (bit 0: enabled, bit 1: online capable)
    uint32_t acpi_processor_uid; ///< ACPI Processor UID
} __attribute__((packed));

/**
 * @brief CPU identification information from ACPI
 */
struct AcpiCpuInfo {
    uint32_t apic_id;           ///< Local APIC or x2APIC ID
    uint32_t processor_uid;     ///< ACPI Processor UID
    bool enabled;               ///< CPU is enabled
    bool is_x2apic;            ///< Uses x2APIC (vs legacy APIC)
};

/**
 * @brief ACPI table parser for CPU identification
 */
class AcpiParser {
public:
    /**
     * @brief Parse MADT table to extract CPU APIC IDs and processor UIDs
     * 
     * Reads and parses the MADT (Multiple APIC Description Table) from
     * /sys/firmware/acpi/tables/APIC to extract Local APIC and x2APIC entries.
     * 
     * Each entry contains:
     * - APIC ID: Hardware identifier for the CPU's local APIC
     * - Processor UID: ACPI processor unique identifier
     * - Flags: Whether CPU is enabled/online
     * 
     * @param cpu_info Output vector of CPU information structures
     * @return AMDCUID_STATUS_SUCCESS on success
     *         AMDCUID_STATUS_ACPI_ERROR if MADT table doesn't exist
     *         AMDCUID_STATUS_PERMISSION_DENIED if access denied (need root)
     *         AMDCUID_STATUS_INVALID_FORMAT if table format is invalid
     */
    static amdcuid_status_t parse_madt(std::vector<AcpiCpuInfo>& cpu_info);

    /**
     * @brief Validate ACPI table header checksum
     * 
     * Verifies that the sum of all bytes in the table equals zero (mod 256).
     * This is the standard ACPI checksum validation.
     * 
     * @param data Pointer to table data
     * @param length Table length in bytes
     * @return true if checksum is valid, false otherwise
     */
    static bool validate_checksum(const uint8_t* data, size_t length);

    /**
     * @brief Get CPU count from MADT
     * 
     * Quick function to count enabled CPUs without full parsing.
     * 
     * @param count Output for CPU count
     * @return AMDCUID_STATUS_SUCCESS on success
     */
    static amdcuid_status_t get_cpu_count(uint32_t& count);

private:
    /**
     * @brief Read ACPI table from sysfs
     * 
     * @param table_name Name of table (e.g., "APIC", "DSDT")
     * @param data Output buffer for table data
     * @return AMDCUID_STATUS_SUCCESS on success
     */
    static amdcuid_status_t read_acpi_table(const char* table_name, std::vector<uint8_t>& data);

    /**
     * @brief Parse individual MADT entry
     * 
     * @param entry Pointer to entry data
     * @param cpu_info Output CPU info if entry is Local APIC or x2APIC
     * @return true if entry was parsed successfully
     */
    static bool parse_madt_entry(const uint8_t* entry, AcpiCpuInfo& cpu_info);
};

#endif // ACPI_PARSER_H
