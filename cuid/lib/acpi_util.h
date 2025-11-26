#ifndef ACPI_UTIL_H
#define ACPI_UTIL_H

#include "cuid.h"
#include <cstdint>
#include <string>
#include <vector>

/**
 * @file acpi_util.h
 * @brief ACPI and SMBIOS/DMI utilities for reading system information
 * 
 * This utility provides functions to read system UUID, serial numbers,
 * and other platform information from ACPI tables and DMI/SMBIOS data
 * exposed through Linux sysfs interfaces.
 * 
 * Primary sources:
 * - /sys/class/dmi/id/ - DMI/SMBIOS information (parsed by kernel)
 * - /sys/firmware/acpi/tables/ - Raw ACPI tables
 * 
 * Similar to pci_util, this avoids external dependencies on acpica library
 * by directly reading from sysfs interfaces.
 */

class AcpiUtil {
public:
    /**
     * @brief Get system UUID from DMI/SMBIOS
     * 
     * Reads the system UUID from /sys/class/dmi/id/product_uuid
     * This is the unique identifier for the platform/motherboard.
     * 
     * @param uuid Output string for UUID in format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
     * @return AMDCUID_STATUS_SUCCESS on success
     *         AMDCUID_STATUS_FILE_NOT_FOUND if sysfs entry doesn't exist
     *         AMDCUID_STATUS_PERMISSION_DENIED if access denied (may need root)
     *         AMDCUID_STATUS_INVALID_ARGUMENT if uuid is null
     */
    static amdcuid_status_t get_system_uuid(std::string &uuid);

    /**
     * @brief Get system/board serial number
     * 
     * Reads serial number from DMI/SMBIOS. Tries multiple sources:
     * 1. Product serial (/sys/class/dmi/id/product_serial)
     * 2. Board serial (/sys/class/dmi/id/board_serial)
     * 3. Chassis serial (/sys/class/dmi/id/chassis_serial)
     * 
     * @param serial Output string for serial number
     * @return AMDCUID_STATUS_SUCCESS on success
     *         AMDCUID_STATUS_FILE_NOT_FOUND if no serial available
     *         AMDCUID_STATUS_PERMISSION_DENIED if access denied (may need root)
     */
    static amdcuid_status_t get_system_serial(std::string &serial);

    /**
     * @brief Get board vendor, name, and version
     * 
     * @param vendor Output string for board vendor
     * @param name Output string for board name
     * @param version Output string for board version
     * @return AMDCUID_STATUS_SUCCESS on success
     *         AMDCUID_STATUS_FILE_NOT_FOUND if information not available
     */
    static amdcuid_status_t get_board_info(std::string &vendor, 
                                           std::string &name, 
                                           std::string &version);

    /**
     * @brief Get BIOS information
     * 
     * @param vendor Output string for BIOS vendor
     * @param version Output string for BIOS version
     * @param date Output string for BIOS release date
     * @return AMDCUID_STATUS_SUCCESS on success
     *         AMDCUID_STATUS_FILE_NOT_FOUND if information not available
     */
    static amdcuid_status_t get_bios_info(std::string &vendor,
                                          std::string &version,
                                          std::string &date);

    /**
     * @brief Get system product information
     * 
     * @param name Output string for product name
     * @param family Output string for product family
     * @return AMDCUID_STATUS_SUCCESS on success
     *         AMDCUID_STATUS_FILE_NOT_FOUND if information not available
     */
    static amdcuid_status_t get_product_info(std::string &name,
                                             std::string &family);

    /**
     * @brief Read raw ACPI table
     * 
     * Reads a raw ACPI table from /sys/firmware/acpi/tables/
     * For advanced use cases requiring custom ACPI table parsing.
     * 
     * Common tables: DSDT, FACP, APIC, HPET, IVRS, CRAT
     * 
     * @param table_name Name of the ACPI table (e.g., "FACP", "DSDT")
     * @param buffer Buffer to store table data
     * @param buffer_size Size of the buffer
     * @param bytes_read Output parameter for actual bytes read
     * @return AMDCUID_STATUS_SUCCESS on success
     *         AMDCUID_STATUS_FILE_NOT_FOUND if table doesn't exist
     *         AMDCUID_STATUS_PERMISSION_DENIED if access denied (requires root)
     *         AMDCUID_STATUS_INVALID_ARGUMENT if parameters invalid
     *         AMDCUID_STATUS_BUFFER_TOO_SMALL if buffer insufficient
     */
    static amdcuid_status_t read_acpi_table(const std::string &table_name,
                                            uint8_t *buffer,
                                            size_t buffer_size,
                                            size_t *bytes_read);

private:
    /**
     * @brief Read a single sysfs file
     * 
     * Helper function to read text content from sysfs files.
     * Trims whitespace and newlines from the result.
     * 
     * @param path Full path to sysfs file
     * @param content Output string for file content
     * @return AMDCUID_STATUS_SUCCESS on success
     *         AMDCUID_STATUS_FILE_NOT_FOUND if file doesn't exist
     *         AMDCUID_STATUS_PERMISSION_DENIED if access denied
     */
    static amdcuid_status_t read_sysfs_file(const std::string &path, 
                                            std::string &content);

    /**
     * @brief Trim whitespace from string
     * 
     * @param str String to trim
     * @return Trimmed string
     */
    static std::string trim(const std::string &str);

    // DMI/SMBIOS sysfs paths
    static constexpr const char* DMI_PATH = "/sys/class/dmi/id/";
    static constexpr const char* ACPI_TABLES_PATH = "/sys/firmware/acpi/tables/";
};

#endif // ACPI_UTIL_H
