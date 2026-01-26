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

#ifndef AMD_CUID_H
#define AMD_CUID_H

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

//! Major version should be changed for every header change that breaks ABI
//! Such as adding/deleting APIs, changing names, fields of structures, etc.
#define AMDCUID_LIB_VERSION_MAJOR 0

//! Minor version should be updated for each API change, but without changing headers
#define AMDCUID_LIB_VERSION_MINOR 1

//! Patch version should be updated for each bug fix or non-API change
#define AMDCUID_LIB_VERSION_PATCH 1

/**
 * @brief Retrieve the version of the CUID library.
 * 
 * Major version should be changed for every header change that breaks ABI such as
 * adding/deleting APIs, changing names, fields of structures, etc. Minor version
 * should be updated for each API change, but without changing headers. Patch version
 * should be updated for each bug fix or non-API change.
 *
 * @param[out] major Pointer to store the major version number.
 * @param[out] minor Pointer to store the minor version number.
 * @param[out] patch Pointer to store the patch version number.
 */
void amdcuid_get_library_version(
    uint32_t* major,
    uint32_t* minor,
    uint32_t* patch
);

/**
 * @brief Retrieve the version string of the CUID library.
 *
 * @return A constant character pointer to the version string. The format is "MAJOR.MINOR.PATCH".
 */
const char* amdcuid_library_version_to_string();

/**
 * @brief Status codes returned by CUID API functions.
 */
typedef enum {
    AMDCUID_STATUS_SUCCESS = 0,                     ///< Operation completed successfully
    AMDCUID_STATUS_FILE_NOT_FOUND = 1,              ///< CUID file not found
    AMDCUID_STATUS_DEVICE_NOT_FOUND = 2,            ///< Device(s) not found
    AMDCUID_STATUS_INVALID_ARGUMENT = 3,            ///< Invalid argument passed to function
    AMDCUID_STATUS_PERMISSION_DENIED = 4,           ///< Insufficient permissions for operation
    AMDCUID_STATUS_UNSUPPORTED = 5,                 ///< Operation or device type not supported on system
    AMDCUID_STATUS_WRONG_DEVICE_TYPE = 6,           ///< Incorrect device type for function
    AMDCUID_STATUS_INSUFFICIENT_SIZE = 7,           ///< Provided buffer or array is too small
    AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND = 8,    ///< Hardware fingerprint could not be found
    AMDCUID_STATUS_KEY_ERROR = 9,                   ///< An error occurred related to the hash key
    AMDCUID_STATUS_HMAC_ERROR = 10,                 ///< An error occurred during HMAC computation
    AMDCUID_STATUS_FILE_ERROR = 11,                 ///< File I/O error occurred when reading or writing the CUID files
    AMDCUID_STATUS_INVALID_FORMAT = 12,             ///< Data format given or read is invalid or malformed
    AMDCUID_STATUS_PCI_ERROR = 13,                  ///< An error occurred while accessing or parsing PCI configuration space
    AMDCUID_STATUS_SMBIOS_ERROR = 14,               ///< An error occurred while accessing or parsing the SMBIOS table
    AMDCUID_STATUS_ACPI_ERROR = 15,                 ///< An error occurred while accessing or parsing the ACPI table
    AMDCUID_STATUS_CPUINFO_ERROR = 16,              ///< An error occurred while accessing or parsing CPUINFO
    AMDCUID_STATUS_IPC_ERROR = 17                   ///< An error occurred during IPC communication with the daemon
} amdcuid_status_t;

/**
 * @brief Convert a CUID status code to a human-readable string.
 * 
 * @param[in] status The CUID status code to convert.
 * @return A constant character pointer to the string representation of the status code.
 */
const char* amdcuid_status_to_string(
    amdcuid_status_t status
);

/**
 * @brief UUIDv8 representation of a CUID. Also serves as an opaque handle type.
 * 
 * This structure holds the 16-byte CUID value in a UUIDv8 format used to uniquely
 * identify devices. The CUID will also function as the handle. Users will use the
 * CUID to query device information and the library will look up the device internally
 * using the given CUID/handle. Handles are created when the library is initialized.
 * Handles will be invalidated when devices are removed or when the library is
 * shutdown. If handles have been invalidated, users must obtain new handles by
 * re-initializing the library if making use of all handles on the system, or
 * by querying for a specific device using amdcuid_get_handle_by_dev_path().
 */
typedef struct {
    uint8_t bytes[16];
} amdcuid_id_t;

/**
 * @brief Convert a CUID to a human-readable string.
 * 
 * @param[in] cuid_value The CUID to convert.
 * @return A constant character pointer to the string representation of the CUID.
 */
const char* amdcuid_id_to_string(
    amdcuid_id_t cuid_value
);

typedef enum {
    AMDCUID_DEVICE_TYPE_NONE      = 0,
    AMDCUID_DEVICE_TYPE_PLATFORM  = 0x1, /* chassis, motherboard */
    AMDCUID_DEVICE_TYPE_CPU       = 0x2, /* CPU core */
    AMDCUID_DEVICE_TYPE_GPU       = 0x3, /* GPU */
    AMDCUID_DEVICE_TYPE_NIC       = 0x4, /* NIC */
    AMDCUID_DEVICE_TYPE_LAST      = 0x4
} amdcuid_device_type_t;

/**
 * @brief Add a new device to the CUID device registry.
 * 
 * This function allows adding a new device to the CUID device registry and CUID
 * files at runtime. It will initiate a request to the CUID daemon to add the
 * device. The device's CUID will be generated by the daemon and returned. In
 * low resource environments, such as containers, the daemon will only run at start up
 * to discover devices, so this function will not work. In those instances, the daemon
 * should be restarted to pick up changes to the devices running in the system.
 * Requires elevated permissions to add the device.
 * 
 * @param[in] dev_path The device path of the new device.
 * @param[in] device_type The type of the device (see amdcuid_device_type_t).
 * @param[out] handle Pointer to an amdcuid_id_t that will be filled with the new device's handle.
 * 
 * @return AMDCUID_STATUS_SUCCESS on success,
 *         AMDCUID_STATUS_INVALID_ARGUMENT if the provided arguments are invalid,
 *         AMDCUID_STATUS_PERMISSION_DENIED if insufficient permissions to add the device
 *         AMDCUID_STATUS_DEVICE_NOT_FOUND if the device could not be found at the specified path
 *         AMDCUID_STATUS_UNSUPPORTED if the device type is not supported
 *         AMDCUID_STATUS_IPC_ERROR if an error occurred during communication with the daemon
 */
amdcuid_status_t amdcuid_add_device(
    const char* dev_path,
    amdcuid_device_type_t device_type,
    amdcuid_id_t* handle
);

/**
 * @brief Remove a device from the CUID device registry.
 * 
 * This function allows removing a device from the CUID device registry and CUID
 * files at runtime. It will initiate a request to the CUID daemon to remove the
 * device. The device's handle will be invalidated. In low resource environments,
 * such as containers, the daemon will only run at start up to discover devices,
 * so this function will not work. In those instances, the daemon should be
 * restarted to pick up changes to the devices running in the system. Requires
 * elevated permissions to remove the device.
 * 
 * @param[in] handle The handle of the device to remove.
 * 
 * @return AMDCUID_STATUS_SUCCESS on success,
 *         AMDCUID_STATUS_DEVICE_NOT_FOUND if the provided handle is invalid,
 *         AMDCUID_STATUS_PERMISSION_DENIED if insufficient permissions to remove the device
 *         AMDCUID_STATUS_IPC_ERROR if an error occurred during communication with the daemon
 */
amdcuid_status_t amdcuid_remove_device(
    amdcuid_id_t handle
);

/**
 * @brief Retrieve the number of CUID handles present in the system.
 *
 * @param[out] count Pointer to a uint32_t that will be set to the number of handles. This will be set to 0 if no devices are found.
 *
 * @return AMDCUID_STATUS_SUCCESS on success,
 *         AMDCUID_STATUS_UNSUPPORTED if no supported devices are found
 */
amdcuid_status_t amdcuid_get_num_handles(
    uint32_t* count
);

/**
 * @brief Retrieve a list of all CUID handles present in the system.
 *
 * The order of the handles in the list is unspecified and may vary between calls.
 *
 * @param[out] handles Pointer to an array of CUID handles. This will be set to nullptr if no devices are found.
 * @param[in] count Pointer to a uint32_t that must be set to the number of handles. The number of handles should be obtained from amdcuid_get_num_handles() prior to calling this function.
 *
 * @return AMDCUID_STATUS_SUCCESS on success,
 *         AMDCUID_STATUS_UNSUPPORTED if no supported devices are found
 */
amdcuid_status_t amdcuid_get_all_handles(
    amdcuid_id_t *handles,
    uint32_t count
);

/**
 * @brief Types of properties that can be queried from a device.
 * 
 * These properties are also stored in the CUID files. Some properties may
 * require elevated permissions to access. Not all device types will support
 * all properties.
 */
typedef enum {
    AMDCUID_QUERY_NONE = 0,                   ///< No query
    AMDCUID_QUERY_PRIMARY_CUID = 1,           ///< Query the primary CUID (uint8_t[16]). The bits will be formatted in the UUIDv8 format. Requires elevated permissions.
    AMDCUID_QUERY_SECONDARY_CUID = 2,         ///< Query the secondary CUID (uint8_t[16]). The bits will be formatted in the UUIDv8 format. This is the user visible CUID in most cases.
    AMDCUID_QUERY_HARDWARE_FINGERPRINT = 3,   ///< Query the hardware fingerprint (aka serial number/id) (uint64_t). Requires elevated permissions.
    AMDCUID_QUERY_DEVICE_PATH = 4,            ///< Query the device path (string)
    AMDCUID_QUERY_DEVICE_TYPE = 5,            ///< Query the device type (). One of amdcuid_device_type_t
    AMDCUID_QUERY_VENDOR_ID = 6,              ///< Query the vendor ID (uint16_t). Supported by all device types.
    AMDCUID_QUERY_DEVICE_ID = 7,              ///< Query the device ID (uint16_t). Supported by GPU, NIC, and CPU device types.
    AMDCUID_QUERY_REVISION_ID = 8,            ///< Query the revision ID (uint16_t). Supported by GPU, NIC, and CPU device types.
    AMDCUID_QUERY_UNIT_ID = 9,                ///< Query the unit ID (uint16_t). Supported by GPU and CPU device type.
    AMDCUID_QUERY_FAMILY = 10,                ///< Query the CPU family (uint16_t). Supported by CPU device type.
    AMDCUID_QUERY_MODEL = 11,                 ///< Query the CPU model (uint16_t). Supported by CPU device type.
    AMDCUID_QUERY_CORE_ID = 12,               ///< Query the core ID (uint16_t). Supported by CPU device type.
    AMDCUID_QUERY_PHYSICAL_ID = 13,           ///< Query the physical package ID (uint16_t). Supported by CPU device type.
    AMDCUID_QUERY_PCI_CLASS = 14,             ///< Query the PCI class (uint16_t). Supported by GPU and NIC device types.
    AMDCUID_QUERY_LAST = 14
} amdcuid_query_t;

/**
 * @brief Maximum buffer size for query operations.
 * 
 * Use this when querying string properties like device path.
 */
#define AMDCUID_MAX_QUERY_BUFFER_SIZE 256

/**
 * @brief Query a specific property of a device identified by its CUID handle.
 * 
 * This function allows querying various properties of a device using its CUID handle. These properties are stored in the
 * CUID files or can be derived from the device itself. Accessing certain properties may require elevated permissions.
 *
 * @param[in] handle The CUID handle of the device to query.
 * @param[in] query The property to query (see amdcuid_query_t).
 * @param[out] data Pointer to a buffer where the queried data will be stored.
 * @param[in/out] length On input, the size in bytes of the buffer pointed to by @p data.
 *                      On output, the actual size in bytes of the data written or required.
 * @return AMDCUID_STATUS_SUCCESS on success
 *         AMDCUID_STATUS_DEVICE_NOT_FOUND if the handle is invalid,
 *         AMDCUID_STATUS_INSUFFICIENT_SIZE if the provided buffer is too small,
 *         AMDCUID_STATUS_PERMISSION_DENIED if insufficient permissions to access the property,
 *         AMDCUID_STATUS_WRONG_DEVICE_TYPE if the property is not applicable to the device type,
 *         AMDCUID_STATUS_INVALID_ARGUMENT if the query type is invalid,
 *         AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND if the hardware fingerprint could not be found.
 */
amdcuid_status_t amdcuid_query_device_property(
    amdcuid_id_t handle,
    amdcuid_query_t query,
    void *data,
    uint32_t *length
);

/**
 * @brief Set the hash key used for HMAC computations on CUIDs.
 * 
 * This function takes the key given and sets it as the HMAC key for future computations.
 * The key is also stored in the default key file location (/opt/cuid/hmac_key.bin). Users can
 * use amdcuid_generate_hash_key() to create a new random key before calling this function.
 * Requires elevated permissions to set the key.
 *
 * @param[in] key Pointer to the HMAC key. This must be 32 bytes in length.
 * @return AMDCUID_STATUS_SUCCESS on success,
 *         AMDCUID_STATUS_PERMISSION_DENIED if insufficient permissions,
 *         AMDCUID_STATUS_KEY_ERROR if there was an error writing the key
 */
amdcuid_status_t amdcuid_set_hash_key(
    const uint8_t key[32]
);

/**
 * @brief Create a new HMAC key for HMAC computations on CUIDs.
 *
 * This function generates a new random HMAC key. Use amdcuid_set_hash_key() to set the key for use in the library
 * to the key generated by this function. Requires elevated permissions to generate the key.
 *
 * @param[out] key Pointer to the buffer where the generated HMAC key will be stored. This must be 32 bytes in length.
 * @return AMDCUID_STATUS_SUCCESS on success
 *         AMDCUID_STATUS_PERMISSION_DENIED if insufficient permissions,
 *         AMDCUID_STATUS_INVALID_ARGUMENT if the arguments are invalid,
 *         AMDCUID_STATUS_KEY_ERROR if there was an error generating the key.
 */
amdcuid_status_t amdcuid_generate_hash_key(
    uint8_t key[32]
);

#ifdef __cplusplus
}
#endif

#endif // AMD_CUID_H