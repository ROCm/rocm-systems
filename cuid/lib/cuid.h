
#ifndef CUID_H
#define CUID_H

#include <cstdint>

typedef enum {
    AMDCUID_DEVICE_TYPE_PLATFORM  = 0, /* chassis, motherboard */
    AMDCUID_DEVICE_TYPE_CPU       = 0x1, /* CPU core */
    AMDCUID_DEVICE_TYPE_GPU       = 0x2, /* GPU */
    AMDCUID_DEVICE_TYPE_NIC       = 0x3,  /* NIC */
    AMDCUID_DEVICE_TYPE_NPU       = 0X4,
    AMDCUID_DEVICE_TYPE_STORAGE   = 0X5,
    AMDCUID_DEVICE_TYPE_MEMORY    = 0X6,
    AMDCUID_DEVICE_TYPE_OTHER     = 0XF,
    AMDCUID_DEVICE_TYPE_UNKNOWN   = 0XFF
} amdcuid_device_type_t;



/**
 * @brief Status codes returned by CUID API functions.
 */
typedef enum {
    AMDCUID_STATUS_SUCCESS = 0,           ///< Operation completed successfully
    AMDCUID_STATUS_FILE_NOT_FOUND = 1,    ///< CUID file not found
    AMDCUID_STATUS_DEVICE_NOT_FOUND = 2,  ///< Device not found for the given CUID
    AMDCUID_STATUS_INVALID_ARGUMENT = 3,  ///< Invalid argument passed to function
    AMDCUID_STATUS_PERMISSION_DENIED = 4, ///< Insufficient permissions for operation
    AMDCUID_STATUS_UNSUPPORTED = 5,       ///< Operation or device type not supported
    AMDCUID_STATUS_NOT_INIT = 6,          ///< CUID library not initialized
    AMDCUID_STATUS_WRONG_DEVICE_TYPE = 7,      ///< Incorrect device type for function
    AMDCUID_STATUS_INSUFFICIENT_SIZE = 8,   ///< Provided buffer or array is too small
    AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND = 9, ///< Hardware fingerprint could not be found
    AMDCUID_STATUS_HW_FINGERPRINT_FORMAT_ERROR = 10, ///< Hardware fingerprint format is incorrect
    AMDCUID_STATUS_HW_FINGERPRINT_PERMISSION_DENIED = 11 ///< Not permitted to access hardware fingerprint
} amdcuid_status_t;


typedef struct {
    uint8_t bytes[16] = {0};
} amdcuid;

/**
 * @brief Salt structure used in CUID generation.
 *
 * This structure holds a 112-bit (14-byte) salt value used in the generation of
 * Component Unique Identifiers (CUIDs). The salt adds an additional layer of uniqueness
 * and security to the CUID generation process.
 */
typedef struct {
    uint8_t bytes[14]; // 112 bits = 14 bytes
} amdcuid_salt_t;


/**
 * @brief Opaque handler pointing to the underlying implementation.
 *
 * This type is used to abstract the internal implementation details of the CUID library.
 * Users interact with this handle to reference internal objects or contexts without
 * needing to know their structure or contents.
 */
typedef struct {
    void *impl;
} amdcuid_handle;



/**
 * @brief Bitmask set of component types for querying multiple device types.
 *
 * These values are used as bitmasks to specify which component types to include in queries.
 * Each value corresponds to a specific device type, and AMDCUID_COMPONENT_TYPE_SET_ALL selects all types.
 */
typedef enum {
    AMDCUID_DEVICE_TYPE_SET_PLATFORM  = 1U << AMDCUID_DEVICE_TYPE_PLATFORM, ///< Platform devices (chassis, motherboard)
    AMDCUID_DEVICE_TYPE_SET_CPU       = 1U << AMDCUID_DEVICE_TYPE_CPU,      ///< CPU devices
    AMDCUID_DEVICE_TYPE_SET_GPU       = 1U << AMDCUID_DEVICE_TYPE_GPU,      ///< GPU devices
    AMDCUID_DEVICE_TYPE_SET_NIC       = 1U << AMDCUID_DEVICE_TYPE_NIC,      ///< NIC devices
    AMDCUID_DEVICE_TYPE_SET_ALL       = -1U                                    ///< All device types
} amdcuid_device_type_set_t;


/**
 * @brief Retrieve a list of CUID handles present in the system.
 *
 * The order of the handles in the list is unspecified and may vary between calls.
 *
 * @param[in] component_types Bitmask of component types to query (use AMDCUID_COMPONENT_TYPE_SET_*).
 * @param[in] handle_count Maximum number of handles to write to @p handles.
 * @param[out] handles If non-NULL, must point to an array of amdcuid_handle with at least @p handle_count elements.
 *                     The caller allocates and frees this array.
 * @param[out] total_available_handles If non-NULL, set to the total number of handles available for the given component_types.
 * @return AMDCUID_STATUS_SUCCESS on success, or an appropriate error code on failure.
 */
amdcuid_status_t amdcuid_get_handles(
    amdcuid_device_type_set_t  component_types,
    uint32_t handle_count,
    amdcuid_handle *handles,
    uint32_t *total_available_handles
);

/**
* @brief Retrieve the primary CUID for a given handle.
*
* Retrieve the primary CUID for a given handle.
* 
* @param[in]     The handle of the device to query
* @param[out]    The primary CUID as an unsigned 128 bit integer
* @return        AMDCUID_STATUS_SUCCESS on success, or an appropriate error code on failure
*/
amdcuid_status_t amdcuid_get_primary_cuid(
    amdcuid_handle handle,
    amdcuid *primary_cuid
);

/**
* @brief Retrieve the secondary CUID for a given handle.
*
* Retrieve the secondary CUID for a given handle.
* 
* @param[in]     The handle of the device to query
* @param[out]    The secondary CUID as an unsigned 128 bit integer
* @return        AMDCUID_STATUS_SUCCESS on success, or an appropriate error code on failure
*/
amdcuid_status_t amdcuid_get_secondary_cuid(
    amdcuid_handle handle,
    amdcuid *secondary_cuid
);

/**
* @brief Retrieve the device type for a given handle.
*
* Retrieve the device type for a given handle.
* 
* @param[in]     The handle of the device to query
* @param[out]    The device type as the enum amdcuid_device_type_t
* @return        AMDCUID_STATUS_SUCCESS on success, or an appropriate error code on failure
*/
amdcuid_status_t amdcuid_get_device_type(
    amdcuid_handle handle,
    amdcuid_device_type_t *dev_type
);

/**
* @brief Retrieve the vendor ID for a given handle.
*
* Retrieve the vendor ID for a given handle.
* 
* @param[in]     The handle of the device to query
* @param[out]    The vendor ID as an unsigned 16 bit integer
* @return        AMDCUID_STATUS_SUCCESS on success, or an appropriate error code on failure
*/
amdcuid_status_t amdcuid_get_vendor_id(
    amdcuid_handle handle,
    uint16_t *vendor_id
);

/**
* @brief Retrieve the revision ID for a given handle.
*
* Retrieve the revision ID for a given handle.
* 
* @param[in]     The handle of the device to query
* @param[out]    The revision ID as unsigned 16 bit integer
* @return        AMDCUID_STATUS_SUCCESS on success, or an appropriate error code on failure
*/
amdcuid_status_t amdcuid_get_revision_id(
    amdcuid_handle handle,
    uint16_t *revision_id
);

/**
* @brief Retrieve the partition ID for a given handle.
*
* Retrieve the partition ID for a given handle.
* 
* @param[in]     The handle of the device to query
* @param[out]    The partition ID as an unsigned 32 bit integer
* @return        AMDCUID_STATUS_SUCCESS on success, or an appropriate error code on failure
*/
amdcuid_status_t amdcuid_get_partition_info(
    amdcuid_handle handle,
    uint32_t partition_info
);

/**
* @brief Retrieve the bdf for a given handle.
*
* Retrieve the bdf for a given handle.
* 
* @param[in]     The handle of the device to query
* @param[out]    The bdf as a char*
* @param[in/out] The length of the string buffer for bdf; On input, the maximum number of characters to write to bdf.
                                                          On output, the number of characters written to bdf
* @return        AMDCUID_STATUS_SUCCESS on success, or an appropriate error code on failure
*/
amdcuid_status_t amdcuid_get_bdf(
    amdcuid_handle handle,
    char *bdf,
    uint32_t *length
);

/**
* @brief Retrieve the render node for a given handle.
*
* Retrieve the render node for a given handle.
* 
* @param[in]     The handle of the device to query
* @param[out]    The render node as a char *
* @param[in/out] The length of the string buffer for render node; On input, the maximum number of characters to write to render node
                                                          On output, the number of characters written to render node
* @return        AMDCUID_STATUS_SUCCESS on success, or an appropriate error code on failure
*/
amdcuid_status_t amdcuid_get_render_node(
    amdcuid_handle handle,
    char *render_node,
    uint32_t *length
);

/**
* @brief Retrieve the core for a given handle.
*
* Retrieve the core for a given handle.
* 
* @param[in]     The handle of the device to query
* @param[out]    The core as an unsigned 16 bit integer
* @return        AMDCUID_STATUS_SUCCESS on success, or an appropriate error code on failure
*/
amdcuid_status_t amdcuid_get_cpu_core(
    amdcuid_handle handle,
    uint16_t core
);

/**
* @brief Retrieve the network interface name for a given handle.
*
* Retrieve the network interface name for a given handle.
* 
* @param[in]     The handle of the device to query
* @param[out]    The network interface name as a char *
* @param[in/out] The length of the string buffer for network interface name; On input, the maximum number of characters to write to network interface name
                                                          On output, the number of characters written to network interface name
* @return        AMDCUID_STATUS_SUCCESS on success, or an appropriate error code on failure
*/
amdcuid_status_t amdcuid_get_network_interface(
    amdcuid_handle handle,
    char *network_interface,
    uint32_t *length
);

/**
 * @brief Retrieve handles whose secondary CUID matches the given string prefix.
 *
 * This helper function allows searching for device handles by a partial (prefix) match
 * of the secondary CUID, provided as a hexadecimal string. All handles whose secondary
 * CUID matches the prefix will be returned.
 *
 * @param[in]  prefix         The hexadecimal string prefix to match (case-insensitive, no "0x" prefix).
 * @param[in]  component_types Bitmask of component types to query (use AMDCUID_COMPONENT_TYPE_SET_*).
 * @param[out] handles        Array to store matching handles (allocated by caller).
 * @param[in]  max_handles    Maximum number of handles to write to @p handles.
 * @param[out] matched_count  Number of handles matched and written to @p handles.
 * @return     AMDCUID_STATUS_SUCCESS on success, or an appropriate error code on failure.
 */
amdcuid_status_t amdcuid_find_handles_by_secondary_cuid_prefix(
    const char *prefix,
    amdcuid_device_type_set_t component_types,
    amdcuid_handle *handles,
    uint32_t max_handles,
    uint32_t *matched_count
);


#endif  // CUID_H
