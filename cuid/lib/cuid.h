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

typedef struct {
    uint8_t bytes[16] = {0};
} amdcuid;

typedef struct {
    uint16_t vendor_id;
    uint16_t family;
    uint16_t model;
    uint16_t device_id;
    uint8_t revision_id;
    uint16_t unit_id;
    uint16_t core;
    uint16_t physical_id;
} amdcuid_cuid_fields_cpu;

typedef struct {
    uint16_t vendor_id;
    uint16_t device_id;
    uint16_t pci_class;
    uint8_t revision_id;
    uint16_t unit_id;
} amdcuid_cuid_fields_gpu;

typedef struct {
    uint16_t vendor_id;
    uint16_t device_id;
    uint16_t pci_class;
    uint8_t revision_id;
} amdcuid_cuid_fields_nic;

typedef struct {
    uint8_t system_information[14]; // System Information (Type1)
} amdcuid_cuid_fields_platform;

typedef struct amdcuid_cuid_fields {
    amdcuid_device_type_t device_type;
    union {
        amdcuid_cuid_fields_cpu cpu;
        amdcuid_cuid_fields_gpu gpu;
        amdcuid_cuid_fields_nic nic;
        amdcuid_cuid_fields_platform platform;
    } fields;
} amdcuid_cuid_fields;

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
    AMDCUID_STATUS_WRONG_DEVICE_TYPE = 6,      ///< Incorrect device type for function
    AMDCUID_STATUS_INSUFFICIENT_SIZE = 7,   ///< Provided buffer or array is too small
    AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND = 8, ///< Hardware fingerprint could not be found
    AMDCUID_STATUS_HW_FINGERPRINT_FORMAT_ERROR = 9, ///< Hardware fingerprint format is incorrect
    AMDCUID_STATUS_HW_FINGERPRINT_PERMISSION_DENIED = 10, ///< Not permitted to access hardware fingerprint
    AMDCUID_STATUS_KEY_ERROR = 11,                          ///< Error reading key file or key file does not exist
    AMDCUID_STATUS_HMAC_ERROR = 12                          ///< HMAC computation failed
} amdcuid_status_t;



/**
 * @brief A wrapper around the secondary CUID to enable device handling/query.
 *
 * This type will help identify devices by their secondary cuid and determine their device type at a glance.
 * When querying devices for additional details, users will provide this type to direct their queries to a particular
 * device.
 */
typedef struct {
    void* impl;
} amdcuid_handle;


/**
 * @brief Bitmask set of device types for querying multiple device types.
 *
 * These values are used as bitmasks to specify which device types to include in queries.
 * Each value corresponds to a specific device type, and AMDCUID_DEVICE_TYPE_SET_ALL selects all types.
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
 * @param[in/out] handle_count On input, the maximum number of handles to write to @p handles.
                               On output, the number of entries it filled, or tried to fill if the provided max number was too small
 * @param[out] handles If non-NULL, must point to an array of amdcuid_handle with at least @p handle_count elements.
 *                     The caller allocates and frees this array.
 * @param[out] total_available_handles If non-NULL, set to the total number of handles available for the given component_types.
 * @return AMDCUID_STATUS_SUCCESS on success, or an appropriate error code on failure.
 */
amdcuid_status_t amdcuid_get_handles(
    amdcuid_device_type_set_t component_types,
    uint32_t *handle_count,
    amdcuid_handle *handles,
    uint32_t *total_available_handles
);

/**
* @brief Retrieve the primary CUID for a given handle. REQUIRES ROOT PRIVILEGES
*
* Retrieve the primary CUID from the /tmp/priv_cuid file for a given handle. This function accesses privileged device information
* and therefore requires ROOT privileges to execute.
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
* @brief Retrieve the secondary CUID for a given device name.
*
* Retrieve the secondary CUID from the /tmp/cuid file for a given device name.
* 
* @param[in]     The name or identifier of the device. The possible identifiers that can be used are listed below:
*                  for GPUS: the render node path (ex.: /sys/class/drm/renderD128/)
*                  for CPUS: the core id (ex.: 0:0)
*                  for NICs: the device name (ex.: /sys/class/net/enp131s0)
* @param[in]     The length of the string given as the device name
* @param[out]    The secondary CUID as an unsigned 128 bit integer
* @return        AMDCUID_STATUS_SUCCESS on success, or an appropriate error code on failure
*/
amdcuid_status_t amdcuid_get_secondary_cuid(
    char *device_name,
    uint32_t *name_len,
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
#endif // CUID_H