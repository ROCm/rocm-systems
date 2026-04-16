/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef ROCSHMEM_AMDSMI_LOADER_HPP_
#define ROCSHMEM_AMDSMI_LOADER_HPP_

#include <cstdint>

namespace rocshmem {

// Forward declarations and type definitions to avoid compile-time dependency on AMD SMI headers
// These definitions match the ABI of libamd_smi.so

// AMD SMI status type
typedef enum {
    AMDSMI_STATUS_SUCCESS = 0,              //!< Call succeeded
    // Library usage errors
    AMDSMI_STATUS_INVAL = 1,                //!< Invalid parameters
    AMDSMI_STATUS_NOT_SUPPORTED = 2,        //!< Command not supported
    AMDSMI_STATUS_NOT_YET_IMPLEMENTED = 3,  //!< Not implemented yet
    AMDSMI_STATUS_FAIL_LOAD_MODULE = 4,     //!< Fail to load lib
    AMDSMI_STATUS_FAIL_LOAD_SYMBOL = 5,     //!< Fail to load symbol
    AMDSMI_STATUS_DRM_ERROR = 6,            //!< Error when call libdrm
    AMDSMI_STATUS_API_FAILED = 7,           //!< API call failed
    AMDSMI_STATUS_TIMEOUT = 8,              //!< Timeout in API call
    AMDSMI_STATUS_RETRY = 9,                //!< Retry operation
    AMDSMI_STATUS_NO_PERM = 10,             //!< Permission Denied
    AMDSMI_STATUS_INTERRUPT = 11,           //!< An interrupt occurred during execution of function
    AMDSMI_STATUS_IO = 12,                  //!< I/O Error
    AMDSMI_STATUS_ADDRESS_FAULT = 13,       //!< Bad address
    AMDSMI_STATUS_FILE_ERROR = 14,          //!< Problem accessing a file
    AMDSMI_STATUS_OUT_OF_RESOURCES = 15,    //!< Not enough memory
    AMDSMI_STATUS_INTERNAL_EXCEPTION = 16,  //!< An internal exception was caught
    AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS = 17, //!< The provided input is out of allowable or safe range
    AMDSMI_STATUS_INIT_ERROR = 18,          //!< An error occurred when initializing internal data structures
    AMDSMI_STATUS_REFCOUNT_OVERFLOW = 19,   //!< An internal reference counter exceeded INT32_MAX
    AMDSMI_STATUS_DIRECTORY_NOT_FOUND = 20, //!< Error when a directory is not found, maps to ENOTDIR
    // Processor related errors
    AMDSMI_STATUS_BUSY = 30,                //!< Processor busy
    AMDSMI_STATUS_NOT_FOUND = 31,           //!< Processor Not found
    AMDSMI_STATUS_NOT_INIT = 32,            //!< Processor not initialized
    AMDSMI_STATUS_NO_SLOT = 33,             //!< No more free slot
    AMDSMI_STATUS_DRIVER_NOT_LOADED = 34,   //!< Processor driver not loaded
    // Data and size errors
    AMDSMI_STATUS_MORE_DATA = 39,           //!< There is more data than the buffer size the user passed
    AMDSMI_STATUS_NO_DATA = 40,             //!< No data was found for a given input
    AMDSMI_STATUS_INSUFFICIENT_SIZE = 41,   //!< Not enough resources were available for the operation
    AMDSMI_STATUS_UNEXPECTED_SIZE = 42,     //!< An unexpected amount of data was read
    AMDSMI_STATUS_UNEXPECTED_DATA = 43,     //!< The data read or provided to function is not what was expected
    //esmi errors
    AMDSMI_STATUS_NON_AMD_CPU = 44,         //!< System has different cpu than AMD
    AMDSMI_STATUS_NO_ENERGY_DRV = 45,       //!< Energy driver not found
    AMDSMI_STATUS_NO_MSR_DRV = 46,          //!< MSR driver not found
    AMDSMI_STATUS_NO_HSMP_DRV = 47,         //!< HSMP driver not found
    AMDSMI_STATUS_NO_HSMP_SUP = 48,         //!< HSMP not supported
    AMDSMI_STATUS_NO_HSMP_MSG_SUP = 49,     //!< HSMP message/feature not supported
    AMDSMI_STATUS_HSMP_TIMEOUT = 50,        //!< HSMP message timed out
    AMDSMI_STATUS_NO_DRV = 51,              //!< No Energy and HSMP driver present
    AMDSMI_STATUS_FILE_NOT_FOUND = 52,      //!< file or directory not found
    AMDSMI_STATUS_ARG_PTR_NULL = 53,        //!< Parsed argument is invalid
    AMDSMI_STATUS_AMDGPU_RESTART_ERR = 54,  //!< AMDGPU restart failed
    AMDSMI_STATUS_SETTING_UNAVAILABLE = 55, //!< Setting is not available
    AMDSMI_STATUS_CORRUPTED_EEPROM = 56,    //!< EEPROM is corrupted
    // General errors
    AMDSMI_STATUS_MAP_ERROR = 0xFFFFFFFE,     //!< The internal library error did not map to a status code
    AMDSMI_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF  //!< An unknown error occurred
} amdsmi_status_t;

// AMD SMI processor handle (opaque pointer)
typedef void* amdsmi_processor_handle;

// AMD SMI GPU fabric info structure (version 1)
// Based on AMD SMI library interface
/**
 * @brief Fabric type
 */
typedef enum {
    AMDSMI_FABRIC_TYPE_UALOE,
    AMDSMI_FABRIC_TYPE_UALLINK
} amdsmi_fabric_type_t;

/**
 * @brief Fabric NHT address mode
 */
typedef enum {
    AMDSMI_FABRIC_NHT_ADDRESS_MODE_SOURCE_ALIASING,
    AMDSMI_FABRIC_NHT_ADDRESS_MODE_SOURCE_IDENTIFICATION
} amdsmi_fabric_nht_address_mode_t;

/**
 * @brief Fabric accelerator vPoD state
 */
typedef enum {
    AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_UNCONFIGURED,
    AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_CONFIGURED,
    AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY,
    AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE,
    AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ERROR
} amdsmi_fabric_accelerator_vpod_state_t;

typedef union {
    struct bdf_ {
        uint64_t function_number : 3;
        uint64_t device_number : 5;
        uint64_t bus_number : 8;
        uint64_t domain_number : 48;
    } bdf;
    struct {
        uint64_t function_number : 3;
        uint64_t device_number : 5;
        uint64_t bus_number : 8;
        uint64_t domain_number : 48;
    };
    uint64_t as_uint;
} amdsmi_bdf_t;

// Constants for fabric info arrays
#define AMDSMI_FABRIC_ACTIVE_ACCELERATORS_BITMAP_SIZE 32
#define AMDSMI_FABRIC_MAX_LOCAL_GPUS 8

/**
 * @brief Fabric device configuration information (version 1)
 */
typedef struct {
    uint32_t accelerator_id; //!< Accelerator identifier (range 0 to 1023)
    amdsmi_fabric_type_t fabric_type; //!< UALOE or UALLINK
    uint32_t bandwidth; //!< Station bandwidth share in Mb/s
    uint32_t latency; //!< Latency in nanoseconds
    uint8_t ppod_id[16]; //!< Physical PoD Identifier (128-bit UUID)
    uint32_t ppod_size; //!< Physical PoD size
    uint32_t vpod_id; //!< Virtual PoD Identifier
    uint32_t vpod_size; //!< Virtual PoD size
    uint32_t vpod_active_accelerators[AMDSMI_FABRIC_ACTIVE_ACCELERATORS_BITMAP_SIZE];
    uint32_t local_accelerators[AMDSMI_FABRIC_MAX_LOCAL_GPUS]; //!< Local Accelerator IDs
    amdsmi_fabric_nht_address_mode_t addr_mode; //!< Source aliasing or identification mode
    amdsmi_fabric_accelerator_vpod_state_t accel_state; //!< Accelerator vPoD State
} amdsmi_fabric_info_v1_t;

typedef struct {
    uint32_t version;
    union {
        amdsmi_fabric_info_v1_t v1;
    };
} amdsmi_fabric_info_ver_t;

/**
 * @brief Fabric device information structure
 */
typedef struct {
    amdsmi_bdf_t bdf;      //!< BDF of the Fabric device
    amdsmi_fabric_info_ver_t info;
    uint32_t reserved[14];
} amdsmi_fabric_info_t;

// AMD SMI initialization flags
typedef enum {
    AMDSMI_INIT_ALL_PROCESSORS = 0xFFFFFFFF,  //!< Initialize all processors
    AMDSMI_INIT_AMD_CPUS       = (1 << 0),    //!< Initialize AMD CPUS
    AMDSMI_INIT_AMD_GPUS       = (1 << 1),    //!< Initialize AMD GPUS
    AMDSMI_INIT_NON_AMD_CPUS   = (1 << 2),    //!< Initialize Non-AMD CPUS
    AMDSMI_INIT_NON_AMD_GPUS   = (1 << 3),    //!< Initialize Non-AMD GPUS
    AMDSMI_INIT_AMD_APUS       = (AMDSMI_INIT_AMD_CPUS | AMDSMI_INIT_AMD_GPUS) /**< Initialize AMD CPUS and GPUS
                                                                                    (Default option) */
} amdsmi_init_flags_t;

/**
 * Structure to hold dynamically loaded AMD SMI function pointers
 */
struct AmdsmiLoader {
  void* amdsmi_handle;

  // Function pointers
  amdsmi_status_t (*init)(uint64_t init_flags);
  amdsmi_status_t (*shut_down)();
  amdsmi_status_t (*get_processor_handle_from_bdf)(const char* bdf, amdsmi_processor_handle* processor_handle);
  amdsmi_status_t (*get_gpu_fabric_info)(amdsmi_processor_handle processor_handle, amdsmi_fabric_info_t* fabric_info);

  AmdsmiLoader();
  ~AmdsmiLoader();

  int init_function_table();
  bool isLoaded() const;
};

}  // namespace rocshmem

#endif  // ROCSHMEM_AMDSMI_LOADER_HPP_
