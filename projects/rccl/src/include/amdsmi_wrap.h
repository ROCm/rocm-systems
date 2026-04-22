// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#ifndef AMDSMI_WRAP_H_
#define AMDSMI_WRAP_H_

#include <cstdint>
#include <ctime>

#include "nccl.h"

/*************************************************************************
 * UALoE Fabric Support
 *
 * The following types and functions provide support for AMD's UALoE
 * scale-up fabric technology.
 * When the full amdsmi fabric API is available in the system headers,
 * define AMDSMI_FABRIC_DIRECT=1 to use the header types/functions directly.
 * Otherwise, the compatibility types below will be used.
 ************************************************************************/
 // Set AMDSMI_DIRECT to 1 at build time if the required amdsmi headers (even before UALoE support) are available,
 // otherwise 0 to use compatibility types defined below

#if __has_include(<amd_smi/amdsmi.h>)
  #define AMDSMI_DIRECT 1
  #include <amd_smi/amdsmi.h>
#else
  #define AMDSMI_DIRECT 0
#endif

// Define when amdsmi fabric API is available in amdsmi headers, otherwise use compatibility types defined below
#ifndef AMDSMI_FABRIC_DIRECT
#define AMDSMI_FABRIC_DIRECT 0
#endif


// Added to facilitate builds when amdsmi headers are not available
// In these cases, the rsmi wrapper will be used by default
#if !AMDSMI_DIRECT
/*************************************************************************
 * Pre-UALoE AMDSMI Definitions
 ************************************************************************/
typedef enum {
    AMDSMI_FW_ID_SMU = 1,                   /**< System Management Unit (power management,
                                                 clock control, thermal monitoring, etc...) */
    AMDSMI_FW_ID_FIRST = AMDSMI_FW_ID_SMU,
    AMDSMI_FW_ID_CP_CE,                     //!< Compute Processor - Command_Engine (fetch, decode, dispatch)
    AMDSMI_FW_ID_CP_PFP,                    //!< Compute Processor - Pixel Front End Processor (pixelating process)
    AMDSMI_FW_ID_CP_ME,                     //!< Compute Processor - Micro Engine (specialize processing)
    AMDSMI_FW_ID_CP_MEC_JT1,                //!< Compute Processor - Micro Engine Controler Job Table 1 (queues, scheduling)
    AMDSMI_FW_ID_CP_MEC_JT2,                //!< Compute Processor - Micro Engine Controler Job Table 2 (queues, scheduling)
    AMDSMI_FW_ID_CP_MEC1,                   //!< Compute Processor - Micro Engine Controler 1 (scheduling, managing resources)
    AMDSMI_FW_ID_CP_MEC2,                   //!< Compute Processor - Micro Engine Controler 2 (scheduling, managing resources)
    AMDSMI_FW_ID_RLC,                       //!< Rasterizer and L2 Cache (rasterization processs)
    AMDSMI_FW_ID_SDMA0,                     //!< System Direct Memory Access 0 (high speed data transfers)
    AMDSMI_FW_ID_SDMA1,                     //!< System Direct Memory Access 1 (high speed data transfers)
    AMDSMI_FW_ID_SDMA2,                     //!< System Direct Memory Access 2 (high speed data transfers)
    AMDSMI_FW_ID_SDMA3,                     //!< System Direct Memory Access 3 (high speed data transfers)
    AMDSMI_FW_ID_SDMA4,                     //!< System Direct Memory Access 4 (high speed data transfers)
    AMDSMI_FW_ID_SDMA5,                     //!< System Direct Memory Access 5 (high speed data transfers)
    AMDSMI_FW_ID_SDMA6,                     //!< System Direct Memory Access 6 (high speed data transfers)
    AMDSMI_FW_ID_SDMA7,                     //!< System Direct Memory Access 7 (high speed data transfers)
    AMDSMI_FW_ID_VCN,                       //!< Video Core Next (encoding and decoding)
    AMDSMI_FW_ID_UVD,                       //!< Unified Video Decoder (decode specific video formats)
    AMDSMI_FW_ID_VCE,                       //!< Video Coding Engine (Encoding video)
    AMDSMI_FW_ID_ISP,                       //!< Image Signal Processor (processing raw image data from sensors)
    AMDSMI_FW_ID_DMCU_ERAM,                 //!< Digital Micro Controller Unit - Embedded RAM (memory used by DMU)
    AMDSMI_FW_ID_DMCU_ISR,                  //!< Digital Micro Controller Unit - Interrupt Service Routine (interrupt handlers)
    AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM,  //!< Rasterizier and L2 Cache Restore List Graphics Processor Memory
    AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM,  //!< Rasterizier and L2 Cache Restore List System RAM Memory
    AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL,     //!< Rasterizier and L2 Cache Restore List Control
    AMDSMI_FW_ID_RLC_V,                     //!< Rasterizier and L2 Cache Virtual memory
    AMDSMI_FW_ID_MMSCH,                     //!< Multi-Media Shader Hardware Scheduler
    AMDSMI_FW_ID_PSP_SYSDRV,                //!< Platform Security Processor System Driver
    AMDSMI_FW_ID_PSP_SOSDRV,                //!< Platform Security Processor Secure Operating System Driver
    AMDSMI_FW_ID_PSP_TOC,                   //!< Platform Security Processor Table of Contents
    AMDSMI_FW_ID_PSP_KEYDB,                 //!< Platform Security Processor Table of Contents
    AMDSMI_FW_ID_DFC,                       //!< Data Fabric Controler (bandwidth and coherency)
    AMDSMI_FW_ID_PSP_SPL,                   //!< Platform Security Processor Secure Program Loader
    AMDSMI_FW_ID_DRV_CAP,                   //!< Driver Capabilities (capabilities, features)
    AMDSMI_FW_ID_MC,                        //!< Memory Contoller (RAM and VRAM)
    AMDSMI_FW_ID_PSP_BL,                    //!< Platform Security Processor Bootloader (initial firmware)
    AMDSMI_FW_ID_CP_PM4,                    //!< Compute Processor Packet Processor 4 (processing command packets)
    AMDSMI_FW_ID_RLC_P,                     //!< Rasterizier and L2 Cache Partition
    AMDSMI_FW_ID_SEC_POLICY_STAGE2,         //!< Security Policy Stage 2 (security features)
    AMDSMI_FW_ID_REG_ACCESS_WHITELIST,      //!< Register Access Whitelist (Prevent unathorizied access)
    AMDSMI_FW_ID_IMU_DRAM,                  //!< Input/Output Memory Management Unit - Dynamic RAM
    AMDSMI_FW_ID_IMU_IRAM,                  //!< Input/Output Memory Management Unit - Instruction RAM
    AMDSMI_FW_ID_SDMA_TH0,                  //!< System Direct Memory Access - Thread Handler 0
    AMDSMI_FW_ID_SDMA_TH1,                  //!< System Direct Memory Access - Thread Handler 1
    AMDSMI_FW_ID_CP_MES,                    //!< Compute Processor - Micro Engine Scheduler
    AMDSMI_FW_ID_MES_KIQ,                   //!< Micro Engine Scheduler - Kernel Indirect Queue
    AMDSMI_FW_ID_MES_STACK,                 //!< Micro Engine Scheduler - Stack
    AMDSMI_FW_ID_MES_THREAD1,               //!< Micro Engine Scheduler - Thread 1
    AMDSMI_FW_ID_MES_THREAD1_STACK,         //!< Micro Engine Scheduler - Thread 1 Stack
    AMDSMI_FW_ID_RLX6,                      //!< Hardware Block RLX6
    AMDSMI_FW_ID_RLX6_DRAM_BOOT,            //!< Hardware Block RLX6 - Dynamic Ram Boot
    AMDSMI_FW_ID_RS64_ME,                   //!< Hardware Block RS64 - Micro Engine
    AMDSMI_FW_ID_RS64_ME_P0_DATA,           //!< Hardware Block RS64 - Micro Engine Partition 0 Data
    AMDSMI_FW_ID_RS64_ME_P1_DATA,           //!< Hardware Block RS64 - Micro Engine Partition 1 Data
    AMDSMI_FW_ID_RS64_PFP,                  //!< Hardware Block RS64 - Pixel Front End Processor
    AMDSMI_FW_ID_RS64_PFP_P0_DATA,          //!< Hardware Block RS64 - Pixel Front End Processor Partition 0 Data
    AMDSMI_FW_ID_RS64_PFP_P1_DATA,          //!< Hardware Block RS64 - Pixel Front End Processor Partition 1 Data
    AMDSMI_FW_ID_RS64_MEC,                  //!< Hardware Block RS64 - Micro Engine Controller
    AMDSMI_FW_ID_RS64_MEC_P0_DATA,          //!< Hardware Block RS64 - Micro Engine Controller Partition 0 Data
    AMDSMI_FW_ID_RS64_MEC_P1_DATA,          //!< Hardware Block RS64 - Micro Engine Controller Partition 1 Data
    AMDSMI_FW_ID_RS64_MEC_P2_DATA,          //!< Hardware Block RS64 - Micro Engine Controller Partition 2 Data
    AMDSMI_FW_ID_RS64_MEC_P3_DATA,          //!< Hardware Block RS64 - Micro Engine Controller Partition 3 Data
    AMDSMI_FW_ID_PPTABLE,                   //!< Power Policy Table (power management policies)
    AMDSMI_FW_ID_PSP_SOC,                   //!< Platform Security Processor - System On a Chip
    AMDSMI_FW_ID_PSP_DBG,                   //!< Platform Security Processor - Debug
    AMDSMI_FW_ID_PSP_INTF,                  //!< Platform Security Processor - Interface
    AMDSMI_FW_ID_RLX6_CORE1,                //!< Hardware Block RLX6 - Core 1
    AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1,      //!< Hardware Block RLX6 Core 1 - Dynamic RAM Boot
    AMDSMI_FW_ID_RLCV_LX7,                  //!< Hardware Block RLCV - Subsystem LX7
    AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST,     //!< Rasterizier and L2 Cache - Save Restore List
    AMDSMI_FW_ID_ASD,                       //!< Asynchronous Shader Dispatcher
    AMDSMI_FW_ID_TA_RAS,                    //!< Trusted Applications - Reliablity Availability and Serviceability
    AMDSMI_FW_ID_TA_XGMI,                   //!< Trusted Applications - Reliablity XGMI
    AMDSMI_FW_ID_RLC_SRLG,                  //!< Rasterizier and L2 Cache - Shared Resource Local Group
    AMDSMI_FW_ID_RLC_SRLS,                  //!< Rasterizier and L2 Cache - Shared Resource Local Segment
    AMDSMI_FW_ID_PM,                        //!< Power Management Firmware
    AMDSMI_FW_ID_DMCU,                      //!< Display Micro-Controller Unit
    AMDSMI_FW_ID_PLDM_BUNDLE,               //!< Platform Level Data Model Firmware Bundle
    AMDSMI_FW_ID__MAX
} amdsmi_fw_block_t;

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

typedef struct {
    uint8_t num_fw_info;
    struct fw_info_list_ {
        amdsmi_fw_block_t fw_id;
        uint64_t fw_version;
        uint64_t reserved[2];
    } fw_info_list[AMDSMI_FW_ID__MAX];
    uint32_t reserved[7];
} amdsmi_fw_info_t;

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

#define AMDSMI_MAX_STRING_LENGTH          256  //!< Maximum length for string buffers

typedef void *amdsmi_processor_handle;
typedef void *amdsmi_socket_handle;
typedef struct {
    uint32_t drm_render; //!< the render node under /sys/class/drm/renderD*
    uint32_t drm_card;   //!< the graphic card device under /sys/class/drm/card*
    uint32_t hsa_id;     //!< the HSA enumeration ID
    uint32_t hip_id;     //!< the HIP enumeration ID
    char hip_uuid[AMDSMI_MAX_STRING_LENGTH];  //!< the HIP unique identifer
} amdsmi_enumeration_info_t;

typedef enum {
    AMDSMI_PROCESSOR_TYPE_UNKNOWN = 0,   //!< Unknown processor type
    AMDSMI_PROCESSOR_TYPE_AMD_GPU,       //!< AMD Graphics processor type
    AMDSMI_PROCESSOR_TYPE_AMD_CPU,       //!< AMD CPU processor type, a physical component that holds the CPU
    AMDSMI_PROCESSOR_TYPE_NON_AMD_GPU,   //!< Non-AMD Graphics processor type
    AMDSMI_PROCESSOR_TYPE_NON_AMD_CPU,   //!< Non-AMD CPU processor type
    AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE,  //!< AMD CPU-Core processor type, individual processing units within the CPU
    AMDSMI_PROCESSOR_TYPE_AMD_APU,       //!< AMD Accelerated processor type, GPU and CPU on a single die
    AMDSMI_PROCESSOR_TYPE_AMD_NIC        //!< AMD Network Interface Card processor type
} processor_type_t;

typedef enum {
    AMDSMI_LINK_TYPE_INTERNAL = 0,        //!< Internal Link Type, within chip
    AMDSMI_LINK_TYPE_PCIE = 1,            //!< Peripheral Component Interconnect Express Link Type
    AMDSMI_LINK_TYPE_XGMI = 2,            //!< GPU Memory Interconnect (multi GPU communication)
    AMDSMI_LINK_TYPE_NOT_APPLICABLE = 3,  //!< Not Applicable Link Type
    AMDSMI_LINK_TYPE_UNKNOWN = 4          //!< Unknown Link Type
} amdsmi_link_type_t;

typedef struct {
    uint32_t major;     //!< Major version
    uint32_t minor;     //!< Minor version
    uint32_t release;   //!< Patch, build or stepping version
    const char *build;  //!< Full Build version string
} amdsmi_version_t;

typedef enum {
    AMDSMI_INIT_ALL_PROCESSORS = 0xFFFFFFFF,  //!< Initialize all processors
    AMDSMI_INIT_AMD_CPUS       = (1 << 0),    //!< Initialize AMD CPUS
    AMDSMI_INIT_AMD_GPUS       = (1 << 1),    //!< Initialize AMD GPUS
    AMDSMI_INIT_NON_AMD_CPUS   = (1 << 2),    //!< Initialize Non-AMD CPUS
    AMDSMI_INIT_NON_AMD_GPUS   = (1 << 3),    //!< Initialize Non-AMD GPUS
    AMDSMI_INIT_AMD_APUS       = (AMDSMI_INIT_AMD_CPUS | AMDSMI_INIT_AMD_GPUS) /**< Initialize AMD CPUS and GPUS
                                                                                    (Default option) */
} amdsmi_init_flags_t;

amdsmi_status_t
amdsmi_init(uint64_t init_flags);
amdsmi_status_t
amdsmi_status_code_to_string(amdsmi_status_t status, const char **status_string);
amdsmi_status_t
amdsmi_get_socket_handles(uint32_t *socket_count, amdsmi_socket_handle* socket_handles);
amdsmi_status_t
amdsmi_get_processor_handles(amdsmi_socket_handle socket_handle,
                                    uint32_t *processor_count,
                                    amdsmi_processor_handle* processor_handles);
amdsmi_status_t
amdsmi_get_processor_type(amdsmi_processor_handle processor_handle, processor_type_t* processor_type);

amdsmi_status_t
amdsmi_get_gpu_enumeration_info(amdsmi_processor_handle processor_handle, amdsmi_enumeration_info_t *info);

amdsmi_status_t
amdsmi_get_gpu_bdf_id(amdsmi_processor_handle processor_handle, uint64_t *bdfid);

amdsmi_status_t
amdsmi_get_processor_handle_from_bdf(amdsmi_bdf_t bdf, amdsmi_processor_handle* processor_handle);

amdsmi_status_t
amdsmi_topo_get_link_type(amdsmi_processor_handle processor_handle_src,
                          amdsmi_processor_handle processor_handle_dst,
                          uint64_t *hops, amdsmi_link_type_t *type);
amdsmi_status_t
amdsmi_topo_get_link_weight(amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst,
                            uint64_t *weight);
amdsmi_status_t
amdsmi_get_minmax_bandwidth_between_processors(amdsmi_processor_handle processor_handle_src,
                                                amdsmi_processor_handle processor_handle_dst,
                                                uint64_t *min_bandwidth,
                                                uint64_t *max_bandwidth);

#endif // !AMDSMI_DIRECT


#if !AMDSMI_FABRIC_DIRECT
/*****************************************************************************/
/** @defgroup rcclFabricCompat Fabric Compatibility Types
 *  Compatibility types for UALoE fabric support when amdsmi fabric API
 *  is not yet available amdsmi headers.
 *  @{
 */

/**
 * @brief Fabric telemetry categories
 */
typedef enum {
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_UNKNOWN         = -1,
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE           = 0, //!< UALOE telemetry
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_SWITCH          = 1, //!< Switch telemetry
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_CRYPTO          = 2, //!< Crypto telemetry
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_PFC             = 3, //!< PFC telemetry
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT         = 4, //!< Network Port telemetry
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_DERIVED_UALOE   = 5, //!< Derived UALOE telemetry
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_DERIVED_NETPORT = 6, //!< Derived Network Port telemetry
    AMDSMI_FABRIC_TELEMETRY_CATEGORY_MAX             = 7  //!< Maximum number of categories
} amdsmi_fabric_telemetry_category_t;

/**
 * @brief Fabric telemetry category bitmask constructor
 */
#define AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK(cat) (1U << AMDSMI_FABRIC_TELEMETRY_CATEGORY_ ## cat)

/**
 * @brief Fabric telemetry item structure
 */
typedef struct {
    uint64_t id;      //!< Identifier of the telemetry item
    uint64_t value;   //!< Value of the telemetry item
} amdsmi_fabric_telemetry_item_t;

/**
 * @brief Fabric textual label structure
 */
typedef struct {
    char text[32];  //!< Textual label content
} amdsmi_fabric_label_t;

/**
 * @brief Fabric telemetry instance structure
 */
typedef struct {
    amdsmi_fabric_label_t name;               //!< Name for this instance
    unsigned logical_idx;                     //!< Logical index for this instance
    unsigned item_count;                      //!< Number of telemetry items in the set
    amdsmi_fabric_telemetry_item_t items[];   //!< Pointer to array of telemetry items
} amdsmi_fabric_telemetry_instance_t;

/**
 * @brief Fabric telemetry dataset structure
 */
typedef struct {
    amdsmi_fabric_telemetry_category_t category;  //!< Telemetry category
    uint64_t generation_count;                    //!< Sequence number incremented each time telemetry is written
    struct timespec timestamp;                    //!< UTC timestamp seconds since epoch
    unsigned instance_count;                      //!< Number of instances for this category
    amdsmi_fabric_telemetry_instance_t *instances[]; //!< Array of pointers to instances
} amdsmi_fabric_telemetry_dataset_t;

/**
 * @brief Fabric telemetry structure
 */
typedef struct {
    amdsmi_fabric_telemetry_dataset_t *datasets[AMDSMI_FABRIC_TELEMETRY_CATEGORY_MAX];
} amdsmi_fabric_telemetry_t;

#define AMDSMI_FABRIC_ACTIVE_ACCELERATORS_BITMAP_SIZE 32
#define AMDSMI_FABRIC_MAX_LOCAL_GPUS 8

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

/*************************************************************************
 * UALoE Fabric Functions
 ************************************************************************/

/**
 *  @brief Get Fabric device information
 *
 *  @ingroup tagFabric
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @details This function retrieves Fabric device information for the specified
 *  processor handle, including the BDF (Bus, Device, Function) identifier.
 *
 *  @param[in] processor_handle - Handle for the target processor
 *
 *  @param[out] info - Pointer to Fabric information structure to be populated.
 *  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_fabric_info(amdsmi_processor_handle processor_handle,
                                           amdsmi_fabric_info_t* info);

/**
 *  @brief Get string name for a telemetry item ID
 *
 *  @ingroup tagFabric
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @details Given a telemetry item ID @p telem_id,
 *  this function returns a pointer to a string containing the human-readable name
 *  for the specified telemetry item. The returned string is statically allocated
 *  and should not be freed by the caller.
 *
 *
 *  @param[in] telem_id The telemetry item ID for which the name is requested
 *
 *  @return const char* | Pointer to string containing the telemetry item name,
 *  or UNKNOWN if the category or telemetry ID is not recognized
 */
const char* amdsmi_fabric_telem_id_to_string(uint64_t telem_id);

/** @} End rcclFabricCompat */
#endif // AMDSMI_FABRIC_DIRECT


/*************************************************************************
 * AMD SMI Fabric Info Cache
 *
 * Similar to ncclNvmlDeviceInfo for NVML, this structure caches fabric
 * information per device for quick access during topology discovery.
 ************************************************************************/

#define AMDSMI_FABRIC_INFO_VERSION_1 1
#define AMDSMI_FABRIC_INFO_CURRENT_VERSION AMDSMI_FABRIC_INFO_VERSION_1

struct amdsmiFabricDeviceInfo {
    bool fabricSupported;                           //!< Whether UALoE fabric is available
    amdsmi_fabric_type_t fabricType;                //!< Fabric type (UALOE or UALLINK)
    amdsmi_fabric_accelerator_vpod_state_t state;   //!< Accelerator vPOD state
    uint32_t acceleratorId;                         //!< Accelerator ID in fabric
    uint32_t bandwidth;                             //!< Bandwidth in Mb/s
    uint32_t latency;                               //!< Latency in nanoseconds
    uint8_t  clusterUuid[16];                       //!< Physical POD ID (128-bit UUID)
    uint32_t ppodSize;                              //!< Physical POD size
    uint32_t cliqueId;                              //!< Virtual POD ID
    uint32_t vpodSize;                              //!< Virtual POD size
};

constexpr int amdsmiFabricMaxDevices = 32;
extern int amdsmiFabricDeviceCount;
extern struct amdsmiFabricDeviceInfo amdsmiFabricDevices[amdsmiFabricMaxDevices];

/*************************************************************************
 * Existing AMD SMI Wrapper Functions
 ************************************************************************/

ncclResult_t amd_smi_init();
ncclResult_t amd_smi_shutdown();
ncclResult_t amd_smi_getNumDevice(uint32_t* num_devs);
ncclResult_t amd_smi_getDevicePciBusIdString(uint32_t deviceIndex, char* pciBusId, size_t len);
ncclResult_t amd_smi_getDeviceIndexByPciBusId(const char* pciBusId, uint32_t* deviceIndex);
ncclResult_t amd_smi_getLinkInfo(int srcDev, int dstDev, amdsmi_link_type_t* type, int *hops, int *count);
ncclResult_t amd_smi_getFirmwareVersion(uint32_t deviceIndex, uint64_t* fwVersion);
/*************************************************************************
 * UALoE Fabric Wrapper Functions
 *
 * These functions wrap the AMD SMI fabric APIs, following the pattern
 * established by nvmlwrap.h for NVML. They provide:
 * - Graceful fallback when fabric is not supported
 * - Error conversion to ncclResult_t
 * - Thread-safe access to fabric information
 ************************************************************************/

/**
 * @brief Ensure fabric subsystem is initialized and info is cached
 *
 * Similar to ncclNvmlEnsureInitialized() for NVML. Initializes the fabric
 * subsystem and caches fabric info for all devices. Thread-safe.
 *
 * @return ncclResult_t ncclSuccess on success
 */
ncclResult_t amd_smi_ensureFabricInitialized();

/**
 * @brief Check if UALoE fabric is supported and active on the system
 *
 * @param[in] deviceIndex GPU device index
 * @param[out] supported Set to true if UALoE fabric is available and active
 * @return ncclResult_t ncclSuccess on success
 */
ncclResult_t amd_smi_isFabricSupported(uint32_t deviceIndex, bool* supported);

/**
 * @brief Get cached fabric device info
 *
 * Returns cached fabric information for quick access during topology discovery.
 * Must call amd_smi_ensureFabricInitialized() first.
 *
 * @param[in] deviceIndex GPU device index
 * @param[out] info Pointer to fabric device info structure
 * @return ncclResult_t ncclSuccess on success
 */
ncclResult_t amd_smi_getFabricDeviceInfo(uint32_t deviceIndex, struct amdsmiFabricDeviceInfo* info);

/**
 * @brief Allocate storage for fabric telemetry data
 *
 * @param[in] deviceIndex GPU device index
 * @param[in] categoryMask Bitmask of telemetry categories
 * @param[out] telemetry Pointer to allocated telemetry structure
 * @return ncclResult_t ncclSuccess on success
 */
ncclResult_t amd_smi_allocFabricTelemetry(uint32_t deviceIndex,
                                          uint32_t categoryMask,
                                          amdsmi_fabric_telemetry_t** telemetry);

/**
 * @brief Get fabric telemetry data
 *
 * @param[in] deviceIndex GPU device index
 * @param[in,out] telemetry Pre-allocated telemetry structure to populate
 * @return ncclResult_t ncclSuccess on success
 */
ncclResult_t amd_smi_getFabricTelemetryData(uint32_t deviceIndex,
                                            amdsmi_fabric_telemetry_t* telemetry);

/**
 * @brief Free fabric telemetry storage
 *
 * @param[in] deviceIndex GPU device index
 * @param[in] telemetry Telemetry structure to free
 * @return ncclResult_t ncclSuccess on success
 */
ncclResult_t amd_smi_freeFabricTelemetry(uint32_t deviceIndex,
                                         amdsmi_fabric_telemetry_t* telemetry);

/**
 * @brief Get string name for a telemetry item ID
 *
 * @param[in] telemId Telemetry item ID
 * @return const char* String name or "UNKNOWN"
 */
const char* amd_smi_fabricTelemIdToString(uint64_t telemId);

/**
 * @brief Get fabric bandwidth for XGMI speed calculations
 *
 * Returns the fabric-reported bandwidth if UALoE is available,
 * otherwise returns 0 to indicate fallback to arch-based defaults.
 *
 * @param[in] deviceIndex GPU device index
 * @param[out] bandwidthMbps Bandwidth in Mb/s (0 if fabric not available)
 * @return ncclResult_t ncclSuccess on success
 */
ncclResult_t amd_smi_getFabricBandwidth(uint32_t deviceIndex, uint32_t* bandwidthMbps);
#endif // AMDSMI_WRAP_H_
