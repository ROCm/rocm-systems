/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */
#ifndef INCLUDE_AMDSMI_H_
#define INCLUDE_AMDSMI_H_

/**
 * @file amdsmi.h
 * @brief AMD System Management Interface API
 */

#include <stdlib.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#include <cstdint>
#ifdef ENABLE_ESMI_LIB
    #include <e_smi/e_smi.h>
#endif

#else
#include <stdint.h>
#endif  // __cplusplus

/**
 * @brief Initialization flags
 *
 * Initialization flags may be OR'd together and passed to ::amdsmi_init().
 */
typedef enum {
  AMDSMI_INIT_ALL_PROCESSORS = 0xFFFFFFFF,  //!< Initialize all processors
  AMDSMI_INIT_AMD_CPUS = (1 << 0),
  AMDSMI_INIT_AMD_GPUS = (1 << 1),
  AMDSMI_INIT_NON_AMD_CPUS = (1 << 2),
  AMDSMI_INIT_NON_AMD_GPUS = (1 << 3),
  AMDSMI_INIT_AMD_APUS = (AMDSMI_INIT_AMD_CPUS | AMDSMI_INIT_AMD_GPUS) // Default option
} amdsmi_init_flags_t;

/* Maximum size definitions AMDSMI */
#define AMDSMI_MAX_MM_IP_COUNT        8
#define AMDSMI_MAX_DATE_LENGTH        32 /**< YYYY-MM-DD:HH:MM:SS.MSC */
#define AMDSMI_MAX_STRING_LENGTH      64
#define AMDSMI_NORMAL_STRING_LENGTH   32
#define AMDSMI_MAX_DEVICES            32
#define AMDSMI_MAX_NAME               32
#define AMDSMI_MAX_DRIVER_VERSION_LENGTH 80
#define AMDSMI_256_LENGTH 256
#define AMDSMI_MAX_CONTAINER_TYPE    2
#define AMDSMI_MAX_CACHE_TYPES 10
#define AMDSMI_MAX_NUM_XGMI_PHYSICAL_LINK 64
#define AMDSMI_MAX_ACCELERATOR_PROFILE    32
#define AMDSMI_MAX_CP_PROFILE_RESOURCES   32
#define AMDSMI_MAX_ACCELERATOR_PARTITIONS 8

#define AMDSMI_GPU_UUID_SIZE 38

/**
 * @brief The following structure holds the gpu metrics values for a device.
 */

/**
 * @brief Unit conversion factor for HBM temperatures
 */
#define CENTRIGRADE_TO_MILLI_CENTIGRADE 1000

/**
 * @brief This should match NUM_HBM_INSTANCES
 */
#define AMDSMI_NUM_HBM_INSTANCES 4

/**
 * @brief This should match MAX_NUM_VCN
 */
#define AMDSMI_MAX_NUM_VCN 4

/**
 * @brief This should match MAX_NUM_CLKS
 */
#define AMDSMI_MAX_NUM_CLKS 4

/**
 * @brief This should match MAX_NUM_XGMI_LINKS
 */
#define AMDSMI_MAX_NUM_XGMI_LINKS 8

/**
 * @brief This should match MAX_NUM_GFX_CLKS
 */
#define AMDSMI_MAX_NUM_GFX_CLKS 8

/**
 * @brief This should match AMDSMI_MAX_AID
 */
#define AMDSMI_MAX_AID 4

/**
 * @brief This should match AMDSMI_MAX_ENGINES
 */
#define AMDSMI_MAX_ENGINES 8

/**
 * @brief This should match AMDSMI_MAX_NUM_JPEG (8*4=32)
 */
#define AMDSMI_MAX_NUM_JPEG 32

/**
 * @brief This should match AMDSMI_MAX_NUM_XCC;
 * XCC - Accelerated Compute Core, the collection of compute units,
 * ACE (Asynchronous Compute Engines), caches,
 * and global resources organized as one unit.
 *
 * Refer to amd.com documentation for more detail:
 * https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
 */
#define AMDSMI_MAX_NUM_XCC 8

/**
 * @brief This should match AMDSMI_MAX_NUM_XCP;
 * XCP - Accelerated Compute Processor,
 * also referred to as the Graphics Compute Partitions.
 * Each physical gpu could have a maximum of 8 separate partitions
 * associated with each (depending on ASIC support).
 *
 * Refer to amd.com documentation for more detail:
 * https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
 */
#define AMDSMI_MAX_NUM_XCP 8

/* string format */
#define AMDSMI_TIME_FORMAT "%02d:%02d:%02d.%03d"
#define AMDSMI_DATE_FORMAT "%04d-%02d-%02d:%02d:%02d:%02d.%03d"

/**
 * @brief library versioning
 */

//! Year should follow the IP driver package version: 22.40/23.10 and similar
#define AMDSMI_LIB_VERSION_YEAR 24

//! Major version should be changed for every header change (adding/deleting APIs, changing names, fields of structures, etc.)
#define AMDSMI_LIB_VERSION_MAJOR 7

//! Minor version should be updated for each API change, but without changing headers
#define AMDSMI_LIB_VERSION_MINOR 0

//! Release version should be set to 0 as default and can be updated by the PMs for each CSP point release
#define AMDSMI_LIB_VERSION_RELEASE 0

#define AMDSMI_LIB_VERSION_CREATE_STRING(YEAR, MAJOR, MINOR, RELEASE) (#YEAR "." #MAJOR "." #MINOR "." #RELEASE)
#define AMDSMI_LIB_VERSION_EXPAND_PARTS(YEAR_STR, MAJOR_STR, MINOR_STR, RELEASE_STR) AMDSMI_LIB_VERSION_CREATE_STRING(YEAR_STR, MAJOR_STR, MINOR_STR, RELEASE_STR)
#define AMDSMI_LIB_VERSION_STRING AMDSMI_LIB_VERSION_EXPAND_PARTS(AMDSMI_LIB_VERSION_YEAR, AMDSMI_LIB_VERSION_MAJOR, AMDSMI_LIB_VERSION_MINOR, AMDSMI_LIB_VERSION_RELEASE)

typedef enum {
  AMDSMI_MM_UVD,
  AMDSMI_MM_VCE,
  AMDSMI_MM_VCN,
  AMDSMI_MM__MAX
} amdsmi_mm_ip_t;

typedef enum {
  AMDSMI_CONTAINER_LXC,
  AMDSMI_CONTAINER_DOCKER,
} amdsmi_container_types_t;

//! opaque handler point to underlying implementation
typedef void *amdsmi_processor_handle;
typedef void *amdsmi_socket_handle;
#ifdef ENABLE_ESMI_LIB
typedef void *amdsmi_cpusocket_handle;
#endif

/**
 * @brief Processor types detectable by AMD SMI
 * AMDSMI_PROCESSOR_TYPE_AMD_CPU      - CPU Socket is a physical component that holds the CPU.
 * AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE - CPU Cores are number of individual processing units within the CPU.
 * AMDSMI_PROCESSOR_TYPE_AMD_APU      - Combination of AMDSMI_PROCESSOR_TYPE_AMD_CPU and integrated GPU on single die
 */
typedef enum {
  AMDSMI_PROCESSOR_TYPE_UNKNOWN = 0,
  AMDSMI_PROCESSOR_TYPE_AMD_GPU,
  AMDSMI_PROCESSOR_TYPE_AMD_CPU,
  AMDSMI_PROCESSOR_TYPE_NON_AMD_GPU,
  AMDSMI_PROCESSOR_TYPE_NON_AMD_CPU,
  AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE,
  AMDSMI_PROCESSOR_TYPE_AMD_APU
} processor_type_t;

/**
 * @brief Error codes returned by amdsmi functions
 */
// Please avoid status codes that are multiples of 256 (256, 512, etc..)
// Return values in the shell get modulo 256 applied, meaning any multiple of 256 ends up as 0
typedef enum {
    AMDSMI_STATUS_SUCCESS = 0,  //!< Call succeeded
    // Library usage errors
    AMDSMI_STATUS_INVAL = 1,  //!< Invalid parameters
    AMDSMI_STATUS_NOT_SUPPORTED = 2,  //!< Command not supported
    AMDSMI_STATUS_NOT_YET_IMPLEMENTED = 3,  //!< Not implemented yet
    AMDSMI_STATUS_FAIL_LOAD_MODULE = 4,  //!< Fail to load lib
    AMDSMI_STATUS_FAIL_LOAD_SYMBOL = 5,  //!< Fail to load symbol
    AMDSMI_STATUS_DRM_ERROR = 6,  //!< Error when call libdrm
    AMDSMI_STATUS_API_FAILED = 7,  //!< API call failed
    AMDSMI_STATUS_TIMEOUT = 8,  //!< Timeout in API call
    AMDSMI_STATUS_RETRY = 9,  //!< Retry operation
    AMDSMI_STATUS_NO_PERM = 10,  //!< Permission Denied
    AMDSMI_STATUS_INTERRUPT = 11,  //!< An interrupt occurred during execution of function
    AMDSMI_STATUS_IO = 12,  //!< I/O Error
    AMDSMI_STATUS_ADDRESS_FAULT = 13,  //!< Bad address
    AMDSMI_STATUS_FILE_ERROR = 14,  //!< Problem accessing a file
    AMDSMI_STATUS_OUT_OF_RESOURCES = 15,  //!< Not enough memory
    AMDSMI_STATUS_INTERNAL_EXCEPTION = 16,  //!< An internal exception was caught
    AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS = 17,  //!< The provided input is out of allowable or safe range
    AMDSMI_STATUS_INIT_ERROR = 18,  //!< An error occurred when initializing internal data structures
    AMDSMI_STATUS_REFCOUNT_OVERFLOW = 19,  //!< An internal reference counter exceeded INT32_MAX
    // Device related errors
    AMDSMI_STATUS_BUSY = 30,  //!< Device busy
    AMDSMI_STATUS_NOT_FOUND = 31,  //!< Device Not found
    AMDSMI_STATUS_NOT_INIT = 32,  //!< Device not initialized
    AMDSMI_STATUS_NO_SLOT = 33,  //!< No more free slot
    AMDSMI_STATUS_DRIVER_NOT_LOADED = 34,  //!< Processor driver not loaded
    // Data and size errors
    AMDSMI_STATUS_NO_DATA = 40,  //!< No data was found for a given input
    AMDSMI_STATUS_INSUFFICIENT_SIZE = 41,  //!< Not enough resources were available for the operation
    AMDSMI_STATUS_UNEXPECTED_SIZE = 42,  //!< An unexpected amount of data was read
    AMDSMI_STATUS_UNEXPECTED_DATA = 43,  //!< The data read or provided to function is not what was expected
    //esmi errors
    AMDSMI_STATUS_NON_AMD_CPU = 44, //!< System has different cpu than AMD
    AMDSMI_STATUS_NO_ENERGY_DRV = 45, //!< Energy driver not found
    AMDSMI_STATUS_NO_MSR_DRV = 46, //!< MSR driver not found
    AMDSMI_STATUS_NO_HSMP_DRV = 47, //!< HSMP driver not found
    AMDSMI_STATUS_NO_HSMP_SUP = 48, //!< HSMP not supported
    AMDSMI_STATUS_NO_HSMP_MSG_SUP = 49, //!< HSMP message/feature not supported
    AMDSMI_STATUS_HSMP_TIMEOUT = 50,  //!< HSMP message is timedout
    AMDSMI_STATUS_NO_DRV = 51,  //!< No Energy and HSMP driver present
    AMDSMI_STATUS_FILE_NOT_FOUND = 52, //!< file or directory not found
    AMDSMI_STATUS_ARG_PTR_NULL = 53,   //!< Parsed argument is invalid
    AMDSMI_STATUS_AMDGPU_RESTART_ERR = 54, //!< AMDGPU restart failed
    AMDSMI_STATUS_SETTING_UNAVAILABLE = 55, //!< Setting is not available

    // General errors
    AMDSMI_STATUS_MAP_ERROR = 0xFFFFFFFE,  //!< The internal library error did not map to a status code
    AMDSMI_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF,  //!< An unknown error occurred
} amdsmi_status_t;

/**
 * @brief Clock types
 */
typedef enum {
  AMDSMI_CLK_TYPE_SYS = 0x0,   //!< System clock
  AMDSMI_CLK_TYPE_FIRST = AMDSMI_CLK_TYPE_SYS,
  AMDSMI_CLK_TYPE_GFX = AMDSMI_CLK_TYPE_SYS,
  AMDSMI_CLK_TYPE_DF,    //!< Data Fabric clock (for ASICs
        //!< running on a separate clock)
  AMDSMI_CLK_TYPE_DCEF,  //!< Display Controller Engine clock
  AMDSMI_CLK_TYPE_SOC,
  AMDSMI_CLK_TYPE_MEM,
  AMDSMI_CLK_TYPE_PCIE,
  AMDSMI_CLK_TYPE_VCLK0,
  AMDSMI_CLK_TYPE_VCLK1,
  AMDSMI_CLK_TYPE_DCLK0,
  AMDSMI_CLK_TYPE_DCLK1,
  AMDSMI_CLK_TYPE__MAX = AMDSMI_CLK_TYPE_DCLK1
} amdsmi_clk_type_t;

/**
 * @brief Accelerator Partition. This enum is used to identify
 * various accelerator partitioning settings.
 */
typedef enum {
  AMDSMI_ACCELERATOR_PARTITION_INVALID = 0,
  AMDSMI_ACCELERATOR_PARTITION_SPX,        //!< Single GPU mode (SPX)- All XCCs work
                                       //!< together with shared memory
  AMDSMI_ACCELERATOR_PARTITION_DPX,        //!< Dual GPU mode (DPX)- Half XCCs work
                                       //!< together with shared memory
  AMDSMI_ACCELERATOR_PARTITION_TPX,        //!< Triple GPU mode (TPX)- One-third XCCs
                                       //!< work together with shared memory
  AMDSMI_ACCELERATOR_PARTITION_QPX,        //!< Quad GPU mode (QPX)- Quarter XCCs
                                       //!< work together with shared memory
  AMDSMI_ACCELERATOR_PARTITION_CPX,        //!< Core mode (CPX)- Per-chip XCC with
                                       //!< shared memory
} amdsmi_accelerator_partition_type_t;

/**
 * @brief Compute Partition. This enum is used to identify
 * various compute partitioning settings.
 */
typedef enum {
  AMDSMI_COMPUTE_PARTITION_INVALID = 0,
  AMDSMI_COMPUTE_PARTITION_SPX,  //!< Single GPU mode (SPX)- All XCCs work
                                 //!< together with shared memory
  AMDSMI_COMPUTE_PARTITION_DPX,  //!< Dual GPU mode (DPX)- Half XCCs work
                                 //!< together with shared memory
  AMDSMI_COMPUTE_PARTITION_TPX,  //!< Triple GPU mode (TPX)- One-third XCCs
                                 //!< work together with shared memory
  AMDSMI_COMPUTE_PARTITION_QPX,   //!< Quad GPU mode (QPX)- Quarter XCCs
                                 //!< work together with shared memory
  AMDSMI_COMPUTE_PARTITION_CPX,  //!< Core mode (CPX)- Per-chip XCC with
                                 //!< shared memory
} amdsmi_compute_partition_type_t;

/**
 * @brief Memory Partitions. This enum is used to identify various
 * memory partition types.
 */
typedef enum {
  AMDSMI_MEMORY_PARTITION_UNKNOWN = 0,
  AMDSMI_MEMORY_PARTITION_NPS1,  //!< NPS1 - All CCD & XCD data is interleaved
                          //!< accross all 8 HBM stacks (all stacks/1).
  AMDSMI_MEMORY_PARTITION_NPS2,  //!< NPS2 - 2 sets of CCDs or 4 XCD interleaved
                          //!< accross the 4 HBM stacks per AID pair
                          //!< (8 stacks/2).
  AMDSMI_MEMORY_PARTITION_NPS4,  //!< NPS4 - Each XCD data is interleaved accross
                          //!< accross 2 (or single) HBM stacks
                          //!< (8 stacks/8 or 8 stacks/4).
  AMDSMI_MEMORY_PARTITION_NPS8,  //!< NPS8 - Each XCD uses a single HBM stack
                          //!< (8 stacks/8). Or each XCD uses a single
                          //!< HBM stack & CCDs share 2 non-interleaved
                          //!< HBM stacks on its AID
                          //!< (AID[1,2,3] = 6 stacks/6).
} amdsmi_memory_partition_type_t;

/**
 * @brief This enumeration is used to indicate from which part of the device a
 * temperature reading should be obtained.
 */
typedef enum {
  AMDSMI_TEMPERATURE_TYPE_EDGE,
  AMDSMI_TEMPERATURE_TYPE_FIRST = AMDSMI_TEMPERATURE_TYPE_EDGE,
  AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
  AMDSMI_TEMPERATURE_TYPE_JUNCTION = AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
  AMDSMI_TEMPERATURE_TYPE_VRAM,
  AMDSMI_TEMPERATURE_TYPE_HBM_0,
  AMDSMI_TEMPERATURE_TYPE_HBM_1,
  AMDSMI_TEMPERATURE_TYPE_HBM_2,
  AMDSMI_TEMPERATURE_TYPE_HBM_3,
  AMDSMI_TEMPERATURE_TYPE_PLX,
  AMDSMI_TEMPERATURE_TYPE__MAX = AMDSMI_TEMPERATURE_TYPE_PLX
} amdsmi_temperature_type_t;

/**
 * @brief The values of this enum are used to identify the various firmware
 * blocks.
 */
typedef enum {
  AMDSMI_FW_ID_SMU = 1,
  AMDSMI_FW_ID_FIRST = AMDSMI_FW_ID_SMU,
  AMDSMI_FW_ID_CP_CE,
  AMDSMI_FW_ID_CP_PFP,
  AMDSMI_FW_ID_CP_ME,
  AMDSMI_FW_ID_CP_MEC_JT1,
  AMDSMI_FW_ID_CP_MEC_JT2,
  AMDSMI_FW_ID_CP_MEC1,
  AMDSMI_FW_ID_CP_MEC2,
  AMDSMI_FW_ID_RLC,
  AMDSMI_FW_ID_SDMA0,
  AMDSMI_FW_ID_SDMA1,
  AMDSMI_FW_ID_SDMA2,
  AMDSMI_FW_ID_SDMA3,
  AMDSMI_FW_ID_SDMA4,
  AMDSMI_FW_ID_SDMA5,
  AMDSMI_FW_ID_SDMA6,
  AMDSMI_FW_ID_SDMA7,
  AMDSMI_FW_ID_VCN,
  AMDSMI_FW_ID_UVD,
  AMDSMI_FW_ID_VCE,
  AMDSMI_FW_ID_ISP,
  AMDSMI_FW_ID_DMCU_ERAM, /*eRAM*/
  AMDSMI_FW_ID_DMCU_ISR,  /*ISR*/
  AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM,
  AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM,
  AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL,
  AMDSMI_FW_ID_RLC_V,
  AMDSMI_FW_ID_MMSCH,
  AMDSMI_FW_ID_PSP_SYSDRV,
  AMDSMI_FW_ID_PSP_SOSDRV,
  AMDSMI_FW_ID_PSP_TOC,
  AMDSMI_FW_ID_PSP_KEYDB,
  AMDSMI_FW_ID_DFC,
  AMDSMI_FW_ID_PSP_SPL,
  AMDSMI_FW_ID_DRV_CAP,
  AMDSMI_FW_ID_MC,
  AMDSMI_FW_ID_PSP_BL,
  AMDSMI_FW_ID_CP_PM4,
  AMDSMI_FW_ID_RLC_P,
  AMDSMI_FW_ID_SEC_POLICY_STAGE2,
  AMDSMI_FW_ID_REG_ACCESS_WHITELIST,
  AMDSMI_FW_ID_IMU_DRAM,
  AMDSMI_FW_ID_IMU_IRAM,
  AMDSMI_FW_ID_SDMA_TH0,
  AMDSMI_FW_ID_SDMA_TH1,
  AMDSMI_FW_ID_CP_MES,
  AMDSMI_FW_ID_MES_KIQ,
  AMDSMI_FW_ID_MES_STACK,
  AMDSMI_FW_ID_MES_THREAD1,
  AMDSMI_FW_ID_MES_THREAD1_STACK,
  AMDSMI_FW_ID_RLX6,
  AMDSMI_FW_ID_RLX6_DRAM_BOOT,
  AMDSMI_FW_ID_RS64_ME,
  AMDSMI_FW_ID_RS64_ME_P0_DATA,
  AMDSMI_FW_ID_RS64_ME_P1_DATA,
  AMDSMI_FW_ID_RS64_PFP,
  AMDSMI_FW_ID_RS64_PFP_P0_DATA,
  AMDSMI_FW_ID_RS64_PFP_P1_DATA,
  AMDSMI_FW_ID_RS64_MEC,
  AMDSMI_FW_ID_RS64_MEC_P0_DATA,
  AMDSMI_FW_ID_RS64_MEC_P1_DATA,
  AMDSMI_FW_ID_RS64_MEC_P2_DATA,
  AMDSMI_FW_ID_RS64_MEC_P3_DATA,
  AMDSMI_FW_ID_PPTABLE,
  AMDSMI_FW_ID_PSP_SOC,
  AMDSMI_FW_ID_PSP_DBG,
  AMDSMI_FW_ID_PSP_INTF,
  AMDSMI_FW_ID_RLX6_CORE1,
  AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1,
  AMDSMI_FW_ID_RLCV_LX7,
  AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST,
  AMDSMI_FW_ID_ASD,
  AMDSMI_FW_ID_TA_RAS,
  AMDSMI_FW_ID_TA_XGMI,
  AMDSMI_FW_ID_RLC_SRLG,
  AMDSMI_FW_ID_RLC_SRLS,
  AMDSMI_FW_ID_PM,
  AMDSMI_FW_ID_DMCU,
  AMDSMI_FW_ID__MAX
} amdsmi_fw_block_t;

typedef enum {
  AMDSMI_VRAM_TYPE_UNKNOWN = 0,
  // HBM
  AMDSMI_VRAM_TYPE_HBM = 1,
  AMDSMI_VRAM_TYPE_HBM2 = 2,
  AMDSMI_VRAM_TYPE_HBM2E = 3,
  AMDSMI_VRAM_TYPE_HBM3 = 4,
  // DDR
  AMDSMI_VRAM_TYPE_DDR2 = 10,
  AMDSMI_VRAM_TYPE_DDR3 = 11,
  AMDSMI_VRAM_TYPE_DDR4 = 12,
  // GDDR
  AMDSMI_VRAM_TYPE_GDDR1 = 17,
  AMDSMI_VRAM_TYPE_GDDR2 = 18,
  AMDSMI_VRAM_TYPE_GDDR3 = 19,
  AMDSMI_VRAM_TYPE_GDDR4 = 20,
  AMDSMI_VRAM_TYPE_GDDR5 = 21,
  AMDSMI_VRAM_TYPE_GDDR6 = 22,
  AMDSMI_VRAM_TYPE_GDDR7 = 23,
  AMDSMI_VRAM_TYPE__MAX = AMDSMI_VRAM_TYPE_GDDR7
} amdsmi_vram_type_t;

typedef enum {
    AMDSMI_VRAM_VENDOR__PLACEHOLDER0,
    AMDSMI_VRAM_VENDOR__SAMSUNG,
    AMDSMI_VRAM_VENDOR__INFINEON,
    AMDSMI_VRAM_VENDOR__ELPIDA,
    AMDSMI_VRAM_VENDOR__ETRON,
    AMDSMI_VRAM_VENDOR__NANYA,
    AMDSMI_VRAM_VENDOR__HYNIX,
    AMDSMI_VRAM_VENDOR__MOSEL,
    AMDSMI_VRAM_VENDOR__WINBOND,
    AMDSMI_VRAM_VENDOR__ESMT,
    AMDSMI_VRAM_VENDOR__PLACEHOLDER1,
    AMDSMI_VRAM_VENDOR__PLACEHOLDER2,
    AMDSMI_VRAM_VENDOR__PLACEHOLDER3,
    AMDSMI_VRAM_VENDOR__PLACEHOLDER4,
    AMDSMI_VRAM_VENDOR__PLACEHOLDER5,
    AMDSMI_VRAM_VENDOR__MICRON,
} amdsmi_vram_vendor_type_t;

/**
 * @brief This structure represents a range (e.g., frequencies or voltages).
 */
typedef struct {
    uint64_t lower_bound;      //!< Lower bound of range
    uint64_t upper_bound;      //!< Upper bound of range
    uint64_t reserved[2];
} amdsmi_range_t;

typedef struct {
  uint8_t xgmi_lanes;
  uint64_t xgmi_hive_id;
  uint64_t xgmi_node_id;
  uint32_t index;
  uint32_t reserved[9];
} amdsmi_xgmi_info_t;

typedef struct {
  uint32_t vram_total;
  uint32_t vram_used;
  uint32_t reserved[2];
} amdsmi_vram_usage_t;
/**
 * @brief This structure hold violation status information.
 */
typedef struct {
  uint64_t reference_timestamp;  //!< Represents CPU timestamp in microseconds (uS)
  uint64_t violation_timestamp;  //!< Violation time in milliseconds (ms)
  uint64_t acc_counter;          //!< Current accumulated counter; Max uint64 means unsupported
  uint64_t acc_prochot_thrm;     //!< Current accumulated processor hot violation count; Max uint64 means unsupported
  uint64_t acc_ppt_pwr;          //!< PVIOL; Current accumulated Package Power Tracking (PPT) count; Max uint64 means unsupported
  uint64_t acc_socket_thrm;      //!< TVIOL; Current accumulated Socket thermal count; Max uint64 means unsupported
  uint64_t acc_vr_thrm;          //!< Current accumulated voltage regulator count; Max uint64 means unsupported
  uint64_t acc_hbm_thrm;         //!< Current accumulated High Bandwidth Memory (HBM) thermal count; Max uint64 means unsupported
  uint64_t per_prochot_thrm;     //!< Processor hot violation % (greater than 0% is a violation); Max uint64 means unsupported
  uint64_t per_ppt_pwr;          //!< PVIOL; Package Power Tracking (PPT) violation % (greater than 0% is a violation); Max uint64 means unsupported
  uint64_t per_socket_thrm;      //!< TVIOL; Socket thermal violation % (greater than 0% is a violation); Max uint64 means unsupported
  uint64_t per_vr_thrm;          //!< Voltage regulator violation % (greater than 0% is a violation); Max uint64 means unsupported
  uint64_t per_hbm_thrm;         //!< High Bandwidth Memory (HBM) thermal violation % (greater than 0% is a violation); Max uint64 means unsupported
  uint8_t active_prochot_thrm;   //!< Processor hot violation; 1 = active 0 = not active; Max uint8 means unsupported
  uint8_t active_ppt_pwr;        //!< Package Power Tracking (PPT) violation; 1 = active 0 = not active; Max uint8 means unsupported
  uint8_t active_socket_thrm;    //!< Socket thermal violation; 1 = active 0 = not active; Max uint8 means unsupported
  uint8_t active_vr_thrm;        //!< Voltage regulator violation; 1 = active 0 = not active; Max uint8 means unsupported
  uint8_t active_hbm_thrm;       //!< High Bandwidth Memory (HBM) thermal violation; 1 = active 0 = not active; Max uint8 means unsupported
  uint64_t reserved[30];         // Reserved for new violation info
} amdsmi_violation_status_t;
typedef struct {
  amdsmi_range_t supported_freq_range;
  amdsmi_range_t current_freq_range;
  uint32_t reserved[8];
} amdsmi_frequency_range_t;

typedef union {
  struct {
    uint64_t function_number : 3;
    uint64_t device_number : 5;
    uint64_t bus_number : 8;
    uint64_t domain_number : 48;
  };
  uint64_t as_uint;
} amdsmi_bdf_t;

typedef enum {
  AMDSMI_CARD_FORM_FACTOR_PCIE,
  AMDSMI_CARD_FORM_FACTOR_OAM,
  AMDSMI_CARD_FORM_FACTOR_CEM,
  AMDSMI_CARD_FORM_FACTOR_UNKNOWN
} amdsmi_card_form_factor_t;

typedef struct {
  struct pcie_static_ {
    uint16_t max_pcie_width;              //!< maximum number of PCIe lanes
    uint32_t max_pcie_speed;              //!< maximum PCIe speed
    uint32_t pcie_interface_version;      //!< PCIe interface version
    amdsmi_card_form_factor_t slot_type;  //!< card form factor
    uint64_t reserved[10];
  } pcie_static;
  struct pcie_metric_ {
    uint16_t pcie_width;                  //!< current PCIe width
    uint32_t pcie_speed;                  //!< current PCIe speed in MT/s
    uint32_t pcie_bandwidth;              //!< current instantaneous PCIe bandwidth in Mb/s
    uint64_t pcie_replay_count;           //!< total number of the replays issued on the PCIe link
    uint64_t pcie_l0_to_recovery_count;   //!< total number of times the PCIe link transitioned from L0 to the recovery state
    uint64_t pcie_replay_roll_over_count; //!< total number of replay rollovers issued on the PCIe link
    uint64_t pcie_nak_sent_count;         //!< total number of NAKs issued on the PCIe link by the device
    uint64_t pcie_nak_received_count;     //!< total number of NAKs issued on the PCIe link by the receiver
    uint32_t pcie_lc_perf_other_end_recovery_count;  //!< PCIe other end recovery counter
    uint64_t reserved[12];
  } pcie_metric;
  uint64_t reserved[32];
} amdsmi_pcie_info_t;

typedef struct {
  uint64_t power_cap;           //!< current power cap (uW)
  uint64_t default_power_cap;   //!< default power cap (uW)
  uint64_t dpm_cap;             //!< dpm power cap (MHz)
  uint64_t min_power_cap;       //!< minimum power cap (uW)
  uint64_t max_power_cap;       //!< maximum power cap (uW)
  uint64_t reserved[3];
} amdsmi_power_cap_info_t;

typedef struct {
  char    name[AMDSMI_MAX_STRING_LENGTH];
  char    build_date[AMDSMI_MAX_DATE_LENGTH];
  char    part_number[AMDSMI_MAX_STRING_LENGTH];
  char    version[AMDSMI_NORMAL_STRING_LENGTH];
  uint32_t reserved[16];
} amdsmi_vbios_info_t;

/**
 * @brief cache properties
 */
typedef enum {
  AMDSMI_CACHE_PROPERTY_ENABLED = 0x00000001,
  AMDSMI_CACHE_PROPERTY_DATA_CACHE = 0x00000002,
  AMDSMI_CACHE_PROPERTY_INST_CACHE = 0x00000004,
  AMDSMI_CACHE_PROPERTY_CPU_CACHE = 0x00000008,
  AMDSMI_CACHE_PROPERTY_SIMD_CACHE = 0x00000010,
} amdsmi_cache_property_type_t;

typedef struct {
  uint32_t num_cache_types;
  struct cache_ {
    uint32_t cache_properties;  // amdsmi_cache_property_type_t which is a bitmask
    uint32_t cache_size; /* In KB */
    uint32_t cache_level;
    uint32_t max_num_cu_shared;  /* Indicates how many Compute Units share this cache instance */
    uint32_t num_cache_instance;  /* total number of instance of this cache type */
    uint32_t reserved[3];
  } cache[AMDSMI_MAX_CACHE_TYPES];
  uint32_t reserved[15];
} amdsmi_gpu_cache_info_t;

typedef struct {
  uint8_t num_fw_info;
  struct fw_info_list_ {
    amdsmi_fw_block_t fw_id;
    uint64_t fw_version;
    uint64_t reserved[2];
  } fw_info_list[AMDSMI_FW_ID__MAX];
  uint32_t reserved[7];
} amdsmi_fw_info_t;

typedef struct {
  char  market_name[AMDSMI_256_LENGTH];
  uint32_t vendor_id;   //< Use 32 bit to be compatible with other platform.
  char vendor_name[AMDSMI_MAX_STRING_LENGTH];
  uint32_t subvendor_id;   //< The subsystem vendor id
  uint64_t device_id;   //< The device id of a GPU
  uint32_t rev_id;
  char asic_serial[AMDSMI_NORMAL_STRING_LENGTH];
  uint32_t oam_id;   //< 0xFFFF if not supported
  uint32_t num_of_compute_units;   //< 0xFFFFFFFF if not supported
  uint64_t target_graphics_version;  //< 0xFFFFFFFFFFFFFFFF if not supported
  uint32_t reserved[15];
} amdsmi_asic_info_t;

typedef struct {
  uint64_t kfd_id;  //< 0xFFFFFFFFFFFFFFFF if not supported
  uint32_t node_id;  //< 0xFFFFFFFF if not supported
  uint32_t current_partition_id;  //< 0xFFFFFFFF if not supported
  uint32_t reserved[12];
} amdsmi_kfd_info_t;

/**
 * @brief Possible Memory Partition Modes.
 * This union is used to identify various memory partitioning settings.
 */
typedef union {
  struct nps_flags_ {
    uint32_t nps1_cap :1;  // bool 1 = true; 0 = false; Max uint32 means unsupported
    uint32_t nps2_cap :1;  // bool 1 = true; 0 = false; Max uint32 means unsupported
    uint32_t nps4_cap :1;  // bool 1 = true; 0 = false; Max uint32 means unsupported
    uint32_t nps8_cap :1;  // bool 1 = true; 0 = false; Max uint32 means unsupported
    uint32_t reserved :28;
  } amdsmi_nps_flags_t;

  uint32_t nps_cap_mask;
} amdsmi_nps_caps_t;

/**
 * @brief Possible Memory Partition Modes.
 * This union is used to identify various memory partitioning settings.
 */
typedef struct {
  amdsmi_accelerator_partition_type_t  profile_type;   // SPX, DPX, QPX, CPX and so on
  uint32_t num_partitions;  // On MI300X, SPX: 1, DPX: 2, QPX: 4, CPX: 8, length of resources array
  uint32_t profile_index;
  amdsmi_nps_caps_t memory_caps;             // Possible memory partition capabilities
  uint32_t num_resources;                    // length of index_of_resources_profile
  uint32_t resources[AMDSMI_MAX_ACCELERATOR_PARTITIONS][AMDSMI_MAX_CP_PROFILE_RESOURCES];
  uint64_t reserved[6];
} amdsmi_accelerator_partition_profile_t;

typedef enum {
  AMDSMI_LINK_TYPE_INTERNAL,
  AMDSMI_LINK_TYPE_XGMI,
  AMDSMI_LINK_TYPE_PCIE,
  AMDSMI_LINK_TYPE_NOT_APPLICABLE,
  AMDSMI_LINK_TYPE_UNKNOWN
} amdsmi_link_type_t;

typedef struct {
  uint32_t num_links;   //!< number of links
  struct _links {
    amdsmi_bdf_t bdf;
    uint32_t bit_rate;  //!< current link speed in Gb/s
    uint32_t max_bandwidth;   //!< max bandwidth of the link
    amdsmi_link_type_t link_type;   //!< type of the link
    uint64_t read;  //!< total data received for each link in KB
    uint64_t write;   //!< total data transfered for each link in KB
    uint64_t reserved[2];
  } links[AMDSMI_MAX_NUM_XGMI_PHYSICAL_LINK];
  uint64_t reserved[7];
} amdsmi_link_metrics_t;

typedef struct {
  amdsmi_vram_type_t vram_type;
  amdsmi_vram_vendor_type_t vram_vendor;
  uint64_t vram_size;
  uint32_t vram_bit_width;
  uint64_t reserved[5];
} amdsmi_vram_info_t;


typedef struct {
  char  driver_version[AMDSMI_MAX_STRING_LENGTH];
  char  driver_date[AMDSMI_MAX_STRING_LENGTH];
  char  driver_name[AMDSMI_MAX_STRING_LENGTH];
} amdsmi_driver_info_t;

typedef struct {
  char  model_number[AMDSMI_256_LENGTH];
  char  product_serial[AMDSMI_NORMAL_STRING_LENGTH];
  char  fru_id[AMDSMI_NORMAL_STRING_LENGTH];
  char  product_name[AMDSMI_256_LENGTH];
  char  manufacturer_name[AMDSMI_MAX_STRING_LENGTH];
  uint32_t reserved[32];
} amdsmi_board_info_t;

typedef struct {
  uint32_t current_socket_power;
  uint32_t average_socket_power;
  uint32_t gfx_voltage;   // GFX voltage measurement in mV
  uint32_t soc_voltage;  // SOC voltage measurement in mV
  uint32_t mem_voltage;  // MEM voltage measurement in mV
  uint32_t power_limit;  // The power limit;
  uint32_t reserved[11];
} amdsmi_power_info_t;

typedef struct {
  uint32_t clk;
  uint32_t min_clk;
  uint32_t max_clk;
  uint8_t clk_locked;
  uint8_t clk_deep_sleep;
  uint32_t reserved[4];
} amdsmi_clk_info_t;

/**
 * amdsmi_engine_usage_t:
 * This structure holds common
 * GPU activity values seen in both BM or
 * SRIOV
 **/
typedef struct {
  uint32_t gfx_activity;
  uint32_t umc_activity;
  uint32_t mm_activity;
  uint32_t reserved[13];
} amdsmi_engine_usage_t;
typedef uint32_t amdsmi_process_handle_t;


typedef struct {
  char name[AMDSMI_NORMAL_STRING_LENGTH];
  amdsmi_process_handle_t pid;
  uint64_t mem; /** in bytes */
  struct engine_usage_ {
    uint64_t gfx;
    uint64_t enc;
    uint32_t reserved[12];
  } engine_usage; /** How much time the process spend using these engines in ns */
  struct memory_usage_ {
    uint64_t gtt_mem;
    uint64_t cpu_mem;
    uint64_t vram_mem;
    uint32_t reserved[10];
  } memory_usage; /** in bytes */
  char container_name[AMDSMI_NORMAL_STRING_LENGTH];
  uint32_t reserved[4];
} amdsmi_proc_info_t;

/**
 * @brief IO Link P2P Capability
 */
typedef struct {
  uint8_t is_iolink_coherent;    // 1 = true, 0 = false, UINT8_MAX = Not defined.
  uint8_t is_iolink_atomics_32bit;
  uint8_t is_iolink_atomics_64bit;
  uint8_t is_iolink_dma;
  uint8_t is_iolink_bi_directional;
} amdsmi_p2p_capability_t;


//! Guaranteed maximum possible number of supported frequencies
#define AMDSMI_MAX_NUM_FREQUENCIES 33

//! Maximum possible value for fan speed. Should be used as the denominator
//! when determining fan speed percentage.
#define AMDSMI_MAX_FAN_SPEED 255

//! The number of points that make up a voltage-frequency curve definition
#define AMDSMI_NUM_VOLTAGE_CURVE_POINTS 3
/**
 * @brief PowerPlay performance levels
 */
typedef enum {
  AMDSMI_DEV_PERF_LEVEL_AUTO = 0,       //!< Performance level is "auto"
  AMDSMI_DEV_PERF_LEVEL_FIRST = AMDSMI_DEV_PERF_LEVEL_AUTO,

  AMDSMI_DEV_PERF_LEVEL_LOW,              //!< Keep PowerPlay levels "low",
                                        //!< regardless of workload
  AMDSMI_DEV_PERF_LEVEL_HIGH,             //!< Keep PowerPlay levels "high",
                                        //!< regardless of workload
  AMDSMI_DEV_PERF_LEVEL_MANUAL,           //!< Only use values defined by manually
                                        //!< setting the AMDSMI_CLK_TYPE_SYS speed
  AMDSMI_DEV_PERF_LEVEL_STABLE_STD,       //!< Stable power state with profiling
                                        //!< clocks
  AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK,      //!< Stable power state with peak clocks
  AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK,  //!< Stable power state with minimum
                                        //!< memory clock
  AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK,  //!< Stable power state with minimum
                                        //!< system clock
  AMDSMI_DEV_PERF_LEVEL_DETERMINISM,      //!< Performance determinism state

  AMDSMI_DEV_PERF_LEVEL_LAST = AMDSMI_DEV_PERF_LEVEL_DETERMINISM,

  AMDSMI_DEV_PERF_LEVEL_UNKNOWN = 0x100   //!< Unknown performance level
} amdsmi_dev_perf_level_t;

/**
 * @brief Available clock types.
 */


/**
 * Event counter types
 */

/**
 * @brief Handle to performance event counter
 */
typedef uintptr_t amdsmi_event_handle_t;

/**
 * Event Groups
 *
 * @brief Enum denoting an event group. The value of the enum is the
 * base value for all the event enums in the group.
 */
typedef enum {
  AMDSMI_EVNT_GRP_XGMI = 0,         //!< Data Fabric (XGMI) related events
  AMDSMI_EVNT_GRP_XGMI_DATA_OUT = 10,  //!< XGMI Outbound data
  AMDSMI_EVNT_GRP_INVALID = 0xFFFFFFFF
} amdsmi_event_group_t;

/**
 * Event types
 * @brief Event type enum. Events belonging to a particular event group
 * ::amdsmi_event_group_t should begin enumerating at the ::amdsmi_event_group_t
 * value for that group.
 */
typedef enum {
  AMDSMI_EVNT_FIRST = AMDSMI_EVNT_GRP_XGMI,

  AMDSMI_EVNT_XGMI_FIRST = AMDSMI_EVNT_GRP_XGMI,
  AMDSMI_EVNT_XGMI_0_NOP_TX = AMDSMI_EVNT_XGMI_FIRST,  //!< NOPs sent to neighbor 0
  AMDSMI_EVNT_XGMI_0_REQUEST_TX,                    //!< Outgoing requests to
                                                  //!< neighbor 0
  AMDSMI_EVNT_XGMI_0_RESPONSE_TX,                   //!< Outgoing responses to
                                                  //!< neighbor 0
  /**
   * @brief
   *
   * Data beats sent to neighbor 0; Each beat represents 32 bytes.<br><br>
   *
   * XGMI throughput can be calculated by multiplying a BEATs event
   * such as ::AMDSMI_EVNT_XGMI_0_BEATS_TX by 32 and dividing by
   * the time for which event collection occurred,
   * ::amdsmi_counter_value_t.time_running (which is in nanoseconds). To get
   * bytes per second, multiply this value by 10<sup>9</sup>.<br>
   * <br>
   * Throughput = BEATS/time_running * 10<sup>9</sup>  (bytes/second)<br>
   */
  // ie, Throughput = BEATS/time_running 10^9  bytes/sec
  AMDSMI_EVNT_XGMI_0_BEATS_TX,
  AMDSMI_EVNT_XGMI_1_NOP_TX,                        //!< NOPs sent to neighbor 1
  AMDSMI_EVNT_XGMI_1_REQUEST_TX,                        //!< Outgoing requests to
                                                  //!< neighbor 1
  AMDSMI_EVNT_XGMI_1_RESPONSE_TX,                   //!< Outgoing responses to
                                                  //!< neighbor 1
  AMDSMI_EVNT_XGMI_1_BEATS_TX,                      //!< Data beats sent to
                                                  //!< neighbor 1; Each beat
                                                  //!< represents 32 bytes

  AMDSMI_EVNT_XGMI_LAST = AMDSMI_EVNT_XGMI_1_BEATS_TX,   // 5

  AMDSMI_EVNT_XGMI_DATA_OUT_FIRST = AMDSMI_EVNT_GRP_XGMI_DATA_OUT,  // 10

  /*
   * @brief Events in the AMDSMI_EVNT_GRP_XGMI_DATA_OUT group measure
   * the number of beats sent on an XGMI link. Each beat represents
   * 32 bytes. AMDSMI_EVNT_XGMI_DATA_OUT_n represents the number of
   * outbound beats (each representing 32 bytes) on link n.<br><br>
   *
   * XGMI throughput can be calculated by multiplying a event
   * such as ::AMDSMI_EVNT_XGMI_DATA_OUT_n by 32 and dividing by
   * the time for which event collection occurred,
   * ::amdsmi_counter_value_t.time_running (which is in nanoseconds). To get
   * bytes per second, multiply this value by 10<sup>9</sup>.<br>
   * <br>
   * Throughput = BEATS/time_running * 10<sup>9</sup>  (bytes/second)<br>
   */
  // ie, Throughput = BEATS/time_running 10^9  bytes/sec
  AMDSMI_EVNT_XGMI_DATA_OUT_0 = AMDSMI_EVNT_XGMI_DATA_OUT_FIRST,
  AMDSMI_EVNT_XGMI_DATA_OUT_1,   //!< Outbound beats to neighbor 1
  AMDSMI_EVNT_XGMI_DATA_OUT_2,   //!< Outbound beats to neighbor 2
  AMDSMI_EVNT_XGMI_DATA_OUT_3,   //!< Outbound beats to neighbor 3
  AMDSMI_EVNT_XGMI_DATA_OUT_4,   //!< Outbound beats to neighbor 4
  AMDSMI_EVNT_XGMI_DATA_OUT_5,   //!< Outbound beats to neighbor 5
  AMDSMI_EVNT_XGMI_DATA_OUT_LAST = AMDSMI_EVNT_XGMI_DATA_OUT_5,

  AMDSMI_EVNT_LAST = AMDSMI_EVNT_XGMI_DATA_OUT_LAST,
} amdsmi_event_type_t;

/**
 * Event counter commands
 */
typedef enum {
  AMDSMI_CNTR_CMD_START = 0,  //!< Start the counter
  AMDSMI_CNTR_CMD_STOP,       //!< Stop the counter; note that this should not
                            //!< be used before reading.
} amdsmi_counter_command_t;

/**
 * Counter value
 */
typedef struct {
  uint64_t value;            //!< Counter value
  uint64_t time_enabled;     //!< Time that the counter was enabled
                             //!< (in nanoseconds)
  uint64_t time_running;     //!< Time that the counter was running
                             //!< (in nanoseconds)
} amdsmi_counter_value_t;

/**
 * Event notification event types
 */
typedef enum {
  AMDSMI_EVT_NOTIF_NONE = 0,     //!< Not used
  AMDSMI_EVT_NOTIF_VMFAULT = 1,  //!< VM page fault
  AMDSMI_EVT_NOTIF_FIRST = AMDSMI_EVT_NOTIF_VMFAULT,
  AMDSMI_EVT_NOTIF_THERMAL_THROTTLE = 2,
  AMDSMI_EVT_NOTIF_GPU_PRE_RESET = 3,
  AMDSMI_EVT_NOTIF_GPU_POST_RESET = 4,
  AMDSMI_EVT_NOTIF_RING_HANG = 5,

  AMDSMI_EVT_NOTIF_LAST = AMDSMI_EVT_NOTIF_RING_HANG
} amdsmi_evt_notification_type_t;

/**
 * Macro to generate event bitmask from event id
 */
#define AMDSMI_EVENT_MASK_FROM_INDEX(i) (1ULL << ((i) - 1))

//! Maximum number of characters an event notification message will be
#define MAX_EVENT_NOTIFICATION_MSG_SIZE 64

/**
 * Event notification data returned from event notification API
 */
typedef struct {
    amdsmi_processor_handle processor_handle;        //!< Handler of device that corresponds to the event
    amdsmi_evt_notification_type_t event;     //!< Event type
    char message[MAX_EVENT_NOTIFICATION_MSG_SIZE];  //!< Event message
} amdsmi_evt_notification_data_t;

/**
 * @brief Temperature Metrics.  This enum is used to identify various
 * temperature metrics. Corresponding values will be in millidegress
 * Celcius.
 */
typedef enum {
  AMDSMI_TEMP_CURRENT = 0x0,   //!< Temperature current value.
  AMDSMI_TEMP_FIRST = AMDSMI_TEMP_CURRENT,

  AMDSMI_TEMP_MAX,             //!< Temperature max value.
  AMDSMI_TEMP_MIN,             //!< Temperature min value.
  AMDSMI_TEMP_MAX_HYST,        //!< Temperature hysteresis value for max limit.
                             //!< (This is an absolute temperature, not a
                             //!< delta).
  AMDSMI_TEMP_MIN_HYST,        //!< Temperature hysteresis value for min limit.
                             //!< (This is an absolute temperature,
                             //!<  not a delta).
  AMDSMI_TEMP_CRITICAL,        //!< Temperature critical max value, typically
                             //!<  greater than corresponding temp_max values.
  AMDSMI_TEMP_CRITICAL_HYST,   //!< Temperature hysteresis value for critical
                             //!<  limit. (This is an absolute temperature,
                             //!<  not a delta).
  AMDSMI_TEMP_EMERGENCY,       //!< Temperature emergency max value, for chips
                             //!<  supporting more than two upper temperature
                             //!<  limits. Must be equal or greater than
                             //!<  corresponding temp_crit values.
  AMDSMI_TEMP_EMERGENCY_HYST,  //!< Temperature hysteresis value for emergency
                             //!<  limit. (This is an absolute temperature,
                             //!<  not a delta).
  AMDSMI_TEMP_CRIT_MIN,        //!< Temperature critical min value, typically
                             //!<  lower than corresponding temperature
                             //!<  minimum values.
  AMDSMI_TEMP_CRIT_MIN_HYST,   //!< Temperature hysteresis value for critical
                             //!< minimum limit. (This is an absolute
                             //!< temperature, not a delta).
  AMDSMI_TEMP_OFFSET,          //!< Temperature offset which is added to the
                             //!  temperature reading by the chip.
  AMDSMI_TEMP_LOWEST,          //!< Historical minimum temperature.
  AMDSMI_TEMP_HIGHEST,         //!< Historical maximum temperature.

  AMDSMI_TEMP_LAST = AMDSMI_TEMP_HIGHEST
} amdsmi_temperature_metric_t;

/**
 * @brief Voltage Metrics.  This enum is used to identify various
 * Volatge metrics. Corresponding values will be in millivolt.
 *
 */
typedef enum {
  AMDSMI_VOLT_CURRENT = 0x0,               //!< Voltage current value.

  AMDSMI_VOLT_FIRST = AMDSMI_VOLT_CURRENT,
  AMDSMI_VOLT_MAX,                         //!< Voltage max value.
  AMDSMI_VOLT_MIN_CRIT,                    //!< Voltage critical min value.
  AMDSMI_VOLT_MIN,                         //!< Voltage min value.
  AMDSMI_VOLT_MAX_CRIT,                    //!< Voltage critical max value.
  AMDSMI_VOLT_AVERAGE,                     //!< Average voltage.
  AMDSMI_VOLT_LOWEST,                      //!< Historical minimum voltage.
  AMDSMI_VOLT_HIGHEST,                     //!< Historical maximum voltage.

  AMDSMI_VOLT_LAST = AMDSMI_VOLT_HIGHEST
} amdsmi_voltage_metric_t;

/**
 * @brief This ennumeration is used to indicate which type of
 * voltage reading should be obtained.
 */
typedef enum {
  AMDSMI_VOLT_TYPE_FIRST = 0,

  AMDSMI_VOLT_TYPE_VDDGFX = AMDSMI_VOLT_TYPE_FIRST,  //!< Vddgfx GPU
                                                 //!< voltage
  AMDSMI_VOLT_TYPE_LAST = AMDSMI_VOLT_TYPE_VDDGFX,
  AMDSMI_VOLT_TYPE_INVALID = 0xFFFFFFFF            //!< Invalid type
} amdsmi_voltage_type_t;

/**
 * @brief Pre-set Profile Selections. These bitmasks can be AND'd with the
 * ::amdsmi_power_profile_status_t.available_profiles returned from
 * :: amdsmi_get_gpu_power_profile_presets to determine which power profiles
 * are supported by the system.
 */
typedef enum {
  AMDSMI_PWR_PROF_PRST_CUSTOM_MASK = 0x1,        //!< Custom Power Profile
  AMDSMI_PWR_PROF_PRST_VIDEO_MASK = 0x2,         //!< Video Power Profile
  AMDSMI_PWR_PROF_PRST_POWER_SAVING_MASK = 0x4,  //!< Power Saving Profile
  AMDSMI_PWR_PROF_PRST_COMPUTE_MASK = 0x8,       //!< Compute Saving Profile
  AMDSMI_PWR_PROF_PRST_VR_MASK = 0x10,           //!< VR Power Profile

  //!< 3D Full Screen Power Profile
  AMDSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK = 0x20,
  AMDSMI_PWR_PROF_PRST_BOOTUP_DEFAULT = 0x40,    //!< Default Boot Up Profile
  AMDSMI_PWR_PROF_PRST_LAST = AMDSMI_PWR_PROF_PRST_BOOTUP_DEFAULT,

  //!< Invalid power profile
  AMDSMI_PWR_PROF_PRST_INVALID = 0xFFFFFFFFFFFFFFFF
} amdsmi_power_profile_preset_masks_t;

/**
 * @brief This enum is used to identify different GPU blocks.
 */
typedef enum {
  AMDSMI_GPU_BLOCK_INVALID =   0x0000000000000000,  //!< Used to indicate an
                                                    //!< invalid block
  AMDSMI_GPU_BLOCK_FIRST =     0x0000000000000001,

  AMDSMI_GPU_BLOCK_UMC = AMDSMI_GPU_BLOCK_FIRST,    //!< UMC block
  AMDSMI_GPU_BLOCK_SDMA =      0x0000000000000002,  //!< SDMA block
  AMDSMI_GPU_BLOCK_GFX =       0x0000000000000004,  //!< GFX block
  AMDSMI_GPU_BLOCK_MMHUB =     0x0000000000000008,  //!< MMHUB block
  AMDSMI_GPU_BLOCK_ATHUB =     0x0000000000000010,  //!< ATHUB block
  AMDSMI_GPU_BLOCK_PCIE_BIF =  0x0000000000000020,  //!< PCIE_BIF block
  AMDSMI_GPU_BLOCK_HDP =       0x0000000000000040,  //!< HDP block
  AMDSMI_GPU_BLOCK_XGMI_WAFL = 0x0000000000000080,  //!< XGMI block
  AMDSMI_GPU_BLOCK_DF =        0x0000000000000100,  //!< DF block
  AMDSMI_GPU_BLOCK_SMN =       0x0000000000000200,  //!< SMN block
  AMDSMI_GPU_BLOCK_SEM =       0x0000000000000400,  //!< SEM block
  AMDSMI_GPU_BLOCK_MP0 =       0x0000000000000800,  //!< MP0 block
  AMDSMI_GPU_BLOCK_MP1 =       0x0000000000001000,  //!< MP1 block
  AMDSMI_GPU_BLOCK_FUSE =      0x0000000000002000,  //!< Fuse block
  AMDSMI_GPU_BLOCK_MCA =       0x0000000000004000,  //!< MCA block
  AMDSMI_GPU_BLOCK_VCN =       0x0000000000008000,  //!< VCN block
  AMDSMI_GPU_BLOCK_JPEG =      0x0000000000010000,  //!< JPEG block
  AMDSMI_GPU_BLOCK_IH =        0x0000000000020000,  //!< IH block
  AMDSMI_GPU_BLOCK_MPIO =      0x0000000000040000,  //!< MPIO block

  AMDSMI_GPU_BLOCK_LAST = AMDSMI_GPU_BLOCK_MPIO,    //!< The highest bit position
                                                    //!< for supported blocks
  AMDSMI_GPU_BLOCK_RESERVED =  0x8000000000000000
} amdsmi_gpu_block_t;


/**
 * @brief The clk limit type
 */
typedef enum  {
           CLK_LIMIT_MIN,
           CLK_LIMIT_MAX
} amdsmi_clk_limit_type_t;

/**
 * @brief The current ECC state
 */
typedef enum {
  AMDSMI_RAS_ERR_STATE_NONE = 0,   //!< No current errors
  AMDSMI_RAS_ERR_STATE_DISABLED,   //!< ECC is disabled
  AMDSMI_RAS_ERR_STATE_PARITY,     //!< ECC errors present, but type unknown
  AMDSMI_RAS_ERR_STATE_SING_C,     //!< Single correctable error
  AMDSMI_RAS_ERR_STATE_MULT_UC,    //!< Multiple uncorrectable errors
  AMDSMI_RAS_ERR_STATE_POISON,     //!< Firmware detected error and isolated
                                 //!< page. Treat as uncorrectable.
  AMDSMI_RAS_ERR_STATE_ENABLED,    //!< ECC is enabled

  AMDSMI_RAS_ERR_STATE_LAST = AMDSMI_RAS_ERR_STATE_ENABLED,
  AMDSMI_RAS_ERR_STATE_INVALID = 0xFFFFFFFF
} amdsmi_ras_err_state_t;

/**
 * @brief Types of memory
 */
typedef enum {
  AMDSMI_MEM_TYPE_FIRST = 0,

  AMDSMI_MEM_TYPE_VRAM = AMDSMI_MEM_TYPE_FIRST,  //!< VRAM memory
  AMDSMI_MEM_TYPE_VIS_VRAM,                    //!< VRAM memory that is visible
  AMDSMI_MEM_TYPE_GTT,                         //!< GTT memory

  AMDSMI_MEM_TYPE_LAST = AMDSMI_MEM_TYPE_GTT
} amdsmi_memory_type_t;

/**
 * @brief The values of this enum are used as frequency identifiers.
 */
typedef enum {
  AMDSMI_FREQ_IND_MIN = 0,  //!< Index used for the minimum frequency value
  AMDSMI_FREQ_IND_MAX = 1,  //!< Index used for the maximum frequency value
  AMDSMI_FREQ_IND_INVALID = 0xFFFFFFFF  //!< An invalid frequency index
} amdsmi_freq_ind_t;

/**
 * @brief XGMI Status
 */
typedef enum {
  AMDSMI_XGMI_STATUS_NO_ERRORS = 0,
  AMDSMI_XGMI_STATUS_ERROR,
  AMDSMI_XGMI_STATUS_MULTIPLE_ERRORS,
} amdsmi_xgmi_status_t;

/**
 * @brief Bitfield used in various AMDSMI calls
 */
typedef uint64_t amdsmi_bit_field_t;

/**
 * @brief Reserved Memory Page States
 */
typedef enum {
  AMDSMI_MEM_PAGE_STATUS_RESERVED = 0,  //!< Reserved. This gpu page is reserved
                                      //!<  and not available for use
  AMDSMI_MEM_PAGE_STATUS_PENDING,       //!< Pending. This gpu page is marked
                                      //!<  as bad and will be marked reserved
                                      //!<  at the next window.
  AMDSMI_MEM_PAGE_STATUS_UNRESERVABLE   //!< Unable to reserve this page
} amdsmi_memory_page_status_t;

/**
 * @brief Types for IO Link
 */
typedef enum {
  AMDSMI_IOLINK_TYPE_UNDEFINED      = 0,          //!< unknown type.
  AMDSMI_IOLINK_TYPE_PCIEXPRESS     = 1,          //!< PCI Express
  AMDSMI_IOLINK_TYPE_XGMI           = 2,          //!< XGMI
  AMDSMI_IOLINK_TYPE_NUMIOLINKTYPES,              //!< Number of IO Link types
  AMDSMI_IOLINK_TYPE_SIZE           = 0xFFFFFFFF  //!< Max of IO Link types
} amdsmi_io_link_type_t;

/**
 * @brief The utilization counter type
 */
typedef enum {
  AMDSMI_UTILIZATION_COUNTER_FIRST = 0,
  //!< Corse grain activity counters
  AMDSMI_COARSE_GRAIN_GFX_ACTIVITY  = AMDSMI_UTILIZATION_COUNTER_FIRST,
  AMDSMI_COARSE_GRAIN_MEM_ACTIVITY,    //!< Memory Activity
  AMDSMI_COARSE_DECODER_ACTIVITY,      //!< Decoder Activity
  //!< Fine grain activity counters
  AMDSMI_FINE_GRAIN_GFX_ACTIVITY = 100,
  AMDSMI_FINE_GRAIN_MEM_ACTIVITY = 101,
  AMDSMI_FINE_DECODER_ACTIVITY   = 102,
  AMDSMI_UTILIZATION_COUNTER_LAST = AMDSMI_FINE_DECODER_ACTIVITY
} amdsmi_utilization_counter_type_t;

/**
 * @brief Power types
 */
typedef enum {
  AMDSMI_AVERAGE_POWER = 0,            //!< Average Power
  AMDSMI_CURRENT_POWER,                //!< Current / Instant Power
  AMDSMI_INVALID_POWER = 0xFFFFFFFF    //!< Invalid / Undetected Power
} amdsmi_power_type_t;

/**
 * @brief The utilization counter data
 */
//! The max number of values per counter type
#define AMDSMI_MAX_UTILIZATION_VALUES 4
typedef struct {
  amdsmi_utilization_counter_type_t type;              //!< Utilization counter type
  uint64_t value;                                      //!< Coarse grain activity counter value (average)
  uint64_t fine_value[AMDSMI_MAX_UTILIZATION_VALUES];  //!< Utilization counter value
  uint16_t fine_value_count;
} amdsmi_utilization_counter_t;


/**
 * @brief Reserved Memory Page Record
 */
typedef struct {
  uint64_t page_address;                  //!< Start address of page
  uint64_t page_size;                     //!< Page size
  amdsmi_memory_page_status_t status;       //!< Page "reserved" status
} amdsmi_retired_page_record_t;

/**
 * @brief Number of possible power profiles that a system could support
 */
#define AMDSMI_MAX_NUM_POWER_PROFILES (sizeof(amdsmi_bit_field_t) * 8)

/**
 * @brief This structure contains information about which power profiles are
 * supported by the system for a given device, and which power profile is
 * currently active.
 */
typedef struct {
    /**
    * Which profiles are supported by this system
    */
    amdsmi_bit_field_t available_profiles;

    /**
    * Which power profile is currently active
    */
    amdsmi_power_profile_preset_masks_t current;

    /**
    * How many power profiles are available
    */
    uint32_t num_profiles;
} amdsmi_power_profile_status_t;

/**
 * @brief This structure holds information about clock frequencies.
 */
typedef struct {
    /**
     * Deep Sleep frequency is only supported by some GPUs
     */
    bool has_deep_sleep;

    /**
     * The number of supported frequencies
     */
    uint32_t num_supported;

    /**
     * The current frequency index
     */
    uint32_t current;

    /**
     * List of frequencies.
     * Only the first num_supported frequencies are valid.
     */
    uint64_t frequency[AMDSMI_MAX_NUM_FREQUENCIES];
} amdsmi_frequencies_t;

/**
 * @brief The dpm policy.
 */
typedef struct {
  uint32_t policy_id;
  char policy_description[AMDSMI_MAX_NAME];
} amdsmi_dpm_policy_entry_t;

#define AMDSMI_MAX_NUM_PM_POLICIES 32

/**
 * @brief This structure holds information about dpm policies.
 */
typedef struct {
    /**
     * The number of supported policies
     */
    uint32_t num_supported;

    /**
     * The current policy index
     */
    uint32_t current;

    /**
     * List of policies.
     * Only the first num_supported policies are valid.
     */
    amdsmi_dpm_policy_entry_t policies[AMDSMI_MAX_NUM_PM_POLICIES];
} amdsmi_dpm_policy_t;

/**
 * @brief This structure holds information about the possible PCIe
 * bandwidths. Specifically, the possible transfer rates and their
 * associated numbers of lanes are stored here.
 */
typedef struct {
    /**
     * Transfer rates (T/s) that are possible
     */
    amdsmi_frequencies_t transfer_rate;

    /**
     * List of lanes for corresponding transfer rate.
     * Only the first num_supported bandwidths are valid.
     */
    uint32_t lanes[AMDSMI_MAX_NUM_FREQUENCIES];
} amdsmi_pcie_bandwidth_t;

/**
 * @brief This structure holds version information.
 */
typedef struct {
    uint32_t year;      //!< Last 2 digits of the Year released
    uint32_t major;     //!< Major version
    uint32_t minor;     //!< Minor version
    uint32_t release;   //!< Patch, build or stepping version
    const char *build;  //!< Full Build version string
} amdsmi_version_t;

/**
 * @brief This structure represents a point on the frequency-voltage plane.
 */
typedef struct {
    uint64_t frequency;      //!< Frequency coordinate (in Hz)
    uint64_t voltage;        //!< Voltage coordinate (in mV)
} amdsmi_od_vddc_point_t;

/**
 * @brief This structure holds 2 ::amdsmi_range_t's, one for frequency and one for
 * voltage. These 2 ranges indicate the range of possible values for the
 * corresponding ::amdsmi_od_vddc_point_t.
 */
typedef struct {
    amdsmi_range_t freq_range;  //!< The frequency range for this VDDC Curve point
    amdsmi_range_t volt_range;  //!< The voltage range for this VDDC Curve point
} amdsmi_freq_volt_region_t;

/**
 * ::AMDSMI_NUM_VOLTAGE_CURVE_POINTS number of ::amdsmi_od_vddc_point_t's
 */
typedef struct {
    /**
     * Array of ::AMDSMI_NUM_VOLTAGE_CURVE_POINTS ::amdsmi_od_vddc_point_t's that
     * make up the voltage frequency curve points.
     */
    amdsmi_od_vddc_point_t vc_points[AMDSMI_NUM_VOLTAGE_CURVE_POINTS];
} amdsmi_od_volt_curve_t;

/**
 * @brief This structure holds the frequency-voltage values for a device.
 */
typedef struct {
  amdsmi_range_t curr_sclk_range;          //!< The current SCLK frequency range
  amdsmi_range_t curr_mclk_range;          //!< The current MCLK frequency range;
                                           //!< (upper bound only)
  amdsmi_range_t sclk_freq_limits;         //!< The range possible of SCLK values
  amdsmi_range_t mclk_freq_limits;         //!< The range possible of MCLK values

  /**
   * @brief The current voltage curve
   */
  amdsmi_od_volt_curve_t curve;
  uint32_t num_regions;                //!< The number of voltage curve regions
} amdsmi_od_volt_freq_data_t;

/**
 * @brief The following structures hold the gpu metrics values for a device.
 */

/**
 * @brief Size and version information of metrics data
 */
typedef struct {
  // TODO(amd) Doxygen documents
  // Note: This should match: AMDGpuMetricsHeader_v1_t
  /// \cond Ignore in docs.
  uint16_t      structure_size;
  uint8_t       format_revision;
  uint8_t       content_revision;
  /// \endcond
} amd_metrics_table_header_t;


/**
 * @brief The following structures hold the gpu statistics for a device.
 */
struct amdsmi_gpu_xcp_metrics_t {
  /* Utilization Instantaneous (%) */
  uint32_t gfx_busy_inst[AMDSMI_MAX_NUM_XCC];
  uint16_t jpeg_busy[AMDSMI_MAX_NUM_JPEG];
  uint16_t vcn_busy[AMDSMI_MAX_NUM_VCN];

  /* Utilization Accumulated (%) */
  uint64_t gfx_busy_acc[AMDSMI_MAX_NUM_XCC];
};


typedef struct {
  // TODO(amd) Doxygen documents
  // Note:  This structure is extended to fit the needs of different GPU metric
  //        versions when exposing data through the structure.
  //        Depending on the version, some data members will hold data, and
  //        some will not. A good example is the set of 'current clocks':
  //          - current_gfxclk, current_socclk, current_vclk0, current_dclk0
  //        These are single-valued data members, up to version 1.3.
  //        For version 1.4 and up these are multi-valued data members (arrays)
  //        and their counterparts;
  //          - current_gfxclks[], current_socclks[], current_vclk0s[],
  //            current_dclk0s[]
  //        will hold the data
  /// \cond Ignore in docs.
  amd_metrics_table_header_t common_header;

  /*
   * v1.0 Base
   */

  // Temperature (C)
  uint16_t temperature_edge;
  uint16_t temperature_hotspot;
  uint16_t temperature_mem;
  uint16_t temperature_vrgfx;
  uint16_t temperature_vrsoc;
  uint16_t temperature_vrmem;

  // Utilization (%)
  uint16_t average_gfx_activity;
  uint16_t average_umc_activity;    // memory controller
  uint16_t average_mm_activity;     // UVD or VCN

  // Power (W) /Energy (15.259uJ per 1ns)
  uint16_t average_socket_power;
  uint64_t energy_accumulator;      // v1 mod. (32->64)

  // Driver attached timestamp (in ns)
  uint64_t system_clock_counter;    // v1 mod. (moved from top of struct)

  // Average clocks (MHz)
  uint16_t average_gfxclk_frequency;
  uint16_t average_socclk_frequency;
  uint16_t average_uclk_frequency;
  uint16_t average_vclk0_frequency;
  uint16_t average_dclk0_frequency;
  uint16_t average_vclk1_frequency;
  uint16_t average_dclk1_frequency;

  // Current clocks (MHz)
  uint16_t current_gfxclk;
  uint16_t current_socclk;
  uint16_t current_uclk;
  uint16_t current_vclk0;
  uint16_t current_dclk0;
  uint16_t current_vclk1;
  uint16_t current_dclk1;

  // Throttle status
  uint32_t throttle_status;

  // Fans (RPM)
  uint16_t current_fan_speed;

  // Link width (number of lanes) /speed (0.1 GT/s)
  uint16_t pcie_link_width;         // v1 mod.(8->16)
  uint16_t pcie_link_speed;         // in 0.1 GT/s; v1 mod. (8->16)


  /*
   * v1.1 additions
   */
  uint32_t gfx_activity_acc;        // new in v1
  uint32_t mem_activity_acc;        // new in v1
  uint16_t temperature_hbm[AMDSMI_NUM_HBM_INSTANCES];  // new in v1


  /*
   * v1.2 additions
   */
  // PMFW attached timestamp (10ns resolution)
  uint64_t firmware_timestamp;


  /*
   * v1.3 additions
   */
  // Voltage (mV)
  uint16_t voltage_soc;
  uint16_t voltage_gfx;
  uint16_t voltage_mem;

  // Throttle status
  uint64_t indep_throttle_status;


  /*
   * v1.4 additions
   */
  // Power (Watts)
  uint16_t current_socket_power;

  // Utilization (%)
  uint16_t vcn_activity[AMDSMI_MAX_NUM_VCN];

  // Clock Lock Status. Each bit corresponds to clock instance
  uint32_t gfxclk_lock_status;

  // XGMI bus width and bitrate (in GB/s)
  uint16_t xgmi_link_width;
  uint16_t xgmi_link_speed;

  // PCIE accumulated bandwidth (GB/sec)
  uint64_t pcie_bandwidth_acc;

  // PCIE instantaneous bandwidth (GB/sec)
  uint64_t pcie_bandwidth_inst;

  // PCIE L0 to recovery state transition accumulated count
  uint64_t pcie_l0_to_recov_count_acc;

  // PCIE replay accumulated count
  uint64_t pcie_replay_count_acc;

  // PCIE replay rollover accumulated count
  uint64_t pcie_replay_rover_count_acc;

  // XGMI accumulated data transfer size(KiloBytes)
  uint64_t xgmi_read_data_acc[AMDSMI_MAX_NUM_XGMI_LINKS];
  uint64_t xgmi_write_data_acc[AMDSMI_MAX_NUM_XGMI_LINKS];

  // XGMI accumulated data transfer size(KiloBytes)
  uint16_t current_gfxclks[AMDSMI_MAX_NUM_GFX_CLKS];
  uint16_t current_socclks[AMDSMI_MAX_NUM_CLKS];
  uint16_t current_vclk0s[AMDSMI_MAX_NUM_CLKS];
  uint16_t current_dclk0s[AMDSMI_MAX_NUM_CLKS];

  /*
   * v1.5 additions
   */
  // JPEG activity percent (encode/decode)
  uint16_t jpeg_activity[AMDSMI_MAX_NUM_JPEG];

  // PCIE NAK sent accumulated count
  uint32_t pcie_nak_sent_count_acc;

  // PCIE NAK received accumulated count
  uint32_t pcie_nak_rcvd_count_acc;

  /*
   * v1.6 additions
   */
  /* Accumulation cycle counter */
  uint64_t accumulation_counter;

  /**
   * Accumulated throttler residencies
   */
  uint64_t prochot_residency_acc;
  /**
   * Accumulated throttler residencies
   *
   * Prochot (thermal) - PPT (power)
   * Package Power Tracking (PPT) violation % (greater than 0% is a violation);
   * aka PVIOL
   *
   * Ex. PVIOL/TVIOL calculations
   * Where A and B are measurments recorded at prior points in time.
   * Typically A is the earlier measured value and B is the latest measured value.
   *
   * PVIOL % = (PptResidencyAcc (B) - PptResidencyAcc (A)) * 100/ (AccumulationCounter (B) - AccumulationCounter (A))
   * TVIOL % = (SocketThmResidencyAcc (B) -  SocketThmResidencyAcc (A)) * 100 / (AccumulationCounter (B) - AccumulationCounter (A))
  */
  uint64_t ppt_residency_acc;
  /**
   * Accumulated throttler residencies
   *
   * Socket (thermal)	-
   * Socket thermal violation % (greater than 0% is a violation);
   * aka TVIOL
   *
   * Ex. PVIOL/TVIOL calculations
   * Where A and B are measurments recorded at prior points in time.
   * Typically A is the earlier measured value and B is the latest measured value.
   *
   * PVIOL % = (PptResidencyAcc (B) - PptResidencyAcc (A)) * 100/ (AccumulationCounter (B) - AccumulationCounter (A))
   * TVIOL % = (SocketThmResidencyAcc (B) -  SocketThmResidencyAcc (A)) * 100 / (AccumulationCounter (B) - AccumulationCounter (A))
  */
  uint64_t socket_thm_residency_acc;
  uint64_t vr_thm_residency_acc;
  uint64_t hbm_thm_residency_acc;

  /* Number of current partition */
  uint16_t num_partition;

  /* XCP (Graphic Cluster Partitions) metrics stats */
  struct amdsmi_gpu_xcp_metrics_t xcp_stats[AMDSMI_MAX_NUM_XCP];

  /* PCIE other end recovery counter */
  uint32_t pcie_lc_perf_other_end_recovery;

  /// \endcond
} amdsmi_gpu_metrics_t;


#define  MAX_AMDSMI_NAME_LENGTH 64

/**
 * @brief This structure holds the name value pairs
 */
typedef struct {
    char name[MAX_AMDSMI_NAME_LENGTH];     //!< Name
    uint64_t value;   //!< Use uint64_t to make it universal
} amdsmi_name_value_t;

/**
 * @brief This register type for register table
 */
typedef enum {
  AMDSMI_REG_XGMI,
  AMDSMI_REG_WAFL,
  AMDSMI_REG_PCIE,
  AMDSMI_REG_USR,
  AMDSMI_REG_USR1,
} amdsmi_reg_type_t;

/**
 * @brief This structure holds ras feature
 */
typedef struct {
  uint32_t ras_eeprom_version;
  // PARITY error(bit 0), Single Bit correctable (bit1),
  // Double bit error detection (bit2), Poison (bit 3).
  uint32_t ecc_correction_schema_flag;    //!< ecc_correction_schema mask
} amdsmi_ras_feature_t;

/**
 * @brief This structure holds error counts.
 */
typedef struct {
  uint64_t correctable_count;   //!< Accumulated correctable errors
  uint64_t uncorrectable_count;  //!< Accumulated uncorrectable errors
  uint64_t deferred_count;  //!< Accumulated deferred errors
  uint64_t reserved[5];
} amdsmi_error_count_t;

/**
 * @brief This structure contains information specific to a process.
 */
typedef struct {
    uint32_t process_id;      //!< Process ID
    uint32_t pasid;           //!< PASID
    uint64_t vram_usage;      //!< VRAM usage
    uint64_t sdma_usage;      //!< SDMA usage in microseconds
    uint32_t cu_occupancy;    //!< Compute Unit usage in percent
} amdsmi_process_info_t;


typedef struct {
    uint32_t count;
    amdsmi_processor_handle processor_list[AMDSMI_MAX_DEVICES];
    uint64_t reserved[15];
} amdsmi_topology_nearest_t;


//! Place-holder "variant" for functions that have don't have any variants,
//! but do have monitors or sensors.
#define AMDSMI_DEFAULT_VARIANT 0xFFFFFFFFFFFFFFFF

#ifdef ENABLE_ESMI_LIB
/**
 * @brief This structure holds SMU Firmware version information.
 */
typedef struct {
        uint8_t debug;      //!< SMU fw Debug version number
        uint8_t minor;      //!< SMU fw Minor version number
        uint8_t major;      //!< SMU fw Major version number
        uint8_t unused;     //!< reserved fields
} amdsmi_smu_fw_version_t;

/**
 * @brief DDR bandwidth metrics.
 */
typedef struct {
        uint32_t max_bw;    //!< DDR Maximum theoritical bandwidth in GB/s
        uint32_t utilized_bw;   //!< DDR bandwidth utilization in GB/s
        uint32_t utilized_pct;  //!< DDR bandwidth utilization in % of theoritical max
} amdsmi_ddr_bw_metrics_t;

/**
 * @brief temperature range and refresh rate metrics of a DIMM
 */
typedef struct {
    uint8_t range : 3;  //!< temp range[2:0](3 bit data)
    uint8_t ref_rate : 1;   //!< DDR refresh rate mode[3](1 bit data)
} amdsmi_temp_range_refresh_rate_t;

/**
 * @brief DIMM Power(mW), power update rate(ms) and dimm address
 */
typedef struct {
    uint16_t power : 15;            //!< Dimm power consumption[31:17](15 bits data)
    uint16_t update_rate : 9;       //!< Time since last update[16:8](9 bit data)
    uint8_t dimm_addr;              //!< Dimm address[7:0](8 bit data)
} amdsmi_dimm_power_t;

/**
 * @brief DIMM temperature(°C) and update rate(ms) and dimm address
 */
typedef struct {
    uint16_t sensor : 11;           //!< Dimm thermal sensor[31:21](11 bit data)
    uint16_t update_rate : 9;       //!< Time since last update[16:8](9 bit data)
    uint8_t dimm_addr;              //!< Dimm address[7:0](8 bit data)
    float temp;         //!< temperature in degree celcius
} amdsmi_dimm_thermal_t;

/**
 * @brief xGMI Bandwidth Encoding types
 */
typedef enum {
    AGG_BW0 = 1, //!< Aggregate Bandwidth
    RD_BW0 = 2,      //!< Read Bandwidth
    WR_BW0 = 4       //!< Write Bandwdith
} amdsmi_io_bw_encoding_t;

/**
 * @brief LINK name and Bandwidth type Information.It contains
 * link names i.e valid link names are
 * "P0", "P1", "P2", "P3", "P4", "G0", "G1", "G2", "G3", "G4"
 * "G5", "G6", "G7"
 * Valid bandwidth types 1(Aggregate_BW), 2 (Read BW), 4 (Write BW).
 */
typedef struct {
    amdsmi_io_bw_encoding_t bw_type;         //!< Bandwidth Type Information [1, 2, 4]
    char *link_name;            //!< Link name [P0, P1, G0, G1 etc]
} amdsmi_link_id_bw_type_t;

/**
 * @brief max and min LCLK DPM level on a given NBIO ID.
 * Valid max and min DPM level values are 0 - 1.
 */
typedef struct {
    uint8_t max_dpm_level;          //!< Max LCLK DPM level[15:8](8 bit data)
    uint8_t min_dpm_level;          //!< Min LCLK DPM level[7:0](8 bit data)
} amdsmi_dpm_level_t;

/**
 * @brief HSMP Metrics table (supported only with hsmp proto version 6).
 */
typedef struct __attribute__((__packed__)){
    uint32_t accumulation_counter;

    /* TEMPERATURE */
    uint32_t max_socket_temperature;
    uint32_t max_vr_temperature;
    uint32_t max_hbm_temperature;
    uint64_t max_socket_temperature_acc;
    uint64_t max_vr_temperature_acc;
    uint64_t max_hbm_temperature_acc;

    /* POWER */
    uint32_t socket_power_limit;
    uint32_t max_socket_power_limit;
    uint32_t socket_power;

    /* ENERGY */
    uint64_t timestamp;
    uint64_t socket_energy_acc;
    uint64_t ccd_energy_acc;
    uint64_t xcd_energy_acc;
    uint64_t aid_energy_acc;
    uint64_t hbm_energy_acc;

    /* FREQUENCY */
    uint32_t cclk_frequency_limit;
    uint32_t gfxclk_frequency_limit;
    uint32_t fclk_frequency;
    uint32_t uclk_frequency;
    uint32_t socclk_frequency[4];
    uint32_t vclk_frequency[4];
    uint32_t dclk_frequency[4];
    uint32_t lclk_frequency[4];
    uint64_t gfxclk_frequency_acc[8];
    uint64_t cclk_frequency_acc[96];

    /* FREQUENCY RANGE */
    uint32_t max_cclk_frequency;
    uint32_t min_cclk_frequency;
    uint32_t max_gfxclk_frequency;
    uint32_t min_gfxclk_frequency;
    uint32_t fclk_frequency_table[4];
    uint32_t uclk_frequency_table[4];
    uint32_t socclk_frequency_table[4];
    uint32_t vclk_frequency_table[4];
    uint32_t dclk_frequency_table[4];
    uint32_t lclk_frequency_table[4];
    uint32_t max_lclk_dpm_range;
    uint32_t min_lclk_dpm_range;

    /* XGMI */
    uint32_t xgmi_width;
    uint32_t xgmi_bitrate;
    uint64_t xgmi_read_bandwidth_acc[8];
    uint64_t xgmi_write_bandwidth_acc[8];

    /* ACTIVITY */
    uint32_t socket_c0_residency;
    uint32_t socket_gfx_busy;
    uint32_t dram_bandwidth_utilization;
    uint64_t socket_c0_residency_acc;
    uint64_t socket_gfx_busy_acc;
    uint64_t dram_bandwidth_acc;
    uint32_t max_dram_bandwidth;
    uint64_t dram_bandwidth_utilization_acc;
    uint64_t pcie_bandwidth_acc[4];

    /* THROTTLERS */
    uint32_t prochot_residency_acc;
    uint32_t ppt_residency_acc;
    uint32_t socket_thm_residency_acc;
    uint32_t vr_thm_residency_acc;
    uint32_t hbm_thm_residency_acc;
    uint32_t spare;

    /* New items at the end to maintain driver compatibility */
    uint32_t gfxclk_frequency[8];
} amdsmi_hsmp_metrics_table_t;

/**
 * @brief hsmp frequency limit source names
 */
static char* const amdsmi_hsmp_freqlimit_src_names[] = {
        "cHTC-Active",
        "PROCHOT",
        "TDC limit",
        "PPT Limit",
        "OPN Max",
        "Reliability Limit",
        "APML Agent",
        "HSMP Agent"
};
#endif

/*****************************************************************************/
/** @defgroup InitShutAdmin Initialization and Shutdown
 *  These functions are used for initialization of AMD SMI and clean up when done.
 *  @{
 */

/**
 *  @brief Initialize the AMD SMI library
 *
 *  @platform{gpu_bm_linux}  @platform{host} @platform{cpu_bm}  @platform{guest_1vf}
 *  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @details This function initializes the library and the internal data structures,
 *  including those corresponding to sources of information that SMI provides.
 *
 *  The @p init_flags decides which type of processor
 *  can be discovered by ::amdsmi_get_socket_handles(). AMDSMI_INIT_AMD_GPUS returns
 *  sockets with AMD GPUS, and AMDSMI_INIT_AMD_GPUS | AMDSMI_INIT_AMD_CPUS returns
 *  sockets with either AMD GPUS or CPUS.
 *  Currently, only AMDSMI_INIT_AMD_GPUS is supported.
 *
 *  @param[in] init_flags Bit flags that tell SMI how to initialze. Values of
 *  ::amdsmi_init_flags_t may be OR'd together and passed through @p init_flags
 *  to modify how AMDSMI initializes.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_init(uint64_t init_flags);

/**
 *  @brief Shutdown the AMD SMI library
 *
 *  @platform{gpu_bm_linux}  @platform{host} @platform{cpu_bm}  @platform{guest_1vf}
 *  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @details This function shuts down the library and internal data structures and
 *  performs any necessary clean ups.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_shut_down(void);

/** @} End InitShutAdmin */

/*****************************************************************************/
/** @defgroup DiscQueries Discovery Queries
 *  These functions provide discovery of the sockets.
 *  @{
 */

/**
 *  @brief Get the list of socket handles in the system.
 *
 *  @platform{gpu_bm_linux}  @platform{host} @platform{cpu_bm}  @platform{guest_1vf}
 *  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @details Depends on what flag is passed to ::amdsmi_init.  AMDSMI_INIT_AMD_GPUS
 *  returns sockets with AMD GPUS, and AMDSMI_INIT_AMD_GPUS | AMDSMI_INIT_AMD_CPUS returns
 *  sockets with either AMD GPUS or CPUS.
 *  The socket handles can be used to query the processor handles in that socket, which
 *  will be used in other APIs to get processor detail information or telemtries.
 *
 *  @param[in,out] socket_count As input, the value passed
 *  through this parameter is the number of ::amdsmi_socket_handle that
 *  may be safely written to the memory pointed to by @p socket_handles. This is the
 *  limit on how many socket handles will be written to @p socket_handles. On return, @p
 *  socket_count will contain the number of socket handles written to @p socket_handles,
 *  or the number of socket handles that could have been written if enough memory had been
 *  provided.
 *  If @p socket_handles is NULL, as output, @p socket_count will contain
 *  how many sockets are available to read in the system.
 *
 *  @param[in,out] socket_handles A pointer to a block of memory to which the
 *  ::amdsmi_socket_handle values will be written. This value may be NULL.
 *  In this case, this function can be used to query how many sockets are
 *  available to read in the system.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_socket_handles(uint32_t *socket_count,
                amdsmi_socket_handle* socket_handles);

#ifdef ENABLE_ESMI_LIB
/**
 *  @brief Get the list of cpu handles in the system.
 *
 *  @platform{cpu_bm}
 *
 *  @details Depends on AMDSMI_INIT_AMD_CPUS flag passed to ::amdsmi_init.
 *  The processor handles can be used in other APIs to get processor detail information.
 *
 *  @param[in,out] cpu_count As input, the value passed
 *  through this parameter is the number of ::amdsmi_processor_handle that
 *  may be safely written to the memory pointed to by @p processor_handles. This is the
 *  limit on how many processor handles will be written to @p processor_handles. On return, @p
 *  socket_count will contain the number of processor handles written to @p processor_handles,
 *  or the number of processor handles that could have been written if enough memory had been
 *  provided.
 *  If @p processor_handles is NULL, as output, @p cpu_count will contain
 *  how many processors are available to read in the system.
 *
 *  @param[in,out] socket_handles A pointer to a block of memory to which the
 *  ::amdsmi_processor_handle values will be written. This value may be NULL.
 *  In this case, this function can be used to query how many processors are
 *  available to read in the system.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_handles(uint32_t *cpu_count,
                                       amdsmi_processor_handle *processor_handles);
#endif

/**
 *  @brief Get information about the given socket
 *
 *  @platform{gpu_bm_linux}  @platform{host}  @platform{guest_1vf}
 *  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @details This function retrieves socket information. The @p socket_handle must
 *  be provided to retrieve the Socket ID.
 *
 *  @param[in] socket_handle a socket handle
 *
 *  @param[out] name The id of the socket.
 *
 *  @param[in] len the length of the caller provided buffer @p name.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_socket_info(
                amdsmi_socket_handle socket_handle,
                size_t len, char *name);

#ifdef ENABLE_ESMI_LIB
/**
 *  @brief Get information about the given processor
 *
 *  @platform{cpu_bm}
 *
 *  @details This function retrieves processor information. The @p processor_handle must
 *  be provided to retrieve the processor ID.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[out] name The id of the processor.
 *
 *  @param[in] len the length of the caller provided buffer @p name.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_processor_info(
                amdsmi_processor_handle processor_handle,
                size_t len, char *name);

/**
 *  @brief Get respective processor counts from the processor handles
 *
 *  @platform{cpu_bm}
 *
 *  @details This function retrieves respective processor counts information.
 *  The @p processor_handle must be provided to retrieve the processor ID.
 *
 *  @param[in] processor_handles A pointer to a block of memory to which the
 *  ::amdsmi_processor_handle values will be written. This value may be NULL.
 *
 *  @param[in] processor_count total processor count per socket
 *
 *  @param[out] nr_cpusockets Total number of cpu sockets
 *
 *  @param[out] nr_cpucores Total number of cpu cores
 *
 *  @param[out] nr_gpus Total number of gpu devices
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_processor_count_from_handles(amdsmi_processor_handle* processor_handles,
                                                        uint32_t* processor_count, uint32_t* nr_cpusockets,
                                                        uint32_t* nr_cpucores, uint32_t* nr_gpus);

/**
 *  @brief Get processor list as per processor type
 *
 *  @platform{cpu_bm}
 *
 *  @details This function retrieves processor list as per the processor type
 *  from the total processor handles list.
 *  The @p list of processor_handles and processor type must be provided.
 *
 *  @param[in] socket_handle socket handle
 *
 *  @param[in] processor_type processor type
 *
 *  @param[out] processor_handles list of processor handles as per processor type
 *
 *  @param[out] processor_count processor count as per processor type selected
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_processor_handles_by_type(amdsmi_socket_handle socket_handle,
                                                     processor_type_t processor_type,
                                                     amdsmi_processor_handle* processor_handles,
                                                     uint32_t* processor_count);
#endif

/**
 *  @brief Get the list of the processor handles associated to a socket.
 *
 *  @platform{gpu_bm_linux}  @platform{host} @platform{guest_1vf}
 *  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @details This function retrieves the processor handles of a socket. The
 *  @p socket_handle must be provided for the processor. A socket may have mulitple different
 *  type processors: An APU on a socket have both CPUs and GPUs.
 *  Currently, only AMD GPUs are supported.
 *
 *  The number of processor count is returned through @p processor_count
 *  if @p processor_handles is NULL. Then the number of @p processor_count can be pass
 *  as input to retrieval all processors on the socket to @p processor_handles.
 *
 *  @param[in] socket_handle The socket to query
 *
 *  @param[in,out] processor_count As input, the value passed
 *  through this parameter is the number of ::amdsmi_processor_handle's that
 *  may be safely written to the memory pointed to by @p processor_handles. This is the
 *  limit on how many processor handles will be written to @p processor_handles. On return, @p
 *  processor_count will contain the number of processor handles written to @p processor_handles,
 *  or the number of processor handles that could have been written if enough memory had been
 *  provided.
 *  If @p processor_handles is NULL, as output, @p processor_count will contain
 *  how many processors are available to read for the socket.
 *
 *  @param[in,out] processor_handles A pointer to a block of memory to which the
 *  ::amdsmi_processor_handle values will be written. This value may be NULL.
 *  In this case, this function can be used to query how many processors are
 *  available to read.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_processor_handles(amdsmi_socket_handle socket_handle,
                                    uint32_t *processor_count,
                                    amdsmi_processor_handle* processor_handles);

#ifdef ENABLE_ESMI_LIB
/**
 *  @brief Get the list of the cpu core handles in a system.
 *
 *  @platform{cpu_bm}
 *
 *  @details This function retrieves the cpu core handles of a system.
 *  @param[in,out] cores_count As input, the value passed
 *  through this parameter is the number of ::amdsmi_processor_handle's that
 *  may be safely written to the memory pointed to by @p processor_handles. This is the
 *  limit on how many core handles will be written to @p processor_handles. On return, @p
 *  cores_count will contain the number of core processor handles written to @p processor_handles,
 *  or the number of core processor handles that could have been written if enough memory had been
 *  provided.
 *  If @p processor_handles is NULL, as output, @p processor_count will contain
 *  how many cpu cores are available to read in the system.
 *
 *  @param[in,out] processor_handles A pointer to a block of memory to which the
 *  ::amdsmi_processor_handle values will be written. This value may be NULL.
 *  In this case, this function can be used to query how many processors are
 *  available to read.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
*/
amdsmi_status_t amdsmi_get_cpucore_handles(uint32_t *cores_count,
                                           amdsmi_processor_handle* processor_handles);
#endif

/**
 *  @brief Get the processor type of the processor_handle
 *
 *  @platform{gpu_bm_linux}  @platform{host} @platform{cpu_bm}  @platform{guest_1vf}
 *  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @details This function retrieves the processor type. A processor_handle must be provided
 *  for that processor.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[out] processor_type a pointer to processor_type_t to which the processor type
 *  will be written. If this parameter is nullptr, this function will return
 * ::AMDSMI_STATUS_INVAL.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_processor_type(amdsmi_processor_handle processor_handle,
              processor_type_t* processor_type);

/**
 *  @brief Get processor handle with the matching bdf.
 *
 *  @platform{gpu_bm_linux}  @platform{host} @platform{guest_1vf}
 *  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @details Given bdf info @p bdf, this function will get
 *  the processor handle with the matching bdf.
 *
 *  @param[in] bdf The bdf to match with corresponding processor handle.
 *
 *  @param[out] processor_handle processor handle with the matching bdf.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_processor_handle_from_bdf(amdsmi_bdf_t bdf,
              amdsmi_processor_handle* processor_handle);

/** @} End DiscQueries */

/*****************************************************************************/
/** @defgroup IDQuer Identifier Queries
 *  These functions provide identification information.
 *  @{
 */

/**
 *  @brief Get the device id associated with the device with provided device
 *  handler.
 *
 *  @platform{gpu_bm_linux}  @platform{guest_1vf}  @platform{guest_mvf}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint32_t @p id,
 *  this function will write the device id value to the uint64_t pointed to by
 *  @p id. This ID is an identification of the type of device, so calling this
 *  function for different devices will give the same value if they are kind
 *  of device. Consequently, this function should not be used to distinguish
 *  one device from another. amdsmi_get_gpu_bdf_id() should be used to get a
 *  unique identifier.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] id a pointer to uint64_t to which the device id will be
 *  written
 * If this parameter is nullptr, this function will return
 * ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 * arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 * provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_id(amdsmi_processor_handle processor_handle, uint16_t *id);

/**
 *  @brief Get the device revision associated with the device
 *
 *  @platform{gpu_bm_linux} @platform{guest_1vf}  @platform{guest_mvf}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a
 *  uint16_t @p revision to which the revision id will be written
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[out] revision a pointer to uint16_t to which the device revision
 *  will be written
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_revision(amdsmi_processor_handle processor_handle, uint16_t *revision);

/**
 *  @brief Get the name string for a give vendor ID
 *
 *  @platform{gpu_bm_linux}  @platform{guest_1vf}  @platform{guest_mvf}
 *
 *  @details Given a processor handle @p processor_handle, a pointer to a caller provided
 *  char buffer @p name, and a length of this buffer @p len, this function will
 *  write the name of the vendor (up to @p len characters) buffer @p name. The
 *  @p id may be a device vendor or subsystem vendor ID.
 *
 *  If the integer ID associated with the vendor is not found in one of the
 *  system files containing device name information (e.g.
 *  /usr/share/misc/pci.ids), then this function will return the hex vendor ID
 *  as a string. Updating the system name files can be accompplished with
 *  "sudo update-pciids".
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] name a pointer to a caller provided char buffer to which the
 *  name will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[in] len the length of the caller provided buffer @p name.
 *
 *  @note ::AMDSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire name. In this case, only @p len bytes will
 *  be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_vendor_name(amdsmi_processor_handle processor_handle, char *name,
                                                                  size_t len);

/**
 *  @brief Get the vram vendor string of a device.
 *
 *  @platform{gpu_bm_linux}  @platform{guest_1vf}  @platform{guest_mvf}
 *
 *  @details This function retrieves the vram vendor name given a processor handle
 *  @p processor_handle, a pointer to a caller provided
 *  char buffer @p brand, and a length of this buffer @p len, this function
 *  will write the vram vendor of the device (up to @p len characters) to the
 *  buffer @p brand.
 *
 *  If the vram vendor for the device is not found as one of the values
 *  contained within amdsmi_get_gpu_vram_vendor, then this function will return
 *  the string 'unknown' instead of the vram vendor.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] brand a pointer to a caller provided char buffer to which the
 *  vram vendor will be written
 *
 *  @param[in] len the length of the caller provided buffer @p brand.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_vram_vendor(amdsmi_processor_handle processor_handle, char *brand,
                                                                uint32_t len);

/**
 *  @brief Get the subsystem device id associated with the device with
 *  provided processor handle.
 *
 * @platform{gpu_bm_linux} @platform{guest_1vf}  @platform{guest_mvf}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint32_t @p id,
 *  this function will write the subsystem device id value to the uint64_t
 *  pointed to by @p id.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] id a pointer to uint64_t to which the subsystem device id
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_subsystem_id(amdsmi_processor_handle processor_handle, uint16_t *id);

/**
 *  @brief Get the name string for the device subsytem
 *
 *  @platform{gpu_bm_linux}  @platform{guest_1vf}  @platform{guest_mvf}
 *
 *  @details Given a processor handle @p processor_handle, a pointer to a caller provided
 *  char buffer @p name, and a length of this buffer @p len, this function
 *  will write the name of the device subsystem (up to @p len characters)
 *  to the buffer @p name.
 *
 *  If the integer ID associated with the sub-system is not found in one of the
 *  system files containing device name information (e.g.
 *  /usr/share/misc/pci.ids), then this function will return the hex sub-system
 *  ID as a string. Updating the system name files can be accompplished with
 *  "sudo update-pciids".
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] name a pointer to a caller provided char buffer to which the
 *  name will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.

 *  @param[in] len the length of the caller provided buffer @p name.
 *
 *  @note ::AMDSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire name. In this case, only @p len bytes will
 *  be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_subsystem_name(amdsmi_processor_handle processor_handle, char *name, size_t len);

/** @} End IDQuer */

/*****************************************************************************/
/** @defgroup PCIeQuer PCIe Queries
 *  These functions provide information about PCIe.
 *  @{
 */

/**
 *  @brief Get the list of possible PCIe bandwidths that are available. It is not
 *  supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a to an
 *  ::amdsmi_pcie_bandwidth_t structure @p bandwidth, this function will fill in
 *  @p bandwidth with the possible T/s values and associated number of lanes,
 *  and indication of the current selection.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] bandwidth a pointer to a caller provided
 *  ::amdsmi_pcie_bandwidth_t structure to which the frequency information will be
 *  written
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_pci_bandwidth(amdsmi_processor_handle processor_handle,
                              amdsmi_pcie_bandwidth_t *bandwidth);

/**
 *  @brief Get the unique PCI device identifier associated for a device
 *
 *  @platform{gpu_bm_linux}  @platform{guest_1vf}
 *
 *  @details Give a processor handle @p processor_handle and a pointer to a uint64_t @p
 *  bdfid, this function will write the Bus/Device/Function PCI identifier
 *  (BDFID) associated with device @p processor_handle to the value pointed to by
 *  @p bdfid.
 *
 *  The format of @p bdfid will be as follows:
 *
 *      BDFID = ((DOMAIN & 0xFFFFFFFF) << 32) | ((Partition & 0xF) << 28)
 *              | ((BUS & 0xFF) << 8) | ((DEVICE & 0x1F) <<3 )
 *              | (FUNCTION & 0x7)
 *
 *  | Name         | Field   | KFD property       KFD -> PCIe ID (uint64_t)
 *  -------------- | ------- | ---------------- | ---------------------------- |
 *  | Domain       | [63:32] | "domain"         | (DOMAIN & 0xFFFFFFFF) << 32  |
 *  | Partition id | [31:28] | "location id"    | (LOCATION & 0xF0000000)      |
 *  | Reserved     | [27:16] | "location id"    | N/A                          |
 *  | Bus          | [15: 8] | "location id"    | (LOCATION & 0xFF00)          |
 *  | Device       | [ 7: 3] | "location id"    | (LOCATION & 0xF8)            |
 *  | Function     | [ 2: 0] | "location id"    | (LOCATION & 0x7)             |
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] bdfid a pointer to uint64_t to which the device bdfid value
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_bdf_id(amdsmi_processor_handle processor_handle, uint64_t *bdfid);

/**
 *  @brief Get the NUMA node associated with a device
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a int32_t @p
 *  numa_node, this function will retrieve the NUMA node value associated
 *  with device @p processor_handle and store the value at location pointed to by
 *  @p numa_node.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] numa_node pointer to location where NUMA node value will
 *  be written.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_topo_numa_affinity(amdsmi_processor_handle processor_handle,
                                                    int32_t *numa_node);

/**
 *  @brief Get PCIe traffic information. It is not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Give a processor handle @p processor_handle and pointers to a uint64_t's, @p
 *  sent, @p received and @p max_pkt_sz, this function will write the number
 *  of bytes sent and received in 1 second to @p sent and @p received,
 *  respectively. The maximum possible packet size will be written to
 *  @p max_pkt_sz.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] sent a pointer to uint64_t to which the number of bytes sent
 *  will be written in 1 second. If pointer is NULL, it will be ignored.
 *
 *  @param[in,out] received a pointer to uint64_t to which the number of bytes
 *  received will be written. If pointer is NULL, it will be ignored.
 *
 *  @param[in,out] max_pkt_sz a pointer to uint64_t to which the maximum packet
 *  size will be written. If pointer is NULL, it will be ignored.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_pci_throughput(amdsmi_processor_handle processor_handle, uint64_t *sent,
                                    uint64_t *received, uint64_t *max_pkt_sz);

/**
 *  @brief Get PCIe replay counter
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint64_t @p
 *  counter, this function will write the sum of the number of NAK's received
 *  by the GPU and the NAK's generated by the GPU to memory pointed to by @p
 *  counter.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] counter a pointer to uint64_t to which the sum of the NAK's
 *  received and generated by the GPU is written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_pci_replay_counter(amdsmi_processor_handle processor_handle,
                                                           uint64_t *counter);

/** @} End PCIeQuer */

/*****************************************************************************/
/** @defgroup PCIeCont PCIe Control
 *  These functions provide some control over PCIe.
 *  @{
 */

/**
 *  @brief Control the set of allowed PCIe bandwidths that can be used. It is not
 *  supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a 64 bit bitmask @p bw_bitmask,
 *  this function will limit the set of allowable bandwidths. If a bit in @p
 *  bw_bitmask has a value of 1, then the frequency (as ordered in an
 *  ::amdsmi_frequencies_t returned by :: amdsmi_get_clk_freq()) corresponding
 *  to that bit index will be allowed.
 *
 *  This function will change the performance level to
 *  ::AMDSMI_DEV_PERF_LEVEL_MANUAL in order to modify the set of allowable
 *  band_widths. Caller will need to set to ::AMDSMI_DEV_PERF_LEVEL_AUTO in order
 *  to get back to default state.
 *
 *  All bits with indices greater than or equal to the value of the
 *  ::amdsmi_frequencies_t::num_supported field of ::amdsmi_pcie_bandwidth_t will be
 *  ignored.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] bw_bitmask A bitmask indicating the indices of the
 *  bandwidths that are to be enabled (1) and disabled (0). Only the lowest
 *  ::amdsmi_frequencies_t::num_supported (of ::amdsmi_pcie_bandwidth_t) bits of
 *  this mask are relevant.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_pci_bandwidth(amdsmi_processor_handle processor_handle,
                                              uint64_t bw_bitmask);

/** @} End PCIeCont */

/*****************************************************************************/
/** @defgroup PowerQuer Power Queries
 *  These functions provide information about power usage.
 *  @{
 */

/**
 *  @brief Get the energy accumulator counter of the processor with provided
 *  processor handle. It is not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a pointer to a uint64_t
 *  @p power, and a pointer to a uint64_t @p timestamp, this function will write
 *  amount of energy consumed to the uint64_t pointed to by @p power,
 *  and the timestamp to the uint64_t pointed to by @p timestamp.
 *  This function accumulates all energy consumed.
 *
 *  @param[in] processor_handle a processor handle
 *  @param[in,out] counter_resolution resolution of the counter @p power in
 *  micro Joules
 *
 *  @param[in,out] energy_accumulator a pointer to uint64_t to which the energy
 *  counter will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[in,out] timestamp a pointer to uint64_t to which the timestamp
 *  will be written. Resolution: 1 ns.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_energy_count(amdsmi_processor_handle processor_handle, uint64_t *energy_accumulator,
                          float *counter_resolution, uint64_t *timestamp);

/** @} End PowerQuer */

/*****************************************************************************/
/** @defgroup PowerCont Power Control
 *  These functions provide ways to control power usage.
 *  @{
 */
/**
 *  @brief Set the maximum gpu power cap value. It is not supported on virtual
 *  machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details This function will set the power cap to the provided value @p cap.
 *  @p cap must be between the minimum and maximum power cap values set by the
 *  system, which can be obtained from ::amdsmi_dev_power_cap_range_get.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a processor has more than one sensor, it could be greater than 0.
 *
 *  @param[in] cap a uint64_t that indicates the desired power cap, in
 *  microwatts
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
 amdsmi_set_power_cap(amdsmi_processor_handle processor_handle, uint32_t sensor_ind, uint64_t cap);

/**
 *  @brief Set the power performance profile. It is not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details This function will attempt to set the current profile to the provided
 *  profile, given a processor handle @p processor_handle and a @p profile. The provided
 *  profile must be one of the currently supported profiles, as indicated by a
 *  call to :: amdsmi_get_gpu_power_profile_presets()
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] reserved Not currently used. Set to 0.
 *
 *  @param[in] profile a ::amdsmi_power_profile_preset_masks_t that hold the mask
 *  of the desired new power profile
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
 amdsmi_set_gpu_power_profile(amdsmi_processor_handle processor_handle, uint32_t reserved,
                                   amdsmi_power_profile_preset_masks_t profile);

/** @} End PowerCont*/

/*****************************************************************************/
/** @defgroup MemQuer Memory Queries
 *  These functions provide information about memory systems.
 *  @{
 */

/**
 *  @brief Get the total amount of memory that exists
 *
 *  @platform{gpu_bm_linux}  @platform{guest_1vf}  @platform{guest_mvf}
 *
 *  @details Given a processor handle @p processor_handle, a type of memory @p mem_type, and
 *  a pointer to a uint64_t @p total, this function will write the total amount
 *  of @p mem_type memory that exists to the location pointed to by @p total.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] mem_type The type of memory for which the total amount will be
 *  found
 *
 *  @param[in,out] total a pointer to uint64_t to which the total amount of
 *  memory will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_memory_total(amdsmi_processor_handle processor_handle, amdsmi_memory_type_t mem_type,
                                                             uint64_t *total);

/**
 *  @brief Get the current memory usage
 *
 *  @platform{gpu_bm_linux}  @platform{guest_1vf}  @platform{guest_mvf}
 *
 *  @details This function will write the amount of @p mem_type memory that
 *  that is currently being used to the location pointed to by @p used.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] mem_type The type of memory for which the amount being used will
 *  be found
 *
 *  @param[in,out] used a pointer to uint64_t to which the amount of memory
 *  currently being used will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_memory_usage(amdsmi_processor_handle processor_handle, amdsmi_memory_type_t mem_type,
                                                              uint64_t *used);

/**
 * @brief Get the bad pages of a processor. It is not supported on virtual
 * machine guest
 *
 * @platform{gpu_bm_linux} @platform{host}
 *
 * @details This call will query the device @p processor_handle for the
 * number of bad pages (written to @p num_pages address). The results are
 * written to address held by the @p info pointer.
 * The first call to this API returns the number of bad pages which
 * should be used to allocate the buffer that should contain the bad page
 * records.
 * @param[in] processor_handle a processor handle
 * @param[out] num_pages Number of bad page records.
 * @param[out] info The results will be written to the
 * amdsmi_retired_page_record_t pointer.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_bad_page_info(amdsmi_processor_handle processor_handle, uint32_t *num_pages,
                              amdsmi_retired_page_record_t *info);

/**
 *  @brief Returns RAS features info.
 *
 *  @platform{gpu_bm_linux}  @platform{host}
 *
 *  @param[in] processor_handle Device handle which to query
 *
 *  @param[out] ras_feature RAS features that are currently enabled and supported on
 *  the processor. Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_ras_feature_info(
  amdsmi_processor_handle processor_handle, amdsmi_ras_feature_t *ras_feature);


/**
 * @brief Returns if RAS features are enabled or disabled for given block. It is not
 * supported on virtual machine guest
 *
 * @platform{gpu_bm_linux}
 *
 * @details Given a processor handle @p processor_handle, this function queries the
 * state of RAS features for a specific block @p block. Result will be written
 * to address held by pointer @p state.
 *
 * @param[in] processor_handle Device handle which to query
 *
 * @param[in] block Block which to query
 *
 * @param[in,out] state A pointer to amdsmi_ras_err_state_t to which the state
 * of block will be written.
 * If this parameter is nullptr, this function will return
 * ::AMDSMI_STATUS_INVAL if the function is supported with the provided
 * arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 * provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_ras_block_features_enabled(amdsmi_processor_handle processor_handle,
                                            amdsmi_gpu_block_t block,
                                            amdsmi_ras_err_state_t *state);

/**
 *  @brief Get information about reserved ("retired") memory pages. It is not supported on
 *  virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, this function returns retired page
 *  information @p records corresponding to the device with the provided processor
 *  handle @p processor_handle. The number of retired page records is returned through @p
 *  num_pages. @p records may be NULL on input. In this case, the number of
 *  records available for retrieval will be returned through @p num_pages.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] num_pages a pointer to a uint32. As input, the value passed
 *  through this parameter is the number of ::amdsmi_retired_page_record_t's that
 *  may be safely written to the memory pointed to by @p records. This is the
 *  limit on how many records will be written to @p records. On return, @p
 *  num_pages will contain the number of records written to @p records, or the
 *  number of records that could have been written if enough memory had been
 *  provided.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[in,out] records A pointer to a block of memory to which the
 *  ::amdsmi_retired_page_record_t values will be written. This value may be NULL.
 *  In this case, this function can be used to query how many records are
 *  available to read.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_memory_reserved_pages(amdsmi_processor_handle processor_handle,
                                      uint32_t *num_pages,
                                      amdsmi_retired_page_record_t *records);

/** @} End MemQuer */

/** @defgroup PhysQuer Physical State Queries
 *  These functions provide information about the physical characteristics of
 *  the device.
 *  @{
 */

/**
 *  @brief Get the fan speed in RPMs of the device with the specified processor
 *  handle and 0-based sensor index. It is not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint32_t
 *  @p speed, this function will write the current fan speed in RPMs to the
 *  uint32_t pointed to by @p speed
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[in,out] speed a pointer to uint32_t to which the speed will be
 *  written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_fan_rpms(amdsmi_processor_handle processor_handle,
                                          uint32_t sensor_ind, int64_t *speed);

/**
 *  @brief Get the fan speed for the specified device as a value relative to
 *  ::AMDSMI_MAX_FAN_SPEED. It is not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint32_t
 *  @p speed, this function will write the current fan speed (a value
 *  between 0 and the maximum fan speed, ::AMDSMI_MAX_FAN_SPEED) to the uint32_t
 *  pointed to by @p speed
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[in,out] speed a pointer to uint32_t to which the speed will be
 *  written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_fan_speed(amdsmi_processor_handle processor_handle,
                                        uint32_t sensor_ind, int64_t *speed);

/**
 *  @brief Get the max. fan speed of the device with provided processor handle. It is
 *  not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint32_t
 *  @p max_speed, this function will write the maximum fan speed possible to
 *  the uint32_t pointed to by @p max_speed
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[in,out] max_speed a pointer to uint32_t to which the maximum speed
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_fan_speed_max(amdsmi_processor_handle processor_handle,
                                    uint32_t sensor_ind, uint64_t *max_speed);

/**
 *  @brief Get the temperature metric value for the specified metric, from the
 *  specified temperature sensor on the specified device. It is not supported on
 *  virtual machine guest
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @details Given a processor handle @p processor_handle, a sensor type @p sensor_type, a
 *  ::amdsmi_temperature_metric_t @p metric and a pointer to an int64_t @p
 *  temperature, this function will write the value of the metric indicated by
 *  @p metric and @p sensor_type to the memory location @p temperature.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_type part of device from which temperature should be
 *  obtained. This should come from the enum ::amdsmi_temperature_type_t
 *
 *  @param[in] metric enum indicated which temperature value should be
 *  retrieved
 *
 *  @param[in,out] temperature a pointer to int64_t to which the temperature
 *  will be written, in millidegrees Celcius.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_temp_metric(amdsmi_processor_handle processor_handle,
                      amdsmi_temperature_type_t sensor_type,
                      amdsmi_temperature_metric_t metric, int64_t *temperature);

/**
 *  @brief Returns gpu cache info.
 *
 *  @platform{gpu_bm_linux}  @platform{host}
 *
 *  @param[in] processor_handle PF of a processor for which to query
 *
 *  @param[out] info reference to the cache info struct.
 *  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_cache_info(
      amdsmi_processor_handle processor_handle, amdsmi_gpu_cache_info_t *info);

/**
 *  @brief Get the voltage metric value for the specified metric, from the
 *  specified voltage sensor on the specified device. It is not supported on
 *  virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a sensor type @p sensor_type, a
 *  ::amdsmi_voltage_metric_t @p metric and a pointer to an int64_t @p
 *  voltage, this function will write the value of the metric indicated by
 *  @p metric and @p sensor_type to the memory location @p voltage.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_type part of device from which voltage should be
 *  obtained. This should come from the enum ::amdsmi_voltage_type_t
 *
 *  @param[in] metric enum indicated which voltage value should be
 *  retrieved
 *
 *  @param[in,out] voltage a pointer to int64_t to which the voltage
 *  will be written, in millivolts.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_volt_metric(amdsmi_processor_handle processor_handle,
                               amdsmi_voltage_type_t sensor_type,
                              amdsmi_voltage_metric_t metric, int64_t *voltage);

/** @} End PhysQuer */

/*****************************************************************************/
/** @defgroup PhysCont Physical State Control
 *  These functions provide control over the physical state of a device.
 *  @{
 */

/**
 *  @brief Reset the fan to automatic driver control. It is not supported on virtual
 *  machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details This function returns control of the fan to the system
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_reset_gpu_fan(amdsmi_processor_handle processor_handle, uint32_t sensor_ind);

/**
 *  @brief Set the fan speed for the specified device with the provided speed,
 *  in RPMs. It is not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a integer value indicating
 *  speed @p speed, this function will attempt to set the fan speed to @p speed.
 *  An error will be returned if the specified speed is outside the allowable
 *  range for the device. The maximum value is 255 and the minimum is 0.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[in] speed the speed to which the function will attempt to set the fan
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_fan_speed(amdsmi_processor_handle processor_handle,
      uint32_t sensor_ind, uint64_t speed);

/** @} End PhysCont */

/*****************************************************************************/
/** @defgroup PerfQuer Clock, Power and Performance Queries
 *  These functions provide information about clock frequencies and
 *  performance.
 *  @{
 */

/**
 *  @brief Get coarse grain utilization counter of the specified device
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, the array of the utilization counters,
 *  the size of the array, this function returns the coarse grain utilization counters
 *  and timestamp.
 *  The counter is the accumulated percentages. Every milliseconds the firmware calculates
 *  % busy count and then accumulates that value in the counter. This provides minimally
 *  invasive coarse grain GPU usage information.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] utilization_counters Multiple utilization counters can be retreived with a single
 *  call. The caller must allocate enough space to the utilization_counters array. The caller also
 *  needs to set valid AMDSMI_UTILIZATION_COUNTER_TYPE type for each element of the array.
 *  ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the provided arguments.
 *
 *  If the function reutrns AMDSMI_STATUS_SUCCESS, the counter will be set in the value field of
 *  the amdsmi_utilization_counter_t.
 *
 *  @param[in] count The size of @p utilization_counters array.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_utilization_count(amdsmi_processor_handle processor_handle,
                amdsmi_utilization_counter_t utilization_counters[],
                uint32_t count,
                uint64_t *timestamp);

/**
 *  @brief Get the performance level of the device. It is not supported on virtual
 *  machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details This function will write the ::amdsmi_dev_perf_level_t to the uint32_t
 *  pointed to by @p perf, for a given processor handle @p processor_handle and a pointer
 *  to a uint32_t @p perf.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] perf a pointer to ::amdsmi_dev_perf_level_t to which the
 *  performance level will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_perf_level(amdsmi_processor_handle processor_handle,
                                                 amdsmi_dev_perf_level_t *perf);

/**
 *  @brief Enter performance determinism mode with provided processor handle. It is
 *  not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and @p clkvalue this function
 *  will enable performance determinism mode, which enforces a GFXCLK frequency
 *  SoftMax limit per GPU set by the user. This prevents the GFXCLK PLL from
 *  stretching when running the same workload on different GPUS, making
 *  performance variation minimal. This call will result in the performance
 *  level ::amdsmi_dev_perf_level_t of the device being
 *  ::AMDSMI_DEV_PERF_LEVEL_DETERMINISM.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] clkvalue Softmax value for GFXCLK in MHz.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_set_gpu_perf_determinism_mode(amdsmi_processor_handle processor_handle, uint64_t clkvalue);

/**
 *  @brief Get the overdrive percent associated with the device with provided
 *  processor handle. It is not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint32_t @p od,
 *  this function will write the overdrive percentage to the uint32_t pointed
 *  to by @p od
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] od a pointer to uint32_t to which the overdrive percentage
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */

amdsmi_status_t amdsmi_get_gpu_overdrive_level(amdsmi_processor_handle processor_handle,
                                                uint32_t *od);

/**
 *  @brief Get the GPU memory clock overdrive percent associated with the device with provided
 *  processor handle. It is not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint32_t @p od,
 *  this function will write the overdrive percentage to the uint32_t pointed
 *  to by @p od
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] od a pointer to uint32_t to which the GPU memory clock overdrive percentage
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_mem_overdrive_level(amdsmi_processor_handle processor_handle,
                                                uint32_t *od);

/**
 *  @brief Get the list of possible system clock speeds of device for a
 *  specified clock type. It is not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}  @platform{guest_1vf}
 *
 *  @details Given a processor handle @p processor_handle, a clock type @p clk_type, and a
 *  pointer to a to an ::amdsmi_frequencies_t structure @p f, this function will
 *  fill in @p f with the possible clock speeds, and indication of the current
 *  clock speed selection.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] clk_type the type of clock for which the frequency is desired
 *
 *  @param[in,out] f a pointer to a caller provided ::amdsmi_frequencies_t structure
 *  to which the frequency information will be written. Frequency values are in
 *  Hz.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_clk_freq(amdsmi_processor_handle processor_handle,
                             amdsmi_clk_type_t clk_type, amdsmi_frequencies_t *f);

/**
 *  @brief Reset the gpu associated with the device with provided processor handle. It is not
 *  supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, this function will reset the GPU
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_reset_gpu(amdsmi_processor_handle processor_handle);

/**
 *  @brief This function retrieves the overdrive GFX & MCLK information. If valid
 *  for the GPU it will also populate the voltage curve data. It is not supported
 *  on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a
 *  ::amdsmi_od_volt_freq_data_t structure @p odv, this function will populate @p
 *  odv. See ::amdsmi_od_volt_freq_data_t for more details.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] odv a pointer to an ::amdsmi_od_volt_freq_data_t structure
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_od_volt_info(amdsmi_processor_handle processor_handle,
                                               amdsmi_od_volt_freq_data_t *odv);

/**
 *  @brief Get the 'metrics_header_info' from the GPU metrics associated with the device
 *
 *  @platform{gpu_bm_linux}  @platform{guest_1vf}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a amd_metrics_table_header_t in which
 *  the 'metrics_header_info' will stored
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[inout] header_value a pointer to amd_metrics_table_header_t to which the device gpu
 *  metric unit will be stored
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS is returned upon successful call.
 *          ::AMDSMI_STATUS_NOT_SUPPORTED is returned in case the metric unit
 *            does not exist for the given device
 *
 */
amdsmi_status_t
amdsmi_get_gpu_metrics_header_info(amdsmi_processor_handle processor_handle, amd_metrics_table_header_t* header_value);

/**
 *  @brief This function retrieves the gpu metrics information. It is not supported
 *  on virtual machine guest
 *
 *  @platform{gpu_bm_linux}  @platform{guest_1vf}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a
 *  ::amdsmi_gpu_metrics_t structure @p pgpu_metrics, this function will populate
 *  @p pgpu_metrics. See ::amdsmi_gpu_metrics_t for more details.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] pgpu_metrics a pointer to an ::amdsmi_gpu_metrics_t structure
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_metrics_info(amdsmi_processor_handle processor_handle,
                                            amdsmi_gpu_metrics_t *pgpu_metrics);

/**
 *  @brief Get the pm metrics table with provided device index.
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a device handle @p processor_handle, @p pm_metrics pointer,
 *  and @p num_of_metrics pointer,
 *  this function will write the pm metrics name value pair
 *  to the array at @p pm_metrics and the number of metrics retreived to @p num_of_metrics
 *  Note: the library allocated memory for pm_metrics, and user must call
 *  free(pm_metrics) to free it after use.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[inout] pm_metrics A pointerto an array to hold multiple PM metrics. On successs,
 *  the library will allocate memory of pm_metrics and write metrics to this array.
 *  The caller must free this memory after usage to avoid memory leak.
 *
 *  @param[inout] num_of_metrics a pointer to uint32_t to which the number of
 *  metrics is allocated for pm_metrics array as input, and the number of metrics retreived
 *  as output. If this parameter is NULL, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS call was successful
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
 *
 */
amdsmi_status_t amdsmi_get_gpu_pm_metrics_info(
                      amdsmi_processor_handle processor_handle,
                      amdsmi_name_value_t** pm_metrics,
                      uint32_t *num_of_metrics);

/**
 *  @brief Get the register metrics table with provided device index and register type.
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a device handle @p processor_handle, @p reg_type, @p reg_metrics pointer,
 *  and @p num_of_metrics pointer,
 *  this function will write the register metrics name value pair
 *  to the array at @p reg_metrics and the number of metrics retreived to @p num_of_metrics
 *  Note: the library allocated memory for reg_metrics, and user must call
 *  free(reg_metrics) to free it after use.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] reg_type The register type
 *
 *  @param[inout] reg_metrics A pointerto an array to hold multiple register metrics. On successs,
 *  the library will allocate memory of reg_metrics and write metrics to this array.
 *  The caller must free this memory after usage to avoid memory leak.
 *
 *  @param[inout] num_of_metrics a pointer to uint32_t to which the number of
 *  metrics is allocated for reg_metrics array as input, and the number of metrics retreived
 *  as output. If this parameter is NULL, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS call was successful
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
 *
 */
amdsmi_status_t amdsmi_get_gpu_reg_table_info(
                      amdsmi_processor_handle processor_handle,
                      amdsmi_reg_type_t reg_type,
                      amdsmi_name_value_t** reg_metrics,
                      uint32_t *num_of_metrics);

/**
 *  @brief This function sets the clock range information. It is not supported on virtual
 *  machine guest
 *
 *  @deprecated ::amdsmi_set_gpu_clk_limit() should be used, with an
 *  interface that set the min_value and then max_value.
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a minimum clock value @p minclkvalue,
 *  a maximum clock value @p maxclkvalue and a clock type @p clkType this function
 *  will set the sclk|mclk range
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] minclkvalue value to apply to the clock range. Frequency values
 *  are in MHz.
 *
 *  @param[in] maxclkvalue value to apply to the clock range. Frequency values
 *  are in MHz.
 *
 *  @param[in] clkType AMDSMI_CLK_TYPE_SYS | AMDSMI_CLK_TYPE_MEM range type
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_clk_range(amdsmi_processor_handle processor_handle,
                                          uint64_t minclkvalue,
                                          uint64_t maxclkvalue,
                                          amdsmi_clk_type_t clkType);

/**
 *  @brief This function sets the clock sets the clock min/max level
 *
 *  @platform{gpu_bm_linux} @platform{guest_1vf}
 *
 *  @details Given a processor handle @p processor_handle, a clock type @p clk_type,
 *  a value @p clk_value needs to be set, and the @p level indicates min or max
 *  clock you want to set, this function the clock limit.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] clk_type AMDSMI_CLK_TYPE_SYS, AMDSMI_CLK_TYPE_MEM and so on
 *
 *  @param[in] limit_type AMDSMI_FREQ_IND_MIN|AMDSMI_FREQ_IND_MAX to set the
 *  minimum (0) or maximum (1) speed.
 *
 *  @param[in] clk_value value to apply to. Frequency values are in MHz.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_clk_limit(amdsmi_processor_handle processor_handle,
                                         amdsmi_clk_type_t clk_type,
                                         amdsmi_clk_limit_type_t limit_type,
                                         uint64_t clk_value);

/**
 *  @brief Frees heap memory allocated by reg_table and pm_metrics
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Frees heap memory.
 *
 *  @param[in] p a pointer to the memory to free.
 *
 */
void amdsmi_free_name_value_pairs(void *p);

/**
 *  @brief This function sets the clock frequency information. It is not supported on
 *  virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a frequency level @p level,
 *  a clock value @p clkvalue and a clock type @p clkType this function
 *  will set the sclk|mclk range
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] level AMDSMI_FREQ_IND_MIN|AMDSMI_FREQ_IND_MAX to set the
 *  minimum (0) or maximum (1) speed.
 *
 *  @param[in] clkvalue value to apply to the clock range. Frequency values
 *  are in MHz.
 *
 *  @param[in] clkType AMDSMI_CLK_TYPE_SYS | AMDSMI_CLK_TYPE_MEM range type
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_od_clk_info(amdsmi_processor_handle processor_handle,
                                        amdsmi_freq_ind_t level,
                                        uint64_t clkvalue,
                                        amdsmi_clk_type_t clkType);

/**
 *  @brief This function sets  1 of the 3 voltage curve points. It is not supported
 *  on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a voltage point @p vpoint
 *  and a voltage value @p voltvalue this function will set voltage curve point
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] vpoint voltage point [0|1|2] on the voltage curve
 *
 *  @param[in] clkvalue clock value component of voltage curve point.
 *  Frequency values are in MHz.
 *
 *  @param[in] voltvalue voltage value component of voltage curve point.
 *  Voltage is in mV.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_od_volt_info(amdsmi_processor_handle processor_handle,
                                              uint32_t vpoint,
                                              uint64_t clkvalue,
                                              uint64_t voltvalue);

/**
 *  @brief This function will retrieve the current valid regions in the
 *  frequency/voltage space. It is not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a pointer to an unsigned integer
 *  @p num_regions and a buffer of ::amdsmi_freq_volt_region_t structures, @p
 *  buffer, this function will populate @p buffer with the current
 *  frequency-volt space regions. The caller should assign @p buffer to memory
 *  that can be written to by this function. The caller should also
 *  indicate the number of ::amdsmi_freq_volt_region_t structures that can safely
 *  be written to @p buffer in @p num_regions.
 *
 *  The number of regions to expect this function provide (@p num_regions) can
 *  be obtained by calling :: amdsmi_get_gpu_od_volt_info().
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] num_regions As input, this is the number of
 *  ::amdsmi_freq_volt_region_t structures that can be written to @p buffer. As
 *  output, this is the number of ::amdsmi_freq_volt_region_t structures that were
 *  actually written.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[in,out] buffer a caller provided buffer to which
 *  ::amdsmi_freq_volt_region_t structures will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_od_volt_curve_regions(amdsmi_processor_handle processor_handle,
                      uint32_t *num_regions, amdsmi_freq_volt_region_t *buffer);

/**
 *  @brief Get the list of available preset power profiles and an indication of
 *  which profile is currently active. It is not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a
 *  ::amdsmi_power_profile_status_t @p status, this function will set the bits of
 *  the ::amdsmi_power_profile_status_t.available_profiles bit field of @p status to
 *  1 if the profile corresponding to the respective
 *  ::amdsmi_power_profile_preset_masks_t profiles are enabled. For example, if both
 *  the VIDEO and VR power profiles are available selections, then
 *  ::AMDSMI_PWR_PROF_PRST_VIDEO_MASK AND'ed with
 *  ::amdsmi_power_profile_status_t.available_profiles will be non-zero as will
 *  ::AMDSMI_PWR_PROF_PRST_VR_MASK AND'ed with
 *  ::amdsmi_power_profile_status_t.available_profiles. Additionally,
 *  ::amdsmi_power_profile_status_t.current will be set to the
 *  ::amdsmi_power_profile_preset_masks_t of the profile that is currently active.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[in,out] status a pointer to ::amdsmi_power_profile_status_t that will be
 *  populated by a call to this function
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
 amdsmi_get_gpu_power_profile_presets(amdsmi_processor_handle processor_handle, uint32_t sensor_ind,
                                         amdsmi_power_profile_status_t *status);

/** @} End PerfQuer */

/*****************************************************************************/
/** @defgroup PerfCont Clock, Power and Performance Control
 *  These functions provide control over clock frequencies, power and
 *  performance.
 *  @{
 */

/**
 *  @brief Set the PowerPlay performance level associated with the device with
 *  provided processor handle with the provided value. It is not supported
 *  on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and an ::amdsmi_dev_perf_level_t @p
 *  perf_level, this function will set the PowerPlay performance level for the
 *  device to the value @p perf_lvl.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] perf_lvl the value to which the performance level should be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
 amdsmi_set_gpu_perf_level(amdsmi_processor_handle processor_handle, amdsmi_dev_perf_level_t perf_lvl);

/**
 *  @brief Set the overdrive percent associated with the device with provided
 *  processor handle with the provided value. See details for WARNING. It is
 *  not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and an overdrive level @p od,
 *  this function will set the overdrive level for the device to the value
 *  @p od. The overdrive level is an integer value between 0 and 20, inclusive,
 *  which represents the overdrive percentage; e.g., a value of 5 specifies
 *  an overclocking of 5%.
 *
 *  The overdrive level is specific to the gpu system clock.
 *
 *  The overdrive level is the percentage above the maximum Performance Level
 *  to which overclocking will be limited. The overclocking percentage does
 *  not apply to clock speeds other than the maximum. This percentage is
 *  limited to 20%.
 *
 *   ******WARNING******
 *  Operating your AMD GPU outside of official AMD specifications or outside of
 *  factory settings, including but not limited to the conducting of
 *  overclocking (including use of this overclocking software, even if such
 *  software has been directly or indirectly provided by AMD or otherwise
 *  affiliated in any way with AMD), may cause damage to your AMD GPU, system
 *  components and/or result in system failure, as well as cause other problems.
 *  DAMAGES CAUSED BY USE OF YOUR AMD GPU OUTSIDE OF OFFICIAL AMD SPECIFICATIONS
 *  OR OUTSIDE OF FACTORY SETTINGS ARE NOT COVERED UNDER ANY AMD PRODUCT
 *  WARRANTY AND MAY NOT BE COVERED BY YOUR BOARD OR SYSTEM MANUFACTURER'S
 *  WARRANTY. Please use this utility with caution.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] od the value to which the overdrive level should be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_overdrive_level(amdsmi_processor_handle processor_handle, uint32_t od);

/**
 * @brief Control the set of allowed frequencies that can be used for the
 * specified clock. It is not supported on virtual machine guest
 *
 * @platform{gpu_bm_linux}
 *
 * @details Given a processor handle @p processor_handle, a clock type @p clk_type, and a
 * 64 bit bitmask @p freq_bitmask, this function will limit the set of
 * allowable frequencies. If a bit in @p freq_bitmask has a value of 1, then
 * the frequency (as ordered in an ::amdsmi_frequencies_t returned by
 *  amdsmi_get_clk_freq()) corresponding to that bit index will be
 * allowed.
 *
 * This function will change the performance level to
 * ::AMDSMI_DEV_PERF_LEVEL_MANUAL in order to modify the set of allowable
 * frequencies. Caller will need to set to ::AMDSMI_DEV_PERF_LEVEL_AUTO in order
 * to get back to default state.
 *
 * All bits with indices greater than or equal to
 * ::amdsmi_frequencies_t::num_supported will be ignored.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] clk_type the type of clock for which the set of frequencies
 *  will be modified
 *
 *  @param[in] freq_bitmask A bitmask indicating the indices of the
 *  frequencies that are to be enabled (1) and disabled (0). Only the lowest
 *  ::amdsmi_frequencies_t.num_supported bits of this mask are relevant.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_clk_freq(amdsmi_processor_handle processor_handle,
                             amdsmi_clk_type_t clk_type, uint64_t freq_bitmask);

/**
 * @brief Get the soc pstate policy for the processor
 *
 * @platform{gpu_bm_linux} @platform{guest_1vf}
 *
 * @details Given a processor handle @p processor_handle, this function will write
 * current soc pstate  policy settings to @p policy. All the processors at the same socket
 * will have the same policy.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in, out] policy the soc pstate  policy for this processor.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_soc_pstate(amdsmi_processor_handle processor_handle,
                             amdsmi_dpm_policy_t* policy);

/**
 * @brief Set the soc pstate  policy for the processor
 *
 * @platform{gpu_bm_linux} @platform{guest_1vf}
 *
 * @details Given a processor handle @p processor_handle and a soc pstate  policy @p policy_id,
 * this function will set the soc pstate  policy for this processor. All the processors at
 * the same socket will be set to the same policy.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] policy_id the soc pstate  policy id to set. The id is the id in
 *  amdsmi_dpm_policy_entry_t, which can be obtained by calling
 *  amdsmi_get_soc_pstate()
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_soc_pstate(amdsmi_processor_handle processor_handle,
                             uint32_t policy_id);

/**
 * @brief Get the xgmi per-link power down policy parameter for the processor
 *
 * @platform{gpu_bm_linux}
 *
 * @details Given a processor handle @p processor_handle, this function will write
 * current xgmi plpd settings to @p policy. All the processors at the same socket
 * will have the same policy.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in, out] policy the xgmi plpd for this processor.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_xgmi_plpd(amdsmi_processor_handle processor_handle,
                             amdsmi_dpm_policy_t* xgmi_plpd);

/**
 * @brief Set the xgmi per-link power down policy parameter for the processor
 *
 * @platform{gpu_bm_linux}
 *
 * @details Given a processor handle @p processor_handle and a dpm policy @p plpd_id,
 * this function will set the xgmi plpd for this processor. All the processors at
 * the same socket will be set to the same policy.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] xgmi_plpd_id the xgmi plpd id to set. The id is the id in
 *  amdsmi_dpm_policy_entry_t, which can be obtained by calling
 *  amdsmi_get_xgmi_plpd()
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_xgmi_plpd(amdsmi_processor_handle processor_handle,
                             uint32_t plpd_id);


/**
 * @brief Get the status of the Process Isolation
 *
 * @platform{gpu_bm_linux} @platform{guest_1vf}
 *
 * @details Given a processor handle @p processor_handle, this function will write
 * current process isolation status to @p pisolate. The 0 is the process isolation
 * disabled, and the 1 is the process isolation enabled.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in, out] pisolate the process isolation status.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_process_isolation(amdsmi_processor_handle processor_handle,
                             uint32_t* pisolate);

/**
 * @brief Enable/disable the system Process Isolation
 *
 * @platform{gpu_bm_linux} @platform{guest_1vf}
 *
 * @details Given a processor handle @p processor_handle and a process isolation @p pisolate,
 * flag, this function will set the Process Isolation for this processor. The 0 is the process
 * isolation disabled, and the 1 is the process isolation enabled.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] pisolate the process isolation status to set.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_process_isolation(amdsmi_processor_handle processor_handle,
                             uint32_t pisolate);

/**
 * @brief Clean up local data in LDS/GPRs
 *
 * @platform{gpu_bm_linux} @platform{guest_1vf}
 *
 * @details Given a processor handle @p processor_handle,
 * this function will clean the local data of this processor. This can be called between
 * user logins to prevent information leak.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_clean_gpu_local_data(amdsmi_processor_handle processor_handle);

/** @} End PerfCont */

/*****************************************************************************/
/** @defgroup VersQuer Version Queries
 *  These functions provide version information about various subsystems.
 *  @{
 */

/**
 * @brief Get the build version information for the currently running build of
 * AMDSMI.
 *
 * @platform{gpu_bm_linux} @platform{cpu_bm}  @platform{guest_1vf}  @platform{guest_mvf}
 *
 * @details  Get the major, minor, patch and build string for AMDSMI build
 * currently in use through @p version
 *
 * @param[in,out] version A pointer to an ::amdsmi_version_t structure that will
 * be updated with the version information upon return.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_lib_version(amdsmi_version_t *version);

/** @} End VersQuer */

/*****************************************************************************/
/** @defgroup ErrQuer Error Queries
 *  These functions provide error information about AMDSMI calls as well as
 *  device errors.
 *  @{
 */

/**
 *  @brief Retrieve the error counts for a GPU block. It is not supported on virtual
 *  machine guest
 *
 *  @platform{gpu_bm_linux}  @platform{host}
 *
 *  @details Given a processor handle @p processor_handle, an ::amdsmi_gpu_block_t @p block and a
 *  pointer to an ::amdsmi_error_count_t @p ec, this function will write the error
 *  count values for the GPU block indicated by @p block to memory pointed to by
 *  @p ec.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] block The block for which error counts should be retrieved
 *
 *  @param[in,out] ec A pointer to an ::amdsmi_error_count_t to which the error
 *  counts should be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_ecc_count(amdsmi_processor_handle processor_handle,
                              amdsmi_gpu_block_t block, amdsmi_error_count_t *ec);

/**
 *  @brief Retrieve the enabled ECC bit-mask. It is not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @details Given a processor handle @p processor_handle, and a pointer to a uint64_t @p
 *  enabled_mask, this function will write bits to memory pointed to by
 *  @p enabled_blocks. Upon a successful call, @p enabled_blocks can then be
 *  AND'd with elements of the ::amdsmi_gpu_block_t ennumeration to determine if
 *  the corresponding block has ECC enabled. Note that whether a block has ECC
 *  enabled or not in the device is independent of whether there is kernel
 *  support for error counting for that block. Although a block may be enabled,
 *  but there may not be kernel support for reading error counters for that
 *  block.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] enabled_blocks A pointer to a uint64_t to which the enabled
 *  blocks bits will be written.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_ecc_enabled(amdsmi_processor_handle processor_handle,
                                                    uint64_t *enabled_blocks);

/**
 *  @brief Retrieve the ECC status for a GPU block. It is not supported on virtual machine
 *  guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, an ::amdsmi_gpu_block_t @p block and
 *  a pointer to an ::amdsmi_ras_err_state_t @p state, this function will write
 *  the current state for the GPU block indicated by @p block to memory pointed
 *  to by @p state.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] block The block for which error counts should be retrieved
 *
 *  @param[in,out] state A pointer to an ::amdsmi_ras_err_state_t to which the
 *  ECC state should be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_ecc_status(amdsmi_processor_handle processor_handle,
                                                  amdsmi_gpu_block_t block,
                                                  amdsmi_ras_err_state_t *state);

/**
 *  @brief Get a description of a provided AMDSMI error status
 *
 * @platform{gpu_bm_linux}  @platform{host} @platform{cpu_bm}  @platform{guest_1vf}  @platform{guest_mvf}
 *
 *  @details Set the provided pointer to a const char *, @p status_string, to
 *  a string containing a description of the provided error code @p status.
 *
 *  @param[in] status The error status for which a description is desired
 *
 *  @param[in,out] status_string A pointer to a const char * which will be made
 *  to point to a description of the provided error code
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_status_code_to_string(amdsmi_status_t status, const char **status_string);

/** @} End ErrQuer */

/*****************************************************************************/
/** @defgroup PerfCntr Performance Counter Functions
 *  These functions are used to configure, query and control performance
 *  counting.
 *
 *  These functions use the same mechanisms as the "perf" command line
 *  utility. They share the same underlying resources and have some similarities
 *  in how they are used. The events supported by this API should have
 *  corresponding perf events that can be seen with "perf stat ...". The events
 *  supported by perf can be seen with "perf list"
 *
 *  The types of events available and the ability to count those
 *  events are dependent on which device is being targeted and if counters are
 *  still available for that device, respectively.
 *  ::amdsmi_gpu_counter_group_supported() can be used to see which event types
 *  (::amdsmi_event_group_t) are supported for a given device. Assuming a device
 *  supports a given event type, we can then check to see if there are counters
 *  available to count a specific event with
 *  :: amdsmi_get_gpu_available_counters(). Counters may be occupied by other
 *  perf based programs.
 *
 *  Once it is determined that events are supported and counters are available,
 *  an event counter can be created/destroyed and controlled.
 *
 *  ::amdsmi_gpu_create_counter() allocates internal data structures that will be
 *  used to used to control the event counter, and return a handle to this data
 *  structure.
 *
 *  Once an event counter handle is obtained, the event counter can be
 *  controlled (i.e., started, stopped,...) with ::amdsmi_gpu_control_counter() by
 *  passing ::amdsmi_counter_command_t commands. ::AMDSMI_CNTR_CMD_START starts an
 *  event counter and ::AMDSMI_CNTR_CMD_STOP stops a counter.
 *  ::amdsmi_gpu_read_counter() reads an event counter.
 *
 *  Once the counter is no longer needed, the resources it uses should be freed
 *  by calling ::amdsmi_gpu_destroy_counter().
 *
 *
 *  Important Notes about Counter Values
 *  ====================================
 *  - A running "absolute" counter is kept internally. For the discussion that
 *  follows, we will call the internal counter value at time @a t @a
 *  val<sub>t</sub>
 *  - Issuing ::AMDSMI_CNTR_CMD_START or calling ::amdsmi_gpu_read_counter(), causes
 *  AMDSMI (in kernel) to internally record the current absolute counter value
 *  - ::amdsmi_gpu_read_counter() returns the number of events that have occurred
 *  since the previously recorded value (ie, a relative value,
 *  @a val<sub>t</sub> - val<sub>t-1</sub>) from the issuing of
 *  ::AMDSMI_CNTR_CMD_START or calling ::amdsmi_gpu_read_counter()
 *
 *  Example of event counting sequence:
 *
 *  @latexonly
 *  \pagebreak
 *  @endlatexonly
 *  @code{.cpp}
 *
 *    amdsmi_counter_value_t value;
 *
 *    // Determine if AMDSMI_EVNT_GRP_XGMI is supported for device dv_ind
 *    ret = amdsmi_gpu_counter_group_supported(dv_ind, AMDSMI_EVNT_GRP_XGMI);
 *
 *    // See if there are counters available for device dv_ind for event
 *    // AMDSMI_EVNT_GRP_XGMI
 *
 *    ret =  amdsmi_get_gpu_available_counters(dv_ind,
 *                                 AMDSMI_EVNT_GRP_XGMI, &counters_available);
 *
 *    // Assuming AMDSMI_EVNT_GRP_XGMI is supported and there is at least 1
 *    // counter available for AMDSMI_EVNT_GRP_XGMI on device dv_ind, create
 *    // an event object for an event of group AMDSMI_EVNT_GRP_XGMI (e.g.,
 *    // AMDSMI_EVNT_XGMI_0_BEATS_TX) and get the handle
 *    // (amdsmi_event_handle_t).
 *
 *    ret = amdsmi_gpu_create_counter(dv_ind, AMDSMI_EVNT_XGMI_0_BEATS_TX,
 *                                                          &evnt_handle);
 *
 *    // A program that generates the events of interest can be started
 *    // immediately before or after starting the counters.
 *    // Start counting:
 *    ret = amdsmi_gpu_control_counter(evnt_handle, AMDSMI_CNTR_CMD_START, NULL);
 *
 *    // Wait...
 *
 *    // Get the number of events since AMDSMI_CNTR_CMD_START was issued:
 *    ret = amdsmi_gpu_read_counter(amdsmi_event_handle_t evt_handle, &value)
 *
 *    // Wait...
 *
 *    // Get the number of events since amdsmi_gpu_read_counter() was last called:
 *    ret = amdsmi_gpu_read_counter(amdsmi_event_handle_t evt_handle, &value)
 *
 *    // Stop counting.
 *    ret = amdsmi_gpu_control_counter(evnt_handle, AMDSMI_CNTR_CMD_STOP, NULL);
 *
 *    // Release all resources (e.g., counter and memory resources) associated
 *    with evnt_handle.
 *    ret = amdsmi_gpu_destroy_counter(evnt_handle);
 *  @endcode
 *  @{
 */

/**
 *  @brief Tell if an event group is supported by a given device. It is not supported
 *  on virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and an event group specifier @p
 *  group, tell if @p group type events are supported by the device associated
 *  with @p processor_handle
 *
 *  @param[in] processor_handle processor handle of device being queried
 *
 *  @param[in] group ::amdsmi_event_group_t identifier of group for which support
 *  is being queried
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_gpu_counter_group_supported(amdsmi_processor_handle processor_handle, amdsmi_event_group_t group);

/**
 *  @brief Create a performance counter object
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Create a performance counter object of type @p type for the device
 *  with a processor handle of @p processor_handle, and write a handle to the object to the
 *  memory location pointed to by @p evnt_handle. @p evnt_handle can be used
 *  with other performance event operations. The handle should be deallocated
 *  with ::amdsmi_gpu_destroy_counter() when no longer needed.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] type the ::amdsmi_event_type_t of performance event to create
 *
 *  @param[in,out] evnt_handle A pointer to a ::amdsmi_event_handle_t which will be
 *  associated with a newly allocated counter
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_gpu_create_counter(amdsmi_processor_handle processor_handle, amdsmi_event_type_t type,
                                            amdsmi_event_handle_t *evnt_handle);

/**
 *  @brief Deallocate a performance counter object
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Deallocate the performance counter object with the provided
 *  ::amdsmi_event_handle_t @p evnt_handle
 *
 *  @note This function requires root access
 *
 *  @param[in] evnt_handle handle to event object to be deallocated
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_gpu_destroy_counter(amdsmi_event_handle_t evnt_handle);

/**
 *  @brief Issue performance counter control commands. It is not supported on
 *  virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Issue a command @p cmd on the event counter associated with the
 *  provided handle @p evt_handle.
 *
 *  @note This function requires root access
 *
 *  @param[in] evt_handle an event handle
 *
 *  @param[in] cmd The event counter command to be issued
 *
 *  @param[in,out] cmd_args Currently not used. Should be set to NULL.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_gpu_control_counter(amdsmi_event_handle_t evt_handle,
                                  amdsmi_counter_command_t cmd, void *cmd_args);

/**
 *  @brief Read the current value of a performance counter
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Read the current counter value of the counter associated with the
 *  provided handle @p evt_handle and write the value to the location pointed
 *  to by @p value.
 *
 *  @note This function requires root access
 *
 *  @param[in] evt_handle an event handle
 *
 *  @param[in,out] value pointer to memory of size of ::amdsmi_counter_value_t to
 *  which the counter value will be written
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_gpu_read_counter(amdsmi_event_handle_t evt_handle,
                    amdsmi_counter_value_t *value);

/**
 *  @brief Get the number of currently available counters. It is not supported on
 *  virtual machine guest
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a performance event group @p grp,
 *  and a pointer to a uint32_t @p available, this function will write the
 *  number of @p grp type counters that are available on the device with handle
 *  @p processor_handle to the memory that @p available points to.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] grp an event device group
 *
 *  @param[in,out] available A pointer to a uint32_t to which the number of
 *  available counters will be written
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
 amdsmi_get_gpu_available_counters(amdsmi_processor_handle processor_handle,
                                 amdsmi_event_group_t grp, uint32_t *available);

/** @} End PerfCntr */

/*****************************************************************************/
/** @defgroup SysInfo System Information Functions
 *  These functions are used to configure, query and control performance
 *  counting.
 *  @{
 */

/**
 *  @brief Get process information about processes currently using GPU
 *
 *  @platform{gpu_bm_linux}  @platform{guest_1vf}
 *
 *  @details Given a non-NULL pointer to an array @p procs of
 *  ::amdsmi_process_info_t's, of length *@p num_items, this function will write
 *  up to *@p num_items instances of ::amdsmi_process_info_t to the memory pointed
 *  to by @p procs. These instances contain information about each process
 *  utilizing a GPU. If @p procs is not NULL, @p num_items will be updated with
 *  the number of processes actually written. If @p procs is NULL, @p num_items
 *  will be updated with the number of processes for which there is current
 *  process information. Calling this function with @p procs being NULL is a way
 *  to determine how much memory should be allocated for when @p procs is not
 *  NULL.
 *
 *  @param[in,out] procs a pointer to memory provided by the caller to which
 *  process information will be written. This may be NULL in which case only @p
 *  num_items will be updated with the number of processes found.
 *
 *  @param[in,out] num_items A pointer to a uint32_t, which on input, should
 *  contain the amount of memory in ::amdsmi_process_info_t's which have been
 *  provided by the @p procs argument. On output, if @p procs is non-NULL, this
 *  will be updated with the number ::amdsmi_process_info_t structs actually
 *  written. If @p procs is NULL, this argument will be updated with the number
 *  processes for which there is information.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_compute_process_info(amdsmi_process_info_t *procs, uint32_t *num_items);

/**
 *  @brief Get process information about a specific process
 *
 *  @platform{gpu_bm_linux}  @platform{guest_1vf}
 *
 *  @details Given a pointer to an ::amdsmi_process_info_t @p proc and a process
 *  id
 *  @p pid, this function will write the process information for @p pid, if
 *  available, to the memory pointed to by @p proc.
 *
 *  @param[in] pid The process ID for which process information is being
 *  requested
 *
 *  @param[in,out] proc a pointer to a ::amdsmi_process_info_t to which
 *  process information for @p pid will be written if it is found.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_compute_process_info_by_pid(uint32_t pid, amdsmi_process_info_t *proc);

/**
 *  @brief Get the device indices currently being used by a process
 *
 *   @platform{gpu_bm_linux}  @platform{guest_1vf}
 *
 *  @details Given a process id @p pid, a non-NULL pointer to an array of
 *  uint32_t's @p processor_handleices of length *@p num_devices, this function will
 *  write up to @p num_devices device indices to the memory pointed to by
 *  @p processor_handleices. If @p processor_handleices is not NULL, @p num_devices will be
 *  updated with the number of gpu's currently being used by process @p pid.
 *  If @p processor_handleices is NULL, @p processor_handleices will be updated with the number of
 *  gpus currently being used by @p pid. Calling this function with @p
 *  dv_indices being NULL is a way to determine how much memory is required
 *  for when @p processor_handleices is not NULL.
 *
 *  @param[in] pid The process id of the process for which the number of gpus
 *  currently being used is requested
 *
 *  @param[in,out] dv_indices a pointer to memory provided by the caller to
 *  which indices of devices currently being used by the process will be
 *  written. This may be NULL in which case only @p num_devices will be
 *  updated with the number of devices being used.
 *
 *  @param[in,out] num_devices A pointer to a uint32_t, which on input, should
 *  contain the amount of memory in uint32_t's which have been provided by the
 *  @p processor_handleices argument. On output, if @p processor_handleices is non-NULL, this will
 *  be updated with the number uint32_t's actually written. If @p processor_handleices is
 *  NULL, this argument will be updated with the number devices being used.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_compute_process_gpus(uint32_t pid, uint32_t *dv_indices,
                                                       uint32_t *num_devices);

/** @} End SysInfo */

/*****************************************************************************/
/** @defgroup XGMIInfo XGMI Functions
 *  These functions are used to configure, query and control XGMI.
 *  @{
 */

/**
 *  @brief Retrieve the XGMI error status for a device. It is not supported on
 *  virtual machine guest
 *
 *   @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, and a pointer to an
 *  ::amdsmi_xgmi_status_t @p status, this function will write the current XGMI
 *  error state ::amdsmi_xgmi_status_t for the device @p processor_handle to the memory
 *  pointed to by @p status.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] status A pointer to an ::amdsmi_xgmi_status_t to which the
 *  XGMI error state should be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_gpu_xgmi_error_status(amdsmi_processor_handle processor_handle, amdsmi_xgmi_status_t *status);

/**
 * @brief Reset the XGMI error status for a device. It is not supported on virtual
 * machine guest
 *
 * @platform{gpu_bm_linux}
 *
 * @details Given a processor handle @p processor_handle, this function will reset the
 * current XGMI error state ::amdsmi_xgmi_status_t for the device @p processor_handle to
 * amdsmi_xgmi_status_t::AMDSMI_XGMI_STATUS_NO_ERRORS
 *
 * @param[in] processor_handle a processor handle
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_reset_gpu_xgmi_error(amdsmi_processor_handle processor_handle);

/** @} End SysInfo */

/*****************************************************************************/
/** @defgroup HWTopo Hardware Topology Functions
 *  These functions are used to query Hardware topology.
 *  @{
 */

/**
 *  @brief Return link metric information
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @param[in] processor_handle PF of a processor for which to query
 *
 *  @param[out] link_metrics reference to the link metrics struct.
 *  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_link_metrics(amdsmi_processor_handle processor_handle,
          amdsmi_link_metrics_t *link_metrics);

/**
 *  @brief Retrieve the NUMA CPU node number for a device
 *
 *   @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, and a pointer to an
 *  uint32_t @p numa_node, this function will write the
 *  node number of NUMA CPU for the device @p processor_handle to the memory
 *  pointed to by @p numa_node.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] numa_node A pointer to an uint32_t to which the
 *  numa node number should be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_topo_get_numa_node_number(amdsmi_processor_handle processor_handle, uint32_t *numa_node);

/**
 *  @brief Retrieve the weight for a connection between 2 GPUs
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a source processor handle @p processor_handle_src and
 *  a destination processor handle @p processor_handle_dst, and a pointer to an
 *  uint64_t @p weight, this function will write the
 *  weight for the connection between the device @p processor_handle_src
 *  and @p processor_handle_dst to the memory pointed to by @p weight.
 *
 *  @param[in] processor_handle_src the source processor handle
 *
 *  @param[in] processor_handle_dst the destination processor handle
 *
 *  @param[in,out] weight A pointer to an uint64_t to which the
 *  weight for the connection should be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_topo_get_link_weight(amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst,
                          uint64_t *weight);

/**
 *  @brief Retreive minimal and maximal io link bandwidth between 2 GPUs
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a source processor handle @p processor_handle_src and
 *  a destination processor handle @p processor_handle_dst,  pointer to an
 *  uint64_t @p min_bandwidth, and a pointer to uint64_t @p max_bandiwidth,
 *  this function will write theoretical minimal and maximal bandwidth limits.
 *  API works if src and dst are connected via xgmi and have 1 hop distance.
 *
 *  @param[in] processor_handle_src the source processor handle
 *
 *  @param[in] processor_handle_dst the destination processor handle
 *
 *  @param[in,out] min_bandwidth A pointer to an uint64_t to which the
 *  minimal bandwidth for the connection should be written.
 *
 *  @param[in,out] max_bandwidth A pointer to an uint64_t to which the
 *  maximal bandwidth for the connection should be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
 amdsmi_get_minmax_bandwidth_between_processors(amdsmi_processor_handle processor_handle_src,
                                                  amdsmi_processor_handle processor_handle_dst,
                                                  uint64_t *min_bandwidth,
                                                  uint64_t *max_bandwidth);

/**
 *  @brief Retrieve the hops and the connection type between 2 GPUs
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a source processor handle @p processor_handle_src and
 *  a destination processor handle @p processor_handle_dst, and a pointer to an
 *  uint64_t @p hops and a pointer to an AMDSMI_IO_LINK_TYPE @p type,
 *  this function will write the number of hops and the connection type
 *  between the device @p processor_handle_src and @p processor_handle_dst to the memory
 *  pointed to by @p hops and @p type.
 *
 *  @param[in] processor_handle_src the source processor handle
 *
 *  @param[in] processor_handle_dst the destination processor handle
 *
 *  @param[in,out] hops A pointer to an uint64_t to which the
 *  hops for the connection should be written.
 *
 *  @param[in,out] type A pointer to an ::AMDSMI_IO_LINK_TYPE to which the
 *  type for the connection should be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_topo_get_link_type(amdsmi_processor_handle processor_handle_src,
                        amdsmi_processor_handle processor_handle_dst,
                        uint64_t *hops, amdsmi_io_link_type_t *type);

/**
 *  @brief Return P2P availability status between 2 GPUs
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a source processor handle @p processor_handle_src and
 *  a destination processor handle @p processor_handle_dst, and a pointer to a
 *  bool @p accessible, this function will write the P2P connection status
 *  between the device @p processor_handle_src and @p processor_handle_dst to the memory
 *  pointed to by @p accessible.
 *
 *  @param[in] processor_handle_src the source processor handle
 *
 *  @param[in] processor_handle_dst the destination processor handle
 *
 *  @param[in,out] accessible A pointer to a bool to which the status for
 *  the P2P connection availablity should be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_is_P2P_accessible(amdsmi_processor_handle processor_handle_src,
                          amdsmi_processor_handle processor_handle_dst,
                          bool *accessible);


/**
 *  @brief Retrieve connection type and P2P capabilities between 2 GPUs
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf}  @platform{guest_mvf}
 *
 *  @details Given a source processor handle @p processor_handle_src and
 *  a destination processor handle @p processor_handle_dst, a pointer to an amdsmi_io_link_type_t @p type,
 *  and a pointer to amdsmi_p2p_capability_t @p cap. This function will write the connection type,
 *  and io link capabilities between the device
 *  @p processor_handle_src and @p processor_handle_dst to the memory
 *  pointed to by @p cap and @p type.
 *
 *  @param[in] processor_handle_src the source processor handle
 *
 *  @param[in] processor_handle_dst the destination processor handle
 *
 *  @param[in,out] type A pointer to an ::amdsmi_io_link_type_t to which the
 *  type for the connection should be written.
 *
 *  @param[in,out] type A pointer to an ::amdsmi_p2p_capability_t to which the
 *  io link capabilities should be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_topo_get_p2p_status(amdsmi_processor_handle processor_handle_src,
                           amdsmi_processor_handle processor_handle_dst,
                           amdsmi_io_link_type_t *type, amdsmi_p2p_capability_t *cap);

/** @} End HWTopo */

/*****************************************************************************/
/** @defgroup compute_partition Compute Partition Functions
 *  These functions are used to configure and query the device's
 *  compute parition setting.
 *  @{
 */

/**
 *  @brief Retrieves the current compute partitioning for a desired device
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details
 *  Given a processor handle @p processor_handle and a string @p compute_partition ,
 *  and uint32 @p len , this function will attempt to obtain the device's
 *  current compute partition setting string. Upon successful retreival,
 *  the obtained device's compute partition settings string shall be stored in
 *  the passed @p compute_partition char string variable.
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[inout] compute_partition a pointer to a char string variable,
 *  which the device's current compute partition will be written to.
 *
 *  @param[in] len the length of the caller provided buffer @p compute_partition
 *  , suggested length is 4 or greater.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS call was successful
 *  @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
 *  @retval ::AMDSMI_STATUS_UNEXPECTED_DATA data provided to function is not valid
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *  @retval ::AMDSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire compute partition value. In this case,
 *  only @p len bytes will be written.
 *
 */
amdsmi_status_t
amdsmi_get_gpu_compute_partition(amdsmi_processor_handle processor_handle,
                                  char *compute_partition, uint32_t len);

/**
 *  @brief Modifies a selected device's compute partition setting.
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a type of compute partition
 *  @p compute_partition, this function will attempt to update the selected
 *  device's compute partition setting.
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[in] compute_partition using enum ::amdsmi_compute_partition_type_t,
 *  define what the selected device's compute partition setting should be
 *  updated to.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS call was successful
 *  @retval ::AMDSMI_STATUS_PERMISSION function requires root access
 *  @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
 *  @retval ::AMDSMI_STATUS_SETTING_UNAVAILABLE the provided setting is
 *  unavailable for current device
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *
 */
amdsmi_status_t
amdsmi_set_gpu_compute_partition(amdsmi_processor_handle processor_handle,
                                  amdsmi_compute_partition_type_t compute_partition);

/**
 *  @brief Reverts a selected device's compute partition setting back to its
 *  boot state.
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, this function will attempt to
 *  revert its compute partition setting back to its boot state.
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS call was successful
 *  @retval ::AMDSMI_STATUS_PERMISSION function requires root access
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *
 */
amdsmi_status_t amdsmi_reset_gpu_compute_partition(amdsmi_processor_handle processor_handle);

/** @} */  // end of compute_partition

/*****************************************************************************/
/** @defgroup memory_partition Memory Partition Functions
 *  These functions are used to query and set the device's current memory
 *  partition.
 *  @{
 */

/**
 *  @brief Retrieves the current memory partition for a desired device
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details
 *  Given a processor handle @p processor_handle and a string @p memory_partition ,
 *  and uint32 @p len , this function will attempt to obtain the device's
 *  memory partition string. Upon successful retreival, the obtained device's
 *  memory partition string shall be stored in the passed @p memory_partition
 *  char string variable.
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[inout] memory_partition a pointer to a char string variable,
 *  which the device's memory partition will be written to.
 *
 *  @param[in] len the length of the caller provided buffer @p memory_partition ,
 *  suggested length is 5 or greater.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS call was successful
 *  @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
 *  @retval ::AMDSMI_STATUS_UNEXPECTED_DATA data provided to function is not valid
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *  @retval ::AMDSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire memory partition value. In this case,
 *  only @p len bytes will be written.
 *
 */
amdsmi_status_t
amdsmi_get_gpu_memory_partition(amdsmi_processor_handle processor_handle,
                                  char *memory_partition, uint32_t len);

/**
 *  @brief Modifies a selected device's current memory partition setting.
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a type of memory partition
 *  @p memory_partition, this function will attempt to update the selected
 *  device's memory partition setting.
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[in] memory_partition using enum ::amdsmi_memory_partition_type_t,
 *  define what the selected device's current mode setting should be updated to.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS call was successful
 *  @retval ::AMDSMI_STATUS_PERMISSION function requires root access
 *  @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *  @retval ::AMDSMI_STATUS_AMDGPU_RESTART_ERR could not successfully restart
 *  the amdgpu driver
 *
 */
amdsmi_status_t
amdsmi_set_gpu_memory_partition(amdsmi_processor_handle processor_handle,
                                  amdsmi_memory_partition_type_t memory_partition);

/**
 *  @brief Reverts a selected device's memory partition setting back to its
 *  boot state.
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, this function will attempt to
 *  revert its current memory partition setting back to its boot state.
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS call was successful
 *  @retval ::AMDSMI_STATUS_PERMISSION function requires root access
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *  @retval ::AMDSMI_STATUS_AMDGPU_RESTART_ERR could not successfully restart
 *  the amdgpu driver
 *
 */
amdsmi_status_t amdsmi_reset_gpu_memory_partition(amdsmi_processor_handle processor_handle);

/** @} */  // end of memory_partition

/*****************************************************************************/
/** @defgroup accelerator_partition_profile Accelerator Partition Profile Functions
 *  These functions are used to configure and query the device's
 *  accelerator parition profile setting.
 *  @{
 */
// TODO: declare rest of partition profile functions and complete doc commentary.
/*
  Get the current accelerator partition profile. The function will return current profile.
*/
amdsmi_status_t
amdsmi_get_gpu_accelerator_partition_profile(amdsmi_processor_handle processor_handle,
                                             amdsmi_accelerator_partition_profile_t *profile,
                                             uint32_t *partition_id);

/** @} */  // end of accelerator_partition_profile

/*****************************************************************************/
/** @defgroup EvntNotif Event Notification Functions
 *  These functions are used to configure for and get asynchronous event
 *  notifications.
 *  @{
 */

/**
 * @brief Prepare to collect event notifications for a GPU
 *
 *  @platform{gpu_bm_linux}
 *
 * @details This function prepares to collect events for the GPU with device
 * ID @p processor_handle, by initializing any required system parameters. This call
 * may open files which will remain open until ::amdsmi_stop_gpu_event_notification()
 * is called.
 *
 * @param processor_handle a processor handle corresponding to the device on which to
 * listen for events
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_init_gpu_event_notification(amdsmi_processor_handle processor_handle);

/**
 * @brief Specify which events to collect for a device
 *
 * @platform{gpu_bm_linux}
 *
 * @details Given a processor handle @p processor_handle and a @p mask consisting of
 * elements of ::amdsmi_evt_notification_type_t OR'd together, this function
 * will listen for the events specified in @p mask on the device
 * corresponding to @p processor_handle.
 *
 * @param processor_handle a processor handle corresponding to the device on which to
 * listen for events
 *
 * @param mask Bitmask generated by OR'ing 1 or more elements of
 * ::amdsmi_evt_notification_type_t indicating which event types to listen for,
 * where the amdsmi_evt_notification_type_t value indicates the bit field, with
 * bit position starting from 1.
 * For example, if the mask field is 0x0000000000000003, which means first bit,
 * bit 1 (bit position start from 1) and bit 2 are set, which indicate interest
 * in receiving AMDSMI_EVT_NOTIF_VMFAULT (which has a value of 1) and
 * AMDSMI_EVT_NOTIF_THERMAL_THROTTLE event (which has a value of 2).
 *
 * @note ::AMDSMI_STATUS_INIT_ERROR is returned if
 * ::amdsmi_init_gpu_event_notification() has not been called before a call to this
 * function
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
 amdsmi_set_gpu_event_notification_mask(amdsmi_processor_handle processor_handle, uint64_t mask);

/**
 * @brief Collect event notifications, waiting a specified amount of time
 *
 * @platform{gpu_bm_linux}
 *
 * @details Given a time period @p timeout_ms in milliseconds and a caller-
 * provided buffer of ::amdsmi_evt_notification_data_t's @p data with a length
 * (in ::amdsmi_evt_notification_data_t's, also specified by the caller) in the
 * memory location pointed to by @p num_elem, this function will collect
 * ::amdsmi_evt_notification_type_t events for up to @p timeout_ms milliseconds,
 * and write up to *@p num_elem event items to @p data. Upon return @p num_elem
 * is updated with the number of events that were actually written. If events
 * are already present when this function is called, it will write the events
 * to the buffer then poll for new events if there is still caller-provided
 * buffer available to write any new events that would be found.
 *
 * This function requires prior calls to ::amdsmi_init_gpu_event_notification() and
 * :: amdsmi_set_gpu_event_notification_mask(). This function polls for the
 * occurrance of the events on the respective devices that were previously
 * specified by :: amdsmi_set_gpu_event_notification_mask().
 *
 * @param[in] timeout_ms number of milliseconds to wait for an event
 * to occur
 *
 * @param[in,out] num_elem pointer to uint32_t, provided by the caller. On
 * input, this value tells how many ::amdsmi_evt_notification_data_t elements
 * are being provided by the caller with @p data. On output, the location
 * pointed to by @p num_elem will contain the number of items written to
 * the provided buffer.
 *
 * @param[out] data pointer to a caller-provided memory buffer of size
 * @p num_elem ::amdsmi_evt_notification_data_t to which this function may safely
 * write. If there are events found, up to @p num_elem event items will be
 * written to @p data.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
 amdsmi_get_gpu_event_notification(int timeout_ms,
                     uint32_t *num_elem, amdsmi_evt_notification_data_t *data);

/**
 * @brief Close any file handles and free any resources used by event
 * notification for a GPU
 *
 * @platform{gpu_bm_linux}
 *
 * @details Any resources used by event notification for the GPU with
 * processor handle @p processor_handle will be free with this
 * function. This includes freeing any memory and closing file handles. This
 * should be called for every call to ::amdsmi_init_gpu_event_notification()
 *
 * @param[in] processor_handle The processor handle of the GPU for which event
 * notification resources will be free
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_stop_gpu_event_notification(amdsmi_processor_handle processor_handle);

/** @} End EvntNotif */

/**
 *  @brief          Returns BDF of the given device
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf}  @platform{guest_mvf}
 *  @platform{guest_windows}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     bdf Reference to BDF. Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_device_bdf(amdsmi_processor_handle processor_handle, amdsmi_bdf_t *bdf);

/**
 *  @brief          Returns the UUID of the device
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf}  @platform{guest_mvf}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[in,out]  uuid_length Length of the uuid string. As input, must be
 *                  equal or greater than SMI_GPU_UUID_SIZE and be allocated by
 *                  user. As output it is the length of the uuid string.
 *
 *  @param[out]     uuid Pointer to string to store the UUID. Must be
 *                  allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_device_uuid(amdsmi_processor_handle processor_handle, unsigned int *uuid_length, char *uuid);

/*****************************************************************************/
/** @defgroup swversion     SW Version Information
 *  @{
 */

/**
 *  @brief          Returns the driver version information
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
 *  @platform{guest_windows}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[in,out]  length As input parameter length of the user allocated
 *                  string buffer. As output parameter length of the returned
 *                  string buffer.
 *
 *  @param[out]     info Reference to driver information structure. Must be
 *                  allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_driver_info(amdsmi_processor_handle processor_handle, amdsmi_driver_info_t *info);

/** @} End swversion */

/*****************************************************************************/
/** @defgroup asicinfo     ASIC & Board Static Information
 *  @{
 */

/**
 *  @brief          Returns the ASIC information for the device
 *
 *  @platform{gpu_bm_linux}  @platform{host} @platform{guest_1vf}  @platform{guest_mvf}
 *  @platform{guest_windows}
 *
 *  @details        This function returns ASIC information such as the product name,
 *                  the vendor ID, the subvendor ID, the device ID,
 *                  the revision ID and the serial number.
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     info Reference to static asic information structure.
 *                  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_asic_info(amdsmi_processor_handle processor_handle, amdsmi_asic_info_t *info);

/**
 *  @brief          Returns the KFD (Kernel Fusion Driver) information for the device
 *
 *  @platform{gpu_bm_linux}  @platform{guest_1vf}  @platform{guest_mvf}
 *
 *  @details        This function returns KFD information populated into the amdsmi_kfd_info_t.
 *                  This contains the kfd_id and node_id which allow for the ID and
 *                  index of this device in the KFD.
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     info Reference to kfd information structure.
 *                  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_kfd_info(amdsmi_processor_handle processor_handle, amdsmi_kfd_info_t *info);

/**
 *  @brief Returns vram info
 *
 *  @platform{gpu_bm_linux}  @platform{host} @platform{guest_1vf}  @platform{guest_mvf}
 *
 *  @param[in] processor_handle PF of a processor for which to query
 *
 *  @param[out] info Reference to vram info structure
 *  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_vram_info(
          amdsmi_processor_handle processor_handle, amdsmi_vram_info_t *info);

/**
 *  @brief          Returns the board part number and board information for the requested device
 *
 *   @platform{gpu_bm_linux}  @platform{host} @platform{guest_1vf}  @platform{guest_mvf}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     info Reference to board info structure.
 *                  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_board_info(amdsmi_processor_handle processor_handle, amdsmi_board_info_t *info);

/**
 *  @brief          Returns the power caps as currently configured in the
 *                  system. Power in units of uW.
 *                  It is not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}  @platform{host}
 *
 *  @param[in]      processor_handle Device which to query
 *  @param[in]      sensor_ind A 0-based sensor index. Normally, this will be 0.
 *                  If a device has more than one sensor, it could be greater than 0.
 *  @param[out]     info Reference to power caps information structure. Must be
 *                  allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_power_cap_info(amdsmi_processor_handle processor_handle, uint32_t sensor_ind,
          amdsmi_power_cap_info_t *info);

/**
 *  @brief Returns the PCIe info for the GPU.
 *
 *  @platform{gpu_bm_linux}  @platform{host} @platform{guest_1vf} @platform{guest_windows}
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[out] info Reference to the PCIe information
 *  returned by the library. Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_pcie_info(amdsmi_processor_handle processor_handle,
          amdsmi_pcie_info_t *info);

/**
 *  @brief          Returns XGMI information for the GPU.
 *
 *  @platform{gpu_bm_linux}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     info Reference to xgmi information structure. Must be
 *                  allocated by user.
 *
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_xgmi_info(amdsmi_processor_handle processor_handle, amdsmi_xgmi_info_t *info);

/** @} End asicinfo */

/*****************************************************************************/
/** @defgroup firmwareinfo     Firmware & VBIOS queries
 *  @{
 */

/**
 *  @brief          Returns the firmware versions running on the device.
 *
 *  @platform{gpu_bm_linux}  @platform{host} @platform{guest_1vf}  @platform{guest_mvf}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     info Reference to the fw info. Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_fw_info(amdsmi_processor_handle processor_handle, amdsmi_fw_info_t *info);

/**
 *  @brief          Returns the static information for the vBIOS on the device.
 *
 *  @platform{gpu_bm_linux}  @platform{host} @platform{guest_1vf}  @platform{guest_mvf}
 *  @platform{guest_windows}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     info Reference to static vBIOS information.
 *                  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_vbios_info(amdsmi_processor_handle processor_handle, amdsmi_vbios_info_t *info);

/** @} End fwinfo */

/*****************************************************************************/
/** @defgroup gpumon     GPU Monitoring
 *  @{
 */

/**
 *  @brief          Returns the current usage of the GPU engines (GFX, MM and MEM).
 *                  Each usage is reported as a percentage from 0-100%. It is not
 *                  supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_windows}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     info Reference to the gpu engine usage structure. Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_activity(amdsmi_processor_handle processor_handle, amdsmi_engine_usage_t *info);

/**
 *  @brief          Returns the current power and voltage of the GPU.
 *                  The voltage is in units of mV and the power in units of W.
 *                  It is not supported on virtual machine guest
 *
 *  @platform{gpu_bm_linux}  @platform{host}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     info Reference to the gpu power structure. Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_power_info(amdsmi_processor_handle processor_handle, amdsmi_power_info_t *info);

/**
 *  @brief Returns is power management enabled
 *
 *  @platform{gpu_bm_linux}  @platform{host}
 *
 *  @param[in] processor_handle PF of a processor for which to query
 *
 *  @param[out] enabled Reference to bool. Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_is_gpu_power_management_enabled(amdsmi_processor_handle processor_handle, bool *enabled);

/**
 *  @brief          Returns the measurements of the clocks in the GPU
 *                  for the GFX and multimedia engines and Memory. This call
 *                  reports the averages over 1s in MHz. It is not supported
 *                  on virtual machine guest
 *
 *  @platform{gpu_bm_linux}  @platform{host}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[in]      clk_type Enum representing the clock type to query.
 *
 *  @param[out]     info Reference to the gpu clock structure.
 *                  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_clock_info(amdsmi_processor_handle processor_handle, amdsmi_clk_type_t clk_type, amdsmi_clk_info_t *info);

/**
 *  @brief          Returns the VRAM usage (both total and used memory)
 *                  in MegaBytes.
 *
 *  @platform{gpu_bm_linux} @platform{guest_1vf}  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *
 *  @param[out]     info Reference to vram information.
 *                  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_vram_usage(amdsmi_processor_handle processor_handle, amdsmi_vram_usage_t *info);

/**
 *  @brief          Returns the violations for a processor
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *
 *  @param[in,out]  info Reference to all violation status details available.
 *                  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_violation_status(amdsmi_processor_handle processor_handle,
                            amdsmi_violation_status_t *info);


/** @} End gpumon */

/*****************************************************************************/
/**  @defgroup processinfo     Process information
 *  @{
 */

/**
 *  @brief          Returns the list of processes running on a given GPU including itself.
 *
 *  @platform{gpu_bm_linux}  @platform{guest_1vf}  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @note           The user provides a buffer to store the list and the maximum
 *                  number of processes that can be returned. If the user sets
 *                  max_processes to 0, the current total number of processes will
 *                  replace max_processes param. After that, the function needs to be
 *                  called again, with updated max_processes, to successfully fill the
 *                  process list, which was previously allocated with max_processes
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[in,out]  max_processes Reference to the size of the list buffer in
 *                  number of elements. Returns the return number of elements
 *                  in list or the number of running processes if equal to 0,
 *                  and if given value in param max_processes is less than
 *                  number of processes currently running,
 *                  AMDSMI_STATUS_OUT_OF_RESOURCES will be returned.
 *
 *                  For cases where max_process is not zero (0), it specifies the list's size limit.
 *                  That is, the maximum size this list will be able to hold. After the list is built
 *                  internally, as a return status, we will have AMDSMI_STATUS_OUT_OF_RESOURCES when
 *                  the original size limit is smaller than the actual list of processes running.
 *                  Hence, the caller is aware the list size needs to be resized, or
 *                  AMDSMI_STATUS_SUCCESS otherwise.
 *                  Holding a copy of max_process before it is passed in will be helpful for monitoring
 *                  the allocations done upon each call since the max_process will permanently be changed
 *                  to reflect the actual number of processes running.
 *
 *  @param[out]     list Reference to a user-provided buffer where the process
 *                  list will be returned. This buffer must contain at least
 *                  max_processes entries of type amd_proc_info_list_t. Must be allocated
 *                  by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success,
 *                            | ::AMDSMI_STATUS_OUT_OF_RESOURCES, filled list buffer with data, but number of
 *                                actual running processes is larger than the size provided.
 *
 */
  //  Note: If the reserved size for processes is smaller than the number of
  //        actual processes running. The AMDSMI_STATUS_OUT_OF_RESOURCES is
  //        an indication the caller should handle the situation (resize).
  //        The max_processes is always changed to reflect the actual size of
  //        list of processes running, so the caller knows where it is at.
  //
amdsmi_status_t
amdsmi_get_gpu_process_list(amdsmi_processor_handle processor_handle, uint32_t *max_processes, amdsmi_proc_info_t *list);

/** @} End processinfo */

/*****************************************************************************/
/** @defgroup eccinfo     ECC information
 *  @{
 */

/**
 *  @brief          Returns the total number of ECC errors (correctable,
 *                  uncorrectable and deferred) in the given GPU. It is not supported on
 *                  virtual machine guest
 *
 *  @platform{gpu_bm_linux}  @platform{host}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     ec Reference to ecc error count structure.
 *                  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_total_ecc_count(amdsmi_processor_handle processor_handle, amdsmi_error_count_t *ec);

/** @} End eccinfo */

/**
 *  @brief          Retrieve the set of GPUs that are nearest to a given device
 *                  at a specific interconnectivity level.
 *
 *  @platform{gpu_bm_linux}  @platform{host}
 *
 *  @details        Once called topology_nearest_info will get populated with a list of
 *                  all nearest devices for a given link_type. The list has a count of
 *                  the number of devices found and their respective handles/identifiers.
 *
 *  @param[in]      processor_handle The identifier of the given device.
 *
 *  @param[in]      link_type The amdsmi_link_type_t level to search for nearest GPUs.
 *
 *  @param[in,out]  topology_nearest_info
 *                  .count;
 *                    - When zero, is set to the number of matching GPUs such that .device_list can
 *                    be malloc'd.
 *                    - When non-zero, .device_list will be filled with count number of processor_handle.
 *
 *  @param[out]     .device_list An array of processor_handle for GPUs found at level.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail.
 */
amdsmi_status_t
amdsmi_get_link_topology_nearest(amdsmi_processor_handle processor_handle,
                                 amdsmi_link_type_t link_type,
                                 amdsmi_topology_nearest_t* topology_nearest_info);


#ifdef ENABLE_ESMI_LIB
/*****************************************************************************/
/**  @defgroup energyinfo     Energy information (RAPL MSR)
 *  @{
 */

/**
 *  @brief Get the core energy for a given core.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu core which to query
 *
 *  @param[in,out]    penergy - Input buffer to return the core energy
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_core_energy(amdsmi_processor_handle processor_handle,
                                           uint64_t *penergy);

/**
 *  @brief Get the socket energy for a given socket.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    penergy - Input buffer to return the socket energy
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_energy(amdsmi_processor_handle processor_handle,
                                             uint64_t *penergy);

/** @} End energyinfo */

/*****************************************************************************/
/**  @defgroup systemstatistics     HSMP system statistics
 *  @{
 */

/**
 *  @brief Get Number of threads Per Core.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in,out]    threads_per_core - Input buffer to return the Number of threads Per Core
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_threads_per_core(uint32_t *threads_per_core);

/**
 *  @brief Get SMU Firmware Version.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in,out]    amdsmi_smu_fw - Input buffer to return the firmware version
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_smu_fw_version(amdsmi_processor_handle processor_handle,
                                              amdsmi_smu_fw_version_t *amdsmi_smu_fw);

/**
 *  @brief Get HSMP protocol Version.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in,out]    proto_ver - Input buffer to return the protocol version
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_hsmp_proto_ver(amdsmi_processor_handle processor_handle,
                                              uint32_t *proto_ver);

/**
 *  @brief Get normalized status of the processor's PROCHOT status.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    prochot - Input buffer to return the procohot status.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_prochot_status(amdsmi_processor_handle processor_handle,
                                              uint32_t *prochot);

/**
 *  @brief Get Data fabric clock and Memory clock in MHz.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    fclk - Input buffer to return fclk
 *
 *  @param[in,out]    mclk - Input buffer to return mclk
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_fclk_mclk(amdsmi_processor_handle processor_handle,
                                         uint32_t *fclk, uint32_t *mclk);

/**
 *  @brief Get core clock in MHz.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    cclk - Input buffer to return core clock
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_cclk_limit(amdsmi_processor_handle processor_handle,
                                          uint32_t *cclk);

/**
 *  @brief Get current active frequency limit of the socket.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    freq - Input buffer to return frequency value in MHz
 *
 *  @param[in,out]    src_type - Input buffer to return frequency source name
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_current_active_freq_limit(amdsmi_processor_handle processor_handle,
                                                                uint16_t *freq, char **src_type);

/**
 *  @brief Get socket frequency range.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    fmax - Input buffer to return maximum frequency
 *
 *  @param[in,out]    fmin - Input buffer to return minimum frequency
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_freq_range(amdsmi_processor_handle processor_handle,
                                                 uint16_t *fmax, uint16_t *fmin);

/**
 *  @brief Get socket frequency limit of the core.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu core which to query
 *
 *  @param[in,out]    freq - Input buffer to return frequency.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_core_current_freq_limit(amdsmi_processor_handle processor_handle,
                                                       uint32_t *freq);

/** @} systemstatistics */

/*****************************************************************************/
/**  @defgroup powercont    Power Control
 *  @{
 */

/**
 *  @brief Get the socket power.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    ppower - Input buffer to return socket power
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_power(amdsmi_processor_handle processor_handle,
                                            uint32_t *ppower);

/**
 *  @brief Get the socket power cap.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    pcap - Input buffer to return power cap.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_power_cap(amdsmi_processor_handle processor_handle,
                                                uint32_t *pcap);

/**
 *  @brief Get the maximum power cap value for a given socket.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    pmax - Input buffer to return maximum power limit value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_power_cap_max(amdsmi_processor_handle processor_handle,
                                                    uint32_t *pmax);

/**
 *  @brief Get the SVI based power telemetry for all rails.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    power - Input buffer to return svi based power value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_pwr_svi_telemetry_all_rails(amdsmi_processor_handle processor_handle,
                                                           uint32_t *power);

/**
 *  @brief Set the power cap value for a given socket.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in]		power - Input power limit value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_socket_power_cap(amdsmi_processor_handle processor_handle,
                                                uint32_t pcap);

/**
 *  @brief Set the power efficiency profile policy.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in]		mode - mode to be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_pwr_efficiency_mode(amdsmi_processor_handle processor_handle,
                                                   uint8_t mode);

/** @} powercont */

/*****************************************************************************/
/**  @defgroup perfcont   Performance (Boost limit) Control
 *  @{
 */

/**
 *  @brief Get the core boost limit.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu core which to query
 *
 *  @param[in,out]	pboostlimit - Input buffer to fill the boostlimit value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_core_boostlimit(amdsmi_processor_handle processor_handle,
                                               uint32_t *pboostlimit);

/**
 *  @brief Get the socket c0 residency.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]	pc0_residency - Input buffer to fill the c0 residency value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_c0_residency(amdsmi_processor_handle processor_handle,
                                                   uint32_t *pc0_residency);

/**
 *  @brief Set the core boostlimit value.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu core which to query
 *
 *  @param[in]		boostlimit - boostlimit value to be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_core_boostlimit(amdsmi_processor_handle processor_handle,
                                               uint32_t boostlimit);

/**
 *  @brief Set the socket boostlimit value.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in]		boostlimit - boostlimit value to be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_socket_boostlimit(amdsmi_processor_handle processor_handle,
                                                 uint32_t boostlimit);

/** @} perfcont */

/*****************************************************************************/
/**  @defgroup ddrquer    DDR bandwidth monitor
 *  @{
 */

/**
 *  @brief Get the DDR bandwidth data.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in,out]	ddr_bw - Input buffer to fill ddr bandwidth data
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_ddr_bw(amdsmi_processor_handle processor_handle,
                                      amdsmi_ddr_bw_metrics_t *ddr_bw);

/** @} ddrquer */

/*****************************************************************************/
/**  @defgroup  tempquer   Temperature Query
 *  @{
 */

/**
 *  @brief Get socket temperature.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]	ptmon - Input buffer to fill temperature value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_temperature(amdsmi_processor_handle processor_handle,
                                                  uint32_t *ptmon);

/** @} tempquer */

/*****************************************************************************/
/**  @defgroup  dimmstatistics   Dimm statistics
 *  @{
 */

/**
 *  @brief Get DIMM temperature range and refresh rate.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in]		dimm_addr - DIMM address
 *  @param[in,out]	rate - Input buffer to fill temperature range and refresh rate value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_dimm_temp_range_and_refresh_rate(amdsmi_processor_handle processor_handle,
                                                                uint8_t dimm_addr,
                                                                amdsmi_temp_range_refresh_rate_t *rate);

/**
 *  @brief Get DIMM power consumption.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in]		dimm_addr - DIMM address
 *  @param[in,out]	rate - Input buffer to fill power consumption value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_dimm_power_consumption(amdsmi_processor_handle processor_handle,
                                                      uint8_t dimm_addr,
                                                      amdsmi_dimm_power_t *dimm_pow);

/**
 *  @brief Get DIMM thermal sensor value.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in]		dimm_addr - DIMM address
 *  @param[in,out]	dimm_temp - Input buffer to fill temperature value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_dimm_thermal_sensor(amdsmi_processor_handle processor_handle,
                                                   uint8_t dimm_addr,
                                                   amdsmi_dimm_thermal_t *dimm_temp);

/** @} dimmstatistics */

/*****************************************************************************/
/**  @defgroup xgmibwcont     xGMI bandwidth control
 *  @{
 */

/**
 *  @brief Set xgmi width.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in]		min - Minimum xgmi width to be set
 *  @param[in]		max - maximum xgmi width to be set
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_xgmi_width(amdsmi_processor_handle processor_handle,
                                          uint8_t min, uint8_t max);

/** @} xgmibwcont */

/*****************************************************************************/
/**  @defgroup gmi3widthcont     GMI3 width control
 *  @{
 */

/**
 *  @brief Set gmi3 link width range.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in]		min_link_width - minimum link width to be set.
 *  @param[in]		max_link_width - maximum link width to be set.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_gmi3_link_width_range(amdsmi_processor_handle processor_handle,
                                                     uint8_t min_link_width, uint8_t max_link_width);

/** @} gmi3widthcont */

/*****************************************************************************/
/**  @defgroup pstatecnt     Pstate selection
 *  @{
 */

/**
 *  @brief Enable APB.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_cpu_apb_enable(amdsmi_processor_handle processor_handle);

/**
 *  @brief Disable APB.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in]		pstate - pstate value to be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_cpu_apb_disable(amdsmi_processor_handle processor_handle,
                                       uint8_t pstate);

/**
 *  @brief Set NBIO lclk dpm level value.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in]		nbio_id - nbio index
 *  @param[in]		min - minimum value to be set
 *  @param[in]		max - maximum value to be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_socket_lclk_dpm_level(amdsmi_processor_handle processor_handle,
                                                     uint8_t nbio_id, uint8_t min, uint8_t max);

/**
 *  @brief Get NBIO LCLK dpm level.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in]		nbio_id - nbio index
 *  @param[in,out]	nbio - Input buffer to fill lclk dpm level
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_lclk_dpm_level(amdsmi_processor_handle processor_handle,
                                                     uint8_t nbio_id, amdsmi_dpm_level_t *nbio);

/**
 *  @brief Set pcie link rate.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in]		rate_ctrl - rate control value to be set.
 *  @param[in,out]	prev_mode - Input buffer to fill previous rate control value.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_pcie_link_rate(amdsmi_processor_handle processor_handle,
                                              uint8_t rate_ctrl, uint8_t *prev_mode);

/**
 *  @brief Set df pstate range.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in]		max_pstate - maximum pstate value to be set
 *  @param[in]		min_pstate - minimum pstate value to be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_df_pstate_range(amdsmi_processor_handle processor_handle,
                                               uint8_t max_pstate, uint8_t min_pstate);

/** @} pstatecnt */

/*****************************************************************************/
/**  @defgroup bwquer     Bandwidth monitor
 *  @{
 */

/**
 *  @brief Get current input output bandwidth.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in]		link - link id and bw type to which io bandwidth to be obtained
 *  @param[in,out]	io_bw - Input buffer to fill bandwidth data
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_current_io_bandwidth(amdsmi_processor_handle processor_handle,
                                                    amdsmi_link_id_bw_type_t link, uint32_t *io_bw);

/**
 *  @brief Get current input output bandwidth.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in]		link - link id and bw type to which xgmi bandwidth to be obtained
 *  @param[in,out]	xgmi_bw - Input buffer to fill bandwidth data
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_current_xgmi_bw(amdsmi_processor_handle processor_handle,
                                               amdsmi_link_id_bw_type_t link, uint32_t *xgmi_bw);

/** @} bwquer */

/*****************************************************************************/
/**  @defgroup MetQuer HSMP Metrics Table
 *  @{
 */

/**
 *  @brief Get HSMP metrics table version
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in,out]  metrics_version input buffer to return the HSMP metrics table version.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_hsmp_metrics_table_version(amdsmi_processor_handle processor_handle,
                                                 uint32_t *metrics_version);

/**
 *  @brief Get HSMP metrics table
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in,out]  metrics_table input buffer to return the HSMP metrics table.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_hsmp_metrics_table(amdsmi_processor_handle processor_handle,
                                         amdsmi_hsmp_metrics_table_t *metrics_table);

/** @} MetQuer */

/*****************************************************************************/
/**  @defgroup auxiquer     Auxillary functions
 *  @{
 */

/**
 *  @brief Get first online core on socket.
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]	pcore_ind - Input buffer to fill first online core on socket data
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_first_online_core_on_cpu_socket(amdsmi_processor_handle processor_handle,
                                                       uint32_t *pcore_ind);

/**
 *  @brief Get CPU family.
 *
 *  @param[in,out]	cpu_family - Input buffer to return the cpu family
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_family(uint32_t *cpu_family);

/**
 *  @brief Get CPU model.
 *
 *  @param[in,out]	cpu_model - Input buffer to return the cpu model
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_model(uint32_t *cpu_model);

/**
 *  @brief Get a description of provided AMDSMI error status for esmi errors.
 *
 *  @platform{cpu_bm}
 *
 *  @details Set the provided pointer to a const char *, @p status_string, to
 *  a string containing a description of the provided error code @p status.
 *
 *  @param[in]    status - The error status for which a description is desired.
 *
 *  @param[in,out]    status_string - A pointer to a const char * which will be made
 *  to point to a description of the provided error code
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_esmi_err_msg(amdsmi_status_t status, const char **status_string);
#endif
/** @} auxiquer */
#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // INCLUDE_AMDSMI_H_
