/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef INCLUDE_RDC_RDC_H_
#define INCLUDE_RDC_RDC_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#ifdef __cplusplus

// cstddef include causes issues on older GCC
// use stddef.h instead
#if __GNUC__ < 9
#include <stddef.h>
#else
#include <cstddef>
#endif  // __GNUC__

#include <cstdint>
#else
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif  // __cplusplus

/** \file rdc.h
 *  Main header file for the ROCm RDC library.
 *  All required function, structure, enum, etc. definitions should be defined
 *  in this file.
 *
 *  @brief The rocm_rdc library api is new, and therefore subject to change
 *  either at the ABI or API level. Instead of marking every function prototype
 *  as "unstable", we are instead saying the API is unstable (i.e., changes
 *  are possible) while the major version remains 0. This means that if the
 *  API/ABI changes, we will not increment the major version to 1. Once the
 *  ABI stabilizes, we will increment the major version to 1, and thereafter
 *  increment it on all ABI breaks.
 */

/**
 * @brief Error codes returned by rocm_rdc_lib functions
 */
typedef enum {
  RDC_ST_OK = 0,            //!< Success
  RDC_ST_NOT_SUPPORTED,     //!< Not supported feature
  RDC_ST_SMI_ERROR,         //!< The SMI library error
  RDC_ST_FAIL_LOAD_MODULE,  //!< Fail to load the library
  RDC_ST_INVALID_HANDLER,   //!< Invalid handler
  RDC_ST_BAD_PARAMETER,     //!< A parameter is invalid
  RDC_ST_NOT_FOUND,         //!< Cannot find the value
  RDC_ST_CONFLICT,          //!< Conflict with current state
  RDC_ST_CLIENT_ERROR,      //!< The RDC client error
  RDC_ST_ALREADY_EXIST,     //!< The item already exists
  RDC_ST_MAX_LIMIT,         //!< Max limit recording for the object
  RDC_ST_INSUFF_RESOURCES,  //!< Not enough resources to complete
                            //!<  operation
  RDC_ST_FILE_ERROR,        //!< Failed to access a file
  RDC_ST_NO_DATA,           //!< Data was requested,
                            //!<   but none was found
  RDC_ST_PERM_ERROR,        //!< Insufficient permission to complete
                            //!<   operation
  RDC_ST_CORRUPTED_EEPROM,  //!< EEPROM is corrupted
  RDC_ST_DISABLED_MODULE,   //!< Attempted loading disabled module

  RDC_ST_UNKNOWN_ERROR = 0xFFFFFFFF  //!< Unknown error
} rdc_status_t;

/**
 * @brief rdc operation mode
 * rdc can run in auto mode where background threads will collect metrics.
 * When run in manual mode, the user needs to periodically call
 * rdc_field_update_all for data collection.
 */
typedef enum { RDC_OPERATION_MODE_AUTO = 0, RDC_OPERATION_MODE_MANUAL } rdc_operation_mode_t;

/**
 * @brief type of GPU group
 */
typedef enum {
  RDC_GROUP_DEFAULT = 0,  //!< All GPUs on the Node
  RDC_GROUP_EMPTY         //!< Empty group
} rdc_group_type_t;

/**
 * @brief the type stored in the filed value
 */
typedef enum { INTEGER = 0, DOUBLE, STRING, BLOB } rdc_field_type_t;

//! ID used to represent an invalid GPU
#define GPU_ID_INVALID (-1)
//! Used to specify all GPUs
#define RDC_GROUP_ALL_GPUS (-1000)
//! Used to specify all stats fields
#define RDC_JOB_STATS_FIELDS (-1000)

/**
 * @brief The max rdc field string length
 */
#define RDC_MAX_STR_LENGTH 256

/**
 * @brief The max entities in a group
 */
#define RDC_GROUP_MAX_ENTITIES 64

/**
 * @brief Max number of GPUs supported by RDC
 */
#define RDC_MAX_NUM_DEVICES 128

/**
 * @brief Max number of partitions
 */
#define RDC_MAX_NUM_PARTITIONS 8

/**
 * @brief The max fields in a field group
 */
#define RDC_MAX_FIELD_IDS_PER_FIELD_GROUP 128

/**
 * @brief The max number of groups
 */
#define RDC_MAX_NUM_GROUPS 64

/**
 * @brief The max number of the field groups
 */
#define RDC_MAX_NUM_FIELD_GROUPS 64

/**
 * @brief The max string length occupied by version information
 */
#define RDC_MAX_VERSION_STR_LENGTH 60

/**
 * @brief Max configuration can be collected using the configuration get
 */
#define RDC_MAX_CONFIG_SETTINGS 32

/**
 * These enums are used to specify a particular field to be retrieved.
 */
typedef enum {
  RDC_FI_INVALID = 0,  //!< Invalid field value
  //!< @brief Identifier fields
  RDC_FI_GPU_COUNT = 1,            //!< GPU count in the system
  RDC_FI_DEV_NAME,                 //!< Name of the device
  RDC_FI_OAM_ID,                   //!< OAM ID of the device
  RDC_FI_DEV_ID,                   //!< Device ID
  RDC_FI_REV_ID,                   //!<
  RDC_FI_TARGET_GRAPHICS_VERSION,  //!< Target graphics version
  RDC_FI_NUM_OF_COMPUTE_UNITS,     //!< Number of compute units
  RDC_FI_UUID,                     //!< Device UUID
  RDC_FI_GPU_PARTITION_COUNT,

  /**
   * @brief Frequency related fields
   */
  RDC_FI_GPU_CLOCK = 100,  //!< The current clock for the GPU
  RDC_FI_MEM_CLOCK,        //!< Clock for the memory

  /*
   * @brief Physical monitor fields
   */
  RDC_FI_MEMORY_TEMP = 200,  //!< Memory temperature for the device
  RDC_FI_GPU_TEMP,           //!< Current temperature for the device
  RDC_FI_POWER_USAGE = 300,  //!< Power usage for the device

  /**
   * @brief PCIe related fields
   */
  RDC_FI_PCIE_TX = 400,  //!< PCIe Tx utilization information
  RDC_FI_PCIE_RX,        //!< PCIe Rx utilization information
  // RDC_FI_PCIE_TX, RDC_FI_PCIE_RX are not supported on new ASIC
  // The RDC_FI_PCIE_BANDWIDTH should be used
  RDC_FI_PCIE_BANDWIDTH,  //!< PCIe bandwidth in Mbps

  /**
   * @brief GPU usage related fields
   */
  RDC_FI_GPU_UTIL = 500,            //!< GPU Utilization
  RDC_FI_GPU_MEMORY_USAGE,          //!< Memory usage of the GPU instance
  RDC_FI_GPU_MEMORY_TOTAL,          //!< Total memory of the GPU instance
  RDC_FI_GPU_MM_ENC_UTIL,           //!< Multimedia encoder busy percentage
  RDC_FI_GPU_MM_DEC_UTIL,           //!< Multimedia decoder busy percentage
  RDC_FI_GPU_MEMORY_ACTIVITY,       //!< Memory busy percentage
  RDC_FI_GPU_MEMORY_MAX_BANDWIDTH,  //<! The Memory max bandwidth at current memory clock in
                                    // Mb/Second
  RDC_FI_GPU_MEMORY_CUR_BANDWIDTH,  //<! The Memory current bandwidth in Mb/Second
  RDC_FI_GPU_BUSY_PERCENT,          //<! The GPU busy percentage

  /**
   * @brief GPU page related fields
   */
  RDC_FI_GPU_PAGE_RETRIED = 550,  //!< Retried page of the GPU instance
  /**
   * @brief ECC related fields
   */
  RDC_FI_ECC_CORRECT_TOTAL = 600,  //!< Accumulated correctable ECC errors
  RDC_FI_ECC_UNCORRECT_TOTAL,      //!< Accumulated uncorrectable ECC errors

  RDC_FI_ECC_FIRST = 602,  //!< FIRST Error Correction and Detection field
  RDC_FI_ECC_SDMA_CE = RDC_FI_ECC_FIRST,
  RDC_FI_ECC_SDMA_UE,
  RDC_FI_ECC_GFX_CE,
  RDC_FI_ECC_GFX_UE,
  RDC_FI_ECC_MMHUB_CE,
  RDC_FI_ECC_MMHUB_UE,
  RDC_FI_ECC_ATHUB_CE,
  RDC_FI_ECC_ATHUB_UE,
  RDC_FI_ECC_PCIE_BIF_CE,
  RDC_FI_ECC_PCIE_BIF_UE,
  RDC_FI_ECC_HDP_CE,
  RDC_FI_ECC_HDP_UE,
  RDC_FI_ECC_XGMI_WAFL_CE,
  RDC_FI_ECC_XGMI_WAFL_UE,
  RDC_FI_ECC_DF_CE,
  RDC_FI_ECC_DF_UE,
  RDC_FI_ECC_SMN_CE,
  RDC_FI_ECC_SMN_UE,
  RDC_FI_ECC_SEM_CE,
  RDC_FI_ECC_SEM_UE,
  RDC_FI_ECC_MP0_CE,
  RDC_FI_ECC_MP0_UE,
  RDC_FI_ECC_MP1_CE,
  RDC_FI_ECC_MP1_UE,
  RDC_FI_ECC_FUSE_CE,
  RDC_FI_ECC_FUSE_UE,
  RDC_FI_ECC_UMC_CE,
  RDC_FI_ECC_UMC_UE,
  RDC_FI_ECC_MCA_CE,
  RDC_FI_ECC_MCA_UE,
  RDC_FI_ECC_VCN_CE,
  RDC_FI_ECC_VCN_UE,
  RDC_FI_ECC_JPEG_CE,
  RDC_FI_ECC_JPEG_UE,
  RDC_FI_ECC_IH_CE,
  RDC_FI_ECC_IH_UE,
  RDC_FI_ECC_MPIO_CE,
  RDC_FI_ECC_MPIO_UE,
  RDC_FI_ECC_LAST = RDC_FI_ECC_MPIO_UE,

  // In new ASCI, such as MI300, the XGMI events is not supported
  // Using below XGMI related fields to calculate the bandwidth.
  RDC_FI_XGMI_0_READ_KB = 700,  //!< XGMI_0 accumulated data read size (KB)
  RDC_FI_XGMI_1_READ_KB,        //!< XGMI_1 accumulated data read size (KB)
  RDC_FI_XGMI_2_READ_KB,        //!< XGMI_2 accumulated data read size (KB)
  RDC_FI_XGMI_3_READ_KB,        //!< XGMI_3 accumulated data read size (KB)
  RDC_FI_XGMI_4_READ_KB,        //!< XGMI_4 accumulated data read size (KB)
  RDC_FI_XGMI_5_READ_KB,        //!< XGMI_5 accumulated data read size (KB)
  RDC_FI_XGMI_6_READ_KB,        //!< XGMI_6 accumulated data read size (KB)
  RDC_FI_XGMI_7_READ_KB,        //!< XGMI_7 accumulated data read size (KB)

  RDC_FI_XGMI_0_WRITE_KB,      //!< XGMI_0 accumulated data write size (KB)
  RDC_FI_XGMI_1_WRITE_KB,      //!< XGMI_1 accumulated data write size (KB)
  RDC_FI_XGMI_2_WRITE_KB,      //!< XGMI_2 accumulated data write size (KB)
  RDC_FI_XGMI_3_WRITE_KB,      //!< XGMI_3 accumulated data write size (KB)
  RDC_FI_XGMI_4_WRITE_KB,      //!< XGMI_4 accumulated data write size (KB)
  RDC_FI_XGMI_5_WRITE_KB,      //!< XGMI_5 accumulated data write size (KB)
  RDC_FI_XGMI_6_WRITE_KB,      //!< XGMI_6 accumulated data write size (KB)
  RDC_FI_XGMI_7_WRITE_KB,      //!< XGMI_7 accumulated data write size (KB)
  RDC_FI_XGMI_TOTAL_READ_KB,   //!< XGMI_SUM accumulated data read size (KB)
  RDC_FI_XGMI_TOTAL_WRITE_KB,  //!< XGMI_SUM accumulated data write size (KB)

  /**
   * @brief ROC-profiler related fields
   */
  RDC_FI_PROF_OCCUPANCY_PERCENT = 800,
  RDC_FI_PROF_ACTIVE_CYCLES,
  RDC_FI_PROF_ACTIVE_WAVES,
  RDC_FI_PROF_ELAPSED_CYCLES,
  RDC_FI_PROF_TENSOR_ACTIVE_PERCENT,
  RDC_FI_PROF_GPU_UTIL_PERCENT,
  // metrics with EVAL are divided by time passed
  RDC_FI_PROF_EVAL_MEM_R_BW,
  RDC_FI_PROF_EVAL_MEM_W_BW,
  RDC_FI_PROF_EVAL_FLOPS_16,
  RDC_FI_PROF_EVAL_FLOPS_32,
  RDC_FI_PROF_EVAL_FLOPS_64,
  RDC_FI_PROF_VALU_PIPE_ISSUE_UTIL,
  RDC_FI_PROF_SM_ACTIVE,
  RDC_FI_PROF_OCC_PER_ACTIVE_CU,
  RDC_FI_PROF_OCC_ELAPSED,
  RDC_FI_PROF_EVAL_FLOPS_16_PERCENT,
  RDC_FI_PROF_EVAL_FLOPS_32_PERCENT,
  RDC_FI_PROF_EVAL_FLOPS_64_PERCENT,
  // CPC
  RDC_FI_PROF_CPC_CPC_STAT_BUSY,
  RDC_FI_PROF_CPC_CPC_STAT_IDLE,
  RDC_FI_PROF_CPC_CPC_STAT_STALL,
  RDC_FI_PROF_CPC_CPC_TCIU_BUSY,
  RDC_FI_PROF_CPC_CPC_TCIU_IDLE,
  RDC_FI_PROF_CPC_CPC_UTCL2IU_BUSY,
  RDC_FI_PROF_CPC_CPC_UTCL2IU_IDLE,
  RDC_FI_PROF_CPC_CPC_UTCL2IU_STALL,
  RDC_FI_PROF_CPC_ME1_BUSY_FOR_PACKET_DECODE,
  RDC_FI_PROF_CPC_ME1_DC0_SPI_BUSY,
  RDC_FI_PROF_CPC_UTCL1_STALL_ON_TRANSLATION,
  RDC_FI_PROF_CPC_ALWAYS_COUNT,
  RDC_FI_PROF_CPC_ADC_VALID_CHUNK_NOT_AVAIL,
  RDC_FI_PROF_CPC_ADC_DISPATCH_ALLOC_DONE,
  RDC_FI_PROF_CPC_ADC_VALID_CHUNK_END,
  RDC_FI_PROF_CPC_SYNC_FIFO_FULL_LEVEL,
  RDC_FI_PROF_CPC_SYNC_FIFO_FULL,
  RDC_FI_PROF_CPC_GD_BUSY,
  RDC_FI_PROF_CPC_TG_SEND,
  RDC_FI_PROF_CPC_WALK_NEXT_CHUNK,
  RDC_FI_PROF_CPC_STALLED_BY_SE0_SPI,
  RDC_FI_PROF_CPC_STALLED_BY_SE1_SPI,
  RDC_FI_PROF_CPC_STALLED_BY_SE2_SPI,
  RDC_FI_PROF_CPC_STALLED_BY_SE3_SPI,
  RDC_FI_PROF_CPC_LTE_ALL,
  RDC_FI_PROF_CPC_SYNC_WRREQ_FIFO_BUSY,
  RDC_FI_PROF_CPC_CANE_BUSY,
  RDC_FI_PROF_CPC_CANE_STALL,
  // CPF
  RDC_FI_PROF_CPF_CMP_UTCL1_STALL_ON_TRANSLATION,
  RDC_FI_PROF_CPF_CPF_STAT_BUSY,
  RDC_FI_PROF_CPF_CPF_STAT_IDLE,
  RDC_FI_PROF_CPF_CPF_STAT_STALL,
  RDC_FI_PROF_CPF_CPF_TCIU_BUSY,
  RDC_FI_PROF_CPF_CPF_TCIU_IDLE,
  RDC_FI_PROF_CPF_CPF_TCIU_STALL,
  RDC_FI_PROF_SIMD_UTILIZATION,
  RDC_FI_PROF_KFD_ID,

  /**
   * @brief Raw XGMI counter events
   */
  RDC_EVNT_XGMI_0_NOP_TX = 1000,  //!< NOPs sent to neighbor 0
  RDC_EVNT_XGMI_0_REQ_TX,         //!< Outgoing requests to
                                  //!< neighbor 0
  RDC_EVNT_XGMI_0_RESP_TX,        //!< Outgoing responses to
                                  //!< neighbor 0
  /**
   * @brief
   *
   * Data beats sent to neighbor 0; Each beat represents 32 bytes.<br><br>
   *
   * XGMI throughput can be calculated by multiplying a BEATs event
   * such as ::RDC_EVNT_XGMI_0_BEATS_TX by 32 and dividing by
   * the time for which event collection occurred,
   * ::rdc_gpu_usage_info_t.start_time (which is in nanoseconds). To get
   * bytes per second, multiply this value by 10<sup>9</sup>.<br>
   * <br>
   * Throughput = BEATS/time_running * 10<sup>9</sup>  (bytes/second)<br>
   */
  //  ie, Throughput = BEATS/time_running 10^9  bytes/sec
  RDC_EVNT_XGMI_0_BEATS_TX,
  RDC_EVNT_XGMI_1_NOP_TX,    //!< NOPs sent to neighbor 1
  RDC_EVNT_XGMI_1_REQ_TX,    //!< Outgoing requests to
                             //!< neighbor 1
  RDC_EVNT_XGMI_1_RESP_TX,   //!< Outgoing responses to
                             //!< neighbor 1
  RDC_EVNT_XGMI_1_BEATS_TX,  //!< Data beats sent to
                             //!< neighbor 1; Each beat
                             //!< represents 32 bytes

  // "Composite" events. These events have additional processing beyond
  // the value provided by the amd_smi library.
  RDC_EVNT_XGMI_0_THRPUT = 1500,  //!< Transmit throughput to XGMI
                                  //!< neighbor 0 in byes/sec
  RDC_EVNT_XGMI_1_THRPUT,         //!< Transmit throughput to XGMI
                                  //!< neighbor 1 in byes/sec
  RDC_EVNT_XGMI_2_THRPUT,         //!< Transmit throughput to XGMI
                                  //!< neighbor 2 in byes/sec
  RDC_EVNT_XGMI_3_THRPUT,         //!< Transmit throughput to XGMI
                                  //!< neighbor 3 in byes/sec
  RDC_EVNT_XGMI_4_THRPUT,         //!< Transmit throughput to XGMI
                                  //!< neighbor 4 in byes/sec
  RDC_EVNT_XGMI_5_THRPUT,         //!< Transmit throughput to XGMI
                                  //!< neighbor 5 in byes/sec

  RDC_EVNT_NOTIF_VMFAULT = 2000,  //!< VM page fault
  RDC_EVNT_NOTIF_FIRST = RDC_EVNT_NOTIF_VMFAULT,

  RDC_EVNT_NOTIF_THERMAL_THROTTLE,  //!< Clock frequency has decreased
                                    //!< due to temperature rise
  RDC_EVNT_NOTIF_PRE_RESET,         //!< GPU reset is about to occur
  RDC_EVNT_NOTIF_POST_RESET,        //!< GPU reset just occurred
  RDC_EVNT_NOTIF_MIGRATE_START,
  RDC_EVNT_NOTIF_MIGRATE_END,
  RDC_EVNT_NOTIF_PAGE_FAULT_START,
  RDC_EVNT_NOTIF_PAGE_FAULT_END,
  RDC_EVNT_NOTIF_QUEUE_EVICTION,
  RDC_EVNT_NOTIF_QUEUE_RESTORE,
  RDC_EVNT_NOTIF_UNMAP_FROM_GPU,
  RDC_EVNT_NOTIF_PROCESS_START,
  RDC_EVNT_NOTIF_PROCESS_END,

  RDC_EVNT_NOTIF_LAST = RDC_EVNT_NOTIF_PROCESS_END,

  /**
   * @brief RDC health related fields
   */
  RDC_HEALTH_XGMI_ERROR = 3000,      //!< XGMI one or more errors detected
  RDC_HEALTH_PCIE_REPLAY_COUNT,      //!< Total PCIE replay count
  RDC_HEALTH_RETIRED_PAGE_NUM,       //!< Retired page number
  RDC_HEALTH_PENDING_PAGE_NUM,       //!< Pending page number
  RDC_HEALTH_RETIRED_PAGE_LIMIT,     //!< the threshold of retired page number
  RDC_HEALTH_EEPROM_CONFIG_VALID,    //!< Reads the EEPROM and verifies the checksums
  RDC_HEALTH_POWER_THROTTLE_TIME,    //!< Power throttle status counter
  RDC_HEALTH_THERMAL_THROTTLE_TIME,  //!< Total time in thermal throttle status (microseconds)
  /**
   * @brief RDC CPU related fields
   */
  RDC_FI_CPU_COUNT = 10000,       //!< CPU count
  RDC_FI_CPU_UTIL_TOTAL,          //!< CPU total percentage of time in use
  RDC_FI_CPU_UTIL_USER,           //!< Percent of time in use by the user
  RDC_FI_CPU_UTIL_NICE,           //!< Percent of time in use by low priority programs
  RDC_FI_CPU_UTIL_SYS,            //!< Percent of time in use by the system
  RDC_FI_CPU_UTIL_IRQ,            //!< Percent of time in use by interrupts
  RDC_FI_CPU_TEMP_CURRENT,        //!< Temperature (Celsius)
  RDC_FI_CPU_CLOCK_CURRENT,       //!< Clock speed (KHz)
  RDC_FI_CPU_POWER_UTIL_CURRENT,  //!< Power usage (watts)
  RDC_FI_CPU_POWER_LIMIT,         //!< Power limit (watts)
  RDC_FI_CPU_VENDOR,              //!< Name of the vendor
  RDC_FI_CPU_MODEL,               //!< Name of the model

  /** TODO: Clean up duplicates
   * @brief Below are AI generated from functions
   */
  RDC_FI_CPU_CORE_ENERGY,               //!< CPU core energy consumption (microjoules)
  RDC_FI_CPU_SOCKET_ENERGY,             //!< CPU socket energy consumption (microjoules)
  RDC_FI_CPU_THREADS_PER_CORE,          //!< Number of threads per CPU core
  RDC_FI_CPU_HSMP_DRIVER_VERSION,       //!< HSMP driver version
  RDC_FI_CPU_SMU_FW_VERSION,            //!< SMU firmware version
  RDC_FI_CPU_HSMP_PROTO_VERSION,        //!< HSMP protocol version
  RDC_FI_CPU_PROCHOT_STATUS,            //!< CPU PROCHOT status
  RDC_FI_CPU_FCLK_FREQUENCY,            //!< CPU fabric clock frequency (MHz)
  RDC_FI_CPU_MCLK_FREQUENCY,            //!< CPU memory clock frequency (MHz)
  RDC_FI_CPU_CCLK_LIMIT,                //!< CPU core clock limit (MHz)
  RDC_FI_CPU_SOCKET_ACTIVE_FREQ_LIMIT,  //!< CPU socket active frequency limit (MHz)
  RDC_FI_CPU_SOCKET_FREQ_LIMIT_SRC,     //!< CPU socket frequency limit source type
  RDC_FI_CPU_SOCKET_FREQ_RANGE_MAX,     //!< CPU socket maximum frequency range (MHz)
  RDC_FI_CPU_SOCKET_FREQ_RANGE_MIN,     //!< CPU socket minimum frequency range (MHz)
  RDC_FI_CPU_CORE_FREQ_LIMIT,           //!< CPU core current frequency limit (MHz)
  RDC_FI_CPU_CORE_BOOST_LIMIT,          //!< CPU core boost limit (MHz)
  RDC_FI_CPU_SOCKET_C0_RESIDENCY,       //!< CPU socket C0 residency percentage
  RDC_FI_CPU_DDR_BW_MAX_BW,             //!< CPU DDR maximum bandwidth (MB/s)
  RDC_FI_CPU_DDR_BW_UTILIZED_BW,        //!< CPU DDR utilized bandwidth (MB/s)
  RDC_FI_CPU_DDR_BW_UTILIZED_PCT,       //!< CPU DDR utilized bandwidth percentage
  RDC_FI_CPU_SOCKET_TEMPERATURE,        //!< CPU socket temperature (millidegrees Celsius)
  RDC_FI_CPU_DIMM_TEMP_RANGE,           //!< CPU DIMM temperature range
  RDC_FI_CPU_DIMM_REFRESH_RATE,         //!< CPU DIMM refresh rate
  RDC_FI_CPU_DIMM_POWER_CONSUMPTION,    //!< CPU DIMM power consumption (milliwatts)
  RDC_FI_CPU_DIMM_THERMAL_SENSOR,       //!< CPU DIMM thermal sensor temperature (millidegrees)
  RDC_FI_CPU_SOCKET_LCLK_DPM_LEVEL,     //!< CPU socket LCLK DPM level
  RDC_FI_CPU_IO_BANDWIDTH,              //!< CPU I/O bandwidth (MB/s)
  RDC_FI_CPU_XGMI_BANDWIDTH,            //!< CPU XGMI bandwidth (MB/s)
  RDC_FI_CPU_HSMP_METRICS_VERSION,      //!< HSMP metrics table version
  RDC_FI_CPU_HSMP_METRICS_TABLE,        //!< HSMP metrics table data
  RDC_FI_CPU_FIRST_ONLINE_CORE,         //!< First online core on CPU socket
  RDC_FI_CPU_FAMILY,                    //!< CPU family identifier
  RDC_FI_CPU_MODEL_ID,                  //!< CPU model identifier
  RDC_FI_CPU_CORES_PER_SOCKET,          //!< Number of CPU cores per socket
  RDC_FI_CPU_SOCKET_COUNT,              //!< Number of CPU sockets

} rdc_field_t;

// even and odd numbers are used for correctable and uncorrectable errors
static_assert(RDC_FI_ECC_SDMA_CE % 2 == 0, "Correctable Error enum is not even");
static_assert(RDC_FI_ECC_SDMA_UE % 2 == 1, "Uncorrectable Error enum is not odd");
static_assert(RDC_FI_ECC_MPIO_CE % 2 == 0, "Correctable Error enum is not even");
static_assert(RDC_FI_ECC_MPIO_UE % 2 == 1, "Uncorrectable Error enum is not odd");

#define RDC_EVNT_IS_NOTIF_FIELD(FIELD) \
  ((FIELD) >= RDC_EVNT_NOTIF_FIRST && (FIELD) <= RDC_EVNT_NOTIF_LAST)
/**
 * @brief handlers used in various rdc calls
 */
typedef void* rdc_handle_t;        //!< Handle used for an RDC session
typedef uint32_t rdc_gpu_group_t;  //!< GPU Group ID type
typedef uint32_t rdc_field_grp_t;  //!< Field group ID type

/**
 * @brief Represents attributes corresponding to a device
 */
typedef struct {
  char device_name[RDC_MAX_STR_LENGTH];  //!< Name of the device.
  uint64_t device_id;                    //!< The device id of a GPU
  uint32_t num_of_compute_units;
  uint64_t target_graphics_version;
} rdc_device_attributes_t;

/**
 * @brief Store version information for each component
 */
typedef struct {
  char version[RDC_MAX_VERSION_STR_LENGTH];
} rdc_component_version_t;

/**
 * @brief The structure to store the group info
 */
typedef struct {
  unsigned int count;                   //!< count of GPUs in the group
  char group_name[RDC_MAX_STR_LENGTH];  //!< group name
  /**
   * The list of entities in the group
   */
  uint32_t entity_ids[RDC_GROUP_MAX_ENTITIES];
} rdc_group_info_t;

/**
 * @brief The structure to store summary of data
 */
typedef struct {
  uint64_t max_value;         //!< Maximum value measured
  uint64_t min_value;         //!< Minimum value measured
  uint64_t average;           //!< Average value measured
  double standard_deviation;  //!< The standard deviation
} rdc_stats_summary_t;

/**
 * @brief The structure to hold the GPU usage information
 */
typedef struct {
  uint32_t gpu_id;      //!< GPU_ID_INVALID for summary information
  uint64_t start_time;  //!< The time to start the watching
  uint64_t end_time;    //!< The time to stop the watching

  uint64_t energy_consumed;             //!< GPU Energy consumed
  uint64_t ecc_correct;                 //!< Correctable errors
  uint64_t ecc_uncorrect;               //!< Uncorrectable errors
  rdc_stats_summary_t pcie_tx;          //!< Bytes sent over PCIe stats
  rdc_stats_summary_t pcie_rx;          //!< Bytes received over PCIe stats
  rdc_stats_summary_t pcie_total;       //!< Total PCIe bandwidth stats
                                        //!< pcie_tx/pcie_rx are not available on mi300, max integer
                                        //!< returned, so use pcie_total
  rdc_stats_summary_t power_usage;      //!< GPU Power usage stats
  rdc_stats_summary_t gpu_clock;        //!< GPU Clock speed stats
  rdc_stats_summary_t memory_clock;     //!< Mem. Clock speed stats
  rdc_stats_summary_t gpu_utilization;  //!< GPU Utilization stats
  rdc_stats_summary_t gpu_temperature;  //!< GPU temperature stats

  uint64_t max_gpu_memory_used;            //!< Maximum GPU memory used
  rdc_stats_summary_t memory_utilization;  //!< Memory Utilization statistics
} rdc_gpu_usage_info_t;                    //!< GPU usage statistics

#define MAX_PROCESS_NAME 256
/**
 * @brief The structure to track process start/stop times during a job running
 */
typedef struct {
  uint32_t pid;  //!< Process ID
  char process_name[MAX_PROCESS_NAME];
  uint64_t start_time;  //!< Process start time in microseconds since 1970
  uint64_t stop_time;   //!< Process stop time in microseconds since 1970
} rdc_process_status_info_t;
#define RDC_MAX_NUM_PROCESSES_STATUS 64

/**
 * @brief The structure to hold the job stats
 */
typedef struct {
  uint32_t num_gpus;              //!< Number of GPUs used by job
  rdc_gpu_usage_info_t summary;   //!< Job usage summary statistics
                                  //!< (overall)
  rdc_gpu_usage_info_t gpus[16];  //!< Job usage summary statistics by GPU
  uint32_t num_processes;         //!< Number of processes tracked
  rdc_process_status_info_t
      processes[RDC_MAX_NUM_PROCESSES_STATUS];  //!< Array to track process start/stop times
} rdc_job_info_t;

/**
 * @brief Field value data
 */
typedef union {
  int64_t l_int;
  double dbl;
  char str[RDC_MAX_STR_LENGTH];
} rdc_field_value_data;

/**
 * @brief The structure to store the field value
 */
typedef struct {
  rdc_field_t field_id;        //!< The field id of the value
  int status;                  //!< RDC_ST_OK or error status
  uint64_t ts;                 //!< Timestamp in usec since 1970
  rdc_field_type_t type;       //!< The field type
  rdc_field_value_data value;  //!< Value of the field. Value type
                               //!< depends on the field type.
} rdc_field_value;

/**
 * @brief The structure to store the field group info
 */
typedef struct {
  uint32_t count;                       //!< count of fields in the group
  char group_name[RDC_MAX_STR_LENGTH];  //!< field group name
  /**
   * The list of fields in the group
   */
  rdc_field_t field_ids[RDC_MAX_FIELD_IDS_PER_FIELD_GROUP];
} rdc_field_group_info_t;

/**
 * @brief The structure to store the job info
 */
typedef struct {
  char job_id[RDC_MAX_STR_LENGTH];  //!< job id
  rdc_gpu_group_t group_id;         //!< group name
  uint64_t start_time;              //!< job start time
  uint64_t stop_time;               //!< job stop time
} rdc_job_group_info_t;

/**
 * @brief type of diagnostic level
 */
typedef enum {
  RDC_DIAG_LVL_INVALID = 0,  //!< invalid level
  RDC_DIAG_LVL_SHORT,        //!< take a few seconds to run
  RDC_DIAG_LVL_MED,          //!< take less than 2 minutes to run
  RDC_DIAG_LVL_LONG          //!< take up to 15 minutes to run
} rdc_diag_level_t;

/**
 * @brief type of diagnostic result
 */
typedef enum {
  RDC_DIAG_RESULT_PASS,  //!< The diagnostic test pass
  RDC_DIAG_RESULT_SKIP,  //!< The diagnostic test skipped
  RDC_DIAG_RESULT_WARN,  //!< The diagnostic test has warnings
  RDC_DIAG_RESULT_FAIL   //!< The diagnostic test fail
} rdc_diag_result_t;

/**
 * @brief The test cases to run
 */
typedef enum {
  RDC_DIAG_TEST_FIRST = 0,
  //!< The diagnostic test pass
  RDC_DIAG_COMPUTE_PROCESS = RDC_DIAG_TEST_FIRST,
  RDC_DIAG_COMPUTE_QUEUE,         //!< The Compute Queue is ready
  RDC_DIAG_SYS_MEM_CHECK,         //!< Check System memory
  RDC_DIAG_NODE_TOPOLOGY,         //!< Report node topology
  RDC_DIAG_GPU_PARAMETERS,        //!< GPU parameters in range
  RDC_DIAG_RVS_GST_TEST,          //!< RVS GST test
  RDC_DIAG_RVS_MEMBW_TEST,        //!< RVS bandwidth test
  RDC_DIAG_RVS_H2DD2H_TEST,       //!< RVS Host<->Device transfer speed test
  RDC_DIAG_RVS_IET_TEST,          //!< RVS IET test
  RDC_DIAG_RVS_GST_LONG_TEST,     //!< RVS GST test
  RDC_DIAG_RVS_MEMBW_LONG_TEST,   //!< RVS bandwidth test
  RDC_DIAG_RVS_H2DD2H_LONG_TEST,  //!< RVS Host<->Device transfer speed test
  RDC_DIAG_RVS_IET_LONG_TEST,     //!< RVS IET test
  RDC_DIAG_RVS_CUSTOM,            //!< RVS custom test
  RDC_DIAG_TEST_LAST,
} rdc_diag_test_cases_t;

/**
 * @brief Type of Components
 */
typedef enum {
  RDC_AMDSMI_COMPONENT
  // If needed later, add them one by one
} rdc_component_t;

/**
 * @brief The maximum test cases to run
 */
#define MAX_TEST_CASES (RDC_DIAG_TEST_LAST - RDC_DIAG_TEST_FIRST)

/**
 * @brief The maximum length of the diagnostic messages
 */
#define MAX_DIAG_MSG_LENGTH 4096

/**
 * @brief details of the diagnostic errors
 */
typedef struct {
  char msg[MAX_DIAG_MSG_LENGTH];  //!< The test result details
  uint32_t code;                  //!< The low level error code
} rdc_diag_detail_t;

/**
 * @brief details of the per gpu diagnostic results
 */
typedef struct {
  uint32_t gpu_index;            //!< The GPU index
  rdc_diag_detail_t gpu_result;  //!< The detail results
} rdc_diag_per_gpu_result_t;

/**
 * @brief The diagnostic results for all GPUs
 */
typedef struct {
  rdc_diag_result_t status;         //!< The diagnostic result
  rdc_diag_detail_t details;        //!< The summary details
  rdc_diag_test_cases_t test_case;  //!< The test case to run

  uint32_t per_gpu_result_count;  //!< How many gpu_results
  //!< Result details
  rdc_diag_per_gpu_result_t gpu_results[RDC_MAX_NUM_DEVICES];

  char info[MAX_DIAG_MSG_LENGTH];  //!< Detail information
} rdc_diag_test_result_t;

/**
 * @brief The diagnostic responses for test cases
 */
typedef struct {
  uint32_t results_count;
  rdc_diag_test_result_t diag_info[MAX_TEST_CASES];
} rdc_diag_response_t;

typedef void (*rdc_callback_t)(void*, void*);
typedef struct {
  rdc_callback_t callback;  //!< Callback sends logs for running diagnostics
  void* cookie;  //!< Cookie is used to identify different callbacks and supply them with data
} rdc_diag_callback_t;

/**
 * @brief The policy type to support
 */
typedef enum {
  RDC_POLICY_COND_FIRST = 0,
  RDC_POLICY_COND_MAX_PAGE_RETRIED = RDC_POLICY_COND_FIRST,  //!< Max number of page retired
  RDC_POLICY_COND_THERMAL,  //!< Temperature threshold, millidegree Celsius
  RDC_POLICY_COND_POWER,    //!< Power threshold, unit microwatt
  RDC_POLICY_COND_LAST = RDC_POLICY_COND_POWER,
  RDC_POLICY_COND_MAX
} rdc_policy_condition_type_t;

typedef struct {
  rdc_policy_condition_type_t type;
  int64_t value;
} rdc_policy_condition_t;

typedef enum { RDC_POLICY_ACTION_NONE, RDC_POLICY_ACTION_GPU_RESET } rdc_policy_action_t;

/**
 * @brief The structure to define policy to enforce on GPU.
 */
typedef struct {
  rdc_policy_condition_t condition;  //!< condition to meet
  rdc_policy_action_t action;        //!< Action to take
} rdc_policy_t;

typedef enum {
  RDC_IOLINK_TYPE_UNDEFINED = 0,     //!< unknown type.
  RDC_IOLINK_TYPE_PCIEXPRESS = 1,    //!< PCI Express
  RDC_IOLINK_TYPE_XGMI = 2,          //!< XGMI
  RDC_IOLINK_TYPE_SIZE = 0xFFFFFFFF  //!< Max of IO Link types
} rdc_topology_link_type_t;

/**
 * @brief The link information of the GPU connected to
 */
typedef struct {
  uint32_t gpu_index;
  // amdsmi_topo_get_link_weight
  uint64_t weight;  // the weight for a connection between 2 GPUs
  // minimal and maximal io link bandwidth between 2 GPUs
  // amdsmi_get_minmax_bandwidth_between_processors
  uint64_t min_bandwidth;
  uint64_t max_bandwidth;
  // amdsmi_topo_get_link_type
  uint64_t hops;
  rdc_topology_link_type_t link_type;
  // amdsmi_is_P2P_accessible
  bool is_p2p_accessible;
} rdc_topology_link_info_t;

/**
 * @brief The data in the data structure will be set to max value if it is N/A or error
 */
typedef struct {
  uint32_t num_of_gpus;  // The length of link_infos array
  rdc_topology_link_info_t link_infos[RDC_MAX_NUM_DEVICES];
  // amdsmi_topo_get_numa_node_number
  uint32_t numa_node;  // the NUMA CPU node number for a device
} rdc_device_topology_t;

typedef enum {
  RDC_LINK_STATE_DOWN = 0,
  RDC_LINK_STATE_UP,
  RDC_LINK_STATE_DISABLED,
} rdc_link_state_t;

#define RDC_MAX_NUM_OF_LINKS 16

typedef struct {
  uint32_t gpu_index;
  uint32_t num_of_links;                // The size of the array link_states
  rdc_topology_link_type_t link_types;  // XGMI, PCIe, and so on
  rdc_link_state_t link_states[RDC_MAX_NUM_OF_LINKS];
} rdc_gpu_link_status_t;

typedef struct {
  int32_t num_of_gpus;  // The size of gpus array
  rdc_gpu_link_status_t gpus[RDC_MAX_NUM_DEVICES];
} rdc_link_status_t;

/**
 * @brief type of health watches
 */
typedef enum {
  RDC_HEALTH_WATCH_PCIE = 0x1,      //!< PCIe system watches
  RDC_HEALTH_WATCH_XGMI = 0x2,      //!< XGMI system watches
  RDC_HEALTH_WATCH_MEM = 0x4,       //!< Memory watches
  RDC_HEALTH_WATCH_EEPROM = 0x8,    //!< EEPROM watches
  RDC_HEALTH_WATCH_THERMAL = 0x10,  //!< Temperature watches
  RDC_HEALTH_WATCH_POWER = 0x20,    //!< Power watches
} rdc_health_system_t;

/**
 * @brief type of health result
 */
typedef enum {
  RDC_HEALTH_RESULT_PASS,  //!< The health test pass
  RDC_HEALTH_RESULT_WARN,  //!< The health test has warnings
  RDC_HEALTH_RESULT_FAIL   //!< The health test fail
} rdc_health_result_t;

/**
 * @brief The maximum length of the health messages
 */
#define MAX_HEALTH_MSG_LENGTH 4096

/**
 * 8 replays per minute is the maximum recommended
 */
#define PCIE_MAX_REPLAYS_PERMIN 8

// The error code set at rdc_health_incidents_t.error.code
typedef enum {
  RDC_FR_PCI_REPLAY_RATE = 1000,
  RDC_FR_ECC_UNCORRECTABLE_DETECTED = 1001,
  RDC_FR_PENDING_PAGE_RETIREMENTS = 1002,
  RDC_FR_RETIRED_PAGES_LIMIT = 1003,
  RDC_FR_RETIRED_PAGES_UNCORRECTABLE_LIMIT = 1004,
  RDC_FR_CLOCKS_THROTTLE_THERMAL = 1005,
  RDC_FR_CLOCKS_THROTTLE_POWER = 1006,
  RDC_FR_XGMI_SINGLE_ERROR = 1007,
  RDC_FR_XGMI_MULTIPLE_ERROR = 1008,
  RDC_FR_CORRUPT_EEPROM = 1009
} rdc_health_error_code_t;

/**
 * @brief details of the health errors
 */
typedef struct {
  char msg[MAX_HEALTH_MSG_LENGTH];  //!< The test result details
  uint32_t code;                    //!< The low level error code
} rdc_health_detail_t;

/**
 * @brief details of the per health incidents
 */
typedef struct {
  uint32_t gpu_index;             //!< which GPU in this group have the issue
  rdc_health_system_t component;  //!< which components have the issue
  rdc_health_result_t health;     //!< health diagnosis of this incident
  rdc_health_detail_t error;      //!< The details of the error, rdc_health_error_code_t
} rdc_health_incidents_t;

#define HEALTH_MAX_ERROR_ITEMS 64

/**
 * @brief The health responses for test cases
 */
typedef struct {
  rdc_health_result_t overall_health;  //!< The overall health of this entire host
  unsigned int incidents_count;        //!< The number of health incidents reported in this struct
  rdc_health_incidents_t incidents[HEALTH_MAX_ERROR_ITEMS];  //!< Report of the errors detected
} rdc_health_response_t;

/**
 * @brief property id's for the configuration set/get
 */
typedef enum {
  RDC_CFG_GFX_CLOCK_LIMIT,
  RDC_CFG_MEMORY_CLOCK_LIMIT,
  RDC_CFG_POWER_LIMIT
} rdc_config_type_t;

/**
 * @brief Value mapped to rdc_config_type_t property id for the configuration set/get
 */
typedef struct {
  rdc_config_type_t type;
  uint64_t target_value;
} rdc_config_setting_t;

/**
 * @brief Array of properties collected using the configuration get
 */
typedef struct {
  uint32_t total_settings;
  rdc_config_setting_t settings[RDC_MAX_CONFIG_SETTINGS];
} rdc_config_setting_list_t;

/**
 *  @brief Initialize ROCm RDC.
 *
 *  @details When called, this initializes internal data structures,
 *  including those corresponding to sources of information that RDC provides.
 *  This must be called before rdc_start_embedded() or rdc_connect()
 *
 *  @param[in] init_flags init_flags Bit flags that tell RDC how to initialize.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_init(uint64_t init_flags);

/**
 *  @brief Shutdown ROCm RDC.
 *
 *  @details Do any necessary clean up.
 */
rdc_status_t rdc_shutdown();

/**
 *  @brief Start embedded RDC agent within this process.
 *
 *  @details The RDC is loaded as library so that it does not require rdcd
 *  daemon. In this mode, the user has to periodically call
 *  rdc_field_update_all() when op_mode is RDC_OPERATION_MODE_MANUAL, which
 *  tells RDC to collect the stats.
 *
 *  @param[in] op_mode Operation modes. When RDC_OPERATION_MODE_AUTO, RDC
 * schedules background task to collect the stats. When
 * RDC_OPERATION_MODE_MANUAL, the user needs to call rdc_field_update_all()
 * periodically.
 *
 *  @param[inout] p_rdc_handle Caller provided pointer to rdc_handle_t. Upon
 *  successful call, the value will contain the handler for following API calls.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_start_embedded(rdc_operation_mode_t op_mode, rdc_handle_t* p_rdc_handle);

/**
 *  @brief Stop embedded RDC agent.
 *
 *  @details Stop the embedded RDC agent, and p_rdc_handle becomes
 *  invalid after this call.
 *
 *  @param[in] p_rdc_handle The RDC handler that come from
 *  rdc_start_embedded().
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_stop_embedded(rdc_handle_t p_rdc_handle);

/**
 *  @brief Connect to rdcd daemon
 *
 *  @details This method is used to connect to a remote stand-alone
 *  rdcd daemon.
 *
 *  @param[in] ipAndPort The IP and port of the remote rdcd. The ipAndPort
 *  can be specified in this x.x.x.x:yyyy format, where x.x.x.x is the
 *  IP address and yyyy is the port.
 *
 *  @param[inout] p_rdc_handle Caller provided pointer to rdc_handle_t. Upon
 *  successful call, the value will contain the handler
 *  for following API calls.
 *
 *  @param [in] root_ca The root CA stored in the string in pem format. Set it
 *  as nullptr if the communication is not encrypted.
 *
 *  @param [in] client_cert The client certificate stored in the string in pem
 *  format. Set it as nullptr if the communication is not encrypted.
 *
 *  @param [in] client_key The client key stored in the string in pem format.
 *  Set it as nullptr if the communication is not encrypted.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_connect(const char* ipAndPort, rdc_handle_t* p_rdc_handle, const char* root_ca,
                         const char* client_cert, const char* client_key);

/**
 *  @brief Disconnect from rdcd daemon.
 *
 *  @details Disconnect from rdcd daemon, and p_rdc_handle becomes invalid
 *  after this call.
 *
 *  @param[in] p_rdc_handle The RDC handler that come from rdc_connect().
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_disconnect(rdc_handle_t p_rdc_handle);

/**
 *  @brief Request the RDC to watch the job stats
 *
 *  @details This should be executed as part of job prologue. The summary
 *  job stats can be retrieved using rdc_job_get_stats().
 *  In RDC_OPERATION_MODE_MANUAL, user must call rdc_field_update_all(1)
 *  at least once, before call rdc_job_get_stats()
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The group of GPUs to be watched.
 *
 *  @param[in] job_id The name of the job.
 *
 *  @param[in] update_freq  How often to update this field in usec.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_job_start_stats(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                                 const char job_id[64], uint64_t update_freq);

/**
 *  @brief Get the stats of the job using the job id.
 *
 *  @details The stats can be retrieved at any point when the job is in
 *  process.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] job_id The name of the job.
 *
 *  @param[inout] p_job_info Caller provided pointer to rdc_job_info_t. Upon
 *  successful call, the value will contain the stats of the job.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_job_get_stats(rdc_handle_t p_rdc_handle, const char job_id[64],
                               rdc_job_info_t* p_job_info);

/**
 *  @brief Request RDC to stop watching the stats of the job
 *
 *  @details This should be execute as part of job epilogue. The job Id
 *  remains available to view the stats at any point. You must call
 *  rdc_watch_job_fields() before this call.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] job_id The name of the job.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_job_stop_stats(rdc_handle_t p_rdc_handle, const char job_id[64]);

/**
 *  @brief Request RDC to stop tracking the job given by job_id
 *
 *  @details After this call, you will no longer be able to call
 *  rdc_job_get_stats() on this job_id. But you will be able to reuse
 *  the job_id after this call.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] job_id The name of the job.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_job_remove(rdc_handle_t p_rdc_handle, const char job_id[64]);

/**
 *  @brief Request RDC to stop tracking all the jobs
 *
 *  @details After this call, you will no longer be able to call
 *  rdc_job_get_stats() on any job id. But you will be able to reuse
 *  the any previous used job id after this call.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_job_remove_all(rdc_handle_t p_rdc_handle);

/**
 *  @brief Request RDC to update all fields to be watched.
 *
 *  @details In RDC_OPERATION_MODE_MANUAL, the user must call this method
 *  periodically.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] wait_for_update Whether or not to wait for the update loop to
 *  complete before returning to the caller 1=wait. 0=do not wait.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_field_update_all(rdc_handle_t p_rdc_handle, uint32_t wait_for_update);

/**
 *  @brief Get indexes corresponding to all the devices on the system.
 *
 *  @details Indexes represents RDC GPU Id corresponding to each GPU on the
 * system and is immutable during the lifespan of the engine. The list
 * should be queried again if the engine is restarted.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[out] gpu_index_list Array reference to fill GPU indexes present on
 *  the system.
 *
 *  @param[out] count Number of GPUs returned in gpu_index_list.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_device_get_all(rdc_handle_t p_rdc_handle,
                                uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES], uint32_t* count);

/**
 *  @brief Gets device attributes corresponding to the gpu_index.
 *
 *  @details Fetch the attributes, such as device name, of a GPU.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] gpu_index GPU index corresponding to which the attributes
 *  should be fetched
 *
 *  @param[out] p_rdc_attr GPU attribute corresponding to the gpu_index.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_device_get_attributes(rdc_handle_t p_rdc_handle, uint32_t gpu_index,
                                       rdc_device_attributes_t* p_rdc_attr);

/**
 *  @brief Get version information of components used by rdc.
 *
 *  @details Given a component type, return its version information.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] component Type of Components. See rdc_component_t definition for details.
 *
 *  @param[out] p_rdc_compv Version information of the corresponding component.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_device_get_component_version(rdc_handle_t p_rdc_handle, rdc_component_t component,
                                              rdc_component_version_t* p_rdc_compv);

/**
 *  @brief Create a group contains multiple GPUs
 *
 *  @details This method can create a group contains multiple GPUs. Instead of
 *  executing an operation separately for each GPU, the RDC group enables
 *  the user to execute same operation on all the GPUs present in the group as
 *  a single API call.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] type The type of the group. RDC_GROUP_DEFAULT includes all the
 *  GPUs on the node, and RDC_GROUP_EMPTY creates an empty group.
 *
 *  @param[in] group_name The group name specified as NULL terminated C String
 *
 *  @param[inout] p_rdc_group_id Caller provided pointer to rdc_gpu_group_t.
 *  Upon successful call, the value will contain the group id for following
 *  group API calls.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_gpu_create(rdc_handle_t p_rdc_handle, rdc_group_type_t type,
                                  const char* group_name, rdc_gpu_group_t* p_rdc_group_id);

/**
 *  @brief Add a GPU to the group
 *
 *  @details This method can add a GPU to the group
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The group id to which the GPU will be added.
 *
 *  @param[in] gpu_index The GPU index to be added to the group.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_gpu_add(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                               uint32_t gpu_index);

/**
 *  @brief Get information about a GPU group
 *
 *  @details Get detail information about a GPU group created by
 *  rdc_group_gpu_create
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] p_rdc_group_id The GPU group handler created by
 *  rdc_group_gpu_create
 *
 *  @param[out] p_rdc_group_info The information of the GPU
 *  group p_rdc_group_id.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_gpu_get_info(rdc_handle_t p_rdc_handle, rdc_gpu_group_t p_rdc_group_id,
                                    rdc_group_info_t* p_rdc_group_info);

/**
 *  @brief Used to get information about all GPU groups in the system.
 *
 *  @details Get the list of GPU group ids in the system.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[out] group_id_list Array reference to fill GPU group
 *  ids in the system.
 *
 *  @param[out] count Number of GPU group returned in group_id_list.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_get_all_ids(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id_list[],
                                   uint32_t* count);

/**
 *  @brief Destroy GPU group represented by p_rdc_group_id
 *
 *  @details Delete the logic group represented by p_rdc_group_id
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] p_rdc_group_id The group id
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_gpu_destroy(rdc_handle_t p_rdc_handle, rdc_gpu_group_t p_rdc_group_id);

/**
 *  @brief create a group of fields
 *
 *  @details The user can create a group of fields and perform an operation
 *  on a group of fields at once.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] num_field_ids Number of field IDs that are being provided
 *  in field_ids.
 *
 *  @param[in] field_ids Field IDs to be added to the newly-created
 *  field group.
 *
 *  @param[in] field_group_name Unique name for this group of fields.
 *
 *  @param[out] rdc_field_group_id Handle to the newly-created field group
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_field_create(rdc_handle_t p_rdc_handle, uint32_t num_field_ids,
                                    rdc_field_t* field_ids, const char* field_group_name,
                                    rdc_field_grp_t* rdc_field_group_id);

/**
 *  @brief Get information about a field group
 *
 *  @details Get detail information about a field group created by
 *  rdc_group_field_create
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] rdc_field_group_id The field group handler created by
 *  rdc_group_field_create
 *
 *  @param[out] field_group_info The information of the field group
 *  rdc_field_group_id.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_field_get_info(rdc_handle_t p_rdc_handle, rdc_field_grp_t rdc_field_group_id,
                                      rdc_field_group_info_t* field_group_info);

/**
 *  @brief Used to get information about all field groups in the system.
 *
 *  @details Get the list of field group ids in the system.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[out] field_group_id_list Array reference to fill field group
 *  ids in the system.
 *
 *  @param[out] count Number of field group returned in field_group_id_list.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_field_get_all_ids(rdc_handle_t p_rdc_handle,
                                         rdc_field_grp_t field_group_id_list[], uint32_t* count);

/**
 *  @brief Destroy field group represented by rdc_field_group_id
 *
 *  @details Delete the logic group represented by rdc_field_group_id
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] rdc_field_group_id The field group id
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_field_destroy(rdc_handle_t p_rdc_handle, rdc_field_grp_t rdc_field_group_id);

/**
 *  @brief Request the RDC start recording updates for a given field
 *  collection.
 *
 *  @details Note that the first update of the field will not occur
 *  until the next field update cycle. To force a field update cycle,
 *  user must call rdc_field_update_all(1)
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The group of GPUs to be watched.
 *
 *  @param[in] field_group_id  The collection of fields to record
 *
 *  @param[in] update_freq  How often to update fields in usec.
 *
 *  @param[in] max_keep_age How long to keep data for fields in seconds.
 *
 *  @param[in] max_keep_samples Maximum number of samples to keep. 0=no limit.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_field_watch(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                             rdc_field_grp_t field_group_id, uint64_t update_freq,
                             double max_keep_age, uint32_t max_keep_samples);

/**
 *  @brief Request a latest cached field of a GPU
 *
 *  @details Note that the field can be cached after called rdc_field_watch
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] gpu_index The GPU index.
 *
 *  @param[in] field  The field id
 *
 *  @param[out] value  The field value got from cache.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_field_get_latest_value(rdc_handle_t p_rdc_handle, uint32_t gpu_index,
                                        rdc_field_t field, rdc_field_value* value);

/**
 *  @brief Request a history cached field of a GPU
 *
 *  @details Note that the field can be cached after called rdc_field_watch
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] gpu_index The GPU index.
 *
 *  @param[in] field  The field id
 *
 *  @param[in] since_time_stamp  Timestamp to request values since in
 *  usec since 1970.
 *
 *  @param[out] next_since_time_stamp Timestamp to use for sinceTimestamp
 *  on next call to this function
 *
 *  @param[out] value  The field value got from cache.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_field_get_value_since(rdc_handle_t p_rdc_handle, uint32_t gpu_index,
                                       rdc_field_t field, uint64_t since_time_stamp,
                                       uint64_t* next_since_time_stamp, rdc_field_value* value);

/**
 *  @brief Stop record updates for a given field collection.
 *
 *  @details The cache of those fields will not be updated after this call
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The GPU group id.
 *
 *  @param[in] field_group_id  The field group id.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_field_unwatch(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                               rdc_field_grp_t field_group_id);

/**
 *  @brief Run the diagnostic test cases
 *
 *  @details Run the diagnostic test cases at different levels.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The GPU group id.
 *
 *  @param[in] level  The level decides how long the test will run.
 *  The RDC_DIAG_LVL_SHORT only take a few seconds, and the
 *  the RDC_DIAG_LVL_LONG may take up to 15 minutes.
 *
 *  @param[in] config  Implementation specific configuration.
 *
 *  @param[in] config_size  Length of the configuration.
 *
 *  @param[inout] response  The detail results of the tests run.
 *
 *  @param[inout] callback  Callback for realtime communication
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_diagnostic_run(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                                rdc_diag_level_t level, const char* config, size_t config_size,
                                rdc_diag_response_t* response, rdc_diag_callback_t* callback);

/**
 *  @brief Run one diagnostic test case
 *
 *  @details Run a specific diagnostic test case.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The GPU group id.
 *
 *  @param[in] test_case  The test case to run.
 *
 *  @param[in] config  Implementation specific configuration.
 *
 *  @param[in] config_size  Length of the configuration.
 *
 *  @param[inout] result  The results of the test.
 *
 *  @param[inout] callback  Callback for realtime communication
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_test_case_run(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                               rdc_diag_test_cases_t test_case, const char* config,
                               size_t config_size, rdc_diag_test_result_t* result,
                               rdc_diag_callback_t* callback);

/**
 *  @brief Get a description of a provided RDC error status
 *
 *  @details return the string in human readable format.
 *
 *  @param[in] status The RDC status.
 *
 *  @retval The string to describe the RDC status.
 */
const char* rdc_status_string(rdc_status_t status);

/**
 *  @brief Get the name of a field
 *
 *  @details return the string in human readable format.
 *
 *  @param[in] field_id The field id.
 *
 *  @retval The string to describe the field.
 */
const char* field_id_string(rdc_field_t field_id);

/**
 *  @brief Get the field id from name
 *
 *  @details return the field id from field name.
 *
 *  @param[in] name The field name.
 *
 *  @retval return RDC_FI_INVALID if the field name is invalid.
 */
rdc_field_t get_field_id_from_name(const char* name);

/**
 *  @brief Get a description of a diagnostic result.
 *
 *  @details return the string in human readable format.
 *
 *  @param[in] result The RDC diagnostic result.
 *
 *  @retval The string to describe the RDC diagnostic result.
 */
const char* rdc_diagnostic_result_string(rdc_diag_result_t result);

/**
 *  @brief Set the RDC policy. Each group has multiple policies, these policies can be set by this
 * API one by one. Multiple calls of this API will override the existing policy.
 *
 *  @details Set the RDC policy
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The GPU group id.
 *
 *  @param[in] policy  The policy to set
 *
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_policy_set(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                            rdc_policy_t policy);

#define RDC_MAX_POLICY_SETTINGS 32

/**
 *  @brief Get the RDC policy
 *
 *  @details Get the RDC policy
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The GPU group id.
 *
 *  @param[out] count The size of policies array
 *
 *  @param[out] policies  The policies to get
 *
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_policy_get(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id, uint32_t* count,
                            rdc_policy_t policies[RDC_MAX_POLICY_SETTINGS]);

/**
 *  @brief delete the RDC policy for this group based on condition type
 *
 *  @details clear the RDC policy for this group based on condition type. In a GPU group, only one
 * policy can be set for a specific rdc_policy_condition_type_t
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The GPU group id
 *
 *  @param[in] condition_type The condition type to delete
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */

rdc_status_t rdc_policy_delete(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                               rdc_policy_condition_type_t condition_type);

/**
 * Define the structure is used in RDC policy callback
 */
typedef struct {
  unsigned int version;
  rdc_policy_condition_t condition;  //!< the condition that is meet
  rdc_gpu_group_t group_id;          //!< The group id trigger this callback
  int64_t value;                     //!< The current value that meet the condition
} rdc_policy_callback_response_t;

/**
 * The user data is the rdc_policy_callback_response_t
 */
typedef int (*rdc_policy_register_callback)(rdc_policy_callback_response_t* userData);

/**
 *  @brief Register a function to be called when policy condition is meet.
 *
 *  @details Register the RDC policy callback
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The GPU group id.
 *
 *  @param[in] callback  The callback function to be called when condition meet.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_policy_register(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                                 rdc_policy_register_callback callback);

/**
 *  @brief un-register a policy callback function for a conditioin.
 *
 *  @details Un-register the policy callback for a condition.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The GPU group id.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_policy_unregister(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id);

/**
 *  @brief enable the health check for a group
 *
 *  @details For each group, only one parameter can be set. If you want to
 *  clear the setting for a group, set component == 0x0
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The GPU group id.
 *
 *  @param[in] components  The list of components that should be enabled for health check
 *  for example, RDC_HEALTH_WATCH_THERMAL | RDC_HEALTH_WATCH_POWER
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_health_set(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                            unsigned int components);

/**
 *  @brief get the health check settings of a group
 *
 *  @details get the health check settings of a component
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The GPU group id.
 *
 *  @param[out] components  The list of components that should be enabled for health check
 *  for example, RDC_HEALTH_WATCH_THERMAL | RDC_HEALTH_WATCH_POWER
 *  if it is 0x0, then the health check not set for the group yet.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_health_get(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                            unsigned int* components);

/**
 *  @brief Check health watch results
 *
 *  @details If it has incidents.
 *  For each incident, check the component and error message.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The GPU group id.
 *
 *  @param[inout] response  The detail results of the health.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_health_check(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                              rdc_health_response_t* response);

/**
 *  @brief clear the health watch
 *
 *  @details For each group, clear the setting.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The GPU group id.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_health_clear(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id);

/**
 *  @brief Get the topology of the device
 *
 *  @details topology of the device
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] gpu_index The GPU gpu index.
 *
 *  @param[out] results  The device topology
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_device_topology_get(rdc_handle_t p_rdc_handle, uint32_t gpu_index,
                                     rdc_device_topology_t* results);
/**
 *  @brief Get the link status
 *
 *  @details the link is up or down
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *
 *  @param[out] results
 * lts  The link up or down status
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_link_status_get(rdc_handle_t p_rdc_handle, rdc_link_status_t* results);
/**
 *  @brief Set one configuration
 *
 *  @details Set the given configuration to all nodes belong to the given group
 *
 *  @param[in] p_rdc_handle Node handle
 *
 *  @param[in] group_id Group id to which node belongs
 *
 *  @param[in] setting Configuration to be set for the nodes
 *
 *  @retval RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_config_set(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                            rdc_config_setting_t setting);

/**
 *  @brief Get the configrations
 *
 *  @details Get all the configurations for all nodes belong to the given group
 *
 *  @param[in] p_rdc_handle Node handle
 *
 *  @param[in] group_id Group id to which nodes belong
 *
 *  @param[out] settings List of configurations returned.
 *
 *  @retval RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_config_get(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id,
                            rdc_config_setting_list_t* settings);

/**
 *  @brief Clear the setting
 *
 *  @details Clear all the configurations for the nodes belongs to the given group
 *
 *  @param[in] p_rdc_handle Node handle
 *
 *  @param[in] group_id Group id to which nodes belong
 *
 *  @retval RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_config_clear(rdc_handle_t p_rdc_handle, rdc_gpu_group_t group_id);

const char* get_rocm_path(const char* search_string);

/**
 * @brief The device role
 */
typedef enum {
  RDC_DEVICE_ROLE_PHYSICAL,
  RDC_DEVICE_ROLE_PARTITION_INSTANCE  //!< The partition instance
} rdc_device_role_t;

/**
 * @brief The device type
 */
typedef enum { RDC_DEVICE_TYPE_GPU, RDC_DEVICE_TYPE_CPU } rdc_device_type_t;

typedef struct {
  uint32_t device_index;          //!< Physical device index
  uint32_t instance_index;        //!< Instance or core index
  rdc_device_role_t entity_role;  //!< Physical device or partition instance
  rdc_device_type_t device_type;  //!< Type
} rdc_entity_info_t;

/**
 * @brief The function to decode the entity info from entity index
 * @details
 * | 31 30 29| 28 27 | 21 20 19 ... 12 11 |    10 9 8 7 6 5 4 3 2 1 0 |
 * |---------|-------|--------------------|---------------------------|
 * | Type    | Role  |     Instance       |     Device                |
 * |---------|-------|--------------------|---------------------------|
 *  the 32 bit entity index is crafted based on above structure, this function
 *  will decode them into a data structure
 *
 *  @param[in] entity_index The entity index.
 *
 *  @retval rdc_entity_info_t is returned for decode structure
 */

rdc_entity_info_t rdc_get_info_from_entity_index(uint32_t entity_index);

/**
 * @brief The function to encode the entity info to entity index
 * @details
 * | 31 30 29| 28 27 | 21 20 19 ... 12 11 |    10 9 8 7 6 5 4 3 2 1 0 |
 * |---------|-------|--------------------|---------------------------|
 * | Type    | Role  |     Instance       |     Device                |
 * |---------|-------|--------------------|---------------------------|
 *  the 32 bit entity index is crafted based on above structure, this function
 *  will encode them to index
 *
 *  @param[in] info The entity info to encode.
 *
 *  @retval entity_index is returned
 */
uint32_t rdc_get_entity_index_from_info(rdc_entity_info_t info);

// map from amdsmi_accelerator_partition_resource_type_t
typedef enum {
  RDC_ACCELERATOR_XCC = 0,
  RDC_ACCELERATOR_ENCODER,
  RDC_ACCELERATOR_DECODER,
  RDC_ACCELERATOR_DMA,
  RDC_ACCELERATOR_JPEG,
  RDC_ACCELERATOR_RESOURCE_MAX,
  RDC_ACCELERATOR_LAST = RDC_ACCELERATOR_RESOURCE_MAX
} rdc_instance_resource_type_t;

// map from amdsmi_accelerator_partition_resource_profile_t
typedef struct {
  rdc_instance_resource_type_t resource_type;
  uint32_t partition_resource;  // The resources a partition can be used, which may be shared
  uint32_t num_partitions_share_resource;  // If it is greater than 1, then resource is shared.
} rdc_resource_profile_t;

/**
 *  @brief Query the resource allocation for a device/instance
 *
 *  @details The profile contains detail information how resource is allocated.
 *
 *  As an example, MI300X has 8 XCCs and 4 Decoders, in DPX mode, the physical device is
 *  partitioned to 2 instances, so each instance will have 4 XCC and 2 Decoder and they are
 *  not shared.
 *  [XCC, 4, 0], [DECODER, 2, 0]
 *
 *  If it is CPX mode, the physical device is partitioned to 8 instances, and each instance
 *  have 1 XCC and 2 instances are sharing the same decoder.
 *  [XCC, 1, 0], [DECODER, 1, 1]
 *
 *  If entity_index is the physical device, it should return all resources of the device:
 *  [XCC, 8, 0], [DECODER, 4, 0]
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] entity_index The GPU index to query. It can be physical device or instance.
 *
 *  @param[in] resource_type Which resource type to query
 *
 *  @param[out] profile  The details how the resource is allocated.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_instance_profile_get(rdc_handle_t p_rdc_handle, uint32_t entity_index,
                                      rdc_instance_resource_type_t resource_type,
                                      rdc_resource_profile_t* profile);

/**
 * @brief Get the number of partitions for the specified GPU index.
 *
 * @param[in] p_rdc_handle The RDC handler.
 * @param[in] index The GPU index to query.
 * @param[out] num_partition Pointer to a variable to receive the number of partitions.
 *
 * @retval ::RDC_ST_OK on success.
 */
rdc_status_t rdc_get_num_partition(rdc_handle_t p_rdc_handle, uint32_t index,
                                   uint16_t* num_partition);

/**
 * @brief Check if gpuid is partition string
 *
 * @param[in] s - singular partition string
 * @retval bool - if partition string or not
 */
bool rdc_is_partition_string(const char* s);

/**
 * @brief Parse partition id into physical gpu and partition
 *
 * @param[in] s - singular partition string
 * @param[out] physicalGpu - socket id
 * @param[out] partition - partition id
 *
 * @retval bool - success
 */
bool rdc_parse_partition_string(const char* s, uint32_t* physicalGpu, uint32_t* partition);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // INCLUDE_RDC_RDC_H_
