/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017-2025, Advanced Micro Devices, Inc.
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

#ifndef ROCM_SMI_ROCM_SMI_H_
#define ROCM_SMI_ROCM_SMI_H_

#ifdef __cplusplus
extern "C" {
#include <cstdint>
#else
#include <stdint.h>
#endif  // __cplusplus

#include <stddef.h>
#include <stdbool.h>

#include "rocm_smi/kfd_ioctl.h"

/** \file rocm_smi.h
 *  Main header file for the ROCm SMI library.
 *  All required function, structure, enum, etc. definitions should be defined
 *  in this file.
 *
 *  @brief The rocm_smi library api is new, and therefore subject to change
 *  either at the ABI or API level. Instead of marking every function prototype as "unstable", we are
 *  instead saying the API is unstable (i.e., changes are possible) while the
 *  major version remains 0. This means that if the API/ABI changes, we will
 *  not increment the major version to 1. Once the ABI stabilizes, we will
 *  increment the major version to 1, and thereafter increment it on all ABI
 *  breaks.
 */

//! Guaranteed maximum possible number of supported frequencies
//! (32 normal + 1 sleep frequency)
#define RSMI_MAX_NUM_FREQUENCIES 33

//! Maximum possible value for fan speed. Should be used as the denominator
//! when determining fan speed percentage.
#define RSMI_MAX_FAN_SPEED 255

//! The number of points that make up a voltage-frequency curve definition
#define RSMI_NUM_VOLTAGE_CURVE_POINTS 3


/**
 * @brief Error codes retured by rocm_smi_lib functions
 */
typedef enum {
  RSMI_STATUS_SUCCESS = 0x0,             //!< Operation was successful
  RSMI_STATUS_INVALID_ARGS,              //!< Passed in arguments are not valid
  RSMI_STATUS_NOT_SUPPORTED,             //!< The requested information or
                                         //!< action is not available for the
                                         //!< given input, on the given system
  RSMI_STATUS_FILE_ERROR,                //!< Problem accessing a file. This
                                         //!< may because the operation is not
                                         //!< supported by the Linux kernel
                                         //!< version running on the executing
                                         //!< machine
  RSMI_STATUS_PERMISSION,                //!< Permission denied/EACCESS file
                                         //!< error. Many functions require
                                         //!< root access to run.
  RSMI_STATUS_OUT_OF_RESOURCES,          //!< Unable to acquire memory or other
                                         //!< resource
  RSMI_STATUS_INTERNAL_EXCEPTION,        //!< An internal exception was caught
  RSMI_STATUS_INPUT_OUT_OF_BOUNDS,       //!< The provided input is out of
                                         //!< allowable or safe range
  RSMI_STATUS_INIT_ERROR,                //!< An error occurred when rsmi
                                         //!< initializing internal data
                                         //!< structures
  RSMI_INITIALIZATION_ERROR = RSMI_STATUS_INIT_ERROR,
  RSMI_STATUS_NOT_YET_IMPLEMENTED,       //!< The requested function has not
                                         //!< yet been implemented in the
                                         //!< current system for the current
                                         //!< devices
  RSMI_STATUS_NOT_FOUND,                 //!< An item was searched for but not
                                         //!< found
  RSMI_STATUS_INSUFFICIENT_SIZE,         //!< Not enough resources were
                                         //!< available for the operation
  RSMI_STATUS_INTERRUPT,                 //!< An interrupt occurred during
                                         //!< execution of function
  RSMI_STATUS_UNEXPECTED_SIZE,           //!< An unexpected amount of data
                                         //!< was read
  RSMI_STATUS_NO_DATA,                   //!< No data was found for a given
                                         //!< input
  RSMI_STATUS_UNEXPECTED_DATA,           //!< The data read or provided to
                                         //!< function is not what was expected
  RSMI_STATUS_BUSY,                      //!< A resource or mutex could not be
                                         //!< acquired because it is already
                                         //!< being used
  RSMI_STATUS_REFCOUNT_OVERFLOW,         //!< An internal reference counter
                                         //!< exceeded INT32_MAX
  RSMI_STATUS_SETTING_UNAVAILABLE,       //!< Requested setting is unavailable
                                         //!< for the current device
  RSMI_STATUS_AMDGPU_RESTART_ERR,        //!< Could not successfully restart
                                         //!< the amdgpu driver
  RSMI_STATUS_DRM_ERROR,                 //!< Error when call libdrm
  RSMI_STATUS_FAIL_LOAD_MODULE,          //!< Fail to load lib
  RSMI_STATUS_FAIL_LOAD_SYMBOL,          //!< Fail to load symbol

  RSMI_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF,  //!< An unknown error occurred
} rsmi_status_t;

/**
 * @brief Initialization flags
 *
 * Initialization flags may be OR'd together and passed to ::rsmi_init().
 */

typedef enum {
  RSMI_INIT_FLAG_ALL_GPUS   = 0x1,       //!< Attempt to add all GPUs found
                                         //!< (including non-AMD) to the list
                                         //!< of devices from which SMI
                                         //!< information can be retrieved. By
                                         //!< default, only AMD devices are
                                         //!<  enumerated by RSMI.
  RSMI_INIT_FLAG_THRAD_ONLY_MUTEX = 0x400000000000000,   //!< The mutex limit to thread
  RSMI_INIT_FLAG_RESRV_TEST1 = 0x800000000000000,  //!< Reserved for test
} rsmi_init_flags_t;

/**
 * @brief PowerPlay performance levels
 */
typedef enum {
  RSMI_DEV_PERF_LEVEL_AUTO = 0,       //!< Performance level is "auto"
  RSMI_DEV_PERF_LEVEL_FIRST = RSMI_DEV_PERF_LEVEL_AUTO,

  RSMI_DEV_PERF_LEVEL_LOW,              //!< Keep PowerPlay levels "low",
                                        //!< regardless of workload
  RSMI_DEV_PERF_LEVEL_HIGH,             //!< Keep PowerPlay levels "high",
                                        //!< regardless of workload
  RSMI_DEV_PERF_LEVEL_MANUAL,           //!< Only use values defined by manually
                                        //!< setting the RSMI_CLK_TYPE_SYS speed
  RSMI_DEV_PERF_LEVEL_STABLE_STD,       //!< Stable power state with profiling
                                        //!< clocks
  RSMI_DEV_PERF_LEVEL_STABLE_PEAK,      //!< Stable power state with peak clocks
  RSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK,  //!< Stable power state with minimum
                                        //!< memory clock
  RSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK,  //!< Stable power state with minimum
                                        //!< system clock
  RSMI_DEV_PERF_LEVEL_DETERMINISM,      //!< Performance determinism state

  RSMI_DEV_PERF_LEVEL_LAST = RSMI_DEV_PERF_LEVEL_DETERMINISM,

  RSMI_DEV_PERF_LEVEL_UNKNOWN = 0x100   //!< Unknown performance level
} rsmi_dev_perf_level_t;
/// \cond Ignore in docs.
typedef rsmi_dev_perf_level_t rsmi_dev_perf_level;
/// \endcond
/**
 * @brief Available clock types.
 */

/**
 * @brief Software components
 */
typedef enum {
  RSMI_SW_COMP_FIRST = 0x0,

  RSMI_SW_COMP_DRIVER = RSMI_SW_COMP_FIRST,    //!< Driver

  RSMI_SW_COMP_LAST = RSMI_SW_COMP_DRIVER
} rsmi_sw_component_t;

/**
 * Event counter types
 */

/**
 * @brief Handle to performance event counter
 */
typedef uintptr_t rsmi_event_handle_t;

/**
 * Event Groups
 *
 * @brief Enum denoting an event group. The value of the enum is the
 * base value for all the event enums in the group.
 */
typedef enum {
  RSMI_EVNT_GRP_XGMI = 0,         //!< Data Fabric (XGMI) related events
  RSMI_EVNT_GRP_XGMI_DATA_OUT = 10,  //!< XGMI Outbound data
  RSMI_EVNT_GRP_INVALID = 0xFFFFFFFF
} rsmi_event_group_t;

/**
 * Event types
 * @brief Event type enum. Events belonging to a particular event group
 * ::rsmi_event_group_t should begin enumerating at the ::rsmi_event_group_t
 * value for that group.
 */
typedef enum {
  RSMI_EVNT_FIRST = RSMI_EVNT_GRP_XGMI,

  RSMI_EVNT_XGMI_FIRST = RSMI_EVNT_GRP_XGMI,
  RSMI_EVNT_XGMI_0_NOP_TX = RSMI_EVNT_XGMI_FIRST,  //!< NOPs sent to neighbor 0
  RSMI_EVNT_XGMI_0_REQUEST_TX,                    //!< Outgoing requests to
                                                  //!< neighbor 0
  RSMI_EVNT_XGMI_0_RESPONSE_TX,                   //!< Outgoing responses to
                                                  //!< neighbor 0
  /**
   * @brief
   *
   * Data beats sent to neighbor 0; Each beat represents 32 bytes.<br><br>
   *
   * XGMI throughput can be calculated by multiplying a BEATs event
   * such as ::RSMI_EVNT_XGMI_0_BEATS_TX by 32 and dividing by
   * the time for which event collection occurred,
   * ::rsmi_counter_value_t.time_running (which is in nanoseconds). To get
   * bytes per second, multiply this value by 10<sup>9</sup>.<br>
   * <br>
   * Throughput = BEATS/time_running * 10<sup>9</sup>  (bytes/second)<br>
   */
  // ie, Throughput = BEATS/time_running 10^9  bytes/sec
  RSMI_EVNT_XGMI_0_BEATS_TX,
  RSMI_EVNT_XGMI_1_NOP_TX,                        //!< NOPs sent to neighbor 1
  RSMI_EVNT_XGMI_1_REQUEST_TX,                        //!< Outgoing requests to
                                                  //!< neighbor 1
  RSMI_EVNT_XGMI_1_RESPONSE_TX,                   //!< Outgoing responses to
                                                  //!< neighbor 1
  RSMI_EVNT_XGMI_1_BEATS_TX,                      //!< Data beats sent to
                                                  //!< neighbor 1; Each beat
                                                  //!< represents 32 bytes

  RSMI_EVNT_XGMI_LAST = RSMI_EVNT_XGMI_1_BEATS_TX,   // 5

  RSMI_EVNT_XGMI_DATA_OUT_FIRST = RSMI_EVNT_GRP_XGMI_DATA_OUT,  // 10

  /*
   * @brief Events in the RSMI_EVNT_GRP_XGMI_DATA_OUT group measure
   * the number of beats sent on an XGMI link. Each beat represents
   * 32 bytes. RSMI_EVNT_XGMI_DATA_OUT_n represents the number of
   * outbound beats (each representing 32 bytes) on link n.<br><br>
   *
   * XGMI throughput can be calculated by multiplying a event
   * such as ::RSMI_EVNT_XGMI_DATA_OUT_n by 32 and dividing by
   * the time for which event collection occurred,
   * ::rsmi_counter_value_t.time_running (which is in nanoseconds). To get
   * bytes per second, multiply this value by 10<sup>9</sup>.<br>
   * <br>
   * Throughput = BEATS/time_running * 10<sup>9</sup>  (bytes/second)<br>
   */
  // ie, Throughput = BEATS/time_running 10^9  bytes/sec
  RSMI_EVNT_XGMI_DATA_OUT_0 = RSMI_EVNT_XGMI_DATA_OUT_FIRST,
  RSMI_EVNT_XGMI_DATA_OUT_1,   //!< Outbound beats to neighbor 1
  RSMI_EVNT_XGMI_DATA_OUT_2,   //!< Outbound beats to neighbor 2
  RSMI_EVNT_XGMI_DATA_OUT_3,   //!< Outbound beats to neighbor 3
  RSMI_EVNT_XGMI_DATA_OUT_4,   //!< Outbound beats to neighbor 4
  RSMI_EVNT_XGMI_DATA_OUT_5,   //!< Outbound beats to neighbor 5
  RSMI_EVNT_XGMI_DATA_OUT_LAST = RSMI_EVNT_XGMI_DATA_OUT_5,

  RSMI_EVNT_LAST = RSMI_EVNT_XGMI_DATA_OUT_LAST,
} rsmi_event_type_t;

/**
 * Event counter commands
 */
typedef enum {
  RSMI_CNTR_CMD_START = 0,  //!< Start the counter
  RSMI_CNTR_CMD_STOP,       //!< Stop the counter; note that this should not
                            //!< be used before reading.
} rsmi_counter_command_t;

/**
 * Counter value
 */
typedef struct {
  uint64_t value;            //!< Counter value
  uint64_t time_enabled;     //!< Time that the counter was enabled
                             //!< (in nanoseconds)
  uint64_t time_running;     //!< Time that the counter was running
                             //!< (in nanoseconds)
} rsmi_counter_value_t;

/**
 * Event notification event types
 */
typedef enum {
  RSMI_EVT_NOTIF_NONE = KFD_SMI_EVENT_NONE,        //!< Unused
  RSMI_EVT_NOTIF_VMFAULT = KFD_SMI_EVENT_VMFAULT,  //!< VM page fault
  RSMI_EVT_NOTIF_FIRST = RSMI_EVT_NOTIF_VMFAULT,
  RSMI_EVT_NOTIF_THERMAL_THROTTLE = KFD_SMI_EVENT_THERMAL_THROTTLE,
  RSMI_EVT_NOTIF_GPU_PRE_RESET = KFD_SMI_EVENT_GPU_PRE_RESET,
  RSMI_EVT_NOTIF_GPU_POST_RESET = KFD_SMI_EVENT_GPU_POST_RESET,
  RSMI_EVT_NOTIF_EVENT_MIGRATE_START =  KFD_SMI_EVENT_MIGRATE_START,
  RSMI_EVT_NOTIF_EVENT_MIGRATE_END = KFD_SMI_EVENT_MIGRATE_END,
  RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_START = KFD_SMI_EVENT_PAGE_FAULT_START,
  RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_END = KFD_SMI_EVENT_PAGE_FAULT_END,
  RSMI_EVT_NOTIF_EVENT_QUEUE_EVICTION = KFD_SMI_EVENT_QUEUE_EVICTION,
  RSMI_EVT_NOTIF_EVENT_QUEUE_RESTORE = KFD_SMI_EVENT_QUEUE_RESTORE,
  RSMI_EVT_NOTIF_EVENT_UNMAP_FROM_GPU = KFD_SMI_EVENT_UNMAP_FROM_GPU,
  RSMI_EVT_NOTIF_EVENT_ALL_PROCESS = KFD_SMI_EVENT_ALL_PROCESS,
  RSMI_EVT_NOTIF_LAST = KFD_SMI_EVENT_ALL_PROCESS
} rsmi_evt_notification_type_t;

/**
 * Macro to generate event bitmask from event id
 */
#define RSMI_EVENT_MASK_FROM_INDEX(i) (1ULL << ((i) - 1))

//! Maximum number of characters an event notification message will be
// matches kfd message max size
#define MAX_EVENT_NOTIFICATION_MSG_SIZE 96

/**
 * Event notification data returned from event notification API
 */
typedef struct {
    uint32_t dv_ind;        //!< Index of device that corresponds to the event
    rsmi_evt_notification_type_t event;     //!< Event type
    char message[MAX_EVENT_NOTIFICATION_MSG_SIZE];  //!< Event message
} rsmi_evt_notification_data_t;

/**
 * Clock types
 */
typedef enum {
  RSMI_CLK_TYPE_SYS = 0x0,            //!< System clock
  RSMI_CLK_TYPE_FIRST = RSMI_CLK_TYPE_SYS,
  RSMI_CLK_TYPE_DF,                   //!< Data Fabric clock (for ASICs
                                      //!< running on a separate clock)
  RSMI_CLK_TYPE_DCEF,                 //!< Display Controller Engine clock
  RSMI_CLK_TYPE_SOC,                  //!< SOC clock
  RSMI_CLK_TYPE_MEM,                  //!< Memory clock
  RSMI_CLK_TYPE_PCIE,                 //!< PCIE clock

  // Add new clocks to the end (not in the middle) and update
  // RSMI_CLK_TYPE_LAST
  RSMI_CLK_TYPE_LAST = RSMI_CLK_TYPE_MEM,
  RSMI_CLK_INVALID = 0xFFFFFFFF
} rsmi_clk_type_t;
/// \cond Ignore in docs.
typedef rsmi_clk_type_t rsmi_clk_type;
/// \endcond

/**
 * @brief Compute Partition. This enum is used to identify
 * various compute partitioning settings.
 */
typedef enum {
  RSMI_COMPUTE_PARTITION_INVALID = 0,
  RSMI_COMPUTE_PARTITION_SPX, //!< Single GPU mode (SPX)- All XCCs work
                              //!< together with shared memory
  RSMI_COMPUTE_PARTITION_DPX, //!< Dual GPU mode (DPX)- Half XCCs work
                              //!< together with shared memory
  RSMI_COMPUTE_PARTITION_TPX, //!< Triple GPU mode (TPX)- One-third XCCs
                              //!< work together with shared memory
  RSMI_COMPUTE_PARTITION_QPX, //!< Quad GPU mode (QPX)- Quarter XCCs
                              //!< work together with shared memory
  RSMI_COMPUTE_PARTITION_CPX  //!< Core mode (CPX)- Per-chip XCC with
                              //!< shared memory
} rsmi_compute_partition_type_t;
/// \cond Ignore in docs.
typedef rsmi_compute_partition_type_t rsmi_compute_partition_type;
/// \endcond

/**
 * @brief Memory Partitions. This enum is used to identify various
 * memory partition types.
 */
typedef enum {
  RSMI_MEMORY_PARTITION_UNKNOWN = 0,
  RSMI_MEMORY_PARTITION_NPS1,  //!< NPS1 - All CCD & XCD data is interleaved
                               //!< accross all 8 HBM stacks (all stacks/1).
  RSMI_MEMORY_PARTITION_NPS2,  //!< NPS2 - 2 sets of CCDs or 4 XCD interleaved
                               //!< accross the 4 HBM stacks per AID pair
                               //!< (8 stacks/2).
  RSMI_MEMORY_PARTITION_NPS4,  //!< NPS4 - Each XCD data is interleaved accross
                               //!< accross 2 (or single) HBM stacks
                               //!< (8 stacks/8 or 8 stacks/4).
  RSMI_MEMORY_PARTITION_NPS8,  //!< NPS8 - Each XCD uses a single HBM stack
                               //!< (8 stacks/8). Or each XCD uses a single
                               //!< HBM stack & CCDs share 2 non-interleaved
                               //!< HBM stacks on its AID
                               //!< (AID[1,2,3] = 6 stacks/6).
} rsmi_memory_partition_type_t;
/// \cond Ignore in docs.
typedef rsmi_memory_partition_type_t rsmi_memory_partition_type;
/// \endcond

/**
 * @brief Temperature Metrics.  This enum is used to identify various
 * temperature metrics. Corresponding values will be in millidegress
 * Celcius.
 */
typedef enum {
  RSMI_TEMP_CURRENT = 0x0,   //!< Temperature current value.
  RSMI_TEMP_FIRST = RSMI_TEMP_CURRENT,

  RSMI_TEMP_MAX,             //!< Temperature max value.
  RSMI_TEMP_MIN,             //!< Temperature min value.
  RSMI_TEMP_MAX_HYST,        //!< Temperature hysteresis value for max limit.
                             //!< (This is an absolute temperature, not a
                             //!< delta).
  RSMI_TEMP_MIN_HYST,        //!< Temperature hysteresis value for min limit.
                             //!< (This is an absolute temperature,
                             //!<  not a delta).
  RSMI_TEMP_CRITICAL,        //!< Temperature critical max value, typically
                             //!<  greater than corresponding temp_max values.
  RSMI_TEMP_CRITICAL_HYST,   //!< Temperature hysteresis value for critical
                             //!<  limit. (This is an absolute temperature,
                             //!<  not a delta).
  RSMI_TEMP_EMERGENCY,       //!< Temperature emergency max value, for chips
                             //!<  supporting more than two upper temperature
                             //!<  limits. Must be equal or greater than
                             //!<  corresponding temp_crit values.
  RSMI_TEMP_EMERGENCY_HYST,  //!< Temperature hysteresis value for emergency
                             //!<  limit. (This is an absolute temperature,
                             //!<  not a delta).
  RSMI_TEMP_CRIT_MIN,        //!< Temperature critical min value, typically
                             //!<  lower than corresponding temperature
                             //!<  minimum values.
  RSMI_TEMP_CRIT_MIN_HYST,   //!< Temperature hysteresis value for critical
                             //!< minimum limit. (This is an absolute
                             //!< temperature, not a delta).
  RSMI_TEMP_OFFSET,          //!< Temperature offset which is added to the
                             //!  temperature reading by the chip.
  RSMI_TEMP_LOWEST,          //!< Historical minimum temperature.
  RSMI_TEMP_HIGHEST,         //!< Historical maximum temperature.

  RSMI_TEMP_LAST = RSMI_TEMP_HIGHEST
} rsmi_temperature_metric_t;
/// \cond Ignore in docs.
typedef rsmi_temperature_metric_t rsmi_temperature_metric;
/// \endcond

/**
 * @brief This enumeration is used to indicate from which part of the device a
 * temperature reading should be obtained.
 */
typedef enum {
  RSMI_TEMP_TYPE_FIRST = 0,

  RSMI_TEMP_TYPE_EDGE = RSMI_TEMP_TYPE_FIRST,  //!< Edge GPU temperature
  RSMI_TEMP_TYPE_JUNCTION,                     //!< Junction/hotspot
                                               //!< temperature
  RSMI_TEMP_TYPE_MEMORY,                       //!< VRAM temperature
  RSMI_TEMP_TYPE_HBM_0,                        //!< HBM temperature instance 0
  RSMI_TEMP_TYPE_HBM_1,                        //!< HBM temperature instance 1
  RSMI_TEMP_TYPE_HBM_2,                        //!< HBM temperature instance 2
  RSMI_TEMP_TYPE_HBM_3,                        //!< HBM temperature instance 3
  RSMI_TEMP_TYPE_LAST = RSMI_TEMP_TYPE_HBM_3,
  RSMI_TEMP_TYPE_INVALID = 0xFFFFFFFF          //!< Invalid type
} rsmi_temperature_type_t;

/**
 * @brief Activity (Utilization) Metrics.  This enum is used to identify
 * various activity metrics.
 *
 */
typedef enum {
  /* Utilization */
  RSMI_ACTIVITY_GFX = (0x1 << 0),
  RSMI_ACTIVITY_UMC = (0x1 << 1),   //!< memory controller
  RSMI_ACTIVITY_MM  = (0x1 << 2)    //!< UVD or VCN
} rsmi_activity_metric_t;


/**
 * @brief Voltage Metrics.  This enum is used to identify various
 * Volatge metrics. Corresponding values will be in millivolt.
 *
 */
typedef enum {
  RSMI_VOLT_CURRENT = 0x0,               //!< Voltage current value.

  RSMI_VOLT_FIRST = RSMI_VOLT_CURRENT,
  RSMI_VOLT_MAX,                         //!< Voltage max value.
  RSMI_VOLT_MIN_CRIT,                    //!< Voltage critical min value.
  RSMI_VOLT_MIN,                         //!< Voltage min value.
  RSMI_VOLT_MAX_CRIT,                    //!< Voltage critical max value.
  RSMI_VOLT_AVERAGE,                     //!< Average voltage.
  RSMI_VOLT_LOWEST,                      //!< Historical minimum voltage.
  RSMI_VOLT_HIGHEST,                     //!< Historical maximum voltage.

  RSMI_VOLT_LAST = RSMI_VOLT_HIGHEST
} rsmi_voltage_metric_t;

/**
 * @brief This ennumeration is used to indicate which type of
 * voltage reading should be obtained.
 */
typedef enum {
  RSMI_VOLT_TYPE_FIRST = 0,

  RSMI_VOLT_TYPE_VDDGFX = RSMI_VOLT_TYPE_FIRST,  //!< Vddgfx GPU voltage
  RSMI_VOLT_TYPE_VDDNB,                          //!< Vddnb iGPU
  RSMI_VOLT_TYPE_VDDBOARD,                       //!< Voltage for VDDBOARD

  RSMI_VOLT_TYPE_LAST = RSMI_VOLT_TYPE_VDDBOARD,
  RSMI_VOLT_TYPE_INVALID = 0xFFFFFFFF            //!< Invalid type
} rsmi_voltage_type_t;

/**
 * @brief Pre-set Profile Selections. These bitmasks can be AND'd with the
 * ::rsmi_power_profile_status_t.available_profiles returned from
 * ::rsmi_dev_power_profile_presets_get to determine which power profiles
 * are supported by the system.
 */
typedef enum {
  RSMI_PWR_PROF_PRST_CUSTOM_MASK = 0x1,        //!< Custom Power Profile
  RSMI_PWR_PROF_PRST_VIDEO_MASK = 0x2,         //!< Video Power Profile
  RSMI_PWR_PROF_PRST_POWER_SAVING_MASK = 0x4,  //!< Power Saving Profile
  RSMI_PWR_PROF_PRST_COMPUTE_MASK = 0x8,       //!< Compute Saving Profile
  RSMI_PWR_PROF_PRST_VR_MASK = 0x10,           //!< VR Power Profile

  //!< 3D Full Screen Power Profile
  RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK = 0x20,
  RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT = 0x40,    //!< Default Boot Up Profile
  RSMI_PWR_PROF_PRST_LAST = RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT,

  //!< Invalid power profile
  RSMI_PWR_PROF_PRST_INVALID = 0xFFFFFFFFFFFFFFFF
} rsmi_power_profile_preset_masks_t;
/// \cond Ignore in docs.
typedef rsmi_power_profile_preset_masks_t rsmi_power_profile_preset_masks;
/// \endcond

/**
 * @brief This enum is used to identify different GPU blocks.
 */
typedef enum {
  RSMI_GPU_BLOCK_INVALID =   0x0000000000000000,  //!< Used to indicate an
                                                  //!< invalid block
  RSMI_GPU_BLOCK_FIRST =     0x0000000000000001,

  RSMI_GPU_BLOCK_UMC = RSMI_GPU_BLOCK_FIRST,      //!< UMC block
  RSMI_GPU_BLOCK_SDMA =      0x0000000000000002,  //!< SDMA block
  RSMI_GPU_BLOCK_GFX =       0x0000000000000004,  //!< GFX block
  RSMI_GPU_BLOCK_MMHUB =     0x0000000000000008,  //!< MMHUB block
  RSMI_GPU_BLOCK_ATHUB =     0x0000000000000010,  //!< ATHUB block
  RSMI_GPU_BLOCK_PCIE_BIF =  0x0000000000000020,  //!< PCIE_BIF block
  RSMI_GPU_BLOCK_HDP =       0x0000000000000040,  //!< HDP block
  RSMI_GPU_BLOCK_XGMI_WAFL = 0x0000000000000080,  //!< XGMI block
  RSMI_GPU_BLOCK_DF =        0x0000000000000100,  //!< DF block
  RSMI_GPU_BLOCK_SMN =       0x0000000000000200,  //!< SMN block
  RSMI_GPU_BLOCK_SEM =       0x0000000000000400,  //!< SEM block
  RSMI_GPU_BLOCK_MP0 =       0x0000000000000800,  //!< MP0 block
  RSMI_GPU_BLOCK_MP1 =       0x0000000000001000,  //!< MP1 block
  RSMI_GPU_BLOCK_FUSE =      0x0000000000002000,  //!< Fuse block

  RSMI_GPU_BLOCK_LAST = RSMI_GPU_BLOCK_FUSE,       //!< The highest bit position
                                                  //!< for supported blocks
  RSMI_GPU_BLOCK_RESERVED =  0x8000000000000000
} rsmi_gpu_block_t;
/// \cond Ignore in docs.
typedef rsmi_gpu_block_t rsmi_gpu_block;
/// \endcond

/**
 * @brief The current ECC state
 */
typedef enum {
  RSMI_RAS_ERR_STATE_NONE = 0,   //!< No current errors
  RSMI_RAS_ERR_STATE_DISABLED,   //!< ECC is disabled
  RSMI_RAS_ERR_STATE_PARITY,     //!< ECC errors present, but type unknown
  RSMI_RAS_ERR_STATE_SING_C,     //!< Single correctable error
  RSMI_RAS_ERR_STATE_MULT_UC,    //!< Multiple uncorrectable errors
  RSMI_RAS_ERR_STATE_POISON,     //!< Firmware detected error and isolated
                                 //!< page. Treat as uncorrectable.
  RSMI_RAS_ERR_STATE_ENABLED,    //!< ECC is enabled

  RSMI_RAS_ERR_STATE_LAST = RSMI_RAS_ERR_STATE_ENABLED,
  RSMI_RAS_ERR_STATE_INVALID = 0xFFFFFFFF
} rsmi_ras_err_state_t;

/**
 * @brief Types of memory
 */
typedef enum {
  RSMI_MEM_TYPE_FIRST = 0,

  RSMI_MEM_TYPE_VRAM = RSMI_MEM_TYPE_FIRST,  //!< VRAM memory
  RSMI_MEM_TYPE_VIS_VRAM,                    //!< VRAM memory that is visible
  RSMI_MEM_TYPE_GTT,                         //!< GTT memory

  RSMI_MEM_TYPE_LAST = RSMI_MEM_TYPE_GTT
} rsmi_memory_type_t;

/**
 * @brief The values of this enum are used as frequency identifiers.
 */
typedef enum {
  RSMI_FREQ_IND_MIN = 0,  //!< Index used for the minimum frequency value
  RSMI_FREQ_IND_MAX = 1,  //!< Index used for the maximum frequency value
  RSMI_FREQ_IND_INVALID = 0xFFFFFFFF  //!< An invalid frequency index
} rsmi_freq_ind_t;
/// \cond Ignore in docs.
typedef rsmi_freq_ind_t rsmi_freq_ind;
/// \endcond


/**
 * @brief The values of this enum are used to identify the various firmware
 * blocks.
 */
typedef enum {
  RSMI_FW_BLOCK_FIRST = 0,

  RSMI_FW_BLOCK_ASD = RSMI_FW_BLOCK_FIRST,
  RSMI_FW_BLOCK_CE,
  RSMI_FW_BLOCK_DMCU,
  RSMI_FW_BLOCK_MC,
  RSMI_FW_BLOCK_ME,
  RSMI_FW_BLOCK_MEC,
  RSMI_FW_BLOCK_MEC2,
  RSMI_FW_BLOCK_MES,
  RSMI_FW_BLOCK_MES_KIQ,
  RSMI_FW_BLOCK_PFP,
  RSMI_FW_BLOCK_RLC,
  RSMI_FW_BLOCK_RLC_SRLC,
  RSMI_FW_BLOCK_RLC_SRLG,
  RSMI_FW_BLOCK_RLC_SRLS,
  RSMI_FW_BLOCK_SDMA,
  RSMI_FW_BLOCK_SDMA2,
  RSMI_FW_BLOCK_SMC,
  RSMI_FW_BLOCK_SOS,
  RSMI_FW_BLOCK_TA_RAS,
  RSMI_FW_BLOCK_TA_XGMI,
  RSMI_FW_BLOCK_UVD,
  RSMI_FW_BLOCK_VCE,
  RSMI_FW_BLOCK_VCN,

  RSMI_FW_BLOCK_LAST = RSMI_FW_BLOCK_VCN
} rsmi_fw_block_t;

/**
 * @brief XGMI Status
 */
typedef enum {
  RSMI_XGMI_STATUS_NO_ERRORS = 0,
  RSMI_XGMI_STATUS_ERROR,
  RSMI_XGMI_STATUS_MULTIPLE_ERRORS,
} rsmi_xgmi_status_t;

/**
 * @brief Bitfield used in various RSMI calls
 */
typedef uint64_t rsmi_bit_field_t;
/// \cond Ignore in docs.
typedef rsmi_bit_field_t rsmi_bit_field;
/// \endcond

/**
 * @brief Reserved Memory Page States
 */
typedef enum {
  RSMI_MEM_PAGE_STATUS_RESERVED = 0,  //!< Reserved. This gpu page is reserved
                                      //!<  and not available for use
  RSMI_MEM_PAGE_STATUS_PENDING,       //!< Pending. This gpu page is marked
                                      //!<  as bad and will be marked reserved
                                      //!<  at the next window.
  RSMI_MEM_PAGE_STATUS_UNRESERVABLE   //!< Unable to reserve this page
} rsmi_memory_page_status_t;

/**
 * @brief Types for IO Link
 */
typedef enum _RSMI_IO_LINK_TYPE {
  RSMI_IOLINK_TYPE_UNDEFINED      = 0,          //!< unknown type.
  RSMI_IOLINK_TYPE_PCIEXPRESS,                  //!< PCI Express
  RSMI_IOLINK_TYPE_XGMI,                        //!< XGMI
  RSMI_IOLINK_TYPE_NUMIOLINKTYPES,              //!< Number of IO Link types
  RSMI_IOLINK_TYPE_SIZE           = 0xFFFFFFFF  //!< Max of IO Link types
} RSMI_IO_LINK_TYPE;

//! The CPU node index which will be used in rsmi_topo_get_link_type
//! to query the link type between GPU and CPU
#define CPU_NODE_INDEX 0xFFFFFFFF

/**
 * @brief The utilization counter type
 */
typedef enum {
  RSMI_UTILIZATION_COUNTER_FIRST = 0,
  //!< GFX Activity
  RSMI_COARSE_GRAIN_GFX_ACTIVITY  = RSMI_UTILIZATION_COUNTER_FIRST,
  RSMI_COARSE_GRAIN_MEM_ACTIVITY,    //!< Memory Activity
  RSMI_UTILIZATION_COUNTER_LAST = RSMI_COARSE_GRAIN_MEM_ACTIVITY
} RSMI_UTILIZATION_COUNTER_TYPE;

/**
 * @brief Power types
 */
typedef enum {
  RSMI_AVERAGE_POWER = 0,            //!< Average Power
  RSMI_CURRENT_POWER,                //!< Current / Instant Power
  RSMI_INVALID_POWER = 0xFFFFFFFF    //!< Invalid / Undetected Power
} RSMI_POWER_TYPE;

/**
 * @brief The utilization counter data
 */
typedef struct  {
  RSMI_UTILIZATION_COUNTER_TYPE type;   //!< Utilization counter type
  uint64_t value;                       //!< Utilization counter value
} rsmi_utilization_counter_t;

/**
 * @brief Reserved Memory Page Record
 */
typedef struct {
  uint64_t page_address;                  //!< Start address of page
  uint64_t page_size;                     //!< Page size
  rsmi_memory_page_status_t status;       //!< Page "reserved" status
} rsmi_retired_page_record_t;

/**
 * @brief Number of possible power profiles that a system could support
 */
#define RSMI_MAX_NUM_POWER_PROFILES (sizeof(rsmi_bit_field_t) * 8)

/**
 * @brief This structure contains information about which power profiles are
 * supported by the system for a given device, and which power profile is
 * currently active.
 */
typedef struct {
    /**
    * Which profiles are supported by this system
    */
    rsmi_bit_field_t available_profiles;

    /**
    * Which power profile is currently active
    */
    rsmi_power_profile_preset_masks_t current;

    /**
    * How many power profiles are available
    */
    uint32_t num_profiles;
} rsmi_power_profile_status_t;
/// \cond Ignore in docs.
typedef rsmi_power_profile_status_t rsmi_power_profile_status;
/// \endcond

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
    uint64_t frequency[RSMI_MAX_NUM_FREQUENCIES];
} rsmi_frequencies_t;
/// \cond Ignore in docs.
typedef rsmi_frequencies_t rsmi_frequencies;
/// \endcond

/**
 * @brief This structure holds information about the possible PCIe
 * bandwidths. Specifically, the possible transfer rates and their
 * associated numbers of lanes are stored here.
 */
typedef struct {
    /**
     * Transfer rates (T/s) that are possible
     */
    rsmi_frequencies_t transfer_rate;

    /**
     * List of lanes for corresponding transfer rate.
     * Only the first num_supported bandwidths are valid.
     */
    uint32_t lanes[RSMI_MAX_NUM_FREQUENCIES];
} rsmi_pcie_bandwidth_t;

/// \cond Ignore in docs.
typedef rsmi_pcie_bandwidth_t rsmi_pcie_bandwidth;
/// \endcond

/**
 * @brief This structure holds information about the possible activity
 * averages. Specifically, the utilization counters.
 */
typedef struct {
  /* Utilization */
  uint16_t average_gfx_activity;  //!< Average graphics activity
  uint16_t average_umc_activity;  //!< memory controller
  uint16_t average_mm_activity;   //!< UVD or VCN
} rsmi_activity_metric_counter_t;

/**
 * @brief This structure holds version information.
 */
typedef struct {
    uint32_t major;     //!< Major version
    uint32_t minor;     //!< Minor version
    uint32_t patch;     //!< Patch, build  or stepping version
    const char *build;  //!< Build string
} rsmi_version_t;
/// \cond Ignore in docs.
typedef rsmi_version_t rsmi_version;
/// \endcond
/**
 * @brief This structure represents a range (e.g., frequencies or voltages).
 */
typedef struct {
    uint64_t lower_bound;      //!< Lower bound of range
    uint64_t upper_bound;      //!< Upper bound of range
} rsmi_range_t;
/// \cond Ignore in docs.
typedef rsmi_range_t rsmi_range;
/// \endcond

/**
 * @brief This structure represents a point on the frequency-voltage plane.
 */
typedef struct {
    uint64_t frequency;      //!< Frequency coordinate (in Hz)
    uint64_t voltage;        //!< Voltage coordinate (in mV)
} rsmi_od_vddc_point_t;
/// \cond Ignore in docs.
typedef rsmi_od_vddc_point_t rsmi_od_vddc_point;
/// \endcond

/**
 * @brief This structure holds 2 ::rsmi_range_t's, one for frequency and one for
 * voltage. These 2 ranges indicate the range of possible values for the
 * corresponding ::rsmi_od_vddc_point_t.
 */
typedef struct {
    rsmi_range_t freq_range;  //!< The frequency range for this VDDC Curve point
    rsmi_range_t volt_range;  //!< The voltage range for this VDDC Curve point
} rsmi_freq_volt_region_t;
/// \cond Ignore in docs.
typedef rsmi_freq_volt_region_t rsmi_freq_volt_region;
/// \endcond

/**
 * ::RSMI_NUM_VOLTAGE_CURVE_POINTS number of ::rsmi_od_vddc_point_t's
 */
typedef struct {
    /**
     * Array of ::RSMI_NUM_VOLTAGE_CURVE_POINTS ::rsmi_od_vddc_point_t's that
     * make up the voltage frequency curve points.
     */
    rsmi_od_vddc_point_t vc_points[RSMI_NUM_VOLTAGE_CURVE_POINTS];
} rsmi_od_volt_curve_t;
/// \cond Ignore in docs.
typedef rsmi_od_volt_curve_t rsmi_od_volt_curve;
/// \endcond

/**
 * @brief This structure holds the frequency-voltage values for a device.
 */
typedef struct {
  rsmi_range_t curr_sclk_range;          //!< The current SCLK frequency range
  rsmi_range_t curr_mclk_range;          //!< The current MCLK frequency range;
                                         //!< (upper bound only)
  rsmi_range_t sclk_freq_limits;         //!< The range possible of SCLK values
  rsmi_range_t mclk_freq_limits;         //!< The range possible of MCLK values

  /**
   * @brief The current voltage curve
   */
  rsmi_od_volt_curve_t curve;
  uint32_t num_regions;                //!< The number of voltage curve regions
} rsmi_od_volt_freq_data_t;
/// \cond Ignore in docs.
typedef rsmi_od_volt_freq_data_t rsmi_od_volt_freq_data;
/// \endcond


/**
 * @brief The following structures hold the gpu metrics values for a device.
 */

/**
 * @brief Size and version information of metrics data
 */
struct metrics_table_header_t {
  // TODO(amd) Doxygen documents
  // Note: This should match: AMDGpuMetricsHeader_v1_t
  /// \cond Ignore in docs.
  uint16_t      structure_size;
  uint8_t       format_revision;
  uint8_t       content_revision;
  /// \endcond
};
/// \cond Ignore in docs.
typedef struct metrics_table_header_t metrics_table_header_t;
/// \endcond

/**
 * @brief Unit conversion factor for HBM temperatures
 */
#define CENTRIGRADE_TO_MILLI_CENTIGRADE 1000

/**
 * @brief This should match kRSMI_MAX_NUM_HBM_INSTANCES
 */
#define RSMI_NUM_HBM_INSTANCES 4

/**
 * @brief This should match kRSMI_MAX_NUM_VCNS
 */
#define RSMI_MAX_NUM_VCNS 4

/**
 * @brief This should match kRSMI_MAX_JPEG_ENGINES
 */
#define RSMI_MAX_NUM_JPEG_ENGS 32

/**
 * @brief This should match kRSMI_MAX_NUM_JPEG_ENG_V1
 */
#define RSMI_MAX_NUM_JPEG_ENG_V1 40

/**
 * @brief This should match kRSMI_MAX_NUM_CLKS
 */
#define RSMI_MAX_NUM_CLKS 4

/**
 * @brief This should match kRSMI_MAX_NUM_XGMI_LINKS
 */
#define RSMI_MAX_NUM_XGMI_LINKS 8

/**
 * @brief This should match kRSMI_MAX_NUM_GFX_CLKS
 */
#define RSMI_MAX_NUM_GFX_CLKS 8

/**
 * @brief This should match kRSMI_MAX_NUM_XCC;
 * XCC - Accelerated Compute Core, the collection of compute units,
 * ACE (Asynchronous Compute Engines), caches,
 * and global resources organized as one unit.
 *
 * Refer to amd.com documentation for more detail:
 * https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
 */
#define RSMI_MAX_NUM_XCC 8

/**
 * @brief This should match kRSMI_MAX_NUM_XCP;
 * XCP - Accelerated Compute Processor,
 * also referred to as the Graphics Compute Partitions.
 * Each physical gpu could have a maximum of 8 separate partitions
 * associated with each (depending on ASIC support).
 *
 * Refer to amd.com documentation for more detail:
 * https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
 */
#define RSMI_MAX_NUM_XCP 8

/**
 * @brief The following structures hold the gpu statistics for a device.
 */
struct amdgpu_xcp_metrics_t {
  /*
  * v1.6 additions
  */
  /* Utilization Instantaneous (%) */
  uint32_t gfx_busy_inst[RSMI_MAX_NUM_XCC];
  uint16_t jpeg_busy[RSMI_MAX_NUM_JPEG_ENG_V1];
  uint16_t vcn_busy[RSMI_MAX_NUM_VCNS];

  /* Utilization Accumulated (%) */
  uint64_t gfx_busy_acc[RSMI_MAX_NUM_XCC];

  /*
  * v1.7 additions
  */
  /* Total App Clock Counter Accumulated */
  uint64_t gfx_below_host_limit_acc[RSMI_MAX_NUM_XCC];

  /**
   * v1.8 additions
   */
  uint64_t gfx_below_host_limit_ppt_acc[RSMI_MAX_NUM_XCC];
  uint64_t gfx_below_host_limit_thm_acc[RSMI_MAX_NUM_XCC];
  uint64_t gfx_low_utilization_acc[RSMI_MAX_NUM_XCC];
  uint64_t gfx_below_host_limit_total_acc[RSMI_MAX_NUM_XCC];
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

  /*
   * v1.0 Base
   */
  struct metrics_table_header_t common_header;

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
  uint16_t temperature_hbm[RSMI_NUM_HBM_INSTANCES];  // new in v1


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
  uint16_t vcn_activity[RSMI_MAX_NUM_VCNS];  // VCN instances activity percent (encode/decode)

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
  uint64_t xgmi_read_data_acc[RSMI_MAX_NUM_XGMI_LINKS];
  uint64_t xgmi_write_data_acc[RSMI_MAX_NUM_XGMI_LINKS];

  // XGMI accumulated data transfer size(KiloBytes)
  uint16_t current_gfxclks[RSMI_MAX_NUM_GFX_CLKS];
  uint16_t current_socclks[RSMI_MAX_NUM_CLKS];
  uint16_t current_vclk0s[RSMI_MAX_NUM_CLKS];
  uint16_t current_dclk0s[RSMI_MAX_NUM_CLKS];

  /*
   * v1.5 additions
   */
  // JPEG activity percent (encode/decode)
  uint16_t jpeg_activity[RSMI_MAX_NUM_JPEG_ENGS];

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
   * Socket (thermal) -
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
  struct amdgpu_xcp_metrics_t xcp_stats[RSMI_MAX_NUM_XCP];

  /* PCIE other end recovery counter */
  uint32_t pcie_lc_perf_other_end_recovery;

  /*
  * v1.7 additions
  */
  /* VRAM max bandwidth at max memory clock */
  uint64_t vram_max_bandwidth;

  /* XGMI link status(up/down) */
  uint16_t xgmi_link_status[RSMI_MAX_NUM_XGMI_LINKS];

  /// \endcond
} rsmi_gpu_metrics_t;

/**
 * @brief This structure holds error counts.
 */
typedef struct {
    uint64_t correctable_err;            //!< Accumulated correctable errors
    uint64_t uncorrectable_err;          //!< Accumulated uncorrectable errors
} rsmi_error_count_t;

/**
 * @brief This structure contains information specific to a process.
 */
typedef struct {
    uint32_t process_id;      //!< Process ID
    uint32_t pasid;           //!< PASID
    uint64_t vram_usage;      //!< VRAM usage
    uint64_t sdma_usage;      //!< SDMA usage in microseconds
    uint32_t cu_occupancy;    //!< Compute Unit usage in percent
} rsmi_process_info_t;

//! CU occupancy invalidation value for the GFX revisions not providing cu_occupancy debugfs method
#define CU_OCCUPANCY_INVALID 0xFFFFFFFF

/**
 * @brief Opaque handle to function-support object
 */
typedef struct rsmi_func_id_iter_handle * rsmi_func_id_iter_handle_t;

//! Place-holder "variant" for functions that have don't have any variants,
//! but do have monitors or sensors.
#define RSMI_DEFAULT_VARIANT 0xFFFFFFFFFFFFFFFF

/**
 * @brief This union holds the value of an ::rsmi_func_id_iter_handle_t. The
 * value may be a function name, or an ennumerated variant value of types
 * such as ::rsmi_memory_type_t, ::rsmi_temperature_metric_t, etc.
 */
typedef union id {
        uint64_t id;           //!< uint64_t representation of value
        const char *name;      //!< name string (applicable to functions only)
        union {
            /** Used for ::rsmi_memory_type_t variants */
            rsmi_memory_type_t memory_type;
            /** Used for ::rsmi_temperature_metric_t variants */
            rsmi_temperature_metric_t temp_metric;
            /** Used for ::rsmi_event_type_t variants */
            rsmi_event_type_t evnt_type;
            /** Used for ::rsmi_event_group_t variants */
            rsmi_event_group_t evnt_group;
            /** Used for ::rsmi_clk_type_t variants */
            rsmi_clk_type_t clk_type;
            /** Used for ::rsmi_fw_block_t variants */
            rsmi_fw_block_t fw_block;
            /** Used for ::rsmi_gpu_block_t variants */
            rsmi_gpu_block_t gpu_block_type;
        };
} rsmi_func_id_value_t;

/**
 * @struct rsmi_device_identifiers_t
 * @brief Structure to hold various identifiers for a GPU device.
 *
 * @details This structure contains fields that uniquely identify a GPU device,
 * including its card index, DRM render minor, PCI Bus/Device/Function ID (BDFID),
 * KFD GPU ID, partition ID, and SMI device ID.
 */
typedef struct {
  //!< The card index of the device.
  uint32_t card_index;
  //!< The DRM render minor number of the device.
  uint32_t drm_render_minor;

  //!< The PCI Bus/Device/Function identifier (BDFID) of the device.
  uint64_t bdfid;

  //!< The KFD (Kernel Fusion Driver) GPU ID of the device.
  uint64_t kfd_gpu_id;

  //!< The partition ID of the device.
  uint32_t partition_id;

  //!< The SMI (System Management Interface) device ID.
  uint32_t smi_device_id;

  uint32_t reserved[10];
} rsmi_device_identifiers_t;

/*****************************************************************************/
/** @defgroup InitShutAdmin Initialization and Shutdown
 *  These functions are used for initialization of ROCm SMI and clean up when
 *  done.
 *  @{
 */
/**
 *  @brief Initialize ROCm SMI.
 *
 *  @details When called, this initializes internal data structures,
 *  including those corresponding to sources of information that SMI provides.
 *
 *  @param[in] init_flags Bit flags that tell SMI how to initialze. Values of
 *  ::rsmi_init_flags_t may be OR'd together and passed through @p init_flags
 *  to modify how RSMI initializes.
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 */
rsmi_status_t rsmi_init(uint64_t init_flags);

/**
 *  @brief Shutdown ROCm SMI.
 *
 *  @details Do any necessary clean up.
 */
rsmi_status_t rsmi_shut_down(void);

/** @} */  // end of InitShut

/*****************************************************************************/
/** @defgroup IDQuer Identifier Queries
 *  These functions provide identification information.
 *  @{
 */
/**
 *  @brief Get the number of devices that have monitor information.
 *
 *  @details The number of devices which have monitors is returned. Monitors
 *  are referenced by the index which can be between 0 and @p num_devices - 1.
 *
 *  @param[inout] num_devices Caller provided pointer to uint32_t. Upon
 *  successful call, the value num_devices will contain the number of monitor
 *  devices.
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 */
rsmi_status_t rsmi_num_monitor_devices(uint32_t *num_devices);

/**
 *  @brief Get the device id associated with the device with provided device
 *  index.
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint32_t @p id,
 *  this function will write the device id value to the uint64_t pointed to by
 *  @p id. This ID is an identification of the type of device, so calling this
 *  function for different devices will give the same value if they are kind
 *  of device. Consequently, this function should not be used to distinguish
 *  one device from another. rsmi_dev_pci_id_get() should be used to get a
 *  unique identifier.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] id a pointer to uint64_t to which the device id will be
 *  written
 * If this parameter is nullptr, this function will return
 * ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 * arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 * provided arguments.
 *
 * @retval ::RSMI_STATUS_SUCCESS call was successful
 * @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 * support this function with the given arguments
 * @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_id_get(uint32_t dv_ind, uint16_t *id);

/**
 *  @brief Get the device revision associated with the device
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint32_t to
 *  which the revision will be written
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] revision a pointer to uint32_t to which the device revision
 *  will be written
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t rsmi_dev_revision_get(uint32_t dv_ind, uint16_t *revision);

/**
 *  @brief Get the SKU for a desired device associated with the device with
 *  provided device index.
 *
 *  @details Given a device index @p dv_ind and a pointer to a char @p sku,
 *  this function will attempt to obtain the SKU from the Product Information
 *  FRU chip, present on server ASICs. It will write the sku value to the
 *  char array pointed to by @p sku.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] sku a pointer to char to which the sku will be written
 *
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_sku_get(uint32_t dv_ind, uint16_t *sku);

/**
 *  @brief Get the device vendor id associated with the device with provided
 *  device index.
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint32_t @p id,
 *  this function will write the device vendor id value to the uint64_t pointed
 *  to by @p id.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] id a pointer to uint64_t to which the device vendor id will
 *  be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_vendor_id_get(uint32_t dv_ind, uint16_t *id);

/**
 *  @brief Get the name string of a gpu device.
 *
 *  @details Given a device index @p dv_ind, a pointer to a caller provided
 *  char buffer @p name, and a length of this buffer @p len, this function
 *  will write the name of the device (up to @p len characters) to the buffer
 *  @p name.
 *
 *  If the integer ID associated with the device is not found in one of the
 *  system files containing device name information (e.g.
 *  /usr/share/misc/pci.ids), then this function will return the hex device ID
 *  as a string. Updating the system name files can be accompplished with
 *  "sudo update-pciids".
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] name a pointer to a caller provided char buffer to which the
 *  name will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[in] len the length of the caller provided buffer @p name.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire name. In this case, only @p len bytes will
 *  be written.
 *
 */
rsmi_status_t rsmi_dev_name_get(uint32_t dv_ind, char *name, size_t len);

/**
 *  @brief Get the brand string of a gpu device.
 *
 *  @details Given a device index @p dv_ind, a pointer to a caller provided
 *  char buffer @p brand, and a length of this buffer @p len, this function
 *  will write the brand of the device (up to @p len characters) to the buffer
 *  @p brand.
 *
 *  If the sku associated with the device is not found as one of the values
 *  contained within rsmi_dev_brand_get, then this function will return the
 *  device marketing name as a string instead of the brand name.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] brand a pointer to a caller provided char buffer to which the
 *  brand will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[in] len the length of the caller provided buffer @p brand.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire name. In this case, only @p len bytes will
 *  be written.
 *
 */
rsmi_status_t rsmi_dev_brand_get(uint32_t dv_ind, char *brand, uint32_t len);

/**
 *  @brief Get the name string for a give vendor ID
 *
 *  @details Given a device index @p dv_ind, a pointer to a caller provided
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
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] name a pointer to a caller provided char buffer to which the
 *  name will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[in] len the length of the caller provided buffer @p name.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire name. In this case, only @p len bytes will
 *  be written.
 *
 */
rsmi_status_t rsmi_dev_vendor_name_get(uint32_t dv_ind, char *name, size_t len);


/**
 *  @brief Get the device's market name
 *
 *  @details Given a device index @p dv_ind, a pointer to a caller provided
 *  char buffer @p market_name, and a length of this buffer @p len, this function will
 *  write the name of the market name (up to @p len characters) buffer @p market_name.
 *
 *  @param[inout] market_name a pointer to a caller provided char buffer to which the
 *  market name will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_DRM_ERROR if a DRM error occurs
 *
 *  @param[in] len the length of the caller provided buffer @p name.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_DRM_ERROR if a DRM error occurs
 *  @retval ::RSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire name. In this case, only @p len bytes will
 *  be written.
 *
 */
rsmi_status_t rsmi_dev_market_name_get(uint32_t dv_ind, char *market_name, uint32_t len);

/**
 *  @brief Get the vram vendor string of a gpu device.
 *
 *  @details Given a device index @p dv_ind, a pointer to a caller provided
 *  char buffer @p brand, and a length of this buffer @p len, this function
 *  will write the vram vendor of the device (up to @p len characters) to the
 *  buffer @p brand.
 *
 *  If the vram vendor for the device is not found as one of the values
 *  contained within rsmi_dev_vram_vendor_get, then this function will return
 *  the string 'unknown' instead of the vram vendor.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] brand a pointer to a caller provided char buffer to which the
 *  vram vendor will be written
 *
 *  @param[in] len the length of the caller provided buffer @p brand.
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t rsmi_dev_vram_vendor_get(uint32_t dv_ind, char *brand,
                                                                uint32_t len);

/**
 * @brief Get the serial number string for a device
 *
 * @details Given a device index @p dv_ind, a pointer to a buffer of chars
 * @p serial_num, and the length of the provided buffer @p len, this function
 * will write the serial number string (up to @p len characters) to the buffer
 * pointed to by @p serial_num.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] serial_num a pointer to caller-provided memory to which the
 *  serial number will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.

 *  @param[in] len the length of the caller provided buffer @p serial_num.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire name. In this case, only @p len bytes will
 *  be written.
 *
 */
rsmi_status_t rsmi_dev_serial_number_get(uint32_t dv_ind,
                                              char *serial_num, uint32_t len);
/**
 *  @brief Get the subsystem device id associated with the device with
 *  provided device index.
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint32_t @p id,
 *  this function will write the subsystem device id value to the uint64_t
 *  pointed to by @p id.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] id a pointer to uint64_t to which the subsystem device id
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_subsystem_id_get(uint32_t dv_ind, uint16_t *id);

/**
 *  @brief Get the name string for the device subsytem
 *
 *  @details Given a device index @p dv_ind, a pointer to a caller provided
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
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] name a pointer to a caller provided char buffer to which the
 *  name will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.

 *  @param[in] len the length of the caller provided buffer @p name.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire name. In this case, only @p len bytes will
 *  be written.
 *
 */
rsmi_status_t
rsmi_dev_subsystem_name_get(uint32_t dv_ind, char *name, size_t len);

/**
 *  @brief Get the drm minor number associated with this device
 *
 *  @details Given a device index @p dv_ind, find its render device file
 *  /dev/dri/renderDN where N corresponds to its minor number.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] minor a pointer to a uint32_t into which minor number will
 *  be copied
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *  @retval ::RSMI_STATUS_INIT_ERROR if failed to get minor number during
 *  initialization.
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_dev_drm_render_minor_get(uint32_t dv_ind, uint32_t *minor);

/**
 *  @brief Get the device subsystem vendor id associated with the device with
 *  provided device index.
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint32_t @p id,
 *  this function will write the device subsystem vendor id value to the
 *  uint64_t pointed to by @p id.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] id a pointer to uint64_t to which the device subsystem vendor
 *  id will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t rsmi_dev_subsystem_vendor_id_get(uint32_t dv_ind, uint16_t *id);

/**
 *  @brief Get Unique ID
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint64_t @p
 *  id, this function will write the unique ID of the GPU pointed to @p
 *  id.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] id a pointer to uint64_t to which the unique ID of the GPU
 *  is written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t rsmi_dev_unique_id_get(uint32_t dv_ind, uint64_t *id);

/**
 *  @brief Get the XGMI physical id associated with the device
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint32_t to
 *  which the XGMI physical id will be written
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] id a pointer to uint32_t to which the XGMI physical id
 *  will be written
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t rsmi_dev_xgmi_physical_id_get(uint32_t dv_ind, uint16_t *id);

/**
 *  @brief Get the GUID, also known as the GPU device id,
 *  associated with the provided device index indicated by KFD.
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint64_t
 *  @p guid, this function will write the KFD GPU id value to the
 *  uint64_t pointed to by @p guid.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] guid a pointer to uint64_t to which the KFD gpu id will be
 *  written. If the @p guid parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS. If the GPU ID is not supported with
 *  the device index queried, gpu_id will return MAX UINT64 value an
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED as a response.
 *
 * @retval ::RSMI_STATUS_SUCCESS call was successful
 * @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 * support this function with the given arguments
 * @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_guid_get(uint32_t dv_ind, uint64_t *guid);

/**
 *  @brief Get the node id associated with the provided device index
 *  indicated by KFD.
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint32_t
 *  @p node_id, this function will write the KFD node id value to the
 *  uint32_t pointed to by @p node_id.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] node_id a pointer to uint64_t to which the KFD gpu id will be
 *  written. If the @p node_id parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS. If @p node_id is not supported with
 *  the device index queried, @p node_id will return MAX UINT64 value as an
 *  argument and ::RSMI_STATUS_NOT_SUPPORTED as a response.
 *
 * @retval ::RSMI_STATUS_SUCCESS call was successful
 * @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 * support this function with the given arguments
 * @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_node_id_get(uint32_t dv_ind, uint32_t *node_id);

/**
 * @brief Retrieves the device identifiers for a specific GPU device.
 *
 * @details This function retrieves various identifiers for a GPU device, such as
 * the card index, DRM render minor, BDFID, KFD GPU ID, partition ID, and SMI device ID.
 * The identifiers are written to the provided `rsmi_device_identifiers_t` structure.
 *
 * @param[in] dv_ind a device index.
 *
 * @param[out] identifiers A pointer to a structure of type `rsmi_device_identifiers_t`
 *                         where the device identifiers will be stored. The structure
 *                         contains fields such as:
 *                         - `card_index`: The card index of the device.
 *                         - `drm_render_minor`: The DRM render minor number.
 *                         - `bdfid`: The Bus/Device/Function PCI identifier.
 *                         - `kfd_gpu_id`: The KFD GPU ID.
 *                         - `partition_id`: The partition ID of the device.
 *                         - `smi_device_id`: The SMI device ID.
 *
 * @retval ::RSMI_STATUS_SUCCESS The call was successful, and the device identifiers were retrieved.
 * @retval ::RSMI_STATUS_NOT_SUPPORTED The installed software or hardware does not support this function
 *                                     with the given arguments.
 * @retval ::RSMI_STATUS_INVALID_ARGS The provided arguments are invalid.
 *
 * @note Ensure that the `identifiers` pointer is valid and points to a properly allocated structure
 *       before calling this function.
 */
rsmi_status_t rsmi_dev_device_identifiers_get(uint32_t dv_ind,
                                              rsmi_device_identifiers_t *identifiers);

/** @} */  // end of IDQuer

/*****************************************************************************/
/** @defgroup PCIeQuer PCIe Queries
 *  These functions provide information about PCIe.
 *  @{
 */
/**
 *  @brief Get the list of possible PCIe bandwidths that are available.
 *
 *  @details Given a device index @p dv_ind and a pointer to a to an
 *  ::rsmi_pcie_bandwidth_t structure @p bandwidth, this function will fill in
 *  @p bandwidth with the possible T/s values and associated number of lanes,
 *  and indication of the current selection.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] bandwidth a pointer to a caller provided
 *  ::rsmi_pcie_bandwidth_t structure to which the frequency information will be
 *  written
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *  @retval ::RSMI_STATUS_UNEXPECTED_DATA Data read or provided was not as
 *  expected
 *
 */
rsmi_status_t
rsmi_dev_pci_bandwidth_get(uint32_t dv_ind, rsmi_pcie_bandwidth_t *bandwidth);

/**
 *  @brief Get the unique PCI device identifier associated for a device
 *
 *  @details Give a device index @p dv_ind and a pointer to a uint64_t @p
 *  bdfid, this function will write the Bus/Device/Function PCI identifier
 *  (BDFID) associated with device @p dv_ind to the value pointed to by
 *  @p bdfid.
 *
 *  The format of @p bdfid will be as follows:
 *
 *      BDFID = ((DOMAIN & 0xFFFFFFFF) << 32) | ((Partition & 0xF) << 28)
 *              | ((BUS & 0xFF) << 8) | ((DEVICE & 0x1F) <<3 )
 *              | (FUNCTION & 0x7)
 *
 *  \code{.unparsed}
 *  | Name         | Field   | KFD property       KFD -> PCIe ID (uint64_t)
 *  -------------- | ------- | ---------------- | ---------------------------- |
 *  | Domain       | [63:32] | "domain"         | (DOMAIN & 0xFFFFFFFF) << 32  |
 *  | Partition id | [31:28] | "location id"    | (LOCATION & 0xF0000000)      |
 *  | Reserved     | [27:16] | "location id"    | N/A                          |
 *  | Bus          | [15: 8] | "location id"    | (LOCATION & 0xFF00)          |
 *  | Device       | [ 7: 3] | "location id"    | (LOCATION & 0xF8)            |
 *  | Function     | [ 2: 0] | "location id"    | (LOCATION & 0x7)             |
 *  \endcode
 *
 *  Note: In some devices, the partition ID may be stored in the function bits
 *  BDFID[2:0] instead of BDFID[31:28].
 *
 *  Note: For MI series devices, the function bits are only used to store the
 *  partition ID, but this modified BDF is internal to the ROCm stack.
 *  To the OS, partitions share the same BDF as the unpartitioned device and
 *  have function bits = 0, which can be verified through lspci.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] bdfid a pointer to uint64_t to which the device bdfid value
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t rsmi_dev_pci_id_get(uint32_t dv_ind, uint64_t *bdfid);

/**
 *  @brief Get the NUMA node associated with a device
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint32_t @p
 *  numa_node, this function will retrieve the NUMA node value associated
 *  with device @p dv_ind and store the value at location pointed to by
 *  @p numa_node.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] numa_node pointer to location where NUMA node value will
 *  be written.
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t rsmi_topo_numa_affinity_get(uint32_t dv_ind, int32_t *numa_node);

/**
 *  @brief Get PCIe traffic information
 *
 *  @details Give a device index @p dv_ind and pointers to a uint64_t's, @p
 *  sent, @p received and @p max_pkt_sz, this function will write the number
 *  of bytes sent and received in 1 second to @p sent and @p received,
 *  respectively. The maximum possible packet size will be written to
 *  @p max_pkt_sz.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] sent a pointer to uint64_t to which the number of bytes sent
 *  will be written in 1 second. If pointer is NULL, it will be ignored.
 *
 *  @param[inout] received a pointer to uint64_t to which the number of bytes
 *  received will be written. If pointer is NULL, it will be ignored.
 *
 *  @param[inout] max_pkt_sz a pointer to uint64_t to which the maximum packet
 *  size will be written. If pointer is NULL, it will be ignored.
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 */
rsmi_status_t rsmi_dev_pci_throughput_get(uint32_t dv_ind, uint64_t *sent,
                                    uint64_t *received, uint64_t *max_pkt_sz);

/**
 *  @brief Get PCIe replay counter
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint64_t @p
 *  counter, this function will write the sum of the number of NAK's received
 *  by the GPU and the NAK's generated by the GPU to memory pointed to by @p
 *  counter.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] counter a pointer to uint64_t to which the sum of the NAK's
 *  received and generated by the GPU is written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t rsmi_dev_pci_replay_counter_get(uint32_t dv_ind,
                                                           uint64_t *counter);

/** @} */  // end of PCIeQuer
/*****************************************************************************/
/** @defgroup PCIeCont PCIe Control
 *  These functions provide some control over PCIe.
 *  @{
 */

/**
 *  @brief Control the set of allowed PCIe bandwidths that can be used.
 *
 *  @details Given a device index @p dv_ind and a 64 bit bitmask @p bw_bitmask,
 *  this function will limit the set of allowable bandwidths. If a bit in @p
 *  bw_bitmask has a value of 1, then the frequency (as ordered in an
 *  ::rsmi_frequencies_t returned by ::rsmi_dev_gpu_clk_freq_get()) corresponding
 *  to that bit index will be allowed.
 *
 *  This function will change the performance level to
 *  ::RSMI_DEV_PERF_LEVEL_MANUAL in order to modify the set of allowable
 *  band_widths. Caller will need to set to ::RSMI_DEV_PERF_LEVEL_AUTO in order
 *  to get back to default state.
 *
 *  All bits with indices greater than or equal to the value of the
 *  ::rsmi_frequencies_t::num_supported field of ::rsmi_pcie_bandwidth_t will be
 *  ignored.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] bw_bitmask A bitmask indicating the indices of the
 *  bandwidths that are to be enabled (1) and disabled (0). Only the lowest
 *  ::rsmi_frequencies_t::num_supported (of ::rsmi_pcie_bandwidth_t) bits of
 *  this mask are relevant.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_PERMISSION function requires root access
 *
 */
rsmi_status_t rsmi_dev_pci_bandwidth_set(uint32_t dv_ind, uint64_t bw_bitmask);

/** @} */  // end of PCIeCont

/*****************************************************************************/
/** @defgroup PowerQuer Power Queries
 *  These functions provide information about power usage.
 *  @{
 */
/**
 *  @brief Get the average power consumption of the device with provided
 *  device index.
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint64_t
 *  @p power, this function will write the current average power consumption
 *  (in microwatts) to the uint64_t pointed to by @p power.
 *
 *  @deprecated ::rsmi_dev_power_get() is preferred due to providing
 *  backwards compatibility, which looks at both average and current power
 *  values. Whereas ::rsmi_dev_power_ave_get only looks for average power
 *  consumption. Newer ASICs will support current power only.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[inout] power a pointer to uint64_t to which the average power
 *  consumption will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t
rsmi_dev_power_ave_get(uint32_t dv_ind, uint32_t sensor_ind, uint64_t *power);

/**
 *  @brief Get the current socket power (also known as instant
 *  power) of the device index provided.
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint64_t
 *  @p socket_power, this function will write the current socket power
 *  (in microwatts) to the uint64_t pointed to by @p socket_power.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] socket_power a pointer to uint64_t to which the current
 *  socket power will be written to. If this parameter is nullptr,
 *  this function will return ::RSMI_STATUS_INVALID_ARGS if the function is
 *  supported with the provided, arguments and ::RSMI_STATUS_NOT_SUPPORTED
 *  if it is not supported with the provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t
rsmi_dev_current_socket_power_get(uint32_t dv_ind, uint64_t *socket_power);

/**
 *  @brief A generic get which attempts to retieve current socket power
 *  (also known as instant power) of the device index provided, if not
 *  supported tries to get average power consumed by device. Current
 *  socket power is typically supported by newer devices, whereas average
 *  power is generally reported on older devices. This function
 *  aims to provide backwards compatability depending on device support.
 *
 *  @details Given a device index @p dv_ind, a pointer to a uint64_t
 *  @p power, and @p type this function will write the current socket or
 *  average power (in microwatts) to the uint64_t pointed to by @p power and
 *  a pointer to its @p type RSMI_POWER_TYPE read.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] power a pointer to uint64_t to which the current or average
 *  power will be written to. If this parameter is nullptr,
 *  this function will return ::RSMI_STATUS_INVALID_ARGS if the function is
 *  supported with the provided, arguments and ::RSMI_STATUS_NOT_SUPPORTED
 *  if it is not supported with the provided arguments.
 *
 *  @param[inout] type a pointer to RSMI_POWER_TYPE object. Returns the type
 *  of power retrieved from the device. Current power is ::RSMI_CURRENT_POWER
 *  and average power is ::RSMI_AVERAGE_POWER. If an error occurs,
 *  returns an invalid power type ::RSMI_INVALID_POWER - example device
 *  neither supports average power or current power.
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t rsmi_dev_power_get(uint32_t dv_ind, uint64_t *power,
                                 RSMI_POWER_TYPE *type);

/**
 *  @brief Get the energy accumulator counter of the device with provided
 *  device index.
 *
 *  @details Given a device index @p dv_ind, a pointer to a uint64_t
 *  @p power, and a pointer to a uint64_t @p timestamp, this function will write
 *  amount of energy consumed to the uint64_t pointed to by @p power,
 *  and the timestamp to the uint64_t pointed to by @p timestamp.
 *  The rsmi_dev_power_ave_get() is an average of a short time. This function
 *  accumulates all energy consumed.
 *
 *  @param[in] dv_ind a device index
 *  @param[inout] counter_resolution resolution of the counter @p power in
 *  micro Joules
 *
 *  @param[inout] power a pointer to uint64_t to which the energy
 *  counter will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[inout] timestamp a pointer to uint64_t to which the timestamp
 *  will be written. Resolution: 1 ns.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t
rsmi_dev_energy_count_get(uint32_t dv_ind, uint64_t *power,
                          float *counter_resolution, uint64_t *timestamp);

/**
 *  @brief Get the cap on power which, when reached, causes the system to take
 *  action to reduce power.
 *
 *  @details When power use rises above the value @p power, the system will
 *  take action to reduce power use. The power level returned through
 *  @p power will be in microWatts.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[inout] cap a pointer to a uint64_t that indicates the power cap,
 *  in microwatts
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t
rsmi_dev_power_cap_get(uint32_t dv_ind, uint32_t sensor_ind, uint64_t *cap);

/**
 *  @brief Get the default power cap for the device specified by @p dv_ind.
 *
 *  @details The maximum power cap be temporarily changed by the user. However,
 *  this function always returns the default reset power cap. The power level
 *  returned through @p power will be in microWatts.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] default_cap a pointer to a uint64_t that indicates the default
 *  power cap, in microwatts
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t
rsmi_dev_power_cap_default_get(uint32_t dv_ind, uint64_t *default_cap);

/**
 *  @brief Get the range of valid values for the power cap
 *
 *  @details This function will return the maximum possible valid power cap
 *  @p max and the minimum possible valid power cap @p min
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[inout] max a pointer to a uint64_t that indicates the maximum
 *  possible power cap, in microwatts
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[inout] min a pointer to a uint64_t that indicates the minimum
 *  possible power cap, in microwatts
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_dev_power_cap_range_get(uint32_t dv_ind, uint32_t sensor_ind,
                                                uint64_t *max, uint64_t *min);

/** @} */  // end of PowerQuer

/*****************************************************************************/
/** @defgroup PowerCont Power Control
 *  These functions provide ways to control power usage.
 *  @{
 */
/**
 *  @brief Set the power cap value
 *
 *  @details This function will set the power cap to the provided value @p cap.
 *  @p cap must be between the minimum and maximum power cap values set by the
 *  system, which can be obtained from ::rsmi_dev_power_cap_range_get.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[in] cap a uint64_t that indicates the desired power cap, in
 *  microwatts
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_PERMISSION function requires root access
 *
 */
rsmi_status_t
rsmi_dev_power_cap_set(uint32_t dv_ind, uint32_t sensor_ind, uint64_t cap);

/**
 *  @brief Set the power profile
 *
 *  @details Given a device index @p dv_ind and a @p profile, this function will
 *  attempt to set the current profile to the provided profile. The provided
 *  profile must be one of the currently supported profiles, as indicated by a
 *  call to ::rsmi_dev_power_profile_presets_get()
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] reserved Not currently used. Set to 0.
 *
 *  @param[in] profile a ::rsmi_power_profile_preset_masks_t that hold the mask
 *  of the desired new power profile
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *  @retval ::RSMI_STATUS_PERMISSION function requires root access
 *
 */
rsmi_status_t
rsmi_dev_power_profile_set(uint32_t dv_ind, uint32_t reserved,
                                   rsmi_power_profile_preset_masks_t profile);
/** @} */  // end of PowerCont
/*****************************************************************************/



/*****************************************************************************/
/** @defgroup MemQuer Memory Queries
 *  These functions provide information about memory systems.
 *  @{
 */

/**
 *  @brief Get the total amount of memory that exists
 *
 *  @details Given a device index @p dv_ind, a type of memory @p mem_type, and
 *  a pointer to a uint64_t @p total, this function will write the total amount
 *  of @p mem_type memory that exists to the location pointed to by @p total.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] mem_type The type of memory for which the total amount will be
 *  found
 *
 *  @param[inout] total a pointer to uint64_t to which the total amount of
 *  memory will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_dev_memory_total_get(uint32_t dv_ind, rsmi_memory_type_t mem_type,
                                                             uint64_t *total);

/**
 *  @brief Get the current memory usage
 *
 *  @details Given a device index @p dv_ind, a type of memory @p mem_type, and
 *  a pointer to a uint64_t @p usage, this function will write the amount of
 *  @p mem_type memory that that is currently being used to the location
 *  pointed to by @p used.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] mem_type The type of memory for which the amount being used will
 *  be found
 *
 *  @param[inout] used a pointer to uint64_t to which the amount of memory
 *  currently being used will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_dev_memory_usage_get(uint32_t dv_ind, rsmi_memory_type_t mem_type,
                                                              uint64_t *used);

/**
 *  @brief Get percentage of time any device memory is being used
 *
 *  @details Given a device index @p dv_ind, this function returns the
 *  percentage of time that any device memory is being used for the specified
 *  device.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] busy_percent a pointer to the uint32_t to which the busy
 *  percent will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_dev_memory_busy_percent_get(uint32_t dv_ind, uint32_t *busy_percent);

/**
 *  @brief Get information about reserved ("retired") memory pages
 *
 *  @details Given a device index @p dv_ind, this function returns retired page
 *  information @p records corresponding to the device with the provided device
 *  index @p dv_ind. The number of retired page records is returned through @p
 *  num_pages. @p records may be NULL on input. In this case, the number of
 *  records available for retrieval will be returned through @p num_pages.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] num_pages a pointer to a uint32. As input, the value passed
 *  through this parameter is the number of ::rsmi_retired_page_record_t's that
 *  may be safely written to the memory pointed to by @p records. This is the
 *  limit on how many records will be written to @p records. On return, @p
 *  num_pages will contain the number of records written to @p records, or the
 *  number of records that could have been written if enough memory had been
 *  provided.
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[inout] records A pointer to a block of memory to which the
 *  ::rsmi_retired_page_record_t values will be written. This value may be NULL.
 *  In this case, this function can be used to query how many records are
 *  available to read.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_INSUFFICIENT_SIZE is returned if more records were available
 *  than allowed by the provided, allocated memory.
 */
rsmi_status_t
rsmi_dev_memory_reserved_pages_get(uint32_t dv_ind, uint32_t *num_pages,
                                         rsmi_retired_page_record_t *records);
/** @} */  // end of MemQuer

/** @defgroup PhysQuer Physical State Queries
 *  These functions provide information about the physical characteristics of
 *  the device.
 *  @{
 */
/**
 *  @brief Get the fan speed in RPMs of the device with the specified device
 *  index and 0-based sensor index.
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint32_t
 *  @p speed, this function will write the current fan speed in RPMs to the
 *  uint32_t pointed to by @p speed
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[inout] speed a pointer to uint32_t to which the speed will be
 *  written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_fan_rpms_get(uint32_t dv_ind, uint32_t sensor_ind,
                                                              int64_t *speed);

/**
 *  @brief Get the fan speed for the specified device as a value relative to
 *  ::RSMI_MAX_FAN_SPEED
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint32_t
 *  @p speed, this function will write the current fan speed (a value
 *  between 0 and the maximum fan speed, ::RSMI_MAX_FAN_SPEED) to the uint32_t
 *  pointed to by @p speed
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[inout] speed a pointer to uint32_t to which the speed will be
 *  written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_fan_speed_get(uint32_t dv_ind,
                                        uint32_t sensor_ind, int64_t *speed);

/**
 *  @brief Get the max. fan speed of the device with provided device index.
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint32_t
 *  @p max_speed, this function will write the maximum fan speed possible to
 *  the uint32_t pointed to by @p max_speed
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[inout] max_speed a pointer to uint32_t to which the maximum speed
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_fan_speed_max_get(uint32_t dv_ind,
                                    uint32_t sensor_ind, uint64_t *max_speed);

/**
 *  @brief Get the temperature metric value for the specified metric, from the
 *  specified temperature sensor on the specified device.
 *
 *  @details Given a device index @p dv_ind, a sensor type @p sensor_type, a
 *  ::rsmi_temperature_metric_t @p metric and a pointer to an int64_t @p
 *  temperature, this function will write the value of the metric indicated by
 *  @p metric and @p sensor_type to the memory location @p temperature.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] sensor_type part of device from which temperature should be
 *  obtained. This should come from the enum ::rsmi_temperature_type_t
 *
 *  @param[in] metric enum indicated which temperature value should be
 *  retrieved
 *
 *  @param[inout] temperature a pointer to int64_t to which the temperature
 *  will be written, in millidegrees Celcius.
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_temp_metric_get(uint32_t dv_ind, uint32_t sensor_type,
                      rsmi_temperature_metric_t metric, int64_t *temperature);

/**
 *  @brief Get the voltage metric value for the specified metric, from the
 *  specified voltage sensor on the specified device.
 *
 *  @details Given a device index @p dv_ind, a sensor type @p sensor_type, a
 *  ::rsmi_voltage_metric_t @p metric and a pointer to an int64_t @p
 *  voltage, this function will write the value of the metric indicated by
 *  @p metric and @p sensor_type to the memory location @p voltage.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] sensor_type part of device from which voltage should be
 *  obtained. This should come from the enum ::rsmi_voltage_type_t
 *
 *  @param[in] metric enum indicated which voltage value should be
 *  retrieved
 *
 *  @param[inout] voltage a pointer to int64_t to which the voltage
 *  will be written, in millivolts.
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_volt_metric_get(uint32_t dv_ind,
                               rsmi_voltage_type_t sensor_type,
                              rsmi_voltage_metric_t metric, int64_t *voltage);
/** @} */  // end of PhysQuer

/*****************************************************************************/
/** @defgroup PhysCont Physical State Control
 *  These functions provide control over the physical state of a device.
 *  @{
 */
/**
 *  @brief Reset the fan to automatic driver control
 *
 *  @details This function returns control of the fan to the system
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments

 */
rsmi_status_t rsmi_dev_fan_reset(uint32_t dv_ind, uint32_t sensor_ind);

/**
 *  @brief Set the fan speed for the specified device with the provided speed,
 *  in RPMs.
 *
 *  @details Given a device index @p dv_ind and a integer value indicating
 *  speed @p speed, this function will attempt to set the fan speed to @p speed.
 *  An error will be returned if the specified speed is outside the allowable
 *  range for the device. The maximum value is 255 and the minimum is 0.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[in] speed the speed to which the function will attempt to set the fan
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_PERMISSION function requires root access
 *
 */
rsmi_status_t rsmi_dev_fan_speed_set(uint32_t dv_ind, uint32_t sensor_ind,
                                                              uint64_t speed);

/** @} */  // end of PhysCont
/*****************************************************************************/
/** @defgroup PerfQuer Clock, Power and Performance Queries
 *  These functions provide information about clock frequencies and
 *  performance.
 *  @{
 */

/**
 *  @brief Get percentage of time device is busy doing any processing
 *
 *  @details Given a device index @p dv_ind, this function returns the
 *  percentage of time that the specified device is busy. The device is
 *  considered busy if any one or more of its sub-blocks are working, and idle
 *  if none of the sub-blocks are working.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] busy_percent a pointer to the uint32_t to which the busy
 *  percent will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_dev_busy_percent_get(uint32_t dv_ind, uint32_t *busy_percent);

/**
 *  @brief Get coarse grain utilization counter of the specified device
 *
 *  @details Given a device index @p dv_ind, the array of the utilization counters,
 *  the size of the array, this function returns the coarse grain utilization counters
 *  and timestamp.
 *  The counter is the accumulated percentages. Every milliseconds the firmware calculates
 *  % busy count and then accumulates that value in the counter. This provides minimally
 *  invasive coarse grain GPU usage information.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] utilization_counters Multiple utilization counters can be retreived with a single
 *  call. The caller must allocate enough space to the utilization_counters array. The caller also
 *  needs to set valid RSMI_UTILIZATION_COUNTER_TYPE type for each element of the array.
 *  ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the provided arguments.
 *
 *  If the function reutrns RSMI_STATUS_SUCCESS, the counter will be set in the value field of
 *  the rsmi_utilization_counter_t.
 *
 *  @param[in] count The size of utilization_counters array.
 *
 *  @param[inout] timestamp The timestamp when the counter is retreived. Resolution: 1 ns.
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_utilization_count_get(uint32_t dv_ind,
                rsmi_utilization_counter_t utilization_counters[],
                uint32_t count,
                uint64_t *timestamp);

/**
 *  @brief Get activity metric average utilization counter of the specified device
 *
 *  @details Given a device index @p dv_ind, the activity metric type,
 *  this function returns the requested utilization counters
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] activity_metric_type a metric type
 *
 *  @param[inout] activity_metric_counter Multiple utilization counters can be retrieved with a single
 *  call. The caller must allocate enough space to the rsmi_activity_metric_counter_t structure.
 *
 *  If the function returns RSMI_STATUS_SUCCESS, the requested type will be set in the corresponding
 *  field of the counter will be set in the value field of
 *  the activity_metric_counter_t.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_dev_activity_metric_get(uint32_t dv_ind,
                             rsmi_activity_metric_t activity_metric_type,
                             rsmi_activity_metric_counter_t* activity_metric_counter);

/**
 *  @brief Get activity metric bandwidth average utilization counter of the specified device
 *
 *  @details Given a device index @p dv_ind, the activity metric type,
 *  this function returns the requested utilization counters
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] avg_activity average bandwidth utilization counters can be retrieved
 *
 *  If the function returns RSMI_STATUS_SUCCESS, the requested type will be set in the corresponding
 *  field of the counter will be set in the value field of
 *  the activity_metric_counter_t.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_dev_activity_avg_mm_get(uint32_t dv_ind, uint16_t* avg_activity);

/**
 *  @brief Get the performance level of the device with provided
 *  device index.
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint32_t @p
 *  perf, this function will write the ::rsmi_dev_perf_level_t to the uint32_t
 *  pointed to by @p perf
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] perf a pointer to ::rsmi_dev_perf_level_t to which the
 *  performance level will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_perf_level_get(uint32_t dv_ind,
                                                 rsmi_dev_perf_level_t *perf);

/**
 *  @brief Enter performance determinism mode with provided device index.
 *
 *  @details Given a device index @p dv_ind and @p clkvalue this function
 *  will enable performance determinism mode, which enforces a GFXCLK frequency
 *  SoftMax limit per GPU set by the user. This prevents the GFXCLK PLL from
 *  stretching when running the same workload on different GPUS, making
 *  performance variation minimal. This call will result in the performance
 *  level ::rsmi_dev_perf_level_t of the device being
 *  ::RSMI_DEV_PERF_LEVEL_DETERMINISM.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] clkvalue Softmax value for GFXCLK in MHz.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_perf_determinism_mode_set(uint32_t dv_ind,
                                             uint64_t clkvalue);

/**
 *  @brief Get the overdrive percent associated with the device with provided
 *  device index.
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint32_t @p od,
 *  this function will write the overdrive percentage to the uint32_t pointed
 *  to by @p od
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] od a pointer to uint32_t to which the overdrive percentage
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_overdrive_level_get(uint32_t dv_ind, uint32_t *od);

/**
 *  @brief Get the memory clock overdrive percent associated with the device
 *  with provided device index.
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint32_t @p od,
 *  this function will write the memory overdrive percentage to the uint32_t
 *  pointed to by @p od
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] od a pointer to uint32_t to which the overdrive percentage
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_mem_overdrive_level_get(uint32_t dv_ind, uint32_t *od);

/**
 *  @brief Get the list of possible system clock speeds of device for a
 *  specified clock type.
 *
 *  @details Given a device index @p dv_ind, a clock type @p clk_type, and a
 *  pointer to a to an ::rsmi_frequencies_t structure @p f, this function will
 *  fill in @p f with the possible clock speeds, and indication of the current
 *  clock speed selection.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] clk_type the type of clock for which the frequency is desired
 *
 *  @param[inout] f a pointer to a caller provided ::rsmi_frequencies_t structure
 *  to which the frequency information will be written. Frequency values are in
 *  Hz.
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *  If multiple current frequencies are found, a warning is shown. If no
 *  current frequency is found, it is reflected as -1. If frequencies are not
 *  read from low to high a warning is shown as well.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_UNEXPECTED_DATA Data read or provided was not as
 *  expected
 *
 */
rsmi_status_t rsmi_dev_gpu_clk_freq_get(uint32_t dv_ind,
                             rsmi_clk_type_t clk_type, rsmi_frequencies_t *f);

/**
 *  @brief Reset the gpu associated with the device with provided device index
 *
 *  @details Given a device index @p dv_ind, this function will reset the GPU
 *
 *  @param[in] dv_ind a device index
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_gpu_reset(uint32_t dv_ind);

/**
 *  @brief This function retrieves the voltage/frequency curve information
 *
 *  @details Given a device index @p dv_ind and a pointer to a
 *  ::rsmi_od_volt_freq_data_t structure @p odv, this function will populate @p
 *  odv. See ::rsmi_od_volt_freq_data_t for more details.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] odv a pointer to an ::rsmi_od_volt_freq_data_t structure
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments. In the event where there are some values are missing from
 *  or not available on the device, the respective values will be set to
 *  UINT64_MAX.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t rsmi_dev_od_volt_info_get(uint32_t dv_ind,
                                               rsmi_od_volt_freq_data_t *odv);

/**
 *  @brief This function retrieves the gpu metrics information
 *
 *  @details Given a device index @p dv_ind and a pointer to a
 *  ::rsmi_gpu_metrics_t structure @p pgpu_metrics, this function will populate
 *  @p pgpu_metrics. See ::rsmi_gpu_metrics_t for more details.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] pgpu_metrics a pointer to an ::rsmi_gpu_metrics_t structure
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t rsmi_dev_gpu_metrics_info_get(uint32_t dv_ind,
                                            rsmi_gpu_metrics_t *pgpu_metrics);

/**
 *  @brief This function sets the clock range information
 *
 *  @details Given a device index @p dv_ind, a minimum clock value @p minclkvalue,
 *  a maximum clock value @p maxclkvalue and a clock type @p clkType this function
 *  will set the sclk|mclk range
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] minclkvalue value to apply to the clock range. Frequency values
 *  are in MHz.
 *
 *  @param[in] maxclkvalue value to apply to the clock range. Frequency values
 *  are in MHz.
 *
 *  @param[in] clkType RSMI_CLK_TYPE_SYS | RSMI_CLK_TYPE_MEM range type
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t rsmi_dev_clk_range_set(uint32_t dv_ind, uint64_t minclkvalue,
                                       uint64_t maxclkvalue,
                                       rsmi_clk_type_t clkType);

/**
 *  @brief This function sets the clock min/max level
 *
 *  @details Given a device index @p dv_ind, a clock value @p minclkvalue,
 *  a maximum clock value @p maxclkvalue and a clock type @p clkType this function
 *  will set the sclk|mclk range
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] level RSMI_FREQ_IND_MIN|RSMI_FREQ_IND_MAX
 *
 *  @param[in] clkvalue value to apply to the clock level. Frequency values
 *  are in MHz.
 *
 *  @param[in] clkType RSMI_CLK_TYPE_SYS | RSMI_CLK_TYPE_MEM level type
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t rsmi_dev_clk_extremum_set(uint32_t dv_ind, rsmi_freq_ind_t level,
                                       uint64_t clkvalue,
                                       rsmi_clk_type_t clkType);

/**
 *  @brief This function sets the clock frequency information
 *
 *  @details Given a device index @p dv_ind, a frequency level @p level,
 *  a clock value @p clkvalue and a clock type @p clkType this function
 *  will set the sclk|mclk range
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] level RSMI_FREQ_IND_MIN|RSMI_FREQ_IND_MAX to set the
 *  minimum (0) or maximum (1) speed.
 *
 *  @param[in] clkvalue value to apply to the clock range. Frequency values
 *  are in MHz.
 *
 *  @param[in] clkType RSMI_CLK_TYPE_SYS | RSMI_CLK_TYPE_MEM range type
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t rsmi_dev_od_clk_info_set(uint32_t dv_ind, rsmi_freq_ind_t level,
                                       uint64_t clkvalue,
                                       rsmi_clk_type_t clkType);

/**
 *  @brief This function sets  1 of the 3 voltage curve points.
 *
 *  @details Given a device index @p dv_ind, a voltage point @p vpoint
 *  and a voltage value @p voltvalue this function will set voltage curve point
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] vpoint voltage point [0|1|2] on the voltage curve
 *
 *  @param[in] clkvalue clock value component of voltage curve point.
 *  Frequency values are in MHz.
 *
 *  @param[in] voltvalue voltage value component of voltage curve point.
 *  Voltage is in mV.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t rsmi_dev_od_volt_info_set(uint32_t dv_ind, uint32_t vpoint,
                                        uint64_t clkvalue, uint64_t voltvalue);

/**
 *  @brief This function will retrieve the current valid regions in the
 *  frequency/voltage space.
 *
 *  @details Given a device index @p dv_ind, a pointer to an unsigned integer
 *  @p num_regions and a buffer of ::rsmi_freq_volt_region_t structures, @p
 *  buffer, this function will populate @p buffer with the current
 *  frequency-volt space regions. The caller should assign @p buffer to memory
 *  that can be written to by this function. The caller should also
 *  indicate the number of ::rsmi_freq_volt_region_t structures that can safely
 *  be written to @p buffer in @p num_regions.
 *
 *  The number of regions to expect this function provide (@p num_regions) can
 *  be obtained by calling ::rsmi_dev_od_volt_info_get().
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] num_regions As input, this is the number of
 *  ::rsmi_freq_volt_region_t structures that can be written to @p buffer. As
 *  output, this is the number of ::rsmi_freq_volt_region_t structures that were
 *  actually written.
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[inout] buffer a caller provided buffer to which
 *  ::rsmi_freq_volt_region_t structures will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t rsmi_dev_od_volt_curve_regions_get(uint32_t dv_ind,
                      uint32_t *num_regions, rsmi_freq_volt_region_t *buffer);

/**
 *  @brief Get the list of available preset power profiles and an indication of
 *  which profile is currently active.
 *
 *  @details Given a device index @p dv_ind and a pointer to a
 *  ::rsmi_power_profile_status_t @p status, this function will set the bits of
 *  the ::rsmi_power_profile_status_t.available_profiles bit field of @p status to
 *  1 if the profile corresponding to the respective
 *  ::rsmi_power_profile_preset_masks_t profiles are enabled. For example, if both
 *  the VIDEO and VR power profiles are available selections, then
 *  ::RSMI_PWR_PROF_PRST_VIDEO_MASK AND'ed with
 *  ::rsmi_power_profile_status_t.available_profiles will be non-zero as will
 *  ::RSMI_PWR_PROF_PRST_VR_MASK AND'ed with
 *  ::rsmi_power_profile_status_t.available_profiles. Additionally,
 *  ::rsmi_power_profile_status_t.current will be set to the
 *  ::rsmi_power_profile_preset_masks_t of the profile that is currently active.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[inout] status a pointer to ::rsmi_power_profile_status_t that will be
 *  populated by a call to this function
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_dev_power_profile_presets_get(uint32_t dv_ind, uint32_t sensor_ind,
                                         rsmi_power_profile_status_t *status);

/** @} */  // end of PerfQuer
/*****************************************************************************/

/** @defgroup PerfCont Clock, Power and Performance Control
 *  These functions provide control over clock frequencies, power and
 *  performance.
 *  @{
 */
/**
 *  @brief Set the PowerPlay performance level associated with the device with
 *  provided device index with the provided value.
 *
 *  @deprecated ::rsmi_dev_perf_level_set_v1() is preferred, with an
 *  interface that more closely  matches the rest of the rocm_smi API.
 *
 *  @details Given a device index @p dv_ind and an ::rsmi_dev_perf_level_t @p
 *  perf_level, this function will set the PowerPlay performance level for the
 *  device to the value @p perf_lvl.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] perf_lvl the value to which the performance level should be set
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_PERMISSION function requires root access
 *
 */
rsmi_status_t
rsmi_dev_perf_level_set(uint32_t dv_ind, rsmi_dev_perf_level_t perf_lvl);

/**
 *  @brief Set the PowerPlay performance level associated with the device with
 *  provided device index with the provided value.
 *
 *  @details Given a device index @p dv_ind and an ::rsmi_dev_perf_level_t @p
 *  perf_level, this function will set the PowerPlay performance level for the
 *  device to the value @p perf_lvl.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] perf_lvl the value to which the performance level should be set
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_PERMISSION function requires root access
 *
 */
rsmi_status_t
rsmi_dev_perf_level_set_v1(uint32_t dv_ind, rsmi_dev_perf_level_t perf_lvl);

/**
 *  @brief Set the overdrive percent associated with the device with provided
 *  device index with the provided value. See details for WARNING.
 *
 *  @deprecated This function is deprecated. ::rsmi_dev_overdrive_level_set_v1
 *  has the same functionaltiy, with an interface that more closely
 *  matches the rest of the rocm_smi API.
 *
 *  @details Given a device index @p dv_ind and an overdrive level @p od,
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
 *  @param[in] dv_ind a device index
 *
 *  @param[in] od the value to which the overdrive level should be set
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_PERMISSION function requires root access
 *
 */
rsmi_status_t rsmi_dev_overdrive_level_set(uint32_t dv_ind, uint32_t od);

/**
 *  @brief Set the overdrive percent associated with the device with provided
 *  device index with the provided value. See details for WARNING.
 *
 *  @details Given a device index @p dv_ind and an overdrive level @p od,
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
 *  @param[in] dv_ind a device index
 *
 *  @param[in] od the value to which the overdrive level should be set
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_PERMISSION function requires root access
 *
 */
rsmi_status_t rsmi_dev_overdrive_level_set_v1(uint32_t dv_ind, uint32_t od);

/**
 * @brief Control the set of allowed frequencies that can be used for the
 * specified clock.
 *
 * @details Given a device index @p dv_ind, a clock type @p clk_type, and a
 * 64 bit bitmask @p freq_bitmask, this function will limit the set of
 * allowable frequencies. If a bit in @p freq_bitmask has a value of 1, then
 * the frequency (as ordered in an ::rsmi_frequencies_t returned by
 * rsmi_dev_gpu_clk_freq_get()) corresponding to that bit index will be
 * allowed.
 *
 * This function will change the performance level to
 * ::RSMI_DEV_PERF_LEVEL_MANUAL in order to modify the set of allowable
 * frequencies. Caller will need to set to ::RSMI_DEV_PERF_LEVEL_AUTO in order
 * to get back to default state.
 *
 * All bits with indices greater than or equal to
 * ::rsmi_frequencies_t::num_supported will be ignored.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] clk_type the type of clock for which the set of frequencies
 *  will be modified
 *
 *  @param[in] freq_bitmask A bitmask indicating the indices of the
 *  frequencies that are to be enabled (1) and disabled (0). Only the lowest
 *  ::rsmi_frequencies_t.num_supported bits of this mask are relevant.
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_PERMISSION function requires root access
 *
 */
rsmi_status_t rsmi_dev_gpu_clk_freq_set(uint32_t dv_ind,
                             rsmi_clk_type_t clk_type, uint64_t freq_bitmask);

/** @} */  // end of PerfCont

/*****************************************************************************/
/** @defgroup VersQuer Version Queries
 *  These functions provide version information about various subsystems.
 *  @{
 */

/**
 * @brief Get the build version information for the currently running build of
 * RSMI.
 *
 * @details  Get the major, minor, patch and build string for RSMI build
 * currently in use through @p version
 *
 * @param[inout] version A pointer to an ::rsmi_version_t structure that will
 * be updated with the version information upon return.
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call
 *
 */
rsmi_status_t
rsmi_version_get(rsmi_version_t *version);

/**
 *  @brief Get the driver version string for the current system.
 *
 *  @details Given a software component @p component, a pointer to a char
 *  buffer, @p ver_str, this function will write the driver version string
 *  (up to @p len characters) for the current system to @p ver_str. The caller
 *  must ensure that it is safe to write at least @p len characters to @p
 *  ver_str.
 *
 *  @param[in] component The component for which the version string is being
 *  requested
 *
 *  @param[inout] ver_str A pointer to a buffer of char's to which the version
 *  of @p component will be written
 *
 *  @param[in] len the length of the caller provided buffer @p name.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire name. In this case, only @p len bytes will
 *  be written.
 */
rsmi_status_t
rsmi_version_str_get(rsmi_sw_component_t component, char *ver_str,
                                                                uint32_t len);

/**
 *  @brief Get the VBIOS identifer string
 *
 *  @details Given a device ID @p dv_ind, and a pointer to a char buffer,
 *  @p vbios, this function will write the VBIOS string (up to @p len
 *  characters) for device @p dv_ind to @p vbios. The caller must ensure that
 *  it is safe to write at least @p len characters to @p vbios.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] vbios A pointer to a buffer of char's to which the VBIOS name
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[in] len The number of char's pointed to by @p vbios which can safely
 *  be written to by this function.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_dev_vbios_version_get(uint32_t dv_ind, char *vbios, uint32_t len);

/**
 *  @brief Get the firmware versions for a device
 *
 *  @details Given a device ID @p dv_ind, and a pointer to a uint64_t,
 *  @p fw_version, this function will write the FW Versions as a string (up to @p len
 *  characters) for device @p dv_ind to @p vbios. The caller must ensure that
 *  it is safe to write at least @p len characters to @p vbios.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] block The firmware block for which the version is being requested
 *
 *  @param[inout] fw_version The version for the firmware block
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_dev_firmware_version_get(uint32_t dv_ind, rsmi_fw_block_t block,
                                                        uint64_t *fw_version);

/**
 *  @brief Get the target graphics version for a GPU device
 *
 *  @details Given a device ID @p dv_ind and a uint64_t pointer
 *  @p gfx_version, this function will write the graphics version.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] gfx_version The device graphics version number indicated by
 *  KFD. If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS. If device does not support this value,
 *  will return ::RSMI_STATUS_NOT_SUPPORTED and a maximum UINT64 value as
 *  @p gfx_version.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_target_graphics_version_get(uint32_t dv_ind,
                                                  uint64_t *gfx_version);

/** @} */  // end of VersQuer

/*****************************************************************************/
/** @defgroup ErrQuer Error Queries
 *  These functions provide error information about RSMI calls as well as
 *  device errors.
 *  @{
 */

/**
 *  @brief Retrieve the error counts for a GPU block
 *
 *  @details Given a device index @p dv_ind, an ::rsmi_gpu_block_t @p block and a
 *  pointer to an ::rsmi_error_count_t @p ec, this function will write the error
 *  count values for the GPU block indicated by @p block to memory pointed to by
 *  @p ec.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] block The block for which error counts should be retrieved
 *
 *  @param[inout] ec A pointer to an ::rsmi_error_count_t to which the error
 *  counts should be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_ecc_count_get(uint32_t dv_ind,
                              rsmi_gpu_block_t block, rsmi_error_count_t *ec);

/**
 *  @brief Retrieve the enabled ECC bit-mask
 *
 *  @details Given a device index @p dv_ind, and a pointer to a uint64_t @p
 *  enabled_mask, this function will write bits to memory pointed to by
 *  @p enabled_blocks. Upon a successful call, @p enabled_blocks can then be
 *  AND'd with elements of the ::rsmi_gpu_block_t ennumeration to determine if
 *  the corresponding block has ECC enabled. Note that whether a block has ECC
 *  enabled or not in the device is independent of whether there is kernel
 *  support for error counting for that block. Although a block may be enabled,
 *  but there may not be kernel support for reading error counters for that
 *  block.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] enabled_blocks A pointer to a uint64_t to which the enabled
 *  blocks bits will be written.
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t rsmi_dev_ecc_enabled_get(uint32_t dv_ind,
                                                    uint64_t *enabled_blocks);

/**
 *  @brief Retrieve the ECC status for a GPU block
 *
 *  @details Given a device index @p dv_ind, an ::rsmi_gpu_block_t @p block and
 *  a pointer to an ::rsmi_ras_err_state_t @p state, this function will write
 *  the current state for the GPU block indicated by @p block to memory pointed
 *  to by @p state.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] block The block for which error counts should be retrieved
 *
 *  @param[inout] state A pointer to an ::rsmi_ras_err_state_t to which the
 *  ECC state should be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t rsmi_dev_ecc_status_get(uint32_t dv_ind, rsmi_gpu_block_t block,
                                                  rsmi_ras_err_state_t *state);
/**
 *  @brief Get a description of a provided RSMI error status
 *
 *  @details Set the provided pointer to a const char *, @p status_string, to
 *  a string containing a description of the provided error code @p status.
 *
 *  @param[in] status The error status for which a description is desired
 *
 *  @param[inout] status_string A pointer to a const char * which will be made
 *  to point to a description of the provided error code
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call
 *
 */
rsmi_status_t
rsmi_status_string(rsmi_status_t status, const char **status_string);

/** @} */  // end of ErrQuer

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
 *  ::rsmi_dev_counter_group_supported() can be used to see which event types
 *  (::rsmi_event_group_t) are supported for a given device. Assuming a device
 *  supports a given event type, we can then check to see if there are counters
 *  available to count a specific event with
 *  ::rsmi_counter_available_counters_get(). Counters may be occupied by other
 *  perf based programs.
 *
 *  Once it is determined that events are supported and counters are available,
 *  an event counter can be created/destroyed and controlled.
 *
 *  ::rsmi_dev_counter_create() allocates internal data structures that will be
 *  used to used to control the event counter, and return a handle to this data
 *  structure.
 *
 *  Once an event counter handle is obtained, the event counter can be
 *  controlled (i.e., started, stopped,...) with ::rsmi_counter_control() by
 *  passing ::rsmi_counter_command_t commands. ::RSMI_CNTR_CMD_START starts an
 *  event counter and ::RSMI_CNTR_CMD_STOP stops a counter.
 *  ::rsmi_counter_read() reads an event counter.
 *
 *  Once the counter is no longer needed, the resources it uses should be freed
 *  by calling ::rsmi_dev_counter_destroy().
 *
 *
 *  Important Notes about Counter Values
 *  ====================================
 *  - A running "absolute" counter is kept internally. For the discussion that
 *  follows, we will call the internal counter value at time \a t \a
 *  val<sub>t</sub>
 *  - Issuing ::RSMI_CNTR_CMD_START or calling ::rsmi_counter_read(), causes
 *  RSMI (in kernel) to internally record the current absolute counter value
 *  - ::rsmi_counter_read() returns the number of events that have occurred
 *  since the previously recorded value (ie, a relative value,
 *  \a val<sub>t</sub> - val<sub>t-1</sub>) from the issuing of
 *  ::RSMI_CNTR_CMD_START or calling ::rsmi_counter_read()
 *
 *  Example of event counting sequence:
 *
 *  \latexonly
 *  \pagebreak
 *  \endlatexonly
 *  \code{.cpp}
 *
 *    rsmi_counter_value_t value;
 *
 *    // Determine if RSMI_EVNT_GRP_XGMI is supported for device dv_ind
 *    ret = rsmi_dev_counter_group_supported(dv_ind, RSMI_EVNT_GRP_XGMI);
 *
 *    // See if there are counters available for device dv_ind for event
 *    // RSMI_EVNT_GRP_XGMI
 *
 *    ret = rsmi_counter_available_counters_get(dv_ind,
 *                                 RSMI_EVNT_GRP_XGMI, &counters_available);
 *
 *    // Assuming RSMI_EVNT_GRP_XGMI is supported and there is at least 1
 *    // counter available for RSMI_EVNT_GRP_XGMI on device dv_ind, create
 *    // an event object for an event of group RSMI_EVNT_GRP_XGMI (e.g.,
 *    // RSMI_EVNT_XGMI_0_BEATS_TX) and get the handle
 *    // (rsmi_event_handle_t).
 *
 *    ret = rsmi_dev_counter_create(dv_ind, RSMI_EVNT_XGMI_0_BEATS_TX,
 *                                                          &evnt_handle);
 *
 *    // A program that generates the events of interest can be started
 *    // immediately before or after starting the counters.
 *    // Start counting:
 *    ret = rsmi_counter_control(evnt_handle, RSMI_CNTR_CMD_START, NULL);
 *
 *    // Wait...
 *
 *    // Get the number of events since RSMI_CNTR_CMD_START was issued:
 *    ret = rsmi_counter_read(rsmi_event_handle_t evt_handle, &value)
 *
 *    // Wait...
 *
 *    // Get the number of events since rsmi_counter_read() was last called:
 *    ret = rsmi_counter_read(rsmi_event_handle_t evt_handle, &value)
 *
 *    // Stop counting.
 *    ret = rsmi_counter_control(evnt_handle, RSMI_CNTR_CMD_STOP, NULL);
 *
 *    // Release all resources (e.g., counter and memory resources) associated
 *    with evnt_handle.
 *    ret = rsmi_dev_counter_destroy(evnt_handle);
 *  \endcode
 *  @{
 */

/**
 *  @brief Tell if an event group is supported by a given device
 *
 *  @details Given a device index @p dv_ind and an event group specifier @p
 *  group, tell if @p group type events are supported by the device associated
 *  with @p dv_ind
 *
 *  @param[in] dv_ind device index of device being queried
 *
 *  @param[in] group ::rsmi_event_group_t identifier of group for which support
 *  is being queried
 *
 *  @retval ::RSMI_STATUS_SUCCESS if the device associatee with @p dv_ind
 *  support counting events of the type indicated by @p group.
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  group
 *
 */
rsmi_status_t
rsmi_dev_counter_group_supported(uint32_t dv_ind, rsmi_event_group_t group);

/**
 *  @brief Create a performance counter object
 *
 *  @details Create a performance counter object of type @p type for the device
 *  with a device index of @p dv_ind, and write a handle to the object to the
 *  memory location pointed to by @p evnt_handle. @p evnt_handle can be used
 *  with other performance event operations. The handle should be deallocated
 *  with ::rsmi_dev_counter_destroy() when no longer needed.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] type the ::rsmi_event_type_t of performance event to create
 *
 *  @param[inout] evnt_handle A pointer to a ::rsmi_event_handle_t which will be
 *  associated with a newly allocated counter
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_OUT_OF_RESOURCES unable to allocate memory for counter
 *  @retval ::RSMI_STATUS_PERMISSION function requires root access
 *
 */
rsmi_status_t
rsmi_dev_counter_create(uint32_t dv_ind, rsmi_event_type_t type,
                                            rsmi_event_handle_t *evnt_handle);

/**
 *  @brief Deallocate a performance counter object
 *
 *  @details Deallocate the performance counter object with the provided
 *  ::rsmi_event_handle_t @p evnt_handle
 *
 *  @param[in] evnt_handle handle to event object to be deallocated
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_PERMISSION function requires root access
 *
 */
rsmi_status_t
rsmi_dev_counter_destroy(rsmi_event_handle_t evnt_handle);

/**
 *  @brief Issue performance counter control commands
 *
 *  @details Issue a command @p cmd on the event counter associated with the
 *  provided handle @p evt_handle.
 *
 *  @param[in] evt_handle an event handle
 *
 *  @param[in] cmd The event counter command to be issued
 *
 *  @param[inout] cmd_args Currently not used. Should be set to NULL.
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_PERMISSION function requires root access
 *
 */
rsmi_status_t
rsmi_counter_control(rsmi_event_handle_t evt_handle,
                                  rsmi_counter_command_t cmd, void *cmd_args);

/**
 *  @brief Read the current value of a performance counter
 *
 *  @details Read the current counter value of the counter associated with the
 *  provided handle @p evt_handle and write the value to the location pointed
 *  to by @p value.
 *
 *  @param[in] evt_handle an event handle
 *
 *  @param[inout] value pointer to memory of size of ::rsmi_counter_value_t to
 *  which the counter value will be written
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_PERMISSION function requires root access
 *
 */
rsmi_status_t
rsmi_counter_read(rsmi_event_handle_t evt_handle,
                                                 rsmi_counter_value_t *value);

/**
 *  @brief Get the number of currently available counters
 *
 *  @details Given a device index @p dv_ind, a performance event group @p grp,
 *  and a pointer to a uint32_t @p available, this function will write the
 *  number of @p grp type counters that are available on the device with index
 *  @p dv_ind to the memory that @p available points to.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] grp an event device group
 *
 *  @param[inout] available A pointer to a uint32_t to which the number of
 *  available counters will be written
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_counter_available_counters_get(uint32_t dv_ind,
                                 rsmi_event_group_t grp, uint32_t *available);
/** @} */  // end of PerfCntr

/*****************************************************************************/
/** @defgroup SysInfo System Information Functions
 *  These functions are used to configure, query and control performance
 *  counting.
 *  @{
 */

/**
 *  @brief Get process information about processes currently using GPU
 *
 *  @details Given a non-NULL pointer to an array @p procs of
 *  ::rsmi_process_info_t's, of length *@p num_items, this function will write
 *  up to *@p num_items instances of ::rsmi_process_info_t to the memory pointed
 *  to by @p procs. These instances contain information about each GPU compute
 *  process and their PASID for further analysis or monitoring via
 *  ::rsmi_compute_process_info_by_pid_get().
 *  If @p procs is not NULL, @p num_items will be updated with
 *  the number of processes actually written. If @p procs is NULL, @p num_items
 *  will be updated with the number of processes for which there is current
 *  process information. Calling this function with @p procs being NULL is a way
 *  to determine how much memory should be allocated for when @p procs is not
 *  NULL.
 *
 *  @param[inout] procs a pointer to memory provided by the caller to which
 *  process information will be written. This may be NULL in which case only @p
 *  num_items will be updated with the number of processes found.
 *
 *  @param[inout] num_items A pointer to a uint32_t, which on input, should
 *  contain the amount of memory in ::rsmi_process_info_t's which have been
 *  provided by the @p procs argument. On output, if @p procs is non-NULL, this
 *  will be updated with the number ::rsmi_process_info_t structs actually
 *  written. If @p procs is NULL, this argument will be updated with the number
 *  processes for which there is information.
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_INSUFFICIENT_SIZE is returned if there were more
 *  processes for which information was available, but not enough space was
 *  provided as indicated by @p procs and @p num_items, on input.
 */
rsmi_status_t
rsmi_compute_process_info_get(rsmi_process_info_t *procs, uint32_t *num_items);

/**
 *  @brief Get process information about a specific process
 *
 *  @details Given a pointer to an ::rsmi_process_info_t @p proc and a process
 *  id
 *  @p pid, this function will write the process information for @p pid, if
 *  available, to the memory pointed to by @p proc.
 *
 *  @param[in] pid The process ID for which process information is being
 *  requested
 *
 *  @param[inout] proc a pointer to a ::rsmi_process_info_t to which
 *  process information for @p pid will be written if it is found.
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_NOT_FOUND is returned if there was no process
 *  information
 *  found for the provided @p pid
 *
 */
rsmi_status_t
rsmi_compute_process_info_by_pid_get(uint32_t pid, rsmi_process_info_t *proc);

/**
 *  @brief Get the device indices currently being used by a process
 *
 *  @details Given a process id @p pid, a non-NULL pointer to an array of
 *  uint32_t's @p dv_indices of length *@p num_devices, this function will
 *  write up to @p num_devices device indices to the memory pointed to by
 *  @p dv_indices. If @p dv_indices is not NULL, @p num_devices will be
 *  updated with the number of gpu's currently being used by process @p pid.
 *  If @p dv_indices is NULL, @p dv_indices will be updated with the number of
 *  gpus currently being used by @p pid. Calling this function with @p
 *  dv_indices being NULL is a way to determine how much memory is required
 *  for when @p dv_indices is not NULL.
 *
 *  @param[in] pid The process id of the process for which the number of gpus
 *  currently being used is requested
 *
 *  @param[inout] dv_indices a pointer to memory provided by the caller to
 *  which indices of devices currently being used by the process will be
 *  written. This may be NULL in which case only @p num_devices will be
 *  updated with the number of devices being used.
 *
 *  @param[inout] num_devices A pointer to a uint32_t, which on input, should
 *  contain the amount of memory in uint32_t's which have been provided by the
 *  @p dv_indices argument. On output, if @p dv_indices is non-NULL, this will
 *  be updated with the number uint32_t's actually written. If @p dv_indices is
 *  NULL, this argument will be updated with the number devices being used.
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_INSUFFICIENT_SIZE is returned if there were more
 *  gpu indices that could have been written, but not enough space was
 *  provided as indicated by @p dv_indices and @p num_devices, on input.
 *
 */
rsmi_status_t
rsmi_compute_process_gpus_get(uint32_t pid, uint32_t *dv_indices,
                                            uint32_t *num_devices);

/**
 *  @brief Get the info of a process on a specific device.
 *
 *  @details Given a process id @p pid, a @p dv_ind, this function will
 *  write the process information for pid on the device, if available, to
 *  the memory pointed to by @p proc.
 *
 *  @param[in] pid The process id of the process for which the gpu
 *  currently being used is requested.
 *
 *  @param[in] dv_ind a device index where the process running on.
 *
 *  @param[inout] proc a pointer to memory provided by the caller to which
 *  process information will be written.
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_compute_process_info_by_device_get(uint32_t pid, uint32_t dv_ind,
                                                rsmi_process_info_t *proc);

/** @} */  // end of SysInfo

/*****************************************************************************/
/** @defgroup XGMIInfo XGMI Functions
 *  These functions are used to configure, query and control XGMI.
 *  @{
 */

/**
 *  @brief Retrieve the XGMI error status for a device
 *
 *  @details Given a device index @p dv_ind, and a pointer to an
 *  ::rsmi_xgmi_status_t @p status, this function will write the current XGMI
 *  error state ::rsmi_xgmi_status_t for the device @p dv_ind to the memory
 *  pointed to by @p status.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] status A pointer to an ::rsmi_xgmi_status_t to which the
 *  XGMI error state should be written
 *  If this parameter is nullptr, this function will return
 *  ::RSMI_STATUS_INVALID_ARGS if the function is supported with the provided,
 *  arguments and ::RSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_dev_xgmi_error_status(uint32_t dv_ind, rsmi_xgmi_status_t *status);

/**
 * @brief Reset the XGMI error status for a device
 *
 * @details Given a device index @p dv_ind, this function will reset the
 * current XGMI error state ::rsmi_xgmi_status_t for the device @p dv_ind to
 * rsmi_xgmi_status_t::RSMI_XGMI_STATUS_NO_ERRORS
 *
 * @param[in] dv_ind a device index
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t
rsmi_dev_xgmi_error_reset(uint32_t dv_ind);

/**
 *  @brief Retrieve the XGMI hive id for a device
 *
 *  @details Given a device index @p dv_ind, and a pointer to an
 *  uint64_t @p hive_id, this function will write the current XGMI
 *  hive id for the device @p dv_ind to the memory pointed to by @p hive_id.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] hive_id A pointer to an uint64_t to which the XGMI hive id
 *  should be written
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_dev_xgmi_hive_id_get(uint32_t dv_ind, uint64_t *hive_id);

/** @} */  // end of SysInfo

/*****************************************************************************/
/** @defgroup HWTopo Hardware Topology Functions
 *  These functions are used to query Hardware topology.
 *  @{
 */

/**
 *  @brief Retrieve the NUMA CPU node number for a device
 *
 *  @details Given a device index @p dv_ind, and a pointer to an
 *  uint32_t @p numa_node, this function will write the
 *  node number of NUMA CPU for the device @p dv_ind to the memory
 *  pointed to by @p numa_node.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] numa_node A pointer to an uint32_t to which the
 *  numa node number should be written.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_topo_get_numa_node_number(uint32_t dv_ind, uint32_t *numa_node);

/**
 *  @brief Retrieve the weight for a connection between 2 GPUs
 *
 *  @details Given a source device index @p dv_ind_src and
 *  a destination device index @p dv_ind_dst, and a pointer to an
 *  uint64_t @p weight, this function will write the
 *  weight for the connection between the device @p dv_ind_src
 *  and @p dv_ind_dst to the memory pointed to by @p weight.
 *
 *  @param[in] dv_ind_src the source device index
 *
 *  @param[in] dv_ind_dst the destination device index
 *
 *  @param[inout] weight A pointer to an uint64_t to which the
 *  weight for the connection should be written.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_topo_get_link_weight(uint32_t dv_ind_src, uint32_t dv_ind_dst,
                          uint64_t *weight);

/**
 *  @brief Retreive minimal and maximal io link bandwidth between 2 GPUs
 *
 *  @details Given a source device index @p dv_ind_src and
 *  a destination device index @p dv_ind_dst,  pointer to an
 *  uint64_t @p min_bandwidth, and a pointer to uint64_t @p max_bandiwidth,
 *  this function will write theoretical minimal and maximal bandwidth limits.
 *  API works if src and dst are connected via xgmi and have 1 hop distance.
 *
 *  @param[in] dv_ind_src the source device index
 *
 *  @param[in] dv_ind_dst the destination device index
 *
 *  @param[inout] min_bandwidth A pointer to an uint64_t to which the
 *  minimal bandwidth for the connection should be written.
 *
 *  @param[inout] max_bandwidth A pointer to an uint64_t to which the
 *  maximal bandwidth for the connection should be written.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 */
rsmi_status_t
rsmi_minmax_bandwidth_get(uint32_t dv_ind_src, uint32_t dv_ind_dst,
                          uint64_t *min_bandwidth, uint64_t *max_bandwidth);

/**
 *  @brief Retrieve the hops and the connection type between GPU to GPU/CPU
 *
 *  @details Given a source device index @p dv_ind_src and
 *  a destination device index @p dv_ind_dst, and a pointer to an
 *  uint64_t @p hops and a pointer to an RSMI_IO_LINK_TYPE @p type,
 *  this function will write the number of hops and the connection type
 *  between the device @p dv_ind_src and @p dv_ind_dst to the memory
 *  pointed to by @p hops and @p type.
 *
 *  To query the link type between GPU and CPU, given a source GPU index
 *  @p dev_ind_srcc and a destination device index @p dv_ind_dst
 *  CPU_NODE_INDEX(0xFFFFFFFF), a pointer to an
 *  uint64_t @p hops and a pointer to an RSMI_IO_LINK_TYPE @p type,
 *  this function will write the number of hops and the connection type
 *  between the device @p dv_ind_src and CPU to the memory
 *  pointed to by @p hops and @p type.
 *
 *  @param[in] dv_ind_src the source device index
 *
 *  @param[in] dv_ind_dst the destination device index
 *
 *  @param[inout] hops A pointer to an uint64_t to which the
 *  hops for the connection should be written.
 *
 *  @param[inout] type A pointer to an ::RSMI_IO_LINK_TYPE to which the
 *  type for the connection should be written.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_topo_get_link_type(uint32_t dv_ind_src, uint32_t dv_ind_dst,
                        uint64_t *hops, RSMI_IO_LINK_TYPE *type);

/**
 *  @brief Return P2P availability status between 2 GPUs
 *
 *  @details Given a source device index @p dv_ind_src and
 *  a destination device index @p dv_ind_dst, and a pointer to a
 *  bool @p accessible, this function will write the P2P connection status
 *  between the device @p dv_ind_src and @p dv_ind_dst to the memory
 *  pointed to by @p accessible.
 *
 *  @param[in] dv_ind_src the source device index
 *
 *  @param[in] dv_ind_dst the destination device index
 *
 *  @param[inout] accessible A pointer to a bool to which the status for
 *  the P2P connection availablity should be written.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *
 */
rsmi_status_t
rsmi_is_P2P_accessible(uint32_t dv_ind_src, uint32_t dv_ind_dst,
                       bool *accessible);

/** @} */  // end of HWTopo

/*****************************************************************************/
/** @defgroup ComputePartition Compute Partition Functions
 *  These functions are used to configure and query the device's
 *  compute parition setting.
 *  @{
 */

/**
 *  @brief Retrieves the current compute partitioning for a desired device
 *
 *  @details
 *  Given a device index @p dv_ind and a string @p compute_partition ,
 *  and uint32 @p len , this function will attempt to obtain the device's
 *  current compute partition setting string. Upon successful retreival,
 *  the obtained device's compute partition settings string shall be stored in
 *  the passed @p compute_partition char string variable.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] compute_partition a pointer to a char string variable,
 *  which the device's current compute partition will be written to.
 *
 *  @param[in] len the length of the caller provided buffer @p compute_partition
 *  , suggested length is 4 or greater.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_UNEXPECTED_DATA data provided to function is not valid
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *  @retval ::RSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire compute partition value. In this case,
 *  only @p len bytes will be written.
 *
 */
rsmi_status_t
rsmi_dev_compute_partition_get(uint32_t dv_ind, char *compute_partition,
                               uint32_t len);

/**
 *  @brief Modifies a selected device's compute partition setting.
 *
 *  @details Given a device index @p dv_ind, a type of compute partition
 *  @p compute_partition, this function will attempt to update the selected
 *  device's compute partition setting.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] compute_partition using enum ::rsmi_compute_partition_type_t,
 *  define what the selected device's compute partition setting should be
 *  updated to.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_PERMISSION function requires root access
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_SETTING_UNAVAILABLE the provided setting is
 *  unavailable for current device
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *  @retval ::RSMI_STATUS_BUSY A resource or mutex could not be acquired
 *  because it is already being used - device is busy
 *
 */
rsmi_status_t
rsmi_dev_compute_partition_set(uint32_t dv_ind,
                               rsmi_compute_partition_type_t compute_partition);

/**
 *  @brief Retrieves the partition_id for a desired device
 *
 *  @details
 *  Given a device index @p dv_ind and a uint32_t pointer @p partition_id ,
 *  this function will attempt to obtain the device's partition ID.
 *  Upon successful retreival, the obtained device's partition will be stored
 *  in the passed @p partition_id uint32_t variable. If device does
 *  not support partitions or is in SPX, a @p partition_id ID of 0 shall
 *  be returned.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] partition_id a uint32_t variable,
 *  which the device's partition_id will be written to.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *
 */
rsmi_status_t rsmi_dev_partition_id_get(uint32_t dv_ind, uint32_t *partition_id);

/** @} */  // end of ComputePartition

/*****************************************************************************/
/** @defgroup memory_partition The Memory Partition Functions
 *  These functions are used to query and set the device's current memory
 *  partition.
 *  @{
 */

/**
 *  @brief Retrieves the current memory partition for a desired device
 *
 *  @details
 *  Given a device index @p dv_ind and a string @p memory_partition ,
 *  and uint32 @p len , this function will attempt to obtain the device's
 *  memory partition string. Upon successful retreival, the obtained device's
 *  memory partition string shall be stored in the passed @p memory_partition
 *  char string variable.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] memory_partition a pointer to a char string variable,
 *  which the device's memory partition will be written to.
 *
 *  @param[in] len the length of the caller provided buffer @p memory_partition ,
 *  suggested length is 5 or greater.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_UNEXPECTED_DATA data provided to function is not valid
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *  @retval ::RSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire memory partition value. In this case,
 *  only @p len bytes will be written.
 *
 */
rsmi_status_t
rsmi_dev_memory_partition_get(uint32_t dv_ind, char *memory_partition,
                              uint32_t len);

/**
 *  @brief Retrieves the available memory partition capabilities
 *  for a desired device
 *
 *  @details
 *  Given a device index @p dv_ind and a string @p memory_partition_caps ,
 *  and uint32 @p len , this function will attempt to obtain the device's
 *  available memory partition capabilities string. Upon successful
 *  retreival, the obtained device's available memory partition capablilities
 *  string shall be stored in the passed @p memory_partition_caps
 *  char string variable.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] memory_partition_caps a pointer to a char string variable,
 *  which the device's available memory partition capabilities will be written to.
 *
 *  @param[in] len the length of the caller provided buffer @p len ,
 *  suggested length is 30 or greater.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_UNEXPECTED_DATA data provided to function is not valid
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *  @retval ::RSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire memory partition value. In this case,
 *  only @p len bytes will be written.
 *
 */
rsmi_status_t rsmi_dev_memory_partition_capabilities_get(
                uint32_t dv_ind, char *memory_partition_caps, uint32_t len);

/**
 *  @brief Modifies a selected device's current memory partition setting.
 *
 *  @details Given a device index @p dv_ind and a type of memory partition
 *  @p memory_partition, this function will attempt to update the selected
 *  device's memory partition setting.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] memory_partition using enum ::rsmi_memory_partition_type_t,
 *  define what the selected device's current mode setting should be updated to.
 *
 *  @retval ::RSMI_STATUS_SUCCESS call was successful
 *  @retval ::RSMI_STATUS_PERMISSION function requires root access
 *  @retval ::RSMI_STATUS_INVALID_ARGS the provided arguments are not valid
 *  @retval ::RSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *  @retval ::RSMI_STATUS_AMDGPU_RESTART_ERR could not successfully restart
 *  the amdgpu driver
  *  @retval ::RSMI_STATUS_BUSY A resource or mutex could not be acquired
 *  because it is already being used - device is busy
 *
 */
rsmi_status_t
rsmi_dev_memory_partition_set(uint32_t dv_ind,
                              rsmi_memory_partition_type_t memory_partition);

/** @} */  // end of memory_partition

/*****************************************************************************/
/** @defgroup APISupport Supported Functions
 *  API function support varies by both GPU type and the version of the
 *  installed ROCm stack.  The functions described in this section can be used
 *  to determine, up front, which functions are supported for a given device
 *  on a system. If such "up front" knowledge of support for a function is not
 *  needed, alternatively, one can call a device related function and check the
 *  return code.
 *
 *  Some functions have several variations ("variants") where some variants are
 *  supported and others are not. For example, on a given device,
 *  ::rsmi_dev_temp_metric_get may support some types of temperature metrics
 *  (e.g., ::RSMI_TEMP_CRITICAL_HYST), but not others
 *  (e.g., ::RSMI_TEMP_EMERGENCY).
 *
 *  In addition to a top level of variant support for a function, a function
 *  may have varying support for monitors/sensors. These are considered
 *  "sub-variants" in functions described in this section. Continuing the
 *  ::rsmi_dev_temp_metric_get example, if variant
 *  ::RSMI_TEMP_CRITICAL_HYST is supported, perhaps
 *  only the sub-variant sensors ::RSMI_TEMP_TYPE_EDGE
 *  and ::RSMI_TEMP_TYPE_EDGE are supported, but not
 *  ::RSMI_TEMP_TYPE_MEMORY.
 *
 *  In cases where a function takes in a sensor id parameter but does not have
 *  any "top level" variants, the functions in this section will indicate a
 *  default "variant", ::RSMI_DEFAULT_VARIANT, for the top level variant, and
 *  the various monitor support will be sub-variants of this.
 *
 *  The functions in this section use the "iterator" concept to list which
 *  functions are supported; to list which variants of the supported functions
 *  are supported; and finally which monitors/sensors are supported for a
 *  variant.
 *
 *  Here is example code that prints out all supported functions, their
 *  supported variants and sub-variants. Please see the related descriptions
 *  functions and RSMI types.
 *  \latexonly
 *  \pagebreak
 *  \endlatexonly
 *  \code{.cpp}
 *     rsmi_func_id_iter_handle_t iter_handle, var_iter, sub_var_iter;
 *     rsmi_func_id_value_t value;
 *     rsmi_status_t err;
 *
 *     for (uint32_t i = 0; i < <number of devices>; ++i) {
 *      std::cout << "Supported RSMI Functions:" << std::endl;
 *      std::cout << "\tVariants (Monitors)" << std::endl;
 *
 *      err = rsmi_dev_supported_func_iterator_open(i, &iter_handle);
 *
 *      while (1) {
 *        err = rsmi_func_iter_value_get(iter_handle, &value);
 *        std::cout << "Function Name: " << value.name << std::endl;
 *
 *        err = rsmi_dev_supported_variant_iterator_open(iter_handle, &var_iter);
 *        if (err != RSMI_STATUS_NO_DATA) {
 *          std::cout << "\tVariants/Monitors: ";
 *          while (1) {
 *            err = rsmi_func_iter_value_get(var_iter, &value);
 *            if (value.id == RSMI_DEFAULT_VARIANT) {
 *              std::cout << "Default Variant ";
 *            } else {
 *              std::cout << value.id;
 *            }
 *            std::cout << " (";
 *
 *            err =
 *              rsmi_dev_supported_variant_iterator_open(var_iter, &sub_var_iter);
 *            if (err != RSMI_STATUS_NO_DATA) {
 *
 *              while (1) {
 *                err = rsmi_func_iter_value_get(sub_var_iter, &value);
 *                std::cout << value.id << ", ";
 *
 *                err = rsmi_func_iter_next(sub_var_iter);
 *
 *                if (err == RSMI_STATUS_NO_DATA) {
 *                  break;
 *                }
 *              }
 *              err = rsmi_dev_supported_func_iterator_close(&sub_var_iter);
 *            }
 *
 *            std::cout << "), ";
 *
 *            err = rsmi_func_iter_next(var_iter);
 *
 *            if (err == RSMI_STATUS_NO_DATA) {
 *              break;
 *            }
 *          }
 *          std::cout << std::endl;
 *
 *          err = rsmi_dev_supported_func_iterator_close(&var_iter);
 *        }
 *
 *        err = rsmi_func_iter_next(iter_handle);
 *
 *        if (err == RSMI_STATUS_NO_DATA) {
 *          break;
 *        }
 *      }
 *      err = rsmi_dev_supported_func_iterator_close(&iter_handle);
 *    }
 * \endcode
 *
 *  @{
 */


/**
 * @brief Get a function name iterator of supported RSMI functions for a device
 *
 * @details Given a device index @p dv_ind, this function will write a function
 * iterator handle to the caller-provided memory pointed to by @p handle. This
 * handle can be used to iterate through all the supported functions.
 *
 * Note that although this function takes in @p dv_ind as an argument,
 * ::rsmi_dev_supported_func_iterator_open itself will not be among the
 * functions listed as supported. This is because
 * ::rsmi_dev_supported_func_iterator_open does not depend on hardware or
 * driver support and should always be supported.
 *
 * @param[in] dv_ind a device index of device for which support information is
 * requested
 *
 * @param[inout] handle A pointer to caller-provided memory to which the
 * function iterator will be written.
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t
rsmi_dev_supported_func_iterator_open(uint32_t dv_ind,
                                          rsmi_func_id_iter_handle_t *handle);

/**
 * @brief Get a variant iterator for a given handle
 *
 * @details Given a ::rsmi_func_id_iter_handle_t @p obj_h, this function will
 * write a function iterator handle to the caller-provided memory pointed to
 * by @p var_iter. This handle can be used to iterate through all the supported
 * variants of the provided handle. @p obj_h may be a handle to a function
 * object, as provided by a call to ::rsmi_dev_supported_func_iterator_open, or
 * it may be a variant itself (from a call to
 * ::rsmi_dev_supported_variant_iterator_open), it which case @p var_iter will
 * be an iterator of the sub-variants of @p obj_h (e.g., monitors).
 *
 * This call allocates a small amount of memory to @p var_iter. To free this memory
 * ::rsmi_dev_supported_func_iterator_close should be called on the returned
 * iterator handle @p var_iter when it is no longer needed.
 *
 * @param[in] obj_h an iterator handle for which the variants are being requested
 *
 * @param[inout] var_iter A pointer to caller-provided memory to which the
 * sub-variant iterator will be written.
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t
rsmi_dev_supported_variant_iterator_open(rsmi_func_id_iter_handle_t obj_h,
                                        rsmi_func_id_iter_handle_t *var_iter);

/**
 * @brief Advance a function identifer iterator
 *
 * @details Given a function id iterator handle (::rsmi_func_id_iter_handle_t)
 * @p handle, this function will increment the iterator to point to the next
 * identifier. After a successful call to this function, obtaining the value
 * of the iterator @p handle will provide the value of the next item in the
 * list of functions/variants.
 *
 * If there are no more items in the list, ::RSMI_STATUS_NO_DATA is returned.
 *
 * @param[in] handle A pointer to an iterator handle to be incremented
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 * @retval ::RSMI_STATUS_NO_DATA is returned when list of identifiers has been
 * exhausted
 *
 */
rsmi_status_t
rsmi_func_iter_next(rsmi_func_id_iter_handle_t handle);

/**
 * @brief Close a variant iterator handle
 *
 * @details Given a pointer to an ::rsmi_func_id_iter_handle_t @p handle, this
 * function will free the resources being used by the handle
 *
 * @param[in] handle A pointer to an iterator handle to be closed
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t
rsmi_dev_supported_func_iterator_close(rsmi_func_id_iter_handle_t *handle);

/**
 * @brief Get the value associated with a function/variant iterator
 *
 * @details Given an ::rsmi_func_id_iter_handle_t @p handle, this function
 * will write the identifier of the function/variant to the user provided
 * memory pointed to by @p value.
 *
 * @p value may point to a function name, a variant id, or a monitor/sensor
 * index, depending on what kind of iterator @p handle is
 *
 * @param[in] handle An iterator for which the value is being requested
 *
 * @param[inout] value A pointer to an ::rsmi_func_id_value_t provided by the
 * caller to which this function will write the value assocaited with @p handle
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t
rsmi_func_iter_value_get(rsmi_func_id_iter_handle_t handle,
                                                 rsmi_func_id_value_t *value);

/** @} */  // end of APISupport

/*****************************************************************************/
/** @defgroup EvntNotif Event Notification Functions
 *  These functions are used to configure for and get asynchronous event
 *  notifications.
 *  @{
 */

/**
 * @brief Prepare to collect event notifications for a GPU
 *
 * @details This function prepares to collect events for the GPU with device
 * ID @p dv_ind, by initializing any required system parameters. This call
 * may open files which will remain open until ::rsmi_event_notification_stop()
 * is called.
 *
 * @param dv_ind a device index corresponding to the device on which to
 * listen for events
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 */
rsmi_status_t
rsmi_event_notification_init(uint32_t dv_ind);

/**
 * @brief Specify which events to collect for a device
 *
 * @details Given a device index @p dv_ind and a @p mask consisting of
 * elements of ::rsmi_evt_notification_type_t OR'd together, this function
 * will listen for the events specified in @p mask on the device
 * corresponding to @p dv_ind.
 *
 * @param dv_ind a device index corresponding to the device on which to
 * listen for events
 *
 * @param mask Bitmask generated by OR'ing 1 or more elements of
 * ::rsmi_evt_notification_type_t indicating which event types to listen for,
 * where the rsmi_evt_notification_type_t value indicates the bit field, with
 * bit position starting from 1.
 * For example, if the mask field is 0x0000000000000003, which means first bit,
 * bit 1 (bit position start from 1) and bit 2 are set, which indicate interest
 * in receiving RSMI_EVT_NOTIF_VMFAULT (which has a value of 1) and
 * RSMI_EVT_NOTIF_THERMAL_THROTTLE event (which has a value of 2).
 *
 * @retval ::RSMI_STATUS_INIT_ERROR is returned if
 * ::rsmi_event_notification_init() has not been called before a call to this
 * function
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call
 */
rsmi_status_t
rsmi_event_notification_mask_set(uint32_t dv_ind, uint64_t mask);

/**
 * @brief Collect event notifications, waiting a specified amount of time
 *
 * @details Given a time period @p timeout_ms in milliseconds and a caller-
 * provided buffer of ::rsmi_evt_notification_data_t's @p data with a length
 * (in ::rsmi_evt_notification_data_t's, also specified by the caller) in the
 * memory location pointed to by @p num_elem, this function will collect
 * ::rsmi_evt_notification_type_t events for up to @p timeout_ms milliseconds,
 * and write up to *@p num_elem event items to @p data. Upon return @p num_elem
 * is updated with the number of events that were actually written. If events
 * are already present when this function is called, it will write the events
 * to the buffer then poll for new events if there is still caller-provided
 * buffer available to write any new events that would be found.
 *
 * This function requires prior calls to ::rsmi_event_notification_init() and
 * ::rsmi_event_notification_mask_set(). This function polls for the
 * occurrance of the events on the respective devices that were previously
 * specified by ::rsmi_event_notification_mask_set().
 *
 * @param[in] timeout_ms number of milliseconds to wait for an event
 * to occur
 *
 * @param[inout] num_elem pointer to uint32_t, provided by the caller. On
 * input, this value tells how many ::rsmi_evt_notification_data_t elements
 * are being provided by the caller with @p data. On output, the location
 * pointed to by @p num_elem will contain the number of items written to
 * the provided buffer.
 *
 * @param[out] data pointer to a caller-provided memory buffer of size
 * @p num_elem ::rsmi_evt_notification_data_t to which this function may safely
 * write. If there are events found, up to @p num_elem event items will be
 * written to @p data.
 *
 * @retval ::RSMI_STATUS_SUCCESS The function ran successfully. The events
 * that were found are written to @p data and @p num_elems is updated
 * with the number of elements that were written.
 *
 * @retval ::RSMI_STATUS_NO_DATA No events were found to collect.
 *
 */
rsmi_status_t
rsmi_event_notification_get(int timeout_ms,
                     uint32_t *num_elem, rsmi_evt_notification_data_t *data);

/**
 * @brief Close any file handles and free any resources used by event
 * notification for a GPU
 *
 * @details Any resources used by event notification for the GPU with
 * device index @p dv_ind will be free with this
 * function. This includes freeing any memory and closing file handles. This
 * should be called for every call to ::rsmi_event_notification_init()
 *
 * @param[in] dv_ind The device index of the GPU for which event
 * notification resources will be free
 *
 * @retval ::RSMI_STATUS_INVALID_ARGS resources for the given device have
 * either already been freed, or were never allocated by
 * ::rsmi_event_notification_init()
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call
 */
rsmi_status_t rsmi_event_notification_stop(uint32_t dv_ind);

/** @} */  // end of EvntNotif


/*****************************************************************************/
/** @defgroup GPU Metric Functions
 *  These functions are used to get granular information about all counters
 *  available in GPU Metrics.
 *  @{
 */

/**
 *  @brief Get the 'metrics_header_info' from the GPU metrics associated with the device
 *
 *  @details Given a device index @p dv_ind and a pointer to a metrics_table_header_t in which
 *  the 'metrics_header_info' will stored
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] header_value a pointer to metrics_table_header_t to which the device gpu
 *  metric unit will be stored
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *          ::RSMI_STATUS_NOT_SUPPORTED is returned in case the metric unit
 *            does not exist for the given device
 *
 */
rsmi_status_t
rsmi_dev_metrics_header_info_get(uint32_t dv_ind, metrics_table_header_t* header_value);

/**
 *  @brief Get the 'xcd_counter' from the GPU metrics associated with the device
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint16_t in which
 *  the 'xcd_counter' will stored
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] xcd_counter_value a pointer to uint16_t to which the device gpu
 *  metric unit will be stored
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *          ::RSMI_STATUS_NOT_SUPPORTED is returned in case the metric unit
 *            does not exist for the given device
 *
 */
rsmi_status_t
rsmi_dev_metrics_xcd_counter_get(uint32_t dv_ind, uint16_t* xcd_counter_value);

/**
 *  @brief Get the log from the GPU metrics associated with the device
 *
 *  @details Given a device index @p dv_ind it will log all the gpu metric info
 *  related to the device. The 'logging' feature must be on.
 *
 *  @param[in] dv_ind a device index
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t
rsmi_dev_metrics_log_get(uint32_t dv_ind);

/** @} */  // end of DevMetricsHeaderInfoGet

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // ROCM_SMI_ROCM_SMI_H_
