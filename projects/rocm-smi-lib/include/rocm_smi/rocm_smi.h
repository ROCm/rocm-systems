/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
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
#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_H_

#ifdef __cplusplus
extern "C" {
#include <cstdint>
#else
#include <stdinit.h>
#endif  // __cplusplus

#include <stdint.h>
#include <stddef.h>

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
#define RSMI_MAX_NUM_FREQUENCIES 32

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
                                         //!< given input
  RSMI_STATUS_FILE_ERROR,                //!< Problem accessing a file. This
                                         //!< may because the operation is not
                                         //!< supported by the Linux kernel
                                         //!< version running on the executing
                                         //!< machine
  RSMI_STATUS_PERMISSION,                //!< Permission denied/EACCESS file
                                         //!< error
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
  RSMI_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF,  //!< An unknown error occurred
} rsmi_status_t;

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

  RSMI_DEV_PERF_LEVEL_LAST = RSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK,

  RSMI_DEV_PERF_LEVEL_UNKNOWN = 0x100   //!< Unknown performance level
} rsmi_dev_perf_level_t;
/// \cond Ignore in docs.
typedef rsmi_dev_perf_level_t rsmi_dev_perf_level;
/// \endcond
/**
 * @brief Available clock types.
 */
typedef enum {
  RSMI_CLK_TYPE_SYS = 0x0,            //!< System clock
  RSMI_CLK_TYPE_FIRST = RSMI_CLK_TYPE_SYS,

  RSMI_CLK_TYPE_MEM,                  //!< Memory clock
  RSMI_CLK_TYPE_LAST = RSMI_CLK_TYPE_MEM
} rsmi_clk_type_t;
/// \cond Ignore in docs.
typedef rsmi_clk_type_t rsmi_clk_type;
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
  RSMI_TEMP_MIN_HYST,        //!< Temperature hysteresis value for min limit.
  RSMI_TEMP_CRITICAL,        //!< Temperature critical max value, typically
                             //!<  greater than corresponding temp_max values.
  RSMI_TEMP_CRITICAL_HYST,   //!< Temperature hysteresis value for critical
                             //!<  limit.
  RSMI_TEMP_EMERGENCY,       //!< Temperature emergency max value, for chips
                             //!<  supporting more than two upper temperature
                             //!<  limits. Must be equal or greater than
                             //!<  corresponding temp_crit values.
  RSMI_TEMP_EMERGENCY_HYST,  //!< Temperature hysteresis value for emergency
                             //!<  limit.
  RSMI_TEMP_CRIT_MIN,        //!< Temperature critical min value, typically
                             //!<  lower than corresponding temperature
                             //!<  minimum values.
  RSMI_TEMP_CRIT_MIN_HYST,   //!< Temperature hysteresis value for critical
                             //!<  minimum limit.
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
 * @brief Pre-set Profile Selections. These bitmasks can be AND'd with the
 * ::rsmi_power_profile_status_t.available_profiles returned from
 * ::rsmi_dev_power_profile_presets_get() to determine which power profiles
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
  RSMI_GPU_BLOCK_FIRST = 0,

  RSMI_GPU_BLOCK_UMC = RSMI_GPU_BLOCK_FIRST,
  RSMI_GPU_BLOCK_SDMA,
  RSMI_GPU_BLOCK_GFX,

  RSMI_GPU_BLOCK_LAST = RSMI_GPU_BLOCK_GFX,
} rsmi_gpu_block_t;
/// \cond Ignore in docs.
typedef rsmi_gpu_block_t rsmi_gpu_block;
/// \endcond

typedef struct {
} rsmi_memory_type_t;
/**
 * @brief This values of this enum are used as frequency identifiers.
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
 * @brief Bitfield used in various RSMI calls
 */
typedef uint64_t rsmi_bit_field_t;
/// \cond Ignore in docs.
typedef rsmi_bit_field_t rsmi_bit_field;
/// \endcond

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
 * @brief This structure holds error counts.
 */
typedef struct {
    uint64_t correctable_err;            //!< Accumulated correctable errors
    uint64_t uncorrectable_err;          //!< Accumulated uncorrectable errors
} rsmi_error_count_t;


/**
 *  @brief Initialize ROCm SMI.
 *
 *  @details When called, this initializes internal data structures,
 *  including those corresponding to sources of information that SMI provides.
 *
 *  @param[in] init_flags Bit flags that tell SMI how to initialze. Not
 *  currently used.
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
 *
 */
rsmi_status_t
rsmi_dev_pci_bandwidth_get(uint32_t dv_ind, rsmi_pcie_bandwidth_t *bandwidth);

/**
 * @brief Get percentage of time device is busy doing any processing
 *
 * @details Given a device index @p dv_ind, this function returns the
 * percentage of time that the specified device is busy. The device is
 * considered busy if any one or more of its sub-blocks are working, and idle
 * if none of the sub-blocks are working.
 *
 * @param[in] dv_ind a device index
 *
 * @param[inout] busy_percent a pointer to the uint32_t to which the busy
 * percent will be written
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call
 *
 */
rsmi_status_t
rsmi_dev_busy_percent_get(uint32_t dv_ind, uint32_t *busy_percent);

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
 */
rsmi_status_t rsmi_dev_pci_bandwidth_set(uint32_t dv_ind, uint64_t bw_bitmask);

/**
 *  @brief Get the unique PCI device identifier associated for a device
 *
 *  @details Give a device index @p dv_ind and a pointer to a uint64_t @p
 *  bdfid, this function will write the Bus/Device/Function PCI identifier
 *  (BDFID) associated with device @p dv_ind to the value pointed to by
 *  @p bdfid.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] bdfid a pointer to uint64_t to which the device bdfid value
 *  will be written
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.

 */
rsmi_status_t rsmi_dev_pci_id_get(uint32_t dv_ind, uint64_t *bdfid);

/**
 *  @brief Get PCIe traffic information
 *
 *  @details Give a device index @p dv_ind and pointers to a uint64_t's, @p
 *  sent, @p received and @p max_pkt_sz, this function will write the number
 *  of bytes sent and received in 1 second to @p sent and @p received,
 *  respectively. The maximum possible packet size will be written to
 *  @max_pkt_sz.
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
 */
rsmi_status_t rsmi_dev_pci_throughput_get(uint32_t dv_ind, uint64_t *sent,
                                    uint64_t *received, uint64_t *max_pkt_sz);

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
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t rsmi_dev_id_get(uint32_t dv_ind, uint64_t *id);


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
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t rsmi_dev_perf_level_get(uint32_t dv_ind,
                                                 rsmi_dev_perf_level_t *perf);

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
 *
 */
rsmi_status_t
rsmi_dev_perf_level_set(int32_t dv_ind, rsmi_dev_perf_level_t perf_lvl);

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
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t rsmi_dev_overdrive_level_get(uint32_t dv_ind, uint32_t *od);


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
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t rsmi_dev_overdrive_level_set(int32_t dv_ind, uint32_t od);

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
 *  to which the frequency information will be written
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t rsmi_dev_gpu_clk_freq_get(uint32_t dv_ind,
                             rsmi_clk_type_t clk_type, rsmi_frequencies_t *f);

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
 */
rsmi_status_t rsmi_dev_gpu_clk_freq_set(uint32_t dv_ind,
                             rsmi_clk_type_t clk_type, uint64_t freq_bitmask);
/**
 *  @brief Get the name of a gpu device.
 *
 *  @details Given a device index @p dv_ind, a pointer to a caller provided
 *  char buffer @p name, and a length of this buffer @p len, this function
 *  will write the name of the device (up to @p len characters) buffer @p name.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[inout] name a pointer to a caller provided char buffer to which the
 *  name will be written
 *
 *  @param[in] len the length of the caller provided buffer @p name.
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t rsmi_dev_name_get(uint32_t dv_ind, char *name, size_t len);

/**
 *  @brief Get the temperature metric value for the specified metric, from the
 *  specified temperature sensor on the specified device.
 *
 *  @details Given a device index @p dv_ind, a 0-based sensor index
 *  @p sensor_ind, a ::rsmi_temperature_metric_t @p metric and a pointer to an
 *  int64_t @p temperature, this function will write the value of the metric
 *  indicated by @p metric to the memory location @p temperature.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[in] metric enum indicated which temperature value should be
 *  retrieved
 *
 *  @param[inout] temperature a pointer to int64_t to which the temperature
 *  will be written, in millidegrees Celcius.
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t rsmi_dev_temp_metric_get(uint32_t dv_ind, uint32_t sensor_ind,
                      rsmi_temperature_metric_t metric, int64_t *temperature);

/**
 * @brief Retrieve the error counts for a GPU block
 * 
 * @details Given a device index @p dv_ind, an ::rsmi_gpu_block_t @p block and a
 * pointer to an ::rsmi_error_count_t @p ec, this function will write the error
 * count values for the GPU block indicated by @p block to memory pointed to by
 * @p ec.
 * 
 * @param[in] dv_ind a device index
 * 
 * @param[in] block The block for which error counts should be retrieved
 * 
 * @param[inout] ec A pointer to an ::rsmi_error_count_t to which the error
 * counts should be written
 * 
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 * 
 */
rsmi_status_t rsmi_dev_error_count_get(uint32_t dv_ind,
                              rsmi_gpu_block_t block, rsmi_error_count_t *ec);
/**
 * @brief Reset the fan to automatic driver control
 *
 * @details This function returns control of the fan to the system
 *
 * @param[in] dv_ind a device index
 *
 * @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 * If a device has more than one sensor, it could be greater than 0.
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 */
rsmi_status_t rsmi_dev_fan_reset(uint32_t dv_ind, uint32_t sensor_ind);

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
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t rsmi_dev_fan_rpms_get(uint32_t dv_ind, uint32_t sensor_ind,
                                                              int64_t *speed);

/**
 * @brief Get the fan speed for the specified device in RPMs.
 *
 * @details Given a device index @p dv_ind 
 * this function will get the fan speed.
 *
 * @param[in] dv_ind a device index
 *
 * @details Given a device index @p dv_ind and a pointer to a uint32_t
 * @p speed, this function will write the current fan speed (a value
 * between 0 and 255) to the uint32_t pointed to by @p speed
 *
 * @param[in] dv_ind a device index
 *
 * @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 * If a device has more than one sensor, it could be greater than 0.
 *
 * @param[inout] speed a pointer to uint32_t to which the speed will be
 * written
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
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
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t rsmi_dev_fan_speed_max_get(uint32_t dv_ind,
                                    uint32_t sensor_ind, uint64_t *max_speed);

/**
 * @brief Set the fan speed for the specified device with the provided speed,
 * in RPMs.
 *
 * @details Given a device index @p dv_ind and a integer value indicating
 * speed @p speed, this function will attempt to set the fan speed to @p speed.
 * An error will be returned if the specified speed is outside the allowable
 * range for the device. The maximum value is 255 and the minimum is 0.
 *
 * @param[in] dv_ind a device index
 *
 * @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 * If a device has more than one sensor, it could be greater than 0.
 *
 * @param[in] speed the speed to which the function will attempt to set the fan
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 */
rsmi_status_t rsmi_dev_fan_speed_set(uint32_t dv_ind, uint32_t sensor_ind,
                                                              uint64_t speed);

/**
 * @brief This function retrieves the voltage/frequency curve information
 *
 * @details Given a device index @p dv_ind and a pointer to a
 * ::rsmi_od_volt_freq_data_t structure @p odv, this function will populate @p
 * odv. See ::rsmi_od_volt_freq_data_t for more details.
 *
 * @param[in] dv_ind a device index
 *
 * @param[in] odv a pointer to an ::rsmi_od_volt_freq_data_t structure 
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 */
rsmi_status_t rsmi_dev_od_volt_info_get(uint32_t dv_ind,
                                               rsmi_od_volt_freq_data_t *odv);

/**
 * @brief Set the frequency limits for the specified clock
 *
 * @details Given a device index @p dv_ind, a clock type (::rsmi_clk_type_t)
 * @p clk, and a pointer to a ::rsmi_range_t @p range containing the desired
 * upper and lower frequency limits, this function will attempt to set the
 * frequency limits to those specified in @p range.
 *
 * @param[in] dv_ind a device index
 *
 * @param[in] clk The clock type for which the limits should be imposed.
 *
 * @param[in] range A pointer to the ::rsmi_range_t containing the desired limits
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 */
rsmi_status_t rsmi_dev_od_freq_range_set(uint32_t dv_ind, rsmi_clk_type_t clk,
                                                           rsmi_range_t *range);

/**
 * @brief This function will retrieve the current valid regions in the 
 * frequency/voltage space.
 *
 * @details Given a device index @p dv_ind, a pointer to an unsigned integer
 * @p num_regions and a buffer of ::rsmi_freq_volt_region_t structures, @p
 * buffer, this function will populate @p buffer with the current
 * frequency-volt space regions. The caller should assign @p buffer to memory
 * that can be written to by this function. The caller should also
 * indicate the number of ::rsmi_freq_volt_region_t structures that can safely
 * be written to @p buffer in @p num_regions.
 * 
 * The number of regions to expect this function provide (@p num_regions) can
 * be obtained by calling ::rsmi_dev_od_volt_info_get().
 *
 * @param[in] dv_ind a device index
 *
 * @param[inout] num_regions As input, this is the number of
 * ::rsmi_freq_volt_region_t structures that can be written to @p buffer. As
 * output, this is the number of ::rsmi_freq_volt_region_t structures that were
 * actually written.
 * 
 * @param[inout] buffer a caller provided buffer to which
 * ::rsmi_freq_volt_region_t structures will be written
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 */
rsmi_status_t rsmi_dev_od_volt_curve_regions_get(uint32_t dv_ind,
                      uint32_t *num_regions, rsmi_freq_volt_region_t *buffer);

/**
 *  @brief Get the average power consumption of the device with provided
 *  device index.
 *
 *  @details Given a device index @p dv_ind and a pointer to a uint64_t
 *  @p power, this function will write the current average power consumption to
 *  the uint64_t in microwatts pointed to by @p power. This function requires
 *  root privilege.
 *
 *  @param[in] dv_ind a device index
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[inout] power a pointer to uint64_t to which the average power
 *  consumption will be written
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t
rsmi_dev_power_ave_get(uint32_t dv_ind, uint32_t sensor_ind, uint64_t *power);

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
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t
rsmi_dev_power_cap_get(uint32_t dv_ind, uint32_t sensor_ind, uint64_t *cap);

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
 *
 *  @param[inout] min a pointer to a uint64_t that indicates the minimum
 *  possible power cap, in microwatts
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t
rsmi_dev_power_cap_range_get(uint32_t dv_ind, uint32_t sensor_ind,
                                                uint64_t *max, uint64_t *min);

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
 *  @param[inout] cap a uint64_t that indicates the desired power cap, in
 *  microwatts
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t
rsmi_dev_power_cap_set(uint32_t dv_ind, uint32_t sensor_ind, uint64_t cap);


/**
 * @brief Get the list of available preset power profiles and an indication of
 * which profile is currently active.
 *
 * @details Given a device index @p dv_ind and a pointer to a
 * ::rsmi_power_profile_status_t @p status, this function will set the bits of
 * the ::rsmi_power_profile_status_t.available_profiles bit field of @p status to
 * 1 if the profile corresponding to the respective
 * ::rsmi_power_profile_preset_masks_t profiles are enabled. For example, if both
 * the VIDEO and VR power profiles are available selections, then
 * ::RSMI_PWR_PROF_PRST_VIDEO_MASK AND'ed with
 * ::rsmi_power_profile_status_t.available_profiles will be non-zero as will
 * ::RSMI_PWR_PROF_PRST_VR_MASK AND'ed with
 * ::rsmi_power_profile_status_t.available_profiles. Additionally,
 * ::rsmi_power_profile_status_t.current will be set to the
 * ::rsmi_power_profile_preset_masks_t of the profile that is currently active.
 *
 * @param[in] dv_ind a device index
 *
 * @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 * If a device has more than one sensor, it could be greater than 0.
 *
 * @param[inout] status a pointer to ::rsmi_power_profile_status_t that will be
 * populated by a call to this function
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t
rsmi_dev_power_profile_presets_get(uint32_t dv_ind, uint32_t sensor_ind,
                                         rsmi_power_profile_status_t *status);

/**
 * @brief Set the power profile
 *
 * @details Given a device index @p dv_ind, a sensor index sensor_ind, and a
 * @p profile, this function will attempt to set the current profile to the
 * provided profile. The provided profile must be one of the currently
 * supported profiles, as indicated by a call to
 * ::rsmi_dev_power_profile_presets_get()
 *
 * @param[in] dv_ind a device index
 *
 * @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 * If a device has more than one sensor, it could be greater than 0.
 *
 * @param[in] profile a ::rsmi_power_profile_preset_masks_t that hold the mask
 * of the desired new power profile
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t
rsmi_dev_power_profile_set(uint32_t dv_ind, uint32_t sensor_ind,
                                   rsmi_power_profile_preset_masks_t profile);

/**
 * @brief Get the VBIOS identifer string
 *
 * @details Given a device ID @p dv_ind, and a pointer to a char buffer,
 * @p vbios, this function will write the VBIOS string (up to @p len
 * characters) for device @p dv_ind to @p vbios. The caller must ensure that
 * it is safe to write at least @p len characters to @p vbios.
 *
 * @param[in] dv_ind a device index
 *
 * @param[inout] vbios A pointer to a buffer of char's to which the VBIOS name
 * will be written
 *
 * @param[in] len The number of char's pointed to by @p vbios which can safely
 * be written to by this function.
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call.
 *
 */
rsmi_status_t
rsmi_dev_vbios_version_get(uint32_t dv_ind, char *vbios, uint32_t len);

/**
 * @brief Get a description of a provided RSMI error status
 *
 * @details Set the provided pointer to a const char *, @p status_string, to
 * a string containing a description of the provided error code @p status.
 *
 * @param[in] status The error status for which a description is desired
 *
 * @param[inout] status_string A pointer to a const char * which will be made
 * to point to a description of the provided error code
 *
 * @retval ::RSMI_STATUS_SUCCESS is returned upon successful call
 *
 */
rsmi_status_t
rsmi_status_string(rsmi_status_t status, const char **status_string);

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

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_H_
