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

#ifndef AMD_SMI_INCLUDE_AMD_SMI_UTILS_H_
#define AMD_SMI_INCLUDE_AMD_SMI_UTILS_H_

#include <limits>
#include <type_traits>
#include <string>
#include <utility>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"


#define SMIGPUDEVICE_MUTEX(MUTEX) \
    amd::smi::pthread_wrap _pw(*(MUTEX)); \
    amd::smi::ScopedPthread _lock(_pw, true); \
    if (_lock.mutex_not_acquired()) { \
      return AMDSMI_STATUS_BUSY; \
    }

amdsmi_status_t smi_amdgpu_find_hwmon_dir(amd::smi::AMDSmiGPUDevice* device, std::string* full_path);
amdsmi_status_t smi_amdgpu_get_board_info(amd::smi::AMDSmiGPUDevice* device, amdsmi_board_info_t *info);
amdsmi_status_t smi_amdgpu_get_power_cap(amd::smi::AMDSmiGPUDevice* device, int *cap);
amdsmi_status_t smi_amdgpu_get_ranges(amd::smi::AMDSmiGPUDevice* device, amdsmi_clk_type_t domain, int *max_freq, int *min_freq, int *num_dpm, int *sleep_state_freq);
amdsmi_status_t smi_amdgpu_get_enabled_blocks(amd::smi::AMDSmiGPUDevice* device, uint64_t *enabled_blocks);
amdsmi_status_t smi_amdgpu_get_bad_page_info(amd::smi::AMDSmiGPUDevice* device, uint32_t *num_pages, amdsmi_retired_page_record_t *info);
amdsmi_status_t smi_amdgpu_get_bad_page_threshold(amd::smi::AMDSmiGPUDevice* device, uint32_t *threshold);
amdsmi_status_t smi_amdgpu_validate_ras_eeprom(amd::smi::AMDSmiGPUDevice* device);
amdsmi_status_t smi_amdgpu_get_ecc_error_count(amd::smi::AMDSmiGPUDevice* device, amdsmi_error_count_t *err_cnt);
amdsmi_status_t smi_amdgpu_get_driver_version(amd::smi::AMDSmiGPUDevice* device, int *length, char *version);
amdsmi_status_t smi_amdgpu_get_pcie_speed_from_pcie_type(uint16_t pcie_type, uint32_t *pcie_speed);
amdsmi_status_t smi_amdgpu_get_market_name_from_dev_id(amd::smi::AMDSmiGPUDevice* device, char *market_name);
amdsmi_status_t smi_amdgpu_is_gpu_power_management_enabled(amd::smi::AMDSmiGPUDevice* device, bool *enabled);
std::string smi_split_string(std::string str, char delim);
std::string smi_amdgpu_get_status_string(amdsmi_status_t ret, bool fullStatus);
amdsmi_status_t smi_clear_char_and_reinitialize(char buffer[], uint32_t len,
                                                    std::string newString);
amdsmi_status_t amdsmi_get_gpu_cper_entries_by_path(const char *amdgpu_ring_cper_file, uint32_t severity_mask, char *cper_data, uint64_t *buf_size, amdsmi_cper_hdr_t **cper_hdrs, uint64_t *entry_count, uint64_t *cursor);

/**
 *  @brief Get the device index given the processor handle.
 *
 *  @details Given a processor handle @p processor_handle 
 *  and a pointer to a uint32_t @p device_index will be returned.
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[inout] device_index a pointer to uint32_t to which the matching device
 *  index will be stored
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS is returned upon successful call.
 *          ::AMDSMI_STATUS_INVAL is returned if user provides a null pointer
 *          for device_index.
 *          ::AMDSMI_STATUS_API_FAILED is returned if the corresponding device
 *          index for the processor handle cannot be found.
 */
amdsmi_status_t smi_amdgpu_get_device_index(amdsmi_processor_handle processor_handle,
                                            uint32_t* device_index);

/**
 *  @brief Get total number of devices
 *
 *  @details Given a pointer to a uint32_t @p total_num_devices will be returned
 *
 *  @param[inout] total_num_devices a pointer to uint32_t to which the total number
 *  of devices will be stored
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS is returned upon successful call.
 *          ::AMDSMI_STATUS_INVAL is returned if user provides a null pointer
 *          for total_num_devices.
 */
amdsmi_status_t smi_amdgpu_get_device_count(uint32_t *total_num_devices);

/**
 *  @brief Get the processor handle given the device index.
 *
 *  @details Given a uint32_t @p device_index and a pointer to
 *  a processor handle @p processor_handle, the device index will be used to
 *  find the processor handle of the device and store it in the provided pointer
 *
 *  @param[in] device_index a uint32_t to value to help find the corresponding
 *  processor handle
 *
 *  @param[inout] processor_handle a pointer to amdsmi_processor_handle
 *  which the corresponding processor_handle will be stored
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS is returned upon successful call.
 *          ::AMDSMI_STATUS_INVAL is returned if user provides a null pointer
 *          for processor_handle.
 *          ::AMDSMI_STATUS_API_FAILED is returned if the device_index is cannot
 *          be found.
 */
amdsmi_status_t smi_amdgpu_get_processor_handle_by_index(
                                        uint32_t device_index,
                                        amdsmi_processor_handle *processor_handle);


template<typename>
constexpr bool is_dependent_false_v = false;

template<typename T>
inline constexpr bool is_supported_type_v = (
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::uint8_t>  ||
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::uint16_t> ||
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::uint32_t> ||
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::uint64_t>
);

template<typename T>
constexpr T get_std_num_limit()
{
    if constexpr (is_supported_type_v<T>) {
        return std::numeric_limits<T>::max();
    } else {
        return std::numeric_limits<T>::min();
        static_assert(is_dependent_false_v<T>, "Error: Type not supported...");
    }
}

template<typename T>
constexpr bool is_std_num_limit(T value)
{
    return (value == get_std_num_limit<T>());
}

template<typename T, typename U,  typename V = T>
constexpr T translate_umax_or_assign_value(U source_value, V target_value)
{
    T result{};
    if constexpr (is_supported_type_v<T> && is_supported_type_v<U>) {
        // If the source value is uint<U>::max(), then return is uint<T>::max()
        if (is_std_num_limit(source_value)) {
            result = get_std_num_limit<T>();
        } else {
            result = static_cast<T>(target_value);
        }

        return result;
    } else {
        static_assert(is_dependent_false_v<T>, "Error: Type not supported...");
    }

    return result;
}

#endif  // AMD_SMI_INCLUDE_AMD_SMI_UTILS_H_
