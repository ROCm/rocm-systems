/*
 * Copyright (c) Broadcom Inc All Rights Reserved.
 *
 *  Developed by:
 *            Broadcom Inc
 *
 *            www.broadcom.com
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

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_NO_DRM_NIC_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_NO_DRM_NIC_H_

#include <unistd.h>
#include <vector>
#include <memory>
#include <mutex>  // NOLINT
#include "amd_smi/amdsmi.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace amd::smi {

class AMDSmiNoDrmNIC {
 public:
    amdsmi_status_t init();
    amdsmi_status_t cleanup();
    amdsmi_status_t get_bdf_by_index(uint32_t nic_index, amdsmi_bdf_t *bdf_info) const;
    amdsmi_status_t get_interface_name_by_index(uint32_t nic_index, std::string* interface_name) const;
    amdsmi_status_t get_device_path_by_index(uint32_t nic_index, std::string* device_path) const;
    amdsmi_status_t get_hwmon_path_by_index(uint32_t nic_index, std::string* hwm_path) const;
    std::vector<amdsmi_bdf_t> get_bdfs();
    std::vector<std::string>& get_device_paths();
    std::vector<std::string>& get_hwmon_paths();
    bool check_if_no_drm_is_supported();

   uint32_t get_vendor_id();
    amdsmi_status_t amd_query_nic_info(uint32_t nic_index, amdsmi_brcm_nic_info_t& info);
    amdsmi_status_t amd_query_nic_uuid(std::string devicePath, std::string& version);
    amdsmi_status_t amd_query_nic_temp(std::string hwmonPath, amdsmi_brcm_nic_temperature_metric_t& info);
    amdsmi_status_t amd_query_nic_device(std::string hwmonPath, amdsmi_brcm_nic_hwmon_device_t& info);
    amdsmi_status_t amd_query_nic_power(std::string hwmonPath, amdsmi_brcm_nic_hwmon_power_t& info);
    amdsmi_status_t amd_query_nic_numa_affinity(std::string devicePath, int32_t *numa_node);
    amdsmi_status_t amd_query_nic_cpu_affinity(std::string devicePath, std::string& cpu_affinity);

    amdsmi_status_t amd_query_nic_fw_info(std::string devicePath, amdsmi_brcm_nic_firmware_t& info);
 private:
    // when file is not found, the empty string will be returned
    std::vector<std::string> device_paths_;
    std::vector<std::string> hwmon_paths_;
    std::vector<std::string> interfaces_;
    std::vector<amdsmi_bdf_t> no_drm_bdfs_;
};

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_NO_DRM_NIC_H_
