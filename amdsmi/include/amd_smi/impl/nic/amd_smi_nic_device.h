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

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_NIC_DEVICE_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_NIC_DEVICE_H_

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_processor.h"
#include "amd_smi/impl/nic/amd_smi_no_drm_nic.h"
#include "shared_mutex.h"  // NOLINT
#include "rocm_smi/rocm_smi_logger.h"

namespace amd::smi {

class AMDSmiNICDevice: public AMDSmiProcessor {
 public:

    AMDSmiNICDevice(uint32_t nic_id, amdsmi_bdf_t bdf, AMDSmiNoDrmNIC& no_drm_nic)
      : AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_BRCM_NIC), nic_id_(nic_id), bdf_(bdf), nodrm_(no_drm_nic) {
              if (check_if_no_drm_is_supported()) this->get_no_drm_data();
            }

    ~AMDSmiNICDevice() = default;

    amdsmi_status_t get_no_drm_data();
    pthread_mutex_t* get_mutex();
    uint32_t get_nic_id() const;
    std::string& get_nic_path();
    amdsmi_bdf_t get_bdf();
    bool check_if_no_drm_is_supported() { return nodrm_.check_if_no_drm_is_supported(); }
    uint32_t get_vendor_id();

    amdsmi_status_t amd_query_nic_info(amdsmi_brcm_nic_info_t& info) const;
    amdsmi_status_t amd_query_nic_temp_info(amdsmi_brcm_nic_temperature_metric_t& info) const;
    amdsmi_status_t amd_query_nic_device_info(amdsmi_brcm_nic_hwmon_device_t& info) const;
    amdsmi_status_t amd_query_nic_power_info(amdsmi_brcm_nic_hwmon_power_t& info) const;
    amdsmi_status_t amd_query_nic_uuid(std::string& version) const;
    amdsmi_status_t amd_query_nic_numa_affinity(int32_t *numa_node) const;
    amdsmi_status_t amd_query_nic_cpu_affinity(std::string& cpu_affinity) const;

    amdsmi_status_t amd_query_nic_firmware_info(amdsmi_brcm_nic_firmware_t& info) const;

 private:
    uint32_t nic_id_;
    std::string path_;
    amdsmi_bdf_t bdf_;
    AMDSmiNoDrmNIC& nodrm_;
};

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_NIC_DEVICE_H_
