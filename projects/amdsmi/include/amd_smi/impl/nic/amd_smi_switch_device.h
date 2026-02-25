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

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_SWITCH_DEVICE_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_SWITCH_DEVICE_H_

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_processor.h"
#include "amd_smi/impl/nic/amd_smi_no_drm_switch.h"
#include "shared_mutex.h"  // NOLINT
#include "rocm_smi/rocm_smi_logger.h"

namespace amd::smi {

class AMDSmiSWITCHDevice: public AMDSmiProcessor {
 public:

    AMDSmiSWITCHDevice(uint32_t switch_id, amdsmi_bdf_t bdf, AMDSmiNoDrmSwitch& no_drm_switch)
                : AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_BRCM_SWITCH), switch_id_(switch_id), bdf_(bdf), nodrm_(no_drm_switch) {
              if (check_if_no_drm_is_supported()) this->get_no_drm_data();
            }

    ~AMDSmiSWITCHDevice() = default;

    amdsmi_status_t get_no_drm_data();
    pthread_mutex_t* get_mutex();
    uint32_t get_switch_id() const;
    std::string& get_switch_path();
    amdsmi_bdf_t get_bdf();
    bool check_if_no_drm_is_supported() { return nodrm_.check_if_no_drm_is_supported(); }

    amdsmi_status_t amd_query_switch_link_info(amdsmi_brcm_switch_link_metric_t& info) const;
    amdsmi_status_t amd_query_switch_uuid(std::string& serial) const;
    amdsmi_status_t amd_query_switch_numa_affinity(int32_t *numa_node) const;
    amdsmi_status_t amd_query_switch_cpu_affinity(std::string& cpu_affinity) const;
    amdsmi_status_t amd_query_switch_device_info(amdsmi_brcm_switch_device_metric_t& info) const;
    amdsmi_status_t amd_query_switch_power_info(amdsmi_brcm_switch_power_metric_t& info) const;

 private:
    uint32_t switch_id_;
    std::string path_;
    amdsmi_bdf_t bdf_;
    AMDSmiNoDrmSwitch& nodrm_;
};

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_SWITCH_DEVICE_H_
