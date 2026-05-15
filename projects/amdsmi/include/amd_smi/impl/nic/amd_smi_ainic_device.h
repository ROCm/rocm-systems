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

#pragma once

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_processor.h"
#include <memory>

// User could to get the AI_NIC processor using below existing function
// ret = amdsmi_get_processor_handles_by_type(sockets[i], AMDSMI_PROCESSOR_TYPE_AMD_NIC, nullptr, &tmp_device_count);
// Get the ai nic information using the nic handle

namespace amd::smi {

class AMDSmiAINICDevice: public AMDSmiProcessor {
 public:

/**
 * @brief Main NIC Information
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
    struct AINICInfo{
        amdsmi_nic_asic_info_t asic;
        amdsmi_nic_bus_info_t bus;
        amdsmi_nic_driver_info_t driver;
        amdsmi_nic_numa_info_t numa;
        amdsmi_nic_fw_t versions;
        amdsmi_nic_port_info_t port;
        amdsmi_nic_rdma_devices_info_t rdma_dev;
    };

    AMDSmiAINICDevice(uint32_t nic_idx, const amdsmi_bdf_t &bdf, const AINICInfo &ai_nic_info)
      : AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_NIC)
      , nic_idx_(nic_idx)
      , bdf_(bdf)
      , ai_nic_info_(ai_nic_info) {
    }
    ~AMDSmiAINICDevice() = default;
    amdsmi_status_t amd_query_nic_info(AINICInfo& info) const;
 private:
    uint32_t nic_idx_;
    amdsmi_bdf_t bdf_;
    AINICInfo ai_nic_info_;
};

}  // namespace amd::smi

amdsmi_status_t
amdsmi_get_ainic_info(amdsmi_processor_handle processor_handle, amd::smi::AMDSmiAINICDevice::AINICInfo *info);
