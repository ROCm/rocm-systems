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

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_DEVICE_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_DEVICE_H_

#include <map>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_processor.h"
#include "amd_smi/impl/amd_smi_drm.h"

namespace amd::smi {


// PID, amdsmi_proc_info_t
using GPUComputeProcessList_t = std::map<amdsmi_process_handle_t, amdsmi_proc_info_t>;
using ComputeProcessListClassType_t = uint16_t;

enum class ComputeProcessListType_t : ComputeProcessListClassType_t
{
    kAllProcesses,
    kAllProcessesOnDevice,
};


class AMDSmiGPUDevice: public AMDSmiProcessor {

 public:
    AMDSmiGPUDevice(uint32_t gpu_id, std::string path, amdsmi_bdf_t bdf, AMDSmiDrm& drm):
            AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_GPU), gpu_id_(gpu_id), path_(path), bdf_(bdf), drm_(drm) {}

    AMDSmiGPUDevice(uint32_t gpu_id, AMDSmiDrm& drm):
            AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_GPU), gpu_id_(gpu_id), drm_(drm) {
                if (check_if_drm_is_supported()) this->get_drm_data();
            }
    ~AMDSmiGPUDevice() {
    }

    amdsmi_status_t get_drm_data();
    pthread_mutex_t* get_mutex();
    uint32_t get_gpu_id() const;
    uint32_t get_card_id();            // -e feature + we can get card_id for our internal functions
    uint32_t get_drm_render_minor();   // -e feature + we can get card_id for our internal functions
    uint64_t get_kfd_gpu_id();  // Used to decode vram usage for KFD processes
    std::string& get_gpu_path();
    amdsmi_bdf_t  get_bdf();
    bool check_if_drm_is_supported() { return drm_.check_if_drm_is_supported(); }
    uint32_t get_vendor_id();
    const GPUComputeProcessList_t& amdgpu_get_compute_process_list(ComputeProcessListType_t list_type = ComputeProcessListType_t::kAllProcessesOnDevice);


// New methods for -e feature
    std::string bdf_to_string() const;     // -e feature
    std::vector<uint64_t> get_bitmask_from_numa_node(int32_t node_id, uint32_t size) const;

 private:
    uint32_t gpu_id_;
    std::string path_;
    amdsmi_bdf_t bdf_;
    uint32_t vendor_id_;
    AMDSmiDrm& drm_;
    uint32_t card_index_;
    uint32_t drm_render_minor_;
    uint64_t kfd_gpu_id_;  // Used to decode vram usage for KFD processes
    GPUComputeProcessList_t compute_process_list_;
    int32_t get_compute_process_list_impl(GPUComputeProcessList_t& compute_process_list,
                                          ComputeProcessListType_t list_type);

};


} // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_DEVICE_H_
