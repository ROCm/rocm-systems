/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2023, Advanced Micro Devices, Inc.
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

#include <functional>
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "rocm_smi/rocm_smi_utils.h"


namespace amd {
namespace smi {

uint32_t AMDSmiGPUDevice::get_gpu_id() const {
    return gpu_id_;
}

uint32_t AMDSmiGPUDevice::get_gpu_fd() const {
    return fd_;
}

std::string& AMDSmiGPUDevice::get_gpu_path() {
    return path_;
}

amdsmi_bdf_t AMDSmiGPUDevice::get_bdf() {
    return bdf_;
}

uint32_t AMDSmiGPUDevice::get_vendor_id() {
    return vendor_id_;
}

amdsmi_status_t AMDSmiGPUDevice::get_drm_data() {
    amdsmi_status_t ret;
    uint32_t fd = 0;
    std::string path;
    amdsmi_bdf_t bdf;
    ret = drm_.get_drm_fd_by_index(gpu_id_, &fd);
    if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;
    ret = drm_.get_drm_path_by_index(gpu_id_, &path);
    if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;
    ret = drm_.get_bdf_by_index(gpu_id_, &bdf);
    if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;

    bdf_ = bdf, path_ = path, fd_ = fd;
    vendor_id_ = drm_.get_vendor_id();

    return AMDSMI_STATUS_SUCCESS;
}

pthread_mutex_t* AMDSmiGPUDevice::get_mutex() {
    return amd::smi::GetMutex(gpu_id_);
}

amdsmi_status_t AMDSmiGPUDevice::amdgpu_query_info(unsigned info_id,
                    unsigned size, void *value) const {
    amdsmi_status_t ret;
    uint32_t fd = 0;
    ret = drm_.get_drm_fd_by_index(gpu_id_, &fd);
    if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;

    return drm_.amdgpu_query_info(fd, info_id, size, value);
}

amdsmi_status_t AMDSmiGPUDevice::amdgpu_query_driver_name(std::string& name) const {
    amdsmi_status_t ret;
    uint32_t fd = 0;
    ret = drm_.get_drm_fd_by_index(gpu_id_, &fd);
    if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;

    return drm_.amdgpu_query_driver_name(fd, name);
}

amdsmi_status_t AMDSmiGPUDevice::amdgpu_query_driver_date(std::string& date) const {
    amdsmi_status_t ret;
    uint32_t fd = 0;
    ret = drm_.get_drm_fd_by_index(gpu_id_, &fd);
    if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;

    return drm_.amdgpu_query_driver_date(fd, date);
}

amdsmi_status_t AMDSmiGPUDevice::amdgpu_query_hw_ip(unsigned info_id,
            unsigned hw_ip_type, unsigned size, void *value) const {
    amdsmi_status_t ret;
    uint32_t fd = 0;
    ret = drm_.get_drm_fd_by_index(gpu_id_, &fd);
    if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;

    return drm_.amdgpu_query_hw_ip(fd, info_id, hw_ip_type, size, value);
}

amdsmi_status_t AMDSmiGPUDevice::amdgpu_query_fw(unsigned info_id,
        unsigned fw_type, unsigned size, void *value) const {
    amdsmi_status_t ret;
    uint32_t fd = 0;
    ret = drm_.get_drm_fd_by_index(gpu_id_, &fd);
    if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;

    return drm_.amdgpu_query_fw(fd, info_id, fw_type, size, value);
}

amdsmi_status_t AMDSmiGPUDevice::amdgpu_query_vbios(void *info) const {
    amdsmi_status_t ret;
    uint32_t fd = 0;
    ret = drm_.get_drm_fd_by_index(gpu_id_, &fd);
    if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;;

    return drm_.amdgpu_query_vbios(fd, info);
}

}  // namespace smi
}  // namespace amd

