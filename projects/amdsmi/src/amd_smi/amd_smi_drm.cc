/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2024, Advanced Micro Devices, Inc.
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
#include <sys/types.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <memory>
#include <regex>
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amdgpu_drm.h"
#include "amd_smi/impl/amd_smi_common.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace amd {
namespace smi {

std::string AMDSmiDrm::find_file_in_folder(const std::string& folder,
               const std::string& regex) {
    std::string file_name;
    using dir_ptr = std::unique_ptr<DIR, decltype(&closedir)>;

    struct dirent *dir = nullptr;
    std::regex file_regex(regex);
    auto drm_dir = dir_ptr(opendir(folder.c_str()), &closedir);
    if (drm_dir == nullptr) return file_name;
    std::cmatch m;
    while ((dir = readdir(drm_dir.get())) != NULL) {
        if (std::regex_search(dir->d_name, m, file_regex)) {
            file_name = dir->d_name;
            break;
        }
    }
    return file_name;
}

amdsmi_status_t AMDSmiDrm::init() {
    // A few RAII handler
    using drm_version_ptr = std::unique_ptr<drmVersion,
            decltype(&drmFreeVersion)>;
    // using drm_device_ptr = std::unique_ptr(drmDevicePtr,
    //         decltype(&drmFreeDevice));

    struct dirent *dir = nullptr;
    int fd = -1;


    amdsmi_status_t status = lib_loader_.load("libdrm.so.2");
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    // load symbol from libdrm
    drm_cmd_write_ = nullptr;
    status = lib_loader_.load_symbol(&drm_cmd_write_, "drmCommandWrite");
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    using drmGetDeviceType = int(*)(int, drmDevicePtr*);   // drmGetDevice
    using drmFreeDeviceType = void(*)(drmDevicePtr*);     // drmFreeDevice

    drmGetDeviceType drm_get_device = nullptr;
    drmFreeDeviceType drm_free_device = nullptr;
    drm_get_version_ = nullptr;
    drm_free_version_ = nullptr;

    status = lib_loader_.load_symbol(&drm_get_version_, "drmGetVersion");
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }
    status = lib_loader_.load_symbol(&drm_free_version_, "drmFreeVersion");
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    status = lib_loader_.load_symbol(&drm_get_device, "drmGetDevice");
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }
    status = lib_loader_.load_symbol(&drm_free_device, "drmFreeDevice");
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    /* Need to map the /dev/dri/render* file to /sys/class/drm/card*
       The former is for drm fd and the latter is used for rocm-smi gpu index.
       Here it will search the /sys/class/drm/card0/../renderD128
    */
    amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();
    auto devices = smi.devices();

    bool has_valid_fds = false;
    for (uint32_t i=0; i < devices.size(); i++) {
        auto rocm_smi_device = devices[i];
        std::string render_file_name;
        drmDevicePtr device;

        const std::string regex("renderD([0-9]+)");
        const std::string renderD_folder = "/sys/class/drm/card"
                    + std::to_string(rocm_smi_device->index()) + "/../";

        // looking for /sys/class/drm/card0/../renderD*
        std::string render_name = find_file_in_folder(renderD_folder, regex);
        fd = -1;
        std::string name = "/dev/dri/" + render_name;
        if (render_name != "") {
            fd = open(name.c_str(), O_RDWR | O_CLOEXEC);
        }

        amdsmi_bdf_t bdf;
        if (fd >= 0) {
            auto version = drm_version_ptr(
                drm_get_version_(fd), drm_free_version_);
            if (strcmp("amdgpu", version->name)) {  // only amdgpu
                close(fd);
                fd = -1;
            }
            if (fd  >= 0 && drm_get_device(fd, &device) != 0) {
                drm_free_device(&device);
                close(fd);
                fd = -1;
            }
        }

        drm_fds_.push_back(fd);
        drm_paths_.push_back(render_name);
        // even if fail, still add to prevent mismatch the index
        if (fd < 0) {
            drm_bdfs_.push_back(bdf);
            continue;
        }

        has_valid_fds = true;
        std::ostringstream ss;
        uint64_t bdf_rocm = 0;
        rsmi_dev_pci_id_get(i, &bdf_rocm);
        ss << __PRETTY_FUNCTION__ << " | "
           << "bdf_rocm | Received bdf: "
           << "\nWhole BDF: " << amd::smi::print_unsigned_hex_and_int(bdf_rocm)
           << "\nDomain = "
           << amd::smi::print_unsigned_hex_and_int((bdf_rocm & 0xFFFFFFFF00000000) >> 32)
           << "; \nBus# = " << amd::smi::print_unsigned_hex_and_int((bdf_rocm & 0xFF00) >> 8)
           << "; \nDevice# = "<< amd::smi::print_unsigned_hex_and_int((bdf_rocm & 0xF8) >> 3)
           << "; \nFunction# = " << amd::smi::print_unsigned_hex_and_int((bdf_rocm & 0x7));
        LOG_INFO(ss);
        bdf.function_number = ((bdf_rocm & 0x7));
        bdf.device_number = ((bdf_rocm & 0xF8) >> 3);
        bdf.bus_number = ((bdf_rocm & 0xFF00) >> 8);
        bdf.domain_number = ((bdf_rocm & 0xFFFFFFFF00000000) >> 32);
        ss << __PRETTY_FUNCTION__ << " | " << "Received bdf: Domain = " << bdf.domain_number
           << "; Bus# = " << bdf.bus_number << "; Device# = "<< bdf.device_number
           << "; Function# = " << bdf.function_number;
        LOG_INFO(ss);

        vendor_id = device->deviceinfo.pci->vendor_id;

        drm_bdfs_.push_back(bdf);
        drm_free_device(&device);
    }

    // cannot find any valid fds.
    if (!has_valid_fds) {
        drm_bdfs_.clear();
        return AMDSMI_STATUS_INIT_ERROR;
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiDrm::cleanup() {
    for (unsigned int i=0; i < drm_fds_.size(); i++) {
        close(drm_fds_[i]);
    }

    drm_fds_.clear();
    drm_paths_.clear();
    drm_bdfs_.clear();
    lib_loader_.unload();
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiDrm::amdgpu_query_driver_name(int fd, std::string& driver_name) {
    // RAII handler
    using drm_version_ptr = std::unique_ptr<drmVersion,
            decltype(&drmFreeVersion)>;
    std::lock_guard<std::mutex> guard(drm_mutex_);
    auto version = drm_version_ptr(
                drm_get_version_(fd), drm_free_version_);
    if (version == nullptr) return AMDSMI_STATUS_DRM_ERROR;
    driver_name = version->name;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiDrm::amdgpu_query_driver_date(int fd, std::string& driver_date) {
    // RAII handler
    using drm_version_ptr = std::unique_ptr<drmVersion,
            decltype(&drmFreeVersion)>;
    std::lock_guard<std::mutex> guard(drm_mutex_);
    auto version = drm_version_ptr(
                drm_get_version_(fd), drm_free_version_);
    if (version == nullptr) return AMDSMI_STATUS_DRM_ERROR;
    driver_date = version->date;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiDrm::amdgpu_query_info(int fd, unsigned info_id,
            unsigned size, void *value) {
    if (drm_cmd_write_ == nullptr) return AMDSMI_STATUS_NOT_SUPPORTED;
    std::lock_guard<std::mutex> guard(drm_mutex_);

    struct drm_amdgpu_info request;
    memset(&request, 0, sizeof(request));
    request.return_pointer = (uintptr_t)value;
    request.return_size = size;
    request.query = info_id;
    int status = drm_cmd_write_(fd, DRM_AMDGPU_INFO,
            &request, sizeof(struct drm_amdgpu_info));
    if (status == 0) return AMDSMI_STATUS_SUCCESS;
    return AMDSMI_STATUS_DRM_ERROR;
}

amdsmi_status_t AMDSmiDrm::amdgpu_query_fw(int fd, unsigned info_id,
        unsigned fw_type, unsigned size, void *value) {
    if (drm_cmd_write_ == nullptr) return AMDSMI_STATUS_NOT_SUPPORTED;

    std::lock_guard<std::mutex> guard(drm_mutex_);

    struct drm_amdgpu_info request;
    memset(&request, 0, sizeof(request));
    request.return_pointer = (uintptr_t)value;
    request.return_size = size;
    request.query = info_id;
    request.query_fw.fw_type = fw_type;
    int status = drm_cmd_write_(fd, DRM_AMDGPU_INFO, &request,
                    sizeof(struct drm_amdgpu_info));
    if (status == 0) return AMDSMI_STATUS_SUCCESS;
    return AMDSMI_STATUS_DRM_ERROR;
}

amdsmi_status_t AMDSmiDrm::amdgpu_query_hw_ip(int fd, unsigned info_id,
        unsigned hw_ip_type, unsigned size, void *value) {
    if (drm_cmd_write_ == nullptr) return AMDSMI_STATUS_NOT_SUPPORTED;

    std::lock_guard<std::mutex> guard(drm_mutex_);

    struct drm_amdgpu_info request;
    memset(&request, 0, sizeof(request));
    request.return_pointer = (uintptr_t)value;
    request.return_size = size;
    request.query = info_id;
    request.query_hw_ip.type = hw_ip_type;
    int status = drm_cmd_write_(fd, DRM_AMDGPU_INFO, &request,
                sizeof(struct drm_amdgpu_info));
    if (status == 0) return AMDSMI_STATUS_SUCCESS;
    return AMDSMI_STATUS_DRM_ERROR;
}

amdsmi_status_t AMDSmiDrm::amdgpu_query_vbios(int fd, void *info) {
    if (drm_cmd_write_ == nullptr) return AMDSMI_STATUS_NOT_SUPPORTED;

    std::lock_guard<std::mutex> guard(drm_mutex_);

    struct drm_amdgpu_info request;
    memset(&request, 0, sizeof request);
    request.return_pointer = (uint64_t) info;
    request.return_size = sizeof(drm_amdgpu_info_vbios);
    request.query = AMDGPU_INFO_VBIOS;
    request.vbios_info.type = AMDGPU_INFO_VBIOS_INFO;
    int status = drm_cmd_write_(fd, DRM_AMDGPU_INFO, &request,
                    sizeof(struct drm_amdgpu_info));
    if (status == 0) return AMDSMI_STATUS_SUCCESS;
    return AMDSMI_STATUS_DRM_ERROR;
}


amdsmi_status_t AMDSmiDrm::get_drm_fd_by_index(uint32_t gpu_index, uint32_t *fd_info) const {
    if (gpu_index + 1 > drm_fds_.size()) return AMDSMI_STATUS_NOT_SUPPORTED;
    if (drm_fds_[gpu_index] < 0 ) return AMDSMI_STATUS_NOT_SUPPORTED;
    *fd_info = drm_fds_[gpu_index];
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiDrm::get_bdf_by_index(uint32_t gpu_index, amdsmi_bdf_t *bdf_info) const {
    if (gpu_index + 1 > drm_bdfs_.size()) return AMDSMI_STATUS_NOT_SUPPORTED;
    *bdf_info = drm_bdfs_[gpu_index];
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | gpu_index = " << gpu_index
    << "; \nreceived bdf: Domain = " << bdf_info->domain_number
    << "; \nBus# = " << bdf_info->bus_number
    << "; \nDevice# = " << bdf_info->device_number
    << "; \nFunction# = " << bdf_info->function_number
    << "\nReturning = AMDSMI_STATUS_SUCCESS";
    LOG_INFO(ss);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiDrm::get_drm_path_by_index(uint32_t gpu_index, std::string *drm_path) const {
    if (gpu_index + 1 > drm_paths_.size()) return AMDSMI_STATUS_NOT_SUPPORTED;
    *drm_path = drm_paths_[gpu_index];
    return AMDSMI_STATUS_SUCCESS;
}

std::vector<std::string>& AMDSmiDrm::get_drm_paths() {
    return drm_paths_;
}

bool AMDSmiDrm::check_if_drm_is_supported() {
    return (drm_cmd_write_ != NULL && drm_bdfs_.size() >0) ? true : false;
}

std::vector<amdsmi_bdf_t> AMDSmiDrm::get_bdfs() {
    return drm_bdfs_;
}

uint32_t AMDSmiDrm::get_vendor_id() {
    return vendor_id;
}

}  // namespace smi
}  // namespace amd

