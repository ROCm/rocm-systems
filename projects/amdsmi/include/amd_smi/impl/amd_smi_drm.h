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

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_DRM_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_DRM_H_

#include <unistd.h>

#include <vector>
#include <memory>
#include <mutex>  // NOLINT
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_lib_loader.h"
#include "amd_smi/impl/amdgpu_drm.h"
#include "amd_smi/impl/xf86drm.h"
#include "amd_smi/impl/scoped_fd.h"

namespace amd::smi {

class AMDSmiDrm {
 public:
    amdsmi_status_t init();
    amdsmi_status_t cleanup();
    amdsmi_status_t get_bdf_by_index(uint32_t gpu_index, amdsmi_bdf_t *bdf_info) const;
    amdsmi_status_t get_drm_path_by_index(uint32_t gpu_index, std::string *drm_path) const;
    std::vector<amdsmi_bdf_t> get_bdfs();
    std::vector<std::string>& get_drm_paths();
    bool check_if_drm_is_supported();
    uint32_t get_vendor_id();

 private:
    // when file is not found, the empty string will be returned
    std::string find_file_in_folder(const std::string& folder,
                  const std::string& regex);
    std::vector<std::string> drm_paths_;  // drm path (renderD128 for example)
    std::vector<amdsmi_bdf_t> drm_bdfs_;  // bdf
    uint32_t vendor_id;

    AMDSmiLibraryLoader lib_loader_;  // lazy load libdrm

    std::mutex drm_mutex_;
};


} // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_DRM_H_
