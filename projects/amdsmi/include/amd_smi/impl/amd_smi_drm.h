// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_DRM_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_DRM_H_

#include <libdrm/amdgpu_drm.h>
#include <unistd.h>

#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_lib_loader.h"
#include "amd_smi/impl/scoped_fd.h"
#include "amd_smi/impl/xf86drm.h"

namespace amd::smi {

class AMDSmiDrm {
 public:
  amdsmi_status_t init();
  amdsmi_status_t cleanup();
  amdsmi_status_t get_bdf_by_index(uint32_t gpu_index, amdsmi_bdf_t* bdf_info) const;
  amdsmi_status_t get_drm_path_by_index(uint32_t gpu_index, std::string* drm_path) const;
  std::vector<amdsmi_bdf_t> get_bdfs();
  std::vector<std::string>& get_drm_paths();
  bool check_if_drm_is_supported();

  amdsmi_status_t amdgpu_query_cpu_affinity(const std::string& device_path,
                                            std::string& cpu_affinity);

  uint32_t get_vendor_id();

 private:
  // when file is not found, the empty string will be returned
  std::string find_file_in_folder(const std::string& folder, const std::string& regex);
  std::vector<std::string> drm_paths_;  // drm path (renderD128 for example)
  std::vector<amdsmi_bdf_t> drm_bdfs_;  // bdf
  uint32_t vendor_id;

  AMDSmiLibraryLoader lib_loader_;  // lazy load libdrm

  std::mutex drm_mutex_;
};

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_DRM_H_
