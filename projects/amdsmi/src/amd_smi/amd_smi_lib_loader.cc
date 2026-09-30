// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amd_smi/impl/amd_smi_lib_loader.h"

#include <iostream>

#include "config/amd_smi_config.h"

namespace amd::smi {

std::vector<const char*> libdrm_amdgpu_sonames() {
  return {LIBDRM_AMDGPU_SONAME, "libdrm_amdgpu.so", "librocm_sysdeps_drm_amdgpu.so.1"};
}

AMDSmiLibraryLoader::AMDSmiLibraryLoader() : libHandler_(nullptr) {}

amdsmi_status_t AMDSmiLibraryLoader::load(const char* filename) {
  if (filename == nullptr) {
    return AMDSMI_STATUS_FAIL_LOAD_MODULE;
  }
  return load(std::vector<const char*>{filename});
}

amdsmi_status_t AMDSmiLibraryLoader::load(const std::vector<const char*>& filenames) {
  if (filenames.empty()) {
    return AMDSMI_STATUS_FAIL_LOAD_MODULE;
  }
  if (libHandler_ || library_loaded_) {
    unload();
  }

  std::lock_guard<std::mutex> guard(library_mutex_);
  std::string errors;
  for (const char* filename : filenames) {
    if (filename == nullptr) {
      continue;
    }
    // check if already loaded, return success if it is
    // dlopen(filename, RTLD_NOLOAD) == null only IFF library is not loaded
    void* isLibOpen = dlopen(filename, RTLD_NOLOAD);
    if (isLibOpen != nullptr) {
      // Keep the handle so load_symbol() resolves and unload() balances the
      // refcount that RTLD_NOLOAD incremented.
      libHandler_ = isLibOpen;
      library_loaded_ = true;
      return AMDSMI_STATUS_SUCCESS;
    }
    libHandler_ = dlopen(filename, RTLD_LAZY);
    if (libHandler_) {
      library_loaded_ = true;
      return AMDSMI_STATUS_SUCCESS;
    }
    errors += std::string(filename) + ": " + dlerror() + "; ";
  }
  std::cerr << "Fail to open library (tried " << filenames.size() << " candidate(s)): " << errors
            << std::endl;
  return AMDSMI_STATUS_FAIL_LOAD_MODULE;
}

amdsmi_status_t AMDSmiLibraryLoader::unload() {
  std::lock_guard<std::mutex> guard(library_mutex_);
  if (libHandler_) {
    dlclose(libHandler_);
    libHandler_ = nullptr;
    library_loaded_ = false;
  }
  return AMDSMI_STATUS_SUCCESS;
}

AMDSmiLibraryLoader::~AMDSmiLibraryLoader() { unload(); }

}  // namespace amd::smi
