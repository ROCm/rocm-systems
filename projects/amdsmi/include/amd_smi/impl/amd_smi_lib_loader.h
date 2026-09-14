// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_LIB_LOADER_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_LIB_LOADER_H_
#include <dlfcn.h>

#include <cstring>
#include <iostream>
#include <mutex>  //  NOLINT(build/c++11)
#include <vector>

#include "amd_smi/amdsmi.h"

namespace amd::smi {

// Ordered libdrm_amdgpu soname candidates, so tarball / TheRock installs that
// rename the soname still resolve.
std::vector<const char*> libdrm_amdgpu_sonames();

class AMDSmiLibraryLoader {
 public:
  AMDSmiLibraryLoader();

  amdsmi_status_t load(const char* filename);

  // Tries each candidate in order, returning success on the first that opens.
  // Only logs once all candidates fail.
  amdsmi_status_t load(const std::vector<const char*>& filenames);

  template <typename T>
  amdsmi_status_t load_symbol(T* func_handler, const char* func_name);

  amdsmi_status_t unload();

  ~AMDSmiLibraryLoader();

 private:
  void* libHandler_;
  std::mutex library_mutex_;
  bool library_loaded_ = false;
};

template <typename T>
amdsmi_status_t AMDSmiLibraryLoader::load_symbol(T* func_handler, const char* func_name) {
  if (!libHandler_) {
    return AMDSMI_STATUS_FAIL_LOAD_MODULE;
  }

  if (!func_handler || !func_name) {
    return AMDSMI_STATUS_FAIL_LOAD_SYMBOL;
  }

  std::lock_guard<std::mutex> guard(library_mutex_);

  *reinterpret_cast<void**>(func_handler) = dlsym(libHandler_, func_name);
  if (*func_handler == nullptr) {
    char* error = dlerror();
    std::cerr << "AMDSmiLibraryLoader: Fail to load the symbol " << func_name << ": " << error
              << std::endl;
    return AMDSMI_STATUS_FAIL_LOAD_SYMBOL;
  }

  return AMDSMI_STATUS_SUCCESS;
}

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_LIB_LOADER_H_
