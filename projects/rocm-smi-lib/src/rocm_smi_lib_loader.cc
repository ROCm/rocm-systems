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

#include <iostream>

#include "rocm_smi/rocm_smi_lib_loader.h"

namespace amd {
namespace smi {

ROCmSmiLibraryLoader::ROCmSmiLibraryLoader(): libHandler_(nullptr) {
}

rsmi_status_t ROCmSmiLibraryLoader::load(const char* filename) {
    if (filename == nullptr) {
        return RSMI_STATUS_FAIL_LOAD_MODULE;
    }
    if (libHandler_ || library_loaded_) {
        unload();
    }

    std::lock_guard<std::mutex> guard(library_mutex_);
    // check if already loaded, return success if it is
    // dlopen(filename, RTLD_NOLOAD) == null only IFF library is not loaded
    void* isLibOpen = dlopen(filename, RTLD_NOLOAD);
    if (isLibOpen == nullptr) {
      libHandler_ = dlopen(filename, RTLD_LAZY);
      if (!libHandler_) {
          char* error = dlerror();
          std::cerr << "Fail to open " << filename <<": " << error
                    << std::endl;
          return RSMI_STATUS_FAIL_LOAD_MODULE;
      }
    }
    library_loaded_ = true;

    return RSMI_STATUS_SUCCESS;
}

rsmi_status_t ROCmSmiLibraryLoader::unload() {
        std::lock_guard<std::mutex> guard(library_mutex_);
        if (libHandler_) {
            dlclose(libHandler_);
            libHandler_ = nullptr;
            library_loaded_ = false;
        }
        return RSMI_STATUS_SUCCESS;
}

ROCmSmiLibraryLoader::~ROCmSmiLibraryLoader() {
        unload();
}

}  // namespace smi
}  // namespace amd
