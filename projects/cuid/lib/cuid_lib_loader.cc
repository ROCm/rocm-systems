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

#include "cuid_lib_loader.h"
#include "cuid.h"
#include <iostream>

CuidLibLoader::CuidLibLoader(): libHandler_(nullptr) {
}

amdcuid_status_t CuidLibLoader::load(const char* filename) {
    if (filename == nullptr) {
        return AMDCUID_STATUS_FILE_NOT_FOUND;
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
          return AMDCUID_STATUS_UNSUPPORTED;
      }
    }
    library_loaded_ = true;

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidLibLoader::unload() {
        std::lock_guard<std::mutex> guard(library_mutex_);
        if (libHandler_) {
            dlclose(libHandler_);
            libHandler_ = nullptr;
            library_loaded_ = false;
        }
        return AMDCUID_STATUS_SUCCESS;
}

CuidLibLoader::~CuidLibLoader() {
        unload();
}