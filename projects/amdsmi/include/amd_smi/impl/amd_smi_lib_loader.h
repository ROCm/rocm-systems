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

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_LIB_LOADER_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_LIB_LOADER_H_
#include <dlfcn.h>
#include <cstring>
#include <iostream>
#include <mutex>   //  NOLINT(build/c++11)
#include "amd_smi/amdsmi.h"


namespace amd::smi {
class AMDSmiLibraryLoader {
 public:
     AMDSmiLibraryLoader();

     amdsmi_status_t load(const char* filename);

     template<typename T> amdsmi_status_t load_symbol(T* func_handler,
            const char* func_name);


     amdsmi_status_t unload();

     ~AMDSmiLibraryLoader();

 private:
     void* libHandler_;
     std::mutex library_mutex_;
     bool library_loaded_ = false;
};

template<typename T> amdsmi_status_t AMDSmiLibraryLoader::load_symbol(
            T* func_handler,
            const char* func_name) {
    if (!libHandler_) {
        return AMDSMI_STATUS_FAIL_LOAD_MODULE;
    }

    if (!func_handler || !func_name) {
        return AMDSMI_STATUS_FAIL_LOAD_SYMBOL;
    }

    std::lock_guard<std::mutex> guard(library_mutex_);

    *reinterpret_cast<void**>(func_handler) =
            dlsym(libHandler_, func_name);
    if (*func_handler == nullptr) {
        char* error = dlerror();
        std::cerr << "AMDSmiLibraryLoader: Fail to load the symbol "
                    << func_name << ": " << error << std::endl;
        return AMDSMI_STATUS_FAIL_LOAD_SYMBOL;
    }

    return AMDSMI_STATUS_SUCCESS;
}

} // namespace amd::smi


#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_LIB_LOADER_H_
