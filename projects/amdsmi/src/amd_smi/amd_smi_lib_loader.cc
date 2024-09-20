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

#include "amd_smi/impl/amd_smi_lib_loader.h"
#include <iostream>

namespace amd {
namespace smi {

AMDSmiLibraryLoader::AMDSmiLibraryLoader(): libHandler_(nullptr) {
}

amdsmi_status_t AMDSmiLibraryLoader::load(const char* filename) {
    if (filename == nullptr) {
        return AMDSMI_STATUS_FAIL_LOAD_MODULE;
    }
    if (libHandler_) {
        unload();
    }

    std::lock_guard<std::mutex> guard(library_mutex_);
    libHandler_ = dlopen(filename, RTLD_LAZY);
    if (!libHandler_) {
        char* error = dlerror();
        std::cerr << "Fail to open " << filename <<": " << error
                << std::endl;
        return AMDSMI_STATUS_FAIL_LOAD_MODULE;
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiLibraryLoader::unload() {
        std::lock_guard<std::mutex> guard(library_mutex_);
        if (libHandler_) {
            dlclose(libHandler_);
            libHandler_ = nullptr;
        }
        return AMDSMI_STATUS_SUCCESS;
}

AMDSmiLibraryLoader::~AMDSmiLibraryLoader() {
        unload();
}

}  // namespace rdc
}  // namespace amd
