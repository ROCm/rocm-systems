/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "amdsmi_loader.hpp"
#include "util.hpp"
#include "rocshmem/rocshmem.hpp"

#include <dlfcn.h>

namespace rocshmem {

AmdsmiLoader::AmdsmiLoader()
    : amdsmi_handle(nullptr),
      init(nullptr),
      shut_down(nullptr),
      get_processor_handle_from_bdf(nullptr),
      get_gpu_fabric_info(nullptr) {

  // Try to load the AMD SMI library
  amdsmi_handle = dlopen("libamd_smi.so", RTLD_LAZY);
  if (!amdsmi_handle) {
    LOG_TRACE("Failed to load libamd_smi.so: %s", dlerror());
    return;
  }

  int err = init_function_table();
  if (err != ROCSHMEM_SUCCESS) {
    LOG_TRACE("Could not construct AMD SMI function table");
  }
}

AmdsmiLoader::~AmdsmiLoader() {
  if (amdsmi_handle) {
    dlclose(amdsmi_handle);
  }
}

int AmdsmiLoader::init_function_table() {
  DLSYM_HELPER((*this), amdsmi_, amdsmi_handle, init);
  DLSYM_HELPER((*this), amdsmi_, amdsmi_handle, shut_down);
  DLSYM_HELPER((*this), amdsmi_, amdsmi_handle, get_processor_handle_from_bdf);
  DLSYM_HELPER((*this), amdsmi_, amdsmi_handle, get_gpu_fabric_info);
  return ROCSHMEM_SUCCESS;
}

bool AmdsmiLoader::isLoaded() const {
  return amdsmi_handle != nullptr;
}

}  // namespace rocshmem
