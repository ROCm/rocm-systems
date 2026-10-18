/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef INCLUDE_RDC_LIB_RDCLIBRARYLOADER_H_
#define INCLUDE_RDC_LIB_RDCLIBRARYLOADER_H_
#include <dlfcn.h>

#include <mutex>  //  NOLINT(build/c++11)
#include <string>

#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/RdcLogger.h"

namespace amd {
namespace rdc {
// Loads one library and keeps it until unload(). Loading the same filename again returns the
// library already loaded, with the first call's dlopen_flags, because handlers and modules
// created from it may still be in use. Loading a different filename fails; filenames are
// compared as strings.
class RdcLibraryLoader {
 public:
  RdcLibraryLoader();

  // throws RdcException if lib not found
  rdc_status_t load(const char* filename, int dlopen_flags = RTLD_LAZY);

  template <typename T>
  rdc_status_t load_symbol(T* func_handler, const char* func_name);

  template <typename T>
  rdc_status_t load(const char* filename, T* func_make_handler, int dlopen_flags = RTLD_LAZY);

  rdc_status_t unload();

  ~RdcLibraryLoader();

 private:
  void* libHandler_;
  std::string filename_;
  std::mutex library_mutex_;
};

template <typename T>
rdc_status_t RdcLibraryLoader::load_symbol(T* func_handler, const char* func_name) {
  if (!func_handler || !func_name) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  std::lock_guard<std::mutex> guard(library_mutex_);
  if (!libHandler_) {
    RDC_LOG(RDC_ERROR, "Must load the library before loading the symbol");
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  *reinterpret_cast<void**>(func_handler) = dlsym(libHandler_, func_name);
  if (*func_handler == nullptr) {
    char* error = dlerror();
    RDC_LOG(RDC_ERROR, "RdcLibraryLoader: Fail to load the symbol " << func_name << ": " << error);
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  return RDC_ST_OK;
}

template <typename T>
rdc_status_t RdcLibraryLoader::load(const char* filename, T* func_make_handler, int dlopen_flags) {
  if (filename == nullptr || func_make_handler == nullptr) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  try {
    rdc_status_t status = load(filename, dlopen_flags);
    if (status != RDC_ST_OK) {
      return status;
    }
  } catch (RdcException& e) {
    RDC_LOG(RDC_ERROR, e.what());
    return e.error_code();
  }

  return load_symbol(func_make_handler, "make_handler");
}

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_RDCLIBRARYLOADER_H_
