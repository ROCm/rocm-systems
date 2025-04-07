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

#include "rdc_lib/RdcLibraryLoader.h"

#include "rdc_lib/RdcException.h"

namespace amd {
namespace rdc {

RdcLibraryLoader::RdcLibraryLoader() : libHandler_(nullptr) {}

rdc_status_t RdcLibraryLoader::load(const char* filename) {
  if (filename == nullptr) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }
  if (libHandler_) {
    unload();
  }

  std::lock_guard<std::mutex> guard(library_mutex_);
  libHandler_ = dlopen(filename, RTLD_LAZY);
  if (!libHandler_) {
    char* error = dlerror();
    throw RdcException(
        RDC_ST_FAIL_LOAD_MODULE,
        std::string("Fail to open ") + std::string(filename) + ": " + std::string(error));
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  return RDC_ST_OK;
}

rdc_status_t RdcLibraryLoader::unload() {
  std::lock_guard<std::mutex> guard(library_mutex_);
  if (libHandler_) {
    dlclose(libHandler_);
    libHandler_ = nullptr;
  }
  return RDC_ST_OK;
}

RdcLibraryLoader::~RdcLibraryLoader() { unload(); }

}  // namespace rdc
}  // namespace amd
