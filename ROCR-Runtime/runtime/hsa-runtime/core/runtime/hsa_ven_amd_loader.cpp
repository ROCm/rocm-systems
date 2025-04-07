////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/hsa_ven_amd_loader_impl.h"

#include "core/inc/amd_hsa_loader.hpp"
#include "core/inc/runtime.h"

namespace rocr {

using namespace amd::hsa;
using namespace core;

using loader::CodeObjectReaderImpl;
using loader::Executable;
using loader::LoadedCodeObject;
using loader::Loader;

namespace AMD {

hsa_status_t handleException();

}   // namespace amd

hsa_status_t hsa_ven_amd_loader_query_host_address(
  const void *device_address,
  const void **host_address) {
  try {
    if (!Runtime::runtime_singleton_->IsOpen()) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (nullptr == device_address) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    if (nullptr == host_address) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    uintptr_t udaddr = reinterpret_cast<uintptr_t>(device_address);
    uintptr_t uhaddr = Runtime::runtime_singleton_->loader()->FindHostAddress(udaddr);
    if (0 == uhaddr) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    *host_address = reinterpret_cast<void*>(uhaddr);
    return HSA_STATUS_SUCCESS;
  } catch(...) { return AMD::handleException(); }
}

hsa_status_t hsa_ven_amd_loader_query_segment_descriptors(
  hsa_ven_amd_loader_segment_descriptor_t *segment_descriptors,
  size_t *num_segment_descriptors) {
  try {
    if (!Runtime::runtime_singleton_->IsOpen()) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    // Arguments are checked by the loader.
    return Runtime::runtime_singleton_->loader()->QuerySegmentDescriptors(segment_descriptors, num_segment_descriptors);
  } catch(...) { return AMD::handleException(); }
}

hsa_status_t hsa_ven_amd_loader_query_executable(
  const void *device_address,
  hsa_executable_t *executable) {
  try {
    if (!Runtime::runtime_singleton_->IsOpen()) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }
    if ((nullptr == device_address) || (nullptr == executable)) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    uintptr_t udaddr = reinterpret_cast<uintptr_t>(device_address);
    hsa_executable_t exec = Runtime::runtime_singleton_->loader()->FindExecutable(udaddr);
    if (0 == exec.handle) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    *executable = exec;
    return HSA_STATUS_SUCCESS;
  } catch(...) { return AMD::handleException(); }
}

hsa_status_t hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
  hsa_executable_t executable,
  hsa_status_t (*callback)(
    hsa_executable_t executable,
    hsa_loaded_code_object_t loaded_code_object,
    void *data),
  void *data) {
  try {
    if (!Runtime::runtime_singleton_->IsOpen()) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (nullptr == callback) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    Executable *exec = Executable::Object(executable);
    if (!exec) {
      return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
    }

    return exec->IterateLoadedCodeObjects(callback, data);
  } catch(...) { return AMD::handleException(); }
}

hsa_status_t hsa_ven_amd_loader_loaded_code_object_get_info(
  hsa_loaded_code_object_t loaded_code_object,
  hsa_ven_amd_loader_loaded_code_object_info_t attribute,
  void *value) {
  try {
    if (!Runtime::runtime_singleton_->IsOpen()) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (nullptr == value) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const LoadedCodeObject *lcobj = LoadedCodeObject::Object(loaded_code_object);
    if (!lcobj) {
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }

    switch (attribute) {
      case HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_EXECUTABLE: {
        *((hsa_executable_t*)value) = lcobj->getExecutable();
        break;
      }
      case HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_KIND: {
        *((uint32_t*)value) = lcobj->getAgent().handle == 0
            ? HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_KIND_PROGRAM
            : HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_KIND_AGENT;
        break;
      }
      case HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT: {
        hsa_agent_t agent = lcobj->getAgent();
        if (agent.handle == 0) {
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        }
        *((hsa_agent_t*)value) = agent;
        break;
      }
      case HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE: {
        // TODO Update loader so it keeps track if code object was loaded from a
        // file or memory.
        *((uint32_t*)value) = HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY;
        break;
      }
      case HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE: {
        *((uint64_t*)value) = lcobj->getElfData();
        break;
      }
      case HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE: {
        *((uint64_t*)value) = lcobj->getElfSize();
        break;
      }
      case HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_FILE: {
        // TODO Update loader so it keeps track if code object was loaded from a
        // file or memory.
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        break;
      }
      case HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA: {
        // TODO Check if executable is frozen.
        // This suggests this code should be moved into LoadedCodeObjectImpl::getinfo
        // as is done for other *_get_info methods. Currently LoadedCodeObject has a
        // GetInfo method which is likely not used.
        // Also should this have a *NOT_FROZEN ststus code added?
        // if (state_ != HSA_EXECUTABLE_STATE_FROZEN) {
        //   return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        // }
        *((int64_t*)value) = lcobj->getDelta();
        break;
      }
      case HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE: {
        // TODO Check if executable is frozen.
        *((uint64_t*)value) = lcobj->getLoadBase();
        break;
      }
      case HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE: {
        // TODO Check if executable is frozen.
        *((uint64_t*)value) = lcobj->getLoadSize();
        break;
      }
      case HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH: {
        *(reinterpret_cast<uint32_t*>(value)) = lcobj->getUri().size();
        break;
      }
      case HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI: {
        memcpy(value, lcobj->getUri().c_str(), lcobj->getUri().size());
        break;
      }
      default: {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      }
    }

    return HSA_STATUS_SUCCESS;
  } catch(...) { return AMD::handleException(); }
}

hsa_status_t
hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
    hsa_file_t file,
    size_t offset,
    size_t size,
    hsa_code_object_reader_t *code_object_reader) {
  try {
    if (!Runtime::runtime_singleton_->IsOpen()) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (nullptr == code_object_reader) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    if (size == 0) {
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }

    std::unique_ptr<CodeObjectReaderImpl> reader(
        new (std::nothrow) CodeObjectReaderImpl());
    if (!reader) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    hsa_status_t status = reader->SetFile(file, offset, size);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }

    *code_object_reader = CodeObjectReaderImpl::Handle(reader.release());
    return HSA_STATUS_SUCCESS;
  } catch(...) { return AMD::handleException(); }
}

namespace {

Loader *GetLoader() {
  return Runtime::runtime_singleton_->loader();
}

} // namespace anonymous

hsa_status_t
hsa_ven_amd_loader_iterate_executables(
    hsa_status_t (*callback)(
      hsa_executable_t executable,
      void *data),
    void *data) {
  try {
    if (!Runtime::runtime_singleton_->IsOpen()) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (nullptr == callback) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    return GetLoader()->IterateExecutables(callback, data);
  } catch(...) { return AMD::handleException(); }
}

} // namespace rocr
