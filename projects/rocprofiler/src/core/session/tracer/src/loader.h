/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef ROCTRACER_LOADER_H_
#define ROCTRACER_LOADER_H_

#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <link.h>
#include <unistd.h>

#include "src/utils/helper.h"
#include "src/utils/filesystem.hpp"

namespace fs = rocprofiler::common::filesystem;

namespace roctracer {

// Base loader class
template <typename Loader> class BaseLoader {
 protected:
  BaseLoader(const char* pattern) {
    // Iterate through the process' loaded shared objects and try to dlopen the
    // first entry with a file name starting with the given 'pattern'. This
    // allows the loader to acquire a handle to the target library iff it is
    // already loaded. The handle is used to query symbols exported by that
    // library.

    auto callback = [this, pattern](dl_phdr_info* info) {
      if (handle_ == nullptr &&
          fs::path(info->dlpi_name).filename().string().rfind(pattern, 0) == 0)
        handle_ = ::dlopen(info->dlpi_name, RTLD_LAZY);
    };
    dl_iterate_phdr(
        [](dl_phdr_info* info, size_t size, void* data) {
          (*reinterpret_cast<decltype(callback)*>(data))(info);
          return 0;
        },
        &callback);
  }

  ~BaseLoader() {
    if (handle_ != nullptr) ::dlclose(handle_);
  }

  BaseLoader(const BaseLoader&) = delete;
  BaseLoader& operator=(const BaseLoader&) = delete;

 public:
  bool IsEnabled() const { return handle_ != nullptr; }

  template <typename FunctionPtr> FunctionPtr GetFun(const char* symbol) const {
    assert(IsEnabled());

    auto function_ptr = reinterpret_cast<FunctionPtr>(::dlsym(handle_, symbol));
    if (function_ptr == nullptr)
      rocprofiler::fatal("symbol lookup '%s' failed: %s", symbol, ::dlerror());
    return function_ptr;
  }

  static inline Loader& Instance() {
    static Loader instance;
    return instance;
  }

 private:
  void* handle_;
};

}  // namespace roctracer

// HIP runtime library loader class
namespace roctracer {
#if STATIC_BUILD
__attribute__((weak)) const char* hipKernelNameRef(const hipFunction_t f) { return nullptr; }
__attribute__((weak)) const char* hipKernelNameRefByPtr(const void* hostFunction,
                                                        hipStream_t stream) {
  return nullptr;
}
__attribute__((weak)) int hipGetStreamDeviceId(hipStream_t stream) { return 0; }
__attribute__((weak)) const char* hipGetCmdName(unsigned op) { return nullptr; }
__attribute__((weak)) const char* hipApiName(uint32_t id) { return nullptr; }
__attribute__((weak)) void hipRegisterTracerCallback(int (*function)(activity_domain_t domain,
                                                                     uint32_t operation_id,
                                                                     void* data)) {}

class HipLoader {
 private:
  HipLoader() {}

 public:
  bool IsEnabled() const { return true; }

  int GetStreamDeviceId(hipStream_t stream) const { return hipGetStreamDeviceId(stream); }

  const char* KernelNameRef(const hipFunction_t f) const { return hipKernelNameRef(f); }

  const char* KernelNameRefByPtr(const void* host_function, hipStream_t stream = nullptr) const {
    return hipKernelNameRefByPtr(host_function, stream);
  }

  const char* GetOpName(unsigned op) const { return hipGetCmdName(op); }

  const char* ApiName(uint32_t id) const { return hipApiName(id); }

  void RegisterTracerCallback(int (*callback)(activity_domain_t domain, uint32_t operation_id,
                                              void* data)) const {
    return hipRegisterTracerCallback(callback);
  }

  static inline HipLoader& Instance() {
    static HipLoader instance;
    return instance;
  }
};
#else
class HipLoader : public BaseLoader<HipLoader> {
 private:
  friend HipLoader& BaseLoader::Instance();
  HipLoader() : BaseLoader("libamdhip64.so") {}

 public:
  int GetStreamDeviceId(hipStream_t stream) const {
    static auto function = GetFun<int (*)(hipStream_t stream)>("hipGetStreamDeviceId");
    return function(stream);
  }

  const char* KernelNameRef(const hipFunction_t f) const {
    static auto function = GetFun<const char* (*)(const hipFunction_t f)>("hipKernelNameRef");
    return function(f);
  }

  const char* KernelNameRefByPtr(const void* host_function, hipStream_t stream = nullptr) const {
    static auto function = GetFun<const char* (*)(const void* hostFunction, hipStream_t stream)>(
        "hipKernelNameRefByPtr");
    return function(host_function, stream);
  }

  const char* GetOpName(unsigned op) const {
    static auto function = GetFun<const char* (*)(unsigned op)>("hipGetCmdName");
    return function(op);
  }

  const char* ApiName(uint32_t id) const {
    static auto function = GetFun<const char* (*)(uint32_t id)>("hipApiName");
    return function(id);
  }

  void RegisterTracerCallback(int (*callback)(activity_domain_t domain, uint32_t operation_id,
                                              void* data)) const {
    static auto function = GetFun<void (*)(int (*callback)(
        activity_domain_t domain, uint32_t operation_id, void* data))>("hipRegisterTracerCallback");
    return function(callback);
  }
};
#endif

// ROCTX library loader class
class RocTxLoader : public BaseLoader<RocTxLoader> {
 private:
  friend RocTxLoader& BaseLoader::Instance();
  RocTxLoader() : BaseLoader("libroctx64.so") {}

 public:
  void RegisterTracerCallback(int (*callback)(activity_domain_t domain, uint32_t operation_id,
                                              void* data)) const {
    static auto function =
        GetFun<void (*)(int (*callback)(activity_domain_t domain, uint32_t operation_id,
                                        void* data))>("roctxRegisterTracerCallback");
    return function(callback);
  }
};

}  // namespace roctracer

#endif  // ROCTRACER_LOADER_H_
