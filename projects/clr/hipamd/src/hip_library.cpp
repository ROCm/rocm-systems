/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include "hip/hip_runtime.h"
#include "hip_library.hpp"
#include "hip_platform.hpp"
#include "utils/debug.hpp"

namespace hip {
void LibraryContainer::Register(std::string name, int device, hipKernel_t k) {
  std::scoped_lock<std::mutex> lock(lib_mutex_);
  auto key = std::make_pair(name, device);
  if (kernels_.find(key) == kernels_.end()) {
    kernels_.insert(std::make_pair(std::make_pair(name, device), k));
    if (!hip::PlatformState::instance().RegisterLibraryFunction(k)) {
      LogPrintfInfo("Already registered: %p", k);
    }
  }
}

hipError_t LibraryContainer::Kernel(hipKernel_t* k, std::string name) {
  auto device_id = hip::ihipGetDevice();
  if (auto ki = kernels_.find(std::make_pair(name, device_id)); ki != kernels_.end()) {
    *k = ki->second;
    return hipSuccess;
  }
  auto m = fatbin_->Module(device_id);
  auto f = functions_.find(name);
  if (f == functions_.end()) {
    return hipErrorNotFound;
  }
  auto ret = f->second.get()->getDynFunc(reinterpret_cast<hipFunction_t*>(k), m);

  // Register it, basically make it available for query though the hip context.
  Register(name, device_id, *k);
  return hipSuccess;
}

LibraryContainer::LibraryContainer(const char* code_object) {
  fatbin_ = std::make_shared<hip::FatBinaryInfo>(nullptr, code_object);
}

LibraryContainer::LibraryContainer(const std::string file_name) {
  fatbin_ = std::make_shared<hip::FatBinaryInfo>(file_name.c_str(), nullptr);
}

LibraryContainer::~LibraryContainer() {
  for (const auto& k : kernels_) {
    (void)hip::PlatformState::instance().UnregisterLibraryFunction(k.second);
  }
  kernels_.clear();
}

// BuildIt builds and loads the Library, default behavior is lazy load.
// This function needs to be called before any query on library.
hipError_t LibraryContainer::BuildIt() {
  std::scoped_lock<std::mutex> lock(lib_mutex_);
  if (built_) {
    return hipSuccess;
  }

  if (!fatbin_) {
    return hipErrorInvalidValue;
  }

  int device_id = ihipGetDevice();
  std::vector<hip::Device*> devices = {g_devices[device_id]};
  IHIP_RETURN_ONFAIL(fatbin_->ExtractFatBinaryUsingCOMGR(devices));
  IHIP_RETURN_ONFAIL(fatbin_->BuildProgram(device_id));

  auto program =
      fatbin_->GetProgram(device_id)->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  // Process Functions
  std::vector<std::string> function_names;
  program->getGlobalFuncFromCodeObj(&function_names);
  for (auto& name : function_names) {
    functions_.emplace(std::make_pair(name, std::make_shared<hip::Function>(name)));
  }

  built_ = true;
  return hipSuccess;
}

hipError_t hipLibraryLoadData(hipLibrary_t* library, const void* image, hipJitOption* jitOptions,
                              void** jitOptionsValues, unsigned int numJitOptions,
                              hipLibraryOption* libraryOptions, void** libraryOptionValues,
                              unsigned int numLibraryOptions) {
  HIP_INIT_API(hipLibraryLoadData, library, image, jitOptions, jitOptionsValues, numJitOptions,
               libraryOptions, libraryOptionValues, numLibraryOptions);
  if (library == nullptr || image == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // We do not support JIT options
  if (numJitOptions > 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* l = new hip::LibraryContainer((const char*)image);
  *library = reinterpret_cast<hipLibrary_t>(l);
  HIP_RETURN(hipSuccess);
}

hipError_t hipLibraryLoadFromFile(hipLibrary_t* library, const char* fname,
                                  hipJitOption* jitOptions, void** jitOptionsValues,
                                  unsigned int numJitOptions, hipLibraryOption* libraryOptions,
                                  void** libraryOptionValues, unsigned int numLibraryOptions) {
  HIP_INIT_API(hipLibraryLoadFromFile, library, fname, jitOptions, jitOptionsValues, numJitOptions,
               libraryOptions, libraryOptionValues, numLibraryOptions);
  if (library == nullptr || !std::filesystem::exists(fname) || numJitOptions > 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  auto* l = new hip::LibraryContainer(std::string(fname));
  *library = reinterpret_cast<hipLibrary_t>(l);
  HIP_RETURN(hipSuccess);
}

hipError_t hipLibraryUnload(hipLibrary_t library) {
  HIP_INIT_API(hipLibraryUnload, library);
  if (library == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }
  auto l = reinterpret_cast<hip::LibraryContainer*>(library);
  delete l;
  HIP_RETURN(hipSuccess);
}

hipError_t hipLibraryGetKernelCount(unsigned int* count, hipLibrary_t library) {
  HIP_INIT_API(hipLibraryGetKernelCount, count, library);
  if (library == nullptr || count == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  auto l = reinterpret_cast<hip::LibraryContainer*>(library);
  auto ret = l->BuildIt();
  if (ret != hipSuccess) {
    HIP_RETURN(ret);
  }
  *count = static_cast<int>(l->KernelCount());
  HIP_RETURN(hipSuccess);
}

hipError_t hipLibraryGetKernel(hipKernel_t* kernel, hipLibrary_t library, const char* kname) {
  HIP_INIT_API(hipLibraryGetKernel, kernel, library, kname);
  if (library == nullptr || kname == nullptr || kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  auto l = reinterpret_cast<hip::LibraryContainer*>(library);
  auto ret = l->BuildIt();
  if (ret != hipSuccess) {
    HIP_RETURN(ret);
  }
  ret = l->Kernel(kernel, kname);
  HIP_RETURN(ret);
}
}  // namespace hip
