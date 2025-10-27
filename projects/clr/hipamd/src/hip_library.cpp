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

#include "hip_global.hpp"
#include "hip_library.hpp"
#include "hip_platform.hpp"
#include "utils/debug.hpp"

namespace hip {
// Declare the functions that will be used for managed variable alloc
hipError_t ihipMallocManaged(void** ptr, size_t size, size_t align = 0, bool use_host_ptr = 0);

hipError_t ihipFree(void* ptr);

void LibraryContainer::Register(std::string name, int device, hipKernel_t k) {
  std::scoped_lock<std::mutex> lock(libMutex_);
  auto key = std::make_pair(name, device);
  if (kernels_.find(key) == kernels_.end()) {
    kernels_.insert(std::make_pair(std::make_pair(name, device), k));
    if (!hip::PlatformState::instance().RegisterLibraryFunction(k)) {
      LogPrintfInfo("Already registered: %p", k);
    }
  }
}

hipError_t LibraryContainer::Kernel(hipKernel_t* k, std::string name) {
  auto deviceId = hip::ihipGetDevice();
  if (auto ki = kernels_.find(std::make_pair(name, deviceId)); ki != kernels_.end()) {
    *k = ki->second;
    return hipSuccess;
  }

  auto m = fatbin_->Module(deviceId);
  auto f = functions_.find(name);
  if (f == functions_.end()) {
    return hipErrorNotFound;
  }
  auto ret = f->second.get()->getDynFunc(reinterpret_cast<hipFunction_t*>(k), m);

  // Register it, basically make it available for query though the hip context.
  if (ret == hipSuccess) Register(name, deviceId, *k);
  return ret;
}

hipError_t LibraryContainer::GVariable(const std::string& name, void** ptr, size_t* size) {
  if (name.size() == 0) {
    return hipErrorInvalidValue;
  }
  const auto d = globalVariables_.find(name);
  if (d == globalVariables_.end()) {
    return hipErrorNotFound;
  }

  if (ptr != nullptr) *ptr = d->second->address_;
  if (size != nullptr) *size = d->second->size_;

  return hipSuccess;
}

hipError_t LibraryContainer::MVariable(const std::string& name, void** ptr, size_t* size) {
  if (name.size() == 0) {
    return hipErrorInvalidValue;
  }
  const auto d = managedVariables_.find(name);
  if (d == managedVariables_.end()) {
    return hipErrorNotFound;
  }

  if (ptr != nullptr) *ptr = d->second->maddress_;
  if (size != nullptr) *size = d->second->size_;

  return hipSuccess;
}


LibraryContainer::LibraryContainer(const char* codeObject) {
  fatbin_ = std::make_shared<hip::FatBinaryInfo>(nullptr, codeObject);
}

LibraryContainer::LibraryContainer(const std::string fileName) {
  fatbin_ = std::make_shared<hip::FatBinaryInfo>(fileName.c_str(), nullptr);
}

LibraryContainer::~LibraryContainer() {
  for (const auto& k : kernels_) {
    (void)hip::PlatformState::instance().UnregisterLibraryFunction(k.second);
  }
  kernels_.clear();
  // release managed memory
  for (const auto& g : managedVariables_) {
    assert(hipSuccess == ihipFree(g.second->maddress_));
  }

  // release memory objects from global varaibles
  for (const auto& g : globalVariables_) {
    if (auto& mem = g.second->memory_; mem != nullptr) {
      amd::MemObjMap::RemoveMemObj(g.second->address_);
      mem->release();
    }
  }
  globalVariables_.clear();
}

// `BuildIt` builds and loads the Library, default behavior is lazy load.
// This function needs to be called before any query on library.
hipError_t LibraryContainer::BuildIt() {
  // built_ is already an atomic variable, so we declare scoped lock after this query.
  if (built_) {
    return hipSuccess;
  }

  std::scoped_lock<std::mutex> lock(libMutex_);
  if (!fatbin_) {
    return hipErrorInvalidValue;
  }

  const int deviceId = ihipGetDevice();
  std::vector<hip::Device*> devices = {g_devices[deviceId]};
  IHIP_RETURN_ONFAIL(fatbin_->ExtractFatBinaryUsingCOMGR(devices));
  IHIP_RETURN_ONFAIL(fatbin_->BuildProgram(deviceId));

  auto program =
      fatbin_->GetProgram(deviceId)->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  // Process Functions
  std::vector<std::string> functionNames;
  program->getGlobalFuncFromCodeObj(&functionNames);
  for (auto& name : functionNames) {
    LogPrintfInfo("Parsing functions name: %s", name.c_str());
    functions_.emplace(std::make_pair(name, std::make_shared<hip::Function>(name)));
  }

  // Process Globals/Managed
  std::vector<std::string> globalNames;
  std::vector<std::string> managedNames;
  program->getGlobalVarFromCodeObj(&globalNames);

  // All kernel symbols will have *.kd in the end, compiler also embeds a __hip_cuid_ object
  // remove all *.kd from names and __hip_cuid_*
  globalNames.erase(std::remove_if(globalNames.begin(), globalNames.end(),
                                   [](const std::string& s) {
                                     return s.find(".kd") != std::string::npos ||
                                            s.find("__hip_cuid_") != std::string::npos;
                                   }),
                    globalNames.end());

  // globalNames now either global varaibles or pair of managed variables.
  // managed variables are seen as <name>/<name>.managed.
  // populate managed varaibles in different containers
  for (const auto& name : globalNames) {
    if (name.find(".managed") == std::string::npos) {
      if (std::find(globalNames.begin(), globalNames.end(), std::string{name + ".managed"}) !=
          globalNames.end()) {
        managedNames.push_back(name);
      }
    }
  }

  // Remove all managed names from global names
  // So that our global names just have global device variables
  // We will process managed variables separately
  globalNames.erase(std::remove_if(globalNames.begin(), globalNames.end(),
                                   [&](const std::string& name) {
                                     return name.find(".managed") != std::string::npos ||
                                            std::find(managedNames.begin(), managedNames.end(),
                                                      name) != managedNames.end();
                                   }),
                    globalNames.end());

  // Process all global names
  for (const auto& name : globalNames) {
    amd::Memory* memory;  // memory object
    void* devicePtr;      // corresponding device ptr;
    size_t size;
    if (!program->createGlobalVarObj(&memory, &devicePtr, &size, name.c_str())) {
      return hipErrorInvalidResourceHandle;
    }
    amd::MemObjMap::AddMemObj(devicePtr, memory);
    LogPrintfInfo("Parsing globals name: %s memory object: %p device ptr: %p size: %lld",
                  name.c_str(), memory, devicePtr, size);
    globalVariables_.emplace(name, std::make_shared<DeviceVaraible>(memory, devicePtr, size));
  }

  // Process all managed variable, this is a bit more complicated
  for (const auto& name : managedNames) {
    // First step is to allocate all <name> and <name>.managed variables
    const auto managedName = name + ".managed";
    amd::Memory *memory{nullptr}, *managedMemory{nullptr};  // memory object
    void *devicePtr{nullptr}, *managedDevicePtr{nullptr};   // corresponding device ptr;
    size_t size = 0, managedSize = 0;
    if (!program->createGlobalVarObj(&memory, &devicePtr, &size, name.c_str())) {
      return hipErrorInvalidResourceHandle;
    }
    amd::MemObjMap::AddMemObj(devicePtr, memory);
    LogPrintfInfo(
        "Allocating managed variable name: %s memory object: %p device ptr: %p size: %lld",
        name.c_str(), memory, devicePtr, size);
    globalVariables_.emplace(name, std::make_shared<DeviceVaraible>(memory, devicePtr, size));


    if (!program->createGlobalVarObj(&managedMemory, &managedDevicePtr, &managedSize,
                                     managedName.c_str())) {
      return hipErrorInvalidResourceHandle;
    }
    amd::MemObjMap::AddMemObj(managedDevicePtr, managedMemory);

    LogPrintfInfo(
        "Allocating managed variable name: %s memory object: %p device ptr: %p size: %lld",
        managedName.c_str(), managedMemory, managedDevicePtr, managedSize);
    globalVariables_.emplace(managedName, std::make_shared<DeviceVaraible>(
                                              managedMemory, managedDevicePtr, managedSize));

    // Allocate managed pointers
    void* managedPointer = nullptr;
    IHIP_RETURN_ONFAIL(ihipMallocManaged(&managedPointer, managedSize, 0, 0));

    // Copy initial values
    hip::Stream* stream = hip::getNullStream();
    IHIP_RETURN_ONFAIL(ihipMemcpy(managedPointer, managedDevicePtr, managedSize,
                                  hipMemcpyDeviceToDevice, *stream));

    // initialize ptr with the allocated Ptr
    IHIP_RETURN_ONFAIL(
        ihipMemcpy(devicePtr, &managedPointer, size, hipMemcpyHostToDevice, *stream));

    // Track it
    managedVariables_.emplace(
        name, std::make_shared<ManagedVaraible>(managedDevicePtr, managedPointer, managedSize));
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

hipError_t hipLibraryGetGlobal(void** dptr, size_t* bytes, hipLibrary_t library, const char* name) {
  HIP_INIT_API(hipLibraryGetGlobal, dptr, bytes, library, name);
  if ((dptr == nullptr && bytes == nullptr) || name == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (library == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  auto l = reinterpret_cast<hip::LibraryContainer*>(library);
  auto ret = l->BuildIt();
  if (ret != hipSuccess) {
    HIP_RETURN(ret);
  }
  ret = l->GVariable(std::string{name}, dptr, bytes);
  HIP_RETURN(ret);
}

hipError_t hipLibraryGetManaged(void** dptr, size_t* bytes, hipLibrary_t library,
                                const char* name) {
  HIP_INIT_API(hipLibraryGetManaged, dptr, bytes, library, name);
  if ((dptr == nullptr && bytes == nullptr) || name == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (library == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }
  auto l = reinterpret_cast<hip::LibraryContainer*>(library);
  auto ret = l->BuildIt();
  if (ret != hipSuccess) {
    HIP_RETURN(ret);
  }
  ret = l->MVariable(std::string{name}, dptr, bytes);
  HIP_RETURN(ret);
}
}  // namespace hip
