/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "hip_library.hpp"
#include "hip_platform.hpp"
#include "utils/debug.hpp"

namespace hip {
void LibraryContainer::Register(const std::string &name, int device, hipKernel_t k) {
  std::scoped_lock<std::mutex> lock(lib_mutex_);
  auto key = std::make_pair(name, device);
  if (kernels_.find(key) == kernels_.end()) {
    kernels_.insert(std::make_pair(std::make_pair(name, device), k));
    auto lib = reinterpret_cast<hipLibrary_t>(this);
    if (!hip::PlatformState::Instance().RegisterLibraryFunction(k, lib)) {
      LogPrintfInfo("Already registered: %p", k);
    }
  }
}

hipError_t LibraryContainer::GetKernelName(const char** name, hipKernel_t kernel) {
  std::scoped_lock<std::mutex> lock(lib_mutex_);
  if (kernels_.empty()) {
    return hipErrorInvalidValue;
  }

  for (const auto &it : kernels_) {
    if (it.second == kernel) {
      *name = it.first.first.c_str();
      return hipSuccess;
    }
  }
  return hipErrorInvalidValue;
}

hipError_t LibraryContainer::EnumerateKernels(hipKernel_t* k, unsigned int maxKernels) {
  auto names = dynco_->getFunctionNames();
  auto maxCount = (maxKernels > names.size()) ? names.size() : maxKernels;
  unsigned int count = 0;
  for (const auto& name : names) {
    if (count >= maxCount) break;
    hipKernel_t kern;
    auto ret = Kernel(&kern, name);
    if (ret != hipSuccess) {
      return ret;
    }
    k[count++] = kern;
  }
  return hipSuccess;
}

hipError_t LibraryContainer::Kernel(hipKernel_t* k, const std::string &name) {
  // Use the device the underlying DynCO was loaded for, not ihipGetDevice():
  // the caller may have switched devices since BuildIt(), but our module is
  // single-device. Keying off the active device would mis-attribute cache
  // entries across devices.
  const int device_id = dynco_ ? dynco_->getDeviceId() : hip::ihipGetDevice();
  {
    // Cache hit fast path under lib_mutex_ — Register() writes kernels_
    // under the same mutex, so unlocked reads would race.
    std::scoped_lock<std::mutex> lock(lib_mutex_);
    if (auto ki = kernels_.find(std::make_pair(name, device_id)); ki != kernels_.end()) {
      *k = ki->second;
      return hipSuccess;
    }
  }
  hipFunction_t hfunc = nullptr;
  IHIP_RETURN_ONFAIL(dynco_->getDynFunc(&hfunc, name));
  *k = reinterpret_cast<hipKernel_t>(hfunc);
  // Register() re-takes lib_mutex_; its check-and-insert handles the benign
  // TOCTOU window where a second concurrent caller misses the cache and we
  // both arrive here with the same hfunc.
  Register(name, device_id, *k);
  return hipSuccess;
}

hipError_t LibraryContainer::ResolveKernelForDevice(hipKernel_t* k, const std::string& name,
                                                    int device) {
  if (k == nullptr) {
    return hipErrorInvalidValue;
  }

  if (device < 0 || static_cast<size_t>(device) >= hip::g_devices.size()) {
    return hipErrorInvalidDevice;
  }

  IHIP_RETURN_ONFAIL(BuildIt());
  if (device == DeviceId()) {
    return Kernel(k, name);
  }

  // The first getter or setter for a non-primary device loads its code object. Later calls from
  // either API reuse this entry and the kernel cached internally by Function::GetDynFunc.
  std::scoped_lock<std::mutex> lock(lib_mutex_);
  auto dynco = code_objects_by_device_.find(device);
  if (dynco == code_objects_by_device_.end()) {
    auto target = std::make_unique<hip::DynCO>(device);
    const char* fname = filename_.empty() ? nullptr : filename_.c_str();
    IHIP_RETURN_ONFAIL(target->loadCodeObject(fname, image_, /* init_global_vars */ false));
    dynco = code_objects_by_device_.emplace(device, std::move(target)).first;
  }

  hipFunction_t function = nullptr;
  IHIP_RETURN_ONFAIL(dynco->second->getDynFunc(&function, name));
  *k = reinterpret_cast<hipKernel_t>(function);

  return hipSuccess;
}

size_t LibraryContainer::KernelCount() {
  unsigned int count = 0;
  if (dynco_) {
    (void)dynco_->getFuncCount(&count);
  }
  return count;
}

hipError_t LibraryContainer::GetGlobal(const std::string& name, void** dptr, size_t* bytes) {
  return dynco_->GetGlobal(name, dptr, bytes);
}

hipError_t LibraryContainer::GetManaged(const std::string& name, void** dptr, size_t* bytes) {
  return dynco_->GetManaged(name, dptr, bytes);
}

LibraryContainer::LibraryContainer(const char* code_object) : image_(code_object) {}

LibraryContainer::LibraryContainer(const std::string &file_name) : filename_(file_name) {}

LibraryContainer::~LibraryContainer() {
  // No lock here on purpose: destruction is the user's responsibility to
  // serialize against any in-flight Kernel/Enumerate/GetGlobal calls (same
  // contract CUDA documents for cuLibraryUnload). Locking would falsely
  // suggest safety while the mutex itself is about to be destroyed.
  for (const auto& k : kernels_) {
    (void)hip::PlatformState::Instance().UnregisterLibraryFunction(k.second);
  }
  code_objects_by_device_.clear();
  kernels_.clear();
  // dynco_ unique_ptr destruction frees vars, functions, and the underlying fatbin.
}

// BuildIt builds and loads the Library, default behavior is lazy load.
// This function needs to be called before any query on library.
hipError_t LibraryContainer::BuildIt() {
  if (built_.load(std::memory_order_acquire)) {
    return hipSuccess;
  }
  std::scoped_lock<std::mutex> lock(lib_mutex_);
  if (built_.load(std::memory_order_relaxed)) {
    return hipSuccess;
  }
  if (filename_.empty() && image_ == nullptr) {
    return hipErrorInvalidValue;
  }

  dynco_ = std::make_unique<hip::DynCO>();
  const char* fname = filename_.empty() ? nullptr : filename_.c_str();
  IHIP_RETURN_ONFAIL(dynco_->loadCodeObject(fname, image_));

  built_.store(true, std::memory_order_release);
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

hipError_t hipLibraryGetGlobal(void** dptr, size_t* bytes, hipLibrary_t library,
                               const char* name) {
  HIP_INIT_API(hipLibraryGetGlobal, dptr, bytes, library, name);
  if ((dptr == nullptr && bytes == nullptr) || name == nullptr || strlen(name) == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (library == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }
  auto* l = reinterpret_cast<hip::LibraryContainer*>(library);
  auto ret = l->BuildIt();
  if (ret != hipSuccess) {
    HIP_RETURN(ret);
  }
  HIP_RETURN(l->GetGlobal(std::string{name}, dptr, bytes));
}

hipError_t hipLibraryGetManaged(void** dptr, size_t* bytes, hipLibrary_t library,
                                const char* name) {
  HIP_INIT_API(hipLibraryGetManaged, dptr, bytes, library, name);
  if ((dptr == nullptr && bytes == nullptr) || name == nullptr || strlen(name) == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (library == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }
  auto* l = reinterpret_cast<hip::LibraryContainer*>(library);
  auto ret = l->BuildIt();
  if (ret != hipSuccess) {
    HIP_RETURN(ret);
  }
  HIP_RETURN(l->GetManaged(std::string{name}, dptr, bytes));
}

hipError_t hipLibraryEnumerateKernels(hipKernel_t* kernels, unsigned int numKernels,
                                      hipLibrary_t library) {
  HIP_INIT_API(hipLibraryEnumerateKernels, kernels, numKernels, library);
  if (kernels == nullptr || library == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto l = reinterpret_cast<hip::LibraryContainer*>(library);
  auto ret = l->BuildIt();
  if (ret != hipSuccess) {
    HIP_RETURN(ret);
  }

  if (numKernels == 0) {
    HIP_RETURN(hipSuccess);
  }

  HIP_RETURN(l->EnumerateKernels(kernels, numKernels));
}

hipError_t hipKernelGetLibrary(hipLibrary_t* library, hipKernel_t kernel) {
  HIP_INIT_API(hipKernelGetLibrary, library, kernel);
  if (library == nullptr || kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (!hip::PlatformState::Instance().GetFunctionLibrary(kernel, library)) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipKernelGetName(const char** name, hipKernel_t kernel) {
  HIP_INIT_API(hipKernelGetName, name, kernel);
  if (name == nullptr || kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipLibrary_t library;
  if (!hip::PlatformState::Instance().GetFunctionLibrary(kernel, &library)) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  auto l = reinterpret_cast<hip::LibraryContainer*>(library);
  auto ret = l->GetKernelName(name, kernel);

  HIP_RETURN(ret);
}

hipError_t hipKernelGetParamInfo(hipKernel_t kernel, size_t paramIndex, size_t* paramOffset,
                                 size_t* paramSize ) {
  HIP_INIT_API(hipKernelGetParamInfo, kernel, paramIndex, paramOffset, paramSize);
  if (kernel == nullptr || paramOffset == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const auto* const d_kernel = hip::asKernel(reinterpret_cast<hipFunction_t>(kernel));
  if (d_kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }
  const amd::KernelSignature& signature = d_kernel->signature();
  if (paramIndex >= signature.numParameters()) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const amd::KernelParameterDescriptor& desc = signature.at(paramIndex);
  *paramOffset = desc.offset_;
  if (paramSize != nullptr) {
    *paramSize = desc.size_;
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipKernelGetAttribute(int* pi, hipFunction_attribute attrib, hipKernel_t kernel,
                                 hipDevice_t dev) {
  HIP_INIT_API(hipKernelGetAttribute, pi, attrib, kernel, dev);
  if (pi == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  if (dev < 0 || static_cast<size_t>(dev) >= hip::g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  hipKernel_t target_kernel = kernel;
  hipLibrary_t library = nullptr;
  if (hip::PlatformState::Instance().GetFunctionLibrary(kernel, &library)) {
    auto* container = reinterpret_cast<hip::LibraryContainer*>(library);
    const char* kernel_name = nullptr;
    if (container->GetKernelName(&kernel_name, kernel) != hipSuccess || kernel_name == nullptr) {
      HIP_RETURN(hipErrorInvalidHandle);
    }
    hipError_t status = container->ResolveKernelForDevice(&target_kernel, kernel_name, dev);
    if (status != hipSuccess) {
      HIP_RETURN(status);
    }
  }

  const auto* const d_kernel = hip::asKernel(target_kernel);
  if (d_kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  const auto& device = *hip::g_devices[dev]->devices()[0];

  auto* dev_kernel = d_kernel->getDeviceKernel(device);
  if (dev_kernel == nullptr) {
    HIP_RETURN(hipErrorMissingConfiguration);
  }
  const auto* wrkGrpInfoPtr = dev_kernel->workGroupInfo();
  if (wrkGrpInfoPtr == nullptr) {
    HIP_RETURN(hipErrorMissingConfiguration);
  }
  const auto& wrkGrpInfo = *wrkGrpInfoPtr;
  std::scoped_lock lock(dev_kernel->attributeLock());

  switch (attrib) {
    case HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
      *pi = static_cast<int>(wrkGrpInfo.size_);
      break;
    case HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
      *pi = static_cast<int>(wrkGrpInfo.localMemSize_);
      break;
    case HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
      *pi = static_cast<int>(wrkGrpInfo.constMemSize_ - 1);
      break;
    case HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
      *pi = static_cast<int>(wrkGrpInfo.privateMemSize_);
      break;
    case HIP_FUNC_ATTRIBUTE_NUM_REGS:
      *pi = static_cast<int>(wrkGrpInfo.usedVGPRs_);
      break;
    case HIP_FUNC_ATTRIBUTE_PTX_VERSION:
    case HIP_FUNC_ATTRIBUTE_BINARY_VERSION:
      *pi = device.isa().versionMajor() * 10 + device.isa().versionMinor();
      break;
    case HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA:
      *pi = 0;
      break;
    case HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
      *pi = wrkGrpInfo.kernelPreferredShmemCarveout_;
      break;
    case HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: {
      const int alignmentSize = device.isa().ldsAlignment();
      *pi = amd::alignDown(wrkGrpInfo.kernelMaxDynamicSharedSizeBytes_, alignmentSize);
      break;
    }
    case HIP_FUNC_ATTRIBUTE_CLUSTER_DIM_MUST_BE_SET:
      *pi = wrkGrpInfo.hasClusterAttr_ || wrkGrpInfo.runtimeClusterSize_[0] != 0 ||
                    wrkGrpInfo.runtimeClusterSize_[1] != 0 ||
                    wrkGrpInfo.runtimeClusterSize_[2] != 0;
      break;
    case HIP_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH:
    case HIP_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT:
    case HIP_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: {
      const int index = attrib - HIP_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH;
      *pi = wrkGrpInfo.hasClusterAttr_ ? static_cast<int>(wrkGrpInfo.clusterSize_[index])
                                       : wrkGrpInfo.runtimeClusterSize_[index];
      break;
    }
    case HIP_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED:
      *pi = wrkGrpInfo.nonPortableClusterSizeAllowed_;
      break;
    case HIP_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE:
      *pi = wrkGrpInfo.clusterSchedulingPolicyPreference_;
      break;
    default:
      HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipKernelSetAttribute(hipFunction_attribute attrib, int value, hipKernel_t kernel,
                                 hipDevice_t dev) {
  HIP_INIT_API(hipKernelSetAttribute, attrib, value, kernel, dev);

  if (kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Kernel* d_kernel = hip::asKernel(kernel);
  if (d_kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  auto* currentDevice = hip::getCurrentDevice();
  const auto& devices = currentDevice->devices();
  if (dev < 0 || static_cast<size_t>(dev) >= devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  device::Kernel* deviceKernel = const_cast<device::Kernel*>(
      d_kernel->getDeviceKernel(*devices[dev]));
  if (deviceKernel == nullptr) {
    HIP_RETURN(hipErrorMissingConfiguration);
  }

  device::Kernel::WorkGroupInfo* wrkGrpInfo = deviceKernel->workGroupInfo();

  if (wrkGrpInfo == nullptr) {
    HIP_RETURN(hipErrorMissingConfiguration);
  }
  std::scoped_lock lock(deviceKernel->attributeLock());

  switch (attrib) {
    case HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
    case HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
    case HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
    case HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
    case HIP_FUNC_ATTRIBUTE_NUM_REGS:
    case HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA:
    case HIP_FUNC_ATTRIBUTE_PTX_VERSION:
    case HIP_FUNC_ATTRIBUTE_BINARY_VERSION:
      HIP_RETURN(hipErrorInvalidValue);
      break;
    case HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
      if ((value < 0) || (value > (wrkGrpInfo->availableLDSSize_ - wrkGrpInfo->localMemSize_))) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      wrkGrpInfo->kernelMaxDynamicSharedSizeBytes_ = value;
      wrkGrpInfo->maxDynamicSharedSizeBytes_ = value;
      break;
    case HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
      break;
    default:
      HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipKernelSetAttributeForDevice(hipKernel_t kernel, hipFuncAttribute attr, int value,
                                          int device) {
  HIP_INIT_API(hipKernelSetAttributeForDevice, kernel, attr, value, device);

  if (kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }
  if (device < 0 || static_cast<size_t>(device) >= hip::g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  switch (attr) {
    case hipFuncAttributeMaxDynamicSharedMemorySize:
    case hipFuncAttributePreferredSharedMemoryCarveout:
    case hipFuncAttributeRequiredClusterWidth:
    case hipFuncAttributeRequiredClusterHeight:
    case hipFuncAttributeRequiredClusterDepth:
    case hipFuncAttributeNonPortableClusterSizeAllowed:
    case hipFuncAttributeClusterSchedulingPolicyPreference:
      break;
    default:
      HIP_RETURN(hipErrorInvalidValue);
  }

  // Load code object for the target device if not already loaded
  hipKernel_t target_kernel = kernel;
  hipLibrary_t library = nullptr;
  if (hip::PlatformState::Instance().GetFunctionLibrary(kernel, &library)) {
    auto* container = reinterpret_cast<hip::LibraryContainer*>(library);
    const char* kernel_name = nullptr;
    if (container->GetKernelName(&kernel_name, kernel) != hipSuccess || kernel_name == nullptr) {
      HIP_RETURN(hipErrorInvalidResourceHandle);
    }
    hipError_t status =
        container->ResolveKernelForDevice(&target_kernel, kernel_name, device);
    if (status != hipSuccess) {
      HIP_RETURN(status);
    }
  }

  amd::Kernel* amd_kernel = hip::asKernel(target_kernel);
  if (amd_kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  const amd::Device& amd_device = *hip::g_devices[device]->devices()[0];
  amd::device::Kernel* device_kernel =
      const_cast<amd::device::Kernel*>(amd_kernel->getDeviceKernel(amd_device));
  if (device_kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  std::scoped_lock lock(device_kernel->attributeLock());
  amd::device::Kernel::WorkGroupInfo* work_group_info = device_kernel->workGroupInfo();
  if (work_group_info == nullptr) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  switch (attr) {
    case hipFuncAttributeMaxDynamicSharedMemorySize: {
      if (work_group_info->localMemSize_ > work_group_info->availableLDSSize_) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      const size_t maximum =
          work_group_info->availableLDSSize_ - work_group_info->localMemSize_;
      if (value < 0 || static_cast<size_t>(value) > maximum) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      work_group_info->kernelMaxDynamicSharedSizeBytes_ = value;
      if (!work_group_info->hasFuncMaxDynamicSharedSize_) {
        work_group_info->maxDynamicSharedSizeBytes_ = value;
      }
      break;
    }
    case hipFuncAttributePreferredSharedMemoryCarveout:
      if (value < -1 || value > 100) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      work_group_info->kernelPreferredShmemCarveout_ = value;
      if (!work_group_info->hasFuncPreferredShmemCarveout_) {
        work_group_info->groupMemCarveout_ = value;
      }
      break;
    case hipFuncAttributeRequiredClusterWidth:
    case hipFuncAttributeRequiredClusterHeight:
    case hipFuncAttributeRequiredClusterDepth: {
      if (value < 0) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      const int index = attr - hipFuncAttributeRequiredClusterWidth;
      if (work_group_info->hasClusterAttr_) {
        if (value != static_cast<int>(work_group_info->clusterSize_[index])) {
          HIP_RETURN(hipErrorInvalidValue);
        }
      } else {
        work_group_info->runtimeClusterSize_[index] = value;
      }
      break;
    }
    case hipFuncAttributeNonPortableClusterSizeAllowed:
      work_group_info->nonPortableClusterSizeAllowed_ = value != 0;
      break;
    case hipFuncAttributeClusterSchedulingPolicyPreference:
      if (value < hipClusterSchedulingPolicyDefault ||
          value > hipClusterSchedulingPolicyLoadBalancing) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      work_group_info->clusterSchedulingPolicyPreference_ = value;
      break;
    default:
      HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipKernelGetFunction(hipFunction_t* pFunc, hipKernel_t kernel) {
  HIP_INIT_API(hipKernelGetFunction, pFunc, kernel);

  if (pFunc == nullptr || kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pFunc = reinterpret_cast<hipFunction_t>(kernel);

  HIP_RETURN(hipSuccess);
}
}  // namespace hip
