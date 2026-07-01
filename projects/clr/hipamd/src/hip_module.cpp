/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <fstream>
#include <optional>

#include "hip_internal.hpp"
#include "platform/command.hpp"
#include "platform/ndrange.hpp"
#include "platform/program.hpp"
#include "hip_event.hpp"
#include "hip_platform.hpp"
#include "hip_comgr_helper.hpp"

#if defined(_MSC_VER) && !defined(__clang__)
#include <intsafe.h>
#endif

// returns true if there is overflow
static bool multiplyOverflow(__hip_uint32_t operand,
                             __hip_uint32_t multiplier,
                             __hip_uint32_t& result)
{
#if defined(__GNUC__)
  // gcc, clang and clang-cl
  return __builtin_mul_overflow(operand, multiplier, &result);
#else
  // cl.exe
  return UIntMult(operand, multiplier, &result);
#endif
}

namespace hip {

hipError_t ihipModuleLoadData(hipModule_t* module, const void* mmap_ptr, size_t mmap_size);

extern hipError_t ihipLaunchKernel(const void* hostFunction, dim3 gridDim, dim3 blockDim,
                                   void** args, size_t sharedMemBytes, hipStream_t stream,
                                   hipEvent_t startEvent, hipEvent_t stopEvent, int flags,
                                   dim3 clusterDim = {1, 1, 1},
                                   const amd::DynDataPrefetchConfig* dynDataPrefetchConfig = nullptr);

const std::string& FunctionName(const hipFunction_t f) {
  return hip::asKernel(f)->name();
}

hipError_t hipModuleUnload(hipModule_t hmod) {
  HIP_INIT_API(hipModuleUnload, hmod);
  if (hmod == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }
  CHECK_STREAM_CAPTURE_SUPPORTED();
  HIP_RETURN(PlatformState::Instance().UnloadModule(hmod));
}
hipError_t hipModuleLoadFatBinary(hipModule_t* module, const void* fatbin) {
  HIP_INIT_API(hipModuleLoadFatBinary, module, fatbin);
  HIP_RETURN(PlatformState::Instance().LoadModule(module, 0, fatbin));
  HIP_RETURN(hipSuccess);
}

hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
  HIP_INIT_API(hipModuleLoad, module, fname);

  HIP_RETURN(PlatformState::Instance().LoadModule(module, fname));
}

hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
  HIP_INIT_API(hipModuleLoadData, module, image);
  HIP_RETURN(PlatformState::Instance().LoadModule(module, 0, image));
}

hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image, unsigned int numOptions,
                               hipJitOption* options, void** optionsValues) {
  /* TODO: Pass options to Program */
  HIP_INIT_API(hipModuleLoadDataEx, module, image);
  HIP_RETURN(PlatformState::Instance().LoadModule(module, 0, image));
}

extern hipError_t __hipExtractCodeObjectFromFatBinary(
    const void* data, const std::vector<std::string>& devices,
    std::vector<std::pair<const void*, size_t>>& code_objs);

hipError_t hipModuleGetFunction(hipFunction_t* hfunc, hipModule_t hmod, const char* name) {
  HIP_INIT_API(hipModuleGetFunction, hfunc, hmod, name);

  if (hfunc == nullptr || name == nullptr || strlen(name) == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (hmod == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  if (hipSuccess != PlatformState::Instance().GetDynFunc(hfunc, hmod, name)) {
    LogPrintfInfo("Cannot find the function: %s for module: 0x%x", name, hmod);
    hip::tls.last_command_error_ = hipErrorNotFound;
    return hipErrorNotFound;
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipModuleGetFunctionCount(unsigned int* count, hipModule_t mod) {
  HIP_INIT_API(hipModuleGetFunctionCount, count, mod);

  if (mod == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }
  HIP_RETURN(PlatformState::Instance().GetFuncCount(count, mod));
}

hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                              const char* name) {
  HIP_INIT_API(hipModuleGetGlobal, dptr, bytes, hmod, name);

  if ((dptr == nullptr && bytes == nullptr) || name == nullptr || strlen(name) == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (hmod == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }
  /* Get address and size for the global symbol */
  if (hipSuccess != PlatformState::Instance().GetDynGlobalVar(name, hmod, dptr, bytes)) {
    LogPrintfError("Cannot find global Var: %s for module: 0x%x at device: %d", name, hmod,
                   ihipGetDevice());
    HIP_RETURN(hipErrorNotFound);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipFuncGetAttribute(int* value, hipFunction_attribute attrib, hipFunction_t hfunc) {
  HIP_INIT_API(hipFuncGetAttribute, value, attrib, hfunc);

  if (value == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Kernel* kernel = hip::asKernel(hfunc);
  if (kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  const device::Kernel::WorkGroupInfo* wrkGrpInfo =
      kernel->getDeviceKernel(*(hip::getCurrentDevice()->devices()[0]))->workGroupInfo();
  if (wrkGrpInfo == nullptr) {
    HIP_RETURN(hipErrorMissingConfiguration);
  }

  switch (attrib) {
    case HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
      *value = static_cast<int>(wrkGrpInfo->localMemSize_);
      break;
    case HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
      *value = static_cast<int>(wrkGrpInfo->size_);
      break;
    case HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
      *value = static_cast<int>(wrkGrpInfo->constMemSize_ - 1);
      break;
    case HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
      *value = static_cast<int>(wrkGrpInfo->privateMemSize_);
      break;
    case HIP_FUNC_ATTRIBUTE_NUM_REGS:
      *value = static_cast<int>(wrkGrpInfo->usedVGPRs_);
      break;
    case HIP_FUNC_ATTRIBUTE_PTX_VERSION:
    case HIP_FUNC_ATTRIBUTE_BINARY_VERSION:
      *value = hip::getCurrentDevice()->devices()[0]->isa().versionMajor() * 10 +
               hip::getCurrentDevice()->devices()[0]->isa().versionMinor();
      break;
    case HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA:
      *value = 0;
      break;
    case HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
      *value = static_cast<int>(wrkGrpInfo->availableLDSSize_ - wrkGrpInfo->localMemSize_);
      break;
    case HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
      *value = wrkGrpInfo->groupMemCarveout_;
      break;
    default:
      HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func) {
  HIP_INIT_API(hipFuncGetAttributes, attr, func);

  if (attr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (func == nullptr) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  HIP_RETURN_ONFAIL(PlatformState::Instance().StatCO().GetFuncAttr(attr, func, ihipGetDevice()));

  HIP_RETURN(hipSuccess);
}

inline hipError_t GetDeviceKernel(const void* func, device::Kernel** d_kernel) {
  hipFunction_t h_func = nullptr;

  hipError_t err = PlatformState::Instance().StatCO().GetFunc(&h_func, func, ihipGetDevice());
  if (h_func == nullptr) {
    if (PlatformState::Instance().IsValidDynFunc(func)) {
      h_func = reinterpret_cast<hipFunction_t>(const_cast<void*>(func));
    } else {
      return hipErrorInvalidDeviceFunction;
    }
  }

  amd::Kernel* kernel = hip::asKernel(h_func);

  if (kernel == nullptr) {
    return hipErrorInvalidDeviceFunction;
  }
  *(d_kernel) = const_cast<device::Kernel*>(
      kernel->getDeviceKernel(*(hip::getCurrentDevice()->devices()[0])));
  return hipSuccess;
}

hipError_t hipFuncSetAttribute(const void* func, hipFuncAttribute attr, int value) {
  HIP_INIT_API(hipFuncSetAttribute, func, attr, value);

  if (func == nullptr) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }
  if (attr < 0 || attr > hipFuncAttributeMax) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipFunction_t h_func = nullptr;

  hipError_t err = PlatformState::Instance().StatCO().GetFunc(&h_func, func, ihipGetDevice());
  if (h_func == nullptr) {
    if (PlatformState::Instance().IsValidDynFunc((func))) {
      h_func = reinterpret_cast<hipFunction_t>(const_cast<void*>(func));
    } else {
      HIP_RETURN(hipErrorInvalidDeviceFunction);
    }
  }

  amd::Kernel* kernel = hip::asKernel(h_func);

  if (kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }
  device::Kernel* d_kernel =
      (device::Kernel*)(kernel->getDeviceKernel(*(hip::getCurrentDevice()->devices()[0])));

  if (attr == hipFuncAttributeMaxDynamicSharedMemorySize) {
    if ((value < 0) ||
        (value > (d_kernel->workGroupInfo()->availableLDSSize_ -
                  d_kernel->workGroupInfo()->localMemSize_))) {
      HIP_RETURN(hipErrorInvalidValue);
    }
    d_kernel->workGroupInfo()->maxDynamicSharedSizeBytes_ = value;
  }

  if (attr == hipFuncAttributePreferredSharedMemoryCarveout) {
    if (value < -1 || value > 100) {
      HIP_RETURN(hipErrorInvalidValue);
    }
    d_kernel->workGroupInfo()->groupMemCarveout_ = value;
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t cacheConfig) {
  HIP_INIT_API(hipFuncSetCacheConfig, cacheConfig);

  if (func == nullptr) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }
  if (cacheConfig != hipFuncCachePreferNone && cacheConfig != hipFuncCachePreferShared &&
      cacheConfig != hipFuncCachePreferL1 && cacheConfig != hipFuncCachePreferEqual) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  device::Kernel* d_kernel;
  hipError_t status = GetDeviceKernel(func, &d_kernel);

  if (hipSuccess != status) {
    HIP_RETURN(status);
  }
  d_kernel->workGroupInfo()->groupMemCarveout_ =
      amd::funcCacheToCarveoutPercent(static_cast<uint32_t>(cacheConfig));

  HIP_RETURN(hipSuccess);
}

hipError_t hipFuncSetSharedMemConfig(const void* func, hipSharedMemConfig config) {
  HIP_INIT_API(hipFuncSetSharedMemConfig, func, config);

  if (func == nullptr) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }
  if (config != hipSharedMemBankSizeDefault && config != hipSharedMemBankSizeFourByte &&
      config != hipSharedMemBankSizeEightByte) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // No way to set shared memory config yet

  HIP_RETURN(hipSuccess);
}

hipError_t ihipLaunchKernel_validate(hipFunction_t f, const amd::LaunchParams& launch_params,
                                     void** kernelParams, void** extra, int deviceId,
                                     uint32_t params = 0) {
  if (f == nullptr) {
    LogPrintfError("%s", "Function passed is null");
    return hipErrorInvalidImage;
  }
  if ((kernelParams != nullptr) && (extra != nullptr)) {
    LogPrintfError("%s",
                   "Both, kernelParams and extra Params are provided, only one should be provided");
    return hipErrorInvalidValue;
  }

  if (launch_params.global_[0] == 0 || launch_params.global_[1] == 0 ||
      launch_params.global_[2] == 0) {
    return hipErrorInvalidConfiguration;
  }

  if (launch_params.local_[0] == 0 || launch_params.local_[1] == 0 ||
      launch_params.local_[2] == 0) {
    return hipErrorInvalidConfiguration;
  }

  const amd::Device* device = g_devices[deviceId]->devices()[0];
  const auto& info = device->info();
  if (launch_params.sharedMemBytes_ > info.localMemSizePerCU_) {  // sharedMemPerBlock
    return hipErrorInvalidValue;
  }

  // Make sure dispatch doesn't exceed max workgroup size limit
  if (launch_params.local_.product() > info.maxWorkGroupSize_) {
    return hipErrorInvalidConfiguration;
  }
  amd::Kernel* kernel = hip::asKernel(f);
  if (!kernel) {
    LogPrintfError("%s", "Kernel object is invalid or null, possibly due to architecture mismatch.");
    return hipErrorInvalidValue;
  }
  const amd::KernelSignature& signature = kernel->signature();
  if ((signature.numParameters() > 0) && (kernelParams == nullptr) && (extra == nullptr)) {
    LogPrintfError("%s", "At least one of kernelParams or extra Params should be provided");
    return hipErrorInvalidValue;
  }
  if (!kernel->getDeviceKernel(*device)) {
    return hipErrorInvalidDevice;
  }
  // Make sure the launch params are not larger than if specified launch_bounds
  // If it exceeds, then return a failure
  if (launch_params.local_.product() > kernel->getDeviceKernel(*device)->workGroupInfo()->size_) {
    LogPrintfError("Launch params (%u, %u, %u) are larger than launch bounds (%lu) for kernel %s",
                   launch_params.local_[0], launch_params.local_[1], launch_params.local_[2],
                   kernel->getDeviceKernel(*device)->workGroupInfo()->size_,
                   kernel->name().c_str());
    return hipErrorLaunchFailure;
  }

  if (params & amd::NDRangeKernelCommand::CooperativeGroups) {
    if (!device->info().cooperativeGroups_) {
      return hipErrorLaunchFailure;
    }
    int num_blocks = 0;
    int max_blocks_per_grid = 0;
    int best_block_size = 0;
    int block_size = launch_params.local_.product();
    hipError_t err = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, &max_blocks_per_grid, &best_block_size, *device, f, block_size,
        launch_params.sharedMemBytes_, true);
    if (err != hipSuccess) {
      return err;
    }
    if (((launch_params.global_.product()) / block_size) > unsigned(max_blocks_per_grid)) {
      return hipErrorCooperativeLaunchTooLarge;
    }
  }
  if (params & amd::NDRangeKernelCommand::CooperativeMultiDeviceGroups) {
    if (!device->info().cooperativeMultiDeviceGroups_) {
      return hipErrorLaunchFailure;
    }
  }
  return hipSuccess;
}

// =================================================================================================
bool UpdateNumClustersFromKernel(const hip::Stream* stream, const amd::Kernel* kernel,
                                 amd::LaunchParams& launch_params) {

  const amd::Device& device = stream->vdev()->device();
  amd::device::Kernel* devKernel = const_cast<device::Kernel*>(kernel->getDeviceKernel(device));
  // If cluster size from device kernel is > 1, then we need to update the cluster params.
  if (devKernel->getClusterSize(0) > 1 || devKernel->getClusterSize(1) > 1 ||
      devKernel->getClusterSize(2) > 1) {
    if (!launch_params.UpdateClusterLaunchParams(devKernel->getClusterSize(0),
                                                 devKernel->getClusterSize(1),
                                                 devKernel->getClusterSize(2))) {
      LogPrintfError("This is not a valid Cluster Launch, please recheck parameters"
                     "global[0]: %d, global[1]: %d, global[2]: %d, local[0]: %d, local[1]: %d,"
                     "local[2] :%d, numClusters[0]: %d, numClusters[1]: %d, numClusters[2]: %d",
                      launch_params.global_[0], launch_params.grid_[1],
                      launch_params.global_[2], launch_params.local_[0],
                      launch_params.local_[1], launch_params.local_[2],
                      devKernel->getClusterSize(0), devKernel->getClusterSize(1),
                      devKernel->getClusterSize(2));
      return false;
    }
  }
  return true;
}

// =================================================================================================
hipError_t ihipLaunchKernelCommand(amd::Command*& command, hipFunction_t f,
                                   amd::LaunchParams& launch_params, hip::Stream* stream,
                                   void** kernelParams, void** extra,
                                   hipEvent_t startEvent = nullptr, hipEvent_t stopEvent = nullptr,
                                   uint32_t flags = 0, uint32_t params = 0, uint32_t gridId = 0,
                                   uint32_t numGrids = 0, uint64_t prevGridSum = 0,
                                   uint64_t allGridSum = 0, uint32_t firstDevice = 0) {
  amd::Kernel* kernel = hip::asKernel(f);

  // Check if the kernel metadata has cluster info we need to act on.
  if (!UpdateNumClustersFromKernel(stream, kernel, launch_params)) {
    return hipErrorInvalidValue;
  }

  size_t globalWorkOffset[3] = {0};
  amd::NDRangeContainer ndrange(3, globalWorkOffset, launch_params.global_.Data(),
                                launch_params.local_.Data(), launch_params.cluster_.Data());
  amd::Command::EventWaitList waitList;

  bool profileNDRange = (startEvent != nullptr || stopEvent != nullptr);

  // Flag set to 1 signifies that kernel can be launched in anyorder
  if (flags & hipExtAnyOrderLaunch) {
    params |= amd::NDRangeKernelCommand::AnyOrderLaunch;
  }

  amd::NDRangeKernelCommand* kernelCommand = new amd::NDRangeKernelCommand(
      *stream, waitList, *kernel, ndrange, launch_params.sharedMemBytes_, params, gridId, numGrids,
      prevGridSum, allGridSum, firstDevice, profileNDRange);
  address kernargs = nullptr;
  size_t kernargs_size = 0;
  // 'extra' is a struct that contains the following info: {
  //   HIP_LAUNCH_PARAM_BUFFER_POINTER, kernargs,
  //   HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernargs_size,
  //   HIP_LAUNCH_PARAM_END }
  if (extra != nullptr) {
    assert(kernelParams == nullptr);
    if (extra[0] != HIP_LAUNCH_PARAM_BUFFER_POINTER || extra[2] != HIP_LAUNCH_PARAM_BUFFER_SIZE ||
        extra[4] != HIP_LAUNCH_PARAM_END) {
      kernelCommand->release();
      return hipErrorInvalidValue;
    }
    kernargs = reinterpret_cast<address>(extra[1]);
    kernargs_size = *reinterpret_cast<size_t*>(extra[3]);
    const uint32_t numParams = kernel->signature().numParameters();
    const bool expectsArgs = (numParams > 0);
    const bool hasArgs = (kernargs != nullptr && kernargs_size > 0);
    // we either expected args but got none, or didn’t expect any but got some
    if (expectsArgs == true && hasArgs == false) {
      return hipErrorInvalidValue;
    }
    if (expectsArgs == false && kernargs_size != 0) {
      return hipErrorLaunchOutOfResources;
    }
  }

  if (CL_SUCCESS != kernelCommand->captureHIPArgsAndValidate(kernelParams, kernargs, kernargs_size)) {
    kernelCommand->release();
    return hipErrorOutOfMemory;
  }

  command = kernelCommand;

  return hipSuccess;
}

hipError_t ihipModuleLaunchKernel(hipFunction_t f, amd::LaunchParams& launch_params,
                                  hipStream_t hStream, void** kernelParams, void** extra,
                                  hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags = 0,
                                  uint32_t params = 0, uint32_t gridId = 0, uint32_t numGrids = 0,
                                  uint64_t prevGridSum = 0, uint64_t allGridSum = 0,
                                  uint32_t firstDevice = 0,
                                  const amd::DynDataPrefetchConfig* dynDataPrefetchConfig = nullptr) {
  int deviceId = hip::Stream::DeviceId(hStream);

  // Ensure the stream's device matches the current device,
  // or the grid's assigned device in CooperativeKernelMultiDevice mode
  int targetDevice = (numGrids == 0) ? ihipGetDevice() : gridId;
  if (deviceId != targetDevice) {
    return hipErrorInvalidResourceHandle;
  }
  HIP_RETURN_ONFAIL(PlatformState::Instance().StatCO().InitManagedVarDevicePtr(deviceId));

  if (f == nullptr) {
    LogPrintfError("%s", "Function passed is null");
    return hipErrorInvalidResourceHandle;
  }
  amd::Kernel* kernel = hip::asKernel(f);

  hipError_t status =
      ihipLaunchKernel_validate(f, launch_params, kernelParams, extra, deviceId, params);
  if (status != hipSuccess) {
    return status;
  }

  // Make sure the app doesn't launch a workgroup bigger than the global size
  if (launch_params.global_[0] < launch_params.local_[0]) {
    launch_params.local_[0] = launch_params.global_[0];
  }
  if (launch_params.global_[1] < launch_params.local_[1]) {
    launch_params.local_[1] = launch_params.global_[1];
  }
  if (launch_params.global_[2] < launch_params.local_[2]) {
    launch_params.local_[2] = launch_params.global_[2];
  }

  auto device = g_devices[deviceId]->devices()[0];
  // Check if it's a uniform kernel and validate dimensions
  if (kernel->getDeviceKernel(*device)->getUniformWorkGroupSize()) {
    if (((launch_params.global_[0] % launch_params.local_[0]) != 0) ||
        ((launch_params.global_[1] % launch_params.local_[1]) != 0) ||
        ((launch_params.global_[2] % launch_params.local_[2]) != 0)) {
      return hipErrorInvalidValue;
    }
  }
  amd::Command* command = nullptr;
  hip::Stream* hip_stream = hip::getStream(hStream);
  status = ihipLaunchKernelCommand(command, f, launch_params, hip_stream, kernelParams, extra,
                                   startEvent, stopEvent, flags, params, gridId, numGrids,
                                   prevGridSum, allGridSum, firstDevice);
  if (status != hipSuccess) {
    return status;
  }

  if (dynDataPrefetchConfig != nullptr && dynDataPrefetchConfig->isEnabled()) {
    amd::NDRangeKernelCommand* kernelCmd = static_cast<amd::NDRangeKernelCommand*>(command);
    kernelCmd->setDynDataPrefetchConfig(*dynDataPrefetchConfig);
  }

  if (startEvent != nullptr) {
    hip::Event* eStart = reinterpret_cast<hip::Event*>(startEvent);
    status = eStart->addMarker(hip_stream, nullptr);
    if (status != hipSuccess) {
      return status;
    }
  }

  if (stopEvent != nullptr) {
    hip::Event* eStop = reinterpret_cast<hip::Event*>(stopEvent);
    if (eStop->flags() & hipEventDisableSystemFence) {
      command->setCommandEntryScope(amd::Device::kCacheStateIgnore);
    } else {
      command->setCommandEntryScope(amd::Device::kCacheStateSystem);
    }
    // Enqueue Dispatch and bind the stop event
    command->enqueue();
    eStop->BindCommand(*command);
  } else {
    command->enqueue();
  }

  if (command->status() == CL_INVALID_OPERATION) {
    command->release();
    return hipErrorIllegalState;
  }

  command->release();

  return hipSuccess;
}

hipError_t hipModuleLaunchKernel(hipFunction_t f, uint32_t gridDimX, uint32_t gridDimY,
                                 uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY,
                                 uint32_t blockDimZ, uint32_t sharedMemBytes, hipStream_t hStream,
                                 void** kernelParams, void** extra) {
  HIP_INIT_API(hipModuleLaunchKernel, f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
               blockDimZ, sharedMemBytes, hStream, kernelParams, extra);

  if (!hip::isValid(hStream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }
  CHECK_STREAM_DETACHED_API(hStream);

  int deviceId = hip::Stream::DeviceId(hStream);
  const amd::Device* device = g_devices[deviceId]->devices()[0];

  STREAM_CAPTURE(hipModuleLaunchKernel, hStream, f, gridDimX, gridDimY, gridDimZ, blockDimX,
                 blockDimY, blockDimZ, sharedMemBytes, kernelParams, extra);

  amd::HIPLaunchParams launch_params(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                     sharedMemBytes, *device, 0, 0, 0, 1, 1, 1);
  if (!launch_params.IsValidConfig() ||
       launch_params.local_.product() > device->info().maxWorkGroupSize_) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (sharedMemBytes > device->info().localMemSizePerCU_) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (launch_params.global_[0] == 0 || launch_params.global_[1] == 0
      || launch_params.global_[2] == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (launch_params.local_[0] == 0 || launch_params.local_[1] == 0
                                    || launch_params.local_[2] == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipModuleLaunchKernel(f, launch_params, hStream, kernelParams, extra, nullptr,
                                    nullptr));
}

hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags) {
  HIP_INIT_API(hipExtModuleLaunchKernel, f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
               localWorkSizeX, localWorkSizeY, localWorkSizeZ, sharedMemBytes, hStream,
               kernelParams, extra, startEvent, stopEvent, flags);

  if (!hip::isValid(hStream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }
  CHECK_STREAM_DETACHED_API(hStream);

  int deviceId = hip::Stream::DeviceId(hStream);
  const amd::Device* device = g_devices[deviceId]->devices()[0];

  STREAM_CAPTURE(hipExtModuleLaunchKernel, hStream, f, globalWorkSizeX, globalWorkSizeY,
                 globalWorkSizeZ, localWorkSizeX, localWorkSizeY, localWorkSizeZ, sharedMemBytes,
                 kernelParams, extra, startEvent, stopEvent, flags);

  amd::LaunchParams launch_params(globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX,
                                  localWorkSizeY, localWorkSizeZ, sharedMemBytes, *device, 1, 1, 1, 1,
                                  1, 1, false);

  if (!launch_params.IsValidConfig() ||
       launch_params.local_.product() > device->info().maxWorkGroupSize_) {
    HIP_RETURN(hipErrorInvalidConfiguration);
  }

  if (sharedMemBytes > device->info().localMemSizePerCU_) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (launch_params.global_[0] == 0 || launch_params.global_[1] == 0
      || launch_params.global_[2] == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (launch_params.local_[0] == 0 || launch_params.local_[1] == 0
                                    || launch_params.local_[2] == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipModuleLaunchKernel(f, launch_params, hStream, kernelParams, extra, startEvent,
                                    stopEvent, flags));
}


hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                                    size_t sharedMemBytes, hipStream_t hStream, void** kernelParams,
                                    void** extra, hipEvent_t startEvent, hipEvent_t stopEvent) {
  HIP_INIT_API(hipHccModuleLaunchKernel, f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
               blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra,
               startEvent, stopEvent);

  CHECK_STREAM_DETACHED_API(hStream);

  int deviceId = hip::Stream::DeviceId(hStream);
  const amd::Device* device = g_devices[deviceId]->devices()[0];
  amd::LaunchParams launch_params(globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, blockDimX,
                                  blockDimY, blockDimZ, sharedMemBytes, *device, 1, 1, 1, 1, 1, 1,
                                  false);

  HIP_RETURN(ihipModuleLaunchKernel(f, launch_params, hStream, kernelParams, extra, startEvent,
                                    stopEvent));
}

hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int gridDimX,
                                            unsigned int gridDimY, unsigned int gridDimZ,
                                            unsigned int blockDimX, unsigned int blockDimY,
                                            unsigned int blockDimZ, unsigned int sharedMemBytes,
                                            hipStream_t stream, void** kernelParams) {
  HIP_INIT_API(hipModuleLaunchCooperativeKernel, f, gridDimX, gridDimY, gridDimZ, blockDimX,
               blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams);

  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }
  CHECK_STREAM_DETACHED_API(stream);

  int deviceId = hip::Stream::DeviceId(stream);
  const amd::Device* device = g_devices[deviceId]->devices()[0];

  STREAM_CAPTURE(hipModuleLaunchCooperativeKernel, stream, f, gridDimX, gridDimY, gridDimZ,
                 blockDimX, blockDimY, blockDimZ, sharedMemBytes, kernelParams);

  amd::HIPLaunchParams launch_params(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                     sharedMemBytes, *device, 0, 0, 0, 1, 1, 1);

  if (!launch_params.IsValidConfig() ||
      launch_params.local_.product() > device->info().maxWorkGroupSize_) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (sharedMemBytes > device->info().localMemSizePerCU_) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (launch_params.global_[0] == 0 || launch_params.global_[1] == 0 ||
      launch_params.global_[2] == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (launch_params.local_[0] == 0 || launch_params.local_[1] == 0 ||
      launch_params.local_[2] == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipModuleLaunchKernel(f, launch_params, stream, kernelParams, nullptr, nullptr,
                                    nullptr, 0, amd::NDRangeKernelCommand::CooperativeGroups));
}

hipError_t ihipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams* launchParamsList,
                                                        unsigned int numDevices, unsigned int flags,
                                                        uint32_t extFlags) {
  int numActiveGPUs = 0;
  hipError_t result = hipSuccess;
  result = ihipDeviceGetCount(&numActiveGPUs);

  if ((numDevices == 0) || (numDevices > numActiveGPUs)) {
    return hipErrorInvalidValue;
  }

  if (flags >
      (hipCooperativeLaunchMultiDeviceNoPostSync + hipCooperativeLaunchMultiDeviceNoPreSync)) {
    return hipErrorInvalidValue;
  }

  uint64_t allGridSize = 0;
  std::vector<const amd::Device*> mgpu_list(numDevices);

  for (int i = 0; i < numDevices; ++i) {
    uint32_t blockDims = 0;
    const hipFunctionLaunchParams& launch = launchParamsList[i];
    blockDims = launch.blockDimX * launch.blockDimY * launch.blockDimZ;
    allGridSize += launch.gridDimX * launch.gridDimY * launch.gridDimZ * blockDims;

    // Make sure block dimensions are valid
    if (0 == blockDims) {
      return hipErrorInvalidConfiguration;
    }
    if (launch.hStream != nullptr) {
      // Validate devices to make sure it dosn't have duplicates
      hip::Stream* hip_stream = reinterpret_cast<hip::Stream*>(launch.hStream);
      auto device = &hip_stream->vdev()->device();
      for (int j = 0; j < numDevices; ++j) {
        if (mgpu_list[j] == device) {
          return hipErrorInvalidDevice;
        }
      }
      mgpu_list[i] = device;
    } else {
      return hipErrorInvalidResourceHandle;
    }
  }
  uint64_t prevGridSize = 0;
  uint32_t firstDevice = 0;

  // Sync the execution streams on all devices
  if ((flags & hipCooperativeLaunchMultiDeviceNoPreSync) == 0) {
    for (int i = 0; i < numDevices; ++i) {
      hip::Stream* hip_stream = reinterpret_cast<hip::Stream*>(launchParamsList[i].hStream);
      hip_stream->finish();
    }
  }

  // Grid and Block dimensions should match across devices, as well as sharedMemBytes
  for (uint32_t i = 1; i < numDevices; ++i) {
    if (launchParamsList[i - 1].gridDimX != launchParamsList[i].gridDimX ||
        launchParamsList[i - 1].gridDimY != launchParamsList[i].gridDimY ||
        launchParamsList[i - 1].gridDimZ != launchParamsList[i].gridDimZ ||
        launchParamsList[i - 1].blockDimX != launchParamsList[i].blockDimX ||
        launchParamsList[i - 1].blockDimY != launchParamsList[i].blockDimY ||
        launchParamsList[i - 1].blockDimZ != launchParamsList[i].blockDimZ) {
      return hipErrorInvalidValue;
    }

    if (launchParamsList[i - 1].sharedMemBytes != launchParamsList[i].sharedMemBytes) {
      return hipErrorInvalidValue;
    }
  }

  for (int i = 0; i < numDevices; ++i) {
    const hipFunctionLaunchParams& launch = launchParamsList[i];
    hip::Stream* hip_stream = reinterpret_cast<hip::Stream*>(launch.hStream);

    if (i == 0) {
      // The order of devices in the launch may not match the order in the global array
      for (size_t dev = 0; dev < g_devices.size(); ++dev) {
        // Find the matching device
        if (&hip_stream->vdev()->device() == g_devices[dev]->devices()[0]) {
          // Save ROCclr index of the first device in the launch
          firstDevice = hip_stream->vdev()->device().index();
          break;
        }
      }
    }

    const amd::Device& device = hip_stream->vdev()->device();
    amd::HIPLaunchParams launch_params(launch.gridDimX, launch.gridDimY, launch.gridDimZ,
                                       launch.blockDimX, launch.blockDimY, launch.blockDimZ,
                                       launch.sharedMemBytes, device, 0, 0, 0, 1, 1, 1);

    if (!launch_params.IsValidConfig()) {
      return hipErrorInvalidConfiguration;
    }

    result = ihipModuleLaunchKernel(launch.function, launch_params, launch.hStream,
                                    launch.kernelParams, nullptr, nullptr, nullptr, flags, extFlags,
                                    i, numDevices, prevGridSize, allGridSize, firstDevice);
    if (result != hipSuccess) {
      break;
    }
    prevGridSize += launch_params.global_.product();
  }

  // Sync the execution streams on all devices
  if ((flags & hipCooperativeLaunchMultiDeviceNoPostSync) == 0) {
    for (int i = 0; i < numDevices; ++i) {
      hip::Stream* hip_stream = reinterpret_cast<hip::Stream*>(launchParamsList[i].hStream);
      hip_stream->finish();
    }
  }

  return result;
}

hipError_t hipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams* launchParamsList,
                                                       unsigned int numDevices,
                                                       unsigned int flags) {
  HIP_INIT_API(hipModuleLaunchCooperativeKernelMultiDevice, launchParamsList, numDevices, flags);

  if (launchParamsList == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Validate all streams passed by user
  for (int i = 0; i < numDevices; ++i) {
    if (!hip::isValid(launchParamsList[i].hStream)) {
      HIP_RETURN(hipErrorInvalidValue);
    }
    CHECK_STREAM_DETACHED_API(launchParamsList[i].hStream);
  }

  HIP_RETURN(ihipModuleLaunchCooperativeKernelMultiDevice(
      launchParamsList, numDevices, flags,
      (amd::NDRangeKernelCommand::CooperativeGroups |
       amd::NDRangeKernelCommand::CooperativeMultiDeviceGroups)));
}

hipError_t ihipGetFuncBySymbol(hipFunction_t* functionPtr, const void* symbolPtr) {
  hipError_t hip_error =
      PlatformState::Instance().StatCO().GetFunc(functionPtr, symbolPtr, ihipGetDevice());
  if ((hip_error != hipSuccess) || (functionPtr == nullptr)) {
    return hipErrorInvalidDeviceFunction;
  }
  return hipSuccess;
}

hipError_t hipGetFuncBySymbol(hipFunction_t* functionPtr, const void* symbolPtr) {
  HIP_INIT_API(hipGetFuncBySymbol, functionPtr, symbolPtr);
  HIP_RETURN(ihipGetFuncBySymbol(functionPtr, symbolPtr));
}

hipError_t hipLaunchKernel_common(const void* hostFunction, dim3 gridDim, dim3 blockDim,
                                  void** args, size_t sharedMemBytes, hipStream_t stream,
                                  dim3 clusterDim = {1, 1, 1},
                                  const amd::DynDataPrefetchConfig* dynDataPrefetchConfig = nullptr) {
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  CHECK_STREAM_DETACHED(stream);
  STREAM_CAPTURE(hipLaunchKernel, stream, hostFunction, gridDim, blockDim, args, sharedMemBytes);
  return ihipLaunchKernel(hostFunction, gridDim, blockDim, args, sharedMemBytes, stream, nullptr,
                          nullptr, 0, clusterDim, dynDataPrefetchConfig);
}

hipError_t hipLaunchKernel(const void* hostFunction, dim3 gridDim, dim3 blockDim, void** args,
                           size_t sharedMemBytes, hipStream_t stream) {
  HIP_INIT_API(hipLaunchKernel, hostFunction, gridDim, blockDim, args, sharedMemBytes, stream);
  HIP_RETURN_DURATION(
      hipLaunchKernel_common(hostFunction, gridDim, blockDim, args, sharedMemBytes, stream));
}

hipError_t hipLaunchKernel_spt(const void* hostFunction, dim3 gridDim, dim3 blockDim, void** args,
                               size_t sharedMemBytes, hipStream_t stream) {
  HIP_INIT_API(hipLaunchKernel, hostFunction, gridDim, blockDim, args, sharedMemBytes, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipLaunchKernel_common(hostFunction, gridDim, blockDim, args, sharedMemBytes, stream));
}

hipError_t hipExtLaunchKernel(const void* hostFunction, dim3 gridDim, dim3 blockDim, void** args,
                              size_t sharedMemBytes, hipStream_t stream, hipEvent_t startEvent,
                              hipEvent_t stopEvent, int flags) {
  HIP_INIT_API(hipExtLaunchKernel, hostFunction, gridDim, blockDim, args, sharedMemBytes, stream,
               startEvent, stopEvent, flags);


  if (!hip::isValid(stream) || !hip::isValid(startEvent) || !hip::isValid(stopEvent)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  CHECK_STREAM_DETACHED_API(stream);

  STREAM_CAPTURE(hipExtLaunchKernel, stream, hostFunction, gridDim, blockDim, args, sharedMemBytes,
                 startEvent, stopEvent, flags);
  HIP_RETURN(ihipLaunchKernel(hostFunction, gridDim, blockDim, args, sharedMemBytes, stream,
                              startEvent, stopEvent, flags));
}

hipError_t hipLaunchCooperativeKernel_common(const void* f, dim3 gridDim, dim3 blockDim,
                                             void** kernelParams, uint32_t sharedMemBytes,
                                             hipStream_t hStream) {
  if (!hip::isValid(hStream)) {
    return hipErrorInvalidValue;
  }
  CHECK_STREAM_DETACHED(hStream);

  STREAM_CAPTURE(hipLaunchCooperativeKernel, hStream, f, gridDim, blockDim, kernelParams,
                 sharedMemBytes);

  if (f == nullptr) {
    return hipErrorInvalidDeviceFunction;
  }

  hipFunction_t func = nullptr;
  int deviceId = hip::Stream::DeviceId(hStream);
  hipError_t getStatFuncError = PlatformState::Instance().StatCO().GetFunc(&func, f, deviceId);
  if (getStatFuncError != hipSuccess) {
    return getStatFuncError;
  }
  const amd::Device* device = g_devices[deviceId]->devices()[0];

  amd::HIPLaunchParams launch_params(gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
                                     blockDim.z, sharedMemBytes, *device, 0, 0, 0, 1, 1, 1);

  if (!launch_params.IsValidConfig() ||
      launch_params.local_.product() > device->info().maxWorkGroupSize_) {
    return hipErrorInvalidConfiguration;
  }

  if (sharedMemBytes > device->info().localMemSizePerCU_) {
    return hipErrorCooperativeLaunchTooLarge;
  }

  return ihipModuleLaunchKernel(func, launch_params, hStream, kernelParams, nullptr,
                                nullptr, nullptr, 0, amd::NDRangeKernelCommand::CooperativeGroups);
}

hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDim,
                                      void** kernelParams, uint32_t sharedMemBytes,
                                      hipStream_t hStream) {
  HIP_INIT_API(hipLaunchCooperativeKernel, f, gridDim, blockDim, sharedMemBytes, hStream);
  HIP_RETURN(hipLaunchCooperativeKernel_common(f, gridDim, blockDim, kernelParams, sharedMemBytes,
                                               hStream));
}

hipError_t hipLaunchCooperativeKernel_spt(const void* f, dim3 gridDim, dim3 blockDim,
                                          void** kernelParams, uint32_t sharedMemBytes,
                                          hipStream_t hStream) {
  HIP_INIT_API(hipLaunchCooperativeKernel, f, gridDim, blockDim, sharedMemBytes, hStream);
  PER_THREAD_DEFAULT_STREAM(hStream);
  HIP_RETURN(hipLaunchCooperativeKernel_common(f, gridDim, blockDim, kernelParams, sharedMemBytes,
                                               hStream));
}

hipError_t ihipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices,
                                                  unsigned int flags, uint32_t extFlags) {
  if (launchParamsList == nullptr) {
    return hipErrorInvalidValue;
  }
  if (numDevices > g_devices.size()) {
    return hipErrorInvalidDevice;
  }

  std::vector<hipFunctionLaunchParams> functionLaunchParamsList(numDevices);
  // Convert hipLaunchParams to hipFunctionLaunchParams
  for (int i = 0; i < numDevices; ++i) {
    hipLaunchParams& launch = launchParamsList[i];
    // Validate stream passed by user
    if (!hip::isValid(launch.stream)) {
      return hipErrorInvalidValue;
    }

    if (launch.stream == nullptr || launch.stream == hipStreamLegacy) {
      return hipErrorInvalidResourceHandle;
    }
    CHECK_STREAM_DETACHED(launch.stream);

    // Not supported while stream is capturing
    hip::Stream* s = reinterpret_cast<hip::Stream*>(launch.stream);
    if (s->GetCaptureStatus() == hipStreamCaptureStatusActive) {
      s->SetCaptureStatus(hipStreamCaptureStatusInvalidated);
      return hipErrorStreamCaptureUnsupported;
    }
    if (s->GetCaptureStatus() == hipStreamCaptureStatusInvalidated) {
      return hipErrorStreamCaptureInvalidated;
    }

    hip::Stream* hip_stream = hip::getStream(launch.stream);
    hipFunction_t func = nullptr;
    // The order of devices in the launch may not match the order in the global array
    for (size_t dev = 0; dev < g_devices.size(); ++dev) {
      // Find the matching device and request the kernel function
      if (&hip_stream->vdev()->device() == g_devices[dev]->devices()[0]) {
        IHIP_RETURN_ONFAIL(PlatformState::Instance().StatCO().GetFunc(&func, launch.func, dev));
        break;
      }
    }
    if (func == nullptr) {
      return hipErrorInvalidDeviceFunction;
    }

    // functions should match across all devices
    if (i > 0 && launch.func != launchParamsList[i - 1].func) {
      return hipErrorInvalidValue;
    }

    functionLaunchParamsList[i].function = func;
    functionLaunchParamsList[i].gridDimX = launch.gridDim.x;
    functionLaunchParamsList[i].gridDimY = launch.gridDim.y;
    functionLaunchParamsList[i].gridDimZ = launch.gridDim.z;
    functionLaunchParamsList[i].blockDimX = launch.blockDim.x;
    functionLaunchParamsList[i].blockDimY = launch.blockDim.y;
    functionLaunchParamsList[i].blockDimZ = launch.blockDim.z;
    functionLaunchParamsList[i].sharedMemBytes = launch.sharedMem;
    functionLaunchParamsList[i].hStream = launch.stream;
    functionLaunchParamsList[i].kernelParams = launch.args;
  }

  return ihipModuleLaunchCooperativeKernelMultiDevice(
      functionLaunchParamsList.data(), functionLaunchParamsList.size(), flags, extFlags);
}

hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices,
                                                 unsigned int flags) {
  HIP_INIT_API(hipLaunchCooperativeKernelMultiDevice, launchParamsList, numDevices, flags);

  HIP_RETURN(ihipLaunchCooperativeKernelMultiDevice(
      launchParamsList, numDevices, flags,
      (amd::NDRangeKernelCommand::CooperativeGroups |
       amd::NDRangeKernelCommand::CooperativeMultiDeviceGroups)));
}

hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices,
                                              unsigned int flags) {
  HIP_INIT_API(hipExtLaunchMultiKernelMultiDevice, launchParamsList, numDevices, flags);

  HIP_RETURN(ihipLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags, 0));
}

hipError_t hipModuleGetTexRef(textureReference** texRef, hipModule_t hmod, const char* name) {
  HIP_INIT_API(hipModuleGetTexRef, texRef, hmod, name);

  /* input args check */
  if ((texRef == nullptr) || (name == nullptr) || (strlen(name) == 0)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (hmod == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  amd::Device* device = hip::getCurrentDevice()->devices()[0];
  const device::Info& info = device->info();
  if (!info.imageSupport_) {
    LogPrintfError("Texture not supported on the device %s", info.name_);
    HIP_RETURN(hipErrorNotSupported);
  }

  /* Get address and size for the global symbol */
  if (hipSuccess != PlatformState::Instance().GetDynTexRef(name, hmod, texRef)) {
    LogPrintfError("Cannot get texRef for name: %s at module:0x%x", name, hmod);
    HIP_RETURN(hipErrorNotFound);
  }

  // Texture references created by HIP driver API
  // have the default read mode set to normalized float.
  // have format set to format float
  // set num of channels to 1
  (*texRef)->readMode = hipReadModeNormalizedFloat;
  (*texRef)->format = HIP_AD_FORMAT_FLOAT;
  (*texRef)->numChannels = 1;

  hipError_t err = PlatformState::Instance().RegisterTexRef(*texRef, hmod, std::string(name));

  HIP_RETURN(err);
}


hipError_t hipLinkAddData(hipLinkState_t hip_link_state, hipJitInputType input_type, void* image,
                          size_t image_size, const char* name, unsigned int num_options,
                          hipJitOption* options_ptr, void** option_values) {
  HIP_INIT_API(hipLinkAddData, hip_link_state, image, image_size, name, num_options, options_ptr,
               option_values);

  if (image == nullptr || image_size <= 0) {
    HIP_RETURN(hipErrorInvalidImage);
  }

  if (input_type == hipJitInputCubin || input_type == hipJitInputPtx ||
      input_type == hipJitInputFatBinary || input_type == hipJitInputObject ||
      input_type == hipJitInputLibrary || input_type == hipJitInputNvvm ||
      input_type == hipJitInputLLVMBitcode || input_type == hipJitInputLLVMBundledBitcode ||
      input_type == hipJitInputLLVMArchivesOfBundledBitcode) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  std::string input_name;
  if (name) {
    input_name = name;
  }

  LinkProgram* hip_link_prog_ptr = reinterpret_cast<LinkProgram*>(hip_link_state);

  if (!LinkProgram::isLinkerValid(hip_link_prog_ptr)) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  if (!hip_link_prog_ptr->AddLinkerData(image, image_size, input_name, input_type)) {
    HIP_RETURN(hipErrorInvalidConfiguration);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipLinkAddFile(hipLinkState_t hip_link_state, hipJitInputType input_type,
                          const char* file_path, unsigned int num_options,
                          hipJitOption* options_ptr, void** option_values) {
  HIP_INIT_API(hipLinkAddFile, hip_link_state, input_type, file_path, num_options, options_ptr,
               option_values);

  if (hip_link_state == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  if (input_type == hipJitInputCubin || input_type == hipJitInputPtx ||
      input_type == hipJitInputFatBinary || input_type == hipJitInputObject ||
      input_type == hipJitInputLibrary || input_type == hipJitInputNvvm ||
      input_type == hipJitInputLLVMBitcode || input_type == hipJitInputLLVMBundledBitcode ||
      input_type == hipJitInputLLVMArchivesOfBundledBitcode) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  LinkProgram* hip_link_prog_ptr = reinterpret_cast<LinkProgram*>(hip_link_state);

  if (!LinkProgram::isLinkerValid(hip_link_prog_ptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (!hip_link_prog_ptr->AddLinkerFile(std::string(file_path), input_type)) {
    HIP_RETURN(hipErrorInvalidConfiguration);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipLinkCreate(unsigned int num_options, hipJitOption* options_ptr,
                         void** options_vals_pptr, hipLinkState_t* hip_link_state_ptr) {
  HIP_INIT_API(hipLinkCreate, num_options, options_ptr, options_vals_pptr, hip_link_state_ptr);

  if (hip_link_state_ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (num_options != 0) {
    if (options_ptr == nullptr || options_vals_pptr == nullptr) {
      HIP_RETURN(hipErrorInvalidValue);
    }

    for (int i = 0; i < num_options; ++i) {
      switch (options_ptr[i]) {
        // CUDA only options
        case hipJitOptionMaxRegisters:
        case hipJitOptionThreadsPerBlock:
        case hipJitOptionWallTime:
        case hipJitOptionInfoLogBuffer:
        case hipJitOptionInfoLogBufferSizeBytes:
        case hipJitOptionErrorLogBuffer:
        case hipJitOptionErrorLogBufferSizeBytes:
        case hipJitOptionOptimizationLevel:
        case hipJitOptionTargetFromContext:
        case hipJitOptionTarget:
        case hipJitOptionFallbackStrategy:
        case hipJitOptionGenerateDebugInfo:
        case hipJitOptionLogVerbose:
        case hipJitOptionGenerateLineInfo:
        case hipJitOptionCacheMode:
        case hipJitOptionSm3xOpt:
        case hipJitOptionFastCompile:
        case hipJitOptionGlobalSymbolNames:
        case hipJitOptionGlobalSymbolAddresses:
        case hipJitOptionGlobalSymbolCount:
        case hipJitOptionLto:
        case hipJitOptionFtz:
        case hipJitOptionPrecDiv:
        case hipJitOptionPrecSqrt:
        case hipJitOptionFma:
        case hipJitOptionPositionIndependentCode:
        case hipJitOptionMinCTAPerSM:
        case hipJitOptionMaxThreadsPerBlock:
        case hipJitOptionOverrideDirectiveValues:
        case hipJitOptionNumOptions:
          HIP_RETURN(hipErrorInvalidValue);
        default:
          // everything else is fine
          break;
      }
    }
  }

  std::string name("LinkerProgram");
  LinkProgram* hip_link_prog_ptr = new LinkProgram(name);
  if (!hip_link_prog_ptr->AddLinkerOptions(num_options, options_ptr, options_vals_pptr)) {
    HIP_RETURN(hipErrorInvalidConfiguration);
  }

  *hip_link_state_ptr = reinterpret_cast<hipLinkState_t>(hip_link_prog_ptr);

  HIP_RETURN(hipSuccess);
}

hipError_t hipLinkComplete(hipLinkState_t hip_link_state, void** bin_out, size_t* size_out) {
  HIP_INIT_API(hipLinkComplete, hip_link_state, bin_out, size_out);

  if (bin_out == nullptr || size_out == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  LinkProgram* hip_link_prog_ptr = reinterpret_cast<LinkProgram*>(hip_link_state);

  if (!LinkProgram::isLinkerValid(hip_link_prog_ptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (!hip_link_prog_ptr->LinkComplete(bin_out, size_out)) {
    HIP_RETURN(hipErrorInvalidConfiguration);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipLinkDestroy(hipLinkState_t hip_link_state) {
  HIP_INIT_API(hipLinkDestroy, hip_link_state);

  LinkProgram* hip_link_prog_ptr = reinterpret_cast<LinkProgram*>(hip_link_state);

  if (!LinkProgram::isLinkerValid(hip_link_prog_ptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  delete hip_link_prog_ptr;
  HIP_RETURN(hipSuccess);
}

static hipError_t validateDynDataPrefetchConfig(const hipExtDynDataPrefetchConfig& config,
                                                 uint32_t maxRegions,
                                                 size_t cacheLineSize) {
  constexpr size_t kPrefetchUnit = 256;
  constexpr size_t kMaxBurstSizeUnits = 1u << 11;
  constexpr size_t kMaxNumBursts = 1u << 16;
  constexpr size_t kMaxStrideUnits = (1u << 24) - 1u;

  if (maxRegions == 0) return hipErrorNotSupported;
  if (config.numRegions == 0 || config.numRegions > maxRegions) return hipErrorInvalidValue;
  if (config.temporal != hipExtDynDataPrefetchTemporalRegular &&
      config.temporal != hipExtDynDataPrefetchTemporalHigh) {
    return hipErrorInvalidValue;
  }
  for (unsigned int i = 0; i < config.numRegions; ++i) {
    const hipExtDynDataPrefetchRegion& r = config.regions[i];
    if (r.address == nullptr ||
        reinterpret_cast<uintptr_t>(r.address) % cacheLineSize != 0 ||
        r.width == 0 || r.width % cacheLineSize != 0 || r.width % kPrefetchUnit != 0 ||
        r.stride == 0 || r.stride % cacheLineSize != 0 || r.stride % kPrefetchUnit != 0 ||
        r.stride < r.width ||
        r.height == 0) {
      return hipErrorInvalidValue;
    }
    if ((r.width / kPrefetchUnit) > kMaxBurstSizeUnits ||
        r.height > kMaxNumBursts ||
        (r.stride / kPrefetchUnit) > kMaxStrideUnits) {
      return hipErrorInvalidValue;
    }
  }
  return hipSuccess;
}

static hipError_t convertDynDataPrefetchConfig(const hipExtDynDataPrefetchConfig& hipCfg,
                                               amd::DynDataPrefetchConfig& clrCfg,
                                               const device::Info& devInfo) {
  hipError_t status = validateDynDataPrefetchConfig(
      hipCfg, devInfo.maxDynDataPrefetchRegions_,
      static_cast<size_t>(devInfo.globalMemCacheLineSize_));
  if (status != hipSuccess) return status;
  clrCfg.numRegions = hipCfg.numRegions;
  clrCfg.hints = (hipCfg.temporal == hipExtDynDataPrefetchTemporalHigh) ? 2u : 0u;
  for (uint32_t i = 0; i < hipCfg.numRegions; ++i) {
    clrCfg.regions[i].baseAddress = hipCfg.regions[i].address;
    clrCfg.regions[i].burstSize   = hipCfg.regions[i].width;
    clrCfg.regions[i].numBursts   = static_cast<uint32_t>(hipCfg.regions[i].height);
    clrCfg.regions[i].stride      = static_cast<uint32_t>(hipCfg.regions[i].stride);
  }
  return hipSuccess;
}

hipError_t hipLaunchKernelExC(const hipLaunchConfig_t* config, const void* fPtr, void** args) {
  HIP_INIT_API(hipLaunchKernelExC, config, fPtr, args);
  if (fPtr == nullptr) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  if (config == nullptr) {
    HIP_RETURN(hipErrorInvalidConfiguration);
  }

  CHECK_STREAM_DETACHED_API(config->stream);

  if (config->numAttrs == 0) {
    HIP_RETURN_DURATION(hipLaunchKernel_common(fPtr, config->gridDim, config->blockDim, args,
                                               config->dynamicSmemBytes, config->stream));
  }

  const auto& devInfo = hip::getCurrentDevice()->devices()[0]->info();
  dim3 clusterDims = {1, 1, 1};
  amd::DynDataPrefetchConfig dynDataPrefetchConfig;
  for (size_t attr_idx = 0; attr_idx < config->numAttrs; ++attr_idx) {
    hipLaunchAttribute& attr = config->attrs[attr_idx];
    switch (attr.id) {
      case hipLaunchAttributeCooperative:
        if (attr.val.cooperative != 0) {
          HIP_RETURN_DURATION(
              hipLaunchCooperativeKernel_common(fPtr, config->gridDim, config->blockDim, args,
                                                config->dynamicSmemBytes, config->stream));
        }
        break;
      case hipLaunchAttributeClusterDimension:
        clusterDims.x = attr.val.clusterDim.x;
        clusterDims.y = attr.val.clusterDim.y;
        clusterDims.z = attr.val.clusterDim.z;
        if (clusterDims.x == 0 || clusterDims.y == 0 || clusterDims.z == 0) {
          HIP_RETURN(hipErrorInvalidConfiguration);
        }
        break;
      case hipLaunchAttributeExtDynDataPrefetch: {
        if (attr.val.dynDataPrefetch == nullptr) {
          HIP_RETURN(hipErrorInvalidValue);
        }
        hipError_t status = convertDynDataPrefetchConfig(*attr.val.dynDataPrefetch,
                                                         dynDataPrefetchConfig,
                                                         devInfo);
        if (status != hipSuccess) {
          HIP_RETURN(status);
        }
        break;
      }
      default:
        LogPrintfError("Attribute %u not supported", attr.id);
        HIP_RETURN(hipErrorInvalidValue);
    }
  }

  const auto& deviceInfo = hip::getCurrentDevice()->devices()[0]->info();
  __hip_uint32_t requestedClusterSize;

  if (multiplyOverflow(clusterDims.x,
                       clusterDims.y,
                       requestedClusterSize)) {
    HIP_RETURN(hipErrorInvalidClusterSize);
  }

  if (multiplyOverflow(requestedClusterSize,
                       clusterDims.z,
                       requestedClusterSize)) {
    HIP_RETURN(hipErrorInvalidClusterSize);
  }

  // Check cluster size against device maximum
  if (deviceInfo.clusterMaxSize_ > 0 &&
      requestedClusterSize > deviceInfo.clusterMaxSize_) {
    HIP_RETURN(hipErrorInvalidClusterSize);
  }

  HIP_RETURN_DURATION(hipLaunchKernel_common(fPtr, config->gridDim, config->blockDim, args,
    config->dynamicSmemBytes, config->stream, clusterDims,
    dynDataPrefetchConfig.isEnabled() ? &dynDataPrefetchConfig : nullptr));
}

hipError_t hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG* config, hipFunction_t f,
                                void** kernelParams, void** extra) {
  HIP_INIT_API(hipDrvLaunchKernelEx, config, f, kernelParams, extra);
  if (f == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  if (config == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipStream_t hStream = config->hStream;
  if (!hip::isValid(hStream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }
  CHECK_STREAM_DETACHED_API(hStream);

  STREAM_CAPTURE(hipDrvLaunchKernelEx, hStream, config, f, kernelParams, extra);

  int drvDeviceId = hip::Stream::DeviceId(hStream);
  const amd::Device* drvDevice = g_devices[drvDeviceId]->devices()[0];
  amd::HIPLaunchParams launch_params(config->gridDimX, config->gridDimY, config->gridDimZ,
                                     config->blockDimX, config->blockDimY, config->blockDimZ,
                                     config->sharedMemBytes, *drvDevice, 0, 0, 0, 1, 1, 1);

  if (!launch_params.IsValidConfig()) {
    HIP_RETURN(hipErrorInvalidConfiguration);
  }

  if (config->numAttrs == 0) {
    HIP_RETURN(ihipModuleLaunchKernel(f, launch_params, hStream, kernelParams, extra,
                                      nullptr, nullptr, 0));
  }

  const auto& devInfo = drvDevice->info();
  dim3 clusterDim = {1, 1, 1};
  amd::DynDataPrefetchConfig dynDataPrefetchConfig;
  for (size_t attr_idx = 0; attr_idx < config->numAttrs; ++attr_idx) {
    hipLaunchAttribute& attr = config->attrs[attr_idx];
    switch (attr.id) {
      case hipLaunchAttributeCooperative: {
        if (attr.value.cooperative != 0) {
          HIP_RETURN(ihipModuleLaunchKernel(f, launch_params, hStream, kernelParams,
                                            extra, nullptr, nullptr, 0,
                                            amd::NDRangeKernelCommand::CooperativeGroups));
        }
        break;
      }
      case hipLaunchAttributeClusterDimension: {
        clusterDim.x = attr.value.clusterDim.x;
        clusterDim.y = attr.value.clusterDim.y;
        clusterDim.z = attr.value.clusterDim.z;
        break;
      }
      case hipLaunchAttributeExtDynDataPrefetch: {
        if (attr.value.dynDataPrefetch == nullptr) {
          HIP_RETURN(hipErrorInvalidValue);
        }
        hipError_t status = convertDynDataPrefetchConfig(*attr.value.dynDataPrefetch,
                                                         dynDataPrefetchConfig,
                                                         devInfo);
        if (status != hipSuccess) {
          HIP_RETURN(status);
        }
        break;
      }
      default:
        LogPrintfError("Attribute %u not supported", attr.id);
        HIP_RETURN(hipErrorInvalidValue);
    }
  }

  if (clusterDim.x == 0 || clusterDim.y == 0 || clusterDim.z == 0) {
    HIP_RETURN(hipErrorInvalidConfiguration);
  }

  amd::HIPLaunchParams launch_params_cluster(config->gridDimX, config->gridDimY, config->gridDimZ,
                                          config->blockDimX, config->blockDimY, config->blockDimZ,
                                          config->sharedMemBytes, *drvDevice, 0, 0, 0, clusterDim.x,
                                          clusterDim.y, clusterDim.z);

  HIP_RETURN(ihipModuleLaunchKernel(f, launch_params_cluster, hStream, kernelParams, extra, nullptr,
                                    nullptr, 0, 0, 0, 0, 0, 0, 0,
                                    dynDataPrefetchConfig.isEnabled() ? &dynDataPrefetchConfig : nullptr));
}
}  // namespace hip
