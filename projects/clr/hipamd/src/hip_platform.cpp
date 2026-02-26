/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

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

#include <hip/hip_runtime.h>
#include <hip/texture_types.h>
#include "hip_platform.hpp"
#include "hip_internal.hpp"
#include "platform/program.hpp"
#include "platform/runtime.hpp"
#include "utils/flags.hpp"

#include <unordered_map>
#include <mutex>

namespace hip_impl {
// ================================================================================================
hipError_t ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    int* maxBlocksPerCU, int* numBlocksPerGrid, int* bestBlockSize, const amd::Device& device,
    hipFunction_t func, int inputBlockSize, size_t dynamicSMemSize, bool bCalcPotentialBlkSz) {
  auto* function = hip::DeviceFunc::asFunction(func);
  const auto* kernel = function->kernel();

  const auto* wrkGrpInfo = kernel->getDeviceKernel(device)->workGroupInfo();
  const int maxWorkGroupSize = static_cast<int>(device.info().maxWorkGroupSize_);
  
  if (!bCalcPotentialBlkSz) {
    if (inputBlockSize <= 0) {
      return hipErrorInvalidValue;
    }
    *bestBlockSize = 0;
    if (inputBlockSize > maxWorkGroupSize) {
      *maxBlocksPerCU = 0;
      *numBlocksPerGrid = 0;
      return hipSuccess;
    }
  } else if (inputBlockSize > maxWorkGroupSize || inputBlockSize <= 0) {
    inputBlockSize = maxWorkGroupSize;
  }
  
  // Find wave occupancy per CU => simd_per_cu * GPR usage
  // Limited by SPI 32 per CU, hence 8 per SIMD
  const size_t MaxWavesPerSimd = (device.isa().versionMajor() <= 9) ? 8 : 16;
  const size_t wavefrontSize = wrkGrpInfo->wavefrontSize_;
  const bool adjust_for_wave64 = device.isa().versionMajor() >= 10 && wavefrontSize == 64;
  const uint32_t VgprGranularity = adjust_for_wave64
      ? device.info().vgprAllocGranularity_ >> 1
      : device.info().vgprAllocGranularity_;
  const size_t maxVGPRs = adjust_for_wave64
      ? device.info().vgprsPerSimd_ >> 1
      : device.info().vgprsPerSimd_;
  const size_t VgprWaves = wrkGrpInfo->usedVGPRs_ > 0
      ? maxVGPRs / amd::alignUp(wrkGrpInfo->usedVGPRs_, VgprGranularity)
      : MaxWavesPerSimd;

  if (VgprWaves == 0) {
    // This should not happen ideally, but in case the value is
    // incorrect, it can lead to a crash. By returning error, API can exit gracefully.
    return hipErrorUnknown;
  }

  const size_t GprWaves = wrkGrpInfo->usedSGPRs_ > 0
      ? std::min(VgprWaves, device.info().sgprsPerSimd_ /
                            amd::alignUp(wrkGrpInfo->usedSGPRs_, 16))
      : VgprWaves;

  // The table contains SIMD per CU, not per WGP, so when WGP mode is set
  // on kernel metadata, multiply the number of SIMDs by 2, to account for
  // 2CUs in 1 WGP.
  const uint32_t simdPerCU = wrkGrpInfo->isWGPMode_
      ? device.isa().simdPerCU() * 2
      : device.isa().simdPerCU();

  const size_t alu_occupancy = simdPerCU * std::min(MaxWavesPerSimd, GprWaves);
  const int alu_limited_threads = static_cast<int>(alu_occupancy * wavefrontSize);

  const size_t total_used_lds = wrkGrpInfo->usedLDSSize_ + dynamicSMemSize;
  const int lds_occupancy_wgs = total_used_lds != 0
      ? static_cast<int>(device.info().localMemSize_ / total_used_lds)
      : INT_MAX;
  // Calculate how many blocks of inputBlockSize we can fit per CU
  // Need to align with hardware wavefront size. If they want 65 threads, but
  // waves are 64, then we need 128 threads per block.
  // So this calculates how many blocks we can fit.
  const int aligned_input_size = amd::alignUp(inputBlockSize, wrkGrpInfo->wavefrontSize_);
  *maxBlocksPerCU = alu_limited_threads / aligned_input_size;
  // Unless those blocks are further constrained by LDS size.
  *maxBlocksPerCU = std::min(*maxBlocksPerCU, lds_occupancy_wgs);

  // Return optimal block size: min of ALU limit and requested size
  *bestBlockSize = std::min(alu_limited_threads, aligned_input_size);
  // Calculate blocks per CU for full occupancy
  const int bestBlocksPerCU = alu_limited_threads / (*bestBlockSize);
  const uint32_t maxCUs = (wrkGrpInfo->isWGPMode_ != device.settings().enableWgpMode_)
      ? (wrkGrpInfo->isWGPMode_
          ? device.info().maxComputeUnits_ / 2
          : device.info().maxComputeUnits_ * 2)
      : device.info().maxComputeUnits_;
  *numBlocksPerGrid = maxCUs * std::min(bestBlocksPerCU, lds_occupancy_wgs);

  return hipSuccess;
}
}  // namespace hip_impl

namespace hip {
constexpr unsigned __hipFatMAGIC2 = 0x48495046;  // "HIPF"

// Static member definition
PlatformState* PlatformState::platform_ = nullptr;

struct __CudaFatBinaryWrapper {
  unsigned int magic;
  unsigned int version;
  void* binary;
  void* dummy1;
};

// Forward declarations
hipError_t ihipMallocManaged(void** ptr, size_t size, size_t align = 0, bool use_host_ptr = 0);
hipError_t ihipModuleLaunchKernel(hipFunction_t f, amd::LaunchParams& launch_params,
                                  hipStream_t hStream, void** kernelParams, void** extra,
                                  hipEvent_t startEvent, hipEvent_t stopEvent,
                                  uint32_t flags = 0, uint32_t params = 0,
                                  uint32_t gridId = 0, uint32_t numGrids = 0,
                                  uint64_t prevGridSum = 0, uint64_t allGridSum = 0,
                                  uint32_t firstDevice = 0);

// ================================================================================================
static bool isCompatibleCodeObject(const std::string& codeobj_target_id, const char* device_name) {
  // Workaround for device name mismatch.
  // Device name may contain feature strings delimited by '+', e.g.
  // gfx900+xnack. Currently HIP-Clang does not include feature strings
  // in code object target id in fat binary. Therefore drop the feature
  // strings from device name before comparing it with code object target id.
  const char* feature_loc = std::strchr(device_name, '+');
  if (feature_loc == nullptr) {
    return codeobj_target_id == device_name;
  }
  return codeobj_target_id.compare(0, std::string::npos, device_name,
                                    feature_loc - device_name) == 0;
}

// ================================================================================================
void** __hipRegisterFatBinary(const void* data) {
  const __CudaFatBinaryWrapper* fbwrapper = reinterpret_cast<const __CudaFatBinaryWrapper*>(data);

  // Validate version early
  if (fbwrapper->version != 1) {
    LogPrintfError("Cannot Register fat binary. Invalid version: %u", fbwrapper->version);
    return nullptr;
  }

  bool success = false;
  hip::FatBinaryInfo** fat_binary_info = nullptr;

  // Check for HIPK magic (kpack'd binary with external device code)
  if (fbwrapper->magic == symbols::kHipkMagic) {
    // For HIPK binaries, fbwrapper->binary points to msgpack metadata
    // Route through AddKpackBinary which will error if ROCM_KPACK_ENABLED=OFF
    fat_binary_info = PlatformState::Instance().StatCO().AddKpackBinary(fbwrapper->binary, data, success);
  } else if (fbwrapper->magic == __hipFatMAGIC2) {
    // Normal HIPF path
    fat_binary_info = PlatformState::Instance().StatCO().AddFatBinary(fbwrapper->binary, success);
  } else {
    LogPrintfError("Cannot Register fat binary. Invalid FatMagic: 0x%x", fbwrapper->magic);
    return nullptr;
  }

  return success ? reinterpret_cast<void**>(fat_binary_info) : nullptr;
}

// ================================================================================================
void __hipRegisterFunction(void** modules, const void* hostFunction, char* deviceFunction,
                           const char* deviceName, unsigned int threadLimit, uint3* tid, uint3* bid,
                           dim3* blockDim, dim3* gridDim, int* wSize) {
  auto* fat_binary_modules = reinterpret_cast<hip::FatBinaryInfo**>(modules);
  
  static const bool enable_deferred_loading = []() {
    const char* var = getenv("HIP_ENABLE_DEFERRED_LOADING");
    return var ? atoi(var) != 0 : true;
  }();

  // Compiler might share same hostFunction, so avoid creating duplicate hip::Function.
  // hip::Function is stored in map with hostFunction as key to prevent leaks.
  auto& platform = PlatformState::Instance();
  if (platform.StatCO().GetFuncName(hostFunction) == nullptr) {
    hip::Function* func = new hip::Function(std::string(deviceName), fat_binary_modules);
    hipError_t hip_error = platform.StatCO().RegisterFunction(hostFunction, func);
    guarantee(hip_error == hipSuccess, "Cannot register Static function, error: %d", hip_error);
  }

  if (!enable_deferred_loading) {
    HIP_INIT_VOID();
    
    for (size_t dev_idx = 0; dev_idx < g_devices.size(); ++dev_idx) {
      hipFunction_t hfunc = nullptr;
      hipError_t hip_error = platform.StatCO().GetFunc(&hfunc, hostFunction, dev_idx);
      guarantee(hip_error == hipSuccess, "Cannot retrieve Static function, error: %d", hip_error);
    }
  }
}

// ================================================================================================
// Registers a device-side global variable with host-side shadow copy for
// tracking state between kernel executions.
void __hipRegisterVar(void** modules,       // The device modules containing code object
                      void* var,            // The shadow variable in host code
                      char* hostVar,        // Variable name in host code
                      char* deviceVar,      // Variable name in device code
                      int ext,              // Whether this variable is external
                      size_t size,          // Size of the variable
                      int constant,         // Whether this variable is constant
                      int global) {         // Unknown, always 0
  auto* fat_binary_modules = reinterpret_cast<hip::FatBinaryInfo**>(modules);
  hip::Var* var_ptr = new hip::Var(std::string(hostVar), hip::Var::DeviceVarKind::DVK_Variable,
                                   size, 0, 0, fat_binary_modules);
  hipError_t err = PlatformState::Instance().StatCO().RegisterGlobalVar(var, var_ptr);
  guarantee((err == hipSuccess), "Cannot register Static Global Var, error:%d", err);
}

// ================================================================================================
void __hipRegisterSurface(
    void** modules,       // The device modules containing code object
    void* var,            // The shadow variable in host code
    char* hostVar,        // Variable name in host code
    char* deviceVar,      // Variable name in device code
    int type, int ext) {
  auto* fat_binary_modules = reinterpret_cast<hip::FatBinaryInfo**>(modules);
  hip::Var* var_ptr = new hip::Var(std::string(hostVar), hip::Var::DeviceVarKind::DVK_Surface,
                                   sizeof(surfaceReference), 0, 0, fat_binary_modules);
  hipError_t err = PlatformState::Instance().StatCO().RegisterGlobalVar(var, var_ptr);
  guarantee((err == hipSuccess), "Cannot register Static Glbal Var, err:%d", err);
}

// ================================================================================================
void __hipRegisterManagedVar(
    void* hipModule,  // Pointer to hip module returned from __hipRegisterFatbinary
    void** pointer,   // Pointer to a chunk of managed memory with size \p size and alignment \p
                      // align HIP runtime allocates such managed memory and assign it to \p pointer
    void* init_value,  // Initial value to be copied into \p pointer
    const char* name,  // Name of the variable in code object
    size_t size, unsigned align) {
  static const bool enable_deferred_loading = []() {
#ifdef _WIN32  // Don't defer loading for windows
    return false;
#else
    const char* var = getenv("HIP_ENABLE_DEFERRED_LOADING");
    return var ? atoi(var) != 0 : true;
#endif
  }();

  hip::Var* var_ptr = new hip::Var(std::string(name), hip::Var::DeviceVarKind::DVK_Managed, pointer,
                                   size, align, reinterpret_cast<hip::FatBinaryInfo**>(hipModule));
  hipError_t status = PlatformState::Instance().StatCO().RegisterManagedVar(var_ptr);
  guarantee(status == hipSuccess, "Cannot register Static Managed Var, error: %d", status);

  if (enable_deferred_loading) {
    // Allocate temporary var on host and initialize
    *pointer = amd::Os::reserveMemory(0, size, align, amd::Os::MEM_PROT_RW);
    ::memcpy(*pointer, init_value, size);
  } else {
    HIP_INIT_VOID();
    status = ihipMallocManaged(pointer, size, align, 0);
    var_ptr->setAllocFlag(true);
    if (status == hipSuccess) {
      hip::Stream* stream = hip::getNullStream();
      if (stream != nullptr) {
        status = ihipMemcpy(*pointer, init_value, size, hipMemcpyHostToDevice, *stream);
        guarantee(status == hipSuccess, "Error during memcpy to managed memory, error: %d", status);
      } else {
        ClPrint(amd::LOG_ERROR, amd::LOG_API, "Host Queue is NULL");
      }
    } else {
      guarantee(false, "Error during allocation of managed memory!, error: %d", status);
    }
  }
}

// ================================================================================================
void __hipRegisterTexture(
    void** modules,       // The device modules containing code object
    void* var,            // The shadow variable in host code
    char* hostVar,        // Variable name in host code
    char* deviceVar,      // Variable name in device code
    int type, int norm, int ext) {
  auto* fat_binary_modules = reinterpret_cast<hip::FatBinaryInfo**>(modules);
  hip::Var* var_ptr = new hip::Var(std::string(hostVar), hip::Var::DeviceVarKind::DVK_Texture,
                                   sizeof(textureReference), 0, 0, fat_binary_modules);
  hipError_t err = PlatformState::Instance().StatCO().RegisterGlobalVar(var, var_ptr);
  guarantee((err == hipSuccess), "Cannot register Static Global Var, status: %d", err);
}

// ================================================================================================
void __hipUnregisterFatBinary(void** modules) {
  auto* fat_binary_modules = reinterpret_cast<hip::FatBinaryInfo**>(modules);
  static std::once_flag unregister_device_sync;
  // If SKIP ABORT is set and GPU is in error, dont need to sync streams.
  if (!HIP_SKIP_ABORT_ON_GPU_ERROR || !amd::Device::IsGPUInError()) {
    std::call_once(unregister_device_sync, []() {
      for (const auto& hipDevice : g_devices) {
        // By synchronizing devices ensure that all HSA signal handlers
        // complete before RemoveFatBinary
        hipDevice->SyncAllStreams(true);
      }
    });
  }
  hipError_t err = PlatformState::Instance().StatCO().RemoveFatBinary(fat_binary_modules);
  guarantee((err == hipSuccess), "Cannot Unregister Fat Binary, error:%d", err);
}

// ================================================================================================
hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream) {
  HIP_INIT_API(hipConfigureCall, gridDim, blockDim, sharedMem, stream);

  PlatformState::Instance().ConfigureCall(gridDim, blockDim, sharedMem, stream);

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                      hipStream_t stream) {
  HIP_INIT_API(__hipPushCallConfiguration, gridDim, blockDim, sharedMem, stream);

  PlatformState::Instance().ConfigureCall(gridDim, blockDim, sharedMem, stream);

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem,
                                     hipStream_t* stream) {
  HIP_INIT_API(__hipPopCallConfiguration, gridDim, blockDim, sharedMem, stream);

  ihipExec_t exec;
  PlatformState::Instance().PopExec(exec);
  *gridDim = exec.gridDim_;
  *blockDim = exec.blockDim_;
  *sharedMem = exec.sharedMem_;
  *stream = exec.hStream_;

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset) {
  HIP_INIT_API(hipSetupArgument, arg, size, offset);

  PlatformState::Instance().SetupArgument(arg, size, offset);

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipLaunchByPtr(const void* hostFunction) {
  HIP_INIT_API(hipLaunchByPtr, hostFunction);

  ihipExec_t exec;
  PlatformState::Instance().PopExec(exec);

  const auto* stream = reinterpret_cast<hip::Stream*>(exec.hStream_);
  const auto deviceId = stream ? stream->DeviceId() : ihipGetDevice();
  if (deviceId == -1) {
    LogPrintfError("Wrong DeviceId: %d", deviceId);
    HIP_RETURN(hipErrorNoDevice);
  }

  hipFunction_t func = nullptr;
  const hipError_t hip_error =
      PlatformState::Instance().StatCO().GetFunc(&func, hostFunction, deviceId);
  if (hip_error != hipSuccess || !func) {
    LogPrintfError("Could not retrieve hostFunction: 0x%x", hostFunction);
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  size_t size = exec.arguments_.size();
  void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &exec.arguments_[0],
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};

  STREAM_CAPTURE(hipLaunchByPtr, exec.hStream_, func, exec.blockDim_, exec.gridDim_,
                 exec.sharedMem_, extra);

  const amd::Device* device = g_devices[deviceId]->devices()[0];
  amd::HIPLaunchParams launch_params(exec.gridDim_.x, exec.gridDim_.y, exec.gridDim_.z,
                                           exec.blockDim_.x, exec.blockDim_.y, exec.blockDim_.z,
                                           exec.sharedMem_);
  if (!launch_params.IsValidConfig() ||
      launch_params.local_.product() > device->info().maxWorkGroupSize_) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipModuleLaunchKernel(
      func, launch_params, exec.hStream_, nullptr, extra, nullptr, nullptr));
}

// ================================================================================================
hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol) {
  HIP_INIT_API(hipGetSymbolAddress, devPtr, symbol);

  if (devPtr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  size_t sym_size = 0;
  HIP_RETURN_ONFAIL(PlatformState::Instance().StatCO().GetGlobalVar(symbol, ihipGetDevice(), devPtr,
                                                                &sym_size));

  HIP_RETURN(hipSuccess, *devPtr);
}

// ================================================================================================
hipError_t hipGetSymbolSize(size_t* sizePtr, const void* symbol) {
  HIP_INIT_API(hipGetSymbolSize, sizePtr, symbol);

  if (sizePtr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipDeviceptr_t device_ptr = nullptr;
  HIP_RETURN_ONFAIL(
      PlatformState::Instance().StatCO().GetGlobalVar(symbol, ihipGetDevice(), &device_ptr, sizePtr));

  HIP_RETURN(hipSuccess, *sizePtr);
}

// ================================================================================================
hipError_t ihipCreateGlobalVarObj(const char* name, hipModule_t hmod, amd::Memory** amd_mem_obj,
                                  hipDeviceptr_t* dptr, size_t* bytes) {
  // Get Device Program pointer
  auto* program = as_amd(reinterpret_cast<cl_program>(hmod));
  auto* dev_program = program->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  if (!dev_program) {
    LogPrintfError("Cannot get Device Function for module: 0x%x", hmod);
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }
  
  // Find the global Symbols
  if (!dev_program->createGlobalVarObj(amd_mem_obj, dptr, bytes, name)) {
    LogPrintfError("Cannot create Global Var obj for symbol: %s", name);
    HIP_RETURN(hipErrorInvalidSymbol);
  }

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* f,
                                                    int numBlocks, int blockSize){
  HIP_INIT_API(hipOccupancyAvailableDynamicSMemPerBlock, dynamicSmemSize, f, numBlocks, blockSize);
  if (dynamicSmemSize == nullptr || numBlocks <= 0 || blockSize <= 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipFunction_t func = nullptr;
  const int dev_id = ihipGetDevice();
  const hipError_t hip_error = PlatformState::Instance().StatCO().GetFunc(&func, f, dev_id);

  if (hip_error != hipSuccess || !func) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  auto* function = hip::DeviceFunc::asFunction(func);
  if (function == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  hipDeviceProp_t prop = {0};
  HIP_RETURN_ONFAIL(ihipGetDeviceProperties(&prop, dev_id));

  if (blockSize > prop.maxThreadsPerMultiProcessor) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  const amd::Device& device = *hip::getCurrentDevice()->devices()[dev_id];
  const amd::Kernel& kernel = *function->kernel();
  const auto* wrkGrpInfo = kernel.getDeviceKernel(device)->workGroupInfo();

  const int staticSharedMemoryUsage = wrkGrpInfo->usedLDSSize_;
  const int maxDynamicSharedSizeBytes = wrkGrpInfo->maxDynamicSharedSizeBytes_;
  const int maxNumBlocks = prop.maxThreadsPerMultiProcessor / blockSize;
  const int maxSharedMemoryPerMultiProcessor = prop.maxSharedMemoryPerMultiProcessor - 
      staticSharedMemoryUsage * std::min(numBlocks, maxNumBlocks);
  const int maxDynamicSmemSize = std::min(maxSharedMemoryPerMultiProcessor / maxNumBlocks,
                                          maxDynamicSharedSizeBytes);
  const int alignmentSize = device.isa().ldsAlignment();

  const int dynamic_smem_size = std::max(maxDynamicSmemSize,
                                         std::min(maxSharedMemoryPerMultiProcessor / numBlocks,
                                                  maxDynamicSharedSizeBytes));
  *dynamicSmemSize = amd::alignDown(dynamic_smem_size, alignmentSize);

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize, const void* f,
                                             size_t dynSharedMemPerBlk, int blockSizeLimit) {
  HIP_INIT_API(hipOccupancyMaxPotentialBlockSize, f, dynSharedMemPerBlk, blockSizeLimit);
  if (!gridSize || !blockSize) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipFunction_t func = nullptr;
  const hipError_t hip_error =
      PlatformState::Instance().StatCO().GetFunc(&func, f, ihipGetDevice());
  if (hip_error != hipSuccess || !func) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];
  int max_blocks_per_grid = 0;
  int num_blocks = 0;
  int best_block_size = 0;
  const hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, &max_blocks_per_grid, &best_block_size, device, func, blockSizeLimit,
      dynSharedMemPerBlk, true);
  if (ret == hipSuccess) {
    *blockSize = best_block_size;
    *gridSize = max_blocks_per_grid;
  }
  HIP_RETURN(ret);
}

// ================================================================================================
hipError_t hipModuleOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize, hipFunction_t f,
                                                   size_t dynSharedMemPerBlk, int blockSizeLimit) {
    HIP_INIT_API(hipModuleOccupancyMaxPotentialBlockSize, f, dynSharedMemPerBlk, blockSizeLimit);
    if ((gridSize == nullptr) || (blockSize == nullptr) || (f == nullptr)) {
      HIP_RETURN(hipErrorInvalidValue);
    }
    const amd::Device& device = *hip::getCurrentDevice()->devices()[0];
    int max_blocks_per_grid = 0;
    int num_blocks = 0;
    int best_block_size = 0;
    const hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, &max_blocks_per_grid, &best_block_size, device, f, blockSizeLimit,
        dynSharedMemPerBlk, true);
    if (ret == hipSuccess) {
      *blockSize = best_block_size;
      *gridSize = max_blocks_per_grid;
    }
    HIP_RETURN(ret);
}

// ================================================================================================
hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize,
                                                            hipFunction_t f,
                                                            size_t dynSharedMemPerBlk,
                                                            int blockSizeLimit,
                                                            unsigned int flags) {
  HIP_INIT_API(hipModuleOccupancyMaxPotentialBlockSizeWithFlags, f, dynSharedMemPerBlk,
               blockSizeLimit, flags);
  if ((gridSize == nullptr) || (blockSize == nullptr) || (f == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (flags != hipOccupancyDefault && flags != hipOccupancyDisableCachingOverride) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];
  int max_blocks_per_grid = 0;
  int num_blocks = 0;
  int best_block_size = 0;
  const hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, &max_blocks_per_grid, &best_block_size, device, f, blockSizeLimit,
      dynSharedMemPerBlk, true);
  if (ret == hipSuccess) {
    *blockSize = best_block_size;
    *gridSize = max_blocks_per_grid;
  }
  HIP_RETURN(ret);
}

// ================================================================================================
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, hipFunction_t f,
                                                              int blockSize,
                                                              size_t dynSharedMemPerBlk) {
  HIP_INIT_API(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, f, blockSize,
               dynSharedMemPerBlk);
  if (numBlocks == nullptr || (f == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];
  int max_blocks_per_grid = 0;
  int best_block_size = 0;
  const hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, &max_blocks_per_grid, &best_block_size, device, f, blockSize, dynSharedMemPerBlk,
      false);
  HIP_RETURN(ret);
}

// ================================================================================================
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
  HIP_INIT_API(hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, f, blockSize,
               dynSharedMemPerBlk, flags);
  if (numBlocks == nullptr || (f == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (flags != hipOccupancyDefault && flags != hipOccupancyDisableCachingOverride) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];
  int max_blocks_per_grid = 0;
  int best_block_size = 0;
  const hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, &max_blocks_per_grid, &best_block_size, device, f, blockSize, dynSharedMemPerBlk,
      false);
  HIP_RETURN(ret);
}

// ================================================================================================
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* f,
                                                        int blockSize, size_t dynamicSMemSize) {
  HIP_INIT_API(hipOccupancyMaxActiveBlocksPerMultiprocessor, f, blockSize, dynamicSMemSize);
  if (numBlocks == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipFunction_t func = nullptr;
  hipError_t ret = PlatformState::Instance().StatCO().GetFunc(&func, f, ihipGetDevice());
  if (ret != hipSuccess || !func) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];
  int max_blocks_per_grid = 0;
  int best_block_size = 0;
  ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, &max_blocks_per_grid, &best_block_size, device, func, blockSize, dynamicSMemSize,
      false);
  HIP_RETURN(ret);
}

// ================================================================================================
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* f,
                                                                 int blockSize,
                                                                 size_t dynamicSMemSize,
                                                                 unsigned int flags) {
  HIP_INIT_API(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, f, blockSize, dynamicSMemSize,
               flags);
  if (numBlocks == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (flags != hipOccupancyDefault && flags != hipOccupancyDisableCachingOverride) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipFunction_t func = nullptr;
  hipError_t ret = PlatformState::Instance().StatCO().GetFunc(&func, f, ihipGetDevice());
  if (ret != hipSuccess || !func) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];
  int max_blocks_per_grid = 0;
  int best_block_size = 0;
  ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, &max_blocks_per_grid, &best_block_size, device, func, blockSize, dynamicSMemSize,
      false);
  HIP_RETURN(ret);
}

// ================================================================================================
hipError_t ihipLaunchKernel(const void* hostFunction, dim3 gridDim, dim3 blockDim, void** args,
                            size_t sharedMemBytes, hipStream_t stream, hipEvent_t startEvent,
                            hipEvent_t stopEvent, int flags) {
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  if (hostFunction == nullptr) {
    return hipErrorInvalidDeviceFunction;
  }

  const int deviceId = hip::Stream::DeviceId(stream);

  const auto [hip_error, func] = [&]() -> std::pair<hipError_t, hipFunction_t> {
    hipFunction_t f;
    const hipError_t err = PlatformState::Instance().StatCO().GetFunc(&f, hostFunction, deviceId);
    
    // Propagate specific invalid code object errors
    if (err == hipErrorInvalidKernelFile ||
        err == hipErrorInvalidDeviceFunction ||
        err == hipErrorInvalidImage) {
      return {err, nullptr};
    }
    
    // If successful lookup with valid function, use it
    if (err == hipSuccess && f) {
      return {hipSuccess, f};
    }
    
    // Fallback: assume it's a hip function type
    return {hipSuccess, reinterpret_cast<hipFunction_t>(const_cast<void*>(hostFunction))};
  }();

  if (hip_error != hipSuccess) {
    return hip_error;
  }

  constexpr auto gridDimYZmax = static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1;
  const auto* device = g_devices[deviceId]->devices()[0];
  if (device->isa().versionMajor() >= 12 &&
      (gridDim.y > gridDimYZmax || gridDim.z > gridDimYZmax)) {
    return hipErrorInvalidConfiguration;
  }

  amd::HIPLaunchParams launch_params(gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
                                     blockDim.z, sharedMemBytes);
  if (!launch_params.IsValidConfig()) {
    return hipErrorInvalidConfiguration;
  }

  return ihipModuleLaunchKernel(func, launch_params, stream, args, nullptr, startEvent, stopEvent,
                                flags);
}

// ================================================================================================
// conversion routines between float and half precision
static inline std::uint32_t f32_as_u32(float f) {
  union {
    float f;
    std::uint32_t u;
  } v;
  v.f = f;
  return v.u;
}

// ================================================================================================
static inline float u32_as_f32(std::uint32_t u) {
  union {
    float f;
    std::uint32_t u;
  } v;
  v.u = u;
  return v.f;
}

// ================================================================================================
static inline int clamp_int(int i, int l, int h) { return std::min(std::max(i, l), h); }

// ================================================================================================
// half float, the f16 is in the low 16 bits of the input argument
static inline float __convert_half_to_float(std::uint32_t a) noexcept {
  std::uint32_t u = ((a << 13) + 0x70000000U) & 0x8fffe000U;

  std::uint32_t v =
      f32_as_u32(u32_as_f32(u) * u32_as_f32(0x77800000U) /*0x1.0p+112f*/) + 0x38000000U;

  u = (a & 0x7fff) != 0 ? v : u;

  return u32_as_f32(u) * u32_as_f32(0x07800000U) /*0x1.0p-112f*/;
}

// ================================================================================================
// float half with nearest even rounding
// The lower 16 bits of the result is the bit pattern for the f16
static inline uint32_t __convert_float_to_half(float a) noexcept {
  const uint32_t u = f32_as_u32(a);
  const int e = static_cast<int>((u >> 23) & 0xff) - 127 + 15;
  const uint32_t m = ((u >> 11) & 0xffe) | ((u & 0xfff) != 0);
  const uint32_t i = 0x7c00 | (m != 0 ? 0x0200 : 0);
  const uint32_t n = (static_cast<uint32_t>(e) << 12) | m;
  const uint32_t s = (u >> 16) & 0x8000;
  const int b = clamp_int(1 - e, 0, 13);
  uint32_t d = (0x1000 | m) >> b;
  d |= (d << b) != (0x1000 | m);
  uint32_t v = e < 1 ? d : n;
  v = (v >> 2) + (((v & 0x7) == 3) | ((v & 0x7) > 5));
  v = e > 30 ? 0x7c00 : v;
  v = e == 143 ? i : v;
  return s | v;
}

// ================================================================================================
extern "C"
#if !defined(_MSC_VER)
    __attribute__((weak))
#endif
    float
    __gnu_h2f_ieee(unsigned short h) {
  return __convert_half_to_float((std::uint32_t)h);
}

// ================================================================================================
extern "C"
#if !defined(_MSC_VER)
    __attribute__((weak))
#endif
    unsigned short
    __gnu_f2h_ieee(float f) {
  return (unsigned short)__convert_float_to_half(f);
}

// ================================================================================================
void PlatformState::Init() {
  amd::ScopedLock lock(lock_);
  if (initialized_ || g_devices.empty()) {
    return;
  }
  initialized_ = true;
  statCO_.ResizeForDevices(g_devices.size());
  amd::RuntimeTearDown::RegisterTearDownCallback("PlatformState static fatbin cleanup", [this]() {
    statCO_.RemoveAllFatBinaries();
  });
}

// ================================================================================================
hipError_t PlatformState::LoadModule(hipModule_t* module, const char* fname, const void* image) {
  if (module == nullptr) {
    return hipErrorInvalidValue;
  }

  auto dynCo = std::make_unique<hip::DynCO>();
  const hipError_t hip_error = dynCo->loadCodeObject(fname, image);
  if (hip_error != hipSuccess) {
    return hip_error;
  }

  *module = dynCo->getModule();
  assert(*module != nullptr);

  amd::ScopedLock lock(lock_);
  const auto [it, inserted] = dynCO_map_.try_emplace(*module, dynCo.get());
  if (!inserted) {
    return hipErrorAlreadyMapped;
  }
  dynCo.release();

  return hipSuccess;
}

// ================================================================================================
hipError_t PlatformState::UnloadModule(hipModule_t hmod) {
  amd::ScopedLock lock(lock_);

  if (auto it = dynCO_map_.find(hmod); it == dynCO_map_.end()) {
    return hipErrorNotFound;
  } else {
    delete it->second;
    dynCO_map_.erase(it);  // Iterator-based erase avoids second lookup
  }

  // Remove all texture references associated with this module
  for (auto tex_it = texRef_map_.begin(); tex_it != texRef_map_.end(); ) {
    if (tex_it->second.first == hmod) {
      tex_it = texRef_map_.erase(tex_it);
    } else {
      ++tex_it;
    }
  }

  return hipSuccess;
}

// ================================================================================================
hipError_t PlatformState::GetDynFunc(hipFunction_t* hfunc, hipModule_t hmod,
                                     const char* func_name) {
  if (func_name[0] == '\0') {
    return hipErrorNotFound;
  }

  amd::ScopedLock lock(lock_);

  const auto it = dynCO_map_.find(hmod);
  if (it == dynCO_map_.end()) {
    LogPrintfError("Cannot find the module: 0x%x", hmod);
    return hipErrorNotFound;
  }

  return it->second->getDynFunc(hfunc, func_name);
}

// ================================================================================================
hipError_t PlatformState::GetFuncCount(unsigned int* count, hipModule_t hmod) {
  amd::ScopedLock lock(lock_);

  const auto it = dynCO_map_.find(hmod);
  if (it == dynCO_map_.end()) {
    LogPrintfError("Cannot find the module: 0x%x", hmod);
    return hipErrorNotFound;
  }
  return it->second->getFuncCount(count);
}

// ================================================================================================
bool PlatformState::IsValidDynFunc(const void* hfunc) {
  amd::ScopedLock lock(lock_);
  return std::any_of(dynCO_map_.begin(), dynCO_map_.end(),
                     [hfunc](const auto& entry) { return entry.second->isValidDynFunc(hfunc); });
}

// ================================================================================================
hipError_t PlatformState::GetDynGlobalVar(const char* hostVar, hipModule_t hmod,
                                          hipDeviceptr_t* dev_ptr, size_t* size_ptr) {
  amd::ScopedLock lock(lock_);

  if (hostVar == nullptr) {
    return hipErrorInvalidValue;
  }

  const auto it = dynCO_map_.find(hmod);
  if (it == dynCO_map_.end()) {
    LogPrintfError("Cannot find the module: 0x%x", hmod);
    return hipErrorNotFound;
  }
  if (dev_ptr) {
    *dev_ptr = nullptr;
  }
  IHIP_RETURN_ONFAIL(it->second->getManagedVarPointer(hostVar, dev_ptr, size_ptr));
  // if dev_ptr is nullptr, hostvar is not in managed variable list
  if ((dev_ptr && !*dev_ptr) || (size_ptr && *size_ptr == 0)) {
    auto* dvar = static_cast<hip::DeviceVar*>(nullptr);
    IHIP_RETURN_ONFAIL(it->second->getDeviceVar(&dvar, hostVar));
    if (dev_ptr) {
      *dev_ptr = dvar->device_ptr();
    }
    if (size_ptr) {
      *size_ptr = dvar->size();
    }
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t PlatformState::RegisterTexRef(textureReference* texRef, hipModule_t hmod,
                                         std::string name) {
  amd::ScopedLock lock(lock_);
  texRef_map_.insert(std::make_pair(texRef, std::make_pair(hmod, name)));
  return hipSuccess;
}

// ================================================================================================
hipError_t PlatformState::GetDynTexGlobalVar(textureReference* texRef, hipDeviceptr_t* dev_ptr,
                                             size_t* size_ptr) {
  amd::ScopedLock lock(lock_);

  const auto tex_it = texRef_map_.find(texRef);
  if (tex_it == texRef_map_.end()) {
    LogPrintfError("Cannot find the texRef Entry: 0x%x", texRef);
    return hipErrorNotFound;
  }

  const auto& tex_ref_entry = tex_it->second;
  const auto it = dynCO_map_.find(tex_ref_entry.first);
  if (it == dynCO_map_.end()) {
    LogPrintfError("Cannot find the module: 0x%x", tex_ref_entry.first);
    return hipErrorNotFound;
  }

  hip::DeviceVar* dvar;
  IHIP_RETURN_ONFAIL(it->second->getDeviceVar(&dvar, tex_ref_entry.second));
  *dev_ptr = dvar->device_ptr();
  *size_ptr = dvar->size();

  return hipSuccess;
}

// ================================================================================================
hipError_t PlatformState::GetDynTexRef(const char* hostVar, hipModule_t hmod,
                                       textureReference** texRef) {
  amd::ScopedLock lock(lock_);

  const auto it = dynCO_map_.find(hmod);
  if (it == dynCO_map_.end()) {
    LogPrintfError("Cannot find the module: 0x%x", hmod);
    return hipErrorNotFound;
  }

  hip::DeviceVar* dvar;
  IHIP_RETURN_ONFAIL(it->second->getDeviceVar(&dvar, hostVar));

  if (dvar->size() != sizeof(textureReference)) {
    return hipErrorNotFound;
  }

  dvar->shadowVptr = new texture<char>();
  *texRef = reinterpret_cast<textureReference*>(dvar->shadowVptr);
  return hipSuccess;
}

// ================================================================================================
void PlatformState::SetupArgument(const void* arg, size_t size, size_t offset) {
  auto& arguments = hip::tls.exec_stack_.top().arguments_;
  const size_t required_size = offset + size;
  arguments.resize(std::max(arguments.size(), required_size));
  std::memcpy(&arguments[offset], arg, size);
}

// ================================================================================================
void PlatformState::ConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                  hipStream_t stream) {
  hip::tls.exec_stack_.push(ihipExec_t{gridDim, blockDim, sharedMem, stream});
}

// ================================================================================================
void PlatformState::PopExec(ihipExec_t& exec) {
  exec = std::move(hip::tls.exec_stack_.top());
  hip::tls.exec_stack_.pop();
}

// ================================================================================================
std::shared_ptr<UniqueFD> PlatformState::GetUniqueFileHandle(const std::string& file_path) {
  amd::ScopedLock lock(ufd_lock_);

  auto it = ufd_map_.find(file_path);
  if (it != ufd_map_.end()) {
    return it->second;
  }

  // Get the file desc and file size from amd::Os API
  amd::Os::FileDesc fdesc;
  size_t fsize = 0;
  if (!amd::Os::GetFileHandle(file_path.c_str(), &fdesc, &fsize)) {
    return nullptr;
  }
  
  auto ufd = std::make_shared<UniqueFD>(file_path, fdesc, fsize);
  ufd_map_.emplace(file_path, ufd);
  return ufd;
}

// ================================================================================================
bool PlatformState::CloseUniqueFileHandle(const std::shared_ptr<UniqueFD>& ufd) {
  amd::ScopedLock lock(ufd_lock_);

  // if use_count is 2, then there is 1 entry in the map and the current entry is the last close.
  if (ufd.use_count() == 2) {
    ufd_map_.erase(ufd->fpath_);
    if (!amd::Os::CloseFileHandle(ufd->fdesc_)) {
      return false;
    }
  }
  return true;
}

// ================================================================================================
void* PlatformState::GetDynamicLibraryHandle() {
  amd::ScopedLock lock(lock_);

  if (dynamicLibraryHandle_ != nullptr) {
    return dynamicLibraryHandle_;
  }

#ifdef _WIN32
  static const std::string libName = "amdhip64.dll";
#else
  static const std::string libName = "libamdhip64.so." + std::to_string(HIP_VERSION_MAJOR);
#endif

  dynamicLibraryHandle_ = amd::Os::loadLibrary(libName.c_str());
  return dynamicLibraryHandle_;
}

// ================================================================================================
void PlatformState::SetDynamicLibraryHandle(void* handle) {
  amd::ScopedLock lock(lock_);
  dynamicLibraryHandle_ = handle;
}

}  // namespace hip
