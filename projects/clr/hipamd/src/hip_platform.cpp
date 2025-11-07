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

namespace hip {
constexpr unsigned __hipFatMAGIC2 = 0x48495046;  // "HIPF"
constexpr unsigned __hipFatMAGIC_KPACK = 0x4B504948;  // "HIPK" - kpack'd binary

PlatformState* PlatformState::platform_;  // Initiaized as nullptr by default

// forward declaration of methods required for __hipRegisrterManagedVar
hipError_t ihipMallocManaged(void** ptr, size_t size, size_t align = 0, bool use_host_ptr = 0);

struct __CudaFatBinaryWrapper {
  unsigned int magic;
  unsigned int version;
  void* binary;
  void* dummy1;
};

// Kpack metadata structure
struct KpackMetadata {
  std::vector<std::string> kpack_search_paths;
  std::string kernel_name;
};

// Helper to parse a MessagePack string (fixstr, str8, str16, str32)
static bool parseMsgpackString(const uint8_t*& ptr, const uint8_t* end, std::string& out) {
  if (ptr >= end) return false;

  uint32_t str_len = 0;
  uint8_t format = *ptr;

  if ((format & 0xe0) == 0xa0) {
    // fixstr: 101xxxxx
    str_len = format & 0x1f;
    ptr++;
  } else if (format == 0xd9) {
    // str8: length in next 1 byte
    ptr++;
    if (ptr >= end) return false;
    str_len = *ptr;
    ptr++;
  } else if (format == 0xda) {
    // str16: length in next 2 bytes (big-endian)
    ptr++;
    if (ptr + 2 > end) return false;
    str_len = (static_cast<uint32_t>(ptr[0]) << 8) | ptr[1];
    ptr += 2;
  } else if (format == 0xdb) {
    // str32: length in next 4 bytes (big-endian)
    ptr++;
    if (ptr + 4 > end) return false;
    str_len = (static_cast<uint32_t>(ptr[0]) << 24) |
              (static_cast<uint32_t>(ptr[1]) << 16) |
              (static_cast<uint32_t>(ptr[2]) << 8) |
              ptr[3];
    ptr += 4;
  } else {
    fprintf(stderr, "[HIP] Kpack metadata: expected string, got 0x%02x\n", format);
    return false;
  }

  if (ptr + str_len > end) return false;
  out = std::string(reinterpret_cast<const char*>(ptr), str_len);
  ptr += str_len;
  return true;
}

// Simple MessagePack parser for kpack metadata
// Only handles the specific format we use: map with string keys and string/array values
static bool parseKpackMetadata(const void* data, size_t max_size, KpackMetadata& metadata) {
  const uint8_t* ptr = static_cast<const uint8_t*>(data);
  const uint8_t* end = ptr + max_size;

  if (ptr >= end) return false;

  // Expect fixmap with 2 entries (0x82)
  if (*ptr != 0x82) {
    fprintf(stderr, "[HIP] Kpack metadata: expected fixmap, got 0x%02x\n", *ptr);
    return false;
  }
  ptr++;

  // Parse 2 key-value pairs
  for (int i = 0; i < 2; i++) {
    if (ptr >= end) return false;

    // Read key (string)
    std::string key;
    if (!parseMsgpackString(ptr, end, key)) {
      fprintf(stderr, "[HIP] Kpack metadata: failed to parse key\n");
      return false;
    }

    if (ptr >= end) return false;

    if (key == "kpack_search_paths") {
      // Expect fixarray
      if ((*ptr & 0xf0) != 0x90) {
        fprintf(stderr, "[HIP] Kpack metadata: expected fixarray for kpack_search_paths, got 0x%02x\n", *ptr);
        return false;
      }
      uint8_t arr_len = *ptr & 0x0f;
      ptr++;

      // Parse array elements (all strings)
      for (uint8_t j = 0; j < arr_len; j++) {
        std::string path;
        if (!parseMsgpackString(ptr, end, path)) {
          fprintf(stderr, "[HIP] Kpack metadata: failed to parse search path\n");
          return false;
        }
        metadata.kpack_search_paths.push_back(path);
      }
    } else if (key == "kernel_name") {
      // Parse kernel name string
      if (!parseMsgpackString(ptr, end, metadata.kernel_name)) {
        fprintf(stderr, "[HIP] Kpack metadata: failed to parse kernel_name\n");
        return false;
      }
    } else {
      fprintf(stderr, "[HIP] Kpack metadata: unknown key '%s'\n", key.c_str());
      return false;
    }
  }

  return true;
}

hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                              const char* name);

hipError_t ihipCreateGlobalVarObj(const char* name, hipModule_t hmod, amd::Memory** amd_mem_obj,
                                  hipDeviceptr_t* dptr, size_t* bytes);

extern hipError_t ihipModuleLaunchKernel(hipFunction_t f, amd::LaunchParams& launch_params,
                                         hipStream_t hStream, void** kernelParams, void** extra,
                                         hipEvent_t startEvent, hipEvent_t stopEvent,
                                         uint32_t flags = 0, uint32_t params = 0,
                                         uint32_t gridId = 0, uint32_t numGrids = 0,
                                         uint64_t prevGridSum = 0, uint64_t allGridSum = 0,
                                         uint32_t firstDevice = 0);
static bool isCompatibleCodeObject(const std::string& codeobj_target_id, const char* device_name) {
  // Workaround for device name mismatch.
  // Device name may contain feature strings delimited by '+', e.g.
  // gfx900+xnack. Currently HIP-Clang does not include feature strings
  // in code object target id in fat binary. Therefore drop the feature
  // strings from device name before comparing it with code object target id.
  std::string short_name(device_name);
  auto feature_loc = short_name.find('+');
  if (feature_loc != std::string::npos) {
    short_name.erase(feature_loc);
  }
  return codeobj_target_id == short_name;
}

void** __hipRegisterFatBinary(const void* data) {
  const __CudaFatBinaryWrapper* fbwrapper = reinterpret_cast<const __CudaFatBinaryWrapper*>(data);

  // Check for kpack'd binary by magic value (BEFORE dereferencing binary pointer!)
  if (fbwrapper->magic == __hipFatMAGIC_KPACK && fbwrapper->version == 1) {
    // Parse the MessagePack metadata
    KpackMetadata metadata;
    constexpr size_t MAX_METADATA_SIZE = 4096; // Reasonable upper bound

    if (!parseKpackMetadata(fbwrapper->binary, MAX_METADATA_SIZE, metadata)) {
      return nullptr;
    }

    // HACK: Check for env var pointing to extracted device code
    const char* device_code_path = getenv("HIP_KPACK_DEVICE_CODE");
    if (device_code_path == nullptr) {
      return nullptr;
    }

    // Load the .hsaco file from disk
    FILE* fp = fopen(device_code_path, "rb");
    if (!fp) {
      return nullptr;
    }

    // Get file size
    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // Allocate buffer and read file
    // NOTE: Using new[] to match FatBinaryInfo memory management
    char* hsaco_buffer = new char[file_size];
    size_t bytes_read = fread(hsaco_buffer, 1, file_size, fp);
    fclose(fp);

    if (bytes_read != file_size) {
      delete[] hsaco_buffer;
      return nullptr;
    }

    // Store the hsaco buffer in a static map for lazy loading
    // HACK: Using a static map to store kpack device code until runtime is initialized
    PlatformState::instance().registerKpackDeviceCode(fbwrapper, hsaco_buffer, file_size);

    // Call addFatBinary with initialized=false to defer loading
    // This creates an entry in modules_[fbwrapper] and returns a pointer to it
    bool success = false;
    auto fat_binary_info = PlatformState::instance().addFatBinary(fbwrapper, success);

    if (!success) {
      delete[] hsaco_buffer;
      return nullptr;
    }

    // Set to nullptr for lazy loading - digestFatBinary will be called later
    // when runtime is initialized and kernel is first used
    *fat_binary_info = nullptr;

    // Return the FatBinaryInfo** as expected
    return reinterpret_cast<void**>(fat_binary_info);
  }

  // Check for normal fat binary
  if (fbwrapper->magic != __hipFatMAGIC2 || fbwrapper->version != 1) {
    LogPrintfError("Cannot Register fat binary. FatMagic: %u version: %u ", fbwrapper->magic,
                   fbwrapper->version);
    return nullptr;
  }

  // Normal fat binary path - safe to dereference binary pointer
  bool success{};
  auto fat_binary_info = PlatformState::instance().addFatBinary(fbwrapper->binary, success);
  return success ? reinterpret_cast<void**>(fat_binary_info) : nullptr;
}

void __hipRegisterFunction(hip::FatBinaryInfo** modules, const void* hostFunction,
                           char* deviceFunction, const char* deviceName, unsigned int threadLimit,
                           uint3* tid, uint3* bid, dim3* blockDim, dim3* gridDim, int* wSize) {
  // Check if modules pointer itself is nullptr (shouldn't happen)
  if (modules == nullptr) {
    return;
  }

  // For kpack binaries, *modules will be nullptr but we still need to register the function
  // The device code will be loaded lazily when first accessed

  static int enable_deferred_loading{[]() {
    char* var = getenv("HIP_ENABLE_DEFERRED_LOADING");
    return var ? atoi(var) : 1;
  }()};
  hipError_t hip_error = hipSuccess;
  // Compiler might share same hostFunction and hence it's needless to have another
  // hip::Function and hip::Function is stored in map with hostFunction as key.
  // Creating hip::Function in such case, Leaks it.
  if (PlatformState::instance().getStatFuncName(hostFunction) == nullptr) {
    hip::Function* func = new hip::Function(std::string(deviceName), modules);
    hip_error = PlatformState::instance().registerStatFunction(hostFunction, func);
  }
  guarantee((hip_error == hipSuccess), "Cannot register Static function, error: %d", hip_error);

  if (!enable_deferred_loading) {
    HIP_INIT_VOID();
    hipFunction_t hfunc = nullptr;

    for (size_t dev_idx = 0; dev_idx < g_devices.size(); ++dev_idx) {
      hip_error = PlatformState::instance().getStatFunc(&hfunc, hostFunction, dev_idx);
      guarantee((hip_error == hipSuccess), "Cannot retrieve Static function, error: %d", hip_error);
    }
  }
}

// Registers a device-side global variable.
// For each global variable in device code, there is a corresponding shadow
// global variable in host code. The shadow host variable is used to keep
// track of the value of the device side global variable between kernel
// executions.
void __hipRegisterVar(hip::FatBinaryInfo** modules,  // The device modules containing code object
                      void* var,                     // The shadow variable in host code
                      char* hostVar,                 // Variable name in host code
                      char* deviceVar,               // Variable name in device code
                      int ext,                       // Whether this variable is external
                      size_t size,                   // Size of the variable
                      int constant,                  // Whether this variable is constant
                      int global)                    // Unknown, always 0
{
  hip::Var* var_ptr = new hip::Var(std::string(hostVar), hip::Var::DeviceVarKind::DVK_Variable,
                                   size, 0, 0, modules);
  hipError_t err = PlatformState::instance().registerStatGlobalVar(var, var_ptr);
  guarantee((err == hipSuccess), "Cannot register Static Global Var, error:%d", err);
}

void __hipRegisterSurface(
    hip::FatBinaryInfo** modules,  // The device modules containing code object
    void* var,                     // The shadow variable in host code
    char* hostVar,                 // Variable name in host code
    char* deviceVar,               // Variable name in device code
    int type, int ext) {
  hip::Var* var_ptr = new hip::Var(std::string(hostVar), hip::Var::DeviceVarKind::DVK_Surface,
                                   sizeof(surfaceReference), 0, 0, modules);
  hipError_t err = PlatformState::instance().registerStatGlobalVar(var, var_ptr);
  guarantee((err == hipSuccess), "Cannot register Static Glbal Var, err:%d", err);
}

void __hipRegisterManagedVar(
    void* hipModule,  // Pointer to hip module returned from __hipRegisterFatbinary
    void** pointer,   // Pointer to a chunk of managed memory with size \p size and alignment \p
                      // align HIP runtime allocates such managed memory and assign it to \p pointer
    void* init_value,  // Initial value to be copied into \p pointer
    const char* name,  // Name of the variable in code object
    size_t size, unsigned align) {
  static int enable_deferred_loading{[]() {
#ifdef _WIN32  // Don't defer loading for windows
    return 0;
#else
    char* var = getenv("HIP_ENABLE_DEFERRED_LOADING");
    return var ? atoi(var) : 1;
#endif
  }()};
  hipError_t hip_error = hipSuccess;
  hip::Var* var_ptr = new hip::Var(std::string(name), hip::Var::DeviceVarKind::DVK_Managed, pointer,
                                   size, align, reinterpret_cast<hip::FatBinaryInfo**>(hipModule));
  hipError_t status = PlatformState::instance().registerStatManagedVar(var_ptr);
  guarantee((status == hipSuccess), "Cannot register Static Managed Var, error: %d", status);

  if (enable_deferred_loading) {
    // Allocate temporary var on host and initialize
    *pointer = amd::Os::reserveMemory(0, size, align, amd::Os::MEM_PROT_RW);
    ::memcpy(*pointer, init_value, size);
  } else {
    HIP_INIT_VOID();
    hipError_t status = ihipMallocManaged(pointer, size, align, 0);
    var_ptr->setAllocFlag(true);  // set flag true for managed alloc
    if (status == hipSuccess) {
      hip::Stream* stream = hip::getNullStream();
      if (stream != nullptr) {
        status = ihipMemcpy(*pointer, init_value, size, hipMemcpyHostToDevice, *stream);
        guarantee((status == hipSuccess), "Error during memcpy to managed memory, error:%d!",
                  status);
      } else {
        ClPrint(amd::LOG_ERROR, amd::LOG_API, "Host Queue is NULL");
      }
    } else {
      guarantee(false, "Error during allocation of managed memory!, error: %d", status);
    }
  }
}

void __hipRegisterTexture(
    hip::FatBinaryInfo** modules,  // The device modules containing code object
    void* var,                     // The shadow variable in host code
    char* hostVar,                 // Variable name in host code
    char* deviceVar,               // Variable name in device code
    int type, int norm, int ext) {
  hip::Var* var_ptr = new hip::Var(std::string(hostVar), hip::Var::DeviceVarKind::DVK_Texture,
                                   sizeof(textureReference), 0, 0, modules);
  hipError_t err = PlatformState::instance().registerStatGlobalVar(var, var_ptr);
  guarantee((err == hipSuccess), "Cannot register Static Global Var, status: %d", err);
}

void __hipUnregisterFatBinary(hip::FatBinaryInfo** modules) {
  static std::once_flag unregister_device_sync;
  // If SKIP ABORT is set and GPU is in error, dont need to sync streams.
  if (!HIP_SKIP_ABORT_ON_GPU_ERROR || !amd::Device::IsGPUInError()) {
    std::call_once(unregister_device_sync, []() {
      for (auto& hipDevice : g_devices) {
        // By synchronizing devices ensure that all HSA signal handlers
        // complete before removeFatBinary
        hipDevice->SyncAllStreams(true);
      }
    });
  }
  hipError_t err = PlatformState::instance().removeFatBinary(modules);
  guarantee((err == hipSuccess), "Cannot Unregister Fat Binary, error:%d", err);
}

void __hipRegisterFunction(void** modules, const void* hostFunction, char* deviceFunction,
                           const char* deviceName, unsigned int threadLimit, uint3* tid, uint3* bid,
                           dim3* blockDim, dim3* gridDim, int* wSize) {
  return __hipRegisterFunction(reinterpret_cast<hip::FatBinaryInfo**>(modules), hostFunction,
                               deviceFunction, deviceName, threadLimit, tid, bid, blockDim, gridDim,
                               wSize);
}
void __hipRegisterSurface(void** modules, void* var, char* hostVar, char* deviceVar, int type,
                          int ext) {
  return __hipRegisterSurface(reinterpret_cast<hip::FatBinaryInfo**>(modules), var, hostVar,
                              deviceVar, type, ext);
}
void __hipRegisterTexture(void** modules, void* var, char* hostVar, char* deviceVar, int type,
                          int norm, int ext) {
  return __hipRegisterTexture(reinterpret_cast<hip::FatBinaryInfo**>(modules), var, hostVar,
                              deviceVar, type, norm, ext);
}
void __hipRegisterVar(void** modules, void* var, char* hostVar, char* deviceVar, int ext,
                      size_t size, int constant, int global) {
  return __hipRegisterVar(reinterpret_cast<hip::FatBinaryInfo**>(modules), var, hostVar, deviceVar,
                          ext, size, constant, global);
}
void __hipUnregisterFatBinary(void** modules) {
  return __hipUnregisterFatBinary(reinterpret_cast<hip::FatBinaryInfo**>(modules));
}

hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream) {
  HIP_INIT_API(hipConfigureCall, gridDim, blockDim, sharedMem, stream);

  PlatformState::instance().configureCall(gridDim, blockDim, sharedMem, stream);

  HIP_RETURN(hipSuccess);
}

hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                      hipStream_t stream) {
  HIP_INIT_API(__hipPushCallConfiguration, gridDim, blockDim, sharedMem, stream);

  PlatformState::instance().configureCall(gridDim, blockDim, sharedMem, stream);

  HIP_RETURN(hipSuccess);
}

hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem,
                                     hipStream_t* stream) {
  HIP_INIT_API(__hipPopCallConfiguration, gridDim, blockDim, sharedMem, stream);

  ihipExec_t exec;
  PlatformState::instance().popExec(exec);
  *gridDim = exec.gridDim_;
  *blockDim = exec.blockDim_;
  *sharedMem = exec.sharedMem_;
  *stream = exec.hStream_;

  HIP_RETURN(hipSuccess);
}

hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset) {
  HIP_INIT_API(hipSetupArgument, arg, size, offset);

  PlatformState::instance().setupArgument(arg, size, offset);

  HIP_RETURN(hipSuccess);
}

hipError_t hipLaunchByPtr(const void* hostFunction) {
  HIP_INIT_API(hipLaunchByPtr, hostFunction);

  ihipExec_t exec;
  PlatformState::instance().popExec(exec);

  hip::Stream* stream = reinterpret_cast<hip::Stream*>(exec.hStream_);
  int deviceId = (stream != nullptr) ? stream->DeviceId() : ihipGetDevice();
  if (deviceId == -1) {
    LogPrintfError("Wrong DeviceId: %d", deviceId);
    HIP_RETURN(hipErrorNoDevice);
  }
  hipFunction_t func = nullptr;
  hipError_t hip_error = PlatformState::instance().getStatFunc(&func, hostFunction, deviceId);
  if ((hip_error != hipSuccess) || (func == nullptr)) {
    LogPrintfError("Could not retrieve hostFunction: 0x%x", hostFunction);
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  size_t size = exec.arguments_.size();
  void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &exec.arguments_[0],
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};

  STREAM_CAPTURE(hipLaunchByPtr, exec.hStream_, func, exec.blockDim_, exec.gridDim_,
                 exec.sharedMem_, extra);

  HIP_RETURN(hipModuleLaunchKernel(func, exec.gridDim_.x, exec.gridDim_.y, exec.gridDim_.z,
                                   exec.blockDim_.x, exec.blockDim_.y, exec.blockDim_.z,
                                   exec.sharedMem_, exec.hStream_, nullptr, extra));
}

hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol) {
  HIP_INIT_API(hipGetSymbolAddress, devPtr, symbol);

  hipError_t hip_error = hipSuccess;
  if (devPtr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  size_t sym_size = 0;

  HIP_RETURN_ONFAIL(
      PlatformState::instance().getStatGlobalVar(symbol, ihipGetDevice(), devPtr, &sym_size));

  HIP_RETURN(hipSuccess, *devPtr);
}

hipError_t hipGetSymbolSize(size_t* sizePtr, const void* symbol) {
  HIP_INIT_API(hipGetSymbolSize, sizePtr, symbol);

  if (sizePtr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipDeviceptr_t device_ptr = nullptr;
  HIP_RETURN_ONFAIL(
      PlatformState::instance().getStatGlobalVar(symbol, ihipGetDevice(), &device_ptr, sizePtr));

  HIP_RETURN(hipSuccess, *sizePtr);
}

hipError_t ihipCreateGlobalVarObj(const char* name, hipModule_t hmod, amd::Memory** amd_mem_obj,
                                  hipDeviceptr_t* dptr, size_t* bytes) {
  /* Get Device Program pointer*/
  amd::Program* program = as_amd(reinterpret_cast<cl_program>(hmod));
  device::Program* dev_program = program->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  if (dev_program == nullptr) {
    LogPrintfError("Cannot get Device Function for module: 0x%x", hmod);
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }
  /* Find the global Symbols */
  if (!dev_program->createGlobalVarObj(amd_mem_obj, dptr, bytes, name)) {
    LogPrintfError("Cannot create Global Var obj for symbol: %s", name);
    HIP_RETURN(hipErrorInvalidSymbol);
  }

  HIP_RETURN(hipSuccess);
}
}  // namespace hip

namespace hip_impl {
hipError_t ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    int* maxBlocksPerCU, int* numBlocksPerGrid, int* bestBlockSize, const amd::Device& device,
    hipFunction_t func, int inputBlockSize, size_t dynamicSMemSize, bool bCalcPotentialBlkSz) {
  hip::DeviceFunc* function = hip::DeviceFunc::asFunction(func);
  const amd::Kernel& kernel = *function->kernel();

  const device::Kernel::WorkGroupInfo* wrkGrpInfo = kernel.getDeviceKernel(device)->workGroupInfo();
  if (bCalcPotentialBlkSz == false) {
    if (inputBlockSize <= 0) {
      return hipErrorInvalidValue;
    }
    *bestBlockSize = 0;
    // Make sure the requested block size is smaller than max supported
    if (inputBlockSize > int(device.info().maxWorkGroupSize_)) {
      *maxBlocksPerCU = 0;
      *numBlocksPerGrid = 0;
      return hipSuccess;
    }
  } else {
    if (inputBlockSize > int(device.info().maxWorkGroupSize_) || inputBlockSize <= 0) {
      // The user wrote the kernel to work with a workgroup size
      // bigger than this hardware can support. Or they do not care
      // about the size So just assume its maximum size is
      // constrained by hardware
      inputBlockSize = device.info().maxWorkGroupSize_;
    }
  }
  // Find wave occupancy per CU => simd_per_cu * GPR usage
  size_t MaxWavesPerSimd;

  if (device.isa().versionMajor() <= 9) {
    MaxWavesPerSimd = 8;  // Limited by SPI 32 per CU, hence 8 per SIMD
  } else {
    MaxWavesPerSimd = 16;
  }
  size_t VgprWaves = MaxWavesPerSimd;
  uint32_t VgprGranularity = device.info().vgprAllocGranularity_;
  size_t maxVGPRs = device.info().vgprsPerSimd_;
  size_t wavefrontSize = wrkGrpInfo->wavefrontSize_;
  if (device.isa().versionMajor() >= 10) {
    if (wavefrontSize == 64) {
      maxVGPRs = maxVGPRs >> 1;
      VgprGranularity = VgprGranularity >> 1;
    }
  }
  if (wrkGrpInfo->usedVGPRs_ > 0) {
    VgprWaves = maxVGPRs / amd::alignUp(wrkGrpInfo->usedVGPRs_, VgprGranularity);
  }

  if (VgprWaves == 0) {
    // This should not happen ideally, but in case the value is
    // incorrect, it can lead to a crash. By returning error, API can exit gracefully.
    return hipErrorUnknown;
  }

  size_t GprWaves = VgprWaves;
  if (wrkGrpInfo->usedSGPRs_ > 0) {
    size_t maxSGPRs = device.info().sgprsPerSimd_;
    const size_t SgprWaves = maxSGPRs / amd::alignUp(wrkGrpInfo->usedSGPRs_, 16);
    GprWaves = std::min(VgprWaves, SgprWaves);
  }

  // The table contains SIMD per CU, not per WGP, so when WGP mode is set on kernel metadata,
  // multiply the number of SIMDs by 2, to account for 2CUs in 1 WGP.
  uint32_t simdPerCU = device.isa().simdPerCU();
  if (wrkGrpInfo->isWGPMode_) {
    simdPerCU *= 2;
  }

  const size_t alu_occupancy = simdPerCU * std::min(MaxWavesPerSimd, GprWaves);
  const int alu_limited_threads = alu_occupancy * wrkGrpInfo->wavefrontSize_;

  int lds_occupancy_wgs = INT_MAX;
  const size_t total_used_lds = wrkGrpInfo->usedLDSSize_ + dynamicSMemSize;
  if (total_used_lds != 0) {
    lds_occupancy_wgs = static_cast<int>(device.info().localMemSize_ / total_used_lds);
  }
  // Calculate how many blocks of inputBlockSize we can fit per CU
  // Need to align with hardware wavefront size. If they want 65 threads, but
  // waves are 64, then we need 128 threads per block.
  // So this calculates how many blocks we can fit.
  *maxBlocksPerCU = alu_limited_threads / amd::alignUp(inputBlockSize, wrkGrpInfo->wavefrontSize_);
  // Unless those blocks are further constrained by LDS size.
  *maxBlocksPerCU = std::min(*maxBlocksPerCU, lds_occupancy_wgs);

  // Some callers of this function want to return the block size, in threads, that
  // leads to the maximum occupancy. In that case, inputBlockSize is the maximum
  // workgroup size the user wants to allow, or that the hardware can allow.
  // It is either the number of threads that we are limited to due to occupancy, or
  // the maximum available block size for this kernel, which could have come from the
  // user. e.g., if the user indicates the maximum block size is 64 threads, but we
  // calculate that 128 threads can fit in each CU, we have to give up and return 64.
  *bestBlockSize =
      std::min(alu_limited_threads, amd::alignUp(inputBlockSize, wrkGrpInfo->wavefrontSize_));
  // If the best block size is smaller than the block size used to fit the maximum,
  // then we need to make the grid bigger for full occupancy.
  const int bestBlocksPerCU = alu_limited_threads / (*bestBlockSize);
  uint32_t maxCUs = device.info().maxComputeUnits_;
  if (wrkGrpInfo->isWGPMode_ == false && device.settings().enableWgpMode_ == true) {
    maxCUs *= 2;
  } else if ((wrkGrpInfo->isWGPMode_ == true && device.settings().enableWgpMode_ == false)) {
    maxCUs /= 2;
  }
  // Unless those blocks are further constrained by LDS size.
  *numBlocksPerGrid = (maxCUs * std::min(bestBlocksPerCU, lds_occupancy_wgs));

  return hipSuccess;
}
}  // namespace hip_impl

namespace hip {
hipError_t hipOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* f,
                                                    int numBlocks, int blockSize){
  HIP_INIT_API(hipOccupancyAvailableDynamicSMemPerBlock, dynamicSmemSize, f, numBlocks, blockSize);
  if (dynamicSmemSize == nullptr || numBlocks <= 0 || blockSize <= 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipFunction_t func;
  int dev_id = ihipGetDevice();
  hipError_t hip_error = PlatformState::instance().getStatFunc(&func, f, dev_id);

  if (hip_error != hipSuccess || func == nullptr) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  hip::DeviceFunc* function = hip::DeviceFunc::asFunction(func);
  if (function == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  hipDeviceProp_t prop = {0};
  HIP_RETURN_ONFAIL(ihipGetDeviceProperties(&prop, dev_id));

  const amd::Device& device = *hip::getCurrentDevice()->devices()[dev_id];
  const amd::Kernel& kernel = *function->kernel();
  const device::Kernel::WorkGroupInfo* wrkGrpInfo = kernel.getDeviceKernel(device)->workGroupInfo();

  const int staticSharedMemoryUsage = wrkGrpInfo->usedLDSSize_;
  const int maxDynamicSharedSizeBytes = wrkGrpInfo->maxDynamicSharedSizeBytes_;
  const int maxNumBlocks = static_cast<int>(floor((prop.maxThreadsPerMultiProcessor) / blockSize));
  const int maxSharedMemoryPerMultiProcessor = prop.maxSharedMemoryPerMultiProcessor - staticSharedMemoryUsage * std::min(numBlocks,
                                                                                                                          maxNumBlocks);
  const size_t maxDynamicSmemSize = std::min(static_cast<int>(floor(maxSharedMemoryPerMultiProcessor / maxNumBlocks)),
                                                                      maxDynamicSharedSizeBytes);
  const int alignmentSize = device.isa().ldsAlignment();

  size_t dynamic_smem_size = 0;
  dynamic_smem_size = std::min(static_cast<int>(floor(maxSharedMemoryPerMultiProcessor / numBlocks)),
                               maxDynamicSharedSizeBytes);
  dynamic_smem_size = std::max(maxDynamicSmemSize, dynamic_smem_size);
  *dynamicSmemSize = amd::alignDown(dynamic_smem_size, alignmentSize);

  HIP_RETURN(hipSuccess);
}

hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize, const void* f,
                                             size_t dynSharedMemPerBlk, int blockSizeLimit) {
  HIP_INIT_API(hipOccupancyMaxPotentialBlockSize, f, dynSharedMemPerBlk, blockSizeLimit);
  if ((gridSize == nullptr) || (blockSize == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipFunction_t func = nullptr;
  hipError_t hip_error = PlatformState::instance().getStatFunc(&func, f, ihipGetDevice());
  if ((hip_error != hipSuccess) || (func == nullptr)) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];
  int max_blocks_per_grid = 0;
  int num_blocks = 0;
  int best_block_size = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, &max_blocks_per_grid, &best_block_size, device, func, blockSizeLimit,
      dynSharedMemPerBlk, true);
  if (ret == hipSuccess) {
    *blockSize = best_block_size;
    *gridSize = max_blocks_per_grid;
  }
  HIP_RETURN(ret);
}

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
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, &max_blocks_per_grid, &best_block_size, device, f, blockSizeLimit,
      dynSharedMemPerBlk, true);
  if (ret == hipSuccess) {
    *blockSize = best_block_size;
    *gridSize = max_blocks_per_grid;
  }
  HIP_RETURN(ret);
}

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
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, &max_blocks_per_grid, &best_block_size, device, f, blockSizeLimit,
      dynSharedMemPerBlk, true);
  if (ret == hipSuccess) {
    *blockSize = best_block_size;
    *gridSize = max_blocks_per_grid;
  }
  HIP_RETURN(ret);
}

hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, hipFunction_t f,
                                                              int blockSize,
                                                              size_t dynSharedMemPerBlk) {
  HIP_INIT_API(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, f, blockSize,
               dynSharedMemPerBlk);
  if (numBlocks == nullptr || (f == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];

  int num_blocks = 0;
  int max_blocks_per_grid = 0;
  int best_block_size = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, &max_blocks_per_grid, &best_block_size, device, f, blockSize, dynSharedMemPerBlk,
      false);
  *numBlocks = num_blocks;
  HIP_RETURN(ret);
}

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

  int num_blocks = 0;
  int max_blocks_per_grid = 0;
  int best_block_size = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, &max_blocks_per_grid, &best_block_size, device, f, blockSize, dynSharedMemPerBlk,
      false);
  *numBlocks = num_blocks;
  HIP_RETURN(ret);
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* f,
                                                        int blockSize, size_t dynamicSMemSize) {
  HIP_INIT_API(hipOccupancyMaxActiveBlocksPerMultiprocessor, f, blockSize, dynamicSMemSize);
  if (numBlocks == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipFunction_t func = nullptr;
  hipError_t hip_error = PlatformState::instance().getStatFunc(&func, f, ihipGetDevice());
  if ((hip_error != hipSuccess) || (func == nullptr)) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];

  int num_blocks = 0;
  int max_blocks_per_grid = 0;
  int best_block_size = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, &max_blocks_per_grid, &best_block_size, device, func, blockSize, dynamicSMemSize,
      false);
  *numBlocks = num_blocks;
  HIP_RETURN(ret);
}

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
  hipError_t hip_error = PlatformState::instance().getStatFunc(&func, f, ihipGetDevice());
  if ((hip_error != hipSuccess) || (func == nullptr)) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];

  int num_blocks = 0;
  int max_blocks_per_grid = 0;
  int best_block_size = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, &max_blocks_per_grid, &best_block_size, device, func, blockSize, dynamicSMemSize,
      false);
  *numBlocks = num_blocks;
  HIP_RETURN(ret);
}

hipError_t ihipLaunchKernel(const void* hostFunction, dim3 gridDim, dim3 blockDim, void** args,
                            size_t sharedMemBytes, hipStream_t stream, hipEvent_t startEvent,
                            hipEvent_t stopEvent, int flags) {
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  if (hostFunction == nullptr) {
    return hipErrorInvalidDeviceFunction;
  }

  hipFunction_t func = nullptr;
  int deviceId = hip::Stream::DeviceId(stream);

  hipError_t hip_error =
      PlatformState::instance().getStatFunc(&func, hostFunction, deviceId);
  if ((hip_error != hipSuccess) || (func == nullptr)) {
    // assume its hip function type if we did not get a valid output from static
    // func lookup
    func = reinterpret_cast<hipFunction_t>(const_cast<void *>(hostFunction));
  }

  constexpr auto gridDimYZmax = static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1;
  const auto& isa = g_devices[deviceId]->devices()[0]->isa().versionMajor();
  if (isa >= 12 && (gridDim.y > gridDimYZmax || gridDim.z > gridDimYZmax)) {
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

// conversion routines between float and half precision

static inline std::uint32_t f32_as_u32(float f) {
  union {
    float f;
    std::uint32_t u;
  } v;
  v.f = f;
  return v.u;
}

static inline float u32_as_f32(std::uint32_t u) {
  union {
    float f;
    std::uint32_t u;
  } v;
  v.u = u;
  return v.f;
}

static inline int clamp_int(int i, int l, int h) { return std::min(std::max(i, l), h); }


// half float, the f16 is in the low 16 bits of the input argument

static inline float __convert_half_to_float(std::uint32_t a) noexcept {
  std::uint32_t u = ((a << 13) + 0x70000000U) & 0x8fffe000U;

  std::uint32_t v =
      f32_as_u32(u32_as_f32(u) * u32_as_f32(0x77800000U) /*0x1.0p+112f*/) + 0x38000000U;

  u = (a & 0x7fff) != 0 ? v : u;

  return u32_as_f32(u) * u32_as_f32(0x07800000U) /*0x1.0p-112f*/;
}

// float half with nearest even rounding
// The lower 16 bits of the result is the bit pattern for the f16
static inline std::uint32_t __convert_float_to_half(float a) noexcept {
  std::uint32_t u = f32_as_u32(a);
  int e = static_cast<int>((u >> 23) & 0xff) - 127 + 15;
  std::uint32_t m = ((u >> 11) & 0xffe) | ((u & 0xfff) != 0);
  std::uint32_t i = 0x7c00 | (m != 0 ? 0x0200 : 0);
  std::uint32_t n = ((std::uint32_t)e << 12) | m;
  std::uint32_t s = (u >> 16) & 0x8000;
  int b = clamp_int(1 - e, 0, 13);
  std::uint32_t d = (0x1000 | m) >> b;
  d |= (d << b) != (0x1000 | m);
  std::uint32_t v = e < 1 ? d : n;
  v = (v >> 2) + (((v & 0x7) == 3) | ((v & 0x7) > 5));
  v = e > 30 ? 0x7c00 : v;
  v = e == 143 ? i : v;
  return s | v;
}

extern "C"
#if !defined(_MSC_VER)
    __attribute__((weak))
#endif
    float
    __gnu_h2f_ieee(unsigned short h) {
  return __convert_half_to_float((std::uint32_t)h);
}

extern "C"
#if !defined(_MSC_VER)
    __attribute__((weak))
#endif
    unsigned short
    __gnu_f2h_ieee(float f) {
  return (unsigned short)__convert_float_to_half(f);
}

void PlatformState::init() {
  amd::ScopedLock lock(lock_);
  if (initialized_ || g_devices.empty()) {
    return;
  }
  initialized_ = true;
  for (auto& it : statCO_.vars_) {
    it.second->resize_dVar(g_devices.size());
  }
  for (auto& it : statCO_.managedVars_) {
    for (auto& var : it.second) {
      var->resize_dVar(g_devices.size());
    }
  }
  for (auto& it : statCO_.functions_) {
    it.second->resize_dFunc(g_devices.size());
  }
}

hipError_t PlatformState::loadModule(hipModule_t* module, const char* fname, const void* image) {
  if (module == nullptr) {
    return hipErrorInvalidValue;
  }

  hip::DynCO* dynCo = new hip::DynCO();
  hipError_t hip_error = dynCo->loadCodeObject(fname, image);
  if (hip_error != hipSuccess) {
    delete dynCo;
    return hip_error;
  }

  *module = dynCo->getModule();
  assert(*module != nullptr);

  amd::ScopedLock lock(lock_);
  if (dynCO_map_.find(*module) != dynCO_map_.end()) {
    delete dynCo;
    return hipErrorAlreadyMapped;
  }
  dynCO_map_.insert(std::make_pair(*module, dynCo));

  return hipSuccess;
}

hipError_t PlatformState::unloadModule(hipModule_t hmod) {
  amd::ScopedLock lock(lock_);

  auto it = dynCO_map_.find(hmod);
  if (it == dynCO_map_.end()) {
    return hipErrorNotFound;
  }

  delete it->second;
  dynCO_map_.erase(hmod);

  auto tex_it = texRef_map_.begin();
  while (tex_it != texRef_map_.end()) {
    if (tex_it->second.first == hmod) {
      tex_it = texRef_map_.erase(tex_it);
    } else {
      ++tex_it;
    }
  }

  return hipSuccess;
}

hipError_t PlatformState::getDynFunc(hipFunction_t* hfunc, hipModule_t hmod,
                                     const char* func_name) {
  amd::ScopedLock lock(lock_);

  auto it = dynCO_map_.find(hmod);
  if (it == dynCO_map_.end()) {
    LogPrintfError("Cannot find the module: 0x%x", hmod);
    return hipErrorNotFound;
  }
  if (0 == strlen(func_name)) {
    return hipErrorNotFound;
  }

  return it->second->getDynFunc(hfunc, func_name);
}

hipError_t PlatformState::getFuncCount(unsigned int* count, hipModule_t hmod) {
  amd::ScopedLock lock(lock_);

  auto it = dynCO_map_.find(hmod);
  if (it == dynCO_map_.end()) {
    LogPrintfError("Cannot find the module: 0x%x", hmod);
    return hipErrorNotFound;
  }
  return it->second->getFuncCount(count);
}

bool PlatformState::isValidDynFunc(const void* hfunc) {
  amd::ScopedLock lock(lock_);
  return std::any_of(dynCO_map_.begin(), dynCO_map_.end(),
                     [&](auto& it) { return it.second->isValidDynFunc(hfunc); });
}

hipError_t PlatformState::getDynGlobalVar(const char* hostVar, hipModule_t hmod,
                                          hipDeviceptr_t* dev_ptr, size_t* size_ptr) {
  amd::ScopedLock lock(lock_);

  if (hostVar == nullptr) {
    return hipErrorInvalidValue;
  }

  auto it = dynCO_map_.find(hmod);
  if (it == dynCO_map_.end()) {
    LogPrintfError("Cannot find the module: 0x%x", hmod);
    return hipErrorNotFound;
  }
  if (dev_ptr) {
    *dev_ptr = nullptr;
  }
  IHIP_RETURN_ONFAIL(it->second->getManagedVarPointer(hostVar, dev_ptr, size_ptr));
  // if dev_ptr is nullptr, hostvar is not in managed variable list
  if ((dev_ptr && *dev_ptr == nullptr) || (size_ptr && *size_ptr == 0)) {
    hip::DeviceVar* dvar = nullptr;
    IHIP_RETURN_ONFAIL(it->second->getDeviceVar(&dvar, hostVar));
    if (dev_ptr != nullptr) {
      *dev_ptr = dvar->device_ptr();
    }
    if (size_ptr != nullptr) {
      *size_ptr = dvar->size();
    }
  }
  return hipSuccess;
}

hipError_t PlatformState::registerTexRef(textureReference* texRef, hipModule_t hmod,
                                         std::string name) {
  amd::ScopedLock lock(lock_);
  texRef_map_.insert(std::make_pair(texRef, std::make_pair(hmod, name)));
  return hipSuccess;
}

hipError_t PlatformState::getDynTexGlobalVar(textureReference* texRef, hipDeviceptr_t* dev_ptr,
                                             size_t* size_ptr) {
  amd::ScopedLock lock(lock_);

  auto tex_it = texRef_map_.find(texRef);
  if (tex_it == texRef_map_.end()) {
    LogPrintfError("Cannot find the texRef Entry: 0x%x", texRef);
    return hipErrorNotFound;
  }

  auto it = dynCO_map_.find(tex_it->second.first);
  if (it == dynCO_map_.end()) {
    LogPrintfError("Cannot find the module: 0x%x", tex_it->second.first);
    return hipErrorNotFound;
  }

  hip::DeviceVar* dvar = nullptr;
  IHIP_RETURN_ONFAIL(it->second->getDeviceVar(&dvar, tex_it->second.second));
  *dev_ptr = dvar->device_ptr();
  *size_ptr = dvar->size();

  return hipSuccess;
}

hipError_t PlatformState::getDynTexRef(const char* hostVar, hipModule_t hmod,
                                       textureReference** texRef) {
  amd::ScopedLock lock(lock_);

  auto it = dynCO_map_.find(hmod);
  if (it == dynCO_map_.end()) {
    LogPrintfError("Cannot find the module: 0x%x", hmod);
    return hipErrorNotFound;
  }

  hip::DeviceVar* dvar = nullptr;
  IHIP_RETURN_ONFAIL(it->second->getDeviceVar(&dvar, hostVar));

  if (dvar->size() != sizeof(textureReference)) {
    return hipErrorNotFound;  // Any better way to verify texture type?
  }

  dvar->shadowVptr = new texture<char>();
  *texRef = reinterpret_cast<textureReference*>(dvar->shadowVptr);
  return hipSuccess;
}

void PlatformState::registerKpackDeviceCode(const void* key, char* buffer, size_t size) {
  kpack_device_code_map_[key] = std::make_pair(buffer, size);
}

bool PlatformState::getKpackDeviceCode(const void* key, char*& buffer, size_t& size) {
  auto it = kpack_device_code_map_.find(key);
  if (it != kpack_device_code_map_.end()) {
    buffer = it->second.first;
    size = it->second.second;
    return true;
  }
  return false;
}

hipError_t PlatformState::digestFatBinary(const void* data, hip::FatBinaryInfo*& programs) {
  // HACK: Check if this is a kpack binary
  char* kpack_buffer = nullptr;
  size_t kpack_size = 0;
  if (getKpackDeviceCode(data, kpack_buffer, kpack_size)) {
    // Now runtime should be initialized
    if (g_devices.size() == 0) {
      return hipErrorNoDevice;
    }

    // Create FatBinaryInfo with nullptr image (don't pass the .hsaco, it's not a fat binary!)
    hip::FatBinaryInfo* fb_info = new hip::FatBinaryInfo(nullptr, nullptr);

    // Add device program for each device
    // The kpack_buffer contains a single-arch .hsaco file (ELF), not a fat binary
    for (size_t dev_idx = 0; dev_idx < g_devices.size(); dev_idx++) {
      hipError_t status = fb_info->AddDevProgram(
          g_devices[dev_idx],
          kpack_buffer,
          kpack_size,
          0
      );

      if (status != hipSuccess) {
        delete fb_info;
        return status;
      }
    }

    // Build the program for each device
    for (size_t dev_idx = 0; dev_idx < g_devices.size(); dev_idx++) {
      hipError_t status = fb_info->BuildProgram(dev_idx);

      if (status != hipSuccess) {
        delete fb_info;
        return status;
      }
    }

    programs = fb_info;
    return hipSuccess;
  }

  // Normal fat binary path
  return statCO_.digestFatBinary(data, programs);
}

hip::FatBinaryInfo** PlatformState::addFatBinary(const void* data, bool& success) {
  return statCO_.addFatBinary(data, initialized_, success);
}

hipError_t PlatformState::removeFatBinary(hip::FatBinaryInfo** module) {
  return statCO_.removeFatBinary(module);
}

hipError_t PlatformState::registerStatFunction(const void* hostFunction, hip::Function* func) {
  return statCO_.registerStatFunction(hostFunction, func);
}

hipError_t PlatformState::registerStatGlobalVar(const void* hostVar, hip::Var* var) {
  return statCO_.registerStatGlobalVar(hostVar, var);
}

hipError_t PlatformState::registerStatManagedVar(hip::Var* var) {
  return statCO_.registerStatManagedVar(var);
}

const char* PlatformState::getStatFuncName(const void* hostFunction) {
  return statCO_.getStatFuncName(hostFunction);
}

hipError_t PlatformState::getStatFunc(hipFunction_t* hfunc, const void* hostFunction,
                                      int deviceId) {
  return statCO_.getStatFunc(hfunc, hostFunction, deviceId);
}

hipError_t PlatformState::getStatFuncAttr(hipFuncAttributes* func_attr, const void* hostFunction,
                                          int deviceId) {
  if (func_attr == nullptr) {
    return hipErrorInvalidValue;
  }
  if (hostFunction == nullptr) {
    return hipErrorInvalidDeviceFunction;
  }
  return statCO_.getStatFuncAttr(func_attr, hostFunction, deviceId);
}

hipError_t PlatformState::getStatGlobalVar(const void* hostVar, int deviceId,
                                           hipDeviceptr_t* dev_ptr, size_t* size_ptr) {
  return statCO_.getStatGlobalVar(hostVar, deviceId, dev_ptr, size_ptr);
}

hipError_t PlatformState::initStatManagedVarDevicePtr(int deviceId) {
  return statCO_.initStatManagedVarDevicePtr(deviceId);
}

void PlatformState::setupArgument(const void* arg, size_t size, size_t offset) {
  auto& arguments = hip::tls.exec_stack_.top().arguments_;

  if (arguments.size() < offset + size) {
    arguments.resize(offset + size);
  }

  ::memcpy(&arguments[offset], arg, size);
}

void PlatformState::configureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                  hipStream_t stream) {
  hip::tls.exec_stack_.push(ihipExec_t{gridDim, blockDim, sharedMem, stream});
}

void PlatformState::popExec(ihipExec_t& exec) {
  exec = std::move(hip::tls.exec_stack_.top());
  hip::tls.exec_stack_.pop();
}

std::shared_ptr<UniqueFD> PlatformState::GetUniqueFileHandle(const std::string& file_path) {
  amd::ScopedLock lock(ufd_lock_);

  if (ufd_map_.cend() == ufd_map_.find(file_path)) {
    // Get the file desc and file size from amd::Os API
    amd::Os::FileDesc fdesc;
    size_t fsize = 0;
    if (!amd::Os::GetFileHandle(file_path.c_str(), &fdesc, &fsize)) {
      return nullptr;
    }
    ufd_map_.insert(std::make_pair(file_path, std::make_shared<UniqueFD>(file_path, fdesc, fsize)));
  }

  // we should have an entry at this time.
  return ufd_map_[file_path];
}

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

void* PlatformState::getDynamicLibraryHandle() {
  amd::ScopedLock lock(lock_);

  if (dynamicLibraryHandle_ != nullptr) {
    return dynamicLibraryHandle_;
  }

#ifdef _WIN32
  const char* libName = "amdhip64.dll";
#else
  const char* libName = "libamdhip64.so";
#endif

  dynamicLibraryHandle_ = amd::Os::loadLibrary(libName);
  return dynamicLibraryHandle_;
}

void PlatformState::setDynamicLibraryHandle(void* handle) {
  amd::ScopedLock lock(lock_);
  dynamicLibraryHandle_ = handle;
}

}  // namespace hip
