/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 */

#include "amd_smi/impl/wsl/dxcore_loader.h"

#include <dlfcn.h>
#include <ntstatus.h>

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string_view>

#include "rocm_smi/rocm_smi_logger.h"

namespace wsl {
namespace thunk {
namespace dxcore {

DxcoreLoader::DxcoreLoader()
    : pfn_D3DKMTCreateAllocation2(nullptr),
      pfn_D3DKMTDestroyAllocation2(nullptr),
      pfn_D3DKMTMapGpuVirtualAddress(nullptr),
      pfn_D3DKMTReserveGpuVirtualAddress(nullptr),
      pfn_D3DKMTFreeGpuVirtualAddress(nullptr),
      pfn_D3DKMTCreateDevice(nullptr),
      pfn_D3DKMTDestroyDevice(nullptr),
      pfn_D3DKMTEnumAdapters2(nullptr),
      pfn_D3DKMTQueryAdapterInfo(nullptr),
      pfn_D3DKMTCreateContextVirtual(nullptr),
      pfn_D3DKMTDestroyContext(nullptr),
      pfn_D3DKMTSubmitCommand(nullptr),
      pfn_D3DKMTCreateSynchronizationObject2(nullptr),
      pfn_D3DKMTDestroySynchronizationObject(nullptr),
      pfn_D3DKMTQueryStatistics(nullptr),
      pfn_D3DKMTEscape(nullptr),
      pfn_D3DKMTLock2(nullptr),
      pfn_D3DKMTUnlock2(nullptr),
      pfn_D3DKMTCreatePagingQueue(nullptr),
      pfn_D3DKMTDestroyPagingQueue(nullptr),
      pfn_D3DKMTWaitForSynchronizationObjectFromGpu(nullptr),
      pfn_D3DKMTSignalSynchronizationObjectFromGpu(nullptr),
      pfn_D3DKMTWaitForSynchronizationObjectFromCpu(nullptr),
      pfn_D3DKMTQueryClockCalibration(nullptr),
      pfn_D3DKMTMakeResident(nullptr),
      pfn_D3DKMTEvict(nullptr),
      pfn_D3DKMTShareObjects(nullptr),
      pfn_D3DKMTQueryResourceInfoFromNtHandle(nullptr),
      pfn_D3DKMTOpenResourceFromNtHandle(nullptr),
      pfn_D3DKMTOpenSyncObjectFromNtHandle2(nullptr),
      pfn_D3DKMTCreateHwQueue(nullptr),
      pfn_D3DKMTDestroyHwQueue(nullptr),
      pfn_D3DKMTSubmitCommandToHwQueue(nullptr),
      pfn_D3DKMTEnumAdapters3(nullptr),
      pfn_D3DKMTQueryResourceInfo(nullptr),
      pfn_D3DKMTOpenResource(nullptr),
      pfn_D3DKMTEnumProcesses(nullptr),
      pfn_D3DKMTQueryVideoMemoryInfo(nullptr),
      pfn_D3DKMTCloseAdapter(nullptr),
      dxcore_handle_(nullptr) {}

DxcoreLoader::~DxcoreLoader() { Shutdown(); }

bool DxcoreLoader::Initialize() {
  dlerror();  // Clear error
#if defined(__linux__)
  constexpr std::string_view dxcore_lib_name = "libdxcore.so";
#else
  constexpr std::string_view dxcore_lib_name = "Gdi32.dll";
#endif
  dxcore_handle_ = dlopen(dxcore_lib_name.data(), RTLD_NOW);
  if (!dxcore_handle_) {
    std::ostringstream ss;
    ss << "[DxcoreLoader] Cannot load libdxcore.so: " << dlerror();
    LOG_ERROR(ss);
    return false;
  }

  {
    std::ostringstream ss;
    ss << "[DxcoreLoader] libdxcore.so loaded successfully";
    LOG_INFO(ss);
  }
  if (!LoadDxcoreApis()) {
    // If API loading failed, close the handle to indicate failure
    dlclose(dxcore_handle_);
    dxcore_handle_ = nullptr;
    return false;
  }

  return IsLoaded();
}

void DxcoreLoader::Shutdown() {
  if (dxcore_handle_) {
    // dlclose() returns 0 on success and nonzero on failure -- the
    // opposite convention of the platform-agnostic CloseLib() helper
    // this was ported from, so the branches are inverted here to keep
    // the same success/failure meaning.
    if (dlclose(dxcore_handle_) != 0) {
      std::ostringstream ss;
      ss << "[DxcoreLoader] Cannot unload libdxcore.so: " << dlerror();
      LOG_ERROR(ss);
    } else {
      std::ostringstream ss;
      ss << "[DxcoreLoader] libdxcore.so unloaded successfully";
      LOG_INFO(ss);
    }
    dxcore_handle_ = nullptr;
  }
}

bool DxcoreLoader::LoadDxcoreApis() {
  if (!dxcore_handle_) {
    std::ostringstream ss;
    ss << "[DxcoreLoader] Error: dxcore_handle_ is null";
    LOG_ERROR(ss);
    return false;
  }

  dlerror();  // Clear error

// Load all D3DKMT functions
#define LOAD_DXCORE_API(func_name)                                                   \
  DXCORE_PFN(func_name) = (DXCORE_DEF(func_name)*)dlsym(dxcore_handle_, #func_name); \
  if (!DXCORE_PFN(func_name)) {                                                      \
    std::ostringstream ss;                                                           \
    ss << "[DxcoreLoader] Failed to load " #func_name ": " << dlerror();             \
    LOG_ERROR(ss);                                                                   \
    goto ERROR_LOAD;                                                                 \
  }

  LOAD_DXCORE_API(D3DKMTCreateAllocation2);
  LOAD_DXCORE_API(D3DKMTDestroyAllocation2);
  LOAD_DXCORE_API(D3DKMTMapGpuVirtualAddress);
  LOAD_DXCORE_API(D3DKMTReserveGpuVirtualAddress);
  LOAD_DXCORE_API(D3DKMTFreeGpuVirtualAddress);
  LOAD_DXCORE_API(D3DKMTCreateDevice);
  LOAD_DXCORE_API(D3DKMTDestroyDevice);
  LOAD_DXCORE_API(D3DKMTEnumAdapters2);
  LOAD_DXCORE_API(D3DKMTCloseAdapter);
  LOAD_DXCORE_API(D3DKMTEnumAdapters3);
  LOAD_DXCORE_API(D3DKMTQueryAdapterInfo);
  LOAD_DXCORE_API(D3DKMTCreateContextVirtual);
  LOAD_DXCORE_API(D3DKMTDestroyContext);
  LOAD_DXCORE_API(D3DKMTSubmitCommand);
  LOAD_DXCORE_API(D3DKMTCreateSynchronizationObject2);
  LOAD_DXCORE_API(D3DKMTDestroySynchronizationObject);
  LOAD_DXCORE_API(D3DKMTQueryStatistics);
  LOAD_DXCORE_API(D3DKMTEscape);
  LOAD_DXCORE_API(D3DKMTLock2);
  LOAD_DXCORE_API(D3DKMTUnlock2);
  LOAD_DXCORE_API(D3DKMTCreatePagingQueue);
  LOAD_DXCORE_API(D3DKMTDestroyPagingQueue);
  LOAD_DXCORE_API(D3DKMTWaitForSynchronizationObjectFromGpu);
  LOAD_DXCORE_API(D3DKMTSignalSynchronizationObjectFromGpu);
  LOAD_DXCORE_API(D3DKMTWaitForSynchronizationObjectFromCpu);
  LOAD_DXCORE_API(D3DKMTQueryClockCalibration);
  LOAD_DXCORE_API(D3DKMTMakeResident);
  LOAD_DXCORE_API(D3DKMTEvict);
  LOAD_DXCORE_API(D3DKMTShareObjects);
  LOAD_DXCORE_API(D3DKMTQueryResourceInfoFromNtHandle);
  LOAD_DXCORE_API(D3DKMTQueryResourceInfo);
  LOAD_DXCORE_API(D3DKMTOpenResourceFromNtHandle);
  LOAD_DXCORE_API(D3DKMTOpenSyncObjectFromNtHandle2);
  LOAD_DXCORE_API(D3DKMTOpenResource);
  LOAD_DXCORE_API(D3DKMTCreateHwQueue);
  LOAD_DXCORE_API(D3DKMTDestroyHwQueue);
  LOAD_DXCORE_API(D3DKMTSubmitCommandToHwQueue);

#undef LOAD_DXCORE_API

  // Optional WSL2 dxgkrnl exports. Older libdxcore builds may not expose
  // these, so callers must probe the function pointer before use.
  DXCORE_PFN(D3DKMTEnumProcesses) =
      (DXCORE_DEF(D3DKMTEnumProcesses)*)dlsym(dxcore_handle_, "D3DKMTEnumProcesses");
  DXCORE_PFN(D3DKMTQueryVideoMemoryInfo) =
      (DXCORE_DEF(D3DKMTQueryVideoMemoryInfo)*)dlsym(dxcore_handle_, "D3DKMTQueryVideoMemoryInfo");

  {
    std::ostringstream ss;
    ss << "[DxcoreLoader] All DXCore APIs loaded successfully";
    LOG_INFO(ss);
  }
  return true;
ERROR_LOAD: {
  std::ostringstream ss;
  ss << "[DxcoreLoader] Failed to load DXCore APIs";
  LOG_ERROR(ss);
}
  return false;
}

}  // namespace dxcore
}  // namespace thunk
}  // namespace wsl
