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
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include "amd_smi/impl/wsl/wddm_dxg_types.h"

#define DXCORE_CALL(function_name) wsl::thunk::dxcore::DxcoreLoader::Instance().pfn_##function_name

namespace wsl {
namespace thunk {
namespace dxcore {

/**
 * @brief DxcoreLoader class for dynamic loading of libdxcore.so
 *
 * Singleton loader for the DXCore library. Initialize() unconditionally
 * dlopen()s libdxcore.so; there is no env-var gate and no stub fallback if
 * loading fails (callers must check IsLoaded()/the Initialize() return value).
 * Not thread-safe: Initialize()/Shutdown() must be called from a single
 * thread before concurrent use of Instance().
 *
 * NOTE: libwkmi.a (built from rocr-runtime's copy of this same header, see
 * AMD-ROCm-Internal/wkmi's CMakeLists.txt) inlines calls against this class's
 * member layout and links against its constructor symbol. Both TUs must agree
 * on this header's layout, or expect corrupted data, not a link error. Rerun
 * the WSL functional test suite (WslFunctionalReadOnly.* on real WSL hardware)
 * whenever libwkmi.a is updated to catch this.
 */

// Macro definitions mimicking HSAKMT design
#define DXCORE_DEF(function_name) PFN##function_name
#define DXCORE_PFN(function_name) pfn_##function_name

class DxcoreLoader {
 public:
  // D3DKMT function type definitions
  typedef NTSTATUS(DXCORE_DEF(D3DKMTCreateAllocation2))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTDestroyAllocation2))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTMapGpuVirtualAddress))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTReserveGpuVirtualAddress))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTFreeGpuVirtualAddress))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTCreateDevice))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTDestroyDevice))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTEnumAdapters2))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTCloseAdapter))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTQueryAdapterInfo))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTCreateContextVirtual))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTDestroyContext))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTSubmitCommand))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTCreateSynchronizationObject2))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTDestroySynchronizationObject))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTQueryStatistics))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTEscape))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTLock2))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTUnlock2))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTCreatePagingQueue))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTDestroyPagingQueue))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTWaitForSynchronizationObjectFromGpu))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTSignalSynchronizationObjectFromGpu))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTWaitForSynchronizationObjectFromCpu))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTQueryClockCalibration))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTMakeResident))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTEvict))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTShareObjects))(size_t num_allocations,
                                                   WinResourceHandle* resource,
                                                   OBJECT_ATTRIBUTES* obj_attr, uint32_t flags,
                                                   void** nt_handle);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTQueryResourceInfoFromNtHandle))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTOpenResourceFromNtHandle))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTOpenSyncObjectFromNtHandle2))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTCreateHwQueue))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTDestroyHwQueue))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTSubmitCommandToHwQueue))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTEnumAdapters3))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTQueryResourceInfo))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTOpenResource))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTEnumProcesses))(void* args);
  typedef NTSTATUS(DXCORE_DEF(D3DKMTQueryVideoMemoryInfo))(void* args);

  static DxcoreLoader& Instance() {
    static DxcoreLoader* instance = new DxcoreLoader();
    return (*instance);
  }

  bool Initialize();
  void Shutdown();
  bool IsLoaded() const { return dxcore_handle_ != nullptr; }

  // Function pointer declarations
  DXCORE_DEF(D3DKMTCreateAllocation2) * DXCORE_PFN(D3DKMTCreateAllocation2);
  DXCORE_DEF(D3DKMTDestroyAllocation2) * DXCORE_PFN(D3DKMTDestroyAllocation2);
  DXCORE_DEF(D3DKMTMapGpuVirtualAddress) * DXCORE_PFN(D3DKMTMapGpuVirtualAddress);
  DXCORE_DEF(D3DKMTReserveGpuVirtualAddress) * DXCORE_PFN(D3DKMTReserveGpuVirtualAddress);
  DXCORE_DEF(D3DKMTFreeGpuVirtualAddress) * DXCORE_PFN(D3DKMTFreeGpuVirtualAddress);
  DXCORE_DEF(D3DKMTCreateDevice) * DXCORE_PFN(D3DKMTCreateDevice);
  DXCORE_DEF(D3DKMTDestroyDevice) * DXCORE_PFN(D3DKMTDestroyDevice);
  DXCORE_DEF(D3DKMTEnumAdapters2) * DXCORE_PFN(D3DKMTEnumAdapters2);
  DXCORE_DEF(D3DKMTCloseAdapter) * DXCORE_PFN(D3DKMTCloseAdapter);
  DXCORE_DEF(D3DKMTQueryAdapterInfo) * DXCORE_PFN(D3DKMTQueryAdapterInfo);
  DXCORE_DEF(D3DKMTCreateContextVirtual) * DXCORE_PFN(D3DKMTCreateContextVirtual);
  DXCORE_DEF(D3DKMTDestroyContext) * DXCORE_PFN(D3DKMTDestroyContext);
  DXCORE_DEF(D3DKMTSubmitCommand) * DXCORE_PFN(D3DKMTSubmitCommand);
  DXCORE_DEF(D3DKMTCreateSynchronizationObject2) * DXCORE_PFN(D3DKMTCreateSynchronizationObject2);
  DXCORE_DEF(D3DKMTDestroySynchronizationObject) * DXCORE_PFN(D3DKMTDestroySynchronizationObject);
  DXCORE_DEF(D3DKMTQueryStatistics) * DXCORE_PFN(D3DKMTQueryStatistics);
  DXCORE_DEF(D3DKMTEscape) * DXCORE_PFN(D3DKMTEscape);
  DXCORE_DEF(D3DKMTLock2) * DXCORE_PFN(D3DKMTLock2);
  DXCORE_DEF(D3DKMTUnlock2) * DXCORE_PFN(D3DKMTUnlock2);
  DXCORE_DEF(D3DKMTCreatePagingQueue) * DXCORE_PFN(D3DKMTCreatePagingQueue);
  DXCORE_DEF(D3DKMTDestroyPagingQueue) * DXCORE_PFN(D3DKMTDestroyPagingQueue);
  DXCORE_DEF(D3DKMTWaitForSynchronizationObjectFromGpu) *
      DXCORE_PFN(D3DKMTWaitForSynchronizationObjectFromGpu);
  DXCORE_DEF(D3DKMTSignalSynchronizationObjectFromGpu) *
      DXCORE_PFN(D3DKMTSignalSynchronizationObjectFromGpu);
  DXCORE_DEF(D3DKMTWaitForSynchronizationObjectFromCpu) *
      DXCORE_PFN(D3DKMTWaitForSynchronizationObjectFromCpu);
  DXCORE_DEF(D3DKMTQueryClockCalibration) * DXCORE_PFN(D3DKMTQueryClockCalibration);
  DXCORE_DEF(D3DKMTMakeResident) * DXCORE_PFN(D3DKMTMakeResident);
  DXCORE_DEF(D3DKMTEvict) * DXCORE_PFN(D3DKMTEvict);
  DXCORE_DEF(D3DKMTShareObjects) * DXCORE_PFN(D3DKMTShareObjects);
  DXCORE_DEF(D3DKMTQueryResourceInfoFromNtHandle) * DXCORE_PFN(D3DKMTQueryResourceInfoFromNtHandle);
  DXCORE_DEF(D3DKMTOpenResourceFromNtHandle) * DXCORE_PFN(D3DKMTOpenResourceFromNtHandle);
  DXCORE_DEF(D3DKMTOpenSyncObjectFromNtHandle2) * DXCORE_PFN(D3DKMTOpenSyncObjectFromNtHandle2);
  DXCORE_DEF(D3DKMTCreateHwQueue) * DXCORE_PFN(D3DKMTCreateHwQueue);
  DXCORE_DEF(D3DKMTDestroyHwQueue) * DXCORE_PFN(D3DKMTDestroyHwQueue);
  DXCORE_DEF(D3DKMTSubmitCommandToHwQueue) * DXCORE_PFN(D3DKMTSubmitCommandToHwQueue);
  DXCORE_DEF(D3DKMTEnumAdapters3) * DXCORE_PFN(D3DKMTEnumAdapters3);
  DXCORE_DEF(D3DKMTQueryResourceInfo) * DXCORE_PFN(D3DKMTQueryResourceInfo);
  DXCORE_DEF(D3DKMTOpenResource) * DXCORE_PFN(D3DKMTOpenResource);
  DXCORE_DEF(D3DKMTEnumProcesses) * DXCORE_PFN(D3DKMTEnumProcesses);
  DXCORE_DEF(D3DKMTQueryVideoMemoryInfo) * DXCORE_PFN(D3DKMTQueryVideoMemoryInfo);

 private:
  DxcoreLoader();
  ~DxcoreLoader();

  bool LoadDxcoreApis();

  void* dxcore_handle_;

  // Disable copy
  DxcoreLoader(const DxcoreLoader&) = delete;
  DxcoreLoader& operator=(const DxcoreLoader&) = delete;
};

}  // namespace dxcore
}  // namespace thunk
}  // namespace wsl
