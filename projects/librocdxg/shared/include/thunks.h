////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2020, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef SHARED_THUNKS_H
#define SHARED_THUNKS_H

#include "shared/include/status.h"
#include "shared/include/d3dkmt_types.h"
#include "shared/include/dxcore_loader.h"
#include "shared/include/lda_chain.h"

namespace wsl {
namespace thunk {

inline ErrorCode TranslateNtStatus(NTSTATUS status) {
  switch (status) {
  case STATUS_SUCCESS:
    return ErrorCode::Success;
  case STATUS_PENDING:
    return ErrorCode::NotReady;
  case STATUS_NO_MEMORY:
    return ErrorCode::OutOfMemory;
  case STATUS_DEVICE_REMOVED:
    return ErrorCode::DeviceLost;
  case STATUS_GRAPHICS_NO_VIDEO_MEMORY:
    return ErrorCode::OutOfGpuMemory;
  case STATUS_TIMEOUT:
    return ErrorCode::Timeout;
  case STATUS_BUFFER_TOO_SMALL:
    return ErrorCode::BufferTooSmall;
  case STATUS_INVALID_PARAMETER:
    return ErrorCode::InvalidParams;
  default:
    return ErrorCode::Unknown;
  }
}

namespace d3dthunk {

typedef D3DKMT_CREATEALLOCATION                      CreateAllocationArgs;
typedef D3DKMT_CREATECONTEXT                         CreateContextArgs;
typedef D3DKMT_CREATECONTEXTVIRTUAL                  CreateContextVirtualArgs;
typedef D3DKMT_CREATEPAGINGQUEUE                     CreatePagingQueueArgs;
typedef D3DKMT_CREATESYNCHRONIZATIONOBJECT           CreateSynchronizationObjectArgs;
typedef D3DKMT_CREATESYNCHRONIZATIONOBJECT2          CreateSynchronizationObject2Args;
typedef D3DKMT_CREATEDEVICE                          CreateDeviceArgs;
typedef D3DKMT_ESCAPE                                EscapeArgs;
typedef D3DKMT_EVICT                                 EvictArgs;
typedef D3DKMT_FREEGPUVIRTUALADDRESS                 FreeGpuVirtualAddressArgs;
typedef D3DKMT_LOCK                                  LockArgs;
typedef D3DKMT_LOCK2                                 Lock2Args;
typedef D3DKMT_OPENRESOURCE                          OpenResourceArgs;
typedef D3DKMT_OPENRESOURCEFROMNTHANDLE              OpenResourceFromNtHandleArgs;
typedef D3DKMT_QUERYADAPTERINFO                      QueryAdapterInfoArgs;
typedef D3DKMT_SIGNALSYNCHRONIZATIONOBJECT           SignalSynchronizationObjectArgs;
typedef D3DKMT_SIGNALSYNCHRONIZATIONOBJECT2          SignalSynchronizationObject2Args;
typedef D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMCPU    SignalSynchronizationObjectFromCpuArgs;
typedef D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU2   SignalSynchronizationObjectFromGpuArgs;
typedef D3DKMT_SUBMITCOMMAND                         SubmitCommandArgs;
typedef D3DKMT_UNLOCK                                UnlockArgs;
typedef D3DKMT_UNLOCK2                               Unlock2Args;
typedef D3DKMT_UPDATEGPUVIRTUALADDRESS               UpdateGpuVirtualAddressArgs;
typedef D3DKMT_WAITFORSYNCHRONIZATIONOBJECT          WaitForSynchronizationObjectArgs;
typedef D3DKMT_WAITFORSYNCHRONIZATIONOBJECT2         WaitForSynchronizationObject2Args;
typedef D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU   WaitForSynchronizationObjectFromCpuArgs;
typedef D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU   WaitForSynchronizationObjectFromGpuArgs;
typedef D3DKMT_ACQUIREKEYEDMUTEX                     AcquireKeyedMutexArgs;
typedef D3DKMT_RELEASEKEYEDMUTEX                     ReleaseKeyedMutexArgs;
typedef D3DKMT_OPENKEYEDMUTEX                        OpenKeyedMutexArgs;
typedef D3DKMT_DESTROYKEYEDMUTEX                     DestroyKeyedMutexArgs;
typedef D3DKMT_QUERYVIDEOMEMORYINFO                  QueryVideoMemoryInfoArgs;
typedef D3DKMT_CREATEHWQUEUE                         CreateHwQueueArgs;
typedef D3DKMT_DESTROYHWQUEUE                        DestroyHwQueueArgs;
typedef D3DKMT_SUBMITCOMMANDTOHWQUEUE                SubmitCommandToHwQueueArgs;
typedef D3DKMT_SUBMITPRESENTTOHWQUEUE                SubmitPresentToHwQueueArgs;
typedef D3DKMT_SUBMITSIGNALSYNCOBJECTSTOHWQUEUE      SubmitSignalSyncObjectsToHwQueueArgs;
typedef D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE     SubmitWaitForSyncObjectsToHwQueueArgs;
typedef D3DKMT_CREATESYNCFILE                        CreateSyncFileArgs;

inline ErrorCode MapGpuVirtualAddress(D3DDDI_MAPGPUVIRTUALADDRESS *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTMapGpuVirtualAddress(args)));
}

inline ErrorCode CreateAllocation(CreateAllocationArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTCreateAllocation2(args)));
}

inline ErrorCode DestroyAllocation(
            WinDeviceHandle device,
            WinResourceHandle resource,
            size_t num_allocations,
            const WinAllocationHandle *alloc_handles) {

  D3DKMT_DESTROYALLOCATION2 args{};

  memset(&args, 0, sizeof(args));
  args.hDevice = device;
  if (resource) {
    args.hResource = resource;
  } else {
    args.phAllocationList = alloc_handles;
    args.AllocationCount = num_allocations;
  }

  return TranslateNtStatus(DXCORE_CALL(D3DKMTDestroyAllocation2(&args)));
}

inline ErrorCode ReserveGpuVirtualAddress(D3DDDI_RESERVEGPUVIRTUALADDRESS *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTReserveGpuVirtualAddress(args)));
}

inline ErrorCode ReserveGpuVirtualAddress(WinAdapterHandle handle,
                                          gpusize size,
                                          gpusize base_address,
                                          gpusize *out_addr) {
  D3DDDI_RESERVEGPUVIRTUALADDRESS args{};
  args.hPagingQueue = handle;
  args.Size = size;
  args.BaseAddress = base_address;

  auto code = ReserveGpuVirtualAddress(&args);
  if (code == ErrorCode::Success)
    *out_addr = args.VirtualAddress;
  return code;
}

inline ErrorCode ReserveGpuVirtualAddress(WinAdapterHandle handle,
                                          gpusize size,
                                          gpusize minimum_address,
                                          gpusize maximum_address,
                                          gpusize *out_addr) {
  D3DDDI_RESERVEGPUVIRTUALADDRESS args{};
  args.hPagingQueue = handle;
  args.Size = size;
  args.MinimumAddress = minimum_address;
  args.MaximumAddress = maximum_address;

  auto code = ReserveGpuVirtualAddress(&args);
  if (code == ErrorCode::Success)
    *out_addr = args.VirtualAddress;
  return code;
}

inline ErrorCode FreeGpuVirtualAddress(FreeGpuVirtualAddressArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTFreeGpuVirtualAddress(args)));
}

inline ErrorCode FreeGpuVirtualAddress(WinAdapterHandle handle,
                                       gpusize base_address,
                                       gpusize size) {
  FreeGpuVirtualAddressArgs args{};
  args.hAdapter = handle;
  args.Size = size;
  args.BaseAddress = base_address;
  return FreeGpuVirtualAddress(&args);
}

inline ErrorCode MakeResident(D3DDDI_MAKERESIDENT *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTMakeResident(args)));
}

inline ErrorCode Evict(EvictArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTEvict(args)));
}

inline ErrorCode ShareObjects(size_t num_allocations,
                               WinResourceHandle resource,
                               uint32_t flags,
                               int* dmabuf_fd) {
  OBJECT_ATTRIBUTES obj_attr;
  HANDLE nt_handle;
  ErrorCode ret;

  InitializeObjectAttributes(&obj_attr, nullptr, OBJ_INHERIT, nullptr, nullptr);
  ret = TranslateNtStatus(DXCORE_CALL(D3DKMTShareObjects(num_allocations,
        &resource, &obj_attr, flags, &nt_handle)));
  if (ret == ErrorCode::Success)
    *dmabuf_fd = *(reinterpret_cast<int*>(&nt_handle));
  else
    *dmabuf_fd = -1;

  return ret;
}

inline ErrorCode QueryResourceInfoFromNtHandle(D3DKMT_QUERYRESOURCEINFOFROMNTHANDLE *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTQueryResourceInfoFromNtHandle(args)));
}

inline ErrorCode OpenResourceFromNtHandle(D3DKMT_OPENRESOURCEFROMNTHANDLE *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTOpenResourceFromNtHandle(args)));
}

inline ErrorCode EnumAdapters(D3DKMT_ENUMADAPTERS2 *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTEnumAdapters2(args)));
}

inline ErrorCode QueryAdapterInfo(QueryAdapterInfoArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTQueryAdapterInfo(args)));
}

inline ErrorCode CreateDevice(const LdaChain &, CreateDeviceArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTCreateDevice(args)));
}

inline ErrorCode DestroyDevice(const LdaChain &chain) {
  D3DKMT_DESTROYDEVICE args{};
  args.hDevice = chain.DeviceHandle();
  return TranslateNtStatus(DXCORE_CALL(D3DKMTDestroyDevice(&args)));
}

inline ErrorCode Escape(const LdaChain &chain, EscapeArgs *args) {
  args->hAdapter = chain.AdapterHandle();
  args->hDevice  = chain.DeviceHandle();
  return TranslateNtStatus(DXCORE_CALL(D3DKMTEscape(args)));
}

inline ErrorCode Escape(WinAdapterHandle adapter, WinDeviceHandle device,
                        EscapeArgs *args) {
  args->hAdapter = adapter;
  args->hDevice  = device;
  return TranslateNtStatus(DXCORE_CALL(D3DKMTEscape(args)));
}

typedef D3DKMT_QUERYSTATISTICS QueryStatisticsArgs;
typedef D3DDDI_DESTROYPAGINGQUEUE DestroyPagingQueueArgs;
typedef D3DKMT_DESTROYCONTEXT DestroyContextArgs;
typedef D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU
    SignalSynchronizationObjectFromGpuV1Args;
typedef D3DKMT_QUERYCLOCKCALIBRATION QueryClockCalibrationArgs;

inline ErrorCode QueryStatistics(QueryStatisticsArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTQueryStatistics(args)));
}

inline ErrorCode CreatePagingQueue(CreatePagingQueueArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTCreatePagingQueue(args)));
}

inline ErrorCode DestroyPagingQueue(DestroyPagingQueueArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTDestroyPagingQueue(args)));
}

inline ErrorCode Lock2(Lock2Args *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTLock2(args)));
}

inline ErrorCode Unlock2(Unlock2Args *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTUnlock2(args)));
}

inline ErrorCode CreateContextVirtual(CreateContextVirtualArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTCreateContextVirtual(args)));
}

inline ErrorCode DestroyContext(DestroyContextArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTDestroyContext(args)));
}

inline ErrorCode WaitForSynchronizationObjectFromGpu(
    WaitForSynchronizationObjectFromGpuArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTWaitForSynchronizationObjectFromGpu(args)));
}

inline ErrorCode SignalSynchronizationObjectFromGpu(
    SignalSynchronizationObjectFromGpuV1Args *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTSignalSynchronizationObjectFromGpu(args)));
}

inline ErrorCode WaitForSynchronizationObjectFromCpu(
    WaitForSynchronizationObjectFromCpuArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTWaitForSynchronizationObjectFromCpu(args)));
}

inline ErrorCode CreateSynchronizationObject2(CreateSynchronizationObject2Args *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTCreateSynchronizationObject2(args)));
}

inline ErrorCode DestroySynchronizationObject(
    D3DKMT_DESTROYSYNCHRONIZATIONOBJECT *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTDestroySynchronizationObject(args)));
}

inline ErrorCode QueryClockCalibration(QueryClockCalibrationArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTQueryClockCalibration(args)));
}

inline ErrorCode SubmitCommand(SubmitCommandArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTSubmitCommand(args)));
}

inline ErrorCode CreateHwQueue(CreateHwQueueArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTCreateHwQueue(args)));
}

inline ErrorCode DestroyHwQueue(DestroyHwQueueArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTDestroyHwQueue(args)));
}

inline ErrorCode SubmitCommandToHwQueue(SubmitCommandToHwQueueArgs *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTSubmitCommandToHwQueue(args)));
}

/** Optional WSL2 dxgkrnl export; nullptr if missing. */
inline bool QueryVideoMemoryInfoAvailable() {
  return DXCORE_CALL(D3DKMTQueryVideoMemoryInfo) != nullptr;
}

inline ErrorCode QueryVideoMemoryInfo(void *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTQueryVideoMemoryInfo(args)));
}

inline bool EnumProcessesAvailable() {
  return DXCORE_CALL(D3DKMTEnumProcesses) != nullptr;
}

inline ErrorCode EnumProcesses(void *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTEnumProcesses(args)));
}

} // namespace d3dthunk
} // namespace thunk
} // namespace wsl

#endif
