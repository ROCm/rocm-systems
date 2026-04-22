////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2020-2026, Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "impl/wddm/status.h"
#include "impl/wddm/types.h"
#include "dxcore_loader.h"

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
  case STATUS_INVALID_PARAMETER:
    return ErrorCode::InvalidateParams;
  case STATUS_INVALID_HANDLE:
    return ErrorCode::InvalidHandle;
  default:
    break;
  }
  return ErrorCode::Unknown;
}

namespace d3dthunk {

typedef D3DKMT_CREATEALLOCATION                      CreateAllocationArgs;
typedef D3DKMT_CREATECONTEXT                         CreateContextArgs;
typedef D3DKMT_CREATECONTEXTVIRTUAL                  CreateContextVirtualArgs;
typedef D3DKMT_CREATEPAGINGQUEUE                     CreatePagingQueueArgs;
typedef D3DKMT_CREATESYNCHRONIZATIONOBJECT           CreateSynchronizationObjectArgs;
typedef D3DKMT_CREATESYNCHRONIZATIONOBJECT2          CreateSynchronizationObject2Args;
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
  args.hDevice = device;
  if (resource) {
    args.hResource = resource;
  } else {
    args.phAllocationList = alloc_handles;
    args.AllocationCount = num_allocations;
  }
  // Avoid stalls in VidMM since runtime is responsible for ensuring memory is not in use
  // Note: Otherwise OS will wait on the fence, but fence can't reflect HW state in AQL path
  args.Flags.AssumeNotInUse = 1;
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
inline ErrorCode QueryResourceInfo(D3DKMT_QUERYRESOURCEINFO *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTQueryResourceInfo(args)));
}

inline ErrorCode OpenResource(D3DKMT_OPENRESOURCE *args) {
  return TranslateNtStatus(DXCORE_CALL(D3DKMTOpenResource(args)));
}
} // namespace d3dthunk
} // namespace thunk
} // namespace wsl

