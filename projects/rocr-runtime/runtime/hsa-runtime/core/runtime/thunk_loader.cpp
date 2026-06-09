////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2026, Advanced Micro Devices, Inc. All rights reserved.
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

#include "core/inc/thunk_loader.h"
#include "core/inc/runtime.h"

#include <core/util/os.h>
#include <iostream>
#if defined(__linux__)
#include <dlfcn.h>
#include <fcntl.h>
#endif

namespace rocr {
namespace core {

  std::string ThunkLoader::whoami() {
    is_dtif_ = is_dxg_ = false;
    if (core::Runtime::runtime_singleton_->flag().enable_dtif()) {
      is_dtif_ = true;
#if defined(_WIN32)
      return "dtif64a.dll";
#else
      return "libdtif.so";
#endif
    }

#if defined(__linux__)
    if (core::Runtime::runtime_singleton_->flag().enable_dxg_detection()) {
      int fd = open("/dev/dxg", O_RDWR);
      if (fd >= 0) {
        close(fd);
        is_dxg_ = true;
        return "librocdxg.so";
      }
    }
#else
    is_dxg_ = true;
#endif

    return "";
  }

  ThunkLoader::ThunkLoader()
    : thunk_handle(nullptr),
      library_name(whoami()),
      is_loaded_(false) {
    if (!library_name.empty()) {
      rocr::os::DlError();  // Clear any existing error messages
      thunk_handle = rocr::os::LoadLib(library_name.c_str());
      if (thunk_handle == nullptr) {
        fprintf(stderr, "Cannot load %s, failed:%s\n", library_name.c_str(), rocr::os::DlError());
      } else {
        debug_print("Load %s successully!\n", library_name.c_str());
      }
      is_loaded_ = true;
    }
  }

  ThunkLoader::~ThunkLoader() {
    if (IsSharedLibraryLoaded()
      && (thunk_handle != nullptr)) {
        if (!rocr::os::CloseLib(thunk_handle)) {
          fprintf(stderr, "Cannot unload %s, failed:%s\n", library_name.c_str(), rocr::os::DlError());
        } else {
          debug_print("Unload %s successully!\n", library_name.c_str());
        }
    }
  }

  void ThunkLoader::LoadThunkApiTable() {
    if (IsSharedLibraryLoaded()) {
      rocr::os::DlError(); // Clear any existing error messages

      HSAKMT_PFN(hsaKmtOpenKFD) = (HSAKMT_DEF(hsaKmtOpenKFD)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtOpenKFD");
      if (HSAKMT_PFN(hsaKmtOpenKFD) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtCloseKFD) = (HSAKMT_DEF(hsaKmtCloseKFD)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtCloseKFD");
      if (HSAKMT_PFN(hsaKmtCloseKFD) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtGetVersion) = (HSAKMT_DEF(hsaKmtGetVersion)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtGetVersion");
      if (HSAKMT_PFN(hsaKmtGetVersion) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtAcquireSystemProperties) = (HSAKMT_DEF(hsaKmtAcquireSystemProperties)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtAcquireSystemProperties");
      if (HSAKMT_PFN(hsaKmtAcquireSystemProperties) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtReleaseSystemProperties) = (HSAKMT_DEF(hsaKmtReleaseSystemProperties)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtReleaseSystemProperties");
      if (HSAKMT_PFN(hsaKmtReleaseSystemProperties) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtGetNodeProperties) = (HSAKMT_DEF(hsaKmtGetNodeProperties)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtGetNodeProperties");
      if (HSAKMT_PFN(hsaKmtGetNodeProperties) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtGetNodeMemoryProperties) = (HSAKMT_DEF(hsaKmtGetNodeMemoryProperties)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtGetNodeMemoryProperties");
      if (HSAKMT_PFN(hsaKmtGetNodeMemoryProperties) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtGetNodeCacheProperties) = (HSAKMT_DEF(hsaKmtGetNodeCacheProperties)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtGetNodeCacheProperties");
      if (HSAKMT_PFN(hsaKmtGetNodeCacheProperties) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtGetNodeIoLinkProperties) = (HSAKMT_DEF(hsaKmtGetNodeIoLinkProperties)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtGetNodeIoLinkProperties");
      if (HSAKMT_PFN(hsaKmtGetNodeIoLinkProperties) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtCreateEvent) = (HSAKMT_DEF(hsaKmtCreateEvent)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtCreateEvent");
      if (HSAKMT_PFN(hsaKmtCreateEvent) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtDestroyEvent) = (HSAKMT_DEF(hsaKmtDestroyEvent)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtDestroyEvent");
      if (HSAKMT_PFN(hsaKmtDestroyEvent) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtSetEvent) = (HSAKMT_DEF(hsaKmtSetEvent)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtSetEvent");
      if (HSAKMT_PFN(hsaKmtSetEvent) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtResetEvent) = (HSAKMT_DEF(hsaKmtResetEvent)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtResetEvent");
      if (HSAKMT_PFN(hsaKmtResetEvent) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtQueryEventState) = (HSAKMT_DEF(hsaKmtQueryEventState)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtQueryEventState");
      if (HSAKMT_PFN(hsaKmtQueryEventState) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtWaitOnEvent) = (HSAKMT_DEF(hsaKmtWaitOnEvent)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtWaitOnEvent");
      if (HSAKMT_PFN(hsaKmtWaitOnEvent) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtWaitOnMultipleEvents) = (HSAKMT_DEF(hsaKmtWaitOnMultipleEvents)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtWaitOnMultipleEvents");
      if (HSAKMT_PFN(hsaKmtWaitOnMultipleEvents) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtCreateQueue) = (HSAKMT_DEF(hsaKmtCreateQueue)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtCreateQueue");
      if (HSAKMT_PFN(hsaKmtCreateQueue) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtCreateQueueExt) = (HSAKMT_DEF(hsaKmtCreateQueueExt)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtCreateQueueExt");
      if (HSAKMT_PFN(hsaKmtCreateQueueExt) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtCreateQueueV2) = (HSAKMT_DEF(hsaKmtCreateQueueV2)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtCreateQueueV2");
      if (HSAKMT_PFN(hsaKmtCreateQueueV2) == NULL) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtUpdateQueue) = (HSAKMT_DEF(hsaKmtUpdateQueue)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtUpdateQueue");
      if (HSAKMT_PFN(hsaKmtUpdateQueue) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtDestroyQueue) = (HSAKMT_DEF(hsaKmtDestroyQueue)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtDestroyQueue");
      if (HSAKMT_PFN(hsaKmtDestroyQueue) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtSetQueueCUMask) = (HSAKMT_DEF(hsaKmtSetQueueCUMask)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtSetQueueCUMask");
      if (HSAKMT_PFN(hsaKmtSetQueueCUMask) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtSetMemoryPolicy) = (HSAKMT_DEF(hsaKmtSetMemoryPolicy)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtSetMemoryPolicy");
      if (HSAKMT_PFN(hsaKmtSetMemoryPolicy) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtAllocMemory) = (HSAKMT_DEF(hsaKmtAllocMemory)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtAllocMemory");
      if (HSAKMT_PFN(hsaKmtAllocMemory) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtAllocMemoryAlign) = (HSAKMT_DEF(hsaKmtAllocMemoryAlign)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtAllocMemoryAlign");
      if (HSAKMT_PFN(hsaKmtAllocMemoryAlign) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtFreeMemory) = (HSAKMT_DEF(hsaKmtFreeMemory)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtFreeMemory");
      if (HSAKMT_PFN(hsaKmtFreeMemory) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtAvailableMemory) = (HSAKMT_DEF(hsaKmtAvailableMemory)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtAvailableMemory");
      if (HSAKMT_PFN(hsaKmtAvailableMemory) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtRegisterMemory) = (HSAKMT_DEF(hsaKmtRegisterMemory)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtRegisterMemory");
      if (HSAKMT_PFN(hsaKmtRegisterMemory) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtRegisterMemoryToNodes) = (HSAKMT_DEF(hsaKmtRegisterMemoryToNodes)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtRegisterMemoryToNodes");
      if (HSAKMT_PFN(hsaKmtRegisterMemoryToNodes) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtRegisterMemoryWithFlags) = (HSAKMT_DEF(hsaKmtRegisterMemoryWithFlags)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtRegisterMemoryWithFlags");
      if (HSAKMT_PFN(hsaKmtRegisterMemoryWithFlags) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtRegisterGraphicsHandleToNodes) = (HSAKMT_DEF(hsaKmtRegisterGraphicsHandleToNodes)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtRegisterGraphicsHandleToNodes");
      if (HSAKMT_PFN(hsaKmtRegisterGraphicsHandleToNodes) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtRegisterGraphicsHandleToNodesExt) = (HSAKMT_DEF(hsaKmtRegisterGraphicsHandleToNodesExt)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtRegisterGraphicsHandleToNodesExt");
      if (HSAKMT_PFN(hsaKmtRegisterGraphicsHandleToNodesExt) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtShareMemory) = (HSAKMT_DEF(hsaKmtShareMemory)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtShareMemory");
      if (HSAKMT_PFN(hsaKmtShareMemory) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtRegisterSharedHandle) = (HSAKMT_DEF(hsaKmtRegisterSharedHandle)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtRegisterSharedHandle");
      if (HSAKMT_PFN(hsaKmtRegisterSharedHandle) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtRegisterSharedHandleToNodes) = (HSAKMT_DEF(hsaKmtRegisterSharedHandleToNodes)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtRegisterSharedHandleToNodes");
      if (HSAKMT_PFN(hsaKmtRegisterSharedHandleToNodes) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtProcessVMRead) = (HSAKMT_DEF(hsaKmtProcessVMRead)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtProcessVMRead");
      if (HSAKMT_PFN(hsaKmtProcessVMRead) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtProcessVMWrite) = (HSAKMT_DEF(hsaKmtProcessVMWrite)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtProcessVMWrite");
      if (HSAKMT_PFN(hsaKmtProcessVMWrite) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtDeregisterMemory) = (HSAKMT_DEF(hsaKmtDeregisterMemory)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtDeregisterMemory");
      if (HSAKMT_PFN(hsaKmtDeregisterMemory) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtMapMemoryToGPU) = (HSAKMT_DEF(hsaKmtMapMemoryToGPU)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtMapMemoryToGPU");
      if (HSAKMT_PFN(hsaKmtMapMemoryToGPU) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtMapMemoryToGPUNodes) = (HSAKMT_DEF(hsaKmtMapMemoryToGPUNodes)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtMapMemoryToGPUNodes");
      if (HSAKMT_PFN(hsaKmtMapMemoryToGPUNodes) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtUnmapMemoryToGPU) = (HSAKMT_DEF(hsaKmtUnmapMemoryToGPU)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtUnmapMemoryToGPU");
      if (HSAKMT_PFN(hsaKmtUnmapMemoryToGPU) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtDbgRegister) = (HSAKMT_DEF(hsaKmtDbgRegister)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtDbgRegister");
      if (HSAKMT_PFN(hsaKmtDbgRegister) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtDbgUnregister) = (HSAKMT_DEF(hsaKmtDbgUnregister)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtDbgUnregister");
      if (HSAKMT_PFN(hsaKmtDbgUnregister) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtDbgWavefrontControl) = (HSAKMT_DEF(hsaKmtDbgWavefrontControl)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtDbgWavefrontControl");
      if (HSAKMT_PFN(hsaKmtDbgWavefrontControl) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtDbgAddressWatch) = (HSAKMT_DEF(hsaKmtDbgAddressWatch)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtDbgAddressWatch");
      if (HSAKMT_PFN(hsaKmtDbgAddressWatch) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtDbgEnable) = (HSAKMT_DEF(hsaKmtDbgEnable)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtDbgEnable");
      if (HSAKMT_PFN(hsaKmtDbgEnable) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtDbgDisable) = (HSAKMT_DEF(hsaKmtDbgDisable)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtDbgDisable");
      if (HSAKMT_PFN(hsaKmtDbgDisable) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtDbgGetDeviceData) = (HSAKMT_DEF(hsaKmtDbgGetDeviceData)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtDbgGetDeviceData");
      if (HSAKMT_PFN(hsaKmtDbgGetDeviceData) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtDbgGetQueueData) = (HSAKMT_DEF(hsaKmtDbgGetQueueData)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtDbgGetQueueData");
      if (HSAKMT_PFN(hsaKmtDbgGetQueueData) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtGetClockCounters) = (HSAKMT_DEF(hsaKmtGetClockCounters)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtGetClockCounters");
      if (HSAKMT_PFN(hsaKmtGetClockCounters) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtPmcGetCounterProperties) = (HSAKMT_DEF(hsaKmtPmcGetCounterProperties)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtPmcGetCounterProperties");
      if (HSAKMT_PFN(hsaKmtPmcGetCounterProperties) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtPmcRegisterTrace) = (HSAKMT_DEF(hsaKmtPmcRegisterTrace)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtPmcRegisterTrace");
      if (HSAKMT_PFN(hsaKmtPmcRegisterTrace) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtPmcUnregisterTrace) = (HSAKMT_DEF(hsaKmtPmcUnregisterTrace)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtPmcUnregisterTrace");
      if (HSAKMT_PFN(hsaKmtPmcUnregisterTrace) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtPmcAcquireTraceAccess) = (HSAKMT_DEF(hsaKmtPmcAcquireTraceAccess)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtPmcAcquireTraceAccess");
      if (HSAKMT_PFN(hsaKmtPmcAcquireTraceAccess) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtPmcReleaseTraceAccess) = (HSAKMT_DEF(hsaKmtPmcReleaseTraceAccess)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtPmcReleaseTraceAccess");
      if (HSAKMT_PFN(hsaKmtPmcReleaseTraceAccess) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtPmcStartTrace) = (HSAKMT_DEF(hsaKmtPmcStartTrace)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtPmcStartTrace");
      if (HSAKMT_PFN(hsaKmtPmcStartTrace) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtPmcQueryTrace) = (HSAKMT_DEF(hsaKmtPmcQueryTrace)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtPmcQueryTrace");
      if (HSAKMT_PFN(hsaKmtPmcQueryTrace) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtPmcStopTrace) = (HSAKMT_DEF(hsaKmtPmcStopTrace)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtPmcStopTrace");
      if (HSAKMT_PFN(hsaKmtPmcStopTrace) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtMapGraphicHandle) = (HSAKMT_DEF(hsaKmtMapGraphicHandle)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtMapGraphicHandle");
      if (HSAKMT_PFN(hsaKmtMapGraphicHandle) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtUnmapGraphicHandle) = (HSAKMT_DEF(hsaKmtUnmapGraphicHandle)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtUnmapGraphicHandle");
      if (HSAKMT_PFN(hsaKmtUnmapGraphicHandle) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtSetTrapHandler) = (HSAKMT_DEF(hsaKmtSetTrapHandler)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtSetTrapHandler");
      if (HSAKMT_PFN(hsaKmtSetTrapHandler) == nullptr) goto LOAD_ERROR;

      // only resolved when libhsakmt exposes the RAS-poison opt-in.
      HSAKMT_PFN(hsaKmtSetSigbusDelay) = (HSAKMT_DEF(hsaKmtSetSigbusDelay)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtSetSigbusDelay");

      HSAKMT_PFN(hsaKmtGetTileConfig) = (HSAKMT_DEF(hsaKmtGetTileConfig)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtGetTileConfig");
      if (HSAKMT_PFN(hsaKmtGetTileConfig) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtQueryPointerInfo) = (HSAKMT_DEF(hsaKmtQueryPointerInfo)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtQueryPointerInfo");
      if (HSAKMT_PFN(hsaKmtQueryPointerInfo) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtSetMemoryUserData) = (HSAKMT_DEF(hsaKmtSetMemoryUserData)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtSetMemoryUserData");
      if (HSAKMT_PFN(hsaKmtSetMemoryUserData) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtGetQueueInfo) = (HSAKMT_DEF(hsaKmtGetQueueInfo)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtGetQueueInfo");
      if (HSAKMT_PFN(hsaKmtGetQueueInfo) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtAllocQueueGWS) = (HSAKMT_DEF(hsaKmtAllocQueueGWS)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtAllocQueueGWS");
      if (HSAKMT_PFN(hsaKmtAllocQueueGWS) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtRuntimeEnable) = (HSAKMT_DEF(hsaKmtRuntimeEnable)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtRuntimeEnable");
      if (HSAKMT_PFN(hsaKmtRuntimeEnable) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtRuntimeDisable) = (HSAKMT_DEF(hsaKmtRuntimeDisable)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtRuntimeDisable");
      if (HSAKMT_PFN(hsaKmtRuntimeDisable) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtCheckRuntimeDebugSupport) = (HSAKMT_DEF(hsaKmtCheckRuntimeDebugSupport)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtCheckRuntimeDebugSupport");
      if (HSAKMT_PFN(hsaKmtCheckRuntimeDebugSupport) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtGetRuntimeCapabilities) = (HSAKMT_DEF(hsaKmtGetRuntimeCapabilities)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtGetRuntimeCapabilities");
      if (HSAKMT_PFN(hsaKmtGetRuntimeCapabilities) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtDebugTrapIoctl) = (HSAKMT_DEF(hsaKmtDebugTrapIoctl)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtDebugTrapIoctl");
      if (HSAKMT_PFN(hsaKmtDebugTrapIoctl) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtSPMAcquire) = (HSAKMT_DEF(hsaKmtSPMAcquire)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtSPMAcquire");
      if (HSAKMT_PFN(hsaKmtSPMAcquire) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtSPMRelease) = (HSAKMT_DEF(hsaKmtSPMRelease)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtSPMRelease");
      if (HSAKMT_PFN(hsaKmtSPMRelease) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtSPMSetDestBuffer) = (HSAKMT_DEF(hsaKmtSPMSetDestBuffer)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtSPMSetDestBuffer");
      if (HSAKMT_PFN(hsaKmtSPMSetDestBuffer) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtSVMSetAttr) = (HSAKMT_DEF(hsaKmtSVMSetAttr)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtSVMSetAttr");
      if (HSAKMT_PFN(hsaKmtSVMSetAttr) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtSVMGetAttr) = (HSAKMT_DEF(hsaKmtSVMGetAttr)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtSVMGetAttr");
      if (HSAKMT_PFN(hsaKmtSVMGetAttr) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtSetXNACKMode) = (HSAKMT_DEF(hsaKmtSetXNACKMode)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtSetXNACKMode");
      if (HSAKMT_PFN(hsaKmtSetXNACKMode) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtGetXNACKMode) = (HSAKMT_DEF(hsaKmtGetXNACKMode)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtGetXNACKMode");
      if (HSAKMT_PFN(hsaKmtGetXNACKMode) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtOpenSMI) = (HSAKMT_DEF(hsaKmtOpenSMI)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtOpenSMI");
      if (HSAKMT_PFN(hsaKmtOpenSMI) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtExportDMABufHandle) = (HSAKMT_DEF(hsaKmtExportDMABufHandle)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtExportDMABufHandle");
      if (HSAKMT_PFN(hsaKmtExportDMABufHandle) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtWaitOnEvent_Ext) = (HSAKMT_DEF(hsaKmtWaitOnEvent_Ext)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtWaitOnEvent_Ext");
      if (HSAKMT_PFN(hsaKmtWaitOnEvent_Ext) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtWaitOnMultipleEvents_Ext) = (HSAKMT_DEF(hsaKmtWaitOnMultipleEvents_Ext)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtWaitOnMultipleEvents_Ext");
      if (HSAKMT_PFN(hsaKmtWaitOnMultipleEvents_Ext) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtReplaceAsanHeaderPage) = (HSAKMT_DEF(hsaKmtReplaceAsanHeaderPage)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtReplaceAsanHeaderPage");
      if (HSAKMT_PFN(hsaKmtReplaceAsanHeaderPage) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtReturnAsanHeaderPage) = (HSAKMT_DEF(hsaKmtReturnAsanHeaderPage)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtReturnAsanHeaderPage");
      if (HSAKMT_PFN(hsaKmtReturnAsanHeaderPage) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtGetAMDGPUDeviceHandle) = (HSAKMT_DEF(hsaKmtGetAMDGPUDeviceHandle)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtGetAMDGPUDeviceHandle");
      if (HSAKMT_PFN(hsaKmtGetAMDGPUDeviceHandle) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtPcSamplingQueryCapabilities) = (HSAKMT_DEF(hsaKmtPcSamplingQueryCapabilities)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtPcSamplingQueryCapabilities");
      if (HSAKMT_PFN(hsaKmtPcSamplingQueryCapabilities) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtPcSamplingCreate) = (HSAKMT_DEF(hsaKmtPcSamplingCreate)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtPcSamplingCreate");
      if (HSAKMT_PFN(hsaKmtPcSamplingCreate) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtPcSamplingDestroy) = (HSAKMT_DEF(hsaKmtPcSamplingDestroy)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtPcSamplingDestroy");
      if (HSAKMT_PFN(hsaKmtPcSamplingDestroy) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtPcSamplingStart) = (HSAKMT_DEF(hsaKmtPcSamplingStart)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtPcSamplingStart");
      if (HSAKMT_PFN(hsaKmtPcSamplingStart) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtPcSamplingStop) = (HSAKMT_DEF(hsaKmtPcSamplingStop)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtPcSamplingStop");
      if (HSAKMT_PFN(hsaKmtPcSamplingStop) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtPcSamplingSupport) = (HSAKMT_DEF(hsaKmtPcSamplingSupport)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtPcSamplingSupport");
      if (HSAKMT_PFN(hsaKmtPcSamplingSupport) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtModelEnabled) = (HSAKMT_DEF(hsaKmtModelEnabled)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtModelEnabled");
      if (HSAKMT_PFN(hsaKmtModelEnabled) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtQueueRingDoorbell) = (HSAKMT_DEF(hsaKmtQueueRingDoorbell)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtQueueRingDoorbell");
      if (HSAKMT_PFN(hsaKmtQueueRingDoorbell) == nullptr) goto LOAD_ERROR;

      DRM_PFN(amdgpu_device_initialize) = (DRM_DEF(amdgpu_device_initialize)*)rocr::os::GetExportAddress(thunk_handle, "amdgpu_device_initialize");
      if (DRM_PFN(amdgpu_device_initialize) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtAisReadWriteFile) = (HSAKMT_DEF(hsaKmtAisReadWriteFile)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtAisReadWriteFile");
      if (HSAKMT_PFN(hsaKmtAisReadWriteFile) == nullptr) goto LOAD_ERROR;

#if defined(_WIN32)
      HSAKMT_PFN(hsaKmtGetMemoryHandle) = (HSAKMT_DEF(hsaKmtGetMemoryHandle)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtGetMemoryHandle");
      if (HSAKMT_PFN(hsaKmtGetMemoryHandle) == nullptr) goto LOAD_ERROR;
#endif

      HSAKMT_PFN(hsaKmtHandleImport) = (HSAKMT_DEF(hsaKmtHandleImport)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtHandleImport");
      if (HSAKMT_PFN(hsaKmtHandleImport) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtImportExternalSemaphore) = (HSAKMT_DEF(hsaKmtImportExternalSemaphore)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtImportExternalSemaphore");
      if (HSAKMT_PFN(hsaKmtImportExternalSemaphore) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtDestroyExternalSemaphore) = (HSAKMT_DEF(hsaKmtDestroyExternalSemaphore)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtDestroyExternalSemaphore");
      if (HSAKMT_PFN(hsaKmtDestroyExternalSemaphore) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtMemoryVaMap) = (HSAKMT_DEF(hsaKmtMemoryVaMap)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtMemoryVaMap");
      if (HSAKMT_PFN(hsaKmtMemoryVaMap) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtMemoryVaUnmap) = (HSAKMT_DEF(hsaKmtMemoryVaUnmap)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtMemoryVaUnmap");
      if (HSAKMT_PFN(hsaKmtMemoryVaUnmap) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtMemHandleFree) = (HSAKMT_DEF(hsaKmtMemHandleFree)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtMemHandleFree");
      if (HSAKMT_PFN(hsaKmtMemHandleFree) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtMemoryGetCpuAddr) = (HSAKMT_DEF(hsaKmtMemoryGetCpuAddr)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtMemoryGetCpuAddr");
      if (HSAKMT_PFN(hsaKmtMemoryGetCpuAddr) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtMemoryCpuMap) = (HSAKMT_DEF(hsaKmtMemoryCpuMap)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtMemoryCpuMap");
      if (HSAKMT_PFN(hsaKmtMemoryCpuMap) == nullptr) goto LOAD_ERROR;

      HSAKMT_PFN(hsaKmtGetNodeWallclockFrequency) = (HSAKMT_DEF(hsaKmtGetNodeWallclockFrequency)*)rocr::os::GetExportAddress(thunk_handle, "hsaKmtGetNodeWallclockFrequency");
      if (HSAKMT_PFN(hsaKmtGetNodeWallclockFrequency) == nullptr) goto LOAD_ERROR;

      DRM_PFN(amdgpu_device_deinitialize) = (DRM_DEF(amdgpu_device_deinitialize)*)rocr::os::GetExportAddress(thunk_handle, "amdgpu_device_deinitialize");
      if (DRM_PFN(amdgpu_device_deinitialize) == nullptr) goto LOAD_ERROR;

      DRM_PFN(amdgpu_query_gpu_info) = (DRM_DEF(amdgpu_query_gpu_info)*)rocr::os::GetExportAddress(thunk_handle, "amdgpu_query_gpu_info");
      if (DRM_PFN(amdgpu_query_gpu_info) == nullptr) goto LOAD_ERROR;

      DRM_PFN(amdgpu_bo_cpu_map) = (DRM_DEF(amdgpu_bo_cpu_map)*)rocr::os::GetExportAddress(thunk_handle, "amdgpu_bo_cpu_map");
      if (DRM_PFN(amdgpu_bo_cpu_map) == nullptr) goto LOAD_ERROR;

      DRM_PFN(amdgpu_bo_free) = (DRM_DEF(amdgpu_bo_free)*)rocr::os::GetExportAddress(thunk_handle, "amdgpu_bo_free");
      if (DRM_PFN(amdgpu_bo_free) == nullptr) goto LOAD_ERROR;

      DRM_PFN(amdgpu_bo_export) = (DRM_DEF(amdgpu_bo_export)*)rocr::os::GetExportAddress(thunk_handle, "amdgpu_bo_export");
      if (DRM_PFN(amdgpu_bo_export) == nullptr) goto LOAD_ERROR;

      DRM_PFN(amdgpu_bo_import) = (DRM_DEF(amdgpu_bo_import)*)rocr::os::GetExportAddress(thunk_handle, "amdgpu_bo_import");
      if (DRM_PFN(amdgpu_bo_import) == nullptr) goto LOAD_ERROR;

      DRM_PFN(amdgpu_bo_va_op) = (DRM_DEF(amdgpu_bo_va_op)*)rocr::os::GetExportAddress(thunk_handle, "amdgpu_bo_va_op");
      if (DRM_PFN(amdgpu_bo_va_op) == nullptr) goto LOAD_ERROR;

      DRM_PFN(amdgpu_bo_query_info) = (DRM_DEF(amdgpu_bo_query_info)*)rocr::os::GetExportAddress(thunk_handle, "amdgpu_bo_query_info");
      if (DRM_PFN(amdgpu_bo_query_info) == NULL) goto LOAD_ERROR;

      DRM_PFN(amdgpu_bo_set_metadata) = (DRM_DEF(amdgpu_bo_set_metadata)*)rocr::os::GetExportAddress(thunk_handle, "amdgpu_bo_set_metadata");
      if (DRM_PFN(amdgpu_bo_set_metadata) == NULL) goto LOAD_ERROR;

      DRM_PFN(drmCommandWriteRead) = (DRM_DEF(drmCommandWriteRead)*)rocr::os::GetExportAddress(thunk_handle, "drmCommandWriteRead");
      if (DRM_PFN(drmCommandWriteRead) == nullptr) goto LOAD_ERROR;
      debug_print("Load all DTIF APIs OK!\n");
      return;

LOAD_ERROR:
      fprintf(stderr, "GetExportAddress failed: %s\n", rocr::os::DlError());
    } else {
      HSAKMT_PFN(hsaKmtOpenKFD) = (HSAKMT_DEF(hsaKmtOpenKFD)*)(&hsaKmtOpenKFD);
      HSAKMT_PFN(hsaKmtCloseKFD) = (HSAKMT_DEF(hsaKmtCloseKFD)*)(&hsaKmtCloseKFD);
      HSAKMT_PFN(hsaKmtGetVersion) = (HSAKMT_DEF(hsaKmtGetVersion)*)(&hsaKmtGetVersion);
      HSAKMT_PFN(hsaKmtAcquireSystemProperties) = (HSAKMT_DEF(hsaKmtAcquireSystemProperties)*)(&hsaKmtAcquireSystemProperties);
      HSAKMT_PFN(hsaKmtReleaseSystemProperties) = (HSAKMT_DEF(hsaKmtReleaseSystemProperties)*)(&hsaKmtReleaseSystemProperties);
      HSAKMT_PFN(hsaKmtGetNodeProperties) = (HSAKMT_DEF(hsaKmtGetNodeProperties)*)(&hsaKmtGetNodeProperties);
      HSAKMT_PFN(hsaKmtGetNodeMemoryProperties) = (HSAKMT_DEF(hsaKmtGetNodeMemoryProperties)*)(&hsaKmtGetNodeMemoryProperties);
      HSAKMT_PFN(hsaKmtGetNodeCacheProperties) = (HSAKMT_DEF(hsaKmtGetNodeCacheProperties)*)(&hsaKmtGetNodeCacheProperties);
      HSAKMT_PFN(hsaKmtGetNodeIoLinkProperties) = (HSAKMT_DEF(hsaKmtGetNodeIoLinkProperties)*)(&hsaKmtGetNodeIoLinkProperties);
      HSAKMT_PFN(hsaKmtCreateEvent) = (HSAKMT_DEF(hsaKmtCreateEvent)*)(&hsaKmtCreateEvent);
      HSAKMT_PFN(hsaKmtDestroyEvent) = (HSAKMT_DEF(hsaKmtDestroyEvent)*)(&hsaKmtDestroyEvent);
      HSAKMT_PFN(hsaKmtSetEvent) = (HSAKMT_DEF(hsaKmtSetEvent)*)(&hsaKmtSetEvent);
      HSAKMT_PFN(hsaKmtResetEvent) = (HSAKMT_DEF(hsaKmtResetEvent)*)(&hsaKmtResetEvent);
      HSAKMT_PFN(hsaKmtQueryEventState) = (HSAKMT_DEF(hsaKmtQueryEventState)*)(&hsaKmtQueryEventState);
      HSAKMT_PFN(hsaKmtWaitOnEvent) = (HSAKMT_DEF(hsaKmtWaitOnEvent)*)(&hsaKmtWaitOnEvent);
      HSAKMT_PFN(hsaKmtWaitOnMultipleEvents) = (HSAKMT_DEF(hsaKmtWaitOnMultipleEvents)*)(&hsaKmtWaitOnMultipleEvents);
      HSAKMT_PFN(hsaKmtCreateQueue) = (HSAKMT_DEF(hsaKmtCreateQueue)*)(&hsaKmtCreateQueue);
      HSAKMT_PFN(hsaKmtCreateQueueExt) = (HSAKMT_DEF(hsaKmtCreateQueueExt)*)(&hsaKmtCreateQueueExt);
      HSAKMT_PFN(hsaKmtCreateQueueV2) = (HSAKMT_DEF(hsaKmtCreateQueueV2)*)(&hsaKmtCreateQueueV2);
      HSAKMT_PFN(hsaKmtUpdateQueue) = (HSAKMT_DEF(hsaKmtUpdateQueue)*)(&hsaKmtUpdateQueue);
      HSAKMT_PFN(hsaKmtDestroyQueue) = (HSAKMT_DEF(hsaKmtDestroyQueue)*)(&hsaKmtDestroyQueue);
      HSAKMT_PFN(hsaKmtSetQueueCUMask) = (HSAKMT_DEF(hsaKmtSetQueueCUMask)*)(&hsaKmtSetQueueCUMask);
      HSAKMT_PFN(hsaKmtSetMemoryPolicy) = (HSAKMT_DEF(hsaKmtSetMemoryPolicy)*)(&hsaKmtSetMemoryPolicy);
      HSAKMT_PFN(hsaKmtAllocMemory) = (HSAKMT_DEF(hsaKmtAllocMemory)*)(&hsaKmtAllocMemory);
      HSAKMT_PFN(hsaKmtAllocMemoryAlign) = (HSAKMT_DEF(hsaKmtAllocMemoryAlign)*)(&hsaKmtAllocMemoryAlign);
      HSAKMT_PFN(hsaKmtFreeMemory) = (HSAKMT_DEF(hsaKmtFreeMemory)*)(&hsaKmtFreeMemory);
      HSAKMT_PFN(hsaKmtAvailableMemory) = (HSAKMT_DEF(hsaKmtAvailableMemory)*)(&hsaKmtAvailableMemory);
      HSAKMT_PFN(hsaKmtRegisterMemory) = (HSAKMT_DEF(hsaKmtRegisterMemory)*)(&hsaKmtRegisterMemory);
      HSAKMT_PFN(hsaKmtRegisterMemoryToNodes) = (HSAKMT_DEF(hsaKmtRegisterMemoryToNodes)*)(&hsaKmtRegisterMemoryToNodes);
      HSAKMT_PFN(hsaKmtRegisterMemoryWithFlags) = (HSAKMT_DEF(hsaKmtRegisterMemoryWithFlags)*)(&hsaKmtRegisterMemoryWithFlags);
      HSAKMT_PFN(hsaKmtRegisterGraphicsHandleToNodes) = (HSAKMT_DEF(hsaKmtRegisterGraphicsHandleToNodes)*)(&hsaKmtRegisterGraphicsHandleToNodes);
      HSAKMT_PFN(hsaKmtRegisterGraphicsHandleToNodesExt) = (HSAKMT_DEF(hsaKmtRegisterGraphicsHandleToNodesExt)*)(&hsaKmtRegisterGraphicsHandleToNodesExt);
      HSAKMT_PFN(hsaKmtShareMemory) = (HSAKMT_DEF(hsaKmtShareMemory)*)(&hsaKmtShareMemory);
      HSAKMT_PFN(hsaKmtRegisterSharedHandle) = (HSAKMT_DEF(hsaKmtRegisterSharedHandle)*)(&hsaKmtRegisterSharedHandle);
      HSAKMT_PFN(hsaKmtRegisterSharedHandleToNodes) = (HSAKMT_DEF(hsaKmtRegisterSharedHandleToNodes)*)(&hsaKmtRegisterSharedHandleToNodes);
      HSAKMT_PFN(hsaKmtProcessVMRead) = (HSAKMT_DEF(hsaKmtProcessVMRead)*)(&hsaKmtProcessVMRead);
      HSAKMT_PFN(hsaKmtProcessVMWrite) = (HSAKMT_DEF(hsaKmtProcessVMWrite)*)(&hsaKmtProcessVMWrite);
      HSAKMT_PFN(hsaKmtDeregisterMemory) = (HSAKMT_DEF(hsaKmtDeregisterMemory)*)(&hsaKmtDeregisterMemory);
      HSAKMT_PFN(hsaKmtMapMemoryToGPU) = (HSAKMT_DEF(hsaKmtMapMemoryToGPU)*)(&hsaKmtMapMemoryToGPU);
      HSAKMT_PFN(hsaKmtMapMemoryToGPUNodes) = (HSAKMT_DEF(hsaKmtMapMemoryToGPUNodes)*)(&hsaKmtMapMemoryToGPUNodes);
      HSAKMT_PFN(hsaKmtUnmapMemoryToGPU) = (HSAKMT_DEF(hsaKmtUnmapMemoryToGPU)*)(&hsaKmtUnmapMemoryToGPU);
      HSAKMT_PFN(hsaKmtDbgRegister) = (HSAKMT_DEF(hsaKmtDbgRegister)*)(&hsaKmtDbgRegister);
      HSAKMT_PFN(hsaKmtDbgUnregister) = (HSAKMT_DEF(hsaKmtDbgUnregister)*)(&hsaKmtDbgUnregister);
      HSAKMT_PFN(hsaKmtDbgWavefrontControl) = (HSAKMT_DEF(hsaKmtDbgWavefrontControl)*)(&hsaKmtDbgWavefrontControl);
      HSAKMT_PFN(hsaKmtDbgAddressWatch) = (HSAKMT_DEF(hsaKmtDbgAddressWatch)*)(&hsaKmtDbgAddressWatch);
      HSAKMT_PFN(hsaKmtDbgEnable) = (HSAKMT_DEF(hsaKmtDbgEnable)*)(&hsaKmtDbgEnable);
      HSAKMT_PFN(hsaKmtDbgDisable) = (HSAKMT_DEF(hsaKmtDbgDisable)*)(&hsaKmtDbgDisable);
      HSAKMT_PFN(hsaKmtDbgGetDeviceData) = (HSAKMT_DEF(hsaKmtDbgGetDeviceData)*)(&hsaKmtDbgGetDeviceData);
      HSAKMT_PFN(hsaKmtDbgGetQueueData) = (HSAKMT_DEF(hsaKmtDbgGetQueueData)*)(&hsaKmtDbgGetQueueData);
      HSAKMT_PFN(hsaKmtGetClockCounters) = (HSAKMT_DEF(hsaKmtGetClockCounters)*)(&hsaKmtGetClockCounters);
      HSAKMT_PFN(hsaKmtPmcGetCounterProperties) = (HSAKMT_DEF(hsaKmtPmcGetCounterProperties)*)(&hsaKmtPmcGetCounterProperties);
      HSAKMT_PFN(hsaKmtPmcRegisterTrace) = (HSAKMT_DEF(hsaKmtPmcRegisterTrace)*)(&hsaKmtPmcRegisterTrace);
      HSAKMT_PFN(hsaKmtPmcUnregisterTrace) = (HSAKMT_DEF(hsaKmtPmcUnregisterTrace)*)(&hsaKmtPmcUnregisterTrace);
      HSAKMT_PFN(hsaKmtPmcAcquireTraceAccess) = (HSAKMT_DEF(hsaKmtPmcAcquireTraceAccess)*)(&hsaKmtPmcAcquireTraceAccess);
      HSAKMT_PFN(hsaKmtPmcReleaseTraceAccess) = (HSAKMT_DEF(hsaKmtPmcReleaseTraceAccess)*)(&hsaKmtPmcReleaseTraceAccess);
      HSAKMT_PFN(hsaKmtPmcStartTrace) = (HSAKMT_DEF(hsaKmtPmcStartTrace)*)(&hsaKmtPmcStartTrace);
      HSAKMT_PFN(hsaKmtPmcQueryTrace) = (HSAKMT_DEF(hsaKmtPmcQueryTrace)*)(&hsaKmtPmcQueryTrace);
      HSAKMT_PFN(hsaKmtPmcStopTrace) = (HSAKMT_DEF(hsaKmtPmcStopTrace)*)(&hsaKmtPmcStopTrace);
      HSAKMT_PFN(hsaKmtMapGraphicHandle) = (HSAKMT_DEF(hsaKmtMapGraphicHandle)*)(&hsaKmtMapGraphicHandle);
      HSAKMT_PFN(hsaKmtUnmapGraphicHandle) = (HSAKMT_DEF(hsaKmtUnmapGraphicHandle)*)(&hsaKmtUnmapGraphicHandle);
      HSAKMT_PFN(hsaKmtSetTrapHandler) = (HSAKMT_DEF(hsaKmtSetTrapHandler)*)(&hsaKmtSetTrapHandler);
      HSAKMT_PFN(hsaKmtSetSigbusDelay) = (HSAKMT_DEF(hsaKmtSetSigbusDelay)*)(&hsaKmtSetSigbusDelay);
      HSAKMT_PFN(hsaKmtGetTileConfig) = (HSAKMT_DEF(hsaKmtGetTileConfig)*)(&hsaKmtGetTileConfig);
      HSAKMT_PFN(hsaKmtQueryPointerInfo) = (HSAKMT_DEF(hsaKmtQueryPointerInfo)*)(&hsaKmtQueryPointerInfo);
      HSAKMT_PFN(hsaKmtSetMemoryUserData) = (HSAKMT_DEF(hsaKmtSetMemoryUserData)*)(&hsaKmtSetMemoryUserData);
      HSAKMT_PFN(hsaKmtGetQueueInfo) = (HSAKMT_DEF(hsaKmtGetQueueInfo)*)(&hsaKmtGetQueueInfo);
      HSAKMT_PFN(hsaKmtAllocQueueGWS) = (HSAKMT_DEF(hsaKmtAllocQueueGWS)*)(&hsaKmtAllocQueueGWS);
      HSAKMT_PFN(hsaKmtRuntimeEnable) = (HSAKMT_DEF(hsaKmtRuntimeEnable)*)(&hsaKmtRuntimeEnable);
      HSAKMT_PFN(hsaKmtRuntimeDisable) = (HSAKMT_DEF(hsaKmtRuntimeDisable)*)(&hsaKmtRuntimeDisable);
      HSAKMT_PFN(hsaKmtCheckRuntimeDebugSupport) = (HSAKMT_DEF(hsaKmtCheckRuntimeDebugSupport)*)(&hsaKmtCheckRuntimeDebugSupport);
      HSAKMT_PFN(hsaKmtGetRuntimeCapabilities) = (HSAKMT_DEF(hsaKmtGetRuntimeCapabilities)*)(&hsaKmtGetRuntimeCapabilities);
      HSAKMT_PFN(hsaKmtDebugTrapIoctl) = (HSAKMT_DEF(hsaKmtDebugTrapIoctl)*)(&hsaKmtDebugTrapIoctl);
      HSAKMT_PFN(hsaKmtSPMAcquire) = (HSAKMT_DEF(hsaKmtSPMAcquire)*)(&hsaKmtSPMAcquire);
      HSAKMT_PFN(hsaKmtSPMRelease) = (HSAKMT_DEF(hsaKmtSPMRelease)*)(&hsaKmtSPMRelease);
      HSAKMT_PFN(hsaKmtSPMSetDestBuffer) = (HSAKMT_DEF(hsaKmtSPMSetDestBuffer)*)(&hsaKmtSPMSetDestBuffer);
      HSAKMT_PFN(hsaKmtSVMSetAttr) = (HSAKMT_DEF(hsaKmtSVMSetAttr)*)(&hsaKmtSVMSetAttr);
      HSAKMT_PFN(hsaKmtSVMGetAttr) = (HSAKMT_DEF(hsaKmtSVMGetAttr)*)(&hsaKmtSVMGetAttr);
      HSAKMT_PFN(hsaKmtSetXNACKMode) = (HSAKMT_DEF(hsaKmtSetXNACKMode)*)(&hsaKmtSetXNACKMode);
      HSAKMT_PFN(hsaKmtGetXNACKMode) = (HSAKMT_DEF(hsaKmtGetXNACKMode)*)(&hsaKmtGetXNACKMode);
      HSAKMT_PFN(hsaKmtOpenSMI) = (HSAKMT_DEF(hsaKmtOpenSMI)*)(&hsaKmtOpenSMI);
      HSAKMT_PFN(hsaKmtExportDMABufHandle) = (HSAKMT_DEF(hsaKmtExportDMABufHandle)*)(&hsaKmtExportDMABufHandle);
      HSAKMT_PFN(hsaKmtWaitOnEvent_Ext) = (HSAKMT_DEF(hsaKmtWaitOnEvent_Ext)*)(&hsaKmtWaitOnEvent_Ext);
      HSAKMT_PFN(hsaKmtWaitOnMultipleEvents_Ext) = (HSAKMT_DEF(hsaKmtWaitOnMultipleEvents_Ext)*)(&hsaKmtWaitOnMultipleEvents_Ext);
      HSAKMT_PFN(hsaKmtReplaceAsanHeaderPage) = (HSAKMT_DEF(hsaKmtReplaceAsanHeaderPage)*)(&hsaKmtReplaceAsanHeaderPage);
      HSAKMT_PFN(hsaKmtReturnAsanHeaderPage) = (HSAKMT_DEF(hsaKmtReturnAsanHeaderPage)*)(&hsaKmtReturnAsanHeaderPage);
      HSAKMT_PFN(hsaKmtGetAMDGPUDeviceHandle) = (HSAKMT_DEF(hsaKmtGetAMDGPUDeviceHandle)*)(&hsaKmtGetAMDGPUDeviceHandle);
      HSAKMT_PFN(hsaKmtPcSamplingQueryCapabilities) = (HSAKMT_DEF(hsaKmtPcSamplingQueryCapabilities)*)(&hsaKmtPcSamplingQueryCapabilities);
      HSAKMT_PFN(hsaKmtPcSamplingCreate) = (HSAKMT_DEF(hsaKmtPcSamplingCreate)*)(&hsaKmtPcSamplingCreate);
      HSAKMT_PFN(hsaKmtPcSamplingDestroy) = (HSAKMT_DEF(hsaKmtPcSamplingDestroy)*)(&hsaKmtPcSamplingDestroy);
      HSAKMT_PFN(hsaKmtPcSamplingStart) = (HSAKMT_DEF(hsaKmtPcSamplingStart)*)(&hsaKmtPcSamplingStart);
      HSAKMT_PFN(hsaKmtPcSamplingStop) = (HSAKMT_DEF(hsaKmtPcSamplingStop)*)(&hsaKmtPcSamplingStop);
      HSAKMT_PFN(hsaKmtPcSamplingSupport) = (HSAKMT_DEF(hsaKmtPcSamplingSupport)*)(&hsaKmtPcSamplingSupport);
#if defined(_WIN32)
      HSAKMT_PFN(hsaKmtQueueRingDoorbell) = (HSAKMT_DEF(hsaKmtQueueRingDoorbell)*)(&hsaKmtQueueRingDoorbell);
#endif
      HSAKMT_PFN(hsaKmtModelEnabled) = (HSAKMT_DEF(hsaKmtModelEnabled)*)(&hsaKmtModelEnabled);
      HSAKMT_PFN(hsaKmtAisReadWriteFile) = (HSAKMT_DEF(hsaKmtAisReadWriteFile)*)(&hsaKmtAisReadWriteFile);
#if defined(_WIN32)
      HSAKMT_PFN(hsaKmtGetMemoryHandle) = (HSAKMT_DEF(hsaKmtGetMemoryHandle)*)(&hsaKmtGetMemoryHandle);
#endif
      HSAKMT_PFN(hsaKmtHandleImport) = (HSAKMT_DEF(hsaKmtHandleImport)*)(&hsaKmtHandleImport);
      HSAKMT_PFN(hsaKmtImportExternalSemaphore) = (HSAKMT_DEF(hsaKmtImportExternalSemaphore)*)(&hsaKmtImportExternalSemaphore);
      HSAKMT_PFN(hsaKmtDestroyExternalSemaphore) = (HSAKMT_DEF(hsaKmtDestroyExternalSemaphore)*)(&hsaKmtDestroyExternalSemaphore);
      HSAKMT_PFN(hsaKmtMemoryVaMap) = (HSAKMT_DEF(hsaKmtMemoryVaMap)*)(&hsaKmtMemoryVaMap);
      HSAKMT_PFN(hsaKmtMemoryVaUnmap) = (HSAKMT_DEF(hsaKmtMemoryVaUnmap)*)(&hsaKmtMemoryVaUnmap);
      HSAKMT_PFN(hsaKmtMemHandleFree) = (HSAKMT_DEF(hsaKmtMemHandleFree)*)(&hsaKmtMemHandleFree);
      HSAKMT_PFN(hsaKmtMemoryGetCpuAddr) = (HSAKMT_DEF(hsaKmtMemoryGetCpuAddr)*)(&hsaKmtMemoryGetCpuAddr);
      HSAKMT_PFN(hsaKmtMemoryCpuMap) = (HSAKMT_DEF(hsaKmtMemoryCpuMap)*)(&hsaKmtMemoryCpuMap);
      HSAKMT_PFN(hsaKmtGetNodeWallclockFrequency) = (HSAKMT_DEF(hsaKmtGetNodeWallclockFrequency)*)(&hsaKmtGetNodeWallclockFrequency);

      DRM_PFN(amdgpu_device_initialize) = (DRM_DEF(amdgpu_device_initialize)*)(&amdgpu_device_initialize);
      DRM_PFN(amdgpu_device_deinitialize) = (DRM_DEF(amdgpu_device_deinitialize)*)(&amdgpu_device_deinitialize);
      DRM_PFN(amdgpu_query_gpu_info) = (DRM_DEF(amdgpu_query_gpu_info)*)(&amdgpu_query_gpu_info);
      DRM_PFN(amdgpu_bo_cpu_map) = (DRM_DEF(amdgpu_bo_cpu_map)*)(&amdgpu_bo_cpu_map);
      DRM_PFN(amdgpu_bo_free) = (DRM_DEF(amdgpu_bo_free)*)(&amdgpu_bo_free);
      DRM_PFN(amdgpu_bo_export) = (DRM_DEF(amdgpu_bo_export)*)(&amdgpu_bo_export);
      DRM_PFN(amdgpu_bo_import) = (DRM_DEF(amdgpu_bo_import)*)(&amdgpu_bo_import);
      DRM_PFN(amdgpu_bo_va_op) = (DRM_DEF(amdgpu_bo_va_op)*)(&amdgpu_bo_va_op);
      DRM_PFN(amdgpu_bo_query_info) = (DRM_DEF(amdgpu_bo_query_info)*)(&amdgpu_bo_query_info);
      DRM_PFN(amdgpu_bo_set_metadata) = (DRM_DEF(amdgpu_bo_set_metadata)*)(&amdgpu_bo_set_metadata);
#if defined(__linux__)
      DRM_PFN(drmCommandWriteRead) = (DRM_DEF(drmCommandWriteRead)*)(&drmCommandWriteRead);
#endif
    }
  }

  bool ThunkLoader::CreateThunkInstance() {
    if (!IsDTIF())
      return true;

    DtifCreateFunc* pfnDtifCreate =
        (DtifCreateFunc*)rocr::os::GetExportAddress(thunk_handle, "DtifCreate");
    if (pfnDtifCreate != nullptr) {
      if (pfnDtifCreate("HSA") != nullptr) {
        debug_print("DtifCreate OK!\n");
        return true;
      } else {
        debug_print("DtifCreate failed!\n");
        return false;
      }
    }
    return false;
  }

  bool ThunkLoader::DestroyThunkInstance() {
    if (!IsDTIF())
      return true;

    if (thunk_handle == nullptr)
      return false;

    DtifDestroyFunc* pfnDtifDestroy =
        (DtifDestroyFunc*)rocr::os::GetExportAddress(thunk_handle, "DtifDestroy");
    if (pfnDtifDestroy != nullptr) {
      pfnDtifDestroy();
      debug_print("DtifDestroy OK!\n");
      return true;
    }
    return false;
  }

  bool ThunkLoader::CheckThunkAbi() {
    if (!IsDXG()) return true;

    if (thunk_handle == nullptr) return false;

    DxgAbiCheckFunc* pfnDxgAbiCheck =
        (DxgAbiCheckFunc*)rocr::os::GetExportAddress(thunk_handle, "DxgAbiCheck");
    if (pfnDxgAbiCheck == nullptr) {
      // Old librocdxg without DxgAbiCheck — assume compatible.
      debug_print("DxgAbiCheck not exported, skipping ABI check.\n");
      return true;
    }

    HsaStructureSizes sizes = {};
    sizes.StructureSizes = (HSAuint16)sizeof(HsaStructureSizes);
    sizes.SizeOfHsaNodeProperties = (HSAuint16)sizeof(HsaNodeProperties);
    sizes.SizeOfHsaExternalHandleDesc = (HSAuint16)sizeof(HsaExternalHandleDesc);

    if (pfnDxgAbiCheck(&sizes) != HSAKMT_STATUS_SUCCESS) {
      debug_print("DxgAbiCheck failed!\n");
      return false;
    }
    return true;
  }
}   //  namespace core
}   //  namespace rocr
