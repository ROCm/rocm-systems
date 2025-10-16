////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
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
      return "libdtif.so";
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
    : thunk_handle(NULL),
      library_name(whoami()),
      is_loaded_(false) {
    if (!library_name.empty()) {
      rocr::os::DlError();  // Clear any existing error messages
      thunk_handle = rocr::os::LoadLib(library_name.c_str());
      if (thunk_handle == NULL) {
        fprintf(stderr, "Cannot load %s, failed:%s\n", library_name.c_str(), rocr::os::DlError());
      } else {
        debug_print("Load %s successully!\n", library_name.c_str());
      }
      is_loaded_ = true;
    }
  }

  ThunkLoader::~ThunkLoader() {
    if (IsSharedLibraryLoaded()
      && (thunk_handle != NULL)) {
        if (!rocr::os::CloseLib(thunk_handle)) {
          fprintf(stderr, "Cannot unload %s, failed:%s\n", library_name.c_str(), rocr::os::DlError());
        } else {
          debug_print("Unload %s successully!\n", library_name.c_str());
        }
    }
  }

  void ThunkLoader::LoadThunkApiTable() {
    if (IsSharedLibraryLoaded()) {
#if defined(__linux__)
      dlerror(); // Clear any existing error messages

      HSAKMT_PFN(hsaKmtOpenKFD) = (HSAKMT_DEF(hsaKmtOpenKFD)*)dlsym(thunk_handle, "hsaKmtOpenKFD");
      if (HSAKMT_PFN(hsaKmtOpenKFD) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtCloseKFD) = (HSAKMT_DEF(hsaKmtCloseKFD)*)dlsym(thunk_handle, "hsaKmtCloseKFD");
      if (HSAKMT_PFN(hsaKmtCloseKFD) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtGetVersion) = (HSAKMT_DEF(hsaKmtGetVersion)*)dlsym(thunk_handle, "hsaKmtGetVersion");
      if (HSAKMT_PFN(hsaKmtGetVersion) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtAcquireSystemProperties) = (HSAKMT_DEF(hsaKmtAcquireSystemProperties)*)dlsym(thunk_handle, "hsaKmtAcquireSystemProperties");
      if (HSAKMT_PFN(hsaKmtAcquireSystemProperties) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtReleaseSystemProperties) = (HSAKMT_DEF(hsaKmtReleaseSystemProperties)*)dlsym(thunk_handle, "hsaKmtReleaseSystemProperties");
      if (HSAKMT_PFN(hsaKmtReleaseSystemProperties) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtGetNodeProperties) = (HSAKMT_DEF(hsaKmtGetNodeProperties)*)dlsym(thunk_handle, "hsaKmtGetNodeProperties");
      if (HSAKMT_PFN(hsaKmtGetNodeProperties) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtGetNodeMemoryProperties) = (HSAKMT_DEF(hsaKmtGetNodeMemoryProperties)*)dlsym(thunk_handle, "hsaKmtGetNodeMemoryProperties");
      if (HSAKMT_PFN(hsaKmtGetNodeMemoryProperties) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtGetNodeCacheProperties) = (HSAKMT_DEF(hsaKmtGetNodeCacheProperties)*)dlsym(thunk_handle, "hsaKmtGetNodeCacheProperties");
      if (HSAKMT_PFN(hsaKmtGetNodeCacheProperties) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtGetNodeIoLinkProperties) = (HSAKMT_DEF(hsaKmtGetNodeIoLinkProperties)*)dlsym(thunk_handle, "hsaKmtGetNodeIoLinkProperties");
      if (HSAKMT_PFN(hsaKmtGetNodeIoLinkProperties) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtCreateEvent) = (HSAKMT_DEF(hsaKmtCreateEvent)*)dlsym(thunk_handle, "hsaKmtCreateEvent");
      if (HSAKMT_PFN(hsaKmtCreateEvent) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtDestroyEvent) = (HSAKMT_DEF(hsaKmtDestroyEvent)*)dlsym(thunk_handle, "hsaKmtDestroyEvent");
      if (HSAKMT_PFN(hsaKmtDestroyEvent) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtSetEvent) = (HSAKMT_DEF(hsaKmtSetEvent)*)dlsym(thunk_handle, "hsaKmtSetEvent");
      if (HSAKMT_PFN(hsaKmtSetEvent) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtResetEvent) = (HSAKMT_DEF(hsaKmtResetEvent)*)dlsym(thunk_handle, "hsaKmtResetEvent");
      if (HSAKMT_PFN(hsaKmtResetEvent) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtQueryEventState) = (HSAKMT_DEF(hsaKmtQueryEventState)*)dlsym(thunk_handle, "hsaKmtQueryEventState");
      if (HSAKMT_PFN(hsaKmtQueryEventState) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtWaitOnEvent) = (HSAKMT_DEF(hsaKmtWaitOnEvent)*)dlsym(thunk_handle, "hsaKmtWaitOnEvent");
      if (HSAKMT_PFN(hsaKmtWaitOnEvent) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtWaitOnMultipleEvents) = (HSAKMT_DEF(hsaKmtWaitOnMultipleEvents)*)dlsym(thunk_handle, "hsaKmtWaitOnMultipleEvents");
      if (HSAKMT_PFN(hsaKmtWaitOnMultipleEvents) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtCreateQueue) = (HSAKMT_DEF(hsaKmtCreateQueue)*)dlsym(thunk_handle, "hsaKmtCreateQueue");
      if (HSAKMT_PFN(hsaKmtCreateQueue) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtCreateQueueExt) = (HSAKMT_DEF(hsaKmtCreateQueueExt)*)dlsym(thunk_handle, "hsaKmtCreateQueueExt");
      if (HSAKMT_PFN(hsaKmtCreateQueueExt) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtUpdateQueue) = (HSAKMT_DEF(hsaKmtUpdateQueue)*)dlsym(thunk_handle, "hsaKmtUpdateQueue");
      if (HSAKMT_PFN(hsaKmtUpdateQueue) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtDestroyQueue) = (HSAKMT_DEF(hsaKmtDestroyQueue)*)dlsym(thunk_handle, "hsaKmtDestroyQueue");
      if (HSAKMT_PFN(hsaKmtDestroyQueue) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtSetQueueCUMask) = (HSAKMT_DEF(hsaKmtSetQueueCUMask)*)dlsym(thunk_handle, "hsaKmtSetQueueCUMask");
      if (HSAKMT_PFN(hsaKmtSetQueueCUMask) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtSetMemoryPolicy) = (HSAKMT_DEF(hsaKmtSetMemoryPolicy)*)dlsym(thunk_handle, "hsaKmtSetMemoryPolicy");
      if (HSAKMT_PFN(hsaKmtSetMemoryPolicy) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtAllocMemory) = (HSAKMT_DEF(hsaKmtAllocMemory)*)dlsym(thunk_handle, "hsaKmtAllocMemory");
      if (HSAKMT_PFN(hsaKmtAllocMemory) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtAllocMemoryAlign) = (HSAKMT_DEF(hsaKmtAllocMemoryAlign)*)dlsym(thunk_handle, "hsaKmtAllocMemoryAlign");
      if (HSAKMT_PFN(hsaKmtAllocMemoryAlign) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtFreeMemory) = (HSAKMT_DEF(hsaKmtFreeMemory)*)dlsym(thunk_handle, "hsaKmtFreeMemory");
      if (HSAKMT_PFN(hsaKmtFreeMemory) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtAvailableMemory) = (HSAKMT_DEF(hsaKmtAvailableMemory)*)dlsym(thunk_handle, "hsaKmtAvailableMemory");
      if (HSAKMT_PFN(hsaKmtAvailableMemory) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtRegisterMemory) = (HSAKMT_DEF(hsaKmtRegisterMemory)*)dlsym(thunk_handle, "hsaKmtRegisterMemory");
      if (HSAKMT_PFN(hsaKmtRegisterMemory) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtRegisterMemoryToNodes) = (HSAKMT_DEF(hsaKmtRegisterMemoryToNodes)*)dlsym(thunk_handle, "hsaKmtRegisterMemoryToNodes");
      if (HSAKMT_PFN(hsaKmtRegisterMemoryToNodes) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtRegisterMemoryWithFlags) = (HSAKMT_DEF(hsaKmtRegisterMemoryWithFlags)*)dlsym(thunk_handle, "hsaKmtRegisterMemoryWithFlags");
      if (HSAKMT_PFN(hsaKmtRegisterMemoryWithFlags) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtRegisterGraphicsHandleToNodes) = (HSAKMT_DEF(hsaKmtRegisterGraphicsHandleToNodes)*)dlsym(thunk_handle, "hsaKmtRegisterGraphicsHandleToNodes");
      if (HSAKMT_PFN(hsaKmtRegisterGraphicsHandleToNodes) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtRegisterGraphicsHandleToNodesExt) = (HSAKMT_DEF(hsaKmtRegisterGraphicsHandleToNodesExt)*)dlsym(thunk_handle, "hsaKmtRegisterGraphicsHandleToNodesExt");
      if (HSAKMT_PFN(hsaKmtRegisterGraphicsHandleToNodesExt) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtShareMemory) = (HSAKMT_DEF(hsaKmtShareMemory)*)dlsym(thunk_handle, "hsaKmtShareMemory");
      if (HSAKMT_PFN(hsaKmtShareMemory) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtRegisterSharedHandle) = (HSAKMT_DEF(hsaKmtRegisterSharedHandle)*)dlsym(thunk_handle, "hsaKmtRegisterSharedHandle");
      if (HSAKMT_PFN(hsaKmtRegisterSharedHandle) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtRegisterSharedHandleToNodes) = (HSAKMT_DEF(hsaKmtRegisterSharedHandleToNodes)*)dlsym(thunk_handle, "hsaKmtRegisterSharedHandleToNodes");
      if (HSAKMT_PFN(hsaKmtRegisterSharedHandleToNodes) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtProcessVMRead) = (HSAKMT_DEF(hsaKmtProcessVMRead)*)dlsym(thunk_handle, "hsaKmtProcessVMRead");
      if (HSAKMT_PFN(hsaKmtProcessVMRead) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtProcessVMWrite) = (HSAKMT_DEF(hsaKmtProcessVMWrite)*)dlsym(thunk_handle, "hsaKmtProcessVMWrite");
      if (HSAKMT_PFN(hsaKmtProcessVMWrite) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtDeregisterMemory) = (HSAKMT_DEF(hsaKmtDeregisterMemory)*)dlsym(thunk_handle, "hsaKmtDeregisterMemory");
      if (HSAKMT_PFN(hsaKmtDeregisterMemory) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtMapMemoryToGPU) = (HSAKMT_DEF(hsaKmtMapMemoryToGPU)*)dlsym(thunk_handle, "hsaKmtMapMemoryToGPU");
      if (HSAKMT_PFN(hsaKmtMapMemoryToGPU) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtMapMemoryToGPUNodes) = (HSAKMT_DEF(hsaKmtMapMemoryToGPUNodes)*)dlsym(thunk_handle, "hsaKmtMapMemoryToGPUNodes");
      if (HSAKMT_PFN(hsaKmtMapMemoryToGPUNodes) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtUnmapMemoryToGPU) = (HSAKMT_DEF(hsaKmtUnmapMemoryToGPU)*)dlsym(thunk_handle, "hsaKmtUnmapMemoryToGPU");
      if (HSAKMT_PFN(hsaKmtUnmapMemoryToGPU) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtDbgRegister) = (HSAKMT_DEF(hsaKmtDbgRegister)*)dlsym(thunk_handle, "hsaKmtDbgRegister");
      if (HSAKMT_PFN(hsaKmtDbgRegister) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtDbgUnregister) = (HSAKMT_DEF(hsaKmtDbgUnregister)*)dlsym(thunk_handle, "hsaKmtDbgUnregister");
      if (HSAKMT_PFN(hsaKmtDbgUnregister) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtDbgWavefrontControl) = (HSAKMT_DEF(hsaKmtDbgWavefrontControl)*)dlsym(thunk_handle, "hsaKmtDbgWavefrontControl");
      if (HSAKMT_PFN(hsaKmtDbgWavefrontControl) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtDbgAddressWatch) = (HSAKMT_DEF(hsaKmtDbgAddressWatch)*)dlsym(thunk_handle, "hsaKmtDbgAddressWatch");
      if (HSAKMT_PFN(hsaKmtDbgAddressWatch) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtDbgEnable) = (HSAKMT_DEF(hsaKmtDbgEnable)*)dlsym(thunk_handle, "hsaKmtDbgEnable");
      if (HSAKMT_PFN(hsaKmtDbgEnable) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtDbgDisable) = (HSAKMT_DEF(hsaKmtDbgDisable)*)dlsym(thunk_handle, "hsaKmtDbgDisable");
      if (HSAKMT_PFN(hsaKmtDbgDisable) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtDbgGetDeviceData) = (HSAKMT_DEF(hsaKmtDbgGetDeviceData)*)dlsym(thunk_handle, "hsaKmtDbgGetDeviceData");
      if (HSAKMT_PFN(hsaKmtDbgGetDeviceData) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtDbgGetQueueData) = (HSAKMT_DEF(hsaKmtDbgGetQueueData)*)dlsym(thunk_handle, "hsaKmtDbgGetQueueData");
      if (HSAKMT_PFN(hsaKmtDbgGetQueueData) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtGetClockCounters) = (HSAKMT_DEF(hsaKmtGetClockCounters)*)dlsym(thunk_handle, "hsaKmtGetClockCounters");
      if (HSAKMT_PFN(hsaKmtGetClockCounters) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtPmcGetCounterProperties) = (HSAKMT_DEF(hsaKmtPmcGetCounterProperties)*)dlsym(thunk_handle, "hsaKmtPmcGetCounterProperties");
      if (HSAKMT_PFN(hsaKmtPmcGetCounterProperties) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtPmcRegisterTrace) = (HSAKMT_DEF(hsaKmtPmcRegisterTrace)*)dlsym(thunk_handle, "hsaKmtPmcRegisterTrace");
      if (HSAKMT_PFN(hsaKmtPmcRegisterTrace) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtPmcUnregisterTrace) = (HSAKMT_DEF(hsaKmtPmcUnregisterTrace)*)dlsym(thunk_handle, "hsaKmtPmcUnregisterTrace");
      if (HSAKMT_PFN(hsaKmtPmcUnregisterTrace) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtPmcAcquireTraceAccess) = (HSAKMT_DEF(hsaKmtPmcAcquireTraceAccess)*)dlsym(thunk_handle, "hsaKmtPmcAcquireTraceAccess");
      if (HSAKMT_PFN(hsaKmtPmcAcquireTraceAccess) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtPmcReleaseTraceAccess) = (HSAKMT_DEF(hsaKmtPmcReleaseTraceAccess)*)dlsym(thunk_handle, "hsaKmtPmcReleaseTraceAccess");
      if (HSAKMT_PFN(hsaKmtPmcReleaseTraceAccess) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtPmcStartTrace) = (HSAKMT_DEF(hsaKmtPmcStartTrace)*)dlsym(thunk_handle, "hsaKmtPmcStartTrace");
      if (HSAKMT_PFN(hsaKmtPmcStartTrace) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtPmcQueryTrace) = (HSAKMT_DEF(hsaKmtPmcQueryTrace)*)dlsym(thunk_handle, "hsaKmtPmcQueryTrace");
      if (HSAKMT_PFN(hsaKmtPmcQueryTrace) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtPmcStopTrace) = (HSAKMT_DEF(hsaKmtPmcStopTrace)*)dlsym(thunk_handle, "hsaKmtPmcStopTrace");
      if (HSAKMT_PFN(hsaKmtPmcStopTrace) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtMapGraphicHandle) = (HSAKMT_DEF(hsaKmtMapGraphicHandle)*)dlsym(thunk_handle, "hsaKmtMapGraphicHandle");
      if (HSAKMT_PFN(hsaKmtMapGraphicHandle) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtUnmapGraphicHandle) = (HSAKMT_DEF(hsaKmtUnmapGraphicHandle)*)dlsym(thunk_handle, "hsaKmtUnmapGraphicHandle");
      if (HSAKMT_PFN(hsaKmtUnmapGraphicHandle) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtSetTrapHandler) = (HSAKMT_DEF(hsaKmtSetTrapHandler)*)dlsym(thunk_handle, "hsaKmtSetTrapHandler");
      if (HSAKMT_PFN(hsaKmtSetTrapHandler) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtGetTileConfig) = (HSAKMT_DEF(hsaKmtGetTileConfig)*)dlsym(thunk_handle, "hsaKmtGetTileConfig");
      if (HSAKMT_PFN(hsaKmtGetTileConfig) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtQueryPointerInfo) = (HSAKMT_DEF(hsaKmtQueryPointerInfo)*)dlsym(thunk_handle, "hsaKmtQueryPointerInfo");
      if (HSAKMT_PFN(hsaKmtQueryPointerInfo) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtSetMemoryUserData) = (HSAKMT_DEF(hsaKmtSetMemoryUserData)*)dlsym(thunk_handle, "hsaKmtSetMemoryUserData");
      if (HSAKMT_PFN(hsaKmtSetMemoryUserData) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtGetQueueInfo) = (HSAKMT_DEF(hsaKmtGetQueueInfo)*)dlsym(thunk_handle, "hsaKmtGetQueueInfo");
      if (HSAKMT_PFN(hsaKmtGetQueueInfo) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtAllocQueueGWS) = (HSAKMT_DEF(hsaKmtAllocQueueGWS)*)dlsym(thunk_handle, "hsaKmtAllocQueueGWS");
      if (HSAKMT_PFN(hsaKmtAllocQueueGWS) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtRuntimeEnable) = (HSAKMT_DEF(hsaKmtRuntimeEnable)*)dlsym(thunk_handle, "hsaKmtRuntimeEnable");
      if (HSAKMT_PFN(hsaKmtRuntimeEnable) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtRuntimeDisable) = (HSAKMT_DEF(hsaKmtRuntimeDisable)*)dlsym(thunk_handle, "hsaKmtRuntimeDisable");
      if (HSAKMT_PFN(hsaKmtRuntimeDisable) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtCheckRuntimeDebugSupport) = (HSAKMT_DEF(hsaKmtCheckRuntimeDebugSupport)*)dlsym(thunk_handle, "hsaKmtCheckRuntimeDebugSupport");
      if (HSAKMT_PFN(hsaKmtCheckRuntimeDebugSupport) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtGetRuntimeCapabilities) = (HSAKMT_DEF(hsaKmtGetRuntimeCapabilities)*)dlsym(thunk_handle, "hsaKmtGetRuntimeCapabilities");
      if (HSAKMT_PFN(hsaKmtGetRuntimeCapabilities) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtDebugTrapIoctl) = (HSAKMT_DEF(hsaKmtDebugTrapIoctl)*)dlsym(thunk_handle, "hsaKmtDebugTrapIoctl");
      if (HSAKMT_PFN(hsaKmtDebugTrapIoctl) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtSPMAcquire) = (HSAKMT_DEF(hsaKmtSPMAcquire)*)dlsym(thunk_handle, "hsaKmtSPMAcquire");
      if (HSAKMT_PFN(hsaKmtSPMAcquire) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtSPMRelease) = (HSAKMT_DEF(hsaKmtSPMRelease)*)dlsym(thunk_handle, "hsaKmtSPMRelease");
      if (HSAKMT_PFN(hsaKmtSPMRelease) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtSPMSetDestBuffer) = (HSAKMT_DEF(hsaKmtSPMSetDestBuffer)*)dlsym(thunk_handle, "hsaKmtSPMSetDestBuffer");
      if (HSAKMT_PFN(hsaKmtSPMSetDestBuffer) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtSVMSetAttr) = (HSAKMT_DEF(hsaKmtSVMSetAttr)*)dlsym(thunk_handle, "hsaKmtSVMSetAttr");
      if (HSAKMT_PFN(hsaKmtSVMSetAttr) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtSVMGetAttr) = (HSAKMT_DEF(hsaKmtSVMGetAttr)*)dlsym(thunk_handle, "hsaKmtSVMGetAttr");
      if (HSAKMT_PFN(hsaKmtSVMGetAttr) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtSetXNACKMode) = (HSAKMT_DEF(hsaKmtSetXNACKMode)*)dlsym(thunk_handle, "hsaKmtSetXNACKMode");
      if (HSAKMT_PFN(hsaKmtSetXNACKMode) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtGetXNACKMode) = (HSAKMT_DEF(hsaKmtGetXNACKMode)*)dlsym(thunk_handle, "hsaKmtGetXNACKMode");
      if (HSAKMT_PFN(hsaKmtGetXNACKMode) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtOpenSMI) = (HSAKMT_DEF(hsaKmtOpenSMI)*)dlsym(thunk_handle, "hsaKmtOpenSMI");
      if (HSAKMT_PFN(hsaKmtOpenSMI) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtExportDMABufHandle) = (HSAKMT_DEF(hsaKmtExportDMABufHandle)*)dlsym(thunk_handle, "hsaKmtExportDMABufHandle");
      if (HSAKMT_PFN(hsaKmtExportDMABufHandle) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtWaitOnEvent_Ext) = (HSAKMT_DEF(hsaKmtWaitOnEvent_Ext)*)dlsym(thunk_handle, "hsaKmtWaitOnEvent_Ext");
      if (HSAKMT_PFN(hsaKmtWaitOnEvent_Ext) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtWaitOnMultipleEvents_Ext) = (HSAKMT_DEF(hsaKmtWaitOnMultipleEvents_Ext)*)dlsym(thunk_handle, "hsaKmtWaitOnMultipleEvents_Ext");
      if (HSAKMT_PFN(hsaKmtWaitOnMultipleEvents_Ext) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtReplaceAsanHeaderPage) = (HSAKMT_DEF(hsaKmtReplaceAsanHeaderPage)*)dlsym(thunk_handle, "hsaKmtReplaceAsanHeaderPage");
      if (HSAKMT_PFN(hsaKmtReplaceAsanHeaderPage) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtReturnAsanHeaderPage) = (HSAKMT_DEF(hsaKmtReturnAsanHeaderPage)*)dlsym(thunk_handle, "hsaKmtReturnAsanHeaderPage");
      if (HSAKMT_PFN(hsaKmtReturnAsanHeaderPage) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtGetAMDGPUDeviceHandle) = (HSAKMT_DEF(hsaKmtGetAMDGPUDeviceHandle)*)dlsym(thunk_handle, "hsaKmtGetAMDGPUDeviceHandle");
      if (HSAKMT_PFN(hsaKmtGetAMDGPUDeviceHandle) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtPcSamplingQueryCapabilities) = (HSAKMT_DEF(hsaKmtPcSamplingQueryCapabilities)*)dlsym(thunk_handle, "hsaKmtPcSamplingQueryCapabilities");
      if (HSAKMT_PFN(hsaKmtPcSamplingQueryCapabilities) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtPcSamplingCreate) = (HSAKMT_DEF(hsaKmtPcSamplingCreate)*)dlsym(thunk_handle, "hsaKmtPcSamplingCreate");
      if (HSAKMT_PFN(hsaKmtPcSamplingCreate) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtPcSamplingDestroy) = (HSAKMT_DEF(hsaKmtPcSamplingDestroy)*)dlsym(thunk_handle, "hsaKmtPcSamplingDestroy");
      if (HSAKMT_PFN(hsaKmtPcSamplingDestroy) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtPcSamplingStart) = (HSAKMT_DEF(hsaKmtPcSamplingStart)*)dlsym(thunk_handle, "hsaKmtPcSamplingStart");
      if (HSAKMT_PFN(hsaKmtPcSamplingStart) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtPcSamplingStop) = (HSAKMT_DEF(hsaKmtPcSamplingStop)*)dlsym(thunk_handle, "hsaKmtPcSamplingStop");
      if (HSAKMT_PFN(hsaKmtPcSamplingStop) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtPcSamplingSupport) = (HSAKMT_DEF(hsaKmtPcSamplingSupport)*)dlsym(thunk_handle, "hsaKmtPcSamplingSupport");
      if (HSAKMT_PFN(hsaKmtPcSamplingSupport) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtModelEnabled) = (HSAKMT_DEF(hsaKmtModelEnabled)*)dlsym(thunk_handle, "hsaKmtModelEnabled");
      if (HSAKMT_PFN(hsaKmtModelEnabled) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtQueueRingDoorbell) = (HSAKMT_DEF(hsaKmtQueueRingDoorbell)*)dlsym(thunk_handle, "hsaKmtQueueRingDoorbell");
      if (HSAKMT_PFN(hsaKmtQueueRingDoorbell) == NULL) goto ERROR;

      DRM_PFN(amdgpu_device_initialize) = (DRM_DEF(amdgpu_device_initialize)*)dlsym(thunk_handle, "amdgpu_device_initialize");
      if (DRM_PFN(amdgpu_device_initialize) == NULL) goto ERROR;

      HSAKMT_PFN(hsaKmtAisReadWriteFile) = (HSAKMT_DEF(hsaKmtAisReadWriteFile)*)dlsym(thunk_handle, "hsaKmtAisReadWriteFile");
      if (HSAKMT_PFN(hsaKmtAisReadWriteFile) == NULL) goto ERROR;

      DRM_PFN(amdgpu_device_deinitialize) = (DRM_DEF(amdgpu_device_deinitialize)*)dlsym(thunk_handle, "amdgpu_device_deinitialize");
      if (DRM_PFN(amdgpu_device_deinitialize) == NULL) goto ERROR;

      DRM_PFN(amdgpu_query_gpu_info) = (DRM_DEF(amdgpu_query_gpu_info)*)dlsym(thunk_handle, "amdgpu_query_gpu_info");
      if (DRM_PFN(amdgpu_query_gpu_info) == NULL) goto ERROR;

      DRM_PFN(amdgpu_bo_cpu_map) = (DRM_DEF(amdgpu_bo_cpu_map)*)dlsym(thunk_handle, "amdgpu_bo_cpu_map");
      if (DRM_PFN(amdgpu_bo_cpu_map) == NULL) goto ERROR;

      DRM_PFN(amdgpu_bo_free) = (DRM_DEF(amdgpu_bo_free)*)dlsym(thunk_handle, "amdgpu_bo_free");
      if (DRM_PFN(amdgpu_bo_free) == NULL) goto ERROR;

      DRM_PFN(amdgpu_bo_export) = (DRM_DEF(amdgpu_bo_export)*)dlsym(thunk_handle, "amdgpu_bo_export");
      if (DRM_PFN(amdgpu_bo_export) == NULL) goto ERROR;

      DRM_PFN(amdgpu_bo_import) = (DRM_DEF(amdgpu_bo_import)*)dlsym(thunk_handle, "amdgpu_bo_import");
      if (DRM_PFN(amdgpu_bo_import) == NULL) goto ERROR;

      DRM_PFN(amdgpu_bo_va_op) = (DRM_DEF(amdgpu_bo_va_op)*)dlsym(thunk_handle, "amdgpu_bo_va_op");
      if (DRM_PFN(amdgpu_bo_va_op) == NULL) goto ERROR;

      DRM_PFN(drmCommandWriteRead) = (DRM_DEF(drmCommandWriteRead)*)dlsym(thunk_handle, "drmCommandWriteRead");
      if (DRM_PFN(drmCommandWriteRead) == NULL) goto ERROR;
      debug_print("Load all DTIF APIs OK!\n");
      return;

ERROR:
      fprintf(stderr, "dlsym failed: %s\n", dlerror());
#endif
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

      DRM_PFN(amdgpu_device_initialize) = (DRM_DEF(amdgpu_device_initialize)*)(&amdgpu_device_initialize);
      DRM_PFN(amdgpu_device_deinitialize) = (DRM_DEF(amdgpu_device_deinitialize)*)(&amdgpu_device_deinitialize);
      DRM_PFN(amdgpu_query_gpu_info) = (DRM_DEF(amdgpu_query_gpu_info)*)(&amdgpu_query_gpu_info);
      DRM_PFN(amdgpu_bo_cpu_map) = (DRM_DEF(amdgpu_bo_cpu_map)*)(&amdgpu_bo_cpu_map);
      DRM_PFN(amdgpu_bo_free) = (DRM_DEF(amdgpu_bo_free)*)(&amdgpu_bo_free);
      DRM_PFN(amdgpu_bo_export) = (DRM_DEF(amdgpu_bo_export)*)(&amdgpu_bo_export);
      DRM_PFN(amdgpu_bo_import) = (DRM_DEF(amdgpu_bo_import)*)(&amdgpu_bo_import);
      DRM_PFN(amdgpu_bo_va_op) = (DRM_DEF(amdgpu_bo_va_op)*)(&amdgpu_bo_va_op);
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
    if (pfnDtifCreate != NULL) {
      if (pfnDtifCreate("HSA") != NULL) {
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

    if (thunk_handle == NULL)
      return false;

    DtifDestroyFunc* pfnDtifDestroy =
        (DtifDestroyFunc*)rocr::os::GetExportAddress(thunk_handle, "DtifDestroy");
    if (pfnDtifDestroy != NULL) {
      pfnDtifDestroy();
      debug_print("DtifDestroy OK!\n");
      return true;
    }
    return false;
  }
}   //  namespace core
}   //  namespace rocr
