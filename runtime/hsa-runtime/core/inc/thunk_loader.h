////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HSA_RUNTIME_CORE_INC_THUNK_LOADER_H
#define HSA_RUNTIME_CORE_INC_THUNK_LOADER_H

#include <amdgpu.h>
#include "hsakmt/hsakmttypes.h"

namespace rocr {
namespace core {

#define HSAKMT_DEF(function_name)   PFN##function_name
#define HSAKMT_PFN(function_name)   pfn_##function_name
#define HSAKMT_CALL(function_name)   core::Runtime::runtime_singleton_->thunkLoader()->pfn_##function_name

#define DRM_DEF(function_name)   PFN##function_name
#define DRM_PFN(function_name)   pfn_##function_name
#define DRM_CALL(function_name)   core::Runtime::runtime_singleton_->thunkLoader()->pfn_##function_name

class ThunkLoader {
  public:
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtOpenKFD))(void);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtCloseKFD))(void);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtGetVersion))(HsaVersionInfo* VersionInfo);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtAcquireSystemProperties))(HsaSystemProperties* SystemProperties);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtReleaseSystemProperties))(void);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtGetNodeProperties))(HSAuint32 NodeId, \
                                      HsaNodeProperties* NodeProperties);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtGetNodeMemoryProperties))(HSAuint32 NodeId, \
                                      HSAuint32 NumBanks, \
                                      HsaMemoryProperties* MemoryProperties);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtGetNodeCacheProperties))(HSAuint32 NodeId, \
                                      HSAuint32 ProcessorId, \
                                      HSAuint32 NumCaches, \
                                      HsaCacheProperties* CacheProperties);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtGetNodeIoLinkProperties))(HSAuint32 NodeId, \
                                      HSAuint32 NumIoLinks, \
                                      HsaIoLinkProperties* IoLinkProperties);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtCreateEvent))(HsaEventDescriptor* EventDesc, \
                                      bool ManualReset, \
                                      bool IsSignaled, \
                                      HsaEvent** Event);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtDestroyEvent))(HsaEvent* Event);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtSetEvent))(HsaEvent* Event);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtResetEvent))(HsaEvent* Event);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtQueryEventState))(HsaEvent* Event, \
                                      HSAuint32 Milliseconds);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtWaitOnEvent))(HsaEvent* Event, \
                                      HSAuint32 Milliseconds);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtWaitOnMultipleEvents))(HsaEvent* Events[], \
                                      HSAuint32 NumEvents, \
                                      bool WaitOnAll, \
                                      HSAuint32 Milliseconds);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtCreateQueue))(HSAuint32 NodeId, \
                                      HSA_QUEUE_TYPE Type, \
                                      HSAuint32 QueuePercentage, \
                                      HSA_QUEUE_PRIORITY Priority, \
                                      void* QueueAddress, \
                                      HSAuint64 QueueSizeInBytes, \
                                      HsaEvent* Event, \
                                      HsaQueueResource* QueueResource);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtCreateQueueExt))(HSAuint32 NodeId, \
                                      HSA_QUEUE_TYPE Type, \
                                      HSAuint32 QueuePercentage, \
                                      HSA_QUEUE_PRIORITY Priority, \
                                      HSAuint32 SdmaEngineId, \
                                      void* QueueAddress, \
                                      HSAuint64 QueueSizeInBytes, \
                                      HsaEvent* Event, \
                                      HsaQueueResource* QueueResource);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtUpdateQueue))( HSA_QUEUEID QueueId, \
                                      HSAuint32 QueuePercentage, \
                                      HSA_QUEUE_PRIORITY Priority, \
                                      void* QueueAddress, \
                                      HSAuint64 QueueSize, \
                                      HsaEvent* Event);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtDestroyQueue))(HSA_QUEUEID QueueId);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtSetQueueCUMask))(HSA_QUEUEID QueueId, \
                                      HSAuint32 CUMaskCount, \
                                      HSAuint32* QueueCUMask);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtSetMemoryPolicy))(HSAuint32 Node, \
                                      HSAuint32 DefaultPolicy, \
                                      HSAuint32 AlternatePolicy, \
                                      void* MemoryAddressAlternate, \
                                      HSAuint64 MemorySizeInBytes);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtAllocMemory))(HSAuint32 PreferredNode, \
                                      HSAuint64 SizeInBytes, \
                                      HsaMemFlags MemFlags, \
                                      void** MemoryAddress);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtAllocMemoryAlign))(HSAuint32 PreferredNode, \
                                      HSAuint64 SizeInBytes, \
                                      HSAuint64 Alignment, \
                                      HsaMemFlags emFlags, \
                                      void** MemoryAddress);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtFreeMemory))(void* MemoryAddress, \
                                      HSAuint64 SizeInBytes);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtAvailableMemory))(HSAuint32 Node, \
                                      HSAuint64 *AvailableBytes);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtRegisterMemory))(void* MemoryAddress, \
                                      HSAuint64 MemorySizeInBytes);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtRegisterMemoryToNodes))(void *MemoryAddress, \
                                      HSAuint64 MemorySizeInBytes, \
                                      HSAuint64 NumberOfNodes, \
                                      HSAuint32* NodeArray);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtRegisterMemoryWithFlags))(void *MemoryAddress, \
                                      HSAuint64 MemorySizeInBytes, \
                                      HsaMemFlags MemFlags);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtRegisterGraphicsHandleToNodes))(HSAuint64 GraphicsResourceHandle, \
                                      HsaGraphicsResourceInfo *GraphicsResourceInfo, \
                                      HSAuint64 NumberOfNodes, \
                                      HSAuint32* NodeArray);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtRegisterGraphicsHandleToNodesExt))(HSAuint64 GraphicsResourceHandle, \
                                      HsaGraphicsResourceInfo *GraphicsResourceInfo, \
                                      HSAuint64 NumberOfNodes, \
                                      HSAuint32* NodeArray, \
                                      HSA_REGISTER_MEM_FLAGS RegisterFlags);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtShareMemory))(void *MemoryAddress, \
                                      HSAuint64 SizeInBytes, \
                                      HsaSharedMemoryHandle *SharedMemoryHandle);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtRegisterSharedHandle))(const HsaSharedMemoryHandle *SharedMemoryHandle, \
                                      void **MemoryAddress, \
                                      HSAuint64 *SizeInBytes);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtRegisterSharedHandleToNodes))(const HsaSharedMemoryHandle *SharedMemoryHandle, \
                                      void **MemoryAddress, \
                                      HSAuint64 *SizeInBytes, \
                                      HSAuint64 NumberOfNodes, \
                                      HSAuint32* NodeArray);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtProcessVMRead))(HSAuint32 Pid, \
                                      HsaMemoryRange *LocalMemoryArray, \
                                      HSAuint64 LocalMemoryArrayCount, \
                                      HsaMemoryRange *RemoteMemoryArray, \
                                      HSAuint64 RemoteMemoryArrayCount, \
                                      HSAuint64 *SizeCopied);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtProcessVMWrite))(HSAuint32 Pid, \
                                      HsaMemoryRange *LocalMemoryArray, \
                                      HSAuint64 LocalMemoryArrayCount, \
                                      HsaMemoryRange *RemoteMemoryArray, \
                                      HSAuint64 RemoteMemoryArrayCount, \
                                      HSAuint64 *SizeCopied);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtDeregisterMemory))(void* MemoryAddress);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtMapMemoryToGPU))(void*  MemoryAddress, \
                                      HSAuint64 MemorySizeInBytes, \
                                      HSAuint64* AlternateVAGPU);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtMapMemoryToGPUNodes))(void* MemoryAddress, \
                                      HSAuint64 MemorySizeInBytes, \
                                      HSAuint64* AlternateVAGPU, \
                                      HsaMemMapFlags MemMapFlags, \
                                      HSAuint64 NumberOfNodes, \
                                      HSAuint32* NodeArray);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtUnmapMemoryToGPU))(void* MemoryAddress);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtDbgRegister))(HSAuint32 NodeId);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtDbgUnregister))(HSAuint32 NodeId);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtDbgWavefrontControl))(HSAuint32 NodeId, \
                                      HSA_DBG_WAVEOP Operand, \
                                      HSA_DBG_WAVEMODE Mode, \
                                      HSAuint32 TrapId, \
                                      HsaDbgWaveMessage* DbgWaveMsgRing);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtDbgAddressWatch))(HSAuint32 NodeId, \
                                      HSAuint32 NumWatchPoints, \
                                      HSA_DBG_WATCH_MODE WatchMode[], \
                                      void* WatchAddress[], \
                                      HSAuint64 WatchMask[], \
                                      HsaEvent* WatchEvent[]);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtDbgEnable))(void **runtime_info, \
                                      HSAuint32 *data_size);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtDbgDisable))(void);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtDbgGetDeviceData))(void **data, \
                                      HSAuint32 *n_entries, \
                                      HSAuint32 *entry_size);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtDbgGetQueueData))(void **data, \
                                      HSAuint32 *n_entries, \
                                      HSAuint32 *entry_size, \
                                      bool suspend_queues);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtGetClockCounters))(HSAuint32 NodeId, \
                                      HsaClockCounters* Counters);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtPmcGetCounterProperties))(HSAuint32 NodeId, \
                                      HsaCounterProperties** CounterProperties);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtPmcRegisterTrace))(HSAuint32 NodeId, \
                                      HSAuint32 NumberOfCounters, \
                                      HsaCounter* Counters, \
                                      HsaPmcTraceRoot* TraceRoot);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtPmcUnregisterTrace))(HSAuint32 NodeId, \
                                      HSATraceId TraceId);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtPmcAcquireTraceAccess))(HSAuint32 NodeId, \
                                      HSATraceId TraceId);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtPmcReleaseTraceAccess))(HSAuint32 NodeId, \
                                      HSATraceId TraceId);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtPmcStartTrace))(HSATraceId TraceId, \
                                      void* TraceBuffer, \
                                      HSAuint64 TraceBufferSizeBytes);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtPmcQueryTrace))(HSATraceId TraceId);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtPmcStopTrace))(HSATraceId TraceId);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtMapGraphicHandle))(HSAuint32 NodeId, \
                                      HSAuint64 GraphicDeviceHandle, \
                                      HSAuint64 GraphicResourceHandle, \
                                      HSAuint64 GraphicResourceOffset, \
                                      HSAuint64 GraphicResourceSize, \
                                      HSAuint64* FlatMemoryAddress);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtUnmapGraphicHandle))(HSAuint32 NodeId, \
                                      HSAuint64 FlatMemoryAddress, \
                                      HSAuint64 SizeInBytes);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtSetTrapHandler))(HSAuint32 NodeId, \
                                      void* TrapHandlerBaseAddress, \
                                      HSAuint64 TrapHandlerSizeInBytes, \
                                      void* TrapBufferBaseAddress, \
                                      HSAuint64 TrapBufferSizeInBytes);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtGetTileConfig))(HSAuint32 NodeId, \
                                      HsaGpuTileConfig* config);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtQueryPointerInfo))(const void* Pointer, \
                                      HsaPointerInfo* PointerInfo);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtSetMemoryUserData))(const void* Pointer,  \
                                      void* UserData);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtGetQueueInfo))(HSA_QUEUEID QueueId, \
                                      HsaQueueInfo *QueueInfo);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtAllocQueueGWS))(HSA_QUEUEID QueueId, \
                                      HSAuint32 nGWS, \
                                      HSAuint32 *firstGWS);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtRuntimeEnable))(void* rDebug, \
                                      bool setupTtmp);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtRuntimeDisable))(void);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtCheckRuntimeDebugSupport))(void);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtGetRuntimeCapabilities))(HSAuint32 *caps_mask);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtDebugTrapIoctl))(struct kfd_ioctl_dbg_trap_args *arg, \
                                      HSA_QUEUEID *Queues, \
                                      HSAuint64 *DebugReturn);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtSPMAcquire))(HSAuint32 PreferredNode);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtSPMRelease))(HSAuint32 PreferredNode);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtSPMSetDestBuffer))(HSAuint32 PreferredNode, \
                                      HSAuint32 SizeInBytes, \
                                      HSAuint32* timeout, \
                                      HSAuint32* SizeCopied, \
                                      void *DestMemoryAddress, \
                                      bool *isSPMDataLoss);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtSVMSetAttr))(void *start_addr, \
                                      HSAuint64 size, \
                                      unsigned int nattr, \
                                      HSA_SVM_ATTRIBUTE *attrs);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtSVMGetAttr))(void *start_addr, \
                                      HSAuint64 size, \
                                      unsigned int nattr, \
                                      HSA_SVM_ATTRIBUTE *attrs);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtSetXNACKMode))(HSAint32 enable);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtGetXNACKMode))(HSAint32 * enable);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtOpenSMI))(HSAuint32 NodeId, \
                                      int *fd);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtExportDMABufHandle))(void *MemoryAddress, \
                                      HSAuint64 MemorySizeInBytes, \
                                      int *DMABufFd, \
                                      HSAuint64 *Offset);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtWaitOnEvent_Ext))(HsaEvent* Event, \
                                      HSAuint32 Milliseconds, \
                                      uint64_t *event_age);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtWaitOnMultipleEvents_Ext))(HsaEvent* Events[], \
                                      HSAuint32 NumEvents, \
                                      bool WaitOnAll, \
                                      HSAuint32 Milliseconds, \
                                      uint64_t *event_age);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtReplaceAsanHeaderPage))(void *addr);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtReturnAsanHeaderPage))(void *addr);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtGetAMDGPUDeviceHandle))(HSAuint32 NodeId, \
                                      HsaAMDGPUDeviceHandle *DeviceHandle);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtPcSamplingQueryCapabilities))(HSAuint32 NodeId, \
                                      void *sample_info, \
                                      HSAuint32 sample_info_sz, \
                                      HSAuint32 *sz_needed);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtPcSamplingCreate))(HSAuint32 node_id, \
                                      HsaPcSamplingInfo *sample_info, \
                                      HsaPcSamplingTraceId *traceId);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtPcSamplingDestroy))(HSAuint32 NodeId, \
                                      HsaPcSamplingTraceId traceId);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtPcSamplingStart))(HSAuint32 NodeId, \
                                      HsaPcSamplingTraceId traceId);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtPcSamplingStop))(HSAuint32 NodeId, \
                                      HsaPcSamplingTraceId traceId);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtPcSamplingSupport))(void);
    typedef HSAKMT_STATUS (HSAKMT_DEF(hsaKmtModelEnabled))(bool* enable);

    /* drm API */
    typedef int (DRM_DEF(amdgpu_device_initialize))(int fd, \
                                      uint32_t *major_version, \
                                      uint32_t *minor_version, \
                                      amdgpu_device_handle *device_handle);

    typedef int (DRM_DEF(amdgpu_device_deinitialize))(amdgpu_device_handle device_handle);

    typedef int (DRM_DEF(amdgpu_query_gpu_info))(amdgpu_device_handle dev, \
                                      struct amdgpu_gpu_info *info);

    typedef int (DRM_DEF(amdgpu_bo_cpu_map))(amdgpu_bo_handle bo, \
                                      void **cpu);

    typedef int (DRM_DEF(amdgpu_bo_free))(amdgpu_bo_handle buf_handle);

    typedef int (DRM_DEF(amdgpu_bo_export))(amdgpu_bo_handle bo, \
                                      enum amdgpu_bo_handle_type type, \
                                      uint32_t *shared_handle);

    typedef int (DRM_DEF(amdgpu_bo_import))(amdgpu_device_handle dev, \
                                      enum amdgpu_bo_handle_type type, \
                                      uint32_t shared_handle, \
                                      struct amdgpu_bo_import_result *output);

    typedef int (DRM_DEF(amdgpu_bo_va_op))(amdgpu_bo_handle bo, \
                                      uint64_t offset, \
                                      uint64_t size, \
                                      uint64_t addr, \
                                      uint64_t flags, \
                                      uint32_t op);

    typedef int (DRM_DEF(drmCommandWriteRead))(int fd, \
                                      unsigned long drmCommandIndex, \
                                      void *data, \
                                      unsigned long size);

    ThunkLoader();
    ~ThunkLoader();

    void LoadThunkApiTable();

    HSAKMT_DEF(hsaKmtOpenKFD)* HSAKMT_PFN(hsaKmtOpenKFD);
    HSAKMT_DEF(hsaKmtCloseKFD)* HSAKMT_PFN(hsaKmtCloseKFD);
    HSAKMT_DEF(hsaKmtGetVersion)* HSAKMT_PFN(hsaKmtGetVersion);
    HSAKMT_DEF(hsaKmtAcquireSystemProperties)* HSAKMT_PFN(hsaKmtAcquireSystemProperties);
    HSAKMT_DEF(hsaKmtReleaseSystemProperties)* HSAKMT_PFN(hsaKmtReleaseSystemProperties);
    HSAKMT_DEF(hsaKmtGetNodeProperties)* HSAKMT_PFN(hsaKmtGetNodeProperties);
    HSAKMT_DEF(hsaKmtGetNodeMemoryProperties)* HSAKMT_PFN(hsaKmtGetNodeMemoryProperties);
    HSAKMT_DEF(hsaKmtGetNodeCacheProperties)* HSAKMT_PFN(hsaKmtGetNodeCacheProperties);
    HSAKMT_DEF(hsaKmtGetNodeIoLinkProperties)* HSAKMT_PFN(hsaKmtGetNodeIoLinkProperties);
    HSAKMT_DEF(hsaKmtCreateEvent)* HSAKMT_PFN(hsaKmtCreateEvent);
    HSAKMT_DEF(hsaKmtDestroyEvent)* HSAKMT_PFN(hsaKmtDestroyEvent);
    HSAKMT_DEF(hsaKmtSetEvent)* HSAKMT_PFN(hsaKmtSetEvent);
    HSAKMT_DEF(hsaKmtResetEvent)* HSAKMT_PFN(hsaKmtResetEvent);
    HSAKMT_DEF(hsaKmtQueryEventState)* HSAKMT_PFN(hsaKmtQueryEventState);
    HSAKMT_DEF(hsaKmtWaitOnEvent)* HSAKMT_PFN(hsaKmtWaitOnEvent);
    HSAKMT_DEF(hsaKmtWaitOnMultipleEvents)* HSAKMT_PFN(hsaKmtWaitOnMultipleEvents);
    HSAKMT_DEF(hsaKmtCreateQueue)* HSAKMT_PFN(hsaKmtCreateQueue);
    HSAKMT_DEF(hsaKmtCreateQueueExt)* HSAKMT_PFN(hsaKmtCreateQueueExt);
    HSAKMT_DEF(hsaKmtUpdateQueue)* HSAKMT_PFN(hsaKmtUpdateQueue);
    HSAKMT_DEF(hsaKmtDestroyQueue)* HSAKMT_PFN(hsaKmtDestroyQueue);
    HSAKMT_DEF(hsaKmtSetQueueCUMask)* HSAKMT_PFN(hsaKmtSetQueueCUMask);
    HSAKMT_DEF(hsaKmtSetMemoryPolicy)* HSAKMT_PFN(hsaKmtSetMemoryPolicy);
    HSAKMT_DEF(hsaKmtAllocMemory)* HSAKMT_PFN(hsaKmtAllocMemory);
    HSAKMT_DEF(hsaKmtAllocMemoryAlign)* HSAKMT_PFN(hsaKmtAllocMemoryAlign);
    HSAKMT_DEF(hsaKmtFreeMemory)* HSAKMT_PFN(hsaKmtFreeMemory);
    HSAKMT_DEF(hsaKmtAvailableMemory)* HSAKMT_PFN(hsaKmtAvailableMemory);
    HSAKMT_DEF(hsaKmtRegisterMemory)* HSAKMT_PFN(hsaKmtRegisterMemory);
    HSAKMT_DEF(hsaKmtRegisterMemoryToNodes)* HSAKMT_PFN(hsaKmtRegisterMemoryToNodes);
    HSAKMT_DEF(hsaKmtRegisterMemoryWithFlags)* HSAKMT_PFN(hsaKmtRegisterMemoryWithFlags);
    HSAKMT_DEF(hsaKmtRegisterGraphicsHandleToNodes)* HSAKMT_PFN(hsaKmtRegisterGraphicsHandleToNodes);
    HSAKMT_DEF(hsaKmtRegisterGraphicsHandleToNodesExt)* HSAKMT_PFN(hsaKmtRegisterGraphicsHandleToNodesExt);
    HSAKMT_DEF(hsaKmtShareMemory)* HSAKMT_PFN(hsaKmtShareMemory);
    HSAKMT_DEF(hsaKmtRegisterSharedHandle)* HSAKMT_PFN(hsaKmtRegisterSharedHandle);
    HSAKMT_DEF(hsaKmtRegisterSharedHandleToNodes)* HSAKMT_PFN(hsaKmtRegisterSharedHandleToNodes);
    HSAKMT_DEF(hsaKmtProcessVMRead)* HSAKMT_PFN(hsaKmtProcessVMRead);
    HSAKMT_DEF(hsaKmtProcessVMWrite)* HSAKMT_PFN(hsaKmtProcessVMWrite);
    HSAKMT_DEF(hsaKmtDeregisterMemory)* HSAKMT_PFN(hsaKmtDeregisterMemory);
    HSAKMT_DEF(hsaKmtMapMemoryToGPU)* HSAKMT_PFN(hsaKmtMapMemoryToGPU);
    HSAKMT_DEF(hsaKmtMapMemoryToGPUNodes)* HSAKMT_PFN(hsaKmtMapMemoryToGPUNodes);
    HSAKMT_DEF(hsaKmtUnmapMemoryToGPU)* HSAKMT_PFN(hsaKmtUnmapMemoryToGPU);
    HSAKMT_DEF(hsaKmtDbgRegister)* HSAKMT_PFN(hsaKmtDbgRegister);
    HSAKMT_DEF(hsaKmtDbgUnregister)* HSAKMT_PFN(hsaKmtDbgUnregister);
    HSAKMT_DEF(hsaKmtDbgWavefrontControl)* HSAKMT_PFN(hsaKmtDbgWavefrontControl);
    HSAKMT_DEF(hsaKmtDbgAddressWatch)* HSAKMT_PFN(hsaKmtDbgAddressWatch);
    HSAKMT_DEF(hsaKmtDbgEnable)* HSAKMT_PFN(hsaKmtDbgEnable);
    HSAKMT_DEF(hsaKmtDbgDisable)* HSAKMT_PFN(hsaKmtDbgDisable);
    HSAKMT_DEF(hsaKmtDbgGetDeviceData)* HSAKMT_PFN(hsaKmtDbgGetDeviceData);
    HSAKMT_DEF(hsaKmtDbgGetQueueData)* HSAKMT_PFN(hsaKmtDbgGetQueueData);
    HSAKMT_DEF(hsaKmtGetClockCounters)* HSAKMT_PFN(hsaKmtGetClockCounters);
    HSAKMT_DEF(hsaKmtPmcGetCounterProperties)* HSAKMT_PFN(hsaKmtPmcGetCounterProperties);
    HSAKMT_DEF(hsaKmtPmcRegisterTrace)* HSAKMT_PFN(hsaKmtPmcRegisterTrace);
    HSAKMT_DEF(hsaKmtPmcUnregisterTrace)* HSAKMT_PFN(hsaKmtPmcUnregisterTrace);
    HSAKMT_DEF(hsaKmtPmcAcquireTraceAccess)* HSAKMT_PFN(hsaKmtPmcAcquireTraceAccess);
    HSAKMT_DEF(hsaKmtPmcReleaseTraceAccess)* HSAKMT_PFN(hsaKmtPmcReleaseTraceAccess);
    HSAKMT_DEF(hsaKmtPmcStartTrace)* HSAKMT_PFN(hsaKmtPmcStartTrace);
    HSAKMT_DEF(hsaKmtPmcQueryTrace)* HSAKMT_PFN(hsaKmtPmcQueryTrace);
    HSAKMT_DEF(hsaKmtPmcStopTrace)* HSAKMT_PFN(hsaKmtPmcStopTrace);
    HSAKMT_DEF(hsaKmtMapGraphicHandle)* HSAKMT_PFN(hsaKmtMapGraphicHandle);
    HSAKMT_DEF(hsaKmtUnmapGraphicHandle)* HSAKMT_PFN(hsaKmtUnmapGraphicHandle);
    HSAKMT_DEF(hsaKmtSetTrapHandler)* HSAKMT_PFN(hsaKmtSetTrapHandler);
    HSAKMT_DEF(hsaKmtGetTileConfig)* HSAKMT_PFN(hsaKmtGetTileConfig);
    HSAKMT_DEF(hsaKmtQueryPointerInfo)* HSAKMT_PFN(hsaKmtQueryPointerInfo);
    HSAKMT_DEF(hsaKmtSetMemoryUserData)* HSAKMT_PFN(hsaKmtSetMemoryUserData);
    HSAKMT_DEF(hsaKmtGetQueueInfo)* HSAKMT_PFN(hsaKmtGetQueueInfo);
    HSAKMT_DEF(hsaKmtAllocQueueGWS)* HSAKMT_PFN(hsaKmtAllocQueueGWS);
    HSAKMT_DEF(hsaKmtRuntimeEnable)* HSAKMT_PFN(hsaKmtRuntimeEnable);
    HSAKMT_DEF(hsaKmtRuntimeDisable)* HSAKMT_PFN(hsaKmtRuntimeDisable);
    HSAKMT_DEF(hsaKmtCheckRuntimeDebugSupport)* HSAKMT_PFN(hsaKmtCheckRuntimeDebugSupport);
    HSAKMT_DEF(hsaKmtGetRuntimeCapabilities)* HSAKMT_PFN(hsaKmtGetRuntimeCapabilities);
    HSAKMT_DEF(hsaKmtDebugTrapIoctl)* HSAKMT_PFN(hsaKmtDebugTrapIoctl);
    HSAKMT_DEF(hsaKmtSPMAcquire)* HSAKMT_PFN(hsaKmtSPMAcquire);
    HSAKMT_DEF(hsaKmtSPMRelease)* HSAKMT_PFN(hsaKmtSPMRelease);
    HSAKMT_DEF(hsaKmtSPMSetDestBuffer)* HSAKMT_PFN(hsaKmtSPMSetDestBuffer);
    HSAKMT_DEF(hsaKmtSVMSetAttr)* HSAKMT_PFN(hsaKmtSVMSetAttr);
    HSAKMT_DEF(hsaKmtSVMGetAttr)* HSAKMT_PFN(hsaKmtSVMGetAttr);
    HSAKMT_DEF(hsaKmtSetXNACKMode)* HSAKMT_PFN(hsaKmtSetXNACKMode);
    HSAKMT_DEF(hsaKmtGetXNACKMode)* HSAKMT_PFN(hsaKmtGetXNACKMode);
    HSAKMT_DEF(hsaKmtOpenSMI)* HSAKMT_PFN(hsaKmtOpenSMI);
    HSAKMT_DEF(hsaKmtExportDMABufHandle)* HSAKMT_PFN(hsaKmtExportDMABufHandle);
    HSAKMT_DEF(hsaKmtWaitOnEvent_Ext)* HSAKMT_PFN(hsaKmtWaitOnEvent_Ext);
    HSAKMT_DEF(hsaKmtWaitOnMultipleEvents_Ext)* HSAKMT_PFN(hsaKmtWaitOnMultipleEvents_Ext);
    HSAKMT_DEF(hsaKmtReplaceAsanHeaderPage)* HSAKMT_PFN(hsaKmtReplaceAsanHeaderPage);
    HSAKMT_DEF(hsaKmtReturnAsanHeaderPage)* HSAKMT_PFN(hsaKmtReturnAsanHeaderPage);
    HSAKMT_DEF(hsaKmtGetAMDGPUDeviceHandle)* HSAKMT_PFN(hsaKmtGetAMDGPUDeviceHandle);
    HSAKMT_DEF(hsaKmtPcSamplingQueryCapabilities)* HSAKMT_PFN(hsaKmtPcSamplingQueryCapabilities);
    HSAKMT_DEF(hsaKmtPcSamplingCreate)* HSAKMT_PFN(hsaKmtPcSamplingCreate);
    HSAKMT_DEF(hsaKmtPcSamplingDestroy)* HSAKMT_PFN(hsaKmtPcSamplingDestroy);
    HSAKMT_DEF(hsaKmtPcSamplingStart)* HSAKMT_PFN(hsaKmtPcSamplingStart);
    HSAKMT_DEF(hsaKmtPcSamplingStop)* HSAKMT_PFN(hsaKmtPcSamplingStop);
    HSAKMT_DEF(hsaKmtPcSamplingSupport)* HSAKMT_PFN(hsaKmtPcSamplingSupport);
    HSAKMT_DEF(hsaKmtModelEnabled)* HSAKMT_PFN(hsaKmtModelEnabled);

    DRM_DEF(amdgpu_device_initialize)* DRM_PFN(amdgpu_device_initialize);
    DRM_DEF(amdgpu_device_deinitialize)* DRM_PFN(amdgpu_device_deinitialize);
    DRM_DEF(amdgpu_query_gpu_info)* DRM_PFN(amdgpu_query_gpu_info);
    DRM_DEF(amdgpu_bo_cpu_map)* DRM_PFN(amdgpu_bo_cpu_map);
    DRM_DEF(amdgpu_bo_free)* DRM_PFN(amdgpu_bo_free);
    DRM_DEF(amdgpu_bo_export)* DRM_PFN(amdgpu_bo_export);
    DRM_DEF(amdgpu_bo_import)* DRM_PFN(amdgpu_bo_import);
    DRM_DEF(amdgpu_bo_va_op)* DRM_PFN(amdgpu_bo_va_op);
    DRM_DEF(drmCommandWriteRead)* DRM_PFN(drmCommandWriteRead);

  private:
    void *dtif_handle;
};

}   //  namespace core
}   //  namespace rocr

#endif
