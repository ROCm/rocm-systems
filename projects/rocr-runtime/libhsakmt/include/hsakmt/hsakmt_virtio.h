/*
 * Copyright © 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef HSAKMT_VIRTIO_H
#define HSAKMT_VIRTIO_H

#if defined(__linux__)
#include "hsakmt/linux/kfd_ioctl.h"
#endif

// Forward declaration for HsaKFDContext to avoid dependency issues
typedef struct _HsaKFDContext HsaKFDContext;

#include "hsakmt/hsakmt.h"
#include <libdrm/amdgpu.h>

#ifdef __cplusplus
extern "C" {
#endif

HSAKMT_STATUS HSAKMTAPI vhsaKmtOpenKFD(void);
HSAKMT_STATUS HSAKMTAPI vhsaKmtCloseKFD(void);
HSAKMT_STATUS HSAKMTAPI vhsaKmtAllocMemory(HSAuint32 PreferredNode, HSAuint64 SizeInBytes,
                                           HsaMemFlags MemFlags, void** MemoryAddress);
HSAKMT_STATUS HSAKMTAPI vhsaKmtAllocMemoryAlign(HSAuint32 PreferredNode, HSAuint64 SizeInBytes,
                                                HSAuint64 Alignment, HsaMemFlags MemFlags,
                                                void** MemoryAddress);
HSAKMT_STATUS HSAKMTAPI vhsaKmtFreeMemory(void* MemoryAddress, HSAuint64 SizeInBytes);
HSAKMT_STATUS HSAKMTAPI vhsaKmtMapMemoryToGPUNodes(void* MemoryAddress, HSAuint64 MemorySizeInBytes,
                                                   HSAuint64* AlternateVAGPU,
                                                   HsaMemMapFlags MemMapFlags,
                                                   HSAuint64 NumberOfNodes, HSAuint32* NodeArray);
HSAKMT_STATUS HSAKMTAPI vhsaKmtUnmapMemoryToGPU(void* MemoryAddress);
HSAKMT_STATUS HSAKMTAPI vhsaKmtAvailableMemory(HSAuint32 Node, HSAuint64* AvailableBytes);
HSAKMT_STATUS HSAKMTAPI vhsaKmtMapMemoryToGPU(void* MemoryAddress, HSAuint64 MemorySizeInBytes,
                                              HSAuint64* AlternateVAGPU);
HSAKMT_STATUS HSAKMTAPI vhsaKmtRegisterMemoryWithFlags(void* MemoryAddress,
                                                       HSAuint64 MemorySizeInBytes,
                                                       HsaMemFlags MemFlags);
HSAKMT_STATUS HSAKMTAPI vhsaKmtRegisterMemory(void* MemoryAddress, HSAuint64 MemorySizeInBytes);
HSAKMT_STATUS HSAKMTAPI vhsaKmtRegisterMemoryToNodes(void* MemoryAddress,
                                                     HSAuint64 MemorySizeInBytes,
                                                     HSAuint32 NumberOfNodes, HSAuint32* NodeArray);
HSAKMT_STATUS HSAKMTAPI vhsaKmtDeregisterMemory(void* MemoryAddress);
HSAKMT_STATUS HSAKMTAPI vhsaKmtGetVersion(HsaVersionInfo* v);
HSAKMT_STATUS HSAKMTAPI vhsaKmtAcquireSystemProperties(HsaSystemProperties* SystemProperties);
HSAKMT_STATUS HSAKMTAPI vhsaKmtReleaseSystemProperties(void);
HSAKMT_STATUS HSAKMTAPI vhsaKmtGetNodeProperties(HSAuint32 NodeId,
                                                 HsaNodeProperties* NodeProperties);
HSAKMT_STATUS HSAKMTAPI vhsaKmtGetXNACKMode(HSAint32* enable);
HSAKMT_STATUS HSAKMTAPI vhsaKmtRuntimeEnable(void* rDebug, bool setupTtmp);
HSAKMT_STATUS HSAKMTAPI vhsaKmtRuntimeDisable(void);
HSAKMT_STATUS HSAKMTAPI vhsaKmtGetNodeMemoryProperties(HSAuint32 NodeId, HSAuint32 NumBanks,
                                                       HsaMemoryProperties* MemoryProperties);
HSAKMT_STATUS HSAKMTAPI vhsaKmtGetNodeCacheProperties(HSAuint32 NodeId, HSAuint32 ProcessorId,
                                                      HSAuint32 NumCaches,
                                                      HsaCacheProperties* CacheProperties);
HSAKMT_STATUS HSAKMTAPI vhsaKmtGetNodeIoLinkProperties(HSAuint32 NodeId, HSAuint32 NumIoLinks,
                                                       HsaIoLinkProperties* IoLinkProperties);
HSAKMT_STATUS HSAKMTAPI vhsaKmtGetClockCounters(HSAuint32 NodeId, HsaClockCounters* Counters);
HSAKMT_STATUS HSAKMTAPI vhsaKmtGetAMDGPUDeviceHandle(HSAuint32 NodeId,
                                                     HsaAMDGPUDeviceHandle* DeviceHandle);
HSAKMT_STATUS HSAKMTAPI vhsaKmtQueryPointerInfo(const void* Pointer, HsaPointerInfo* PointerInfo);
HSAKMT_STATUS HSAKMTAPI vhsaKmtGetTileConfig(HSAuint32 NodeId, HsaGpuTileConfig* config);
HSAKMT_STATUS HSAKMTAPI vhsaKmtCreateEvent(HsaEventDescriptor* EventDesc, _Bool ManualReset,
                                           _Bool IsSignaled, HsaEvent** Event);
HSAKMT_STATUS HSAKMTAPI vhsaKmtDestroyEvent(HsaEvent* Event);
HSAKMT_STATUS HSAKMTAPI vhsaKmtSetEvent(HsaEvent* Event);
HSAKMT_STATUS HSAKMTAPI vhsaKmtResetEvent(HsaEvent* Event);
HSAKMT_STATUS HSAKMTAPI vhsaKmtQueryEventState(HsaEvent* Event);
HSAKMT_STATUS HSAKMTAPI vhsaKmtWaitOnMultipleEvents(HsaEvent* Events[], HSAuint32 NumEvents,
                                                    bool WaitOnAll, HSAuint32 Milliseconds);
HSAKMT_STATUS HSAKMTAPI vhsaKmtWaitOnEvent(HsaEvent* Event, HSAuint32 Milliseconds);
HSAKMT_STATUS HSAKMTAPI vhsaKmtWaitOnEvent_Ext(HsaEvent* Event, HSAuint32 Milliseconds,
                                               uint64_t* event_age);
HSAKMT_STATUS HSAKMTAPI vhsaKmtWaitOnMultipleEvents_Ext(HsaEvent* Events[], HSAuint32 NumEvents,
                                                        bool WaitOnAll, HSAuint32 Milliseconds,
                                                        uint64_t* event_age);
HSAKMT_STATUS HSAKMTAPI vhsaKmtSetTrapHandler(HSAuint32 NodeId, void* TrapHandlerBaseAddress,
                                              HSAuint64 TrapHandlerSizeInBytes,
                                              void* TrapBufferBaseAddress,
                                              HSAuint64 TrapBufferSizeInBytes);
HSAKMT_STATUS HSAKMTAPI vhsaKmtCreateQueueExt(HSAuint32 NodeId, HSA_QUEUE_TYPE Type,
                                              HSAuint32 QueuePercentage,
                                              HSA_QUEUE_PRIORITY Priority, HSAuint32 SdmaEngineId,
                                              void* QueueAddress, HSAuint64 QueueSizeInBytes,
                                              HsaEvent* Event, HsaQueueResource* QueueResource);
HSAKMT_STATUS HSAKMTAPI vhsaKmtCreateQueue(HSAuint32 NodeId, HSA_QUEUE_TYPE Type,
                                           HSAuint32 QueuePercentage, HSA_QUEUE_PRIORITY Priority,
                                           void* QueueAddress, HSAuint64 QueueSizeInBytes,
                                           HsaEvent* Event, HsaQueueResource* QueueResource);
HSAKMT_STATUS HSAKMTAPI vhsaKmtDestroyQueue(HSA_QUEUEID QueueId);
HSAKMT_STATUS HSAKMTAPI vhsaKmtUpdateQueue(HSA_QUEUEID QueueId, HSAuint32 QueuePercentage,
                                           HSA_QUEUE_PRIORITY Priority, void* QueueAddress,
                                           HSAuint64 QueueSize, HsaEvent* Event);
HSAKMT_STATUS HSAKMTAPI vhsaKmtGetQueueInfo(HSA_QUEUEID QueueId, HsaQueueInfo* QueueInfo);
HSAKMT_STATUS HSAKMTAPI vhsaKmtSetQueueCUMask(HSA_QUEUEID QueueId, HSAuint32 CUMaskCount,
                                              HSAuint32* QueueCUMask);
HSAKMT_STATUS HSAKMTAPI vhsaKmtAllocQueueGWS(HSA_QUEUEID QueueId, HSAuint32 nGWS,
                                             HSAuint32* firstGWS);
HSAKMT_STATUS HSAKMTAPI vhsaKmtRegisterGraphicsHandleToNodesExt(
    HSAuint64 GraphicsResourceHandle, HsaGraphicsResourceInfo* GraphicsResourceInfo,
    HSAuint64 NumberOfNodes, HSAuint32* NodeArray, HSA_REGISTER_MEM_FLAGS RegisterFlags);
HSAKMT_STATUS HSAKMTAPI vhsaKmtRegisterGraphicsHandleToNodes(
    HSAuint64 GraphicsResourceHandle, HsaGraphicsResourceInfo* GraphicsResourceInfo,
    HSAuint64 NumberOfNodes, HSAuint32* NodeArray);
HSAKMT_STATUS HSAKMTAPI vhsaKmtMapGraphicHandle(HSAuint32 NodeId, HSAuint64 GraphicDeviceHandle,
                                                HSAuint64 GraphicResourceHandle,
                                                HSAuint64 GraphicResourceOffset,
                                                HSAuint64 GraphicResourceSize,
                                                HSAuint64* FlatMemoryAddress);
HSAKMT_STATUS HSAKMTAPI vhsaKmtUnmapGraphicHandle(HSAuint32 NodeId, HSAuint64 FlatMemoryAddress,
                                                  HSAuint64 SizeInBytes);
HSAKMT_STATUS HSAKMTAPI vhsaKmtExportDMABufHandle(void* MemoryAddress, HSAuint64 MemorySizeInBytes,
                                                  int* DMABufFd, HSAuint64* Offset);
HSAKMT_STATUS HSAKMTAPI vhsaKmtGetRuntimeCapabilities(HSAuint32* caps_mask);
HSAKMT_STATUS HSAKMTAPI vhsaKmtModelEnabled(bool* enable);
HSAKMT_STATUS HSAKMTAPI vhsaKmtOpenSMI(HSAuint32 NodeId, int* fd);
HSAKMT_STATUS HSAKMTAPI vhsaKmtSetXNACKMode(HSAint32 enable);
HSAKMT_STATUS HSAKMTAPI vhsaKmtShareMemory(void* MemoryAddress, HSAuint64 SizeInBytes,
                                           HsaSharedMemoryHandle* SharedMemoryHandle);
HSAKMT_STATUS HSAKMTAPI vhsaKmtRegisterSharedHandleToNodes(
    const HsaSharedMemoryHandle* SharedMemoryHandle, void** MemoryAddress, HSAuint64* SizeInBytes,
    HSAuint64 NumberOfNodes, HSAuint32* NodeArray);
HSAKMT_STATUS HSAKMTAPI vhsaKmtRegisterSharedHandle(const HsaSharedMemoryHandle* SharedMemoryHandle,
                                                    void** MemoryAddress, HSAuint64* SizeInBytes);
HSAKMT_STATUS HSAKMTAPI vhsaKmtSetMemoryUserData(const void* Pointer, void* UserData);
HSAKMT_STATUS HSAKMTAPI vhsaKmtSetMemoryPolicy(HSAuint32 Node, HSAuint32 DefaultPolicy,
                                               HSAuint32 AlternatePolicy,
                                               void* MemoryAddressAlternate,
                                               HSAuint64 MemorySizeInBytes);
HSAKMT_STATUS HSAKMTAPI vhsaKmtSVMGetAttr(void* start_addr, HSAuint64 size, unsigned int nattr,
                                          HSA_SVM_ATTRIBUTE* attrs);
HSAKMT_STATUS HSAKMTAPI vhsaKmtSVMSetAttr(void* start_addr, HSAuint64 size, unsigned int nattr,
                                          HSA_SVM_ATTRIBUTE* attrs);
HSAKMT_STATUS HSAKMTAPI vhsaKmtReplaceAsanHeaderPage(void* addr);
HSAKMT_STATUS HSAKMTAPI vhsaKmtReturnAsanHeaderPage(void* addr);
HSAKMT_STATUS HSAKMTAPI vhsaKmtSPMAcquire(HSAuint32 PreferredNode);
HSAKMT_STATUS HSAKMTAPI vhsaKmtSPMRelease(HSAuint32 PreferredNode);
HSAKMT_STATUS HSAKMTAPI vhsaKmtSPMSetDestBuffer(HSAuint32 PreferredNode, HSAuint32 SizeInBytes,
                                                HSAuint32* timeout, HSAuint32* SizeCopied,
                                                void* DestMemoryAddress, bool* isSPMDataLoss);
HSAKMT_STATUS HSAKMTAPI vhsaKmtAisReadWriteFile(void* MemoryAddress, HSAuint64 MemorySizeInBytes,
                                                HSAint32 fd, HSAint64 file_offset,
                                                HsaAisFlags AisFlags, HSAuint64* SizeCopiedInBytes,
                                                HSAint32* status);
HSAKMT_STATUS HSAKMTAPI vhsaKmtProcessVMRead(HSAuint32 Pid, HsaMemoryRange* LocalMemoryArray,
                                             HSAuint64 LocalMemoryArrayCount,
                                             HsaMemoryRange* RemoteMemoryArray,
                                             HSAuint64 RemoteMemoryArrayCount,
                                             HSAuint64* SizeCopied);
HSAKMT_STATUS HSAKMTAPI vhsaKmtProcessVMWrite(HSAuint32 Pid, HsaMemoryRange* LocalMemoryArray,
                                              HSAuint64 LocalMemoryArrayCount,
                                              HsaMemoryRange* RemoteMemoryArray,
                                              HSAuint64 RemoteMemoryArrayCount,
                                              HSAuint64* SizeCopied);

int vamdgpu_query_gpu_info(amdgpu_device_handle dev, void* out);
int vamdgpu_device_initialize(int fd, uint32_t* major_version, uint32_t* minor_version,
                             amdgpu_device_handle* device_handle);
int vamdgpu_device_deinitialize(amdgpu_device_handle device_handle);
int vamdgpu_device_get_fd(amdgpu_device_handle device_handle);
int vdrmCommandWriteRead(int fd, unsigned long drmCommandIndex, void* data, unsigned long size);
int vamdgpu_bo_cpu_map(amdgpu_bo_handle buf_handle, void** cpu);
int vamdgpu_bo_free(amdgpu_bo_handle buf_handle);
int vamdgpu_bo_export(amdgpu_bo_handle buf_handle, enum amdgpu_bo_handle_type type,
                     uint32_t* shared_handle);
int vamdgpu_bo_import(amdgpu_device_handle dev, enum amdgpu_bo_handle_type type,
                     uint32_t shared_handle, struct amdgpu_bo_import_result* output);
int vamdgpu_bo_va_op(amdgpu_bo_handle bo, uint64_t offset, uint64_t size, uint64_t addr,
                    uint64_t flags, uint32_t ops);
int vamdgpu_bo_query_info(amdgpu_bo_handle bo, struct amdgpu_bo_info* info);
int vamdgpu_bo_set_metadata(amdgpu_bo_handle bo, struct amdgpu_bo_metadata* info);

#ifdef __cplusplus
}
#endif

#endif /* HSAKMT_VIRTIO_H */
