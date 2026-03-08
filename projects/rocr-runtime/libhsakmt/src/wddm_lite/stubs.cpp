/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

/*
 * Stub implementations for all unimplemented HSAKMT APIs.
 * Every function returns HSAKMT_STATUS_NOT_SUPPORTED.
 * These will be replaced as features are implemented.
 */

#include "wddm_lite_internal.h"
#include <string.h>

/* ======================================================================
 * Memory management stubs
 * ====================================================================== */

HSAKMT_STATUS HSAKMTAPI
hsaKmtSetMemoryPolicy(HSAuint32 Node, HSAuint32 DefaultPolicy,
                       HSAuint32 AlternatePolicy,
                       void *MemoryAddressAlternate,
                       HSAuint64 MemorySizeInBytes)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtAllocMemory(HSAuint32 Node, HSAuint64 SizeInBytes,
                   HsaMemFlags MemFlags, void **MemoryAddress)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtFreeMemory(void *MemoryAddress, HSAuint64 SizeInBytes)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtAvailableMemory(HSAuint32 Node, HSAuint64 *AvailableBytes)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtRegisterMemory(void *MemoryAddress, HSAuint64 MemorySize)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtRegisterMemoryToNodes(void *MemoryAddress, HSAuint64 MemorySize,
                             HSAuint64 NumberOfNodes,
                             HSAuint32 *NodeArray)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtRegisterMemoryWithFlags(void *MemoryAddress, HSAuint64 MemorySize,
                               HsaMemFlags MemFlags)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtDeregisterMemory(void *MemoryAddress)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtMapMemoryToGPU(void *MemoryAddress, HSAuint64 MemorySize,
                      HSAuint64 *AlternateVAGPU)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtMapMemoryToGPUNodes(void *MemoryAddress, HSAuint64 MemorySize,
                           HSAuint64 *AlternateVAGPU,
                           HsaMemMapFlags MapFlags,
                           HSAuint64 NumberOfNodes,
                           HSAuint32 *NodeArray)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtUnmapMemoryToGPU(void *MemoryAddress)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtRegisterGraphicsHandleToNodes(HSAuint64 GraphicsResourceHandle,
                                     HsaGraphicsResourceInfo *GraphicsResourceInfo,
                                     HSAuint64 NumberOfNodes,
                                     HSAuint32 *NodeArray)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtShareMemory(void *MemoryAddress, HSAuint64 SizeInBytes,
                   HsaSharedMemoryHandle *SharedMemoryHandle)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtRegisterSharedHandle(const HsaSharedMemoryHandle *SharedMemoryHandle,
                            void **MemoryAddress, HSAuint64 *SizeInBytes)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtRegisterSharedHandleToNodes(const HsaSharedMemoryHandle *SharedMemoryHandle,
                                   void **MemoryAddress, HSAuint64 *SizeInBytes,
                                   HSAuint64 NumberOfNodes,
                                   HSAuint32 *NodeArray)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtProcessVMRead(HSAuint32 Pid, HsaMemoryRange *LocalMemoryArray,
                     HSAuint64 LocalMemoryArrayCount,
                     HsaMemoryRange *RemoteMemoryArray,
                     HSAuint64 RemoteMemoryArrayCount,
                     HSAuint64 *SizeCopied)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtProcessVMWrite(HSAuint32 Pid, HsaMemoryRange *LocalMemoryArray,
                      HSAuint64 LocalMemoryArrayCount,
                      HsaMemoryRange *RemoteMemoryArray,
                      HSAuint64 RemoteMemoryArrayCount,
                      HSAuint64 *SizeCopied)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtExportDMABufHandle(void *MemoryAddress, HSAuint64 MemorySize,
                          int *DMABufFd, HSAuint64 *Offset)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtQueryPointerInfo(const void *Pointer,
                        HsaPointerInfo *PointerInfo)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSetMemoryUserData(const void *Pointer, void *UserData)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

/* ======================================================================
 * Queue management stubs
 * ====================================================================== */

HSAKMT_STATUS HSAKMTAPI
hsaKmtCreateQueue(HSAuint32 NodeId, HSA_QUEUE_TYPE Type,
                   HSAuint32 QueuePercentage,
                   HSA_QUEUE_PRIORITY Priority,
                   void *QueueAddress, HSAuint64 QueueSizeInBytes,
                   HsaEvent *Event, HsaQueueResource *QueueResource)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtUpdateQueue(HSA_QUEUEID QueueId, HSAuint32 QueuePercentage,
                   HSA_QUEUE_PRIORITY Priority,
                   void *QueueAddress, HSAuint64 QueueSizeInBytes,
                   HsaEvent *Event)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtDestroyQueue(HSA_QUEUEID QueueId)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSetQueueCUMask(HSA_QUEUEID QueueId, HSAuint32 CUMaskCount,
                      HSAuint32 *QueueCUMask)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetQueueInfo(HSA_QUEUEID QueueId, HsaQueueInfo *QueueInfo)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtAllocQueueGWS(HSA_QUEUEID QueueId, HSAuint32 nGWS,
                     HSAuint32 *FirstGWS)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

/* ======================================================================
 * Event stubs
 * ====================================================================== */

HSAKMT_STATUS HSAKMTAPI
hsaKmtCreateEvent(HsaEventDescriptor *EventDesc, bool ManualReset,
                   bool IsSignaled, HsaEvent **Event)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtDestroyEvent(HsaEvent *Event)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSetEvent(HsaEvent *Event)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtResetEvent(HsaEvent *Event)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtQueryEventState(HsaEvent *Event)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtWaitOnEvent(HsaEvent *Event, HSAuint32 Milliseconds)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtWaitOnMultipleEvents(HsaEvent *Events[], HSAuint32 NumEvents,
                            bool WaitOnAll, HSAuint32 Milliseconds)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

/* ======================================================================
 * Debug stubs
 * ====================================================================== */

HSAKMT_STATUS HSAKMTAPI
hsaKmtDbgRegister(HSAuint32 NodeId)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtDbgUnregister(HSAuint32 NodeId)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtDbgWavefrontControl(HSAuint32 NodeId, HSA_DBG_WAVEOP Operand,
                           HSA_DBG_WAVEMODE Mode,
                           HSAuint32 TrapId, HsaDbgWaveMessage *DbgWaveMsgRing)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtDbgAddressWatch(HSAuint32 NodeId, HSAuint32 NumWatchPoints,
                       HSA_DBG_WATCH_MODE WatchMode[],
                       void *WatchAddress[],
                       HSAuint64 WatchMask[],
                       HsaEvent *WatchEvent[])
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtRuntimeEnable(void *rDebug, bool SetupTtmp)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtRuntimeDisable(void)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetRuntimeCapabilities(HSAuint32 *caps_mask)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

/* ======================================================================
 * Performance counter stubs
 * ====================================================================== */

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetClockCounters(HSAuint32 NodeId, HsaClockCounters *Counters)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtPmcGetCounterProperties(HSAuint32 NodeId,
                               HsaCounterProperties **CounterProperties)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtPmcRegisterTrace(HSAuint32 NodeId,
                        HSAuint32 NumberOfCounters,
                        HsaCounter *Counters,
                        HsaPmcTraceRoot *TraceRoot)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtPmcUnregisterTrace(HSAuint32 NodeId, HSATraceId TraceId)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtPmcAcquireTraceAccess(HSAuint32 NodeId, HSATraceId TraceId)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtPmcReleaseTraceAccess(HSAuint32 NodeId, HSATraceId TraceId)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtPmcStartTrace(HSATraceId TraceId, void *TraceBuffer,
                      HSAuint64 TraceBufferSizeBytes)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtPmcQueryTrace(HSATraceId TraceId)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtPmcStopTrace(HSATraceId TraceId)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

/* ======================================================================
 * Trap handler stubs
 * ====================================================================== */

HSAKMT_STATUS HSAKMTAPI
hsaKmtSetTrapHandler(HSAuint32 Node, void *TrapHandlerBaseAddress,
                      HSAuint64 TrapHandlerSizeInBytes,
                      void *TrapBufferBaseAddress,
                      HSAuint64 TrapBufferSizeInBytes)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

/* ======================================================================
 * SVM stubs
 * ====================================================================== */

HSAKMT_STATUS HSAKMTAPI
hsaKmtSetXNACKMode(HSAint32 enable)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetXNACKMode(HSAint32 *enable)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

/* ======================================================================
 * Topology extras stubs
 * ====================================================================== */

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetNodeWallclockFrequency(HSAuint32 NodeId, uint64_t *Freq)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

/* ======================================================================
 * Additional stubs for completeness
 * ====================================================================== */

HSAKMT_STATUS HSAKMTAPI
hsaKmtReportQueue(HSA_QUEUEID QueueId, HsaQueueReport *QueueReport)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtCreateQueueExt(HSAuint32 NodeId, HSA_QUEUE_TYPE Type,
                      HSAuint32 QueuePercentage, HSA_QUEUE_PRIORITY Priority,
                      HSAuint32 SdmaEngineId, void *QueueAddress,
                      HSAuint64 QueueSizeInBytes, HsaEvent *Event,
                      HsaQueueResource *QueueResource)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtQueueRingDoorbell(HSA_QUEUEID QueueId, HSAuint64 value)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtAllocMemoryAlign(HSAuint32 PreferredNode, HSAuint64 SizeInBytes,
                        HSAuint64 Alignment, HsaMemFlags MemFlags,
                        void **MemoryAddress)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtRegisterGraphicsHandleToNodesExt(HSAuint64 GraphicsResourceHandle,
                                        HsaGraphicsResourceInfo *GraphicsResourceInfo,
                                        HSAuint64 NumberOfNodes,
                                        HSAuint32 *NodeArray,
                                        HSA_REGISTER_MEM_FLAGS RegisterFlags)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

#if defined(_WIN32)
HSAKMT_STATUS HSAKMTAPI
hsaKmtGetMemoryHandle(void *va, void *MemoryAddress,
                       HSAuint64 SizeInBytes, uint64_t *SharedMemoryHandle)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}
#endif

HSAKMT_STATUS HSAKMTAPI
hsaKmtWaitOnEvent_Ext(HsaEvent *Event, HSAuint32 Milliseconds,
                       uint64_t *event_age)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtWaitOnMultipleEvents_Ext(HsaEvent *Events[], HSAuint32 NumEvents,
                                bool WaitOnAll, HSAuint32 Milliseconds,
                                uint64_t *event_age)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtDbgEnable(void **runtime_info, HSAuint32 *data_size)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtDbgDisable(void)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtDbgGetDeviceData(void **data, HSAuint32 *n_entries,
                        HSAuint32 *entry_size)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtDbgGetQueueData(void **data, HSAuint32 *n_entries,
                       HSAuint32 *entry_size, bool suspend_queues)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtCheckRuntimeDebugSupport(void)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtDebugTrapIoctl(struct kfd_ioctl_dbg_trap_args *args,
                      HSA_QUEUEID *Queues, HSAuint64 *DebugReturn)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetTileConfig(HSAuint32 NodeId, HsaGpuTileConfig *config)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSPMAcquire(HSAuint32 PreferredNode)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSPMRelease(HSAuint32 PreferredNode)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSPMSetDestBuffer(HSAuint32 PreferredNode, HSAuint32 SizeInBytes,
                        HSAuint32 *timeout, HSAuint32 *SizeCopied,
                        void *DestMemoryAddress, bool *isSPMDataLoss)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSVMSetAttr(void *start_addr, HSAuint64 size,
                  unsigned int nattr, HSA_SVM_ATTRIBUTE *attrs)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSVMGetAttr(void *start_addr, HSAuint64 size,
                  unsigned int nattr, HSA_SVM_ATTRIBUTE *attrs)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtOpenSMI(HSAuint32 NodeId, int *fd)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtReplaceAsanHeaderPage(void *addr)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtReturnAsanHeaderPage(void *addr)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtPcSamplingSupport(void)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtPcSamplingQueryCapabilities(HSAuint32 NodeId, void *sample_info,
                                   HSAuint32 sample_info_sz,
                                   HSAuint32 *sz_needed)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtPcSamplingCreate(HSAuint32 node_id, HsaPcSamplingInfo *sample_info,
                        HsaPcSamplingTraceId *traceId)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtPcSamplingDestroy(HSAuint32 NodeId, HsaPcSamplingTraceId traceId)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtPcSamplingStart(HSAuint32 NodeId, HsaPcSamplingTraceId traceId)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtPcSamplingStop(HSAuint32 NodeId, HsaPcSamplingTraceId traceId)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtAisReadWriteFile(void *MemoryAddress, HSAuint64 MemorySizeInBytes,
                        HSAint32 fd, HSAint64 file_offset,
                        HsaAisFlags AisFlags, HSAuint64 *SizeCopiedInBytes,
                        HSAint32 *status)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtModelEnabled(bool *enable)
{
    if (enable)
        *enable = false;
    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtMapGraphicHandle(HSAuint32 NodeId, HSAuint64 GraphicDeviceHandle,
                        HSAuint64 GraphicResourceHandle,
                        HSAuint64 GraphicResourceOffset,
                        HSAuint64 GraphicResourceSize,
                        HSAuint64 *FlatMemoryAddress)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtUnmapGraphicHandle(HSAuint32 NodeId, HSAuint64 FlatMemoryAddress,
                          HSAuint64 SizeInBytes)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetAMDGPUDeviceHandle(HSAuint32 NodeId,
                             HsaAMDGPUDeviceHandle *DeviceHandle)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtHandleImport(const HsaExternalHandleDesc *ImportDesc,
                    HsaHandleImportResult *ImportResult,
                    HsaHandleImportFlags *Flags)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtMemoryVaMap(HsaMemoryObjectHandle Handle, HSAuint64 offset,
                   HSAuint64 size, HSAuint64 addr, HsaMemoryMapFlags flags)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtMemoryVaUnmap(HsaMemoryObjectHandle Handle, HSAuint64 offset,
                     HSAuint64 size, HSAuint64 addr)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtMemoryCpuMap(HsaMemoryObjectHandle Handle, void **out_cpu_ptr)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtMemHandleFree(HsaMemoryObjectHandle Handle)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtMemoryGetCpuAddr(HsaAMDGPUDeviceHandle DeviceHandle,
                        HsaMemoryObjectHandle MemoryHandle,
                        HSAint32 *fd, HSAuint64 *cpu_addr)
{
    return HSAKMT_STATUS_NOT_SUPPORTED;
}
