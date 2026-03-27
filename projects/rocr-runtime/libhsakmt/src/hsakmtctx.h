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

#ifndef _HSAKMTCTX_H_
#define _HSAKMTCTX_H_

#include "hsakmt/hsakmttypes.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _HsaKFDContext HsaKFDContext;

/**
  The context-aware version for openning the kfd device.

  "Opens" the HSA kernel driver for user-kernel mode communication.

  On Windows, this function gets a handle to the KFD's AMDKFDIO device object that
  is responsible for user-kernel communication, this handle is used internally by
  the thunk library to send device I/O control to the HSA kernel driver.
  No other thunk library function may be called unless the user-kernel communication
  channel is opened first.

  On Linux this call opens the "/dev/kfd" device file to establish a communication
  path to the kernel.
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtOpenKFDCtx(
    HsaKFDContext **pCtx   //IN/OUT
    );

/**
  The context-aware version for closing the kfd device.

  "Closes" the user-kernel communication path.

  On Windows, the handle obtained by the hsaKmtOpenKFDCtx() function is closed;
  no other communication with the kernel driver is possible after the successful
  execution of the hsaKmtCloseKFDCtx() function. Depending on the failure reason,
  the user-kernel communication path may or may not be still active.

  On Linux the function closes the "dev/kfd" device file.
  No further communication to the kernel driver is allowed until hsaKmtOpenKFDCtx()
  function is called again.
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtCloseKFDCtx( void );

/**
  The function takes a "snapshot" of the topology information within the KFD
  to avoid any changes during the enumeration process.
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtAcquireSystemPropertiesCtx(
    HsaKFDContext         *ctx,               //IN
    HsaSystemProperties*  SystemProperties    //OUT
    );

/**
  Releases the topology "snapshot" taken by hsaKmtAcquireSystemProperties()
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtReleaseSystemPropertiesCtx(
    HsaKFDContext         *ctx              //IN
    );

/**
  Retrieves the discoverable sub-properties for a given HSA
  node. The parameters returned allow the application or runtime to size the
  management structures necessary to store the information.
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtGetNodePropertiesCtx(
    HsaKFDContext          *ctx,              //IN
    HSAuint32              NodeId,            //IN
    HsaNodeProperties*     NodeProperties     //OUT
    );

/**
  Retrieves the memory properties of a specific HSA node.
  the memory pointer passed as MemoryProperties is sized as
  NumBanks * sizeof(HsaMemoryProperties). NumBanks is retrieved with the
  hsaKmtGetNodePropertiesCtx() call.

  Some of the data returned is optional. Not all implementations may return all
  parameters in the hsaMemoryProperties.
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtGetNodeMemoryPropertiesCtx(
    HsaKFDContext         *ctx,               //IN
    HSAuint32             NodeId,             //IN
    HSAuint32             NumBanks,           //IN
    HsaMemoryProperties*  MemoryProperties    //OUT
    );

/**
  Retrieves the cache properties of a specific HSA node and processor ID.
  ProcessorID refers to either a CPU core or a SIMD unit as enumerated earlier
  via the hsaKmtGetNodePropertiesCtx() call.
  The memory pointer passed as CacheProperties is sized as
  NumCaches * sizeof(HsaCacheProperties). NumCaches is retrieved with the
  hsaKmtGetNodePropertiesCtx() call.

  The data returned is optional. Not all implementations may return all
  parameters in the CacheProperties.
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtGetNodeCachePropertiesCtx(
    HsaKFDContext       *ctx,           //IN
    HSAuint32           NodeId,         //IN
    HSAuint32           ProcessorId,    //IN
    HSAuint32           NumCaches,      //IN
    HsaCacheProperties* CacheProperties //OUT
    );

/**
  Retrieves the HSA IO affinity properties of a specific HSA node.
  the memory pointer passed as Properties is sized as
  NumIoLinks * sizeof(HsaIoLinkProperties). NumIoLinks is retrieved with the
  hsaKmtGetNodePropertiesCtx() call.

  The data returned is optional. Not all implementations may return all
  parameters in the IoLinkProperties.
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtGetNodeIoLinkPropertiesCtx(
    HsaKFDContext        *ctx,              //IN
    HSAuint32            NodeId,            //IN
    HSAuint32            NumIoLinks,        //IN
    HsaIoLinkProperties* IoLinkProperties   //OUT
    );


/**
  Creates an operating system event associated with a HSA event ID
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtCreateEventCtx(
    HsaKFDContext       *ctx,           //IN
    HsaEventDescriptor* EventDesc,      //IN
    bool                ManualReset,    //IN
    bool                IsSignaled,     //IN
    HsaEvent**          Event           //OUT
    );

/**
  Destroys an operating system event associated with a HSA event ID
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtDestroyEventCtx(
    HsaKFDContext       *ctx,    //IN
    HsaEvent*           Event    //IN
    );

/**
  Sets the specified event object to the signaled state
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtSetEventCtx(
    HsaKFDContext       *ctx,    //IN
    HsaEvent*           Event    //IN
    );

/**
  Sets the specified event object to the non-signaled state
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtResetEventCtx(
    HsaKFDContext       *ctx,    //IN
    HsaEvent*           Event    //IN
    );

/**
  Queries the state of the specified event object
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtQueryEventStateCtx(
    HsaKFDContext       *ctx,    //IN
    HsaEvent*           Event    //IN
    );

/**
  Checks the current state of the event object. If the object's state is
  nonsignaled, the calling thread enters the wait state.

 The function returns when one of the following occurs:
- The specified event object is in the signaled state.
- The time-out interval elapses.
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtWaitOnEventCtx(
    HsaKFDContext       *ctx,           //IN
    HsaEvent*           Event,          //IN
    HSAuint32           Milliseconds    //IN
    );

/**
  Checks the current state of the event object. If the object's state is
  nonsignaled, the calling thread enters the wait state. event_age can
  help avoiding race conditions.

 The function returns when one of the following occurs:
- The specified event object is in the signaled state.
- The time-out interval elapses.
- Tracking event age
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtWaitOnEvent_ExtCtx(
    HsaKFDContext       *ctx,           //IN
    HsaEvent*           Event,          //IN
    HSAuint32           Milliseconds,   //IN
    uint64_t            *event_age      //IN/OUT
    );

/**
  Checks the current state of multiple event objects.

 The function returns when one of the following occurs:
- Either any one or all of the specified objects are in the signaled state
  - if "WaitOnAll" is "true" the function returns when the state of all
    objects in array is signaled
  - if "WaitOnAll" is "false" the function returns when the state of any
    one of the objects is set to signaled
- The time-out interval elapses.
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtWaitOnMultipleEventsCtx(
    HsaKFDContext       *ctx,           //IN
    HsaEvent*           Events[],       //IN
    HSAuint32           NumEvents,      //IN
    bool                WaitOnAll,      //IN
    HSAuint32           Milliseconds    //IN
    );

/**
  Checks the current state of multiple event objects.
  event_age can help avoiding race conditions.

 The function returns when one of the following occurs:
- Either any one or all of the specified objects are in the signaled state
  - if "WaitOnAll" is "true" the function returns when the state of all
    objects in array is signaled
  - if "WaitOnAll" is "false" the function returns when the state of any
    one of the objects is set to signaled
- The time-out interval elapses.
- Tracking event age
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtWaitOnMultipleEvents_ExtCtx(
    HsaKFDContext       *ctx,           //IN
    HsaEvent*           Events[],       //IN
    HSAuint32           NumEvents,      //IN
    bool                WaitOnAll,      //IN
    HSAuint32           Milliseconds,   //IN
    uint64_t            *event_age      //IN/OUT
    );

/**
  Creates a GPU queue with user-mode access rights
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtCreateQueueCtx(
    HsaKFDContext       *ctx,             //IN
    HSAuint32           NodeId,           //IN
    HSA_QUEUE_TYPE      Type,             //IN
    HSAuint32           QueuePercentage,  //IN
    HSA_QUEUE_PRIORITY  Priority,         //IN
    void*               QueueAddress,     //IN
    HSAuint64           QueueSizeInBytes, //IN
    HsaEvent*           Event,            //IN
    HsaQueueResource*   QueueResource     //OUT
    );

/**
  Creates a GPU queue with user-mode access rights
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtCreateQueueExtCtx(
    HsaKFDContext       *ctx,             //IN
    HSAuint32           NodeId,           //IN
    HSA_QUEUE_TYPE      Type,             //IN
    HSAuint32           QueuePercentage,  //IN
    HSA_QUEUE_PRIORITY  Priority,         //IN
    HSAuint32           SdmaEngineId,     //IN
    void*               QueueAddress,     //IN
    HSAuint64           QueueSizeInBytes, //IN
    HsaEvent*           Event,            //IN
    HsaQueueResource*   QueueResource     //OUT
    );

/**
  Updates a queue
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtUpdateQueueCtx(
    HsaKFDContext       *ctx,             //IN
    HSA_QUEUEID         QueueId,          //IN
    HSAuint32           QueuePercentage,  //IN
    HSA_QUEUE_PRIORITY  Priority,         //IN
    void*               QueueAddress,     //IN
    HSAuint64           QueueSize,        //IN
    HsaEvent*           Event             //IN
    );

/**
  Destroys a queue
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtDestroyQueueCtx(
    HsaKFDContext       *ctx,           //IN
    HSA_QUEUEID         QueueId         //IN
    );

/**
  Set cu mask for a queue
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtSetQueueCUMaskCtx(
    HsaKFDContext       *ctx,           //IN
    HSA_QUEUEID         QueueId,        //IN
    HSAuint32           CUMaskCount,    //IN
    HSAuint32*          QueueCUMask     //IN
    );

HSAKMT_STATUS
HSAKMTAPI
hsaKmtGetQueueInfoCtx(
    HsaKFDContext       *ctx,           //IN
    HSA_QUEUEID         QueueId,        //IN
    HsaQueueInfo        *QueueInfo      //IN
    );

/**
  Allows an HSA process to set/change the default and alternate memory coherency, before starting to dispatch.
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtSetMemoryPolicyCtx(
    HsaKFDContext      *ctx,                      //IN
    HSAuint32          Node,                      //IN
    HSAuint32          DefaultPolicy,             //IN
    HSAuint32          AlternatePolicy,           //IN
    void*              MemoryAddressAlternate,    //IN (page-aligned)
    HSAuint64          MemorySizeInBytes          //IN (page-aligned)
    );

/**
  Allocates a memory buffer that may be accessed by the GPU
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtAllocMemoryCtx(
    HsaKFDContext     *ctx,                   //IN
    HSAuint32         PreferredNode,          //IN
    HSAuint64         SizeInBytes,            //IN  (multiple of page size)
    HsaMemFlags       MemFlags,               //IN
    void**            MemoryAddress           //IN/OUT (page-aligned)
    );

/**
  Allocates a memory buffer with specific alignment that may be accessed by the GPU
  If Alignment is 0, the smallest possible alignment will be used
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtAllocMemoryAlignCtx(
    HsaKFDContext     *ctx,                  //IN
    HSAuint32         PreferredNode,          //IN
    HSAuint64         SizeInBytes,            //IN  (multiple of page size)
    HSAuint64         Alignment,              //IN  (power of 2 and >= page size)
    HsaMemFlags       MemFlags,               //IN
    void**            MemoryAddress           //IN/OUT (page-aligned)
    );

/**
  Frees a memory buffer
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtFreeMemoryCtx(
    HsaKFDContext     *ctx,                 //IN
    void*             MemoryAddress,        //IN (page-aligned)
    HSAuint64         SizeInBytes           //IN
    );

/**
  Inquires memory available for allocation as a memory buffer
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtAvailableMemoryCtx(
    HsaKFDContext     *ctx,                //IN
    HSAuint32         Node,                //IN
    HSAuint64         *AvailableBytes      //OUT
    );

/**
  Registers with KFD a memory buffer that may be accessed by the GPU
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtRegisterMemoryCtx(
    HsaKFDContext     *ctx,               //IN
    void*             MemoryAddress,      //IN (cache-aligned)
    HSAuint64         MemorySizeInBytes   //IN (cache-aligned)
    );


/**
  Registers with KFD a memory buffer that may be accessed by specific GPUs
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtRegisterMemoryToNodesCtx(
    HsaKFDContext    *ctx,               //IN
    void             *MemoryAddress,     //IN (cache-aligned)
    HSAuint64        MemorySizeInBytes,  //IN (cache-aligned)
    HSAuint64        NumberOfNodes,      //IN
    HSAuint32*       NodeArray           //IN
    );


/**
  Registers with KFD a memory buffer with memory attributes
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtRegisterMemoryWithFlagsCtx(
    HsaKFDContext    *ctx,                //IN
    void             *MemoryAddress,      //IN (cache-aligned)
    HSAuint64        MemorySizeInBytes,   //IN (cache-aligned)
    HsaMemFlags      MemFlags             //IN
    );

/**
  Registers with KFD a graphics buffer and returns graphics metadata
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtRegisterGraphicsHandleToNodesCtx(
    HsaKFDContext             *ctx,                         //IN
    HSAuint64                 GraphicsResourceHandle,       //IN
    HsaGraphicsResourceInfo   *GraphicsResourceInfo,        //OUT
    HSAuint64                 NumberOfNodes,                //IN
    HSAuint32*                NodeArray                     //IN
    );

/**
  Similar to hsaKmtRegisterGraphicsHandleToNodes but provides registration
  options via RegisterFlags.
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtRegisterGraphicsHandleToNodesExtCtx(
    HsaKFDContext           *ctx,                         //IN
    HSAuint64               GraphicsResourceHandle,       //IN
    HsaGraphicsResourceInfo *GraphicsResourceInfo,        //OUT
    HSAuint64               NumberOfNodes,                //IN
    HSAuint32*              NodeArray,                    //IN
    HSA_REGISTER_MEM_FLAGS  RegisterFlags                 //IN
    );

/**
 * Export a dmabuf handle and offset for a given memory address
 *
 * Validates that @MemoryAddress belongs to a valid allocation and that the
 * @MemorySizeInBytes doesn't exceed the end of that allocation. Returns a
 * dmabuf fd of the allocation and the offset of MemoryAddress within that
 * allocation. The memory will remain allocated even after the allocation is
 * freed by hsaKmtFreeMemory for as long as a dmabuf fd remains open or any
 * importer of that fd maintains an active reference to the memory.
 */

HSAKMT_STATUS
HSAKMTAPI
hsaKmtExportDMABufHandleCtx(
    HsaKFDContext     *ctx,               //IN
    void              *MemoryAddress,     //IN
    HSAuint64         MemorySizeInBytes,  //IN
    int               *DMABufFd,          //OUT
    HSAuint64         *Offset             //OUT
    );

/**
 Export a memory buffer for sharing with other processes

 NOTE: for the current revision of the thunk spec, SizeInBytes
 must match whole allocation.
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtShareMemoryCtx(
    HsaKFDContext         *ctx,               //IN
    void                  *MemoryAddress,     //IN
    HSAuint64             SizeInBytes,        //IN
    HsaSharedMemoryHandle *SharedMemoryHandle //OUT
);

/**
 Register shared memory handle
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtRegisterSharedHandleCtx(
    HsaKFDContext               *ctx,                //IN
    const HsaSharedMemoryHandle *SharedMemoryHandle, //IN
    void                        **MemoryAddress,     //OUT
    HSAuint64                   *SizeInBytes         //OUT
);

/**
 Register shared memory handle to specific nodes only
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtRegisterSharedHandleToNodesCtx(
    HsaKFDContext               *ctx,                //IN
    const HsaSharedMemoryHandle *SharedMemoryHandle, //IN
    void                        **MemoryAddress,     //OUT
    HSAuint64                   *SizeInBytes,        //OUT
    HSAuint64                   NumberOfNodes,       //OUT
    HSAuint32*                  NodeArray            //OUT
);

/**
  Unregisters with KFD a memory buffer
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtDeregisterMemoryCtx(
    HsaKFDContext     *ctx,           //IN
    void*             MemoryAddress   //IN
    );

/**
  Ensures that the memory is resident and can be accessed by GPU
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtMapMemoryToGPUCtx(
    HsaKFDContext     *ctx,              //IN
    void*             MemoryAddress,     //IN (page-aligned)
    HSAuint64         MemorySizeInBytes, //IN (page-aligned)
    HSAuint64*        AlternateVAGPU     //OUT (page-aligned)
    );

/**
  Ensures that the memory is resident and can be accessed by GPUs
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtMapMemoryToGPUNodesCtx(
    HsaKFDContext     *ctx,                  //IN
    void*             MemoryAddress,         //IN (page-aligned)
    HSAuint64         MemorySizeInBytes,     //IN (page-aligned)
    HSAuint64*        AlternateVAGPU,        //OUT (page-aligned)
    HsaMemMapFlags    MemMapFlags,           //IN
    HSAuint64         NumberOfNodes,         //IN
    HSAuint32*        NodeArray              //IN
    );

/**
  Releases the residency of the memory
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtUnmapMemoryToGPUCtx(
    HsaKFDContext     *ctx,              //IN
    void*             MemoryAddress      //IN (page-aligned)
    );

/**
  Stub for Unmap Graphic Handle
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtUnmapGraphicHandleCtx(
    HsaKFDContext      *ctx,                  //IN
    HSAuint32          NodeId,                //IN
    HSAuint64          FlatMemoryAddress,     //IN
    HSAuint64          SizeInBytes            //IN
    );

/**
 * Get an AMDGPU device handle for a GPU node
 */
HSAKMT_STATUS
HSAKMTAPI
hsaKmtGetAMDGPUDeviceHandleCtx(
    HsaKFDContext           *ctx,          //IN
    HSAuint32               NodeId,        //IN
    HsaAMDGPUDeviceHandle   *DeviceHandle  //OUT
    );

/**
  Sets trap handler and trap buffer to be used for all queues
  associated with the specified NodeId within this process context
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtSetTrapHandlerCtx(
    HsaKFDContext      *ctx,                     //IN
    HSAuint32          NodeId,                   //IN
    void*              TrapHandlerBaseAddress,   //IN
    HSAuint64          TrapHandlerSizeInBytes,   //IN
    void*              TrapBufferBaseAddress,    //IN
    HSAuint64          TrapBufferSizeInBytes     //IN
    );

/**
  Gets image tile configuration.
 */
HSAKMT_STATUS
HSAKMTAPI
hsaKmtGetTileConfigCtx(
    HsaKFDContext       *ctx,       //IN
    HSAuint32           NodeId,     //IN
    HsaGpuTileConfig    *config     //IN/OUT
    );

/**
  Returns information about pointers
*/
HSAKMT_STATUS
HSAKMTAPI
hsaKmtQueryPointerInfoCtx(
    HsaKFDContext     *ctx,            //IN
    const void        *Pointer,        //IN
    HsaPointerInfo    *PointerInfo     //OUT
    );

/**
  Associates user data with a memory allocation
*/
HSAKMT_STATUS
HSAKMTAPI
hsaKmtSetMemoryUserDataCtx(
    HsaKFDContext    *ctx,       //IN
    const void *     Pointer,    //IN
    void *           UserData    //IN
    );

/**
  Allocate GWS resource for a queue
 */

HSAKMT_STATUS
HSAKMTAPI
hsaKmtAllocQueueGWSCtx(
    HsaKFDContext      *ctx,           //IN
    HSA_QUEUEID        QueueId,        //IN
    HSAuint32          nGWS,           //IN
    HSAuint32          *firstGWS       //OUT
    );

HSAKMT_STATUS
HSAKMTAPI
hsaKmtRuntimeEnableCtx(
    HsaKFDContext      *ctx,           //IN
    void*              rDebug,         //IN
    bool               setupTtmp       //IN
    );

HSAKMT_STATUS
HSAKMTAPI
hsaKmtRuntimeDisableCtx(
    HsaKFDContext      *ctx           //IN
    );

HSAKMT_STATUS
HSAKMTAPI
hsaKmtGetRuntimeCapabilitiesCtx(
    HsaKFDContext      *ctx,           //IN
    HSAuint32	         *caps_mask      //OUT
    );

/**
  Enable debug trap.
*/
HSAKMT_STATUS
HSAKMTAPI
hsaKmtDbgEnableCtx(
    HsaKFDContext      *ctx,           //IN
    void               **runtime_info, //Out
    HSAuint32          *data_size      //Out
    );

/**
  Disable debug trap.
*/
HSAKMT_STATUS
HSAKMTAPI
hsaKmtDbgDisableCtx(
    HsaKFDContext      *ctx          //IN
    );

/**
  Get device snapshot.
*/
HSAKMT_STATUS
HSAKMTAPI
hsaKmtDbgGetDeviceDataCtx(
    HsaKFDContext     *ctx,          //IN
    void              **data,        //Out
    HSAuint32         *n_entries,    //Out
    HSAuint32         *entry_size    //Out
    );

/**
  Get queues snapshot.
*/
HSAKMT_STATUS
HSAKMTAPI
hsaKmtDbgGetQueueDataCtx(
    HsaKFDContext     *ctx,           //IN
    void              **data,         //Out
    HSAuint32         *n_entries,     //Out
    HSAuint32         *entry_size,    //Out
    bool              suspend_queues  //In
    );

/**
  Check whether gpu firmware and kernel support debugging
*/
HSAKMT_STATUS
HSAKMTAPI
hsaKmtCheckRuntimeDebugSupportCtx(
    HsaKFDContext     *ctx           //IN
    );

/**
  Debug ops call primarily used for KFD testing
 */
HSAKMT_STATUS
HSAKMTAPI
hsaKmtDebugTrapIoctlCtx(
    HsaKFDContext                  *ctx,           //IN
    struct kfd_ioctl_dbg_trap_args *args,          //IN/OUT
    HSA_QUEUEID                    *Queues,        //IN
    HSAuint64                      *DebugReturn    //OUT
    );

/**
  Gets GPU and CPU clock counters for particular Node
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtGetClockCountersCtx(
    HsaKFDContext     *ctx,           //IN
    HSAuint32         NodeId,         //IN
    HsaClockCounters  *Counters);     //OUT

/**
  Retrieves information on the available HSA counters
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtPmcGetCounterPropertiesCtx(
    HsaKFDContext         *ctx,                //IN
    HSAuint32              NodeId,             //IN
    HsaCounterProperties** CounterProperties   //OUT
    );

/**
  Registers a set of (HW) counters to be used for tracing/profiling
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtPmcRegisterTraceCtx(
    HsaKFDContext      *ctx,                //IN
    HSAuint32           NodeId,             //IN
    HSAuint32           NumberOfCounters,   //IN
    HsaCounter*         Counters,           //IN
    HsaPmcTraceRoot*    TraceRoot           //OUT
    );

/**
  Allows a user mode process to get exclusive access to the defined set of (HW) counters
  used for tracing/profiling
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtPmcAcquireTraceAccessCtx(
    HsaKFDContext      *ctx,                //IN
    HSAuint32          NodeId,              //IN
    HSATraceId         TraceId              //IN
    );

/**
  Allows a user mode process to release exclusive access to the defined set of (HW) counters
  used for tracing/profiling
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtPmcReleaseTraceAccessCtx(
    HsaKFDContext      *ctx,                //IN
    HSAuint32          NodeId,              //IN
    HSATraceId         TraceId              //IN
    );

/* Helper functions for calling KFD SVM ioctl */
HSAKMT_STATUS
HSAKMTAPI
hsaKmtSVMSetAttrCtx(
    HsaKFDContext     *ctx,         //IN
    void              *start_addr,  //IN: Start of the virtual address range (page-aligned)
    HSAuint64         size,         //IN: size (page-aligned)
    unsigned int      nattr,        //IN: number of attributes
    HSA_SVM_ATTRIBUTE *attrs        //IN: array of attributes
    );

HSAKMT_STATUS
HSAKMTAPI
hsaKmtSVMGetAttrCtx(
    HsaKFDContext     *ctx,         //IN
    void              *start_addr,  //IN: Start of the virtual address range (page-aligned)
    HSAuint64         size,         //IN: size (page aligned)
    unsigned int      nattr,        //IN: number of attributes
    HSA_SVM_ATTRIBUTE *attrs        //IN/OUT: array of attributes
    );

HSAKMT_STATUS
HSAKMTAPI
hsaKmtSetXNACKModeCtx(
    HsaKFDContext     *ctx,       //IN
    HSAint32          enable      //IN: enable/disable XNACK node.
    );

HSAKMT_STATUS
HSAKMTAPI
hsaKmtGetXNACKModeCtx(
    HsaKFDContext     *ctx,       //IN
    HSAint32          *enable     //OUT: returns XNACK value.
    );

/**
   Open anonymous file handle to enable events and read SMI events.

   To enable events, write 64bit events mask to fd, event enums as bit index.
   for example, event mask ctx(HSA_SMI_EVENT_MASK_FROM_INDEXCtx(HSA_SMI_EVENT_INDEX_MAX) - 1) to enable all events

   Read event from fd is not blocking, use poll with timeout value to check if event is available.
   Event is dropped if kernel event fifo is full.
*/
HSAKMT_STATUS
HSAKMTAPI
hsaKmtOpenSMICtx(
    HsaKFDContext     *ctx,      //IN
    HSAuint32         NodeId,    //IN: GPU node_id to receive the SMI event from
    int               *fd        //OUT: anonymous file handle
    );

/**
   If this is GPU Mapped memory, remap the first page at this address to be normal system memory

   This is used in ASAN mode to remap the first page of device memory to share host ASAN logic.
   This function is only supported when libhsakmt is compiled in ASAN mode.
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtReplaceAsanHeaderPageCtx(
    HsaKFDContext     *ctx,      //IN
    void              *addr      //IN: Start of the virtual address page
    );

/**
   If this is GPU Mapped memory, remap the first page back to the original GPU memory

   This is used in ASAN mode to remap the first page back to its original mapping.
   This function is only supported when libhsakmt is compiled in ASAN mode.
*/

HSAKMT_STATUS
HSAKMTAPI
hsaKmtReturnAsanHeaderPageCtx(
    HsaKFDContext     *ctx,       //IN
    void              *addr       //IN: Start of the virtual address page
    );

#ifdef __cplusplus
}   //extern "C"
#endif

#endif //_HSAKMTCTX_H_
