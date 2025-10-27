/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2014-2025 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 **********************************************************************************************************************/
/**
 ***********************************************************************************************************************
 * @file  palQueue.h
 * @brief Defines the Platform Abstraction Library (PAL) IQueue interface and related types.
 ***********************************************************************************************************************
 */

#pragma once

#include "pal.h"
#include "palDestroyable.h"
#include "palEvent.h"

namespace Pal
{

// Forward declarations.
class ICmdBuffer;
class IFence;
class IGpuMemory;
class IImage;
class IPrivateScreen;
class IQueueSemaphore;
class IScreen;
class ISwapChain;
struct CmdBufInfo;
struct GpuMemSubAllocInfo;
struct GpuMemoryRef;
struct DoppRef;

enum class VirtualGpuMemAccessMode : uint32;

/// Specifies whether presents are windowed or fullscreen.  This will determine whether the present is performed via a
/// BLT or flip.
enum class PresentMode : uint32
{
    Unknown,
    Windowed,
    Fullscreen,
    Count
};

/// Enumerates the possible overrides for the flip interval.
enum class FlipIntervalOverride : uint32
{
    _None                 = 0, ///< No override.
    Immediate             = 1, ///< Zero frames of flip latency.
    ImmediateAllowTearing = 2, ///< Same as Immediate, but allows tearing (no vsync).
    One                   = 3, ///< One frame of flip latency.
    Two                   = 4, ///< Two frames of flip latency.
    Three                 = 5, ///< Three frames of flip latency.
    Four                  = 6, ///< Four frames of flip latency.
};

/// Defines flags for describing which types of present modes are supported on a given queue.
enum PresentModeSupport : uint32
{
    SupportWindowedPresent          = 0x1,
    SupportWindowedPriorBlitPresent = 0x2,
    SupportFullscreenPresent        = 0x4,
};

/// Defines submit-time bottlenecks which PAL can potentially optimize.
enum class SubmitOptMode : uint32
{
    Default           = 0, ///< PAL will enable optimizations when generally efficient.
    Disabled          = 1, ///< Disable all optimizations that could be detrimental in special cases.
    MinKernelSubmits  = 2, ///< Minimize the overhead of launching command buffers on the CPU and GPU.
    MinGpuCmdOverhead = 3, ///< Minimize the overhead of reading command buffer commands on the GPU.
    Count
};

/// Enumerates vcn instance affinity statuses
enum MmAffinityStatus : uint32
{
    MmAffinityNotAllowed = 0, ///< The specific vcn instance can't be used.
    MmAffinityAllowed    = 1  ///< The specific vcn instance can be used.
};

/// Union describes all vcn instance affinity status.
union MmAffinity
{
    struct
    {
        uint32 vcn0Affinity : 2;  ///< Affinity for instance vcn0
        uint32 vcn1Affinity : 2;  ///< Affinity for instance vcn1
        uint32 reserved     : 28; ///< Reserved (all 0)
    };
    uint32 u32All;
};

/// Structure describing dump information for a command buffer.
struct CmdBufferDumpDesc
{
    EngineType    engineType;       ///< The engine type that this buffer is targeted for.
    QueueType     queueType;        ///< The type of queue that this buffer is being created on.
    SubEngineType subEngineType;    ///< The ID of which sub-engine that this buffer is made for.

    uint32        cmdBufferIdx;     ///< The index into the SubmitInfo ppCmdBuffers array that this
                                    ///  command buffer dump came from.
    union
    {
        struct
        {
            uint8 isPreamble  : 1;  ///< Set if the buffer is an internal preamble command buffer.
            uint8 isPostamble : 1;  ///< Set if the buffer is an internal postamble command buffer.
            uint8 reserved    : 6;  ///< Reserved for future use.
        };
        uint8 u32All;               ///< Flags packed as 8-bit uint.
    } flags;

};

/// Structure describing a command buffer chunk for use while dumping command buffers.
struct CmdBufferChunkDumpDesc
{
    uint32       id;        ///< ID (number) of this command chunk within the command buffer.
    const void*  pCommands; ///< Pointer to the command data.
    size_t       size;      ///< Size of valid data in bytes pointed to in pCommands.
};

/// Definition for command buffer dumping callback.
///
/// @param [in] cmdBufferDesc   Description of the command buffer.
/// @param [in] pChunks         Pointer to an array of command buffer chunk descriptions.
/// @param [in] numChunks       The number of chunks pointed to in pChunks.
typedef void (PAL_STDCALL* CmdDumpCallback)(
    const CmdBufferDumpDesc&      cmdBufferDesc,
    const CmdBufferChunkDumpDesc* pChunks,
    uint32                        numChunks,
    void*                         pUserData);

/// Specifies properties for @ref IQueue creation.  Input structure to IDevice::CreateQueue().
struct QueueCreateInfo
{
    QueueType     queueType;     ///< Selects which type of queue to create.
    EngineType    engineType;    ///< Selects which type of engine to create.
    uint32        engineIndex;   ///< Which instance of the specified engine type to query. For example, there
                                 ///  can be multiple compute queues, so this parameter distinguished between them.
    SubmitOptMode submitOptMode; ///< A hint telling PAL which submit-time bottlenecks should be optimized, if any.
    QueuePriority priority;      ///< A hint telling PAL to create queue with proper priority.
                                 ///  It is only supported if supportQueuePriority is set in DeviceProperties.
                                 ///  In Linux, if we don't have root privilege, the creation with above-Medium
                                 ///  priority will fail. Client should take the corresponding action like retry
                                 ///  with lower priority, if necessary.
    struct
    {
        uint32 aqlQueue                        :  1; ///< Compute queue will process AQL packets and kernels
        uint32 windowedPriorBlit               :  1; ///< All windowed presents on this queue are notifications
                                                     ///  that the client has manually done a blit present
        uint32 tmzOnly                         :  1; ///< This queue allows only TMZ submissions. Required for
                                                     ///  compute TMZ submits.

#if PAL_AMDGPU_BUILD
        uint32 enableGpuMemoryPriorities       :  1; ///< Enables support for GPU memory priorities on this Queue.
                                                     /// This is optional because enabling the feature requires
                                                     /// a small amount of memory overhead per-Queue for
                                                     /// bookkeeping purposes.
#else
        uint32 placeholder2                    :  1; ///< Reserved field. Set to 0.
#endif
        uint32 dispatchTunneling               :  1; ///< This queue uses compute dispatch tunneling.

        uint32 forceWaitIdleOnRingResize       :  1; ///< This queue need to wait for idle before resize RingSet.
                                                     ///  This is intended as a workaround for misbehaving applications.
#if defined(_WIN32)
        uint32 nullRendering                   :  1; ///< Setting this bit makes this queue behave like IfhModeKmd.
#else
        uint32 placeholder3                    :  1; ///< Reserved field. Set to 0.
#endif
        uint32 reserved                        : 25; ///< Reserved for future use.
    };

    uint32 numReservedCu;           ///< The number of reserved compute units for RT CU queue

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 914
    uint32 persistentCeRamOffset;   ///< Byte offset to the beginning of the region of CE RAM which this Queue should
                                    ///  preserve across consecutive submissions.  Must be a multiple of 32.  It is an
                                    ///  error to specify a nonzero value here if the the Device does not support
                                    ///  @ref supportPersistentCeRam for the Engine this Queue will attach to.
    uint32 persistentCeRamSize;     ///< Amount of CE RAM space which this Queue should preserve across consecutive
                                    ///  submissions.  Units are in DWORDs, and this must be a multiple of 8.  It is an
                                    ///  error to specify a nonzero value here if the the Device does not support
                                    ///  @ref supportPersistentCeRam for the Engine this Queue will attach to.
#endif

    uintptr_t aqlPacketList;        ///< Location of the HIP runtime's info about this queue
};

/// Specifies the portion of @ref SubmitInfo that is specific to each sub-queue in a multi-queue object (@see
/// IDevice::CreateMultiQueue).  Effectively, this enables specifying a different set of command buffers for each
/// queue that makes up a gang submission to a multi-queue object.
struct PerSubQueueSubmitInfo
{
    uint32            cmdBufferCount;   ///< Number of command buffers to be submitted (can be 0 if this submit doesn't
                                        ///  involve work for the relevant queue).
    ICmdBuffer*const* ppCmdBuffers;     ///< Array of cmdBufferCount command buffers to be submitted.  Command buffers
                                        ///  that are part of a ganged submit must guarantee the conditions required
                                        ///  for the optimizeExclusiveSubmit flag.
    const CmdBufInfo* pCmdBufInfoList;  ///< Null, or an array of cmdBufferCount structs providing additional
                                        ///  info about the command buffers being submitted.  If non-null,
                                        ///  elements are ignored if their isValid flag is false.
};

/// Specifies all information needed to execute a set of command buffers.  Input structure to IQueue::Submit().
///
/// Some members of this structure are not supported on all platforms.  The client must check the appropriate properties
/// structures to determine if the corresponding features are supported:
/// + pGpuMemoryRefs:    Support is indicated by supportPerSubmitMemRefs in @ref DeviceProperties.
/// + ppBlockIfFlipping: Support is indicated by supportBlockIfFlipping in @ref PlatformProperties.  If it is supported,
///                      the client must not specify a blockIfFlippingCount greater than MaxBlockIfFlippingCount.
///
/// @note If this queue is running in physical submission mode (due to hardware restrictions), the gpuMemRefCount and
///       pGpuMemoryRefs arguments to this method are ignored because the command buffers themselves contain their own
///       GPU memory reference lists.
struct MultiSubmitInfo
{
    const PerSubQueueSubmitInfo* pPerSubQueueInfo;///< Specifies per-subqueue information for the submit.  Typically
                                                  ///  this is a pointer to a single entry specifying the command
                                                  ///  buffers to be submitted on this queue.  For gang submission on
                                                  ///  a multi-queue, this should be an array with one entry per
                                                  ///  sub-queue.  The array size must be less than or equal to the
                                                  ///  queueCount specified when the multi-queue was created and
                                                  ///  the workload specified in each entry will be assigned to the
                                                  ///  corresponding sub-queue.  It is valid to have a cmdBufferCount
                                                  ///  of 0 for sub-queues without work. Can be null if perSubQueueInfo-
                                                  ///  Count is 0.
    uint32                  perSubQueueInfoCount; ///< Number of PerSubqueueSubmitInfo to be submitted. Can be zero if
                                                  ///  there is no work to submit.
    uint32                  gpuMemRefCount;       ///< Number of GPU memory references for this submit.
    const GpuMemoryRef*     pGpuMemoryRefs;       ///< Array of gpuMemRefCount GPU memory references.  Can be null if
                                                  ///  gpuMemRefCount is zero.  The GPU memory objects will be made
                                                  ///  resident for the duration of this submit.
    uint32                  doppRefCount;         ///< Number of DOPP desktop texture references for this submit.
    const DoppRef*          pDoppRefs;            ///< Array of doppRefCount DOPP texture references.  Can be null if
                                                  ///  doppRefCount is zero.
    uint32                  externPhysMemCount;   ///< Number of entries in ppExternPhysMem.
    const IGpuMemory**      ppExternPhysMem;      ///< Array of external physical memory allocations to be initialized
                                                  ///  as part of this submit.  The first submit that references a
                                                  ///  particular external physical memory allocation must include
                                                  ///  that allocation in this list.  Subsequent submits that reference
                                                  ///  the same allocation should not include it in this list, as it
                                                  ///  would trigger redundant GPU page table initialization.
    uint32                  blockIfFlippingCount; ///< Number of GPU memory objects to protect when flipped.
    const IGpuMemory*const* ppBlockIfFlipping;    ///< Array of blockIfFlippingCount GPU memory objects.  Can be null if
                                                  ///  blockIfFlippingCount is zero.  The command buffers will not be
                                                  ///  scheduled to the GPU while a fullscreen (flip) present is queued
                                                  ///  for any of these GPU memory allocations.
    uint32                  fenceCount;           ///< Number of fence objects to be signaled once the last command buffer
                                                  ///  in this submission completes execution.
    IFence**                ppFences;             ///< Array of fence objects. Can be null if fenceCount is zero.
    CmdDumpCallback         pfnCmdDumpCb;         ///< Null, or a callback function to handle the dumping of the
                                                  ///  command buffers used in this submit.
    void*                   pUserData;            ///< Client provided data to be passed to callback.

    uint32                  stackSizeInDwords;    ///< 0, or the max of stack frame size for indirect shaders of the
                                                  ///  pipelines referenced in the command buffers of this submission.
                                                  ///  The size is per native thread. So that the client will have to
                                                  ///  multiply by 2 if a Wave64 shader that needs scratch is used.
                                                  ///  Note that the size will not shrink for the lifetime of the queue
                                                  ///  once it is grown and only affects compute scratch ring.
    const IGpuMemory*       pFreeMuxMemory;       ///< The gpu memory object of the private flip primary surface for the
                                                  ///  FreeMux feature.
};

typedef MultiSubmitInfo SubmitInfo;

/// The value of blockIfFlippingCount in @ref SubmitInfo cannot be greater than this value.
constexpr uint32 MaxBlockIfFlippingCount = 16;

/// Specifies properties for the presentation of an image to the screen.  Input structure to IQueue::PresentDirect().
struct PresentDirectInfo
{
    union
    {
        struct
        {
            uint32 fullscreenDoNotWait :  1; ///< Fail the present immediately if the present queue is full.
            uint32 srcIsTypedBuffer    :  1; ///< True if the source is a typed buffer instead of an image.
            uint32 dstIsTypedBuffer    :  1; ///< True if the destination is a typed buffer instead of an image.
            uint32 notifyOnly          :  1; ///< Indicates that a present occurred outside of PAL. PAL must not
                                             ///  execute a present if this is true but may update internal
                                             ///  tracking state.
            uint32 reserved            : 28; ///< Reserved for future use.
        };
        uint32 u32All;       ///< Flags packed as 32-bit uint.
    } flags;                 ///< Present flags.

    OsWindowHandle hWindow;         ///< Native OS window handle that this image should be presented to.
    PresentMode    presentMode;     ///< Chooses between windowed and fullscreen present.
    uint32         presentInterval; ///< Must be an integer from 0 to 4.  0 indicates that the present should
                                    ///  occur immediately (may tear), and 1-4 indicates the present should
                                    ///  occur after 1 to 4 vertical syncs.  Only valid for fullscreen presents.
    union
    {
        IImage*        pSrcImage;       ///< Optional: The image to be presented.  If null, the present will not
                                        ///  occur but PAL may still call into the OS on certain platforms that
                                        ///  expect it.
        IGpuMemory*    pSrcTypedBuffer; ///< The typed buffer to be presented.  If null, the present will not occur
                                        ///  but PAL may still call into the OS on certain platforms that expect it.
    };
    union
    {
        IImage*        pDstImage;       ///< Optional: copy from the source image to this image.  If null, PAL will
                                        ///  automatically copy into the appropriate platform-specific destination.
                                        ///  This is only supported for windowed mode presents.
        IGpuMemory*    pDstTypedBuffer; ///< The typed buffer to be presented.  If null, the present will not occur
                                        ///  but PAL may still call into the OS on certain platforms that expect it.
    };

};

/// Media stream counter information.
struct MscInfo
{
    uint64 targetMsc;                  ///< if the current MSC is less than <targetMsc>, the buffer swap
                                       ///< will occur when the MSC value becomes equal to <targetMsc>
    uint64 divisor;                    ///< Divisor
                                       ///< the buffer swap will occur the next time the MSC value is
                                       ///< incremented to a value such that MSC % <divisor> = <remainder>
                                       ///< if the current MSC is greater than or equal to <targetMsc>
    uint64 remainder;                  ///< Remainder
};

/// Specifies properties for the presentation of an image to the screen.  Input structure to IQueue::PresentSwapChain().
struct PresentSwapChainInfo
{
    PresentMode presentMode;    ///< Chooses between windowed and fullscreen present.
    IImage*     pSrcImage;      ///< The image to be presented.
    ISwapChain* pSwapChain;     ///< The swap chain associated with the source image.
    uint32      imageIndex;     ///< The index of the source image within the swap chain. Owership of this image
                                ///  index will be released back to the swap chain if this call succeeds.
    uint32      rectangleCount; ///< Number of valid rectangles in the pRectangles array.
    uint32      syncInterval;   ///< Applicable only when syncIntervalOverride is set
                                ///  0 - The presentation occurs immediately, there is no synchronization.
                                ///  1 through 4 - Synchronize presentation after the nth vertical blank.
    const Rect* pRectangles;    ///< Array of rectangles defining the regions which will be updated.
    uint64      presentId;      ///< PresentId functions as an identifier for present operations on a swapchain.
                                ///  If this PresentId is non-zero, then the application can later use this value
                                ///  to refer to that image presentation. A value of zero indicates that this
                                ///  presentation has no associated presentId. A non-zero presentId must be greater
                                ///  than any non-zero presentId passed previously by the application for the same
                                ///  swapchain.
    union
    {
        struct
        {
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 941
            uint32 notifyOnly           :  1; ///< True if it is a notify-only present
#else
            uint32 notifyOnly           :  1; ///< Indicates that a present occurred outside of PAL. PAL must not
                                              ///  execute a present if this is true but may update internal
                                              ///  tracking state.
#endif
            uint32 isTemporaryMono      :  1; ///< True if WS Stereo is enabled, but 3D display mode turned off.
            uint32 turboSyncEnabled     :  1; ///< Whether TurboSync is enabled.
            uint32 syncIntervalOverride :  1; ///< Override default syncInterval with the value in syncInterval
                                              ///  Supported only on Windows wsiPlatforms.
            uint32 reserved             : 28; ///< Reserved for future use.
        };
        uint32 u32All; ///< Flags packed as 32-bit uint.
    } flags;           ///< PresentSwapChainInfo flags.
#if PAL_AMDGPU_BUILD
    MscInfo mscInfo;   ///< Media stream counter information
#endif
};

/// Specifies a mapping from a range of pages in a virtual GPU memory object to a range of pages in a real GPU memory
/// object.  Input to IQueue::RemapVirtualMemoryPages().
///
/// When mapping pages of a virtual GPU memory object to a range of pages in a real GPU memory object on a remote GPU,
/// the client must point pRealGpuMem at a peer GPU memory object created on the input queue's device instead of the
/// actual real GPU memory object created on the remote device.  This is required for two reasons:
///   1. PAL can only view remote GPU memory using peer objects.
///   2. PAL enforces a separation of state between different IDevice object families.
///
/// virtualStartOffset and size must be aligned to the virtualMemPageSize member of @ref DeviceProperties.
/// realStartOffset must be aligned to the realMemAllocGranularity member of @ref DeviceProperties.
struct VirtualMemoryRemapRange
{
    IGpuMemory*             pVirtualGpuMem;     ///< Virtual GPU memory object whose mapping is being updated.
    gpusize                 virtualStartOffset; ///< Start of the page range to be updated, in bytes.
    IGpuMemory*             pRealGpuMem;        ///< Real GPU memory object the virtual range should point at.
    gpusize                 realStartOffset;    ///< Start of the page range in the real GPU memory object, in bytes.
    gpusize                 size;               ///< Size of the mapping range, in bytes.
    VirtualGpuMemAccessMode virtualAccessMode;  ///< Access mode for virtual GPU memory's unmapped pages.
                                                ///  This parameter is ignored on some platforms.
};

/// Specifies a set of page mappings to copy between virtual GPU memory objects. The source and destination can be the
/// same memory object and the source and destination regions may overlap. Input to IQueue::CopyVirtualMemoryPageMappings().
///
/// srcStartOffset, dstStartOffset, and size must be aligned to the virtualMemPageSize member of @ref DeviceProperties.
struct VirtualMemoryCopyPageMappingsRange
{
    IGpuMemory* pSrcGpuMem;     ///< Virtual GPU memory object whose mapping is being copied from.
    gpusize     srcStartOffset; ///< Start of the copy source range, in bytes.
    IGpuMemory* pDstGpuMem;     ///< Virtual GPU memory object whose mapping is being copied to.
    gpusize     dstStartOffset; ///< Start of the copy destination range, in bytes.
    gpusize     size;           ///< Size of the mapping range, in bytes.
};

/// Specifies kernel level information about a context.
struct KernelContextInfo
{
    union
    {
        struct
        {
            uint32 hasDebugVmid        :  1; ///< True if the context has acquired the debug vmid.
            uint32 hasHighPriorityVmid :  1; ///< True if the context has acquired the high priority vmid.
            uint32 reserved            : 30; ///< Reserved for future use.
        };
        uint32 u32All;                       ///< Flags packed as 32-bit uint.
    } flags;                                 ///< Context flags.

    uint64 contextIdentifier;                ///< Kernel scheduler context identifier.
};

/**
 ***********************************************************************************************************************
 * @interface IQueue
 * @brief     Represents a queue of work for a particular GPU engine on a device.
 *
 * An IQueue object is a virtual representation of a hardware engine on the device. Multiple IQueue objects can be
 * created and have work submitted on them in parallel. Work is submitted to a queue through @ref ICmdBuffer objects,
 * and work can be synchronized between multiple queues using @ref IQueueSemaphore objects.
 *
 * @see IDevice::GetQueue()
 ***********************************************************************************************************************
 */
class IQueue : public IDestroyable
{
public:
    /// Submits a group of root command buffers for execution on this queue.
    ///
    /// @param [in] submitInfo Specifies all command buffers to execute along with other residency and synchronization
    ///                        information.  See @ref SubmitInfo for additional, important documentation.
    ///
    /// @returns Success if the command buffer was successfully submitted.  Otherwise, one of the following errors may
    ///          be returned:
    ///          + ErrorInvalidPointer if:
    ///              - any of the array inputs are null when their counts are non-zero.
    ///              - any members of non-null point arrays are null.
    ///          + ErrorTooManyMemoryReferences if the total number of memory references (device/queue global and
    ///            per-command buffer) is too large.
    ///          + ErrorInvalidValue if blockIfFlippingCount is too large.
    ///          + ErrorIncompleteCommandBuffer if any of the submitted command buffers are not properly constructed.
    ///          + ErrorIncompatibleQueue if any submitted command buffer does not match this queue's type (e.g.,
    ///            universal, graphics, DMA).
    virtual Result Submit(
        const MultiSubmitInfo& submitInfo) = 0;

    /// Waits for all previous submission on this queue to complete before control is returned to the caller.
    ///
    /// @returns Success if wait for submissions completed.  Otherwise an error indicates reason for unsuccessful wait,
    ///          for example due to lost device.
    virtual Result WaitIdle() = 0;

    /// Inserts a semaphore signal into the GPU queue.  The semaphore will be signaled once all previously submitted
    /// work on this queue has completed.
    ///
    /// @param [in] pQueueSemaphore     Semaphore to signal.
    /// @param [in] value               timeline Semaphore point value to signal, ignored for non-timeline semaphores.
    ///
    /// @returns Success if the semaphore signal was successfully queued.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorUnknown if the OS scheduler rejects the signal for unknown reasons.
    virtual Result SignalQueueSemaphore(
        IQueueSemaphore* pQueueSemaphore, uint64 value = 0) = 0;

    /// Inserts a semaphore wait into the GPU queue.  The queue will be stalled until the specified semaphore is
    /// signaled.
    ///
    /// @param [in] pQueueSemaphore     Semaphore to wait on.
    /// @param [in] value               timeline semaphore point value to wait on, ignored for non-timeline semaphores.
    ///
    /// @returns Success if the semaphore wait was successfully queued.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorUnknown if the OS scheduler rejects the wait for unknown reasons.
    virtual Result WaitQueueSemaphore(
        IQueueSemaphore* pQueueSemaphore, uint64 value = 0) = 0;

#if PAL_KMT_BUILD
    /// Acquire the keyed mutex of shared GPU memory object (CPU sync) and then wait for the synchronization object of
    /// the shared GPU memory object (GPU sync based on fence). Note that the shared GPU memory object has to be
    /// a D3d11 resource created with (D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX | D3D11_RESOURCE_MISC_SHARED_NTHANDLE)
    /// misc flag.
    ///
    /// @param [in] pGpuMemory  Shared GPU memory object on which keyed mutex and synchronization object are bound.
    ///        [in] key         Key of keyed mutex to be acquired.
    ///        [in] timeout     Timeout interval for keyed mutex acquiring, in milliseconds.
    ///
    /// @ returns Success if the keyed mutex has been successfully acquired and wait for the synchronization object
    ///           has been successfully scheduled. Otherwise, one of the following errors may be returned:
    ///           + ErrorUnknown if either the keyed mutex has not been successfully acuiqred or wait for the
    ///           synchronization object has not been successfully scheduled.
    virtual Result KeyedMutexAcquireSync(
        IGpuMemory*               pGpuMemory,
        uint64                    key,
        std::chrono::milliseconds timeout) = 0;

    /// Signal the synchronization object of shared GPU memory object with bumped fence value and then release the
    /// keyed mutex of shared GPU memory object. Note that the shared GPU memory object has to be a D3d11 resource
    /// created with (D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX | D3D11_RESOURCE_MISC_SHARED_NTHANDLE) misc flag.
    ///
    /// @param [in] pGpuMemory  Shared GPU memory object on which keyed mutex and synchronization object are
    ///                         bound.
    ///        [in] key         Key of keyed mutex to be released.
    ///
    /// @ returns Success if signal of the synchronization object has been successfully scheduled and the keyed mutex
    ///           has been successfully released. Otherwise, one of the following errors may be returned:
    ///           + ErrorUnknown if either signal of the synchronization object has not been successfully scheduled
    ///           or keyed mutex has not been successfully released.
    virtual Result KeyedMutexReleaseSync(
        IGpuMemory* pGpuMemory,
        uint64      key) = 0;
#endif

    /// This function passes application information to KMD for application specific power optimizations.
    /// Power configuration are restored to default when all application queues are destroyed.
    ///
    /// @param [in]  pFileName  Application executable name
    /// @param [in]  pPathName  Path to the application
    ///
    /// @returns Success if the information is passed successfully.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + Unsupported if this function is not available on this OS or if the queue context is null.
    ///          + ErrorUnknown if an unexpected internal error occurs.
    virtual Result UpdateAppPowerProfile(
        const wchar_t* pFileName,
        const wchar_t* pPathName) = 0;

    /// Queues the specified image for presentation on the screen.  This function directly queues the presentation
    /// request based on the input parameters without special synchronization considerations like a swap chain present.
    /// All previous work done on this queue will complete before the image is displayed.
    ///
    /// This function should never be called with a swap chain presentable image because it won't release ownership of
    /// the presentable image index, eventually deadlocking the swap chain.
    ///
    /// Overall support for direct presents can be queried at platform creation time via supportNonSwapChainPresents
    /// in @ref PlatformProperties.  Support for particular present modes is specifed via supportedDirectPresentModes
    /// in @ref DeviceProperties.
    ///
    /// @note  Any images specified in presentInfo must be made resident before calling this function.
    ///
    /// @param [in] presentInfo Specifies the source image and destination window for the present as well as other
    ///                         properties.
    ///
    /// @returns Success if the present was successfully queued.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorInvalidValue if the flip interval is invalid.
    ///          + ErrorInvalidValue if the present mode doesn't match the capabilities of the image.
    ///          + ErrorInvalidFlags if the present flags don't match the capabilities of the image.
    virtual Result PresentDirect(
        const PresentDirectInfo& presentInfo) = 0;

    /// Queues the specified image for presentation on the screen.  This function uses the provided swap chain to
    /// determine exactly how the image should be presented (e.g., can the user see tearing).  See @ref ISwapChain for
    /// more information on swap chain presentation.  All previous work done on this queue will complete before the
    /// image is displayed, but future work may execute before the present is completed because swap chain present
    /// execution may be asynchronous to the queue that initiated present.
    ///
    /// Assuming the presentInfo is valid, this function will always release ownership of the presentable image index
    /// even if PAL encounters an error while executing the present.
    ///
    /// Queue support for swap chain presents is specified via supportsSwapChainPresents in @ref DeviceProperties.
    /// Support for particular PresentModes is queried per SwapChainMode via IDevice::GetSwapChainInfo().
    ///
    /// @note  The source image specified in presentInfo must be made resident before calling this function.
    ///
    /// @param [in] presentInfo Specifies the source image, swap chain, and basic presentation information.
    ///
    /// @returns Success if the present was successfully queued.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorInvalidPointer if the source image or swap chain are null.
    ///          + ErrorInvalidValue if the present mode doesn't match the capabilities of the image or if the image
    ///                              index isn't valid within the swap chain.
    virtual Result PresentSwapChain(
        const PresentSwapChainInfo& presentInfo) = 0;

    /// Inserts a delay of a specified amount of time before processing more commands on this queue.
    ///
    /// Only available on timer queues.  Useful in conjunction with queue semaphores to implement frame pacing.
    ///
    /// @param [in] delay Time, in milliseconds, to delay before processing more commands on this queue.
    ///
    /// @returns Success if the delay was successfully queued.  Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidValue if delay is less than 0.
    virtual Result Delay(
        Util::fmilliseconds delay) = 0;

    /// Inserts a delay of a specified amount of time on this queue after a vsync on a private display object.
    ///
    /// Only available on timer queues.  Useful in conjunction with queue semaphores to implement pacing of GPU and CPU
    /// operations for rendering and presentation in VR as this allows GPU commands of next frame to be sent early but
    /// blocks GPU execution until after vsync.
    ///
    /// @param [in] delay   Time, in microseconds, to delay before processing more commands on this queue.
    /// @param [in] pScreen The private screen object that the vsync is occurring and the delay is waiting on.
    ///
    /// @returns Success if the delay was successfully queued.  Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidValue if delay is less than 0.
    virtual Result DelayAfterVsync(
        Util::fmicroseconds   delay,
        const IPrivateScreen* pScreen) = 0;

    /// Updates page mappings for virtual GPU memory allocations.
    ///
    /// @param [in] rangeCount  Number of ranges to remap (i.e., size of the pRanges array).
    /// @param [in] pRanges     Defines the set of remappings from virtual GPU memory object pages to real GPU
    ///                         memory object pages.
    /// @param [in] doNotWait   If true, then this paging operation will be executed on the Queue immediately, without
    ///                         waiting for any previous rendering to finish first. On platforms that don't support
    ///                         this, the flag will be ignored.
    /// @param [in] pFence      Optional. Pointer to an IFence, which will be signaled after the VA remapping.
    ///
    /// @returns Success if the remappings were executed successfully.  It is assumed that the following conditions are
    ///          met for the input to this function:
    ///          + rangeCount is not 0.
    ///          + The page range for all members of pRanges are valid.
    ///          + pRanges is not null.
    ///          + pVirtualGpuMem is not null for any member of pRanges.
    ///          + pRanges does not specify a real GPU memory object as a virtual GPU memory object or vice versa.
    virtual Result RemapVirtualMemoryPages(
        uint32                         rangeCount,
        const VirtualMemoryRemapRange* pRanges,
        bool                           doNotWait,
        IFence*                        pFence) = 0;

    /// Copies page mappings from one virtual GPU memory object to another.
    ///
    /// @param [in] rangeCount  Number of ranges to copy (i.e., size of the pRanges array).
    /// @param [in] pRanges     Defines the set of page mappings to copy between virtual GPU memory objects.
    /// @param [in] doNotWait   If true, then this paging operation will be executed on the Queue immediately, without
    ///                         waiting for any previous rendering to finish first. On platforms that don't support
    ///                         this, the flag will be ignored.
    ///
    /// @returns Success if the mappings were copied successfully.  It is assumed that the following conditions are
    ///          met for the input to this function:
    ///          + rangeCount is not 0.
    ///          + The page range for all members of pRanges are valid.
    ///          + pRanges is not null.
    ///          + pSrcGpuMem or pDstGpuMem is not null for any member of pRanges.
    ///          + pRanges does not specify a real GPU memory object as source or destination
    virtual Result CopyVirtualMemoryPageMappings(
        uint32                                    rangeCount,
        const VirtualMemoryCopyPageMappingsRange* pRanges,
        bool                                      doNotWait) = 0;

    /// Associates the provided Fence object with the last submission on this queue object. The Fence can be used via
    /// GetStatus() to get the status of the last Submit, however no event will be created/set for the Fence so
    /// WaitForFences() should NOT be called on the fence after this association.
    ///
    /// @see IFence::GetStatus()
    /// @see IFence::WaitForFences()
    ///
    /// @param [in] pFence   Fence object to be associated with the last Submit on this queue
    ///
    /// @returns Success if the association was successful. ErrorUnavailable will be returned in there has not yet been
    ///          a Submit on this queue.
    virtual Result AssociateFenceWithLastSubmit(
        IFence* pFence) = 0;

    /// Set execution priority for the current queue, it allows to elevate execution priority of submitted command
    /// buffers, but it has no effect on command buffers that have already been submitted for execution. Elevating
    /// the queue priority to medium or high would allow to temporary stall a low priority queue execution and execute
    /// its work as soon as the low priority queue starts draining.
    ///
    /// @param [in] priority The priority level of the queue.
    virtual void SetExecutionPriority(
        QueuePriority priority) = 0;

    /// Returns a list of GPU memory allocations used by this queue.
    ///
    /// @param [in,out] pNumEntries    Input value specifies the available size in pAllocInfoList; output value
    ///                                reports the number of GPU memory allocations.
    /// @param [out]    pAllocInfoList If pAllocInfoList=nullptr, then pNumEntries is ignored on input.  On output it
    ///                                will reflect the number of allocations that make up this queue.  If
    ///                                pAllocInfoList!=nullptr, then on input pNumEntries is assumed to be the number
    ///                                of entries in the pAllocInfoList array.  On output, pNumEntries reflects the
    ///                                number of entries in pAllocInfoList that are valid.
    /// @returns Success if the allocation info was successfully written to the buffer.
    ///          + ErrorInvalidValue if the caller provides a buffer size that is different from the size needed.
    ///          + ErrorInvalidPointer if pNumEntries is nullptr.
    virtual Result QueryAllocationInfo(
        size_t*                    pNumEntries,
        GpuMemSubAllocInfo* const  pAllocInfoList) = 0;

    /// Returns the QueueType for the queue
    virtual QueueType Type() const = 0;

    /// Returns the EngineType for the queue
    virtual EngineType GetEngineType() const = 0;

    /// Queries the kernel context info associated with this queue and copies it into pKernelContextInfo.
    ///
    /// Only supported on Windows platforms.
    ///
    /// @param [out] pKernelContextInfo Pointer to a KernelContextInfo struct to copy the information into.
    /// @returns Success if the information is successfully copied into the output struct.
    ///          + ErrorInvalidPointer if pKernelContextInfo is nullptr.
    ///          + ErrorUnavailable if kernel context information is not available on the current platform.
    virtual Result QueryKernelContextInfo(KernelContextInfo* pKernelContextInfo) const = 0;

    /// Returns the value of the associated arbitrary client data pointer.
    /// Can be used to associate arbitrary data with a particular PAL object.
    ///
    /// @returns Pointer to client data.
    void* GetClientData() const
    {
        return m_pClientData;
    }

    /// Sets the value of the associated arbitrary client data pointer.
    /// Can be used to associate arbitrary data with a particular PAL object.
    ///
    /// @param  [in]    pClientData     A pointer to arbitrary client data.
    void SetClientData(
        void* pClientData)
    {
        m_pClientData = pClientData;
    }

protected:
    /// @internal Constructor. Prevent use of new operator on this interface. Client must create objects by explicitly
    /// called the proper create method.
    IQueue() : m_pClientData(nullptr) {}

    /// @internal Destructor.  Prevent use of delete operator on this interface.  Queues will be destroyed when the
    /// associated device is destroyed.
    virtual ~IQueue() { }

private:
    /// @internal Client data pointer. This can have an arbitrary value and can be returned by calling GetClientData()
    /// and set via SetClientData().
    /// For non-top-layer objects, this will point to the layer above the current object.
    void* m_pClientData;
};

} // Pal
