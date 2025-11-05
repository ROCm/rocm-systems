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
 * @file  palGpuMemory.h
 * @brief Defines the Platform Abstraction Library (PAL) IGpuMemory interface and related types.
 ***********************************************************************************************************************
 */

#pragma once

#include "pal.h"
#include "palDestroyable.h"

#if defined(_WIN32)
struct _SECURITY_ATTRIBUTES;
#endif

namespace Pal
{

// Forward declarations.
class IGpuMemory;
class IDevice;
class IImage;
enum class VaRange : uint32;

/// Specifies Base Level priority per GPU memory allocation as a hint to the memory manager in the event it needs to
/// select allocations to page out of their preferred heaps.
enum class GpuMemPriority : uint32
{
    Unused    = 0x0,  ///< Indicates that the allocation is not currently being used at all, and should be the first
                      ///  choice to be paged out.
    VeryLow   = 0x1,  ///< Lowest priority to keep in its preferred heap.
    Low       = 0x2,  ///< Low priority to keep in its preferred heap.
    Normal    = 0x3,  ///< Normal priority to keep in its preferred heap.
    High      = 0x4,  ///< High priority to keep in its preferred heap (e.g., render targets).
    VeryHigh  = 0x5,  ///< Highest priority to keep in its preferred heap.  Last choice to be paged out (e.g., page
                      ///  tables, displayable allocations).
    Count
};

/// Specifies a finer granularity to the base Level priority per GPU memory allocation as a hint to the memory manager
/// in the event it needs to select allocations to page out of their preferred heaps.
enum class GpuMemPriorityOffset : uint32
{
    Offset0  = 0x0, ///< Same priority as Base Level
    Offset1  = 0x1, ///< Next priority from Base Level
    Offset2  = 0x2, ///< Next priority from Base Level.
    Offset3  = 0x3, ///< Next priority from Base Level.
    Offset4  = 0x4, ///< Next priority from Base Level.
    Offset5  = 0x5, ///< Next priority from Base Level.
    Offset6  = 0x6, ///< Next priority from Base Level
    Offset7  = 0x7, ///< Highest priority from Base Level
    Count
};

/// Speicfies access mode for unmapped pages in a virtual Gpu Memory.
enum class VirtualGpuMemAccessMode : uint32
{
    Undefined = 0x0, ///< Used in situations where no special accessMode needed.
    NoAccess  = 0x1, ///< All accesses of unmapped pages will trigger a GPU page fault.
    ReadZero  = 0x2, ///< Reads of unmapped pages return zero, and writes are discarded.
    Count
};

/// Controls the behavior of this allocation with respect to the MALL.
enum class GpuMemMallPolicy : uint32
{
    Default = 0x0,  ///< MALL policy is decided by the driver.
    Never   = 0x1,  ///< This allocation is never put through the MALL.
    Always  = 0x2,  ///< This allocation is always put through the MALL.
};

/// Bitmask of cases where RPM view memory accesses will bypass the MALL.
enum RpmViewsBypassMall : uint32
{
    RpmViewsBypassMallOff         = 0x0, ///< Disable MALL bypass
    RpmViewsBypassMallOnRead      = 0x1, ///< Skip MALL for read access of views created in RPM
    RpmViewsBypassMallOnWrite     = 0x2, ///< Skip MALL for write access of views created in RPM
    RpmViewsBypassMallOnCbDbWrite = 0x4, ///< Control the RPM CB/DB behavior

};

/// Used for specifying a subregion of the allocation as having a different mall policy from the rest of the
/// allocation.
struct GpuMemMallRange
{
    uint32  startPage; ///< Starting 4k page that will obey the specified mallPolicy.
    uint32  numPages;  ///< Number of 4k pages that will obey the specified mallPolicy.
};

/// Specifies flags for @ref IGpuMemory creation.
union GpuMemoryCreateFlags
{
    struct
    {
        uint64 virtualAlloc                 :  1; ///< Create a _virtual_ as opposed to _real_ GPU memory allocation.
                                                  ///  Only VA space will be allocated, and pages must be mapped via
                                                  ///  IQueue::RemapVirtualMemoryPages().
        uint64 shareable                    :  1; ///< Memory can be shared between devices in the same process that
                                                  ///  report the sharedMemory flag from
                                                  /// IDevice::GetMultiGpuCompatibility().
        uint64 interprocess                 :  1; ///< Memory will be visible to other processes
                                                  ///  (they may choose to open it).
        uint64 presentable                  :  1; ///< Memory can be bound to an image that will be used by presents.
        uint64 flippable                    :  1; ///< Memory can be bound to an image that will be used by flip
                                                  ///  presents.
        uint64 stereo                       :  1; ///< Memory will be used for stereo (DXGI or AQBS stereo).
        uint64 globallyCoherent             :  1; ///< Memory needs to be globally coherent,
                                                  ///  indicating the driver must manage both
                                                  ///  CPU caches and GPU caches that are not flushed on
                                                  ///  command buffer boundaries.
        uint64 xdmaBuffer                   :  1; ///< GPU memory will be used for an XDMA cache buffer for
                                                  ///  transferring data
                                                  ///  between GPUs in a multi-GPU configuration.
        uint64 turboSyncSurface             :  1; ///< The memory will be used for TurboSync private swapchain primary.
        uint64 typedBuffer                  :  1; ///< GPU memory will be permanently considered a single
                                                  ///  typed buffer pseudo-object
                                                  ///  with the properties given in typedBufferInfo.
        uint64 globalGpuVa                  :  1; ///< The GPU virtual address must be visible to all devices.
        uint64 useReservedGpuVa             :  1; ///< Use GPU virtual address previously reserved by another
                                                  ///  memory object. It is invalid when using the shadow descriptor
                                                  ///  table VA range.
        uint64 autoPriority                 :  1; ///< Allow the platform to automatically determine the priority of
                                                  ///  this GPU memory allocation. Flag is only valid if the device
                                                  ///  reports that it supports this feature, and will result in an
                                                  ///  error otherwise.
        uint64 busAddressable               :  1; ///< Create Bus Addressable memory. Allow memory to be used by other
                                                  ///  device on the PCIe bus by exposing a write-only bus address.
        uint64 sdiExternal                  :  1; ///< Create External Physical memory from an already allocated memory
                                                  ///  on remote device. Similar to virtual allocations (no physical
                                                  ///  backing) but have an immutable page mapping. The client must
                                                  ///  specify surfaceBusAddr and markerBusAddr either at creation time
                                                  ///  in GpuMemoryCreateInfo or by calling SetSdiRemoteBusAddress
                                                  ///  once before using the GPU memory. The page mappings for an
                                                  ///  allocation with this flag set must be initialized by including a
                                                  ///  reference to it in the ppExternPhysMem list for the first
                                                  ///  submission that references it.
        uint64 sharedViaNtHandle            :  1; ///< Memory will be shared by using Nt handle.
        uint64 peerWritable                 :  1; ///< The memory can be open as peer memory and be writable.
        uint64 tmzProtected                 :  1; ///< The memory is protected using TMZ (Trusted Memory Zone) or HSFB
                                                  ///  (Hybrid Secure Framebuffer). It is not CPU accessible,
                                                  ///  and GPU access is restricted by the hardware such that data
                                                  ///  cannot be copied from protected memory into unprotected memory.
        uint64 placeholder0                 :  1; ///< Placeholder.
        uint64 externalOpened               :  1; ///< Specifies the GPUMemory is opened.
        uint64 restrictedContent            :  1; ///< Specifies the GPUMemory is protected content.
        uint64 restrictedAccess             :  1; ///< Specifies the GPUMemory is restricted shared access resource.
        uint64 crossAdapter                 :  1; ///< Specifies the GPUMemory is shared cross-adapter resource.
        uint64 cpuInvisible                 :  1; ///< By default, PAL makes every allocation CPU visible if all of its
                                                  ///  preferred
                                                  ///< heaps are CPU visible. This flag can be used to override this
                                                  ///  behavior when the client knows the memory will never be mapped
                                                  ///  for CPU access. If this flag is set, calls to IGpuMemory::Map()
                                                  ///  on this object will fail.
        uint64 gl2Uncached                  :  1; ///< Specifies the GPU Memory is un-cached on GPU L2 cache.
                                                  ///  But the memory still would be cached by other cache hierarchy
                                                  ///  like L0, RB caches, L1, and L3.
        uint64 mallRangeActive              :  1; ///< If set, then this allocation will be partially allocated in the
                                                  ///  MALL. If this is set, then the mallPolicy enumeration must be set
                                                  ///  to either "always" or "never".
        uint64 explicitSync                 :  1; ///< If set, shared memory will skip syncs in the kernel and all
                                                  ///  drivers that use this memory must handle syncs explicitly.
        uint64 privPrimary                  :  1; ///< This is a private primary surface gpu memory.
        uint64 privateScreen                :  1; ///< GPU memory will be used for a private screen image.
        uint64 kmdShareUmdSysMem            :  1; ///< UMD will allocate/free a memory buffer to be shared with KMD.
        uint64 deferCpuVaReservation        :  1; ///< KMD will allocate with the "CpuVisibleOnDemand" alloc flag.
                                                  ///  Ignored for non-CPU-visible allocations.
        uint64 placeholder1                 :  1;
        uint64 startVaHintFlag              :  1; ///< startVaHintFlag is set to 1 for passing startVaHint address
                                                  ///  to set baseVirtAddr as startVaHint for memory allocation.
#if PAL_AMDGPU_BUILD
        uint64 initializeToZero             :  1; ///< If set, PAL will request that the host OS zero-initializes
                                                  ///  the allocation upon creation, currently, only GpuHeapLocal and
                                                  ///  GpuHeapInvisible are supported.
        uint64 discardable                  :  1; ///< If set, this gpu memory object can be discarded under memory
                                                  ///  pressure without keeping the content.
#else
        uint64 placeholder2                 :  2;
#endif
        uint64 directCaptureSource          :  1; ///< Memory will be mapped to DirectCapture resource's KMD-managed
                                                  ///  private VA.
        uint64 reserved                     : 28; ///< Reserved for future use.
    };
    uint64     u64All;                            ///< Flags packed as 64-bit uint.
};

/// Specifies properties of a typed buffer pseudo-object. When this is specified in GpuMemoryCreateInfo along with the
/// typedBuffer flag, the GPU memory object has been permanently cast as a single typed buffer.  A typed buffer is very
/// similar to a linear 3D image: it has a format, extent, and row/depth pitch values.
///
/// Note that the typed buffer concept is used in other parts of the PAL interface and some of those instances may not
/// require a permanent typed buffer association.  In such cases multiple typed buffers can be "bound" to one GPU memory
/// object at arbitrary offsets without any need to set the typedBuffer flag or fill out a TypedBufferCreateInfo.
struct TypedBufferCreateInfo
{
    SwizzledFormat swizzledFormat; ///< Pixel format and channel swizzle.
    Extent3d       extent;         ///< Dimensions in pixels WxHxD.
    uint32         rowPitch;       ///< Offset in bytes between the same X position on two consecutive lines.
    uint32         depthPitch;     ///< Offset in bytes between the same X,Y position of two consecutive slices.
    bool           depthIsSubres;  ///< True if the depth slices should be treated as an array of 2D subresources.
};

/// Specifies properties for @ref IGpuMemory creation.  Input structure to IDevice::CreateGpuMemory().
///
/// See the @ref IGpuMemory for additional restrictions on the size, alignment, vaRange, and descrVirtAddr fields.
struct GpuMemoryCreateInfo
{
    GpuMemoryCreateFlags flags;     ///< GPU memory flags.
    gpusize              size;      ///< Amount of GPU memory to allocate in bytes.
    gpusize              alignment; ///< Byte alignment of the allocation's GPU VA. If zero, an alignment matching the
                                    ///  allocation granularity will be used.
    VaRange              vaRange;   ///< Virtual address range for the GPU memory allocation.

    union
    {
        const IGpuMemory* pReservedGpuVaOwner; ///< Must be zero unless "useReservedGpuVa" is true.  It points to the
                                               ///  memory object which previously reserved the GPU VA range to be used
                                               ///  by the new memory object.
        gpusize           descrVirtAddr;       ///< Must be zero unless vaRange is ShadowDescriptorTable, in which case
                                               ///  it must specify the GPU VA of the corresponding DescriptorTable.
                                               ///  It doesn't need to be the base VA of the DescriptorTable allocation
                                               ///  but must be aligned to "alignment".
        gpusize           replayVirtAddr;      ///< Must be zero unless vRange is CaptureReplay, in which case it must
                                               ///  specify the GPU VA of the corresponding memory object.
        gpusize           startVaHint;         ///< Client passes a start VA hint to set as baseVirtAddr. If the given
                                               ///  hint is not properly aligned, find next higher aligned address as
                                               ///  hint. If the hint is available and within right vaRange where
                                               ///  vaRange is VaRange::Default then set baseVirtAddr as hint. If the
                                               ///  hint is unavailable, find the higher available address between
                                               ///  startVaHint and max vaRange. If any of the two cases are failed,
                                               ///  set baseVirtAddr as normal.
    };

    GpuMemPriority       priority;       ///< Hint to the OS paging process on how important it is to keep this
                                         ///  allocation in its preferred heap.
    GpuMemPriorityOffset priorityOffset; ///< Offset from the base level priority. A higher offset means higher priority
                                         ///  within same base Level. Currently supported on Windows only.
    GpuMemMallPolicy     mallPolicy;     ///< Used to control whether or not this allocation will be accessed via the
                                         ///  MALL (memory access last level).  Only valid if "supportsMall" is set in
                                         ///  DeviceProperties.
    GpuMemMallRange      mallRange;      ///< These parameters are only meaningful if flags.mallRangeActive is set.
                                         ///  Any pages outside of this range will use the opposite MALL policy from
                                         ///  what is specified in "mallPolicy".

    /// Describes how the allocation will be accessed. If heapAccess is set to something other than @ref
    /// GpuHeapAccessExplicit, then PAL decides the appropriate heap to allocate memory from based on this member and
    /// @ref heaps is ignored. Otherwise heap selection respects the selection in @ref heaps.
    GpuHeapAccess heapAccess;
    uint32        heapCount;           ///< Number of entries in heaps[]. Must be 0 for virtual allocations.
    GpuHeap       heaps[GpuHeapCount]; ///< List of allowed memory heaps, in order of preference. It will be ignored if
                                       ///  @ref heapAccess is to something other than @ref GpuHeapAccessExplicit.

    /// The pointer to an Image object the memory object will be bound to.  It must be specified in special cases where
    /// a memory object is permanently linked to an Image such as presentable images or shared resources on Windows.
    /// For other cases, it's highly encouraged to provide the image object pointer.
    ///
    /// When @ref compression is set to Default, clients can benefit from PAL's internal logic to choose the optimal
    /// GFX12-style distributed compression setting (enabled or disabled) based on properties of this image (e.g., usage
    /// flags); otherwise PAL will disable the compression by default when detecting nullptr @pImage.
    IImage* pImage;

    TypedBufferCreateInfo typedBufferInfo; ///< This struct must be filled out if the @tref typedBuffer flag is set.
                                           ///  This GPU memory will be permanently considered a typed buffer.

    VirtualGpuMemAccessMode virtualAccessMode; ///< Access mode for virtual GPU memory's unmapped pages, WDDM only.
    gpusize                 surfaceBusAddr;    ///< Surface bus address of Bus Addresable Memory.
                                               ///  Only valid when GpuMemoryCreateFlags::sdiExternal is set.
    gpusize                 markerBusAddr;     ///< Marker bus address of Bus Addresable Memory. The client can:
                                               ///  1. Write to marker
                                               ///  2. Let GPU wait until a value is written to marker before issuing
                                               ///     the next command.
                                               ///  Only valid when GpuMemoryCreateFlags::sdiExternal is set.

    /// Client override for GFX12-style distributed compression. Only meaningful on devices that set the
    /// supportDistributedCompression DeviceProperties flag. By default, PAL will apply a heuristic to determine
    /// whether or not to enable compression based on properties of the memory allocation (e.g., CPU-visibility or
    /// properties of the attached pImage). This mode does not apply to virtual IGpuMemory objects (such object
    /// inherit their compression behavior from their backing physical memory on a page-by-page basis).
    ///
    /// @note If allocating memory intended to back multiple resources (e.g., a heap for DX12 "placed resources") where
    /// pImage is nullptr, the client should set compression to Enable then rely on per-resource or per-view controls to
    /// disable compression for resources as needed (use compressionMode fields in ImageCreateInfo, BufferViewInfo,
    /// ImageViewInfo, etc.).
    ///
    /// If clients intends to enable distributed compression on buffers, buffer base alignment and size should be
    /// aligned to DCC minimum compression unit size (256 bytes); otherwise there is potential corruption issue.
    /// e.g. An allocation holding two buffers (placed resource or driver suballocation). Suppose the first buffer has
    /// size 4Kibytes+128 bytes with compression enabled and the second buffer has size 128bytes with compression
    /// disabled (buffer view compressionMode is ReadBypassWriteDisable). When compressing the trailing 128bytes of
    /// the first buffer, it will compress the second buffer as well. Reading the second buffer will result in corrupted
    /// content.
    TriState compression;
};

/// Specifies properties for @ref IGpuMemory creation.  Input structure to IDevice::CreatePinnedGpuMemory().
///
/// See the @ref IGpuMemory for additional restrictions on the size and vaRange fields.
struct PinnedGpuMemoryCreateInfo
{
    const void*       pSysMem;   ///< Pointer to the system memory that should be pinned for GPU access.  Must be
                                 ///  aligned to realMemAllocGranularity in DeviceProperties.
    size_t            size;      ///< Amount of system memory to pin for GPU access.
    VaRange           vaRange;   ///< Virtual address range for the GPU memory allocation.
    gpusize           alignment; ///< Byte alignment of the allocation's GPU VA.  If zero, an alignment matching the
                                 ///  Platform's allocation granularity will be used.
    GpuMemMallPolicy  mallPolicy; ///< Used to control whether or not this allocation will be
                                  ///  accessed via the MALL (memory access last level).  Only valid
                                  ///  if "supportsMall" is set in DeviceProperties.
    GpuMemMallRange   mallRange;  ///< These parameters are only meaningful if flags.mallRangeActive
                                  ///  is set.  Any pages outside of this range will use the opposite
                                  ///  MALL policy from what is specified in "mallPolicy".
};

/// Specifies properties for @ref IGpuMemory creation.  Input structure to IDevice::CreateSvmGpuMemory().
///
/// See the @ref IGpuMemory for additional restrictions on the size and alignment.
struct SvmGpuMemoryCreateInfo
{
    GpuMemoryCreateFlags    flags;                 ///< GPU memory flags.
    gpusize                 size;                  ///< Amount of SVM memory to allocate in bytes.
                                                   ///  The total amount of SVM memory can't exceed the value set in
                                                   ///  maxSvmSize when the platform is created.
    gpusize                 alignment;             ///< Byte alignment of the allocation's SVM VA.  If zero, an
                                                   ///  alignment matching the allocation granularity will be used.
    const IGpuMemory*        pReservedGpuVaOwner;  ///< Must be zero unless "useReservedGpuVa" is true.  It points to
                                                   ///  the memory object which previously reserved the GPU VA range
                                                   ///  to be used by the new memory object.
    bool                    isUsedForKernel;       ///< Memory will be used to store kernel and execute on gpu.
    GpuMemMallPolicy         mallPolicy;           ///< Used to control whether or not this allocation will be
                                                   ///  accessed via the MALL (memory access last level).  Only valid
                                                   ///  if "supportsMall" is set in DeviceProperties.
    GpuMemMallRange          mallRange;           ///< These parameters are only meaningful if flags.mallRangeActive
                                                  ///  is set.  Any pages outside of this range will use the opposite
                                                  ///  MALL policy from what is specified in "mallPolicy".
};

/// Specifies parameters for opening a shared GPU memory object on another device.
struct GpuMemoryOpenInfo
{
    IGpuMemory* pSharedMem; ///< Shared GPU memory object from another device to open.
};

/// Specifies parameters for opening a GPU memory object on another device for peer-to-peer memory transfers.
struct PeerGpuMemoryOpenInfo
{
    IGpuMemory* pOriginalMem; ///< GPU memory object from another device to open for peer-to-peer memory transfers.
};

/// Specifies parameters for opening another non-PAL device's gpu memory for access from this device.  Input structure to
/// IDevice::OpenExternalSharedGpuMemory().
struct ExternalGpuMemoryOpenInfo
{
    ExternalResourceOpenInfo resourceInfo;    ///< Information describing the external gpuMemory.
    TypedBufferCreateInfo    typedBufferInfo; ///< Information describing the typed buffer information.
    GpuMemMallPolicy         mallPolicy;      ///< Used to control whether or not this allocation will be accessed via
                                              ///  the MALL (memory access last level). Only valid if "supportsMall" is
                                              ///  set in DeviceProperties.
    GpuMemMallRange          mallRange;       ///< These parameters are only meaningful if flags.mallRangeActive is set.
                                              ///  Any pages outside of this range will use the opposite MALL policy
                                              ///  from what is specified in "mallPolicy".
    union
    {
        struct
        {
            uint32 typedBuffer     :  1;  ///< GPU memory will be permanently considered a single typed buffer pseudo-object
                                          ///  with the properties given in typedBufferInfo.
            uint32 gl2Uncached     :  1;  ///< Specifies the GPU Memory is un-cached on GPU L2 cache.
            uint32 mallRangeActive :  1;  ///< If set, then this allocation will be partially allocated in the MALL.
                                          ///  If this is set, then the mallPolicy enumeration must be set to either
                                          ///  "always" or "never".
            uint32 reserved        : 29;  ///< Reserved for future use.
        };
        uint32 u32All;              ///< Flags packed as 32-bit uint.
    } flags;                        ///< External Gpu memory open info flags.
};

/// The fundemental information that describes a GPU memory object that is stored directly in each IGpuMemory.
/// It can be accessed without a virtual call via IGpuMemory::Desc().
struct GpuMemoryDesc
{
    gpusize gpuVirtAddr;         ///< GPU virtual address of the GPU memory allocation.
    gpusize size;                ///< Size of the GPU memory allocation, in bytes.
    gpusize clientSize;          ///< Size of the client requested GPU memory allocation, in bytes.
    gpusize alignment;           ///< Required GPU virtual address alignment, in bytes.
    uint32  heapCount;           ///< Number of entries in heaps[].  Must be 0 for virtual allocations.
    GpuHeap heaps[GpuHeapCount]; ///< List of preferred memory heaps, in order of preference.
    gpusize surfaceBusAddr;      ///< Bus Address of SDI memory surface and marker. These will not be initialized
    gpusize markerBusAddr;       ///  until the memory is made resident. Client needs to call
                                 ///  InitBusAddressableGpuMemory() to query and update before this is valid.
    union
    {
        struct
        {
            uint32 isVirtual    :  1; ///< GPU memory is not backed by physical memory and must be remapped before the
                                      ///  GPU can safely access it. Will also be set for sdiExternal allocations. See
                                      ///  GpuMemoryCreateFlags::sdiExternal
            uint32 isPeer       :  1; ///< GPU memory object was created with @ref IDevice::OpenPeerGpuMemory.
            uint32 isShared     :  1; ///< GPU memory object was created either with
                                      ///  @ref IDevice::OpenExternalSharedGpuMemory or OpenSharedGpuMemory.
                                      ///  This IGpuMemory references memory created either by another process or
                                      ///  another device with the exception of peer access.
            uint32 isExternal   :  1; ///< GPU memory object was created with @ref IDevice::OpenExternalSharedGpuMemory.
                                      ///  This IGpuMemory references memory that was created either by another process
                                      ///  or by a device that doesn't support sharedMemory with this object's device
                                      ///  (i.e., MDA sharing on Windows).
            uint32 isSvmAlloc   :  1; ///< GPU memory is allocated in system memory.
                                      /// Valid only when IOMMUv2 is supported
            uint32 isExecutable :  1; ///< GPU memory is used for execution. Valid only when IOMMUv2 is supported
            uint32 isExternPhys :  1; ///< GPU memory is External Physical memory

            uint32 placeholder0        :  1; ///< Reserved for future memory flag

            uint32 isCompressed :  1; ///< Set for physical allocations where UMD requested PTE.D=1 to enable
                                      ///  GFX12-style distributed compression.
            uint32 reserved     : 23; ///< Reserved for future use
        };
        uint32 u32All;               ///< Flags packed as 32-bit uint.
    } flags;                         ///< GPU memory desc flags.

    uint64 uniqueId; ///< Unique ID given to each GPU memory object, allows client tracking of GPU memory allocations.
};

/// Defines GPU memory sub allocation info. Contains a GPU memory handle to the whole memory. And the offset and size
/// shows where is the sub allocated memory.
struct GpuMemSubAllocInfo
{
    gpusize     address;     ///< Start address of the memory, not including the offset.
    gpusize     offset;      ///< Offset from the start address of the memory.
    gpusize     size;        ///< Size of the memory.
};

/// Specifies a GPU memory object and flags with more specific usage details.  An array of these structures is specified
/// to PAL residency operations.
///
/// @see IDevice::AddGpuMemoryReferences
/// @see IQueue::Submit
struct GpuMemoryRef
{
    union
    {
        struct
        {
            uint32 readOnly :  1;  ///< The allocation will not be written using this reference.
            uint32 reserved : 31;  ///< Reserved for future use.
        };
        uint32 u32All;             ///< Flags packed as 32-bit uint.
    } flags;                       ///< GPU memory reference flags.

    IGpuMemory* pGpuMemory;        ///< The GPU memory object referenced by this residency operation.
};

/// Specifies a Display Output Post-Processing (DOPP) allocation that will be referenced by a submission along with
/// additional info describing how it will be used.
///
/// @see IQueue::Submit
struct DoppRef
{
    union
    {
        struct
        {
            uint32 pfpa        :  1;  ///< Access to this DOPP allocation will be redirected to the primary pending
                                      ///  present (i.e., pre-flip primary access).  If not set, access will
                                      ///  refer to the current onscreen primary.
            uint32 lastPfpaCmd :  1;  ///< This submission will be the last access of this pfpa allocation
                                      ///  for this frame.  The pfpa interval will end once this submit
                                      ///  completes, allowing the corresponding vidPnSource to flip.
                                      ///  This flag is invalid if the pfpa flag is not set.
            uint32 reserved    : 30;  ///< Reserved for future use.
        };
        uint32 u32All;             ///< Flags packed as 32-bit uint.
    } flags;                       ///< GPU memory reference flags.

    IGpuMemory* pGpuMemory;        ///< The GPU memory object referenced by this residency operation.
};

/// Specifies the types of the exporting memory.
enum class ExportHandleType : uint32
{
    Default        = 0,            ///< Let PAL choose the export type
#if PAL_AMDGPU_BUILD
    FileDescriptor,                ///< Export using a Linux file descriptor
    Kms,                           ///< Export through KMS
#endif
};

/// Specifies parameters for export a GPUMemory NT handle from its name.
struct GpuMemoryExportInfo
{
#if PAL_KMT_BUILD
    const _SECURITY_ATTRIBUTES* pSecurityAttributes; ///< It specifies the security descriptor and the inheritable
                                                     ///  attribute.
    const wchar_t*              pNtObjectName;       ///< A name to NT handle, if the object is exported as a NT
                                                     ///  handle with a name, and then the handle can be acquired
                                                     ///  via this name.
    uint32                      accessFlags;         ///< Desried access rights of GPU memory.
#endif
    ExportHandleType            exportType;         ///< Type of handle to use for exporting the memory.
};

/**
 ***********************************************************************************************************************
 * @interface IGpuMemory
 * @brief     Interface representing a GPU-accessible memory allocation.
 *
 * Depending on creation parameters, this could correspond to:
 *
 * + A _real_ memory object, corresponding directly to a physical allocation made on this device (whether it resides in
 *   a local or non-local heap).
 * + A _virtual_ memory object, only consisting of virtual address space that can be mapped on a page basis to pages in
 *   _real_ memory objects via IQueue::RemapVirtualMemoryPages.
 * + Pinned memory, a _real_ memory object created by pinning down client system memory.
 * + Peer memory, a _real_ memory object corresponding to GPU memory that is likely local to another GPU.  Only copy
 *   operations (peer-to-peer transfers) are allowed with this memory.
 * + Opened/shared memory, a _real_ memory object that is fully shared between multiple GPUs, residing in a non-local
 *   heap.
 * + External shared memory, a _real_ memory object that was created by an external process and is fully shared between
 *   multiple GPUs.
 *
 * @see IDevice::CreateGpuMemory
 * @see IDevice::CreatePinnedGpuMemory
 * @see IDevice::OpenSharedGpuMemory
 * @see IDevice::OpenPeerGpuMemory
 * @see IDevice::OpenExternalSharedGpuMemory
 *
 *
 * All of these kinds of GPU memory are assigned a set of fundemental properties specified in GpuMemoryDesc which are
 * either specified by the client or by PAL.  There are specific rules these properties must follow; those rules are
 * documented here to avoid duplication.  Violating these rules will cause the device's corresponding "get size"
 * functions to return an error code, the create/open functions may not validate their arguments.
 *
 *
 * With the exception of external memory objects being opened, PAL will adjust size and base alignments as necessary
 * to meet device requirements. Typically this means going out to OS page boundaries. The client is no longer required
*  to query device requirements and align for PAL.
 *
 * Note that the device alignment requirements apply equally to GPU VAs.  However, other kinds of alignment
 * restrictions (e.g., IGpuMemoryBindable's requirements) may only apply to one of those two properties.  When creating
 * GPU memory objects the client must be careful to set the "alignment" field to the alignment of the GPU VA.
 *
 *
 * Second, the client can't directly specify a memory object's GPU VA but must specify its VA range, limiting which
 * portions of the VA space can be used.  Note that non-external shared and peer GPU memory objects will use the
 * original memory's VA range.  External shared GPU memory always uses the default VA range.
 *
 * The ShadowDescriptorTable VA range is special because it pairs the shadow GPU memory to an existing descriptor GPU
 * memory.  The client must specify the GPU VA of the corresponding DescriptorTable memory when creating a shadow GPU
 * memory object via descrVirtAddr; it must satisfy the alignment requirements of the shadow GPU memory.  Both GPU
 * memory objects must be created on the same device.  Note that descrVirtAddr can be offset into the descriptor
 * allocation such that multiple shadow GPU memory objects correspond to one larger descriptor GPU memory object.
 *
 *
 * The client can further influence the GPU VA of shared and peer GPU memory objects. If the globalGpuVa flag is set
 * when the original GPU memory object is created, PAL will assign any shared or peer GPU memory objects that same VA.
 * Note that globalGpuVa is only supported if globalGpuVaSupport is set in DeviceProperties.
 ***********************************************************************************************************************
 */
class IGpuMemory : public IDestroyable
{
public:
    /// Sets a new priority for this GPU memory object.
    ///
    /// This call is not available for virtual or pinned memory.
    ///
    /// @param [in] priority       New base priority for the GPU memory object.
    ///
    /// @param [in] priorityOffset New priority offset for the GPU memory object. This is a small bias that can be
    ///                            used by the OS to raise the importance of an allocation when there are
    ///                            multiple allocations in the same base priority level. You can think of it as
    ///                            the fractional bits of the priority level.
    ///
    /// @returns Success if the priority was successfully updated.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorUnavailable if this is a virtual or pinned GPU memory object.
    virtual Result SetPriority(
        GpuMemPriority       priority,
        GpuMemPriorityOffset priorityOffset) = 0;

    /// Makes the GPU memory available for CPU access and gives the client a pointer to reference it.
    ///
    /// The allocation should be unmapped by the client once CPU access is complete, although it _is_ legal to keep an
    /// allocation mapped while the GPU references the allocation from a command buffer.
    ///
    /// It is legal to map the allocation multiple times concurrently.  Mapping is not available for pinned or virtual
    /// memory objects.  This call is thread safe for calls referencing this memory object.
    ///
    /// @see Unmap.
    ///
    /// @param [out] ppData CPU pointer to the GPU memory object.
    ///
    /// @returns Success if the map succeeded.  Otherwise, *ppData will not be valid and one of the following errors may
    ///          be returned.
    ///          + ErrorInvalidPointer if ppData is null.
    ///          + ErrorGpuMemoryMapFailed if the object is busy and cannot be mapped by the OS.
    ///          + ErrorNotMappable if the memory object cannot be mapped due to some of its heaps not having the CPU
    ///            visible flag set.
    ///          + ErrorUnavailable if the memory object is not a real allocation.
    virtual Result Map(
        void** ppData) = 0;

    /// Removes CPU access from a previously mapped GPU memory object.
    ///
    /// This call is thread safe for calls referencing the same memory object.
    ///
    /// @see Map
    ///
    /// @returns Success if the unmap succeeded.  Otherwise, one of the following errors may be returned:
    ///          + ErrorGpuMemoryUnmapFailed if the GPU memory object cannot be unlocked.
    ///          + ErrorUnavailable if the GPU memory object is not a real allocation.
    virtual Result Unmap() = 0;

#if PAL_KMT_BUILD || PAL_AMDGPU_BUILD
    /// Returns an OS-specific handle which can be used to refer to this GPU memory object across processes. This will
    /// return a null or invalid handle if the object was not created with the @ref interprocess create flag set.
    ///
    /// @note This function is only available for Linux builds or KMT builds.
    ///
    /// @param [in] handleInfo       The info is used to open handle.
    ///
    /// @returns An OS-specific handle which can be used to access the GPU memory object across processes.
    virtual OsExternalHandle ExportExternalHandle(const GpuMemoryExportInfo& exportInfo) const = 0;
#endif

    /// Returns a structure containing some fundemental information that describes this GPU memory object.
    ///
    /// @returns A reference to this allocation's GpuMemoryDesc.
    const GpuMemoryDesc& Desc() const { return m_desc; }

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

    /// Set SDI remote surface bus address and marker bus address.
    ///
    /// This GPU memory object must have been created with the sdiExternal flag set and with the GpuMemoryCreateInfo
    /// surfaceBusAddr and markerBusAddr fields both set to zero. This function allows clients to defer setting those
    /// addresses until after creation. It must be called exactly once to permanently bind the given SDI addresses to
    /// this GPU memory object.
    ///
    /// @warning An sdiExternal GPU memory object is not complete until its given its SDI addresses! The gpuVirtAddr
    ///          field in this GPU memory's GpuMemoryDesc will not be valid until this function is called!
    ///
    /// @param [in] surfaceBusAddr Surface bus address of Bus Addressable Memory.
    /// @param [in] markerBusAddr  Marker bus address of Bus Addressable Memory. The client can write to the marker
    ///                            and have the GPU wait until a value is written to marker before continuing.
    ///
    /// @returns Success if succeeded. Otherwise, one of the following errors may be returned:
    ///          + ErrorUnavailable if the GPU memory object is not external physical memory or it has already been set.
    ///          + ErrorInvalidValue if one of the input params is 0.
    ///          + One of the escape call failed error.
    virtual Result SetSdiRemoteBusAddress(gpusize surfaceBusAddr, gpusize markerBusAddr) = 0;

protected:
    /// @internal Constructor. Prevent use of new operator on this interface. Client must create objects by explicitly
    /// called the proper create method.
    IGpuMemory() : m_pClientData(nullptr) {}

    /// @internal Destructor.  Prevent use of delete operator on this interface.  Client must destroy objects by
    /// explicitly calling IDestroyable::Destroy() and is responsible for freeing the system memory allocated for the
    /// object on their own.
    virtual ~IGpuMemory() { }

    GpuMemoryDesc m_desc; ///< Information that describes this GPU memory object.

private:
    /// @internal Client data pointer. This can have an arbitrary value and can be returned by calling GetClientData()
    /// and set via SetClientData().
    /// For non-top-layer objects, this will point to the layer above the current object.
    void* m_pClientData;
};

} // Pal
