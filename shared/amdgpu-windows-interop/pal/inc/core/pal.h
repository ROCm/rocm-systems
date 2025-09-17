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
 * @file  pal.h
 * @brief Common include for the Platform Abstraction Library (PAL) interface.  Defines common types, enums, etc.
 ***********************************************************************************************************************
 */

#pragma once

#include "palFormat.h"
#include "palSysUtil.h"

// Forward declarations of global types (must be done outside of Pal namespace).
#if (PAL_KMT_BUILD) && !defined(__unix__)
struct HMONITOR__;
struct HWND__;
#endif

#if PAL_KMT_BUILD
struct _SECURITY_ATTRIBUTES;
#endif

/// Library-wide namespace encapsulating all PAL entities.
namespace Pal
{

typedef Util::int8    int8;     ///< 8-bit integer.
typedef Util::int16   int16;    ///< 16-bit integer.
typedef Util::int32   int32;    ///< 32-bit integer.
typedef Util::int64   int64;    ///< 64-bit integer.
typedef Util::uint8   uint8;    ///< Unsigned 8-bit integer.
typedef Util::uint16  uint16;   ///< Unsigned 16-bit integer.
typedef Util::uint32  uint32;   ///< Unsigned 32-bit integer.
typedef Util::uint64  uint64;   ///< Unsigned 64-bit integer.
typedef Util::gpusize gpusize;  ///< Used to specify GPU addresses and sizes of GPU allocations.  This differs from
                                ///  size_t since the GPU still uses 64-bit addresses on a 32-bit OS.
typedef Util::Result  Result;   ///< The PAL core and utility companion share the same result codes for convenience.

typedef Util::Rational Rational; ///< A ratio of two unsigned integers.

#if defined(_WIN32)
typedef HMONITOR__* OsDisplayHandle;  ///< OsDisplayHandle corresponds to an HMONITOR on Windows.
typedef HWND__*     OsWindowHandle;   ///< OsWindowHandle corresponds to an HWND on Windows.
typedef void*       OsExternalHandle; ///< OsExternalHandle corresponds to a generic HANDLE on Windows

typedef uint32  OsVideoSessionHandle;   ///< OsVideoSessionHandle corresponds to a video session handle on Vulkan.

constexpr OsWindowHandle NullWindowHandle = nullptr; ///< Value representing a null or invalid window handle.
#elif defined(__unix__)

typedef void*   OsDisplayHandle;        ///< The Display Handle for Linux except X11 platform
typedef uint32  OsExternalHandle;       ///< OsExternalHandle corresponds to a generic handle on linux
typedef uint32  OsVideoSessionHandle;   ///< OsVideoSessionHandle corresponds to a video session handle on linux.

/// OsWindowHandle corresponds to a window on X-Windows or surface on Wayland.
union OsWindowHandle
{
    void*  pSurface;  ///< Native surface handle in wayland is a pointer.
    uint64 win;       ///< Native window handle in X is a 32-bit integer (but stored here as 64 bit).
};
constexpr OsWindowHandle NullWindowHandle = {nullptr}; ///< Value representing a null or invalid window handle.

// don't check for the Linux Platform type; just compare the larger member of the union
inline bool operator==(const Pal::OsWindowHandle& lhs, const Pal::OsWindowHandle& rhs)
    { return (lhs.pSurface == rhs.pSurface); }
inline bool operator!=(const Pal::OsWindowHandle& lhs, const Pal::OsWindowHandle& rhs)
    { return (lhs.pSurface != rhs.pSurface); }
#else
#error "Unsupported OS platform detected!"
#endif

#if PAL_CLIENT_EXAMPLE
typedef void*  AddrHandle;      ///< Corresponds to an ADDR_HANDLE.
#endif

constexpr uint32 InvalidVidPnSourceId     = ~0u; ///< In cases where PAL cannot abstract a Windows VidPnSourceId, this
                                                 ///  represents an invalid value. (Note: zero is a valid value.)

constexpr uint32 MaxVertexBuffers         = 32;  ///< Maximum number of vertex buffers per pipeline.
constexpr uint32 MaxColorTargets          = 8;   ///< Maximum number of color targets.
constexpr uint32 MaxStreamOutTargets      = 4;   ///< Maximum number of stream output target buffers.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 936
constexpr uint32 MaxDescriptorSets        = 2;   ///< Maximum number of descriptor sets.
#endif
constexpr uint32 MaxMsaaRasterizerSamples = 16;  ///< Maximum number of MSAA samples supported by the rasterizer.
constexpr uint32 MaxAvailableEngines      = 12;  ///< Maximum number of engines for a particular engine type.
constexpr uint32 MaxNumPlanes             = 3;   ///< Maximum number of format planes.

constexpr uint64 InternalApiPsoHash       = UINT64_MAX;  ///< Default Hash for PAL internal pipelines.

/// Specifies a category of GPU engine.  Each category corresponds directly to a hardware engine. There may be multiple
/// engines available for a given type; the available engines on a particular GPU can be queried via
/// Device::GetProperties, returned in DeviceProperties.engineProperties[].
enum EngineType : uint32
{
    /// Corresponds to the graphics hardware engine (a.k.a. graphcis ring a.k.a 3D).
    EngineTypeUniversal,

    /// Corresponds to asynchronous compute engines (ACE).
    EngineTypeCompute,

    /// Corresponds to SDMA engines.
    EngineTypeDma,

    /// Virtual engine that only supports inserting sleeps, used for implementing frame-pacing.
    EngineTypeTimer,

    /// Number of engine types.
    EngineTypeCount,
};

/// Specifies a category of GPU work.  Each queue type only supports specific types of work. Determining which
/// QueueTypes are supported on which engines can be queried via IDevice::GetProperties, returned in
/// DeviceProperties.engineProperties[].
enum QueueType : uint32
{
    /// Supports graphics commands (draws), compute commands (dispatches), and copy commands.
    QueueTypeUniversal,

    /// Supports compute commands (dispatches), and copy commands.
    QueueTypeCompute,

    /// Supports copy commands.
    QueueTypeDma,

    /// Virtual engine that only supports inserting sleeps, used for implementing frame pacing.
    /// This is a software-only queue.
    QueueTypeTimer,

    /// Number of queue types.
    QueueTypeCount,
};

/// Defines flags for describing which queues are supported.
enum QueueTypeSupport : uint32
{
    SupportQueueTypeUniversal   = (1 << static_cast<uint32>(QueueTypeUniversal)),
    SupportQueueTypeCompute     = (1 << static_cast<uint32>(QueueTypeCompute)),
    SupportQueueTypeDma         = (1 << static_cast<uint32>(QueueTypeDma)),
    SupportQueueTypeTimer       = (1 << static_cast<uint32>(QueueTypeTimer)),

};

// Many command buffers break down into multiple command streams targeting internal sub-engines. For example, Universal
// command buffers build a primary stream (DE) but may also build a second stream for async compute engine (ACE).
enum class SubEngineType : uint32
{
    Primary        = 0, // Subqueue that is the queue itself, rather than an ancillary queue.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 914
    AsyncCompute   = 1, // Auxiliary ACE subqueue, together with a primary subqueue forms a "ganged" submit.
    ConstantEngine = 2, // CP constant update engine that runs in parallel with draw engine.
                        // Internal usage only.
#else
    ConstantEngine = 1, // CP constant update engine that runs in parallel with draw engine.
    AsyncCompute   = 2, // Auxiliary ACE subqueue, together with a primary subqueue forms a "ganged" submit.
    Pup            = 3, // Subqueue that is the queue itself but for PUP-style packets, rather than an
                        // ancillary queue
#endif
    Count,
};

/// Defines the execution priority for a queue, specified either at queue creation or via IQueue::SetExecutionPriority()
/// on platforms that support it.  QueuePriority::Normal corresponds to the default priority.
enum class QueuePriority : uint32
{
    Normal   =  0,  ///< Normal priority (default).
    Idle     =  1,  ///< Idle, or low priority (lower than Normal).
    Medium   =  2,  ///< Medium priority (higher than Normal).
    High     =  3,  ///< High priority (higher than Normal).
    Realtime =  4,  ///< Real time priority (higher than Normal).
    Count
};

/// Defines flags for describing which queue priority levels are supported.
enum QueuePrioritySupport : uint32
{
    SupportQueuePriorityNormal   = (1 << static_cast<uint32>(QueuePriority::Normal)),
    SupportQueuePriorityIdle     = (1 << static_cast<uint32>(QueuePriority::Idle)),
    SupportQueuePriorityMedium   = (1 << static_cast<uint32>(QueuePriority::Medium)),
    SupportQueuePriorityHigh     = (1 << static_cast<uint32>(QueuePriority::High)),
    SupportQueuePriorityRealtime = (1 << static_cast<uint32>(QueuePriority::Realtime)),
};

/// Selects one of a few possible memory heaps accessible by a GPU.
enum GpuHeap : uint32
{
    GpuHeapLocal         = 0x0,  ///< Local heap visible to the CPU.
    GpuHeapInvisible     = 0x1,  ///< Local heap not visible to the CPU.
    GpuHeapGartUswc      = 0x2,  ///< GPU-accessible uncached system memory.
    GpuHeapGartCacheable = 0x3,  ///< GPU-accessible cached system memory.
    GpuHeapCount
};

/// Describes the desired access for a memory allocation.
enum GpuHeapAccess : uint32
{
    GpuHeapAccessExplicit       = 0x0, ///< Memory access is not known. Heaps will be explicitly defined.
    GpuHeapAccessCpuNoAccess    = 0x1, ///< Memory access from CPU not required.
    GpuHeapAccessGpuMostly      = 0x2, ///< Memory optimized for reads/writes from GPU and accessible from CPU.
    GpuHeapAccessCpuReadMostly  = 0x3, ///< Memory optimized for reads from CPU.
    GpuHeapAccessCpuWriteMostly = 0x4, ///< Memory optimized for writes from CPU.
    GpuHeapAccessCpuMostly      = 0x5, ///< Memory optimized for read/writes from CPU.
    GpuHeapAccessCount
};

#if defined(__unix__)
/// Describes possible handle types.
enum class HandleType : uint32
{
    GemFlinkName      = 0x0, ///< GEM flink name (needs DRM authentication, used by DRI2)
    Kms               = 0x1, ///< KMS handle which is used by all driver ioctls
    DmaBufFd          = 0x2, ///< DMA-buf fd handle
    KmsNoImport       = 0x3, ///< Deprecated in favour of and same behaviour as HandleTypeDmaBufFd, use that instead of this
};
#endif

/// Comparison function determines how a pass/fail condition is determined between two values.  For depth/stencil
/// comparison, the first value comes from source data and the second value comes from destination data.
enum class CompareFunc : uint8
{
    Never        = 0x0,
    Less         = 0x1,
    Equal        = 0x2,
    LessEqual    = 0x3,
    Greater      = 0x4,
    NotEqual     = 0x5,
    GreaterEqual = 0x6,
    _Always      = 0x7,

    // Unfortunately for Linux clients, X.h includes a "#define Always 2" macro.  Clients have their choice of either
    // undefing Always before including this header or using _Always when dealing with PAL.
#ifndef Always
    Always       = _Always,
#endif

    Count
};

/// Defines an offset into a 2D pixel region.
struct Offset2d
{
    int32 x;  ///< X offset.
    int32 y;  ///< Y offset.
};

/// Defines an offset into a 3D pixel region.
struct Offset3d
{
    int32 x;  ///< X offset.
    int32 y;  ///< Y offset.
    int32 z;  ///< Z offset.
};

/// Defines an floating-point offset into a 3D pixel region.
struct Offset3dFloat
{
    float x;  ///< X offset.
    float y;  ///< Y offset.
    float z;  ///< Z offset.
};

/// Defines a width and height for a 2D image region. The dimensions could be pixels, blocks, or bytes
/// depending on context, so be sure to check documentation for the PAL interface of interest to be sure you
/// get it right.
struct Extent2d
{
    uint32 width;   ///< Width of region.
    uint32 height;  ///< Height of region.
};

/// Defines a signed width and height, for a 2D image region. The dimensions could be pixels, blocks, or bytes
/// depending on context, so be sure to check documentation for the PAL interface of interest to be sure you
/// get it right.
struct SignedExtent2d
{
    int32 width;    ///< Width of region.
    int32 height;   ///< Height of region.
};

/// Defines a width, height, and depth for a 3D image region. The dimensions could be pixels, blocks, or bytes
/// depending on context, so be sure to check documentation for the PAL interface of interest to be sure you
/// get it right.
struct Extent3d
{
    uint32 width;   ///< Width of region.
    uint32 height;  ///< Height of region.
    uint32 depth;   ///< Depth of region.
};

constexpr bool operator==(const Extent3d& x, const Extent3d& y)
{
    return (x.width == y.width) && (x.height == y.height) && (x.depth == y.depth);
}

constexpr bool operator!=(const Extent3d& x, const Extent3d& y) { return (x == y) == false; }

/// Defines a signed width, height, and depth for a 3D image region. The dimensions could be pixels, blocks, or bytes
/// depending on context, so be sure to check documentation for the PAL interface of interest to be sure you
/// get it right.
struct SignedExtent3d
{
    int32 width;    ///< Width of region.
    int32 height;   ///< Height of region.
    int32 depth;    ///< Depth of region.
};

/// Defines a floating-point width, height, and depth for a 3D image region. The dimensions could be pixels, blocks, or
/// bytes depending on context, so be sure to check documentation for the PAL interface of interest to be sure you
/// get it right.
struct Extent3dFloat
{
    float width;    ///< Width of region.
    float height;   ///< Height of region.
    float depth;    ///< Depth of region.
};

/// Defines a region in 1D space.
struct Range
{
    int32  offset;  ///< Starting position.
    uint32 extent;  ///< Region size.
};

/// Defines a rectangular region in 2D space.
struct Rect
{
    Offset2d offset;  ///< Top left corner.
    Extent2d extent;  ///< Rectangle width and height.
};

/// Defines a cubic region in 3D space.
struct Box
{
    Offset3d offset;  ///< Top left front corner.
    Extent3d extent;  ///< Box width, height and depth.
};

/// ShaderHash represents a 128-bit shader hash.
struct ShaderHash
{
    uint64 lower;   ///< Lower 64-bits of hash
    uint64 upper;   ///< Upper 64-bits of hash
};

/// PipelineHash represents a concatenated pair of 64-bit hashes.
struct PipelineHash
{
    uint64 stable;   ///< Lower 64-bits of hash.  "Stable" portion, suitable for e.g. shader replacement use cases.
    uint64 unique;   ///< Upper 64-bits of hash.  "Unique" portion, suitable for e.g. pipeline cache use cases.
};

/// Common shader pre and post compilation stats.
struct CommonShaderStats
{
    uint32  numUsedVgprs;               ///< Number of VGPRs used by this shader
    uint32  numUsedSgprs;               ///< Number of SGPRs used by this shader

    uint32  ldsSizePerThreadGroup;      ///< LDS size per thread group in bytes.
    size_t  ldsUsageSizeInBytes;        ///< LDS usage by this shader.

    size_t  scratchMemUsageInBytes;     ///< Amount of scratch mem used by this shader.
    gpusize gpuVirtAddress;             ///< Gpu mem address of shader ISA code.

    union
    {
        struct
        {
            uint32 isWave32 :  1;  ///< If set, specifies that the shader is compiled in wave32 mode.
            uint32 reserved : 31;  ///< Reserved for future use.
        };
        uint32 u32All;  ///< Flags packed as a 32-bit uint.
    } flags;            ///< Shader compilation stat flags.
};

/// Per-thread stack sizes
struct CompilerStackSizes
{
    uint32 backendSize;  ///< Managed by compiler backend
    uint32 frontendSize; ///< Managed by compiler frontend
};

///@{
/// Determines whether two ShaderHashes or PipelineHashes are equal.
///
/// @param  [in]    hash1    The first 128-bit shader hash or pipeline hash
/// @param  [in]    hash2    The second 128-bit shader hash or pipeline hash
///
/// @returns True if the hashes are equal.
constexpr bool ShaderHashesEqual(const ShaderHash hash1, const ShaderHash hash2)
    { return ((hash1.lower  == hash2.lower)  && (hash1.upper  == hash2.upper)); }
constexpr bool operator==(const ShaderHash hash1, const ShaderHash hash2)
    { return ((hash1.lower  == hash2.lower)  && (hash1.upper  == hash2.upper)); }
constexpr bool operator!=(const ShaderHash hash1, const ShaderHash hash2)
    { return ((hash1.lower  != hash2.lower)  || (hash1.upper  != hash2.upper)); }
constexpr bool PipelineHashesEqual(const PipelineHash hash1, const PipelineHash hash2)
    { return ((hash1.stable == hash2.stable) && (hash1.unique == hash2.unique)); }
///@}

///@{
/// Determines whether the given ShaderHash or PipelineHash is non-zero.
///
/// @param  [in]    hash    A 128-bit shader hash or pipeline hash
///
/// @returns True if the hash is non-zero.
constexpr bool ShaderHashIsNonzero(const ShaderHash hash)     { return ((hash.upper  | hash.lower)  != 0); }
constexpr bool PipelineHashIsNonzero(const PipelineHash hash) { return ((hash.stable | hash.unique) != 0); }
///@}

/// Specifies the Display Output Post-Processing (DOPP) desktop texture information, which are provided by OpenGL via
/// interop.  The DOPP is an OpenGL extension to allow its client to access the desktop texture directly without the
/// need of copying to system memory.  This is only supported on Windows.
struct DoppDesktopInfo
{
    gpusize gpuVirtAddr;    ///< The VA of the dopp desktop texture. Set to 0 for the non-dopp resource.
    uint32  vidPnSourceId;  ///< Display source id of the dopp desktop texture.
};

/// Specifies the Direct Capture resource information. Direct Capture is an extension that allows to access on-screen
/// primary, motion vectors, depth, and camera matrix directly. This is only supported on Windows.
struct DirectCaptureInfo
{
    uint32  vidPnSourceId;  ///< VidPnSource ID of the on-screen primary.
    union
    {
        struct
        {
            uint32 preflip              :  1;  ///< Requires pre-flip primary access
            uint32 postflip             :  1;  ///< Requires post-flip primary access. A DirectCapture resource cannot
                                               ///  have pre-flip and post-flip access at the same time
            uint32 accessDesktop        :  1;  ///< Requires acces to the desktop
            uint32 shared               :  1;  ///< This resource will be shared between APIs
            uint32 frameGenRatio        :  4;  ///< Frame generation ratio
            uint32 paceGeneratedFrame   :  1;  ///< Requires pacing the generated frames
            uint32 requiresDisplayDcc   :  1;  ///< Requires display dcc support
            uint32 requestMotionVectors :  1;  ///< Request DirectCapture access to motion vector data if available
            uint32 requestDepth         :  1;  ///< Request DirectCapture access to depth data if available
            uint32 requestCamera        :  1;  ///< Request DirectCapture access to camera matrix data if available
            uint32 initMotionVectors    :  1;  ///< Initialize the DirectCapture resource to access motion vector data
            uint32 initDepth            :  1;  ///< Initialize the DirectCapture resource to access depth data
            uint32 initCamera           :  1;  ///< Initialize the DirectCapture resource to access camera matrix
            uint32 requestHudLessImage  :  1;  ///< Request DirectCapture access to HUD less image if available
            uint32 initHudLessImage     :  1;  ///< Initialize the DirectCapture resource to access HUD less image
            uint32 reserved             : 14;
        };
        uint32 u32All;
    } usageFlags;

    OsExternalHandle hNewFrameEvent;   ///< Event to notify of a new frame available for pre-flip or post-flip access
    OsExternalHandle hFatalErrorEvent; ///< Event to notify of a fatal error
};

/// Specifies parameters for opening a shared GPU resource from a non-PAL device or non-local process.
struct ExternalResourceOpenInfo
{
    OsExternalHandle hExternalResource;         ///< External GPU resource from another non-PAL device to open.
#if defined(__unix__)
    HandleType       handleType;                ///< Type of the external GPU resource to be opened.
#endif

    union
    {
        struct
        {
            uint32 ntHandle           :  1; ///< The provided hExternalResource is an NT handle instead of a default
                                            ///  KMT handle.
            uint32 androidHwBufHandle :  1; ///< The provided hExternalResource is android hardware buffer handle
                                            ///  instead of fd.
            uint32 isDopp             :  1; ///< This is a Dopp texture, doppDesktopInfo is in use.
            uint32 isDirectCapture    :  1; ///< This is a Direct Capture resource, directCaptureInfo is in use.
            uint32 globalGpuVa        :  1; ///< The GPU virtual address must be visible to all devices.
            uint32 reserved           : 27; ///< Reserved for future use.
        };
        uint32 u32All;            ///< Flags packed as 32-bit uint.
    } flags;                      ///< External resource open flags.

    union
    {
        DoppDesktopInfo   doppDesktopInfo;      ///< The information of dopp desktop texture.
        DirectCaptureInfo directCaptureInfo;    ///< The information of direct capture resource.
    };
};

/// Packed pixel display enumeration.
///
/// In the medical imaging market space, there are several 10-bit per component color and grayscale displays
/// available.In addition to being high precision, these displays tend to be very high resolution.For grayscale
/// displays,one method of getting high pixel resolution in 10b precision is a proprietary method called
/// "packed pixel".Each of these packed pixel formats packs two/three 10-bit luminance values into a single
/// R8G8B8 pixel.
///
/// Example Displays:
///
///     EIZO GS510
///     NEC MD21GS
///     TOTOKU ME55Xi2
///     FIMI 3/5MP
///
///
///   The enumerations are named in a way to describe the format of the packed pixels. Names for
///   formats with two or three pixels packed into a single word (corresponding to a simple RGB pixel)
///   follow this convention:
///
///       LLLLLL_RRRRRR (L=left pixel, R=right pixel) or
///       LLL_MMM_RRR (L=left pixel, M=middle pixel, R=right pixel)
///
///   The bit order for a pixel follows this convention:
///
///       (ColorBand)MSB(ColorBand)LSB
///
///   For example: G70B54 means that the MSBs are in 7-0 of the green channel, and the LSBs
///   are stored in bits 5-4.
///
enum class PackedPixelType : uint32
{
    NotPacked = 0,          ///< Pixels not packed, for standard color RGB8 monitor
    SplitG70B54_R70B10,     ///< 10-bit mono, split screen
    SplitB70G10_R70G76,     ///< 10-bit mono, split screen
    G70B54_R70B10,          ///< 10-bit mono, 2 adjacent pixels
    B70R32_G70R76,          ///< 10-bit mono, 2 adjacent pixels
    B70R30_G70R74,          ///< 12-bit mono, 2 adjacent pixels
    B70_G70_R70,            ///< 8-bit mono, 3 adjacent pixels
    R70G76,                 ///< 10-bit mono, single pixel
    G70B54,                 ///< 10-bit mono, single pixel
    Native,                 ///< 10-bit color, without packing
};

/// Enumerates the logging priority levels supported by PAL.
enum class LogLevel : uint32
{
    Debug = 0, ///< Debug messages
    Verbose,   ///< High frequency messages
    Info,      ///< Low frequency messages
    Alert,     ///< Warnings
    Error,     ///< Critical issues
    Always     ///< All messages
};

/// Enumerates all log categories explicitly defined by PAL
enum class LogCategory : uint64
{
    Correctness = 0, ///< Application correctness
    Performance,     ///< Application performance
    Internal,        ///< Internal logging
    Display,         ///< Display Info
    Count
};

/// String table used to register log categories
constexpr const char* LogCategoryTable[] =
{
    "Correctness",
    "Performance",
    "Internal",
    "Display"
};

/// Typedef for log category masks.
typedef uint64 LogCategoryMask;

/// Log category mask for messages related to application correctness
constexpr LogCategoryMask LogCategoryMaskCorrectness = (1 << static_cast<uint32>(LogCategory::Correctness));

/// Log category mask for messages related to application performance
constexpr LogCategoryMask LogCategoryMaskPerformance = (1 << static_cast<uint32>(LogCategory::Performance));

/// Log category mask for messages related to internal messages
constexpr LogCategoryMask LogCategoryMaskInternal    = (1 << static_cast<uint32>(LogCategory::Internal));

/// Log category mask for messages related to display information (e.g. HDR format)
constexpr LogCategoryMask LogCategoryMaskDisplay = (1 << static_cast<uint32>(LogCategory::Display));

/// Defines the modes that the GPU Profiling layer can be enabled with. If the GpuProfilerMode is
/// GpuProfilerTraceEnabledTtv or GpuProfilerTraceEnabledRgp, then the GpuProfilerConfig_TraceModeMask is examined to
/// configure the trace type (spm, sqtt or both) requested.
enum GpuProfilerMode : uint32
{
    GpuProfilerDisabled              = 0, ///< Gpu Profiler is disabled.
    GpuProfilerCounterAndTimingOnly  = 1, ///< Traces are disabled but perf counter and timing operations are enabled.
    GpuProfilerTraceEnabledTtv       = 2, ///< Traces are output in format (.csv, .out) for Thread trace viewer.
    GpuProfilerTraceEnabledRgp       = 3, ///< Trace data is output as .rgp file for Radeon Gpu Profiler.
};

// Defines the trigger keys for capturing the GPU profiler.
typedef Util::KeyCode GpuProfilerCaptureTriggerKey;

#define PAL_EVENT_LOGGING_VERSION 528

/// This enumeration identifies the source/owner of a resource object, used for event logging.
enum ResourceOwner : uint32
{
    ResourceOwnerApplication = 0,    ///< The resource is owned by the application
    ResourceOwnerPalClient   = 1,    ///< The resource is owned by the PAL client
    ResourceOwnerPal         = 2,    ///< The resource is owned by PAL
    ResourceOwnerUnknown     = 3,    ///< The resource owner is unknown
};

/// This enumeration lists the usage/category of a resource object to give context in event logging.
enum ResourceCategory : uint32
{
    ResourceCategoryApplication = 0,    ///< The resource is used by the application.
    ResourceCategoryRpm         = 1,    ///< The resource is used by RPM
    ResourceCategoryProfiling   = 2,    ///< The resource is used for profiling (e.g. SQTT, SPM, etc)
    ResourceCategoryDebug       = 3,    ///< The resource is used for debug purposes
    ResourceCategoryRayTracing  = 4,    ///< The resource is used for ray tracing
    ResourceCategoryVideo       = 5,    ///< The resource is used for video encode/decode
    ResourceCategoryMisc        = 6,    ///< Miscellaneous, resource doesn't fit in any of the above categories
    ResourceCategoryUnknown     = 7,    ///< The resource category is unknown
};

/// Set of information about resource ownership and usage, used for event logging.
struct ResourceEventInfo
{
    ResourceOwner    owner;     ///< Resource owner
    ResourceCategory category;  ///< Resource category
};

/// General purpose on/off/default tri-state enum.
enum class TriState : uint8
{
    Default = 0,  ///< Let implementation decide whether to enable or disable
    Enable  = 1,  ///< Force enable
    Disable = 2,  ///< Force disable
    Count
};

/// Defines the modes that the GPU Profiling layer can be enabled with.
/**
 ***********************************************************************************************************************
 * @mainpage
 *
 * Introduction
 * ------------
 * The Platform Abstraction Library (PAL) provides hardware and OS abstractions for Radeon (GCN+) user-mode 3D graphics
 * drivers.  The level of abstraction is chosen to support performant driver implementations of several APIs while
 * hiding the client from hardware and operating system details.
 *
 * PAL client drivers will have no HW-specific code; their responsibility is to translate API/DDI commands into PAL
 * commands as efficiently as possible.  This means that the client should be unaware of hardware registers, PM4
 * commands, SP3 shaders, etc.  However, PAL is an abstraction of AMD hardware only, so many things in the PAL interface
 * have an obvious correlation to hardware features.
 *
 * PAL client drivers should have little OS-specific code.  PAL and its companion utility collection provide
 * OS abstractions for almost everything a client might need, but there are some cases where this is unavoidable:
 *
 * + Handling dynamic library infrastructure.  I.e., the client has to implement DllMain() on Windows, etc.
 * + OS-specific APIs or extensions.  DX may have Windows-specific functionality in the core API, and Vulkan may
 *   export certain OS-specific features as extensions (like for presenting contents to the screen).
 * + Single OS clients (e.g., DX) may choose to make OS-specific calls directly simply out of convenience with no down
 *   side.
 *
 *
 * The following diagram illustrates the software stack when running a 3D application with a PAL-based UMD.  Non-AMD
 * components are in gray, UMD client code is blue, AMD static libs linked into the UMD are green, and the AMD KMD
 * is in red.
 *
 * @image html swStack.png
 *
 * PAL is a relatively _thick_ abstraction layer, typically accounting for the majority of code (excluding SC) in any
 * particular UMD built on PAL.  The level of abstraction tends to be higher in areas where client APIs are similar,
 * and lower (closer to hardware) in areas where client APIs diverge significantly.  The overall philosophy is to share
 * as much code as possible without impacting client driver performance.  Our committed goal is that CPU-limited
 * performance should be within 5% of what a native solution could achieve, and GPU-limited performance should be within
 * 2%.
 *
 * PAL uses a C++ interface.  The public interface is defined in .../pal/inc, and client must _only_ include headers
 * from that directory.  The interface is spread over many header files - typically one per class - in order to clarify
 * dependencies and reduce build times.  There are two sub-directories in .../pal/inc:
 *
 * + <b>.../pal/inc/core</b>    - Defines the PAL Core (see @ref Overview).
 * + <b>.../pal/inc/gpuUtil</b> - Defines the PAL GPU Utility Collection (see @ref GpuUtilOverview).
 * + <b>.../pal/inc/util</b>    - Defines the PAL Utility Collection (see @ref UtilOverview).
 *
 *
 * @copydoc VersionHistory
 *
 * Next: @ref Build
 ***********************************************************************************************************************
 */

/**
 ***********************************************************************************************************************
 * @page Overview PAL Core Overview
 *
 * ### Introduction
 * PAL's core interface is defined in the @ref Pal namespace, and defines an object-oriented model for interacting with
 * the GPU and OS.  The interface closely resembles the Vulkan and DX12 APIs.  Some common features of these
 * APIs that are central to the PAL interface:
 *
 * - All shader stages, and some additional "shader adjacent" state, are glommed together into a monolithic pipeline
 *   object.
 * - Explicit, free-threaded command buffer generation.
 * - Support for multiple, asynchronous engines for executing GPU work (graphics, compute, DMA).
 * - Explicit system and GPU memory management.
 * - Flexible shader resource binding model.
 * - Explicit management of stalls, cache flushes, and compression state changes.
 *
 * However, as a common component supporting multiple APIs, the PAL interface tends to be lower level in places where
 * client APIs diverge.
 *
 * ### Settings
 * The PAL library has a number of configuration settings available for the client to modify either programmatically
 * or via external settings.  PAL also includes infrastructure for building/loading client-specific settings.
 * See @ref Settings for a detailed description of this support.
 *
 * ### Initialization
 * The first step to interacting with the PAL core is creating an IPlatform object and enumerating IDevice objects
 * representing GPUs attached to the system and, optionally, IScreen objects representing displays attached to the
 * system.  See @ref LibInit for a detailed description.
 *
 * ### System Memory Allocation
 * Clients have a lot of control over PAL's system memory allocations.  Most PAL objects require the client to provide
 * system memory; the client first calls a GetSize() method and then passes a pointer to PAL on the actual create call.
 * Further, when PAL needs to make an internal allocation, it will optionally call a client callback, which can be
 * specified on platform creation.  This callback will specify a category for the allocation, which may imply an
 * expected lifetime.
 *
 * ### Interface Classes
 * The following diagram illustrates the relationship of some key PAL interfaces and how they interact to render a
 * typical frame in a modern game.  Below that is a listing of all of PAL's interface classes, and a very brief
 * description of their purpose.  Follow the link for each interface to see detailed reference documentation.
 *
 * @image html scheduling.png
 *
 * - __OS Abstractions__
 *   + _IPlatform_: Root-level object created by clients that interact with PAL.  Mostly responsible for enumerating
 *                  devices and screens attached to the system and returning any system-wide properties.<br><br>
 *   + _IDevice_: Configurable context for querying properties of a particular GPU and interacting with it.  Acts as a
 *                factory for almost all other PAL objects.<br><br>
 *   + _IQueue_: A device has one or more _engines_ which are able to issue certain types of work.  Tahiti, for example,
 *               has 1 universal engine (supports graphics, compute, or copy commands), 2 compute engines (support
 *               compute or copy commands), and 2 DMA engines (support only copy commands).  An IQueue object is a
 *               context for submitting work on a particular engine.  This mainly takes the form of submitting command
 *               buffers and presenting images to the screen.  Work performed in a queue will be started in order, but
 *               work executed on different queues (even if the queues reference the same engine) is not guaranteed
 *               to be ordered without explicit synchronization.<br><br>
 *   + _IQueueSemaphore_: Queue semaphores can be signaled and waited on from an IQueue in order to control execution
 *                        order between queues.<br><br>
 *   + _IFence_: Used for coarse-grain CPU/GPU synchronization.  Fences can be signalled from the GPU as part of a
 *               command buffer submission on a queue, then waited on from the CPU.<br><br>
 *   + _IGpuMemory_: Represents a GPU-accessible memory allocation.  Can either be virtual (only VA allocation which
 *                   must be explicitly mapped via an IQueue operation) or physical.  Residency of physical allocations
 *                   must be managed by the client either globally for a device (IDevice::AddGpuMemoryReferences) or by
 *                   specifying allocations referenced by command buffers at submit.<br><br>
 *   + _ICmdAllocator_: GPU memory allocation pool used for backing an ICmdBuffer.  The client is free to create one
 *                      allocator per device, or one per thread to remove thread contention.<br><br>
 *   + _IScreen_: Represents a display attached to the system.  Mostly used for managing full-screen flip
 *                presents.<br><br>
 *   + _IPrivateScreen_: Represents a display that is not otherwise visible to the OS, typically a VR head mounted
 *                       display.<br><br>
 * - __Hardware IP Abstractions__
 *    + __All IP__
 *      - _ICmdBuffer_: Clients build command buffers to execute the desired work on the GPU, and submit them on a
 *                      corresponding queue.  Different types of work can be executed depending on the _queueType_ of
 *                      the command buffer (graphics work, compute work, DMA work).<br><br>
 *      - _IImage_: Images are a 1D, 2D, or 3D collection of pixels (i.e., _texture_) that can be accessed by the
 *                  GPU in various ways: texture sampling, BLT source/destination, UAV, etc.<br><br>
 *    + __GFXIP-only__
 *      - _IShader_: Container for shader byte code used as an input to pipeline creation.  No compilation occurs
 *                   until an IPipeline is created.  Currently, AMDIL is the only supported input language.<br><br>
 *      - _IPipeline_: Comprised of all shader stages (CS for compute, VS/HS/DS/GS/PS for graphics), resource mappings
 *                     describing how user data entries are to be used by the shaders, and some other fixed-function
 *                     state like depth/color formats, blend enable, MSAA enable, etc.<br><br>
 *      - _IColorTargetView_: IImage view allowing the image to be bound as a color target (i.e., RTV.).<br><br>
 *      - _IDepthStencilView_: IImage view allowing the image to be bound as a depth/stencil target (i.e., DSV).<br><br>
 *      - _IGpuEvent_: Used for fine-grained (intra-command buffer) synchronization between the CPU and GPU.  GPU
 *                     events can be set/reset from either the CPU or GPU and waited on from either.<br><br>
 *      - _IQueryPool_: Collection of query slots for tracking occlusion or pipeline stats query results.<br><br>
 *      - __Dynamic State Objects__: _IColorBlendState_, _IDepthStencilState_, _IMsaaState_, _IScissorState_,
 *                                   and _IViewportState_ define logical collections of related fixed function graphics
 *                                   state, similar to DX11.<br><br>
 *      - _IPerfExperiment_: Used for gathering performance counter and thread trace data.<br><br>
 *      - _IBorderColorPalette_: Provides a collection of indexable colors for use by samplers that clamp to an
 *                               arbitrary border color.<br><br>
 * - __Common Base Classes__
 *   + _IDestroyable_: Defines a _Destroy()_ method for the PAL interface.  Calling _Destroy()_ will release any
 *                     internally allocated resources for the object, but the client is still responsible for freeing
 *                     the system memory provided for the object.<br><br>
 *   + _IGpuMemoryBindable_: Defines a set of methods for binding GPU memory to the object.  Interfaces that inherit
 *                           _IGpuMemoryBindable_ require GPU memory in order to be used by the GPU.  The client
 *                           must query the requirements (e.g., alignment, size, heaps) and allocate/bind GPU memory
 *                           for the object.  _IGpuMemoryBindable_ inherits from _IDestroyable_.<br><br>
 *
 * ### %Format Info
 * Several helper methods are available for dealing with image formats in the @ref Formats namespace.
 *
 * ### Graphics/Compute Execution Model
 * Most graphics/compute work is defined by first binding a set of states then issuing a draw or dispatch command to
 * kick off the work.  The complete set of graphics states available in PAL is illustrated below; compute is a subset
 * of this that only includes the pipeline, user data entries, and border color palette.
 *
 * @image html stateBreakdown.jpg
 *
 * Most of these correspond directly to a PAL interface object above, and these items are bound by calling a
 * corresponding _CmdBind...()_ method in the ICmdBuffer interface.  The states marked in yellow and orange, however,
 * are _immediate_ states for which there is no object, you just specify the required state values in the corresponding
 * _CmdSet...()_ method in the ICmdBuffer interface.
 *
 * User data entries are the way that input resources are specified for the pipeline on an upcoming draw/dispatch.  This
 * mapping is complicated, and is described fully in @ref ResourceBinding.
 *
 * A final complication worth noting is that PAL provides no implicit surface synchronization.  The client is
 * responsible for explicitly inserting barriers to resolve data hazards, flush/invalidate caches, and ensure images
 * are in the proper compression state.  For more detail, see ICmdBuffer::CmdReleaseThenAcquire, CmdRelease, CmdAcquire,
 * CmdReleaseEvent, CmdAcquireEvent and AcquireReleaseInfo.
 *
 ***********************************************************************************************************************
 */

} // Pal
