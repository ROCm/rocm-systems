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
 * @file  palCmdBuffer.h
 * @brief Defines the Platform Abstraction Library (PAL) ICmdBuffer interface and related types.
 ***********************************************************************************************************************
 */

#pragma once

#include "pal.h"
#include "palDevice.h"
#include "palGpuMemory.h"
#include "palImage.h"
#include "palMsaaState.h"
#include "palPipeline.h"
#include "palQueryPool.h"
#include "palCmdTracking.h"

/// HSA kernel dispatch packet typedef
typedef struct hsa_kernel_dispatch_packet_s hsa_kernel_dispatch_packet_t;
/// AMD kernel code typedef
typedef struct amd_kernel_code_s amd_kernel_code_t;
/// AMD kernel descriptor
namespace llvm {namespace amdhsa {struct kernel_descriptor_t;} }

namespace Util
{
class VirtualLinearAllocator;
class Event;
}

namespace Pal
{

// Forward declarations.
class IBorderColorPalette;
class ICmdAllocator;
class ICmdBuffer;
class IColorBlendState;
class IColorTargetView;
class IDepthStencilState;
class IDepthStencilView;
class IGpuEvent;
class IGpuMemory;
class IIndirectCmdGenerator;
class IMsaaState;
class IPerfExperiment;
class IQueue;
class IQueryPool;
enum class PerfTraceMarkerType : uint32;
enum class PointOrigin : uint32;

struct VideoCodecInfo;
struct VideoCodecAuxInfo;

/// Specifies a pipeline bind point (i.e., compute or graphics).
enum class PipelineBindPoint : uint32
{
    Compute     = 0x0,
    Graphics    = 0x1,
    Count
};

/// Fully specifies a type of graphics primitive and vertex ordering for geometry.
enum class PrimitiveTopology : uint8
{
    PointList        = 0x0,
    LineList         = 0x1,
    LineStrip        = 0x2,
    TriangleList     = 0x3,
    TriangleStrip    = 0x4,
    RectList         = 0x5,  ///< Each rect is three 2D axis-aligned rectangle vertices.
    QuadList         = 0x6,
    QuadStrip        = 0x7,
    LineListAdj      = 0x8,
    LineStripAdj     = 0x9,
    TriangleListAdj  = 0xA,
    TriangleStripAdj = 0xB,
    Patch            = 0xC,
    TriangleFan      = 0xD,
    LineLoop         = 0xE,
    Polygon          = 0xF,
    TwoDRectList     = 0x10, ///< Each rect is the bounding box of an arbitrary 2D triangle.
                             ///  Support is optional, see support2DRectList in DeviceProperties.
    Count
};

/// Specifies how triangle primitives should be rasterized.
enum class FillMode : uint8
{
    Points    = 0x0,
    Wireframe = 0x1,
    Solid     = 0x2,
    Count
};

/// Specifies the triangle face direction that should result in culled primitives.
enum class CullMode : uint8
{
    _None         = 0x0,  ///< All triangles are rasterized.
    Front         = 0x1,  ///< Front facing triangles are culled.
    Back          = 0x2,  ///< Back facing triangles are culled.
    FrontAndBack  = 0x3,  ///< All triangles are culled.

    // Unfortunately for Linux clients, X.h includes a "#define None 0" macro.  Clients have their choice of either
    // undefing None before including this header or using _None when dealing with PAL.
#ifndef None
    None = _None,         ///< All triangles are rasterized.
#endif
};

/// Specifies vertex winding order corresponding to a front facing triangle.  @see CullMode.
enum class FaceOrientation : uint8
{
    Ccw = 0x0,  ///< Counter-clockwise vertex winding primitives are front facing.
    Cw  = 0x1   ///< Clockwise vertex winding primitives are front facing.
};

/// Specifies which vertex of a primitive is the _provoking vertex_.  This impacts which vertex's "flat" VS outputs
/// are passed to the PS (i.e., flat shading).
enum class ProvokingVertex : uint8
{
    First = 0x0,
    Last  = 0x1
};

/// Specifies bit size of each element in an index buffer.
enum class IndexType : uint32
{
    Idx8  = 0x0,
    Idx16 = 0x1,
    Idx32 = 0x2,
    Count
};

/// Specifies a memory atomic operation that can be performed from command buffers with ICmdBuffer::CmdMemoryAtomic().
enum class AtomicOp : uint32
{
    AddInt32  = 0x00,
    SubInt32  = 0x01,
    MinUint32 = 0x02,
    MaxUint32 = 0x03,
    MinSint32 = 0x04,
    MaxSint32 = 0x05,
    AndInt32  = 0x06,
    OrInt32   = 0x07,
    XorInt32  = 0x08,
    IncUint32 = 0x09,
    DecUint32 = 0x0A,
    AddInt64  = 0x0B,
    SubInt64  = 0x0C,
    MinUint64 = 0x0D,
    MaxUint64 = 0x0E,
    MinSint64 = 0x0F,
    MaxSint64 = 0x10,
    AndInt64  = 0x11,
    OrInt64   = 0x12,
    XorInt64  = 0x13,
    IncUint64 = 0x14,
    DecUint64 = 0x15,
    Count
};

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 928
/// Specifies the point in the GPU pipeline where an action should take place.
///
/// Relevant operations include setting GPU events, waiting on GPU events in hardware, or writing timestamps.
///
/// @note The numeric value of these enums are ordered such that a "newState < oldState" comparison will generally yield
///        true if a stall is necessary to resolve a hazard between those two pipe points.  This guideline does not
///        hold up when comparing PreRasterization or PostPs with PostCs, as CS work is not properly pipelined with
///        graphics shader work.
///
/// @see ICmdBuffer::CmdSetEvent()
/// @see ICmdBuffer::CmdResetEvent()
/// @see ICmdBuffer::CmdPredicateEvent()
/// @see ICmdBuffer::CmdBarrier()
/// @see ICmdBuffer::CmdWriteTimestamp()
/// @see ICmdBuffer::CmdWriteImmediate()
enum HwPipePoint : uint32
{
    HwPipeTop              = 0x0,                   ///< Earliest possible point in the GPU pipeline (CP PFP), can be
                                                    ///  used as wait point for indirect args and index buffer fetch.
    HwPipePostPrefetch     = 0x1,                   ///< Indirect arguments have been fetched for all prior
                                                    ///  draws/dispatches (CP ME).
    HwPipePreRasterization = 0x2,                   ///< All prior generated VS/HS/DS/GS waves have completed, can be
                                                    ///  used as release point for VB/IB fetch and streamout target.
    HwPipePostPs           = 0x3,                   ///< All prior generated PS waves have completed.
                                                    ///  Only valid as a pipe point to wait on (release point).
    HwPipePreColorTarget   = 0x4,                   ///< Represents the same point in pipe to HwPipePostPs, but provides
                                                    ///  clients with a better option to accurately specify the pipeline
                                                    ///  sync request. And PAL uses it as entry-point to add partial
                                                    ///  flushes to prevent write-after-read hazard from corner cases.
                                                    ///  Only valid as a wait point (acquire point).
    HwPipePreIndexBuffer   = HwPipeTop,             ///< As late as possible before index buffer fetches (CP PFP).
    HwPipePostIndexBuffer  = HwPipePreRasterization,///< All prior index buffer fetches have completed.

    // The following points apply to compute-specific work:
    HwPipePreCs            = HwPipePostPrefetch,    ///< As late as possible before CS waves are launched (CP ME).
    HwPipePostCs           = 0x5,                   ///< All prior generated CS waves have completed.

    // The following points apply to BLT-specific work:
    HwPipePreBlt           = HwPipePostPrefetch,    ///< As late as possible before BLT operations are launched.
    HwPipePostBlt          = 0x6,                   ///< All prior requested BLTs have completed.

    HwPipeBottom           = 0x7,                   ///< All prior GPU work (graphics, compute, or BLT) has completed.
    HwPipePointCount
};
#endif

/// Bitmask values that can be OR'ed together to specify a synchronization scope.  See srcStageMask and dstStageMask in
/// @ref AcquireReleaseInfo.
///
/// When specifying an execution dependency at a synchronization point where previous operations must *happen-before*
/// future operations, a mask of these flags specifies a *synchronization scope* that restricts which stages of prior
/// draws, dispatches, or BLTs must *happen-before* which stages of future draws, dispatches, or BLTs.
///
/// Note that flag numerical order does not indicate any happens-before or happens-after relationships. Clients should
/// not compare flags numerically to judge execution order, only barriers can guarantee execution ordering.
enum PipelineStageFlag : uint32
{
    PipelineStageTopOfPipe         = 0x00000001,
    PipelineStageFetchIndirectArgs = 0x00000002,
    PipelineStagePostPrefetch      = 0x00000004,
    PipelineStageFetchIndices      = 0x00000008,
    PipelineStageStreamOut         = 0x00000010,
    PipelineStageVs                = 0x00000020,
    PipelineStageHs                = 0x00000040,
    PipelineStageDs                = 0x00000080,
    PipelineStageGs                = 0x00000100,
    PipelineStagePs                = 0x00000200,
    PipelineStageSampleRate        = 0x00000400,
    PipelineStageEarlyDsTarget     = 0x00000800,
    PipelineStageLateDsTarget      = 0x00001000,
    PipelineStageColorTarget       = 0x00002000,
    PipelineStageCs                = 0x00004000,
    PipelineStageBlt               = 0x00008000,
    PipelineStageBottomOfPipe      = 0x00010000,
    PipelineStageDsTarget          = PipelineStageEarlyDsTarget | PipelineStageLateDsTarget,
    PipelineStageAllStages         = 0x0001FFFF
};

/// Bitmask values that can be ORed together to specify all potential usages of an image at a point in time.  Such a
/// mask should be specified in the usages field of ImageLayout.  These combined usages can be examined by PAL to infer
/// the layout (i.e., compression state) of the image.
///
/// @note There is no layout corresponding to CmdClear*().  The layout flags passed to those functions will determine
///       the expected image layout at that time, and the CmdClear*() implementation will execute a clear that keeps the
///       layout the same.
enum ImageLayoutUsageFlags : uint32
{
    LayoutUninitializedTarget     = 0x00000001,  ///< Initial state of any image that can be used as a color or
                                                 ///  depth/stencil target.  A layout transition out of this state will
                                                 ///  likely result in a mask RAM initialization BLT.  If this bit is
                                                 ///  set, no other bits may be set.
    LayoutColorTarget             = 0x00000002,  ///< Color target bound via CmdBindTargets().  This bit is exclusive
                                                 ///  with LayoutDepthStencilTarget.
    LayoutDepthStencilTarget      = 0x00000004,  ///< Depth/stencil target bound via CmdBindTargets().  This bit is
                                                 ///  exclusive with LayoutColorTarget.
    LayoutShaderRead              = 0x00000008,  ///< Any shader read state including texture, UAV, constant buffer,
                                                 ///  vertex buffer.
    LayoutShaderFmaskBasedRead    = 0x00000010,  ///< Images in this state support the load_fptr AMD IL instruction,
                                                 ///  which will read decompressed fmask in order to access compressed
                                                 ///  MSAA color data from a shader.
    LayoutShaderWrite             = 0x00000020,  ///< Writeable UAV.
    LayoutCopySrc                 = 0x00000040,  ///< CmdCopyImage(), CmdCopyImageToMemory(), CmdScaledCopyImage or
                                                 ///  CmdCopyTiledImageToMemory() source image.
    LayoutCopyDst                 = 0x00000080,  ///< CmdCopyImage(), CmdCopyMemoryToImage(), CmdScaledCopyImage or
                                                 ///  CmdCopyMemoryToTiledImage() destination image.
    LayoutResolveSrc              = 0x00000100,  ///< CmdResolveImage() source.
    LayoutResolveDst              = 0x00000200,  ///< CmdResolveImage() destination.
    LayoutPresentWindowed         = 0x00000400,  ///< Windowed-mode IQueue::Present().
    LayoutPresentFullscreen       = 0x00000800,  ///< Fullscreen (flip) present.  Layout must be supported by the
                                                 ///  display engine.
    LayoutUncompressed            = 0x00001000,  ///< Metadata fully decompressed/expanded layout
    LayoutSampleRate              = 0x00002000,  ///< CmdBindSampleRateImage() source.
    LayoutAllUsages               = 0x00003FFF
};

/// Bitmask values that can be ORed together to specify all potential engines an image might be used on.  Such a
/// mask should be specified in the engines field of ImageLayout.
///
/// If the client API is unable to determine which engines might be used, it should specify all possible engines
/// corresponding to the usage flags.
enum ImageLayoutEngineFlags : uint32
{
    LayoutUniversalEngine       = 0x1,
    LayoutComputeEngine         = 0x2,
    LayoutDmaEngine             = 0x4,
    LayoutVideoEncodeEngine     = 0x8,
    LayoutVideoDecodeEngine     = 0x10,
    LayoutVideoJpegDecodeEngine = 0x20,
    LayoutAllEngines            = 0x3F
};

/// Bitmask values that can be ORed together to specify previous output usage and upcoming input usages of an image or
/// GPU memory in a ICmdBuffer::CmdBarrier() call to ensure cache coherency between those usages.
enum CacheCoherencyUsageFlags : uint32
{
    CoherCpu                = 0x00000001,     ///< Data read or written by CPU.
    CoherShaderRead         = 0x00000002,     ///< Data read by a GPU shader.
    CoherShaderWrite        = 0x00000004,     ///< Data written by a GPU shader.
    CoherCopySrc            = 0x00000008,     ///< Source of a ICmdBuffer::CmdCopy*() call.
    CoherCopyDst            = 0x00000010,     ///< Destination of a ICmdBuffer::CmdCopy*() call.
    CoherColorTarget        = 0x00000020,     ///< Color target.
    CoherDepthStencilTarget = 0x00000040,     ///< Depth stencil target.
    CoherResolveSrc         = 0x00000080,     ///< Source of a CmdResolveImage() call.
    CoherResolveDst         = 0x00000100,     ///< Destination of a CmdResolveImage() call.
    CoherClear              = 0x00000200,     ///< Destination of a CmdClear() call.
    CoherIndirectArgs       = 0x00000400,     ///< Source argument data read by CmdDrawIndirect() and similar functions.
    CoherIndexData          = 0x00000800,     ///< Index buffer data.
    CoherQueueAtomic        = 0x00001000,     ///< Destination of a CmdMemoryAtomic() call.
    CoherTimestamp          = 0x00002000,     ///< Destination of a CmdWriteTimestamp() call.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 914
    CoherStreamOut          = 0x00004000,     ///< Data written as stream output.
    CoherMemory             = 0x00008000,     ///< Data read or written directly from/to memory
    CoherSampleRate         = 0x00010000,     ///< CmdBindSampleRateImage() source.
    CoherPresent            = 0x00020000,     ///< Source of present.
    CoherCp                 = 0x00080000,     ///< HW Command Processor (CP) encompassing the front - end command
    CoherAllUsages          = 0x000FFFFF,     ///<  processing of any queue, including SDMA.
#else
    CoherCeLoad             = 0x00004000,     ///< Source of a CmdLoadCeRam() call.
    CoherCeDump             = 0x00008000,     ///< Destination of CmdDumpCeRam() call.
    CoherStreamOut          = 0x00010000,     ///< Data written as stream output.
    CoherMemory             = 0x00020000,     ///< Data read or written directly from/to memory
    CoherSampleRate         = 0x00040000,     ///< CmdBindSampleRateImage() source.
    CoherPresent            = 0x00080000,     ///< Source of present.
    CoherCp                 = 0x00200000,     ///< HW Command Processor (CP) encompassing the front - end command
    CoherAllUsages          = 0x003FFFFF,     ///<  processing of any queue, including SDMA.
#endif

    CoherShader             = CoherShaderRead | CoherShaderWrite,
    CoherCopy               = CoherCopySrc    | CoherCopyDst,
    CoherResolve            = CoherResolveSrc | CoherResolveDst,
};

/// Bitmask values for the flags parameter of ICmdBuffer::CmdClearColorImage().
enum ClearColorImageFlags : uint32
{
    ColorClearAutoSync   = 0x00000001, ///< PAL will automatically insert required barrier synchronization before
                                       ///  and after the clear assuming all subresources to be cleared are currently
                                       ///  ready for rendering as a color target (as is required by API convention in
                                       ///  DX12).  Allows reduced sync costs in some situations since PAL knows
                                       ///  the details of how the clear will be performed.
    ColorClearForceSlow  = 0x00000002, ///< Force these to use slow clears.
    ColorClearSkipIfSlow = 0x00000004, ///< Only issue the clear if it is a fast clear.
    ColorClearAllFlags   = 0x00000007  ///< Clients should NOT use it, for internal static_assert purpose only.
};

/// Bitmask values for the flags parameter of ICmdBuffer::CmdClearDepthStencil().
enum ClearDepthStencilFlags : uint32
{
    DsClearAutoSync = 0x00000001,   ///< PAL will automatically insert required barrier synchronization before
                                    ///  and after the clear assuming all subresources to be cleared are currently
                                    ///  ready for rendering as a depth/stencil target (as is required by API convention
                                    ///  in DX12).  Allows reduced sync costs in some situations since PAL knows the
                                    ///  details of how the clear will be performed.
    DsClearAllFlags = 0x00000001    ///< Clients should NOT use it, for internal static_assert purpose only.
};

/// Bitmask values for the flags parameter of ICmdBuffer::CmdResolveImage().
enum ResolveImageFlags : uint32
{
    ImageResolveInvertY     = 0x00000001,   ///< PAL will invert the y-axis (flip upside down) of the resolved region to
                                            ///  the destination image.
    ImageResolveDstAsSrgb   = 0x00000002,   ///< If set, a non-srgb destination image will be treated as srgb format.
                                            ///  The flag cannot be set when @ref ImageResolveDstAsNorm is set.
    ImageResolveDstAsNorm   = 0x00000004,   ///< If set, a srgb destination image will be treated as non-srgb format.
                                            ///  The flag cannot be set when @ref ImageResolveDstAsSrgb is set.
    ImageResolveSrcAsNorm   = 0x00000008,   ///< If set, a srgb source image will be treated as non-srgb format.
    ImageResolveAllFlags    = 0x0000000F    ///< Clients should NOT use it, for internal static_assert purpose only.
};

/// Specifies properties for creation of an ICmdBuffer object.  Input structure to IDevice::CreateCmdBuffer().
struct CmdBufferCreateInfo
{
    ICmdAllocator*  pCmdAllocator; ///< The command buffer will use this command allocator to allocate all GPU memory
                                   ///  If the client specifies a null pCmdAllocator, it must call ICmdBuffer::Reset
                                   ///  with a non-null pCmdAllocator before calling ICmdBuffer::Begin.
    QueueType       queueType;     ///< Type of queue commands in this command buffer will target.
                                   ///  This defines the set of allowed actions in the command buffer.
    QueuePriority   queuePriority; ///< Priority level of the queue this command buffer will target.
    EngineType      engineType;    ///< Type of engine the queue commands will run on.

    union
    {
        struct
        {
            /// Indicates that this command buffer will be a "nested" command buffer, instead of a normal, "root"
            /// command buffer.  Nested command buffers differ from root command buffers in how they are sent to the
            /// GPU for execution: root command buffers must be submitted to the hardware by calling
            /// @ref IQueue::Submit, whereas nested command buffers can only be submitted by being executed by a root
            /// command buffer.
            ///
            /// Currently, only Universal and Compute command buffers can be nested. Nesting DMA command buffers is
            /// meaningless and unsupported.  It is an error to attempt to create a nested DMA command buffer.
            ///
            /// @see ICmdBuffer::CmdExecuteNestedCmdBuffers.
            uint32 nested                     :  1;

            /// Dedicated CUs are reserved for this queue. Thus we have to skip CU mask programming.
            uint32 realtimeComputeUnits       :  1;

            /// Target queue uses dispatch tunneling.
            uint32 dispatchTunneling          :  1;

            /// Indicates that each subsequent Dispatch command is desired to be executed in alternating
            /// order of forward and reverse workgroup walk order. This can improve cache locality when
            /// subsequent Dispatches consume data from the previous Dispatch and the overall footprint
            /// does not fit in cache.
            /// This is a best effort as not all implementations or Queues may support this.
            uint32 dispatchPingPongWalk       :  1;

            /// Reserved for future use.
            uint32 reserved                   : 28;
        };

        /// Flags packed as 32-bit uint.
        uint32  u32All;

    }  flags;   ///< Command buffer creation flags.
};

/// Specifies which states will not be bound in a nested command buffer, and instead must be inherited from the calling
/// root-level command buffer.
union InheritedStateFlags
{
    struct
    {
        /// Color and depth target views are inherited from the root-level command buffer. The nested command buffer
        /// should not modify this state.
        uint32 targetViewState :  1;

        /// Occlusion query is inherited from the root-level command buffer. The nested command buffer
        /// should not modify this state.
        uint32 occlusionQuery  :  1;

        /// Predication is inherited from the root-level command buffer. The nested command buffer should not modify
        /// this state.
        uint32  predication    :  1;

        /// Reserved for future usage.
        uint32 reserved        : 29;
    };

    /// Flags packed as 32-bit uint.
    uint32 u32All;
};

/// Specifies parameters inherited from primary command buffer into nested command buffer.
struct InheritedStateParams
{
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 891
    uint32                      colorTargetCount;                            ///< Number of color targets bound in the
                                                                             ///  root-level command buffer.
    SwizzledFormat              colorTargetSwizzledFormats[MaxColorTargets]; ///< Format and swizzle for each color
                                                                             ///  target.
    uint32                      sampleCount[MaxColorTargets];                ///< Sample count for each color target.
#endif
    InheritedStateFlags         stateFlags;                                  ///< States that are inherited from the
                                                                             ///  calling root-level command buffer.
};

/// Specifies optional hints to control command buffer building optimizations.
union CmdBufferBuildFlags
{
    struct
    {
        /// Optimize command buffer building for large sets of draw or dispatch operations that are GPU front-end
        /// limited.  These optimizations include removing redundant PM4 commands and reducing the VGT prim group size.
        /// This flag might increase the CPU overhead of building command buffers.
        uint32 optimizeGpuSmallBatch           :  1;

        /// Optimize command buffer building for exclusive command buffer submission.  Command buffers built with this
        /// flag cannot be submitted if they have already been submitted previously unless the caller guarantees that
        /// they are no longer in use.  This flag allows PAL to modify the contents of command buffers during
        /// submission.
        uint32 optimizeExclusiveSubmit         :  1;

        /// Optimize command buffer building for single command buffer submission.  Command buffers built with this flag
        /// cannot be submitted more than once.  This flag allows PAL to modify the contents of command buffers during
        /// submission.  This flag is a stricter version of optimizeExclusiveSubmit, it is not necessary to set
        /// optimizeExclusiveSubmit if this flag is set.
        uint32 optimizeOneTimeSubmit           :  1;

        /// Indicates that the client is providing custom tessellation distribution settings. If set, it is the clients
        /// responsibility to ensure all 5 (isoline, triangle, quad, donut, trapezoid) factors are provided.
        uint32 optimizeTessDistributionFactors :  1;

        /// Attempt to prefetch shader code into cache before launching draws or dispatches with a freshly bound
        /// pipeline object.  This optimization might increase the CPU overhead of building command buffers and/or
        /// introduce additional front-end GPU bottlenecks.
        uint32 prefetchShaders                 :  1;

        /// Attempt to prefetch the command buffer into cache to avoid bottlenecking the GPU front-end.
        /// This optimization might slightly increase the overhead of some GPU copies and other front-end reads/writes.
        uint32 prefetchCommands                :  1;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 914
        /// Indicates the command buffer will use one or more constant engine commands: CmdLoadCeRam(), CmdDumpCeRam(),
        /// or CmdWriteCeRam()
        uint32 usesCeRamCmds                   :  1;
#else
        uint32 placeholder914                  :  1;
#endif

        /// Indicates that the client would prefer that this nested command buffer not be launched using an IB2 packet.
        /// The calling command buffer will either inline this command buffer into itself or use IB chaining based on if
        /// the optimizeExclusiveSubmit flag is also set. This flag is ignored for root command buffers.
        uint32 disallowNestedLaunchViaIb2      :  1;

        /// placeholder
        uint32 placeholder1                    :  2;

        /// Enable TMZ mode to allow reading TMZ protected allocations. If this command buffer attempts to write
        /// non-TMZ memory, the results are undefined. Only valid for graphics and compute.
        uint32  enableTmz                      :  1;

        uint32 placeholder3                    :  1;

        /// If set, internal operations such as blits, copies, etc. will not affect active Query results.
        /// Otherwise they may affect the results.
        uint32 disableQueryInternalOps         :  1;

        uint32 optimizeContextStatesPerBin     :  1;
        uint32 optimizePersistentStatesPerBin  :  1;

        /// Reserved for future use.
        uint32 reserved                        :  16;
    };

    /// Flags packed as 32-bit uint.
    uint32 u32All;
};

/// Specifies tessellation accum factors.
union TessDistributionFactors
{
    struct
    {
        /// The following 3 factors are used by hardware when distributed tessellation is active: the min tess factors for
        /// each patch processed by a VGT are accumulated. When the sum exceeds this threshold, the next patch is sent to a
        /// different VGT.
        uint32 isoDistributionFactor   : 8;
        uint32 triDistributionFactor   : 8; ///< Recommended to be higher than quad factor.
        uint32 quadDistributionFactor  : 8;
        /// Used by the hardware when distributed tessellation is in DONUT mode: the min tess factor for each patch is
        /// tested against this threshold to determine whether a patch gets split up. If the patch isn't split, it still
        /// increments the accumulator for the Patch distribution factor.
        uint32 donutDistributionFactor : 5;
        /// Used when the distribution mode is TRAPEZOID for quad and tri domain types. The number of donuts in the patch
        /// are compared against this value to detemine whether this donut gets split up into trapezoids (needs the patch to
        /// be in donut mode). A value of 0 or 1 will be treated as 2. The innermost donut is never allowed to be broken
        /// into trapezoids.
        uint32 trapDistributionFactor  : 3;
    };

    /// Values packed as 32-bit uint.
    uint32 u32All;
};

/// Specifies options that direct command buffer building.
struct CmdBufferBuildInfo
{
    /// Command buffer build flags, specifies optional hints to control command buffer build optimizations.
    CmdBufferBuildFlags flags;

    /// Command buffer inherited state and params. If non-null, related state is assumed set in root-level and nested
    /// command buffer should not modify the software states. Any software params that may be needed within nested
    /// command buffer needs to be provided here.
    const InheritedStateParams* pInheritedState;

    /// If non-null, the command buffer will begin with all states set as they are in this previously built command
    /// buffer. Any state specified in pInheritedState is excluded if it is also provided.
    const ICmdBuffer* pStateInheritCmdBuffer;

    /// Optional allocator for PAL to use when allocating temporary memory during command buffer building.  PAL will
    /// stop using this allocator once command building ends.  If no allocator is provided PAL will use an internally
    /// managed allocator instead which may be less efficient.  PAL will use this allocator in two ways:
    /// + Temporary storage within a single command building call.  PAL will rewind the allocator before returning to
    ///   free all memory allocated within the call.
    /// + Temporary storage for the entire command building period.  When Begin() is called, PAL will save the current
    ///   position of the allocator and rewind the allocator to that point when End() is called.  If the client also
    ///   wishes to allocate temporary storage that lasts between command building function calls they must allocate it
    ///   before calling Begin() or PAL will accidentally free it.
    Util::VirtualLinearAllocator* pMemAllocator;

    /// Optional tessellation distribution factors that will overwrite PAL set defaults. Clients must also set the
    /// optimizeTessDistributionFactors flag for these custom factors to take effect.
    /// Nested command buffers inherit this value from the primary.
    TessDistributionFactors clientTessDistributionFactors;

    /// Number of context states per PBB bin.
    /// Client must also set @ref CmdBufferBuildFlags::optimizeContextStatesPerBin for this to take effect.
    uint8 contextStatesPerBin;

    /// Number of persistent states per PBB bin.
    /// Client must also set @ref CmdBufferBuildFlags::optimizePersistentStatesPerBin for this to take effect.
    uint8 persistentStatesPerBin;

    /// Client/app data handle. This can have an arbitrary value and is used to uniquely identify this command buffer.
    uint64 execMarkerClientHandle;
};

/// Specifies info on how a compute shader should use resources.
struct DynamicComputeShaderInfo
{
    float maxWavesPerCu; ///< Limits the number of waves in flight per compute unit.  This can be used to selectively
                         ///  throttle certain workloads that bottleneck multiqueue applications.  For ease of use, a
                         ///  value of zero means no limit is set.  The remaining valid values are in the range (0, 40]
                         ///  and specify the maximum number of waves per compute unit.  If the hardware has one wave
                         ///  limit control for multiple shader stages PAL will select the most strict limit.
                         ///  This option is converted internally to set set HW WavesPerSh setting and the non-integer
                         ///  maxWavesPerCu value provides more flexibility to allow arbitrary WavesPerSh value; for
                         ///  example specify less number of waves than number of CUs per shader array.

    uint32 maxThreadGroupsPerCu; ///< Override the maximum number of threadgroups that a particular CS can run on,
                                 ///  throttling it, to enable more graphics work to complete.  0 disables the limit.

    uint32 tgScheduleCountPerCu; ///< Override the number of threadgroups to schedule on a single compute unit before
                                 ///  moving to the next compute unit. 0 selects optimal default.

    uint32 ldsBytesPerTg; ///< Override the amount of LDS space used per thread-group for this pipeline, in bytes.
                          ///  Zero indicates that the LDS size determined at pipeline-compilation time will be used.
};

/// Specifies info on how a graphics shader should use resources.
struct DynamicGraphicsShaderInfo
{
    float maxWavesPerCu; ///< Limits the number of waves in flight per compute unit.  This can be used to selectively
                         ///  throttle certain workloads that bottleneck multiqueue applications.  For ease of use, a
                         ///  value of zero means no limit is set.  The remaining valid values are in the range (0, 40]
                         ///  and specify the maximum number of waves per compute unit.  If the hardware has one wave
                         ///  limit control for multiple shader stages PAL will select the most strict limit.
                         ///  This option is converted internally to set HW WavesPerSh setting and the non-integer
                         ///  maxWavesPerCu value provides more flexibility to allow arbitrary WavesPerSh value; for
                         ///  example specify less number of waves than number of CUs per shader array.
};

/// Specifies dynamic states of a graphics pipeline
struct DynamicGraphicsState
{
    uint32             colorWriteMask;               ///< Color target write mask. 4b / RT (8 count)
    struct
    {
        uint32         switchWinding              : 1; ///< Whether to reverse vertex ordering for tessellation.
        uint32         depthClipNearEnable        : 1; ///< Enable clipping based on Near Z coordinate.
        uint32         depthClipFarEnable         : 1; ///< Enable clipping based on Far Z coordinate.
        uint32         alphaToCoverageEnable      : 1; ///< Enable alpha to coverage.
        uint32         perpLineEndCapsEnable      : 1; ///< Forces the use of perpendicular line end caps as opposed to
                                                       ///  axis-aligned line end caps during line rasterization.
        uint32         rasterizerDiscardEnable    : 1; ///< Whether to kill all rasterized pixels.
        uint32         dualSourceBlendEnable      : 1; ///< Enable dual source blend
        uint32         vertexBufferCount          : 6; ///< Number vertex buffer slots accessed by this pipeline
        LogicOp        logicOp                    : 4; ///< Logic operation to perform.
        DepthRange     depthRange                 : 1; ///< Specifies Z dimensions of screen space (i.e., post viewport
                                                       ///  transform: 0 to 1 or -1 to 1).
        DepthClampMode depthClampMode             : 2; ///< Depth clamping behavior.
        uint32         reserved1                  : 7; ///< Reserved
        uint32         reserved                   : 5; ///< Reserved for future use.
    };

    union
    {
        struct
        {
            uint32 depthClampMode          :  1;  ///< Whether to enable dynamic state depthClampMode.
            uint32 depthRange              :  1;  ///< Whether to enable dynamic state depthRange.
            uint32 logicOp                 :  1;  ///< Whether to enable dynamic state logicOp.
            uint32 colorWriteMask          :  1;  ///< Whether to enable dynamic state colorWriteMask.
            uint32 switchWinding           :  1;  ///< Whether to enable dynamic state switchWinding.
            uint32 depthClipMode           :  1;  ///< Whether to enable dynamic state depthClipNear/FarEnable.
            uint32 alphaToCoverageEnable   :  1;  ///< Whether to enable dynamic state alphaToCoverageEnable.
            uint32 perpLineEndCapsEnable   :  1;  ///< Whether to enable dynamic state perpLineEndCapsEnable.
            uint32 rasterizerDiscardEnable :  1;  ///< Whether to enable dynamic state rasterizerDiscardEnable.
            uint32 dualSourceBlendEnable   :  1;  ///< Whether to enable dynamic state dualSourceBlendEnable
            uint32 vertexBufferCount       :  1;  ///< Whether to enable dynamic state vertexBufferCount.
            uint32 reserved1               :  1;  ///< Reserved.
            uint32 reserved                : 20;  ///< Reserved for future use.
        };
        uint32     u32All;
    } enable;
};

/// Specifies info on how graphics shaders should use resources.
struct DynamicGraphicsShaderInfos
{
    union
    {
        // VS/HS/DS/GS or TS/MS are active
        struct
        {
            DynamicGraphicsShaderInfo vs;  ///< Dynamic Vertex shader information.
            DynamicGraphicsShaderInfo hs;  ///< Dynamic Hull shader information.
            DynamicGraphicsShaderInfo ds;  ///< Dynamic Domain shader information.
            DynamicGraphicsShaderInfo gs;  ///< Dynamic Geometry shader information.
        };
        struct
        {
            DynamicGraphicsShaderInfo ts;  ///< Dynamic Task shader information.
            DynamicGraphicsShaderInfo ms;  ///< Dynamic Mesh shader information.
        };
    };

    DynamicGraphicsShaderInfo ps;  ///< Dynamic Pixel shader information.

    union
    {
        struct
        {
            uint8 vs       : 1; // If set, there is dynamic VS shader info.
            uint8 hs       : 1; // If set, there is dynamic HS shader info.
            uint8 ds       : 1; // If set, there is dynamic DS shader info.
            uint8 gs       : 1; // If set, there is dynamic GS shader info.
            uint8 ps       : 1; // If set, there is dynamic PS shader info.
            uint8 ts       : 1; // If set, there is dynamic TS shader info.
            uint8 ms       : 1; // If set, there is dynamic MS shader info.
            uint8 reserved : 1; // Reserved.
        };
        uint8 u8All;
    } enable;
};

/// Specifies parameters for binding a pipeline.
/// @see ICmdBuffer::CmdBindPipeline
struct PipelineBindParams
{
    PipelineBindPoint pipelineBindPoint; ///< Specifies which type of pipeline is to be bound (compute or graphics).
    const IPipeline*  pPipeline;         ///< New pipeline to be bound.  Can be null in order to unbind a previously
                                         ///  bound pipeline without binding a new one.
    uint64 apiPsoHash;                   ///< 64-bit identifier provided by client driver based on the Pipeline State
                                         ///  Object. There exists a many-to-one correlation for ApiPsoHash to
                                         ///  internalPipelineHash to map the two.
    union
    {
        DynamicComputeShaderInfo   cs;   ///< Dynamic Compute shader information.

        struct
        {
            DynamicGraphicsShaderInfos gfxShaderInfo;
            DynamicGraphicsState       gfxDynState;
        };
    };
};

/// Specifies per-MRT color target view and current image state.  Used as input to ICmdBuffer::CmdBindTargets().
struct ColorTargetBindInfo
{
    const IColorTargetView* pColorTargetView;  ///< Color target view to bind.
    ImageLayout             imageLayout;       ///< Specifies the current image layout based on bitmasks of currently
                                               ///  allowed operations and engines that may perform those operations.
                                               ///  At minimum, the LayoutColorTarget usage flag and
                                               ///  LayoutUniversalEngine engine flag must be set.
};

/// Specifies depth/stencil view and current image state of the depth and stencil planes.  Used as input to
/// ICmdBuffer::CmdBindTargets().
struct DepthStencilBindInfo
{
    const IDepthStencilView* pDepthStencilView;  ///< Depth/stencil target view to bind.
    ImageLayout              depthLayout;        ///< Specifies the current image layout of the depth plane based on
                                                 ///  bitmasks of currently allowed operations and engines that may
                                                 ///  perform those operations.  At minimum, the
                                                 ///  LayoutDepthStencilTarget usage flag and LayoutUniversalEngine
                                                 ///  engine flag must be set.  Ignored if the specified view does not
                                                 ///  have a depth plane.
    ImageLayout              stencilLayout;      ///< Specifies the current image layout of the stencil plane based on
                                                 ///  bitmasks of currently allowed operations and engines that may
                                                 ///  perform those operations.  At minimum, the
                                                 ///  LayoutDepthStencilTarget usage flag and LayoutUniversalEngine
                                                 ///  engine flag must be set.  Ignored if the specified view does not
                                                 ///  have a stencil plane.
};

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 928
/// Represents a GPU memory or image transition as part of a barrier.
///
/// A single transition will ensure cache coherency of dirty data in the specific set of source caches with the
/// specified set of destination caches. The source and destination designation is relative to the barrier itself
/// and does not indicate whether a particular cache is a read or write cache.
///
/// Typically a transition flushes written data from the source caches into the destination caches and thus the source
/// cache mask typically only contains write caches. However, the client is encouraged to include flags for any prior
/// read-only caches accesses as PAL may be able to optimize its cache operations.
///
/// If the both cache masks are zero the client is indicating that no cache coherency operations are required but PAL
/// may still issue cache operations for internal reasons.
///
/// In addition, the client can change an image's layout usage/engine flags which may result in a metadata blt.
///
/// @note There is no range provided to control the range of addresses that will be flushed/invalidated in GPU caches.
struct BarrierTransition
{

    uint32 srcCacheMask; ///< Bitmask of @ref CacheCoherencyUsageFlags describing previous write operations whose
                         ///  results need to be visible for subsequent operations. Flags for prior read operations
                         ///  may be included as well and may be used for internal optimizations.
    uint32 dstCacheMask; ///< Bitmask of @ref CacheCoherencyUsageFlags describing the operations expected to read
                         ///  and/or write data flushed from the caches indicated by the srcCacheMask.

    struct
    {
        const IImage* pImage;      ///< If non-null, indicates this transition only applies to the specified image.
                                   ///  The remaining members of this structure are ignored if this member is null.
        SubresRange   subresRange; ///< Subset of pImage this transition applies to. If newLayout includes @ref
                                   ///  LayoutUninitializedTarget this range must cover all subresources of pImage
                                   ///  unless the perSubresInit image create flag was specified.
        ImageLayout   oldLayout;   ///< Specifies the current image layout based on bitmasks of allowed operations and
                                   ///  engines up to this point.  These masks imply the previous compression state. No
                                   ///  usage flags should ever be set in oldLayout.usages that correspond to usages
                                   ///  that are not supported by the engine that is performing the transition.  The
                                   ///  queue type performing the transition must be set in oldLayout.engines.
        ImageLayout   newLayout;   ///< Specifies the upcoming image layout based on bitmasks of allowed operations and
                                   ///  engines after this point.  These masks imply the upcoming compression state.
                                   ///  point.  This usage mask implies the upcoming compressions state.  A difference
                                   ///  between oldLayoutUsageMask and newLayoutUsageMask may result in a
                                   ///  decompression.

        /// Specifies a custom sample pattern over a 2x2 pixel quad.  The position for each sample is specified on a
        /// grid where the pixel center is <0,0>, the top left corner of the pixel is <-8,-8>, and <7,7> is the maximum
        /// valid position (not quite to the bottom/right border of the pixel).
        /// Specifies a custom sample pattern over a 2x2 pixel quad. Can be left null for non-MSAA images or when
        /// a valid MsaaQuadSamplePattern is bound prior to the CmdBarrier call.
        const MsaaQuadSamplePattern* pQuadSamplePattern;

    } imageInfo; ///< Image-specific transition information.
};

/// Describes a barrier as inserted by a call to ICmdBuffer::CmdBarrier().
///
/// A barrier can be used to 1) stall GPU execution at a specified point to resolve a data hazard, 2) flush/invalidate
/// GPU caches to ensure data coherency, and/or 3) compress/decompress image resources as necessary when changing how
/// the GPU will use the image.
///
/// This structure directly specifies how #1 is performed.  #2 and #3 are managed by the list of @ref BarrierTransition
/// structures passed in pTransitions.
struct BarrierInfo
{
    /// Determine at what point the GPU should stall until all specified waits and transitions have completed.  If the
    /// specified wait point is unavailable, PAL will wait at the closest available earlier point.
    HwPipePoint        waitPoint;

    uint32             pipePointWaitCount;           ///< Number of entries in pPipePoints.
    const HwPipePoint* pPipePoints;                  ///< The barrier will stall until the hardware pipeline has cleared
                                                     ///  up to each point specified in this array.  One entry in this
                                                     ///  array is typically enough, but CS and GFX operate in parallel
                                                     ///  at certain stages.

    uint32             gpuEventWaitCount;            ///< Number of entries in ppGpuEvents.
    const IGpuEvent**  ppGpuEvents;                  ///< The barrier will stall until each GPU event in this array is
                                                     ///  in the set state.

    uint32             rangeCheckedTargetWaitCount;  ///< Number of entries in ppTargets.
    const IImage**     ppTargets;                    ///< The barrier will stall until all previous rendering with any
                                                     ///  color or depth/stencil image in this list bound as a target
                                                     ///  has completed. If one of the targets is a nullptr it will
                                                     ///  perform a full range sync.

    uint32                   transitionCount;        ///< Number of entries in pTransitions.
    const BarrierTransition* pTransitions;           ///< List of image/memory transitions to process.  See
                                                     ///  @ref BarrierTransition. The same subresource should never
                                                     ///  be specified more than once in the list of transitions.
                                                     ///  PAL assumes that all specified subresources are unique.

    uint32  globalSrcCacheMask; ///< This is a global bitmask of @ref CacheCoherencyUsageFlags which is combined
                                ///  (bitwise logical union) with the @ref srcCacheMask field belonging to every
                                ///  element in @ref pTransitions. If this is zero or if there are no transitions,
                                ///  then no global cache flags are applied during every transition.

    uint32  globalDstCacheMask; ///< This is a global bitmask of @ref CacheCoherencyUsageFlags which is combined
                                ///  (bitwise logical union) with the @ref dstCacheMask field belonging to every
                                ///  element in @ref pTransitions. If this is zero or if there are no transitions,
                                ///  then no global cache flags are applied during every transition.

    uint32 reason; ///< The reason that the barrier was invoked.
};
#endif

/// Specifies execution dependencies, *availability* and/or *visibility* operations on a section of an IGpuMemory
/// object that does not contain valid IImage data. PAL may assume image data is not present and skip certain
/// cache operations.
///
/// PAL specifies these execution dependencies using pairs of synchronization scope bitmasks of
/// @ref PipelineStageFlag values. The barrier's execution dependencies are only applied to state in this barrier.
/// Memory coherency operations or layout transitions in other barriers will ignore this barrier's execution
/// dependencies.
///
/// PAL specifies these operations using pairs of access scope bitmasks of @ref CacheCoherencyUsageFlags values.
/// The source mask (named srcAccessMask or srcGlobalAccessMask) describes which prior write operations should be made
/// available (i.e., written back from local caches to the LLC). The destination mask (named dstAccessMask or
/// dstGlobalAccessMask) describes which upcoming read/write operations that need visibility (i.e., invalidate
/// corresponding local caches above the LLC). These masks may be zero if no cache operations are needed.
///
/// In general, PAL executes the availability and visibility operations in isolation because the CmdRelease functions
/// require that the destination masks be zero and the CmdAcquire functions require that the source masks be zero.
/// In essence, CmdRelease implements the availability operations and CmdAcquire implements the visibility operations.
/// However, CmdReleaseThenAcquire sees both masks and thus can optimize its cache operations.
///
/// To facilitate cache optimizations, the client is encouraged to add flags corresponding to prior read operations
/// in the relevant source mask(s). Unlike the usual write operation flags, these read flags are entirely optional
/// and do not impact correctness; if they are omitted PAL will simply issue the full set of cache operations.
/// If they are provided PAL may detect cases where future read operations use the same caches as the prior read
/// operations and thus can skip the usual visibility operations.
///
/// Note that,
///   1. If the client does provide read operation flags in a source mask they *must* guarantee that the same flags
///      were provided to a prior barrier's destination mask(s). Incorrect behavior may occur otherwise.
///   2. One @ref MemBarrier or @ImgBarrier object can only be applied to a single resource otherwise PAL's internal
///      optimization may be incorrect. Don't OR multiple resource transitions' stage or access mask into one
///      @ref MemBarrier or @ImgBarrier when making PAL barrier call. However, you are allowed to OR multiple resource
///      transitions' stage or access mask into the global transition mask.
///
/// This struct is used by @ref AcquireReleaseInfo.
struct MemBarrier
{
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 914
    union
    {
        struct
        {
            uint32 globallyAvailable :  1; ///< Normally, data made available is in the GPU LLC.  When this bit is
                                           ///  set, available means in memory, available to all clients in the
                                           ///  system.  This is useful for rare cases like mid command buffer
                                           ///  synchronization with the CPU or another external device.
            uint32 reserved          : 31; ///< Reserved for future use.
        };
        uint32 u32All;                     ///< Flags packed as a 32-bit uint.
    } flags;                               ///< Flags controlling the memory barrier.
#endif

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 880
    GpuMemSubAllocInfo memory;             ///< Specifies a portion of an IGpuMemory object this memory barrier affects.
                                           ///  Zero values of memory structure indicate full range barrier operations.
#endif

    uint32             srcStageMask;       ///< Bitmask of PipelineStageFlag values defining the synchronization
                                           ///  scope that must be confirmed complete as part of a release.  Must be
                                           ///  0 when passed in to CmdAcquire or CmdAcquireEvent.
    uint32             dstStageMask;       ///< Bitmask of PipelineStageFlag values defining the synchronization
                                           ///  scope of operations to be performed after the acquire.  Must be
                                           ///  0 when passed in to CmdRelease or CmdReleaseEvent.

    uint32             srcAccessMask;      ///< CacheCoherencyUsageFlags mask which defines the access scope for the
                                           ///  availability operation, as defined in the struct comment header.
                                           ///  This mask must be 0 when passed to CmdAcquire or CmdAcquireEvent.
    uint32             dstAccessMask;      ///< CacheCoherencyUsageFlags mask which defines the access scope for the
                                           ///  visibility operation, as defined in the struct comment header.
                                           ///  This must be 0 when passed to CmdRelease or CmdReleaseEvent.
};

/// Specifies required layout transition, execution dependencies, *availability*, and/or *visibility* operations on a
/// subresource of an IImage object.
///
/// See the header comment on @ref MemBarrier for a full description of the execution dependencies, availability and
/// visibility operations, including what rules the clients must follow when filling out srcAccessMask and
/// dstAccessMask.
///
/// This struct is used by @ref AcquireReleaseInfo.
struct ImgBarrier
{
    const IImage* pImage;        ///< Relevant image resource for this barrier.
    SubresRange   subresRange;   ///< Selects a range of planes/slices/mips the barrier affects.  If newLayout
                                 ///  includes @ref LayoutUninitializedTarget this range must cover all subresources of
                                 ///  pImage unless the perSubresInit image create flag was specified.

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 880
    Box           box;           ///< Restricts the barrier to a sub-section of each subresource.  The Z offset/extent
                                 ///  must be 0 for 1D/2D images, and the Y offset/extent must be 0 for 1D images.  A
                                 ///  box with zero extents will be ignored, and the barrier will affect the entire
                                 ///  subresource range.  This box may be used to restrict ranges of cache flushes or
                                 ///  invalidations, or may restrict what data is decompressed.  However, the
                                 ///  implementation may not be able to optimize particular cases and may expand the
                                 ///  barrier to cover the entire subresource range.  Specifying a subregion with a box
                                 ///  when newLayout includes @ref LayoutUninitializedTarget is not supported.
#endif

    uint32        srcStageMask;  ///< Bitmask of PipelineStageFlag values defining the synchronization
                                 ///  scope that must be confirmed complete as part of a release.  Must be
                                 ///  0 when passed in to CmdAcquire or CmdAcquireEvent.
    uint32        dstStageMask;  ///< Bitmask of PipelineStageFlag values defining the synchronization
                                 ///  scope of operations to be performed after the acquire.  Must be
                                 ///  0 when passed in to CmdRelease or CmdReleaseEvent.

    uint32        srcAccessMask; ///< CacheCoherencyUsageFlags mask which defines the access scope for the
                                 ///  availability operation, as defined in the struct comment header.
                                 ///  This mask must be 0 when passed to CmdAcquire or CmdAcquireEvent.
    uint32        dstAccessMask; ///< CacheCoherencyUsageFlags mask which defines the access scope for the
                                 ///  visibility operation, as defined in the struct comment header.
                                 ///  This must be 0 when passed to CmdRelease or CmdReleaseEvent.

    ImageLayout   oldLayout;     ///< Specifies the current image layout based on bitmasks of allowed operations and
                                 ///  engines up to this point.  These masks imply the previous compression state. No
                                 ///  usage flags should ever be set in oldLayout.usages that correspond to usages
                                 ///  that are not supported by the engine that is performing the transition.  The
                                 ///  engine type performing the transition must be set in oldLayout.engines. Can set
                                 ///  both oldLayout and newLayout to zero value for no layout transition case.
    ImageLayout   newLayout;     ///< Specifies the upcoming image layout based on bitmasks of allowed operations and
                                 ///  engines after this point.  These masks imply the upcoming compression state.
                                 ///  point.  A difference between oldLayoutUsageMask and newLayoutUsageMask may result
                                 ///  in a decompression.  PAL's implementation will ensure the results of any layout
                                 ///  operations are consistent with the requested availability and visibility
                                 ///  operations. Can set both oldLayout and newLayout to zero value for no layout
                                 ///  transition case.

    /// Specifies a custom sample pattern over a 2x2 pixel quad.  The position for each sample is specified on a grid
    /// where the pixel center is <0,0>, the top left corner of the pixel is <-8,-8>, and <7,7> is the maximum valid
    /// position (not quite to the bottom/right border of the pixel).  Specifies a custom sample pattern over a 2x2
    /// pixel quad. Can be left null for non-MSAA images or when a valid IMsaaState is bound prior to the barrier
    /// call.
    const MsaaQuadSamplePattern* pQuadSamplePattern;
};

/// Input structure to CmdRelease(), CmdReleaseEvent(), CmdAcquire(), CmdAcquireEvent(), and CmdReleastThenAcquire().
/// It describes the execution dependencies, memory dependencies, and image layout transitions that must be resolved.
///
/// Global transition doesn't have buffer or image info so it will assume the worst case and the barrier operations may
/// not be optimal (e.g. metadata may be misaligned and need issue LLC flush/invalidation). It's suggested that if
/// clients know the buffer or image info, try setting up the barrier call with the full buffer or image transition
/// info (including stageMask and accessMask) instead of global transition for optimal performance.
///
/// Clients may OR multiple MemBarrier into a single MemBarrier on full range barrier cases for simple and saving CPU
/// overhead. To allow more optimization chances (e.g. skip unnecessary stalls for read only transitions) in PAL,
/// it's suggested to split the single grouped MemBarrier into two separate grouped MemBarriers: one is read only
/// MemBarrier and the other is writeable MemBarrier; both are then passed together to the barrier call.
struct AcquireReleaseInfo
{
    uint32               srcGlobalStageMask;  ///< Bitmask of PipelineStageFlag values defining the global
                                              ///  synchronization scope that must be confirmed complete as part of a
                                              ///  release.  Must be 0 when passed in to CmdAcquire or CmdAcquireEvent.
    uint32               dstGlobalStageMask;  ///< Bitmask of PipelineStageFlag values defining the global
                                              ///  synchronization scope of operations to be performed after the
                                              ///  acquire.  Must be 0 when passed in to CmdRelease or CmdReleaseEvent.

    uint32               srcGlobalAccessMask; ///< *Access scope* for the global availability operation.  Serves the
                                              ///  same purpose as srcAccessMask in @ref MemoryBarrier, but will cause
                                              ///  all relevant caches to be flushed without range checking.
                                              ///  This mask must be 0 when passed to CmdAcquire or CmdAcquireEvent.
    uint32               dstGlobalAccessMask; ///< *Access scope* for the global visibility operation.  Serves the
                                              ///  same purpose as dstAccessMask in @ref MemoryBarrier, but will cause
                                              ///  all relevant caches to be invalidated without range checking.
                                              ///  This must be 0 when passed to CmdRelease or CmdReleaseEvent.

    uint32               memoryBarrierCount;  ///< Number of entries in pMemoryBarriers.
    const MemBarrier*    pMemoryBarriers;     ///< Describes memory dependencies specific to a range of a particular
                                              ///  IGpuMemory object.

    uint32               imageBarrierCount;   ///< Number of entries in pImageBarriers.
    const ImgBarrier*    pImageBarriers;      ///  Describes memory dependencies and image layout transitions required
                                              ///  for a subresource range of a particular IImage object.
    uint32 reason;                            ///< The reason that the barrier was invoked.
                                              ///  See @ref Developer::BarrierReason for internal reason codes, though
                                              ///  clients may define their own as well
};

/// Specifies barrier type, global (potentially mixed cases of buffer and image), buffer or image.
enum class BarrierType : uint32
{
    Global,
    Buffer,
    Image
};

/// Number of all HW opaque release token types.
constexpr uint32 NumReleaseTokenTypes = 4;

/// Synchronization token structure for CmdRelease() and CmdAcquire().
///
/// Clients should pass the ReleaseToken returned by CmdRelease() to CmdAcquire() directly without changing the value.
/// If a resource with given subresource range has multiple ReleaseToken, all related ReleaseToken should be passed to
/// CmdAcquire().
///
/// Passing ReleaseToken { .fenceValue = N; .type = T } into CmdAcquire() will wait for all prior releases with
/// .fenceValue <= N for .type == T. Resource with a large number of subresources may introduce lots of ReleaseToken
/// potentially (e.g. released per subresource). No need to track all ReleaseToken for each resource since clients
/// can optimize this based on the fact that release type and fenceValue are exposed for each ReleaseToken: define a
/// ReleaseToken array with size @ref NumReleaseTokenTypes, only track ReleaseToken with the largest fenceValue per
/// each release type; and then passing the tracked array ReleaseToken values to CmdAcquire() is enough.
union ReleaseToken
{
    struct
    {
        uint32 fenceValue : 24; ///< Release fence value per token type.
        uint32 type       : 8;  ///< Release token type (HW opaque). Note that please increase the number of bits if
                                ///  it can't hold all types, see @ref NumReleaseTokenTypes for details.
    };

    uint32 u32All;
};

/// Specifies parameters for a copy from one range of a source GPU memory allocation to a range of the same size in a
/// destination GPU memory allocation.  Used as an input to ICmdBuffer::CmdCopyMemory().
struct MemoryCopyRegion
{
    gpusize srcOffset;  ///< Offset in bytes into the source GPU memory allocation to copy data from.
    gpusize dstOffset;  ///< Offset in bytes into the destination GPU memory allocation to copy data to.
    gpusize copySize;   ///< Amount of data to copy in bytes.
};

/// Specifies parameters for an image copy from one region in a source image subresource to a region of the same size in
/// a destination image subresource.  Used as input to ICmdBuffer::CmdCopyImage().
/// If the region describes a copy between a 2D and a 3D image, extent.depth and numSlices must be equal and may be
/// larger than 1.
struct ImageCopyRegion
{
    SubresId srcSubres;  ///< Selects the source subresource.
    Offset3d srcOffset;  ///< Offset to the start of the chosen region in the source subresource.
    SubresId dstSubres;  ///< Selects the destination subresource.
    Offset3d dstOffset;  ///< Offset to the start of the chosen region in the destination
                         ///  subresource.
    Extent3d extent;     ///< Size of the copy region in pixels.
    uint32   numSlices;  ///< Number of slices the copy will span.
};

/// Specifies parameters for a copy between an image and a GPU memory allocation.  The same structure is used regardless
/// of direction, an input for both ICmdBuffer::CmdCopyImageToMemory() and ICmdBuffer::CmdCopyMemoryToImage().
struct MemoryImageCopyRegion
{
    SubresId imageSubres;         ///< Selects the image subresource.
    Offset3d imageOffset;         ///< Pixel offset to the start of the chosen subresource region.
    Extent3d imageExtent;         ///< Size of the image region in pixels.
    uint32   numSlices;           ///< Number of slices the copy will span.
    gpusize  gpuMemoryOffset;     ///< Offset in bytes to the start of the copy region in the GPU memory allocation.
    gpusize  gpuMemoryRowPitch;   ///< Offset in bytes between the same X position on two consecutive lines.
    gpusize  gpuMemoryDepthPitch; ///< Offset in bytes between the same X,Y position of two consecutive slices.
    SwizzledFormat swizzledFormat;///< If not Undefined, reinterpret both subresources using this format and swizzle.
};

/// Specifies parameters for a copy between a PRT and a GPU memory allocation.  The same structure is used regardless
/// of direction, an input for both ICmdBuffer::CmdCopyTiledImageToMemory() and ICmdBuffer::CmdCopyMemoryToTiledImage().
struct MemoryTiledImageCopyRegion
{
    SubresId imageSubres;         ///< Selects the image subresource; must not be a part of the packed mip tail.
    Offset3d imageOffset;         ///< Tile offset to the start of the chosen subresource region.
    Extent3d imageExtent;         ///< Size of the image region in tiles.
    uint32   numSlices;           ///< Number of slices the copy will span.
    gpusize  gpuMemoryOffset;     ///< Offset in bytes to the start of the copy region in the GPU memory allocation.
    gpusize  gpuMemoryRowPitch;   ///< Offset in bytes between the same X position on two consecutive lines.
    gpusize  gpuMemoryDepthPitch; ///< Offset in bytes between the same X,Y position of two consecutive slices.
};

/// Used by copy operations to temporarily interpret a range of GPU memory as a "typed buffer".  A typed buffer is
/// essentially a linear image with a caller-defined row pitch and depth pitch.  Typed buffer copies do not require
/// the GPU memory objects to be created with the "typedBuffer" flag.
struct TypedBufferInfo
{
    SwizzledFormat swizzledFormat; ///< The pixels in this buffer have this format.
    gpusize        offset;         ///< Offset in bytes to the start of the copy region in the buffer's GPU memory
                                   ///  allocation.
    gpusize        rowPitch;       ///< Offset in bytes between the same X position on two consecutive lines.
    gpusize        depthPitch;     ///< Offset in bytes between the same X,Y position of two consecutive slices.
};

/// Specifies parameters for a copy from one region of a typed buffer to a region of the same size in a destination
/// typed buffer.  Used as an input to ICmdBuffer::CmdCopyTypedBuffer().
struct TypedBufferCopyRegion
{
    TypedBufferInfo srcBuffer; ///< How to interpret the source GPU memory allocation as a typed buffer.
    TypedBufferInfo dstBuffer; ///< How to interpret the destination GPU memory allocation as a typed buffer.
    Extent3d        extent;    ///< Size of the copy region in pixels.
};

/// Specifies parameters for a scaled copy between an image and a typed buffer.  The same structure is used regardless
/// of direction, an input for ICmdBuffer::CmdScaledCopyTypedBufferToImage().
struct TypedBufferImageScaledCopyRegion
{
    SubresId        imageSubres;    ///< Selects the image subresource.
    Offset2d        imageOffset;    ///< Pixel offset to the start of the chosen subresource region.
    Extent2d        imageExtent;    ///< Size of the image region in pixels.
    TypedBufferInfo bufferInfo;     ///< How to interpret the GPU memory allocation as a typed buffer.
    Extent2d        bufferExtent;   ///< Size of the typed buffer region in pixels.
    SwizzledFormat  swizzledFormat; ///< If not Undefined, reinterpret both subresources using this format and swizzle.
};

/// Specifies parameters for a scaled image copy from one region in a source image subresource to a region in the
/// destination image subresource.  Used as an input to ICmdBuffer::CmdScaledCopyImage.
struct ImageScaledCopyRegion
{
    SubresId           srcSubres;      ///< Selects the source subresource.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 887
    uint32             srcSlices;      ///< Number of source image slices to read across.
#endif
    union
    {
        Offset3d       srcOffset;      ///< Offset to the start of the chosen region in the source subresource.
        Offset3dFloat  srcOffsetFloat; ///< Alternative representation in floating point.
    };
    union
    {
        SignedExtent3d srcExtent;      ///< Signed size of the source region in pixels.  A negative size indicates
                                       ///  a copy in the reverse direction.
        Extent3dFloat  srcExtentFloat; ///< Alternative representation in floating point.
    };

    SubresId           dstSubres;      ///< Selects the destination subresource.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 887
    uint32             dstSlices;      ///< Number of destination image slices to write.
#endif
    union
    {
        Offset3d       dstOffset;      ///< Offset to the start of the chosen region in the destination subresource.
        Offset3dFloat  dstOffsetFloat; ///< Alternative representation in floating point.
    };
    union
    {
        SignedExtent3d dstExtent;      ///< Signed size of the destination region in pixels.  A negative size
                                       ///  indicates a copy in the reverse direction.
        Extent3dFloat  dstExtentFloat; ///< Alternative representation in floating point.
    };

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 887
    uint32             numSlices;      ///< Number of slices the copy will span.
#endif
    SwizzledFormat     swizzledFormat; ///< If not Undefined, reinterpret both subresources using this format and swizzle.
                                       ///  The specified format needs to have been included in the "pViewFormats" list
                                       ///  specified at image-creation time, otherwise the result might be incorrect.
};

/// Specifies parameters for a color-space-conversion copy from one region in a source image subresource to a region in
/// a destination image subresource.  Used as an input to ICmdBuffer::CmdColorSpaceConversionCopy.
struct ColorSpaceConversionRegion
{
    Offset2d        srcOffset;      ///< Offset to the start of the chosen region in the source subresource(s).
    SignedExtent2d  srcExtent;      ///< Signed size of the source region in pixels.  A negative size indicates a copy
                                    ///  in the reverse direction.
    Offset2d        dstOffset;      ///< Offset to the start of the chosen region in the destination subresource(s).
    SignedExtent2d  dstExtent;      ///< Signed size of the destination region in pixels.  A negative size indicates a
                                    ///  copy in the reverse direction.
    SubresId        rgbSubres;      ///< Selects the first subresource of the RGB image where the copy will begin.  This
                                    ///  can either be the source or destination of the copy, depending on whether the
                                    ///  copy is performing an RGB->YUV or YUV->RGB conversion.
    uint32          yuvStartSlice;  ///< Array slice of the YUV image where the copy will begin.  All planes of planar
                                    ///  YUV images will be implicitly involved in the copy.  This can either be the
                                    ///  source or destination of the copy, depending on whether the copy is performing
                                    ///  an RGB->YUV or YUV->RGB conversion.
    uint32          sliceCount;     ///< Number of slices the copy will span.
};

/// Specifies the color-space-conversion table used when converting between YUV and RGB Image formats.  Used as an input
/// to ICmdBuffer:CmdColorSpaceConversionCopy.
struct ColorSpaceConversionTable
{
    float table[3][4];  ///< Values forming the conversion table matrix, which has three rows and four columns. For RGB
                        ///  to YUV conversions, the conversion shader uses the following expressions to evaluate the
                        ///  YUV color:
                        ///   Y = dot( [R G B 1], [row #0] )
                        ///   U = dot( [R G B 1], [row #1] )
                        ///   V = dot( [R G B 1], [row #2] )
                        ///  For YUV to RGB conversions, the conversion shader uses the following expressions to
                        ///  evaluate the RGB color:
                        ///   R = dot( [Y U V 1], [row #0] )
                        ///   G = dot( [Y U V 1], [row #1] )
                        ///   B = dot( [Y U V 1], [row #2] )
                        ///  A fourth row is not needed because alpha is copied directly between the RGB and YUV colors.
};

/// Default color-space-conversion table usable by PAL clients when calling ICmdBuffer::CmdColorSpaceConverionCopy
/// to perform a YUV to RGB color space conversion.  Represents the BT.601 standard (standard-definition TV).
extern const ColorSpaceConversionTable DefaultCscTableYuvToRgb;

/// Default color-space-conversion table usable by PAL clients when calling ICmdBuffer::CmdColorSpaceConverionCopy
/// to perform a RGB to YUV color space conversion.  Represents the BT.601 standard (standard-definition TV).
extern const ColorSpaceConversionTable DefaultCscTableRgbToYuv;

/// Specifies flags controlling GPU copy behavior.  Format related flags are ignored by DMA queues.
enum CopyControlFlags : uint32
{
    CopyFormatConversion  = 0x1, ///< Requests that the copy convert between two compatible formats. This is ignored
                                 ///  unless both formats support @ref FormatFeatureFormatConversion.
    CopyRawSwizzle        = 0x2, ///< If possible, raw copies will swizzle from the source channel format into the
                                 ///  destination channel format (e.g., RGBA to BGRA).
    CopyEnableScissorTest = 0x4, ///< If set, do scissor test using the specified scissor rectangle.
    CopyControlAllFlags   = 0x7  ///< Clients should NOT use it, for internal static_assert purpose only.
};

/// Specifies parameters for a resolve of one region in an MSAA source image to a region of the same size in a single
/// sample destination image.  Used as an input to ICmdBuffer::CmdResolveImage().
struct ImageResolveRegion
{
    uint32         srcPlane;        ///< The source color, depth, or stencil plane.
    uint32         srcSlice;        ///< Selects the source starting slice
    Offset3d       srcOffset;       ///< Offset to the start of the chosen region in the source subresource.
    uint32         dstPlane;        ///< The destination color, depth, or stencil plane.
    uint32         dstMipLevel;     ///< Selects destination mip level.
    uint32         dstSlice;        ///< Selects the destination starting slice
    Offset3d       dstOffset;       ///< Offset to the start of the chosen region in the destination subresource.
    Extent3d       extent;          ///< Size of the resolve region in pixels.
    uint32         numSlices;       ///< Number of slices to be resolved
    SwizzledFormat swizzledFormat;  ///< If not Undefined, reinterpret both subresources using this format and swizzle.
                                    ///  The format must match both subresource's native formats.

    const MsaaQuadSamplePattern* pQuadSamplePattern; ///< Specifies sample pattern for MSAA depth image. It must be a
                                                     ///  valid pointer if image was created with sampleLocsAlwaysKnown
                                                     ///  flag set.
};

/// A list of the types of PRT+ resolves that can be performed.
enum class PrtPlusResolveType : uint32
{
    Decode = 0x0, ///< Translate from AMD HW format to format of destination image.
    Encode = 0x1, ///< Translate from source image to AMD HW format
    Count  = 0x2,
};

/// Input structure to the CmdResolvePrtPlusImage function
struct PrtPlusImageResolveRegion
{
    Offset3d  srcOffset;       ///< Offset to the start of the chosen region in the source subresource.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 938
    SubresId  srcSubresId;       ///< Selects the source subresource
#else
    uint32    srcMipLevel;     ///< Selects source mip level
    uint32    srcSlice;        ///< Selects the source starting slice
#endif

    Offset3d  dstOffset;       ///< Offset to the start of the chosen region in the destination subresource.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 938
    SubresId  dstSubresId;       ///< Selects the destination subresource
#else
    uint32    dstMipLevel;     ///< Selects destination mip level
    uint32    dstSlice;        ///< Selects the destination starting slice
#endif

    Extent3d  extent;          ///< Size of the resolve region in pixels.
    uint32    numSlices;       ///< Number of slices to be resolved
};

/// Input structure to ICmdBuffer::CmdResolvePrtPlusImageToBuffer()
struct PrtPlusImageToBufferResolveRegion
{
    SubresId  srcSubresId; ///< Selects the source subresource

    gpusize   dstOffset;   ///< Offset into destination subresource

    Extent3d  extent;      ///< Size of the resolve region in pixels.
    uint32    numSlices;   ///< Number of slices to be resolved
};

/// Input structure to ICmdBuffer::CmdResolvePrtPlusBufferToImage()
struct PrtPlusBufferToImageResolveRegion
{
    gpusize   srcOffset;   ///< Offset into source subresource

    SubresId  dstSubresId; ///< Selects the destination subresource

    Extent3d  extent;      ///< Size of the resolve region in pixels.
    uint32    numSlices;   ///< Number of slices to be resolved
};

/// Specifies parameters for a resolve of one region in an MSAA source image to a region of the same size in a single
/// sample destination image.  Used as an input to ICmdBuffer::CmdResolveImage().
enum class ResolveMode : uint32
{
    Average     = 0x0,   ///< Resolve result is an average of all the individual samples
    Minimum     = 0x1,   ///< Resolve result is the minimum value of all individual samples
    Maximum     = 0x2,   ///< Resolve result is the maximum value of all individual samples
    Count       = 0x4,
};

/// Specifies width of immediate data to be written out.
enum class ImmediateDataWidth : uint32
{
    ImmediateData32Bit = 0x0,
    ImmediateData64Bit = 0x1,

    Count              = 0x2,
};

/// Specifies flags controlling GPU query behavior.
union QueryControlFlags
{
    struct
    {
        /// Controls accuracy of query data collection. Available only for occlusion queries.  If set, occlusion query
        /// is guaranteed to return imprecise non-zero value if any samples pass the depth and stencil test.  Using
        /// imprecise occlusion query results could improve rendering performance while an occlusion query is active.
        uint32 impreciseData :  1;
        uint32 reserved      : 31;  ///< Reserved for future use.
    };
    uint32 u32All;                  ///< Flags packed as 32-bit uint.
};

/// Specifies layout of GPU memory used as an input to CmdDrawIndirectMulti.
struct DrawIndirectArgs
{
    uint32 vertexCount;    ///< Number of vertices to draw.
    uint32 instanceCount;  ///< Number of instances to draw.
    uint32 firstVertex;    ///< Starting index value for the draw.  Indices passed to the vertex shader will range from
                           ///  firstVertex to firstVertex + vertexCount - 1.
    uint32 firstInstance;  ///< Starting instance for the draw.  Instace IDs passed to the vertex shader will range from
                           ///  firstInstance to firstInstance + instanceCount - 1.
};

/// Specifies layout of GPU memory used as an input to CmdDrawIndexedIndirectMulti.
///
/// Indices passed to the vertex shader will be:
///
/// + IndexBuffer[firstIndex] + vertexOffset
/// + IndexBuffer[firstIndex + 1] + vertexOffset,
/// + ...
/// + IndexBuffer[firstIndex + indexCount - 1] + vertexOffset
struct DrawIndexedIndirectArgs
{
    uint32 indexCount;     ///< Number of vertices to draw.
    uint32 instanceCount;  ///< Number of instances to draw.
    uint32 firstIndex;     ///< Starting index buffer slot for the draw.
    int32  vertexOffset;   ///< Offset added to the index fetched from the index buffer before it is passed to the
                           ///  vertex shader.
    uint32 firstInstance;  ///< Starting instance for the draw.  Instace IDs passed to the vertex shader will range from
                           ///  firstInstance to firstInstance + instanceCount - 1.
};

/// Specifies layout of GPU memory used as an input to CmdDispatchIndirect.
struct DispatchIndirectArgs
{
    uint32 x;  ///< Threadgroups to dispatch in the X dimension.
    uint32 y;  ///< Threadgroups to dispatch in the Y dimension.
    uint32 z;  ///< Threadgroups to dispatch in the Z dimension.
};

/// Specifies layout of GPU memory used as an input to CmdDispatchMeshIndirect.
using DispatchMeshIndirectArgs = DispatchIndirectArgs;

/// Specifies the GPU virtual address of an array and the stride in bytes between array elements.
struct CpuVirtAddrAndStride
{
    const void*  pCpuVirtAddr;  ///< CPU virtual address of the 0th array element.
    struct
    {
        uint64  stride : 32;    ///< Distance between array elements in bytes.
        uint64  _pad   : 32;    ///< Padding for structure alignment.
    };
};

/// Specifies the GPU virtual address of an array and the stride in bytes between array elements.
struct GpuVirtAddrAndStride
{
    gpusize  gpuVirtAddr;       ///< GPU virtual address of the 0th array element.
    struct
    {
        uint64  stride : 32;    ///< Distance between array elements in bytes.
        uint64  _pad   : 32;    ///< Padding for structure alignment.
    };
};

/// Flags to describe a dispatch
union DispatchInfoFlags
{
    struct
    {
        uint32 devDriverOverlay      : 1;   ///< Flag indicates this dispatch draws the DevDriver overlay
        uint32 reserved              : 31;  ///< Reserved for future use.
    };
    uint32 u32All;                  ///< Flags packed as 32-bit uint.
};

/// Specifies the different stages at which a combiner can choose between different shading rates.
enum class VrsCombinerStage : uint32
{
    ProvokingVertex, ///< Chooses between the shading rate specified by the VrsRateParams struct and the shader
                     ///  rate provided by the provoking vertex.
    Primitive,       ///< Chooses between previous combiner stage and the shader rate associated with the primitive
    Image,           ///< Chooses between previous combiner stage and the shader rate associated with an image
    PsIterSamples,   ///< Chooses between previous combiner stage and the PS_ITER_SAMPLES rate.
    Max
};

/// Specifies the different possible shading rates.  Not all are supported on all HW; see the supportedVrsRates
/// entry in the gfxipProperties structure.
enum class VrsShadingRate : uint32
{
    _16xSsaa  = 0x0,
    _8xSsaa   = 0x1,
    _4xSsaa   = 0x2,
    _2xSsaa   = 0x3,
    _1x1      = 0x4,
    _1x2      = 0x5,
    _2x1      = 0x6,
    _2x2      = 0x7,
    Count
};

/// Indices into the centerOffset array member of the VrsCenterState structure.
enum class VrsCenterRates : uint32
{
    _1x1 = 0x0,
    _1x2 = 0x1,
    _2x1 = 0x2,
    _2x2 = 0x3,
    Max  = 0x4,
};

/// Specifies the different ways in which a combiner can choose between two different shading rate inputs.
enum class VrsCombiner : uint32
{
    Passthrough = 0, ///< Keep previous shading rate.
    Override    = 1, ///< C.xy = B.xy
    Min         = 2, ///< min(A.xy, B.xy)
    Max         = 3, ///< max(A.xy, B.xy)
    Sum         = 4, ///< min(maxRate, A.xy + B.xy)
    Count
};

/// Structure for defining paramters to the CmdSetPerDrawVrsRate function.
struct VrsRateParams
{
    /// The shading rate to be bound to the render state.
    VrsShadingRate  shadingRate;

    /// The state of all the combiners.
    VrsCombiner     combinerState[static_cast<uint32>(VrsCombinerStage::Max)];

    union
    {
        struct
        {
            uint32  exposeVrsPixelsMask :  1; ///< Controls how the shader input mask of a coarse pixel is generated.
                                              ///   0 : Bitwise OR of all fine pixel`s mask
                                              ///   1 : Pack fine pixels` coverage mask into iMask.  Layout based
                                              ///       on VRS rate
            uint32  reserved            : 31;
        };

        uint32  u32All;   ///< Flags packed as 32-bit uint.
    } flags;              ///< Flags controlling VRS rate parameters
};

/// Structure for defininig paramters to the CmdSetVrsCenterState function.
struct VrsCenterState
{
    /// The offset is scaled by the coarse pixel size and then added to the center location
    /// Center offsets are specified as two 4 bits signed integer value representing a location on a 16x16 grid gd.
    /// The offset is scaled by the coarse pixel size and then added to the center location
    /// 1x1, 1x2, 2x1 and 2x2 shading rates can all have their own unique offsets
    Offset2d  centerOffset[static_cast<uint32>(VrsCenterRates::Max)];

    union
    {
        struct
        {
            uint32  overrideCenterSsaa    :  1; ///< Override center interpolants to be evaluated at the sample
                                                ///  position.
            uint32  overrideCentroidSsaa  :  1; ///< Override centroid interpolants to be evaluated at the centroid
                                                ///  of each sample group being iterated (simply the sample position
                                                ///  in the typical case of 1-sample groups).
            uint32  alwaysComputeCentroid :  1; ///< Don't assume the centroid of a fully covered shading region is
                                                ///  the center.  It is possible all samples could be lit but the
                                                ///  center is not lit for certain combinations of centerOffset[]
                                                ///  values and programmable sample positions
            uint32  reserved              : 29; ///< Reserved for future HW
        };

        uint32  u32All;   ///< Flags packed as 32-bit uint.
    } flags;              ///< Flags controlling VRS center state
};

/// @internal
/// Function pointer type definition for setting pipeline-accessible user data entries to the specified values. Each
/// command buffer object has one such callback per pipeline bind point, so the bind point is implicit.
///
/// @see ICmdBuffer::CmdSetUserData().
typedef void (PAL_STDCALL *CmdSetUserDataFunc)(
    ICmdBuffer*   pCmdBuffer,
    uint32        firstEntry,
    uint32        entryCount,
    const uint32* pEntryValues);

/// @internal Function pointer type definition for issuing non-indexed draws.
///
/// @see ICmdBuffer::CmdDraw().
typedef void (PAL_STDCALL *CmdDrawFunc)(
    ICmdBuffer* pCmdBuffer,
    uint32      firstVertex,
    uint32      vertexCount,
    uint32      firstInstance,
    uint32      instanceCount,
    uint32      drawId);

/// @internal Function pointer type definition for issuing draws auto.
///
/// @see ICmdBuffer::CmdDrawOpaque().
typedef void (PAL_STDCALL *CmdDrawOpaqueFunc)(
    ICmdBuffer* pCmdBuffer,
    gpusize     streamOutFilledSizeVa,
    uint32      streamOutOffset,
    uint32      stride,
    uint32      firstInstance,
    uint32      instanceCount);

/// @internal Function pointer type definition for issuing indexed draws.
///
/// @see ICmdBuffer::CmdDrawIndexed().
typedef void (PAL_STDCALL *CmdDrawIndexedFunc)(
    ICmdBuffer* pCmdBuffer,
    uint32      firstIndex,
    uint32      indexCount,
    int32       vertexOffset,
    uint32      firstInstance,
    uint32      instanceCount,
    uint32      drawId);

/// @internal Function pointer type definition for issuing indirect draws.
///
/// @see ICmdBuffer::CmdDrawIndirectMulti().
typedef void (PAL_STDCALL *CmdDrawIndirectMultiFunc)(
    ICmdBuffer*           pCmdBuffer,
    GpuVirtAddrAndStride  gpuVirtAddrAndStride,
    uint32                maximumCount,
    gpusize               countGpuAddr);

/// @internal Function pointer type definition for issuing indexed, indirect draws.
///
/// @see ICmdBuffer::CmdDrawIndexedIndirectMulti().
typedef void (PAL_STDCALL *CmdDrawIndexedIndirectMultiFunc)(
    ICmdBuffer*           pCmdBuffer,
    GpuVirtAddrAndStride  gpuVirtAddrAndStride,
    uint32                maximumCount,
    gpusize               countGpuAddr);

/// @internal Function pointer type definition for issuing direct dispatches.
///
/// @see ICmdBuffer::CmdDispatch().
typedef void (PAL_STDCALL *CmdDispatchFunc)(
    ICmdBuffer*       pCmdBuffer,
    DispatchDims      size,
    DispatchInfoFlags infoFlags);

/// @internal Function pointer type definition for issuing indirect dispatches.
///
/// @see ICmdBuffer::CmdDispatchIndirect().
typedef void (PAL_STDCALL *CmdDispatchIndirectFunc)(
    ICmdBuffer* pCmdBuffer,
    gpusize     gpuVirtAddr);
/// @internal Function pointer type definition for issuing direct dispatches with threadgroup offsets.
///
/// @see ICmdBuffer::CmdDispatchOffset().
typedef void (PAL_STDCALL *CmdDispatchOffsetFunc)(
    ICmdBuffer*  pCmdBuffer,
    DispatchDims offset,
    DispatchDims launchSize,
    DispatchDims logicalSize);

/// @internal Function pointer type definition for issuing direct mesh dispatches.
///
/// @see ICmdBuffer::CmdDispatchMesh().
typedef void (PAL_STDCALL *CmdDispatchMeshFunc)(
    ICmdBuffer*  pCmdBuffer,
    DispatchDims size);

/// @internal Function pointer type definition for issuing indirect mesh dispatches.
///
/// @see ICmdBuffer::CmdDispatchMeshIndirectMulti().
typedef void (PAL_STDCALL *CmdDispatchMeshIndirectMultiFunc)(
    ICmdBuffer*          pCmdBuffer,
    GpuVirtAddrAndStride gpuVirtAddrAndStride,
    uint32               maximumCount,
    gpusize              countGpuAddr);

/// This struct provides the parameters of all the supported features for kernel dispatch
struct DispatchAqlParams
{
    const hsa_kernel_dispatch_packet_t*  pAqlPacket;    ///< Pointer to AQL packet contains the essential
                                                        ///  information (size of workgroup, grid, data
                                                        ///  segments, handle of kernel code object, kernel
                                                        ///  arguments) of the kernel to be dispatched.
    gpusize                              scratchAddr;   ///< GPU VM scratch buffer address
    uint32                               scratchSize;   ///< Scratch buffer size
    uint32                               scratchOffset; ///< Scratch buffer offset from the base for generic
                                                        ///  address space
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 920
    const llvm::amdhsa::kernel_descriptor_t* pCpuAqlCode; ///< AMD kernel descriptor on CPU for PM4 emulation
#else
    const amd_kernel_code_t*             pCpuAqlCode;   ///< AMD kernel code object on CPU for PM4 emulation
#endif
    gpusize                              hsaQueueVa;    ///< GPU VM address where amd_queue_t is allocated
    uint32                               wavesPerSh;    ///< Waves Per Shade Array
    bool                                 useAtc;        ///< Indicates whether ATC bit in registers should be set
                                                        ///  ATC bit is used for indicating if the address
                                                        ///  is GPUVM(=0) or SVM(=1). Valid only in gfx6 and older
    uint64                               kernargSegmentSize;  ///< The size of kerarg segment that holds the
                                                              ///  values of the arguments to the kernels
    uint32                               workitemPrivateSegmentSize;  ///< The amount of fixed private address
                                                                      ///  space memory required for a workitem.
    uint32                               aqlPacketIndex; ///< AQL ID in QueueCreateInfo.aqlPacketList for debugger.
                                                         ///  Note: debugger support enabled for Navi3x+

};

/// @internal Function pointer type definition for issuing AQL dispatches.
///
/// @see ICmdBuffer::CmdDispatchAql().
typedef void (PAL_STDCALL *CmdDispatchAqlFunc)(
    ICmdBuffer*                 pCmdBuffer,
    const DispatchAqlParams&    dispatchInfo);

/// Specifies input assembler state for draws.
/// @see ICmdBuffer::CmdSetInputAssemblyState
struct InputAssemblyStateParams
{
    PrimitiveTopology topology;                     ///< Defines how vertices should be interpretted and rendered by
                                                    ///  the graphics pipeline.
    uint8             patchControlPoints;           ///< # of control points per patch. [0-32] valid. Should be set to
                                                    ///  0 by clients if topology is not PrimitiveTopology::Patch.
    bool              primitiveRestartEnable;       ///< Enables the index specified by primitiveRestartIndex to _cut_
                                                    ///  a primitive (i.e., triangle strip) and begin a new primitive
                                                    ///  with the next index.
    bool              primitiveRestartMatchAllBits; ///< Specifies which bits from primitiveRestartIndex to use.
                                                    ///  false - only check relevant bits based on index type
                                                    ///  true  - check all 32 bits irrespective of index type
    uint32            primitiveRestartIndex;        ///< When primitiveRestartEnable is true, this is the index value
                                                    ///  that will restart a primitive.  When using a 16-bit index
                                                    ///  buffer, the upper 16 bits of this value will be ignored.
};

/// Specifies parameters for controlling triangle rasterization.
/// @see ICmdBuffer::CmdSetTriangleRasterState
struct TriangleRasterStateParams
{
    struct
    {
        FillMode        frontFillMode   : 2; ///< Whether front-facing triangles should be rendered solid or wireframe.
        FillMode        backFillMode    : 2; ///< Whether back-facing triangles should be rendered solid or wireframe.
        CullMode        cullMode        : 2; ///< Specifies which, if any, triangles should be culled based on whether
                                             ///  they are front or back facing.
        FaceOrientation frontFace       : 1; ///< Specifies the vertex winding that results in a front-facing triangle.
        ProvokingVertex provokingVertex : 1; ///< Specifies whether the first or last vertex of a primitive is the
    };                                       ///  provoking vertex as it affects flat shading.
    union
    {
        struct
        {
            uint8 frontDepthBiasEnable : 1; ///< Enable depth bias (i.e. polygon offset) for front-facing
                                            ///  triangle-based primitives
            uint8 backDepthBiasEnable  : 1; ///< Enable depth bias (i.e. polygon offset) for back-facing
                                            ///  triangle-based primitives
            uint8 reserved             : 6; ///< Reserved for future use.
        };
        uint8 u8All;                        ///< Flags packed as 8-bit uint.
    } flags;                                ///< Triangle raster state flags.
};

/// Specifies parameters for controlling point and line rasterization.
/// @see ICmdBuffer::CmdSetPointLineRasterState
struct PointLineRasterStateParams
{
    float pointSize;    ///< Width of a point primitive in pixels.
    float lineWidth;    ///< Width of a line primitive in pixels.
    float pointSizeMin; ///< Minimum width of a point primitive in pixels.
    float pointSizeMax; ///< Maximum width of a point primitive in pixels.
};

/// Specifies parameters for controlling line stippling.
/// @see ICmdBuffer::CmdSetLineStippleState
struct LineStippleStateParams
{
    uint16 lineStippleValue; ///< Line stipple bit pattern.
    uint32 lineStippleScale; ///< Line stipple repeat factor.
};

/// Specifies paramters for setting up depth bias. Depth Bias is used to ensure a primitive can properly be displayed
/// (without Z fighting) in front (or behind) of the previously rendered co-planar primitive.  This is useful for decal
/// or shadow rendering.
/// @see ICmdBuffer::CmdSetDepthBiasState
struct DepthBiasParams
{
    float depthBias;            ///< Base depth bias to be added to each fragment's Z value.  In units of the
                                ///  minimum delta representable in the bound depth buffer.
    float depthBiasClamp;       ///< Maximum allowed depth bias result.  Prevents polygons viewed at a sharp value
                                ///  from generating very large biases.
    float slopeScaledDepthBias; ///< Factor multiplied by the depth slope (change in Z coord per x/y pixel) to
                                ///  create more bias for "steep" polygons.  This result is applied to the final
                                ///  Z value in addition to the base depthBias parameter.
};

/// Specifies parameters for setting the value range to be used for depth bounds testing.
/// @see ICmdBuffer::CmdSetDepthBounds
struct DepthBoundsParams
{
    float min; ///< Minimum depth value in passing range (closest).
    float max; ///< Maximum depth value in passing range (farthest).
};

/// Specifies parameters for setting bit-masks applied to stencil buffer reads and writes.
/// @see ICmdBuffer::CmdSetStencilRefMasks
struct StencilRefMaskParams
{

    uint8 frontRef;        ///< Stencil reference value for front-facing polygons.
    uint8 frontReadMask;   ///< Bitmask to restrict stencil buffer reads for front-facing polygons.
    uint8 frontWriteMask;  ///< Bitmask to restrict stencil buffer writes for front-facing polygons.
    uint8 frontOpValue;    ///< Stencil operation value for front-facing polygons.
                           ///  This is the value used as a parameter for a given stencil operation.
                           ///  For example: StencilOp::IncWrap will use this value when incrementing the current
                           ///  stencil contents. Typically, this would be set to one, but on AMD hardware,
                           ///  this register is 8 bits so there is a greater flexibility.

    uint8 backRef;         ///< Stencil reference value for back-facing polygons.
    uint8 backReadMask;    ///< Bitmask to restrict stencil buffer reads for back-facing polygons.
    uint8 backWriteMask;   ///< Bitmask to restrict stencil buffer writes for back-facing polygons.
    uint8 backOpValue;     ///< Stencil operation value for back-facing polygons - See description of frontOpValue
                           ///  for further details.
    union
    {
        uint8 u8All;                         ///< Flags packed as a 8-bit uint.
        struct
        {
            uint8 updateFrontRef        : 1; ///< Updating reference value for front-facing polygons.
            uint8 updateFrontReadMask   : 1; ///< Updating read mask value for front-facing polygons.
            uint8 updateFrontWriteMask  : 1; ///< Updating write mask value for front-facing polygons.
            uint8 updateFrontOpValue    : 1; ///< Updating stencil op value for front-facing polygons.
            uint8 updateBackRef         : 1; ///< Updating reference value for back-facing polygons.
            uint8 updateBackReadMask    : 1; ///< Updating read mask value for back-facing polygons.
            uint8 updateBackWriteMask   : 1; ///< Updating write mask value for back-facing polygons.
            uint8 updateBackOpValue     : 1; ///< Updating stencil op value for back-facing polygons.
        };
    } flags;               ///< Flags to indicate which of the stencil state values are being updated.
};

/// HiS always exposes two pretests.
constexpr uint32 NumHiSPretests = 2;

/// Hierarchical stencil (HiS) allows work to be discarded by the stencil test at tile rate in certain cases.
/// In order to use HiS, the client will define a set of pretests that will be performed whenever a particular stencil
/// buffer is written.  The stencil image will track the results of the pretest for each 8x8 tile, keeping a record of
/// whether any pixel in the tile "may-pass" or "may-fail" the specified pretest.  When stencil testing is enabled,
/// the hardware may be able to discard whole tiles early based on what it can glean from the HiS pretest states.
///
/// Each stencil image has two pretest slots per mip level.  Pretest slots are reset when an initialization barrier
/// targets their mip level on the stencil plane.  The client can then pass this struct to @ref CmdUpdateHiSPretests
/// to bind one or more valid pretests.  It is legal to bind a pretest over a reset slot at any point.
///
/// @warning Except in special cases, it is illegal to bind a pretest on top of an existing pretest.
///
/// It is only legal to bind a new pretest on top of an existing pretest if:
/// 1. All array slices within the given mip have been reset using an initialization barrier.
/// 2. The client guarantees that they will rewrite all stencil values in all array slices within the given mip
///    before the next draw with stencil testing enabled by doing either:
///      a. One or more calls to @ref CmdClearDepthStencil.
///      b. One or more draws with the stencil test disabled and stencil writes enabled.
///
/// Once pretests are selected via @ref CmdUpdateHiSPretests the client should keep track of which tests were enabled
/// on each stencil image and provide them to every call to @ref CmdClearDepthStencil.  This is optional but PAL will
/// not be able to generate HiS optimized clears unless it is given the current pretests.
///
/// @warning The pretests provided to @ref CmdUpdateHiSPretests are applied to all mips of all subresource ranges.
///          If the client varies pretests between mips they must guarantee that the given pretests were bound to all
///          mips in the given subresource ranges.
///
/// This feature works best if the future stencil test behavior is known, either directly told via an API extension
/// or via an app profile in the client layer. For example, if the application 1) clears stencil, 2) does a pass to
/// write stencil, 3) then does a final pass that masks rendering based on the stencil value being > 0, ideally we
/// would choose a pretest of func=Greater, mask=0xFF, and value=0 so that #2 would update the stencil image with
/// per-tile data that lets #3 be accelerated at maximum effeciency.
///
/// In absence of app-specific knowledge, the following algorithm may be a good generic approach:
/// 1. When the stencil image is cleared, set pretest #0 to func=Equal, mask=0xFF, and value set to the clear value.
/// 2. On the first draw with stencil writes enabled, set pretest #1 with the mask set to the app's current stencil
///    mask, and
///      a. If the stencil op is INC or DEC, set func=GreaterEqual and value the same as in #1.
///      b. If the stencil op is REPLACE, set func=Equal and set value to the app's current stencil ref value.
///
/// Note that HiS can only be beneficial for GPU performance so clients that do not want to implement app profiles or
/// generic heuristics should at least hard-code both tests to something simple.
struct HiSPretests
{
    struct
    {
        CompareFunc func;    ///< This function is used to compare the pretest value with the image's stencil value.
                             ///  The expression is evaluated with the pretest value as the left-hand operand and the
                             ///  image's stencil value as the right-hand operand.
        uint8       mask;    ///< This value is ANDed with both stencil values before evaluating the comparison.
        uint8       value;   ///< The pretest value, used as the left-hand operand in the comparison.
        bool        isValid; ///< True if this pretest contains valid information.  Set to false to skip this test.
    } test[NumHiSPretests];  ///< The set of pretest slots.
};

/// Specifies coordinates for setting up single user clip plane.
/// @see ICmdBuffer::CmdSetUserClipPlanes
struct UserClipPlane
{
    float x; ///< Plane coordinate x
    float y; ///< Plane coordinate y
    float z; ///< Plane coordinate z
    float w; ///< Plane coordinate w
};

/// Specifies parameters for setting the constant factor to be used by the blend hardware when programmed with the
/// Blend::ConstantColor, Blend::OneMinusConstantColor, Blend::ConstantAlpha, or Blend::OneMinusConstantAlpha blend
/// coefficients.
/// @see ICmdBuffer::CmdSetBlendConst
struct BlendConstParams
{
    float blendConst[4];  ///< 4-component RGBA float specifying the new blend constant.
};

/// Specifies the parameters for a single viewport
struct Viewport
{
    float       originX;   ///< X coordinate for the viewport's origin.
    float       originY;   ///< Y coordinate for the viewport's origin.
    float       width;     ///< Width of the viewport.
    float       height;    ///< Height of the viewport.
    float       minDepth;  ///< Minimum depth value of the viewport.  Must be in the [0..1] range.
    float       maxDepth;  ///< Maximum depth value of the viewport.  Must be in the [0..1] range.
    PointOrigin origin;    ///< Origin of the viewport relative to NDC. UpperLeft or LowerLeft.
};

/// Specifies the viewport transform parameters for setting a single viewport.
/// @see ICmdBuffer::CmdSetViewport
struct ViewportParams
{
    uint32     count;                   ///< Number of viewports.
    float      horzDiscardRatio;        ///< The ratio between guardband discard rect width and viewport width.
                                        ///  For all guard band ratio settings, values less than 1.0f are illegal.
                                        ///  Value FLT_MAX opens the guardband as wide as the HW supports.
                                        ///  Value 1.0f disables the guardband.
    float      vertDiscardRatio;        ///< The ratio between guardband discard rect height and viewport height.
    float      horzClipRatio;           ///< The ratio between guardband clip rect width and viewport width.
    float      vertClipRatio;           ///< The ratio between guardband clip rect height and viewport height.
    DepthRange depthRange;              ///< Specifies the target range of Z values
    // Define viewports array at the end of the structure as it is common to only access the first N from the CPU.
    Viewport   viewports[MaxViewports]; ///< Array of desciptors for each viewport.
};

/// Specifies the parameters for specifing the scissor rectangle.
struct ScissorRectParams
{
    uint32 count;                   ///< Number of scissor rectangles.
    Rect   scissors[MaxViewports];  ///< Array of scissor regions corresponding to each viewport.
};

/// Specifies parameters for setting the global scissor rectangle.
/// @see ICmdBuffer::CmdSetGlobalScissor
struct GlobalScissorParams
{
    Rect scissorRegion; ///< Rectangle of the global scissor window.
};

/// Specifies parameters for binding the color targets and depth target.
/// @see ICmdBuffer::CmdBindTargets
struct BindTargetParams
{
    uint32                      colorTargetCount;               ///< Number of color targets to bind.
    ColorTargetBindInfo         colorTargets[MaxColorTargets];  ///< Array of color target descriptors.
    DepthStencilBindInfo        depthTarget;                    ///< Describes the depth target bind info.
};

/// Specifies parameters for binding the stream-output targets.
/// @see ICmdBuffer::CmdBindStreamOutTargets
struct BindStreamOutTargetParams
{
    struct
    {
        gpusize  gpuVirtAddr;       ///< GPU virtual address of this stream-output target.  Must be DWORD-aligned.  If
                                    ///  this is zero, 'size' is ignored and the target is considered un-bound.
        gpusize  size;              ///< Size of this stream-output target, in bytes.  Must be DWORD-aligned.
    } target[MaxStreamOutTargets];  ///< Describes the stream-output target for each buffer slot.
};

/// Specifies the different types of predication ops available.
enum class PredicateType : uint32
{
    Zpass     = 1, ///< Enable occlusion predicate
    PrimCount = 2, ///< Enable streamout predicate
    Boolean64 = 3, ///< CP PFP treats memory as a 64bit integer which is either false (0) or true, DX12 style.
    Boolean32 = 4, ///< CP PFP treats memory as a 32bit integer which is either false (0) or true, Vulkan style.
    Count
};

/// Bitfield structure used to specify masks for functions that operate on depth and/or stencil planes of an image.
union DepthStencilSelectFlags
{
    struct
    {
        /// Select Depth.
        uint32 depth : 1;

        /// Select Stencil.
        uint32 stencil : 1;

        /// Reserved for future usage.
        uint32 reserved : 30;
    };

    /// Flags packed as 32-bit uint.
    uint32 u32All;
};

/// Specifies information related to clearing a bound color target.  Input structure to CmdClearBoundColorTargets().
struct BoundColorTarget
{
    uint32         targetIndex;    ///< Render target index where the target image is currently bound.
    SwizzledFormat swizzledFormat; ///< Format and swizzle of the target image.
    uint32         samples;        ///< Sample count for the target.
    uint32         fragments;      ///< Fragment count for the target.
    ClearColor     clearValue;     ///< clear color value.
};

/// Specifies clear region to clear a bound target. Input structure to CmdClearBoundColorTargets() and
/// CmdClearBoundDepthStencilTargets()
struct ClearBoundTargetRegion
{
    Rect    rect;                       ///< The 2D region to clear.
    uint32  startSlice;                 ///< The starting slice to clear.
    uint32  numSlices;                  ///< The number of slices to clear.
};

/// Specifies flags controlling CmdSaveComputeState and CmdRestoreComputeState.  PAL clients must be aware that saving
/// and restoring specific state in a nested command buffer may not be supported.  The rule is simple: if the client
/// requires that the caller leak the given state to the callee, PAL will not support saving and restoring that state.
enum ComputeStateFlags : uint32
{
    ComputeStatePipelineAndUserData = 0x1, ///< Selects the bound compute pipeline, all non-indirect user data, and all
                                           ///  kernel arguments (if applicable). Note that the current user data will
                                           ///  be invalidated on CmdSaveComputeState.
    ComputeStateBorderColorPalette  = 0x2, ///< Selects the bound border color pallete that affects compute pipelines.
    ComputeStateAll                 = 0x3, ///< Selects all state
};

/// Provides dynamic command buffer flags during submission
/// The following flags are used for Frame Pacing when delay time is configured to be caculated by KMD.
/// (Currently DX clients require this).
/// For clients that do not need Frame Pacing with KMD caculated delay time, they can ignore these flags:
///
/// - frameBegin and frameEnd : Client's presenting queue should track its present state,
///   and set frameBegin flag on the first command buffer after present,
///   set frameEnd flag on the the last command buffer before present. (Could be the Present command buffer itself.)
///   We don't need to set them on queues other than the presenting queue.
/// - P2PCmd : Mark a P2P copy command. KMD could use this flag for adjustments for its frame time calculation.
///   For the current frame time algorithm, clients should only set this flag on SW compositing copy command.
///   But KMD may adjust their algorithm, and clients should update the flag depending on KMD needs.
///
/// The following flags are used for Direct Capture.
///
/// - captureBegin and captureEnd : Direct capture info should be filled if any of these is set. And captureEnd flag
///   also notifies KMD that the on-screen primary is safe to release.
struct CmdBufInfo
{
    union
    {
        struct
        {
            uint32 isValid                 : 1;  ///< Indicate if this CmdBufInfo is valid and should be submitted
            uint32 frameBegin              : 1;  ///< First command buffer after Queue creation or Present.
            uint32 dfSpmTraceBegin         : 1;  ///< This command buffer begins a DF SPM trace.
            uint32 dfSpmTraceEnd           : 1;  ///< This command buffer ends a DF SPM trace.
            uint32 frameEnd                : 1;  ///< Last command buffer before Present.
            uint32 p2pCmd                  : 1;  ///< Is P2P copy command. See CmdBufInfo comments for details.
            uint32 captureBegin            : 1;  ///< This command buffer begins a Direct Capture frame capture.
            uint32 captureEnd              : 1;  ///< This command buffer ends a Direct Capture frame capture.
            uint32 rayTracingExecuted      : 1;  ///< This command buffer contains ray tracing work.
            uint32 preflip                 : 1;  ///< This command buffer has pre-flip access to DirectCapture resource
            uint32 postflip                : 1;  ///< This command buffer has post-flip access to DirectCapture resource
            uint32 privateFlip             : 1;  ///< Need to flip to a private primary surface for DirectCapture feature
            uint32 vpBltExecuted           : 1;  ///< This command buffer comtains VP Blt work.
            uint32 disableDccRejected      : 1;  ///< Reject KMD's DisableDcc request to avoid writing to front buffer.
            uint32 noFlip                  : 1;  ///< No flip when DirectCapture access submission completes
            uint32 frameGenIndex           : 4;  ///< Index of the DirectCapture feature generated frames
            uint32 noRenderPresent         : 1;  ///< Last command buffer before present which is no render present or not
            uint32 motionVectorPropChanged : 1;  ///< Indicates whether motion vector properties changed
            uint32 depthPropChanged        : 1;  ///< Indicates whether depth properties changed
            uint32 cameraPropChanged       : 1;  ///< Indicates whether camera matrix properties changed
            uint32 capturePrimary          : 1;  ///< Has Direct Capture primary surface capture
            uint32 captureMotionVector     : 1;  ///< Has Direct Capture motion vector capture
            uint32 captureDepth            : 1;  ///< Has Direct Capture depth capture
            uint32 captureCamera           : 1;  ///< Has Direct Capture camera matrix capture
            uint32 hudLessImagePropChanged : 1;  ///< Indicates whether HUD less image properties changed
            uint32 captureHudLessImage     : 1;  ///< Has Direct Capture HUD less image capture
            uint32 reserved                : 3;  ///< Reserved for future usage.
        };
        uint32 u32All;                           ///< Flags packed as uint32.
    };

    const IGpuMemory*  pPrimaryMemory;      ///< The primary's gpu memory object used for passing its allocation handle
                                            ///  to KMD for pre-flip primary access (PFPA). If frame metadata flags
                                            ///  specifies that primaryHandle should be sent, clients should set this to
                                            ///  current frame pending primary's IGpuMemory object on the creating GPU
                                            ///  for the frameEnd command. Otherwise set this to nullptr.
    const IGpuMemory*  pDirectCapMemory;    ///< The Direct Capture gpu memory object. It should be set if flag
                                            ///  captureBegin or captureEnd is set. Otherwise set this to nullptr.
    const IGpuMemory*  pPrivFlipMemory;     ///< The gpu memory object of the private flip primary surface for the
                                            ///  DirectCapture feature.
    const Util::Event* pEarlyPresentEvent;  ///< The 'early present' event object. This variable can be nullptr.
    uint64             frameIndex;          ///< The frame index of this command buffer. It is only required for the
                                            ///  DirectCapture feature
    uint32             vidPnSourceId;       ///< The display source id for the DirectCapture feature. Clients must set
                                            ///  a valid vidPnSourceId when privateFlip flag is set and pDirectCapMemory
                                            ///  is nullptr.
    uint64             frameId;             ///< Present frame index, incremented at each present
    const IGpuMemory*  pMotionVectorMemory; ///< The motion vector gpu memory object for the DirectCapture feature.
    const IGpuMemory*  pDepthMemory;        ///< The depth gpu memory object for the DirectCapture feature.
    const IGpuMemory*  pCameraMemory;       ///< The camera gpu memory object for the DirectCapture feature.
    const IGpuMemory*  pHudLessImageMemory; ///< The HUD less image gpu memory object for DirectCapture.
};

/// Specifies rotation angle between two images.  Used as input to ICmdBuffer::CmdScaledCopyImage.
enum class ImageRotation : uint32
{
    Ccw0   = 0x0,   ///< Counter clockwise degree 0
    Ccw90  = 0x1,   ///< Counter clockwise degree 90
    Ccw180 = 0x2,   ///< Counter clockwise degree 180
    Ccw270 = 0x3,   ///< Counter clockwise degree 270
    Count
};

/// Describes a color-key value which can control a pixel get copied or ignored during a CmdScaledCopyImage operation.
struct ColorKey
{
    uint32 u32Color[4]; ///< The color value for each channel
};

/// Uniquely identifies the target of the a Present operation (swap chain / destination window / etc.) so that PAL's debug
/// layers can track frames-per-second or other statistics correctly when applications render to multiple displays or
/// windows.  Client drivers which don't care about this can always specify a key value of 0.
using UniquePresentKey = uint64;

/// Convert an OS window handle to a unique present key.
inline UniquePresentKey PresentKeyFromOsWindowHandle(OsWindowHandle handle)
#if defined(_WIN32)
    { return reinterpret_cast<UniquePresentKey>(handle); }
#else
    { return handle.win; }
#endif
/// Convert any pointer to a unique present key.
template <typename T>
constexpr inline UniquePresentKey PresentKeyFromPointer(T* ptr) { return reinterpret_cast<UniquePresentKey>(ptr); }

/// Specifies the input parameters for debug overlay's visual confirm. This struct is not functional.
/// The client is expected to default initialize this struct and then fill out any state that makes
/// sense under its presentation model. PAL will process any valid input and ignore fields that are
/// default initialized.
struct CmdPostProcessDebugOverlayInfo
{
    PresentMode presentMode;           ///< The Presentation Mode of the application.
    WsiPlatform wsiPlatform;           ///< The WsiPlatform that Swap Chain works upon
    UniquePresentKey presentKey;       ///< Identifies the window/swap chain, etc. used to present.
};

/// Specifies the input parameters for ICmdBuffer::CmdPostProcessFrame.
struct CmdPostProcessFrameInfo
{
    union
    {
        struct
        {
            uint32 srcIsTypedBuffer :  1; ///< True if the source is a typed buffer instead of an image.
            uint32 reserved         : 31; ///< Reserved for future usage.
        };
        uint32     u32All;                ///< Flags packed as uint32.
    } flags;

    union
    {
        const IImage*     pSrcImage;       ///< The image to postprocess (prior to presenting).
        const IGpuMemory* pSrcTypedBuffer; ///< The typed buffer to postprocess.
                                           ///  Must have been created as a typed buffer.
    };

    CmdPostProcessDebugOverlayInfo      debugOverlay;
    FullScreenFrameMetadataControlFlags fullScreenFrameMetadataControlFlags;

    Pal::ImageLayout srcImageLayout;
};

/// External flags for ScaledCopyImage.
union ScaledCopyFlags
{
    struct
    {
        uint32 srcColorKey    : 1;  ///< If set, enables source color-keying by using the value in the ColorKey member.
                                    ///  That is, any pixel in the source image that matches the color key should not be
                                    ///  copied to the destination image, and all of the source pixels that do not match
                                    ///  the color key should be copied. Mutually exclusive with dstColorKey.
        uint32 dstColorKey    : 1;  ///< If set, enables destination color-keying by using the value in the ColorKey
                                    ///  member. That is, any pixel in the destination image that matches the color key
                                    ///  should be replaced with the corresponding pixel from the source image, and all of
                                    ///  the destination pixels that do not match the color key should not be replaced.
                                    ///  Mutually exclusive with srcColorKey.
        uint32 srcAlpha       : 1;  ///< If set, use alpha channel in source surface as blend factor.
                                    ///  color = src alpha * src color + (1.0 - src alpha) * dst color.
        uint32 dstAsSrgb      : 1;  ///< If set, a non-srgb destination image will be treated as srgb format.
                                    ///  Cannot be set if @ref dstAsNorm is set.
        uint32 dstAsNorm      : 1;  ///< If set, a srgb destination image will be treated as non-srgb format.
                                    ///  Cannot be set if @ref dstAsSrgb is set.
        uint32 scissorTest    : 1;  ///< If set, do scissor test using the specified scissor rectangle.
        uint32 coordsInFloat  : 1;  ///< If set, copy regions are represented in floating point type.
        uint32 srcAsNorm      : 1;  ///< If set, an srgb source image will be treated as non-srgb format.
                                    ///  Cannot be set if @ref srcAsSrgb is set.
        uint32 srcAsSrgb      : 1;  ///< If set, a non-srgb source image will be treated as srgb format.
                                    ///  Cannot be set if @ref srcAsNorm is set.
        uint32 reserved       : 23; ///< reserved for future usage.
    };
    uint32 u32All;                  ///< Flags packed as uint32.
};

/// Input structure to @ref ICmdBuffer::CmdScaledCopyImage. Specifies parameters needed to execute CmdScaledCopyImage.
struct ScaledCopyInfo
{
    const IImage*                   pSrcImage;      ///< The source image to blt from.
    ImageLayout                     srcImageLayout; ///< The source image layout.
    const IImage*                   pDstImage;      ///< The dest image to blt to.
    ImageLayout                     dstImageLayout; ///< The dest image layout.
    uint32                          regionCount;    ///< Copy region array size.
    const ImageScaledCopyRegion*    pRegions;       ///< Region array to copy.
    TexFilter                       filter;         ///< Controlling how a given texture is sampled.
    ImageRotation                   rotation;       ///< Rotation option between two images.
    const ColorKey*                 pColorKey;      ///< Color key value.
    const Rect*                     pScissorRect;   ///< Scissor test rectangle.
    ScaledCopyFlags                 flags;          ///< Copy flags, identifies the type of blt to peform.
};

/// Input structure to @ref ICmdBuffer::CmdGenerateMipmaps. Specifies parameters needed to execute CmdGenerateMipmaps.
struct GenMipmapsInfo
{
    const IImage*  pImage;         ///< Populate mips in this image by reading from existing higher-level mips.
    ImageLayout    baseMipLayout;  ///< The layout of all slices in the read-only base mip; must include LayoutCopySrc.
    ImageLayout    genMipLayout;   ///< The layout of all slices and mips that will be generated; must include
                                   ///  LayoutCopySrc and LayoutCopyDst.
    SubresRange    range;          ///< Which subresources should be generated from earlier mips. The starting mipLevel
                                   ///  must never be zero because there would be no larger mip to read.
    TexFilter      filter;         ///< Controls texture sampling during mip generation. Linear texture filtering is
                                   ///  only supported for images with non-integer formats.
    SwizzledFormat swizzledFormat; ///< If not Undefined, reinterpret all subresources using this format and swizzle.
                                   ///  The specified format needs to have been included in the "pViewFormats" list
                                   ///  specified at image-creation time, otherwise the result might be incorrect.
};

/// Defines a single memory range to prefetch using CmdPrimeGpuCaches.
struct PrimeGpuCacheRange
{
    gpusize gpuVirtAddr;          ///< Base GPU virtual address to be prefetched.
    gpusize size;                 ///< Number of bytes to prefetch.  Clients should keep range sizes small relative
                                  ///  to the GPU caches (e.g., tccSizeInBytes); the PAL implementation may clamp
                                  ///  prefetched ranges if they are too large for the cache being prefetched.
    uint32  usageMask;            ///< Bitmask of CacheCoherencyUsageFlags defining the usage to prefetch for.
                                  ///  E.g., if the mask includes CoherShader, then PAL will attempt to prefetch
                                  ///  into caches that are on the shader core's data path.  This mask must be a
                                  ///  subset of the dstCacheMask specified in the last barrier operation executed
                                  ///  on this memory range.  Performing the cache prefetch is considered a read
                                  ///  operation of the specified usage, and so must be properly accounted for
                                  ///  in future barrier memory dependencies for this range.
    bool    addrTranslationOnly;  ///< If set, only the address translation caches (i.e., TLB) will be primed;
                                  ///  no data caches will be affected.  If this is set, the prefetch operation
                                  ///  has no bearing on barrier execution or memory dependencies.
};

/// Magic number tag for payloads in command buffer dumps
constexpr uint32 CmdBufferPayloadSignature = 0x1337F77D;

/// Maximum size, in DWORDs, of payload data in command buffer dumps.
constexpr uint32 MaxPayloadSize = 254;

/// Payload types used in special embedded NOP packets.
enum class CmdBufferPayloadType : uint32
{
    Integer             = 0,    ///< Payload consists of a single 32-bit signed integer.
    UnsignedInteger     = 1,    ///< Payload consists of a single 32-bit unsigned integer.
    Integer64           = 2,    ///< Payload consists of a single 64-bit signed integer.
    UnsignedInteger64   = 3,    ///< Payload consists of a single 64-bit unsigned integer.
    Float               = 4,    ///< Payload consists of a single 32-bit floating point number.
    Double              = 5,    ///< Payload consists of a single 64-bit double precision floating point number.
    Pointer             = 6,    ///< Payload consists of a single 64-bit pointer address.
    String              = 7,    ///< Payload consists of a variable length string. Must contain null-terminator.
    Binary              = 8,    ///< Payload consists of DWORD-aligned binary data.
};

/// Structure layout for embedded CmdBuffer payloads. This can be embedded into the command stream with the
/// @ref ICmdBuffer::CmdNop() function.
struct CmdBufferPayload
{
    uint32               signature;     ///< Magic number tag indicating the structure to follow.
    uint32               payloadSize;   ///< Size of the NOP packet (one DWORD) plus the sizeof this structure and the
                                        ///  payload data to follow.
                                        ///  This value is in DWORDs. Payload size is expected to be under
                                        ///  MaxPayloadSize.
    CmdBufferPayloadType type;          ///< The type of payload.
    uint32               payload[1];    ///< Initial DWORD of payload data with the other data to follow.
};

/// Flags controlling which sub-queue(s) of a command buffer should insert an RGP trace marker.  Zeroing out this
/// union is invalid, because RGP markers must be sent to at least one sub-queue.
union RgpMarkerSubQueueFlags
{
    struct
    {
        uint32  includeMainSubQueue    :  1; ///< If set, includes the main sub-queue in the RGP marker.
        uint32  includeGangedSubQueues :  1; ///< If set, includes any ganged sub-queues in the RGP marker.
        uint32  reserved               : 30; ///< Reserved for future use.
    };
    uint32  u32All; ///< Flags packed into a uint32
};

/**
 ***********************************************************************************************************************
 * @interface ICmdBuffer
 * @brief     Contains GPU rendering and other commands recorded by PAL on the client's behalf.
 *
 * A command buffer can be executed by the GPU multiple times and recycled, provided the command buffer is not pending
 * execution on the GPU when it is recycled.
 *
 * Command buffers are fully independent and there is no persistence of GPU state between submitted command buffers.
 * When a new command buffer is recorded, the state is undefined.  All relevant state must be explicitly set by the
 * client before state-dependent operations such as draws and dispatches.
 *
 * @see IDevice::CreateCmdBuffer()
 ***********************************************************************************************************************
 */
class ICmdBuffer : public IDestroyable
{
public:
    /// Resets the command buffer's previous contents and state, then puts it in the _building_ _state_, allowing new
    /// commands to be recorded.
    ///
    /// If this is a root command buffer, the state will be reset to a "clean slate" with nothing bound.  If this is a
    /// nested command buffer, the state is set to an "undefined" state so that all render state can be inherited from
    /// any root command buffer which executes this one.
    ///
    /// @param [in] info Controls how PAL will generate commands for this command buffer.  E.g., specifies whether the
    ///                  command buffer may be submitted more than once, and controls options for optimizing PM4, etc.
    ///
    /// @returns Success if the command buffer was successfully reset and put into the _building_ _state_.  Otherwise,
    ///          one of the following error codes may be returned:
    ///          + ErrorInvalidFlags if invalid flags are set in the flags parameter.
    ///          + ErrorIncompleteCommandBuffer if the command buffer is already in the _building_ _state_.
    virtual Result Begin(
        const CmdBufferBuildInfo& info) = 0;

    /// Completes recording of a command buffer in the _building_ _state_, making it _executable_.
    ///
    /// @returns Success if the command buffer was successfully made _executable_.  Otherwise, one of the following
    ///          errors may be returned:
    ///          + ErrorIncompleteCommandBuffer if the command buffer is not in the _building_ _state_.
    ///          + ErrorBuildingCommandBuffer if some error occurred while building the command buffer, and it could not
    ///            be made _executable_.  If this error is returned, the command buffer can not be submitted.
    virtual Result End() = 0;

    /// Explicitly resets a command buffer, releasing any internal resources associated with it.
    ///
    /// This call must be used to reset command buffers that have previously reported a ErrorIncompleteCommandBuffer
    /// error.
    ///
    /// @note @ref Begin will implicitly cause a command buffer to be reset in addition to putting it in the
    ///       _building_ _state_.  This method just gives a way to release resources between when the client knows
    ///       it is done with the command buffer and when it is ready to reuse this command buffer object for
    ///       recording new commands.
    ///
    /// @param [in] pCmdAllocator If non-null, all future GPU memory allocations will be done using this allocator.
    ///                           Otherwise the command buffer will continue to use its current command allocator.
    ///
    /// @param [in] returnGpuMemory If true then all GPU memory associated with this command buffer will be returned
    ///                             to the allocator upon reset. If false data chunks will be retained and reused.
    ///                             Note: This flag must be true if changing command allocators.
    ///
    /// @warning If returnGpuMemory is false, the client must guarantee that this command buffer is not queued for
    ///          execution, is not currently being executed, and that all other command buffers that have referenced
    ///          this command buffer in a @ref CmdExecuteNestedCmdBuffers call have also been reset.
    ///
    /// @returns Success if the command buffer was successfully reset.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorUnknown if an internal PAL error occurs.
    virtual Result Reset(ICmdAllocator* pCmdAllocator, bool returnGpuMemory) = 0;

    /// Queries how many DWORDs of embedded data the command buffer can allocate in one call to CmdAllocateEmbeddedData.
    ///
    /// This a property of the command buffer and its associated command allocator; it may change if the caller
    /// specifies a different command allocator on Reset().
    ///
    /// @returns How many DWORDs of embedded data the command buffer can allocate at once.
    virtual uint32 GetEmbeddedDataLimit() const = 0;

    /// Queries how many DWORDs of embedded data the command buffer can allocate in one call to
    /// CmdAllocateLargeEmbeddedData.
    ///
    /// @returns Number of DWORDs that can be allocated in one call to CmdAllocateLargeEmbeddedData
    virtual uint32 GetLargeEmbeddedDataLimit() const = 0;

    /// Binds a graphics or compute pipeline to the current command buffer state.
    ///
    /// Graphics pipelines must be compiled for the PAL ABI. Compute pipelines must either be compiled for the PAL ABI
    /// or the HSA ABI, if it's supported. HSA ABI support is indicated by supportHsaAbi in @ref DeviceProperties.
    ///
    /// PAL ABI pipelines and HSA ABI pipelines use different mechanisms to bind inputs and outputs. PAL ABI pipelines
    /// use user data entries set by @ref CmdSetUserData. HSA ABI pipelines use kernel arguments set by @ref
    /// CmdSetKernelArguments. Binding or unbinding a compute pipeline can implicitly modify the user data and kernel
    /// argument state, please read the @ref CmdSetUserData and @ref CmdSetKernelArguments documentation for details.
    ///
    /// @param [in] params Parameters necessary to manage dynamic pipeline shader information.
    virtual void CmdBindPipeline(
        const PipelineBindParams& params) = 0;

    /// Binds the specified MSAA state object to the current command buffer state.
    ///
    /// @param [in] pMsaaState New MSAA state to be bound.  Can be null in order to unbind a previously bound MSAA state
    ///                        object without binding a new one.
    virtual void CmdBindMsaaState(
        const IMsaaState* pMsaaState) = 0;

    /// Saves a copy of all of the current command buffer state that is used by graphics workloads. This feature is
    /// intended to give PAL clients a convenient way to issue their own internal graphics workloads without modifying
    /// the application-facing state.
    ///
    /// PAL cannot save multiple layers of state, each call to CmdSaveGraphicsState must be followed by a call to
    /// CmdRestoreGraphicsState before the next call to CmdSaveGraphicsState. Any barriers, resolves, blits, etc are not
    /// allowed while the state is pushed.
    ///
    /// This function can only be called on command buffers that support graphics workloads. All query counters will be
    /// disabled until CmdRestoreGraphicsState is called.
    virtual void CmdSaveGraphicsState() = 0;

    /// Restores all of the command buffer state that is used by graphics workloads. This feature is intended to
    /// give PAL clients a convenient way to issue their own internal graphics workloads without modifying the
    /// application-facing state.
    ///
    /// A call to this function must be preceded by a call to CmdSaveGraphicsState
    ///
    /// This function can only be called on command buffers that support graphics workloads. All previously disabled
    /// query counters will be reactivated.
    virtual void CmdRestoreGraphicsState() = 0;

    /// Sets the shading rate in the command buffer along with the state of the various combiners.
    ///
    /// @param [in] rateParams     Nwe VRS shading rate parameters to be bound.
    virtual void CmdSetPerDrawVrsRate(
        const VrsRateParams&  rateParams) = 0;

    /// Setup parameters regarding how pixel center will be evaluated with VRS.
    ///
    /// @param [in] centerState     Nwe VRS parameters to be bound that control how pixel center is defined.
    virtual void CmdSetVrsCenterState(
        const VrsCenterState&  centerState) = 0;

    /// Binds the shading rate data in the specified image into the pipeline for use with VRS.  Only relevant if the
    /// combiner stage for VrsCombinerStage is set to something other than Passthrough.
    ///
    /// This binding point requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageSampleRate
    /// - CacheCoherency: @ref CoherSampleRate
    /// - ImageLayout:    @ref LayoutSampleRate
    ///
    /// @param [in] pImage   Image that contains sample rate data.  Pointer can be NULL to force 1x1 shading rate.
    virtual void CmdBindSampleRateImage(
        const IImage*  pImage) = 0;

    /// Binds the specified color/blend state object to the current command buffer state.
    ///
    /// @param [in] pColorBlendState New color/blend state to be bound.  Can be null in order to unbind a previously
    ///                              bound color/blend state object without binding a new one.
    virtual void CmdBindColorBlendState(
        const IColorBlendState* pColorBlendState) = 0;

    /// Binds the specified depth/stencil state object to the current command buffer state.
    ///
    /// @param [in] pDepthStencilState New depth/stencil state to be bound.  Can be null in order to unbind a previously
    ///                                bound depth/stencil state object without binding a new one.
    virtual void CmdBindDepthStencilState(
        const IDepthStencilState* pDepthStencilState) = 0;

    /// Sets the value range to be used for depth bounds testing.
    ///
    /// The depth bounds test is enabled in the graphics pipeline.  When enabled, an additional check will be done that
    /// will reject a pixel if the pre-existing depth value stored at its destination location is outside of the
    /// specified bounds.  Applications would typically use this feature to optimize shadow volume rendering.
    ///
    /// @param [in] params  Parameters necessary to set the depth bounds (such as min/max depth).
    virtual void CmdSetDepthBounds(
        const DepthBoundsParams& params) = 0;

    /// Sets pipeline-accessible user data to the specified values.
    ///
    /// The values set in user data entries will be interpreted based on the resource mapping specified for each shader
    /// in the currently bound pipeline.  For example, the client can write virtual addresses of tables containing
    /// SRDs, immediate SRDs that can be loaded without an indirection, or even a small number of immediate ALU
    /// constants.
    ///
    /// The user data values are only used by PAL ABI pipelines. Almost all pipelines used by PAL clients are compiled
    /// for the PAL ABI, but PAL also supports HSA ABI compute pipelines which use @ref CmdSetKernelArguments instead.
    /// When an HSA ABI pipeline is bound the current compute user data entries are saved and will be restored if the
    /// client later binds a PAL ABI compute pipeline.
    ///
    /// @warning It's illegal to set compute user data if an HSA ABI pipeline is currently bound.
    ///
    /// If no compute pipeline is currently bound PAL assumes the client will bind a PAL ABI pipeline and thus accepts
    /// user data bindings. Graphics user data are unaffected by all of this because graphics pipelines can only use
    /// the PAL ABI.
    ///
    /// @see PipelineShaderInfo
    /// @see ResourceMappingNode
    /// @ingroup ResourceBinding
    ///
    /// @param [in] bindPoint    Specifies which type of user-date is to be set (i.e., compute or graphics).
    /// @param [in] firstEntry   First user data entry to be updated.
    /// @param [in] entryCount   Number of user data entries to update; size of the pEntryValues array. Must be greater
    ///                          than zero, and (firstEntry + entryCount) must not extend beyond MaxUserDataEntries.
    /// @param [in] pEntryValues Array of 32-bit values to be copied into user data.
    void CmdSetUserData(
        PipelineBindPoint bindPoint,
        uint32            firstEntry,
        uint32            entryCount,
        const uint32*     pEntryValues)
    { (m_funcTable.pfnCmdSetUserData[static_cast<uint32>(bindPoint)])(this, firstEntry, entryCount, pEntryValues); }

    /// Copies all pipeline-accessible user-data from one bind point to another.  It is invalid if the source and
    /// dest parameters refer to the same bind point.
    ///
    /// @see CmdSetUserData for how the user-date entries will be interpreted by the pipeline.
    ///
    /// @param [in] source  Specifies which bind point to copy from.
    /// @param [in] dest    Specifies which bind point to copy into.
    virtual void CmdDuplicateUserData(
        PipelineBindPoint source,
        PipelineBindPoint dest) = 0;

    /// Sets one or more HSA code object kernel argument values.
    ///
    /// If the currently bound compute pipeline was compiled using the HSA compute ABI this function must be used to
    /// bind that pipeline's arguments. The argument position and value types are static properties of the pipeline
    /// and must be known by the client.
    ///
    /// @note Calling @ref CmdBindPipeline invalidates all prior kernel argument bindings, even if the new pipeline
    ///       also uses the HSA ABI. Any kernel arguments that the client intends to share between pipelines must
    ///       be manually rebound.
    ///
    /// @warning It's illegal to call this function if no compute pipeline is bound or if the bound compute pipeline
    ///          uses a different ABI (e.g., the PAL compute ABI).
    ///
    /// @ingroup ResourceBinding
    ///
    /// @param [in] firstArg  The zero-based position of the first kernel argument to bind.
    /// @param [in] argCount  Number of kernel arguments this call binds.
    /// @param [in] ppValues  Array of pointers to kernel argument values.
    virtual void CmdSetKernelArguments(
        uint32            firstArg,
        uint32            argCount,
        const void*const* ppValues) = 0;

    /// Changes one or more of the command buffer's active vertex buffers.
    ///
    /// @note  If bufferViews.offsetMode is false, PAL will construct SRDs for each bound vertex buffer which are
    ///        equivalent to the client calling @ref IDevice::CreateUntypedBufferViewSrd on each element of the
    ///        pBuffers parameter.
    ///
    /// Note that vertex buffers require use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageVs
    /// - CacheCoherency: @ref CoherShaderRead
    ///
    /// @param [in] bufferViews Vertex buffer view descriptors.This parameter defines which vertex mode is used through
    ///                         @ref VertexBufferViews::offsetMode. VertexBufferViews::pVertexBufferViews or
    ///                         VertexBufferViews::pBufferViewInfos must not be nullptr.
    virtual void CmdSetVertexBuffers(
        const VertexBufferViews& bufferViews) = 0;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 877
    void CmdSetVertexBuffers(
        uint32                firstBuffer,
        uint32                bufferCount,
        const BufferViewInfo* pBuffers)
    {
        const VertexBufferViews bufferViews =
        {
            .firstBuffer = firstBuffer,
            .bufferCount = bufferCount,
            .offsetMode = false,
            .pBufferViewInfos = pBuffers
        };
        CmdSetVertexBuffers(bufferViews);
    }
#endif

    /// Binds a range of memory for use as index data (i.e., binds an index buffer).
    ///
    /// The GPU virtual address must be index element aligned: 2-byte aligned for 16-bit indices or 4-byte aligned for
    /// 32-bit indices.
    ///
    /// The index buffer binding point requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageFetchIndices
    /// - CacheCoherency: @ref CoherIndexData
    ///
    /// @param [in] gpuAddr    GPU virtual address of the index data.  Can be zero to unbind the previously bound data.
    /// @param [in] indexCount Maximum number of indices in the index data; the GPU may read less indices.
    /// @param [in] indexType  Specifies whether to use 8-bit, 16-bit or 32-bit index data.
    virtual void CmdBindIndexData(
        gpusize   gpuAddr,
        uint32    indexCount,
        IndexType indexType) = 0;

    /// Binds color and depth/stencil targets to the current command buffer state.
    ///
    /// The current layout of each target must also be specified.
    ///
    /// The color target binding points require use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageColorTarget
    /// - CacheCoherency: @ref CoherColorTarget
    /// - ImageLayout:    @ref LayoutColorTarget
    ///
    /// The depth and stencil target binding points require use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageEarlyDsTarget and/or @ref PipelineStageLateDsTarget
    /// - CacheCoherency: @ref CoherDepthStencilTarget
    /// - ImageLayout:    @ref LayoutDepthStencilTarget
    ///
    /// @param [in] params Parameters representing the color and depth/stencil targets to bind to the command buffer.
    virtual void CmdBindTargets(
        const BindTargetParams& params) = 0;

    /// Binds stream-output target buffers to the current command buffer state.
    ///
    /// At draw-time, the stream-output targets must be consistent with the soState parameters specified by the
    /// currently bound graphics pipeline.
    ///
    /// The stream-output target buffers require use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageStreamOut
    /// - CacheCoherency: @ref CoherStreamOut
    ///
    /// @param [in] params Parameters representing the stream-output target buffers to bind to the command buffer.
    virtual void CmdBindStreamOutTargets(
        const BindStreamOutTargetParams& params) = 0;

    /// Sets the constant factor to be used by the blend hardware when programmed with the Blend::ConstantColor,
    /// Blend::OneMinusConstantColor, Blend::ConstantAlpha, or Blend::OneMinusConstantAlpha blend coefficients.
    ///
    /// @param [in] params Parameters representing the blend constant factor.
    virtual void CmdSetBlendConst(
        const BlendConstParams& params) = 0;

    /// Sets input assembly state for upcoming draws in this command buffer.
    ///
    /// At draw-time, the topology specified with this method must be consistent with the _topologyInfo_ parameters
    /// specified by the currently bound graphics pipeline.
    ///
    /// @param [in] params Parameters representing the input assembly state for upcoming draws.
    virtual void CmdSetInputAssemblyState(
        const InputAssemblyStateParams& params) = 0;

    /// Sets parameters controlling triangle rasterization.
    ///
    /// @param [in] params Parameters to set the triangle raster state (such as fill/cull mode).
    virtual void CmdSetTriangleRasterState(
        const TriangleRasterStateParams& params) = 0;

    /// Sets parameters controlling point and line rasterization.
    ///
    /// @param [in] params Parameters to set the point and line rasterization state (such as pointSize and lineWidth).
    virtual void CmdSetPointLineRasterState(
        const PointLineRasterStateParams& params) = 0;

    /// Sets parameters controlling line stippling.
    ///
    /// @param [in] params Parameters to set the line stipple state.
    virtual void CmdSetLineStippleState(
        const LineStippleStateParams& params) = 0;

    /// Sets depth bias parameters.
    ///
    /// Depth bias is used to ensure a primitive can properly be displayed (without Z fighting) in front (or behind)
    /// of the previously rendered co-planar primitive.  This is useful for decal or shadow rendering.
    ///
    /// @param [in] params  Parameters for setting the depth bias (such as depth bias, depth bias clamp, and slope
    ///                     scaled depth bias).
    virtual void CmdSetDepthBiasState(
        const DepthBiasParams& params) = 0;

    /// Sets stencil reference values and mask buffer reads and writes in upcoming draws. Separate reference values
    /// can be specified for front-facing and back-facing polygons. Update flags should be set for state which needs to
    /// be updated. All other state will be preserved.
    /// Setting all the values (reference, read/write masks and stencil op) in the StencilRefMaskParams together
    /// takes the faster path.
    /// Setting either the ref value, read/write masks or the stencil op value individually takes the slower
    /// read-modify-write path.
    ///
    /// @param [in] params  Parameters for setting the stencil read and write masks.
    virtual void CmdSetStencilRefMasks(
        const StencilRefMaskParams& params) = 0;

    /// Sets user defined clip planes, should only be called on universal command buffers.
    ///
    /// @param [in] firstPlane The index of first plane in user define clip plane array.
    /// @param [in] planeCount The count of planes in plane array.
    /// @param [in] pPlanes    Pointer to plane array.
    virtual void CmdSetUserClipPlanes(
        uint32               firstPlane,
        uint32               planeCount,
        const UserClipPlane* pPlanes) = 0;

    /// Sets clip rects, should only be called on universal command buffers.
    ///
    /// @param [in] clipRule  16-bit clip rule bits are used to determine if pixel shall be discarded or retained.
    ///                       For each pixel, a 4-bit index is computed based on which clip rects the pixel is
    ///                       inside (bitN represents rectN). Then uses this index to check the corresponding bit
    ///                       in clip rule for this pixel - 0 for discarded, 1 for retained.
    /// @param [in] rectCount The count of rectangles in rect list. This must be less than or equal to
    ///                       MaxClipRects (4).
    /// @param [in] pRectList Pointer to the rect list.
    virtual void CmdSetClipRects(
        uint16      clipRule,
        uint32      rectCount,
        const Rect* pRectList) = 0;

    /// Sets user defined MSAA quad-pixel sample pattern, should only be called on universal command buffers
    /// This should be called before clearing, rendering, barriering and resolving of MSAA DepthStencil image.
    ///
    /// @param [in] numSamplesPerPixel Number of samples per pixel
    /// @param [in] quadSamplePattern  The input msaa sample pattern
    virtual void CmdSetMsaaQuadSamplePattern(
        uint32                       numSamplesPerPixel,
        const MsaaQuadSamplePattern& quadSamplePattern) = 0;

    /// Sets the specified viewports to the current command buffer state.
    ///
    /// @param [in] params  Parameters for setting the specified number of viewports.
    virtual void CmdSetViewports(
        const ViewportParams& params) = 0;

    /// Sets the scissor regions corresponding to each viewport to the current command buffer state.
    ///
    /// @param [in] params  Parameters for setting the specified number of scissor regions.
    virtual void CmdSetScissorRects(
        const ScissorRectParams& params) = 0;

    /// Sets the global scissor rectangle.
    ///
    /// @param [in] params  Parameters for setting the global scissor rectangle from the top left to bottom right
    ///                     coordinate.
    virtual void CmdSetGlobalScissor(
        const GlobalScissorParams& params) = 0;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 928
    /// Inserts a barrier in the current command stream that can stall GPU execution, flush/invalidate caches, or
    /// decompress images before further, dependent work can continue in this command buffer.
    ///
    /// This operation does not honor the command buffer's predication state, if active.
    ///
    /// @param [in] barrierInfo See @ref BarrierInfo for detailed information.
    virtual void CmdBarrier(
        const BarrierInfo& barrierInfo) = 0;
#endif
    /// Perform source pipeline stage and cache access optimization based on the acquire/release interface.
    ///
    /// @param [in]     barrierType    Barrier transition type @ref BarrierType.
    /// @param [in]     pImage         Image pointer for image transition, required when @ref BarrierType is
    ///                                BarrierType::Image.
    /// @param [in/out] pSrcStageMask  A source mask of ORed @ref PipelineStageFlag to optimize, can't be null.
    /// @param [in/out] pSrcAccessMask A source mask of ORed @ref CacheCoherencyUsageFlags to optimize, can't be null.
    /// @param [in/out] pDstStageMask  A destination mask of ORed @ref PipelineStageFlag to optimize, can't be null.
    /// @param [in/out] pDstAccessMask A destination mask of ORed @ref CacheCoherencyUsageFlags to optimize.
    ///
    /// @returns If need flush and invalidate GL2 cache.
    ///
    /// @note PipelineStageBlt will be converted to more accurate stage(s) based on the underlying implementation of
    ///       outstanding BLTs, but will be left as PipelineStageBlt if the internal outstanding BLTs can't be expressed
    ///       as a client-facing PipelineStage (e.g., if there are CP DMA BLTs in flight).
    virtual bool OptimizeAcqRelReleaseInfo(
        BarrierType   barrierType,
        const IImage* pImage,
        uint32*       pSrcStageMask,
        uint32*       pSrcAccessMask,
        uint32*       pDstStageMask,
        uint32*       pDstAccessMask) const = 0;

    /// Performs the release portion of an acquire/release-based barrier.  This releases a set of resources from their
    /// current usage, while CmdAcquire() is expected to be called to acquire access to the resources for future,
    /// different usage.
    ///
    /// Conceptually, this method will:
    ///   - Ensure the specified source synchronization scope has completed.
    ///   - Ensure all specified resources are available in memory.  The availability operation will flush all
    ///     write-back caches to the last-level-cache.
    ///   - Perform any requested layout transitions.
    ///
    /// Once all of these operations are complete, the release issues a timestamp event that signals the operation
    /// completion.  The event type and timestamp value is returned to caller in a packed uint32 token.  A corresponding
    /// CmdAcquire() call is expected to wait on one or a list of such synchronization tokens and perform any necessary
    /// visibility operations and/or layout transitions that could not be predicted at release-time.
    ///
    /// @param [in]  releaseInfo  Describes the synchronization scope, availability operations, and required layout
    ///                           transitions.
    /// @returns Synchronization token for the release operation.  Pass this token to CmdAcquire to confirm completion.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 885
    virtual uint32 CmdRelease(
#else
    virtual ReleaseToken CmdRelease(
#endif
        const AcquireReleaseInfo& releaseInfo) = 0;

    /// Performs the acquire portion of an acquire/release-based barrier.  This acquire a set of resources for a new
    /// set of usages, assuming CmdRelease() was called to release access for the resource's past usage.
    ///
    /// Conceptually, this method will:
    ///   - Ensure all specified resources are visible in memory.  The visibility operation will invalidate all
    ///     relevant caches above the last-level-cache.
    ///   - Perform any requested layout transitions.
    ///   - Ensure the release(s) have completed by waiting on the synchronization token of the release operation.
    ///
    /// @param [in] acquireInfo    Describes the synchronization scope, visibility operations, and the required layout
    ///                            layout transitions.
    /// @param [in] syncTokenCount Number of entries in pSyncTokens, can be zero if no valid release token.
    /// @param [in] pSyncTokens    Array of synchronization tokens, as returned from CmdRelease, to confirm completion.
    ///                            The token value(s) must have been returned by a CmdRelease call in the same command
    ///                            buffer. pSyncTokens can be null if syncTokenCount is 0.
    virtual void CmdAcquire(
        const AcquireReleaseInfo& acquireInfo,
        uint32                    syncTokenCount,
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 885
        const uint32*             pSyncTokens) = 0;
#else
        const ReleaseToken*       pSyncTokens) = 0;
#endif

    /// Performs the release portion of an acquire/release event-based barrier.  This releases a set of resources from
    /// their current usage, while CmdAcquireEvent() is expected to be called to acquire access to the resources for
    /// future, different usage.
    ///
    /// Conceptually, this method will:
    ///   - Ensure the specified source synchronization scope has completed.
    ///   - Ensure all specified resources are available in memory.  The availability operation will flush all
    ///     write-back caches to the last-level-cache.
    ///   - Perform any requested layout transitions.
    ///
    /// Once all of these operations are complete, the specified IGpuEvent object will be signaled.  A corresponding
    /// CmdAcquireEvent() call is expected to wait on this event and perform any necessary visibility operations and/or
    /// layout transitions that could not be predicted at release-time.
    ///
    /// @param [in] releaseInfo Describes the synchronization scope, availability operations, and required layout
    ///                         transitions.
    /// @param [in] pGpuEvent   Event to be signaled once the release has completed.  Must be a valid (non-null) GPU
    ///                         event pointer. Call CmdRelease()/CmdAcquire() pair instead if want to release/acquire
    ///                         something but no GPU event is available.
    virtual void CmdReleaseEvent(
        const AcquireReleaseInfo& releaseInfo,
        const IGpuEvent*          pGpuEvent) = 0;

    /// Performs the acquire portion of an acquire/release event-based barrier.  This acquire a set of resources for a
    /// new set of usages, assuming CmdReleaseEvent() was called to release access for the resource's past usage.
    ///
    /// Conceptually, this method will:
    ///   - Ensure the release(s) have completed by waiting for the specified IGpuEvent early enough in the pipeline to
    ///     support the specified destination synchronization scope.
    ///   - Ensure all specified resources are visible in memory.  The visibility operation will invalidate all
    ///     relevant caches above the last-level-cache.
    ///   - Perform any requested layout transitions.
    ///
    /// @param [in] acquireInfo    Describes the synchronization scope, visibility operations, and the required layout
    ///                            layout transitions.
    /// @param [in] gpuEventCount  Number of entries in pGpuEvents.
    /// @param [in] ppGpuEvents    Array of one or more events to wait on.  Typically these will be set via
    ///                            CmdReleaseEvent(), but it's valid to wait on an event set through a different means
    ///                            like CmdSetEvent() from CPU side. Must be a valid (non-null) pointer to an array of
    ///                            gpuEventCount valid GPU event pointers. Call CmdReleaseThenAcquire() instead if wait
    ///                            to acquire something but no GPU event is available.
    virtual void CmdAcquireEvent(
        const AcquireReleaseInfo& acquireInfo,
        uint32                    gpuEventCount,
        const IGpuEvent* const*   ppGpuEvents) = 0;

    /// Conceptually equivalent to calling CmdRelease() followed immediately by CmdAcquire(),  but it potentially has
    /// better performance than calling CmdRelease()/CmdAcquire() directly.  Can be called in cases where the client/
    /// application cannot detect separate release and acquire points for a transition.
    ///
    /// @param [in] barrierInfo  Describes the synchronization scopes, availability/visibility operations, and the
    ///                          required layout transitions.
    virtual void CmdReleaseThenAcquire(
        const AcquireReleaseInfo& barrierInfo) = 0;

    /// Issues an instanced, non-indexed draw call using the command buffer's currently bound graphics state.  Results
    /// in instanceCount * vertexCount vertices being processed.
    ///
    /// It is an error if the currently bound pipeline contains a mesh and/or task shader.
    ///
    /// @param [in] firstVertex   Starting index value for the draw.  Indices passed to the vertex shader will range
    ///                           from firstVertex to firstVertex + vertexCount - 1.
    /// @param [in] vertexCount   Number of vertices to draw.  If zero, the draw will be discarded.
    /// @param [in] firstInstance Starting instance for the draw.  Instance IDs passed to the vertex shader will range
    ///                           from firstInstance to firstInstance + instanceCount - 1.
    /// @param [in] instanceCount Number of instances to draw.  If zero, the draw will be discarded.
    /// @param [in] drawId        Draw index for the draw.
    void CmdDraw(
        uint32 firstVertex,
        uint32 vertexCount,
        uint32 firstInstance,
        uint32 instanceCount,
        uint32 drawId)
    {
        m_funcTable.pfnCmdDraw(this, firstVertex, vertexCount, firstInstance, instanceCount, drawId);
    }

    /// Issues draw opaque call using the command buffer's currently bound graphics state.
    /// Uses the stream-out target of a previous draw as the input vertex data.
    /// the number of vertices = (streamOutFilledSize (value of streamOutFilledSizeVa) - streamOutOffset) / stride
    ///
    /// It is an error if the currently bound pipeline contains a mesh and/or task shader.
    ///
    /// @param [in] streamOutFilledSizeVa gpuAddress of streamOut filled size for streamOut buffer.
    /// @param [in] streamOutOffset       the offset of begin of streamOut as vertex.
    /// @param [in] stride                stride for stream data as vertex.
    /// @param [in] firstInstance         Starting instance for the draw. Instance IDs passed to the vertex shader
    ///                                   will range from firstInstance to firstInstance + instanceCount - 1.
    /// @param [in] instanceCount         Number of instances to draw.  If zero, the draw will be discarded.
    void CmdDrawOpaque(
        gpusize streamOutFilledSizeVa,
        uint32  streamOutOffset,
        uint32  stride,
        uint32  firstInstance,
        uint32  instanceCount)
    {
        m_funcTable.pfnCmdDrawOpaque(this,
                                     streamOutFilledSizeVa,
                                     streamOutOffset,
                                     stride,
                                     firstInstance,
                                     instanceCount);
    }

    /// Issues an instanced, indexed draw call using the command buffer's currently bound graphics state.  Results in
    /// instanceCount * indexCount vertices being processed.
    ///
    /// It is an error if the currently bound pipeline contains a mesh and/or task shader.
    ///
    /// Indices passed to the vertex shader will be:
    ///
    /// + IndexBuffer[firstIndex] + vertexOffset
    /// + IndexBuffer[firstIndex + 1] + vertexOffset,
    /// + ...
    /// + IndexBuffer[firstIndex + indexCount - 1] + vertexOffset
    ///
    /// @param [in] firstIndex    Starting index buffer slot for the draw.
    /// @param [in] indexCount    Number of vertices to draw.  If zero, the draw will be discarded.
    /// @param [in] vertexOffset  Offset added to the index fetched from the index buffer before it is passed to the
    ///                           vertex shader.
    /// @param [in] firstInstance Starting instance for the draw.  Instance IDs passed to the vertex shader will range
    ///                           from firstInstance to firstInstance + instanceCount - 1.
    /// @param [in] instanceCount Number of instances to draw.  If zero, the draw will be discarded.
    /// @param [in] drawId        Draw index for the draw.
    void CmdDrawIndexed(
        uint32 firstIndex,
        uint32 indexCount,
        int32  vertexOffset,
        uint32 firstInstance,
        uint32 instanceCount,
        uint32 drawId)
    {
        m_funcTable.pfnCmdDrawIndexed(this, firstIndex, indexCount, vertexOffset, firstInstance, instanceCount, drawId);
    }

    /// Issues instanced, non-indexed draw calls using the command buffer's currently bound graphics state.  The draw
    /// arguments come from GPU memory. This command will issue count draw calls, using the provided stride to find
    /// the next indirect args structure in gpuMemory.  Each draw call will be discarded if its vertexCount or
    /// instanceCount is zero.
    ///
    /// The layout of the argument data is defined in the @ref DrawIndirectArgs structure.
    ///
    /// It is an error if the currently bound pipeline contains a mesh and/or task shader.
    ///
    /// This function requires use of the following barrier flags on the indirect memory:
    /// - PipelineStage:  @ref PipelineStageFetchIndirectArgs
    /// - CacheCoherency: @ref CoherIndirectArgs
    ///
    /// @see CmdDraw
    /// @see DrawIndirectArgs
    ///
    /// @param [in] gpuVirtAddrAndStride  GPU virtual address where the indirect argument data is located and stride in
    ///                                   memory from one structure to another.
    ///                                   The virtual address must be 4 byte aligned.
    /// @param [in] maximumCount          Maximum count of data structures to loop through.  If countGpuAddr
    ///                                   is nonzero, the value at that memory location is clamped to
    ///                                   this maximum. If countGpuAddr is zero, then the number of draws
    ///                                   issued exactly matches this number.
    /// @param [in] countGpuAddr          GPU virtual address where the number of draws is stored.
    ///                                   Must be 4-byte aligned.
    void CmdDrawIndirectMulti(
        GpuVirtAddrAndStride gpuVirtAddrAndStride,
        uint32               maximumCount,
        gpusize              countGpuAddr)
    {
        m_funcTable.pfnCmdDrawIndirectMulti(this, gpuVirtAddrAndStride, maximumCount, countGpuAddr);
    }

    /// Issues instanced, indexed draw calls using the command buffer's currently bound graphics state.  The draw
    /// arguments come from GPU memory. This command will issue count draw calls, using the provided stride to find
    /// the next indirect args structure in gpuMemory.  Each draw call will be discarded if its indexCount or
    /// instanceCount is zero.
    ///
    /// The layout of the argument data is defined in the @ref DrawIndexedIndirectArgs structure.
    ///
    /// It is an error if the currently bound pipeline contains a mesh and/or task shader.
    ///
    /// This function requires use of the following barrier flags on the indirect memory:
    /// - PipelineStage:  @ref PipelineStageFetchIndirectArgs
    /// - CacheCoherency: @ref CoherIndirectArgs
    ///
    /// @see CmdDrawIndexed
    /// @see DrawIndexedIndirectArgs
    ///
    /// @param [in] gpuVirtAddrAndStride  GPU virtual address where the indirect argument data is located and stride in
    ///                                   memory from one structure to another.
    ///                                   The virtual address must be 4 byte aligned.
    /// @param [in] maximumCount          Maximum count of data structures to loop through.  If countGpuAddr
    ///                                   is nonzero, the value at that memory location is clamped to
    ///                                   this maximum. If countGpuAddr is zero, then the number of draws
    ///                                   issued exactly matches this number.
    /// @param [in] countGpuAddr          GPU virtual address where the number of draws is stored.
    ///                                   Must be 4-byte aligned.
    void CmdDrawIndexedIndirectMulti(
        GpuVirtAddrAndStride gpuVirtAddrAndStride,
        uint32               maximumCount,
        gpusize              countGpuAddr)
    {
        m_funcTable.pfnCmdDrawIndexedIndirectMulti(this, gpuVirtAddrAndStride, maximumCount, countGpuAddr);
    }

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 909
    /// Dispatches a compute workload of the given dimensions using the command buffer's currently bound compute state.
    ///
    /// The thread group size is defined in the compute shader.
    ///
    /// Supports PAL ABI and HSA ABI pipelines.
    ///
    /// @param [in] size Thread groups to dispatch. If any components are zero the dispatch will be discarded.
    inline void CmdDispatch(
        DispatchDims size)
    {
        m_funcTable.pfnCmdDispatch(this, size, {});
    }
#endif

    /// Dispatches a compute workload of the given dimensions using the command buffer's currently bound compute state.
    ///
    /// The thread group size is defined in the compute shader.
    ///
    /// Supports PAL ABI and HSA ABI pipelines.
    ///
    /// @param [in] size Thread groups to dispatch. If any components are zero the dispatch will be discarded.
    /// @param [in] infoFlags Additional information about the dispatch.
    void CmdDispatch(
        DispatchDims      size,
        DispatchInfoFlags infoFlags)
    {
        m_funcTable.pfnCmdDispatch(this, size, infoFlags);
    }

    /// Dispatches a compute workload using the command buffer's currently bound compute state.  The dimensions of the
    /// workload come from GPU memory.  The dispatch will be discarded if any of its dimensions are zero.
    ///
    /// The layout of the argument data is defined in the @ref DispatchIndirectArgs structure.
    ///
    /// @warning Does not support HSA ABI pipelines.
    ///
    /// This function requires use of the following barrier flags on the indirect memory:
    /// - PipelineStage:  @ref PipelineStageFetchIndirectArgs
    /// - CacheCoherency: @ref CoherIndirectArgs
    ///
    /// @see CmdDispatch
    /// @see DispatchIndirectArgs
    ///
    /// @param [in] gpuVirtAddr  GPU virtual memory address where the indirect argument data is located.
    ///                          The virtual address must be 4-byte aligned.
    void CmdDispatchIndirect(
        gpusize gpuVirtAddr)
    {
        m_funcTable.pfnCmdDispatchIndirect(this, gpuVirtAddr);
    }

    /// Dispatches a compute workload of the given dimensions and offsets using the command buffer's currently bound
    /// compute state. This command allows targeting regions of thread groups without adding the offset computations in
    /// the shader.
    ///
    /// The caller may also provide a logical thread group count which is larger than the number of groups actually
    /// launched. If the shader reads the dispatch's thread group count from PAL metadata it will see the logical size,
    /// not the launch size.
    ///
    /// The combination of an offset, launch size, and logical size give the caller enough flexibility to take an
    /// incoming dispatch, split it up into sub-dispatches, and execute those sub-dispatches using multiple
    /// CmdDispatchOffset calls in whatever execution pattern they would like. Note that such an optimization
    /// would not work if the shader has global logic that does make assumptions about thread group launch order.
    ///
    /// The thread group size is defined in the compute shader.
    ///
    /// Supports PAL ABI and HSA ABI pipelines.
    ///
    /// @param [in] offset      The thread groups offsets. Set them to zero if you don't want an offset.
    /// @param [in] launchSize  Thread groups to dispatch. If any components are zero the dispatch will be discarded.
    /// @param [in] logicalSize The thread group dimensions reported to the shader via metadata.
    void CmdDispatchOffset(
        DispatchDims offset,
        DispatchDims launchSize,
        DispatchDims logicalSize)
    {
        m_funcTable.pfnCmdDispatchOffset(this, offset, launchSize, logicalSize);
    }

    /// Dispatches a mesh shader workload using the command buffer's currently bound graphics state.  It is an error if
    /// the currently bound graphics pipeline does not contain a mesh and/or task shader.
    ///
    /// The thread group size is defined in the mesh shader or task shader.
    ///
    /// @param [in] size Thread groups to dispatch. If any components are zero the dispatch will be discarded.
    void CmdDispatchMesh(
        DispatchDims size)
    {
        m_funcTable.pfnCmdDispatchMesh(this, size);
    }

    /// Dispatches a mesh shader workload using the command buffer's currently bound graphics state.  It is an error if
    /// the currently bound graphics pipeline does not contain a mesh shader.  The dimensions of the workload come from
    /// GPU memory.  The dispatch will be discarded if any of its dimensions are zero.
    ///
    /// The layout of the argument data is defined in the @ref DispatchMeshIndirectArgs structure.
    ///
    /// This function requires use of the following barrier flags on the indirect memory:
    /// - PipelineStage:  @ref PipelineStageFetchIndirectArgs
    /// - CacheCoherency: @ref CoherIndirectArgs
    ///
    /// @see CmdDispatchMesh
    /// @see DispatchMeshIndirectArgs
    ///
    /// @param [in] gpuVirtAddrAndStride  GPU virtual address where the indirect argument data is located and stride in
    ///                                   memory from one structure to another.
    ///                                   The virtual address must be 4 byte aligned.
    /// @param [in] maximumCount          Maximum count of data structures to loop through.  If countGpuAddr
    ///                                   is nonzero, the value at that memory location is clamped to
    ///                                   this maximum. If countGpuAddr is zero, then the number of draws
    ///                                   issued exactly matches this number.
    /// @param [in] countGpuAddr          GPU virtual address where the number of draws is stored.
    ///                                   Must be 4-byte aligned.
    void CmdDispatchMeshIndirectMulti(
        GpuVirtAddrAndStride gpuVirtAddrAndStride,
        uint32               maximumCount,
        gpusize              countGpuAddr)
    {
        m_funcTable.pfnCmdDispatchMeshIndirectMulti(this, gpuVirtAddrAndStride, maximumCount, countGpuAddr);
    }

    /// Copies multiple regions from one GPU memory allocation to another.
    ///
    /// None of the destination regions are allowed to overlap each other, nor are destination and source regions
    /// allowed to overlap when the source and destination GPU memory allocations are the same.  Any illegal overlapping
    /// will cause undefined results.
    ///
    /// This call should be used for buffer memory copy only; don't use it for image memory.
    ///
    /// For best performance, offsets and copy sizes should be 4-byte aligned.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopySrc for the source and @ref CoherCopyDst for the destination.
    ///
    /// @param [in] srcGpuMemory  GPU memory allocation where the source regions are located.
    /// @param [in] dstGpuMemory  GPU memory allocation where the destination regions are located.
    /// @param [in] regionCount   Number of regions to copy; size of the pRegions array.
    /// @param [in] pRegions      Array of copy regions, each entry specifying a source offset, destination offset, and
    ///                           copy size.
    virtual void CmdCopyMemory(
        const IGpuMemory&       srcGpuMemory,
        const IGpuMemory&       dstGpuMemory,
        uint32                  regionCount,
        const MemoryCopyRegion* pRegions) = 0;

    /// Copies multiple regions from one GPU memory virtual address to another.
    ///
    /// @note  The CmdCopyMemory() path should be preferred because it contains more optimizations due to more
    ///        knowledge about the memory itself that is lost when only virtual addresses are passed in.
    ///
    ///
    /// None of the destination regions are allowed to overlap each other, nor are destination and source regions
    /// allowed to overlap when the source and destination GPU memory virtual address are the same.  Any illegal
    /// overlapping will cause undefined results.
    ///
    /// For best performance, addresses, offsets, and copy sizes should be 4-byte aligned.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopySrc for the source and @ref CoherCopyDst for the destination.
    ///
    /// @param [in] srcGpuVirtAddr  GPU memory vitrual address where the source regions are located.
    /// @param [in] dstGpuVirtAddr  GPU memory virtual address where the destination regions are located.
    /// @param [in] regionCount     Number of regions to copy; size of the pRegions array.
    /// @param [in] pRegions        Array of copy regions, each entry specifynig a source offset, destination offset,
    ///                             and copy size.
    virtual void CmdCopyMemoryByGpuVa(
        gpusize                 srcGpuVirtAddr,
        gpusize                 dstGpuVirtAddr,
        uint32                  regionCount,
        const MemoryCopyRegion* pRegions) = 0;

    /// Copies multiple regions from one image to another.
    ///
    /// The source and destination subresource of a particular region are not allowed to be the same, and will produce
    /// undefined results.  Additionally, destination subresources cannot be present more than once per CmdCopyImage()
    /// call.
    ///
    /// For compressed images, the compression block size is used as the pixel size.  For compressed images, the image
    /// extents are specified in compression blocks.
    ///
    /// The source and destination images must to be of the same type (1D, 2D or 3D), or optionally 2D and 3D with the
    /// number of slices matching the depth.  MSAA source and destination images must have the same number of samples.
    ///
    /// Each region must satisfy these restrictions.
    /// - srcOffset >= 0 and dstOffset >= 0
    /// - srcOffset + extent <= srcSubres's extent
    /// - dstOffset + extent <= dstSubres's extent
    ///
    /// Images copied via this function must have x/y/z offsets and width/height/depth extents aligned to the minimum
    /// tiled copy alignment specified in @ref DeviceProperties for the engine this function is executed on.  Note that
    /// the DMA engine supports tiled copies regardless of the alignment; the reported minimum tiled copy alignments
    /// are an indication of the minimum alignments for which the copy will be performant.
    ///
    /// When the per-engine capability flag supportsMismatchedTileTokenCopy (@see DeviceProperties) is false,
    /// CmdCopyImage is only valid between two subresources that share the same tileToken (@see SubresLayout).
    ///
    /// Note that the copy can go through clone copy automatically if,
    ///  - Both source and destination images are created with @ref ImageCreateInfo::flags::cloneable = 1
    ///  - Both source and destination images have same @ref ImageCreateInfo
    ///  - Source image's layout is compatible with destination images' layout
    ///  - This is a full image copy
    ///  - Copy flags @ref CopyControlFlags required to be 0.
    ///
    /// Basically clone copy clones all subresources' data of one image object in another while preserving the image
    /// layout. It does raw copy on image data and metadata; and tries to keep the metadata (like DCC/HiZ/HiS)
    /// unchanged but may be not true due to different HW design.
    /// e.g. Client compression (fragment and ZPlane compression) will be missed during the compute based raw copy.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopySrc for the source and @ref CoherCopyDst for the destination.
    /// - ImageLayout:    @ref LayoutCopySrc for the source and @ref LayoutCopyDst for the destination.
    ///
    /// @param [in] srcImage       Image where source regions reside.
    /// @param [in] srcImageLayout Current allowed usages and engines for the source image.  These masks must include
    ///                            LayoutCopySrc and the ImageLayoutEngineFlags corresponding to the engine this
    ///                            function is being called on.
    /// @param [in] dstImage       Image where destination regions reside.
    /// @param [in] dstImageLayout Current allowed usages and engines for the destination image.  These masks must
    ///                            include LayoutCopyDst and the ImageLayoutEngineFlags corresponding to the engine this
    ///                            function is being called on.
    /// @param [in] regionCount    Number of regions to copy; size of the pRegions array.
    /// @param [in] pRegions       Array of copy regions, each entry specifying a source subresource, destination
    ///                            subresource, source x/y/z offset, destination x/y/z offset, and copy size in the
    ///                            x/y/z dimensions.
    /// @param [in] pScissorRect   Rectangle for scissor test.
    /// @param [in] flags          A mask of ORed @ref CopyControlFlags that can be used to control copy behavior.
    virtual void CmdCopyImage(
        const IImage&          srcImage,
        ImageLayout            srcImageLayout,
        const IImage&          dstImage,
        ImageLayout            dstImageLayout,
        uint32                 regionCount,
        const ImageCopyRegion* pRegions,
        const Rect*            pScissorRect,
        uint32                 flags) = 0;

    /// Copies data directly (without format conversion) from a GPU memory object to an image.
    ///
    /// For compressed images, the extents are specified in compression blocks.
    ///
    /// The size of the data copied from memory is implicitly derived from the image extents.
    ///
    /// The source memory offset has to be aligned to the smaller of the copied texel size or 4 bytes.  A destination
    /// subresource cannot be present more than once per CmdCopyMemoryToImage() call.
    ///
    /// Each region's imageOffset must be >= 0 and imageOffset + imageExtent must be <= imageSubres's extent.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopySrc for the source and @ref CoherCopyDst for the destination.
    /// - ImageLayout:    @ref LayoutCopyDst
    ///
    /// @param [in] srcGpuMemory   GPU memory where the source data is located.
    /// @param [in] dstImage       Image where destination data will be written.
    /// @param [in] dstImageLayout Current allowed usages and engines for the destination image.  These masks must
    ///                            include LayoutCopyDst and the ImageLayoutEngineFlags corresponding to the engine this
    ///                            function is being called on.
    /// @param [in] regionCount    Number of regions to copy; size of the pRegions array.
    /// @param [in] pRegions       Array of copy regions, each entry specifying a source offset, a destination
    ///                            subresource, destination x/y/z offset, and copy size in the x/y/z dimensions.
    virtual void CmdCopyMemoryToImage(
        const IGpuMemory&            srcGpuMemory,
        const IImage&                dstImage,
        ImageLayout                  dstImageLayout,
        uint32                       regionCount,
        const MemoryImageCopyRegion* pRegions) = 0;

    /// Copies data directly (without format conversion) from an image to a GPU memory object.
    ///
    /// For compressed images, the extents are specified in compression blocks.
    ///
    /// The size of the data copied to memory is implicitly derived from the image extents.
    ///
    /// The destination memory offset has to be aligned to the smaller of the copied texel size or 4 bytes.  A
    /// destination region cannot be present more than once per CmdCopyImageToMemory() call.
    ///
    /// Each region's imageOffset must be >= 0 and imageOffset + imageExtent must be <= imageSubres's extent.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopySrc for the source and @ref CoherCopyDst for the destination.
    /// - ImageLayout:    @ref LayoutCopySrc
    ///
    /// @param [in] srcImage       Image where source data will be read from.
    /// @param [in] srcImageLayout Current allowed usages and engines for the source image.  These masks must include
    ///                            LayoutCopySrc and the ImageLayoutEngineFlags corresponding to the engine this
    ///                            function is being called on.
    /// @param [in] dstGpuMemory   GPU memory where the destination data will be written.
    /// @param [in] regionCount    Number of regions to copy; size of the pRegions array.
    /// @param [in] pRegions       Array of copy regions, each entry specifying a destination offset, a source
    ///                            subresource, source x/y/z offset, and copy size in the x/y/z dimensions.
    virtual void CmdCopyImageToMemory(
        const IImage&                srcImage,
        ImageLayout                  srcImageLayout,
        const IGpuMemory&            dstGpuMemory,
        uint32                       regionCount,
        const MemoryImageCopyRegion* pRegions) = 0;

    /// Copies data directly (without format conversion) from a GPU memory object to a PRT.
    ///
    /// The image offset and extents are in units of tiles.  @see ImageMemoryLayout for the size of a tile in texels.
    /// This function always copies entire tiles, even if parts of the tile are internal padding.
    ///
    /// This function cannot be used to copy any subresources stored in the packed mip tail.  Other copy functions that
    /// operate in texels like the generic CmdCopyMemoryToImage() should be used instead.
    ///
    /// The size of the data copied from memory is implicitly derived from the image extents.
    ///
    /// The source memory offset has to be aligned to the smaller of the copied texel size or 4 bytes.  A destination
    /// subresource cannot be present more than once per CmdCopyMemoryToTiledImage() call.
    ///
    /// Each region's imageOffset must be >= 0 and imageOffset + imageExtent must be <= imageSubres's extent.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopySrc for the source and @ref CoherCopyDst for the destination.
    /// - ImageLayout:    @ref LayoutCopyDst
    ///
    /// @param [in] srcGpuMemory   GPU memory where the source data is located.
    /// @param [in] dstImage       Image where destination data will be written.  Must have the "prt" flag set.
    /// @param [in] dstImageLayout Current allowed usages and engines for the destination image.  These masks must
    ///                            include LayoutCopyDst and the ImageLayoutEngineFlags corresponding to the engine this
    ///                            function is being called on.
    /// @param [in] regionCount    Number of regions to copy; size of the pRegions array.
    /// @param [in] pRegions       Array of copy regions, each entry specifying a source offset, a destination
    ///                            subresource, destination x/y/z offset, and copy size in the x/y/z dimensions.
    virtual void CmdCopyMemoryToTiledImage(
        const IGpuMemory&                 srcGpuMemory,
        const IImage&                     dstImage,
        ImageLayout                       dstImageLayout,
        uint32                            regionCount,
        const MemoryTiledImageCopyRegion* pRegions) = 0;

    /// Copies data directly (without format conversion) from a PRT to a GPU memory object.
    ///
    /// The image offset and extents are in units of tiles.  @see ImageMemoryLayout for the size of a tile in texels.
    /// This function always copies entire tiles, even if parts of the tile are internal padding.
    ///
    /// This function cannot be used to copy any subresources stored in the packed mip tail.  Other copy functions that
    /// operate in texels like the generic CmdCopyImageToMemory() should be used instead.
    ///
    /// The size of the data copied to memory is implicitly derived from the image extents.
    ///
    /// The destination memory offset has to be aligned to the smaller of the copied texel size or 4 bytes.  A
    /// destination region cannot be present more than once per CmdCopyTiledImageToMemory() call.
    ///
    /// Each region's imageOffset must be >= 0 and imageOffset + imageExtent must be <= imageSubres's extent.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopySrc for the source and @ref CoherCopyDst for the destination.
    /// - ImageLayout:    @ref LayoutCopySrc
    ///
    /// @param [in] srcImage       Image where source data will be read from.
    /// @param [in] srcImageLayout Current allowed usages and queues for the source image.  These masks must include
    ///                            LayoutCopySrc and the ImageLayoutEngineFlags corresponding to the engine this
    ///                            function is being called on.
    /// @param [in] dstGpuMemory   GPU memory where the destination data will be written.
    /// @param [in] regionCount    Number of regions to copy; size of the pRegions array.
    /// @param [in] pRegions       Array of copy regions, each entry specifying a destination offset, a source
    ///                            subresource, source x/y/z offset, and copy size in the x/y/z dimensions.
    virtual void CmdCopyTiledImageToMemory(
        const IImage&                     srcImage,
        ImageLayout                       srcImageLayout,
        const IGpuMemory&                 dstGpuMemory,
        uint32                            regionCount,
        const MemoryTiledImageCopyRegion* pRegions) = 0;

    /// Copies multiple regions directly (without format conversion) from one typed buffer to another.
    ///
    /// For compressed formats, the extents are specified in compression blocks.
    ///
    /// The buffer memory offsets have to be aligned to the smaller of their texel sizes or 4 bytes.
    ///
    /// None of the destination regions are allowed to overlap each other, nor are destination and source regions
    /// allowed to overlap when the source and destination GPU memory allocations are the same.  Any illegal overlapping
    /// will cause undefined results.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopySrc for the source and @ref CoherCopyDst for the destination.
    ///
    /// @param [in] srcGpuMemory GPU memory where the source data is located.
    /// @param [in] dstGpuMemory GPU memory where the destination data will be written.
    /// @param [in] regionCount  Number of regions to copy; size of the pRegions array.
    /// @param [in] pRegions     Array of copy regions, each entry specifying a destination offset, a source offset,
    ///                          and copy size in the x/y/z dimensions.
    virtual void CmdCopyTypedBuffer(
        const IGpuMemory&            srcGpuMemory,
        const IGpuMemory&            dstGpuMemory,
        uint32                       regionCount,
        const TypedBufferCopyRegion* pRegions) = 0;

    /// Copies data directly (without format conversion) from a 2D typed buffer to a 2D image.
    ///
    /// For compressed images, the extents are specified in compression blocks.
    ///
    /// The source memory offset has to be aligned to the smaller of the copied texel size or 4 bytes.  A destination
    /// subresource cannot be present more than once per CmdScaledCopyTypedBufferToImage() call.
    ///
    /// MSAA resource is unsupported. The client must resolve both resources before calling this function.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopySrc for the source and @ref CoherCopyDst for the destination.
    /// - ImageLayout:    @ref LayoutCopyDst
    ///
    /// @param [in] srcGpuMemory   GPU memory where the source data is located.
    /// @param [in] dstImage       Image where destination data will be written.
    /// @param [in] dstImageLayout Current allowed usages and engines for the destination image.  These masks must
    ///                            include LayoutCopyDst and the ImageLayoutEngineFlags corresponding to the engine this
    ///                            function is being called on.
    /// @param [in] regionCount    Number of regions to copy; size of the pRegions array.
    /// @param [in] pRegions       Array of copy regions, each entry specifying a source offset, copy size of source
    ///                            region, a destination offset, destination subresource, and copy size of destination
    ///                            region.
    virtual void CmdScaledCopyTypedBufferToImage(
        const IGpuMemory&                       srcGpuMemory,
        const IImage&                           dstImage,
        ImageLayout                             dstImageLayout,
        uint32                                  regionCount,
        const TypedBufferImageScaledCopyRegion* pRegions) = 0;

    /// Copies a GPU register content to a GPU memory location.
    ///
    /// The destination memory offset has to be aligned to 4 bytes.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStagePostPrefetch
    /// - CacheCoherency: @ref CoherMemory
    ///
    /// @param [in] srcRegisterOffset Source register offset in bytes
    /// @param [in] dstGpuMemory      GPU memory where the destination data will be written.
    /// @param [in] dstOffset         Destination memory offset in bytes.
    virtual void CmdCopyRegisterToMemory(
        uint32            srcRegisterOffset,
        const IGpuMemory& dstGpuMemory,
        gpusize           dstOffset) = 0;

    /// Copies multiple scaled regions from one image to another.
    ///
    /// The source and destination subresource of a particular region are not allowed to be the same, and will produce
    /// undefined results.  Additionally, destination subresources cannot be present more than once per
    /// CmdScaledCopyImage() call.
    ///
    /// For compressed images, the compression block size is used as the pixel size.  For compressed images, the image
    /// extents are specified in compression blocks.
    ///
    /// The source and destination images must to be of the same type (1D, 2D or 3D). Both single sampled images and
    /// MSAA images are supported.
    ///
    /// Linear texture filtering is only supported for images with non-integer formats.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopySrc for the source and @ref CoherCopyDst for the destination.
    /// - ImageLayout:    @ref LayoutCopySrc for the source and @ref LayoutCopyDst for the destination.
    ///
    /// @param [in] copyInfo       Specifies parameters needed to execute CmdScaledCopyImage. See
    ///                            @ref ScaledCopyInfo for more information.
    virtual void CmdScaledCopyImage(
        const ScaledCopyInfo& copyInfo) = 0;

    /// Automatically generates texture data for a range of subresources such that they may be used as intermediate
    /// images in a mipmap chain. The existing values in mip N are used to generate mip N+1.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopySrc for the base mip, @ref CoherCopySrc and @ref CoherCopyDst for the others.
    /// - ImageLayout:    @ref LayoutCopySrc for the base mip, @ref LayoutCopySrc and @ref LayoutCopyDst for the others.
    ///
    /// @param [in] genInfo The parameters for CmdGenerateMipmaps. See @ref GenMipmapsInfo for more information.
    virtual void CmdGenerateMipmaps(
        const GenMipmapsInfo& genInfo) = 0;

    /// Copies multiple scaled regions from one image to another, converting between RGB and YUV color spaces during
    /// the copy.  The exact conversion between YUV and RGB is controlled by a caller-specified color-space-conversion
    /// table.
    ///
    /// The source and destination images must both be of the 2D type.  Only single-sampled images are supported.
    /// One of the two images involved must have an RGB color format, and the other must have a YUV color format.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopySrc for the source and @ref CoherCopyDst for the destination.
    /// - ImageLayout:    @ref LayoutCopySrc for the source and @ref LayoutCopyDst for the destination.
    ///
    /// @param [in] srcImage       Images where source region reside.  If this is a YUV image, the destination must be
    ///                            RGB, and this copy will convert YUV to RGB.  Otherwise, the destination must be YUV,
    ///                            and the copy will convert RGB to YUV.
    /// @param [in] srcImageLayout Current allowed usages and engines for the source image.  These masks must include
    ///                            LayoutCopySrc and the ImageLayoutEngineFlags corresponding to the engine this
    ///                            function is being called on.
    /// @param [in] dstImage       Image where destination regions reside.  If this is a YUV image, the source must be
    ///                            RGB, and this copy will convert RGB to YUV.  Otherwise, the source must be YUV and
    ///                            the copy will convert YUV to RGB.
    /// @param [in] dstImageLayout Current allowed usages and engines for the destination image.  These masks must
    ///                            include LayoutCopyDst and the ImageLayoutEngineFlags corresponding to the engine this
    ///                            function is being called on.
    /// @param [in] regionCount    Number of regions to copy; size of the pRegions array.
    /// @param [in] pRegions       Array of conversion-copy regions, each entry specifying a source x/y/z offset, source
    ///                            x/y/z extent, destination x/y/z offset, destination x/y/z extent, RGB subresource and
    ///                            YUV subresource(s).
    /// @param [in] filter         Texture filtering for shader sample instruction.
    /// @param [in] cscTable       Color-space-conversion table which controls how YUV data is converted to a specific
    ///                            RGB representation and vice-versa.
    virtual void CmdColorSpaceConversionCopy(
        const IImage&                     srcImage,
        ImageLayout                       srcImageLayout,
        const IImage&                     dstImage,
        ImageLayout                       dstImageLayout,
        uint32                            regionCount,
        const ColorSpaceConversionRegion* pRegions,
        TexFilter                         filter,
        const ColorSpaceConversionTable&  cscTable) = 0;

    /// Clones data of one image object in another while preserving the image layout.
    ///
    /// The source and destination images must be created with identical creation parameters and must specify the
    /// cloneable flag. The clone operation clones all subresources.
    ///
    /// Both resources can be in any layout before the clone operation.  After the clone, the source image state is left
    /// intact and the destination image layout becomes the same as the source.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopySrc for the source and @ref CoherCopyDst for the destination.
    ///
    /// @param [in] srcImage Source image.
    /// @param [in] dstImage Destination image.
    virtual void CmdCloneImageData(
        const IImage& srcImage,
        const IImage& dstImage) = 0;

    /// Directly updates a range of GPU memory with a small amount of host data.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopyDst
    ///
    /// The client is responsible for choosing the proper method for optimal performance. If updating data size is less
    /// equal than 8 bytes, CmdWriteImmediate() is preferred.
    ///
    /// @param [in] dstGpuMemory  GPU memory object to be updated.
    /// @param [in] dstOffset     Byte offset into the GPU memory object to be udpated.  Must be a multiple of 4.
    /// @param [in] dataSize      Amount of data to write, in bytes.  Must be a multiple of 4.
    /// @param [in] pData         Pointer to host data to be copied into the GPU memory.
    virtual void CmdUpdateMemory(
        const IGpuMemory& dstGpuMemory,
        gpusize           dstOffset,
        gpusize           dataSize,
        const uint32*     pData) = 0;

    /// Updates marker surface with a DWORD value to indicate an event completion.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStagePostPrefetch
    /// - CacheCoherency: @ref CoherCp
    ///
    /// @param [in] dstGpuMemory  GPU memory object to be updated.
    /// @param [in] offset        Byte offset into marker address
    /// @param [in] value         Marker DWORD value to be copied to the bus addressable or external physical memory.
    virtual void CmdUpdateBusAddressableMemoryMarker(
        const IGpuMemory& dstGpuMemory,
        gpusize           offset,
        uint32            value) = 0;

    /// Fills a range of GPU memory with the provided 32-bit data.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopyDst
    ///
    /// @param [in] dstGpuMemory  GPU memory object to be filled.
    /// @param [in] dstOffset     Byte offset into the GPU memory object to be filled.  Must be a multiple of 4.
    /// @param [in] fillSize      Size to fill, in bytes.  Must be a multiple of 4.
    /// @param [in] data          32-bit value to be repeated in the filled range.
    virtual void CmdFillMemory(
        const IGpuMemory& dstGpuMemory,
        gpusize           dstOffset,
        gpusize           fillSize,
        uint32            data) = 0;

    /// Interprets a range of GPU memory as a color buffer and clears it to the specified clear color.
    ///
    /// The maximum clear range is determined by the buffer offset and buffer extent; if any Ranges are specified they
    /// must be specified in texels with respect to the beginning of the buffer and must not exceed its extent.
    /// With 96-bit formats, bufferOffset must be specified in bytes.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageCs is expected but the more general @ref PipelineStageBlt is also OK.
    /// - CacheCoherency: @ref CoherShader is expected but the more general @ref CoherClear is also OK.
    ///
    /// @param [in] gpuMemory     GPU memory to be cleared.
    /// @param [in] color         Specifies the clear color data and how to interpret it.
    /// @param [in] bufferFormat  The format of the color data in the buffer.
    /// @param [in] bufferOffset  The offset to the beginning of the buffer, in units of texels
    ///                           (or bytes for 96-bit texels).
    /// @param [in] bufferExtent  The extent of the buffer, in units of texels.
    /// @param [in] rangeCount    Number of ranges within the buffer to clear; size of the pRanges array.
    ///                           If zero, the entire view will be cleared and pRanges will be ignored.
    /// @param [in] pRanges       Array of ranges within the GPU memory to clear.
    virtual void CmdClearColorBuffer(
        const IGpuMemory& gpuMemory,
        const ClearColor& color,
        SwizzledFormat    bufferFormat,
        uint32            bufferOffset,
        uint32            bufferExtent,
        uint32            rangeCount = 0,
        const Range*      pRanges    = nullptr) = 0;

    /// Clears the currently bound color targets to the specified clear color.
    ///
    /// This will always result in a slow clear and should only be used when the actual image being cleared is unknown.
    /// In practice, this is the case when vkCmdClearColorAttachments() is called in a secondary command buffer in
    /// Vulkan where the color attachments are inherited.
    ///
    /// This requires regionCount being specified since resource size is for sure to be known. The bound color targets
    /// shouldn't have UndefinedSwizzledFormat as their swizzle format.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageColorTarget
    /// - CacheCoherency: @ref CoherColorTarget
    /// - ImageLayout:    @ref LayoutColorTarget
    ///
    /// @param [in] colorTargetCount      Number of bound color target that needs to be cleared.
    /// @param [in] pBoundColorTargets    Color target information for the bound color targets.
    /// @param [in] regionCount           Number of volumes within the image to clear; size of the pClearRegions array.
    ///                                   This need to be non-zero.
    /// @param [in] pClearRegions         Array of volumes within the subresources to clear.
    virtual void CmdClearBoundColorTargets(
        uint32                          colorTargetCount,
        const BoundColorTarget*         pBoundColorTargets,
        uint32                          regionCount,
        const ClearBoundTargetRegion*   pClearRegions) = 0;

    /// Clears a color image to the specified clear color.
    ///
    /// If any Boxes have been specified, all subresource ranges must contain a single, identical mip level.
    ///
    /// The imageLayout can include any valid layout (e.g. not @ref LayoutUninitializedTarget) but it is wise to stick
    /// to layouts that are likely to support compression like @ref LayoutColorTarget.
    ///
    /// This function requires use of the following barrier flags if @ref flags includes @ref ColorClearAutoSync:
    /// - PipelineStage:  @ref PipelineStageColorTarget
    /// - CacheCoherency: @ref CoherColorTarget
    /// - ImageLayout:    @ref LayoutColorTarget
    /// Otherwise the following barrier flags must be used:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherClear
    ///
    /// @param [in] image       Image to be cleared.
    /// @param [in] imageLayout Current allowed usages and engines for the target image.
    /// @param [in] color       Specifies the clear color data and how to interpret it.
    /// @param [in] clearFormat If clearFormat.format is Undefined (e.g. if UndefinedSwizzledFormat is provided), do not
    ///                         reinterpret the subresources' formats. Otherwise, the subresources' formats will be
    ///                         reinterpreted according to this parameter. The specified format needs to have been
    ///                         included in the "pViewFormats" list specified at image-creation time, otherwise
    ///                         corruption may occur.
    /// @param [in] rangeCount  Number of subresource ranges to clear; size of the pRanges array.
    /// @param [in] pRanges     Array of subresource ranges to clear.
    /// @param [in] boxCount    Number of volumes within the image to clear; size of the pBoxes array.
    ///                         If zero, entire subresources will be cleared and pBoxes will be ignored.
    /// @param [in] pBoxes      Array of volumes within the subresources to clear.
    /// @param [in] flags       Mask of ClearColorImageFlags values controlling behavior of the clear.
    virtual void CmdClearColorImage(
        const IImage&         image,
        ImageLayout           imageLayout,
        const ClearColor&     color,
        const SwizzledFormat& clearFormat,
        uint32                rangeCount,
        const SubresRange*    pRanges,
        uint32                boxCount,
        const Box*            pBoxes,
        uint32                flags) = 0;

    /// Clears the currently bound depth/stencil targets to the specified clear values.
    ///
    /// This will always result in a slow clear and should only be used when the actual image being cleared is unknown.
    /// In practice, this is the case when vkCmdClearColorAttachments() is called in a secondary command buffer in
    /// Vulkan where the color attachments are inherited.
    ///
    /// This requires regionCount being specified since resource size is for sure to be known.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageEarlyDsTarget and/or @ref PipelineStageLateDsTarget
    /// - CacheCoherency: @ref CoherDepthStencilTarget
    /// - ImageLayout:    @ref LayoutDepthStencilTarget
    ///
    /// @param [in] depth            Depth clear value.
    /// @param [in] stencil          Stencil clear value.
    /// @param [in] stencilWriteMask Stencil write mask to clear specific stencil planes.
    /// @param [in] samples          Sample count.
    /// @param [in] fragments        Fragment count.
    /// @param [in] flag             Select to depth, stencil or depth and stencil.
    /// @param [in] regionCount      Number of volumes within the bound depth/stencil target to clear.
    /// @param [in] pClearRegions    Array of volumes within the subresources to clear.
    virtual void CmdClearBoundDepthStencilTargets(
        float                         depth,
        uint8                         stencil,
        uint8                         stencilWriteMask,
        uint32                        samples,
        uint32                        fragments,
        DepthStencilSelectFlags       flag,
        uint32                        regionCount,
        const ClearBoundTargetRegion* pClearRegions) = 0;

    /// Clears a depth/stencil image to the specified clear values.
    ///
    /// If any Rects have been specified, all subresource ranges must contain a single, identical mip level.
    ///
    /// The layouts can include any valid layout (e.g. not @ref LayoutUninitializedTarget) but it is wise to stick to
    /// layouts that are likely to support compression like @ref LayoutDepthStencilTarget.
    ///
    /// This function requires use of the following barrier flags if @ref flags includes @ref DsClearAutoSync:
    /// - PipelineStage:  @ref PipelineStageEarlyDsTarget and/or @ref PipelineStageLateDsTarget
    /// - CacheCoherency: @ref CoherDepthStencilTarget
    /// - ImageLayout:    @ref LayoutDepthStencilTarget
    /// Otherwise the following barrier flags must be used:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherClear
    ///
    /// @param [in] image            Image to be cleared.
    /// @param [in] depth            Depth clear value.
    /// @param [in] depthLayout      Current allowed usages and engines for the depth plane.
    /// @param [in] stencil          Stencil clear value.
    /// @param [in] stencilWriteMask Write-mask to apply to the stencil subresource ranges during the clear.
    /// @param [in] stencilLayout    Current allowed usages and engines for the stencil plane.
    /// @param [in] rangeCount       Number of subresource ranges to clear; size of the pRanges array.
    /// @param [in] pRanges          Array of subresource ranges to clear.
    /// @param [in] rectCount        Number of areas within the image to clear; size of the pRects array.  If zero,
    ///                              the entire subresources will be cleared and pRects will be ignored.
    /// @param [in] pRects           Array of areas within the subresources to clear.
    /// @param [in] flags            Mask of ClearDepthStencilFlags values controlling behavior of the clear.
    virtual void CmdClearDepthStencil(
        const IImage&      image,
        ImageLayout        depthLayout,
        ImageLayout        stencilLayout,
        float              depth,
        uint8              stencil,
        uint8              stencilWriteMask,
        uint32             rangeCount,
        const SubresRange* pRanges,
        uint32             rectCount,
        const Rect*        pRects,
        uint32             flags) = 0;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 910
    /// Clears a range of GPU memory to the specified clear color using the specified buffer view SRD.
    ///
    /// The maximum clear range is determined by the view; if any Ranges are specified they must fit within the view's
    /// range. The view must support shader writes.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageCs
    /// - CacheCoherency: @ref CoherShader
    ///
    /// @note You may use the more general @ref PipelineStageBlt and @ref CoherClear if you wish but they may result in
    ///       higher barrier overhead.
    ///
    /// @param [in] gpuMemory      GPU memory to be cleared.
    /// @param [in] color          Specifies the clear color data and how to interpret it.
    /// @param [in] pBufferViewSrd The image view SRD that will be used to interpret the image.
    /// @param [in] rangeCount     Number of ranges within the GPU memory to clear; size of the pRanges array.
    ///                            If zero, the entire view will be cleared and pRanges will be ignored.
    /// @param [in] pRanges        Array of ranges within the GPU memory to clear.
    virtual void CmdClearBufferView(
        const IGpuMemory& gpuMemory,
        const ClearColor& color,
        const void*       pBufferViewSrd,
        uint32            rangeCount = 0,
        const Range*      pRanges    = nullptr) = 0;

    /// Clears an image to the specified clear color using the specified image view SRD.
    ///
    /// The clear subresouce range is determined by the view; if any Rects have been specified, the image view must
    /// contain a single mip level. The view must support shader writes.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageCs but the more general @ref PipelineStageBlt is also OK.
    /// - CacheCoherency: @ref CoherShader but the more general @ref CoherClear is also OK.
    /// - ImageLayout:    @ref LayoutShaderWrite
    ///
    /// @param [in] image         Image to be cleared.
    /// @param [in] imageLayout   Current allowed usages and engines for the image, must include LayoutShaderWrite.
    /// @param [in] color         Specifies the clear color data and how to interpret it.
    /// @param [in] pImageViewSrd The image view SRD that will be used to interpret the image.
    /// @param [in] rectCount     Number of volumes within the image to clear; size of the pRects array.
    ///                           If zero, entire subresources will be cleared and pRects will be ignored.
    /// @param [in] pRects        Array of volumes within the subresources to clear. The begin and end slices to be
    ///                           cleard are from SubresRange in pImageViewSrd.
    virtual void CmdClearImageView(
        const IImage&     image,
        ImageLayout       imageLayout,
        const ClearColor& color,
        const void*       pImageViewSrd,
        uint32            rectCount = 0,
        const Rect*       pRects    = nullptr) = 0;
#endif

    /// Resolves multiple regions of a multisampled image to a single-sampled image.
    ///
    /// The source image must be a 2D multisampled image and the destination must be a single-sampled image.
    /// The formats of the source and destination images must match unless all regions specify a valid format.
    ///
    /// For color images, if the source image has an integer numeric format, a single sample is copied (sample 0).
    ///
    /// For depth/stencil images, the resolve is performed by simply copying sample 0 from every source pixel to the
    /// destination pixel.
    ///
    /// The same subresource may not appear more than once in the specified array of regions.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherResolveSrc for the source and @ref CoherResolveDst for the destination.
    /// - ImageLayout:    @ref LayoutResolveSrc for the source and @ref LayoutResolveDst for the destination.
    ///
    /// @param [in] srcImage       MSAA source image.
    /// @param [in] srcImageLayout Current allowed usages and engines for the source image.  These masks must include
    ///                            LayoutResolveSrc and the ImageLayoutEngineFlags corresponding to the engine this
    ///                            function is being called on.
    /// @param [in] dstImage       Single-sample destination image.
    /// @param [in] dstImageLayout Current allowed usages and engines for the destination image.  These masks must
    ///                            include LayoutResolveDst and the ImageLayoutEngineFlags corresponding to the engine
    ///                            this function is being called on.
    /// @param [in] regionCount    Number of regions to resolve; size of the pRegions array.
    /// @param [in] resolveMode    Resolve mode
    /// @param [in] pRegions       Specifies src/dst subresources and rectangles.
    /// @param [in] flags          Mask of ResolveImageFlags values controlling behavior of the resolve.
    virtual void CmdResolveImage(
        const IImage&             srcImage,
        ImageLayout               srcImageLayout,
        const IImage&             dstImage,
        ImageLayout               dstImageLayout,
        ResolveMode               resolveMode,
        uint32                    regionCount,
        const ImageResolveRegion* pRegions,
        uint32                    flags) = 0;

    /// Resolves multiple regions of a Sampler Feedback map to another image.
    ///
    /// The source image must be:
    /// Decode: A 2D Feedback map, in which case the destination image will hold the decoded sampled data
    /// Encode: A texture, in which case the destination will hold the encoded sampler map
    ///
    /// The formats of the source and destination images must be 8bpp
    ///
    /// The same subresource may not appear more than once in the specified array of regions.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherShaderRead for the source and @ref CoherShaderWrite for the destination.
    /// - ImageLayout:    @ref LayoutShaderRead for the source and @ref LayoutShaderWrite for the destination.
    ///
    /// @param [in] srcImage       Source image, depends on resolve type
    /// @param [in] srcImageLayout Current allowed usages and engines for the source image.  These masks must
    ///                            include LayoutShaderRead and the ImageLayoutEngineFlags corresponding to the engine
    ///                            this function is being called on.
    /// @param [in] dstImage       Destination image, depends on resolve type
    /// @param [in] dstImageLayout Current allowed usages and engines for the destination image.  These masks must
    ///                            include LayoutShaderWrite and the ImageLayoutEngineFlags corresponding to the engine
    ///                            this function is being called on.
    /// @param [in] resolveType    Resolve type
    /// @param [in] regionCount    Number of regions to resolve; size of the pRegions array.
    /// @param [in] pRegions       Specifies src/dst subresources and rectangles
    virtual void CmdResolvePrtPlusImage(
        const IImage&                    srcImage,
        ImageLayout                      srcImageLayout,
        const IImage&                    dstImage,
        ImageLayout                      dstImageLayout,
        PrtPlusResolveType               resolveType,
        uint32                           regionCount,
        const PrtPlusImageResolveRegion* pRegions) = 0;

    /// Encodes a buffer into a Sampler Feedback image.
    ///
    /// This interface only supports encoding raw data from a buffer to an encoded Sampler Feedback map
    ///
    /// The format of the data in the source buffer and destination image must be 8bpp
    ///
    /// The same subresource may not appear more than once in the specified array of regions.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherShaderRead for the source and @ref CoherShaderWrite for the destination.
    /// - ImageLayout:    @ref LayoutShaderWrite for the destination.
    ///
    /// @param [in] srcBuffer      Source buffer, depends on the resolve type
    /// @param [in] dstImage       Destination image, depends on resolve type
    /// @param [in] dstImageLayout Current allowed usages and engines for the destination image.  These masks must
    ///                            include LayoutShaderWrite and the ImageLayoutEngineFlags corresponding to the engine
    ///                            this function is being called on.
    /// @param [in] regionCount    Number of regions to resolve, size of pRegions array
    /// @param [in] pRegions       Specifies src/dst subresources and rectangles
    virtual void CmdResolvePrtPlusBufferToImage(
        const IGpuMemory&                        srcBuffer,
        const IImage&                            dstImage,
        ImageLayout                              dstImageLayout,
        uint32                                   regionCount,
        const PrtPlusBufferToImageResolveRegion* pRegions) = 0;

    /// Decodes a Sampler Feedback map to a buffer
    ///
    /// This interface only supports decoding a Sampler Feedback map to buffer
    ///
    /// The format of the data in the source image and destination buffer must be 8bpp
    ///
    /// The same subresource may not appear more than once in the specified array of regions.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherShaderRead for the source and @ref CoherShaderWrite for the destination.
    /// - ImageLayout:    @ref LayoutShaderRead for the source
    ///
    /// @param [in] srcImage       Source image, depends on the resolve type
    /// @param [in] srcImageLayout Current allowed usages and engines for the source image.  These masks must
    ///                            include LayoutShaderRead and the ImageLayoutEngineFlags corresponding to the engine
    ///                            this function is being called on.
    /// @param [in] dstBuffer      Destination buffer, depends on resolve type
    /// @param [in] regionCount    Number of regions to resolve; size of the pRegions array.
    /// @param [in] pRegions       Specifies src/dst subresources and rectangles.
    virtual void CmdResolvePrtPlusImageToBuffer(
        const IImage&                            srcImage,
        ImageLayout                              srcImageLayout,
        const IGpuMemory&                        dstBuffer,
        uint32                                   regionCount,
        const PrtPlusImageToBufferResolveRegion* pRegions) = 0;

    /// Puts the specified event into the _set_ state when all prior GPU work has progressed past the given stages.
    ///
    /// @note Clients should use this version if they're using the CmdRelease/Acquire APIs.
    ///
    /// @param [in] gpuEvent  GPU event to be set.
    /// @param [in] stageMask A bitmask of @ref PipelineStageFlag values which defines a synchronization scope that
    ///                       restricts which stages of prior GPU work must happen before the event is set. The set
    ///                       will be performed at the earliest possible stage after the prior stages.
    virtual void CmdSetEvent(
        const IGpuEvent& gpuEvent,
        uint32           stageMask) = 0;

    /// Puts the specified event into the _reset_ state when all prior GPU work has progressed past the given stages.
    ///
    /// @note Clients should use this version if they're using the CmdRelease/Acquire APIs.
    ///
    /// @param [in] gpuEvent  GPU event to be reset.
    /// @param [in] stageMask A bitmask of @ref PipelineStageFlag values which defines a synchronization scope that
    ///                       restricts which stages of prior GPU work must happen before the event is reset. The
    ///                       reset will be performed at the earliest possible stage after the prior stages.
    virtual void CmdResetEvent(
        const IGpuEvent& gpuEvent,
        uint32           stageMask) = 0;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 900
    /// Puts the specified GPU event into the _set_ state when all previous GPU work reaches the specified point in the
    /// pipeline.
    ///
    /// @note Clients may use this version if they're using the legacy @ref CmdBarrier API.
    ///
    /// @param [in] gpuEvent GPU event to be set.
    /// @param [in] setPoint Point in the graphics pipeline where the GPU event will be _set_, indicating all prior
    ///                      issued GPU work has reached at least this point in the pipeline.  If the GPU doesn't
    ///                      support this operation at the exact specified point, the set will be performed at the
    ///                      earliest possible point _after_ the specified point.
    inline void CmdSetEvent(
        const IGpuEvent& gpuEvent,
        HwPipePoint      setPoint)
        { CmdSetEvent(gpuEvent, HwPipePointToStage[setPoint]); }

    /// Puts the specified GPU event into the _reset_ state when all previous GPU work reaches the specified point in
    /// the pipeline.
    ///
    /// @note Clients may use this version if they're using the legacy @ref CmdBarrier API.
    ///
    /// @param [in] gpuEvent   GPU event to be reset.
    /// @param [in] resetPoint Point in the graphics pipeline where the GPU event will be _reset_, indicating all prior
    ///                        issued GPU work has reached at least this point in the pipeline.  If the GPU doesn't
    ///                        support this operation at the exact specified point, the reset will be performed at the
    ///                        earliest possible point _after_ the specified point.
    inline void CmdResetEvent(
        const IGpuEvent& gpuEvent,
        HwPipePoint      resetPoint)
        { CmdResetEvent(gpuEvent, HwPipePointToStage[resetPoint]); }
#endif

    /// Predicate the subsequent jobs in the command buffer if the event is set.
    ///
    /// @param [in] gpuEvent   GPU event to be checked.
    virtual void CmdPredicateEvent(
        const IGpuEvent&  gpuEvent) = 0;

    /// Performs the specified 32- or 64-bit memory operation.  These operations are atomic with respect to shader
    /// atomic operations.
    ///
    /// The data size (32-bit or 64-bit) is determined by the operation type.  For 32-bit atomics, only the lower
    /// 32-bits of srcData is used.
    ///
    /// The destination GPU memory offset must be 4-byte aligned for 32-bit atomics and 8-byte aligned for 64-bit
    /// atomics.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  @ref PipelineStagePostPrefetch
    /// - CacheCoherency: @ref CoherQueueAtomic
    ///
    /// @param [in] dstGpuMemory  Destination GPU memory object.
    /// @param [in] dstOffset     Offset into the memory object where the atomic will be performed.
    /// @param [in] srcData       Source data for the atomic operation.  Use depends on the atomicOp.
    /// @param [in] atomicOp      Specifies which atomic operation to perform.  @see AtomicOp.
    virtual void CmdMemoryAtomic(
        const IGpuMemory& dstGpuMemory,
        gpusize           dstOffset,
        uint64            srcData,
        AtomicOp          atomicOp) = 0;

    /// Starts a query operation for the given slot of a query pool.
    ///
    /// The query slot must have been previously cleared with CmdResetQueryPool() before starting a query.
    ///
    /// @note Queries may not span multiple command buffers.
    ///
    /// @param [in] queryPool  Query pool for this query.
    /// @param [in] queryType  The type of query this operation will produce.
    /// @param [in] slot       Slot in pQueryPool where the results of this query should be accumulated.
    /// @param [in] flags      Flags controlling query behavior.  @see QueryControlFlags.
    virtual void CmdBeginQuery(
        const IQueryPool& queryPool,
        QueryType         queryType,
        uint32            slot,
        QueryControlFlags flags) = 0;

    /// Stops a query operation for the given slot of a query pool.
    ///
    /// The query slot must have an open query on it when this is called.
    ///
    /// @param [in] queryPool  Query pool for this query.
    /// @param [in] queryType  The type of query this operation will produce.
    /// @param [in] slot       Slot in pQueryPool where the query is running.
    virtual void CmdEndQuery(
        const IQueryPool& queryPool,
        QueryType         queryType,
        uint32            slot) = 0;

    /// Resolves the results of a range of queries to the specified query type into the specified GPU memory location.
    ///
    /// This function requires use of the following barrier flags on @ref dstGpuMemory:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopyDst
    ///
    /// This operation does not honor the command buffer's predication state, if active.
    ///
    /// @param [in] queryPool     Query pool holding the source queries.
    /// @param [in] flags         Flags that control the result data layout and how the results are retrieved.
    /// @param [in] queryType     The type of queries this resolve will produce.
    /// @param [in] startQuery    First slot in pQueryPool to resolve.
    /// @param [in] queryCount    Number of query pool slots to resolve.
    /// @param [in] dstGpuMemory  Destination GPU memory object.
    /// @param [in] dstOffset     4-byte aligned offset into pDstGpuMemory where the results should be written.
    /// @param [in] dstStride     4-byte aligned stride between where results are written into pDstGpuMemory.
    virtual void CmdResolveQuery(
        const IQueryPool& queryPool,
        QueryResultFlags  flags,
        QueryType         queryType,
        uint32            startQuery,
        uint32            queryCount,
        const IGpuMemory& dstGpuMemory,
        gpusize           dstOffset,
        gpusize           dstStride) = 0;

    /// Rests a range of slots in a query pool.  A query slot must be reset each time before a query can be started
    /// using that slot.
    ///
    /// @param [in] queryPool  Query pool to be reset.
    /// @param [in] startQuery First slot in pQueryPool to be reset.
    /// @param [in] queryCount Number of slots to reset.
    virtual void CmdResetQueryPool(
        const IQueryPool& queryPool,
        uint32            startQuery,
        uint32            queryCount) = 0;

    /// Writes a GPU performance timestamp to memory when all prior GPU work has progressed past the given stages.
    ///
    /// The timestamp data is a 64-bit value that increments once per clock. @ref timestampFrequency in DeviceProperties
    /// reports the frequency the timestamps are clocked at. Timestamps are only supported by engines that report
    /// @ref supportsTimestamps in DeviceProperties.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  the same flag(s) specified in @ref stageMask.
    /// - CacheCoherency: @ref CoherTimestamp
    ///
    /// @note Clients should use this version if they're using the CmdRelease/Acquire APIs.
    ///
    /// @param [in] stageMask    A bitmask of @ref PipelineStageFlag values which defines a synchronization scope that
    ///                          restricts which stages of prior GPU work must happen before the timestamp is written.
    ///                          The timestamp will be performed at the earliest possible stage after the prior stages.
    ///                          Note that the SDMA engine only supports bottom-of-pipe timestamps.
    /// @param [in] dstGpuMemory GPU memory object where timestamp should be written.
    /// @param [in] dstOffset    Offset into pDstGpuMemory where the timestamp should be written.  Must be aligned to
    ///                          minTimestampAlignment in DeviceProperties.
    virtual void CmdWriteTimestamp(
        uint32            stageMask,
        const IGpuMemory& dstGpuMemory,
        gpusize           dstOffset) = 0;

    /// Writes an immediate value to memory when all prior GPU work has progressed past the given stages.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  the same flag(s) specified in @ref stageMask.
    /// - CacheCoherency: @ref CoherCp
    ///
    /// @note Clients should use this version if they're using the CmdRelease/Acquire APIs.
    ///
    /// @param [in] stageMask          A bitmask of @ref PipelineStageFlag values which defines a synchronization scope
    ///                                that restricts which stages of prior GPU work must happen before the immediate
    ///                                value is written. The write will be occur at the earliest possible stage after
    ///                                the prior stages. Note that the SDMA engine only supports bottom-of-pipe writes.
    /// @param [in] data               Value to be written to gpu address.
    /// @param [in] ImmediateDataWidth Size of the data to be written out.
    /// @param [in] address            GPU address where immediate value should be written.
    virtual void CmdWriteImmediate(
        uint32             stageMask,
        uint64             data,
        ImmediateDataWidth dataSize,
        gpusize            address) = 0;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 900
    /// Writes a HwPipePostPrefetch or HwPipeBottom timestamp to the specified memory location.
    ///
    /// The timestamp data is a 64-bit value that increments once per clock.  timestampFrequency in DeviceProperties
    /// reports the frequency the timestamps are clocked at.
    ///
    /// Timestamps are only supported by engines that report supportsTimestamps in DeviceProperties.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  the same flag specified in @ref pipePoint.
    /// - CacheCoherency: @ref CoherTimestamp
    ///
    /// @note Clients may use this version if they're using the legacy @ref CmdBarrier API.
    ///
    /// @param [in] pipePoint    Specifies where in the pipeline the timestamp should be sampled and written. The only
    ///                          valid choices are HwPipePostPrefetch and HwPipeBottom. HwPipePostPrefetch timestamps
    ///                          are not supported on the SDMA engine, so all timestamps will be executed as
    ///                          bottom-of-pipe.
    /// @param [in] dstGpuMemory GPU memory object where timestamp should be written.
    /// @param [in] dstOffset    Offset into pDstGpuMemory where the timestamp should be written.  Must be aligned to
    ///                          minTimestampAlignment in DeviceProperties.
    inline void CmdWriteTimestamp(
        HwPipePoint       pipePoint,
        const IGpuMemory& dstGpuMemory,
        gpusize           dstOffset)
        { CmdWriteTimestamp(HwPipePointToStage[pipePoint], dstGpuMemory, dstOffset); }

    /// Writes a top-of-pipe or bottom-of-pipe immediate value to the specified memory location.
    ///
    /// This function requires use of the following barrier flags:
    /// - PipelineStage:  the same flag specified in @ref pipePoint.
    /// - CacheCoherency: @ref CoherCp
    ///
    /// @note Clients may use this version if they're using the legacy @ref CmdBarrier API.
    ///
    /// @param [in] pipePoint          Specifies where in the pipeline the timestamp should be sampled and written.
    ///                                The only valid choices are HwPipeTop, HwPipePostPrefetch and HwPipeBottom.
    ///                                Top-of-pipe timestamps are not supported on the SDMA engine, so all timestamps
    ///                                will be executed as bottom-of-pipe.
    /// @param [in] data               Value to be written to gpu address.
    /// @param [in] ImmediateDataWidth Size of the data to be written out.
    /// @param [in] address            GPU address where immediate value should be written.
    inline void CmdWriteImmediate(
        HwPipePoint        pipePoint,
        uint64             data,
        ImmediateDataWidth dataSize,
        gpusize            address)
        { CmdWriteImmediate(HwPipePointToStage[pipePoint], data, dataSize, address); }
#endif

    /// Loads the current stream-out buffer-filled-sizes stored on the GPU from memory, typically from a target of a
    /// prior CmdSaveBufferFilledSizes() call.
    ///
    /// Note that barriers shouldn't be necessary in normal stream-out workflows. However, if the client wishes to use
    /// the @ref gpuVirtAddr allocations in shaders or PAL blts they must use the following barrier flags:
    /// - PipelineStage:  @ref PipelineStagePostPrefetch
    /// - CacheCoherency: @ref CoherCp
    ///
    /// @param [in] gpuVirtAddr Array of GPU virtual addresses to load each counter from.  If any of these are zero,
    ///                         the corresponding filled-size counter is not loaded.
    virtual void CmdLoadBufferFilledSizes(
        const gpusize (&gpuVirtAddr)[MaxStreamOutTargets]) = 0;

    /// Saves the current stream-out buffer-filled-sizes into GPU memory.
    ///
    /// Note that barriers shouldn't be necessary in normal stream-out workflows. However, if the client wishes to use
    /// the @ref gpuVirtAddr allocations in shaders or PAL blts they must use the following barrier flags:
    /// - PipelineStage:  @ref PipelineStagePostPrefetch
    /// - CacheCoherency: @ref CoherCp
    ///
    /// @param [in] gpuVirtAddr Array of GPU virtual addresses to save each counter into.  If any of these are zero,
    ///                         the corresponding filled-size counter is not saved.
    virtual void CmdSaveBufferFilledSizes(
        const gpusize (&gpuVirtAddr)[MaxStreamOutTargets]) = 0;

    /// Set the offset to buffer-filled-size for a stream-out target.
    ///
    /// @param [in] bufferId   Stream-out buffer ID, it could be in the range [0, MaxStreamOutTargets).
    /// @param [in] offset     The value to be written into the buffer filled size counter.
    ///
    virtual void CmdSetBufferFilledSize(
        uint32  bufferId,
        uint32  offset) = 0;

    /// Binds the specified border color palette for use by samplers.
    ///
    /// @param [in] pipelineBindPoint Specifies which pipeline type is affected (i.e., graphics or compute).
    /// @param [in] pPalette          Border color palette object to bind.
    virtual void CmdBindBorderColorPalette(
        PipelineBindPoint          pipelineBindPoint,
        const IBorderColorPalette* pPalette) = 0;

    /// Sets predication for this command buffer to use the specified GPU memory location. Any draw, dispatch or copy
    /// operation between this command and the corresponding reset/disable call will be skipped if the value in spec-
    /// ified location matches the passed-in predicated value
    ///
    /// This function requires use of the following barrier flags on @ref pGpuMemory:
    /// - PipelineStage:  @ref PipelineStageFetchIndirectArgs
    /// - CacheCoherency: @ref CoherIndirectArgs
    ///
    /// @param [in] pQueryPool     pointer to QueryPool obj, not-nullptr means this is a QueryPool based predication
    ///                                - Zpass/Occlusion based predication
    ///                                - or PrimCount/Streamout based predication
    /// @param [in] slot           Slot to use for setting occlusion predication, valid when pQueryPool is not nullptr
    /// @param [in] pGpuMemory     GPU memory object for the predication value, only valid when pQueryPool is nullptr
    /// @param [in] offset         GPU memory offset for the predication value
    /// @param [in] predType       Predication type.
    /// @param [in] predPolarity   Controls the polarity of the predication test
    ///                                true  = draw_if_visible_or_no_overflow
    ///                                false = draw_if_not_visible_or_overflow
    /// @param [in] waitResults    Hint only valid for Zpass/Occlusion.
    ///                                false = wait_until_final_zpass_written
    ///                                true  = draw_if_not_final_zpass_written
    /// @param [in] accumulateData true(1) = allow_accumulation of Zpass and PrimCount across command buffer boundaries.
    ///
    /// pQueryPool and gpuVirtAddr should be exclusively set, when both are nullptr/0, other params will be ignored
    /// and it means to reset/disable predication so that the following commands can perform normally.
    virtual void CmdSetPredication(
        IQueryPool*         pQueryPool,
        uint32              slot,
        const IGpuMemory*   pGpuMemory,
        gpusize             offset,
        PredicateType       predType,
        bool                predPolarity,
        bool                waitResults,
        bool                accumulateData) = 0;

    /// Suspend/resume any active predication for this command buffer
    ///
    /// @param [in] suspend     Controls if predication should be paused
    ///                             true  = suspend active predication
    ///                             false = resume active predication
    ///
    /// Any suspended predication must be resumed prior to disabling predication using CmdSetPredication with pQueryPool
    /// and gpuVirtAddr with nullptr/0. This is only valid on universal and compute command buffers.
    virtual void CmdSuspendPredication(
        bool suspend) = 0;

    /// Begins a conditional block in the current command buffer. All commands between this and the corresponding
    /// CmdEndIf() (or CmdElse() if it is present) command are executed if the specified condition is true.
    ///
    /// This function requires use of the following barrier flags on @ref gpuMemory:
    /// - PipelineStage:  @ref PipelineStageFetchIndirectArgs
    /// - CacheCoherency: @ref CoherIndirectArgs
    ///
    /// @param [in] gpuMemory    GPU memory object containing the memory location to be tested.
    /// @param [in] offset       Offset within the memory object where the tested memory location begins.
    /// @param [in] data         Source data to compare against the value in GPU memory.
    /// @param [in] mask         Mask to apply to the GPU memory (via bitwise AND) prior to comparison.
    /// @param [in] compareFunc  Function controlling how the data operands are compared.
    virtual void CmdIf(
        const IGpuMemory& gpuMemory,
        gpusize           offset,
        uint64            data,
        uint64            mask,
        CompareFunc       compareFunc) = 0;

    /// Begins a conditional block in the current command buffer. All commands between this and the corresponding
    /// CmdEndIf() command are executed if the condition specified in the innermost active conditional block are false.
    virtual void CmdElse() = 0;

    /// Ends the innermost active conditional block in the current command buffer.
    virtual void CmdEndIf() = 0;

    /// Begins a while loop in the current command buffer. All commands between this and the corresponding CmdEndWhile()
    /// command are executed repeatedly as long as the specified condition remains true.
    ///
    /// This function requires use of the following barrier flags on @ref gpuMemory:
    /// - PipelineStage:  @ref PipelineStageFetchIndirectArgs
    /// - CacheCoherency: @ref CoherIndirectArgs
    ///
    /// @param [in] gpuMemory    GPU memory object containing the memory location to be tested.
    /// @param [in] offset       Offset within the memory object where the tested memory location begins.
    /// @param [in] data         Source data to compare against the value in GPU memory.
    /// @param [in] mask         Mask to apply to the GPU memory (via bitwise AND) prior to comparison.
    /// @param [in] compareFunc  Function controlling how the data operands are compared.
    virtual void CmdWhile(
        const IGpuMemory& gpuMemory,
        gpusize           offset,
        uint64            data,
        uint64            mask,
        CompareFunc       compareFunc) = 0;

    /// Ends the innermost active while loop in the current command buffer.
    virtual void CmdEndWhile() = 0;

    /// Stalls a command buffer execution based on a condition that compares an immediate value with value coming from a
    /// GPU register.
    ///
    /// The client (or application) is supposed to do necessary barriers before calling this function, but for now this
    /// is only need to wait some display or timer related registers.
    ///
    /// @param [in] registerOffset The offset in bytes of GPU register to be tested.
    /// @param [in] data           Source data to compare against the value of GPU register.
    /// @param [in] mask           Mask to apply to the GPU memory (via bitwise AND) prior to comparison.
    /// @param [in] compareFunc    Function controlling how the data operands are compared. CompareFunc::Never shouldn't
    ///                            be used as the hardware does not support it.
    virtual void CmdWaitRegisterValue(
        uint32      registerOffset,
        uint32      data,
        uint32      mask,
        CompareFunc compareFunc) = 0;

    /// Stalls a command buffer execution based on a condition that compares an immediate value with value coming from a
    /// GPU memory location.
    ///
    /// The client (or application) is expected to transiton the memory to proper state before calling this function.
    /// The memory location for the condition must be 4-byte aligned.
    /// This function requires use of the following barrier flags on @ref gpuVirtAddr:
    /// - PipelineStage:  @ref PipelineStagePostPrefetch
    /// - CacheCoherency: @ref CoherCp
    ///
    /// @param [in] gpuVirtAddr  GPU memory address containing the data to be tested.
    /// @param [in] data         Source data to compare against the value in GPU memory.
    /// @param [in] mask         Mask to apply to the GPU memory (via bitwise AND) prior to comparison.
    /// @param [in] compareFunc  Function controlling how the data operands are compared. CompareFunc::Never should not
    ///                          be used as the hardware does not support it.
    virtual void CmdWaitMemoryValue(
        gpusize     gpuVirtAddr,
        uint32      data,
        uint32      mask,
        CompareFunc compareFunc) = 0;

    /// Stalls a command buffer execution until an external device writes to the marker surface in the GPU bus
    /// addressable memory location.
    ///
    /// This function requires use of the following barrier flags on @ref gpuMemory:
    /// - PipelineStage:  @ref PipelineStagePostPrefetch
    /// - CacheCoherency: @ref CoherCp
    ///
    /// @param [in] gpuMemory    GPU memory object containing the memory location to be tested.
    /// @param [in] data         Source data to compare against the value in GPU memory.
    /// @param [in] mask         Mask to apply to the GPU memory (via bitwise AND) prior to comparison.
    /// @param [in] compareFunc  Function controlling how the data operands are compared. CompareFunc::Never should not
    ///                          be used as the hardware does not support it.
    virtual void CmdWaitBusAddressableMemoryMarker(
        const IGpuMemory& gpuMemory,
        uint32            data,
        uint32            mask,
        CompareFunc       compareFunc) = 0;

    /// Begins the specified performance experiment.
    ///
    /// @param [in] pPerfExperiment Performance experiment to begin.
    virtual void CmdBeginPerfExperiment(
        IPerfExperiment* pPerfExperiment) = 0;

    /// Updates the sqtt token mask on the specified performance experiment.
    ///
    /// @param [in] pPerfExperiment Performance experiment to update.
    /// @param [in] tokenConfig updated token and reg mask to apply.
    ///
    /// @note: This function is only valid to call if pPerfExperiment is a thread trace experiment that is currently
    //         active.
    virtual void CmdUpdatePerfExperimentSqttTokenMask(
        IPerfExperiment*              pPerfExperiment,
        const ThreadTraceTokenConfig& tokenConfig) = 0;

    /// Updates the sqtt token mask on all running traces, if any.
    ///
    /// @note This may overwrite the stall settings (making them more conservative)
    /// @param [in] tokenConfig updated token and reg mask to apply.
    virtual void CmdUpdateSqttTokenMask(
        const ThreadTraceTokenConfig& tokenConfig) = 0;

    /// Ends the specified performance experiment.
    ///
    /// @param [in] pPerfExperiment Performance experiment to end.
    virtual void CmdEndPerfExperiment(
        IPerfExperiment* pPerfExperiment) = 0;

    /// Inserts a trace marker into the command buffer.
    ///
    /// A trace marker can be inserted to mark particular points of interest in a command buffer to be viewed with the
    /// trace data collected in a performance experiment.
    ///
    /// @param [in] markerType Selects one of two generic marker categories ("A" or "B").
    /// @param [in] markerData 32-bit marker value to be inserted.
    virtual void CmdInsertTraceMarker(
        PerfTraceMarkerType markerType,
        uint32              markerData) = 0;

    /// Inserts a set of SQ thread trace markers for consumption by the Radeon GPU Profiler (RGP).
    ///
    /// Only supported on Universal and Compute engines.
    ///
    /// @param [in] numDwords Number of dwords in pData to be inserted as SQTT markers.
    /// @param [in] pData     SQTT marker data.  See the RGP SQTT Instrumentation Specification for details on how this
    ///                       data should be formatted.
    virtual void CmdInsertRgpTraceMarker(
        RgpMarkerSubQueueFlags subQueueFlags,
        uint32                 numDwords,
        const void*            pData) = 0;

    /// This function is to be used to copy the DF SPM (MALL SPM) data from the output buffers to an accessible buffer.
    /// The buffer that HW outputs to is allocated with a special KMD flag and therefore cannot be the same as the
    /// normal IPerfExperiment buffer so we need a special command to get the data.
    ///
    /// The bulk of the implementation for this is done by the KMD. They are in charge of starting and stopping the
    /// trace as well as all of the register programming. When KMD recieves a dfSpmTraceEnd bit from a CmdBufInfo
    /// flag, they will wait for the command buffer to be completely idle before stopping the trace. Therefore, a
    /// CmdEndPerfExperiment call does not stop this particular sample, the end of a command buffer with a
    /// dfSpmTraceEnd does. This means that calling CmdCopyDfSpmTraceData in the same command buffer as
    /// dfSpmTraceEnd will give you incorrect data. The sample will still be in progress when the copy happens.
    /// You must call CmdCopyDfSpmTraceData in a separate command buffer after one where the dfSpmTraceEnd bit is
    /// set.
    ///
    /// There is also a metadata buffer that does not need a special KMD flag. It is also stored in a separate buffer
    /// and is copied along with the output buffer with this command. It contains a uint32 trace size, a uint32 pad,
    /// a uint64 start trace GPU timestamp and a uint64 stop trace GPU timestamp and is placed at the beginning of the
    /// dstGpuMemory.
    ///
    /// The minimum size of the dstGpuMemory should be the size of the metadata struct plus the size of the DF SPM
    /// ringSize given to the perf experiment. The SPM data may not fill the entire memory, but the client is
    /// responsible for parsing the data.
    ///
    /// This function requires use of the following barrier flags on @ref dstGpuMemory:
    /// - PipelineStage:  @ref PipelineStageBlt
    /// - CacheCoherency: @ref CoherCopyDst
    ///
    /// @param [in] perfExperiment  The perfExperiment that we will be copying the data from
    /// @param [in] dstGpuMemory    The memory location that the DF SPM trace data will be copied to.
    /// @param [in] dstOffset       The offset into the destination memory that the data will be copied to.
    virtual void CmdCopyDfSpmTraceData(
        const IPerfExperiment& perfExperiment,
        const IGpuMemory&      dstGpuMemory,
        gpusize                dstOffset) = 0;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 914
    /// Loads data from the provided GPU Memory object into Constant Engine RAM.
    ///
    /// @param [in] srcGpuMemory  GPU Memory object containing the source data to be loaded to CE RAM.
    /// @param [in] memOffset     Offset within the memory object where the source data is located,
    ///                           must be 32-byte aligned.
    /// @param [in] ramOffset     Byte offset destination in CE RAM where the data should be loaded,
    ///                           must be 32-byte aligned.
    /// @param [in] dwordSize     Number of DWORDs that should be loaded into CE RAM, must be a multiple of 8.
    void CmdLoadCeRam(
        const IGpuMemory& srcGpuMemory,
        gpusize           memOffset,
        uint32            ramOffset,
        uint32            dwordSize) {}

    /// Dumps data from Constant Engine RAM to the provided GPU Memory address which may be located in a GPU ring buffer
    /// managed by the CE. The CE can be used to automatically handle the synchronization between the DE and CE when
    /// manipulating a GPU ring buffer. In order for PAL to instruct the CE to handle this, we need to know the current
    /// position (entry) within the ring buffer being dumped to, as well as the total size (in entries) of the ring.
    ///
    /// @param [in] dstGpuMemory  GPU Memory object destination where the data should be dumped from CE RAM.
    /// @param [in] memOffset     Offset within the memory object where data should be dumped, must be 4 byte aligned.
    /// @param [in] ramOffset     Byte offset source in CE RAM for data that should be dumped, must be 4 byte aligned.
    /// @param [in] dwordSize     Number of DWORDs that should be dumped from CE RAM into GPU Memory
    /// @param [in] currRingPos   Current position (ring entry) in the GPU ring buffer being managed by the CE which the
    ///                           dump location corresponds to.
    /// @param [in] ringSize      Number of entries in the GPU ring buffer being managed by the CE. If the memory being
    ///                           dumped into is not managed in a ring-like fashion, this should be set to zero.
    void CmdDumpCeRam(
        const IGpuMemory& dstGpuMemory,
        gpusize           memOffset,
        uint32            ramOffset,
        uint32            dwordSize,
        uint32            currRingPos,
        uint32            ringSize) {}

    /// Writes CPU data to Constant Engine RAM
    ///
    /// @param [in] pSrcData   Pointer to the source CPU data to be written to CE RAM.
    /// @param [in] ramOffset  Byte offset in CE RAM where the data should be written, must be 4 byte aligned.
    /// @param [in] dwordSize  Number of DWORDs that should be written from pSrcData into CE RAM.
    virtual void CmdWriteCeRam(
        const void* pSrcData,
        uint32      ramOffset,
        uint32      dwordSize) {}
#endif

    /// Allocates a chunk of command space that the client can use to embed constant data directly in the command
    /// buffer's backing memory. The returned CPU address is valid until ICmdBuffer::End() is called. The GPU address
    /// is valid until ICmdBuffer::Reset() or ICmdBuffer::Begin() and must only be referenced by work contained within
    /// this command buffer (e.g., as an SRD table address).
    ///
    /// @param [in]  sizeInDwords       Size of the embedded data space in DWORDs. It must be less than or equal to the
    ///                                 value reported by GetEmbeddedDataLimit().
    /// @param [in]  alignmentInDwords  Minimum GPU address alignment of the embedded space in DWORDs.
    /// @param [out] pGpuAddress        The GPU address of the embedded space.
    ///
    /// @returns The DWORD-aligned CPU address of the embedded space.
    virtual uint32* CmdAllocateEmbeddedData(
        uint32   sizeInDwords,
        uint32   alignmentInDwords,
        gpusize* pGpuAddress) = 0;

    /// Allocates a chunk of command space that the client can use to embed constant data directly in the command
    /// buffer's backing memory. The returned CPU address is valid until ICmdBuffer::End() is called. The GPU address
    /// is valid until ICmdBuffer::Reset() or ICmdBuffer::Begin() and must only be referenced by work contained within
    /// this command buffer (e.g., as an SRD table address).
    ///
    /// @param [in]  sizeInDwords       Size of the embedded data space in DWORDs. It must be less than or equal to the
    ///                                 value reported by GetLargeEmbeddedDataLimit().
    /// @param [in]  alignmentInDwords  Minimum GPU address alignment of the embedded space in DWORDs.
    /// @param [out] pGpuAddress        The GPU address of the embedded space.
    ///
    /// @returns The DWORD-aligned CPU address of the embedded space.
    virtual uint32* CmdAllocateLargeEmbeddedData(
        uint32   sizeInDwords,
        uint32   alignmentInDwords,
        gpusize* pGpuAddress) = 0;

    /// Get memory from scratch memory and bind to GPU event. For now only GpuEventPool and CmdBuffer's internal
    /// GpuEvent use this path to allocate and bind GPU memory. These usecases assume the bound GPU memory is GPU access
    /// only, so client is responsible for resetting the event from GPU, and cannot call Set(), Reset(), GetStatus().
    ///
    /// @param [in]  pGpuEvent  The GPU event that needs to bind a memory. Must not be nullptr.
    ///
    /// @returns Success if the GPU event successfully binds a GPU memory.  Otherwise, one of the following errors may
    ///          be returned:
    ///          + ErrorUnknown if an internal PAL error occurs.
    virtual Result AllocateAndBindGpuMemToEvent(
        IGpuEvent* pGpuEvent) = 0;

    /// Issues commands to prime GPU caches shortly before accessing the specified GPU address range(s).  The benefit of
    /// this prefetching is likely to be platform-dependent based on the GPU's cache hierarchy, memory subsystem,
    /// available prefetching tools in hardware, etc., so caller beware.
    ///
    /// This operation may read data from memory into caches and therefore counts as a general BLT SRC operation with
    /// regard to barrier execution and memory dependencies.
    ///
    /// @param [in] rangeCount  Number of entries in pRanges.
    /// @param [in] pRanges     Array of structs defining a memory range and properties controlling prefetching of that
    ///                         range.
    virtual void CmdPrimeGpuCaches(
        uint32                    rangeCount,
        const PrimeGpuCacheRange* pRanges) = 0;

    /// Issues commands which execute the specified group of nested command buffers.  The observable behavior of this
    /// operation should be indiscernible from directly recording the nested command buffers' commands directly into
    /// this command buffer.  Naturally, the queue type of the nested command buffers must match this command buffer.
    ///
    /// Conceptually, executing a nested command buffer is similar to calling a subroutine: the root command buffer is
    /// like the "caller", while the nested ones are the "callees".
    ///
    /// If any nested command buffers were allocated from a @ref ICmdAllocator with @ref autoMemoryReuse enabled,
    /// resetting or destroying those nested command buffers will render them retroactively uncallable. This effectively
    /// makes the caller command buffer invalid and illegal to submit even if it was otherwise valid and executable in
    /// the past. If the nested command allocator has autoMemoryReuse disabled, the calls to reset nested command
    /// buffers remain valid until the allocator itself is reset.
    ///
    /// State inheritance/leakage between the caller and callee(s) has the following behavior:
    /// + The callee only inherits the state specified in the callee CmdBufferBuildInfo.  It is up to the client to
    ///   bind any default state necessary when they called @ref ICmdBuffer::Begin() to begin building the callee.
    ///   By default no state is inherited and all state must be specified by the client.
    /// + The callee leaks any render and resource-binding state back into the caller after it completes.  It is up to
    ///   the client to rebind the caller's state after this operation completes if they don't want state leakage.
    /// + Both of the above points apply in between callees, if more than one command buffer is being executed by this
    ///   call.
    ///
    /// @param [in]     cmdBufferCount  Number of nested command buffers to execute.  (i.e., size of the ppCmdBuffers
    ///                                 array).  This must be at least one, otherwise making this call is pointless.
    /// @param [in,out] ppCmdBuffers    Array of nested command buffers to execute.  It is an error condition if any
    ///                                 of the following are true: (Debug assertions are used to check them.)
    ///                                 + ppCmdBuffers is null.
    ///                                 + Any member of ppCmdBuffers is null.
    ///                                 + Any member of ppCmdBuffers is a root command buffer, or has a different
    ///                                   queue type than this command buffer.
    virtual void CmdExecuteNestedCmdBuffers(
        uint32            cmdBufferCount,
        ICmdBuffer*const* ppCmdBuffers) = 0;

    /// Saves a copy of some set of the current command buffer state that is used by compute workloads. This feature is
    /// intended to give PAL clients a convenient way to issue their own internal compute workloads without modifying
    /// the application-facing state.
    ///
    /// PAL cannot save multiple layers of state, each call to CmdSaveComputeState must be followed by a call to
    /// CmdRestoreComputeState before the next call to CmdSaveComputeState.
    ///
    /// This function can only be called on command buffers that support compute workloads. All query counters will be
    /// disabled until CmdRestoreComputeState is called.
    ///
    /// @param [in] stateFlags  A mask of ORed @ref ComputeStateFlags indicating which state to save.
    virtual void CmdSaveComputeState(
        uint32 stateFlags) = 0;

    /// Restores some set of the command buffer state that is used by compute workloads. This feature is intended to
    /// give PAL clients a convenient way to issue their own internal compute workloads without modifying the
    /// application-facing state.
    ///
    /// A call to this function must be preceded by a call to CmdSaveComputeState and the save stateFlags must contain
    /// all restore stateFlags, otherwise the values of the restored state are undefined.
    ///
    /// This function can only be called on command buffers that support compute workloads. All previously disabled
    /// query counters will be reactivated.
    ///
    /// @param [in] stateFlags  A mask of ORed @ref ComputeStateFlags indicating which state to restore.
    virtual void CmdRestoreComputeState(
        uint32 stateFlags) = 0;

    /// Issues commands which complete two tasks: using the provided @ref IIndirectCmdGenerator object to translate the
    /// indirect argument buffer into a format understandable by the GPU; and then executing the generated commands.
    ///
    /// The virtual address must be 4-byte aligned.
    ///
    /// The indirect argument data offset in memory must be 4-byte aligned. The expected layout of the argument data
    /// is defined by the @ref IIndirectCmdGenerator object.
    ///
    /// It is unsafe to call this method on a command buffer which was not begun with either the optimizeOneTimeSubmit
    /// or optimizeExclusiveSubmit flags. This is because there is a potential race condition if the same command buffer
    /// is generating indirect commands on multiple Queues simultaneously.
    ///
    /// This function requires use of the following barrier flags on the indirect memory:
    /// - PipelineStage:  @ref PipelineStageFetchIndirectArgs
    /// - CacheCoherency: @ref CoherIndirectArgs
    ///
    /// @param [in] generator     Indirect command generator object which can translate the indirect argument buffer
    ///                           into a command buffer format which the GPU can understand.
    /// @param [in] gpuVirtAddr   Gpu virtual address where the indirect argument data is located.
    /// @param [in] maximumCount  Maximum count of data structures to loop through.  If countGpuAddr is nonzero, the
    ///                           value at that memory location is clamped to this maximum. If countGpuAddr is zero,
    ///                           Then the number of draws issued exactly matches this number.
    /// @param [in] countGpuAddr  GPU virtual address where the number of draws is stored.  Must be 4-byte aligned.
    virtual void CmdExecuteIndirectCmds(
        const IIndirectCmdGenerator& generator,
        gpusize                      gpuVirtAddr,
        uint32                       maximumCount,
        gpusize                      countGpuAddr) = 0;

    /// Updates one or more HiS pretests bound to the given stencil image within a range of mip levels.
    /// See @ref HiSPretests for a summary of HiS.
    ///
    /// @warning Improper use of pretests can cause corruption.  Please see @ref HiSPretests for more information.
    ///
    /// @param [in] image    The stencil image that will receive the new pretest(s).
    /// @param [in] pretests The new pretest(s).
    /// @param [in] firstMip The beginning of the mip range which will receive the new pretest(s).
    /// @param [in] numMips  The number of mips in the mip range which will receive the new pretest(s).
    virtual void CmdUpdateHiSPretests(
        const IImage*      pImage,
        const HiSPretests& pretests,
        uint32             firstMip,
        uint32             numMips) = 0;

    /// Reserve @ref CommandDataAlloc space for external command packets up to a size of @ref sizeInDwords.
    /// This method is only supported on command buffers for the following queue types:
    ///
    /// @warning @ref CmdCommitSpace must be called once after this function is called.
    //           Failing to pair up these function calls will result in undefined behavior.
    ///
    /// @param [in] sizeInDwords       Size of the command buffer space to reserve in dwords.
    ///                                If this param is 0, the default command stream reserve limit will be used.
    /// @param [in] reserveInNewChunk  Selection to reserve space in a new chunk or current chunk.
    ///
    /// @returns A pointer to the reserved command space.
    virtual uint32* CmdReserveSpace(
        uint32 sizeInDwords,
        bool reserveInNewChunk) = 0;

    /// Ensure data is commited the command buffer and unused space is reclaimed.
    /// This method is only supported on command buffers for the following queue types:
    ///
    /// @param [in] pCmdSpace  Pointer to the next unused dword in the command buffer.
    virtual void CmdCommitSpace(
        uint32* pCmdSpace) = 0;

    /// Executes any internal postprocessing commands to be performed on a frame, such as drawing the dev driver
    /// overlay.  Calling this prior to presenting (via any path) is a requirement, and must be prior to or
    /// concurrent with frameEnd if FSFM is applicable.  This must be called using the image that will be the
    /// source of the present.
    ///
    /// @param [in]  postProcessInfo  Information about the frame to be postprocessed.
    /// @param [out] pAddedGpuWork    (Optional) Set to true if commands were added as part of this call.
    virtual void CmdPostProcessFrame(
        const CmdPostProcessFrameInfo& postProcessInfo,
        bool*                          pAddedGpuWork) = 0;

    /// Inserts a string embedded inside a NOP packet with a signature that is recognized by tools and can be printed
    /// inside a command buffer disassembly. Note that this is a real NOP that will really be submitted to the GPU
    /// and executed (skipped over) by CP. It will be visible in kernel debugging as well as offline debug dumps.
    ///
    /// The maximum length of a string that may be embedded in the command buffer is currently 128 characters,
    /// including the NUL-terminator. This is defined in the internal command buffer class in MaxCommentStringLength.
    ///
    /// @param [in] pComment        Pointer to NUL-terminated string that will be inserted into the command buffer.
    virtual void CmdCommentString(
        const char* pComment) = 0;

    /// Inserts the specified payload embedded inside a NOP packet. Note that this is a real NOP that will be submitted
    /// to the GPU and executed (skipped over) by CP. It will be visible in kernel debugging as well as offline debug
    /// dumps.
    ///
    /// @param [in] pPayload    Pointer to binary data to embed.
    /// @param [in] payloadSize Size of the payload in DWORDs, expected to be under MaxPayloadSize.
    virtual void CmdNop(
        const void* pPayload,
        uint32      payloadSize) = 0;

    /// Marks the begin or end of a user-defined region of GPU work; analyzed post-mortem in crash-dump analysis tools.
    /// Each 'Begin' marker must be paired with a corresponding 'End' marker; however, markers may be nested by
    /// inserting multiple 'Begin' markers consecutively.
    ///
    /// @warning This function is a no-op if Crash Analysis mode is not enabled.
    ///
    /// @param [in] isBegin         Whether this is a 'Begin' marker (true) or an 'End' marker (false).
    /// @param [in] sourceId        The application layer ID at which the marker is being created:
    ///                               0x0 => Application
    ///                               0x1 => API (e.g. DX12, Vulkan, etc.)
    ///                               0x2 => PAL
    ///                             Developers may use IDs within the range of 10 - 15 to define a custom
    ///                             application layer.
    /// @param [in] pMarkerName     A NULL-terminated string containing a name for this marker, used for annotation
    ///                             purposes in external tools. Only valid for 'Begin' markers, and will be ignored if
    ///                             isBeginMarker is false.
    /// @param [in] markerNameSize  Size of the marker string, in bytes.
    ///
    /// @returns Non-zero counter value of the embedded execution marker.
    ///          If Crash Analysis mode is disabled, this will always return zero.
    virtual uint32 CmdInsertExecutionMarker(
        bool         isBegin,
        uint8        sourceId,
        const char*  pMarkerName,
        uint32       markerNameSize) = 0;

    /// Performs the virtual queue handshake. The host queue will do the following:
    ///  - Wait until the parent kernel is done
    ///  - Change the parent kernel state
    ///  - CP waits until the child counter is 0
    ///  - CP sends the termination signal to the device queue
    /// NOTE: Available for compute queues when created with aqlQueue set in the QueueCreateInfo.
    ///
    /// @param [in] parentState         Address of the parent kernel state
    /// @param [in] newStateValue       The new state value of the parent kernel
    /// @param [in] parentChildCounter  Address of the parent child counter
    /// @param [in] signal              Address of the virtual queue signal
    /// @param [in] dedicatedQueue      Runtime uses a dedicated queue for the scheduler
    ///
    /// @note This function is to support OpenCL AQL submissions.
    virtual void  CmdVirtualQueueHandshake(
        gpusize parentState,
        uint32  newStateValue,
        gpusize parentChildCounter,
        gpusize signal,
        bool    dedicatedQueue) = 0;

    /// Returns GPU address of the loop start with dispatch templates. The pointer will be passed to the
    /// scheduler kernel for the update of dispatch templates.
    /// Also initializes common registers for each dispatch template.
    /// NOTE: Available for compute queues when created with aqlQueue set in the QueueCreateInfo.
    ///
    /// @note This function is to support OpenCL AQL submissions.
    virtual gpusize CmdVirtualQueueDispatcherStart() = 0;

    /// Programs CP iterator with dispatch templates for device enqueue in OpenCL2.0
    /// NOTE: Available for compute queues when created with aqlQueue set in the QueueCreateInfo.
    ///
    /// @param [in] signal          Address for the termination signal
    /// @param [in] loopStart       GPU address of the loop start for CP
    /// @param [in] numTemplates    The number of dispatch templates
    ///
    /// @note This function is to support OpenCL AQL submissions.
    virtual void CmdVirtualQueueDispatcherEnd(
        gpusize signal,
        gpusize loopStart,
        uint32  numTemplates) = 0;

    /// Emulates AQL dispatch with PM4 commands.
    /// NOTE: Available for compute queues when created with aqlQueue set in the QueueCreateInfo.
    ///
    /// @param [in] dispatchInfo    Pointer to kernel dispatch info
    ///
    /// @note This function is to support OpenCL AQL submissions.
    void CmdDispatchAql(
        const DispatchAqlParams& dispatchInfo)
    {
        m_funcTable.pfnCmdDispatchAql(this, dispatchInfo);
    }

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 888
    /// XDMA was retired starting in gfx10 so this function has no use anymore.
    inline void CmdXdmaWaitFlipPending() {}
#endif

    /// Starts thread-trace/counter-collection - used by GPS Shim's OpenShimInterface via DXCP
    /// Only valid for the GPU Profiler layer (which is enabled separately by the GPS Shim during usage of these
    /// functions)
    /// Only valid for per-draw granularity and hence non-RGP thread-trace formats.
    /// The caller is responsible for setting up valid GPU Profiler panel settings.
    virtual void CmdStartGpuProfilerLogging() = 0;

    /// Stops thread-trace/counter-collection - used by GPS Shim's OpenShimInterface via DXCP
    /// Only valid for the GPU Profiler layer (which is enabled separately by the GPS Shim during usage of these
    /// functions)
    /// Only valid for per-draw granularity and hence non-RGP thread-trace formats.
    /// The caller is responsible for setting up valid GPU Profiler panel settings.
    virtual void CmdStopGpuProfilerLogging() = 0;

    /// Set a mask to control which view instances are enabled for subsequent draws, should only be called on
    /// universal command buffers.
    ///
    /// @param [in] mask     The mask to control which view instances are enabled.
    virtual void CmdSetViewInstanceMask(uint32 mask) = 0;

    /// Get used size of all chunks in bytes for given CmdAllocType. For CommandDataAlloc with multi-queue scheme, the
    /// size reported will be the sum of all command streams associated with the command buffer. It's legal to call
    /// this function while in the command building state.
    ///
    /// @param [in] type    Allocation type for ICmdAllocator
    ///
    /// @returns Used allocation data size in bytes for provided CmdAllocType.
    virtual uint32 GetUsedSize(
        CmdAllocType type) const = 0;

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

    /// Gets the internal unique Id of the command buffer.
    /// This function was originally only for internal debugging, and the Id is not unique across different queue types.
    /// However, CmdDisassembly::ICmdBufferReporting requires a way to differentiate cmdLists that matches
    /// the meaning of UniqueId(), in particular, not requiring uniqueness across different queue types.
    ///
    /// @returns Unique Id of the command buffer
    virtual uint32 UniqueId() const = 0;

    /// Get the number of arrays of tracking data (for correlation) held by this ICmdBuffer.
    /// For TrackCmdLocationBefore and TrackCmdLocationAfter below, only (idx < GetNumTrackingArrays())
    /// can yield a non-nullptr result
    ///
    /// @detail If GetNumTrackingArrays() returns 0 but GetTrackedCmdLocationArraySizeInBytes() does not, it will
    /// be possible to create them on this ICmdBuffer using CreateTrackedCmdLocationArray
    ///
    /// @returns the number of a TrackedCmdLocationArray's held by the ICmdBuffer.
    virtual uint32 GetNumTrackingArrays() const = 0;

    /// Get the number of bytes required by CreateTrackedCmdLocationArray.
    ///
    /// @detail The value returned here accomdates the full number of TrackedCmdLocationArray's to be
    ///         created, from a single contiguous allocation.
    ///         If allocation has not yet occured, (GetNumTrackingArrays() == 0).
    ///         If (GetTrackedCmdLocationArraySizeInBytes() > 0) &&  (GetNumTrackingArrays() == 0)
    ///         this ICmdBuffer supports TrackedCmdLocationArray's, but has not yet allocated them
    ///         If (GetTrackedCmdLocationArraySizeInBytes() == 0), this ICmdBuffer does not support
    ///         TrackedCmdLocationArray's
    ///
    /// @returns 0 if TrackedCmdLocationArray's are not supported
    ///         The total number of bytes required requied by CreateTrackedCmdLocationArray otherwise.
    virtual uint32 GetTrackedCmdLocationArraySizeInBytes() const = 0;

    /// Uses the memory pMemory to initialize GetNumTrackingArrays() TrackedCmdLocationArray's on this
    /// ICmdBuffer.
    ///
    /// @param  [in] pMemory    Address of memory allocated for the purpose of creating TrackedCmdLocationArray's
    ///                         This memory should be at larger than GetTrackedCmdLocationArraySizeInBytes()
    ///                         bytes.
    ///
    /// @returns    Result::Success:                Indicates creation was successful
    ///             Result::Unsupported:            This ICmdBuffer does not support TrackedCmdLocationArray's
    ///                 Confirm (GetTrackedCmdLocationArraySizeInBytes() > 0) before using this function
    ///             Result::ErrorInvalidPointer:    pMemory == nullptr
    ///             Result::AlreadyExists:          TrackedCmdLocationArray's have already been allocated on
    ///                 this ICmdBuffer
    ///             other:                          Error values originating from Util::Vector::Reserve() or
    ///                 Util::Vector::PushBack()
    ///
    virtual Result CreateTrackedCmdLocationArray(
        void* pMemory) = 0;

    /// Executes the destructors for all TrackedCmdLocationArray's owned by this ICmdBuffer. This should be
    /// called prior to deleting the memory pMemory that was originally provided to CreateTrackedCmdLocationArray
    ///
    /// @param  [out] ppAllocatedMemory
    ///             If (ppAllocatedMemory != nullptr) && (GetNumTrackingArrays() > 0)
    ///                 The original value for pMemory provided in CreateTrackedCmdLocationArray(pMemory) will
    ///                 by returned in *ppAllocatedMemory. ie *ppAllocatedMemory = pMemory
    ///             If (ppAllocatedMemory != nullptr)  && (GetNumTrackingArrays() == 0),
    ///                 *ppAllocatedMemory  = nullptr;
    ///
    virtual void DestroyTrackedCmdLocationArray(
        void** ppAllocatedMemory) = 0;

    /// Creates a new TrackedCmdLocation in the TrackedCmdLocationArray corresponding to idx
    /// of type TrackedCmdLocationMode::Begin. This location will be furnished with a pointer to the address of
    /// the next PM4Packet to be created on the CmdBuffer referred to by idx, and m_event == eventId.
    ///
    /// @detail Note there is two potential valid corner cases.
    ///         (pBeforeResult->Get()->m_correlateInternal.m_ptr. == 0)
    ///             Indicates there is a commandstream but it has not begun building PM4Packets
    ///             This implicitly refers to the baseAddress of the commandstream, o0nce building begins
    ///         (pBeforeResult->Get()->m_correlateInternal.m_ptr == TrackedCmdLocation::NoCorrespondingBaseAddress)
    ///             Indicates there is not yet an associated commandstream.
    ///
    /// @param [in] idx             The idx corresponding to GetTrackingArray(idx).
    ///                                 Only (idx < GetNumTrackingArrays()) will yield non-trivial results.
    /// @param [in] eventId         The code for the event being tracked
    /// @param [out] pBeforeResult  If successful, returns a TrackedCmdLocationRef to a TrackedCmdLocation within
    ///                                 the TrackedCmdLocationArray corresponding to idx
    ///                                 Otherwise, if (pBeforeResult != nullptr), is initialized to the default for
    ///                                 TrackedCmdLocationRef()
    ///
    /// @returns
    ///     Pal::Result::Success                if successful
    ///     Pal::Result::ErrorInvalidPointer    if (pBeforeResult == nullptr)
    ///     Pal::Result::Unsupported            if not supported by this implementation of palCmdBuffer
    ///     Pal::Result::ErrorInvalidValue      if (idx >= NumCmdStreams())
    ///     Pal::Result::NotFound               if there is no tracking array corresponding to idx
    ///                                             This can occur if CreateTrackedCmdLocationArray has not been
    ///                                             called - which may mean the feature is disabled
    ///     Pal::Result ErrorOutOfMemory         if the TrackedCmdLocationArray corresponding to idx is unable
    ///                                             to allocate memory
    ///
    ///
    virtual Pal::Result TrackCmdLocationBefore(
        uint32 idx,
        uint8 eventId,
        CmdDisassembly::TrackedCmdLocationRef* pBeforeResult) = 0;

    /// Similar to TrackCmdLocationBefore, TrackCmdLocationAfter creates a TrackedCmdLocation, of type
    /// TrackedCmdLocationMode::End or TrackedCmdLocationMode::Delta. This location will be furnished with a
    /// pointer to the address of the next PM4Packet to be created on the CmdBuffer referred to by idx. Type
    /// TrackedCmdLocationMode::Delta will only occur if parameter before is the last TrackedCmdLocation
    /// for the TrackedCmdLocationArray corresponding to idx, has the same m_event == eventId, and the change in
    /// pointer address is small enough to be represented in 6 bits.
    ///
    /// @detail Note there is two potential valid corner cases.
    ///         (pAfterResult->Get()->m_correlateInternal.m_ptr. == 0)
    ///             Indicates there is a commandstream but it has not begun building PM4Packets
    ///             This implicitly refers to the baseAddress of the commandstream, o0nce building begins
    ///         (pAfterResult->Get()->m_correlateInternal.m_ptr == TrackedCmdLocation::NoCorrespondingBaseAddress)
    ///             Indicates there is not yet an associated commandstream.
    ///         In both of these cases, in parameter "before" had the same value for m_correlateInternal.m_ptr.
    ///         and "before" referred to the most recent TrackedCmdLocation, this tracked location will be of
    ///         type TrackedCmdLocationMode::Delta, with (before.Get()->m_correlateInternal.m_deltaInDWords == 0)
    ///
    /// @param [in] idx             The idx corresponding to GetTrackingArray(idx).
    ///                                 Only (idx < GetNumTrackingArrays()) can yield non-trivial results.
    /// @param [in] eventId         The code for the event being tracked
    /// @param [in] before          The corresponding location generated by TrackCmdLocationBefore
    ///                                 This may be CmdDisassembly::TrackedCmdLocationRef() if no location from
    ///                                 TrackCmdLocationBefore before exists (such as on Reset)
    /// @param [out] pAfterResult   Returns a TrackedCmdLocationRef to a TrackedCmdLocation within the TrackedCmdLocationArray
    ///                                 corresponding to idx
    ///
    /// @returns
    ///     Pal::Result::Success                if successful
    ///     Pal::Result::ErrorInvalidPointer    if (pAfterResult == nullptr)
    ///     Pal::Result::Unsupported            if not supported by this implementation of palCmdBuffer
    ///     Pal::Result::ErrorInvalidValue      if (idx >= NumCmdStreams())
    ///     Pal::Result::ErrorInvalidValue      if (eventId != before.m_correlateInternal.m_event)
    ///     Pal::Result::NotFound               if there is no tracking array corresponding to idx
    ///                                             This can occur if CreateTrackedCmdLocationArray has not been
    ///                                             called - which may mean the feature is disabled
    ///     Pal::Result ErrorOutOfMemory         if the TrackedCmdLocationArray corresponding to idx is unable
    ///                                             to allocate memory
    ///

    virtual Pal::Result TrackCmdLocationAfter(
        uint32 idx,
        uint8 eventId,
        CmdDisassembly::TrackedCmdLocationRef before,
        CmdDisassembly::TrackedCmdLocationRef* pAfterResult) = 0;

    /// An accessor function for the TrackedCmdLocationArray corresponding to idx
    ///
    /// @param idx  There is a CmdDisassembly::TrackedCmdLocationArray* corresponding to each
    ///             sub-cmdBuffer for this cmdBuffer. This idx indexes these in the same fashion.
    ///             Only idx < GetNumTrackingArrays() can yield non-nullptr results.
    ///
    /// @returns the TrackedCmdLocationArray corresponding to idx
    virtual CmdDisassembly::TrackedCmdLocationArray* GetTrackingArray(
        uint32 idx) const = 0;

    /// TrackClientEvent operates similarly to TrackCmdLocationBefore and TrackCmdLocationAfter above but
    /// operates on all TrackedCmdLocationArray's on this CmdBuffer. The TrackedCmdLocation generated here
    /// corresponds to type TrackedCmdLocationMode::ClientEvent, where clientId is a value the client is using
    /// to track this cmdBuffer, and eventId refers to some event the client is tracking outside of driver.
    ///
    /// @detail The first call to TrackClientEvent will include a TrackedCmdLocation with
    /// (m_mode == TrackedCmdLocationMode::ClientId), to define clientId on this cmdBuffer
    /// All calls will generate the TrackedCmdLocationMode::ClientEventId with
    ///     (m_clientEvent.m_clientEventId == clientEventId)
    /// and be followed with a TrackedCmdLocationMode::Delta location with (m_eventId == PostClientEvent (0xFF))
    ///
    /// @param [in] clientId
    /// @param [in] eventId
    ///
    /// @returns
    ///     Result::Success         if the tracked client event was successfully recorded
    ///     Result::Unsupported     if the implementation of ICmdBuffer does not support tracking
    ///     Result::ErrorInvalidPointer if there was an error encountered determining the cmdList correlation
    ///                             requested. This is likely to be an out-of-memory situation.
    ///     Result::AlreadyExists   if registering clientId occured multiple times. This should only occur for
    ///                             race conditions, if the code calling TrackClientEvent is not threadsafe
    virtual Result TrackClientEvent(
        uint64 clientId,
        uint64 clientEventId) = 0;

protected:
    /// @internal Constructor. Prevent use of new operator on this interface. Client must create objects by explicitly
    /// called the proper create method.
    ICmdBuffer() : m_pClientData(nullptr)
    {
    }

    /// @internal Destructor.  Prevent use of delete operator on this interface.  Client must destroy objects by
    /// explicitly calling IDestroyable::Destroy() and is responsible for freeing the system memory allocated for the
    /// object on their own.
    virtual ~ICmdBuffer() { }

    /// Structure for function pointers for the ICmdBuffer::Cmd* functions.
    struct CmdBufferFnTable
    {
        /// CmdSetUserData function pointers for each pipeline bind point.
        CmdSetUserDataFunc              pfnCmdSetUserData[static_cast<uint32>(PipelineBindPoint::Count)];

        CmdDrawFunc                      pfnCmdDraw;                      ///< CmdDraw function pointer.
        CmdDrawOpaqueFunc                pfnCmdDrawOpaque;                ///< CmdDrawOpaque function pointer.
        CmdDrawIndexedFunc               pfnCmdDrawIndexed;               ///< CmdDrawIndexed function pointer.
        CmdDrawIndirectMultiFunc         pfnCmdDrawIndirectMulti;         ///< CmdDrawIndirectMulti function pointer.
        CmdDrawIndexedIndirectMultiFunc  pfnCmdDrawIndexedIndirectMulti;  ///< CmdDrawIndexedIndirectMulti func pointer.
        CmdDispatchFunc                  pfnCmdDispatch;                  ///< CmdDispatch function pointer.
        CmdDispatchIndirectFunc          pfnCmdDispatchIndirect;          ///< CmdDispatchIndirect function pointer.
        CmdDispatchOffsetFunc            pfnCmdDispatchOffset;            ///< CmdDispatchOffset function pointer.
        CmdDispatchMeshFunc              pfnCmdDispatchMesh;              ///< CmdDispatchmesh function pointer.
        CmdDispatchMeshIndirectMultiFunc pfnCmdDispatchMeshIndirectMulti; ///< CmdDispatchMeshIndirect function pointer.
        CmdDispatchAqlFunc               pfnCmdDispatchAql;                ///< CmdDispatchAql function pointer.
    } m_funcTable;     ///< Function pointer table for Cmd* functions.

private:
    /// @internal Client data pointer. This can have an arbitrary value and can be returned by calling GetClientData()
    /// and set via SetClientData().
    /// For non-top-layer objects, this will point to the layer above the current object.
    void* m_pClientData;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 900
    /// @internal Some back-compat glue for some of the HwPipePoint interfaces in this file.
    static constexpr uint32 HwPipePointToStage[] =
    {
            PipelineStageTopOfPipe,    // HwPipeTop              = 0x0
            PipelineStagePostPrefetch, // HwPipePostPrefetch     = 0x1
            PipelineStageVs,           // HwPipePreRasterization = 0x2
            PipelineStagePs,           // HwPipePostPs           = 0x3
            PipelineStageLateDsTarget, // HwPipePreColorTarget   = 0x4
            PipelineStageCs,           // HwPipePostCs           = 0x5
            PipelineStageBlt,          // HwPipePostBlt          = 0x6
            PipelineStageBottomOfPipe, // HwPipeBottom           = 0x7
    };
#endif
};

} // Pal
