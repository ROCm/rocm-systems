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
 * @file  palPipeline.h
 * @brief Defines the Platform Abstraction Library (PAL) IPipeline interface and related types.
 ***********************************************************************************************************************
 */

#pragma once

#include "pal.h"
#include "palGpuMemoryBindable.h"
#include "palDestroyable.h"
#include "palImage.h"
#include "palShaderLibrary.h"
#include "palSpan.h"
#include <utility>

namespace Util
{
namespace Abi
{
union ApiHwShaderMapping;
enum class HardwareStage : uint32;
}

namespace HsaAbi
{
struct KernelArgument;
}
}

namespace Pal
{
struct GpuMemSubAllocInfo;
enum class PrimitiveTopology : uint8;

/// Specifies a shader type (i.e., what stage of the pipeline this shader was written for).
enum class ShaderType : uint32
{
    Compute = 0,
    Task,
    Vertex,
    Hull,
    Domain,
    Geometry,
    Mesh,
    Pixel,

    Count
};

/// Number of shader program types supported by PAL.
constexpr uint32 NumShaderTypes = static_cast<uint32>(ShaderType::Count);

/// Maximum number of viewports.
constexpr uint32 MaxViewports = 16;

/// Maximum number of supported stream-output declaration entries by any PAL device.
constexpr uint32 MaxStreamOutEntries = 512;

/// Specifies a general primitive category without differentiating between a strip or list and without specifying
/// whether a the primitive will include adjacency info or not.
enum class PrimitiveType : uint32
{
    Point    = 0x0,
    Line     = 0x1,
    Triangle = 0x2,
    Rect     = 0x3,
    Quad     = 0x4,
    Patch    = 0x5,
    Count
};

/// Specifies the target range of Z values after viewport transform.
enum class DepthRange : uint32
{
    ZeroToOne        = 0x0,
    NegativeOneToOne = 0x1,
};

/// Specifies whether the v/t texture coordinates of a point sprite map 0 to 1 from top to bottom or bottom to top.
enum class PointOrigin : uint32
{
    UpperLeft = 0x0,
    LowerLeft = 0x1,
    Count
};

/// Specifies primitive's shade mode.
enum class ShadeMode : uint32
{
    Gouraud = 0x0,      ///< Gouraud shading mode, pixel shader input is interpolation of vertex
    Flat    = 0x1,      ///< Flat shading mode, pixel shader input from provoking vertex
    Count
};

/// Defines a logical operation applied between the color coming from the pixel shader and the current value in the
/// target image.
enum class LogicOp : uint32
{
    Copy         = 0x0,
    Clear        = 0x1,
    And          = 0x2,
    AndReverse   = 0x3,
    AndInverted  = 0x4,
    Noop         = 0x5,
    Xor          = 0x6,
    Or           = 0x7,
    Nor          = 0x8,
    Equiv        = 0x9,
    Invert       = 0xA,
    OrReverse    = 0xB,
    CopyInverted = 0xC,
    OrInverted   = 0xD,
    Nand         = 0xE,
    Set          = 0xF,
};

/// Shader Engine Dispatch Interleave Size
///
/// This determines how many Threads or Threadgroups are sent to one SE before switching to the next SE.
/// Work is always distributed in Threadgroups though.
///
/// The 1D values are specified in Threads and the Threadgroups are walked in a 1D typewriter fashion.
/// The 2D values are specified in Threadgroups and also walked in typewriter fashion (in groups of the 2D pattern).
///
/// Clients should check for 1D and 2D support separately in:
///   - DeviceProperties::gfxipProperties::flags::support1dDispatchInterleave
///   - DeviceProperties::gfxipProperties::flags::support2dDispatchInterleave
///
/// Default will result in "Disable" for chips which do not support 1D or 2D.
/// Disable means that every Threadgroup is issued to the next SE.
enum class DispatchInterleaveSize : uint32
{
    Default,
    Disable,

    _1D_64_Threads,
    _1D_128_Threads,
    _1D_256_Threads,
    _1D_512_Threads,

    _2D_1x1_ThreadGroups,
    _2D_1x2_ThreadGroups,
    _2D_1x4_ThreadGroups,
    _2D_1x8_ThreadGroups,
    _2D_1x16_ThreadGroups,

    _2D_2x1_ThreadGroups,
    _2D_2x2_ThreadGroups,
    _2D_2x4_ThreadGroups,
    _2D_2x8_ThreadGroups,

    _2D_4x1_ThreadGroups,
    _2D_4x2_ThreadGroups,
    _2D_4x4_ThreadGroups,

    _2D_8x1_ThreadGroups,
    _2D_8x2_ThreadGroups,

    _2D_16x1_ThreadGroups,

    Count,
};

/// Specifies whether to override binning setting for pipeline. Enum value of Default follows the PBB global setting.
/// Enable or Disable value overrides PBB global setting for the pipeline and sets binning accordingly.
enum class BinningOverride : uint32
{
    Default = 0x0,
    Disable = 0x1,
    Enable  = 0x2,
    Count
};

/// GPU behavior is controlled by LDS_GROUP_SIZE.
enum class LdsPsGroupSizeOverride : uint32
{
    Default     = 0x0,
    SingleWave  = 0x1,
    DoubleWaves = 0x2
};

/// Tri-state enum which controls enabling or disabling a feature or behavior, or letting PAL select a sensible default
enum class OverrideMode : int32
{
    Default  = -1, ///< PAL selects the default behavior, which could be either enabled or disabled.
    Disabled = 0,  ///< Force to disabled. Equal to set to False.
    Enabled  = 1,  ///< Force to enabled. Equal to set to True.
};

/// Enumerates the depth clamping modes a pipeline can use.
enum class DepthClampMode : uint32
{
    Viewport    = 0x0,  ///< Clamps to the viewport min/max depth bounds
    _None       = 0x1,  ///< Disables depth clamping
#if PAL_BUILD_SUPPORT_DEPTHCLAMPMODE_ZERO_TO_ONE
    ZeroToOne   = 0x2,  ///< Clamps between 0.0 and 1.0.
#endif

    // Unfortunately for Linux clients, X.h includes a "#define None 0" macro.  Clients have their choice of either
    // undefing None before including this header or using _None when dealing with PAL.
#ifndef None
    None = _None,       ///< Disables depth clamping
#endif
};

/// Common flags controlling creation of both compute and graphics pipeline.
union PipelineCreateFlags
{
    struct
    {
        uint32 clientInternal        :  1; ///< Internal pipeline not created by the application.
        uint32 reverseWorkgroupOrder :  1; ///< Indicates that any Dispatch using this pipeline should execute in
                                           ///  reverse workgroup order. This superceeds the flag on the CommandBuffer
                                           ///  (dispatchPingPongWalk) - always forcing reverse workgroup order! This
                                           ///  is a best effort as not all implementations or Queues may support this.
        uint32 reserved              : 30; ///< Reserved for future use.
    };
    uint32 u32All;                         ///< Flags packed as 32-bit uint.
};

/// Constant definining the max number of view instance count that is supported.
constexpr uint32 MaxViewInstanceCount = 6;

/// Specifies graphic pipeline view instancing state.
struct ViewInstancingDescriptor
{
    uint32         viewInstanceCount;                           ///< The view instance count of the graphic pipeline
    uint32         viewId[MaxViewInstanceCount];                ///< The view instance ids.
    uint32         renderTargetArrayIdx[MaxViewInstanceCount];  ///< The instance render target array index, can be
                                                                ///  used in hardware accelerated stereo rendering.
    uint16         viewportArrayIdx[MaxViewInstanceCount];      ///< The instance viewport array index, can be
                                                                ///  used in hardware accelerated stereo rendering.
    bool           enableMasking;                               ///< Indicate whether instance masking is enabled.
};

// Specifies the input parameters for the MSAA coverage out feature.  MSAA coverage out is used in conjunction with a
// single sampled color image.  This feature exports a mask indicating which samples would have been used if the
// image had been multi-sampled.  The mask is exported to the specified channel of the MRT pointing to the rendered
// image.  That is, the MRT must be an active bound render target.  This MSAA mask data can then be post-processed.
struct MsaaCoverageOutDescriptor
{
    union
    {
        struct
        {
            uint32  enable        :  1; ///< Set to true to enable render target channel output
            uint32  numSamples    :  4; ///< Number of samples to export
            uint32  mrt           :  3; ///< Which MRT to export to.
            uint32  channel       :  2; ///< Which channel to export to (x = 0, y = 1, z = 2, w = 3)
            uint32  reserved      : 22;
        };

        uint32  u32All;
    } flags;
};

/// Specifies properties about an indirect function belonging to a compute @ref IPipelne object.  Part of the input
/// structure to IDevice::CreateComputePipeline().
struct ComputePipelineIndirectFuncInfo
{
    const char*  pSymbolName; ///< ELF Symbol name for the associated function.  Must not be null.
    gpusize      gpuVirtAddr; ///< [out] GPU virtual address of the function.  This is computed by PAL during
                              ///  pipeline creation.
};

/// Specifies properties for creation of a compute @ref IPipeline object.  Input structure to
/// IDevice::CreateComputePipeline().
struct ComputePipelineCreateInfo
{
    PipelineCreateFlags flags;                 ///< Flags controlling pipeline creation.

    const void*         pPipelineBinary;       ///< Pointer to Pipeline ELF binary implementing the Pipeline ABI
                                               ///  interface. The Pipeline ELF contains pre-compiled shaders,
                                               ///  register values, and additional metadata.
    size_t              pipelineBinarySize;    ///< Size of Pipeline ELF binary in bytes.
    uint32              maxFunctionCallDepth;  ///< Maximum depth for indirect function calls. Not used for a new
                                               ///  path ray-tracing pipeline as the compiler has pre-calculated
                                               ///  stack requirements.
    bool disablePartialDispatchPreemption; ///< Prevents scenarios where a subset of the dispatched thread groups are
                                           ///  preempted and the remaining thread groups run to completion. This
                                           ///  can occur when thread group granularity preemption is available and
                                           ///  instruction level (CWSR) is not. This setting is useful for allowing
                                           ///  dispatches with interdependent thread groups.
    DispatchInterleaveSize interleaveSize; ///< Controls how many thread groups are sent to one SE before switching to
                                           ///  the next one.

    /// PAL expects a fixed 3D thread group size for each compute pipeline but the HSA ABI supports dynamic group sizes.
    /// If this pipeline's ELF binary metadata doesn't specify a fixed thread group size, this should be used to force
    /// a particular thread group size. If this extent is set to all zeros PAL will use the metadata's group size.
    /// This field is not supported on PAL ABI ELFs, it should be set to all zeros.
    Extent3d threadsPerGroup;
    TriState groupLaunchGuarantee; ///< Force the group launch guarantee mechanism on or off. This feature will throttle
                                   ///  issuing of low priority waves when it detects too many higher priority waves are
                                   ///  failing to schedule due to resource contraints.

    const char* pKernelName; ///< When create pipeline with hsa ELF binary of multiple kernels, need to set one
                             ///  kernel to create the pipeline. null means only one kernel in ELF binary.

};

/// Specifies information about the viewport behavior of an assembled graphics pipeline.  Part of the input
/// structure @ref GraphicsPipelineCreateInfo.
struct ViewportInfo
{
    bool       depthClipNearEnable; ///< Enable clipping based on Near Z coordinate.
    bool       depthClipFarEnable;  ///< Enable clipping based on Far Z coordinate.
    DepthRange depthRange;          ///< Specifies Z dimensions of screen space (i.e., post viewport transform:
                                    ///  0 to 1 or -1 to 1).
};

/// Specifies edgeRule for rasterization
enum class EdgeRuleMode : uint32
{
    D3dCompliant    = 0x0,   ///< Use rasterization edge-rules which comply with the D3D spec.
    OpenGlDefault   = 0x1,   ///< Use rasterization edge-rules compatible with the default OpenGL driver.
};

/// Specifies Rasterizer state in properties for creation of a graphics
struct RasterizerState
{
    PointOrigin     pointCoordOrigin;          ///< Controls texture coordinate orientation for point sprites.
    bool            expandLineWidth;           ///< If true, line primitives will have their width expanded by 1/cos(a)
                                               ///  where a is the minimum angle from horizontal or vertical.
                                               ///  This can be used in conjunction with PS patching for a client to
                                               ///  implement line antialiasing.
    ShadeMode       shadeMode;                 ///< Specifies shading mode, Gouraud or Flat
    bool            rasterizeLastLinePixel;    ///< Specifies whether to draw last pixel in a line.
    bool            outOfOrderPrimsEnable;     ///< Enables out-of-order primitive rasterization.  PAL silently
                                               ///  ignores this if it is unsupported in hardware.
    bool            perpLineEndCapsEnable;     ///< Forces the use of perpendicular line end caps as opposed to
                                               ///  axis-aligned line end caps during line rasterization.
    BinningOverride binningOverride;           ///< Binning setting for this pipeline.

    DepthClampMode  depthClampMode;            ///< Depth clamping behavior

    union
    {
        struct
        {
            uint8 clipDistMaskValid : 1; ///< Whether or not @ref clipDiskMask, below, is valid.
            uint8 cullDistMaskValid : 1; ///< Whether or not @ref cullDistMask, below, is valid.
            uint8 reserved : 6;
        };
        uint8 u8All;                    ///< All the flags as a single value.
    } flags;

    uint8         cullDistMask;           ///< Mask of which cullDistance exports to leave enabled.
    uint8         clipDistMask;           ///< Mask of which clipDistance exports to leave enabled.
    bool          dx10DiamondTestDisable; ///< Disable DX10 diamond test during line rasterization.
    EdgeRuleMode  edgeRule;
};

/// Specifies Per-MRT color target info in olor target state
struct ColorTargetInfo
{
    SwizzledFormat swizzledFormat;      ///< Color target format and channel swizzle. Set the format to invalid
                                        ///  if no color target will be bound at this slot.
    uint8          channelWriteMask;    ///< Color target write mask.  Bit 0 controls the red channel, bit 1 is
                                        ///  green, bit 2 is blue, and bit 3 is alpha.
    bool           forceAlphaToOne;     ///< Treat alpha as one regardless of the shader output.  Ignored unless
                                        ///  supportAlphaToOne is set in DeviceProperties.
};

/// Specifies color target state in properties for creation of a graphics
struct ColorTargetState
{
    bool    alphaToCoverageEnable;           ///< Enable alpha to coverage.
    bool    dualSourceBlendEnable;           ///< Blend state bound at draw time will use a dual source blend mode.
    LogicOp logicOp;                         ///< Logic operation to perform.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 904
    bool    uavExportSingleDraw;             ///< When UAV export is enabled, acts as a hint that only a single draw
                                             ///  is done on a color target with this or subsequent pipelines before
                                             ///  a barrier. Improves performance by allowing pipelines to overlap.
#endif

    ColorTargetInfo target[MaxColorTargets]; ///< Per-MRT color target info.
};

/// Specifies properties for creation of a graphics @ref IPipeline object.  Input structure to
/// IDevice::CreateGraphicsPipeline().
struct GraphicsPipelineCreateInfo
{
    PipelineCreateFlags flags;                 ///< Flags controlling pipeline creation.

    const void*         pPipelineBinary;       ///< Pointer to Pipeline ELF binary implementing the Pipeline ABI
                                               ///  interface. The Pipeline ELF contains pre-compiled shaders,
                                               ///  register values, and additional metadata.
    size_t              pipelineBinarySize;    ///< Size of Pipeline ELF binary in bytes.
    const IShaderLibrary** ppShaderLibraries;  ///< An array of graphics @ref IShaderLibrary object. pPipelineBinary
                                               ///  and ppShaderLibraries can't be valid at the same time.
    size_t              numShaderLibraries;    ///< Number of graphics shaderLibrary object in ppShaderLibraries.
    bool                useLateAllocVsLimit;   ///< If set, use the specified lateAllocVsLimit instead of PAL internally
                                               ///  determining the limit.
    uint32              lateAllocVsLimit;      ///< The number of VS waves that can be in flight without having param
                                               ///  cache and position buffer space. If useLateAllocVsLimit flag is set,
                                               ///  PAL will use this limit instead of the PAL-specified limit.
    bool                useLateAllocGsLimit;   ///< If set, use the specified lateAllocVsLimit instead of PAL internally
                                               ///  determining the limit.
    uint32              lateAllocGsLimit;      ///< Controls GS LateAlloc val (for pos/prim allocations NOT param cache)
                                               ///  on NGG pipelines. Can be no more than 127.
    struct
    {
        struct
        {
            PrimitiveType primitiveType;      ///< Basic primitive category: points, line, triangles, patches.
            bool          topologyIsPolygon;  ///< Indicates that triangle primitives are combined to represent more
                                              ///  complex polygons. Only valid for triangle primitive types.
            uint32        patchControlPoints; ///< Number of control points per patch. Only required if primitiveType
                                              ///  is PrimitiveType::Patch.
        } topologyInfo; ///< Various information about the primitive topology that will be used with this pipeline.
                        ///  All of this info must be consistent with the full topology specified by
                        ///  ICmdBuffer::SetPrimitiveTopology() when drawing with this pipeline bound.

        /// Number of vertex buffer slots which are accessed by this pipeline.  Behavior is undefined if the pipeline
        /// tries to access a vertex buffer slot outside the range [0, vertexBufferCount).  It is generally advisable
        /// to make this the minimum value possible because that reduces the number of vertex buffer slots PAL has to
        /// maintain for this pipeline when recording command buffers.
        uint32 vertexBufferCount;
    } iaState;                   ///< Input assembler state.

    RasterizerState  rsState;    ///< Rasterizer state.
    ColorTargetState cbState;    ///< Color target state.

    ViewInstancingDescriptor  viewInstancingDesc;  ///< Descriptor describes view instancing state
                                                   ///  of the graphics pipeline
    MsaaCoverageOutDescriptor coverageOutDesc;     ///< Descriptor describes input parameters for MSAA coverage out.
    ViewportInfo              viewportInfo;        ///< Viewport info.
    DispatchInterleaveSize    taskInterleaveSize;  ///< Ignored for pipelines without a task shader. For pipelines with
                                                   ///  a task shader, controls how many thread groups are sent to one
                                                   ///  SE before switching to the next one.
    LdsPsGroupSizeOverride ldsPsGroupSizeOverride; ///< Whether to override ldsPsGroupSize setting for pipeline.

    TriState groupLaunchGuarantee; ///< Force the group launch guarantee mechanism on or off. This feature will throttle
                                   ///  issuing of low priority waves when it detects too many higher priority waves are
                                   ///  failing to schedule due to resource contraints.
    bool     noForceReZ;           ///< Disables the ability for PAL to force ReZ modes outside of what was chosen by
                                   ///  the compiler for this pipeline.
};

/// The graphic pipeline view instancing information. This is used to determine if hardware accelerated stereo rendering
/// can be enabled for a graphic pipeline.
struct GraphicPipelineViewInstancingInfo
{
    union
    {
        struct
        {
            uint32 shaderUseViewId                  : 1;   ///< If any shader in pipeline uses view id.
            uint32 gsExportRendertargetArrayIndex   : 1;   ///< If gs exports render target array index,
                                                           ///  must be 0 if there is no gs.
            uint32 gsExportViewportArrayIndex       : 1;   ///< If gs exports viewport array index,
                                                           ///  must be 0 if there is no gs.
            uint32 reserved                         : 29;  ///< Reserved for future use.
        };
        uint32 apiShaderFlags;
    };

    const ViewInstancingDescriptor* pViewInstancingDesc;    ///< View Instancing descriptor
};

/// Reports properties of a compiled pipeline.  This includes hashes for the pipeline and shaders that the client can
/// use to correlate PAL pipeline/shader dumps with corresponding API-level pipelines/shaders.
struct PipelineInfo
{
    const char* pName;                  ///< Non-owning pointer to a null-terminated string containing human-readable
                                        ///< name of this pipeline. nullptr indicates no name was provided.

    PipelineHash internalPipelineHash;  ///< 128-bit identifier extracted from this pipeline's ELF binary, composed of
                                        ///  the state the compiler decided was appropriate to identify the compiled
                                        ///  shaders.  The lower 64 bits are "stable"; the upper 64 bits are "unique".

    struct
    {
        ShaderHash hash;      ///< Unique 128-bit identifier for this shader.  0 indicates there is no shader bound for
                              ///  the corresponding shader stage.
    } shader[NumShaderTypes]; ///< Array of per-shader pipeline properties.

    union
    {
        struct
        {
            uint32 hsaAbi    : 1;  ///< This pipeline uses the HSA ABI (i.e. bind arguments not user-data)
            uint32 usesCps   : 1;  ///< This pipeline uses continuations passing shaders (CPS). In an archive pipeline,
                                   ///  this bit is set iff it is set in any constituent pipeline.
            uint32 cpsGlobal : 1;  ///< If using continuations passing shaders (CPS), stack is in global rather than
                                   ///  scratch. In an archive pipeline, this bit is set iff it is set in any
                                   ///  constituent pipeline.
            uint32 reserved  : 29; ///< Reserved for future use.
        };
        uint32 u32All;            ///< All flags combined as a single uint32.
    } flags;                      ///< Pipeline properties.

    struct
    {
        union
        {
            struct
            {
                uint32 perSampleShading : 1;    ///< Shader instructions want per-sample execution.
                uint32 usesSampleMask   : 1;    ///< Shader is using sample mask.
                uint32 enablePops       : 1;    ///< Primitive order pixel shader is enabled.
                uint32 reserved         : 29;   ///< Reserved for future use.
            };
            uint32 u32All;                      ///< All flags combined as a single uint32.
        } flags;
    } ps;                                       ///< Pixel shader properties.

    uint64 resourceMappingHash; ///< 64-bit hash of the resource mapping used when compiling the pipeline,
                                ///  if available (0 otherwise).

    /// The GPU address and size in bytes of the traceRays table, or { 0, 0 } if none.
    /// The format of the table is private to the compiler.
    /// Compiled code requires a buffer descriptor for the traceRays table to be passed in to the launch kernel
    /// in a particular place.
    gpusize traceRaysTable;
    gpusize traceRaysTableSize;

    /// Pointer to the the traceRays table for debugging purpose, or nullptr if none.
    const char* pTraceRaysTable;

    /// Pointer to and size in bytes of the shader identifiers table, or { nullptr, 0 } if none.
    /// The table has an entry for each export and then each hit group specified when the pipeline was compiled.
    /// Each entry has a pointer to the shader or hit group name in the same block of data (always 64 bits even
    /// on a 32-bit host), then the 32 byte shader identifier whose format is agreed between the compiler and
    /// GPURT's library code, for a total of 40 bytes per entry.
    const void* pShaderIdentifiers;
    size_t      shaderIdentifiersSize;

    uint32 unifiedRgsNameHash; ///< 32-bit hash of unified RGS name, 0 otherwise

};

/// A structure that represents any 3D arrangement of threads or thread groups as part of a compute shader dispatch.
///
/// This structure is halfway between Extent3d and Offset3d, depending on the context it may represent an offset or
/// an extent. Essentially it's meaning is tied to the concept of 3D thread or thread group grids rather than generic
/// contexts like "extent" or "offset". Whether it represents threads or thread groups is also context specific.
struct DispatchDims
{
    uint32 x; ///< Threads or thread groups in the X dimension.
    uint32 y; ///< Threads or thread groups in the Y dimension.
    uint32 z; ///< Threads or thread groups in the Z dimension.

    /// Computes the volume of this 3D arrangement of threads or thread groups.
    ///
    /// @returns the total number of threads or threads groups this struct represents.
    uint32 Flatten() const { return x * y * z; }
};

// There are some places where we'd like to directly cast DispatchDims to an array of three uint32s.
static_assert(sizeof(DispatchDims) == sizeof(uint32) * 3, "DispatchDims not castable to uint32*");

/// Component-wise addition of two DispatchDims.
///
/// @param [in] l  The left-hand argument.
/// @param [in] r  The right-hand argument.
///
/// @returns A new DispatchDims which contains the sum of 'l' and 'r' along each dimension.
inline DispatchDims operator+(DispatchDims l, DispatchDims r) { return {l.x + r.x, l.y + r.y, l.z + r.z}; }

/// Component-wise addition of one DispatchDims into another.
///
/// @param [in] l  The left-hand argument.
/// @param [in] r  The right-hand argument.
///
/// @returns A reference to 'l' after it is updated to the sum of 'l' and 'r'.
inline DispatchDims& operator+=(DispatchDims& l, DispatchDims r) { return l = (l + r); }

/// Component-wise multiplication of two DispatchDims.
///
/// @param [in] l  The left-hand argument.
/// @param [in] r  The right-hand argument.
///
/// @returns A new DispatchDims which contains the product of 'l' and 'r' along each dimension.
inline DispatchDims operator*(DispatchDims l, DispatchDims r) { return {l.x * r.x, l.y * r.y, l.z * r.z}; }

/// Component-wise multiplication of one DispatchDims into another.
///
/// @param [in] l  The left-hand argument.
/// @param [in] r  The right-hand argument.
///
/// @returns A reference to 'l' after it is updated to the product of 'l' and 'r'.
inline DispatchDims& operator*=(DispatchDims& l, DispatchDims r) { return l = (l * r); }

/// Used to represent API level shader stage.
enum ShaderStageFlagBits : uint32
{
    ApiShaderStageCompute  = (1u << static_cast<uint32>(ShaderType::Compute)),
    ApiShaderStageTask     = (1u << static_cast<uint32>(ShaderType::Task)),
    ApiShaderStageVertex   = (1u << static_cast<uint32>(ShaderType::Vertex)),
    ApiShaderStageHull     = (1u << static_cast<uint32>(ShaderType::Hull)),
    ApiShaderStageDomain   = (1u << static_cast<uint32>(ShaderType::Domain)),
    ApiShaderStageGeometry = (1u << static_cast<uint32>(ShaderType::Geometry)),
    ApiShaderStageMesh     = (1u << static_cast<uint32>(ShaderType::Mesh)),
    ApiShaderStagePixel    = (1u << static_cast<uint32>(ShaderType::Pixel)),
};

/// Reports shader stats. Multiple bits set in the shader stage mask indicates that multiple shaders have been combined
/// due to HW support. The same information will be repeated for both the constituent shaders in this case.
struct ShaderStats
{
    uint32             shaderStageMask;        ///< Indicates the stages of the pipeline this shader is
                                               /// used for. If multiple bits are set, it implies
                                               /// shaders were merged. See @ref ShaderStageFlagBits.
    CommonShaderStats  common;                 ///< The shader compilation parameters for this shader.
    /// Maximum number of VGPRs the compiler was allowed to use for this shader.  This limit will be the minimum
    /// of any architectural restriction and any client-requested limit intended to increase the number of waves in
    /// flight.
    uint32             numAvailableVgprs;
    /// Maximum number of SGPRs the compiler was allowed to use for this shader.  This limit will be the minimum
    /// of any architectural restriction and any client-requested limit intended to increase the number of waves in
    /// flight.
    uint32             numAvailableSgprs;
    size_t             isaSizeInBytes;         ///< Size of the shader ISA disassembly for this shader.
    ShaderHash         palShaderHash;          ///< Internal hash of the shader compilation data used by PAL.

    union
    {
        struct
        {
            uint32 writesUAV   : 1;     ///< This shader performs writes to UAVs.
            uint32 writesDepth : 1;     ///< Indicates explicit depth writes performed by the shader stage.
            uint32 streamOut   : 1;     ///< The shader performs stream out of shader generated data.
            uint32 reserved    : 29;    ///< Reserved for future use.
        };
        uint32 u32All;                  ///< All flags combined as a single uint32.

    } shaderOperations;                 ///< Flags depicting shader operations.

    struct
    {
        DispatchDims numThreadsPerGroup; ///< Number of compute threads per thread group in X, Y, and Z dimensions.
    } cs;                                ///< Parameters specific to compute shader only.

    union
    {
        struct
        {
            uint8 copyShaderPresent : 1; ///< Indicates that the copy shader data is valid.
            uint8 reserved          : 7; ///< Reserved for future use.
        };
        uint8 u8All;                     ///< All the flags as a single value.
    } flags;                             ///< Flags related to this shader data.

    CommonShaderStats  copyShader;       ///< This data is valid only when the copyShaderPresent flag above is set.
};

 /**
  ***********************************************************************************************************************
  * @interface IPipeline
  * @brief     Monolithic object containing all shaders and a large amount of "shader adjacent" state.  Separate concrete
  *            implementations will support compute or graphics pipelines.
  *
  * @see IDevice::CreateComputePipeline()
  * @see IDevice::CreateGraphicsPipeline()
  * @see IDevice::LoadPipeline()
  ***********************************************************************************************************************
  */
class IPipeline : public IDestroyable
{
public:
    /// Returns PAL-computed properties of this pipeline and its corresponding shaders.
    ///
    /// @returns Property structure describing this pipeline.
    virtual const PipelineInfo& GetInfo() const = 0;

    /// Returns a list of GPU memory allocations used by this pipeline.
    ///
    /// @param [in,out] pNumEntries    Input value specifies the available size in pAllocInfoList; output value
    ///                                reports the number of GPU memory allocations.
    /// @param [out]    pAllocInfoList If pAllocInfoList=nullptr, then pNumEntries is ignored on input.  On output it
    ///                                will reflect the number of allocations that make up this pipeline.  If
    ///                                pAllocInfoList!=nullptr, then on input pNumEntries is assumed to be the number
    ///                                of entries in the pAllocInfoList array.  On output, pNumEntries reflects the
    ///                                number of entries in pAllocInfoList that are valid.
    /// @returns Success if the allocation info was successfully written to the buffer.
    ///          + ErrorInvalidValue if the caller provides a buffer size that is different from the size needed.
    ///          + ErrorInvalidPointer if pNumEntries is nullptr.
    virtual Result QueryAllocationInfo(
        size_t*                    pNumEntries,
        GpuMemSubAllocInfo* const  pAllocInfoList) const = 0;

    /// Gives the client access to the resource ID used for internal Pal events.
    /// EX: Resource Create, Resource Bind, Resource Destroy.
    ///
    /// @returns The Resource ID.
    virtual const void* GetResourceId() const = 0;

    /// Obtains the binary code object for this pipeline.
    ///
    /// @param [in, out] pSize  Represents the size of the shader ISA code.
    ///
    /// @param [out] pBuffer    If non-null, the pipeline ELF is written in the buffer. If null, the size required
    ///                         for the pipeline ELF is given out in the location pSize.
    ///
    /// @returns Success if the pipeline binary was fetched successfully.
    ///          +ErrorUnavailable if the pipeline binary was not fetched successfully.
    virtual Result GetCodeObject(
        uint32*  pSize,
        void*    pBuffer) const = 0;

    /// Obtains the pointer of code object with ELF format according to the shader type. Returned ELF object is not
    /// guaranteed to be unique with different shader type, because a single code object can contain multiple shaders.
    ///
    /// @param [in] shaderType The shader stage for which the code object are requested.
    /// @param [out] pSize     The size of the ELF binary.
    ///
    /// @returns The pointer of ELF binary which contains requested shader stage.
    virtual const void* GetCodeObjectWithShaderType(
        ShaderType shaderType,
        size_t*    pSize) const = 0;

    /// Obtains the shader pre and post compilation stats/params for the specified shader stage.
    ///
    /// @param [in]  shaderType The shader stage for which the stats are requested.
    ///
    /// @param [out] pShaderStats Pointer to the ShaderStats structure which will be filled with the shader stats for
    ///                           the shader stage mentioned in shaderType. This cannot be nullptr.
    /// @param [in]  getDisassemblySize If set to true performs disassembly on the shader binary code and reports the
    ///                                 size of the disassembly string in ShaderStats::isaSizeInBytes. Else reports 0.
    /// @returns Success if the stats were successfully obtained for this shader, including the shader disassembly size.
    ///          +ErrorUnavailable if a wrong shader stage for this pipeline was specified, or if some internal error
    ///                           occured.
    virtual Result GetShaderStats(
        ShaderType   shaderType,
        ShaderStats* pShaderStats,
        bool         getDisassemblySize) const = 0;

    /// Obtains the compiled shader ISA code for the shader stage specified.
    ///
    /// @param [in]  shaderType The shader stage for which the shader cache entry is requested.
    ///
    /// @param [in, out] pSize  Represents the size of the shader ISA code.
    ///
    /// @param [out] pBuffer    If non-null, the shader ISA code is written in the buffer. If null, the size required
    ///                         for the shader ISA is given out in the location pSize.
    ///
    /// @returns Success if the shader ISA code was fetched successfully.
    ///          +ErrorUnavailable if the shader ISA code was not fetched successfully.
    virtual Result GetShaderCode(
        ShaderType shaderType,
        size_t*    pSize,
        void*      pBuffer) const = 0;

    /// Obtains the generated performance data for the shader stage specified.
    ///
    /// @param [in]      hardwareStage  The hardware stage of the shader which the performance data is requested.
    /// @param [in, out] pSize          Represents the size of the performance data.
    /// @param [out]     pBuffer        If non-null, the performance data is written in the buffer. If null, the size
    ///                                 required for the performance data is given out in the location pSize.
    ///
    /// @returns Success if the performance data was fetched successfully.
    ///          +ErrorUnavailable if the performance data was not fetched successfully.
    virtual Result GetPerformanceData(
        Util::Abi::HardwareStage hardwareStage,
        size_t*                  pSize,
        void*                    pBuffer) = 0;

    /// Notifies PAL that this pipeline may make indirect function calls to any function contained within any of the
    /// specified @ref IShaderLibrary objects.  This gives PAL a chance to perform any late linking steps required to
    /// valid execution of the possible function calls (this could include adjusting hardware resources such as GPRs
    /// or LDS space for the pipeline).
    ///
    /// This may be called multiple times on the same pipeline object.  Subsequent calls do not invalidate the result
    /// of previous calls.
    ///
    /// This must be called prior to binding this pipeline to a command buffer which will make function calls into any
    /// shader function contained within any of the specified libraries.  Failure to comply is an error and will result
    /// in undefined behavior.
    ///
    /// Currently only supported on compute pipelines.
    ///
    /// @param [in] ppLibraryList  List of @ref IShaderLibrary object to link with.
    /// @param [in] libraryCount   Number of valid library objects in the ppLibraryList array.
    ///
    /// @returns Success if the operation is successful.  Other return codes may include:
    ///          + ErrorUnavailable if called on a graphics pipeline.
    ///          + ErrorBadPipelineData if any of the libraries in ppLibraryList are not compatible with this pipeline.
    ///            Reasons for incompatibility include (but are not limited to) different user-data mappings, different
    ///            wavefront sizes, and other reasons.
    virtual Result LinkWithLibraries(
        const IShaderLibrary*const* ppLibraryList,
        uint32                      libraryCount) = 0;

    /// Sets the stack size for indirect function calls made by this pipeline. This may be smaller than or equal to the
    /// stack size already determined during pipeline creation or during an earlier call to LinkWithLibraries() because
    /// the client has access to more information about which functions contained in those libraries (or in the pipeline
    /// itself) are actually going to be called.
    ///
    /// Note that a future call to LinkWithLibraries() will invalidate this value and this should
    /// be called again.
    ///
    /// @param [in] stackSizeInBytes  Client-specified stack size, in bytes.
    virtual void SetStackSizeInBytes(
        uint32 stackSizeInBytes) = 0;

    /// Retrieve the stack sizes managed by compiler, including the frontend stack and the backend stack.
    ///
    /// @param [out] pSizes  To be filled with both the frontend stack size and the backend stack size, in bytes.
    ///
    /// @returns SUCCESS
    virtual Result GetStackSizes(
        CompilerStackSizes* pSizes) const = 0;

    /// Returns the API shader type to hardware stage mapping for the pipeline.
    ///
    /// @returns The appropriate mapping for this pipeline.
    virtual Util::Abi::ApiHwShaderMapping ApiHwShaderMapping() const = 0;

    /// Given the zero-based position of a kernel argument, return a pointer to that argument's metadata.
    ///
    /// @note Only compute pipelines using the HSA ABI have kernel arguments.
    ///
    /// @param [in] index  The zero-based position of the kernel argument to query.
    ///
    /// @returns A pointer to the kernel argument's metadata, or null if this pipeline doesn't have this argument.
    virtual const Util::HsaAbi::KernelArgument* GetKernelArgument(uint32 index) const = 0;

    /// Returns the value of the associated arbitrary client data pointer.
    /// Can be used to associate arbitrary data with a particular PAL object.
    ///
    /// @returns Pointer to client data.
    void* GetClientData() const { return m_pClientData; }

    /// Sets the value of the associated arbitrary client data pointer.
    /// Can be used to associate arbitrary data with a particular PAL object.
    ///
    /// @param  [in]    pClientData     A pointer to arbitrary client data.
    void SetClientData(
        void* pClientData)
    {
        m_pClientData = pClientData;
    }

    /// Get the array of underlying pipelines that this pipeline contains. For a normal non-multi-pipeline,
    /// this returns a single-entry array pointing to the same IPipeline. For a multi-pipeline compiled in
    /// dynamic launch mode, this returns an empty array. The contents of the returned array remain valid
    /// until the IPipeline is destroyed.
    ///
    /// @returns The array of underlying pipelines.
    virtual Util::Span<const IPipeline* const> GetPipelines() const = 0;

    /// Get the array of underlying shader libraries that this pipeline contains. For a normal non-multi-pipeline,
    /// this returns the empty array. The contents of the returned array remain valid until the IPipeline is
    /// destroyed.
    ///
    /// @returns The array of underlying shader libraries.
    virtual Util::Span<const IShaderLibrary* const> GetLibraries() const { return {}; }

protected:
    /// @internal Constructor. Prevent use of new operator on this interface. Client must create objects by explicitly
    /// called the proper create method.
    IPipeline() : m_pClientData(nullptr) {}

    /// @internal Destructor.  Prevent use of delete operator on this interface.  Client must destroy objects by
    /// explicitly calling IDestroyable::Destroy() and is responsible for freeing the system memory allocated for the
    /// object on their own.
    virtual ~IPipeline() { }

private:
    /// @internal Client data pointer. This can have an arbitrary value and can be returned by calling GetClientData()
    /// and set via SetClientData().
    /// For non-top-layer objects, this will point to the layer above the current object.
    void* m_pClientData;

    IPipeline(const IPipeline&) = delete;
    IPipeline& operator=(const IPipeline&) = delete;
};

} // Pal
