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
 * @file  palDevice.h
 * @brief Defines the Platform Abstraction Library (PAL) IDevice interface and related types.
 ***********************************************************************************************************************
 */

#pragma once

#include "palCmdAllocator.h"
#include "palDestroyable.h"
#include "palFence.h"
#include "palFile.h"
#include "palGpuMemory.h"
#include "palImage.h"
#include "palInlineFuncs.h"
#include "palLib.h"
#include "palPerfExperiment.h"
#include "palPipeline.h"
#include "palQueue.h"
#include <chrono>

#if PAL_KMT_BUILD
struct _SECURITY_ATTRIBUTES;
#endif

namespace Util
{
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
class IDevice;
class IFence;
class IGpuEvent;
class IGpuMemory;
class IImage;
class IIndirectCmdGenerator;
class IMsaaState;
class IPerfExperiment;
class IPipeline;
class IPrivateScreen;
class IQueryPool;
class IQueue;
class IQueueSemaphore;
class IShaderLibrary;
class ISwapChain;
struct BorderColorPaletteCreateInfo;
struct CmdAllocatorCreateInfo;
struct CmdBufferCreateInfo;
struct ColorBlendStateCreateInfo;
struct ColorTargetViewCreateInfo;
struct ComputePipelineCreateInfo;
struct DepthStencilStateCreateInfo;
struct DepthStencilViewCreateInfo;
struct ExternalImageOpenInfo;
struct ExternalGpuMemoryOpenInfo;
struct ExternalQueueSemaphoreOpenInfo;
struct ExternalResourceOpenInfo;
struct GpuEventCreateInfo;
struct GpuMemoryCreateInfo;
struct GpuMemoryOpenInfo;
struct GpuMemoryRef;
struct GraphicsPipelineCreateInfo;
struct ImageCreateInfo;
struct IndirectCmdGeneratorCreateInfo;
struct MsaaStateCreateInfo;
struct MsaaQuadSamplePattern;
struct PeerGpuMemoryOpenInfo;
struct PeerImageOpenInfo;
struct PerfExperimentCreateInfo;
struct PinnedGpuMemoryCreateInfo;
struct PresentableImageCreateInfo;
struct PrivateScreenCreateInfo;
struct PrivateScreenNotifyInfo;
struct QueryPoolCreateInfo;
struct QueueCreateInfo;
struct QueueSemaphoreCreateInfo;
struct QueueSemaphoreOpenInfo;
struct ShaderLibraryCreateInfo;
struct SwapChainCreateInfo;
struct SwapChainProperties;
struct SvmGpuMemoryCreateInfo;
struct GraphicPipelineViewInstancingInfo;
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 881
enum class WsiPlatform : uint32;
#else
enum WsiPlatform : uint32;
#endif
enum class PipelineBindPoint : uint32;
enum class VaRange : uint32;
enum class SwapChainMode : uint32;
namespace CmdDisassembly
{
    class ICmdBufferReporting;
}

/// Maximum string length for GPU names.  @see DeviceProperties.
constexpr uint32 MaxDeviceName = 256;

/// Maximum number of indirect user-data tables managed by PAL's command buffer objects.  @see DeviceFinalizeInfo.
constexpr uint32 MaxIndirectUserDataTables = 1;

/// Maximum number of supported entries in the MSAA sample pattern palette.  See IDevice::SetSamplePatternPalette().
constexpr uint32 MaxSamplePatternPaletteEntries = 16;

/// Maximum number of supported units in the gpu. These can be much larger than the actual values, but useful for arrays.
constexpr uint32 MaxShaderEngines       = 32;
/// Maximum number of supported subunits each Shader Engine splits into (SH or SA, depending on generation)
constexpr uint32 MaxShaderArraysPerSe   = 2;

/// Size of the Active Pixel Packer Mask in DWORDs
constexpr uint32 ActivePixelPackerMaskDwords = 4;

/// Maximum number of pixel packers per SE expected by PAL
constexpr uint32 MaxPixelPackerPerSe = 4;

/// Defines host flags for Semaphore/Fence Array wait
enum HostWaitFlags : uint32
{
    HostWaitAny                = 0x1,  ///< if set this bit, return after any signle semaphore/fence in the array has
                                       ///  completed. if not set, wait for completion of all semaphores/fences in the
                                       ///  array before returning.
};

/// Specifies what type of GPU a particular IDevice is (i.e., discrete vs. integrated).
enum class GpuType : uint32
{
    Unknown    = 0x0,  ///< The GPU type can't be determined and is unknown.
    Integrated = 0x1,  ///< Integrated GPU (i.e., APU).
    Discrete   = 0x2,  ///< Discrete GPU.
    Virtual    = 0x3,  ///< Virtualized GPU.
    Count
};

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 888
/// Specifies which operating-system-support IP level (OSSIP) this device has.
enum class OssIpLevel : uint32
{
    _None    = 0x0,   ///< @internal The device does not have an OSSIP block, or its level cannot be determined

    // Unfortunately for Linux clients, X.h includes a "#define None 0" macro.  Clients have their choice of either
    // undefing None before including this header or using _None when dealing with PAL.
#ifndef None
    None     = _None, ///< The device does not have an OSSIP block, or its level cannot be determined
#endif

    OssIp2_4 = 0x3,
    OssIp4   = 0x4,
};

/// Specifies which VCE IP level this device has.
enum class VceIpLevel : uint32
{
    _None    = 0x0,  ///< @internal The device does not have an VCEIP block, or its level cannot be determined

    // Unfortunately for Linux clients, X.h includes a "#define None 0" macro.  Clients have their choice of either
    // undefing None before including this header or using _None when dealing with PAL.
#ifndef None
    None     = _None, ///< The device does not have an VCEIP block, or its level cannot be determined
#endif

    VceIp1   = 0x1,
    VceIp2   = 0x2,
    VceIp3   = 0x3,
    VceIp3_1 = 0x4,
    VceIp3_4 = 0x5,
    VceIp4   = 0x6,
};

/// Specifies which UVD IP level this device has.
enum class UvdIpLevel : uint32
{
    _None    = 0x0,   ///< @internal The device does not have an UVDIP block, or its level cannot be determined

    // Unfortunately for Linux clients, X.h includes a "#define None 0" macro.  Clients have their choice of either
    // undefing None before including this header or using _None when dealing with PAL.
#ifndef None
    None     = _None, ///< The device does not have an UVDIP block, or its level cannot be determined
#endif

    UvdIp3_2 = 0x1,
    UvdIp4   = 0x2,
    UvdIp4_2 = 0x2,
    UvdIp5   = 0x3,
    UvdIp6   = 0x4,
    UvdIp6_2 = 0x5,
    UvdIp6_3 = 0x6,
    UvdIp7   = 0x7,
    UvdIp7_2 = 0x8,
};
#endif

/// Specifies which VCN IP level this device has.
enum class VcnIpLevel : uint32
{
    _None    = 0x0,   ///< @internal The device does not have an VCNIP block, or its level cannot be determined

    // Unfortunately for Linux clients, X.h includes a "#define None 0" macro.  Clients have their choice of either
    // undefing None before including this header or using _None when dealing with PAL.
#ifndef None
    None     = _None, ///< The device does not have an VCNIP block, or its level cannot be determined
#endif
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 888
#else // PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 888
    VcnIp1   = 0x1,
#endif
};

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 888
/// Specifies which SPU IP level this device has.
enum class SpuIpLevel : uint32
{
    _None    = 0x0,   ///< @internal The device does not have an SPUIP block, or its level cannot be determined
#ifndef None
    None     = _None, ///< The device does not have an SPUIP block, or its level cannot be determined
#endif
    SpuIp    = 0x1,
};
#endif

/// Specifies which PSP IP level this device has.
enum class PspIpLevel : uint32
{
    _None    = 0x0,   ///< @internal The device does not have an PSPIP block, or its level cannot be determined
#ifndef None
    None     = _None, ///< The device does not have an PSPIP block, or its level cannot be determined
#endif
    PspIp10  = 0x1,
};

/// Specified video decode type
enum class VideoDecodeType : uint32
{
    H264                = 0x0,      ///< H264 VLD
    Vc1                 = 0x1,      ///< VC1 VLD
    Mpeg2Idct           = 0x2,      ///< Partial MPEG2 decode (IT+MP)
    Mpeg2Vld            = 0x3,      ///< Full MPEG2 decode (RE+IT+MP+DB)
    Mpeg4               = 0x4,      ///< MPEG4
    Wmv9                = 0x5,      ///< WMV9 IDCT
    Mjpeg               = 0x6,      ///< Motion JPEG
    Hevc                = 0x7,      ///< HEVC
    Vp9                 = 0x8,      ///< VP9
    Hevc10Bit           = 0x9,      ///< HEVC 10bit
    Vp910Bit            = 0xa,      ///< VP9 10bit
    Av1                 = 0xb,      ///< AV1 8/10bit
    Av112Bit            = 0xc,      ///< AV1 12bit
    Count,
};

/// Video CODEC to use for encoding
enum class VideoEncodeCodec : uint32
{
    H264               = 0x0,       ///< H.264
    H265               = 0x1,       ///< H.265
    Av1                = 0x2,       ///< AV1
    Count
};

/// Specifies a virtual address range memory should be allocated in.
enum class VaRange : uint32
{
    Default,                ///< Default VA range.  Choose this for most allocations.
    DescriptorTable,        ///< Place the allocation in a 4GB VA range reserved by PAL for descriptor tables.  Knowing
                            ///  an allocation is allocated in this range, only one user data entry is required to
                            ///  specify a descriptor table.  @see ResourceMappingNodeType.
    ShadowDescriptorTable,  ///< Place the allocation in a 4GB VA range reserved by PAL for "shadow" descriptor tables.
                            ///  A shadow descriptor table is an additional table with the same layout as its parent
                            ///  descriptor table that can hold infrequently needed data like fmask SRDs or UAV counter
                            ///  data.  This scheme allows the client and SC to work out a known location for
                            ///  infrequently needed data without wasting a user data entry or wasting half of every
                            ///  descriptor cache line.
                            ///  Only supported if DeviceProperties::gpuMemoryProperties::flags::shadowDescVaSupport is
                            ///  set.
    Svm,                    ///< Place the allocation in a VA range reserved by PAL for shared virtual memory(SVM).
                            ///  This is a GPU VA range that is reserved also on the CPU-side.
                            ///  The size of reserved VA is set by PAL client by calling CreatePlatform.
    CaptureReplay,          ///< Place the allocation in a VA range reserved for capture and playback.
    Count,
};

/// Enumerates tmz(trusted memory zone) support level.
enum class TmzSupportLevel : uint32
{
    None          = 0, ///< TMZ not supported.
    PerQueue      = 1, ///< Enable TMZ mode per queue.
    PerSubmission = 2, ///< Enable TMZ mode per submission.
    PerCommandOp  = 3  ///< Enable TMZ mode per command operation.
};

/// How to interpret a single bit in a swizzle equation.
union SwizzleEquationBit
{
    struct
    {
        uint8 valid   : 1;  ///< Indicates whether this channel setting is valid.
        uint8 channel : 2;  ///< 0 for x channel, 1 for y channel, 2 for z channel.
        uint8 index   : 5;  ///< The channel index.
    };
    uint8 u8All;            ///< The above values packed in an 8-bit uint.
};

constexpr uint32 SwizzleEquationMaxBits = 20;   ///< Swizzle equations will consider no more than this many bits.
constexpr uint8  InvalidSwizzleEqIndex  = 0xFF; ///< Indicates an invalid swizzle equation index in the equation table.
constexpr uint8  LinearSwizzleEqIndex   = 0xFE; ///< An invalid eq. index indicating a row-major, linear memory layout.

/// Texture fetch meta-data capabilities bitfield definition, used with tcCompatibleMetaData setting
enum TexFetchMetaDataCaps : uint32
{
    TexFetchMetaDataCapsNoAaColor = 0x00000001,
    TexFetchMetaDataCapsMsaaColor = 0x00000002,
    TexFetchMetaDataCapsFmask = 0x00000004,
    TexFetchMetaDataCapsNoAaDepth = 0x00000008,
    TexFetchMetaDataCapsMsaaDepth = 0x00000010,
    TexFetchMetaDataCapsAllowStencil = 0x00000020,
    TexFetchMetaDataCapsAllowZ16 = 0x00000040,
};

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 936
/// Catalyst AI setting enums
enum CatalystAiSettings : uint32
{
    CatalystAiDisable = 0,
    CatalystAiEnable = 1,
    CatalystAiMaximum = 2,
};
#endif

/// Texture Filter optimization enum values
enum TextureFilterOptimizationSettings : uint32
{
    TextureFilterOptimizationsDisabled = 0,
    TextureFilterOptimizationsEnabled = 1,
    TextureFilterOptimizationsAggressive = 2,
};

/// Distribution Tess Mode enum values
enum DistributionTessMode : uint32
{
    DistributionTessOff = 0,
    DistributionTessDefault = 1,
    DistributionTessPatch = 2,
    DistributionTessDonut = 3,
    DistributionTessTrapezoid = 4,
    DistributionTessTrapezoidOnly = 5,
};

/// Defines the context roll optimization flags
enum ContextRollOptimizationFlags : uint32
{
    OptFlagNone = 0x00000000,
    PadParamCacheSpace = 0x00000001,
};

/// Defines the initial value to use for DCC metadata
enum class DccInitialClearKind {
    Uncompressed = 0x0,
    OpaqueBlack  = 0x1,
    OpaqueWhite  = 0x2,
    ForceBit     = 0x10,
    ForceOpaqueBlack = (ForceBit | OpaqueBlack),
    ForceOpaqueWhite = (ForceBit | OpaqueWhite),
};

/// Enum defining the different scopes (i.e. registry locations) where settings values are stored
enum InternalSettingScope : uint32
{
    PrivateDriverKey   = 0x0,
    PublicPalKey       = 0x1,
    PrivatePalKey      = 0x2,
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 888
    PublicCatalystKey  = 0x3,
    PrivatePalGfx9Key  = 0x4,
    PrivatePalGfx12Key = 0x5,
#else // PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 888
    PrivatePalGfx6Key  = 0x3,
    PrivatePalGfx9Key  = 0x4,
    PublicCatalystKey  = 0x5,
    PrivatePalGfx12Key = 0x7,
#endif
    PublicPalFile      = 0x9,
    RootDriverKey      = 0xa,
};

/// Enum defining override states for feature settings.
enum class FeatureOverride : uint32
{
    Default  = 0,  ///< Default setting state.
    Enabled  = 1,  ///< (Force) enabled state.  Default may change itself to this state.
    Disabled = 2   ///< (Force) disabled state.  Default may change itself to this state.
};

/// Enum bitmask defining externally-controlled (e.g. by Radeon Settings/KMD) driver feature settings.
enum RsFeatureType : uint32
{
    RsFeatureTypeTurboSync = (1u << 0),
    RsFeatureTypeChill     = (1u << 1),
    RsFeatureTypeDelag     = (1u << 2),
    RsFeatureTypeBoost     = (1u << 4),
    RsFeatureTypeProVsr    = (1u << 5),
};

/// Output structure containing information about the requested RsFeatureType (singular).
union RsFeatureInfo
{
    /// Global TurboSync settings.
    struct
    {
        bool enabled;    ///< Specifies whether TurboSync is enabled globally.
    } turboSync;

    /// Global Chill settings.
    struct
    {
        bool enabled;    ///< Specifies whether Chill is enabled globally.
        uint32 hotkey;   ///< If nonzero, specifies the virtual key code assigned to Chill.
        uint32 minFps;   ///< Specifies the global Chill minimum FPS limit.
        uint32 maxFps;   ///< Specifies the global Chill maximum FPS limit.
    } chill;

    /// Global Delag settings.
    struct
    {
        bool enabled;    ///< Specifies whether Delag is enabled globally.
        uint32 hotkey;   ///< If nonzero, specifies the virtual key code assigned to Delag.
        uint32 hotkeyInd;///< If nonzero, specifies the virtual key code assigned to Delag's indicator.
        uint32 limitFps; ///< Specifies the global Delag FPS limit.
        uint32 level;    ///< Specifies the global Delag level.
    } delag;

    /// Global Boost settings.
    struct
    {
        bool enabled;            ///< Specifies whether Boost is enabled globally.
        uint32 hotkey;           ///< If nonzero, specifies the virtual key code assigned to Boost.
        uint32 hotkeyInd;        ///< If nonzero, specifies the virtual key code assigned to Boost's indicator.
        uint32 minRes;           ///< Specifies the global Boost minimum resolution.
        bool adaptiveVrsEnabled; ///< Specifies whether BoostAdaptiveVrs is enabled globally.
    } boost;

    /// Global ProVsr settings.
    struct
    {
        bool enabled;    ///< Specifies whether ProVsr is enabled globally.
        uint32 hotkey;   ///< If nonzero, specifies the virtual key code assigned to ProVsr.
    } proVsr;

};

/// High-dynamic range (HDR) surface display modes.  Used to indicate the HDR display standard for a particular swap
/// chain texture format and screen colorspace/transfer function combination.
enum class HdrDisplayMode : uint32
{
    Sdr       = 0,  ///< Standard dynamic range; non-HDR compatible (default).
    Hdr10     = 1,  ///< HDR10 PQ.  Requires 10:10:10:2 swap chain.
    ScRgb     = 2,  ///< scRGB HDR (Microsoft and FreeSync2 linear mode).  1.0 = 80 nits, 125.0 = 10000 nits.
                    ///  Requires FP16 swapchain.
    FreeSync2 = 3,  ///< FreeSync2 HDR10 Gamma 2.2.  Requires 10:10:10:2 swap chain.
};

static constexpr uint32 MaxMiscStrLen = 61;

/// Whether to use graphics or compute for performing fast clears on depth stencil views.
enum class FastDepthStencilClearMode : uint8
{
    Default,    ///< Compute or graphics will be chosen at the driver's discretion
    Graphics,   ///< Graphics will always be used
    Compute     ///< Compute will always be used
};

enum DeferredBatchBinMode : uint32
{
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 875
    DeferredBatchBinCustom = 0,
    DeferredBatchBinAccurate = 1
#else
    DeferredBatchBinDisabled = 0,
    DeferredBatchBinCustom = 1,
    DeferredBatchBinAccurate = 2
#endif
};

/// PWS enable mode: e.g. disabled, fully enabled or partially enabled.
enum class PwsMode : uint32
{
    Disabled = 0,           ///< PWS feature is disabled
    Enabled = 1,            ///< PWS feature is fully enabled if HW supports.
    NoLateAcquirePoint = 2  ///< PWS feature is enabled with PWS counter only if HW supports, no late acquire points.
};

#if defined(__unix__)
/// Defines the supported VM always valid modes.
enum VmAlwaysValidEnable : uint32
{
    VmAlwaysValidDefaultEnable = 0,  ///< Only enable the optimization when kernel driver can support
    VmAlwaysValidForceDisable = 1,   ///< Force the optimization always disabled
    VmAlwaysValidForceEnable = 2     ///< Force the optimization always enabled
};
#endif

/// Alignment mode for accessing graphics/compute buffers.
/// This doesn't change the alignment of the whole buffer, just the alignment at which accesses must be made.
enum class BufferAlignmentMode : uint8
{
    Default,  ///< KMD (and therefore PAL) picks alignment requirement. Client should be prepared for anything.
    Dword,    ///< Hardware will automatically align requests to the smaller of: element-size or DWORD.
    Unaligned ///< Any request alignment is allowed.
};

enum TemporalHintsMrtBehavior : uint8
{
    TemporalHintsDynamicRt = 0x0, ///< Enable Dynamic RT Temporal hints. PAL chooses NT vs RT based on heuristics.
    TemporalHintsStaticRt  = 0x1, ///< Regular temporal for both near and far read/write caches.
    TemporalHintsStaticNt  = 0x2, ///< Non-temporal (re-use not expected) for both near and far read/write caches.
};

/// Client-controllable behavior for Gfx12-specific software workaround to HiSZ hardware bug.
/// Allows the client to override PAL defaults for performance or profiled reasons.
enum class HiSZWorkaroundBehavior : uint8
{
    Default = 0x0,                  ///< Let PAL decide what the default is.
    ForceDisableAllWar,             ///< Force disable all workarounds.
                                    ///  Note: This should rarely be chosen and carries an extremely high risk of issue.
                                    ///        Should only be used when application has been profiled to guarantee no
                                    ///        risk of issue.
    ForceHiSZDisableBasedWar,       ///< Force the disable HiZ/S based workaround behavior.
    ForceHiSZEventBasedWar,         ///< Force the event-after-draw workaround behavior.
                                    ///  Note: This carries a risk that the hang may still be seen.
    ForceHiSZDisableBaseWarWithReZ  ///< Force the disable HiZ/S based workaround behavior, but with an added
                                    ///  optimization to force ZOrder mode to EarlyZThenReZ to reclaim some performance.
                                    ///  This will apply to all graphics pipelines that trigger the workaround
                                    ///  condition, except those pipelines that have set
                                    ///  GraphicsPipelineCreateInfo::noForceReZ. This ZOrder change will occur only when
                                    ///  HiZ/S is forcibly disabled by the workaround; otherwise the ZOrder remains what
                                    ///  was chosen by the compiler.
};

/// Pal settings that are client visible and editable.
struct PalPublicSettings
{
    /// Maximum border color palette size supported by any queue.
    uint32 borderColorPaletteSizeLimit;

    /// Whether to use graphics or compute for performing fast clears on depth stencil views.
    FastDepthStencilClearMode fastDepthStencilClearMode;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 936
    /// Forces all serialized loads (LoadPipeline or LoadCompoundState) to fail.
    bool forceLoadObjectFailure;
#endif

    /// Controls the distribution mode for tessellation, which affects how patches are processed by different VGT
    /// units. 0: None - No distribution across VGTs (legacy mode). 1: Default - Optimal settings are chosen depending
    /// on the gfxip. 2: Patch - Individual patches are distributed to different VGTs. 3: Donut - Patches are split
    /// into donuts and distributed to different VGTs. 4: Trapezoid - Patches from donuts are split into trapezoids and
    /// distributed to different VGTs. Falls back to donut mode if HW does not support this mode. 5: Trapezoid only -
    /// Distribution turned off if HW does not support this mode.
    uint32 distributionTessMode;

    /// Flags that control PAL optimizations to reduce context rolls. 0: Optimization disabled. 1: Pad parameter cache
    /// space. Sets VS export count and PS interpolant number to per-command buffer maximum value. Reduces context rolls
    /// at the expense of parameter cache space.
    uint32 contextRollOptimizationFlags;

    /// The number of unbound descriptor debug srds to allocate. To detect reads of unbound descriptor within arrays,
    /// multiple debug srds can be allocated.
    uint32 unboundDescriptorDebugSrdCount;

    /// Disables compilation of internal PAL shaders. It can be enabled only if a PAL client won't use any of PAL blit
    /// functionalities on gfx/compute engines.
    bool disableResourceProcessingManager;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 936
    /// Controls app detect and image quality altering optimizations exposed by CCC.
    uint32 catalystAI;
#endif

    /// Controls texture filtering optimizations exposed by CCC.
    uint32 textureOptLevel;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 936
    /// Disables SC initialization. It can be enabled only if a PAL client won't use SC for shader compilation and
    /// provide direct ISA binaries(usually AQL path).
    bool disableScManager;
#endif

    /// Information about the client performing the rendering. For example: Rendered By PAL (0.0.1)
    char renderedByString[MaxMiscStrLen];

    /// Debug information that the client or tester might want reported.
    char miscellaneousDebugString[MaxMiscStrLen];

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 936
    /// Allows SC to make optimizations at the expense of IEEE compliance.
    bool allowNonIeeeOperations;

    /// Controls whether shaders should execute one atomic instruction per wave for UAV append/consume operations.
    /// If false, one atomic will be executed per thread.
    bool appendBufPerWaveAtomic;
#endif

    /// Bitmask of cases where texture compatible meta data will be used Single-sample color surface: 0x00000001 MSAA
    /// color surface: 0x00000002 FMask data: 0x00000004 Single-sample depth surface: 0x00000008 MSAA depth surface:
    /// 0x00000010 Allow stencil: 0x00000020 Allow Z-16 surfs 0x00000040
    uint32 tcCompatibleMetaData;

    /// Specifies the threshold below which CmdCopyMemory() is executed via a CpDma BLT, in bytes. CPDMA copies have
    /// lower overhead than CS/Gfx copies, but less throughput for large copies.
    uint32 cpDmaCmdCopyMemoryMaxBytes;

    /// Forces high performance state for allocated queues. Note: currently supported in Windows only.
    bool forceHighClocks;

    /// When submitting multiple command buffers in a single grQueueSubmit call, the ICD will patch the command streams
    /// so that the command buffers are chained together instead of submitting through KMD multiple times. This setting
    /// limits the number of command buffers that will be chained together; reduce to prevent problems due to long
    /// running submits.
    uint32 cmdBufBatchedSubmitChainLimit;

    /// Flags that control PAL's command allocator residency optimizations. If a command allocation isn't optimized PAL
    /// will wait for it to become resident at creation. 0x1 - Wait for command data to become resident at Submit-time.
    /// 0x2 - Wait for embedded data to become resident at Submit-time. 0x4 - Wait for marker data to become resident at
    /// Submit-time.
    uint32 cmdAllocResidency;

    /// Overrides max queued frames allowed
    uint32 maxQueuedFrames;

    /// Maximum number of presentable images per adapter(including LDA chain) which is recommended. If app exceeds the
    /// presentable image number threshold, awarning may be reported.
    uint32 presentableImageNumberThreshold;

    /// Provides a hint to PAL that client knows that every individual depth stencil surfaces are always cleared with
    /// same values.If TRUE, per-tile tracking of exp/clear will be enabled (requires HTile).
    bool hintInvariantDepthStencilClearValues;

    /// Provides a hint to PAL that PAL should disable color compression on surfaces that are smaller than or equal to
    /// this setting (setting * setting) in size.
    uint32 hintDisableSmallSurfColorCompressionSize;

    /// Disables Escape call to KMD. This is a temporary setting for experimentation that is expected to break features
    /// that currently needs Escape call.
    bool disableEscapeCall;

    /// A hint to the Windows OS that this application has submissions that are expected to run for a long time. This
    /// tells the OS that checking elapsed execution time is not a good way to judge if the GPU is hung.
    ///
    /// If this flag is set, Windows shouldn't TDR long submissions on uncontested hardware queues. Submissions on
    /// contested queues must preempt within the time limit to avoid a TDR. Setting this may also extend the TDR timer
    /// on compute queues.
    bool longRunningSubmissions;

    /// Disables MCBP on demand. This is a temporary setting until ATOMIC_MEM packet issue with MCBP is resolved.
    bool disableCommandBufferPreemption;

    /// Disable the fast clear eliminate skipping optimization.  This optimization will conservatively track the usage
    /// of clear values to allow the vast majority of images that never clear to a value that isn't TC-compatible to
    /// skip the CPU and front-end GPU overhead of issuing a predicated fast clear eliminate BLT.
    bool disableSkipFceOptimization;

    /// Sets the minimum BPP of surfaces which will have DCC enabled
    uint32 dccBitsPerPixelThreshold;

    /// See largePageSizeInBytes in DeviceProperties. This limit defines how large an allocation must be to have
    /// PAL automatically pad allocation starting virtual address alignments to enable this optimization. By
    /// default, PAL will use the KMD-reported limit.
    gpusize largePageMinSizeForVaAlignmentInBytes;

    /// See largePageSizeInBytes in DeviceProperties. This limit defines how large an allocation must be to have
    /// PAL automatically pad allocation sizes to fill an integral number of large pages. By default, PAL will
    /// use the KMD-reported limit.
    gpusize largePageMinSizeForSizeAlignmentInBytes;

    /// Makes the unbound descriptor debug srd 0 so the hardware drops the load and ignores it instead of pagefaulting.
    /// Used to workaround incorrect app behavior.
    bool zeroUnboundDescDebugSrd;

    /// Preferred heap for uploading client pipelines. Default is set to @ref GpuHeap::GpuHeapInvisible. Setting is
    /// ignored for internal pipelines and are uploaded to @ref GpuHeap::GpuHeapLocal.
    GpuHeap pipelinePreferredHeap;

    bool depthClampBasedOnZExport;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 928
    /// Force the PreColorTarget to an earlier PreRasterization point if used as a wait point. This is to prevent a
    /// write-after-read hazard for a corner case: shader exports from distinct packers are not ordered. Advancing
    /// wait point from PreColorTarget to PostPrefetch could cause over-sync due to extra  VS/PS_PARTIAL_FLUSH
    /// inserted. It is default to false, but client drivers may choose to app-detect to enable if see corruption.
    bool forceWaitPointPreColorToPostPrefetch;
#endif

    /// Allows the client to disable debug overlay visual confirm after DebugOverlay::Platform is created when the
    /// panel setting DebugOverlayEnabled is globally set but a certain application might need to turn off visual
    /// confirm to make the screen not too noisy.
    bool disableDebugOverlayVisualConfirm;

    bool enableExecuteIndirectPacket;

    /// Offers flexibility to the client to choose Graphics vs Compute engine for Indirect Command Generation
    /// (Shader path) based on performance and other factors. The default is false since we have seen perf gains using
    /// the ACE.
    bool disableExecuteIndirectAceOffload;

    /// Value to initialize metadata for DCC surfaces to, if they are compressable. This has no effect on non-DCC
    /// images. Images whose initial layout is not compressable are only affected if this is "forced".
    ///  0x00 - Uncompressed (default)
    ///  0x01 - Opaque Black
    ///  0x02 - Opaque White
    ///  0x11 - Forced Opaque Black
    ///  0x12 - Forced Opaque White
    uint32 dccInitialClearKind;

    /// Allows the client to not create internal VrsImage. Pal internal will create a 16M image as vrsImageSize.
    bool disableInternalVrsImage;

    /// Allows the client to control binning persistent and context states per bin.
    /// A value of 0 tells PAL to pick the number of states per bin.
    uint32 binningPersistentStatesPerBin;
    uint32 binningContextStatesPerBin;

    /// This key controls if binning will be disabled when the PS may kill pixels.
    OverrideMode disableBinningPsKill;

    /// Controls GS LateAlloc val (for pos/prim allocations NOT param cache) on NGG pipelines. Can be no more than 127.
    uint32 nggLateAllocGs;

    /// Bitmask of cases where RPM view memory accesses will bypass the MALL
    /// RpmViewsBypassMallOff (0x0): Disable MALL bypass
    /// RpmViewsBypassMallOnRead (0x1): Skip MALL for read access of views created in RPM
    /// RpmViewsBypassMallOnWrite (0x2): Skip MALL for write access of views created in RPM
    /// RpmViewsBypassMallOnCbDbWrite (0x4): Control the RPM CB/DB behavior
    RpmViewsBypassMall rpmViewsBypassMall;

    /// Optimize color export format for depth only rendering. Only applicable for RB+ parts
    bool optDepthOnlyExportRate;

    /// Controls whether or not we should expand Hi-Z to full range rather than doing fine-grain resummarize
    /// operations.  Expanding Hi-Z leaves the Hi-Z data in a less optimal state but is a much faster operation
    /// than the fine-grain resummarize.
    bool expandHiZRangeForResummarize;

    /// Control whether to have command buffer emit SQTT marker events. Useful for client driver to perform SQTT
    /// dump without the involvement of dev driver.
    bool enableSqttMarkerEvent;

    /// Controls the value of CB_COLOR0_ATTRIB.LIMIT_COLOR_FETCH_TO_256B_MAX. This bit limits CB fetch to 256B on cache
    /// miss, regardless of sector size.
    bool limitCbFetch256B;

    /// Controls whether or not deferred batch binning is enabled 0 : Batch binning always disabled 1 : Use custom bin
    /// sizes 2 : Optimal.
    DeferredBatchBinMode binningMode;

    /// Controls the custom batch bin size.Only used when deferredBatchBinMode == 1 High word is for x, low word is for
    /// y. Default is 128x128. Values must be power of two between 16 and 512.
    uint32               customBatchBinSize;

    /// Maximum number of primitives per batch. The maximum value is 1024.
    uint32               binningMaxPrimPerBatch;

    /// Controls PWS enable mode: disabled, fully enabled or partially enabled. Only take effect if HW supports PWS.
    PwsMode pwsMode;

    /// Controls the MaxScratchRingSizeBaseline, which is really just the maximum size of the scratch ring
    gpusize maxScratchRingSizeBaseline;

    /// Controls the maximum size of the scratch ring allocation
    uint32 maxScratchRingSizeScalePct;

#if defined(__unix__)
    /// Whether enable vm-always-valid feature on Linux while allocating Bo
    VmAlwaysValidEnable  enableVmAlwaysValid;
#endif

    /// Alignment mode for graphics and compute buffers.
    /// Clients must check Pal::DeviceProperties::osProperties::flags::forceAlignmentSupported
    /// to see if anything other than default will work.
    BufferAlignmentMode hardwareBufferAlignmentMode;

    // Disallows putting the shader ring in system memory for performance purposes
    // This is done by un-listing GpuHeapGartUswc as a possible heap for the shader rings.
    // Instead only allowing GpuHeapInvisible and GpuHeapLocal
    bool forceShaderRingToVMem;

    /// If the client sets this to true they promise they've done exhaustive testing on every ASIC to prove that this
    /// application can use AC01 fast clears safely. This should never be forced to true unconditionally.
    bool ac01WaNotNeeded;

    /// Toggles whether or not image copies will prefer using the graphics pipeline. This setting does not force all
    /// copies to use graphics or compute, it changes what method will be selected in cases where either could be used.
    bool preferGraphicsImageCopy;

    /// Bitmask to control adding Waits around Flush events
    /// This is public setting to allow AppDetect to override of 'WaitOnFlush' setting.
    /// Setting 'waitOnFlush' for actual workarounds is strongly discouraged because:
    ///  1) It has a negative performance impact.
    ///  2) Waits effect the timing and pipeline execution which can hide underlying hw/fw/sw bugs.
    /// Issues resolved by added waits should be root caused.
    uint32 waitOnFlush;

    /// Provides the ability for mall to be alloc-ed/noalloc-ed using the dynamic or static behavior.
    TemporalHintsMrtBehavior temporalHintsMrtBehavior;

    /// Allows the client to change the behavior of the Gfx12 HW Bug that impacts HiS and HiZ.
    /// This should never be forced away from default unconditionally.
    HiSZWorkaroundBehavior hiSZWorkaroundBehavior;

    /// For event-based HiSZ workarounds (Gfx12), specifies the timeouts supported by the Scan Converter and Depth Block
    /// for their tile summarizer controller.
    /// Ignored unless @ref hiSZWorkaroundBehavior is set to ForceHiSZEventBasedWar.
    /// If set to 0, PAL picks a default value.
    /// Note: Choosing any value other than 0 carries with it **significant risk** of hangs, as the timeout value
    ///       determined by PAL is the most optimal to avoid the most hangs. Any deviation from the default must be
    ///       thoroughly tested and is not guaranteed to be safe!
    uint32 tileSummarizerTimeout;

#if PAL_KMT_BUILD
    /// If client sets it to false, native fence will be disabled in OS wide no matter OS or hardware supports
    /// native fence or not.
    bool enableNativeFence;
#endif

    /// Whether HiZ/HiS is valid after depth/stencil image is copied as destination.
    ///  - If false, this is not guaranteed, PAL will issue necessary HiZ/HiS resummarization at later client barrier
    ///    call time when transition to another HiZ/HiS valid layout like DepthStencilTarget.
    ///  - If true, PAL does necessary HiZ/HiS fixup after copy if needed and HiZ/HiS will be always valid after copy.
    ///    At later client barrier call time, no HiZ/HiS resummarization will happen.
    bool makeLayoutCopyDstHiSZCompatible;
};

/// Defines the modes that the GPU Profiling layer can use when its buffer fills.
enum GpuProfilerStallMode : uint32
{
    GpuProfilerStallAlways = 0,     ///< Always stall to get accurate trace data
    GpuProfilerStallLoseDetail = 1, ///< Lose register-level detail if under pressure to avoid stalls
    GpuProfilerStallNever = 2,      ///< Never stall, miss trace packets
};

/// Describes the equations needed to interpret the raw memory of a tiled texture.
struct SwizzleEquation
{
    SwizzleEquationBit addr[SwizzleEquationMaxBits]; ///< Address setting: each bit is the result of addr ^ xor ^ xor2.
    SwizzleEquationBit xor1[SwizzleEquationMaxBits]; ///< xor setting.
    SwizzleEquationBit xor2[SwizzleEquationMaxBits]; ///< xor2 setting.
    uint32             numBits;                      ///< The number of bits in the equation.
    bool               stackedDepthSlices;           ///< True if depth slices are treated as being stacked vertically
                                                     ///  prior to swizzling.
};

/// Specifies the hardware features supported for PRT (sparse images).
enum PrtFeatureFlags : uint32
{
    PrtFeatureBuffer                = 0x00000001,   ///< Indicates support for sparse buffers
    PrtFeatureImage2D               = 0x00000002,   ///< Indicates support for sparse 2D images
    PrtFeatureImage3D               = 0x00000004,   ///< Indicates support for sparse 3D images
    PrtFeatureImageMultisampled     = 0x00000008,   ///< Indicates support for sparse multisampled images
    PrtFeatureImageDepthStencil     = 0x00000010,   ///< Indicates support for sparse depth/stencil images
    PrtFeatureShaderStatus          = 0x00000020,   ///< Indicates support for residency status in shader instructions
    PrtFeatureShaderLodClamp        = 0x00000040,   ///< Indicates support for LOD clamping in shader instructions
    PrtFeatureUnalignedMipSize      = 0x00000080,   ///< Indicates support for non-miptail levels with dimensions that
                                                    ///  aren't integer multiples of the tile size as long as they are
                                                    ///  at least as large as a single tile
    PrtFeaturePerSliceMipTail       = 0x00000100,   ///< Indicates support for per-slice miptail (slice-major order)

    PrtFeatureTileAliasing          = 0x00000200,   ///< Indicates support for aliasing tiles (without metadata)
    PrtFeatureStrictNull            = 0x00000400,   ///< Indicates whether reads of unmapped tiles always return zero
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 936
    PrtFeatureNonStandardImage3D    = 0x00000800,   ///< Indicates support for sparse 3D images restricted to
                                                    ///  non-standard tile shapes that match the tile mode block depth
    PrtFeaturePrtPlus               = 0x00001000,   ///< Indicates that this image supports use of residency maps.
#else
    PrtFeaturePrtPlus               = 0x00000800,   ///< Indicates that this image supports use of residency maps.
#endif
};

/// Describe the settings' scope accessible by clients.
enum class SettingScope
{
    Driver,   ///< For settings specific to a UMD
    Global,   ///< For global settings controlled by CCC
    File,     ///< For settings that are only read from a file
    Root,     ///< For settings under HKR
};

/// Big Software (BigSW) Release information structure
/// Software release management uses this version # to control a rollout of big SW features together.
struct BigSoftwareReleaseInfo
{
    uint32 majorVersion;         ///< BigSW Release Major version
    uint32 minorVersion;         ///< BigSW Release Minor version.
    uint32 miscControl;          ///< BigSW Release miscellaneous control.
};

/// Virtual display capabilities as determined by the OS. The reported values bound the valid ranges of values supported
/// by the @ref VirtualDisplayInfo structure passed in to @ref IDevice::CreateVirtualDisplay.
struct VirtualDisplayCapabilities
{
    uint32   maxVirtualDisplays; ///< The maximum number of virtual display supported
    Rational minRefreshRate;     ///< The minimum refresh rate
    Rational maxRefreshRate;     ///< The maximum refresh rate
};

/// The properties of a specific virtual display
struct VirtualDisplayProperties
{
    bool isVirtualDisplay; ///< True, if it's a virtual display
};

/// Enumerates all of the types of local video memory which could be associated with a GPU.
enum class LocalMemoryType : uint32
{
    Unknown = 0,
    Ddr2,
    Ddr3,
    Ddr4,
    Gddr5,
    Gddr6,
    Hbm,
    Hbm2,
    Hbm3,
    Lpddr4,
    Lpddr5,
    Ddr5,
    Count
};

/// Bitmask of all MSAA/EQAA types supported, in terms of samples (S) and shaded fragments (F)
enum MsaaFlags : uint16
{
    MsaaS1F1  = 0x0001,
    MsaaS2F1  = 0x0002,
    MsaaS4F1  = 0x0004,
    MsaaS8F1  = 0x0008,
    MsaaS16F1 = 0x0010,
    MsaaAllF1 = 0x001F,

    MsaaS2F2  = 0x0020,
    MsaaS4F2  = 0x0040,
    MsaaS8F2  = 0x0080,
    MsaaS16F2 = 0x0100,
    MsaaAllF2 = 0x01E0,

    MsaaS4F4  = 0x0200,
    MsaaS8F4  = 0x0400,
    MsaaS16F4 = 0x0800,
    MsaaAllF4 = 0x0E00,

    MsaaS8F8  = 0x1000,
    MsaaS16F8 = 0x2000,
    MsaaAllF8 = 0x3000,

    MsaaAll   = 0x3FFF,
};

/// Supported RTIP version enumeration
enum class RayTracingIpLevel : uint32
{
    _None   = 0x0,   ///< The device does not have an RayTracing Ip Level
#ifndef None
    None    = _None, ///< The device does not have an RayTracing Ip Level
#endif
    RtIp1_0 = 0x1,   ///< First Implementation of HW RT
    RtIp1_1 = 0x2,   ///< Added computation of triangle barycentrics into HW
    RtIp2_0 = 0x3,   ///< Added more Hardware RayTracing features, such as BoxSort, PointerFlag, etc
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 888
    RtIp3_0 = 0x4,   ///< Added high precision box node, HW instance node, dual intersect ray, BVH8 intersect ray,
                     ///  LDS stack push 8 pop 1, and LDS stack push 8 pop 2
    RtIp3_1 = 0x5,   ///< Added improved bvh footprints (change to node pointer, 128 Byte primitive structure format,
                     ///  128 Byte Quantized box node, obb support, wide sort)
#else // PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 888
    RtIp3_0 = 0x4,   ///< Added high precision box node, HW instance node, dual intersect ray, BVH8 intersect ray,
                     ///  LDS stack push 8 pop 1, and LDS stack push 8 pop 2
    RtIp3_1 = 0x6,   ///< Added improved bvh footprints (change to node pointer, 128 Byte primitive structure format,
                     ///  128 Byte Quantized box node, obb support, wide sort)
#endif
};

/// Which ip version a component has reported
struct IpTriple
{
    uint32 major    : 8;     /// Major revision value
    uint32 minor    : 8;     /// Minor revision value
    uint32 stepping : 16;    /// Stepping value

    constexpr operator uint32() const
    {
        return ((stepping      ) |
                (minor    << 16) |
                (major    << 24));
    }
};

/// Reports various properties of a particular IDevice to the client.  @see IDevice::GetProperties.
struct DeviceProperties
{
    uint32     vendorId;                     ///< Vendor ID (should always be 0x1002 for AMD).
    uint32     deviceId;                     ///< GPU device ID (e.g., Hawaii XT = 0x67B0).
    uint32     revisionId;                   ///< GPU revision.  HW-specific value differentiating between different
                                             ///  SKUs or revisions.  Corresponds to one of the PRID_* revision IDs.
    uint32     eRevId;                       ///< GPU emulation/internal revision ID.
    AsicRevision revision;                   ///< ASIC revision.
    GpuType    gpuType;                      ///< Type of GPU (discrete vs. integrated)
    uint16     gpuPerformanceCapacity;       ///< Portion of GPU assigned in virtualized system (SRIOV)
                                             ///< 0-65535, 0 invalid (not virtualized), 1 min, 65535 max
    GfxIpLevel gfxLevel;                     ///< IP level of this GPU's GFX block
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 888
    OssIpLevel ossLevel;                     ///< IP level of this GPU's OSS block
    VceIpLevel vceLevel;                     ///< IP level of this GPU's VCE block
    UvdIpLevel uvdLevel;                     ///< IP level of this GPU's UVD block
#endif
    VcnIpLevel vcnLevel;                     ///< IP level of this GPU's VCN block
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 888
    SpuIpLevel spuLevel;                     ///< IP level of this GPU's SPU block
#endif
    PspIpLevel pspLevel;                     ///< IP level of this GPU's PSP block
    uint32     gfxStepping;                  ///< Stepping level of this GPU's GFX block
    IpTriple   gfxTriple;                    ///< Full GFX IP level (major.minor.step) of this GPU
    char       gpuName[MaxDeviceName];       ///< Null terminated string identifying the GPU.
    uint32     gpuIndex;                     ///< Device's index in a linked adapter chain.
    uint32     maxGpuMemoryRefsResident;     ///< Maximum number of GPU memory references that can be resident
                                             ///  at any time.  Memory references set both via IQueue and IDevice
                                             ///  (via AddGpuMemoryReferences() or Submit()) count against this limit.
    uint64     timestampFrequency;           ///< Frequency of the device's timestamp counter in Hz.
                                             ///  @see ICmdBuffer::CmdWriteTimestamp.
    uint32     attachedScreenCount;          ///< Number of screen attached to the device.
    uint32     maxSemaphoreCount;            ///< Queue semaphores cannot have a signal count higher than this value.
                                             ///  For example, one indicates that queue semaphores are binary.
    PalPublicSettings settings;              ///< Public settings that the client has the option of overriding
#if PAL_CLIENT_EXAMPLE
    AddrHandle hAddrlib;                     ///< Handle to addrlib for directed image tests.
#endif

    struct
    {
        union
        {
            struct
            {
                /// This engine supports timestamps (ICmdBuffer::CmdWriteTimestamp()).
                uint32 supportsTimestamps              :  1;

                /// This engine supports ICmdBuffer::CmdSetPredication() based on Streamout/Occlusion query
                uint32 supportsQueryPredication        :  1;

                /// This engine supports ICmdBuffer::CmdSetPredication() based on a 32-bit GPU memory allocation
                uint32 supports32bitMemoryPredication  :  1;

                /// This engine supports ICmdBuffer::CmdSetPredication() based on a 64-bit GPU memory allocation
                uint32 supports64bitMemoryPredication  :  1;

                /// This engine supports ICmdBuffer::If(), Else() and EndIf() calls.
                uint32 supportsConditionalExecution    :  1;

                /// This engine supports ICmdBuffer::While() and EndWhile() calls.
                uint32 supportsLoopExecution           :  1;

                /// This engine supports ICmdBuffer::CmdWaitRegisterValue(), WaitMemoryValue() and
                /// CopyRegisterToMemory() calls.
                uint32 supportsRegMemAccess            :  1;

                /// This engine supports ICmdBuffer::CmdCopyImage() between optimally tiled images with
                /// mismatched tiling tokens.
                uint32 supportsMismatchedTileTokenCopy :  1;

                /// This engine supports ICmdBuffer::Barrier() calls that transition out of the @ref
                /// LayoutUninitializedTarget layout.
                uint32 supportsImageInitBarrier        :  1;

                /// This engine supports ICmdBuffer::Barrier() calls that transition out of the @ref
                /// LayoutUninitializedTarget layout for individual subresources. If this is not set and
                /// supportsImageInitBarrier is set, the subresource range must span the entire image.
                uint32 supportsImageInitPerSubresource :  1;

                /// This engine does not support any virtual memory features. IQueue::RemapVirtualMemoryPages and
                /// IQueue::CopyVirtualPageMappings are not supported on Queues using this engine.
                uint32 runsInPhysicalMode              :  1;

                /// Indicates whether this engine can do virtual memory remap or not.
                uint32 supportVirtualMemoryRemap       :  1;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 914
                /// Indicates whether this Queues using this engine can maintain the contents of CE RAM across
                /// consecutive submissions.  If this is not set, the client must not specify a nonzero value for
                /// either @ref QueueCreateInfo::persistentCeRamSize or @ref QueueCreateInfo::persistentCeRamOffset.
                uint32 supportPersistentCeRam          :  1;
#else
                uint32 reserved914                     :  1;
#endif

                /// Indicates whether the engine supports the command allocator tracks which chunk is idle.
                uint32 supportsTrackBusyChunks         :  1;

                /// Indicates whether the engine can safely access non-resident ranges of resources.
                uint32 supportsUnmappedPrtPageAccess   :  1;

                /// This engine supports clear or copy with MSAA depth-stencil destination
                uint32 supportsClearCopyMsaaDsDst      :  1;

                /// Reserved for future use.
                uint32 reserved                        : 16;
            };
            uint32 u32All;                  ///< Flags packed as 32-bit uint.
        } flags;                            ///< Engines property flags.

        struct
        {
            union
            {
                struct
                {
                    uint32 exclusive                :  1;  ///< Engine is exclusively owned by one client at a time.
                    uint32 mustUseDispatchTunneling :  1;  ///< Queues created on this engine must use dispatch
                                                           ///  tunneling.
                    /// Indicates whether this engine instance can be used for gang submission workloads via
                    /// a multi-queue.
                    /// @see IDevice::CreateMultiQueue.
                    uint32 supportsMultiQueue       :  1;
                    uint32 hwsEnabled               :  1;
                    uint32 isHighPriority           :  1;
                    uint32 reserved                 : 27;  ///< Reserved for future use.
                };
                uint32 u32All;                        ///< Flags packed as 32-bit uint.
            } flags;                                  ///< Capabilities property flags.

            uint32 queuePrioritySupport;              ///< Mask of QueuePrioritySupport flags indicating which queue
                                                      ///  priority levels are supported by this engine.
            uint32 dispatchTunnelingPrioritySupport;  ///< Mask of QueuePrioritySupport flags indicating which queue
                                                      ///  priority levels support dispatch tunneling on this engine.
            uint32 maxFrontEndPipes;                  ///< Up to this number of IQueue objects can be consumed in
                                                      ///  parallel by the front-end of this engine instance. It will
                                                      ///  only be greater than 1 on hardware scheduled engine backed
                                                      ///  by multiple hardware pipes/threads.
        } capabilities[MaxAvailableEngines];          ///< Lists each engine of this type (up to engineCount) and their
                                                      ///  properties.

        uint32   engineCount;                   ///< Number available engines of this type.
        uint32   queueSupport;                  ///< Mask of QueueTypeSupport flags indicating which queues are
                                                ///  supported by this engine.
        uint32   maxBorderColorPaletteSize;     ///< Maximum size of a border color palette on this engine.
        uint32   controlFlowNestingLimit;       ///< Maximum depth of command-buffer control flow nesting on this
                                                ///  engine.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 914
        uint32   ceRamSizeAvailable;            ///< Size, in bytes, of constant engine RAM available on this engine.
#endif
        Extent3d minTiledImageCopyAlignment;    ///< Minimum alignments (pixels) for X/Y/Z/Width/Height/Depth for
                                                ///  ICmdBuffer::CmdCopyImage() between optimally tiled images.
        Extent3d minTiledImageMemCopyAlignment; ///< Minimum alignments (bytes) for X/Y/Z/Width/Height/Depth for
                                                ///  ICmdBuffer::CmdCopyImage() with an optimally tiled image and a
                                                ///  linearly tiled image. Also applies to
                                                ///  ICmdBuffer::CmdCopyImageToMemory() or
                                                ///  ICmdBuffer::CmdCopyMemoryToImage() with an optimally tiled image.
        Extent3d minLinearMemCopyAlignment;     ///< Minimum alignments (bytes) for X/Y/Z/Width/Height/Depth for
                                                ///  ICmdBuffer::CmdCopyTypedBuffer().
        uint32   minTimestampAlignment;         ///< If supportsTimestamps is set, this is the minimum address alignment
                                                ///  in bytes of the dstOffset in ICmdBuffer::CmdWriteTimestamp().
        uint32   maxNumDedicatedCu;             ///< The maximum number of dedicated CUs for the real time audio queue
        uint32   maxNumDedicatedCuPerQueue;     ///< The maximum number of dedicated CUs per queue
        uint32   dedicatedCuGranularity;        ///< The granularity at which compute units can be dedicated to a queue
        /// Specifies the suggested heap preference clients should use when creating an @ref ICmdAllocator that will
        /// allocate command space for this engine type.  These heap preferences should be specified in the allocHeap
        /// parameter of @ref CmdAllocatorCreateInfo.  Clients are free to ignore these defaults and use their own
        /// heap preferences, but may suffer a performance penalty.
        GpuHeap preferredCmdAllocHeaps[CmdAllocatorTypeCount];

        /// Indicate which queue supports per-command, per-submit, or per-queue TMZ based on the queue type.
        TmzSupportLevel tmzSupportLevel;
    } engineProperties[EngineTypeCount];    ///< Lists available engines on this device and their properties.

    struct
    {
        union
        {
            struct
            {
                /// This queue supports IQueue::PresentSwapChain() calls.  Note that a queue may support swap chain
                /// presents even if the supportedDirectPresentModes flags below indicate no support for direct
                /// presents; instead swap chain PresentMode support is queried via GetSwapChainInfo.
                uint32 supportsSwapChainPresents  :  1;
                uint32 reserved744                :  1;
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 927
                /// Set if the queue supports additional split barrier feature on top of basic acquire/release
                /// interface support. This provides CmdAcquire() and CmdRelease() to implement split barriers.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 893
                ///  Note: supportReleaseAcquireInterface is a prerequisite to supportSplitReleaseAcquire.
#endif
                uint32 supportSplitReleaseAcquire :  1;
#else
                uint32 reserved927                :  1;
#endif

                /// Reserved for future use.
                uint32 reserved                   : 29;
            };
            uint32 u32All;                    ///< Flags packed as 32-bit uint.
        } flags;                              ///< Queue property flags.

        uint32   supportedDirectPresentModes; ///< A mask of PresentModeSupport flags indicating support for various
                                              ///  PresentModes when calling IQueue::PresentDirect().
    } queueProperties[QueueTypeCount];        ///< Lists the properties of all queues supported by PAL.

    struct
    {
        union
        {
            struct
            {
                /// Indicates support for virtual GPU memory allocations.  @see IQueue::RemapVirtualMemoryPages.
                uint32 virtualRemappingSupport          :  1;

                /// Indicates support for pinning system memory for access as GPU memory.
                /// @see IDevice::PinSystemMemory.
                uint32 pinningSupport                   :  1;

                /// Indicates support pinned memory which is host-mapped from foreign device.
                uint32 supportHostMappedForeignMemory   :  1;

                /// Indicates whether specifying memory references at Submit time is supported. If not supported
                /// all memory references must be manged via IDevice or IQueue AddGpuMemoryReferences()
                uint32 supportPerSubmitMemRefs          :  1;

                /// Indicates support for GPU virtual addresses that are visible to all devices.
                uint32 globalGpuVaSupport               :  1;

                /// Indicates support for Shared Virtual Memory VA range.
                uint32 svmSupport                       :  1;

                /// Indicates support for shadow desc VA range.
                uint32 shadowDescVaSupport              :  1;

                /// Indicates support for IOMMUv2. Fine grain SVM is not supported without IOMMU.
                /// PAL client needs to check this flag before using fine grain SVM.
                /// IOMMU is a memory management unit (MMU) that connects a direct-memory-access-capable
                /// (DMA-capable) I/O bus to the main memory.
                uint32 iommuv2Support                   :  1;

                /// Indiciates that the platform supports automatic GPU memory priority management.
                uint32 autoPrioritySupport              :  1;

                /// Indicates KMD has enabled HBCC(High Bandwidth Cache Controller) page migration support.  This means
                /// shaders must be compiled such that all memory clauses can be replayed in response to an XNACK.
                uint32 pageMigrationEnabled             :  1;
                /// Indicates TMZ (or HSFB) protected memory allocations are supported.
                uint32 supportsTmz                      :  1;

                /// Memory allocations on this device support MALL (memory access last level); essentially
                /// the lowest level cache possible.
                uint32 supportsMall                     : 1;

                /// Support for querying page fault information
                uint32 supportPageFaultInfo             : 1;

                /// Indicates if this device supports GFX12-style distributed compression. Client can control
                /// whether distributed compression is enabled or not per IGpuMemory object using the
                /// distributedCompression field in @ref GpuMemoryCreateInfo.
                uint32 supportDistributedCompression    : 1;

                /// All GPU memory allocations in all heaps are always initialized to zero on creation.
                /// The @ref initializeToZero flag is not needed when this is true.
                uint32 alwaysInitializedToZero : 1;

                /// Reserved for future use.
                uint32 reserved : 17;
            };
            uint32 u32All;           ///< Flags packed as 32-bit uint.
        } flags;                     ///< GPU memory property flags.

        gpusize realMemAllocGranularity;    ///< The addresses and sizes of "real" GPU memory objects must be aligned
                                            ///  to at least this many bytes.
        gpusize virtualMemAllocGranularity; ///< The addresses and sizes of virtual GPU memory objects must be aligned
                                            ///  to at least this many bytes.
        gpusize virtualMemPageSize;         ///< Size in bytes of a virtual GPU memory page.
                                            ///  @see IQueue::RemapVirtualMemoryPages.
        gpusize fragmentSize;               ///< Size in bytes of a video memory fragment.  If GPU memory object
                                            ///  addresses and sizes are aligned to at least this value, VA translation
                                            ///  will be a bit faster.  It is aligned to the allocation granularities.
        gpusize largePageSizeInBytes;       ///< The large page optimization will allow compatible allocations to
                                            ///  potentially be upgraded to a page size larger than 64KiB to reduce TLB
                                            ///  pressure.  PAL will automatically pad the size and alignment of some
                                            ///  allocations to enable this optimization;
                                            ///  see largePageMinSizeForAlignmentInBytes in PalPublicSettings.
        gpusize maxVirtualMemSize;          ///< Total virtual GPU memory available (total VA space size).
        gpusize maxPhysicalMemSize;         ///< Total VRAM available (Local + Invisible + non-Local heap sizes).
        gpusize vaStart;                    ///< Starting address of the GPU's virtual address space.
        gpusize vaEnd;                      ///< Ending address of the GPU's virtual address space.
        gpusize descTableVaStart;           ///< Starting address of the descriptor table's virtual address space
        gpusize shadowDescTableVaStart;     ///< Starting address of the shadow descriptor table's virtual address space
        gpusize privateApertureBase;        ///< Private memory base address for generic address space (Windows only).
        gpusize sharedApertureBase;         ///< Shared memory base address for generic address space (Windows only).

        gpusize busAddressableMemSize;      ///< SDI/DirectGMA GPU aperture size set in CCC
        gpusize maxLocalMemSize;            ///< Total VRAM available on the GPU (Local + Invisible heap sizes).
        LocalMemoryType localMemoryType;    ///< Type of local memory used by the GPU.
        gpusize maxCaptureReplaySize;       ///< Total virtual GPU available for Capture/Replay
        gpusize barSize;                    ///< Total VRAM which can be accessed by the CPU.

        struct
        {
            float  maxMemClock;      ///< Maximum GPU memory clock in MHz. For DX builds this value is valid only after
                                     ///  the device has been finalized.
            uint32 memPerfRating;    ///< Precomputed performance rating of memory operations.
            uint32 vramBusBitWidth;  ///< Memory bus width.
            uint32 memOpsPerClock;   ///< Memory operations per clock.
        } performance;               ///< Performance-related memory properties.

    } gpuMemoryProperties;           ///< Memory properties for this device.

    struct
    {
        union
        {
            struct
            {
                /// Images created on this device supports AQBS stereo mode, this AQBS stereo mode doesn't apply to the
                /// array-based stereo feature supported by Presentable images.
                uint32 supportsAqbsStereoMode       :  1;

                /// Set if images created on this device support being created with corner sampling.
                uint32 supportsCornerSampling       :  1;

                /// Load Bearing placeholder, do not touch.
                uint32 placeholder0                 :  1;

                /// Reserved for future use.
                uint32 reserved                     : 29;
            };
            uint32 u32All;              ///< Flags packed as 32-bit uint.
        } flags;                        ///< GPU memory property flags.

        Extent3d        maxDimensions;  ///< Maximum supported width/height/depth for an image.
        uint32          maxArraySlices; ///< Maximum supported number of array slices for a 1D or 2D image.
        PrtFeatureFlags prtFeatures;    ///< PRT features supported by the hardware.
        gpusize         prtTileSize;    ///< Size, in bytes, of a PRT tile.
        MsaaFlags       msaaSupport;    ///< Bitflags for MSAA sample/fragment count support.
        uint8           maxMsaaFragments; ///< Max number of MSAA fragments per pixel (may have more samples).
        uint8           numSwizzleEqs;  ///< How many swizzle equations are in pSwizzleEqs.
        Extent2d        vrsTileSize;    ///< Pixel dimensions of a VRS tile.  0x0 indicates image-based shading rate
                                        ///  is not supported.
        const SwizzleEquation* pSwizzleEqs; ///< These describe how to interpret device-dependent tiling modes.

        bool     tilingSupported[static_cast<size_t>(ImageTiling::Count)]; ///< If each image tiling is supported.
    } imageProperties;                  ///< Image properties for this device.

    struct
    {
        /// Maximum number of available shader-accessible user data entries. @see PipelineShaderInfo.
        uint32 maxUserDataEntries;
        uint32 maxThreadGroupSize;  ///< Per-device limit on threads per threadgroup for compute shaders.
        /// Some hardware supported by PAL has a bug which can cause a GPU hang if async compute enginesare used while
        /// compute shaders with > maxAsyncComputeThreadGroupSize are in flight on any queue. This reports the
        /// maximum "safe" limit on threads per threadgroup for compute shaders for this device if the client wishes to
        /// use async compute engines. Note that the bug can occur if the following conditions are met:
        ///  (a) Async compute workloads are running *somewhere* on the GPU, in any process;
        ///  (b) Some compute workloads on either the async compute engine or on the universal engine have a threads per
        ///      threadgroup amount which exceeds maxAsyncComputeThreadGroupSize.
        ///
        /// It is up to the client to choose how to work around this bug. They are free to either limit applications to
        /// only creating compute shaders with <= maxAsyncComputeThreadGroupSize threads per group, or to avoid using
        /// the async compute engines at all.
        ///
        /// If this value equals maxThreadGroupSize, then the device does not have this bug and the client can use
        /// any compute shader on any queue.
        uint32 maxAsyncComputeThreadGroupSize;

        uint32 maxComputeThreadGroupCountX; ///< Maximum number of thread groups supported
        uint32 maxComputeThreadGroupCountY; ///< Maximum number of thread groups supported
        uint32 maxComputeThreadGroupCountZ; ///< Maximum number of thread groups supported

        uint32 maxBufferViewStride; ///< Maximum stride, in bytes, that can be specified in a buffer view.

        uint32 hardwareContexts;    ///< Number of distinct state contexts available for graphics workloads.  Mostly
                                    ///  irrelevant to clients, but may be useful to tools.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 914
        uint32 ceRamSize;           ///< Maximum on-chip CE RAM size in bytes.
#endif
        uint32 maxPrimgroupSize;    ///< Maximum primitive group size.
        uint32 supportedVrsRates;   ///< Bitmask of VrsShadingRate enumerations indicating which modes are supported.

        uint32 mallSizeInBytes;     ///< Size of total MALL (Memory Attached Last Level - L3) cache in bytes.

        uint32 gl2UncachedCpuCoherency; ///< If supportGl2Uncached is set, then this is a bitmask of all
                                        ///  CacheCoherencyUsageFlags that will be coherent with CPU reads/writes.
                                        ///  Note that reporting CoherShader only means that GLC accesses will be
                                        ///  CPU coherent.
                                        ///  Note: Only valid if @ref supportGl2Uncached is true.

        uint32 maxGsOutputVert;             ///< Maximum number of GS output vertices.
        uint32 maxGsTotalOutputComponents;  ///< Maximum number of GS output components totally.
        uint32 maxGsInvocations;            ///< Maximum number of GS prim instances, corresponding to geometry shader
                                            ///  invocation in glsl.

        RayTracingIpLevel       rayTracingIp;       ///< HW RayTracing IP version

        uint32                  cpUcodeVersion;   ///< Command processor feature version.
        uint32                  pfpUcodeVersion;  ///< Command processor, graphics prefetch firmware version.

        union
        {
            struct
            {
                uint64 support8bitIndices                 :  1; ///< Hardware natively supports 8bit indices
                uint64 support16BitInstructions           :  1; ///< Hardware supports FP16 and INT16 instructions
                uint64 supportBorderColorSwizzle          :  1; ///< Hardware supports border color swizzle
                uint64 supportDoubleRate16BitInstructions :  1; ///< Hardware supports double rate packed math
                uint64 supportFp16Fetch                   :  1; ///< Hardware supports FP16 texture fetches
                uint64 supportFp16Dot2                    :  1; ///< Hardware supports a paired FP16 dot product.
                uint64 supportConservativeRasterization   :  1; ///< Hardware supports conservative rasterization
                uint64 supportImplicitPrimitiveShader     :  1; ///< Device supports implicit compiling of the
                                                                ///  hardware vertex shader as a primitive shader to
                                                                ///  perform culling and compaction optimizations in
                                                                ///  the shader.
                uint64 supportMeshShader                  :  1; ///< Indicates support for mesh shaders.
                uint64 supportTaskShader                  :  1; ///< Indicates support for task shaders.
                uint64 supportMsFullRangeRtai             :  1; ///< HW supports full range render target array
                                                                ///  index for Mesh Shaders.
                uint64 supportPrtBlendZeroMode            :  1; ///< Blend zero mode support.
                uint64 supports2BitSignedValues           :  1; ///< Hardware natively supports 2-bit signed values.
                uint64 supportPrimitiveOrderedPs          :  1; ///< Hardware supports primitive ordered UAV
                                                                ///  accesses in the PS.
                uint64 supportPatchTessDistribution       :  1; ///< Hardware supports patch level tessellation
                                                                ///  distribution among VGTs.
                uint64 supportDonutTessDistribution       :  1; ///< Hardware supports donut granularity of
                                                                ///  tessellation distribution among VGTs.
                uint64 supportTrapezoidTessDistribution   :  1; ///< Hardware supports trapezoid granularity of
                                                                ///  tessellation distribution among VGTs.
                uint64 supportSingleChannelMinMaxFilter   :  1; ///< Hardware supports min/max filtering that can
                                                                ///  return one channel at a time.
                uint64 supportPerChannelMinMaxFilter      :  1; ///< Hardware returns min/max value on a per-channel
                                                                ///  basis.
                uint64 supportRgpTraces                   :  1; ///< Hardware supports RGP traces.
                uint64 supportMsaaCoverageOut             :  1; ///< Set if HW supports MSAA coverage feature
                uint64 supportPostDepthCoverage           :  1; ///< Set if HW supports post depth coverage feature
                uint64 supportSpiPrefPriority             :  1; ///< Set if HW supports preference priority.
                uint64 supportWaveBreakSize               :  1; ///< The HW supports specifying the wavebreak size
                                                                ///  in the pixel shader pipeline.
                uint64 supportsPerShaderStageWaveSize     :  1; ///< If set, the "waveSize" setting in the
                                                                ///  @ref PipelineShaderInfo structure is meaningful.
                uint64 placeholder2                       :  1; ///< Placeholder for backward compatibility, no use it.
                uint64 supportSpp                         :  1; ///< Hardware supports Shader Profiling for Power.
                uint64 timestampResetOnIdle               :  1; ///< GFX timestamp resets after idle between
                                                                ///  submissions. The client cannot assume that
                                                                ///  timestamps will increase monotonically across
                                                                ///  command buffer submissions.
                uint64 support1xMsaaSampleLocations       :  1; ///< HW supports 1xMSAA custom quad sample patterns
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 893
                uint64 supportReleaseAcquireInterface     :  1; ///< Set if HW supports the basic functionalities of
                                                                ///  acquire/release-based barrier interface. This
                                                                ///  provides CmdReleaseThenAcquire() as a convenient
                                                                ///  way to replace the legacy barrier interface's
                                                                ///  CmdBarrier() to handle single point barriers.
#else
                uint64 placeholder4                       :  1; ///< Placeholder for backward compatibility, no use it.
#endif
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 883
                uint64 supportSplitReleaseAcquire         :  1; ///< Set if HW supports additional split barrier feature
                                                                ///  on top of basic acquire/release interface support.
                                                                ///  This provides CmdAcquire() and CmdRelease() to
                                                                ///  implement split barriers.
                                                                ///  Note: supportReleaseAcquireInterface is a
                                                                ///  prerequisite to supportSplitReleaseAcquire.
#else
                uint64 placeholder3                       :  1; ///< Placeholder for backward compatibility, no use it.
#endif
                uint64 supportGl2Uncached                 :  1; ///< Indicates support for the allocation of GPU L2
                                                                ///  un-cached memory. @see gl2UncachedCpuCoherency
                uint64 supportOutOfOrderPrimitives        :  1; ///< HW supports higher throughput for out of order
                uint64 supportIntersectRayBarycentrics    :  1; ///< HW supports the ray intersection mode which
                                                                ///  returns triangle barycentrics.
                uint64 supportFloat32BufferAtomics        :  1; ///< Hardware supports float32 buffer atomics
                uint64 supportFloat32ImageAtomics         :  1; ///< Hardware supports float32 image atomics
                uint64 supportFloat32BufferAtomicAdd      :  1; ///< Hardware supports float32 buffer atomic add
                uint64 supportFloat32ImageAtomicAdd       :  1; ///< Hardware supports float32 image atomic add
                uint64 supportFloat64Atomics              :  1; ///< Hardware supports float64 atomics
                uint64 supportFloat32ImageAtomicMinMax    :  1; ///< Hardware supports float32 image atomic min and max
                uint64 supportFloat64BufferAtomicMinMax   :  1; ///< Hardware supports float64 buffer atomic min and max
                uint64 supportFloat64SharedAtomicMinMax   :  1; ///< Hardware supports float64 shared atomic min and max
                uint64 support64BitInstructions           :  1; ///< Hardware supports 64b instructions
                uint64 supportShaderSubgroupClock         :  1; ///< HW supports clock functions across subgroup.
                uint64 supportShaderDeviceClock           :  1; ///< HW supports clock functions across device.
                uint64 supportAlphaToOne                  :  1; ///< HW supports forcing PS output alpha channel to 1
                uint64 supportCaptureReplay               :  1; ///< HW supports captureReplay
                uint64 supportSortAgnosticBarycentrics    :  1; ///< HW supports sort-agnostic Barycentrics for PS
                uint64 supportVrsWithDsExports            :  1; ///< If true, asic support coarse VRS rates
                                                                ///  when z or stencil exports are enabled
                uint64 supportRayTraversalStack           :  1; ///< HW assisted ray tracing traversal stack support
                uint64 supportPointerFlags                :  1; ///< Ray tracing HW supports flags embedded in the node
                                                                ///  pointer bits
                uint64 supportTextureGatherBiasLod        :  1; ///< HW supports SQ_IMAGE_GATHER4_L_O
                uint64 supportInt8Dot                     :  1; ///< Hardware supports a dot product 8bit.
                uint64 supportInt4Dot                     :  1; ///< Hardware supports a dot product 4bit.
                uint64 supportMixedSignIntDot             :  1; ///< Hardware supports a integer dot product with mixed
                                                                ///  sign inputs.
                uint64 support2DRectList                  :  1; ///< HW supports PrimitiveTopology::TwoDRectList.
                uint64 supportHsaAbi                      :  1; ///< PAL supports HSA ABI compute pipelines.
                uint64 supportImageViewMinLod             :  1; ///< Indicates image srd supports min_lod.
                uint64 supportStaticVmid                  :  1; ///< Indicates support for static-VMID
                uint64 support3dUavZRange                 :  1; ///< HW supports read-write ImageViewSrds of 3D images
                                                                ///  with zRange specified.
                uint64 supportCooperativeMatrix           :  1; ///< HW supports cooperative matrix
                uint64 support1dDispatchInterleave        :  1; ///< Indicates support for 1D Dispatch Interleave.
                uint64 support2dDispatchInterleave        :  1; ///< Indicates support for 2D Dispatch Interleave.
                uint64 supportBFloat16                    :  1; ///< HW supports bf16 instructions.
                uint64 supportFloat8                      :  1; ///< HW supports float 8-bit instructions.
                uint64 supportInt4                        :  1; ///< HW supports integer 4-bit instructions.
                uint64 supportCooperativeMatrix2          :  1; ///< HW supports Gfx12 extension cooperative matrix.
                uint64 placeholder14                      :  2;
                uint64 reserved                           :  59; ///< Reserved for future use.
            };
            uint64 u64All[2];           ///< Flags packed as 32-bit uint.
        } flags;                     ///< Device IP property flags.

        struct
        {
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 892
            union
            {
                uint32 bufferView;
#endif
                uint32 typedBufferView; ///< Size in bytes (and required alignment) of a typed buffer view SRD.
                                        ///  @see IDevice::CreateTypedBufferViewSrds().
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 892
            };
#endif
            uint32 untypedBufferView;  ///< Size in bytes (and required alignment) of a untyped buffer view SRD.
                                       ///  @see IDevice::CreateUntypedBufferViewSrds().
            uint32 imageView;          ///< Size in bytes (and required alignment) of an image view SRD.
                                       ///  @see IDevice::CreateImageViewSrds().
            uint32 fmaskView;          ///< Size in bytes (and required alignment) of an fmask view SRD.
                                       ///  @see IDevice::CreateFmaskViewSrds().  This value can be zero to denote
                                       ///  a lack of fMask support.
            uint32 sampler;            ///< Size in bytes (and required alignment) of a sampler SRD.
                                       ///  @see IDevice::CreateSamplerSrds().
            uint32 bvh;                ///< Size in bytes (and required alignment) of a BVH SRD
                                       ///  Will be zero if HW doesn't support ray-tracing capabilities.
                                       ///  @see IDevice::CreateBvhSrds().
        } srdSizes;                    ///< Sizes for various types of _shader resource descriptor_ (SRD).

        struct
        {
            const void* pNullBufferView;  ///< Pointer to null buffer view srd
            const void* pNullImageView;   ///< Pointer to null image view srd
            const void* pNullFmaskView;   ///< Pointer to null fmask view srd.  This pointer can be nullptr to
                                          ///  indicate a lack of fMask support.
            const void* pNullSampler;     ///< Pointer to null sampler srd
        } nullSrds;                       ///< Null SRDs are used to drop shader writes or read 0

        struct
        {
            float  maxGpuClock;      ///< Maximum GPU engine clock in MHz. For DX builds this value is valid only after
                                     ///  the device has been finalized.
            float  aluPerClock;      ///< Maximum shader ALU operations per clock.
            float  texPerClock;      ///< Maximum texture fetches per clock.
            float  primsPerClock;    ///< Maximum primitives processed per clock.
            float  pixelsPerClock;   ///< Maximum pixels processed per clock.
            uint32 gfxipPerfRating;  ///< Precomputed performance rating of the GfxIp block.
        } performance;               ///< Performance-related device properties.

        struct
        {
            union
            {
                struct
                {
                    uint32 eccProtectedGprs                    :  1; ///< Whether or not the GPU has ECC protection
                                                                     ///< on its VGPR's
                    uint32 placeholder0                        :  1;
                    uint32 reserved                            : 30; ///< Reserved for future use.
                };
                uint32     u32All;                ///< Flags packed as a 32-bit unsigned integer.
            } flags;

            uint32 numShaderEngines;        ///< Number of non-harvested shader engines.
            uint32 numShaderArrays;         ///< Number of shader arrays.
            uint32 numCusPerShaderArray;    ///< Number of CUs per shader array that are actually usable.
            uint32 maxCusPerShaderArray;    ///< Maximum number of CUs per shader array. Count of physical CUs prior to
                                            ///< harvesting CUs for yield in certain variants of ASICs (ex: Fiji PRO).
            uint32 numSimdsPerCu;           ///< Number of SIMDs per compute unit.
            uint32 numWavefrontsPerSimd;    ///< Number of wavefront slots in each SIMD.
            uint32 numActiveRbs;            ///< Number of active Renderbackends
            uint32 nativeWavefrontSize;     ///< The native wavefront size.
            uint32 minWavefrontSize;        ///< The smallest supported wavefront size.
            uint32 maxWavefrontSize;        ///< All powers of two between the min size and max size are supported.
            uint32 numAvailableSgprs;       ///< Number of available SGPRs.
            uint32 sgprsPerSimd;            ///< Number of physical SGPRs per SIMD.
            uint32 minSgprAlloc;            ///< Minimum number of SGPRs that can be allocated by a wave.
            uint32 sgprAllocGranularity;    ///< SGPRs are allocated in groups of this size.  Meaning, if your shader
                                            ///  only uses 1 SGPR, you will still end up reserving this number of
                                            ///  SGPRs.
            uint32 numAvailableVgprs;       ///< Number of available VGPRs.
            uint32 vgprsPerSimd;            ///< Number of physical VGPRs per SIMD.
            uint32 minVgprAlloc;            ///< Minimum number of VGPRs that can be allocated by a wave.
            uint32 vgprAllocGranularity;    ///< VGPRs are allocated in groups of this size.  Meaning, if your shader
                                            ///  only uses 1 VGPR, you will still end up reserving this number of
                                            ///  VGPRs. On hardware where wave32 is available, the granularity for a
                                            ///  wave64 shader is half of this value, but the VGPR allocation is
                                            ///  double. The same number of total physical registers is allocated for
                                            ///  each unit of allocation with either wave size.
            uint32 ldsSizePerCu;            ///< Local Data Store size available in bytes per CU.
            uint32 ldsSizePerThreadGroup;   ///< Local Data Store size available in bytes per thread-group.
            uint32 ldsGranularity;          ///< Local Data Store allocation granularity expressed in bytes.
            uint32 gsPrimBufferDepth;       ///< Hardware configuration for the GS prim buffer depth.
            uint32 gsVgtTableDepth;         ///< Hardware configuration for the GS VGT table depth.
            uint32 numOffchipTessBuffers;   ///< Number of offchip buffers that are used for offchip tessellation to
                                            ///  pass data between shader stages.
            uint32 offchipTessBufferSize;   ///< Size of each buffer used for passing data between shader stages when
                                            ///  tessellation passes data using off-chip memory.
            uint32 tessFactorBufSizePerSe;  ///< Size of GPU's the tessellatio-factor buffer, per shader engine.
            uint32 tccSizeInBytes;          ///< Size of total L2 TCC cache in bytes.
            uint32 tcpSizeInBytes;          ///< Size of one L1 TCP cache in bytes. There is one TCP per CU.
            uint32 maxLateAllocVsLimit;     ///< Maximum number of VS waves that can be in flight without
                                            ///  having param cache and position buffer space.
            uint32 shaderPrefetchBytes;     ///< Number of bytes the SQ will prefetch, if any.
            uint32 gl1cSizePerSa;           ///< Size in bytes of GL1 cache per SA.
            uint32 instCacheSizePerCu;      ///< Size in bytes of instruction cache per CU/WGP.
            uint32 scalarCacheSizePerCu;    ///< Size in bytes of scalar cache per CU/WGP.
            uint32 numAvailableCus;         ///< Total number of CUs that are actually usable.
            uint32 numPhysicalCus;          ///< Count of physical CUs prior to harvesting.
            /// Mask of active pixel packers. The mask is 128 bits wide, assuming a max of 32 SEs and a max of 4 pixel
            /// packers (indicated by a single bit each) per SE.
            uint32 activePixelPackerMask[ActivePixelPackerMaskDwords];
            /// Mask of present, non-harvested CUs (Virtual Layout)
            uint32 activeCuMask[MaxShaderEngines][MaxShaderArraysPerSe];
        } shaderCore;                       ///< Properties of computational power of the shader engine.

    } gfxipProperties;

    struct
    {
        union
        {
            struct
            {
                uint32 supportTurboSync           :  1; ///< Whether TurboSync is supported by KMD
                uint32 enableUmdFpsCap            :  1; ///< Whether UMD FPS CAP enabled
                uint32 isCwgSupported             :  1; ///< KMD supports Creator Who Game (CWG) feature
                uint32 isGamingDriver             :  1; ///< KMD works in gaming mode
#if defined(_WIN32)
                uint32 canShareSemaphoreKmtHandle :  1; ///< OS supports sharing queue semaphores by KMT handle.
#else
                uint32 placeholder0               :  1;
#endif
                uint32 ifhModeEnabled             :  1; ///< Whether the IFH mode is enabled
                uint32 requireFrameEnd            :  1; ///< If the client must tag the last command buffer
                                                        ///  submission in each frame with a @ref CmdBufInfo with
                                                        ///  the frameEnd flag set.
                uint32 supportNativeHdrWindowing  :  1; ///< Support HDR presentation that does not require FSE.
                uint32 flipQueueSupportsDecodeDst :  1; ///< If set, Decode destination images are supported
                                                        ///  in the OS flip-queue.
                uint32 supportFreeMux             :  1; ///< Whether FreeMux is supported by KMD
                uint32 isDataCenterBoard          :  1; ///< Whether the current board in use is a Data Center board.
                                                        ///  This is meant to support a unified VDI/CG driver package.
#if defined(__unix__)
                uint32 hasPrimaryDrmNode          :  1; ///< Set if the device has a primary DRM node.
                uint32 hasRenderDrmNode           :  1; ///< Set if the device has a render DRM node.
#else
                uint32 placeholder1               :  2;
#endif
                uint32 forceAlignmentSupported    :  1; ///< If PalPublicSettings::hardwareBufferAlignmentMode
                                                        ///  has any effect.
                uint32 reserved                   : 18; ///< Reserved for future use.
            };
            uint32 u32All;                              ///< Flags packed as 32-bit uint.
        } flags;                                        ///< OS-specific property flags.

#if (PAL_KMT_BUILD)
        int32  luidHighPart;        ///< High 32 bits of the LUID (locally unique identifier) for this device.
        uint32 luidLowPart;         ///< Low 32 bits of the LUID (locally unique identifier) for this device.
        bool   supportNtHandle;     ///< Support export/import resource with NT handle.
        bool   supportNativefence;  ///< Expose the capability of native fence support to client in DeviceProperties.
#endif

        union
        {
            struct
            {
                uint32 support                 : 1;  ///< Support Timeline type semaphore.
                uint32 supportHostQuery        : 1;  ///< Support Timeline type semaphore host query.
                uint32 supportHostWait         : 1;  ///< Support Timeline type semaphore host wait.
                uint32 supportHostSignal       : 1;  ///< Support Timeline type semaphore host signal.
                uint32 supportWaitBeforeSignal : 1;  ///< Support Timeline type semaphore wait before signal.

                uint32 reserved                : 27; ///< Reserved for future use.
            };
            uint32 u32All;
        } timelineSemaphore;

#if defined(__unix__)
        bool   supportOpaqueFdSemaphore; ///< Support export/import semaphore as opaque fd in linux KMD.
        bool   supportSyncFileSemaphore; ///< Support export/import semaphore as sync file in linux KMD.
        bool   supportSyncFileFence;     ///< Support export/import fence as sync file in linux KMD.
#endif

        bool   supportQueuePriority;        ///< Support create queue with priority
        bool   supportDynamicQueuePriority; ///< Support set the queue priority through IQueue::SetExecutionPriority

#if (PAL_KMT_BUILD || PAL_AMDGPU_BUILD)
        bool   supportMemoryBudgetQuery;    ///< Support memory budget query through IDevice::QueryGpuMemoryBudgetInfo
#endif

#if defined(_WIN32)
        bool   supportArbitaryPrtMapUnmap;  ///< Support arbitary prt map unmap operation.
#endif

        uint32                     umdFpsCapFrameRate;   ///< The frame rate of the UMD FPS CAP
        VirtualDisplayCapabilities virtualDisplayCaps;   ///< Capabilities of virtual display, it's provided by KMD

        union
        {
            struct
            {
                uint32 supportDevice                  : 1;  ///< GPU time domain
                uint32 supportClockMonotonic          : 1;  ///< POSIX CLOCK_MONOTONIC time domain
                uint32 supportClockMonotonicRaw       : 1;  ///< POSIX CLOCK_MONOTONIC_RAW time domain
                uint32 supportQueryPerformanceCounter : 1;  ///< Windows Query Performance Counter time domain

                uint32 reserved                       : 28; ///< Reserved for future use.
            };
            uint32 u32All;
        } timeDomains;

#if defined(_WIN32)
        char driverStorePath[Util::MaxPathStrLen];    ///< Driver store path
#endif

#if defined(__unix__)
        int64 primaryDrmNodeMajor; ///< DRM primary node major number.
        int64 primaryDrmNodeMinor; ///< DRM primary node minor number.
        int64 renderDrmNodeMajor;  ///< DRM render node major number.
        int64 renderDrmNodeMinor;  ///< DRM render node minor number.
#endif
        union
        {
            struct
            {
                uint32 supportPostflip   : 1;  ///< KMD support DirectCapture post-flip access
                uint32 supportPreflip    : 1;  ///< KMD support DirectCapture pre-flip access
                uint32 supportRSync      : 1;  ///< KMD support RSync
                uint32 maxFrameGenRatio  : 4;  ///< Maximum frame generation ratio or zero if not supported
                uint32 supportNonPrimary : 1;  ///< KMD support non-primary DirectCapture auxiliary data
                uint32 reserved          : 24; ///< Reserved for future use.
            };
            uint32 u32All;
        } directCapture;
    } osProperties;                 ///< OS-specific properties of this device.

    struct
    {
        uint32 domainNumber;                ///< PCI bus number.
        uint32 busNumber;                   ///< PCI bus number.
        uint32 deviceNumber;                ///< PCI device number.
        uint32 functionNumber;              ///< PCI function number.

        union
        {
            struct
            {
                uint32 gpuConnectedViaThunderbolt :  1; ///< Device is an externally housed GPU connected to the system
                                                        ///  via Thunderbolt. This will drastically impact CPU read and
                                                        ///  write performance of memory in the @ref GpuHeapLocal heap.
                uint32 gpuEmulatedInSoftware      :  1; ///< Device is really a software package which emulates the
                                                        ///  GPU.  This is meant for pre-silicon development.
                uint32 gpuEmulatedInHardware      :  1; ///< Device is a hardware emulated GPU.  This is meant for
                                                        ///  pre-silicon development.
                uint32 gpuVirtualization          :  1; ///< Set if running under VM.
                uint32 atomicOpsSupported         :  1; ///< Set if pcie atomic is supported.
                uint32 reserved                   : 27; ///< Reserved for future use.
            };
            uint32 u32All;                  ///< Flags packed as 32-bit uint.
        } flags;                            ///< PCI bus property flags.
    } pciProperties;                        ///< PCI bus properties of this device.

    BigSoftwareReleaseInfo bigSoftwareReleaseInfo;   ///< Big Software (BigSW) Release Version information
};

/// Defines callback function to notify client of private screen changes.
typedef void (PAL_STDCALL *TopologyChangeNotificationFunc)(void* pClient);

/// Defines callback function to notify client of the private screen removal.
typedef void (PAL_STDCALL *DestroyNotificationFunc)(void* pOwner);

/// Specifies the private screen topology change notification data.
struct PrivateScreenNotifyInfo
{
    void*                          pClient;       ///< Pointer to client, PAL use this pointer as parameter when PAL
                                                  ///  calls callback pfnOnTopology.
    TopologyChangeNotificationFunc pfnOnTopology; ///< Pointer to client provided function. PAL should call this when
                                                  ///  the topology change happens and let the client handle the change.
    DestroyNotificationFunc        pfnOnDestroy;  ///< Pointer to client provdided function. PAL should call this when
                                                  ///  a private screen object is to be destroyed. The pOwner data is
                                                  ///  passed at @ref IPrivateScreen::BindOwner() time.
};

/// Specifies fullscreen frame metadata control flags. Used for the KMD to notify clients about which types of frame
/// metadata it needs to send to KMD. The meaning depends on the context:
/// - During device finalization, client can set the flags indicating the specified metadata 'is supported' by client.
/// - During present, client can query these flags that indicate which metadata 'is enabled' currently so that the
///   client should send them to the KMD.
union FullScreenFrameMetadataControlFlags
{
    struct
    {
        uint32 timerNodeSubmission       :  1; ///< Timer node submission, used for cases such as FRTC/FP/PFPA.
        uint32 frameBeginFlag            :  1; ///< FrameBegin flag on CmdBufInfo, see CmdBufInfo for details.
        uint32 frameEndFlag              :  1; ///< FrameEnd flag on CmdBufInfo, see CmdBufInfo for details.
        uint32 primaryHandle             :  1; ///< Pending primary handle for pre-flip primary access (PFPA)
        uint32 p2pCmdFlag                :  1; ///< P2P copy command. See CmdBufInfo comments for details.
        uint32 forceSwCfMode             :  1; ///< Force software crossfire mode.
        uint32 postFrameTimerSubmission  :  1; ///< It indicates whether the timer node submission at frame N is to
                                               ///  synchronize the flip of frame N (postFrameTimerSubmission == TRUE)
                                               ///  or N+1 (postFrameTimerSubmission == FALSE).
                                               ///  It's only valid when timerNodeSubmission is also set.
        uint32 useHp3dForDwm             :  1; ///< KMD Informs (DX11) UMD to use HP3D for DWM or not (Output only).
        uint32 expandDcc                 :  1; ///< KMD notifies UMD to expand DCC (Output only).
        uint32 enableTurboSyncForDwm     :  1; ///< Indicates DWM should turn on TurboSync(Output only).
        uint32 enableDwmFrameMetadata    :  1; ///< When cleared, no frame metadata should be sent for DWM(Output only).
        uint32 flipIntervalOverride      :  3; ///< KMD-UMD interface FLIP_INTERVAL_OVERRIDE, for KMD to request flip
                                               ///  interval override from UMD.
        uint32 disableFreeMux            :  1; ///< KMD notifies UMD to disable FreeMux.
        uint32 maxFrameLatency           :  2; ///< KMD can notify UMD to override the frame latency of an app.
        uint32 sendMotionVectors         :  1; ///< Send the motion vector in CmdBufInfo once per frame
        uint32 sendDepth                 :  1; ///< Send the depth buffer in CmdBufInfo once per frame
        uint32 sendCameraMatrix          :  1; ///< Send the camera matrix in CmdBufInfo once per frame
        uint32 sendHudLessImage          :  1; ///< Send the HUD less image in CmdBufInfo once per frame
        uint32 reserved                  : 11; ///< Reserved for future use.

    };
    uint32 u32All;    ///< Flags packed as 32-bit uint.
};

/// Indicates the desired UMD behavior with timer node submission.
/// This is used to distinguish FP Vsync On + FreeSync Off case from HSync or FreeSync cases, the former case doesn't
/// hold flip while the later cases do.
enum class TimerNodeMode : uint32
{
    Unspecified,       ///< Unspecified, client can decide what to do with the timer submission.
    ForceFlipHold,     ///< Client must hold flip with the timer submission
};

/// Specifies fullscreen frame metadata control data. Including FullScreenFrameMetadataControlFlags plus extended data.
/// According to KMD's design, the difference is that 'flags' can be used to indicate 'client caps' during device
/// initialization, while the 'data' is only passed from KMD to UMD.
struct PerSourceFrameMetadataControl
{
    FullScreenFrameMetadataControlFlags flags;                   ///< The frame metadata control flags
    TimerNodeMode                       timerNodeSubmissionMode; ///< Desired UMD behavior with timer node submission
};

/// Specifies the texture optimization level to use for an image.
///
/// @ingroup ResourceBinding
enum class ImageTexOptLevel : uint32
{
    Default = 0,        ///< Use device default setting
    Disabled,           ///< Disable texture filter optimization
    Enabled,            ///< Enable texture filter optimization
    Maximum,            ///< Maximum texture filter optimization
    Count
};

/// Specifies properties for @ref IDevice finalization.  Input structure to IDevice::Finalize().
struct DeviceFinalizeInfo
{
    union
    {
        struct
        {
            uint32 supportPrivateScreens        :  1; ///< Initializes private screen support.
            uint32 requireFlipStatus            :  1; ///< Requires to initialize flip status shared memory
            uint32 requireFrameMetadata         :  1; ///< Requires to initialize frame metadata flags shared memory.
                                                      ///  Clients should only set this flag on the master device in an
                                                      ///  LDA chain.
            uint32 internalGpuMemAutoPriority   :  1; ///< Forces internal GPU memory allocation priorities to be
                                                      ///  determined automatically. It is an error to set this flag
                                                      ///  if the device does not report that it supports this feature.
            uint32 reserved                     : 28; ///< Reserved for future use.
        };
        uint32 u32All;                    ///< Flags packed as 32-bit uint.
    } flags;                              ///< Device finalization flags.

    /// Specifies which engines of each type should be created for the device.
    struct
    {
        uint32 engines; ///< A mask of which engines are requested.
    } requestedEngineCounts[EngineTypeCount];

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 914
    /// Bytes of CE RAM to be used by the client for each engine type. This value must be <= ceRamSizeAvailable reported
    /// for that engine type. In the case where more than one engine of a given type is requested it is assumed each
    /// engine of that type will use this amount of CE RAM so the total size of (ceRamSizeUsed * queueCounts) must be <=
    /// ceRamSizeAvailable for that engine type. Each entry must be either zero or a multiple of 32 bytes.
    size_t ceRamSizeUsed[EngineTypeCount];
#endif

    /// @see PrivateScreenNotifyInfo
    /// Private screen notify info, must be filled when supportPrivateScreens=1. The client pointer and callback are to
    /// be saved in device. PAL should call the callback when there is any topology (hotplug) change with the client
    /// pointer as parameter.
    PrivateScreenNotifyInfo  privateScreenNotifyInfo;

    /// Fullscreen frame metadata control flags indicating the types of metadata that the client supports.
    /// During adapter initialization, capable KMD notifies clients that it supports frame metadata,
    /// clients should then set these flags on device finalization info, indicating which types of metadata the client
    /// supports.
    FullScreenFrameMetadataControlFlags supportedFullScreenFrameMetadata;

    /// Specify the texture optimization level which only applies to internally-created views by PAL (e.g., for BLTs),
    /// client-created views must use the texOptLevel parameter in ImageViewInfo.
    ImageTexOptLevel internalTexOptLevel;
};

/// Reports the compatibility and available features when using two particular devices in a multi-GPU system.  Output
/// structure from IDevice::GetMultiGpuCompatibility().
struct GpuCompatibilityInfo
{
    union
    {
        struct
        {
            uint32 gpuFeatures         :  1;  ///< The devices have an exact feature match: same internal tiling, same
                                              ///  pipeline binary data, etc.
            uint32 iqMatch             :  1;  ///< Devices produce images with same precision.
            uint32 peerTransferWrite   :  1;  ///< Peer-to-peer transfers write are supported.  See
                                              ///  IDevice::OpenPeerMemory() and IDevice::OpenPeerImage().
            uint32 peerTransferRead    :  1;  ///< Peer-to-peer transfers based on xmgi are supported.
                                              ///  See IDevice::OpenPeerMemory() and IDevice::OpenPeerImage().
            uint32 sharedMemory        :  1;  ///< Devices can share memory objects with.  IDevice::OpenSharedMemory().
            uint32 sharedSync          :  1;  ///< Devices can share queue semaphores with
                                              ///  IDevice::OpenSharedQueueSemaphore().
            uint32 shareThisGpuScreen  :  1;  ///< Either device can present to this device.  Means that the device
                                              ///  indicated by the otherDevice param in
                                              ///  IDevice::GetMultiGpuCompatibility() can present to the device the
                                              ///  method was called on.
            uint32 shareOtherGpuScreen :  1;  ///< Either device can present to the other device.  Means that the
                                              ///  device IDevice::GetMultiGpuCompatibility() was called on can present
                                              ///  to the GPU indicated by the otherGpu param.
            uint32 peerEncode          :  1;  ///< whether encoding HW can access FB memory of remote GPU in chain
            uint32 peerDecode          :  1;  ///< whether decoding HW can access FB memory of remote GPU in chain
            uint32 peerTransferProtected : 1; ///< whether protected content can be transferred over P2P
            uint32 crossGpuCoherency   :  1;  ///< whether remote FB memory can be accessed without need for cache flush
            uint32 reserved            : 20;  ///< Reserved for future use.
        };
        uint32 u32All;                        ///< Flags packed as 32-bit uint.
    } flags;                                  ///< GPU compatibility flags.
};

/// Reports properties of a GPU memory heap.
///
/// @note The performance ratings represent an approximate memory throughput for a particular access scenario, but
///       should not be taken as an absolute performance metric.
struct GpuMemoryHeapProperties
{
    union
    {
        struct
        {
            uint32 cpuVisible       :  1;  ///< Accessible with IGpuMemory::Map()
            uint32 cpuGpuCoherent   :  1;  ///< Cache coherent between the CPU and GPU.
            uint32 cpuUncached      :  1;  ///< Not cached by CPU, but could still be GPU cached.
            uint32 cpuWriteCombined :  1;  ///< CPU write-combined memory.
            uint32 holdsPinned      :  1;  ///< GPU memory objects created by IDevice::CreatePinnedGpuMemory() are in
                                           ///  this heap.
            uint32 shareable        :  1;  ///< GPU memory objects in this heap can be shared between multiple devices.
            uint32 supportsTmz      :  1;  ///< This heap supports TMZ allocations.
            uint32 reserved         : 25;  ///< Reserved for future use.
        };
        uint32 u32All;                     ///< Flags packed as 32-bit uint.
    } flags;                               ///< GPU memory heap property flags.

    gpusize logicalSize;                   ///< Size of the heap in bytes. If HBCC is enabled, certain heaps may be
                                           ///  virtualized and the logical size will exceed the physical size.
    gpusize physicalSize;                  ///< Physical size of the heap in bytes
};

/// Reports properties of a specific GPU block required for interpretting performance experiment data from that block.
/// See @ref PerfExperimentProperties.
struct GpuBlockPerfProperties
{
    bool   available;               ///< If performance data is available for this block.
    uint32 instanceCount;           ///< How many instances of this block are in the device.
    uint32 maxEventId;              ///< Maximum event ID for this block.
    uint32 maxGlobalOnlyCounters;   ///< Number of counters available only for global counts.
    uint32 maxGlobalSharedCounters; ///< Total counters available including state shared between global and SPM.
    uint32 maxSpmCounters;          ///< Counters available for streaming only.

    /// If the instance group size is equal to one, every block instance has its own independent counter hardware.
    /// PAL guarantees this is true for all non-DF blocks.
    ///
    /// Otherwise the instance group size will be a value greater than one which indicates how many sequential
    /// instances share the same counter hardware. The client must take care to not enable too many counters within
    /// each of these groups.
    ///
    /// For example, the DfMall block may expose 16 instances with 8 global counters but define a group size of 16.
    /// In that case all instances are part of one massive group which uses one pool of counter state such that no
    /// combination of DfMall counter configurations can exceed 8 global counters.
    uint32 instanceGroupSize;
};

/// Reports performance experiment capabilities of a device.  Returned by IDevice::GetPerfExperimentProperties().
struct PerfExperimentProperties
{
    PerfExperimentDeviceFeatureFlags features; ///< Performance experiment device features.

    size_t maxSqttSeBufferSize;   ///< SQTT buffer size per shader engine.
    size_t sqttSeBufferAlignment; ///< SQTT buffer size and base address alignment.
    uint32 shaderEngineCount;     ///< Number of shader engines.

    /// Reports availability and properties of each device block.
    GpuBlockPerfProperties blocks[static_cast<size_t>(GpuBlock::Count)];
};

/// Reports maximum alignments for images created with a @ref ImageTiling::Linear tiling mode assuming the images'
/// elements are no larger than maxElementSize.
struct LinearImageAlignments
{
    uint16 maxElementSize; ///< Maximum element size in bytes.
    uint16 baseAddress;    ///< Minimum required base address alignment in bytes.
    uint16 rowPitch;       ///< Minimum required row pitch alignment in bytes.
    uint16 depthPitch;     ///< Minimum required depth pitch alignment in bytes.
};

/// Specifies image view type (i.e., 1D, 2D, 3D, or cubemap).
///
/// @ingroup ResourceBinding
enum class ImageViewType : uint32
{
    Tex1d    = 0x0,
    Tex2d    = 0x1,
    Tex3d    = 0x2,
    TexCube  = 0x3,

    Count
};

/// Enumeration which defines the mode for magnification and minification sampling
///
/// @ingroup ResourceBinding
enum XyFilter : uint32
{
    XyFilterPoint = 0,          ///< Use single point sampling
    XyFilterLinear,             ///< Use linear sampling
    XyFilterAnisotropicPoint,   ///< Use anisotropic with single point sampling
    XyFilterAnisotropicLinear,  ///< Use anisotropic with linear sampling
    XyFilterCount
};

/// Enumeration which defines the mode for volume texture sampling
///
/// @ingroup ResourceBinding
enum ZFilter : uint32
{
    ZFilterNone = 0,            ///< Disable Z filtering
    ZFilterPoint,               ///< Use single point sampling
    ZFilterLinear,              ///< Use linear sampling
    ZFilterCount
};

/// Enumeration which defines the mode for mip-map texture sampling
///
/// @ingroup ResourceBinding
enum MipFilter : uint32
{
    MipFilterNone = 0,          ///< Disable Mip filtering
    MipFilterPoint,             ///< Use single point sampling
    MipFilterLinear,            ///< Use linear sampling
    MipFilterCount
};

/// Specifies parameters for an image view descriptor controlling how a given texture is sampled
///
/// @ingroup ResourceBinding
struct TexFilter
{
    union
    {
        struct
        {
            uint32 magnification : 2;  ///< Used with enum XyFilter for Plane magnification filtering
            uint32 minification  : 2;  ///< Used with enum XyFilter for Plane minification filtering
            uint32 zFilter       : 2;  ///< Used with enum ZFilter for volume texture filtering
            uint32 mipFilter     : 2;  ///< Used with enum MipFilter for mip-map filtering
            uint32 reserved      : 24; ///< Reserved for future use
        };
        uint32 u32All;                 ///< Value of flags bitfield
    };
};

/// Determines if "TexFilter" should be ignored or not.
enum class TexFilterMode : uint32
{
    Blend = 0x0, ///< Use the filter method specified by the TexFilter enumeration
    Min   = 0x1, ///< Use the minimum value returned by the sampler, no blending op occurs
    Max   = 0x2, ///< Use the maximum value returned by the sampler, no blending op occurs
    Count
};

/// Specifies how texture coordinates outside of texture boundaries are interpreted.
///
/// @ingroup ResourceBinding
enum class TexAddressMode : uint32
{
    Wrap                    = 0x0,  ///< Repeat the texture.
    Mirror                  = 0x1,  ///< Mirror the texture by flipping it at every other coordinate interval.
    Clamp                   = 0x2,  ///< Clamp the texture to the texture's edge pixel.
    MirrorOnce              = 0x3,  ///< Mirror the texture once then clamp.
    ClampBorder             = 0x4,  ///< Clamp the texture to the border color specified in the sampler.
    MirrorClampHalfBorder   = 0x5,  ///< Mirror the texture once then clamp the texture to half of the edge color.
    ClampHalfBorder         = 0x6,  ///< Clamp the texture to half of the edge color.
    MirrorClampBorder       = 0x7,  ///< Mirror the texture once then clamp the texture to the samler's border color.
    Count
};

/// Specifies how a border color should be chosen when the TexAddressClampBorder texture addressing is used by a
/// sampler.
///
/// @ingroup ResourceBinding
enum class BorderColorType : uint32
{
    White            = 0x0,  ///< White border color (1.0, 1.0, 1.0, 1.0).
    TransparentBlack = 0x1,  ///< Transparent black border color (0.0, 0.0, 0,0, 0.0).
    OpaqueBlack      = 0x2,  ///< Opaque black border color (0.0, 0.0, 0.0, 1.0).
    PaletteIndex     = 0x3,  ///< Fetch border color from the border color palette.
    Count
};

/// Residency maps are helper surfaces used in conjunction with PRT+.  They reflect the resident mip levels
/// associated with a given UV region of the parent image.
enum class PrtMapAccessType : uint32
{
    Raw                 = 0x0, ///< Read / write the map image as a normal image.
    Read                = 0x1, ///< Read the residency map as floating point data
    WriteMin            = 0x2, ///< Write the residency map with min(existing,new)
    WriteMax            = 0x3, ///< Write the residency map with max(existing,new)
    WriteSamplingStatus = 0x4, ///< Write to the sampling status map.
    Count
};

/// Specifies compression behavior for an IImage or image/buffer view.
enum class CompressionMode : uint32
{
    Default                = 0,  ///< Let PAL choose behavior.
    ReadEnableWriteEnable  = 1,  ///< Override default to force read and write compression on.
    ReadEnableWriteDisable = 2,  ///< Support reading compressed data, but force any writes to be uncompressed (keeping
                                 ///  physical metadata consistent).
    ReadBypassWriteDisable = 3,  ///< Bypass physical metadata on reads (assume decompressed), all writes will be
                                 ///  uncompressed and will write physical metatdata marking updated blocks as being
                                 ///  uncompressed. This mode is intended to handle placed resources that do not
                                 ///  want compression in memory allocations that have distributed compression enabled.
                                 ///  WARNING: Using this mode to read compressed data will result in corruption.
    Count,
};

/// Specifies client compression behavior for an IImage.
enum class ClientCompressionMode : uint32
{
    Default          = 0,  ///< Let implementation decide whether to enable or disable
    Enable           = 1,  ///< Force enable
    Disable          = 2,  ///< Force disable
    DisableClearOnly = 3,  ///< Force enable for all image views except for image clears
    Count,
};

/// Specifies parameters for a buffer view descriptor that control how a range of GPU memory is viewed by a shader.
///
/// Input to either CreateTypedBufferViewSrds() or CreateUntypedBufferViewSrds().  Used for any buffer descriptor,
/// including read-only shader resources, UAVs, vertex buffers, etc.  The usage of stride and format depends on the
/// expected shader instruction access:
///
/// + _Typed buffer_ access must set a valid format and channel mapping.
/// + _Raw buffer_ access is indicated by setting an invalid format and setting stride to 1.
/// + _Structured buffer_ access is indicated by setting an invalid format and setting stride to any value except 1.  A
///   stride of 0 maps all view accesses to the first structure stored in memory.
///
/// _Typed buffer_ SRD's must be created using @ref IDevice::CreateTypedBufferViewSrds().
/// _Raw buffer_ and _structured buffer_ SRD's must be created using @ref IDevice::CreateUntypedBufferViewSrds().
///
/// If necessary, PAL will adjust the out of bounds read/write behavior to match the client's API requirements based on
/// the client defines - PAL_CLIENT_VULKAN, etc.
///
/// @ingroup ResourceBinding
struct BufferViewInfo
{
    gpusize         gpuAddr;         ///< GPU memory virtual address where the buffer view starts, in bytes.
                                     ///  Must be aligned to bytes-per-element for typed access.
    gpusize         range;           ///< Restrict the buffer view to this many bytes.  Will be rounded down to a
                                     ///< multiple of the stride.
    gpusize         stride;          ///< Stride in bytes.  Must be aligned to bytes-per-element for typed access.
    SwizzledFormat  swizzledFormat;  ///< Format and channel swizzle for typed access. Must be Undefined for structured
                                     ///  or raw access.
    CompressionMode compressionMode; ///< Specify GFX12-style distributed compression mode override for this view.
                                     ///  Only relevant if the backing memory pages enable compression.

    union
    {
        struct
        {
            /// Set to have this surface independently bypass the MALL for read and / or write operations.
            /// If set, this overrides the GpuMemMallPolicy specified at memory allocation time.  Meaningful
            /// only on GPUs that have supportsMall set in DeviceProperties.
            uint32 bypassMallRead  : 1;
            uint32 bypassMallWrite : 1;
            uint32 reserved        : 30; ///< Reserved for future use
        };
        uint32 u32All;                   ///< Value of flags bitfield
    } flags;
};

/// D3D12_VERTEX_BUFFER_VIEW structure. Match vertex offset mode.
/// @ref ICmdBuffer::CmdSetVertexBuffers function.
struct VertexBufferView
{
    gpusize gpuva;         ///< GPU virtual address to vertex buffer.
    uint32  sizeInBytes;   ///< The size in bytes of the buffer.
    uint32  strideInBytes; ///< The size in bytes of each vertex entry.
};

/// Specifies an array of vertex buffer view descriptors. PAL supports multiple vertex buffer modes which change how
/// the GPU memory is viewed by a shader. This struct wraps up the choice between these modes.
///
/// If offsetMode is true PAL will create "vertex offset mode" SRDs using the @ref VertexBufferView array. This new SRD
/// has stronger robustness. It avoids that the last vertex is removed when buffer size is not multiple of the stride.
///
/// @see ICmdBuffer::CmdSetVertexBuffers
struct VertexBufferViews
{
    uint32 firstBuffer; ///< First vertex buffer slot to change. Must be less than @ref MaxVertexBuffers.
    uint32 bufferCount; ///< The length of the buffer view array.
    bool   offsetMode;  ///< If true, pVertexBufferViews must be used, otherwise pBufferViewInfos is used.

    union
    {
        const VertexBufferView* pVertexBufferViews; ///< Offset mode vertex buffer views. Valid if offsetMode is true.
        const BufferViewInfo*   pBufferViewInfos;   ///< Index mode vertex buffer views. Valid if offsetMode is false.
    };
};

/// Specifies parameters for an image view descriptor controlling how a shader will view the specified image.
///
/// Input to CreateImageViewSrd().  Used for any image view descriptor, including read-only shader resources and UAVs.
///
/// @ingroup ResourceBinding
struct ImageViewInfo
{
    const IImage*  pImage;         ///< Image associated with the view.
    ImageViewType  viewType;       ///< 1D, 2D, 3D, or Cubemap.  Typically this should match the image type, but a
                                   ///  Cubemap view can be imposed on a 2D array image.
    SwizzledFormat swizzledFormat; ///< Specifies the image view format and channel swizzle. Must be compatible (same
                                   ///  bit-widths per channel) with the image's base format.
                                   ///  @note: YUV formats are invalid for an ImageView. A format should be chosen to be
                                   ///  compatible with either the luma or chroma plane(s) of the YUV format.
    SubresRange    subresRange;    ///< Specifies a subset of subresources to include in the view.  If the base Image
                                   ///  has a YUV planar format, the number of array slices in the range must be 1.
                                   ///  If zRange feature is used, the number of mips in the range must be 1.
    float          minLod;         ///< Minimum mip level of detail to use for this view.

    uint32         samplePatternIdx;  ///< Index into the currently bound MSAA sample pattern palette to be
                                      ///  read/evaluated when samplepos shader instructions are executed on this
                                      ///  view.  Can be ignored if the samplepos shadinstruction will not be used.
                                      ///  Must be less than MaxSamplePatternPaletteEntries.  See
                                      ///  IDevice::SetSamplePatternPalette().
    Range          zRange;            ///< Specifies the z offset and z range.

    ImageTexOptLevel texOptLevel;     ///< Specific the texture optimization level.

    const IImage*    pPrtParentImg;   ///< Meaningful only if "mapAccess" is not "raw".
    PrtMapAccessType mapAccess;       ///< Type of access to be done if "pImage" is a PRT+ meta-data image.
                                      ///  See @ref ImageCreateInfo

    ImageLayout possibleLayouts; ///< Union of all possible layouts this view can be in while accessed by this view.
                                 ///  (ie. what can be done with this SRD without having a layout transition?)
                                 ///  In DX, for example, it's possible that a texture SRV could be accessed in a state
                                 ///  with all other read-only usages allowed, but a UAV must exclusively be accessed
                                 ///  in the UNORDERED_ACCESS state.
                                 ///  The primary purpose of this flag is to avoid compressed shader writes if a
                                 ///  different usage does not support compression and PAL won't get an opportunity to
                                 ///  decompress it (ie. a transition in a barrier)

    CompressionMode compressionMode; ///< Specify GFX12-style distributed compression mode override for this view.
                                     ///  Only relevant if the backing IImage resource and its bound memory pages
                                     ///  enable compression. ReadBypassWriteDisable is only valid if compressionMode
                                     ///  in ImageCreateInfo disables compressed write.

    union
    {
        struct
        {
            /// Set to have this surface independently bypass the MALL for read and / or write operations.
            /// If set, this overrides the GpuMemMallPolicy specified at memory allocation time.  Meaningful
            /// only on GPUs that have supportsMall set in DeviceProperties.
            uint32 bypassMallRead  : 1;
            uint32 bypassMallWrite : 1;

            uint32 zRangeValid     : 1;  ///< whether z offset/ range value is valid.
            uint32 includePadding  : 1;  ///< Whether internal padding should be included in the view range.

            uint32 reserved        : 28; ///< Reserved for future use
        };
        uint32 u32All;                   ///< Value of flags bitfield
    } flags;                             ///< Image view flags.
};

/// Specifies parameters controlling execution of sample instructions in a shader.  Input to CreateSamplerSrd().
///
/// @ingroup ResourceBinding
struct SamplerInfo
{
    TexFilterMode   filterMode;               ///< Min/max filtering modes
    TexFilter       filter;                   ///< Filtering to apply to texture fetches.
    TexAddressMode  addressU;                 ///< Addressing mode for U texture coords outside of the [0..1] range.
    TexAddressMode  addressV;                 ///< Addressing mode for V texture coords outside of the [0..1] range.
    TexAddressMode  addressW;                 ///< Addressing mode for W texture coords outside of the [0..1] range.
    float           mipLodBias;               ///< Bias for mipmap level of detail selection.
    uint32          maxAnisotropy;            ///< Anisotropy value clamp when the filter mode is TexFilterAnisotropic.
    CompareFunc     compareFunc;              ///< Comparison function to apply to fetched data.
    float           minLod;                   ///< High-resolution mipmap LOD clamp.
    float           maxLod;                   ///< Low-resolution mipmap LOD clamp.
    BorderColorType borderColorType;          ///< Selects border color when an address mode is TexAddressClampBorder.
    uint32          borderColorPaletteIndex;  ///< Choose color from the border color palette when borderColorType is
                                              ///  BorderColorPalette.
    float anisoThreshold;                     ///< Opt-in, flags.useAnisoThreshold == 1 and flags.preciseAniso == 0.
                                              ///  The value should be computed taking account the maxAnisotropy
                                              ///  setting. This is a high resolution value which is quantized and
                                              ///  clamped down to 3 bits to the domain [0.0, 0.875] for current Hw.
                                              ///  We can interpret the functioning of the threshold value as follows.
                                              ///  maxAnisotropy per-pixel can be 1, 2, 4, 8 or 16 (N).
                                              ///  During sampling, the initial count (or S) is computed in Hw for each
                                              ///  quad and the domain for current Hw is [0-16].
                                              ///  Final sample count = min(pow(2, ceil(log2(S - anisoThreshold))), N)
                                              ///  Note: when flags.useAnisoThreshold == 0, Pal will ignore this value
                                              ///  and instead use a maximum of 0.25 at the highest anisotropic setting.
                                              ///  It is important to be aware that this feature tunes quality vs
                                              ///  performance, so care should be taken to not degrade image quality
                                              ///  'noticeably' when enabling using this feature
    uint32 perfMip;                           ///< Controls the value of the PERF_MIP field in Sampler SRD's.
                                              ///  This field basically controls the Fractional part of the LOD
                                              ///  calculation. if LOD is fractional so let us say 1.23, in this case
                                              ///  you must avg.out your samples from both MIP 1 and 2.But if PERF_MIP
                                              ///  is set to nonzero the HW will perform an optimization and may fetch
                                              ///  from only 1 MIP.

    // These values are used to define a filtering line used when sampling a residency map.  The defined
    // slopes in both the X (U) and Y (V) directions are to avoid visible disconnects when sampling between
    // different samples.
    Offset2d   uvOffset;                      ///< u/v offset value selectors.  Values specified are in
                                              ///  log2 of fractions of pixel.  i.e., 1 / (1 << x).  Not all values
                                              ///  are supported by all HW.
    Offset2d   uvSlope;                       ///< u/v slope value selectors.  Supported slope values are
                                              ///  specified in degrees.  In the case of a 3D image, the supplied
                                              ///  uvSlope.y is interpreted as wSlope.
                                              ///      0   2.5
                                              ///      1   3
                                              ///      2   4
                                              ///      3   5
                                              ///      4   8
                                              ///      5   16
                                              ///      6   32
                                              ///      7   64
                                              ///      other values:  unsupported

    union
    {
        struct
        {
            uint32 mgpuIqMatch         : 1;  ///< Enables image compatibility for MGPU scenarios where paired devices
                                             ///  come from different hardware families.
            uint32 preciseAniso        : 1;  ///< Anisotropic filtering should prefer precision over speed.
            uint32 unnormalizedCoords  : 1;  ///< If set then always use unnormalized texture coordinates instead of
                                             ///  zero to one.  Only works under certain conditions (no mip filtering,
                                             ///  no computed LOD, no offsets, only edge or border clamp address modes)
            uint32 truncateCoords      : 1;  ///< If set then hardware will truncate mantissa instead of
                                             ///  rounding to nearest even in float point to fixed point
                                             ///  texture coordinate conversion
            uint32 seamlessCubeMapFiltering : 1;  ///< If set then there's filtering across the edges of the cube map.
            uint32 prtBlendZeroMode    : 1;  ///< Allow unmapped PRT texels to be treated as zero and blended with
                                             ///  mapped texels. If set to 0, the destination of the sample instruction
                                             ///  is written with all 0s when TFE == 0; if set to 1, Treat unmapped
                                             ///  texels as zeros and blend them with other mapped texels, write the
                                             ///  result of this sample instruction to the destination GPRs.
            uint32 useAnisoThreshold   : 1;  ///< If set, Hw will use the value assigned in anisoThreshold, but
                                             ///  only if preciseAniso is set to 0, also.

            /// This allows the sampler to turn off overriding anisotropic filtering when the resource view contains a
            /// single mipmap level.  Not all graphics IP supports overriding anisotropic filtering, and this flag will
            /// be ignored for such GPUs.
            uint32 disableSingleMipAnisoOverride : 1;

            uint32 forResidencyMap     : 1; ///< Set if the surface being sampled is a residency map used in PRTs.
                                            ///  Only meaningful if the corresponding ImageView's mapAccess is set to
                                            ///  "read". Only valid for devices that report the "PrtFeaturePrtPlus"
                                            ///  flag.
            uint32 reserved            : 23; ///< Reserved for future use
        };
        uint32 u32All;                ///< Value of flags bitfield
    } flags;
};

/// Specifies which heuristic should be utilized for sorting children when box sorting is enabled
enum class BoxSortHeuristic : uint32
{
    ClosestFirst    = 0x0,  ///< Traversal is ordered to enter the children that
                            ///< intersect the ray closer to the ray origin first.
                            ///< This is good baseline option. Default option for RT IP 1.x.
    LargestFirst    = 0x1,  ///< Traversal is ordered to enter the children that have the largest
                            ///< interval where the box intersects the ray first.
                            ///< Good for shadow rays with terminate on first hit.
    ClosestMidPoint = 0x2,  ///< Traversal is ordered to enter the children that have a midpoint in the interval
                            ///< where the box intersects that has the lowest intersection time before clamping(
                            ///< Good for reflection rays.
    Disabled        = 0x3,  ///< Box sort and heuristic are disabled.
    Count
};

/// Specifies parameter for creating a BvH (bounding volume hierarchy, used by ray-trace) descriptor
struct BvhInfo
{
    const IGpuMemory*  pMemory;      ///< Memory object holding the BVH nodes
    gpusize            offset;       ///< Offset from memory address specified by pMemory.  Combination of
                                     ///  pMemory address and the offset must be 256 byte aligned.
    gpusize            numNodes;     ///< Number of nodes in the view
    uint32             boxGrowValue; ///< Number of ULPs (unit in last place) to be added during ray-box test.

    BoxSortHeuristic   boxSortHeuristic;   ///< Specifies which heuristic should be utilized for
                                           ///< sorting children when box sorting is enabled
    union
    {
        struct
        {
            uint32 useZeroOffset        :  1; ///< If set, SRD address is programmed to zero
            uint32 returnBarycentrics   :  1; ///< When enabled, ray intersection will return triangle barycentrics.
                                              ///  Note: Only valid if @see supportIntersectRayBarycentrics is true.

            /// Set to have this surface independently bypass the MALL for read and / or write operations.
            /// If set, this overrides the GpuMemMallPolicy specified at memory allocation time.  Meaningful
            /// only on GPUs that have supportsMall set in DeviceProperties.
            uint32 bypassMallRead       :  1;
            uint32 bypassMallWrite      :  1;
            uint32 pointerFlags         :  1; ///< If set, flags are encoded in the node pointer bits
            uint32 highPrecisionBoxNode :  1; ///< If set, enable 64-byte high precision box node
            uint32 wideSort             :  1; ///< If set, enable wide sort
            uint32 hwInstanceNode       :  1; ///< If set, enable hardware instance node
            uint32 sortTrianglesFirst   :  1; ///< If set, triangle nodes are treated specially during child sorting
            uint32 compressedFormatEn   :  1; ///< If set, enable compressed format support. This include enable
                                              ///  support for compressed primitive packets, BVH8-128B box nodes,
                                              ///  and changes to triangle intersection test return data.
            uint32 reserved             : 22; ///< Reserved for future HW
        };

        uint32  u32All; ///< Flags packed as 32-bit uint.
    } flags;            ///< BVH creation flags.
};

/// Specifies parameters for an fmask view descriptor.
///
/// Input to CreateFmaskViewSrd().  Allows the client to access fmask from a shader using the load_fptr IL instruction.
///
/// @ingroup ResourceBinding
struct FmaskViewInfo
{
    const IImage* pImage;          ///< Image associated with the fmask view.
    uint32        baseArraySlice;  ///< First slice in the view.
    uint32        arraySize;       ///< Number of slices in the view.

    union
    {
        struct
        {
            uint32 shaderWritable : 1;  ///< True if used with an image that has been transitioned to a shader-
                                        ///  writable image state (e.g. [Graphics|Compute][WriteOnly|ReadWrite])
            uint32 reserved       : 31; ///< Reserved for future use
        };
        uint32     u32All;              ///< Value of flags bitfield
    } flags;                            ///< Fmask view flags
};

/// Element of the multisample pattern representing a sample position (X, Y), type of SamplePatternPalette, which
/// matches the layout defined by SC.
struct SamplePos
{
    float x;                  ///< x coordinate of sample position.
    float y;                  ///< y coordinate of sample position.
    uint32 reserved1;         ///< reserved for future use
    uint32 reserved2;         ///< reserved for future use
};

/// Specifies a palette of MSAA sample patterns used by the client.  Input to SetSamplePatternPalette, which is used
/// to implement samplepos shader instruction support.
typedef SamplePos SamplePatternPalette[MaxSamplePatternPaletteEntries][MaxMsaaRasterizerSamples];

/// Provides a GPU timestamp along with the corresponding CPU timestamps, for use in calibrating CPU and GPU timelines.
struct CalibratedTimestamps
{
    uint64 gpuTimestamp;                  ///< GPU timestamp value compatible with ICmdBuffer::CmdWriteTimestamp().
    uint64 cpuClockMonotonicTimestamp;    ///< POSIX CLOCK_MONOTONIC timestamp
    uint64 cpuClockMonotonicRawTimestamp; ///< POSIX CLOCK_MONOTONIC_RAW timestamp
    uint64 cpuQueryPerfCounterTimestamp;  ///< Windows QueryPerformanceCounter timestamp
    uint64 maxDeviation;                  ///< Maximum deviation in nanoseconds between the GPU and CPU timestamps
};

/// Specifies connector types
enum class DisplayConnectorType : uint32
{
    Unknown = 0,    ///< Unknown connector type
    Vga,            ///< VGA
    DviD,           ///< DVI_D
    DviI,           ///< DVI_I
    Hdmi,           ///< HDMI
    Dp,             ///< DP
    Edp,            ///< EDP
    Minidp,         ///< MINI_DP
    Count
};

/// Specifies properties for display connectors connected to GPU
struct DisplayConnectorProperties
{
    DisplayConnectorType type;        ///< Connector type - VGA, DVI, HDMI, DP etc
};

/// Specifies pre-defined power profile which is used to communicate with KMD/PPLib and set correspond power states.
enum class PowerProfile : uint32
{
    Default   = 0,      ///< Default power profile.
    VrCustom  = 1,      ///< Power profile used by custom VR scenario.
    VrDefault = 2,      ///< Power profile used by default VR scenario.
    Idle      = 3,      ///< Power profile used for forced DPM0, in case HMD is taken off but the game is still running.
    Count
};

/// Fine-grain power switch info.
struct PowerSwitchInfo
{
    uint32 time;        ///< Time in microseconds, relative to the frame start at V-sync. Clients should consider the
                        ///  powerSwitchLatency value reported in @ref PrivateScreenProperties when specifying
                        ///  switch times.

    uint32 performance; ///< Performance to be set (between 0-100), which is mapped to a certain DPM level by KMD.
};

/// Maximum number of power switch info allowed in one custom power profile.
static constexpr uint32 MaxNumPowerSwitchInfo = 5;

/// Fine-grain power management for dynamic power mode. This structure specifies multiple DPM states to be cycled
/// through each frame.
struct CustomPowerProfile
{
    IPrivateScreen*  pScreen;       ///< Dynamic power mode needs V-sync so a private screen object is needed.
    uint32           numSwitchInfo; ///< Number of discrete DPM states to cycle through per frame.  Number entries
                                    ///  in switchInfo[] and actualSwitchInfo[].

    PowerSwitchInfo  switchInfo[MaxNumPowerSwitchInfo];  ///< Specifies the set of power states to cycle through each
                                                         ///  frame. Each entry specifies an offset into the frame where
                                                         ///  the DPM state should be switched, and a rough performance
                                                         ///  requirement value which will be translated into an
                                                         ///  appropriate DPM state by KMD.
    PowerSwitchInfo  actualSwitchInfo[MaxNumPowerSwitchInfo]; ///< The actual set of power states that KMD/PPLib sets.
};

/// Flags for IDevice::AddGpuMemoryReferences().  Depending on their residency model, a client may set these flags as
/// directed by the application or hard-code them to a single value.  Driver-internal memory references should be marked
/// as CantTrim unless the client explicitly handles trim support.
///
/// Note that the CantTrim and MustSucceed flags are based on the same WDDM flags; it is expected that PAL will ignore
/// them on non-WDDM platforms.
enum GpuMemoryRefFlags : uint32
{
    GpuMemoryRefCantTrim    = 0x1, ///< The caller can't or won't free this allocation on OS request.
    GpuMemoryRefMustSucceed = 0x2, ///< Hint to the OS that we can't process a failure here, this may result in a TDR.
    GpuMemoryRefAllFlags    = 0x3  ///< Clients should NOT use it, for internal static_assert purpose only.
};

/// Specifies input arguments for IDevice::GetPrimaryInfo(). Client must specify a display ID and properties of the
/// primary surface that will drive that display in order to query capabilities.
struct GetPrimaryInfoInput
{
    uint32          vidPnSrcId;                 ///< Video present source id.
    uint32          width;                      ///< Primary surface width.
    uint32          height;                     ///< Primary surface height.
    SwizzledFormat  swizzledFormat;             ///< Format and swizzle of the primary surface.
    Rational        refreshRate;                ///< Video refresh rate, this is only valid if refreshRateValid is set.
    union
    {
        struct
        {
            uint32 qbStereoRequest              :  1;   ///< Going to set a stereo mode.
            uint32 refreshRateValid             :  1;   ///< Refresh rate is valid.
            uint32 freeSyncInCrossFireSupport   :  1;   ///< True if client supports FreeSync in CrossFire.
            uint32 useKmdCalcFramePacing        :  1;   ///< True if client uses KMD frame pacing. If so, the client
                                                        ///  creates a timer queue to delay the present, and the delay
                                                        ///  value is calculated by KMD.
            uint32 reserved                     : 28;   ///< Reserved for future use.
        };
        uint32 u32All;                      ///< Flags packed as 32-bit uint.
    } flags;                                ///< get primary surface info input flags.
};

/// Specifies output arguments for IDevice::GetStereoDisplayModes(), returning supported stereo mode
struct StereoDisplayModeOutput
{
    Extent2d          extent;           ///< Dimensions in pixels WxH.
    Rational          refreshRate;      ///< Refresh rate.
    SwizzledFormat    format;           ///< Format and swizzle of the primary surface.
};

/// Specifies output arguments for IDevice::GetActive10BitPackedPixelMode(), returning which, if any, 10-bit
/// display mode is active.
struct Active10BitPackedPixelModeOutput
{
    bool             isInWs10BitMode;           ///< Whether the workstation 10-bit feature is enabled.
    bool             notifyKmd10bitsPresent;    ///< When in 10-bit mode and at present time, if the
                                                ///  client driver sees a 10-bit to 8-bit surface blt,
                                                ///  it needs to call RequestKmdReinterpretAs10Bit() to
                                                ///  inform the KMD that the dst surface must be reinterpreted
                                                ///  as 10-bits per channel for all KMD-initiated BLTs.
    PackedPixelType  packedPixelType;           ///< Format of the packed pixels.
    uint32           pixelPackRatio;            ///< The number of 10-bit pixels that are packed into one 8-8-8-8
                                                ///  format pixel.
};

/// Specifies primary surface stereo mode.
enum StereoMode : uint32
{
    StereoModeHwAlignedViews = 0,  ///< The stereo views are HW aligned on the display.
    StereoModeSwPackedViews  = 1,  ///< The layout of the stereo views on the display are determined by the client.
    StereoModeNotSupported   = 2,  ///< Not support stereo mode
    StereoModeSideBySide     = 3,  ///< The two stereo views are put side by side on the display.
    StereoModeTopBottom      = 4   ///< One stereo view is on the top of the display, and the other is on the bottom.
};

/// Enumerates the supported workstation stereo modes.
enum class WorkstationStereoMode : uint32
{
    Disabled,
    ViaConnector,             ///< Active Stereo for 3 Pin VESA connector.
    ViaBlueLine,              ///< Blue line Active Stereo for laptops.
    Passive,                  ///< Passive Stereo (Dual head).
    PassiveInvertRightHoriz,  ///< Passive Stereo with Horizontal Invert (Dual Head).
    PassiveInvertRightVert,   ///< Passive Stereo with Vertical Invert (Dual Head).
    Auto,                     ///< Auto Stereo Vertical Interleaved.
    AutoHoriz,                ///< Auto Stereo Horizontal Interleaved.
    AutoCheckerboard,         ///< Auto Stereo Checkerboard Interleaved.
    AutoTsl,                  ///< Tridelity SL Auto Stereo.
    Count,
};

/// Specifies output arguments for IDevice::GetPrimaryInfo(), returning capabilitiy information for a display in
/// a particular mode.
struct GetPrimaryInfoOutput
{
    uint32          tilingCaps;                ///< Tiling caps supported by this primary surface.
    StereoMode      stereoMode;                ///< Stereo mode supported by this primary surface.
    uint32          mallCursorCacheSize;       ///< Size of the mall cursor cache in bytes
    union
    {
        struct
        {
            /// MGPU flag: this primary surface supports DVO HW compositing mode.
            uint32 dvoHwMode                    :  1;
            /// MGPU flag: this primary surface supports XDMA HW compositing mode.
            uint32 xdmaHwMode                   :  1;
            /// MGPU flag: this primary surface supports client doing SW compositing mode.
            uint32 swMode                       :  1;
            /// MGPU flag: this primary surface supports freesync.
            uint32 isFreeSyncEnabled            :  1;
            /// Single-GPU flag: gives hint to the client that they should use rotated tiling mode.
            uint32 hwRotationPortraitMode       :  1;
            /// Single-GPU flag: this primary surface supports non local heap.
            uint32 displaySupportsNonLocalHeap  :  1;
            /// Reserved for future use.
            uint32  reserved                    : 26;
        };
        uint32 u32All;  ///< Flags packed as 32-bit uint.
    } flags;            ///< get primary surface support info output flags.
};

/// Specifies different clock modes that the device can be set to.
enum class DeviceClockMode : uint32
{
    Default        = 0,   ///< Device clocks and other power settings are restored to default.
    Query          = 1,   ///< Queries the current device clock ratios. Leaves the clock mode of the device unchanged.
    Profiling      = 2,   ///< Scale down from peak ratio. Clocks are set to a constant amount which is
                          ///  known to be power and thermal sustainable. The engine/memory clock ratio
                          ///  will be kept the same as much as possible.
    MinimumMemory  = 3,   ///< Memory clock is set to the lowest available level. Engine clock is set to
                          ///  thermal and power sustainable level.
    MinimumEngine  = 4,   ///< Engine clock is set to the lowest available level. Memory clock is set to
                          ///  thermal and power sustainable level.
    Peak           = 5,   ///< Clocks set to maximum when possible. Fan set to maximum. Note: Under power
                          ///  and thermal constraints device will clock down.
    QueryProfiling = 6,   ///< Queries the profiling device clock ratios. Leaves the clock mode of the device unchanged.
    QueryPeak      = 7,   ///< Queries the peak device clock ratios. Leaves the clock mode of the device unchanged.
    Count
};

/// Specifies input argument to IDeive::SetClockMode. The caller can read the clock ratios the device is currently
/// running by querying using the mode DeviceClockMode::DeviceClockModeQuery.
struct SetClockModeOutput
{
    uint32 memoryClockFrequency; /// Current mem clock (absolute) value in Mhz
    uint32 engineClockFrequency; /// Current gpu core clock (absolute) value in Mhz
};

/// Specifies input argument to IDeive::SetClockMode. The caller must specify the mode in which to set the device.
struct SetClockModeInput
{
    DeviceClockMode clockMode; ///< Used to specify the clock mode for the device.
};

/// Specifies primary surface MGPU compositing mode.
enum MgpuMode : uint32
{
    MgpuModeOff  = 0,  ///< MGPU compositing mode off, the client does not do SW compositing at all, e.g. AFR disabled.
    MgpuModeSw   = 1,  ///< MGPU SW compositing mode, the client handle the SW compositing.
    MgpuModeDvo  = 2,  ///< MGPU DVO HW compositing mode
    MgpuModeXdma = 3,  ///< MGPU XDMA HW compositing mode
    MgpuModeCount
};

/// Specifies input arguments for IDevice::SetMgpuMode(). A client set a particular MGPU compositing mode and whether
/// frame pacing is enabled for a display.
struct SetMgpuModeInput
{
    uint32      vidPnSrcId;             ///< Video present source id.
    MgpuMode    mgpuMode;               ///< Primary surface MGPU compositing mode.
    bool        isFramePacingEnabled;   ///< True if frame pacing enabled. If so, the client creates a timer queue
                                        ///  to delay the present, and the delay value is calculated by KMD.
};

constexpr uint32 XdmaMaxDevices = 8;    ///< Maximum number of Devices for XDMA compositing.

/// Specifies XDMA cache buffer info for each gpu.
struct XdmaBufferInfo
{
    uint32 bufferSize;      ///< XDMA cache buffer size of each device
    uint32 startAlignment;  ///< XDMA cache buffer start alignment of each device
};

/// Specifies output arguments for IDevice::GetXdmaInfo(), returning the XDMA cache buffer information of each GPU for
/// a display.
struct GetXdmaInfoOutput
{
    XdmaBufferInfo  xdmaBufferInfo[XdmaMaxDevices]; ///< Output XDMA cache buffer info
};

/// Specifies flipping status flags on a specific VidPnSource. It's Windows specific.
union FlipStatusFlags
{
    struct
    {
        uint32 immediate : 1;  ///< Is immediate flip
        uint32 dwmFlip   : 1;  ///< Is DWM conducted flip
        uint32 iFlip     : 1;  ///< Is independent exclusive flip
        uint32 reserved  : 29; ///< Reserved for future use.
    };
    uint32 u32All;             ///< Flags packed as 32-bit uint.
};

/// Specifies the VSync mode of virtual display.
enum class VirtualDisplayVSyncMode : uint32
{
    Default   = 0,  ///< Using the default VSync mode based on refresh rate
    Immediate = 1,  ///< The presentation should be executed immediately without waiting for vsync to display
    HMD       = 2,  ///< Using HMD VSync, the HMD is specified by pPrivateScreen
    Count
};

/// The VirtualDisplayInfo is provided by application and KMD uses it to create a virtual display.
/// @see IDevice::CreateVirtualDisplay.
struct VirtualDisplayInfo
{
    uint32                  width;              ///< Horizontal dimension in pixels
    uint32                  height;             ///< Vertical dimension in pixels
    Rational                refreshRate;        ///< Refresh rate of virtual display
    VirtualDisplayVSyncMode vsyncMode;          ///< VSync mode
    uint32                  vsyncOffset;        ///< VSync front porch location in pixels or lines.
                                                ///  It's needed when VSyncMode is HMD
    Pal::IPrivateScreen*    pPrivateScreen;     ///< A pointer to IPrivateScreen.
                                                ///  It's needed when VSyncMode is HMD
};

/// Function pointer type definition for creating a buffer view SRD.
///
/// @see IDevice::CreateTypedBufferViewSrds()/CreateUntypedBufferViewSrds().
///
/// @param [in]  pDevice         Pointer to the device this function is called on.
/// @param [in]  count           Number of buffer view SRDs to create; size of the pBufferViewInfo array.
/// @param [in]  pBufferViewInfo Array of buffer view descriptions directing SRD construction.
/// @param [out] pOut            Client-provided space where opaque, hardware-specific SRD data is written.
///
/// @ingroup ResourceBinding
typedef void (PAL_STDCALL *CreateBufferViewSrdsFunc)(
    const IDevice*        pDevice,
    uint32                count,
    const BufferViewInfo* pBufferViewInfo,
    void*                 pOut);

/// Function pointer type definition for creating an image view SRD.
///
/// @see IDevice::CreateImageViewSrds().
///
/// @param [in]  pDevice      Pointer to the device this function is called on.
/// @param [in]  count        Number of buffer view SRDs to create; size of the pImageViewInfo array.
/// @param [in]  pImgViewInfo Array of image view descriptions directing SRD construction.
/// @param [out] pOut         Client-provided space where opaque, hardware-specific SRD data is written.
///
/// @ingroup ResourceBinding
typedef void (PAL_STDCALL *CreateImageViewSrdsFunc)(
    const IDevice*       pDevice,
    uint32               count,
    const ImageViewInfo* pImgViewInfo,
    void*                pOut);

/// Function pointer type definition for creating a fmask view SRD.
///
/// @see IDevice::CreateFmaskViewSrds().
///
/// @param [in]  pDevice        Pointer to the device this function is called on.
/// @param [in]  count          Number of fmask view SRDs to create; size of the pFmaskViewInfo array.
/// @param [in]  pFmaskViewInfo Array of fmask view descriptions directing SRD construction.
/// @param [out] pOut           Client-provided space where opaque, hardware-specific SRD data is written.
///
/// @ingroup ResourceBinding
typedef void (PAL_STDCALL *CreateFmaskViewSrdsFunc)(
    const IDevice*       pDevice,
    uint32               count,
    const FmaskViewInfo* pFmaskViewInfo,
    void*                pOut);

/// Function pointer type definition for creating a sampler SRD.
///
/// @see IDevice::CreateSamplerSrds().
///
/// @param [in]  pDevice      Pointer to the device this function is called on.
/// @param [in]  count        Number of sampler SRDs to create; size of the pSamplerInfo array.
/// @param [in]  pSamplerInfo Array of sampler descriptions directing SRD construction.
/// @param [out] pOut         Client-provided space where opaque, hardware-specific SRD data is written.
///
/// @ingroup ResourceBinding
typedef void (PAL_STDCALL *CreateSamplerSrdsFunc)(
    const IDevice*      pDevice,
    uint32              count,
    const SamplerInfo*  pSamplerInfo,
    void*               pOut);

/// Function pointer type definition for creating a ray tracing SRD.
///
/// @see IDevice::CreateBvhSrds().
///
/// @param [in]  pDevice   Pointer to the device this function is called on.
/// @param [in]  count     Number of BVH SRDs to create; size of the pBvhInfo array.
/// @param [in]  pBvhInfo  Array of BVH descriptions directing SRD construction.
/// @param [out] pOut      Client-provided space where opaque, hardware-specific SRD data is written.
///
/// @ingroup ResourceBinding
typedef void (PAL_STDCALL *CreateBvhSrdsFunc)(
    const IDevice*  pDevice,
    uint32          count,
    const BvhInfo*  pBvhInfo,
    void*           pOut);

/// Decode a buffer SRD back into most of the BufferViewInfo used to create it via Create{Typed,Untyped}BufferViewSrds.
///
/// @param [in]  pDevice
/// @param [in]  pBufferViewSrd Pointer to the SRD. Should not be in dedicated GPU memory.
/// @param [out] pViewInfo      Pointer to memory to be filled with decoded info.
typedef void (PAL_STDCALL *DecodeBufferViewSrdFunc)(
    const IDevice*  pDevice,
    const void*     pBufferViewSrd,
    BufferViewInfo* pViewInfo);

/// Structure filled in by DecodeImageViewSrd().
struct DecodedImageSrd
{
    SwizzledFormat swizzledFormat; ///< SRD's swizzled format.
    SubresRange    subresRange;    ///< SRD's subresource range; the array-range is always { 0, 1 } for 3D images.
    Range          zRange;         ///< z-range of the SRD's subresRange.startSubres.
};

/// Decode an image SRD back into the main parameters used to create it via CreateImageViewSrds().
///
/// @param [in]  pDevice
/// @param [in]  pImage          The same image the SRD was created on.
/// @param [in]  pImageViewSrd  Pointer to the SRD. Should not be in dedicated GPU memory.
/// @param [out] pDecodedInfo   Pointer to memory to be filled with decoded info.
typedef void (PAL_STDCALL *DecodeImageViewSrdFunc)(
    const IDevice*   pDevice,
    const IImage*    pImage,
    const void*      pImageViewSrd,
    DecodedImageSrd* pDecodedInfo);

/// Function pointer table for SRD methods.
struct DeviceInterfacePfnTable
{
    CreateBufferViewSrdsFunc pfnCreateTypedBufViewSrds;     ///< Typed Buffer view SRD creation function pointer.
    CreateBufferViewSrdsFunc pfnCreateUntypedBufViewSrds;   ///< Untyped Buffer view SRD creation function ptr.
    CreateImageViewSrdsFunc  pfnCreateImageViewSrds;        ///< Image view SRD creation function pointer.
    CreateFmaskViewSrdsFunc  pfnCreateFmaskViewSrds;        ///< Fmask View SRD creation function pointer.
    CreateSamplerSrdsFunc    pfnCreateSamplerSrds;          ///< Sampler SRD creation function pointer.
    CreateBvhSrdsFunc        pfnCreateBvhSrds;              ///< BVH SRD creation function pointer.
    DecodeBufferViewSrdFunc  pfnDecodeBufferViewSrd;        ///< Buffer SRD decode function pointer.
    DecodeImageViewSrdFunc   pfnDecodeImageViewSrd;         ///< Image SRD decode function pointer.
};

/// Specifies output arguments for IDevice::QueryWorkstationCaps(), returning worksation feature information
/// on this device workstation board.
union WorkStationCaps
{
    struct
    {
        uint32 workStationBoard             : 1;  ///< Running a workstation driver on a workstation board.
                                                  ///  On workstation boards that support CWG (Creator Who Game),
                                                  ///  the user can switch to a Gaming/consumer driver on the
                                                  ///  workstation board, and then this will be false.
        uint32 supportWorkstationAppPerfOpt : 1;  ///< Workstation boards have optimizations for kinds of workstation
                                                  ///  applications. These optimization is enabled if it is set.
        uint32 supportWorkstationEdgeFlag   : 1;  ///< Workstation boards have a DX9 feature that edge flag can be
                                                  ///  exported via point size output in VS. The feature is enabled
                                                  ///  if it is set.
        uint32 reserved                     : 29; ///< Reserved for future use.
    };
    uint32 u32All;         ///< Flags packed as 32-bit uint.
};

/// FrameLock/GenLock support state enum
enum class FlglSupport : uint32
{
    NotAvailable  = 0,   ///< FL/GL not supported by the GPU
    NotConnected  = 1,   ///< FL/GL support available in the GPU, but is not connected to a GLSync board
    Available     = 2,   ///< FL/GL support available and connected
    Count
};

/// Container structure for FrameLock/GenLock state.
struct FlglState
{
    union
    {
        struct
        {
            uint32 genLockEnabled    : 1;   ///< True if genlock is currently enabled. Genlock is a system-wide setting
                                            ///< in CCC. Genlock provides a singal source (which is used in framelock)
            uint32 frameLockEnabled  : 1;   ///< True if (KMD) framelock is currently enabled.
                                            ///< Framelock is the mechanism to sync all presents in multiple adapters.
            uint32 isTimingMaster    : 1;   ///< True if the display being driven by the current adapter is the timing
                                            ///< master in a genlock configuration
            uint32 reserved          : 29;  ///< Reserved for future use.
        };
        uint32 u32All;  ///< Packed 32-bit uint value.
    };
    FlglSupport support;         ///< The state of the FLGL support in current adapter
    uint32      firmwareVersion; ///< Firmware version number of the GLSync hardware (S400 board), if available
};

/// GlSync setting mask definition, used with GlSyncConfig
enum GlSyncConfigMask : uint32
{
    GlSyncConfigMaskSignalSource    = 0x00000001,
    GlSyncConfigMaskSyncField       = 0x00000002,
    GlSyncConfigMaskSampleRate      = 0x00000004,
    GlSyncConfigMaskSyncDelay       = 0x00000008,
    GlSyncConfigMaskTriggerEdge     = 0x00000010,
    GlSyncConfigMaskScanRateCoeff   = 0x00000020,
    GlSyncConfigMaskFrameLockCntl   = 0x00000040,
    GlSyncConfigMaskSigGenFrequency = 0x00000080
};

/// specify GLSYNC framelock control state
enum GlSyncFrameLockCtrl : uint32
{
    GlSyncFrameLockCntlNone                 = 0x00000000,
    GlSyncFrameLockCntlEnable               = 0x00000001,
    GlSyncFrameLockCntlDisable              = 0x00000002,
    GlSyncFrameLockCntlResetSwapCounter     = 0x00000004,
    GlSyncFrameLockCntlAckSwapCounter       = 0x00000008,
    GlSyncFrameLockCntlVersionKmd           = 0x00000010
};

/// Specifies GlSync Signal Source
enum GlSyncSignalSource : uint32
{
    GlSyncSignalSourceGpuMask   = 0x0FF,
    GlSyncSignalSourceUndefined = 0x100,
    GlSyncSignalSourceFreerun   = 0x101,
    GlSyncSignalSourceBncPort   = 0x102,
    GlSyncSignalSourceRj45Port1 = 0x103,
    GlSyncSignalSourceRj45Port2 = 0x104
};

/// Specifies GlSync Sync Field
enum GlSyncSyncField : uint8
{
    GlSyncSyncFieldUndefined    = 0,
    GlSyncSyncFieldBoth         = 1,
    GlSyncSyncField1            = 2
};

/// Specifies GlSync Sync Trigger Edge
enum GlSyncTriggerEdge : uint8
{
    GlSyncTriggerEdgeUndefined  = 0,
    GlSyncTriggerEdgeRising     = 1,
    GlSyncTriggerEdgeFalling    = 2,
    GlSyncTriggerEdgeBoth       = 3
};

/// Specifies GlSync scan rate coefficient/multiplier options
enum GlSyncScanRateCoeff : uint8
{
    GlSyncScanRateCoeffUndefined    = 0,
    GlSyncScanRateCoeffx5           = 1,
    GlSyncScanRateCoeffx4           = 2,
    GlSyncScanRateCoeffx3           = 3,
    GlSyncScanRateCoeffx5Div2       = 4,
    GlSyncScanRateCoeffx2           = 5,
    GlSyncScanRateCoeffx3Div2       = 6,
    GlSyncScanRateCoeffx5Div4       = 7
};

/// Container structure for FrameLock/GenLock config.
struct GlSyncConfig
{
    uint32 validMask;           ///< Mask that specifies which settings are actually referred in the structure.
                                ///  GlSyncConfigMask*
    uint32 syncDelay;           ///< Delay of sync signal in microseconds
    uint32 framelockCntlVector; ///< Vector of Framelock control bits. GlSyncFrameLockCntl*
    uint32 signalSource;        ///< Source of sync signal. Can be House Sync, RJ45 Port or GPUPort.
                                ///  GlSyncSignalSource* or GPUPort Index
    uint8  sampleRate;          ///< Number of VSyncs per sample. 0 - no sampling, syncronized by singal VSync.
    uint8  syncField;           ///< Sync to Field 1 or to both Fields when input signal is interlaced.
                                ///  GlSyncSyncField*
    uint8  triggerEdge;         ///< Which edge should be used as trigger. GlSyncTriggerEdge*
    uint8  scanRateCoeff;       ///< Scan Rate Multiplier applied to original sync signal. GlSyncScanRateCoeff*
    uint32 sigGenFrequency;     ///< Frequency in mHz of internal signal generator
};

#if PAL_KMT_BUILD
/// The type of NT object, it decides which KMT function should be called to retrieve the NT handle.
enum class ExternalObjectType : uint32
{
    Allocation = 0, ///< Specifies the NT object is an allocation, such as GPUMemory etc.
    SyncObject = 1, ///< Specifies the NT object is a sync object, such as semaphore and fence etc.
};

/// Specifies the NT handle attributes when opening from name.
struct ExternalHandleInfo
{
    ExternalObjectType          objectType;          ///< The type of NT object, it decides which KMT function should
                                                     ///  be called to retrieve the NT handle.
    const _SECURITY_ATTRIBUTES* pSecurityAttributes; ///< It specifies the security descriptor and the inheritable
                                                     ///   attribute.
    const wchar_t*              pNtObjectName;       ///< A name to NT handle, if the object is exported as a NT handle
                                                     ///  with a name, and then the handle can be acquired via this
                                                     ///  name.
    uint32                      accessFlags;         ///< Desried access rights of the object the handle refers to.
};
#endif

#if (PAL_KMT_BUILD || PAL_AMDGPU_BUILD)
/// Gpu heap group enumeration. One heap group contains several pal GpuHeap.
enum GpuHeapGroup : uint32
{
    GpuHeapGroupLocal     = 0x0,    /// Local heap group includes GpuHeapLocal and GpuHeapInvisible on Windows.
                                    /// But GpuHeapInvisible is not included on Linux.
    GpuHeapGroupNonLocal  = 0x1,    /// NonLocal heap group includes GpuHeapGartUswc and GpuHeapGartCacheable.
#if PAL_AMDGPU_BUILD
    GpuHeapGroupInvisible = 0x2,    /// This is used on Linux as GpuHeapLocal and GpuHeapInvisible are not combined.
#endif
    GpuHeapGroupCount,
};

/// Struct for querying current gpu memory usage info and budget info.
struct GpuMemoryBudgetInfo
{
#if PAL_AMDGPU_BUILD
    gpusize systemUsage[GpuHeapGroupCount];    /// Current total memory usage of specified heap group of whole system.
#else
    gpusize usage[GpuHeapGroupCount];          /// Current total memory usage of specified heap group of current process.
    gpusize budget[GpuHeapGroupCount];         /// Current total memory budget of specified heap group of the device which
                                               /// implies how much memory the device can allocate from that heap group
                                               /// before allocations may fail or cause performance degradation, including
                                               /// all allocated memory. Budget might be affected by OS status and other
                                               /// processes.
#endif
};
#endif

/// Reclaim allocation result enumeration.
enum class ReclaimResult : uint8
{
    Ok           = 0, ///< Reclaim result is OK.
    Discarded    = 1, ///< Reclaim result is discarded.
    NotCommitted = 2, ///< Reclaim result is not committed.
    Count
};

#if defined(_WIN32)
/// KMD has an interface in place to support UMD registration of Events to be signaled based on various
/// system-level conditions. Each enum here represents a certain condition which will cause KMD to signal
/// the event Clients previously registered through RegisterEvent.
enum class EventTrackingType : uint32
{
    ShaderInterrupt = 0,
    EarlyPresent    = 1,
    Count
};

/// Clients specify the Util::Event and EventTrackingType it wants to register with PAL in this struct.
/// Under the condition specified by the trackingType, pEvent will be signaled by KMD.
struct RegisterEventInfo
{
    Util::Event*      pEvent;
    EventTrackingType trackingType;
};

/// As an input of UnregisterEvent to specify the event and trackingType Clients want to unregister from KMD.
struct UnregisterEventInfo
{
    Util::Event*      pEvent;
    EventTrackingType trackingType;
};

/// Output parameter of RegisterEvent. There's one struct for each type of EventTrackingType.
union RegisterEventOutputInfo
{
    struct
    {
        uint32  eventId;
        gpusize eventMailboxGpuVa;
    } shaderInterrupt;

    struct
    {
        uint32  eventId;
    } earlyPresent;
};
#endif

/// Contains the page fault status of the GPU.
struct PageFaultStatus
{
    union
    {
        struct
        {
            uint32 pageFault :  1;  ///< Set if there was a GPU page fault.
            uint32 readFault :  1;  ///< Set if the page fault was during a read operation.
            uint32 reserved  :  30; ///< Reserved for future use.
        };
        uint32 u32All;
    } flags;

    gpusize faultAddress; ///< GPU virtual address where page fault occurred.  Ignored if @ref pageFault is not set.
};

/// Input to the RegisterRuntimeState call, which allows the HIP runtime to pass information to KMD which can be shared
/// with the HIP debugger.
struct HipRuntimeSetup
{
    const void* pRdebug;  ///< Address of the r_debug structure in the runtime
    uint32 runtimeState;  ///< Runtime-specific enum indicating runtime state
    uint32 ttmpSetupHint; ///< Hint indicating that ttmp values should be initialized
};

/**
 ***********************************************************************************************************************
 * @interface IDevice
 * @brief     Interface representing a client-configurable context for a particular GPU.
 *
 * This object becomes the root of all client/PAL interaction to get work done on that GPU.  The main functionality
 * provided by the device object:
 *
 * + Creation of all other PAL objects.
 * + GPU memory management.
 ***********************************************************************************************************************
 */
class IDevice
{
public:
    /// Get the maximum alignments for images created with a @ref ImageTiling::Linear tiling mode assuming the images'
    /// elements are no larger than pAlignments->maxElementSize.
    ///
    /// @param [out] pAlignments  Its maxElementSize will be used to fill the rest of its members with valid alignments.
    ///
    /// @returns Success if pAlignments was filled with data.  Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidPointer if pAlignments is null.
    ///          + ErrorInvalidValue if pAlignments->maxElementSize is zero.
    virtual Result GetLinearImageAlignments(
        LinearImageAlignments* pAlignments) const = 0;

    /// Fills out a structure with details on the properties of this device.  This includes capability flags,
    /// supported engines/queues, performance characteristics, etc.  This should only be called after a client has
    /// called @ref CommitSettingsAndInit().
    ///
    /// @see DeviceProperties
    ///
    /// @param [out] pInfo Properties structure to be filled out by PAL based on properties of this device.
    ///
    /// @returns Success if the device properties were successfully returned in pInfo.  Otherwise, one of the
    ///          following errors may be returned:
    ///          + ErrorInvalidPointer if pInfo is null.
    virtual Result GetProperties(
        DeviceProperties* pInfo) const = 0;

    /// Checks and returns execution state of the device. Currently unsupported for DX clients and
    /// will return Unavailable if called by those clients.
    ///
    /// @param [out] pPageFaultStatus  This struct is filled out when this function returns ErrorGpuPageFaultDetected.
    ///
    /// @returns Success if device is operational and running.  Otherwise, one of the following errors may be
    ///          + ErrorDeviceLost if device is lost, reset or not responding,
    ///          + ErrorInvalidValue if failed to get device reset state,
    ///          + ErrorOutOfGpuMemory if ran out of GPU memory,
    ///          + ErrorGpuPageFaultDetected if page fault was detected,
    ///          + ErrorUnknown if device is in unknown state.
    virtual Result CheckExecutionState(
        PageFaultStatus* pPageFaultStatus) = 0;

    /// Returns this devices client-visible settings structure initialized with appropriate defaults.  Clients can
    /// modify parameters in this structure as they wish in order to modify PAL's behavior for this device.  After
    /// modifying settings, the client must call CommitSettingsAndInit() before creating finalizing the device.
    ///
    /// @warning The returned value points to an internal PAL structure.  Modifying data using this pointer after
    ///          calling CommitSettingsAndInit() will result in undefined behavior.
    ///
    /// @returns Pointer to this devices public settings for examination and/or modification by the client.
    virtual PalPublicSettings* GetPublicSettings() = 0;

    /// Reads a specific setting from the operating system specific source (e.g. registry or config file).
    ///
    /// @param [in]  pSettingName Name of the setting. Must be null-terminated.
    /// @param [in]  settingScope The scope of settings accessible.
    /// @param [in]  valueType    The type of the setting to return (e.g. bool or int).
    /// @param [out] pValue       Buffer to write data that was read. Must be non-null.
    /// @param [out] bufferSz     Size of string buffer (pValue). Only necessary for ValueType::Str.
    /// @param [out] pInRegistry  If not null, this will be set to true if the setting was read from the registry.
    ///
    /// @returns True if the read of specified setting is successful. False indicates failure.
    virtual bool ReadSetting(
        const char*     pSettingName,
        SettingScope    settingScope,
        Util::ValueType valueType,
        void*           pValue,
        size_t          bufferSz = 0,
        bool*           pInRegistry = nullptr) const = 0;

#if defined(_WIN32)
    /// Clients call this function to register a Util::Event() object to be signaled based on specified conditions.
    /// For now, we will only add one condition: shader interrupt.
    ///
    /// @param [in] input               RegisterEventInfo which contains Util::Event client wants to be signaled and
    ///                                 EventTrackingType.
    /// @param [out] pOutput            Depending on the trackingType in input, the pOutput may refers different info
    ///                                 which clients may need. Refer to the definition of RegisterEventOutputInfo
    ///                                 for details.
    /// @returns Success if no errors occurred.
    virtual Result RegisterEvent(
        const RegisterEventInfo& input,
        RegisterEventOutputInfo* pOutput) = 0;

    /// Clients call this function to unregister an event previously registered with RegisterEvent().
    ///
    /// @param [in] input UnregisterEventInfo which contains Util::Event and tracing type the client
    ///                   wants to unregister.
    /// @returns Success if no errors occurred.
    virtual Result UnregisterEvent(
        const UnregisterEventInfo& input) = 0;

#endif

    /// Indicates that the client has finished overriding public settings so the settings struct can be finalized and
    /// any late-stage initialization can be done. This method must be called before @ref IDevice::Finalize() can be
    /// called.
    ///
    /// @note The only functions in IDevice that are able to be called before CommitSettingsAndInit():
    ///       + GetLinearImageAlignments()
    ///       + GetPublicSettings()
    ///       + ReadSetting()
    ///
    /// @note Finalizing the settings may override values set by the client.  This can occur if:
    ///       + Invalid settings, either because they are not supported by hardware or are somehow self-conflicting,
    ///         will be overridden.
    ///       + Settings specified in the private settings will override client-specified settings.
    ///
    /// @returns Success if settings have been committed successfully and any late-stage initialization is completed
    ///          successfully as well.
    virtual Result CommitSettingsAndInit() = 0;

    /// Returns the largest possible GPU memory alignment requirement for any IGpuMemoryBindable object created on this
    /// device.
    ///
    /// This is useful for clients that may want to allocate generic GPU memory rafts up front to support many objects
    /// without creating all of those objects to query their alignment requirements ahead of time.  In practice, most
    /// objects have a much smaller alignment requirement than the allocation granularity, but images may require more
    /// than the allocation granularity on some devices.
    ///
    /// @returns Largest possible GPU memory byte alignment for an IGpuMemoryBindable object on this device.
    virtual gpusize GetMaxGpuMemoryAlignment() const = 0;

    /// Indicates that the client is able to finalize the initialization of this device with the requisite information.
    /// This method must be called before any of the factory creation methods may be called.
    ///
    /// @note The only functions in IDevice that are able to be called before Finalize():
    ///         + The functions listed in IDevice::CommitSettingsAndInit().
    ///         + GetMaxGpuMemoryAlignment()
    ///         + GetProperties()
    ///
    /// @param [in] finalizeInfo Device finalization properties.
    ///
    /// @returns Success if final initialization is successful.
    virtual Result Finalize(
        const DeviceFinalizeInfo& finalizeInfo) = 0;

    /// Cleans up all internal state, undoing any work done by CommitSettingsAndInit() and Finalize(). Following a call
    /// to this function, the device will be in its initial state as if it was re-enumerated; the client may requery
    /// settings and build up the device for further use. If the client doesn't call this function, it will be called
    /// automatically when IPlatform::Destroy() is called or when devices are re-enumerated.
    ///
    /// This function provides clients with a way to return devices to a trival state, one in which they have no
    /// lingering OS or kernel driver dependencies. If a client pairs external state (e.g., an OS handle) with their
    /// devices they may be required to call this function when they destroy their API device objects.
    ///
    /// It is expected that all PAL objects created by the device have already been destroyed (e.g. GPU memory, queues),
    /// if not, the device may fall into an illegal state and the client will experience undefined behavior.
    ///
    /// @returns Success if no errors occurred.
    virtual Result Cleanup() = 0;

    /// Returns if dual-source blending can be enabled. It checks the ColorBlendStateCreateInfo for any src1 blending
    /// options. Then it checks if we are going to override those src1 options because the blend func is
    /// min or max.
    ///
    /// @param [in] createInfo The ColorBlendStateCreateInfo that is checked for conditions that call for dual-source
    ///                        blending.
    ///
    /// @returns true if the blend state calls for dual-source blending to be enabled.
    virtual bool CanEnableDualSourceBlend(
        const ColorBlendStateCreateInfo& createInfo) const = 0;

    /// Specifies how many frames can be placed in the presentation queue.  This limits how many frames the CPU can get
    /// in front of the device.
    ///
    /// @param [in] maxFrames Maximum number of frames that can be batched.  Specifying a value of 0 resets the limit to
    ///                       a default system value (3 frames on Windows).
    ///
    /// @returns Success if the limit was successfully adjusted.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorUnavailable if this function is not available on this OS.
    virtual Result SetMaxQueuedFrames(
        uint32 maxFrames) = 0;

    /// Compares this device against another device object to determine how compatible they are for multi-GPU
    /// operations.
    ///
    /// @param [in]  otherDevice Device to determine MGPU compatibility with.
    /// @param [out] pInfo       Result compatibility info.
    ///
    /// @returns Success if the compatibility info was successfully returned in pInfo.  Otherwise, one of the following
    ///          errors may be returned:
    ///          + ErrorInvalidPointer if pInfo is null.
    virtual Result GetMultiGpuCompatibility(
        const IDevice&        otherDevice,
        GpuCompatibilityInfo* pInfo) const = 0;

    /// Reports properties of all GPU memory heaps available to this device (e.g., size, whether it is CPU visible or
    /// not, performance characteristics, etc.).
    ///
    /// @param [out] info Properties of each GPU heap available to this device, indexed by the GPU ID defined in
    ///                   @ref GpuHeap.  If a particular heap is unavailable, its entry will report a size of 0.
    ///
    /// @returns Success if the heap properties were successfully queried and returned in info[].  Otherwise, one of the
    ///          following errors may be returned:
    ///          + ErrorUnknown if an unexpected internal error occured.
    virtual Result GetGpuMemoryHeapProperties(
        GpuMemoryHeapProperties info[GpuHeapCount]) const = 0;

    /// Reports all format and tiling mode related properties for this device.
    ///
    /// @param [out] pInfo  Output properties.
    ///
    /// @returns Success if the properties were successfully queried and returned in pProperties.  Otherwise, one of the
    ///          following errors may be returned:
    ///          + ErrorInvalidPointer if pInfo is null.
    virtual Result GetFormatProperties(
        MergedFormatPropertiesTable* pInfo) const = 0;

    /// Reports performance experiment related properties for this device.
    ///
    /// Enumerates the GPU family, blocks, capabilities, etc..
    ///
    /// @param [out] pProperties Output properties.
    ///
    /// @returns Success if the properties were successfully queried and returned in pProperties.  Otherwise, one of the
    ///          following errors may be returned:
    ///          + ErrorInvalidPointer if pProperties is null.
    virtual Result GetPerfExperimentProperties(
        PerfExperimentProperties* pProperties) const = 0;

    /// Fills out the default MSAA quad sample pattern for the given sample count.
    ///
    /// @param [in]  samples             The number of valid samples in the sample pattern. Must be a power of two.
    /// @param [out] pQuadSamplePattern  Fill this with the default pattern.
    ///
    /// @returns Success if @ref pQuadSamplePattern was filled with the default sample pattern.
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidPointer if @ref pQuadSamplePattern is null.
    ///          + ErrorInvalidValue if @ref samples is not a supported power of two.
    ///          + ErrorUnavailable if this device lacks GfxIp support.
    virtual Result GetDefaultSamplePattern(
        uint32                 samples,
        MsaaQuadSamplePattern* pQuadSamplePattern) const = 0;

    /// Adds a list of per-device memory object references that persist across command buffer submissions. It is the
    /// responsibility of the client to make sure that all required memory references have been added before submitting
    /// the command buffer that uses on them. References can be added at the device, queue or specified at submit time.
    /// gpuMemRefCount and ppGpuMemory cannot be 0/null. PAL will assert and crash if these values are invalid.   If
    /// multiple references are provided for the same memory, PAL will retain the safest set of GpuMemoryRef flags.
    ///
    /// see @ref IQueue::Submit()
    ///
    /// @param [in] gpuMemRefCount Number of memory references in the memory reference list, must be non-zero.
    /// @param [in] pGpuMemoryRefs Array of gpuMemRefCount GPU memory references.
    /// @param [in] pQueue         Optional IQueue that the memory references will be used on, used to optimize
    ///                            residency operations, can be null. Note, if a queue is specified here the same queue
    ///                            should be specified in RemoveGpuMemoryReferences.
    /// @param [in] flags          Flags from GpuMemoryRefFlags that will apply to all memory object references.
    ///
    /// @returns Success if the memory references were successfully added. Can also return NotReady if the client
    ///          passes in a valid pPagingFence pointer and the operation doesn't complete before the function returns.
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorTooManyMemoryReferences if gpuMemRefCount will cause the total reference count to exceed the
    ///            limit of this device.
    ///          + ErrorOutOfMemory if GPU memory objects will not fit in available GPU memory space (i.e. GPU Memory
    ///            is overcommitted).
    virtual Result AddGpuMemoryReferences(
        uint32              gpuMemRefCount,
        const GpuMemoryRef* pGpuMemoryRefs,
        IQueue*             pQueue,
        uint32              flags
        ) = 0;

    /// Removes a list of per-device memory object references that have previously been added via
    /// IDevice::AddGpuMemoryReferences(). PAL is responsible for ensuring that timestamps have been retired prior to
    /// actually performing any residency operations related to removal of a memory reference, so clients are free to
    /// call this function without regard for command buffer use. Memory references are reference counted, so an
    /// individual memory reference will only be removed when the total internal reference count reaches zero.
    /// gpuMemoryCount and ppGpuMemory cannot be 0/null, PAL will assert and crash if these values are invalid.
    ///
    /// @param [in] gpuMemoryCount Number of memory objects in the memory reference list (size of ppGpuMemory array).
    ///                            This count must be greater than zero.
    /// @param [in] ppGpuMemory    Array of GPU memory references.
    /// @param [in] pQueue         Optional IQueue that the memory references were used on, used to optimize residency
    ///                            operations, can be null. Note, if a queue was specified in AddGpuMemoryReferences,
    ///                            pQueue must match.
    ///
    /// @returns Success if the memory references were successfully updated.
    virtual Result RemoveGpuMemoryReferences(
        uint32            gpuMemoryCount,
        IGpuMemory*const* ppGpuMemory,
        IQueue*           pQueue
        ) = 0;

    /// Queries the Device for the total amount of referenced GPU memory for each heap type.  These totals include all
    /// memory added to the Device or any Queue using @ref AddGpuMemoryReferences and not yet removed using @ref
    /// RemoveGpuMemoryReferences.  Internal PAL allocations are included in these totals, but memory referenced using
    /// the per-submit list in @ref IQueue::Submit is not included in these amounts.
    ///
    /// The intended use for this interface is for clients to be able to manage budgeting of resident GPU memory.
    ///
    /// @param [out] referencedGpuMemTotal Array containing the total amount of referenced GPU memory for each GPU
    ///              memory heap.
    virtual void GetReferencedMemoryTotals(
        gpusize  referencedGpuMemTotal[GpuHeapCount]) const = 0;

    /// Get primary surface MGPU support information based upon primary surface create info and input flags provided
    /// by client.
    ///
    /// This function should not be called by clients that rely on PAL for compositor management.  Basically, if your
    /// client uses the IScreen's interface to take full screen exclusive mode, then don't call this.
    ///
    /// @param [in]     primaryInfoInput        Primary surface info input arguments.
    /// @param [in,out] pPrimaryInfoOutput      Primary surface info output arguments.
    ///
    /// @returns Success if the primary surface MGPU support information were successfully queried.
    virtual Result GetPrimaryInfo(
        const GetPrimaryInfoInput&  primaryInfoInput,
        GetPrimaryInfoOutput*       pPrimaryInfoOutput) const = 0;

    /// Returns the supported stereo modes list.
    ///
    /// @param [in,out] pStereoModeCount Input value specifies the maximum number of stereo modes to enumerate, and the
    ///                                  output value specifies the total number of stereo modes that were enumerated
    ///                                  in pStereoModeList.  The input value is ignored if pStereoModeList is null.
    ///                                  This pointer must not be null.
    /// @param [out]    pStereoModeList  Output list of stereo modes.  Can be null, in which case the total number of
    ///                                  available modes will be written to pStereoModeCount.
    ///
    /// @returns Success if the display modes were successfully queried and the results were reported in
    ///          pStereoModeCount/pStereoModeList.  Otherwise, one of the following errors may be returned:
    ///          + Unsupported if stereo mode is not supported, or the stereo modes can't be queried.
    ///          + ErrorOutOfMemory if temp memeory allocation failed.
    virtual Result GetStereoDisplayModes(
        uint32*                   pStereoModeCount,
        StereoDisplayModeOutput*  pStereoModeList) const = 0;

    /// Returns the currently selected Workstation stereo mode on Windows OS.
    ///
    /// @param [out]    pWsStereoMode    Output currently selected Workstation Stereo mode.
    ///
    /// @returns Success if the currently selected Workstation stereo mode  were successfully queried
    ///          and the results were reported in pWsStereoMode.
    virtual Result GetWsStereoMode(WorkstationStereoMode* pWsStereoMode) const = 0;

    /// Return information about active workstation support for 10-bit (potentially packed pixel) displays.
    ///
    /// @param [out]    pMode  Output reports if the workstation 10-bit display feature is enabled, and if so,
    ///                        details on any required pixel packing.
    ///
    /// @returns Success if the 10-bits and packed-pixel format were successfully queried and the result were
    ///          reported in pMode.
    virtual Result GetActive10BitPackedPixelMode(
        Active10BitPackedPixelModeOutput*  pMode) const = 0;

    /// Inform the KMD that this allocation must be reinterpreted as 10-bits per channel for the all
    /// KMD-initiated BLTs.
    ///
    /// When in 10-bit mode and at present time, if the dx9p driver sees a 10-bit to 8-bit surface blt,
    /// it will use this interface to inform the KMD that the blt dst surface must be reinterpreted as
    /// 10-bits per channel for the all KMD-initiated BLTs.
    ///
    /// @param [in] pGpuMemory    The dst GPU memory reference which will be marked as 10 bits format.
    ///
    /// @returns Success if the KMD has been sucessfully notified.
    virtual Result RequestKmdReinterpretAs10Bit(
        const IGpuMemory* pGpuMemory) const = 0;

    /// Set or query device clock mode.
    ///
    /// This function can be called by clients to set the device engine and memory clocks to certain pre-defined ratios.
    /// If a call to restore the device clocks to default does not occur, the device stays in the previously set mode.
    ///
    /// @param [in]  setClockModeInput        Specify the clock mode to set the device to.
    /// @param [out] pSetClockModeOutput      @b Optional - Output device clock mode. If not nullptr, it is used
    ///                                       to query the current clock mode the device is running in.
    ///
    /// @returns Success if the device clock mode query/set request was successful.
    virtual Result SetClockMode(
        const SetClockModeInput& setClockModeInput,
        SetClockModeOutput*      pSetClockModeOutput) = 0;

    /// Request to enable/disable static VMID for the device.
    ///
    /// The function must be called with enable = true before a profiling session starts and enable = false after a
    /// profiling session ends. It may be called any time, though it is illegal to disable without a prior corresponding
    //  enable/acquire. Only after this returns success (when enabling) can the driver make submissions targeting the
    /// static VMID.
    ///
    /// @param [in] enable Specifies whether acquiring or releasing the static VMID
    ///
    /// @returns Success if the static VMID acquire/release request was successful.
    virtual Result SetStaticVmidMode(
        bool enable) = 0;

    /// Set up MGPU compositing mode of a display provided by client.
    ///
    /// This function should not be called by clients that rely on PAL for compositor management.  Basically, if your
    /// client uses the IScreen's interface to take full screen exclusive mode, then don't call this.
    ///
    /// @param [in] setMgpuModeInput        Set MGPU compositing mode input arguments.
    ///
    /// @returns Success if the MGPU compositing mode were successfully set.
    virtual Result SetMgpuMode(
        const SetMgpuModeInput& setMgpuModeInput) const = 0;

    /// Get XDMA cache buffer information of each GPU based upon video present source ID provided by client.
    ///
    /// This function should not be called by clients that rely on PAL for compositor management.  Basically, if your
    /// client uses the IScreen's interface to take full screen exclusive mode, then don't call this.
    ///
    /// @param [in]     vidPnSrcId              Video present source id.
    /// @param [in]     gpuMemory               Primary surface GPU memory.
    /// @param [in,out] pGetXdmaInfoOutput      Set XDMA cache buffer info output arguments.
    ///
    /// @returns Success if the XDMA cache buffer information were successfully queried.
    virtual Result GetXdmaInfo(
        uint32              vidPnSrcId,
        const IGpuMemory&   gpuMemory,
        GetXdmaInfoOutput*  pGetXdmaInfoOutput) const = 0;

    /// Polls current fullscreen frame metadata controls on given vidPnSourceId, including extended data.
    ///
    /// The function is used by clients that support frame metadata through KMD-UMD shared memory.
    /// It polls the frame metadata shared memory for the given VidPnSource. Indicating which types of metadata
    /// the UMD should send to KMD.
    /// Clients should only call this function on the master device in an LDA chain.
    ///
    /// @param [in]  vidPnSrcId             Video present source id
    /// @param [out] pFrameMetadataControl  @b Optional - Output frame metadata controls. Clients can pass null to
    ///                                     check if the buffer is initialized successfully and if the
    ///                                     vidPnSrcId is valid.
    ///
    /// @returns Success if the metadata controls on the given vidPnSrcId was successfully polled.
    ///          Otherwise, one of the following erros may be returned:
    ///          + ErrorInvalidValue if vidPnSrcId is invalid (out of range)
    ///          + ErrorUnavailable if no implementation on current platform or if metadata shared buffer is null.
    virtual Result PollFullScreenFrameMetadataControl(
        uint32                         vidPnSrcId,
        PerSourceFrameMetadataControl* pFrameMetadataControl) const = 0;

    /// Get flip status flags and a flag indicating if current device owns the flags. (DX only)
    ///
    /// The function is used by clients that need flip status polling through KMD-UMD shared memory.
    /// It provides caller the FlipStatusFlags on the given VidPnSource. Note that the flag returned is only a hint,
    /// and can have a one frame delay during flip status transition.
    ///
    /// @param [in]  vidPnSrcId    Video present source id.
    /// @param [out] pFlipFlags    Output flip flags on given vidPnSrcId. Must not be null.
    /// @param [out] pIsFlipOwner  Output indicating if the current device owns the flip flags.
    ///
    /// @returns Success if flipping flags on given vidPnSrcId was successfully polled.
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidValue if vidPnSrcId is invalid.
    ///          + ErrorUnavailable if no implementation on current platform.
    ///          + ErrorInitializationFailed if flip status shared buffer was failed to initialize.
    virtual Result GetFlipStatus(
        uint32           vidPnSrcId,
        FlipStatusFlags* pFlipFlags,
        bool*            pIsFlipOwner) const = 0;

    /// Resets the specified set of fences.
    ///
    /// All fences must be reset before passing them to a submission command.
    ///
    /// @param [in] fenceCount Number of fences to reset.
    /// @param [in] ppFences   Array of fences to reset.
    ///
    /// @returns Success if the specified fences have been successfully reset.
    ///
    /// @note The function assumes that neither ppFences is null nor that any of the elements of the array pointed by
    /// ppFences are null.
    virtual Result ResetFences(
        uint32              fenceCount,
        IFence*const*       ppFences) const = 0;

    /// Stalls the current thread until one or all of the specified fences have been reached by the device.
    ///
    /// If waitAll is true all fences must have been submitted at least once before this is called;
    /// otherwise at least one fence must have been submitted.  Using a zero timeout value returns
    /// immediately and can be used to determine the status of a set of fences without stalling.
    ///
    /// @param [in] fenceCount Number of fences to wait for (i.e., size of the ppFences array).
    /// @param [in] ppFences   Array of fences to be waited on.
    /// @param [in] waitAll    If true, wait for completion of all fences in the array before returning; if false,
    ///                        return after any single fence in the array has completed.
    /// @param [in] timeout    This method will return after this many nanoseconds even if the fences do not complete.
    ///
    /// @returns Success if the specified fences have been reached, or Timeout if the fences have not been reached but
    ///          the specified timeout time has elapsed.  Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidPointer if:
    ///              - ppFences is null.
    ///              - Any member of the ppFences array is null.
    ///          + ErrorInvalidValue if:
    ///              - fenceCount is zero.
    ///          + ErrorFenceNeverSubmitted if:
    ///              - Any of the specified fences haven't been submitted.
    virtual Result WaitForFences(
        uint32                   fenceCount,
        const IFence*const*      ppFences,
        bool                     waitAll,
        std::chrono::nanoseconds timeout) const = 0;

    /// Stalls the current thread until one or all of the specified Semaphores have been reached by the device.
    ///
    /// Using a zero timeout value returns immediately and can be used to determine the status of a set of semaphores
    /// without stalling.
    ///
    /// @param [in] semaphoreCount Number of semaphores to wait for (i.e., size of the ppFences array).
    /// @param [in] ppSemaphores   Array of semaphores to be waited on.
    /// @param [in] pValues        Array of semaphores's value to be waited on.
    /// @param [in] flags          Combination of zero or more @ref HostWaitFlags values describing the behavior of this
    ///                            wait operation. See @ref HostWaitFlags for more details.
    /// @param [in] timeout        This method will return after this many nanoseconds even if the semaphores do not
    ///                            complete.
    ///
    /// @returns Success if the specified semaphores have been reached, or Timeout if the semaphores have not been
    ///          reached but the specified timeout time has elapsed.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorInvalidPointer if:
    ///              - ppSemaphores is null.
    ///              - Any member of the ppSemaphores array is null.
    ///          + ErrorInvalidValue if:
    ///              - semaphoreCount is zero.
    virtual Result WaitForSemaphores(
        uint32                       semaphoreCount,
        const IQueueSemaphore*const* ppSemaphores,
        const uint64*                pValues,
        uint32                       flags,
        std::chrono::nanoseconds     timeout) const = 0;

    /// Correlates a GPU timestamp with the corresponding CPU timestamps, for tighter CPU/GPU timeline synchronization
    ///
    /// @param [out] pCalibratedTimestamps  Reports a current GPU timestamp along with the CPU timestamps at the time
    ///                                     that GPU timestamp was written.  The CPU timestamps are OS-specific.  Also
    ///                                     reports a maximum deviation between the captured timestamps in nanoseconds.
    ///
    /// @returns Success if the request was successful.  Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidPointer if:
    ///              - pCalibratedTimestamps is null.
    ///          + ErrorUnavailable if:
    ///              - unable to capture timestamps for all requested time domains.
    virtual Result GetCalibratedTimestamps(
        CalibratedTimestamps* pCalibratedTimestamps) const = 0;

    /// Binds the specified GPU memory as a trap handler for the specified pipeline type.  This GPU memory must hold
    /// shader machine code (i.e., the client must generate HW-specific shader binaries through some external means,
    /// probably the SP3 assembler).
    ///
    /// The same trap handler will be installed for all shader stages that are part of the pipeline.  A trap handler
    /// will only ever be executed for shaders that set the trapPresent bit in @ref PipelineShaderInfo.
    ///
    /// @param [in] pipelineType Select compute or graphics pipeline.  If graphics, this trap handler will be installed
    ///                          for _all_ hardware shader stages.
    /// @param [in] pGpuMemory   GPU memory allocation holding the trap handler.
    /// @param [in] offset       Offset in bytes into pGpuMemory where the trap handler shader code begins.  Must be
    ///                          256 byte aligned.
    virtual void BindTrapHandler(
        PipelineBindPoint pipelineType,
        IGpuMemory*       pGpuMemory,
        gpusize           offset) = 0;

    /// Binds the specified GPU memory location as a trap buffer for the specified pipeline type.  This GPU memory will
    /// be available to the trap handler as scratch memory to use as it chooses.  The same trap buffer will be installed
    /// for all shader stages that are part of the pipeline.
    ///
    /// There is no size parameter for the trap buffer.  The client is responsible for ensuring that the trap handler
    /// only reads/writes data within the bounds designated for trap buffer usage.
    ///
    /// @param [in] pipelineType Select compute or graphics pipeline.  If graphics, this trap buffer will be installed
    ///                          for _all_ hardware shader stages.
    /// @param [in] pGpuMemory   GPU memory allocation holding the trap buffer range.
    /// @param [in] offset       Offset in bytes into pGpuMemory where the trap buffer range starts.  Must be 256 byte
    ///                          aligned.
    virtual void BindTrapBuffer(
        PipelineBindPoint pipelineType,
        IGpuMemory*       pGpuMemory,
        gpusize           offset) = 0;

    /// Get the swap chain information for creating a swap chain and presenting an image.
    ///
    /// @param [in]  hDisplay                Display handle of the local window system.
    /// @param [in]  hWindow                 Window handle of the local window system.
    /// @param [in]  wsiPlatform             WSI Platform the swapchain supposed to work on
    /// @param [in,out] pSwapChainProperties Contains swap chain information.
    ///
    /// @returns Success if get swap chain information successfully.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorUnknown if an unexpected internal error occurs.
    virtual Result GetSwapChainInfo(
        OsDisplayHandle      hDisplay,
        OsWindowHandle       hWindow,
        WsiPlatform          wsiPlatform,
        SwapChainProperties* pSwapChainProperties) = 0;

    /// Determines if the given window system requirement is supported by the underlying wsiPlatform.
    ///
    /// @param [in] hDisplay                  Display handle of the local window system.
    /// @param [in] wsiPlatform               WSI Platform the request supposed to send to
    /// @param [in] visualId                  Requested visual information which may not needed for some wsiPlatforms
    ///
    /// @returns Success if the request is supported. Otherwise, one of the following erros may be returned:
    ///         + Unsupported
    virtual Result DeterminePresentationSupported(
        OsDisplayHandle      hDisplay,
        WsiPlatform          wsiPlatform,
        int64                visualId) = 0;

    /// Returns a mask of SwapChainModeSupport flags for each present mode. The swapchain modes are different for each
    /// WsiPlatform.
    ///
    /// @param [in]  wsiPlatform             WSI Platform the swapchain is supposed to work on.
    /// @param [in]  mode                    The swap chain will use this present mode.
    ///
    /// @returns Returns a mask of SwapChainModeSupport.
    virtual uint32 GetSupportedSwapChainModes(
        WsiPlatform wsiPlatform,
        PresentMode mode) const = 0;

    /// Returns a mask of SwapChainModeSupport flags for each present mode. The swapchain modes are different for each
    /// WsiPlatform.
    ///
    /// @param [in]      wsiPlatform             WSI Platform the swapchain is supposed to work on.
    /// @param [in]      mode                    The swap chain will use this present mode.
    ///
    /// @returns Returns a mask of SwapChainModeSupport.
    virtual uint32 GetCompatiblePresentModes(
        WsiPlatform          wsiPlatform,
        SwapChainMode        mode) const = 0;

    /// Determines if the given information corresponds to an external shared image.
    ///
    /// Some clients may not know if a given external shared resource is a simple GPU memory allocation or an image; it
    /// is expected they will call this function to determine which set of open functions they must call.
    ///
    /// @param [in]  openInfo  The open info describing the external shared resource.
    /// @param [out] pIsImage  Its contents will be set to true if the external shared resource is an image.
    ///
    /// @returns Success if PAL was able to determine whether or not the resource is an image.  Otherwise, one of the
    ///          following errors may be returned:
    ///          + ErrorInvalidPointer if pIsImage is null.
    ///          + ErrorUnknown if an unexpected internal error occurs.
    virtual Result DetermineExternalSharedResourceType(
        const ExternalResourceOpenInfo& openInfo,
        bool*                           pIsImage) const = 0;

    /// @name FactoryMethods Device Factory Methods
    ///
    /// The following set of IDevice methods is the interface through which almost all PAL objects are created.
    ///
    /// PAL does not allocate its own system memory for these objects.  Instead, the client must query the amount of
    /// system memory required for the object then provide a pointer where PAL will construct the object.
    ///
    /// This approach allows the client to roll the PAL object into its own allocations without unnecessary heap
    /// allocations and cache misses.  It can also allocate many objects in a single memory space without PAL
    /// involvement.
    ///
    /// @{

    /// Determines the amount of system memory required for a queue object.  An allocation of this amount of memory
    /// must be provided in the pPlacementAddr parameter of CreateQueue().
    ///
    /// @param [in]  createInfo  Properties of the new queue such as engine type and engine index.
    /// @param [out] pResult     The validation result if pResult is non-null. This argument can be null to avoid the
    ///                          additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an IQueue object with the specified properties.
    ///          A return value of 0 indicates the createInfo was invalid.
    virtual size_t GetQueueSize(
        const QueueCreateInfo& createInfo,
        Result*                pResult) const = 0;

    /// Creates a queue object.
    ///
    /// @param [in]  createInfo     Properties of the new queue such as engine type and engine index.
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetQueueSize() with the same
    ///                             create info.
    /// @param [out] ppQueue        Constructed queue object.  When successful, the returned address will be the same
    ///                             as specified in pPlacementAddr.
    ///
    /// @returns Success if the queue was successfully created.  Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppQueue is null.
    ///          + ErrorInvalidValue if the create info's engineType is invalid or if the engineIndex is invalid.
    virtual Result CreateQueue(
        const QueueCreateInfo& createInfo,
        void*                  pPlacementAddr,
        IQueue**               ppQueue) = 0;

    /// Determines the amount of system memory required for a multi-queue object.  An allocation of this amount of
    /// memory must be provided in the pPlacementAddr parameter of CreateMultiQueue().
    ///
    /// @param [in]  queueCount  Number of queues in the gang; matches number of entries in pCreateInfo.
    /// @param [in]  pCreateInfo Properties of each queue to create for this gang (engine type, etc.).  The first
    ///                          entry in this array describes the master queue which will be used to execute all
    ///                          IQueue interfaces except for MultiSubmit().
    /// @param [out] pResult     The validation result if pResult is non-null. This argument can be null to avoid the
    ///                          additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an multi-queue IQueue object with the specified
    ///          properties.  A return value of 0 indicates the createInfo was invalid.
    virtual size_t GetMultiQueueSize(
        uint32                 queueCount,
        const QueueCreateInfo* pCreateInfo,
        Result*                pResult) const = 0;

    /// Creates a multi-queue (i.e., gang submission queue) object.  The resulting version of the IQueue interface
    /// is composed of multiple hardware queues which can be atomically submitted to as a group.  When this is done,
    /// it is safe to use IGpuEvent objects to tightly synchronize work done across queues in a single call to Submit().
    /// This can allow the client to tightly schedule asynchronous workloads for maximum efficiency that isn't possible
    /// across queues using IQueueSemaphore objects.
    ///
    /// @param [in]  queueCount     Number of queues in the gang; matches number of entries in the pCreateInfo array.
    /// @param [in]  pCreateInfo    Properties of each queue to create for this gang (engine type, etc.).  The first
    ///                             entry in this array describes the master queue which will be used to execute all
    ///                             IQueue interfaces except for the ganged-portion of a Submit() (e.g., Present()).
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetMultiQueueSize() with the same
    ///                             arguments.
    /// @param [out] ppQueue        Constructed multi queue object.
    ///
    /// @returns Success if the multi queue was successfully created.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorInvalidValue if queueCount is less than 2.
    ///          + ErrorInvalidQueueType if any of the created sub-queues are not multi-queue compatible.  This is
    ///            indicated by the supportsMultiQueue engineProperties flag in @ref DeviceProperties.
    ///          + ErrorInvalidPointer if pCreateInfo, pPlacementAddr or ppQueue is null.
    ///          + ErrorInvalidValue if any create info's configuration is invalid.
    virtual Result CreateMultiQueue(
        uint32                 queueCount,
        const QueueCreateInfo* pCreateInfo,
        void*                  pPlacementAddr,
        IQueue**               ppQueue) = 0;

    /// Determines the amount of system memory required for a GPU memory object.
    ///
    /// An allocation of this amount of memory must be provided in the pPlacementAddr parameter of CreateGpuMemory().
    ///
    /// @param [in]  createInfo Data controlling the GPU memory properties, such as size, alignment, and allowed heaps.
    /// @param [out] pResult    The validation result if pResult is non-null. This argument can be null to avoid the
    ///                         additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an IGpuMemory object with the specified properties.  A
    ///          return value of 0 indicates the createInfo was invalid.
    virtual size_t GetGpuMemorySize(
        const GpuMemoryCreateInfo& createInfo,
        Result*                    pResult) const = 0;

    /// Creates an @ref IGpuMemory object with the requested properties.
    ///
    /// This method can create either _real_ or _virtual_ GPU memory allocations.
    ///
    /// @param [in]  createInfo     Data controlling the GPU memory properties, such as size, alignment, and allowed
    ///                             heaps.
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetGpuMemorySize() with the same
    ///                             createInfo param.
    /// @param [out] ppGpuMemory    Constructed GPU memory object.  When successful, the returned address will be the
    ///                             same as specified in pPlacementAddr.
    ///
    /// @returns Success if the GPU memory was successfully created.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + TooManyFlippableAllocations if the GPU memory was successfully created, but the client has reached
    ///            the limit of flippable allocations for this Device.  This is a warning that future flippable GPU
    ///            memory may fail to be created due to internal OS limitations.
    ///          + ErrorInvalidPointer if pPlacementAddr or ppGpuMemory is null.
    ///          + ErrorInvalidMemorySize if createInfo.size is invalid.
    ///          + ErrorInvalidAlignment if createInfo.alignment is invalid.
    ///          + ErrorInvalidValue if createInfo.heapCount is 0 for real allocations or non-0 for virtual allocations.
    ///          + ErrorOutOfGpuMemory if the allocation failed due to a lack of GPU memory.
    ///          + ErrorUnavailable if the Reserve Gpu Virtual Address failed.
    virtual Result CreateGpuMemory(
        const GpuMemoryCreateInfo& createInfo,
        void*                      pPlacementAddr,
        IGpuMemory**               ppGpuMemory) = 0;

    /// Determines the amount of system memory required for a pinned GPU memory object.
    ///
    /// An allocation of this amount of memory must be provided in the pPlacementAddr parameter of
    /// CreatePinnedGpuMemory().
    ///
    /// @param [in]  createInfo Data controlling the GPU memory properties, such as size and the allocation to pin.
    /// @param [out] pResult    The validation result if pResult is non-null. This argument can be null to avoid the
    ///                         additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an IGpuMemory object pinned with the specified
    ///          properties.
    virtual size_t GetPinnedGpuMemorySize(
        const PinnedGpuMemoryCreateInfo& createInfo,
        Result*                          pResult) const = 0;

    /// Pins a segment of system memory in place and create an @ref IGpuMemory object allowing access by the GPU.
    ///
    /// @param [in]  createInfo     Data controlling the GPU memory properties, such as size and the allocation to pin.
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetPinnedGpuMemorySize() with the
    ///                             same params.
    /// @param [out] ppGpuMemory    Constructed GPU memory object.  When successful, the returned address will be the
    ///                             same as specified in pPlacementAddr.
    ///
    /// @returns Success if the system memory was successfully pinned and a corresponding GPU memory object was created.
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidPointer if createInfo.pSysMem, pPlacementAddr, or ppGpuMemory is null, or if
    ///            createInfo.pSysMem is not allocation granularity aligned.
    ///          + ErrorInvalidMemorySize if createInfo.memSize is not allocation granularity aligned.
    ///          + ErrorOutOfMemory if the creation failed because the system memory could not be pinned.
    virtual Result CreatePinnedGpuMemory(
        const PinnedGpuMemoryCreateInfo& createInfo,
        void*                            pPlacementAddr,
        IGpuMemory**                     ppGpuMemory) = 0;

    /// Determines the amount of system memory required for a SVM memory object
    ///
    /// An allocation of this amount of memory must be provided in the pPlacementAddr parameter of
    /// CreateSvmGpuMemory().
    ///
    /// @param [in]  createInfo Data controlling the SVM memory properties, such as size and type of SVM buffer.
    /// @param [out] pResult    The validation result if pResult is non-null. This argument can be null to avoid the
    ///                         additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an IGpuMemory object with the specified properties.
    virtual size_t GetSvmGpuMemorySize(
        const SvmGpuMemoryCreateInfo& createInfo,
        Result*                       pResult) const = 0;

    /// Creates an SVM (Shared Virtual Memory) IGpuMemory object.
    /// The basic idea of SVM is to create system memory that has the same CPU and GPU virtual address
    /// (i.e., "pointer is a pointer").  This can work in two modes: fine-grain, or coarse-grain.
    ///
    /// Fine-grain (Single-GPU): The client should just call this function with pReservedGpuVaOwner set to null.
    /// PAL will allocate GPU-accessible system memory that will have the same CPU virtual address
    /// (as returned by IGpuMemory::Map()) as GPU virtual address
    /// (as returned in the gpuVirtAddr value returned by IGpuMemory::Desc()).
    ///
    /// Fine-grain (MGPU): The client can call this function with pReservedGpuVaOwner set to IGpuMemory object
    /// allocated on the first device and receive mapping to the same GPU VA location on another device.
    ///
    /// Coarse-grain: In this mode, there are actually two separate IGpuMemory objects.
    /// The client should first create the "local" GPU memory object by calling IDevice::CreateGpuMemory() to create
    /// a standard GPU memory object in the VaRange::Svm VA space.  Next, the client should create the "staging" GPU
    /// memory by calling this function (CreateSvmGpuMemory) with pReserveGpuVaOwner pointing to the "local" GPU
    /// memory object.  PAL will create system memory for the "staging" GPU memory with a CPU virtual address matching
    /// the "local" GPU memory's GPU virtual address.  The GPU virtual address of the "staging" GPU memory is
    /// arbitrarily assigned.  The client is responsible for managing the contents of the two related allocations
    /// per their API rules.
    ///
    /// @param [in]  createInfo     Data controlling the SVM memory properties, such as size and location of SVM buffer.
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetSvmGpuMemorySize() with the
    ///                             same params.
    /// @param [out] ppGpuMemory    Constructed GPU memory object.  When successful, the returned address will be the
    ///                             same as specified in pPlacementAddr.
    ///
    /// @returns Success if the SVM buffer was successfully created and a corresponding GPU memory object was created.
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr, or ppGpuMemory is null.
    ///          + ErrorInvalidMemorySize if createInfo.memSize is not allocation granularity aligned.
    ///          + ErrorOutOfMemory if the creation failed because there is not enough GPU memory
    ///            or the system memory could not be pinned.
    virtual Result CreateSvmGpuMemory(
        const SvmGpuMemoryCreateInfo& createInfo,
        void*                         pPlacementAddr,
        IGpuMemory**                  ppGpuMemory) = 0;

    /// Determines the amount of system memory required for a GPU memory object created by opening an allocation from a
    /// different GPU.
    ///
    /// An allocation of this amount of memory must be provided in the pPlacementAddr parameter of
    /// OpenSharedGpuMemory().
    ///
    /// @param [in]  openInfo Specifies a handle to a shared GPU memory object to open.
    /// @param [out] pResult  The validation result if pResult is non-null. This argument can be null to avoid the
    ///                       additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for opening a shared IGpuMemory object with the specified
    ///          properties.  A return value of 0 indicates the openInfo was invalid.
    virtual size_t GetSharedGpuMemorySize(
        const GpuMemoryOpenInfo& openInfo,
        Result*                  pResult) const = 0;

    /// Opens a shareable GPU memory object created on another device for use on this device.
    ///
    /// @param [in]  openInfo       Specifies a handle to a shared GPU memory object to open.
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetSharedGpuMemorySize() with the
    ///                             same params.
    /// @param [out] ppGpuMemory    Constructed GPU memory object.  When successful, the returned address will be the
    ///                             same as specified in pPlacementAddr.
    ///
    /// @returns Success if the shared memory was successfully opened for access on this device.  Otherwise, one of the
    ///          following errors may be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppGpuMemory is null.
    ///          + ErrorNotShareable if the specified memory object was not marked as shareable on creation.
    virtual Result OpenSharedGpuMemory(
        const GpuMemoryOpenInfo& openInfo,
        void*                    pPlacementAddr,
        IGpuMemory**             ppGpuMemory) = 0;

    /// Determines the amount of system memory required for a external GPU memory object created by opening
    ///  an allocation from a compatible device, such as D3D device.
    ///
    /// @param [out] pResult  The validation result if pResult is non-null. This argument can be null to avoid the
    ///                       additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for opening a shared IGpuMemory object with the specified
    ///          properties.  A return value of 0 indicates the openInfo was invalid.
    virtual size_t GetExternalSharedGpuMemorySize(
        Result* pResult) const = 0;

    /// Opens an external shared memory object which is created by a compatible device, such as D3D device.
    /// There could be more than one underlying allocations in the shared memory object, only one allocation
    /// created on the device's GPU will be opened, other allocations will be ignored.
    ///
    /// @param [in]  openInfo       Open info.
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetExternalSharedGpuMemorySize()
    ///                              with the same params.
    /// @param [out] pMemCreateInfo Return CreateInfo of the external shared GPU memory.
    /// @param [out] ppGpuMemory    Constructed GPU memory object.  When successful, the returned address will be the
    ///                             same as specified in pPlacementAddr.
    ///
    /// @returns Success if the shared memory was successfully opened for access on this device.  Otherwise, one of the
    ///          following errors may be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr, ppGpuMemory or ppGpuMemory is null.
    ///          + ErrorNotShareable if none of allocations in the shared memory object is created on the device's GPU.
    virtual Result OpenExternalSharedGpuMemory(
        const ExternalGpuMemoryOpenInfo& openInfo,
        void*                            pPlacementAddr,
        GpuMemoryCreateInfo*             pMemCreateInfo,
        IGpuMemory**                     ppGpuMemory) = 0;

    /// Determines the amount of system memory required for a proxy GPU memory object to a GPU memory object on a
    /// different GPU.  An allocation of this amount of memory must be provided in the pPlacementAddr parameter of
    /// OpenPeerGpuMemory().
    ///
    /// @param [in]  openInfo Specifies a handle to a GPU memory object to open for peer-to-peer transfer access.
    /// @param [out] pResult  The validation result if pResult is non-null. This argument can be null to avoid the
    ///                       additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for opening a peer IGpuMemory object with the specified
    ///          properties.  A return value of 0 indicates the openInfo was invalid.
    virtual size_t GetPeerGpuMemorySize(
        const PeerGpuMemoryOpenInfo& openInfo,
        Result*                      pResult) const = 0;

    /// Opens previously created GPU memory object for peer access on another device.
    ///
    /// @param [in]  openInfo       Specifies a handle to a shared GPU memory object to open.
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetPeerGpuMemorySize() with the same
    ///                             params.
    /// @param [out] ppGpuMemory    Constructed GPU memory object.  When successful, the returned address will be the
    ///                             same as specified in pPlacementAddr.
    ///
    /// @returns Success if the memory was successfully opened for peer access on this device.  Otherwise, one of the
    ///          following errors may be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr, ppGpuMemory, or openInfo.pOriginalMem is null.
    virtual Result OpenPeerGpuMemory(
        const PeerGpuMemoryOpenInfo& openInfo,
        void*                        pPlacementAddr,
        IGpuMemory**                 ppGpuMemory) = 0;

    /// Determines the amount of system memory required for an image object.  An allocation of this amount of memory
    /// must be provided in the pPlacementAddr parameter of CreateImage().
    ///
    /// @param [in]  createInfo Properties of the new image such as pixel format and dimensions.
    /// @param [out] pResult    The validation result if pResult is non-null. This argument can be null to avoid the
    ///                         additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an @ref IImage object with the specified properties.  A
    ///          return value of 0 indicates the createInfo was invalid.
    virtual size_t GetImageSize(
        const ImageCreateInfo& createInfo,
        Result*                pResult) const = 0;

    /// Determines if the image will have faster clone copy (than normal CS or graphics copy) in @ref CmdCopyImage().
    /// If this function returns true, clients can set @ref ImageCreateInfo::flags::cloneable = 1 to enable clone
    /// copy support in @ref CmdCopyImage(); however whether @ref CmdCopyImage() goes through clone copy or not depends
    /// on multiple factors, please see doxygen in @ref CmdCopyImage() for more details.
    ///
    /// Note that clients are free to set this flag to 1 regardless; it still enables clone copy potentially but can't
    /// guarantee better performance. If clients find more faster clone copy cases, feel free to improve this function
    /// to cover these kinds of images.
    ///
    /// Due to rely on @ref ImageCreateInfo values to make decision, ideally clients should set all other members
    /// for @ref ImageCreateInfo first, and then call this function to setup cloneable flag last before creating image.
    ///
    /// @param [in]  createInfo Properties of the new image.
    ///
    /// @returns true if image prefers clone copy in CmdCopyImage(); otherwise goes through normal copy.
    virtual bool ImagePrefersCloneCopy(
        const ImageCreateInfo& createInfo) const = 0;

    /// Creates an @ref IImage object with the requested properties.
    ///
    /// @param [in]  createInfo     Properties of the new image such as pixel format and dimensions.
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetImageSize() with the same
    ///                             createInfo param.
    /// @param [out] ppImage        Constructed image object.  When successful, the returned address will be the same as
    ///                             specified in pPlacementAddr.
    ///
    /// @returns Success if the image was successfully created.  Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidValue if:
    ///              - The image dimensions are invalid based on the image type.
    ///              - The image dimensions are not properly aligned for compressed formats.
    ///              - The number of samples is invalid for the image type and format.
    ///              - MSAA is enabled for an image that doesn't support color or depth usage.
    ///              - MSAA images have more than one mip level.
    ///              - The array size is zero, non-1 for 3D images, or beyond the max number of slices for 1D or 2D
    ///                images.
    ///              - The number of mipmaps is invalid for the image dimensions.
    ///          + ErrorInvalidPointer if pPlacementAddr or ppImage is null.
    ///          + ErrorInvalidFormat if:
    ///              - The format doesn't support the usage flags.
    ///              - A 1D image specifies a compressed format.
    ///          + ErrorInvalidFlags if:
    ///              - The color target and depth/stencil usages are specified simultaneously.
    ///              - The color target flag is set for a 1D image.
    ///              - The depth/stencil flag is set for a non-2D image.
    virtual Result CreateImage(
        const ImageCreateInfo& createInfo,
        void*                  pPlacementAddr,
        IImage**               ppImage) = 0;

    /// Determines the amount of system memory required for a presentable image object (and an associated memory
    /// object).  Allocations of these amounts of memory must be provided in the pImagePlacementAddr and
    /// pGpuMemoryPlacementAddr parameters of CreatePresentableImage().
    ///
    /// Only images created through this interface are valid sources for IQueue::Present().
    ///
    /// @param [in]  createInfo     Properties of the image to create such as width/height and pixel format.
    /// @param [out] pImageSize     Size, in bytes, of system memory required for the IImage.
    ///                             Should be specified to the pImagePlacementAddr argument of CreatePresentableImage().
    /// @param [out] pGpuMemorySize Size, in bytes, of system memory required for a IGpuMemory object attached to the
    ///                             presentable IImage.  Should be specified to the pGpuMemoryPlacementAddr argument
    ///                             of CreatePresentableImage().
    /// @param [out] pResult        The validation result if pResult is non-null. This argument can be null to avoid the
    ///                             additional validation.
    virtual void GetPresentableImageSizes(
        const PresentableImageCreateInfo& createInfo,
        size_t*                           pImageSize,
        size_t*                           pGpuMemorySize,
        Result*                           pResult) const = 0;

    /// Creates a presentable image. Presentable image must have internally bound GPU memory allocated as OS needs the
    /// information of image/memory via OS callbacks.
    ///
    /// @param [in]  createInfo              Properties of the image to create such as width/height and pixel format.
    /// @param [in]  pImagePlacementAddr     Pointer to the location where PAL should construct this object.  There must
    ///                                      be as much size available here as reported by calling
    ///                                      GetPresentableImageSizes().
    /// @param [in]  pGpuMemoryPlacementAddr Pointer to the location where PAL should construct a IGpuMemory associated
    ///                                      with this presentable image.  There must be as much size available here as
    ///                                      reported by calling GetPresentableImageSizes().
    /// @param [out] ppImage                 Constructed image object.
    /// @param [out] ppGpuMemory             Constructed memory object.  This object is only valid for specifying in a
    ///                                      memory reference list.  It must be destroyed when the image is destroyed.
    ///
    /// @returns Success if the image was successfully created.  Otherwise, one of the following errors may be returned:
    ///          + TooManyFlippableAllocations if the image was successfully created, but the client has reached the
    ///            limit of flippable allocations for this Device.  This is a warning that future presentable Images
    ///            may fail to be created due to internal OS limitations.
    ///          + ErrorTooManyPresentableImages if the swap chain cannot be associated with more presentable images.
    ///          + ErrorInvalidPointer if pImagePlacementAddr, pGpuMemoryPlacementAddr, ppImage, or ppGpuMemory is null.
    ///          + ErrorInvalidValue if:
    ///              - The image dimensions are invalid.
    ///              - The refresh rate is invalid for a fullscreen image.
    ///          + ErrorInvalidFormat if the format doesn't support presentation.
    virtual Result CreatePresentableImage(
        const PresentableImageCreateInfo& createInfo,
        void*                             pImagePlacementAddr,
        void*                             pGpuMemoryPlacementAddr,
        IImage**                          ppImage,
        IGpuMemory**                      ppGpuMemory) = 0;

    /// Determines the amount of system memory required for an image object (and an associated memory object) opened for
    /// peer access to an image created on another GPU.  Allocations of these amounts of memory must be provided in the
    /// pImagePlacementAddr and pGpuMemoryPlacementAddr parameters of OpenPeerImage().
    ///
    /// @param [in]  openInfo           Specifies the image to be opened for peer access from another GPU.
    /// @param [out] pPeerImageSize     Size, in bytes, of system memory required for a peer IImage.  Should be
    ///                                 specified to the pImagePlacementAddr argument to OpenPeerImage().
    /// @param [out] pPeerGpuMemorySize Size, in bytes, of system memory required for a dummy IGpuMemory object attached
    ///                                 to a peer IImage.  Should be specified to the pGpuMemoryPlacementAddr argument
    ///                                 to OpenPeerImage().
    /// @param [out] pResult            The validation result if pResult is non-null. This argument can be null to avoid
    ///                                 the additional validation.
    virtual void GetPeerImageSizes(
        const PeerImageOpenInfo& openInfo,
        size_t*                  pPeerImageSize,
        size_t*                  pPeerGpuMemorySize,
        Result*                  pResult) const = 0;

    /// Creates an @ref IImage object as a proxy to an IImage on another GPU to be used for peer-to-peer transfers.
    ///
    /// @note The @ref IImage object provided in the @ref PeerImageOpenInfo must be bound to an existing
    ///       @ref IGpuMemory object prior to opening a peer version of it.
    ///       If the new image needs to bind to an existing peer memory allocation, as when images are suballocated,
    ///       pGpuMemoryPlacementAddr must equal nullptr and ppGpuMemory must point to an existing IGpuMemory object
    ///       that was previously opened to reference the same memory from the current device.
    /// @param [in]  openInfo                Specifies the image to be opened for peer access from another GPU.
    /// @param [in]  pImagePlacementAddr     Pointer to the location where PAL should construct this object.  There must
    ///                                      be as much size available here as reported by calling GetPeerImageSizes().
    /// @param [in]  pGpuMemoryPlacementAddr If nonzero, this is a pointer to the location where PAL should construct a
    ///                                      IGpuMemory to be associated with this peer image. There must be as much
    ///                                      size available here as reported by calling GetPeerImageSizes().
    /// @param [out] ppImage                 Constructed image object.
    /// @param [in]  ppGpuMemory             Constructed dummy memory object.  This object is only valid for specifying
    ///                                      in a memory reference list.
    ///
    /// @returns Success if the image was successfully created.  Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidPointer if pImagePlacementAddr, pGpuMemoryPlacementAddr, ppImage, ppGpuMemory, or
    ///            openInfo.pOriginalImage is null.
    virtual Result OpenPeerImage(
        const PeerImageOpenInfo& openInfo,
        void*                    pImagePlacementAddr,
        void*                    pGpuMemoryPlacementAddr,
        IImage**                 ppImage,
        IGpuMemory**             ppGpuMemory) = 0;

    /// Determines the amount of system memory required for an external shared image object (and an associated memory
    /// object).  Allocations of these amounts of memory must be provided in the pImagePlacementAddr and
    /// pGpuMemoryPlacementAddr parameters of OpenExternalSharedImage().
    ///
    /// @param [in]  openInfo       Specifies the external image to be opened.
    /// @param [out] pImageSize     Size, in bytes, of system memory required for pImagePlacementAddr.
    /// @param [out] pGpuMemorySize Size, in bytes, of system memory required for pGpuMemoryPlacementAddr.
    /// @param [out] pImgCreateInfo If non-null, it will be filled out with information describing the shared image.
    /// @returns Success if the shared image was successfully opened for access on this device.  Otherwise, one of the
    ///          following errors may be returned:
    ///          + ErrorInvalidPointer if pImageSize or pGpuMemorySize is null.
    ///          + ErrorNotShareable if none of allocations in the shared image is created on the device's GPU.
    virtual Result GetExternalSharedImageSizes(
        const ExternalImageOpenInfo& openInfo,
        size_t*                      pImageSize,
        size_t*                      pGpuMemorySize,
        ImageCreateInfo*             pImgCreateInfo) const = 0;

    /// Opens an external shared image object which was created by a compatible device, such as D3D device.
    /// There could be more than one underlying allocations in the shared image object, only one allocation
    /// created on the device's GPU will be opened, other allocations will be ignored.
    ///
    /// @param [in]  openInfo                Specifies the external image to be opened.
    /// @param [in]  pImagePlacementAddr     Pointer to the location where PAL should construct the image object.
    ///                                      There must be as much space available here as reported by calling
    ///                                      GetExternalSharedImageSizes() with the same params.
    /// @param [in]  pGpuMemoryPlacementAddr Pointer to the location where PAL should construct the GPU memory object.
    ///                                      There must be as much space available here as reported by calling
    ///                                      GetExternalSharedImageSizes() with the same params.
    /// @param [out] pMemCreateInfo          If non-null, it is filled with information describing the external GPU
    ///                                      memory that backs this external image.
    /// @param [out] ppImage                 Constructed image object.  When successful, the returned address will
    ///                                      be the same as specified in pImagePlacementAddr.
    /// @param [out] ppGpuMemory             Constructed GPU memory object.  When successful, the returned address will
    ///                                      be the same as specified in pGpuMemoryPlacementAddr.
    /// @returns Success if the shared image was successfully opened for access on this device. Otherwise, one of the
    ///          following errors may be returned:
    ///          + ErrorInvalidPointer if pImagePlacementAddr, pGpuMemoryPlacementAddr, ppImage or ppGpuMemory is null.
    ///          + ErrorNotShareable if none of allocations in the shared image object is created on the device's GPU.
    virtual Result OpenExternalSharedImage(
        const ExternalImageOpenInfo& openInfo,
        void*                        pImagePlacementAddr,
        void*                        pGpuMemoryPlacementAddr,
        GpuMemoryCreateInfo*         pMemCreateInfo,
        IImage**                     ppImage,
        IGpuMemory**                 ppGpuMemory) = 0;

    /// Determines the amount of system memory required for a color target view object.  An allocation of this amount of
    /// memory must be provided in the pPlacementAddr parameter of CreateColorTargetView().
    ///
    /// Unlike most creation methods in this class, GetColorTargetViewSize() does not require a ColorTargetCreateInfo
    /// parameter.  PAL must guarantee that all color target view objects are the same size in order to support DX12,
    /// where these views are treated similarly to SRDs.
    ///
    /// @param [out] pResult            The validation result if pResult is non-null. This argument can be null to avoid
    ///                                 the additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an IColorTargetView object.
    virtual size_t GetColorTargetViewSize(
        Result* pResult) const = 0;

    /// Creates an @ref IColorTargetView object with the requested properties.
    ///
    /// @param [in]  createInfo        Properties of the color target view to create.
    /// @param [in]  pPlacementAddr    Pointer to the location where PAL should construct this object.  There must be as
    ///                                much size available here as reported by calling GetColorTargetViewSize().
    /// @param [out] ppColorTargetView Constructed color target view object.  When successful, the returned address will
    ///                                be the same as specified in pPlacementAddr.
    ///
    /// @returns Success if the color target view was successfully created.  Otherwise, one of the following
    ///          errors may be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr, ppColorTargetView, or createInfo.pImage is null.
    ///          + ErrorInvalidValue if:
    ///              - The base slice is invalid for the given image object and view type.
    ///              - The number of array slices is zero or the range of slices is too large for the specified image.
    ///              - The mip level is invalid for the given image object.
    ///          + ErrorInvalidImage if the image object doesn't have the color target access flag set.
    virtual Result CreateColorTargetView(
        const ColorTargetViewCreateInfo& createInfo,
        void*                            pPlacementAddr,
        IColorTargetView**               ppColorTargetView) const = 0;

    /// Determines the amount of system memory required for a depth/stencil view object.  An allocation of this amount
    /// of memory must be provided in the pPlacementAddr parameter of CreateDepthStencilView().
    ///
    /// Unlike most creation methods in this class, GetDepthStencilViewSize() does not require a
    /// DepthStencilViewCreateInfo parameter.  PAL must guarantee that all color target view objects are the same size
    /// in order to support DX12, where these views are treated similarly to SRDs.
    ///
    /// @param [out] pResult            The validation result if pResult is non-null. This argument can be null to avoid
    ///                                 the additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an IDepthStencilView object.
    virtual size_t GetDepthStencilViewSize(
        Result* pResult) const = 0;

    /// Creates an @ref IDepthStencilView object with the requested properties.
    ///
    /// @param [in]  createInfo         Properties of the depth/stencil view to create.
    /// @param [in]  pPlacementAddr     Pointer to the location where PAL should construct this object.  There must be
    ///                                 as much size available here as reported by calling GetDepthStencilViewSize().
    /// @param [out] ppDepthStencilView Constructed depth/stencil view object.  When successful, the returned address
    ///                                 will be the same as specified in pPlacementAddr.
    ///
    /// @returns Success if the depth/stencil view was successfully created.  Otherwise, one of the following errors may
    ///          be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr, ppDepthStencilView, or createInfo.pImage is null.
    ///          + ErrorInvalidValue if:
    ///              - The base slice is invalid for the given image object and view type.
    ///              - The number of array slices is zero or the range of slices is too large for the specified image.
    ///              - The mip level is invalid for the given image object.
    ///          + ErrorInvalidImage if the image object doesn't have the depth/stencil target access flag set.
    virtual Result CreateDepthStencilView(
        const DepthStencilViewCreateInfo& createInfo,
        void*                             pPlacementAddr,
        IDepthStencilView**               ppDepthStencilView) const = 0;

    /// Creates one or more typed buffer view _shader resource descriptors (SRDs)_ in memory provided by the client.
    ///
    /// The client is responsible for providing _count_ times the amount of memory reported by srdSizes.typedBufferView
    /// in DeviceProperties, and must also ensure the provided memory is aligned to the size of one SRD.
    ///
    /// The SRD can be created in either system memory or pre-mapped GPU memory.  If updating GPU memory, the client
    /// must ensure there are no GPU accesses of this memory in flight before calling this method.
    ///
    /// The generated buffer view SRD allows a range of a GPU memory allocation to be accessed by a shader, and should
    /// be setup based on shader usage as described in @ref BufferViewInfo.  The client should put the resulting SRD
    /// in an appropriate location based on the shader resource mapping specified by the bound pipeline, either directly
    /// in user data (ICmdBuffer::CmdSetUserData()) or a table in GPU memory indirectly referenced by user data.
    ///
    /// For performance reasons, this method returns void and does minimal error-checking. However, in debug builds,
    /// to assist clients' debug efforts, the following conditions will be checked with runtime assertions:
    ///     + If pBufferViewInfo or pOut, is null.
    ///     + If count is 0.
    ///     + If pBufferViewInfo[].format is Undefined.
    ///     + If pBufferViewInfo[].stride does not match the size of an element of that format.
    ///     + If pBufferViewInfo[].gpuAddr is 0.
    ///     + If pBufferViewInfo[].gpuAddr is not properly aligned to Min(4, pBufferViewInfo[].stride).
    ///
    /// @param [in]  count           Number of buffer view SRDs to create; size of the pBufferViewInfo array.
    /// @param [in]  pBufferViewInfo Array of buffer view descriptions directing SRD construction.
    /// @param [out] pOut            Client-provided space where opaque, hardware-specific SRD data is written.
    ///
    /// @ingroup ResourceBinding
    void CreateTypedBufferViewSrds(
        uint32                count,
        const BufferViewInfo* pBufferViewInfo,
        void*                 pOut) const
        { m_pfnTable.pfnCreateTypedBufViewSrds(this, count, pBufferViewInfo, pOut); }

    /// Creates one or more untyped buffer view _shader resource descriptors (SRDs)_ in memory provided by the client.
    /// These SRDs can be accessed in a shader as either _raw_ or _structured_ views.
    ///
    /// The client is responsible for providing _count_ times the amount of memory reported by srdSizes.untypedBufferView
    /// in DeviceProperties, and must also ensure the provided memory is aligned to the size of one SRD.
    ///
    /// The SRD can be created in either system memory or pre-mapped GPU memory.  If updating GPU memory, the client
    /// must ensure there are no GPU accesses of this memory in flight before calling this method.
    ///
    /// The generated buffer view SRD allows a range of a GPU memory allocation to be accessed by a shader, and should
    /// be setup based on shader usage as described in @ref BufferViewInfo.  The client should put the resulting SRD
    /// in an appropriate location based on the shader resource mapping specified by the bound pipeline, either directly
    /// in user data (ICmdBuffer::CmdSetUserData()) or a table in GPU memory indirectly referenced by user data.
    ///
    /// For performance reasons, this method returns void and does minimal error-checking. However, in debug builds,
    /// to assist clients' debug efforts, the following conditions will be checked with runtime assertions:
    ///     + If pBufferViewInfo or pOut, is null.
    ///     + If count is 0.
    ///     + If pBufferViewInfo[].format is not Undefined.
    ///     + If pBufferViewInfo[].gpuAddr is 0.
    ///     + If pBufferViewInfo[].gpuAddr is not properly aligned to Min(4, pBufferViewInfo[].stride).
    ///
    /// @param [in]  count           Number of buffer view SRDs to create; size of the pBufferViewInfo array.
    /// @param [in]  pBufferViewInfo Array of buffer view descriptions directing SRD construction.
    /// @param [out] pOut            Client-provided space where opaque, hardware-specific SRD data is written.
    ///
    /// @ingroup ResourceBinding
    void CreateUntypedBufferViewSrds(
        uint32                count,
        const BufferViewInfo* pBufferViewInfo,
        void*                 pOut) const
        { m_pfnTable.pfnCreateUntypedBufViewSrds(this, count, pBufferViewInfo, pOut); }

    /// Validates image view SRD input parameters.  Error checking for image view SRDs is handled by a separate
    /// function for performance reasons and to avoid rechecking parameters that the client knows are correct when
    /// rebuilding SRDs.
    ///
    /// @param [in] viewInfo        Input image view SRD parameter info.
    ///
    /// @returns Success if the parameters pass validation.  Otherwise, one of the following errors may be returned:
    ///     + ErrorImagePlaneUnavailable if the requested image plane specified in the view is not available on
    ///       the image.
    ///     + ErrorImageNotShaderAccessible if the image does not have a shader-readable or shader-writable usage.
    ///     + ErrorInvalidFormatSwizzle if the view's channel swizzle specifies components not available in the view
    ///       format.
    ///     + ErrorInvalidBaseMipLevel if the view's start subresource has a mip level larger than the number of
    ///       of available mip levels.
    ///     + ErrorFormatIncompatibleWithImageFormat if the view's format is not compatible with the image's format.
    ///       This can happen if:
    ///         - For color plane views, the bit-depths of the two formats are not equal.
    ///     + ErrorFormatIncompatibleWithImagePlane if the view's format is not compatible with the image's plane.
    ///       This can happen if:
    ///         - For depth plane views, the bit-depths of the view format and the depth component of the image
    ///           are not equal.
    ///         - For stencil plane views, the bit-depths of the view format and the stencil component of the image
    ///           are not equal.
    ///     + ErrorInvalidViewArraySize if:
    ///         - The view array size is 0.
    ///         - The image type is 3D and the view array size is not 1.
    ///     + ErrorViewTypeIncompatibleWithImageType if:
    ///         - The image type is 1D and the view type is not 1D
    ///         - The image type is 2D and the view type is not 2D or cubemap
    ///         - The image type is 3D and the view type is not 3D
    ///     + ErrorInsufficientImageArraySize if the number of viewed array slices is more than available on
    ///       the image.
    ///     + ErrorCubemapIncompatibleWithMsaa if the view type is a cubemap view and the image has
    ///       multiple samples.
    ///     + ErrorCubemapNonSquareFaceSize if the view type is a cubemap view and the image 2D extents are not
    ///       square.
    ///     + ErrorInvalidViewBaseSlice
    ///         - If the image type is 3D and the view base slice is not 0.
    ///
    /// @ingroup ResourceBinding
    virtual Result ValidateImageViewInfo(const ImageViewInfo& viewInfo) const = 0;

    /// Creates one or more image view _shader resource descriptors (SRDs)_ in memory provided by the client.
    ///
    /// The client is responsible for providing _count_ times the amount of memory reported by srdSizes.imageView
    /// in DeviceProperties, and must also ensure the provided memory is aligned to the size of one SRD.
    ///
    /// The SRD can be created in either system memory or pre-mapped GPU memory.  If updating GPU memory, the client
    /// must ensure there are no GPU accesses of this memory in flight before calling this method.
    ///
    /// The generated image view SRD allows a set of subresources in an image to be accessed by a shader, and should
    /// be setup as described in @ref ImageViewInfo.  The client should put the resulting SRD in an appropriate
    /// location based on the shader resource mapping specified by the bound pipeline, either directly in user data
    /// (ICmdBuffer::CmdSetUserData()) or a table in GPU memory indirectly referenced by user data.
    ///
    /// @warning SRDs for Planar YUV images will include padding if pImageViewInfo->subresRange.numSlices > 1
    ///
    /// @param [in]  count        Number of buffer view SRDs to create; size of the pImageViewInfo array.
    /// @param [in]  pImgViewInfo Array of image view descriptions directing SRD construction.
    /// @param [out] pOut         Client-provided space where opaque, hardware-specific SRD data is written.
    ///
    /// @ingroup ResourceBinding
    void CreateImageViewSrds(
        uint32               count,
        const ImageViewInfo* pImgViewInfo,
        void*                pOut) const
        { m_pfnTable.pfnCreateImageViewSrds(this, count, pImgViewInfo, pOut); }

    /// Validates an fmask view SRD input parameters.  Error checking for fmask view SRDs is handled by a separate
    /// function for performance reasons and to avoid rechecking parameters that the client knows are correct when
    /// rebuilding SRDs.
    ///
    /// @param [in] viewInfo        Input image view SRD parameter info.
    ///
    /// @returns Success if the parameters pass validation.  Otherwise, one of the following errors may be returned:
    ///     + ErrorImageFmaskUnavailable if the image does not have an FMask.
    ///     + ErrorInvalidViewArraySize if the view array size is 0.
    ///     + ErrorViewTypeIncompatibleWithImageType if the image type is not 2D.
    ///     + ErrorInsufficientImageArraySize if the view base array slice and size define an out of bounds array range.
    ///
    /// @ingroup ResourceBinding
    virtual Result ValidateFmaskViewInfo(const FmaskViewInfo& viewInfo) const = 0;

    /// Creates one or more fmask view _shader resource descriptors (SRDs)_ in memory provided by the client.
    ///
    /// The client is responsible for providing _count_ times the amount of memory reported by srdSizes.fmaskView
    /// in DeviceProperties, and must also ensure the provided memory is aligned to the size of one SRD.
    ///
    /// The SRD can be created in either system memory or pre-mapped GPU memory.  If updating GPU memory, the client
    /// must ensure there are no GPU accesses of this memory in flight before calling this method.
    ///
    /// The generated fmask view SRD allows a range of image slices to be accessed bo the load_fptr IL instruction,
    /// which allows a shader to read compressed MSAA data at the expense of a texture indirection.  This SRD should be
    /// setup as described in @ref FmaskViewInfo.  The client should put the resulting SRD in an appropriate
    /// location based on the shader resource mapping specified by the bound pipeline, either directly in user data
    /// (ICmdBuffer::CmdSetUserData()) or a table in GPU memory indirectly referenced by user data.
    ///
    /// @param [in]  count          Number of fmask view SRDs to create; size of the pFmaskViewInfo array.
    /// @param [in]  pFmaskViewInfo Array of fmask view descriptions directing SRD construction.
    /// @param [out] pOut           Client-provided space where opaque, hardware-specific SRD data is written.
    ///
    /// @ingroup ResourceBinding
    void CreateFmaskViewSrds(
        uint32               count,
        const FmaskViewInfo* pFmaskViewInfo,
        void*                pOut) const
        { m_pfnTable.pfnCreateFmaskViewSrds(this, count, pFmaskViewInfo, pOut); }

    /// Validates a sampler SRD input parameters.  Error checking for sampler SRDs is handled by a separate function for
    /// performance reasons and to avoid rechecking parameters that the client knows are correct when rebuilding SRDs.
    ///
    /// @param [in] samplerInfo Input sampler SRD parameter info.
    ///
    /// @returns Success if the parameters pass validation.  Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidValue if:
    ///              - The max anisotropy or LOD bias value is outside of the legal range.
    ///              - The min/max LOD values are outside the legal range or if the max LOD is smaller than the min LOD.
    ///              - The border color palette index is out of the legal range.
    ///
    /// @ingroup ResourceBinding
    virtual Result ValidateSamplerInfo(const SamplerInfo& samplerInfo) const = 0;

    /// Creates one or more sampler _shader resource descriptors (SRDs)_ in memory provided by the client.
    ///
    /// The client is responsible for providing _count_ times the amount of memory reported by srdSizes.sampler in
    /// DeviceProperties, and must also ensure the provided memory is aligned to the size of one SRD.
    ///
    /// The SRD can be created in either system memory or pre-mapped GPU memory.  If updating GPU memory, the client
    /// must ensure there are no GPU accesses of this memory in flight before calling this method.
    ///
    /// The generated sampler SRD controlls execution of sample instructions in a shader, and should be setup as
    /// described in @ref SamplerInfo.  The client should put the resulting SRD in an appropriate location based on the
    /// shader resource mapping specified by the bound pipeline, either directly in user data
    /// (ICmdBuffer::CmdSetUserData()) or a table in GPU memory indirectly referenced by user data.
    ///
    /// @param [in]  count        Number of sampler SRDs to create; size of the pSamplerInfo array.
    /// @param [in]  pSamplerInfo Array of sampler descriptions directing SRD construction.
    /// @param [out] pOut         Client-provided space where opaque, hardware-specific SRD data is written.
    ///
    /// @returns Success if the sampler SRD data was successfully written to pOut.  Otherwise, one of the following
    ///          errors may be returned:
    ///          + ErrorInvalidPointer if pSamplerInfo or pOut is null.
    ///          + ErrorInvalidValue if:
    ///              - The max anisotropy or LOD bias value is outside of the legal range.
    ///              - The min/max LOD values are outside the legal range or if the max LOD is smaller than the min LOD.
    ///              - The border color palette index is out of the legal range.
    ///
    /// @ingroup ResourceBinding
    void CreateSamplerSrds(
        uint32             count,
        const SamplerInfo* pSamplerInfo,
        void*              pOut) const
        { m_pfnTable.pfnCreateSamplerSrds(this, count, pSamplerInfo, pOut); }

    /// Creates one or more _BVH resource descriptors (SRDs)_ in memory provided by the client.
    ///
    /// The client is responsible for providing _count_ times the amount of memory reported by srdSizes.bvhInfo in
    /// DeviceProperties, and must also ensure the provided memory is aligned to the size of one SRD.
    ///
    /// The SRD can be created in either system memory or pre-mapped GPU memory.  If updating GPU memory, the client
    /// must ensure there are no GPU accesses of this memory in flight before calling this method.
    ///
    /// The generated BVH SRD controls execution of ray trace instructions in a shader, and should be setup as
    /// described in @ref BvhInfo.  The client should put the resulting SRD in an appropriate location based on
    /// the shader resource mapping specified by the bound pipeline, either directly in user data
    /// (ICmdBuffer::CmdSetUserData()) or a table in GPU memory indirectly referenced by user data.
    ///
    /// @param [in]  count    Number of BVH SRDs to create; size of the pBvhInfo array.
    /// @param [in]  pBvhInfo Array of BVH (bounding volume hierarchy) descriptions directing SRD construction.
    /// @param [out] pOut     Client-provided space where opaque, hardware-specific SRD data is written.
    ///
    /// @returns Success if the sampler SRD data was successfully written to pOut.  Otherwise, one of the following
    ///          errors may be returned:
    ///          + ErrorInvalidPointer if pBvhInfo or pOut is null.
    ///
    /// @ingroup ResourceBinding
    void CreateBvhSrds(
        uint32         count,
        const BvhInfo* pBvhInfo,
        void*          pOut) const
    {
        m_pfnTable.pfnCreateBvhSrds(this, count, pBvhInfo, pOut);
    }

    /// Decode a buffer SRD back into most of the BufferViewInfo used to create it via Create{Typed,Untyped}BufferViewSrds.
    ///
    /// @param [in]  pBufferViewSrd Pointer to the SRD. Should not be in dedicated GPU memory.
    /// @param [out] pViewInfo      Pointer to memory to be filled with decoded info.
    void DecodeBufferViewSrd(
        const void*     pBufferViewSrd,
        BufferViewInfo* pViewInfo) const
    {
        m_pfnTable.pfnDecodeBufferViewSrd(this, pBufferViewSrd, pViewInfo);
    }

    /// Decode an image SRD back into the main parameters used to create it via CreateImageViewSrds().
    ///
    /// @param [in]  image          The same image the SRD was created on.
    /// @param [in]  pImageViewSrd  Pointer to the SRD. Should not be in dedicated GPU memory.
    /// @param [out] pDecodedInfo   Pointer to memory to be filled with decoded info.
    void DecodeImageViewSrd(
        const IImage&    image,
        const void*      pImageViewSrd,
        DecodedImageSrd* pDecodedInfo) const
    {
        m_pfnTable.pfnDecodeImageViewSrd(this, &image, pImageViewSrd, pDecodedInfo);
    }

    /// The MSAA sample pattern palette is a client-managed table of sample patterns that might be in use by the app.
    ///
    /// The only purpose of this palette is to implement the samplepos shader instruction.  This instruction returns the
    /// position of a particular sample based on the sample pattern of the current rasterizer state or a particular
    /// specified resource.  When this instruction is executed, the shader will determine the correct palette index
    /// as specified in the pipeline (see samplePatternIdx in the rsState structure inside GraphicsPipelineCreateInfo)
    /// or in the image view SRD (see samplePatternIdx in ImageViewInfo).  The shader will then return the position
    /// for the specified sample in the specified entry of the currently bound sample pattern palette as set with this
    /// function.
    ///
    /// The initial bound sample pattern palette values are undefined.  A palette entry must be specified before it is
    /// referenced by a samplepos instruction.
    ///
    /// @warning The samplepos instruction and sample pattern palette both assume every pixel has the same pattern.
    ///          This may not be accurate if the application uses custom sample patterns that aren't identical for
    ///          every pixel in the quad.
    ///
    /// @param [in] palette Small set of sample patterns.  Each sample pattern consists of an x,y tuple per sample
    ///                     describing where it is located in a pixel.  The coordinate system is described in
    ///                     MsaaStateCreateInfo.
    ///
    /// @returns Success if the palette was successfully updated.  An error is only possible due to an internal error,
    ///          such as a failure to allocate GPU memory for a new table.
    virtual Result SetSamplePatternPalette(
        const SamplePatternPalette& palette) = 0;

    /// Determines the amount of system memory required for a border color palette object.  An allocation of this amount
    /// of memory must be provided in the pPlacementAddr parameter of CreateBorderColorPalette().
    ///
    /// @param [in]  createInfo Border color palette creation info (specifies number of entries).
    /// @param [out] pResult    The validation result if pResult is non-null. This argument can be null to avoid the
    ///                         additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an IBorderColorPalette object with the specified
    ///          properties.  A return value of 0 indicates the createInfo was invalid.
    virtual size_t GetBorderColorPaletteSize(
        const BorderColorPaletteCreateInfo& createInfo,
        Result*                             pResult) const = 0;

    /// Creates a border color palette object.
    ///
    /// @param [in]  createInfo     Border color palette creation info (number of entries).
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetBorderColorPaletteSize() with the
    ///                             same createInfo param.
    /// @param [out] ppPalette      Constructed border color palette object.  When successful, the returned address will
    ///                             be the same as specified in pPlacementAddr.
    ///
    /// @returns Success if the border color palette was successfully created.  Otherwise, one of the following errors
    ///          may be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppPalette is null.
    ///          + ErrorInvalidValue if the palette size is too large to be used on any queue on this device.
    virtual Result CreateBorderColorPalette(
        const BorderColorPaletteCreateInfo& createInfo,
        void*                               pPlacementAddr,
        IBorderColorPalette**               ppPalette) const = 0;

    /// Determines the amount of system memory required for a compute pipeline object.  An allocation of this amount of
    /// memory must be provided in the pPlacementAddr parameter of CreateComputePipeline().
    ///
    /// @param [in]  createInfo Pipeline properties including shaders and descriptor set mappings.
    /// @param [out] pResult    The validation result if pResult is non-null. This argument can be null to avoid the
    ///                         additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an IPipeline object with the specified properties.  A
    ///          return value of 0 indicates the createInfo was invalid.
    virtual size_t GetComputePipelineSize(
        const ComputePipelineCreateInfo& createInfo,
        Result*                          pResult) const = 0;

    /// Creates a compute @ref IPipeline object with the requested properties.
    ///
    /// @param [in]     createInfo     Pipeline properties including shaders and descriptor set mappings.
    /// @param [in]     pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                                much size available here as reported by calling GetComputePipelineSize() with the
    ///                                same createInfo param.
    /// @param [out]    ppPipeline     Constructed pipeline object.  When successful, the returned address will be the
    ///                                same as specified in pPlacementAddr.
    ///
    /// @returns Success if the pipeline was successfully created.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorInvalidPointer if:
    ///              - pPlacementAddr or ppPipeline is null.
    ///              - A required shader pointer is null.
    ///              - The link time constant data pointer is null.
    ///          + ErrorInvalidValue if:
    ///              - The link constant buffer info pointer isn't consistent with the link constant buffer count value.
    ///              - The dynamic memory view mapping slot object type is not unused, resource, of UAV.
    ///          + ErrorUnsupportedShaderIlVersion if an incorrect shader type is used in any shader stage.
    virtual Result CreateComputePipeline(
        const ComputePipelineCreateInfo& createInfo,
        void*                            pPlacementAddr,
        IPipeline**                      ppPipeline) = 0;

    /// Determines the amount of system memory required for a shader library object.  An allocation of this amount of
    /// memory must be provided in the pPlacementAddr parameter of CreateShaderLibrary().
    ///
    /// @param [in]  createInfo Library creation parameters including ELF code object and other items.
    /// @param [out] pResult    The validation result if pResult is non-null.  This argument can be null to avoid the
    ///                         additonal validation.
    ///
    /// @returns Size, in bytes, of system memory required for an IShaderLibrary object with the specified properties.
    ///          A return value of zero indicates the createInfo was invalid.
    virtual size_t GetShaderLibrarySize(
        const ShaderLibraryCreateInfo& createInfo,
        Result*                        pResult) const = 0;

    /// Creates a @ref IShaderLibrary object with the requested properties.
    ///
    /// @param [in]  createInfo     Library creation parameters including ELF code object and other items.
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetShaderLibrarySize() with the
    ///                             same createInfo parameter.
    /// @param [out] ppLibrary      Constructed library object.  When successful, the returned address will be the same
    ///                             as specified in pPlacementAddr.
    ///
    /// @returns Success if the library was successfully created.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorInvalidPointer if:
    ///              - pPlacementAddr or ppLibrary is null.
    ///              - Required code object pointer is null.
    virtual Result CreateShaderLibrary(
        const ShaderLibraryCreateInfo& createInfo,
        void*                          pPlacementAddr,
        IShaderLibrary**               ppLibrary) = 0;

    /// Determines the amount of system memory required for a graphics pipeline object.  An allocation of this amount of
    /// memory must be provided in the pPlacementAddr parameter of CreateGraphicsPipeline().
    ///
    /// @param [in]  createInfo Pipeline properties including shaders and descriptor set mappings.
    /// @param [out] pResult    The validation result if pResult is non-null. This argument can be null to avoid
    ///                         the additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an IPipeline object with the specified properties.  A
    ///          return value of 0 indicates the createInfo was invalid.
    virtual size_t GetGraphicsPipelineSize(
        const GraphicsPipelineCreateInfo& createInfo,
        Result*                           pResult) const = 0;

    /// Creates a graphics @ref IPipeline object with the requested properties.
    /// If this pipeline is created from a set of @IShaderLibrary objects, it is the caller's responsibility to ensure
    /// that the lifetime of the pipeline does not exceed the lifetime of the libraries themselves.
    ///
    /// @param [in]  createInfo     Pipeline properties including shaders and descriptor set mappings.
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetGraphicsPipelineSize() with the
    ///                             same createInfo param.
    /// @param [out] ppPipeline     Constructed pipeline object.  When successful, the returned address will be the same
    ///                             as specified in pPlacementAddr.
    ///
    /// @returns Success if the pipeline was successfully created.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorInvalidPointer if:
    ///              - pPlacementAddr or ppPipeline is null.
    ///              - A required shader pointer is null.
    ///              - The link time constant data pointer is null.
    ///          + ErrorInvalidValue if:
    ///              - The number of control points is invalid for a tessellation pipeline.
    ///              - Logic operations are enabled while some of the color targets enable blending.
    ///              - The dual source blend enable doesn't match expectations for color target and blend enable setup.
    ///              - The link constant buffer info pointer isn't consistent with the link constant buffer count value.
    ///              - The dynamic memory view mapping slot object type is not unused, resource, of UAV.
    ///          + ErrorInvalidFormat if:
    ///              - Blending is enabled by the color target format doesn't support blending.
    ///              - Logic operations are enabled by an incompatible format is used.
    ///          + ErrorUnsupportedShaderIlVersion if an incorrect shader type is used in any shader stage.
    virtual Result CreateGraphicsPipeline(
        const GraphicsPipelineCreateInfo& createInfo,
        void*                             pPlacementAddr,
        IPipeline**                       ppPipeline) = 0;

    /// Determines the amount of system memory required for a MSAA state object.  An allocation of this amount of memory
    /// must be provided in the pPlacementAddr parameter of CreateMsaaState().
    ///
    /// @returns Size, in bytes, of system memory required for an @ref IMsaaState object.
    ///          This value will always be non-zero if the device has GfxIp support.
    virtual size_t GetMsaaStateSize() const = 0;

    /// Creates an @ref IMsaaState object with the requested properties.
    ///
    /// @param [in]  createInfo      Properties of the MSAA state object to create.
    /// @param [in]  pPlacementAddr  Pointer to the location where PAL should construct this object.  There must be as
    ///                              much size available here as reported by calling GetMsaaStateSize() with the same
    ///                              createInfo param.
    /// @param [out] ppMsaaState     Constructed MSAA state object.  When successful, the returned address will be the
    ///                              same as specified in pPlacementAddr.
    ///
    /// @returns Success if the MSAA state was successfully created.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppMsaaState is null.
    ///          + ErrorInvalidValue if:
    ///              - The number of samples is unsupported.
    virtual Result CreateMsaaState(
        const MsaaStateCreateInfo& createInfo,
        void*                      pPlacementAddr,
        IMsaaState**               ppMsaaState) const = 0;

    /// Determines the amount of system memory required for a color blend state object.  An allocation of this amount of
    /// memory must be provided in the pPlacementAddr parameter of CreateColorBlendState().
    ///
    /// @returns Size, in bytes, of system memory required for an @ref IColorBlendState object.
    ///          This value will always be non-zero if the device has GfxIp support.
    virtual size_t GetColorBlendStateSize() const = 0;

    /// Creates an @ref IColorBlendState object with the requested properties.
    ///
    /// @param [in]  createInfo        Properties of the color blend state object to create.
    /// @param [in]  pPlacementAddr    Pointer to the location where PAL should construct this object.  There must be as
    ///                                much size available here as reported by calling GetColorBlendStateSize() with the
    ///                                same createInfo param.
    /// @param [out] ppColorBlendState Constructed color blend state object.  When successful, the returned address will
    ///                                be the same as specified in pPlacementAddr.
    ///
    /// @returns Success if the color blend state was successfully created.  Otherwise, one of the following errors may
    ///          be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppColorBlendState is null.
    ///          + ErrorInvalidValue if:
    ///              - An unsupported blend function is used with dual source blending.
    virtual Result CreateColorBlendState(
        const ColorBlendStateCreateInfo& createInfo,
        void*                            pPlacementAddr,
        IColorBlendState**               ppColorBlendState) const = 0;

    /// Determines the amount of system memory required for a depth/stencil state object.  An allocation of this amount
    /// of memory must be provided in the pPlacementAddr parameter of CreateDepthStencilState().
    ///
    /// @returns Size, in bytes, of system memory required for an @ref IDepthStencilState object.
    ///          This value will always be non-zero if the device has GfxIp support.
    virtual size_t GetDepthStencilStateSize() const = 0;

    /// Creates an @ref IDepthStencilState object with the requested properties.
    ///
    /// @param [in]  createInfo          Properties of the depth/stencil state object to create.
    /// @param [in]  pPlacementAddr      Pointer to the location where PAL should construct this object.  There must be
    ///                                  as much size available here as reported by calling GetDepthStencilStateSize()
    ///                                  with the same createInfo param.
    /// @param [out] ppDepthStencilState Constructed depth/stencil state object.  When successful, the returned address
    ///                                  will be the same as specified in pPlacementAddr.
    ///
    /// @returns Success if the depth/stencil state was successfully created.  Otherwise, one of the following errors
    ///          may be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppDepthStencilState is null.
    ///          + ErrorInvalidValue if:
    ///              - Depth bounds is enabled and the depth range is invalid.
    virtual Result CreateDepthStencilState(
        const DepthStencilStateCreateInfo& createInfo,
        void*                              pPlacementAddr,
        IDepthStencilState**               ppDepthStencilState) const = 0;

    /// Determines the amount of system memory required for a queue semaphore object.  An allocation of this amount of
    /// memory must be provided in the pPlacementAddr parameter of CreateQueueSemaphore().
    ///
    /// @param [in]  createInfo Data controlling the queue semaphore properties, such as an initial semaphore count.
    /// @param [out] pResult    The validation result if pResult is non-null. This argument can be null to avoid the
    ///                         additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an IQueueSemaphore object with the specified properties.
    ///          A return value of 0 indicates the createInfo was invalid.
    virtual size_t GetQueueSemaphoreSize(
        const QueueSemaphoreCreateInfo& createInfo,
        Result*                         pResult) const = 0;

    /// Creates an @ref IQueueSemaphore object with the requested properties.
    ///
    /// @param [in]  createInfo       Data controlling the queue semaphore properties, such as an initial semaphore
    ///                               count.
    /// @param [in]  pPlacementAddr   Pointer to the location where PAL should construct this object.  There must be as
    ///                               much size available here as reported by calling GetQueueSemaphoreSize() with the
    ///                               same createInfo param.
    /// @param [out] ppQueueSemaphore Constructed queue semaphore object.  When successful, the returned address will be
    ///                               the same as specified in pPlacementAddr.
    ///
    /// @returns Success if the queue semaphore was successfully created.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppGpuMemory is null.
    ///          + ErrorInvalidValue if createInfo.initialCount is outside of the [0..31] range.
    virtual Result CreateQueueSemaphore(
        const QueueSemaphoreCreateInfo& createInfo,
        void*                           pPlacementAddr,
        IQueueSemaphore**               ppQueueSemaphore) = 0;

    /// Determines the amount of system memory required for a queue semaphore object created by opening a semaphore
    /// from a different device.  An allocation of this amount of memory must be provided in the pPlacementAddr
    /// parameter of OpenSharedQueueSemaphore().
    ///
    /// @param [in]  openInfo Specifies a handle to a shared queue semaphore object to open.
    /// @param [out] pResult  The validation result if pResult is non-null. This argument can be null to avoid the
    ///                       additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for opening a shared IQueueSemaphore object with the
    ///          specified properties.  A return value of 0 indicates the openInfo was invalid.
    virtual size_t GetSharedQueueSemaphoreSize(
        const QueueSemaphoreOpenInfo& openInfo,
        Result*                       pResult) const = 0;

    /// Opens a shareable queue semaphore object created on another device for use on this device.
    ///
    /// @param [in]  openInfo         Specifies a handle to a queue semaphore memory object to open.
    /// @param [in]  pPlacementAddr   Pointer to the location where PAL should construct this object.  There must be as
    ///                               much size available here as reported by calling GetSharedQueueSemaphoreSize() with
    ///                               the same params.
    /// @param [out] ppQueueSemaphore Constructed queue semaphore object.  When successful, the returned address will be
    ///                               the same as specified in pPlacementAddr.
    ///
    /// @returns Success if the shared semaphore was successfully opened for access on this device.  Otherwise, one of
    ///          the following errors may be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppQueueSemaphore is null.
    ///          + ErrorNotShareable if the specified queue semaphore object was not marked as shareable on creation.
    virtual Result OpenSharedQueueSemaphore(
        const QueueSemaphoreOpenInfo& openInfo,
        void*                         pPlacementAddr,
        IQueueSemaphore**             ppQueueSemaphore) = 0;

    /// Determines the amount of system memory required for a queue semaphore object created by opening a semaphore from
    /// a different API which isn't a PAL client.  An allocation of this amount of memory must be provided in the
    /// pPlacementAddr parameter of OpenExternalSharedQueueSemaphore().
    ///
    /// @param [in]  openInfo Specifies a handle to a shared queue semaphore object to open.
    /// @param [out] pResult  The validation result if pResult is non-null. This argument can be null to avoid
    ///                       the additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for opening a shared IQueueSemaphore object with the
    ///          specified properties.  A return value of 0 indicates the openInfo was invalid.
    virtual size_t GetExternalSharedQueueSemaphoreSize(
        const ExternalQueueSemaphoreOpenInfo& openInfo,
        Result*                               pResult) const = 0;

    /// Opens a shareable queue semaphore object created on another API which isn't a PAL client for use on this device.
    ///
    /// @param [in]  openInfo         Specifies a handle to a queue semaphore memory object to open and flags.
    /// @param [in]  pPlacementAddr   Pointer to the location where PAL should construct this object.  There must be as
    ///                               much size available here as reported by calling GetSharedQueueSemaphoreSize() with
    ///                               the same params.
    /// @param [out] ppQueueSemaphore Constructed queue semaphore object.  When successful, the returned address will be
    ///                               the same as specified in pPlacementAddr.
    ///
    /// @returns Success if the shared semaphore was successfully opened for access on this device.  Otherwise, one of
    ///          the following errors may be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppQueueSemaphore is null.
    virtual Result OpenExternalSharedQueueSemaphore(
        const ExternalQueueSemaphoreOpenInfo& openInfo,
        void*                                 pPlacementAddr,
        IQueueSemaphore**                     ppQueueSemaphore) = 0;

#if PAL_KMT_BUILD
    /// Opens an external NT handle from name.
    ///
    /// @param [in]  handleInfo             Specifies the NT handle attributes when opening from name.
    /// @param [out] pHandle                A NT handle returned by runtime.
    ///
    /// @returns Success if the NT handle was successfully opened.  Otherwise, one of
    ///          the following errors may be returned:
    ///          + ErrorInvalidValue if the name or attributes is invaild.
    virtual Result OpenExternalHandleFromName(
        const ExternalHandleInfo& handleInfo,
        OsExternalHandle*         pHandle) = 0;
#endif

#if (PAL_KMT_BUILD || PAL_AMDGPU_BUILD)
    /// Query current gpu memory usage info and budget info of specified heap group of the device.
    ///
    /// @param [out]  pInfo                  Heap usage and budget info reported from Os.
    virtual Result QueryGpuMemoryBudgetInfo(
        GpuMemoryBudgetInfo* pInfo) = 0;
#endif

    /// Determines the amount of system memory required for an IFence object.  An allocation of this amount of memory
    /// must be provided in the pPlacementAddr parameter of CreateFence().
    ///
    /// @param [out] pResult    The validation result if pResult is non-null. This argument can be null to avoid
    ///                         the additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an IFence object.
    virtual size_t GetFenceSize(
        Result* pResult) const = 0;

    /// Creates a GPU fence object.
    ///
    /// @param [in]  createInfo         Data controlling the fence properties
    /// @param [in]  pPlacementAddr     Pointer to the location where PAL should construct this object.  There must be as
    ///                                 much size available here as reported by calling GetFenceSize().
    /// @param [out] ppFence            Constructed fence object.  When successful, the returned address will be
    ///                                 the same as specified in pPlacementAddr.
    ///
    /// @returns Success if the fence was successfully created.  Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppFence is null.
    virtual Result CreateFence(
        const FenceCreateInfo& createInfo,
        void*                  pPlacementAddr,
        IFence**               ppFence) const = 0;

    /// Opens a fence wihich was shared by another Device.
    ///
    /// @param  [in] openInfo         A reference to FenceOpenInfo, the handle is used if it's not null, or the
    ///                               event is opened via name.
    /// @param [in]  pPlacementAddr   Pointer to the location where PAL should construct this object.  There must be as
    ///                               much size available here as reported by calling GetFenceSize().
    /// @param [out] ppFence          Constructed fence object.  When successful, the returned address will be
    ///                               the same as specified in pPlacementAddr.
    /// @returns Success if the event was successfully reconstructed, otherwise an appropriate error code.
    virtual Result OpenFence(
        const FenceOpenInfo& openInfo,
        void*                pPlacementAddr,
        IFence**             ppFence) const = 0;

    /// Determines the amount of system memory required for an IGpuEvent object.  An allocation of this amount of memory
    /// must be provided in the pPlacementAddr parameter of CreateGpuEvent().
    ///
    /// @param [in]  createInfo Properties of the GPU event object to create.
    /// @param [out] pResult    The validation result if pResult is non-null. This argument can be null to avoid
    ///                         the additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an IGpuEvent object.
    virtual size_t GetGpuEventSize(
        const GpuEventCreateInfo& createInfo,
        Result*                   pResult) const = 0;

    /// Creates a GPU event object.
    ///
    /// @param [in]  createInfo     Properties of the GPU event object to create.
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetEventSize().
    /// @param [out] ppGpuEvent     Constructed event object.  When successful, the returned address will be the same as
    ///                             specified in pPlacementAddr.
    ///
    /// @returns Success if the event was successfully created.  Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppEvent is null.
    virtual Result CreateGpuEvent(
        const GpuEventCreateInfo& createInfo,
        void*                     pPlacementAddr,
        IGpuEvent**               ppGpuEvent) = 0;

    /// Determines the amount of system memory required for a query pool object.  An allocation of this amount of memory
    /// must be provided in the pPlacementAddr parameter of CreateQueryPool().
    ///
    /// @param [in]  createInfo Data controlling the query pool, such as what type of queries and how many slots are in
    ///                         the pool.
    /// @param [out] pResult    The validation result if pResult is non-null. This argument can be null to avoid the
    ///                         additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an IQueryPool object with the specified properties.  A
    ///          return value of 0 indicates the createInfo was invalid.
    virtual size_t GetQueryPoolSize(
        const QueryPoolCreateInfo& createInfo,
        Result*                    pResult) const = 0;

    /// Creates an @ref IQueryPool object with the requested properties.
    ///
    /// @param [in]  createInfo     Data controlling the query pool, such as what type of queries and how many slots are
    ///                             in the pool.
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetQueryPoolSize() with the same
    ///                             createInfo param.
    /// @param [out] ppQueryPool    Constructed query pool object.  When successful, the returned address will be the
    ///                             same as specified in pPlacementAddr.
    ///
    /// @returns Success if the query pool was successfully created.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppQueryPool is null.
    ///          + ErrorInvalidValue if createInfo.numSlots is zero.
    virtual Result CreateQueryPool(
        const QueryPoolCreateInfo& createInfo,
        void*                      pPlacementAddr,
        IQueryPool**               ppQueryPool) const = 0;

    /// Determines the amount of system memory required for a command allocator object.  An allocation of this amount of
    /// memory must be provided in the pPlacementAddr parameter of CreateCmdAllocator().
    ///
    /// @param [in]  createInfo Command allocator properties including GPU memory allocation sizes.
    /// @param [out] pResult    The validation result if pResult is non-null. This argument can be null to avoid the
    ///                         additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an ICmdAllocator object with the specified properties.
    ///          A return value of 0 indicates the createInfo was invalid.
    virtual size_t GetCmdAllocatorSize(
        const CmdAllocatorCreateInfo& createInfo,
        Result*                       pResult) const = 0;

    /// Creates a command allocator object that can allocate GPU memory with the specified properties for use by command
    /// buffer objects.
    ///
    /// @param [in]  createInfo     Command allocator properties including GPU memory allocation sizes.
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetCmdAllocatorSize() with the
    ///                             same createInfo param.
    /// @param [out] ppCmdAllocator Constructed command allocator object.  When successful, the returned address will be
    ///                             the same as specified in pPlacementAddr.
    ///
    /// @returns Success if the command allocator was successfully created.  Otherwise, one of the following errors may
    ///          be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppCmdAllocator is null.
    virtual Result CreateCmdAllocator(
        const CmdAllocatorCreateInfo& createInfo,
        void*                         pPlacementAddr,
        ICmdAllocator**               ppCmdAllocator) = 0;

    /// Determines the amount of system memory required for a command buffer object.  An allocation of this amount of
    /// memory must be provided in the pPlacementAddr parameter of CreateCmdBuffer().
    ///
    /// @param [in]  createInfo Command buffer properties including the target queue type.
    /// @param [out] pResult    The validation result if pResult is non-null. This argument can be null to avoid
    ///                         the additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an ICmdBuffer object with the specified properties.  A
    ///          return value of 0 indicates the createInfo was invalid.
    virtual size_t GetCmdBufferSize(
        const CmdBufferCreateInfo& createInfo,
        Result*                    pResult) const = 0;

    /// Creates a command buffer object that can build work intended for a particular queue type.
    ///
    /// @param [in]  createInfo     Command buffer properties including the target queue type.
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetCmdBufferSize() with the same
    ///                             createInfo param.
    /// @param [out] ppCmdBuffer    Constructed command buffer object.  When successful, the returned address will be
    ///                             the same as specified in pPlacementAddr.
    ///
    /// @returns Success if the command buffer was successfully created.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppCmdBuffer is null.
    virtual Result CreateCmdBuffer(
        const CmdBufferCreateInfo& createInfo,
        void*                      pPlacementAddr,
        ICmdBuffer**               ppCmdBuffer) = 0;

    /// Determines the amount of system memory required for an indirect command generator object.  An allocation of this
    /// amount must be provided in the pPlacementAddr parameter of CreateIndirectCmdGenerator().
    ///
    /// @param [in]  createInfo Indirect command generator properties.
    /// @param [out] pResult    The validation result if pResult is non-null.  This argument can be null to avoid the
    ///                         additional validation steps.
    ///
    /// @returns Size, in bytes, of system memory required for an IIndirectCmdGenerator object with the specified
    ///          properties.  A return value of zero indicates the createInfo was invalid.
    virtual size_t GetIndirectCmdGeneratorSize(
        const IndirectCmdGeneratorCreateInfo& createInfo,
        Result*                               pResult) const = 0;

    /// Creates an indirect command generator object which can translate an application-specified command buffer into a
    /// format understandable by the GPU.
    ///
    /// @param [in]  createInfo
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetIndirectCmdGeneratorSize() with
    ///                             the same createInfo param.
    /// @param [out] ppGenerator    Constructed indirect command generator object.  When successful, the returned
    ///                             address will be the same as specified in pPlacementAddr.
    ///
    /// @returns Success if the command generator was successfully created.  Otherwise, one of the following errors may
    ///          be returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppGenerator is null.
    virtual Result CreateIndirectCmdGenerator(
        const IndirectCmdGeneratorCreateInfo& createInfo,
        void*                                 pPlacementAddr,
        IIndirectCmdGenerator**               ppGenerator) const = 0;

    /// Determines the amount of system memory required for a perf experiment object.  An allocation of this amount of
    /// memory must be provided in the pPlacementAddr parameter of CreatePerfExperiment().
    ///
    /// @param [in]  createInfo Properties of the performance experiment to be created.
    /// @param [out] pResult    The validation result if pResult is non-null. This argument can be null to avoid
    ///                         the additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an IPerfExperiment object with the specified properties.
    ///          A return value of 0 indicates the createInfo was invalid.
    virtual size_t GetPerfExperimentSize(
        const PerfExperimentCreateInfo& createInfo,
        Result*                         pResult) const = 0;

    /// Creates a performance experiment object that can gather performance counter and trace data for a specific span
    /// of a command buffer.
    ///
    /// @param [in]  createInfo       Properties of the performance experiment to be created.
    /// @param [in]  pPlacementAddr   Pointer to the location where PAL should construct this object.  There must be as
    ///                               much size available here as reported by calling GetPerfExperimentSize() with the
    ///                               same createInfo param.
    /// @param [out] ppPerfExperiment Constructed performance experiment object.  When successful, the returned address
    ///                               will be the same as specified in pPlacementAddr.
    ///
    /// @returns Success if the perf experiment was successfully created.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorInvalidPointer if pPlacementAddr or ppCmdBuffer is null.
    virtual Result CreatePerfExperiment(
        const PerfExperimentCreateInfo& createInfo,
        void*                           pPlacementAddr,
        IPerfExperiment**               ppPerfExperiment) const = 0;

    /// Gets @ref IPrivateScreen objects owned by this device. Private screens are screens not exposed through standard
    /// OS mechanisms. This function should be called again when any of the private screens are plugged or unplugged.
    /// The first call to this function enumerates all private screens and stores in device object as well. The next
    /// call triggered by hot-plug event enumerates private screens again but only destroys removed ones and creates
    /// new private screen objects for newly-added ones. The hash code generated at enumeration time is used as id of
    /// private screens. If the id of an enumerated private screen already exists, it is treated as unchanged. The EDID
    /// array and display index are used to generate MD5 hash code.
    ///
    /// @param [out]  pNumScreens  Pointer to the number of private sceens, note that this number does not mean first
    ///                            *pNumScreens elements in ppScreens are valid but just a hint that total *pNumScreens
    ///                            out of MaxPrivateScreens are valid.
    /// @param [out]  ppScreens    Pointer to the array of private screens. The client must pass in the pointer to an
    ///                            array of at least MaxPrivateScreens pointers to IPrivateScreen.
    ///
    /// @returns Success if the private screens are correctly retrieved. Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorUnavailable if the device does not support private screen functionalities.
    virtual Result GetPrivateScreens(
        uint32*          pNumScreens,
        IPrivateScreen** ppScreens) = 0;

    /// Registers an emulated @ref IPrivateScreen objects owned by this device. An emulated private screen doesn't have
    /// a physical display hardware connected to the GPU. This could be useful for debugging unusual configurations or
    /// using in automation systems when no real HMDs are available.
    ///
    /// @param [in]   createInfo   Properties of to create an emulated private screen object.
    /// @param [out]  pTargetId    Pointer to returned emulated private screen target id.
    ///
    /// @returns Success if the emulated private screen is correctly created. Otherwise, one of the following errors may
    ///          be returned:
    ///          + ErrorTooManyPrivateScreens if the device cannot create an emulated private screen.
    virtual Result AddEmulatedPrivateScreen(
        const PrivateScreenCreateInfo& createInfo,
        uint32*                        pTargetId) = 0;

    /// Removes an emulated @ref IPrivateScreen objects owned by this device.
    ///
    /// @param [in]  targetId       Target id of emulated private screen to be removed.
    ///
    /// @returns Success if the emulated private screen is correctly removed. Otherwise, one of the following errors may
    ///          be returned:
    ///          + ErrorUnknown if any unknown error occurs.
    virtual Result RemoveEmulatedPrivateScreen(
        uint32 targetId) = 0;

    /// Determines the amount of system memory required for a private screen image object (and an associated memory
    /// object).  Allocations of these amounts of memory must be provided in the pImagePlacementAddr and
    /// pGpuMemoryPlacementAddr parameters of CreatePrivateScreenImage().
    ///
    /// Only images created through this interface are valid sources for IPrivateScreen::Present().
    ///
    /// @param [in]  createInfo     Properties of the image to create such as width/height and pixel format.
    /// @param [out] pImageSize     Size, in bytes, of system memory required for the IImage.
    ///                             Should be specified to the pImagePlacementAddr argument of CreatePresentableImage().
    /// @param [out] pGpuMemorySize Size, in bytes, of system memory required for a dummy IGpuMemory object attached to
    ///                             the private screen IImage. Should be specified to the pGpuMemoryPlacementAddr
    ///                             argument of CreatePrivateScreenImage().
    /// @param [out] pResult        The validation result if pResult is non-null. This argument can be null to avoid
    ///                             the additional validation.
    virtual void GetPrivateScreenImageSizes(
        const PrivateScreenImageCreateInfo& createInfo,
        size_t*                             pImageSize,
        size_t*                             pGpuMemorySize,
        Result*                             pResult) const = 0;

    /// Creates private screen presentable image. A private screen presentable image is similar to a regular presentable
    /// image but can only be presented on the private screens. It has some implicit properties relative to standard
    /// images, such as mipLevels=1, arraySize=1, numSamples=1 and etc. It also requires its bound GPU memory to be
    /// pinned before presenting.
    ///
    /// @param [in]  createInfo              Create info.
    /// @param [in]  pImagePlacementAddr     Pointer to the location where PAL should construct this object. There must
    ///                                      be as much size available here as reported by calling
    ///                                      GetPrivateScreenImageSizes().
    /// @param [in]  pGpuMemoryPlacementAddr Pointer to the location where PAL should construct a IGpuMemory associated
    ///                                      with this peer image.  There must be as much size available here as
    ///                                      reported by calling GetPrivateScreenImageSizes().
    /// @param [out] ppImage                 Constructed image object.
    /// @param [out] ppGpuMemory             Constructed dummy memory object.  This object is only valid for specifying
    ///                                      in a memory reference list.
    ///
    /// @returns Success if the image was successfully created.  Otherwise, one of the following errors may be returned:
    ///          + ErrorPrivateScreenInvalidFormat if the format isn't supported on the private screen.
    ///          + ErrorPrivateScreenRemoved if the private screen was removed.
    virtual Result CreatePrivateScreenImage(
        const PrivateScreenImageCreateInfo& createInfo,
        void*                               pImagePlacementAddr,
        void*                               pGpuMemoryPlacementAddr,
        IImage**                            ppImage,
        IGpuMemory**                        ppGpuMemory) = 0;

    /// Determines the amount of system memory required for an ISwapChain object. An allocation of this amount of memory
    /// must be provided in the pPlacementAddr parameter of CreateSwapChain().
    ///
    /// @param [in]  createInfo All the information related with this swap chain.
    /// @param [out] pResult    The validation result if pResult is non-null. This argument can be null to avoid
    ///                         the additional validation.
    ///
    /// @returns Size, in bytes, of system memory required for an ISwapChain object.
    virtual size_t GetSwapChainSize(
        const SwapChainCreateInfo& createInfo,
        Result*                    pResult) const = 0;

    /// Create swap chain object based on the local window system. It doesn't include creation of presentable images.
    /// The presentable images should be associated with SwapChain object when presentable image is created.
    ///
    /// @param [in]  createInfo     All the information related with this swap chain.
    /// @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
    ///                             much size available here as reported by calling GetSwapChainSize().
    /// @param [out] ppSwapChain    Constructed swapchain object.  When successful, the returned address will be the
    ///                             same as specified in pPlacementAddr.
    ///
    /// @returns Success if create swap chain instance successfully.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorUnknown if an unexpected internal error occurs.
    virtual Result CreateSwapChain(
        const SwapChainCreateInfo& createInfo,
        void*                      pPlacementAddr,
        ISwapChain**               ppSwapChain) = 0;

    /// Sets a power profile for this device.
    ///
    /// @param [in]      profile A profile is a pre-defined configuration indicates how KMD/PPLib is notified to work,
    ///                  e.g. raise or lower the GPU clock etc.
    /// @param [in,out]  pInfo Custom power profile info needed for VrCustom mode, can be null for other modes. Note the
    ///                  actualSwitchInfo[] field is output part of @ref CustomPowerProfile.
    ///
    /// @returns Success if the profile is set successfully.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorUnavailable if this function is not available on this OS.
    ///          + ErrorUnknown if an unexpected internal error occurs.
    virtual Result SetPowerProfile(
        PowerProfile        profile,
        CustomPowerProfile* pInfo) = 0;

    /// Sends an escape call to the KMD to enable power optimizations for DirectML/ROCm workloads. The client must call
    /// this function when the context for any ROCm or DirectML workload is created or destroyed.
    ///
    /// @param [in] enableOptimization Set to true if called during context creation (enable power optimizations),
    ///                                false if context destroyed (restore defaults).
    ///
    /// @returns Success if the power optimization is set successfully. Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorUnavailable if this function is not available on this OS.
    ///          + ErrorUnknown if an unexpected internal error occurs.
    virtual Result SetMlPowerOptimization(
        bool enableOptimization) const = 0;

    /// Queries workstation caps on this device.
    ///
    /// @param [out]  pCaps Pointer to location where pal should write back workstation caps.
    ///
    /// @returns Success if wokstation caps is got from KMD successfully.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorOutOfMemory if out of system memory.
    virtual Result QueryWorkStationCaps(
        WorkStationCaps* pCaps) const = 0;

    /// Queries display connectors installed on the GPU
    ///
    /// @param [in,out]  pConnectorCount Input value specifies the maximum number of connectors to enumerate, and the
    ///                                  output value specifies the total number of display modes that were enumerated
    ///                                  in pConnectors.  The input value is ignored if pConnectors is null.
    ///                                  This pointer must not be null.
    /// @param [out]     pConnectors     Output list of connectors.  Can be null, in which case the total number of
    ///                                  available connectors will be written to pConnectorCount.
    ///
    ///
    /// @returns Success if the profile is set successfully.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorUnavailable if this function is not available on this OS.
    ///          + ErrorUnknown if an unexpected internal error occurs.
    virtual Result QueryDisplayConnectors(
        uint32*                     pConnectorCount,
        DisplayConnectorProperties* pConnectors) = 0;

    /// Queries the dispatch kernel source code
    ///
    /// @returns Returns the source code for the kernel dispatch function used in
    ///          the kernel scheduler for device enqueue (CL C code) or nullptr if it's not available.
    ///
    /// @note This function is to support OpenCL AQL submissions.
    virtual const char* GetDispatchKernelSource() const = 0;

    /// @}

    /// Query the Flgl state from the device. Device will query the connectivity of GLSync card and return
    /// the state. Pal internal state of Flgl will be updated.
    ///
    /// @param [out]    pState  Pointer to the location that PAL should write the internal flgl states back.
    ///
    /// @returns Success if query returns with success. Otherwise, one of the following errors may returned:
    ///          + ErrorOutOfMemory if out of system memory.
    ///          + ErrorUnknown if an unexpected internal error occurs.
    virtual Result FlglQueryState(
       FlglState* pState) = 0;

    /// Set the Flgl config of the device.
    ///
    /// @param [in]    glSyncConfig  const reference to the config struct.
    ///
    /// @returns Success if setting returns with success. Otherwise, one of the following errors may returned:
    ///          + ErrorUnknown if an unexpected internal error occurs.
    ///          + ErrorUnsuppported if the this GenLock function is not available.
    virtual Result FlglSetSyncConfiguration(
        const GlSyncConfig& glSyncConfig) = 0;

    /// Get the Flgl config of the device.
    /// This function cannot be called if FlglState's support value is FlglSupport::NotAvailable.
    ///
    /// @param [out]    pGlSyncConfig  Pointer to the location that PAL should write the config back.
    ///
    /// @returns Success if query returns with success. Otherwise, one of the following errors may returned:
    ///          + ErrorUnknown if an unexpected internal error occurs.
    ///          + ErrorInvalidPointer if pGlSyncConfig is null poiter.
    virtual Result FlglGetSyncConfiguration(
        GlSyncConfig* pGlSyncConfig) const  = 0;

    /// Set the Framelock to disable or enable. Client should call this interface first to enable/disable Flgl.
    /// This function cannot be called if FlglState's support value is FlglSupport::NotAvailable.
    ///
    /// @param [in]    enable           If true enables KMD framelock, otherwise disables framelock.
    ///
    /// @returns Success if framelock enable/disable successfully. Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorUnavailable if this function is not supported on this Asic.
    ///          + ErrorUnknown if an unexpected internal error occurs.
    virtual Result FlglSetFrameLock(
       bool enable) = 0;

    /// Set the Genlock to disable or enable.
    /// This function cannot be called if FlglState's support value is FlglSupport::NotAvailable.
    ///
    /// @param [in]    enable           If true enables the genlock, otherwise disables genlock.
    ///
    /// @returns Success if genlock enable/disable successfully. Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorUnavailable if this function is not supported on this Asic.
    ///          + ErrorUnknown if an unexpected internal error occurs.
    virtual Result FlglSetGenLock(
        bool enable) = 0;

    /// Reset the framelock HW counter. The following counter operations are directly submit to hardware via I2C
    /// interface Pal doesn't store the counter internally. Client should manage the counter
    ///
    /// @returns Success if the HW counter is reset successfully.  Otherwise, one of the following errors may be
    ///          returned:
    ///          + ErrorUnavailable if this function is not available on this Asic.
    ///          + ErrorUnknown if an unexpected internal error occurs.
    virtual Result FlglResetFrameCounter() const = 0;

    /// Check if a reset/discontinuity of HW framecounter occurs. If returns false, there is no need to query HW frame
    /// counter, client should update its software counter instead. If returns true, client is required to query HW
    /// counter and adjusts its software counter accordingly.
    ///
    /// @param [out]    pReset   Pointer to the location that PAL should write the reset status back.
    ///
    /// @returns Success if the reset status is returned.  Otherwise, one of the following errors may be returned:
    ///          + ErrorUnavailable if this function is not available on this Asic.
    ///          + ErrorUnknown if an unexpected internal error occurs.
    virtual Result FlglGetFrameCounterResetStatus(
        bool* pReset) const = 0;

    /// Get the framelock HW counter.
    ///
    /// @param [out]    pValue   Pointer to the location that PAL should write the frame counter value back.
    /// @param [out]    pReset   Pointer to the location that PAL should write the frame counter reset state.
    ///
    /// @returns Success if the frame counter is returned.  Otherwise, one of the following errors may be returned:
    ///          + ErrorUnavailable if this function is not available on this Asic.
    ///          + ErrorUnknown if an unexpected internal error occurs.
    virtual Result FlglGetFrameCounter(
        uint64* pValue,
        bool*   pReset) const = 0;

    /// Checks if the specified externally-controlled feature settings have changed since the last time the function was
    /// called.
    ///
    /// This is intended to be a lightweight function that can be called per frame per feature.  If the function
    /// returns Result::Success and (*pRsFeaturesChanged & RsFeatureTypeXX) != 0, then the user changed some related
    /// settings in the UI.
    ///
    /// If TurboSync has updated, the client should first try to re-read the application profile settings by calling
    /// IPlatform::QueryRawApplicationProfile() with client = User3D.  If that returns Unsupported, then fall back
    /// to device-wide TurboSync settings read via GetRsFeatureGlobalSettings().
    ///
    /// If Chill has updated, call IPlatform::QueryRawApplicationProfile() with client = Chill to re-read the
    /// system app profiles and then with client = User3D for any per-user Chill overrides, and additionally
    /// call GetRsFeatureGlobalSettings() to get the Chill enabled state.
    ///
    /// If Delag has updated, call IPlatform::QueryRawApplicationProfile() with client = User3D to get the enabled
    /// state, and additionally call GetRsFeatureGlobalSettings() to get the Delag hotkey.
    ///
    ///
    /// @param [in]  rsFeatures          Bitmask of RsFeatureType value(s) to query.  Use UINT_MAX to poll all.
    /// @param [out] pRsFeaturesChanged  Bitmask of queried RsFeatureTypes that have changed since last polling.
    ///
    /// @returns Success if the call succeeded.
    virtual Result DidRsFeatureSettingsChange(
        uint32  rsFeatures,
        uint32* pRsFeaturesChanged) = 0;

    /// Gets externally-controlled per-device settings for the requested RsFeatureType.
    ///
    /// @param [in]  rsFeature       Feature type to request information for (singular, not a mask).
    /// @param [out] pRsFeatureInfo  Settings related to the specified RsFeatureType.
    ///
    /// @returns Success if the call succeeded.
    virtual Result GetRsFeatureGlobalSettings(
        RsFeatureType  rsFeature,
        RsFeatureInfo* pRsFeatureInfo) = 0;

    /// Update Chill Status (last active time stamp). After every frame, UMD needs to generate a time stamp and inform
    /// KMD through the shared memory, if the time stamp changes between 2 frames, it means Chill is active and KMD
    /// needs to adjust power through PSM.
    ///
    /// @param [in]  lastChillActiveTimeStampUs     the last Chill active time stamp in microseconds to set
    ///
    /// @returns Success if the call succeeded.
    virtual Result UpdateChillStatus(
        uint64 lastChillActiveTimeStampUs) = 0;

    /// Make the Bus Addressable allocations available to be accessed by remote device.
    /// Exposes the surface and marker bus addresses for each allocation. These bus addresses can be accessed by
    /// calling @ref IGpuMemory::Desc() on the appropriate object.
    /// Client drivers must call @ref AddGpuMemoryReferences() for all relevant allocations before calling this.
    ///
    /// @param [in]  pQueue         Queue used by PAL for performing this operation.
    /// @param [in]  gpuMemCount    Number of GPU memory allocations to expose to remote devices.
    /// @param [in]  ppGpuMemList   Array of gpuMemCount IGpuMemory objects.
    ///
    /// @returns Success if bus addresses are available by calling @ref IGpuMemory::Desc() on all IGpuMemory objects
    virtual Result InitBusAddressableGpuMemory(
        IQueue*           pQueue,
        uint32            gpuMemCount,
        IGpuMemory*const* ppGpuMemList) = 0;

    /// Create virtual display. Virtual display is similar to the regular display (IScreen), the difference is the
    /// virtual display doesn't have a physical monitor connected. When CreateVirtualDisplay is called, KMD will
    /// generate a hot-plug-in event to notify application a new display is added. Then the app/client will call PAL to
    /// re-querythe attached screens and they will find a new one in the list that is pretend, but they can use it just
    /// like a normal display.
    ///
    /// @param [in]  virtualDisplayInfo   Virtual display creation infomation.
    /// @param [out] pScreenTargetId      The screen target ID returned by KMD
    ///
    /// @returns Success if the call succeeded.
    virtual Result CreateVirtualDisplay(
        const VirtualDisplayInfo& virtualDisplayInfo,
        uint32*                   pScreenTargetId) = 0;

    /// Destroy virtual display. When DestroyVirtualDisplay is called KMD will generate a hot-plug-out event to notify
    /// application a virtual display is removed, it also will be removed from the display list, and app/client can't
    /// uses it anymore.
    ///
    /// @param [in]  screenTargetId  Screen target ID.
    ///
    /// @returns Success if the call succeeded.
    virtual Result DestroyVirtualDisplay(
        uint32     screenTargetId) = 0;

    /// Query virtual display Properties from screen target Id.
    ///
    /// @param [in]  screenTargetId             Screen target ID.
    /// @param [out] pVirtualDisplayProperties  A pointer to VirtualDisplayProperties
    ///
    /// @returns Success if the call succeeded.
    virtual Result GetVirtualDisplayProperties(
        uint32                    screenTargetId,
        VirtualDisplayProperties* pProperties) = 0;

    /// Determines if hardware accelerated stereo rendering can be enabled for given graphic pipeline.
    /// If hardware accelerate stereo rendering can be enabled, client doesn't need to do shader patching
    /// which includes translating view id intrinsic to user data slot, outputing render target
    /// array index and viewport array index in shader closest to scan converter.
    ///
    /// @param [in]  viewInstancingInfo  Graphic pipeline view instancing information.
    ///
    /// @returns True if hardware accelerated stereo rendering can be enabled, False otherwise.
    virtual bool DetermineHwStereoRenderingSupported(
        const GraphicPipelineViewInstancingInfo& viewInstancingInfo) const = 0;

    /// Get the UTF-8 file path used to put all files for cache purpose.
    /// On Windows, the returned path depends on Util::GetProcessIntegrityLevel and Util::IsProcessInAppContainer.
    ///
    /// @returns Pointer to cache file path.
    virtual const char* GetCacheFilePath() const = 0;

    /// Get the UTF-8 file path used to put all files for debug purpose (such as logs, dumps, replace shader).
    ///
    /// @returns Pointer to debug file path.
    virtual const char* GetDebugFilePath() const = 0;

    /// Queries the base driver Radeon Software Version string (as shown in Radeon Settings).
    ///
    /// @param [out]  pBuffer           A non-null pointer to the buffer where the string will be written.
    /// @param [in]   bufferLength      The byte size of the string buffer (must be non-zero).
    ///
    /// @returns Success if the string was successfully retrieved. Otherwise, one of the following errors
    ///          may be returned:
    ///          + Unsupported if this function is not available on this environment.
    ///          + NotFound if the Radeon Software Version string is not present.
    ///          + ErrorInvalidValue if nullptr was passed for pBuffer or 0 for bufferLength.
    virtual Result QueryRadeonSoftwareVersion(
        char*  pBuffer,
        size_t bufferLength) const = 0;

    /// Queries the base Driver Release Version string.
    ///
    /// @param [out]  pBuffer           A non-null pointer to the buffer where the string will be written.
    /// @param [in]   bufferLength      The byte size of the string buffer (must be non-zero).
    ///
    /// @returns Success if the string was successfully retrieved. Otherwise, one of the following errors
    ///          may be returned:
    ///          + Unsupported if this function is not available on this environment.
    ///          + NotFound if the Release Version string is not present.
    ///          + ErrorInvalidValue if nullptr was passed for pBuffer or 0 for bufferLength.
    virtual Result QueryReleaseVersion(
        char* pBuffer,
        size_t bufferLength) const = 0;

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

#if defined(__unix__)
    /// Gets the modifier list and modifier count according to image format.
    ///
    /// @param [in]    format             Image format.
    /// @param [out]   pModifierCount     Modifier count.
    /// @param [out]   pModifiersList     A pointer to the modifier list.
    virtual void GetModifiersList(
        ChNumFormat format,
        uint32*     pModifierCount,
        uint64*     pModifiersList) const = 0;
#endif

    /// Passes the HIP runtime state setup to KMD on behalf of the HIP runtime.
    ///
    /// @param [in] runtimeState  A structure containing runtime state information to pass to KMD
    ///
    /// @returns Result for error handling.
    virtual Result RegisterHipRuntimeState(const HipRuntimeSetup& runtimeState) const = 0;

    /// Sets the second-level trap handler for HIP
    ///
    /// @param [in] pTrapHandlerCode   A pointer to the piece of memory containing the trap handler code
    ///                                This may be nullptr, which indicates that there is no secondary trap handler.
    /// @param [in] codeOffset         An offset, in bytes, into the pTrapHandlerCode's memory region
    /// @param [in] pTrapHandlerMemory A pointer to the piece of memory containing the trap handler's memory
    ///                                This may be nullptr, which indicates that there is no valid trap handler
    ///                                memory.
    /// @param [in] memoryOffset       An offset, in bytes, into the pTrapHandlerMemory's memory region
    ///
    /// @returns Result for error handling.
    virtual Result SetHipTrapHandler(
        const IGpuMemory* pTrapHandlerCode,
        gpusize           codeOffset,
        const IGpuMemory* pTrapHandlerMemory,
        gpusize           memoryOffset) const = 0;

    /// Sets the CmdDisassembly::ICmdBufferReporting associated with this device
    /// Currently only one can be registered at a time - but this could be changed in the future
    ///
    /// @param [in] pInterface              A pointer to the CmdDisassembly::ICmdBufferReporting being
    ///                                         registered
    /// @return Result::Success             if successful
    ///         Result::Unsupported         if feature is not supported
    ///         Result::AlreadyExists       if there is already a CmdDisassembly::ICmdBufferReporting
    ///                                         registered with this devide
    ///         Result::ErrorInvalidValue   if pInterface == nullptr
    ///
    virtual Result RegisterCmdReportingInterface(
        CmdDisassembly::ICmdBufferReporting* pInterface)
    {
        return Result::Unsupported;
    }

    /// Resets the CmdDisassembly::ICmdBufferReporting associated with this device to nullptr
    /// Currently only one can be registered at a time - but this could be changed in the future
    ///
    /// @param [in] pInterface              A pointer to the CmdDisassembly::ICmdBufferReporting that
    ///                                         was registered
    /// @return Result::Success             if successful
    ///         Result::Unsupported         if feature is not supported
    ///         Result::ErrorInvalidValue   if pInterface == nullptr or pInterface was not the last
    ///                                         CmdDisassembly::ICmdBufferReporting registered
    ///
    virtual Result UnregisterCmdReportingInterface(
        CmdDisassembly::ICmdBufferReporting* pInterface)
    {
        return Result::Unsupported;
    }

protected:
    /// @internal Constructor. Prevent use of new operator on this interface. Client must create objects by explicitly
    /// called the proper create method.
    IDevice() : m_pClientData(nullptr) {}

    /// @internal Destructor.  Prevent use of delete operator on this interface.  Device objects will be destroyed when
    ///           @ref IPlatform::Destroy() is called.
    virtual ~IDevice() { }

    DeviceInterfacePfnTable m_pfnTable; ///< SRD function pointer table.

private:
    /// @internal Client data pointer. This can have an arbitrary value and can be returned by calling GetClientData()
    /// and set via SetClientData().
    /// For non-top-layer objects, this will point to the layer above the current object.
    void* m_pClientData;
};

/**
 ***********************************************************************************************************************
 * @defgroup ResourceBinding Resource Binding Model
 *
 * _Resource binding_ refers to the process of binding resources (textures, UAVs, samplers, etc.) for access by shaders
 * in a pipeline.  This is an area where 3D APIs diverge significantly.  PAL's model is designed to minimally abstract
 * the underlying hardware in a way that allows performant implementations by each client driver.
 *
 * ### Hardware User Data
 * GCN hardware has 16 _user data_ registers that act as a generic interface for passing values from a command buffer to
 * a shader.  User data registers are set to their desired value via packets in a command buffer, then the specified
 * values are loaded from the user data registers into shader GPRs when a wave is launched.  Since the user data is just
 * arbitrary generic data, this method can be used to pass any type of data a client may want to specify directly from
 * a command buffer, for example:
 *
 * - __Constant value__ - a 32-bit floating point or integer constant could be written into user data then be used
 *   directly by the shader.
 * - __Shader resource descriptor (SRD)__ - 4 or 8 dwords of consecutive data could be an SRD which will be used as a
 *   t#, s#, etc. by the shader.
 * - __Pointer__ - The user data could be an arbitrary GPU virtual address where a table of constants, SRDs, etc. are
 *   stored.
 *
 * The 3D driver and shader compiler are responsible for working together to define how resources referenced in a shader
 * should be mapped to user data bound in a command buffer.
 *
 * ### PAL User Data
 * PAL only lightly abstracts the hardware user data concept.  DeviceProperties reports the number of user data entries
 * supported on the device in maxUserDataEntries.  Note that some clients may require more user data entries than there
 * are physical user data registers - PAL will manage "spilling" of user data entries to GPU memory if necessary.
 *
 * User data entries are set in a command buffer by calling ICmdBuffer::CmdSetUserData().
 *
 * ### Shader User Data Mapping
 * When creating a pipeline, the client must specify how the user data entries set in a command buffer map to resources
 * referenced by each shader in the pipeline.  This is done in the pUserDataNodes array of PipelineShaderInfo.
 *
 * The resource mapping is built as a graph of _resource mapping nodes_ where the root nodes in the graph correspond
 * to the user data entries.  Each node fits in one of the following categories:
 *
 * - __SRD__: A 4 or 8 dword descriptor describing a shader resource.  The mapping specifies the type and slot the SRD
 *   corresponds to (e.g., UAV 3 or sampler 7).
 * - __Descriptor table pointer__: A GPU virtual address pointing at an array of other nodes.  Typically this will be a
 *   pointer to GPU memory containing just SRDs, but tables are free to be built hierarchically such that tables have
 *   pointers to other tables in them.
 * - __Inline constants__: 32-bit constants loaded directly byu the shader.  The mapping specified the CB slot that
 *   should load the constant (e.g., cb3[1]).
 * - __Unused__: A particular shader may not use all entries in a user data layout, and those should be marked unused.
 *
 * The following image illustrates a simple user data mapping:
 *
 * @image html userDataMapping.png
 *
 * ### Building Descriptor Tables
 * The client is responsible for building specifying SRDs and pointers to GPU memory in order to execute the shader
 * resource mapping specified during pipeline creation.  SRDs can be created with several methods provided by IDevice:
 *
 * - CreateTypedBufferViewSrds()
 * - CreateUntypedBufferViewSrds()
 * - CreateImageViewSrds()
 * - CreateFmaskViewSrds()
 * - CreateSamplerSrds()
 *
 * The size required for each of these SRD types is returned in the srdSizes structure in DeviceProperties.
 *
 * When building descriptor tables in GPU memory, the client will need to retrieve a virtual address of the GPU memory
 * where the tables exist in order to reference them from user data or from other descriptor tables.  IGpuMemory
 * provides the GetVirtAddr() method for this purpose.
 ***********************************************************************************************************************
 */

} // Pal
