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
 * @file  palPerfExperiment.h
 * @brief Defines the Platform Abstraction Library (PAL) IPerfExperiment interface and related types.
 ***********************************************************************************************************************
 */

#pragma once

#include "pal.h"
#include "palGpuMemoryBindable.h"

namespace Pal
{

/// Specifies a particular block on the GPU to gather counters for.
enum class GpuBlock : uint32
{
    Cpf      = 0x0,
    Ia       = 0x1,
    Vgt      = 0x2,
    Pa       = 0x3,
    Sc       = 0x4,
    Spi      = 0x5,
    Sq       = 0x6,
    Sx       = 0x7,
    Ta       = 0x8,
    Td       = 0x9,
    Tcp      = 0xA,
    Tcc      = 0xB,
    Tca      = 0xC,
    Db       = 0xD,
    Cb       = 0xE,
    Gds      = 0xF,
    Srbm     = 0x10,
    Grbm     = 0x11,
    GrbmSe   = 0x12,
    Rlc      = 0x13,
    Dma      = 0x14,
    Mc       = 0x15,
    Cpg      = 0x16,
    Cpc      = 0x17,
    Wd       = 0x18,
    Tcs      = 0x19,
    Atc      = 0x1A,
    AtcL2    = 0x1B,
    McVmL2   = 0x1C,
    Ea       = 0x1D,
    Rpb      = 0x1E,
    Rmi      = 0x1F,
    Umcch    = 0x20,
    Ge       = 0x21,
    Gl1a     = 0x22,
    Gl1c     = 0x23,
    Gl1cg    = 0x24,
    Gl2a     = 0x25, // TCA is used in Gfx9, and changed to GL2A in Gfx10
    Gl2c     = 0x26, // TCC is used in Gfx9, and changed to GL2C in Gfx10
    Cha      = 0x27,
    Chc      = 0x28,
    Chcg     = 0x29,
    Gus      = 0x2A,
    Gcr      = 0x2B,
    Ph       = 0x2C,
    UtcL1    = 0x2D,
    Ge1      = Ge,
    GeDist   = 0x2E,
    GeSe     = 0x2F,
    DfMall   = 0x30, // The DF subblocks have unique instances and event IDs but they all share the DF's perf counters.
    SqWgp    = 0x31, // SQ counters that can be sampled at WGP granularity.
    Pc       = 0x32,
    Gl1xa    = 0x33,
    Gl1xc    = 0x34,
    Wgs      = 0x35,
    EaCpwd   = 0x36,
    EaSe     = 0x37,
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 926
    RlcUser  = 0x38,
#else
    RlcLocal = 0x38,
#endif
    Count
};

/// Distinguishes between global and streaming performance monitor (SPM) counters.
enum class PerfCounterType : uint32
{
    Global = 0x0, ///< Represents the traditional summary perf counters.
    Spm    = 0x1, ///< Represents streaming performance counters.
    Spm32  = 0x2, ///< Represents 32bit streaming performance counters
    Count
};

/// Reports the type of data the hardware writes for a particular counter.
enum class PerfCounterDataType : uint32
{
    Uint32 = 0x0,
    Uint64 = 0x1,
    Count
};

/// Distinguishes between normal thread traces and streaming performance monitor (SPM) traces.
enum class PerfTraceType : uint32
{
    ThreadTrace = 0x0,
    SpmTrace    = 0x1,
    Count
};

/// Mask values ORed together to choose which shader stages a performance experiment should sample.
enum PerfExperimentShaderFlags
{
    PerfShaderMaskPs  = 0x01,
    PerfShaderMaskVs  = 0x02,
    PerfShaderMaskGs  = 0x04,
    PerfShaderMaskEs  = 0x08,
    PerfShaderMaskHs  = 0x10,
    PerfShaderMaskLs  = 0x20,
    PerfShaderMaskCs  = 0x40,
    PerfShaderMaskAll = 0x7f,
};

/// Selects one of generic performance trace markers, which the client can use to track data of its own choosing.
enum class PerfTraceMarkerType : uint32
{
    SqttA = 0x0,
    SqttB = 0x1,
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 874
    A = SqttA,
    B = SqttB,
#endif
    SpmA  = 0x2,
    SpmB  = 0x3,
    SpmC  = 0x4,
    SpmD  = 0x5,
    Count
};

/// Specifies available features in device for supporting performance measurements.
union PerfExperimentDeviceFeatureFlags
{
    struct
    {
        uint32 counters          :  1; ///< Device supports performance counters.
        uint32 threadTrace       :  1; ///< Device supports thread traces.
        uint32 spmTrace          :  1; ///< Device supports streaming perf monitor traces.
        uint32 dfSpmTrace        :  1; ///< Device supports streaming df perf monitor traces.
        uint32 supportPs1Events  :  1; ///< The thread trace HW of this Device is capable of producing event tokens
                                       ///  from the second PS backend of SC.
        uint32 sqttBadScPackerId :  1; ///< Hardware is affected by bug causing the packer ID specified in new PS waves
                                       ///  to be incorrect in SQ thread trace data.
        uint32 reserved          : 26; ///< Reserved for future use.
    };
    uint32     u32All;                 ///< Feature flags packed as 32-bit uint.
};

/// Specifies properties for a perf counter being added to a perf experiment.  Input structure to
/// IPerfExperiment::AddCounter().
///
/// A note for GpuBlock::SqWgp
/// Client of palPerfExperiment may configure counters of GpuBlock::SqWgp based on a per-wgp granularity
/// only if the following are disabled: GFXOFF, virtualization/SRIOV, VDDGFX (power down features), clock gating (CGCG)
/// and power gating. PAL expose this feature to clients.
/// If any of the conditions above cannot be met, it's the client's job to set all WGPs in the same SE to the same
/// perf counter programming. In this case, GpuBlock::SqWgp's perf counter works on a per-SE granularity.
/// Strictly speaking, it's not true that the counters work on a per-SE granularity when those power features
/// are enabled. It's all still per-WGP in HW, we just can't support different counter configs within the same SE.
/// The counter data is still reported per WGP (not aggregated for the whole SE).
///
struct PerfCounterInfo
{
    PerfCounterType              counterType; ///< Type of counter to add.
    GpuBlock                     block;       ///< Which block to reference.
    uint32                       instance;    ///< Instance of that block in the device.
    uint32                       eventId;     ///< Which event ID to track.

    // Some blocks have additional per-counter controls. They must be properly programmed when adding counters for
    // the relevant blocks. It's recommended to zero them out when not in use.
    union
    {
        struct
        {
            uint32 eventQualifier;   ///< The DF counters have an event-specific qualifier bitfield.
        } df;

        struct
        {
            uint16 eventThreshold;   ///< Threshold value for those UMC counters having event-specific threshold.
            uint8  eventThresholdEn; ///< Threshold enable (0 for disabled,1 for <threshold,2 for >threshold).
            uint8  rdWrMask;         ///< Read/Write mask select (1 for Read, 2 for Write).
        } umc;

        uint32 rs64Cntl; ///< CP blocks CPG and CPC have events that can be further filtered for processor events

        uint32 u32All; ///< Union value for copying, must be increased in size if any element of the union exceeds
    } subConfig;
};

/// Specifies properties for setting up a streaming performance counter trace. Input structure to
/// IPerfExperiment::AddSpmTrace().
struct SpmTraceCreateInfo
{
    uint32                 spmInterval;       ///< Interval between each sample in terms of GPU sclks. Minimum of 32.
    gpusize                ringSize;          ///< Suggested size of the SPM output ring buffer in bytes. PAL may use
                                              ///  a smaller ring in practice but it cannot exceed this size.
    uint32                 numPerfCounters;   ///< Number of performance counters to be collected in this trace.
    const PerfCounterInfo* pPerfCounterInfos; ///< Array of size numPerfCounters of PerfCounterInfo(s).
};

/// Reports layout of a single global perf counter sample.
struct GlobalSampleLayout
{
    GpuBlock            block;             ///< Type of GPU block.
    uint32              instance;          ///< Which instance of that type of GPU block.
    uint32              slot;              ///< Slot varies in meaning per block.
    uint32              eventId;           ///< Sampled event ID.
    PerfCounterDataType dataType;          ///< What type of data is written (e.g., 32-bit uint).
    gpusize             beginValueOffset;  ///< Offset in bytes where the sample data begins.
    gpusize             endValueOffset;    ///< Offset in bytes where the sample data ends.
};

/// Describes the layout of global perf counter data in memory.
struct GlobalCounterLayout
{
    uint32             sampleCount;  ///< Number of samples described in samples[].
    GlobalSampleLayout samples[1];   ///< Describes the layout of each sample.  This structure is repeated (sampleCount
                                     ///  - 1) additional times.
};

/// Enumeration of SQ Thread trace token types. All versions of Thread Trace (TT) are represented. If an unsupported
/// token is enabled, no error is reported.
enum ThreadTraceTokenTypeFlags : Pal::uint32
{
    Misc         = 0x00000001, ///< A miscellaneous event has been sent. TT 2.3
    Timestamp    = 0x00000002, ///< Timestamp tokens. TT 2.3
    Reg          = 0x00000004, ///< Register activity token. TT 2.3
    WaveStart    = 0x00000008, ///< A wavefront has started. TT 2.3
    WaveAlloc    = 0x00000010, ///< Output space has been allocated for vertex position or color/Z. TT 2.3.
    RegCsPriv    = 0x00000020, ///< There has been a compute pipeline private data, state or threadgroup update. TT 2.3.
    WaveEnd      = 0x00000040, ///< Wavefront completion. TT 2.3
    Event        = 0x00000080, ///< An event has reached the top of a shader stage. TT 2.3
    EventCs      = 0x00000100, ///< An event has reached the top of a compute shader stage. TT 2.3
    EventGfx1    = 0x00000200, ///< An event has reached the top of a shader stage for the second GFX pipe. TT 2.3
    Inst         = 0x00000400, ///< The shader has executed an instruction. TT 2.3
    InstPc       = 0x00000800, ///< The shader has explicitly written the PC value. TT 2.3
    InstUserData = 0x00001000, ///< The shader has written user data into the thread trace buffer. TT 2.3
    Issue        = 0x00002000, ///< Provides information about instruction scheduling. TT 2.3
    Perf         = 0x00004000, ///< The performance counter delta has been updated. TT 2.3 and below only.
    RegCs        = 0x00008000, ///< A compute  state update packet has been received by the SPI. TT 2.3
    VmemExec     = 0x00010000, ///< A previously issued VMEM instruction is now being sent to LDS/TA. TT 3.0
    AluExec      = 0x00020000, ///< A previously issued VALU instruction is now being executed. TT 3.0
    ValuInst     = 0x00040000, ///< A VALU instruction has been issued. TT 3.0.
    WaveRdy      = 0x00080000, ///< Mask of which waves became ready this cycle but did not issue an instruction. TT 3.0
    Immed1       = 0x00100000, ///< One wave issued an immediate instruction this cycle. TT 3.0.
    Immediate    = 0x00200000, ///< One or more waves have issued an immediate instruction this cycle. TT 3.0.
    UtilCounter  = 0x00400000, ///< A new set of utilization counter values. TT 3.0.
    RealTime     = 0x00800000, ///< Output realtime. TT 3.3.
    All          = 0xFFFFFFFF  ///< Enable all the above tokens.
};

/// Enumeration of register types whose reads/writes can be traced. Register reads are disabled by default as it can
/// generate a lot of traffic and cause the GPU to hang.
enum ThreadTraceRegTypeFlags : Pal::uint32
{
    EventRegs             = 0x00000001, ///< Event registers. TT 2.3.
    DrawRegs              = 0x00000002, ///< Draw registers. TT 2.3.
    DispatchRegs          = 0x00000004, ///< Dispatch registers. TT 2.3.
    UserdataRegs          = 0x00000008, ///< UserData Registers. Must be explicitly requested in TT 2.3.
    MarkerRegs            = 0x00000010, ///< Thread trace marker data regs. TT 2.3.
    ShaderConfigRegs      = 0x00000020, ///< Shader configuration state. TT 3.0.
    ShaderLaunchStateRegs = 0x00000040, ///< Shader program launch state. TT 3.0.
    GraphicsPipeStateRegs = 0x00000080, ///< Graphics pipeline state. TT 3.0.
    AsyncComputeRegs      = 0x00000100, ///< Async compute registers. TT 3.0.
    GraphicsContextRegs   = 0x00000200, ///< Graphics context registers. TT 3.0.
    OtherConfigRegs       = 0x00000400, ///< Other regs. TT 2.3.
    AllRegWrites          = 0x000007FF, ///< All reg writes other than OtherBusRegs.
    OtherBusRegs          = 0x00000800, ///< All write activity over gfx and compute buses. Debug only. TT 3.0.
    AllRegReads           = 0x00001000, ///< Not encouraged to be enabled. This can cause a GPU hang.
    AllReadsAndWrites     = 0xFFFFFFFF  ///< All reads and writes. Not encouraged. This can cause a GPU hang.
};

/// Represents thread trace token types and register types that can be enabled to be reported in the trace data. If
/// a particular token type or reg type is unsupported, no error is returned and the thread trace is configured with
/// the minimum supported tokens in the user provided config.
struct ThreadTraceTokenConfig
{
    /// Mask of ThreadTraceTokenTypeFlags
    uint32 tokenMask;

    /// Mask of ThreadTraceRegTypeFlags
    uint32 regMask;
};

/// Specifies properties for a perf trace being added to a perf experiment.  Input structure to
/// IPerfExperiment::AddThreadTrace().
struct ThreadTraceInfo
{
    PerfTraceType              traceType;    ///< Type of trace to add.
    uint32                     instance;     ///< Selected trace instance.

    union
    {
        struct
        {
            // Options common to all traces
            uint32 bufferSize                            :  1;

            // Thread trace only options
            uint32 threadTraceTargetSh                   :  1;
            uint32 threadTraceTargetCu                   :  1;
            uint32 threadTraceSh0CounterMask             :  1;
            uint32 threadTraceSh1CounterMask             :  1;
            uint32 threadTraceSimdMask                   :  1;
            uint32 threadTraceVmIdMask                   :  1;
            uint32 threadTraceRandomSeed                 :  1;
            uint32 threadTraceShaderTypeMask             :  1;
            uint32 threadTraceIssueMask                  :  1;
            uint32 threadTraceWrapBuffer                 :  1;
            uint32 threadTraceStallBehavior              :  1;
            uint32 threadTraceTokenConfig                :  1;
            uint32 threadTraceStallAllSimds              :  1;
            uint32 threadTraceExcludeNonDetailShaderData :  1;
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 899
            uint32 threadTraceEnableExecPop              :  1;
#else
            uint32 placeholder2                          :  1;
#endif
            uint32 reserved                              : 16;
        };
        uint32 u32All;
    } optionFlags;

    struct
    {
        // Options common to all traces
        size_t                    bufferSize;

        // Thread trace only options
        ThreadTraceTokenConfig    threadTraceTokenConfig;
        uint32                    threadTraceTargetSh;
        uint32                    threadTraceTargetCu;
        uint32                    threadTraceSh0CounterMask;
        uint32                    threadTraceSh1CounterMask;
        uint32                    threadTraceSimdMask;
        uint32                    threadTraceVmIdMask;
        uint32                    threadTraceRandomSeed;
        PerfExperimentShaderFlags threadTraceShaderTypeMask;
        uint32                    threadTraceIssueMask;
        bool                      threadTraceWrapBuffer;
        uint32                    threadTraceStallBehavior;
        bool                      threadTraceStallAllSimds;
        bool                      threadTraceExcludeNonDetailShaderData;
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 899
        bool                      threadTraceEnableExecPop;
#endif
    } optionValues;
};

/// Reports thread trace data written when the trace is stopped (copied from internal SQ registers).
struct ThreadTraceInfoData
{
    uint32 curOffset;     ///< Contents of SQ_THREAD_TRACE_WPTR register.
    uint32 traceStatus;   ///< Contents of SQ_THREAD_TRACE_STATUS register.
    uint32 writeCounter;  ///< Contents of SQ_THREAD_TRACE_CNTR register.
};

/// Describes the layout of a single shader engine's thread trace data.
struct ThreadTraceSeLayout
{
    uint32  shaderEngine;  ///< Shader engine index.
    uint32  computeUnit;   ///< Compute unit index.
    gpusize infoOffset;    ///< Offset to ThreadTraceInfoData in memory.
    gpusize infoSize;      ///< Size in bytes reserved for ThreadTraceInfoData.
    gpusize dataOffset;    ///< Offset in bytes to the actual trace data.
    gpusize dataSize;      ///< Amount of trace data, in bytes.
};

/// Describes how the thread trace data is laid out.
struct ThreadTraceLayout
{
    uint32              traceCount;  ///< Number of entries in traces[].
    ThreadTraceSeLayout traces[1];   ///< ThreadTraceSeLayout repeated (traceCount - 1) times.
};

/// Describes a single SPM counter instance.
struct SpmCounterData
{
    GpuBlock gpuBlock; ///< The kind of GPU block this counter measured.
    uint32   instance; ///< Which specific global block instance this counter measured.
    uint32   eventId;  ///< The event that was measured by this counter.
    uint32   offsetLo; ///< Byte offset within each sample to the lower 16-bit half of the counter data.
    uint32   offsetHi; ///< Byte offset within each sample to the upper 16-bit half of the counter data.
    bool     is32Bit;  ///< If the client must combine the independent 16-bit halves into a single 32-bit value.
                       ///  If this is false offsetLo points to the full 16-bit data value and offsetHi is ignored.
};

/// All information required to parse the counter data out of a SpmTrace results buffer.
///
/// Note that the hardware will continue to write samples to the SPM ring buffer even if it runs out of unused space.
/// The hardware will simply wrap the ring's write pointer back around to the first sample's location. Each subsequent
/// sample will overwrite the oldest sample in the ring. When the trace is finished we will have at most @ref
/// maxNumSamples valid samples.
///
/// PAL doesn't zero out the ring memory so it's generally hard for the client to distinguish valid samples from random
/// data present in unused sample locations. PAL does guarantee that the final sample location in the ring has its
/// timestamp zeroed out before the SPM trace starts. This means this last timestamp will only be non-zero if the ring
/// has completely filled up and the WrPtr has wrapped one or more times. The client must inspect this timestamp when
/// parsing the sample data:
/// 1. The last timestamp is zero. The ring did not wrap. The oldest sample is at @ref sampleOffset. The ring's write
///    pointer tells us how many samples were written. From the write pointer onwards the ring contains invalid data.
/// 2. The last timestamp is non-zero. The ring did wrap. The ring's write pointer points to the oldest sample,
///    effectively a random sample offset into the ring. The full ring contains valid sample data but it's not in
///    oldest-to-newest order, it's shifted. The client can walk the ring from the write pointer's location (wrapping
///    as they go) to parse all @ref maxNumSamples samples out in oldest-to-newest order.
struct SpmTraceLayout
{
    gpusize offset;           ///< Byte offset into the bound GPU memory where the spm trace data begins.
                              ///  The @ref wrPtrOffset and @ref sampleOffset are relative to this value.
    uint32  wrPtrOffset;      ///< Byte offset within SPM trace data to the HW's write pointer (WrPtr) DWORD.
                              ///  The WrPtr's value is an offset relative to @ref sampleOffset. Don't assume this is
                              ///  a byte offset (see @ref wrPtrGranularity). The WrPtr's value shows where the HW's
                              ///  theoretical next sample would go. This value may wrap back to zero if the HW runs of
                              ///  space in the SPM ring buffer.
    uint32  wrPtrGranularity; ///< The WrPtr's granularity. Multiply WrPtr's value by this value to get a byte offset.
    uint32  sampleOffset;     ///< Byte offset within the SPM trace data to the array of samples. The HW will write the
                              ///  first sample here but it will be overwritten if the ring wraps (see the top comment).
    uint32  sampleStride;     ///< The distance between consecutive samples in bytes. May include empty padding.
    uint32  maxNumSamples;    ///< The maximum number of samples the HW can write before wrapping. The SPM ring buffer
                              ///  ends at sampleOffset + sampleStride * maxNumSamples.
    uint32  numCounters;      ///< The true length of counterData. The client must allocate extra memory for the array.

    SpmCounterData counterData[1]; ///< The layout and identity of the counters in the samples.
};

/// Represents the information that is stored in the DF SPM trace metadata buffer.
struct DfSpmTraceMetadataLayout
{
    uint32 numRecordPairs; ///< The number of 64-byte blocks written by this trace. There are two time segments
                           ///< per 64-byte block so we have to check the lastSpmPkt bit to see which half of
                           ///< the last 64-byte block is the last packet.
    uint32 padding;        ///< Padding to match what the compiler does by default.
    uint64 beginTimestamp; ///< The DF timestamp at the start of the DF SPM trace.
    uint64 endTimestamp;   ///< The DF timestamp at the finish of the DF SPM trace.
};

/// Specifies properties for creation of an @ref IPerfExperiment object.  Input structure to
/// IDevice::CreatePerfExperiment().
struct PerfExperimentCreateInfo
{
    union
    {
        struct
        {
            uint32 cacheFlushOnCounterCollection :  1;
            uint32 sampleInternalOperations      :  1;
            uint32 sqShaderMask                  :  1;
            uint32 sqWgpShaderMask               :  1;
            uint32 reserved                      : 28;
        };
        uint32 u32All;
    } optionFlags;

    struct
    {
        bool                      cacheFlushOnCounterCollection;
        bool                      sampleInternalOperations;
        PerfExperimentShaderFlags sqShaderMask;    ///< GpuBlock::Sq counters only look at these shader types.
        PerfExperimentShaderFlags sqWgpShaderMask; ///< GpuBlock::SqWgp counters only look at these shader types.
    } optionValues;
};

/**
 ***********************************************************************************************************************
 * @interface IPerfExperiment
 * @brief     Set of performance profiling activities to be performed over a specific range of commands in a command
 *            buffer.
 *
 * @warning The details of building a performance experiment are not very well documented here.  Please see your local
 *          hardware performance expert for more details until this documentation can be fully fleshed out.
 *
 * @see IDevice::CreatePerfExperiment
 ***********************************************************************************************************************
 */
class IPerfExperiment : public IGpuMemoryBindable
{
public:
    /// Adds the specified performance counter to be tracked as part of this perf experiment.
    ///
    /// @param [in] counterInfo Specifies which counter to add: which hardware block, instance, any options, etc.
    ///
    /// @returns Success if the counter was successfully added to the experiment, otherwise an appropriate error code.
    virtual Result AddCounter(
        const PerfCounterInfo& counterInfo) = 0;

    /// Queries the layout of counter results in memory for this perf experiment.
    ///
    /// @param [out] pLayout Layout describing the begin and end offset of each counter in the resulting GPU memory once
    ///                      this perf experiment is executed.  Should correspond with counters added via AddCounter().
    ///
    /// @returns Success if the layout was successfully returned in pLayout, otherwise an appropriate error code.
    virtual Result GetGlobalCounterLayout(
        GlobalCounterLayout* pLayout) const = 0;

    /// Addes the specified thread trace to be recorded as part of this perf experiment.
    ///
    /// @param [in] traceInfo Specifies what type of trace to record, which block instance to trace, and options, etc.
    ///
    /// @returns Success if the trace was successfully added to the experiment, otherwise an appropriate error code.
    virtual Result AddThreadTrace(
        const ThreadTraceInfo& traceInfo) = 0;

    /// Adds the specified DfSpmTrace to be recorded as part of this perf experiment.
    ///
    /// @param [in] dfSpmCreateInfo Specifies the parameters of the df spm trace and
    /// provides the list of perf counters.
    ///
    /// @returns Success if the df spm trace was successfully added to the experiment,
    /// otherwise and appropriate error code.
    virtual Result AddDfSpmTrace(
        const SpmTraceCreateInfo& dfSpmCreateInfo) = 0;

    /// Adds the specified SpmTrace to be recorded as part of this perf experiment.
    ///
    /// @param [in] spmCreateInfo Specifies the parameters of the spm trace and provides the list of perf counters.
    ///
    /// @returns Success if the spm trace was successfully added to the experiment, otherwise an appropriate error code.
    virtual Result AddSpmTrace(
        const SpmTraceCreateInfo& spmCreateInfo) = 0;

    /// Queries the layout of thread trace results in memory for this perf experiment.
    ///
    /// @param [out] pLayout Layout describing how the results of each thread trace will be written to GPU memory when
    ///                      this perf experiment is executed.  Should correspond with counters added via AddTrace().
    ///
    /// @returns Success if the layout was successfully returned in pLayout, otherwise an appropriate error code.
    virtual Result GetThreadTraceLayout(
        ThreadTraceLayout* pLayout) const = 0;

    /// Queries the layout of streaming counter trace results in memory for this perf experiment.
    ///
    /// The caller is expected to call this function twice. The first time with pLayout->numCounters = 0 which prompts
    /// PAL to only set numCounters to the correct number of SPM counters and return. The second call with a non-zero
    /// numCounters prompts PAL to fill out the full structure and counterData array.
    ///
    /// Note that @ref SpmTraceLayout contains a variable length array. The caller must allocate enough memory for
    /// an additional "numCounters - 1" copies of @ref SpmCounterData.
    ///
    /// @param [out] pLayout Layout describing the layout of the streaming counter trace results in the resulting
    ///                      GPU memory once this perf experiment is executed.
    ///
    /// @returns Success if the layout was successfully returned in pLayout, otherwise an appropriate error code.
    virtual Result GetSpmTraceLayout(
        SpmTraceLayout* pLayout) const = 0;

    /// Finalizes the performance experiment preparing it for execution.
    ///
    /// @returns Success if the operation executed successfully, otherwise an appropriate error code.
    virtual Result Finalize() = 0;

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
    IPerfExperiment() : m_pClientData(nullptr) {}

    /// @internal Destructor.  Prevent use of delete operator on this interface.  Client must destroy objects by
    /// explicitly calling IDestroyable::Destroy() and is responsible for freeing the system memory allocated for the
    /// object on their own.
    virtual ~IPerfExperiment() { }

private:
    /// @internal Client data pointer. This can have an arbitrary value and can be returned by calling GetClientData()
    /// and set via SetClientData().
    /// For non-top-layer objects, this will point to the layer above the current object.
    void* m_pClientData;
};

} // Pal
