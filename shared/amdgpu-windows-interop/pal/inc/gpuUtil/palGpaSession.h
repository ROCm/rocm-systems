/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2016-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
 * @file  palGpaSession.h
 * @brief PAL GPU utility GpaSession class.
 ***********************************************************************************************************************
 */

#pragma once

#include "palDeque.h"
#include "palDevice.h"
#include "palGpuUtil.h"
#include "palHashSet.h"
#include "palMutex.h"
#include "palPipeline.h"
#include "palVector.h"
#include "palPlatform.h"
#include "palSysMemory.h"
#include "palGpuMemory.h"
#include "palMemTrackerImpl.h"

// Forward declarations.
namespace Pal
{
    class  ICmdAllocator;
    class  ICmdBuffer;
    class  IDevice;
    class  IGpuEvent;
    class  IGpuMemory;
    class  IPerfExperiment;
    class  IQueue;
    class  IQueueSemaphore;
    struct GlobalCounterLayout;
    struct MultiSubmitInfo;
    struct ThreadTraceLayout;
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 900
    enum   PipelineStageFlag : uint32;
#else
    enum   HwPipePoint : uint32;
#endif
}
struct SqttFileChunkCpuInfo;
struct SqttFileChunkAsicInfo;
struct SqttCodeObjectDatabaseRecord;

struct GpuMemoryInfo;

namespace GpuUtil
{
// Sample id initialization value.
constexpr Pal::uint32 InvalidSampleId = 0xFFFFFFFF;

/// The available states of GpaSession
enum class GpaSessionState : Pal::uint32
{
    Reset      = 0,
    Building   = 1,
    Complete   = 2,
    Ready      = 3,
};

/// The various ways you can change trace options after it has started.
enum class UpdateSampleTraceMode : Pal::uint32
{
    MinimalToFullMask      = 0, ///< Used to convert a minimal trace (needed for context in compute presents) to a full
                                ///   trace according to the options in the active trace. Requires enableSampleUpdates.
                                ///  Additionally, this must be called between BeginSample() and EndSample() and
                                ///   queue timing must also be enabled on the GpaSession when this function is called.
    StartInstructionTrace  = 1, ///< Used to enable instruction-level trace globally at any time. Can be run without an
                                ///  active sample. Useful for targeting specific parts of a frame.
    StopInstructionTrace   = 2, ///< Used to disable instruction-level trace globally at any time. Can be run without an
                                ///  active sample.
};

/// Specifies basic type of sample to perfom - either a normal set of "global" perf counters, or a trace consisting
/// of SQ thread trace and/or streaming performance counters.
enum class GpaSampleType : Pal::uint32
{
    None       = 0x0,  ///< No profile will be done.
    Cumulative = 0x1,  ///< One 64-bit result will be returned per performance counter representing the cumulative delta
                       ///  for that counter over the sample period.  Cumulative samples must begin and end in the same
                       ///  command buffer.
    Trace      = 0x2,  ///< A GPU memory buffer will be filled with hw-specific SQ thread trace and/or streaming
                       ///  performance counter data.  Trace samples may span multiple command buffers.
    Timing     = 0x3,  ///< Two 64-bit results will be recorded in beginTs and endTs to gather timestamp data.
    Query      = 0x4,  ///< A set of 11 pipeline stats will be collected.
    Count
};

/// Specifies a specific performance counter to be sampled with GpaSession::BeginSample() and GpaSession::EndSample().
///
/// This identifies a specific counter in a particular HW block instance, e.g., TCC instance 3 counter #19.  It is up
/// to the client to know the meaning of a particular counter, e.g., TCC #19 is TCC_PERF_SEL_MISS on Fiji.  Eventually,
/// PAL may want to support certain counters without the client needing HW-specific knowledge (i.e., select an enum
/// called L2MissRate from PAL rather than needing to know that counter is TCC #19 on Fiji), but GPA currently works in
/// this low-level mode with other drivers, and wants to keep the flexibility.
struct PerfCounterId
{
    Pal::GpuBlock block;    ///< Which GPU block to reference (e.g., CB, DB, TCC).
    Pal::uint32   instance; ///< Which instance of the specified GPU block to sample.  E.g., Tahiti has 12 TCC blocks
                            ///  (this number is returned per-block in the @ref Pal::GpuBlockPerfProperties structure).
                            ///  There is no shortcut to get results for all instances of block in the whole chip, the
                            ///  client must explicitly sample each instance and sum the results.
    Pal::uint32   eventId;  ///< Counter ID to sample.  Note that the meaning of a particular eventId for a block can
                            ///  change between chips.

    union
    {
        struct
        {
            Pal::uint32 spm32Bit : 1;  ///< For SPM counters, collect in 32bit instead of 16bit
            Pal::uint32 reserved : 31; ///< Reserved for future use
        };

        Pal::uint32 u32All; ///< Union value for copying

    } flags;

    // Some blocks have additional per-counter controls. They must be properly programmed when adding counters for
    // the relevant blocks. It's recommended to zero them out when not in use.
    union
    {
        struct
        {
            Pal::uint32 eventQualifier;   ///< The DF counters have an event-specific qualifier bitfield.
        } df;

        struct
        {
            Pal::uint16 eventThreshold;   ///< Threshold value for those UMC counters having event-specific threshold.
            Pal::uint8  eventThresholdEn; ///< Threshold enable (0 for disabled,1 for <threshold,2 for >threshold)
            Pal::uint8  rdWrMask;         ///< Read/Write mask select (1 for Read, 2 for Write).
        } umc;

        Pal::uint32 rs64Cntl; ///< CP blocks CPG and CPC have events that can be further filtered for processor events

        Pal::uint32 u32All; ///< Union value for copying, must be increased in size if any element of the union exceeds
    } subConfig;
};

/// Defines a set of flags for a particular gpa session.
union GpaSessionFlags
{
    struct
    {
        /// Enables timing of queue operations via Timed* functions.
        Pal::uint32 enableQueueTiming        : 1;

        /// Enables sample updates via the UpdateSampleTraceParams function.
        Pal::uint32 enableSampleUpdates      : 1;

        /// Indicates that the client will use the internal Timed*QueueSemaphore() functions for queue semaphore timing
        /// data. When not set it indicates the client will provide ETW data via the ExteralTimed* functions.
        Pal::uint32 useInternalQueueSemaphoreTiming : 1;

        /// Reserved for future use.
        Pal::uint32 reserved                 : 29;
    };

    /// Flags packed as 32-bit uint.
    Pal::uint32 u32All;
};

/// Specifies options that direct the gpa session behavior.
struct GpaSessionBeginInfo
{
    /// Gpa Session flags used to control behavior.
    GpaSessionFlags flags;
};

/// Input structure for CmdBeginGpuProfilerSample.
///
/// Defines a set of global performance counters and/or SQ thread trace data to be sampled.
struct GpaSampleConfig
{
    /// Selects what type of data should be gathered for this sample.  This can either be _cumulative_ to gather
    /// simple deltas for the specified set of perf counters over the sample period, or it can be _trace_ to generate
    /// a blob of RGP-formatted data containing SQ thread trace and/or streaming performance monitor data.
    GpaSampleType type;

    union
    {
        struct
        {
            Pal::uint32 sampleInternalOperations      :  1;  ///< Include BLTs and internal driver operations in the
                                                             ///  results.
            Pal::uint32 cacheFlushOnCounterCollection :  1;  ///< Insert cache flush and invalidate events before and
                                                             ///  after every sample.
            Pal::uint32 sqShaderMask                  :  1;  ///< If sqShaderMask is valid.
            Pal::uint32 sqWgpShaderMask               :  1;  ///< If sqWgpShaderMask is valid.
            Pal::uint32 reserved                      : 28;  ///< Reserved for future use.
        };
        Pal::uint32 u32All;                                  ///< Bit flags packed as uint32.
    } flags;                                                 ///< Bit flags controlling sample operation for all sample
                                                             ///  types.

    Pal::PerfExperimentShaderFlags sqShaderMask;    ///< Which shader stages are sampled by GpuBlock::Sq counters.
                                                    ///< Only used if flags.sqShaderMask is set to 1.
    Pal::PerfExperimentShaderFlags sqWgpShaderMask; ///< Which shader stages are sampled by GpuBlock::SqWgp counters.
                                                    ///< Only used if flags.sqWgpShaderMask is set to 1.

    struct
    {
        /// Number of entries in pIds.
        Pal::uint32 numCounters;

        /// List of performance counters to be gathered for a sample.  If the sample type is _cumulative_ this will
        /// result in "global" perf counters being sampled at the beginning of the sample period; if the sample type
        /// is _trace_ this will result in SPM data being added to the sample's resulting RGP blob.
        ///
        /// Note that it is up to the client to respect the hardware counter limit per block.  This can be
        /// determined by the maxGlobalOnlyCounters, maxGlobalSharedCounters, maxSpmCounters, and instanceGroupSize
        /// fields of @ref Pal::GpuBlockPerfProperties.
        const PerfCounterId* pIds;

        /// Period for SPM sample collection in cycles.  Only relevant for _trace_ samples.
        Pal::uint32  spmTraceSampleInterval;

        /// Maximum amount of GPU memory in bytes this sample can allocate for SPM data.  Only relevant for _trace_
        /// samples.
        Pal::gpusize gpuMemoryLimit;
    } perfCounters;  ///< Performance counter selection (valid for both _cumulative_ and _trace_ samples).

    struct
    {
        /// Number of entries in pIds.
        Pal::uint32  numCounters;

        /// Period for DF SPM sample collection in nano seconds.
        Pal::uint32  sampleInterval;

        /// Maximum amount of GPU memory in bytes this sample can allocate for DF SPM data.
        Pal::gpusize gpuMemoryLimit;

        /// List of performance counters to be gathered for a df sample. This has to be separate from the list
        /// list of normal counters because it is a completely different mechanism for gathering data.
        ///
        /// Note that it is up to the client to respect the hardware counter limit per block.  This can be
        /// determined by the maxSpmCounters fields of
        /// @ref Pal::GpuBlockPerfProperties.
        const PerfCounterId* pIds;
    } dfSpmPerfCounters;

    struct
    {
        union
        {
            struct
            {
                Pal::uint32 enable                     :  1;  ///< Include SQTT data in the trace.
                Pal::uint32 supressInstructionTokens   :  1;  ///< Prevents capturing instruction-level SQTT tokens,
                                                              ///  significantly reducing the amount of sample data.
                Pal::uint32 stallMode                  :  2;  ///< Describes behavior when buffer full
                Pal::uint32 stallAllSimds              :  1;  ///< Stall all SIMDs for thread trace stall.
                Pal::uint32 excludeNonDetailShaderData :  1;  ///< Only emit shader tokens from the SIMD that have been
                                                              ///  selected for detail instruction tracing
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 899
                Pal::uint32 enableExecPopTokens        :  1;  ///< Output exec tokens
#else
                Pal::uint32 placeholder2               :  1;
#endif
                Pal::uint32 reserved                   : 25;  ///< Reserved for future use.
            };
            Pal::uint32 u32All;                             ///< Bit flags packed as uint32.
        } flags;                                            ///< Bit flags controlling SQTT samples.
        Pal::uint32 seMask;          ///< Mask that determines which specific SEs to run Thread trace on.
                                     ///  If 0, all SEs are enabled
        Pal::uint32 seDetailedMask;  ///< Mask that selects which specific SEs to reveal Thread trace detailed info.
                                     ///  If 0, all SEs will reveal detailed thread trace
        Pal::gpusize gpuMemoryLimit; ///< Maximum amount of GPU memory in bytes this sample can allocate for the SQTT
                                     ///  buffer.  If 0, allocate maximum size to prevent dropping tokens toward the
                                     ///  end of the sample.
        Pal::uint32 tokenMask;       ///< Mask indicating which SQTT tokens are requested for capture. If a tokenMask is
                                     ///  not provided, PAL will default to collecting all tokens or tokens except
                                     ///  instruction tokens if the supressInstructionTokens flag is set. Instruction
                                     ///  tokens will always be filtered out if supressInstructionTokens = true.
    } sqtt;  ///< SQ thread trace configuration (only valid for _trace_ samples).

    struct
    {
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 900
        Pal::PipelineStageFlag preSample;  ///< The pipeline stage in the GPU pipeline where the begin timestamp should
                                           ///  take place.
        Pal::PipelineStageFlag postSample; ///< The pipeline stage in the GPU pipeline where the end timestamp should
                                           ///  take place.
#else
        Pal::HwPipePoint preSample;   ///< The point in the GPU pipeline where the begin timestamp should take place.
        Pal::HwPipePoint postSample;  ///< The point in the GPU pipeline where the end timestamp should take place.
#endif
    } timing;  ///< Timestamp configuration. (only valid for timing samples)
};

/// Extra metadata about a command buffer submission
struct TimedSubmitInfo
{
    const Pal::uint64* pApiCmdBufIds;  ///< Array of api specific command buffer ids
    const Pal::uint32* pSqttCmdBufIds; ///< Array of sqtt command buffer ids
    Pal::uint64 frameIndex;            ///< The global frame index for the application.
};

/// Extra metadata about a queue semaphore operation
struct TimedQueueSemaphoreInfo
{
    Pal::uint64 semaphoreID; ///< Api specific id associated with a semaphore.
};

/// Extra metadata about a queue present operation
struct TimedQueuePresentInfo
{
    Pal::uint64 presentID; ///< Api specific id associated with a present.
};

/// Struct for storing information about gpu clock speeds.
struct GpuClocksSample
{
    Pal::uint32 gpuEngineClockSpeed; // Current speed of the gpu engine clock in MHz
    Pal::uint32 gpuMemoryClockSpeed; // Current speed of the gpu memory clock in MHz
};

/// Struct for storing CPU-side allocations of Pal::IPerfExperiment's.
struct PerfExperimentMemory
{
    void*  pMemory;     // Memory allocated for an IPerfExperiment.
    size_t memorySize;  // Size of the memory allocated in pMemory.
};

/// Struct for supplying API-dependent information about pipelines.
struct RegisterPipelineInfo
{
    Pal::uint64 apiPsoHash;  ///< Client-provided PSO hash.
};

/// Struct for supplying API-dependent information about libraries.
struct RegisterLibraryInfo
{
    Pal::uint64 apiHash;      ///< Client-provided api hash.
};

/// Struct for supplying Elf binary.
struct ElfBinaryInfo
{
    const void* pBinary;          ///< FAT Elf binary.
    Pal::uint32 binarySize;       ///< FAT Elf binary size.
    Pal::IGpuMemory* pGpuMemory;  ///< GPU Memory where the compiled ISA resides.
    Pal::gpusize offset;          ///< Offset inside GPU memory object
    Pal::uint64 originalHash;     ///< Original source/binary hash.
    Pal::uint64 compiledHash;     ///< Compiled binary hash.
};

/// Enumeration of RGP trace profiling modes
enum class TraceProfilingMode : Pal::uint32
{
    Present     = 0,    ///< Present triggered capture
    UserMarkers = 1,    ///< Capture triggered by user marker
    FrameNumber = 2,    ///< Capture based on frame number
    Tags        = 3,    ///< Tag based capture
};

/// Constant defines the maximum length for a user marker string.
static constexpr Pal::uint32 UserMarkerStringLength = 256;

/// Defines data specific to each profiling mode used to capture an RGP trace.
union TraceProfilingModeData
{
    struct
    {
        char start[UserMarkerStringLength];     ///< User marker string used to start trace capture.
        char end[UserMarkerStringLength];       ///< User marker string used to end trace capture.
    } userMarkerData;

    struct
    {
        Pal::uint32 start;                      ///< Frame number used to start the trace.
        Pal::uint32 end;                        ///< Frame number used to end the trace.
    } frameNumberData;

    struct
    {
        Pal::uint64 start;                      ///< Tag used to start the trace.
        Pal::uint64 end;                        ///< Tag used to end the trace.
    } tagData;
};

/// Enumerates the different instruction level data modes for an RGP trace
enum class InstructionTraceMode : Pal::uint32
{
    Disabled  = 0, ///< Instruction level data was disabled for trace.
    FullFrame = 1, ///< Instruction level data was enabled for the full trace.
    ApiPso    = 2, ///< Instruction level data was enabled only for a single API PSO.
};

/// Defines the data used to control enabling of instruction level data.
struct InstructionTraceModeData
{
    Pal::uint64 apiPsoHash;     ///< Hash of the API PSO targeted for instruction level data.
};

/// Struct for supplying API specific information about an RGP trace
struct SampleTraceApiInfo
{
    TraceProfilingMode       profilingMode;             ///< Profiling mode used to trigger the trace.
    TraceProfilingModeData   profilingModeData;         ///< Profiling mode specific data.
    InstructionTraceMode     instructionTraceMode;      ///< Instruction trace mode for the trace.
    InstructionTraceModeData instructionTraceModeData;  ///< Instruction trace mode data.
};

/// An enumeration of the API types.
enum class ApiType : Pal::uint32
{
    DirectX12 = 0,    ///< Represents DirectX12 API type.
    Vulkan    = 1,    ///< Represents Vulkan API type.
    Generic   = 2,    ///< Represents Generic API type.
    OpenCl    = 3,    ///< Represents OpenCL API type.
    Hip       = 5,    ///< Represents HIP API type.
};

/// Struct used for storing SQTT-specific trace information
struct SqttTraceInfo
{
    Pal::uint32 shaderEngine;    ///< Shader engine index
    Pal::uint32 computeUnit;     ///< Compute unit index
    Pal::uint32 sqttVersion;     ///< SQTT version
    Pal::uint64 bufferSize;      ///< SQTT trace buffer size
};

/// Struct used for storing SPM-specific trace information
struct SpmTraceInfo
{
    Pal::uint32 numSpmCounters;  ///< The number of SPM counters sampled in the trace
    Pal::uint32 numTimestamps;   ///< The number of timestamps that samples were taken
    Pal::uint32 sampleFrequency; ///< The SPM counter sampling frequency
};

/// Struct used for storing QueueTimings-specific trace information
struct QueueTimingsTraceInfo
{
    Pal::uint32 numQueueInfoRecords;
    Pal::uint32 numQueueEventRecords;
    Pal::uint32 queueInfoTableSize;
    Pal::uint32 queueEventTableSize;
};

/**
***********************************************************************************************************************
* @class GpaSession
* @brief Helper class providing common driver functionality required by all PAL clients that support the GPUPerfAPI
*        (GPA).  Abstracts IPerfExperiment creation, memory management, completion confirmation, and results reporting
*        at a level convenient for GPA.  Each PAL client driver will need to publish an API extension exposing this
*        support for use by GPA.
*
* A GpaSession is a container for a set of _samples_ of performance counter and/or SQ thread trace data.  Its main
* purpose is to manage resources (IPerfExperiments and their backing system/GPU memory) in an efficient manner that is
* consistent with command buffer management in modern APIs.  Consider GpaSession as a peer of DX12's command
* allocator or Vulkan's command pool objects.
*
* Basic flow of usage:
*     - Newly create sessions are in the _reset_ state.
*     - A session is moved from the _reset_ state to the _building_ state by calling Begin().
*     - Samples are added to a session by specifying desired data for each query and marking a begin and end location
*       in ICmdBuffers as they are built.  Internally required resources, like GPU memory where counters will be
*       written, are allocated from internal pools managed by the session.
*     - A session is moved from the _building_ state to the _complete_ state by calling End().
*     - The application will submit all command buffers referenced by the session.
*     - The session is confirmed as _ready_, either using standard PAL fences to confirm all assocated submission have
*       completed, or by polling IsReady() on the session.
*     - Results for all samples in the session can be queried via GetResults().
*     - Reset() should be called once results have been gathered and before building a new session.  Resources are
*       retained by the session object for use in the newly built session.  The session object must be destroyed in
*       order to fully release all resource back to the system.
*
* Cumulative-type samples may not span multiple command buffers, as other apps could interfere with the counts and
* there the final data doesn't have time-based visibility to detect that happened.
*
* @warning GpaSession is not thread safe.  Performing samples in command buffers being built simultaneously by multiple
*          threads should use multiple GpaSession objects.
***********************************************************************************************************************
*/
class GpaSession
{
    typedef Pal::IPlatform             GpaAllocator;
public:
    typedef Util::Deque<PerfExperimentMemory, GpaAllocator> PerfExpMemDeque;

    /// Constructor.
    GpaSession(
        Pal::IPlatform*      pPlatform,
        Pal::IDevice*        pDevice,
        Pal::uint16          apiMajorVer,
        Pal::uint16          apiMinorVer,
        ApiType              apiType,
        Pal::uint16          rgpInstrumentationSpecVer = 0,
        Pal::uint16          rgpInstrumentationApiVer  = 0,
        PerfExpMemDeque*     pAvailablePerfExpMem      = nullptr);

    ~GpaSession();

    /// Copy constructor creates an empty copy of a session.
    ///
    /// Newly constructed session copies the GPU memory allocations and their layout from the source session, making
    /// this a valid destination for a CopyResults command.  This new object is effectively in the _complete_ state.
    ///
    /// The purpose of such objects is to handle sampling data from bundles or nested command buffers where the same
    /// set of commands might be executed multiple times from a single root-level command buffer.  The client should
    /// note such cases, and create a copy of the bundle's session for each invocation, then call CopyResults() from
    /// the original session into the copy after the invocation.
    ///
    /// @param [in] src Session to be copied.  Must either be in the _complete_ or _ready_ state.
    explicit GpaSession(const GpaSession& src);

    /// Initialize the newly constructed GPA session.
    Pal::Result Init();

    /// Registers a queue with the GpaSession that will be submitted to using TimedSubmit. This must be called on any
    /// queues that are submitted to via the Timed* functions. For Timed* signal and wait queue semaphore events, a
    /// valid queueContext will be required (queueContext not equal to 0).
    Pal::Result RegisterTimedQueue(Pal::IQueue* pQueue,
                                   Pal::uint64 queueId,
                                   Pal::uint64 queueContext);

    /// Unregisters a queue prior to object destruction, and ensure that associated resources are destroyed. Work can
    /// no longer be submitted on the queue after this has been called.
    Pal::Result UnregisterTimedQueue(Pal::IQueue* pQueue);

    Pal::Result TimedSubmit(Pal::IQueue*                pQueue,
                            const Pal::MultiSubmitInfo& submitInfo,
                            const TimedSubmitInfo&      timedSubmitInfo);

    /// Executes a timed queue semaphore signal through the given queue.  The HW time is measured when the queue semaphore
    /// is signaled.
    Pal::Result TimedSignalQueueSemaphore(Pal::IQueue* pQueue,
                                          Pal::IQueueSemaphore* pQueueSemaphore,
                                          const TimedQueueSemaphoreInfo& timedSignalInfo,
                                          Pal::uint64  value = 0);

    /// Executes a timed queue semaphore wait through the given queue.  The HW time is measured when the queue semaphore
    /// wait finishes.
    Pal::Result TimedWaitQueueSemaphore(Pal::IQueue* pQueue,
                                        Pal::IQueueSemaphore* pQueueSemaphore,
                                        const TimedQueueSemaphoreInfo& timedWaitInfo,
                                        Pal::uint64  value = 0);

    /// Injects a timed queue present event.
    Pal::Result TimedQueuePresent(Pal::IQueue* pQueue,
                                  const TimedQueuePresentInfo& timedPresentInfo);

    /// Injects a timed wait queue semaphore event using information supplied by an external source.
    /// A valid queueContext (queueContext not equal to 0) is needed for this function.
    Pal::Result ExternalTimedWaitQueueSemaphore(Pal::uint64 queueContext,
                                                Pal::uint64 cpuSubmissionTimestamp,
                                                Pal::uint64 cpuCompletionTimestamp,
                                                const TimedQueueSemaphoreInfo& timedWaitInfo);

    /// Injects a timed signal queue semaphore event using information supplied by an external source.
    /// A valid queueContext (queueContext not equal to 0) is needed for this function.
    Pal::Result ExternalTimedSignalQueueSemaphore(Pal::uint64 queueContext,
                                                  Pal::uint64 cpuSubmissionTimestamp,
                                                  Pal::uint64 cpuCompletionTimestamp,
                                                  const TimedQueueSemaphoreInfo& timedSignalInfo);

    /// Queries the engine and memory clocks from DeviceProperties
    Pal::Result SampleGpuClocks(GpuClocksSample* pGpuClocksSample) const;

    /// Samples the timing clocks if queue timing is enabled and adds a clock sample entry to the current session.
    Pal::Result SampleTimingClocks();

    /// Moves the session from the _reset_ state to the _building_ state.
    ///
    /// Invalid to call Begin() on a session that isn't in the _reset_ state.
    ///
    /// @param [in] info Information about the gpa sessions desired behavior.
    ///
    /// @returns Success if the session was successfully moved to the _building_ state.  Otherwise, possible errors
    ///          include:
    ///          + ErrorUnavailable if the sessions isn't current in the _reset_ state.
    Pal::Result Begin(const GpaSessionBeginInfo& info);

    /// Moves the session from the _building_ state to the _complete_ state.
    ///
    /// Invalid to call End() on a session that isn't in the _building_ state.  The implementation _may_ insert GPU
    /// commands into the specified pCmdBuf - in the case of a session that spans multiple command buffers, the
    /// command buffer specified to End() _must_ be the last command buffer of the session that is submitted.
    ///
    /// @param [in] pCmdBuf Last (normally _only_) command buffer of the session.  Can be used by implementation
    ///                     to insert GPU commands required after all samples are inserted (e.g., to confirm session
    ///                     completion).
    ///
    /// @returns Success if the session was successfully moved to the _complete_ state.  Otherwise, possible errors
    ///          include:
    ///          + ErrorUnavailable if the sessions isn't current in the _building_ state.
    Pal::Result End(Pal::ICmdBuffer* pCmdBuf);

    /// Marks the beginning of a range of GPU operations to be measured and specifies what data should be recorded.
    ///
    /// It is possible the sample will not succeed due to internal memory allocation failure, etc.  In those cases,
    /// the session will be marked invalid and no sample commands will be inserted.  Reporting of this error is
    /// delayed until GetResults().
    ///
    /// A note for GpuBlock::SqWgp
    /// Client of palPerfExperiment may configure counters of GpuBlock::SqWgp based on a per-wgp granularity
    /// only if the following are disabled: GFXOFF, virtualization/SRIOV, VDDGFX (power down features), clock
    /// gating (CGCG) and power gating. PAL expose this feature to clients.
    /// If any of the conditions above cannot be met, it's the client's job to set all WGPs in the same SE to the same
    /// perf counter programming. In this case, GpuBlock::SqWgp's perf counter works on a per-SE granularity.
    /// Strictly speaking, it's not true that the counters work on a per-SE granularity when those power features
    /// are enabled. It's all still per-WGP in HW, we just can't support different counter configs within the same SE.
    /// The counter data is still reported per WGP (not aggregated for the whole SE).
    ///
    /// Check the following two documents for details:
    ///
    /// @param [in]  pCmdBuf      Command buffer to issue the begin sample commands.  All operations performed
    ///                           between executing the BeginSample() and EndSample() GPU commands will contribute to
    ///                           the sample results.
    /// @param [in]  sampleConfig Describes what data should be sampled.
    /// @param [out] pSampleId    An ID corresponding to this sample.  This ID should be recorded and passed back to
    ///                           EndSample() when the sampled command buffer range is complete.  This ID should also
    ///                           be passed to GetResults() when the session is in the _ready_ state in order to get
    ///                           the results of this sample.
    ///
    /// @returns Success if the update was successful.  Unsupported if the sample config type is not supported.
    ///          Otherwise, possible errors include:
    ///          + ErrorInvalidPointer if pCmdBuf or pSampleId is nullptr.
    Pal::Result BeginSample(
        Pal::ICmdBuffer*       pCmdBuf,
        const GpaSampleConfig& sampleConfig,
        Pal::uint32*           pSampleId);

    /// Updates the trace parameters for a specific sample.
    ///
    /// @param [in] pCmdBuf    Command buffer to issue the update commands.
    /// @param [in] sampleId   Identifies the sample to be updated, if required by the mode.  This should be a value
    ///                        returned by BeginSample(), and must correspond to a thread trace sample.
    /// @param [in] updateMode The way the sample parameters should be set. Some modes have additional restrictions.
    ///                        @see UpdateSampleTraceMode
    ///
    /// @returns Success if the update was successful.  Otherwise, possible errors
    ///          include:
    ///          + ErrorInvalidPointer if pCmdBuf is nullptr.
    ///          + ErrorInvalidObjectType if a sample is required and the sample associated with sampleId is not a
    ///                                   trace sample.
    Pal::Result UpdateSampleTraceParams(
        Pal::ICmdBuffer*          pCmdBuf,
        Pal::uint32               sampleId,
        UpdateSampleTraceMode     updateMode);

    /// Marks the end of a range of command buffer operations to be measured.
    ///
    /// @param [in] pCmdBuf  Command buffer to issue the end sample commands.  All operations performed between
    ///                      executing the BeginSample() and EndSample() GPU commands will contribute to the sample
    ///                      results.  _Cumulative_ samples (i.e., global performance counter samples) must never span
    ///                      multiple command buffers (EndSample() should be called in the same command buffer as
    ///                      BeginSample()).
    /// @param [in] sampleId Identifies the sample to be ended.  This should be the value returned by BeginSample()
    ///                      for the sample that is being ended.
    ///
    /// @note BeginSample() must be called before EndSample() _and_ the GPU commands inserted by BeginSample() must be
    ///       executed before the command inserted by EndSample().  Since a session is a single-threaded object, this
    ///       will normally happen naturally.
    void EndSample(
        Pal::ICmdBuffer* pCmdBuf,
        Pal::uint32      sampleId);

    /// Copies the DF SPM trace buffer to the GpaSession result buffer
    ///
    /// @param [in] pCmdBuf  Command buffer to issue the copy commands.
    /// @param [in] sampleId Identifies the sample to be copied.
    /// @note This must be called after a command buffer with the dfSpmTraceEnd CmdBufInfo flag
    ///       and with a separate command buffer. DF SPM traces are on a per command buffer granularity
    ///       because they are started and stopped by the KMD.
    void CopyDfSpmTraceResults(
        Pal::ICmdBuffer* pCmdBuf,
        Pal::uint32      sampleId);

    /// Provides API specific information about an RGP trace.
    ///
    /// @param [in] traceApiInfo  Const reference to the struct of API specific information.
    /// @param [in] sampleId      Sample ID (returned by BeginSample) for the RGP trace type sample info is being
    ///                           provided for.
    void SetSampleTraceApiInfo(
        const SampleTraceApiInfo& traceApiInfo,
        Pal::uint32               sampleId) const;

    /// Reports if GPU execution of this session has completed and results are _ready_ for querying from the CPU via
    /// GetResults().
    ///
    /// @returns true if all samples in the session have completed GPU execution.
    bool IsReady() const;

    /// Reports results of a particular sample.  Only valid for sessions in the _ready_ state.
    ///
    /// Results will be formatted depending on the sample type:
    /// + Cumulative: Results will be an array of uint64 values in the order of perf counter IDs specified by
    ///               BeginSample().
    /// + SqThreadTrace: Results will be a binary blob in the RGP file format.
    ///
    /// @param [in]     sampleId     Sample to be reported.  Corresponds to value returned by BeginSample().
    /// @param [in,out] pSizeInBytes If pData is non-null, the input value of *pSizeInBytes is the amount of space
    ///                              available in pData, and *pSizeInBytes will be set to the amount of space written
    ///                              to pData.  If pData is null, *pSizeInBytes will be set to the amount of space
    ///                              required.
    /// @param [out]    pData        Can be null to query how much size is required (should only be necessary when
    ///                              getting RGP data).  If non-null, the sample results will be written to this
    ///                              location.
    ///
    /// @returns Success if the sample results are successfully written to pData (or, if pData is null, the required
    ///          size is successfully written to pSizeInBytes).  Otherwise, possible errors include:
    ///          + ErrorUnavailable if the session is not in the _ready_ state.
    ///          + ErrorOutOfGpuMemory if the session wasn't properly built due to running out of GPU memory resources.
    ///          + ErrorInvalidMemorySize if *pSizeInBytes isn't big enough to hold the results.
    Pal::Result GetResults(
        Pal::uint32 sampleId,
        size_t*     pSizeInBytes,
        void*       pData) const;

    /// Retrieves the SQTT results. Only valid for sessions in the _complete_ state.
    ///
    /// @param [in]     sampleId       Sample to be reported.  Corresponds to value returned by BeginSample().
    /// @param [in]     traceIndex     The index of the trace to get.
    /// @param [out]    pTraceInfoOut  Optional pointer to a structure which will be written with information about the trace.
    /// @param [in,out] pSizeInBytes   If pData is non-null, the input value of *pSizeInBytes is the amount of space
    ///                                available in pData, and *pSizeInBytes will be set to the amount of space written
    ///                                to pData.  If pData is null, *pSizeInBytes will be set to the amount of space
    ///                                required.
    /// @param [out]    pData          Can be null to query how much size is required.
    ///                                If non-null, the sample results will be written to this location.
    ///
    /// @returns Success if the sample results are successfully written to pData (or, if pData is null, the required
    ///          size is successfully written to pSizeInBytes).  Otherwise, possible errors include:
    ///          + ErrorUnavailable if the session is not in the _ready_ state.
    ///          + NotFound if the given index is not valid.
    ///          + ErrorOutOfGpuMemory if the session wasn't properly built due to running out of GPU memory resources.
    ///          + ErrorInvalidMemorySize if *pSizeInBytes isn't big enough to hold the results.
    //           + ErrorInvalidPointer if pSizeInBytes is NULL.
    Pal::Result GetSqttTraceData(
        Pal::uint32    sampleId,
        Pal::uint32    traceIndex,
        SqttTraceInfo* pTraceInfo,
        size_t*        pSizeInBytes,
        void*          pData) const;

    /// Retrieves the SPM trace results of a particular sample. Only valid for 'Trace' type samples and sessions
    /// in the _complete_ state.
    ///
    /// Results in the output buffer are a binary blob formatted according to the RGP specification.
    /// The data layout of the populated output buffer is as follows:
    ///     - Timestamps array        [size: "numTimestamps * sizeof(uint64)" bytes]
    ///     - SpmCounterInfo array    [size: "numSpmCounters * sizeof(SpmCounterInfo)" bytes]
    ///     - SPM Counter Data matrix [size: "*pSizeInBytes - (timestamps array + SpmCounterInfo array size)" bytes]
    ///
    /// The SPM Counter Data matrix is laid out linearly in a row-major format. There are "numSpmCounters" rows and
    /// "numTimestamps" columns. Each element in the matrix is either 16- or 32-bits, based on the "dataSize" field
    /// of the corresponding "SpmCounterInfo" entry.
    ///
    /// @param [in]     sampleId          Sample to be reported.  Corresponds to value returned by BeginSample().
    /// @param [out]    pTraceInfo        Optional. If non-null, this structure is populated with trace metadata.
    /// @param [in,out] pSizeInBytes      If pData is non-null, the input value of *pSizeInBytes is the amount of space
    ///                                   available in pData.
    ///                                   If pData is null, *pSizeInBytes will be set to the amount of space
    ///                                   required.
    /// @param [out]    pData             Can be null to query how much size is required.
    ///                                   If non-null, the sample results will be written to this location.
    ///
    /// @returns Success if the sample results are successfully written to pData (or, if pData is null, the required
    ///          size is successfully written to pSizeInBytes).  Otherwise, possible errors include:
    ///          + ErrorUnavailable if the session is not in the _ready_ state.
    ///          + ErrorOutOfGpuMemory if the session wasn't properly built due to running out of GPU memory resources.
    ///          + ErrorInvalidMemorySize if *pSizeInBytes isn't big enough to hold the results.
    Pal::Result GetSpmTraceData(
        Pal::uint32   sampleId,
        SpmTraceInfo* pTraceInfo,
        size_t*       pSizeInBytes,
        void*         pData) const;

    /// Retrieves the Queue Timings data from the active GpaSession.
    /// Only valid when the GpaSession had `enableQueueTiming` flag set.
    ///
    /// @param [out]    pTraceInfo        Optional. If non-null, this structure is populated with metadata.
    /// @param [in,out] pSizeInBytes      If pData is non-null, the input value of *pSizeInBytes is the amount of space
    ///                                   available in pData.
    ///                                   If pData is null, *pSizeInBytes will be set to the amount of space
    ///                                   required.
    /// @param [out]    pData             Can be null to query how much size is required.
    ///                                   If non-null, the sample results will be written to this location.
    ///
    /// @returns Success if the sample results are successfully written to pData (or, if pData is null, the required
    ///          size is successfully written to pSizeInBytes).  Otherwise, possible errors include:
    ///          + ErrorUnavailable if the session was not configured with `enableQueueTiming`.
    Pal::Result GetQueueTimingsData(
        QueueTimingsTraceInfo* pTraceInfo,
        size_t*                pSizeInBytes,
        void*                  pData) const;

    /// Moves the session to the _reset_ state, marking all sessions resources as unused and available for reuse when
    /// the session is re-built.
    ///
    /// @warning This function cannot be called when the session is queued for execution on the GPU.  The client must
    ///          confirm this is not the case using IsReady(), fences, etc.
    ///
    /// @returns Success if the session was successfully moved to the _reset_ state.  Otherwise, possible errors
    ///          include:
    ///          + ErrorUnknown if an internal PAL error occurs.
    Pal::Result Reset();

    /// Uses the GPU to copy results from a nested command buffer's session into a root-level command buffer's per-
    /// invocation session data.
    ///
    /// This command will implicitly wait for the source session (as specified in the copy constructor) to be complete
    /// then use the GPU to update this session's data.  This allows the client to get accurate sample data in the
    /// case where a nested command buffer is launched multiple times from the same root-level command buffer.
    ///
    /// The session remains in the _complete_ state after calling this, and the client should submit the commands
    /// and verify their completion to move to the _ready_ state.
    ///
    /// @param pCmdBuf Command buffer where the session copy should be performed.
    void CopyResults(Pal::ICmdBuffer* pCmdBuf);

    /// Register pipeline with GpaSession for obtaining shader dumps and load events in the RGP file.
    ///
    /// @param [in] pPipeline   The PAL pipeline to be tracked.
    /// @param [in] clientInfo  API-dependent information for this pipeline to also be recorded.
    ///
    /// @returns Success if the pipeline has been registered with GpaSession successfully.
    ///          + AlreadyExists if a duplicate pipeline is provided.
    Pal::Result RegisterPipeline(const Pal::IPipeline* pPipeline, const RegisterPipelineInfo& clientInfo);

    /// Unregister pipeline with GpaSession for obtaining unload events in the RGP file.
    /// This should be called immediately before destroying the PAL pipeline object.
    ///
    /// @param [in] pPipeline  The PAL pipeline to be tracked.
    ///
    /// @returns Success if the pipeline has been unregistered with GpaSession successfully.
    Pal::Result UnregisterPipeline(const Pal::IPipeline* pPipeline);

    /// Register library with GpaSession for obtaining shader dumps and load events in the RGP file.
    ///
    /// @param [in] pLibrary   The PAL library to be tracked.
    /// @param [in] clientInfo API-dependent information for this library to also be recorded.
    ///
    /// @returns Success if the library has been registered with GpaSession successfully.
    ///          + AlreadyExists if a duplicate library is provided.
    Pal::Result RegisterLibrary(const Pal::IShaderLibrary* pLibrary, const RegisterLibraryInfo& clientInfo);

    /// Unregister library with GpaSession for obtaining unload events in the RGP file.
    /// This should be called immediately before destroying the PAL library object.
    ///
    /// @param [in] pLibrary The PAL library to be tracked.
    ///
    /// @returns Success if the library has been unregistered with GpaSession successfully.
    Pal::Result UnregisterLibrary(const Pal::IShaderLibrary* pLibrary);

    /// Register ELF binary with GpaSession for obtaining kernel dumps and load events in the RGP file.
    ///
    /// @param [in] elfBinaryInfo Contains information about the Elf binary to be recorded.
    ///
    /// @returns Success if the Elf binary has been registered with GpaSession successfully.
    Pal::Result RegisterElfBinary(const ElfBinaryInfo& elfBinaryInfo);

    /// Unregister Elf binary with GpaSession for obtaining unload events in the RGP file.
    /// This should be called immediately before destroying the Elf binary.
    ///
    /// @param [in] elfBinaryInfo  Contains the elf binary info to be removed from tracking.
    ///
    /// @returns Success if the library has been unregistered with GpaSession successfully.
    Pal::Result UnregisterElfBinary(const ElfBinaryInfo& elfBinaryInfo);

    /// Given a Pal device, validate a list of perfcounters.
    ///
    /// @param [in] pDevice      a given device
    /// @param [in] pCounters    a list of perf counters.
    /// @param [in] numCounters  perf counter counts.
    ///
    /// @returns Success if counters are valid.
    Pal::Result ValidatePerfCounters(Pal::IDevice*        pDevice,
                                     const PerfCounterId* pCounters,
                                     const Pal::uint32    numCounters);

private:
    // Tracking structure for a single IGpuMemory allocation owned by a GpaSession::GpaSession. In particular, it
    // tracks the associated CPU pointer since these allocations remain mapped for CPU access for their lifetime.
    struct GpuMemoryInfo
    {
        Pal::IGpuMemory* pGpuMemory;
        void*            pCpuAddr;
    };

    // Event type for code object load events
    enum class CodeObjectLoadEventType
    {
        LoadToGpuMemory     = 0,
        UnloadFromGpuMemory
    };

    // Represents all information to be contained in one SqttCodeObjectLoaderEventRecord
    struct CodeObjectLoadEventRecord
    {
        CodeObjectLoadEventType eventType;
        Pal::uint64             baseAddress;
        Pal::ShaderHash         codeObjectHash;
        Pal::uint64             timestamp;
    };

    // Represents all information to be contained in one SqttPsoCorrelationRecord
    struct PsoCorrelationRecord
    {
        Pal::uint64        apiPsoHash;
        Pal::PipelineHash  internalPipelineHash;
    };

    // Registers a single (non-archive) pipeline with the GpaSession. Returns AlreadyExists on duplicate PAL pipeline.
    Pal::Result RegisterSinglePipeline(const Pal::IPipeline* pPipeline, const RegisterPipelineInfo& clientInfo);

    // Unregisters a single (non-archive) pipeline from the GpaSession.
    Pal::Result UnregisterSinglePipeline(const Pal::IPipeline* pPipeline);

    Pal::IDevice*const            m_pDevice;                    // Device associated with this GpaSession.
    Pal::DeviceProperties         m_deviceProps;
    Pal::SetClockModeOutput       m_peakClockFrequency;         // Output of query for stable peak, values in Mhz
    Pal::PerfExperimentProperties m_perfExperimentProps;
    Pal::uint32                   m_timestampAlignment;         // Pre-calculated timestamp data alignment.
    ApiType                       m_apiType;                    // API type, e.g. Vulkan, used in RGP dumps.
    Pal::uint16                   m_apiMajorVer;                // API major version, used in RGP dumps.
    Pal::uint16                   m_apiMinorVer;                // API minor version, used in RGP dumps.
    Pal::uint16                   m_instrumentationSpecVersion; // Spec version of RGP instrumetation.
    Pal::uint16                   m_instrumentationApiVersion;  // Api version of RGP instrumetation.

    Pal::IGpuEvent*               m_pGpuEvent;
    GpaSessionState               m_sessionState;

    const GpaSession* const       m_pSrcSession;                // source session for session created via copy c'tor

    // Tracks the current GPU memory object and offset being sub-allocated for AcquireGpuMem().
    GpuMemoryInfo                 m_curGartGpuMem;
    Pal::gpusize                  m_curGartGpuMemOffset;
    GpuMemoryInfo                 m_curLocalGpuMem;
    Pal::gpusize                  m_curLocalGpuMemOffset;
    GpuMemoryInfo                 m_curInvisGpuMem;
    Pal::gpusize                  m_curInvisGpuMemOffset;

    // Locks for the local-invisible, gart and local memory subdivision (and their pools)
    Util::Mutex m_gartGpuMemLock;
    Util::Mutex m_localGpuMemLock;
    Util::Mutex m_invisGpuMemLock;

    // Counts number of samples that are active in this GpaSession.
    Pal::uint32                   m_sampleCount;

    Pal::IPlatform*const          m_pPlatform;                  // Platform associated with this GpaSesion.

    // GartHeap / LocalHeap / InvisHeap GPU chunk pools.
    Util::Deque<GpuMemoryInfo, GpaAllocator> m_availableGartGpuMem;
    Util::Deque<GpuMemoryInfo, GpaAllocator> m_busyGartGpuMem;
    Util::Deque<GpuMemoryInfo, GpaAllocator> m_availableLocalGpuMem;
    Util::Deque<GpuMemoryInfo, GpaAllocator> m_busyLocalGpuMem;
    Util::Deque<GpuMemoryInfo, GpaAllocator> m_availableInvisGpuMem;
    Util::Deque<GpuMemoryInfo, GpaAllocator> m_busyInvisGpuMem;

    struct SampleItem;
    class PerfSample;
    class CounterSample;
    class TraceSample;
    class TimingSample;
    class QuerySample;

    Util::Vector<SampleItem*, 16, GpaAllocator> m_sampleItemArray;
    PerfExpMemDeque* m_pAvailablePerfExpMem;

    // Unique pipelines registered with this GpaSession.
    Util::HashSet<Pal::uint64, GpaAllocator, Util::JenkinsHashFunc> m_registeredPipelines;
    // Unique API PSOs registered with this GpaSession.
    Util::HashSet<Pal::uint64, GpaAllocator, Util::JenkinsHashFunc> m_registeredApiHashes;

    // List of cached pipeline code object records that will be copied to the final database at the end of a trace
    Util::Deque<SqttCodeObjectDatabaseRecord*, GpaAllocator>  m_codeObjectRecordsCache;
    // List of pipeline code object records that were registered during a trace
    Util::Deque<SqttCodeObjectDatabaseRecord*, GpaAllocator>  m_curCodeObjectRecords;

    // List of cached code object load event records that will be copied to the final database at the end of a trace
    Util::Deque<CodeObjectLoadEventRecord, GpaAllocator>  m_codeObjectLoadEventRecordsCache;
    // List of code object load event records that were registered during a trace
    Util::Deque<CodeObjectLoadEventRecord, GpaAllocator>  m_curCodeObjectLoadEventRecords;

    // List of cached PSO correlation records that will be copied to the final database at the end of a trace
    Util::Deque<PsoCorrelationRecord, GpaAllocator>  m_psoCorrelationRecordsCache;
    // List of PSO correlation records that were registered during a trace
    Util::Deque<PsoCorrelationRecord, GpaAllocator>  m_curPsoCorrelationRecords;

    Util::RWLock m_registerPipelineLock;

    // Event type for timed queue events
    enum class TimedQueueEventType : Pal::uint32
    {
        Submit,
        Signal,
        Wait,
        Present,
        ExternalSignal,
        ExternalWait
    };

    // Struct that contains information about a specific timed queue event.
    struct TimedQueueEventItem
    {
        TimedQueueEventType eventType;      // Type of event
        Pal::uint64         cpuTimestamp;   // Time when the event was processed on the cpu
        Pal::uint64         apiId;          // The api specific id for the queue event
        Pal::uint32         sqttCmdBufId;   // The sqtt command buffer id value associated with a submit event
        Pal::uint32         submitSubIndex; // The sub index of an event within a submission event.
        Pal::uint32         queueIndex;     // The index of the associated queue in the m_timedQueuesArray
        Pal::uint64         frameIndex;     // The index of the current frame being rendered
        union
        {
            struct
            {
                GpuMemoryInfo memInfo[2];   // The gpu memory for the timestamps associated with the event
                Pal::gpusize  offsets[2];   // Memory offsets for the associated timestamp gpu memory
            } gpuTimestamps;

            Pal::uint64       cpuCompletionTimestamp; // The time when the event completed on the cpu
        };
    };

    // Struct for keeping track of timed operation on a specific queue
    struct TimedQueueState
    {
        Pal::IQueue*                                 pQueue;               // Pal Queue
        Pal::uint64                                  queueId;              // Api specific queue id
        Pal::uint64                                  queueContext;         // Api specific queue context
        Pal::QueueType                               queueType;            // Queue type
        Pal::EngineType                              engineType;           // Engine type
        bool                                         valid;                // Used to track if the queue is valid
        Util::Deque<Pal::ICmdBuffer*, GpaAllocator>* pAvailableCmdBuffers; // List of available cmdbuffers
        Util::Deque<Pal::ICmdBuffer*, GpaAllocator>* pBusyCmdBuffers;      // List of busy cmdbuffers
        Pal::IFence*                                 pFence;               // Used to track queue
                                                                           // operations
    };

    // Flags for the current session.
    GpaSessionFlags m_flags;

    // Array containing all of the queues registered for timing operations
    Util::Vector<TimedQueueState*, 8, GpaAllocator> m_timedQueuesArray;
    Util::RWLock m_timedQueuesArrayLock;

    // List of timed queue events for the current session
    Util::Vector<TimedQueueEventItem, 16, GpaAllocator> m_queueEvents;
    Util::Mutex m_queueEventsLock;

    // List of timestamp calibration samples
    Util::Vector<Pal::CalibratedTimestamps, 4, GpaAllocator> m_timestampCalibrations;

    // The most recent gpu clocks sample
    GpuClocksSample m_lastGpuClocksSample;

    // Internal command allocator used for timing command buffers
    Pal::ICmdAllocator* m_pCmdAllocator;

    // Finds the TimedQueueState associated with pQueue.
    Pal::Result FindTimedQueue(Pal::IQueue* pQueue,
                               TimedQueueState** ppQueueState,
                               Pal::uint32* pQueueIndex);

    // Finds the TimedQueueState associated with queueContext.
    Pal::Result FindTimedQueueByContext(Pal::uint64 queueContext,
                                        TimedQueueState** ppQueueState,
                                        Pal::uint32* pQueueIndex);

    /// Injects an external timed queue semaphore operation event
    Pal::Result ExternalTimedQueueSemaphoreOperation(Pal::uint64 queueContext,
                                                     Pal::uint64 cpuSubmissionTimestamp,
                                                     Pal::uint64 cpuCompletionTimestamp,
                                                     const TimedQueueSemaphoreInfo& timedSemaphoreInfo,
                                                     bool isSignalOperation);

    /// Converts a CPU timestamp to a GPU timestamp using a CalibratedTimestamps struct
    Pal::uint64 ConvertCpuTimestampToGpuTimestamp(Pal::uint64                      cpuTimestamp,
                                                  const Pal::CalibratedTimestamps& calibration) const;

    /// Extracts a GPU timestamp from a queue event
    Pal::uint64 ExtractGpuTimestampFromQueueEvent(const TimedQueueEventItem& queueEvent) const;

    // Creates a new command buffer for use on pQueue
    Pal::Result CreateCmdBufferForQueue(Pal::IQueue* pQueue,
                                        Pal::ICmdBuffer** ppCmdBuffer);

    // Acquires a command buffer from the TimedQueueState's command buffer pool
    Pal::Result AcquireTimedQueueCmdBuffer(TimedQueueState* pQueueState,
                                           Pal::ICmdBuffer** ppCmdBuffer);

    // Recycles busy command buffers in pQueueState
    Pal::Result RecycleTimedQueueCmdBuffers(TimedQueueState* pQueueState);

    // Preallocates a fixed number of command buffers for pQueueState and adds them to the command buffer pool
    Pal::Result PreallocateTimedQueueCmdBuffers(TimedQueueState* pQueueState,
                                                Pal::uint32 numCmdBuffers);

    // Resets all per session state in pQueueState
    Pal::Result ResetTimedQueueState(TimedQueueState* pQueueState);

    // Destroys the memory and resources for pQueueState
    void DestroyTimedQueueState(TimedQueueState* pQueueState);

    // Helper function to import one sample item from a source session to copy session.
    Pal::Result ImportSampleItem(const SampleItem* pSrcSampleItem);

    // Acquires a range of queue-owned GPU memory for use by the next command buffer submission.
    Pal::Result AcquireGpuMem(
        Pal::gpusize            size,
        Pal::gpusize            alignment,
        Pal::GpuHeap            heapType,
        Pal::GpuMemMallPolicy   mallPolicy,
        GpuMemoryInfo*          pGpuMem,
        Pal::gpusize*           pOffset);

    // Acquires a GpaSession-owned performance experiment based on the device's active perf counter requests.
    Pal::Result AcquirePerfExperiment(
        GpaSession::SampleItem* pSampleItem,
        const GpaSampleConfig&  sampleConfig,
        GpuMemoryInfo*          pGpuMem,
        Pal::gpusize*           pOffset,
        GpuMemoryInfo*          pSecondaryGpuMem,
        Pal::gpusize*           pSecondaryOffset,
        Pal::gpusize*           pHeapSize,
        Pal::IPerfExperiment**  ppExperiment);

    // Acquires a session-owned pipeline stats query.
    Pal::Result AcquirePipeStatsQuery(
        GpuMemoryInfo*          pGpuMem,
        Pal::gpusize*           pOffset,
        Pal::gpusize*           pHeapSize,
        Pal::IQueryPool**       ppQuery);

    // Dump SQ thread trace data in rgp format
    Pal::Result DumpRgpData(const GpaSampleConfig* pTraceConfig,
                            TraceSample*           pTraceSample,
                            void*                  pRgpOutput,
                            size_t*                pTraceSize) const;

    // Dumps the spm trace data in the buffer provided.
    Pal::Result AppendSpmTraceData(TraceSample*  pTraceSample,
                                   size_t        bufferSize,
                                   void*         pData,
                                   Pal::gpusize* pSizeInBytes) const;

    // Dumps the df spm trace data in the buffer provided.
    Pal::Result AppendDfSpmTraceData(TraceSample*  pTraceSample,
                                     size_t        bufferSize,
                                     void*         pData,
                                     Pal::gpusize* pSizeInBytes) const;

    Pal::Result AddCodeObjectLoadEvent(const Pal::IPipeline* pPipeline, CodeObjectLoadEventType eventType);
    Pal::Result AddCodeObjectLoadEvent(const Pal::IShaderLibrary* pLibrary, CodeObjectLoadEventType eventType);
    Pal::Result AddCodeObjectLoadEvent(const ElfBinaryInfo& elfBinaryInfo, CodeObjectLoadEventType eventType);

    // Recycle used Gart rafts and put back to available pool
    void RecycleGartGpuMem();

    // Recycle used Local rafts and put back to available pool
    void RecycleLocalGpuMem();

    // Recycle used Invisible rafts and put back to available pool
    void RecycleInvisGpuMem();

    // Destroy and free one sample item and its sub-items.
    void FreeSampleItem(GpaSession::SampleItem* pSampleItem);

    // Destroy and free the m_sampleItemArray and associated memory allocation
    void FreeSampleItemArray();

    // Destroy the sub-items in m_sampleItemArray but keep associated memory allocations.
    void RecycleSampleItemArray();

    // Helper function to destroy the GpuMemoryInfo object
    void DestroyGpuMemoryInfo(GpuMemoryInfo* pGpuMemoryInfo);

    PAL_DISALLOW_DEFAULT_CTOR(GpaSession);
    GpaSession& operator =(const GpaSession&);
};
} // GpuUtil
