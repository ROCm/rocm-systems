/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#include "palGpuUtil.h"
#include "palTraceSession.h"
#include "palGpaSession.h"

#include <atomic>

struct SqttQueueEventRecord;
struct SqttQueueInfoRecord;

namespace Pal
{
class Platform;
}

namespace GpuUtil
{
namespace TraceChunk
{

/// "QueueInfo" RDF chunk identifier & version
constexpr char        QueueInfoChunkId[TextIdentifierSize] = "QueueInfo";
constexpr Pal::uint32 QueueInfoChunkVersion                = 1;

/// Enum describing logical queue types
enum class QueueType : Pal::uint8
{
    Unknown        = 0,
    Universal      = 1,
    Compute        = 2,
    Dma            = 3,
    Encode         = 4,
    Decode         = 5,
    Security       = 6,
    VideoProcessor = 7
};

/// Enum describing hardware engine types
enum class HwEngineType : Pal::uint8
{
    Unknown               = 0,
    Universal             = 1,
    Compute               = 2,
    ExclusiveCompute      = 3,
    Dma                   = 4,
    Decode                = 5,
    Encode                = 6,
    HighPriorityUniversal = 7,
    HighPriorityGraphics  = 8,
    Security              = 9,
    Vpe                   = 10
};

/// Structure describing a queue's properties
struct QueueInfo
{
    Pal::uint32  pciId;        ///< The ID of the GPU queried
    Pal::uint64  queueId;      ///< API-specific queue ID
    Pal::uint64  queueContext; ///< OS-level queue context value from Windows KMD to correlate with ETW data.
                               ///  Only applicable to D3D on Windows; 0 otherwise.
    QueueType    queueType;    ///< The logical queue type
    HwEngineType engineType;   ///< The hardware engine that the queue is mapped to
};

// ------------------------------------------------------------------------------------------- //

/// "QueueEvent" RDF chunk identifier & version
constexpr char        QueueEventChunkId[TextIdentifierSize] = "QueueEvent";
constexpr Pal::uint32 QueueEventChunkVersion                = 1;

/// The type of queue-level timings event
enum class QueueEventType : Pal::uint32
{
    CmdBufSubmit    = 0,
    SignalSemaphore = 1,
    WaitSemaphore   = 2,
    Present         = 3
};

/// Structure describing a queue-level timings event
struct QueueEvent
{
    Pal::uint32    pciId;          ///< The ID of the GPU queried
    Pal::uint64    queueId;        ///< The API-specific queue ID which triggered the event
    QueueEventType eventType;      ///< The type of the queue-timing event
    Pal::uint32    sqttCmdBufId;   ///< [`CmdBufSubmit` only; 0 otherwise]
                                   ///  SQTT command buffer ID matching CmdBufStart user data marker
    Pal::uint64    frameIndex;     ///< [`CmdBufSubmit` & `Present` only; 0 otherwise]
                                   ///  Global frame index incremented for each "Present" call
    Pal::uint32    submitSubIndex; ///< [`CmdBufSubmit` only; 0 otherwise]
                                   ///  Sub-index of event within submission.
                                   ///  When there is only one CmdBuffer per submission, `submitSubIndex` is 0.
                                   ///  When there are multiple command buffers per submission, `submitSubIndex`
                                   ///  is incremented by one for each command buffer within the submission.
    Pal::uint64    apiEventId;     ///< [`CmdBufSubmit`] API-specific command buffer ID signaled
                                   ///  [`SignalSemaphore`] API-specific semaphore ID signaled
                                   ///  [`WaitSemaphore`] API-specific semaphore ID waited on
                                   ///  [`Present`] N/A (set to 0)
    Pal::uint64    cpuTimestamp;   ///< CPU start timestamp of when this event is triggered in clock cycle units
    Pal::uint64    gpuTimestamp1;  ///< [`CmdBufSubmit`] GPU timestamp when the HW execution of command buffer began
                                   ///  [`SignalSemaphore`] GPU timestamp when the HW signaled the queue semaphore
                                   ///  [`WaitSemaphore`] GPU timestamp when HW finished waiting on the semaphore
                                   ///  [`Present`] GPU timestamp when HW processed the Present call
                                   ///
                                   ///  All timestamps are expressed in clock cycle units.
    Pal::uint64    gpuTimestamp2;  ///< [`CmdBufSubmit` only; 0 otherwise]
                                   ///  GPU timestamp when the HW execution of command buffer finished
};

} // namespace TraceChunk

// QueueTimings Trace Source name & version
constexpr char        QueueTimingsTraceSourceName[]  = "queuetimings";
constexpr Pal::uint32 QueueTimingsTraceSourceVersion = 2;

// =====================================================================================================================
// This trace source captures queue timings data through GPA session & produces "QueueInfo" and "QueueEvent" RDF chunks
class QueueTimingsTraceSource : public ITraceSource
{
public:
    explicit QueueTimingsTraceSource(Pal::IPlatform* pPlatform);
    virtual ~QueueTimingsTraceSource();

    // ==== TraceSource Native Functions ========================================================================== //
    Pal::Result Init(Pal::IDevice* pDevice);

    Pal::Result RegisterTimedQueue(Pal::IQueue* pQueue,
                                   Pal::uint64 queueId,
                                   Pal::uint64 queueContext);

    Pal::Result UnregisterTimedQueue(Pal::IQueue* pQueue);

    Pal::Result TimedSubmit(Pal::IQueue*                pQueue,
                            const Pal::MultiSubmitInfo& submitInfo,
                            const TimedSubmitInfo&      timedSubmitInfo);

    Pal::Result TimedSignalQueueSemaphore(Pal::IQueue* pQueue,
                                          Pal::IQueueSemaphore* pQueueSemaphore,
                                          const TimedQueueSemaphoreInfo& timedSignalInfo,
                                          Pal::uint64  value = 0);

    Pal::Result TimedWaitQueueSemaphore(Pal::IQueue* pQueue,
                                        Pal::IQueueSemaphore* pQueueSemaphore,
                                        const TimedQueueSemaphoreInfo& timedWaitInfo,
                                        Pal::uint64  value = 0);

    Pal::Result TimedQueuePresent(Pal::IQueue*                 pQueue,
                                  const TimedQueuePresentInfo& timedPresentInfo);

    Pal::Result ExternalTimedWaitQueueSemaphore(Pal::uint64 queueContext,
                                                Pal::uint64 cpuSubmissionTimestamp,
                                                Pal::uint64 cpuCompletionTimestamp,
                                                const TimedQueueSemaphoreInfo& timedWaitInfo);

    Pal::Result ExternalTimedSignalQueueSemaphore(Pal::uint64 queueContext,
                                                  Pal::uint64 cpuSubmissionTimestamp,
                                                  Pal::uint64 cpuCompletionTimestamp,
                                                  const TimedQueueSemaphoreInfo& timedSignalInfo);

    bool IsTimingInProgress() const;

    // ==== Base Class Overrides =================================================================================== //
    virtual void OnConfigUpdated(DevDriver::StructuredValue* pJsonConfig) override { };

    virtual Pal::uint64 QueryGpuWorkMask() const override { return 0; }

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 908
    virtual void OnTraceAccepted(Pal::uint32 gpuIndex, Pal::ICmdBuffer* pCmdBuf) override;
#else
    virtual void OnTraceAccepted() override;
#endif
    virtual void OnTraceBegin(Pal::uint32 gpuIndex, Pal::ICmdBuffer* pCmdBuf) override { };
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 939
    virtual void OnPostambleEnd(
        Pal::uint32      gpuIndex,
        Pal::ICmdBuffer* pCmdBuf) override;
    virtual void OnTraceEnd(
        Pal::uint32      gpuIndex,
        Pal::ICmdBuffer* pCmdBuf) override {};
#else
    virtual void OnTraceEnd(
        Pal::uint32      gpuIndex,
        Pal::ICmdBuffer* pCmdBuf) override;
#endif
    virtual void OnTraceFinished() override;

    virtual const char* GetName() const override { return QueueTimingsTraceSourceName; }
    virtual Pal::uint32 GetVersion() const override { return QueueTimingsTraceSourceVersion; }

private:
    void WriteQueueInfoChunks(
        const SqttQueueInfoRecord* pQueueInfoRecords,
        size_t                     numQueueInfoRecords);

    void WriteQueueEventChunks(
        const SqttQueueInfoRecord*  pQueueInfoRecords,
        size_t                      numQueueInfoRecords,
        const SqttQueueEventRecord* pQueueEventRecords,
        size_t                      numQueueEventRecords);

    void ReportInternalError(const char* pErrorMsg, Pal::Result result);

    Pal::IPlatform* const m_pPlatform;        // IPlatform owning the parent TraceSession
    GpaSession*           m_pGpaSession;      // Handle to GpaSession object for tracking queue timings
    bool                  m_traceIsHealthy;   // Internal flag for tracking resource and state health
    std::atomic<bool>     m_timingInProgress; // Flag for tracking if queue timings operations are ongoing

};

} // namespace GpuUtil
