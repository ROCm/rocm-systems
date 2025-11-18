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

#include "palTraceSession.h"

namespace Pal
{
class IPlatform;
class IQueue;
class ICmdBuffer;
class Device;
}

namespace GpuUtil
{

/// Supported render operations used to advance the trace
enum RenderOp : Pal::uint8
{
    RenderOpDraw     = (1u << 0),
    RenderOpDispatch = (1u << 1)
};

/// Structure used to batch submit render operations on queue submission
/// This struct should have a `*Count` field for each @ref RenderOp enumeration above
struct RenderOpCounts
{
    Pal::uint32 drawCount;
    Pal::uint32 dispatchCount;
};

constexpr Pal::uint32 RenderOpTraceControllerVersion = 4;
constexpr char        RenderOpTraceControllerName[]  = "renderop";

// =====================================================================================================================
class RenderOpTraceController : public ITraceController
{
public:
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 896
    using RenderOp = GpuUtil::RenderOp;
#endif
    RenderOpTraceController(Pal::IPlatform* pPlatform, Pal::IDevice* pDevice);
    virtual ~RenderOpTraceController();

    virtual const char* GetName() const override { return RenderOpTraceControllerName; }
    virtual Pal::uint32 GetVersion() const override { return RenderOpTraceControllerVersion; }

    virtual void OnConfigUpdated(DevDriver::StructuredValue* pJsonConfig) override;
    virtual Pal::Result OnTraceRequested() override;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 908
    virtual Pal::Result OnPreparationGpuWork(Pal::uint32 gpuIndex, Pal::ICmdBuffer** ppCmdBuf) override;
#endif
    virtual Pal::Result OnBeginGpuWork(Pal::uint32 gpuIndex, Pal::ICmdBuffer** ppCmdBuffer) override;
    virtual Pal::Result OnEndGpuWork(Pal::uint32 gpuIndex, Pal::ICmdBuffer** ppCmdBuffer) override;
    virtual Pal::Result OnEndPostambleGpuWork(
        Pal::uint32       gpuIndex,
        Pal::ICmdBuffer** ppCmdBuffer) override;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 896
    void RecordRenderOp(Pal::IQueue* pQueue, RenderOp renderOp);
#endif

    void FinishTrace();

    // Cancel the trace currently in progress.
    virtual Pal::Result OnTraceCanceled() override;

    /// This function must be called by client drivers implementing the RenderOp controller.
    /// On every queue submission, this function is called with the cumulative counts of render operations
    /// recorded into that queue's command buffers.
    /// Based on the controller's internal mask, set by the user during trace configuration,
    /// the trace controller may advance its state.
    void RecordRenderOps(Pal::IQueue* pQueue, const RenderOpCounts& renderOpCounts);

private:
    /// Controls whether the trace proceeds on absolute render op counts or relative
    enum class CaptureMode : Pal::uint8
    {
        Relative = 0, ///< Relative to when the trace request is received
        Absolute      ///< Absolute render op index
    };

    Pal::Result AcceptTrace();
    Pal::Result BeginTrace();

    Pal::Result SubmitBeginTraceGpuWork() const;
    Pal::Result SubmitEndTraceGpuWork();
    Pal::Result SubmitEndPostambleGpuWork();

    Pal::Result WaitForTraceEndGpuWorkCompletion() const;
    Pal::Result CreateFence(Pal::IFence** ppFence) const;
    Pal::Result CreateCommandBuffer(bool traceEnd, Pal::ICmdBuffer** ppCmdBuf) const;
    Pal::Result CreateCmdAllocator();

    void OnRenderOpUpdated(Pal::uint64 countRecorded);
    void FreeResources();
    void AbortTrace();

    Pal::IPlatform* const m_pPlatform;             // Platform associated with this TraceController
    Pal::IDevice*         m_pDevice;               // Device associated with this TraceController
    Pal::ICmdAllocator*   m_pCmdAllocator;         // Command allocator for the TraceController

    TraceSession*         m_pTraceSession;         // TraceSession owning this TraceController
    Pal::uint64           m_supportedGpuMask;      // Bit mask of GPU indices that are capable of participating in the trace
    Pal::uint8            m_renderOpMask;          // Bitmask of RenderOp modes, indicating which are accepted
    CaptureMode           m_captureMode;           // Modality for determining the starting renderop index of the trace
    Pal::uint64           m_renderOpCount;         // The "global" count, incremented on every render op
    Pal::uint64           m_prepStartRenderOp;     // Relative or absolute render op number indicating trace begin
    Pal::uint64           m_numPrepRenderOps;      // Number of "warm-up" frames before the start frame
    Pal::uint64           m_captureRenderOpCount;  // Number of frames to wait before ending the trace
    Pal::uint64           m_renderOpTraceAccepted; // The frame number when the trace was accepted

    Util::Mutex           m_renderOpLock;          // Lock over UpdateFrame/OnFrameUpdated
    Pal::IQueue*          m_pQueue;                // The queue being used to submit Begin/End GPU trace command buffers
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 908
    Pal::ICmdBuffer*      m_pCmdBufTracePrepare;   // Command buffer for recording during the prep phase
#endif
    Pal::ICmdBuffer*      m_pCmdBufTraceBegin;     // Command buffer to submit Trace Begin
    Pal::ICmdBuffer*      m_pCmdBufTraceEnd;       // Command buffer to submit Trace End
    Pal::ICmdBuffer*      m_pCmdBufPostambleEnd;   // Command buffer to submit Postamble End
    Pal::IFence*          m_pFenceTraceEnd;        // Fence to wait for Trace End command buffer completion
    Pal::IFence*          m_pFencePostambleEnd;    // Fence to wait for Postamble End command buffer completion
};

} // namespace GpuUtil
