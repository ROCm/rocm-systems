/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
 * @file  palTraceSession.h
 * @brief PAL GPU utility TraceSession class.
 ***********************************************************************************************************************
 */

#pragma once

#include "palPlatform.h"
#include "palDeque.h"
#include "palDevice.h"
#include "palGpuUtil.h"
#include "palHashMap.h"
#include "palMutex.h"
#include "palPipeline.h"
#include "palSysMemory.h"
#include "palGpuMemory.h"
#include "palMemTrackerImpl.h"
#include "palVector.h"

struct rdfStream;
struct rdfChunkFileWriter;

namespace DevDriver
{
class IStructuredWriter;
class IStructuredReader;
class StructuredValue;
}

namespace GpuUtil
{

class ITraceController;
class ITraceSource;

constexpr Pal::uint16 TextIdentifierSize = 16;

/// Information required to create a new chunk of trace data in a TraceSession
///
/// This data inside this structure is expected to be produced by trace source implementations. The specific fields
/// included within this structure are intended to support compatibility with the Radeon Data Format (RDF) spec.
struct TraceChunkInfo
{
    char        id[TextIdentifierSize]; ///<      Text identifier of the chunk
    Pal::uint32 version;                ///<      Version number of the chunk
    const void* pHeader;                ///< [in] Pointer to a buffer that contains the header data for the chunk
    Pal::int64  headerSize;             ///<      Size of the buffer pointed to by pHeader
    const void* pData;                  ///< [in] Pointer to a buffer that contains the data for the chunk
    Pal::int64  dataSize;               ///<      Size of the buffer pointed to by pData
    bool        enableCompression;      ///<      Indicates if the chunk's data should be compressed or not
};

/// The available states of TraceSession
enum class TraceSessionState : Pal::uint32
{
    Ready             = 0, ///< New trace ready to begin
    Requested         = 1, ///< A trace has been requested and awaiting acceptance
    Preparing         = 2, ///< Trace has been accepted and is preparing resources before beginning
    Running           = 3, ///< Trace is in progress
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 939
    Postamble         = 4, ///< The detailed frame trace has ended but its data has not yet been written
                           ///  into the session. Some trace sources may still collect data during this time.
    PostambleWaiting  = 5, ///< Waiting for Postamble to complete.
    Completed         = 6, ///< Trace has fully completed. RDF trace data is ready to be pulled out by CollectTrace().
    Count             = 7
#else
    Waiting           = 4, ///< Trace has ended, but data has not been written into the session
    Completed         = 5, ///< Trace has fully completed. RDF trace data is ready to be pulled out by CollectTrace().
    Count             = 6
#endif
};

/// Defines the type of payload. Currently only strings are supported but in the future can include JSON, structs, etc.
enum class TraceErrorPayload : Pal::uint32
{
    None,        //< Should be set when there is no additional information to be sent with the error
    ErrorString  //< Should be set when the error payload is string data
};

/// Chunk header for the error tracing chunk
struct TraceErrorHeader
{
    char              chunkId[TextIdentifierSize]; ///< Text identifier of the failing chunk
    Pal::uint32       chunkIndex;                  ///< Chunk index of the failing chunk
    Pal::Result       resultCode;                  ///< PAL Result code of the failure
    TraceErrorPayload payloadType;                 ///< Type of error chunk payload
};

constexpr char ErrorChunkTextIdentifier[TextIdentifierSize]  = "TraceError";
constexpr Pal::uint32 ErrorTraceChunkVersion                 = 1;

/**
***********************************************************************************************************************
* @interface ITraceController
* @brief Interface that allows for control of a trace operation through TraceSession.
*
* Trace controllers are responsible for driving the high-level steps of a trace operation. Users of this interface are
* expected to create their own implementation of this interface, register it with a TraceSession, then call the
* following TraceSession functions to drive the trace process:
*
* TraceSession::AcceptTrace
* TraceSession::BeginTrace
* TraceSession::EndTrace
* TraceSession::EndPostamble
* TraceSession::FinishTrace
***********************************************************************************************************************
*/
class ITraceController
{
public:
    /// Returns the name of the controller
    ///
    /// @returns the name of the controller as a null terminated string
    virtual const char* GetName() const = 0;

    /// Returns the version of the controller
    ///
    /// @returns the version of the controller as an unsigned integer value
    virtual Pal::uint32 GetVersion() const = 0;

    /// Called by the associated session to update the current trace configuration
    ///
    /// @param [in] pJsonConfig  Configuration data formatted as json and stored as DevDriver's StructuredValue object
    virtual void OnConfigUpdated(DevDriver::StructuredValue* pJsonConfig) = 0;

    /// Called by the associated session to notify the controller that a trace has been requested and it can take
    /// control of the TraceSession when desired.
    virtual Pal::Result OnTraceRequested() = 0;

    /// Called by the associated session to notify the controller that a trace has been canceled and it can start
    /// canceling the trace when ready.
    virtual Pal::Result OnTraceCanceled() = 0;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 908
    /// Called by TraceSession to indicate that GPU work is required on the indicated GPU during the preparation phase.
    /// The command buffer must be ready to record commands; however, the trace controller should not submit it
    /// until the trace begins.
    ///
    /// The controller MUST return a valid command buffer that is ready to record commands for the target GPU
    /// upon successful completion of this function via ppCmdBuf.
    ///
    /// This function will be called once per trace for each GPU that's considered relevant by the current set of
    /// trace sources.
    ///
    /// Note: This command buffer should be submitted at the same time as the command buffer provided in
    /// `OnBeginGpuWork`. They may be the same command buffer or separate; the goal is to allow trace sources
    /// to frontload recording GPU work before the trace formally begins.
    ///
    /// Note: The command buffer provided by this function does not need to be a new command buffer. It just needs
    ///       to be capable of recording new commands.
    ///
    /// @param [in]  gpuIndex   The index of the target GPU
    /// @param [out] ppCmdBuf   A command buffer that can be used to record GPU work before a trace starts executing.
    ///                         Note that this command buffer shouldn't be submitted until the trace begins.
    ///
    /// @returns Success if the command buffer was successfully returned
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorUnknown if an internal PAL error occurs.
    virtual Pal::Result OnPreparationGpuWork(Pal::uint32 gpuIndex, Pal::ICmdBuffer** ppCmdBuf) = 0;
#endif

    /// Called by TraceSession to indicate that GPU work is required to begin a trace on the indicated GPU
    ///
    /// The controller MUST return a valid command buffer that is ready to record commands for the target GPU
    /// upon successful completion of this function via ppCmdBuf.
    ///
    /// This function will be called once per trace for each GPU that's considered relevant by the current set of
    /// trace sources.
    ///
    /// Note: The command buffer provided by this function does not need to be a new command buffer. It just needs
    ///       to be capable of recording new commands.
    ///
    /// @param [in]  gpuIndex   The index of the target GPU
    /// @param [out] ppCmdBuf   A command buffer that can be used to perform any GPU work required to begin the trace
    ///
    /// @returns Success if the command buffer was successfully returned
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorUnknown if an internal PAL error occurs.
    virtual Pal::Result OnBeginGpuWork(Pal::uint32 gpuIndex, Pal::ICmdBuffer** ppCmdBuf) = 0;

    /// Called by TraceSession to indicate that GPU work is required to end a trace on the indicated GPU
    ///
    /// The controller MUST return a valid command buffer that is ready to record commands for the target GPU
    /// upon successful completion of this function via ppCmdBuf.
    ///
    /// This function will be called once per trace for each GPU that's considered relevant by the current set of
    /// trace sources.
    ///
    /// Note: The command buffer provided by this function does not need to be a new command buffer. It just needs
    ///       to be capable of recording new commands.
    ///
    /// @param [in]  gpuIndex   The index of the target GPU
    /// @param [out] ppCmdBuf   A command buffer that can be used to perform any GPU work required to end the trace
    ///
    /// @returns Success if the command buffer was successfully returned
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorUnknown if an internal PAL error occurs.
    virtual Pal::Result OnEndGpuWork(Pal::uint32 gpuIndex, Pal::ICmdBuffer** ppCmdBuf) = 0;

    /// Called by TraceSession to indicate that GPU work is required to end the postamble on the indicated GPU
    ///
    /// The controller MUST return a valid command buffer that is ready to record commands for the target GPU
    /// upon successful completion of this function via ppCmdBuf.
    ///
    /// This function will be called once per trace for each GPU that's considered relevant by the current set of
    /// trace sources.
    ///
    /// Note: The command buffer provided by this function does not need to be a new command buffer. It just needs
    ///       to be capable of recording new commands.
    ///
    /// @param [in]  gpuIndex   The index of the target GPU
    /// @param [out] ppCmdBuf   A command buffer that can be used to perform any GPU work required to end the postamble
    ///
    /// @returns Success if the command buffer was successfully returned
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorUnknown if an internal PAL error occurs.
    virtual Pal::Result OnEndPostambleGpuWork(
        Pal::uint32       gpuIndex,
        Pal::ICmdBuffer** ppCmdBuf) = 0;
};

/**
***********************************************************************************************************************
* @interface ITraceSource
* @brief Interface that enables developers to emit arbitrary data chunks into a trace through TraceSession.
*
* Trace sources are used to implement any surrounding logic required to produce a trace data chunk. Users of this
* interface are expected to create their own implementation of this interface, register it with a TraceSession, then
* call TraceSession::WriteDataChunk during a trace operation whenever a data chunk should be produced.
***********************************************************************************************************************
*/
class ITraceSource
{
public:
    /// Called by the associated session to update the current trace configuration
    ///
    /// @param [in] pJsonConfig  Configuration data formatted as json and stored as DevDriver's StructuredValue object
    virtual void OnConfigUpdated(DevDriver::StructuredValue* pJsonConfig) = 0;

    /// Returns a bitmask that represents which GPUs are relevant to this trace source
    ///
    /// If the bit at index N is set, GPU N must execute work on the GPU in order to produce trace data
    virtual Pal::uint64 QueryGpuWorkMask() const = 0;

    /// Called by the associated session to notify the source that a new trace has been accepted
    ///
    /// The source may use this notification to do any preparation work that might be required before the trace begins.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 908
    /// A command buffer is provided for the trace source to insert any work into. Note that the work will not be
    /// submitted until the trace begins (at the same time as `OnTraceBegin`). This allows for frontloading of
    /// expensive operations, such as the construction of a GpaSession sample, that would affect runtime speed
    /// or behavior during trace exeecution.
    ///
    /// @param [in] gpuIndex The index of the GPU that owns pCmdBuf
    /// @param [in] pCmdBuf  A command buffer that can be used to record any GPU work required during the
    ///                      preparation phase of the trace. Not submitted until `OnTraceBegin`.
    virtual void OnTraceAccepted(Pal::uint32 gpuIndex, Pal::ICmdBuffer* pCmdBuf) = 0;
#else
    virtual void OnTraceAccepted() = 0;
#endif

    /// Called by the associated session to notify the source that it should begin a trace
    ///
    /// The source should use the provided command buffer to execute any GPU work that's required for the source to
    /// begin a trace operation.
    ///
    /// In situations where multiple GPUs are present, this function will be called for all GPUs that are expected to
    /// participate in the trace. All GPUs that begin a trace are required to end it later. Sources are not expected
    /// to handle cases where the begin/end function calls are mismatched during a trace operation.
    ///
    /// @param [in] gpuIndex The index of the GPU that owns pCmdBuf
    /// @param [in] pCmdBuf  A command buffer that can be used to perform any GPU work required to begin the trace
    virtual void OnTraceBegin(Pal::uint32 gpuIndex, Pal::ICmdBuffer* pCmdBuf) = 0;

    /// Called by the associated session to notify the source that it should end the current trace
    ///
    /// The source should use the provided command buffer to execute any GPU work that's required for the source to
    /// end a trace operation.
    ///
    /// The command buffer associated with the OnTraceBegin function is not guaranteed to have finished GPU execution
    /// when this function is called. The command buffer associated with this function is also not guaranteed to finish
    /// execution until OnTraceFinished is called.
    ///
    /// In situations where multiple GPUs are present, this function will be called for all GPUs that are expected to
    /// participate in the trace.
    ///
    /// @param [in] gpuIndex The index of the GPU that owns pCmdBuf
    /// @param [in] pCmdBuf  A command buffer that can be used to perform any GPU work required to end the trace
    virtual void OnTraceEnd(Pal::uint32 gpuIndex, Pal::ICmdBuffer* pCmdBuf) = 0;

    /// Called by the associated session to notify the source that it should end the postamble
    ///
    /// The source should use the provided command buffer to execute any GPU work that's required for the source to
    /// end its postamble operation.
    ///
    /// The command buffer associated with the OnTraceBegin and OnTraceEnd functions are not guaranteed to have
    /// finished GPU execution when this function is called. The command buffer associated with this function is also
    /// not guaranteed to finish execution until OnTraceFinished is called.
    ///
    /// In situations where multiple GPUs are present, this function will be called for all GPUs that are expected to
    /// participate in the trace.
    ///
    /// @param [in] gpuIndex The index of the GPU that owns pCmdBuf
    /// @param [in] pCmdBuf  A command buffer that can be used to perform any GPU work required to end the postamble
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 939
    virtual void OnPostambleEnd(
        Pal::uint32      gpuIndex,
        Pal::ICmdBuffer* pCmdBuf) = 0;
#endif

    /// Called by the associated session to notify the source that the current trace has finished
    ///
    /// When this function is called, all prior command buffers provided to the source during the trace operation have
    /// finished execution. The source should use this function to collect any data generated by the GPU and emit it
    /// via TraceSession::WriteDataChunk.
    virtual void OnTraceFinished() = 0;

    /// Returns the name of the source
    ///
    /// @returns the name of the source as a null terminated string
    virtual const char* GetName() const = 0;

    /// Returns the version of the source
    ///
    /// @returns the version of the source as an unsigned integer value
    virtual Pal::uint32 GetVersion() const = 0;

    /// Whether multiple instances of the trace source are allowed
    ///
    /// @returns true if multiple instances of this trace sources can co-exist in one session, false otherwise.
    virtual bool AllowMultipleInstances() const { return false; }
};

/**
***********************************************************************************************************************
* @class TraceSession
* @brief Helper class providing common driver functionality for collecting arbitrary data traces.
*
* Due to the global nature of the trace functionality, only one TraceSession is typically used at a time.
* An interface to acquire a session exists on IPlatform. Users who need to interact with an instance of this object
* should expect to acquire it there.
*
* @see IPlatform::GetTraceSession()
***********************************************************************************************************************
*/
class TraceSession final
{
public:
    /// Constructor.
    ///
    /// @param [in] pPlatform Platform associated with this TraceSesion
    TraceSession(Pal::IPlatform* pPlatform);

    /// Destructor
    ~TraceSession();

    /// Initialize the trace session before requesting a trace.
    ///
    /// @returns Success if initalization was successful, or ErrorUnknown upon failure.
    Pal::Result Init();

    /// Returns whether tracing has been formally enabled via UberTrace or not.
    /// If 'true', this means that tool-side applications have requested this
    /// TraceSession to capture traces. This has implications for PAL clients.
    ///
    /// @returns True if tracing has been enabled, and false otherwise.
    bool IsTracingEnabled() const { return m_tracingEnabled; }

    /// Attempts to update the current trace configuration
    ///
    /// This function will only succeed if there is currently to trace in progress
    ///
    /// TODO: The JSON configuration interface will likely be replaced with driver settings in the future
    ///
    /// @param [in] pData    Buffer that stores the Json-formatted configuration data
    /// @param [in] dataSize Configuration data-size
    ///
    /// @returns Success if the trace configuration was successfully updated.
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorUnknown if an internal PAL error occurs.
    ///          + ErrorUnavailable if a trace is currently in progress
    ///          + ErrorInvalidPointer pData is nullptr
    ///          + ErrorInvalidParameter pData is not valid json
    Pal::Result UpdateTraceConfig(const void* pData, size_t dataSize);

    /// Attempts to request a new trace operation on the trace session.
    ///
    /// Once a trace is successfully requested, it will become available for a registered trace controller to accept.
    /// When a controller accepts the trace, it becomes responsible for managing the rest of the trace operation and
    /// notifying the session upon trace completion.
    ///
    /// Since the session can only run a single trace at a time, this function will not succeed if another trace is
    /// is already requested or in progress.
    ///
    /// @returns Success if the trace operation was successfully requested.
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorUnknown if an internal PAL error occurs.
    ///          + ErrorUnavailable if there is a trace in progress already and a new one cannot be started
    Pal::Result RequestTrace();

    /// Cancels a trace currently in progress.
    ///
    /// @returns Success if the trace was successfully canceled.
    ///          Otherwise, one of the following errors may be returned:
    ///          + NotReady if the trace is not ready to be canceled.
    ///          + ErrorUnknown if an internal PAL error occurs.
    Pal::Result CancelTrace();

    /// Cleans up the RDF chunk stream and makes it ready for a new trace again.
    ///
    /// @returns Success if the trace session and rdf streams were successfully cleaned up and returned to the
    ///          initialization state
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorUnknown if an internal PAL error occurs.
    Pal::Result CleanupChunkStream();

    /// Attempts to consume any trace data stored within the trace session.
    ///
    /// This function will only successfully return trace data after a trace operation is completed on the session.
    ///
    /// TODO: This function should be replaced with one that uses a callback so we can avoid needing to store the trace
    ///       data into memory twice.
    ///
    /// @param [out]    pData     (Optional) Destination buffer to copy the trace data into
    ///                                      If this parameter is nullptr, the size of the trace data in bytes will be
    ///                                      returned via pDataSize instead of consuming any trace data.
    /// @param [in/out] pDataSize            If pData is nullptr, then this parameter is used to return the trace data
    ///                                      size in bytes.
    ///                                      If pData is valid, this parameter represents the size of the buffer
    ///                                      pointed to by pData.
    ///
    /// @returns Success if the trace data was successfully consumed or the size of the trace data was returned.
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorUnknown if an internal error occurs in PAL or an unknown error is thrown by external library
    ///          + ErrorUnavailable if trace data is not available for collection at this time
    ///          + ErrorInvalidPointer if nullptr is passed as pDataSize
    ///          + ErrorInvalidMemorySize if *pDataSize indicates that pData is too small to contain the trace data
    Pal::Result CollectTrace(void* pData, size_t* pDataSize);

    /// Attempts to register a trace controller
    ///
    /// Once registered, trace controllers can receive configuration updates from the session.
    /// They may also manage the trace operation by calling AcceptTrace, BeginTrace, EndTrace, EndPostamble and FinishTrace.
    ///
    /// Trace controllers can only be registered when there is no trace in progress
    ///
    /// @param [in] pController The trace controller to register with the session
    ///
    /// @returns Success if the controller was successfully registered.
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorUnknown if an internal PAL error occurs.
    ///          + AlreadyExists if this controller has already been registered
    ///          + ErrorUnavailable if a trace is in progress
    ///          + ErrorInvalidPointer if nullptr is passed as pController
    Pal::Result RegisterController(ITraceController* pController);

    /// Attempts to unregister a previously registered trace controller
    ///
    /// @param [in] pController The trace controller to unregister from the session
    ///
    /// @returns Success if the controller was successfully unregistered.
    ///          Otherwise, one of the following errors may be returned:
    ///          + NotFound if the provided controller was not previously registered
    ///          + ErrorUnknown if an internal PAL error occurs.
    ///          + ErrorUnavailable if a trace is in progress
    Pal::Result UnregisterController(ITraceController* pController);

    /// Attempts to register a trace source
    ///
    /// Once registered, trace sources can receive configuration updates from the session.
    /// They may also emit data during trace operations by calling WriteDataChunk.
    ///
    /// Trace sources can only be registered when there is no trace in progress
    ///
    /// @param [in] pSource The trace source to register with the session
    ///
    /// @returns Success if the source was successfully registered.
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorUnknown if an internal PAL error occurs.
    ///          + AlreadyExists if this source has already been registered
    ///          + ErrorUnavailable if a trace is in progress
    ///          + ErrorInvalidPointer if nullptr is passed as pSource
    Pal::Result RegisterSource(ITraceSource* pSource);

    /// Attempts to unregister a previously registered trace source
    ///
    /// @param [in] pSource The trace source to unregister from the session
    ///
    /// @returns Success if the source was successfully unregistered.
    ///          Otherwise, one of the following errors may be returned:
    ///          + NotFound if the provided source was not previously registered
    ///          + ErrorUnknown if an internal PAL error occurs.
    ///          + ErrorUnavailable if a trace is in progress
    Pal::Result UnregisterSource(ITraceSource* pSource);

    /// Attempts to accept a previously requested trace with the provided controller
    ///
    /// Once a trace is successfully accepted by a controller, that controller becomes responsible for managing the
    /// rest of the trace operation. Also, once a requested trace is accepted by a controller, no other controllers
    /// will be able to accept that trace. Accept is a "consuming" operation.
    ///
    /// @param [in] pController      The trace controller to accept the trace with
    /// @param [in] supportedGpuMask Bit mask of GPU indices that are capable of participating in the trace
    ///
    /// The GPU mask provided to this function is used to determine which GPUs will be involved in the trace. In order
    /// to decide which GPUs require GPU work, the session creates a combined mask from all registered sources and
    /// checks it against the mask provided by this function. Only GPUs that are present in both masks will be able to
    /// submit GPU work during the trace.
    ///
    /// @returns Success if the trace was successfully accepted.
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorUnknown if an internal PAL error occurs.
    ///          + ErrorUnavailable if no trace has been requested or a trace is currently in progress
    ///          + ErrorInvalidPointer if nullptr is passed as pController
    Pal::Result AcceptTrace(ITraceController* pController, Pal::uint64 supportedGpuMask);

    /// Begins the trace that was previously accepted by the provided controller
    ///
    /// This function MUST be called after a successful call to AcceptTrace. When this function is called, the session
    /// will communicate with all registered trace sources and instruct them to begin the trace operation. The provided
    /// trace controller will be notified if any GPU work is required via ITraceController::OnBeginGpuWork. The command
    /// buffers returned by OnBeginGpuWork will be passed to each relevant trace source to record required work.
    ///
    /// The command buffers generated in response to this this call MUST be submitted BEFORE the command buffers
    /// generated in response to the EndTrace call!
    ///
    /// In situations where multiple GPUs are present, the OnBeginGpuWork function will be called once per GPU index,
    /// for all GPUs that are relevant for the current trace sources.
    ///
    /// @returns Success if the trace was successfully started.
    ///          Otherwise, the error generated by OnBeginGpuWork will be returned.
    Pal::Result BeginTrace();

    /// Ends the trace that was previously started by the provided controller
    ///
    /// This function MUST be called after BeginTrace. When this function is called, the session will communicate with
    /// all registered trace sources and instruct them to end the trace operation. The provided trace controller will
    /// trace controller will be notified if any GPU work is required via ITraceController::OnEndGpuWork. The command
    /// buffers returned by OnEndGpuWork will be passed to each relevant trace source to record required work.
    ///
    /// The command buffers generated in response to this this call MUST be submitted AFTER the command buffers
    /// generated in response to the previous BeginTrace call! The generated command buffers MUST also complete
    /// execution on the GPU BEFORE FinishTrace is called!
    ///
    /// In situations where multiple GPUs are present, the OnEndGpuWork function will be called once per GPU index
    /// for all GPUs that are relevant for the current trace sources.
    ///
    /// The Trace Session will enter Postamble phase after EndTrace is called.
    ///
    /// @returns Success if the trace was successfully ended.
    ///          Otherwise, the error generated by OnEndGpuWork will be returned.
    Pal::Result EndTrace();

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 939
    /// Ends the postamble phase, which typically runs until the detailed trace data is available.
    /// This function MUST be called after EndTrace. When this function is called, the session will communicate with
    /// all registered trace sources and notify them of the end of the postamble phase. The provided trace controller
    /// will be notified if any GPU work is required via ITraceController::OnEndPostambleGpuWork. The command
    /// buffers returned by OnEndPostambleGpuWork will be passed to each relevant trace source to record required work.
    ///
    /// The command buffers generated in response to this this call MUST be submitted AFTER the command buffers
    /// generated in response to the previous EndTrace call! The generated command buffers MUST also complete
    /// execution on the GPU BEFORE FinishPostamble is called!
    ///
    /// In situations where multiple GPUs are present, the OnEndPostambleGpuWork function will be called once per GPU index
    /// for all GPUs that are relevant for the current trace sources.
    ///
    /// @returns Success if the trace was successfully ended.
    ///          Otherwise, the error generated by OnEndPostambleGpuWork will be returned.
    Pal::Result EndPostamble();
#endif

    /// Notifies the session that the trace operation started by the provided controller has finished.
    ///
    /// This function MUST be called after EndPostamble. When this function is called, the session will communicate with
    /// all registered trace sources and notify them that all GPU work is complete. This notification is typically
    /// used by sources to retrieve data produced by the GPU and write it into the session's trace data.
    void FinishTrace();

    /// Writes a chunk of trace data into the session.
    ///
    /// Trace sources are expected to call this function whenever they produce a new data chunk that should be added
    /// into the session's trace data.
    ///
    /// This function may ONLY be called AFTER the BeginTrace function returns and BEFORE the FinishTrace call returns!
    ///
    /// @param [in] pSource  The trace source that generated the provided data chunk
    /// @param [in] info     Information about the provided chunk that will be written into the trace data
    ///
    /// @returns Success if the incoming data chunk was successfully written/appended into the current data stream.
    ///          Otherwise, one of the following errors may be returned:
    ///          + ErrorUnknown if an internal error occurs in PAL or an unknown error is thrown by external library
    Pal::Result WriteDataChunk(ITraceSource* pSource, const TraceChunkInfo& info);

    /// Returns the current TraceSession state
    ///
    /// @returns Enum value of the current TraceSessionState
    TraceSessionState GetTraceSessionState() const
    {
        return m_sessionState;
    }

    /// Sets the TraceSession state based on external operations
    ///
    /// @param [in] sessionState TraceSessionState value to be assigned as the current state
    void SetTraceSessionState(TraceSessionState sessionState)
    {
        m_sessionState = sessionState;
    }

    /// Returns the current active controller
    ///
    /// @returns Pointer to the current active controller driving the TraceSession
    ITraceController* GetActiveController() const
    {
        return m_pActiveController;
    }

    /// Reports an error encountered during an active trace by inserting a "TraceError" chunk to the trace stream
    ///
    /// If, during a trace or the construction of an RDF chunk, an error is encountered and a chunk that was
    /// expected to be written can no longer be, this function may be called to insert an error chunk in place
    /// of the expected chunk.
    ///
    /// @param [in] chunkId      Text identifier of the failed RDF chunk
    /// @param [in] pPayload     Pointer to the data sent for the error
    ///                          If the payloadType is a string, the string must be null-terminated
    /// @param [in] payloadSize  Size of the data in the payload
    /// @param [in] payloadType  Type of payload data represented by `pPayload`
    /// @param [in] errorResult  The PAL result code of the encountered error
    ///
    /// @returns Success if the error chunk was written successfully
    Pal::Result ReportError(
        const char        chunkId[TextIdentifierSize],
        const void*       pPayload,
        Pal::uint64       payloadSize,
        TraceErrorPayload payloadType,
        Pal::Result       errorResult);

    /// Explicitly activates this TraceSession for managing traces.
    ///
    /// This should be called during Platform Init in response to a tool-side request to enable UberTrace tracing.
    /// This signals that an active connection has been made to tool-side applications and that profiling via
    /// PAL Trace should be prioritized in client drivers.
    void EnableTracing()
    {
        m_tracingEnabled = true;
    }

    /// Returns a pointer to a byte array containing the trace configuration.
    ///
    /// @param [out] pTraceConfigSize  Sets *pTraceConfigSize to the number of bytes in the trace config
    ///
    /// @returns A pointer to the trace configuration data
    const void* GetTraceConfig(size_t* pTraceConfigSize) const
    {
        PAL_ASSERT(pTraceConfigSize != nullptr);
        (*pTraceConfigSize) = m_configDataSize;
        return m_pConfigData;
    }

    /// Indicates if a cancel-trace signal has been received and that a cancelation is in progress.
    ///
    /// @return true if a cancelation is in progress.
    bool IsCancelingTrace() const { return m_cancelingTrace; }

private:
    typedef Pal::IPlatform TraceAllocator;

    Pal::IPlatform* const         m_pPlatform; // Platform associated with this TraceSesion
    DevDriver::IStructuredReader* m_pReader;   // Stores the current JSON-based config of the TraceSession

    // RW Locks for trace sources, controllers, and RDF streams
    Util::RWLock                  m_registerTraceSourceLock;
    Util::RWLock                  m_registerTraceControllerLock;
    Util::RWLock                  m_chunkAppendLock;

    // Trace sources registered with this TraceSession.
    using TraceSourcesVec = Util::Vector<ITraceSource*, 16, TraceAllocator>;
    TraceSourcesVec m_registeredTraceSources;

    // TraceSources and corresponding configs
    typedef Util::HashMap <const char*,
                           DevDriver::StructuredValue*,
                           TraceAllocator,
                           Util::StringJenkinsHashFunc,
                           Util::StringEqualFunc> TraceSourcesConfigMap;
    TraceSourcesConfigMap m_traceSourcesConfigs;

    // Unique trace controllers registered with this TraceSession.
    typedef Util::HashMap <const char*,
                           ITraceController*,
                           TraceAllocator,
                           Util::StringJenkinsHashFunc,
                           Util::StringEqualFunc> TraceControllersMap;
    TraceControllersMap m_registeredTraceControllers;

    ITraceController*   m_pActiveController; // The controller currently driving the TraceSession.
                                             // We can have only one active controller at a time.
    TraceSessionState   m_sessionState;      // Current state of the TraceSession
    rdfChunkFileWriter* m_pChunkFileWriter;  // Helper struct that manages create chunk file streams
                                             // and write data chunks
    rdfStream*          m_pCurrentStream;    // Active RDF stream for writing chunks
    Pal::int32          m_currentChunkIndex; // The current chunk index of the RDF stream
    bool                m_tracingEnabled;    // Flag indicating UberTrace tracing is enabled tool-side
    void*               m_pConfigData;       // Buffer containing the cached trace configurationn
    size_t              m_configDataSize;    // Size of the cached trace config buffer
    bool                m_cancelingTrace;    // Indicates that a cancel signal has been received and trace cancelation
                                             // is in progress.
};
} // GpuUtil
