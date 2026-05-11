/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ROCm-native TraceSession — a stripped-down port of GpuUtil::TraceSession from PAL.
 * Manages trace state, RDF chunk-file serialisation, and a set of pluggable
 * IRocTraceController / IRocTraceSource objects.
 *
 * PAL equivalent: pal/inc/gpuUtil/palTraceSession.h
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

// Forward-declare opaque RDF types so consumers of this header do not need
// amdrdf.h on their include path.  The full definition is only needed in
// roctracesession.cpp which includes amdrdf.h directly.
struct rdfStream;
struct rdfChunkFileWriter;

// RDF chunk identifier size (matches RDF_IDENTIFIER_SIZE in amdrdf.h)
static constexpr int kRdfIdentifierSize = 16;

namespace roc {

// ================================================================================================
// Mirror of GpuUtil::TraceSessionState (palTraceSession.h)
enum class RocTraceSessionState : uint32_t {
  Ready     = 0,  ///< No trace in flight; ready for a new request
  Requested = 1,  ///< Tool called RequestTrace(); awaiting controller acceptance
  Preparing = 2,  ///< Controller accepted; preparation dispatches in flight
  Beginning = 3,  ///< Begin GPU commands being submitted
  Running   = 4,  ///< Detailed trace capture in progress
  Completed = 5,  ///< Trace finished; RDF data ready for CollectTrace()
};

// ================================================================================================
// Forward declarations
class RocTraceSession;

// ================================================================================================
// IRocTraceController — mirrors GpuUtil::ITraceController.
// One controller drives the trace lifecycle (accept, begin, end, finish).
class IRocTraceController {
public:
  virtual ~IRocTraceController() = default;
  virtual const char* GetName()    const = 0;
  virtual uint32_t    GetVersion() const = 0;

  /// Called when a trace is requested; controller calls session->AcceptTrace() to accept.
  virtual bool OnTraceRequested(RocTraceSession* pSession) = 0;
  /// Called when trace is cancelled before it starts.
  virtual void OnTraceCanceled() = 0;
  /// Called when all GPU work is complete and sources have emitted their chunks.
  virtual void OnTraceFinished() = 0;
};

// ================================================================================================
// IRocTraceSource — mirrors GpuUtil::ITraceSource.
// Sources write RDF chunks when the trace finishes.
class IRocTraceSource {
public:
  virtual ~IRocTraceSource() = default;
  virtual const char* GetName()    const = 0;
  virtual uint32_t    GetVersion() const = 0;

  /// Called when a trace begins; sources may set up per-trace bookkeeping.
  virtual void OnTraceBegin() = 0;
  /// Called when the trace ends; sources may flush bookkeeping.
  virtual void OnTraceEnd() = 0;
  /// Called when the trace is fully finished; source should call WriteDataChunk() here.
  virtual void OnTraceFinished(RocTraceSession* pSession) = 0;
};

// ================================================================================================
// RocTraceSession — lightweight ROCm port of GpuUtil::TraceSession.
//
// Thread-safety: public methods are serialised by m_mutex.
class RocTraceSession {
public:
  RocTraceSession();
  ~RocTraceSession();

  /// Initialise the RDF stream and chunk-file writer.  Must be called once before use.
  bool Init();

  // ── Tool-driven entry points (called via RocUberTraceService RPC) ────────────

  /// Mark tracing as enabled.  Mirrors GpuUtil::TraceSession::EnableTracing() which
  /// is invoked by UberTraceService::EnableTracing() in response to a tool RPC call.
  void EnableTracing();

  /// Returns true if the tool has called EnableTracing().
  bool IsTracingEnabled() const;

  /// Request a new trace.  Notifies the registered controller.
  /// Returns false if a trace is already in flight or no controller is registered.
  bool RequestTrace();

  /// Cancel an in-flight trace.
  bool CancelTrace();

  /// Collect serialised RDF trace data.
  /// Call with pData == nullptr first to obtain the required size, then again
  /// with a caller-allocated buffer.  Mirrors TraceSession::CollectTrace().
  bool CollectTrace(void* pData, size_t* pDataSize);

  // ── Controller / source registration ────────────────────────────────────────

  bool RegisterController(IRocTraceController* pController);
  bool RegisterSource(IRocTraceSource* pSource);
  void UnregisterController(IRocTraceController* pController);
  void UnregisterSource(IRocTraceSource* pSource);

  // ── Called by the controller to advance the state machine ───────────────────

  /// Controller calls this from OnTraceRequested() to accept the trace.
  bool AcceptTrace(IRocTraceController* pController);

  /// Driver calls this when preparation dispatches are done; advances to Running.
  bool BeginTrace();

  /// Driver calls this when capture dispatches are done; advances to Completed.
  bool EndTrace();

  // ── State accessors ──────────────────────────────────────────────────────────

  RocTraceSessionState GetState() const;

  /// Returns true if a controller has been registered.
  bool HasController() const { return m_pController != nullptr; }

  // ── Data emission (called by IRocTraceSource::OnTraceFinished) ───────────────

  /// Write one RDF chunk into the current trace stream.
  /// chunkId must be exactly RDF_IDENTIFIER_SIZE (16) bytes, zero-padded.
  bool WriteDataChunk(const char chunkId[kRdfIdentifierSize],
                      uint32_t   version,
                      const void* pHeader,  size_t headerSize,
                      const void* pData,    size_t dataSize);

private:
  void ResetStream();

  mutable std::mutex           m_mutex;
  bool                         m_tracingEnabled;
  RocTraceSessionState         m_state;

  // RDF chunk-file serialisation
  rdfStream*           m_pStream;
  rdfChunkFileWriter*  m_pChunkWriter;

  // Registered controller (one at a time, like PAL)
  IRocTraceController* m_pController;

  // Registered sources
  std::vector<IRocTraceSource*> m_sources;
};

} // namespace roc
