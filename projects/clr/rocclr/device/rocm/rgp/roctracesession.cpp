/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ROCm-native TraceSession implementation.
 * PAL equivalent: pal/src/gpuUtil/traceSession.cpp
 */

#ifdef ROC_GPUOPEN

#include "device/rocm/rgp/roctracesession.hpp"
#include "device/rocm/rocdevice.hpp"
#include "amdrdf.h"

#include <algorithm>
#include <cstring>

namespace roc {

// ================================================================================================
RocTraceSession::RocTraceSession()
    : m_tracingEnabled(false),
      m_state(RocTraceSessionState::Ready),
      m_pStream(nullptr),
      m_pChunkWriter(nullptr),
      m_pController(nullptr) {}

// ================================================================================================
RocTraceSession::~RocTraceSession() {
  ResetStream();
}

// ================================================================================================
bool RocTraceSession::Init() {
  // Create the in-memory RDF stream that will hold the serialised trace data.
  // Mirrors TraceSession::RequestTrace() → rdfStreamCreateMemoryStream().
  const int r = rdfStreamCreateMemoryStream(&m_pStream);
  if (r != rdfResultOk || m_pStream == nullptr) {
    return false;
  }
  return true;
}

// ================================================================================================
void RocTraceSession::EnableTracing() {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_tracingEnabled = true;
}

// ================================================================================================
bool RocTraceSession::IsTracingEnabled() const {
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_tracingEnabled;
}

// ================================================================================================
bool RocTraceSession::RequestTrace() {
  std::lock_guard<std::mutex> lock(m_mutex);

  if (m_state != RocTraceSessionState::Ready) {
    return false;  // trace already in flight
  }
  if (m_pController == nullptr) {
    return false;  // no controller registered
  }

  m_state = RocTraceSessionState::Requested;

  // Notify controller — it will call AcceptTrace() if it accepts.
  // Release the lock before calling back to avoid deadlock.
  IRocTraceController* ctrl = m_pController;
  m_mutex.unlock();
  const bool accepted = ctrl->OnTraceRequested(this);
  m_mutex.lock();

  if (!accepted) {
    m_state = RocTraceSessionState::Ready;
  }
  return accepted;
}

// ================================================================================================
bool RocTraceSession::CancelTrace() {
  std::lock_guard<std::mutex> lock(m_mutex);

  if (m_state == RocTraceSessionState::Ready ||
      m_state == RocTraceSessionState::Completed) {
    return false;
  }

  const RocTraceSessionState prev = m_state;
  m_state = RocTraceSessionState::Ready;
  ResetStream();

  // Notify controller if it had already accepted.
  if (prev != RocTraceSessionState::Requested && m_pController != nullptr) {
    IRocTraceController* ctrl = m_pController;
    m_mutex.unlock();
    ctrl->OnTraceCanceled();
    m_mutex.lock();
  }

  return true;
}

// ================================================================================================
bool RocTraceSession::AcceptTrace(IRocTraceController* pController) {
  std::lock_guard<std::mutex> lock(m_mutex);

  if (m_state != RocTraceSessionState::Requested) {
    return false;
  }
  if (pController != m_pController) {
    return false;  // only the registered controller may accept
  }

  // Re-create a fresh RDF stream + chunk writer for this trace.
  ResetStream();

  if (rdfStreamCreateMemoryStream(&m_pStream) != rdfResultOk) {
    m_state = RocTraceSessionState::Ready;
    return false;
  }
  if (rdfChunkFileWriterCreate(m_pStream, &m_pChunkWriter) != rdfResultOk) {
    rdfStreamClose(&m_pStream);
    m_state = RocTraceSessionState::Ready;
    return false;
  }

  m_state = RocTraceSessionState::Preparing;

  // Notify sources that a trace has begun.
  for (IRocTraceSource* src : m_sources) {
    src->OnTraceBegin();
  }

  return true;
}

// ================================================================================================
bool RocTraceSession::BeginTrace() {
  std::lock_guard<std::mutex> lock(m_mutex);

  if (m_state != RocTraceSessionState::Preparing) {
    return false;
  }
  m_state = RocTraceSessionState::Running;
  return true;
}

// ================================================================================================
bool RocTraceSession::EndTrace() {
  std::lock_guard<std::mutex> lock(m_mutex);

  if (m_state != RocTraceSessionState::Running) {
    return false;
  }

  // Notify sources that the trace capture has ended.
  for (IRocTraceSource* src : m_sources) {
    src->OnTraceEnd();
  }

  // Let sources emit their RDF chunks.
  for (IRocTraceSource* src : m_sources) {
    m_mutex.unlock();
    src->OnTraceFinished(this);
    m_mutex.lock();
  }

  // Let controller clean up.
  if (m_pController != nullptr) {
    IRocTraceController* ctrl = m_pController;
    m_mutex.unlock();
    ctrl->OnTraceFinished();
    m_mutex.lock();
  }

  // Finalise the RDF chunk file.
  if (m_pChunkWriter != nullptr) {
    rdfChunkFileWriterDestroy(&m_pChunkWriter);
    m_pChunkWriter = nullptr;
  }

  m_state = RocTraceSessionState::Completed;
  return true;
}

// ================================================================================================
bool RocTraceSession::CollectTrace(void* pData, size_t* pDataSize) {
  std::lock_guard<std::mutex> lock(m_mutex);

  if (m_state != RocTraceSessionState::Completed || m_pStream == nullptr) {
    return false;
  }

  int64_t streamSize = 0;
  if (rdfStreamGetSize(m_pStream, &streamSize) != rdfResultOk) {
    return false;
  }

  if (pData == nullptr) {
    // First call: return required size.
    *pDataSize = static_cast<size_t>(streamSize);
    return true;
  }

  if (*pDataSize < static_cast<size_t>(streamSize)) {
    return false;
  }

  // Seek to start and read all data.
  if (rdfStreamSeek(m_pStream, 0) != rdfResultOk) {
    return false;
  }

  int64_t bytesRead = 0;
  if (rdfStreamRead(m_pStream, static_cast<int64_t>(*pDataSize), pData, &bytesRead) != rdfResultOk) {
    return false;
  }

  *pDataSize = static_cast<size_t>(bytesRead);

  // Reset back to Ready so another trace can be requested.
  ResetStream();
  m_state = RocTraceSessionState::Ready;

  return true;
}

// ================================================================================================
bool RocTraceSession::WriteDataChunk(const char  chunkId[kRdfIdentifierSize],
                                     uint32_t    version,
                                     const void* pHeader,   size_t headerSize,
                                     const void* pData,     size_t dataSize) {
  std::lock_guard<std::mutex> lock(m_mutex);

  if (m_pChunkWriter == nullptr) {
    return false;
  }

  rdfChunkCreateInfo info = {};
  memcpy(info.identifier, chunkId, kRdfIdentifierSize);
  info.headerSize  = static_cast<int64_t>(headerSize);
  info.pHeader     = pHeader;
  info.compression = rdfCompressionNone;
  info.version     = version;

  int index = 0;
  return rdfChunkFileWriterWriteChunk(m_pChunkWriter, &info,
                                      static_cast<int64_t>(dataSize), pData,
                                      &index) == rdfResultOk;
}

// ================================================================================================
bool RocTraceSession::RegisterController(IRocTraceController* pController) {
  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_pController != nullptr) {
    return false;  // only one controller at a time (like PAL)
  }
  m_pController = pController;
  return true;
}

// ================================================================================================
void RocTraceSession::UnregisterController(IRocTraceController* pController) {
  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_pController == pController) {
    m_pController = nullptr;
  }
}

// ================================================================================================
bool RocTraceSession::RegisterSource(IRocTraceSource* pSource) {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_sources.push_back(pSource);
  return true;
}

// ================================================================================================
void RocTraceSession::UnregisterSource(IRocTraceSource* pSource) {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_sources.erase(std::remove(m_sources.begin(), m_sources.end(), pSource),
                  m_sources.end());
}

// ================================================================================================
RocTraceSessionState RocTraceSession::GetState() const {
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_state;
}

// ================================================================================================
void RocTraceSession::ResetStream() {
  if (m_pChunkWriter != nullptr) {
    rdfChunkFileWriterDestroy(&m_pChunkWriter);
    m_pChunkWriter = nullptr;
  }
  if (m_pStream != nullptr) {
    rdfStreamClose(&m_pStream);
    m_pStream = nullptr;
  }
}

} // namespace roc

#endif // ROC_GPUOPEN
