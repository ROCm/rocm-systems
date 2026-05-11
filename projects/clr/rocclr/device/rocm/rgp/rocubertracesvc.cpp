/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ROCm UberTraceService implementation.
 * PAL equivalent: pal/src/gpuUtil/uberTraceService.cpp
 */

#ifdef ROC_GPUOPEN

#include "device/rocm/rgp/rocubertracesvc.hpp"

#include <cstdlib>
#include <cstring>

namespace roc {

// ================================================================================================
RocUberTraceService::RocUberTraceService(RocTraceSession* pSession)
    : m_pSession(pSession),
      m_traceActive(false) {}

// ================================================================================================
// Mirrors GpuUtil::UberTraceService::EnableTracing():
//   m_pPlatform->GetTraceSession()->EnableTracing();
DD_RESULT RocUberTraceService::EnableTracing() {
  m_pSession->EnableTracing();
  return DD_RESULT_SUCCESS;
}

// ================================================================================================
// Stub — no configurable params yet.
DD_RESULT RocUberTraceService::QueryTraceParams(const DDByteWriter& /*writer*/) {
  return DD_RESULT_SUCCESS;
}

// ================================================================================================
// Stub — config parsing not yet implemented.
DD_RESULT RocUberTraceService::ConfigureTraceParams(const void* /*pParamBuffer*/,
                                                    size_t      /*paramBufferSize*/) {
  return DD_RESULT_SUCCESS;
}

// ================================================================================================
// Mirrors GpuUtil::UberTraceService::RequestTrace():
//   m_traceActive = true;
//   m_traceInactiveEvent.Reset();
//   Result result = m_pPlatform->GetTraceSession()->RequestTrace();
DD_RESULT RocUberTraceService::RequestTrace() {
  m_traceActive.store(true);
  return m_pSession->RequestTrace() ? DD_RESULT_SUCCESS : DD_RESULT_DD_GENERIC_UNAVAILABLE;
}

// ================================================================================================
// Mirrors GpuUtil::UberTraceService::CancelTrace().
DD_RESULT RocUberTraceService::CancelTrace() {
  m_pSession->CancelTrace();
  m_traceActive.store(false);
  return DD_RESULT_SUCCESS;
}

// ================================================================================================
// Mirrors GpuUtil::UberTraceService::CollectTrace():
//   calls CollectTrace twice (size query then data), writes via DDByteWriter.
DD_RESULT RocUberTraceService::CollectTrace(const DDByteWriter& writer) {
  // First call: query required size.
  size_t dataSize = 0;
  if (!m_pSession->CollectTrace(nullptr, &dataSize)) {
    return DD_RESULT_DD_GENERIC_UNAVAILABLE;
  }

  void* pData = malloc(dataSize);
  if (pData == nullptr) {
    return DD_RESULT_DD_GENERIC_INSUFFICIENT_MEMORY;
  }

  // Second call: retrieve the data.
  DD_RESULT result = DD_RESULT_SUCCESS;
  if (!m_pSession->CollectTrace(pData, &dataSize)) {
    result = DD_RESULT_DD_GENERIC_UNAVAILABLE;
  }

  if (result == DD_RESULT_SUCCESS) {
    result = writer.pfnBegin(writer.pUserdata, &dataSize);
  }

  if (result == DD_RESULT_SUCCESS) {
    result = writer.pfnWriteBytes(writer.pUserdata, pData, dataSize);
  }

  writer.pfnEnd(writer.pUserdata, result);

  free(pData);

  // Mirror GpuUtil::UberTraceService: reset active flag once collection is done
  // and the session is back to Ready.
  if (result == DD_RESULT_SUCCESS &&
      m_pSession->GetState() == RocTraceSessionState::Ready) {
    m_traceActive.store(false);
  }

  return result;
}

} // namespace roc

#endif // ROC_GPUOPEN
