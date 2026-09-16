/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ROCm UberTraceService implementation.
 * PAL equivalent: pal/src/gpuUtil/uberTraceService.cpp
 */

#ifdef ROC_GPUOPEN

#include "device/rocm/rgp/rocubertracesvc.hpp"

#include "util/ddStructuredReader.h"  // DevDriver::IStructuredReader / StructuredValue

#include <cstdio>
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
// Parse RDP's RDF JSON trace config into m_config.
//
// PAL equivalent: GpuUtil::TraceSession::UpdateTraceConfig (traceSession.cpp) fans the same JSON
// out to RenderOpTraceController::OnConfigUpdated (controller.config) and
// GpuPerfExperimentTraceSource::OnSqttConfigUpdated (sources[gpuperfexp].config.sqtt). The ROC
// port has no controller/source objects, so we parse those same keys directly into the flat
// RocTraceConfig consumed by RocUberTraceCaptureMgr::OnTraceRequested().
//
// Only fields whose JSON key is present are overwritten; absent keys keep RocTraceConfig defaults.
DD_RESULT RocUberTraceService::ConfigureTraceParams(const void* pParamBuffer,
                                                    size_t      paramBufferSize) {
  if (pParamBuffer == nullptr || paramBufferSize == 0) {
    return DD_RESULT_DD_GENERIC_INVALID_PARAMETER;
  }

  DevDriver::IStructuredReader* pReader = nullptr;
  if (DevDriver::IStructuredReader::CreateFromJson(pParamBuffer, paramBufferSize,
                                                   DevDriver::Platform::GenericAllocCb,
                                                   &pReader) != DevDriver::Result::Success) {
    return DD_RESULT_DD_GENERIC_INVALID_PARAMETER;
  }

  const DevDriver::StructuredValue root = pReader->GetRoot();

  // ── Controller (controller.config; older tools nest it under controllers[0]) ─────────────
  // Mirrors RenderOpTraceController::OnConfigUpdated. For HIP/OpenCL compute the controller name
  // is "renderop" with renderOpMode "dispatch"; we don't need to key off those here.
  DevDriver::StructuredValue controller = root["controller"];
  if (controller.IsNull()) {
    controller = root["controllers"][0];
  }
  if (!controller.IsNull()) {
    const DevDriver::StructuredValue cfg = controller["config"];
    DevDriver::StructuredValue value;

    // captureMode "absolute" → dispatch-index window; "relative" (default) → false.
    if (cfg.GetValueByKey("captureMode", &value)) {
      char buffer[32] = {'\0'};
      if (value.GetStringCopy(buffer)) {
        m_config.indexMode = (strcmp(buffer, "absolute") == 0);
      }
    }
    if (cfg.GetValueByKey("preparationStartRenderOp", &value)) {
      m_config.captureStartIndex = value.GetUint32Or(m_config.captureStartIndex);
    }
    if (cfg.GetValueByKey("numPrepRenderOps", &value)) {
      m_config.numPrepDispatches = value.GetUint32Or(m_config.numPrepDispatches);
    }
    if (cfg.GetValueByKey("captureRenderOpCount", &value)) {
      m_config.captureDispatchCount = value.GetUint32Or(m_config.captureDispatchCount);
      if (m_config.captureDispatchCount < 1) {  // can't capture 0 dispatches
        m_config.captureDispatchCount = 1;
      }
    }
  }

  // ── Sources: find the "gpuperfexp" source, read its config.sqtt ──────────────────────────
  // Mirrors GpuPerfExperimentTraceSource::OnSqttConfigUpdated.
  const DevDriver::StructuredValue sources = root["sources"];
  const size_t numSources = sources.GetArrayLength();
  for (size_t i = 0; i < numSources; ++i) {
    const DevDriver::StructuredValue src = sources[i];
    const char* pName = src["name"].GetStringPtr();
    if (pName == nullptr || strcmp(pName, "gpuperfexp") != 0) {
      continue;
    }

    const DevDriver::StructuredValue sqtt = src["config"]["sqtt"];
    DevDriver::StructuredValue value;

    if (sqtt.GetValueByKey("memoryLimitInMb", &value)) {
      m_config.sqttMemoryLimitInMb =
          static_cast<uint32_t>(value.GetUint64Or(m_config.sqttMemoryLimitInMb));
    }
    if (sqtt.GetValueByKey("enableInstructionTokens", &value)) {
      m_config.enableInstructionTokens = value.GetBoolOr(m_config.enableInstructionTokens);
    }
    if (sqtt.GetValueByKey("seMask", &value)) {
      m_config.seMask = value.GetUint32Or(m_config.seMask);
    }
    // TODO: parse sqtt.spm.perfCounters (SPM counter list) once ROCr SPM capture is wired up.
    break;
  }

  DevDriver::IStructuredReader::Destroy(&pReader);

  // Plain fprintf (not gated by NDEBUG) so parsed params are visible in every build config,
  // including Release/RelWithDebInfo, during on-hardware trace tests.
  fprintf(stderr,
          "[CLR-Ctrl] ConfigureTraceParams: indexMode=%d startIdx=%u numPrep=%u dispCount=%u"
          " memMb=%u seMask=0x%x instrTokens=%d codeObj=%d\n",
          m_config.indexMode, m_config.captureStartIndex, m_config.numPrepDispatches,
          m_config.captureDispatchCount, m_config.sqttMemoryLimitInMb, m_config.seMask,
          m_config.enableInstructionTokens, m_config.captureCodeObjects);

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
