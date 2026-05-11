/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ROCm implementation of the DriverUtils RPC service.
 * PAL equivalent: pal/src/core/rpcDriverUtilsService/
 *
 * DriverUtils exposes driver-utility RPCs to developer tools (RGP, RDP).
 * Methods that are PAL-specific (QueryPalDriverInfo, overlay, DbgLog) are
 * stubbed as no-ops; EnableTracing and EnableDriverFeatures forward to the
 * RocTraceSession so tools can enable tracing via either UberTrace or DriverUtils.
 */

#pragma once

#ifdef ROC_GPUOPEN

#include "g_DriverUtilsService.h"
#include "device/rocm/rgp/roctracesession.hpp"

namespace roc {

// ================================================================================================
// RocDriverUtilsService — ROCm implementation of DriverUtils::IDriverUtilsService.
//
// Mirrors PAL's platform-level DriverUtils service.  PAL-specific methods
// (overlay string, DbgLog, QueryPalDriverInfo) are no-ops on ROCm.
class RocDriverUtilsService final : public DriverUtils::IDriverUtilsService {
public:
  explicit RocDriverUtilsService(RocTraceSession* pSession)
      : m_pSession(pSession) {}
  ~RocDriverUtilsService() override = default;

  // Informs driver that a tool is collecting trace data.
  // Mirrors PAL: Platform::EnableTracing() → TraceSession::EnableTracing()
  DD_RESULT EnableTracing() override {
    if (m_pSession != nullptr) {
      m_pSession->EnableTracing();
    }
    return DD_RESULT_SUCCESS;
  }

  // Crash analysis mode is not supported on ROCm — stub.
  DD_RESULT EnableCrashAnalysisMode() override { return DD_RESULT_SUCCESS; }

  // PAL-specific driver info query — not applicable on ROCm.
  DD_RESULT QueryPalDriverInfo(const DDByteWriter& /*writer*/) override {
    return DD_RESULT_SUCCESS;
  }

  // Feature enable bitmask — forward tracing bit to TraceSession, ignore others.
  DD_RESULT EnableDriverFeatures(const void* /*pParamBuffer*/,
                                 size_t      /*paramBufferSize*/) override {
    // PAL parses a bitmask here (Tracing, CrashAnalysis, RTShaderTokens, DebugVmid).
    // On ROCm we only support tracing; forward to EnableTracing() for simplicity.
    return EnableTracing();
  }

  // Overlay and DbgLog methods are PAL/Windows-display specific — stubs.
  DD_RESULT SetOverlayString(const void*, size_t) override { return DD_RESULT_SUCCESS; }
  DD_RESULT SetDbgLogSeverityLevel(const void*, size_t) override { return DD_RESULT_SUCCESS; }
  DD_RESULT SetDbgLogOriginationMask(const void*, size_t) override { return DD_RESULT_SUCCESS; }
  DD_RESULT ModifyDbgLogOriginationMask(const void*, size_t) override { return DD_RESULT_SUCCESS; }

private:
  RocTraceSession* m_pSession;
};

} // namespace roc

#endif // ROC_GPUOPEN
