/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ROCm UberTraceService — forwards UberTrace RPC calls to RocTraceSession.
 * PAL equivalent: pal/src/gpuUtil/uberTraceService.h
 */

#pragma once

#include "device/rocm/rgp/roctracesession.hpp"
#include "UberTraceService.h"  // auto-generated UberTrace::IService

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace roc {

// ================================================================================================
// Trace configuration parsed from ConfigureTraceParams() JSON.
// Mirrors the fields used by RocUberTraceCaptureMgr::OnTraceRequested().
struct RocTraceConfig {
  uint32_t numPrepDispatches       = 2;     ///< Preparation dispatches before SQTT starts
  uint32_t sqttMemoryLimitInMb     = 64;    ///< SQTT ring buffer size in MB per SE
  uint32_t seMask                  = 0xF;  ///< Shader-engine enable mask
  bool     enableInstructionTokens = true; ///< Capture instruction-level SQTT tokens
  bool     captureCodeObjects      = true; ///< Attach code objects to the .rgp file
  bool     indexMode               = false;///< Use dispatch-index window instead of frame
  uint32_t captureStartIndex       = 0;    ///< Dispatch index to start capture (indexMode)
  uint32_t captureDispatchCount    = 0;    ///< Number of dispatches to capture (0 = unlimited)
};

// ================================================================================================
// RocUberTraceService — ROCm implementation of UberTrace::IService.
//
// All RPC methods simply forward to RocTraceSession, mirroring
// GpuUtil::UberTraceService (pal/src/gpuUtil/uberTraceService.cpp).
class RocUberTraceService final : public UberTrace::IService {
public:
  explicit RocUberTraceService(RocTraceSession* pSession);
  ~RocUberTraceService() override = default;

  // ── UberTrace::IService RPC methods ─────────────────────────────────────────

  /// Called by a tool to enable UberTrace-based tracing on this driver.
  /// Sets RocTraceSession::m_tracingEnabled = true.
  /// PAL equivalent: GpuUtil::UberTraceService::EnableTracing()
  DD_RESULT EnableTracing() override;

  /// Returns current trace parameters as JSON (stub — no params yet).
  DD_RESULT QueryTraceParams(const DDByteWriter& writer) override;

  /// Applies JSON trace configuration from the tool.
  DD_RESULT ConfigureTraceParams(const void* pParamBuffer, size_t paramBufferSize) override;

  /// Requests a trace capture.  Sets m_traceActive = true.
  DD_RESULT RequestTrace() override;

  /// Cancels an in-flight trace.
  DD_RESULT CancelTrace() override;

  /// Collects serialised RDF trace data and streams it to the tool.
  DD_RESULT CollectTrace(const DDByteWriter& writer) override;

  // ── State query ──────────────────────────────────────────────────────────────

  /// True between RequestTrace() and the end of CollectTrace() — mirrors
  /// GpuUtil::UberTraceService::IsTraceActive().
  bool IsTraceActive() const { return m_traceActive.load(); }

  /// Returns the trace configuration set by ConfigureTraceParams().
  const RocTraceConfig& GetTraceConfig() const { return m_config; }

  /// Block until CollectTrace() completes or timeout_ms elapses.
  /// Returns true if collection finished, false on timeout.
  bool WaitForCollectDone(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lock(m_collectMutex);
    return m_collectCv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                [this] { return !m_collecting.load(); });
  }

private:
  RocTraceSession*          m_pSession;
  std::atomic<bool>         m_traceActive;
  RocTraceConfig            m_config;
  std::mutex                m_collectMutex;
  std::condition_variable   m_collectCv;
  std::atomic<bool>         m_collecting{false};
};

} // namespace roc
