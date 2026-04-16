//===-- aegisbit/ActiveDispatch.h - In-flight dispatch record ---*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Record describing a kernel dispatch that the tracing engine has
/// intercepted and redirected to its patched equivalent. Extracted from
/// TracingEngine.h so that DispatchRegistry can reference it without pulling
/// in the full engine header.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_ACTIVE_DISPATCH_H
#define AEGISBIT_ACTIVE_DISPATCH_H

#include "aegisbit/CoalescingAnalyzer.h" // SiteInfo
#include "aegisbit/KernelLauncher.h"     // DispatchParams

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace aegisbit {

/// Information about an active (in-flight) kernel dispatch.
struct ActiveDispatch {
  /// Unique dispatch identifier.
  uint32_t DispatchID = 0;

  /// Kernel name.
  std::string KernelName;

  /// Dispatch parameters.
  DispatchParams Params;

  /// Start time of dispatch.
  std::chrono::steady_clock::time_point StartTime;

  /// Extended kernel arguments (original + TraceArgs).
  std::vector<uint8_t> ExtendedKernarg;

  /// Original kernel object (for restoration if needed).
  uint64_t OriginalKernelObject = 0;

  /// Patched kernel object.
  uint64_t PatchedKernelObject = 0;

  /// GPU memory for extended kernarg (from pool).
  void *GpuKernargPtr = nullptr;

  /// Our completion signal handle (for detecting dispatch completion).
  uint64_t CompletionSignalHandle = 0;

  /// App's original completion signal handle (to forward to after trace
  /// collection).
  uint64_t OriginalSignalHandle = 0;

  /// Per-site instruction metadata for coalescing reports.
  std::vector<SiteInfo> SiteMap;

  /// Which patched-ELF variant was selected for this dispatch.  Zero for
  /// single-variant (replay-disabled) workloads.  Used by the sink at
  /// completion to route counters into the canonical per-PC table.
  uint32_t VariantID = 0;
};

} // namespace aegisbit

#endif // AEGISBIT_ACTIVE_DISPATCH_H
