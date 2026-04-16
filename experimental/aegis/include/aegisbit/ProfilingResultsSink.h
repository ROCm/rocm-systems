//===-- aegisbit/ProfilingResultsSink.h - Results aggregator ----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Thread-safe collector for VMEM coalescing and LDS bank-conflict summaries
/// produced on each dispatch completion. Emits a single JSON file on
/// `flush(Path)`. Extracted from TracingEngine so the aggregation logic can
/// be unit tested without the HSA runtime.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_PROFILING_RESULTS_SINK_H
#define AEGISBIT_PROFILING_RESULTS_SINK_H

#include "aegisbit/CoalescingAnalyzer.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace aegisbit {

struct PersistentTraceBuffer;

/// Per-kernel accumulator keyed by the **original** program counter of each
/// instrumented site.  Variant-local site indices collapse into one canonical
/// per-PC entry so instrumentation-replay (K complementary patched ELFs per
/// kernel) can merge samples from every variant into a single report.
///
/// Duplicate-PC policy:
///   - `OnGpuReduce` — sums `TotalCacheLines`, `TotalUniqueBanks`, and
///     `TotalSamples`.  Same PC appearing in multiple variants or repeated
///     dispatches accumulates additively.
///   - `FullCapture` — not handled by this data model yet; that path stays
///     on the legacy per-summary append in `ProfilingResultsSink` until the
///     strategy reaches production.
///
/// `FirstSeenSiteID` / `FirstSeenInfo` retain the first `SiteInfo` observed
/// for each PC (for source mapping, element size, instruction name) so the
/// JSON output can still emit per-site rows even though the accumulator is
/// keyed by PC.
struct CanonicalKernelAccum {
  struct VMEMEntry {
    uint64_t OriginalPC = 0;
    uint32_t FirstSeenSiteID = 0;
    uint64_t TotalCacheLines = 0;
    uint64_t TotalSamples = 0;
    SiteInfo FirstSeenInfo;
  };
  struct LDSEntry {
    uint64_t OriginalPC = 0;
    uint32_t FirstSeenSiteID = 0;
    uint64_t TotalUniqueBanks = 0;
    uint64_t TotalSamples = 0;
    SiteInfo FirstSeenInfo;
  };

  /// Merge one VMEM reading.  Duplicate PCs (same or different variants)
  /// accumulate `TotalCacheLines` and `TotalSamples`; `FirstSeenSiteID`
  /// and `FirstSeenInfo` remain stable after the first insert.
  void addVMEMSample(const SiteInfo &Info, uint64_t CacheLines,
                     uint64_t Samples);

  /// Merge one LDS reading.  `UniqueBanks` is the aggregated max-lanes-per-
  /// bank counter emitted by the on-GPU reducer.
  void addLDSSample(const SiteInfo &Info, uint64_t UniqueBanks,
                    uint64_t Samples);

  std::unordered_map<uint64_t, VMEMEntry> VMEMByPC;
  std::unordered_map<uint64_t, LDSEntry> LDSByPC;
};

class ProfilingResultsSink {
public:
  /// Interpret a completed dispatch's persistent trace buffer and append the
  /// derived VMEM and (optionally) LDS summaries. Selects the analysis path
  /// based on `PB.Config.Strategy`:
  ///   - `OnGpuReduce` — per-site 8-byte counters (cache-lines / unique-banks
  ///     + sample count).
  ///   - `FullCapture` — raw address-record buffer sized by the atomic write
  ///     offset in `PB.CounterPtr`.
  /// Logs per-kernel summaries via `RuntimeConfig::log`. Safe to call from
  /// any thread; aggregation is serialized internally.
  ///
  /// No-op when `PB.BufferPtr` is null.
  ///
  /// \param VariantID Index of the patched-ELF variant that produced this
  ///        buffer.  Non-zero only when instrumentation replay is enabled
  ///        (`AEGISBIT_REPLAY`).  Present for telemetry only — canonical
  ///        aggregation is keyed by original PC, so samples from any
  ///        variant merge into the same per-kernel report.
  void ingest(const PersistentTraceBuffer &PB,
              const std::vector<SiteInfo> &SiteMap,
              const std::string &KernelName,
              uint32_t VariantID = 0);

  /// Append a VMEM coalescing summary produced on dispatch completion.
  void addVMEM(CoalescingSummary Summary);

  /// Append an LDS bank-conflict summary produced on dispatch completion.
  void addLDS(LDSSummary Summary);

  /// Current counts (handy for tests and stats).
  size_t vmemCount() const;
  size_t ldsCount() const;

  /// Write the accumulated summaries to JSON and clear both buffers.
  /// No-op if Path is empty.
  void flush(const std::string &Path);

  /// Drop any accumulated summaries without writing.
  void clear();

  /// Access the current canonical-by-PC table for \p KernelName.  Returns
  /// nullptr if the kernel has not yet been ingested.  Intended for tests
  /// and diagnostics only; aggregation is serialized on the sink's mutex
  /// so the returned pointer is safe to read without external locking only
  /// as long as no other thread is calling `ingest` / `flush` / `clear`.
  const CanonicalKernelAccum *findCanonical(const std::string &KernelName) const;

private:
  mutable std::mutex Mutex;
  std::vector<CoalescingSummary> VMEM;
  std::vector<LDSSummary> LDS;

  /// Per-kernel canonical accumulators keyed by original PC.  Populated from
  /// every `ingest` call across all variants; consumed by `flush`.
  std::unordered_map<std::string, CanonicalKernelAccum> ByKernel;
};

} // namespace aegisbit

#endif // AEGISBIT_PROFILING_RESULTS_SINK_H
