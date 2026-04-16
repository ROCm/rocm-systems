//===-- ProfilingResultsSink.cpp - Results aggregator implementation ------===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/ProfilingResultsSink.h"

#include "aegisbit/PersistentBufferCache.h"
#include "aegisbit/RuntimeConfig.h"
#include "aegisbit/Types.h"

#include <algorithm>
#include <cstdint>

namespace aegisbit {

void CanonicalKernelAccum::addVMEMSample(const SiteInfo &Info,
                                         uint64_t CacheLines,
                                         uint64_t Samples) {
  auto [It, Inserted] = VMEMByPC.try_emplace(Info.OriginalPC);
  VMEMEntry &E = It->second;
  if (Inserted) {
    E.OriginalPC = Info.OriginalPC;
    E.FirstSeenSiteID = Info.SiteID;
    E.FirstSeenInfo = Info;
  }
  E.TotalCacheLines += CacheLines;
  E.TotalSamples += Samples;
}

void CanonicalKernelAccum::addLDSSample(const SiteInfo &Info,
                                        uint64_t UniqueBanks,
                                        uint64_t Samples) {
  auto [It, Inserted] = LDSByPC.try_emplace(Info.OriginalPC);
  LDSEntry &E = It->second;
  if (Inserted) {
    E.OriginalPC = Info.OriginalPC;
    E.FirstSeenSiteID = Info.SiteID;
    E.FirstSeenInfo = Info;
  }
  E.TotalUniqueBanks += UniqueBanks;
  E.TotalSamples += Samples;
}

void ProfilingResultsSink::ingest(const PersistentTraceBuffer &PB,
                                  const std::vector<SiteInfo> &SiteMap,
                                  const std::string &KernelName,
                                  uint32_t VariantID) {
  if (!PB.BufferPtr)
    return;

  RuntimeConfig &Cfg = RuntimeConfig::getInstance();

  if (PB.Config.Strategy == PayloadStrategy::OnGpuReduce) {
    // Per-site reduced counters: {total_cache_lines u32, total_samples u32}
    // for VMEM, {total_unique_banks u32, total_samples u32} for LDS.
    auto VMEMSummary = CoalescingAnalyzer::analyzeOnGpuCounters(
        PB.BufferPtr, SiteMap, KernelName);
    CoalescingAnalyzer::printReport(VMEMSummary);
    addVMEM(VMEMSummary);

    // Canonical (per-PC) aggregation: variant-local site indices collapse
    // into one canonical entry keyed by the original kernel PC, so multiple
    // patched-ELF variants contribute additively to the same per-site row.
    {
      const auto *Counters =
          reinterpret_cast<const uint32_t *>(PB.BufferPtr);
      std::lock_guard<std::mutex> Lock(Mutex);
      auto &Accum = ByKernel[KernelName];
      for (const auto &SI : SiteMap) {
        uint32_t CacheLines = Counters[SI.SiteID * 2];
        uint32_t Samples    = Counters[SI.SiteID * 2 + 1];
        if (Samples == 0)
          continue;
        if (SI.IsLDS)
          Accum.addLDSSample(SI, CacheLines, Samples);
        else
          Accum.addVMEMSample(SI, CacheLines, Samples);
      }
    }

    Cfg.log("Profiler (OnGpuReduce): " + KernelName + " v" +
            std::to_string(VariantID) + " — " +
            std::to_string(VMEMSummary.NumSites) + " VMEM sites, " +
            std::to_string(
                static_cast<int>(VMEMSummary.OverallEfficiency * 100)) +
            "% efficiency");

    auto LDSSumm = CoalescingAnalyzer::analyzeOnGpuCountersLDS(
        PB.BufferPtr, SiteMap, KernelName);
    if (LDSSumm.NumSites > 0) {
      CoalescingAnalyzer::printLDSReport(LDSSumm);
      addLDS(LDSSumm);

      Cfg.log("Profiler (OnGpuReduce): " + KernelName + " v" +
              std::to_string(VariantID) + " — " +
              std::to_string(LDSSumm.NumSites) + " LDS sites, " +
              "avg conflict_cycles=" +
              std::to_string(
                  static_cast<int>(LDSSumm.OverallAvgConflictCycles)));
    }
    return;
  }

  // FullCapture: analyze raw address record buffer. NumRecords is derived
  // from the atomic write-offset counter in `PB.CounterPtr`; we clamp to
  // the buffer capacity so a runaway kernel can't crash the analysis.
  if (!PB.CounterPtr)
    return;

  uint64_t CounterBytes = *reinterpret_cast<uint64_t *>(PB.CounterPtr);
  uint32_t MaxRecords =
      static_cast<uint32_t>(PB.BufferSize / TraceConfig::RecordSize);
  uint32_t NumRecords =
      static_cast<uint32_t>(CounterBytes / TraceConfig::RecordSize);
  if (NumRecords > MaxRecords) {
    Cfg.log("WARNING: Trace buffer overflow — " + std::to_string(NumRecords) +
            " records written, buffer has " + std::to_string(MaxRecords) +
            " capacity. Clamping.");
    NumRecords = MaxRecords;
  }
  if (NumRecords == 0)
    return;

  std::vector<SiteInfo> VMEMSiteMap;
  bool HasLDS = false;
  for (const auto &SI : SiteMap) {
    if (SI.IsLDS)
      HasLDS = true;
    else
      VMEMSiteMap.push_back(SI);
  }

  if (!VMEMSiteMap.empty()) {
    auto Metrics = CoalescingAnalyzer::analyzeBuffer(PB.BufferPtr, NumRecords,
                                                    VMEMSiteMap);
    auto Summary =
        CoalescingAnalyzer::summarize(Metrics, KernelName, VMEMSiteMap);
    CoalescingAnalyzer::printReport(Summary);
    addVMEM(Summary);

    Cfg.log("Profiler: " + KernelName + " — " + std::to_string(NumRecords) +
            " records, " + std::to_string(Summary.NumSites) +
            " VMEM sites, " +
            std::to_string(static_cast<int>(Summary.OverallEfficiency * 100)) +
            "% efficiency");
  }

  if (HasLDS) {
    auto LDSMetrics =
        CoalescingAnalyzer::analyzeLDSBuffer(PB.BufferPtr, NumRecords, SiteMap);
    if (!LDSMetrics.empty()) {
      auto LDSSumm =
          CoalescingAnalyzer::summarizeLDS(LDSMetrics, KernelName, SiteMap);
      CoalescingAnalyzer::printLDSReport(LDSSumm);
      addLDS(LDSSumm);

      Cfg.log("Profiler: " + KernelName + " — " +
              std::to_string(LDSSumm.TotalRecords) + " LDS records, " +
              std::to_string(LDSSumm.NumSites) + " LDS sites, " +
              "avg conflict cycles=" +
              std::to_string(
                  static_cast<int>(LDSSumm.OverallAvgConflictCycles)));
    }
  }
}

void ProfilingResultsSink::addVMEM(CoalescingSummary Summary) {
  std::lock_guard<std::mutex> Lock(Mutex);
  VMEM.push_back(std::move(Summary));
}

void ProfilingResultsSink::addLDS(LDSSummary Summary) {
  std::lock_guard<std::mutex> Lock(Mutex);
  LDS.push_back(std::move(Summary));
}

size_t ProfilingResultsSink::vmemCount() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return VMEM.size();
}

size_t ProfilingResultsSink::ldsCount() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return LDS.size();
}

namespace {

/// Convert a per-kernel `CanonicalKernelAccum` into a `CoalescingSummary`
/// suitable for `CoalescingAnalyzer::writeJSON`.  Mirrors the efficiency
/// calculation used by `CoalescingAnalyzer::analyzeOnGpuCounters` so that
/// the canonical-by-PC output is byte-for-byte comparable to the legacy
/// single-variant path.
CoalescingSummary canonicalToVMEMSummary(const std::string &KernelName,
                                         const CanonicalKernelAccum &A) {
  CoalescingSummary S;
  S.KernelName = KernelName;

  uint64_t TotalSamples = 0;
  double WeightedEfficiency = 0.0;

  for (const auto &[PC, E] : A.VMEMByPC) {
    if (E.TotalSamples == 0)
      continue;
    float AvgCacheLines =
        static_cast<float>(E.TotalCacheLines) / static_cast<float>(E.TotalSamples);

    float IdealCacheLines =
        static_cast<float>(64u * E.FirstSeenInfo.ElemSize) / 128.0f;
    if (IdealCacheLines < 1.0f)
      IdealCacheLines = 1.0f;

    float Efficiency =
        AvgCacheLines > 0.0f
            ? std::min(1.0f, IdealCacheLines / AvgCacheLines)
            : 0.0f;

    SiteStats SS;
    SS.SiteID = E.FirstSeenSiteID;
    SS.TotalAccesses = static_cast<uint32_t>(E.TotalSamples);
    SS.AvgEfficiency = Efficiency;
    SS.AvgCacheLines = AvgCacheLines;
    SS.MinCacheLines = static_cast<uint32_t>(AvgCacheLines);
    SS.MaxCacheLines = static_cast<uint32_t>(AvgCacheLines + 0.5f);
    SS.CoalescedCount = (Efficiency > 0.9f) ? SS.TotalAccesses : 0;
    SS.StridedCount = 0;
    SS.ScatteredCount = (Efficiency <= 0.9f) ? SS.TotalAccesses : 0;
    S.PerSite.push_back(SS);
    S.SiteMap.push_back(E.FirstSeenInfo);

    TotalSamples += E.TotalSamples;
    WeightedEfficiency += static_cast<double>(Efficiency) * E.TotalSamples;
  }

  S.NumSites = static_cast<uint32_t>(S.PerSite.size());
  S.TotalRecords = static_cast<uint32_t>(TotalSamples);
  S.OverallEfficiency = TotalSamples > 0
      ? static_cast<float>(WeightedEfficiency / TotalSamples) : 0.0f;
  return S;
}

LDSSummary canonicalToLDSSummary(const std::string &KernelName,
                                 const CanonicalKernelAccum &A) {
  LDSSummary S;
  S.KernelName = KernelName;

  double OverallSum = 0.0;
  uint64_t OverallCount = 0;

  for (const auto &[PC, E] : A.LDSByPC) {
    if (E.TotalSamples == 0)
      continue;
    float AvgConflictCycles =
        static_cast<float>(E.TotalUniqueBanks) /
        static_cast<float>(E.TotalSamples);

    LDSSiteStats SS;
    SS.SiteID = E.FirstSeenSiteID;
    SS.TotalAccesses = static_cast<uint32_t>(E.TotalSamples);
    SS.AvgConflictCycles = AvgConflictCycles;
    SS.MinConflictCycles = static_cast<uint32_t>(AvgConflictCycles);
    SS.MaxConflictCycles = static_cast<uint32_t>(AvgConflictCycles + 0.5f);
    SS.ConflictFreeCount = 0;
    SS.ConflictCount = 0;
    SS.HasPerSampleBreakdown = false;
    S.PerSite.push_back(SS);
    S.SiteMap.push_back(E.FirstSeenInfo);

    OverallSum += static_cast<double>(AvgConflictCycles) * E.TotalSamples;
    OverallCount += E.TotalSamples;
  }

  S.NumSites = static_cast<uint32_t>(S.PerSite.size());
  S.TotalRecords = static_cast<uint32_t>(OverallCount);
  S.OverallAvgConflictCycles = OverallCount > 0
      ? static_cast<float>(OverallSum / OverallCount) : 0.0f;
  return S;
}

} // namespace

void ProfilingResultsSink::flush(const std::string &Path) {
  if (Path.empty())
    return;

  std::unordered_map<std::string, CanonicalKernelAccum> LocalByKernel;
  std::vector<CoalescingSummary> LegacyVMEM;
  std::vector<LDSSummary> LegacyLDS;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    LocalByKernel.swap(ByKernel);
    LegacyVMEM.swap(VMEM);
    LegacyLDS.swap(LDS);
  }

  if (!LocalByKernel.empty()) {
    // Canonical path: one summary per kernel built from the per-PC table
    // (variant-local site indices already merged during ingest).
    std::vector<CoalescingSummary> VMEMResults;
    std::vector<LDSSummary> LDSResults;
    VMEMResults.reserve(LocalByKernel.size());
    LDSResults.reserve(LocalByKernel.size());
    for (const auto &[Name, Accum] : LocalByKernel) {
      if (!Accum.VMEMByPC.empty())
        VMEMResults.push_back(canonicalToVMEMSummary(Name, Accum));
      if (!Accum.LDSByPC.empty())
        LDSResults.push_back(canonicalToLDSSummary(Name, Accum));
    }
    CoalescingAnalyzer::writeJSON(Path, VMEMResults, LDSResults);
    return;
  }

  // Fallback for callers/tests that populated the legacy per-dispatch
  // summary vectors directly via `addVMEM` / `addLDS` without going through
  // `ingest`.  Preserves pre-canonicalization behavior.
  CoalescingAnalyzer::writeJSON(Path, LegacyVMEM, LegacyLDS);
}

void ProfilingResultsSink::clear() {
  std::lock_guard<std::mutex> Lock(Mutex);
  VMEM.clear();
  LDS.clear();
  ByKernel.clear();
}

const CanonicalKernelAccum *
ProfilingResultsSink::findCanonical(const std::string &KernelName) const {
  std::lock_guard<std::mutex> Lock(Mutex);
  auto It = ByKernel.find(KernelName);
  if (It == ByKernel.end())
    return nullptr;
  return &It->second;
}

} // namespace aegisbit
