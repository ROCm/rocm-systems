//===-- CoalescingAnalyzer.cpp - Memory Coalescing Analysis --------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/CoalescingAnalyzer.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <unordered_map>

namespace aegisbit {

namespace {
std::string readSourceLine(const std::string &File, uint32_t Line) {
  // Intentionally leaked to survive static destruction — the signal monitor
  // thread may still be running when atexit handlers destroy function-local
  // statics, causing a use-after-destroy crash.
  static auto &Cache =
      *new std::unordered_map<std::string, std::vector<std::string>>();
  auto &Lines = Cache[File];
  if (Lines.empty()) {
    std::ifstream In(File);
    if (!In.is_open()) return {};
    std::string L;
    while (std::getline(In, L))
      Lines.push_back(std::move(L));
  }
  if (Line == 0 || Line > Lines.size()) return {};
  const auto &L = Lines[Line - 1];
  auto Start = L.find_first_not_of(" \t");
  if (Start == std::string::npos) return {};
  return L.substr(Start);
}
} // anonymous namespace

uint32_t CoalescingAnalyzer::inferElemSize(const std::string &Name) {
  // DS instruction element sizes (must come before DWORD checks)
  if (Name.find("DS_") == 0) {
    if (Name.find("_B128") != std::string::npos) return 16;
    if (Name.find("_B96")  != std::string::npos) return 12;
    if (Name.find("_B64")  != std::string::npos) return 8;
    if (Name.find("_B32")  != std::string::npos) return 4;
    if (Name.find("_B16")  != std::string::npos ||
        Name.find("_U16")  != std::string::npos ||
        Name.find("_I16")  != std::string::npos) return 2;
    if (Name.find("_B8")   != std::string::npos ||
        Name.find("_U8")   != std::string::npos ||
        Name.find("_I8")   != std::string::npos) return 1;
    // DS atomics (DS_ADD_U32, DS_AND_B32, etc.) operate on 32-bit values
    if (Name.find("_U32")  != std::string::npos ||
        Name.find("_I32")  != std::string::npos ||
        Name.find("_F32")  != std::string::npos) return 4;
    if (Name.find("_U64")  != std::string::npos ||
        Name.find("_F64")  != std::string::npos) return 8;
    return 4; // safe default for DS
  }

  // Match most-specific patterns first (DWORDX4 before DWORD).
  // AMDGPU names: GLOBAL_LOAD_DWORDX4, BUFFER_STORE_DWORDX2_OFFEN_gfx90a, etc.
  if (Name.find("DWORDX4") != std::string::npos) return 16;
  if (Name.find("DWORDX3") != std::string::npos) return 12;
  if (Name.find("DWORDX2") != std::string::npos) return 8;
  if (Name.find("DWORD")   != std::string::npos) return 4;

  // 16-bit loads/stores: SHORT, D16, D16_HI
  if (Name.find("SHORT") != std::string::npos) return 2;
  if (Name.find("D16")   != std::string::npos) return 2;

  // 8-bit loads/stores: BYTE, UBYTE, SBYTE
  if (Name.find("BYTE") != std::string::npos) return 1;

  return 4; // safe default
}

AccessMetrics CoalescingAnalyzer::analyzeAccess(const uint64_t Addresses[64],
                                                 uint32_t SiteID,
                                                 uint32_t ElemSize) {
  AccessMetrics M;
  M.SiteID = SiteID;

  std::set<uint64_t> CacheLineSet;
  std::set<uint64_t> SectorSet;
  std::set<uint64_t> UniqueAddrSet;

  uint32_t ActiveLanes = 0;
  int64_t PrevAddr = -1;
  int64_t Stride = 0;
  bool StrideConsistent = true;
  bool FirstStride = true;

  for (int Lane = 0; Lane < 64; Lane++) {
    uint64_t Addr = Addresses[Lane];
    if (Addr == 0) continue;

    ActiveLanes++;
    UniqueAddrSet.insert(Addr);

    // Account for all cache lines and sectors spanned by this element.
    uint64_t FirstCL = Addr / CacheLineSize;
    uint64_t LastCL  = (Addr + ElemSize - 1) / CacheLineSize;
    for (uint64_t CL = FirstCL; CL <= LastCL; CL++)
      CacheLineSet.insert(CL);

    uint64_t FirstSec = Addr / SectorSize;
    uint64_t LastSec  = (Addr + ElemSize - 1) / SectorSize;
    for (uint64_t S = FirstSec; S <= LastSec; S++)
      SectorSet.insert(S);

    if (PrevAddr >= 0) {
      int64_t CurStride = static_cast<int64_t>(Addr) - PrevAddr;
      if (FirstStride) {
        Stride = CurStride;
        FirstStride = false;
      } else if (CurStride != Stride) {
        StrideConsistent = false;
      }
    }
    PrevAddr = static_cast<int64_t>(Addr);
  }

  M.ActiveLanes = ActiveLanes;
  M.CacheLines = static_cast<uint32_t>(CacheLineSet.size());
  M.Sectors = static_cast<uint32_t>(SectorSet.size());

  uint32_t UniqueAddrs = static_cast<uint32_t>(UniqueAddrSet.size());
  M.BytesRequested = UniqueAddrs * ElemSize;
  M.BytesFetched = M.CacheLines * CacheLineSize;
  M.Efficiency = M.BytesFetched > 0
                     ? static_cast<float>(M.BytesRequested) / M.BytesFetched
                     : 0.0f;

  if (ActiveLanes <= 1 || UniqueAddrs <= 1) {
    M.AccessPattern = AccessMetrics::Coalesced;
  } else if (StrideConsistent && Stride == static_cast<int64_t>(ElemSize)) {
    M.AccessPattern = AccessMetrics::Coalesced;
  } else if (StrideConsistent && Stride != 0) {
    M.AccessPattern = AccessMetrics::Strided;
  } else if (M.CacheLines <= 2) {
    M.AccessPattern = AccessMetrics::Coalesced;
  } else {
    M.AccessPattern = AccessMetrics::Scattered;
  }

  return M;
}

struct RawRecord {
  uint32_t SiteID;
  uint32_t Padding;
  uint64_t Addresses[64];
};
static_assert(sizeof(RawRecord) == TraceConfig::RecordSize);

std::vector<AccessMetrics>
CoalescingAnalyzer::analyzeBuffer(const void *Buffer, uint32_t NumRecords,
                                   uint32_t ElemSize) {
  std::vector<AccessMetrics> Results;
  Results.reserve(NumRecords);

  auto *Records = reinterpret_cast<const RawRecord *>(Buffer);
  for (uint32_t i = 0; i < NumRecords; i++) {
    Results.push_back(analyzeAccess(Records[i].Addresses, Records[i].SiteID, ElemSize));
  }

  return Results;
}

std::vector<AccessMetrics>
CoalescingAnalyzer::analyzeBuffer(const void *Buffer, uint32_t NumRecords,
                                   const std::vector<SiteInfo> &SiteMap) {
  // Build SiteID → ElemSize lookup.
  std::unordered_map<uint32_t, uint32_t> ElemSizeMap;
  for (const auto &SI : SiteMap)
    ElemSizeMap[SI.SiteID] = SI.ElemSize;

  std::vector<AccessMetrics> Results;
  Results.reserve(NumRecords);

  auto *Records = reinterpret_cast<const RawRecord *>(Buffer);
  for (uint32_t i = 0; i < NumRecords; i++) {
    uint32_t SiteID = Records[i].SiteID;
    auto It = ElemSizeMap.find(SiteID);
    uint32_t ES = (It != ElemSizeMap.end()) ? It->second : 4;
    Results.push_back(analyzeAccess(Records[i].Addresses, SiteID, ES));
  }

  return Results;
}

static CoalescingSummary
summarizeImpl(const std::vector<AccessMetrics> &Metrics) {
  CoalescingSummary S;
  S.TotalRecords = static_cast<uint32_t>(Metrics.size());

  std::map<uint32_t, std::vector<const AccessMetrics *>> BySite;
  for (const auto &M : Metrics) {
    BySite[M.SiteID].push_back(&M);
  }

  S.NumSites = static_cast<uint32_t>(BySite.size());

  for (auto &[ID, Recs] : BySite) {
    SiteStats SS;
    SS.SiteID = ID;
    SS.TotalAccesses = static_cast<uint32_t>(Recs.size());

    double SumCL = 0, SumSec = 0, SumEff = 0;

    for (const auto *M : Recs) {
      SumCL += M->CacheLines;
      SumSec += M->Sectors;
      SumEff += M->Efficiency;
      SS.MinCacheLines = std::min(SS.MinCacheLines, M->CacheLines);
      SS.MaxCacheLines = std::max(SS.MaxCacheLines, M->CacheLines);

      S.TotalBytesRequested += M->BytesRequested;
      S.TotalBytesFetched += M->BytesFetched;

      switch (M->AccessPattern) {
      case AccessMetrics::Coalesced: SS.CoalescedCount++; break;
      case AccessMetrics::Strided:   SS.StridedCount++;   break;
      case AccessMetrics::Scattered: SS.ScatteredCount++; break;
      }
    }

    SS.AvgCacheLines = static_cast<float>(SumCL / Recs.size());
    SS.AvgSectors = static_cast<float>(SumSec / Recs.size());
    SS.AvgEfficiency = static_cast<float>(SumEff / Recs.size());
    S.PerSite.push_back(SS);
  }

  S.OverallEfficiency = S.TotalBytesFetched > 0
      ? static_cast<float>(S.TotalBytesRequested) /
            static_cast<float>(S.TotalBytesFetched)
      : 0.0f;

  return S;
}

CoalescingSummary
CoalescingAnalyzer::summarize(const std::vector<AccessMetrics> &Metrics) {
  return summarizeImpl(Metrics);
}

CoalescingSummary
CoalescingAnalyzer::summarize(const std::vector<AccessMetrics> &Metrics,
                               const std::string &KernelName,
                               const std::vector<SiteInfo> &SiteMap) {
  CoalescingSummary S = summarizeImpl(Metrics);
  S.KernelName = KernelName;
  S.SiteMap = SiteMap;
  return S;
}

CoalescingSummary CoalescingAnalyzer::analyzeOnGpuCounters(
    const void *Buffer,
    const std::vector<SiteInfo> &SiteMap,
    const std::string &KernelName) {
  CoalescingSummary S;
  S.KernelName = KernelName;
  S.SiteMap = SiteMap;

  const auto *Counters = reinterpret_cast<const uint32_t *>(Buffer);

  uint32_t TotalSamples = 0;
  double WeightedEfficiency = 0;

  for (uint32_t i = 0; i < SiteMap.size(); ++i) {
    const auto &SI = SiteMap[i];
    if (SI.IsLDS) continue;

    uint32_t SiteID = SI.SiteID;
    uint32_t TotalCacheLines = Counters[SiteID * 2];
    uint32_t Samples = Counters[SiteID * 2 + 1];
    if (Samples == 0) continue;

    float AvgCacheLines = static_cast<float>(TotalCacheLines) / Samples;

    float IdealCacheLines = static_cast<float>(64 * SI.ElemSize) / 128.0f;
    if (IdealCacheLines < 1.0f) IdealCacheLines = 1.0f;

    float Efficiency = std::min(1.0f, IdealCacheLines / AvgCacheLines);

    SiteStats SS;
    SS.SiteID = SiteID;
    SS.TotalAccesses = Samples;
    SS.AvgEfficiency = Efficiency;
    SS.AvgCacheLines = AvgCacheLines;
    SS.MinCacheLines = static_cast<uint32_t>(AvgCacheLines);
    SS.MaxCacheLines = static_cast<uint32_t>(AvgCacheLines + 0.5f);
    SS.CoalescedCount = (Efficiency > 0.9f) ? Samples : 0;
    SS.StridedCount = 0;
    SS.ScatteredCount = (Efficiency <= 0.9f) ? Samples : 0;
    S.PerSite.push_back(SS);

    TotalSamples += Samples;
    WeightedEfficiency += Efficiency * Samples;
  }

  S.NumSites = static_cast<uint32_t>(S.PerSite.size());
  S.TotalRecords = TotalSamples;
  S.OverallEfficiency = TotalSamples > 0
      ? static_cast<float>(WeightedEfficiency / TotalSamples) : 0.0f;

  return S;
}

LDSSummary CoalescingAnalyzer::analyzeOnGpuCountersLDS(
    const void *Buffer,
    const std::vector<SiteInfo> &SiteMap,
    const std::string &KernelName) {
  LDSSummary S;
  S.KernelName = KernelName;
  S.SiteMap = SiteMap;

  const auto *Counters = reinterpret_cast<const uint32_t *>(Buffer);

  double OverallSum = 0;
  uint32_t OverallCount = 0;

  for (uint32_t i = 0; i < SiteMap.size(); ++i) {
    const auto &SI = SiteMap[i];
    if (!SI.IsLDS) continue;

    uint32_t SiteID = SI.SiteID;
    uint32_t TotalMaxLanes = Counters[SiteID * 2];
    uint32_t Samples = Counters[SiteID * 2 + 1];
    if (Samples == 0) continue;

    // The GPU payload now returns max-lanes-per-bank per sample.
    // The accumulated counter is the sum of that across all samples.
    // Average gives the mean conflict cycle count directly.
    float AvgConflictCycles = static_cast<float>(TotalMaxLanes) / Samples;

    LDSSiteStats SS;
    SS.SiteID = SiteID;
    SS.TotalAccesses = Samples;
    SS.AvgConflictCycles = AvgConflictCycles;
    SS.MinConflictCycles = static_cast<uint32_t>(AvgConflictCycles);
    SS.MaxConflictCycles = static_cast<uint32_t>(AvgConflictCycles + 0.5f);
    // With aggregated counters we only know the average N-way conflict
    // degree. Mark per-sample breakdown as unavailable (both zero) so the
    // report shows "N-way" style output instead of a misleading 0%.
    SS.ConflictFreeCount = 0;
    SS.ConflictCount = 0;
    SS.HasPerSampleBreakdown = false;
    S.PerSite.push_back(SS);

    OverallSum += static_cast<double>(AvgConflictCycles) * Samples;
    OverallCount += Samples;
  }

  S.NumSites = static_cast<uint32_t>(S.PerSite.size());
  S.TotalRecords = OverallCount;
  S.OverallAvgConflictCycles = OverallCount > 0
      ? static_cast<float>(OverallSum / OverallCount) : 0.0f;

  return S;
}

static const char *patternStr(AccessMetrics::Pattern P) {
  switch (P) {
  case AccessMetrics::Coalesced: return "coalesced";
  case AccessMetrics::Strided:   return "strided";
  case AccessMetrics::Scattered: return "SCATTERED";
  }
  return "unknown";
}

void CoalescingAnalyzer::printReport(const CoalescingSummary &Summary) {
  fprintf(stderr, "\n");
  if (!Summary.KernelName.empty())
    fprintf(stderr, "=== VMEM Coalescing: %s ===\n",
            Summary.KernelName.c_str());
  else
    fprintf(stderr, "=== VMEM Coalescing ===\n");
  fprintf(stderr, "Overall efficiency: %.1f%%  (%u sites, %u samples)\n\n",
          Summary.OverallEfficiency * 100.0f,
          Summary.NumSites, Summary.TotalRecords);

  std::map<uint32_t, const SiteInfo *> InfoMap;
  for (const auto &SI : Summary.SiteMap)
    InfoMap[SI.SiteID] = &SI;

  struct LineGroup {
    std::string SourceFile;
    std::string SourceFileFull;
    uint32_t SourceLine = 0;
    uint32_t NumInstructions = 0;
    uint32_t Loads = 0;
    uint32_t Stores = 0;
    double EfficiencySum = 0;
    double CacheLinesSum = 0;
    uint32_t ScatteredTotal = 0;
    uint32_t StridedTotal = 0;
    uint32_t CoalescedTotal = 0;
  };
  std::vector<LineGroup> Groups;
  std::map<std::string, size_t> GroupIdx;

  for (const auto &SS : Summary.PerSite) {
    const SiteInfo *Info = nullptr;
    auto It = InfoMap.find(SS.SiteID);
    if (It != InfoMap.end())
      Info = It->second;

    if (!Info) continue;

    std::string Key;
    if (Info->SourceLine != 0) {
      Key = Info->SourceFile + ":" + std::to_string(Info->SourceLine);
    } else {
      Key = std::string(Info->InstrName) + " @ 0x" +
            std::to_string(Info->PC);
    }

    auto [GIt, Inserted] = GroupIdx.try_emplace(Key, Groups.size());
    if (Inserted) {
      Groups.push_back({});
      Groups.back().SourceFile = Info->SourceFile;
      Groups.back().SourceFileFull = Info->SourceFileFull;
      Groups.back().SourceLine = Info->SourceLine;
    }
    auto &G = Groups[GIt->second];
    G.NumInstructions++;
    if (Info->IsLoad) G.Loads++;
    else G.Stores++;
    G.EfficiencySum += SS.AvgEfficiency;
    G.CacheLinesSum += SS.AvgCacheLines;
    G.ScatteredTotal += SS.ScatteredCount;
    G.StridedTotal += SS.StridedCount;
    G.CoalescedTotal += SS.CoalescedCount;
  }

  for (const auto &G : Groups) {
    const char *Pattern = "coalesced";
    if (G.ScatteredTotal > G.CoalescedTotal && G.ScatteredTotal > G.StridedTotal)
      Pattern = "SCATTERED";
    else if (G.StridedTotal > G.CoalescedTotal)
      Pattern = "strided";

    float AvgEff = G.NumInstructions > 0
        ? static_cast<float>(G.EfficiencySum / G.NumInstructions) : 0;
    float AvgCL = G.NumInstructions > 0
        ? static_cast<float>(G.CacheLinesSum / G.NumInstructions) : 0;

    const char *Kind = "load";
    if (G.Stores > 0 && G.Loads > 0) Kind = "load+store";
    else if (G.Stores > 0) Kind = "store";

    std::string Code = readSourceLine(G.SourceFileFull, G.SourceLine);
    std::string Label = !Code.empty() ? Code
        : G.SourceFile + ":" + std::to_string(G.SourceLine);

    fprintf(stderr, "  %-50s %u×%-5s eff=%-4.0f%%  cachelines=%-2.0f  %s\n",
            Label.c_str(), G.NumInstructions, Kind,
            AvgEff * 100.0f, AvgCL, Pattern);
  }
  fprintf(stderr, "\n");
}

//===----------------------------------------------------------------------===//
// LDS bank conflict analysis
//===----------------------------------------------------------------------===//

LDSAccessMetrics CoalescingAnalyzer::analyzeLDSAccess(
    const uint64_t Addresses[64], uint32_t SiteID,
    uint32_t ElemSize, uint16_t DSOffset) {
  LDSAccessMetrics M;
  M.SiteID = SiteID;

  // bank -> set of unique addresses in that bank
  std::set<uint32_t> BankAddrs[LDSAccessMetrics::NumBanks];
  uint32_t ActiveLanes = 0;

  for (int Lane = 0; Lane < 64; Lane++) {
    uint64_t Raw = Addresses[Lane];
    if (Raw == 0) continue;
    ActiveLanes++;

    uint32_t Addr = static_cast<uint32_t>(Raw) + DSOffset;
    uint32_t Bank = (Addr / LDSAccessMetrics::BankWidth) % LDSAccessMetrics::NumBanks;
    BankAddrs[Bank].insert(Addr);
  }

  M.ActiveLanes = ActiveLanes;
  uint32_t MaxUniquePerBank = 0;
  uint32_t BanksUsed = 0;
  for (unsigned B = 0; B < LDSAccessMetrics::NumBanks; B++) {
    if (!BankAddrs[B].empty()) {
      BanksUsed++;
      MaxUniquePerBank = std::max(MaxUniquePerBank,
                                   static_cast<uint32_t>(BankAddrs[B].size()));
    }
  }

  M.ConflictCycles = std::max(1u, MaxUniquePerBank);
  M.BanksUsed = BanksUsed;

  uint32_t TotalUniquePerBank = 0;
  for (unsigned B = 0; B < LDSAccessMetrics::NumBanks; B++)
    TotalUniquePerBank += static_cast<uint32_t>(BankAddrs[B].size());
  M.BroadcastLanes = (ActiveLanes > TotalUniquePerBank)
                          ? ActiveLanes - TotalUniquePerBank : 0;

  return M;
}

std::vector<LDSAccessMetrics>
CoalescingAnalyzer::analyzeLDSBuffer(const void *Buffer, uint32_t NumRecords,
                                      const std::vector<SiteInfo> &SiteMap) {
  std::unordered_map<uint32_t, const SiteInfo *> LDSSiteMap;
  for (const auto &SI : SiteMap)
    if (SI.IsLDS)
      LDSSiteMap[SI.SiteID] = &SI;

  std::vector<LDSAccessMetrics> Results;
  auto *Records = reinterpret_cast<const RawRecord *>(Buffer);
  for (uint32_t i = 0; i < NumRecords; i++) {
    uint32_t SiteID = Records[i].SiteID;
    auto It = LDSSiteMap.find(SiteID);
    if (It == LDSSiteMap.end())
      continue; // skip non-LDS records
    const SiteInfo *SI = It->second;
    Results.push_back(analyzeLDSAccess(Records[i].Addresses, SiteID,
                                        SI->ElemSize, SI->DSOffset0));
  }

  return Results;
}

LDSSummary
CoalescingAnalyzer::summarizeLDS(const std::vector<LDSAccessMetrics> &Metrics,
                                  const std::string &KernelName,
                                  const std::vector<SiteInfo> &SiteMap) {
  LDSSummary S;
  S.KernelName = KernelName;
  S.TotalRecords = static_cast<uint32_t>(Metrics.size());
  S.SiteMap = SiteMap;

  std::map<uint32_t, std::vector<const LDSAccessMetrics *>> BySite;
  for (const auto &M : Metrics)
    BySite[M.SiteID].push_back(&M);

  S.NumSites = static_cast<uint32_t>(BySite.size());

  double OverallSum = 0;
  uint32_t OverallCount = 0;

  for (auto &[ID, Recs] : BySite) {
    LDSSiteStats SS;
    SS.SiteID = ID;
    SS.TotalAccesses = static_cast<uint32_t>(Recs.size());

    double SumCycles = 0;
    for (const auto *M : Recs) {
      SumCycles += M->ConflictCycles;
      SS.MinConflictCycles = std::min(SS.MinConflictCycles, M->ConflictCycles);
      SS.MaxConflictCycles = std::max(SS.MaxConflictCycles, M->ConflictCycles);
      if (M->ConflictCycles == 1)
        SS.ConflictFreeCount++;
      else
        SS.ConflictCount++;
    }

    SS.AvgConflictCycles = static_cast<float>(SumCycles / Recs.size());
    OverallSum += SumCycles;
    OverallCount += SS.TotalAccesses;
    S.PerSite.push_back(SS);
  }

  S.OverallAvgConflictCycles = OverallCount > 0
      ? static_cast<float>(OverallSum / OverallCount) : 0.0f;

  return S;
}

void CoalescingAnalyzer::printLDSReport(const LDSSummary &Summary) {
  fprintf(stderr, "\n");
  if (!Summary.KernelName.empty())
    fprintf(stderr, "=== LDS Bank Conflicts: %s ===\n",
            Summary.KernelName.c_str());
  else
    fprintf(stderr, "=== LDS Bank Conflicts ===\n");
  fprintf(stderr, "%u sites, %u samples\n\n",
          Summary.NumSites, Summary.TotalRecords);

  std::map<uint32_t, const SiteInfo *> InfoMap;
  for (const auto &SI : Summary.SiteMap)
    InfoMap[SI.SiteID] = &SI;

  struct LineGroup {
    std::string SourceFile;
    std::string SourceFileFull;
    uint32_t SourceLine = 0;
    uint32_t NumInstructions = 0;
    uint32_t Loads = 0;
    uint32_t Stores = 0;
    double ConflictCyclesSum = 0;
    uint32_t ConflictCount = 0;
    uint32_t ConflictFreeCount = 0;
    uint32_t TotalAccesses = 0;
    bool HasPerSampleBreakdown = true;
  };
  std::vector<LineGroup> Groups;
  std::map<std::string, size_t> GroupIdx;

  for (const auto &SS : Summary.PerSite) {
    const SiteInfo *Info = nullptr;
    auto It = InfoMap.find(SS.SiteID);
    if (It != InfoMap.end())
      Info = It->second;

    std::string Key;
    if (Info && Info->SourceLine != 0) {
      Key = Info->SourceFile + ":" + std::to_string(Info->SourceLine);
    } else if (Info) {
      Key = std::string(Info->InstrName) + " @ 0x" +
            std::to_string(Info->PC);
    } else {
      continue;
    }

    auto [GIt, Inserted] = GroupIdx.try_emplace(Key, Groups.size());
    if (Inserted) {
      Groups.push_back({});
      Groups.back().SourceFile = Info ? Info->SourceFile : "";
      Groups.back().SourceFileFull = Info ? Info->SourceFileFull : "";
      Groups.back().SourceLine = Info ? Info->SourceLine : 0;
    }
    auto &G = Groups[GIt->second];
    G.NumInstructions++;
    if (Info && Info->IsLoad) G.Loads++;
    else G.Stores++;
    G.ConflictCyclesSum += SS.AvgConflictCycles;
    G.ConflictCount += SS.ConflictCount;
    G.ConflictFreeCount += SS.ConflictFreeCount;
    G.TotalAccesses += SS.TotalAccesses;
    if (!SS.HasPerSampleBreakdown) G.HasPerSampleBreakdown = false;
  }

  for (const auto &G : Groups) {
    float AvgNWay = G.NumInstructions > 0
        ? static_cast<float>(G.ConflictCyclesSum / G.NumInstructions) : 0;

    const char *Kind = "load";
    if (G.Stores > 0 && G.Loads > 0) Kind = "load+store";
    else if (G.Stores > 0) Kind = "store";

    std::string Code = readSourceLine(G.SourceFileFull, G.SourceLine);
    std::string Label = !Code.empty() ? Code
        : G.SourceFile + ":" + std::to_string(G.SourceLine);

    fprintf(stderr, "  %-50s %u×%-5s", Label.c_str(), G.NumInstructions, Kind);
    if (AvgNWay <= 1.0f) {
      fprintf(stderr, "  no conflicts\n");
    } else if (G.HasPerSampleBreakdown) {
      float ConflictFreePct = G.TotalAccesses > 0
          ? 100.0f * G.ConflictFreeCount / G.TotalAccesses : 0.0f;
      fprintf(stderr, "  avg_n-way=%.1f  conflict_free=%.0f%%\n",
              AvgNWay, ConflictFreePct);
    } else {
      fprintf(stderr, "  avg_n-way=%.1f\n", AvgNWay);
    }
  }
  fprintf(stderr, "\n");
}

//===----------------------------------------------------------------------===//
// JSON output
//===----------------------------------------------------------------------===//

namespace {

/// Escape a string for JSON output (handles \, ", and control characters).
std::string jsonEscape(const std::string &S) {
  std::string Out;
  Out.reserve(S.size() + 8);
  for (char C : S) {
    switch (C) {
    case '"':  Out += "\\\""; break;
    case '\\': Out += "\\\\"; break;
    case '\n': Out += "\\n";  break;
    case '\r': Out += "\\r";  break;
    case '\t': Out += "\\t";  break;
    default:
      if (static_cast<unsigned char>(C) < 0x20) {
        char Buf[8];
        snprintf(Buf, sizeof(Buf), "\\u%04x", static_cast<unsigned>(C));
        Out += Buf;
      } else {
        Out += C;
      }
    }
  }
  return Out;
}

struct JSONLineGroup {
  std::string SourceFile;
  std::string SourceFileFull;
  uint32_t SourceLine = 0;
  std::string InstrName;
  bool IsLoad = true;
  uint32_t NumInstructions = 0;
};

struct VMEMLineGroup : JSONLineGroup {
  double EfficiencySum = 0;
  double CacheLinesSum = 0;
  uint32_t ScatteredTotal = 0;
  uint32_t StridedTotal = 0;
  uint32_t CoalescedTotal = 0;
};

struct LDSLineGroup : JSONLineGroup {
  double ConflictCyclesSum = 0;
  uint32_t ConflictFreeCount = 0;
  uint32_t TotalAccesses = 0;
  bool HasPerSampleBreakdown = true;
};

} // anonymous namespace

void CoalescingAnalyzer::writeJSON(
    const std::string &Path,
    const std::vector<CoalescingSummary> &VMEMResults,
    const std::vector<LDSSummary> &LDSResults) {

  FILE *F = fopen(Path.c_str(), "w");
  if (!F) {
    fprintf(stderr, "[aegisbit] WARNING: cannot open JSON output file: %s\n",
            Path.c_str());
    return;
  }

  // Build a unified kernel list: merge VMEM and LDS results by kernel name.
  std::map<std::string, std::pair<const CoalescingSummary *, const LDSSummary *>>
      KernelMap;
  for (const auto &V : VMEMResults)
    KernelMap[V.KernelName].first = &V;
  for (const auto &L : LDSResults)
    KernelMap[L.KernelName].second = &L;

  fprintf(F, "{\n  \"version\": 1,\n  \"kernels\": [\n");

  bool FirstKernel = true;
  for (const auto &[Name, Pair] : KernelMap) {
    if (!FirstKernel) fprintf(F, ",\n");
    FirstKernel = false;

    fprintf(F, "    {\n      \"name\": \"%s\"", jsonEscape(Name).c_str());

    // --- VMEM coalescing ---
    if (Pair.first) {
      const auto &S = *Pair.first;

      std::map<uint32_t, const SiteInfo *> InfoMap;
      for (const auto &SI : S.SiteMap)
        InfoMap[SI.SiteID] = &SI;

      // Source-line grouping (same logic as printReport)
      std::vector<VMEMLineGroup> Groups;
      std::map<std::string, size_t> GIdx;

      for (const auto &SS : S.PerSite) {
        auto It = InfoMap.find(SS.SiteID);
        if (It == InfoMap.end()) continue;
        const SiteInfo *Info = It->second;

        std::string Key;
        if (Info->SourceLine != 0)
          Key = Info->SourceFile + ":" + std::to_string(Info->SourceLine);
        else
          Key = Info->InstrName + " @ 0x" + std::to_string(Info->PC);

        auto [GIt, Inserted] = GIdx.try_emplace(Key, Groups.size());
        if (Inserted) {
          VMEMLineGroup G;
          G.SourceFile = Info->SourceFile;
          G.SourceFileFull = Info->SourceFileFull;
          G.SourceLine = Info->SourceLine;
          G.InstrName = Info->InstrName;
          G.IsLoad = Info->IsLoad;
          Groups.push_back(G);
        }
        auto &G = Groups[GIt->second];
        G.NumInstructions++;
        if (!Info->IsLoad) G.IsLoad = false;
        G.EfficiencySum += SS.AvgEfficiency;
        G.CacheLinesSum += SS.AvgCacheLines;
        G.ScatteredTotal += SS.ScatteredCount;
        G.StridedTotal += SS.StridedCount;
        G.CoalescedTotal += SS.CoalescedCount;
      }

      fprintf(F, ",\n      \"vmem_coalescing\": {\n");
      fprintf(F, "        \"overall_efficiency_pct\": %.1f,\n",
              S.OverallEfficiency * 100.0f);
      fprintf(F, "        \"num_sites\": %u,\n", S.NumSites);
      fprintf(F, "        \"total_samples\": %u,\n", S.TotalRecords);
      fprintf(F, "        \"sites\": [\n");

      for (size_t i = 0; i < Groups.size(); ++i) {
        const auto &G = Groups[i];
        float AvgEff = G.NumInstructions > 0
            ? static_cast<float>(G.EfficiencySum / G.NumInstructions) : 0;
        float AvgCL = G.NumInstructions > 0
            ? static_cast<float>(G.CacheLinesSum / G.NumInstructions) : 0;

        const char *Pattern = "coalesced";
        if (G.ScatteredTotal > G.CoalescedTotal &&
            G.ScatteredTotal > G.StridedTotal)
          Pattern = "scattered";
        else if (G.StridedTotal > G.CoalescedTotal)
          Pattern = "strided";

        std::string Code = readSourceLine(G.SourceFileFull, G.SourceLine);

        if (i > 0) fprintf(F, ",\n");
        fprintf(F, "          {\n");
        fprintf(F, "            \"site_id\": %zu,\n", i);
        fprintf(F, "            \"source_file\": \"%s\",\n",
                jsonEscape(G.SourceFile).c_str());
        fprintf(F, "            \"source_line\": %u,\n", G.SourceLine);
        fprintf(F, "            \"source_text\": \"%s\",\n",
                jsonEscape(Code).c_str());
        fprintf(F, "            \"instruction\": \"%s\",\n",
                jsonEscape(G.InstrName).c_str());
        fprintf(F, "            \"is_load\": %s,\n",
                G.IsLoad ? "true" : "false");
        fprintf(F, "            \"num_instructions\": %u,\n",
                G.NumInstructions);
        fprintf(F, "            \"avg_efficiency_pct\": %.1f,\n",
                AvgEff * 100.0f);
        fprintf(F, "            \"avg_cachelines\": %.1f,\n", AvgCL);
        fprintf(F, "            \"pattern\": \"%s\"\n", Pattern);
        fprintf(F, "          }");
      }

      fprintf(F, "\n        ]\n      }");
    }

    // --- LDS bank conflicts ---
    if (Pair.second) {
      const auto &S = *Pair.second;

      std::map<uint32_t, const SiteInfo *> InfoMap;
      for (const auto &SI : S.SiteMap)
        InfoMap[SI.SiteID] = &SI;

      // Source-line grouping (same logic as printLDSReport)
      std::vector<LDSLineGroup> Groups;
      std::map<std::string, size_t> GIdx;

      for (const auto &SS : S.PerSite) {
        auto It = InfoMap.find(SS.SiteID);
        if (It == InfoMap.end()) continue;
        const SiteInfo *Info = It->second;

        std::string Key;
        if (Info->SourceLine != 0)
          Key = Info->SourceFile + ":" + std::to_string(Info->SourceLine);
        else
          Key = Info->InstrName + " @ 0x" + std::to_string(Info->PC);

        auto [GIt, Inserted] = GIdx.try_emplace(Key, Groups.size());
        if (Inserted) {
          LDSLineGroup G;
          G.SourceFile = Info->SourceFile;
          G.SourceFileFull = Info->SourceFileFull;
          G.SourceLine = Info->SourceLine;
          G.InstrName = Info->InstrName;
          G.IsLoad = Info->IsLoad;
          Groups.push_back(G);
        }
        auto &G = Groups[GIt->second];
        G.NumInstructions++;
        if (!Info->IsLoad) G.IsLoad = false;
        G.ConflictCyclesSum += SS.AvgConflictCycles;
        G.ConflictFreeCount += SS.ConflictFreeCount;
        G.TotalAccesses += SS.TotalAccesses;
        if (!SS.HasPerSampleBreakdown) G.HasPerSampleBreakdown = false;
      }

      fprintf(F, ",\n      \"lds_bank_conflicts\": {\n");
      fprintf(F, "        \"num_sites\": %u,\n", S.NumSites);
      fprintf(F, "        \"total_samples\": %u,\n", S.TotalRecords);
      fprintf(F, "        \"overall_avg_n_way\": %.1f,\n",
              S.OverallAvgConflictCycles);
      fprintf(F, "        \"sites\": [\n");

      for (size_t i = 0; i < Groups.size(); ++i) {
        const auto &G = Groups[i];
        float AvgNWay = G.NumInstructions > 0
            ? static_cast<float>(G.ConflictCyclesSum / G.NumInstructions) : 0;

        std::string Code = readSourceLine(G.SourceFileFull, G.SourceLine);

        if (i > 0) fprintf(F, ",\n");
        fprintf(F, "          {\n");
        fprintf(F, "            \"site_id\": %zu,\n", i);
        fprintf(F, "            \"source_file\": \"%s\",\n",
                jsonEscape(G.SourceFile).c_str());
        fprintf(F, "            \"source_line\": %u,\n", G.SourceLine);
        fprintf(F, "            \"source_text\": \"%s\",\n",
                jsonEscape(Code).c_str());
        fprintf(F, "            \"instruction\": \"%s\",\n",
                jsonEscape(G.InstrName).c_str());
        fprintf(F, "            \"is_load\": %s,\n",
                G.IsLoad ? "true" : "false");
        fprintf(F, "            \"num_instructions\": %u,\n",
                G.NumInstructions);
        fprintf(F, "            \"avg_n_way\": %.1f,\n", AvgNWay);
        if (G.HasPerSampleBreakdown) {
          float ConflictFreePct = G.TotalAccesses > 0
              ? 100.0f * G.ConflictFreeCount / G.TotalAccesses : 0.0f;
          fprintf(F, "            \"conflict_free_pct\": %.1f\n",
                  ConflictFreePct);
        } else {
          fprintf(F, "            \"conflict_free_pct\": null\n");
        }
        fprintf(F, "          }");
      }

      fprintf(F, "\n        ]\n      }");
    }

    fprintf(F, "\n    }");
  }

  fprintf(F, "\n  ]\n}\n");
  fclose(F);
}

} // namespace aegisbit
