//===-- aegisbit/CoalescingAnalyzer.h - Memory Coalescing Analysis *- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Analyzes per-lane memory addresses captured by the instrumented trampoline
/// to compute memory coalescing and bank conflict metrics.
///
/// AMDGPU memory hierarchy:
///   L2 cache line = 128 bytes (4 sectors × 32 bytes)
///   A wavefront (64 lanes) accessing consecutive 4-byte elements = 256 bytes
///     = 2 cache lines, 8 sectors → perfectly coalesced
///
/// Metrics:
///   - Cache lines touched per access (fewer = better coalescing)
///   - Sectors touched per access (directly maps to memory transactions)
///   - Coalescing efficiency = useful bytes / fetched bytes
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_COALESCING_ANALYZER_H
#define AEGISBIT_COALESCING_ANALYZER_H

#include "aegisbit/Types.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace aegisbit {

/// Per-access coalescing metrics for a single trace record.
struct AccessMetrics {
  uint32_t SiteID = 0;          ///< Instrumentation site ID
  uint32_t ActiveLanes = 0;     ///< Number of lanes with non-zero addresses
  uint32_t CacheLines = 0;      ///< Unique 128-byte cache lines touched
  uint32_t Sectors = 0;         ///< Unique 32-byte sectors touched
  uint32_t BytesRequested = 0;  ///< Useful bytes = UniqueAddrs × element_size
  uint32_t BytesFetched = 0;    ///< Total bytes fetched = CacheLines × 128
  float Efficiency = 0.0f;      ///< BytesRequested / BytesFetched (0.0 - 1.0)

  /// Access pattern classification.
  enum Pattern { Coalesced, Strided, Scattered };
  Pattern AccessPattern = Scattered;
};

/// Aggregated coalescing statistics per instrumentation site.
struct SiteStats {
  uint32_t SiteID = 0;
  uint32_t TotalAccesses = 0;

  float AvgCacheLines = 0.0f;
  float AvgSectors = 0.0f;
  float AvgEfficiency = 0.0f;

  uint32_t MinCacheLines = UINT32_MAX;
  uint32_t MaxCacheLines = 0;

  uint32_t CoalescedCount = 0;
  uint32_t StridedCount = 0;
  uint32_t ScatteredCount = 0;
};

/// Metadata mapping a site ID to the originating instruction.
struct SiteInfo {
  uint32_t SiteID = 0;
  uint64_t PC = 0;             ///< Program counter in the patched kernel
                               ///< (post-padding; stable across one ELF)
  /// Byte offset of the instrumented instruction within the *original*
  /// kernel (before any pre-kernel padding / prologue shift).  Stable
  /// across patched-ELF variants of the same kernel and used as the
  /// canonical identifier for instrumentation replay's `ExcludedPCs` set.
  uint64_t OriginalPC = 0;
  std::string InstrName;       ///< e.g. "GLOBAL_LOAD_DWORDX2"
  bool IsLoad = true;          ///< true = load, false = store
  uint32_t ElemSize = 4;       ///< Element size in bytes (inferred from instruction)

  bool IsLDS = false;          ///< true for DS (LDS) instructions
  uint16_t DSOffset0 = 0;     ///< Constant byte offset from DS instruction
  uint16_t DSOffset1 = 0;     ///< Second offset for DS_*2 dual-address instructions
  bool IsDualDS = false;       ///< true for DS_READ2/DS_WRITE2

  std::string SourceFile;      ///< Short filename (basename, from DWARF)
  std::string SourceFileFull;  ///< Full path (from DWARF, for source reading)
  uint32_t SourceLine = 0;     ///< Source line (from DWARF, 0 if unavailable)
  uint32_t SourceColumn = 0;   ///< Source column (from DWARF, 0 if unavailable)
};

/// Per-access LDS bank conflict metrics.
struct LDSAccessMetrics {
  uint32_t SiteID = 0;
  uint32_t ActiveLanes = 0;
  uint32_t ConflictCycles = 0;     ///< Max unique addresses in any single bank
  uint32_t BanksUsed = 0;          ///< Number of distinct banks accessed
  uint32_t BroadcastLanes = 0;     ///< Lanes served by broadcast (duplicate addr in same bank)

  static constexpr uint32_t NumBanks = 32;
  static constexpr uint32_t BankWidth = 4; // bytes
};

/// Aggregated LDS bank conflict stats per site.
struct LDSSiteStats {
  uint32_t SiteID = 0;
  uint32_t TotalAccesses = 0;
  float AvgConflictCycles = 0.0f;    ///< Average N-way conflict (1.0 = no conflicts)
  uint32_t MinConflictCycles = UINT32_MAX;
  uint32_t MaxConflictCycles = 0;
  uint32_t ConflictFreeCount = 0;    ///< Accesses with ConflictCycles == 1
  uint32_t ConflictCount = 0;        ///< Accesses with ConflictCycles > 1
  bool HasPerSampleBreakdown = true; ///< false when only aggregate counters are available
};

/// LDS bank conflict summary for a kernel.
struct LDSSummary {
  std::string KernelName;
  uint32_t TotalRecords = 0;
  uint32_t NumSites = 0;
  float OverallAvgConflictCycles = 0.0f;
  std::vector<LDSSiteStats> PerSite;
  std::vector<SiteInfo> SiteMap;
};

/// Overall kernel coalescing summary.
struct CoalescingSummary {
  std::string KernelName;
  uint32_t TotalRecords = 0;
  uint32_t NumSites = 0;
  float OverallEfficiency = 0.0f;
  uint64_t TotalBytesRequested = 0;
  uint64_t TotalBytesFetched = 0;
  std::vector<SiteStats> PerSite;
  std::vector<SiteInfo> SiteMap;
};

/// Analyzes per-lane memory addresses from the instrumented trampoline's
/// trace buffer to compute memory coalescing metrics.
class CoalescingAnalyzer {
public:
  /// AMDGPU L2 cache line size.
  static constexpr uint32_t CacheLineSize = 128;

  /// AMDGPU L2 sector size (minimum fetch granularity).
  static constexpr uint32_t SectorSize = 32;

  /// Analyze a single trace record.
  /// \param Addresses Array of 64 per-lane addresses from one wavefront access.
  /// \param SiteID    Instrumentation site identifier.
  /// \param ElemSize  Element size in bytes (4 for dword, 8 for dwordx2, etc.)
  static AccessMetrics analyzeAccess(const uint64_t Addresses[64],
                                     uint32_t SiteID,
                                     uint32_t ElemSize = 4);

  /// Analyze an entire trace buffer (uniform element size).
  /// \param Buffer    Pointer to trace records (TraceConfig::RecordSize each).
  /// \param NumRecords Number of records in the buffer.
  /// \param ElemSize  Element size in bytes for all accesses.
  /// \return Per-record metrics.
  static std::vector<AccessMetrics> analyzeBuffer(
      const void *Buffer, uint32_t NumRecords, uint32_t ElemSize = 4);

  /// Analyze an entire trace buffer with per-site element sizes.
  /// Each record's SiteID is looked up in \p SiteMap to determine ElemSize.
  static std::vector<AccessMetrics> analyzeBuffer(
      const void *Buffer, uint32_t NumRecords,
      const std::vector<SiteInfo> &SiteMap);

  /// Infer element size in bytes from an AMDGPU instruction name.
  ///
  /// Examples:
  ///   GLOBAL_LOAD_DWORD        → 4
  ///   GLOBAL_LOAD_DWORDX2      → 8
  ///   BUFFER_STORE_DWORDX4     → 16
  ///   GLOBAL_LOAD_USHORT       → 2
  ///   GLOBAL_LOAD_UBYTE        → 1
  ///   GLOBAL_LOAD_SHORT_D16    → 2
  static uint32_t inferElemSize(const std::string &InstrName);

  /// Compute aggregated per-site statistics from per-record metrics.
  static CoalescingSummary summarize(const std::vector<AccessMetrics> &Metrics);

  /// Compute aggregated per-site statistics, attaching instruction metadata.
  static CoalescingSummary summarize(const std::vector<AccessMetrics> &Metrics,
                                     const std::string &KernelName,
                                     const std::vector<SiteInfo> &SiteMap);

  /// Print a human-readable report to stderr.
  static void printReport(const CoalescingSummary &Summary);

  /// Analyze on-GPU counter buffer from OnGpuReduce strategy.
  /// Buffer layout: per site_id, 8 bytes = {total_cache_lines u32, total_samples u32}.
  /// Returns a CoalescingSummary with efficiency computed from counter data.
  static CoalescingSummary analyzeOnGpuCounters(
      const void *Buffer,
      const std::vector<SiteInfo> &SiteMap,
      const std::string &KernelName);

  /// Analyze on-GPU reduced LDS counters for bank conflicts.
  /// Buffer layout matches analyzeOnGpuCounters: per site_id,
  /// 8 bytes = {total_unique_banks u32, total_samples u32}.
  /// Returns an LDSSummary with conflict metrics estimated from unique bank counts.
  static LDSSummary analyzeOnGpuCountersLDS(
      const void *Buffer,
      const std::vector<SiteInfo> &SiteMap,
      const std::string &KernelName);

  /// Analyze a single LDS trace record for bank conflicts.
  /// \param Addresses  64 per-lane addresses (low 32 bits = LDS byte offset)
  /// \param SiteID     Site identifier
  /// \param ElemSize   Element size in bytes
  /// \param DSOffset   Constant offset added to each address (from DS instruction encoding)
  static LDSAccessMetrics analyzeLDSAccess(const uint64_t Addresses[64],
                                            uint32_t SiteID,
                                            uint32_t ElemSize = 4,
                                            uint16_t DSOffset = 0);

  /// Analyze LDS records from a trace buffer using per-site metadata.
  static std::vector<LDSAccessMetrics> analyzeLDSBuffer(
      const void *Buffer, uint32_t NumRecords,
      const std::vector<SiteInfo> &SiteMap);

  /// Summarize LDS bank conflict metrics.
  static LDSSummary summarizeLDS(const std::vector<LDSAccessMetrics> &Metrics,
                                  const std::string &KernelName,
                                  const std::vector<SiteInfo> &SiteMap);

  /// Print LDS bank conflict report to stderr.
  static void printLDSReport(const LDSSummary &Summary);

  /// Write all profiling results as structured JSON to a file.
  /// Combines VMEM coalescing and LDS bank conflict data for all kernels.
  static void writeJSON(const std::string &Path,
                        const std::vector<CoalescingSummary> &VMEMResults,
                        const std::vector<LDSSummary> &LDSResults);
};

} // namespace aegisbit

#endif // AEGISBIT_COALESCING_ANALYZER_H
