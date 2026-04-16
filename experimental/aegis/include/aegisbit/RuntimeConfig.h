//===-- aegisbit/RuntimeConfig.h - Runtime Configuration --------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Runtime configuration for AegisBit tracing.
/// Parses environment variables to control tracing behavior and low-level
/// transform/debug knobs. All AEGISBIT_* environment variables read by the
/// library are declared here and parsed once in RuntimeConfig::initialize().
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_RUNTIME_CONFIG_H
#define AEGISBIT_RUNTIME_CONFIG_H

#include "aegisbit/Types.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace aegisbit {

/// Debug / diagnostic knobs.
/// All fields default to "no-effect" (log off, dumps disabled, no limits,
/// sentinel -1/INT_MAX/UINT32_MAX for integer filters).
struct DebugFlags {
  /// AEGISBIT_LOG: 0 = off, 1 = info, 2 = verbose. Any non-empty
  /// non-"0"/"false" value sets LogEnabled=true.
  int LogLevel = 0;

  /// AEGISBIT_DUMP_INPUT_ELF: when non-empty, the input code object is
  /// written to "<value>_<KernelName>.bin" (plus a .meta sidecar) before
  /// patching. Empty = disabled.
  std::string DumpInputELFPrefix;

  /// AEGISBIT_DUMP_ELF: when non-empty, the patched ELF is written to this
  /// path. Empty = disabled.
  std::string DumpELFPath;

  /// AEGISBIT_DUMP_BLOBS: when non-empty, payload blobs are written into
  /// this directory. Empty = disabled.
  std::string DumpBlobsDir;

  /// AEGISBIT_MAX_SITES: cap on instrumentation sites per kernel.
  /// 0 or unset = no cap.
  uint32_t MaxSites = 0;

  /// AEGISBIT_MAX_LDS: cap on LDS sites per kernel.
  /// 0 or unset = no cap.
  uint32_t MaxLDSSites = 0;

  /// AEGISBIT_SKIP_LDS_FIRST: skip the first N LDS sites per kernel.
  int SkipLDSFirst = 0;

  /// AEGISBIT_SKIP_SIGNAL: diagnostic — do not forward completion signals.
  bool SkipSignal = false;

  /// AEGISBIT_GPU_FILTER: select a specific GPU agent by index.
  /// Negative = no filter.
  int GPUFilter = -1;

  /// AEGISBIT_BODY_SITE_ONLY: when >= 0, only this site has a body island
  /// (debug bisection).
  int BodySiteOnly = -1;

  /// AEGISBIT_MAX_BODY_SITES: cap on the number of sites that receive a
  /// body island. UINT32_MAX = unlimited.
  uint32_t MaxBodySites = UINT32_MAX;

  /// AEGISBIT_DRY_PAYLOAD: payload dry-run level (99 = full payload).
  int DryPayloadLevel = 99;

  /// AEGISBIT_COUNT_MODE: counting-loop variant selector (99 = default).
  int CountMode = 99;

  /// AEGISBIT_REPLAY: instrumentation replay — build multiple patched-ELF
  /// variants per kernel and rotate them across dispatches to improve
  /// coverage on large kernels.
  ///   0 / unset — off (single-variant legacy path).
  ///   N  (>0)   — always build exactly `N` variants.
  /// Populated by the env-var parser; see phase 5 of the replay plan.
  uint32_t ReplayVariants = 0;
  /// AEGISBIT_REPLAY=auto: cap + enable plateau detection
  /// (stop building variants when one adds no new PCs).
  bool ReplayAuto = false;

  /// AEGISBIT_REPLAY_MAX: hard upper bound on the number of variants
  /// `AEGISBIT_REPLAY=auto` will build before plateau detection forces a
  /// stop.  Plateau is the real terminator; this is a safety valve for
  /// pathologically sparse kernels and GPU-memory-constrained environments.
  /// Only consulted when `ReplayAuto` is true.  Default 32.
  uint32_t ReplayMax = 32;
};

/// Instrumentation transform toggles. All bool defaults are false.
/// These select alternative code paths in the trampoline/relay emitters
/// and the jump-strategy heuristics.
struct TransformFlags {
  /// AEGISBIT_FORCE_RELAY: force relay-stub path in zero-SGPR mode.
  bool ForceRelay = false;

  /// AEGISBIT_FORCE_SWAPPC: always upgrade to SwapPC shared-body mode.
  bool ForceSwapPC = false;

  /// AEGISBIT_MINIMAL_RELAY: minimal relay stub (no body island).
  bool MinimalRelay = false;

  /// AEGISBIT_NOP_RELAY: NOP-only relay stub (isolate instruction cost).
  bool NopRelay = false;

  /// AEGISBIT_VCC_ONLY_RELAY: VCC-only save/restore relay (no SCC).
  bool VccOnlyRelay = false;

  /// AEGISBIT_NO_BODY_JUMP: suppress jump from relay stub into body island.
  bool NoBodyJump = false;

  /// AEGISBIT_NOOP_TRAMPOLINE: empty trampoline body (no payload).
  bool NoopTrampoline = false;

  /// AEGISBIT_SBRANCH_BODY: use s_branch for body entry/return.
  bool SBranchBody = false;

  /// AEGISBIT_ACCVGPR_SPILL: legacy AccVGPR spill path (instead of scratch).
  bool AccVGPRSpill = false;

  /// AEGISBIT_BODY_NEAR_STUBS: force body island placement near relay stubs.
  bool BodyNearStubs = false;

  /// AEGISBIT_STRATEGY: payload strategy override string.
  /// "full_capture" selects PayloadStrategy::FullCapture; anything else
  /// (including empty) selects OnGpuReduce.
  std::string StrategyOverride;
};

/// Runtime configuration for AegisBit tracing.
///
/// Configuration is loaded from environment variables:
/// - AEGISBIT_ENABLED: Enable/disable tracing (default: 1)
/// - AEGISBIT_MODE: Instrumentation mode (only MEMORY_ONLY supported)
/// - AEGISBIT_KERNELS: Comma-separated kernel name patterns (default: *)
/// - AEGISBIT_OUTPUT: Output directory for traces (default: ./aegisbit_traces/)
/// - AEGISBIT_BUFFER_MB: Trace buffer size in MB (default: 64)
/// - AEGISBIT_LOG: Enable debug logging (default: 0)
/// - Debug knobs: see DebugFlags.
/// - Transform toggles: see TransformFlags.
///
/// Usage:
/// \code
///   RuntimeConfig::initialize();
///   if (RuntimeConfig::getInstance().Enabled) {
///     if (RuntimeConfig::getInstance().shouldTraceKernel("myKernel")) {
///       // Trace this kernel
///     }
///   }
/// \endcode
struct RuntimeConfig {
  /// Whether tracing is enabled
  bool Enabled = true;

  /// Instrumentation mode
  InstrumentationMode Mode = InstrumentationMode::MEMORY_ONLY;

  /// Kernel name patterns to trace (supports wildcards)
  std::vector<std::string> KernelPatterns;

  /// Output directory for trace files
  std::filesystem::path OutputDir;

  /// Trace buffer size in bytes
  size_t BufferSizeBytes = 64 * 1024 * 1024;

  /// Enable debug logging (shorthand for Debug.LogLevel > 0)
  bool LogEnabled = false;

  /// Path for structured JSON output of profiling results (empty = disabled).
  /// Set via AEGISBIT_JSON_OUTPUT env var.
  std::string JSONOutputPath;

  /// Grouped debug/diagnostic flags.
  DebugFlags Debug;

  /// Grouped transform/codegen toggles.
  TransformFlags Transform;

  /// Get the singleton instance. The first call triggers initialize() so
  /// that call sites which do not explicitly initialize still see the
  /// current environment.
  static RuntimeConfig& getInstance();

  /// Initialize (or re-initialize) configuration from environment
  /// variables. Safe to call multiple times — subsequent calls refresh
  /// the config.
  static void initialize();

  /// Check if a kernel should be traced based on patterns.
  /// \param KernelName Name of the kernel
  /// \return true if the kernel matches any pattern
  bool shouldTraceKernel(llvm::StringRef KernelName) const;

  /// Get buffer size in MB (convenience accessor)
  size_t getBufferSizeMB() const { return BufferSizeBytes / (1024 * 1024); }

  /// Log a message if logging is enabled (LogLevel >= 1).
  void log(const std::string& Message) const;

  /// Log a message only when LogLevel >= MinLevel. Useful for verbose
  /// traces that should be gated on AEGISBIT_LOG=2.
  void logAt(int MinLevel, const std::string& Message) const;

private:
  RuntimeConfig() = default;

  /// Match a kernel name against a pattern (supports * wildcard)
  static bool matchPattern(llvm::StringRef Name, llvm::StringRef Pattern);
};

} // namespace aegisbit

#endif // AEGISBIT_RUNTIME_CONFIG_H
