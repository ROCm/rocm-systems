//===-- RuntimeConfig.cpp - Runtime Configuration ---------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of runtime configuration loading from environment
/// variables. All AEGISBIT_* env vars are parsed here and exposed via
/// RuntimeConfig / RuntimeConfig::Debug / RuntimeConfig::Transform so that
/// call sites do not read getenv() directly.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/RuntimeConfig.h"
#include "llvm/ADT/StringRef.h"
#include <atomic>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <sstream>

namespace aegisbit {

namespace {

/// Interpret a string as a boolean, using the same semantics as the legacy
/// env-var parsing: any value other than "0" / "false" (case-sensitive) is
/// considered true.
bool parseBool(const char* Env) {
  if (!Env) return false;
  std::string V(Env);
  return V != "0" && V != "false";
}

/// Parse an int from the environment variable \p Name, returning \p Default
/// when the variable is unset or unparseable.
int getEnvInt(const char* Name, int Default) {
  const char* Env = std::getenv(Name);
  if (!Env) return Default;
  try {
    return std::stoi(std::string(Env));
  } catch (...) {
    return Default;
  }
}

uint32_t getEnvU32(const char* Name, uint32_t Default) {
  const char* Env = std::getenv(Name);
  if (!Env) return Default;
  try {
    long V = std::stol(std::string(Env));
    if (V < 0) return Default;
    return static_cast<uint32_t>(V);
  } catch (...) {
    return Default;
  }
}

/// True iff the env var is set to a non-empty value (and not "0"/"false").
bool getEnvFlag(const char* Name) {
  return parseBool(std::getenv(Name));
}

/// Get the raw string value of the env var, or "" if unset.
std::string getEnvStr(const char* Name) {
  const char* Env = std::getenv(Name);
  return Env ? std::string(Env) : std::string();
}

std::atomic<bool> InitializedOnce{false};

} // namespace

RuntimeConfig& RuntimeConfig::getInstance() {
  static RuntimeConfig Instance;
  // Lazy first-time initialization so call sites that read RuntimeConfig
  // without first calling initialize() still observe the current
  // environment. initialize() may still be called explicitly to re-read.
  if (!InitializedOnce.load(std::memory_order_acquire)) {
    bool Expected = false;
    if (InitializedOnce.compare_exchange_strong(Expected, true)) {
      initialize();
    }
  }
  return Instance;
}

void RuntimeConfig::initialize() {
  // Mark as initialized first so the lazy-init branch in getInstance()
  // is skipped when we call it below. This also lets explicit
  // re-initialization (e.g. from tests that setenv then call
  // initialize()) re-parse the environment into the existing singleton.
  InitializedOnce.store(true, std::memory_order_release);
  RuntimeConfig& Cfg = getInstance();

  // AEGISBIT_ENABLED (default: 1)
  if (const char* Env = std::getenv("AEGISBIT_ENABLED")) {
    Cfg.Enabled = parseBool(Env);
  } else {
    Cfg.Enabled = true;
  }

  // AEGISBIT_LOG (default: 0). Accepts "0"/"false" -> off, "1"/"2"/... ->
  // numeric log level. Any non-numeric non-false value is treated as level 1.
  {
    const char* Env = std::getenv("AEGISBIT_LOG");
    if (!Env || !parseBool(Env)) {
      Cfg.Debug.LogLevel = 0;
    } else {
      try {
        Cfg.Debug.LogLevel = std::stoi(std::string(Env));
      } catch (...) {
        Cfg.Debug.LogLevel = 1;
      }
      if (Cfg.Debug.LogLevel < 0) Cfg.Debug.LogLevel = 0;
    }
    Cfg.LogEnabled = Cfg.Debug.LogLevel > 0;
  }

  // AEGISBIT_MODE (only MEMORY_ONLY supported)
  Cfg.Mode = InstrumentationMode::MEMORY_ONLY;

  // AEGISBIT_KERNELS (default: *)
  Cfg.KernelPatterns.clear();
  if (const char* Env = std::getenv("AEGISBIT_KERNELS")) {
    std::string Patterns(Env);
    std::stringstream SS(Patterns);
    std::string Pattern;
    while (std::getline(SS, Pattern, ',')) {
      size_t Start = Pattern.find_first_not_of(" \t");
      size_t End = Pattern.find_last_not_of(" \t");
      if (Start != std::string::npos && End != std::string::npos) {
        Cfg.KernelPatterns.push_back(Pattern.substr(Start, End - Start + 1));
      } else if (!Pattern.empty()) {
        Cfg.KernelPatterns.push_back(Pattern);
      }
    }
  }
  if (Cfg.KernelPatterns.empty()) {
    Cfg.KernelPatterns.push_back("*");
  }

  // AEGISBIT_OUTPUT (default: ./aegisbit_traces/)
  if (const char* Env = std::getenv("AEGISBIT_OUTPUT")) {
    Cfg.OutputDir = std::filesystem::path(Env);
  } else {
    Cfg.OutputDir = std::filesystem::path("./aegisbit_traces/");
  }

  // AEGISBIT_JSON_OUTPUT (default: empty = disabled)
  if (const char* Env = std::getenv("AEGISBIT_JSON_OUTPUT")) {
    Cfg.JSONOutputPath = Env;
  } else {
    Cfg.JSONOutputPath.clear();
  }

  // AEGISBIT_BUFFER_MB (default: 64)
  if (const char* Env = std::getenv("AEGISBIT_BUFFER_MB")) {
    try {
      size_t MB = std::stoull(Env);
      if (MB > 0 && MB <= 4096) {
        Cfg.BufferSizeBytes = MB * 1024 * 1024;
      } else {
        Cfg.BufferSizeBytes = 64 * 1024 * 1024;
      }
    } catch (...) {
      Cfg.BufferSizeBytes = 64 * 1024 * 1024;
    }
  } else {
    Cfg.BufferSizeBytes = 64 * 1024 * 1024;
  }

  // --- Debug flags ---
  Cfg.Debug.DumpInputELFPrefix = getEnvStr("AEGISBIT_DUMP_INPUT_ELF");
  Cfg.Debug.DumpELFPath        = getEnvStr("AEGISBIT_DUMP_ELF");
  Cfg.Debug.DumpBlobsDir       = getEnvStr("AEGISBIT_DUMP_BLOBS");
  Cfg.Debug.MaxSites           = getEnvU32("AEGISBIT_MAX_SITES", 0);
  Cfg.Debug.MaxLDSSites        = getEnvU32("AEGISBIT_MAX_LDS", 0);
  Cfg.Debug.SkipLDSFirst       = getEnvInt("AEGISBIT_SKIP_LDS_FIRST", 0);
  Cfg.Debug.SkipSignal         = getEnvFlag("AEGISBIT_SKIP_SIGNAL");
  Cfg.Debug.GPUFilter          = getEnvInt("AEGISBIT_GPU_FILTER", -1);
  Cfg.Debug.BodySiteOnly       = getEnvInt("AEGISBIT_BODY_SITE_ONLY", -1);
  Cfg.Debug.MaxBodySites       = getEnvU32("AEGISBIT_MAX_BODY_SITES",
                                            UINT32_MAX);
  Cfg.Debug.DryPayloadLevel    = getEnvInt("AEGISBIT_DRY_PAYLOAD", 99);
  Cfg.Debug.CountMode          = getEnvInt("AEGISBIT_COUNT_MODE", 99);

  // AEGISBIT_REPLAY: instrumentation-replay knob.  Accepts:
  //   unset / "0" / "false"   — single-variant legacy path.
  //   "auto"                  — plateau-capped multi-variant (up to
  //                             AEGISBIT_REPLAY_MAX, default 32).
  //   N (positive integer)    — exactly N variants.
  // The single-variant path is byte-for-byte equivalent to the pre-replay
  // flow (ensureVariants collapses to one bundle, VariantSelector returns 0).
  {
    Cfg.Debug.ReplayVariants = 0;
    Cfg.Debug.ReplayAuto = false;
    if (const char *Env = std::getenv("AEGISBIT_REPLAY")) {
      std::string V(Env);
      if (V == "auto" || V == "AUTO") {
        Cfg.Debug.ReplayAuto = true;
      } else if (parseBool(Env)) {
        try {
          long N = std::stol(V);
          if (N > 0)
            Cfg.Debug.ReplayVariants = static_cast<uint32_t>(N);
        } catch (...) {
          // Non-numeric, non-"auto" truthy value — ignore.
        }
      }
    }
    // AEGISBIT_REPLAY_MAX: safety cap for auto mode.  Plateau detection
    // does the real work; this only matters for pathologically sparse
    // kernels (or GPU-memory-tight environments that want a lower bound).
    // Garbage values fall back to the default 32.
    Cfg.Debug.ReplayMax = getEnvU32("AEGISBIT_REPLAY_MAX", 32);
    if (Cfg.Debug.ReplayMax == 0)
      Cfg.Debug.ReplayMax = 32;
  }

  // --- Transform flags ---
  Cfg.Transform.ForceRelay      = getEnvFlag("AEGISBIT_FORCE_RELAY");
  Cfg.Transform.ForceSwapPC     = getEnvFlag("AEGISBIT_FORCE_SWAPPC");
  Cfg.Transform.MinimalRelay    = getEnvFlag("AEGISBIT_MINIMAL_RELAY");
  Cfg.Transform.NopRelay        = getEnvFlag("AEGISBIT_NOP_RELAY");
  Cfg.Transform.VccOnlyRelay    = getEnvFlag("AEGISBIT_VCC_ONLY_RELAY");
  Cfg.Transform.NoBodyJump      = getEnvFlag("AEGISBIT_NO_BODY_JUMP");
  Cfg.Transform.NoopTrampoline  = getEnvFlag("AEGISBIT_NOOP_TRAMPOLINE");
  Cfg.Transform.SBranchBody     = getEnvFlag("AEGISBIT_SBRANCH_BODY");
  Cfg.Transform.AccVGPRSpill    = getEnvFlag("AEGISBIT_ACCVGPR_SPILL");
  Cfg.Transform.BodyNearStubs   = getEnvFlag("AEGISBIT_BODY_NEAR_STUBS");
  Cfg.Transform.StrategyOverride = getEnvStr("AEGISBIT_STRATEGY");

  if (Cfg.LogEnabled) {
    std::cerr << "[aegisbit] Configuration loaded:\n"
              << "  Enabled: " << (Cfg.Enabled ? "true" : "false") << "\n"
              << "  Mode: MEMORY_ONLY\n"
              << "  Output: " << Cfg.OutputDir.string() << "\n"
              << "  Buffer: " << Cfg.getBufferSizeMB() << " MB\n"
              << "  LogLevel: " << Cfg.Debug.LogLevel << "\n"
              << "  Replay: "
              << (Cfg.Debug.ReplayAuto
                      ? std::string("auto (max ") +
                            std::to_string(Cfg.Debug.ReplayMax) + ")"
                      : (Cfg.Debug.ReplayVariants > 0
                             ? std::to_string(Cfg.Debug.ReplayVariants)
                             : std::string("off")))
              << "\n"
              << "  Kernels: ";
    for (size_t i = 0; i < Cfg.KernelPatterns.size(); ++i) {
      if (i > 0) std::cerr << ", ";
      std::cerr << Cfg.KernelPatterns[i];
    }
    std::cerr << "\n";
  }
}

bool RuntimeConfig::shouldTraceKernel(llvm::StringRef KernelName) const {
  if (!Enabled) {
    return false;
  }

  for (const auto& Pattern : KernelPatterns) {
    if (matchPattern(KernelName, Pattern)) {
      return true;
    }
  }
  return false;
}

void RuntimeConfig::log(const std::string& Message) const {
  if (LogEnabled) {
    std::cerr << "[aegisbit] " << Message << "\n";
  }
}

void RuntimeConfig::logAt(int MinLevel, const std::string& Message) const {
  if (Debug.LogLevel >= MinLevel) {
    std::cerr << "[aegisbit] " << Message << "\n";
  }
}

bool RuntimeConfig::matchPattern(llvm::StringRef Name, llvm::StringRef Pattern) {
  if (Pattern == "*") {
    return true;
  }
  if (Pattern == Name) {
    return true;
  }
  if (Pattern.empty()) {
    return Name.empty();
  }

  // .kd suffix tolerance: `DispatchInterceptor` strips the ".kd" suffix from
  // kernel descriptor symbol names before filtering, but the `tools/aegisbit`
  // CLI auto-generates filter patterns with the ".kd" suffix for discovered
  // Triton kernels. Treat `"foo"` and `"foo.kd"` as equivalent so the filter
  // still matches the trimmed name.
  if (Pattern.ends_with(".kd") && Pattern.drop_back(3) == Name) {
    return true;
  }
  if (Name.ends_with(".kd") && Name.drop_back(3) == Pattern) {
    return true;
  }

  // Simple wildcard matching supporting * at start, end, or both
  // Pattern: "*suffix" - ends with suffix
  // Pattern: "prefix*" - starts with prefix
  // Pattern: "*middle*" - contains middle
  // Pattern: "exact" - exact match
  //
  // Case-insensitive matching for better usability with camelCase names

  bool StartsWithStar = Pattern.starts_with("*");
  bool EndsWithStar = Pattern.ends_with("*");

  if (StartsWithStar && EndsWithStar && Pattern.size() > 1) {
    llvm::StringRef Middle = Pattern.drop_front(1).drop_back(1);
    return Name.contains_insensitive(Middle);
  } else if (StartsWithStar) {
    llvm::StringRef Suffix = Pattern.drop_front(1);
    return Name.ends_with_insensitive(Suffix);
  } else if (EndsWithStar) {
    llvm::StringRef Prefix = Pattern.drop_back(1);
    return Name.starts_with_insensitive(Prefix);
  } else {
    return Name == Pattern;
  }
}

} // namespace aegisbit
