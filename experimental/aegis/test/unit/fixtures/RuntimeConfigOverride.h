//===-- RuntimeConfigOverride.h - RuntimeConfig Test Helper -----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Test-only helper that lets unit tests exercise transform/debug code
/// paths by overriding RuntimeConfig flags without going through the
/// environment. Intended usage:
///
/// \code
///   RuntimeConfigOverride Override;
///   Override.config().Transform.ForceRelay = true;
///   // ... run code under test ...
///   // RuntimeConfig is automatically restored on destruction.
/// \endcode
///
/// The destructor re-initializes RuntimeConfig from the current
/// environment so successive tests start from a clean state.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_TEST_RUNTIME_CONFIG_OVERRIDE_H
#define AEGISBIT_TEST_RUNTIME_CONFIG_OVERRIDE_H

#include "aegisbit/RuntimeConfig.h"

#include <utility>

namespace aegisbit {

/// RAII helper that installs a mutable view of the process-wide
/// RuntimeConfig singleton and restores defaults on destruction.
class RuntimeConfigOverride {
public:
  /// Snapshot current configuration then reset fields to library defaults
  /// (all flags off, standard output path, default buffer size). Tests can
  /// then mutate .Debug / .Transform / etc via config().
  RuntimeConfigOverride() : Previous(RuntimeConfig::getInstance()) {
    RuntimeConfig &Cfg = RuntimeConfig::getInstance();
    Cfg.Enabled = true;
    Cfg.Mode = InstrumentationMode::MEMORY_ONLY;
    Cfg.KernelPatterns = {"*"};
    Cfg.OutputDir = "./aegisbit_traces/";
    Cfg.BufferSizeBytes = 64 * 1024 * 1024;
    Cfg.LogEnabled = false;
    Cfg.JSONOutputPath.clear();
    Cfg.Debug = DebugFlags{};
    Cfg.Transform = TransformFlags{};
  }

  /// Restore the prior configuration.
  ~RuntimeConfigOverride() {
    RuntimeConfig::getInstance() = Previous;
  }

  RuntimeConfigOverride(const RuntimeConfigOverride &) = delete;
  RuntimeConfigOverride &operator=(const RuntimeConfigOverride &) = delete;

  /// Mutable accessor for the global config while the override is live.
  RuntimeConfig &config() { return RuntimeConfig::getInstance(); }

private:
  RuntimeConfig Previous;
};

} // namespace aegisbit

#endif // AEGISBIT_TEST_RUNTIME_CONFIG_OVERRIDE_H
