//===-- aegisbit/DispatchRegistry.h - Active dispatch map -------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Thread-safe registry of in-flight kernel dispatches keyed by a per-engine
/// monotonically-increasing dispatch ID.
///
/// The registry owns only the storage (map + mutex + ID counter). Resource
/// cleanup (signals, kernarg pool return, forwarding to the application's
/// original signal) remains with `TracingEngine::cleanupDispatch` because it
/// intentionally behaves differently during normal completion vs. shutdown.
///
/// Extracted from TracingEngine.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_DISPATCH_REGISTRY_H
#define AEGISBIT_DISPATCH_REGISTRY_H

#include "aegisbit/ActiveDispatch.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aegisbit {

class DispatchRegistry {
public:
  /// Allocate a fresh dispatch ID.
  uint32_t nextId() { return NextID.fetch_add(1); }

  /// Insert a new in-flight dispatch. Overwrites on duplicate ID (the caller
  /// guarantees uniqueness via `nextId`).
  void insert(uint32_t Id, ActiveDispatch Dispatch) {
    std::lock_guard<std::mutex> Lock(Mutex);
    Dispatches[Id] = std::move(Dispatch);
  }

  /// Remove and return the dispatch with the given ID, if any.
  std::optional<ActiveDispatch> take(uint32_t Id) {
    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = Dispatches.find(Id);
    if (It == Dispatches.end())
      return std::nullopt;
    ActiveDispatch Out = std::move(It->second);
    Dispatches.erase(It);
    return Out;
  }

  /// Visit every active dispatch under the registry's internal lock.
  /// The visitor must be side-effect-light — it runs while the lock is held
  /// and therefore blocks `insert`, `take`, and other calls.
  ///
  /// Signature: `void(uint32_t Id, const ActiveDispatch&)`.
  template <typename F> void forEach(F &&Fn) const {
    std::lock_guard<std::mutex> Lock(Mutex);
    for (const auto &[Id, Dispatch] : Dispatches)
      Fn(Id, Dispatch);
  }

  /// Remove and return every active dispatch. Used by `TracingEngine::finalize`
  /// to iterate dispatches for shutdown cleanup outside the lock.
  std::vector<std::pair<uint32_t, ActiveDispatch>> drainAll() {
    std::lock_guard<std::mutex> Lock(Mutex);
    std::vector<std::pair<uint32_t, ActiveDispatch>> Out;
    Out.reserve(Dispatches.size());
    for (auto &[Id, Dispatch] : Dispatches)
      Out.emplace_back(Id, std::move(Dispatch));
    Dispatches.clear();
    return Out;
  }

  /// Current in-flight count.
  size_t size() const {
    std::lock_guard<std::mutex> Lock(Mutex);
    return Dispatches.size();
  }

private:
  std::atomic<uint32_t> NextID{0};
  mutable std::mutex Mutex;
  std::unordered_map<uint32_t, ActiveDispatch> Dispatches;
};

} // namespace aegisbit

#endif // AEGISBIT_DISPATCH_REGISTRY_H
