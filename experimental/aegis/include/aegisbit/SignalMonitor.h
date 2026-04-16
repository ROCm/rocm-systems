//===-- aegisbit/SignalMonitor.h - Completion signal monitor ----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Background thread that polls HSA completion signals of in-flight
/// dispatches held by a `DispatchRegistry`, invoking caller-supplied
/// callbacks on completion or timeout.
///
/// Extracted from TracingEngine. The monitor owns only its thread and a
/// stop flag; it calls into HSA via `hsa_signal_load_relaxed` and therefore
/// must be stopped before the HSA runtime tears down. The engine's finalize
/// path enforces this ordering by calling `stop()` before any HSA pool
/// teardown.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_SIGNAL_MONITOR_H
#define AEGISBIT_SIGNAL_MONITOR_H

#include "aegisbit/ActiveDispatch.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>

namespace aegisbit {

class DispatchRegistry;

class SignalMonitor {
public:
  /// Invoked (outside the registry lock) when a dispatch's completion signal
  /// drops to zero. The dispatch is still present in the registry; the
  /// callback is expected to call `DispatchRegistry::take(ID)` itself (the
  /// engine's `onDispatchComplete` does exactly that).
  using CompletionCallback = std::function<void(uint32_t DispatchID)>;

  /// Invoked after the monitor has removed a timed-out dispatch from the
  /// registry. The callback owns `Dispatch` and is responsible for any
  /// resource cleanup (signal destruction, kernarg release, stats update).
  using TimeoutCallback =
      std::function<void(uint32_t DispatchID, ActiveDispatch Dispatch)>;

  /// Construct a monitor bound to a registry. `OnCompleted`/`OnTimedOut`
  /// must outlive the monitor; in the engine they are captured lambdas that
  /// call the engine's own methods.
  SignalMonitor(DispatchRegistry &Registry, CompletionCallback OnCompleted,
                TimeoutCallback OnTimedOut, int TimeoutSeconds);

  /// Not copyable/movable — holds a thread and atomic state.
  SignalMonitor(const SignalMonitor &) = delete;
  SignalMonitor &operator=(const SignalMonitor &) = delete;

  ~SignalMonitor();

  /// Spawn the monitor thread. Idempotent — subsequent calls are no-ops
  /// until `stop()` is called.
  void start();

  /// Signal the monitor to stop and join its thread. Idempotent. Safe to
  /// call from any thread except the monitor thread itself.
  void stop();

  /// Whether the monitor thread is currently running.
  bool isRunning() const { return Running.load(); }

private:
  void loop();

  DispatchRegistry &Registry;
  CompletionCallback OnCompleted;
  TimeoutCallback OnTimedOut;
  int TimeoutSeconds;

  std::atomic<bool> StopRequested{false};
  std::atomic<bool> Running{false};
  std::thread Thread;
};

} // namespace aegisbit

#endif // AEGISBIT_SIGNAL_MONITOR_H
