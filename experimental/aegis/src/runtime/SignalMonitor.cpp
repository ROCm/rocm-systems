//===-- SignalMonitor.cpp - Completion signal monitor impl ----------------===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/SignalMonitor.h"
#include "aegisbit/DispatchRegistry.h"
#include "aegisbit/RuntimeConfig.h"

#ifdef AEGISBIT_HAS_GPU
#include <hsa/hsa.h>
#endif

#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace aegisbit {

SignalMonitor::SignalMonitor(DispatchRegistry &Registry,
                             CompletionCallback OnCompleted,
                             TimeoutCallback OnTimedOut, int TimeoutSeconds)
    : Registry(Registry), OnCompleted(std::move(OnCompleted)),
      OnTimedOut(std::move(OnTimedOut)), TimeoutSeconds(TimeoutSeconds) {}

SignalMonitor::~SignalMonitor() {
  // Best-effort stop — the engine's finalize() is expected to stop() us
  // explicitly in the correct ordering.  A destructor-time stop is the
  // failsafe for test fixtures that construct a SignalMonitor directly.
  stop();
}

void SignalMonitor::start() {
  if (Running.exchange(true))
    return; // Already running.

  StopRequested.store(false);
  Thread = std::thread([this]() { loop(); });
}

void SignalMonitor::stop() {
  if (!Running.load())
    return; // Never started or already stopped.

  StopRequested.store(true);
  if (Thread.joinable())
    Thread.join();
  Running.store(false);
}

void SignalMonitor::loop() {
  RuntimeConfig &Cfg = RuntimeConfig::getInstance();
  Cfg.log("Signal monitor thread started");

  while (!StopRequested.load()) {
    std::vector<uint32_t> CompletedIDs;
    std::vector<uint32_t> TimedOutIDs;
    auto Now = std::chrono::steady_clock::now();

    Registry.forEach([&](uint32_t ID, const ActiveDispatch &Dispatch) {
      if (Dispatch.CompletionSignalHandle == 0)
        return;
#ifdef AEGISBIT_HAS_GPU
      hsa_signal_t Signal = {Dispatch.CompletionSignalHandle};
      hsa_signal_value_t Value = hsa_signal_load_relaxed(Signal);
      if (Value == 0) {
        CompletedIDs.push_back(ID);
      } else {
        auto Elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            Now - Dispatch.StartTime);
        if (Elapsed.count() >= TimeoutSeconds)
          TimedOutIDs.push_back(ID);
      }
#else
      (void)Now;
#endif
    });

    // Completion: outside the registry lock. The callback is responsible for
    // atomically removing the dispatch via `DispatchRegistry::take` — if it
    // races with another path (e.g. timeout below) the registry guarantees
    // only one wins.
    for (uint32_t ID : CompletedIDs) {
      Cfg.log("Signal monitor: dispatch " + std::to_string(ID) + " completed");
      OnCompleted(ID);
    }

    // Timeout: we own the removal so the callback receives the dispatch by
    // value and handles cleanup + stats.
    for (uint32_t ID : TimedOutIDs) {
      Cfg.log("WARNING: Dispatch " + std::to_string(ID) +
              " timed out after " + std::to_string(TimeoutSeconds) +
              "s — cleaning up resources");
      if (auto Taken = Registry.take(ID))
        OnTimedOut(ID, std::move(*Taken));
    }

    if (CompletedIDs.empty() && TimedOutIDs.empty())
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  Cfg.log("Signal monitor thread exiting");
}

} // namespace aegisbit
