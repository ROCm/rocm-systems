#include "core/inc/clock_sync.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

#include "core/inc/agent.h"
#include "core/inc/amd_gpu_agent.h"
#include "core/inc/runtime.h"
#include "lttng/rocm_trace_emit.h"

namespace rocr {
namespace clock_sync {

namespace {

std::atomic<bool> g_shutdown{false};
std::thread g_poller_thread;

void poller_loop() {
#if defined(HSA_ENABLE_LTTNG_UST) && HSA_ENABLE_LTTNG_UST
  // Configurable sync interval. Default is 100ms.
  auto tick_interval = std::chrono::milliseconds(100);
  const char* env_interval = getenv("HSA_CLOCK_SYNC_INTERVAL_MS");
  if (env_interval != nullptr) {
    int parsed_interval = std::atoi(env_interval);
    if (parsed_interval > 0) {
      tick_interval = std::chrono::milliseconds(parsed_interval);
    }
  }

  while (!g_shutdown.load(std::memory_order_relaxed)) {
    if (!rocm_trace_disabled() && lttng_ust_tracepoint_enabled(rocm_hsa, clock_sync)) {
      // Iterate over all GPU agents in the system and emit a clock sync record
      // for each. gpu_agents() is already filtered to GPU-class agents only,
      // so no kAmdGpuDevice device-type check is needed.
      for (core::Agent* agent : core::Runtime::runtime_singleton_->gpu_agents()) {
        if (agent == nullptr) continue;
        auto* gpu = static_cast<AMD::GpuAgentInt*>(agent);

        HsaClockCounters clocks{};
        hsa_status_t err = gpu->driver().GetClockCounters(gpu->node_id(), &clocks);
        if (err == HSA_STATUS_SUCCESS) {
          rocm_trace_emit_hsa_clock_sync(gpu->node_id(),
                                         clocks.GPUClockCounter,
                                         clocks.SystemClockCounter);
        }
      }
    }

    std::this_thread::sleep_for(tick_interval);
  }
#endif
}

}  // namespace

void init() {
  g_shutdown.store(false, std::memory_order_relaxed);
  g_poller_thread = std::thread(poller_loop);
}

void shutdown() {
  g_shutdown.store(true, std::memory_order_release);
  if (g_poller_thread.joinable()) {
    g_poller_thread.join();
  }
}

}  // namespace clock_sync
}  // namespace rocr