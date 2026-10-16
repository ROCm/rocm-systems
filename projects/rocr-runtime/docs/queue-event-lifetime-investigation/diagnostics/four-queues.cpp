// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Standalone ROCr queue-event lifetime reproducer on physical KFD hardware.
// No rocJITsu, HIP, GEMM library, or GPU kernel dispatch is required.
#include <hsa/hsa.h>

#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <pthread.h>
#include <unistd.h>
#include <unordered_map>

namespace {
struct EventWrites {
  pthread_t main_thread = pthread_self();
  std::mutex mutex;
  std::unordered_map<uint32_t, unsigned> pending;
  std::atomic<unsigned> delays{0};
  std::atomic<unsigned> overlaps{0};
};

// Keep instrumentation available during library static teardown as well.
EventWrites* const writes = new EventWrites;

hsa_status_t FindGpu(hsa_agent_t agent, void* data) {
  hsa_device_type_t type;
  const hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (status == HSA_STATUS_SUCCESS && type == HSA_DEVICE_TYPE_GPU)
    *static_cast<hsa_agent_t*>(data) = agent;
  return status;
}

__attribute__((destructor)) void CheckLateWrites() {
  if (writes->overlaps.load() != 0) {
    std::fprintf(stderr, "Queue event destroyed during an outstanding callback write\n");
    std::_Exit(1);
  }
}
}  // namespace

extern "C" int ioctl(int fd, unsigned long request, ...) {
  va_list args;
  va_start(args, request);
  void* argument = va_arg(args, void*);
  va_end(args);
  static const auto next =
      reinterpret_cast<int (*)(int, unsigned long, void*)>(dlsym(RTLD_NEXT, "ioctl"));
  if (next == nullptr) std::_Exit(1);

  // KFD SET_EVENT (0x4b0a) and DESTROY_EVENT (0x4b09) both start with
  // uint32_t event_id. Ignore the size/direction bits in the ioctl number.
  const bool delayed = (request & 0xffff) == 0x4b0a && argument != nullptr &&
      !pthread_equal(pthread_self(), writes->main_thread);
  const bool destroyed = (request & 0xffff) == 0x4b09 && argument != nullptr;
  const uint32_t id = delayed || destroyed ? *static_cast<uint32_t*>(argument) : 0;
  if (delayed) {
    {
      std::lock_guard<std::mutex> lock(writes->mutex);
      ++writes->pending[id];
    }
    ++writes->delays;
    // Allow native queue destruction to progress before this callback's final
    // write completes. Forward the real ioctl unchanged after the delay.
    usleep(100000);
  }
  if (destroyed) {
    std::lock_guard<std::mutex> lock(writes->mutex);
    if (writes->pending[id] != 0) ++writes->overlaps;
  }
  const int result = next(fd, request, argument);
  const int saved_errno = errno;
  if (delayed && result != 0)
    std::fprintf(stderr, "Delayed KFD SET_EVENT id=%u failed: result=%d errno=%d\n", id, result,
                 saved_errno);
  if (delayed) {
    std::lock_guard<std::mutex> lock(writes->mutex);
    --writes->pending[id];
  }
  errno = saved_errno;
  return result;
}

int main() {
  if (hsa_init() != HSA_STATUS_SUCCESS) return 1;
  hsa_agent_t gpu{};
  if (hsa_iterate_agents(FindGpu, &gpu) != HSA_STATUS_SUCCESS) return 1;
  if (gpu.handle == 0) {
    hsa_shut_down();
    std::fprintf(stderr, "No GPU agent; run on KFD or under CPU emulation\n");
    return 77;
  }
  for (unsigned iteration = 0; iteration != 4; ++iteration) {
    hsa_queue_t* queue = nullptr;
    if (hsa_queue_create(gpu, 64, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0, 0, &queue) !=
        HSA_STATUS_SUCCESS)
      return 1;
    if (hsa_queue_destroy(queue) != HSA_STATUS_SUCCESS) return 1;
  }
  if (hsa_shut_down() != HSA_STATUS_SUCCESS) return 1;
  std::fprintf(stderr, "Queue event lifetime: delayed writes=%u, overlapping destroys=%u\n",
               writes->delays.load(), writes->overlaps.load());
  if (writes->overlaps.load() != 0) return 1;
  if (writes->delays.load() == 0) {
    std::fprintf(stderr, "No background KFD event writes exercised\n");
    return 77;
  }
  return 0;
}
