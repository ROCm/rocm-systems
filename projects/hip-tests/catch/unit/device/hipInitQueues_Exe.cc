/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Child of Unit_hipInit_Positive_NoQueuesAtInit. Tracks hardware queue lifetimes by intercepting
// the KFD create/destroy ioctls and reports them as "key=value" pairs:
//   init_queues     queues retained after hipInit and device queries
//   used_queues     queues created after those queries, through a kernel on device 0
//   tracking_errors duplicate live IDs or unmatched successful destructions
// Returns 0 when it could report, 1 on a HIP error (printed to stderr).

#include <hip/hip_runtime.h>

#include <dlfcn.h>
#include <linux/kfd_ioctl.h>

#include <cerrno>
#include <cstdarg>
#include <iostream>
#include <mutex>
#include <set>

#define CHECK(call)                                                                                \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    if (err != hipSuccess) {                                                                       \
      std::cerr << #call << " failed: " << hipGetErrorString(err) << std::endl;                    \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

namespace {

// Match by type and number only: the size encoded in a request grows with its args struct,
// which differs between kernel and ROCr headers. The queue_id fields retain their offsets.
constexpr unsigned kKfdIoctlType = 'K';
constexpr unsigned kKfdIoctlCreateQueue = 0x02;
constexpr unsigned kKfdIoctlDestroyQueue = 0x03;

struct QueueSnapshot {
  size_t live;
  size_t created;
  size_t errors;
};

struct QueueTracker {
  std::mutex mutex;
  std::set<uint32_t> live_ids;
  size_t created = 0;
  size_t errors = 0;

  void recordCreate(uint32_t id) {
    ++created;
    if (!live_ids.insert(id).second) ++errors;
  }

  void recordDestroy(uint32_t id) {
    if (live_ids.erase(id) != 1) ++errors;
  }

  QueueSnapshot snapshot() {
    std::lock_guard<std::mutex> lock(mutex);
    return {live_ids.size(), created, errors};
  }
};

QueueTracker& queueTracker() {
  // Library destructors can destroy queues after main and after this executable's statics.
  // Keep the interception state alive until process teardown.
  static auto* const tracker = new QueueTracker;
  return *tracker;
}

__global__ void emptyKernel() {}

}  // namespace

// Interposes libc's ioctl(), which the thunk (libhsakmt) in ROCr calls for every KFD request.
extern "C" __attribute__((visibility("default"))) int ioctl(int fd, unsigned long request, ...) {
  va_list ap;
  va_start(ap, request);
  void* arg = va_arg(ap, void*);
  va_end(ap);
  using IoctlFn = int (*)(int, unsigned long, ...);
  static const IoctlFn real = reinterpret_cast<IoctlFn>(dlsym(RTLD_NEXT, "ioctl"));
  if (real == nullptr) {
    errno = ENOSYS;
    return -1;
  }
  const bool create = _IOC_NR(request) == kKfdIoctlCreateQueue;
  const bool destroy = _IOC_NR(request) == kKfdIoctlDestroyQueue;
  if (_IOC_TYPE(request) != kKfdIoctlType || (!create && !destroy)) {
    return real(fd, request, arg);
  }

  auto& tracker = queueTracker();
  // Keep native operations and their observations in the same order, including queue-ID reuse.
  std::lock_guard<std::mutex> lock(tracker.mutex);
  const int ret = real(fd, request, arg);
  const int saved_errno = errno;
  if (ret == 0) {
    if (create) {
      tracker.recordCreate(static_cast<kfd_ioctl_create_queue_args*>(arg)->queue_id);
    } else {
      tracker.recordDestroy(static_cast<kfd_ioctl_destroy_queue_args*>(arg)->queue_id);
    }
  }
  errno = saved_errno;
  return ret;
}

int main() {
  CHECK(hipInit(0));
  int devices = 0;
  CHECK(hipGetDeviceCount(&devices));
  for (int i = 0; i < devices; ++i) {
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, i));
  }
  const QueueSnapshot initialized = queueTracker().snapshot();

  CHECK(hipSetDevice(0));
  emptyKernel<<<1, 1>>>();
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());

  const QueueSnapshot used = queueTracker().snapshot();
  std::cout << "init_queues=" << initialized.live
            << " used_queues=" << used.created - initialized.created
            << " tracking_errors=" << used.errors << std::endl;
  return 0;
}
