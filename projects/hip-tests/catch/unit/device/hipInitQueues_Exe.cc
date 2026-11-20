/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Child of Unit_hipInit_Positive_NoQueuesAtInit. Counts the hardware queues this process creates,
// by intercepting the KFD create-queue ioctl, and reports them as "key=value" pairs:
//   init_queues  queues created by hipInit, hipGetDeviceCount and hipGetDeviceProperties of every
//                device
//   used_queues  queues created up to the end of a kernel on device 0
// Returns 0 when it could report, 1 on a HIP error (printed to stderr).

#include <hip/hip_runtime.h>

#include <dlfcn.h>
#include <linux/ioctl.h>

#include <atomic>
#include <cstdarg>
#include <iostream>

#define CHECK(call)                                                                                \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    if (err != hipSuccess) {                                                                       \
      std::cerr << #call << " failed: " << hipGetErrorString(err) << std::endl;                    \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

namespace {

// AMDKFD_IOC_CREATE_QUEUE (linux/kfd_ioctl.h). Matched by type and number only: the size encoded in
// the request grows with the args struct, which differs between kernel and ROCr headers.
constexpr unsigned kKfdIoctlType = 'K';
constexpr unsigned kKfdIoctlCreateQueue = 0x02;

std::atomic<int> created_queues{0};

__global__ void emptyKernel() {}

}  // namespace

// Interposes libc's ioctl(), which the thunk (libhsakmt) in ROCr calls for every KFD request.
extern "C" __attribute__((visibility("default"))) int ioctl(int fd, unsigned long request, ...) {
  va_list ap;
  va_start(ap, request);
  void* arg = va_arg(ap, void*);
  va_end(ap);
  using IoctlFn = int (*)(int, unsigned long, void*);
  static const IoctlFn real = reinterpret_cast<IoctlFn>(dlsym(RTLD_NEXT, "ioctl"));
  const int ret = real(fd, request, arg);
  if (ret == 0 && _IOC_TYPE(request) == kKfdIoctlType && _IOC_NR(request) == kKfdIoctlCreateQueue) {
    ++created_queues;
  }
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
  const int init_queues = created_queues;

  CHECK(hipSetDevice(0));
  emptyKernel<<<1, 1>>>();
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());

  std::cout << "init_queues=" << init_queues << " used_queues=" << created_queues << std::endl;
  return 0;
}
