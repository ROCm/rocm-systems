// SPDX-License-Identifier: MIT
// Linux LD_PRELOAD timing helper for the HSA queue-lifecycle workload.
// Delays background KFD SET_EVENT calls, then forwards the real ioctl.
#include <linux/kfd_ioctl.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>

extern "C" int ioctl(int fd, unsigned long request, ...) noexcept {
  const int incoming_errno = errno;
  // ROCr/thunk use pointer arguments for the ioctls in this workload.
  va_list args;
  va_start(args, request);
  void* argument = va_arg(args, void*);
  va_end(args);
  using Ioctl = int (*)(int, unsigned long, ...);
  static const Ioctl real_ioctl = reinterpret_cast<Ioctl>(dlsym(RTLD_NEXT, "ioctl"));
  if (real_ioctl == nullptr) {
    std::fputs("Cannot resolve real ioctl\n", stderr);
    _exit(2);
  }

  // In this single-process Linux program, the main thread's TID equals PID.
  const bool background_set = request == AMDKFD_IOC_SET_EVENT && syscall(SYS_gettid) != getpid();
  if (background_set) {
    const char* setting = std::getenv("KFD_SET_EVENT_DELAY_US");
    const unsigned long delay = setting ? std::strtoul(setting, nullptr, 10) : 100000;
    timespec remaining{static_cast<time_t>(delay / 1000000),
                       static_cast<long>((delay % 1000000) * 1000)};
    if (delay != 0)
      while (nanosleep(&remaining, &remaining) == -1 && errno == EINTR) {
      }
  }
  errno = incoming_errno;
  const int result = real_ioctl(fd, request, argument);
  const int saved_errno = errno;
  if (request == AMDKFD_IOC_DESTROY_EVENT)
    std::fprintf(stderr, "Real KFD DESTROY_EVENT(event_id=%u) returned %d\n",
                 static_cast<kfd_ioctl_destroy_event_args*>(argument)->event_id, result);
  if (background_set && result == -1)
    std::fprintf(stderr, "Real KFD SET_EVENT(event_id=%u) returned -1, errno=%d\n",
                 static_cast<kfd_ioctl_set_event_args*>(argument)->event_id, saved_errno);
  errno = saved_errno;
  return result;
}
