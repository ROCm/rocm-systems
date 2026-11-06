// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_UTIL_KFD_PMC_ENABLE_H_
#define SRC_UTIL_KFD_PMC_ENABLE_H_

// Uses rocprofiler-sdk's vendored copy of the kernel's uapi/linux/kfd_ioctl.h (struct
// kfd_ioctl_profiler_args and friends) for the PMC/VERSION ops needed to issue the KFD
// profiler interface v2 dispatch-mode PMC request (KFD_IOC_PROFILER_PMC with lock=0,
// perfcount_enable=1) before dispatching. This is a plain header with no function bodies to
// link against (just struct/enum/macro declarations mirroring the kernel uapi header), so
// including it does not pull ctrl_v3 into linking against the main rocprofiler-sdk library or
// its plugin init path (see CMakeLists.txt: direct HSA calls only). Use the real header rather
// than a hand-trimmed struct to prevent future struct mismatches: omitting union members (e.g.
// pc_sample) changes the compiler-inferred alignment/padding of kfd_ioctl_profiler_args, silently
// shifting every field's offset relative to what the kernel expects.
#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace kfd_pmc {

// The KFD profiler interface version at which KFD_IOC_PROFILER_PMC starts forcing the perfmon
// clock on, per the "drm/amdkfd: Force perfmon clock on for profiler PMC requests" KFD commit.
// Mirrors kPerfmonClockMinVersion in rocprofiler-sdk's counters/ioctl.cpp on the (unmerged)
// dispatch-mode branch. Named after the perfmon-clock behavior itself, not after dispatch mode:
// the lock=0/perfcount_enable=1 request already existed at version 1, just without forcing the
// clock on, so this is a feature gate rather than a dispatch-mode-support gate. Hardcoded rather
// than taken from KFD_IOC_PROFILER_VERSION_NUM in the vendored kfd_ioctl.h above: that macro is
// still 1 on this branch (the dispatch-mode kernel/SDK work that bumps it to 2 hasn't merged
// here yet), even though the KFD driver under test already implements version-2 behavior.
constexpr uint32_t kPerfmonClockMinVersion = 2;

inline int OpenKfd() {
  static int fd = []() -> int {
    int f = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if (f < 0) {
      fprintf(stderr,
              "Warning: failed to open /dev/kfd for dispatch-mode perfmon-clock enable "
              "(errno=%d: %s)\n",
              errno, strerror(errno));
    }
    return f;
  }();
  return fd;
}

inline unsigned long ProfilerIoctlRequest(int fd) {
  static const unsigned long request = [fd]() -> unsigned long {
    kfd_ioctl_get_version_args args{};
    if (fd < 0 || ioctl(fd, AMDKFD_IOC_GET_VERSION, &args) != 0)
      return AMDKFD_IOC_PROFILER;
    if (args.major_version > 1 || (args.major_version == 1 && args.minor_version >= 23))
      return AMDKFD_IOWR(0x28, struct kfd_ioctl_profiler_args);
    return AMDKFD_IOC_PROFILER;
  }();
  return request;
}

// Maps an HSA driver node id (HSA_AMD_AGENT_INFO_DRIVER_NODE_ID) to the KFD gpu_id the
// profiler ioctl expects, via the same per-node sysfs file KFD itself exposes.
inline uint32_t GpuIdForDriverNode(uint32_t driver_node_id) {
  char path[128];
  snprintf(path, sizeof(path), "/sys/class/kfd/kfd/topology/nodes/%u/gpu_id", driver_node_id);
  FILE* f = fopen(path, "r");
  if (!f) return 0;
  uint32_t gpu_id = 0;
  if (fscanf(f, "%u", &gpu_id) != 1) gpu_id = 0;
  fclose(f);
  return gpu_id;
}

// Issues the dispatch-mode PMC request (KFD_IOC_PROFILER_PMC with lock=0, perfcount_enable=1),
// which forces the perfmon clock on for this process without requiring CAP_PERFMON, on kernels
// new enough to support it (profiler interface v2+). There is no corresponding "disable": the
// kernel only clears the request when this process's KFD context is torn down. ctrl_v3 is an
// NPI bring-up diagnostic, not a production path, so every skip/failure below is reported to
// stderr rather than silently no-op'd -- callers need to know exactly why counters/PC sampling
// might still read 0 on a new ASIC, not just that they do.
inline void EnableDispatchPerfmonClock(uint32_t driver_node_id) {
  int fd = OpenKfd();
  if (fd < 0) return;

  unsigned long request = ProfilerIoctlRequest(fd);

  kfd_ioctl_profiler_args ver_args{};
  ver_args.op = KFD_IOC_PROFILER_VERSION;
  if (ioctl(fd, request, &ver_args) != 0) {
    fprintf(stderr,
            "Warning: dispatch-mode perfmon-clock enable skipped for driver_node_id=%u: "
            "KFD_IOC_PROFILER_VERSION query failed (errno=%d: %s)\n",
            driver_node_id, errno, strerror(errno));
    return;
  }
  if (ver_args.version < kPerfmonClockMinVersion) {
    fprintf(stderr,
            "Warning: dispatch-mode perfmon-clock enable skipped for driver_node_id=%u: "
            "KFD profiler interface version %u < required %u\n",
            driver_node_id, ver_args.version, kPerfmonClockMinVersion);
    return;
  }

  uint32_t gpu_id = GpuIdForDriverNode(driver_node_id);
  if (gpu_id == 0) {
    fprintf(stderr,
            "Warning: dispatch-mode perfmon-clock enable skipped for driver_node_id=%u: "
            "failed to resolve KFD gpu_id\n",
            driver_node_id);
    return;
  }

  kfd_ioctl_profiler_args pmc_args{};
  pmc_args.op                   = KFD_IOC_PROFILER_PMC;
  pmc_args.pmc.gpu_id           = gpu_id;
  pmc_args.pmc.lock             = 0;
  pmc_args.pmc.perfcount_enable = 1;
  if (ioctl(fd, request, &pmc_args) != 0) {
    fprintf(stderr,
            "Warning: dispatch-mode perfmon-clock enable failed (gpu_id=%u, errno=%d: %s)\n",
            gpu_id, errno, strerror(errno));
  }
}

}  // namespace kfd_pmc

#endif  // SRC_UTIL_KFD_PMC_ENABLE_H_
