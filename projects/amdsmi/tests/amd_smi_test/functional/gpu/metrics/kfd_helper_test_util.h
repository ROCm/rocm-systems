// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared by the process-list tests that fork a helper and make it a GPU process
// through KFD: the ioctls it needs, its setup report and the rule for which
// setup failures skip a test, and finding and stopping the helper.

#ifndef AMDSMI_TESTS_FUNCTIONAL_GPU_METRICS_KFD_HELPER_TEST_UTIL_H_
#define AMDSMI_TESTS_FUNCTIONAL_GPU_METRICS_KFD_HELPER_TEST_UTIL_H_

#include <dirent.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "rocm_smi/kfd_ioctl.h"

namespace amdsmi_test {

// Kernel UAPI numbers of two ioctls missing from rocm_smi's kfd_ioctl.h. The
// system <linux/kfd_ioctl.h> is not used: on some distributions it includes
// <drm/drm.h>, which is not installed.
inline constexpr unsigned long kIocAcquireVm = AMDKFD_IOW(0x15, struct kfd_ioctl_acquire_vm_args);
inline constexpr unsigned long kIocAllocMemoryOfGpu =
    AMDKFD_IOWR(0x16, struct kfd_ioctl_alloc_memory_of_gpu_args);
static_assert(sizeof(kfd_ioctl_acquire_vm_args) == 8 &&
                  sizeof(kfd_ioctl_alloc_memory_of_gpu_args) == 40,
              "KFD UAPI layout changed");

inline constexpr char kKfdProcRoot[] = "/sys/class/kfd/kfd/proc/";

// What a forked helper reports once set up: the step that failed, with errno.
struct HelperReport {
  enum Step { kReady, kOpenKfd, kOpenRenderNode, kAcquireVm, kAllocVram, kDupFd };
  int step = -1;  // stays -1 without a complete report
  int err = 0;
};

// Only these mean the machine cannot run the test: no access to the device
// files, or not enough memory. Any other failed step fails the test.
inline bool IsEnvironmentFailure(const HelperReport& report) {
  if (report.err == ENOMEM) return true;
  return (report.step == HelperReport::kOpenKfd || report.step == HelperReport::kOpenRenderNode) &&
         (report.err == EACCES || report.err == EPERM || report.err == ENOENT ||
          report.err == ENODEV || report.err == ENXIO);
}

// The VRAM KFD holds for process `pid` on the GPU with KFD id `kfd_gpu_id`, or 0.
inline uint64_t KfdVram(const std::string& pid, uint32_t kfd_gpu_id) {
  std::ifstream file(kKfdProcRoot + pid + "/vram_" + std::to_string(kfd_gpu_id));
  uint64_t vram = 0;
  return (file >> vram) ? vram : 0;
}

// KFD names processes by host PID, which differs from fork()'s result inside a
// container's PID namespace, so find the KFD processes holding exactly the
// helper's VRAM, as `holds_helper_vram(pid)` tells.
template <typename HoldsHelperVram>
std::vector<pid_t> FindHelperKfdPids(pid_t helper, HoldsHelperVram holds_helper_vram) {
  if (holds_helper_vram(std::to_string(helper))) return {helper};
  std::vector<pid_t> found;
  if (DIR* dir = opendir(kKfdProcRoot)) {
    while (const dirent* entry = readdir(dir)) {
      const std::string name = entry->d_name;
      const bool is_pid =
          !name.empty() && std::all_of(name.begin(), name.end(),
                                       [](unsigned char c) { return std::isdigit(c) != 0; });
      if (is_pid && holds_helper_vram(name)) found.push_back(static_cast<pid_t>(std::stol(name)));
    }
    closedir(dir);
  }
  return found;
}

// The same for a helper that holds `vram` on one GPU.
inline std::vector<pid_t> FindHelperKfdPids(pid_t helper, uint32_t kfd_gpu_id, uint64_t vram) {
  return FindHelperKfdPids(
      helper, [&](const std::string& pid) { return KfdVram(pid, kfd_gpu_id) == vram; });
}

// Kills the helper on scope exit and waits until KFD has released it, so the
// tests that follow (partition changes, for example) do not find it.
struct HelperStopper {
  pid_t pid;
  pid_t kfd_pid;
  ~HelperStopper() {
    kill(pid, SIGKILL);
    const std::string path = kKfdProcRoot + std::to_string(kfd_pid);
    for (int i = 0; i < 500; ++i) {
      if (waitpid(pid, nullptr, WNOHANG) != 0 && access(path.c_str(), F_OK) != 0) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
};

}  // namespace amdsmi_test

#endif  // AMDSMI_TESTS_FUNCTIONAL_GPU_METRICS_KFD_HELPER_TEST_UTIL_H_
