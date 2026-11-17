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
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
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
inline constexpr char kKfdNodesRoot[] = "/sys/class/kfd/kfd/topology/nodes/";

// The value the library reports for a statistic that is not available.
inline constexpr uint32_t kNotAvailable32 = std::numeric_limits<uint32_t>::max();
inline constexpr uint64_t kNotAvailable64 = std::numeric_limits<uint64_t>::max();

// What a forked helper reports once set up: the step that failed, with errno.
struct HelperReport {
  enum Step { kReady, kOpenKfd, kOpenRenderNode, kAcquireVm, kAllocVram, kDupFd, kMapVram };
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

// Memory sizes in bytes.
struct DrmMemory {
  uint64_t vram = 0;
  uint64_t gtt = 0;
  uint64_t cpu = 0;
};

// The memory that the DRM client which process `pid` opened on `render_node`
// reports in its fdinfo; all 0 without one.
inline DrmMemory DrmMemoryBytes(pid_t pid, const std::string& render_node) {
  const std::string dir = "/proc/" + std::to_string(pid);
  DIR* fds = opendir((dir + "/fd").c_str());
  if (fds == nullptr) return {};
  DrmMemory kib;
  while (const dirent* entry = readdir(fds)) {
    char target[256] = {};
    const std::string link = dir + "/fd/" + entry->d_name;
    if (readlink(link.c_str(), target, sizeof(target) - 1) <= 0 || render_node != target) continue;
    std::ifstream fdinfo(dir + "/fdinfo/" + entry->d_name);
    for (std::string line; std::getline(fdinfo, line);) {
      sscanf(line.c_str(), "drm-memory-vram: %" SCNu64, &kib.vram);
      sscanf(line.c_str(), "drm-memory-gtt: %" SCNu64, &kib.gtt);
      sscanf(line.c_str(), "drm-memory-cpu: %" SCNu64, &kib.cpu);
    }
    break;
  }
  closedir(fds);
  return {kib.vram * 1024, kib.gtt * 1024, kib.cpu * 1024};
}

// Whether amdgpu places what KFD allocates as VRAM on the GPU with KFD id
// `kfd_gpu_id` and render node `render_node` in GTT, as it does on an APU with
// less VRAM than GTT. KFD then offers more local memory than the GPU has VRAM.
inline bool KfdVramIsInGtt(uint32_t kfd_gpu_id, const std::string& render_node) {
  uint64_t kfd_memory = 0;
  if (DIR* nodes = opendir(kKfdNodesRoot)) {
    while (const dirent* node = readdir(nodes)) {
      const std::string dir = kKfdNodesRoot + std::string(node->d_name);
      std::ifstream gpu_id_file(dir + "/gpu_id");
      uint64_t gpu_id = 0;
      if (node->d_name[0] == '.' || !(gpu_id_file >> gpu_id) || gpu_id != kfd_gpu_id) continue;
      if (DIR* banks = opendir((dir + "/mem_banks").c_str())) {
        while (const dirent* bank = readdir(banks)) {
          if (bank->d_name[0] == '.') continue;
          std::ifstream properties(dir + "/mem_banks/" + bank->d_name + "/properties");
          for (std::string line; std::getline(properties, line);) {
            uint64_t size = 0;
            if (sscanf(line.c_str(), "size_in_bytes %" SCNu64, &size) == 1) kfd_memory += size;
          }
        }
        closedir(banks);
      }
    }
    closedir(nodes);
  }
  const std::string node_name = render_node.substr(render_node.rfind('/') + 1);
  std::ifstream vram_file("/sys/class/drm/" + node_name + "/device/mem_info_vram_total");
  uint64_t vram = 0;
  return (vram_file >> vram) && kfd_memory > vram;
}

// Whether this process runs in the initial PID namespace, where KFD's process
// directories are named by its PIDs. Elsewhere, as in a container that does not
// share the host's, the library lists this namespace's processes by their PIDs
// here, with their KFD statistics not available.
inline bool InInitialPidNamespace() {
  // The initial PID namespace has a fixed inode number (PROC_PID_INIT_INO in
  // the kernel's include/linux/proc_ns.h).
  constexpr char kInitialPidNamespace[] = "pid:[4026531836]";
  char link[64] = {};
  return readlink("/proc/self/ns/pid", link, sizeof(link) - 1) > 0 &&
         std::string(link) == kInitialPidNamespace;
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
