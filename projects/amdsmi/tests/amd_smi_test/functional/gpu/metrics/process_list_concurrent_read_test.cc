// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "process_list_concurrent_read.h"

#include <dirent.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rocm_smi/kfd_ioctl.h"
#include "test_common.h"

namespace {

using Clock = std::chrono::steady_clock;

// Kernel UAPI numbers of two ioctls missing from rocm_smi's kfd_ioctl.h. The
// system <linux/kfd_ioctl.h> is not used: on some distributions it includes
// <drm/drm.h>, which is not installed.
constexpr unsigned long kIocAcquireVm = AMDKFD_IOW(0x15, struct kfd_ioctl_acquire_vm_args);
constexpr unsigned long kIocAllocMemoryOfGpu =
    AMDKFD_IOWR(0x16, struct kfd_ioctl_alloc_memory_of_gpu_args);
static_assert(sizeof(kfd_ioctl_acquire_vm_args) == 8 &&
                  sizeof(kfd_ioctl_alloc_memory_of_gpu_args) == 40,
              "KFD UAPI layout changed");

const char kKfdProcRoot[] = "/sys/class/kfd/kfd/proc/";
constexpr size_t kMaxProcs = 512;
constexpr auto kPhaseBudget = std::chrono::seconds(20);
const uint64_t kPageSize = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));

// VRAM the helper holds on GPU `gpu`: 2 MiB steps tell the GPUs apart, and
// `tag` pages tell concurrent runs of this test apart.
uint64_t HelperVram(size_t gpu, uint32_t tag) { return (gpu + 1) * (2ULL << 20) + tag * kPageSize; }

struct HelperGpu {
  uint32_t kfd_gpu_id;
  std::string render_node;
};

// The helper's setup steps; it reports the one that failed, with errno.
enum HelperStep { kReady, kOpenKfd, kOpenRenderNode, kAcquireVm, kAllocVram };

// Runs in the forked helper, system calls only: allocates HelperVram(i, tag)
// on GPU i through KFD, writes {step, errno} to `fd`, then waits to be killed.
[[noreturn]] void RunHelper(const std::vector<HelperGpu>& gpus, uint32_t tag, pid_t parent,
                            int fd) {
  // Never outlive the test, even if it crashes. The signal follows the forking
  // thread, which is gtest's main thread.
  if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) _exit(1);
  int report[2] = {kReady, 0};
  const int kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (kfd < 0) report[0] = kOpenKfd;
  for (size_t i = 0; report[0] == kReady && i < gpus.size(); ++i) {
    const int drm = open(gpus[i].render_node.c_str(), O_RDWR | O_CLOEXEC);
    kfd_ioctl_acquire_vm_args vm{};
    vm.drm_fd = static_cast<uint32_t>(drm);
    vm.gpu_id = gpus[i].kfd_gpu_id;
    kfd_ioctl_alloc_memory_of_gpu_args mem{};
    mem.va_addr = (1ULL << 40) + (static_cast<uint64_t>(i) << 32);
    mem.size = HelperVram(i, tag);
    mem.gpu_id = gpus[i].kfd_gpu_id;
    mem.flags =
        static_cast<uint32_t>(KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                              KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE);
    if (drm < 0) {
      report[0] = kOpenRenderNode;
    } else if (ioctl(kfd, kIocAcquireVm, &vm) != 0) {
      report[0] = kAcquireVm;
    } else if (ioctl(kfd, kIocAllocMemoryOfGpu, &mem) != 0) {
      report[0] = kAllocVram;
    }
  }
  if (report[0] != kReady) report[1] = errno;
  if (write(fd, report, sizeof(report)) != sizeof(report) || report[0] != kReady) _exit(1);
  for (;;) pause();
}

// Only these mean the machine cannot run the test: no access to the device
// files, or not enough memory.
bool IsEnvironmentFailure(const int report[2]) {
  const int step = report[0];
  const int err = report[1];
  if (err == ENOMEM) return true;
  return (step == kOpenKfd || step == kOpenRenderNode) &&
         (err == EACCES || err == EPERM || err == ENOENT || err == ENODEV || err == ENXIO);
}

// Names of the processes KFD tracks, sorted.
std::vector<std::string> KfdProcesses() {
  std::vector<std::string> names;
  if (DIR* dir = opendir(kKfdProcRoot)) {
    while (const dirent* entry = readdir(dir)) {
      if (entry->d_name[0] != '.') names.emplace_back(entry->d_name);
    }
    closedir(dir);
  }
  std::sort(names.begin(), names.end());
  return names;
}

uint64_t KfdVram(const std::string& pid, uint32_t kfd_gpu_id) {
  std::ifstream file(kKfdProcRoot + pid + "/vram_" + std::to_string(kfd_gpu_id));
  uint64_t vram = 0;
  return (file >> vram) ? vram : 0;
}

// KFD names processes by host PID, which differs from fork()'s result inside a
// container's PID namespace, so find the entries holding exactly the helper's
// VRAM.
std::vector<pid_t> FindHelperKfdPids(const std::vector<HelperGpu>& gpus, uint32_t tag,
                                     pid_t helper) {
  auto holds_helper_vram = [&](const std::string& pid) {
    for (size_t i = 0; i < gpus.size(); ++i) {
      if (KfdVram(pid, gpus[i].kfd_gpu_id) != HelperVram(i, tag)) return false;
    }
    return true;
  };
  if (holds_helper_vram(std::to_string(helper))) return {helper};
  std::vector<pid_t> found;
  for (const std::string& name : KfdProcesses()) {
    const bool is_pid =
        std::all_of(name.begin(), name.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
    if (is_pid && holds_helper_vram(name)) found.push_back(static_cast<pid_t>(std::stol(name)));
  }
  return found;
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

enum Outcome { kRight, kWrongVram, kMissing, kEmpty, kError, kNumOutcomes };

// Looks the helper up in one GPU's process list.
Outcome QueryOnce(amdsmi_processor_handle gpu, pid_t pid, uint64_t vram,
                  std::vector<amdsmi_proc_info_t>* procs) {
  uint32_t count = static_cast<uint32_t>(procs->size());
  amdsmi_status_t status = amdsmi_get_gpu_process_list(gpu, &count, procs->data());
  if (status == AMDSMI_STATUS_OUT_OF_RESOURCES) {
    procs->resize(count + 64);
    count = static_cast<uint32_t>(procs->size());
    status = amdsmi_get_gpu_process_list(gpu, &count, procs->data());
  }
  if (status != AMDSMI_STATUS_SUCCESS) return kError;
  if (count == 0) return kEmpty;
  for (uint32_t i = 0; i < count && i < procs->size(); ++i) {
    if ((*procs)[i].pid == static_cast<uint32_t>(pid)) {
      return (*procs)[i].mem == vram ? kRight : kWrongVram;
    }
  }
  return kMissing;
}

struct Tally {
  std::atomic<int> count[kNumOutcomes] = {};
  int failures() const {
    return count[kWrongVram] + count[kMissing] + count[kEmpty] + count[kError];
  }
  std::string str() const {
    std::ostringstream ss;
    ss << count[kWrongVram] << " with another VRAM value, " << count[kMissing]
       << " without the helper, " << count[kEmpty] << " empty, " << count[kError]
       << " failed calls, of " << failures() + count[kRight];
    return ss.str();
  }
};

}  // namespace

TestProcessListConcurrentRead::TestProcessListConcurrentRead() : TestBase() {
  set_title("AMDSMI Process List Concurrent Read Test");
  set_description(
      "This test verifies that amdsmi_get_gpu_process_list reports a process "
      "that uses every GPU with each GPU's own memory usage, both when the GPUs "
      "are queried in turn and when several threads query them at once.");
}

TestProcessListConcurrentRead::~TestProcessListConcurrentRead(void) {}

void TestProcessListConcurrentRead::SetUp(void) {
  TestBase::SetUp();
  return;
}

void TestProcessListConcurrentRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestProcessListConcurrentRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestProcessListConcurrentRead::Close() { TestBase::Close(); }

void TestProcessListConcurrentRead::Run(void) {
  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  const size_t num_gpus = num_monitor_devs();
  if (num_gpus == 0) GTEST_SKIP() << "No GPUs found";
  std::vector<HelperGpu> gpus;
  for (size_t i = 0; i < num_gpus; ++i) {
    amdsmi_kfd_info_t kfd_info{};
    amdsmi_enumeration_info_t enum_info{};
    amdsmi_status_t status = amdsmi_get_gpu_kfd_info(processor_handles_[i], &kfd_info);
    if (status == AMDSMI_STATUS_SUCCESS) {
      status = amdsmi_get_gpu_enumeration_info(processor_handles_[i], &enum_info);
    }
    if (status == AMDSMI_STATUS_NOT_SUPPORTED) GTEST_SKIP() << "GPU " << i << " has no KFD info";
    ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS) << "GPU " << i;
    // The library reports a KFD id of UINT64_MAX, or a render minor of 0 or
    // UINT32_MAX, when it does not know the node.
    if (kfd_info.kfd_id == std::numeric_limits<uint64_t>::max() || enum_info.drm_render == 0 ||
        enum_info.drm_render == std::numeric_limits<uint32_t>::max()) {
      GTEST_SKIP() << "GPU " << i << " has no known KFD node or render node";
    }
    gpus.push_back({static_cast<uint32_t>(kfd_info.kfd_id),
                    "/dev/dri/renderD" + std::to_string(enum_info.drm_render)});
  }

  const uint32_t tag = std::random_device{}() % 255 + 1;
  int fds[2];
  ASSERT_EQ(pipe(fds), 0);
  const pid_t parent = getpid();
  const pid_t helper = fork();
  if (helper < 0) {
    close(fds[0]);
    close(fds[1]);
    FAIL() << "fork failed, errno " << errno;
  }
  if (helper == 0) {
    close(fds[0]);
    RunHelper(gpus, tag, parent, fds[1]);
  }
  close(fds[1]);
  HelperStopper stopper{helper, helper};
  int report[2] = {-1, 0};  // stays -1 without a complete report
  pollfd ready{fds[0], POLLIN, 0};
  if (poll(&ready, 1, 30000) == 1) {
    if (read(fds[0], report, sizeof(report)) != sizeof(report)) report[0] = -1;
  }
  close(fds[0]);
  if (report[0] == -1) FAIL() << "The helper exited or did not report within 30 s";
  if (report[0] != kReady) {
    if (IsEnvironmentFailure(report)) {
      GTEST_SKIP() << "Cannot use the GPU through KFD: step " << report[0] << ", errno "
                   << report[1];
    }
    FAIL() << "Helper setup failed: step " << report[0] << ", errno " << report[1];
  }
  const std::vector<pid_t> kfd_pids = FindHelperKfdPids(gpus, tag, helper);
  // A second match is another run of this test that drew the same tag.
  if (kfd_pids.size() > 1) GTEST_SKIP() << "Another process holds the helper's exact VRAM";
  ASSERT_EQ(kfd_pids.size(), 1u) << "KFD does not report exactly the helper's VRAM";
  const pid_t kfd_pid = kfd_pids[0];
  stopper.kfd_pid = kfd_pid;

  // Lists cached before the helper started are served for up to
  // AMDSMI_PROCESS_INFO_CACHE_MS, so wait until every GPU lists it.
  const char* cache_ms = std::getenv("AMDSMI_PROCESS_INFO_CACHE_MS");
  const auto listed_by =
      Clock::now() + std::chrono::seconds(10) +
      std::chrono::milliseconds(cache_ms ? std::strtoul(cache_ms, nullptr, 10) : 0);
  std::vector<amdsmi_proc_info_t> procs(kMaxProcs);
  for (size_t i = 0; i < num_gpus; ++i) {
    Outcome outcome = QueryOnce(processor_handles_[i], kfd_pid, HelperVram(i, tag), &procs);
    while (outcome != kRight && Clock::now() < listed_by) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      outcome = QueryOnce(processor_handles_[i], kfd_pid, HelperVram(i, tag), &procs);
    }
    // In a container rocm_smi may list this namespace's PIDs instead of KFD's.
    const Outcome by_own_pid =
        kfd_pid == helper ? kMissing
                          : QueryOnce(processor_handles_[i], helper, HelperVram(i, tag), &procs);
    if (outcome != kRight && (by_own_pid == kRight || by_own_pid == kWrongVram)) {
      GTEST_SKIP() << "The library lists the helper by this PID namespace's PID";
    }
    ASSERT_EQ(outcome, kRight) << "GPU " << i << " does not list the helper with its VRAM";
  }

  // Query the GPUs in turn: each list must carry that GPU's own numbers. At the
  // default cache period this mostly checks attribution; a cross-GPU mix-up of
  // cached data also shows here with AMDSMI_PROCESS_INFO_CACHE_MS=100.
  Tally in_turn;
  const int rounds = std::max(5, 400 / static_cast<int>(num_gpus));
  const auto in_turn_end = Clock::now() + kPhaseBudget;
  for (int round = 0; round < rounds && Clock::now() < in_turn_end; ++round) {
    for (size_t i = 0; i < num_gpus && Clock::now() < in_turn_end; ++i) {
      ++in_turn.count[QueryOnce(processor_handles_[i], kfd_pid, HelperVram(i, tag), &procs)];
    }
  }
  EXPECT_EQ(in_turn.failures(), 0) << "Queries in turn: " << in_turn.str();

  // Two unsynchronized readers per GPU.
  Tally concurrent;
  const size_t num_readers = 2 * num_gpus;
  const int calls = std::max(20, 4000 / static_cast<int>(num_readers));
  const auto concurrent_end = Clock::now() + kPhaseBudget;
  std::vector<std::thread> readers;
  for (size_t r = 0; r < num_readers; ++r) {
    readers.emplace_back([&, gpu = r % num_gpus] {
      std::vector<amdsmi_proc_info_t> list(kMaxProcs);
      for (int c = 0; c < calls && Clock::now() < concurrent_end; ++c) {
        ++concurrent
              .count[QueryOnce(processor_handles_[gpu], kfd_pid, HelperVram(gpu, tag), &list)];
      }
    });
  }
  for (auto& reader : readers) reader.join();
  EXPECT_EQ(concurrent.failures(), 0) << "Concurrent queries: " << concurrent.str();
  IF_VERB(STANDARD) {
    std::cout << "\t**Queries in turn: " << in_turn.str()
              << "\n\t**Concurrent queries: " << concurrent.str() << std::endl;
  }
}
