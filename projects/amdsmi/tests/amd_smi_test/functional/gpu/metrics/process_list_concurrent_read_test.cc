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
// This file must not include rocm_smi headers: rocm_smi/kfd_ioctl.h uses the
// same include guard and lacks the memory ioctls.
#if __has_include(<linux/kfd_ioctl.h>)
#include <linux/kfd_ioctl.h>
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

#if defined(AMDKFD_IOC_ACQUIRE_VM) && defined(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU)
namespace {

using Clock = std::chrono::steady_clock;

const char kKfdProcRoot[] = "/sys/class/kfd/kfd/proc/";
constexpr size_t kMaxProcs = 512;
constexpr auto kPhaseBudget = std::chrono::seconds(20);

// VRAM the helper holds on GPU `gpu`: 2 MiB steps tell the GPUs apart, and
// `tag` pages tell concurrent runs of this test apart.
uint64_t HelperVram(size_t gpu, uint32_t tag) { return (gpu + 1) * (2ULL << 20) + tag * 4096ULL; }

struct HelperGpu {
  uint32_t kfd_gpu_id;
  std::string render_node;
};

// Runs in the forked helper, system calls only: allocates HelperVram(i, tag)
// on GPU i through KFD, writes {0, 0} or {failed step, errno} to `fd`, then
// waits to be killed.
[[noreturn]] void RunHelper(const std::vector<HelperGpu>& gpus, uint32_t tag, pid_t parent,
                            int fd) {
  // Never outlive the test, even if it crashes. The signal follows the forking
  // thread, which is gtest's main thread.
  if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) _exit(1);
  int report[2] = {0, 0};
  const int kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (kfd < 0) report[0] = 1;
  for (size_t i = 0; report[0] == 0 && i < gpus.size(); ++i) {
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
      report[0] = 2;
    } else if (ioctl(kfd, AMDKFD_IOC_ACQUIRE_VM, &vm) != 0) {
      report[0] = 3;
    } else if (ioctl(kfd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &mem) != 0) {
      report[0] = 4;
    }
  }
  if (report[0] != 0) report[1] = errno;
  if (write(fd, report, sizeof(report)) != sizeof(report) || report[0] != 0) _exit(1);
  for (;;) pause();
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
// container's PID namespace, so find the one entry holding exactly the
// helper's VRAM. Returns 0 if there is not exactly one.
pid_t FindHelperKfdPid(const std::vector<HelperGpu>& gpus, uint32_t tag, pid_t helper) {
  auto holds_helper_vram = [&](const std::string& pid) {
    for (size_t i = 0; i < gpus.size(); ++i) {
      if (KfdVram(pid, gpus[i].kfd_gpu_id) != HelperVram(i, tag)) return false;
    }
    return true;
  };
  if (holds_helper_vram(std::to_string(helper))) return helper;
  pid_t found = 0;
  for (const std::string& name : KfdProcesses()) {
    const bool is_pid =
        std::all_of(name.begin(), name.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
    if (!is_pid || !holds_helper_vram(name)) continue;
    if (found != 0) return 0;
    found = static_cast<pid_t>(std::stol(name));
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

// Reports whether any GPU process started or exited while it ran. The library
// returns an empty list when that happens during its scan, so an empty list is
// a failure only if nothing changed.
class ChurnWatch {
 public:
  ChurnWatch() : start_(KfdProcesses()), thread_([this] { Watch(); }) {}
  ~ChurnWatch() { Stop(); }
  bool Stop() {
    done_ = true;
    if (thread_.joinable()) thread_.join();
    return changed_;
  }

 private:
  void Watch() {
    while (!done_) {
      if (KfdProcesses() != start_) changed_ = true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  const std::vector<std::string> start_;
  std::atomic<bool> done_{false};
  std::atomic<bool> changed_{false};
  std::thread thread_;
};

enum Outcome { kRight, kWrongVram, kMissing, kEmpty, kError, kNumOutcomes };

// Looks the helper up in one GPU's process list.
Outcome QueryHelper(amdsmi_processor_handle gpu, pid_t pid, uint64_t vram,
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
  int failures(bool count_empty) const {
    return count[kWrongVram] + count[kMissing] + count[kError] +
           (count_empty ? count[kEmpty].load() : 0);
  }
  std::string str() const {
    std::ostringstream ss;
    ss << count[kWrongVram] << " with another VRAM value, " << count[kMissing]
       << " without the helper, " << count[kError] << " failed calls, " << count[kEmpty]
       << " empty lists, of " << failures(true) + count[kRight];
    return ss.str();
  }
};

}  // namespace
#endif

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

#if !defined(AMDKFD_IOC_ACQUIRE_VM) || !defined(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU)
  GTEST_SKIP() << "The system headers lack the KFD memory ioctls";
#else
  const size_t num_gpus = num_monitor_devs();
  if (num_gpus == 0) GTEST_SKIP() << "No GPUs found";
  std::vector<HelperGpu> gpus;
  for (size_t i = 0; i < num_gpus; ++i) {
    amdsmi_kfd_info_t kfd_info{};
    amdsmi_enumeration_info_t enum_info{};
    if (amdsmi_get_gpu_kfd_info(processor_handles_[i], &kfd_info) != AMDSMI_STATUS_SUCCESS ||
        amdsmi_get_gpu_enumeration_info(processor_handles_[i], &enum_info) !=
            AMDSMI_STATUS_SUCCESS) {
      GTEST_SKIP() << "No KFD id or render node for GPU " << i;
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
  int report[2] = {-1, 0};  // stays -1 if the helper times out
  pollfd ready{fds[0], POLLIN, 0};
  const bool ok = poll(&ready, 1, 30000) == 1 &&
                  read(fds[0], report, sizeof(report)) == sizeof(report) && report[0] == 0;
  close(fds[0]);
  if (!ok) {
    GTEST_SKIP() << "Cannot allocate VRAM through KFD: step " << report[0] << ", errno "
                 << report[1];
  }
  const pid_t kfd_pid = FindHelperKfdPid(gpus, tag, helper);
  if (kfd_pid == 0) GTEST_SKIP() << "KFD does not report exactly the helper's VRAM";
  stopper.kfd_pid = kfd_pid;

  // Lists cached before the helper started are served for up to
  // AMDSMI_PROCESS_INFO_CACHE_MS, so wait until every GPU lists it.
  const char* cache_ms = std::getenv("AMDSMI_PROCESS_INFO_CACHE_MS");
  const auto listed_by =
      Clock::now() + std::chrono::seconds(10) +
      std::chrono::milliseconds(cache_ms ? std::strtoul(cache_ms, nullptr, 10) : 0);
  std::vector<amdsmi_proc_info_t> procs(kMaxProcs);
  for (size_t i = 0; i < num_gpus; ++i) {
    Outcome outcome = QueryHelper(processor_handles_[i], kfd_pid, HelperVram(i, tag), &procs);
    while (outcome != kRight && Clock::now() < listed_by) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      outcome = QueryHelper(processor_handles_[i], kfd_pid, HelperVram(i, tag), &procs);
    }
    // In a container rocm_smi may list this namespace's PIDs instead of KFD's.
    const Outcome by_own_pid =
        kfd_pid == helper ? kMissing
                          : QueryHelper(processor_handles_[i], helper, HelperVram(i, tag), &procs);
    if (outcome != kRight && (by_own_pid == kRight || by_own_pid == kWrongVram)) {
      GTEST_SKIP() << "The library lists the helper by this PID namespace's PID";
    }
    ASSERT_EQ(outcome, kRight) << "GPU " << i << " does not list the helper with its VRAM";
  }

  // Query the GPUs in turn: each list must carry that GPU's own numbers. At the
  // default cache period this mostly checks attribution; a cross-GPU mix-up of
  // cached data also shows here with AMDSMI_PROCESS_INFO_CACHE_MS=100.
  Tally in_turn;
  ChurnWatch in_turn_churn;
  const int rounds = std::max(5, 400 / static_cast<int>(num_gpus));
  const auto in_turn_end = Clock::now() + kPhaseBudget;
  for (int round = 0; round < rounds && Clock::now() < in_turn_end; ++round) {
    for (size_t i = 0; i < num_gpus; ++i) {
      ++in_turn.count[QueryHelper(processor_handles_[i], kfd_pid, HelperVram(i, tag), &procs)];
    }
  }
  EXPECT_EQ(in_turn.failures(!in_turn_churn.Stop()), 0) << "Queries in turn: " << in_turn.str();

  // Two unsynchronized readers per GPU.
  Tally concurrent;
  ChurnWatch concurrent_churn;
  const size_t num_readers = 2 * num_gpus;
  const int calls = std::max(20, 4000 / static_cast<int>(num_readers));
  const auto concurrent_end = Clock::now() + kPhaseBudget;
  std::vector<std::thread> readers;
  for (size_t r = 0; r < num_readers; ++r) {
    readers.emplace_back([&, gpu = r % num_gpus] {
      std::vector<amdsmi_proc_info_t> list(kMaxProcs);
      for (int c = 0; c < calls && Clock::now() < concurrent_end; ++c) {
        ++concurrent
              .count[QueryHelper(processor_handles_[gpu], kfd_pid, HelperVram(gpu, tag), &list)];
      }
    });
  }
  for (auto& reader : readers) reader.join();
  EXPECT_EQ(concurrent.failures(!concurrent_churn.Stop()), 0)
      << "Concurrent queries: " << concurrent.str();
  IF_VERB(STANDARD) {
    std::cout << "\t**Queries in turn: " << in_turn.str()
              << "\n\t**Concurrent queries: " << concurrent.str() << std::endl;
  }
#endif
}
