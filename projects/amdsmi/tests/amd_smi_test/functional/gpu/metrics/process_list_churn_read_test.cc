// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "process_list_churn_read.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rocm_smi/kfd_ioctl.h"
#include "test_common.h"

namespace {

using Clock = std::chrono::steady_clock;

// Kernel UAPI numbers of two ioctls missing from rocm_smi's kfd_ioctl.h.
constexpr unsigned long kIocAcquireVm = AMDKFD_IOW(0x15, struct kfd_ioctl_acquire_vm_args);
constexpr unsigned long kIocAllocMemoryOfGpu =
    AMDKFD_IOWR(0x16, struct kfd_ioctl_alloc_memory_of_gpu_args);
static_assert(sizeof(kfd_ioctl_acquire_vm_args) == 8 &&
                  sizeof(kfd_ioctl_alloc_memory_of_gpu_args) == 40,
              "KFD UAPI layout changed");

const char kKfdProcRoot[] = "/sys/class/kfd/kfd/proc/";
constexpr size_t kMaxProcs = 512;
constexpr int kChurnThreads = 3;
constexpr auto kChurnTime = std::chrono::seconds(8);

// Makes the calling process a GPU process: opens /dev/kfd, binds the render node
// and, given a size, allocates that much VRAM, which the GPU's process list needs
// to count the process as using it. System calls only, so a forked child may run
// it. Returns errno, or 0.
int BindKfd(uint32_t kfd_gpu_id, const char* render_node, uint64_t vram) {
  const int kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (kfd < 0) return errno;
  const int drm = open(render_node, O_RDWR | O_CLOEXEC);
  if (drm < 0) return errno;
  kfd_ioctl_acquire_vm_args vm{};
  vm.drm_fd = static_cast<uint32_t>(drm);
  vm.gpu_id = kfd_gpu_id;
  if (ioctl(kfd, kIocAcquireVm, &vm) != 0) return errno;
  if (vram == 0) return 0;
  kfd_ioctl_alloc_memory_of_gpu_args mem{};
  mem.va_addr = 1ULL << 40;
  mem.size = vram;
  mem.gpu_id = kfd_gpu_id;
  mem.flags =
      static_cast<uint32_t>(KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                            KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE);
  return ioctl(kfd, kIocAllocMemoryOfGpu, &mem) == 0 ? 0 : errno;
}

// One short-lived GPU process: it binds, holds for 2 ms, then exits.
void RunChurnProcess(uint32_t kfd_gpu_id, const char* render_node) {
  const pid_t child = fork();
  if (child == 0) {
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    BindKfd(kfd_gpu_id, render_node, 0);
    const timespec hold{0, 2000000};
    nanosleep(&hold, nullptr);
    _exit(0);
  }
  if (child > 0) waitpid(child, nullptr, 0);
}

struct HelperStopper {
  pid_t pid;
  ~HelperStopper() {
    kill(pid, SIGKILL);
    const std::string path = kKfdProcRoot + std::to_string(pid);
    for (int i = 0; i < 500; ++i) {
      if (waitpid(pid, nullptr, WNOHANG) != 0 && access(path.c_str(), F_OK) != 0) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
};

}  // namespace

TestProcessListChurnRead::TestProcessListChurnRead() : TestBase() {
  set_title("AMDSMI Process List Churn Read Test");
  set_description(
      "This test verifies that amdsmi_get_gpu_process_list keeps listing a "
      "running GPU process while other GPU processes start and exit.");
}

TestProcessListChurnRead::~TestProcessListChurnRead(void) {}

void TestProcessListChurnRead::SetUp(void) {
  TestBase::SetUp();
  return;
}

void TestProcessListChurnRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestProcessListChurnRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestProcessListChurnRead::Close() { TestBase::Close(); }

void TestProcessListChurnRead::Run(void) {
  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }
  if (num_monitor_devs() == 0) GTEST_SKIP() << "No GPUs found";

  amdsmi_processor_handle gpu = processor_handles_[0];
  amdsmi_kfd_info_t kfd_info{};
  amdsmi_enumeration_info_t enum_info{};
  amdsmi_status_t status = amdsmi_get_gpu_kfd_info(gpu, &kfd_info);
  if (status == AMDSMI_STATUS_SUCCESS) status = amdsmi_get_gpu_enumeration_info(gpu, &enum_info);
  if (status == AMDSMI_STATUS_NOT_SUPPORTED) GTEST_SKIP() << "GPU 0 has no KFD info";
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  if (kfd_info.kfd_id == std::numeric_limits<uint64_t>::max() || enum_info.drm_render == 0 ||
      enum_info.drm_render == std::numeric_limits<uint32_t>::max()) {
    GTEST_SKIP() << "GPU 0 has no known KFD node or render node";
  }
  const uint32_t kfd_gpu_id = static_cast<uint32_t>(kfd_info.kfd_id);
  const std::string render_node = "/dev/dri/renderD" + std::to_string(enum_info.drm_render);

  // The helper is a GPU process that lives through the whole test.
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
    if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) _exit(1);
    const int err = BindKfd(kfd_gpu_id, render_node.c_str(), 2ULL << 20);
    if (write(fds[1], &err, sizeof(err)) != sizeof(err) || err != 0) _exit(1);
    for (;;) pause();
  }
  close(fds[1]);
  HelperStopper stopper{helper};
  int err = -1;  // stays -1 without a report
  pollfd ready{fds[0], POLLIN, 0};
  if (poll(&ready, 1, 30000) == 1) {
    if (read(fds[0], &err, sizeof(err)) != sizeof(err)) err = -1;
  }
  close(fds[0]);
  if (err == -1) FAIL() << "The helper exited or did not report within 30 s";
  if (err == EACCES || err == EPERM || err == ENOENT || err == ENODEV || err == ENXIO ||
      err == ENOMEM) {
    GTEST_SKIP() << "Cannot use the GPU through KFD: errno " << err;
  }
  ASSERT_EQ(err, 0) << "The helper could not bind to the GPU";
  // KFD names processes by host PID, so under another PID namespace it does not
  // list the helper by the PID seen here.
  if (access((kKfdProcRoot + std::to_string(helper)).c_str(), F_OK) != 0) {
    GTEST_SKIP() << "KFD does not list the helper under its own PID";
  }

  std::vector<amdsmi_proc_info_t> procs(kMaxProcs);
  auto query = [&](bool* listed) {
    uint32_t count = static_cast<uint32_t>(procs.size());
    const amdsmi_status_t st = amdsmi_get_gpu_process_list(gpu, &count, procs.data());
    *listed = false;
    for (uint32_t i = 0; st == AMDSMI_STATUS_SUCCESS && i < count && i < procs.size(); ++i) {
      *listed |= procs[i].pid == static_cast<uint32_t>(helper);
    }
    return st;
  };
  // A list cached before the helper started is served for up to
  // AMDSMI_PROCESS_INFO_CACHE_MS, so wait until it shows the helper.
  bool listed = false;
  for (const auto give_up = Clock::now() + std::chrono::seconds(10);
       !listed && Clock::now() < give_up;) {
    if (query(&listed) != AMDSMI_STATUS_SUCCESS || !listed) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  ASSERT_TRUE(listed) << "GPU 0 does not list the helper";

  std::atomic<bool> stop{false};
  std::atomic<int> churned{0};
  std::vector<std::thread> churn;
  for (int t = 0; t < kChurnThreads; ++t) {
    churn.emplace_back([&] {
      while (!stop) {
        RunChurnProcess(kfd_gpu_id, render_node.c_str());
        ++churned;
      }
    });
  }
  int calls = 0;
  int unlisted = 0;
  std::map<amdsmi_status_t, int> errors;
  for (const auto end = Clock::now() + kChurnTime; Clock::now() < end; ++calls) {
    const amdsmi_status_t st = query(&listed);
    if (st != AMDSMI_STATUS_SUCCESS) {
      ++errors[st];
    } else if (!listed) {
      ++unlisted;
    }
  }
  stop = true;
  for (auto& t : churn) t.join();

  std::ostringstream failed_calls;
  for (const auto& [st, n] : errors) failed_calls << " status " << st << " x" << n;
  EXPECT_TRUE(errors.empty()) << "Failed calls:" << failed_calls.str() << " of " << calls;
  EXPECT_EQ(unlisted, 0) << "Lists without the helper, of " << calls << " calls";
  IF_VERB(STANDARD) {
    std::cout << "\t**" << calls << " calls while " << churned
              << " GPU processes started and exited" << std::endl;
  }
}
