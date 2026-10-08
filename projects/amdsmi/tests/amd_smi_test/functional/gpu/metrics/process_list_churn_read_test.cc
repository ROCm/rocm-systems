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
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "kfd_helper_test_util.h"
#include "rocm_smi/kfd_ioctl.h"
#include "test_common.h"

namespace {

using amdsmi_test::FindHelperKfdPids;
using amdsmi_test::HelperReport;
using amdsmi_test::HelperStopper;
using amdsmi_test::IsEnvironmentFailure;
using amdsmi_test::kIocAcquireVm;
using amdsmi_test::kIocAllocMemoryOfGpu;
using Clock = std::chrono::steady_clock;

constexpr size_t kMaxProcs = 512;
constexpr uint64_t kPageSize = 4096;
constexpr int kChurnThreads = 3;
constexpr auto kChurnTime = std::chrono::seconds(8);
// Fewest churn processes that must bind for a run to count; AMDSMI CI runs bound 285 to 486.
constexpr int kMinChurnProcesses = 100;

// Makes the calling process a GPU process: opens /dev/kfd, binds the render node
// and, given a size, allocates that much VRAM, which the GPU's process list needs
// to count the process as using it, and reports the step that failed, with errno.
// System calls only, so a forked child may run it.
HelperReport BindKfd(uint32_t kfd_gpu_id, const char* render_node, uint64_t vram) {
  const int kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (kfd < 0) return {HelperReport::kOpenKfd, errno};
  const int drm = open(render_node, O_RDWR | O_CLOEXEC);
  if (drm < 0) return {HelperReport::kOpenRenderNode, errno};
  kfd_ioctl_acquire_vm_args vm{};
  vm.drm_fd = static_cast<uint32_t>(drm);
  vm.gpu_id = kfd_gpu_id;
  if (ioctl(kfd, kIocAcquireVm, &vm) != 0) return {HelperReport::kAcquireVm, errno};
  if (vram == 0) return {HelperReport::kReady, 0};
  kfd_ioctl_alloc_memory_of_gpu_args mem{};
  mem.va_addr = 1ULL << 40;
  mem.size = vram;
  mem.gpu_id = kfd_gpu_id;
  mem.flags =
      static_cast<uint32_t>(KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                            KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE);
  if (ioctl(kfd, kIocAllocMemoryOfGpu, &mem) != 0) return {HelperReport::kAllocVram, errno};
  return {HelperReport::kReady, 0};
}

// One short-lived GPU process: it binds, holds for 2 ms, then exits. Returns
// whether it bound.
bool RunChurnProcess(uint32_t kfd_gpu_id, const char* render_node) {
  const pid_t child = fork();
  if (child == 0) {
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    const bool bound = BindKfd(kfd_gpu_id, render_node, 0).step == HelperReport::kReady;
    const timespec hold{0, 2000000};
    nanosleep(&hold, nullptr);
    _exit(bound ? 0 : 1);
  }
  int wstatus = 0;
  return child > 0 && waitpid(child, &wstatus, 0) == child && WIFEXITED(wstatus) &&
         WEXITSTATUS(wstatus) == 0;
}

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

  // The helper is a GPU process that lives through the whole test. The `tag`
  // pages tell concurrent runs of this test apart, and the 6 MiB base keeps it
  // apart from the other process-list tests' helpers.
  const uint64_t helper_vram = (6ULL << 20) + (std::random_device{}() % 255 + 1) * kPageSize;
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
    const HelperReport report = BindKfd(kfd_gpu_id, render_node.c_str(), helper_vram);
    if (write(fds[1], &report, sizeof(report)) != sizeof(report) ||
        report.step != HelperReport::kReady) {
      _exit(1);
    }
    for (;;) pause();
  }
  close(fds[1]);
  HelperStopper stopper{helper, helper};
  HelperReport report;
  pollfd ready{fds[0], POLLIN, 0};
  if (poll(&ready, 1, 30000) == 1) {
    if (read(fds[0], &report, sizeof(report)) != sizeof(report)) report.step = -1;
  }
  close(fds[0]);
  if (report.step == -1) FAIL() << "The helper exited or did not report within 30 s";
  if (report.step != HelperReport::kReady) {
    if (IsEnvironmentFailure(report)) {
      GTEST_SKIP() << "Cannot use the GPU through KFD: step " << report.step << ", errno "
                   << report.err;
    }
    FAIL() << "Helper setup failed: step " << report.step << ", errno " << report.err;
  }
  const std::vector<pid_t> kfd_pids = FindHelperKfdPids(helper, kfd_gpu_id, helper_vram);
  // A second match is another run of this test that drew the same tag.
  if (kfd_pids.size() > 1) GTEST_SKIP() << "Another process holds the helper's exact VRAM";
  ASSERT_EQ(kfd_pids.size(), 1u) << "KFD does not report exactly the helper's VRAM";
  const pid_t kfd_pid = kfd_pids[0];
  stopper.kfd_pid = kfd_pid;

  std::vector<amdsmi_proc_info_t> procs(kMaxProcs);
  auto query = [&](pid_t pid, bool* listed) {
    uint32_t count = static_cast<uint32_t>(procs.size());
    amdsmi_status_t st = amdsmi_get_gpu_process_list(gpu, &count, procs.data());
    if (st == AMDSMI_STATUS_OUT_OF_RESOURCES) {
      procs.resize(count + 64);
      count = static_cast<uint32_t>(procs.size());
      st = amdsmi_get_gpu_process_list(gpu, &count, procs.data());
    }
    *listed = false;
    for (uint32_t i = 0; st == AMDSMI_STATUS_SUCCESS && i < count && i < procs.size(); ++i) {
      *listed |= procs[i].pid == static_cast<uint32_t>(pid);
    }
    return st;
  };
  // A list cached before the helper started is served for up to
  // AMDSMI_PROCESS_INFO_CACHE_MS, so wait until it shows the helper.
  const char* cache_ms = std::getenv("AMDSMI_PROCESS_INFO_CACHE_MS");
  bool listed = false;
  for (const auto give_up =
           Clock::now() + std::chrono::seconds(10) +
           std::chrono::milliseconds(cache_ms ? std::strtoul(cache_ms, nullptr, 10) : 0);
       !listed && Clock::now() < give_up;) {
    if (query(kfd_pid, &listed) != AMDSMI_STATUS_SUCCESS || !listed) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  // In a container rocm_smi may list this namespace's PIDs instead of KFD's.
  bool listed_by_own_pid = false;
  if (!listed && kfd_pid != helper && query(helper, &listed_by_own_pid) == AMDSMI_STATUS_SUCCESS &&
      listed_by_own_pid) {
    GTEST_SKIP() << "The library lists the helper by this PID namespace's PID";
  }
  ASSERT_TRUE(listed) << "GPU 0 does not list the helper";

  std::atomic<bool> stop{false};
  std::atomic<int> churned{0};
  std::atomic<int> churn_failed{0};
  std::vector<std::thread> churn;
  for (int t = 0; t < kChurnThreads; ++t) {
    churn.emplace_back([&] {
      while (!stop) {
        if (RunChurnProcess(kfd_gpu_id, render_node.c_str())) {
          ++churned;
        } else {
          ++churn_failed;
        }
      }
    });
  }
  int calls = 0;
  int unlisted = 0;
  std::map<amdsmi_status_t, int> errors;
  for (const auto end = Clock::now() + kChurnTime; Clock::now() < end; ++calls) {
    const amdsmi_status_t st = query(kfd_pid, &listed);
    if (st != AMDSMI_STATUS_SUCCESS) {
      ++errors[st];
    } else if (!listed) {
      ++unlisted;
    }
  }
  stop = true;
  for (auto& t : churn) t.join();

  std::ostringstream failed_calls;
  int failed = 0;
  for (const auto& [st, n] : errors) {
    failed_calls << " status " << st << " x" << n;
    failed += n;
  }
  // Processes that start or exit while the list is read never fail the call.
  EXPECT_EQ(failed, 0) << "Failed calls:" << failed_calls.str() << " of " << calls;
  EXPECT_EQ(unlisted, 0) << "Lists without the helper, of " << calls << " calls";
  // Without GPU processes coming and going the test proves nothing.
  EXPECT_GE(churned, kMinChurnProcesses)
      << churned << " churn processes bound to the GPU, " << churn_failed << " did not";
  EXPECT_GT(churned, churn_failed)
      << churned << " churn processes bound to the GPU, " << churn_failed << " did not";
  IF_VERB(STANDARD) {
    std::cout << "\t**" << calls << " calls while " << churned
              << " GPU processes started and exited" << std::endl;
  }
}
