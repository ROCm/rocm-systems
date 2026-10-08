// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "process_memory_read.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
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

// Extra fds the helper opens on its one DRM client, as dup(), fork() or fd passing do.
constexpr int kDupFds = 3;

// The helper also reports its DRM fd, whose fdinfo the test reads.
struct MemoryHelperReport : HelperReport {
  int drm_fd = -1;
};

// Runs in the forked helper, system calls only: allocates `vram` on the GPU
// through KFD, adds kDupFds fds for the same DRM client, reports, then waits to be killed.
[[noreturn]] void RunHelper(uint32_t kfd_gpu_id, const std::string& render_node, uint64_t vram,
                            pid_t parent, int fd) {
  if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) _exit(1);
  MemoryHelperReport report;
  report.step = HelperReport::kReady;
  const int kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  const int drm = kfd < 0 ? -1 : open(render_node.c_str(), O_RDWR | O_CLOEXEC);
  kfd_ioctl_acquire_vm_args vm{};
  vm.drm_fd = static_cast<uint32_t>(drm);
  vm.gpu_id = kfd_gpu_id;
  kfd_ioctl_alloc_memory_of_gpu_args mem{};
  mem.va_addr = 1ULL << 40;
  mem.size = vram;
  mem.gpu_id = kfd_gpu_id;
  mem.flags =
      static_cast<uint32_t>(KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                            KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE);
  if (kfd < 0) {
    report.step = HelperReport::kOpenKfd;
  } else if (drm < 0) {
    report.step = HelperReport::kOpenRenderNode;
  } else if (ioctl(kfd, kIocAcquireVm, &vm) != 0) {
    report.step = HelperReport::kAcquireVm;
  } else if (ioctl(kfd, kIocAllocMemoryOfGpu, &mem) != 0) {
    report.step = HelperReport::kAllocVram;
  } else {
    for (int i = 0; i < kDupFds; ++i) {
      if (dup(drm) < 0) {
        report.step = HelperReport::kDupFd;
        break;
      }
    }
  }
  if (report.step != HelperReport::kReady) report.err = errno;
  report.drm_fd = drm;
  if (write(fd, &report, sizeof(report)) != sizeof(report) || report.step != HelperReport::kReady) {
    _exit(1);
  }
  for (;;) pause();
}

struct DrmMemoryKiB {
  uint64_t vram = 0;
  uint64_t gtt = 0;
  uint64_t cpu = 0;
  bool operator==(const DrmMemoryKiB& o) const {
    return vram == o.vram && gtt == o.gtt && cpu == o.cpu;
  }
};

// The DRM memory one fd's client reports, in KiB as fdinfo prints it.
bool ReadDrmMemory(pid_t pid, int fd, DrmMemoryKiB* out) {
  std::ifstream fdinfo("/proc/" + std::to_string(pid) + "/fdinfo/" + std::to_string(fd));
  int keys = 0;
  for (std::string line; std::getline(fdinfo, line);) {
    const char* l = line.c_str();
    keys += sscanf(l, "drm-memory-vram: %" SCNu64, &out->vram) == 1;
    keys += sscanf(l, "drm-memory-gtt: %" SCNu64, &out->gtt) == 1;
    keys += sscanf(l, "drm-memory-cpu: %" SCNu64, &out->cpu) == 1;
  }
  return keys == 3;
}

}  // namespace

TestProcessMemoryRead::TestProcessMemoryRead() : TestBase() {
  set_title("AMDSMI Process Memory Read Test");
  set_description(
      "This test verifies that amdsmi_get_gpu_process_list reports a process's "
      "VRAM, GTT and CPU memory in bytes, counting its DRM client once however "
      "many fds refer to it. When the process's fdinfo is in another PID "
      "namespace, it verifies the VRAM KFD reports for the process instead.");
}

TestProcessMemoryRead::~TestProcessMemoryRead(void) {}

void TestProcessMemoryRead::SetUp(void) {
  TestBase::SetUp();
  return;
}

void TestProcessMemoryRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestProcessMemoryRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestProcessMemoryRead::Close() { TestBase::Close(); }

void TestProcessMemoryRead::Run(void) {
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
  // An odd size, so another process is unlikely to hold exactly the same.
  const uint64_t vram = (4ULL << 20) + (std::random_device{}() % 255 + 1) *
                                           static_cast<uint64_t>(sysconf(_SC_PAGESIZE));

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
    RunHelper(kfd_gpu_id, render_node, vram, parent, fds[1]);
  }
  close(fds[1]);
  HelperStopper stopper{helper, helper};
  MemoryHelperReport report;
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
  const std::vector<pid_t> kfd_pids = FindHelperKfdPids(helper, kfd_gpu_id, vram);
  // A second match is another run of this test that drew the same size.
  if (kfd_pids.size() > 1) GTEST_SKIP() << "Another process holds the helper's exact VRAM";
  ASSERT_EQ(kfd_pids.size(), 1u) << "KFD does not report exactly the helper's VRAM";
  const pid_t kfd_pid = kfd_pids[0];
  stopper.kfd_pid = kfd_pid;
  DrmMemoryKiB expected;
  ASSERT_TRUE(ReadDrmMemory(helper, report.drm_fd, &expected)) << "No DRM memory in fdinfo";
  // Equal sizes prove nothing when fdinfo reports none.
  ASSERT_GT(expected.vram + expected.gtt + expected.cpu, 0u) << "fdinfo reports no DRM memory";

  // A list cached before the helper started is served for up to
  // AMDSMI_PROCESS_INFO_CACHE_MS, so query until it shows the helper.
  const char* cache_ms = std::getenv("AMDSMI_PROCESS_INFO_CACHE_MS");
  std::vector<amdsmi_proc_info_t> procs(512);
  // Looks `pid` up in GPU 0's process list.
  auto find = [&](pid_t pid, amdsmi_proc_info_t* entry) {
    uint32_t count = static_cast<uint32_t>(procs.size());
    amdsmi_status_t list_status = amdsmi_get_gpu_process_list(gpu, &count, procs.data());
    if (list_status == AMDSMI_STATUS_OUT_OF_RESOURCES) {
      procs.resize(count + 64);
      count = static_cast<uint32_t>(procs.size());
      list_status = amdsmi_get_gpu_process_list(gpu, &count, procs.data());
    }
    for (uint32_t i = 0; list_status == AMDSMI_STATUS_SUCCESS && i < count && i < procs.size();
         ++i) {
      if (procs[i].pid == static_cast<uint32_t>(pid)) {
        *entry = procs[i];
        return true;
      }
    }
    return false;
  };
  amdsmi_proc_info_t found{};
  bool listed = false;
  for (const auto give_up =
           Clock::now() + std::chrono::seconds(10) +
           std::chrono::milliseconds(cache_ms ? std::strtoul(cache_ms, nullptr, 10) : 0);
       !listed && Clock::now() < give_up;) {
    listed = find(kfd_pid, &found);
    if (!listed) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  // In a container rocm_smi may list this namespace's PIDs instead of KFD's.
  amdsmi_proc_info_t by_own_pid{};
  if (!listed && kfd_pid != helper && find(helper, &by_own_pid)) {
    GTEST_SKIP() << "The library lists the helper by this PID namespace's PID";
  }
  ASSERT_TRUE(listed) << "GPU 0 does not list the helper";
  DrmMemoryKiB now;
  ASSERT_TRUE(ReadDrmMemory(helper, report.drm_fd, &now));
  ASSERT_TRUE(now == expected) << "The helper's DRM memory changed during the query";

  // A KFD process reports, as mem, the VRAM KFD holds for it.
  EXPECT_EQ(found.mem, vram);
  if (kfd_pid == helper) {
    // fdinfo sizes are KiB, and the helper's 1 + kDupFds fds all name one client.
    EXPECT_EQ(found.memory_usage.vram_mem, expected.vram * 1024);
    EXPECT_EQ(found.memory_usage.gtt_mem, expected.gtt * 1024);
    EXPECT_EQ(found.memory_usage.cpu_mem, expected.cpu * 1024);
  } else {
    // Listed by its host PID, which this PID namespace's /proc does not have: the
    // library cannot read the helper's fdinfo and reports the VRAM KFD holds for it.
    EXPECT_EQ(found.memory_usage.vram_mem, vram);
    std::cout << "\t**fdinfo sizes not compared: the helper's fdinfo is in another PID "
                 "namespace"
              << std::endl;
  }
  IF_VERB(STANDARD) {
    std::cout << "\t**fdinfo KiB vram " << expected.vram << ", gtt " << expected.gtt << ", cpu "
              << expected.cpu << "; listed bytes vram " << found.memory_usage.vram_mem << ", gtt "
              << found.memory_usage.gtt_mem << ", cpu " << found.memory_usage.cpu_mem << ", mem "
              << found.mem << std::endl;
  }
}
