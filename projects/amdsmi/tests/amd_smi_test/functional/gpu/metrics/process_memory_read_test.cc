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
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
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
// Extra fds the helper opens on its one DRM client, as dup(), fork() or fd passing do.
constexpr int kDupFds = 3;

enum HelperStep { kReady, kOpenKfd, kOpenRenderNode, kAcquireVm, kAllocVram, kDupFd };

struct HelperReport {
  int step = -1;  // stays -1 without a complete report
  int err = 0;
  int drm_fd = -1;
};

// Runs in the forked helper, system calls only: allocates `vram` on the GPU
// through KFD, adds kDupFds fds for the same DRM client, reports, then waits to be killed.
[[noreturn]] void RunHelper(uint32_t kfd_gpu_id, const std::string& render_node, uint64_t vram,
                            pid_t parent, int fd) {
  if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) _exit(1);
  HelperReport report;
  report.step = kReady;
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
    report.step = kOpenKfd;
  } else if (drm < 0) {
    report.step = kOpenRenderNode;
  } else if (ioctl(kfd, kIocAcquireVm, &vm) != 0) {
    report.step = kAcquireVm;
  } else if (ioctl(kfd, kIocAllocMemoryOfGpu, &mem) != 0) {
    report.step = kAllocVram;
  } else {
    for (int i = 0; i < kDupFds; ++i) {
      if (dup(drm) < 0) {
        report.step = kDupFd;
        break;
      }
    }
  }
  if (report.step != kReady) report.err = errno;
  report.drm_fd = drm;
  if (write(fd, &report, sizeof(report)) != sizeof(report) || report.step != kReady) _exit(1);
  for (;;) pause();
}

uint64_t KfdVram(pid_t pid, uint32_t kfd_gpu_id) {
  std::ifstream file(kKfdProcRoot + std::to_string(pid) + "/vram_" + std::to_string(kfd_gpu_id));
  uint64_t vram = 0;
  return (file >> vram) ? vram : 0;
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

TestProcessMemoryRead::TestProcessMemoryRead() : TestBase() {
  set_title("AMDSMI Process Memory Read Test");
  set_description(
      "This test verifies that amdsmi_get_gpu_process_list reports a process's "
      "VRAM, GTT and CPU memory in bytes, counting its DRM client once however "
      "many fds refer to it.");
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
  HelperStopper stopper{helper};
  HelperReport report;
  pollfd ready{fds[0], POLLIN, 0};
  if (poll(&ready, 1, 30000) == 1) {
    if (read(fds[0], &report, sizeof(report)) != sizeof(report)) report.step = -1;
  }
  close(fds[0]);
  if (report.step == -1) FAIL() << "The helper exited or did not report within 30 s";
  if (report.step != kReady) {
    // No access to the device files, or not enough memory: this machine cannot run the test.
    const bool no_access = (report.step == kOpenKfd || report.step == kOpenRenderNode) &&
                           (report.err == EACCES || report.err == EPERM || report.err == ENOENT ||
                            report.err == ENODEV || report.err == ENXIO);
    if (no_access || report.err == ENOMEM) {
      GTEST_SKIP() << "Cannot use the GPU through KFD: step " << report.step << ", errno "
                   << report.err;
    }
    FAIL() << "Helper setup failed: step " << report.step << ", errno " << report.err;
  }
  // KFD names processes by host PID. Under another PID namespace the library
  // cannot open the helper's fdinfo, so there is nothing to compare.
  if (KfdVram(helper, kfd_gpu_id) != vram) {
    GTEST_SKIP() << "KFD lists the helper by another PID; its fdinfo is not readable here";
  }
  DrmMemoryKiB expected;
  ASSERT_TRUE(ReadDrmMemory(helper, report.drm_fd, &expected)) << "No DRM memory in fdinfo";

  // A list cached before the helper started is served for up to
  // AMDSMI_PROCESS_INFO_CACHE_MS, so query until it shows the helper.
  std::vector<amdsmi_proc_info_t> procs(512);
  amdsmi_proc_info_t found{};
  bool listed = false;
  for (const auto give_up = Clock::now() + std::chrono::seconds(10);
       !listed && Clock::now() < give_up;) {
    uint32_t count = static_cast<uint32_t>(procs.size());
    amdsmi_status_t list_status = amdsmi_get_gpu_process_list(gpu, &count, procs.data());
    if (list_status == AMDSMI_STATUS_OUT_OF_RESOURCES) {
      procs.resize(count + 64);
      count = static_cast<uint32_t>(procs.size());
      list_status = amdsmi_get_gpu_process_list(gpu, &count, procs.data());
    }
    if (list_status == AMDSMI_STATUS_SUCCESS) {
      for (uint32_t i = 0; i < count && i < procs.size(); ++i) {
        if (procs[i].pid == static_cast<uint32_t>(helper)) {
          found = procs[i];
          listed = true;
        }
      }
    }
    if (!listed) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_TRUE(listed) << "GPU 0 does not list the helper";
  DrmMemoryKiB now;
  ASSERT_TRUE(ReadDrmMemory(helper, report.drm_fd, &now));
  ASSERT_TRUE(now == expected) << "The helper's DRM memory changed during the query";

  // fdinfo sizes are KiB, and the helper's 1 + kDupFds fds all name one client.
  EXPECT_EQ(found.memory_usage.vram_mem, expected.vram * 1024);
  EXPECT_EQ(found.memory_usage.gtt_mem, expected.gtt * 1024);
  EXPECT_EQ(found.memory_usage.cpu_mem, expected.cpu * 1024);
  // A KFD process reports, as mem, the VRAM KFD holds for it.
  EXPECT_EQ(found.mem, vram);
  IF_VERB(STANDARD) {
    std::cout << "\t**fdinfo KiB vram " << expected.vram << ", gtt " << expected.gtt << ", cpu "
              << expected.cpu << "; listed bytes vram " << found.memory_usage.vram_mem << ", gtt "
              << found.memory_usage.gtt_mem << ", cpu " << found.memory_usage.cpu_mem << ", mem "
              << found.mem << std::endl;
  }
}
