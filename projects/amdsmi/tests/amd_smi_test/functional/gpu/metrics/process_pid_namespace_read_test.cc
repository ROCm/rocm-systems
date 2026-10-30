// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "process_pid_namespace_read.h"

#include <fcntl.h>
#include <grp.h>
#include <gtest/gtest.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "kfd_helper_test_util.h"
#include "rocm_smi/kfd_ioctl.h"
#include "test_common.h"

namespace {

using amdsmi_test::DrmVramBytes;
using amdsmi_test::FindHelperKfdPids;
using amdsmi_test::HelperReport;
using amdsmi_test::HelperStopper;
using amdsmi_test::InInitialPidNamespace;
using amdsmi_test::IsEnvironmentFailure;
using amdsmi_test::kIocAcquireVm;
using amdsmi_test::kIocAllocMemoryOfGpu;
using amdsmi_test::kNotAvailable32;
using amdsmi_test::kNotAvailable64;
using Clock = std::chrono::steady_clock;

// Room for the processes and GPUs that the queries return. kSpareProcs more
// entries cover processes that start between a count and a read, or a list
// longer than kMaxProcs.
constexpr uint32_t kMaxProcs = 512;
constexpr uint32_t kSpareProcs = 64;
constexpr uint32_t kMaxGpus = 64;
// How long a forked process may take to set up, or to finish its queries.
constexpr int kForkedTimeoutMs = 30000;
// How long a process list cached before the helper started may still be
// served (beyond AMDSMI_PROCESS_INFO_CACHE_MS), and how often to query it.
constexpr auto kListWait = std::chrono::seconds(10);
constexpr auto kPollInterval = std::chrono::milliseconds(10);
// The highest fd a forked child closes, whatever RLIMIT_NOFILE allows.
constexpr int kMaxInheritedFds = 65536;
// The user and group "nobody", for a caller that may not read root's processes.
constexpr uid_t kNobodyUid = 65534;
constexpr gid_t kNobodyGid = 65534;

// Closes the fds a forked child inherited, except `keep`, so that it holds only
// the files it opens itself. System calls only.
void CloseInheritedFds(int keep) {
  rlimit limit{};
  const int max_fd = getrlimit(RLIMIT_NOFILE, &limit) == 0 && limit.rlim_cur < kMaxInheritedFds
                         ? static_cast<int>(limit.rlim_cur)
                         : kMaxInheritedFds;
  for (int fd = 3; fd < max_fd; ++fd) {
    if (fd != keep) close(fd);
  }
}

// Kills a forked child of this test and waits for it to exit.
void KillAndReap(pid_t pid) {
  kill(pid, SIGKILL);
  waitpid(pid, nullptr, 0);
}

// Runs in the forked helper, system calls only: holds `vram` on one GPU through
// KFD, mapped there so that the DRM client of that GPU's render node reports
// it too, and no other GPU file. Writes {step, errno} to `fd`, then waits to be
// killed.
[[noreturn]] void RunHelper(uint32_t kfd_gpu_id, const char* render_node, uint64_t vram,
                            pid_t parent, int fd) {
  if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) _exit(1);
  CloseInheritedFds(fd);
  HelperReport report{};
  report.step = HelperReport::kReady;
  const int kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  const int drm = kfd < 0 ? -1 : open(render_node, O_RDWR | O_CLOEXEC);
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
  uint32_t map_gpu_id = kfd_gpu_id;
  kfd_ioctl_map_memory_to_gpu_args map{};
  map.device_ids_array_ptr = reinterpret_cast<uint64_t>(&map_gpu_id);
  map.n_devices = 1;
  if (kfd < 0) {
    report.step = HelperReport::kOpenKfd;
  } else if (drm < 0) {
    report.step = HelperReport::kOpenRenderNode;
  } else if (ioctl(kfd, kIocAcquireVm, &vm) != 0) {
    report.step = HelperReport::kAcquireVm;
  } else if (ioctl(kfd, kIocAllocMemoryOfGpu, &mem) != 0) {
    report.step = HelperReport::kAllocVram;
  } else {
    map.handle = mem.handle;
    if (ioctl(kfd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map) != 0) report.step = HelperReport::kMapVram;
  }
  if (report.step != HelperReport::kReady) report.err = errno;
  if (write(fd, &report, sizeof(report)) != sizeof(report) || report.step != HelperReport::kReady) {
    _exit(1);
  }
  for (;;) pause();
}

// Runs in the forked bystander, system calls only: a process of this PID
// namespace that does not use a GPU. Writes one byte to `fd` once it holds no
// inherited file, then waits to be killed.
[[noreturn]] void RunBystander(pid_t parent, int fd) {
  if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) _exit(1);
  CloseInheritedFds(fd);
  const char ready = 1;
  if (write(fd, &ready, 1) != 1) _exit(1);
  for (;;) pause();
}

// Forks a bystander and waits until it reports. Returns its PID, or -1.
pid_t StartBystander(pid_t parent) {
  int fds[2];
  if (pipe(fds) != 0) return -1;
  const pid_t pid = fork();
  if (pid == 0) RunBystander(parent, fds[1]);
  close(fds[1]);
  char ready = 0;
  pollfd ready_poll{fds[0], POLLIN, 0};
  const bool up =
      pid > 0 && poll(&ready_poll, 1, kForkedTimeoutMs) == 1 && read(fds[0], &ready, 1) == 1;
  close(fds[0]);
  if (pid > 0 && !up) KillAndReap(pid);
  return up ? pid : -1;
}

// Starts a bystander with the PID `pid` of this namespace by setting
// ns_last_pid, which needs CAP_SYS_ADMIN. Returns its PID, or -1 if it could
// not get `pid`.
pid_t StartBystanderAt(pid_t pid, pid_t parent) {
  const int fd = open("/proc/sys/kernel/ns_last_pid", O_WRONLY | O_CLOEXEC);
  if (fd < 0) return -1;
  const std::string last = std::to_string(pid - 1);
  const bool set = write(fd, last.data(), last.size()) == static_cast<ssize_t>(last.size());
  close(fd);
  const pid_t bystander = set ? StartBystander(parent) : -1;
  if (bystander > 0 && bystander != pid) {
    KillAndReap(bystander);
    return -1;
  }
  return bystander;
}

// Whether a process of this PID namespace has the PID `pid`.
bool IsLocalPid(pid_t pid) { return kill(pid, 0) == 0 || errno != ESRCH; }

// What RunUnprivileged() got from the library.
struct UnprivilegedResult {
  bool dropped_root = false;
  amdsmi_status_t by_pid = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t gpus = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t list = AMDSMI_STATUS_SUCCESS;
  bool helper_listed = false;
};

// Runs in a forked child of root: becomes the user and group nobody, who may
// not read the open files of root's process `helper`, queries the library, set
// up before the fork, about it, and writes what it got to `fd`.
[[noreturn]] void RunUnprivileged(pid_t helper, pid_t parent, int fd) {
  UnprivilegedResult result{};
  result.dropped_root =
      setgroups(0, nullptr) == 0 && setgid(kNobodyGid) == 0 && setuid(kNobodyUid) == 0;
  // Set after setuid(), which clears it.
  if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) _exit(1);
  if (result.dropped_root) {
    amdsmi_process_info_t info{};
    result.by_pid = amdsmi_get_gpu_compute_process_info_by_pid(helper, &info);
    uint32_t gpu_indices[kMaxGpus] = {};
    uint32_t gpu_count = kMaxGpus;
    result.gpus = amdsmi_get_gpu_compute_process_gpus(helper, gpu_indices, &gpu_count);
    amdsmi_process_info_t procs[kMaxProcs] = {};
    uint32_t num_items = kMaxProcs;
    result.list = amdsmi_get_gpu_compute_process_info(procs, &num_items);
    for (uint32_t i = 0; result.list == AMDSMI_STATUS_SUCCESS && i < num_items; ++i) {
      if (procs[i].process_id == static_cast<uint32_t>(helper)) result.helper_listed = true;
    }
  }
  _exit(write(fd, &result, sizeof(result)) == sizeof(result) ? 0 : 1);
}

}  // namespace

TestProcessPidNamespaceRead::TestProcessPidNamespaceRead() : TestBase() {
  set_title("AMDSMI Process PID Namespace Read Test");
  set_description(
      "This test verifies that in a PID namespace other than the initial one, as "
      "in a container that does not share the host's, the process APIs report a "
      "GPU process of that namespace by its PID there, with its own name, GPU and "
      "memory and its KFD statistics not available; that they report nothing for "
      "the PID KFD knows it by or for a process of the namespace that does not use "
      "a GPU; and that a caller who may not read its open files does not see it.");
}

TestProcessPidNamespaceRead::~TestProcessPidNamespaceRead(void) {}

void TestProcessPidNamespaceRead::SetUp(void) {
  TestBase::SetUp();
  return;
}

void TestProcessPidNamespaceRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestProcessPidNamespaceRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestProcessPidNamespaceRead::Close() { TestBase::Close(); }

void TestProcessPidNamespaceRead::Run(void) {
  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }
  if (InInitialPidNamespace()) {
    GTEST_SKIP() << "Needs a PID namespace other than the initial one, as in a container that "
                    "does not share the host's";
  }
  const size_t num_gpus = num_monitor_devs();
  if (num_gpus == 0) GTEST_SKIP() << "No GPUs found";

  amdsmi_processor_handle gpu = processor_handles_[0];
  amdsmi_kfd_info_t kfd_info{};
  amdsmi_enumeration_info_t enum_info{};
  amdsmi_status_t status = amdsmi_get_gpu_kfd_info(gpu, &kfd_info);
  if (status == AMDSMI_STATUS_SUCCESS) status = amdsmi_get_gpu_enumeration_info(gpu, &enum_info);
  if (status == AMDSMI_STATUS_NOT_SUPPORTED) GTEST_SKIP() << "GPU 0 has no KFD info";
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  if (kfd_info.kfd_id == kNotAvailable64 || enum_info.drm_render == 0 ||
      enum_info.drm_render == kNotAvailable32) {
    GTEST_SKIP() << "GPU 0 has no known KFD node or render node";
  }
  const uint32_t kfd_gpu_id = static_cast<uint32_t>(kfd_info.kfd_id);
  const std::string render_node = "/dev/dri/renderD" + std::to_string(enum_info.drm_render);
  // A random page count tells concurrent runs of this test apart. The 5 MiB
  // base tells its helper from the other process tests' with 4 KiB pages.
  const uint64_t vram = (5ULL << 20) + (std::random_device{}() % 255 + 1) *
                                           static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
  const pid_t parent = getpid();

  int fds[2];
  ASSERT_EQ(pipe(fds), 0);
  const pid_t helper = fork();
  if (helper < 0) {
    close(fds[0]);
    close(fds[1]);
    FAIL() << "fork failed, errno " << errno;
  }
  if (helper == 0) RunHelper(kfd_gpu_id, render_node.c_str(), vram, parent, fds[1]);
  close(fds[1]);
  HelperStopper helper_stopper{helper, 0};
  HelperReport report{};  // step -1 until the helper reports
  pollfd ready_poll{fds[0], POLLIN, 0};
  if (poll(&ready_poll, 1, kForkedTimeoutMs) == 1) {
    if (read(fds[0], &report, sizeof(report)) != sizeof(report)) report.step = -1;
  }
  close(fds[0]);
  if (report.step == -1) {
    FAIL() << "The helper exited or did not report within " << kForkedTimeoutMs << " ms";
  }
  if (report.step != HelperReport::kReady) {
    if (IsEnvironmentFailure(report)) {
      GTEST_SKIP() << "Cannot use the GPU through KFD: step " << report.step << ", errno "
                   << report.err;
    }
    FAIL() << "Helper setup failed: step " << report.step << ", errno " << report.err;
  }
  const uint64_t drm_vram = DrmVramBytes(helper, render_node);
  ASSERT_GE(drm_vram, vram) << "GPU 0's fdinfo misses the helper's VRAM";
  // KFD knows the helper by its PID in the initial PID namespace. Two matches
  // are another run of this test that drew the same size; then, as without a
  // match, the checks of that PID are left out.
  const std::vector<pid_t> kfd_pids = FindHelperKfdPids(helper, kfd_gpu_id, vram);
  const pid_t kfd_pid = kfd_pids.size() == 1 ? kfd_pids[0] : 0;
  helper_stopper.kfd_pid = kfd_pid;

  // A process of this namespace that does not use a GPU. When it can, it takes
  // the PID KFD knows the helper by, under which a library that read KFD's
  // process directories would report the helper's data.
  pid_t bystander = -1;
  if (kfd_pid > 0 && !IsLocalPid(kfd_pid)) bystander = StartBystanderAt(kfd_pid, parent);
  if (bystander < 0) bystander = StartBystander(parent);
  ASSERT_GT(bystander, 0) << "Could not start the bystander";
  HelperStopper bystander_stopper{bystander, 0};
  // Unless another process of this namespace has KFD's PID for the helper, no
  // API may report anything for it.
  const bool check_kfd_pid = kfd_pid > 0 && (bystander == kfd_pid || !IsLocalPid(kfd_pid));
  if (!check_kfd_pid) {
    std::cout << "\t**KFD's PID for the helper is " << (kfd_pid > 0 ? "a process here" : "unknown")
              << ": the checks that nothing is reported for it are left out" << std::endl;
  }
  IF_VERB(STANDARD) {
    std::cout << "\t**Helper PID " << helper << ", KFD's PID for it " << kfd_pid
              << ", bystander PID " << bystander << std::endl;
  }

  std::vector<amdsmi_proc_info_t> procs(kMaxProcs);
  // Looks `pid` up in one GPU's process list: sets *listed, and *entry if it
  // is listed, and returns the list's status.
  auto find = [&](amdsmi_processor_handle handle, pid_t pid, bool* listed,
                  amdsmi_proc_info_t* entry) {
    *listed = false;
    uint32_t count = static_cast<uint32_t>(procs.size());
    amdsmi_status_t list_status = amdsmi_get_gpu_process_list(handle, &count, procs.data());
    if (list_status == AMDSMI_STATUS_OUT_OF_RESOURCES) {
      procs.resize(count + kSpareProcs);
      count = static_cast<uint32_t>(procs.size());
      list_status = amdsmi_get_gpu_process_list(handle, &count, procs.data());
    }
    for (uint32_t i = 0; list_status == AMDSMI_STATUS_SUCCESS && i < count && i < procs.size();
         ++i) {
      if (procs[i].pid == static_cast<uint32_t>(pid)) {
        *entry = procs[i];
        *listed = true;
        break;
      }
    }
    return list_status;
  };
  // A list cached before the helper started is served for up to
  // AMDSMI_PROCESS_INFO_CACHE_MS, so query until it shows the helper.
  const char* cache_ms = std::getenv("AMDSMI_PROCESS_INFO_CACHE_MS");
  amdsmi_proc_info_t found{};
  bool listed = false;
  amdsmi_status_t list_status = AMDSMI_STATUS_SUCCESS;
  for (const auto give_up =
           Clock::now() + kListWait +
           std::chrono::milliseconds(cache_ms ? std::strtoul(cache_ms, nullptr, 10) : 0);
       !listed && Clock::now() < give_up;) {
    list_status = find(gpu, helper, &listed, &found);
    if (!listed) std::this_thread::sleep_for(kPollInterval);
  }
  ASSERT_EQ(list_status, AMDSMI_STATUS_SUCCESS) << "GPU 0's process list failed";
  ASSERT_TRUE(listed) << "GPU 0 does not list the helper by its PID in this namespace";

  // The name is the helper's own: a fork of this test, it runs the same file.
  char exe[PATH_MAX] = {};
  ASSERT_GT(readlink("/proc/self/exe", exe, sizeof(exe) - 1), 0);
  EXPECT_EQ(std::string(found.name, strnlen(found.name, sizeof(found.name))),
            std::string(exe).substr(0, sizeof(found.name) - 1));
  // Its VRAM is what its own DRM client reports: the VRAM it mapped, with the
  // GPU's page tables for it.
  EXPECT_EQ(found.memory_usage.vram_mem, drm_vram);
  EXPECT_EQ(found.cu_occupancy, kNotAvailable32);
  EXPECT_EQ(found.evicted_time, kNotAvailable32);
  EXPECT_EQ(found.sdma_usage, kNotAvailable64);
  amdsmi_proc_info_t other{};
  bool other_listed = false;
  EXPECT_EQ(find(gpu, bystander, &other_listed, &other), AMDSMI_STATUS_SUCCESS);
  EXPECT_FALSE(other_listed) << "GPU 0 lists the bystander";
  if (check_kfd_pid) {
    EXPECT_EQ(find(gpu, kfd_pid, &other_listed, &other), AMDSMI_STATUS_SUCCESS);
    EXPECT_FALSE(other_listed) << "GPU 0 lists KFD's PID for the helper";
  }
  // The helper uses GPU 0 only: it holds no other GPU's render node.
  for (size_t i = 1; i < num_gpus; ++i) {
    EXPECT_EQ(find(processor_handles_[i], helper, &other_listed, &other), AMDSMI_STATUS_SUCCESS);
    EXPECT_FALSE(other_listed) << "GPU " << i << " lists the helper";
  }

  // The KFD processes of this namespace.
  uint32_t num_items = 0;
  ASSERT_EQ(amdsmi_get_gpu_compute_process_info(nullptr, &num_items), AMDSMI_STATUS_SUCCESS);
  std::vector<amdsmi_process_info_t> kfd_procs(num_items + kSpareProcs);
  num_items = static_cast<uint32_t>(kfd_procs.size());
  ASSERT_EQ(amdsmi_get_gpu_compute_process_info(kfd_procs.data(), &num_items),
            AMDSMI_STATUS_SUCCESS);
  bool helper_listed = false;
  for (uint32_t i = 0; i < num_items; ++i) {
    const amdsmi_process_info_t& proc = kfd_procs[i];
    EXPECT_NE(proc.process_id, static_cast<uint32_t>(bystander)) << "The bystander is listed";
    if (check_kfd_pid) {
      EXPECT_NE(proc.process_id, static_cast<uint32_t>(kfd_pid)) << "KFD's PID is listed";
    }
    if (proc.process_id != static_cast<uint32_t>(helper)) continue;
    helper_listed = true;
    EXPECT_EQ(proc.vram_usage, kNotAvailable64);
    EXPECT_EQ(proc.sdma_usage, kNotAvailable64);
    EXPECT_EQ(proc.cu_occupancy, kNotAvailable32);
    EXPECT_EQ(proc.evicted_time, kNotAvailable32);
  }
  EXPECT_TRUE(helper_listed) << "amdsmi_get_gpu_compute_process_info does not list the helper";

  amdsmi_process_info_t info{};
  ASSERT_EQ(amdsmi_get_gpu_compute_process_info_by_pid(helper, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.process_id, static_cast<uint32_t>(helper));
  EXPECT_EQ(info.vram_usage, kNotAvailable64);
  EXPECT_EQ(info.sdma_usage, kNotAvailable64);
  EXPECT_EQ(info.cu_occupancy, kNotAvailable32);
  EXPECT_EQ(info.evicted_time, kNotAvailable32);
  EXPECT_EQ(amdsmi_get_gpu_compute_process_info_by_pid(bystander, &info), AMDSMI_STATUS_NOT_FOUND);
  if (check_kfd_pid) {
    EXPECT_EQ(amdsmi_get_gpu_compute_process_info_by_pid(kfd_pid, &info), AMDSMI_STATUS_NOT_FOUND);
  }

  uint32_t gpu_indices[kMaxGpus] = {};
  uint32_t gpu_count = kMaxGpus;
  ASSERT_EQ(amdsmi_get_gpu_compute_process_gpus(helper, gpu_indices, &gpu_count),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(gpu_count, 1u) << "The helper uses GPU 0 only";
  gpu_count = kMaxGpus;
  EXPECT_EQ(amdsmi_get_gpu_compute_process_gpus(bystander, gpu_indices, &gpu_count),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(gpu_count, 0u) << "The bystander uses no GPU";
  if (check_kfd_pid) {
    gpu_count = kMaxGpus;
    EXPECT_EQ(amdsmi_get_gpu_compute_process_gpus(kfd_pid, gpu_indices, &gpu_count),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(gpu_count, 0u) << "KFD's PID for the helper names, here, a process without a GPU";
  }

  // A caller that may not read the helper's open files, as a user other than
  // root, gets NO_PERM for it and a KFD process list without it. Only root can
  // start such a caller. The library starts no thread, so the child may call it
  // after fork().
  if (geteuid() != 0) {
    std::cout << "\t**Not root: the checks of a caller without access are left out" << std::endl;
    return;
  }
  int result_fds[2];
  ASSERT_EQ(pipe(result_fds), 0);
  const pid_t caller = fork();
  if (caller < 0) {
    close(result_fds[0]);
    close(result_fds[1]);
    FAIL() << "fork failed, errno " << errno;
  }
  if (caller == 0) RunUnprivileged(helper, parent, result_fds[1]);
  close(result_fds[1]);
  UnprivilegedResult result{};
  pollfd result_poll{result_fds[0], POLLIN, 0};
  const bool reported = poll(&result_poll, 1, kForkedTimeoutMs) == 1 &&
                        read(result_fds[0], &result, sizeof(result)) == sizeof(result);
  close(result_fds[0]);
  KillAndReap(caller);
  ASSERT_TRUE(reported) << "The caller without access exited or did not report within "
                        << kForkedTimeoutMs << " ms";
  if (!result.dropped_root) {
    std::cout << "\t**Cannot become user nobody: the checks of a caller without access are "
                 "left out"
              << std::endl;
    return;
  }
  EXPECT_EQ(result.by_pid, AMDSMI_STATUS_NO_PERM)
      << "amdsmi_get_gpu_compute_process_info_by_pid for the helper";
  EXPECT_EQ(result.gpus, AMDSMI_STATUS_NO_PERM)
      << "amdsmi_get_gpu_compute_process_gpus for the helper";
  EXPECT_EQ(result.list, AMDSMI_STATUS_SUCCESS) << "amdsmi_get_gpu_compute_process_info";
  EXPECT_FALSE(result.helper_listed) << "amdsmi_get_gpu_compute_process_info lists the helper";
}
