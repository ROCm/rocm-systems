// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Reads processes from a mock /proc and a mock KFD process root, so the
// fdinfo-based numbers are checked without a GPU or another process. In CI
// containers without the host PID namespace, a real process's fdinfo cannot be
// read, so the functional tests skip that comparison there.

#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_kfd_process_list.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "amd_smi/impl/fdinfo.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_kfd.h"
#include "rocm_smi/rocm_smi_main.h"
#include "test_fixture_utils.h"

namespace {

using amd::smi::AMDSmiDrm;
using amd::smi::AMDSmiGPUDevice;
using amd::smi::testing::ScopedOverride;
using amd::smi::testing::ScopedProcessorRegistration;

// An address no real GPU is likely to have, and the same address as fdinfo
// prints it.
constexpr char kBdfText[] = "0000:fe:1f.7";
constexpr char kOtherBdfText[] = "0000:c3:00.0";
constexpr uint64_t kKfdGpuId = 41651;
constexpr uint64_t kOtherKfdGpuId = 1234;

amdsmi_bdf_t TestBdf() {
  amdsmi_bdf_t bdf{};
  bdf.bus_number = 0xfe;
  bdf.device_number = 0x1f;
  bdf.function_number = 7;
  return bdf;
}

amdsmi_bdf_t OtherTestBdf() {
  amdsmi_bdf_t bdf{};
  bdf.bus_number = 0xc3;
  return bdf;
}

// Points the KFD process root at a mock tree for one test.
class ScopedKfdProcRoot {
 public:
  explicit ScopedKfdProcRoot(const std::string& path) : path_(path) {
    amd::smi::SetKFDProcPathRootForTesting(path_.c_str());
  }
  ~ScopedKfdProcRoot() { amd::smi::SetKFDProcPathRootForTesting(nullptr); }

  ScopedKfdProcRoot(const ScopedKfdProcRoot&) = delete;
  ScopedKfdProcRoot& operator=(const ScopedKfdProcRoot&) = delete;

 private:
  std::string path_;
};

// Writes /proc/<pid> for a process with one amdgpu DRM client on the GPU at
// pdev: 1 MiB of VRAM, 2 MiB of GTT, 3 KiB of CPU memory.
void AddProcess(FakeSysfsTree* proc, uint32_t pid, const std::string& pdev) {
  const std::string dir = std::to_string(pid);
  proc->WriteFile(dir + "/fdinfo/3",
                  "pos:\t0\nflags:\t02100002\ndrm-driver:\tamdgpu\ndrm-client-id:\t7\n"
                  "drm-pdev:\t" +
                      pdev +
                      "\ndrm-memory-vram:\t1024 KiB\ndrm-memory-gtt:\t2 MiB\n"
                      "drm-memory-cpu:\t3 KiB\ndrm-engine-gfx:\t5000 ns\ndrm-engine-enc:\t60 ns\n");
  proc->WriteFile(dir + "/cgroup", "0::/user.slice\n");
  std::filesystem::create_symlink("/usr/bin/fake_gpu_app", proc->root() + "/" + dir + "/exe");
}

constexpr uint64_t kVram = 1024 * 1024;
constexpr uint64_t kGtt = 2 * 1024 * 1024;
constexpr uint64_t kCpu = 3 * 1024;

// Writes the per-GPU files that KFD creates in a process's directory for every
// GPU when the process opens /dev/kfd, used or not: here for this GPU and
// another one.
void AddKfdFilesForEveryGpu(FakeSysfsTree* kfd, const std::string& pid) {
  for (const uint64_t gpu_id : {kKfdGpuId, kOtherKfdGpuId}) {
    const std::string id = std::to_string(gpu_id);
    kfd->WriteFile(pid + "/vram_" + id, "0\n");
    kfd->WriteFile(pid + "/sdma_" + id, "0\n");
    kfd->WriteFile(pid + "/stats_" + id + "/cu_occupancy", "0\n");
    kfd->WriteFile(pid + "/stats_" + id + "/evicted_ms", "0\n");
  }
}

// A GPU for the process lists: the rocm_smi device at sysfs/<card>, which has
// the address `bdf` and, as discovery sets it on a real GPU, the KFD id
// `kfd_gpu_id`; by default this test GPU's.
class TestGpu {
 public:
  explicit TestGpu(const std::string& card = "card9", const amdsmi_bdf_t& bdf = TestBdf(),
                   uint64_t kfd_gpu_id = kKfdGpuId)
      : rocm_device_(sysfs_.root() + "/" + card),
        gpu_device_(rocm_device_.index(), card, bdf, drm_),
        registration_(&gpu_device_) {
    amd::smi::RocmSMI::getInstance().devices()[rocm_device_.index()]->set_kfd_gpu_id(kfd_gpu_id);
  }
  uint32_t index() const { return rocm_device_.index(); }
  amdsmi_processor_handle handle() const { return registration_.handle(); }

 private:
  FakeSysfsTree sysfs_;
  ScopedRocmSmiDevice rocm_device_;
  AMDSmiDrm drm_;
  AMDSmiGPUDevice gpu_device_;
  ScopedProcessorRegistration registration_;
};

class MockProcessRoots {
 public:
  MockProcessRoots()
      : proc_root_(amd::smi::test_hooks::ProcRoot, amd::smi::test_hooks::SetProcRoot, proc_.root()),
        kfd_root_(kfd_.root()) {}

  FakeSysfsTree& proc() { return proc_; }
  FakeSysfsTree& kfd() { return kfd_; }

 private:
  FakeSysfsTree proc_;
  FakeSysfsTree kfd_;
  ScopedOverride<std::string> proc_root_;
  ScopedKfdProcRoot kfd_root_;
};

TEST(GpuUnit, ProcessInfoSumsTheFdinfoOfAMockProcess) {
  MockProcessRoots roots;
  constexpr uint32_t kPid = 4242;
  AddProcess(&roots.proc(), kPid, kBdfText);
  // A VRAM size unlike fdinfo's, so the test shows which one is reported.
  roots.kfd().WriteFile("4242/vram_" + std::to_string(kKfdGpuId), "7\n");

  amdsmi_proc_info_t info{};
  ASSERT_EQ(gpuvsmi_get_pid_info(TestBdf(), kPid, info, kKfdGpuId), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.pid, kPid);
  EXPECT_STREQ(info.name, "/usr/bin/fake_gpu_app");
  EXPECT_EQ(info.memory_usage.vram_mem, kVram);
  EXPECT_EQ(info.memory_usage.gtt_mem, kGtt);
  EXPECT_EQ(info.memory_usage.cpu_mem, kCpu);
  EXPECT_EQ(info.mem, kVram + kGtt + kCpu);
  EXPECT_EQ(info.engine_usage.gfx, 5000U);
  EXPECT_EQ(info.engine_usage.enc, 60U);
}

TEST(GpuUnit, ProcessInfoFindsTheGpuInFdinfoWhenKfdDoesNotListIt) {
  MockProcessRoots roots;
  constexpr uint32_t kPid = 4243;
  AddProcess(&roots.proc(), kPid, kBdfText);
  roots.kfd().WriteFile("4243/vram_" + std::to_string(kOtherKfdGpuId), "4096\n");

  amdsmi_proc_info_t info{};
  ASSERT_EQ(gpuvsmi_get_pid_info(TestBdf(), kPid, info, kKfdGpuId), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, kVram);
}

TEST(GpuUnit, ProcessInfoOfAProcessOnAnotherGpuIsNotFound) {
  MockProcessRoots roots;
  constexpr uint32_t kPid = 4244;
  AddProcess(&roots.proc(), kPid, kOtherBdfText);
  roots.kfd().WriteFile("4244/vram_" + std::to_string(kOtherKfdGpuId), "4096\n");

  amdsmi_proc_info_t info{};
  EXPECT_EQ(gpuvsmi_get_pid_info(TestBdf(), kPid, info, kKfdGpuId), AMDSMI_STATUS_NOT_FOUND);
}

TEST(GpuUnit, ProcessListPlacesAProcessWhoseKfdGpusCannotBeRead) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // No process's KFD queue names a readable GPU, and each has the KFD files of
  // both GPUs. The first one's fdinfo shows this GPU, the second one's another
  // GPU; the third has no /proc entry (it exited), and the fourth shows another
  // GPU and has an fdinfo entry whose fd was closed during the read.
  constexpr uint32_t kOnThisGpu = 4250;
  constexpr uint32_t kOnAnotherGpu = 4251;
  constexpr uint32_t kWithoutProcEntry = 4256;
  constexpr uint32_t kWithClosedFd = 4257;
  for (const uint32_t pid : {kOnThisGpu, kOnAnotherGpu, kWithoutProcEntry, kWithClosedFd}) {
    roots.kfd().WriteFile(std::to_string(pid) + "/queues/0/gpuid", "not-a-gpu-id\n");
    AddKfdFilesForEveryGpu(&roots.kfd(), std::to_string(pid));
  }
  AddProcess(&roots.proc(), kOnThisGpu, kBdfText);
  AddProcess(&roots.proc(), kOnAnotherGpu, kOtherBdfText);
  AddProcess(&roots.proc(), kWithClosedFd, kOtherBdfText);
  const std::string closed_fd = roots.proc().root() + "/" + std::to_string(kWithClosedFd);
  std::filesystem::create_symlink(closed_fd + "/fdinfo/gone", closed_fd + "/fdinfo/8");

  TestGpu gpu;
  WaitOutProcessListCache();
  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  ASSERT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_processes, 1U);
  EXPECT_EQ(list[0].pid, kOnThisGpu);
  EXPECT_EQ(list[0].memory_usage.vram_mem, kVram);
  EXPECT_EQ(list[0].mem, kVram + kGtt + kCpu);
  // Its KFD statistics are not available.
  EXPECT_EQ(list[0].cu_occupancy, std::numeric_limits<decltype(list[0].cu_occupancy)>::max());
  EXPECT_EQ(list[0].evicted_time, std::numeric_limits<decltype(list[0].evicted_time)>::max());
  EXPECT_EQ(list[0].sdma_usage, std::numeric_limits<decltype(list[0].sdma_usage)>::max());
}

TEST(GpuUnit, ProcessListFailsForARunningProcessThatCannotBePlaced) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // KFD cannot say which GPU this running process uses, and its fdinfo cannot be
  // read either: a file stands where the directory should be.
  roots.kfd().WriteFile("4252/queues/0/gpuid", "not-a-gpu-id\n");
  AddKfdFilesForEveryGpu(&roots.kfd(), "4252");
  roots.proc().WriteFile("4252/fdinfo", "not a directory\n");

  TestGpu gpu;
  WaitOutProcessListCache();
  // The error from reading its GPUs (a queue GPU id that is not a number), not
  // a list without it.
  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  EXPECT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list),
            AMDSMI_STATUS_UNEXPECTED_DATA);
  // With no GPU's list read, the by-PID list returns it too.
  amdsmi_processor_handle handle = gpu.handle();
  uint32_t num_pids = 4;
  amdsmi_proc_info_by_pid_t by_pid[4] = {};
  EXPECT_EQ(amdsmi_get_gpu_process_list_by_pid(&handle, 1, by_pid, &num_pids),
            AMDSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, ProcessListFailsWhenAnFdinfoFileCannotBeRead) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // KFD cannot say which GPU this running process uses. Its readable fdinfo
  // file shows another GPU, but one that cannot be read (a symlink loop) might
  // have shown this one.
  roots.kfd().WriteFile("4258/queues/0/gpuid", "not-a-gpu-id\n");
  AddKfdFilesForEveryGpu(&roots.kfd(), "4258");
  AddProcess(&roots.proc(), 4258, kOtherBdfText);
  std::filesystem::create_symlink("9", roots.proc().root() + "/4258/fdinfo/9");

  TestGpu gpu;
  WaitOutProcessListCache();
  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  EXPECT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list),
            AMDSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, ProcessListLeavesOutAProcessWhoseGpusTheCallerMayNotRead) {
  if (geteuid() == 0) GTEST_SKIP() << "Root reads files whatever their mode";
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // The caller may read neither process's KFD queue. The fdinfo of the first
  // shows this GPU; that of the second cannot be read either.
  constexpr uint32_t kShown = 4259;
  constexpr uint32_t kHidden = 4260;
  for (const uint32_t pid : {kShown, kHidden}) {
    const std::string dir = std::to_string(pid);
    roots.kfd().WriteFile(dir + "/queues/0/gpuid", std::to_string(kKfdGpuId) + "\n");
    AddKfdFilesForEveryGpu(&roots.kfd(), dir);
    std::filesystem::permissions(roots.kfd().root() + "/" + dir + "/queues/0/gpuid",
                                 std::filesystem::perms::none);
    AddProcess(&roots.proc(), pid, kBdfText);
  }
  std::filesystem::permissions(roots.proc().root() + "/4260/fdinfo", std::filesystem::perms::none);

  TestGpu gpu;
  WaitOutProcessListCache();
  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  const amdsmi_status_t status = amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list);
  std::filesystem::permissions(roots.proc().root() + "/4260/fdinfo",
                               std::filesystem::perms::owner_all);
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_processes, 1U);
  EXPECT_EQ(list[0].pid, kShown);
}

// Lists KFD processes as KFD does, except that the first listing after
// g_listings is reset fails.
std::atomic<int> g_listings{0};
int FailFirstListing(rsmi_process_info_t* procs, uint32_t num_allocated, uint32_t* num_found) {
  return g_listings++ == 0 ? EIO : amd::smi::GetProcessInfo(procs, num_allocated, num_found);
}

class ScopedKfdProcessLister {
 public:
  explicit ScopedKfdProcessLister(SmiKfdProcessListerFn lister) {
    smi_amdgpu_set_kfd_process_lister_for_testing(lister);
  }
  ~ScopedKfdProcessLister() { smi_amdgpu_set_kfd_process_lister_for_testing(nullptr); }

  ScopedKfdProcessLister(const ScopedKfdProcessLister&) = delete;
  ScopedKfdProcessLister& operator=(const ScopedKfdProcessLister&) = delete;
};

TEST(GpuUnit, ProcessListByPidLeavesOutOnlyAGpuWhoseListFails) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // KFD places the process on the second GPU through a queue.
  constexpr uint32_t kPid = 4261;
  roots.kfd().WriteFile("4261/queues/0/gpuid", std::to_string(kOtherKfdGpuId) + "\n");
  AddKfdFilesForEveryGpu(&roots.kfd(), "4261");
  AddProcess(&roots.proc(), kPid, kOtherBdfText);

  TestGpu failing;
  TestGpu other("card10", OtherTestBdf(), kOtherKfdGpuId);
  ScopedKfdNode kfd_node(kOtherKfdGpuId, other.index());
  WaitOutProcessListCache();
  // The first GPU's list fails, as when reading KFD's process directory fails.
  g_listings = 0;
  ScopedKfdProcessLister lister(FailFirstListing);
  amdsmi_processor_handle handles[] = {failing.handle(), other.handle()};
  uint32_t num_pids = 4;
  amdsmi_proc_info_by_pid_t by_pid[4] = {};
  const amdsmi_status_t status = amdsmi_get_gpu_process_list_by_pid(handles, 2, by_pid, &num_pids);
  // Both GPUs' lists were read: the first one's failed, the second one's did not.
  EXPECT_EQ(g_listings.load(), 2);
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_pids, 1U);
  EXPECT_EQ(by_pid[0].pid, kPid);
  ASSERT_EQ(by_pid[0].num_gpus, 1U);
  EXPECT_EQ(by_pid[0].gpus[0].gpu_index, other.index());
}

// Fails every listing: the first with EIO, the others because the caller may
// not read KFD's process directory.
int FailEveryListing(rsmi_process_info_t*, uint32_t, uint32_t*) {
  return g_listings++ == 0 ? EIO : EACCES;
}

TEST(GpuUnit, ProcessListByPidReturnsTheFirstErrorWhenNoGpuListIsRead) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  TestGpu first;
  TestGpu second("card10", OtherTestBdf(), kOtherKfdGpuId);
  ScopedKfdNode kfd_node(kOtherKfdGpuId, second.index());
  WaitOutProcessListCache();
  g_listings = 0;
  ScopedKfdProcessLister lister(FailEveryListing);
  amdsmi_processor_handle handles[] = {first.handle(), second.handle()};
  uint32_t num_pids = 4;
  amdsmi_proc_info_by_pid_t by_pid[4] = {};
  const amdsmi_status_t status = amdsmi_get_gpu_process_list_by_pid(handles, 2, by_pid, &num_pids);
  // Both GPUs' lists were read; the first one's error (EIO) is returned.
  EXPECT_EQ(g_listings.load(), 2);
  EXPECT_EQ(status, AMDSMI_STATUS_UNEXPECTED_SIZE);
}

}  // namespace
