// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Reads processes from a mock /proc and a mock KFD process root, so the
// fdinfo-based numbers are checked without a GPU or another process. In CI
// containers without the host PID namespace, a real process's fdinfo cannot be
// read, so the functional tests skip that comparison there.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

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
constexpr uint64_t kUnrelatedKfdGpuId = 5678;

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
        registration_(&gpu_device_),
        card_(card) {
    amd::smi::RocmSMI::getInstance().devices()[rocm_device_.index()]->set_kfd_gpu_id(kfd_gpu_id);
  }
  uint32_t index() const { return rocm_device_.index(); }
  amdsmi_processor_handle handle() const { return registration_.handle(); }
  // Writes `content` to the file at `path` in the GPU's sysfs directory.
  void WriteSysfsFile(const std::string& path, const std::string& content) {
    sysfs_.WriteFile(card_ + "/" + path, content);
  }

 private:
  FakeSysfsTree sysfs_;
  ScopedRocmSmiDevice rocm_device_;
  AMDSmiDrm drm_;
  AMDSmiGPUDevice gpu_device_;
  ScopedProcessorRegistration registration_;
  std::string card_;
};

// A mock /proc, KFD process root and KFD topology (no nodes unless a test adds
// them), so that no list reads the host's.
class MockProcessRoots {
 public:
  MockProcessRoots()
      : proc_root_(amd::smi::test_hooks::ProcRoot, amd::smi::test_hooks::SetProcRoot, proc_.root()),
        kfd_root_(kfd_.root()),
        topology_root_(amd::smi::test_hooks::KfdNodesPathRoot,
                       amd::smi::test_hooks::SetKfdNodesPathRoot, topology_.root()) {}

  FakeSysfsTree& proc() { return proc_; }
  FakeSysfsTree& kfd() { return kfd_; }
  FakeSysfsTree& topology() { return topology_; }

 private:
  FakeSysfsTree proc_;
  FakeSysfsTree kfd_;
  FakeSysfsTree topology_;
  ScopedOverride<std::string> proc_root_;
  ScopedKfdProcRoot kfd_root_;
  ScopedOverride<std::string> topology_root_;
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

TEST(GpuUnit, ProcessListVramIsTheLargerCountAndMemTheSum) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // KFD places the processes on this GPU through a queue. For the first it
  // reports less VRAM than its fdinfo does (7 bytes), for the second more (it
  // also counts allocations its DRM client has not mapped), for the third the
  // largest size it reports (all-ones reads as unknown), so that mem would wrap.
  constexpr uint32_t kFdinfoMore = 4253;
  constexpr uint32_t kKfdMore = 4254;
  constexpr uint32_t kKfdLargest = 4255;
  constexpr uint64_t kKfdVram = 8 * 1024 * 1024;
  constexpr uint64_t kLargestVram = std::numeric_limits<uint64_t>::max() - 1;
  for (const auto& [pid, kfd_vram] : {std::pair<uint32_t, uint64_t>{kFdinfoMore, 7},
                                      std::pair<uint32_t, uint64_t>{kKfdMore, kKfdVram},
                                      std::pair<uint32_t, uint64_t>{kKfdLargest, kLargestVram}}) {
    const std::string dir = std::to_string(pid);
    roots.kfd().WriteFile(dir + "/queues/0/gpuid", std::to_string(kKfdGpuId) + "\n");
    AddKfdFilesForEveryGpu(&roots.kfd(), dir);
    roots.kfd().WriteFile(dir + "/vram_" + std::to_string(kKfdGpuId), std::to_string(kfd_vram));
    AddProcess(&roots.proc(), pid, kBdfText);
  }

  // KFD's node offers as much memory as the GPU has VRAM, as on a discrete GPU.
  roots.topology().WriteFile("0/mem_banks/0/properties", "heap_type 1\nsize_in_bytes 536870912\n");
  TestGpu gpu;
  gpu.WriteSysfsFile("device/mem_info_vram_total", "536870912\n");
  ScopedKfdNode kfd_node(kKfdGpuId, gpu.index());
  WaitOutProcessListCache();
  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  ASSERT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_processes, 3U);
  std::vector<uint32_t> pids{list[0].pid, list[1].pid, list[2].pid};
  std::sort(pids.begin(), pids.end());
  EXPECT_EQ(pids, (std::vector<uint32_t>{kFdinfoMore, kKfdMore, kKfdLargest}));
  for (const amdsmi_proc_info_t& proc : {list[0], list[1], list[2]}) {
    const uint64_t vram = proc.pid == kKfdLargest ? kLargestVram
                          : proc.pid == kKfdMore  ? kKfdVram
                                                  : kVram;
    EXPECT_EQ(proc.memory_usage.vram_mem, vram) << "pid " << proc.pid;
    EXPECT_EQ(proc.memory_usage.gtt_mem, kGtt) << "pid " << proc.pid;
    // A sum that does not fit stops one below all-ones.
    const uint64_t mem = proc.pid == kKfdLargest ? kLargestVram : vram + kGtt + kCpu;
    EXPECT_EQ(proc.mem, mem) << "pid " << proc.pid;
  }
}

TEST(GpuUnit, ProcessListTakesKfdVramAsGttWhereAmdgpuPlacesItInGtt) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // As on an APU with less VRAM than GTT, KFD offers more local memory than the
  // GPU has VRAM, and amdgpu places what KFD allocates as VRAM in GTT. KFD
  // reports more of it for the first process than its fdinfo's GTT, less for the
  // second.
  constexpr uint32_t kKfdMore = 4262;
  constexpr uint32_t kFdinfoMore = 4263;
  constexpr uint64_t kKfdVram = 8 * 1024 * 1024;
  for (const auto& [pid, kfd_vram] : {std::pair<uint32_t, uint64_t>{kKfdMore, kKfdVram},
                                      std::pair<uint32_t, uint64_t>{kFdinfoMore, 7}}) {
    const std::string dir = std::to_string(pid);
    roots.kfd().WriteFile(dir + "/queues/0/gpuid", std::to_string(kKfdGpuId) + "\n");
    AddKfdFilesForEveryGpu(&roots.kfd(), dir);
    roots.kfd().WriteFile(dir + "/vram_" + std::to_string(kKfdGpuId), std::to_string(kfd_vram));
    AddProcess(&roots.proc(), pid, kBdfText);
  }
  roots.topology().WriteFile("0/mem_banks/0/properties",
                             "heap_type 1\nsize_in_bytes 14040223744\n");

  TestGpu gpu;
  gpu.WriteSysfsFile("device/mem_info_vram_total", "536870912\n");
  ScopedKfdNode kfd_node(kKfdGpuId, gpu.index());
  WaitOutProcessListCache();
  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  ASSERT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_processes, 2U);
  for (const amdsmi_proc_info_t& proc : {list[0], list[1]}) {
    ASSERT_TRUE(proc.pid == kKfdMore || proc.pid == kFdinfoMore) << "pid " << proc.pid;
    const uint64_t gtt = proc.pid == kKfdMore ? kKfdVram : kGtt;
    EXPECT_EQ(proc.memory_usage.vram_mem, kVram) << "pid " << proc.pid;
    EXPECT_EQ(proc.memory_usage.gtt_mem, gtt) << "pid " << proc.pid;
    EXPECT_EQ(proc.mem, kVram + gtt + kCpu) << "pid " << proc.pid;
  }
}

TEST(GpuUnit, ProcessListKeepsKfdVramAsVramWhenTheVramTotalIsNotANumber) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // KFD's node offers more memory than an APU's VRAM would be, but the GPU's
  // VRAM total is not a number, which rocm_smi throws on: the list still
  // succeeds, with KFD's count as VRAM.
  constexpr uint32_t kPid = 4265;
  constexpr uint64_t kKfdVram = 8 * 1024 * 1024;
  const std::string dir = std::to_string(kPid);
  roots.kfd().WriteFile(dir + "/queues/0/gpuid", std::to_string(kKfdGpuId) + "\n");
  AddKfdFilesForEveryGpu(&roots.kfd(), dir);
  roots.kfd().WriteFile(dir + "/vram_" + std::to_string(kKfdGpuId), std::to_string(kKfdVram));
  AddProcess(&roots.proc(), kPid, kBdfText);
  roots.topology().WriteFile("0/mem_banks/0/properties",
                             "heap_type 1\nsize_in_bytes 14040223744\n");

  TestGpu gpu;
  gpu.WriteSysfsFile("device/mem_info_vram_total", "x\n");
  ScopedKfdNode kfd_node(kKfdGpuId, gpu.index());
  WaitOutProcessListCache();
  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  ASSERT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_processes, 1U);
  EXPECT_EQ(list[0].memory_usage.vram_mem, kKfdVram);
  EXPECT_EQ(list[0].memory_usage.gtt_mem, kGtt);
}

TEST(GpuUnit, ProcessListTakesKfdVramAsGttWhereTheGpuHasNoVramTotal) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // As on an MI300A in native mode: amdgpu manages no VRAM, so the GPU has no
  // VRAM total, and it places what KFD allocates as VRAM in GTT.
  constexpr uint32_t kPid = 4275;
  constexpr uint64_t kKfdVram = 8 * 1024 * 1024;
  const std::string dir = std::to_string(kPid);
  roots.kfd().WriteFile(dir + "/queues/0/gpuid", std::to_string(kKfdGpuId) + "\n");
  AddKfdFilesForEveryGpu(&roots.kfd(), dir);
  roots.kfd().WriteFile(dir + "/vram_" + std::to_string(kKfdGpuId), std::to_string(kKfdVram));
  AddProcess(&roots.proc(), kPid, kBdfText);
  roots.topology().WriteFile("0/mem_banks/0/properties",
                             "heap_type 1\nsize_in_bytes 14040223744\n");

  TestGpu gpu;
  amd::smi::RocmSMI::getInstance().devices()[gpu.index()]->set_bdfid((0xfeULL << 8) |
                                                                     (0x1fULL << 3) | 7);
  ScopedKfdNode kfd_node(kKfdGpuId, gpu.index());
  WaitOutProcessListCache();
  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  ASSERT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_processes, 1U);
  EXPECT_EQ(list[0].memory_usage.vram_mem, kVram);
  EXPECT_EQ(list[0].memory_usage.gtt_mem, kKfdVram);
  EXPECT_EQ(list[0].mem, kVram + kKfdVram + kCpu);
}

TEST(GpuUnit, ProcessListTakesKfdVramAsGttOnEveryPartitionOfSuchAnApu) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // As on an APU in native mode split into partitions: only the first
  // partition's render node has the GPU's VRAM total, 0, and KFD offers each
  // partition more. KFD places the process on the second partition.
  constexpr uint32_t kPid = 4264;
  constexpr uint64_t kKfdVram = 8 * 1024 * 1024;
  const std::string dir = std::to_string(kPid);
  roots.kfd().WriteFile(dir + "/queues/0/gpuid", std::to_string(kKfdGpuId) + "\n");
  AddKfdFilesForEveryGpu(&roots.kfd(), dir);
  roots.kfd().WriteFile(dir + "/vram_" + std::to_string(kKfdGpuId), std::to_string(kKfdVram));
  AddProcess(&roots.proc(), kPid, kBdfText);
  roots.topology().WriteFile("0/mem_banks/0/properties",
                             "heap_type 1\nsize_in_bytes 14040223744\n");

  // rocm_smi's address of a partition holds its ID in bits 31:28 or, with some
  // drivers, in the PCI function.
  constexpr uint64_t kBdfid = (0xfeULL << 8) | (0x1fULL << 3);
  for (const uint64_t second_bdfid : {kBdfid | (uint64_t{1} << 28), kBdfid | uint64_t{1}}) {
    // Another GPU, listed before them, with more VRAM than KFD offers: only the
    // first partition's total may count.
    TestGpu other("card8", OtherTestBdf(), kUnrelatedKfdGpuId);
    other.WriteSysfsFile("device/mem_info_vram_total", "68719476736\n");
    TestGpu first("card9", TestBdf(), kOtherKfdGpuId);
    first.WriteSysfsFile("device/mem_info_vram_total", "0\n");
    TestGpu second("card10", TestBdf(), kKfdGpuId);
    auto& devices = amd::smi::RocmSMI::getInstance().devices();
    devices[other.index()]->set_bdfid(0xc300);
    devices[first.index()]->set_bdfid(kBdfid);
    devices[second.index()]->set_bdfid(second_bdfid);
    ScopedKfdNode kfd_node(kKfdGpuId, second.index());
    WaitOutProcessListCache();
    uint32_t num_processes = 4;
    amdsmi_proc_info_t list[4] = {};
    ASSERT_EQ(amdsmi_get_gpu_process_list(second.handle(), &num_processes, list),
              AMDSMI_STATUS_SUCCESS);
    ASSERT_EQ(num_processes, 1U);
    EXPECT_EQ(list[0].pid, kPid);
    EXPECT_EQ(list[0].memory_usage.vram_mem, kVram) << "address " << std::hex << second_bdfid;
    EXPECT_EQ(list[0].memory_usage.gtt_mem, kKfdVram) << "address " << std::hex << second_bdfid;
    EXPECT_EQ(list[0].mem, kVram + kKfdVram + kCpu) << "address " << std::hex << second_bdfid;
  }
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

TEST(GpuUnit, ProcessListFailsWhenReadingAnFdinfoFileFails) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // As in ProcessListFailsWhenAnFdinfoFileCannotBeRead, except that the file
  // that might have shown this GPU opens and then fails to read (a directory).
  roots.kfd().WriteFile("4264/queues/0/gpuid", "not-a-gpu-id\n");
  AddKfdFilesForEveryGpu(&roots.kfd(), "4264");
  AddProcess(&roots.proc(), 4264, kOtherBdfText);
  std::filesystem::create_directory(roots.proc().root() + "/4264/fdinfo/9");

  TestGpu gpu;
  WaitOutProcessListCache();
  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  EXPECT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list),
            AMDSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, ProcessListFailsWhenAnFdinfoFileOfAProcessOnTheGpuCannotBeRead) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // KFD places the process on this GPU through a queue. One of its fdinfo files
  // shows this GPU; another opens and then fails to read (a directory), so its
  // usage would be missing from the sizes.
  roots.kfd().WriteFile("4265/queues/0/gpuid", std::to_string(kKfdGpuId) + "\n");
  AddKfdFilesForEveryGpu(&roots.kfd(), "4265");
  AddProcess(&roots.proc(), 4265, kBdfText);
  std::filesystem::create_directory(roots.proc().root() + "/4265/fdinfo/9");

  TestGpu gpu;
  ScopedKfdNode kfd_node(kKfdGpuId, gpu.index());
  WaitOutProcessListCache();
  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  EXPECT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list),
            AMDSMI_STATUS_FILE_ERROR);
}

TEST(GpuUnit, ProcessListFailsWhenTheFdinfoOfAProcessOnTheGpuCannotBeOpened) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // KFD places the process on this GPU through a queue, and its fdinfo directory
  // cannot be opened (a file stands in its place), as with EMFILE when no fd is
  // left: its sizes would read as 0.
  roots.kfd().WriteFile("4268/queues/0/gpuid", std::to_string(kKfdGpuId) + "\n");
  AddKfdFilesForEveryGpu(&roots.kfd(), "4268");
  roots.proc().WriteFile("4268/fdinfo", "not a directory\n");

  TestGpu gpu;
  ScopedKfdNode kfd_node(kKfdGpuId, gpu.index());
  WaitOutProcessListCache();
  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  EXPECT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list),
            AMDSMI_STATUS_DIRECTORY_NOT_FOUND);
}

TEST(GpuUnit, ProcessListFailsWhenTheFdinfoOfAProcessWithoutKfdFilesForTheGpuCannotBeOpened) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // KFD places the process on this GPU through a queue but has its per-GPU files
  // only for another GPU, so its fdinfo is searched for this GPU, and that
  // directory cannot be opened (a file stands in its place).
  roots.kfd().WriteFile("4272/queues/0/gpuid", std::to_string(kKfdGpuId) + "\n");
  roots.kfd().WriteFile("4272/vram_" + std::to_string(kOtherKfdGpuId), "0\n");
  roots.proc().WriteFile("4272/fdinfo", "not a directory\n");

  TestGpu gpu;
  ScopedKfdNode kfd_node(kKfdGpuId, gpu.index());
  WaitOutProcessListCache();
  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  EXPECT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list),
            AMDSMI_STATUS_DIRECTORY_NOT_FOUND);
}

TEST(GpuUnit, ProcessListFailsWhenAnFdinfoFileOfAProcessWithoutKfdFilesForTheGpuFailsToRead) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // As in ProcessListFailsWhenTheFdinfoOfAProcessWithoutKfdFilesForTheGpuCannotBeOpened,
  // except that the directory opens: its readable file shows another GPU, and one
  // that opens and then fails to read (a directory) might have shown this one.
  roots.kfd().WriteFile("4274/queues/0/gpuid", std::to_string(kKfdGpuId) + "\n");
  roots.kfd().WriteFile("4274/vram_" + std::to_string(kOtherKfdGpuId), "0\n");
  AddProcess(&roots.proc(), 4274, kOtherBdfText);
  std::filesystem::create_directory(roots.proc().root() + "/4274/fdinfo/9");

  TestGpu gpu;
  ScopedKfdNode kfd_node(kKfdGpuId, gpu.index());
  WaitOutProcessListCache();
  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  EXPECT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list),
            AMDSMI_STATUS_FILE_ERROR);
}

TEST(GpuUnit, ProcessListGivesOnlyKfdsCountWhenTheCallerMayNotReadAnFdinfoFile) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // KFD places the process on this GPU through a queue. The caller may read one
  // of its fdinfo files but not another, as for a process it may not trace: the
  // sizes summed from the readable one are not the process's usage, so only
  // KFD's VRAM count is listed.
  constexpr uint64_t kKfdVram = 8 * 1024 * 1024;
  roots.kfd().WriteFile("4273/queues/0/gpuid", std::to_string(kKfdGpuId) + "\n");
  AddKfdFilesForEveryGpu(&roots.kfd(), "4273");
  roots.kfd().WriteFile("4273/vram_" + std::to_string(kKfdGpuId), std::to_string(kKfdVram));
  AddProcess(&roots.proc(), 4273, kBdfText);
  roots.proc().WriteFile("4273/fdinfo/9", "drm-pdev:\t" + std::string(kBdfText) + "\n");
  std::filesystem::permissions(roots.proc().root() + "/4273/fdinfo/9",
                               std::filesystem::perms::none);

  TestGpu gpu;
  ScopedKfdNode kfd_node(kKfdGpuId, gpu.index());
  WaitOutProcessListCache();
  // Root reads files whatever their mode, so the list is read as another user.
  auto list_process = [&] {
    uint32_t num_processes = 4;
    amdsmi_proc_info_t list[4] = {};
    EXPECT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(num_processes, 1U);
    EXPECT_EQ(list[0].pid, 4273U);
    EXPECT_EQ(list[0].memory_usage.vram_mem, kKfdVram);
    EXPECT_EQ(list[0].memory_usage.gtt_mem, 0U);
    EXPECT_EQ(list[0].memory_usage.cpu_mem, 0U);
    return AMDSMI_STATUS_SUCCESS;
  };
  const amdsmi_status_t status = amd::smi::testing::RunAsUnprivilegedUser(list_process);
  std::filesystem::permissions(roots.proc().root() + "/4273/fdinfo/9",
                               std::filesystem::perms::owner_all);
  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
}

TEST(GpuUnit, ProcessListKeepsAProcessOnTheGpuThatItsFdinfoDoesNotShow) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // KFD places the process on this GPU through a queue, but has its per-GPU
  // files only for another GPU, and its fdinfo shows only that GPU: the fdinfo
  // read finds nothing for this GPU, which leaves the process listed with KFD's
  // information, not the list failed.
  roots.kfd().WriteFile("4269/queues/0/gpuid", std::to_string(kKfdGpuId) + "\n");
  roots.kfd().WriteFile("4269/vram_" + std::to_string(kOtherKfdGpuId), "0\n");
  AddProcess(&roots.proc(), 4269, kOtherBdfText);

  TestGpu gpu;
  ScopedKfdNode kfd_node(kKfdGpuId, gpu.index());
  WaitOutProcessListCache();
  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  ASSERT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_processes, 1U);
  EXPECT_EQ(list[0].pid, 4269U);
}

TEST(GpuUnit, ProcessListFailsWhenAnFdinfoFileOfAProcessPlacedByFdinfoCannotBeRead) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // As in ProcessListFailsWhenAnFdinfoFileOfAProcessOnTheGpuCannotBeRead, except
  // that KFD cannot say which GPU the process uses, so its fdinfo places it.
  roots.kfd().WriteFile("4266/queues/0/gpuid", "not-a-gpu-id\n");
  AddKfdFilesForEveryGpu(&roots.kfd(), "4266");
  AddProcess(&roots.proc(), 4266, kBdfText);
  std::filesystem::create_directory(roots.proc().root() + "/4266/fdinfo/9");

  TestGpu gpu;
  WaitOutProcessListCache();
  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  EXPECT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list),
            AMDSMI_STATUS_FILE_ERROR);
}

TEST(GpuUnit, ProcessListKeepsAProcessOnTheGpuWhoseFdinfoTheCallerMayNotRead) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // KFD places the process on this GPU through a queue, and the caller may not
  // read its fdinfo, as for another user's process: it is listed with KFD's
  // VRAM count.
  constexpr uint64_t kKfdVram = 8 * 1024 * 1024;
  roots.kfd().WriteFile("4267/queues/0/gpuid", std::to_string(kKfdGpuId) + "\n");
  AddKfdFilesForEveryGpu(&roots.kfd(), "4267");
  roots.kfd().WriteFile("4267/vram_" + std::to_string(kKfdGpuId), std::to_string(kKfdVram));
  AddProcess(&roots.proc(), 4267, kBdfText);
  std::filesystem::permissions(roots.proc().root() + "/4267/fdinfo", std::filesystem::perms::none);

  TestGpu gpu;
  ScopedKfdNode kfd_node(kKfdGpuId, gpu.index());
  WaitOutProcessListCache();
  // Root reads files whatever their mode, so the list is read as another user.
  auto list_process = [&] {
    uint32_t num_processes = 4;
    amdsmi_proc_info_t list[4] = {};
    EXPECT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(num_processes, 1U);
    EXPECT_EQ(list[0].pid, 4267U);
    EXPECT_EQ(list[0].memory_usage.vram_mem, kKfdVram);
    return AMDSMI_STATUS_SUCCESS;
  };
  const amdsmi_status_t status = amd::smi::testing::RunAsUnprivilegedUser(list_process);
  std::filesystem::permissions(roots.proc().root() + "/4267/fdinfo",
                               std::filesystem::perms::owner_all);
  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
}

TEST(GpuUnit, ProcessListLeavesOutAProcessWhoseGpusTheCallerMayNotRead) {
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
  // Root reads files whatever their mode, so the list is read as another user.
  auto list_readable_process = [&] {
    uint32_t num_processes = 4;
    amdsmi_proc_info_t list[4] = {};
    EXPECT_EQ(amdsmi_get_gpu_process_list(gpu.handle(), &num_processes, list),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(num_processes, 1U);
    EXPECT_EQ(list[0].pid, kShown);
    return AMDSMI_STATUS_SUCCESS;
  };
  const amdsmi_status_t status = amd::smi::testing::RunAsUnprivilegedUser(list_readable_process);
  std::filesystem::permissions(roots.proc().root() + "/4260/fdinfo",
                               std::filesystem::perms::owner_all);
  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
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

// Lists KFD processes as KFD does, counting the listings in g_listings.
int CountListing(rsmi_process_info_t* procs, uint32_t num_allocated, uint32_t* num_found) {
  ++g_listings;
  return amd::smi::GetProcessInfo(procs, num_allocated, num_found);
}

TEST(GpuUnit, ProcessListOfAGpuIsNeverServedFromAnotherGpusCache) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // A process on each of two GPUs, placed by its KFD queue.
  constexpr uint32_t kOnFirst = 4270;
  constexpr uint32_t kOnSecond = 4271;
  for (const auto& [pid, gpu_id, pdev] :
       {std::tuple<uint32_t, uint64_t, const char*>{kOnFirst, kKfdGpuId, kBdfText},
        std::tuple<uint32_t, uint64_t, const char*>{kOnSecond, kOtherKfdGpuId, kOtherBdfText}}) {
    roots.kfd().WriteFile(std::to_string(pid) + "/queues/0/gpuid", std::to_string(gpu_id) + "\n");
    AddKfdFilesForEveryGpu(&roots.kfd(), std::to_string(pid));
    AddProcess(&roots.proc(), pid, pdev);
  }
  TestGpu first;
  TestGpu second("card10", OtherTestBdf(), kOtherKfdGpuId);
  ScopedKfdNode first_node(kKfdGpuId, first.index());
  ScopedKfdNode second_node(kOtherKfdGpuId, second.index());
  ScopedKfdProcessLister lister(CountListing);
  WaitOutProcessListCache();
  g_listings = 0;
  // Back to back, within the cache period (1 ms by default), so a list cached
  // for one GPU would be served for the other.
  auto listed_pid = [](amdsmi_processor_handle handle) {
    uint32_t num_processes = 4;
    amdsmi_proc_info_t list[4] = {};
    EXPECT_EQ(amdsmi_get_gpu_process_list(handle, &num_processes, list), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(num_processes, 1U);
    return list[0].pid;
  };
  const uint32_t first_pid = listed_pid(first.handle());
  const uint32_t second_pid = listed_pid(second.handle());
  const uint32_t first_pid_again = listed_pid(first.handle());
  // Every call listed KFD's processes anew: they came further apart than the
  // cache period, or it is 0, so no list could come from a cache.
  if (g_listings == 3) {
    GTEST_SKIP() << "No list was served from a cache";
  }
  EXPECT_EQ(first_pid, kOnFirst);
  EXPECT_EQ(second_pid, kOnSecond);
  EXPECT_EQ(first_pid_again, kOnFirst);
}

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

TEST(GpuUnit, ProcessListByPidSucceedsWhenTheOnlyListReadIsEmpty) {
  ScopedAmdsmiInit smi_init;
  MockProcessRoots roots;
  // No process uses a GPU, so the second GPU's list is read and empty.
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
  // A list was read, so the first one's error is not returned.
  EXPECT_EQ(g_listings.load(), 2);
  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(num_pids, 0U);
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
