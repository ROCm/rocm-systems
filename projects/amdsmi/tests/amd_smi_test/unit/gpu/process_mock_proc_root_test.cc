// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Reads processes from a mock /proc and a mock KFD process root, so the
// fdinfo-based numbers are checked without a GPU or another process. In CI
// containers without the host PID namespace, a real process's fdinfo cannot be
// read, so the functional tests skip that comparison there.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "amd_smi/impl/fdinfo.h"
#include "rocm_smi/rocm_smi_kfd.h"
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

class MockProcessRoots {
 public:
  MockProcessRoots()
      : proc_root_(smi_amdgpu_proc_root, smi_amdgpu_set_proc_root_for_testing, proc_.root() + "/"),
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
  // Neither process's KFD queue names a readable GPU. The first one's fdinfo
  // shows this GPU, the second one's another GPU.
  constexpr uint32_t kOnThisGpu = 4250;
  constexpr uint32_t kOnAnotherGpu = 4251;
  roots.kfd().WriteFile("4250/queues/0/gpuid", "not-a-gpu-id\n");
  roots.kfd().WriteFile("4251/queues/0/gpuid", "not-a-gpu-id\n");
  AddProcess(&roots.proc(), kOnThisGpu, kBdfText);
  AddProcess(&roots.proc(), kOnAnotherGpu, kOtherBdfText);

  FakeSysfsTree sysfs;
  ScopedRocmSmiDevice rocm_device(sysfs.root() + "/card9");
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), "card9", TestBdf(), drm);
  ScopedProcessorRegistration registration(&gpu_device);

  uint32_t num_processes = 4;
  amdsmi_proc_info_t list[4] = {};
  ASSERT_EQ(amdsmi_get_gpu_process_list(registration.handle(), &num_processes, list),
            AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_processes, 1U);
  EXPECT_EQ(list[0].pid, kOnThisGpu);
  EXPECT_EQ(list[0].memory_usage.vram_mem, kVram);
  EXPECT_EQ(list[0].mem, kVram + kGtt + kCpu);
  // Its KFD statistics are not available.
  EXPECT_EQ(list[0].cu_occupancy, std::numeric_limits<decltype(list[0].cu_occupancy)>::max());
  EXPECT_EQ(list[0].evicted_time, std::numeric_limits<decltype(list[0].evicted_time)>::max());
  EXPECT_EQ(list[0].sdma_usage, std::numeric_limits<decltype(list[0].sdma_usage)>::max());
}

}  // namespace
