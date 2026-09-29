/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

// GetProcessInfoForPID() walks every KFD GPU. These tests build a fake
// /sys/class/kfd/kfd/proc tree so a missing cu_occupancy file can be checked
// without a GPU. rsmi_compute_process_info_by_pid_get() passes the full node
// set; rsmi_compute_process_info_by_device_get() passes a one-element set.
// Both call GetProcessInfoForPID(), so the one-element cases cover the
// per-device API.

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_kfd.h"
#include "rocm_smi/rocm_smi_main.h"

namespace {

// KFD ids from the MI308X + Raphael node in the cgroup report. The smaller id
// is visited first because GetProcessInfoForPID sorts the set.
constexpr uint64_t kRaphaelGpuId = 36657;
constexpr uint64_t kMi308GpuId = 41338;
constexpr uint32_t kPid = 424242;

class ProcSysfs {
 public:
  ProcSysfs() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "rsmi-cu-occupancy-XXXXXX").string();
    if (mkdtemp(pattern.data()) == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    root_ = pattern;
    amd::smi::SetKFDProcPathRootForTest(root_.string());
    std::filesystem::create_directory(proc_dir());
    Write("pasid", "7\n");
  }

  ~ProcSysfs() {
    amd::smi::SetKFDProcPathRootForTest("");
    std::error_code ec;
    std::filesystem::remove_all(root_, ec);
  }

  void Write(const std::string& rel, const std::string& data) const {
    auto path = proc_dir() / rel;
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out.is_open()) {
      throw std::runtime_error("failed to open " + path.string());
    }
    out << data;
    if (!out.good()) {
      throw std::runtime_error("failed to write " + path.string());
    }
  }

  void Occupancy(uint64_t gpu_id, const std::string& data) const {
    Write("stats_" + std::to_string(gpu_id) + "/cu_occupancy", data);
  }

  void Vram(uint64_t gpu_id, const std::string& data) const {
    Write("vram_" + std::to_string(gpu_id), data);
  }

  void Sdma(uint64_t gpu_id, const std::string& data) const {
    Write("sdma_" + std::to_string(gpu_id), data);
  }

  std::filesystem::path OccupancyPath(uint64_t gpu_id) const {
    return proc_dir() / ("stats_" + std::to_string(gpu_id)) / "cu_occupancy";
  }

 private:
  std::filesystem::path proc_dir() const { return root_ / std::to_string(kPid); }

  std::filesystem::path root_;
};

class NodeMapGuard {
 public:
  explicit NodeMapGuard(std::vector<std::pair<uint64_t, uint32_t>> nodes) {
    auto& map = amd::smi::RocmSMI::getInstance().kfd_node_map();
    for (const auto& node : nodes) {
      ids_.push_back(node.first);
      auto kfd = std::make_shared<amd::smi::KFDNode>(0);
      kfd->set_cu_count(node.second);
      map[node.first] = kfd;
    }
  }

  ~NodeMapGuard() {
    auto& map = amd::smi::RocmSMI::getInstance().kfd_node_map();
    for (uint64_t id : ids_) {
      map.erase(id);
    }
  }

 private:
  std::vector<uint64_t> ids_;
};

int ReadProc(const std::unordered_set<uint64_t>& gpus, rsmi_process_info_t* proc) {
  std::unordered_set<uint64_t> gpu_set = gpus;
  return amd::smi::GetProcessInfoForPID(kPid, proc, &gpu_set);
}

}  // namespace

TEST(ProcessCuOccupancy, KeepsSampleWhenLaterGpuHasNoFile) {
  ProcSysfs sysfs;
  // Raphael is the lower id, so its file is read first. The MI308X file is
  // absent. The old code cleared the Raphael sample on that ENOENT.
  NodeMapGuard nodes({{kRaphaelGpuId, 8}, {kMi308GpuId, 100}});
  sysfs.Occupancy(kRaphaelGpuId, "4\n");
  sysfs.Vram(kRaphaelGpuId, "1000\n");
  sysfs.Vram(kMi308GpuId, "2000\n");
  sysfs.Sdma(kRaphaelGpuId, "5\n");
  sysfs.Sdma(kMi308GpuId, "7\n");

  rsmi_process_info_t proc{};
  ASSERT_EQ(ReadProc({kRaphaelGpuId, kMi308GpuId}, &proc), 0);
  // 4 occupied CUs / 8 CUs. The MI308X's 100 CUs are not in the denominator.
  EXPECT_EQ(proc.cu_occupancy, 50u);
  EXPECT_EQ(proc.vram_usage, 3000u);
  EXPECT_EQ(proc.sdma_usage, 12u);
  EXPECT_EQ(proc.pasid, 7u);
}

TEST(ProcessCuOccupancy, KeepsSampleWhenEarlierGpuHasNoFile) {
  ProcSysfs sysfs;
  // Missing file is visited first (Raphael), valid file second (MI308X).
  NodeMapGuard nodes({{kRaphaelGpuId, 8}, {kMi308GpuId, 100}});
  sysfs.Occupancy(kMi308GpuId, "79\n");
  sysfs.Vram(kMi308GpuId, "2337208\n");
  sysfs.Vram(kRaphaelGpuId, "4096\n");
  sysfs.Sdma(kMi308GpuId, "3\n");

  rsmi_process_info_t proc{};
  ASSERT_EQ(ReadProc({kRaphaelGpuId, kMi308GpuId}, &proc), 0);
  EXPECT_EQ(proc.cu_occupancy, 79u);
  EXPECT_EQ(proc.vram_usage, 2341304u);
  EXPECT_EQ(proc.sdma_usage, 3u);
}

TEST(ProcessCuOccupancy, InvalidOnlyWhenEveryGpuLacksAFile) {
  ProcSysfs sysfs;
  NodeMapGuard nodes({{kRaphaelGpuId, 8}, {kMi308GpuId, 100}});
  sysfs.Vram(kRaphaelGpuId, "10\n");
  sysfs.Vram(kMi308GpuId, "20\n");
  sysfs.Sdma(kRaphaelGpuId, "1\n");
  sysfs.Sdma(kMi308GpuId, "2\n");

  rsmi_process_info_t proc{};
  ASSERT_EQ(ReadProc({kRaphaelGpuId, kMi308GpuId}, &proc), 0);
  EXPECT_EQ(proc.cu_occupancy, CU_OCCUPANCY_INVALID);
  EXPECT_EQ(proc.vram_usage, 30u);
  EXPECT_EQ(proc.sdma_usage, 3u);
}

TEST(ProcessCuOccupancy, AggregatesOnlyGpusThatPublishedAFile) {
  ProcSysfs sysfs;
  constexpr uint64_t kThirdGpu = 99999;
  NodeMapGuard nodes({{kRaphaelGpuId, 50}, {kMi308GpuId, 100}, {kThirdGpu, 1000}});
  sysfs.Occupancy(kRaphaelGpuId, "10\n");
  sysfs.Occupancy(kMi308GpuId, "30\n");
  sysfs.Vram(kRaphaelGpuId, "1\n");
  sysfs.Vram(kMi308GpuId, "2\n");
  sysfs.Vram(kThirdGpu, "4\n");
  sysfs.Sdma(kThirdGpu, "9\n");

  rsmi_process_info_t proc{};
  ASSERT_EQ(ReadProc({kRaphaelGpuId, kMi308GpuId, kThirdGpu}, &proc), 0);
  // (10 + 30) * 100 / (50 + 100) = 26. The third GPU's 1000 CUs are excluded.
  EXPECT_EQ(proc.cu_occupancy, 26u);
  EXPECT_EQ(proc.vram_usage, 7u);
  EXPECT_EQ(proc.sdma_usage, 9u);
}

TEST(ProcessCuOccupancy, ZeroOccupancyIsARealSample) {
  ProcSysfs sysfs;
  NodeMapGuard nodes({{kMi308GpuId, 100}});
  sysfs.Occupancy(kMi308GpuId, "0\n");
  sysfs.Vram(kMi308GpuId, "64\n");

  rsmi_process_info_t proc{};
  ASSERT_EQ(ReadProc({kMi308GpuId}, &proc), 0);
  EXPECT_EQ(proc.cu_occupancy, 0u);
  EXPECT_EQ(proc.vram_usage, 64u);
}

TEST(ProcessCuOccupancy, MalformedOccupancyReturnsEinval) {
  ProcSysfs sysfs;
  NodeMapGuard nodes({{kRaphaelGpuId, 8}, {kMi308GpuId, 100}});
  sysfs.Occupancy(kRaphaelGpuId, "not-a-number\n");
  sysfs.Occupancy(kMi308GpuId, "79\n");
  sysfs.Vram(kMi308GpuId, "50\n");

  rsmi_process_info_t proc{};
  EXPECT_EQ(ReadProc({kRaphaelGpuId, kMi308GpuId}, &proc), EINVAL);
}

TEST(ProcessCuOccupancy, OpenFailureIsNotSkipped) {
  ProcSysfs sysfs;
  NodeMapGuard nodes({{kRaphaelGpuId, 8}, {kMi308GpuId, 100}});
  sysfs.Occupancy(kRaphaelGpuId, "4\n");
  sysfs.Occupancy(kMi308GpuId, "79\n");
  ASSERT_EQ(chmod(sysfs.OccupancyPath(kRaphaelGpuId).c_str(), 0), 0);

  rsmi_process_info_t proc{};
  int err = ReadProc({kRaphaelGpuId, kMi308GpuId}, &proc);
  EXPECT_EQ(err, EACCES);
  EXPECT_NE(err, 0);
}

TEST(ProcessCuOccupancy, PerDeviceSetIgnoresOtherGpusOnDisk) {
  ProcSysfs sysfs;
  NodeMapGuard nodes({{kRaphaelGpuId, 8}, {kMi308GpuId, 100}});
  sysfs.Occupancy(kMi308GpuId, "79\n");
  sysfs.Vram(kMi308GpuId, "100\n");
  sysfs.Vram(kRaphaelGpuId, "5\n");
  sysfs.Sdma(kMi308GpuId, "8\n");
  sysfs.Sdma(kRaphaelGpuId, "9\n");

  rsmi_process_info_t on_gpu{};
  ASSERT_EQ(ReadProc({kMi308GpuId}, &on_gpu), 0);
  EXPECT_EQ(on_gpu.cu_occupancy, 79u);
  EXPECT_EQ(on_gpu.vram_usage, 100u);
  EXPECT_EQ(on_gpu.sdma_usage, 8u);

  rsmi_process_info_t on_igpu{};
  ASSERT_EQ(ReadProc({kRaphaelGpuId}, &on_igpu), 0);
  EXPECT_EQ(on_igpu.cu_occupancy, CU_OCCUPANCY_INVALID);
  EXPECT_EQ(on_igpu.vram_usage, 5u);
  EXPECT_EQ(on_igpu.sdma_usage, 9u);
}

TEST(ProcessCuOccupancy, WideMultiplyDoesNotWrap) {
  ProcSysfs sysfs;
  NodeMapGuard nodes({{kMi308GpuId, 100}});
  // 50000000 * 100 does not fit in uint32 before the division.
  sysfs.Occupancy(kMi308GpuId, "50000000\n");

  rsmi_process_info_t proc{};
  ASSERT_EQ(ReadProc({kMi308GpuId}, &proc), 0);
  EXPECT_EQ(proc.cu_occupancy, 50000000u);
}
