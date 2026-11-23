// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Hardware-free coverage for gpuvsmi_get_self_info(): inject a fake
// /proc/self/fdinfo directory and assert totals, dedupe, physical-BDF
// matching, NOT_FOUND, and KFD unavailable sentinels.

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/fdinfo.h"

namespace {

class ScopedTempDir {
 public:
  ScopedTempDir() {
    std::string tmpl =
        (std::filesystem::temp_directory_path() / "drm_fdinfo_self_info_XXXXXX").string();
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (char* result = mkdtemp(buf.data())) {
      path_ = result;
    }
  }
  ~ScopedTempDir() {
    if (!path_.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(path_, ec);
    }
  }
  const std::filesystem::path& path() const { return path_; }
  std::string slash_path() const { return path_.string() + "/"; }

 private:
  std::filesystem::path path_;
};

void WriteFdinfo(const std::filesystem::path& dir, const std::string& name,
                 const std::string& content) {
  std::ofstream(dir / name) << content;
}

amdsmi_bdf_t MakeBdf(uint32_t domain, uint32_t bus, uint32_t device, uint32_t function) {
  amdsmi_bdf_t bdf{};
  bdf.domain_number = domain;
  bdf.bus_number = bus & 0xffu;
  bdf.device_number = device & 0x1fu;
  bdf.function_number = function & 0x7u;
  return bdf;
}

std::string AmdgpuClient(const std::string& pdev, const std::string& client_key,
                         const std::string& vram, bool use_client_id) {
  std::string text = "drm-driver:\tamdgpu\ndrm-pdev:\t" + pdev + "\n";
  if (use_client_id) {
    text += "drm-client-id:\t" + client_key + "\n";
  } else {
    text += "pasid:\t" + client_key + "\n";
  }
  text += "drm-total-vram:\t" + vram + "\n";
  text += "drm-total-gtt:\t4 KiB\n";
  text += "drm-total-cpu:\t1 KiB\n";
  text += "drm-engine-gfx:\t100 ns\n";
  text += "drm-engine-enc:\t20 ns\n";
  return text;
}

}  // namespace

TEST(SystemUnit, GetSelfInfoSumsDistinctClients) {
  ScopedTempDir dir;
  ASSERT_FALSE(dir.path().empty());
  WriteFdinfo(dir.path(), "3",
              AmdgpuClient("0000:05:00.0", "1", "10 MiB", /*use_client_id=*/true));
  WriteFdinfo(dir.path(), "4",
              AmdgpuClient("0000:05:00.0", "2", "6 MiB", /*use_client_id=*/true));
  // Nonnumeric and non-DRM noise must be ignored.
  WriteFdinfo(dir.path(), "not-an-fd", "pos:\t0\n");
  WriteFdinfo(dir.path(), "5", "pos:\t0\nflags:\t02\n");

  amdsmi_proc_info_t info{};
  ASSERT_EQ(gpuvsmi_get_self_info(MakeBdf(0, 0x05, 0, 0), info, dir.slash_path().c_str()),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, (10ULL << 20) + (6ULL << 20));
  EXPECT_EQ(info.memory_usage.gtt_mem, 2 * (4ULL << 10));
  EXPECT_EQ(info.memory_usage.cpu_mem, 2 * (1ULL << 10));
  EXPECT_EQ(info.mem, info.memory_usage.vram_mem + info.memory_usage.gtt_mem +
                          info.memory_usage.cpu_mem);
  EXPECT_EQ(info.engine_usage.gfx, 200ULL);
  EXPECT_EQ(info.engine_usage.enc, 40ULL);
  EXPECT_EQ(info.pid, static_cast<uint32_t>(getpid()));
  EXPECT_EQ(info.cu_occupancy, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(info.evicted_time, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(info.sdma_usage, std::numeric_limits<uint64_t>::max());
}

TEST(SystemUnit, GetSelfInfoDedupesDuplicateClientDescriptors) {
  ScopedTempDir dir;
  ASSERT_FALSE(dir.path().empty());
  const std::string body = AmdgpuClient("0000:05:00.0", "7", "8 MiB", /*use_client_id=*/true);
  WriteFdinfo(dir.path(), "10", body);
  WriteFdinfo(dir.path(), "11", body);

  amdsmi_proc_info_t info{};
  ASSERT_EQ(gpuvsmi_get_self_info(MakeBdf(0, 0x05, 0, 0), info, dir.slash_path().c_str()),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 8ULL << 20);
  EXPECT_EQ(info.engine_usage.gfx, 100ULL);
}

TEST(SystemUnit, GetSelfInfoDedupesByPasidWithoutClientId) {
  ScopedTempDir dir;
  ASSERT_FALSE(dir.path().empty());
  const std::string body = AmdgpuClient("0000:05:00.0", "32825", "3 MiB", /*use_client_id=*/false);
  WriteFdinfo(dir.path(), "8", body);
  WriteFdinfo(dir.path(), "9", body);

  amdsmi_proc_info_t info{};
  ASSERT_EQ(gpuvsmi_get_self_info(MakeBdf(0, 0x05, 0, 0), info, dir.slash_path().c_str()),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 3ULL << 20);
}

TEST(SystemUnit, GetSelfInfoIgnoresOtherDevices) {
  ScopedTempDir dir;
  ASSERT_FALSE(dir.path().empty());
  WriteFdinfo(dir.path(), "3",
              AmdgpuClient("0000:05:00.0", "1", "10 MiB", /*use_client_id=*/true));
  WriteFdinfo(dir.path(), "4",
              AmdgpuClient("0000:06:00.0", "2", "20 MiB", /*use_client_id=*/true));

  amdsmi_proc_info_t info{};
  ASSERT_EQ(gpuvsmi_get_self_info(MakeBdf(0, 0x05, 0, 0), info, dir.slash_path().c_str()),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 10ULL << 20);

  memset(&info, 0, sizeof(info));
  EXPECT_EQ(gpuvsmi_get_self_info(MakeBdf(0, 0x07, 0, 0), info, dir.slash_path().c_str()),
            AMDSMI_STATUS_NOT_FOUND);
}

// Partition handles may encode the partition index in function_number; fdinfo
// still prints physical .0. Matching must force function to 0.
TEST(SystemUnit, GetSelfInfoMatchesPhysicalBdfForPartitionHandle) {
  ScopedTempDir dir;
  ASSERT_FALSE(dir.path().empty());
  WriteFdinfo(dir.path(), "3",
              AmdgpuClient("0000:0c:00.0", "1", "5 MiB", /*use_client_id=*/true));

  amdsmi_proc_info_t info{};
  ASSERT_EQ(gpuvsmi_get_self_info(MakeBdf(0, 0x0c, 0, 3), info, dir.slash_path().c_str()),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 5ULL << 20);
}

TEST(SystemUnit, GetSelfInfoReturnsNotFoundWhenNoClients) {
  ScopedTempDir dir;
  ASSERT_FALSE(dir.path().empty());
  WriteFdinfo(dir.path(), "3", "pos:\t0\nflags:\t02\n");

  amdsmi_proc_info_t info{};
  EXPECT_EQ(gpuvsmi_get_self_info(MakeBdf(0, 0x05, 0, 0), info, dir.slash_path().c_str()),
            AMDSMI_STATUS_NOT_FOUND);
}
