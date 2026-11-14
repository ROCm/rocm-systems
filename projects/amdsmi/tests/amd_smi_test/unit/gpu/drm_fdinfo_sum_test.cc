// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <stdlib.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_utils.h"

namespace {

constexpr char kBdf[] = "0000:03:00.0";

// A stand-in for /proc/<pid>/fdinfo, removed with its files at the end.
class FdinfoDir {
 public:
  FdinfoDir() {
    std::string path = (std::filesystem::temp_directory_path() / "amdsmi_fdinfo_XXXXXX").string();
    if (mkdtemp(path.data()) != nullptr) path_ = path;
  }
  ~FdinfoDir() {
    std::error_code ec;
    if (!path_.empty()) std::filesystem::remove_all(path_, ec);
  }
  FdinfoDir(const FdinfoDir&) = delete;
  FdinfoDir& operator=(const FdinfoDir&) = delete;

  const std::string& path() const { return path_; }
  void Add(const std::string& fd, const std::string& text) const {
    std::ofstream(path_ + "/" + fd) << text;
  }

 private:
  std::string path_;
};

// The fdinfo of an fd for an amdgpu DRM client, without a drm-client-id line
// when client is negative.
std::string AmdgpuFdinfo(const std::string& pdev, int64_t client, const std::string& vram,
                         const std::string& gtt, const std::string& cpu, uint64_t gfx_ns,
                         uint64_t enc_ns) {
  std::string text =
      "pos:\t0\nflags:\t02100002\nmnt_id:\t26\nino:\t1094\n"
      "drm-driver:\tamdgpu\n";
  if (client >= 0) text += "drm-client-id:\t" + std::to_string(client) + "\n";
  text += "drm-pdev:\t" + pdev + "\ndrm-memory-vram:\t" + vram + "\ndrm-memory-gtt: \t" + gtt +
          "\ndrm-memory-cpu: \t" + cpu + "\ndrm-engine-gfx:\t" + std::to_string(gfx_ns) +
          " ns\ndrm-engine-enc:\t" + std::to_string(enc_ns) + " ns\n";
  return text;
}

amdsmi_proc_info_t Sum(const FdinfoDir& dir) {
  amdsmi_proc_info_t info{};
  EXPECT_EQ(AMDSMI_STATUS_SUCCESS, smi_amdgpu_sum_drm_fdinfo(dir.path(), kBdf, &info));
  return info;
}

}  // namespace

// Each fd of a DRM client (dup, fork, fd passing) lists all of the client's
// usage.
TEST(GpuUnit, DrmFdinfoSumCountsEachClientOnce) {
  FdinfoDir dir;
  ASSERT_FALSE(dir.path().empty());
  for (const char* fd : {"3", "4", "7", "12"}) {
    dir.Add(fd, AmdgpuFdinfo(kBdf, 7, "1234 KiB", "567 KiB", "89 KiB", 111222333, 44455));
  }
  const amdsmi_proc_info_t info = Sum(dir);
  EXPECT_EQ(uint64_t{1234} * 1024, info.memory_usage.vram_mem);
  EXPECT_EQ(uint64_t{567} * 1024, info.memory_usage.gtt_mem);
  EXPECT_EQ(uint64_t{89} * 1024, info.memory_usage.cpu_mem);
  EXPECT_EQ(uint64_t{111222333}, info.engine_usage.gfx);
  EXPECT_EQ(uint64_t{44455}, info.engine_usage.enc);
}

TEST(GpuUnit, DrmFdinfoSumAddsClients) {
  FdinfoDir dir;
  ASSERT_FALSE(dir.path().empty());
  dir.Add("3", AmdgpuFdinfo(kBdf, 7, "1234 KiB", "567 KiB", "89 KiB", 111222333, 44455));
  dir.Add("4", AmdgpuFdinfo(kBdf, 7, "1234 KiB", "567 KiB", "89 KiB", 111222333, 44455));
  dir.Add("5", AmdgpuFdinfo(kBdf, 8, "2001 KiB", "13 KiB", "5 KiB", 7000000, 101));
  const amdsmi_proc_info_t info = Sum(dir);
  EXPECT_EQ(uint64_t{1234 + 2001} * 1024, info.memory_usage.vram_mem);
  EXPECT_EQ(uint64_t{567 + 13} * 1024, info.memory_usage.gtt_mem);
  EXPECT_EQ(uint64_t{89 + 5} * 1024, info.memory_usage.cpu_mem);
  EXPECT_EQ(uint64_t{111222333 + 7000000}, info.engine_usage.gfx);
  EXPECT_EQ(uint64_t{44455 + 101}, info.engine_usage.enc);
}

// Without a client id, fds cannot be matched to clients, so each one counts.
TEST(GpuUnit, DrmFdinfoSumCountsEachFdWithoutClientId) {
  FdinfoDir dir;
  ASSERT_FALSE(dir.path().empty());
  dir.Add("3", AmdgpuFdinfo(kBdf, -1, "17 KiB", "3 KiB", "1 KiB", 500, 7));
  dir.Add("4", AmdgpuFdinfo(kBdf, -1, "29 KiB", "2 KiB", "6 KiB", 900, 13));
  const amdsmi_proc_info_t info = Sum(dir);
  EXPECT_EQ(uint64_t{17 + 29} * 1024, info.memory_usage.vram_mem);
  EXPECT_EQ(uint64_t{3 + 2} * 1024, info.memory_usage.gtt_mem);
  EXPECT_EQ(uint64_t{1 + 6} * 1024, info.memory_usage.cpu_mem);
  EXPECT_EQ(uint64_t{500 + 900}, info.engine_usage.gfx);
  EXPECT_EQ(uint64_t{7 + 13}, info.engine_usage.enc);
}

// Only DRM clients on the GPU asked about count.
TEST(GpuUnit, DrmFdinfoSumSkipsOtherFiles) {
  FdinfoDir dir;
  ASSERT_FALSE(dir.path().empty());
  dir.Add("3", AmdgpuFdinfo(kBdf, 7, "1 MiB", "4096", "0 KiB", 100, 10));
  dir.Add("4", AmdgpuFdinfo("0000:c3:00.0", 9, "999 KiB", "999 KiB", "999 KiB", 999, 999));
  dir.Add("5", "pos:\t0\nflags:\t02\nmnt_id:\t15\nino:\t42\n");
  dir.Add("6", "drm-client-id:\t11\ndrm-memory-vram:\t999 KiB\ndrm-engine-gfx:\t999 ns\n");
  dir.Add(".7", AmdgpuFdinfo(kBdf, 12, "999 KiB", "999 KiB", "999 KiB", 999, 999));
  const amdsmi_proc_info_t info = Sum(dir);
  EXPECT_EQ(uint64_t{1} << 20, info.memory_usage.vram_mem);
  EXPECT_EQ(uint64_t{4096}, info.memory_usage.gtt_mem);
  EXPECT_EQ(uint64_t{0}, info.memory_usage.cpu_mem);
  EXPECT_EQ(uint64_t{100}, info.engine_usage.gfx);
  EXPECT_EQ(uint64_t{10}, info.engine_usage.enc);
}

// amdgpu lists a drm-engine-<ip> line only for an engine that the client has
// used, so each fd's values start from zero.
TEST(GpuUnit, DrmFdinfoSumStartsEachFdAtZero) {
  FdinfoDir dir;
  ASSERT_FALSE(dir.path().empty());
  const std::string head = std::string("drm-driver:\tamdgpu\ndrm-pdev:\t") + kBdf + "\n";
  dir.Add("3", head + "drm-client-id:\t7\ndrm-memory-vram:\t4 KiB\ndrm-engine-gfx:\t1000 ns\n");
  dir.Add("4", head + "drm-client-id:\t8\ndrm-memory-gtt:\t8 KiB\ndrm-engine-enc:\t30 ns\n");
  const amdsmi_proc_info_t info = Sum(dir);
  EXPECT_EQ(uint64_t{4} * 1024, info.memory_usage.vram_mem);
  EXPECT_EQ(uint64_t{8} * 1024, info.memory_usage.gtt_mem);
  EXPECT_EQ(uint64_t{0}, info.memory_usage.cpu_mem);
  EXPECT_EQ(uint64_t{1000}, info.engine_usage.gfx);
  EXPECT_EQ(uint64_t{30}, info.engine_usage.enc);
}

TEST(GpuUnit, DrmFdinfoSumWithoutDirectory) {
  FdinfoDir dir;
  ASSERT_FALSE(dir.path().empty());
  amdsmi_proc_info_t info{};
  EXPECT_EQ(AMDSMI_STATUS_NO_PERM, smi_amdgpu_sum_drm_fdinfo(dir.path() + "/missing", kBdf, &info));
  EXPECT_EQ(uint64_t{0}, info.memory_usage.vram_mem);
}
