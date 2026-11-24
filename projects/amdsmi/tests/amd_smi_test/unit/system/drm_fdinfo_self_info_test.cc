// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Hardware-free coverage for gpuvsmi_get_self_info() and the public
// amdsmi_get_gpu_process_info_self() wiring. Injects fake /proc/self/fdinfo and
// /proc/self/fd trees (and, for the public API, FakeKfdTopology + SelfFd* hooks).
// /dev/null and /dev/zero stand in for two render nodes: the collector matches
// descriptors by character device number, which a test cannot create for DRM.

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "amd_smi/impl/fdinfo.h"
#include "test_fixture_utils.h"

namespace {

namespace fs = std::filesystem;

constexpr char kRenderNode[] = "/dev/null";
constexpr char kSiblingNode[] = "/dev/zero";

class ScopedTempDir {
 public:
  ScopedTempDir() {
    std::string tmpl = (fs::temp_directory_path() / "drm_fdinfo_self_info_XXXXXX").string();
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (char* result = mkdtemp(buf.data())) {
      path_ = result;
    }
  }
  ~ScopedTempDir() {
    if (!path_.empty()) {
      std::error_code ec;
      fs::remove_all(path_, ec);
    }
  }
  const fs::path& path() const { return path_; }
  std::string slash_path() const { return path_.string() + "/"; }

 private:
  fs::path path_;
};

dev_t DeviceNumber(const char* path) {
  struct stat st{};
  EXPECT_EQ(stat(path, &st), 0) << path;
  return st.st_rdev;
}

void WriteFile(const fs::path& path, const std::string& content) {
  fs::create_directories(path.parent_path());
  std::ofstream(path) << content;
}

// fdinfo/<name> holds `fdinfo_body`; fd/<name> links to `target`, as /proc/self/fd does.
void WriteFdClient(const fs::path& fdinfo_dir, const fs::path& fd_dir, const std::string& name,
                   const fs::path& target, const std::string& fdinfo_body) {
  WriteFile(fdinfo_dir / name, fdinfo_body);
  fs::create_directories(fd_dir);
  std::error_code ec;
  fs::create_symlink(target, fd_dir / name, ec);
  ASSERT_FALSE(ec) << ec.message();
}

// Compute partitions of one GPU all print the same drm-pdev.
std::string AmdgpuClient(const std::string& client_key, const std::string& vram,
                         bool use_client_id) {
  std::string text = "drm-driver:\tamdgpu\ndrm-pdev:\t0000:05:00.0\n";
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
  ScopedTempDir fdinfo_dir;
  ScopedTempDir fd_dir;
  ASSERT_FALSE(fdinfo_dir.path().empty());
  ASSERT_FALSE(fd_dir.path().empty());
  WriteFdClient(fdinfo_dir.path(), fd_dir.path(), "3", kRenderNode,
                AmdgpuClient("1", "10 MiB", /*use_client_id=*/true));
  WriteFdClient(fdinfo_dir.path(), fd_dir.path(), "4", kRenderNode,
                AmdgpuClient("2", "6 MiB", /*use_client_id=*/true));
  WriteFile(fdinfo_dir.path() / "not-an-fd", "pos:\t0\n");
  WriteFdClient(fdinfo_dir.path(), fd_dir.path(), "5", kRenderNode, "pos:\t0\nflags:\t02\n");

  amdsmi_proc_info_t info{};
  ASSERT_EQ(gpuvsmi_get_self_info(DeviceNumber(kRenderNode), info, fdinfo_dir.slash_path().c_str(),
                                  fd_dir.slash_path().c_str()),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, (10ULL << 20) + (6ULL << 20));
  EXPECT_EQ(info.memory_usage.gtt_mem, 2 * (4ULL << 10));
  EXPECT_EQ(info.memory_usage.cpu_mem, 2 * (1ULL << 10));
  EXPECT_EQ(info.mem,
            info.memory_usage.vram_mem + info.memory_usage.gtt_mem + info.memory_usage.cpu_mem);
  EXPECT_EQ(info.engine_usage.gfx, 200ULL);
  EXPECT_EQ(info.engine_usage.enc, 40ULL);
  EXPECT_EQ(info.pid, static_cast<uint32_t>(getpid()));
  EXPECT_EQ(info.cu_occupancy, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(info.evicted_time, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(info.sdma_usage, std::numeric_limits<uint64_t>::max());
}

TEST(SystemUnit, GetSelfInfoDedupesDuplicateClientDescriptors) {
  ScopedTempDir fdinfo_dir;
  ScopedTempDir fd_dir;
  ASSERT_FALSE(fdinfo_dir.path().empty());
  const std::string body = AmdgpuClient("7", "8 MiB", /*use_client_id=*/true);
  WriteFdClient(fdinfo_dir.path(), fd_dir.path(), "10", kRenderNode, body);
  WriteFdClient(fdinfo_dir.path(), fd_dir.path(), "11", kRenderNode, body);

  amdsmi_proc_info_t info{};
  ASSERT_EQ(gpuvsmi_get_self_info(DeviceNumber(kRenderNode), info, fdinfo_dir.slash_path().c_str(),
                                  fd_dir.slash_path().c_str()),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 8ULL << 20);
  EXPECT_EQ(info.engine_usage.gfx, 100ULL);
}

TEST(SystemUnit, GetSelfInfoDedupesByPasidWithoutClientId) {
  ScopedTempDir fdinfo_dir;
  ScopedTempDir fd_dir;
  ASSERT_FALSE(fdinfo_dir.path().empty());
  const std::string body = AmdgpuClient("32825", "3 MiB", /*use_client_id=*/false);
  WriteFdClient(fdinfo_dir.path(), fd_dir.path(), "8", kRenderNode, body);
  WriteFdClient(fdinfo_dir.path(), fd_dir.path(), "9", kRenderNode, body);

  amdsmi_proc_info_t info{};
  ASSERT_EQ(gpuvsmi_get_self_info(DeviceNumber(kRenderNode), info, fdinfo_dir.slash_path().c_str(),
                                  fd_dir.slash_path().c_str()),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 3ULL << 20);
}

// Sibling partitions share drm-pdev, so only the render node tells them apart.
TEST(SystemUnit, GetSelfInfoCountsOnlyTheHandlesRenderNode) {
  ScopedTempDir fdinfo_dir;
  ScopedTempDir fd_dir;
  ASSERT_FALSE(fdinfo_dir.path().empty());
  WriteFdClient(fdinfo_dir.path(), fd_dir.path(), "3", kRenderNode,
                AmdgpuClient("1", "10 MiB", /*use_client_id=*/true));
  WriteFdClient(fdinfo_dir.path(), fd_dir.path(), "4", kSiblingNode,
                AmdgpuClient("2", "20 MiB", /*use_client_id=*/true));
  // Not a character device, even though its fdinfo looks like an amdgpu client.
  WriteFdClient(fdinfo_dir.path(), fd_dir.path(), "5", fdinfo_dir.path() / "3",
                AmdgpuClient("3", "30 MiB", /*use_client_id=*/true));

  amdsmi_proc_info_t info{};
  ASSERT_EQ(gpuvsmi_get_self_info(DeviceNumber(kRenderNode), info, fdinfo_dir.slash_path().c_str(),
                                  fd_dir.slash_path().c_str()),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 10ULL << 20);

  ASSERT_EQ(gpuvsmi_get_self_info(DeviceNumber(kSiblingNode), info, fdinfo_dir.slash_path().c_str(),
                                  fd_dir.slash_path().c_str()),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 20ULL << 20);
}

TEST(SystemUnit, GetSelfInfoReturnsNotFoundWhenNoClients) {
  ScopedTempDir fdinfo_dir;
  ScopedTempDir fd_dir;
  ASSERT_FALSE(fdinfo_dir.path().empty());
  WriteFdClient(fdinfo_dir.path(), fd_dir.path(), "3", kRenderNode, "pos:\t0\nflags:\t02\n");
  WriteFdClient(fdinfo_dir.path(), fd_dir.path(), "4", kSiblingNode,
                AmdgpuClient("1", "10 MiB", /*use_client_id=*/true));

  amdsmi_proc_info_t info{};
  EXPECT_EQ(gpuvsmi_get_self_info(DeviceNumber(kRenderNode), info, fdinfo_dir.slash_path().c_str(),
                                  fd_dir.slash_path().c_str()),
            AMDSMI_STATUS_NOT_FOUND);
}

TEST(SystemUnit, GetSelfInfoPropagatesReadFailure) {
  ScopedTempDir fdinfo_dir;
  ScopedTempDir fd_dir;
  ASSERT_FALSE(fdinfo_dir.path().empty());
  // A numeric directory cannot be read as an fdinfo file.
  ASSERT_TRUE(fs::create_directory(fdinfo_dir.path() / "99"));
  fs::create_directories(fd_dir.path());
  std::error_code ec;
  fs::create_symlink(kRenderNode, fd_dir.path() / "99", ec);
  ASSERT_FALSE(ec) << ec.message();

  amdsmi_proc_info_t info{};
  EXPECT_EQ(gpuvsmi_get_self_info(DeviceNumber(kRenderNode), info, fdinfo_dir.slash_path().c_str(),
                                  fd_dir.slash_path().c_str()),
            AMDSMI_STATUS_FILE_ERROR);
}

TEST(SystemUnit, GetGpuProcessInfoSelfPublicApi) {
  constexpr uint32_t kNode = 0;
  constexpr uint32_t kCard = 0;
  constexpr uint64_t kGpuId = 4242;
  constexpr uint32_t kRender = 128;  // kFirstRenderMinor + card 0

  FakeSysfsTree tree;
  WriteGpuNode(tree, kNode, kCard, kGpuId);
  const dev_t render_dev = DeviceNumber(kRenderNode);
  tree.WriteFile(
      "renderD" + std::to_string(kRender) + "/dev",
      std::to_string(major(render_dev)) + ":" + std::to_string(minor(render_dev)) + "\n");

  ScopedTempDir fdinfo_dir;
  ScopedTempDir fd_dir;
  ASSERT_FALSE(fdinfo_dir.path().empty());
  WriteFdClient(fdinfo_dir.path(), fd_dir.path(), "7", kRenderNode,
                AmdgpuClient("1", "12 MiB", /*use_client_id=*/true));
  WriteFdClient(fdinfo_dir.path(), fd_dir.path(), "8", kSiblingNode,
                AmdgpuClient("2", "20 MiB", /*use_client_id=*/true));

  FakeKfdTopology topology(tree);
  amd::smi::testing::ScopedOverride<std::string> fdinfo_guard(
      amd::smi::test_hooks::SelfFdinfoDir, amd::smi::test_hooks::SetSelfFdinfoDir,
      fdinfo_dir.slash_path());
  amd::smi::testing::ScopedOverride<std::string> fd_guard(
      amd::smi::test_hooks::SelfFdDir, amd::smi::test_hooks::SetSelfFdDir, fd_dir.slash_path());

  EXPECT_EQ(amdsmi_get_gpu_process_info_self(topology.handle(kNode), nullptr), AMDSMI_STATUS_INVAL);

  amdsmi_proc_info_t info{};
  ASSERT_EQ(amdsmi_get_gpu_process_info_self(topology.handle(kNode), &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 12ULL << 20);
  EXPECT_EQ(info.pid, static_cast<uint32_t>(getpid()));
  EXPECT_EQ(info.cu_occupancy, std::numeric_limits<uint32_t>::max());
}

TEST(SystemUnit, GetGpuProcessInfoSelfNeedsRenderNodeDevNumber) {
  constexpr uint32_t kNode = 0;
  FakeSysfsTree tree;
  WriteGpuNode(tree, kNode, /*card_idx=*/0, /*gpu_id=*/4242);
  FakeKfdTopology topology(tree);

  amdsmi_proc_info_t info{};
  EXPECT_EQ(amdsmi_get_gpu_process_info_self(topology.handle(kNode), &info),
            AMDSMI_STATUS_NOT_SUPPORTED);
}
