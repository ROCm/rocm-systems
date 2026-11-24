// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Hardware-free coverage for gpuvsmi_get_self_info() and the public
// amdsmi_get_gpu_process_info_self() wiring. Tests open real descriptors and a
// fake reader supplies their fdinfo text (and, for the public API,
// FakeKfdTopology + the SelfFdinfoReaderHook). /dev/null and /dev/zero stand
// in for two render nodes: the collector matches descriptors by character
// device number, which a test cannot create for DRM.

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_self_fdinfo.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "test_fixture_utils.h"

namespace {

constexpr char kRenderNode[] = "/dev/null";
constexpr char kSiblingNode[] = "/dev/zero";

class ScopedFd {
 public:
  explicit ScopedFd(const char* path) : fd_(open(path, O_RDONLY | O_CLOEXEC)) {
    EXPECT_GE(fd_, 0) << path;
  }
  ~ScopedFd() {
    if (fd_ >= 0) close(fd_);
  }
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;
  int get() const { return fd_; }

 private:
  int fd_;
};

dev_t DeviceNumber(const char* path) {
  struct stat st{};
  EXPECT_EQ(stat(path, &st), 0) << path;
  return st.st_rdev;
}

// What the fake reader returns, keyed by descriptor number. Unlisted
// descriptors read as a non-DRM file.
struct FakeFdinfo {
  std::map<int, std::string> text;
  std::map<int, amdsmi_status_t> status;
  // After the first read, replace the descriptor with a duplicate of this one,
  // as another thread closing it and opening a different file would.
  std::map<int, int> reuse_as;
};

FakeFdinfo* g_fake = nullptr;

amdsmi_status_t ReadFakeFdinfo(int fd, std::string* text) {
  if (auto it = g_fake->status.find(fd); it != g_fake->status.end()) return it->second;
  auto it = g_fake->text.find(fd);
  *text = it == g_fake->text.end() ? "pos:\t0\nflags:\t02\n" : it->second;
  if (auto reuse = g_fake->reuse_as.find(fd); reuse != g_fake->reuse_as.end()) {
    EXPECT_EQ(dup2(reuse->second, fd), fd);
    g_fake->text[fd] = g_fake->text[reuse->second];
    g_fake->reuse_as.erase(reuse);
  }
  return AMDSMI_STATUS_SUCCESS;
}

class ScopedFakeFdinfo {
 public:
  ScopedFakeFdinfo() { g_fake = &fake_; }
  ~ScopedFakeFdinfo() { g_fake = nullptr; }
  FakeFdinfo* operator->() { return &fake_; }

 private:
  FakeFdinfo fake_;
};

amdsmi_status_t GetSelfInfo(const char* render_node, amdsmi_proc_info_t* info) {
  return gpuvsmi_get_self_info(DeviceNumber(render_node), *info, ReadFakeFdinfo);
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
  ScopedFakeFdinfo fake;
  ScopedFd a(kRenderNode), b(kRenderNode), other(kRenderNode);
  fake->text[a.get()] = AmdgpuClient("1", "10 MiB", /*use_client_id=*/true);
  fake->text[b.get()] = AmdgpuClient("2", "6 MiB", /*use_client_id=*/true);

  amdsmi_proc_info_t info{};
  ASSERT_EQ(GetSelfInfo(kRenderNode, &info), AMDSMI_STATUS_SUCCESS);
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
  ScopedFakeFdinfo fake;
  ScopedFd a(kRenderNode), b(kRenderNode);
  fake->text[a.get()] = fake->text[b.get()] = AmdgpuClient("7", "8 MiB", /*use_client_id=*/true);

  amdsmi_proc_info_t info{};
  ASSERT_EQ(GetSelfInfo(kRenderNode, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 8ULL << 20);
  EXPECT_EQ(info.engine_usage.gfx, 100ULL);
}

TEST(SystemUnit, GetSelfInfoDedupesByPasidWithoutClientId) {
  ScopedFakeFdinfo fake;
  ScopedFd a(kRenderNode), b(kRenderNode);
  fake->text[a.get()] = fake->text[b.get()] =
      AmdgpuClient("32825", "3 MiB", /*use_client_id=*/false);

  amdsmi_proc_info_t info{};
  ASSERT_EQ(GetSelfInfo(kRenderNode, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 3ULL << 20);
}

// Sibling partitions share drm-pdev, so only the render node tells them apart.
TEST(SystemUnit, GetSelfInfoCountsOnlyTheHandlesRenderNode) {
  ScopedFakeFdinfo fake;
  ScopedFd mine(kRenderNode), sibling(kSiblingNode), not_a_device("/proc/self/stat");
  fake->text[mine.get()] = AmdgpuClient("1", "10 MiB", /*use_client_id=*/true);
  fake->text[sibling.get()] = AmdgpuClient("2", "20 MiB", /*use_client_id=*/true);
  fake->text[not_a_device.get()] = AmdgpuClient("3", "30 MiB", /*use_client_id=*/true);

  amdsmi_proc_info_t info{};
  ASSERT_EQ(GetSelfInfo(kRenderNode, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 10ULL << 20);

  ASSERT_EQ(GetSelfInfo(kSiblingNode, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 20ULL << 20);
}

// A sibling's descriptor number reused for this render node between the fdinfo
// read and the device check must not count the sibling's client here.
TEST(SystemUnit, GetSelfInfoSkipsDescriptorReusedForThisDevice) {
  ScopedFakeFdinfo fake;
  ScopedFd mine(kRenderNode), sibling(kSiblingNode);
  fake->text[mine.get()] = AmdgpuClient("1", "10 MiB", /*use_client_id=*/true);
  fake->text[sibling.get()] = AmdgpuClient("2", "20 MiB", /*use_client_id=*/true);
  fake->reuse_as[sibling.get()] = mine.get();

  amdsmi_proc_info_t info{};
  ASSERT_EQ(GetSelfInfo(kRenderNode, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 10ULL << 20);
}

TEST(SystemUnit, GetSelfInfoSkipsDescriptorReusedForAnotherDevice) {
  ScopedFakeFdinfo fake;
  ScopedFd mine(kRenderNode), sibling(kSiblingNode);
  fake->text[mine.get()] = AmdgpuClient("1", "10 MiB", /*use_client_id=*/true);
  fake->text[sibling.get()] = AmdgpuClient("2", "20 MiB", /*use_client_id=*/true);
  fake->reuse_as[mine.get()] = sibling.get();

  amdsmi_proc_info_t info{};
  EXPECT_EQ(GetSelfInfo(kRenderNode, &info), AMDSMI_STATUS_NOT_FOUND);
}

// Kernels 5.14+ print the open file's inode; it must be the one fstat saw.
TEST(SystemUnit, GetSelfInfoChecksFdinfoInode) {
  ScopedFakeFdinfo fake;
  ScopedFd mine(kRenderNode);
  struct stat st{};
  ASSERT_EQ(fstat(mine.get(), &st), 0);
  const std::string client = AmdgpuClient("1", "10 MiB", /*use_client_id=*/true);

  amdsmi_proc_info_t info{};
  fake->text[mine.get()] = client + "ino:\t" + std::to_string(st.st_ino) + "\n";
  ASSERT_EQ(GetSelfInfo(kRenderNode, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 10ULL << 20);

  fake->text[mine.get()] = client + "ino:\t" + std::to_string(st.st_ino + 1) + "\n";
  EXPECT_EQ(GetSelfInfo(kRenderNode, &info), AMDSMI_STATUS_NOT_FOUND);
}

// Closing any descriptor of a file releases the process's POSIX record locks on
// it, so the scan must not open and close its own descriptors for the files it
// inspects. Another process checks that the lock is still held.
TEST(SystemUnit, GetSelfInfoKeepsRecordLocks) {
  char path[] = "/tmp/amdsmi_self_info_lock_XXXXXX";
  const int locked = mkstemp(path);
  ASSERT_GE(locked, 0);
  unlink(path);
  struct flock lock{};
  lock.l_type = F_WRLCK;
  lock.l_whence = SEEK_SET;
  ASSERT_EQ(fcntl(locked, F_SETLK, &lock), 0);

  amdsmi_proc_info_t info{};
  EXPECT_EQ(gpuvsmi_get_self_info(DeviceNumber(kRenderNode), info), AMDSMI_STATUS_NOT_FOUND);

  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    struct flock probe{};
    probe.l_type = F_WRLCK;
    probe.l_whence = SEEK_SET;
    _exit(fcntl(locked, F_GETLK, &probe) == 0 && probe.l_type == F_WRLCK ? 0 : 1);
  }
  int wstatus = 0;
  ASSERT_EQ(waitpid(child, &wstatus, 0), child);
  EXPECT_TRUE(WIFEXITED(wstatus) && WEXITSTATUS(wstatus) == 0) << "record lock was released";
  close(locked);
}

TEST(SystemUnit, GetSelfInfoReturnsNotFoundWhenNoClients) {
  ScopedFakeFdinfo fake;
  ScopedFd non_drm(kRenderNode), sibling(kSiblingNode);
  fake->text[sibling.get()] = AmdgpuClient("1", "10 MiB", /*use_client_id=*/true);

  amdsmi_proc_info_t info{};
  EXPECT_EQ(GetSelfInfo(kRenderNode, &info), AMDSMI_STATUS_NOT_FOUND);
}

TEST(SystemUnit, GetSelfInfoSkipsDescriptorClosedDuringRead) {
  ScopedFakeFdinfo fake;
  ScopedFd closed(kRenderNode), mine(kRenderNode);
  fake->status[closed.get()] = AMDSMI_STATUS_NOT_FOUND;
  fake->text[mine.get()] = AmdgpuClient("1", "10 MiB", /*use_client_id=*/true);

  amdsmi_proc_info_t info{};
  ASSERT_EQ(GetSelfInfo(kRenderNode, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 10ULL << 20);
}

TEST(SystemUnit, GetSelfInfoPropagatesReadFailure) {
  ScopedFakeFdinfo fake;
  ScopedFd failing(kRenderNode), mine(kRenderNode);
  fake->status[failing.get()] = AMDSMI_STATUS_FILE_ERROR;
  fake->text[mine.get()] = AmdgpuClient("1", "10 MiB", /*use_client_id=*/true);

  amdsmi_proc_info_t info{};
  EXPECT_EQ(GetSelfInfo(kRenderNode, &info), AMDSMI_STATUS_FILE_ERROR);
}

TEST(SystemUnit, ReadFdinfoFileReportsDescriptorClosedAfterOpen) {
  const int target = open(kRenderNode, O_RDONLY | O_CLOEXEC);
  ASSERT_GE(target, 0);
  const std::string path = "/proc/self/fdinfo/" + std::to_string(target);
  const int open_fdinfo = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  ASSERT_GE(open_fdinfo, 0);
  std::string text;
  EXPECT_EQ(amd::smi::ReadFdinfoFile(open_fdinfo, &text), AMDSMI_STATUS_SUCCESS);
  EXPECT_NE(text.find("flags:"), std::string::npos);
  close(open_fdinfo);

  const int closing_fdinfo = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  ASSERT_GE(closing_fdinfo, 0);
  ASSERT_EQ(close(target), 0);
  EXPECT_EQ(amd::smi::ReadFdinfoFile(closing_fdinfo, &text), AMDSMI_STATUS_NOT_FOUND);
  close(closing_fdinfo);
}

TEST(SystemUnit, GetSelfInfoRejectsInvalidCounters) {
  ScopedFakeFdinfo fake;
  ScopedFd mine(kRenderNode), sibling(kSiblingNode);
  fake->text[mine.get()] = AmdgpuClient("1", "garbage", /*use_client_id=*/true);
  fake->text[sibling.get()] = AmdgpuClient("2", "20 MiB", /*use_client_id=*/true);

  amdsmi_proc_info_t info{};
  EXPECT_EQ(GetSelfInfo(kRenderNode, &info), AMDSMI_STATUS_UNEXPECTED_DATA);
  // Bad data on another device does not fail this one.
  ASSERT_EQ(GetSelfInfo(kSiblingNode, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 20ULL << 20);
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

  ScopedFakeFdinfo fake;
  ScopedFd mine(kRenderNode), sibling(kSiblingNode);
  fake->text[mine.get()] = AmdgpuClient("1", "12 MiB", /*use_client_id=*/true);
  fake->text[sibling.get()] = AmdgpuClient("2", "20 MiB", /*use_client_id=*/true);

  FakeKfdTopology topology(tree);
  amd::smi::testing::ScopedOverride<SelfFdinfoReader> reader_guard(
      amd::smi::test_hooks::SelfFdinfoReaderHook, amd::smi::test_hooks::SetSelfFdinfoReaderHook,
      ReadFakeFdinfo);

  EXPECT_EQ(amdsmi_get_gpu_process_info_self(topology.handle(kNode), nullptr), AMDSMI_STATUS_INVAL);

  amdsmi_proc_info_t info{};
  ASSERT_EQ(amdsmi_get_gpu_process_info_self(topology.handle(kNode), &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.memory_usage.vram_mem, 12ULL << 20);
  EXPECT_EQ(info.pid, static_cast<uint32_t>(getpid()));
  EXPECT_EQ(info.cu_occupancy, std::numeric_limits<uint32_t>::max());

  fake->text[mine.get()] = AmdgpuClient("1", "garbage", /*use_client_id=*/true);
  EXPECT_EQ(amdsmi_get_gpu_process_info_self(topology.handle(kNode), &info),
            AMDSMI_STATUS_UNEXPECTED_DATA);
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
