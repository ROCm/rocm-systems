// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Reads the processes of a PID namespace other than the initial one, as in a
// container, from a mock /proc, so that the namespaced listing, GPU lookup and
// process lists run without a GPU, a container or another process. A mock fd
// links to a character device of this machine, which statx() then reports as
// it would through /proc/<pid>/fd.

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_kfd.h"
#include "rocm_smi/rocm_smi_main.h"
#include "test_fixture_utils.h"

namespace {

using amd::smi::AMDSmiDrm;
using amd::smi::AMDSmiGPUDevice;
using amd::smi::testing::ScopedOverride;
using amd::smi::testing::ScopedProcessorRegistration;

// A GPU no machine is likely to have, by address and by KFD id, and the KFD id
// of a second one. A queue doorbell's mmap offset holds the KFD id in 16 bits.
constexpr char kBdfText[] = "0000:fe:1e.3";
constexpr char kOtherBdfText[] = "0000:fd:00.0";
constexpr uint64_t kKfdGpuId = 41671;
constexpr uint64_t kOtherKfdGpuId = 52813;
constexpr uint64_t kUnknownKfdGpuId = 1239;
constexpr uint64_t kNotAvailable64 = std::numeric_limits<uint64_t>::max();
constexpr uint32_t kNotAvailable32 = std::numeric_limits<uint32_t>::max();

amdsmi_bdf_t TestBdf() {
  amdsmi_bdf_t bdf{};
  bdf.bus_number = 0xfe;
  bdf.device_number = 0x1e;
  bdf.function_number = 3;
  return bdf;
}

// A character device of this machine: its device number, and the device of its
// file system and its inode, which /proc/<pid>/maps shows for a mapping of it.
struct CharDevice {
  std::string path;
  uint32_t major = 0;
  uint32_t minor = 0;
  uint32_t fs_major = 0;
  uint32_t fs_minor = 0;
  uint64_t inode = 0;
};

std::optional<CharDevice> StatCharDevice(const char* path) {
  struct statx stx{};
  if (statx(AT_FDCWD, path, 0, STATX_TYPE | STATX_INO, &stx) != 0 || !S_ISCHR(stx.stx_mode)) {
    return std::nullopt;
  }
  return CharDevice{
      path,       stx.stx_rdev_major, stx.stx_rdev_minor, stx.stx_dev_major, stx.stx_dev_minor,
      stx.stx_ino};
}

// Stand in for the two GPUs' render nodes, /dev/zero and /dev/full: they share
// a major, as render nodes do, and have minors other than 0 (a device without a
// known render node has minor 0). /dev/kfd's stand-in has another major.
struct GpuFiles {
  CharDevice kfd;
  CharDevice render_node;
  CharDevice other_render_node;
};

std::optional<GpuFiles> FindGpuFiles() {
  const auto node = StatCharDevice("/dev/zero");
  const auto other_node = StatCharDevice("/dev/full");
  if (!node || !other_node || node->major != other_node->major || node->minor == 0 ||
      other_node->minor == 0) {
    return std::nullopt;
  }
  for (const char* path : {"/dev/ptmx", "/dev/tty", "/dev/console"}) {
    const auto kfd = StatCharDevice(path);
    if (kfd && kfd->major != node->major) return GpuFiles{*kfd, *node, *other_node};
  }
  return std::nullopt;
}

// Which GPU's render node a mock process holds open.
enum class Node { kNone, kThisGpu, kOtherGpu };

// A PID namespace's /proc with GPU processes, and the GPUs they use: the
// rocm_smi device at sysfs/card9 with this test GPU's address, KFD id and render
// node, the one at sysfs/card10 with the other GPU's KFD id and render node, and
// entries in KFD's node map as discovery makes for real GPUs. A mock KFD process
// tree has a directory for each of the namespace's PIDs, as the host's processes
// of those PIDs would; the namespace's reads must not use them.
class MockPidNamespace {
 public:
  explicit MockPidNamespace(const GpuFiles& files)
      : files_(files),
        rocm_device_(sysfs_.root() + "/card9"),
        other_rocm_device_(sysfs_.root() + "/card10"),
        kfd_node_(kKfdGpuId, rocm_device_.index()),
        other_kfd_node_(kOtherKfdGpuId, other_rocm_device_.index()),
        gpu_device_(rocm_device_.index(), "card9", TestBdf(), drm_),
        registration_(&gpu_device_),
        amdsmi_proc_root_(amd::smi::test_hooks::ProcRoot, amd::smi::test_hooks::SetProcRoot,
                          proc_.root()),
        topology_root_(amd::smi::test_hooks::KfdNodesPathRoot,
                       amd::smi::test_hooks::SetKfdNodesPathRoot, topology_.root()) {
    auto& devices = amd::smi::RocmSMI::getInstance().devices();
    devices[rocm_device_.index()]->set_kfd_gpu_id(kKfdGpuId);
    devices[rocm_device_.index()]->set_drm_render_minor(files_.render_node.minor);
    devices[other_rocm_device_.index()]->set_kfd_gpu_id(kOtherKfdGpuId);
    devices[other_rocm_device_.index()]->set_drm_render_minor(files_.other_render_node.minor);
    proc_.WriteFile("devices", "Character devices:\n" + std::to_string(files_.kfd.major) +
                                   " kfd\n" + std::to_string(files_.render_node.major) +
                                   " drm\n\nBlock devices:\n");
    amd::smi::SetProcPathRootForTesting(proc_.root().c_str());
    amd::smi::SetKFDProcPathRootForTesting(kfd_.root().c_str());
  }
  ~MockPidNamespace() {
    amd::smi::SetKFDProcPathRootForTesting(nullptr);
    amd::smi::SetProcPathRootForTesting(nullptr);
  }

  MockPidNamespace(const MockPidNamespace&) = delete;
  MockPidNamespace& operator=(const MockPidNamespace&) = delete;

  uint32_t index() const { return rocm_device_.index(); }
  uint32_t other_index() const { return other_rocm_device_.index(); }
  amdsmi_processor_handle handle() const { return registration_.handle(); }

  // Adds process `pid`: with /dev/kfd open if `kfd`; with the render node of the
  // GPU `node` names open, its DRM client holding `vram_kib` KiB of VRAM; and
  // with a queue doorbell of the GPU of KFD id `doorbell_gpu` mapped from its
  // /dev/kfd unless that id is 0.
  void AddProcess(uint32_t pid, bool kfd, Node node, uint64_t doorbell_gpu, uint64_t vram_kib = 0) {
    const std::string dir = std::to_string(pid);
    const std::string path = proc_.root() + "/" + dir;
    std::filesystem::create_directories(path + "/fd");
    std::filesystem::create_directories(path + "/fdinfo");
    std::filesystem::create_symlink("/usr/bin/fake_namespace_app", path + "/exe");
    proc_.WriteFile(dir + "/cgroup", "0::/\n");
    // A file that is not a device, as every process holds some.
    std::filesystem::create_symlink(proc_.root() + "/devices", path + "/fd/0");
    if (kfd) std::filesystem::create_symlink(files_.kfd.path, path + "/fd/3");
    if (node != Node::kNone) {
      const bool other = node == Node::kOtherGpu;
      std::filesystem::create_symlink(
          other ? files_.other_render_node.path : files_.render_node.path, path + "/fd/4");
      proc_.WriteFile(dir + "/fdinfo/4",
                      "pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t" + dir + "\ndrm-pdev:\t" +
                          (other ? kOtherBdfText : kBdfText) + "\ndrm-memory-vram:\t" +
                          std::to_string(vram_kib) +
                          " KiB\ndrm-memory-gtt:\t2 MiB\ndrm-memory-cpu:\t3 KiB\n");
    }
    std::string maps = "55d0c0000000-55d0c0001000 r-xp 00000000 08:02 131 /usr/bin/app\n";
    if (doorbell_gpu != 0) {
      const uint64_t offset = (uint64_t{3} << 62) | (doorbell_gpu << 46);
      char line[128];
      snprintf(line, sizeof(line),
               "7f0000000000-7f0000002000 rw-s %" PRIx64 " %02x:%02x %" PRIu64 " /dev/kfd\n",
               offset, files_.kfd.fs_major, files_.kfd.fs_minor, files_.kfd.inode);
      maps += line;
    }
    proc_.WriteFile(dir + "/maps", maps);
    kfd_.WriteFile(dir + "/vram_" + std::to_string(kKfdGpuId), "7\n");
  }

  // Makes <proc>/<entry> a link to `target`, in place of what is there.
  void LinkEntry(const std::string& entry, const std::string& target) {
    std::filesystem::remove_all(proc_.root() + "/" + entry);
    std::filesystem::create_symlink(target, proc_.root() + "/" + entry);
  }

 private:
  GpuFiles files_;
  FakeSysfsTree proc_;
  FakeSysfsTree kfd_;
  FakeSysfsTree sysfs_;
  FakeSysfsTree topology_;  // no KFD nodes, so that no list reads the host's
  ScopedRocmSmiDevice rocm_device_;
  ScopedRocmSmiDevice other_rocm_device_;
  ScopedKfdNode kfd_node_;
  ScopedKfdNode other_kfd_node_;
  AMDSmiDrm drm_;
  AMDSmiGPUDevice gpu_device_;
  ScopedProcessorRegistration registration_;
  ScopedOverride<std::string> amdsmi_proc_root_;
  ScopedOverride<std::string> topology_root_;
};

// The processes of the mock namespace.
constexpr uint32_t kQueueAndNode = 4301;  // a queue on the GPU and its render node
constexpr uint32_t kNodeOnly = 4302;      // the GPU's render node, no queue
constexpr uint32_t kQueueOnly = 4303;     // a queue on the GPU, no render node
constexpr uint32_t kUnknownQueue = 4304;  // a queue on a GPU that is not here
constexpr uint32_t kKfdOnly = 4305;       // /dev/kfd and nothing else
constexpr uint32_t kNoKfd = 4306;         // the GPU's render node, but not /dev/kfd

void AddProcesses(MockPidNamespace* ns) {
  ns->AddProcess(kQueueAndNode, true, Node::kThisGpu, kKfdGpuId, 1024);
  ns->AddProcess(kNodeOnly, true, Node::kThisGpu, 0, 2048);
  ns->AddProcess(kQueueOnly, true, Node::kNone, kKfdGpuId);
  ns->AddProcess(kUnknownQueue, true, Node::kNone, kUnknownKfdGpuId);
  ns->AddProcess(kKfdOnly, true, Node::kNone, 0);
  ns->AddProcess(kNoKfd, false, Node::kThisGpu, 0, 512);
}

TEST(GpuUnit, PidNamespaceKfdProcessesAreThoseHoldingDevKfd) {
  const auto files = FindGpuFiles();
  if (!files) GTEST_SKIP() << "No character devices to stand for /dev/kfd and render nodes";
  ScopedAmdsmiInit smi_init;
  MockPidNamespace ns(*files);
  AddProcesses(&ns);

  uint32_t num_items = 0;
  ASSERT_EQ(amdsmi_get_gpu_compute_process_info(nullptr, &num_items), AMDSMI_STATUS_SUCCESS);
  std::vector<amdsmi_process_info_t> procs(num_items);
  ASSERT_EQ(amdsmi_get_gpu_compute_process_info(procs.data(), &num_items), AMDSMI_STATUS_SUCCESS);
  std::vector<uint32_t> pids;
  for (uint32_t i = 0; i < num_items; ++i) {
    pids.push_back(procs[i].process_id);
    // Not the 7 bytes in KFD's directory of the same PID, another process's.
    EXPECT_EQ(procs[i].vram_usage, kNotAvailable64);
    EXPECT_EQ(procs[i].sdma_usage, kNotAvailable64);
    EXPECT_EQ(procs[i].cu_occupancy, kNotAvailable32);
    EXPECT_EQ(procs[i].evicted_time, kNotAvailable32);
  }
  std::sort(pids.begin(), pids.end());
  EXPECT_EQ(pids,
            (std::vector<uint32_t>{kQueueAndNode, kNodeOnly, kQueueOnly, kUnknownQueue, kKfdOnly}));

  amdsmi_process_info_t info{};
  ASSERT_EQ(amdsmi_get_gpu_compute_process_info_by_pid(kQueueAndNode, &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.process_id, kQueueAndNode);
  EXPECT_EQ(info.vram_usage, kNotAvailable64);
  EXPECT_EQ(amdsmi_get_gpu_compute_process_info_by_pid(kNoKfd, &info), AMDSMI_STATUS_NOT_FOUND);
}

TEST(GpuUnit, PidNamespaceProcessGpusComeFromQueuesThenRenderNodes) {
  const auto files = FindGpuFiles();
  if (!files) GTEST_SKIP() << "No character devices to stand for /dev/kfd and render nodes";
  ScopedAmdsmiInit smi_init;
  MockPidNamespace ns(*files);
  AddProcesses(&ns);

  // The GPU indices amdsmi_get_gpu_compute_process_gpus() reports for `pid`.
  auto gpus_of = [](uint32_t pid) {
    std::vector<uint32_t> indices(8);
    auto count = static_cast<uint32_t>(indices.size());
    EXPECT_EQ(amdsmi_get_gpu_compute_process_gpus(pid, indices.data(), &count),
              AMDSMI_STATUS_SUCCESS)
        << "pid " << pid;
    indices.resize(std::min<size_t>(count, indices.size()));
    return indices;
  };
  const std::vector<uint32_t> this_gpu{ns.index()};
  EXPECT_EQ(gpus_of(kQueueAndNode), this_gpu);
  EXPECT_EQ(gpus_of(kQueueOnly), this_gpu);
  EXPECT_EQ(gpus_of(kNodeOnly), this_gpu);
  // A queue GPU id that names no GPU here places a process nowhere.
  EXPECT_TRUE(gpus_of(kUnknownQueue).empty());
  EXPECT_TRUE(gpus_of(kKfdOnly).empty());
  EXPECT_TRUE(gpus_of(kNoKfd).empty());

  // The other GPU's render node places a process there; a queue on this GPU
  // outweighs it; a queue GPU id that names no GPU here does not.
  constexpr uint32_t kNodeThere = 4307;
  constexpr uint32_t kQueueHereNodeThere = 4308;
  constexpr uint32_t kUnknownQueueNodeHere = 4309;
  ns.AddProcess(kNodeThere, true, Node::kOtherGpu, 0);
  ns.AddProcess(kQueueHereNodeThere, true, Node::kOtherGpu, kKfdGpuId);
  ns.AddProcess(kUnknownQueueNodeHere, true, Node::kThisGpu, kUnknownKfdGpuId);
  EXPECT_EQ(gpus_of(kNodeThere), std::vector<uint32_t>{ns.other_index()});
  EXPECT_EQ(gpus_of(kQueueHereNodeThere), this_gpu);
  EXPECT_EQ(gpus_of(kUnknownQueueNodeHere), this_gpu);
}

TEST(GpuUnit, PidNamespaceProcessListsHoldTheGpuProcesses) {
  const auto files = FindGpuFiles();
  if (!files) GTEST_SKIP() << "No character devices to stand for /dev/kfd and render nodes";
  ScopedAmdsmiInit smi_init;
  MockPidNamespace ns(*files);
  AddProcesses(&ns);
  WaitOutProcessListCache();

  uint32_t num_processes = 8;
  amdsmi_proc_info_t list[8] = {};
  ASSERT_EQ(amdsmi_get_gpu_process_list(ns.handle(), &num_processes, list), AMDSMI_STATUS_SUCCESS);
  std::map<uint32_t, amdsmi_proc_info_t> listed;
  for (uint32_t i = 0; i < num_processes; ++i) listed[list[i].pid] = list[i];
  ASSERT_EQ(listed.size(), 3U);
  ASSERT_EQ(listed.count(kQueueAndNode), 1U);
  ASSERT_EQ(listed.count(kNodeOnly), 1U);
  ASSERT_EQ(listed.count(kQueueOnly), 1U);
  constexpr uint64_t kGtt = 2 * 1024 * 1024;
  constexpr uint64_t kCpu = 3 * 1024;
  const amdsmi_proc_info_t& with_queue = listed[kQueueAndNode];
  EXPECT_STREQ(with_queue.name, "/usr/bin/fake_namespace_app");
  EXPECT_EQ(with_queue.memory_usage.vram_mem, 1024U * 1024U);
  EXPECT_EQ(with_queue.memory_usage.gtt_mem, kGtt);
  EXPECT_EQ(with_queue.memory_usage.cpu_mem, kCpu);
  EXPECT_EQ(with_queue.mem, 1024U * 1024U + kGtt + kCpu);
  EXPECT_EQ(with_queue.cu_occupancy, kNotAvailable32);
  EXPECT_EQ(with_queue.evicted_time, kNotAvailable32);
  EXPECT_EQ(with_queue.sdma_usage, kNotAvailable64);
  EXPECT_EQ(listed[kNodeOnly].mem, 2048U * 1024U + kGtt + kCpu);
  // Without the render node it has no DRM client, so no memory to report.
  EXPECT_EQ(listed[kQueueOnly].mem, 0U);

  amdsmi_processor_handle handle = ns.handle();
  uint32_t num_pids = 8;
  amdsmi_proc_info_by_pid_t merged[8] = {};
  ASSERT_EQ(amdsmi_get_gpu_process_list_by_pid(&handle, 1, merged, &num_pids),
            AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_pids, 3U);
  for (uint32_t i = 0; i < num_pids; ++i) {
    ASSERT_EQ(listed.count(merged[i].pid), 1U) << "pid " << merged[i].pid;
    ASSERT_EQ(merged[i].num_gpus, 1U);
    EXPECT_EQ(merged[i].gpus[0].gpu_index, ns.index());
    EXPECT_EQ(merged[i].gpus[0].mem, listed[merged[i].pid].mem);
  }
}

// A link that open() fails with EPERM, as /proc mounted with hidepid=noaccess
// does for another user's process: this process's /proc/self/map_files entry
// for a page it maps from a file, which only a caller with CAP_SYS_ADMIN or
// CAP_CHECKPOINT_RESTORE may follow.
class EpermLink {
 public:
  EpermLink() : file_(tmpfile()), page_size_(static_cast<size_t>(sysconf(_SC_PAGESIZE))) {
    if (file_ == nullptr || ftruncate(fileno(file_), static_cast<off_t>(page_size_)) != 0) return;
    page_ = mmap(nullptr, page_size_, PROT_READ, MAP_SHARED, fileno(file_), 0);
    if (page_ == MAP_FAILED) return;
    const auto start = reinterpret_cast<uintptr_t>(page_);
    char path[64];
    snprintf(path, sizeof(path), "/proc/self/map_files/%" PRIxPTR "-%" PRIxPTR, start,
             start + page_size_);
    path_ = path;
  }
  ~EpermLink() {
    if (page_ != MAP_FAILED) munmap(page_, page_size_);
    if (file_ != nullptr) fclose(file_);
  }

  EpermLink(const EpermLink&) = delete;
  EpermLink& operator=(const EpermLink&) = delete;

  // Empty if the page could not be mapped.
  const std::string& path() const { return path_; }

  // Whether open() of the link fails with EPERM for this caller.
  bool DeniesThisCaller() const {
    const int fd = open(path_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) close(fd);
    return fd < 0 && errno == EPERM;
  }

 private:
  FILE* file_;
  size_t page_size_;
  void* page_ = MAP_FAILED;
  std::string path_;
};

TEST(GpuUnit, PidNamespaceListsLeaveOutAProcessThatHidepidDenies) {
  const auto files = FindGpuFiles();
  if (!files) GTEST_SKIP() << "No character devices to stand for /dev/kfd and render nodes";
  EpermLink denied;
  ASSERT_FALSE(denied.path().empty()) << "Mapping a page of a temporary file failed";
  ScopedAmdsmiInit smi_init;
  MockPidNamespace ns(*files);
  AddProcesses(&ns);
  // Opening /proc/<pid> fails with EPERM for the first process, opening its fd
  // directory for the second, and opening its maps for the third, which has
  // /dev/kfd open and a queue on the GPU.
  constexpr uint32_t kDenied = 4310;
  constexpr uint32_t kFdDenied = 4311;
  constexpr uint32_t kMapsDenied = 4312;
  ns.LinkEntry(std::to_string(kDenied), denied.path());
  ns.AddProcess(kFdDenied, true, Node::kThisGpu, kKfdGpuId);
  ns.LinkEntry(std::to_string(kFdDenied) + "/fd", denied.path());
  ns.AddProcess(kMapsDenied, true, Node::kNone, kKfdGpuId);
  ns.LinkEntry(std::to_string(kMapsDenied) + "/maps", denied.path());
  WaitOutProcessListCache();

  // A caller with CAP_SYS_ADMIN or CAP_CHECKPOINT_RESTORE, such as root, may
  // open the link, so the lists are read as an unprivileged user.
  // NOT_SUPPORTED stands for a caller the link does not deny.
  auto read_lists = [&] {
    if (!denied.DeniesThisCaller()) return AMDSMI_STATUS_NOT_SUPPORTED;
    // The lists leave out the first two rather than fail. The third is a KFD
    // process, and with its queues unreadable and no render node open, no GPU
    // lists it.
    uint32_t num_items = 0;
    EXPECT_EQ(amdsmi_get_gpu_compute_process_info(nullptr, &num_items), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(num_items, 6U);
    uint32_t num_processes = 8;
    amdsmi_proc_info_t list[8] = {};
    EXPECT_EQ(amdsmi_get_gpu_process_list(ns.handle(), &num_processes, list),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(num_processes, 3U);
    // A lookup of each reports that the caller may not read it.
    for (const uint32_t pid : {kDenied, kFdDenied, kMapsDenied}) {
      uint32_t gpu_count = 0;
      EXPECT_EQ(amdsmi_get_gpu_compute_process_gpus(pid, nullptr, &gpu_count),
                AMDSMI_STATUS_NO_PERM)
          << "pid " << pid;
    }
    amdsmi_process_info_t info{};
    EXPECT_EQ(amdsmi_get_gpu_compute_process_info_by_pid(kDenied, &info), AMDSMI_STATUS_NO_PERM);
    return AMDSMI_STATUS_SUCCESS;
  };
  const amdsmi_status_t status = amd::smi::testing::RunAsUnprivilegedUser(read_lists);
  if (status == AMDSMI_STATUS_NOT_SUPPORTED) {
    GTEST_SKIP() << "Opening a /proc/self/map_files link does not fail with EPERM for an "
                    "unprivileged caller";
  }
  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
}

}  // namespace
