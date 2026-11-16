// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "test_fixture_utils.h"

#include <gtest/gtest.h>
#include <linux/limits.h>
#include <sys/inotify.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "amd_smi/impl/amd_smi_processor.h"
#include "amd_smi/impl/amd_smi_system.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_main.h"

namespace fs = std::filesystem;

// The kernel numbers render nodes from 128 (drivers/gpu/drm/drm_drv.c).
constexpr uint32_t kFirstRenderMinor = 128;

FakeSysfsTree::FakeSysfsTree() {
  std::string tmpl = "/tmp/amdsmi_fake_sysfs_XXXXXX";
  const char* dir = mkdtemp(tmpl.data());
  if (dir == nullptr) {
    ADD_FAILURE() << "FakeSysfsTree: mkdtemp() failed";
    return;
  }
  root_ = dir;
  // mkdtemp() makes the root owner-only; an unprivileged caller (RunAsUnprivilegedUser) must still
  // traverse it to reach the files.
  std::error_code ec;
  fs::permissions(root_,
                  fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec |
                      fs::perms::others_read | fs::perms::others_exec,
                  ec);
  if (ec) {
    ADD_FAILURE() << "FakeSysfsTree: making the root traversable failed: " << ec.message();
  }
}

FakeSysfsTree::~FakeSysfsTree() {
  if (root_.empty()) return;
  std::error_code ec;
  fs::remove_all(root_, ec);
}

void FakeSysfsTree::WriteFile(const std::string& relative_path, const std::string& content) {
  if (root_.empty()) return;  // mkdtemp() already failed; ADD_FAILURE()'d, avoid touching cwd.
  fs::path full_path = fs::path(root_) / relative_path;
  std::error_code ec;
  fs::create_directories(full_path.parent_path(), ec);
  if (ec) {
    ADD_FAILURE() << "FakeSysfsTree::WriteFile(" << relative_path
                  << "): create_directories() failed: " << ec.message();
    return;
  }
  std::ofstream out(full_path);
  out << content;
  out.close();
  if (!out) {
    ADD_FAILURE() << "FakeSysfsTree::WriteFile(" << relative_path << "): write failed";
  }
}

std::string FakeSysfsTree::ReadFile(const std::string& relative_path) const {
  std::ifstream in(fs::path(root_) / relative_path);
  std::string line;
  std::getline(in, line);
  return line;
}

void FakeSysfsTree::RemoveFile(const std::string& relative_path) {
  if (root_.empty()) return;
  std::error_code ec;
  fs::remove(fs::path(root_) / relative_path, ec);
}

void FakeSysfsTree::SetPermissions(const std::string& relative_path, mode_t mode) {
  if (root_.empty()) return;
  if (chmod((fs::path(root_) / relative_path).c_str(), mode) != 0) {
    ADD_FAILURE() << "FakeSysfsTree::SetPermissions(" << relative_path << ", " << mode
                  << "): chmod() failed: " << strerror(errno);
  }
}

namespace amd::smi::testing {

ScopedEnvVar::ScopedEnvVar(const char* name, const char* value) : name_(name) {
  if (const char* old = std::getenv(name)) old_value_ = old;
  setenv(name, value, /*overwrite=*/1);
}

ScopedEnvVar::~ScopedEnvVar() {
  if (old_value_) {
    setenv(name_.c_str(), old_value_->c_str(), /*overwrite=*/1);
  } else {
    unsetenv(name_.c_str());
  }
}

ScopedProcessorRegistration::ScopedProcessorRegistration(amd::smi::AMDSmiProcessor* processor)
    : processor_(processor) {
  amd::smi::AMDSmiSystem::getInstance().register_processor_for_testing(processor_);
}

ScopedProcessorRegistration::~ScopedProcessorRegistration() {
  amd::smi::AMDSmiSystem::getInstance().unregister_processor_for_testing(processor_);
}

amdsmi_processor_handle ScopedProcessorRegistration::handle() const {
  return reinterpret_cast<amdsmi_processor_handle>(processor_);
}

ScopedFileWatch::ScopedFileWatch(const std::string& path, uint32_t event_mask) {
  inotify_fd_ = inotify_init1(IN_NONBLOCK);
  if (inotify_fd_ < 0) {
    return;  // IsAvailable() reports false; nothing else to clean up.
  }
  watch_fd_ = inotify_add_watch(inotify_fd_, path.c_str(), event_mask);
  if (watch_fd_ < 0) {
    close(inotify_fd_);
    inotify_fd_ = -1;
    return;
  }
}

ScopedFileWatch::~ScopedFileWatch() {
  if (watch_fd_ >= 0) {
    inotify_rm_watch(inotify_fd_, watch_fd_);
  }
  if (inotify_fd_ >= 0) {
    close(inotify_fd_);
  }
}

void ScopedFileWatch::Drain() {
  if (inotify_fd_ < 0) return;
  // Per man 7 inotify: struct inotify_event has a variable-length trailing
  // name, so the buffer must hold several whole entries, properly aligned.
  constexpr std::size_t kMaxBufferedEvents = 16;
  constexpr std::size_t kBufferSize =
      kMaxBufferedEvents * (sizeof(struct inotify_event) + NAME_MAX + 1);
  alignas(struct inotify_event) std::array<char, kBufferSize> buf;
  for (;;) {
    ssize_t len = read(inotify_fd_, buf.data(), buf.size());
    if (len < 0 && errno == EINTR) {
      continue;
    }
    if (len <= 0) {
      // len < 0 with EAGAIN/EWOULDBLOCK just means no more events are
      // pending right now -- not an error worth distinguishing here.
      break;
    }
    for (char* ptr = buf.data(); ptr < buf.data() + len;) {
      auto* event = reinterpret_cast<struct inotify_event*>(ptr);
      for (std::size_t bit = 0; bit < kBitsPerMask; ++bit) {
        if (event->mask & (1u << bit)) ++event_tally_[bit];
      }
      ptr += sizeof(struct inotify_event) + event->len;
    }
  }
}

uint32_t ScopedFileWatch::Count(uint32_t mask) {
  Drain();
  uint32_t total = 0;
  for (std::size_t bit = 0; bit < kBitsPerMask; ++bit) {
    if (mask & (1u << bit)) total += event_tally_[bit];
  }
  return total;
}

uint32_t ScopedFileWatch::OpenCount() { return Count(IN_OPEN); }

uint32_t ScopedFileWatch::WriteCount() { return Count(IN_CLOSE_WRITE); }

amdsmi_status_t RunAsUnprivilegedUser(const std::function<amdsmi_status_t()>& fn) {
  if (geteuid() != 0) return fn();

  constexpr uid_t kNobody = 65534;
  int fds[2];
  if (pipe(fds) != 0) {
    ADD_FAILURE() << "RunAsUnprivilegedUser: pipe() failed: " << strerror(errno);
    return AMDSMI_STATUS_UNKNOWN_ERROR;
  }
  const pid_t child = fork();
  if (child < 0) {
    ADD_FAILURE() << "RunAsUnprivilegedUser: fork() failed: " << strerror(errno);
    close(fds[0]);
    close(fds[1]);
    return AMDSMI_STATUS_UNKNOWN_ERROR;
  }
  if (child == 0) {
    close(fds[0]);
    amdsmi_status_t status = AMDSMI_STATUS_UNKNOWN_ERROR;
    // gid first: after setuid() the process can no longer change its gid. gtest state does not
    // cross fork(), so a failure here is only visible through stderr and the returned status.
    if (setgid(kNobody) != 0 || setuid(kNobody) != 0) {
      std::cerr << "RunAsUnprivilegedUser: dropping to uid " << kNobody
                << " failed: " << strerror(errno) << std::endl;
    } else {
      status = fn();
      // A failure recorded in the child never reaches the parent; send it as a bad status.
      if (::testing::Test::HasFailure()) status = AMDSMI_STATUS_UNKNOWN_ERROR;
    }
    const ssize_t written = write(fds[1], &status, sizeof(status));
    static_cast<void>(written);
    _exit(0);
  }
  close(fds[1]);
  amdsmi_status_t status = AMDSMI_STATUS_UNKNOWN_ERROR;
  const ssize_t got = read(fds[0], &status, sizeof(status));
  close(fds[0]);
  waitpid(child, nullptr, 0);
  if (got != static_cast<ssize_t>(sizeof(status))) {
    ADD_FAILURE() << "RunAsUnprivilegedUser: the child did not report a status";
    return AMDSMI_STATUS_UNKNOWN_ERROR;
  }
  return status;
}

}  // namespace amd::smi::testing

ScopedRocmSmiDevice::ScopedRocmSmiDevice(const std::string& path) {
  auto& devices = amd::smi::RocmSMI::getInstance().devices();
  index_ = static_cast<uint32_t>(devices.size());
  devices.push_back(std::make_shared<amd::smi::Device>(path, nullptr));
  devices.back()->set_drm_render_minor(render_minor());

  const fs::path card_dir(path);
  std::error_code ec;
  fs::create_directory_symlink(card_dir.filename(), RenderLink(card_dir.parent_path().string()),
                               ec);
  if (ec) {
    ADD_FAILURE() << "ScopedRocmSmiDevice(" << path
                  << "): renderD symlink failed: " << ec.message();
  } else {
    link_dir_ = card_dir.parent_path().string();
  }
}

uint32_t ScopedRocmSmiDevice::render_minor() const { return kFirstRenderMinor + index_; }

ScopedRocmSmiDevice::~ScopedRocmSmiDevice() {
  if (!link_dir_.empty()) {
    std::error_code ec;
    fs::remove(RenderLink(link_dir_), ec);
  }
  auto& devices = amd::smi::RocmSMI::getInstance().devices();
  if (index_ < devices.size()) devices.resize(index_);
}

ScopedAmdsmiInit::ScopedAmdsmiInit() {
  EXPECT_EQ(amdsmi_init(kNoHardwareDiscoveryInitFlags), AMDSMI_STATUS_SUCCESS);
}

ScopedAmdsmiInit::~ScopedAmdsmiInit() { amdsmi_shut_down(); }

namespace {

// A gpu_id of 0 marks a CPU node in KFD's topology.
constexpr uint64_t kCpuNodeGpuId = 0;

// A KFD location_id packs the PCI bus above an 8-bit device and function.
constexpr uint32_t kBusShift = 8;

// /sys/class/kfd/kfd/topology/nodes/<node_idx>/gpu_id
void WriteGpuId(FakeSysfsTree& tree, uint32_t node_idx, uint64_t gpu_id) {
  tree.WriteFile(std::to_string(node_idx) + "/gpu_id", std::to_string(gpu_id) + "\n");
}

}  // namespace

// The DRM entries pair the node with its card: renderD<minor> holds the card and render node it
// belongs to.
void WriteGpuNode(FakeSysfsTree& tree, uint32_t node_idx, uint32_t card_idx, uint64_t gpu_id) {
  const std::string render_minor = std::to_string(kFirstRenderMinor + card_idx);
  const std::string drm_dir = "renderD" + render_minor + "/device/drm/";
  WriteGpuId(tree, node_idx, gpu_id);
  tree.WriteFile(std::to_string(node_idx) + "/properties",
                 "hive_id 0\nsimd_arrays_per_engine 1\narray_count 1\ncu_per_simd_array 1\n"
                 "location_id " +
                     std::to_string((node_idx + 1) << kBusShift) + "\ndomain 0\ndrm_render_minor " +
                     render_minor + "\n");
  tree.WriteFile(std::to_string(node_idx) + "/io_links/.keep", "");
  tree.WriteFile(drm_dir + "card" + std::to_string(card_idx) + "/.keep", "");
  tree.WriteFile(drm_dir + "renderD" + render_minor + "/.keep", "");
  std::error_code ec;
  fs::create_directory_symlink("renderD" + render_minor,
                               fs::path(tree.root()) / ("card" + std::to_string(card_idx)), ec);
  if (ec)
    ADD_FAILURE() << "WriteGpuNode(" << card_idx << "): card symlink failed: " << ec.message();
}

void WriteCpuNode(FakeSysfsTree& tree, uint32_t node_idx) {
  WriteGpuId(tree, node_idx, kCpuNodeGpuId);
  tree.WriteFile(std::to_string(node_idx) + "/io_links/.keep", "");
}

void WriteIoLink(FakeSysfsTree& tree, uint32_t node_idx, uint32_t link_idx, uint32_t node_to,
                 amd::smi::IO_LINK_TYPE type, uint64_t weight) {
  std::ostringstream props;
  props << "type " << static_cast<uint64_t>(type) << "\n"
        << "node_from " << node_idx << "\n"
        << "node_to " << node_to << "\n"
        << "weight " << weight << "\n"
        << "flags 1\n"
        << "min_bandwidth 100\n"
        << "max_bandwidth 200\n";
  tree.WriteFile(std::to_string(node_idx) + "/io_links/" + std::to_string(link_idx) + "/properties",
                 props.str());
}

FakeKfdTopology::FakeKfdTopology(FakeSysfsTree& tree)
    : kfd_root_guard_(amd::smi::test_hooks::KfdNodesPathRoot,
                      amd::smi::test_hooks::SetKfdNodesPathRoot, tree.root()),
      drm_root_guard_(amd::smi::test_hooks::SysfsDrmRoot, amd::smi::test_hooks::SetSysfsDrmRoot,
                      tree.root()),
      tmp_dir_guard_(amd::smi::test_hooks::AppTmpDir, amd::smi::test_hooks::SetAppTmpDir,
                     tree.root()) {
  const amdsmi_status_t init_status = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  if (init_status != AMDSMI_STATUS_SUCCESS) {
    throw std::runtime_error("amdsmi_init() failed with status " + std::to_string(init_status));
  }

  uint32_t socket_count = 0;
  EXPECT_EQ(amdsmi_get_socket_handles(&socket_count, nullptr), AMDSMI_STATUS_SUCCESS);
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  EXPECT_EQ(amdsmi_get_socket_handles(&socket_count, sockets.data()), AMDSMI_STATUS_SUCCESS);
  for (auto socket : sockets) {
    uint32_t processor_count = 0;
    EXPECT_EQ(amdsmi_get_processor_handles(socket, &processor_count, nullptr),
              AMDSMI_STATUS_SUCCESS);
    std::vector<amdsmi_processor_handle> processors(processor_count);
    EXPECT_EQ(amdsmi_get_processor_handles(socket, &processor_count, processors.data()),
              AMDSMI_STATUS_SUCCESS);
    for (auto processor : processors) {
      amdsmi_kfd_info_t info{};
      EXPECT_EQ(amdsmi_get_gpu_kfd_info(processor, &info), AMDSMI_STATUS_SUCCESS);
      handles_[info.node_id] = processor;
    }
  }
}

FakeKfdTopology::~FakeKfdTopology() { amdsmi_shut_down(); }
