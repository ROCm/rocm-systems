// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "test_fixture_utils.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <linux/limits.h>
#include <sys/inotify.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <system_error>

#include "amd_smi/impl/amd_smi_processor.h"
#include "amd_smi/impl/amd_smi_system.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_main.h"

namespace fs = std::filesystem;

FakeSysfsTree::FakeSysfsTree() {
  std::string tmpl = "/tmp/amdsmi_fake_sysfs_XXXXXX";
  const char* dir = mkdtemp(tmpl.data());
  if (dir == nullptr) {
    ADD_FAILURE() << "FakeSysfsTree: mkdtemp() failed";
    return;
  }
  root_ = dir;
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

ScopedFakeIoctl* ScopedFakeIoctl::active_ = nullptr;

ScopedFakeIoctl::ScopedFakeIoctl() : previous_active_(active_) { active_ = this; }

ScopedFakeIoctl::~ScopedFakeIoctl() { active_ = previous_active_; }

void ScopedFakeIoctl::RegisterHandler(unsigned long request, Handler handler) {
  handlers_[request] = std::move(handler);
}

int ScopedFakeIoctl::CallCount(unsigned long request) const {
  const auto call = calls_.find(request);
  return call == calls_.end() ? 0 : call->second;
}

int ScopedFakeIoctl::Dispatch(unsigned long request, void* arg) {
  for (ScopedFakeIoctl* fake = active_; fake != nullptr; fake = fake->previous_active_) {
    const auto handler = fake->handlers_.find(request);
    if (handler != fake->handlers_.end()) {
      ++fake->calls_[request];
      return handler->second(arg);
    }
  }
  errno = ENOTTY;
  return -1;
}

}  // namespace amd::smi::testing

ScopedRocmSmiDevice::ScopedRocmSmiDevice(const std::string& path) {
  auto& devices = amd::smi::RocmSMI::getInstance().devices();
  index_ = static_cast<uint32_t>(devices.size());
  devices.push_back(std::make_shared<amd::smi::Device>(path, nullptr));
}

ScopedRocmSmiDevice::~ScopedRocmSmiDevice() {
  auto& devices = amd::smi::RocmSMI::getInstance().devices();
  if (index_ < devices.size()) devices.resize(index_);
}

ScopedAmdsmiInit::ScopedAmdsmiInit() {
  EXPECT_EQ(amdsmi_init(kNoHardwareDiscoveryInitFlags), AMDSMI_STATUS_SUCCESS);
}

ScopedAmdsmiInit::~ScopedAmdsmiInit() { amdsmi_shut_down(); }

ScopedFakeKfdEventStream::ScopedFakeKfdEventStream(uint32_t rocm_smi_device_index)
    : device_index_(rocm_smi_device_index) {
  // Installed first so that no failure below can leave the library using the real /dev/kfd.
  kfd_tree_.WriteFile("kfd", "");
  path_override_.emplace(amd::smi::test_hooks::KfdDevicePath,
                         amd::smi::test_hooks::SetKfdDevicePath, kfd_tree_.root() + "/kfd");
  ioctl_.RegisterHandler<kfd_ioctl_get_version_args>(AMDKFD_IOC_GET_VERSION, [](auto& args) {
    args.major_version = KFD_IOCTL_MAJOR_VERSION;
    args.minor_version = KFD_IOCTL_MINOR_VERSION;
    return 0;
  });
  ioctl_.RegisterHandler<kfd_ioctl_smi_events_args>(AMDKFD_IOC_SMI_EVENTS, [this](auto& args) {
    if (device_fd_ < 0) {
      errno = EIO;
      return -1;
    }
    // The library now owns this end and closes it when notification stops.
    args.anon_fd = static_cast<uint32_t>(device_fd_);
    device_fd_ = -1;
    return 0;
  });
  int fds[2];
  // A socket pair, not a pipe: the library both reads events from and writes the mask to its fd.
  // Non-blocking so its fgets() loop ends with EAGAIN once the queued lines are read.
  if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0, fds) != 0) {
    ADD_FAILURE() << "ScopedFakeKfdEventStream: socketpair() failed: " << strerror(errno);
    return;
  }
  device_fd_ = fds[0];
  stream_fd_ = fds[1];
}

ScopedFakeKfdEventStream::~ScopedFakeKfdEventStream() {
  // A test that failed before calling stop leaves the library holding the stream; release it so
  // the next test starts clean. The library closes the fd it was given.
  auto& devices = amd::smi::RocmSMI::getInstance().devices();
  if (device_index_ < devices.size() && devices[device_index_]->evt_notif_anon_fd() != -1) {
    rsmi_event_notification_stop(device_index_);
  }
  if (device_fd_ >= 0) close(device_fd_);  // never handed to the library
  if (stream_fd_ >= 0) close(stream_fd_);
}

void ScopedFakeKfdEventStream::Write(uint32_t event, const std::string& payload) {
  if (stream_fd_ < 0) return;  // construction already reported the failure

  // The library writes each mask as one uint64_t; the newest one wins, as in the kernel.
  uint64_t mask;
  while (read(stream_fd_, &mask, sizeof(mask)) == static_cast<ssize_t>(sizeof(mask))) {
    mask_ = mask;
  }
  constexpr uint32_t kMaskBits = std::numeric_limits<uint64_t>::digits;
  const bool has_mask_bit = event >= 1 && event <= kMaskBits;
  if (has_mask_bit && (mask_ & (1ULL << (event - 1))) == 0) return;

  std::ostringstream line;
  line << std::hex << event << ' ' << payload << '\n';
  const std::string text = line.str();
  if (write(stream_fd_, text.data(), text.size()) != static_cast<ssize_t>(text.size())) {
    ADD_FAILURE() << "ScopedFakeKfdEventStream::Write(): write() failed: " << strerror(errno);
  }
}

extern "C" int __real_ioctl(int fd, unsigned long request, ...);

extern "C" int __wrap_ioctl(int fd, unsigned long request, ...) {
  va_list args;
  va_start(args, request);
  void* arg = va_arg(args, void*);
  va_end(args);
  if (amd::smi::testing::ScopedFakeIoctl::active_ == nullptr) return __real_ioctl(fd, request, arg);
  return amd::smi::testing::ScopedFakeIoctl::Dispatch(request, arg);
}
