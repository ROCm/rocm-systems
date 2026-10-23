// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/kfd_open_file.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"

#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace rocjitsu {

KfdOpenFile::KfdOpenFile(SimulatedKfd &driver, uint32_t primary_process_id)
    : driver_(driver), primary_process_id_(primary_process_id), process_id_(primary_process_id) {}

KfdOpenFile::~KfdOpenFile() {
  if (process_id_ != primary_process_id_)
    driver_.close(process_id_);
}

int KfdOpenFile::create_process() {
  std::lock_guard lock(mutex_);
  if (process_id_ != primary_process_id_)
    return -EINVAL;
  const uint32_t process_id = driver_.open_secondary_process();
  if (process_id == 0)
    return -errno;
  process_id_ = process_id;
  return 0;
}

uint32_t KfdOpenFile::process_id() const {
  std::lock_guard lock(mutex_);
  return process_id_;
}

uint32_t SimulatedKfd::open_secondary_process() {
  std::lock_guard lock(process_mutex_);
  const uint32_t process_id = next_process_id_++;
  auto process =
      std::make_shared<KfdProcess>(process_id, static_cast<uint32_t>(gpus_.size()), true);
  process->set_client_pid(static_cast<pid_t>(getpid()));
  process->event_state_.reset();
  if (!register_process_address_spaces(process, 0, true)) {
    errno = EIO;
    return 0;
  }
  processes_.emplace(process_id, process);
  {
    std::lock_guard interrupt_lock(interrupt_mutex_);
    event_dispatch_[process_id] = &process->event_state_;
  }
  return process_id;
}

int SimulatedKfd::open_file() {
  const int primary_fd = open();
  if (primary_fd < 0)
    return -1;
  const int application_fd =
      (libc_passthrough().fcntl ? libc_passthrough().fcntl(primary_fd, F_DUPFD_CLOEXEC, 0)
                                : ::fcntl(primary_fd, F_DUPFD_CLOEXEC, 0));
  if (application_fd < 0) {
    const int saved_errno = errno;
    close();
    errno = saved_errno;
    return -1;
  }
  track_local_file(application_fd, std::make_shared<KfdOpenFile>(*this, local_process_id()));
  return application_fd;
}

std::shared_ptr<LinuxKfd::OpenFile> SimulatedKfd::local_file(int descriptor) {
  std::lock_guard lock(local_files_mutex_);
  auto it = local_files_.find(descriptor);
  return it == local_files_.end() ? nullptr : it->second;
}

void SimulatedKfd::track_local_file(int descriptor, std::shared_ptr<OpenFile> file) {
  std::lock_guard lock(local_files_mutex_);
  local_files_.emplace(descriptor, std::move(file));
}

void SimulatedKfd::untrack_local_file(int descriptor) {
  std::shared_ptr<OpenFile> file;
  {
    std::lock_guard lock(local_files_mutex_);
    auto node = local_files_.extract(descriptor);
    if (!node.empty())
      file = std::move(node.mapped());
  }
  // Final-close teardown enters process/queue locks, outside the file-table lock.
}

uint32_t SimulatedKfd::process_id_for_file(int descriptor) {
  const auto file = std::static_pointer_cast<KfdOpenFile>(local_file(descriptor));
  return file ? file->process_id() : (descriptor == fd() ? local_process_id() : 0);
}

int SimulatedKfd::ioctl_file(int descriptor, unsigned long request, void *arg) {
  const auto file = std::static_pointer_cast<KfdOpenFile>(local_file(descriptor));
  if (canonical_ioctl_request(request) == AMDKFD_IOC_CREATE_PROCESS)
    return file && arg ? file->create_process() : -EINVAL;
  if (!file && descriptor != fd())
    return -EBADF;
  const int result = ioctl(file ? file->process_id() : local_process_id(), request, arg);
  if (result == 0 && file && canonical_ioctl_request(request) == AMDKFD_IOC_GET_VERSION)
    static_cast<kfd_ioctl_get_version_args *>(arg)->minor_version = KFD_IOCTL_MINOR_VERSION;
  return result;
}

void *SimulatedKfd::mmap_file(int descriptor, void *addr, size_t length, int prot, int flags,
                              off_t offset) {
  const auto file = std::static_pointer_cast<KfdOpenFile>(local_file(descriptor));
  if (!file && descriptor != fd()) {
    errno = EBADF;
    return MAP_FAILED;
  }
  return mmap(file ? file->process_id() : local_process_id(), addr, length, prot, flags, offset);
}

} // namespace rocjitsu
