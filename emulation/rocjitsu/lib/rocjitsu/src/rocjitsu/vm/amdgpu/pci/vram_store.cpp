// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/vram_store.h"

#include "util/log.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <format>
#include <limits>

namespace rocjitsu {

VramStore::VramStore(const std::string &name, uint64_t bytes, uint64_t aperture_bytes)
    : bytes_(bytes), aperture_bytes_(aperture_bytes) {
  if (bytes_ == 0 || aperture_bytes_ == 0 || aperture_bytes_ > bytes_ ||
      bytes_ > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
    util::Logger::warn(
        std::format("{}: invalid video memory sizing: {} backing bytes, {} aperture bytes", name,
                    bytes_, aperture_bytes_));
    return;
  }
  fd_ = ::memfd_create("rocjitsu-vram", MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (fd_ < 0 || ::ftruncate(fd_, static_cast<off_t>(bytes_)) != 0) {
    util::Logger::warn(std::format("{}: cannot back the video memory aperture", name));
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    return;
  }

  void *mapped = ::mmap(nullptr, aperture_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (mapped == MAP_FAILED) {
    util::Logger::warn(std::format("{}: cannot map the video memory aperture", name));
    ::close(fd_);
    fd_ = -1;
    return;
  }
  mapping_ = static_cast<std::byte *>(mapped);

  // Sealed because the descriptor is shared with a client that could otherwise
  // shrink it and leave this process reading past the end of the file.
  //
  // A failure here releases both the mapping and the descriptor rather than
  // returning with them held: the object is unusable either way, and leaving
  // them open would leak a mapping of the whole aperture for the life of the
  // process.
  if (::fcntl(fd_, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) != 0) {
    util::Logger::warn(std::format("{}: cannot seal the video memory backing", name));
    ::munmap(mapping_, aperture_bytes_);
    mapping_ = nullptr;
    ::close(fd_);
    fd_ = -1;
  }
}

VramStore::~VramStore() {
  if (mapping_ != nullptr) {
    ::munmap(mapping_, aperture_bytes_);
  }
  if (fd_ >= 0) {
    ::close(fd_);
  }
}

bool VramStore::read(uint64_t offset, uint32_t &value) const {
  const std::lock_guard lock(access_mutex_);
  if (fd_ < 0 || offset > bytes_ || sizeof(value) > bytes_ - offset) {
    return false;
  }
  return ::pread(fd_, &value, sizeof(value), static_cast<off_t>(offset)) ==
         static_cast<ssize_t>(sizeof(value));
}

bool VramStore::read_all(std::span<std::byte> bytes, uint64_t at) const {
  const std::lock_guard lock(access_mutex_);
  if (fd_ < 0 || at > bytes_ || bytes.size() > bytes_ - at) {
    return false;
  }
  std::size_t done = 0;
  while (done < bytes.size()) {
    const ssize_t read =
        ::pread(fd_, bytes.data() + done, bytes.size() - done, static_cast<off_t>(at + done));
    if (read < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (read == 0) {
      return false;
    }
    done += static_cast<std::size_t>(read);
  }
  return true;
}

bool VramStore::write(uint64_t offset, uint32_t value) {
  const std::lock_guard lock(access_mutex_);
  if (fd_ < 0 || offset > bytes_ || sizeof(value) > bytes_ - offset) {
    return false;
  }
  return ::pwrite(fd_, &value, sizeof(value), static_cast<off_t>(offset)) ==
         static_cast<ssize_t>(sizeof(value));
}

bool VramStore::write_all(std::span<const std::byte> bytes, uint64_t at) {
  const std::lock_guard lock(access_mutex_);
  if (fd_ < 0 || at > bytes_ || bytes.size() > bytes_ - at) {
    return false;
  }
  std::size_t done = 0;
  while (done < bytes.size()) {
    const ssize_t wrote =
        ::pwrite(fd_, bytes.data() + done, bytes.size() - done, static_cast<off_t>(at + done));
    if (wrote < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (wrote == 0) {
      return false;
    }
    done += static_cast<std::size_t>(wrote);
  }
  return true;
}

std::optional<VramAtomicLoadResult> VramStore::atomic_load(uint64_t offset, uint32_t width) {
  if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || offset % width != 0)
    return std::nullopt;

  const std::lock_guard lock(access_mutex_);
  if (fd_ < 0 || offset > bytes_ || width > bytes_ - offset)
    return std::nullopt;

  if (mapping_ != nullptr && offset <= aperture_bytes_ && width <= aperture_bytes_ - offset) {
    std::byte *target = mapping_ + offset;
    const uint64_t value = width == sizeof(uint64_t)
                               ? std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(target))
                                     .load(std::memory_order_acquire)
                               : std::atomic_ref<uint32_t>(*reinterpret_cast<uint32_t *>(target))
                                     .load(std::memory_order_acquire);
    return VramAtomicLoadResult{.value = value};
  }

  VramAtomicLoadResult result;
  std::array<std::byte, sizeof(uint64_t)> raw{};
  if (::pread(fd_, raw.data(), width, static_cast<off_t>(offset)) != static_cast<ssize_t>(width))
    return std::nullopt;
  std::memcpy(&result.value, raw.data(), width);
  return result;
}

bool VramStore::atomic_store(uint64_t offset, uint32_t width, uint64_t value) {
  if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || offset % width != 0)
    return false;

  const std::lock_guard lock(access_mutex_);
  if (fd_ < 0 || offset > bytes_ || width > bytes_ - offset)
    return false;

  if (mapping_ != nullptr && offset <= aperture_bytes_ && width <= aperture_bytes_ - offset) {
    std::byte *target = mapping_ + offset;
    if (width == sizeof(uint64_t)) {
      std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(target))
          .store(value, std::memory_order_release);
    } else {
      std::atomic_ref<uint32_t>(*reinterpret_cast<uint32_t *>(target))
          .store(static_cast<uint32_t>(value), std::memory_order_release);
    }
    return true;
  }

  std::array<std::byte, sizeof(uint64_t)> raw{};
  std::memcpy(raw.data(), &value, width);
  return ::pwrite(fd_, raw.data(), width, static_cast<off_t>(offset)) ==
         static_cast<ssize_t>(width);
}

std::optional<VramAtomicCompareExchangeResult>
VramStore::compare_exchange(uint64_t offset, uint32_t width, uint64_t expected, uint64_t desired) {
  if ((width != sizeof(uint32_t) && width != sizeof(uint64_t)) || offset % width != 0)
    return std::nullopt;

  const std::lock_guard lock(access_mutex_);
  if (fd_ < 0 || offset > bytes_ || width > bytes_ - offset)
    return std::nullopt;

  VramAtomicCompareExchangeResult result{.observed = expected, .exchanged = false};
  if (mapping_ != nullptr && offset <= aperture_bytes_ && width <= aperture_bytes_ - offset) {
    std::byte *target = mapping_ + offset;
    if (width == sizeof(uint64_t)) {
      uint64_t observed = expected;
      result.exchanged = std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(target))
                             .compare_exchange_strong(observed, desired, std::memory_order_acq_rel,
                                                      std::memory_order_acquire);
      result.observed = observed;
    } else {
      uint32_t observed = static_cast<uint32_t>(expected);
      result.exchanged =
          std::atomic_ref<uint32_t>(*reinterpret_cast<uint32_t *>(target))
              .compare_exchange_strong(observed, static_cast<uint32_t>(desired),
                                       std::memory_order_acq_rel, std::memory_order_acquire);
      result.observed = observed;
    }
    return result;
  }

  std::array<std::byte, sizeof(uint64_t)> raw{};
  if (::pread(fd_, raw.data(), width, static_cast<off_t>(offset)) != static_cast<ssize_t>(width))
    return std::nullopt;
  std::memcpy(&result.observed, raw.data(), width);
  const uint64_t mask = width == sizeof(uint64_t) ? UINT64_MAX : UINT32_MAX;
  if ((result.observed & mask) != (expected & mask))
    return result;
  std::memcpy(raw.data(), &desired, width);
  if (::pwrite(fd_, raw.data(), width, static_cast<off_t>(offset)) != static_cast<ssize_t>(width))
    return std::nullopt;
  result.exchanged = true;
  return result;
}

} // namespace rocjitsu
