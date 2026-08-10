// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sealed_memfd.h
/// @brief RAII wrapper for a sealable anonymous memory file (memfd).
///
/// Both Bar0Vram and Bar2Doorbell need the same sequence:
///   memfd_create → ftruncate → optional write → F_ADD_SEALS
/// SealedMemfd encapsulates that setup so it isn't duplicated.

#ifndef ROCJITSU_VFU_SEALED_MEMFD_H_
#define ROCJITSU_VFU_SEALED_MEMFD_H_

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
#ifndef MFD_ALLOW_SEALING
#define MFD_ALLOW_SEALING 0x0002U
#endif

namespace rocjitsu::vfu {

/// @brief Owning RAII wrapper around a sealed anonymous memfd.
///
/// On construction: creates the memfd, truncates to @p size, optionally
/// fills every byte with @p init_byte, then adds F_SEAL_SHRINK|F_SEAL_GROW.
/// Sealing is required so QEMU's vfio-user client can create a dma-buf from
/// the fd without further resize. The destructor closes the fd.
class SealedMemfd {
public:
  /// @param name      Label passed to memfd_create (shown in /proc/*/fd/).
  /// @param size      Size in bytes.
  /// @param init_byte If >= 0, every byte is initialised to this value before
  ///                  sealing. Pass -1 (the default) to leave memory zeroed.
  explicit SealedMemfd(const char *name, size_t size, int init_byte = -1)
      : size_(size) {
    fd_ = static_cast<int>(
        syscall(SYS_memfd_create, name, MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (fd_ < 0) {
      std::perror("memfd_create");
      return;
    }
    if (ftruncate(fd_, static_cast<off_t>(size_)) != 0) {
      std::perror("ftruncate (memfd)");
      ::close(fd_);
      fd_ = -1;
      return;
    }
    if (init_byte >= 0) {
      void *p = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd_, 0);
      if (p != MAP_FAILED) {
        std::memset(p, init_byte, size_);
        ::munmap(p, size_);
      }
    }
    if (fcntl(fd_, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW) != 0)
      std::perror("F_ADD_SEALS (memfd) — dma-buf path may not work");
  }

  ~SealedMemfd() {
    if (fd_ >= 0)
      ::close(fd_);
  }

  SealedMemfd(const SealedMemfd &) = delete;
  SealedMemfd &operator=(const SealedMemfd &) = delete;

  /// @brief Returns the file descriptor, or -1 if initialization failed.
  int fd() const { return fd_; }

  /// @brief Size passed to the constructor.
  size_t size() const { return size_; }

  /// @brief True if the fd was created successfully.
  bool valid() const { return fd_ >= 0; }

  /// @brief Map @p len bytes at @p byte_offset for writing, call @p fn(ptr),
  ///        then unmap. Returns false if the mmap fails.
  ///
  /// The offset is page-aligned internally; @p byte_offset must be
  /// page-aligned by the caller (the sealing allows MAP_SHARED writes).
  template <typename Fn>
  bool with_mapping(uint64_t byte_offset, size_t len, int prot, Fn &&fn) {
    if (fd_ < 0)
      return false;
    void *p = ::mmap(nullptr, len, prot, MAP_SHARED, fd_,
                     static_cast<off_t>(byte_offset));
    if (p == MAP_FAILED)
      return false;
    fn(p);
    ::munmap(p, len);
    return true;
  }

private:
  int fd_ = -1;
  size_t size_;
};

} // namespace rocjitsu::vfu

#endif // ROCJITSU_VFU_SEALED_MEMFD_H_
