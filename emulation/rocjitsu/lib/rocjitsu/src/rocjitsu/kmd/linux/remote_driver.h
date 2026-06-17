// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_REMOTE_DRIVER_H_
#define ROCJITSU_KMD_LINUX_REMOTE_DRIVER_H_

/// @file remote_driver.h
/// @brief Client-side RPC stub that forwards KFD ioctls to the rocjitsu daemon.
///
/// @details Implements the Driver interface by serializing ioctl requests over
/// a Unix domain socket to the daemon process. GPU memory is shared via memfds
/// passed through SCM_RIGHTS. The client mmaps these memfds locally at the
/// addresses ROCR's FMM expects.

#include "rocjitsu/vm/driver.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

/// @brief Client-side driver that forwards ioctls to the rocjitsu daemon.
///
/// @details Connects to the daemon over a Unix domain socket and forwards
/// KFD ioctls via RPC. Owned by InterposerContext, not a global singleton.
class RemoteDriver : public Driver {
public:
  /// @brief Construct from an already-connected Unix socket fd.
  explicit RemoteDriver(int sock_fd);

  ~RemoteDriver() override;

  /// @brief Get the daemon's sysfs topology directory path.
  [[nodiscard]] const std::string &topology_path() const { return topology_path_; }

  /// @brief Get the daemon's DRM sysfs directory path.
  [[nodiscard]] const std::string &drm_path() const { return drm_path_; }

  /// @brief Find a stored memfd that covers the given GPUVM address.
  /// @details Used by the interposer to intercept anonymous MAP_FIXED at
  /// addresses that have daemon-shared memfd mappings.
  /// @param addr The target address to look up.
  /// @param length The mapping length.
  /// @param[out] memfd_out The memfd covering this address.
  /// @param[out] memfd_offset The offset within the memfd.
  /// @returns true if a matching allocation was found.
  [[nodiscard]] bool find_memfd_for_addr(void *addr, size_t length, int *memfd_out,
                                         off_t *memfd_offset);

  /// @brief Perform the RPC handshake with the daemon.
  /// @details Sends RPC_HANDSHAKE, receives the topology path and gpu_id,
  /// and creates a synthetic memfd to use as the KFD fd.
  /// @retval >=0 Synthetic KFD fd on success.
  /// @retval -1 Handshake failed (socket error or daemon rejected).
  int open() override;

  /// @brief Send RPC_CLOSE to the daemon.
  /// @retval 0 Success.
  /// @retval -1 Socket error.
  int close() override;

  /// @brief Forward a KFD ioctl to the daemon via RPC_IOCTL.
  /// @param request The AMDKFD_IOC_* ioctl number.
  /// @param arg Pointer to the ioctl args struct (read and possibly modified).
  /// @retval 0 Success.
  /// @retval negative Negative errno from the daemon's ioctl dispatch, or -1 on
  /// socket error.
  int ioctl(unsigned long request, void *arg) override;

  /// @brief Forward an mmap request to the daemon via RPC_MMAP.
  /// @param addr Requested mapping address (may include MAP_FIXED).
  /// @param length Length in bytes to map.
  /// @param prot Memory protection flags.
  /// @param flags Mapping flags.
  /// @param offset KFD mmap offset encoding.
  /// @retval non-MAP_FAILED Pointer to the locally mapped memory.
  /// @retval MAP_FAILED Mapping failed; errno is set.
  void *mmap(void *addr, size_t length, int prot, int flags, off_t offset) override;

  /// @brief Forward a munmap request to the daemon via RPC_MUNMAP.
  /// @param addr Address of the mapping to unmap.
  /// @param length Length in bytes to unmap.
  /// @retval 0 Success.
  /// @retval -ENOENT Address not found in daemon's mappings.
  /// @retval -1 Socket error.
  int munmap(void *addr, size_t length) override;

private:
  int send_ioctl(unsigned long request, void *arg);
  int send_mmap(void *addr, size_t length, int prot, int flags, off_t offset, int *memfd_out);

  int sock_ = -1;                    ///< Unix socket connection to the daemon.
  uint32_t next_id_ = 0;             ///< Monotonic request ID counter (for debugging).
  std::string topology_path_;        ///< Daemon's sysfs topology directory path.
  std::string drm_path_;             ///< Daemon's DRM sysfs directory path.
  std::atomic<bool> closing_{false}; ///< Set by close() to break WAIT_EVENTS loops.
  int shutdown_efd_ = -1;            ///< eventfd written by close() to wake WAIT_EVENTS pollers.

  /// @brief Serializes all RPC send+recv pairs on sock_.
  /// @details ROCR is multithreaded — concurrent ioctl/mmap calls interleave
  /// socket writes without this lock, corrupting the RPC stream.
  std::mutex rpc_mutex_;

  /// @brief Memfds received from the daemon during ALLOC_MEMORY, keyed by handle.
  std::unordered_map<uint64_t, int> handle_memfds_;

  /// @brief Maps GPUVM addresses to allocation memfds for anonymous MAP_FIXED
  /// interception. When ROCR's FMM does anonymous MAP_FIXED at a GPUVM address
  /// that already has a daemon-shared memfd, we use the memfd instead to
  /// preserve cross-process sharing. Keyed by alloc va_addr, value is memfd.
  struct AllocRange {
    uint64_t va;
    uint64_t size;
    int memfd;
  };
  std::vector<AllocRange> alloc_ranges_;
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_REMOTE_DRIVER_H_
