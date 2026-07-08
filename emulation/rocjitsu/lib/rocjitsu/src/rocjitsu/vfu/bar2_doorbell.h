// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file bar2_doorbell.h
/// @brief BAR2 doorbell region handler for the rocjitsu vfio-user GPU.
///
/// The doorbell BAR (64-bit prefetchable, 2 MB) is how the guest amdgpu
/// driver signals the GPU's CP when new AQL packets are ready. Each queue
/// is assigned a unique 8-byte slot within this BAR.
///
/// This handler traps all writes from the guest and forwards them to the
/// rocjitsu SimulatedDriver's doorbell dispatch path, which triggers the
/// CommandProcessor to fetch and process new packets.

#ifndef ROCJITSU_VFU_BAR2_DOORBELL_H_
#define ROCJITSU_VFU_BAR2_DOORBELL_H_

#include <cstdint>
#include <cstddef>
#include <functional>
#include <sys/types.h>  // ssize_t

// Forward-declare libvfio-user types.
typedef struct vfu_ctx vfu_ctx_t;

namespace rocjitsu::vfu {

/// @brief BAR2 doorbell region.
///
/// Handles trapped doorbell writes from the guest and dispatches them into
/// the rocjitsu CP via a caller-supplied callback. The doorbell page is also
/// backed by a memfd so the guest can mmap it and write doorbells without
/// socket round-trips on supported platforms.
class Bar2Doorbell {
public:
  /// @brief Callback type invoked on each doorbell write from the guest.
  /// @param doorbell_offset  Byte offset within the doorbell BAR (slot address).
  /// @param value            64-bit value written by the guest CP driver.
  using RingCallback = std::function<void(uint32_t doorbell_offset, uint64_t value)>;

  /// @brief Construct the doorbell BAR handler.
  /// @param size  Doorbell BAR size in bytes (typically 2 MB).
  /// @param cb    Callback invoked on each doorbell write.
  Bar2Doorbell(uint64_t size, RingCallback cb);
  ~Bar2Doorbell();

  Bar2Doorbell(const Bar2Doorbell &) = delete;
  Bar2Doorbell &operator=(const Bar2Doorbell &) = delete;

  /// @brief Register this BAR with the libvfio-user context.
  /// @returns 0 on success, -1 on failure.
  int setup(vfu_ctx_t *ctx);

  /// @brief libvfio-user region access callback (static trampoline).
  /// @note Signature matches vfu_region_access_cb_t: loff_t is long on Linux.
  static ssize_t access_cb(vfu_ctx_t *ctx, char *buf, size_t count,
                            long offset, bool is_write);

  /// @brief The backing memfd for direct doorbell page mmap.
  int fd() const { return memfd_; }

private:
  ssize_t handle_access(char *buf, size_t count, long offset, bool is_write);

  uint64_t size_;
  int memfd_ = -1;
  RingCallback ring_cb_;
};

} // namespace rocjitsu::vfu

#endif // ROCJITSU_VFU_BAR2_DOORBELL_H_
