// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file bar0_vram.h
/// @brief BAR0 VRAM aperture handler for the rocjitsu vfio-user GPU.
///
/// Exposes rocjitsu's GpuMemory as a memfd-backed 64-bit prefetchable BAR0.
/// The guest can mmap BAR0 directly (no per-access trapping) because BAR0
/// is registered with VFU_REGION_FLAG_MEM and a backing fd.

#ifndef ROCJITSU_VFU_BAR0_VRAM_H_
#define ROCJITSU_VFU_BAR0_VRAM_H_

#include <cstdint>

// Forward-declare libvfio-user types.
typedef struct vfu_ctx vfu_ctx_t;

namespace rocjitsu {
namespace amdgpu {
class GpuMemory;
} // namespace amdgpu
} // namespace rocjitsu

namespace rocjitsu::vfu {

/// @brief BAR0 VRAM aperture.
///
/// Creates and owns a memfd that backs the full VRAM aperture. Immediately after
/// creation, the IP discovery binary for GFX9.4.4 is written at
/// (vram_size - DISCOVERY_TMR_OFFSET) so the amdgpu driver can read it via
/// amdgpu_device_vram_access() during ip_discovery_init().
class Bar0Vram {
public:
  /// @brief Create the VRAM BAR.
  /// @param size  Size of the VRAM aperture in bytes (must be power-of-two).
  explicit Bar0Vram(uint64_t size);
  ~Bar0Vram();

  Bar0Vram(const Bar0Vram &) = delete;
  Bar0Vram &operator=(const Bar0Vram &) = delete;

  /// @brief Register this BAR with the libvfio-user context.
  /// @returns 0 on success, -1 on failure.
  int setup(vfu_ctx_t *ctx);

  /// @brief The backing memfd file descriptor.
  int fd() const { return memfd_; }

  /// @brief Advertised BAR size in bytes.
  uint64_t size() const { return size_; }

private:
  uint64_t size_;
  int memfd_ = -1;
};

} // namespace rocjitsu::vfu

#endif // ROCJITSU_VFU_BAR0_VRAM_H_
