// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file bar5_mmio.h
/// @brief BAR5 MMIO register model for the rocjitsu vfio-user GPU.
///
/// Handles trapped MMIO register reads/writes from the guest amdgpu driver.
/// Implements the minimal register subset required for GPU initialization and
/// queue operation. Unknown registers return 0 and are logged.

#ifndef ROCJITSU_VFU_BAR5_MMIO_H_
#define ROCJITSU_VFU_BAR5_MMIO_H_

#include <cstdint>
#include <cstddef>
#include <sys/types.h>  // ssize_t
#include <atomic>

// Forward-declare libvfio-user types.
typedef struct vfu_ctx vfu_ctx_t;

namespace rocjitsu::vfu {

/// @brief MMIO register model for BAR5 (256 KB register window).
///
/// Maintains a flat 256 KB shadow register array backed by host memory.
/// Most registers are read/write shadows; selected registers (e.g. GRBM_STATUS,
/// SMC response, SDMA version) return fixed "plausible" values from the model.
///
/// The amdgpu driver iteratively polls registers during hw_init. This model
/// handles the first wave of required probes and logs any unrecognized access
/// so register coverage can be extended incrementally.
class MmioModel {
public:
  /// @param vram_fd    memfd backing BAR0 VRAM (borrowed, not owned).
  /// @param vram_size  Size of the VRAM region in bytes.
  explicit MmioModel(int vram_fd, uint64_t vram_size);
  ~MmioModel() = default;

  /// @brief Read handler — called by vfio-user when guest reads BAR5.
  /// @param buf    Buffer to fill with read data.
  /// @param count  Number of bytes to read (must be 4 for register accesses).
  /// @param offset Byte offset within BAR5.
  /// @returns Bytes read (== count) on success, -1 on error.
  ssize_t read(char *buf, size_t count, long offset);

  /// @brief Write handler — called by vfio-user when guest writes BAR5.
  /// @param buf    Buffer containing the value to write.
  /// @param count  Number of bytes to write (must be 4 for register accesses).
  /// @param offset Byte offset within BAR5.
  /// @returns Bytes written (== count) on success, -1 on error.
  ssize_t write(const char *buf, size_t count, long offset);

  /// @brief libvfio-user region access callback (static trampoline).
  /// @note Signature matches vfu_region_access_cb_t: loff_t is long on Linux.
  static ssize_t access_cb(vfu_ctx_t *ctx, char *buf, size_t count,
                            long offset, bool is_write);

private:
  uint32_t read_register(uint32_t byte_offset);
  void write_register(uint32_t byte_offset, uint32_t value);

  /// @brief Read an indirectly-addressed register (via MM_INDEX/MM_DATA).
  /// Returns 0 for unknown indirect addresses.
  uint32_t read_indirect(uint32_t dword_index);

  /// @brief Shadow register array: one uint32_t per 4-byte register slot.
  static constexpr size_t kRegArraySize = 256 * 1024 / 4;
  uint32_t regs_[kRegArraySize]{};

  /// @brief Last dword index written to MM_INDEX (for MM_DATA reads).
  uint32_t mm_index_{0};

  /// @brief High 32 bits of the MM indirect address (written to MM_INDEX_HI).
  /// Combined with mm_index_ to form a 64-bit byte address into VRAM.
  uint32_t mm_index_hi_{0};

  /// @brief memfd backing BAR0 VRAM (borrowed). -1 if not available.
  int vram_fd_{-1};

  /// @brief Size of the VRAM region in bytes.
  uint64_t vram_size_{0};

  /// @brief 64-bit monotonically increasing timestamp counter (GPU clock stub).
  uint64_t timestamp_counter_{0};
};

} // namespace rocjitsu::vfu

#endif // ROCJITSU_VFU_BAR5_MMIO_H_
