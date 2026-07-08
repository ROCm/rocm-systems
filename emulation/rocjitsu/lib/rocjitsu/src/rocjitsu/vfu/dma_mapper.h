// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dma_mapper.h
/// @brief Guest DMA memory mapper for the rocjitsu vfio-user GPU.
///
/// When QEMU registers guest memory ranges (DMA_MAP), this component makes
/// them visible to the rocjitsu GPU's virtual address space so the GPU can
/// read kernel descriptors, AQL ring buffers, and output buffers directly.

#ifndef ROCJITSU_VFU_DMA_MAPPER_H_
#define ROCJITSU_VFU_DMA_MAPPER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

// Forward-declare libvfio-user types.
typedef struct vfu_ctx vfu_ctx_t;
typedef struct vfu_dma_info vfu_dma_info_t;

namespace rocjitsu {
class SimulatedDriver;
} // namespace rocjitsu

namespace rocjitsu::vfu {

/// @brief Tracks and applies guest DMA memory mappings into rocjitsu's GPU VA space.
///
/// Each DMA_MAP message from QEMU carries an IOVA (guest physical address) range
/// and an optional host virtual address (when the guest memory is accessible via
/// shared memfd). On DMA_MAP, the range is registered into the SimulatedDriver's
/// guest-process page table so the CP can fetch AQL packets and the CUs can
/// access kernel arguments from guest memory.
class DmaMapper {
public:
  /// @brief Construct the DMA mapper.
  /// @param driver  The SimulatedDriver that owns the GPU process/VA space.
  /// @param process_id  The rocjitsu process ID for the guest.
  explicit DmaMapper(SimulatedDriver &driver, uint32_t process_id);
  ~DmaMapper();

  DmaMapper(const DmaMapper &) = delete;
  DmaMapper &operator=(const DmaMapper &) = delete;

  /// @brief DMA_MAP callback from libvfio-user.
  static void dma_register(vfu_ctx_t *ctx, vfu_dma_info_t *info);

  /// @brief DMA_UNMAP callback from libvfio-user.
  static void dma_unregister(vfu_ctx_t *ctx, vfu_dma_info_t *info);

private:
  void on_register(vfu_dma_info_t *info);
  void on_unregister(vfu_dma_info_t *info);

  struct Mapping {
    uint64_t iova;     ///< Guest physical (IOVA) base address.
    size_t   length;   ///< Length of the region in bytes.
    void *   vaddr;    ///< Host virtual address (nullptr if not accessible).
  };

  SimulatedDriver &driver_;
  uint32_t process_id_;
  std::vector<Mapping> mappings_;
};

} // namespace rocjitsu::vfu

#endif // ROCJITSU_VFU_DMA_MAPPER_H_
