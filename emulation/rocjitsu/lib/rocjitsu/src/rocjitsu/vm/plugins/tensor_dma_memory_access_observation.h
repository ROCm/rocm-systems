// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file tensor_dma_memory_access_observation.h
/// @brief Completed global-memory accesses performed by one tensor DMA instruction.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace rocjitsu::amdgpu {

/// @brief Callback-lifetime view of tensor-DMA global element addresses.
///
/// @details Address generation is deferred until a consumer supplies its final
/// destination. This lets consumers enforce their own capacity limits before
/// retaining any per-element state. The view and its backing context are valid
/// only for the duration of the observation callback.
class TensorDmaAddressView {
public:
  using CopyFunction = bool (*)(const void *context, std::span<uint64_t> destination);

  TensorDmaAddressView() = default;

  /// Construct a view over an existing callback-lifetime span.
  TensorDmaAddressView(std::span<const uint64_t> addresses)
      : context_(addresses.data()), size_(addresses.size()), copy_(&copy_span) {}

  /// Construct a lazily generated view over callback-lifetime context.
  static TensorDmaAddressView deferred(size_t size, const void *context, CopyFunction copy) {
    return TensorDmaAddressView(size, context, copy);
  }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  bool valid() const { return empty() || copy_ != nullptr; }

  /// Copy all addresses into an exactly sized destination.
  bool copy_to(std::span<uint64_t> destination) const {
    if (destination.size() != size_)
      return false;
    return empty() || (copy_ != nullptr && copy_(context_, destination));
  }

private:
  TensorDmaAddressView(size_t size, const void *context, CopyFunction copy)
      : context_(context), size_(size), copy_(copy) {}

  static bool copy_span(const void *context, std::span<uint64_t> destination) {
    const auto *source = static_cast<const uint64_t *>(context);
    for (size_t i = 0; i < destination.size(); ++i)
      destination[i] = source[i];
    return true;
  }

  const void *context_ = nullptr;
  size_t size_ = 0;
  CopyFunction copy_ = nullptr;
};

/// @brief The in-bounds global requests attempted by one tensor DMA instruction.
///
/// @details The address view contains only in-bounds global element base
/// addresses, in the exact order in which the implementation attempted them.
/// Duplicates are preserved. The callback is emitted only after the instruction
/// and any descriptor-requested atomic-barrier arrival return normally. Memory
/// access outcomes are intentionally not filtered, matching the observer's
/// boundary.
///
/// The view borrows execution-owned context and is valid only for the duration
/// of the callback. A consumer that keeps addresses must copy them during the
/// callback.
struct TensorDmaMemoryAccessObservation {
  /// @brief Instruction mnemonic; points at static storage.
  std::string_view mnemonic;
  uint64_t pc = 0;              ///< PC of the issuing wavefront.
  uint32_t compute_unit_id = 0; ///< Component id of the issuing compute unit.
  uint32_t dispatch_id = 0;
  uint32_t queue_id = 0;
  uint32_t workgroup_id = 0;
  uint32_t wavefront_id = 0; ///< Wavefront slot within the compute unit.
  uint32_t process_id = 0;   ///< VMID of the global addresses.
  uint32_t element_size_bytes = 0;
  bool is_load = true; ///< True for global-to-LDS, false for LDS-to-global.
  TensorDmaAddressView addresses;
  uint32_t tile_dim0 = 0; ///< Descriptor tile-dimension-zero extent.
  /// Dimension-one tile extent; gather uses the valid-index count.
  uint32_t tile_dim1 = 0;
  uint32_t data_size = 0;         ///< Raw element-size code; bytes = 1 << data_size.
  int64_t tensor_dim0_stride = 0; ///< Descriptor dimension-zero stride in bytes.
  int64_t tensor_dim1_stride = 0; ///< Descriptor dimension-one stride in bytes.
};

} // namespace rocjitsu::amdgpu
