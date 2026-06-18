// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file memory_interface.h
/// @brief Abstract byte-addressed memory controller interface.

#ifndef SIMDOJO_COMPONENTS_MEMORY_INTERFACE_H_
#define SIMDOJO_COMPONENTS_MEMORY_INTERFACE_H_

#include <cstdint>

namespace simdojo {

/// @brief Byte-addressed memory controller interface.
///
/// Implemented by controllers that serve memory requests: memory controllers,
/// cache controllers, and scratchpad memories. NOT implemented by backing
/// stores (SparseMemory, Cache<>) which are data structures wrapped by
/// controllers.
class MemoryInterface {
public:
  virtual ~MemoryInterface() = default;

  /// @brief Read bytes from the given address into a destination buffer.
  /// @param addr Start address to read from.
  /// @param dst Destination buffer for the read data.
  /// @param size Number of bytes to read.
  virtual void read(uint64_t addr, uint8_t *dst, uint32_t size) = 0;

  /// @brief Write bytes from a source buffer to the given address.
  /// @param addr Start address to write to.
  /// @param src Source buffer containing data to write.
  /// @param size Number of bytes to write.
  virtual void write(uint64_t addr, const uint8_t *src, uint32_t size) = 0;
};

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_MEMORY_INTERFACE_H_
