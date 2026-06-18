// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_RISC_V_MEMORY_H_
#define ROCJITSU_VM_RISC_V_MEMORY_H_

#include "simdojo/components/sparse_memory.h"

#include <string>
#include <utility>

namespace rocjitsu {
namespace risc_v {

/// @brief RISC-V address space backed by simdojo::SparseMemory.
///
/// Per-hart memory component. Inherits all read/write/load functionality
/// from simdojo::SparseMemory directly.
class Memory : public simdojo::SparseMemory {
public:
  explicit Memory(std::string name) : simdojo::SparseMemory(std::move(name)) {}
};

} // namespace risc_v
} // namespace rocjitsu

#endif // ROCJITSU_VM_RISC_V_MEMORY_H_
