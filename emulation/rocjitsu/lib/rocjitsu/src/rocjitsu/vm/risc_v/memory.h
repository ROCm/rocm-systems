// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

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
