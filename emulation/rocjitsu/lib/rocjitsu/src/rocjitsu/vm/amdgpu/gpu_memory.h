// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_memory.h
/// @brief Physical sparse backing for the AMDGPU model.

#pragma once

#include "simdojo/components/sparse_memory.h"

#include <string>
#include <utility>

namespace rocjitsu::amdgpu {

/// @brief Physical sparse backing bytes for an emulated GPU.
/// @details This component deliberately has no request port and interprets no
/// virtual-address identity. Memory controllers adapt transport requests to
/// GpuVm, whose address-space implementations resolve them before accessing
/// this backing store.
class GpuMemory : public simdojo::SparseMemory {
public:
  explicit GpuMemory(std::string name) : simdojo::SparseMemory(std::move(name)) {}
};

} // namespace rocjitsu::amdgpu
