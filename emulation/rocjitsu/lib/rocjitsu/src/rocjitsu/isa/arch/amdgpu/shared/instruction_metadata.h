// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/generated/shared/operations.h"

#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Decoded addressing family, independent of issue and completion policy.
/// @details FLAT instructions retain Flat until addresses are resolved at issue.
/// The encoded segment selects Global or Scratch when it fixes the address space.
/// LDS transfers keep their source addressing family; their decoded destination
/// and memory issue metadata describe the separate LDS effect.
enum class MemoryAddressing : uint8_t {
  None,
  Scalar,
  Flat,
  Global,
  Scratch,
  Buffer,
  TypedBuffer,
  Image,
  Lds,
  Gds,
  DirectLds,
  Tensor,
  Cluster,
};

/// @brief Architectural memory operation, independent of implemented pipelines.
/// @details Atomic denotes a read-modify-write operation whether or not it returns
/// data to a register. Decoded destination operands describe any returned value.
/// These categories do not imply execution support or a particular wait counter.
enum class MemoryKind : uint8_t {
  None,
  Load,
  Store,
  Atomic,
  Sample,
  Bvh,
  Cache,
  Prefetch,
  Probe,
  Other,
};

} // namespace rocjitsu::amdgpu
