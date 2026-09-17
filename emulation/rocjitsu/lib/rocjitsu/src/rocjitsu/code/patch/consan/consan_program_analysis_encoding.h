// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_encoding.h
/// @brief Target-neutral instruction facts consumed by program analysis.

#pragma once

#include <cstdint>
#include <optional>

namespace rocjitsu::consan {

/// Target-neutral meaning of one decoded cache-maintenance instruction.
enum class CacheOperation : uint8_t {
  Unsupported,
  Release,
  Acquire,
  AcquirePairPrefix,
  AcquirePairCompletion,
};

struct CacheOperationEncoding {
  CacheOperation operation = CacheOperation::Unsupported;
  bool ordinary_acquire_mutation_supported = false;
};

/// Target-normalized effect of one exact zero-count wait instruction.
struct WaitInstructionEncoding {
  bool bounded_release_counter_form = false;
  bool drains_load = false;
  bool drains_store = false;
  bool drains_lds = false;
  bool release_boundary = false;
};

struct ScratchComponentEncoding {
  uint16_t vector_address_vgpr = 0;
  uint16_t load_data_vgpr = 0;
  uint16_t store_data_vgpr = 0;
  uint16_t scalar_address_sgpr = 0;
  uint32_t immediate_offset = 0;
};

struct PrivateComponentEncoding {
  uint16_t address_vgpr = 0;
  uint16_t load_data_vgpr = 0;
  uint16_t store_data_vgpr = 0;
  uint32_t immediate_offset = 0;
};

struct LaneTransferEncoding {
  uint32_t lane_selector = 0;
  int value_source_operand = 0;
};

struct AccvgprTransferEncoding {
  std::optional<uint16_t> accumulator_vgpr;
};

enum class TargetDecodeStatus : uint8_t {
  UnsupportedArchitecture,
  UnsupportedEncodingSize,
  Decoded,
};

enum class EncodedFlatSegment : uint8_t {
  Unspecified,
  Private,
  Global,
};

/// Target-normalized causal visibility scope carried by memory operations.
///
/// This is ConSan's conservative synchronization vocabulary, not an ISA cache-
/// scope encoding. Exact encoded fields remain target-owned. Common analysis,
/// policy, modes, and validation consume only this semantic enum.
enum class MemoryScope : uint8_t {
  Wavefront,
  Workgroup,
  Agent,
  System,
};

[[nodiscard]] constexpr bool memory_scope_is_supported(MemoryScope scope) {
  switch (scope) {
  case MemoryScope::Wavefront:
  case MemoryScope::Workgroup:
  case MemoryScope::Agent:
  case MemoryScope::System:
    return true;
  }
  return false;
}

[[nodiscard]] constexpr bool memory_scope_is_agent_or_system(MemoryScope scope) {
  return scope == MemoryScope::Agent || scope == MemoryScope::System;
}

/// Target-normalized FLAT fields shared by access and ordinary-memory
/// inventory. The target owner validates raw padding and form bits once;
/// semantic inventory owners decide how the decoded operation is used.
struct VectorMemoryEncoding {
  uint32_t raw_op = 0;
  uint32_t raw_saddr = 0;
  uint32_t raw_nv = 0;
  bool raw_scale_offset = false;
  uint32_t raw_sve = 0;
  uint32_t raw_vaddr = 0;
  uint32_t raw_vsrc = 0;
  uint32_t raw_vdst = 0;
  int32_t raw_ioffset = 0;
  uint32_t raw_segment = 0;
  uint32_t raw_scope = 0;
  uint32_t raw_th = 0;
  std::optional<MemoryScope> scope;
  EncodedFlatSegment encoded_segment = EncodedFlatSegment::Unspecified;
  std::optional<uint16_t> scalar_provenance_sgpr;
  bool scope_follows_address_space = false;
  bool workgroup_acquire_ordering = false;
  bool exact_size = false;
  bool ordinary_well_formed = false;
  bool ordinary_requires_complete_registers = false;
  bool ordinary_mutation_supported = false;
};

struct VectorMemoryDecode {
  TargetDecodeStatus status = TargetDecodeStatus::UnsupportedArchitecture;
  VectorMemoryEncoding encoding;
};

struct BufferMemoryEncoding {
  int32_t raw_ioffset = 0;
  uint32_t raw_scope = 0;
  uint32_t raw_th = 0;
  std::optional<MemoryScope> scope;
  uint32_t raw_rsrc = 0;
  uint32_t raw_soffset = 0;
  bool raw_offen = false;
  bool raw_idxen = false;
  uint32_t raw_vaddr = 0;
  uint16_t address_sgpr = 0;
  std::optional<uint16_t> address_vgpr;
  uint16_t data_vgpr = 0;
  bool well_formed = false;
};

struct BufferMemoryDecode {
  TargetDecodeStatus status = TargetDecodeStatus::UnsupportedArchitecture;
  BufferMemoryEncoding encoding;
};

struct DirectLdsTransferEncoding {
  bool writes_lds = false;
  std::optional<uint8_t> address_source_operand;
  std::optional<uint16_t> memory_address_vgpr;
  uint16_t memory_address_vgpr_count = 1;
  uint32_t m0_address_mask = 0xffffffffu;
  int32_t immediate_byte_offset = 0;
};

} // namespace rocjitsu::consan
