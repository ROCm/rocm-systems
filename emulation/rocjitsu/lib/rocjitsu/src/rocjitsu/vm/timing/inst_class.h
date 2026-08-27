// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file inst_class.h
/// @brief What an instruction is, at the granularity timing cares about.
///
/// @details rocjitsu's Instruction carries an encoding, an opcode, a mnemonic
/// and a handful of flags, but no notion of which execution unit an
/// instruction occupies or what it costs. Those are timing concepts, so they
/// live here rather than in the ISA layer, where they would have to be
/// regenerated per target and would bind the decoder to one performance model.
///
/// The split is by cost and by resource, not by encoding: two instructions in
/// the same encoding that occupy different units are different classes, and two
/// in different encodings that behave identically are the same class.

#pragma once

#include <cstdint>

namespace rocjitsu::timing {

/// @brief The class of an instruction.
///
/// @details Loads and stores are separate classes despite sharing a unit,
/// because they land on different wait counters and complete differently.
enum class InstClass : std::uint8_t {
  /// @brief The observer could not classify this opcode.
  ///
  /// @details First so that a zero-initialized class is the expensive one. This
  /// is not "an instruction that does nothing interesting" — it is "the
  /// observer does not know", and a model is required to charge it
  /// `unknown.issue_cycles`, which the config resolves to the most expensive
  /// class it names. An opcode nobody has classified must make a run read slow
  /// and look suspicious; charging it zero would make coverage gaps
  /// indistinguishable from a fast kernel. See MachineSpec.
  Unknown,
  VectorAlu,          ///< Ordinary lane-parallel arithmetic.
  ScalarAlu,          ///< Uniform arithmetic on the scalar unit.
  Transcendental,     ///< Reciprocal, square root, exponential and friends.
  MatrixMultiply,     ///< MFMA, WMMA and SWMMAC — the matrix pipe.
  LdsRead,            ///< Local data share read.
  LdsWrite,           ///< Local data share write.
  VectorMemoryRead,   ///< Global, buffer, scratch or flat load.
  VectorMemoryWrite,  ///< Global, buffer, scratch or flat store.
  VectorMemoryAtomic, ///< Atomic read-modify-write to global memory.
  ScalarMemory,       ///< Scalar (constant) cache access.
  TensorMemory,       ///< Tensor data-mover transfer.
  Export,             ///< Export or position/parameter write.
  Branch,             ///< Direct or indirect control transfer.
  WaitCounter,        ///< s_waitcnt and the per-counter waits.
  DelayAlu,           ///< s_delay_alu and s_wait_alu.
  Barrier,            ///< Workgroup barrier arrive and wait.
  Message,            ///< s_sendmsg and similar side-band operations.
  Nop,                ///< s_nop and encoded no-ops.
  Terminate,          ///< s_endpgm and friends.
  Count,              ///< Number of classes; not a class itself.
};

inline constexpr std::size_t kNumInstClasses = static_cast<std::size_t>(InstClass::Count);

/// @brief Fixed-point denominator for front-end occupancy.
///
/// @details Front-end cost per instruction is a fraction of a cycle and the
/// accumulator it feeds counts whole cycles, so it is carried in sixteenths.
inline constexpr std::uint64_t kFrontEndScale = 16;

/// @brief The hardware unit an instruction occupies while it issues.
///
/// @details Distinct from InstClass because several classes share a unit: a
/// load and a store both occupy the vector-memory issue path, and a wait
/// occupies nothing. Throughput contention is tracked per unit, so this is the
/// key a model stores its issue-port state under.
enum class FunctionalUnit : std::uint8_t {
  None, ///< Occupies no issue port — waits, barriers, no-ops, terminate.
  VectorAlu,
  ScalarAlu,
  Transcendental,
  MatrixMultiply,
  LocalDataShare,
  VectorMemory,
  ScalarMemory,
  Branch,
  Export,
  Count,
};

inline constexpr std::size_t kNumFunctionalUnits = static_cast<std::size_t>(FunctionalUnit::Count);

/// @brief Config-key and report name for a class. Stable; it is part of the
///        architecture config file's vocabulary.
inline constexpr const char *inst_class_name(InstClass value) {
  switch (value) {
  case InstClass::Unknown:
    return "unknown";
  case InstClass::VectorAlu:
    return "vector_alu";
  case InstClass::ScalarAlu:
    return "scalar_alu";
  case InstClass::Transcendental:
    return "transcendental";
  case InstClass::MatrixMultiply:
    return "matrix_multiply";
  case InstClass::LdsRead:
    return "lds_read";
  case InstClass::LdsWrite:
    return "lds_write";
  case InstClass::VectorMemoryRead:
    return "vector_memory_read";
  case InstClass::VectorMemoryWrite:
    return "vector_memory_write";
  case InstClass::VectorMemoryAtomic:
    return "vector_memory_atomic";
  case InstClass::ScalarMemory:
    return "scalar_memory";
  case InstClass::TensorMemory:
    return "tensor_memory";
  case InstClass::Export:
    return "export";
  case InstClass::Branch:
    return "branch";
  case InstClass::WaitCounter:
    return "wait_counter";
  case InstClass::DelayAlu:
    return "delay_alu";
  case InstClass::Barrier:
    return "barrier";
  case InstClass::Message:
    return "message";
  case InstClass::Nop:
    return "nop";
  case InstClass::Terminate:
    return "terminate";
  case InstClass::Count:
    break;
  }
  return "unknown";
}

inline constexpr const char *functional_unit_name(FunctionalUnit value) {
  switch (value) {
  case FunctionalUnit::None:
    return "none";
  case FunctionalUnit::VectorAlu:
    return "vector_alu";
  case FunctionalUnit::ScalarAlu:
    return "scalar_alu";
  case FunctionalUnit::Transcendental:
    return "transcendental";
  case FunctionalUnit::MatrixMultiply:
    return "matrix_multiply";
  case FunctionalUnit::LocalDataShare:
    return "local_data_share";
  case FunctionalUnit::VectorMemory:
    return "vector_memory";
  case FunctionalUnit::ScalarMemory:
    return "scalar_memory";
  case FunctionalUnit::Branch:
    return "branch";
  case FunctionalUnit::Export:
    return "export";
  case FunctionalUnit::Count:
    break;
  }
  return "unknown";
}

/// @brief The unit a class occupies.
///
/// @details Total, so callers never need a fallback. Unknown maps to VectorAlu
/// rather than None: an unclassified opcode must contend for something, and
/// the vector pipe is the one every kernel is already pressuring, so charging
/// it there cannot be cancelled out by an idle port.
inline constexpr FunctionalUnit unit_for_class(InstClass value) {
  switch (value) {
  case InstClass::Unknown:
  case InstClass::VectorAlu:
    return FunctionalUnit::VectorAlu;
  case InstClass::ScalarAlu:
    return FunctionalUnit::ScalarAlu;
  case InstClass::Transcendental:
    return FunctionalUnit::Transcendental;
  case InstClass::MatrixMultiply:
    return FunctionalUnit::MatrixMultiply;
  case InstClass::LdsRead:
  case InstClass::LdsWrite:
    return FunctionalUnit::LocalDataShare;
  case InstClass::VectorMemoryRead:
  case InstClass::VectorMemoryWrite:
  case InstClass::VectorMemoryAtomic:
  case InstClass::TensorMemory:
    return FunctionalUnit::VectorMemory;
  case InstClass::ScalarMemory:
    return FunctionalUnit::ScalarMemory;
  case InstClass::Branch:
    return FunctionalUnit::Branch;
  case InstClass::Export:
    return FunctionalUnit::Export;
  case InstClass::WaitCounter:
  case InstClass::DelayAlu:
  case InstClass::Barrier:
  case InstClass::Message:
  case InstClass::Nop:
  case InstClass::Terminate:
  case InstClass::Count:
    return FunctionalUnit::None;
  }
  return FunctionalUnit::VectorAlu;
}

/// @brief Whether a class issues a request some wait counter tracks.
inline constexpr bool class_is_memory(InstClass value) {
  switch (value) {
  case InstClass::LdsRead:
  case InstClass::LdsWrite:
  case InstClass::VectorMemoryRead:
  case InstClass::VectorMemoryWrite:
  case InstClass::VectorMemoryAtomic:
  case InstClass::ScalarMemory:
  case InstClass::TensorMemory:
  case InstClass::Export:
    return true;
  default:
    return false;
  }
}

} // namespace rocjitsu::timing
