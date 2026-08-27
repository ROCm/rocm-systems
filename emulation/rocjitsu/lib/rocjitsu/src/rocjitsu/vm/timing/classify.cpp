// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/classify.h"

#include "rocjitsu/isa/instruction.h"

#include <array>
#include <cctype>
#include <optional>

namespace rocjitsu::timing {
namespace {

bool has_prefix(std::string_view text, std::string_view prefix) { return text.starts_with(prefix); }

bool contains(std::string_view text, std::string_view needle) {
  return text.find(needle) != std::string_view::npos;
}

/// @brief Whether a vector-memory mnemonic names an atomic.
///
/// @details Tested before load and store because an atomic mnemonic usually
/// contains neither word, and the ones that do -- a returning atomic -- are
/// still atomics for timing: they occupy the memory path for the whole round
/// trip whether or not the result is written back.
bool is_atomic_name(std::string_view mnemonic) { return contains(mnemonic, "atomic"); }

InstClass vector_memory_class(std::string_view mnemonic) {
  if (is_atomic_name(mnemonic))
    return InstClass::VectorMemoryAtomic;
  if (contains(mnemonic, "store"))
    return InstClass::VectorMemoryWrite;
  // Everything else in these encodings moves data toward the wavefront: loads,
  // sample and gather operations, and the prefetch-shaped ones that still
  // occupy the return path.
  return InstClass::VectorMemoryRead;
}

/// @brief Split a local-data-share mnemonic into read or write.
///
/// @details A returning atomic reads as well as writes and is classified as a
/// read, because the wavefront waits for its result and waiting is what a model
/// has to get right. A non-returning one is a pure write nothing waits on.
InstClass lds_class(std::string_view mnemonic) {
  if (contains(mnemonic, "read") || contains(mnemonic, "load") || contains(mnemonic, "_rtn"))
    return InstClass::LdsRead;
  // The lane-crossing permutes name neither a read nor a load, but they return
  // a value to a vector register and the wavefront waits for it, so they cost
  // what a read costs. ds_swizzle has the same shape. Calling them writes would
  // drop them off the counter their consumer waits on.
  if (contains(mnemonic, "permute") || contains(mnemonic, "swizzle"))
    return InstClass::LdsRead;
  if (contains(mnemonic, "write") || contains(mnemonic, "store"))
    return InstClass::LdsWrite;
  // Bare atomics, appends and consumes update memory without returning.
  return InstClass::LdsWrite;
}

/// @brief Transcendental operation stems, matched after the `v_` prefix and an
///        optional packed infix.
///
/// @details These run on the separate low-throughput pipe rather than the main
/// vector unit. Listed by operation name because that is what is stable across
/// targets; the encoding that carries them is not.
constexpr std::array<std::string_view, 10> kTranscendentalStems = {
    "rcp", "rsq", "sqrt", "log", "exp", "sin", "cos", "rcp_iflag", "tanh", "exp2",
};

/// @brief Whether a vector mnemonic names a transcendental.
///
/// @details The stem is looked for at a bounded position rather than anywhere
/// in the string, so `v_rcp_f32`, `v_pk_rcp_f16` and `v_rcp_iflag_f32` all
/// match while a longer name that merely contains the letters does not.
bool is_transcendental_name(std::string_view mnemonic) {
  std::string_view body = mnemonic.substr(2); // Past the leading "v_".
  if (has_prefix(body, "pk_"))
    body = body.substr(3);
  for (std::string_view stem : kTranscendentalStems) {
    if (!has_prefix(body, stem))
      continue;
    const std::string_view rest = body.substr(stem.size());
    if (rest.empty() || rest.front() == '_')
      return true;
  }
  return false;
}

/// @brief Classify a scalar (`s_`) mnemonic.
///
/// @details Falls through to ScalarAlu rather than to Unknown, and that is a
/// classification rather than a guess: the `s_` prefix is itself evidence of
/// which unit the instruction occupies, and everything on the scalar unit that
/// is *not* scalar arithmetic is named above.
InstClass scalar_class(std::string_view mnemonic, const rocjitsu::Instruction &inst) {
  if (has_prefix(mnemonic, "s_endpgm") || (inst.flags() & rocjitsu::PROGRAM_TERMINATOR) != 0)
    return InstClass::Terminate;
  if (has_prefix(mnemonic, "s_nop"))
    return InstClass::Nop;
  // Before the wait family: s_wait_alu and s_delay_alu carry the waitcnt flag
  // but describe an ALU dependency, not outstanding memory.
  if (has_prefix(mnemonic, "s_delay_alu") || has_prefix(mnemonic, "s_wait_alu"))
    return InstClass::DelayAlu;
  if (has_prefix(mnemonic, "s_waitcnt") || has_prefix(mnemonic, "s_wait_"))
    return InstClass::WaitCounter;
  if (has_prefix(mnemonic, "s_barrier"))
    return InstClass::Barrier;
  if (has_prefix(mnemonic, "s_sendmsg") || has_prefix(mnemonic, "s_ttracedata") ||
      has_prefix(mnemonic, "s_incperflevel") || has_prefix(mnemonic, "s_decperflevel"))
    return InstClass::Message;
  if (has_prefix(mnemonic, "s_load") || has_prefix(mnemonic, "s_store") ||
      has_prefix(mnemonic, "s_buffer_") || has_prefix(mnemonic, "s_dcache") ||
      has_prefix(mnemonic, "s_atomic") || has_prefix(mnemonic, "s_prefetch") ||
      has_prefix(mnemonic, "s_atc_probe"))
    return InstClass::ScalarMemory;
  if (has_prefix(mnemonic, "s_branch") || has_prefix(mnemonic, "s_cbranch") ||
      has_prefix(mnemonic, "s_setpc") || has_prefix(mnemonic, "s_swappc") ||
      has_prefix(mnemonic, "s_call") || has_prefix(mnemonic, "s_getpc"))
    return InstClass::Branch;
  return InstClass::ScalarAlu;
}

/// @brief Classify a vector (`v_`) mnemonic.
InstClass vector_class(std::string_view mnemonic, const rocjitsu::Instruction &inst) {
  if (inst.is_mfma() || has_prefix(mnemonic, "v_mfma") || has_prefix(mnemonic, "v_smfmac") ||
      has_prefix(mnemonic, "v_wmma") || has_prefix(mnemonic, "v_swmmac"))
    return InstClass::MatrixMultiply;
  if (is_transcendental_name(mnemonic))
    return InstClass::Transcendental;
  return InstClass::VectorAlu;
}

/// @brief Classify by mnemonic family, or nothing when no family matches.
std::optional<InstClass> class_by_family(std::string_view mnemonic) {
  if (has_prefix(mnemonic, "ds_"))
    return lds_class(mnemonic);
  if (has_prefix(mnemonic, "global_") || has_prefix(mnemonic, "flat_") ||
      has_prefix(mnemonic, "scratch_") || has_prefix(mnemonic, "buffer_") ||
      has_prefix(mnemonic, "tbuffer_") || has_prefix(mnemonic, "image_"))
    return vector_memory_class(mnemonic);
  if (has_prefix(mnemonic, "tensor_"))
    return InstClass::TensorMemory;
  if (has_prefix(mnemonic, "exp") || has_prefix(mnemonic, "export"))
    return InstClass::Export;
  return std::nullopt;
}

/// @brief Read one run of decimal digits, advancing @p cursor past it.
/// @returns How many digits were consumed.
std::size_t read_number(std::string_view text, std::size_t &cursor, std::uint64_t &value) {
  value = 0;
  std::size_t digits = 0;
  while (cursor < text.size() && std::isdigit(static_cast<unsigned char>(text[cursor])) != 0) {
    value = value * 10 + static_cast<std::uint64_t>(text[cursor] - '0');
    ++cursor;
    ++digits;
  }
  return digits;
}

} // namespace

InstClass classify(const rocjitsu::Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();

  // The scalar and vector prefixes cover the overwhelming majority of a kernel
  // and need the instruction's flags to resolve, so they come first and
  // separately from the name-only families.
  if (has_prefix(mnemonic, "s_"))
    return scalar_class(mnemonic, inst);
  if (has_prefix(mnemonic, "v_"))
    return vector_class(mnemonic, inst);
  if (const std::optional<InstClass> by_family = class_by_family(mnemonic))
    return *by_family;

  // Fall back to the flags rocjitsu does set, so an instruction whose name has
  // never been seen is still placed on the right unit when its metadata says
  // enough. Branch precedes the memory flag because an indirect branch carries
  // neither a recognizable name nor a memory role.
  if (inst.is_barrier())
    return InstClass::Barrier;
  if (inst.is_waitcnt())
    return InstClass::WaitCounter;
  if (inst.is_branch() ||
      (inst.flags() & (rocjitsu::INDIRECT_BRANCH | rocjitsu::INDIRECT_CALL)) != 0)
    return InstClass::Branch;
  if ((inst.flags() & rocjitsu::PROGRAM_TERMINATOR) != 0)
    return InstClass::Terminate;
  // A memory op of unknown direction is called a read: a read is the direction
  // the wavefront stalls on, and mis-costing a store as a load overstates the
  // dependency rather than dropping it.
  if (inst.is_memory_op())
    return InstClass::VectorMemoryRead;

  return InstClass::Unknown;
}

InstClass refine_with_memory_space(InstClass initial, bool is_local, bool is_load) {
  switch (initial) {
  case InstClass::VectorMemoryRead:
  case InstClass::VectorMemoryWrite:
  case InstClass::LdsRead:
  case InstClass::LdsWrite:
    if (is_local)
      return is_load ? InstClass::LdsRead : InstClass::LdsWrite;
    return is_load ? InstClass::VectorMemoryRead : InstClass::VectorMemoryWrite;
  default:
    // An atomic, a scalar access and a tensor transfer are already on the only
    // path they can take.
    return initial;
  }
}

std::uint64_t matrix_macs(std::string_view mnemonic) {
  std::size_t index = 0;
  while (index < mnemonic.size()) {
    if (std::isdigit(static_cast<unsigned char>(mnemonic[index])) == 0) {
      ++index;
      continue;
    }
    std::uint64_t dimension[3] = {0, 0, 0};
    std::size_t parsed = 0;
    std::size_t cursor = index;
    while (parsed < 3) {
      std::uint64_t value = 0;
      if (read_number(mnemonic, cursor, value) == 0)
        break;
      dimension[parsed++] = value;
      if (parsed < 3) {
        // The leading 32 of `f32` is a number that is not followed by an `x`,
        // which is exactly how it is told apart from a shape.
        if (cursor < mnemonic.size() && mnemonic[cursor] == 'x')
          ++cursor;
        else
          break;
      }
    }
    if (parsed == 3) {
      std::uint64_t macs = dimension[0] * dimension[1] * dimension[2];
      // A trailing block count multiplies the shape.
      if (cursor + 1 < mnemonic.size() && mnemonic[cursor] == '_') {
        std::size_t scan = cursor + 1;
        std::uint64_t blocks = 0;
        read_number(mnemonic, scan, blocks);
        if (blocks > 1 && scan < mnemonic.size() && mnemonic[scan] == 'b')
          macs *= blocks;
      }
      return macs;
    }
    index = cursor > index ? cursor : index + 1;
  }
  return 0;
}

MatrixType matrix_type_of(std::string_view mnemonic) {
  const std::size_t last = mnemonic.rfind('_');
  if (last == std::string_view::npos)
    return MatrixType::Default;
  const std::string_view tail = mnemonic.substr(last + 1);
  if (tail == "f64")
    return MatrixType::F64;
  if (tail == "f32" || tail == "xf32")
    return MatrixType::F32;
  if (tail == "f16")
    return MatrixType::F16;
  if (tail == "bf16")
    return MatrixType::BF16;
  if (tail == "fp8" || tail == "bf8" || tail == "f8" || tail == "fp4" || tail == "bf6" ||
      tail == "fp6")
    return MatrixType::Narrow;
  if (tail == "i8" || tail == "iu8" || tail == "i4" || tail == "iu4" || tail == "i32")
    return MatrixType::Integer;
  return MatrixType::Default;
}

} // namespace rocjitsu::timing
