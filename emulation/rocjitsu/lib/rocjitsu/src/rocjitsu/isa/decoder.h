// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file decoder.h
/// @brief Instruction decoder with optional pool-backed allocation.

#ifndef ROCJITSU_ISA_DECODER_H_
#define ROCJITSU_ISA_DECODER_H_

#include "rocjitsu/base/api.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decode_result.h"
#include "rocjitsu/isa/execution_backend.h"
#include "util/arena_alloc.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace rocjitsu {

class Instruction;
class IsaTargetRegistry;
struct IsaExecutionBackend;

/// @brief Instruction decoder with optional pool allocator.
///
/// By default, decoded instructions are heap-allocated.  Call
/// ``enable_pool()`` to route Instruction::operator new/delete through
/// the decoder's O(1) free-list pool.  Only enable the pool when all
/// decoded instructions will be deleted before the decoder is destroyed
/// (e.g., the ComputeUnit simulation loop).
class Decoder {
public:
  using Pool = util::ArenaAlloc<512, 128>;

  virtual ~Decoder();

  /// @brief Decode a binary instruction.
  /// @param[in] inst Pointer to the binary instruction encoding.
  /// @param[in] emit_error Diagnostic destination for rejected encodings.
  /// @returns A decoded instruction (pool or heap allocated), or failure.
  virtual DecodeResult decode(const rj_code_binary_inst_t *inst,
                              const DecodeErrorEmitter &emit_error) = 0;

  /// @brief Decode silently, for expected speculative rejection.
  DecodeResult decode(const rj_code_binary_inst_t *inst) {
    return decode(inst, DecodeErrorEmitter{});
  }

  /// @brief Maximum encoded instruction width and decode lookahead, in 32-bit words.
  /// @returns A nonzero bound covering both every raw-pointer read and the size of every
  /// successfully decoded instruction.
  virtual std::size_t max_instruction_words() const = 0;

  /// @brief Decode a binary instruction and record its source text offset.
  ///
  /// @details The generated ISA decoders construct instructions from raw
  /// encoding words. This overload keeps source-location assignment in the
  /// decoder API, which is the boundary where callers know both the encoding and
  /// its location in a larger text stream.
  /// @param[in] inst Pointer to the binary instruction encoding.
  /// @param[in] src_loc Source byte offset in the decoded text stream.
  /// @param[in] emit_error Diagnostic destination for rejected encodings.
  /// @returns A decoded instruction (pool or heap allocated), or failure.
  DecodeResult decode(const rj_code_binary_inst_t *inst, uint64_t src_loc,
                      const DecodeErrorEmitter &emit_error = {});

  /// @brief Create a decoder for the given architecture.
  static std::unique_ptr<Decoder> create(rj_code_arch_t arch);

  /// @brief Create a decoder from an explicitly scoped registry and open ID.
  static std::unique_ptr<Decoder> create(const IsaTargetRegistry &registry,
                                         std::string_view target_id);

  /// @brief Create a decoder from a built-in architecture in a scoped registry.
  static std::unique_ptr<Decoder> create(const IsaTargetRegistry &registry, rj_code_arch_t arch);

  /// @brief Create a decoder for a concrete public GPU target.
  static std::unique_ptr<Decoder> create(const IsaTargetRegistry &registry,
                                         rj_code_target_id_t target);

  /// @brief Enable pool allocation for decoded instructions.
  ///
  /// When active, Instruction::operator new/delete route through the
  /// decoder's pool for O(1) alloc/free.  Only enable when the caller
  /// guarantees all instructions will be deleted before the decoder
  /// is destroyed (e.g., the ComputeUnit hot path).
  void enable_pool() {
    activate_pool([](void *p, size_t s) -> void * { return static_cast<Pool *>(p)->allocate(s); },
                  [](void *p, void *ptr) { static_cast<Pool *>(p)->deallocate(ptr); }, &pool_);
  }

  /// @brief Disable pool allocation; future allocations use the heap.
  void disable_pool();

protected:
  using AllocFn = void *(*)(void *, size_t);
  using DeallocFn = void (*)(void *, void *);

  static void activate_pool(AllocFn alloc, DeallocFn dealloc, void *pool);
  static Result validate_instruction_operands(const Instruction &inst,
                                              const DecodeErrorEmitter &emit_error);

  Pool pool_;
};

/// @brief ISA-parameterized decoder.
template <typename Isa> class IsaDecoder final : public Decoder {
public:
  using Decoder::decode;

  explicit IsaDecoder(const IsaExecutionBackend *execution_backend = nullptr,
                      uint64_t isa_features = 0)
      : execution_backend_(execution_backend), isa_features_(isa_features) {}

  DecodeResult decode(const rj_code_binary_inst_t *inst,
                      const DecodeErrorEmitter &emit_error) override {
    ScopedIsaExecutionBackend scope(execution_backend_);
    DecodeResult result = Isa::Decoder::decode(inst, emit_error);
    if (result.failed()) [[unlikely]]
      return Result::failure();
    const uint64_t missing_features = result.value()->required_isa_features() & ~isa_features_;
    if (missing_features != 0) [[unlikely]]
      return emit_error.emit() << "instruction requires unavailable target ISA features (mask "
                               << missing_features << ")";
    if (validate_instruction_operands(*result.value(), emit_error).failed()) [[unlikely]]
      return Result::failure();
    return result;
  }

  std::size_t max_instruction_words() const override { return Isa::Decoder::kMaxInstructionWords; }

private:
  const IsaExecutionBackend *execution_backend_;
  const uint64_t isa_features_;
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_DECODER_H_
