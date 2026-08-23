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

/// @brief Instruction decoder with optional worker-local pool allocation.
///
/// By default, decoded instructions are heap-allocated. Simulation workers
/// enable one shared grow-on-demand pool for all decoders on that worker. Every
/// pool-backed instruction must be destroyed on the allocating worker before
/// the worker pool is disabled.
class Decoder {
public:
  // CDNA5 carry instructions with five inline operands are 672 bytes. Grow in
  // slabs so the many CUs sharing a simulation thread can cover peak in-flight
  // demand without reserving a large fixed buffer in every decoder.
  using Pool = util::GrowingArenaAlloc<768, 128>;

  virtual ~Decoder() = default;

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

  /// @brief Enable the current worker thread's shared decoder pool.
  /// @details Compatibility wrapper for enable_thread_pool(). The pool belongs
  /// to the thread rather than this Decoder instance.
  void enable_pool() { enable_thread_pool(); }

  /// @brief Enable the pool shared by decoders on the current worker thread.
  ///
  /// Compute units are constructed before the simulation worker starts. All
  /// CUs assigned to one worker therefore share this pool, allowing decoded
  /// instructions to remain live while other CUs or deferred pipelines run.
  static void enable_thread_pool();

  /// @brief Disable and release the current worker thread's shared pool.
  /// @warning Every instruction allocated from the pool must already be destroyed.
  static void disable_thread_pool();

  /// @brief Compatibility wrapper for disable_thread_pool().
  void disable_pool() { disable_thread_pool(); }

protected:
  using AllocFn = void *(*)(void *, size_t);
  using DeallocFn = void (*)(void *, void *);

  static void activate_pool(AllocFn alloc, DeallocFn dealloc, void *pool);
  static Result validate_instruction_operands(const Instruction &inst,
                                              const DecodeErrorEmitter &emit_error);
};

/// @brief ISA-parameterized decoder.
template <typename Isa> class IsaDecoder final : public Decoder {
public:
  using Decoder::decode;

  explicit IsaDecoder(const IsaExecutionBackend *execution_backend = nullptr)
      : execution_backend_(execution_backend) {}

  DecodeResult decode(const rj_code_binary_inst_t *inst,
                      const DecodeErrorEmitter &emit_error) override {
    ScopedIsaExecutionBackend scope(execution_backend_);
    DecodeResult result = Isa::Decoder::decode(inst, emit_error);
    if (result.failed()) [[unlikely]]
      return Result::failure();
    if (validate_instruction_operands(*result.value(), emit_error).failed()) [[unlikely]]
      return Result::failure();
    return result;
  }

  std::size_t max_instruction_words() const override { return Isa::Decoder::kMaxInstructionWords; }

private:
  const IsaExecutionBackend *execution_backend_;
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_DECODER_H_
