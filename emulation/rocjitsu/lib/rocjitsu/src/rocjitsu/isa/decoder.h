// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file decoder.h
/// @brief Instruction decoder with optional pool-backed allocation.

#ifndef ROCJITSU_ISA_DECODER_H_
#define ROCJITSU_ISA_DECODER_H_

#include "rocjitsu/base/api.h"
#include "rocjitsu/code/rj_code.h"
#include "util/arena_alloc.h"

#include <cstdint>
#include <memory>

namespace rocjitsu {

class Instruction;

/// @brief Instruction decoder with optional pool allocator.
///
/// By default, decoded instructions are heap-allocated.  Call
/// ``enable_pool()`` to route Instruction::operator new/delete through
/// the decoder's O(1) free-list pool.  Only enable the pool when all
/// decoded instructions will be deleted before the decoder is destroyed
/// (e.g., the ComputeUnit simulation loop).
class Decoder {
public:
  virtual ~Decoder();

  /// @brief Decode a binary instruction.
  /// @param[in] inst Pointer to the binary instruction encoding.
  /// @returns Decoded Instruction pointer (pool or heap allocated).
  virtual Instruction *decode(const rj_code_binary_inst_t *inst) = 0;

  /// @brief Decode a binary instruction and record its source text offset.
  ///
  /// @details The generated ISA decoders construct instructions from raw
  /// encoding words. This overload keeps source-location assignment in the
  /// decoder API, which is the boundary where callers know both the encoding and
  /// its location in a larger text stream.
  /// @param[in] inst Pointer to the binary instruction encoding.
  /// @param[in] src_loc Source byte offset in the decoded text stream.
  /// @returns Decoded Instruction pointer (pool or heap allocated).
  Instruction *decode(const rj_code_binary_inst_t *inst, uint64_t src_loc);

  /// @brief Create a decoder for the given architecture.
  static std::unique_ptr<Decoder> create(rj_code_arch_t arch);

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
  void disable_pool() { deactivate_pool(); }

protected:
  using Pool = util::ArenaAlloc<512, 128>;
  using AllocFn = void *(*)(void *, size_t);
  using DeallocFn = void (*)(void *, void *);

  static void activate_pool(AllocFn alloc, DeallocFn dealloc, void *pool);
  static void deactivate_pool();

  Pool pool_;
};

/// @brief ISA-parameterized decoder.
template <typename Isa> class IsaDecoder final : public Decoder {
public:
  using Decoder::decode;

  Instruction *decode(const rj_code_binary_inst_t *inst) override {
    auto result = Isa::Decoder::decode(inst);
    return result.release();
  }
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_DECODER_H_
