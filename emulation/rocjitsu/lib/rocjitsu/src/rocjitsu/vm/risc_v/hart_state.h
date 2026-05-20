// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_RISC_V_HART_STATE_H_
#define ROCJITSU_VM_RISC_V_HART_STATE_H_

#include "rocjitsu/vm/thread_context.h"

#include <cstdint>
#include <unordered_map>

namespace rocjitsu {
namespace risc_v {

class Memory;

/// @brief RISC-V hart architectural state.
///
/// Derives from ThreadContext so that execute() methods can
/// static_cast the ThreadContext& parameter to HartState&.
struct HartState : public ThreadContext {
  int64_t xreg[32] = {};
  uint64_t freg[32] = {};
  // pc is inherited from ThreadContext.
  uint64_t next_pc = 0;
  bool halted = false;

  bool reservation_valid = false;
  uint64_t reservation_addr = 0;

  std::unordered_map<uint16_t, uint64_t> csrs;

  int64_t read_xreg(int idx) const { return (idx == 0) ? 0 : xreg[idx]; }
  void write_xreg(int idx, int64_t val) {
    if (idx != 0)
      xreg[idx] = val;
  }

  uint64_t read_freg(int idx) const { return freg[idx]; }
  void write_freg(int idx, uint64_t val) { freg[idx] = val; }

  uint64_t read_csr(uint16_t addr) const;
  void write_csr(uint16_t addr, uint64_t val);
};

/// @brief Convenience: cast a ThreadContext& to HartState*.
inline HartState *as_hart(ThreadContext &ctx) { return static_cast<HartState *>(&ctx); }

/// @brief Get the current thread's memory.
Memory *current_memory();

/// @brief Set the current thread's memory.
void set_current_memory(Memory *m);

// Sign-extend 32-bit result to 64 bits (common for W-instructions).
inline int64_t sext32(int32_t val) { return static_cast<int64_t>(val); }

// NaN-box a 32-bit float value into a 64-bit FP register.
inline uint64_t nan_box(uint32_t f) { return 0xFFFFFFFF00000000ULL | f; }

// Unbox a NaN-boxed 32-bit float; returns canonical NaN if not properly boxed.
inline uint32_t unbox(uint64_t d) {
  return (d >> 32) == 0xFFFFFFFF ? static_cast<uint32_t>(d) : 0x7FC00000u;
}

} // namespace risc_v
} // namespace rocjitsu

#endif // ROCJITSU_VM_RISC_V_HART_STATE_H_
