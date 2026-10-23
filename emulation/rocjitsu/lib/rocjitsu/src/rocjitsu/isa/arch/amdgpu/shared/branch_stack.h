// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file branch_stack.h
/// @brief CDNA1-4 conditional branch stack for S_CBRANCH_{G,I}_FORK and S_CBRANCH_JOIN.

#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <bit>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// A waiting group of lanes and where they resume.
struct BranchStackEntry {
  uint64_t lanes;
  uint64_t resume_pc;
};

/// @brief Physical SGPR holding branch stack entry @p csp.
/// @details Entry n occupies the wave's s[4n..4n+3]:
///   s[4n:4n+1]   BranchStackEntry::lanes
///   s[4n+2:4n+3] BranchStackEntry::resume_pc
inline uint32_t branch_stack_entry_sgpr(const Wavefront &wf, uint32_t csp) {
  return wf.sgpr_alloc().base + csp * 4;
}

/// @brief Push @p entry at MODE.CSP, then increment CSP.
inline void push_branch_stack(Wavefront &wf, BranchStackEntry entry) {
  const uint32_t sgpr = branch_stack_entry_sgpr(wf, wf.cond_stack_ptr());
  RegisterAccess regs(wf);
  regs.write_sgpr64(sgpr, entry.lanes);
  regs.write_sgpr64(sgpr + 2, entry.resume_pc);
  wf.set_cond_stack_ptr(wf.cond_stack_ptr() + 1);
}

/// @brief Decrement MODE.CSP, then pop the entry it points to.
inline BranchStackEntry pop_branch_stack(Wavefront &wf) {
  // Read CSP back after the decrement so it wraps as a 3-bit field.
  wf.set_cond_stack_ptr(wf.cond_stack_ptr() - 1);
  const uint32_t sgpr = branch_stack_entry_sgpr(wf, wf.cond_stack_ptr());
  RegisterAccess regs(wf);
  return {regs.read_sgpr64(sgpr), regs.read_sgpr64(sgpr + 2)};
}

/// @brief Execute S_CBRANCH_G_FORK or S_CBRANCH_I_FORK.
/// @param compare Lanes that take the branch (S0).
/// @returns Address of the next instruction to execute.
inline uint64_t cbranch_fork(Wavefront &wf, uint64_t compare, uint64_t target, uint64_t next_pc) {
  const uint64_t exec = wf.exec_raw();
  const uint64_t pass = compare & exec;
  const uint64_t fail = ~compare & exec;

  // Case 1: Every lane takes the same route
  if (pass == exec)
    return target;
  if (fail == exec)
    return next_pc;

  // Case 2: Run the side with fewer active lanes first (tie is broken by the pass side)
  const bool fail_runs_first = std::popcount(fail) < std::popcount(pass);
  const BranchStackEntry running =
      fail_runs_first ? BranchStackEntry{fail, next_pc} : BranchStackEntry{pass, target};
  const BranchStackEntry waiting =
      fail_runs_first ? BranchStackEntry{pass, target} : BranchStackEntry{fail, next_pc};
  push_branch_stack(wf, waiting);
  wf.write_exec(running.lanes);
  return running.resume_pc;
}

/// @brief Execute S_CBRANCH_JOIN.
/// @param saved_csp MODE.CSP captured before the matching fork (S0).
/// @returns Address of the next instruction to execute. Falling through leaves
/// EXEC as the last path set it; the program restores the pre-fork mask.
inline uint64_t cbranch_join(Wavefront &wf, uint32_t saved_csp, uint64_t next_pc) {
  if (wf.cond_stack_ptr() == saved_csp) // second arrival
    return next_pc;

  // first arrival
  const BranchStackEntry waiting = pop_branch_stack(wf);
  wf.write_exec(waiting.lanes);
  return waiting.resume_pc;
}

} // namespace rocjitsu::amdgpu
