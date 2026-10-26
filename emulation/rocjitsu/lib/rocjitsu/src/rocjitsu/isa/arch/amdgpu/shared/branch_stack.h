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

/// The manual documents a six-entry stack. Running the smaller side first keeps
/// any valid program within it, so a deeper push means a malformed program.
inline constexpr uint32_t kBranchStackDepth = 6;

/// @brief Warn when the entry at @p sgpr does not fit in the wave's SGPRs.
/// @details Out-of-allocation SGPR writes are dropped and reads return zero, so
/// the matching join would resume at address 0 with EXEC cleared.
inline void check_branch_stack_entry_allocated(const Wavefront &wf, uint32_t sgpr) {
  if (!wf.sgpr_alloc().contains(sgpr, 4))
    wf.report_undefined_behavior("branch stack entry lies outside the wave's SGPR allocation");
}

/// @brief Push @p entry at MODE.CSP, then increment CSP.
inline void push_branch_stack(Wavefront &wf, BranchStackEntry entry) {
  const uint32_t csp = wf.cond_stack_ptr();
  if (csp >= kBranchStackDepth)
    wf.report_undefined_behavior("fork pushes past the six-entry branch stack");
  const uint32_t sgpr = branch_stack_entry_sgpr(wf, csp);
  check_branch_stack_entry_allocated(wf, sgpr);
  RegisterAccess regs(wf);
  regs.write_sgpr64(sgpr, entry.lanes);
  regs.write_sgpr64(sgpr + 2, entry.resume_pc);
  wf.set_cond_stack_ptr(csp + 1);
}

/// @brief Decrement MODE.CSP, then pop the entry it points to.
inline BranchStackEntry pop_branch_stack(Wavefront &wf) {
  // Read CSP back after the decrement so it wraps as a 3-bit field.
  wf.set_cond_stack_ptr(wf.cond_stack_ptr() - 1);
  const uint32_t sgpr = branch_stack_entry_sgpr(wf, wf.cond_stack_ptr());
  check_branch_stack_entry_allocated(wf, sgpr);
  RegisterAccess regs(wf);
  return {regs.read_sgpr64(sgpr), regs.read_sgpr64(sgpr + 2)};
}

/// @brief Execute S_CBRANCH_G_FORK or S_CBRANCH_I_FORK.
/// @details Updates EXEC and the branch stack; the caller must still set the PC
/// to the returned address to complete the instruction.
/// @param compare Per-lane branch condition: lanes whose bit is set take the
/// branch. Only active lanes are considered.
/// @param target Resolved branch target PC.
/// @param next_pc Fall-through PC, the address of the instruction after the fork.
/// @returns Address of the next instruction to execute.
[[nodiscard]] inline uint64_t cbranch_fork(Wavefront &wf, uint64_t compare, uint64_t target,
                                           uint64_t next_pc) {
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
/// @details Updates EXEC and the branch stack when it resumes a waiting path;
/// the caller must still set the PC to the returned address. Falling through
/// leaves EXEC as the last path set it; the program restores the pre-fork mask.
/// @param saved_csp MODE.CSP value the program saved before the matching fork.
/// @param next_pc Fall-through PC, the address of the instruction after the join.
/// @returns Address of the next instruction to execute.
[[nodiscard]] inline uint64_t cbranch_join(Wavefront &wf, uint32_t saved_csp, uint64_t next_pc) {
  if (wf.cond_stack_ptr() == saved_csp) // second arrival
    return next_pc;
  // Below the saved value, no entry of the matching fork is left to pop.
  if (wf.cond_stack_ptr() < saved_csp)
    wf.report_undefined_behavior("join finds MODE.CSP below its saved value");

  // first arrival
  const BranchStackEntry waiting = pop_branch_stack(wf);
  wf.write_exec(waiting.lanes);
  return waiting.resume_pc;
}

} // namespace rocjitsu::amdgpu
