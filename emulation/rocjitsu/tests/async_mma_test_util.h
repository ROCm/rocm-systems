// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"

#include <array>
#include <span>

namespace mma_test {
using namespace rocjitsu;
// Exercise the production ticket API with caller-owned instructions. Keep the
// last instruction on the issuer and join every accepted job before returning.
inline unsigned execute_independent(amdgpu::matrix_coexecution::SharedPool &pool,
                                    std::span<Instruction *> instructions, void *context) {
  assert(instructions.size() <= 8);
  std::array<amdgpu::matrix_coexecution::SharedPool::Ticket, 8> tickets{};
  std::exception_ptr error;
  size_t error_index = instructions.size();
  unsigned submitted = 0;
  for (size_t i = 0; i != instructions.size(); ++i) {
    if (i + 1 < instructions.size() && (tickets[i] = pool.submit(*instructions[i], context))) {
      ++submitted;
      continue;
    }
    try {
      instructions[i]->execute(*instructions[i], context);
    } catch (...) {
      error = std::current_exception();
      error_index = i;
      break;
    }
  }
  for (size_t i = 0; i != instructions.size(); ++i) {
    if (tickets[i]) {
      auto helper_error = pool.finish(tickets[i]);
      if (helper_error && i < error_index) {
        error = helper_error;
        error_index = i;
      }
    }
  }
  if (error)
    std::rethrow_exception(error);
  return submitted;
}
} // namespace mma_test
