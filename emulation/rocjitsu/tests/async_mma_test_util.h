// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"

#include <gtest/gtest.h>

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
  unsigned submitted = 0;
  for (size_t i = 0; i != instructions.size(); ++i) {
    if (i + 1 < instructions.size() && (tickets[i] = pool.submit(*instructions[i], context))) {
      ++submitted;
      continue;
    }
    instructions[i]->execute(*instructions[i], context);
  }
  for (auto ticket : tickets)
    if (ticket) {
      EXPECT_TRUE(pool.finish(ticket).succeeded());
    }
  return submitted;
}
} // namespace mma_test
