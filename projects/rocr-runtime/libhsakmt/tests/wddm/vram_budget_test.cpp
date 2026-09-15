// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#include "impl/wddm/vram_budget.h"

#include <cstdint>
#include <iostream>
#include <limits>

namespace {

int failures = 0;

void ExpectEqual(uint64_t actual, uint64_t expected) {
  if (actual == expected) return;

  std::cerr << "expected " << expected << ", got " << actual << '\n';
  ++failures;
}

}  // namespace

int main() {
  using wsl::thunk::AvailableVramBudget;

  ExpectEqual(AvailableVramBudget(900, 200, 1000), 700);
  ExpectEqual(AvailableVramBudget(1200, 100, 1000), 1000);
  ExpectEqual(AvailableVramBudget(100, 100, 1000), 0);
  ExpectEqual(AvailableVramBudget(100, 200, 1000), 0);
  ExpectEqual(AvailableVramBudget(900, 200, 0), 0);
  ExpectEqual(AvailableVramBudget(std::numeric_limits<uint64_t>::max(),
                                  std::numeric_limits<uint64_t>::max() - 1,
                                  std::numeric_limits<uint64_t>::max()),
              1);
  return failures == 0 ? 0 : 1;
}
