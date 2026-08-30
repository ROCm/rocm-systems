/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

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
