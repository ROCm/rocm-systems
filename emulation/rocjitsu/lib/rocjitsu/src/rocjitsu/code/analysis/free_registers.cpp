// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/free_registers.h"

#include "util/bit.h"

#include <bit>
#include <cassert>
#include <cstddef>

namespace rocjitsu {

std::optional<uint16_t> find_free_run(const RegisterSet &unavailable, RegClass cls, uint8_t count,
                                      uint16_t search_start, uint16_t base_alignment,
                                      uint32_t bound) {
  assert(count > 0 && "Must request at least one register");
  // util::align_up asserts this too, but only in a debug build of util. A
  // non-power-of-two would make align_up's mask arithmetic stop being a rounding.
  assert(std::has_single_bit(base_alignment) && "Register tuple alignment must be a power of two");
  // RegisterSet answers false for every class it does not model, so searching
  // one of those would find the entire space free and hand back base 0.
  assert((cls == RegClass::SGPR || cls == RegClass::VGPR || cls == RegClass::ACC_VGPR) &&
         "free-register search requires a class RegisterSet tracks");
  // Widened to size_t before the bound test so a run ending at the top of the
  // 16-bit index space cannot wrap into a false fit.
  for (size_t base =
           util::align_up(static_cast<size_t>(search_start), static_cast<size_t>(base_alignment));
       base + count <= bound; base += base_alignment) {
    if (!unavailable.intersects({cls, static_cast<uint16_t>(base), count})) {
      return static_cast<uint16_t>(base);
    }
  }
  return std::nullopt;
}

std::optional<uint16_t> find_free_sgpr(const RegisterSet &unavailable, uint32_t bound,
                                       uint16_t search_start) {
  return find_free_run(unavailable, RegClass::SGPR, /*count=*/1, search_start,
                       /*base_alignment=*/1, bound);
}

std::optional<uint16_t> find_free_sgpr_pair(const RegisterSet &unavailable, uint32_t bound,
                                            uint16_t search_start) {
  return find_free_run(unavailable, RegClass::SGPR, /*count=*/2, search_start,
                       /*base_alignment=*/2, bound);
}

} // namespace rocjitsu
