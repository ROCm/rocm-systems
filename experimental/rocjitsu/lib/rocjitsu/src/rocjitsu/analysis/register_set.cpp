// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/register_set.h"

#include <algorithm>

namespace rocjitsu {

namespace {

template <size_t N> void set_range(std::bitset<N> &bits, size_t base, size_t width) {
  for (size_t i = 0; i < width && base + i < N; ++i)
    bits.set(base + i);
}

template <size_t N> void reset_range(std::bitset<N> &bits, size_t base, size_t width) {
  for (size_t i = 0; i < width && base + i < N; ++i)
    bits.reset(base + i);
}

template <size_t N>
[[nodiscard]] bool contains_range(const std::bitset<N> &bits, size_t base, size_t width) {
  for (size_t i = 0; i < width; ++i) {
    if (base + i >= N || !bits.test(base + i))
      return false;
  }
  return true;
}

template <size_t N> void subtract(std::bitset<N> &lhs, const std::bitset<N> &rhs) { lhs &= ~rhs; }

} // namespace

std::optional<size_t> RegisterSet::special_offset(RegisterRef ref) {
  switch (ref.cls) {
  case RegClass::EXEC:
    return kSpecialExecBase + ref.index;
  case RegClass::VCC:
    return kSpecialVccBase + ref.index;
  case RegClass::SCC:
    return kSpecialSccBase;
  case RegClass::M0:
    return kSpecialM0Base;
  case RegClass::FLAT_SCRATCH:
    return kSpecialFlatScratchBase + ref.index;
  case RegClass::TTMP:
    return kSpecialTtmpBase + ref.index;
  case RegClass::PC:
    return kSpecialPcBase;
  default:
    return std::nullopt;
  }
}

void RegisterSet::expand(RegisterRef ref) {
  const size_t width = std::max<size_t>(1, ref.width);
  switch (ref.cls) {
  case RegClass::SGPR:
    set_range(sgprs_, ref.index, width);
    break;
  case RegClass::VGPR:
    set_range(vgprs_, ref.index, width);
    break;
  case RegClass::ACC_VGPR:
    set_range(acc_vgprs_, ref.index, width);
    break;
  default:
    if (auto offset = special_offset(ref))
      set_range(special_, *offset, width);
    break;
  }
}

void RegisterSet::erase(RegisterRef ref) {
  const size_t width = std::max<size_t>(1, ref.width);
  switch (ref.cls) {
  case RegClass::SGPR:
    reset_range(sgprs_, ref.index, width);
    break;
  case RegClass::VGPR:
    reset_range(vgprs_, ref.index, width);
    break;
  case RegClass::ACC_VGPR:
    reset_range(acc_vgprs_, ref.index, width);
    break;
  default:
    if (auto offset = special_offset(ref))
      reset_range(special_, *offset, width);
    break;
  }
}

bool RegisterSet::contains(RegisterRef ref) const {
  const size_t width = std::max<size_t>(1, ref.width);
  switch (ref.cls) {
  case RegClass::SGPR:
    return contains_range(sgprs_, ref.index, width);
  case RegClass::VGPR:
    return contains_range(vgprs_, ref.index, width);
  case RegClass::ACC_VGPR:
    return contains_range(acc_vgprs_, ref.index, width);
  default:
    if (auto offset = special_offset(ref))
      return contains_range(special_, *offset, width);
    return false;
  }
}

bool RegisterSet::none() const {
  return sgprs_.none() && vgprs_.none() && acc_vgprs_.none() && special_.none();
}

bool RegisterSet::intersects(const RegisterSet &rhs) const {
  return (sgprs_ & rhs.sgprs_).any() || (vgprs_ & rhs.vgprs_).any() ||
         (acc_vgprs_ & rhs.acc_vgprs_).any() || (special_ & rhs.special_).any();
}

RegisterSet &RegisterSet::operator|=(const RegisterSet &rhs) {
  sgprs_ |= rhs.sgprs_;
  vgprs_ |= rhs.vgprs_;
  acc_vgprs_ |= rhs.acc_vgprs_;
  special_ |= rhs.special_;
  return *this;
}

RegisterSet &RegisterSet::operator&=(const RegisterSet &rhs) {
  sgprs_ &= rhs.sgprs_;
  vgprs_ &= rhs.vgprs_;
  acc_vgprs_ &= rhs.acc_vgprs_;
  special_ &= rhs.special_;
  return *this;
}

RegisterSet &RegisterSet::operator-=(const RegisterSet &rhs) {
  subtract(sgprs_, rhs.sgprs_);
  subtract(vgprs_, rhs.vgprs_);
  subtract(acc_vgprs_, rhs.acc_vgprs_);
  subtract(special_, rhs.special_);
  return *this;
}

} // namespace rocjitsu
