// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/register_set.h"

#include <algorithm>
#include <bit>

namespace rocjitsu {

namespace {

// Mark a contiguous range of register indices in one register-class bitset.
template <size_t N> void set_range(std::bitset<N> &bits, size_t base, size_t width) {
  for (size_t i = 0; i < width && base + i < N; ++i)
    bits.set(base + i);
}

// Clear a contiguous range of register indices in one register-class bitset.
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

template <size_t N>
[[nodiscard]] bool intersects_range(const std::bitset<N> &bits, size_t base, size_t width) {
  for (size_t i = 0; i < width; ++i) {
    if (base + i < N && bits.test(base + i))
      return true;
  }
  return false;
}

template <size_t N> void subtract(std::bitset<N> &lhs, const std::bitset<N> &rhs) { lhs &= ~rhs; }

} // namespace

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
  case RegClass::EXEC:
  case RegClass::VCC:
  case RegClass::SCC:
  case RegClass::M0:
  case RegClass::FLAT_SCRATCH:
  case RegClass::PC:
    // Special singleton: index/width are meaningless, so just set its bit.
    special_regs_ |= special_bit(ref.cls);
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
  case RegClass::EXEC:
  case RegClass::VCC:
  case RegClass::SCC:
  case RegClass::M0:
  case RegClass::FLAT_SCRATCH:
  case RegClass::PC:
    special_regs_ &= static_cast<uint16_t>(~special_bit(ref.cls));
    break;
  }
}

void RegisterSet::clear_class(RegClass cls) {
  switch (cls) {
  case RegClass::SGPR:
    sgprs_.reset();
    break;
  case RegClass::VGPR:
    vgprs_.reset();
    break;
  case RegClass::ACC_VGPR:
    acc_vgprs_.reset();
    break;
  case RegClass::EXEC:
  case RegClass::VCC:
  case RegClass::SCC:
  case RegClass::M0:
  case RegClass::FLAT_SCRATCH:
  case RegClass::PC:
    special_regs_ &= static_cast<uint16_t>(~special_bit(cls));
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
  case RegClass::EXEC:
  case RegClass::VCC:
  case RegClass::SCC:
  case RegClass::M0:
  case RegClass::FLAT_SCRATCH:
  case RegClass::PC:
    return (special_regs_ & special_bit(ref.cls)) != 0;
  }
  return false; // unreachable for a valid RegClass; a new class trips -Wswitch first
}

bool RegisterSet::none() const {
  return sgprs_.none() && vgprs_.none() && acc_vgprs_.none() && special_regs_ == 0;
}

size_t RegisterSet::size() const {
  return ordinary_size() + static_cast<size_t>(std::popcount(special_regs_));
}

size_t RegisterSet::ordinary_size() const {
  return sgprs_.count() + vgprs_.count() + acc_vgprs_.count();
}

bool RegisterSet::intersects(RegisterRef ref) const {
  const size_t width = std::max<size_t>(1, ref.width);
  switch (ref.cls) {
  case RegClass::SGPR:
    return intersects_range(sgprs_, ref.index, width);
  case RegClass::VGPR:
    return intersects_range(vgprs_, ref.index, width);
  case RegClass::ACC_VGPR:
    return intersects_range(acc_vgprs_, ref.index, width);
  default:
    return false;
  }
}

bool RegisterSet::intersects(const RegisterSet &rhs) const {
  return (sgprs_ & rhs.sgprs_).any() || (vgprs_ & rhs.vgprs_).any() ||
         (acc_vgprs_ & rhs.acc_vgprs_).any() || (special_regs_ & rhs.special_regs_) != 0;
}

RegisterSet &RegisterSet::operator|=(const RegisterSet &rhs) {
  sgprs_ |= rhs.sgprs_;
  vgprs_ |= rhs.vgprs_;
  acc_vgprs_ |= rhs.acc_vgprs_;
  special_regs_ |= rhs.special_regs_;
  return *this;
}

RegisterSet &RegisterSet::operator&=(const RegisterSet &rhs) {
  sgprs_ &= rhs.sgprs_;
  vgprs_ &= rhs.vgprs_;
  acc_vgprs_ &= rhs.acc_vgprs_;
  special_regs_ &= rhs.special_regs_;
  return *this;
}

RegisterSet &RegisterSet::operator-=(const RegisterSet &rhs) {
  subtract(sgprs_, rhs.sgprs_);
  subtract(vgprs_, rhs.vgprs_);
  subtract(acc_vgprs_, rhs.acc_vgprs_);
  special_regs_ &= static_cast<uint16_t>(~rhs.special_regs_);
  return *this;
}

} // namespace rocjitsu
