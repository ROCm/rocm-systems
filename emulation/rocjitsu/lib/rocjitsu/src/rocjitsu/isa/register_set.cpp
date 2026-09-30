// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/register_set.h"

#include <algorithm>
#include <bit>

namespace rocjitsu {

namespace {

// Mark a contiguous range of register indices in one register-class bitset.
template <size_t N, size_t Words>
void set_range(std::array<std::bitset<64>, Words> &bits, size_t base, size_t width) {
  for (size_t i = 0; i < width && base + i < N; ++i)
    bits[(base + i) / 64].set((base + i) % 64);
}

// Clear a contiguous range of register indices in one register-class bitset.
template <size_t N, size_t Words>
void reset_range(std::array<std::bitset<64>, Words> &bits, size_t base, size_t width) {
  for (size_t i = 0; i < width && base + i < N; ++i)
    bits[(base + i) / 64].reset((base + i) % 64);
}

template <size_t N, size_t Words>
[[nodiscard]] bool contains_range(const std::array<std::bitset<64>, Words> &bits, size_t base,
                                  size_t width) {
  for (size_t i = 0; i < width; ++i) {
    if (base + i >= N || !bits[(base + i) / 64].test((base + i) % 64))
      return false;
  }
  return true;
}

template <size_t N, size_t Words>
[[nodiscard]] bool intersects_range(const std::array<std::bitset<64>, Words> &bits, size_t base,
                                    size_t width) {
  for (size_t i = 0; i < width; ++i) {
    if (base + i < N && bits[(base + i) / 64].test((base + i) % 64))
      return true;
  }
  return false;
}

template <size_t Words, typename Operation>
void combine(std::array<std::bitset<64>, Words> &lhs, const std::array<std::bitset<64>, Words> &rhs,
             Operation operation) {
  for (size_t word = 0; word < Words; ++word)
    lhs[word] = operation(lhs[word], rhs[word]);
}

template <size_t Words> bool bits_none(const std::array<std::bitset<64>, Words> &bits) {
  return std::ranges::all_of(bits, [](const auto &word) { return word.none(); });
}

template <size_t Words> size_t count(const std::array<std::bitset<64>, Words> &bits) {
  size_t result = 0;
  for (const auto &word : bits)
    result += word.count();
  return result;
}

template <size_t Words>
bool intersects_bits(const std::array<std::bitset<64>, Words> &lhs,
                     const std::array<std::bitset<64>, Words> &rhs) {
  for (size_t word = 0; word < Words; ++word)
    if ((lhs[word] & rhs[word]).any())
      return true;
  return false;
}

} // namespace

void RegisterSet::expand(RegisterRef ref) {
  const size_t width = std::max<size_t>(1, ref.width);
  switch (ref.cls) {
  case RegClass::SGPR:
    set_range<REGISTER_SET_MAX_SGPRS>(sgprs_, ref.index, width);
    break;
  case RegClass::VGPR:
    set_range<REGISTER_SET_MAX_VGPRS>(vgprs_, ref.index, width);
    break;
  case RegClass::ACC_VGPR:
    set_range<REGISTER_SET_MAX_ACC_VGPRS>(acc_vgprs_, ref.index, width);
    break;
  case RegClass::TTMP:
    // Trap temporaries are known to the ISA but not tracked by this set: no
    // bitset and no mask bit, so index/width are dropped (as they are on the
    // general def/use path). Explicit arm keeps the switch exhaustive.
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
    reset_range<REGISTER_SET_MAX_SGPRS>(sgprs_, ref.index, width);
    break;
  case RegClass::VGPR:
    reset_range<REGISTER_SET_MAX_VGPRS>(vgprs_, ref.index, width);
    break;
  case RegClass::ACC_VGPR:
    reset_range<REGISTER_SET_MAX_ACC_VGPRS>(acc_vgprs_, ref.index, width);
    break;
  case RegClass::TTMP: // untracked: nothing to clear
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
    sgprs_ = {};
    break;
  case RegClass::VGPR:
    vgprs_ = {};
    break;
  case RegClass::ACC_VGPR:
    acc_vgprs_ = {};
    break;
  case RegClass::TTMP: // untracked: nothing to clear
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
    return contains_range<REGISTER_SET_MAX_SGPRS>(sgprs_, ref.index, width);
  case RegClass::VGPR:
    return contains_range<REGISTER_SET_MAX_VGPRS>(vgprs_, ref.index, width);
  case RegClass::ACC_VGPR:
    return contains_range<REGISTER_SET_MAX_ACC_VGPRS>(acc_vgprs_, ref.index, width);
  case RegClass::TTMP: // untracked: never present
    return false;
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
  return bits_none(sgprs_) && bits_none(vgprs_) && bits_none(acc_vgprs_) && special_regs_ == 0;
}

size_t RegisterSet::size() const {
  return ordinary_size() + static_cast<size_t>(std::popcount(special_regs_));
}

size_t RegisterSet::ordinary_size() const {
  return count(sgprs_) + count(vgprs_) + count(acc_vgprs_);
}

bool RegisterSet::intersects(RegisterRef ref) const {
  const size_t width = std::max<size_t>(1, ref.width);
  switch (ref.cls) {
  case RegClass::SGPR:
    return intersects_range<REGISTER_SET_MAX_SGPRS>(sgprs_, ref.index, width);
  case RegClass::VGPR:
    return intersects_range<REGISTER_SET_MAX_VGPRS>(vgprs_, ref.index, width);
  case RegClass::ACC_VGPR:
    return intersects_range<REGISTER_SET_MAX_ACC_VGPRS>(acc_vgprs_, ref.index, width);
  default:
    return false;
  }
}

bool RegisterSet::intersects(const RegisterSet &rhs) const {
  return intersects_bits(sgprs_, rhs.sgprs_) || intersects_bits(vgprs_, rhs.vgprs_) ||
         intersects_bits(acc_vgprs_, rhs.acc_vgprs_) || (special_regs_ & rhs.special_regs_) != 0;
}

RegisterSet &RegisterSet::operator|=(const RegisterSet &rhs) {
  combine(sgprs_, rhs.sgprs_, [](auto a, auto b) { return a | b; });
  combine(vgprs_, rhs.vgprs_, [](auto a, auto b) { return a | b; });
  combine(acc_vgprs_, rhs.acc_vgprs_, [](auto a, auto b) { return a | b; });
  special_regs_ |= rhs.special_regs_;
  return *this;
}

RegisterSet &RegisterSet::operator&=(const RegisterSet &rhs) {
  combine(sgprs_, rhs.sgprs_, [](auto a, auto b) { return a & b; });
  combine(vgprs_, rhs.vgprs_, [](auto a, auto b) { return a & b; });
  combine(acc_vgprs_, rhs.acc_vgprs_, [](auto a, auto b) { return a & b; });
  special_regs_ &= rhs.special_regs_;
  return *this;
}

RegisterSet &RegisterSet::operator-=(const RegisterSet &rhs) {
  combine(sgprs_, rhs.sgprs_, [](auto a, auto b) { return a & ~b; });
  combine(vgprs_, rhs.vgprs_, [](auto a, auto b) { return a & ~b; });
  combine(acc_vgprs_, rhs.acc_vgprs_, [](auto a, auto b) { return a & ~b; });
  special_regs_ &= static_cast<uint16_t>(~rhs.special_regs_);
  return *this;
}

} // namespace rocjitsu
