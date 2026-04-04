// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_BIT_H_
#define UTIL_BIT_H_

#include "util/meta_programming.h"

#include <bit>
#include <cassert>
#include <limits>
#include <type_traits>

namespace util {

/// @brief Generate an n-bit mask justified to the LSB.
/// @param[in] n The number of bits in the mask.
/// @returns The bit mask.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
inline T mask(int n) {
  assert(n <= std::numeric_limits<T>::digits);

  if (n < std::numeric_limits<T>::digits) {
    const T one{1};
    const T m{(one << n) - 1};
    return m;
  }
  return std::numeric_limits<T>::max();
}

/// @brief Generate a bit mask in place from first to last inclusive.
/// @param[in] first The first bit (LSB) of the mask.
/// @param[in] last The last bit (MSB) of the mask.
/// @returns The bit mask.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
inline T mask(int first, int last) {
  assert(last < std::numeric_limits<T>::digits);
  assert(first >= 0);
  assert(last >= first);

  return (~mask<T>(first) & mask<T>(last + 1));
}

/// @brief Insert bits from bit_val into val from first to last inclusive.
/// @param[in] val The val into which we are inserting bits.
/// @param[in] bit_val The bits we are inserting into val.
/// @param[in] first The first bit (LSB) we are inserting into.
/// @param[in] last The last bit (MSB) we are inserting into.
/// @returns The value with bit_val inserted between first and last.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
inline T insert_bits(T val, T bit_val, int first, int last) {
  assert(last < std::numeric_limits<T>::digits);
  assert(first >= 0);
  assert(last >= first);

  const T insert_mask{mask<T>(first, last)};
  val = (((bit_val << first) & insert_mask) | (val & ~insert_mask));
  return val;
}

/// @brief Sets the bits from first to last inclusive in val.
/// @param[in] val The value whose bits will be set.
/// @param[in] first The first bit (LSB) to set.
/// @param[in] last The last bit (MSB) to set.
/// @returns The value with bits from first to last set.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
inline T set_bits(T val, int first, int last) {
  assert(last < std::numeric_limits<T>::digits);
  assert(first >= 0);
  assert(last >= first);

  return (val | mask<T>(first, last));
}

/// @brief Sets the bit specified by position in val.
/// @param[in] val The value whose bit will be set.
/// @param[in] position The bit to set.
/// @returns The value with bit at position set.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
inline T set_bit(T val, int position) {
  assert(position < std::numeric_limits<T>::digits);
  assert(position >= 0);

  return (val | mask<T>(position, position));
}

/// @brief Clears the bits from first to last inclusive in val.
/// @param[in] val The value whose bits will be cleared.
/// @param[in] first The first bit (LSB) to cleared.
/// @param[in] last The last bit (MSB) to cleared.
/// @returns The value with bits from first to last cleared.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
inline T clear_bits(T val, int first, int last) {
  assert(last < std::numeric_limits<T>::digits);
  assert(first >= 0);
  assert(last >= first);

  return (val & ~mask<T>(first, last));
}

/// @brief Clears the bit specified by position.
/// @param[in] val The value whose bit will be cleared.
/// @param[in] position The bit to clear.
/// @returns The value with bit at position cleared.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
inline T clear_bit(T val, int position) {
  assert(position < std::numeric_limits<T>::digits);
  assert(position >= 0);

  return (val & ~mask<T>(position, position));
}

/// @brief Extract bits from first to last inclusive from val and justify to the LSB.
/// @param val The value to extract bits from.
/// @param first The first bit (LSB) to extact.
/// @param last The last bit (MSB) to extact.
/// @returns The extracted bits.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
inline T bits(T val, int first, int last) {
  assert(last < std::numeric_limits<T>::digits);
  assert(first >= 0);
  assert(last >= first);

  return ((val >> first) & mask<T>(last - first + 1));
}

/// @brief Extract bit specified by position from val and justify to the LSB.
/// @param val The value to extract the bit from.
/// @param position The bit to extact.
/// @returns The extracted bit.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
inline T bit(T val, int position) {
  assert(position < std::numeric_limits<T>::digits);
  assert(position >= 0);

  return bits(val, position, position);
}

/// @brief Extend bits of val starting from the LSB up to num_bits - 1.
/// @details The bits are replicated as many times as possibe based on
/// the size of T. The number of bits should be a power of two.
/// @param[in] val Value to be extended.
/// @param[in] num_bits The number of of bits in val to extend.
/// @returns Value containing the replicated bits.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
inline T extend_bits(T val, int num_bits) {
  assert(num_bits <= std::numeric_limits<T>::digits);
  assert(!(std::numeric_limits<T>::digits % num_bits));

  T res{0};
  const int num_chunks(std::numeric_limits<T>::digits / num_bits);

  if (num_chunks == 1) {
    return val;
  }

  const int last{num_bits - 1};
  const T bit_val{bits(val, 0, last)};

  for (int i = 0; i < num_chunks; ++i) {
    res |= bit_val << i * num_bits;
  }

  return res;
}

/// @brief Return the popcount of val.
/// @param[in] val Value whose bits are being counted.
/// @returns Number of 1 bits.
template <typename T> inline int popcount(T val) { return std::popcount<T>(val); }

/// @brief lead_zero_count. Return the leading zero count of val.
/// @param[in] val Value whose bits are being counted.
/// @returns Number of leading 0 bits.
template <typename T> inline int lead_zero_count(T val) { return std::countl_zero<T>(val); }

/// @brief Return the trailing zero count of val.
/// @param[in] val Value whose bits are being counted.
/// @returns Number of trailing 0 bits.
template <typename T> inline int trail_zero_count(T val) { return std::countr_zero<T>(val); }

/// @brief Check if a number is a power of 2.
/// @param[in] val Value to check.
/// @returns true if @p val is a power of 2.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
constexpr inline bool is_power_of_2(T val) {
  return std::has_single_bit(val);
}

} // namespace util

#endif // UTIL_BIT_H_
