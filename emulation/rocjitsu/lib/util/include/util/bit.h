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
/// @param first The first bit (LSB) to extract.
/// @param last The last bit (MSB) to extract.
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
/// @param position The bit to extract.
/// @returns The extracted bit.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
inline T bit(T val, int position) {
  assert(position < std::numeric_limits<T>::digits);
  assert(position >= 0);

  return bits(val, position, position);
}

/// @brief Extend bits of val starting from the LSB up to num_bits - 1.
/// @details The bits are replicated as many times as possible based on
/// the size of T. The number of bits should be a power of two.
/// @param[in] val Value to be extended.
/// @param[in] num_bits The number of bits in val to extend.
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

/// @brief Extract bits from a multi-word array at an arbitrary bit offset.
/// @param[in] words Pointer to an array of unsigned integer words.
/// @param[in] bit_offset Bit position of the first bit to extract (0-based across all words).
/// @param[in] bit_width Number of bits to extract (must be <= digits of T).
/// @returns The extracted bits, right-justified.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
inline T extract_bits_from_words(const T *words, int bit_offset, int bit_width) {
  static constexpr int word_bits = std::numeric_limits<T>::digits;
  assert(bit_offset >= 0);
  assert(bit_width >= 0 && bit_width <= word_bits);

  if (bit_width == 0)
    return T{0};
  int word_idx = bit_offset / word_bits;
  int local_offset = bit_offset % word_bits;
  T m = bit_width >= word_bits ? std::numeric_limits<T>::max()
                               : static_cast<T>((T{1} << bit_width) - 1);
  if (local_offset + bit_width <= word_bits)
    return (words[word_idx] >> local_offset) & m;
  T lo = words[word_idx] >> local_offset;
  T hi = words[word_idx + 1] << (word_bits - local_offset);
  return (lo | hi) & m;
}

/// @brief Insert bits into a multi-word array at an arbitrary bit offset.
/// @param[in,out] words Pointer to an array of unsigned integer words.
/// @param[in] bit_offset Bit position of the first bit to write (0-based across all words).
/// @param[in] bit_width Number of bits to write (must be <= digits of T).
/// @param[in] value The value to insert (only the lowest bit_width bits are used).
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
inline void insert_bits_into_words(T *words, int bit_offset, int bit_width, T value) {
  static constexpr int word_bits = std::numeric_limits<T>::digits;
  assert(bit_offset >= 0);
  assert(bit_width >= 0 && bit_width <= word_bits);

  if (bit_width == 0)
    return;
  int word_idx = bit_offset / word_bits;
  int local_offset = bit_offset % word_bits;
  T m = bit_width >= word_bits ? std::numeric_limits<T>::max()
                               : static_cast<T>((T{1} << bit_width) - 1);
  value &= m;
  if (local_offset + bit_width <= word_bits) {
    words[word_idx] = (words[word_idx] & ~(m << local_offset)) | (value << local_offset);
    return;
  }
  int lo_bits = word_bits - local_offset;
  T lo_mask = static_cast<T>((T{1} << lo_bits) - 1);
  words[word_idx] =
      (words[word_idx] & ~(lo_mask << local_offset)) | ((value & lo_mask) << local_offset);
  int hi_bits = bit_width - lo_bits;
  T hi_mask = static_cast<T>((T{1} << hi_bits) - 1);
  words[word_idx + 1] = (words[word_idx + 1] & ~hi_mask) | ((value >> lo_bits) & hi_mask);
}

/// @brief Check if a number is a power of 2.
/// @param[in] val Value to check.
/// @returns true if @p val is a power of 2.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
constexpr inline bool is_power_of_2(T val) {
  return std::has_single_bit(val);
}

/// @brief Round @p val up to the next multiple of @p alignment.
/// @param alignment Must be a power of 2.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
constexpr inline T align_up(T val, T alignment) {
  assert(std::has_single_bit(alignment));
  return (val + alignment - 1) & ~(alignment - 1);
}

/// @brief Divide @p numerator by @p divisor, rounding up.
/// @details Asserts that @p divisor is non-zero; this is the strict,
/// general-purpose form. Call sites that must tolerate a zero divisor from
/// untrusted guest input should use ceil_div_or_one() instead.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
constexpr inline T ceil_div(T numerator, T divisor) {
  assert(divisor != 0);
  return numerator / divisor + (numerator % divisor != 0 ? T{1} : T{0});
}

/// @brief Divide @p numerator by @p divisor, rounding up, returning 1 when
/// @p divisor is 0.
/// @details Tolerant variant for paths that parse malformed guest input (e.g.
/// a dispatch packet with a zero workgroup dimension): the emulator must not
/// abort on bad guest data, so a zero divisor yields 1 rather than asserting.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
constexpr inline T ceil_div_or_one(T numerator, T divisor) {
  return divisor == 0 ? T{1} : ceil_div(numerator, divisor);
}

/// @brief Return true when @p val is aligned to @p alignment.
/// @param alignment Must be a power of 2.
template <typename T>
  requires metaprogramming::IsUnsignedInt<T>
constexpr inline bool is_aligned(T val, T alignment) {
  assert(std::has_single_bit(alignment));
  return (val & (alignment - 1)) == 0;
}

} // namespace util

#endif // UTIL_BIT_H_
