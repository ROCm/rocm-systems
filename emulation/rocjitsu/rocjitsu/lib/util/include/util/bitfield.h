// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_BITFIELD_H_
#define UTIL_BITFIELD_H_

#include "util/bit.h"
#include "util/meta_programming.h"

#include <bitset>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

namespace util {

template <size_t num_bits>
  requires(num_bits > 0)
class Bitfield;

template <size_t num_bf_bits, size_t first, size_t last, bool Mutable = true> class BitfieldMember;

template <size_t num_bits>
  requires(num_bits > 0)
class Bitfield {
public:
  // Align array-based bitfield to 32b.
  static constexpr size_t array_bytes = ((num_bits + 31) / 32) * 4;
  static constexpr size_t size_bits_ = num_bits;
  static constexpr size_t size_bytes_ = (size_bits_ / std::numeric_limits<uint8_t>::digits);

  using Storage = typename std::conditional<
      num_bits <= 8, uint8_t,
      typename std::conditional<
          num_bits <= 16, uint16_t,
          typename std::conditional<num_bits <= 32, uint32_t,
                                    typename std::conditional<num_bits <= 64, uint64_t,
                                                              uint8_t[array_bytes]>::type>::type>::
          type>::type;

  template <int n> struct BitsType {
    static_assert(n <= 64, "Trying to use too many bits for BitsType.");
    static_assert(n >= 0, "Trying to negative value for BitsType.");

    using Type = typename std::conditional<
        n <= 8, uint8_t,
        typename std::conditional<
            n <= 16, uint16_t,
            typename std::conditional<n <= 32, uint32_t, uint64_t>::type>::type>::type;
  };

  Bitfield() = default;

  Bitfield(const Bitfield &other) { init(other.data); }

  explicit Bitfield(Storage data_) { init(data_); }

  Bitfield &operator=(const Bitfield &other)
    requires metaprogramming::IsIntegral<Storage>
  {
    data = other.data;
    return *this;
  }

  Bitfield &operator=(const Bitfield &other)
    requires metaprogramming::IsArray<Storage>
  {
    std::memcpy(data, other.data, sizeof(data));
    return *this;
  }

  template <typename T>
    requires(metaprogramming::IsIntegral<T> && metaprogramming::IsIntegral<Storage>)
  Bitfield &operator=(T val) {
    data = static_cast<Storage>(val);
    return *this;
  }

  /// @brief Get the bits between first (f) and last (l) inclusive from integral storage.
  /// @tparam f First (LSB) bit position.
  /// @tparam l Last (MSB) bit position.
  /// @tparam T Type used for return.
  /// @returns Retrieved bits.
  template <int f, int l, typename T = typename BitsType<l - f + 1>::Type>
    requires metaprogramming::IsIntegral<Storage>
  T get() const {
    static constexpr int n(l - f + 1);
    static_assert(n > 0, "Trying to read 0 or negative bit count.");
    static_assert(n <= num_bits, "Trying to read too many bits.");
    static_assert(f < std::numeric_limits<Storage>::digits, "First bit is out of bounds.");
    static_assert(l < std::numeric_limits<Storage>::digits, "Last bit is out of bounds.");
    static_assert(f >= 0, "First bit is out of bounds.");
    static_assert(l >= 0, "Last bit is out of bounds.");

    T bits{(data >> f) & mask<T>(n)};
    return bits;
  }

  /// @brief Get the bits between first (f) and last (l) inclusive from array storage.
  /// @tparam f First (LSB) bit position.
  /// @tparam l Last (MSB) bit position.
  /// @tparam T Type used for return.
  /// @returns Retrieved bits.
  template <int f, int l, typename T = typename BitsType<l - f + 1>::Type>
    requires metaprogramming::IsArray<Storage>
  T get() const {
    static constexpr int n(l - f + 1);
    static const int elem_bits(std::numeric_limits<T>::digits);
    static const int first_elem(f / elem_bits);
    static const int last_elem(l / elem_bits);
    static const int bit_offset(f % elem_bits);
    static const bool split(first_elem != last_elem);
    static_assert(n > 0, "Trying to read 0 or negative bit count.");
    static_assert(n <= num_bits, "Trying to read too many bits.");
    static_assert(f < sizeof(Storage) * 8, "First bit is out of bounds.");
    static_assert(l < sizeof(Storage) * 8, "Last bit is out of bounds.");
    static_assert(f >= 0, "First bit is out of bounds.");
    static_assert(l >= 0, "Last bit is out of bounds.");

    T bits{0};

    auto read_elem = [&](int idx) -> T {
      T v{};
      std::memcpy(&v, reinterpret_cast<const uint8_t *>(data) + idx * sizeof(T), sizeof(T));
      return v;
    };

    if (split) {
      int first_bits(elem_bits - bit_offset);
      int last_bits(n - first_bits);

      bits |= ((read_elem(first_elem) >> bit_offset) & mask<T>(first_bits));
      bits |= ((read_elem(last_elem) & mask<T>(last_bits)) << first_bits);
    } else {
      bits |= ((read_elem(first_elem) >> bit_offset) & mask<T>(n));
    }

    return bits;
  }

  template <typename T>
    requires metaprogramming::IsIntegral<Storage>
  void set(T val, int first, int last) {
    data = insert_bits<Storage>(data, static_cast<Storage>(val), first, last);
  }

  /// @brief Return a read-write proxy for the bit range [f, l].
  template <size_t f, size_t l> BitfieldMember<num_bits, f, l, true> member() {
    return BitfieldMember<num_bits, f, l, true>(*this);
  }

  /// @brief Return a read-only proxy for the bit range [f, l].
  template <size_t f, size_t l> BitfieldMember<num_bits, f, l, false> member() const {
    return BitfieldMember<num_bits, f, l, false>(*this);
  }

  /// @brief Convert the entire bitfield to an integral type.
  template <typename T>
    requires(metaprogramming::IsIntegral<T> && !std::is_same_v<T, bool> &&
             metaprogramming::IsIntegral<Storage> && sizeof(T) >= size_bytes_)
  operator T() const {
    return static_cast<T>(get<0, num_bits - 1>());
  }

  explicit operator bool() const
    requires metaprogramming::IsIntegral<Storage>
  {
    return data != 0;
  }

  bool operator==(const Bitfield &rhs) const
    requires metaprogramming::IsIntegral<Storage>
  {
    return data == rhs.data;
  }

  Bitfield operator~() const
    requires metaprogramming::IsIntegral<Storage>
  {
    return Bitfield(static_cast<Storage>(~data));
  }

  Bitfield operator|(const Bitfield &rhs) const
    requires metaprogramming::IsIntegral<Storage>
  {
    return Bitfield(static_cast<Storage>(data | rhs.data));
  }

  Bitfield operator&(const Bitfield &rhs) const
    requires metaprogramming::IsIntegral<Storage>
  {
    return Bitfield(static_cast<Storage>(data & rhs.data));
  }

  Bitfield operator^(const Bitfield &rhs) const
    requires metaprogramming::IsIntegral<Storage>
  {
    return Bitfield(static_cast<Storage>(data ^ rhs.data));
  }

  Bitfield &operator|=(const Bitfield &rhs)
    requires metaprogramming::IsIntegral<Storage>
  {
    data |= rhs.data;
    return *this;
  }

  Bitfield &operator&=(const Bitfield &rhs)
    requires metaprogramming::IsIntegral<Storage>
  {
    data &= rhs.data;
    return *this;
  }

  Bitfield &operator^=(const Bitfield &rhs)
    requires metaprogramming::IsIntegral<Storage>
  {
    data ^= rhs.data;
    return *this;
  }

  void print() const
    requires metaprogramming::IsIntegral<Storage>
  {
    std::cout << std::bitset<std::numeric_limits<Storage>::digits>(data).to_string() << std::endl;
  }

  void print() const
    requires metaprogramming::IsArray<Storage>
  {
    for (int i = array_bytes - 1; i >= 0; --i) {
      std::cout << std::bitset<8>(data[i]).to_string() << " ";

      if (!(i % 8)) {
        std::cout << std::endl;
      }
    }

    std::cout << std::endl;
  }

private:
  void init(Storage data_)
    requires metaprogramming::IsIntegral<Storage>
  {
    data = static_cast<Storage>(data_);
  }

  void init(const uint8_t *data_)
    requires metaprogramming::IsArray<Storage>
  {
    std::memcpy(reinterpret_cast<uint8_t *>(data), data_, sizeof(data));
  }

  Storage data{0};
};

/// @brief Proxy for accessing a range of bits [first, last] within a Bitfield.
/// @details Returned by Bitfield::member<f, l>(). Supports read via implicit
/// conversion and write via operator= and compound assignment. The Mutable
/// parameter controls whether write operations are available.
template <size_t num_bf_bits, size_t first, size_t last, bool Mutable> class BitfieldMember {
public:
  static constexpr size_t num_member_bits = last - first + 1;
  using BfType = Bitfield<num_bf_bits>;
  using BfRef = std::conditional_t<Mutable, BfType &, const BfType &>;
  using ValueType = typename BfType::template BitsType<last - first + 1>::Type;

  BitfieldMember(BfRef bf) : bf_(bf) {}

  /// @brief Assign an integral value to this bit range.
  template <typename T>
    requires(Mutable && metaprogramming::IsIntegral<T>)
  BitfieldMember &operator=(T val) {
    bf_.set(val, first, last);
    return *this;
  }

  /// @brief Bitwise OR-assign into this bit range.
  template <typename T>
    requires(Mutable && metaprogramming::IsIntegral<T>)
  BitfieldMember &operator|=(T val) {
    bf_.set(static_cast<ValueType>(value() | val), first, last);
    return *this;
  }

  /// @brief Bitwise AND-assign into this bit range.
  template <typename T>
    requires(Mutable && metaprogramming::IsIntegral<T>)
  BitfieldMember &operator&=(T val) {
    bf_.set(static_cast<ValueType>(value() & val), first, last);
    return *this;
  }

  /// @brief Bitwise XOR-assign into this bit range.
  template <typename T>
    requires(Mutable && metaprogramming::IsIntegral<T>)
  BitfieldMember &operator^=(T val) {
    bf_.set(static_cast<ValueType>(value() ^ val), first, last);
    return *this;
  }

  /// @brief Read the bit range as its natural unsigned type.
  ValueType value() const { return bf_.template get<first, last>(); }

  /// @brief Implicit conversion to any integral type large enough to hold the value.
  template <typename T>
    requires(metaprogramming::IsIntegral<T> && !std::is_same_v<T, bool>)
  operator T() const {
    static_assert(std::numeric_limits<T>::digits >= num_member_bits,
                  "Destination type is too small to hold member's value.");
    return static_cast<T>(bf_.template get<first, last>());
  }

  explicit operator bool() const { return value() != 0; }

  template <typename T>
    requires metaprogramming::IsIntegral<T>
  bool operator==(T rhs) const {
    return static_cast<T>(bf_.template get<first, last>()) == rhs;
  }

  template <typename T>
    requires metaprogramming::IsIntegral<T>
  auto operator<=>(T rhs) const {
    return static_cast<T>(bf_.template get<first, last>()) <=> rhs;
  }

private:
  BfRef bf_;
};

template <size_t size_bits>
std::ostream &operator<<(std::ostream &os, const Bitfield<size_bits> &bf) {
  os << bf.template get<0, size_bits - 1>();
  return os;
}

template <size_t size, size_t f, size_t l, bool M>
std::ostream &operator<<(std::ostream &os, const BitfieldMember<size, f, l, M> &member) {
  auto v = member.value();
  if constexpr (sizeof(decltype(v)) == 1)
    os << static_cast<unsigned>(v);
  else
    os << v;
  return os;
}

} // namespace util

#endif // UTIL_BITFIELD_H_
