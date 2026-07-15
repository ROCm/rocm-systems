// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vector_reg.h
/// @brief Fixed-width SIMD vector register type with elementwise arithmetic.

#ifndef SIMDOJO_COMPONENTS_VECTOR_REG_H_
#define SIMDOJO_COMPONENTS_VECTOR_REG_H_

#include "util/bit.h"
#include "util/meta_programming.h"
#include "util/simd.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <utility>

namespace simdojo {

/// @brief Fixed-width SIMD vector register with elementwise arithmetic operators.
///
/// @details Stores NUM_ELEMS elements of type VecElem in a std::array. Supports
/// elementwise arithmetic with other VectorReg instances of the same element
/// count, as well as scalar broadcast operations. Total size must be a power
/// of 2.
///
/// @tparam NUM_ELEMS Number of elements in the vector register.
/// @tparam VecElem Element type (e.g., uint32_t, float).
template <size_t NUM_ELEMS, typename VecElem> class VectorReg {
public:
  static_assert(util::is_power_of_2(NUM_ELEMS * sizeof(VecElem)),
                "VectorReg total size must be a power of 2.");

  static constexpr size_t num_elems_ = NUM_ELEMS;
  using Storage = std::array<VecElem, NUM_ELEMS>;

  VectorReg() = default;
  VectorReg(Storage data) : data_(std::move(data)) {}

  VecElem operator[](size_t idx) const { return data_[idx]; }
  VecElem &operator[](size_t idx) { return data_[idx]; }

  /// SIMD load of `native<T>` from W contiguous lanes starting at `lane_base`.
  /// The storage pointer never escapes the object. T is a 32-bit lane type.
  template <typename T> util::native<T> simd_load(size_t lane_base) const {
    static_assert(sizeof(T) == sizeof(VecElem), "simd_load: T must match element width");
    return util::load<T>(reinterpret_cast<const uint32_t *>(&data_[lane_base]));
  }

  /// Masked SIMD store of `v` into W contiguous lanes at `lane_base`; bit i of
  /// `mask` enables lane `lane_base + i`.
  template <typename T> void simd_store(size_t lane_base, util::native<T> v, uint64_t mask) {
    static_assert(sizeof(T) == sizeof(VecElem), "simd_store: T must match element width");
    util::masked_store<T>(reinterpret_cast<uint32_t *>(&data_[lane_base]), v, mask);
  }

  /// Narrow (native_width64-wide) SIMD load of a 32-bit lane type from W
  /// contiguous lanes at `lane_base`, used by the mixed-width f64<->32-bit glue.
  /// The storage pointer never escapes the object.
  template <typename T> util::narrow32<T> simd_load_narrow(size_t lane_base) const {
    static_assert(sizeof(T) == sizeof(VecElem), "simd_load_narrow: T must match element width");
    return util::load_narrow<T>(reinterpret_cast<const uint32_t *>(&data_[lane_base]));
  }

  /// Masked narrow (native_width64-wide) SIMD store of a 32-bit lane type into W
  /// contiguous lanes at `lane_base`; bit i of `mask` enables lane `lane_base + i`.
  template <typename T>
  void simd_store_narrow(size_t lane_base, util::narrow32<T> v, uint64_t mask) {
    static_assert(sizeof(T) == sizeof(VecElem), "simd_store_narrow: T must match element width");
    util::masked_store_narrow<T>(reinterpret_cast<uint32_t *>(&data_[lane_base]), v, mask);
  }

  /// 64-bit-lane SIMD load combining this register (lo, bits [31:0]) with `hi`
  /// (bits [63:32]) over W = native_width64 lanes at `lane_base`. T is a 64-bit
  /// lane type. The two storage pointers never escape the objects.
  template <typename T> util::native<T> simd_load64(const VectorReg &hi, size_t lane_base) const {
    static_assert(sizeof(T) == sizeof(uint64_t), "simd_load64: T must be a 64-bit lane type");
    return util::load64<T>(reinterpret_cast<const uint32_t *>(&data_[lane_base]),
                           reinterpret_cast<const uint32_t *>(&hi.data_[lane_base]));
  }

  /// Masked 64-bit-lane SIMD store of `v` into this register (lo) + `hi` over W =
  /// native_width64 lanes at `lane_base`; bit i of `mask` enables lane
  /// `lane_base + i`. The two storage pointers never escape the objects.
  template <typename T>
  void simd_store64(VectorReg &hi, size_t lane_base, util::native<T> v, uint64_t mask) {
    static_assert(sizeof(T) == sizeof(uint64_t), "simd_store64: T must be a 64-bit lane type");
    util::masked_store64<T>(reinterpret_cast<uint32_t *>(&data_[lane_base]),
                            reinterpret_cast<uint32_t *>(&hi.data_[lane_base]), v, mask);
  }

  template <size_t OtherN, typename OtherVecElem>
    requires(num_elems_ == VectorReg<OtherN, OtherVecElem>::num_elems_)
  VectorReg &operator=(const VectorReg<OtherN, OtherVecElem> &vreg) {
    for (size_t i = 0; i < num_elems_; ++i)
      data_[i] = static_cast<VecElem>(vreg[i]);
    return *this;
  }

  template <size_t OtherN, typename OtherVecElem>
    requires(num_elems_ == VectorReg<OtherN, OtherVecElem>::num_elems_)
  VectorReg operator+(const VectorReg<OtherN, OtherVecElem> &rhs) const {
    return elementwise(rhs, [](VecElem a, OtherVecElem b) { return static_cast<VecElem>(a + b); });
  }

  template <size_t OtherN, typename OtherVecElem>
    requires(num_elems_ == VectorReg<OtherN, OtherVecElem>::num_elems_)
  VectorReg operator-(const VectorReg<OtherN, OtherVecElem> &rhs) const {
    return elementwise(rhs, [](VecElem a, OtherVecElem b) { return static_cast<VecElem>(a - b); });
  }

  template <size_t OtherN, typename OtherVecElem>
    requires(num_elems_ == VectorReg<OtherN, OtherVecElem>::num_elems_)
  VectorReg operator*(const VectorReg<OtherN, OtherVecElem> &rhs) const {
    return elementwise(rhs, [](VecElem a, OtherVecElem b) { return static_cast<VecElem>(a * b); });
  }

  template <size_t OtherN, typename OtherVecElem>
    requires(num_elems_ == VectorReg<OtherN, OtherVecElem>::num_elems_)
  VectorReg operator/(const VectorReg<OtherN, OtherVecElem> &rhs) const {
    return elementwise(rhs, [](VecElem a, OtherVecElem b) { return static_cast<VecElem>(a / b); });
  }

  template <typename Scalar>
    requires util::metaprogramming::IsArithmetic<Scalar>
  VectorReg operator+(Scalar s) const {
    return scalar_op(s, [](VecElem a, Scalar b) { return static_cast<VecElem>(a + b); });
  }

  template <typename Scalar>
    requires util::metaprogramming::IsArithmetic<Scalar>
  VectorReg operator-(Scalar s) const {
    return scalar_op(s, [](VecElem a, Scalar b) { return static_cast<VecElem>(a - b); });
  }

  template <typename Scalar>
    requires util::metaprogramming::IsArithmetic<Scalar>
  VectorReg operator*(Scalar s) const {
    return scalar_op(s, [](VecElem a, Scalar b) { return static_cast<VecElem>(a * b); });
  }

  template <typename Scalar>
    requires util::metaprogramming::IsArithmetic<Scalar>
  VectorReg operator/(Scalar s) const {
    return scalar_op(s, [](VecElem a, Scalar b) { return static_cast<VecElem>(a / b); });
  }

  template <size_t OtherN, typename OtherVecElem>
    requires(num_elems_ == VectorReg<OtherN, OtherVecElem>::num_elems_)
  VectorReg &operator+=(const VectorReg<OtherN, OtherVecElem> &rhs) {
    return *this = *this + rhs;
  }

  template <size_t OtherN, typename OtherVecElem>
    requires(num_elems_ == VectorReg<OtherN, OtherVecElem>::num_elems_)
  VectorReg &operator-=(const VectorReg<OtherN, OtherVecElem> &rhs) {
    return *this = *this - rhs;
  }

  template <size_t OtherN, typename OtherVecElem>
    requires(num_elems_ == VectorReg<OtherN, OtherVecElem>::num_elems_)
  VectorReg &operator*=(const VectorReg<OtherN, OtherVecElem> &rhs) {
    return *this = *this * rhs;
  }

  template <size_t OtherN, typename OtherVecElem>
    requires(num_elems_ == VectorReg<OtherN, OtherVecElem>::num_elems_)
  VectorReg &operator/=(const VectorReg<OtherN, OtherVecElem> &rhs) {
    return *this = *this / rhs;
  }

  template <typename Scalar>
    requires util::metaprogramming::IsArithmetic<Scalar>
  VectorReg &operator+=(Scalar s) {
    return *this = *this + s;
  }

  template <typename Scalar>
    requires util::metaprogramming::IsArithmetic<Scalar>
  VectorReg &operator-=(Scalar s) {
    return *this = *this - s;
  }

  template <typename Scalar>
    requires util::metaprogramming::IsArithmetic<Scalar>
  VectorReg &operator*=(Scalar s) {
    return *this = *this * s;
  }

  template <typename Scalar>
    requires util::metaprogramming::IsArithmetic<Scalar>
  VectorReg &operator/=(Scalar s) {
    return *this = *this / s;
  }

private:
  template <size_t OtherN, typename OtherVecElem, typename Op>
  VectorReg elementwise(const VectorReg<OtherN, OtherVecElem> &rhs, Op op) const {
    VectorReg result;
    for (size_t i = 0; i < num_elems_; ++i)
      result[i] = op((*this)[i], rhs[i]);
    return result;
  }

  template <typename Scalar, typename Op> VectorReg scalar_op(Scalar s, Op op) const {
    VectorReg result;
    for (size_t i = 0; i < num_elems_; ++i)
      result[i] = op((*this)[i], s);
    return result;
  }

  friend std::ostream &operator<<(std::ostream &os, const VectorReg<NUM_ELEMS, VecElem> &vec_reg) {
    for (size_t i = 0; i < num_elems_; ++i)
      os << vec_reg.data_[i] << " ";
    return os;
  }

  Storage data_{};
};

/// @brief Commutative scalar-on-left arithmetic (scalar + vec, scalar * vec, etc.).
template <typename Scalar, size_t NUM_ELEMS, typename VecElem>
  requires util::metaprogramming::IsArithmetic<Scalar>
VectorReg<NUM_ELEMS, VecElem> operator+(Scalar s, const VectorReg<NUM_ELEMS, VecElem> &v) {
  return v + s;
}

template <typename Scalar, size_t NUM_ELEMS, typename VecElem>
  requires util::metaprogramming::IsArithmetic<Scalar>
VectorReg<NUM_ELEMS, VecElem> operator*(Scalar s, const VectorReg<NUM_ELEMS, VecElem> &v) {
  return v * s;
}

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_VECTOR_REG_H_
