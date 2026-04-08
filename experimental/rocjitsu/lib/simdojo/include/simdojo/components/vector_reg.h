// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vector_reg.h
/// @brief Fixed-width SIMD vector register type with elementwise arithmetic.

#ifndef SIMDOJO_COMPONENTS_VECTOR_REG_H_
#define SIMDOJO_COMPONENTS_VECTOR_REG_H_

#include "util/bit.h"
#include "util/meta_programming.h"

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
