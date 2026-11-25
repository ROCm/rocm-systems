/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCL_M2N_CHECKED_MATH_H_
#define NCCL_M2N_CHECKED_MATH_H_

#include <cstdint>
#include <cstddef>

static inline bool m2nCheckedAddSize(size_t a, size_t b, size_t* out) {
  if (out == nullptr) {
    return false;
  }
  if (a > SIZE_MAX - b) {
    return false;
  }
  *out = a + b;
  return true;
}

static inline bool m2nCheckedMulSize(size_t a, size_t b, size_t* out) {
  if (out == nullptr) {
    return false;
  }
  if (a != 0 && b > SIZE_MAX / a) {
    return false;
  }
  *out = a * b;
  return true;
}

#endif /* NCCL_M2N_CHECKED_MATH_H_ */
