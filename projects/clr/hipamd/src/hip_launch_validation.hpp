/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_LAUNCH_VALIDATION_HPP_
#define HIP_LAUNCH_VALIDATION_HPP_

#include <hip/hip_runtime.h>
#include "platform/ndrange.hpp"

//! One entry in a per-entry-point launch-error rule table: which violation bits it matches, and
//! the hipError_t to return when any of them is set.
struct LaunchErrorRule {
  uint16_t violations;   //!< bits this rule matches
  hipError_t error;      //!< code to return if violation occurred
};

//! Return the error for the FIRST rule matching any set bit; hipSuccess if none.
//! Rules are listed in each caller's existing check order, to prevent API breaks
template <size_t N>
inline hipError_t MapLaunchViolations(uint16_t violations, const LaunchErrorRule (&rules)[N]) {
  if (violations == amd::kLaunchOk) return hipSuccess;
  for (const auto& rule : rules) {
    if (violations & rule.violations) return rule.error;
  }
  return hipSuccess;
}

//! Helper specifying most used bits to be checked
static constexpr uint16_t kConfigBits = amd::kGridOverflow | amd::kBlockOverflow |
                                        amd::kClusterOverflow | amd::kClusterIndivisible;

#endif  // HIP_LAUNCH_VALIDATION_HPP_
