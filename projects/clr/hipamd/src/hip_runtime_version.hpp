/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_RUNTIME_VERSION_HPP
#define HIP_RUNTIME_VERSION_HPP

#include <cstddef>

namespace hip::runtime_version {

extern const char kBuildName[];
extern const char kGitHash[];
extern const size_t kBuildId;

}  // namespace hip::runtime_version

#endif  // HIP_RUNTIME_VERSION_HPP
