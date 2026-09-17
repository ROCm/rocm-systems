// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_version.h
/// @brief Build and source identity shared by all RocJITsu consumers.

#ifndef ROCJITSU_BASE_RJ_VERSION_H_
#define ROCJITSU_BASE_RJ_VERSION_H_

#include "rocjitsu/base/rj_compiler.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Return the immutable identity embedded when RocJITsu is built.
///
/// The formatted string contains the project version, Git revision, commit
/// date, and commit title. Git values are `"unknown"` when the build did not
/// have repository metadata, such as an unqualified source archive.
///
/// @return A non-null, null-terminated string with static lifetime.
RJ_API_EXPORT const char *rj_get_version_string(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_BASE_RJ_VERSION_H_
