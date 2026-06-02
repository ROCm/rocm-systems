/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if __cplusplus >= 202002L
#include <concepts>
#define HIPFILE_REQUIRES(...) requires(__VA_ARGS__)
#else
#define HIPFILE_REQUIRES(...)
#endif
