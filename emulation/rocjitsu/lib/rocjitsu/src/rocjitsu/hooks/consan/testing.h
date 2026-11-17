// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

/// @brief Counter snapshot exposed only by libhsa_consan_testable.so.
/// @details Production uses the same counter storage internally but does not
/// export the inspection function. This interface is not a production ABI.
struct rj_consan_statistics {
  uint64_t submitted;
  uint64_t forwarded;
  uint64_t copied;
  uint64_t written;
  uint64_t reused;
};
using rj_consan_statistics_fn = rj_consan_statistics (*)();
