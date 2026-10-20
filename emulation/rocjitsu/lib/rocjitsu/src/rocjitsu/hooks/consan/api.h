// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

/// @brief Controlled-runner interfaces resolved from the loaded ConSan plugin.
/// @details Zero is success. Both functions run while HSA is alive; finish
/// closes admissions, drains ownership handoffs and publishes a closed capture.
/// drain keeps admissions open and allows tests to require actual storage reuse.
/// Expected failures return status codes; unexpected C++ exceptions terminate
/// the process, without promising recovery or completion of outstanding work.
using rj_consan_drain_fn = int (*)();
using rj_consan_finish_fn = int (*)();

struct rj_consan_statistics {
  uint64_t submitted = 0;
  uint64_t forwarded = 0;
  uint64_t copied = 0;
  uint64_t written = 0;
  uint64_t reused = 0;
};
using rj_consan_statistics_fn = rj_consan_statistics (*)();
