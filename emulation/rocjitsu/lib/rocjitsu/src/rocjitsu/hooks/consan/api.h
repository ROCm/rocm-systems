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
/// A rejected tracked dispatch is not submitted to HSA. If it has a completion
/// signal, the plugin stores -1 to release waiters; only exactly 0 means normal
/// completion. A wait for a value less than 1 alone does not establish success.
/// The caller must check completion before reading kernel output, then call
/// finish and check its status before destroying remaining HSA signals, queues,
/// memory or executables and before hsa_shut_down. A successful drain permits
/// earlier release of a completed dispatch's signal. A nonzero finish is not a saved
/// capture; it must be reported even if earlier dispatches completed normally.
using rj_consan_drain_fn = int (*)();
using rj_consan_finish_fn = int (*)();

struct rj_consan_statistics {
  uint64_t submitted;
  uint64_t forwarded;
  uint64_t copied;
  uint64_t written;
  uint64_t reused;
};
using rj_consan_statistics_fn = rj_consan_statistics (*)();
