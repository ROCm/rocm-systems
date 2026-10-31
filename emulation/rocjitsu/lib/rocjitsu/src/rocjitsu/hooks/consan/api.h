// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @brief Controlled-runner interfaces resolved from the loaded ConSan plugin.
/// @details Zero is success. Both functions run while HSA is alive; finish
/// closes admissions, drains ownership handoffs and publishes a closed capture.
/// drain keeps admissions open while waiting for previously submitted launches
/// to complete collection and writing. It permits bounded report-storage reuse
/// without ending the capture; finish is terminal and cannot replace it.
/// Expected failures return status codes; unexpected C++ exceptions terminate
/// the process, without promising recovery or completion of outstanding work.
/// Signals successfully created through the hooked hsa_signal_create or, when
/// available, hsa_amd_signal_create after plugin installation are tracked. A
/// dispatch requires a fresh, dedicated completion initialized to 1. Preexisting,
/// imported, other-API or hook-bypassing signals are untracked and cannot be used
/// as dispatch completions. A rejected dispatch is not submitted to HSA; its
/// tracked completion receives -1 unless already owned by pending work. Rejected
/// batches and barriers discarded after failure notify tracked AND/OR barrier
/// completions by the same rule. Healthy barriers pass through. Other packet
/// types are discarded without completion notification, even for tracked signals.
/// Untracked signals receive no notification. Use bounded waits; -1 does not
/// satisfy a barrier dependency requiring 0. Only exactly 0 means normal
/// completion; a wait for a value less than 1 alone does not establish success.
/// The caller must check completion before reading kernel output, then call
/// finish and check its status before destroying remaining HSA signals, queues,
/// memory or executables and before hsa_shut_down. A successful drain permits
/// earlier release of a completed dispatch's signal. A nonzero finish is not a saved
/// capture; it must be reported even if earlier dispatches completed normally.
using rj_consan_drain_fn = int (*)();
using rj_consan_finish_fn = int (*)();
