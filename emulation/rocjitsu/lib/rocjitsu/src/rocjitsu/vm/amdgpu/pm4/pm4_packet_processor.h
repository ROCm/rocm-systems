// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <array>
#include <cstdint>
#include <functional>
#include <optional>

namespace rocjitsu::amdgpu {

struct ComputeQueueRecord;
class GpuVm;

/// @brief CP services used by packet execution.
/// @details Queue state remains owned by CP. All callbacks must be populated and
/// run synchronously during process_pm4_packets(); the processor does not retain
/// the context or callbacks.
struct Pm4ExecutionContext {
  /// @brief Target architecture for architecture-specific packet behavior.
  /// @details INVALID represents a CP without a compute unit.
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  /// @brief XCD index used by PRED_EXEC in CDNA vendor command buffers.
  uint32_t xcc_id = 0;
  /// @brief Flush GPU caches before packet memory effects and stream completion.
  std::function<void()> flush_caches;
  /// @brief Admit a dispatch through CP using X, Y, Z, and the initiator word.
  /// @details The initiator determines whether dimensions count threads or groups.
  /// CP appends admitted work to queue.dispatches.entries; processing pauses
  /// until CP retires those entries.
  std::function<void(const std::array<uint32_t, 4> &)> dispatch;
  /// @brief Preserve the owner's retry scheduling at the original execution points.
  /// @details Set queue.command_retry_pending and arrange a later CP scheduling
  /// turn. CP clears the flag before resuming processing.
  std::function<void()> retry;
  /// @brief Wake CP promptly after stream completion or yielding runnable work.
  /// @details Reset retry backoff and arrange a scheduling turn without marking
  /// this queue blocked; packet processing may continue in the current turn.
  std::function<void()> wake;
  /// @brief Cancel through CP at the original execution points, including exception scope.
  /// @details Set queue.faulted, cancel pending work, and notify failed submissions.
  /// The processor sets queue.publication_faulted before calling this callback
  /// on terminal cursor-publication failures; CP preserves that failure state.
  std::function<void()> fault_queue;
};

/// @brief Process a bounded turn of a native ring or DRM indirect-buffer stream.
/// @details Fetch, opcode effects, IB traversal, and cursor publication live here.
/// The caller owns serialization, scheduling, dispatch admission, and cancellation.
/// For native rings, CP captures queue.command_access, initializes the consumer
/// cursor, creates root submissions from doorbells, and retries pending cursor
/// publication even while execution is suspended. This processor traverses those
/// submissions and nested IBs, captures access for submitted streams as needed,
/// and commits/publishes packet retirement through queue.read_pointer_journal.
/// queue.commands and queue.command_access retain progress across turns; the
/// retry and dispatch callbacks must update the shared fields documented above.
/// An unready VM snapshot is released before retry so a later GART publication
/// can be observed. Ready snapshots stay pinned through execution and publication.
/// A VM must be attached before commands can execute; idle queues do not access it.
/// Returns when idle, waiting for dispatch retirement, blocked, faulted, or at the
/// packet budget. Retry and fault handling are reported through context callbacks.
/// Standard exceptions during execution are translated to fault_queue(); exceptions
/// from cancellation before execution or from the catch handler propagate.
/// @param queue CP-owned queue whose command state and cursor are updated in place.
/// @param gpu_vm Borrowed VM used to acquire pinned access to the queue address space.
/// @param context Borrowed architecture and synchronous CP services for this turn.
void process_pm4_packets(ComputeQueueRecord &queue, GpuVm *gpu_vm,
                         const Pm4ExecutionContext &context);

/// @brief Convert a native ring-relative producer cursor into its monotonic epoch.
/// @param producer_cursor Ring-relative cursor or an already monotonic cursor.
/// @param reference_cursor Monotonic consumer cursor defining the current ring epoch.
/// @param ring_dwords Ring capacity in dwords.
/// @return A monotonic cursor at or after the reference, at most one ring ahead;
/// nullopt for zero capacity, arithmetic overflow, or a cursor outside that window.
[[nodiscard]] std::optional<uint64_t> normalize_pm4_producer_cursor(uint64_t producer_cursor,
                                                                    uint64_t reference_cursor,
                                                                    uint64_t ring_dwords) noexcept;

} // namespace rocjitsu::amdgpu
