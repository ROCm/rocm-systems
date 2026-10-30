// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/capture.h"

#include <functional>
#include <memory>

namespace rocjitsu::consan {

enum class RuntimeStatus {
  Success,
  PendingCompletion,
  PendingVisibility,
  InvalidTransition,
  InvalidFacts,
  ForwardFailed,
  CopyUnavailable,
  CopyFailed,
  ShortCopy,
  WriteFailed,
  ZeroWriteProgress,
  CloseFailed,
  PublishFailed,
};

struct CopyResult {
  RuntimeStatus status = RuntimeStatus::CopyFailed;
  size_t bytes = 0;
};

struct OwnedSnapshot {
  std::shared_ptr<const Manifest> facts;
  CapturedExecution execution;
};

/// @brief One dispatch's observer/collector handoff, independent of HSA handles.
/// @details The owner synchronizes these calls between its observer and collector.
/// A failure never certifies reuse. Forwarding precedes the potentially slow
/// copy, is attempted once, and is independent of writer progress.
class DispatchLifecycle {
public:
  DispatchLifecycle(std::shared_ptr<const Manifest> facts, CapturedExecution envelope);

  /// @brief Validate and arm a prepared envelope without submitting an HSA packet.
  /// @details Success fixes the required snapshot extent. Invalid facts leave
  /// the phase unchanged; the owner must not publish a packet on failure.
  [[nodiscard]] RuntimeStatus submitted();
  /// @brief Observe completion and system visibility before forwarding once.
  /// @details Pending results allow another observation. Success is idempotent
  /// after forwarding/copying; a failed or throwing forward callback is terminal
  /// because it may already have affected the application's completion signal.
  [[nodiscard]] RuntimeStatus observe(bool completed, bool system_visible,
                                      const std::function<bool()> &forward);
  /// @brief Copy a forwarded execution into independent owning host storage.
  /// @details Success alone permits the owner to release the report lease,
  /// before any blocking capture write. Failure or an exception is terminal;
  /// neither forwarding nor a partial copy certifies reuse or permits retry.
  [[nodiscard]] RuntimeStatus snapshot(const std::function<CopyResult(std::span<uint8_t>)> &copy,
                                       OwnedSnapshot &destination);

private:
  enum class Phase {
    Prepared,
    Submitted, // Envelope admitted; the owner may not have published its packet yet.
    Completed, // Completion observed, but system visibility may still be pending.
    Forwarded, // Visibility established and the forwarding callback succeeded.
    Copied,    // Independent bytes/facts handed to the collector's destination.
    Failed,    // Ambiguous forwarding/copy attempts must never be retried.
  };
  std::shared_ptr<const Manifest> facts_;
  CapturedExecution envelope_;
  Phase phase_ = Phase::Prepared;
  size_t report_bytes_ = 0;
};

/// @brief Plugin-bound synchronous I/O used by SnapshotWriter's encoded stream.
/// @details CaptureEncoder owns framing/encoding; the plugin binds real file
/// operations here. Callbacks report ordinary I/O failures by their results.
/// They must not retain borrowed byte spans after returning.
struct CaptureSink {
  // Consume at most bytes.size(): short writes are retried, negative fails,
  // and zero fails rather than allowing an infinite no-progress loop.
  std::function<int64_t(std::span<const uint8_t>)> write;
  // Flush/close once, including after write failure; true certifies both.
  std::function<bool()> close;
  // Called only after End, all writes, and close succeed. True certifies
  // publication under the final name without overwriting an existing file.
  std::function<bool()> publish;
};

/// @brief Thread-confined blocking writer over the shared capture encoding.
/// @details Never diagnoses GPU contents. Close/write/publication failure is
/// sticky and cannot report a saved capture. The sink keeps partial output
/// unpublished; only a complete, successfully closed file is published.
class SnapshotWriter {
public:
  /// @brief Bind all three callbacks and write the encoded owning header once.
  /// @details Call once on a fresh writer; no failed start is retryable.
  [[nodiscard]] RuntimeStatus start(const CaptureHeader &header, CaptureSink sink);
  /// @brief Validate matching owning facts and synchronously save one snapshot.
  /// @details The GPU allocation is no longer needed. A failed/partial write
  /// poisons the writer; retrying append cannot repair the encoded stream.
  [[nodiscard]] RuntimeStatus append(const OwnedSnapshot &snapshot);
  /// @brief Before finish, prohibit saved success while preserving the first error.
  void cancel(RuntimeStatus reason);
  /// @brief Write End when valid, close once, then publish only on full success.
  /// @details The caller must finish a started writer even after an error.
  /// Repeated calls return the closed result without repeating I/O callbacks.
  [[nodiscard]] RuntimeStatus finish();

private:
  RuntimeStatus write_all(std::span<const uint8_t> bytes);
  CaptureEncoder encoder_;
  CaptureHeader header_;
  CaptureSink sink_;
  RuntimeStatus status_ = RuntimeStatus::Success;
  bool started_ = false;
  bool closed_ = false;
};

} // namespace rocjitsu::consan
