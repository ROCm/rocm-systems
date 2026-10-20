// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/hooks/consan/runtime_core.h"
#include "rocjitsu/code/consan/encoding.h"

namespace rocjitsu::consan {

DispatchLifecycle::DispatchLifecycle(std::shared_ptr<const Manifest> facts,
                                     CapturedExecution envelope)
    : facts_(std::move(facts)), envelope_(std::move(envelope)) {}

RuntimeStatus DispatchLifecycle::submitted() {
  if (phase_ != Phase::Prepared)
    return RuntimeStatus::InvalidTransition;
  if (facts_ == nullptr)
    return RuntimeStatus::InvalidFacts;
  const auto layout =
      plan_launch(*facts_, envelope_.dimensions, envelope_.group_bytes, envelope_.private_bytes);
  if (layout.failed())
    return RuntimeStatus::InvalidFacts;
  // Admission starts with an identity and no snapshot, not previously copied bytes.
  if (envelope_.serial == 0 || envelope_.storage_slot == 0 || envelope_.storage_generation == 0 ||
      !envelope_.report.empty())
    return RuntimeStatus::InvalidFacts;
  // Only a valid layout can establish the extent promised by this context.
  if (wire::read_u64(envelope_.context, wire::kContextSerial) != envelope_.serial ||
      wire::read_u64(envelope_.context, wire::kContextReportBytes) != layout.value().report_bytes)
    return RuntimeStatus::InvalidFacts;
  report_bytes_ = layout.value().report_bytes;
  phase_ = Phase::Submitted;
  return RuntimeStatus::Success;
}

RuntimeStatus DispatchLifecycle::observe(bool completed, bool system_visible,
                                         const std::function<bool()> &forward) {
  if (phase_ == Phase::Forwarded || phase_ == Phase::Copied)
    return RuntimeStatus::Success;
  if (phase_ != Phase::Submitted && phase_ != Phase::Completed)
    return RuntimeStatus::InvalidTransition;
  if (completed)
    phase_ = Phase::Completed;
  if (phase_ == Phase::Submitted)
    return RuntimeStatus::PendingCompletion;
  if (!system_visible)
    return RuntimeStatus::PendingVisibility;
  // Set a terminal state before invoking external code. An ambiguous forwarding
  // failure cannot cause a retry that decrements the application's signal twice.
  phase_ = Phase::Failed;
  if (!forward || !forward())
    return RuntimeStatus::ForwardFailed;
  phase_ = Phase::Forwarded;
  return RuntimeStatus::Success;
}

RuntimeStatus DispatchLifecycle::snapshot(const std::function<CopyResult(std::span<uint8_t>)> &copy,
                                          OwnedSnapshot &destination) {
  if (phase_ != Phase::Forwarded)
    return RuntimeStatus::InvalidTransition;
  phase_ = Phase::Failed;
  if (!copy)
    return RuntimeStatus::CopyUnavailable;
  std::vector<uint8_t> bytes(report_bytes_);
  const CopyResult copied = copy(bytes);
  if (copied.status != RuntimeStatus::Success)
    return RuntimeStatus::CopyFailed;
  if (copied.bytes != report_bytes_)
    return RuntimeStatus::ShortCopy;
  envelope_.report = std::move(bytes);
  envelope_.completion = kCompletionVisibleForwarded;
  destination = {facts_, std::move(envelope_)};
  phase_ = Phase::Copied;
  return RuntimeStatus::Success;
}

RuntimeStatus SnapshotWriter::write_all(std::span<const uint8_t> bytes) {
  while (!bytes.empty()) {
    const int64_t written = sink_.write(bytes);
    if (written < 0 || static_cast<uint64_t>(written) > bytes.size())
      return status_ = RuntimeStatus::WriteFailed;
    if (written == 0)
      return status_ = RuntimeStatus::ZeroWriteProgress;
    bytes = bytes.subspan(static_cast<size_t>(written));
  }
  return RuntimeStatus::Success;
}

RuntimeStatus SnapshotWriter::start(const CaptureHeader &header, CaptureSink sink) {
  if (started_ || !sink.write || !sink.close || !sink.publish)
    return status_ = RuntimeStatus::InvalidTransition;
  const auto prefix = encoder_.start(header);
  if (prefix.failed())
    return status_ = RuntimeStatus::InvalidFacts;
  header_ = header;
  sink_ = std::move(sink);
  started_ = true;
  return write_all(prefix.value());
}

RuntimeStatus SnapshotWriter::append(const OwnedSnapshot &snapshot) {
  if (status_ != RuntimeStatus::Success)
    return status_;
  if (!started_ || closed_ || snapshot.facts == nullptr || *snapshot.facts != header_.manifest)
    return status_ = RuntimeStatus::InvalidFacts;
  const auto frame = encoder_.append(snapshot.execution);
  if (frame.failed())
    return status_ = RuntimeStatus::InvalidFacts;
  if (write_all(frame.value()) != RuntimeStatus::Success)
    return status_;
  return RuntimeStatus::Success;
}

RuntimeStatus SnapshotWriter::finish() {
  if (!started_)
    return status_ = RuntimeStatus::InvalidTransition;
  if (closed_)
    return status_;
  if (status_ == RuntimeStatus::Success) {
    const auto end = encoder_.finish();
    if (end.failed())
      status_ = RuntimeStatus::InvalidFacts;
    else
      (void)write_all(end.value());
  }
  closed_ = true;
  const bool closed = sink_.close();
  if (!closed && status_ == RuntimeStatus::Success)
    status_ = RuntimeStatus::CloseFailed;
  if (status_ == RuntimeStatus::Success) {
    if (!sink_.publish())
      status_ = RuntimeStatus::PublishFailed;
  }
  return status_;
}

void SnapshotWriter::cancel(RuntimeStatus reason) {
  if (status_ == RuntimeStatus::Success && reason != RuntimeStatus::Success)
    status_ = reason;
}

} // namespace rocjitsu::consan
