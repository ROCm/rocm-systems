// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/sdma_queue_runner.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>
#include <utility>

namespace rocjitsu::amdgpu {
namespace {

SdmaQueueServiceOutcome map(VmAccessOutcome outcome) {
  switch (outcome) {
  case VmAccessOutcome::Complete:
    return SdmaQueueServiceOutcome::Drained;
  case VmAccessOutcome::Unavailable:
    return SdmaQueueServiceOutcome::Unavailable;
  case VmAccessOutcome::Faulted:
    return SdmaQueueServiceOutcome::Faulted;
  case VmAccessOutcome::Malformed:
    return SdmaQueueServiceOutcome::Malformed;
  }
  return SdmaQueueServiceOutcome::Malformed;
}

SdmaQueueServiceOutcome map(SdmaExecutionOutcome outcome) {
  switch (outcome) {
  case SdmaExecutionOutcome::Complete:
    return SdmaQueueServiceOutcome::Drained;
  case SdmaExecutionOutcome::Unavailable:
    return SdmaQueueServiceOutcome::Unavailable;
  case SdmaExecutionOutcome::Faulted:
    return SdmaQueueServiceOutcome::Faulted;
  case SdmaExecutionOutcome::Malformed:
    return SdmaQueueServiceOutcome::Malformed;
  }
  return SdmaQueueServiceOutcome::Malformed;
}

} // namespace

SdmaQueueRunner::SdmaQueueRunner(GpuVm &gpu_vm, SdmaQueueRunnerConfig config, SdmaExecutor executor)
    : gpu_vm_(&gpu_vm), config_(std::move(config)), executor_(std::move(executor)) {
  if (config_.initial_cursor) {
    cursor_ = *config_.initial_cursor;
    cursor_initialized_ = true;
  }
}

SdmaQueueRunner::SdmaQueueRunner(GpuVm &gpu_vm, SdmaQueueRunnerConfig config,
                                 SdmaPacketDialect dialect, SdmaExecutorCallbacks callbacks)
    : SdmaQueueRunner(gpu_vm, std::move(config), SdmaExecutor(dialect, std::move(callbacks))) {}

SdmaQueueRunner::~SdmaQueueRunner() = default;
SdmaQueueRunner::SdmaQueueRunner(SdmaQueueRunner &&) noexcept = default;

std::optional<SdmaQueueServiceOutcome> SdmaQueueRunner::validate_configuration() const {
  if (!config_.address_space || config_.ring_bytes == 0 ||
      (config_.ring_base % sizeof(uint32_t)) != 0 || (config_.ring_bytes % sizeof(uint32_t)) != 0 ||
      config_.read_pointer_address == 0 || (config_.read_pointer_address % sizeof(uint64_t)) != 0 ||
      config_.ring_base > std::numeric_limits<uint64_t>::max() - (config_.ring_bytes - 1) ||
      (config_.initial_cursor && (*config_.initial_cursor % sizeof(uint32_t)) != 0)) {
    return SdmaQueueServiceOutcome::Malformed;
  }
  return std::nullopt;
}

SdmaQueueServiceOutcome SdmaQueueRunner::latch(SdmaQueueServiceOutcome outcome) {
  if (outcome == SdmaQueueServiceOutcome::Faulted || outcome == SdmaQueueServiceOutcome::Malformed)
    terminal_ = outcome;
  return outcome;
}

SdmaQueueServiceOutcome SdmaQueueRunner::initialize_cursor() {
  if (cursor_initialized_)
    return SdmaQueueServiceOutcome::Drained;
  const AtomicLoadResult loaded =
      access_->atomic_load(config_.read_pointer_address, sizeof(uint64_t));
  if (loaded.outcome != VmAccessOutcome::Complete)
    return map(loaded.outcome);
  if ((loaded.value % sizeof(uint32_t)) != 0)
    return SdmaQueueServiceOutcome::Malformed;
  cursor_ = loaded.value;
  cursor_initialized_ = true;
  return SdmaQueueServiceOutcome::Drained;
}

SdmaQueueServiceOutcome SdmaQueueRunner::fetch(uint64_t producer) {
  if (!batch_active_) {
    if ((producer % sizeof(uint32_t)) != 0 || producer < cursor_ ||
        producer - cursor_ > config_.ring_bytes)
      return SdmaQueueServiceOutcome::Malformed;
    batch_active_ = true;
    fetch_end_cursor_ = producer;
    fetch_bytes_.resize(static_cast<std::size_t>(producer - cursor_));
  } else if (producer < fetch_end_cursor_) {
    return SdmaQueueServiceOutcome::Malformed;
  }

  while (fetch_progress_ < fetch_bytes_.size()) {
    const uint64_t pointer = cursor_ + fetch_progress_;
    const uint64_t ring_offset = pointer % config_.ring_bytes;
    const std::size_t chunk =
        std::min<std::size_t>(fetch_bytes_.size() - fetch_progress_,
                              static_cast<std::size_t>(config_.ring_bytes - ring_offset));
    const VmAccessOutcome outcome = access_->read(
        config_.ring_base + ring_offset, std::span(fetch_bytes_).subspan(fetch_progress_, chunk),
        fetch_segment_progress_);
    if (outcome != VmAccessOutcome::Complete)
      return map(outcome);
    fetch_progress_ += chunk;
    fetch_segment_progress_ = 0;
  }

  if (fetched_words_.empty() && !fetch_bytes_.empty()) {
    fetched_words_.resize(fetch_bytes_.size() / sizeof(uint32_t));
    std::memcpy(fetched_words_.data(), fetch_bytes_.data(), fetch_bytes_.size());
  }
  return SdmaQueueServiceOutcome::Drained;
}

SdmaQueueServiceOutcome SdmaQueueRunner::publish_cursor() {
  if (!publication_pending_)
    return SdmaQueueServiceOutcome::Drained;
  const VmAccessOutcome outcome =
      access_->atomic_store(config_.read_pointer_address, sizeof(uint64_t), cursor_);
  if (outcome == VmAccessOutcome::Complete)
    publication_pending_ = false;
  return map(outcome);
}

void SdmaQueueRunner::clear_batch() {
  batch_active_ = false;
  fetch_end_cursor_ = 0;
  fetch_bytes_.clear();
  fetch_progress_ = 0;
  fetch_segment_progress_ = 0;
  fetched_words_.clear();
  fetched_word_offset_ = 0;
  terminal_after_publication_.reset();
}

SdmaQueueServiceOutcome SdmaQueueRunner::service(uint64_t producer) {
  if (terminal_)
    return *terminal_;
  if (const auto invalid = validate_configuration())
    return latch(*invalid);
  if (!access_) {
    access_ = gpu_vm_->snapshot(config_.address_space);
    if (!access_)
      return latch(SdmaQueueServiceOutcome::Faulted);
  }

  const SdmaQueueServiceOutcome initialized = initialize_cursor();
  if (initialized != SdmaQueueServiceOutcome::Drained)
    return initialized == SdmaQueueServiceOutcome::Unavailable ? initialized : latch(initialized);

  if (batch_active_ && producer < fetch_end_cursor_)
    return latch(SdmaQueueServiceOutcome::Malformed);

  if (publication_pending_) {
    const SdmaQueueServiceOutcome published = publish_cursor();
    if (published != SdmaQueueServiceOutcome::Drained)
      return published == SdmaQueueServiceOutcome::Unavailable ? published : latch(published);
    if (terminal_after_publication_)
      return latch(*terminal_after_publication_);
    // The prior call may have stopped solely because retirement publication
    // was unavailable. Complete the consumed batch before attempting to start
    // another packet from its now-empty suffix.
    if (!fetched_words_.empty() && fetched_word_offset_ == fetched_words_.size()) {
      clear_batch();
      access_.reset();
      if (cursor_ == producer)
        return SdmaQueueServiceOutcome::Drained;
      access_ = gpu_vm_->snapshot(config_.address_space);
      if (!access_)
        return latch(SdmaQueueServiceOutcome::Faulted);
    }
  }

  for (;;) {
    if (!executor_.pending() && fetched_words_.empty()) {
      if (cursor_ == producer) {
        clear_batch();
        access_.reset();
        return SdmaQueueServiceOutcome::Drained;
      }
      const SdmaQueueServiceOutcome fetched = fetch(producer);
      if (fetched != SdmaQueueServiceOutcome::Drained)
        return fetched == SdmaQueueServiceOutcome::Unavailable ? fetched : latch(fetched);
    }

    const SdmaExecutionResult result =
        executor_.pending()
            ? executor_.resume()
            : executor_.start(
                  std::span<const uint32_t>(fetched_words_).subspan(fetched_word_offset_),
                  *access_);
    if (result.outcome == SdmaExecutionOutcome::Unavailable)
      return SdmaQueueServiceOutcome::Unavailable;

    const SdmaQueueServiceOutcome execution_outcome = map(result.outcome);
    if (result.retire_packet) {
      const std::size_t remaining = fetched_words_.size() - fetched_word_offset_;
      if (result.packet_dwords == 0 || result.packet_dwords > remaining)
        return latch(SdmaQueueServiceOutcome::Malformed);
      cursor_ += result.packet_dwords * sizeof(uint32_t);
      fetched_word_offset_ += result.packet_dwords;
      publication_pending_ = true;
      if (execution_outcome != SdmaQueueServiceOutcome::Drained)
        terminal_after_publication_ = execution_outcome;
      const SdmaQueueServiceOutcome published = publish_cursor();
      if (published != SdmaQueueServiceOutcome::Drained)
        return published == SdmaQueueServiceOutcome::Unavailable ? published : latch(published);
      if (terminal_after_publication_)
        return latch(*terminal_after_publication_);
    } else if (execution_outcome != SdmaQueueServiceOutcome::Drained) {
      return latch(execution_outcome);
    } else {
      return latch(SdmaQueueServiceOutcome::Malformed);
    }

    if (fetched_word_offset_ == fetched_words_.size()) {
      clear_batch();
      access_.reset();
      if (cursor_ == producer)
        return SdmaQueueServiceOutcome::Drained;
      access_ = gpu_vm_->snapshot(config_.address_space);
      if (!access_)
        return latch(SdmaQueueServiceOutcome::Faulted);
    }
  }
}

bool SdmaQueueRunner::in_flight() const {
  return access_.has_value() || batch_active_ || executor_.pending() || publication_pending_;
}

bool SdmaQueueRunner::reconfigure(SdmaQueueRunnerConfig config) {
  if (in_flight())
    return false;
  config_ = std::move(config);
  reset();
  return true;
}

void SdmaQueueRunner::reset() {
  executor_.reset();
  access_.reset();
  cursor_ = config_.initial_cursor.value_or(0);
  cursor_initialized_ = config_.initial_cursor.has_value();
  publication_pending_ = false;
  terminal_.reset();
  clear_batch();
}

} // namespace rocjitsu::amdgpu
