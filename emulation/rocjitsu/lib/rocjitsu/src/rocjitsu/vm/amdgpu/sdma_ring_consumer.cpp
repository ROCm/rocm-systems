// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/sdma_ring_consumer.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>
#include <utility>

namespace rocjitsu::amdgpu {
namespace {

SdmaRingStatus map(VmAccessOutcome outcome) {
  switch (outcome) {
  case VmAccessOutcome::Complete:
    return SdmaRingStatus::Idle;
  case VmAccessOutcome::Unavailable:
    return SdmaRingStatus::Blocked;
  case VmAccessOutcome::Faulted:
    return SdmaRingStatus::Faulted;
  case VmAccessOutcome::Malformed:
    return SdmaRingStatus::Malformed;
  }
  return SdmaRingStatus::Malformed;
}

SdmaRingStatus map(PacketProcessStatus status) {
  switch (status) {
  case PacketProcessStatus::Complete:
    return SdmaRingStatus::Idle;
  case PacketProcessStatus::NeedInput:
  case PacketProcessStatus::Blocked:
    return SdmaRingStatus::Blocked;
  case PacketProcessStatus::Faulted:
    return SdmaRingStatus::Faulted;
  case PacketProcessStatus::Malformed:
  case PacketProcessStatus::Unsupported:
    return SdmaRingStatus::Malformed;
  }
  return SdmaRingStatus::Malformed;
}

} // namespace

SdmaRingConsumer::SdmaRingConsumer(GpuVm &gpu_vm, SdmaRingConfig config,
                                   SdmaPacketProcessor packet_processor)
    : gpu_vm_(&gpu_vm), config_(std::move(config)),
      ring_reader_(config_.ring_base, config_.ring_bytes),
      cursor_journal_(config_.read_pointer_address, config_.initial_cursor),
      packet_processor_(std::move(packet_processor)) {}

SdmaRingConsumer::SdmaRingConsumer(GpuVm &gpu_vm, SdmaRingConfig config, SdmaPacketDialect dialect,
                                   SdmaPacketCallbacks callbacks)
    : SdmaRingConsumer(gpu_vm, std::move(config),
                       SdmaPacketProcessor(dialect, std::move(callbacks))) {}

SdmaRingConsumer::~SdmaRingConsumer() = default;
SdmaRingConsumer::SdmaRingConsumer(SdmaRingConsumer &&) noexcept = default;

std::optional<SdmaRingStatus> SdmaRingConsumer::validate_configuration() const {
  if (!config_.address_space || config_.ring_bytes == 0 ||
      (config_.ring_base % sizeof(uint32_t)) != 0 || (config_.ring_bytes % sizeof(uint32_t)) != 0 ||
      config_.read_pointer_address == 0 || (config_.read_pointer_address % sizeof(uint64_t)) != 0 ||
      config_.ring_base > std::numeric_limits<uint64_t>::max() - (config_.ring_bytes - 1) ||
      (config_.initial_cursor && (*config_.initial_cursor % sizeof(uint32_t)) != 0)) {
    return SdmaRingStatus::Malformed;
  }
  return std::nullopt;
}

SdmaRingStatus SdmaRingConsumer::latch(SdmaRingStatus outcome) {
  if (outcome == SdmaRingStatus::Faulted || outcome == SdmaRingStatus::Malformed) {
    terminal_ = outcome;
    if (!cursor_journal_.publication_pending())
      clear_in_flight();
  }
  return outcome;
}

SdmaRingStatus SdmaRingConsumer::initialize_cursor() {
  const VmAccessOutcome outcome = cursor_journal_.initialize(*access_);
  if (outcome != VmAccessOutcome::Complete)
    return map(outcome);
  if ((cursor_journal_.cursor() % sizeof(uint32_t)) != 0)
    return SdmaRingStatus::Malformed;
  return SdmaRingStatus::Idle;
}

SdmaRingStatus SdmaRingConsumer::ensure_packet_bytes(uint64_t producer,
                                                     std::size_t required_bytes) {
  if (required_bytes == 0 || (required_bytes % sizeof(uint32_t)) != 0 ||
      required_bytes > config_.ring_bytes || producer < cursor() ||
      required_bytes > producer - cursor()) {
    return SdmaRingStatus::Malformed;
  }
  if (required_bytes < packet_bytes_.size())
    return SdmaRingStatus::Malformed;

  packet_bytes_.resize(required_bytes);
  const VmAccessOutcome outcome =
      ring_reader_.read(*access_, cursor(), packet_bytes_, packet_fetch_progress_);
  if (outcome != VmAccessOutcome::Complete)
    return map(outcome);

  packet_words_.resize(packet_bytes_.size() / sizeof(uint32_t));
  std::memcpy(packet_words_.data(), packet_bytes_.data(), packet_bytes_.size());
  return SdmaRingStatus::Idle;
}

SdmaRingStatus SdmaRingConsumer::publish_cursor() { return map(cursor_journal_.publish()); }

void SdmaRingConsumer::clear_packet() {
  packet_processor_.reset(continuation_);
  packet_bytes_.clear();
  packet_fetch_progress_ = 0;
  packet_words_.clear();
  pending_retirement_.reset();
  terminal_after_publication_.reset();
}

void SdmaRingConsumer::clear_in_flight() {
  clear_packet();
  access_.reset();
}

SdmaRingStatus SdmaRingConsumer::service(uint64_t producer, std::size_t packet_budget) {
  if (terminal_)
    return *terminal_;
  if (packet_budget == 0)
    return SdmaRingStatus::Runnable;
  if (const std::optional<SdmaRingStatus> invalid = validate_configuration())
    return latch(*invalid);
  if (!access_) {
    access_ = gpu_vm_->snapshot(config_.address_space);
    if (!access_)
      return latch(SdmaRingStatus::Faulted);
  }

  const SdmaRingStatus initialized = initialize_cursor();
  if (initialized != SdmaRingStatus::Idle)
    return initialized == SdmaRingStatus::Blocked ? initialized : latch(initialized);

  if ((producer % sizeof(uint32_t)) != 0 || producer < cursor() ||
      producer - cursor() > config_.ring_bytes)
    return latch(SdmaRingStatus::Malformed);
  const std::size_t published_bytes = static_cast<std::size_t>(producer - cursor());
  if (!cursor_journal_.publication_pending() && packet_bytes_.size() > published_bytes)
    return latch(SdmaRingStatus::Malformed);

  if (cursor_journal_.publication_pending()) {
    const SdmaRingStatus published = publish_cursor();
    if (published != SdmaRingStatus::Idle)
      return published == SdmaRingStatus::Blocked ? published : latch(published);
    const std::optional<SdmaRingStatus> terminal = terminal_after_publication_;
    clear_packet();
    access_.reset();
    if (terminal)
      return latch(*terminal);
    if (cursor() == producer)
      return SdmaRingStatus::Idle;
    access_ = gpu_vm_->snapshot(config_.address_space);
    if (!access_)
      return latch(SdmaRingStatus::Faulted);
  }

  std::size_t retired_packets = 0;
  for (;;) {
    if (cursor() == producer) {
      clear_packet();
      access_.reset();
      return SdmaRingStatus::Idle;
    }

    const std::size_t available_bytes = static_cast<std::size_t>(producer - cursor());
    SdmaPacketProcessResult result;
    if (pending_retirement_) {
      result = *pending_retirement_;
    } else if (continuation_.pending()) {
      result = packet_processor_.process(
          {.available_dwords = {}, .access = *access_, .continuation = continuation_});
      const PacketProcessResult &continued = result.packet_result();
      const std::size_t validation_bytes =
          continued.status == PacketProcessStatus::NeedInput ? 0 : available_bytes;
      if (!valid_packet_process_result(continued, validation_bytes, sizeof(uint32_t)))
        return latch(SdmaRingStatus::Malformed);
    } else {
      const SdmaRingStatus fetched =
          ensure_packet_bytes(producer, std::max(packet_bytes_.size(), sizeof(uint32_t)));
      if (fetched != SdmaRingStatus::Idle)
        return fetched == SdmaRingStatus::Blocked ? fetched : latch(fetched);

      for (;;) {
        result = packet_processor_.process(
            {.available_dwords = packet_words_, .access = *access_, .continuation = continuation_});
        const PacketProcessResult &probe_result = result.packet_result();
        const std::size_t validation_bytes = probe_result.status == PacketProcessStatus::NeedInput
                                                 ? packet_bytes_.size()
                                                 : available_bytes;
        if (!valid_packet_process_result(probe_result, validation_bytes, sizeof(uint32_t)))
          return latch(SdmaRingStatus::Malformed);
        if (probe_result.status != PacketProcessStatus::NeedInput)
          break;
        const SdmaRingStatus expanded = ensure_packet_bytes(producer, probe_result.required_bytes);
        if (expanded != SdmaRingStatus::Idle)
          return expanded == SdmaRingStatus::Blocked ? expanded : latch(expanded);
      }
    }
    const PacketProcessResult &packet_result = result.packet_result();
    if (packet_result.status == PacketProcessStatus::NeedInput)
      return latch(SdmaRingStatus::Malformed);
    if (packet_result.status == PacketProcessStatus::Blocked)
      return SdmaRingStatus::Blocked;

    const SdmaRingStatus processing_status = map(packet_result.status);
    if (packet_result.retirement == PacketRetirement::Retire) {
      // COND_EXE can determine its skipped retirement extent from the header
      // alone. Preserve that completed decision, but do not advance beyond the
      // producer-visible ring extent. Retrying must not re-read the condition
      // after the producer publishes the skipped dwords.
      if (packet_result.retirement_bytes > config_.ring_bytes)
        return latch(SdmaRingStatus::Malformed);
      if (packet_result.retirement_bytes > available_bytes) {
        pending_retirement_ = result;
        return SdmaRingStatus::Blocked;
      }
      const uint64_t retired_cursor = cursor() + packet_result.retirement_bytes;
      cursor_journal_.retire(retired_cursor, *access_);
      if (processing_status != SdmaRingStatus::Idle)
        terminal_after_publication_ = processing_status;
      const SdmaRingStatus published = publish_cursor();
      if (published != SdmaRingStatus::Idle)
        return published == SdmaRingStatus::Blocked ? published : latch(published);
      const std::optional<SdmaRingStatus> terminal = terminal_after_publication_;
      clear_packet();
      access_.reset();
      if (terminal)
        return latch(*terminal);
      ++retired_packets;
    } else if (processing_status != SdmaRingStatus::Idle) {
      return latch(processing_status);
    } else {
      return latch(SdmaRingStatus::Malformed);
    }

    if (cursor() == producer)
      return SdmaRingStatus::Idle;
    if (retired_packets >= packet_budget)
      return SdmaRingStatus::Runnable;
    access_ = gpu_vm_->snapshot(config_.address_space);
    if (!access_)
      return latch(SdmaRingStatus::Faulted);
  }
}

bool SdmaRingConsumer::in_flight() const {
  return access_.has_value() || !packet_bytes_.empty() || continuation_.pending() ||
         cursor_journal_.publication_pending();
}

bool SdmaRingConsumer::reconfigure(SdmaRingConfig config) {
  if (in_flight())
    return false;
  config_ = std::move(config);
  ring_reader_ = CircularRingReader(config_.ring_base, config_.ring_bytes);
  cursor_journal_ = ConsumerCursorJournal(config_.read_pointer_address, config_.initial_cursor);
  reset();
  return true;
}

void SdmaRingConsumer::reset() {
  clear_in_flight();
  cursor_journal_.reset(config_.initial_cursor);
  terminal_.reset();
}

} // namespace rocjitsu::amdgpu
