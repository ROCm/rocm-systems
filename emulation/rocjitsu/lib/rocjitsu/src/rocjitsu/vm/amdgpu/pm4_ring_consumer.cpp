// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pm4_ring_consumer.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace rocjitsu::amdgpu {
namespace {

Pm4RingStatus map(VmAccessOutcome outcome) {
  switch (outcome) {
  case VmAccessOutcome::Complete:
    return Pm4RingStatus::Complete;
  case VmAccessOutcome::Unavailable:
    return Pm4RingStatus::Blocked;
  case VmAccessOutcome::Faulted:
    return Pm4RingStatus::Faulted;
  case VmAccessOutcome::Malformed:
    return Pm4RingStatus::Malformed;
  }
  return Pm4RingStatus::Malformed;
}

Pm4RingStatus map(const Pm4PacketProcessResult &result) {
  if (result.diagnostic == Pm4PacketDiagnostic::UnsupportedPacket)
    return Pm4RingStatus::UnsupportedPacket;
  if (result.diagnostic == Pm4PacketDiagnostic::RegisterWriteRejected)
    return Pm4RingStatus::RegisterWriteRejected;
  switch (result.packet.status) {
  case PacketProcessStatus::Complete:
    return Pm4RingStatus::Complete;
  case PacketProcessStatus::NeedInput:
  case PacketProcessStatus::Blocked:
    return Pm4RingStatus::Blocked;
  case PacketProcessStatus::Faulted:
    return Pm4RingStatus::Faulted;
  case PacketProcessStatus::Malformed:
    return Pm4RingStatus::Malformed;
  case PacketProcessStatus::Unsupported:
    return Pm4RingStatus::UnsupportedPacket;
  }
  return Pm4RingStatus::Malformed;
}

} // namespace

Pm4RingConsumer::Pm4RingConsumer(Pm4PacketProcessor &packet_processor, GpuVm &gpu_vm,
                                 AddressSpaceHandle address_space, uint64_t ring_base,
                                 uint64_t ring_bytes, uint64_t consumer_pointer_address,
                                 std::optional<uint64_t> initial_consumer_cursor)
    : packet_processor_(&packet_processor), gpu_vm_(&gpu_vm), address_space_(address_space),
      ring_base_(ring_base), ring_bytes_(ring_bytes),
      cursor_journal_(consumer_pointer_address, initial_consumer_cursor, sizeof(uint32_t)) {}

bool Pm4RingConsumer::in_flight() const noexcept {
  return access_.has_value() || transaction_producer_cursor_.has_value() ||
         cursor_journal_.publication_pending();
}

bool Pm4RingConsumer::publication_pending() const noexcept {
  return cursor_journal_.publication_pending();
}

uint64_t Pm4RingConsumer::consumer_cursor() const noexcept { return cursor_journal_.cursor(); }

QueuePrepareCloseStatus Pm4RingConsumer::prepare_close() noexcept {
  if (cursor_journal_.publication_pending())
    return publication_faulted_ ? QueuePrepareCloseStatus::Faulted : QueuePrepareCloseStatus::Busy;
  reset();
  return QueuePrepareCloseStatus::Ready;
}

void Pm4RingConsumer::reconfigure(uint64_t ring_base, uint64_t ring_bytes) noexcept {
  reset();
  ring_base_ = ring_base;
  ring_bytes_ = ring_bytes;
}

void Pm4RingConsumer::release_transaction() noexcept {
  access_.reset();
  transaction_producer_cursor_.reset();
  terminal_after_publication_.reset();
}

void Pm4RingConsumer::reset() noexcept {
  release_transaction();
  cursor_journal_.reset();
  publication_faulted_ = false;
}

Pm4RingResult Pm4RingConsumer::consume(uint64_t producer_cursor, std::size_t packet_budget) {
  Pm4RingResult result{.read_pointer = cursor_journal_.cursor()};
  if (packet_budget == 0) {
    result.status = Pm4RingStatus::Complete;
    return result;
  }
  if (ring_base_ == 0 || ring_bytes_ == 0 || (ring_bytes_ % sizeof(uint32_t)) != 0) {
    return result;
  }
  if (!access_) {
    access_ = gpu_vm_->snapshot(address_space_);
    if (!access_) {
      result.status = Pm4RingStatus::Faulted;
      return result;
    }
    transaction_producer_cursor_ = producer_cursor;
  }
  if (!transaction_producer_cursor_) {
    result.status = Pm4RingStatus::Faulted;
    release_transaction();
    return result;
  }

  const VmAccessOutcome initialized = cursor_journal_.initialize(*access_);
  if (initialized == VmAccessOutcome::Unavailable) {
    result.status = Pm4RingStatus::Blocked;
    return result;
  }
  if (initialized != VmAccessOutcome::Complete) {
    result.status = map(initialized);
    release_transaction();
    return result;
  }
  result.read_pointer = cursor_journal_.cursor();

  const uint64_t transaction_producer_cursor = *transaction_producer_cursor_;
  const uint64_t ring_dwords = ring_bytes_ / sizeof(uint32_t);
  if (transaction_producer_cursor < cursor_journal_.cursor() ||
      transaction_producer_cursor - cursor_journal_.cursor() > ring_dwords) {
    result.status = Pm4RingStatus::Malformed;
    release_transaction();
    return result;
  }

  if (cursor_journal_.publication_pending()) {
    const VmAccessOutcome published = cursor_journal_.publish();
    result.read_pointer = cursor_journal_.cursor();
    if (published == VmAccessOutcome::Unavailable) {
      result.status = Pm4RingStatus::Blocked;
      return result;
    }
    if (published != VmAccessOutcome::Complete) {
      publication_faulted_ = published != VmAccessOutcome::Complete;
      result.status = map(published);
      if (!cursor_journal_.publication_pending())
        release_transaction();
      return result;
    }
    if (terminal_after_publication_) {
      result = *terminal_after_publication_;
      result.read_pointer = cursor_journal_.cursor();
      release_transaction();
      return result;
    }
  }

  if (cursor_journal_.cursor() == transaction_producer_cursor) {
    result.status = Pm4RingStatus::Complete;
    result.read_pointer = cursor_journal_.cursor();
    release_transaction();
    return result;
  }

  result = consume_packets(*access_, cursor_journal_.cursor(), transaction_producer_cursor,
                           packet_budget);
  if (result.read_pointer != cursor_journal_.cursor()) {
    cursor_journal_.retire(result.read_pointer, *access_);
    if (result.status != Pm4RingStatus::Complete && result.status != Pm4RingStatus::Blocked) {
      terminal_after_publication_ = result;
    }
    const VmAccessOutcome published = cursor_journal_.publish();
    if (published == VmAccessOutcome::Unavailable) {
      result.status = Pm4RingStatus::Blocked;
      return result;
    }
    if (published != VmAccessOutcome::Complete || terminal_after_publication_) {
      publication_faulted_ = published != VmAccessOutcome::Complete;
      if (published == VmAccessOutcome::Complete) {
        result = *terminal_after_publication_;
        result.read_pointer = cursor_journal_.cursor();
      } else {
        result.status = map(published);
      }
      if (!cursor_journal_.publication_pending())
        release_transaction();
      return result;
    }
  }

  result.read_pointer = cursor_journal_.cursor();
  if (result.status == Pm4RingStatus::Blocked)
    return result;
  if (result.status != Pm4RingStatus::Complete ||
      cursor_journal_.cursor() != transaction_producer_cursor) {
    release_transaction();
    return result;
  }
  release_transaction();
  return result;
}

Pm4RingResult Pm4RingConsumer::consume_packets(const GpuVmAccess &access, uint64_t read_pointer,
                                               uint64_t write_pointer,
                                               std::size_t packet_budget) const {
  Pm4RingResult ring_result{.read_pointer = read_pointer};
  const uint64_t ring_dwords = ring_bytes_ / sizeof(uint32_t);
  if (ring_base_ == 0 || ring_dwords == 0 ||
      ring_dwords > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t) ||
      ring_base_ > std::numeric_limits<uint64_t>::max() - ring_dwords * sizeof(uint32_t) ||
      write_pointer < read_pointer || write_pointer - read_pointer > ring_dwords) {
    return ring_result;
  }

  const CircularRingReader ring_reader(ring_base_, ring_dwords * sizeof(uint32_t));
  std::size_t retired_packets = 0;
  while (ring_result.read_pointer != write_pointer) {
    std::array<std::byte, 3 * sizeof(uint32_t)> raw_packet{};
    const uint64_t byte_cursor = (ring_result.read_pointer % ring_dwords) * sizeof(uint32_t);
    const VmAccessOutcome header_outcome =
        ring_reader.read(access, byte_cursor, std::span(raw_packet).first(sizeof(uint32_t)));
    if (header_outcome != VmAccessOutcome::Complete) {
      ring_result.status = map(header_outcome);
      return ring_result;
    }

    std::array<uint32_t, 3> packet = std::bit_cast<std::array<uint32_t, 3>>(raw_packet);
    Pm4PacketProcessResult packet_result =
        packet_processor_->process({.available_dwords = std::span(packet).first(1)});
    std::size_t supplied_bytes = sizeof(uint32_t);
    ring_result.packet_header = packet_result.packet_header;
    ring_result.register_dword = packet_result.register_dword;
    if (!valid_packet_process_result(packet_result.packet_result(), sizeof(uint32_t),
                                     sizeof(uint32_t))) {
      throw std::logic_error("PM4 processor returned an invalid common packet result");
    }

    if (packet_result.packet_result().status == PacketProcessStatus::NeedInput) {
      if (packet_result.packet_result().required_bytes > raw_packet.size() ||
          packet_result.packet_result().required_bytes / sizeof(uint32_t) >
              write_pointer - ring_result.read_pointer) {
        ring_result.status = Pm4RingStatus::Malformed;
        return ring_result;
      }
      const VmAccessOutcome packet_outcome = ring_reader.read(
          access, byte_cursor,
          std::span(raw_packet).first(packet_result.packet_result().required_bytes));
      if (packet_outcome != VmAccessOutcome::Complete) {
        ring_result.status = map(packet_outcome);
        return ring_result;
      }
      packet = std::bit_cast<std::array<uint32_t, 3>>(raw_packet);
      supplied_bytes = packet_result.packet_result().required_bytes;
      packet_result = packet_processor_->process(
          {.available_dwords = std::span(packet).first(supplied_bytes / sizeof(uint32_t))});
      ring_result.packet_header = packet_result.packet_header;
      ring_result.register_dword = packet_result.register_dword;
      if (!valid_packet_process_result(packet_result.packet_result(), supplied_bytes,
                                       sizeof(uint32_t))) {
        throw std::logic_error("PM4 processor returned an invalid common packet result");
      }
    }

    ring_result.status = map(packet_result);
    const PacketProcessResult &common_result = packet_result.packet_result();
    if (common_result.retirement != PacketRetirement::Retire)
      return ring_result;
    if (common_result.retirement_bytes > supplied_bytes) {
      ring_result.status = Pm4RingStatus::Malformed;
      return ring_result;
    }
    ring_result.read_pointer += common_result.retirement_bytes / sizeof(uint32_t);
    if (++retired_packets >= packet_budget)
      break;
  }

  ring_result.status = Pm4RingStatus::Complete;
  return ring_result;
}

} // namespace rocjitsu::amdgpu
