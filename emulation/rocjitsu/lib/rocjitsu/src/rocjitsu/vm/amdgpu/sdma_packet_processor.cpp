// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/sdma_packet_processor.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

enum class SdmaPacketExecutionOutcome {
  Complete,
  NeedInput,
  Unavailable,
  Faulted,
  Malformed,
};

struct SdmaPacketExecutionResult {
  SdmaPacketExecutionOutcome outcome = SdmaPacketExecutionOutcome::Malformed;
  std::size_t packet_dwords = 0;
  std::size_t required_dwords = 0;
  bool operation_committed = false;
  bool completion_published = false;
  bool retire_packet = false;
};

constexpr uint8_t kOpNop = 0;
constexpr uint8_t kOpCopy = 1;
constexpr uint8_t kOpWrite = 2;
constexpr uint8_t kOpIndirect = 4;
constexpr uint8_t kOpFence = 5;
constexpr uint8_t kOpTrap = 6;
constexpr uint8_t kOpPollRegmem = 8;
constexpr uint8_t kOpConditionalExecute = 9;
constexpr uint8_t kOpAtomic = 10;
constexpr uint8_t kOpConstantFill = 11;
constexpr uint8_t kOpTimestamp = 13;
constexpr uint8_t kOpRegisterWrite = 14;
constexpr uint8_t kOpGcr = 17;
constexpr uint8_t kOpHdpFlush = 0x26;

constexpr uint8_t kSubopLinear = 0;
constexpr uint8_t kSubopFence64 = 2;
constexpr uint8_t kSubopPollMemory64 = 5;

constexpr std::size_t kTransferBytes = 4096;
constexpr std::size_t kCopyLinearDwords = 7;
constexpr std::size_t kCopyBroadcastDwords = 9;
constexpr std::size_t kFenceDwords = 4;
constexpr std::size_t kFence64Dwords = 5;
constexpr std::size_t kTrapDwords = 2;
constexpr std::size_t kPollDwords = 6;
constexpr std::size_t kPoll64Dwords = 8;
constexpr std::size_t kAtomicDwords = 8;
constexpr std::size_t kFillDwords = 5;
constexpr std::size_t kTimestampDwords = 3;
constexpr std::size_t kIndirectDwords = 6;
constexpr std::size_t kConditionalDwords = 5;
constexpr std::size_t kRegisterWriteDwords = 3;
constexpr std::size_t kLegacyGcrDwords = 5;
constexpr std::size_t kGfx1250GcrDwords = 6;
constexpr std::size_t kWaitDwords = 7;
constexpr std::size_t kCopyBodyDwords = 6;
constexpr std::size_t kSignalDwords = 5;

uint64_t join(uint32_t low, uint32_t high) {
  return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
}

bool compare(uint32_t function, uint64_t value, uint64_t reference) {
  switch (function) {
  case 0:
    return true;
  case 1:
    return value < reference;
  case 2:
    return value <= reference;
  case 3:
    return value == reference;
  case 4:
    return value != reference;
  case 5:
    return value >= reference;
  case 6:
    return value > reference;
  default:
    return true;
  }
}

bool valid_range(uint64_t address, uint64_t size) {
  return size != 0 && size - 1 <= std::numeric_limits<uint64_t>::max() - address;
}

SdmaPacketExecutionOutcome map_outcome(VmAccessOutcome outcome) {
  switch (outcome) {
  case VmAccessOutcome::Complete:
    return SdmaPacketExecutionOutcome::Complete;
  case VmAccessOutcome::Unavailable:
    return SdmaPacketExecutionOutcome::Unavailable;
  case VmAccessOutcome::Faulted:
    return SdmaPacketExecutionOutcome::Faulted;
  case VmAccessOutcome::Malformed:
    return SdmaPacketExecutionOutcome::Malformed;
  }
  return SdmaPacketExecutionOutcome::Malformed;
}

} // namespace

class SdmaPacketContinuation::Impl {
public:
  enum class Kind {
    Nop,
    Copy,
    Fence,
    Trap,
    Poll,
    Atomic,
    Fill,
    Timestamp,
    Gcr,
    HdpFlush,
    Write,
    Indirect,
    Conditional,
    RegisterWrite,
  };

  class Frame {
  public:
    std::vector<uint32_t> words;
    std::size_t at = 0;
    bool root = false;
  };

  class Operation {
  public:
    Kind kind = Kind::Nop;
    std::size_t dwords = 0;
    std::size_t skip_dwords = 0;
    uint32_t phase = 0;
    uint32_t function = 0;
    uint32_t value32 = 0;
    uint32_t mask32 = 0;
    uint32_t fill_size = 0;
    uint64_t address = 0;
    uint64_t source = 0;
    uint64_t reference = 0;
    uint64_t mask = 0;
    uint64_t value = 0;
    uint64_t count = 0;
    uint64_t completed = 0;
    std::array<uint64_t, 2> destinations{};
    std::size_t destination_count = 0;
    std::size_t destination_index = 0;
    std::size_t io_progress = 0;
    std::size_t chunk_size = 0;
    std::array<std::byte, kTransferBytes> scratch{};
    std::vector<std::byte> payload;
    std::vector<std::byte> indirect;
    std::optional<SdmaCacheLease> cache_lease;
    bool wait_enabled = false;
    bool signal_enabled = false;
    bool signal_decrement = false;
    uint64_t signal_address = 0;
    uint64_t signal_data = 0;
    bool cas_expected_valid = false;
    uint64_t cas_expected = 0;
    bool completes_signal = false;
    bool timestamp_captured = false;
    bool retire_on_fault = false;
    uint64_t mailbox = 0;
    uint32_t event_id = 0;
    std::array<std::byte, sizeof(uint64_t)> mailbox_bytes{};
    std::array<std::byte, sizeof(uint32_t)> event_bytes{};
    std::size_t mailbox_read_progress = 0;
    std::size_t event_read_progress = 0;
  };

  void clear() {
    frames.clear();
    operation.reset();
    root_dwords = 0;
    operation_committed = false;
    completion_published = false;
    retire_packet = false;
    pending = false;
  }

  std::vector<Frame> frames;
  std::optional<Operation> operation;
  std::size_t root_dwords = 0;
  bool operation_committed = false;
  bool completion_published = false;
  bool retire_packet = false;
  bool pending = false;
};

class SdmaPacketProcessor::Impl {
public:
  explicit Impl(SdmaPacketDialect dialect, SdmaPacketCallbacks callbacks)
      : dialect_(dialect), callbacks_(std::move(callbacks)) {}

  using State = SdmaPacketContinuation::Impl;
  using Kind = State::Kind;
  using Frame = State::Frame;
  using Operation = State::Operation;

  struct PacketExtent {
    SdmaPacketExecutionOutcome outcome = SdmaPacketExecutionOutcome::Malformed;
    std::size_t dwords = 0;
  };

  SdmaPacketExecutionResult process(std::span<const uint32_t> words, const GpuVmAccess &access,
                                    State &state) {
    if (state_ != nullptr)
      return {};
    state_ = &state;
    access_ = &access;
    struct Invocation {
      Impl &owner;
      ~Invocation() {
        owner.access_ = nullptr;
        owner.state_ = nullptr;
      }
    } invocation{*this};

    if (state_->pending)
      return run();

    const PacketExtent extent = packet_extent(words);
    if (extent.outcome != SdmaPacketExecutionOutcome::Complete) {
      return {.outcome = extent.outcome,
              .packet_dwords = extent.dwords,
              .required_dwords =
                  extent.outcome == SdmaPacketExecutionOutcome::NeedInput ? extent.dwords : 0};
    }
    clear();
    state_->frames.push_back(
        {.words = std::vector<uint32_t>(words.begin(), words.begin() + extent.dwords),
         .at = 0,
         .root = true});
    state_->pending = true;
    return run();
  }

  void clear() {
    if (state_ != nullptr)
      state_->clear();
  }

  void clear(State &state) noexcept { state.clear(); }

private:
  SdmaPacketExecutionResult result(SdmaPacketExecutionOutcome outcome) const {
    return {.outcome = outcome,
            .packet_dwords = state_->root_dwords,
            .operation_committed = state_->operation_committed,
            .completion_published = state_->completion_published,
            .retire_packet = state_->retire_packet};
  }

  SdmaPacketExecutionResult finish(SdmaPacketExecutionOutcome outcome) {
    if (outcome == SdmaPacketExecutionOutcome::Complete) {
      state_->completion_published = true;
      state_->retire_packet = true;
    } else if ((outcome == SdmaPacketExecutionOutcome::Faulted ||
                outcome == SdmaPacketExecutionOutcome::Malformed) &&
               (state_->operation_committed ||
                (state_->operation && state_->operation->retire_on_fault))) {
      // Never expose an already-committed prefix for replay. Copy data faults
      // also preserve the established CP policy of retiring the failed copy.
      state_->retire_packet = true;
    }
    const SdmaPacketExecutionResult final = result(outcome);
    clear();
    return final;
  }

  bool has(const Frame &frame, std::size_t count) const {
    return frame.at <= frame.words.size() && count <= frame.words.size() - frame.at;
  }

  uint32_t word(const Frame &frame, std::size_t offset) const {
    return frame.words[frame.at + offset];
  }

  bool gfx11_plus() const { return dialect_ != SdmaPacketDialect::Legacy; }

  PacketExtent packet_extent(std::span<const uint32_t> words) const {
    const auto need = [&](std::size_t dwords) {
      return PacketExtent{.outcome = words.size() < dwords ? SdmaPacketExecutionOutcome::NeedInput
                                                           : SdmaPacketExecutionOutcome::Complete,
                          .dwords = dwords};
    };
    if (words.empty())
      return need(1);

    const uint32_t header = words.front();
    const uint8_t opcode = header & 0xff;
    const uint8_t subopcode = (header >> 8) & 0xff;
    switch (opcode) {
    case kOpNop:
      return need(((header >> 16) & 0x3fff) + 1);
    case kOpCopy: {
      constexpr uint8_t kSubopLinearBroadcast = 16;
      if (subopcode != kSubopLinear && subopcode != kSubopLinearBroadcast)
        return {};
      if (gfx11_plus() && (header & ((1u << 30) | (1u << 31))) != 0) {
        const std::size_t copy_base = 1 + ((header & (1u << 30)) != 0 ? kWaitDwords : 0);
        const std::size_t signal_base = copy_base + kCopyBodyDwords;
        return need(signal_base + ((header & (1u << 31)) != 0 ? kSignalDwords : 0));
      }
      const bool broadcast =
          subopcode == kSubopLinearBroadcast ||
          (gfx11_plus() ? (header & (1u << 27)) != 0 : (header & (1u << 28)) != 0);
      return need(broadcast ? kCopyBroadcastDwords : kCopyLinearDwords);
    }
    case kOpFence:
      return need(gfx11_plus() && subopcode == kSubopFence64 ? kFence64Dwords : kFenceDwords);
    case kOpTrap:
      return need(kTrapDwords);
    case kOpPollRegmem:
      return need(gfx11_plus() && subopcode == kSubopPollMemory64 ? kPoll64Dwords : kPollDwords);
    case kOpAtomic:
      return ((header >> 25) & 0x7f) == 47 ? need(kAtomicDwords) : PacketExtent{};
    case kOpConstantFill:
      return need(kFillDwords);
    case kOpTimestamp:
      return need(kTimestampDwords);
    case kOpRegisterWrite:
      return need(kRegisterWriteDwords);
    case kOpGcr:
      return need(dialect_ == SdmaPacketDialect::Gfx1250 ? kGfx1250GcrDwords : kLegacyGcrDwords);
    case kOpHdpFlush:
      return need(1);
    case kOpWrite: {
      if (subopcode != kSubopLinear)
        return {};
      if (words.size() < 4)
        return need(4);
      const std::size_t count = (words[3] & 0x000fffff) + std::size_t{1};
      if (count > std::numeric_limits<std::size_t>::max() - 4)
        return {};
      return need(4 + count);
    }
    case kOpIndirect:
      return need(kIndirectDwords);
    case kOpConditionalExecute:
      return need(kConditionalDwords);
    default:
      return {};
    }
  }

  SdmaPacketExecutionOutcome decode(Frame &frame) {
    if (!has(frame, 1))
      return SdmaPacketExecutionOutcome::Malformed;

    const PacketExtent extent =
        packet_extent(std::span<const uint32_t>(frame.words).subspan(frame.at));
    if (extent.outcome != SdmaPacketExecutionOutcome::Complete)
      return SdmaPacketExecutionOutcome::Malformed;

    Operation operation;
    operation.dwords = extent.dwords;
    const uint32_t header = word(frame, 0);
    const uint8_t opcode = header & 0xff;
    const uint8_t subopcode = (header >> 8) & 0xff;

    switch (opcode) {
    case kOpNop:
      operation.kind = Kind::Nop;
      break;
    case kOpCopy: {
      operation.kind = Kind::Copy;
      constexpr uint8_t kSubopLinearBroadcast = 16;
      if (subopcode != kSubopLinear && subopcode != kSubopLinearBroadcast)
        return SdmaPacketExecutionOutcome::Malformed;
      if (gfx11_plus() && (header & ((1u << 30) | (1u << 31))) != 0) {
        operation.wait_enabled = (header & (1u << 30)) != 0;
        operation.signal_enabled = (header & (1u << 31)) != 0;
        const std::size_t copy_base = 1 + (operation.wait_enabled ? kWaitDwords : 0);
        const std::size_t signal_base = copy_base + kCopyBodyDwords;
        if (operation.wait_enabled) {
          operation.function = word(frame, 1) & 0x7;
          operation.address = join(word(frame, 2) & ~0x7u, word(frame, 3));
          operation.reference = join(word(frame, 4), word(frame, 5));
          operation.mask = join(word(frame, 6), word(frame, 7));
        }
        operation.count = (word(frame, copy_base) & 0x3fffffff) + uint64_t{1};
        operation.source = join(word(frame, copy_base + 2), word(frame, copy_base + 3));
        operation.destinations[0] = join(word(frame, copy_base + 4), word(frame, copy_base + 5));
        operation.destination_count = 1;
        if (!valid_range(operation.source, operation.count) ||
            !valid_range(operation.destinations[0], operation.count))
          return SdmaPacketExecutionOutcome::Malformed;
        if (operation.signal_enabled) {
          const uint32_t signal_op = word(frame, signal_base) & 0x7f;
          operation.signal_address =
              join(word(frame, signal_base + 1) & ~0x7u, word(frame, signal_base + 2));
          operation.signal_data = join(word(frame, signal_base + 3), word(frame, signal_base + 4));
          operation.signal_decrement = operation.signal_address > 0x1000 && signal_op == 0x70;
        }
        break;
      }
      const bool broadcast =
          subopcode == kSubopLinearBroadcast ||
          (gfx11_plus() ? (header & (1u << 27)) != 0 : (header & (1u << 28)) != 0);
      const uint32_t count_mask =
          broadcast || dialect_ == SdmaPacketDialect::Legacy ? 0x003fffff : 0x3fffffff;
      operation.count = (word(frame, 1) & count_mask) + uint64_t{1};
      operation.source = join(word(frame, 3), word(frame, 4));
      operation.destinations[0] = join(word(frame, 5), word(frame, 6));
      operation.destination_count = broadcast ? 2 : 1;
      if (broadcast)
        operation.destinations[1] = join(word(frame, 7), word(frame, 8));
      if (!valid_range(operation.source, operation.count) ||
          !valid_range(operation.destinations[0], operation.count) ||
          (broadcast && !valid_range(operation.destinations[1], operation.count)))
        return SdmaPacketExecutionOutcome::Malformed;
      break;
    }
    case kOpFence:
      operation.kind = Kind::Fence;
      if (gfx11_plus() && subopcode == kSubopFence64) {
        operation.address = join(word(frame, 1) & ~0x7u, word(frame, 2));
        operation.value = join(word(frame, 3), word(frame, 4));
        operation.count = sizeof(uint64_t);
      } else {
        operation.address = join(word(frame, 1), word(frame, 2));
        operation.value = word(frame, 3);
        operation.count = sizeof(uint32_t);
      }
      break;
    case kOpTrap:
      operation.kind = Kind::Trap;
      operation.value32 = word(frame, 1) & 0x0fffffff;
      break;
    case kOpPollRegmem:
      operation.kind = Kind::Poll;
      if (gfx11_plus() && subopcode == kSubopPollMemory64) {
        operation.function = (header >> 28) & 0x7;
        operation.address = join(word(frame, 1) & ~0x7u, word(frame, 2));
        operation.reference = join(word(frame, 3), word(frame, 4));
        operation.mask = join(word(frame, 5), word(frame, 6));
        operation.count = sizeof(uint64_t);
        operation.value32 = 1;
      } else {
        operation.function = (header >> 28) & 0x7;
        operation.address = join(word(frame, 1), word(frame, 2));
        operation.reference = word(frame, 3);
        operation.mask = word(frame, 4);
        operation.count = sizeof(uint32_t);
        operation.value32 = (header >> 31) & 1;
      }
      break;
    case kOpAtomic:
      operation.kind = Kind::Atomic;
      operation.address = join(word(frame, 1), word(frame, 2));
      operation.value = join(word(frame, 3), word(frame, 4));
      operation.completes_signal =
          static_cast<int64_t>(operation.value) < 0 && callbacks_.deliver_interrupt;
      break;
    case kOpConstantFill:
      operation.kind = Kind::Fill;
      operation.address = join(word(frame, 1), word(frame, 2));
      operation.value32 = word(frame, 3);
      operation.count = (word(frame, 4) & 0x3fffffff) + uint64_t{1};
      operation.fill_size = (header >> 30) & 0x3;
      break;
    case kOpTimestamp:
      operation.kind = Kind::Timestamp;
      operation.address = join(word(frame, 1), word(frame, 2));
      break;
    case kOpGcr:
      operation.kind = Kind::Gcr;
      operation.value32 = word(frame, dialect_ == SdmaPacketDialect::Gfx1250 ? 3 : 2);
      break;
    case kOpHdpFlush:
      operation.kind = Kind::HdpFlush;
      break;
    case kOpWrite: {
      operation.kind = Kind::Write;
      const std::size_t count = (word(frame, 3) & 0x000fffff) + std::size_t{1};
      if (count > std::numeric_limits<std::size_t>::max() / 4)
        return SdmaPacketExecutionOutcome::Malformed;
      operation.address = join(word(frame, 1), word(frame, 2));
      operation.count = count * sizeof(uint32_t);
      if (!valid_range(operation.address, operation.count))
        return SdmaPacketExecutionOutcome::Malformed;
      operation.payload.resize(static_cast<std::size_t>(operation.count));
      std::memcpy(operation.payload.data(), &frame.words[frame.at + 4], operation.payload.size());
      break;
    }
    case kOpIndirect: {
      operation.kind = Kind::Indirect;
      const std::size_t count = word(frame, 3) & 0x000fffff;
      if (count > std::numeric_limits<std::size_t>::max() / sizeof(uint32_t))
        return SdmaPacketExecutionOutcome::Malformed;
      operation.address = join(word(frame, 1), word(frame, 2));
      operation.indirect.resize(count * sizeof(uint32_t));
      if (!operation.indirect.empty() && !valid_range(operation.address, operation.indirect.size()))
        return SdmaPacketExecutionOutcome::Malformed;
      break;
    }
    case kOpConditionalExecute:
      operation.kind = Kind::Conditional;
      operation.address = join(word(frame, 1), word(frame, 2));
      operation.reference = word(frame, 3);
      operation.skip_dwords = word(frame, 4) & 0x3fff;
      break;
    case kOpRegisterWrite:
      operation.kind = Kind::RegisterWrite;
      operation.address = word(frame, 1);
      operation.value32 = word(frame, 2);
      break;
    default:
      return SdmaPacketExecutionOutcome::Malformed;
    }

    if (!has(frame, operation.dwords))
      return SdmaPacketExecutionOutcome::Malformed;
    if (frame.root) {
      state_->root_dwords = operation.dwords;
      // COND_EXE controls the dwords following its five-dword header. Keep the
      // available root stream until its predicate determines whether those
      // dwords execute as later packets or retire as one skipped region.
      if (operation.kind != Kind::Conditional)
        frame.words.resize(frame.at + operation.dwords);
    }
    state_->operation.emplace(std::move(operation));
    return SdmaPacketExecutionOutcome::Complete;
  }

  SdmaPacketExecutionOutcome cache_before_write(Operation &operation) {
    if (!operation.cache_lease && callbacks_.acquire_cache_maintenance)
      operation.cache_lease.emplace(
          callbacks_.acquire_cache_maintenance(SdmaCacheOperation::WritebackInvalidate));
    return SdmaPacketExecutionOutcome::Complete;
  }

  SdmaPacketExecutionOutcome atomic_fetch_add(Operation &operation, uint64_t address,
                                              uint64_t amount) {
    if (!operation.cas_expected_valid) {
      const AtomicLoadResult loaded = access_->atomic_load(address, sizeof(uint64_t));
      if (loaded.outcome != VmAccessOutcome::Complete)
        return map_outcome(loaded.outcome);
      operation.cas_expected = loaded.value;
      operation.cas_expected_valid = true;
    }
    for (;;) {
      const AtomicCompareExchangeResult exchanged = access_->compare_exchange(
          address, sizeof(uint64_t), operation.cas_expected, operation.cas_expected + amount);
      if (exchanged.outcome != VmAccessOutcome::Complete)
        return map_outcome(exchanged.outcome);
      if (exchanged.exchanged) {
        operation.cas_expected_valid = false;
        state_->operation_committed = true;
        return SdmaPacketExecutionOutcome::Complete;
      }
      operation.cas_expected = exchanged.observed;
    }
  }

  SdmaPacketExecutionOutcome execute_copy(Operation &operation) {
    if (operation.phase == 0) {
      if (operation.wait_enabled && operation.address > 0x1000) {
        const VmAccessOutcome outcome =
            access_->read(operation.address, std::span(operation.scratch).first(sizeof(uint64_t)),
                          operation.io_progress);
        if (outcome != VmAccessOutcome::Complete)
          return map_outcome(outcome);
        uint64_t value = 0;
        std::memcpy(&value, operation.scratch.data(), sizeof(value));
        operation.io_progress = 0;
        if (!compare(operation.function, value & operation.mask, operation.reference))
          return SdmaPacketExecutionOutcome::Unavailable;
      }
      operation.phase = 1;
    }
    if (operation.phase == 1) {
      if (operation.signal_decrement) {
        const VmAccessOutcome outcome = access_->read(
            operation.signal_address, std::span(operation.scratch).first(sizeof(uint64_t)),
            operation.io_progress);
        if (outcome != VmAccessOutcome::Complete)
          return map_outcome(outcome);
        operation.io_progress = 0;
      }
      operation.phase = 2;
    }
    if (operation.phase == 2) {
      operation.retire_on_fault = true;
      operation.phase = 3;
    }
    if (operation.phase == 3)
      cache_before_write(operation);
    while (operation.phase == 3 && operation.completed < operation.count) {
      if (operation.chunk_size == 0)
        operation.chunk_size = static_cast<std::size_t>(
            std::min<uint64_t>(kTransferBytes, operation.count - operation.completed));
      if (operation.destination_index == 0) {
        const VmAccessOutcome read = access_->read(
            operation.source + operation.completed,
            std::span(operation.scratch).first(operation.chunk_size), operation.io_progress);
        if (read != VmAccessOutcome::Complete)
          return map_outcome(read);
        operation.io_progress = 0;
        operation.destination_index = 1;
      }
      while (operation.destination_index <= operation.destination_count) {
        const uint64_t destination =
            operation.destinations[operation.destination_index - 1] + operation.completed;
        const VmAccessOutcome write = access_->write(
            destination, std::span<const std::byte>(operation.scratch).first(operation.chunk_size),
            operation.io_progress);
        if (operation.io_progress != 0)
          state_->operation_committed = true;
        if (write != VmAccessOutcome::Complete)
          return map_outcome(write);
        state_->operation_committed = true;
        operation.io_progress = 0;
        ++operation.destination_index;
      }
      operation.completed += operation.chunk_size;
      operation.chunk_size = 0;
      operation.destination_index = 0;
    }
    operation.phase = 4;
    if (operation.signal_decrement) {
      cache_before_write(operation);
      const SdmaPacketExecutionOutcome outcome = atomic_fetch_add(
          operation, operation.signal_address, uint64_t{0} - operation.signal_data);
      if (outcome != SdmaPacketExecutionOutcome::Complete)
        return outcome;
    }
    return SdmaPacketExecutionOutcome::Complete;
  }

  SdmaPacketExecutionOutcome execute_fill(Operation &operation) {
    cache_before_write(operation);
    while (operation.completed < operation.count) {
      if (operation.chunk_size == 0) {
        operation.chunk_size = static_cast<std::size_t>(
            std::min<uint64_t>(kTransferBytes, operation.count - operation.completed));
        const std::array<std::byte, sizeof(uint32_t)> pattern =
            std::bit_cast<std::array<std::byte, sizeof(uint32_t)>>(operation.value32);
        for (std::size_t index = 0; index < operation.chunk_size; ++index) {
          operation.scratch[index] =
              dialect_ == SdmaPacketDialect::Gfx1250 || operation.fill_size == 2
                  ? pattern[(operation.completed + index) % pattern.size()]
                  : pattern[0];
        }
      }
      const VmAccessOutcome outcome =
          access_->write(operation.address + operation.completed,
                         std::span<const std::byte>(operation.scratch).first(operation.chunk_size),
                         operation.io_progress);
      if (operation.io_progress != 0)
        state_->operation_committed = true;
      if (outcome != VmAccessOutcome::Complete)
        return map_outcome(outcome);
      state_->operation_committed = true;
      operation.completed += operation.chunk_size;
      operation.chunk_size = 0;
      operation.io_progress = 0;
    }
    return SdmaPacketExecutionOutcome::Complete;
  }

  SdmaPacketExecutionOutcome execute_write(Operation &operation) {
    cache_before_write(operation);
    const VmAccessOutcome outcome =
        access_->write(operation.address, operation.payload, operation.io_progress);
    if (operation.io_progress != 0)
      state_->operation_committed = true;
    if (outcome != VmAccessOutcome::Complete)
      return map_outcome(outcome);
    state_->operation_committed = true;
    return SdmaPacketExecutionOutcome::Complete;
  }

  SdmaPacketExecutionOutcome execute_atomic(Operation &operation) {
    // Preserve the established CP behavior for the reserved low-address range.
    // In particular, do not derive signal metadata below address zero.
    if (operation.address <= 0x1000)
      return SdmaPacketExecutionOutcome::Complete;
    const uint64_t signal_base = operation.address - sizeof(uint64_t);
    if (operation.completes_signal && operation.phase == 0) {
      VmAccessOutcome outcome =
          access_->read(signal_base + 16, operation.mailbox_bytes, operation.mailbox_read_progress);
      if (outcome != VmAccessOutcome::Complete)
        return map_outcome(outcome);
      std::memcpy(&operation.mailbox, operation.mailbox_bytes.data(), sizeof(operation.mailbox));
      operation.phase = 1;
    }
    if (operation.completes_signal && operation.phase == 1) {
      VmAccessOutcome outcome =
          access_->read(signal_base + 24, operation.event_bytes, operation.event_read_progress);
      if (outcome != VmAccessOutcome::Complete)
        return map_outcome(outcome);
      std::memcpy(&operation.event_id, operation.event_bytes.data(), sizeof(operation.event_id));
      operation.phase = 2;
    }
    if (!operation.completes_signal)
      operation.phase = 2;
    if (operation.phase == 2) {
      cache_before_write(operation);
      const SdmaPacketExecutionOutcome outcome =
          atomic_fetch_add(operation, operation.address, operation.value);
      if (outcome != SdmaPacketExecutionOutcome::Complete)
        return outcome;
      operation.phase = 3;
    }
    if (operation.completes_signal && operation.phase == 3 && operation.mailbox != 0) {
      cache_before_write(operation);
      const VmAccessOutcome outcome =
          access_->atomic_store(operation.mailbox, sizeof(uint64_t), operation.event_id);
      if (outcome != VmAccessOutcome::Complete)
        return map_outcome(outcome);
      operation.phase = 4;
    } else if (operation.phase == 3) {
      operation.phase = 4;
    }
    if (operation.completes_signal && operation.phase == 4 && operation.event_id != 0) {
      const VmAccessOutcome outcome = callbacks_.deliver_interrupt(operation.event_id);
      if (outcome != VmAccessOutcome::Complete)
        return map_outcome(outcome);
    }
    state_->completion_published = true;
    return SdmaPacketExecutionOutcome::Complete;
  }

  SdmaPacketExecutionOutcome execute(Operation &operation, Frame &frame) {
    switch (operation.kind) {
    case Kind::Nop:
    case Kind::HdpFlush:
      return SdmaPacketExecutionOutcome::Complete;
    case Kind::Copy:
      return execute_copy(operation);
    case Kind::Fence: {
      cache_before_write(operation);
      std::memcpy(operation.scratch.data(), &operation.value,
                  static_cast<std::size_t>(operation.count));
      const VmAccessOutcome outcome = access_->write(
          operation.address, std::span<const std::byte>(operation.scratch).first(operation.count),
          operation.io_progress);
      if (operation.io_progress != 0)
        state_->operation_committed = true;
      return map_outcome(outcome);
    }
    case Kind::Trap: {
      if (!callbacks_.deliver_interrupt)
        return SdmaPacketExecutionOutcome::Faulted;
      const VmAccessOutcome outcome = callbacks_.deliver_interrupt(operation.value32);
      if (outcome == VmAccessOutcome::Complete) {
        state_->operation_committed = true;
        state_->completion_published = true;
      }
      return map_outcome(outcome);
    }
    case Kind::Poll: {
      uint64_t value = 0;
      if (operation.value32 != 0) {
        const AtomicLoadResult loaded = access_->atomic_load(operation.address, operation.count);
        if (loaded.outcome != VmAccessOutcome::Complete)
          return map_outcome(loaded.outcome);
        value = loaded.value;
      } else {
        if (!callbacks_.read_register)
          return SdmaPacketExecutionOutcome::Complete;
        const SdmaPacketRegisterReadResult loaded =
            callbacks_.read_register(static_cast<uint32_t>(operation.address));
        if (loaded.outcome != VmAccessOutcome::Complete)
          return map_outcome(loaded.outcome);
        value = loaded.value;
      }
      return compare(operation.function, value & operation.mask,
                     operation.reference & operation.mask)
                 ? SdmaPacketExecutionOutcome::Complete
                 : SdmaPacketExecutionOutcome::Unavailable;
    }
    case Kind::Atomic:
      return execute_atomic(operation);
    case Kind::Fill:
      return execute_fill(operation);
    case Kind::Timestamp: {
      if (operation.address <= 0x1000)
        return SdmaPacketExecutionOutcome::Complete;
      cache_before_write(operation);
      if (!operation.timestamp_captured) {
        operation.value = callbacks_.timestamp ? callbacks_.timestamp() : 0;
        operation.timestamp_captured = true;
      }
      std::memcpy(operation.scratch.data(), &operation.value, sizeof(operation.value));
      const VmAccessOutcome outcome = access_->write(
          operation.address, std::span<const std::byte>(operation.scratch).first(sizeof(uint64_t)),
          operation.io_progress);
      if (operation.io_progress != 0)
        state_->operation_committed = true;
      return map_outcome(outcome);
    }
    case Kind::Gcr: {
      const bool gfx1250 = dialect_ == SdmaPacketDialect::Gfx1250;
      const uint32_t wb = gfx1250 ? (1u << 15) : (1u << 31);
      const uint32_t inv = gfx1250 ? ((1u << 14) | (1u << 13)) : ((1u << 30) | (1u << 29));
      if ((operation.value32 & wb) != 0) {
        if (callbacks_.acquire_cache_maintenance) {
          [[maybe_unused]] SdmaCacheLease lease =
              callbacks_.acquire_cache_maintenance(SdmaCacheOperation::WritebackInvalidate);
          state_->operation_committed = true;
        }
      } else if ((operation.value32 & inv) != 0) {
        if (callbacks_.acquire_cache_maintenance) {
          [[maybe_unused]] SdmaCacheLease lease =
              callbacks_.acquire_cache_maintenance(SdmaCacheOperation::Invalidate);
          state_->operation_committed = true;
        }
      }
      return SdmaPacketExecutionOutcome::Complete;
    }
    case Kind::Write:
      return execute_write(operation);
    case Kind::Indirect: {
      if (state_->frames.size() > kMaxIndirectDepth)
        return SdmaPacketExecutionOutcome::Malformed;
      const VmAccessOutcome outcome =
          access_->read(operation.address, operation.indirect, operation.io_progress);
      if (outcome != VmAccessOutcome::Complete)
        return map_outcome(outcome);
      std::vector<uint32_t> words(operation.indirect.size() / sizeof(uint32_t));
      if (!operation.indirect.empty())
        std::memcpy(words.data(), operation.indirect.data(), operation.indirect.size());
      frame.at += operation.dwords;
      state_->operation.reset();
      state_->frames.push_back({.words = std::move(words), .at = 0, .root = false});
      return SdmaPacketExecutionOutcome::Complete;
    }
    case Kind::Conditional: {
      const VmAccessOutcome outcome =
          access_->read(operation.address, std::span(operation.scratch).first(sizeof(uint32_t)),
                        operation.io_progress);
      if (outcome != VmAccessOutcome::Complete)
        return map_outcome(outcome);
      uint32_t value = 0;
      std::memcpy(&value, operation.scratch.data(), sizeof(value));
      if (value != operation.reference) {
        const std::size_t next = frame.at + operation.dwords;
        if (frame.root) {
          if (operation.skip_dwords > std::numeric_limits<std::size_t>::max() - operation.dwords)
            return SdmaPacketExecutionOutcome::Malformed;
          state_->root_dwords = operation.dwords + operation.skip_dwords;
          // The ring owns the skipped commands and validates the reported
          // retirement extent. Its continuation retains only the COND_EXE packet.
          operation.skip_dwords = 0;
        } else if (next > frame.words.size() || operation.skip_dwords > frame.words.size() - next) {
          return SdmaPacketExecutionOutcome::Malformed;
        }
      } else {
        operation.skip_dwords = 0;
      }
      return SdmaPacketExecutionOutcome::Complete;
    }
    case Kind::RegisterWrite: {
      if (!callbacks_.write_register)
        return SdmaPacketExecutionOutcome::Faulted;
      const VmAccessOutcome outcome =
          callbacks_.write_register(static_cast<uint32_t>(operation.address), operation.value32);
      if (outcome == VmAccessOutcome::Complete)
        state_->operation_committed = true;
      return map_outcome(outcome);
    }
    }
    return SdmaPacketExecutionOutcome::Malformed;
  }

  SdmaPacketExecutionResult run() {
    while (!state_->frames.empty()) {
      Frame &frame = state_->frames.back();
      if (frame.at == frame.words.size()) {
        const bool root = frame.root;
        state_->frames.pop_back();
        if (root)
          return finish(SdmaPacketExecutionOutcome::Complete);
        continue;
      }
      if (frame.at > frame.words.size())
        return finish(SdmaPacketExecutionOutcome::Malformed);
      if (!state_->operation) {
        const SdmaPacketExecutionOutcome decoded = decode(frame);
        if (decoded != SdmaPacketExecutionOutcome::Complete)
          return finish(decoded);
      }
      Operation *operation = &*state_->operation;
      const std::size_t dwords = operation->dwords;
      const Kind kind = operation->kind;
      const SdmaPacketExecutionOutcome outcome = execute(*operation, frame);
      if (outcome == SdmaPacketExecutionOutcome::Unavailable) {
        operation->cache_lease.reset();
        return result(outcome);
      }
      if (outcome != SdmaPacketExecutionOutcome::Complete)
        return finish(outcome);
      if (kind == Kind::Indirect)
        continue;
      frame.at += dwords + operation->skip_dwords;
      state_->operation.reset();
    }
    return finish(SdmaPacketExecutionOutcome::Complete);
  }

  SdmaPacketDialect dialect_;
  SdmaPacketCallbacks callbacks_;
  State *state_ = nullptr;
  const GpuVmAccess *access_ = nullptr;
};

SdmaPacketContinuation::SdmaPacketContinuation() : impl_(std::make_unique<Impl>()) {}

SdmaPacketContinuation::~SdmaPacketContinuation() = default;

SdmaPacketContinuation::SdmaPacketContinuation(SdmaPacketContinuation &&other) noexcept = default;

SdmaPacketContinuation &
SdmaPacketContinuation::operator=(SdmaPacketContinuation &&other) noexcept = default;

bool SdmaPacketContinuation::pending() const noexcept { return impl_ != nullptr && impl_->pending; }

SdmaPacketProcessor::SdmaPacketProcessor(SdmaPacketDialect dialect, SdmaPacketCallbacks callbacks)
    : impl_(std::make_unique<Impl>(dialect, std::move(callbacks))) {}

SdmaPacketProcessor::~SdmaPacketProcessor() = default;

SdmaPacketProcessor::SdmaPacketProcessor(SdmaPacketProcessor &&other) noexcept = default;

SdmaPacketProcessor &SdmaPacketProcessor::operator=(SdmaPacketProcessor &&other) noexcept = default;

namespace {

SdmaPacketProcessResult public_result(const SdmaPacketExecutionResult &result) {
  return {.packet = {.status = result.outcome == SdmaPacketExecutionOutcome::Complete
                                   ? PacketProcessStatus::Complete
                               : result.outcome == SdmaPacketExecutionOutcome::NeedInput
                                   ? PacketProcessStatus::NeedInput
                               : result.outcome == SdmaPacketExecutionOutcome::Unavailable
                                   ? PacketProcessStatus::Blocked
                               : result.outcome == SdmaPacketExecutionOutcome::Faulted
                                   ? PacketProcessStatus::Faulted
                                   : PacketProcessStatus::Malformed,
                     .retirement =
                         result.retire_packet ? PacketRetirement::Retire : PacketRetirement::Hold,
                     .retirement_bytes = result.packet_dwords * sizeof(uint32_t),
                     .required_bytes = result.required_dwords * sizeof(uint32_t)},
          .operation_committed = result.operation_committed,
          .completion_published = result.completion_published};
}

} // namespace

SdmaPacketProcessResult SdmaPacketProcessor::process(Request request) {
  const SdmaPacketExecutionResult result =
      impl_ != nullptr && request.continuation.impl_ != nullptr
          ? impl_->process(request.available_dwords, request.access, *request.continuation.impl_)
          : SdmaPacketExecutionResult{};
  return public_result(result);
}

void SdmaPacketProcessor::reset(SdmaPacketContinuation &continuation) noexcept {
  if (impl_ != nullptr && continuation.impl_ != nullptr)
    impl_->clear(*continuation.impl_);
}

} // namespace rocjitsu::amdgpu
