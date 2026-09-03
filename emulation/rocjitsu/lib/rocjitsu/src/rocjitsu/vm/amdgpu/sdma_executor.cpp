// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/sdma_executor.h"

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

SdmaExecutionOutcome map_outcome(VmAccessOutcome outcome) {
  switch (outcome) {
  case VmAccessOutcome::Complete:
    return SdmaExecutionOutcome::Complete;
  case VmAccessOutcome::Unavailable:
    return SdmaExecutionOutcome::Unavailable;
  case VmAccessOutcome::Faulted:
    return SdmaExecutionOutcome::Faulted;
  case VmAccessOutcome::Malformed:
    return SdmaExecutionOutcome::Malformed;
  }
  return SdmaExecutionOutcome::Malformed;
}

} // namespace

class SdmaExecutor::Impl {
public:
  explicit Impl(SdmaPacketDialect dialect, SdmaExecutorCallbacks callbacks)
      : dialect_(dialect), callbacks_(std::move(callbacks)) {}

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

  struct Frame {
    std::vector<uint32_t> words;
    std::size_t at = 0;
    bool root = false;
  };

  struct Operation {
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
    bool cache_prepared = false;
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

  SdmaExecutionResult start(std::span<const uint32_t> words, GpuVmAccess access) {
    if (pending_)
      return result(SdmaExecutionOutcome::Malformed);
    clear();
    access_.emplace(std::move(access));
    frames_.push_back(
        {.words = std::vector<uint32_t>(words.begin(), words.end()), .at = 0, .root = true});
    pending_ = true;
    return run();
  }

  SdmaExecutionResult resume() {
    if (!pending_)
      return result(SdmaExecutionOutcome::Malformed);
    return run();
  }

  bool pending() const { return pending_; }

  void clear() {
    access_.reset();
    frames_.clear();
    operation_.reset();
    root_dwords_ = 0;
    operation_committed_ = false;
    completion_published_ = false;
    retire_packet_ = false;
    pending_ = false;
  }

private:
  SdmaExecutionResult result(SdmaExecutionOutcome outcome) const {
    return {.outcome = outcome,
            .packet_dwords = root_dwords_,
            .operation_committed = operation_committed_,
            .completion_published = completion_published_,
            .retire_packet = retire_packet_};
  }

  SdmaExecutionResult finish(SdmaExecutionOutcome outcome) {
    if (outcome == SdmaExecutionOutcome::Complete) {
      completion_published_ = true;
      retire_packet_ = true;
    } else if ((outcome == SdmaExecutionOutcome::Faulted ||
                outcome == SdmaExecutionOutcome::Malformed) &&
               (operation_committed_ || (operation_ && operation_->retire_on_fault))) {
      // Never expose an already-committed prefix for replay. Copy data faults
      // also preserve the established CP policy of retiring the failed copy.
      retire_packet_ = true;
    }
    const SdmaExecutionResult final = result(outcome);
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

  SdmaExecutionOutcome decode(Frame &frame) {
    if (!has(frame, 1))
      return SdmaExecutionOutcome::Malformed;

    Operation op;
    const uint32_t header = word(frame, 0);
    const uint8_t opcode = header & 0xff;
    const uint8_t subopcode = (header >> 8) & 0xff;

    switch (opcode) {
    case kOpNop:
      op.kind = Kind::Nop;
      op.dwords = ((header >> 16) & 0x3fff) + 1;
      break;
    case kOpCopy: {
      op.kind = Kind::Copy;
      constexpr uint8_t kSubopLinearBroadcast = 16;
      if (subopcode != kSubopLinear && subopcode != kSubopLinearBroadcast)
        return SdmaExecutionOutcome::Malformed;
      if (gfx11_plus() && (header & ((1u << 30) | (1u << 31))) != 0) {
        op.wait_enabled = (header & (1u << 30)) != 0;
        op.signal_enabled = (header & (1u << 31)) != 0;
        const std::size_t copy_base = 1 + (op.wait_enabled ? kWaitDwords : 0);
        const std::size_t signal_base = copy_base + kCopyBodyDwords;
        op.dwords = signal_base + (op.signal_enabled ? kSignalDwords : 0);
        if (!has(frame, op.dwords))
          return SdmaExecutionOutcome::Malformed;
        if (op.wait_enabled) {
          op.function = word(frame, 1) & 0x7;
          op.address = join(word(frame, 2) & ~0x7u, word(frame, 3));
          op.reference = join(word(frame, 4), word(frame, 5));
          op.mask = join(word(frame, 6), word(frame, 7));
        }
        op.count = (word(frame, copy_base) & 0x3fffffff) + uint64_t{1};
        op.source = join(word(frame, copy_base + 2), word(frame, copy_base + 3));
        op.destinations[0] = join(word(frame, copy_base + 4), word(frame, copy_base + 5));
        op.destination_count = 1;
        if (!valid_range(op.source, op.count) || !valid_range(op.destinations[0], op.count))
          return SdmaExecutionOutcome::Malformed;
        if (op.signal_enabled) {
          const uint32_t signal_op = word(frame, signal_base) & 0x7f;
          op.signal_address =
              join(word(frame, signal_base + 1) & ~0x7u, word(frame, signal_base + 2));
          op.signal_data = join(word(frame, signal_base + 3), word(frame, signal_base + 4));
          op.signal_decrement = op.signal_address > 0x1000 && signal_op == 0x70;
        }
        break;
      }
      const bool broadcast =
          subopcode == kSubopLinearBroadcast ||
          (gfx11_plus() ? (header & (1u << 27)) != 0 : (header & (1u << 28)) != 0);
      op.dwords = broadcast ? kCopyBroadcastDwords : kCopyLinearDwords;
      if (!has(frame, op.dwords))
        return SdmaExecutionOutcome::Malformed;
      const uint32_t count_mask =
          broadcast || dialect_ == SdmaPacketDialect::Legacy ? 0x003fffff : 0x3fffffff;
      op.count = (word(frame, 1) & count_mask) + uint64_t{1};
      op.source = join(word(frame, 3), word(frame, 4));
      op.destinations[0] = join(word(frame, 5), word(frame, 6));
      op.destination_count = broadcast ? 2 : 1;
      if (broadcast)
        op.destinations[1] = join(word(frame, 7), word(frame, 8));
      if (!valid_range(op.source, op.count) || !valid_range(op.destinations[0], op.count) ||
          (broadcast && !valid_range(op.destinations[1], op.count)))
        return SdmaExecutionOutcome::Malformed;
      break;
    }
    case kOpFence:
      op.kind = Kind::Fence;
      if (gfx11_plus() && subopcode == kSubopFence64) {
        op.dwords = kFence64Dwords;
        if (!has(frame, op.dwords))
          return SdmaExecutionOutcome::Malformed;
        op.address = join(word(frame, 1) & ~0x7u, word(frame, 2));
        op.value = join(word(frame, 3), word(frame, 4));
        op.count = sizeof(uint64_t);
      } else {
        op.dwords = kFenceDwords;
        if (!has(frame, op.dwords))
          return SdmaExecutionOutcome::Malformed;
        op.address = join(word(frame, 1), word(frame, 2));
        op.value = word(frame, 3);
        op.count = sizeof(uint32_t);
      }
      break;
    case kOpTrap:
      op.kind = Kind::Trap;
      op.dwords = kTrapDwords;
      if (has(frame, op.dwords))
        op.value32 = word(frame, 1) & 0x0fffffff;
      break;
    case kOpPollRegmem:
      op.kind = Kind::Poll;
      if (gfx11_plus() && subopcode == kSubopPollMemory64) {
        op.dwords = kPoll64Dwords;
        if (!has(frame, op.dwords))
          return SdmaExecutionOutcome::Malformed;
        op.function = (header >> 28) & 0x7;
        op.address = join(word(frame, 1) & ~0x7u, word(frame, 2));
        op.reference = join(word(frame, 3), word(frame, 4));
        op.mask = join(word(frame, 5), word(frame, 6));
        op.count = sizeof(uint64_t);
        op.value32 = 1;
      } else {
        op.dwords = kPollDwords;
        if (!has(frame, op.dwords))
          return SdmaExecutionOutcome::Malformed;
        op.function = (header >> 28) & 0x7;
        op.address = join(word(frame, 1), word(frame, 2));
        op.reference = word(frame, 3);
        op.mask = word(frame, 4);
        op.count = sizeof(uint32_t);
        op.value32 = (header >> 31) & 1;
      }
      break;
    case kOpAtomic:
      op.kind = Kind::Atomic;
      op.dwords = kAtomicDwords;
      if (!has(frame, op.dwords))
        return SdmaExecutionOutcome::Malformed;
      if (((header >> 25) & 0x7f) != 47)
        return SdmaExecutionOutcome::Malformed;
      op.address = join(word(frame, 1), word(frame, 2));
      op.value = join(word(frame, 3), word(frame, 4));
      op.completes_signal = static_cast<int64_t>(op.value) < 0 && callbacks_.deliver_interrupt;
      break;
    case kOpConstantFill:
      op.kind = Kind::Fill;
      op.dwords = kFillDwords;
      if (!has(frame, op.dwords))
        return SdmaExecutionOutcome::Malformed;
      op.address = join(word(frame, 1), word(frame, 2));
      op.value32 = word(frame, 3);
      op.count = (word(frame, 4) & 0x3fffffff) + uint64_t{1};
      op.fill_size = (header >> 30) & 0x3;
      break;
    case kOpTimestamp:
      op.kind = Kind::Timestamp;
      op.dwords = kTimestampDwords;
      if (has(frame, op.dwords))
        op.address = join(word(frame, 1), word(frame, 2));
      break;
    case kOpGcr:
      op.kind = Kind::Gcr;
      op.dwords = dialect_ == SdmaPacketDialect::Gfx1250 ? kGfx1250GcrDwords : kLegacyGcrDwords;
      if (has(frame, op.dwords))
        op.value32 = word(frame, dialect_ == SdmaPacketDialect::Gfx1250 ? 3 : 2);
      break;
    case kOpHdpFlush:
      op.kind = Kind::HdpFlush;
      op.dwords = 1;
      break;
    case kOpWrite: {
      op.kind = Kind::Write;
      if (subopcode != kSubopLinear || !has(frame, 4))
        return SdmaExecutionOutcome::Malformed;
      const std::size_t count = (word(frame, 3) & 0x000fffff) + std::size_t{1};
      op.dwords = 4 + count;
      if (!has(frame, op.dwords) || count > std::numeric_limits<std::size_t>::max() / 4)
        return SdmaExecutionOutcome::Malformed;
      op.address = join(word(frame, 1), word(frame, 2));
      op.count = count * sizeof(uint32_t);
      if (!valid_range(op.address, op.count))
        return SdmaExecutionOutcome::Malformed;
      op.payload.resize(static_cast<std::size_t>(op.count));
      std::memcpy(op.payload.data(), &frame.words[frame.at + 4], op.payload.size());
      break;
    }
    case kOpIndirect: {
      op.kind = Kind::Indirect;
      op.dwords = kIndirectDwords;
      if (!has(frame, op.dwords))
        return SdmaExecutionOutcome::Malformed;
      const std::size_t count = word(frame, 3) & 0x000fffff;
      if (count > std::numeric_limits<std::size_t>::max() / sizeof(uint32_t))
        return SdmaExecutionOutcome::Malformed;
      op.address = join(word(frame, 1), word(frame, 2));
      op.indirect.resize(count * sizeof(uint32_t));
      if (!op.indirect.empty() && !valid_range(op.address, op.indirect.size()))
        return SdmaExecutionOutcome::Malformed;
      break;
    }
    case kOpConditionalExecute:
      op.kind = Kind::Conditional;
      op.dwords = kConditionalDwords;
      if (!has(frame, op.dwords))
        return SdmaExecutionOutcome::Malformed;
      op.address = join(word(frame, 1), word(frame, 2));
      op.reference = word(frame, 3);
      op.skip_dwords = word(frame, 4) & 0x3fff;
      break;
    case kOpRegisterWrite:
      op.kind = Kind::RegisterWrite;
      op.dwords = kRegisterWriteDwords;
      if (has(frame, op.dwords)) {
        op.address = word(frame, 1);
        op.value32 = word(frame, 2);
      }
      break;
    default:
      return SdmaExecutionOutcome::Malformed;
    }

    if (!has(frame, op.dwords))
      return SdmaExecutionOutcome::Malformed;
    if (frame.root) {
      root_dwords_ = op.dwords;
      // COND_EXE controls the dwords following its five-dword header. Keep the
      // available root stream until its predicate determines whether those
      // dwords execute as later packets or retire as one skipped region.
      if (op.kind != Kind::Conditional)
        frame.words.resize(frame.at + op.dwords);
    }
    operation_.emplace(std::move(op));
    return SdmaExecutionOutcome::Complete;
  }

  SdmaExecutionOutcome cache_before_write(Operation &op) {
    if (!op.cache_prepared) {
      if (callbacks_.maintain_caches)
        callbacks_.maintain_caches(SdmaCacheOperation::WritebackInvalidate);
      op.cache_prepared = true;
    }
    return SdmaExecutionOutcome::Complete;
  }

  SdmaExecutionOutcome atomic_fetch_add(Operation &op, uint64_t address, uint64_t amount) {
    if (!op.cas_expected_valid) {
      const AtomicLoadResult loaded = access_->atomic_load(address, sizeof(uint64_t));
      if (loaded.outcome != VmAccessOutcome::Complete)
        return map_outcome(loaded.outcome);
      op.cas_expected = loaded.value;
      op.cas_expected_valid = true;
    }
    for (;;) {
      const AtomicCompareExchangeResult exchanged = access_->compare_exchange(
          address, sizeof(uint64_t), op.cas_expected, op.cas_expected + amount);
      if (exchanged.outcome != VmAccessOutcome::Complete)
        return map_outcome(exchanged.outcome);
      if (exchanged.exchanged) {
        op.cas_expected_valid = false;
        operation_committed_ = true;
        return SdmaExecutionOutcome::Complete;
      }
      op.cas_expected = exchanged.observed;
    }
  }

  SdmaExecutionOutcome execute_copy(Operation &op) {
    if (op.phase == 0) {
      if (op.wait_enabled && op.address > 0x1000) {
        const VmAccessOutcome outcome = access_->read(
            op.address, std::span(op.scratch).first(sizeof(uint64_t)), op.io_progress);
        if (outcome != VmAccessOutcome::Complete)
          return map_outcome(outcome);
        uint64_t value = 0;
        std::memcpy(&value, op.scratch.data(), sizeof(value));
        op.io_progress = 0;
        if (!compare(op.function, value & op.mask, op.reference))
          return SdmaExecutionOutcome::Unavailable;
      }
      op.phase = 1;
    }
    if (op.phase == 1) {
      if (op.signal_decrement) {
        const VmAccessOutcome outcome = access_->read(
            op.signal_address, std::span(op.scratch).first(sizeof(uint64_t)), op.io_progress);
        if (outcome != VmAccessOutcome::Complete)
          return map_outcome(outcome);
        op.io_progress = 0;
      }
      op.phase = 2;
    }
    if (op.phase == 2) {
      cache_before_write(op);
      op.retire_on_fault = true;
      op.phase = 3;
    }
    while (op.phase == 3 && op.completed < op.count) {
      if (op.chunk_size == 0)
        op.chunk_size =
            static_cast<std::size_t>(std::min<uint64_t>(kTransferBytes, op.count - op.completed));
      if (op.destination_index == 0) {
        const VmAccessOutcome read = access_->read(
            op.source + op.completed, std::span(op.scratch).first(op.chunk_size), op.io_progress);
        if (read != VmAccessOutcome::Complete)
          return map_outcome(read);
        op.io_progress = 0;
        op.destination_index = 1;
      }
      while (op.destination_index <= op.destination_count) {
        const uint64_t destination = op.destinations[op.destination_index - 1] + op.completed;
        const VmAccessOutcome write =
            access_->write(destination, std::span<const std::byte>(op.scratch).first(op.chunk_size),
                           op.io_progress);
        if (op.io_progress != 0)
          operation_committed_ = true;
        if (write != VmAccessOutcome::Complete)
          return map_outcome(write);
        operation_committed_ = true;
        op.io_progress = 0;
        ++op.destination_index;
      }
      op.completed += op.chunk_size;
      op.chunk_size = 0;
      op.destination_index = 0;
    }
    op.phase = 4;
    if (op.signal_decrement) {
      const SdmaExecutionOutcome outcome =
          atomic_fetch_add(op, op.signal_address, uint64_t{0} - op.signal_data);
      if (outcome != SdmaExecutionOutcome::Complete)
        return outcome;
    }
    return SdmaExecutionOutcome::Complete;
  }

  SdmaExecutionOutcome execute_fill(Operation &op) {
    cache_before_write(op);
    while (op.completed < op.count) {
      if (op.chunk_size == 0) {
        op.chunk_size =
            static_cast<std::size_t>(std::min<uint64_t>(kTransferBytes, op.count - op.completed));
        const auto pattern = std::bit_cast<std::array<std::byte, sizeof(uint32_t)>>(op.value32);
        for (std::size_t index = 0; index < op.chunk_size; ++index) {
          op.scratch[index] = dialect_ == SdmaPacketDialect::Gfx1250 || op.fill_size == 2
                                  ? pattern[(op.completed + index) % pattern.size()]
                                  : pattern[0];
        }
      }
      const VmAccessOutcome outcome = access_->write(
          op.address + op.completed, std::span<const std::byte>(op.scratch).first(op.chunk_size),
          op.io_progress);
      if (op.io_progress != 0)
        operation_committed_ = true;
      if (outcome != VmAccessOutcome::Complete)
        return map_outcome(outcome);
      operation_committed_ = true;
      op.completed += op.chunk_size;
      op.chunk_size = 0;
      op.io_progress = 0;
    }
    return SdmaExecutionOutcome::Complete;
  }

  SdmaExecutionOutcome execute_write(Operation &op) {
    cache_before_write(op);
    const VmAccessOutcome outcome = access_->write(op.address, op.payload, op.io_progress);
    if (op.io_progress != 0)
      operation_committed_ = true;
    if (outcome != VmAccessOutcome::Complete)
      return map_outcome(outcome);
    operation_committed_ = true;
    return SdmaExecutionOutcome::Complete;
  }

  SdmaExecutionOutcome execute_atomic(Operation &op) {
    // Preserve the established CP behavior for the reserved low-address range.
    // In particular, do not derive signal metadata below address zero.
    if (op.address <= 0x1000)
      return SdmaExecutionOutcome::Complete;
    const uint64_t signal_base = op.address - sizeof(uint64_t);
    if (op.completes_signal && op.phase == 0) {
      VmAccessOutcome outcome =
          access_->read(signal_base + 16, op.mailbox_bytes, op.mailbox_read_progress);
      if (outcome != VmAccessOutcome::Complete)
        return map_outcome(outcome);
      std::memcpy(&op.mailbox, op.mailbox_bytes.data(), sizeof(op.mailbox));
      op.phase = 1;
    }
    if (op.completes_signal && op.phase == 1) {
      VmAccessOutcome outcome =
          access_->read(signal_base + 24, op.event_bytes, op.event_read_progress);
      if (outcome != VmAccessOutcome::Complete)
        return map_outcome(outcome);
      std::memcpy(&op.event_id, op.event_bytes.data(), sizeof(op.event_id));
      op.phase = 2;
    }
    if (!op.completes_signal)
      op.phase = 2;
    if (op.phase == 2) {
      cache_before_write(op);
      const SdmaExecutionOutcome outcome = atomic_fetch_add(op, op.address, op.value);
      if (outcome != SdmaExecutionOutcome::Complete)
        return outcome;
      op.phase = 3;
    }
    if (op.completes_signal && op.phase == 3 && op.mailbox != 0) {
      const VmAccessOutcome outcome =
          access_->atomic_store(op.mailbox, sizeof(uint64_t), op.event_id);
      if (outcome != VmAccessOutcome::Complete)
        return map_outcome(outcome);
      op.phase = 4;
    } else if (op.phase == 3) {
      op.phase = 4;
    }
    if (op.completes_signal && op.phase == 4 && op.event_id != 0) {
      const VmAccessOutcome outcome = callbacks_.deliver_interrupt(op.event_id);
      if (outcome != VmAccessOutcome::Complete)
        return map_outcome(outcome);
    }
    completion_published_ = true;
    return SdmaExecutionOutcome::Complete;
  }

  SdmaExecutionOutcome execute(Operation &op, Frame &frame) {
    switch (op.kind) {
    case Kind::Nop:
    case Kind::HdpFlush:
      return SdmaExecutionOutcome::Complete;
    case Kind::Copy:
      return execute_copy(op);
    case Kind::Fence: {
      cache_before_write(op);
      std::memcpy(op.scratch.data(), &op.value, static_cast<std::size_t>(op.count));
      const VmAccessOutcome outcome = access_->write(
          op.address, std::span<const std::byte>(op.scratch).first(op.count), op.io_progress);
      if (op.io_progress != 0)
        operation_committed_ = true;
      return map_outcome(outcome);
    }
    case Kind::Trap: {
      if (!callbacks_.deliver_interrupt)
        return SdmaExecutionOutcome::Faulted;
      const VmAccessOutcome outcome = callbacks_.deliver_interrupt(op.value32);
      if (outcome == VmAccessOutcome::Complete) {
        operation_committed_ = true;
        completion_published_ = true;
      }
      return map_outcome(outcome);
    }
    case Kind::Poll: {
      uint64_t value = 0;
      if (op.value32 != 0) {
        const AtomicLoadResult loaded = access_->atomic_load(op.address, op.count);
        if (loaded.outcome != VmAccessOutcome::Complete)
          return map_outcome(loaded.outcome);
        value = loaded.value;
      } else {
        if (!callbacks_.poll_register)
          return SdmaExecutionOutcome::Faulted;
        return map_outcome(callbacks_.poll_register(static_cast<uint32_t>(op.address),
                                                    static_cast<uint32_t>(op.reference),
                                                    static_cast<uint32_t>(op.mask), op.function));
      }
      return compare(op.function, value & op.mask, op.reference & op.mask)
                 ? SdmaExecutionOutcome::Complete
                 : SdmaExecutionOutcome::Unavailable;
    }
    case Kind::Atomic:
      return execute_atomic(op);
    case Kind::Fill:
      return execute_fill(op);
    case Kind::Timestamp: {
      if (op.address <= 0x1000)
        return SdmaExecutionOutcome::Complete;
      cache_before_write(op);
      if (!op.timestamp_captured) {
        op.value = callbacks_.timestamp ? callbacks_.timestamp() : 0;
        op.timestamp_captured = true;
      }
      std::memcpy(op.scratch.data(), &op.value, sizeof(op.value));
      const VmAccessOutcome outcome =
          access_->write(op.address, std::span<const std::byte>(op.scratch).first(sizeof(uint64_t)),
                         op.io_progress);
      if (op.io_progress != 0)
        operation_committed_ = true;
      return map_outcome(outcome);
    }
    case Kind::Gcr: {
      const bool gfx1250 = dialect_ == SdmaPacketDialect::Gfx1250;
      const uint32_t wb = gfx1250 ? (1u << 15) : (1u << 31);
      const uint32_t inv = gfx1250 ? ((1u << 14) | (1u << 13)) : ((1u << 30) | (1u << 29));
      if ((op.value32 & wb) != 0) {
        if (callbacks_.maintain_caches) {
          callbacks_.maintain_caches(SdmaCacheOperation::WritebackInvalidate);
          operation_committed_ = true;
        }
      } else if ((op.value32 & inv) != 0) {
        if (callbacks_.maintain_caches) {
          callbacks_.maintain_caches(SdmaCacheOperation::Invalidate);
          operation_committed_ = true;
        }
      }
      return SdmaExecutionOutcome::Complete;
    }
    case Kind::Write:
      return execute_write(op);
    case Kind::Indirect: {
      if (frames_.size() > kMaxIndirectDepth)
        return SdmaExecutionOutcome::Malformed;
      const VmAccessOutcome outcome = access_->read(op.address, op.indirect, op.io_progress);
      if (outcome != VmAccessOutcome::Complete)
        return map_outcome(outcome);
      std::vector<uint32_t> words(op.indirect.size() / sizeof(uint32_t));
      if (!op.indirect.empty())
        std::memcpy(words.data(), op.indirect.data(), op.indirect.size());
      frame.at += op.dwords;
      operation_.reset();
      frames_.push_back({.words = std::move(words), .at = 0, .root = false});
      return SdmaExecutionOutcome::Complete;
    }
    case Kind::Conditional: {
      const VmAccessOutcome outcome =
          access_->read(op.address, std::span(op.scratch).first(sizeof(uint32_t)), op.io_progress);
      if (outcome != VmAccessOutcome::Complete)
        return map_outcome(outcome);
      uint32_t value = 0;
      std::memcpy(&value, op.scratch.data(), sizeof(value));
      if (value != op.reference) {
        const std::size_t next = frame.at + op.dwords;
        if (next > frame.words.size() || op.skip_dwords > frame.words.size() - next)
          return SdmaExecutionOutcome::Malformed;
      } else {
        op.skip_dwords = 0;
      }
      if (frame.root) {
        root_dwords_ = op.dwords + op.skip_dwords;
        frame.words.resize(frame.at + root_dwords_);
      }
      return SdmaExecutionOutcome::Complete;
    }
    case Kind::RegisterWrite: {
      if (!callbacks_.write_register)
        return SdmaExecutionOutcome::Faulted;
      const VmAccessOutcome outcome =
          callbacks_.write_register(static_cast<uint32_t>(op.address), op.value32);
      if (outcome == VmAccessOutcome::Complete)
        operation_committed_ = true;
      return map_outcome(outcome);
    }
    }
    return SdmaExecutionOutcome::Malformed;
  }

  SdmaExecutionResult run() {
    while (!frames_.empty()) {
      Frame &frame = frames_.back();
      if (frame.at == frame.words.size()) {
        const bool root = frame.root;
        frames_.pop_back();
        if (root)
          return finish(SdmaExecutionOutcome::Complete);
        continue;
      }
      if (frame.at > frame.words.size())
        return finish(SdmaExecutionOutcome::Malformed);
      if (!operation_) {
        const SdmaExecutionOutcome decoded = decode(frame);
        if (decoded != SdmaExecutionOutcome::Complete)
          return finish(decoded);
      }
      Operation *op = &*operation_;
      const std::size_t dwords = op->dwords;
      const Kind kind = op->kind;
      const SdmaExecutionOutcome outcome = execute(*op, frame);
      if (outcome != SdmaExecutionOutcome::Complete)
        return outcome == SdmaExecutionOutcome::Unavailable ? result(outcome) : finish(outcome);
      if (kind == Kind::Indirect)
        continue;
      frame.at += dwords + op->skip_dwords;
      operation_.reset();
    }
    return finish(SdmaExecutionOutcome::Complete);
  }

  SdmaPacketDialect dialect_;
  SdmaExecutorCallbacks callbacks_;
  std::optional<GpuVmAccess> access_;
  std::vector<Frame> frames_;
  std::optional<Operation> operation_;
  std::size_t root_dwords_ = 0;
  bool operation_committed_ = false;
  bool completion_published_ = false;
  bool retire_packet_ = false;
  bool pending_ = false;
};

SdmaExecutor::SdmaExecutor(SdmaPacketDialect dialect, SdmaExecutorCallbacks callbacks)
    : impl_(std::make_unique<Impl>(dialect, std::move(callbacks))) {}

SdmaExecutor::~SdmaExecutor() = default;

SdmaExecutor::SdmaExecutor(SdmaExecutor &&other) noexcept = default;

SdmaExecutor &SdmaExecutor::operator=(SdmaExecutor &&other) noexcept = default;

SdmaExecutionResult SdmaExecutor::start(std::span<const uint32_t> available_dwords,
                                        GpuVmAccess access) {
  return impl_ != nullptr ? impl_->start(available_dwords, std::move(access))
                          : SdmaExecutionResult{};
}

SdmaExecutionResult SdmaExecutor::resume() {
  return impl_ != nullptr ? impl_->resume() : SdmaExecutionResult{};
}

bool SdmaExecutor::pending() const { return impl_ != nullptr && impl_->pending(); }

void SdmaExecutor::reset() {
  if (impl_ != nullptr)
    impl_->clear();
}

} // namespace rocjitsu::amdgpu
