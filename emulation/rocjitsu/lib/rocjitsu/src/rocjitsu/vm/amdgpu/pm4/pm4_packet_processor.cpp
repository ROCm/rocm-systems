// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pm4/pm4_packet_processor.h"

#include "rocjitsu/vm/amdgpu/atomic_op.h"
#include "rocjitsu/vm/amdgpu/circular_ring_reader.h"
#include "rocjitsu/vm/amdgpu/dispatch_entry.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/hsa_clock.h"
#include "rocjitsu/vm/amdgpu/pm4/pm4_clear_state.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <format>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace rocjitsu::amdgpu {

std::optional<uint64_t> normalize_pm4_producer_cursor(uint64_t producer_cursor,
                                                      uint64_t reference_cursor,
                                                      uint64_t ring_dwords) noexcept {
  if (ring_dwords == 0)
    return std::nullopt;
  if (producer_cursor >= reference_cursor && producer_cursor - reference_cursor <= ring_dwords)
    return producer_cursor;
  if (producer_cursor >= ring_dwords)
    return std::nullopt;

  const uint64_t epoch = reference_cursor - reference_cursor % ring_dwords;
  if (producer_cursor > std::numeric_limits<uint64_t>::max() - epoch)
    return std::nullopt;
  uint64_t normalized = epoch + producer_cursor;
  if (normalized < reference_cursor) {
    if (normalized > std::numeric_limits<uint64_t>::max() - ring_dwords)
      return std::nullopt;
    normalized += ring_dwords;
  }
  if (normalized - reference_cursor > ring_dwords)
    return std::nullopt;
  return normalized;
}

void process_pm4_packets(ComputeQueueRecord &queue, GpuVm *gpu_vm,
                         const Pm4ExecutionContext &context) {
  if (queue.faulted || queue.command_retry_pending)
    return;
  auto &state = queue.commands;
  if (state.submissions.empty())
    return;
  if (state.submissions.front().failure->failed.load(std::memory_order_acquire)) {
    context.fault_queue();
    return;
  }
  if (!queue.dispatches.entries.empty())
    return;
  try {
    if (!queue.command_access)
      queue.command_access = gpu_vm->snapshot_pinned(queue.address_space);
    const auto &access = queue.command_access;
    // Bound one event's packet work, including IB chains.
    const uint32_t packet_budget = queue.submission_queue ? 4096 : 256;
    for (uint32_t budget = 0; budget < packet_budget && !state.submissions.empty(); ++budget) {
      if (!access)
        throw std::runtime_error("PM4 queue has no GPU address space");
      // A snapshot captured before GART publication can never become ready.
      if (!access->info().ready) {
        queue.command_access.reset();
        context.retry();
        return;
      }
      // A committed root packet must publish before another packet can execute.
      const auto published = queue.read_pointer_journal.publish();
      if (published == VmAccessOutcome::Unavailable) {
        context.retry();
        return;
      }
      if (published != VmAccessOutcome::Complete) {
        queue.publication_faulted = true;
        util::Logger::warn("PM4 read pointer publication failed");
        context.fault_queue();
        return;
      }
      auto &submission = state.submissions.front();
      if (submission.ready && !submission.ready()) {
        context.retry();
        return;
      }
      if (submission.buffers.empty()) {
        context.flush_caches();
        if (submission.complete)
          submission.complete(true);
        state.submissions.pop_front();
        queue.command_access.reset();
        if (!state.submissions.empty())
          queue.command_access = gpu_vm->snapshot_pinned(queue.address_space);
        continue;
      }
      auto &ib = submission.buffers.front();
      if (!ib.dwords) {
        submission.buffers.pop_front();
        continue;
      }
      const auto read_commands = [&](uint64_t offset, std::span<std::byte> bytes) {
        return ib.ring_bytes
                   ? CircularRingReader(ib.ring_base, ib.ring_bytes).read(*access, offset, bytes)
                   : access->read(offset, bytes);
      };
      uint32_t header = 0;
      auto outcome =
          read_commands(ib.address, {reinterpret_cast<std::byte *>(&header), sizeof(header)});
      if (outcome == VmAccessOutcome::Unavailable) {
        context.retry();
        return;
      }
      if (outcome != VmAccessOutcome::Complete)
        throw std::runtime_error("unmapped PM4 command buffer");
      const uint32_t type = header >> 30;
      const uint32_t count = type == 2 || header == 0xffff1000 ? 1 : ((header >> 16) & 0x3fff) + 2;
      if (count > ib.dwords || (type != 2 && type != 3))
        throw std::runtime_error("invalid PM4 packet size or type");
      std::vector<uint32_t> words(count - 1);
      if (!words.empty()) {
        outcome = read_commands(ib.address + 4,
                                {reinterpret_cast<std::byte *>(words.data()), words.size() * 4});
        if (outcome == VmAccessOutcome::Unavailable) {
          context.retry();
          return;
        }
        if (outcome != VmAccessOutcome::Complete)
          throw std::runtime_error("unmapped PM4 packet payload");
      }
      ib.address += count * 4;
      ib.dwords -= count;
      // Save root retirement before an IB command replaces the front frame.
      const uint32_t root_ring_bytes = ib.ring_bytes;
      const uint64_t root_cursor = ib.address / 4;
      const auto require = [&](size_t size) {
        if (words.size() != size)
          throw std::runtime_error(std::format("PM4 opcode {:#x} expects {} payload words, got {}",
                                               (header >> 8) & 0xff, size, words.size()));
      };
      const auto address = [&](size_t index) {
        return uint64_t{words[index]} | (uint64_t{words[index + 1]} << 32);
      };
      const uint32_t opcode =
          type == 2 || header == 0xffff1000 ? uint32_t(Pm4Opcode::Nop) : (header >> 8) & 0xff;
      util::Logger::vm([&](auto &os) {
        os << "PM4 packet " << std::hex << opcode << " payload:";
        for (uint32_t word : words)
          os << ' ' << word;
      });
      if ((header & 1) && !state.predicate_pass)
        continue;
      switch (static_cast<Pm4Opcode>(opcode)) {
      case Pm4Opcode::Nop:
        break;
      case Pm4Opcode::ContextControl:
      case Pm4Opcode::PfpSyncMe:
        if (!submission.graphics_engine)
          throw std::runtime_error("graphics state packet on compute engine");
        // The single CP retires preceding work before these packets.
        if (opcode == uint32_t(Pm4Opcode::ContextControl))
          require(2);
        else
          require(1);
        break;
      case Pm4Opcode::ClearState: {
        require(1);
        const uint32_t command = words[0];
        const bool reset = command == 0 || command == 3;
        const bool push = command == 1 || command == 3;
        // GFX12 has push/pop only. Reset values are qualified for GFX11.
        if (!submission.graphics_engine || command > 3 ||
            (reset && (context.arch == ROCJITSU_CODE_ARCH_INVALID ||
                       (context.arch != ROCJITSU_CODE_ARCH_RDNA3 &&
                        context.arch != ROCJITSU_CODE_ARCH_RDNA3_5))))
          throw std::runtime_error("unsupported CLEAR_STATE mode or engine");
        if (push) {
          if (state.saved_context_registers)
            throw std::runtime_error("nested CLEAR_STATE push is unsupported");
          state.saved_context_registers =
              std::make_unique<ComputeCommandState::ContextRegisters>(state.context_registers);
        }
        if (command == 2) {
          if (!state.saved_context_registers)
            throw std::runtime_error("CLEAR_STATE pop without a saved context");
          state.context_registers = *state.saved_context_registers;
          state.saved_context_registers.reset();
        } else if (reset) {
          reset_gfx11_context_registers(state.context_registers);
        }
        break;
      }
      case Pm4Opcode::CondExec: {
        require(4);
        if ((words[0] & 3) || words[2] || (words[3] & ~0x3fffu) || words[3] > ib.dwords)
          throw std::runtime_error("invalid COND_EXEC address, control, or extent");
        context.flush_caches();
        uint32_t value = 0;
        if (access->read(address(0), {reinterpret_cast<std::byte *>(&value), sizeof(value)}) !=
            VmAccessOutcome::Complete)
          throw std::runtime_error("PM4 COND_EXEC read failed");
        if (!value) {
          ib.address += uint64_t{words[3]} * 4;
          ib.dwords -= words[3];
        }
        break;
      }
      case Pm4Opcode::SetPredication: {
        require(3);
        const uint32_t operation = (words[0] >> 16) & 7;
        // Boolean predicates use the common GFX9+ packet layout.
        // Query accumulation and its CONTINUE/HINT controls are not modeled.
        if (!submission.graphics_engine || (words[0] & ~0x70100u) ||
            (operation != 0 && operation != 3 && operation != 4))
          throw std::runtime_error("unsupported SET_PREDICATION control");
        state.predicate_pass = true;
        if (operation) {
          const size_t bytes = operation == 4 ? 4 : 8;
          if (address(1) % bytes)
            throw std::runtime_error("unaligned SET_PREDICATION address");
          context.flush_caches();
          uint64_t value = 0;
          if (access->read(address(1), {reinterpret_cast<std::byte *>(&value), bytes}) !=
              VmAccessOutcome::Complete)
            throw std::runtime_error("PM4 SET_PREDICATION read failed");
          state.predicate_pass = (value != 0) == bool(words[0] & (1u << 8));
        }
        break;
      }
      case Pm4Opcode::SetBase:
        require(3);
        if (!submission.graphics_engine || words[0] != 1)
          throw std::runtime_error("unsupported SET_BASE index");
        state.indirect_base = address(1);
        break;
      case Pm4Opcode::WriteData: {
        if (words.size() < 4)
          throw std::runtime_error("invalid WRITE_DATA payload");
        const uint32_t destination = (words[0] >> 8) & 15;
        if ((destination != 1 && destination != 2 && destination != 5) || (words[0] & (1u << 16)))
          throw std::runtime_error("unsupported WRITE_DATA destination");
        context.flush_caches();
        if (access->write(address(1), {reinterpret_cast<const std::byte *>(words.data() + 3),
                                       (words.size() - 3) * 4}) != VmAccessOutcome::Complete)
          throw std::runtime_error("PM4 WRITE_DATA failed");
        break;
      }
      case Pm4Opcode::AtomicMem: {
        require(8);
        // GFX9+ ME/MEC share the integer TC operations. Cache policy does not
        // change their values. Loop-until-compare commands require CP retry
        // state and are rejected rather than executed as a single pass.
        if ((words[0] & ~0x0600007fu) || (words[7] & ~0x1fffu))
          throw std::runtime_error("unsupported ATOMIC_MEM control");
        const uint32_t operation = words[0] & 0x1f;
        const uint32_t width = (words[0] & 0x20) ? 8 : 4;
        if ((operation != 7 && operation != 8 && (operation < 15 || operation > 25)) ||
            address(1) % width)
          throw std::runtime_error("unsupported ATOMIC_MEM operation or alignment");
        context.flush_caches();
        // TC opcodes are shared by the ME and MEC. Reuse shader-memory atomic
        // arithmetic after decoding the packet's operation and operand order.
        static constexpr AtomicOp tc_operations[] = {AtomicOp::ADD,  AtomicOp::SUB,  AtomicOp::SMIN,
                                                     AtomicOp::UMIN, AtomicOp::SMAX, AtomicOp::UMAX,
                                                     AtomicOp::AND,  AtomicOp::OR,   AtomicOp::XOR,
                                                     AtomicOp::INC,  AtomicOp::DEC};
        const AtomicOp atomic = operation == 7   ? AtomicOp::SWAP
                                : operation == 8 ? AtomicOp::CMPSWAP
                                                 : tc_operations[operation - 15];
        const auto mutate = [&]<typename T>() {
          return access->atomic_modify(address(1), sizeof(T), [&](std::span<std::byte> bytes) {
            T value;
            std::memcpy(&value, bytes.data(), sizeof(value));
            value = apply_int_atomic(atomic, value, static_cast<T>(address(3)),
                                     static_cast<T>(address(5)));
            std::memcpy(bytes.data(), &value, sizeof(value));
          });
        };
        const auto outcome = width == 8 ? mutate.template operator()<uint64_t>()
                                        : mutate.template operator()<uint32_t>();
        if (outcome != VmAccessOutcome::Complete)
          throw std::runtime_error("PM4 ATOMIC_MEM access failed");
        break;
      }
      case Pm4Opcode::ReleaseMem: {
        require(7);
        context.flush_caches();
        const uint32_t selection = words[1] >> 29;
        uint64_t value = address(4);
        if (selection == 3)
          value = hsa_system_timestamp();
        else if (selection != 0 && selection != 1 && selection != 2)
          throw std::runtime_error("unsupported RELEASE_MEM data source");
        const size_t bytes = selection == 0 ? 0 : selection == 1 ? 4 : 8;
        if (bytes && access->write(address(2), {reinterpret_cast<const std::byte *>(&value),
                                                bytes}) != VmAccessOutcome::Complete)
          throw std::runtime_error("PM4 RELEASE_MEM failed");
        break;
      }
      case Pm4Opcode::CopyData: {
        require(5);
        const uint32_t source = words[0] & 15, destination = (words[0] >> 8) & 15;
        const size_t bytes = (words[0] & (1u << 16)) ? 8 : 4;
        if (destination != 1 && destination != 2 && destination != 5)
          throw std::runtime_error("unsupported COPY_DATA destination");
        context.flush_caches();
        uint64_t value = 0;
        if (source == 0) {
          if (!submission.graphics_engine || words[2] || words[1] > 0x3ffff)
            throw std::runtime_error("unsupported COPY_DATA register source");
          for (uint32_t i = 0; i < bytes / 4; ++i) {
            const uint32_t reg = words[1] + i;
            uint32_t data;
            if (reg >= 0x2c00 && reg < 0x3000)
              data = state.sh_registers[reg - 0x2c00];
            else if (reg >= 0xa000 && reg < 0xc000)
              data = state.context_registers[reg - 0xa000];
            else if (reg >= 0xc000 && reg < 0x10000)
              data = state.uconfig_registers[reg - 0xc000];
            else
              throw std::runtime_error("COPY_DATA register outside modeled apertures");
            value |= uint64_t{data} << (32 * i);
          }
        } else if (source == 5)
          value = address(1);
        else if (source == 9)
          value = hsa_system_timestamp();
        else if (source == 1 || source == 2) {
          if (access->read(address(1), {reinterpret_cast<std::byte *>(&value), bytes}) !=
              VmAccessOutcome::Complete)
            throw std::runtime_error("PM4 COPY_DATA read failed");
        } else {
          throw std::runtime_error("unsupported COPY_DATA source");
        }
        if (access->write(address(3), {reinterpret_cast<const std::byte *>(&value), bytes}) !=
            VmAccessOutcome::Complete)
          throw std::runtime_error("PM4 COPY_DATA write failed");
        break;
      }
      case Pm4Opcode::WaitRegMem: {
        require(6);
        if (((words[0] >> 4) & 3) != 1)
          throw std::runtime_error("unsupported WAIT_REG_MEM register space");
        context.flush_caches();
        uint32_t value = 0;
        const auto loaded =
            access->read(address(1), {reinterpret_cast<std::byte *>(&value), sizeof(value)});
        if (loaded == VmAccessOutcome::Unavailable) {
          ib.address -= count * 4;
          ib.dwords += count;
          context.retry();
          return;
        }
        if (loaded != VmAccessOutcome::Complete)
          throw std::runtime_error("PM4 WAIT_REG_MEM read failed");
        value &= words[4];
        uint32_t reference = words[3] & words[4];
        bool ready;
        switch (words[0] & 7) {
        case 0:
          ready = true;
          break;
        case 1:
          ready = value < reference;
          break;
        case 2:
          ready = value <= reference;
          break;
        case 3:
          ready = value == reference;
          break;
        case 4:
          ready = value != reference;
          break;
        case 5:
          ready = value >= reference;
          break;
        case 6:
          ready = value > reference;
          break;
        default:
          throw std::runtime_error("invalid WAIT_REG_MEM comparison");
        }
        if (!ready) {
          ib.address -= count * 4;
          ib.dwords += count;
          context.retry();
          return;
        }
        break;
      }
      case Pm4Opcode::LoadUconfigReg:
      case Pm4Opcode::LoadShReg:
      case Pm4Opcode::LoadContextReg: {
        if (words.size() < 4 || words.size() % 2 || (words[0] & 3) || (words[1] & 0xffff0000u))
          throw std::runtime_error("invalid shadow register load payload");
        if (opcode != uint32_t(Pm4Opcode::LoadShReg) && !submission.graphics_engine)
          throw std::runtime_error("graphics register load on compute engine");
        const std::span<uint32_t> registers = opcode == uint32_t(Pm4Opcode::LoadUconfigReg)
                                                  ? std::span<uint32_t>(state.uconfig_registers)
                                              : opcode == uint32_t(Pm4Opcode::LoadShReg)
                                                  ? std::span<uint32_t>(state.sh_registers)
                                                  : std::span<uint32_t>(state.context_registers);
        for (size_t i = 2; i < words.size(); i += 2) {
          const uint32_t first = words[i], count = words[i + 1];
          if (!count || (count & ~0x3fffu) || first >= registers.size() ||
              count > registers.size() - first)
            throw std::runtime_error("invalid shadow register load range");
        }
        context.flush_caches();
        for (size_t i = 2; i < words.size(); i += 2) {
          const uint32_t first = words[i], count = words[i + 1];
          // Shadow storage is indexed by the register offset, not packed by
          // the order of the ranges in this packet.
          if (access->read(address(0) + uint64_t{first} * 4,
                           {reinterpret_cast<std::byte *>(registers.data() + first), count * 4}) !=
              VmAccessOutcome::Complete)
            throw std::runtime_error("PM4 shadow register load read failed");
        }
        break;
      }
      case Pm4Opcode::LoadShRegIndex:
      case Pm4Opcode::LoadContextRegIndex: {
        require(4);
        if (opcode == uint32_t(Pm4Opcode::LoadContextRegIndex) && !submission.graphics_engine)
          throw std::runtime_error("graphics register load on compute engine");
        const uint32_t first = words[2], count = words[3];
        const std::span<uint32_t> registers = opcode == uint32_t(Pm4Opcode::LoadShRegIndex)
                                                  ? std::span<uint32_t>(state.sh_registers)
                                                  : std::span<uint32_t>(state.context_registers);
        // Direct-address mode, contiguous values (no register/value pairs).
        if ((words[0] & 3) || !count || first >= registers.size() ||
            count > registers.size() - first)
          throw std::runtime_error("unsupported register load range or mode");
        context.flush_caches();
        if (access->read(address(0), {reinterpret_cast<std::byte *>(registers.data() + first),
                                      count * 4}) != VmAccessOutcome::Complete)
          throw std::runtime_error("PM4 register load read failed");
        break;
      }
      case Pm4Opcode::SetShReg:
      case Pm4Opcode::SetShRegIndex: {
        if (words.size() < 2)
          throw std::runtime_error("invalid shader register payload");
        const uint32_t index = words[0] >> 28;
        const uint32_t first = words[0] & 0xffff;
        const bool interleave = submission.graphics_engine && index == 2 && first == 0x22f &&
                                words.size() == 2 && context.arch != ROCJITSU_CODE_ARCH_INVALID &&
                                context.arch == ROCJITSU_CODE_ARCH_RDNA4;
        // Index 3 applies the KMD CU mask to RSRC3/4. CU affinity does not
        // change functional shader results; retain the programmed resources.
        // GFX12 index 2 updates the dispatch-interleave shadow. Scheduling
        // interleave likewise does not change register values or shader results.
        if ((words[0] & 0x0fff0000) ||
            (index &&
             (opcode != uint32_t(Pm4Opcode::SetShRegIndex) || (index != 3 && !interleave))) ||
            first >= state.sh_registers.size() ||
            words.size() - 1 > state.sh_registers.size() - first)
          throw std::runtime_error("invalid shader register range or index");
        std::copy(words.begin() + 1, words.end(), state.sh_registers.begin() + first);
        break;
      }
      case Pm4Opcode::ContextRegRmw:
        require(3);
        if (!submission.graphics_engine || words[0] >= state.context_registers.size())
          throw std::runtime_error("invalid CONTEXT_REG_RMW register or engine");
        state.context_registers[words[0]] =
            (state.context_registers[words[0]] & ~words[1]) | (words[2] & words[1]);
        break;
      case Pm4Opcode::PrimeUtcl2:
        require(4);
        // PAL also sets the PFP engine selector on compute queues, where the
        // hardware ignores it. Retain the same accepted control bits there.
        if ((words[0] & ~0x4000000fu) || (words[1] & 0xfffu) || (words[3] & ~0x3fffu) ||
            address(1) > UINT64_MAX - (uint64_t{words[3]} << 12))
          throw std::runtime_error("invalid PRIME_UTCL2 control or range");
        // Translation prefetch has no data result. Address translation already
        // completes synchronously when an instruction accesses memory.
        break;
      case Pm4Opcode::SetShRegPairs: // SET_SH_REG_PAIRS
        if (words.size() % 2)
          throw std::runtime_error("invalid SET_SH_REG_PAIRS payload");
        for (size_t i = 0; i < words.size(); i += 2) {
          if (words[i] >= state.sh_registers.size())
            throw std::runtime_error("invalid SET_SH_REG_PAIRS register");
          state.sh_registers[words[i]] = words[i + 1];
        }
        break;
      case Pm4Opcode::SetContextReg:
      case Pm4Opcode::SetContextRegPairs:
      case Pm4Opcode::SetContextRegPairsPacked:
      case Pm4Opcode::SetUconfigReg:
      case Pm4Opcode::SetUconfigRegIndex:
      case Pm4Opcode::SetUconfigRegPairs: {
        if (opcode == uint32_t(Pm4Opcode::SetUconfigReg)) {
          if (!queue.packet_callbacks.write_uconfig_register && !queue.submission_queue)
            throw std::runtime_error("SET_UCONFIG_REG requires a register write callback");
          if (queue.packet_callbacks.write_uconfig_register) {
            if (words.size() != 2 || (words[0] & ~0xffffu)) {
              util::Logger::warn("invalid SET_UCONFIG_REG payload");
              context.fault_queue();
              return;
            }
            const auto result =
                queue.packet_callbacks.write_uconfig_register(0xc000 + words[0], words[1]);
            if (result == Pm4RegisterWriteStatus::Blocked) {
              ib.address -= count * 4;
              ib.dwords += count;
              context.retry();
              return;
            }
            if (result != Pm4RegisterWriteStatus::Complete) {
              util::Logger::warn("SET_UCONFIG_REG write rejected");
              context.fault_queue();
              return;
            }
          }
        }
        if (opcode == uint32_t(Pm4Opcode::SetUconfigRegPairs) && !queue.submission_queue)
          throw std::runtime_error("SET_UCONFIG_REG_PAIRS is unsupported on a native ring");
        const bool context = opcode == uint32_t(Pm4Opcode::SetContextReg) ||
                             opcode == uint32_t(Pm4Opcode::SetContextRegPairs) ||
                             opcode == uint32_t(Pm4Opcode::SetContextRegPairsPacked);
        if (context && !submission.graphics_engine)
          throw std::runtime_error("graphics state packet on compute engine");
        std::span<uint32_t> registers = context ? std::span<uint32_t>(state.context_registers)
                                                : std::span<uint32_t>(state.uconfig_registers);
        const auto write = [&](uint32_t reg, uint32_t value) {
          if (reg >= registers.size())
            throw std::runtime_error("graphics register outside aperture");
          registers[reg] = value;
        };
        if (opcode == uint32_t(Pm4Opcode::SetContextRegPairsPacked)) {
          if (words.empty() || words[0] == 0 || (words[0] & 1) ||
              uint64_t{words[0]} / 2 * 3 + 1 != words.size())
            throw std::runtime_error("invalid packed graphics register payload");
          for (size_t i = 1; i < words.size(); i += 3) {
            write(words[i] & 0xffff, words[i + 1]);
            write(words[i] >> 16, words[i + 2]);
          }
        } else if (opcode == uint32_t(Pm4Opcode::SetContextRegPairs) ||
                   opcode == uint32_t(Pm4Opcode::SetUconfigRegPairs)) {
          if (words.empty() || words.size() % 2)
            throw std::runtime_error("invalid graphics register pairs");
          for (size_t i = 0; i < words.size(); i += 2)
            write(words[i], words[i + 1]);
        } else {
          if (words.size() < 2)
            throw std::runtime_error("invalid graphics register payload");
          const uint32_t index = words[0] >> 28;
          const uint32_t first = words[0] & 0xffff;
          if ((words[0] & 0x0fff0000) || (index && (context || index > 4)))
            throw std::runtime_error(
                std::format("unsupported graphics register index {:#x}", words[0]));
          for (size_t i = 1; i < words.size(); ++i)
            write(first + i - 1, words[i]);
        }
        break;
      }
      case Pm4Opcode::AcquireMem: // ACQUIRE_MEM: earlier dispatches and DMA are already retired.
        require(7);
        context.flush_caches();
        break;
      case Pm4Opcode::EventWrite: {
        const uint32_t event = words[0] & 0x3f;
        const uint32_t event_index = (words[0] >> 8) & 15;
        if (event == 15 && event_index >= 8 && event_index <= 11) {
          require(3);
          const bool supports_streamout_query = context.arch != ROCJITSU_CODE_ARCH_INVALID &&
                                                (context.arch == ROCJITSU_CODE_ARCH_RDNA3 ||
                                                 context.arch == ROCJITSU_CODE_ARCH_RDNA3_5);
          if (!submission.graphics_engine || !supports_streamout_query ||
              words[0] != (15 | (event_index << 8)) || (words[1] & 7))
            throw std::runtime_error("unsupported PM4 streamout query event");
          context.flush_caches();
          // Compute execution has no streamout counter producer. Return valid
          // zero samples until graphics supplies the counters.
          constexpr uint64_t kSampleValid = uint64_t{1} << 63;
          constexpr std::array<uint64_t, 2> values{kSampleValid, kSampleValid};
          if (access->write(address(1), std::as_bytes(std::span{values})) !=
              VmAccessOutcome::Complete)
            throw std::runtime_error("PM4 streamout query write failed");
          break;
        }
        if (submission.graphics_engine && event == 56) {
          // PIXEL_PIPE_STAT_CONTROL configures graphics counters, not a memory write.
          require(3);
          break;
        }
        require(1);
        // Counter START/STOP events also configure compute counters. SQ_NON_EVENT
        // drains graphics pipeline messages without writing a sampled result.
        if (event != 7 && event != 23 && event != 24 && event != 25 && event != 26 &&
            !(submission.graphics_engine &&
              (event == 15 || event == 16 || event == 36 || event == 38 || event == 44 ||
               event == 46 || event == 49)))
          throw std::runtime_error(std::format("unsupported PM4 EVENT_WRITE event {}", event));
        context.flush_caches();
        break;
      }
      case Pm4Opcode::StreamoutStatsQuery: {
        require(5);
        if (!submission.graphics_engine || context.arch == ROCJITSU_CODE_ARCH_INVALID ||
            context.arch != ROCJITSU_CODE_ARCH_RDNA4 || (words[0] & 7) || (words[3] & 7) ||
            words[2] > 3 || address(0) > UINT64_MAX - 79)
          throw std::runtime_error("unsupported PM4 streamout statistics query");
        // GFX12 shaders maintain four needed/written counter pairs in ordinary memory,
        // following four dwords of streamout buffer offsets.
        context.flush_caches();
        std::array<uint64_t, 2> values{};
        if (access->read(address(0) + 16 + 16 * words[2],
                         std::as_writable_bytes(std::span{values})) != VmAccessOutcome::Complete)
          throw std::runtime_error("PM4 streamout query read failed");
        for (auto &value : values)
          value |= uint64_t{1} << 63;
        if (access->write(address(3), std::as_bytes(std::span{values})) !=
            VmAccessOutcome::Complete)
          throw std::runtime_error("PM4 streamout query write failed");
        break;
      }
      case Pm4Opcode::DmaData: {
        require(6);
        const uint32_t src_select = (words[0] >> 29) & 3;
        const uint32_t dst_select = (words[0] >> 20) & 3;
        const uint32_t bytes = words[5] & 0x3ffffff;
        if (!bytes || dst_select == 2) // DMA drain or prefetch only.
          break;
        if (dst_select != 0 && dst_select != 3)
          throw std::runtime_error("unsupported PM4 DMA destination");
        context.flush_caches();
        if (src_select == 2) {
          std::vector<uint8_t> data(bytes);
          for (uint32_t i = 0; i < bytes; ++i)
            data[i] = words[1] >> ((i % 4) * 8);
          if (access->write(address(3), std::as_bytes(std::span(data))) !=
              VmAccessOutcome::Complete)
            throw std::runtime_error("PM4 DMA fill failed");
        } else if (src_select == 0 || src_select == 3) {
          std::vector<std::byte> data(bytes);
          if (access->read(address(1), data) != VmAccessOutcome::Complete ||
              access->write(address(3), data) != VmAccessOutcome::Complete)
            throw std::runtime_error("PM4 DMA copy failed");
        } else {
          throw std::runtime_error("unsupported PM4 DMA source");
        }
        break;
      }
      case Pm4Opcode::IndirectBuffer: {
        require(3);
        const bool chained = words[2] & (1u << 20);
        const uint32_t depth = ib.depth + (chained ? 0 : 1);
        if (++submission.indirect_expansions > Pm4Submission::kMaxIndirectExpansions ||
            depth >= Pm4Submission::kMaxIndirectDepth) {
          context.fault_queue();
          return;
        }
        // CHAIN jumps at the current IB level; a non-chained IB returns here.
        // Other root IBs supplied by the same CS remain queued behind this one.
        if (words[2] & (1u << 20))
          submission.buffers.pop_front();
        submission.buffers.push_front({address(0), words[2] & 0xfffff, depth});
        break;
      }
      case Pm4Opcode::DispatchDirectInterleaved:
        if (!submission.graphics_engine || context.arch == ROCJITSU_CODE_ARCH_INVALID ||
            context.arch != ROCJITSU_CODE_ARCH_RDNA4)
          throw std::runtime_error("interleaved dispatch requires a GFX12 graphics queue");
        [[fallthrough]];
      case Pm4Opcode::DispatchDirect:
        require(4);
        context.dispatch({words[0], words[1], words[2], words[3]});
        break;
      case Pm4Opcode::DispatchIndirectInterleaved:
        if (!submission.graphics_engine || context.arch == ROCJITSU_CODE_ARCH_INVALID ||
            context.arch != ROCJITSU_CODE_ARCH_RDNA4)
          throw std::runtime_error("interleaved dispatch requires a GFX12 graphics queue");
        [[fallthrough]];
      case Pm4Opcode::DispatchIndirect: {
        require(submission.graphics_engine ? 2 : 3);
        context.flush_caches();
        const uint64_t arguments =
            submission.graphics_engine ? state.indirect_base + words[0] : address(0);
        std::array<uint32_t, 4> dimensions{0, 0, 0, words.back()};
        if (access->read(arguments, {reinterpret_cast<std::byte *>(dimensions.data()), 12}) !=
            VmAccessOutcome::Complete)
          throw std::runtime_error("PM4 DISPATCH_INDIRECT read failed");
        context.dispatch(dimensions);
        break;
      }
      default:
        throw std::runtime_error(std::format("unsupported PM4 opcode {:#x}", opcode));
      }
      if (root_ring_bytes) {
        // An IB enters a child frame; only the root packet advances the root ring.
        // The journal retains this stream's snapshot until writeback is durable.
        uint64_t retired = root_cursor;
        queue.read_pointer_journal.retire(retired, retired % (root_ring_bytes / 4), *access);
        const auto published = queue.read_pointer_journal.publish();
        if (published == VmAccessOutcome::Unavailable) {
          context.retry();
          return;
        }
        if (published != VmAccessOutcome::Complete) {
          queue.publication_faulted = true;
          util::Logger::warn("PM4 read pointer publication failed");
          context.fault_queue();
          return;
        }
      }
      if (!queue.dispatches.entries.empty())
        return;
    }
    if (!state.submissions.empty()) {
      context.retry();
    }
  } catch (const std::exception &error) {
    util::Logger::warn("PM4 queue failed: ", error.what());
    context.fault_queue();
  }
}

} // namespace rocjitsu::amdgpu
