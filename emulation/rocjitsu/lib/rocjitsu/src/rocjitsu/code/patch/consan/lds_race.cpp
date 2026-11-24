// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/lds_race.h"
#include "rocjitsu/code/patch/consan/prepare.h"
#include "rocjitsu/isa/isa_family.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/shared/ds2_addressing.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <unordered_map>

namespace rocjitsu::consan {
namespace {
Result reject_instruction(InputError &error, InputErrorCode code, std::string message,
                          std::optional<uint64_t> offset = std::nullopt) {
  error = {.code = code, .text_offset = offset, .message = std::move(message)};
  return Result::failure();
}

bool admitted_instruction(const Instruction &instruction) {
  switch (instruction.amdgpu_operation()) {
  case amdgpu::Operation::S_MUL_I32:
  case amdgpu::Operation::S_LOAD_DWORD:
  case amdgpu::Operation::S_LOAD_DWORDX2:
  case amdgpu::Operation::S_MOV_B64:
  case amdgpu::Operation::S_AND_SAVEEXEC_B64:
  case amdgpu::Operation::S_AND_B64:
  case amdgpu::Operation::S_OR_B64:
  case amdgpu::Operation::S_DELAY_ALU:
  case amdgpu::Operation::GLOBAL_STORE_DWORD:
  case amdgpu::Operation::DS_READ_B32:
  case amdgpu::Operation::DS_WRITE_B32:
  case amdgpu::Operation::S_WAIT_KMCNT:
  case amdgpu::Operation::S_WAIT_STORECNT:
  case amdgpu::Operation::S_WAIT_DSCNT:
  case amdgpu::Operation::S_WAIT_LOADCNT_DSCNT:
  case amdgpu::Operation::S_WAIT_STORECNT_DSCNT:
  case amdgpu::Operation::S_WAIT_LOADCNT:
  case amdgpu::Operation::S_WAIT_XCNT:
  case amdgpu::Operation::S_LOAD_B32:
  case amdgpu::Operation::S_LOAD_B64:
  case amdgpu::Operation::S_WAITCNT:
  case amdgpu::Operation::S_WAITCNT_VSCNT:
  case amdgpu::Operation::S_MOV_B32:
  case amdgpu::Operation::S_AND_SAVEEXEC_B32:
  case amdgpu::Operation::S_AND_B32:
  case amdgpu::Operation::S_OR_B32:
  case amdgpu::Operation::S_CMP_EQ_U32:
  case amdgpu::Operation::S_ADD_U32:
  case amdgpu::Operation::S_SUB_U32:
  case amdgpu::Operation::S_ADD_CO_U32:
  case amdgpu::Operation::S_SUB_CO_U32:
  case amdgpu::Operation::S_BRANCH:
  case amdgpu::Operation::S_CBRANCH_SCC0:
  case amdgpu::Operation::S_CBRANCH_SCC1:
  case amdgpu::Operation::S_CBRANCH_VCCZ:
  case amdgpu::Operation::S_CBRANCH_VCCNZ:
  case amdgpu::Operation::S_CBRANCH_EXECZ:
  case amdgpu::Operation::S_CBRANCH_EXECNZ:
  case amdgpu::Operation::S_NOP:
  case amdgpu::Operation::S_ENDPGM:
  case amdgpu::Operation::GLOBAL_STORE_B32:
  case amdgpu::Operation::DS_STORE_B32:
  case amdgpu::Operation::DS_LOAD_B32:
    return true;
  case amdgpu::Operation::V_CMPX_EQ_U32:
  case amdgpu::Operation::V_ADD_U32:
  case amdgpu::Operation::V_NOP:
  case amdgpu::Operation::V_MOV_B32:
  case amdgpu::Operation::V_LSHRREV_B32:
  case amdgpu::Operation::V_LSHLREV_B32:
  case amdgpu::Operation::V_ADD_NC_U32:
  case amdgpu::Operation::V_AND_B32:
  case amdgpu::Operation::V_CMP_EQ_U32:
  case amdgpu::Operation::V_CMP_GT_U32:
    break;
  default:
    return false;
  }

  // The qualified vector forms use VOP1/VOP2/VOPC rather than VOP3. Their
  // first word has bit 31 clear; literals and DPP extensions do not change it.
  // SDWA has the same base format but was never part of this admission set.
  if (instruction.raw_encoding() == nullptr || instruction.size() < 4 ||
      (instruction.raw_encoding()[0] & (uint32_t{1} << 31)) != 0) {
    return false;
  }
  amdgpu::RegisterModifiers modifiers;
  instruction.amdgpu_register_modifiers(modifiers);
  return !modifiers.sdwa;
}

Result reject(InputError &error, std::string message) {
  error = {.code = InputErrorCode::ResourceLimit, .text_offset = {}, .message = std::move(message)};
  return Result::failure();
}

struct NativeLdsForm {
  uint32_t width = 0;
  LdsRangeShape shape = LdsRangeShape::Single;
};

NativeLdsForm native_lds_form(const Instruction &instruction) {
  if (instruction.amdgpu_memory_kind() == amdgpu::MemoryKind::Atomic) {
    if (instruction.amdgpu_memory_addressing() != amdgpu::MemoryAddressing::Lds)
      return {};
    switch (instruction.amdgpu_operation()) {
    case amdgpu::Operation::DS_ADD_U32:
      return {.width = 4};
    case amdgpu::Operation::DS_ADD_U64:
      return {.width = 8};
    default:
      return {};
    }
  }
  switch (instruction.amdgpu_operation()) {
  case amdgpu::Operation::DS_LOAD_B32:
  case amdgpu::Operation::DS_STORE_B32:
  case amdgpu::Operation::DS_READ_B32:
  case amdgpu::Operation::DS_WRITE_B32:
    return {.width = 4};
  case amdgpu::Operation::DS_LOAD_U8:
  case amdgpu::Operation::DS_LOAD_I8:
  case amdgpu::Operation::DS_STORE_B8:
  case amdgpu::Operation::DS_READ_U8:
  case amdgpu::Operation::DS_READ_I8:
  case amdgpu::Operation::DS_WRITE_B8:
  case amdgpu::Operation::DS_LOAD_U8_D16:
  case amdgpu::Operation::DS_LOAD_U8_D16_HI:
  case amdgpu::Operation::DS_LOAD_I8_D16:
  case amdgpu::Operation::DS_LOAD_I8_D16_HI:
  case amdgpu::Operation::DS_STORE_B8_D16_HI:
  case amdgpu::Operation::DS_READ_U8_D16:
  case amdgpu::Operation::DS_READ_U8_D16_HI:
  case amdgpu::Operation::DS_READ_I8_D16:
  case amdgpu::Operation::DS_READ_I8_D16_HI:
  case amdgpu::Operation::DS_WRITE_B8_D16_HI:
    return {.width = 1};
  case amdgpu::Operation::DS_LOAD_U16:
  case amdgpu::Operation::DS_LOAD_I16:
  case amdgpu::Operation::DS_STORE_B16:
  case amdgpu::Operation::DS_READ_U16:
  case amdgpu::Operation::DS_READ_I16:
  case amdgpu::Operation::DS_WRITE_B16:
  case amdgpu::Operation::DS_LOAD_U16_D16:
  case amdgpu::Operation::DS_LOAD_U16_D16_HI:
  case amdgpu::Operation::DS_STORE_B16_D16_HI:
  case amdgpu::Operation::DS_READ_U16_D16:
  case amdgpu::Operation::DS_READ_U16_D16_HI:
  case amdgpu::Operation::DS_WRITE_B16_D16_HI:
    return {.width = 2};
  case amdgpu::Operation::DS_LOAD_B64:
  case amdgpu::Operation::DS_STORE_B64:
  case amdgpu::Operation::DS_READ_B64:
  case amdgpu::Operation::DS_WRITE_B64:
    return {.width = 8};
  case amdgpu::Operation::DS_LOAD_B96:
  case amdgpu::Operation::DS_STORE_B96:
  case amdgpu::Operation::DS_READ_B96:
  case amdgpu::Operation::DS_WRITE_B96:
    return {.width = 12};
  case amdgpu::Operation::DS_LOAD_B128:
  case amdgpu::Operation::DS_STORE_B128:
  case amdgpu::Operation::DS_READ_B128:
  case amdgpu::Operation::DS_WRITE_B128:
    return {.width = 16};
  case amdgpu::Operation::DS_READ_B64_TR_B4:
  case amdgpu::Operation::DS_LOAD_TR4_B64:
    return {.width = 8, .shape = LdsRangeShape::Transpose4};
  case amdgpu::Operation::DS_READ_B96_TR_B6:
  case amdgpu::Operation::DS_LOAD_TR6_B96:
    return {.width = 12, .shape = LdsRangeShape::Transpose6};
  case amdgpu::Operation::DS_READ_B64_TR_B8:
  case amdgpu::Operation::DS_LOAD_TR8_B64:
    return {.width = 8, .shape = LdsRangeShape::Transpose8};
  case amdgpu::Operation::DS_READ_B64_TR_B16:
    return {.width = 8, .shape = LdsRangeShape::Transpose16};
  case amdgpu::Operation::DS_LOAD_TR16_B128:
    return {.width = 16, .shape = LdsRangeShape::Transpose16};
  default:
    break;
  }
  const amdgpu::MemoryKind kind = instruction.amdgpu_memory_kind();
  if (kind == amdgpu::MemoryKind::Load || kind == amdgpu::MemoryKind::Store) {
    const std::optional<amdgpu::Ds2Addressing> addressing =
        amdgpu::ds2_addressing(instruction.amdgpu_operation());
    if (addressing) {
      return {.width = addressing->element_bytes,
              .shape = addressing->stride64 ? LdsRangeShape::DualStride64 : LdsRangeShape::Dual};
    }
  }
  return {};
}

// A dual load writes a single tuple. A dual store can take two noncontiguous
// source tuples, so each encoded register must fit the original allocation.
std::optional<RegisterRef> allocated_vgpr_tuple(const Operand *operand, uint32_t words,
                                                uint32_t allocated_vgprs) {
  if (operand == nullptr) {
    return std::nullopt;
  }
  const std::optional<RegisterRef> reg = operand->decoded_vgpr();
  if (!reg || reg->width != words || uint32_t{reg->index} + words > allocated_vgprs) {
    return std::nullopt;
  }
  return reg;
}

} // namespace

Result qualify_lds_race_entry(const InputProgram &program, InputError &error) {
  const uint32_t group_bytes = program.kernel.descriptor.group_segment_fixed_size;
  if (group_bytes < 4 || group_bytes > 65536) {
    return reject_instruction(error, InputErrorCode::UnsupportedEntry,
                              "native LDS requires an LDS extent in [4, 65536] bytes");
  }
  return Result::success();
}

Result qualify_lds_race_block(const BasicBlock &block, InputError &error) {
  // Defer the conditional self-loop to whole-program barrier qualification.
  for (const BasicBlock *successor : block.successors()) {
    if ((successor != &block && successor->start_offset() <= block.start_offset()) ||
        (successor == &block && block.successors().size() != 2)) {
      return reject_instruction(
          error, InputErrorCode::UnsupportedFlow,
          "only a conditional single-block backedge can enter barrier qualification",
          block.start_offset());
    }
  }
  return Result::success();
}

FailureOr<std::vector<const BasicBlock *>>
qualify_lds_race_cfg(std::span<const std::unique_ptr<BasicBlock>> blocks, InputError &error) {
  // A trampoline return can point backward without revisiting any executed block.
  // Defer only conditional self-edges to the later whole-program barrier gate;
  // every other edge must form a DAG. This ordering alone does not admit a loop.
  std::unordered_map<const BasicBlock *, size_t> incoming;
  incoming.reserve(blocks.size());
  for (const std::unique_ptr<BasicBlock> &block : blocks) {
    incoming.emplace(block.get(), 0);
  }
  for (const std::unique_ptr<BasicBlock> &block : blocks) {
    for (const BasicBlock *successor : block->successors()) {
      if (successor == block.get() && block->successors().size() == 2) {
        continue;
      }
      const std::unordered_map<const BasicBlock *, size_t>::iterator edge =
          incoming.find(successor);
      if (edge == incoming.end()) {
        return reject_instruction(error, InputErrorCode::UnsupportedFlow,
                                  "CFG successor leaves the selected entry", block->start_offset());
      }
      ++edge->second;
    }
  }
  std::vector<const BasicBlock *> ready;
  ready.reserve(blocks.size());
  for (const std::unique_ptr<BasicBlock> &block : blocks) {
    if (incoming.find(block.get())->second == 0) {
      ready.push_back(block.get());
    }
  }
  for (size_t index = 0; index < ready.size(); ++index) {
    for (const BasicBlock *successor : ready[index]->successors()) {
      if (successor == ready[index] && ready[index]->successors().size() == 2) {
        continue;
      }
      size_t &count = incoming.find(successor)->second;
      if (--count == 0) {
        ready.push_back(successor);
      }
    }
  }
  if (ready.size() != blocks.size()) {
    for (const std::unique_ptr<BasicBlock> &block : blocks) {
      if (incoming.find(block.get())->second != 0) {
        return reject_instruction(error, InputErrorCode::UnsupportedFlow,
                                  "MVP does not admit CFG cycles", block->start_offset());
      }
    }
  }
  return ready;
}

Result classify_lds_race_instruction(const Instruction &instruction, rocjitsu::IsaFamily arch,
                                     uint32_t allocated_vgprs,
                                     std::vector<NativeLdsAccess> &accesses, InputError &error) {
  const std::string_view name = instruction.mnemonic();
  const amdgpu::Operation operation = instruction.amdgpu_operation();
  const amdgpu::MemoryAddressing addressing = instruction.amdgpu_memory_addressing();
  const uint64_t offset = instruction.src_loc();
  if ((uses_split_workgroup_barrier(arch) && (operation == amdgpu::Operation::S_BARRIER_SIGNAL ||
                                              operation == amdgpu::Operation::S_BARRIER_WAIT)) ||
      (!uses_split_workgroup_barrier(arch) && operation == amdgpu::Operation::S_BARRIER)) {
    return Result::success(); // The complete original group/CFG is qualified below.
  }
  const bool atomic = instruction.amdgpu_memory_kind() == amdgpu::MemoryKind::Atomic;
  const NativeLdsForm form = native_lds_form(instruction);
  if ((instruction.flags() & BARRIER) != 0 || (atomic && form.width == 0) ||
      name.starts_with("s_barrier") || name.starts_with("s_sendmsg")) {
    return reject_instruction(error, InputErrorCode::UnsupportedSynchronization,
                              "unmodeled synchronization: " + std::string(name), offset);
  }
  const uint32_t width = form.width;
  if ((addressing == amdgpu::MemoryAddressing::Lds || addressing == amdgpu::MemoryAddressing::Gds ||
       addressing == amdgpu::MemoryAddressing::DirectLds) &&
      width == 0) {
    return reject_instruction(error, InputErrorCode::UnsupportedAccess,
                              "unsupported LDS form: " + std::string(name), offset);
  }
  if (width == 0 && !admitted_instruction(instruction)) {
    return reject_instruction(error, InputErrorCode::UnsupportedInstruction,
                              "instruction outside MVP: " + std::string(name), offset);
  }
  if (width != 0) {
    // CDNA3/4 place GDS at bit 16; RDNA and CDNA5 use bit 17.
    const uint32_t gds_bit =
        (arch == rocjitsu::IsaFamily::Cdna3 || arch == rocjitsu::IsaFamily::Cdna4) ? 16 : 17;
    if (instruction.size() != 8 || (instruction.raw_encoding()[0] & (1u << gds_bit)) != 0) {
      return reject_instruction(error, InputErrorCode::UnsupportedAccess, "non-native LDS form",
                                offset);
    }
    const bool read = instruction.amdgpu_memory_kind() == amdgpu::MemoryKind::Load;
    const bool dual = is_dual(form.shape);
    const bool transpose = is_transpose(form.shape);
    if (transpose && (!read || (arch != IsaFamily::Cdna4 && arch != IsaFamily::Cdna5) ||
                      instruction.num_dst_operands() != 1 || instruction.num_src_operands() != 2))
      return reject_instruction(error, InputErrorCode::UnsupportedAccess,
                                "unsupported LDS transpose operands or target", offset);
    const bool dual_store = dual && !read;
    if (dual && (instruction.num_dst_operands() != 1 ||
                 instruction.num_src_operands() != (dual_store ? 3 : 2))) {
      return reject_instruction(error, InputErrorCode::UnsupportedAccess,
                                "malformed two-address LDS operands", offset);
    }

    const std::optional<RegisterRef> address_reg =
        allocated_vgpr_tuple(instruction.src_operand(0), 1, allocated_vgprs);
    const uint32_t words_per_range = std::max(width / 4, 1u);
    const Operand *value = read ? instruction.dst_operand(0) : instruction.src_operand(1);
    const uint32_t value_words = read && dual ? 2 * words_per_range : words_per_range;
    const std::optional<RegisterRef> value_reg =
        allocated_vgpr_tuple(value, value_words, allocated_vgprs);
    if (!address_reg || !value_reg ||
        (dual_store &&
         !allocated_vgpr_tuple(instruction.src_operand(2), words_per_range, allocated_vgprs))) {
      return reject_instruction(error, InputErrorCode::UnsupportedAccess,
                                "LDS operands exceed allocation", offset);
    }

    if (transpose && arch == IsaFamily::Cdna4 && width == 8 && value_reg->index % 2 != 0)
      return reject_instruction(error, InputErrorCode::UnsupportedAccess,
                                "CDNA4 B64 transpose requires an even destination VGPR", offset);

    uint16_t immediate = static_cast<uint16_t>(instruction.raw_encoding()[0]);
    LdsRangeRecipe ranges{.shape = form.shape};
    if (dual) {
      ranges = {.shape = form.shape,
                .offset0 = static_cast<uint8_t>(immediate),
                .offset1 = static_cast<uint8_t>(immediate >> 8)};
      // Both ranges share a pre-instruction base and mask. Replay applies each
      // encoded element offset using this form's width and stride.
      immediate = 0;
    }
    accesses.push_back({.site = static_cast<uint32_t>(accesses.size()),
                        .original_offset = offset,
                        .kind = atomic ? AccessKind::Atomic
                                : read ? AccessKind::Read
                                       : AccessKind::Write,
                        .address_vgpr = address_reg->index,
                        .immediate_bytes = immediate,
                        .width = width,
                        .lds_ranges = ranges});
  }
  return Result::success();
}

Result qualify_lds_race_barrier(InputProgram &program, std::span<BasicBlock *const> blocks,
                                InputError &error) {
  std::vector<const BasicBlock *> barrier_blocks;
  uint32_t members = 0;
  const rocjitsu::IsaFamily arch = program.target->arch;
  const bool split = uses_split_workgroup_barrier(arch);
  for (BasicBlock *block : blocks) {
    std::vector<const Instruction *> instructions;
    for (const Instruction &instruction : block->instructions()) {
      instructions.push_back(&instruction);
      if (instruction.amdgpu_operation() == amdgpu::Operation::S_BARRIER ||
          instruction.amdgpu_operation() == amdgpu::Operation::S_BARRIER_SIGNAL ||
          instruction.amdgpu_operation() == amdgpu::Operation::S_BARRIER_WAIT) {
        ++members;
      }
    }
    for (size_t index = 0; index < instructions.size(); ++index) {
      if (instructions[index]->amdgpu_operation() !=
          (split ? amdgpu::Operation::S_BARRIER_SIGNAL : amdgpu::Operation::S_BARRIER)) {
        continue;
      }
      size_t wait_index = index;
      bool intervening_memory_or_barrier = false;
      if (split) {
        ++wait_index;
        while (wait_index < instructions.size() &&
               instructions[wait_index]->amdgpu_operation() != amdgpu::Operation::S_BARRIER_WAIT) {
          const Instruction &instruction = *instructions[wait_index];
          if (instruction.is_memory_op() || instruction.is_barrier()) {
            intervening_memory_or_barrier = true;
          }
          ++wait_index;
        }
      }
      // Require the original guest drain: DBI envelopes add their own waits,
      // which must not make an undrained guest barrier qualify.
      if (index == 0 || wait_index == instructions.size() || intervening_memory_or_barrier ||
          instructions[index - 1]->size() != 4 || instructions[index]->size() != 4 ||
          instructions[wait_index]->size() != 4) {
        return reject_instruction(
            error, InputErrorCode::UnsupportedSynchronization,
            "requires original LDS drain/barrier groups without intervening memory "
            "operations or barriers",
            instructions[index]->src_loc());
      }
      const Instruction &drain = *instructions[index - 1];
      bool valid_group = false;
      if (split) {
        const uint32_t signal = arch == rocjitsu::IsaFamily::Rdna4
                                    ? rdna4::build_sop1(rdna4::kSBarrierSignalSop1,
                                                        {.ssrc0 = rdna4::OPR_SSRC_NEG_INT_MIN})[0]
                                    : cdna5::build_sop1(cdna5::kSBarrierSignalSop1,
                                                        {.ssrc0 = cdna5::OPR_SSRC_NEG_INT_MIN})[0];
        const uint32_t wait_opcode = arch == rocjitsu::IsaFamily::Rdna4 ? rdna4::kSBarrierWaitSopp
                                                                        : cdna5::kSBarrierWaitSopp;
        const amdgpu::Operation drain_operation = drain.amdgpu_operation();
        const bool ds_wait = drain_operation == amdgpu::Operation::S_WAIT_DSCNT ||
                             drain_operation == amdgpu::Operation::S_WAIT_LOADCNT_DSCNT ||
                             drain_operation == amdgpu::Operation::S_WAIT_STORECNT_DSCNT;
        const FailureOr<std::optional<amdgpu::WaitFields>> fields =
            amdgpu::WaitCounterPolicy::explicit_wait_fields(drain, arch);
        // A combined wait may leave unrelated memory pending, but its original
        // DS threshold must drain every LDS access before the guest barrier.
        valid_group = ds_wait && fields.succeeded() && fields.value() &&
                      (*fields.value())[amdgpu::WaitCounterPolicy::counter_index(
                          amdgpu::WaitCounterKind::Ds)] == 0 &&
                      instructions[index]->raw_encoding()[0] == signal &&
                      instructions[wait_index]->raw_encoding()[0] ==
                          build_sopp_encoding(arch, wait_opcode, 0xffff).value();
      } else {
        const uint32_t opcode = arch == rocjitsu::IsaFamily::Rdna3
                                    ? rdna3::kSBarrierSopp
                                    : (arch == rocjitsu::IsaFamily::Cdna3 ? cdna3::kSBarrierSopp
                                                                          : cdna4::kSBarrierSopp);
        // LGKM's bit positions differ on gfx9 and gfx11. Decode the original
        // field so stronger waits (including s_waitcnt 0) also qualify.
        valid_group =
            drain.amdgpu_operation() == amdgpu::Operation::S_WAITCNT &&
            amdgpu::decode_legacy_waitcnt(drain.raw_encoding()[0] & 0xffff, arch).lgkmcnt == 0 &&
            instructions[index]->raw_encoding()[0] == build_sopp_encoding(arch, opcode, 0).value();
      }
      if (!valid_group) {
        return reject_instruction(
            error, InputErrorCode::UnsupportedSynchronization,
            "requires an original zero LDS drain and canonical full-workgroup "
            "barrier identity",
            instructions[index]->src_loc());
      }
      barrier_blocks.push_back(block);
      program.barriers.push_back(
          BarrierFacts{.original_drain = instructions[index - 1]->src_loc(),
                       .original_signal = instructions[index]->src_loc(),
                       .original_wait = instructions[wait_index]->src_loc()});
    }
  }
  // One block containing every group gives each completed loop cycle the same
  // group order, allowing a retained phase ordinal to identify its occurrence.
  const BasicBlock *loop = nullptr;
  for (const BasicBlock *block : blocks) {
    for (const BasicBlock *successor : block->successors()) {
      if (successor == block) {
        if (loop || block->successors().size() != 2 || members == 0) {
          return reject_instruction(error, InputErrorCode::UnsupportedFlow,
                                    "requires one conditional single-block barrier loop");
        }
        loop = block;
      }
    }
  }
  if (loop) {
    if (std::ranges::any_of(barrier_blocks,
                            [&](const BasicBlock *block) { return block != loop; })) {
      return reject_instruction(error, InputErrorCode::UnsupportedFlow,
                                "all repeated barrier groups must belong to the single loop block");
    }
    program.repeated = true;
  }
  if (members == 0) {
    return Result::success();
  }
  if (program.maximum_workgroup_size <= program.wave_size) {
    return reject_instruction(
        error, InputErrorCode::UnsupportedEntry,
        "barrier profile requires a maximum workgroup size permitting two waves");
  }
  if (members != (split ? 2u : 1u) * program.barriers.size() || program.barriers.empty() ||
      program.barriers.size() > kSyncCapacity) {
    return reject_instruction(error, InputErrorCode::UnsupportedSynchronization,
                              "incomplete or repeated workgroup barrier group");
  }
  // For the acyclic profile, deleting the group block must leave no path
  // from the hardware entry to an exit. This also rejects divergent bypasses;
  // scalar barriers themselves execute independently of the current EXEC.
  const auto entry =
      std::ranges::find(blocks, program.kernel.entry_text_offset, &BasicBlock::start_offset);
  if (entry == blocks.end()) {
    return reject_instruction(error, InputErrorCode::UnsupportedSynchronization,
                              "missing barrier entry");
  }
  // A preheader may bypass the loop for zero iterations. Do not require loop
  // groups to dominate exits or infer uniform trip counts; classification needs
  // actual matching completion witnesses from every required wave.
  if (!program.repeated) {
    for (const BasicBlock *barrier_block : barrier_blocks) {
      std::vector<const BasicBlock *> pending{*entry};
      std::set<const BasicBlock *> visited;
      while (!pending.empty()) {
        const BasicBlock *block = pending.back();
        pending.pop_back();
        if (block == barrier_block || !visited.insert(block).second) {
          continue;
        }
        if (block->successors().empty()) {
          return reject_instruction(error, InputErrorCode::UnsupportedSynchronization,
                                    "workgroup barrier does not converge on every exit",
                                    block->start_offset());
        }
        for (const BasicBlock *successor : block->successors()) {
          pending.push_back(successor);
        }
      }
    }
  }
  std::ranges::sort(program.barriers, {}, &BarrierFacts::original_signal);
  return Result::success();
}

FailureOr<uint32_t> lds_race_row_stride(size_t sites, InputError &error, bool has_barriers) {
  if (sites > ContractLimits{}.sites) {
    return reject(error, "no bounded LDS report-row layout");
  }
  const uint64_t stride = report_row_bytes(sites, has_barriers);
  if (stride > ContractLimits{}.report_bytes - kReportHeaderBytes) {
    return reject(error, "no bounded LDS report-row layout");
  }
  return static_cast<uint32_t>(stride);
}

Result validate_lds_race_accesses(const InputProgram &program, const Manifest &manifest,
                                  std::span<BasicBlock *const> blocks,
                                  std::span<const uint8_t> prepared_text, InputError &error) {
  std::set<uint64_t> barrier_offsets;
  std::set<uint64_t> access_offsets;
  for (BasicBlock *block : blocks) {
    for (const Instruction &instruction : block->instructions()) {
      const amdgpu::Operation operation = instruction.amdgpu_operation();
      if (operation == amdgpu::Operation::S_BARRIER ||
          operation == amdgpu::Operation::S_BARRIER_SIGNAL ||
          operation == amdgpu::Operation::S_BARRIER_WAIT) {
        barrier_offsets.insert(instruction.src_loc());
      }
      if (native_lds_form(instruction).width != 0) {
        access_offsets.insert(instruction.src_loc());
      }
    }
  }
  std::set<uint64_t> expected_barriers;
  for (const BarrierFacts &barrier : manifest.barriers) {
    expected_barriers.insert(barrier.prepared_signal);
    expected_barriers.insert(barrier.prepared_wait);
    if (std::memcmp(program.image.data() + program.text_file_offset + barrier.original_signal,
                    prepared_text.data() + barrier.prepared_signal, 4) != 0 ||
        std::memcmp(program.image.data() + program.text_file_offset + barrier.original_wait,
                    prepared_text.data() + barrier.prepared_wait, 4) != 0) {
      return reject(error, "final reachable barrier operations differ from original group");
    }
  }
  if (barrier_offsets != expected_barriers) {
    return reject(error, "final reachable barrier multiplicity differs from original");
  }
  if (access_offsets.size() != manifest.sites.size()) {
    return reject(error, "final reachable guest access multiplicity differs from original");
  }
  for (const SiteFacts &site : manifest.sites) {
    if (!access_offsets.contains(site.prepared_guest_offset) ||
        std::memcmp(program.image.data() + program.text_file_offset + site.original_offset,
                    prepared_text.data() + site.prepared_guest_offset, 8) != 0) {
      return reject(error, "final guest LDS operation/map differs from original bytes");
    }
  }
  return Result::success();
}

} // namespace rocjitsu::consan
